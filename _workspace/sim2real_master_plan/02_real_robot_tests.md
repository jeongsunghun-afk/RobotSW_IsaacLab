# 컴포넌트 2: 실기 검증 테스트

> Sim-to-Real 마스터 플랜의 컴포넌트 2. **실제 로봇으로 미리 해볼 수 있는 단계적 검증 테스트** 메뉴.
> 배포 직전/직후, 실기에서 무엇을 어떤 순서로 확인하는가를 표준 절차 깊이로 정리한다.
>
> **대상**: legged 로봇 전반(2족/4족) 범용.
> **실기 측정 자원**: 관절 pos/vel(encoder) + 관절 토크/전류 실측. 자유보행 단계에서 base IMU(자세/각속도). base lin_vel은 estimator로 추정.
> **2-stage scope**: ① 고정 베이스 leg 리그 → ② full robot 자유보행.
>
> 각 테스트는 **목적 / 근거(논문·출처) / sim 비교 메트릭 / 안전 가드 / 판정 메트릭(pass·fail)** 으로 기술한다.

---

## 0. 위치 짓기 — 다른 컴포넌트와의 관계

이 컴포넌트는 "측정 그 자체"(C3 system ID)나 "측정으로 sim을 고치는 것"(C3/C4)이 아니라, **배포 가능성(deployability)을 사다리식으로 올려가며 실기에서 직접 확인하는 게이트 시퀀스**다.

- **C3(학습 전 system ID)** 와 데이터를 공유한다: 여기서 수집한 step/sine/excitation trajectory가 C3의 actuator/관성/마찰 fitting 입력이 된다. 즉 테스트 T2·T3·T5는 "검증"인 동시에 C3의 "데이터 수집 패스"다.
- **C1(배포 파이프라인)** 의 export 산출물(JIT/ONNX 정책)을 T4에서 처음 실기 토크로 실행해 본다.
- **C4(학습 후 gap 축소)** 의 real2sim 비교 루프는 여기서 정의한 메트릭(추종 오차, 토크 곡선 RMSE 등)을 재사용한다.

**관통 원칙 — 사다리 + 안전 우선**: 가장 안전(통신·고정·무부하)에서 가장 위험(자유보행) 순으로 진행하고, **각 단계의 pass 게이트를 통과해야 다음 단계로 넘어간다.** 실패 시 다음 단계로 가지 않는다. 이는 Lee et al.(ANYmal, Science Robotics 2019)과 Hwangbo et al.이 강조한 "actuator/in-air/on-ground" 단계적 bottom-up 검증 흐름과, [PACE](https://github.com/leggedrobotics/pace-sim2real)의 "표준 encoder만으로 actuator→joint→로봇 순으로 식별·검증"하는 구조를 따른다.

**테스트 사다리 한눈에**

| # | 테스트 | Stage | 위험도 | 사람 개입 | 주 측정 |
|---|--------|-------|--------|-----------|---------|
| T1 | 통신/주파수 벤치마크 | 사전(로봇 무관) | 무 | - | 루프 주파수/지터/레이턴시 |
| T2 | 단일 관절 step/sine 추종 | Stage 1 | 저 | 고정 | pos/vel/torque |
| T3 | 정지(중력보상) 토크·마찰 측정 | Stage 1 | 저 | 고정 | 정지 torque/current |
| T4 | 무부하(공중 매단) 정책 실행 | Stage 1.5 | 중 | 매단 + e-stop | torque saturation, 정책 출력 |
| T5 | 고정 리그 excitation 재생 (system ID) | Stage 1 | 저~중 | 고정 | trajectory torque/pos |
| T6 | full robot: IMU·정적 균형·점진 보행 | Stage 2 | 고 | 거치대→te
ther→자유 | IMU, 관절, estimator |

> **재사용 명시**: T1~T5 는 `scripts/real2sim/` ZMQ 브리지(`sim_runner.py` 50Hz, `controller.py` GUI 슬라이더+모션 클립 재생) + `utils/benchmark.py`(주파수/레이턴시/지터) + `Isaac-R2S-HindLeg-v0`(순수 포지션 제어, RL 없음) 위에서 그대로 또는 약간의 일반화로 돌린다. T6 만 새 인프라(IMU 수신, estimator, on-robot 정책 runner)가 필요하다. ZMQ 브리지를 실기 모터 드라이버로 향하게 하려면 `utils/robot_interface.py`의 ROS2/CAN skeleton을 채우면 된다(Phase 2).

---

## Stage 1 — 고정 베이스 leg 리그

> 로봇 base(또는 단일 leg)를 리그에 단단히 고정. 낙상·자세 붕괴 위험이 원천 차단되므로 actuator 거동을 안전하게 한계까지 밀어붙일 수 있다. R_Skeleton Hind Leg real2sim 패턴이 정확히 이 단계.

### T1 — 통신 / 제어 루프 주파수 벤치마크

**목적**: 정책을 올리기 전에 **제어 루프 자체가 실시간 보장을 만족**하는지 확인한다. legged RL 정책은 보통 50Hz(IMU/저수준은 더 높게)로 동작하며, 주기 지터·드롭·레이턴시는 학습이 가정한 dynamics와 어긋나 곧바로 sim-to-real gap이 된다. 사다리의 0번째 칸 — 통신이 불안정하면 그 위 모든 테스트의 측정이 오염된다.

**근거**: 실기에서 IMU는 500Hz, 정책은 50Hz로 도는 것이 표준 구성이며, 정책은 IMU·encoder raw 측정에 의존한다([wheeled-legged deployment 보고](https://arxiv.org/html/2405.01792v1)). 제어 지연·latency를 sim에서 모델링하는 것이 reality gap을 줄이는 핵심이라는 점은 Tan et al.(RSS 2018, [Sim-to-Real: Learning Agile Locomotion for Quadruped Robots](https://www.roboticsproceedings.org/rss14/p10.pdf))에서 확립되었다 — 즉 실기 latency를 **측정**해 두어야 sim에서 같은 값을 주입할 수 있다(C3/C4 연결).

**sim 비교 메트릭**: `utils/benchmark.py`의 `FrequencyMonitor`(current_hz, period_ms, jitter_ms)와 `LatencyTracker`(mean/max/p95). sim_runner 단독(루프백) vs 실기 모터 드라이버 연결 시의 latency 분포를 비교 → 실기 추가 latency = `real_loop_latency - sim_loop_latency` 를 C3의 `action latency` DR/모델 파라미터로 넘긴다.

**안전 가드**: 모터 토크 enable 전에 수행(드라이버 통신만 검증). 통신 게이트 미통과 시 토크 enable 금지.

**판정 메트릭 (pass/fail)** (기존 real2sim 기준 그대로 채택):
- PASS: `current_hz ≥ 49.5` (목표 50Hz의 99%), `mean_latency < 5ms`, `max_latency < 15ms`, `jitter(σ) < 1ms`, 60초 연속 무드롭(frame drop = 0).
- FAIL: 위 중 하나라도 미달 → 버스 대역폭/스레드 우선순위/직렬화 점검 후 재측정.

---

### T2 — 단일 관절 step / sine 추종 테스트 (sim vs real)

**목적**: 정책·전신을 배제하고 **단일 관절 닫힌 루프(PD setpoint→실기 거동)** 가 sim과 얼마나 일치하는지를 본다. 지연(dead time), overshoot, steady-state error, 대역폭(주파수 응답)을 정량화한다. 이것이 actuator gap의 1차 진단이며, 어긋남의 부호/형태가 곧 system ID(C3)에 무엇을 고쳐야 하는지를 알려준다.

**근거**: 사인파 관절 목표로 파라미터를 fitting하고, **다른 gain + 랜덤 step**으로 재현 검증하는 절차는 legged sim2real system ID의 표준이다([Sampling-Based System ID with Active Exploration](https://arxiv.org/html/2505.14266), [PACE](https://github.com/leggedrobotics/pace-sim2real)). 실제 actuator는 대역폭 한계·tracking 부정확으로 sim보다 토크 프로파일이 덜 정확하다는 것이 gap의 주원인([systematic sim-to-real for diverse legged robots](https://www.researchgate.net/publication/395355402)).

**sim 비교 메트릭** (`Isaac-R2S-HindLeg-v0` 또는 일반화한 단일관절 R2S env에서 동일 setpoint 재생):
- **Step 응답**: rise time, dead time(지연), overshoot %, settling time, steady-state error(°).
- **Sine sweep(예: 0.5–5Hz)**: gain(진폭비) dB와 phase lag(°)의 Bode 비교 → 실기 대역폭과 위상 지연 추정.
- 동일 입력에 대한 sim/real **위치 궤적 RMSE(°)** 와 **토크 궤적 RMSE(Nm)**.

**안전 가드**: 관절별 소프트 limit 내 setpoint만. `sim_runner.py`의 slew-rate 제한을 실기 송신에도 적용(급격한 setpoint jump 금지). 각 관절 단독, 점진적 진폭 증가. 토크/전류 실시간 모니터 — effort limit의 일정 비율(예: 80%) 초과 시 자동 정지.

**판정 메트릭 (pass/fail)**:
- PASS: step에서 sim 대비 dead time 추가분 ≤ 1 control step(20ms), overshoot 차이 ≤ 5%p, steady-state error ≤ 관절 분해능 + 1°, 위치 궤적 RMSE가 사전 합의 임계(예: ROM의 5%) 이내. Sine에서 대역폭(–3dB) sim/real 비 ≥ 0.7.
- FAIL: 일관된 부호의 dead time(latency 모델 부족), 또는 큰 steady-state offset(마찰/중력보상 누락) → 측정값을 C3 actuator 모델/latency로 환류.

---

### T3 — 정지(중력보상) 토크 · 관절 마찰 측정

**목적**: 관절을 여러 정지 자세로 고정하고 **유지에 필요한 정지 토크**와, 아주 느린 운동에서의 **breakaway(정지마찰) / Coulomb·점성 마찰**을 실측한다. 정지 토크에서 중력 성분을 빼면 마찰/오프셋 단서가 나오고, 이는 관성(C3의 link mass/CoM)과 마찰 파라미터의 직접 입력이 된다.

**근거**: 질량·치수·actuator dynamics와 함께 **모터 마찰을 전용 실험으로 정밀 측정**해 simulator에 반영하는 것이 reality gap을 좁히는 핵심이라고 Tan et al.(RSS 2018)이 명시한다. ANYmal 계열은 마찰·hysteresis·lag가 비선형이라 motor current를 토크 proxy로 쓰기 부정확하다는 점이 알려져 있어([UAN/actuator net 논의](https://arxiv.org/html/2410.16591v1), [Hwangbo et al. Science Robotics 2019](https://www.science.org/doi/10.1126/scirobotics.aau5872)), **토크·전류 동시 실측**(우리 자원)이 중요하다.

**sim 비교 메트릭**:
- 자세별 **정지 유지 토크 vs sim의 중력 토크(URDF 관성 기반)** 차이 → 잔차 = 마찰/모델 오차.
- 초저속 왕복에서 **friction map**: breakaway 토크(Nm), Coulomb 마찰(상수), 점성 계수(Nm·s/rad).
- 측정 current↔torque 관계(토크 상수·비선형성) 곡선.

**안전 가드**: 정적 자세 유지 + 초저속만(동역학 없음). 리그 고정 필수. 토크 ramp는 effort limit 이내, saturation 근처 접근 시 정지.

**판정 메트릭 (pass/fail)**:
- PASS: 마찰 보정을 sim에 반영한 뒤 자세별 정지 토크 sim/real 잔차 RMSE가 effort limit의 ≤ 5%. breakaway/Coulomb/viscous 3-파라미터가 반복 측정 간 ≤ 10% 산포로 재현.
- FAIL: 큰 자세 의존 잔차(관성/CoM 오류 → C3 asset 보정) 또는 비반복적 마찰(케이블/하네스 간섭 점검).

---

### T5 — 고정 리그 excitation trajectory 재생 (system ID 데이터 수집)

**목적**: 단일 관절을 넘어 **다관절 동시 가진(excitation) trajectory** (chirp/멀티사인/PRBS, 또는 미리 만든 모션 클립)를 고정 리그에서 재생해, C3의 dynamic parameter identification(관성·마찰·actuator net 학습)에 쓸 **풍부한 dynamics 데이터**를 안전하게 수집한다. T2/T3가 "정적·단일"이라면 T5는 "동적·다관절"이다.

**근거**: actuator → 전신 in-air trajectory → 지상 보행으로 이어지는 **bottom-up dynamic parameter identification**이 ANYmal 계열의 표준 흐름이며([Hwangbo et al. 2019](https://arxiv.org/pdf/1901.08652)), **능동 excitation trajectory**로 파라미터 식별 정밀도를 높이는 것이 최근 정설이다([Active Exploration system ID](https://arxiv.org/html/2505.14266)). 식별용 trajectory와 검증용 trajectory를 **분리**(다른 gain·random step)해야 overfitting을 피한다([PACE](https://github.com/leggedrobotics/pace-sim2real)).

**sim 비교 메트릭**:
- 동일 excitation 입력에 대한 sim/real **관절 토크 시계열 RMSE / 정규화 NRMSE**, 위치·속도 궤적 RMSE.
- C3 모델(해석적 actuator 모델 또는 학습 actuator net)을 fit한 뒤의 **held-out trajectory 예측 오차** (식별/검증 분리 필수).

**안전 가드**: 진폭을 단계적으로 키움(저→고). slew-rate·토크 saturation 모니터링. 검증 trajectory는 식별과 다른 gain으로(데이터 누수 방지). 리그 고정.

**판정 메트릭 (pass/fail)**:
- PASS: held-out 검증 trajectory에서 토크 NRMSE가 사전 합의 임계(예: ≤ 10–15%) 이내이고, 식별 파라미터가 trajectory 종류 간 일관(부호·크기 안정).
- FAIL: 검증 trajectory에서 발산하는 예측 → 모델 구조 부족(선형 모델→actuator net 승격), 또는 미모델 latency/compliance.

> **C3 환류**: T2·T3·T5 데이터는 컴포넌트 3의 actuator 모델·관성·마찰 fitting 입력. 본 컴포넌트는 "검증 게이트", C3는 "그 데이터로 sim을 고치는 절차"로 역할 분리.

---

## Stage 1.5 — 무부하(공중 매단) 정책 실행 (배포 전 마지막 안전 게이트)

### T4 — 학습된 정책 무부하 실행 (suspended / on-stand)

**목적**: export된 RL 정책(C1 산출물, JIT/ONNX)을 **로봇을 공중에 매달거나 거치대에 올려 발이 지면에 닿지 않는 상태**로 처음 실행한다. 낙상·자기충돌·과토크 없이 **정책이 실기 관절에 보내는 토크/명령이 안전 범위 안인지, sim에서 본 동작 패턴(보행 위상·관절 궤적)과 닮았는지**를 확인한다. 자유보행(T6) 직전의 마지막 안전 게이트.

**근거**: 토크 기반 정책을 실기에 올릴 때 **scaled 출력 clipping**이 안전의 핵심이고, **moderate(보수적) joint feedback gain**이 sim-to-real 전이와 하드웨어 보호에 유리하다는 것이 경험칙([DecAP, torque-based legged](https://arxiv.org/pdf/2310.05714); [Deep Compliant Control](https://crl.ethz.ch/papers/hartmann2024deep.pdf)). in-air 단계를 명시적 검증 단계로 두는 bottom-up 흐름은 Hwangbo et al.의 actuator→in-air→on-ground 구조와 일치. (⚠ C1 메모: ONNX export가 `action_scale` 미반영 — GitHub #2636 — 이므로 T4에서 **실기로 나가는 최종 토크에 action_scale이 정확히 한 번** 곱해졌는지 반드시 육안·수치 확인.)

**sim 비교 메트릭**: 같은 (가짜) 관측을 sim과 실기에 동시에 먹였을 때(또는 동일 시드 rollout)
- **정책 출력 분포**: 명령 관절 위치/토크의 평균·분산·peak가 sim rollout과 동일 범위인가.
- **무부하 보행 위상**: 다리 swing 주기·관절 궤적 형태가 sim과 정성적으로 일치하는가(공중이라 접촉은 없지만 패턴은 보임).
- **saturation 빈도**: 토크가 effort limit에 닿는 비율.

**안전 가드 (가장 중요)**:
- 물리적 매단/거치대 + **즉시 e-stop**(사람 손) 상시 대기.
- **토크 saturation clip**: 정책 출력 토크를 effort limit의 보수적 비율(예: 50–70%)로 1차 clip 후 점진 상향.
- gain을 학습값보다 낮춰 시작(moderate gain) 후 단계적 복원.
- 첫 실행은 짧게(수 초), 이상 시 즉시 중단. action_scale 이중곱/누락 사전 점검.

**판정 메트릭 (pass/fail)**:
- PASS: 전 구간 토크가 clip 한계 미접촉(또는 saturation 비율 사전 임계 이하, 예 ≤ 2%), 발산·진동·고주파 chatter 없음, 관절 궤적이 sim 무부하 rollout과 정성 일치, NaN/통신 드롭 0.
- FAIL: 토크 saturation 빈발, 진동/발산, sim과 전혀 다른 패턴 → 정책 재학습/DR 강화(C4) 또는 actuator 모델 보정(C3)로 회귀. **T6(자유보행) 진입 금지.**

---

## Stage 2 — Full Robot 자유보행

> base를 풀고 발이 지면에 닿는다. 낙상 위험이 실재하므로 **거치대→tether(안전줄)→자유** 순으로 구속을 점진 해제한다. 여기서부터 **base IMU**(자세·각속도)가 관측에 들어오고, 직접 측정이 어려운 **base linear velocity는 estimator module로 추정**(parkour RMA 패턴: policy=추정값 입력, critic/discriminator=실제값).

### T6 — IMU 캘리브레이션 → 정적 균형 → 점진적 보행

세부 단계(각 하위 게이트 통과 후 다음으로):

**T6a — IMU 캘리브레이션 & 정렬**
- **목적**: base IMU의 bias·축 정렬·중력 방향을 보정. IMU 자세/각속도가 학습이 가정한 frame과 어긋나면 정책이 즉시 균형을 잃는다.
- **근거**: 정책은 IMU·encoder raw 측정에 의존하며 IMU는 보통 고주파(≈500Hz)로 동작([wheeled-legged](https://arxiv.org/html/2405.01792v1)). legged 로봇용 IMU spatial-temporal 캘리브레이션이 별도 연구 주제일 만큼 중요([A2I-Calib](https://arxiv.org/pdf/2503.06844)).
- **판정 메트릭**: 정지 수평 상태에서 roll/pitch bias ≤ 1°, gyro bias가 정적 드리프트 임계 이하, 중력 벡터가 base frame과 정렬. (PASS 전 T6b 금지.)

**T6b — 정적 균형 / 자세 유지 (default & 명령 자세)**
- **목적**: tether/거치대 하에서 정책(또는 PD 기본자세)이 **선 자세를 정적으로 유지**하는지. 미세 진동·드리프트·한쪽 쏠림을 본다.
- **근거**: moderate gain + 토크 clip 하의 정적 stand가 보행 전 표준 bring-up([Deep Compliant Control](https://crl.ethz.ch/papers/hartmann2024deep.pdf)).
- **sim 비교 메트릭**: base IMU roll/pitch 표준편차(°), 관절 토크 평균/분산 vs sim 정적 stand. **estimator 검증**: 정지 상태이므로 base lin_vel 추정값 ≈ 0이어야 함(estimator sanity check) — 추정 lin_vel이 0 근처에 머무는지로 estimator 건전성 1차 확인.
- **판정 메트릭**: roll/pitch σ ≤ 2–3°, 무발산, 정지 시 추정 lin_vel 절댓값이 임계(예 ≤ 0.1 m/s) 이내. FAIL 시 gain/관측 부호/IMU 정렬 재점검.

**T6c — 점진적 보행 (제자리→저속→명령 속도, tether→자유)**
- **목적**: 제자리 stepping → 저속 전진 → 명령 속도/방향 추종으로 단계 상향. tether로 시작해 안정 확인 후 자유.
- **근거**: 작은 명령부터 점진 상향하는 on-ground 검증은 Hwangbo et al.의 마지막 단계이자 표준. 상태 추정 + DR로 sim-to-real을 메우는 구성([Olympus](https://arxiv.org/pdf/2503.03574) 등 다수)의 실기 확인 단계.
- **sim 비교 메트릭**:
  - **명령 추종 오차**: 명령 vs 실제(estimator) base 속도 RMSE(m/s), yaw rate RMSE.
  - **estimator 정확도**: (가능 시 모캡/외부 추적과) 추정 lin_vel 비교, 또는 적분 위치 드리프트.
  - **gait 지표**: 접촉 타이밍/스텝 주기·CoT가 sim과 동일 범위인가.
  - **토크 saturation 비율**, IMU 자세 안정성.
- **안전 가드**: tether(안전줄)·낙하 보호, e-stop 상시. 명령 크기·속도를 단계적 상향, 각 단계 통과 후만 다음. 토크 clip 유지하며 점진 복원. 낙상 1회라도 발생 시 해당 명령 영역에서 중단·원인 분석.
- **판정 메트릭 (pass/fail)**:
  - PASS: tether 없이 60초+ 안정 보행, 명령 속도 추종 RMSE가 사전 임계(예 ≤ 0.15 m/s) 이내, 낙상 0, 토크 saturation 비율 임계 이하, IMU 자세 발산 없음, estimator 추정 lin_vel이 외부 기준과 합리적 일치.
  - FAIL: 낙상/발산/큰 추종 오차 → C4(DR 강화·real2sim 비교 루프)와 C3(actuator/관성 재식별)로 회귀.

> **estimator 환류**: T6b/T6c에서 추정 lin_vel과 실제(외부 기준)의 괴리는 estimator module 재학습 신호. 설계 원칙(policy=추정값, critic/disc=실제값)을 깨지 않도록, 배포 시 정책에는 추정값만 들어가는지 export 단계에서 재확인(C1).

---

## 요약 — 진행 게이트와 회귀 경로

```
T1(통신) ─pass→ T2(step/sine) ─pass→ T3(정지토크/마찰) ─pass→ T5(excitation)
                                                              │ (데이터)→ C3 system ID
  └──────────────── 모두 pass ───────────────→ T4(무부하 정책) ─pass→
                                                              │ fail→ C3/C4 회귀
  T6a(IMU) ─pass→ T6b(정적균형) ─pass→ T6c(점진보행/자유)
                                              │ fail→ C4(DR·real2sim), C3(재식별) 회귀
```

- **사다리·게이트**: 안전(통신)→고정 actuator→무부하 정책→자유보행. 각 단계 pass 없이는 다음 진입 금지.
- **인프라 재사용**: T1~T5 = `scripts/real2sim/` ZMQ 브리지 + `utils/benchmark.py` + `Isaac-R2S-HindLeg-v0`(일반화). T6만 IMU 수신·estimator·on-robot 정책 runner 신규(ROS2/CAN은 `robot_interface.py` skeleton 확장 = Phase 2).
- **C3/C4 연결**: T2·T3·T5는 검증인 동시에 C3 system ID 데이터 패스. 모든 sim/real 비교 메트릭(추종 RMSE, 토크 곡선 RMSE/NRMSE)은 C4 real2sim 비교 루프가 재사용.
- **공통 안전 가드**: 토크 saturation clip(보수→점진 복원), moderate gain 시작, slew-rate 제한, e-stop·tether, action_scale 이중곱/누락 확인(C1 ONNX #2636 주의).

### 참고 문헌
- Tan et al., *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots*, RSS 2018 — https://www.roboticsproceedings.org/rss14/p10.pdf
- Hwangbo et al., *Learning Agile and Dynamic Motor Skills for Legged Robots*, Science Robotics 2019 — https://www.science.org/doi/10.1126/scirobotics.aau5872 (arXiv https://arxiv.org/pdf/1901.08652)
- PACE: sim-to-real with standard joint encoders — https://github.com/leggedrobotics/pace-sim2real
- *Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real* — https://arxiv.org/html/2505.14266
- *Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots* — https://www.researchgate.net/publication/395355402
- DecAP: *Decaying Action Priors for Torque-Based Legged Locomotion* — https://arxiv.org/pdf/2310.05714
- Hartmann et al., *Deep Compliant Control for Legged Robots* — https://crl.ethz.ch/papers/hartmann2024deep.pdf
- A2I-Calib: *Anti-noise Active Multi-IMU Spatial-temporal Calibration for Legged Robots* — https://arxiv.org/pdf/2503.06844
- *Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots* — https://arxiv.org/html/2405.01792v1
- *Cycloidal QDD Actuator with Learning-based Torque Estimation* (UAN/actuator net) — https://arxiv.org/html/2410.16591v1

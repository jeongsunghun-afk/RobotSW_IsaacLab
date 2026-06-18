# 컴포넌트 4: 학습 중/후 gap 축소 (DR + real2sim 루프)

> **이 컴포넌트의 위치**: 컴포넌트 3(system ID)이 마스터 플랜의 무게중심이다. system ID는 actuator/관성/마찰의 **점추정(point estimate)** 을 측정해 sim의 nominal 파라미터를 실기에 맞춘다. 그러나 점추정에는 항상 **잔여 불확실성**이 남는다 — 측정 노이즈, 비정상성(온도·마모로 변하는 마찰/게인), 모델 구조 오차(접촉·백래시·케이블 토크 등 미모델링 항), 페이로드/지형 변화. 컴포넌트 4는 이 **"system ID 후 남는 gap"** 을 두 갈래로 다룬다.
>
> - **(a) 학습 중 (a priori robustness)**: domain randomization으로 잔여 불확실성 구간을 학습 분포에 흡수 → 정책이 측정값 근방의 파라미터 변동에 강건.
> - **(b) 배포 후 (a posteriori correction)**: real2sim 비교 루프로 실기 rollout과 sim을 정량 비교 → 어느 파라미터가 여전히 틀렸는지 역추적 → DR 범위/asset 재보정.
>
> **핵심 원칙**: DR은 system ID를 **대체하지 않는다. 보완한다.** system ID 없이 넓은 DR만 쓰면 정책이 지나치게 보수적("평균을 위한 정책")이 되고, system ID 없이 좁은 DR을 쓰면 실기 파라미터가 분포 밖(OOD)이라 전이 실패한다. 올바른 설계는 **measured-centered DR** — system ID가 측정한 값을 분포의 **중심**으로 두고, 측정 불확실성만큼만 ± 폭을 준다.

---

## 4.1 Domain Randomization 설계 (measured-centered)

### 4.1.1 설계 철학: 왜 measured-centered인가

DR의 원형은 Tobin 2017(시각)과 Peng 2018·Tan 2018(동역학)이다. 초기 DR은 "실기 파라미터를 모르니 넓게 뿌려서 그 안에 실기가 들어오게 한다"는 **uninformed wide randomization** 이었다. 그러나 넓은 DR에는 명확한 trade-off가 있다:

- **장점**: 분포가 넓으면 실기 파라미터가 분포 안(in-distribution)에 들어올 확률이 높아 전이 성공.
- **단점 (보수성 / conservativeness)**: 정책이 "모든 가능한 동역학에서 그럭저럭 동작하는" 보수적 평균 정책으로 수렴 → nominal 동역학에서의 성능(추종 정확도, 효율, 자연스러움)이 떨어진다. 극단적으로 넓으면 학습 자체가 불안정해진다.

→ **해법(우리 채택)**: 컴포넌트 3에서 system ID로 각 파라미터의 **측정 평균 μ와 측정 불확실성 σ**(또는 신뢰구간)를 얻은 뒤, DR 범위를 `[μ − kσ, μ + kσ]`(혹은 measured value × log-uniform scale)로 **좁게, 측정값 중심**으로 설정한다. 이렇게 하면 (i) 실기가 분포 안에 있고 (ii) 분포가 좁아 보수성이 최소화된다.

- **근거**: Peng et al. 2018("Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", ICRA) — link mass / joint damping / friction / controller gain을 랜덤화한 정책이 실기로 zero-shot 전이. LSTM(recurrent) 정책이 0.89 성공률로 feed-forward를 능가 → "랜덤화 + 시간 history로 online 동역학 추정"의 원형. Tan et al. 2018("Sim-to-Real: Learning Agile Locomotion for Quadruped Robots", RSS, Laikago) — DR을 **정확한 actuator model + latency 시뮬레이션 + system ID**와 **결합**해야 전이가 된다는 점을 명시. 즉 DR 단독이 아니라 system ID와 짝을 이뤄야 한다(우리 measured-centered 철학의 출처).
- **근거(보수성 trade-off)**: Sobanbabu et al. 2025("Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning", SPI-Active) 및 IsaacLab DR 토론(#2813) — "넓은 휴리스틱 DR은 과도하게 보수적인 정책을 낳는다. system ID/실측으로 현실적 랜덤화 경계를 설정하라. 좁게 시작해 점진적으로 넓혀라(curriculum)."
- **검증**: 동일 정책을 (i) nominal-only, (ii) wide-DR, (iii) measured-centered narrow-DR로 학습 → sim에서 추종 오차/CoT 비교(보수성 측정) + real2sim 루프(4.3)로 실기 전이 성공 여부 비교. measured-centered가 "nominal 성능 ≈ narrow"이면서 "전이 성공 ≈ wide"이면 목표 달성.

### 4.1.2 랜덤화할 파라미터 카탈로그 (기존 EventCfg 재사용·일반화)

우리 코드에는 이미 4~5항 DR이 `EventCfg`(IsaacLab manager-based event 패턴)로 구현되어 있다. 새 legged 로봇은 이 패턴을 **그대로 복제하고 system ID 측정값으로 범위만 교체**한다. 현재 구현(검증된 실측 범위):

| 카테고리 | mdp 함수 | 대상 | Go2 현재 범위 | Hind Leg 현재 범위 | mode |
|---|---|---|---|---|---|
| **마찰** | `randomize_rigid_body_material` | 전체 robot body(`.*`) | static (0.8,0.8) / dynamic (0.6,0.6) / restitution 0 | static (0.4,1.5) / dynamic (0.3,1.2) | startup |
| **base 질량** | `randomize_rigid_body_mass` | base | add (−1.0, +10.0) kg | add (−1.0, +3.0) kg | startup |
| **CoM 오프셋** | `randomize_rigid_body_com` | base | x±0.15, y±0.05, z±0.05 m | x±0.08, y±0.04, z±0.02 m | startup |
| **actuator gain** | `randomize_actuator_gains` | 전체 joint | stiffness ×(0.75,1.5), damping ×(0.3,3.0), log_uniform | stiffness ×(0.75,1.5), damping ×(0.75,1.5), log_uniform | reset |
| **외란 push** | `push_by_setting_velocity` | base | (Go2엔 미적용) | vx,vy ±0.5 m/s, 4~8s 간격 | interval |

> 파일: `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_env_cfg.py`(EventCfg, 라인 29–75), `.../hind_leg/hind_leg_env_cfg.py`(EventCfg, 라인 27–86). DR 함수 카탈로그: `_workspace/parkour_dr_research.md`.

**measured-centered로 일반화하는 방법** (새 로봇 도입 시 각 항을 어떻게 채우나):

1. **Dynamics — 관성(mass/CoM/inertia)**
   - system ID(컴포넌트 3)로 base/link 질량·CoM을 실측 → `add_base_mass`를 측정 평균 중심의 좁은 add 범위(예: ±측정σ + 페이로드 변동분)로, `randomize_com`을 측정 CoM ± 캘리브레이션 불확실성으로 설정. inertia tensor는 IsaacLab event엔 직접 항이 적으므로 CAD/측정값을 asset에 nominal로 박고(컴포넌트 3) 질량·CoM 랜덤화로 간접 흡수.
   - 근거: Peng 2018(link mass 랜덤화), Tan 2018(관성 system ID). 검증: real2sim에서 base 정지/스윙 시 base accel·자세 divergence가 분포 안에 들어오는지.
2. **Dynamics — 마찰(friction)**
   - 발–지면 마찰은 system ID로 정확히 측정하기 가장 어렵고 비정상적(지면별·습도별 변동) → **마찰만은 상대적으로 넓게** 두는 것이 합리적(measured-centered의 예외). 현재 Hind Leg의 (0.3~1.5) 폭이 이 철학.
   - 근거: RMA(Kumar 2021)·legged_gym(Rudin/ETH)이 마찰을 넓게 랜덤화하고 history로 흡수. 검증: 다양한 지면(매끈/거친) 실기 보행에서 슬립 발생률.
3. **Actuator gain (stiffness/damping)**
   - 컴포넌트 3의 actuator system ID로 실제 PD 게인(또는 actuator network)을 측정 → 측정 게인을 nominal로 박고, log_uniform ×(0.75,1.5) 같은 **곱셈 스케일**로 온도·마모에 의한 게인 drift를 흡수. 곱셈 스케일은 측정값 중심을 자동 보존하므로 measured-centered와 잘 맞는다.
   - 근거: Tan 2018(actuator model + 게인 랜덤화), Peng 2018(controller gain). 검증: 고정 리그(Stage 1) real2sim에서 step-response 추종이 randomize된 sim 분포 안에 드는지.
4. **Latency / delay (지연)**
   - 실기에는 센서→정책→actuator 사이 통신·연산 지연이 있다. sim은 기본 0지연 → 학습 시 action/obs에 1~수 step 랜덤 지연을 주입해야 전이가 된다. 우리 real2sim 브리지(`benchmark.py`)가 e2e latency를 실측하므로(4.3), 측정한 latency 분포를 학습 지연 범위로 직접 사용 → 이것도 measured-centered.
   - 근거: Tan 2018(latency 시뮬레이션이 전이의 핵심), legged_gym(control latency 랜덤화). 검증: 측정 latency를 sim에 넣은 정책 vs 안 넣은 정책의 실기 안정성(발진 여부).
   - 우리 코드 현황: EventCfg에 명시적 latency 항이 아직 없음 → **로드맵 항목**(action delay buffer 또는 obs delay를 env에 추가).
5. **Observation noise (관측 노이즈)**
   - encoder pos/vel, IMU 자세/각속도에 Gaussian 노이즈를 주입 → 실기 센서 노이즈에 강건. 노이즈 σ는 실기 센서 스펙 또는 정지 상태 실측 분산에서 가져온다(measured-centered).
   - 근거: legged_gym/ANYmal(actor obs에 Gaussian noise), Lee et al. 2020(ANYmal blind). 검증: 노이즈 주입 정책의 실기 자세 추정 안정성.
   - 우리 코드 현황: IsaacLab `ObservationCfg`의 `noise` 필드 또는 env에서 주입 → **로드맵 항목**(현재 DR은 dynamics 중심, obs noise 항 보강 필요).
6. **외란 push (external perturbation)**
   - `push_by_setting_velocity`로 주기적 속도 외란 → 밀침·미끄러짐 같은 미모델링 외란에 강건. Hind Leg는 이미 적용(±0.5 m/s, 4~8s). 새 로봇도 동일 패턴, 크기는 로봇 질량/보행 안정성에 맞춰 조정.
   - 근거: Peng 2018(perturbation), Tan 2018(perturbation). 검증: 실기 push recovery 테스트(컴포넌트 2의 무부하/부하 push 실험).

**2-stage 매핑**:
- **Stage 1 (고정 리그)**: latency·actuator gain·obs noise 랜덤화가 핵심(base는 고정이라 mass/CoM/push 무의미). real2sim 고정 리그 패턴에서 step-response 직접 비교로 게인·지연 DR 범위를 보정.
- **Stage 2 (full robot 자유보행)**: 위 6항 전부 활성. mass/CoM/마찰/push가 추가되고, estimator(4.2)가 history로 잔여 동역학을 online 흡수.

---

## 4.2 Adaptation 기법 (gap의 능동적 흡수)

DR이 "분포를 넓혀 강건하게"라면, adaptation은 "실시간으로 현재 동역학을 추정해 맞춰가는" 능동적 보완이다. 우리 코드의 estimator/history encoder가 이 역할을 한다.

### 4.2.1 RMA — online system ID via history encoder

- **원리**: Kumar et al. 2021("RMA: Rapid Motor Adaptation for Legged Robots", RSS). 학습 시 privileged 환경 파라미터(마찰·질량·지형 등)를 인코딩한 latent `z`를 만들고, 배포 시엔 측정 불가한 이 latent를 **최근 proprioception history**로부터 adaptation module(history encoder)이 회귀하도록 학습. 결과적으로 정책이 "최근 관측 history → 현재 동역학 latent"를 **수십 ms 안에 online으로 system ID**해 적응. A1 로봇에 sim-only 학습으로 fine-tuning 없이 다양한 지형 zero-shot 전이.
- **우리 코드 연결**: `actor_critic_parkour.py`의 `StateHistoryEncoder`(라인 23–78)가 정확히 이 패턴. proprio history(10/20/50 step)를 Conv1d로 압축한 `history_latent`를 actor 입력에 concat → 정책이 history로 동역학을 암묵 추정. DR(4.1)이 만든 파라미터 변동 분포를 history encoder가 online으로 식별해 흡수하는 것이 RMA의 본질이며, **DR과 adaptation은 짝**이다(DR 없이 history만으론 추정할 변동이 없고, history 없이 DR만이면 보수적 평균 정책에 그친다).
- **검증**: 학습 중 history latent → 실제 randomized 파라미터(마찰/질량) 사이의 회귀 정확도(probe). 실기에선 지면 바꿔가며 동일 정책의 보행 성공률(RMA 논문 메트릭).

### 4.2.2 Estimator module — base linear velocity 추정

- **원리**: 자유보행(Stage 2)에서 base linear velocity는 실기에서 직접 측정이 어렵다(IMU는 자세·각속도만 신뢰, 적분 적용 시 드리프트). 별도 estimator MLP가 proprioception으로부터 base lin_vel을 회귀 추정해 정책에 공급한다. **저장소 원칙: policy=추정값, critic/discriminator=실제값.** 학습 시 critic은 ground-truth lin_vel(privileged)을 받아 정확한 value를 학습하고, actor는 estimator의 추정값만 받아 배포 조건과 일치 → 배포 시 actor가 변함없이 동작.
- **우리 코드 연결**:
  - `rsl_rl/rsl_rl/modules/estimator.py` — `Estimator`(MLP, hidden [256,128,64], ELU). proprio(정책 obs) → priv_explicit 추정.
  - `actor_critic_parkour.py`(ActorCriticRMA) — `priv_explicit` = lin_vel_b(3, ×2.0) + ang_vel_b(3, ×0.25) = 6D. 학습/추론 모두 actor에 직접 concat. critic은 ground-truth priv_explicit + priv_latent을 받음.
  - `exporter_parkour.py`(`_TorchPolicyExporter`, 라인 183–187) — export 시 estimator를 번들: `priv_explicit_raw = estimator(proprio)` → normalizer → actor. 배포 정책이 추정 lin_vel로 자립.
  - **메모리 주의(검증된 함정)**: AMP 러너가 estimator를 빌드하지 않으면 actor가 실제 lin_vel을 직접 입력받게 되어 배포 불가 → estimator 빌드는 러너 책임, 누락 시 재학습 필요(메모리: "Parkour AMP 러너 estimator 누락 버그 fix").
- **근거**: Ji et al. 2022("Concurrent Training of a Control Policy and a State Estimator", RA-L) — 정책과 state estimator 동시 학습이 동적·강건 보행을 가능케 함. RMA의 latent 추정과 같은 계열(추정 대상이 명시적 lin_vel일 뿐).
- **검증**: estimator 추정 lin_vel vs sim ground-truth의 RMSE. 실기에선 모션캡처/외부 트래킹이 있으면 추정 lin_vel과 비교, 없으면 적분 IMU와의 일관성·보행 안정성으로 간접 검증.

### 4.2.3 AMP — 동작 분포 정규화가 gap에 기여하는 방식

- **원리**: AMP(Adversarial Motion Priors)는 discriminator가 reference motion 분포와 정책 motion을 구분하도록 학습하고, 정책은 그에 적대적으로 reference-like 동작을 내도록 보상받는다. sim-to-real 관점의 기여는 두 가지: (i) 정책 동작을 **부드럽고 일관된(자연스러운) 분포로 정규화** → 고주파 떨림·비물리적 토크 스파이크를 억제해 실기 actuator가 추종 가능한 동작만 생성(전이성↑), (ii) Cost of Transport를 낮춰 에너지 효율적 gait 유도.
- **근거**: Escontrela et al. 2022(AMP for legged), Wu et al. 2024("AMP for Go1, proprioception-only, flat→challenging terrain zero-shot transfer") — IMU+encoder만으로 실기 전이, AMP 정책이 velocity tracking·gait 일관성에서 baseline 능가. AMP는 본질적으로 DR/estimator와 **직교 보완**: DR=동역학 변동 흡수, estimator=상태 추정, AMP=동작 품질·전이성 정규화.
- **우리 코드 연결**: parkour/go2_imitation AMP 스택. **주의**: discriminator는 별도 extras 경로(amp_obs)로 **실제값**을 받는다(estimator·mirror data-aug 적용 안 함). policy=추정, disc=실제 원칙 일관.
- **검증**: 실기 보행의 동작 부드러움(joint jerk, 토크 스펙트럼 고주파 성분), CoT, velocity tracking 오차를 AMP-on/off로 비교.

---

## 4.3 real2sim 비교 루프 (a posteriori 측정·보정)

DR+adaptation으로도 남는 gap을 **배포 후 정량 측정**해 sim/asset/DR을 닫는 루프. 우리 `scripts/real2sim/` 인프라가 골격이다.

### 4.3.1 기존 인프라 (재사용)

| 파일 | 역할 |
|---|---|
| `scripts/real2sim/sim_runner.py` | IsaacLab 백엔드. 50Hz 루프, ZMQ PULL(setpoint)/PUB(state), slew-rate 제한. state = {pos(5), vel(5), torque(5), timestamp} |
| `scripts/real2sim/controller.py` | PyQt5 GUI. 관절 슬라이더 + 모션 클립(JSON) 재생, pos/vel/torque 실시간 그래프(30Hz), DataLogger 연동 |
| `scripts/real2sim/utils/zmq_bridge.py` | `ZMQSimBridge`(PULL 5555 CONFLATE / PUB 5556 SNDHWM=1), `ZMQControllerBridge`(PUSH/SUB). setpoint=5×float32, state=JSON |
| `scripts/real2sim/utils/robot_interface.py` | `RobotInterface` 추상화 + `MockRobotInterface` + `ROS2RobotInterface`(Phase 2 skeleton, 미구현) |
| `scripts/real2sim/utils/benchmark.py` | `FrequencyMonitor`(hz/period/jitter), `LatencyTracker`(e2e/state latency), `BenchmarkSession`. 게이트: hz≥49.5, mean latency<5ms, max<15ms, jitter<1ms |
| `scripts/real2sim/utils/data_logger.py` | CSV: timestamp + 5관절×(pos,vel,torque) + setpoint_applied. 200행마다 flush. 출력 `logs/real2sim/*.csv` |
| `scripts/real2sim/INTEGRATION_GUIDE.md` | Phase 1 ZMQ-only, sim(py3.11)/controller(py3.10) 분리 env, 50Hz, 실행 절차 |

> 현재 대상은 R_Skeleton Hind Leg(5관절). **일반화 방향**: `robot_interface.py`의 추상화를 통해 관절 수/이름을 로봇별로 파라미터화하고, `ROS2RobotInterface`를 실기 드라이버에 연결(Phase 2)하면 동일 루프가 임의 legged 로봇에 적용된다.

### 4.3.2 비교 루프 절차 (real-to-sim parameter back-out)

핵심 아이디어: **같은 입력(action/setpoint 시퀀스)을 실기와 sim에 모두 먹이고, state trajectory의 divergence를 측정해 어느 파라미터가 틀렸는지 역추적한다.**

1. **실기 rollout 로깅**: 실기(또는 고정 리그)에 setpoint/action 시퀀스를 보내고 `data_logger.py`로 pos/vel/torque/timestamp를 CSV 기록. 자유보행이면 IMU 자세·각속도도 함께.
2. **sim replay**: 동일 setpoint/action 시퀀스를 `sim_runner.py`로 sim에 그대로 재생(open-loop replay) → sim state trajectory 기록. PD 게인·초기조건을 실기와 일치시킴.
3. **divergence 측정**:
   - **per-joint trajectory divergence**: 관절별 pos/vel RMSE, torque RMSE(시간축 정렬 후). 어느 관절이 가장 벌어지는지 → 그 관절 actuator/마찰/관성 파라미터 의심.
   - **base divergence**(Stage 2): base 자세·각속도·(추정)속도 divergence.
   - **return gap**: 동일 정책의 sim return vs 실기 episodic 성과 차이.
4. **파라미터 역추적 (back-out)**: divergence 패턴 → 책임 파라미터 매핑.
   - 위치는 맞는데 **torque가 체계적으로 다르다** → actuator 게인/effort 모델 오차(컴포넌트 3 actuator system ID 재측정).
   - **고주파 추종 지연/발진** → latency 추정 오차(`benchmark.py`의 실측 latency를 DR 범위에 반영).
   - **느린 drift / 가라앉음** → 질량·CoM·마찰 오차(`add_base_mass`/`randomize_com`/material 범위 보정).
   - 발 슬립/접촉 튐 → 마찰·restitution.
5. **보정 적용**:
   - **asset 보정**(컴포넌트 3으로 피드백): nominal 값이 틀렸으면 asset cfg(`rga.py` 등)의 질량/CoM/게인/effort를 갱신.
   - **DR 범위 보정**: 실기가 sim 분포 **밖**에 있었으면 해당 항의 DR 범위를 measured-centered로 재설정(중심 이동 또는 폭 확대). 분포 안이었는데도 전이 실패면 다른 항 의심.
6. **재학습 → 재배포 → 재측정**: 닫힌 루프. 수렴 기준은 4.4 메트릭이 목표 이하로 떨어질 때까지.

- **근거**: BayesSim(Ramos 2019) 및 Sobanbabu 2025(SPI-Active) — 실기 rollout으로 randomization 파라미터 분포를 베이지안/샘플링으로 업데이트하는 real-to-sim 보정 계열. "Towards bridging the gap"(2025) — full-robot in-air trajectory 검증으로 실/sim 관절궤적 near-overlap 달성(고정 리그 replay 검증의 직접 선례). RSR/Real-Sim-Real loop(2025) — 닫힌 real2sim 루프 프레임워크.
- **검증**: 루프 반복마다 per-joint RMSE·return gap이 단조 감소하는지. 고정 리그(Stage 1)에서 step/sweep response가 실/sim near-overlap(예: pos RMSE < 측정 노이즈 수준)에 도달하면 actuator 보정 완료.

### 4.3.3 자동화 방향 (로드맵)

현재 인프라는 **수동 비교**(GUI로 보내고 그래프로 눈으로 봄)에 가깝다. 자동화 로드맵:

- **(a) replay 자동화**: 로깅된 실기 action CSV를 `sim_runner.py`에 open-loop로 자동 재생하는 배치 스크립트 → CSV in, sim-state CSV out.
- **(b) divergence 자동 산출**: 두 CSV(실기/sim)를 정렬·정합해 per-joint RMSE/torque-gap/return-gap을 자동 계산하는 분석 스크립트(`benchmark.py` 확장).
- **(c) 파라미터 sweep / fitting**: sim 파라미터(게인/마찰/질량)를 grid 또는 베이지안 최적화로 sweep해 divergence를 최소화하는 값을 자동 추정(SPI-Active/BayesSim식) → asset·DR 범위를 자동 제안.
- **(d) ROS2 실기 연동**: `ROS2RobotInterface` 구현으로 sim(mock) 대신 실기 state를 직접 로깅(Phase 2, 비목표지만 로드맵엔 포함).

---

## 4.4 검증 메트릭 종합 (sim-real gap 정량화)

각 기법의 효과를 한 곳에서 측정하는 지표 모음. 컴포넌트 4의 "검증" 라인들이 여기로 수렴한다.

| 메트릭 | 정의 | 무엇을 검증 | Stage |
|---|---|---|---|
| **per-joint trajectory divergence** | 동일 action replay 후 관절별 pos/vel RMSE | system ID·actuator DR·asset 충실도 | 1(우선)·2 |
| **per-joint torque divergence** | 관절별 torque RMSE(실/sim) | actuator 모델·effort·게인 정확도 | 1·2 |
| **return gap** | sim return − 실기 episodic 성과 | 정책 전체 전이성(종합) | 2 |
| **estimator RMSE** | 추정 base lin_vel vs ground-truth(또는 외부트래킹) | estimator module 정확도(4.2.2) | 2 |
| **latency / jitter** | e2e·state latency, 50Hz jitter(`benchmark.py`) | 통신·지연 모델, latency DR 범위 | 1·2 |
| **DR coverage(in-distribution rate)** | 실기 파라미터가 DR 분포 안에 든 비율 | DR 범위가 실기를 덮는가(measured-centered) | 1·2 |
| **history-latent probe accuracy** | history latent → randomized 파라미터 회귀 정확도 | RMA online 적응 작동 여부(4.2.1) | 2 |
| **동작 품질(jerk / torque 고주파 / CoT)** | joint jerk, 토크 스펙트럼, Cost of Transport | AMP 정규화·전이 가능 동작(4.2.3) | 2 |
| **외란 회복(push recovery)** | push 후 자세 회복 성공률 | push DR·강건성 | 2 |
| **루프 수렴 곡선** | real2sim 반복별 RMSE/return-gap 추이 | a posteriori 보정 루프 수렴(4.3) | 1·2 |

> **사용 원칙**: Stage 1(고정 리그)에서는 per-joint pos/vel/torque divergence와 latency가 1차 게이트 — actuator·게인·지연을 닫는다. Stage 2(자유보행)에서 return gap·estimator RMSE·동작 품질·push recovery가 추가되어 full 정책 전이성을 판정한다. real2sim 루프 수렴 곡선이 단조 감소하면 컴포넌트 4의 a posteriori 보정이 작동 중이라는 증거다.

---

## 참고문헌

- Tobin et al. 2017, "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS. (DR 원형 — 시각)
- Peng et al. 2018, "Sim-to-Real Transfer of Robotic Control with Dynamics Randomization", ICRA. arXiv:1710.06537. (동역학 DR + recurrent online 추정)
- Tan et al. 2018, "Sim-to-Real: Learning Agile Locomotion for Quadruped Robots", RSS. arXiv:1804.10332. (DR + actuator model + latency + system ID 결합, Laikago)
- Kumar et al. 2021, "RMA: Rapid Motor Adaptation for Legged Robots", RSS. arXiv:2107.04034. (history encoder online system ID)
- Ji et al. 2022, "Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion", RA-L. arXiv:2202.05481. (state estimator 동시 학습)
- Escontrela et al. 2022 / Wu et al. 2024, Adversarial Motion Priors for legged sim-to-real. (AMP 동작 정규화·전이)
- Ramos et al. 2019, "BayesSim", RSS. (real-to-sim 파라미터 분포 베이지안 업데이트)
- Sobanbabu et al. 2025, "Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning" (SPI-Active). arXiv:2505.14266. (active sysID + 보수성 trade-off)
- Zhao et al. 2020, "Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey", SSCI. arXiv:2009.13303. (서베이/taxonomy)
- Salvato et al. 2021, "Crossing the Reality Gap: A Survey on Sim-to-Real Transferability of Robot Controllers in RL", IEEE Access. (서베이/reality gap)

> 내부 인프라 참조: `_workspace/parkour_dr_research.md`(DR 함수 카탈로그), `source/isaaclab_tasks/.../go2/go2_env_cfg.py`·`.../hind_leg/hind_leg_env_cfg.py`(EventCfg), `scripts/real2sim/*`, `rsl_rl/rsl_rl/modules/estimator.py`·`actor_critic_parkour.py`, `source/isaaclab_rl/.../rsl_rl/exporter_parkour.py`.

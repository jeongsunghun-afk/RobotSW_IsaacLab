# Sim2Real 심화 리뷰 — 카테고리 B: Parametric (#7–#9)

> 리뷰어: Sim2Real 심화 리뷰어 (Opus 4.8)
> 작성일: 2026-06-18
> 대상 환경: IsaacLab / MJX (Go2, R_Skeleton 등)
> 방법: 3편 모두 arXiv 원문(HTML/abs)을 WebFetch로 실제 정독 후 작성. 스니펫 추론 배제.

**번호 교정 (중요)**: 작업 지시의 #8 링크 `arXiv 2403.03212` 는 **오류**다. 해당 ID는 입자물리 논문("modular ton-scale pixel-readout liquid argon TPC")이다. KAIST Shin et al. 토크-클리핑 논문의 실제 arXiv는 **2312.17507** ("Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion"), IEEE RAM 게재명은 "Reinforcement Learning for High-Speed Quadrupedal Locomotion With Motor Operating Region Constraints"이다. 본 리뷰는 올바른 논문(2312.17507)을 대상으로 작성했다.

---

## #7. Impedance Matching: Enabling an RL-Based Running Jump in a Quadruped Robot (Guan, Yu, Zhu, Kim — UMass Amherst, Ubiquitous Robots / IEEE 2024)

- **확정 링크**: https://arxiv.org/abs/2404.15096 (HTML: https://arxiv.org/html/2404.15096v1)
- **접근 분류**: Parametric — 주파수영역 임피던스 매칭으로 sim 파라미터(특히 PD gain·rotor inertia)를 실측에 정합. DR은 매칭 결과의 분산 위에 얹는 보조 수단.
- **우선순위**: **상**. 강복님이 보유한 "토크/전류 실측 + encoder" 자산과 가장 직접적으로 정합되는 방법론. 동역학 식별의 정량적 가이드라인을 제공.

### (1) 요약
PD 제어 관절을 2차 스프링-댐퍼계로 모델링하고, 실로봇과 시뮬레이터 각각에 chirp 입력을 가해 **Bode plot(크기만)**을 측정한 뒤, 두 Bode를 자연주파수 근방에서 MSE 최소화하도록 sim의 Kp/Kd를 grid search로 맞춘다(impedance matching). 동시에 geared actuator의 핵심인 **rotor inertia(반사 관성)**를 Isaac Gym `armature`로 주입하는 것이 고주파 전이의 결정적 요소임을 밝힌다. 매칭으로 얻은 관절별 게인 분산을 DR 범위 산정 근거로 사용. PPO + Net2Net 2단계 학습(걷기→점프)으로 MIT Mini-Cheetah Vision(12kg)에서 **전방 55cm gap / 최대 38cm 높이** 점프(궤적최적화 물리한계의 85%) 및 전후 2m/s·횡 1m/s 보행을 실기 시연.

### (2) Motivation
RL 보행은 성숙했지만 **진짜 동적(점프 등)** 동작은 sim-to-real gap 때문에 실기 시연이 드물다. 무차별 domain randomization은 (a) 어느 파라미터를 얼마나 흔들지 원칙이 없고, (b) 과도하면 정책이 보수적·저성능이 된다. 저자들은 "DR 범위와 파라미터 선택의 **구조적 가이드라인**"이 필요하다고 보고, 제어이론(임피던스/주파수응답)을 그 가이드라인으로 도입한다. 특히 geared 액추에이터의 rotor inertia가 고주파(점프 같은 충격성) 거동을 지배하는데 URDF 기본 모델은 이를 누락한다는 점을 지적.

### (3) Method + 수식 / 파이프라인 / 학습 / 실험

**관절 임피던스 모델 (2차계)**
- 식(1): `I·θ̈ + b·θ̇ = τ = Kp(θ_des − θ) + Kd(θ̇_des − θ̇)`
- 식(2) 폐루프: `I·θ̈ + (Kd + b)·θ̇ + Kp·θ = Kp·θ_des + Kd·θ̇_des`
- 임피던스 = PD에서 emergent하는 강성(Kp)·감쇠(Kd). 자연주파수 `ω_n = √(Kp/I)`, 감쇠비 `ζ = (Kd+b)/(2√(Kp·I))` 관점에서 sim·real을 정합.

**주파수응답(chirp) 식별 절차**
- 로봇 몸체를 고정 플랫폼에 고박, 측정 대상 관절 외 전부 PD로 정지 유지(무릎 측정 시 thigh link도 고정해 무릎 동역학 격리).
- chirp: 0.1–25 Hz 로그 sweep, 진폭 0.25 rad, 12관절 개별. 상한 25Hz는 sim 샘플링 50Hz의 Nyquist 기준.
- 측정/지령 위치 → MATLAB로 전달함수 추정 → **크기 Bode만** 사용(위상은 sim/real 시간지연 거동이 달라 무시).

**매칭(grid search)**
- sim에서 Kp 15–30 N·m/rad, Kd 0.1–0.7 N·m·s/rad 의 **50×50 그리드** 탐색.
- 기준: 자연주파수 근방 주파수대(무릎 0.1–15Hz, hip ab/flex 1–10Hz)에서 real vs sim Bode의 **MSE 최소** 게인쌍 선택.
- 결과(Table I) 평균: hip ab/ad Kp=20/Kd=0.45, hip flex Kp=17.5/Kd=0.4, knee Kp=21.5/Kd=0.55. (실하드웨어 기본 게인은 Kp=17/Kd=0.4였음 → 매칭으로 상향 보정.)

**Rotor inertia(핵심)**
- "geared 액추에이터 고주파 전이는 rotor inertia 없이는 매우 어렵다."
- Isaac Gym `armature`(질량행렬 대각에 가산되는 상수)로 반영. 반사 관성 = 모터 관성 × 기어비².
  - 무릎: `0.000072 × 9.33² ≈ 0.0063`
  - hip: `0.000072 × 6² ≈ 0.0026`

**DR 범위 = 매칭 분산에서 도출**
- Kp 섭동 `U(−2.0, +2.0) N·m/rad`, Kd 섭동 `U(−0.05, +0.05)` — 12관절 매칭 게인의 관측 분산 폭에 맞춤.
- 추가 DR: 지면마찰 U(0.05,3.0), shank 길이 U(0.18,0.20)m, base mass U(−0.4,1.6)kg, decimation(시간지연) U(−2,+2) step.

**학습**
- PPO. actor/critic [1024,512,256], estimator [256,128].
- 센싱/추론 50Hz, 액추에이터 PD 40kHz.
- 2단계: Phase1 걷기 → Phase2 Net2Net으로 점프 뉴런 추가(입력층에 0 row 삽입), jump command 0→1→0(걷기→점프→착지). 50/50 걷기/점프 혼합으로 catastrophic forgetting 방지.
- obs: 추정속도, 발접촉확률, base 각속도, projected gravity, x/y/yaw 속도지령, jump 지령, 관절위치/속도, 이전 action. 관절 히스토리 t, t−0.02, t−0.04.
- action: 12 목표관절각.
- 점프 reward(Table II): dense = −std(F_foot) (점프유도, 학습후 약화), sparse = `exp(−(V_liftoff − V_liftoff,des)²/2)` scale 250, 목표 이륙속도 2.5 m/s.

**실기 실험·평가**
- 플랫폼: MIT Mini-Cheetah Vision, 30cm/12kg, 관절 최대토크 17N·m. 액추에이터 모델은 rotor inertia·back-emf·건성마찰·버스전압 한계를 dynamometer 실측으로 반영.
- 평가: motion capture로 점프 높이(트래커 min/max 차)·거리(X-Y 변곡점 차) 측정, 속도당 5회.
- 결과: 전방 55cm / 후방 50cm / 횡 40cm gap, 최대 높이 38cm(=궤적최적화 45cm의 85%).
- 매칭 효과: naive 정책 대비 점프거리 표준편차 230%, 높이 표준편차 148% → 매칭이 일관성(분산↓) 개선.

### (4) 문제점·한계·개선 여지
- **위상 무시**: 크기 Bode만 매칭 → 시간지연/위상 동역학(고주파 안정성, 충격 타이밍)을 정합 못 함. 점프 같은 비선형 충격 구간에는 선형 LTI 가정이 깨질 수 있음.
- **선형 2차계 가정**: 건성마찰·기어 백래시·방향성 효율(DTE, #9) 같은 비선형은 b 하나로 뭉뚱그림. low-stiffness 컴플라이언트 영역에서 부정확.
- **per-joint 격리 식별**: 멀티바디 연성(coupled inertia, #8의 HFE/KFE coupling)이 누락. 실제 점프는 전관절 동시 부하.
- **매칭이 정적 자세 식별**: 실제 점프 시 부하·온도에 따라 게인이 변하나 1회 식별로 고정.
- **DR 범위가 작음**(Kp±2): 매칭이 정확하다는 전제. 식별 오류 시 robustness 마진 부족.

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder)
- **즉시 가치**: 강복님은 이미 **전류(→토크) + encoder(위치/속도)**를 실측 가능. 식(1)의 `τ = Kt·I`로 chirp 응답에서 **실측 토크 기반 Bode**를 직접 구성 가능 → 본 논문보다 한 단계 정확한(지령위치가 아닌 실토크) 임피던스 식별 가능.
- **Go2 적용 절차**:
  1. 다리 고박 → 관절별 chirp(0.1–25Hz, 0.25rad) → encoder θ, 전류 I 로깅.
  2. `Kt·I` → 실토크, MATLAB/Python(scipy) tf 추정 → 크기 Bode.
  3. IsaacLab actuator cfg의 `stiffness`(Kp)/`damping`(Kd) + **`armature`**(= 모터관성×기어비²)를 grid/optimize로 Bode MSE 최소화.
  4. 매칭 게인 분산 → cfg의 DR 범위(`actuator stiffness/damping randomization`)로 설정.
- **메모리 정합**: hind_leg I_eff 실측(hip0.163…calf0.030, 84배 비)은 이미 본 논문의 `armature` 주입 철학과 일치. inertia-scaled gain 권고도 `ω_n=√(Kp/I)` 일정화 관점에서 본 논문이 이론적 backing 제공.
- **주의**: armature는 IsaacLab `ImplicitActuator`에서 의미 추정이 위험(메모리 feedback). 실측 반사관성 값으로만 설정하고 런타임 검증할 것. parkour의 actuator_mode2 약화와 충돌 가능 — Go2 deploy 제약 메모 준수.

### sim 구현 난이도: **중**
- 필요 인프라: (1) 실로봇 chirp 가진 + encoder/전류 동기 로깅 리그(다리 고박 지그), (2) tf/Bode 추정 스크립트(scipy.signal 또는 MATLAB), (3) IsaacLab actuator cfg의 stiffness/damping/**armature** 노출(이미 지원), (4) grid/Bayesian 매칭 루프. 코어 미수정. armature는 cfg 한 줄. 가장 큰 비용은 실기 chirp 데이터 수집(하드웨어 의존).

---

## #8. Actuator-Constrained Reinforcement Learning for High-Speed Quadrupedal Locomotion (Young-Ha Shin et al. — KAIST, IEEE RAM 2024)

- **확정 링크**: https://arxiv.org/abs/2312.17507 (HTML: https://arxiv.org/html/2312.17507v1) | IEEE RAM: https://ieeexplore.ieee.org/document/10752794/
- **접근 분류**: Parametric — DC모터 회로식에서 유도한 **Motor Operating Region(MOR)** (사다리꼴 τ-ω 영역) 안으로 토크를 clipping하여 sim에서 실현불가 상태천이를 차단.
- **우선순위**: **상**. 고속/고부하에서의 토크-속도 한계가 sim-to-real gap의 주범임을 정량 입증. 강복님 토크/전류 실측으로 MOR 파라미터(R, Kt, Kv, V_bus) 직접 식별 가능.

### (1) 요약
URDF의 직사각형 토크 한계 대신, DC모터 전압식에서 유도한 **사다리꼴 MOR**(고속에서 토크 가용범위 축소, 회생제동 2·4사분면은 더 넓음)을 학습 중 제약으로 적용. policy의 목표관절각 → PD → 관절토크 → **gearbox 행렬**로 모터공간 변환 → MOR 위반 시 clip → 관절공간 복원 → 적용. 더불어 한쪽 모터 saturation이 성능 병목임을 발견해 대각보행 강제 gait reward로 토크를 다리에 고르게 분산. 결과: KAIST Hound(45kg)가 트레드밀 **6.5 m/s**(전기모터 4족 최고속, MIT Cheetah2의 6.4 갱신), 100m 19.87s, CoT 0.29.

### (2) Motivation
고속 주행은 모터가 토크-속도 한계의 모서리에서 작동. URDF는 토크를 속도무관 상수로 제한(직사각형)하지만 실모터는 고속에서 가용토크가 급감(사다리꼴). 이 불일치 때문에 sim 정책은 고속에서 "실재 불가능한 토크"에 의존하다 실기에서 추락(without MOR 정책은 5m/s에서 낙상). 즉 sim-to-real gap의 핵심을 **actuator의 τ-ω 가용영역 모델링 누락**으로 규정.

### (3) Method + 수식 / 파이프라인 / 학습 / 실험

**MOR 유도 (DC모터)**
- quasi-static(인덕턴스 변화 무시): `V_applied = (R/Kt)·τ + (1/Kv)·ω`
- 가용영역: `−V_bus ≤ (R/Kt)·τ + (1/Kv)·ω ≤ V_bus`
- 강자성 코어 비선형(전류-토크) 고려: 선형 이탈 20% 지점을 `τ_peak`로 정의 → τ-ω 평면에서 **사다리꼴**(회생제동 2·4사분면 폭 넓음).

**Gearbox 변환 (HFE/KFE coupling)**
- Hound는 HFE/KFE 액추에이터가 coupled → 관절↔모터 공간 변환 필요(superposition).
- 속도: `[ω_HFE,m; ω_KFE,m] = M⁻¹ [ω_HFE,j; ω_KFE,j]`, 토크: `[τ_HFE,m; τ_KFE,m] = M⁻ᵀ [τ_HFE,j; τ_KFE,j]`, 여기서 `M = [[1/G_HFE, 0],[−1/(G_HFE·G_KFE), 1/G_KFE]]`.

**클리핑 파이프라인**
- policy 목표각 → PD(**Kp=50, Kd=1.0**) → 관절토크 → gearbox로 모터공간 → MOR 위반 시 clip → 관절공간 복원 → 적용. 전류-토크 비선형은 실측 fit한 2차근사로 보상.

**Gait reward (토크 균형)**
- 한쪽 모터 saturation이 성능 병목 → 대각 교대접촉 강제:
  - `if C_f,1 ∧ C_f,4 ∧ ¬C_f,2 ∧ ¬C_f,3 → C_gait`
  - `elif ¬C_f,1 ∧ ¬C_f,4 ∧ C_f,2 ∧ C_f,3 → C_gait`
  - `else → 0`, C_gait=1.2 (1,4=한 대각쌍 / 2,3=반대 대각쌍).

**학습**
- PPO 계열 concurrent(policy/value [256,128,64], estimator [256,128]).
- reward 13항(Table I): lin_vel `6.0·exp(−‖V_des−V‖²)`, ang_vel `3.0·exp(−1.5(θ̇_des−θ̇)²)`, contact_vel −6.0, foot_slip −0.16, torque reg −2e-4, base motion `−4.0(0.8V_z²+0.2|θ̇_x|+0.2|θ̇_y|)`, action smoothness 2항.
- obs: IMU/encoder/contact → estimator가 body velocity·foot height·contact prob 추정. action: 목표관절각.
- **DR**: body mass/inertia/CoM 랜덤화. 커리큘럼 속도 U(−0.3,1.5)→U(−1.4,7.0)m/s. 에피소드 75%는 정지시작 후 1초간 점증, 25%는 이전 에피소드 종단상태 시작(최대 2m/s 지령변화). 고속일수록 lin/ang_vel reward 가중↑.
- 컴퓨트: 400 env, 4s 에피소드, 0.01s 제어, iter당 16만 샘플, RTX3080Ti 6시간.

**실기·평가 (KAIST Hound 45kg)**
- 온보드 P350 Tiny 100Hz, Elmo Platinum 100V/70A 2kHz EtherCAT, 81V 버스.
- ablation 최대속도(Table II): full **6.5**, − gait 6.0, − custom foot 5.5, − MOR **4.5**.
- 6.5m/s에서 CoT 0.29(15 stride), stance에 회생제동 관측. 야외 트랙 100m 19.87s, 9/9 성공.
- gap 분석(Fig9): Δreward_{sim-to-real}이 MOR 정책은 3.5m/s 이상에서 plateau, non-MOR은 계속 증가하다 5m/s 낙상.
- 경량 발: calf 질량 38.0%·pitch 관성 37.4%로 감축.

### (4) 문제점·한계·개선 여지
- **quasi-static 가정**: 인덕턴스 변화·열에 의한 R·Kt 변동 무시. 장시간 고속서 thermal derating 미반영.
- **coupling 모델 특수성**: gearbox 행렬은 Hound의 HFE/KFE 특정 구조. Go2(독립 관절)엔 단순화되지만, serial-elastic·벨트 구동은 재유도 필요.
- **clip의 미분 불연속**: MOR 경계에서 토크 clip은 gradient를 죽일 수 있음(우리 parkour clip 경험과 유사 risk) — 정책이 경계 근처서 학습 신호 약화 가능.
- **20% 선형이탈 τ_peak 정의**가 휴리스틱. 모터마다 재측정 필요.
- **회생제동 전력 한계** 미모델(배터리 충전 수용 한계).

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder)
- **MOR 파라미터 직접 식별**: 강복님 전류 I·전압·encoder ω로 `V = (R/Kt)τ + (1/Kv)ω` 의 R, Kt, Kv 회귀 fit 가능. V_bus는 배터리 실측. → Go2 모터의 실제 사다리꼴 MOR 도출.
- **IsaacLab 구현**: 현재 IsaacLab `ImplicitActuator`/`DCMotor`는 직사각형(`effort_limit`, `velocity_limit`)만 지원. MOR는 **post-PD 토크에 사다리꼴 clip**을 거는 커스텀 actuator 또는 env step 내 토크 후처리로 추가(코어 미수정, `direct/_common` 또는 task-level helper).
- **고속/고부하 task에 우선**: Go2 parkour의 점프·도약(pronk, 메모리)이나 고속 보행에서 효과 큼. 저속 평지엔 MOR 거의 비활성(영향 작음).
- **주의(clip risk)**: parkour total_reward clip·contact_duty clamp가 gradient 죽임을 이미 경험. MOR clip도 동일 risk → 경계 근방 soft penalty(위반 토크 페널티)와 병행 고려. 또한 "contact sensor obs 추가 금지"(sim-to-real) 메모 준수 — gait reward는 contact **state**만 쓰면 OK(obs 팽창 아님).
- **gait reward**: 우리 parkour의 fore-hind/대각 협응 부재 문제(메모리)와 직결. Hound식 대각 강제는 우리 feet_gait_pairing 재활성화·anti-pronk 설계와 합류 가능.

### sim 구현 난이도: **중상**
- 필요 인프라: (1) 모터 R/Kt/Kv 식별(강복님 전류+전압+encoder 데이터 + 회귀), (2) 사다리꼴 MOR clip을 거는 커스텀 actuator/토크 후처리 모듈(IsaacLab actuator 확장 — DCMotor 상속 또는 env step hook), (3) (Go2는 독립관절이라) gearbox 행렬은 단순 스칼라 기어비로 축약 가능. coupling 로봇이면 행렬 유도 비용↑. (4) gait reward(reward-worker, contact state만).

---

## #9. Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less Gear-Driven Humanoid (Masuda & Takahashi — Preferred Networks, IROS 2022)

- **확정 링크**: https://arxiv.org/abs/2204.03897 (HTML: https://arxiv.org/html/2204.03897)
- **접근 분류**: Parametric — 고감속 기어의 **방향성 전달효율(DTE)** 모델 + **실패 데이터 기반 system re-identification**으로 토크센서 없이 컴플라이언트 보행 전이.
- **우선순위**: **중상**. Go2(중감속)보다 R_Skeleton/휴머노이드·고감속·토크센서 부재 플랫폼에 특히 유효. 강복님 자산(전류→토크 proxy, encoder)이 DTE 식별에 정확히 맞물림. 컴플라이언스(low Kp) 전이가 핵심 차별점.

### (1) 요약
고감속 기어(353.5:1)는 backdrivability가 나빠 컴플라이언트(저게인) 제어를 sim-to-real 전이하기 어렵고, 불안정 보행 중엔 표준 system ID가 실패한다. 저자는 (a) 기어 손실을 **forward/backward 비대칭 효율(DTE)**로 모델링한 actuator + Stribeck 마찰, (b) **2단계 식별**(정적 excitation으로 1차 식별 → 전이한 정책이 task 수행 중 만든 **실패 포함 데이터**로 Wasserstein 기반 re-identification)을 제안. SAC로 ROBOTIS-OP3(토크센서 없음)에서 외란·5mm 불균일 지면 보행을 힘/접촉센서 없이 달성. DTE·재식별 모두 제거하면 전이 실패(0/3), 둘 다 있으면 3/3 성공.

### (2) Motivation
- 토크센서 없는 저가 휴머노이드(Dynamixel 구동)에서 **컴플라이언트(외란 흡수) 보행**을 RL로 전이하려면 저PD게인 영역의 정확한 동역학이 필수.
- 고감속 기어는 forward(모터→관절)와 backward(외력→모터) 전달효율이 크게 다름(backdrivability 저하) → 대칭 효율 가정 시 컴플라이언스 거동이 sim≠real.
- 보행은 본질적으로 불안정 → 안정 excitation만으론 task 분포를 못 덮음. **실패까지 포함한 task 데이터**로 재식별해야 분포정합.

### (3) Method + 수식 / 파이프라인 / 학습 / 실험

**DC모터 + DTE actuator 모델**
- `τ = Kt·I`, `I = (V_pwm − V_backemf)/R_ter`, `V_backemf = Kt·q̇`.
- **DTE(방향성 전달효율)** brake torque `τ_brake(τ_m, τ_a, η_fw, η_bw)`:
  - forward 구동(모터 토크 지배): 손실 `−L_fw = −(1−η_fw)·τ_m`
  - backward 구동(부하 토크 지배): 손실 `−L_bw = −(1−η_bw)·τ_a`
  - antagonistic(상충): `τ_m + τ_a + τ_brake = 0`
  - → backdrivability를 η_bw로 명시. (테스트 기어 353.5:1)
- **Stribeck 마찰**: `τ_fric = f_c + s·(f_s − f_c) + k_v·|q̇|`, `s = exp(|q̇|)/q̇_static` (정→동마찰 전이).
- **가변게인 PD**: 정책이 목표각 + **P게인(0.1–6.0)**을 관절별 지령, D=0.1 고정. 저게인으로 토크센서 없이 passive compliance 구현.

**2단계 system identification**
- Phase1 (정적 excitation): 무릎 1.47→0.6rad squat ~0.5Hz. 목적함수 식(5) `min_φ L_exc(O_sim(φ), O_real)`, 오차 식(7) = base 자세/속도 오차 + 관절 각/속도/전류 오차(N관절×T스텝). **TPE(Optuna) 2000 trial**.
- Phase2: SAC로 정책 학습 후 실기 전이.
- Phase3 (실패 기반 재식별): 다목적 식(6) `min_φ {L_exc(O_sim,O_real), W(R_sim, R_real)}` — 1목적: excitation 오차 유지, 2목적: **reward 분포를 Wasserstein 거리로 정합**(전이 정책이 task 수행 중 만든 실패 포함 데이터 사용). **NSGA-II 3000 trial**.

**DR / 식별 파라미터 범위(Table I)**
- Kt [0.003,0.009], R_ter [4.0,9.0]Ω, armature [0.0025,0.011]kg·m², η_fw=1.0 고정, **η_bw [0.6,1.0]**(핵심 DTE), f_c [0.01,0.25]Nm, k_v [0.0025,0.15], f_s = f_c+[0,0.25], base mass [0,0.5]kg, CoM x/z ±0.02m. (DR baseline은 이 범위 균등샘플.)

**학습**
- **SAC**(연속). FC 2-hidden ReLU: balancing 256, walking 1024. balancing 1M step(~5h), walking 3M step(~15h).
- action 20D(10관절 × [목표각, P게인]).
- obs: 발 6 keypoint 위치/속도 36 + 이전 지령 20 + body Euler 3 + body 각속도 3 + (walking) 주기 phase 2. history n=1(balance)/n=3(walk).
- reward 식(8): `R = K_bipedal·R_bipedal + K_cmd·R_cmd + K_smooth·R_smt + 1`. K_bipedal 0/0.4, K_cmd 0.6(속도추종), K_smooth 0.1(action smooth + 전류최소화).
- sim: MuJoCo 1ms timestep.

**실기·평가 (ROBOTIS-OP3)**
- 51cm/~3.5kg/20DOF, Dynamixel XM430-W350(stall 4.1Nm@12V), PD 125Hz, 정책추론 31.25Hz.
- balancing: 보드 ±6° tilt(학습 ±10°), 외란 1kg를 회전중심 0.23m에 배치. 성공기준 sim reward의 ≥80%(10시도).
- walking: 0.3m/s, 1.0s cycle, support:swing 0.7:0.3, **5mm 토이브릭 불균일 지면**. 성공기준 10s 보행 8/10 무낙상. ROBOTIS 내장 보행은 불균일 지면 전부 실패.
- 결과: excitation-only 정책 0/3 → 재식별 후 3/3 성공. **DTE 제거 시 재식별해도 실패**, **DR baseline도 0/3 실패**. 힘/접촉센서 없이 외란·불균일 지면 통과(backdrivability 활용).

### (4) 문제점·한계·개선 여지
- **소형 서보 특화**: Dynamixel(4.1Nm) 스케일. Go2/대형 BLDC의 열·전류 비선형엔 재식별 필요.
- **재식별 비용**: NSGA-II 3000 trial × sim rollout = 무겁다. 실패 데이터 수집도 실기 시도 필요(낙상 리스크).
- **η_fw=1 고정**: forward 손실 무시(저감속이면 부정확). 우리 중감속엔 η_fw도 식별 권장.
- **전류 obs 의존**: obs에 관절 전류 포함(우리 "contact sensor obs 금지" 정신과는 다른 축이지만, 실기 전류센서 가용성 전제). 전류 노이즈/드리프트에 민감.
- **저속(0.3m/s)·소외란** 검증 — 고속/대외란 일반화 미입증.

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder)
- **DTE 식별이 강복님 자산과 정합**: forward/backward 효율은 전류(→토크)·encoder로 직접 측정 가능(저속 정·역 부하 sweep). η_bw가 backdrivability/컴플라이언스 거동을 지배 → R_Skeleton·휴머노이드 컴플라이언트 task에 핵심.
- **MJX/MuJoCo 친화**: 본 논문은 MuJoCo 기반. 우리 MJX 트랙에서 DTE/Stribeck를 actuator gainprm/biasprm 또는 커스텀 actuator로 이식하기 비교적 직접적.
- **실패-기반 재식별**: 우리 parkour/recovery처럼 **불안정 task에서 표준 ID 실패** 상황에 적용 가능. 단, Wasserstein reward 정합 + NSGA-II 파이프라인은 신규 인프라.
- **가변 P게인 action**: 정책이 컴플라이언스(Kp)를 능동 조절 → Go2 fall-recovery(메모리: "고정T 느린복구"→strong penalty 추적)나 안전접촉 task에서 충돌흡수에 유용. 단 action 차원 2배(목표각+게인) → IsaacLab actuator가 per-step variable stiffness 지원 필요.
- **주의**: η_fw=1 고정은 우리 중감속(Go2 기어비 낮음)엔 부적절 — η_fw도 식별. 전류 obs는 sim-to-real 가용성 확인 후.

### sim 구현 난이도: **상**
- 필요 인프라: (1) DTE actuator(forward/backward 비대칭 효율) + Stribeck 마찰 모델을 IsaacLab actuator 또는 MJX actuator로 구현(커스텀, 코어 미수정). IsaacLab 기본 actuator엔 비대칭 효율 개념 없음 → 신규. (2) 가변 stiffness action 경로(per-step Kp 지령) — IsaacLab `ImplicitActuator`는 정적 게인이라 확장 필요. (3) 2단계 식별 파이프라인: TPE(Optuna) + **NSGA-II 다목적 + Wasserstein reward 정합** — 식별 하니스 신규 구축, sim rollout 수천 회. (4) 실기 excitation + 실패 데이터 수집 리그. 가장 무거운 항목은 재식별 다목적 최적화 + variable-stiffness actuator.

---

## 종합 비교 (카테고리 B)

| | #7 Impedance Matching | #8 MOR Torque Clipping | #9 DTE + 실패 재식별 |
|---|---|---|---|
| 핵심 식별 대상 | Kp/Kd/rotor inertia (Bode 매칭) | R/Kt/Kv/V_bus (사다리꼴 MOR) | η_fw/η_bw(DTE), Stribeck, Kt/R |
| 식별 신호 | chirp Bode(크기) | τ-ω 회로식 회귀 | 정적 excitation + 실패 task 데이터 |
| 핵심 기여 | 고주파 전이(점프), DR 가이드 | 고속 전이(τ-ω 한계) | 컴플라이언스 전이(토크센서無) |
| RL | PPO + Net2Net | PPO concurrent | SAC + 가변게인 |
| 플랫폼 | Mini-Cheetah(12kg) | KAIST Hound(45kg) | ROBOTIS-OP3(3.5kg) |
| 강복님 자산 정합 | 매우 높음(전류→실토크 Bode) | 높음(전류로 R/Kt/Kv) | 높음(전류로 DTE 효율) |
| sim 난이도 | 중 | 중상 | 상 |

**적용 권고 순서**: #7(즉시·저비용, armature+게인 매칭) → #8(고속/도약 task 발생 시 MOR clip) → #9(R_Skeleton/휴머노이드 컴플라이언스 트랙에서). 셋 다 강복님 토크/전류 실측을 식별 입력으로 공유하므로, **1회 chirp+정·역 부하 sweep 실험으로 세 방법의 파라미터(Kp/Kd/armature, R/Kt/Kv, η_fw/η_bw)를 동시 추출**하는 통합 식별 프로토콜이 비용 효율적이다.

### 출처
- #7: https://arxiv.org/abs/2404.15096 , https://arxiv.org/html/2404.15096v1
- #8: https://arxiv.org/abs/2312.17507 , https://arxiv.org/html/2312.17507v1 , https://ieeexplore.ieee.org/document/10752794/
- #9: https://arxiv.org/abs/2204.03897 , https://arxiv.org/html/2204.03897

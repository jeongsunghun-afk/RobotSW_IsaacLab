# 카테고리 C 심화 리뷰: System ID / Differentiable Simulation (#13–#15)

> 작성: Sim2Real 논문 심화 리뷰어 (강복님 IsaacLab/MJX 적용 관점)
> 원문 3편을 arXiv PDF / HTML 전문으로 직접 정독하여 작성. 스니펫 추론 아님.
> 대상 환경: IsaacLab(RL 학습) + MuJoCo MJX(미분가능 SysID)

---

## #13. Achieving Precise and Reliable Locomotion with Differentiable Simulation-Based System Identification (Kovalev, Chaikovskaia, Davydenko, Gorbachev — MIPT, arXiv 2025; IROS 2025)

- **확정 링크**: https://arxiv.org/abs/2508.04696 (PDF: https://arxiv.org/pdf/2508.04696). 코드: https://wavegit.mipt.ru/Slavoch/mjx_sysid . IROS 2025 게재 (참조 [13]에서 본인들이 IROS 2025로 명기).
- **접근 분류**: 미분가능 시뮬(MJX) 기반 gradient SysID — actuator built-in 파라미터(armature/frictionloss/damping) 식별. **토크 센서 불필요, trajectory(위치/속도)만 사용**.
- **우선순위**: ★★★ (강복님 "토크/전류 실측 없이 gradient 식별 경로"의 가장 직접적이고 미니멀한 레퍼런스. 가장 먼저 재현 권장)

### (1) 요약
이족(bipedal) 보행에서 sim-to-real trajectory drift를 줄이기 위해, **시스템 식별 단계를 RL 학습 루프 안에 통합**하는 control framework를 제안한다. 핵심은 전통적 SysID가 의존하는 직접 토크 측정을 제거하고, **오직 trajectory(위치·속도)와 control input만으로** 시스템 파라미터를 추정한다는 점이다. 미분가능 시뮬레이터 MuJoCo-XLA(MJX)로 시뮬-실제 trajectory 불일치를 최소화하도록 파라미터를 최적화한다. Mini π 로봇에서 검증: rotational deviation 75% 감소, command 방향 travel distance 46% 증가(baseline RL 대비). 식별 대상은 mass/inertia 같은 기본 물성과 friction/delay 같은 비선형 거동(신경망 근사 가능)이지만, 실제 실험에서는 MJX built-in의 armature/frictionloss/damping을 최적화했다.

### (2) Motivation
- 전통 SysID는 토크 센서에 의존 → (a) 센서가 비싸고 노이즈가 많으며 항상 장착되어 있지 않음, (b) 로봇 동작 중 실시간 측정이 어려움, (c) 통제된 실험 환경을 요구해 실용성 낮음.
- Domain randomization(DR)은 대안이지만 시뮬 모델 자체를 개선하지 못하고 over-conservative(agility 희생) 정책을 낳음.
- ActuatorNet([11], Hwangbo et al.)은 토크 데이터로 motor dynamics를 학습 → 여전히 토크 측정 의존.
- 따라서 "토크 측정 없이, 미분가능 시뮬로 RL 루프에 통합 가능한, scalable한 SysID"가 빈 자리. MJX가 가장 널리 쓰이는 RL 시뮬이자 미분가능이므로 토대로 선택.

### (3) Method + 수식 + 학습 방법 + 실험 방법

**문제 정의 (Problem Formulation).** 상태 `s_i`, 제어 입력 `a_i`. 실제 시스템을 완벽 모델링하는 유일 파라미터 `z*`(예: inertia)가 존재한다고 가정하면 예측 다음 상태가 실제와 일치:
```
s_{i+1} = s'_{i+1} = Φ_{z*}(s_i, a_i, δ)      ... (1)
```
`Φ_z`는 적분 스킴(integration scheme), `δ`는 prediction step size. 식별을 다음 trajectory-matching 최적화로 공식화:
```
z* = argmin_z  Σ_{i=1}^{N} ||s'_i - s_i||²_2
       s.t.  s'_{i+1} = Φ_z(s'_i, a_i, δ),  s'_0 = s_0      ... (2)
```
수치 적분 오차 누적을 막기 위해 trajectory를 길이 N의 다중 segment(M개)로 나눠 segment 내부에서만 매칭:
```
z* = argmin_z  Σ_{j=0}^{M} Σ_{i=0}^{N} ||s'_{i,j} - s_{i,j}||²_2
       s.t.  s'_{i+1,j} = Φ_z(s'_{i,j}, a_{i,j}, δ),  s'_{0,j} = s_{0,j}      ... (3)
```
→ **segment-wise rollout으로 integrator error를 차단**하는 것이 실용적 핵심 트릭. 실험에서 N=4(segment 길이), Euler integrator 사용(RK4 대비 정확도 손실 없이 4배 빠른 수렴).

**무엇을 식별하는가.** 처음엔 motor output torque를 신경망(z로 파라미터화)으로 근사 시도 → high variance(특히 data-sparse 영역), 다양한 데이터 수집이 위험·손상 유발. 그래서 **MJX built-in motor 파라미터 z = {armature, frictionloss, damping}** 직접 최적화로 선회(신경망 유연성은 포기하되 대규모 데이터셋 불필요).

**데이터 생성 (Dataset generation).**
- Mini π 로봇(HTDW-5047-36-NE 모터, High Torque Robotics [25])에서 실제 모터 1개를 추출 → 테이블에 고정, 알려진 mass/inertia의 rod를 부하로 장착.
- 목표 속도 `{v_i^des}`를 Fourier mode 합으로 생성(mode 수·계수·주파수 랜덤 샘플 → 다양성 확보), 모터 max velocity로 clip, 적분하여 desired angle `{q_i^des}` 획득.
- 입력 명령 `a_i = q_i^des`  ... (4), PD 제어기로 처리.
- **PD gain을 의도적으로 낮춤**(기본 P=40~80 → Kp=20, Kd=1): motor intrinsic dynamics가 더 두드러지게 하기 위함.
- 실제 모터에 action 인가 → 상태 `s_i = [q_i, v_i]^T`(측정 각도·속도) ... (5) 수집. 전체 dataset을 1ms time step(δ)으로 resample.

**모터 식별 결과.** train/test 분리, 학습셋은 길이 N=4 segment로 분할. MJX는 초기 파라미터에 매우 민감(비현실적 초기값 → next-state 예측 불안정). 최적 z*는 초기 추정 대비 loss ≈20% 감소. 최적화 파라미터로 시뮬 모터 초기화 후 unseen test set의 제어 입력 적용 → **optimized vs baseline(frictionloss=0, damping=0, armature 최소) 비교 시 MSE가 약 8배 작음**(Fig.4). 파라미터(frictionloss, damping, armature)는 epoch에 따라 부드럽게 수렴(Fig.3).

**실기 검증 (전체 로봇).** pre-trained RL 정책을 (i) 실제 (ii) baseline 시뮬(joint param=0) (iii) optimized 시뮬 세 곳에서 테스트. 각 7초 × 10회, 시간·관절위치·관절속도만 의존(모든 control input은 0). 결과: 실제 로봇은 시간에 따라 점진적 drift → optimized 모델은 이를 재현, baseline은 못 잡고 넘어짐. **mean Euclidean displacement: optimized 0.16m vs baseline 1.13m (86% 개선, Table I)**.

**RL 정책 학습 적용.** LeggedGym([11]) 내장 reward로 randomization 없이 optimized 모델 위에서 walking 정책 학습 vs unadjusted baseline 모델 정책. 실기 배포 시 angular velocity scale만 실제에서 절반으로(IMU 노이즈 보정), command linear velocity는 max. 각 7초 × 10회. **결과(Table I): baseline은 과도 회전으로 1.12m 전진·rotational deviation 109°; refined는 1.64m 전진·deviation 27°. → travel +46%, rotation -75%**.

### (4) 문제점·한계·개선 여지
- **모델 가정 한계(저자 명시)**: 식 (1)은 시뮬-실제 차이가 전적으로 파라미터 벡터 z로 포착된다고 가정. mass/inertia 변동이나 외란이 z에 인코딩 안 되면 식 (1)이 깨져 부정확. 정확한 등식은 비현실적이며, Φ_z가 "충분히 가까운" 근사이기만 하면 됨.
- **단일 모터 식별 중심**: 전체 로봇 multi-joint 동시 식별이 아니라 추출 모터 1개로 dataset 생성. 전신 결합 dynamics·관절 간 coupling 식별은 미확장.
- **contact dynamics 미식별**: kinematic trajectory만 사용 → 접촉 모델(지면 friction, 임팩트)은 식별 대상 밖. terrain interaction 확장은 향후 과제.
- **MJX 초기값 민감성**: 비현실적 초기 추정 시 발산. 좋은 nominal 모델이 사전 필요.
- **규모/검증 폭**: Mini π 단일 플랫폼, 단순 평지, 7초 짧은 trial. agility task나 거친 지형 일반화 미검증.
- **discussion의 향후 방향**: historical motor data로 unobservable state(온도 의존성, command delay) 근사; 추정 파라미터를 constrained DR range로 활용해 DR과 결합.

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder, gradient 식별 경로)
- **가장 직접적인 토대**: 강복님이 가진 encoder(위치/속도)만으로 actuator 식별이 가능함을 가장 작은 셋업(모터 1개 + 알려진 rod 부하)으로 증명. Go2 같은 12-DOF 직접 적용 전, **단일 actuator bench 식별**부터 시작하는 안전 경로를 제공.
- **토크/전류 실측의 위치**: 본 논문은 "토크 불필요"가 셀링 포인트지만, 강복님은 토크/전류 실측 capability를 보유 → **본 방법의 gradient 식별을 ground-truth(토크 기반 검증)와 cross-validate** 가능. 즉 "trajectory-only 식별값"이 "토크 기반 식별값"과 얼마나 일치하는지 정량화하면 본 논문이 못한 검증을 보강.
- **MJX 식별 → IsaacLab 학습 분리 워크플로**: 식별은 MJX(미분가능)에서, RL 학습은 IsaacLab에서. 식별된 armature/frictionloss/damping을 IsaacLab actuator cfg(`ImplicitActuator`의 armature/friction/damping)로 매핑하여 주입. (CLAUDE.md의 `*_env_cfg.py`에 반영, 코어 미수정)
- **핵심 트릭 차용**: (a) segment-wise rollout(N=4)으로 integrator drift 차단, (b) 식별용 데이터는 **PD gain을 낮춰** intrinsic dynamics 노출, (c) Fourier-mode 속도 궤적으로 excitation 다양성 확보. 이 3개는 강복님 식별 파이프라인 설계 시 즉시 채택 권장.
- **주의**: parkour처럼 contact-heavy task엔 본 방법만으로 부족(contact 미식별). flat locomotion / actuator gain 보정엔 최적.

### sim 구현 난이도: **하~중**
- 필요 인프라: MuJoCo MJX(JAX), 단일 actuator XML 모델, encoder 데이터 로깅(이미 보유), JAX optimizer(Adam). 코드 공개(MIPT git)되어 재현 부담 낮음.
- 난이도가 "중"으로 오르는 지점: MJX gradient의 초기값 민감성 튜닝, IsaacLab actuator cfg와 MJX 파라미터의 의미 매핑(armature/damping 단위 일치 확인), 전신 multi-joint 확장.

---

## #14. Closing Sim-to-Real Gap for Heavy-loaded Humanoid Agile Motion Skills via Differentiable Simulation — HALO (Wang, Zhang, Xie, Yu, Song, Bai, Zhu — Zhejiang Univ / TeleAI / SJTU / Lumos Robotics, arXiv 2026)

- **확정 링크**: https://arxiv.org/abs/2603.15084 (PDF: https://arxiv.org/pdf/2603.15084). Project: "HALO-Humanoid". **프레임워크명 HALO = HeAvy-LOaded humanoid motion control**.
- **접근 분류**: 미분가능 시뮬(MuJoCo XLA) 기반 **2단계 gradient SysID** — Stage1 nominal base 모델 캘리브레이션(전 파라미터: mass/CoM/inertia/damping/frictionloss) → Stage2 payload(torso·hand link의 mass·CoM) 식별. **토크/MoCap 불필요, 관절 encoder + 단일 발 고정 제약으로 trajectory만 사용**.
- **우선순위**: ★★★ (강복님 "2단계 nominal→payload 식별" 요구의 정확한 청사진. payload/관성 변화 대응이 목표라면 1순위. 단 humanoid·heavy-load 특화)

### (1) 요약
실세계 humanoid는 알려지지 않은 payload를 운반하며 이는 total mass/CoM/inertia를 크게 바꿔(systematic·large-scale mismatch) RL 정책 성능을 저하시킨다. HALO는 **미분가능 시뮬 MuJoCo XLA 기반 2단계 gradient SysID**를 RL 제어 파이프라인에 통합한다. Stage1은 무부하 데이터로 nominal 모델의 intrinsic 불일치(마모/캘리브레이션 drift)를 캘리브레이션, Stage2는 부하 데이터로 payload mass 분포를 식별한다. 두 효과를 분리(decouple)하는 것이 안정·정확 식별의 핵심. **부동 베이스(floating-base) humanoid에서 외부 센싱 없이 식별하기 위해 한 발을 기계적으로 고정**(2 vises)하고 forward kinematics로 전신 trajectory를 복원하는 "sensor-minimal" 데이터 수집을 도입. Unitree G1(약 132cm, nominal 35kg, +10kg payload)에서 zero-shot 전이: gait precision +45.45~73.33%, in-place 90° yaw jumping +72.97%, 도전적 motion tracking 100% 성공.

### (2) Motivation
- Heavy payload는 작은 stochastic noise가 아니라 **구조적·대규모 dynamic mismatch**(mass/CoM/inertia 동시 이동). DR로 hedge하면 over-conservative → 로봇의 물리 능력을 다 못 씀.
- 전통 SysID는 floating-base humanoid에 부적합: (a) joint torque sensor나 external MoCap 같은 전용 장비 의존, (b) subsystem 분리·부분 분해 요구 → 복잡 플랫폼에 비현실적.
- Data-driven residual("neural patch")은 학습된 시나리오엔 좋지만 다양한 task 일반화 실패.
- **중심 난제**: nominal model mismatch와 load-induced change의 entanglement. 분리 안 하면 base 오차를 payload에 잘못 귀속.
- Differentiable sim은 fixed-base manipulator/quadcopter/quadruped/humanoid actuator([13]=본 카테고리 #13) 식별엔 성공했으나, **heavy-payload humanoid(거대 inertial shift + 비선형 dynamics)로의 스케일업은 미해결**.

### (3) Method + 수식 + 학습 방법 + 실험 방법

**3단계 파이프라인**: (i) Data Collection → (ii) Gradient-Based Identification(2-stage) → (iii) Heavy-Load Skill Acquisition(RL).

**Data Collection (sensor-minimal).**
- floating-base humanoid는 외부 MoCap 없이 global state 복원이 어려움 → **한 발을 2개 vise로 고정**, forward kinematics로 전신 trajectory 재구성. 시뮬에서도 동일 kinematic 제약 적용.
- fixed-foot 궤적을 휴리스틱 설계 → 그것을 imitate하는 exploration policy를 nominal 모델 + wide DR로 학습([III-C]). 실제 로봇에서 이 정책으로 데이터 수집(관절 위치 + 명령 = PD에 준 target joint position).
- **센서 노이즈로 발 높이 불일치 발생**(Fig.3: 좌우 발 높이가 sensor noise로 어긋남) → 매 timestep root·lower-body joint increment에 대한 **constrained QP**를 풀어 foot-height discrepancy를 최소 보정.

**문제 공식화.** 물리 파라미터 `θ ∈ R^d`(link mass, CoM position, inertia, joint damping). 실제/시뮬 dynamics:
```
s^r_{i+1} = Φ_real(s^r_i, a^r_i; θ^r)      ... (1)
s^s_{i+1} = Φ_sim(s^s_i, a^r_i; θ)         ... (2)   (control은 실제 기록 action으로 고정)
```
수집된 실제 trajectory `τ^r = {s^r_0, a^r_0, ..., s^r_N}`에 대해 시뮬 trajectory `τ^s(θ)`가 일치하도록 θ 추정:
```
min_θ  L_total(θ) = L_track(θ) + R(θ)      ... (3)
```
Φ_sim의 미분가능성을 이용한 Gradient Descent:
```
θ_{k+1} = θ_k - η ∇_θ L_total(θ_k)        ... (4)
```

**Tracking Loss.** 완전 궤적에서 길이 N segment를 B개 uniform sampling, 각 시작점에서 시뮬 상태 리셋 후 N step rollout. 전체 body + payload 영향 큰 upper-body subset `U`에 가중:
```
L_track(θ) = L_track^all(θ) + α^u · L_track^upper(θ)                         ... (5)
L_track^all(θ)   = (1/B) Σ_b Σ_t Σ_{j∈B} || x^s_{b,t,j}(θ) - x^r_{b,t,j} ||²_2   ... (6)
L_track^upper(θ) = (1/B) Σ_b Σ_t Σ_{j∈U} || x^s_{b,t,j}(θ) - x^r_{b,t,j} ||²_2   ... (7)
```
`x ∈ R³`는 body j의 Cartesian 위치(시뮬/실제). B=전 body, U=상체 link(torso, hands).

**Regularization(파라미터가 nominal에서 과도 이탈 방지).**
```
R_total = λ_com·R_com + λ_mass·R_mass + λ_damp·R_damp + λ_fric·R_fric           ... (8)
R_com  = Σ_{i∈link} || p_i - p_i^base ||²_2
R_mass = Σ_{i∈link} (m_i - m_i^base)²
R_damp = Σ_{k∈joint} φ(α_{d,k}; 0.8, 1.2)
R_fric = Σ_{k∈joint} φ(α_{f,k}; 0.8, 1.2)
φ(α; l, u) = max(0, l-α)² + max(0, α-u)²                                        ... (9)
```
→ CoM/mass는 nominal로부터의 deviation을 L2 penalty, damping/friction은 box-constraint(0.8~1.2 배 범위 밖만 penalty).

**2단계 식별 (핵심).** 무부하 trajectory로 nominal에서 직접 부하 파라미터까지 한 번에 식별하면 base mismatch를 payload에 잘못 귀속(biased). 그래서:
- **Stage1 (Base Model Calibration)**: 무부하 trajectory로 **전체 파라미터 set**(전 link의 mass/CoM/inertia/damping/frictionloss) 최적화 → calibrated base model.
- **Stage2 (Payload Parameter Identification)**: calibrated base를 초기값으로, 부하 trajectory로 **payload 관련 파라미터만**(torso·hand link의 mass·CoM) 최적화. inertia tensor 최적화 복잡성은 회피하면서 heavy-load gap을 충분히 감소.

**Skill Acquisition.** 식별 파라미터로 mjlab([34]) 기반 motion imitation RL. reference robot motion을 imitation target으로 사전 수집, MLP 정책을 PPO로 학습(joint state + body pose가 reference와 정렬되도록 tracking reward 최대화). 정확 식별 덕에 **minimal DR**만으로 zero-shot 전이.

**실험.** Unitree G1(132cm, nominal 35kg). 시뮬: 3개 perturbation setting으로 payload 모사. 실기: torso link에 6kg diving counterweight + 양 wrist에 2kg씩(Fig.1a). DR 설정 Table I(nominal vs wide range), HALO learning rate mass=0.03, CoM=0.0002.
- **Q1 (vs CMA-ES, Table II)**: 경부하(setting1,2)는 양쪽 유사 수렴. **극부하(setting3) CMA-ES는 특정 seed에서 비현실적 local optima(예: Torso mass groundtruth +12.0인데 CMA-ES +12.43±9.24, Torso CoM x -0.0616±0.18로 부호 오류)에 빠지나, HALO는 reference 부근으로 robust 수렴(+11.99±0.0, CoM +0.0200±0.0)**.
- **Q2 (motion tracking, Table III)**: 지표 E_g-mpjpe(global, mm)/E_mpjpe(local, mm)/E_vel(root vel, m/s). HALO가 WDR·CM(Calibrated Mass) baseline을 전 지표 압도. Steady-state: HALO 52.47/44.87/0.098 vs WDR 94.91/70.54/0.157. High-agility: HALO 78.83/59.54/0.106 vs WDR 132.89/81.54/0.136. 평가셋: AMASS/LAFAN1/HuB 20 sequence.
- **Q3 (2-stage 효과, Table IV)**: Reference torso 13.82kg. Two-stage 13.83±0.43(거의 정확) vs One-stage 12.93±0.32. L-hand: 2.25 ref, 2-stage 2.13 vs 1-stage 3.22(과대). → coarse-to-fine 분리가 global error를 local payload에서 분리.
- **Q4 (실기, Table V/VI)**: Scenario1 bidirectional walking + in-place 90° jumping. 지표 E_fpos(max forward pos err)/E_epos(end-point residual)/E_ang(angular tracking, deg). HALO walking: 0.12/0.26/11.3° vs CM 0.22/0.45/41.8°, WDR 0.45/0.89/N/A(WDR은 점프 takeoff 실패). **E_fpos -73.33%/-45.45%, E_epos -70.79%/-42.22%, jumping E_ang -72.97%(vs CM)**. Scenario2(swallow balancing, side kicking, roundhouse kicking): HALO 10/10·10/10·10/10 vs CM 5/10·7/10·5/10, WDR 0/10·0/10·0/10.

### (4) 문제점·한계·개선 여지
- **inertia tensor 미식별(Stage2)**: payload는 mass·CoM만 식별, inertia tensor는 회피. fast spinning/임팩트 dominant skill에선 부족할 수 있음(저자가 "복잡성 회피" 위해 의도적 생략 명시).
- **fixed-foot 데이터 수집의 제약**: 한 발 vise 고정이 sensor-minimal엔 영리하나, 실제 운영 환경에서의 데이터(동적 보행 중) 분포와 다를 수 있음(domain shift). 수집 가능한 motion 범위 제한.
- **QP 보정 휴리스틱**: foot-height QP 보정은 센서 노이즈 대응 add-on. 노이즈 모델 부정확 시 식별 데이터 오염 가능.
- **regularization 범위 의존**: damping/friction box [0.8,1.2], CoM/mass deviation penalty가 nominal 품질에 의존. nominal이 나쁘면 penalty가 정답에서 멀어지게 당김.
- **단일 플랫폼·payload 형태**: G1 + rigid counterweight. 유체/관절형/비강체 payload(물탱크, 흔들리는 짐)는 미검증. CLAUDE.md 메모리상 강복님 task는 Go2(quadruped) 중심 → humanoid 결과의 직접 이식엔 morphology gap.
- **offline 식별**: 배포 전 offline 계산 필요(실시간 적응 아님). payload가 trial 중 바뀌면 재식별 필요.

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder, gradient 식별 경로)
- **2단계 식별 청사진을 그대로 채택**: "Stage1 base 캘리브레이션(전 파라미터) → Stage2 payload만(mass·CoM)"의 decouple은 강복님이 payload·tool을 다루는 모든 robot에 직접 적용. base 오차를 payload에 오귀속하는 함정을 회피하는 검증된 레시피.
- **encoder-only 식별 + 토크 cross-check**: HALO는 fixed-foot kinematics로 trajectory만 사용. 강복님은 여기에 **토크/전류 실측을 Stage1 검증축으로 추가** 가능 — 식별된 mass/CoM/damping이 토크 기반 inverse-dynamics 추정과 일치하는지 확인하면 HALO가 못한 ground-truth 검증 확보. 특히 inertia tensor를 strong하게 식별하려면 토크 신호가 필요하므로, 강복님 셋업은 본 논문의 "inertia 회피" 한계를 메울 수 있음.
- **MJX 식별 → IsaacLab RL 분리**: #13과 동일. 식별 θ(link mass/CoM/inertia/damping/friction)를 IsaacLab cfg(`*_env_cfg.py`의 rigid body props, actuator cfg)로 주입. minimal DR만 적용해 zero-shot 지향.
- **Loss 설계 차용**: (a) upper-body(또는 payload 부착부) 가중 tracking loss(식 5의 α^u), (b) nominal-anchored regularization(식 8~9)으로 비현실적 식별 방지, (c) box-constraint φ로 damping/friction을 물리 타당 범위에 가둠. 이 셋은 강복님 식별 안정화에 즉시 유용.
- **Go2 적용 시 주의**: humanoid fixed-foot 제약은 quadruped엔 부적합 → quadruped는 #13식 모터-bench 식별 또는 고정 지그 + 가벼운 부하 식별로 변형 필요. payload가 Go2 manipulator(예: Z1 arm) 부착이면 #15(UAN)와 결합 검토.

### sim 구현 난이도: **상**
- 필요 인프라: MuJoCo XLA(JAX) + 미분가능 multi-body rollout, segment-wise gradient, 2-stage optimization 스케줄러, fixed-foot 데이터 수집 지그(물리 하드웨어), forward-kinematics 전신 복원 + foot-height QP solver, mjlab/IsaacLab RL 파이프라인 연동.
- 상으로 평가하는 이유: (a) floating-base 전신 미분가능 식별은 단일 actuator(#13)보다 훨씬 복잡(gradient 안정성·메모리), (b) 하드웨어 데이터 수집(vise 고정 + 안전 motion) 셋업 비용, (c) 2-stage 분리·regularization 튜닝, (d) humanoid → Go2 적용 시 재설계. 코드 미공개(project page만)로 재현 부담 큼.

---

## #15. Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation (Fey, Margolis, Peticco, Agrawal — MIT CSAIL / Improbable AI, arXiv 2025)

- **확정 링크**: https://arxiv.org/abs/2502.10894 (HTML 전문: https://arxiv.org/html/2502.10894v1). Project: https://uan.csail.mit.edu . Thesis(상세): dspace.mit.edu/.../fey-...-2025-thesis.pdf .
- **접근 분류**: SysID(파라미터 추정)가 아니라 **learned residual actuator model(UAN, Unsupervised Actuator Net)** — actuator dynamics를 corrective torque로 보정. **토크 센서 불필요(unsupervised)**. + task-reward RL의 pretrain/finetune.
- **우선순위**: ★★ (카테고리 C의 "diff/SysID" 정의에선 약간 벗어남 = parameter 식별이 아닌 residual NN. 하지만 "토크 없이 actuator gap 메우기" + harmonic drive 비선형 대응 + task-reward 설계는 강복님에게 보완적 가치. #13/#14 식별이 안 닿는 복잡 actuator에 유용)

### (1) 요약
Athletic loco-manipulation(공 멀리 던지기, 빠르게 들기 등)은 reference tracking reward를 넘어 **task reward**(목표 지향, 동적·파워풀)를 요구한다. 그러나 task reward만 쓰면 (a) reward hacking(시뮬 결함 악용), (b) exploration 방향성 부족 문제가 생긴다. 2단계 파이프라인 제안: **① UAN(Unsupervised Actuator Net)** — 실세계 데이터로 복잡 actuation(특히 harmonic drive 비선형)의 sim-to-real gap을 **토크 센싱 없이** 메워 reward hacking을 억제(학습 거동이 robust·전이 가능하도록); **② pretrain-finetune** — reference trajectory를 초기 hint로 exploration을 유도. Unitree B2 quadruped + 개조 Z1 Pro arm(19 DOF)에서 lift/throw/drag를 시뮬→실제 고충실도 전이.

### (2) Motivation
- Tracking reward는 단순히 reference를 따라가게 할 뿐, "공을 최대한 멀리 던져라" 같은 **본질적 athletic agility/power**를 끌어내지 못함.
- Task reward는 동적 거동을 유도하지만: (a) **reward hacking** — 정책이 시뮬의 부정확한 actuator dynamics를 악용해 실기 전이 실패, (b) **방향성 없는 exploration** — 고차원 dynamic task에서 학습 비효율.
- 특히 Z1 Pro arm의 **harmonic reduction drive**는 비선형 friction·hysteresis·lag를 가져 standard actuator 모델(또는 motor current 추정 기반)로 포착 불가 → 시뮬-실제 actuator gap이 reward hacking의 주원인.

### (3) Method + 학습 방법 + 실험 방법

**로봇 플랫폼.** Unitree B2 quadruped + 개조 Unitree Z1 Pro arm. **19 actuated joint**(다리 3×4 + arm 6 + gripper 1). B2 ~65cm/60kg, arm ~74cm/6.8kg.

**UAN (Unsupervised Actuator Net).**
- 구조: **2-layer MLP [128,128], ELU**. 5ms 시뮬 timestep마다 동작.
- 입력: 각 actuator의 **과거 20스텝(=100ms) position·velocity error history `e`**. 출력: corrective torque **δτ = π_UAN(e)**.
- **Unsupervised 학습(토크 센서 불필요)**: 실제 actuator의 결과(상태 전이)와, 보정 토크를 더한 시뮬의 결과가 일치하도록 최소화:
```
min_{π_UAN}  || f_real(s, τ) - f_sim(s, τ + π_UAN(e)) ||
```
즉 토크 자체를 supervised target으로 쓰지 않고(=토크 측정 불필요), **실제 상태 trajectory를 재현하는 corrective torque를 학습**. 학습은 Isaac Sim 4096 병렬 환경에서 PPO, reward = `r^sim-to-real_t + r^smoothness_t`(전자는 실제-시뮬 관절 위치 차 최소화).
- 효과: harmonic drive의 nonlinear friction/hysteresis/lag, actuator lag를 포착 — motor current 추정 기반 방법이 못 잡는 영역.

**2단계 학습 파이프라인.**
- **Stage1 Pretrain (WBC)**: whole-body controller를 random base velocity + end-effector pose command 추종으로 사전학습. 정책 3-layer MLP [512,512,512], observation history H=10(200ms).
- **Stage2 Finetune**: pretrained 정책으로 task-specific 학습 초기화, **낮은 lr(1×10⁻⁵)**. **strict reference tracking을 강제하지 않음** — 초기엔 reference를 따라(exploration 보조), 후기엔 task 성능 최대화를 위해 reference에서 이탈하도록 학습(reference = "초기 hint").

**Task reward(3종).**
- **Ball Throwing**: 100g 공을 최대한 멀리. 실기 약 20m.
- **Dumbbell Snatch(들기)**: randomized mass(0~10kg) 학습 → 시뮬 8kg까지 일관 lift. 실기 5lb·10lb 덤벨 성공.
- **Sled Pull(끌기)**: 저항력 cart. 시뮬 150kg까지. 실기 113N friction을 10m 끌기, 230N 저항은 0.5m만.

**실기 평가.** UAN 학습 정책이 domain randomization 등 baseline 대비 **실기에서 가장 먼 던지기 + 가장 작은 sim-to-real gap** 달성.

### (4) 문제점·한계·개선 여지
- **reference trajectory 의존(저자 명시)**: finetune이 task reference를 요구 → 모든 morphology/task에 reference가 있지 않음.
- **per-task engineering**: task마다 reward function·object 시뮬을 따로 설계해야 함(자동화 안 됨).
- **arm actuator 한정**: UAN은 현재 arm(harmonic drive) 보정 중심. 다른 subsystem·구조적 무결성(structural integrity) 모델링은 미래 과제.
- **SysID 아님(파라미터 비해석성)**: UAN은 black-box residual → 식별된 물리 파라미터를 얻지 못함. 다른 task/시뮬로의 transfer나 해석 가능성은 #13/#14보다 약함(과적합 위험).
- **heavy resistance 한계**: sled pull 230N에서 0.5m로 급락 — 고저항 영역 transfer 약함.
- **실기 정량 지표 빈약**: throw "약 20m" 등 정성·근사치 위주, tracking error 같은 엄밀 지표는 #14 대비 적음.

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder, gradient 식별 경로)
- **#13/#14의 보완재**: 강복님 메인 경로는 gradient SysID(파라미터 식별)지만, **harmonic drive·복잡 transmission처럼 파라미터화가 어려운 actuator**(예: Go2에 붙일 arm/gripper)에는 UAN식 residual이 더 적합. "식별 가능한 건 #13/#14로, 식별 안 되는 잔차는 UAN으로" 하이브리드 가능.
- **토크 없이 actuator gap 메우기**: UAN의 unsupervised loss(`f_real` vs `f_sim(τ+δτ)`)는 강복님 encoder 데이터만으로 동작. 단, 강복님은 토크/전류 실측 보유 → **UAN δτ 출력을 실측 토크와 비교**해 residual이 물리적으로 타당한지 sanity-check(본 논문이 못한 검증).
- **IsaacLab 친화적**: UAN 학습이 이미 **Isaac Sim 4096 env PPO**로 수행됨 → 강복님 IsaacLab 워크플로에 가장 자연스럽게 얹힘(MJX 의존 없음). actuator residual을 IsaacLab actuator 모델에 plug-in하는 형태.
- **task-reward 설계 교훈**: pretrain(reference tracking) → finetune(낮은 lr로 reference 이탈 허용) 레시피는 강복님 parkour/athletic task에서 reward hacking을 줄이며 dynamic 거동을 끌어내는 데 직접 응용 가능. CLAUDE.md 메모리의 AMP/imitation 트랙과도 호환(reference를 hint로만 쓰는 철학).
- **주의**: SysID가 목표라면 1급 시민이 아님(파라미터를 안 줌). "정책 전이용 actuator 보정"으로 한정 사용.

### sim 구현 난이도: **중**
- 필요 인프라: Isaac Sim/IsaacLab(보유) + PPO(rsl_rl 보유), actuator residual NN을 시뮬 actuator loop에 주입하는 hook, 실기 actuator excitation 데이터 수집(encoder, 토크 선택). MJX·미분가능 시뮬 불필요(장점).
- 중으로 평가하는 이유: (a) UAN을 IsaacLab actuator 모델에 통합하는 코드(δτ 주입 지점)가 필요, (b) unsupervised loss용 실기 데이터 수집·정렬, (c) per-task reward·object 시뮬 엔지니어링. 코드 미공개지만 IsaacLab 네이티브라 #14보다 재현 쉬움.

---

## 종합 비교 (강복님 의사결정용)

| 항목 | #13 Kovalev | #14 HALO | #15 UAN |
|---|---|---|---|
| 본질 | gradient SysID (param) | 2단계 gradient SysID (param) | residual actuator NN |
| 식별 대상 | armature/frictionloss/damping (모터 1개) | mass/CoM/inertia/damping/friction → payload mass·CoM | (식별 안 함) corrective torque δτ |
| 토크 측정 | 불필요 | 불필요 | 불필요(unsupervised) |
| 시뮬 | MJX(미분) | MuJoCo XLA(미분) | Isaac Sim(PPO) |
| 플랫폼 | Mini π biped | Unitree G1 humanoid +10kg | B2+Z1 quadruped-manip 19DOF |
| 대표 결과 | rotation -75%, travel +46% | tracking -45~73%, jump -72.97%, skill 10/10 | throw ~20m, lift 10lb |
| 난이도 | 하~중 | 상 | 중 |
| 강복님 우선순위 | ★★★ 먼저 재현 | ★★★ payload면 1순위 | ★★ 보완재 |

**권장 경로**: #13으로 단일 actuator gradient 식별 파이프라인을 IsaacLab/MJX에 안착(토크 실측으로 cross-validate) → #14의 2단계 decouple 구조로 payload·전신 확장 → 파라미터화 어려운 actuator는 #15 UAN residual로 보강.

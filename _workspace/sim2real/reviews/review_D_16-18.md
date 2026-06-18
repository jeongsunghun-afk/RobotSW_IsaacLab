# 카테고리 D: Real2Sim 심화 리뷰 (#16-18)

> 리뷰어: Sim2Real 논문 심화 리뷰어 / 작성일: 2026-06-18
> 원칙: 3편 모두 arxiv 원문(ar5iv HTML 본문)을 실제로 읽고 작성. 스니펫 추론 배제.
> 공통 적용 맥락: IsaacLab / MJX 환경, 강복님 real2sim 닫힌 루프 C4 (divergence 감지 → 역추적 → 재보정)

---

## #16. Closing the Sim-to-Real Loop: Adapting Simulation Randomization with Real World Experience (Chebotar et al., NVIDIA, ICRA 2019)

- **확정 링크**: https://arxiv.org/abs/1810.05687 (ar5iv 본문 확인)
- **접근 분류**: Real2Sim (실기 rollout로 시뮬 randomization 분포 적응 → SimOpt)
- **우선순위**: 카테고리 D의 기준선(seminal). real2sim 닫힌 루프의 원형.

### (1) 요약
SimOpt은 도메인 랜덤화(DR) 분포를 손으로 튜닝하는 대신, **소수의 실기 rollout과 정책 학습을 번갈아(interleave)** 수행하며 시뮬 파라미터 분포 `p_phi`를 자동 적응시킨다. 핵심 루프는 (a) 현재 분포 `p_phi_i`에서 DR로 정책 `pi_theta` 학습 → (b) 학습된 정책을 실기에서 굴려 실기 관측 궤적 수집 → (c) 시뮬 궤적과 실기 궤적의 불일치 `D`를 최소화하도록 분포를 `p_phi_{i+1}`로 갱신, 이때 KL 제약으로 한 스텝 변화량을 제한. ABB Yumi swing-peg-in-hole, Franka drawer-opening 두 실기 과제에서 2~3 SimOpt 반복만으로 zero-shot 전이 성공률을 90~100%까지 끌어올렸다.

### (2) Motivation
- DR은 sim-to-real 갭을 메우는 강력한 수단이지만 **랜덤화 범위·평균을 사람이 수동 튜닝**해야 하고, 너무 넓으면 정책이 보수적·저성능, 너무 좁으면 전이 실패한다.
- 실기 데이터로 시뮬 파라미터를 한 번 SysID하는 고전적 방법은 모델 불완전성(미관측 동역학) 때문에 정확한 점추정이 어렵다. SimOpt은 **점추정이 아니라 "실기를 잘 설명하는 파라미터 분포"** 를 찾는 것으로 문제를 재정의한다.
- 즉 "실기를 가장 잘 재현하는 DR 분포"를 학습 → 그 분포로 학습한 정책이 실기에 강건하게 전이.

### (3) Method + 수식/파이프라인 + 학습/실험 방법

**최적화 목표 (Eq. 3):**
```
min_{phi_{i+1}}  E_{xi ~ p_{phi_{i+1}}} [ E_{pi_theta, p_{phi_i}} [ D(tau^ob_xi, tau^ob_real) ] ]
  s.t.  D_KL( p_{phi_{i+1}} || p_{phi_i} )  <=  epsilon
```
- `xi`: 시뮬 파라미터 벡터(질량/마찰/댐핑/compliance/action scale 등), `p_phi = N(mu, Sigma)` 가우시안.
- 실기 궤적 `tau^ob_real`은 직전 분포 `p_phi_i`로 학습한 정책으로 수집. KL 제약 `epsilon`이 분포의 한 스텝 이동을 trust-region처럼 제한해 정책-분포 mismatch 폭주를 방지.

**불일치 함수 D (Eq. 4) — 시간정렬 처리:**
```
D(tau_xi, tau_real) = w_L1 * sum_i | W (o_{i,xi} - o_{i,real}) |
                    + w_L2 * sum_i || W (o_{i,xi} - o_{i,real}) ||_2^2
```
- 관측 feature별 가중 L1+L2 결합. `W`는 차원별 importance weight.
- **시간정렬**: 명시적 동기화는 불필요하되, 거리 계산에 **가우시안 필터를 적용**해 궤적 간 미세 misalignment를 흡수한다(완전한 distribution-level은 아니고 "soft 시간정렬"). 이 점이 #17(완전 시간정렬 불요)과의 핵심 차이.

**분포 갱신 알고리즘 — REPS:**
- **Relative Entropy Policy Search(REPS)** 를 사용한 gradient-free black-box 최적화. 샘플 `xi ~ p_phi`와 각 비용 `c(xi)=D(...)`로부터 가우시안 `(mu, Sigma)`를 갱신하되 위 KL 제약을 만족. 저비용 샘플에 더 큰 가중을 주는 importance-weighting 형태로 분포를 이동.

**정책 학습:**
- 병렬 **PPO**(멀티-GPU). 첫 SimOpt 반복은 200 PPO iteration, 이후 직전 가중치를 warm-start해 **10 iteration으로 축소**. swing-peg: 100 RL iter ≈ 7분, drawer: 200 RL iter ≈ 22분.

**실험 인프라/데이터 양:**
- 시뮬레이터: **NVIDIA Flex**(GPU 고충실 물리). **64 GPU × GPU당 150 agent** 병렬.
- 실기: swing-peg = **ABB Yumi 7-DoF**, drawer = **Franka Panda 7-DoF**.
- **SimOpt 반복당 실기 rollout 3회**. 총 2~3 SimOpt 반복으로 수렴 (즉 실기 데이터 = 수~십 회 rollout 수준, 극히 적음).
- 관측: swing-peg = 7-DoF 관절각 + peg 3D 위치 / drawer = 7D 관절각 + 손잡이 3D 위치.

**랜덤화 파라미터(Tables I, II):** joint compliance/damping(각 7D), gripper compliance/damping, action scaling(7D), drawer compliance/damping, handle friction; (peg) rope torsion/bending, rope segment width/length/friction/density, peg scale/friction/mass, peg-box scale/friction. 적응 예: drawer handle friction 0.001 → 2.13, rope segment width 0.004 → 0.007.

### (4) 실기 평가 메트릭 + 정량 결과
- **Swing-peg-in-hole**: 20 trial 평가. **2 SimOpt 반복 후 90% 성공**. (반복0: hole 못 맞춤 → 반복1: 도달하나 삽입 각도 부족 → 반복2: 90%.)
- **Drawer-opening**: 20 trial 평가. 갱신 후 **항상(100%) 서랍 개방**. (이전: gripper 방향 어긋남, finger slip.)
- 메트릭 = **task success rate**(이진), 그리고 정성적으로 gripper 정렬·삽입 각도.

### (4-한계) 문제점·한계·개선 여지
- 평가가 **이진 success rate + 20 trial**로 표본이 작고 통계적 신뢰구간 없음. tracking error 같은 연속 지표 부재.
- 실기에서 **peg/handle 3D 위치 관측이 필요**(외부 비전/모캡). proprioception-only가 아니라 외부 센싱 의존 → 비용·설정 부담. (#17이 이 점을 정면 비판·해결.)
- **D가 soft 시간정렬에 의존**: 불안정/혼돈적 동역학에서 작은 차이가 발산하면 시간정렬 기반 거리가 무의미해질 위험(#17의 주장과 직결). 보행 같은 불안정 시스템엔 적용이 까다로움.
- **정적 파라미터 분포만** 적응(질량·마찰·compliance 등). actuator lag·비선형 잔차 같은 구조적 모델 오차는 분포 폭으로만 흡수 → 표현력 한계.
- REPS는 차원이 커지면(여기 수십 D) 수렴이 느려질 수 있고, KL `epsilon`·norm weight 등 하이퍼파라미터 민감.

### (5) 우리 적용 (강복님 real2sim 닫힌 루프 C4)
- **SimOpt이 C4의 원형 그 자체**: divergence(D) 측정 → 역추적(어떤 `xi`가 실기를 설명하나) → 재보정(`p_phi` 갱신) 루프가 1:1 대응.
- IsaacLab 적용: 우리는 이미 4096 env 병렬 PPO + DR(base mass, PD gain, friction)이 갖춰져 있어 SimOpt의 "분포로 학습"부분은 그대로 재사용 가능. 추가로 필요한 건 **(a) 실기 rollout 수집 → (b) D(Eq.4) 계산 → (c) REPS로 DR 분포 mu/Sigma 갱신** 모듈.
- 단, **#17의 교훈을 결합**할 것: 우리 Go2/R_Skeleton는 보행(불안정)이라 SimOpt의 soft-시간정렬 D보다 **#17의 proprioceptive distribution matching(시간정렬 불요)** 가 더 안전. → C4의 divergence 메트릭은 #17식 Wasserstein, 분포 갱신 optimizer는 SimOpt식 REPS 또는 #17식 CMA-ES 중 택일.
- KL trust-region(`epsilon`)은 우리 루프에서도 "한 번에 DR 분포를 과하게 흔들지 않는" 안전장치로 도입 가치 높음.

### sim 구현 난이도: **중**
- 필요 인프라: IsaacLab 대규모 병렬 PPO(이미 보유) + 실기 rollout 로깅 파이프라인 + REPS 구현(외부 라이브러리/직접 구현) + D 계산 모듈 + DR 분포를 런타임에 외부에서 주입·갱신하는 hook(env_cfg의 randomization range를 iteration마다 갱신). NVIDIA Flex 특수성은 IsaacLab/PhysX로 대체 가능. 외부 비전(물체 위치)은 보행 과제엔 불필요하므로 우리 케이스에선 오히려 부담 감소.

---

## #17. Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching (arXiv 2026)

- **확정 링크**: https://arxiv.org/abs/2604.11090 (정식 제목: "Simulator Adaptation for Sim-to-Real Learning of Legged Locomotion via Proprioceptive Distribution Matching", 2026-04-13)
- **접근 분류**: Real2Sim (실기 proprioception 분포로 시뮬레이터 적응)
- **우선순위**: **5순위** (단, 우리 Go2/IsaacLab/PPO 스택과 가장 직접적으로 일치 → 실전 가치는 최상위)

### (1) 요약
보행 로봇 sim-to-real을 위해 실기와 시뮬 rollout을 **"관절 관측·행동의 분포(distribution)"로 비교**해 시뮬레이터를 적응시킨다. 핵심은 궤적을 시간순서 없는 표본 집합으로 보고 **시간정렬 불요·외부 센싱 불요(proprioception only)** 의 Wasserstein 거리로 매칭하는 것. Unitree Go2 + IsaacLab + PPO 스택에서 **5분 미만의 실기 데이터**로 spring-joint·이족(hind-leg) 보행 같은 어려운 케이스의 drift를 90%+ 감소시켰다.

### (2) Motivation
- 기존 DR은 **정적 모델 파라미터만 랜덤화** → 표현 가능한 동역학이 제한적.
- 기존 state-matching(궤적 일치)은 **모캡·privileged 센싱·정밀 초기조건**을 요구 → 비용·설정 부담.
- 결정적으로, **시간정렬 궤적 비교는 불안정 시스템에서 붕괴**: 작은 차이가 빠르게 큰 편차로 누적(보행이 정확히 그 케이스). → "시간정렬을 버리고 분포로 비교"가 동기.

### (3) Method + 수식/파이프라인 + 학습/실험 방법

**분포 매칭 메트릭 — Wasserstein(1D marginal 근사), 시간정렬 불요:**
```
W(P, Q) = inf_eta  sum_{i=1..n} || x_i - y_{eta(i)} ||
```
- 관절 위치/속도/행동 각 차원별 **1D Wasserstein**을 계산(표본 정렬 후 `O(n log n)`), 차원 평균. 다차원 OT를 푸는 비용 회피.
- **궤적을 순서 없는 표본으로 취급** → 시간정렬·동기화 완전 불요. 시간순서를 제거해도 **gait 패턴·접촉 통계·actuator lag·damping 효과** 같은 장기 행동 특성은 분포에 보존된다는 것이 핵심 통찰.

**시뮬레이터 수정 3종(무엇을 적응시키나):**
1. **FricArm**: 관절 정적 마찰 + armature(정적 파라미터).
2. **ActionDelta**: 신경망 `Delta_a(s_t | theta_hat)` 이 관절 목표 위치에 잔차 추가.
3. **ResidAct**: 신경망 `Delta_t(s_t | theta_hat)` 이 1kHz 제어에서 토크 잔차 추가.
- 신경망: 2-layer MLP, hidden [8,4], per-joint. → 단순 정적 파라미터를 넘어 **상태 의존 동역학 잔차**까지 적응(SimOpt 대비 표현력 확장).

**최적화기 — CMA-ES (gradient-free):**
- FricArm: 200 iter, 초기 공분산 0.1(friction)/0.05(armature).
- ActionDelta/ResidAct: 10,000 iter, 공분산 0.25, 파라미터 [-1,1] bound.
- 각 평가 = 시뮬에서 **64 궤적 × 4초**.

**정책 학습:** PPO, 20,000 iter, 4096 env, 1.966B sample. 정책망 3-layer MLP [128,128,128]. 출력 = 12 관절 목표(50Hz) → PD 제어(1kHz). 입력 = 관절 proprioception + base 각속도(IMU) + gravity vector + 속도 명령 + last action. 학습 시 DR = base mass, PD gain, friction.

**실기 데이터 양/방법:** **64 궤적 × 4초 ≈ 총 5분 미만**. 무작위 속도 명령, **모캡 없음**, proprioception만 로깅(base pose 미사용).

**테스트 시나리오:** (1) 모델 파라미터 shift(known friction/armature delta), (2) Spring Joint(calf 가상 스프링), (3) 이족 보행(hind-leg only).

### (4) 실기 평가 메트릭 + 정량 결과
메트릭 = **명령 속도 추종 시 위치 drift(m)**.
- **Spring Joint**: 원정책 +x(1m/s) drift −2.0m, +y는 완전 실패, −y −0.898m → ResidAct+Wasserstein 미세조정 후 +x −0.177m(91%↓), +y 0.21m(성공으로 전환), −y −0.492m(45%↓).
- **이족(bipedal) 보행**: 원정책 forward −0.4m / backward 0.6m / lateral(−y 0.5m/s) −0.5m → 조정 후 forward −0.025m(94%↓), lateral 0.0m(완전 보정), backward 0.05m(92%↓).
- **Sim-to-sim 파라미터 복원**(노이즈 σ=5.0 N·m 토크 + 0.02~0.1s 시간 shift): Wasserstein 18.75% 평균 오차 vs MatchO(∞) 86.28% vs privileged MatchS(∞) 37.81% → **시간정렬 기반 baseline 대비 압도적**.

### (4-한계) 문제점·한계·개선 여지
- **치명적 실패 모드(Digit 휴머노이드)**: 초기 sim-to-real 전이가 충분히 성공해 **관련 state space에 유효 실기 궤적이 존재해야만** 작동. 베이스라인 정책이 아예 붕괴(backward 보행 실패)하면 실기 데이터에 매칭할 유효 표본이 없어, optimizer가 **시뮬에서 임의 실패를 유발하는 무의미한 수정**을 학습(분포만 맞추고 물리적 의미 없음).
- 즉 **"실패 영역·미탐사 state space를 분포에 편입하는 원리적 메커니즘 부재"** (저자 명시).
- CMA-ES 10,000 iter는 비용이 큼(특히 신경망 잔차). 1D marginal Wasserstein은 차원 간 상관(joint coupling)을 놓침.
- 분포 매칭은 **divergence의 "양"은 잘 잡지만 "원인 파라미터 역추적"은 약함**(black-box) → C4의 "역추적" 단계엔 SimOpt식 해석가능 파라미터화 보완 필요.

### (5) 우리 적용 (강복님 real2sim 닫힌 루프 C4)
- **우리 스택과 1:1 일치**(Go2 + IsaacLab + PPO + DR[base mass/PD gain/friction]). C4의 **divergence 메트릭으로 proprioceptive 1D Wasserstein이 최적 후보** — 우리 Go2 보행은 불안정계라 시간정렬 D(#16)보다 안전하고, **모캡 불요·5분 데이터**라 실기 운용 비용 최소.
- C4 루프 매핑: divergence = Wasserstein(실기 proprio 분포 vs 시뮬), 역추적 = 어떤 시뮬 수정(FricArm/ResidAct)이 분포를 좁히나, 재보정 = CMA-ES로 그 수정 파라미터 최적화.
- **메모리 경고(3-leg)와 직접 연결**: 우리 parkour 3-leg 비대칭은 "정책이 학습한 평형"이므로, ResidAct 같은 상태의존 잔차로 실기 동역학 갭을 메우면 3-leg 평형이 실기에서 깨지는지 진단 가능.
- **한계 회피책**: 우리도 "베이스라인이 실기에서 최소한 안 넘어지는" 정책에만 적용해야 함(Digit 실패 교훈). 붕괴 케이스는 SimOpt(#16)·#18로 우선 안정화 후 적용.

### sim 구현 난이도: **하~중**
- 필요 인프라: **이미 보유한 것 대부분 재사용**(IsaacLab Go2 USD, 4096 env PPO, DR). 신규 = (a) 1D Wasserstein 분포 거리 모듈(scipy/직접, 경량), (b) CMA-ES 래퍼(pycma), (c) FricArm/ResidAct 시뮬 수정 hook(IsaacLab actuator/joint에 잔차 주입 — ResidAct은 1kHz 토크 주입이라 약간 손이 감), (d) 실기 proprio 로깅. **MJX로도 직행 가능**(미분 불요, black-box라 어떤 시뮬에도 적용). 우리 카테고리 D 중 진입장벽 최저.

---

## #18. Real-time Model Predictive Control and System Identification Using Differentiable Physics Simulation (Chen/Werling/Wu/Liu, Stanford, RA-L 2023)

> 번호 교정: 의뢰서의 "Jin et al."는 오기. 실제 저자는 **Sirui Chen, Keenon Werling, Albert Wu, C. Karen Liu** (arXiv:2202.09834, RA-L 2023).

- **확정 링크**: https://arxiv.org/abs/2202.09834 (ar5iv 본문 확인)
- **접근 분류**: Real2Sim (online SysID로 시뮬 파라미터를 실기에 맞춤) + 실시간 제어 결합
- **우선순위**: 카테고리 D 내 "미분물리 기반 online SysID+MPC" 트랙.

### (1) 요약
**미분 가능 물리(NimblePhysics)** 를 써서 **online SysID와 MPC(iLQR)를 실시간으로 동시 수행**한다. 매 제어 주기에 들어오는 관측으로 (a) 시뮬 파라미터(질량/관성/COM/댐핑)를 미분물리 gradient로 보정하고 (b) 같은 미분 시뮬로 iLQR 최적제어를 푼다. 핵심 기여 3가지 — (i) **추정 confidence 평가**(동역학 방정식의 수치해석으로 파라미터 관측가능성 판정 → 신뢰할 때만 갱신), (ii) **적응적 최적화 윈도우 스케줄링**(제어가 소비보다 빠르게 계산되도록 미래 시점 예약), (iii) 실기 7-DOF 암에서 페이로드 변화에 즉시 적응. 확률적 최적화 대비 **80배 빠름**.

### (2) Motivation
- 배포 후에도 로봇 모델·제어기를 **지속 개선**해야 하나, 기존 SysID는 오프라인·일회성이라 동적 변화(페이로드 추가/제거 등)에 대응 못 함.
- 미분물리는 gradient로 SysID·제어를 효율화하지만 **실시간성(제어 주기 내 수렴)** 과 **잡음 하 파라미터 신뢰성**이 미해결 → 이 둘을 정면으로 해결.

### (3) Method + 수식/파이프라인 + 학습/실험 방법

**미분 시뮬레이터:** NimblePhysics. 접촉은 LCP로 처리하나 **"모호한 contact gradient + 거친 loss landscape"**(Sec. V)를 명시적으로 인정 — contact-rich 과제에서 iLQR 수렴이 들쭉날쭉.

**online SysID loss(윈도우 H 상태 일치):**
```
mu* = argmin_mu  || sim(q0, q0_dot, u_{0:H}; mu, H) - q_{0:H} ||
```
- 과거 윈도우 H의 관측 상태열과 시뮬 예측 상태열의 위치 오차 최소화. gradient는 미분물리로 역전파.

**MPC 목표(iLQR):**
```
f_task = (1/2) sum_k (x_k - x_bar)^T Q_r (x_k - x_bar)
       + (1/2) (x_T - x_bar)^T Q_f (x_T - x_bar)
       + (1/2) sum_k u_k^T R u_k
```
- 추종(running Q_r) + 종단(Q_f) + 제어정규화(R)의 2차 비용.

**confidence 평가(수치해석 기반 관측가능성):** Lagrangian `M(q;mu) q_ddot + c(q,q_dot;mu) + g(q;mu) = f` 에서 파라미터별 **여기(excitation)/관측가능성**을 컬럼 norm으로 산출:
- mass conf: 궤적상 `||A[:,k] + B[:,k]||` 합(A=관성 가속 결합, B=중력 결합).
- inertia conf: 각 Jacobian 가중 `||J_w^T Diag(...) ||` 의 rank 분석.
- COM conf(Eq.7): `||S_k[:,i]|| + ||G_k[:,i]||` (S=가속 효과, G=중력 효과).
- 갱신 규칙: `mu_bar = (w_bar*mu_bar + w*mu*)/(w_bar + w*)` — 새 confidence `w*`가 `eps1` 초과 **그리고** 기존과 차이가 `eps2` 미만일 때만 추정 반영(잘못된 점프 방지).

**적응적 윈도우 스케줄링(실시간성):**
```
t_hat = t + t_e * N_iter
```
- t=현재시각, t_e=iLQR iteration당 측정 소요, N_iter=직전 solve의 iteration 수. → 계획 buffer가 소비되기 전 보충 보장. cartpole에서 고정 타이밍 8.20s vs 적응 6.31s.

**실험(과제/로봇/제어율/데이터):**
| 과제 | 시스템 | 제어율 | 실기 여부 |
|------|--------|--------|-----------|
| Cartpole swing-up | 2-link, 질량 미지 | ~48Hz (iLQR 20.8ms) | 시뮬 |
| Inverted double pendulum | 3-link, 댐핑 미지 | 동일 | 시뮬 |
| **Robot arm (Rokae Xmate3P 7-DOF)** | 페이로드 변화 | ~20Hz | **실기 O** |
| Elastic rod | 4-link 스프링 체인 | 동일 | 시뮬 |
| Hopper(SLIP, 계단) | contact-rich | 시뮬 | 시뮬 |

- **실기 실험만 hardware**: 수직 sinusoidal 추종 중 페이로드 추가/제거. online SysID = **tracking error 2cm** vs offline SysID = **7cm**.

### (4) 실기 평가 메트릭 + 정량 결과
- 실기 메트릭 = **tracking error(cm)**: online 2cm vs offline 7cm (≈3.5배 개선).
- 시뮬 메트릭 = **time-to-goal(s, mean±std, Table II)**: ours가 모든 과제 최소 — cartpole 6.31±0.42(naive 7.57±0.87, UP-OSI/DuST 실패), InvDP 3.53±0.22, arm 2.10±0.20, rod 4.60±0.07. UP-OSI/DuST 같은 baseline은 다수 과제에서 실패.
- 속도: **확률적 최적화 제어기 대비 80배 빠름**.

### (4-한계) 문제점·한계·개선 여지
- **실기 검증이 단일 과제(7-DOF 암 추종)에 국한**, 나머지는 시뮬. 보행/접촉 실기 검증 없음.
- **Contact-rich에 취약**: contact gradient 모호 + 거친 loss landscape로 iLQR 수렴 편차 큼(Hopper 등). → 보행처럼 접촉 빈번한 과제엔 그대로 못 씀.
- 미분물리 시뮬(Nimble)에 **모델 구조 의존**: IsaacLab/MJX와 다른 엔진. 우리 스택에 직접 이식 불가, MJX의 미분 경로로 재구현 필요.
- horizon T에 민감(H, eps1, eps2엔 강건). 파라미터화가 **저차원 물리량(질량/관성/COM/댐핑)에 국한** → 신경망 잔차 같은 고표현 갭은 못 메움.

### (5) 우리 적용 (강복님 real2sim 닫힌 루프 C4)
- **C4의 "역추적" 단계에 가장 강한 도구**: divergence를 미분물리 gradient로 **어느 파라미터(질량/COM/댐핑)가 갭을 만드는지 직접 역추적** + confidence 평가로 **관측 불가능한 파라미터의 잘못된 보정을 차단**. #16/#17의 black-box(REPS/CMA-ES)가 못 주는 해석가능·고효율 보정.
- **MJX 결합 시나리오**: MJX는 미분 가능 → Nimble의 online SysID loss/iLQR를 MJX로 재현하면, IsaacLab(대규모 RL 학습) + MJX(미분 SysID/MPC)의 **이중 스택 C4** 구성 가능. 단 우리 robot은 보행(접촉)이라 #18의 contact 한계가 직격 → **floating-base 보행보다 fixed-base 조작/단순 동역학에 먼저 적용** 권장.
- **confidence 평가는 우리 루프에 직수입 가치 높음**: 실기 데이터가 특정 파라미터를 여기(excite)하지 못하면 보정하지 말 것 — #17의 "미탐사 state space 무의미 보정" 실패를 원리적으로 방지하는 게이트로 활용.

### sim 구현 난이도: **상**
- 필요 인프라: **미분 가능 물리 엔진**(Nimble 그대로는 우리 스택 밖 → **MJX로 재구현 필수**), iLQR/미분 trajectory optimizer, online SysID loss + gradient 파이프라인, confidence 평가(동역학 방정식 수치해석 구현), 실시간 스케줄러, 실기 저지연 제어 루프(20~48Hz). IsaacLab은 미분 미지원이라 RL은 IsaacLab·SysID/MPC는 MJX로 분리해야 하고, 접촉 gradient 처리까지 더해 카테고리 D 중 진입장벽 최고. 보행 적용 전 contact-gradient 안정화가 선결.

---

## 종합 비교표 (C4 루프 관점)

| | #16 SimOpt | #17 Proprio-Dist-Match | #18 Diff-Physics MPC+SysID |
|---|---|---|---|
| divergence 메트릭 | 가중 L1+L2 + 가우시안 soft-정렬 | 1D Wasserstein (시간정렬 불요) | 미분물리 상태 오차 (윈도우 H) |
| 역추적(원인) | REPS black-box | CMA-ES black-box | **미분 gradient (해석가능)** |
| 재보정 대상 | DR 분포 (정적) | 정적+신경망 잔차 | 물리 파라미터(질량/관성/COM/댐핑) |
| 시간정렬 | soft(필터) | **불요** | 필요(윈도우 정렬) |
| 외부 센싱 | 필요(물체 위치) | **불요(proprio)** | IMU/엔코더 |
| 실기 데이터 | rollout 3회×반복 | **5분 미만** | 단일 암 추종 |
| 불안정/보행 적합 | 약함 | **강함** | 약함(contact 취약) |
| 우리 스택 일치 | 중(Flex→PhysX) | **최상(Go2/IsaacLab/PPO)** | 낮음(MJX 재구현) |
| 구현 난이도 | 중 | 하~중 | 상 |

**C4 권장 합성**: divergence/메트릭 = **#17 Wasserstein(보행 안전)**, 한 스텝 변화 제한 = **#16 KL trust-region**, 원인 역추적·관측가능성 게이트 = **#18 confidence 평가**(가능하면 MJX 미분 gradient). 셋의 강점을 단계별로 결합하는 것이 우리 Go2/R_Skeleton 닫힌 루프에 최적.

---

## 출처
- #16: https://arxiv.org/abs/1810.05687 (ar5iv 본문)
- #17: https://arxiv.org/abs/2604.11090 / https://arxiv.org/html/2604.11090v1
- #18: https://arxiv.org/abs/2202.09834 (ar5iv 본문)

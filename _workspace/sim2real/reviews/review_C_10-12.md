# Sim2Real 심화 리뷰 — 카테고리 C: System Identification (#10–#12)

> 작성: 2026-06-18 / 리뷰어: Sim2Real 논문 심화 리뷰어
> 대상 환경: IsaacLab + MJX. 강복님 적용 관점 = 토크/전류 실측 + encoder, excitation 설계 T5, 토크 불필요 교차검증 경로.
> 모든 내용은 arXiv 원문(abstract + HTML/PDF 본문)을 실제로 fetch하여 확인. 추론/추정 부분은 명시.

**번호 교정 주의 (중요):**
- #10의 실제 제목은 *"PACE"*가 아니라 **"Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots"** (Bjelonic, Tischhauser, Hutter, IJRR 투고). 과제 표에 적힌 "PACE"는 별칭/오기일 가능성이 높음 — 본문 어디에도 "PACE" 약어 미확인.
- #12의 robot/actuator는 초기 자동요약이 "Unitree A1"로 잘못 뽑았으나, 원문 확인 결과 **ROKI-2 모터(210:1 고감속) + miniPi 로봇**이 정답. 시뮬레이터는 MuJoCo가 아니라 **MJX(JAX MuJoCo)**.
- #12 다운스트림 수치는 "oracle 대비"가 아니라 **"baseline 대비"** travel distance +46%, rotational deviation −75%.

---

## #10. Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots (Bjelonic, Tischhauser, Hutter; ETH Zurich, IJRR 투고 2025)

- **확정 링크**: https://arxiv.org/abs/2509.06342
- **접근 분류**: arxiv abstract OK / PDF(>10MB) 직접 fetch 실패 → **HTML 풀텍스트(arxiv.org/html/2509.06342v1)로 본문 확인 완료**
- **우선순위**: **4순위** (가장 시스템 전체적·인프라 무거움)

### (1) 요약
PMSM(영구자석 동기모터) 기반 1차원리 에너지 모델을 sim-to-real RL에 통합하여, 다이내믹스 도메인 랜덤화 없이 **per-joint 파라미터 식별(4n+1차원)** 만으로 sim-to-real gap을 닫는 체계적 프레임워크. bottom-up 식별(단일 액추에이터 → 공중 전신 trajectory → 지상 보행)과 4-term reward(velocity tracking + PMSM energy + foot-touchdown + collision)를 결합. 3개 주력 플랫폼 + 10개 추가 로봇(총 13종)에 zero-shot 전이, ANYmal CoT를 1.86→1.27로 **32% 절감**.

### (2) Motivation
- 기존 sim-to-real은 다이내믹스 랜덤화로 gap을 "덮는다"(robust but conservative). 이 논문은 gap을 **식별로 닫는다(identify, not randomize)**.
- 모터 에너지 손실(Joule heating, 인버터 스위칭)이 보행 자체보다 더 많은 에너지를 먹는다는 관찰 → reward에 물리 기반 에너지 항을 넣어 자연스럽게 효율적 gait를 유도. (ANYmal D 분해: CoE 0.50 / CoD 0.24 / 보행 CoL 0.53 — 절반 미만만 실제 locomotion.)
- 13종의 이질적 로봇(pseudo-direct-drive, SEA, 3D 프린트 가변비 레버, hand 등)에 동일 파이프라인이 통한다는 일반성 입증이 목표.

### (3) Method + 수식 + 학습/식별 + 실험

**최소 파라미터 집합 (per-joint 4개 + 글로벌 1개 = 4n+1):**
```
p = [I_a, d, τ_f, q̃_b, T_d]ᵀ ∈ ℝ^(4n+1)
```
- `I_a`: armature/effective inertia (로터 관성 + 기계 보정)
- `d`: viscous damping (모터+기어 점성)
- `τ_f`: Coulomb friction (부하 무관 마찰)
- `q̃_b`: joint position bias (펌웨어 오프셋)
- `T_d`: command delay (통신 지연, ~7.5 ms 측정) — 글로벌 1개

**PMSM 에너지 모델 (reward의 핵심):**
```
P_total = P_el + P_mech + P_pot
P_el   = Σ R_j i_q,j²  =  Σ τ_j² · R_j / (r_j² k_i,j²)      # Joule heating, 토크²로 환산
P_mech = τᵀq̇            (τᵀq̇ > 0)
       = k_regen·τᵀq̇    (τᵀq̇ < 0)                          # 회생 제동 계수
P_pot  = Σ m_b g v_b,z
```
- `R_j` 코일저항, `r_j` 기어비, `k_i,j` 모터상수. 전류를 직접 안 재고 **토크를 전류 손실로 환산**하는 것이 포인트.
- 속도 정규화: `γ_v = 1/(‖v̂_B‖² + 1)` 로 속도대역 간 손실 균형.

**4-term reward:**
```
r = c_v r_v + c_c r_c + κ(c_e r_e + c_ftd r_ftd),   κ = 1 − exp(−λt)  (지수 스케줄)
r_v   = exp(−‖v̂_B,xy − v_B,xy‖²/σ_v) + exp(−(ω̂_yaw − ω_yaw)²/σ_v)   # velocity tracking
r_e   = γ_v · P_total                                                # 물리기반 에너지
r_ftd = foot touchdown velocity penalty                              # 착지 충격
r_c   = collision penalty (joint-limit/환경 충돌, binary)
```
원문 인용: *"Thanks to fitted dynamics parameters, we use only four terms."* 즉 식별이 잘 되므로 reward를 단순하게 유지 가능.

**Bottom-up 식별 (3단계):**
1. **단일 액추에이터**: 알려진 관성 질량을 단 고정 rig, 완전 전압원 조건에서 `{I_a, d, τ_f}` 식별. current-loop bandwidth ~346 Hz 검증.
2. **공중 전신 trajectory**: base 고정·다리 자유, **chirp 신호 0.1–10 Hz**(접촉 cross-coupling 회피)로 가진 → 4n+1 전체 동시 식별. 손실 `ℓ_e = (1/k) Σ ‖q_real − q_sim‖²`, **Isaac Gym 4096 병렬 환경**, **CMA-ES** 10–24 iter, RTX 3080 단일 GPU.
3. **지상 보행**: 식별 파라미터로 zero-shot 배포 검증(추가 식별 없음, 평가 전용).

**다이내믹스 랜덤화 사용 안 함** (명시): *"We do not use dynamics randomization."* 대신 task-level 변동(지면 마찰, push, terrain)만 랜덤화.

**실험/메트릭**: 13종(주력 3종 Tytan/ANYmal/Minimal + Aibo, NAO, ALMA, Spacehopper, LEVA, Magnecko v1/v2, GR-1, Allegro Hand, Ability Hand). 메트릭 = **Cost of Transport(CoT)**. ANYmal D CoT 1.27(vs 1.86, −32%), Tytan 0.97(최고효율). zero-shot 보행/달리기(~3 m/s, 배터리 32A 제한)/2족 밸런싱/계단(Minimal 20단 46초). 공중 검증서 real-sim trajectory 거의 완전 일치.

### (4) 문제점·한계·개선 여지
- **물리 파라미터(R_j, k_i,j, r_j)를 데이터시트/별도 측정으로 알아야** P_el 환산 가능 → 우리처럼 저가/미지 모터엔 추가 측정 부담.
- 단계 1이 **전용 test rig(알려진 관성, 전압원)** 를 요구 — "토크/전류 측정 불필요" 노선과 반대. 이 논문은 오히려 actuator-level에서 풍부한 계측을 가정.
- chirp 가진은 base-fixed 공중 상태라 **접촉/지면 다이내믹스(마찰, 충돌 복원)는 식별 안 됨** → 여전히 task-level 랜덤화로 덮음.
- 식별 정확도/CoT가 ANYmal급 고품질 SEA·pseudo-direct-drive에 유리. 백래시·gear 비선형 큰 저가 액추에이터로의 일반화 근거는 13종 deploy의 "성공 여부"만 제시, 정량 식별오차는 주력 3종 위주.

### (5) 우리 적용 (강복님)
- **토크/전류 실측 + encoder 보유 시 최적 fit.** R_j·k_i,j·r_j를 우리가 측정/추정 가능하면 PMSM 에너지 reward를 그대로 이식 가능. 우리 Go2/R_Skeleton actuator의 P_el 항을 reward에 넣어 효율 gait 유도 → 기존 IsaacLab reward에 **단일 energy term 추가**로 시작 가능.
- **excitation 설계 T5 연결**: 단계 2의 base-fixed chirp(0.1–10 Hz)는 우리 T5 excitation 설계의 직접 템플릿. IsaacLab에서 base를 fix하고 다리만 chirp로 가진하는 calibration env를 별도 task로 만들면 됨.
- **토크 불필요 교차검증 경로 관점에선 약함**: 이 논문은 actuator rig 계측에 의존하므로, "토크 없이"는 #12가 본가. #10은 우리가 토크/전류를 실제로 잴 수 있을 때의 **상한선(high-fidelity 경로)** 으로 채택.
- per-joint `q̃_b`(position bias)와 `T_d`(command delay) 식별은 우리 sim-to-real에서 자주 무시하는 항인데, IsaacLab actuator cfg에 추가하면 저비용 고효과 가능성.

### sim 구현 난이도: **상** — 필요 인프라
- Isaac Gym/IsaacLab 병렬 + **CMA-ES 식별 루프**(외부 nevergrad/cma 라이브러리) + per-joint 파라미터를 actuator model에 주입하는 cfg 후크.
- **전용 actuator test rig + base-fixed chirp calibration env** + (P_el용) 모터 전기 파라미터 계측. 13종 일반화는 불필요하나, 단일 로봇이라도 rig+chirp+CMA-ES 3중 인프라가 필요해 상.

---

## #11. SPI-Active: Sampling-Based System Identification with Active Exploration for Legged Robot Sim2Real Learning (Sobanbabu, He, He, Yang, Shi; CMU, arXiv 2025)

- **확정 링크**: https://arxiv.org/abs/2505.14266
- **접근 분류**: arxiv abstract OK / **HTML 풀텍스트(arxiv.org/html/2505.14266v1) 본문 확인 완료**
- **우선순위**: **2순위** (active excitation 설계 T5에 가장 직결)

### (1) 요약
두 단계 sim-to-real 식별 프레임워크. **Stage 1**: 대규모 병렬 샘플링(CMA-ES)으로 state-prediction error를 최소화해 관성(10차원, log-Cholesky) + 모터 강도(per-joint κ) 식별. **Stage 2**: **Fisher Information을 최대화(A-optimal: tr(F⁻¹) 최소화)** 하도록 멀티행동 정책의 command 시퀀스를 최적화해 정보량 큰 real trajectory를 능동 수집 → 재식별. Go2(4.7kg 페이로드)·G1 휴머노이드에서 jump/tracking 4개 task에 **42–63% 성능 향상**.

### (2) Motivation
- 수동 데이터(임의 보행)는 일부 다이내믹스 모드만 자극 → 파라미터 추정 불확실성 큼. **"무엇을 측정할지"를 능동 설계**(active exploration)하면 같은 데이터량으로 식별 정확도 ↑.
- IsaacGym 같은 RL 주력 시뮬레이터는 **미분 불가능하지만 대규모 병렬** → gradient 대신 **sampling 기반 최적화(CMA-ES)** 가 자연스러운 선택이라는 실용 논거.
- 미분 시뮬(#12)과 대비되는 "non-differentiable + parallel" 진영의 대표.

### (3) Method + 수식/파이프라인 + 학습/식별 + 실험

**식별 파라미터:**
- 관성 `θ_in`: log-Cholesky 분해 **10차원** — `ϕ = [α, d₁, d₂, d₃, s₁₂, s₂₃, s₁₃, t₁, t₂, t₃]ᵀ ∈ ℝ¹⁰` (질량 + CoM 3D + 관성텐서 6독립성분, 물리적 타당성 보장).
- 모터 `θ_mo`: per-joint κ. Go2는 hip/thigh/calf **3종 κ**.

**모터(토크 감쇠) 모델 — Eq.3:**
```
τ_motor = κ · tanh(τ_PD / κ),   τ_PD = K_p(q_target − q) − K_d q̇
```
κ가 작을수록 saturation 빨라짐 = 모터 강도/포화 식별.

**Stage 1 — sampling 기반 식별 (Eq.4, CMA-ES):**
```
J(θ, {c_k}) = Σ_k Σ_t ‖x^r_{t+1,k} − x_{t+1,k}‖²_Wx + ‖θ − θ₀‖²_Wθ
```
- H-horizon **multi-step sequential prediction error** + prior 정규화.
- 매 iter B개 후보 θ를 **IsaacGym에서 병렬 평가**, CMA-ES로 진화.
- Stage 1 데이터: heuristic motion prior + 다양한 사전학습 RL 정책으로 수집(검증엔 ~60초 teleop).

**Stage 2 — active exploration (Fisher Information):**
```
FIM (Eq.7):   F(θ*, π) ≈ σ⁻² · E[ Σ_t (∂f/∂θ)(∂f/∂θ)ᵀ ]
목적 (Eq.6):  c*_{1:T} = argmin_{c_{1:T}} tr( F(θ*, π)⁻¹ )       # A-optimal design
```
- **exploration 정책을 새로 학습하지 않고**, 멀티행동 정책의 **command 시퀀스 c만 최적화**해 다양한 모드를 자극. solver도 **동일 CMA-ES**.
- tr(F⁻¹) 최소 = 파라미터 추정 분산 최소 → 다음 라운드 데이터 정보량 극대화.

**실험/메트릭**: Go2(4.7kg payload) + G1 휴머노이드. Task = Forward Jump(0.85m), Yaw Jump(135°), Velocity Tracking(2D twist), Attitude Tracking(roll/pitch). 정규화 오차(작을수록 좋음, Vanilla=1.00): Forward Jump 0.48(−52%), Yaw Jump 0.37(−63%), Velocity 0.58(−42%), Attitude 0.73(−27%). 예측: root pos 0.67, per-joint angle 0.72. **42–63% 향상** = 4개 task 범위.

### (4) 문제점·한계·개선 여지
- FIM의 `∂f/∂θ`(민감도)를 **non-differentiable 시뮬에서 어떻게 얻는가**가 핵심 비용. 유한차분/샘플 근사면 차원·노이즈에 약함 — 본문은 근사식만 제시, 계산 디테일은 가벼움.
- `σ⁻²` 노이즈 모델을 등분산 가정 → 실제 센서/접촉 노이즈 이질성 무시.
- 관성 10차원 + per-joint κ만 식별, **마찰/damping/지연은 명시적 비포함**(모터 강도에 일부 흡수). #10·#12 대비 액추에이터 다이내믹스 표현력 낮음.
- κ tanh 모델은 단순(steady-state 포화만). 백래시·속도의존 마찰 미표현.
- Active exploration이 **CMA-ES inner loop를 매 라운드 도는 비용** + real-robot에서 excitation 시퀀스 실행 안전성(점프/공격적 command) 이슈.

### (5) 우리 적용 (강복님)
- **excitation 설계 T5의 정공법.** FIM tr(F⁻¹) 최소화로 "어떤 command가 가장 정보량 큰가"를 정량 설계 → 우리 T5 excitation을 **임의 chirp(#10)에서 FIM-최적 command로 업그레이드** 가능. IsaacLab + CMA-ES면 미분 불필요해 현 인프라와 호환.
- **토크 불필요 교차검증 경로에 부합**: state-prediction error(encoder 기반 x = q, q̇, root state)만 쓰므로 토크/전류 없이 식별 가능. 강복님이 토크/전류를 실측하면 그건 **추가 검증 채널**로, 식별 자체는 encoder만으로 성립.
- **κ tanh 모터모델은 우리 Go2 actuator cfg에 즉시 이식 가능**(per-joint scaling, 3~12 파라미터). 가장 가벼운 첫걸음.
- 단점 보완: FIM 민감도는 우리가 **#12의 MJX 미분으로 정확히 계산**해 SPI-Active의 active 설계와 결합하는 하이브리드가 매력적(아래 #12 적용 참고).

### sim 구현 난이도: **중** — 필요 인프라
- IsaacLab(IsaacGym) 병렬 + **CMA-ES(cma/nevergrad)** + state-prediction loss rollout 비교 + log-Cholesky 관성 reparam.
- Stage 2 FIM은 추가 구현(민감도 근사 + tr(F⁻¹) 최적화 inner loop). 미분 시뮬 불필요해 #12보다 인프라 가벼움 → 중.

---

## #12. Trajectory-based actuator identification via differentiable simulation (Kovalev, Chaikovskaia, Davydenko, Gorbachev; arXiv 2026)

- **확정 링크**: https://arxiv.org/abs/2604.10351
- **접근 분류**: arxiv abstract OK / **PDF 직접 fetch 성공(2.9MB) + HTML v2 본문 교차확인 완료**
- **우선순위**: **3순위** (미분 시뮬·토크리스 식별, MJX 직결)

### (1) 요약
**토크 센서·전류/전압·모터 내부 접근 없이 encoder 모션만으로** 액추에이터 모델을 fit. 식별을 trajectory-matching 문제로 두고 **MJX(미분 가능 MuJoCo)** 를 통해 backprop. 구조적 파라미터(PD gain + armature 3개)부터 신경망 액추에이터까지 통합 파이프라인. ROKI-2의 **210:1 고감속 액추에이터**(임베디드 PD 내장) held-out trajectory에서 stand-trained supervised 대비 MAE **14.20 → 7.54 mrad(1.88×)** 개선. miniPi 보행 다운스트림서 **travel distance +46%, rotational deviation −75%(baseline 대비)**.

### (2) Motivation
- 고품질 액추에이터 모델은 보통 **전용 test stand + 토크 센싱**을 요구 → 비용·접근성 장벽. 또 stand 데이터는 steady-state 위주라 동적 거동 일반화 약함.
- encoder(commanded position, measured angle/velocity)만 가지고, **미분 시뮬 gradient로 actuator+simulator 파라미터를 직접 fit** → stand·토크센서 제거.
- 미분 가능 시뮬(MJX) 진영의 actuator-ID 대표. #11(sampling)과 정반대 접근.

### (3) Method + 수식/파이프라인 + 학습/식별 + 실험

**식별 = trajectory matching, backprop through sim:**
```
ℒ_batch(z) = (1/MN) Σ_i Σ_j ‖W (s'_{i,j} − s_{i,j})‖²₂      (초기상태 일치 제약)
```
- s = (joint angle, velocity). commanded position 입력 → MJX rollout → measured와 비교.
- **MJX(JAX MuJoCo) explicit Euler integrator** 통해 `θ_a`로 gradient 역전파.
- 최적화: **Adam(Optax), lr 1e-2, M=2000 trajectory segment/step, 3000 epoch**.

**고감속 + 임베디드 PD 액추에이터 모델:**
```
u  = K_p(q_des − q) + K_d(q̇_des − q̇)
DC: L_a (di/dt) = u − K_e q̇ − i R_a
 q̈ = N_g/(I_m + N_g² I_a) · [ N_g K_m i − b q̇ − sign(q̇) f_c ]
```
고감속 극한(quasi-static)에서 관성·마찰 지배. (구조 파라미터 버전은 이 중 핵심 소수만 fit.)

**모델 클래스(통합 파이프라인):**
- **TrajID-Param**: PD gain (k_p, k_v) + armature = **3 파라미터** (compact structured).
- **TrajID-NN**: feed-forward MLP **[3,32,32,1]** (neural actuator mapping).
- **Torque-Oracle**: per-timestep 자유 토크 시퀀스(상한 참조).
- 비교 baseline: **Bench-Sup**(stand supervised), **Param-ES**(gradient-free), **Residual-RL**(action correction).

**실험/메트릭**:
- 하드웨어: **ROKI-2 모터(210:1, hand-tuned 임베디드 컨트롤러)**, 다운스트림은 **miniPi 로봇**(동일 액추에이터 클래스).
- held-out trajectory MAE(작을수록 좋음): Bench-Sup **14.20±11.50 mrad** → TrajID-NN **7.54±4.57 mrad**(≈1.88×). steady-state 지배 stand baseline보다 동적 정합 우수.
- **다운스트림 보행(baseline 대비)**: travel distance 1.12±0.13 m → 1.64±0.08 m(**+46%**), rotational deviation 109±14.3° → 27±6.4°(**−75%**).

### (4) 문제점·한계·개선 여지
- 단일 액추에이터 클래스(ROKI-2 210:1)·단일 로봇(miniPi)만 검증 → **일반화/스케일(다관절 동시 식별, 접촉 포함) 미입증**. 보고된 trajectory도 base 단순.
- MJX 미분 = **메모리/속도 비용**(explicit Euler, BPTT through rollout). 긴 horizon·접촉 풍부 시 gradient 폭주/불안정 위험(본문은 단순 trajectory라 회피).
- TrajID-NN이 최고 MAE지만 **물리 해석성 상실**·과적합 위험. structured(3-param)와의 trade-off는 제시되나 sim-to-real robustness 비교는 제한적.
- "torque-free"가 강점이나, 임베디드 PD를 모델이 정확히 알아야(또는 동시 식별) 하는 의존성 존재.

### (5) 우리 적용 (강복님)
- **토크 불필요 교차검증 경로의 정확한 본가.** 우리가 토크/전류를 실측하더라도, #12 경로(encoder만 + MJX backprop)로 **독립 식별을 돌려 토크기반 결과와 교차검증** 가능 → 강복님 요구의 "토크 불필요 교차검증"을 그대로 구현.
- **MJX 환경 직결**: 우리 MJX 인프라에서 actuator+sim 파라미터를 trajectory-matching loss로 backprop. IsaacLab actuator cfg(armature, damping, friction, PD)를 MJX로 미분 식별 후 IsaacLab으로 역이식.
- **#11과의 하이브리드 제안**: #12의 MJX gradient로 SPI-Active(#11)의 FIM 민감도 `∂f/∂θ`를 **정확히 계산**(샘플 근사 대신) → active excitation(T5) 설계 품질 ↑. 즉 식별=#12(미분), 가진설계=#11(FIM), 효율reward=#10(PMSM)의 3편 결합이 우리 풀스택 로드맵.
- structured(TrajID-Param 3개)부터 시작 권장: armature+PD 3파라미터 fit은 우리 Go2에 즉시 적용 가능, NN버전은 나중.

### sim 구현 난이도: **중상** — 필요 인프라
- **MJX(JAX MuJoCo) + Optax(Adam)** 미분 식별 파이프라인. BPTT through rollout 메모리 관리.
- encoder 로그(commanded pos / measured q, q̇) 수집 + MJX 모델에 actuator 파라미터 노출 + (선택)MLP actuator. 미분 안정화(short horizon, gradient clip) 필요 → 중상. (MJX를 이미 쓰면 #11보다 코드량 적으나 미분 안정성 디버깅이 난점.)

---

## 종합 비교 (한 줄 요약)

| # | 접근 | 식별 엔진 | 핵심 무기 | 토크 필요? | 우리 적용 핵심 |
|---|------|----------|----------|-----------|--------------|
| #10 | identify-not-randomize | IsaacGym + CMA-ES | PMSM 에너지 reward + 4n+1 per-joint | rig 계측 필요(상한경로) | 효율 reward + chirp T5 템플릿 |
| #11 | sampling + active | IsaacGym + CMA-ES | FIM tr(F⁻¹) excitation 설계 | 불필요(encoder state) | **T5 가진설계 정공법** |
| #12 | differentiable | **MJX** + Adam | trajectory-match backprop | **불필요(encoder only)** | **토크리스 교차검증 + MJX 직결** |

**권장 통합 로드맵**: 식별=#12(MJX 미분, 토크리스) → 가진설계=#11(FIM, T5) → 효율 gait=#10(PMSM energy reward). 강복님 토크/전류 실측은 #12 encoder-only 식별의 **독립 검증 채널**로 사용.

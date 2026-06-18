# 카테고리 E (Foundational) 심화 리뷰 — #19, #20

> 작성: Sim2Real 논문 심화 리뷰어 / 대상 인프라: IsaacLab + MJX
> 원문 확인 방법: WebSearch + WebFetch로 실제 원문/배포본 확인. (#19은 arxiv 미배포 — 번호 교정 결과 본문 참조)

---

## #19. Differentiable Simulation for Physical System Identification (Le Lidec, Kalevatykh, Laptev, Schmid, Carpentier, RA-L 2021 / ICRA 2021)

- **확정 링크**:
  - 저널(IEEE Xplore): https://ieeexplore.ieee.org/document/9363565/ — IEEE RA-L, vol. 6, no. 2, pp. 3413–3420, 2021. DOI: `10.1109/LRA.2021.3062323`
  - 오픈액세스 PDF(HAL/Inria): https://hal.science/hal-03025616 / https://inria.hal.science/hal-03025616
- **번호 교정 (중요)**: 의뢰서의 "arxiv 2103.xxxxx" 추정은 **오류**. 본 논문은 **arXiv에 배포되지 않았다.** 정식 식별자는 IEEE DOI `10.1109/LRA.2021.3062323`이며, 프리프린트는 **HAL `hal-03025616`** 로만 공개되어 있다. 제목도 의뢰서의 "Differentiable Simulation for Physical System Identification"으로 정확(대소문자만 출판본은 소문자 "simulation ... identification").
- **접근 분류**: Foundational (미분가능 시뮬레이션 + SysID의 이론적 토대). 알고리즘/네트워크 개선이 아니라 "시뮬레이터 자체를 미분해서 동역학 파라미터를 식별"하는 방법론.
- **우선순위**: 중. 강복님 SysID-Diff 트랙의 **이론적 앵커**이지만, IsaacLab/MJX 직접 적용에는 시뮬레이터 backend 교체/미분 경로 확보라는 큰 인프라 비용이 따름.

### (1) 요약
마찰 접촉(frictional contact)을 정확하게 풀면서 **동시에 미분 가능**한 강체 시뮬레이터를 제안한다. 핵심은 (a) 마찰 접촉의 **Nonlinear Complementarity Problem(NCP)** 을 LCP로 선형 근사하지 않고, **staggered projections** 알고리즘을 확장하여 Coulomb 마찰 원뿔(friction cone)과 탄성 충돌을 정확히 처리한 것, (b) 이 접촉 풀이를 **QCQP(Quadratically Constrained QP) 시퀀스**로 정식화하고 그 해에 대한 **해석적 미분(analytical derivatives)** 을 implicit differentiation으로 유도한 것이다. 이렇게 얻은 gradient를 시뮬레이션 출력(궤적)과 관측(실제 비디오)의 오차에 대해 역전파하여, **마찰계수 μ와 물체 질량 M 같은 물리 파라미터를 gradient-based로 식별**한다. 합성 및 실제(비디오 마커) 실험에서 마찰계수/질량을 정확히 추정함을 보였고, **파라미터 관측가능성(observability)** 문제를 명시적으로 논한다.

### (2) Motivation
- **고전 시뮬레이터의 정확도 손실**: 대부분의 물리 엔진은 마찰 접촉을 풀기 위해 NCP를 **LCP로 캐스팅**(다면체로 마찰 원뿔 근사)하여 계산을 단순화한다. 이는 물리적 부정확을 낳는다.
- **미분 불가능성**: 이런 접촉 풀이는 LCP pivoting, projection 등 **비평활(non-smooth) 연산**을 포함해 autodiff로 곧장 미분이 안 된다. SysID/제어 최적화에 쓰려면 gradient가 필요한데 기존 경로가 막혀 있다.
- **목표**: "정확한 접촉 물리"와 "미분 가능성"을 **동시에** 달성하여, 시뮬레이터를 미분 가능한 추론 엔진으로 만들고, 이를 통해 실제 관측(영상)으로부터 동역학 파라미터를 자동 식별한다.

### (3) Method + 수식/파이프라인 + 학습 방법 + 실험 방법

**접촉/마찰 모델 (contact 모델이 gradient 거동에 미치는 영향)**
- 마찰 접촉은 본질적으로 **NCP** (속도-임펄스 상보성 + Coulomb cone). 정확히 풀려면 비선형 원뿔 제약을 유지해야 한다.
- 본 논문은 **staggered projections**(Kaufman et al. 류)를 확장: 접촉 normal 임펄스와 friction 임펄스를 교대(staggered)로 투영하며 수렴시키되, **friction cone과 탄성(restitution) 충돌**을 함께 다루도록 일반화. 각 sub-step이 **QCQP**(2차 목적 + 2차(cone) 제약)로 표현된다.
- **gradient 거동에 대한 함의**: LCP(다면체 근사)는 facet 경계에서 미분이 끊기고 부호가 튀는(non-smooth) gradient를 만든다. NCP를 원뿔로 정확히 유지하면 접촉 상태(정지/슬립)가 바뀌는 경계가 더 매끄럽게 다뤄지고, **active set이 일정한 영역 안에서는 KKT 조건이 매끄럽게 성립** → implicit function theorem 적용이 깔끔해진다. 즉 **정확한 접촉 정식화 자체가 "well-behaved gradient"의 전제조건**이다.
- **well-behaved gradient의 조건 (논문 관점 정리)**: ① 접촉 풀이가 QCQP의 KKT 시스템으로 표현되어 implicit diff가 가능할 것, ② 미분 시점에서 **active constraint set이 고정(부드러운 국소영역)** 일 것 — 정지↔슬립, 접촉 on/off 전환 순간에는 gradient가 불연속이 됨, ③ 관측이 해당 파라미터에 대해 **observable**(궤적이 파라미터에 민감)할 것. ②③가 깨지면 gradient는 정의되더라도 식별에 쓸모없는(0이거나 무한대로 튀는) 값이 된다.

**해석적 미분 (analytical derivatives)**
- 접촉 풀이를 QCQP의 **KKT 최적성 조건** F(z\*, θ)=0 (z\*=해, θ=동역학 파라미터: μ, M, …)로 본다.
- **Implicit Function Theorem**: ∂z\*/∂θ = −(∂F/∂z)⁻¹ (∂F/∂θ). 즉 forward pass에서 푼 해의 KKT 시스템을 한 번 더 선형으로 풀어 **해석적 gradient**를 얻는다(autodiff로 풀이 루프를 펼치지 않음 → 메모리/정확도 이득).
- 전체 시뮬레이션은 이 접촉-스텝을 시간축으로 합성한 미분가능 chain → ∂(궤적)/∂θ 를 backprop.

**SysID 정식화 + 모델 학습 방법**
- loss: 시뮬레이션 궤적과 관측 궤적의 차이를 최소화. 대략
  L(θ) = Σ_t ‖ x_sim,t(θ) − x_obs,t ‖²  (관측 = 실제 비디오에서 마커로 추출한 물체 pose 시퀀스)
- 최적화: θ ← θ − η ∂L/∂θ, gradient는 위 해석적 미분으로 계산 (gradient descent).
- 식별 대상 파라미터: **Coulomb 마찰계수 μ**, **물체 질량 M** (등 동역학 파라미터). 신경망 학습이 아니라 **물리 파라미터 자체를 변수로 두는 최적화**(미분가능 시뮬을 통한 inverse problem).

**실험 방법 / 메트릭**
- **합성(synthetic)**: ground-truth μ, M을 아는 시뮬 시나리오에서 추정값이 진값에 수렴하는지(추정 오차) 확인. finite-difference / 근사 미분 대비 정확도·안정성 비교 맥락.
- **실제(real)**: 실제 물체가 미끄러지는/충돌하는 장면을 **비디오로 촬영**, 마커로 물체 궤적을 추출 → 시뮬을 그 궤적에 맞추도록 μ, M을 식별.
- **메트릭**: 식별 파라미터의 추정 정확도(추정 μ, M vs 측정/GT), 궤적 재현 오차(L 값 감소), 그리고 **관측가능성 분석**(궤적이 파라미터를 충분히 드러내지 못하면 식별이 불안정함을 정량/정성적으로 논의).

### (4) 문제점·한계·개선 여지
- **observability 의존**: 논문 스스로 강조 — 운동 궤적이 μ/M에 민감하지 않은 시나리오(예: 슬립이 거의 없는 운동)에서는 gradient가 정보가 없어 식별 실패/모호. 정지마찰만 있는 구간에서는 μ가 식별 불가.
- **비평활 전이 구간**: 접촉 on/off, 정지↔슬립 전환 순간 gradient 불연속 → 최적화 landscape가 국소최소/평탄구간을 가짐. 본 논문은 정확한 NCP로 완화하지만 근본적으로 남는 문제(후속 연구들이 randomized smoothing 등으로 보완).
- **확장성/속도**: QCQP 시퀀스를 풀고 KKT 역행렬을 매 스텝 계산 → 대규모 병렬(수천 env) RL용 시뮬과는 결이 다름. rigid-body 소수 물체 시나리오 중심.
- **rigid contact + 점/마커 관측 가정**: deformable, 복잡 형상, 비전 노이즈가 큰 실제 로봇 전신 식별로의 일반화는 검증 범위 밖.
- **개선 여지**: 다물체/관절 로봇으로 확장, smoothing/stochastic relaxation 결합으로 비평활 구간 gradient 안정화, 신경망 residual dynamics와 결합.

### (5) 우리 적용 (강복님: SysID-Diff 이론 기반, 식별 동역학 → 실기 상태추정 연결)
- **이론 앵커로서의 위치**: 본 논문은 "미분가능 시뮬로 **동역학 파라미터(μ, 질량, 관성)** 를 실측에서 식별"하는 정통 프레임. 강복님 트랙의 첫 단계(실기 데이터 → sim 파라미터 보정 = domain alignment)의 이론적 근거.
- **#20과의 연결 (식별 동역학 → 실기 상태추정)**: #19로 **sim의 동역학을 실기에 맞춰 캘리브레이션**(μ, 링크 질량/관성, 접촉 파라미터) → 그 보정된 sim에서 #20 방식으로 **state estimator(base velocity, contact, foot height)** 를 학습하면, estimator가 보는 sim 분포가 실기에 더 가까워져 sim2real gap이 줄어든다. 즉 **#19 = 동역학 정합, #20 = 그 정합된 동역학 위에서의 상태추정 학습**이라는 2단 파이프라인이 자연스럽다.
- **IsaacLab/MJX 현실론**:
  - IsaacLab(PhysX)은 미분가능하지 않다 → #19를 곧이곧대로 IsaacLab 안에서 구현 불가. **MJX(MuJoCo-XLA)** 가 JAX 기반으로 **미분가능**하므로 #19식 gradient-based SysID는 **MJX 트랙에서** 구현하는 것이 정석.
  - 권장 분업: **MJX = SysID(동역학 식별/캘리브레이션) 엔진**, **IsaacLab = 대규모 병렬 RL/estimator 학습 엔진**. MJX로 식별한 μ, 질량, 관성, 접촉 파라미터를 IsaacLab cfg(`*_env_cfg.py`의 material/mass/inertia, friction)로 **이식**.
  - 다만 MJX 접촉은 soft/convex 모델이라 #19의 정확 NCP/staggered-projection과 다름 → "정확 접촉 식별"을 그대로 재현하긴 어렵고, **MJX의 기본 미분경로로 마찰/질량을 fit하는 실용 버전**으로 차용하는 것이 현실적.
- **구체 활용 시나리오 (Go2 등)**: 실기 Go2의 짧은 슬립/충돌 모션을 모션캡처/로그로 확보 → MJX에서 발-지면 마찰계수와 링크 질량을 미분가능 fit → 캘리브레이션된 파라미터를 IsaacLab domain randomization의 **중심값**으로 설정(randomization 범위는 식별 불확실성으로). observability가 약한 파라미터는 식별 대신 randomization 유지.

### sim 구현 난이도: 상 — 필요 인프라
- **상** 이유: 미분가능 접촉 시뮬 backend 필요(IsaacLab/PhysX로는 불가, MJX/Brax/Warp 등으로 별도 트랙 구축), QCQP+KKT 해석미분은 직접 구현 시 난도 높음(MJX 기본 autodiff로 대체하면 정확도는 양보), 실기 관측(비디오 마커/모션 로그) 파이프라인과 observability 분석이 추가로 필요.
- 필요 인프라: JAX + MJX(또는 Brax/diffsim), 미분가능 forward dynamics, gradient 기반 파라미터 optimizer(Adam/L-BFGS), 실기 궤적 수집·정렬(time-sync, marker/pose extraction), 식별 결과를 IsaacLab cfg로 옮기는 브리지 스크립트.

---

## #20. Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged Locomotion (Ji, Mun, Kim, Hwangbo, KAIST, RA-L 2022 / ICRA 2022)

- **확정 링크**: https://arxiv.org/abs/2202.05481 (PDF: https://arxiv.org/pdf/2202.05481). IEEE RA-L vol. 7, no. 2, April 2022, 페이지 4630–4637; ICRA 2022 발표. (의뢰서 링크 정확)
- **접근 분류**: Foundational (proprioception-only 상태추정 + 정책 동시학습의 표준 레시피). sim2real 배포의 핵심 빌딩블록.
- **우선순위**: 상. estimator-policy 동시학습은 IsaacLab의 rsl_rl 트랙(이미 parkour estimator 경험 보유)과 직접 호환되며, 우리 sim2real 배포 파이프라인에 곧장 적용 가능.

### (1) 요약
사족보행 로봇(Mini Cheetah)에서 **제어 정책(policy)** 과 **상태추정기(state estimator)** 를 **동시에(concurrently)** 학습한다. 정책은 PD 목표 관절각을 출력하고, estimator는 proprioception+IMU만으로 **base 선속도, 발 높이(foot height), 접촉 확률(contact probability)** 을 예측한다. 두 네트워크는 **gradient를 분리**하여(estimator의 supervised loss는 estimator로만, policy는 PPO로만 학습) 동시에 최적화한다. 추정된 상태를 정책 입력으로 써서 외부 센서 없이 실기 배포가 가능하다. **평지 최대 3.75 m/s, 마찰계수 0.22 미끄러운 판 위 3.54 m/s**의 강건/고속 보행을 실기에서 달성.

### (2) Motivation
- **privileged state 부재**: sim에서는 base 선속도·접촉·발 높이 같은 특권 정보가 공짜지만 실기에는 직접 센서가 없다(특히 base 선속도). 이를 추정 없이 쓰면 sim2real에서 무너진다.
- **추정-제어 결합 문제**: 별도로 학습한 estimator를 사후에 붙이면 정책이 estimator의 오차 분포에 적응하지 못한다. **동시학습**하면 정책이 estimator의 실제 추정 분포(노이즈 포함)에 자연스럽게 강건해진다.
- **목표**: proprioception+IMU만으로 동작하는, 고속이면서 저마찰/험지에 강건한 사족보행을 실기에 전이.

### (3) Method + 구조/loss + 학습 + 실험 메트릭

**동시학습 구조 (policy + base-state estimator)**
- **두 MLP 네트워크**:
  - Policy: 입력 ≈ 48차원(proprioception + estimator가 추정한 상태), hidden [256, 256] ReLU, 출력 12차원(12-DOF PD 목표각).
  - State estimator: 입력 ≈ proprioception + IMU(가속도, 각속도), hidden [256, 256] ReLU, 출력 ≈ base 선속도(2~3), foot height(4), contact probability(4).
- **데이터 흐름**: estimator가 매 스텝 base velocity/foot height/contact를 추정 → 이 추정치가 정책의 관측으로 들어감(특권정보의 **학습된 대체물**). 실기에서는 GT 특권정보 없이 이 추정치만으로 동작.

**loss 함수 / gradient 분리**
- **Estimator loss (supervised regression, MSE)**:
  L_est = Σ ‖ ŝ − s_true ‖²,  ŝ = (base 선속도, foot height, contact prob), s_true = 시뮬레이터 GT.
- **Policy loss**: 표준 **PPO** (clipped surrogate + entropy 정규화).
- **gradient 분리(핵심 설계)**: estimator loss는 **estimator 네트워크로만 backprop**, policy의 PPO gradient는 **policy로만** 흐른다. 두 gradient가 서로 침범하지 않음 → estimator는 정책 보상에 오염되지 않고 순수 회귀로, 정책은 추정치를 입력으로만 받아(추정치 쪽으로 gradient 안 보냄) 학습. 한 epoch 안에서 **PPO 업데이트 ↔ supervised estimator 업데이트를 번갈아** 수행.

**플랫폼 / 시뮬레이터 / 도메인 랜덤화**
- 로봇: **Unitree(MIT) Mini Cheetah**, 12-DOF(다리당 3).
- 시뮬레이터: **RaiSim**(고충실 접촉 동역학; Hwangbo 그룹).
- Domain randomization: 마찰계수 0.22–1.5, 질량 ±20%, 모터 지연/latency, 경사 최대 ~30°. 평지→험지/경사 curriculum.

**실험 / 메트릭 / 결과**
- 메트릭: 최대 전진 속도, 마찰 변화에 대한 강건성, 경사/장애물 성공률, 정성적 험지 주파성.
- baseline: privileged(GT 상태) oracle 정책, proprioception-only(추정 없음), 대체 estimator 구조/순차학습.
- 결과: **평지 최대 3.75 m/s**, **마찰 0.22 저마찰 판 위 3.54 m/s**, 경사·바위 지형 안정 주파, 실기 전이 시 sim 성능의 큰 손실 없이 동작. **동시학습이 순차학습(정책→estimator)보다 우수** — 상호 이득 확인.

### (4) 문제점·한계·개선 여지
- **접촉 추정 정확도 의존**: contact/foot-height 추정이 틀리면 성능 저하. 험지 센싱이 나쁘면 무너진다.
- **분포 밖 마찰 일반화 약화**: 학습 범위(0.22–1.5) 밖(<0.2, >1.5)에서 성능 저하.
- **동적 환경 미대응**: 움직이는 장애물/동적 지형 처리 없음.
- **이중 네트워크 추론 오버헤드**: policy-only 대비 ~2–3× 추론 비용.
- **외부 지형 정보 없음**: proprioception+IMU만 → 사전 지형 예측 불가(blind locomotion 한계).
- **개선 여지**: 시각/높이맵 추가, estimator에 시계열(history/RNN) 강화, RMA/teacher-student와 결합, 동시학습의 추정-정책 상호 gradient를 더 정교하게.

### (5) 우리 적용 (강복님: 식별 동역학 → 실기 상태추정 연결)
- **직접 적용성 매우 높음**: 우리 rsl_rl 트랙은 이미 parkour estimator 경험(base lin_vel 추정, AMP 러너 estimator 버그 fix 이력)이 있어 본 레시피와 거의 동형. **gradient 분리 + supervised MSE estimator + PPO policy 동시학습**은 우리 코드 패턴과 호환.
- **#19와의 결합**: #19(MJX 미분가능 SysID)로 발-지면 마찰·링크 질량/관성을 실기에 맞춰 식별 → 그 캘리브레이션 값으로 IsaacLab DR 중심 설정 → #20식 estimator(base velocity/contact/foot height)를 그 정합된 sim에서 학습. **"동역학 식별(#19) → 상태추정 학습(#20) → 실기 배포"** 의 정합된 sim2real 체인이 강복님 트랙의 골격이 됨.
- **IsaacLab 구현 매핑**:
  - estimator 출력: base lin_vel(우리 priv 정보 중 핵심) + contact prob + foot height. 우리 `_get_observations`/priv_latent 구조에 추가.
  - loss 분리: rsl_rl algorithm `update()`에서 estimator supervised loss를 별도 optimizer로(detach 위치 주의 — estimator 입력은 detach, GT는 sim에서 직접). 우리 메모리의 "AMP estimator는 actor만 추정값, disc/critic 실제값" 원칙과 정합.
  - DR: 마찰 0.22–1.5, 질량 ±20%, 모터 latency를 IsaacLab cfg에 반영(중심값은 #19 식별값).
- **주의(우리 제약 메모 연계)**: contact sensor를 **obs에 직접 넣지 말 것**(sim2real 금지 항목) — #20도 contact을 **센서가 아니라 estimator로 추정**한다는 점이 우리 제약과 정확히 일치. 이 논문은 "contact을 obs로 직접 주지 않고 추정한다"는 우리 원칙의 좋은 레퍼런스.

### sim 구현 난이도: 중 — 필요 인프라
- **중** 이유: 새 시뮬 backend 불필요(IsaacLab 그대로), 추가 estimator MLP + supervised loss + gradient 분리만 구현하면 됨. 우리 rsl_rl estimator 경험으로 진입장벽 낮음. GT 라벨(base velocity/contact/foot height)은 IsaacLab에서 이미 접근 가능.
- 필요 인프라: rsl_rl actor-critic + estimator 동시학습 루프(우리 기보유), estimator용 별도 optimizer/loss, DR cfg(마찰/질량/latency), 실기 배포용 ONNX/JIT export(우리 exporter 경험 보유), 그리고 (#19 결합 시) MJX 식별값 → IsaacLab cfg 브리지.

---

## 부록: 두 논문의 sim2real 체인 요약 (강복님 트랙)

```
실기 로그/비디오
   │  (마커/pose, 짧은 슬립·충돌 모션)
   ▼
[#19] MJX 미분가능 SysID  →  μ, 링크 질량/관성, 접촉 파라미터 식별
   │  (observability 약한 파라미터는 식별 대신 DR 유지)
   ▼
IsaacLab cfg 보정 (식별값 = DR 중심값)
   │
   ▼
[#20] policy + state estimator 동시학습 (base vel/contact/foot height 추정)
   │  (gradient 분리: estimator=MSE, policy=PPO)
   ▼
ONNX/JIT export → 실기 Go2 배포 (proprioception+IMU only)
```

- 핵심 메시지: **#19 = 동역학을 실기에 맞추는 단계(정합)**, **#20 = 정합된 동역학 위에서 센서 없이 상태를 복원하는 단계(추정)**. 둘을 직렬로 쓰면 "정확한 sim 분포 + 그 분포에 강건한 추정/정책"이 되어 sim2real gap을 양쪽에서 줄인다.

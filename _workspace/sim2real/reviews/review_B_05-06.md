# Sim2Real 심화 리뷰 — 카테고리 B: Analytic/Parametric Actuator Models

> 리뷰어: Sim2Real 심화 리뷰어 (arxiv 원문 직접 정독 기반, 스니펫 추론 배제)
> 작성일: 2026-06-18
> 환경 전제: IsaacLab / MJX
> 담당: #5 Tan et al. (Google, RSS 2018), #6 Lee et al. (KAIST, RA-L 2026)

---

## #5. Sim-to-Real: Learning Agile Locomotion for Quadruped Robots (Tan, Zhang, Coumans, Iscen, Bai, Hafner, Bohez, Vanhoucke — Google Brain/X/DeepMind, RSS 2018)

- **확정 링크**: https://arxiv.org/abs/1804.10332 (PDF: https://arxiv.org/pdf/1804.10332, RSS14 p10, video: youtube.com/watch?v=lUZUr7jxoqM)
- **접근 분류**: Analytic/Parametric — 이상적 DC 모터 해석식(EMF/전류/토크) + piecewise 토크-전류 보정 + 명시적 latency 보간 모델
- **우선순위**: 최상위(★★★★★). Sim2Real actuator/latency/randomization 파이프라인의 정전(canonical) 레퍼런스. 이후 거의 모든 legged sim2real 논문(ANYmal, Hwangbo 2019, parkour류)이 이 actuator+latency+DR 삼각형을 계승. 강복님 PMSM 토크/전류 실측 어젠다와 정확히 같은 좌표계.

### (1) 요약
직접구동(direct-drive) 8액추에이터 4족 로봇 Minitaur에 대해, 시뮬레이션에서만 학습한 정책을 추가 실기 fine-tuning 없이 trotting/galloping 두 agile gait로 전이시킨 시스템 논문. reality gap을 두 축으로 좁힘: (A) 시뮬레이터 fidelity 향상 — URDF 질량/관성 식별 + 해석적 actuator 모델(이상 DC 모터식 + piecewise 토크-전류 saturation) + 명시적 latency 모델, (B) robust 정책 학습 — dynamics randomization(Table I), 외란 주입(130~220N), **compact observation space**. 핵심 ablation: latency 모델과 actuator 모델 둘 중 하나라도 빠지면 실기 동작 실패. observation을 12D→4D(IMU만)로 줄였을 때 sim 성능은 낮아지나 **실기 reality gap은 오히려 좁아짐**(overfitting 억제). 학습 gait가 expert handcrafted 대비 에너지 35%(gallop)/23%(trot) 절감.

### (2) Motivation
- Under-actuated 4족 로봇의 agile motion은 잦은 contact switching으로 control space가 조각나 manual tuning이 지난함. 실기 직접 학습은 낙상 위험·리셋 비용 때문에 비현실적 → sim 학습이 더 빠르고 안전.
- 그러나 reality gap(모델 불일치)이 locomotion에서 특히 증폭됨: agile motion의 잦은 접촉 전환에서 작은 모델 오차가 분기(bifurcation)되어 control을 깨뜨림.
- 저자들의 핵심 진단: 선행 연구들이 간과한 **부정확한 actuator 모델**과 **latency 미모델링**이 reality gap의 두 주요 원인. (시뮬레이터의 default constraint-based actuator는 overdamped → agile gait가 안 나옴.)
- 부가 목표: 사용자에게 "완전 from-scratch ↔ 사용자 지정 gait" 전 스펙트럼 controllability 제공.

### (3) Method + 수식/파이프라인 + 학습/실험 방법

**플랫폼 / HW 아키텍처**
- Minitaur (Ghost Robotics): 4족, 다리당 2 direct-drive 모터(총 8개, sagittal plane). 모터 encoder + IMU 탑재.
- 제어 체인: Nvidia Jetson TX2(NN 정책 추론) ↔ UART ↔ STM32 ARM MCU(센서 수집 + 모터 명령). 비실시간 OS 때문에 control loop가 가변 주파수 **~150–200Hz**.
- 시뮬레이터: PyBullet(Bullet) — articulated rigid body, contact/joint limit/actuator 제약, semi-implicit 적분.

**관측/행동 공간 (compact 설계가 핵심 기여)**
- 관측 o: base의 roll, pitch, 두 축 각속도, 그리고 8 모터각. IMU yaw는 drift로 **제외**, 모터 속도는 noisy해서 제외 → 의도적으로 compact.
- 행동: position control 모드. **leg space** `(s, e)`(swing, extension)로 정의 후 motor space로 매핑:
  - `θ₁ = e + s`, `θ₂ = e − s`
  - leg space를 쓰는 이유: motor space에서는 self-collision으로 valid action이 nonconvex하게 흩어짐 → leg space에서 사각형으로 prune 용이.
- **하이브리드 정책 (full controllability)**: `a(t, o) = ā(t) + π(o)`
  - `ā(t)`: open-loop 주기 신호(사용자 지정 gait reference), `π(o)`: NN feedback(balance 학습). 둘 다 0이면 from-scratch.

**보상 함수**
- `r = (pₙ − pₙ₋₁)·d − w·Δt·|τ·q̇|` (eq.2)
  - 1항: 목표 진행방향 d로의 이동거리(속도), 2항: 에너지(토크×속도) 페널티. `w = 0.008` 고정(전 실험). episode = 1000 step 또는 base tilt > 0.5 rad에서 종료.

**Actuator 모델 (해석적 — 본 논문 핵심)**
1. Bullet position control 제약: `e_{n+1} = k_p(q̄ − q_{n+1}) + k_d(q̄̇ − q̇_{n+1})` (eq.4). 주의: Bullet은 **time step 끝 시점**의 각/속도로 제약을 만족(implicit) → 큰 gain에서 sim은 안정하나 실기는 발산. 이 implicit/explicit 차이가 reality gap의 한 축.
2. 이상 DC 모터 동역학으로 교체:
   - `τ = Kₜ I` (eq.5 토크-전류 선형)
   - `I = (V_pwm − V_emf)/R` (eq.5)
   - `V_emf = Kₜ q̇` (eq.6) — back-EMF
   - `Kₜ`(토크상수), `R`(권선저항)은 actuator 스펙에서.
3. **Piecewise 토크-전류 비선형 보정**: 선형 `τ = KₜI`는 ideal motor에서만 성립, 실제는 전류 증가 시 **토크 포화(saturation)**. 그래서 piecewise-linear 함수를 fitting → sim에서 PWM→전류(eq.5,6) 계산 후 이 piecewise 함수로 토크 lookup. (이 보정 전에는 실기 Minitaur가 발에 주저앉거나 다리를 못 듦.)
4. position control 시 PWM도 PD servo로: `V_pwm = V(k_p(q̄ − qₙ) + k_d(q̄̇ − q̇ₙ))` (eq.7). `V`=배터리 전압. Ghost Robotics MCU의 PD 구현 참고하여 target velocity `q̄̇ = 0` 설정.
- 검증 실험: 모터에 sine 목표궤적 인가, 신규 actuator 모델의 sim 궤적이 실측 ground truth와 일치(Fig.4).

**Latency 모델 (해석적 보간)**
- Bullet은 명령 즉시 반영·센서 즉시 보고(instantaneous) → sim의 feedback 안정영역이 실기보다 훨씬 큼 → sim 정책이 실기서 oscillate/발산.
- 모델링: 관측 이력 `{(tᵢ, Oᵢ)}`(`tᵢ = iΔt`) 유지. 현재 step n에서 컨트롤러가 관측 필요 시, `tᵢ ≤ nΔt − t_latency ≤ t_{i+1}` 인 인접 두 관측 Oᵢ, O_{i+1}을 찾아 **선형 보간**.
- **latency 실측 실험**: 1 step PWM spike를 인가해 모터가 살짝 움직이게 하고, spike 송신 시점 ↔ 모터 움직임 보고 시점의 시간차를 측정. 두 계층 latency 분리 측정 → **MCU PD servo: 3ms(낮음), TX2 locomotion 컨트롤러: 약 15–19ms(높음)**. 이 실측값으로 sim latency 설정.

**URDF 질량/관성 식별 절차**
- Minitaur를 **분해(disassemble)** → 각 링크 치수 측정 + 무게 측정 + center of mass 측정 → URDF에 반영.
- 관성(inertia) 측정은 어려우므로 **각 링크의 형상+질량으로부터 균일밀도(uniform density) 가정하에 추정**. (그래서 관성은 다른 파라미터보다 불확실 → DR에서 50~150% 넓게 randomize.)

**Dynamics randomization (Table I) — 실측 + 안전마진**
| parameter | lower | upper |
|---|---|---|
| mass | 80% | 120% |
| motor friction | 0 Nm | 0.05 Nm |
| inertia | 50% | 150% |
| motor strength | 80% | 120% |
| control step | 3 ms | 20 ms |
| latency | 0 ms | 40 ms |
| battery voltage | 14.0 V | 16.8 V |
| contact friction | 0.5 | 1.25 |
| IMU bias | −0.05 rad | 0.05 rad |
| IMU noise (std) | 0 rad | 0.05 rad |
- mass/motor friction은 system ID로 측정한 conservative 범위. inertia는 균일밀도 추정이라 불확실 → 더 넓게. control step/latency는 비실시간 OS 변동, battery voltage는 충전상태 반영. motor friction은 시스템 ID 실험으로 측정.
- **외란 주입**: 시뮬레이션 매 200 step(=1.2s)마다 base에 perturbation force 주입, 10 step(0.06s) 지속, 방향 random, 크기 **130~220N**. 균형 회복 학습.

**학습 절차 (RL)**
- 알고리즘: **PPO**(on-policy, 병렬화 용이). 정책=2 hidden layer FC NN, 크기는 hyperparam search.
- 매 iteration마다 25 roll-out(각 최대 1000 step) 병렬 수집. 최대 **7M sim step** 도달 시 종료. (Table II: trotting obs 4D / policy net (125,89) / value (89,55) / 4.35h; galloping obs 12D / policy (185,95) / value (95,85) / 3.25h.)

**실기 평가 메트릭**
- reality gap 정량화: success rate(전 episode 1000 step≈6s 균형) 대신 **expected return(eq.2)을 sim/real 양쪽에서 계산한 차이**를 reality gap 지표로 채택(Koos et al.).
- 신뢰도 위해 각 실험당 **100 controller(다른 hyperparam/seed) 학습 → sim return 상위 3개를 실기 배포 → 각 3회 = 9회 평균**을 expected return으로.

**주요 결과 (수치)**
- Galloping: 자동 emergence. sim 1.34 m/s(2.48 body length/s), 실기 1.18 m/s(2.18 BL/s). 다수 seed가 gallop으로 수렴.
- Trotting: open-loop reference로 유도. `s̄(t) = 0.3 sin(4πt)`, `ē(t) = 0.35 sin(4πt) + 2` (한 대각쌍이 같은 곡선, 다른 쌍은 180° 위상). feedback bound는 swing/extension 모두 [−0.25, 0.25] rad. 12D obs: sim 0.50 m/s(0.93 BL/s) / 실기 0.60 m/s(1.11 BL/s). 4D obs 재학습 후 sim-real 일치 향상.
- 에너지(Table III, 학습 vs handcrafted): trot 0.60 m/s @ 71.78W (handcraft 0.56 m/s @ 92.72W), gallop 1.18 m/s @ 188.79W (handcraft 1.21 m/s @ 290.00W). 학습 gait가 **gallop 35% / trot 23% 전력 절감**.
- **Ablation 1 (시뮬레이터 fidelity, Fig.6)**: baseline sim / baseline+perturbation / improved+perturbation 세 그룹. 앞 두 그룹은 실기서 실패(sim-real 막대 차 큼), improved+perturbation만 sim≈real. **즉 actuator 모델+latency 없으면 robust 정책으로도 reality gap 못 넘음.** 본문 명시: "accurate actuator model and latency simulation are both important. Without either of them, the learned controllers do not work on the real robot."
- **Ablation 2 (randomization, Fig.7)**: no-randomization 정책은 body inertia가 default서 벗어나면 return 급락, randomized 정책은 40~160% inertia 전반에 평탄(robust). 단 randomization은 trade-off(평균/peak return 하락) — "not a free meal".
- **Ablation 3 (observation, Fig.8/9)**: large obs(12D)는 sim 성능↑이나 실기 reality gap↑(train obs 분포 sparse → 실기 미경험 관측서 낙상). small obs(4D)+randomization이 실기 최고(상위 3 정책 9회 모두 3m+ trot/균형). compact obs가 overfitting 억제 → 전이 핵심.

### (4) 문제점·한계·개선 여지
- **Direct-drive·저DOF 특수성**: Minitaur는 8 direct-drive 모터의 단순 동역학(gear backlash/series-elastic 거의 없음). 이상 DC 모터식 + piecewise saturation이 잘 맞는 이유. **고기어비/SEA/유압 등 복잡 액추에이터에는 부족**(→ #6이 이 빈틈을 정확히 공략).
- **Piecewise 토크-전류 fitting 절차 불투명**: piecewise 함수의 break point/구간수/측정 프로토콜이 논문에 디테일 부족. 재현성 약함.
- **관성 식별의 근본 부정확성**: 균일밀도 가정 → 실제 분포와 괴리. DR(50~150%)로 덮지만 정밀 평가로는 미흡.
- **task 단순화**: flat ground 위 전진속도 최대화만. 지형/비전/동적 속도·방향 전환 없음(저자 future work로 명시).
- **DR trade-off 인정**: randomization이 conservative gait 유발(optimality↓). 자동 범위 선택 메커니즘 없음 — 수동 안전마진.
- **개선 여지**: (a) actuator를 해석식 + 잔차(residual) 학습 하이브리드로(데이터 효율+OOD), (b) 관성을 진자/CAD 기반 정밀식별, (c) latency를 분포로(현재 0~40ms uniform이나 비실시간 jitter는 더 복잡).

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder, PMSM 에너지 reward, 모터모델 nominal 주입)
- **토크/전류 실측 → 모터모델 nominal 주입**: 본 논문의 `τ = Kₜ I`, `I = (V_pwm − V_emf)/R`, `V_emf = Kₜ q̇`가 우리 PMSM에 거의 직결. 강복님이 토크/전류 실측 곡선을 갖고 있다면 → **piecewise 토크-전류 saturation 함수를 fit하여 IsaacLab actuator로 inject**가 1차 액션. IsaacLab의 `ActuatorBase`/`DCMotor`/custom actuator로 이 lookup을 구현(Implicit/Explicit actuator 의미는 사용자 confirm 필요 — 추정 금지). MJX에서는 actuator gain/forcerange를 piecewise로 직접 줄 수 없으므로 control wrapper에서 토크 후처리(clamp/lookup)로 구현.
- **encoder 기반 q̇ 노이즈/latency**: 본 논문이 모터속도를 noisy하다고 obs서 제외한 점, latency를 관측이력 선형보간으로 모델링한 점은 **우리 encoder 실측 noise/delay를 그대로 DR로 옮길 근거**. encoder latency를 spike test로 실측 → IsaacLab obs history 보간 또는 action delay buffer로 주입. (단, 우리 deploy 제약상 contact sensor obs 추가는 금지 — 기존 메모리 준수.)
- **PMSM 에너지 reward**: 본 논문의 `−w·Δt·|τ·q̇|`(w=0.008)는 mechanical power 페널티. PMSM은 **전기적 손실(I²R copper loss)**이 추가로 크므로, 단순 |τ·q̇| 대신 **전기 입력전력 `V·I` 또는 `|τ·q̇| + I²R`** 형태로 확장하면 실측 전류와 정합도↑. nominal Kₜ/R 주입 시 sim에서 I를 복원 가능하므로 I²R reward 구현 가능. w 튜닝 둔감성(본 논문 보고)은 우리 reward weight 단위 함정 메모리와 함께 신중히(step_dt 이중곱 금지).
- **DR 테이블 채택**: Table I를 우리 로봇 system ID 실측값 + 안전마진으로 재작성. 특히 motor_strength(80~120%), latency(우리 실측±), battery voltage(우리 셀 범위), IMU bias/noise는 즉시 이식 가치.
- **compact observation 교훈**: 우리 정책 obs 차원 팽창 경계(input 팽창 dealbreaker 메모리)와 동일 결론 — large obs는 sim 과적합/실기 reality gap↑. 본 논문은 그 정량 ablation 근거 제공.

### sim 구현 난이도: 중 — 필요 인프라
- **이유**: 해석식 자체는 단순(중하). 그러나 (a) piecewise 토크-전류 곡선을 얻으려면 **실측 dyno/전류센서 데이터**, (b) latency는 **spike test 측정 장비**, (c) IsaacLab custom actuator/MJX control wrapper 구현이 필요. 코어 수정 없이 actuator config + obs/action delay buffer 수준이면 IsaacLab에서 중. MJX는 actuator 표현 제약으로 wrapper 작업이 추가되어 중상으로 올라갈 수 있음.
- **필요 인프라**: 토크/전류 실측 셋업(이미 강복님 어젠다), encoder latency 측정, IsaacLab `ActuatorBase` 파생 클래스(piecewise lookup + back-EMF), DR randomization config(Table I 형식), obs history 보간 또는 action delay 버퍼.

---

## #6. Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model (Lee, Kim, Kim, Park, Lee, Cho, Hwangbo — KAIST, RA-L 2026)

- **확정 링크**: https://arxiv.org/abs/2601.11143 (HTML: https://arxiv.org/html/2601.11143v1, PDF: https://arxiv.org/pdf/2601.11143, IEEE Xplore: ieeexplore.ieee.org/document/11206408). **번호 교정 결과: 2601.11143 정확(의심 불필요). IEEE RA-L Vol.10 No.12 pp.12421–12428, 2026-01-16 게재.**
- **접근 분류**: Analytic/Parametric — 유압 동역학 기반 **해석적(neural 아님) actuator 모델**(force/torque domain 4-계수 식). neural baseline(MLP/LSTM/GRU) 대비 우월성을 명시적으로 비교.
- **우선순위**: 상(★★★★). #5의 direct-drive 한계를 정면 보완 — **복잡·고관성 액추에이터(유압)를 해석식으로 RL에 inject**한 최신 사례. 강복님 PMSM도 "복잡 액추에이터의 해석 모델 + 실측 fitting + nominal 주입" 패턴이 동일하므로 방법론 템플릿으로 직결. 다만 유압 특수 항(impact 보정)은 PMSM과 물리적으로 다름 → 선택적 차용.

### (1) 요약
**300kg 초과·길이 1.8m+ 초중량 유압 4족 로봇**(다리당 3DOF, 관절당 2 실린더 bidirectional pulling, 총 24 실린더, 단일 펌프)에 RL locomotion을 최초로 sim2real 전이한 논문. 핵심은 유압 동역학을 4개 경험계수(k₁~k₄)로 압축한 **해석적 actuator 모델** — 12 액추에이터 토크를 **<1μs**에 예측하여 RL 환경서 고속 처리 가능. neural baseline(MLP/LSTM/GRU)보다 정확도(RMSE)·학습속도(>3배) 모두 우월. **단 20초** 실로봇 locomotion 데이터(1000Hz)로 계수 fitting. Raisim서 PPO 학습(400 env, 20k iter, ~10h). 실기서 0.4/1.0 m/s 명령추종, 무재학습 선회·횡보 성공. position-PID+제안모델 조합만 자연스러운 보행 달성(torque-PID는 1.0m/s서 전복, stiff position-PID는 다리 못 들고 진동).

### (2) Motivation
- 유압 4족은 고출력·고내하중이나 **액추에이터 동역학이 매우 복잡**(밸브/압력/유량/실린더 비선형, 단일펌프 부하 결합, 부하·이력 의존 마찰). 기존 RL sim2real(전기모터/SEA 중심, 예: #5, Hwangbo 2019)이 가정한 단순 actuator식이 안 맞음.
- neural actuator net(Hwangbo식)은 정확하나 (a) OOD 일반화 약함(RL 탐색 중 빈번 낙상→분포 밖 토크 다발), (b) 추론 느림(RL 환경 속도 병목), (c) 격리 단일실린더 측정이 legged엔 비현실적(부하가 configuration/motion history 의존).
- 목표: **해석적이면서(OOD 견고·초고속) 다중액추에이터 결합까지 포착**하는 모델로 초중량 유압 4족의 안정·강건 command-tracking locomotion을 실기 전이.

### (3) Method + 수식/파이프라인 + 학습/실험 방법

**해석적 유압 actuator 모델 (핵심 기여)**
- 선행 단일실린더식을 다중액추에이터로 확장. force domain:
  - `Δf = k₁Δx − k₂ f − k₃ ẋ + k₄ Δx · max(−f·sgn(Δx), 0)` (eq.4)
- torque domain 최종형(스프로킷 반경 R로 변환):
  - `τ_next = k₁R²(q_des − q) + (1 − k₂)τ − k₃R²q̇ + k₄R(q_des − q)·max(−τ·sgn(q_des − q), 0)` (eq.7)
- 항 해석:
  - `k₁Δx` (k₁R²항): 위치의존 lever 효과(g(x)를 상수로 단순화)
  - `k₂f` ((1−k₂)τ): 힘의존 압력강하(선형 근사)
  - `k₃ẋ` (k₃R²q̇): 속도의존 저항(마찰/damping)
  - **k₄ 항: 반대부호 하중 하 impact 보정** — 부하 급반전(RL 탐색 중 잦은 낙상·충돌)에서의 sudden load reversal 포착. 이 항이 유압 특유의 핵심 차별점.
- 계수 k₁~k₄: 실데이터 curve fitting(최적화 방법 상세는 논문 미기재).
- **속도**: 12 액추에이터 토크 예측 **<1μs** → RL 환경서 실시간.

**시스템 식별 절차 (실측)**
- 데이터: RL 학습된 locomotion 정책으로 **실로봇을 단 20초** 구동(정책 action rate 100Hz).
- 센서 기록: **1000Hz**(관절 위치/속도/토크/목표위치). 고주파로 actuator 동역학 정밀 포착.
- 설계 의도: **격리 단일실린더 시험 회피** — "joint load가 configuration·motion history에 의존"하므로, 실제 다중액추에이터 상호작용을 포함한 whole-robot 궤적으로 fitting. (neural baseline은 이런 격리 시나리오가 필요해 legged엔 비현실적이라는 게 논점.)
- 검증(Table III): 0.4/1.0 m/s RMSE 및 |τⱼ|>50N·m 구간 MAPE로 정확도 측정(작은 토크의 MAPE 왜곡 회피 위해 50N·m threshold).

**Latency / URDF·관성**
- 명시적 latency 모델 별도 제시는 약함. low-level position PID가 HW서 1000Hz, RL 정책 100Hz로 동작(주파수 분리). (#5만큼 명시적 latency 보간식은 없음 — 본 논문 한계.)
- URDF/관성 텐서·질량분포 수치 미공개. 로봇 스펙: 질량 >300kg, 길이 >1.8m, 4다리×3DOF(roll/pitch/knee), 총 24 유압실린더(관절당 2개 bidirectional pulling), 단일 펌프. 스프로킷 반경 R 사용하나 수치 미기재.

**RL 학습 절차**
- 알고리즘: **PPO**. actor(MLP, 12D desired joint position 출력) + critic + **concurrent state estimator**(supervised로 동시학습) 구조.
- 관측 (eq.8): `oₜ = (ω, qₜ, q̇ₜ, q_{t-1}^des, cmd)` — base 각속도, 관절 위치/속도, 이전 action(desired position), 속도 명령. **compact observation** 지향(#5와 동일 철학).
- 보상 (Table I/II, global+local):
  - global: command tracking(속도오차 exponential, xy+yaw), base height(h₀ 유지), base motion(vz·roll/pitch 각속도 페널티), torque regularization(‖τ‖² + clipped torque), joint nominal position(‖qₜ−q_nom‖, 정지 시 가중↑), joint vel/accel smoothness.
  - local(per-foot): flight phase 페널티(전족 공중), airtime tuning(stance/swing duration), foot slip(‖v_foot,xy‖² 접촉중), foot clearance(swing중), GRF smoothness.
  - curriculum: factor c_f를 학습 중 증가.
- domain randomization: observation·kinematics randomization으로 외란 견고화(구체 범위 논문 미상세 — 한계).
- 학습 config: **400 병렬 env(flat terrain)**, init state per-env randomize, 정책+estimator 동시학습, 종료조건=발/정강이 외 body 접지. HW: AMD Threadripper 3990X + RTX 3090. **20,000 iter ≈ 10h**, 시뮬레이터 **Raisim**.

**실기 평가 메트릭/결과**
- 시나리오: 0.4 / 1.0 m/s, 실내 평지 ~30m 직진, 선회·횡보(무재학습).
- actuator 정확도(Table III, RMSE / MAPE@|τ|>50): **제안 7.45 / 9.15–10.8, MAPE 4.77–5.72%**; MLP 36.0/47.1–57.7, 20.9–33.2%; LSTM 33.3/41.1–51.2, 19.8–28.7%; GRU 27.4/34.4–42.1, 15.6–24.0%. → 제안이 압도.
- locomotion 정성:
  - torque-PID: 1.0m/s서 actuator lag로 추종실패·전방 전복.
  - stiff position-PID: 안정하나 다리 못 들어 장애·비자연 진동.
  - **position-PID + 제안모델: 부드러운 자연보행, 1.0m/s 안정, 선회·횡보 무재학습 성공**(상한속도는 시험장 크레인 안전 제약).
- 학습속도: 제안 ~10h vs neural baseline 36–48h → **>3배 가속**.

### (4) 문제점·한계·개선 여지
- **유압 특화**: k₄ impact 보정·압력강하 선형근사는 유압 고유. PMSM/전기모터엔 물리적으로 그대로 안 맞음(우리는 back-EMF·copper loss·토크포화가 주). 식 구조는 차용하되 항은 재정의 필요.
- **latency 모델 부재**: #5 대비 명시적 latency 보간이 약함. 비실시간/통신 지연 효과 미흡 — 유압 응답이 본질적으로 느려 상대적으로 덜 치명적일 수 있으나, 정량 분석 없음.
- **DR 범위 비공개**: observation/kinematics randomization 범위 미상세 → 재현·전이 일반화 평가 곤란.
- **평가 협소**: flat terrain·0.4/1.0 m/s·~30m만. 급방향전환/큰 외란/거친 지형 미시험(저자 명시 한계). 장기 내구성(actuator wear)·유압 변동 미평가.
- **단일 20초 데이터 fitting의 커버리지**: 학습된 정책 1회 구동 데이터라 동작 분포 편향 가능. fitting 최적화 방법·계수 신뢰구간 미보고.
- **개선 여지**: (a) 해석식 + residual neural 하이브리드(OOD 견고 + 미모델 잔차 보정), (b) 명시적 latency/펌프 결합 동역학 추가, (c) 다지형·외란 robustness ablation, (d) model-based 해석성과 RL robustness 결합 컨트롤러(저자 future work).

### (5) 우리 적용 (강복님: 토크/전류 실측 + encoder, PMSM 에너지 reward, 모터모델 nominal 주입)
- **방법론 템플릿 직결**: "복잡 액추에이터 → 해석식으로 압축 → 실로봇 짧은 구동 데이터로 계수 fitting → sim에 nominal 주입 → 초고속 RL"의 파이프라인이 강복님 PMSM 어젠다와 1:1 대응. PMSM은 #5의 DC식(`τ=KₜI`, back-EMF, R)에 토크포화 piecewise를 얹는 게 자연스럽고, **fitting/주입/검증 절차(Table III식 RMSE+MAPE@threshold 평가)**는 본 논문을 그대로 차용.
- **whole-robot 짧은 데이터 fitting 채택**: PMSM도 부하가 configuration 의존이면, 격리 dyno만이 아니라 **실로봇 locomotion 20~수십초 @1000Hz 로깅으로 계수 fit**이 유효(본 논문 핵심 교훈). 우리 토크/전류/encoder 동시 로깅 셋업이면 즉시 적용.
- **PMSM 에너지 reward**: 본 논문 reward의 torque regularization(‖τ‖² + clipped torque)을 기반으로, 우리는 **전기적 손실(`V·I` 또는 `I²R`)** 항으로 확장(nominal Kₜ/R로 I 복원). #5의 mechanical power와 결합해 `mech + copper loss`가 PMSM 정합 최적.
- **<1μs 해석식의 가치**: IsaacLab/MJX 대규모 병렬에서 actuator 모델이 병목이 안 되려면 해석식(neural 아님)이 유리 — 본 논문이 정량 입증(>3배 학습가속). 우리도 piecewise lookup + back-EMF 해석식으로 가야 throughput 유지.
- **concurrent state estimator**: 본 논문이 obs서 직접 못 얻는 상태를 estimator로 동시학습 — 우리 parkour AMP estimator 누락 fix 경험(메모리)과 정합. PMSM 토크/속도 추정에도 동일 패턴 적용 가능(disc=explicit, policy=estimator 원칙 유지).
- **차용 시 주의**: k₄ impact 보정·압력강하 항은 PMSM에 직접 이식 금지(유압 특유). 식 골격(`τ_next = f(q_des−q, τ, q̇)` 재귀형)과 fitting·평가 프로토콜만 차용.

### sim 구현 난이도: 중상 — 필요 인프라
- **이유**: 해석식 자체는 가벼우나 (a) 우리 PMSM용으로 항을 **재유도**(back-EMF/copper/saturation 기반)해야 함(유압식 그대로 안 됨), (b) 실로봇 1000Hz 토크/전류/encoder 동시 로깅 인프라, (c) 계수 fitting 파이프라인(curve fit + Table III식 검증), (d) IsaacLab custom actuator 또는 MJX control wrapper로 재귀형 토크식 통합, (e) concurrent estimator 도입 시 알고리즘(rsl_rl) 수정. #5 대비 fitting/검증/estimator 추가로 한 단계 위.
- **필요 인프라**: 고주파 동기 데이터로거(토크/전류/q/q̇/q_des @1000Hz), 계수 fitting 스크립트(+RMSE/MAPE@threshold 평가), IsaacLab `ActuatorBase` 파생(재귀 토크식) 또는 MJX wrapper, PPO+state estimator 런너(rsl_rl), DR config.

---

## 카테고리 B 종합 비교 (참고)

| 항목 | #5 Tan (2018, direct-drive) | #6 Lee (2026, hydraulic) |
|---|---|---|
| actuator 모델 | 이상 DC식 + piecewise 토크포화 | 유압 4계수 해석식(k₁~k₄, impact 보정) |
| neural 대비 | (해당 없음) | MLP/LSTM/GRU 대비 RMSE↓·학습 >3배↑ |
| 시스템 ID | 분해·계량·균일밀도 관성, motor friction 실측 | 실로봇 20초@1000Hz whole-robot fitting |
| latency | 명시적(spike test 3ms/15–19ms, obs 보간) | 약함(주파수 분리만) |
| DR | Table I 상세(실측+마진) | 범위 비공개 |
| obs | compact(4D~12D), 작을수록 전이↑ ablation | compact(eq.8) |
| sim/RL | PyBullet + PPO, 7M step | Raisim + PPO + estimator, 20k iter/10h |
| 실기 | Minitaur, trot 0.6/gallop 1.18 m/s | 300kg+ 유압, 0.4/1.0 m/s, 무재학습 선회·횡보 |
| 우리 차용 | DC식+포화+latency+DR 테이블+compact obs | fitting/주입/평가 프로토콜+해석식 throughput+estimator |

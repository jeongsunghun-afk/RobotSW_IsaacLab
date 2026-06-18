# Task #2 — Latent Motion Prior 2단계 방법 조사

> **질문**: 평지 자연 보행 데이터로 사전학습한 prior를 parkour(지형) 학습에 재사용하는 방법은?
> **핵심 제약**: ① policy observation 입력 확장(input팽창) = dealbreaker, ② Go2 실배포(proprioception only, 12-DOF), ③ parkour env 재작성 금지, ④ total_reward clip(min=0)
> 작성: lit-priors / 2026-06-11 / 증거등급 [LIT]=문헌확인, [HYP]=추론

---

## 0. 핵심 판정 프레임: "두 종류의 prior 재사용"

평지 데이터로 학습한 prior를 지형 task에 쓰는 방식은 메커니즘상 **두 부류**로 갈리며, 우리 제약(input팽창 금지 + 단일 deploy policy) 충족 여부가 정반대다.

| 부류 | 구조 | 지형 정보 위치 | deploy policy 입력 | input팽창? |
|---|---|---|---|---|
| **(A) Hierarchical latent-action** (Tencent, ASE, CALM, MCP) | high-level RL이 latent z 출력 → **frozen low-level decoder**가 z→action | high-level **policy 입력**에 들어감 | high-level이 보는 obs (proprio+terrain) | **조건부 판정**(아래 §6) |
| **(B) Frozen prior = 보상/감독 신호** (GMP, DeepMimic류 tracking) | prior가 reference trajectory 생성 → **reward로만** 사용, policy는 그대로 joint target 출력 | reward 계산부 (policy 외부) | 기존 proprio (불변) | **No** (단, reference를 obs에 넣으면 Yes) |

> **결론 선출**: 우리 제약에는 **(B) 부류(prior를 reward로만)가 압도적으로 안전**하다. (A) 부류는 "action space가 latent로 바뀌는" 구조라 deploy 파이프라인(RMA+DAGGER recurrent)과 충돌 가능성이 크다(§6).

---

## 1. Tencent "Lifelike Agility and Play" (PMC/EPMC/SEPMC) — 가장 직접적 선행연구 [LIT]

- **출처**: Han, Lei et al., *"Lifelike agility and play in quadrupedal robots using reinforcement learning and generative pre-trained models"*, **Nature Machine Intelligence 6(7), 2024**. arXiv:2308.15143. 코드 공개(github.com/Tencent-RoboticsX/lifelike-agility-and-play). [LIT 실존확인]

### 3단계 구조와 freeze 전략 [LIT]
1. **Primitive Motion Controller (PMC)** — **평지 동물 mocap(Labrador retriever, 약 30분, 120fps: 걷기/뛰기/점프/앉기/계단)** 으로 사전학습. 구조는 **VQ-VAE 기반 VQ-PMC**: encoder가 (proprioception + 미래 trajectory target)→latent, **vector quantization**으로 K개 discrete embedding 중 선택, decoder가 모터명령 생성. latent = "discrete motor primitive". (저자: discrete 표현이 Gaussian보다 표현력 높음)
2. **Environmental-level PMC (EPMC)** — **PMC를 freeze**하고 그 위에 학습. 지형정보(height map 25×13 + depth 25×13 + 거리 ray)는 **EPMC(상위) 입력**으로 들어감. EPMC 출력 = **PMC latent embedding에 대한 categorical 분포** → frozen decoder 구동. 각 장애물(creeping/계단/허들/블록)을 개별 학습 후 **distillation으로 단일 EPMC 통합**.
3. **Strategic-level (SEPMC)** — PMC+EPMC 모두 freeze, 멀티에이전트 tag 게임용 상위 navigation 명령만 학습.

### 우리 질문 관점 [LIT/HYP]
- **prior가 평지에서 캡처하는 것**: 동물스러운 모터 primitive(자연 gait의 시간적 협응) — 정확히 우리가 부족한 것. [LIT]
- **지형 일반화 메커니즘**: 평지 primitive를 frozen으로 두고, **상위가 지형 맥락에 맞춰 primitive를 "호출/조합"** → 지형 모션 데이터 없이도 자연스러운 통과가 emergent. [LIT]
- **freeze vs fine-tune**: 엄격한 freeze(하위 절대 안 건드림). [LIT]
- **input팽창 판정**: 지형정보는 *상위* 입력 → 우리가 단일 deploy policy로 chain하면 "policy 입력에 지형"은 우리 parkour와 동일(이미 height scan 있음). 단, **action space가 discrete latent로 바뀜** = 구조적 재작성. [HYP]
- **실로봇**: MAX robot(14kg) zero-shot 전이 성공(계단/creeping/허들/블록 + 게임). [LIT]
- **한계(우리 적용)**: ① env/policy 구조를 3단계 hierarchy로 재설계 필요 → "env 재작성 금지"와 직접 충돌. ② RMA+DAGGER recurrent + height-scan teacher 구조를 hierarchy로 갈아끼우는 비용 큼. ③ VQ discrete action은 우리 continuous PPO action head와 비호환.

> **시사점**: 개념적으로 우리 질문의 정답에 가장 가깝지만(평지 mocap→지형 emergent, 실로봇 검증), **구조 채택 비용이 제약(env 재작성 금지)과 정면충돌**. → 통째 채택보다 "primitive를 reward/regularizer로 빌리는" 경량화가 현실적(§5).

---

## 2. ASE / CALM — adversarial skill embedding을 평지 데이터로 학습 후 downstream [LIT]

- **ASE**: Peng, X.B. et al., *"ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters"*, **SIGGRAPH/ACM TOG 41(4), 2022**, arXiv:2205.01906. [LIT]
- **CALM**: Tessler, C. et al., *"CALM: Conditional Adversarial Latent Models for Directable Virtual Characters"*, **SIGGRAPH 2023**. [LIT]

### 구조 [LIT]
- 1단계: mocap으로 **low-level policy π(a|s,z)** + skill encoder를 adversarial imitation(AMP류 discriminator) + unsupervised RL diversity로 사전학습. latent **z = 연속 skill 임베딩**.
- 2단계 downstream: **high-level policy가 z를 출력**, frozen low-level이 z→action. CALM은 추가로 motion encoder로 z를 직접 명령(directable).

### 우리 질문 관점 [LIT/HYP]
- **prior 캡처**: 다양한 skill의 연속 매니폴드(z 하나가 하나의 동작 스타일). [LIT]
- **지형 일반화**: high-level이 task보상(지형 통과) 받으며 z 시퀀스를 탐색. [LIT]
- **freeze**: low-level freeze. [LIT]
- **input팽창 판정**: §1과 동일 — **action space가 latent z로 치환**되는 게 본질. policy 입력 자체는 안 늘 수 있으나 우리 deploy(단일 continuous-action recurrent)와 구조 비호환. [HYP]
- **실로봇**: ASE/CALM 모두 **시뮬레이션 캐릭터 애니메이션**(실로봇 전이 없음). sim-to-real gap·proprio-only 제약 미검증 = 우리 Go2 배포 직접근거 약함. [LIT]
- **한계**: graphics 도메인(관측·구동 풍부), 12-DOF proprio-only Go2와 거리 큼. adversarial latent prior는 mode collapse·z 탐색 불안정 보고 다수. [LIT/HYP]

> **시사점**: ASE/CALM의 핵심 차용가치는 "구조"가 아니라 **AMP discriminator로 평지 스타일을 prior로 만드는 아이디어**. 이건 Task #1(AMP) 영역과 합류. 우리에겐 hierarchy 없이 discriminator만 reward로 쓰는 경로가 더 맞음.

---

## 3. FLD (Fourier Latent Dynamics) — 주기모션 구조적 latent [LIT]

- **출처**: Li, Chenhao; Stanger-Jones, E.; Heim, S.; Kim, Sangbae, *"FLD: Fourier Latent Dynamics for Structured Motion Representation and Learning"*, **ICLR 2024 (spotlight)**, arXiv:2402.13820. 코드 공개(mit-biomimetics/fld). [LIT]

### 구조 [LIT]
- 주기/준주기 모션을 **주파수 영역**에서 연속 파라미터화(PAE 계열 phase 표현 확장). latent dynamics가 모션을 연속 매니폴드로 표현 → **interpolation·미관측 타깃 일반화**.
- 적용: **MIT Humanoid** motion tracking(Isaac Gym). 학습 중 미관측 타깃까지 online tracking.

### 우리 질문 관점 [LIT/HYP]
- **prior 캡처**: 평지 주기보행의 **위상·주파수 구조** → trot/pace 같은 gait의 시간 일관성. [LIT]
- **지형 일반화 가능성**: FLD latent는 "주기모션 매니폴드"라 **비주기적 지형 통과(점프·등반)에는 표현력 부족**. 저자도 periodic/quasi-periodic 가정 명시. [LIT] parkour는 본질적으로 aperiodic 이벤트 多 → 직접 일반화 약함. [HYP]
- **사용형태**: tracking reward의 reference 생성기로 쓰임 → **(B)부류(보상신호)에 가깝게 운용 가능**. [LIT/HYP]
- **input팽창 판정**: tracking이라 target latent/phase를 obs에 넣는 게 통상 → 넣으면 input팽창, **reward로만 쓰면 회피 가능**. [HYP]
- **실로봇**: 논문 자체는 sim(MIT Humanoid). 단 phase/PAE 계열은 사족보행 sim-to-real 사례 존재(별도). [LIT]

> **시사점**: FLD는 "자연스러운 평지 gait의 위상구조"를 싸게 prior화하는 도구로 매력적이나, **aperiodic parkour 일반화가 약점**. 평지 gait regularizer 용도로 부분 차용 가치는 있음.

---

## 4. MCP — hierarchical low-level prior 재사용(곱셈적 합성) [LIT]

- **출처**: Peng, X.B.; Chang, M.; Zhang, G.; Abbeel, P.; Levine, S., *"MCP: Learning Composable Hierarchical Control with Multiplicative Compositional Policies"*, **NeurIPS 2019**, arXiv:1905.09808. [LIT]

### 구조 [LIT]
- 1단계: motion imitation 사전학습으로 다수 **primitive** 추출.
- 2단계: high-level이 primitive들의 **가중치(게이팅)** 출력, 여러 primitive를 **곱셈적으로 동시 활성**해 합성 → 새 task(드리블·운반) 해결.

### 우리 질문 관점 [LIT/HYP]
- **prior 캡처**: 재사용·재조합 가능한 모터 primitive 집합. [LIT]
- **지형 일반화**: high-level이 지형보상 받으며 primitive 합성. [LIT]
- **input팽창/구조**: §1·§2와 동일한 hierarchical latent-action 부류 → **action space가 게이팅으로 바뀜**, 우리 deploy 비호환. [HYP]
- **실로봇**: sim 캐릭터(실로봇 없음). [LIT]
- **한계**: 2019년 그래픽스 세팅, proprio-only 사족 미검증. Tencent가 사실상 이 계보의 로봇·실배포 후속.

> **시사점**: MCP는 Tencent 계보의 원형. 독립 채택 이유 약함 — Tencent를 대표로 보면 됨.

---

## 5. 최신(2024–2026) 로봇 motion-prior 사례 — **(B)부류가 주류로 수렴** [LIT]

### 5-1. GMP: Natural Humanoid Locomotion with Generative Motion Prior [LIT]
- arXiv:2503.09015, **IROS 2025**. 프로젝트: sites.google.com/view/humanoid-gmp. [LIT]
- **구조**: human mocap retarget → **CVAE를 offline 학습 → policy 학습 중 frozen online generator**로 미래 reference trajectory 생성 → **motion guidance reward**(joint angle·keypoint)로 dense supervision. policy는 **joint target 직접 출력**(latent 아님), PD 변환. sim+real 검증.
- **input팽창 판정**: ⚠️ **GMP 원본은 reference m_{t+1}을 obs에 추가** → state = (proprio, command, reference). **그대로면 input팽창 발생**. 단 **reference를 obs에서 빼고 reward로만 쓰면 회피 가능**(우리 변형 필요). [LIT for 원본, HYP for 변형]
- **시사점**: "frozen prior = reward, policy 구조 불변"이 핵심. action space·네트워크는 그대로라 **RMA+DAGGER와 충돌 없음**. obs 추가만 제거하면 우리 제약 부합.

### 5-2. T-GMP: Terrain-conditioned Generative Motion Priors [LIT]
- arXiv:2606.06944 (2026). *"Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion"*. [LIT 실존확인, 내용 미정독]
- **의의**: GMP를 **지형 조건부**로 확장 = "평지 prior를 지형에 맞춰 호출"의 최신 직계 후보. 우리 질문과 가장 시기적으로 근접. (정독 권장 — Task 후속)

### 5-3. Parkour in the Wild (multi-expert distill + RL fine-tune) [LIT]
- arXiv:2505.11164 (2025). terrain별 expert 학습 → **DAgger로 단일 foundation policy distill → 실스캔 지형서 RL fine-tune**. [LIT]
- **의의**: motion prior는 아니지만 **"사전학습 정책 재사용→지형 적응"의 2단계 패턴이 우리 RMA+DAGGER와 동형**. 즉 우리 파이프라인 자체가 이미 distill+finetune 친화적임을 시사. [HYP]

### 5-4. Quadruped Parkour via Terrain-Conditional AMP [LIT]
- MDPI Applied Sciences 16(7):3448. *"Parkour Learning for Quadrupeds via Terrain-Conditional Adversarial Motion Priors"*. [LIT 실존확인] → **Task #1(AMP)과 합류**. 사족+지형+AMP 조합 실존 증거.

### 5-5. Humanoid Parkour Learning [LIT]
- arXiv:2406.10759 (2024). 휴머노이드 parkour, **모션 reference 없이** RL(우리 baseline과 동일 철학). prior 미사용 대조군.

---

## 6. "2단계 latent 구조"가 우리 single-stage RMA+DAGGER와 결합 가능한가 [HYP]

우리 deploy 파이프라인: **teacher(height-scan)→PPO→DAGGER로 proprio-only recurrent student distill, action=12 joint target**.

- **(A) Hierarchical latent-action 채택 시**(Tencent/ASE/CALM/MCP):
  - high-level RL의 action space가 **latent z**가 됨. deploy 시 student는 "z→frozen decoder→joint"를 chain해야 함.
  - **충돌점 1**: DAGGER는 student가 teacher action을 모방 — teacher action이 z면 student도 z를 출력해야 하고, frozen decoder까지 deploy에 묶임 → **현 single-head continuous policy 구조 재작성**. "env/구조 재작성 금지"와 충돌.
  - **충돌점 2**: VQ discrete(Tencent)는 continuous PPO와 비호환; ASE/CALM latent는 adversarial 학습 불안정.
  - **판정**: **비권장**(구조비용↑, 제약충돌, 실로봇 사족 직접근거는 Tencent만).

- **(B) Frozen prior = reward 채택 시**(GMP류 / FLD-as-reward / AMP discriminator):
  - policy **action space·네트워크·obs(reference 미추가 시) 전부 불변**. prior는 reward 계산에만 개입.
  - **RMA+DAGGER 완전 호환**: teacher reward에 "평지 prior 부합도" 항 추가 → PPO 학습 → 기존 DAGGER 그대로.
  - **주의 — clip(min=0) 상호작용**: prior 보상은 **양수 bonus**로 설계해야 total_reward clip(min=0)에 안 먹힘. penalty(음수)로 넣으면 clip floor에 흡수돼 무력화 위험(메모리 `feedback_parkour_reward_weight_units` 교훈과 동일 함정).
  - **판정**: **권장**. 단 reference/latent를 obs에 넣는 순간 input팽창 → **반드시 reward-only로 운용**.

---

## 7. 우리 질문에 대한 시사점 Top 3

1. **prior는 "action space"가 아니라 "reward"로 빌려라 (B부류 채택).** Tencent/ASE/CALM/MCP의 hierarchical latent-action은 개념적으로는 정답에 가깝지만 **action space를 latent로 바꿔 우리 RMA+DAGGER·continuous policy·"env 재작성 금지"와 정면충돌**. 반면 **GMP(IROS2025)·DeepMimic류 tracking·AMP discriminator**처럼 frozen 평지 prior를 **reward 신호로만** 쓰면 policy 입력·구조·action 전부 불변 → 제약 4개 모두 충족. [HYP, 근거 §6]

2. **input팽창의 진짜 위험은 "reference를 obs에 넣는 것"이다.** GMP 원본조차 reference를 state에 추가(input팽창). 우리는 **prior를 obs에서 빼고 reward 계산부에만** 두는 변형이 필수. 이 한 가지 설계결정이 "평지 prior 재사용"을 dealbreaker로 만들지/안 만들지를 가른다. [LIT(원본)+HYP(변형)]

3. **가장 직접적 선행연구(Tencent)는 "구조"보다 "교훈"을 빌려라 + 최신 직계는 T-GMP.** Tencent는 평지 동물mocap→지형 emergent→실로봇(MAX)을 **end-to-end로 입증한 유일 사례**라 reward 설계의 타깃("자연 primitive 부합도")을 정당화하는 근거로 인용가치 최상. 다만 통째 hierarchy 채택은 비용·제약충돌로 비권장. **지형조건부로 가장 근접한 T-GMP(arXiv:2606.06944, 2026)는 정독 후속 권장**. [LIT]

---

## 부록 — 인용 실존 확인표

| # | 논문 | ID/출처 | 확인 |
|---|---|---|---|
| 1 | Lifelike Agility and Play (Tencent) | arXiv:2308.15143 / Nature Mach. Intell. 6(7) 2024 | [LIT] ✔ 코드공개 |
| 2 | ASE | arXiv:2205.01906 / SIGGRAPH TOG 41(4) 2022 | [LIT] ✔ |
| 3 | CALM | SIGGRAPH 2023, Tessler et al. | [LIT] ✔ |
| 4 | FLD | arXiv:2402.13820 / ICLR 2024 spotlight | [LIT] ✔ 코드공개 |
| 5 | MCP | arXiv:1905.09808 / NeurIPS 2019 | [LIT] ✔ |
| 6 | GMP | arXiv:2503.09015 / IROS 2025 | [LIT] ✔ |
| 7 | T-GMP | arXiv:2606.06944 (2026) | [LIT] ✔ 실존, 내용 미정독 |
| 8 | Parkour in the Wild | arXiv:2505.11164 (2025) | [LIT] ✔ |
| 9 | Quadruped Parkour Terrain-Conditional AMP | MDPI Appl.Sci. 16(7):3448 | [LIT] ✔ → Task#1 |
| 10 | Humanoid Parkour Learning | arXiv:2406.10759 (2024) | [LIT] ✔ 대조군 |

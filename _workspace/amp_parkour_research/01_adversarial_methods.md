# Task #1 — AMP 계열 Adversarial Imitation의 지형/불완전 데이터 확장 조사

**작성**: lit-adversarial (team: amp-parkour-research)
**날짜**: 2026-06-11
**핵심 질문**: *"평지 자연 보행 mocap/모션 데이터만으로 parkour(험지) 학습에 style/imitation 신호를 어떻게 넣을 것인가?"*

**증거등급 범례**: `[LIT]` = WebSearch/WebFetch로 문헌 실존·내용 확인됨 / `[HYP]` = 본 작성자의 추론 / `[미확인]` = 실존/내용 검증 실패

---

## 0. 결론 먼저 (TL;DR)

- 평지 demo → 험지 task로 style 신호를 전이하는 것은 **실증된 방향**이다. 핵심 4갈래:
  1. **Discriminator feature를 terrain-invariant하게 좁히기** (root-relative joint-space, base height 제외/완화) — AMP의 핵심 설계 자유도. `[LIT]`
  2. **Wasserstein discriminator (WGAN-GP) + soft boundary** — 평지 분포와 험지 행동의 불일치(distribution mismatch)·partial demo에 더 강건. WASABI/HumanMimic. `[LIT]`
  3. **Multi-critic로 style reward를 task reward와 분리** — clip(min=0) 충돌을 회피하는 우리 제약에 가장 적합. RobotKeyframing. `[LIT]`
  4. **Terrain-conditioning** (discriminator/policy에 terrain feature 주입) — 단, policy 입력 팽창 dealbreaker와 충돌 가능 (discriminator-only면 허용). `[LIT]`
- 우리 제약에서 가장 안전한 조합: **discriminator-only amp_obs (입력팽창 없음) + WGAN soft-boundary + multi-critic 분리** (자세한 Top 3은 §8).

---

## 1. WASABI — Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations

- **출처** `[LIT]`: Li, Günther, Geyer, Martius 외. CoRL 2022. arXiv:**2206.11693**. 코드: github.com/martius-lab/wasabi. 로봇: **Solo 8** (8-DOF quadruped), cross-platform ANYmal 전이도 보고.
- **우리 질문과의 직접 관련성**: 제목 그대로 *rough + partial* demo(사람이 손으로 들고 흔든, 관절 정보 없고 물리적으로 로봇과 비호환인 시연)에서 reward를 뽑는다. "데이터가 task와 정확히 맞지 않는다"는 우리 상황(평지 demo vs 험지 행동)과 동형(同型).

### 핵심 메커니즘 `[LIT]`
- **Discriminator feature가 base-level만** 사용: base linear velocity `v`, base angular velocity `ω`, gravity projection `g` (base frame), base height `z`. **관절 정보 불필요**. → 시연이 부분적/비호환이어도 공통으로 관측 가능한 저차원 신호만 매칭.
- **여러 timestep(H=2,4,8,16) concatenation**으로 horizon을 늘려 mode collapse 방지.
- **Wasserstein GAN (WGAN-GP)** 사용. AMP의 기본 LSGAN 대비:
  - LSGAN: `min_D  E_M[(D(o,o')−1)²] + E_π[(D(Φ(s),Φ(s'))+1)²]`
  - WASABI: `min_D  −E_M[D(o,o')] + E_π[D(Φ(s),Φ(s'))]` (+ gradient penalty)
  - 이유: LSGAN 출력은 ±1 대칭이라 **거리 정보(얼마나 먼가)를 잃어** "처음엔 도달 불가능해 보이는" 동적 모션에 reward 신호가 빈약. WGAN은 earth-mover's distance 근사 → **더 정보량 많은 reward + vanishing gradient 회피**.
- **Reward 정규화**: `r^I = (D(Φ(s^H)) − μ̂) / σ̂` (running mean/std 정규화).

### 실로봇 검증 `[LIT]`
- SoloLeap / SoloWave / SoloBackFlip 3개 task가 sim→real 전이 성공 (DTW 거리로 평가). SoloStandUp은 hip abduction 부재로 실패.

### 우리 제약 하 적합성 / 한계
- **적합 `[HYP]`**: discriminator가 base-level만 보므로 amp_obs를 proprioception 안에서 구성 가능 → **deploy 제약(proprioception only) 호환**, policy 입력 팽창 없음.
- **clip(min=0) 충돌 주의 `[HYP]`**: WGAN reward는 **부호가 없고(음수 가능)** 정규화 후 0 평균. 우리 total_reward clip(min=0)에 그대로 더하면 음수 style이 잘려 buggy. → **별도 critic 분리 필수**(§6) 또는 reward shift(+offset) 필요.
- **한계 `[LIT]`**: 시연 비호환 정도(degree of incompatibility)에 대한 robustness는 저자들도 정밀 분석 안 함 → 평지↔험지 gap이 너무 크면 보장 없음. 평가 metric(DTW) task별 튜닝 필요.

---

## 2. Escontrela et al. 2022 — AMP Make Good Substitutes for Complex Reward Functions

- **출처** `[LIT]`: Escontrela, Peng, Yu, Zhang, Iscen, Goldberg, Abbeel. IROS 2022. arXiv:**2203.15103**. 로봇: **Unitree A1**.
- **데이터** `[LIT]`: **German Shepherd mocap 수 초** 분량. (평지 보행 위주; 프로젝트 페이지에서 험지 mocap 언급 없음.)
- **핵심 주장** `[LIT]`: 손으로 짠 수십 항 reward 없이도 style reward(=discriminator) + 단순 task reward(목표 속도)만으로 자연스러운 gait·에너지 효율·자연스러운 gait transition을 얻고 **sim-to-real 성공**.
- **지형 일반화 범위 `[HYP/미확인]`**: 공개 페이지에서 A1을 **험지/계단/갭에서 검증했다는 명시 근거를 찾지 못함**. 주로 평지/자연 보행 품질·CoT 개선이 강조됨. → "AMP가 험지까지 자동 일반화한다"는 주장은 **이 논문 근거로는 단정 불가 [미확인]**. 우리 질문(험지 전이)에 직접 답을 주는 논문은 아니고, **"평지 mocap만으로 좋은 style 대체 reward가 된다"**는 토대를 제공.
- **시사점 `[HYP]`**: 우리가 가진 "평지 자연 보행 데이터"가 style reward source로 충분하다는 1차 근거. 단, 험지 전이는 §1/§5/§7의 추가 장치 필요.

---

## 3. Multiple AMP / 스킬별 Discriminator 변형

### 3-1. Advanced Skills through Multiple Adversarial Motion Priors `[LIT]`
- **출처**: Vollenweider et al. arXiv:**2203.14912**. 로봇: **wheeled-legged ANYmal (16-DOF)**.
- **메커니즘** `[LIT]`: skill마다 별도 discriminator `D^i` + 별도 motion/rollout buffer. 한 번에 하나의 style만 활성(one-hot command). `r_t = r_t^task + r_t^style`.
- **Discriminator feature(50-D)** `[LIT]`: base lin/ang vel, gravity dir(base frame), base height, joint pos/vel, wheel position(base-frame). → §5 feature 선택의 표준 set.
- **Mode collapse 해법** `[LIT]`: 여러 prior가 같은 task를 cover하면 policy가 더 쉬운 style로 쏠리거나 hybrid가 됨 → **discrete one-hot style switching**으로 해결.
- **지형 `[LIT/미확인]`**: rough terrain 학습을 robustness 차원에서 언급하나 상세 부족.

### 3-2. Conditional AMP for Multi-Skill (CAMP) `[LIT]`
- **출처**: "Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors". arXiv:**2509.21810**. 로봇: **Unitree Go2 (우리와 동일 플랫폼!)**.
- **메커니즘** `[LIT]`: skill embedding을 discriminator에 conditioning(conditional discriminator) + 별도 skill discriminator(cosine similarity reward)로 gait type 정렬. mode collapse 방지.
- **데이터** `[LIT]`: **평지만**. mocap 대신 **model-based controller로 trot/pace/bound/pronk를 평지에서 생성**(retargeting 불필요). → 우리처럼 "평지 데이터로 다양한 스킬"을 만드는 패턴의 직접 선례.
- **검증** `[LIT]`: Go2에 파라미터 튜닝 없이 직접 전이, ~91% joint tracking.
- **한계** `[LIT]`: 평지·합성 데이터 한정 → 복잡 실지형 적용성은 저자도 미해결로 명시.
- **우리 시사점 `[HYP]`**: pronk/split-jump 문제를 "원하는 gait를 conditional하게 명령"하는 식으로 우회 가능. 단 skill embedding을 policy에 주면 입력팽창 → discriminator-only conditioning이면 회피 가능.

---

## 4. HumanMimic / Wasserstein Adversarial Imitation — soft boundary가 분포 불일치에 강건한가?

- **출처** `[LIT]`: "HumanMimic: Learning Natural Locomotion and Transitions for Humanoid Robot via Wasserstein Adversarial Imitation". arXiv:**2309.14225**. ICRA 2024.
- **핵심 메커니즘** `[LIT]`: Wasserstein-1 distance(IPM) + **novel soft boundary constraint**로 discriminator 학습 안정화·mode collapse 방지. unified primitive-skeleton retargeting으로 형태 차이 흡수.
- **분포 불일치 강건성에 대한 직접 증거** `[LIT]`: **transition 모션이 데이터에 없어도** 속도 변화에 따라 자연스러운 pattern 간 전이가 **emergent**하게 나타남 → 즉 "데이터에 없는 영역으로의 보간/외삽"을 soft-boundary WGAN이 더 잘 다룬다는 증거. 우리 질문("평지 데이터엔 없는 험지 행동")에 **간접적이지만 강한 시사**.
- **왜 soft boundary가 LSGAN보다 분포 불일치에 강한가 `[LIT+HYP]`**:
  - `[LIT]` WGAN은 두 분포가 겹치지 않아도(=disjoint support, 평지 vs 험지가 정확히 그 케이스) gradient가 살아있음(LSGAN/JS는 saturate).
  - `[LIT]` soft boundary는 critic 출력을 hard clamp하지 않고 완만히 제약 → 학습 초기 큰 분포차에서 발산/collapse 억제.
  - `[HYP]` 따라서 "평지 demo와 초기 험지 행동의 거리가 큰" 우리 상황에서 표준 AMP(LSGAN)보다 reward 신호가 끊기지 않을 가능성이 높음.
- **우리 제약 하 적합성 `[HYP]`**: WGAN critic 출력은 unbounded·부호 있음 → §1과 동일하게 **clip(min=0)과 직접 충돌**. multi-critic 분리 또는 reward 정규화+offset 필수.

---

## 5. Discriminator Feature 선택 전략 — terrain-invariant style 신호 만들기

**이 섹션이 우리 질문의 기술적 심장부.** "discriminator가 root height/global pose를 못 보고 joint-space relative feature만 보면 지형 변화에 invariant한 style이 되는가?" → **본질적으로 Yes, 설계로 제어 가능.** `[LIT+HYP]`

### 표준 AMP feature set (Peng et al. 2021, arXiv:2104.02180) `[LIT]`
- root local lin/ang velocity (character frame), 각 joint local rotation, joint local velocity, end-effector 3D 위치(character frame). **task-specific feature 없음** → motion prior가 task annotation 없이 학습됨(설계 의도).

### legged_gym 계열 표준 amp_obs `[LIT]`
- joint pos, joint vel, foot/leg end position(FK, base-frame), base frame lin/ang vel, **base height**.

### Feature 선택이 지형 invariance를 결정하는 원리 `[LIT+HYP]`
- **base height `z`를 빼거나 약화** `[HYP]`: 험지에서는 절대/상대 base height가 지형에 따라 변동 → discriminator가 z를 보면 "평지에서의 z 분포"를 강요해 험지 적응을 방해. z 제외 시 지형 invariant.
  - 근거 `[LIT]`: WASABI는 z를 포함하지만 base-level만 쓰는 **저차원** 구성으로 partial/rough demo를 다룸 → feature를 좁힐수록 비호환 데이터에 강해진다는 방향성 확인.
- **root-relative / body-frame joint-space만 매칭** `[LIT+HYP]`: joint pos/vel은 본질적으로 지형과 무관한 "어떻게 다리를 움직이나(coordination)" 신호 → 평지에서 배운 다리 협응 style을 험지에서도 강제 가능. global pose(world position/heading) 제외는 표준 AMP 설계에 이미 내장(local frame).
- **trade-off `[HYP]`**: feature를 너무 좁히면(예: joint vel만) discriminator가 pronk vs trot을 구분 못 해 style 신호가 무력화. 너무 넓히면(z, contact 포함) 험지에서 over-constrain. → **joint pos/vel + body-frame foot pos + body-frame lin/ang vel, base height는 제외 또는 저가중** 이 우리 sweet spot 후보.

### 구현 비교(legged_gym AMP 계열) `[LIT]`
| 구현/논문 | amp_obs 구성 | base height | 지형 적용 |
|---|---|---|---|
| Peng 2021 (AMP) | joint rot/vel + EE pos + root vel (local) | 없음(캐릭터) | 평지/물리캐릭터 |
| Escontrela 2022 (A1) | joint pos/vel + EE + base vel + base height | 포함 | 주로 평지 |
| Multi-AMP (ANYmal) | +wheel pos, 50-D | 포함 | rough(언급) |
| WASABI (Solo8) | base vel/ω/g/z만 (관절X) | 포함 but 저차원 | agile skill |
| CAMP (Go2) | joint pos/vel+base vel+base height+foot, 43-D | 포함 | 평지만 |

→ **결론 `[HYP]`**: "joint-space relative만 보면 terrain-invariant"는 **부분적으로 맞고 설계로 달성 가능**하나, 문헌에서 대부분 base height를 포함한다. **우리 가설(z 제외)은 합리적이나 직접 ablation한 논문은 못 찾음 [미확인] → 우리가 A/B로 검증할 가치 있는 novel 포인트.**

---

## 6. Style vs Task Reward 균형 — multi-critic, scheduling, constraint 결합

### 6-1. RobotKeyframing (multi-critic) `[LIT]`
- **출처**: Zargarbashi, Cheng, Kang, Sumner, Coros (ETH Zürich). CoRL 2024 (PMLR v270). arXiv:**2407.11562**.
- **메커니즘** `[LIT]`: dense reward(추종)와 sparse reward(keyframe 도달)의 혼합을 **multi-critic RL**로 처리. 각 reward group마다 **별도 critic(별도 value head/advantage)** → 서로 다른 scale·sparsity의 reward를 **합쳐서 하나의 scalar로 만들지 않음**.
- **검증** `[LIT]`: sim + hardware. multi-critic이 single-critic 대비 hyperparameter 튜닝 부담을 크게 줄임.
- **우리 제약에 결정적 `[HYP]`**: 우리 total_reward는 **clip(min=0)**. AMP style(부호 있는 WGAN/LSGAN reward)을 여기 더하면 음수가 잘려 망가짐. **multi-critic으로 style을 별도 critic으로 분리하면 clip(min=0)을 거치지 않아 충돌 자체가 소멸.** → 우리 codebase의 가장 깔끔한 통합 경로.

### 6-2. Terrain 난이도 기반 style weight scheduling `[HYP]`
- 문헌에서 "terrain difficulty → style weight curriculum"을 명시한 논문은 **이번 조사에서 직접 확인 못 함 [미확인]**.
- `[HYP]` 합리적 설계: 쉬운 지형(평지·낮은 단)에서 style weight 높게(평지 demo 분포와 가까움) → 어려운 지형(높은 step/gap)에서 style weight 낮춰 task가 지배. terrain_levels(curriculum)에 weight를 연동. 단 multi-critic을 쓰면 weight scheduling 없이도 advantage 정규화로 일부 자동 균형.

### 6-3. Constraint 기반 결합 (CaT/CAT) `[HYP/메모리]`
- 메모리상 CaT(IROS24, constraints-as-terminations)·KAIST IPO가 3-leg fix 후보로 검토됨 `[프로젝트 메모]`. AMP style을 reward가 아닌 **constraint(분포 일탈 시 페널티/termination)**로 거는 변형은 clip(min=0)과 충돌을 줄이는 또 다른 길 `[HYP]`. 단 본 task 범위 밖이라 인용만.

---

## 7. 2024–2026 최신: AMP를 quadruped 험지/parkour에 실제 적용한 사례

### 7-1. Terrain-Conditional AMP for Quadruped Parkour `[LIT]` ★직접 선례
- **출처**: "Parkour Learning for Quadrupeds via Terrain-Conditional Adversarial Motion Priors". MDPI Applied Sciences 16(7):3448, 2026-04-02.
- **메커니즘** `[LIT]`: discriminator와 policy에 **explicit terrain observation을 conditioning** → terrain-aware motion. teacher(privileged terrain height) → student(forward depth) **distillation** (우리 RMA/DAGGER 구조와 동일 계열).
- **결과** `[LIT]`: platform/gap/stair/slope/debris에서 **pure AMP·pure RL 대비 빠른 수렴 + 높은 success rate**.
- **우리 제약 충돌 `[HYP]`**: terrain feature를 **policy에 conditioning = 입력 팽창 dealbreaker와 충돌**. 단 **discriminator에만 terrain conditioning**하면 policy 입력 불변 유지 가능(team-lead 지적대로 amp_obs는 policy 입력 아님). → "discriminator-only terrain-conditional AMP"가 우리 호환 변형.

### 7-2. Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility `[LIT]` ★우리 질문과 가장 동일
- **출처**: arXiv:**2505.16084** (2025). 제목이 곧 우리 질문.
- **메커니즘** `[LIT]`: **평지 demo에서 배운 motion prior를 style 신호로 사용**, terrain 정보는 **policy state에 conditioning**해 평지 prior를 환경에 맞게 **modulate**(새 primitive를 처음부터 배우지 않음). discriminator는 task 성능이 아닌 **style**에 집중. style은 학습 내내 유지하되 험지 적응은 task reward가 주도(adaptive weighting).
- **검증** `[LIT]`: 실로봇에서 slope·obstacle 포함 다양 지형 전이 성공, motion prior 없는 baseline 대비 traversal 효율·자연스러움 향상.
- **우리 시사점 `[HYP]`**: "평지 데이터 → 험지 task"가 **실로봇으로 작동함을 직접 증명한 가장 가까운 논문.** terrain conditioning이 policy에 들어가는 부분만 우리 제약(입력팽창)과 협의 필요 → discriminator-side로 옮기거나, terrain은 이미 우리 height-scan/depth로 들어오는 채널 재사용.

### 7-3. 기타 확인된 최신 `[LIT]`
- **T-GMP**: "Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion" arXiv:**2606.06944** (team #5가 정독 — OOD 해법 관점). humanoid지만 terrain-conditioned generative prior의 OOD 처리 참고.
- **PUMA**: arXiv:2601.15995 (foothold prior, parkour) — AMP 아닌 foothold prior 계열. 메모리상 preprint 격리 대상.
- **BCAMP**: MDPI Appl.Sci. 15(6):3356 — behavior-controllable AMP (Go2). controllability 참고.
- DreamWaQ류 + AMP 직접 결합 명시 사례는 **이번 조사에서 확정 못 함 [미확인]**.

---

## 8. 우리 질문에 대한 시사점 — Top 3

### Top 1. Discriminator-only amp_obs + terrain-invariant feature 설계 (입력팽창 0, deploy 호환) `[LIT+HYP]`
- amp_obs를 **proprioception 안의 joint pos/vel + body-frame foot pos + body-frame lin/ang vel**로 구성하고 **base height z는 제외 또는 저가중**.
- 근거: WASABI(base-level 저차원으로 partial demo 처리) + AMP feature 설계 자유도 + Multi-AMP/CAMP의 표준 set.
- **policy 입력 불변** → team-lead가 명시한 "discriminator 전용 amp_obs는 policy 입력이 아니므로 입력팽창 dealbreaker 해당 없음" 을 100% 활용. deploy(proprioception only)도 만족(amp_obs는 학습 시에만 discriminator로).
- **novelty/검증포인트**: z 제외가 terrain-invariance를 실제로 개선하는지 직접 ablation한 문헌을 못 찾음 → 우리 A/B 실험의 기여 포인트.

### Top 2. WGAN/soft-boundary discriminator + multi-critic 분리 (clip(min=0) 충돌 원천 제거) `[LIT+HYP]`
- discriminator를 **LSGAN이 아닌 WGAN-GP(WASABI) 또는 soft-boundary Wasserstein(HumanMimic)**으로 → 평지 demo와 험지 행동의 **disjoint distribution**에서도 gradient 유지.
- style reward를 **별도 critic(RobotKeyframing multi-critic)**으로 분리 → 부호 있는 style reward가 우리 **total_reward clip(min=0)을 거치지 않음** → 충돌 소멸. (대안: reward에 +offset/정규화 후 합산, 단 깨지기 쉬움.)
- 이 조합이 우리 두 제약(clip, deploy)을 동시에 푸는 가장 견고한 경로.

### Top 3. "평지 prior를 terrain이 modulate" 패러다임 채택 — 단 conditioning은 discriminator-side로 `[LIT+HYP]`
- 직접 선례 2건: **Motion Priors Reimagined(2505.16084)**, **Terrain-Conditional AMP parkour(MDPI 2026)** 가 "평지/일반 데이터 → 험지 parkour"를 실증.
- 다만 두 논문은 terrain feature를 **policy**에 conditioning → 우리 입력팽창 dealbreaker와 충돌.
- **우리 변형 `[HYP]`**: terrain conditioning을 **discriminator에만** 적용(terrain별로 "자연스러움 기준"을 다르게) + policy는 기존 height-scan/depth 채널 그대로(추가 입력 0). 이러면 평지 style을 험지 난이도에 맞춰 완화하면서 policy 인터페이스 불변.
- 보너스: pronk/split-jump 회피를 위해 **CAMP식 gait-conditional discriminator**(평지 trot/walk를 명령)로 "원하는 보행 style"을 직접 지정 가능(skill embedding도 discriminator-side로).

---

## 부록 A. 인용 실존 확인 표

| # | 논문 | ID/출처 | 확인 |
|---|---|---|---|
| 1 | WASABI | arXiv:2206.11693, CoRL 2022, martius-lab/wasabi | `[LIT]` abstract+ar5iv 본문 |
| 2 | AMP Good Substitutes | arXiv:2203.15103, IROS 2022 | `[LIT]` 검색+프로젝트페이지 (험지검증 [미확인]) |
| 3 | Multiple AMP | arXiv:2203.14912 | `[LIT]` ar5iv 본문 |
| 3b| Conditional AMP (CAMP) | arXiv:2509.21810, Go2 | `[LIT]` html 본문 |
| 4 | HumanMimic (WGAN soft boundary) | arXiv:2309.14225, ICRA 2024 | `[LIT]` html 본문 |
| 5 | AMP 원논문 (feature set) | arXiv:2104.02180, Peng 2021 | `[LIT]` 검색 |
| 6 | RobotKeyframing (multi-critic) | arXiv:2407.11562, CoRL 2024, ETH | `[LIT]` 검색+OpenReview |
| 7a| Terrain-Conditional AMP parkour | MDPI Appl.Sci.16(7):3448, 2026 | `[LIT]` 검색 요약 |
| 7b| Motion Priors Reimagined | arXiv:2505.16084 | `[LIT]` pdf 본문 |
| 7c| T-GMP | arXiv:2606.06944 | `[LIT]` 검색 (team#5 정독) |
| 7d| PUMA / BCAMP | arXiv:2601.15995 / MDPI 15(6):3356 | `[LIT]` 검색 (preprint/참고) |

## 부록 B. 미확인/추가 검증 필요 항목
- `[미확인]` Escontrela A1의 명시적 **험지** 검증 여부 (평지 중심으로 보임).
- `[미확인]` "base height 제외가 terrain-invariance를 개선"하는 직접 ablation 문헌 → **우리 실험 기여 가능 영역**.
- `[미확인]` terrain difficulty 연동 style-weight scheduling 명시 논문.
- `[미확인]` DreamWaQ + AMP 직접 결합 사례.

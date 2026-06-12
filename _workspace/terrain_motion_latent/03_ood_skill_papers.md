# 조사C: OOD task의 latent skill 확장 / RL fine-tuning 선행연구

> 담당: paper-c (Task #3)
> 목적: 제안 알고리즘 **②-2**(지형/모션 AE의 latent 위에서 RL 학습, 데이터 없는 OOD task=계단/gap/crawl을 RL feedback으로 latent embedding 공간에 "새로운 skill"로 mapping)의 이론적·실증적 성립 근거 점검.
> 핵심 질문: **"RL feedback만으로 학습된 latent skill 공간에 새로운 skill을 추가하는 것"이 선행연구에서 (a) 직접 가능한가, (b) prior 재학습이 필요한가, (c) residual이 필요한가?**
>
> 표기 규칙: 직접 fetch/search로 확인한 내용과 미확인 내용을 구분. 기관/arXiv ID 중 이번 세션에서 직접 검증 못 한 항목은 "(미확인)" 표기.

---

## 요약 결론 (먼저)

선행연구 종합 시 ②-2의 핵심 가정 — **"고정된 latent skill 공간 위에서 high-level policy를 RL로 학습하면 OOD task가 자동으로 새 skill로 매핑된다"** — 는 **부분적으로만 성립**한다.

- **성립하는 경우**: OOD task가 기존 latent의 **보간/조합(interpolation/recombination)**으로 표현 가능할 때. (ASE/CALM/PULSE의 downstream RL이 이 방식. terrain traversal 같은 task도 "기존 walk/run/jump skill의 재배열"이면 frozen latent + high-level RL로 해결됨.)
- **실패하는 경우**: OOD task가 latent **coverage 밖의 새 동역학**을 요구할 때(예: 데이터에 없는 가파른 등반, crawl 자세). PULSE는 이를 명시적으로 보고함("VR tracking 등 일부는 latent 사용 시 from-scratch보다 성능 저하"). 이때는 **(b) prior 재학습** 또는 **(c) residual** 이 필요.
- **실무적 정답(2024~2025 quadruped 선행연구의 수렴점)**: 순수 latent re-mapping이 아니라 **frozen motion prior + high-level residual** 구조. 특히 우리 과제와 가장 가까운 ANYmal 논문(CoRL 2025)이 이 구조로 평지 prior를 복잡 지형으로 확장하는 데 성공. → **②-2는 "residual을 추가한 형태"로 재설계해야 안전.**

상세 근거는 아래 논문별 정리 참조.

---

## 1. Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility ★최우선 관련

- **저자/기관**: (미확인 — 실로봇 ANYmal-D 사용, ETH/ANYbotics 생태계로 추정)
- **Venue/연도**: CoRL 2025 (PMLR 305:3762–3777)
- **arXiv**: 2505.16084 / 프로젝트: https://anymalprior.github.io/
- **확인 경로**: WebSearch + abstract/프로젝트 페이지 fetch (세부 아키텍처는 본문 PDF 미정독 = 일부 미확인)

**핵심 아이디어 (확인됨)**
- 2단 hierarchical RL. (1) low-level policy를 **평지에서 동물 모션(walk/pace/canter) 모방**으로 사전학습 → motion prior 확립. (2) **goal-conditioned high-level policy**가 이 prior 위에 **residual correction**을 학습 → perceptive locomotion, 국소 장애물 회피, goal-directed navigation을 복잡·험지에서 수행.
- 시뮬레이션: residual이 점진적으로 어려워지는 험지에 적응하면서도 **prior가 부여한 locomotion 특성(자연스러움)을 보존**. 실로봇(ANYmal-D): 평지 동물 모션 skill을 복잡 지형으로 일반화, 부드럽고 효율적 보행 확인.

**②-2와의 관련성**: **직접적 지지 + 설계 수정 근거(가장 중요한 레퍼런스).**
- 우리 과제(평지 walk/run/trot/jump prior → 계단/gap 등 OOD 지형)와 **거의 동일한 문제 설정·로봇 클래스(quadruped)**.
- 단, "latent에 새 skill을 mapping"이 아니라 **action/behavior 공간의 residual correction**으로 OOD를 해결. 즉 선행연구의 실증적 정답은 ②-2의 순수 latent 방식이 아니라 **(c) residual** 쪽.

**차용 가능 요소**
- "평지 prior 동결 + high-level이 perception(height) 받아 residual" 구조를 그대로 골격으로 채택.
- residual이 prior 특성을 보존하도록 하는 motion regularization(자연스러움 유지) 항.

**한계**
- (확인) 논문이 보이는 OOD는 "험지 보행/장애물 회피" 수준 — crawl·대규모 자세 변경 같은 **극단적 OOD skill 창발 여부는 본문 미정독으로 미확인**.
- residual의 적용 공간(action vs latent vs state embedding)이 abstract/페이지에 미명시 → **본문 PDF 정독 필요(후속 작업)**.

---

## 2. PULSE: Universal Humanoid Motion Representations for Physics-Based Control ★메커니즘 직접 근거

- **저자/기관**: Zhengyi Luo 외 (CMU / NVIDIA / Reality Labs 계열, 일부 미확인)
- **Venue/연도**: ICLR 2024 (spotlight)
- **arXiv**: 2310.04582 / 코드: github.com/ZhengyiLuo/PULSE
- **확인 경로**: HTML(v2) 직접 fetch — downstream 수식·실패 모드까지 확인

**핵심 아이디어 (확인됨)**
- AMASS 전체 모터 skill을 **variational information bottleneck(VAE encoder-decoder)** 으로 32차원 확률적 latent에 distill. coverage 99.8%.
- **proprioception에 조건화된 learnable prior** R(z|s^p)를 함께 학습(고정 Gaussian이 아님).
- Downstream task: high-level policy π_task(z_t | s^p, s^g)가 latent를 출력 → **frozen decoder D가 새 동역학으로 작동**. 최종 action `a_t = D(π_task(z) + μ^p)` — 즉 high-level 출력이 **prior 평균 μ^p에 더해져** decoding됨(latent 공간 내 modulation).
- KL 정규화 `D_KL(E(z|·)‖R(z|s^p))`는 **PULSE 사전학습 단계**에서 latent을 prior에 붙임.

**②-2와의 관련성**: **메커니즘 지지 + OOD 실패 모드 직접 증거.**
- ②-2가 말하는 "latent 위 high-level RL로 새 task 수행"의 **표준 구현체**. 특히 latent을 **prior mean에 더하는(=residual-in-latent)** 형태라는 점이 핵심 — 순수 "새 latent 좌표로 점프"가 아니라 prior 근방 modulation.
- 동시에 **반례적 경고**: 저자가 "π_PULSE는 100% imitation 성공률 미달", "VR tracking 등 일부 motion class는 latent 사용 시 from-scratch보다 성능 저하"라고 명시 → **latent coverage 밖 OOD는 latent만으로 표현 불가**. 이게 ②-2의 가장 큰 위험.

**차용 가능 요소**
- **proprioception-conditioned learnable prior**(고정 Gaussian 대신) — OOD에서 prior가 상태에 따라 유연.
- downstream RL을 **latent-residual(z + μ^p) 형태**로 구성하는 패턴.
- 사전학습 KL로 latent 공간을 조밀·연속적으로 만들어 OOD 보간 가능성 확보.

**한계**
- coverage 밖 task는 latent 우회로 인해 오히려 손해 → ②-2처럼 "데이터 없는 OOD"를 노릴 때 정면으로 부딪히는 한계.
- humanoid·MoCap 도메인. quadruped·height-paired 데이터 전이는 우리가 직접 검증 필요.

---

## 3. ASE: Large-Scale Reusable Adversarial Skill Embeddings (foundational)

- **저자/기관**: Xue Bin Peng 외 (NVIDIA / UC Berkeley / Univ. of Toronto, 일부 미확인)
- **Venue/연도**: SIGGRAPH 2022
- **arXiv**: 2205.01906 (이번 세션 직접 미확인 — ID는 일반 지식 기반)
- **확인 경로**: WebSearch 스니펫 (PADL/PULSE 비교 맥락)

**핵심 아이디어 (확인됨/일반지식)**
- GAIL + mutual-information(skill latent ↔ trajectory) 목적으로 **재사용 가능한 adversarial skill embedding(low-level)** 학습. 이후 **high-level controller를 downstream task에서 RL로 학습**(frozen low-level 위).
- ②-2가 차용하려는 "latent skill 공간 + 그 위 downstream RL" 패러다임의 **원조**. → 2022년이지만 **반드시 포함**: ②-2 구조 자체가 ASE 계보이며, 이후 모든 후속(CALM/PULSE/Motion Priors Reimagined)이 ASE의 한계(coverage·directability)를 보완하는 흐름이라 맥락 파악에 필수.

**②-2와의 관련성**: **패러다임 지지(원형 제공).** 단 ASE의 downstream RL도 **기존 skill의 조합**으로 task를 풀 뿐, latent coverage 밖 새 skill을 RL이 "발명"한다는 주장은 아님 → ②-2의 강한 가정은 ASE가 보증하지 않음.

**차용 가능 요소**: latent-conditioned low-level + high-level task RL의 2단 구조; MI 기반 latent 다양성 확보.

**한계**: latent 공간이 **데이터 분포에 갇힘**(directability·OOD 약함) → CALM/PULSE가 이를 개선한 이유. ASE 단독으로 OOD skill 확장 근거로 쓰면 과대해석.

---

## 4. CALM: Conditional Adversarial Latent Models for Directable Virtual Characters

- **저자/기관**: Chen Tessler 외 (NVIDIA / Technion, 일부 미확인)
- **Venue/연도**: SIGGRAPH 2023
- **arXiv**: 2305.02195 (검색 결과로 확인)
- **확인 경로**: WebSearch 스니펫

**핵심 아이디어 (확인됨)**
- raw MoCap을 latent로 encoding하고, latent → physically-simulated skill로 decoding하는 **motion-conditioned policy**를 conditional imitation 목적으로 학습. **directable interface** 제공(원하는 모션으로 조향).
- 여러 클립을 통합해 다기능 agent(locomotion·acrobatics·martial arts) 구성.

**②-2와의 관련성**: **부분 지지.** latent을 **조향 가능(directable)** 하게 만들어 high-level이 latent을 명령처럼 쓰는 구조 — ②-2의 "latent에 task 신호 주입"과 정합. 단 CALM도 **데이터 내 skill의 조합·조향**이지 데이터 밖 skill 생성은 아님.

**차용 가능 요소**: conditional encoder로 latent을 "지형/모션 명령"에 조건화하는 설계 → 우리의 (지형 height, 모션) paired 데이터와 자연스럽게 결합 가능.

**한계**: 데이터 coverage 밖 OOD에 대한 보증 없음. SIGGRAPH(애니메이션) 도메인 — sim-to-real·perception 미고려.

---

## 5. Latent Action Priors for Locomotion with Deep RL

- **저자/기관**: Hausdörfer 외 (기관 미확인)
- **Venue/연도**: 2024 (arXiv preprint; venue 미확인)
- **arXiv**: 2410.03246
- **확인 경로**: WebSearch 스니펫 (abstract 수준)

**핵심 아이디어 (확인됨)**
- **단 1개의 open-loop gait cycle** 시연에서 simple autoencoder로 **action 공간의 latent prior**를 추출 → DRL 가이드. latent 값이 [-1,1]에 머물도록 보조 loss.
- 수동 style reward(imitation)와 latent action prior를 결합. **agent가 시연의 reward 수준에 갇히지 않고**, transfer task에서 성능이 크게 향상.
- Mujoco 벤치마크 + **Unitree A1(quadruped)·H1(humanoid)** 평가.

**②-2와의 관련성**: **지지(특히 "데이터 밖으로 나갈 수 있다"는 점).** latent prior가 **탐색을 가이드하되 reward를 통해 시연 너머로 일반화**됨을 보임 → ②-2의 "prior 활용 + RL feedback으로 변형/확장" 직관과 정합. 소량 시연으로도 prior 구성 가능 → 우리 jump 모션처럼 데이터 적은 skill에 유용.

**차용 가능 요소**: 소량/단일 사이클로 latent prior 부트스트랩; latent 범위 제약 loss; **prior가 RL의 탐색 bias로만 작동하고 reward가 OOD 확장을 견인**하는 구도.

**한계**: action 공간 latent(저수준)이라 ASE/PULSE급 고수준 skill 추상화는 아님. venue·정량 일반화 범위 abstract 수준만 확인(미확인 부분 존재).

---

## 6. AdaptNet: Policy Adaptation for Physics-Based Character Control

- **저자/기관**: (미확인 — NVIDIA 계열 추정)
- **Venue/연도**: SIGGRAPH Asia 2023 (연도 확인, venue 추정)
- **arXiv**: 2310.00239
- **확인 경로**: abstract fetch

**핵심 아이디어 (확인됨)**
- 사전학습된 RL controller "위에" 2단 적응: (1) **state embedding(내부 latent) 증강**으로 소폭 행동 변화, (2) **policy network layer 변경**으로 대폭 변환. 새 locomotion style·task target·morphology·**"환경의 광범위한 변화(extensive changes in environment)"** 에 적응. from-scratch 대비 학습시간 대폭 단축.

**②-2와의 관련성**: **지지(어댑터/residual 경로의 직접 사례).** "내부 latent에 주입 + residual layer"라는 **(a)latent 주입 + (c)residual의 혼합**으로 OOD(환경 변화 포함)에 적응. ②-2를 "순수 frozen latent RL" 대신 **어댑터 기반**으로 구현할 때의 청사진.

**차용 가능 요소**: state-embedding 주입(소폭) + residual layer(대폭)의 **2-tier 적응** — OOD 정도에 따라 적응 강도 조절. 평지 prior 동결 + 지형용 어댑터.

**한계**: abstract만 확인(실패 모드 미명시). 애니메이션 도메인. 적응은 여전히 **추가 학습(어댑터 파라미터)** 을 요구 → "RL feedback만으로 즉시 새 skill"이 아님을 시사.

---

## 7. 데이터 확장 계열: Self-Imitation / Iterative Dataset Expansion

- **대표 연구(확인됨, 개념 수준)**:
  - **Collect-and-Infer 패러다임**: 성공 rollout을 학습셋에 재투입 → 확장된 데이터로 새 모델 재학습(iterative improvement).
  - **SART (Self-Augmented Robot Trajectory)**: 단일 시연에서 안전 경계 내 자율 augmentation으로 데이터 확장. arXiv 2509.09893.
  - **SEIL (Self-Evolved Imitation Learning)**: 시뮬 상호작용으로 성공 trajectory를 새 demo로 수집해 few-shot 모델 점진 개선. arXiv 2509.19460.
  - **Self-Imitation by Planning / Collect-and-Infer**: improve→add→imitate 루프.
- **확인 경로**: WebSearch 스니펫 (manipulation 중심, quadruped parkour 직접 사례는 미확인)

**②-2와의 관련성**: **결정적 보완책 — "OOD를 ID로 끌어들이기".**
- ②-2의 약점(데이터 없는 OOD)을 정면으로 푸는 정통 경로: RL로 얻은 **OOD 성공 trajectory를 데이터셋에 추가 → prior를 재학습/확장**. 이는 위 질문 (b)"prior 재학습"의 구체적 절차.
- 우리 메모리의 amp_parkour 리서치(`self-imitation demo + 품질 게이트`)와도 일치.

**차용 가능 요소**: 계단/gap에서 RL이 만든 성공 모션 중 **품질 게이트 통과분만** 데이터에 추가 → AE/prior 재학습 → 다음 라운드 latent coverage 확장(반복). "improve→add→retrain" 루프.

**한계**: 품질 게이트 없으면 저품질 모션이 prior를 오염(메모리 경고와 동일). manipulation 사례라 quadruped 모션 품질 기준은 우리가 정의해야 함.

---

## 8. (선택) Forward-Backward Representation / Behavioral Foundation Model — zero-shot 새 task

- **대표 연구(확인됨, 개념 수준)**:
  - **BFM-Zero**: reward-free 상호작용 + unlabeled offline 데이터로 compact representation 학습 → **zero-shot prompting으로 다양한 downstream task**. arXiv 2511.04131 (2025, humanoid).
  - **DVFB (Dual-Value Forward-Backward)**: zero-shot 일반화 + fine-tuning 적응 동시. ICLR 2025 (openreview 0QnKnt411O). FB의 **탐색 부족 → 데이터 다양성 부족 → 일반화 제약** 문제를 분석.
- **확인 경로**: WebSearch 스니펫

**②-2와의 관련성**: **대안 패러다임(약한 지지).**
- FB는 occupancy measure를 forward/backward로 분해해 **test-time에 임의 reward를 zero-shot으로** 해결 → "새 task = latent/representation 공간의 새 좌표"라는 ②-2 직관의 **이론적 사촌**.
- 단 DVFB가 지적하듯 **탐색이 빈약하면 OOD task 일반화가 제약** → ②-2도 동일 위험(latent을 채운 데이터가 평지뿐이면 지형 OOD를 못 메움).

**차용 가능 요소**: zero-shot 후 **fine-tuning 적응**의 2-phase 구도(DVFB) — frozen latent로 일단 풀고, 안 되는 OOD만 적응.

**한계**: FB/URL은 주로 reward-free·상태 기반. 모션 품질(자연스러움) 보장 메커니즘이 약함 → 우리 "자연스러운 parkour" 목표와 직접 맞지 않음. 2025 신생 분야로 quadruped parkour 실증 미확인.

---

## 조사 종합: ②-2 설계가 성립하는 조건과 실패 모드

### Q. "RL feedback만으로 latent에 새 skill을 추가"가 선행연구에서 어떻게 다뤄지는가?

**(a) 직접 가능 — 단, 조건부.**
- OOD task가 **기존 latent의 보간·조합으로 표현 가능**할 때만, frozen latent + high-level RL로 해결됨(ASE/CALM/PULSE downstream의 표준 결과).
- 즉 "계단·gap"이 본질적으로 **walk/run/trot/jump의 시공간적 재배열**이라면 ②-2의 순수 형태가 작동할 가능성 있음. (우리 jump 데이터 보유가 여기서 강점.)
- 반례 경고: PULSE는 coverage 밖 motion class에서 **latent이 오히려 from-scratch보다 손해**라고 직접 보고. → "데이터에 전혀 없는 동역학(예: crawl 자세, 매우 가파른 등반)"은 RL feedback만으로 latent에 안정적으로 안 생김.

**(b) prior 재학습이 필요한 경우 — OOD가 coverage 밖일 때의 정통 해법.**
- Self-imitation / iterative dataset expansion(7번): RL로 만든 **OOD 성공 trajectory를 데이터에 추가 → AE/prior 재학습**으로 OOD를 ID로 흡수. 품질 게이트 필수.
- 이것이 "데이터 없는 OOD"를 정면으로 해결하는 가장 신뢰도 높은 경로.

**(c) residual이 필요한 경우 — 2024~2025 quadruped 선행연구의 실증적 수렴점.**
- **Motion Priors Reimagined(ANYmal, CoRL 2025, 1번)**: 우리와 동일 설정에서 **평지 prior 동결 + high-level residual**로 복잡 지형 확장 성공. 순수 latent re-mapping이 아니라 residual.
- **PULSE(2번)**: downstream을 `z + μ^p`의 **latent-residual** 형태로 구현.
- **AdaptNet(6번)**: state-embedding 주입 + residual layer의 2-tier 어댑터로 환경 변화 적응.

### 권고: ②-2의 안전한 재설계 방향

1. **순수 "latent re-mapping"으로 가지 말 것.** 선행연구 어디에서도 "frozen latent + RL feedback만으로 coverage 밖 새 skill이 안정적으로 창발"한다는 강한 증거는 없음. (ASE/CALM/PULSE 모두 "기존 skill 조합" 수준, PULSE는 OOD 손해를 명시.)
2. **②-2 = "latent prior + residual"로 구성.** 1순위 레퍼런스는 ANYmal 논문: 평지 walk/run/trot/jump prior 동결 → 지형(height)-conditioned high-level이 **latent-residual(PULSE식 z+μ) 또는 action-residual(ANYmal식)** 을 출력. residual 크기에 motion regularization을 걸어 자연스러움 보존.
3. **coverage 밖 극단 OOD(crawl 등)는 (b) 데이터 확장 루프 병행.** RL 성공 모션을 품질 게이트로 거른 뒤 (지형,모션) 데이터에 추가 → AE/prior 재학습 → latent coverage를 단계적으로 확장(self-imitation, 7번). 우리 메모리의 amp_parkour self-imitation 처방과 일치.
4. **learnable, 상태(proprio/terrain)-conditioned prior** 채택(PULSE) — 고정 Gaussian prior보다 OOD 적응 여지가 큼. (지형 height, 모션) paired 데이터가 conditional prior 학습에 그대로 활용 가능(CALM식 conditioning).
5. **latent 공간 정규화(KL, 범위 제약)** 로 OOD에서도 latent이 prior 근방에 머물게 해 모션 붕괴 방지(SPiRL/PULSE/Latent Action Priors의 공통 처방). 단 residual/어댑터가 prior 밖으로 나갈 통로는 열어둘 것.

### 실패 모드 체크리스트 (학습 시 모니터링)
- **latent 붕괴/이탈**: high-level이 prior coverage 밖 latent을 쏘면 decoder가 비정상 모션 출력(PULSE 보고). → KL/범위 제약 + residual clip.
- **prior 오염**: 데이터 확장 루프에서 품질 낮은 OOD 모션을 무게검증 없이 추가하면 평지 prior까지 망가짐(메모리 경고). → 품질 게이트 필수.
- **탐색 빈약 → OOD 미충원**: 데이터가 평지뿐이면 지형 OOD를 latent이 못 메움(DVFB 분석). → terrain curriculum + intrinsic/exploration 신호.
- **residual이 prior를 덮어씀**: residual에 정규화 없으면 prior 특성(자연스러움) 상실 → "from-scratch와 다름없음". ANYmal식 motion regularization 필요.

---

## 참고 링크 (markdown)

- Motion Priors Reimagined (CoRL 2025): https://arxiv.org/abs/2505.16084 · https://anymalprior.github.io/
- PULSE (ICLR 2024): https://arxiv.org/abs/2310.04582 · https://github.com/ZhengyiLuo/PULSE
- ASE (SIGGRAPH 2022): https://arxiv.org/abs/2205.01906 *(arXiv ID 직접 미검증)*
- CALM (SIGGRAPH 2023): https://arxiv.org/pdf/2305.02195
- Latent Action Priors (2024): https://arxiv.org/abs/2410.03246
- AdaptNet (SIGGRAPH Asia 2023): https://arxiv.org/abs/2310.00239
- SART (2025): https://arxiv.org/abs/2509.09893 · SEIL (2025): https://arxiv.org/html/2509.19460v1
- BFM-Zero (2025): https://arxiv.org/pdf/2511.04131 · DVFB (ICLR 2025): https://openreview.net/forum?id=0QnKnt411O
- (참고) SPiRL / Accelerating RL with Learned Skill Priors: https://arxiv.org/pdf/2010.11944

> 후속 권고(미완): 1번(ANYmal)·6번(AdaptNet)은 residual 적용 공간(action vs latent vs state-embedding)이 ②-2 최종 설계에 직결되므로 **본문 PDF 정독으로 확정 필요**. 이번 조사는 abstract/HTML/검색 스니펫 기반이라 해당 세부는 "미확인" 표기.

# 조사 A — Latent Motion Prior / Steerable Imitation 계열 논문 조사

> 작성: paper-a (팀 go2-latent-parkour) · 2026-06-12
> 대상 연구: Unitree Go2 사족보행 로봇의 parkour(지형 극복)를 **자연스러운 모션**으로 수행하는 정책 학습.
> 사용자 제안: ① 지형 AE + 모션 AE + (지형,모션) joint AE → ②-1 AE로 reference 생성 후 IL, ②-2 AE latent를 RL state/action으로 사용해 OOD task(계단/gap/crawl)를 새 skill로 latent에 mapping.

본 문서는 **motion latent space(AE/VAE/VQ-VAE/conditional AE) 기반 모방학습 제어기** 분야를 조사한다. Anchor 논문을 정밀 요약한 뒤, latent skill embedding 계열(ASE/CALM/PULSE/MaskedMimic 등)과 그 2024+ 후속, 특히 **사족보행에 latent motion prior를 적용한 최신 연구**를 우선 정리한다.

---

## 0. 한눈에 보는 비교표

| # | 논문 | 기관/venue/연도 | latent 구조 | conditioning | 로봇/캐릭터 | 사용자 설정 관련성 |
|---|------|----------------|-------------|--------------|------------|------------------|
| 1 (anchor) | Walk like Dogs (Steerable Imitation from Unstructured Animal Motions) | RAI Institute / arXiv 2507.00677 / 2025 | **hyperspherical VAE (vMF), 18-D, MoE decoder** | 이전 state + latent; 상위 RL은 속도/turn command | **Unitree Go2 (실기)** | ★★★★★ Go2 + 비라벨 모션 + steerable + latent+RL 2단 |
| 2 | Motion Priors Reimagined (Flat→Complex Quadruped) | ETH RSL (Hutter) / arXiv 2505.16084 / 2025 | **VAE latent (평지 모션)** | 상위 RL이 지형 feature로 latent 출력 | ANYmal C | ★★★★★ 평지 prior→계단/gap RL 적응 = ②-2 그 자체 |
| 3 | PULSE (Universal Humanoid Motion Representations) | Meta Reality Labs + CMU / ICLR 2024 (2310.04582) | **conditional VAE + learnable prior, 32-D, action space** | proprioception-conditioned prior; downstream RL은 prior mean에 residual | 휴머노이드(sim) | ★★★★ learnable prior + residual RL 설계가 ②-2 핵심 청사진 |
| 4 | MaskedMimic (Unified Control via Masked Inpainting) | NVIDIA / SIGGRAPH Asia·TOG 2024 (2409.14393) | conditional latent (VAE-style) + masked inpainting | 부분 pose/joystick/path/**terrain**/text(멀티모달 마스킹) | 휴머노이드(sim), 다양 지형 | ★★★★ "부분조건→완성모션" 생성 = ②-1 reference 생성 관점 |
| 5 | Lifelike Agility and Play | Tencent Robotics X / Nature Machine Intelligence 2024 (2308.15143) | **VQ-VAE 이산 primitive (VQ-PMC)** | 환경레벨이 이산 embedding 선택, 미래 trajectory conditioning | MAX 사족로봇(실기) | ★★★★★ 이산 latent prior→새 task(계단/jump/creep) RL = 3계층 = ②-2 |
| ctx | ASE / CALM / C·ASE | NVIDIA·UC Berkeley / SIGGRAPH 2022–23 | adversarial skill embedding (구면 latent) | latent(skill) z, (C·ASE는 이산 condition) | 휴머노이드(sim) | ★★★ latent skill space의 시조, 비교 baseline |

> ★ = 사용자 설정과의 관련성(5점). 상세는 각 절 참조.

---

## 1. [Anchor] Walk like Dogs: Learning Steerable Imitation Controllers for Legged Robots from Unlabeled Motion Data

- **저자/기관**: Dongho Kang 등, Robotics and AI Institute (RAI Institute, ex-Boston Dynamics AI Institute)
- **venue/연도**: arXiv 2507.00677 (v1 제목 "Learning Steerable Imitation Controllers from Unstructured Animal Motions"), 2025
- **링크**: https://arxiv.org/abs/2507.00677 · html: https://arxiv.org/html/2507.00677v2 · video: https://www.youtube.com/watch?v=DukyUGNYf5A

### 핵심 아이디어 — 3단 파이프라인
**비라벨(unlabeled) 개 모션 캡처** 데이터에서, 별도의 mode 라벨/개수/스위칭 규칙 없이 **steerable(속도·방향 명령으로 조종 가능)** 사족 보행기를 학습. 명령 속도에 따라 pace→trot→gallop **gait 전환이 emergent**하게 나타난다.

1. **Kino-dynamic retargeting (Stage 1)**: 원시 동물 모션 → 로봇 호환 데이터로 변환.
   - Kinematics: base height/roll/pitch에 scaling factor 적용(예 αz=0.81), **constrained IK**로 base pose 오차 + swing-foot 오차 최소화. 제약: stance foot anchor 고정, swing foot·무릎이 항상 지면 위, joint limit 준수.
   - Dynamics: **MPC(MJPC + iLQG)**로 동역학적 실현 가능성 보정(T=2.0s, dt=0.01s). 결과를 retargeted motion DB에 저장.
   - 효과: unit-vector 방식 대비 limb penetration / contact foot slip / joint limit 위반 제거.

2. **Steerable motion synthesis (Stage 2)**:
   - **Hyperspherical VAE**가 state-transition을 latent로 임베딩. **18-D latent**를 **von Mises-Fisher(vMF) 분포**로 구조화(구면 latent).
   - Decoder = **MoE(전문가 6개 + gating)**, 각 expert/encoder는 FC 2-layer×256, gating 2-layer×64.
   - VAE는 "이전 state + latent vector"로 현재 state를 재구성. State vector(49-D) = base height/orientation/lin·ang vel/foot pos/joint angle/joint speed.
   - **상위 RL 정책(PPO)** 이 사용자 steering 명령(전진·turn 속도)과 직전 모션 state를 받아 latent z̃∈R¹⁸ 를 출력 → z = z̃/‖z̃‖₂ 로 구면에 투영 → decoder가 reference 모션 생성.
   - 보상: `r = exp(−(v_fwd−c_fwd)²/0.25 − (ψ̇−c_turn)²/0.1)` (속도·turn 추종 지수보상, 모드 스위칭 규칙 없음).

3. **Motion tracking (Stage 3)**: **residual RL 정책**이 합성된 reference 모션을 실로봇에서 joint position correction으로 실행.

### 결과 / 플랫폼
- **Unitree Go2 실기** 배포. dog 모션 13,076 pose(+좌우 미러).
- 실시간 user-steerable locomotion + gait switching(pace/trot/gallop 자동전환). sim 기준 base velocity RMSE 0.11 m/s.

### 한계
- **데이터 밀도 의존**: 학습 데이터 sparse 영역(고속 gallop, pace↔trot 전이)에서 일반화 저하.
- synthesis가 **kinematic 전용**이라 고속에서 motion artifact(과격한 동작) 발생 → "data sparsity에 강건하면서 physically grounded한 synthesis"가 future work.

### 사용자 설정과의 관련성 / 차용 요소
- **거의 동일한 출발점**: Go2 + 비라벨 평지 모션 + (지형 제외) latent 모션 prior + 상위 RL steering. 사용자 ②-2(latent를 RL로 조종)의 **검증된 레퍼런스 아키텍처**.
- **차용**: ① "VAE latent + 상위 PPO가 latent를 출력"하는 2단 분리 구조, ② **구면(vMF) latent**가 mode-collapse 없이 multi-modal gait를 담는다는 점, ③ **kino-dynamic retargeting**(사용자의 jump 모션을 Go2에 물리적으로 정합시키는 데 직접 활용 가능), ④ residual tracking 정책 분리.
- **한계/주의**: 본 논문엔 **지형(exteroception) conditioning이 없다**. 사용자 핵심인 (지형,모션) joint 표현·OOD 지형 일반화는 미해결 영역으로 남김 → 본 연구의 차별점이 바로 여기.

---

## 2. Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility

- **저자/기관**: Zewei Zhang, Chenhao Li, Takahiro Miki, **Marco Hutter** (ETH Zürich, Robotic Systems Lab)
- **venue/연도**: arXiv 2505.16084, 2025
- **링크**: https://arxiv.org/abs/2505.16084 · pdf: https://arxiv.org/pdf/2505.16084

### 핵심 아이디어
**평지에서 학습한 motion prior(VAE latent)** 를, 계단·gap 등 복잡 지형으로 **RL로 적응**시킨다.
1. **VAE**가 평지 locomotion trajectory를 latent로 인코딩(unsupervised, task 명세 불필요).
2. **상위 RL 정책(PPO)** 이 **terrain feature + proprioception**을 받아 latent code를 출력.
3. **frozen decoder**가 latent → motor command로 복원하여 장애물(계단/gap/slope) 통과.
- RL 목적함수 = task 보상 + prior 구조와의 일관성 유지 → 평지 모션의 **스타일/자연스러움을 보존**하면서 지형 적응.

### 결과 / 한계
- **ANYmal C**. 계단/gap/slope/혼합 지형 통과. scratch 학습 대비 sample-efficient, unseen 장애물 일반화.
- 한계: RL 시 충분한 terrain diversity 필요, 분포 밖 극한 장애물·sim-to-real gap 미해결.

### 사용자 설정과의 관련성 / 차용 요소
- **②-2의 직접적 선례**: "평지 latent prior → 지형 조건부 상위 RL이 latent 출력 → 계단/gap 통과"는 사용자 파이프라인과 **구조적으로 동일**. 본 연구가 "비슷한 게 이미 되는가?"의 답.
- **차용**: ① decoder **freeze + 상위 RL latent 출력** 패턴, ② **prior 일관성 정규화**(latent를 prior 근방에 묶어 자연스러움 유지)로 OOD 지형에서도 모션 prior 붕괴 방지 — 사용자 ②-2의 "기존 prior를 변형해 새 skill mapping"에 핵심.
- **차별화 포인트(우리 기여 여지)**: 본 연구는 **지형을 latent에 함께 임베딩하지 않고**(모션 prior만 VAE, 지형은 상위 RL의 obs), 사용자 제안 ①의 **(지형,모션) joint AE** 는 한 단계 더 나아간 것. "joint latent가 분리형보다 이득이 있는가"가 검증 과제.

---

## 3. PULSE: Universal Humanoid Motion Representations for Physics-Based Control

- **저자/기관**: Zhengyi Luo, Jinkun Cao, Josh Merel, Alexander Winkler, Jing Huang, Kris Kitani, Weipeng Xu (Meta Reality Labs Research + CMU)
- **venue/연도**: ICLR 2024 (arXiv 2310.04582)
- **링크**: https://arxiv.org/abs/2310.04582 · html: https://arxiv.org/html/2310.04582v2

### 핵심 아이디어
대규모 모션(AMASS)을 **conditional VAE + learnable prior**로 압축해 **재사용 가능한 universal latent motor representation**을 만든다.
- 구성: **Encoder** E(proprio+goal → 대각 Gaussian latent), **Decoder** D(latent+proprio → motor action, **action space에서 직접** 동작), **Learnable Prior** R(proprioception-conditioned Gaussian — 표준 VAE의 zero-mean prior를 대체).
- **latent 32-D**(humanoid 69-DOF의 절반). reconstruction loss 없이 action space에서 distill.
- 학습: 강한 imitator(PHC+) → **online distillation**(student rollout을 teacher가 action annotate)으로 supervised 학습(RL과 supervised 혼합은 latent를 noisy하게 만들어 회피). loss = action recon + KL(↔learnable prior) + temporal 정규화.
- **downstream**: decoder·prior **freeze**, 상위 task 정책이 **prior mean μ에 대한 residual** latent를 출력: `a_task = D(π_task(z|s_p,s_g) + μ_p)`. → 탐색을 물리적으로 그럴듯한 모션 분포에 grounding → 수렴 속도·자연스러움 향상.

### 결과 / 한계
- AMASS imitation 97.1% 성공(36.1mm). 생성 task(speed/strike/reach/지형 통과)에서 ASE·CALM latent baseline 능가, adversarial reward 없이 human-like. 한계: 정보 bottleneck으로 100% fidelity 불가, latent 비해석성.

### 사용자 설정과의 관련성 / 차용 요소
- **②-2 설계의 교과서**: "**learnable(=conditional) prior** + downstream RL이 prior mean에 **residual**을 더해 새 task를 latent에 매핑"은 사용자가 말한 "기존 motion prior를 새 task에 맞게 변형"을 **수학적으로 구현하는 정확한 레시피**.
- **차용**: ① **proprioception-conditioned learnable prior**(사용자는 여기에 **terrain-conditioned prior** 로 확장하면 ① joint AE 정신과 통합 가능), ② **residual-on-prior** 탐색(OOD task에서도 모션 붕괴 없이 새 skill 탐색), ③ RL+supervised 혼합이 latent를 망친다는 경고 → ②-1(IL로 reference 생성) vs ②-2(RL)의 latent 오염 분리 설계 근거.
- **한계**: 휴머노이드 + action-space latent(recon loss 無). 사용자는 (지형,모션) **재구성(reconstruction)** 기반 AE를 원하므로, PULSE식 distillation과 사용자식 recon-AE는 **latent 품질 trade-off**(distill=물리적, recon=데이터 충실) 검토 필요.

---

## 4. MaskedMimic: Unified Physics-Based Character Control Through Masked Motion Inpainting

- **저자/기관**: Chen Tessler, Ofir Nabati, Gal Chechik, Yunrong Guo, **Xue Bin Peng** (NVIDIA, Tel-Aviv Lab)
- **venue/연도**: SIGGRAPH Asia 2024 / ACM TOG 2024 (arXiv 2409.14393)
- **링크**: https://research.nvidia.com/labs/par/project/maskedmimic.html · arXiv: https://arxiv.org/abs/2409.14393

### 핵심 아이디어
제어를 **"masked motion inpainting"** 문제로 통일: 부분적으로 주어진 조건(일부 joint target, joystick, path, **terrain**, text style)에서 **나머지 전신 모션을 완성**하는 단일 컨트롤러를 RL로 학습. 학습 중 입력을 무작위로 마스킹 → 추론 시 임의의 부분 조건 조합 수용(compositional).

### 결과 / 관련성
- 휴머노이드(sim), 다양 지형. 부분 joint target → physically feasible 전신 모션, 멀티모달 조건 합성.
- **사용자 관련성(②-1)**: 사용자의 "AE로 **reference dataset 생성**"은 본질적으로 **조건부 모션 생성(inpainting)** — 지형 height + 일부 motion 조건을 주면 나머지 trajectory를 완성하는 MaskedMimic식 프레이밍이 ②-1 reference 생성기의 강력한 후보.
- **차용**: ① **마스킹 학습**으로 "지형만 주어진 OOD(계단/crawl)"와 "지형+모션 주어진 평지"를 한 모델로 처리 → 사용자의 paired/unpaired 혼합 데이터에 적합, ② terrain을 conditioning 모달리티로 직접 취급.
- **한계**: 휴머노이드·그래픽스 중심(실로봇/sim-to-real 미초점), 본 페이지 수준에선 terrain 표현 세부 미공개(전체 논문 확인 필요 — **일부 미확인**).

---

## 5. Lifelike Agility and Play in Quadrupedal Robots using RL and Generative Pre-trained Models

- **저자/기관**: Lei Han, Qingxu Zhu 등 (Tencent Robotics X)
- **venue/연도**: Nature Machine Intelligence, 2024 (arXiv 2308.15143)
- **링크**: https://arxiv.org/abs/2308.15143 · html: https://arxiv.org/html/2308.15143v2

### 핵심 아이디어 — 3계층 hierarchical
1. **Primitive-level**: 동물 모션캡처로 **VQ-VAE 이산 primitive(VQ-PMC: Vector Quantized Primitive Motor Control)** 학습. conditional encoder(proprio + 미래 target trajectory → 이산 latent code), **frozen decoder**(proprio + 양자화된 embedding → 제어신호).
2. **Environmental-level**: frozen primitive decoder 재사용. 환경 정책이 **이산 embedding에 대한 categorical 분포**를 출력 → 새 task(계단/jump/creep)를 sparse-reward PPO로 빠르게 적응. multi-expert distillation으로 통합.
3. **Strategic-level**: 멀티에이전트 chase-tag 등 고수준 전략(방향/속도 명령).

### 결과 / 한계
- **MAX 사족로봇(실기)**, zero-shot sim-to-real. 복잡 장애물 통과 + 경쟁적 chase-tag emergent 전략.
- 한계: 모션캡처 취득 비용, 초기 실세계 배포 시 외부 mocap 의존(onboard camera distill로 완화), 환경레벨에서 여전히 curriculum/reward 설계 필요.

### 사용자 설정과의 관련성 / 차용 요소
- **②-2 + 이산 latent의 모범 사례(사족·실기)**: "동물 모션 → 이산 latent prior(freeze) → 상위 RL이 latent 선택으로 **새 task(계단/jump/creep)** 학습"은 사용자 OOD 확장 아이디어와 **정확히 일치**, 그것도 **실제 사족로봇**에서.
- **차용**: ① **VQ-VAE 이산 latent**(사용자 ①의 모션 AE 대안 — 이산 코드북은 "새 skill = 새 코드/코드조합" 매핑이 직관적이고 mode collapse가 적음), ② **categorical 상위 정책**으로 OOD task를 기존 primitive 조합으로 표현, ③ 3계층 분리(primitive/env/strategy)는 사용자의 (모션 prior)/(지형 적응)/(task) 분리와 대응.
- **시사**: 사용자가 연속 Gaussian AE(②-2) 대신 **VQ-VAE 이산 코드북**을 쓰면 "계단/gap/crawl을 새 코드로 추가"하는 게 더 깔끔할 수 있음 → 설계 선택지로 제시.

---

## 6. [Context] 기반 latent skill embedding 계열 (ASE / CALM / C·ASE)

2024+ 우선 원칙에 따라 요약만 — 단, 위 최신 연구들이 모두 이 계열의 후손이므로 **반드시 필요한 baseline/계보**로 포함.

- **ASE: Adversarial Skill Embeddings** (Peng 등, NVIDIA/UC Berkeley, SIGGRAPH 2022, arXiv 2205.01906): 비라벨 모션에서 **구면 latent skill space** 를 적대적(AMP discriminator)으로 학습. 상위 정책이 latent z를 선택해 다양 skill 재사용. → **latent skill space의 시조**. 사용자 latent의 "구면(hyperspherical)" 선택(anchor도 vMF 구면 사용)의 근거.
- **CALM: Conditional Adversarial Latent Models** (Tessler 등, NVIDIA, SIGGRAPH 2023, arXiv 2305.02195): semantically meaningful latent + **directable** 정책. 모션 인코더가 motion→latent, 정책이 latent 추종. → "directable/steerable"의 직접 선조(anchor "steerable"의 어원적 맥락).
- **C·ASE: Conditional ASE** (SIGGRAPH Asia 2023): ASE에 **이산 condition(skill 라벨)** 을 추가해 명시적 skill 제어 + 재구성 정확도 향상. → 사용자가 "skill을 명시적으로 호출"하고 싶을 때의 conditioning 설계 참고.
- (보조) **VQ-PMC / Multiple-AMP / Generalized Animal Imitator**: 이산 코드북·다중 스타일·동물 모방 prior 변형들. Lifelike Agility(§5)가 대표.

**왜 포함했나(2024 이전인데)**: 위 모든 2024–2025 연구의 **latent 구조(구면/이산/조건부)·재사용(상위 RL이 latent 출력) 패턴이 ASE/CALM에서 정립**되었기 때문. 사용자 ①·②의 설계 용어(latent skill, conditioning, directable)가 이 계보에서 나옴.

---

## 7. 조사 종합 — 사용자 제안 ①(3종 AE) 및 ②-1/②-2 관점에서 이 분야가 시사하는 것

### (A) 제안 ① — 지형 AE + 모션 AE + (지형,모션) joint AE
- **모션 AE는 검증됨**: anchor(vMF VAE 18-D), PULSE(conditional VAE+learnable prior 32-D), Lifelike(VQ-VAE 이산)가 모두 "모션 prior를 latent로 압축 후 재사용"을 실증. **latent 분포 선택지 3가지**가 분명히 존재:
  - **구면(vMF/hyperspherical)** — anchor·ASE. multi-modal gait를 mode-collapse 없이 담음. **사족 steerable에 실적 최다 → 1순위 추천**.
  - **Gaussian + learnable prior** — PULSE. prior를 **조건부(proprio/terrain)** 로 만들 수 있어 ①의 joint 정신과 자연 통합.
  - **VQ-VAE 이산 코드북** — Lifelike. "OOD task = 새 코드 추가"가 직관적, 실사족 실증.
- **(지형,모션) joint AE는 거의 미개척**: 조사된 최신 연구 대부분은 **모션만 latent로, 지형은 상위 RL의 obs**로 분리(§2 Motion Priors Reimagined가 대표). 사용자의 **joint latent** 는 한 단계 더 나아간 것 → **차별화/기여 지점이자 동시에 검증 부담**. 권고: "분리형(모션 latent + 지형 obs)"을 **baseline**으로 먼저 세우고, joint latent의 추가 이득을 ablation으로 입증하는 전략(MaskedMimic식 terrain-as-condition이 중간 지점).
- **retargeting을 ① 앞단에 필수 배치**: anchor의 kino-dynamic retargeting(IK+MPC)이 없으면 동물/평지 모션과 Go2 동역학 불일치로 latent 품질이 무너짐. 사용자 jump 모션의 물리 정합에 직접 필요.

### (B) 제안 ②-1 — AE로 reference dataset 생성 후 IL
- **조건부 모션 생성 = inpainting 프레이밍**(MaskedMimic, §4)이 가장 적합. "지형 height(+일부 모션) → 나머지 trajectory 완성"으로 reference를 생성하면 paired/unpaired 혼합 데이터(사용자가 수동 페어링한 jump+terrain)와 잘 맞음.
- **주의(공통 경고)**: PULSE가 명시 — **RL과 supervised(IL)를 한 latent에 섞으면 latent가 noisy**해짐. ②-1(순수 생성/IL)과 ②-2(RL)를 **분리된 latent/단계**로 두거나, decoder를 freeze하고 상위만 분리하는 설계 권장.

### (C) 제안 ②-2 — AE latent로 RL, OOD task를 새 skill로 latent에 mapping
- **이 분야의 합의된 정답에 가장 가까운 부분**. 세 가지 검증된 패턴:
  1. **decoder freeze + 상위 RL이 latent 출력**(anchor·Motion Priors Reimagined·PULSE·Lifelike 공통). OOD 지형/task를 기존 prior 위에서 RL로 탐색.
  2. **residual-on-prior**(PULSE): 상위 RL이 prior mean에 residual을 더함 → OOD에서도 모션 붕괴 없이 "기존 prior를 변형해 새 skill" — **사용자 문구와 정확히 일치**. 1순위 차용.
  3. **prior 일관성 정규화**(Motion Priors Reimagined): latent를 prior 근방에 묶어 자연스러움 유지 + OOD 적응 동시 달성.
- **OOD(계단/gap/crawl) 매핑의 현실**: §2·§5가 이미 계단·gap·jump·creep을 latent prior 위 RL로 해냄 → **사용자 ②-2는 실현 가능성이 높음**. 단 두 연구 모두 "RL 시 terrain diversity/curriculum 필요"를 명시 → 사용자도 parkour curriculum이 필요.
- **이산 vs 연속**: OOD를 "새 skill 추가"로 명시적으로 다루려면 **VQ-VAE 코드북(Lifelike)** 이 연속 latent보다 운영상 깔끔할 수 있음 — 설계 분기점으로 명시.

### (D) 핵심 한 줄
> **anchor(Go2 steerable VAE+RL) + Motion Priors Reimagined(평지prior→지형RL) + PULSE(learnable prior+residual) + Lifelike(이산 prior→OOD RL, 실사족)** 의 결합이 사용자 파이프라인의 **검증된 구성요소 전부**를 이미 제공한다. 사용자의 **진짜 신규성·기여는 "(지형,모션) joint AE"** 한 곳에 집중되며, 나머지(모션 latent, retargeting, latent+RL, OOD 확장)는 차용으로 빠르게 조립 가능하다.

---

### 참고 링크 모음
- Walk like Dogs (anchor): https://arxiv.org/abs/2507.00677
- Motion Priors Reimagined: https://arxiv.org/abs/2505.16084
- PULSE: https://arxiv.org/abs/2310.04582
- MaskedMimic: https://arxiv.org/abs/2409.14393 · https://research.nvidia.com/labs/par/project/maskedmimic.html
- Lifelike Agility (Nature MI 2024): https://arxiv.org/abs/2308.15143
- ASE: https://arxiv.org/abs/2205.01906 · CALM: https://arxiv.org/abs/2305.02195

> **미확인 표기**: MaskedMimic의 terrain 표현 세부 구조는 프로젝트 페이지 수준에서만 확인(전체 논문 2409.14393 정밀 검증은 미수행). 그 외 §1–§3, §5는 arXiv 본문 fetch로 직접 확인.

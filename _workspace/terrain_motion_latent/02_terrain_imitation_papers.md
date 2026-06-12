# 조사B — Terrain-aware Imitation / Parkour + IL 결합 논문 조사

> 작성: team worker `paper-b` (Task #2) · 2026-06-12
> 대상 연구: Unitree Go2 사족보행 parkour를 **자연스러운 모션**으로 수행하는 정책 학습. 코드베이스 = Extreme Parkour(CMU) 파생.
> 사용자 보유 자산: 평지 walk/run/trot 모션 + (장애물 정보 없는) jump 모션 → 사용자가 수동으로 장애물을 만들고 jump 모션 주변 지형 height를 paired로 데이터화 → **(지형, 모션) paired 데이터셋**.
> 제안 알고리즘: ① 지형 AE + 모션 AE + (지형,모션) joint AE → ②-1 AE로 reference dataset 생성 후 IL, ②-2 AE latent로 RL (OOD task=계단/gap/crawl은 RL feedback으로 latent에 새 skill mapping).

---

## 0. 핵심 요약 (먼저 읽기)

- 사용자가 언급한 **"Hybrid Imitation Learning"** anchor 논문은 **HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos (CMU/NVIDIA/SFU, 2025)** 로 식별됨. 명칭 일치 + 방법 일치(모션 tracking + adversarial IL 결합, **수동 box geometry 배치로 영상 affordance 재현 → 장면을 motion과 paired**, scene point cloud를 "spatial phase"로 사용). 다만 **휴먼 캐릭터(SMPL) 시뮬레이션**이며 실로봇/사족보행이 아님 → 방법론은 직접 차용 가능하나 morphology gap 존재.
  - 후보가 하나 더 있음: **HIL이라는 약어를 쓰는 다른 논문은 미발견**. 다만 "Hybrid Imitation Learning" 키워드로 검색 시 위 논문이 유일하게 정확 일치. (조사A/C에서 다룰 latent/OOD 계열과 구분.)
- 사용자 설정과 가장 직접적으로 부합하는 핵심 메커니즘 4가지가 문헌에서 확인됨:
  1. **수동 장애물 배치로 motion-scene paired 데이터 구성** → HIL.
  2. **지형 height를 discriminator/motion prior에 conditioning** → Terrain-Conditional AMP(MDPI 2026).
  3. **평지 모션 prior를 frozen low-level로 두고 지형 적응은 high-level residual로** → Motion Priors Reimagined(EPFL/ETH 2025).
  4. **지형 위 reference motion 생성/retargeting (box 위 backflip 등)** → Spatio-Temporal Motion Retargeting(2024).

---

## 1. [ANCHOR] HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos

- **저자/기관**: Jiashun Wang, Yifeng Jiang, Haotian Zhang, Chen Tessler, Davis Rempe, Jessica Hodgins, Xue Bin Peng — **Carnegie Mellon University / NVIDIA / Simon Fraser University**
- **Venue/연도**: arXiv 2025 (arXiv:2505.12619) — *미확인: 정식 학회 채택 여부*
- **링크**: https://arxiv.org/abs/2505.12619 · https://arxiv.org/html/2505.12619v1

### 핵심 아이디어
하나의 통합 프레임워크에서 **두 학습 모드를 병렬 멀티태스크 환경으로 동시 학습**:
1. **Motion Tracking 모드**: reference parkour 모션을 frame-by-frame 추종 → 개별 스킬을 정밀하게 습득.
2. **Adversarial Imitation Learning (AMP-style) 모드**: 장애물 코스에서 target-following → 적응성·스킬 합성(composition) 확보.
> 저자 주장: "각 기법 단독으로는 suboptimal" — tracking 단독=환경 적응 불가, AMP 단독=mode collapse/품질 저하. **하이브리드로 상호 보완.**

### 지형 표현 방식 / 모션 결합 방식
- **Unified observation = scene 정보를 spatial "phase" 변수로 사용**: 캐릭터 state + agent-centric **point cloud**(장면 geometry). tracking 시 motion phase 추론 + 장애물 인지를 동시에 제공.
- **Scene Annotation**: 영상 속 affordance를 재현하도록 **box geometry를 수동 배치** → (motion, scene) paired 구성. ← **사용자의 "수동으로 장애물 만들어 height paired" 방식과 정확히 동일한 철학.**
- Point cloud에서 root에 가장 가까운 N개 점을 정책/discriminator 입력으로 사용.
- **Discriminator가 motion naturalness + scene context를 함께 평가** (terrain-conditioned discriminator의 한 형태).
- 아키텍처: Transformer policy + PointNet scene encoder, MLP critic(task indicator), MLP discriminator(state transition + scene context).
- 데이터: YouTube 19클립(총 30초), 15개 parkour 스킬. TRAM(vision pose estimation)→physics tracking refine.
- 보강 기법: **PSI(Perturbed State Initialization)** — reference 시작 state에 가우시안 노이즈 → state 이탈/스킬 전환 robust(=mode collapse 완화). 모드별 early termination.

### 사용자 설정과의 관련성 (높음)
- **paired (scene, motion) 데이터 구성 철학이 동일** → 사용자의 "수동 장애물 + height paired" 정당화 근거.
- **tracking + adversarial 결합**이 사용자 ②-1(reference 생성·IL)과 ②-2(RL) 의 결합 논리를 그대로 뒷받침.
- scene을 "phase"로 쓰는 발상은 사용자 joint AE(지형↔모션 결합 latent)의 동기와 통함.

### 차용 가능한 요소
- 단일 프레임워크에서 **tracking loss + AMP discriminator 병행** (②-1/②-2 통합 학습의 청사진).
- **scene point cloud를 spatial phase로** → 사용자가 height-map 대신/병행해 쓸 수 있는 표현.
- PSI: latent embedding 기반 정책의 robustness 확보용.

### 한계점
- **휴먼 SMPL 캐릭터 / 시뮬 only — 실로봇·사족보행 아님** (가장 큰 gap).
- 학습이 **순차 배열 장애물 가정** → 복잡한 레이아웃 일반화 약함.
- 다른 레이아웃/희소 장애물 타입 일반화 어려움, 넘어질 때 부자연 복구 잔존.

---

## 2. Parkour Learning for Quadrupeds via Terrain-Conditional Adversarial Motion Priors

- **저자/기관**: *미확인 (MDPI 페이지 403으로 본문 확인 불가, 검색 스니펫 기반)*
- **Venue/연도**: **Applied Sciences (MDPI), 2026** (Vol.16, 3448)
- **링크**: https://www.mdpi.com/2076-3417/16/7/3448

### 핵심 아이디어 (검색 스니펫 기반, 일부 미확인)
- **Terrain-Conditional AMP**: 사족로봇이 **외부 terrain height에 따라 motion style을 동적으로 적응**하도록 discriminator/motion prior를 **지형 geometry에 conditioning**.
- 즉 AMP discriminator의 입력에 terrain 정보를 함께 넣어, "평지에서는 이 style, 장애물에서는 저 style"을 환경 기하에 맞춰 전환.

### 사용자 설정과의 관련성 (매우 높음 — 명칭·주제 직격)
- 사용자 제안 "(지형, 모션) paired"의 **discriminator-side 구현 레퍼런스**. ②-2 RL에서 AMP를 쓸 경우 terrain conditioning을 어떻게 넣는지 직접 참조.

### 차용 가능한 요소
- **terrain-conditioned discriminator 입력 설계** (height/scandots를 reference·policy transition과 함께 discriminator에 주입).

### 한계점
- *본문 미확인* — 구체 아키텍처/터레인 표현/Go2 적용 여부/실로봇 검증 여부 **미확인**. MDPI Applied Sciences는 top-tier 로보틱스 학회 대비 검증 강도 낮을 수 있어 **방법 인용 시 1차 출처 본문 재확인 필요**.

---

## 3. Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility

- **저자/기관**: Zewei Zhang, Chenhao Li, Takahiro Miki, Marco Hutter — **EPFL / ETH Zürich (AI Center, Robotic Systems Lab)**
- **Venue/연도**: arXiv preprint 2025 (arXiv:2505.16084v2)
- **링크**: https://arxiv.org/abs/2505.16084 · https://arxiv.org/html/2505.16084v2

### 핵심 아이디어
**계층적 2단계**: (1) **평지 동물 MoCap만으로 low-level 모션 prior 사전학습**(imitation), (2) high-level 정책이 **residual correction**을 학습해 거친 지형에서 perceptive locomotion·장애물 회피·goal navigation 수행.
- low-level은 **frozen** → latent embedding을 출력해 basic skill 명령.
- high-level teacher: **16-dim latent command + 12-dim joint residual** 출력.
- **residual penalty** `w_res · Σ(a_res)²` 로 motion style에서의 과도 이탈 억제.
- 모션 압축: **FLD(Fourier Latent Dynamics) encoder** 로 trajectory→저차원 latent.
- teacher(noiseless)→student(noisy) distillation (GRU belief encoder).

### 지형 표현 방식
- 각 발 주변 **elevation scan** + downsampled Velodyne LiDAR(sparse conical) + 작은 MLP들이 terrain/privileged(접촉, 마찰) 인코딩.

### 사용자 설정과의 관련성 (매우 높음 — ②-2 직격)
- **"평지 모션을 latent prior로 frozen → OOD 지형은 RL residual로 확장"** 이 사용자 ②-2(OOD task를 RL feedback으로 latent에 새 skill mapping)와 **개념적으로 거의 동일**. 사용자 joint AE latent = 이 논문의 frozen low-level latent에 대응.

### 차용 가능한 요소
- **frozen latent prior + RL residual** 구조 (사용자 ②-2의 직접 구현 템플릿).
- **residual penalty**로 자연스러움 보존 ↔ 적응성 trade-off 제어 (사용자 "자연스러운 모션" 목표에 핵심).
- FLD로 모션 AE를 구성하는 구체 기법 (사용자 모션 AE 대안).

### 한계점
- **mode collapse — 단일 low-level gait에 습관적으로 의존** (사용자 자연스러움/다양성 목표에 직접 위협. 대응책 설계 필요).
- 학습 시나리오 제한적, **gap/stepping-stone/overhang 처리 불가** (← 사용자 OOD 목표인 gap/crawl과 정확히 겹치는 미해결 영역 = 차별화 기회).
- residual penalty weight 튜닝 민감.

---

## 4. SF-TIM: Enhancing Quadrupedal Jumping Agility by Combining Terrain Imagination and Measurement

- **저자/기관**: Ze Wang, Yang Li, Long Xu, Hao Shi, Zunwang Ma, Zhen Chu, Chao Li, Fei Gao, Kailun Yang, Kaiwei Wang — (Zhejiang Univ. 계열 추정, *기관 명시 미확인*)
- **Venue/연도**: arXiv 2024 (arXiv:2408.00486)
- **링크**: https://arxiv.org/abs/2408.00486 · https://arxiv.org/pdf/2408.00486

### 핵심 아이디어
사족 **점프(jump) agility** 특화. 두 축 결합:
- **Terrain Imagination**: 직접 센서 없이 terrain 특성을 예측/상상 → 선제적 trajectory 계획.
- **Terrain Measurement**: 실제 센서 피드백으로 예측 보정.
- terrain elevation으로 점프 trajectory 최적화, PPO 기반 RL.

### 사용자 설정과의 관련성 (중)
- **jump 모션 + terrain height**라는 사용자 핵심 자산과 task가 정확히 일치(점프). "장애물 height를 모션과 결합해 점프 학습"의 실증 사례.
- imagination/measurement 이원화는 사용자가 height paired 데이터로 "지형 상상(AE 복원)"을 한다는 발상과 통함.

### 차용 가능한 요소
- **terrain height-conditioned jump reference** 설계 + imagination(예측) vs measurement(실측) 분리 학습.

### 한계점
- jump 단일 스킬 중심 → 다양한 parkour 스킬/스킬 합성은 범위 밖.
- 본문 내 기관/정량 비교 일부 **미확인**.

---

## 5. ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots

- **저자/기관**: David Hoeller, Nikita Rudin, Dhionis Sako, Marco Hutter — **ETH Zürich (RSL)**
- **Venue/연도**: **Science Robotics 2024** (arXiv 2023, arXiv:2306.14874)
- **링크**: https://www.science.org/doi/10.1126/scirobotics.adi7566 · https://arxiv.org/abs/2306.14874

### 핵심 아이디어
- 장애물 타입별 **저수준 스킬(walk/jump/climb/crouch) 학습** → **high-level 정책이 지형에 맞춰 스킬 선택·제어**.
- **perception 모듈**이 occluded/noisy 센서로부터 장애물 형상 재구성 → scene understanding.

### 사용자 설정과의 관련성 (중)
- "스킬 단위로 학습 후 high-level이 지형 따라 선택"은 사용자 latent-skill 매핑(②-2 OOD 스킬 추가)의 **모듈식 대안 관점**. 단, 이 논문은 IL(reference motion)이 아닌 **순수 RL skill + selector** 라는 점이 사용자(IL 기반)와 차이.

### 차용 가능한 요소
- **terrain → skill selection** 계층 구조, perception으로 occluded terrain 복원(사용자 지형 AE의 motivation 보강).

### 한계점
- reference motion 자연스러움 자체를 다루지 않음(스킬은 RL로 emergent) → "자연스러운 모션" 목표에는 간접적.

---

## 6. Spatio-Temporal Motion Retargeting for Quadruped Robots

- **저자/기관**: Taerim Yoon 외 (KAIST 계열 추정, *정확 기관 미확인*)
- **Venue/연도**: arXiv 2024 (arXiv:2404.11557)
- **링크**: https://arxiv.org/abs/2404.11557 · https://taerimyoon.me/Spatio-Temporal-Motion-Retargeting-for-Quadruped-Robots/

### 핵심 아이디어
- 노이즈 많은 모션 소스(핸드헬드 영상 등)를 **로봇 morphology/물리 특성에 맞는 robot-specific motion으로 retarget**.
- **SMR(Spatial Motion Retargeting)**: keypoint trajectory→전신 모션, kinematic artifact 보정.
- **TMR(Temporal Motion Retargeting)**: dynamics 제약 하에 모션 refine.
- **지형 인지 retargeting 시연: box 위에서 BackFlip** 등 지형 위 모션 생성 → 4종 로봇 실배포.

### 사용자 설정과의 관련성 (높음 — ②-1 reference 생성 직격)
- 사용자 ②-1 "AE로 reference dataset 생성"의 **모션 측 품질 보증 도구**. 평지 walk/run/trot/jump 모션을 **Go2 morphology + 장애물 지형에 맞게 retarget**해 paired reference를 정제하는 단계에 직접 적용 가능.
- "box 위 backflip" = **지형 위 reference motion 생성**의 직접 사례.

### 차용 가능한 요소
- SMR/TMR 2단계로 **(지형,모션) paired reference의 물리적 타당성·morphology 정합** 보증 → AE 생성 reference의 noise/artifact 제거 전처리.

### 한계점
- retargeting은 **단일 모션 변환**에 집중 → latent 스킬 공간 학습/OOD 확장은 범위 밖(조사A/C가 보완).

---

## 7. [보조/baseline] Extreme Parkour with Legged Robots (CMU)

- **저자/기관**: Xuxin Cheng, Kexin Shi, Ananye Agarwal, Deepak Pathak — **Carnegie Mellon University**
- **Venue/연도**: ICRA 2024 (arXiv 2023, arXiv:2309.14341)
- **링크**: https://arxiv.org/abs/2309.14341
- **포함 이유 (2024 이전이지만 필수)**: **사용자 코드베이스의 직접 baseline**. 본 조사 논문들과의 비교 기준점이며, 본 연구의 "additive 개선만 허용" 제약(MEMORY)의 기반. scandots(privileged) → depth distillation, single unified RL 정책으로 다양한 장애물 극복.
- **자연스러움 한계**: reward shaping 기반 emergent gait → 3-leg/pronk 등 부자연 gait 발생(사용자 기존 관측과 일치). **본 연구가 IL(reference motion)로 해결하려는 바로 그 gap.**
- *주의(MEMORY)*: Walk These Ways(input 팽창)·Robot Parkour Learning(env 재작성)은 사용자가 dealbreaker로 거부 → 본 조사에서 차용 대상에서 제외.

---

## 8. 조사 종합 — (지형,모션) paired 데이터 활용 & ②-1 reference 생성 관점

### (A) 사용자 paired 데이터 구성은 문헌상 검증된 접근
- **HIL의 "수동 box 배치로 motion-scene paired"** 가 사용자 방식과 동일 → 정당성 확보. 다만 HIL은 point cloud를 spatial phase로 썼고, 사용자는 height를 직접 paired → **표현 선택지(point cloud vs height-map/scandots)** 를 ②에서 정해야 함. Extreme Parkour 파생이므로 **scandots/height-map 유지가 input 팽창 회피 측면에서 유리**.

### (B) ②-1 (AE로 reference 생성 후 IL) 에 직접 쓸 도구
1. **Spatio-Temporal Motion Retargeting(§6)** = AE가 생성한 reference의 **물리·morphology 정합 후처리**. AE 생성물의 artifact를 SMR/TMR로 정제하면 IL 타겟 품질↑.
2. **HIL(§1)** = tracking + AMP를 한 프레임워크에 결합하는 청사진. ②-1 IL을 단독으로 두지 말고 **AMP discriminator를 병행**해 reference 밖 상태에서의 품질 유지(HIL의 핵심 교훈: tracking 단독은 적응 실패).
3. **품질 게이트 필수**: AE 생성 reference가 비물리적이면 IL이 붕괴 → retargeting 검증 + discriminator naturalness score를 게이트로.

### (C) ②-2 (latent로 RL, OOD 스킬 확장) 에 직접 쓸 도구
1. **Motion Priors Reimagined(§3)** 이 사용자 ②-2와 거의 동형: **frozen 모션 latent + RL residual**. 사용자 joint AE latent를 frozen prior로, 계단/gap/crawl은 high-level residual로 RL 확장 → **그대로 구현 템플릿**. 단 이 논문도 **gap/stepping-stone 미해결** → 사용자 OOD 목표가 곧 **차별화 포인트**.
2. **Terrain-Conditional AMP(§2)** = ②-2에서 AMP를 쓸 때 **terrain height를 discriminator에 conditioning** 하는 방법(본문 재확인 필요).
3. **ANYmal Parkour(§5)** = OOD 스킬을 **새 모듈/selector로 추가**하는 대안 관점(latent 매핑 vs 명시적 skill library).

### (D) 가장 경계해야 할 공통 실패 모드 = **mode collapse**
- §1(HIL), §3(Motion Priors)이 모두 mode collapse를 명시적 한계로 보고 → 사용자의 "자연스러운/다양한 모션" 목표에 최대 위협.
- 대응: **PSI(§1)**, **residual penalty 튜닝(§3)**, AMP discriminator의 style 다양성 유지. joint AE latent가 mode collapse하지 않도록 latent 정규화(조사A/C latent 계열 기법과 연계).

### (E) 차별화 여지 (문헌 공백)
- 사용자 접근의 신규성 = **지형 AE + 모션 AE + joint AE → 단일 latent에서 reference 생성(②-1)과 RL 확장(②-2)을 동시 수행**. 문헌은 (i) paired 데이터(HIL), (ii) frozen prior+residual(Motion Priors), (iii) terrain-conditioned discriminator(MDPI), (iv) retargeting(STMR)을 **개별적으로** 다룸. 이를 **joint AE latent 하나로 통합 + gap/crawl OOD 확장** 하는 것은 직접 선례 미발견 → 기여 가능 영역.

---

### 출처 (Sources)
- [HIL: Hybrid Imitation Learning of Diverse Parkour Skills from Videos](https://arxiv.org/abs/2505.12619)
- [Parkour Learning for Quadrupeds via Terrain-Conditional Adversarial Motion Priors (MDPI Applied Sciences 2026)](https://www.mdpi.com/2076-3417/16/7/3448)
- [Motion Priors Reimagined: Adapting Flat-Terrain Skills for Complex Quadruped Mobility](https://arxiv.org/abs/2505.16084)
- [SF-TIM: Combining Terrain Imagination and Measurement for Quadruped Jumping](https://arxiv.org/abs/2408.00486)
- [ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots (Science Robotics 2024)](https://www.science.org/doi/10.1126/scirobotics.adi7566)
- [Spatio-Temporal Motion Retargeting for Quadruped Robots](https://arxiv.org/abs/2404.11557)
- [Extreme Parkour with Legged Robots (CMU, ICRA 2024)](https://arxiv.org/abs/2309.14341)

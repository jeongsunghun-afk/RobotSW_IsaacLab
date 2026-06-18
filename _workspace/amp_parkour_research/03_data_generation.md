# Task #3 — 데이터 측 해법: 지형(parkour) 모션 데이터 생성/증강 조사

**작성**: lit-datagen (team amp-parkour-research)
**일자**: 2026-06-11
**질문**: AMP류 imitation으로 자연스러운 gait를 학습하고 싶은데 **지형 모션 데이터가 없다**(평지 자연 보행만 보유). 지형용 reference 데이터를 **만들어내는(생성/증강)** 방법은 무엇이 있나?
**제약**: Go2 12-DOF 실배포 전제 / 기존 env 재작성 금지 / 대규모 신규 인프라는 비용으로 명시

**증거등급 표기**: `[LIT]` = 논문/코드로 검증된 사실, `[HYP]` = 우리 적용에 대한 추론/가설

---

## 0. 요약 (먼저 읽기)

지형 모션 데이터를 만드는 길은 크게 5갈래이며, **비용↑·품질↑** 순으로 정렬하면:

| # | 접근 | 대표 연구 | 신규 인프라 | Go2 적용 난이도 | 데이터 품질 |
|---|------|----------|------------|----------------|------------|
| 5 | Kinematic terrain-warping (편법) | PARC 전처리 / Imitating Animals(2308.03273) | 낮음 (IK + heightmap offset) | 낮음 | **낮음** (dynamics 불일치 → artifact) |
| 3 | 1차 정책 rollout 재활용 (self-imitation) | PALo(2503.04462), Motion Priors Reimagined(2505.16084) | **거의 0** (우리 RL+AMP 그대로) | **낮음** | 중 (정책 품질에 종속) |
| 2 | Model-based TO retargeting | OPT-Mimic(2210.01247, code), TAMOLS(2206.14049) | 중 (TO solver 필요) | 중 | 높음 (동역학 일관) |
| 4 | 생성모델 (diffusion/VAE) | VAE-Loco(2205.01179), 캐릭터 diffusion | 높음 | 높음 | 가변 |
| 1 | **PARC**: 생성↔물리추적 부트스트랩 | PARC(2505.04002, SIGGRAPH25) | **매우 높음** (~1개월 GPU + mocap + 풀파이프) | 높음 | **높음** (4갈래 중 최고, but humanoid) |

핵심 통찰: **PARC가 "데이터측 해법"의 정점이지만 우리 비용 제약(Go2 12-DOF, env 재작성 금지)에는 과하다.** 우리 상황에 가장 fit한 것은 **#3 (1차 정책 rollout 재활용)** — 우리가 이미 가진 RL 파이프라인과 AMP 인프라만으로 추가 인프라 거의 없이 지형 reference를 부트스트랩한다. PARC의 **핵심 아이디어(물리추적으로 kinematic artifact를 교정)**를 #3/#5에 부분 차용하는 것이 현실적이다.

---

## 1. PARC — Physics-based Augmentation with RL `[LIT]` (검증 완료)

**검증**: 실존 확인. Michael Xu, Yi Shi, KangKang Yin, Xue Bin Peng (Simon Fraser University + NVIDIA), **SIGGRAPH Conference Papers '25** (2025-08, Vancouver). arXiv [2505.04002](https://arxiv.org/abs/2505.04002), project [michaelx.io/parc](https://xbpeng.github.io/projects/PARC/index.html), [ACM DL](https://dl.acm.org/doi/10.1145/3721238.3730616). 우리 task 지시서의 추정("Xu et al., SIGGRAPH 2025")이 **정확**.

### 1.1 방법 (검증된 세부)

소량 mocap → 지형 적응 데이터셋을 **반복 부트스트랩**하는 closed loop:

```
D⁰ (seed) ──┐
            ▼
  [G^i 학습] diffusion motion generator를 D^(i-1)로 학습
            ▼
  [생성] 새 procedural terrain 위에 kinematic motion M̃^i 합성 (iter당 1000~2000개)
            ▼
  [물리 추적] PPO tracking policy π^i가 M̃^i를 sim에서 모방 시도
            ▼
  [필터] 성공 추적분(엔드포인트 도달)만 = 물리보정된 M^i
            ▼
  D^i ← D^(i-1) ∪ M^i  ──→ 다음 iteration
```

- **Terrain 표현** `[LIT]`: 2.5D heightmap, 격자 0.4m, generator 입력은 캐릭터 로컬프레임으로 canonicalize한 **31×31 point grid**.
- **생성 디테일** `[LIT]`: batch 64 시퀀스 생성, contact loss + penetration loss + path-incompletion penalty 휴리스틱으로 선별. **Blended denoising**(terrain-only ↔ frame-conditioned 보간, s=0.65)으로 temporal smoothness와 terrain compliance 균형. 선별 모션은 jitter/penetration 줄이는 kinematic optimization 후 물리추적.
- **Seed 데이터** `[LIT]`: **14분 7초** parkour mocap(Beyond Capture Studios) + UE Game Animation Sample 5.5초. 스킬: climbing/vaulting/running on flat·bumpy/platforms/stairs. **수동** terrain 재구성 + contact label. PARC iteration 진입 전, 휴리스틱 terrain adaptation + 물리추적으로 **clip당 50 spatial variation** 증강.
- **캐릭터** `[LIT]`: **humanoid**(DOF 명시 없음; root 3D pos + 3D rot(exp map) + joint J×3 rot + J×3 pos + per-joint binary contact). 정책망 2048→1024→512.

### 1.2 한계 (검증된 self-stated) `[LIT]`

1. **자연스럽지 않은 거동 위험 잔존** — "PARC does not fully eliminate the risk of unnatural behaviors." (← 우리 핵심 우려와 직결: 생성 데이터의 artifact를 style로 학습)
2. **Procedural terrain만** — 실세계 다양성/복잡성 결여.
3. **Non-real-time** — A6000에서 0.5초 clip 생성에 ~12초. 로보틱스 closed-loop planning 불가.
4. **품질 천장** — 저품질 생성모션은 tracker가 못 따라가 교정 불가.
5. **연산 비용** — 4 iteration에 **A6000 단일 GPU 약 1개월**.

### 1.3 우리 적용 평가 `[HYP]`

- **장점**: "소량 평지 데이터 → 지형 데이터"라는 우리 질문에 **개념적으로 가장 직접적**. 물리추적 필터가 dynamics-feasible만 남기므로 #5 단순 warping의 artifact 문제를 구조적으로 해결.
- **dealbreaker**:
  - (a) **humanoid 캐릭터 애니메이션** 파이프라인. Go2 12-DOF로 옮기려면 generator/tracker/contact-labeling 전부 재구현 = **대규모 신규 인프라**(제약 위반).
  - (b) seed에 **parkour mocap 14분 + 수동 terrain/contact 라벨링** 필요 — 우리는 지형 mocap이 없음(질문의 전제).
  - (c) **~1개월 GPU + 비실시간** 생성기.
- **결론**: 전체 PARC 도입은 **비용 제약 초과로 비권장**. 단 **핵심 메커니즘 "kinematic 생성 → 물리추적으로 교정 → 성공분만 데이터화"는 경량 차용 가능**(아래 #3와 결합). 즉 generator를 diffusion 대신 우리의 1차 RL 정책으로 대체하면 PARC의 부트스트랩 정신을 인프라 0에 가깝게 재현.

---

## 2. Model-based Trajectory Optimization 기반 retargeting `[LIT]`

평지 gait를 지형 위로 warp하되, **모델기반 TO로 동역학 일관성을 복원**해 reference를 만드는 길.

### 2.1 OPT-Mimic `[LIT]` (코드 있음)
- 검증: Fuchioka, Xie, van de Panne (UBC), **ICRA/RA-L 2023**. arXiv [2210.01247](https://arxiv.org/abs/2210.01247), [project](https://www.cs.ubc.ca/~van/papers/2022-opt-mimic/), **코드 [github.com/yunifuchioka/opt-mimic-traj-opt](https://github.com/yunifuchioka/opt-mimic-traj-opt)**.
- 방법: **Single Rigid Body(SRB) 모델 TO**로 open-loop trajectory 생성 → imitation RL로 추적 → **Solo 8** 실로봇 zero-shot 전이(trot, front hop, 180 backflip, biped stepping).
- 우리 관점 `[HYP]`: mocap 없이 **TO만으로 reference 생성** 가능함을 실증. 단 시연 behavior는 dynamic skill이지 **terrain-traversal warping이 아님**. SRB TO에 heightmap 제약을 추가하면 지형 reference 생성기로 확장 가능하나, **TO solver 구축 = 중간 비용 신규 인프라**.

### 2.2 TAMOLS `[LIT]`
- 검증: Jenelten, Grandia, ... (ETH RSL). arXiv [2206.14049](https://arxiv.org/pdf/2206.14049). Terrain-Aware Motion Optimization for Legged Systems.
- 방법: **heightmap 제약 하에서 base pose + footholds를 동시 TO**. 이미 지형을 직접 입력으로 받는 모델기반 plan.
- 우리 관점 `[HYP]`: TAMOLS류 plan의 출력(base+foot trajectory)을 **AMP reference로 export**하면 "지형 위 동역학 일관 reference"가 된다. 단 ETH의 풀 TO 스택 의존 = 비용. **참고 가치는 높으나 직접 이식은 무겁다.**

### 2.3 Imitating Animals (terrain-adaptive) `[LIT]`
- 검증: arXiv [2308.03273](https://arxiv.org/pdf/2308.03273) "Learning Terrain-Adaptive Locomotion with Agile Behaviors by Imitating Animals".
- 방법: **AMP + IK retargeting** — dog mocap keypoint에서 base pos/orient 산출, **foothold를 robot end-effector와 pairing** 후 나머지 IK. 평지 dog mocap을 로봇에 옮기는 표준 retargeting을 지형 거동 학습과 결합.
- 우리 관점 `[HYP]`: **foothold-pairing IK retargeting**이 우리 #5(kinematic warping)의 구체 레시피. 평지 reference의 foot target을 heightmap에 snap → IK로 joint 복원. **저비용, 단 dynamics는 보장 안 됨**(아래 #5 한계).

---

## 3. ★ 1차 정책 rollout 재활용 (self-imitation / expert iteration) `[LIT]` — **우리에 최적**

부자연스러워도 **지형을 극복하는 1차 RL 정책**을 먼저 학습 → 그 성공 trajectory를 **2차 학습의 AMP demo로 재활용**. 우리가 이미 가진 RL+AMP 인프라만으로 지형 reference를 부트스트랩 = **신규 인프라 거의 0**.

### 3.1 PALo `[LIT]`
- 검증: arXiv [2503.04462](https://arxiv.org/pdf/2503.04462) "PALo: Learning Posture-Aware Locomotion for Quadruped Robots" (2025).
- 방법(검색+abstract 검증, PDF 압축으로 일부 `[HYP]`): **평지에서 auxiliary reward로 먼저 안정 gait 학습 → 학습된 정책 rollout에서 AMP data 수집 → 그 안정 gait 데이터를 복잡 지형 학습에 재사용**. terrain/reward/command curriculum 동반.
- 우리 관점: **우리 질문의 거의 정확한 답.** "평지 자연 보행 데이터만 보유 → 그걸로 평지 정책 학습 → rollout을 지형 학습 demo로 승격." **quadruped, AMP, 우리 인프라와 호환.**

### 3.2 Motion Priors Reimagined `[LIT]`
- 검증: arXiv [2505.16084](https://arxiv.org/html/2505.16084v2), project [anymalprior.github.io](https://anymalprior.github.io/). **ANYmal-D**, 실로봇 배포.
- 방법: (1) animal mocap을 IK retarget → (2) **FLD(Fourier Latent Dynamics) encoder + low-level policy를 평지에서 prior 학습** → (3) **frozen low-level 위에 high-level teacher가 latent command + joint residual을 지형 적응용으로 출력** → (4) GRU belief encoder로 student distillation.
- **새 reference를 생성하지 않고 residual correction을 학습** — 평지 prior의 자연스러운 스타일을 보존하며 지형 적응. 성공률 75~95%, baseline 대비 CoT↓, 실세계 smooth 전이.
- **한계** `[LIT]`: **mode collapse**(보통 single gait + residual에 의존), gap/overhang 같은 극한 지형 불가, low-level skill 전범위 미탐색.
- 우리 관점 `[HYP]`: 이건 데이터 생성보다 **prior 재사용**(Task #2 영역과 겹침)이지만, "평지 데이터 → 지형"의 또 다른 인프라-경량 경로. **데이터를 새로 만드는 대신 residual로 우회**하는 대안이라는 점이 시사적.

### 3.3 일반 원리 `[LIT]`
- IL survey ([Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1678567/full)): reference는 mocap뿐 아니라 **TO·hand-design·학습된 정책 rollout**에서 얻을 수 있음을 정리. self-imitation/expert-iteration이 정당한 reference source임을 뒷받침.

### 3.4 우리 적용 평가 `[HYP]`
- **비용**: 최저. 1차 정책 = 이미 가진 parkour RL. AMP 인프라 = 이미 보유(Task #4 확인 영역). 추가는 **rollout 수집 + "자연스러운 부분집합" 필터링 로직**뿐.
- **핵심 리스크 = 필터링**: 1차 정책이 3-leg gait/pronk 등 부자연 거동을 내면(우리 MEMORY의 알려진 문제) **그 artifact를 그대로 demo로 승격 → style로 학습**(self-reinforcing artifact). → **필터 기준이 생명**: contact symmetry, duty-factor, impact force, CoT 등 **명시적 품질 게이트**로 자연스러운 stride만 추출해야. PARC의 "성공 추적분만" 필터와 같은 정신.
- **PARC 차용**: rollout을 그대로 쓰지 말고, **kinematic 정제(foot clearance/penetration 제거) → 물리추적 재검증 → 통과분만** 파이프라인을 넣으면 품질↑.

---

## 4. 생성모델 기반 (diffusion / VAE) `[LIT]`

terrain-conditioned 생성으로 지형 모션 합성.

- **PARC** = 사실상 이 범주의 SOTA(diffusion generator). 위 §1.
- **VAE-Loco** `[LIT]` (arXiv [2205.01179](https://arxiv.org/abs/2205.01179)): **단일 trot 스타일** VAE, latent drive signal로 cadence/step-height/stance 연속 변조. **physics-feasible canonical trajectory로 학습.** 단 **평지 trot 한정**, 지형 conditioning 아님. → 지형 데이터 생성기로는 직접 부적합하나, **gait 파라미터 disentangle 합성**은 데이터 증강 보조로 참고 가능 `[HYP]`.
- **캐릭터 애니메이션 diffusion** `[LIT]`: CLoSD(ICLR25), PDP(SIGGRAPH Asia24), Diffuse-CLoC(2503.11801), Robot Motion Diffusion Model(SIGGRAPH Asia24). 대부분 **humanoid/character**이고 **명시적 terrain-conditioned quadruped는 희소** — 검색상 PARC가 유일하게 직접적. 
- 우리 관점 `[HYP]`: diffusion/VAE 생성기를 Go2용으로 새로 학습 = **데이터+인프라 비용 큼**, 그리고 **생성 artifact를 style로 학습**하는 동일 리스크. **물리추적 필터 없이는 위험.** 비권장(단독). PARC가 이 범주를 이미 "생성+물리필터"로 통합했으므로, 생성모델을 쓸 거면 PARC 구조를 따라야 함.

---

## 5. 단순 kinematic 편법 — terrain-aware warping `[LIT]`

평지 모션의 base/foot height를 terrain heightfield에 맞춰 **offset/snap**하는 최저비용 증강.

- **누가 했나** `[LIT]`:
  - PARC **전처리 단계**가 정확히 이것: "휴리스틱 terrain adaptation"으로 clip당 50 spatial variation 생성. **단, PARC는 이후 물리추적으로 교정**해야 쓸 만해진다고 명시 → 즉 **kinematic warping 단독은 불충분**함을 PARC가 스스로 증언.
  - Imitating Animals(2308.03273): **foothold-robot EE pairing + IK** 로 footstep을 지형에 맞춤(§2.3).
  - TAMOLS류 heightmap 제약(§2.2)은 warping을 "최적화"로 격상한 버전.
- **레시피** `[HYP]`: 평지 reference의 (1) swing foot target을 heightmap 표면에 snap, (2) base height를 지지 발 평균 + 공칭 stance height로 remap, (3) IK로 12-DOF 복원, (4) contact phase는 보존.
- **어디서 깨지나** `[LIT]`/`[HYP]`:
  - **Dynamics 불일치**: kinematic offset은 CoM 가속/접촉력/모멘텀을 무시 → 물리적으로 infeasible(특히 **큰 장애물/점프/큰 height gap**). AMP가 이런 데이터를 style로 학습하면 **infeasible target을 흉내내려다 불안정**.
  - **접촉 타이밍 왜곡**: 평지 contact schedule을 지형에 강제하면 발이 지형보다 일찍/늦게 닿음 → penetration 또는 floating.
  - PARC가 "kinematic optimization + 물리추적"을 둔 이유가 바로 이 깨짐 지점 보정.
- 우리 관점: **가장 싸고 가장 위험.** 작은 지형 변화(완만 slope, 낮은 step)엔 OK, parkour급 장애물엔 artifact 심함. **단독 사용 비권장, #3의 전처리 정제 단계로만 사용 권장.**

---

## 6. 비용 / 품질 매트릭스 (종합)

| 접근 | 필요 인프라 | Go2 12-DOF 적용 난이도 | 데이터 품질 | artifact→style 리스크 | env 재작성 |
|------|------------|----------------------|------------|---------------------|-----------|
| **#1 PARC 풀도입** | diffusion gen + 물리tracker + mocap + 라벨링, ~1개월 GPU | **높음**(humanoid 파이프 이식) | **최고** | 낮음(물리필터) | 큼 |
| **#2 TO retargeting** | TO solver(SRB/whole-body), heightmap 제약 | 중 | 높음(동역학 일관) | 낮음 | 중 (TO 스택 추가) |
| **#3 정책 rollout 재활용** | **거의 0**(기존 RL+AMP) + 필터 로직 | **낮음** | 중(정책 품질 종속) | **중~높음**(필터가 좌우) | **거의 없음** |
| **#4 생성모델 단독** | diffusion/VAE 학습 데이터+코드 | 높음 | 가변 | **높음**(필터 없으면) | 중 |
| **#5 kinematic warping** | IK + heightmap offset | **낮음** | **낮음**(dynamics 불일치) | **높음** | 거의 없음 |

**리스크 공통분모**: §1.2-1·§3.4·§4·§5 모두 **"생성/warp한 artifact를 AMP가 style로 학습"** 하는 동일 함정. **품질 게이트(물리추적 재검증 또는 명시적 kinematic 품질 메트릭)** 가 모든 경로의 성패를 가른다. PARC의 본질적 기여가 바로 이 게이트의 자동화(물리추적 필터)임.

---

## 7. 우리 질문에 대한 시사점 Top 3

### ① 우리 비용 제약에 최적해는 PARC 풀도입이 아니라 **#3 (1차 정책 rollout 재활용)** `[HYP]`
지형 mocap이 없고(질문 전제), env 재작성 금지·대규모 인프라 비용 명시인 우리에겐 **PARC(humanoid 파이프 + mocap + 1개월 GPU)는 과대비용**. 대신 **이미 가진 parkour RL 1차 정책의 성공 rollout을 AMP demo로 승격**하는 self-imitation(PALo 2503.04462가 quadruple로 실증)이 **신규 인프라 ~0**으로 지형 reference를 만든다. Task #4(AMP 인프라 통합성)와 직접 연결해 검토 권장.

### ② **품질 필터가 모든 데이터 생성 경로의 생사를 가른다** — "artifact를 style로 학습" 함정 `[LIT]+[HYP]`
PARC조차 "unnatural behavior 위험 잔존"을 명시(§1.2-1). 우리 1차 parkour 정책은 MEMORY상 **3-leg gait/pronk** 등 알려진 부자연 거동을 냄 → 무필터로 rollout을 demo화하면 **그 artifact를 자연스러운 style로 학습하는 self-reinforcing 붕괴**. 따라서 #3 채택 시 **명시적 품질 게이트 필수**: contact symmetry, duty-factor, impact force, CoT, foot-clearance 등으로 "자연스러운 부분집합"만 추출(우리 MEMORY의 stride/duty 진단 메트릭 재활용 가능).

### ③ **PARC의 핵심 메커니즘만 경량 차용**: "kinematic 생성 → 물리추적 교정 → 성공분만 데이터화" `[HYP]`
단순 kinematic warping(#5)은 dynamics 불일치로 parkour급 장애물에서 깨지고(PARC 스스로 전처리 후 물리추적 필요성 증언), 생성모델 단독(#4)은 artifact 위험. **두 약점을 PARC식 물리추적 필터가 동시에 해소.** 우리는 diffusion generator를 새로 만들 필요 없이, **generator 자리에 ①의 1차 RL 정책 또는 #5의 warped reference를 두고, 물리추적(또는 우리 sim에서 추적 성공/실패 판정)으로 게이트**하면 PARC의 부트스트랩 정신을 인프라 최소로 재현 가능. 즉 권장 스택: **평지 데이터 → 평지/지형 1차 정책 → rollout 수집 → kinematic 정제 → 우리 sim 물리추적 재검증 → 통과분만 AMP demo**.

---

## 부록: 인용 검증 상태

| 인용 | 상태 | 출처 |
|------|------|------|
| PARC (Xu et al., SIGGRAPH 2025) | `[LIT]` 검증 | [arXiv 2505.04002](https://arxiv.org/abs/2505.04002), [ACM 10.1145/3721238.3730616](https://dl.acm.org/doi/10.1145/3721238.3730616), [project](https://xbpeng.github.io/projects/PARC/index.html) |
| OPT-Mimic (Fuchioka et al., RA-L 2023) | `[LIT]` 검증 (+코드) | [arXiv 2210.01247](https://arxiv.org/abs/2210.01247), [code](https://github.com/yunifuchioka/opt-mimic-traj-opt) |
| TAMOLS (Jenelten/Grandia et al., ETH) | `[LIT]` 검증 | [arXiv 2206.14049](https://arxiv.org/pdf/2206.14049) |
| Imitating Animals terrain-adaptive | `[LIT]` 검증 | [arXiv 2308.03273](https://arxiv.org/pdf/2308.03273) |
| PALo (2025) | `[LIT]` 검증 (PDF 압축으로 일부 메커니즘 `[HYP]`) | [arXiv 2503.04462](https://arxiv.org/pdf/2503.04462) |
| Motion Priors Reimagined (ANYmal-D) | `[LIT]` 검증 | [arXiv 2505.16084](https://arxiv.org/html/2505.16084v2), [project](https://anymalprior.github.io/) |
| VAE-Loco | `[LIT]` 검증 | [arXiv 2205.01179](https://arxiv.org/abs/2205.01179) |
| IL for legged locomotion survey | `[LIT]` 검증 | [Frontiers 2025](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1678567/full) |
| 캐릭터 diffusion (CLoSD/PDP/Diffuse-CLoC) | `[LIT]` 실존, 우리 task 직접성 낮음 | 검색 확인, 미심층 |

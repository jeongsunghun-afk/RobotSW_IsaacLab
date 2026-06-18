# Task #5 — T-GMP 정독: terrain-conditioned motion prior의 OOD 해법 분석

> **대상**: Guo, Junhong et al., *"T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion"*, **arXiv:2606.06944, 2026-06-05**. (저자: Junhong Guo, Hao Hu, Chen Chen, Haoxuan Han, Linao Gong, Xin Yang, Zhicheng He, Yao Su, Fenghua He) — **[LIT] 실존확인 ✔ (abstract+본문 WebFetch 정독)**
> **로봇**: **휴머노이드** (28-DoF, ~55kg, 1.66m, LiDAR Mid-360). 사족 아님 — 메커니즘만 전이 분석.
> 작성: lit-priors / 2026-06-11 / [LIT]=문헌확인, [HYP]=추론

---

## 0. 한 줄 결론

T-GMP는 우리 핵심난제(**평지 walk 데이터에 OOD인 장애물 동작을 discriminator가 처벌**)에 대한 **정확히 올바른 메커니즘 = "terrain-conditioned discriminator"를 제시**한다. **그러나 그 해법은 "지형 모션 데모가 존재한다"를 전제**하며, T-GMP는 그것을 **privileged expert policy(RL) + 접근가능 지형 MoCap**으로 마련한다. 즉 **"평지 데이터만으로 terrain-conditioned prior를 만든다"는 우리 시나리오를 직접 풀지는 않는다** — 지형 데모 확보(Task #3)가 선결조건. 단, **우리 parkour teacher(height-scan) policy가 곧 그 privileged expert** 역할을 할 수 있다는 강한 합류점이 있다(§5).

---

## 1. 질문(1): 평지 데이터만으로 terrain-conditioned prior를 만드는가? → **아니오** [LIT]

T-GMP의 데이터는 **평지-only가 아니다**. 두 소스를 결합한다:
1. **Privileged policy data** — "terrain-conditioned locomotion data": 도전적 지형(계단/경사/beam)에서 **expert(privileged) policy가 생성**한 상태-지형 시퀀스.
2. **MoCap data** — "flat, gap, stage 등 **MoCap 촬영이 가능한 지형**에서 수집" 후 로봇 kinematics로 retarget.
   - 총 expert 데이터 **29.6분(88.8k frames)**.

**구조**: **β-VAE 확장 CVAE**. local height map을 **2-layer CNN(f_cnn)**으로 terrain embedding `h_temb,c`로 추출 → **decoder가 (latent z, h_temb,c)를 함께 입력**: `ŝ_{t:t+T} = Decoder(z_t, h_temb,c)`. encoder도 학습 시 state 시퀀스 + height map feature로 조건화. abstract 표현: *"captures a terrain-conditioned latent motion manifold from a few expert state-terrain demonstrations using a CVAE."*

> **판정**: T-GMP의 "terrain-conditioned" 능력의 원천은 **지형 데모 자체**다. 평지 prior를 지형으로 "외삽"하는 게 아니라, **지형 데모로 manifold를 채워서** 지형별 스타일을 분리한다. 우리가 가진 평지 walk만으로는 manifold의 지형 영역이 비어 있어 이 메커니즘이 작동 안 함. [LIT+HYP]

---

## 2. 질문(2): 우리 OOD 난제(평지 prior가 정당한 점프/계단 동작을 처벌)를 푸는가? → **메커니즘은 정확히 그것, 단 데이터 전제 하에** [LIT]

### T-GMP의 OOD 해법 = **terrain-conditioned discriminator** [LIT]
- AMP를 **terrain-conditioned discriminator**로 확장: `D(s_t^π, s_{t+1}^π | h_temb,d)` — **discriminator가 local terrain feature에 조건화**됨.
- style reward: `r_amp = max[0, 1 − (D(...)−1)² / 4]` (양수, AMP 표준형).
- 핵심 효과(본문): *"the terrain-conditioned discriminator prevents style collapse when multiple motion priors exist simultaneously"*, *"stair-climbing behaviors aren't penalized as out-of-distribution flat-walking violations."*
- t-SNE 증거: "highly compact intra-terrain clusters, clearly separable inter-terrain distributions" — 지형별로 "real" 분포가 분리됨.

### 우리 난제와의 정합 [LIT→HYP]
- **우리 현 상태**(code-feasibility 확인): AMP가 `_flat_env_mask`로 **평지 env에만 적용**, 장애물 동작은 평지 walk 분포에 OOD → discriminator가 정당한 점프/계단을 처벌.
- **T-GMP의 답**: discriminator를 **terrain에 조건화**하면, 계단 terrain일 때 "real" 분포 = **계단 모션**이 되어 정당한 계단 동작이 OOD가 아니게 됨. → **우리 난제를 정면으로 해결하는 메커니즘**.
- **그러나 결정적 전제**: 계단 terrain 조건의 "real" 분포를 채우려면 **계단 모션 데모가 있어야** 한다. **데모가 없으면** terrain-conditioned discriminator는 그 지형에서 진짜 샘플이 없어 의미 있는 style reward를 못 줌 → 사실상 "그 지형에선 discriminator off" = **우리 현재 `_flat_env_mask`와 거의 동일한 효과로 퇴화**. [HYP, 강한 근거]

> **판정**: T-GMP는 "discriminator를 끄지 말고 **지형 데모로 켜라**"고 답한다. 우리가 평지 데모만 가지면 T-GMP의 우월점(지형에서 **양(+)의 가이드**)이 사라지고 우리 현 masking과 수렴. **OOD 해결의 본질은 알고리즘이 아니라 "지형 데모 확보"**임을 T-GMP가 입증.

---

## 3. 질문(3): reward-only 변형 가능(policy obs 불변)인가? → **부분적. discriminator측은 OK, policy측은 input팽창 내포** [LIT]

T-GMP는 **두 곳**에서 terrain conditioning을 쓴다 — 이 둘을 분리해야 한다:

| 위치 | 무엇 | deploy 영향 | 우리 제약 |
|---|---|---|---|
| **Discriminator** `D(·\|h_temb,d)` | **reward 계산용**, 학습시간 only | deploy에 **불포함** | ✅ reward-only, input팽창 무관 |
| **Policy observation** | `O_t=[o_{t−4..t}]` + **terrain embedding `h_temb,p` concat** | deploy policy 입력 | ⚠️ **input팽창** |
| **Policy output** | **joint residual** `q*=q0+a` (latent 아님) | — | ✅ action space 불변 |

- **좋은 소식**: OOD를 푸는 **terrain-conditioned discriminator는 100% 학습시간 reward 신호** → deploy policy obs/action에 영향 없음. **이 부분만 떼어내면 reward-only 변형 가능**.
- **나쁜 소식**: T-GMP 원본 policy는 terrain embedding을 **obs에 concat** → 그 부분은 **input팽창**. 우리는 이걸 채택하면 안 됨.
- **우리 대응**: policy측 terrain embedding은 **우리 기존 height-scan(teacher) / RMA latent로 대체** → 새 input팽창 없음. **terrain conditioning은 discriminator에만** 도입. (단 discriminator가 보는 terrain feature는 학습시간 privileged 정보라 deploy 무관 → 자유롭게 사용 가능). [HYP, 구조적으로 타당]

> **판정**: **"terrain-conditioned discriminator만 additive 도입 + policy obs는 불변(기존 height-scan 재사용)"** 으로 reward-only 변형 성립. 이게 우리 제약(input팽창 금지·action 불변·deploy proprio-only)을 모두 만족하는 유일한 차용 경로.

---

## 4. 질문(4): Go2-ParkourImitation-v0 스택에 additive로 붙는가? → **discriminator측은 additive 가능, 데이터·로봇차이 주의** [HYP]

- **호환 요소**:
  - 우리 AMP 인프라(`_flat_env_mask` 기반 discriminator)는 이미 존재 → **terrain conditioning은 discriminator 입력에 terrain feature(또는 terrain-id) 추가 + env별 real-batch를 지형별로 분리**하는 additive 변경. env 재작성 불필요(reward/discriminator 측).
  - policy obs/action 불변 → RMA+DAGGER·deploy 파이프라인 그대로(§3).
  - clip(min=0): AMP style reward는 이미 `max[0,·]` 양수형 → 우리 total_reward clip과 충돌 없음(메모 `feedback_parkour_reward_weight_units` 함정 회피, bonus로 설계).
- **막힘점 / 선결조건**:
  1. **지형 모션 데모 부재**(우리 핵심 제약). terrain-conditioned discriminator를 "켜진 상태"로 쓰려면 계단/gap/step 모션 데모 필요 → **Task #3(데이터 생성)과 직결**. 데모 없으면 §2처럼 현 masking으로 퇴화.
  2. **로봇 차이**: T-GMP=휴머노이드 28-DoF + LiDAR perception. 우리=Go2 12-DoF proprio-only. discriminator는 학습시간이라 perception 차이 무관하나, **데모 retarget·foothold penalty 등 휴머노이드 특화 항은 직접 이식 불가**.
  3. T-GMP의 "Foothold Penalty"는 휴머노이드 발 배치용 → 사족엔 재설계 필요(또는 우리 기존 contact reward로 대체).

> **판정**: **discriminator의 terrain-conditioning은 우리 AMP 스택에 additive로 이식 가능**하나, **실효를 보려면 지형 데모가 선결**. 데모 확보 시 우리 난제의 직접 처방이 됨. 데모 없으면 가치 제한적.

---

## 5. 핵심 합류점 — 우리 parkour teacher가 곧 T-GMP의 "privileged expert" [HYP]

T-GMP의 지형 데모 출처 = "privileged policy data: 지형에서 expert policy가 생성한 state-terrain 시퀀스". **우리는 이미 height-scan을 보는 parkour teacher policy를 가지고 있다.** 그것이 (불완전하더라도) 계단/step/gap을 통과하는 **state 궤적을 롤아웃으로 생성**할 수 있다.

→ **잠재 레시피(우리 맥락 재구성)** [HYP]:
1. parkour teacher(또는 baseline RL)로 각 지형에서 **state 궤적 롤아웃** = 지형 데모 자동생성(MoCap 불필요, Task #3 영역).
2. 평지 walk MoCap(우리 보유) + 위 지형 롤아웃 = terrain-conditioned discriminator의 "real" 분포.
3. discriminator를 **terrain-id/feature 조건화** → 평지=자연보행 처벌, 지형=해당 지형 스타일 허용 → **OOD 처벌 해소 + 평지 자연성 유지**.
4. policy obs/action 불변 → deploy 그대로.

> 단 주의: teacher 롤아웃이 pronk/split-jump 같은 **부자연 동작이면 그것을 "정답"으로 학습**시키는 자기참조 위험. 데모 품질 게이팅 필요(메모 `project_parkour_pronk_measurement` 맥락). 이 위험·실현성은 **Task #3(데이터 생성)과 code-feasibility가 판단할 영역** — 본 Task에서는 "T-GMP가 이 경로를 정당화한다"까지만 보고.

---

## 6. 우리 질문에 대한 시사점 Top 3

1. **T-GMP는 우리 OOD 난제의 정확한 처방(terrain-conditioned discriminator)을 [LIT]로 입증한다.** "discriminator를 평지에서 끄는"(우리 현 `_flat_env_mask`) 대신 **지형에 조건화해 지형별 real 분포로 켜라** → 정당한 계단/점프가 OOD로 처벌 안 됨. abstract·본문·t-SNE 모두 일치. [LIT]

2. **단, 이 해법은 "평지 데이터만"으로는 미완성 — 지형 데모가 선결조건.** T-GMP도 평지로 외삽하지 않고 **privileged expert + 접근가능 지형 MoCap**으로 지형 manifold를 채운다. 우리가 평지만 쓰면 terrain-conditioned discriminator는 우리 현 masking으로 퇴화. **즉 OOD 해결의 병목은 알고리즘이 아니라 데이터** → Task #3로 핸드오프. [LIT+HYP]

3. **차용 경로는 명확: "terrain-conditioned discriminator만 additive, policy obs/action 불변".** discriminator는 학습시간 신호라 input팽창·proprio-only deploy 제약과 무관. policy측 terrain embedding(원본 input팽창)은 거부하고 **우리 기존 height-scan/RMA로 대체**. AMP style reward는 양수형이라 clip(min=0)과도 호환. **우리 parkour teacher가 곧 지형 데모 생성기**가 될 수 있어 데모 부재를 부분 우회할 잠재 경로 존재(품질 게이팅 필요, Task#3 검증). [HYP]

---

## 부록 — 출처/실존 확인

| 항목 | 값 | 등급 |
|---|---|---|
| 논문 | T-GMP: Terrain-conditioned Generative Motion Priors for Versatile and Natural Humanoid Locomotion | [LIT] ✔ |
| arXiv | 2606.06944 (2026-06-05) | [LIT] ✔ abstract+HTML 본문 정독 |
| 저자 | Junhong Guo, Hao Hu, Chen Chen, Haoxuan Han, Linao Gong, Xin Yang, Zhicheng He, Yao Su, Fenghua He | [LIT] ✔ |
| 로봇 | 휴머노이드 28-DoF ~55kg 1.66m, LiDAR Mid-360 | [LIT] ✔ |
| 데이터 | privileged expert terrain data + MoCap(flat/gap/stage), 29.6분/88.8k frames | [LIT] ✔ |
| 구조 | β-VAE 확장 CVAE, decoder(z, h_temb,c); terrain-conditioned AMP discriminator; Foothold Penalty | [LIT] ✔ |
| 결과 | beam +17.01%p, stairs +9.18%p 성공률 | [LIT] ✔ |

> 접근성: arXiv HTML(2606.06944v1) 및 abstract 페이지 모두 정상 접근, 정독 완료. 미확인/추론 부분은 본문에 [HYP] 명시.

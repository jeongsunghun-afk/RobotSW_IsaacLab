# 00 — SYNTHESIS: 평지 보행 데이터로 Parkour까지 자연스러운 움직임 학습하기

> 작성: team-lead (amp-parkour-research) · 2026-06-11
> 입력: `01_adversarial_methods.md`(lit-adversarial) · `02_latent_priors.md`(lit-priors) · `03_data_generation.md`(lit-datagen) · `04_codebase_feasibility.md`(code-feasibility) · `05_tgmp_deep_read.md`(lit-priors)
> 질문: "지형 모션 데이터가 없을 때, 평지 자연 보행 데이터(AMP)만으로 parkour 학습에 자연스러움을 넣을 수 있는가?"
> 증거등급: [FACT]=코드/측정 확인 · [LIT]=문헌 검증(arXiv ID 실존확인) · [HYP]=가설

---

## 1. 전제를 바꾸는 발견: 통합은 이미 되어 있다 [FACT]

`Go2-ParkourImitation-v0`가 이미 구현·등록·학습 이력까지 존재 (04 §1):
- `Go2ParkourImitationEnv` + `PPOAMP(PPOParkour)` + `OnPolicyRunnerParkourAMP`, 평지 trot/walk pkl 데모 포함
- `logs/rsl_rl/parkour_imitation_go2/`에 2026-05-27~28 run 9개
- **amp_obs는 `extras["amp_obs"]` 전용, policy obs(42-dim) 불변** → input팽창 dealbreaker 비해당, deploy 영향 0
- **AMP reward는 task reward의 clip(min=0) 바깥에서 additive** → signed reward 충돌 구조적 회피 (IMPLEMENTATION_STATUS의 WASABI clip 우려는 이 경로엔 비적용으로 정정)
- PPOAMP가 PPOParkour를 상속해 RMA encoder/DAGGER와 클래스 충돌 0 (ActorCriticRMA는 is_recurrent=False)

**따라서 질문은 "AMP를 parkour에 붙일 수 있나"가 아니라 "평지 데이터 기반 style 신호를 장애물 구간까지 확장할 수 있나"이다.**

## 2. 진짜 난제 (사용자 직관이 정확했던 부분) [FACT]

현재 AMP는 `_flat_env_mask`로 **평지 env에서만** 적용 중. 장애물로 확장하면:
> 평지 walk 데이터 기준으로 점프/계단 동작은 OOD → discriminator가 정당한 parkour 동작을 처벌 → task reward와 충돌.

## 3. 문헌의 수렴 (4개 조사 독립 결론)

### 3.1 직접 선례가 존재한다 — "불가능한 문제"가 아님 [LIT]
- **Motion Priors Reimagined** (arXiv:2505.16084): 평지 스킬을 험지 quadruped로 적응, 실로봇 실증
- **Terrain-Conditional AMP parkour** (MDPI Appl.Sci. 16:3448, 2026): AMP로 quadruped parkour 실증
- **CAMP** (arXiv:2509.21810, **Go2 동일 플랫폼**): 평지 합성 데이터 + gait-conditional discriminator
- **T-GMP** (arXiv:2606.06944, humanoid): terrain-conditioned discriminator의 가장 정밀한 레시피
- 단, 위 다수가 terrain 정보를 **policy에 conditioning** → 우리 input팽창 dealbreaker와 충돌. **conditioning은 discriminator 쪽에만 적용**하는 변형이 우리 경로 (discriminator는 학습시간 전용이라 obs/deploy 불변).

### 3.2 알고리즘 측 해법 3종 (05·01 합류)
| 레버 | 메커니즘 | 우리 비용 |
|------|---------|----------|
| **Terrain-conditioned discriminator** D(s,s′\|terrain) | "계단에선 계단 모션이 real" → 정당한 장애물 동작이 OOD 처벌 안 받음 | 중 (amp_obs에 terrain feature 추가 — policy 불변) |
| **WGAN/soft-boundary** (WASABI·HumanMimic) | disjoint 분포에서도 gradient 유지 → 평지 demo↔험지 행동 불일치에 LSGAN보다 강건 | **하 — WASABI 플래그 이미 구현됨** [FACT] |
| **Discriminator feature 선택** | base height z·global pose 제외, joint-space body-frame만 매칭 → 지형 불변 style | 하 (amp_obs 구성 변경). z-제외 ablation 직접 문헌 [미확인] → 우리 기여 포인트 |

보조: **multi-critic 분리** (RobotKeyframing, arXiv:2407.11562) — style reward를 별도 critic으로 → clip 경로 원천 분리. 현 구조로도 충돌은 회피되므로 후순위.

### 3.3 데이터 측 해법 — 알고리즘만으론 미완성 [LIT]
T-GMP 정독의 핵심 경고: **terrain-conditioned discriminator는 지형 데모가 선결조건.** 평지 데이터만 주면 현 `_flat_env_mask`와 사실상 동치로 퇴화. 지형 demo를 채우는 비용 오름차순:
1. **Self-imitation (권장)**: 1차 정책(현 parkour teacher)의 성공 rollout 중 품질 게이트 통과분을 demo로 승격 — PALo(arXiv:2503.04462, quadruped 실증). 신규 인프라 ~0. **우리 height-scan teacher가 곧 T-GMP의 privileged expert 역할** [FACT+LIT]
2. PARC(arXiv:2505.04002) 경량 차용: 생성기↔물리검증 부트스트랩 루프의 구조만 (풀 도입은 humanoid 파이프+1개월 A6000이라 비권장)
3. TO retargeting (OPT-Mimic 등): 동역학 일관but solver 비용 중
4. 단순 kinematic warping: 최저비용·최고위험 (parkour급 장애물에서 깨짐)

⚠️ **공통 치명 함정 — artifact 자기강화**: 우리 1차 정책은 pronk/split-jump artifact 보유 → 무필터 승격 시 나쁜 스타일이 demo가 되어 자기강화 붕괴. **품질 게이트가 생사 결정** — `measure_pronk_cost.py`의 기존 메트릭(airborne_frac, retk/flight, contact symmetry, CoT)을 게이트로 재사용 [FACT].

### 3.4 기각된 경로
- **Hierarchical latent-action 구조** (Tencent Lifelike Agility PMC, ASE/CALM, MCP): action space가 latent로 바뀜 → RMA+DAGGER·env 재작성 금지와 정면충돌. Tencent는 "평지 prior→지형 적응이 실로봇에서 된다"는 **정당화 근거로만** 인용 [LIT]
- **T-GMP 원본의 policy-side terrain embedding concat**: input팽창 → 거부, 기존 height-scan/RMA로 대체
- **순수 평지 외삽** (데이터 0 추가): 모든 문헌이 지형 demo 또는 conditioning을 사용 — 평지 demo만으로 장애물 style을 직접 일반화한 검증 사례 없음 [LIT]

## 4. 권장 로드맵 (비용 오름차순, 한 번에 하나씩)

> 모든 단계: policy obs/action 불변(input팽창 0), env 재작성 없음(additive), clip(min=0) 호환(양수형 AMP reward), Go2 deploy 영향 0.

| 단계 | 내용 | 변경 규모 | 검증 |
|------|------|----------|------|
| **Step 0** | `Go2-ParkourImitation-v0`에서 (a) amp_obs에서 base-height/global 성분 제거(terrain-invariant style feature), (b) WASABI WGAN 플래그 ON, (c) `_flat_env_mask` 완화 A/B | cfg+amp_obs 구성 (소) | measure_pronk_cost 메트릭 + goal rate 게이트 |
| **Step 1** | Self-imitation 데이터 부트스트랩: teacher 성공 rollout → 품질 게이트(airborne_frac/retk/CoT/대칭) → 지형 demo셋 생성 스크립트 | 스크립트 1개 (중) | 게이트 통과율, demo 분포 통계 |
| **Step 2** | Terrain-conditioned discriminator: amp_obs에 local terrain feature 추가(discriminator 전용) + Step 1 demo로 지형 manifold 채움 | amp_obs+disc 입력 (중) | 장애물 구간 style reward 분포, t-SNE |
| **Step 3 (선택)** | Multi-critic 분리(RobotKeyframing) 또는 gait-conditional discriminator(CAMP) | 알고리즘 (대) | Step 2 한계 확인 후에만 |

기존 트랙과의 관계: gait reward 스택(`_workspace/gait_design/00_DESIGN.md`)과 **상호 배타가 아님** — reward 스택은 "최소 footfall 제약", AMP 경로는 "분포 수준 style". 단 한 번에 하나씩 원칙 유지: Step 0은 reward 스택 실험과 별도 run으로.

## 5. 남은 불확실성 (정직성)
- Step 0의 "feature 선택만으로 mask 제거 가능"은 [HYP] — 직접 ablation 문헌 미확인, A/B로만 판정
- Self-imitation 품질 게이트의 임계값은 미정 — 너무 빡빡하면 demo 고갈, 느슨하면 artifact 유입
- T-GMP·Terrain-Conditional AMP는 각각 humanoid·타 quadruped — Go2 12-DOF 직접 검증은 CAMP뿐
- 기존 9개 parkour_imitation run의 결과 품질은 본 조사에서 미분석 (Step 0 전에 로그 확인 권장)

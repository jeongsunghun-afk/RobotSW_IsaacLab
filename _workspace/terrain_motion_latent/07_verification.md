# 07 — 문서 체인(01~06) 검증 보고서

> 작성: team worker `verifier` (Task #7) · 2026-06-12
> 대상: `_workspace/terrain_motion_latent/01~06` 문서 체인
> 방법: 06을 05/04에 대조(V1·V2·V6), 인용 3건 원문 fetch(V3), 코드 경로 ls/grep 실측(V4), 06 §3 표 점검(V5)
> 입장: 산출물 옹호 금지. 결함만 보고. 모든 결함 주장에 근거(문서 위치/코드 경로/fetch 결과) 첨부.

---

## (0) 종합 판정: ✅ **PASS**

**치명결함(critical) 0건. major 0건. minor 3건.**

06 최종 프레임워크는 05의 판정(①축약·②-1 조건부·②-2 가정 기각)과 P0~P2 개선사항을 **누락·왜곡 없이** 반영했고, hard 제약 8개가 본문 설계와 일치하며, load-bearing 인용 3건이 원문과 정확히 일치하고, 수정/불변 대상 코드 경로가 모두 실재한다. ②-1/Stage2b의 방법론적 긴장은 06이 **스스로 명시·격리**하여 정직성을 유지했다. 발견된 3건은 모두 minor(설계 명료성·우선순위 표기·명명)로, 진행을 막지 않는다.

---

## (1) V1~V6 결과

### V1. 05 판정·P0~P2 반영 여부 → ✅ PASS

**05 판정 → 06 반영 대조**

| 05 판정 | 06 반영 위치 | 일치 |
|---------|-------------|------|
| ① 모션 AE ✅ | (0)#1, Stage1 (conditional VAE 본체) | ✓ |
| ① 지형 AE(decoder) ❌ 과잉 | (0)#1 "지형 reconstruction decoder 폐기", (6)표 ① | ✓ |
| ① joint AE(대칭) ⚠️→conditional | Stage1 "대칭 joint AE ❌→conditional VAE", (6)표 ① | ✓ |
| ②-1 ⚠️ 조건부 | Stage2 ②-1, (3)#1, (6)표 ②-1 | ✓ |
| ②-2 원안 가정 기각 | (0)#4, Stage3, (6)표 ②-2 "frozen latent 자동 mapping 기각" | ✓ |
| D 제약 다수 충돌 | (5) 8개 제약 준수표 + §2-B TENSION | ✓ |

**P0~P2 → 06 반영 대조**

| 항목 | 05 내용 | 06 반영 위치 | 일치 |
|------|---------|-------------|------|
| P0-1 | 3종→conditional 1종 | (0)#1, Stage1, (5)#1, (6)① | ✓ |
| P0-2 | ②-2 = action-residual 1순위, latent-residual 비권고 | §2-B Stage2b(⚠️opt-in 강등), (3)#3 | △ 반영(우선순위 강등, minor-1) |
| P0-3 | latent disc-side, obs 팽창 회피 | (0)#2, Stage2 default, (5)#2 | ✓ |
| P0-4 | Go2-ParkourImitation-v0 additive | (0)#6, Stage2, (5)#1 | ✓ |
| P1-5 | self-imitation 루프 | Stage3 전체, (6)신규행 | ✓ |
| P1-6 | 품질게이트 ②-1·self-imitation 양쪽 | GATE-0/2/3, (3)#1 | ✓ |
| P1-7 | terrain curriculum + latent KL/범위 + residual clip | Stage3, (4) 리스크표 2b/3 | ✓ |
| P2-8 | 순수 IL 금지, tracking+AMP, coverage 내부 보간, STMR | Stage2 ②-1, (6)②-1 | ✓ |
| P2-9 | 분리형 baseline ablation 선행 | GATE-1 #4("미입증 가설" 표기), (4) Stage1 fallback | ✓ |
| P2-10 | mode collapse 모니터링, PSI | GATE-2 #3 (fore-hind/PSI) | ✓ |

→ 10개 항목 전부 반영. P0-2만 "1순위 권고"가 "opt-in default-OFF"로 격상 강등(minor-1, 아래 결함목록). **단 06이 이 강등을 §2-B에서 명시적으로 사유와 함께 기술**하여 왜곡이 아닌 의도적 설계 변경임을 투명하게 표기함.

### V2. Hard 제약 8개 ↔ 본문 일치 + §2 TENSION 논리 정합 → ✅ PASS

**8개 제약 체크리스트(06 §5) vs 본문 실제 설계**

| 제약 | §5 주장 | 본문 일치 근거 | 판정 |
|------|--------|---------------|------|
| #1 env 무재작성 | additive on Go2-ParkourImitation-v0 | Stage2 default가 기존 env 위 additive, crawl만 별도 PR(Stage3) | ✓ |
| #2 obs 팽창 금지 | latent=disc-only | Stage2 "policy 입력 concat 안 함", crawl obs는 critic-only/disc-side 한정(167행) | ✓ |
| #3 contact sensor 금지 | 신규 contact 무 | amp_obs=motion frame(dof/root/foot)만 | ✓ |
| #4 latent-action 치환 금지 | actor 불변+additive residual | §2-B 본문과 일치(아래 상세) | ✓ |
| #5 python 직접실행 금지 | ./isaaclab.sh -p | §2 명령어 전부 isaaclab.sh | ✓ |
| #6 conda env | isaac-parkour/isaac-5.1, 1회 실측 | §2 226행 실측 권고 | ✓ |
| #7 코어 수정 금지 | source/isaaclab/ 미수정 | DO-NOT-TOUCH 목록 명시 | ✓ |
| #8 reward clip | AMP additive clip 바깥 | Stage2 amp reward additive, task에 latent 미합산 | ✓ |

→ 체크리스트만 적고 본문이 위반하는 사례 **없음**.

**§2 TENSION(action-residual ↔ 제약#4) 처리의 논리 정합성**

- **구조 격리는 논리적으로 일관되고, 제약#4 준수 목적에는 충분하다.** 제약#4가 금지하는 것은 "action space를 latent로 *치환*하는 hierarchical 구조(ASE/CALM/MCP식)". 06 Stage2b는 기존 `ActorCriticRMA`가 12-dim joint를 **직접 출력**하는 구조를 유지한 채 12-dim residual head를 **additive로만** 얹는다(action = actor_out + a_res). decoder를 계층화하지 않으므로 "치환"이 아니다 → #4 핵심을 건드리지 않음. obs 46/action 12 차원 불변도 본문과 일치. default OFF + 사용자 sign-off 게이트로 위험 격리.
- **단, 효과(자연스러움) 주장에는 mechanistic 공백 존재(minor-2)**: 2505.16084의 자연스러움은 *frozen latent decoder가 생성한 base joint*에서 나오고 a_res는 그 위 보정이다. 06은 그 frozen decoder를 actor에서 제거했으므로 "residual을 frozen motion prior 방향으로 정규화"의 *기준 prior*가 본문상 미정의다. 즉 #4 **준수**는 충분하나, "2505.16084의 자연스러움 보존 효과를 취한다"는 **효과 주장은 근거 약함**. → 06 자신이 (4) 리스크표에서 "효과 無면 즉시 비활성"으로 선제 헤지하고 정직성 표기 #2로 team-lead 보고 대상 명시 → 위험 contained. minor.

### V3. 인용 스팟체크 (원문 HTML fetch 3건) → ✅ 3/3 일치

| 인용 | 06의 서술 | 원문 fetch 결과 | 판정 |
|------|----------|----------------|------|
| **2505.16084** Motion Priors Reimagined | "action-residual 12d + latent command 16d, w_res≈-0.1, frozen decoder; gap/crawl/jump 미해결"(286행) | html v2: latent command **16-D** + joint residual **12-D**, r_res=w_res·Σ(a_res)², **w_res=-0.1**(Table2), future work=**jumping/crawling/gaps/stepping stones/overhanging** | ✅ 정확 |
| **2505.12619** HIL | "수동 box paired, tracking+AMP 병행, PSI"(288행) | html v1: "manually position basic box geometries to replicate interaction affordances", tracking+adversarial 결합, **PSI** 사용, SMPL humanoid sim | ✅ 정확 |
| **2507.00677** Walk like Dogs | "Go2 실기 vMF-VAE 18D, kino-dynamic retargeting"(285행) | html v2: **vMF** latent, **18-D** (z∈ℝ¹⁸), **Unitree Go2 hardware**, MoE decoder(6 expert), kino-dynamic retargeting | ✅ 정확 |

→ 세 인용의 수치(16d/12d/-0.1/18D)·구조·미해결 영역 모두 원문과 일치. 과대인용·수치 오류 없음.

### V4. 코드 anchoring 정합 (ls/grep 실측) → ✅ PASS

**MODIFY 대상(존재해야 함) — 전부 실재:**
- `rsl_rl/rsl_rl/modules/amp_discriminator.py` ✓ (`class AMPDiscriminator`:12, `compute_amp_reward`:54 확인)
- `.../parkour_imitation/parkour_imitation_env.py` ✓ (`_flat_env_mask`:91, `_update_amp_obs_buf`, amp_obs 43/step·430 확인)
- `.../parkour_imitation/parkour_imitation_env_cfg.py` ✓
- `rsl_rl/rsl_rl/algorithms/ppo_amp.py` ✓
- `.../parkour_imitation/agents/rsl_rl_amp_cfg.py` ✓ (amp_weight=0.3, gradient_penalty_coef=5.0, disc_loss_type=ls_gan 실측 일치)
- (opt-in 2b) `rsl_rl/rsl_rl/modules/actor_critic_parkour.py` ✓ (`class ActorCriticRMA`:81, scan encoder dims[..,32]:93 확인)

**DO-NOT-TOUCH(존재) — 전부 실재:** `source/isaaclab/`, `.../direct/parkour/parkour_env.py`, `rsl_rl/rsl_rl/algorithms/ppo_parkour.py` ✓
**보조 인용:** `estimator.py`의 `DiscriminatorLSD`:75, `DiscriminatorContDIAYN`:101 실재 ✓
**NEW 대상(아직 없어야 함):** `motion_terrain_ae.py`, `train_motion_ae.py` — 부재 확인 ✓ (= 신규로 올바르게 표기)

→ 존재하지 않는 경로를 MODIFY/DO-NOT-TOUCH로 지목한 사례 **없음**.

### V5. 사용자 결정 5항목 표(06 §3) → ✅ PASS

5개 항목 모두 **권고 Default + 근거 + 뒤집을 경우 영향** 3열을 갖춤. 질문만 나열하고 끝난 항목 **없음**.

| # | Default | 근거 유무 |
|---|---------|----------|
| 1 ②-1 의도 | coverage 내부 보간+retargeting(해석2+3), 해석1 금지 | 05 §2 ✓ |
| 2 latent 위치 | discriminator-only | 05 §4-1, 04 §6-A ✓ |
| 3 residual 공간 | 기본 무, action-residual opt-in, latent-residual 비권고 | 05 §4·P0-2 ✓ |
| 4 paired 규모 | jump 우선+stair/gap 소량, 최소 수십 클립 | HIL 19클립, PULSE 99.8% ✓ |
| 5 OOD 우선순위 | stair→gap→jump-refine→crawl | 05 §6-5, crawl_plan ✓ |

### V6. 미확인·불일치의 사실 둔갑 여부 → ✅ PASS

- **MDPI Terrain-Cond AMP(403 미확인)**: 06 부록(291행) "본문 미확인(403), 개념 방향성만 인용", 정직성 표기 #3 재확인, §2 MODIFY표도 "(개념·02 §2)" 한정. → 확정 사실로 둔갑 **안 함** ✓
- **04 §7 obs 차원 불일치**: 06은 parkour policy obs를 일관되게 **46**으로 사용(분쟁 없는 값). amp history length(2 vs 10) 분쟁값은 06이 단정하지 않고 "amp_obs 43-dim/step"만 사용. §2 226행에서 "04 §7 불일치 항목과 별개"로 conda env 실측 권고하며 04 §7을 미해결로 인지. → 둔갑 **안 함** ✓
- **05 PULSE "VR tracking from-scratch보다 나쁘다"(가설/부분확인)**: 06은 이 강한 주장을 인용하지 않고 PULSE를 learnable prior 근거로만 사용. → 과대인용 **없음** ✓

---

## (2) 결함 목록

| # | severity | 위치 | 내용 | 수정안 |
|---|----------|------|------|--------|
| minor-1 | minor | 06 (0)#5·§2-B Stage2b·(3)#3 | 05 P0-2는 action-residual을 ②-2 구현의 **1순위(P0) 권고**로 둠. 06은 이를 **opt-in default-OFF Stage2b**로 강등하고, default ②-2 경로(disc+self-imitation)에는 residual이 **전혀 없음**. 06이 사유를 명시(제약#4 긴장)하여 왜곡은 아니나, "P0 우선순위" → "opt-in"의 격상 강등은 06만 읽는 독자에게 05의 우선순위를 오인시킬 여지. | (0) 요약에 "P0-2를 제약#4 긴장으로 opt-in으로 *재분류*함"을 한 줄 명시(현재는 (5)#4·§2-B에 분산). 이미 투명하므로 표기 강화만 권고. |
| minor-2 | minor | 06 §2-B(140행), (4) 2b 리스크 | Stage2b의 격리형 "normal actor + additive residual head"는 2505.16084의 frozen latent decoder(자연스러움의 실제 출처)를 제거함. 따라서 "residual을 frozen motion prior 방향으로 정규화"의 *기준 prior*가 본문상 미정의 → "2505.16084 자연스러움 보존 효과를 취한다"는 효과 주장 근거 약함. (제약#4 *준수*는 충분, 효과 주장만 약함) | Stage2b 활성 시 a_res가 무엇을 기준으로 정규화되는지(예: frozen AE decoder 출력 또는 직전 action) 1줄 정의 추가. 06 (4)표의 "효과 無면 즉시 비활성" 헤지로 위험은 이미 contained. |
| minor-3 | minor(cosmetic) | 06 14·93행 | "scan_encoder(187→32)"로 표기하나 실제 코드 심볼은 `scandot_encoder`(actor_critic_parkour.py:156, dims[..,32]). 입력 187→출력 32 의미는 정확. 04 §2.2에서 상속된 명명 부정확. | 구현 단계에서 `scandot_encoder`로 정정(설계 의미에는 영향 없음). |

---

## (3) 검증하지 못한 항목 (정직성)

1. **05의 "VR tracking 완화" 재대조**: PULSE Table 2 원문 수치 직접 재대조는 본 검증 범위에서 미수행(05가 이미 [가설/부분확인]으로 표기했고 06이 해당 강한 주장을 인용하지 않으므로 영향 없음).
2. **amp_history_length 실 런타임 값(2 vs 10)**: 04 §7의 미해결 항목. 코드 실행 없이는 확정 불가. 06이 이 분쟁값을 단정하지 않아 검증 부담 낮음 — 단 **구현 전 1회 실측 필요**(06도 권고).
3. **parkour_imitation 실제 conda env(isaac-5.1 vs isaac-parkour)**: 06 226행이 "1회 실측 확인" 권고. 본 검증에서 런타임 미실행으로 미확정.
4. **NEW 스크립트 부모 디렉토리 쓰기 가능성/`Go2MotionLib` 인터페이스 호환**: 시그니처 스케치만 검증, 실제 motion_lib API 인자 일치는 구현 시 validate-code 필요.
5. **설계의 학습 성능(수렴·성공률)**: 본 task는 문서 정합성 검증이며, 학습 결과 검증이 아님 — 범위 밖.

---

> **검증 요지**: 06은 05의 비판을 재반박 없이 충실 반영했고, 인용·코드·제약·정직성 표기 모두 정합. 유일한 방법론적 긴장(action-residual↔#4)은 06이 스스로 격리·명시·헤지하여 critical로 비화하지 않음. **PASS(minor 3건, 모두 표기/명료성 수준)**.

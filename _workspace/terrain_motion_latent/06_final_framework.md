# 06 — 최종 알고리즘 프레임워크 + IsaacLab 코드 anchored 구현 방안

> 작성: team worker `framework` (Task #6) · 2026-06-12
> 입력(전부 정독·반영): 01_motion_latent_papers.md · 02_terrain_imitation_papers.md · 03_ood_skill_papers.md · 04_codebase_assets.md · **05_algorithm_review.md(판정·P0~P2 그대로 반영)** · plans/parkour_crawl_plan.md
> 위치 기준: 모든 경로 `/home/lgb/IsaacLab` 절대경로. 본 문서는 **설계 문서**이며 코드 수정 없음.
> 입장: 05 점검 결과를 재반박하지 않는다. **단 1건의 방법론적 긴장(action-residual ↔ 제약#4)** 을 §2-B에서 명시하고 team-lead에 별도 보고한다(아래 ⚠️TENSION).

---

## (0) 최종 프레임워크 한 장 요약

### 핵심 결정 (05의 P0~P2를 그대로 confirm)

1. **3종 AE → "terrain-conditioned motion AE 1종"으로 축약** (05 P0-1). 지형 reconstruction decoder 폐기, 대칭 joint AE 폐기. 지형은 **conditioning encoder**(코드의 `scandot_encoder` 재사용)로만 진입.
2. **latent의 정책 영향 경로 = discriminator-side(amp_obs) terrain-conditioning** (05 P0-3). 이것이 **기본(default) 경로** → 제약#2(obs 팽창)·제약#4(latent-action 치환) 모두 회피.
3. **②-1 = "데이터 없는 OOD를 decode로 새로 생성"이 아니라 coverage 내부 보간 + retargeting + 품질게이트** (05 P2-8). 순수 IL 금지, tracking+AMP 병행(HIL 구조).
4. **OOD(②-2) = self-imitation 데이터 확장 루프**(05 P1-5)가 진짜 메커니즘. "frozen latent에 RL이 새 skill 자동 mapping"이라는 원안 가정은 기각(05 §3).
5. **action-space residual(2505.16084)** 은 **opt-in 고급 트랙(Stage 2b)** 으로만 제공 — 기본 프레임워크는 actor 구조 불변(아래 ⚠️TENSION).
   - ※ **우선순위 이력**: 05 점검은 이 action-residual을 ②-2 구현의 **P0(1순위) 권고**로 두었다. 본 프레임워크는 제약#4 TENSION 격리를 위해 이를 **opt-in default-OFF로 재분류**했다(강등 사유 §2-B). 즉 "선행연구 권고 1순위"이되 "본 코드베이스 제약상 기본 비활성"이라는 이중 위상임을 명시한다.
6. **모든 것을 `Go2-ParkourImitation-v0`(이미 존재·학습 이력) 위에 additive**로 구축(05 P0-4, 제약#1).

### 파이프라인 다이어그램 (텍스트)

```
┌─ STAGE 0 ── 데이터 제작 (선결, 모든 것의 병목) ───────────────────────────┐
│  보유: 평지 walk/run/trot/pace(pkl) + jump(지형정보 無)                    │
│  제작: 수동 장애물 배치 → (지형 height, 모션) paired pkl  [HIL 2505.12619] │
│  후처리: kino-dynamic / STMR retargeting [2507.00677 / 2404.11557]        │
│  ▶ GATE-0: measure_pronk_cost 물리/품질 게이트 통과분만 채택              │
└──────────────────────────────────────────────────────────────────────────┘
                                   │ (terrain,motion) pkl
                                   ▼
┌─ STAGE 1 ── terrain-conditioned motion AE 학습 (offline) ─────────────────┐
│  Encoder E(motion_frame, terrain_feat) → z  (Gaussian, dim≈16)            │
│  Decoder D(z, terrain_feat) → motion_frame   ← FROZEN 이후 단계           │
│  learnable prior R(z | s_proprio, terrain_feat)  [PULSE 2310.04582]        │
│  지형은 scandot_encoder(187→32) 패턴 재사용한 conditioning encoder           │
│  ▶ GATE-1: recon error↓ + latent KL 정상 + 평지 coverage 재현             │
└──────────────────────────────────────────────────────────────────────────┘
                    │ frozen D · prior R · (terrain,motion) demo
                    ▼
┌─ STAGE 2 ── 본학습: parkour_imitation 위 terrain-conditioned AMP ─────────┐
│  policy = 기존 ActorCriticRMA (46-dim obs, 12-dim joint)  ← 구조 불변     │
│  ②-1 경로: frozen D로 coverage-내부 reference 생성 → AMP expert buffer    │
│  핵심: AMP discriminator를 terrain-feat에 conditioning                    │
│        [MDPI Terrain-Cond AMP(개념) + HIL scene-cond disc]                │
│  AMP reward = clip 바깥 additive (제약#8) · flat_env_mask → terrain-가중  │
│  ▶ GATE-2: 평지 style 유지 + 장애물 success↑ + mode-collapse 미발생       │
│  (opt-in 2b) action-residual head 12d, w_res≈-0.1  ⚠️제약#4 긴장          │
└──────────────────────────────────────────────────────────────────────────┘
                    │ RL 성공 rollout
                    ▼
┌─ STAGE 3 ── OOD 확장: self-imitation 루프 (coverage 점진 확장) ───────────┐
│  순서: stair → gap → jump-refine → crawl(지형추가 선행)                    │
│  RL 성공 rollout → ▶GATE-3(measure_pronk_cost) → (terrain,motion) pkl 승격│
│         → AE 재학습(Stage1로 복귀) → demo/coverage 확장 → 다음 curriculum │
│  crawl = 천장 mesh + height_scan 마스킹 선행 (crawl_plan Phase1~5)         │
└──────────────────────────────────────────────────────────────────────────┘
        ▲───────────────── 루프 (improve → gate → add → retrain) ───────────┘
```

> **한 줄 요약**: 검증된 차용 요소(모션 AE + terrain-conditioned prior + AMP + self-imitation)만으로 골격을 짜되, 사용자 원안의 3개 미검증 주장(대칭 3종 AE / decode로 OOD 생성 / frozen latent 자동 mapping)은 각각 (1종 conditional AE / coverage-내부 보간 / self-imitation 루프)로 **대체**한다. latent의 정책 영향은 **discriminator-side**로 제한해 hard 제약을 모두 만족한다.

---

## (1) Stage 0~3 상세

### Stage 0 — 데이터 제작 + retargeting + 품질 게이트 (선결 과제, 최대 실무 리스크)

**근거**: 04 §3 — 지형 모션 **현재 0개**(parkour_imitation는 평지 trot/walk 12 pkl만). 05 §4-2 — "데이터 없이는 joint AE도 reference 생성도 불가, 이 갭이 ①②에 선행".

| 항목 | 내용 | 근거 |
|------|------|------|
| 입력 자산 | 평지 walk/run/trot/pace pkl (smr_mirror 18개·go2 7개) + jump(지형 無) | 04 §3 |
| paired 제작 | 수동 장애물 배치 → jump 모션 주변 height를 paired. **HIL의 "수동 box geometry 배치" 철학과 동일** | HIL 2505.12619 (02 §1) |
| retargeting | jump/지형 모션을 Go2 동역학에 정합: kino-dynamic(IK+MPC) 또는 STMR(SMR+TMR) | 2507.00677 §Stage1 (01 §1) / STMR 2404.11557 (02 §6) |
| pkl 포맷 | 기존 18-col(`root_pos3+root_euler3+joint_pos12`) 유지, `calc_motion_frame` 호환 | 04 §3 motion_lib.py:16-29 |
| **GATE-0** | `measure_pronk_cost` 메트릭(impact BW / saturation / CoT) + retargeting 물리 위반(관절한계·penetration·foot-slip) 0 | 04 §4.1, 메모리 pronk 측정 |

**검증 게이트 (→ Stage 1 진입 조건)**: GATE-0 통과 pkl이 **최소 임계 규모** 이상(아래 사용자 결정 §3 #4). 통과 못 한 모션은 폐기(prior 오염 방지, 05 §3-2).

> ⚠️ 리스크: paired 데이터가 희소하면(jump 소량) conditional AE manifold가 overfit. PULSE조차 99.8% coverage를 AMASS 전량으로 달성(05 §1-3). → Stage 3 self-imitation이 데이터 부트스트랩의 유일 현실 경로.

---

### Stage 1 — Terrain-conditioned motion AE 1종 (offline 사전학습)

**근거**: 05 P0-1(3→1 축약). 01 §7-A(분리형 baseline 우선). PULSE learnable prior(2310.04582) / CALM conditioning(2305.02195).

**구조 (대칭 joint AE ❌ → conditional VAE)**:
- **Encoder** `E(motion_frame, terrain_feat) → (μ, logσ²)` : motion 측만 latent화.
- **Decoder** `D(z, terrain_feat) → motion_frame` : 이후 단계에서 **freeze**.
- **Learnable prior** `R(z | s_proprio, terrain_feat)` : 고정 Gaussian 대신 상태/지형 조건부(PULSE). → 사용자 "(지형,모션) joint" 의도를 **비대칭 conditioning**으로 흡수(05 §1-3: 지형=조건, 모션=결과).
- **지형 conditioning encoder** = 코드의 `scandot_encoder(187→32)` 패턴 **재사용**(04 §2.2, 별도 지형 AE 신설 금지).

**latent 분포 선택 (05 §1-1이 설계자 택1로 남김 → 본 문서 확정)**:

| 후보 | 근거 | 권고 |
|------|------|------|
| **Gaussian + terrain/proprio-conditioned learnable prior** | PULSE(2310.04582) — conditioning과 자연 통합, KL로 연속·조밀 latent | **✅ 1순위 default** (conditional 구조에 가장 정합) |
| 구면 vMF (hyperspherical) | anchor 2507.00677(Go2 실기 18D) — 사족 steerable 실적 최다, mode-collapse 강건 | mode-collapse 발생 시 fallback |
| VQ-VAE 이산 코드북 | Lifelike 2308.15143(실사족) — "OOD=새 코드" 운영 직관적 | OOD를 명시적 코드로 다루고 싶을 때 대안 |

- latent dim ≈ **16**(2505.16084) ~ 18(2507.00677) → **16 채택**.
- 입력 motion_frame = 기존 amp_obs 43-dim 구성요소(dof_pos12+dof_vel12+root_height1+root_lin_vel3+root_ang_vel3+foot_pos_local12)와 정렬(04 §1.3) → discriminator 재사용 용이.

**검증 게이트 (→ Stage 2 진입 조건)** GATE-1:
1. reconstruction error 수렴(평지 모션 재현).
2. latent KL 정상 범위(붕괴·폭주 없음).
3. **평지 coverage 재현 테스트**: 학습 demo를 encode→decode 시 원본과 근사(motion-analyzer로 hip-pitch/base-z 분포 대조).
4. (ablation 권고, 05 P2-9) 분리형 baseline(모션 latent + 지형 obs) 대비 conditional의 이득을 **이후 입증** — 현 시점엔 "미입증 가설"로 표기.

---

### Stage 2 — 본학습: `Go2-ParkourImitation-v0` 위 terrain-conditioned AMP

**근거**: 05 P0-2/P0-3/P0-4. 04 §1.3(parkour_imitation 자산)·§6-A(disc-side 접점)·§2.1(AMP 융합공식). amp_parkour_research 00_SYNTHESIS(04 §4.1).

**기본(default) 경로 — discriminator-side latent conditioning**:
- **policy = 기존 `ActorCriticRMA`** (46-dim obs, 12-dim joint 출력) — **구조 불변**. 제약#1/#2/#4 동시 만족.
- **②-1 reference 생성**: Stage1 frozen decoder `D`로 **coverage-내부 보간**(05 §2 해석2) reference 생성 → STMR 후처리(05 P2-8) → GATE 통과분만 AMP **expert buffer**에 투입. **OOD 외삽 decode 금지**(05 §2 해석1 = ❌).
- **terrain-conditioned discriminator**: `AMPDiscriminator` 입력 = 기존 amp_obs(43/step) **+ terrain_feat**(disc 전용). "평지엔 이 style, 장애물엔 저 style"을 지형 기하에 맞춰 전환 → 현재 `_flat_env_mask`가 평지에만 style을 주던 한계(04 §1.3, 장애물 구간 style=0) 해소.
- **AMP reward = clip 바깥 additive**(제약#8): `total = task_reward + amp_weight(0.3)·mask·disc_reward` (04 §2.1). `_flat_env_mask` → **terrain-conditioned weighting**으로 확장(04 §6-F).
- latent은 **policy 입력에 새 차원으로 concat하지 않음** → 제약#2 회피(05 §4-1 허용케이스 1).

**하이퍼파라미터(근거 인용 값)**:
- `amp_weight=0.3`, `reward_coef=2.0*0.02`, `gradient_penalty_coef=5.0`, `discriminator_hidden_dims=[1024,512]`, `disc_loss_type=ls_gan`(기존 parkour_imitation cfg, 04 §2.1).
- learnable prior KL weight: PULSE식(2310.04582) — 소량부터 튜닝.

**검증 게이트 (→ Stage 3 진입 조건)** GATE-2:
1. **평지 style 보존**: 평지 env에서 기존 parkour_imitation 대비 자연스러움 비퇴행(measure_pronk_cost).
2. 장애물 success rate(goal_reached) 상승.
3. **mode-collapse 미발생**: fore-hind 협응 / PSI 지표(02 §8-D, 05 P2-10) 모니터링.

#### ⚠️TENSION — opt-in Stage 2b: action-space residual (05 P0-2 충실 반영 + 긴장 명시)

> 05 P0-2는 "②-2를 action-space residual(2505.16084: 16d latent command + 12d joint residual + w_res≈-0.1)로 구현"을 1순위 권고하며, 이것이 latent-space residual(PULSE)보다 **제약#4 충돌이 "가장 적다"**(=0 아님)고 판정한다.
>
> **방법론적 긴장(team-lead 보고 대상)**: 2505.16084의 원형은 *frozen low-level decoder가 latent command로 base joint를 생성하고 거기에 a_res를 더하는* **계층적(hierarchical) latent-action 구조**다. 이는 제약#4(ASE/CALM/MCP식 hierarchical latent-action 치환 거부)·제약#1(RMA+DAGGER 단일루프·env 무재작성)과 **부분 충돌**한다. 05 자신도 "충돌 가장 적음"이지 "충돌 없음"이라 하지 않았다.
>
> **본 프레임워크의 해소책**: 기본 경로는 actor를 계층화하지 **않는다**. Stage 2b를 원할 경우, 기존 `ActorCriticRMA`의 12-dim joint 출력에 **소형 residual head(12d)를 additive로만** 얹고(latent→decoder로 action을 *치환*하지 않음), `w_res≈-0.1`(2505.16084 Eq.3)로 정규화한다. obs 46-dim·action 12-dim **불변** 유지.
> **정규화 기준 prior의 정의(명시)**: 본 격리형에는 2505.16084의 *frozen latent decoder*(원 논문에서 자연스러움의 실제 출처)가 **없다**. 따라서 a_res penalty `w_res·Σ(a_res)²`의 기준은 frozen motion prior가 아니라 **기존 `ActorCriticRMA` actor 자신의 출력(a_res=0)** 이다 — 즉 "terrain-conditioned discriminator(Stage2)로 이미 style이 빚어진 base action에서의 *이탈*을 penalize"하는 구조다.
> **효과 주장 톤다운(가설)**: 그러므로 "2505.16084의 자연스러움 보존 효과를 그대로 취한다"는 주장은 **본 격리형에서 직접 보증되지 않는 가설**이다. 기대 효과(base style 유지하며 지형 적응)는 base actor의 style 품질에 의존하며, 효과가 확인되지 않으면 (4) 리스크표대로 **즉시 비활성**한다. 제약#4 *준수*는 충분(action-space 치환 없음)하되, *효과*는 미입증으로 표기.
> **사용자 sign-off 필요**: Stage 2b는 default OFF. 활성화 = 사용자가 위 긴장을 수용한다는 명시 결정(§3 #3) 이후에만.

---

### Stage 3 — OOD 확장: self-imitation 루프 (원안 ②-2의 진짜 메커니즘)

**근거**: 05 P1-5/P1-6(원안에서 빠진 가장 중요한 조각). 조사C #7(self-imitation). 메모리 amp_parkour self-imitation 처방. 2505.16084 미해결 영역(gap/stepping-stone/overhang/jump/crawl) = 사용자 OOD와 정확히 일치(05 §3-1).

**루프 (improve → gate → add → retrain)**:
1. Stage 2 정책으로 OOD 지형 RL rollout 수집.
2. **GATE-3**: `measure_pronk_cost` 품질 게이트 통과분만 채택(05 P1-6, 게이트 없으면 prior 오염).
3. 통과 rollout을 (terrain, motion) pkl로 **승격** → demo set 확장.
4. **AE 재학습**(Stage 1 복귀) → latent coverage 확장 → discriminator·prior 갱신.
5. terrain curriculum 다음 tier로 진행(parkour 기존 curriculum 재사용, 05 P1-7).

**OOD task 우선순위 (05 §6-5 = "가장 가치 높은 1개에 집중")**:

| 순서 | task | 지형 존재 | 비고 |
|------|------|----------|------|
| 1 | **stair** | ✅ 기존 parkour terrain | 기존 자산 재사용, walk/trot 재배열에 가까워 self-imitation 성공률↑ |
| 2 | **gap** | ✅ 기존 | jump 모션 보유가 강점(05 §C). 2505.16084 미해결 영역 → 차별화 |
| 3 | **jump-refine** | ✅ 기존 hurdle | 보유 jump 모션 품질 강화(SF-TIM 2408.00486 imagination/measurement 참고) |
| 4 | **crawl** | ❌ **지형 추가 선행 필요** | 천장 mesh + height_scan 마스킹(crawl_plan Phase1~5). **OOD 중 가장 마지막** |

**crawl 선결 조건 (crawl_plan.md 반영)**:
- 천장(overhang) box mesh 추가 + 양옆 벽(우회 차단) — `parkour_terrains.py`에 `parkour_crawl_terrain` 신규(crawl_plan §2, Phase1).
- **height_scan 오염 처리**: 천장 cell에서 ray가 천장에 먼저 맞아 "지면이 솟았다" 오인 → 마스킹(옵션a) + ceiling clearance obs(옵션b) (crawl_plan §4, Phase2/4). **단 obs 추가는 제약#2 주의** → critic-only(asymmetric) 또는 disc-side로 한정.
- edge mask z 상한 필터(crawl_plan §5, Phase2).
- crawl reference 모션 부재 → Stage3 self-imitation으로 부트스트랩하거나 별도 crawl demo 제작(crawl_plan §8).

**검증 게이트** GATE-3(각 라운드): 승격 모션이 GATE-0 동급 품질 + AE 재학습 후 평지 coverage 비퇴행(catastrophic forgetting 방지).

---

## (2) 코드 구현 방안 (파일 수준, 코드 anchored)

> 04 §6 접점 후보 A~G를 활용. **클래스/함수 시그니처 스케치만** 제시(전체 코드 작성 금지). 전 경로 절대경로.

### 새로 만들 파일 (NEW)

| 파일 | 목적 | 시그니처 스케치 | 근거 접점 |
|------|------|----------------|----------|
| `/home/lgb/IsaacLab/rsl_rl/rsl_rl/modules/motion_terrain_ae.py` | terrain-conditioned motion AE 모듈 | `class TerrainConditionedMotionAE(nn.Module): def encode(self, motion_frame, terrain_feat)->(mu,logvar); def decode(self, z, terrain_feat)->motion_frame; def prior(self, s_proprio, terrain_feat)->(mu_p,logvar_p); def kl_loss(...); def recon_loss(...)` | 04 §6-C(DiscriminatorLSD/ContDIAYN 재사용 가능) |
| `/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/train_motion_ae.py` | AE offline 사전학습 스크립트 | `def main(): lib=Go2MotionLib(...); ae=TerrainConditionedMotionAE(...); for batch: loss=recon+β·kl; save_ckpt()` | 04 §3 motion_lib.py 인터페이스 |
| `/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/generate_reference_from_ae.py` | ②-1 reference 생성(coverage-내부 보간+STMR+GATE) | `def generate(ae_ckpt, terrain_cond, n): z=interp_in_coverage(...); frames=ae.decode(z,terrain_cond); frames=retarget(frames); if quality_gate(frames): write_pkl(...)` | 04 §6-E, 05 P2-8 |
| `/home/lgb/IsaacLab/scripts/reinforcement_learning/rsl_rl/promote_self_imitation.py` | Stage3 성공 rollout → pkl 승격 | `def promote(rollouts): passed=[r for r in rollouts if measure_pronk_cost(r) <= thr]; write_terrain_motion_pkl(passed)` | 04 §6-E, 조사C #7 |

### 수정할 파일 (MODIFY — additive only)

| 파일 | 변경 | 불변 규칙 | 근거 |
|------|------|----------|------|
| `/home/lgb/IsaacLab/rsl_rl/rsl_rl/modules/amp_discriminator.py` | `AMPDiscriminator` 입력에 terrain_feat 추가(disc 전용). `compute_amp_reward` 시그니처에 terrain conditioning 인자 | input/output shape는 **disc 내부만** 확장, policy 무관 | 04 §6-A, MDPI Terrain-Cond AMP(개념·02 §2) |
| `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/parkour_imitation_env.py` | `_update_amp_obs_buf`에 terrain_feat 추가(amp_obs, extras 경로). `_flat_env_mask`→terrain-conditioned weighting. (opt) self-imitation rollout export hook | **policy obs(46) 절대 불변**. amp_obs는 `extras["amp_obs"]` 전용 | 04 §1.3·§6-A·§6-F |
| `.../parkour_imitation/parkour_imitation_env_cfg.py` | AMP cfg 추가: `terrain_cond_disc: bool=False`, demo dir, ae_ckpt path, latent_dim | `@configclass` 유지, 상속 필드 불변 | 04 §1.3 |
| `/home/lgb/IsaacLab/rsl_rl/rsl_rl/algorithms/ppo_amp.py` | `update_amp(expert_batch, policy_batch)`에 terrain-cond 배치 전달. (opt-in 2b) residual penalty 항 `w_res·Σa_res²` | clip 바깥 additive 경로 유지, DAGGER loop 불변 | 04 §2.1·§6-D, 2505.16084 Eq.3 |
| `.../parkour_imitation/agents/rsl_rl_amp_cfg.py` (또는 해당 agent cfg) | 하이퍼파라미터: `latent_dim=16`, `w_res=-0.1`, `kl_weight`, `terrain_cond=True`, `amp_weight=0.3` | 한 번에 1~2개씩 튜닝(hyperparam-worker 규칙) | 2505.16084 / 2310.04582 |
| (opt-in 2b만) `/home/lgb/IsaacLab/rsl_rl/rsl_rl/modules/actor_critic_parkour.py` | `ActorCriticRMA`에 12-dim residual head(플래그 가드, **default OFF**). 출력 = 기존 action + a_res | **obs 46·action 12 불변**, latent→action 치환 금지 | ⚠️TENSION §2-B |

### 건드리면 안 되는 파일 (DO NOT TOUCH)

- `/home/lgb/IsaacLab/source/isaaclab/**` — 코어(제약#7).
- `.../direct/parkour/parkour_env.py`·`parkour_env_cfg.py` — baseline reward/obs/terrain 구조(제약#1, parkour_imitation가 상속). **crawl 지형 추가 시에만** crawl_plan 범위 내 additive(별도 PR 권고).
- `/home/lgb/IsaacLab/rsl_rl/rsl_rl/algorithms/ppo_parkour.py` 의 RMA/DAGGER 단일루프 코어 — distill 로직 불변.
- deploy actor 46-dim obs 경로 — **차원 변경 금지**(제약#2).

### 학습 명령어

```bash
# Stage 1 — AE 사전학습 (offline). 직접 conda 사용 시 parkour 서브프로젝트는 isaac-parkour env(메모리),
#           본 IsaacLab 메인은 isaac-5.1. 안전 경로는 항상 ./isaaclab.sh -p (제약#5/#6).
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_motion_ae.py \
  --motion_dir <terrain_motion_pkl_dir> --latent_dim 16

# Stage 2 — 본학습 (terrain-conditioned AMP, additive)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-ParkourImitation-v0 --num_envs 4096 \
  --logger wandb --wandb-project IsaacLab-terrain-latent

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2-ParkourImitation-v0 --num_envs 32

# Stage 3 — self-imitation 승격 (rollout 수집 후)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/promote_self_imitation.py --run <log_dir>
```
> ⚠️ 검증 필요: parkour_imitation가 메인 IsaacLab(`source/isaaclab_tasks`)에 있어 통상 `isaac-5.1`로 도는지, parkour 서브프로젝트(`isaac-parkour`)로 도는지 1회 실측 확인(04 §7 불일치 항목과 별개). 본 문서는 `./isaaclab.sh -p`를 표준으로 둔다.

---

## (3) 사용자 결정 필요 항목 (각각 권고 default + 근거; 사용자 뒤집기 가능)

> 05 §6의 미해결 질문 5개에 대해 "권고 default + 근거"를 확정. **표의 모든 default는 사용자가 뒤집을 수 있는 결정사항**이다.

| # | 결정 항목 | **권고 Default** | 근거 | 뒤집을 경우 영향 |
|---|----------|-----------------|------|-----------------|
| 1 | **②-1 "AE로 reference 생성"의 의도** | **coverage 내부 보간 + retargeting(해석2+3)**. "미관측 OOD 지형을 decode로 새로 생성(해석1)"은 **금지** | AE 외삽=비물리 모션, coverage 밖 garbage(05 §2). OOD는 Stage3 self-imitation으로 | 해석1 채택 시 IL 오염 위험 → GATE 강화 필수, 비권고 |
| 2 | **latent을 policy 입력 vs discriminator-only** | **discriminator-only(amp_obs)** | 제약#2 회피 가장 안전(05 §4-1, 04 §6-A) | policy concat 시 deploy obs 팽창=제약#2 위반. 굳이 넣으려면 기존 encoder/history-distill로 46-dim 불변 보장 |
| 3 | **residual을 둘 것인가 / 어느 공간** | **기본=residual 없음(disc-side)**. 원하면 **action-residual opt-in(Stage2b)**, latent-residual(PULSE)은 **비권고** | latent-residual=제약#4 충돌(05 §4·P0-2). action-residual도 ⚠️긴장 존재 → 사용자 sign-off 필요 | latent-residual 수용 시 메모리상 과거 거부 이력과 충돌 — 명시 동의 필요 |
| 4 | **paired 데이터 규모/종류** | **jump 우선 + stair/gap 소량**, mirror aug 포함 **최소 수십 클립급**(HIL 19클립 기준) | 희소 pair는 conditional manifold overfit(05 §1-3, PULSE 99.8%는 전량) | jump만 제작 시 stair/gap latent coverage 부족 → Stage3 의존도↑ |
| 5 | **OOD 목표 우선순위** | **stair → gap → jump-refine → crawl** (crawl 최후, 지형추가 선행) | 기존 지형 재사용 + 보유 모션 근접도(05 §6-5, crawl_plan). 가치 높은 1개 집중 | crawl 우선 시 천장 mesh+height_scan 작업이 전체를 블로킹(crawl_plan Phase1~5) |

---

## (4) 리스크 / Fallback (stage별 최대 1~2개)

| Stage | 최대 리스크 | Fallback |
|-------|------------|----------|
| **0 데이터** | paired 데이터 희소 → conditional manifold overfit/mode-collapse (05 §1-3) | (a) Stage1을 분리형 baseline(모션 latent + 지형 obs, 01 §7-A)으로 회귀 — joint/conditional 이득 미입증 시. (b) Stage3 self-imitation으로 데이터 부트스트랩 우선 |
| **1 AE** | conditional이 분리형 대비 이득 無(미입증 가설, 05 P2-9) | ablation으로 분리형 baseline에 회귀. latent 분포는 Gaussian→vMF(mode-collapse 시)→VQ(OOD 코드화 시) 순 교체 |
| **2 본학습** | terrain-cond discriminator가 OOD를 "non-expert"로 처벌(평지 demo 기준, 04 §4.1) / mode-collapse(02 §8-D) | (a) WGAN/soft-boundary(WASABI 이미 구현, 04 §4.1)로 완화. (b) disc feature에서 base-height/global 제외. (c) action-residual opt-in 보류, disc-side만 유지 |
| **2b residual** | ⚠️제약#4 긴장 / residual 폭주로 스타일 붕괴(2505.16084 w_res=0이면) | default OFF 유지. 활성 시 w_res≈-0.1 + residual clip. 효과 無면 즉시 비활성(리스크 격리) |
| **3 OOD** | self-imitation prior 오염(저품질 rollout 승격, 메모리·05 P1-6) / crawl height_scan 오염(crawl_plan §4) | GATE-3 강화(measure_pronk_cost 임계 상향). 평지 coverage 비퇴행 회귀 테스트. crawl은 마스킹(옵션a)+critic-only obs(옵션b)로 격리 |

---

## (5) Hard 제약 준수 체크리스트 (04 §5의 8개)

| 제약 | 본 설계의 준수 방식 |
|------|--------------------|
| **#1 env 무재작성·additive only** | `Go2-ParkourImitation-v0` 위 additive(Stage2). 새 latent env 신설 없음. crawl 지형만 별도 PR로 분리(crawl_plan) |
| **#2 policy obs 팽창 금지** | latent은 **discriminator-only(amp_obs, extras 전용)**. deploy actor 46-dim 불변. crawl ceiling obs는 critic-only/disc-side로 한정 |
| **#3 contact sensor obs 금지** | 기존 contact_filt 외 추가 없음. amp_obs는 motion frame(dof/root/foot_pos)만, 신규 contact sensor 미도입 |
| **#4 latent-ACTION 치환 금지** | 기본 경로 actor 구조 불변(latent→action 치환 없음). Stage2b는 action에 *additive residual*만(치환 아님) + ⚠️긴장 명시·사용자 sign-off 게이트 |
| **#5 python 직접 실행 금지** | 모든 학습 `./isaaclab.sh -p`(§2 명령어) |
| **#6 conda env** | parkour 서브프로젝트=isaac-parkour, 메인=isaac-5.1. `./isaaclab.sh -p`로 통일, 1회 실측 확인 권고 |
| **#7 코어 수정 금지** | `source/isaaclab/**` 미수정. rsl_rl/parkour_imitation 레벨만 |
| **#8 reward clip(min=0)** | latent/style 신호 = AMP additive(clip 바깥). task reward에 latent 직접 합산 안 함 → floor death 회피 |

---

## (6) 원안 ①/②-1/②-2 대비 변경점 요약표

| 원안 | 원안 내용 | **최종 프레임워크** | 변경 이유 (근거) |
|------|----------|-------------------|-----------------|
| **①** | 지형 AE + 모션 AE + (지형,모션) 대칭 joint AE **3종** | **terrain-conditioned motion AE 1종** (지형=conditioning encoder, scandot_encoder 재사용) | 지형 reconstruction decoder 불필요(지형=주어지는 입력), 대칭 joint AE 직접 선례 부재·희소 데이터 overfit (05 P0-1, §1-2/1-3) |
| **②-1** | AE decoder로 (미관측 지형 포함) reference dataset 생성 → 순수 IL | **coverage 내부 보간 + STMR retargeting + GATE → tracking+AMP 병행**(순수 IL ❌) | AE 외삽=비물리, 순수 IL=covariate shift(HIL "tracking 단독 실패") (05 P2-8, §2) |
| **②-2** | frozen latent + RL → OOD를 RL feedback이 latent에 **자동 새 skill mapping** | **(기본) terrain-conditioned discriminator로 style 유도 + (핵심) self-imitation 데이터 확장 루프**. action-residual은 opt-in | "frozen latent 자동 mapping" 보증한 선행연구 없음. 최근접 2505.16084는 action-residual 사용 + gap/crawl/jump를 미해결로 명시 (05 §3, P1-5) |
| (공통) | — | **품질 게이트를 ②-1 생성·Stage3 승격 양쪽 의무화** + terrain curriculum + latent KL/범위 제약 | prior 오염·latent 이탈·mode-collapse 방지 (05 P1-6/P1-7, 조사C 실패모드) |
| (신규) | 원안에 없음 | **self-imitation 루프가 OOD의 진짜 메커니즘** (improve→gate→add→retrain) | 원안에서 빠진 "가장 중요한 조각" — gap/crawl/jump는 residual만으로 안 됨 (05 P1-5) |

---

### 부록 — 근거 논문/코드 인덱스 (load-bearing)

- **2507.00677** Walk like Dogs (Go2 실기 vMF-VAE 18D, kino-dynamic retargeting) — Stage1 latent/retargeting
- **2505.16084** Motion Priors Reimagined (action-residual 16d latent+12d joint, w_res=-0.1, frozen decoder; gap/crawl/jump 미해결) — Stage2b/OOD 근거
- **2310.04582** PULSE (conditional VAE + learnable prior, KL; RL+IL latent 혼합 경고) — Stage1 prior/②-1·②-2 분리
- **2505.12619** HIL (수동 box paired, tracking+AMP 병행, PSI) — Stage0 데이터/Stage2 구조
- **2404.11557** STMR (SMR+TMR retargeting, box 위 모션) — Stage0/②-1 후처리
- **2308.15143** Lifelike (VQ-VAE 이산, 실사족 OOD) — latent 분포 fallback
- **MDPI 2076-3417/16/7/3448** Terrain-Conditional AMP — **본문 미확인(403)**, **개념 방향성만** 인용(구체 아키텍처/Go2적용 인용 금지)
- 코드: `parkour_imitation_env.py`·`amp_discriminator.py`·`ppo_amp.py`·`actor_critic_parkour.py`·`estimator.py(DiscriminatorLSD/ContDIAYN)`·`motion_lib.py` (전부 04 문서 경로)

> **정직성 표기**: (1) "conditional이 분리형보다 낫다"=미입증 가설(ablation 필요). (2) action-residual의 제약#4 긴장=본 문서 §2-B에서 명시, team-lead 보고. (3) MDPI Terrain-Cond AMP=본문 미확인, 개념만. (4) parkour_imitation conda env=1회 실측 확인 권고.
</content>
</invoke>

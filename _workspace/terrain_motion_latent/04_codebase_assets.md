# 04 — 코드베이스/데이터 자산 + 제약 정리

> 작성: code-scout (team go2-latent-parkour, task #4) · 2026-06-12
> 목적: latent AE(autoencoder) 기반 terrain-motion 파이프라인 설계자가 "지금 무엇이 있고, 무엇을 건드리면 안 되는지"를 한 문서로 파악하게 한다.
> 범위: 읽기 전용 조사. 모든 경로는 `/home/lgb/IsaacLab` 기준 절대경로. 코드 수정 없음.
> 증거 등급: **[FACT]** = 코드 직접 확인 · **[DOC]** = 환경 CLAUDE.md/docstring 확인 · **[?]** = 확인 불가/불일치(아래 §7 참조).

---

## 1. 환경 자산 (Direct RL env)

### 1.1 등록된 task 목록 [FACT]

| Task ID | 디렉토리 | env 클래스 | runner cfg | 성격 |
|---------|---------|-----------|-----------|------|
| `Go2-Parkour-Direct-v0` | `direct/parkour/` | `Go2ParkourEnv` | `Go2ParkourPPORunnerCfg` | **baseline** (Extreme Parkour 파생, RMA+DAGGER) |
| `Go2-Parkour-Direct-SPO` | `direct/parkour/` | `Go2ParkourEnv` | `...SPO...` | parkour + SPO variant (default off) |
| `Go2-Parkour-Direct-LCP` | `direct/parkour/` | `Go2ParkourEnv` | `...LCP...` | parkour + LCP variant |
| `Go2-Parkour-Direct-MoE` | `direct/parkour/` | `Go2ParkourEnv` | `...MoE...` | parkour + MoE actor |
| `Go2-Parkour-Symmetry` | `direct/parkour/` | `Go2ParkourEnv` | `...Symmetry...` | parkour + L/R mirror data-aug |
| **`Go2-ParkourImitation-v0`** | `direct/parkour_imitation/` | `Go2ParkourImitationEnv` | `Go2ParkourImitationPPOAMPRunnerCfg` | **parkour + AMP 하이브리드 (이미 존재·학습 이력 有)** |
| `Go2-Imitation-v0` | `direct/go2_imitation/` | `Go2ImitationEnv` | `Go2ImitationPPORunnerCfg` | 평지 AMP+steering (MimicKit TaskSteeringEnv 재구현) |
| `Go2-Imitation-WASABI-v0` | `direct/go2_imitation/` | `Go2ImitationEnv` | `...WASABIPPORunnerCfg` | 위와 동일 env, WASABI(WGAN) runner |
| `Go2-Imitation-Tracking-v0` | `direct/go2_imitation_tracking/` | (tracking env) | tracking runner | go2_imitation 복제 후 command를 body-frame 속도추종으로 변경 |

> **핵심 발견**: `Go2-ParkourImitation-v0`가 `parkour/`가 아닌 **별도 디렉토리 `direct/parkour_imitation/`**에 이미 구현·등록되어 있고, `logs/rsl_rl/parkour_imitation_go2/`에 2026-05-27~28 학습 run 7개 존재 [FACT]. 즉 "parkour env에 AMP 붙이기"는 이미 landed 상태이며, latent 파이프라인은 이 위에 additive로 얹는 것이 자연스럽다.

### 1.2 parkour baseline env (`direct/parkour/`) [FACT/DOC]

- **env 본체**: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` (1878줄)
- **cfg**: `.../parkour/parkour_env_cfg.py` (705줄) — 클래스 `ParkourEnvCfg`
- **terrain**: `.../parkour/parkour_terrains.py`, terrain_type=`generator`, `PARKOUR_TERRAINS_CFG`. 서브지형 클래스 상수 존재 (`TERRAIN_CLASS_FLAT=0` … `TERRAIN_CLASS_CRAWL=7`).
- **mdp 모듈**: `.../parkour/mdp/` (symmetry.py 등)

**Observation 구조 (dict 기반, runner가 obs_groups로 사용)** [FACT, parkour_env_cfg.py:416–434]:

| obs group | 차원 | 내용 |
|-----------|------|------|
| `policy` (proprio) | **46** (= 42 + 4) | projected_gravity(3)+commands(3)+joint_pos(12)+joint_vel(12)+prev_actions(12) 계열. cfg에 `num_proprio=46`, `observation_space=42+4` |
| `scan` (height_scan) | **187** | RayCaster height scan |
| `priv_explicit` | 6 | root_lin_vel_b(3)·2.0 + root_ang_vel_b(3)·0.25 |
| `priv_latent` | 37 | base_friction(1)+foot_friction(8)+base_mass(1)+base_com(3)+joint_stiffness_ratio(12)+joint_damping_ratio(12) |
| `history` | 420 | history_len(10) × num_proprio(42) |
| critic total | 272 | policy+scan+priv_explicit+priv_latent |

> 주의: cfg 주석은 `num_proprio=42+4=46`, deploy actor obs는 survey_v2 기준 46-dim. height scan(187)은 검증 완료(정상). **contact_filt(4발 debounced boolean)이 이미 policy obs에 포함됨** (survey_v2 anchor) — contact 정보는 obs 추가 금지 제약과 별개로 *기존* 보유분.

**Command / goal 구조** [FACT, parkour_env_cfg.py:595–700]:
- `command_cfg`: forward-only — `lin_vel_x_range=[0.3,1.5]`, lin_vel_y=0, ang_vel=0 (Genesis original parkour). `resampling_time_s=4.0`.
- **goal/waypoint 시스템**: `num_goals=8`, `num_future_goal_obs=2`(lookahead), `goal_distance=1.0m`, `next_goal_threshold=0.2`, `reach_goal_delay=0.1s`. terrain 함수가 build time에 `terrain_goals` 채움.

**Reward 구조 (14-term, Extreme Parkour 정렬)** [FACT, parkour_env_cfg.py:607–656]:
- 양수: `tracking_goal_vel`(1.5), `tracking_yaw`(0.5), `feet_gait_pairing`(**0.0=비활성**)
- penalty: `lin_vel_z_l2`(-1.0), `ang_vel_xy_l2`(-0.05), `orientation_l2`(-1.0), `dof_acc_l2`(-2.5e-7), `collision`(-10), `action_rate_l2`(-0.1), `delta_torques`(-1e-7), `torques_l2`(-1e-5), `hip_pos`(-0.5), `dof_error_l2`(-0.04), `feet_stumble`(-1.0), `feet_edge`(-1.0), `feet_dragging`(-0.1 뒷발만)
- opt-in(현재 0 또는 소량): `air_time_cap`(-0.0), `contact_duty_deficit`(-0.0), `positive_work`(-1e-3)
- **total reward에 clip(min=0) 적용** [FACT, 메모리·survey_v2] → signed reward 항은 floor에 죽음. AMP는 이 clip *바깥*에서 additive로 합산되어 충돌 회피(§2.1 참조).

### 1.3 parkour_imitation 하이브리드 env (`direct/parkour_imitation/`) [FACT]

- **cfg**: `.../parkour_imitation/parkour_imitation_env_cfg.py` — `ParkourImitationEnvCfg(ParkourEnvCfg)`. parkour의 모든 필드(terrain/robot/sensor/reward/obs 46) **상속·불변**, AMP 파라미터만 추가.
- **env**: `.../parkour_imitation/parkour_imitation_env.py` — `Go2ParkourImitationEnv`.
- **AMP discriminator obs (43-dim/step)** [FACT, env_cfg:27–54, env.py:14–15]:
  `dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel_b(3) + root_ang_vel_b(3) + foot_pos_local(12)` = 43
- **amp_obs 전달 경로**: `extras["amp_obs"]` (shape [N, history×43]), `extras["terminal_amp_obs"]`로 runner에 넘김. **policy obs(46)에는 전혀 들어가지 않음** → deploy/input 불변 [FACT, env.py:131–151].
- **`_flat_env_mask`** [FACT, env.py:26,91,125]: `[N]` bool, True=평지(TERRAIN_CLASS_FLAT) env. AMP reward와 discriminator gradient 모두 평지 env에서만 적용(나머지 마스킹). **= 현재 장애물 구간엔 style 신호 0** (이게 latent/terrain-conditional 확장의 핵심 표적).
- **모션**: `amp_motion_pkl="imitation/go2"` (디렉토리 → 다수 pkl auto-discover).

### 1.4 go2_imitation 평지 AMP env (`direct/go2_imitation/`) [FACT/DOC]

- **cfg**: `Go2ImitationEnvCfg` — terrain_type=`plane`(평지 only), episode 10s, 200Hz/50Hz.
- **policy obs = 44+6 = 50** [FACT, cfg:60]: projected_gravity(3)+steering(local_tar_dir 2 + tar_speed 1 + local_face_dir 2)+joint_pos_offset(12)+joint_vel(12)+actions(12) (+6 옵션). (CLAUDE.md는 44로 기술 — §7 불일치.)
- **AMP disc obs = 49/step** [FACT, cfg:65]: dof_pos(12)+dof_vel(12)+root_height(1)+root_lin_vel(3)+root_ang_vel(3)+foot_pos_local(12)+root_rot_tan_norm(6). `num_amp_observations=10` → flatten 490.
- **motion_lib**: `Go2MotionLib` (`.../go2_imitation/motion_lib.py`, 450줄). MimicKit MotionLib 인터페이스: `sample_motions / sample_times / calc_motion_frame`.
- **현재 사용 데이터셋**: `MOTION_FILES_DIR = imitation/smr_mirror_pkl` [FACT, cfg:30] (주석엔 imitation/go2도 후보로 기재).
- steering task: tar_dir(랜덤 각도)+tar_speed[0.5,3.0]+face_dir, 4~7s마다 재샘플. 항상 RSI reset.

> **Tracking 변형** (`go2_imitation_tracking/`): go2_imitation 복제 후 command를 steering→body-frame 속도추종(vx,vy=0,yaw_rate)으로 교체. obs 50→48. 동일 motion_lib/심링크.

---

## 2. 알고리즘 자산 (rsl_rl)

위치: `/home/lgb/IsaacLab/rsl_rl/rsl_rl/`

### 2.1 algorithms/ [FACT]

| 파일 | 핵심 클래스 | latent 설계 접점 |
|------|-----------|-----------------|
| `ppo.py` | `PPO` | 기본 |
| `ppo_parkour.py` | `PPOParkour` | **RMA-style**: priv_encoder(37→20)+history_encoder를 DAGGER로 distill(`update_dagger`, 별도 optimizer). optional `estimator`(priv_explicit 예측) 보유. is_recurrent=False. ← parkour baseline 알고리즘 |
| `ppo_amp.py` | `PPOAMPBase(PPO)`, **`PPOAMP(PPOParkour)`** | AMP 융합. `PPOAMP`가 `PPOParkour` 상속 → RMA/DAGGER와 충돌 0. `update_amp(expert_batch, policy_batch)` + `update_dagger`. `disc_loss_type`/`disc_reward_type` = `ls_gan`/`bce`/`wgan` 선택 가능. **AMP reward는 task reward의 clip 바깥에서 additive** (parkour total=task + amp_weight·flat_mask·disc_reward) |
| `ppo_parkour_original.py` | 원본 보존 | 참고용 |
| `distillation.py` | distillation | student-teacher |

**parkour_imitation AMP 융합 공식** [FACT, rsl_rl_amp_cfg.py:13–14, 117–128]:
```
total = task_reward + amp_weight(0.3) * flat_env_mask.float() * disc_reward
```
disc cfg: `discriminator_hidden_dims=[1024,512]`, `gradient_penalty_coef=5.0`, `reward_coef=2.0*0.02`, `disc_loss_type=ls_gan`, `disc_reward_type=ls_gan`. `task_reward_lerp` 계열은 OnPolicyRunnerParkourAMP에서 **무시**됨.

### 2.2 modules/ [FACT]

| 파일 | 클래스 | 역할 / latent 접점 |
|------|--------|-------------------|
| `actor_critic_parkour.py` | `ActorCriticRMA`, `StateHistoryEncoder` | baseline actor-critic. `StateHistoryEncoder`=1D-Conv temporal encoder(proprio×10→20). scan_encoder(187→32), priv_encoder(37→20). actor input = proprio + priv_explicit + priv_latent_enc + scan_latent. **← latent를 흘려넣기 자연스러운 지점(encoder 계열)** |
| `actor_critic_parkour_moe.py` | `ActorCriticRMAMoE(ActorCriticRMA)` | MoE actor (multi-expert) |
| `actor_critic_parkour_original.py` | 원본 | 참고 |
| `amp_discriminator.py` | `AMPDiscriminator` | MLP(input→hidden[1024,512]→1). `compute_amp_reward` (ls_gan/bce/wgan), gradient penalty, `EmpiricalNormalization`(WGAN reward). **← terrain-conditional/latent-conditional discriminator 확장 지점** |
| `estimator.py` | `Estimator`, **`Discriminator`, `DiscriminatorLSD`, `DiscriminatorContDIAYN`** | priv_explicit estimator + **latent-skill discovery용 discriminator(LSD/DIAYN) 이미 존재** ← latent skill 설계 직접 재사용 후보 |
| `depth_backbone.py` | `RecurrentDepthBackbone`, `StackDepthEncoder`, `DepthOnlyFCBackbone58x87` | 깊이 CNN backbone (**parkour baseline은 미사용** — height-scan MLP 경로) |
| `actor_critic.py` / `_recurrent.py` / `_cnn.py` | `ActorCritic` 등 | 범용 |
| `symmetry.py` | mirror data-aug | Go2-Parkour-Symmetry용 |
| `rnd.py`, `student_teacher*.py` | RND, distill | 보조 |

---

## 3. 모션 데이터 현황 [FACT]

루트: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation/imitation/`

| 디렉토리 | 포맷 | 내용 | 사용처 |
|---------|------|------|--------|
| `go2/` | **.pkl** (7개) | go2_walk0~3, go2_run, go2_trot, go2_pace | go2_imitation 후보(현재 미사용) |
| `smr_mirror_pkl/` | **.pkl** (18개) | walk/walk1/walk2/walk_turn + trot0/1 + run0/1/2, 각 `_mirror` 쌍 | **go2_imitation 현재 사용** (cfg MOTION_FILES_DIR) |
| `new_dataset_walk/` | .txt (15개) | walk 모션 raw | 변환 소스 |
| `new_dataset_50/`, `new_dataset_smr/`, `new_dataset_single/`, `new_dataset_smr_mirror/`, `smr_mirror_new_go2/` | .txt | 데이터셋 변형들 | 변환 파이프라인 중간물 |
| `convert_pkl_to_npz.py`, `convert_smr_to_npz.py` | 스크립트 | pkl↔npz/smr 변환 | 데이터 가공 |

**parkour_imitation 전용 모션** [FACT]: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/imitation/go2/` — **trot0/1, walk, walk1/2, walk_turn (+각 _mirror), 총 12개 .pkl**. 즉 **평지 trot/walk 데모만** 존재 (점프/계단/gap 등 지형 모션은 **없음**).

**PKL 프레임 포맷 (18 값/frame)** [FACT, motion_lib.py:16–29]:
- `[0:3]` root_pos(x,y,z), `[3:6]` root_euler(rpy), `[6:18]` joint_pos(12 DOF).
- 속도/발위치: finite-difference + Go2 FK 자동 계산.
- `calc_motion_frame` 반환: root_pos[N,3](world), root_quat[N,4](wxyz), root_lin_vel/ang_vel[N,3](body), dof_pos/vel[N,12], foot_pos_local[N,4,3](base-local, [FL,FR,RL,RR]).

> **종류 요약**: 보유 = walk · trot · run · pace · walk_turn (모두 **평지**). 부재 = jump · stair · gap · 계단 등 **지형 모션 전무**. → latent/terrain-conditional 설계 시 "지형 데모 부재"가 선결 제약 (§4 amp_parkour 조사 결론과 일치).

---

## 4. 기존 조사문서 요약

### 4.1 `_workspace/amp_parkour_research/00_SYNTHESIS.md` (2026-06-11)
- "평지 보행 데이터(AMP)만으로 parkour에 자연스러움을 넣을 수 있는가"가 주제. **통합은 이미 완료**(`Go2-ParkourImitation-v0` 존재, amp_obs=extras 전용·policy 불변·clip 바깥 additive).
- 진짜 난제: 현 AMP는 `_flat_env_mask`로 **평지 env에만** 적용 → 장애물 확장 시 평지 demo 기준 점프/계단은 **OOD → discriminator가 정당한 parkour 동작 처벌**.
- 문헌 수렴: Motion Priors Reimagined / Terrain-Conditional AMP / CAMP(Go2) / T-GMP. 해법 3축 = ①terrain-conditioned **discriminator**(policy 아닌 disc에만 conditioning) ②WGAN/soft-boundary(WASABI 이미 구현) ③disc feature 선택(base-height/global 제외).
- **치명 함정**: 지형 demo는 self-imitation(teacher 성공 rollout 승격)으로 채우되 pronk/split-jump artifact 자기강화 위험 → **품질 게이트 필수**(measure_pronk_cost 메트릭 재사용).
- 권장 로드맵 Step0~3: disc feature 정제+WASABI ON+mask 완화 → self-imitation demo → terrain-conditioned disc → (선택)multi-critic/gait-conditional.

### 4.2 `_workspace/parkour_research_survey_v2.md` (2026-06-10)
- 코드 anchored 재조사(v1 "약하다" 거부 후). 하드 제약 = **deploy-obs 무팽창 / env 무재작성** 강제.
- Anchor: 네트워크=`ActorCriticRMA`(MLP[512,256,128], depth CNN 미사용, 1D-Conv StateHistoryEncoder, GRU/transformer 없음), 알고리즘=`PPOParkour`(RMA+DAGGER 단일루프), deploy actor 46-dim, **contact_filt 이미 obs 포함**.
- 진단: 3-leg = **reward-positive local optimum** + RL_calf dead-action 가능(plasticity collapse).
- 권장: CaT(IROS24, 3줄)·KAIST IPO(T-RO24) 공개코드로 constraint 채널 → reward clip과 직교. 선행으로 Shrink+Perturb+LayerNorm(plasticity 복원). preprint(GPO 등)는 격리.

### 4.3 `_workspace/gait_design/00_DESIGN.md` (2026-06-10)
- 문제[FACT]: 평지 bounding 수렴 / 장애물 4발 pronk(stair 27.7% airborne) / split-jump 전후협응 부재 / CoT stair 1.49×. 안전은 임계 이하 → safety penalty NO-GO.
- Root cause: reward에 **footfall-pattern 신호 부재**. Extreme Parkour는 footfall 미강제 → pronk는 예견된 실패모드. mirror aug는 L/R만 강제(fore-hind 미강제).
- 권장 2축: **축A** fore-hind 협응 reward(Ding et al. 2024, Go2 검증) · **축B** anti-pronk/min-stance(clip-safe). phase-clock obs 추가 = 입력팽창 BLOCKED. 순수 additive reward만.

---

## 5. 사용자 HARD 제약 (설계 시 위반 금지) [지시·메모리 기반]

1. **baseline = Extreme Parkour 파생 코드 유지.** env 전면 재작성 금지 — **additive 개선만** 허용.
2. **정책(policy) obs input 팽창 금지.** deploy actor obs 차원을 늘리는 변경 불가.
   - **예외**: AMP의 `amp_obs`(disc 전용 입력, `extras["amp_obs"]`)는 policy 입력이 아니므로 팽창 제약 비해당. terrain feature를 **discriminator 쪽**에만 넣는 것은 허용.
3. **contact sensor obs 추가 금지** (sim-to-real). (단 기존 contact_filt는 이미 obs에 있던 보유분.)
4. **Hierarchical latent-ACTION 치환 금지** (메모리): action space를 latent로 바꾸는 ASE/CALM/MCP식 구조는 RMA+DAGGER·env 무재작성과 정면충돌 → 거부됨.
5. **python 직접 실행 금지.** 반드시 `./isaaclab.sh -p ...` 사용.
6. **conda env**: parkour 서브프로젝트(Isaaclab_Parkour)는 `isaac-parkour`. 본 IsaacLab은 `isaac-5.1`(기존 isaac는 kit symlink 깨짐).
7. **`source/isaaclab/` 코어 직접 수정 금지.**
8. (참고) total_reward clip(min=0)은 A/B 공통 의도된 설계 — silent bug 아님. signed reward 항은 이 floor에 죽으므로 latent/style 신호는 **clip 바깥(AMP additive) 또는 constraint 채널**로 넣어야 안전.

---

## 6. "새 latent AE 파이프라인을 붙인다면 자연스러운 접점" (파일 경로 수준 — 설계 아님)

> 아래는 **후보 위치만** 나열. 구체 설계는 task #5/#6 설계자 몫.

| 접점 후보 | 파일 경로 | 근거 (왜 자연스러운가) |
|----------|----------|----------------------|
| **(A) Discriminator 입력 측 (terrain/latent conditioning)** | `/home/lgb/IsaacLab/rsl_rl/rsl_rl/modules/amp_discriminator.py` + `.../parkour_imitation/parkour_imitation_env.py`(`_update_amp_obs_buf`, amp_obs 43-dim 구성) | amp_obs는 policy 불변·disc 전용 → 입력팽창 제약 비해당. latent code/terrain feature를 여기 추가 가능 |
| **(B) Latent encoder를 actor 보조 입력으로** | `/home/lgb/IsaacLab/rsl_rl/rsl_rl/modules/actor_critic_parkour.py`(`ActorCriticRMA`, scan/priv/history encoder) | 이미 다수 encoder가 latent를 actor에 concat하는 구조. **단 policy obs 팽창 제약 주의** — 기존 priv_latent/scan_latent 경로(deploy 시 history로 distill) 안에 녹여야 안전 |
| **(C) Latent-skill discriminator 재사용** | `/home/lgb/IsaacLab/rsl_rl/rsl_rl/modules/estimator.py`(`DiscriminatorLSD`, `DiscriminatorContDIAYN`) | LSD/DIAYN 류 latent-skill discriminator가 **이미 구현되어 있음** → AE/skill latent 설계에 직접 차용 가능 |
| **(D) AMP 융합/loss 지점** | `/home/lgb/IsaacLab/rsl_rl/rsl_rl/algorithms/ppo_amp.py`(`PPOAMP.update_amp`, `update_dagger`) | latent reconstruction loss나 latent-conditioned style reward를 융합할 자리. clip 바깥 additive 경로 유지 |
| **(E) 지형 demo 생성(self-imitation)** | `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/imitation/go2/`(현재 평지 pkl 12개) + `.../go2_imitation/imitation/convert_*.py` + motion_lib.py(18-col pkl 포맷) | latent AE 학습용 지형 모션이 **현재 전무** → teacher rollout을 pkl로 승격하는 파이프라인이 데이터 측 접점. 품질 게이트 필수 |
| **(F) flat_env_mask 확장** | `.../parkour_imitation/parkour_imitation_env.py`(`_flat_env_mask`/`_update_flat_env_mask`) | 현재 style 신호가 평지 env에만 적용. latent/terrain-conditional로 장애물 구간까지 확장하는 toggle 지점 |
| **(G) reward 측(footfall/coordination, 보완)** | `.../parkour/parkour_env_cfg.py`(`reward_scales`) + `parkour_env.py`(`_get_rewards`) | latent style과 상호배타 아님(gait_design 권장). 단 clip(min=0) 주의 → 양수형/constraint 채널 |

---

## 7. 확인 불가 / 불일치 (정직성)

- **go2_imitation policy obs 차원**: cfg `observation_space=44+6=50` vs CLAUDE.md 표 "44". cfg가 +6 옵션(`include_rel_track_obs` 등) 반영한 듯 — 실제 런타임 값은 코드 실행 확인 필요. [?]
- **parkour_imitation AMP history length**: cfg `amp_history_length=2`(env_cfg.py:49) vs env/disc 주석·amp_obs flatten 차원이 **430 = 10×43**(10 history) 기준으로 기술됨(env.py:9,100 주석). 둘이 불일치 — 실 런타임 history 깊이는 코드 실행으로 확정 필요. amp_obs **per-step 43-dim**은 일관 확인. [?]
- **parkour policy obs 42 vs 46**: cfg `observation_space=42+4`, `num_proprio=42+4=46`, survey_v2 anchor "deploy actor 46-dim". 42는 height_scan/priv 제외 proprio 코어, 46이 deploy actor. obs_groups 분해는 runner에서 최종 결정. [FACT, 단 명명 혼동 주의]
- 9개 → 실측 **7개** parkour_imitation run 디렉토리(2026-05-27~28). 로그 품질 내용은 본 task 범위 외(미분석).

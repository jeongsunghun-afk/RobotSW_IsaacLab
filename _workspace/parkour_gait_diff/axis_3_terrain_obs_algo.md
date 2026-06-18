# Axis 3: Terrain / Observation / Algo — Parkour A vs B

> Scope: terrain geometry & curriculum, observation/command, episode/termination, algo & network.
> Cross-axis tie-ins (reward weight, actuator) handled by worker-1 / worker-2.
> Hard constraints respected: `height_scan` not ranked; contact-sensor obs not proposed; `actuator_mode=2` not ranked; no "regression" framing; no "direct vs manager" framing.

---

## 3.1 Terrain (sub-parameter level)

### 3.1.1 Generator-level

| Param | A (direct/parkour) | B (extreme_parkour teacher) | Status | Jump relevance |
|---|---|---|---|---|
| Tile size (x×y) | `20.0 × 4.0` m  (`parkour_env_cfg.py:45`) | `16.0 × 4.0` m  (`extreme_parkour/config/parkour.py:5`) | DIFF | A is 25% longer along x → more obstacles per tile, longer travel per episode → harder per-episode load |
| num_rows × num_cols | `11 × 40` (`parkour_env_cfg.py:47-48`) | `10 × 40` (B) | ≈SAME | minor; curriculum granularity ~10 steps each |
| horizontal_scale | `0.05` (`parkour_env_cfg.py:49`) | `0.08` teacher / `0.10` student (`parkour_student_cfg.py:20`) | DIFF | A grids finer (5 cm vs 8 cm); B coarser by design to work around `isaac-sim` issue #2187 |
| vertical_scale | `0.005` | `0.005` | SAME | — |
| slope_threshold | `0.75` | `1.5` | DIFF | A asserts edges harder; auxiliary, low ranking |
| curriculum | `True` | `True` | SAME | — |
| max_init_terrain_level | **`3`** (`parkour_env_cfg.py:434`) | **`2`** (`default_cfg.py:48`) | DIFF | A spawns up to row 3 (≈ difficulty 0.30) on iter 0; B caps at row 2 (≈ 0.22). A's "easy pool" is *less easy* — more agents see obstacles immediately |
| advance / regress threshold | `0.8 / 0.4 * expected_dist` (`parkour_env.py:1270-1271`) | `0.8 / 0.4 * threshold` (`parkour_event.py:143-144`) | SAME | identical rule |
| max-level wrap | `randint(0, max_level)` on reach (`parkour_env.py:1276-1280`) | `randint(0, max_terrain_level)` on reach (`parkour_event.py:149-150`) | SAME | — |
| `random_difficulty` (train) | n/a — pure curriculum | `False` (default; PLAY uses `True`) | SAME | — |
| `apply_roughness` field | NOT present in A configs | `True` on every sub-terrain (`extreme_parkour/config/parkour.py:18,24,32,40,47`) | **DIFF** | **B applies ±2–6 cm Brownian-style ground noise to every tile**; A surfaces are pristine smooth boxes. Smooth-only training under-trains "walk over textured ground" gait. |

### 3.1.2 Sub-terrain types & proportions

| Sub-terrain | A proportion | A function | B proportion | B function |
|---|---|---|---|---|
| flat (no obstacle) | **0.10** | `parkour_jump_hurdle_terrain` w/ `flat=True` (`parkour_env_cfg.py:61-76`) | **0.20** | `ExtremeParkourHurdleTerrainCfg` w/ `apply_flat=True` (`parkour.py:30-37`) |
| hurdle | 0.20 | central raised box, open side passages (`parkour_terrains.py:51-156`) | 0.20 | central raised band, open side passages (`extreme_parkour_terrians.py:117-168`) |
| step (asc/desc) | 0.20 | `MeshParkourStepTerrainCfg` | 0.20 | `ExtremeParkourStepTerrainCfg` |
| gap / pit | **0.30** | `MeshParkourGapTerrainCfg` (trench-only, no side pit) | 0.20 | `ExtremeParkourGapTerrainCfg` (trench **+ lateral pit**, narrow center walk) |
| stair (cycle) | 0.20 | `MeshParkourStairTerrainCfg` | — | absent |
| extreme parkour (lateral-jitter inclined stones) | absent | — | **0.20** | `ExtremeParkourTerrainCfg` (`parkour_terrain` w/ stones + incline_height + lateral jitter) |
| All other (stones, beam, crawl, slope, zigzag, rough_blocks) | **0.0** (registered but inactive) | `parkour_env_cfg.py:143-244` | not registered | — |

Key observation:
- **A allocates 0.30 to gap** (highest single weight); B allocates 0.20.
- **A allocates 0.10 to pure-walk flat**; B allocates 0.20 (twice the walking gradient signal).
- A has no `extreme_parkour` (stepping-stone-with-incline) terrain. B has no `parkour_stair`.
- A's inactive terrain types (stones/beam/crawl/slope/zigzag/rough_blocks) are dead code with proportion 0.0 — irrelevant to current behavior.

### 3.1.3 Difficulty interpolation per sub-terrain — gap width vs leg reach

Go2 max stride reach during walking ≈ 0.30 m; sprint stride ≈ 0.35–0.40 m.
At gap > ~30 cm the robot cannot keep both supporting feet on a platform — only a hop/jump succeeds.

| Sub-terrain | Param | A | B | Jump-required? |
|---|---|---|---|---|
| **gap** | gap width range | linear `(0.05, 0.50)` m  (`parkour_env_cfg.py:110`) | `0.1 + 0.7 * difficulty` → `(0.10, 0.80)` m  (`parkour.py:22`) | **A diff=0**: 5 cm (walk). **A diff=1**: 50 cm (jump-only). **B diff=0**: 10 cm (walk). **B diff=1**: 80 cm (jump-only). A is *easier* at hardest gap; B is much harder. |
| gap | side pit structure | **trench-only** (full-width depression between platforms, walkable corridor) | **trench + lateral pit** (rows 102-103 in `extreme_parkour_terrians.py` zero out `:mid_y+rand_y-half_valid_width` and `mid_y+rand_y+half_valid_width:`) → narrow walkable strip in y | B forces lateral precision in addition to gap-crossing; A only forces forward jump |
| gap | platform length between gaps | `(1.2, 1.6)` m (`parkour_env_cfg.py:111`) | `(0.8, 1.5)` m  (`parkour.py:19`) | A has longer recovery platforms → easier landing-then-walk; B compresses → faster successive jumps |
| **hurdle** | height range | linear `(0.05, 0.30)` m  (`parkour_env_cfg.py:82`) | `(0.1 + 0.1d, 0.15 + 0.25d)` m → diff=0: `[0.10, 0.15]`, diff=1: `[0.20, 0.40]` (`parkour.py:28`) | **B's easy hurdle (10–15 cm) is taller than A's easy hurdle (5 cm)**; A row-0 has near-zero hurdle (essentially flat). At hardest, B reaches 40 cm — much taller than A's 30 cm. |
| hurdle | side-passage half-width | default (~`0.4–0.8` m) — central hurdle + ~1.2 m open side passages on 4 m terrain (`parkour_terrains.py:21-24`) | `half_valid_width = (0.4, 0.8)` → central hurdle 80–160 cm wide, open side passages ~1.2 m | SAME structural pattern (central hurdle, side bypass possible if policy steers around) |
| hurdle | hurdle thickness along x | from `MeshParkourHurdleTerrainCfg` default | `stone_len = 0.1 + 0.3*difficulty` → 10–40 cm  (`parkour.py:28` via `extreme_parkour_terrains_cfg.py:28`) | B's thicker hurdles at hard difficulty are harder to clear in one stride |
| **step** | step height | `(0.10, 0.45)` linear  (`parkour_env_cfg.py:97`) | `0.1 + 0.35*difficulty` → `(0.10, 0.45)` (`parkour.py:43`) | **SAME** — both reach 45 cm at hardest, well beyond Go2 standing hip (≈30 cm). MUST jump. |
| step | step-segment x-length | `x_length_range=(0.4, 0.8)` m | `x_range=(0.3, 1.5)` m | B has wider variance → less periodic, harder to memorize a hop cadence |
| **stair** (A-only) | step depth × height | `width=(0.25, 0.40)`, `height=(0.05, 0.20)` (`parkour_env_cfg.py:120-124`) | n/a | mild — 5–20 cm steps walkable. 20% of A's terrains. |
| **extreme parkour stones** (B-only) | x/y stone offset + incline | `x=(-0.1, 0.1+0.3d)`, `y=(0.2, 0.3+0.1d)`, `incline_height=0.25d` (`parkour.py:46-52`) | n/a | B exposes stepping-stones with sloped tops + lateral spacing — pushes lateral foot placement + balance |

### 3.1.4 Apply-flat / start-platform spawn

| Param | A | B | Status |
|---|---|---|---|
| Start platform length | `platform_length` in each cfg (default 2.0–2.5 m) | `platform_len = 2.5` (`parkour_terrain_generator_cfg.py:15`) | ≈SAME |
| Start patches (`FlatPatchSamplingCfg`) | `num_patches=2`, `patch_radius=0.5`, `max_height_diff=0.05` — explicit in every sub-terrain | uses `terrain_origins` only | DIFF (cosmetic; A places spawn points more permissively) |
| Roughness on start platform | none (smooth) | yes (rough noise applied to whole tile) | DIFF |

---

## 3.2 Observation / Command

### 3.2.1 Proprio dim composition

| Component | A (42 dim, `parkour_env.py:852-863`) | B (53 dim, `observations.py:68-82`) | Jump relevance |
|---|---|---|---|
| `root_ang_vel_b` | **absent** from policy obs (placed in `priv_explicit`, critic only) | `× 0.25` in proprio | **HIGH** — actor cannot feedback-stabilize body angular rate during jump take-off / landing. Critic sees it; policy does not. |
| roll, pitch (IMU) | implicit via `projected_gravity_b` (3 dim) | explicit `imu_obs = (roll, pitch)` (2 dim) | LOW — `projected_gravity` is information-equivalent for tilt |
| delta_yaw (target − heading) | 1 dim (wrapped) | `0*delta_yaw` (1, masked) + `delta_yaw` (1) | A drops the zero-channel; functionally same |
| next_delta_yaw | 1 dim | 1 dim | SAME |
| commanded vx | `commands[:, 0:1]` 1 dim | `0*commands[:,0:2]` (2 masked) + `commands[:, 0:1]` 1 dim | ≈SAME |
| **terrain-class flags** | absent | `env_idx_tensor` (is-NOT-flat) + `invert_env_idx_tensor` (is-flat) 2 dim (`observations.py:59-60, 76-77`) | **MED** — B tells the policy directly "this episode is flat terrain". A's policy cannot distinguish flat vs hurdle terrain from obs and must rely on `height_scan` + history. |
| joint_pos − default | 12 | 12 | SAME |
| joint_vel × 0.05 | 12 | 12 | SAME |
| prev action / current action | 12 (`self._actions`) | 12 (`action_history_buf[:, -1]`) | SAME |
| contact-fill | **forbidden / absent** per memory constraint | 4 dim foot contact filter | Cannot add to A. Acknowledged. |
| **proprio total** | **42** | **53** | A is missing 11 dims (ang_vel 3 + terrain flag 2 + foot contact 4 + ang_vel-related 2). |
| History length | 10 frames (`parkour_env_cfg.py:406`) | 10 frames (`parkour_mdp_cfg.py:58`) | SAME |

### 3.2.2 Scan / privileged / history

| Group | A | B | Status |
|---|---|---|---|
| height scan grid | 1.6 m × 1.0 m @ 0.10 m → 17 × 11 = **187** cells | 1.65 m × 1.5 m @ 0.15 m → 12 × 11 = **132** cells | DIFF (do not rank per constraint) |
| `priv_explicit` | `lin_vel_b*2 (3) + ang_vel_b*0.25 (3)` = 6 (`parkour_env.py:898-904`) | `lin_vel_b*2 + 0*lin_vel + 0*lin_vel` = 9 (`observations.py:115-118`) | A places ang_vel here (not in proprio) — see 3.2.1 |
| `priv_latent` | `foot_friction (8) + base_mass (1) + base_com (3)` = 12 (`parkour_env.py:918-925`) | `body_mass (1) + body_com (3) + friction (1) + joint_stiff_ratio (12) + joint_damp_ratio (12)` = 29 (`observations.py:120-136`) | DIFF — B exposes per-joint actuator gain ratios (RMA conditioning). Not jump-causal. |
| proprio history `_obs_history_buffer` | `(N, 10, 42)`, encoder = `StateHistoryEncoder` | `(N, 10, 53)`, encoder = `StateHistoryEncoder` | SAME architecture; A 11 dims slimmer |

### 3.2.3 Command / standing-still

| Param | A (`parkour_env_cfg.py:549-558`) | B (`parkour_mdp_cfg.py:22-34`) | Jump relevance |
|---|---|---|---|
| `lin_vel_x_range` | `[0.3, 1.0]` m/s | `(0.3, 0.8)` m/s | A allows 25% higher top-speed command → encourages faster gait |
| `lin_vel_y_range` | `[0.0, 0.0]` | not commanded directly | SAME |
| `ang_vel_range` | `[0.0, 0.0]` | n/a | SAME |
| `heading` | NOT commanded (yaw via goal-tracking only) | `heading = (-1.6, 1.6)` rad, converted to ang_vel via `heading_control_stiffness=0.8` | **MED** — B explicitly commands a heading target; A has no heading channel. A's only yaw signal is `delta_yaw` (target − robot) in obs + `tracking_yaw` reward. |
| `small_commands_to_zero` | absent | `True` with `lin_vel_clip=0.2`, `ang_vel_clip=0.4` (`parkour_mdp_cfg.py:30-33`, `uniform_parkour_command.py:64-66`) | **MED** — B explicitly trains stand-still at low commanded velocity; A's policy never sees a "stand still" command target. |
| Resampling time | `6.0 s` (`parkour_env_cfg.py:558`) | `(6.0, 6.0)` s (`parkour_mdp_cfg.py:25`) | SAME |
| Time-out / cmd-zero coupling | none | small-cmd → zero clamp in command term | A always has `commands[0]≥0.3` m/s → policy never has a "0 cmd → don't move" gradient |

---

## 3.3 Episode / Termination

| Param | A (`parkour_env_cfg.py`, `parkour_env.py:1111-1134`) | B (`terminations.py:25-43`) | Status |
|---|---|---|---|
| `episode_length_s` | `20.0` | `20.0` | SAME |
| decimation / physics rate | `4 / 200 Hz` → 50 Hz policy | `4 / 200 Hz` → 50 Hz policy | SAME |
| termination_height | `< -0.20` m (`parkour_env.py:1123`) | `< -0.25` m (`terminations.py:37`) | A trips earlier on fall — auxiliary |
| tilt cutoff | `_term_tilt`: `max_tilt = 1.309` rad ≈ 75° (`parkour_env_cfg.py:602`) | roll `> 1.5` rad ≈ 86°, pitch `> 1.5` rad (`terminations.py:32-33`) | A is *tighter* by 11°. Aggressive jump-landings or stumbles trip A more often → discourages "land sideways" recoveries. Minor but contributes to fail-fast on jump landing. |
| base-contact termination | **disabled** (`parkour_env.py:1131` — comment shows `_term_base_contact` was excluded) | not used | SAME |
| goal-reached "success" termination | yes (`_term_goal_reached`, `terminated=True` so value bootstrap=0) | yes (`reach_goal_cutoff` → `time_out_buf`) | A treats success as terminated (no bootstrap); B as time_out (bootstrap). Cross-axis concern (value scaling) → worker-1/-2 may cite. |
| `termination_grace_steps` | `5` (skip terms for first 5 policy steps) (`parkour_env.py:1132-1133`) | none | A protects against spawn-jitter false-terms; B does not. Minor advantage A. |

---

## 3.4 Algorithm / Network

| Param | A (`agents/rsl_rl_ppo_cfg.py`) | B teacher (`agents/rsl_teacher_ppo_cfg.py`) | Status |
|---|---|---|---|
| `class_name` runner | `OnPolicyRunnerParkour` | `ParkourRslRlOnPolicyRunnerCfg` → PPOWithExtractor | SAME structurally |
| Algorithm class | `PPOParkour` | `PPOWithExtractor` (same DAGGER+priv_reg) | SAME paradigm |
| `num_steps_per_env` | `24` | `24` | SAME |
| `max_iterations` | `50000` | `50000` | SAME |
| `learning_rate` | `2.0e-4`, `schedule="adaptive"`, `desired_kl=0.01` | `2.0e-4`, `schedule="adaptive"`, `desired_kl=0.01` | SAME |
| `clip_param` | `0.2` | `0.2` | SAME |
| `entropy_coef` | `0.01` | `0.01` | SAME |
| `num_learning_epochs` | `5` | `5` | SAME |
| `num_mini_batches` | `4` | `4` | SAME |
| `gamma` | `0.99` | `0.99` | SAME |
| `lam` | `0.95` | `0.95` | SAME |
| `value_loss_coef` | `1.0` | `1.0` | SAME |
| `use_clipped_value_loss` | `True` | `True` | SAME |
| `max_grad_norm` | `1.0` | `1.0` | SAME |
| `init_noise_std` | `1.0` | `1.0` | SAME |
| Actor MLP | `[512, 256, 128]` ELU (`rsl_rl_ppo_cfg.py:62-64`) | `[512, 256, 128]` ELU | SAME |
| Critic MLP | `[512, 256, 128]` ELU | `[512, 256, 128]` ELU | SAME |
| `scan_encoder_dims` | implicit in `ActorCriticRMA`; matches `[128, 64, 32]` (env's RMA encoder family) | `[128, 64, 32]` | SAME by design (already-aligned RMA) |
| `priv_encoder_dims` | matches `[64, 20]` in RMA | `[64, 20]` | SAME |
| `estimator.hidden_dims` | `[128, 64]`, `lr=1e-3`, `train_with_estimated_states=True` (`rsl_rl_ppo_cfg.py:50-54`) | `[128, 64]` (`rsl_teacher_ppo_cfg.py:32-34`) | SAME (already-aligned) |
| `state_history_encoder` | `StateHistoryEncoder` (from rsl_rl modules) | `StateHistoryEncoder` (`parkour_rl_cfg.py:32`) | SAME |
| `dagger_update_freq` | configured via runner (default 20 in PPOParkour family) | `20` (`rsl_teacher_ppo_cfg.py:49`) | SAME |
| `priv_reg_coef_schedual` | not explicit in YAML (handled inside `PPOParkour`) | `[0.0, 0.1, 2000.0, 3000.0]` (`rsl_teacher_ppo_cfg.py:50`) | unknown for A → MAY differ. Memory says this is already aligned. |
| empirical_normalization | `False` | `False` | SAME |
| `clip_actions` | `10.0` (runner) but env-level `clip_actions = 4.8` (`parkour_env_cfg.py:388`) | `clip = (-4.8, 4.8)` in `ActionsCfg` (`parkour_mdp_cfg.py:349`) | SAME effective |
| Delayed action | env applies `action_scale=0.25`, no explicit delay buffer | `DelayedJointPositionActionCfg(action_delay_steps=[1,1], history_length=8, use_delay=True)` (`parkour_mdp_cfg.py:339-350`) | **DIFF — action delay** — see worker-2 (action capability) axis. Flagged here for completeness only. |

---

## Top 3 suspects (ranked across 3.1–3.4 — jump-bias drivers visible in this axis)

> Note: this is one of three parallel axes. Reward weighting (axis 1) and actuator/action (axis 2) may rank higher overall; this list is jump-causal within terrain/obs/algo.

### 1. Actor cannot see body angular velocity → no in-air attitude feedback
- **Where**: `parkour_env.py:852-863` proprio composition vs `parkour_env.py:898-904` `priv_explicit`.
- **A**: `root_ang_vel_b` lives in `priv_explicit` (critic-only RMA channel). The actor receives only `projected_gravity_b` (static tilt) — no rate information.
- **B**: `root_ang_vel_b * 0.25` is the FIRST 3 dims of proprio (`observations.py:69`) — actor sees it every step.
- **Hypothesis**: Without ang_vel in proprio, the policy learns an open-loop pre-shaped jump (`pitch_pos_clearance` + `feet_swing_height` configured by reward, but no feedback to attenuate when no obstacle is present). When the same policy lands on flat terrain it has no reactive damping signal → policy reproduces its jump pre-shape because that is the dominant gait it can articulate. The critic's privileged ang_vel guides the gradient but cannot create the actor's runtime reflex.
- **Confidence**: HIGH (well-known RMA design point; standard parkour proprio includes ang_vel — see https://arxiv.org/abs/2309.14341 reference cited by B `extreme_parkour_terrians.py:13`).
- **Verification**: move `root_lin_vel_b` to keep in `priv_explicit` but copy `root_ang_vel_b * 0.25` into `proprio` (re-set `num_proprio = 45`, update runner `obs_groups` and `StateHistoryEncoder` input dim). Re-train; expect flat-ground gait to lose jump-spike.

### 2. Terrain proportions starve walking gradient (flat 0.10 + gap 0.30)
- **Where**: `parkour_env_cfg.py:54-132` `sub_terrains` dict.
- **A**: flat 0.10, hurdle 0.20, step 0.20, **gap 0.30**, stair 0.20. Gap is the heaviest-weighted obstacle terrain; flat is the lightest walk terrain.
- **B**: flat **0.20**, hurdle 0.20, step 0.20, gap 0.20, parkour-stones 0.20. Flat is *twice* the weight; gap is one-third lower.
- **Hypothesis**: 90% of A's training tiles contain obstacles requiring partial jumping; only 10% provide a clean walk-only gradient. With reward terms (`tracking_goal_vel`, `feet_air_time` from `e81b71c80ef`) summing positive on both walk and jump strategies but jumping yielding larger forward progress per step on obstacle tiles, the policy converges to "always jump" as the locally-dominant gait, which then carries over to the 10% flat slice that is too small to pull it back.
- **Confidence**: HIGH — direct sub-parameter difference, no inference needed.
- **Verification**: change A's `parkour_flat.proportion = 0.20` and `parkour_gap.proportion = 0.20` (re-normalize: keep sum=1.0, e.g. flat 0.20 / hurdle 0.20 / step 0.20 / gap 0.20 / stair 0.20). Re-train from scratch. Expected: jump on flat reduces, walk gait emerges on flat tiles.

### 3. No `apply_roughness` on A terrains + no terrain-class flag in proprio
- **Where**:
   - A: no roughness field anywhere in `parkour_env_cfg.py` or `parkour_terrains.py`; smooth axis-aligned box meshes only.
   - B: `apply_roughness=True` on every sub-terrain (`extreme_parkour/config/parkour.py:18,24,32,40,47`) → `random_uniform_terrain(...)` overlays ±2–6 cm noise on every cell (`extreme_parkour_terrians.py:29-63`).
   - A proprio has no `is_flat / is_obstacle` indicator; B does (`observations.py:59-60`).
- **Hypothesis**: A's policy never sees ground texture variance. A "walk" gait optimized on perfectly smooth boxes is brittle — small perturbations in the deployed flat tile cause the policy to fall back to the more robust learned behaviour (the obstacle-clearing jump). B's policy walks on rough ground for 100% of training, so a "smooth flat" tile is just a special low-noise case of the rough-walk gait.
- Plus: A's policy has no observational signal "this episode is flat" — the policy must infer terrain class from height_scan (memory says it works) + history. Without the explicit flag, generalization can collapse to a single all-purpose-jump policy.
- **Confidence**: MED — multiple coupled effects.
- **Verification**:
  - V3a: add a `random_uniform_terrain`-style height-field roughness to A's box-mesh terrains (write a thin helper that adds a triangulated rough patch on top of the existing trimesh box terrain).
  - V3b: add a `terrain_class` integer (or one-hot) to A's proprio (`parkour_env.py:852` cat order). Re-train.

### Honourable mentions (not top-3 but visible in this axis)
- **Heading command + small-cmd-to-zero clamp absent in A**: B trains "stand still on small cmd" gradient; A always commands ≥ 0.3 m/s vx → no stand-still equilibrium → policy never settles on a non-jumping idle gait. Likely additive to #2.
- **`max_init_terrain_level = 3` in A vs 2 in B**: A starts agents on more difficult rows on iter 0; with already-narrow flat slice this strengthens the jump bias. Minor.
- **Action delay**: B uses `DelayedJointPositionActionCfg(action_delay_steps=[1,1], history_length=8)`; A is undelayed. Cross-axis (worker-2).
- **Tighter tilt cutoff in A (`max_tilt=1.309` vs B `1.5` rad)**: A penalises high-tilt landings/recoveries more aggressively → may suppress walking gaits that briefly tilt past 75°. Minor.

---

## Verification suggestions (concrete, single-knob)

| Test | Knob | Expected signal |
|---|---|---|
| T1 (own #1) | Add `root_ang_vel_b * 0.25` to A's proprio (proprio: 42 → 45 dim; update `num_proprio`, history buffer, encoder input dim) | Jump-on-flat amplitude reduces; walk gait stabilizes on flat tiles |
| T2 (own #2) | Re-balance A `sub_terrains` proportion to `(flat=0.20, hurdle=0.20, step=0.20, gap=0.20, stair=0.20)` | Walk gait emerges on flat; jump retained on gap/hurdle/step |
| T3 (own #3a) | Wrap A's trimesh box terrains with a height-field roughness overlay (±2–4 cm) | Walk gait gains robustness; flat tile gait stops mirroring jump |
| T4 (own #3b) | Add `is_flat` boolean (or `terrain_class` one-hot — 5 dims) to A proprio | Policy learns to conditionally select walk-vs-jump by terrain |
| T5 (HM-1) | Add `small_commands_to_zero` clamp at `lin_vel_clip=0.2` to A's `_resample_commands` and set `lin_vel_x_range=(0.0, 1.0)` | Stand-still equilibrium learned; flat tile may show pause-then-walk |
| T6 (HM-2) | Set A `max_init_terrain_level=2` (matches B teacher) | Earlier walk gradient; may reduce jump fixation early |

> Suggested order: T1, T2 in parallel (independent code paths). T3, T4, T5 only if T1+T2 don't fully resolve jump-on-flat.

---

## Already-aligned items (mark, don't rank)

- **RMA estimator + obs_groups encoder structure**: `Go2ParkourPPORunnerCfg.obs_groups` (`rsl_rl_ppo_cfg.py:40-47`) and `Estimator` (`rsl_rl_ppo_cfg.py:50-54`) mirror B's `parkour_rl_cfg.py` design. (Commit `c7261bd37f9`.)
- **Reward term set**: 14-term Genesis/extreme-parkour set; matches `TeacherRewardsCfg` (`parkour_mdp_cfg.py:117-244`). (Commit `25d2071196f`.)
- **`feet_air_time` reward**: added in commit `e81b71c80ef` to break flat-terrain drag.
- **Actor / critic MLP, encoder dims, PPO hyperparams**: identical to B teacher numerically.
- **History length (10), `num_proprio` placeholder shape, height_scan resolution as a sensor design choice** (do not rank `height_scan` per memory constraint).
- **Episode length (20 s), decimation (4), physics rate (200 Hz)**: identical to B.
- **Curriculum advance/regress rule (0.8 / 0.4 × expected/threshold distance)**: identical formula (`parkour_env.py:1270-1271` vs `parkour_event.py:143-144`).

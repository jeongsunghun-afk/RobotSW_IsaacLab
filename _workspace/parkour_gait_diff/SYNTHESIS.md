# Parkour A vs B Gait Diff — Synthesis (top-3 root cause)

> Worker-4 cross-axis integration of axis_1_reward.md, axis_2_action_physics.md, axis_3_terrain_obs_algo.md.
> Hard constraints obeyed: no `actuator_mode=2` ranking, no `height_scan` ranking, no contact-sensor obs
> proposal, no "regression" framing, no "direct vs manager" framing, `feet_air_time` not ranked
> (aligned post-25d2071196f — both lack it).

---

## TL;DR (3-sentence)

A's policy collapses to a jumping gait because **on obstacle terrain the vertical-velocity penalty
is 5× weaker than B's** (`lin_vel_z_l2 × 0.1` vs B's `× 0.5` non-flat multiplier, parkour_env.py:1001),
**and `torch.clip(total_reward, min=0.)` at L1108 zero-clips every step where penalties dominate**,
which collectively make jumping cheap and bad outcomes free. The reward asymmetry is amplified by a
**terrain mix biased against walking** (A: flat 0.10, gap 0.30 vs B: flat 0.20, gap 0.20) and by
**zero joint-pos reset noise** that pins all 4096 envs to an identical upright stance every episode,
foreclosing any exploration trajectory that would seed a walking branch. Fix in this order — H1, H2,
H3 — each is a single-config change with one-variable verification.

---

## Master diff (verified, cross-axis)

| Axis | Sub-parameter | A | B | Status | Suspected mechanism |
|---|---|---|---|---|---|
| 1 | `lin_vel_z_l2` non-flat multiplier (env.py:1001) | **× 0.1** | **× 0.5** | **DIFF — 5×** | Cheap vertical motion on hurdle/step/gap/stair → jumping pays ~0.008/step vs walking's 0.040/step (B) |
| 1 | total-reward clip (env.py:1108) | `torch.clip(total_reward, min=0.)` | none — raw sum | **DIFF — structural** | Floors all net-negative steps to 0 → eliminates negative gradient → "fall-and-jump" and "walk-cleanly" look equivalent when penalties dominate |
| 1 | `ang_vel_xy_l2` non-flat multiplier (env.py:1003) | **× 0.5** | **× 1.0** (no conditional) | DIFF — 2× | Pitch oscillation during take-off/landing softened in A only |
| 1 | `action_rate_l2` weight (cfg:569) | **−0.05** | **−0.10** | DIFF — 2× | Impulsive (slam-extend-snap) control half-priced |
| 2 | joint-pos reset noise (env.py:1182) | **none** — `default_joint_pos` verbatim | **±5 %** multiplicative (`position_range=(0.95,1.05)`) | DIFF | All 4096 envs identical stance every reset → exploration entropy at episode start ≈ 0 |
| 2 | hip-channel action scale (env.py:517) | **× 0.5** (hip joints only) | **× 1.0** (uniform across 12 joints) | DIFF | Half lateral-balance authority → walking abduction recovery weaker |
| 2 | `friction_combine_mode` (cfg:414, 439) | `multiply` | `average` | DIFF | A's slip-coef spread larger at low-end (0.4×1.0=0.4) → take-off slip variance higher |
| 3 | `root_ang_vel_b` in actor proprio (env.py:852-863) | **absent** (placed in priv_explicit, critic-only) | present `× 0.25` (observations.py:69) | DIFF | Actor has no closed-loop body-rate feedback → jump pre-shape open-loop |
| 3 | terrain proportions (env_cfg.py:54-132) | flat **0.10**, hurdle 0.20, step 0.20, gap **0.30**, stair 0.20 | flat **0.20**, hurdle 0.20, step 0.20, gap **0.20**, extreme_parkour 0.20 | DIFF | Only 10 % of A tiles offer pure walk gradient; gap (jump-mandatory) heaviest weighted |
| 3 | `apply_roughness` per sub-terrain | absent | `True` on every sub-terrain (±2–6 cm noise) | DIFF | A walks on perfectly smooth boxes only → walking gait brittle to any texture |
| 3 | `is_flat` / terrain-class flag in actor proprio | absent | 2 dims (`env_idx_tensor`, `invert_env_idx_tensor`) | DIFF | No explicit terrain-class signal → no conditional gait selection |
| 3 | `lin_vel_x` command upper bound (cfg:550) | `[0.3, 1.0]` | `(0.3, 0.8)` | DIFF — 25 % higher in A | Faster commanded gait pressure increases jump-pays-better margin |
| 3 | heading command + `small_commands_to_zero` | absent | heading ±1.6 rad, vel-clip 0.2 | DIFF | A has no stand-still equilibrium gradient |
| 3 | `max_init_terrain_level` (cfg:434) | **3** | **2** | DIFF (minor) | A's iter-0 difficulty pool slightly harder |
| 3 | tilt cutoff (cfg:602 / terminations.py:32) | 1.309 rad ≈ 75° | 1.5 rad ≈ 86° | DIFF (minor) | Sideways landings fail-fast in A; suppresses walking recoveries that briefly over-tilt |

---

## Top-3 root cause hypotheses (ranked)

### H1 — Reward asymmetry against vertical velocity (compound: `lin_vel_z_l2 × 0.1` + `clip(total_reward, min=0)`) — confidence **H**

- **Mechanism (how it produces jumping).**
  Two reward-side levers in A act on the same exploration loop:
  - (i) `lin_vel_z_l2 *= (is_flat + is_non_flat * 0.1)` at parkour_env.py:1001 — on every obstacle
    terrain class (hurdle/step/gap/stair, i.e. 90 % of training tiles), the vertical-velocity
    penalty is 10 % of nominal. B applies 50 %. Per-step cost at `|v_z|≈2 m/s`: A = `−0.008`,
    B = `−0.040` → over a 10–15-step jump cycle B pays ~0.5, A pays ~0.1, easily covered by the
    `tracking_goal_vel = +1.5` bonus.
  - (ii) `torch.clip(total_reward, min=0.)` at parkour_env.py:1108 — every step where penalties
    exceed bonuses is floored to 0. A net-negative jump-and-stumble step is reward-equivalent to
    a clean walk step → policy sees "jumping fails ≈ walking succeeds", so the higher-variance
    jumping policy is not down-weighted in the on-policy gradient.
  Together: jumping is *cheap* on obstacle tiles and *risk-free in the limit* on bad steps.
  Walking has no equivalent boost — it pays the same dof_acc / torque overhead and gains no
  multiplier relief.
- **Evidence.**
  - parkour_env.py:1001 — `lin_vel_z_l2 = lin_vel_z_l2 * (is_flat + is_non_flat * 0.1)`
  - parkour_env.py:1108 — `return torch.clip(total_reward, min=0.)`
  - B reference: rewards.py:143–145 (× 0.5), no clip on totals.
- **Compounds with.**
  H2 (terrain proportions: 90 % of A's tiles are obstacle terrain → the soft multiplier is the
  effective penalty 90 % of the time). H3 (zero reset noise: even when the policy *could* discover
  a walking branch, no initial-state diversity seeds it).
- **Single-variable verification test.**
  Two-line patch, single training run:
  ```python
  # parkour_env.py:1001
  lin_vel_z_l2 = lin_vel_z_l2 * (is_flat + is_non_flat * 0.5)   # 0.1 → 0.5
  # parkour_env.py:1108
  return total_reward                                            # remove the clip
  ```
  Expected within 500–1000 iters: mean peak `|v_z|` ↓ ~30–50 %, "flight phase" (all-feet-airborne)
  fraction ↓ on hurdle/step tiles, `tracking_goal_vel` curve still rises.
- **Cost.** Low — one training run, two-line patch.
- **Risk if wrong.** Trivial revert (git checkout of two lines). Worst-case symptom: policy now
  fails to clear the hardest gap difficulty because vertical motion is over-penalised — observable
  in 200 iters via `parkour_gap` success rate.

### H2 — Terrain proportions starve the walking gradient (flat 0.10 + gap 0.30 → 0.20 + 0.20) — confidence **H**

- **Mechanism.**
  Only 10 % of A's training tiles provide a clean walking gradient (`parkour_flat`); 90 % require
  some obstacle traversal. Gap — the single obstacle that *cannot* be solved by walking once
  width > ~30 cm — is allocated the heaviest weight (0.30) in A vs 0.20 in B, and A's hardest gap
  (50 cm) is still within Go2 standing reach while B's is 80 cm. So the marginal training
  experience in A is "gap that demands jumping" rather than "flat that rewards walking". Even if
  H1 were fixed, the policy gradient is averaged over a sample distribution that explicitly
  under-weights walking.
- **Evidence.**
  - parkour_env_cfg.py:54-132 — A `sub_terrains` dict: flat `proportion=0.10`,
    hurdle 0.20, step 0.20, **gap 0.30**, stair 0.20.
  - B reference: extreme_parkour/config/parkour.py:5-52 — flat `proportion=0.20`,
    each obstacle 0.20.
  - Gap math: parkour_env_cfg.py:110 — `gap_width=(0.05, 0.50)`; Go2 max stride reach ≈ 0.30 m.
- **Compounds with.**
  H1 (the soft `lin_vel_z` multiplier acts on these 90 % obstacle tiles — re-weighting more flat
  tiles directly reduces the surface area on which jumping is cheap). H3 (with reset-state
  variance, more flat tiles would expose more "walk-from-half-crouch" trajectories).
- **Single-variable verification test.**
  One-config edit in parkour_env_cfg.py: re-balance proportions to flat=0.20, hurdle=0.20,
  step=0.20, gap=0.20, stair=0.20 (sum = 1.0). No code change. One training run.
  Expected: walking gait emerges on flat tiles within 1k iters; obstacle clearance retained on
  the 80 %.
- **Cost.** Low — config-only, no code touched, full revert is a single-file diff.
- **Risk if wrong.** Negligible. If anything, this also tests whether H1 is necessary at all —
  with B-like terrain distribution, the soft `lin_vel_z` may matter less.

### H3 — Zero joint-pos reset noise forecloses walk-gait exploration — confidence **M-H**

- **Mechanism.**
  parkour_env.py:1182 sets `joint_pos = default_joint_pos[env_ids]` verbatim. All 4096 envs reset
  to the identical upright (.1/-.1 hip, .8/1.0 thigh, -1.5 calf) stance every episode. The PPO
  policy at episode step 0 sees an effectively noiseless initial proprio for every env — the only
  diversity comes from terrain row/col + incoming push (which arrives 8 s in). Jumping is the
  policy's first-discovered "go forward" mode from an upright stance because the legs are already
  loaded against the ground for an extension push. Walking requires the policy to ever try a
  half-crouched or off-default initial pose during exploration — without per-reset joint
  perturbation, the search distribution never samples that basin. B applies ±5 % multiplicative
  noise per joint (parkour_mdp_cfg.py:267-274), seeding diverse initial proprios across the 4096
  envs every reset.
- **Evidence.**
  - parkour_env.py:1182 — `joint_pos = self._robot.data.default_joint_pos[env_ids]` (no noise).
  - B reference: parkour_mdp_cfg.py:267-274 — `position_range=(0.95, 1.05)` via
    `reset_joints_by_scale`.
- **Compounds with.**
  H1 (with cheaper jumping, the policy locks onto jumping faster from any initial state — but
  noisy resets dilute that lock-in). H2 (more walking tiles + diverse starts is the precondition
  for walking to be discovered at all).
- **Single-variable verification test.**
  Two-line patch:
  ```python
  # parkour_env.py:1182
  joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
  joint_pos *= 1.0 + (torch.rand_like(joint_pos) * 0.1 - 0.05)
  ```
  Expected: episodic-reward variance ↑ in first 200 iters (good — wider exploration), then
  faster convergence on flat tiles.
- **Cost.** Low — two lines, one training run.
- **Risk if wrong.** Trivial. May increase early-iter termination rate by ~5–10 %; recoverable.

---

## Ruled out (with reason)

- **`actuator_mode=2` gain set (Kp=25 / Kd=0.5 / peak=23.5 N·m / vel_limit=30 / armature=0.01).**
  User has empirically verified this configuration trained successfully in the past
  (project_parkour_actuator_mode2_verified.md). The cfg comment at parkour_env_cfg.py:494-512 also
  declares this the deliberate "weakened" mode. Not a root cause candidate.
- **`height_scan` (parkour_env.py:585).** Verified normal in
  project_parkour_height_scan_verified.md. Not a root cause candidate.
- **Direct vs manager-style workflow.** Framework distractor; B's manager-style and A's direct
  hand-rolled reward pipeline both compute identical math (verified term-by-term in axis_1_reward.md
  §3). Not a root cause candidate.
- **`feet_air_time` absence.** Both A and B currently lack this term (A removed it in commit
  25d2071196f to align with B). Confirmed RULED OUT.
- **Contact-sensor observation.** Forbidden by project memory (sim-to-real). Not proposable.
- **Regression framing of A.** A has never trained successfully
  (project_parkour_A_never_trained.md); this is a structural diff vs B, not a regression to fix.

---

## Already aligned (do not rank)

- 14-term reward set (term names, function forms, 12 of 14 weights). Confirmed alignment except
  `action_rate_l2` weight and the two non-flat multipliers (covered in H1 / honourable mention).
- RMA estimator + obs_groups encoder structure (commit c7261bd37f9).
- B-style domain randomization scaffolding (commit c7261bd37f9): push DR (8 s, ±0.5 m/s),
  base mass DR (-1, +3 kg), foot-friction DR.
- Sim timing: `sim.dt = 1/200`, `decimation = 4`, policy_dt = 20 ms.
- Action transform: `action_scale = 0.25`, `clip_actions = 4.8`, `use_default_offset = True`.
- PPO hyperparams: `lr = 2e-4` adaptive (`desired_kl = 0.01`), `clip = 0.2`, `entropy = 0.01`,
  `5` epochs × `4` minibatches × `24` steps/env, `γ = 0.99`, `λ = 0.95`, `max_grad_norm = 1.0`.
- Actor/critic MLPs `[512, 256, 128]` ELU, scan encoder `[128, 64, 32]`, priv encoder `[64, 20]`.
- History length = 10, `StateHistoryEncoder` family.
- Episode length `20 s`, curriculum advance/regress `0.8 / 0.4 × expected_distance`.

---

## Open questions for user

1. **Is `torch.clip(total_reward, min=0.)` at parkour_env.py:1108 intentional?**
   It is not present in B (raw manager sum) and not documented in any commit message in the
   recent history. It silently invalidates every negative-gradient signal once a step's penalty
   sum exceeds its bonus sum — including the carefully-weighted negative penalties the team has
   been tuning. If it was added defensively (e.g. to suppress catastrophic-reward spikes during
   early training), can it be replaced with a clip on a per-term basis or a lower bound like
   `min=-10.` rather than `0.`?
2. **Are the Genesis-faithful non-flat multipliers in A (`0.1×` on `lin_vel_z`, `0.5×` on
   `ang_vel_xy`) a deliberate inheritance, or were they copied without intent to depart from B?**
   The env.py:1000 / 1002 comments cite "Genesis line 1437-1445" as justification. If the
   training target is B's gait (not Genesis'), H1 says: align with B.
3. **Is the hip-channel `0.5×` scaling (env.py:517) load-bearing for sim-to-real, or vestigial
   from the extreme-parkour reference code?** Worker-2 flagged this as M-confidence (V2 test).
   Knowing the design intent decides whether to bundle this into the sweep matrix or leave it.

---

## Verification matrix (suggested execution order)

| # | Patch | Files / lines | Cost | Expected signal |
|---|---|---|---|---|
| T1 | **H1**: change `0.1` → `0.5` AND remove `clip(min=0)` | parkour_env.py:1001, 1108 | 1 run | Mean peak `|v_z|` ↓, flight-phase fraction ↓, walking emerges on obstacle terrain |
| T2 | **H2**: rebalance terrain proportions | parkour_env_cfg.py:54-132 (5 `proportion=` fields) | 1 run | Walk gait emerges on flat tiles; obstacle clearance unchanged |
| T3 | **H3**: ±5 % joint-pos reset noise | parkour_env.py:1182 | 1 run | Higher early-iter reward variance, faster convergence after 200 iters |
| T4 | Combined H1 + H2 + H3 | as above | 1 run | If jumping persists after T4, root cause lies in axis untested here (likely H3 of axis-3: missing `root_ang_vel_b` in actor proprio) |
| T5 (fallback) | Move `root_ang_vel_b * 0.25` from `priv_explicit` to actor proprio | parkour_env.py:852-863 (cat), num_proprio 42→45, history buffer dim, encoder input dim | 1 run | Closed-loop attitude reflex during jump → smaller pitch oscillation |

Suggested order: **T1 → T2 → T3 → T4 → T5**. T1 has the highest leverage per line of code changed.

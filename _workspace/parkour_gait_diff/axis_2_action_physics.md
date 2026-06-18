# Axis 2: Action Capability / Actuator / Physics — Parkour A vs B

> **Scope.** Sub-parameter diff of everything that shapes the action→torque→impulse pipeline:
> simulation timing, action transform, PD gains, motor limits, contact physics, and the reset
> distribution that determines initial state for jump exploration.
>
> **Diff legend:**
> - **verified diff** = read directly from code (file:line cited)
> - **hypothesis** = behavioral consequence inferred; needs runtime check
> - **aligned** = numerically identical between A and B
>
> **Hard constraints obeyed** (project memory):
> - `actuator_mode=2` gains delta is documented in the table but routed to **RULED OUT**.
> - Torque-envelope math is not used to claim weakening (empirically rejected).

---

## Side-by-side table

### Simulation timing & action pipeline

| Sub-parameter | A value | B value | Status | Jump-vs-walk relevance |
|---|---|---|---|---|
| `sim.dt` | `1/200 = 0.005 s` (parkour_env_cfg.py:409, 411) | `0.005 s` (parkour_teacher_cfg.py:51) | **aligned** | Identical 200 Hz physics. |
| `decimation` | `4` (parkour_env_cfg.py:384) | `4` (parkour_teacher_cfg.py:48) | **aligned** | Identical 50 Hz policy step. |
| `render_interval` | `4` (parkour_env_cfg.py:412) | `=decimation` (parkour_teacher_cfg.py:52) | **aligned** | — |
| **policy_dt = sim.dt × decim** | **0.020 s** | **0.020 s** | **aligned** | Equal control bandwidth → torque application window identical. |
| `action_scale` | `0.25` (parkour_env_cfg.py:385) | `scale=0.25` on `DelayedJointPositionActionCfg` (parkour_mdp_cfg.py:343) | **aligned** | Same per-step joint pos delta authority. |
| `clip_actions` | `4.8` (parkour_env_cfg.py:388) | `clip = {'.*':(-4.8,4.8)}` (parkour_mdp_cfg.py:349) | **aligned** | Same ±4.8 rad-scaled clamp. |
| Hip joint scale | **A reduces hip channel ×0.5** before adding default (parkour_env.py:516-517) | **no per-joint scaling** — uniform 0.25 across all 12 joints (joint_actions.py:54) | **verified diff** | A halves hip-abduction amplitude in action→torque path. Asymmetric reduction in lateral-balance authority. |
| `use_default_offset` | `+ default_joint_pos` added after scaling (parkour_env.py:524) | `use_default_offset=True`, `_offset = default_joint_pos` (joint_actions.py:25-26, 54) | **aligned** | Identical offset convention. |
| Action delay | none (raw `actions` used same step) | `use_delay=True, action_delay_steps=[1,1]` → 1-step (20 ms) ring-buffer delay (joint_actions.py:42-46) | **verified diff** | B injects 20 ms sensorimotor delay (DR for real Go2 onboard latency). Too short to gate jump strategy; sim-to-real flavor, not capability. |
| Per-step action smoothness buffer | `_processed_actions / _last / _last_last` tracked (parkour_env.py:97-99, 519-524) | `_action_history_buf` length 8 (joint_actions.py:27) | **verified diff (bookkeeping)** | Used by reward — out of Axis 2 scope (see Axis 1). |

### Joint PD gains, motor limits, armature

| Sub-parameter | A value (mode=2 — active) | B value | Status | Jump-vs-walk relevance |
|---|---|---|---|---|
| Actuator class | `DCMotorCfg` (stock IsaacLab, parkour_env_cfg.py:518-529) | `ParkourDCMotorCfg` ← `IdealPDActuator` (parkour_actuator_pd.py:20; default_cfg.py:65-85) | **verified diff (impl)** | B inherits from `IdealPDActuator` and reimplements `_clip_effort` using `self._saturation_effort * (1 - vel/vel_limit)` — same formula as A's `DCMotor`. Functionally equivalent torque envelope. |
| `stiffness` (Kp) | `25.0` uniform (parkour_env_cfg.py:495) | `40.0` uniform (default_cfg.py:82) | **verified diff** | See RULED OUT. |
| `damping` (Kd) | `0.5` uniform (parkour_env_cfg.py:496) | `1.0` uniform (default_cfg.py:83) | **verified diff** | See RULED OUT. |
| `friction` | `0.0` (parkour_env_cfg.py:497) | `0.0` (default_cfg.py:84) | **aligned** | — |
| `armature` | `0.01` (parkour_env_cfg.py:528) | not set → defaults to `0.0` (default_cfg.py:65-85, no armature kwarg) | **verified diff (minor)** | A adds 0.01 reflected inertia. Small effect on torque ringing; negligible for jump peak impulse. |
| `effort_limit` (hip) | `23.5` (parkour_env_cfg.py:501) | `35.0` (default_cfg.py:68) | **verified diff** | See RULED OUT. |
| `effort_limit` (thigh) | `23.5` (parkour_env_cfg.py:502) | `40.0` (default_cfg.py:69) | **verified diff** | See RULED OUT. |
| `effort_limit` (calf) | `23.5` (parkour_env_cfg.py:503) | `40.0` (default_cfg.py:70) | **verified diff** | See RULED OUT. |
| `saturation_effort` (hip) | `23.5` scalar (parkour_env_cfg.py:498) | `35.0` per-joint dict (default_cfg.py:73) | **verified diff** | See RULED OUT. |
| `saturation_effort` (thigh/calf) | `23.5` scalar | `45.0` per-joint dict (default_cfg.py:74-75) | **verified diff** | See RULED OUT. |
| `velocity_limit` (hip) | `30.0` (parkour_env_cfg.py:508) | `52.4` (default_cfg.py:78) | **verified diff** | See RULED OUT. |
| `velocity_limit` (thigh/calf) | `30.0` (parkour_env_cfg.py:509-511) | `30.1` (default_cfg.py:79-80) | **aligned (≈)** | Identical. |
| `effort_limit_sim` | set equal to effort_limit (parkour_env_cfg.py:521) | not set | **verified diff (cosmetic)** | A also writes PhysX-side joint effort limit; B leaves PhysX default unbounded → torque clip purely software via `_clip_effort`. Practical envelope identical. |

### Spawn / reset distribution

| Sub-parameter | A value | B value | Status | Jump-vs-walk relevance |
|---|---|---|---|---|
| Robot init z (`init_state.pos`) | `(0,0,0.27)` from `UNITREE_GO2_CFG` (unitree.py:159) → then +0.05 in reset (parkour_env.py:1186) → effective spawn z ≈ 0.32 m | same `UNITREE_GO2_CFG` (default_cfg.py:33), reset moves to `origin - (terrain_size+3, 0, 0)` (events.py:47-61); no extra z buffer | **verified diff (semantic)** | A drops the robot 0.05 m onto terrain; B places it precisely at default. Neither blocks jump. |
| `joint_pos` reset | `joint_pos = default_joint_pos[env_ids]` — **no noise** (parkour_env.py:1182, 1189) | `reset_joints_by_scale` with `position_range=(0.95, 1.05)` — ±5 % multiplicative noise per joint (parkour_mdp_cfg.py:267-274) | **verified diff** | **A starts every episode in identical stance.** No init-state diversity → policy never sees a half-crouched start that would expose the jump branch. See Top 3 #1. |
| `joint_vel` reset | `default_joint_vel = 0` (parkour_env.py:1183) | `velocity_range=(0.0, 0.0)` — zero (parkour_mdp_cfg.py:271) | **aligned** | Both start at rest. |
| Root vel reset | `default_root_state[:, 7:]` = zero (parkour_env.py:1188) | `root_states[:, 7:13]` = zero (events.py:61) | **aligned** | — |
| `soft_joint_pos_limit_factor` | `0.9` (inherited UNITREE_GO2_CFG, unitree.py:169) | `0.9` (same) | **aligned** | — |
| `enabled_self_collisions` | `True` (unitree.py:155) | `True` (default_cfg.py:64, explicitly re-enabled) | **aligned** | — |
| Push DR | `push_robot` interval 8 s, vel ±0.5 m/s (parkour_env_cfg.py:344-352) | `push_by_setting_velocity` interval 8 s, vel ±0.5 m/s (parkour_mdp_cfg.py:321-327) | **aligned** | Identical episodic push. |
| Base mass DR | `(-1.0, +3.0)` add (parkour_env_cfg.py:317-325) | `(-1.0, +3.0)` add (parkour_mdp_cfg.py:296-304) | **aligned** | — |
| Base CoM DR | x ±0.08, y ±0.04, z ±0.02 (parkour_env_cfg.py:329-340) | x/y/z ±0.02 (parkour_mdp_cfg.py:305-312) | **verified diff (range)** | A has wider x-CoM range (8 cm vs 2 cm). Pushes the pitch axis harder. Not directly jump-vs-walk. |
| Actuator-gain DR | excluded by design (cfg comment, parkour_env_cfg.py:283-287) | excluded — commented out as "you will get a bad result" (parkour_mdp_cfg.py:286-295) | **aligned** | Both intentionally skip. |

### Contact physics / friction

| Sub-parameter | A value | B value | Status | Jump-vs-walk relevance |
|---|---|---|---|---|
| `sim.physics_material.friction_combine_mode` | `"multiply"` (parkour_env_cfg.py:414) | `"average"` (default_cfg.py:51) | **verified diff** | Multiply with foot DR (0.4×1.0=0.4 worst case) produces stronger slip extremes; average bounds the spread. Affects take-off traction. See Top 3 #3. |
| `sim.physics_material.restitution_combine_mode` | `"multiply"` (parkour_env_cfg.py:415) | `"average"` (default_cfg.py:52) | **verified diff** | Both robots use restitution=0, so combine mode is moot in practice. |
| `terrain.physics_material.friction_combine_mode` | `"multiply"` (parkour_env_cfg.py:439) | `=scene.terrain.physics_material` (parkour_teacher_cfg.py:53) → `"average"` | **verified diff** | Same direction as above. |
| Base static/dynamic friction | both = 1.0 (parkour_env_cfg.py:418-419) | both = 1.0 (default_cfg.py:53-54) | **aligned** | — |
| Foot friction DR | static (0.4,1.5), dynamic (0.3,1.2) (parkour_env_cfg.py:295-298) | range (0.6, 2.0) — applied to *all* bodies via `body_names=".*"` (parkour_mdp_cfg.py:275-283) | **verified diff** | B's DR is wider and applied to the whole robot, not just feet. Combined with `average` combine, effective contact friction range is similar magnitude but distribution different. |
| Body friction DR (non-foot) | base only, (0.6,1.2)/(0.5,1.0) (parkour_env_cfg.py:303-313) | all bodies (0.6, 2.0) | **verified diff** | — |
| `restitution_range` | (0.0, 0.0) (parkour_env_cfg.py:297) | not configured → default 0 | **aligned** | — |
| `solver_position_iteration_count` | `4` (unitree.py:155) | `4` (same, unitree.py:155) | **aligned** | — |
| `solver_velocity_iteration_count` | `0` | `0` | **aligned** | — |
| `gravity` | sim default (−9.81 m/s² z) | sim default | **aligned** | — |

---

## Top 3 suspects on this axis (ranked)

> Ordered by likelihood of being the per-axis lever that gates jump emergence. The
> **dominant** explanation almost certainly lies on Axis 1 (reward) or Axis 3 (terrain
> curriculum / obs), but among Axis-2-only deltas the following matter most.

### 1. **Reset stance has zero diversity in A → no exploration entropy at episode start** — confidence H

- **Mechanism.** B perturbs every joint by ±5 % multiplicative each reset (parkour_mdp_cfg.py:267-274). A loads `default_joint_pos` verbatim (parkour_env.py:1182). With 4096 envs starting in identical (.1/-.1 hip, .8/1.0 thigh, -1.5 calf) stance, the PPO policy receives an essentially noiseless initial proprio for every reset — the action-policy gradient at step 0 is averaged over near-identical states. The only diversity comes from terrain row/col and incoming velocity push, which arrive too late to seed a take-off pose.
- Jump is a discrete strategy: it requires the policy to discover "crouch → extend hard against ground" within a brief crouch window. Without per-reset joint noise, the policy never lands in the crouch basin during random exploration; it only ever sees the upright nominal stance.
- **Verification.** Single-variable test: add to A's `_reset_idx`
  ```python
  joint_pos = joint_pos + (torch.rand_like(joint_pos) * 0.1 - 0.05) * joint_pos.abs()
  ```
  (equivalent to B's ±5 % scale). Train 1 k iters; compare per-terrain success rate, esp. `parkour_hurdle` and `parkour_gap`.

### 2. **A's hip-channel 0.5× action scaling halves lateral-balance authority** — confidence M

- **Mechanism.** parkour_env.py:516-517 silently halves the action delta on the four hip joints before adding `default_joint_pos`. Effective hip action authority is `0.25 × 0.5 × 4.8 = 0.6 rad` peak vs `0.25 × 4.8 = 1.2 rad` peak for thigh/calf. B applies a uniform `0.25 × 4.8 = 1.2 rad` peak on all 12 joints (joint_actions.py:54). For Go2, jumps benefit from coordinated thigh/calf push, but **landing recovery** and **mid-air body orientation** rely heavily on hip abduction. If hip authority is half, the policy may avoid airborne phases that require strong recovery, biasing toward stay-on-ground (walk).
- This is widely used in extreme-parkour reference codebases (Cheng et al. ExtremeParkour), so the value isn't novel; what matters is that **the diff vs B exists and is on the action-capability path**. May be functionally important on `parkour_gap` / `parkour_hurdle`.
- **Verification.** Single-variable test: comment out line 517 in A. Train; compare jump frequency and `parkour_gap`/`parkour_hurdle` clear rate. Watch for orientation_l2 penalty increase (the reason hip reduction is typically used).

### 3. **`friction_combine_mode = "multiply"` (A) vs `"average"` (B) widens take-off slip distribution** — confidence M-L

- **Mechanism.** With foot static friction sampled in (0.4, 1.5) and terrain at 1.0, A's `multiply` gives effective coefficients (0.4, 1.5); B's `average` gives (0.7, 1.25). On the low end, A's foot can slip during the propulsive phase of a jump → push-off energy bleeds into translation, robot fails the obstacle → over time, policy learns "don't jump, slip risk too high → walk around / through". B's bounded range never punishes a committed jump that hard.
- This is a *secondary* gate: the policy must first try jumping before slip can punish it. If suspects #1/#2 already prevent jump discovery, fixing #3 alone won't help.
- **Verification.** Set A's `friction_combine_mode="average"` in both `sim.physics_material` (parkour_env_cfg.py:414) and `terrain.physics_material` (parkour_env_cfg.py:439). Train; track `tracking_goal_vel` curve on the gap / hurdle terrain classes.

---

## Verification suggestions (single-variable tests)

Run each in isolation (no other change), 500–1000 iter smoke train, eval on the 5-terrain curriculum:

- **Test V1 (highest priority)** — Inject ±5 % joint-pos reset noise:
  ```python
  # parkour_env.py, replace line 1182:
  joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
  joint_pos *= 1.0 + (torch.rand_like(joint_pos) * 0.1 - 0.05)
  ```
  Expected if hypothesis correct: faster ramp-up of episodic reward, jump frequency > 0 on hurdle/gap, terrain curriculum advances past level 3.

- **Test V2** — Remove hip-scale reduction:
  ```python
  # parkour_env.py:517 — comment out
  # scaled[:, self._hip_joint_ids] *= 0.5
  ```
  Expected: more aggressive hip motion in logs (mean |hip_action|↑), possibly more orientation penalty initially. If gap/hurdle success rises, hip reduction was the bottleneck.

- **Test V3** — Switch to `average` friction combine + restrict foot DR to B's symmetric range `(0.6, 2.0)`:
  ```python
  # parkour_env_cfg.py:414, 439
  friction_combine_mode="average",
  # parkour_env_cfg.py:295-298
  "static_friction_range": (0.6, 2.0),
  "dynamic_friction_range": (0.6, 2.0),
  ```
  Expected: less variance in early-training success rate on gap.

- **Test V4 (combination, only after V1 individually shows no signal)** — V1 + V2 together. If still no jump emerges, root cause is **outside Axis 2** (look to reward set / goal yaw / terrain class proportion — see Axis 1, Axis 3).

---

## Already-aligned items (mark, don't rank)

- **Sim timing**: dt=1/200, decimation=4, policy_dt=20 ms — identical.
- **action_scale=0.25, clip_actions=4.8** — identical.
- **`use_default_offset`** — both add `default_joint_pos` after scaling.
- **Solver iters** (4/0), **gravity**, **soft_joint_pos_limit_factor=0.9**, **self_collisions=True** — identical.
- **Push DR** (interval 8 s, vel ±0.5 m/s) — identical.
- **Base mass DR** (-1, +3 kg) — identical.
- **Restitution=0** everywhere — combine_mode diff is moot for restitution.
- **Actuator-gain DR** — both intentionally excluded.
- **Velocity limit on thigh/calf** (30.0 vs 30.1) — effectively identical.
- **RMA estimator, B-style domain randomization** — aligned in commit c7261bd37f9 (per project memory).

---

## RULED OUT (with reason)

- **`actuator_mode=2` gain delta (Kp 25 vs 40, Kd 0.5 vs 1.0, peak 23.5 N·m vs 35–45 N·m per joint, vel_limit 30 vs 52.4 on hip).** User has empirically verified these values trained successfully in the past. The cfg comment block at parkour_env_cfg.py:494-512 also documents this as the deliberate "weakened" mode that should not be ranked as root cause. Diff is logged in the table for worker-4's master diff but **not** in Top 3.
- **Per-joint differentiated effort/saturation in B (hip 35/calf 45) vs A's uniform 23.5.** Same gate as above — uniform 23.5 was previously sufficient. Worth noting in master diff but not a hypothesis here.
- **`armature 0.01 vs 0.0`** — third-order effect on torque transients; nowhere near the jump-vs-walk decision boundary.
- **`effort_limit_sim` set in A, not in B** — both end up clipping torque to identical envelope through different code paths (A via PhysX joint limit + DCMotor `_clip_effort`; B via `ParkourDCMotor._clip_effort` only). User-empirical verification on the gain set covers this.
- **Action delay `[1,1]` in B (20 ms).** Too small to switch jump→walk strategy. It's sim-to-real onboard-latency DR, not capability.
- **Robot init z (+0.05 m drop in A).** 0.05 m onto rigid terrain is below the contact-restitution scale; physics settling happens during the `termination_grace_steps=5` window. Does not gate jump.

---

## Notes for synthesis (worker-4)

- All three Top-3 candidates here are **exploration / authority** issues, not raw actuator strength. They are consistent with the hard-constraint framing that A's actuators are not too weak in the steady-state torque sense.
- If Axis 1 (reward) and Axis 3 (terrain/curriculum) show stronger candidates, they likely dominate; this axis contributes **modifiers** (V1 especially) that may unblock learning once the upstream pressure is right.
- The hip-scale reduction (V2) is a single line of code and should be in the first sweep matrix alongside whatever Axis 1 / 3 candidates are picked.

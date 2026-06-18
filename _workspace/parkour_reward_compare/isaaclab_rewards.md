# IsaacLab Parkour Reward Analysis

**Source:** `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`
**Cfg:**  `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py`
**Env class:** `Go2ParkourEnv(DirectRLEnv)` — direct-style task, no manager-based reward terms.
**Frequencies:** physics `dt = 1/200 s`, `decimation = 4`, so **`step_dt = 4/200 = 0.02 s`** (50 Hz policy). `episode_length_s = 20.0`.

---

## 1. Summary of Reward Terms (cfg.reward_scales)

Table reads scale values **verbatim** from `ParkourEnvCfg.reward_scales` (no value judgments).

| # | Term name (cfg key) | Sign / role                | Scale (`reward_scales[...]`) | Active? |
|---|---------------------|-----------------------------|------------------------------|---------|
| 1 | `tracking_goal_vel`     | bonus — toward-goal speed     | `+1.0`         | yes |
| 2 | `tracking_yaw`          | bonus — heading match         | `+0.5`         | yes |
| 3 | `tracking_lin_vel_xy_exp` | bonus — body-frame lin vel  | `0.0`          | off (parkour weakening) |
| 4 | `tracking_ang_vel_z_exp`  | bonus — body-frame yaw rate | `0.0`          | off (parkour weakening) |
| 5 | `lin_vel_z_l2`          | penalty                       | `-1.0`         | yes (×0.1 on non-flat) |
| 6 | `ang_vel_xy_l2`         | penalty                       | `-0.05`        | yes (×0.5 on non-flat) |
| 7 | `orientation_l2`        | penalty                       | `-1.0`         | yes (flat only) |
| 8 | `dof_acc_l2`            | penalty                       | `-2.5e-7`      | yes |
| 9 | `collision`             | penalty                       | `-10.0`        | yes |
| 10 | `action_rate_l2`       | penalty                       | `-0.0` (literal `-0.`) | off |
| 11 | `delta_torques`        | penalty                       | `-1.0e-7`      | yes |
| 12 | `torques_l2`           | penalty                       | `-1e-5`        | yes |
| 13 | `hip_pos`              | penalty                       | `-0.0`         | off |
| 14 | `dof_error_l2`         | penalty                       | `-0.0`         | off |
| 15 | `feet_stumble`         | penalty                       | `-0.0`         | off |
| 16 | `feet_edge`            | penalty                       | `-0.0`         | off |
| 17 | `termination`          | penalty (early-term)          | `-100.0`       | yes |
| 18 | `feet_dragging`        | penalty                       | `-0.0`         | off |
| 19 | `action_smoothness_1`  | penalty                       | `-0.0`         | off |
| 20 | `action_smoothness_2`  | penalty                       | `-0.0`         | off |
| 21 | `base_height`          | penalty (flat only)           | `0.0`          | off |
| 22 | `feet_air_time`        | bonus — trot inducer (flat only) | `0.0`       | off |

**Count:** 22 reward terms registered in `cfg.reward_scales`. Every key in the dict is iterated by `_get_rewards()` and multiplied by `step_dt`.

`cfg.tracking_sigma = 0.2`, `cfg.dragging_velocity_threshold = 0.05 m/s`, `cfg.base_height_target = 0.34 m`.

---

## 2. Conditional Terrain-Class Masks (applied **inside** `_get_rewards`)

Before any reward is assembled, env-class is split into:

```python
is_flat     = (self._env_class == TERRAIN_CLASS_FLAT).float()   # 0 = flat
is_non_flat = 1.0 - is_flat
```

`self._env_class` (shape `[num_envs]`, int64) is set in `__init__` via the
`(col_idx → class_id)` LUT computed from `cfg.terrain.terrain_generator.sub_terrains` proportions, and refreshed in `_reset_idx` after curriculum updates.

Multiplicative masks applied to specific terms (Genesis-faithful):
- `lin_vel_z_l2     *= is_flat + is_non_flat * 0.1`     (10× weaker on obstacles)
- `ang_vel_xy_l2    *= is_flat + is_non_flat * 0.5`     (2× weaker on obstacles)
- `orientation_l2   *= is_flat`                          (zero on non-flat)
- `dof_error_l2     *= is_flat + is_non_flat * 0`        (flat-only)
- `base_height      *= is_flat`                          (flat-only)
- `feet_air_time    *= is_flat + is_non_flat * 0`        (flat-only)

`feet_edge` is gated by `(self._terrain_levels > 3).float()`.

---

## 3. Per-term Detail

### 3.1 `tracking_goal_vel`  (parkour-specific, Task #4)
- **Inputs:**
  - `self._robot.data.root_lin_vel_w[:, :2]` — world-frame linear velocity, XY only `[N, 2]`.
  - `self._target_pos_rel` — relative goal vector in world-XY (set in `_update_goals`) `[N, 2]`.
  - `self._commands[:, 0]` — forward velocity command `[N]`.
- **Formula:**
  ```
  goal_dir       = target_pos_rel / (‖target_pos_rel‖ + 1e-5)              # [N, 2]
  proj_forward   = Σ_xy ( vel_w_xy * goal_dir )                            # [N]
  commanded_spd  = |commands[:,0]|
  r              = min(proj_forward, commanded_spd) / (commanded_spd+1e-5)
  r              = where(commanded_spd > 1e-3, r, 0)
  ```
  (No clamp(min=0); negative when robot moves opposite to goal.)
- **Shape:** `[N]`
- **Scale:** `+1.0`

### 3.2 `tracking_yaw`  (parkour-specific)
- **Inputs:** `self._robot.data.heading_w` `[N]`; `self._target_yaw` `[N]` (from `_update_goals`).
- **Formula:** `r = exp( -|target_yaw - heading_w| )`  — no wrap (`atan2(sin,cos)` line is commented out), no speed gate (comment: "stand-still local optimum prevented by tracking_goal_vel").
- **Shape:** `[N]`
- **Scale:** `+0.5`

### 3.3 `tracking_lin_vel_xy_exp`
- **Inputs:** `self._commands[:, :2]` `[N,2]`; `self._robot.data.root_lin_vel_b[:, :2]` `[N,2]` (body frame).
- **Formula:**
  ```
  err = Σ ( (commands[:, :2] - root_lin_vel_b[:, :2])² )      # [N]
  r   = exp( -err / tracking_sigma )                          # tracking_sigma = 0.2
  ```
- **Shape:** `[N]`
- **Scale:** `0.0`

### 3.4 `tracking_ang_vel_z_exp`
- **Inputs:** `self._commands[:, 2]` `[N]`; `self._robot.data.root_ang_vel_b[:, 2]` `[N]`.
- **Formula:** `r = exp( -(commands[:,2] - ang_vel_b[:,2])² / tracking_sigma )`.
- **Shape:** `[N]`
- **Scale:** `0.0`

### 3.5 `lin_vel_z_l2`
- **Inputs:** `self._robot.data.root_lin_vel_b[:, 2]` `[N]` (body-frame z).
- **Formula:** `p = (vz_b)²`, then `p *= is_flat + 0.1*is_non_flat`.
- **Shape:** `[N]`
- **Scale:** `-1.0`

### 3.6 `ang_vel_xy_l2`
- **Inputs:** `self._robot.data.root_ang_vel_b[:, :2]` `[N,2]`.
- **Formula:** `p = Σ (ω_b_xy²)`, then `p *= is_flat + 0.5*is_non_flat`.
- **Shape:** `[N]`
- **Scale:** `-0.05`

### 3.7 `orientation_l2`
- **Inputs:** `self._robot.data.projected_gravity_b[:, :2]` `[N,2]`.
- **Formula:** `p = Σ (g_proj_xy²) * is_flat`. (Zero on non-flat.)
- **Shape:** `[N]`
- **Scale:** `-1.0`

### 3.8 `dof_acc_l2`
- **Inputs:** `self._robot.data.joint_vel` `[N,12]`; `self._prev_joint_vel` `[N,12]` (snapshot from previous step, stored in `__init__` and updated each `_get_rewards`).
- **Formula:**
  ```
  joint_acc = (joint_vel - prev_joint_vel) / step_dt          # [N, 12]
  p         = Σ_j ( joint_acc² )                              # [N]
  ```
  Update line: `self._prev_joint_vel = self._robot.data.joint_vel.clone()` at top of `_get_rewards`.
- **Shape:** `[N]`
- **Scale:** `-2.5e-7`

### 3.9 `collision`
- **Inputs:** `self._contact_sensor.data.net_forces_w_history[:, :, self._undesired_contact_body_ids]` `[N, hist=3, n_bodies, 3]`.
  - `self._undesired_contact_body_ids` = bodies matching `base, .*thigh, .*calf, .*hip, Head_upper, Head_lower`.
- **Formula:**
  ```
  is_contact = max_{hist}( ‖F‖ ) > 0.1                        # [N, n_bodies] bool
  collision  = Σ_b ( is_contact )                             # [N] float
  ```
- **Shape:** `[N]` float
- **Scale:** `-10.0`

### 3.10 `action_rate_l2`
- **Inputs:** `self._actions`, `self._previous_actions` `[N, 12]`.
- **Formula:** `p = ‖actions − prev_actions‖₂`  (L2 **norm**, not sum-of-squares — comment in code emphasizes this differs from Genesis sum-square).
- **Shape:** `[N]`
- **Scale:** `-0.0`

### 3.11 `delta_torques`
- **Inputs:** `self._robot.data.applied_torque` `[N, 12]`; `self._last_applied_torque` `[N, 12]` (buffer in `__init__`, updated in `_get_rewards` after use).
- **Formula:** `p = Σ_j ( (τ_t − τ_{t−1})² )`.
- **Shape:** `[N]`
- **Scale:** `-1.0e-7`

### 3.12 `torques_l2`
- **Inputs:** `self._robot.data.applied_torque` `[N, 12]`.
- **Formula:** `p = Σ_j (τ²)`.
- **Shape:** `[N]`
- **Scale:** `-1e-5`

### 3.13 `hip_pos`
- **Inputs:** `self._robot.data.joint_pos[:, self._hip_joint_ids]`, `self._robot.data.default_joint_pos[:, self._hip_joint_ids]`. `self._hip_joint_ids` (long tensor) computed in `__init__` by name match `"hip"`.
- **Formula:** `p = Σ_hip ( (q_hip − q_hip_default)² )`.
- **Shape:** `[N]`
- **Scale:** `-0.0`

### 3.14 `dof_error_l2`
- **Inputs:** `self._robot.data.joint_pos`, `self._robot.data.default_joint_pos` `[N, 12]`.
- **Formula:** `p = Σ_j ( (q − q_default)² )`, then `p *= is_flat + 0*is_non_flat`. (Flat-only.)
- **Shape:** `[N]`
- **Scale:** `-0.0`

### 3.15 `feet_stumble`
- **Inputs:** `self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids]` `[N, 4, 3]` (history index `0` = latest substep).
  - `self._feet_ids` matches `.*foot` (4 bodies).
- **Formula:**
  ```
  feet_F   = net_forces_w_history[:, 0, feet_ids]              # [N, 4, 3]
  stumble  = any_feet( ‖F_xy‖ > 4 * |F_z| )                    # [N] float
  ```
- **Shape:** `[N]` float (0/1)
- **Scale:** `-0.0`

### 3.16 `feet_edge`  (Genesis x_edge_mask port — Option B)
- **Inputs:**
  - `self._contact_sensor.data.net_forces_w_history[:, 0, self._feet_ids]` `[N, 4, 3]` (latest substep).
  - `self._last_contacts` `[N, 4]` bool (OR with current contact, then `self._last_contacts = contact`).
  - `self._robot.data.body_pos_w[:, self._feet_ids, :]` `[N, 4, 3]` (world position of feet).
  - `self.x_edge_mask` `[n_x, n_y]` bool (built once in `_build_edge_mask`).
  - `self._edge_mask_origin` `[2]`, `self._edge_mask_inv_scale` scalar.
  - `self._terrain_levels` `[N]`.
- **Formula:**
  ```
  contact      = ‖F‖ > 2.0                                     # [N, 4] bool
  contact_filt = contact | last_contacts                        # [N, 4]  (Genesis OR)
  last_contacts ← contact                                       # update buffer
  feet_xy      = body_pos_w[:, feet_ids, :2]                   # [N, 4, 2]
  grid_xy      = round((feet_xy - origin) * inv_scale - 0.5).long()
  grid_xy      = clamp(grid_xy, 0, mask_dim-1)
  feet_at_edge = x_edge_mask[grid_xy[..., 0], grid_xy[..., 1]] # [N, 4] bool
  feet_at_edge = contact_filt & feet_at_edge
  r            = (terrain_levels > 3).float() * Σ_feet ( feet_at_edge.float() )
  ```
- **Shape:** `[N]` (zero for envs at curriculum level ≤ 3)
- **Scale:** `-0.0`

### 3.17 `termination`  (early-term penalty)
- **Inputs:** `self._term_base_contact`, `self._term_tilt`, `self._term_low_height` (bool `[N]`), all set by `_get_dones` each step (see §5).
- **Formula:** `t = (term_base_contact | term_tilt | term_low_height).float()`. Note: applied **before** the `~grace` mask of `_get_dones`, so penalty can fire during the grace period too.
- **Shape:** `[N]`
- **Scale:** `-100.0`

### 3.18 `feet_dragging`
- **Inputs:**
  - `self._robot.data.body_link_lin_vel_w[:, self._feet_ids, :2]` `[N, 4, 2]` (world XY link velocity at foot frame).
  - `contact_filt` (`[N,4]`) reused from feet_edge block.
  - `cfg.dragging_velocity_threshold = 0.05 m/s`.
- **Formula:**
  ```
  feet_speed  = ‖feet_lin_vel_xy‖                                          # [N, 4]
  p           = Σ_feet ( feet_speed * contact_filt * (feet_speed > thr) )  # [N]
  ```
- **Shape:** `[N]`
- **Scale:** `-0.0`

### 3.19 `feet_air_time`  (anymal_c / R_Skeleton style, NEW 2026-05-13)
- **Inputs:**
  - `self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]` `[N, 4]` bool — rising edge.
  - `self._contact_sensor.data.last_air_time[:, self._feet_ids]` `[N, 4]` — seconds airborne before this contact event.
  - `self._commands[:, :2]` `[N, 2]` (gate by speed cmd magnitude > 0.1).
- **Formula:**
  ```
  r = Σ_feet ( (last_air_time - 0.5) * first_contact )                    # [N]
  r = r * ( ‖cmd_xy‖ > 0.1 ).float()
  r = r * ( is_flat + 0 * is_non_flat )                                   # flat-only
  ```
- **Shape:** `[N]`
- **Scale:** `0.0`

### 3.20 `action_smoothness_1`
- **Inputs:** `self._processed_actions`, `self._last_processed_actions` `[N, 12]` (snapshots taken in `_pre_physics_step`).
- **Formula:**
  ```
  mask  = (last_processed_actions != 0).float()
  p     = Σ_j ( (a_t - a_{t-1})² * mask )                                  # [N]
  ```
- **Shape:** `[N]`
- **Scale:** `-0.0`

### 3.21 `action_smoothness_2`
- **Inputs:** `self._processed_actions`, `self._last_processed_actions`, `self._last_last_processed_actions` `[N, 12]`.
- **Formula:**
  ```
  mask1 = (last_processed != 0).float()
  mask2 = (last_last_processed != 0).float()
  diff2 = (a_t - 2*a_{t-1} + a_{t-2})²
  p     = Σ_j ( diff2 * mask1 * mask2 )                                    # [N]
  ```
- **Shape:** `[N]`
- **Scale:** `-0.0`

### 3.22 `base_height`  (flat-only)
- **Inputs:** `self._robot.data.root_link_pos_w[:, 2]`, `self._terrain.env_origins[:, 2]`, `cfg.base_height_target = 0.34`.
- **Formula:** `p = (z_base − z_origin − 0.34)² * is_flat`. (Squared deviation around target stance height; zero on non-flat.)
- **Shape:** `[N]`
- **Scale:** `0.0`

### 3.23 `feet_stumble` is term 3.15 (kept for completeness — see above).

---

## 4. Reward Accumulation

```python
total_reward = torch.zeros(self.num_envs, device=self.device)
for key, value in reward_values.items():
    scaled = self.cfg.reward_scales[key] * self.step_dt * value
    self._episode_sums[key] += scaled
    total_reward += scaled
return total_reward
```

- **Scaling pattern:** `per_term_scale × step_dt × value` for **every** term, including bonuses, penalties, and the discrete `termination` event.
- **`step_dt`** = `decimation × sim_dt` = `4 × (1/200)` = **0.02 s**.
- **No global clip** before/after summation. No per-term clip beyond what the term-internal formula does (`feet_edge` gating, `tracking_goal_vel` `where(...)`, etc.).
- `self._episode_sums` (`__init__`) is a `dict[str → Tensor[N]]` zero-initialised with one key per reward; summed at every step and flushed in `_reset_idx` (logged as `Episode_Reward/<key> / max_episode_length_s`).

---

## 5. Termination Penalty Handling

`_get_dones()` computes 4 boolean masks (each `[N]`):

| Flag | Source | Threshold |
|------|--------|-----------|
| `self._term_base_contact` | `max_hist(‖F‖) > 5.0` on `self._base_id` (body name `base`) | 5.0 N |
| `self._term_tilt`         | `Σ (projected_gravity_b[:, :2]²) > 0.99` | `≈ sin²(1.5 rad)` |
| `self._term_low_height`   | `root_link_pos_w[:, 2] < cfg.termination_height` (-0.2) | -0.2 m |
| `self._term_goal_reached` | set in `_update_goals` when robot holds at last waypoint long enough | success flag |

```python
terminated = self._term_base_contact | self._term_tilt | self._term_low_height
grace      = episode_length_buf < cfg.termination_grace_steps     # 5 steps
terminated = (terminated & ~grace) | self._term_goal_reached
```

**Inside the reward,** the `termination` term uses the **un-grace-gated** OR:
```python
termination = (self._term_base_contact | self._term_tilt | self._term_low_height).float()
```
So an env that triggers a failure during the first 5 steps **still receives the −100 × step_dt = −2.0 step penalty per step it remains failed**, even though it is not actually reset (grace blocks the `terminated` flag, not the reward term). `_term_goal_reached` is **not** included in the reward (no positive bonus for goal completion via this term).

The `termination` reward is applied via the standard scaled accumulation: `−100.0 × 0.02 × {0,1}` = `−2.0` per step in which the failure conditions are met.

---

## 6. `__init__` Buffer Table (reward-relevant)

| Buffer | dtype | shape | init value | Reset in `_reset_idx` | Consumers (rewards) |
|--------|-------|-------|------------|------------------------|----------------------|
| `_actions` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `action_rate_l2` |
| `_previous_actions` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `action_rate_l2` (also set in `_pre_physics_step`) |
| `_processed_actions` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `action_smoothness_1/2` |
| `_last_processed_actions` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `action_smoothness_1/2` |
| `_last_last_processed_actions` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `action_smoothness_2` |
| `_last_applied_torque` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `delta_torques` |
| `_prev_joint_vel` | float32 | `[N, 12]` | 0 | yes (`= 0.0`) | `dof_acc_l2` (also updated each `_get_rewards`) |
| `_commands` | float32 | `[N, 3]` | 0 | resampled via `_resample_commands` | `tracking_goal_vel` (cmd[0]), `tracking_lin_vel_xy_exp`, `tracking_ang_vel_z_exp`, `feet_air_time` gate |
| `_last_contacts` | bool | `[N, 4]` | `False` | yes (`= False`) | `feet_edge`, `feet_dragging` |
| `_episode_sums` | dict[str→float32 [N]] | per key | 0 | flushed to extras and zeroed for `env_ids` | logging only |
| `_term_base_contact` | bool | `[N]` | `False` | set each step in `_get_dones` | `termination` reward, logging |
| `_term_tilt`         | bool | `[N]` | `False` | set each step in `_get_dones` | `termination` reward, logging |
| `_term_low_height`   | bool | `[N]` | `False` | set each step in `_get_dones` | `termination` reward, logging |
| `_term_goal_reached` | bool | `[N]` | `False` | yes (`= False`); set in `_update_goals` | logging only (NOT part of `termination` reward) |
| `_target_pos_rel` | float32 | `[N, 2]` | 0 | yes (`= 0.0`); refreshed in `_update_goals` | `tracking_goal_vel` |
| `_target_yaw` | float32 | `[N]` | 0 | yes (`= 0.0`); refreshed in `_update_goals` | `tracking_yaw` |
| `_env_class` | int64 | `[N]` | from `_col_to_class[_terrain_types]` | refreshed after curriculum in `_reset_idx` | mask for `lin_vel_z_l2`, `ang_vel_xy_l2`, `orientation_l2`, `dof_error_l2`, `base_height`, `feet_air_time` |
| `_terrain_levels` | int64 | `[N]` | from `self._terrain.terrain_levels` | updated by `_update_terrain_curriculum` | `feet_edge` gate |
| `_hip_joint_ids` | long | `[4]` | indices of `"hip"` joints | static | `hip_pos` |
| `_feet_ids` (find_bodies `.*foot`) | list[int] | 4 | static | static | `feet_stumble`, `feet_edge`, `feet_dragging`, `feet_air_time` |
| `_base_id` (find_bodies `"base"`) | list[int] | 1 | static | static | `_term_base_contact` |
| `_undesired_contact_body_ids` | list[int] | n | static (`base, .*thigh, .*calf, .*hip, Head_upper, Head_lower`) | static | `collision` |
| `x_edge_mask` | bool | `[n_x, n_y]` | built in `_build_edge_mask` | static | `feet_edge` |
| `_edge_mask_origin`, `_edge_mask_inv_scale` | float / scalar | `[2]` / scalar | from terrain extents | static | `feet_edge` index math |

Termination masks (`_term_base_contact`, `_term_tilt`, `_term_low_height`) are NOT explicitly cleared in `_reset_idx`; they are unconditionally overwritten on every call to `_get_dones()` at the beginning of the next step (env-wide assignment, not in-place index update), so stale carry-over is safe.

---

## 7. Quick Glossary of IsaacLab Data Accessors Used

| Accessor | Frame | Shape | Used by |
|----------|-------|-------|---------|
| `robot.data.root_lin_vel_w[:, :2]` | world | `[N, 2]` | `tracking_goal_vel` |
| `robot.data.root_lin_vel_b[:, :2]` | body | `[N, 2]` | `tracking_lin_vel_xy_exp` |
| `robot.data.root_lin_vel_b[:, 2]`  | body | `[N]`    | `lin_vel_z_l2` |
| `robot.data.root_ang_vel_b[:, 2]`  | body | `[N]`    | `tracking_ang_vel_z_exp` |
| `robot.data.root_ang_vel_b[:, :2]` | body | `[N, 2]` | `ang_vel_xy_l2` |
| `robot.data.heading_w` | world | `[N]` | `tracking_yaw` |
| `robot.data.projected_gravity_b[:, :2]` | body | `[N, 2]` | `orientation_l2`, `_term_tilt` |
| `robot.data.joint_pos`, `joint_vel`, `default_joint_pos`, `applied_torque` | joint | `[N, 12]` | `dof_acc_l2`, `dof_error_l2`, `hip_pos`, `torques_l2`, `delta_torques` |
| `robot.data.body_pos_w[:, feet_ids, :]` | world | `[N, 4, 3]` | `feet_edge` foot XY |
| `robot.data.body_link_lin_vel_w[:, feet_ids, :2]` | world | `[N, 4, 2]` | `feet_dragging` |
| `robot.data.root_link_pos_w[:, 2]` | world | `[N]` | `base_height`, `_term_low_height` |
| `terrain.env_origins[:, 2]` | world | `[N]` | `base_height` |
| `contact_sensor.data.net_forces_w_history` | world | `[N, hist=3, n_bodies, 3]` | `collision` (max over hist), `feet_stumble` (idx 0), `feet_edge` (idx 0), `_term_base_contact` (max over hist) |
| `contact_sensor.compute_first_contact(step_dt)` | — | `[N, n_bodies]` bool | `feet_air_time` |
| `contact_sensor.data.last_air_time` | — | `[N, n_bodies]` | `feet_air_time` |

---

## 8. Notes / Quirks worth flagging

1. **`action_rate_l2` is L2 *norm* (not sum-square)**, comment explicitly flags this as the parkour deviation from Genesis sum-of-squares.
2. **`tracking_goal_vel` uses world-frame velocity** projected onto world-frame goal direction (not body frame).
3. **`feet_edge` reuses `_last_contacts`** which is also updated inside the feet_edge block, so any *subsequent* reward in the same step sees the updated buffer (only `feet_dragging` reads `contact_filt`, and it does so after the update — but `contact_filt` is a local variable, so this is fine).
4. **`termination` reward fires during grace period** (the grace mask only blocks the `_get_dones` terminated output, not the reward term value).
5. **`feet_air_time` is flat-only** by explicit `(is_flat + is_non_flat * 0)` multiplier — different from the typical anymal-c implementation which is always on.
6. **No goal-reach bonus** in the reward list — `_term_goal_reached` only triggers episode termination via `_get_dones`, no positive scalar reward.
7. **`hip_pos`, `dof_error_l2`, `feet_stumble`, `feet_edge`, `feet_dragging`, `action_rate_l2`, `action_smoothness_1/2`, `base_height`, `feet_air_time` all currently 0.0 scale** — registered but disabled. Active scales: `tracking_goal_vel=1.0`, `tracking_yaw=0.5`, `lin_vel_z_l2=-1.0`, `ang_vel_xy_l2=-0.05`, `orientation_l2=-1.0`, `dof_acc_l2=-2.5e-7`, `collision=-10.0`, `delta_torques=-1.0e-7`, `torques_l2=-1e-5`, `termination=-100.0`.

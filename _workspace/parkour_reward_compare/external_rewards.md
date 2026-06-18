# External Parkour Rewards Analysis — `parkour_isaaclab/envs/mdp/rewards.py`

> **Source**: `/home/lgb/Isaaclab_Parkour/parkour_isaaclab/envs/mdp/rewards.py` (221 lines)
> **Scale config**: `/home/lgb/Isaaclab_Parkour/parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py` (Teacher / Student RewardsCfg)
> **Analyst**: ext-analyzer · team `parkour-reward-compare`

---

## 0. Meta — File-level Architecture

### 0.1 Imports
```python
import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors    import ContactSensor
from isaaclab.assets     import Articulation
from isaaclab.utils.math import euler_xyz_from_quat, wrap_to_pi, quat_apply
from parkour_isaaclab.envs.mdp.parkours import ParkourEvent
from collections.abc import Sequence

if TYPE_CHECKING:
    from parkour_isaaclab.envs       import ParkourManagerBasedRLEnv
    from isaaclab.managers           import RewardTermCfg

import cv2          # used only for commented-out edge-mask debug viz
import numpy as np  # idem
```

### 0.2 MDP Manager Pattern
- Uses standard **IsaacLab manager-based MDP** style.
  - Stateless functions = plain `def reward_xxx(env, ...)` returning `torch.Tensor[(N,)]`.
  - Stateful terms = subclass `ManagerTermBase` with `__init__` / `reset(env_ids)` / `__call__`.
- Reward terms are wired in cfg via `RewTerm(func=..., weight=..., params={...})`.
- Env type is `ParkourManagerBasedRLEnv` (custom subclass exposing `env.parkour_manager`, `env.action_manager`, `env.command_manager`).
- Per-env terrain context is obtained through `env.parkour_manager.get_term(parkour_name)` → `ParkourEvent` (provides `target_pos_rel`, `target_yaw`, `env_per_terrain_name`, `terrain`, terrain pixel grid).

### 0.3 Helper / Shared State
- No free-standing helper functions — everything is either a reward function or a manager-term class.
- Stateful classes own their previous-step buffers:
  - `reward_feet_edge` → `feet_at_edge` (also read back by `scripts/rsl_rl/evaluation.py`).
  - `reward_action_rate` → `previous_actions[N, 2, num_joints]`.
  - `reward_dof_acc` → `previous_joint_vel[N, 2, num_joints]` + `self.dt = decimation * sim.dt`.
  - `reward_delta_torques` → `previous_torque[N, 2, num_joints]`.
  - All implement `reset(env_ids)` to zero their slots on env-reset.

### 0.4 Reward Roster (14 terms — all wired in `TeacherRewardsCfg`; only `reward_collision` is also in `StudentRewardsCfg`)

| # | Reward | Type | Sign | Teacher weight | Student weight |
|---|---|---|---:|---:|---:|
| 1 | `reward_feet_edge` | class | − | `-1.0` | n/a |
| 2 | `reward_torques` | fn | − | `-1.0e-5` | n/a |
| 3 | `reward_dof_error` | fn | − | `-0.04` | n/a |
| 4 | `reward_hip_pos` | fn | − | `-0.5` | n/a |
| 5 | `reward_ang_vel_xy` | fn | − | `-0.05` | n/a |
| 6 | `reward_action_rate` | class | − | `-0.1` | n/a |
| 7 | `reward_dof_acc` | class | − | `-2.5e-7` | n/a |
| 8 | `reward_lin_vel_z` | fn | − | `-1.0` | n/a |
| 9 | `reward_orientation` | fn | − | `-1.0` | n/a |
| 10 | `reward_feet_stumble` | fn | − | `-1.0` | n/a |
| 11 | `reward_tracking_goal_vel` | fn | + | `+1.5` | n/a |
| 12 | `reward_tracking_yaw` | fn | + | `+0.5` | n/a |
| 13 | `reward_delta_torques` | class | − | `-1.0e-7` | n/a |
| 14 | `reward_collision` | fn | − | `-10.0` | `-0.0` |

> Source for weights: `parkour_mdp_cfg.py:108–244`. Student stage zeroes `reward_collision` and registers no other reward (distillation stage relies on imitation loss).

---

## 1. Term-by-term Detail

### 1. `reward_feet_edge` (class)
**Signature**
```python
class reward_feet_edge(ManagerTermBase):
    def __call__(self, env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg, parkour_name: str)
```
**State / sensor sources**
- `self.asset.data.body_state_w[:, asset_cfg.body_ids, 0:2]` → world XY of each foot body.
- `self.contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]`  → current (idx 0) foot contact force `(N, 4, 3)`.
- `self.contact_sensor.data.net_forces_w_history[:, -1, sensor_cfg.body_ids]` → last history-buffer entry `(N, 4, 3)`.
- `self.parkour_event.terrain.terrain_levels` → per-env terrain difficulty curriculum integer.
- `self.x_edge_masks_tensor` → precomputed boolean pixel grid `(total_width_pixels, total_length_pixels)` of all "edge" cells (built from `terrain_generator_class.x_edge_maskes` at init).
- Grid indexing constants captured at `__init__`: `horizontal_scale`, `rows_offset = size_x * num_rows / 2`, `cols_offset = size_y * num_cols / 2`.

**Formula**
```
fx = round((foot_x + rows_offset) / horizontal_scale).long().clip(0, W-1)
fy = round((foot_y + cols_offset) / horizontal_scale).long().clip(0, H-1)
feet_at_edge   = x_edge_masks_tensor[fx, fy]              # (N, 4) bool
contact_now    = ||F_t||_2  > 2.0                          # (N, 4) bool
contact_prev   = ||F_{t-H}||_2 > 2.0                       # (N, 4) bool
contact_filt   = contact_now OR contact_prev               # (N, 4) bool
self.feet_at_edge = contact_filt AND feet_at_edge          # (N, 4) bool
rew = (terrain_levels > 3) * sum_{foot}(feet_at_edge)      # (N,)
```
**Return shape** `(num_envs,)` (int-cast from bool sum; auto-cast on weight multiply).
**Scale** Teacher `-1.0` (penalty active only when terrain curriculum level > 3).

---

### 2. `reward_torques` (fn)
**Signature** `reward_torques(env, asset_cfg=SceneEntityCfg("robot"))`
**State** `asset.data.applied_torque` — per-joint torque applied by actuator, shape `(N, J)`.
**Formula** `sum_{j} torque_j^2`  →  `(N,)`
**Scale** Teacher `-1.0e-5`.

---

### 3. `reward_dof_error` (fn)
**Signature** `reward_dof_error(env, asset_cfg=SceneEntityCfg("robot"))`
**State** `asset.data.joint_pos`, `asset.data.default_joint_pos` `(N, J)`.
**Formula** `sum_{j} (joint_pos_j − default_joint_pos_j)^2`
**Return** `(N,)`
**Scale** Teacher `-0.04`.

---

### 4. `reward_hip_pos` (fn)
**Signature** `reward_hip_pos(env, asset_cfg=SceneEntityCfg("robot", joint_names=".*_hip_joint"))`
**State** `asset.data.joint_pos[:, asset_cfg.joint_ids]`, `asset.data.default_joint_pos[:, asset_cfg.joint_ids]` `(N, 4)`.
**Formula** `sum_{j ∈ hip joints} (joint_pos_j − default_pos_j)^2`
**Return** `(N,)`
**Scale** Teacher `-0.5` (hip joint set `".*_hip_joint"` from cfg).

---

### 5. `reward_ang_vel_xy` (fn)
**Signature** `reward_ang_vel_xy(env, asset_cfg=SceneEntityCfg("robot"))`
**State** `asset.data.root_ang_vel_b[:, :2]` — root **body-frame** angular velocity, x/y components.
**Formula** `sum_{i∈{x,y}} ω_i^2`  →  `(N,)`
**Scale** Teacher `-0.05`.

---

### 6. `reward_action_rate` (class)
**Signature** `__call__(env, asset_cfg)`
**State** internal `previous_actions[N, 2, J]` (rolling window of last two actions).
**Source for new action** `env.action_manager.get_term('joint_pos').raw_actions` `(N, J)`.
**Formula**
```
prev[:,0,:] = prev[:,1,:]
prev[:,1,:] = raw_actions
rew = || prev[:,1,:] − prev[:,0,:] ||_2     # L2 over joint dim → (N,)
```
**Note** Uses `torch.norm(..., dim=1)` (not sum-of-squares). Reset zeros both slots.
**Scale** Teacher `-0.1`.

---

### 7. `reward_dof_acc` (class)
**Signature** `__call__(env, asset_cfg)`
**State** internal `previous_joint_vel[N, 2, J]`, `self.dt = env.cfg.decimation * env.cfg.sim.dt`.
**Source** `asset.data.joint_vel` `(N, J)`.
**Formula**
```
prev[:,0,:] = prev[:,1,:]
prev[:,1,:] = joint_vel
acc = (prev[:,1,:] − prev[:,0,:]) / dt
rew = sum_{j} acc_j^2                       # (N,)
```
**Scale** Teacher `-2.5e-7`.

---

### 8. `reward_lin_vel_z` (fn)
**Signature** `reward_lin_vel_z(env, parkour_name, asset_cfg=SceneEntityCfg("robot"))`
**State**
- `asset.data.root_lin_vel_b[:, 2]` — root body-frame z-velocity.
- `parkour_event.env_per_terrain_name` — per-env recent-terrain-name buffer; last column `[:, -1]` = current terrain.
**Formula**
```
rew = v_z^2                                  # (N,)
rew[ terrain_names[:, -1] != 'parkour_flat' ] *= 0.5   # halve penalty on non-flat (obstacle) terrains
```
**Scale** Teacher `-1.0`.

---

### 9. `reward_orientation` (fn)
**Signature** `reward_orientation(env, parkour_name, asset_cfg=SceneEntityCfg("robot"))`
**State**
- `asset.data.projected_gravity_b[:, :2]` — gravity projected into body frame, x/y components (flat = (0,0)).
- `parkour_event.env_per_terrain_name` last column.
**Formula**
```
rew = sum_{i∈{x,y}} g_proj_i^2               # (N,)
rew[ terrain_names[:, -1] != 'parkour_flat' ] = 0.   # disable on obstacle terrains
```
**Scale** Teacher `-1.0` (effectively only penalizes tilt on `parkour_flat`).

---

### 10. `reward_feet_stumble` (fn)
**Signature** `reward_feet_stumble(env, sensor_cfg)`
**State** `contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]` `(N, 4, 3)` — current-step foot contact forces in world frame.
**Formula**
```
F_xy  = ||F[:,:, :2]||_2                     # (N, 4) horizontal force magnitude
F_z   = |F[:,:, 2]|                          # (N, 4) vertical magnitude
flag  = (F_xy > 4 * F_z)                     # per foot bool
rew   = any_over_foot(flag).float()          # (N,)
```
i.e. fires if **any** foot has horizontal contact force exceeding 4× the vertical force (toe-stub heuristic).
**Scale** Teacher `-1.0`.

---

### 11. `reward_tracking_goal_vel` (fn)
**Signature** `reward_tracking_goal_vel(env, parkour_name, asset_cfg=SceneEntityCfg("robot"))`
**State**
- `parkour_event.target_pos_rel` `(N, 2)` — XY vector from base to next parkour goal.
- `asset.data.root_vel_w[:, :2]` `(N, 2)` — root **world-frame** linear velocity, xy.
- `env.command_manager.get_command('base_velocity')[:, 0]` `(N,)` — commanded forward speed.
**Formula**
```
target_dir = target_pos_rel / (||target_pos_rel||_2 + 1e-5)   # unit vector toward goal (N, 2)
proj_vel   = <target_dir, root_vel_w_xy>                       # (N,)
rew        = min(proj_vel, command_vel) / (command_vel + 1e-5) # (N,)
```
Clamps the "useful" forward speed at the commanded value, then normalizes — saturates at 1.0 when robot moves toward goal at exactly the commanded speed.
**Scale** Teacher `+1.5` (positive reward).

---

### 12. `reward_tracking_yaw` (fn)
**Signature** `reward_tracking_yaw(env, parkour_name, asset_cfg=SceneEntityCfg("robot"))`
**State**
- `asset.data.root_quat_w` `(N, 4)` quaternion in `(w, x, y, z)` order — code accesses `q[:,0]` (w), `q[:,1]` (x), `q[:,2]` (y), `q[:,3]` (z).
- `parkour_event.target_yaw` `(N,)` — desired yaw toward next goal.
**Formula**
```
yaw = atan2( 2 * (q_w*q_z + q_x*q_y),
             1 − 2 * (q_y^2 + q_z^2) )      # standard ZYX yaw from quaternion
rew = exp( − | target_yaw − yaw | )         # (N,)
```
> Note: uses raw difference, no `wrap_to_pi` — discontinuity at ±π.
**Scale** Teacher `+0.5` (positive reward).

---

### 13. `reward_delta_torques` (class)
**Signature** `__call__(env, asset_cfg)`
**State** internal `previous_torque[N, 2, J]`.
**Source** `asset.data.applied_torque` `(N, J)`.
**Formula**
```
prev[:,0,:] = prev[:,1,:]
prev[:,1,:] = applied_torque
rew = sum_{j} (prev[:,1,:] − prev[:,0,:])^2  # (N,)
```
Torque jerk penalty (Δτ²).
**Scale** Teacher `-1.0e-7`.

---

### 14. `reward_collision` (fn)
**Signature** `reward_collision(env, sensor_cfg)`
**State** `contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]` `(N, B, 3)`. Cfg `sensor_cfg.body_names = ["base", ".*_calf", ".*_thigh"]` (i.e. base + all calves + all thighs = 9 bodies for Go2).
**Formula**
```
rew = sum_{b}  1{ ||F_b||_2 > 0.1 }          # (N,) integer count of "colliding" tracked bodies
```
**Scale** Teacher `-10.0`, Student `-0.0` (disabled in distillation).

---

## 2. Cross-cutting Observations (factual, no value judgement)

- **Sign convention**: 12 of 14 terms return non-negative scalars and are paired with negative weights → penalties. Only `reward_tracking_goal_vel` and `reward_tracking_yaw` carry positive weights → bonuses.
- **Curriculum gating**: `reward_feet_edge` is the only term explicitly gated by `terrain_levels > 3`. `reward_lin_vel_z` and `reward_orientation` are modulated by the per-env terrain name (`parkour_flat` vs. other).
- **History-dependent terms**: 4 terms keep their own `previous_*` buffers (action_rate, dof_acc, delta_torques) or read `net_forces_w_history[..., -1, ...]` (feet_edge). All four implement `reset(env_ids)` for env-resets.
- **Contact-force threshold values** used in this file: `2.0` N (feet_edge), `0.1` N (collision), `4×|F_z|` ratio (feet_stumble).
- **Reductions**: most penalties use `sum_{j} (·)^2` (joint-axis L2²). Exceptions: `reward_action_rate` uses `torch.norm` (L2, not squared), `reward_tracking_yaw` uses `exp(-|·|)`, `reward_feet_stumble` uses `torch.any` over feet.
- **Quat convention**: code treats `root_quat_w` as `(w, x, y, z)` — consistent with IsaacLab's `Articulation.data.root_quat_w`.
- **Frames used**: body-frame angular vel (`root_ang_vel_b`), body-frame z-lin-vel (`root_lin_vel_b[:,2]`), **world-frame** xy lin-vel for goal tracking (`root_vel_w[:, :2]`), world-frame contact forces, world-frame foot positions (`body_state_w`).
- **Student vs Teacher**: `StudentRewardsCfg` registers only `reward_collision` with weight `-0.0` — effectively no reward shaping during distillation; learning signal comes from imitation loss elsewhere.

---

## 3. File / line cross-reference

| Term | Definition (line) | Cfg wiring (file:line, weight) |
|---|---|---|
| `reward_feet_edge` | rewards.py:19–64 | parkour_mdp_cfg.py:148, weight `-1.0` |
| `reward_torques` | rewards.py:66–71 | parkour_mdp_cfg.py:157, weight `-1e-5` |
| `reward_dof_error` | rewards.py:73–78 | parkour_mdp_cfg.py:164, weight `-0.04` |
| `reward_hip_pos` | rewards.py:80–86 | parkour_mdp_cfg.py:171, weight `-0.5` |
| `reward_ang_vel_xy` | rewards.py:88–93 | parkour_mdp_cfg.py:178, weight `-0.05` |
| `reward_action_rate` | rewards.py:95–112 | parkour_mdp_cfg.py:185, weight `-0.1` |
| `reward_dof_acc` | rewards.py:114–133 | parkour_mdp_cfg.py:192, weight `-2.5e-7` |
| `reward_lin_vel_z` | rewards.py:135–145 | parkour_mdp_cfg.py:199, weight `-1.0` |
| `reward_orientation` | rewards.py:147–157 | parkour_mdp_cfg.py:207, weight `-1.0` |
| `reward_feet_stumble` | rewards.py:159–167 | parkour_mdp_cfg.py:215, weight `-1.0` |
| `reward_tracking_goal_vel` | rewards.py:169–182 | parkour_mdp_cfg.py:222, weight `+1.5` |
| `reward_tracking_yaw` | rewards.py:184–194 | parkour_mdp_cfg.py:230, weight `+0.5` |
| `reward_delta_torques` | rewards.py:196–213 | parkour_mdp_cfg.py:238, weight `-1e-7` |
| `reward_collision` | rewards.py:215–221 | parkour_mdp_cfg.py:108 (student `-0.0`) & :141 (teacher `-10.0`) |

— end of report —

# IsaacLab Parkour Sim Cfg — Analysis Report

**Scope**: `ParkourEnvCfg` and full transitive import chain (Go2 articulation, terrain configs, sensors, randomization).
**Source root**: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`
**Read-only audit. No values modified.**

---

## 0. Import Chain (parkour_env_cfg.py imports)

`/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py:10-38`

```
parkour_env_cfg.py
├── isaaclab.envs.mdp                          (mdp:10)            → randomize_rigid_body_material
├── isaaclab.sim                               (sim_utils:11)      → RigidBodyMaterialCfg / MdlFileCfg / UsdFileCfg / RigidBodyPropertiesCfg / ArticulationRootPropertiesCfg
├── isaaclab.terrains                          (terrain_gen:12)    → MeshParkour*TerrainCfg (used as terrain_gen.MeshParkourHurdleTerrainCfg …)
├── isaaclab.actuators.DCMotorCfg              (:13)
├── isaaclab.assets.ArticulationCfg            (:14)
├── isaaclab.envs.DirectRLEnvCfg, ViewerCfg    (:15)
├── isaaclab.managers.EventTermCfg as EventTerm, SceneEntityCfg  (:16-17)
├── isaaclab.scene.InteractiveSceneCfg         (:18)
├── isaaclab.sensors.ContactSensorCfg, RayCasterCfg, patterns  (:19)
├── isaaclab.sim.PhysxCfg, SimulationCfg       (:20)
├── isaaclab.terrains.FlatPatchSamplingCfg, TerrainImporterCfg  (:21)
├── isaaclab.terrains.terrain_generator_cfg.TerrainGeneratorCfg (:22)
├── isaaclab.utils.configclass                 (:23)
├── isaaclab_assets.robots.unitree.UNITREE_GO2_CFG (:28)
│       → /home/lgb/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:140
└── .parkour_terrains                          (:30-38)
        ├── parkour_jump_hurdle_terrain        (terrain build function)
        ├── MeshParkourSteppingStonesTerrainCfg
        ├── MeshParkourBalanceBeamTerrainCfg
        ├── MeshParkourCrawlTerrainCfg
        ├── MeshParkourSlopeTerrainCfg
        ├── MeshParkourZigzagHurdlesTerrainCfg
        └── MeshParkourRoughBlocksTerrainCfg
```

Sibling files:
- `parkour_env.py` (92,647 B) – DirectRLEnv impl
- `parkour_terrains.py` (55,433 B) – Mesh*TerrainCfg + builder functions
- `agents/rsl_rl_ppo_cfg.py` – PPO runner cfg
- `CLAUDE.md` – task notes

There is **no separate `scene_cfg.py` / `terrain_cfg.py` / `mdp.py`**; everything lives in `parkour_env_cfg.py` + `parkour_terrains.py`.

---

## 1. SimulationCfg
`parkour_env_cfg.py:341-359`

| Field | Value | Unit |
|---|---|---|
| `dt` (cfg.dt mirror) | `1/200 = 0.005` | s (line 341) |
| `sim.dt` | `1/200 = 0.005` | s (line 343) |
| `sim.render_interval` | `4` | physics steps / render (line 344) |
| `sim.gravity` | *(default — not overridden)* | inherits SimulationCfg default `(0.0, 0.0, -9.81)` |
| `physics_material.friction_combine_mode` | `"average"` | (line 348; alt `"multiply"` commented out :346) |
| `physics_material.restitution_combine_mode` | `"average"` | (line 349) |
| `physics_material.static_friction` | `1.0` | (line 350) |
| `physics_material.dynamic_friction` | `1.0` | (line 351) |
| `physics_material.restitution` | `0.0` | (line 352) |
| `physx.gpu_max_rigid_patch_count` | `2**23 = 8 388 608` | (line 355) |
| `physx.gpu_found_lost_pairs_capacity` | `2**23` | (line 356) |
| `physx.gpu_total_aggregate_pairs_capacity` | `2**23` | (line 357) |

Other PhysX solver fields (timestep type, bounce threshold, friction offset, GPU buffers not listed) are **left at IsaacLab `PhysxCfg` defaults**.

> Effective policy rate: `1 / (dt * decimation) = 1 / (0.005 * 4) = 50 Hz`.

---

## 2. Scene
`parkour_env_cfg.py:385`

```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=4096, env_spacing=4.0, replicate_physics=True
)
```

| Field | Value | Unit |
|---|---|---|
| `num_envs` | `4096` | – |
| `env_spacing` | `4.0` | m |
| `replicate_physics` | `True` | – |

Lights / sky / ground:
- No explicit `lights`, `sky_light`, or `ground` cfg attributes on the scene.
- Light creation is delegated to env runtime (`parkour_env.py:349`: `light_cfg.func("/World/Light", light_cfg)`).
- "Ground" is provided by the `terrain` TerrainImporter, **not** by a separate `GroundPlaneCfg`.

---

## 3. TerrainImporterCfg + TerrainGeneratorCfg

### 3a. TerrainImporterCfg
`parkour_env_cfg.py:362-382`

| Field | Value |
|---|---|
| `prim_path` | `/World/ground` |
| `terrain_type` | `"generator"` |
| `terrain_generator` | `PARKOUR_TERRAINS_CFG` (defined :44-246) |
| `max_init_terrain_level` | `3` (line 366) |
| `collision_group` | `-1` |
| `physics_material` | `friction_combine_mode="average"`, `restitution_combine_mode="average"`, `static_friction=1.0`, `dynamic_friction=1.0`, `restitution=0.0` (:368-376) |
| `visual_material` | MDL `{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl`, `project_uvw=True` (:377-380) |
| `debug_vis` | `False` |
| `color` | *(not specified — inherits TerrainImporterCfg default)* |

### 3b. TerrainGeneratorCfg (`PARKOUR_TERRAINS_CFG`)
`parkour_env_cfg.py:44-246`

| Field | Value | Unit |
|---|---|---|
| `size` | `(20.0, 4.0)` | m × m (per tile) |
| `border_width` | `20.0` | m |
| `num_rows` | `11` | – |
| `num_cols` | `40` | – |
| `horizontal_scale` | `0.05` | m / heightfield px |
| `vertical_scale` | `0.005` | m / unit |
| `slope_threshold` | `0.75` | – |
| `use_cache` | `False` | – |
| `curriculum` | `True` (row 0 easiest → row 10 hardest) | – |

### 3c. Sub-terrains (proportions / class indices)
`parkour_env_cfg.py:54-245` + class-index constants `:253-266`

| Name | Class ID | Proportion | Cfg type (line) | Key params |
|---|---|---|---|---|
| `parkour_flat`            | 0 | **0.1** | `MeshParkourHurdleTerrainCfg` (:61) | `flat=True`, `num_hurdles=8`, `num_goals=8`, `hurdle_height_range=(0.0, 0.0)`, `x_spacing_range=(1.0, 1.5)` |
| `parkour_hurdle`          | 1 | **0.2** | `MeshParkourHurdleTerrainCfg` (:77) | `num_hurdles=8`, `num_goals=8`, `hurdle_height_range=(0.05, 0.30)`, `x_spacing_range=(1.0, 1.5)` |
| `parkour_step`            | 2 | **0.2** | `MeshParkourStepTerrainCfg` (:92) | `num_steps=8`, `num_goals=8`, `x_length_range=(0.4, 0.8)`, `step_height_range=(0.10, 0.45)` |
| `parkour_gap`             | 3 | **0.3** | `MeshParkourGapTerrainCfg` (:106) | `num_gaps=8`, `num_goals=8`, `gap_length_range=(0.05, 0.5)`, `platform_length_range=(1.2, 1.6)` |
| `parkour_stair`           | 4 | **0.2** | `MeshParkourStairTerrainCfg` (:120) | `stair_width_range=(0.25, 0.40)`, `stair_height_range=(0.05, 0.20)`, `num_goals=8` |
| `parkour_stepping_stones` | 5 | 0.0 | `MeshParkourSteppingStonesTerrainCfg` (:143) | `platform_length=2.5`, `num_stones=8`, `stone_size_xy_range=(0.20, 0.40)`, `stone_height_range=(0.05, 0.15)`, `gap_length_range=(0.20, 0.45)`, `lateral_jitter_range=(0.0, 0.30)` |
| `parkour_balance_beam`    | 6 | 0.0 | `MeshParkourBalanceBeamTerrainCfg` (:160) | `platform_length=2.5`, `platform_height=0.15`, `beam_width_range=(0.20, 0.50)`, `beam_height=0.15`, `max_segments=3`, `y_shift_per_segment_range=(0.0, 0.40)` |
| `parkour_crawl`           | 7 | 0.0 | `MeshParkourCrawlTerrainCfg` (:177) | `platform_length=2.5`, `num_crawls=3`, `ceiling_height_range=(0.28, 0.50)`, `ceiling_length_x=1.2`, `ceiling_thickness=0.10`, `ceiling_top_extra=0.50`, `corridor_width=1.2`, `side_wall_height=1.0`, `side_walls=True`, `x_spacing_range=(1.0, 2.0)` |
| `parkour_slope`           | 8 | 0.0 | `MeshParkourSlopeTerrainCfg` (:198) | `platform_length=2.5`, `slope_angle_deg_range=(5.0, 25.0)`, `slope_length=4.0`, `flat_top_length=1.5` |
| `parkour_zigzag_hurdles`  | 9 | 0.0 | `MeshParkourZigzagHurdlesTerrainCfg` (:213) | `platform_length=2.5`, `num_hurdles=6`, `hurdle_thickness=0.30`, `hurdle_height_range=(0.10, 0.25)`, `corridor_width=2.0`, `x_spacing_range=(1.5, 2.4)` |
| `parkour_rough_blocks`    | 10 | 0.0 | `MeshParkourRoughBlocksTerrainCfg` (:230) | `platform_length=2.5`, `block_size=0.30`, `block_height_range=(0.0, 0.10)`, `block_density_range=(0.6, 0.9)` |

**Active sub-terrain sum**: `0.1 + 0.2 + 0.2 + 0.3 + 0.2 = 1.0`.

All sub-terrains include `flat_patch_sampling = {"init_positions": FlatPatchSamplingCfg(num_patches=2, patch_radius=0.5, max_height_diff=0.05)}`.

`num_goals=8` is explicit on every sub-terrain (matches `ParkourEnvCfg.num_goals = 8`, line 519).

---

## 4. Robot ArticulationCfg

### 4a. UNITREE_GO2_CFG (external)
`/home/lgb/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:140-181`

```python
UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",  # :142
        activate_contact_sensors=True,                                  # :143
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),                                                              # :144-152
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,                               # :154 (Go2 unique vs Go1/A1=False)
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.27),                                           # :158
        joint_pos={
            ".*L_hip_joint": 0.1,                                       # :160
            ".*R_hip_joint": -0.1,                                      # :161
            "F[L,R]_thigh_joint": 0.8,                                  # :162
            "R[L,R]_thigh_joint": 1.0,                                  # :163
            ".*_calf_joint": -1.5,                                      # :164
        },
        joint_vel={".*": 0.0},                                          # :166
    ),
    soft_joint_pos_limit_factor=0.9,                                    # :168
    actuators={
        "base_legs": DCMotorCfg(                                        # :170-179
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
            armature=0.01,
        ),
    },
)
```

### 4b. ParkourEnvCfg robot override
`parkour_env_cfg.py:391` — `robot = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")`

### 4c. Parkour actuator override (`__post_init__`)
`parkour_env_cfg.py:404-451`

**Mode switch**: `_actuator_mode = 2` (`:404`) — **active branch is mode 2**.

| Field | mode=1 (inactive) | mode=2 (ACTIVE) | Notes |
|---|---|---|---|
| `stiffness` (Kp) | 40.0 | **25.0** (:423) | N·m/rad |
| `damping` (Kd) | 1.0 | **0.5** (:424) | N·m·s/rad |
| `friction` | 0.0 | **0.0** (:425) | – |
| `saturation_effort` | 35.0 | **23.5** (:426) | N·m (scalar) |
| `effort_limit` (hip / thigh / calf) | 35 / 40 / 40 | **23.5 / 23.5 / 23.5** (:427-431) | N·m, dict per joint regex |
| `velocity_limit` (hip / thigh / calf) | 52.4 / 30.1 / 30.1 | **30.0 / 30.0 / 30.0** (:432-436) | rad/s |
| `armature` | – (only in __post_init__) | **0.01** (:450) | kg·m² |
| `joint_names_expr` | – | `[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]` (:443) | – |

`__post_init__` rebuilds `self.robot.actuators` with `dict(self.robot.actuators)` (avoids mutating shared global) then assigns a fresh `DCMotorCfg` (:441-451).

> Mode=2 effectively **restores stock Go2 numbers** (stiffness=25, damping=0.5, effort=23.5, velocity=30.0, armature=0.01) over the inherited `base_legs` actuator.

---

## 5. Sensors

### 5a. ContactSensorCfg
`parkour_env_cfg.py:454-459`

| Field | Value |
|---|---|
| `prim_path` | `/World/envs/env_.*/Robot/.*` |
| `history_length` | `3` |
| `update_period` | `0.005` s |
| `track_air_time` | `True` |
| `debug_vis` | *(default — not set)* |
| body filter / `filter_prim_paths_expr` | *(not specified)* |

### 5b. RayCasterCfg (height scanner)
`parkour_env_cfg.py:461-468`

| Field | Value |
|---|---|
| `prim_path` | `/World/envs/env_.*/Robot/base` |
| `offset.pos` | `(0.375, 0.0, 20.0)` m (forward of base, raised) |
| `ray_alignment` | `"yaw"` |
| `pattern_cfg` | `patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0])` |
| `debug_vis` | `False` |
| `mesh_prim_paths` | `["/World/ground"]` |
| `update_period` | *(default 0.0)* |
| `drift_range` | *(default)* |

Grid math: `(1.6/0.1 + 1) * (1.0/0.1 + 1) = 17 * 11 = 187` → matches `num_scan_obs=187` (line 336).

### 5c. Cameras
None defined (no `Camera*Cfg` in this file).

---

## 6. DirectRL essentials
`parkour_env_cfg.py:316-339`

| Field | Value | Source line |
|---|---|---|
| `episode_length_s` | `20.0` s | :317 |
| `decimation` | `4` | :318 |
| `action_scale` | `0.25` | :319 |
| `action_space` | `12` | :320 |
| `clip_actions` | `10.0` | :321 |
| `observation_space` | `42` (policy obs dim; runtime overrides with dict obs_groups) | :331 |
| `state_space` | `0` | :332 |
| `num_proprio` | `42` (= 3+1+1+1+12+12+12 per comment) | :335 |
| `num_scan_obs` | `187` | :336 |
| `num_priv_obs` | `14` (lin_vel_b 3 + ang_vel_b 3 + foot_friction 4 feet × 2 = 8) | :337 |
| `history_len` | `10` | :338 |

Comment at :323-330 documents the runner-side dict obs groups:
- `policy` 46 (3+3+2+2+12+12+12)
- `scan` 187
- `priv` 4 (placeholder)
- `history` `history_len * num_proprio = 10 * 46 = 460`
- `critic = policy + scan + priv + history = 46 + 187 + 4 + 460 = 697`

> Note the inconsistency between `num_proprio = 42` (:335) and the inline policy-46 doc-comment (:325); reported as-is.

---

## 7. `reward_scales` (full dump)
`parkour_env_cfg.py:478-501`

```python
reward_scales: dict = {
    "tracking_goal_vel":     1.5,        # was 1.2, Genesis original=1.5
    "tracking_yaw":          0.5,        # was 0.7, Genesis original=0.5
    "tracking_lin_vel_xy_exp": 0.0,      # parkour-specific weakening
    "tracking_ang_vel_z_exp":  0.0,      # parkour-specific weakening
    "lin_vel_z_l2":         -1.0,
    "ang_vel_xy_l2":        -0.05,
    "orientation_l2":       -1.0,
    "dof_acc_l2":           -2.5e-7,
    "collision":           -10.0,
    "action_rate_l2":       -0.1,        # comment notes prior -0.1 (10x error)
    "delta_torques":        -1.0e-7,
    "torques_l2":           -1.0e-5,
    "hip_pos":              -0.0,
    "dof_error_l2":         -0.0,
    "feet_stumble":         -1.0,
    "feet_edge":            -1.0,
    "termination":        -100.0,        # was -0.0, Genesis original=-100.0
    "feet_dragging":        -0.0,
    "action_smoothness_1":  -0.0,
    "action_smoothness_2":  -0.0,
    "base_height":           0.0,        # flat-only; disabled by default
    "feet_air_time":         0.0,        # NEW 2026-05-13 (anymal_c/R_Skeleton pattern)
}
```

Companion reward parameters:
| Field | Value | Line |
|---|---|---|
| `tracking_sigma` | `0.2` | :504 |
| `yaw_reward_speed_lower` | `0.05` m/s | :509 |
| `yaw_reward_speed_upper` | `0.15` m/s | :510 |
| `dragging_velocity_threshold` | `0.05` m/s | :513 |
| `base_height_target` | `0.34` m | :516 |

---

## 8. EventCfg / Domain Randomization
`parkour_env_cfg.py:274-288` (class), `:388` (instance binding)

```python
@configclass
class EventCfg:
    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range":  (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range":      (0.0, 0.0),
            "num_buckets":            64,
        },
    )

events: EventCfg = EventCfg()   # :388
```

**Only one randomization event**: per-foot static/dynamic friction at startup (per-env, 64-bucket LUT). Restitution range is degenerate (0,0).

**No other domain randomization fields** were found:
- No `randomize_friction_range` / `add_noise` / `noise_scales` / `motor_strength_range` / `added_mass_range` cfg attributes.
- `parkour_env.py` confirms: only references to randomization are comments around `randomize_rigid_body_material` (`parkour_env.py:192, 840-861`); no env-side noise application referencing cfg-level noise scales.
- Cfg comments at `:530-533` explicitly mark the following as **future work (commented out, not active)**:
  - `friction_range: [0.6, 2.0]`
  - `added_mass_range: [0.0, 5.0]`
  - `motor_strength_range: [0.8, 1.2]`

---

## 9. Command / goal configuration

### 9a. `command_cfg`
`parkour_env_cfg.py:471-475`

```python
command_cfg: dict = {
    "lin_vel_x_range": [0.3, 1.0],   # forward only
    "lin_vel_y_range": [0.0, 0.0],   # no lateral
    "ang_vel_range":   [0.0, 0.0],   # no yaw
}
```

**No** `resample_period` / `forward_vel_curriculum` cfg field. Env reads ranges directly (`parkour_env.py:1290-1295`); resampling, if any, is handled in env code, not exposed via cfg.

### 9b. Parkour goal cfg
`parkour_env_cfg.py:519-528`

| Field | Value | Unit |
|---|---|---|
| `num_goals` | `8` | – |
| `num_future_goal_obs` | `2` (lookahead) | – |
| `goal_distance` | `1.0` | m |
| `next_goal_threshold` | `0.2` | m |
| `reach_goal_delay` | `0.1` | s |
| `goal_z` | `0.3` | m |
| `termination_height` | `-0.2` | m (relative offset; cfg comment at `:526` clarifies grace-step gating) |
| `termination_grace_steps` | `5` | policy steps |
| `max_tilt` | `1.5` | rad |
| `terrain_curriculum` | `True` | – |

---

## 10. Viewer / debug cfg
`parkour_env_cfg.py:301-314`

```python
viewer: ViewerCfg = ViewerCfg(
    origin_type="world",
    asset_name="robot",
    env_index=0,
    eye=(0.0, -2.5, 0.8),
    lookat=(0.0, 0.0, 0.3),
)
debug_vis: bool = False                          # :310
enable_keyboard_view_switch: bool = True         # :311
debug_print_contacts: bool = False               # :312 (P-key toggles at runtime)
debug_vis_edge_mask: bool = False                # :313 (play.py forces True)
debug_vis_edge_mask_radius_m: float = 5.0        # :314
```

---

## Appendix A — Class hierarchy snapshot

```
ParkourEnvCfg(DirectRLEnvCfg)
├── viewer: ViewerCfg
├── debug_vis / enable_keyboard_view_switch / debug_print_contacts / debug_vis_edge_mask*
├── episode_length_s, decimation, action_scale, action_space, clip_actions
├── observation_space, state_space, num_proprio, num_scan_obs, num_priv_obs, history_len
├── dt, sim: SimulationCfg(physx=PhysxCfg, physics_material=RigidBodyMaterialCfg)
├── terrain: TerrainImporterCfg(terrain_generator=PARKOUR_TERRAINS_CFG)
├── scene: InteractiveSceneCfg(num_envs=4096, env_spacing=4.0)
├── events: EventCfg(foot_physics_material=EventTerm[startup])
├── robot: ArticulationCfg (UNITREE_GO2_CFG.replace + DCMotorCfg override in __post_init__)
├── contact_sensor: ContactSensorCfg
├── height_scanner: RayCasterCfg(GridPatternCfg)
├── command_cfg: dict[lin_vel_x/y, ang_vel]
├── reward_scales: dict (22 keys)
├── tracking_sigma, yaw_reward_speed_lower/upper, dragging_velocity_threshold, base_height_target
└── num_goals, num_future_goal_obs, goal_distance, next_goal_threshold, reach_goal_delay,
    goal_z, termination_height, termination_grace_steps, max_tilt, terrain_curriculum
```

## Appendix B — Effective control-loop rates

| Quantity | Value |
|---|---|
| Physics dt | 1/200 s = 5 ms |
| Decimation | 4 |
| Policy dt | 4 × 5 ms = 20 ms (50 Hz) |
| Render interval | 4 physics steps (= 1 per policy step) |
| Episode length | 20 s → 1000 policy steps / episode |
| Contact sensor update | 5 ms (= every physics step) |

## Appendix C — Files inspected

| Path | Purpose |
|---|---|
| `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` | Primary config (533 lines) |
| `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` | Env impl (cfg consumer; randomization & cfg usage cross-checked) |
| `/home/lgb/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py` | `UNITREE_GO2_CFG` definition (line 140-181) |
| `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_terrains.py` | (referenced, sub-terrain Cfg types — not opened; parameters consumed in `parkour_env_cfg.py:54-245`) |
| `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/CLAUDE.md` | Task-specific notes (header context) |

---

*End of audit. Values not present in source explicitly marked as `(default)` or `(not specified)`. No mutations performed.*

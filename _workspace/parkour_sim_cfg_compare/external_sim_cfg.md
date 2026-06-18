# External Parkour Sim Config Dump

**Source repo**: `/home/lgb/Isaaclab_Parkour/` (manager-based RL, Extreme-Parkour adaptation for IsaacLab)
**Workflow target dumped**: Teacher (primary); Student/Eval/Play variants noted where they override.
**Mode**: read-only, factual extraction.

---

## 1. File tree (cfg-relevant only)

```
/home/lgb/Isaaclab_Parkour/
├── parkour_isaaclab/                                       # framework layer
│   ├── envs/
│   │   ├── parkour_manager_based_env_cfg.py                # ParkourManagerBasedEnvCfg
│   │   ├── parkour_manager_based_rl_env_cfg.py             # ParkourManagerBasedRLEnvCfg
│   │   ├── parkour_manager_based_rl_env.py
│   │   └── mdp/
│   │       ├── observations.py                             # ExtremeParkourObservations / image_features / delta_yaw_ok
│   │       ├── events.py                                   # reset_root_state, randomize_*, push_by_setting_velocity, random_camera_position
│   │       ├── rewards.py
│   │       ├── terminations.py
│   │       ├── parkour_actions/actions_cfg.py              # DelayedJointPositionActionCfg
│   │       ├── parkour_commands/parkour_command_cfg.py     # ParkourCommandCfg
│   │       └── parkours/parkour_events_cfg.py              # ParkourEventsCfg (goal/marker mgmt)
│   ├── managers/parkour_manager_term_cfg.py                # ParkourTermCfg
│   ├── actuators/parkour_actuator_cfg.py                   # ParkourDCMotorCfg (DCMotor + saturation override)
│   └── terrains/
│       ├── parkour_terrain_generator_cfg.py                # ParkourTerrainGeneratorCfg, ParkourSubTerrainBaseCfg
│       └── extreme_parkour/
│           ├── extreme_parkour_terrains_cfg.py             # ExtremeParkour{Rough,Gap,Hurdle,Step,Demo,Terrain}Cfg
│           └── config/parkour.py                           # EXTREME_PARKOUR_TERRAINS_CFG instance
└── parkour_tasks/parkour_tasks/
    ├── default_cfg.py                                      # ParkourDefaultSceneCfg, CAMERA_CFG, CAMERA_USD_CFG, VIEWER
    └── extreme_parkour_task/config/go2/
        ├── parkour_mdp_cfg.py                              # Commands/Events/Obs/Rewards/Terminations/Actions Cfg classes
        ├── parkour_teacher_cfg.py                          # UnitreeGo2TeacherParkourEnvCfg + _EVAL + _PLAY
        └── parkour_student_cfg.py                          # UnitreeGo2StudentParkourEnvCfg + _EVAL + _PLAY
```

### Inheritance / composition chain (Teacher)

```
ManagerBasedEnvCfg               (IsaacLab core)
        ▲
ManagerBasedRLEnvCfg             (IsaacLab core)
        ▲
ParkourManagerBasedEnvCfg                 ← parkour_manager_based_env_cfg.py
ParkourManagerBasedRLEnvCfg               ← parkour_manager_based_rl_env_cfg.py
        ▲
UnitreeGo2TeacherParkourEnvCfg            ← parkour_teacher_cfg.py:33
        ├─ scene: ParkourTeacherSceneCfg (extends ParkourDefaultSceneCfg)
        │    ├─ robot ← UNITREE_GO2_CFG (overridden actuator='base_legs': ParkourDCMotorCfg)
        │    ├─ terrain ← TerrainImporterCfg(class_type=ParkourTerrainImporter,
        │    │                generator=EXTREME_PARKOUR_TERRAINS_CFG)
        │    ├─ height_scanner ← RayCasterCfg
        │    └─ contact_forces ← ContactSensorCfg
        ├─ observations: TeacherObservationsCfg (single ExtremeParkourObservations term)
        ├─ actions: ActionsCfg (DelayedJointPositionActionCfg, use_delay=False@teacher)
        ├─ commands: CommandsCfg (ParkourCommandCfg)
        ├─ rewards: TeacherRewardsCfg (12 terms)
        ├─ terminations: TerminationsCfg (1 term)
        ├─ parkours: ParkourEventsCfg (custom 'parkour' manager)
        └─ events: EventCfg (8 event terms)

UnitreeGo2TeacherParkourEnvCfg_EVAL extends Teacher → 5x5 grid, debug_vis, 256 envs
UnitreeGo2TeacherParkourEnvCfg_PLAY extends EVAL    → 16 envs, episode 60s, no push
```

---

## 2. SimulationCfg / `sim`

Set in `parkour_teacher_cfg.py:48-54` (teacher) / `parkour_student_cfg.py:54-60` (student).

| Field | Value | Source |
|-------|-------|--------|
| `sim.dt` | `0.005` (200 Hz) | `parkour_teacher_cfg.py:51` |
| `sim.render_interval` | `self.decimation` = `4` | `parkour_teacher_cfg.py:52` |
| `sim.physics_material` | `self.scene.terrain.physics_material` (re-bound from terrain) | `parkour_teacher_cfg.py:53` |
| `sim.physx.gpu_max_rigid_patch_count` | `10 * 2**18` = 2,621,440 | `parkour_teacher_cfg.py:54` |
| `sim.gravity` (default) | `(0.0, 0.0, -9.81)` | `simulation_cfg.py:371` (inherited default) |
| `sim.device` (default) | `"cuda:0"` | `simulation_cfg.py:355` |
| `sim.use_fabric` (default) | `True` | `simulation_cfg.py:392` |
| `sim.physx.solver_type` (default) | `1` (TGS) | `simulation_cfg.py:37` |

All other PhysX fields are at IsaacLab defaults (only `gpu_max_rigid_patch_count` is overridden in cfg).

---

## 3. Scene (InteractiveSceneCfg) — Teacher

`ParkourTeacherSceneCfg(num_envs=6144, env_spacing=1.)` at `parkour_teacher_cfg.py:34`.

| Field | Value | Source |
|-------|-------|--------|
| `num_envs` | `6144` (teacher); `192` (student); `256` (eval); `16` (play) | `parkour_teacher_cfg.py:34`, `parkour_student_cfg.py:40`, etc. |
| `env_spacing` | `1.0` | `parkour_teacher_cfg.py:34` |
| `replicate_physics` | (default; not overridden) | InteractiveSceneCfg base |
| `sky_light` | `DomeLightCfg(intensity=750.0, texture=kloofendal_43d_clear_puresky_4k.hdr)` | `default_cfg.py:36-41` |
| Self-collisions | enabled (`enabled_self_collisions=True`) | `default_cfg.py:64` |

### 3a. Sky / lights
```python
# default_cfg.py:35-41
sky_light = AssetBaseCfg(
    prim_path="/World/skyLight",
    spawn=sim_utils.DomeLightCfg(
        intensity=750.0,
        texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
    ),
)
```

---

## 4. Terrain

### 4a. TerrainImporterCfg — `default_cfg.py:43-62`

| Field | Value |
|-------|-------|
| `class_type` | `ParkourTerrainImporter` |
| `prim_path` | `"/World/ground"` |
| `terrain_type` | `"generator"` |
| `terrain_generator` | `None` at base; set to `EXTREME_PARKOUR_TERRAINS_CFG` in `ParkourTeacherSceneCfg.__post_init__` (`parkour_teacher_cfg.py:30`) |
| `max_init_terrain_level` | `2` (eval/play override to `None`) |
| `collision_group` | `-1` |
| `physics_material.static_friction` | `1.0` |
| `physics_material.dynamic_friction` | `1.0` |
| `physics_material.friction_combine_mode` | `"average"` |
| `physics_material.restitution_combine_mode` | `"average"` |
| `visual_material` | `TilesMarbleSpiderWhiteBrickBondHoned.mdl` (project_uvw=True, texture_scale=(0.25, 0.25)) |
| `debug_vis` | `False` |

### 4b. TerrainGeneratorCfg — `parkour_isaaclab/terrains/extreme_parkour/config/parkour.py:4-60`

```python
EXTREME_PARKOUR_TERRAINS_CFG = ParkourTerrainGeneratorCfg(
    size=(16.0, 4.0),
    border_width=20.0,
    num_rows=10,                  # student override → 10; eval override → 5
    num_cols=40,                  # student override → 20; eval override → 5
    horizontal_scale=0.08,        # note: comment says original 0.05 but bumped for IsaacLab issue #2187
    vertical_scale=0.005,
    slope_threshold=1.5,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,              # forced True again in __post_init__ (line 58)
    sub_terrains={ ... },
)
```

Also inherits `num_goals: int = 8` from `ParkourTerrainGeneratorCfg` (`parkour_terrain_generator_cfg.py:23`).

### 4c. Sub-terrains (default proportions in `config/parkour.py`)

| Key | Class | proportion (default) | Key params |
|-----|-------|----------------------|------------|
| `parkour_gap` | `ExtremeParkourGapTerrainCfg` | `0.2` | `x_range=(0.8, 1.5)`, `half_valid_width=(0.6, 1.2)`, `gap_size='0.1 + 0.7*difficulty'`, `gap_depth=(0.2, 1)` (line 16-22) |
| `parkour_hurdle` | `ExtremeParkourHurdleTerrainCfg` | `0.2` | `x_range=(1.2, 2.2)`, `half_valid_width=(0.4, 0.8)`, `hurdle_height_range='0.1+0.1*difficulty, 0.15+0.25*difficulty'` (line 23-29) |
| `parkour_flat` | `ExtremeParkourHurdleTerrainCfg` (apply_flat=True) | `0.2` | same hurdle params but flattened; `hurdle_height_range='0.1+0.1*difficulty, 0.15+0.15*difficulty'` (line 30-37) |
| `parkour_step` | `ExtremeParkourStepTerrainCfg` | `0.2` | `x_range=(0.3, 1.5)`, `half_valid_width=(0.5, 1)`, `step_height='0.1 + 0.35*difficulty'` (line 38-44) |
| `parkour` | `ExtremeParkourTerrainCfg` | `0.2` | `x_range='-0.1, 0.1+0.3*difficulty'`, `y_range='0.2, 0.3+0.1*difficulty'`, `stone_len='0.9 - 0.3*difficulty, 1 - 0.2*difficulty'`, `incline_height='0.25*difficulty'`, `last_incline_height='incline_height + 0.1 - 0.1*difficulty'` (line 45-53), default `pit_depth=(0.2, 1)`, `stone_width=1.0`, `last_stone_len=1.6` |
| `parkour_demo` | `ExtremeParkourDemoTerrainCfg` | `0.0` | demo-only (line 54-57) |

Common rough-terrain base (`ExtremeParkourRoughTerrainCfg`, `extreme_parkour_terrains_cfg.py:6-16`):
- `apply_roughness=True`, `apply_flat=False`, `downsampled_scale=0.075`,
- `noise_range=(0.02, 0.06)`, `noise_step=0.005`,
- `x_range=(0.8,1.5)`, `y_range=(-0.4,0.4)`, `half_valid_width=(0.6,1.2)`,
- `pad_width=0.1`, `pad_height=0.0`.

Sub-terrain base (`ParkourSubTerrainBaseCfg`, `parkour_terrain_generator_cfg.py:9-19`):
- `border_width=0.0`, `horizontal_scale=0.05` (overridden to 0.08), `vertical_scale=0.005`,
- `platform_len=2.5`, `platform_height=0.`, `slope_threshold=1.5`, `edge_width_thresh=0.05`,
- `use_simplified=False` (student forces `True`).

### 4d. EVAL / PLAY terrain overrides
- `_EVAL` (`parkour_teacher_cfg.py:74-87`): `max_init_terrain_level=None`, `num_rows=5`, `num_cols=5`, `random_difficulty=True`, `difficulty_range=(0.0,1.0)`. Sub-terrain proportions to `0.25` for parkour/hurdle/step/gap; `noise_range=(0.02,0.02)`.
- `_PLAY` (`parkour_teacher_cfg.py:90-108`): `difficulty_range=(0.7,1.0)`, all sub-terrains `proportion=0.2` except `parkour_flat=0.0`.

---

## 5. Robot — ArticulationCfg + Actuator

### 5a. Robot composition

```python
# default_cfg.py:33
robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
```

Where `UNITREE_GO2_CFG` is the **IsaacLab stock** Go2 cfg (`/home/lgb/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py:140-181`):

| Field | Value | Source |
|-------|-------|--------|
| `spawn.usd_path` | `f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"` | unitree.py:142 |
| `spawn.activate_contact_sensors` | `True` | unitree.py:143 |
| `rigid_props.disable_gravity` | `False` | unitree.py:145 |
| `rigid_props.retain_accelerations` | `False` | unitree.py:146 |
| `rigid_props.linear_damping` | `0.0` | unitree.py:147 |
| `rigid_props.angular_damping` | `0.0` | unitree.py:148 |
| `rigid_props.max_linear_velocity` | `1000.0` | unitree.py:149 |
| `rigid_props.max_angular_velocity` | `1000.0` | unitree.py:150 |
| `rigid_props.max_depenetration_velocity` | `1.0` | unitree.py:151 |
| `articulation_props.enabled_self_collisions` | `True` (stock) — re-asserted in `default_cfg.py:64` | unitree.py:154 + default_cfg.py:64 |
| `articulation_props.solver_position_iteration_count` | `4` | unitree.py:154 |
| `articulation_props.solver_velocity_iteration_count` | `0` | unitree.py:154 |
| `init_state.pos` | `(0.0, 0.0, 0.27)` | unitree.py:158 |
| `init_state.joint_pos` | `{".*L_hip_joint":0.1, ".*R_hip_joint":-0.1, "F[L,R]_thigh_joint":0.8, "R[L,R]_thigh_joint":1.0, ".*_calf_joint":-1.5}` | unitree.py:159-165 |
| `init_state.joint_vel` | `{".*": 0.0}` | unitree.py:166 |
| `soft_joint_pos_limit_factor` | `0.9` | unitree.py:168 |

### 5b. Actuator override — `default_cfg.py:65-85` (ParkourDCMotorCfg, replaces stock `base_legs`)

```python
self.robot.actuators['base_legs'] = ParkourDCMotorCfg(
    joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
    effort_limit={'.*_hip_joint':35.0, '.*_thigh_joint':40.0, '.*_calf_joint':40.0},
    saturation_effort={'.*_hip_joint':35.0, '.*_thigh_joint':45.0, '.*_calf_joint':45.0},
    velocity_limit={'.*_hip_joint':52.4, '.*_thigh_joint':30.1, '.*_calf_joint':30.1},
    stiffness=40.0,
    damping=1.0,
    friction=0.0,
)
```

`ParkourDCMotorCfg` (`parkour_actuator_cfg.py:13-19`) extends `DCMotorCfg`; adds `saturation_effort: dict[str, float] | None = None` and binds `class_type = parkour_actuator_pd.ParkourDCMotor`. The stock `armature=0.01` from `UNITREE_GO2_CFG` is **dropped** because the whole `base_legs` actuator entry is replaced (no merge — full reassignment in line 65).

> Stock Go2 actuator (for reference, NOT the value used here):
> `DCMotorCfg(effort_limit=23.5, saturation_effort=23.5, velocity_limit=30.0, stiffness=25.0, damping=0.5, friction=0.0, armature=0.01)` (unitree.py:170-178)

---

## 6. Sensors

### 6a. ContactSensorCfg — `parkour_teacher_cfg.py:22-27`

```python
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",
    history_length=2,
    track_air_time=True,
    debug_vis=False,
    force_threshold=1.0,
)
```
`update_period` set to `sim.dt * decimation = 0.005 * 4 = 0.02 s` in `__post_init__` (`parkour_teacher_cfg.py:57`).
No body name filter (uses `Robot/.*` so contact data is computed for all bodies; reward/obs terms select bodies via `SceneEntityCfg`).

### 6b. RayCasterCfg (height scanner) — `parkour_teacher_cfg.py:14-21`

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
    attach_yaw_only=True,
    pattern_cfg=patterns.GridPatternCfg(resolution=0.15, size=[1.65, 1.5]),
    debug_vis=False,
    mesh_prim_paths=["/World/ground"],
)
```
- `update_period = 0.02 s`.
- Grid size 1.65 × 1.5 m at 0.15 m resolution → 12 × 11 = **132** rays (matches `measured_heights[:, 132]` in observations.py:42).
- `attach_yaw_only=True`: scan rotates with robot yaw only.
- Height computed in obs: `clip(pos_w[:,2] - ray_hits_w[...,2] - 0.3, -1, 1)` (`observations.py:139`).

### 6c. Depth camera (Student only) — `default_cfg.py:88-106` (`CAMERA_CFG`)

```python
CAMERA_CFG = RayCasterCameraCfg(
    prim_path='{ENV_REGEX_NS}/Robot/base',
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08),
        rot=quat_from_euler_xyz_tuple(deg→rad([180, 70, -90])),
        convention="ros",
    ),
    depth_clipping_behavior='max',
    pattern_cfg=PinholeCameraPatternCfg(
        focal_length=11.041, horizontal_aperture=20.955, vertical_aperture=12.240,
        height=60, width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance=2.0,
)
```
- Resized to **(58, 87)** in obs (`parkour_mdp_cfg.py:86`), `buffer_len=2`, sampled every 5 steps (`observations.py:174`).
- `update_period = 0.02 s` (student `__post_init__`).

### 6d. USD camera asset (`CAMERA_USD_CFG`, only used in `_EVAL`) — `default_cfg.py:108-115`
Cosmetic `d435.usd` prim at `{ENV_REGEX_NS}/Robot/base/d435`, not a sensor.

---

## 7. Decimation / Episode

| Field | Value | Source |
|-------|-------|--------|
| `decimation` | `4` (policy dt = 0.005 * 4 = 0.02 s, 50 Hz) | `parkour_teacher_cfg.py:48`, `parkour_student_cfg.py:54` |
| `episode_length_s` | `20.0` (teacher train); `60.0` (PLAY); `20.0` (EVAL) | `parkour_teacher_cfg.py:49`, `:71`, `:96` |
| episode steps (train) | `ceil(20.0 / 0.02) = 1000` | derived |

---

## 8. ActionsCfg — `parkour_mdp_cfg.py:339-350`

```python
joint_pos = DelayedJointPositionActionCfg(
    asset_name="robot",
    joint_names=[".*"],
    scale=0.25,
    use_default_offset=True,
    action_delay_steps=[1, 1],
    delay_update_global_steps=24 * 8000,   # = 192_000
    history_length=8,
    use_delay=True,                        # teacher overrides → False (parkour_teacher_cfg.py:59-60)
    clip={'.*': (-4.8, 4.8)},
)
```
- Teacher `__post_init__`: `use_delay=False`, `history_length=1`. Student keeps `use_delay=True`, `history_length=8` (`parkour_student_cfg.py:66-67`).
- Extends `JointPositionActionCfg` (IsaacLab core). All 12 leg joints (`.*`).

---

## 9. ObservationsCfg

### 9a. TeacherObservationsCfg — `parkour_mdp_cfg.py:43-61`

Single PolicyCfg term:

```python
extreme_parkour_observations = ObsTerm(
    func=observations.ExtremeParkourObservations,
    params={
        "asset_cfg": SceneEntityCfg("robot"),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
        "parkour_name": 'base_parkour',
        "history_length": 10,
    },
    clip=(-100, 100),
)
```

Internally this single term produces a concatenated vector (`observations.py:68-90`):

| Slice | Content | Notes |
|-------|---------|-------|
| 0-2  | `root_ang_vel_b * 0.25` | base angular velocity (scaled) |
| 3-4  | `imu_obs` (roll, pitch, wrapped) | |
| 5    | `0 * delta_yaw` | placeholder zero |
| 6    | `delta_yaw` | target_yaw - yaw (cleared to 0 in returned `obs_buf[:, 6:8]=0` AFTER copy into observations) |
| 7    | `delta_next_yaw` | next goal yaw delta |
| 8-9  | `0 * commands[:, 0:2]` | zero block |
| 10   | `commands[:, 0:1]` | lin_vel_x command |
| 11   | `env_idx_tensor` | bool (≠ parkour_flat) |
| 12   | `invert_env_idx_tensor` | bool (== parkour_flat) |
| 13-24| `joint_pos - default_joint_pos` (12 joints) | |
| 25-36| `joint_vel * 0.05` (12 joints) | |
| 37-48| previous action (12 joints, from action history) | |
| 49-52| `_get_contact_fill()` for `.*_foot` (4 feet, `(contact-0.5)`) | from `contact_forces.net_forces_w_history[:,0,..]`, threshold 2 N |

`obs_buf` size = `3 + 2 + 3 + 4 + 36 + 5 = 53` (observations.py:39).
Then concat:
- `measured_heights` (132) — from height_scanner
- `priv_explicit` (9) — `base_lin_vel * 2.0` then `0*base_lin_vel` twice (observations.py:115-118)
- `priv_latent` (29) — body mass [1] + body com [3] + friction [1] + (joint_stiffness/default - 1) [12] + (joint_damping/default - 1) [12]
- history buffer flat `(53 * history_length=10) = 530`

**Total per-env obs vector = 53 + 132 + 9 + 29 + 530 = 753** (no explicit `noise` term; clip=(-100, 100)).

### 9b. StudentObservationsCfg — `parkour_mdp_cfg.py:63-103`

- `policy` group: same ExtremeParkourObservations term (history_length=10).
- `depth_camera` group (`DepthCameraPolicyCfg`): `image_features` term over `depth_camera` sensor, `resize=(58, 87)`, `buffer_len=2`, `debug_vis=True`.
- `delta_yaw_ok` group (`DeltaYawOkPolicyCfg`): `obervation_delta_yaw_ok`, threshold `0.6`.

No noise / mod terms attached to any group.

---

## 10. EventsCfg / Domain Randomization — `parkour_mdp_cfg.py:258-336`

| Event | Mode | Trigger params | Body / asset | Range |
|-------|------|----------------|--------------|-------|
| `reset_root_state` | `reset` | — | robot | calls `events.reset_root_state` with `offset=3.0`; positions = `default_root_state[:,0:3] + origin - (terrain_size_y + 3.0, 0, 0)` (events.py:47-61) |
| `reset_robot_joints` | `reset` | — | robot | `reset_joints_by_scale(position_range=(0.95, 1.05), velocity_range=(0.0, 0.0))` |
| `physics_material` | `startup` | — | `robot, body=".*"` | `friction_range=(0.6, 2.0)`, `num_buckets=64` |
| (commented out) `randomize_actuator_gains` | — | "we don't use this event" — disabled per comment (parkour_mdp_cfg.py:285-295) | — | — |
| `randomize_rigid_body_mass` | `startup` | — | `body=base` | `(-1.0, 3.0)`, `operation="add"` |
| `randomize_rigid_body_com` | `startup` | — | `body=base` | `com_range={x:(-0.02,0.02), y:(-0.02,0.02), z:(-0.02,0.02)}` |
| `random_camera_position` | `startup` | — | `depth_camera` | `rot_noise_range={pitch:(-5, 5)}` deg, convention='ros' (teacher disables this in `__post_init__`:61) |
| `push_by_setting_velocity` | `interval` | `interval_range_s=(8., 8.)`, `is_global_time=True` | robot | `velocity_range={x:(-0.5, 0.5), y:(-0.5, 0.5)}` |
| `base_external_force_torque` | `reset` | — | `body=base` | `force_range=(0.0, 0.0)`, `torque_range=(-0.0, 0.0)` (effectively no-op as configured) |

EVAL overrides (`parkour_teacher_cfg.py:80-82`):
- `randomize_rigid_body_com = None`
- `randomize_rigid_body_mass = None`
- `push_by_setting_velocity.interval_range_s = (6., 6.)`

PLAY override (`parkour_teacher_cfg.py:102`): `push_by_setting_velocity = None`.

---

## 11. CommandsCfg — `parkour_mdp_cfg.py:18-34`

```python
base_velocity = ParkourCommandCfg(
    asset_name="robot",
    resampling_time_range=(6.0, 6.0),
    heading_control_stiffness=0.8,
    ranges=ParkourCommandCfg.Ranges(
        lin_vel_x=(0.3, 0.8),
        heading=(-1.6, 1.6),
    ),
    clips=ParkourCommandCfg.Clips(
        lin_vel_clip=0.2,
        ang_vel_clip=0.4,
    ),
)
```

`ParkourCommandCfg` defaults (`parkour_command_cfg.py:10-38`):
- `heading_control_stiffness: float = 1.0` (overridden to 0.8 above).
- `small_commands_to_zero: bool = True`.
- visualizer markers: `BLUE_ARROW_X_MARKER_CFG` / `GREEN_ARROW_X_MARKER_CFG`, scale (0.5,0.5,0.5).

EVAL: `resampling_time_range = (60., 60.)`, `debug_vis=True` (parkour_teacher_cfg.py:73, 83).
No `rel_standing_envs` field (not part of this cfg).

### 11a. ParkourEventsCfg (custom "parkour" manager term) — `parkour_mdp_cfg.py:36-41`, `parkour_events_cfg.py:48-78`

```python
base_parkour = parkours.ParkourEventsCfg(
    asset_name='robot',
    # defaults from parkour_events_cfg.py:
    #   num_future_goal_obs = 2
    #   arrow_num = 8
    #   reach_goal_delay = 0.1
    #   next_goal_threshold = 0.2
    #   debug_vis = False  (EVAL sets True, parkour_teacher_cfg.py:72)
)
```
This term drives `target_yaw`, `next_target_yaw`, current/next goal positions used by reward + obs.

---

## 12. Rewards

### 12a. TeacherRewardsCfg — `parkour_mdp_cfg.py:117-244` (12 terms)

| Term | weight | bodies / params |
|------|--------|-----------------|
| `reward_collision` | `-10.0` | `contact_forces, body=["base", ".*_calf", ".*_thigh"]` |
| `reward_feet_edge` | `-1.0` | `asset_cfg.body=["FL_foot","FR_foot","RL_foot","RR_foot"]`, `sensor=contact_forces .*_foot`, `parkour_name='base_parkour'` |
| `reward_torques` | `-0.00001` | robot |
| `reward_dof_error` | `-0.04` | robot |
| `reward_hip_pos` | `-0.5` | robot, `joint=.*_hip_joint` |
| `reward_ang_vel_xy` | `-0.05` | robot |
| `reward_action_rate` | `-0.1` | robot |
| `reward_dof_acc` | `-2.5e-7` | robot |
| `reward_lin_vel_z` | `-1.0` | robot, `parkour_name='base_parkour'` |
| `reward_orientation` | `-1.0` | robot, `parkour_name='base_parkour'` |
| `reward_feet_stumble` | `-1.0` | `contact_forces, body=.*_foot` |
| `reward_tracking_goal_vel` | `+1.5` | robot, `parkour_name='base_parkour'` |
| `reward_tracking_yaw` | `+0.5` | robot, `parkour_name='base_parkour'` |
| `reward_delta_torques` | `-1.0e-7` | robot |

### 12b. StudentRewardsCfg — `parkour_mdp_cfg.py:106-114`
Only one term:
```python
reward_collision (weight=-0.0, contact_forces body=["base", ".*_calf", ".*_thigh"])
```
(EVAL inherits TeacherRewardsCfg.)

### 12c. TerminationsCfg — `parkour_mdp_cfg.py:246-256`
```python
total_terminates = DoneTerm(
    func=terminations.terminate_episode,
    time_out=True,
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```
Single termination term (timeout-flagged; logic inside `terminations.terminate_episode`, not dumped here as task scope ends at cfg).

---

## 13. Viewer / Debug

```python
# default_cfg.py:116-120
VIEWER = ViewerCfg(
    eye=(-0., 2.6, 1.6),
    asset_name="robot",
    origin_type='asset_root',
)
```
Set in EVAL and PLAY only (`parkour_teacher_cfg.py:65, 91`; student EVAL `parkour_student_cfg.py:73`). Train cfg uses default `ViewerCfg()`.

Custom `ui_window_class_type = ParkourManagerBasedRLEnvWindow` (`parkour_manager_based_rl_env_cfg.py:15`).

Debug flags:
- `parkour_teacher_cfg.py:19`, `:25`, `:62`: `debug_vis=False` on sensors (train).
- EVAL: `parkours.base_parkour.debug_vis=True`, `commands.base_velocity.debug_vis=True` (parkour_teacher_cfg.py:72-73).

---

## 14. Cross-cfg constants summary (single-glance reference)

| Topic | Train | EVAL | PLAY |
|-------|-------|------|------|
| Teacher `num_envs` | 6144 | 256 | 16 |
| Student `num_envs` | 192 | 256 | 16 |
| Teacher `episode_length_s` | 20.0 | 20.0 | 60.0 |
| Teacher `decimation` / `sim.dt` | 4 / 0.005 | (same) | (same) |
| Teacher `use_delay` | False | False | False |
| Student `use_delay` | True | True | True |
| Terrain grid (rows × cols) | 10 × 40 (teacher), 10 × 20 (student) | 5 × 5 | 5 × 5 |
| `difficulty_range` | (0,1) | (0,1) random | (0.7, 1.0) random |
| `max_init_terrain_level` | 2 | None | None |
| Push interval | (8., 8.) | (6., 6.) | None |
| Resampling time | (6., 6.) s | (60., 60.) s | (60., 60.) s |

---

## 15. Notes / open items (factual only, no judgments)

- `parkour_manager_based_env_cfg.py:7` — `ParkourManagerBasedEnvCfg` only adds `parkours: object = MISSING` over IsaacLab's `ManagerBasedEnvCfg`. The actual env class is `ParkourManagerBasedRLEnv` (`parkour_isaaclab/envs/parkour_manager_based_rl_env.py`, not read here).
- The `randomize_actuator_gains` event is **defined but commented out** with note "we don't use this event, If you use this, you will get a bad result" (`parkour_mdp_cfg.py:285-295`).
- `base_external_force_torque` is registered but range is `(0, 0)` — effectively no-op as configured.
- `ParkourDCMotorCfg` replaces the `base_legs` actuator entry completely (`default_cfg.py:65`) — stock `armature=0.01` is dropped (no merge). `class_type` resolves to `ParkourDCMotor` (logic in `parkour_actuator_pd.py`, not part of cfg dump).
- Height-scan tensor is sized `(num_envs, 132)` matching the grid (12 × 11), with `0.3 m` height offset subtracted in obs (`observations.py:139`).
- `EXTREME_PARKOUR_TERRAINS_CFG.horizontal_scale=0.08` — comment in source: *"original scale is 0.05, But Computing issue in IsaacLab see https://github.com/isaac-sim/IsaacLab/issues/2187"*.
- Student `__post_init__` (`parkour_student_cfg.py:21-34`) forces `use_simplified=True` for all sub-terrains and `horizontal_scale=0.1` (overrides the 0.08 in EXTREME_PARKOUR_TERRAINS_CFG).
- No explicit `noise` term on any observation (only `clip=(-100, 100)` on the teacher obs term).
- IsaacLab `SimulationCfg` defaults inherited (not overridden): `gravity=(0,0,-9.81)`, `device="cuda:0"`, `use_fabric=True`, `physx.solver_type=1` (TGS), `physx.min/max_position_iteration_count=1/255`, etc. (`simulation_cfg.py:37, 355, 366, 371, 392`).

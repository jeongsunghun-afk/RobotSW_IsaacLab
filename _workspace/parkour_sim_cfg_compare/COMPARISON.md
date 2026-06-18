# Parkour Simulation Config 비교 보고서

> **A**: `/home/lgb/Isaaclab_Parkour/` (Manager-based RL, Extreme-Parkour 포팅, Teacher/Student 분리)
> **B**: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` (DirectRL, 단일 cfg)
> **자료**: `external_sim_cfg.md` (26 KB) + `isaaclab_sim_cfg.md` (20 KB) — 본 파일은 두 측을 1:1 매칭해 차이를 정리.
> **읽기 전용. 값 판단/권장 없음.**

---

## 0. 가장 큰 구조적 차이 (TL;DR)

| 측면 | A (외부) | B (IsaacLab parkour) |
|---|---|---|
| Env baseclass | `ManagerBasedRLEnv` 계열 → `ParkourManagerBasedRLEnv` | `DirectRLEnv` → `Go2ParkourEnv` |
| Cfg 분산 | **여러 파일 + 상속 체인** (`parkour_isaaclab/` framework + `parkour_tasks/extreme_parkour_task/config/go2/`) | **단일 파일** `parkour_env_cfg.py` + sibling `parkour_terrains.py` |
| Workflow variant | **Teacher / Student / EVAL / PLAY** 4가지 cfg 클래스 | 단일 cfg (PLAY는 `play.py` runtime override) |
| Robot cfg 결합 | `default_cfg.py`에서 `UNITREE_GO2_CFG.replace + base_legs 통째 교체` | `__post_init__`에서 `replace + DCMotorCfg`로 base_legs 재구성, `_actuator_mode` 분기 |
| Actuator 강도 (현재 active) | Kp=40, Kd=1.0, effort={hip:35, thigh:40, calf:40}, vel={52.4, 30.1, 30.1}, armature dropped | **`_actuator_mode=2` 활성** → Kp=25, Kd=0.5, effort=23.5/all, vel=30/all, armature=0.01 (stock Go2 값) |
| Sub-terrain 패턴 | 6종(gap/hurdle/flat/step/parkour/demo), 각 0.2 (demo=0.0) | 11종 정의, 5종 활성(flat 0.1, hurdle 0.2, step 0.2, gap 0.3, stair 0.2), 6종 0.0 |
| Domain randomization | **8 event** (friction/mass/com/push/joints/root/camera/force) | **1 event** (foot physics material만, startup) |
| Command space | x lin + heading (yaw control via stiffness 0.8), resample 6s | x lin only (0.3~1.0), no yaw/lateral, resample cfg 없음 |
| Observation 결합 | 단일 `ExtremeParkourObservations` term (53+132+9+29+530=753) | 다중 obs group (policy 46 + scan 187 + priv 4 + history 460 = 697 dict) |

---

## 1. SimulationCfg 비교

| 필드 | A | B | 일치? |
|---|---|---|---|
| `sim.dt` | `0.005` s (200 Hz) | `0.005` s (200 Hz) | ✅ |
| `sim.render_interval` | `self.decimation` = 4 | `4` | ✅ |
| `sim.gravity` | default `(0,0,-9.81)` | default `(0,0,-9.81)` | ✅ |
| `sim.device` | default `cuda:0` | (override 없음, 동일) | ✅ |
| `physics_material.friction_combine_mode` | (terrain의 material 사용, `"average"` 명시: `default_cfg.py:51-56`) | `"average"` | ✅ |
| `physics_material.restitution_combine_mode` | `"average"` | `"average"` | ✅ |
| `physics_material.static_friction` | `1.0` | `1.0` | ✅ |
| `physics_material.dynamic_friction` | `1.0` | `1.0` | ✅ |
| `physics_material.restitution` | (default 0) | `0.0` | ✅ |
| `physx.gpu_max_rigid_patch_count` | `10 * 2**18 = 2,621,440` | `2**23 = 8,388,608` | ❌ B가 약 3.2배 큼 |
| `physx.gpu_found_lost_pairs_capacity` | default | `2**23 = 8,388,608` | ❌ B만 override |
| `physx.gpu_total_aggregate_pairs_capacity` | default | `2**23 = 8,388,608` | ❌ B만 override |
| `physx.solver_type` | default 1 (TGS) | default 1 (TGS) | ✅ |
| 그 외 PhysX | 모두 default | 모두 default | ✅ |

**요점**: 물리 dt/gravity/material/solver는 동일. 차이는 **GPU 버퍼 크기**만 — B는 3개 GPU buffer를 `2**23`로 키워둠. A는 patch_count 하나만 약 절반 크기로 설정.

---

## 2. Scene 비교

| 필드 | A (Teacher train) | B | 일치? |
|---|---|---|---|
| `num_envs` | `6144` (teacher) / 192 (student) / 256 (eval) / 16 (play) | `4096` | ❌ |
| `env_spacing` | `1.0` m | `4.0` m | ❌ B가 4배 |
| `replicate_physics` | default | `True` | ≈ (둘 다 True) |
| Sky / 조명 | `DomeLightCfg(intensity=750, kloofendal_43d_clear_puresky_4k.hdr)` 명시 (`default_cfg.py:36-41`) | scene cfg에 없음 — env runtime에서 `parkour_env.py:349`로 생성 (cfg화 안됨) | ❌ |
| Ground | TerrainImporter로 제공 | TerrainImporter로 제공 | ✅ |
| Self-collisions | 활성 (`default_cfg.py:64`, stock Go2도 True) | 활성 (stock Go2 True 그대로) | ✅ |

---

## 3. Terrain 비교

### 3.1 TerrainImporterCfg

| 필드 | A | B | 일치? |
|---|---|---|---|
| `prim_path` | `/World/ground` | `/World/ground` | ✅ |
| `terrain_type` | `"generator"` | `"generator"` | ✅ |
| `class_type` | **`ParkourTerrainImporter` (커스텀)** | `TerrainImporter` (default) | ❌ |
| `max_init_terrain_level` | `2` (train) / None (eval/play) | `3` | ❌ |
| `collision_group` | `-1` | `-1` | ✅ |
| `physics_material` (static/dynamic/restitution) | 1.0 / 1.0 / 0 | 1.0 / 1.0 / 0 | ✅ |
| `visual_material` | `TilesMarbleSpiderWhiteBrickBondHoned.mdl`, texture_scale (0.25, 0.25) | `Shingles_01.mdl` | ❌ 단순 시각 차이 |
| `debug_vis` | False (eval True) | False | ✅ |

### 3.2 TerrainGeneratorCfg

| 필드 | A (`EXTREME_PARKOUR_TERRAINS_CFG`) | B (`PARKOUR_TERRAINS_CFG`) | 일치? |
|---|---|---|---|
| `size` (per tile) | `(16.0, 4.0)` m | `(20.0, 4.0)` m | ❌ B의 길이가 4 m 더 김 |
| `border_width` | `20.0` m | `20.0` m | ✅ |
| `num_rows` | `10` (eval/play 5) | `11` | ≈ (1 row 차) |
| `num_cols` | `40` (eval/play 5) | `40` | ✅ |
| `horizontal_scale` | `0.08` m/px ("원래 0.05이나 IsaacLab #2187 회피") | `0.05` m/px | ❌ A가 더 굵음 |
| `vertical_scale` | `0.005` m | `0.005` m | ✅ |
| `slope_threshold` | `1.5` | `0.75` | ❌ B가 더 엄격(절반) |
| `difficulty_range` | `(0.0, 1.0)` | (cfg 미노출, generator default) | — |
| `use_cache` | `False` | `False` | ✅ |
| `curriculum` | `True` (post_init에서 강제) | `True` | ✅ |
| `num_goals` (generator-level) | `8` (cfg inherited) | `8` (개별 sub-terrain에 각각 명시) | ✅ |

### 3.3 활성 sub-terrains

| A (proportion) | B (proportion) | 매칭 가능성 |
|---|---|---|
| `parkour_gap` 0.2 | `parkour_gap` 0.3 | 1:1 (B가 더 높은 빈도) |
| `parkour_hurdle` 0.2 | `parkour_hurdle` 0.2 | 1:1 |
| `parkour_flat` 0.2 (hurdle class에 apply_flat=True) | `parkour_flat` 0.1 (hurdle class에 flat=True, hurdle_height=0~0) | 1:1, B가 절반 빈도 |
| `parkour_step` 0.2 | `parkour_step` 0.2 | 1:1 |
| `parkour` 0.2 (stepping-stone 다리) | — | A 전용 (B에 동등 없음) |
| `parkour_demo` 0.0 | — | A 전용 (demo only) |
| — | `parkour_stair` 0.2 | B 전용 |
| — | `parkour_stepping_stones` 0.0, `parkour_balance_beam` 0.0, `parkour_crawl` 0.0, `parkour_slope` 0.0, `parkour_zigzag_hurdles` 0.0, `parkour_rough_blocks` 0.0 | B 전용 (모두 비활성) |

**활성 종류 수**: A=5종, B=5종. 절반은 매칭, 절반은 다름. **B는 `parkour_stair` 보유, A는 stepping-stone 형태의 `parkour` 보유**가 가장 큰 차이.

### 3.4 Sub-terrain 파라미터 (매칭되는 것만 비교)

| Term | A 값 | B 값 |
|---|---|---|
| **hurdle** `hurdle_height_range` | difficulty-dependent `(0.1+0.1·d, 0.15+0.25·d)` | static `(0.05, 0.30)` |
| **hurdle** `x_spacing` | `x_range=(1.2, 2.2)` (생성 간격) | `x_spacing_range=(1.0, 1.5)` |
| **hurdle** `num_hurdles` | (사양에 명시 없음; goal=8 기반) | `8` |
| **step** height | difficulty-dependent `(0.1+0.35·d)` (max ~0.45) | static `(0.10, 0.45)` |
| **step** `x_length` | `x_range=(0.3, 1.5)` | `x_length_range=(0.4, 0.8)` |
| **gap** `gap_length` | difficulty-dependent `(0.1+0.7·d, ...)` (max ~0.8) | static `(0.05, 0.5)` |
| **gap** valid width / platform | `half_valid_width=(0.6, 1.2)` | `platform_length_range=(1.2, 1.6)` |
| **flat** (hurdle-based) | `hurdle_height=(0.1+0.1·d, 0.15+0.15·d)` (작은 hurdle 흔적 남음) | `hurdle_height_range=(0.0, 0.0)` (완전 평지) |
| Roughness | `apply_roughness=True, noise_range=(0.02, 0.06), noise_step=0.005` (모든 sub-terrain 공통) | sub-terrain 클래스 자체가 다른 구조 — roughness 파라미터 없음 (flat patch만) |
| `pad_width`/`pad_height` | `0.1`/`0.0` (모든 sub-terrain 공통) | 없음 |
| FlatPatchSampling | 없음 | `init_positions = FlatPatchSamplingCfg(num_patches=2, patch_radius=0.5, max_height_diff=0.05)` (모든 sub-terrain에 동일하게 첨부) |

**핵심 차이**:
- A의 sub-terrain은 **difficulty-scaling formula**로 난이도가 자라남 + roughness가 모든 terrain에 항상 깔림.
- B의 sub-terrain은 **고정 range에서 random sampling** + flat patch가 항상 깔림 (스폰 가능 지점 사전 계산).

---

## 4. Robot + Actuator 비교 (가장 큰 actuator 차이 주의)

### 4.1 Articulation 본체 (USD + rigid/articulation props)

| 필드 | A | B | 일치? |
|---|---|---|---|
| `usd_path` | `Robots/Unitree/Go2/go2.usd` | 동일 | ✅ |
| `activate_contact_sensors` | True | True | ✅ |
| `rigid_props` (모든 sub-field) | stock Go2 그대로 | stock Go2 그대로 | ✅ |
| `articulation_props.enabled_self_collisions` | True | True | ✅ |
| `articulation_props.solver_position_iteration_count` | 4 | 4 | ✅ |
| `articulation_props.solver_velocity_iteration_count` | 0 | 0 | ✅ |
| `init_state.pos` | `(0, 0, 0.27)` | `(0, 0, 0.27)` | ✅ |
| `init_state.joint_pos` (hip ±0.1, thigh F/R 0.8/1.0, calf -1.5) | stock | stock | ✅ |
| `init_state.joint_vel` | `{".*": 0.0}` | 동일 | ✅ |
| `soft_joint_pos_limit_factor` | `0.9` | `0.9` | ✅ |

### 4.2 Actuator (base_legs) — **여기가 핵심 차이**

A는 `default_cfg.py:65`에서 `base_legs` actuator를 통째로 `ParkourDCMotorCfg`로 교체. B는 `__post_init__`에서 `_actuator_mode` 분기를 통해 `DCMotorCfg`로 재구성. **현재 활성 모드는 mode=2**.

| 필드 | A (active) | B (`_actuator_mode=2`, active) | B (`_actuator_mode=1`, inactive) | Stock Go2 (reference) |
|---|---|---|---|---|
| Class | `ParkourDCMotorCfg` (DCMotor + saturation override 가능) | `DCMotorCfg` | `DCMotorCfg` | `DCMotorCfg` |
| `stiffness` (Kp) | **40.0** | **25.0** | 40.0 | 25.0 |
| `damping` (Kd) | **1.0** | **0.5** | 1.0 | 0.5 |
| `effort_limit` (hip / thigh / calf) | **35 / 40 / 40** | **23.5 / 23.5 / 23.5** | 35 / 40 / 40 | 23.5 (scalar) |
| `saturation_effort` (hip / thigh / calf) | **35 / 45 / 45** | **23.5** (scalar) | 35.0 | 23.5 (scalar) |
| `velocity_limit` (hip / thigh / calf) | **52.4 / 30.1 / 30.1** | **30.0 / 30.0 / 30.0** | 52.4 / 30.1 / 30.1 | 30.0 (scalar) |
| `armature` | **dropped** (재할당으로 stock 0.01 소실) | **0.01** | 0.01 | 0.01 |
| `friction` | 0.0 | 0.0 | 0.0 | 0.0 |

**해석 (사실 기록만)**:
- A의 active actuator는 **더 강한** envelope (Kp 40, effort 35~40, velocity 30~52, joint-specific) + armature 없음.
- B의 active actuator(mode 2)는 **stock Go2 수치와 동일** (Kp 25, effort 23.5 scalar, velocity 30 scalar, armature 0.01 보존).
- B에는 mode 1 분기가 **존재하지만 비활성** — 그 값이 A와 거의 동일.
- A는 hip/thigh/calf 관절별 다른 effort/velocity (dict 형태). B mode 2는 모든 관절 scalar 동일.

---

## 5. Sensors 비교

### 5.1 ContactSensor

| 필드 | A | B | 일치? |
|---|---|---|---|
| `prim_path` | `{ENV_REGEX_NS}/Robot/.*` (모든 body) | `/World/envs/env_.*/Robot/.*` (동일 의미) | ✅ |
| `history_length` | `2` | `3` | ❌ |
| `update_period` | `0.02` s (post_init: `sim.dt * decimation`) | `0.005` s (cfg 명시) | ❌ A=20ms, B=5ms |
| `track_air_time` | `True` | `True` | ✅ |
| `force_threshold` | `1.0` N | (cfg 미명시 → default) | — |
| `debug_vis` | False | (default) | ✅ |

**Reward 영향 (참고용 — reward 비교 보고서와 cross-check)**: `feet_stumble`, `feet_edge`, `collision`이 history index를 사용. B는 history 3-step 보유 → reward에서 `max over history` 가능. A는 history 2-step만 보유 → reward는 idx 0 / idx -1 두 슬라이스만 사용.

### 5.2 RayCaster / Height scanner

| 필드 | A | B | 일치? |
|---|---|---|---|
| `prim_path` | `{ENV_REGEX_NS}/Robot/base` | `/World/envs/env_.*/Robot/base` (동일) | ✅ |
| `offset.pos` | `(0.375, 0.0, 20.0)` | `(0.375, 0.0, 20.0)` | ✅ |
| `attach_yaw_only` / `ray_alignment` | `attach_yaw_only=True` | `ray_alignment="yaw"` (의미 동일) | ✅ |
| Pattern type | `GridPatternCfg` | `GridPatternCfg` | ✅ |
| Pattern `resolution` | `0.15` m | `0.10` m | ❌ |
| Pattern `size` | `[1.65, 1.5]` m | `[1.6, 1.0]` m | ❌ |
| 결과 ray 수 | `12 × 11 = 132` rays | `17 × 11 = 187` rays | ❌ B가 더 촘촘 + ray 55개 많음 |
| `update_period` | `0.02` s | default (0.0 → 매 step) | ❌ |
| `mesh_prim_paths` | `["/World/ground"]` | `["/World/ground"]` | ✅ |
| `debug_vis` | False (eval True 가능) | False | ✅ |
| Height 후처리 | `clip(pos_w[:,2] − ray_hits_w[:,2] − 0.3, -1, 1)` (obs단) | (obs단에서 처리; cfg에는 0.3 offset 미명시) | ≈ 동일 의도 |

### 5.3 Camera

| 필드 | A | B |
|---|---|---|
| Depth camera | `RayCasterCameraCfg(d435 spec, 60×106 → resized 58×87, buffer_len=2, every 5 steps)` — **student 전용** | **없음** |
| USD camera asset | EVAL만 `d435.usd` 첨부 (장식) | 없음 |

---

## 6. Decimation / Episode

| 필드 | A | B | 일치? |
|---|---|---|---|
| `decimation` | `4` | `4` | ✅ |
| Policy dt | `0.02` s (50 Hz) | `0.02` s (50 Hz) | ✅ |
| `episode_length_s` | `20.0` (teacher train), `60.0` (PLAY), `20.0` (EVAL) | `20.0` | ✅ (train 동일) |
| Steps per episode | 1000 | 1000 | ✅ |

---

## 7. Actions 비교

| 필드 | A (teacher) | B | 일치? |
|---|---|---|---|
| Action term type | `DelayedJointPositionActionCfg` (extends `JointPositionActionCfg`) | direct env `_apply_action`에서 `joint_pos_target = default + action * scale` 적용 | ≈ |
| `scale` | `0.25` | `action_scale = 0.25` | ✅ |
| `clip` | `(-4.8, 4.8)` per joint | `clip_actions = 10.0` (전역 abs clip) | ❌ A=±4.8, B=±10 |
| `action_space` | 12 (`.*` joints) | `12` | ✅ |
| `use_delay` | **False (teacher)** / True (student, `history_length=8`, `action_delay_steps=[1,1]`) | **delay 없음 (즉시 적용)** | ✅ (teacher 동일) |
| `use_default_offset` | True | (env에서 default + scaled action 형태로 적용 — 동등) | ✅ |
| Action history (cfg-level) | teacher 1, student 8 | `_actions`/`_previous_actions` 1-step (action_smoothness용은 `_last_processed_actions`, `_last_last_processed_actions`까지) | A teacher와 같은 단순 1-step |

---

## 8. Observations 비교

### 8.1 결합 구조

| 측면 | A | B |
|---|---|---|
| Top-level group | 단일 ObsTerm `extreme_parkour_observations`가 내부적으로 다 합쳐 반환 | 다중 group: `policy`, `scan`, `priv`, `history`, `critic` (runtime dict) |
| Noise term | 없음 (clip만) | env 코드 내 noise injection 여부 코드 검사 필요 (cfg-level noise_scales 미정의) |
| Clip | `(-100, 100)` | (cfg level 명시 안 됨; env 내부에서 처리) |

### 8.2 차원 (per env)

| 구성 요소 | A | B |
|---|---|---|
| Proprio (obs_buf 내부) | **53** (ang_vel 3 + imu 2 + delta_yaw 3 + cmd 4 + flat flag 2 + joint_pos 12 + joint_vel 12 + prev_action 12 + foot_contact 4 − placeholders) | **46** policy group 또는 42 `num_proprio` (cfg 내 두 값 불일치 노트, line 285-289) |
| Height scan | **132** | **187** |
| Priv (explicit) | **9** (= base_lin_vel × 2 패딩 형태) | **4** placeholder (cfg에 4로 표기, 실제 priv는 `num_priv_obs=14`) |
| Priv (latent) | **29** (mass 1 + com 3 + friction 1 + (k/k_def − 1) 12 + (d/d_def − 1) 12) | **14** (lin_vel 3 + ang_vel 3 + foot_friction 8) — A의 priv_explicit/latent와 의미가 다름 |
| History flat | `53 × 10 = 530` | `history_len × num_proprio = 10 × 46 = 460` |
| **합계** | **753** (단일 obs vector) | **697** critic / 46 policy / 187 scan / 4 priv / 460 history (dict 분리) |

### 8.3 Proprio 항목 매핑 (대략)

| 의미 | A 슬라이스 | B 위치 (`parkour_env.py`의 obs 조립) |
|---|---|---|
| Base ang vel (body) | 0-2 (`* 0.25` 스케일) | policy group에 포함 |
| Roll/pitch (IMU) | 3-4 | policy group에 포함 |
| delta_yaw / delta_next_yaw | 5-7 (5는 placeholder zero, 6/7은 yaw deltas, 단 obs return 전 `obs_buf[:,6:8]=0` 처리) | (yaw deltas obs 사용) |
| Command | 8-10 (8-9 zero, 10 = cmd_x) | `_commands[:, :3]` |
| Flat flag | 11-12 (env_idx + invert) | `_env_class` 기반 |
| joint_pos − default | 13-24 | 동일 |
| joint_vel × 0.05 | 25-36 | 동일 (스케일 동일 여부는 env 코드 확인 필요) |
| Previous action | 37-48 | `_previous_actions` |
| Foot contact (4) | 49-52 (`contact − 0.5` 형태) | binary contact for `_feet_ids` |

**핵심 차이**: A는 obs vector 안에 IMU/cmd-zero/flat flag placeholder까지 다 박혀 있는 53-dim 단일 벡터 + history 10 누적 → 단일 730+ dim 입력. B는 group별 분리 (policy/scan/priv/history)로 critic-only 정보가 명시적으로 분리됨.

---

## 9. EventsCfg / Domain Randomization 비교 (가장 큰 차이 중 하나)

### 9.1 활성 event 수

| | A (Teacher train) | B |
|---|---|---|
| 활성 event 개수 | **8** | **1** |

### 9.2 항목별 매트릭스

| Event | A (값) | B |
|---|---|---|
| `physics_material` (friction) | startup, body `.*` (전체), friction `(0.6, 2.0)`, 64 buckets | startup, **foot only** (`.*foot`), static_friction `(0.4, 1.5)`, dynamic_friction `(0.3, 1.2)`, restitution `(0, 0)`, 64 buckets |
| `reset_root_state` | reset 모드, `events.reset_root_state(offset=3.0)` 커스텀 | env `_reset_idx` 내부 처리 (cfg 노출 없음) |
| `reset_robot_joints` | reset 모드, `position_range=(0.95, 1.05)`, `velocity_range=(0,0)` | env `_reset_idx` 내부 처리 (cfg 노출 없음, 코드 확인 필요) |
| `randomize_rigid_body_mass` | startup, body=base, `(-1.0, 3.0)`, `operation="add"` | **없음** (cfg 주석에 `added_mass_range: [0.0, 5.0]` 비활성 명시) |
| `randomize_rigid_body_com` | startup, body=base, com x/y/z `(-0.02, 0.02)` | 없음 |
| `random_camera_position` | startup, depth_camera, pitch `(-5°, 5°)` | (카메라 없음) |
| `push_by_setting_velocity` | interval `(8., 8.)` s, global_time=True, x/y `(-0.5, 0.5)` m/s | 없음 (코드 주석에 `friction_range: [0.6, 2.0]` 미적용) |
| `base_external_force_torque` | reset, `(0,0)` (no-op) | 없음 |
| `randomize_actuator_gains` | 명시적으로 주석 처리 ("we don't use this … bad result") | cfg 주석에 `motor_strength_range: [0.8, 1.2]` 비활성 명시 |

**요점**: **A는 robust-locomotion 표준 randomization 세트를 거의 다 보유 (push, mass, com, friction, joint resample)**. **B는 사실상 foot friction 랜덤화 하나만 활성**, 나머지는 cfg에 주석으로만 존재.

---

## 10. Commands 비교

| 필드 | A | B |
|---|---|---|
| Command 종류 | `ParkourCommandCfg` (linear vel + heading) | dict `command_cfg` (linear vel range, ang vel range) |
| `lin_vel_x` 범위 | `(0.3, 0.8)` m/s | `(0.3, 1.0)` m/s |
| `lin_vel_y` 범위 | 없음 (y 명령 자체 없음) | `(0.0, 0.0)` (명시적으로 0) |
| Heading / yaw 명령 | `heading=(-1.6, 1.6)` rad + `heading_control_stiffness=0.8` (ω_z = stiffness × angle err) | `ang_vel_range=(0.0, 0.0)` (ang vel 명령 자체 없음) |
| Resample period | `(6.0, 6.0)` s (eval `(60, 60)`) | **cfg 미정의** (resample 로직이 env에 있을 경우 코드 검사 필요) |
| Clip | `lin_vel_clip=0.2`, `ang_vel_clip=0.4` | (cfg 없음, 코드 내부 처리) |
| `small_commands_to_zero` | True | (default 동작은 env 코드 확인 필요) |
| Standing-envs / `rel_standing_envs` | 미정의 (이 cfg에 필드 없음) | 미정의 |

**핵심**: A는 **heading-driven yaw 제어** (target yaw에서 stiffness로 yaw rate 생성). B는 **순수 forward velocity 추종**, yaw 명령 자체 없음 (`tracking_yaw`는 parkour goal 방향 추종으로 따로 처리).

---

## 11. Parkour Goal cfg (양쪽 모두 존재)

| 필드 | A | B |
|---|---|---|
| `num_goals` | 8 | 8 |
| `num_future_goal_obs` | 2 | 2 |
| `next_goal_threshold` | 0.2 m | 0.2 m |
| `reach_goal_delay` | 0.1 s | 0.1 s |
| `goal_z` | 미명시 (default) | 0.3 m |
| Goal arrow markers | `arrow_num=8`, BLUE/GREEN | (env 내부 viz는 별도) |
| `termination_height` | 미명시 (별도 cfg 없음) | -0.2 m |
| `termination_grace_steps` | 미명시 | 5 |
| `max_tilt` | 미명시 | 1.5 rad |

→ Goal lookahead와 threshold는 양쪽 동일. termination 관련 파라미터는 B에만 cfg-level로 노출.

---

## 12. Reward 비교 (간략 — 자세한 건 이전 보고서 참조)

| | A (Teacher) | B |
|---|---|---|
| 등록 term 수 | 12 (TeacherRewardsCfg) | 22 (cfg.reward_scales) |
| 활성 term | 12 모두 활성 | 10 (나머지 12는 scale 0) |
| 누적 방식 | RewardManager가 `weight × value` 합 | `scale × step_dt × value` 합 (모든 term에 step_dt=0.02 추가 곱) |

자세한 매칭/수식 비교: `/home/lgb/IsaacLab/_workspace/parkour_reward_compare/COMPARISON.md` 참조.

---

## 13. Viewer / Debug

| 필드 | A | B |
|---|---|---|
| Viewer | train: default `ViewerCfg()`; EVAL/PLAY: `eye=(0., 2.6, 1.6)`, `origin_type='asset_root'`, asset_name=robot | `eye=(0, -2.5, 0.8)`, `lookat=(0, 0, 0.3)`, `origin_type='world'`, env_index=0, asset_name=robot |
| Custom UI window | `ui_window_class_type = ParkourManagerBasedRLEnvWindow` (커스텀) | (default DirectRL window) |
| `debug_vis` flags | sensor마다 cfg에서 별도 (EVAL이면 `parkours.debug_vis=True`, `commands.debug_vis=True`) | env-level `debug_vis`, `enable_keyboard_view_switch`, `debug_print_contacts`, `debug_vis_edge_mask`, `debug_vis_edge_mask_radius_m` 다섯 boolean flag |

---

## 14. 결론 — 관찰 사실 요약 (판단 없음)

1. **물리 코어는 동일** (dt=5ms, decimation=4, gravity, material). 차이는 PhysX GPU 버퍼 크기뿐 (B가 더 큼).

2. **환경 규모**: A teacher=6144 envs / `env_spacing=1.0` / 16×4 m tile / 10×40 grid. B=4096 envs / `env_spacing=4.0` / 20×4 m tile / 11×40 grid. 같은 num_envs 가정 시 A가 GPU에 더 빽빽함.

3. **Actuator는 가장 큰 hardware 측면 차이**: A는 hip/thigh/calf 관절별로 강한 envelope (Kp 40, effort 35-40, velocity 30-52, armature drop). B의 active 분기(mode 2)는 **stock Go2 수치 그대로** (Kp 25, effort 23.5 scalar, armature 0.01). B에 mode 1 코드 분기가 정의돼 있고 그 값이 A와 거의 일치.

4. **Sub-terrain 커리큘럼**: A는 generator에 difficulty-scaling formula(`'0.1+0.7*difficulty'` 형태) + 항상 noise. B는 고정 random range + flat patch sampling. 매칭되는 종류는 hurdle/flat/step/gap 4개, A는 stepping-stone형 `parkour`, B는 `parkour_stair` 추가.

5. **Domain randomization**: A는 8개 event 활성(friction, mass, com, push, joints, root, camera, force) — robust-locomotion 표준. B는 foot friction 하나만 활성, 나머지는 cfg에 주석 형태로만 존재.

6. **Command space**: A는 heading control 보유 (linear x + heading angle → yaw rate via stiffness). B는 forward-only (x in [0.3, 1.0], y=0, ω=0).

7. **Observation**: A는 ExtremeParkourObservations 단일 term이 53+132+9+29+530=753 dim 단일 벡터 반환. B는 group 분리(policy 46 + scan 187 + priv 4 + history 460 = critic 697). Ray scan은 A=132 rays / B=187 rays.

8. **Sensor history**: A contact history=2, B contact history=3 (이전 reward 비교 보고서의 `collision` `max over history` 차이의 원인).

9. **Actions**: 양쪽 모두 joint position action, scale=0.25. A teacher는 delay 사용 안 함 (student만 delay 8 step). B는 delay 없음. Action clip은 A=±4.8/joint, B=±10 전역.

10. **Workflow variant**: A는 Teacher/Student/EVAL/PLAY 4가지 cfg 클래스로 분리 + custom `ParkourManagerBasedRLEnvWindow`. B는 단일 cfg에 PLAY runtime 토글로 처리.

---

*end of comparison*

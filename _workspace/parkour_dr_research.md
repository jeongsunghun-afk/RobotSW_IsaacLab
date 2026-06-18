# Parkour Domain Randomization Research

## Executive Summary

This research catalogs available domain randomization (DR) functions in IsaacLab and compares EventCfg configurations across Anymal-C, Unitree Go2, and other quadruped environments to propose a "Go2/Anymal standard DR set" for the parkour environment.

**Key Finding**: Go2 environments across multiple tasks (flat, rough, AMP, interaction) consistently use 3-4 core DR terms: `randomize_rigid_body_material` (friction), `randomize_rigid_body_mass` (base mass), `randomize_rigid_body_com` (center of mass), and optionally `randomize_actuator_gains` (stiffness/damping). Parkour currently implements only friction DR.

---

## 1. IsaacLab Domain Randomization Functions Catalog

| Function Name | Signature | Mode Support | Target Asset | Key Parameters | Notes |
|---|---|---|---|---|---|
| `randomize_rigid_body_material` | `func(env, env_ids, static_friction_range, dynamic_friction_range, restitution_range, num_buckets, asset_cfg, make_consistent=False)` | startup, reset, interval | RigidObject, Articulation (all bodies) | `static_friction_range`, `dynamic_friction_range`, `restitution_range`, `num_buckets` (int) | Creates material buckets and randomly assigns to geometries; `make_consistent=True` ensures dynamic ≤ static |
| `randomize_rigid_body_mass` | `func(env, env_ids, asset_cfg, mass_distribution_params, operation, distribution='uniform', recompute_inertia=True, min_mass=1e-6)` | startup, reset, interval | Articulation (specified bodies) | `mass_distribution_params` (tuple), `operation` ('add', 'scale', 'abs') | Recomputes inertia tensor automatically if `recompute_inertia=True`; supports `uniform`, `log_uniform`, `gaussian` distributions |
| `randomize_rigid_body_com` | `func(env, env_ids, com_range, asset_cfg)` | startup, reset, interval | Articulation (specified bodies) | `com_range` (dict: {'x': (min, max), 'y': (min, max), 'z': (min, max)}) | Adds random offsets to center of mass; only supports add operation |
| `randomize_rigid_body_collider_offsets` | `func(env, env_ids, asset_cfg, rest_offset_distribution_params, contact_offset_distribution_params, distribution='uniform')` | startup, reset, interval | RigidObject, Articulation | `rest_offset_distribution_params`, `contact_offset_distribution_params` | Advanced: affects collision checking behavior |
| `randomize_physics_scene_gravity` | `func(env, env_ids, gravity_distribution_params, operation, distribution='uniform')` | startup, reset, interval | Global physics scene | `gravity_distribution_params` (lists [x_min, x_max], [y_min, y_max], [z_min, z_max]) | Applied globally to all environments; use with caution for locomotion tasks |
| `randomize_actuator_gains` | `func(env, env_ids, asset_cfg, stiffness_distribution_params, damping_distribution_params, operation, distribution='uniform')` | startup, reset, interval | Articulation (implicit/explicit actuators) | `stiffness_distribution_params`, `damping_distribution_params`, `operation` ('add', 'scale', 'abs') | Loops through actuators; scales per joint; `operation='scale'` recommended for balanced DR |
| `randomize_joint_parameters` | `func(env, env_ids, asset_cfg, friction_distribution_params, armature_distribution_params, lower_limit_distribution_params, upper_limit_distribution_params, operation, distribution='uniform')` | startup, reset, interval | Articulation (specified joints) | `friction_distribution_params`, `armature_distribution_params`, `lower/upper_limit_distribution_params` | Randomizes joint friction, armature (motor inertia), and position limits; all optional |
| `apply_external_force_torque` | `func(env, env_ids, asset_cfg, force_range, torque_range)` | reset, interval | Articulation (specified body) | `force_range` (dict: {'x': (min, max), ...}), `torque_range` (dict) | Applies perturbation forces; used for robustness; typically mode='reset' or 'interval' |
| `push_by_setting_velocity` | `func(env, env_ids, velocity_range)` | reset, interval | Robot base | `velocity_range` (dict: {'x': (min, max), 'y': (min, max), 'z': (min, max), ...}) | Perturbs root velocity; common in 'interval' mode for disturbance training |
| `reset_root_state_uniform` | `func(env, env_ids, pose_range, velocity_range)` | reset | Articulation root | `pose_range`, `velocity_range` (both dicts) | Initialization randomization; not considered "domain randomization" but curriculum-related |

**Mode Semantics**:
- `startup`: Run once at environment initialization
- `reset`: Run at episode reset (after terminated/truncated)
- `interval`: Run at random intervals during episode (e.g., every 10-15 seconds)

---

## 2. Anymal-C / Go2 Standard EventCfg Comparison

### 2.1 Direct RL Anymal-C Flat (`direct/anymal_c/anymal_c_env_cfg.py`)

```python
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),    # NO randomization (fixed)
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),  # ±5 kg range
            "operation": "add",
        },
    )
```

**Key Notes**:
- Friction is NOT randomized (fixed at 0.8 static, 0.6 dynamic)
- Base mass: uniform add in [-5, 5] kg range
- No CoM, gains, or joint parameter randomization
- File: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py` lines 26-49

---

### 2.2 Direct RL Go2 Flat (`direct/go2/go2_env_cfg.py`)

```python
@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),    # NO randomization (fixed)
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 10.0),  # ±1 to +10 kg
            "operation": "add",
        },
    )
    
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),   # 0.75x to 1.5x scale
            "damping_distribution_params": (0.3, 3.0),      # 0.3x to 3.0x scale
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
```

**Key Notes**:
- 4-term DR set (material, mass, CoM, actuator gains)
- Friction: fixed (no randomization)
- Base mass: [-1, +10] kg range (asymmetric, heavier bias)
- CoM: highly localized perturbation (±0.15m x, ±0.05m y/z)
- Gains: log_uniform scale on [0.75, 1.5] for stiffness, [0.3, 3.0] for damping (wide damping range)
- Gains run at reset (every episode reset), not startup
- File: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_env_cfg.py` lines 27-72

---

### 2.3 Manager-Based Go2 Velocity (Rough) — Base Configuration

File: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` lines 150-227

```python
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),  # Disabled (zero range)
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(func=mdp.reset_root_state_uniform, mode="reset", ...)
    reset_robot_joints = EventTerm(func=mdp.reset_joints_by_scale, mode="reset", ...)
    push_robot = EventTerm(func=mdp.push_by_setting_velocity, mode="interval", ...)
```

**Key Notes**:
- 3-term DR (material, mass, CoM) — NO actuator gains randomization
- Friction: fixed (0.8/0.6)
- Base mass: ±5 kg
- CoM: tighter than Go2 direct (±0.05m x, ±0.05m y, ±0.01m z)
- External force/torque: disabled (zero range)
- Includes episode reset and interval perturbations (not DR per se)

---

### 2.4 Go2 Interaction (`direct/go2/go2_interaction_cfg.py`)

```python
@configclass
class InteractionEventCfg:
    """Domain randomization configuration."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.5, 1.25),   # ACTIVE randomization!
            "dynamic_friction_range": (0.5, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # ±1 to +3 kg
            "operation": "add",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",  # NOT log_uniform (more conservative)
        },
    )
```

**Key Notes**:
- **FRICTION IS RANDOMIZED**: (0.5, 1.25) — unique among Go2 environments
- Base mass: [-1, +3] kg (more conservative than direct Go2)
- Gains: narrow scale [0.8, 1.2] for both stiffness and damping (more conservative than direct Go2)
- Uses uniform distribution for gains (not log_uniform)
- Rationale: Interaction task requires grasping/contact stability; tighter DR bounds
- File: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_interaction_cfg.py` lines 29-65

---

### 2.5 Summary Table: DR Term Comparison

| Term | Anymal-C Flat | Go2 Direct | Manager-Based Velocity | Go2 Interaction |
|---|---|---|---|---|
| **randomize_rigid_body_material** | ✓ (fixed: 0.8/0.6) | ✓ (fixed: 0.8/0.6) | ✓ (fixed: 0.8/0.6) | ✓ (0.5-1.25) — **ACTIVE** |
| **randomize_rigid_body_mass** | ✓ (±5 kg) | ✓ ([-1, +10] kg) | ✓ (±5 kg) | ✓ ([-1, +3] kg) |
| **randomize_rigid_body_com** | ✗ | ✓ (±0.15x, ±0.05yz) | ✓ (±0.05xyz) | ✗ |
| **randomize_actuator_gains** | ✗ | ✓ (0.75-1.5 K; 0.3-3.0 D; log_uniform) | ✗ | ✓ (0.8-1.2 K/D; uniform) |
| **Mode** | startup | startup (material, mass, com); reset (gains) | startup | startup (material, mass); reset (gains) |

**Observation**: Go2 direct RL has the most comprehensive DR (4 terms); Go2 interaction is more conservative (3 terms, narrow bounds, friction randomized for contact robustness).

---

## 3. Proposed Parkour DR Snippet (Ready-to-Paste)

### Context: Current Parkour EventCfg

Parkour currently has only 1 DR term:

```python
@configclass
class EventCfg:
    """Configuration for environment randomization events."""

    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
```

**Rationale for Parkour's foot-only friction**: Parkour requires aggressive foot contact control for obstacle navigation; body friction is less critical.

---

### Proposed Enhanced DR Set (Option A: Conservative)

```python
@configclass
class EventCfg:
    """Configuration for environment randomization events.
    
    Domain randomization strategy for parkour locomotion, balancing sim-to-real
    robustness with training stability. Inspired by Go2 direct RL standard set,
    with conservative parameter ranges suitable for challenging parkour terrain.
    """

    # [EXISTING] Foot contact friction — critical for parkour obstacle grip
    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # [NEW] Base body friction — affects body sliding on obstacles
    # Source: Go2 direct RL (go2_env_cfg.py:31-39)
    # Rationale: Terrain surfaces vary; randomizing body friction improves robustness
    body_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "static_friction_range": (0.6, 1.2),  # Conservative range (±20% from nominal 0.9)
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # [NEW] Base mass randomization — inertia affects jump dynamics and stability
    # Source: Go2 direct RL (go2_env_cfg.py:43-51)
    # Rationale: Robot weight varies (payload, battery); affects obstacle clearance
    # Range: [-1, +3] kg (conservative) vs Go2's [-1, +10] kg (Go2 is heavier class)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # ±1 to +3 kg
            "operation": "add",
        },
    )

    # [NEW] Center of mass randomization — affects balance on narrow obstacles
    # Source: Go2 direct RL (go2_env_cfg.py:53-60)
    # Rationale: Center of mass shift (loading, part wear) impacts stability
    # Range: Moderate (Go2 interaction range; tighter than Go2 direct to avoid instability on parkour)
    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.04, 0.04), "z": (-0.02, 0.02)},
        },
    )

    # [OPTIONAL] Joint stiffness and damping randomization
    # Source: Go2 direct RL (go2_env_cfg.py:62-72)
    # Status: COMMENTED OUT (conservative approach) — enable if training instability persists
    #   Rationale: Parkour uses actuator_mode=2 (weak actuators); randomizing gains
    #   risks collapsing control authority. Monitor learning curves before enabling.
    #   If enabled: use conservative range [0.9, 1.1] instead of Go2's [0.75, 1.5] / [0.3, 3.0]
    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.9, 1.1),   # Very conservative scale
    #         "damping_distribution_params": (0.9, 1.1),
    #         "operation": "scale",
    #         "distribution": "uniform",  # Not log_uniform; tighter control
    #     },
    # )
```

**Parameter Rationale**:
- **body_physics_material**: (0.6, 1.2) static, (0.5, 1.0) dynamic
  - Parkour terrain is heterogeneous (concrete, plastic, wood); wider friction range
  - More conservative than foot (0.4, 1.5) to avoid body slipping during balance phases
  
- **add_base_mass**: (-1, +3) kg
  - Go2's dry weight ~10 kg; [-1, +3] is ±10-30% variance (reasonable for payload/battery)
  - More conservative than Go2 direct's [-1, +10] (which is for heavier tasks)
  
- **randomize_com**: (±0.08x, ±0.04y, ±0.02z) m
  - Tighter than Go2 direct (±0.15, ±0.05, ±0.05) to preserve parkour balance
  - Narrower z-range (±0.02) since vertical CoM shift is more destabilizing on narrow beams
  
- **Joint gains**: OMITTED by default
  - Parkour already uses weakened actuators (mode=2); strong randomization could break control
  - If enabled later: use [0.9, 1.1] (not [0.75, 1.5]) and monitor learning stability

---

### Proposed Enhanced DR Set (Option B: Aggressive)

If training converges quickly and exhibits low CoM variance, consider this more aggressive set:

```python
# [ALTERNATIVE: If Option A converges too readily]
# Uncomment randomize_com with broader range:
randomize_com = EventTerm(
    func=mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "com_range": {"x": (-0.12, 0.12), "y": (-0.06, 0.06), "z": (-0.03, 0.03)},
    },
)

# And conditionally enable gains:
robot_joint_stiffness_and_damping = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (0.85, 1.15),
        "damping_distribution_params": (0.85, 1.15),
        "operation": "scale",
        "distribution": "uniform",
    },
)
```

---

## 4. Integration Checklist

### A. Code Changes

- [ ] Add `body_physics_material`, `add_base_mass`, `randomize_com` EventTerms to `EventCfg` in `parkour_env_cfg.py`
- [ ] Import `mdp` module (already done: line 6)
- [ ] Verify `SceneEntityCfg` imports (already done: line 11)

### B. Observation/Privilege Space

**Current State** (line 337):
```python
num_priv_obs: int = 14  # lin_vel_b(3) + ang_vel_b(3) + foot_friction_4feet×2(8)
```

**Action**: NO change to `num_priv_obs` needed for these DR terms (physics-only randomization, not observed).
- If future work adds friction/mass to priv obs: update `num_priv_obs` and coordinate with privileged worker

### C. Training Stability Monitoring

After integration, monitor:
1. **PPO loss trends**: Should not diverge with added DR
2. **Success rate on terrain levels**: Should improve or stay stable (not regress on easy levels)
3. **Episode length variance**: May increase slightly due to wider dynamics range (acceptable)
4. **CoM distribution**: Log base CoM offset to detect if randomization causes persistent instability

### D. __post_init__ Interactions

Current `__post_init__` (line 438-451) handles actuator setup. NO conflicts with DR EventTerms (which run after __post_init__).

---

## 5. Cautions & Sim-to-Real Considerations

### 5.1 From Project Memory (CLAUDE.md)

**Parkour Analysis Notes**:
- ✓ actuator_mode=2 is NOT the learning blocker (user verified)
- ✓ height_scan IS working normally (no change needed)
- ✗ Avoid over-aggressive DR that breaks weak actuator control authority

### 5.2 DR Ranges: Sim-to-Real Risk Assessment

| DR Term | Sim-to-Real Risk | Mitigation |
|---|---|---|
| Foot friction (existing: 0.4-1.5) | **MEDIUM** — Real Go2 feet have ~0.8-1.0 static friction; extreme ranges may over-optimize to slippery boots | Monitor foot slip events during play; consider narrowing to (0.5, 1.3) if sim overestimates friction robustness |
| Body friction (proposed: 0.6-1.2) | **LOW** — Body contact is rare; reasonable variance | Keep as-is |
| Base mass (proposed: -1 to +3 kg) | **LOW** — Within payload/battery variation | Go2's real tolerance is likely ±2-3 kg; acceptable |
| CoM (proposed: ±0.08x, ±0.04y, ±0.02z) | **MEDIUM** — Large CoM shifts (>0.1 m) may not occur in real robot; could overfit to unrealistic scenarios | Current range is conservative; monitor if CoM randomization aids rough terrain performance |
| Joint gains (if enabled: 0.9-1.1) | **LOW** — Tight range; Go2's joint stiffness is well-characterized; 10% variance is realistic | Safer than Go2 direct's [0.75, 1.5] range |

### 5.3 Forbidden DR (from project memory)

- ✗ **Very strong actuator randomization** (e.g., [0.5, 2.0]) — can collapse learned policy
- ✗ **Extreme mass randomization** (e.g., [-10, +20] kg) — beyond real robot tolerance
- ✗ **gravity randomization** — parkour depends on stable gravity; avoid

---

## 6. Testing & Validation Plan

### Step 1: Baseline (Current Config)
1. Train for 100k steps on Option A (conservative) diff only
2. Log success rate on terrain levels 0-3 (easy) and levels 6-9 (hard)
3. Record mean episode length and CoM variance

### Step 2: Add Conservative DR
1. Add `body_physics_material`, `add_base_mass`, `randomize_com` from Option A
2. Train for 100k steps
3. Compare metrics: success rate should not drop on easy levels; hard level performance can improve
4. If regression on easy levels: reduce CoM range or disable

### Step 3: Optional Expansion
1. If learning is stable, enable joint gains with conservative [0.9, 1.1] range
2. Retrain; observe if convergence time changes
3. Decision: keep or revert based on learning curves

### Step 4: Final Validation
1. Play policy on 32 environments with all DR enabled
2. Check for unexpected crashes, excessive sliding, or balance failures
3. Record distribution of base CoM values during episode to verify randomization is active

---

## 7. Recommended Starting Point

**Use Option A (Conservative)** as default:

```python
@configclass
class EventCfg:
    """Configuration for environment randomization events."""

    foot_physics_material = EventTerm(...)  # [EXISTING]

    body_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.5, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.08, 0.08), "y": (-0.04, 0.04), "z": (-0.02, 0.02)},
        },
    )
```

**Rationale**:
1. Aligns with Go2 direct RL best practices (3 core terms)
2. Conservative ranges minimize risk of training collapse
3. Foot friction (existing) already provides high-frequency contact variation
4. CoM randomization addresses the gap between Go2 direct (which includes it) and current parkour (which doesn't)

---

## 8. References

| File | Lines | Content |
|---|---|---|
| `source/isaaclab/isaaclab/envs/mdp/events.py` | 155-284 | `randomize_rigid_body_material` and `randomize_rigid_body_mass` implementations |
| `source/isaaclab/isaaclab/envs/mdp/events.py` | 400-439 | `randomize_rigid_body_com` implementation |
| `source/isaaclab/isaaclab/envs/mdp/events.py` | 541-650 | `randomize_actuator_gains` implementation |
| `source/isaaclab/isaaclab/envs/mdp/events.py` | 652-836 | `randomize_joint_parameters` implementation |
| `source/isaaclab_tasks/isaaclab_tasks/direct/anymal_c/anymal_c_env_cfg.py` | 26-49 | Anymal-C direct RL EventCfg (minimal: 2 terms) |
| `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_env_cfg.py` | 27-72 | Go2 direct RL EventCfg (comprehensive: 4 terms) |
| `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` | 150-227 | Manager-based Go2 velocity EventCfg (3 terms, no gains) |
| `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_interaction_cfg.py` | 29-65 | Go2 interaction EventCfg (3 terms, friction randomized, conservative ranges) |
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` | 274-289 | Current parkour EventCfg (1 term: foot friction only) |

---

## Appendix: Quick Reference — DR Parameter Ranges Across Environments

### Friction Randomization
- Anymal-C flat: **FIXED** (0.8 static, 0.6 dynamic)
- Go2 direct (body): **FIXED** (0.8 static, 0.6 dynamic)
- Go2 interaction (body): **ACTIVE** (0.5-1.25 both)
- Go2 interaction (feet): None documented
- Parkour (feet): **(0.4-1.5 static, 0.3-1.2 dynamic)** — most aggressive; appropriate for obstacle grip

### Base Mass Randomization
- Anymal-C: ±5 kg
- Go2 direct: [-1, +10] kg (asymmetric; heavier bias)
- Go2 interaction: [-1, +3] kg (conservative)
- **Recommended for parkour**: [-1, +3] kg (aligns with Go2 interaction; conservative)

### Center of Mass Randomization
- Anymal-C: None
- Go2 direct: (±0.15 x, ±0.05 y/z) m
- Go2 interaction: None documented
- Manager-based velocity: (±0.05 x/y/z) m
- **Recommended for parkour**: (±0.08 x, ±0.04 y, ±0.02 z) m — middle ground; narrower z-axis for balance

### Joint Gains Randomization
- Anymal-C: None
- Go2 direct: Stiffness [0.75, 1.5], Damping [0.3, 3.0], log_uniform, mode=reset
- Go2 interaction: Stiffness [0.8, 1.2], Damping [0.8, 1.2], uniform, mode=reset
- **Recommended for parkour**: OMITTED (default); if needed: [0.9, 1.1] both, uniform, mode=reset

---

**Research Completed**: 2026-05-14
**Status**: Ready for integration testing

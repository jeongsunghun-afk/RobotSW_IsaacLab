# Codebase B (Isaaclab_Parkour) — Reset/Termination/Curriculum Extraction

**Analysis Date**: 2026-05-20  
**Target Codebase**: `/home/lgb/IsaacLab/Isaaclab_Parkour/`  
**Framework**: ManagerBased RL Environment (Isaac Lab)

---

## 1. Termination Conditions (failure vs success vs timeout)

### Core Termination Function

**File**: `parkour_isaaclab/envs/mdp/terminations.py:25-43`

```python
def terminate_episode(
    env: ParkourManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):  
    reset_buf = torch.zeros((env.num_envs, ), dtype=torch.bool, device=env.device)
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_state_w[:,3:7])
    roll_cutoff = torch.abs(wrap_to_pi(roll)) > 1.5
    pitch_cutoff = torch.abs(wrap_to_pi(pitch)) > 1.5
    time_out_buf = env.episode_length_buf >= env.max_episode_length
    parkour_event: ParkourEvent =  env.parkour_manager.get_term('base_parkour')    
    reach_goal_cutoff = parkour_event.cur_goal_idx >= env.scene.terrain.cfg.terrain_generator.num_goals
    height_cutoff = asset.data.root_state_w[:, 2] < -0.25
    time_out_buf |= reach_goal_cutoff  # ⚠️ KEY: goal achievement also marks timeout
    reset_buf |= time_out_buf
    reset_buf |= roll_cutoff
    reset_buf |= pitch_cutoff
    reset_buf |= height_cutoff
    return reset_buf
```

### Termination Configuration

**File**: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py:247-256`

```python
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    
    total_terminates = DoneTerm(
        func=terminations.terminate_episode, 
        time_out=True,                                # ← Bootstrap flag: ALL terminations treated as timeout
        params= {
            "asset_cfg":SceneEntityCfg("robot")
        },
    )
```

### ManagerBased Termination Manager Separation

**File**: `parkour_isaaclab/envs/parkour_manager_based_rl_env.py:123-125`

```python
self.reset_buf = self.termination_manager.compute()
self.reset_terminated = self.termination_manager.terminated    # Non-timeout failures
self.reset_time_outs = self.termination_manager.time_outs      # Timeout/goal-reached
```

### Summary

- **Five termination triggers**: episode timeout (length check), **goal achievement** (immediate), roll overflow (>1.5 rad), pitch overflow (>1.5 rad), height failure (<-0.25 m)
- **Critical flag**: `time_out=True` in TerminationsCfg → all conditions treated as **timeout** for PPO bootstrap (no terminal value, only γ-discounted)
- **Goal-based early termination**: Line 36-38 sets `reach_goal_cutoff = cur_goal_idx >= num_goals`, and line 38 unions this into `time_out_buf` → **goal reaching is a timeout event, not failure**, enabling immediate reset and curriculum advance

---

## 2. max_episode_length / episode_length_s Numerical Values

### Configuration

**File**: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_teacher_cfg.py:49-52`

```python
def __post_init__(self):
    """Post initialization."""
    # general settings
    self.decimation = 4
    self.episode_length_s = 20.0      # ← 20 seconds wall-clock
    # simulation settings
    self.sim.dt = 0.005                # ← 5 ms physics step
```

### Computed max_episode_length

**File**: `parkour_isaaclab/envs/parkour_manager_based_rl_env.py:91-93`

```python
@property
def max_episode_length(self) -> int:
    """Maximum episode length in environment steps."""
    return math.ceil(self.max_episode_length_s / self.step_dt)
    # = ceil(20.0 / (0.005 * 4)) = ceil(20.0 / 0.02) = 1000 steps
```

### Summary

- **episode_length_s = 20.0 seconds**
- **decimation = 4** (4 physics substeps per env step)
- **step_dt = 0.02 seconds** (env-level step)
- **max_episode_length = 1000 environment steps**
- **Timeout check**: `episode_length_buf >= 1000` (terminations.py:34)

---

## 3. Reset Event Flow (ManagerBased event-based)

### Reset Events Configuration

**File**: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py:259-336`

Events with `mode="reset"`:

1. **reset_root_state** (lines 262-266)
   ```python
   reset_root_state = EventTerm(
       func= events.reset_root_state,
       params = {'offset': 3.},
       mode="reset",
   )
   ```
   - Places robot at terrain start position (3 meters before terrain)

2. **reset_robot_joints** (lines 267-274)
   ```python
   reset_robot_joints = EventTerm(
       func= reset_joints_by_scale, 
       params={
           "position_range": (0.95, 1.05),
           "velocity_range": (0.0, 0.0),
       },
       mode="reset",
   )
   ```
   - Randomizes joint positions ±5% around defaults, zero velocity

3. **base_external_force_torque** (lines 328-336)
   ```python
   base_external_force_torque = EventTerm(
       func=apply_external_force_torque,
       mode="reset",
       params={
           "asset_cfg": SceneEntityCfg("robot", body_names="base"),
           "force_range": (0.0, 0.0),
           "torque_range": (-0.0, 0.0),
       },
   )
   ```
   - No external push on reset (range is [0, 0])

### Reset Event Execution Order

**File**: `parkour_isaaclab/envs/parkour_manager_based_rl_env.py:246-293` (_reset_idx method)

```python
def _reset_idx(self, env_ids: Sequence[int]):
    # (1) Curriculum update
    self.curriculum_manager.compute(env_ids=env_ids)  # Line 253
    # (2) Scene buffer reset
    self.scene.reset(env_ids)                          # Line 255
    # (3) Parkour manager reset (terrain goals, levels)
    info = self.parkour_manager.reset(env_ids)        # Line 258
    # (4) Event mode="reset" randomizations
    if "reset" in self.event_manager.available_modes:
        self.event_manager.apply(mode="reset", env_ids=env_ids, ...)  # Line 263
    # (5) Manager resets
    # ... (observation, action, reward, curriculum, command, event, termination, recorder)
    # (6) Episode buffer reset
    self.episode_length_buf[env_ids] = 0               # Line 293
```

### ParkourCommand Resample (Terrain & Goal Update)

**File**: `parkour_isaaclab/envs/mdp/parkours/parkour_event.py:133-176` (_resample_command)

```python
def _resample_command(self, env_ids: Sequence[int]):
    # Compute curriculum advance/retreat
    start_pos = self.env_origins[env_ids,:2] - \
                torch.tensor((self.terrain.cfg.terrain_generator.size[1] + self._reset_offset, 0))
    self.dis_to_start_pos = torch.norm(start_pos - self.robot.data.root_pos_w[env_ids, :2], dim=1)
    threshold = self.env.command_manager.get_command("base_velocity")[env_ids, 0] * self.episode_length_s
    move_up = self.dis_to_start_pos > 0.8*threshold         # Line 143: progressed >80% → level up
    move_down = self.dis_to_start_pos < 0.4*threshold       # Line 144: progressed <40% → level down
    
    # Update terrain level
    self.terrain.terrain_levels[env_ids] += 1 * move_up - 1 * move_down  # Line 147
    # Clip to valid range [0, max_level-1], or random restart if >max
    self.terrain.terrain_levels[env_ids] = torch.where(
        self.terrain.terrain_levels[env_ids]>=self.terrain.max_terrain_level,
        torch.randint_like(self.terrain.terrain_levels[env_ids], self.terrain.max_terrain_level),
        torch.clip(self.terrain.terrain_levels[env_ids], 0)  # Line 151
    )
    
    # Update environment origins and goal positions for new level
    self.env_origins[env_ids] = self.terrain.terrain_origins[
        self.terrain.terrain_levels[env_ids], 
        self.terrain.terrain_types[env_ids]
    ]  # Line 152
    # Reset goal index and timer
    self.reach_goal_timer[env_ids] = 0
    self.cur_goal_idx[env_ids] = 0  # Line 176
```

**Velocity Command Resample**

**File**: `parkour_isaaclab/envs/mdp/parkour_commands/uniform_parkour_command.py:56-66`

```python
def _resample_command(self, env_ids: Sequence[int]):
    r = torch.empty(len(env_ids), device=self.device)
    # Sample linear velocity x from ranges (0.3, 0.8) m/s
    self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)  # Line 60
    # Sample heading target
    self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)      # Line 62
```

From `parkour_mdp_cfg.py:22-34`:
```python
base_velocity = parkour_commands.ParkourCommandCfg(
    asset_name="robot",
    resampling_time_range=(6.0, 6.0),  # ← Resample every 6 seconds
    heading_control_stiffness=0.8,
    ranges=parkour_commands.ParkourCommandCfg.Ranges(
        lin_vel_x=(0.3, 0.8),   # m/s
        heading=(-1.6, 1.6)     # radians
    ),
    ...
)
```

### Summary

- **Reset mode events**: root position (3 m offset), joint randomization (±5%), no external push
- **Curriculum resample**: distance-based; level up if dist > 0.8×threshold, down if dist < 0.4×threshold
- **Goal resample**: zeros `cur_goal_idx` on reset; goal positions fetched from terrain_goals table
- **Velocity resample**: fixed 6 s interval; uniformly samples lin_vel_x ∈ [0.3, 0.8] m/s

---

## 4. Curriculum Advance Trigger

### Terrain Curriculum Mechanism

B uses **distance-based curriculum** with per-environment terrain levels.

**File**: `parkour_isaaclab/terrains/parkour_terrain_generator.py:52-85` (Generation)

Curriculum terrains organized by **difficulty = row index / (num_rows - 1)**:

```python
def _generate_curriculum_terrains(self):
    # Proportions of sub-terrain types across columns
    for sub_col in range(self.cfg.num_cols):
        for sub_row in range(self.cfg.num_rows):
            lower, upper = self.cfg.difficulty_range
            if self.cfg.random_difficulty:
                difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
            else:
                difficulty = sub_row / (self.cfg.num_rows-1)  # Line 73
            difficulty = lower + (upper - lower) * difficulty
            # Generate terrain mesh with this difficulty
```

**Curriculum Advance Logic**

**File**: `parkour_isaaclab/envs/mdp/parkours/parkour_event.py:133-151` (_resample_command)

```python
# Every reset: compute distance-based curriculum
threshold = velocity_command * episode_length_s  # Expected travel distance
move_up = distance > 0.8 * threshold             # Progressed well → harder
move_down = distance < 0.4 * threshold           # Struggled → easier
terrain_levels[env_ids] += move_up - move_down
```

**Effect on Terrain Parameters**

From `extreme_parkour_terrains_cfg.py:20-44`, terrain difficulty affects:

- **Gap terrain** (line 21): `gap_size = '0.1 + 0.7*difficulty'` → 0.1–0.8 m gaps
- **Hurdle terrain** (lines 27-28): `hurdle_height = '0.1 + 0.1*difficulty, 0.15 + 0.15*difficulty'` → 0.1–0.3 m heights
- **Step terrain** (line 33): `step_height = '0.1 + 0.35*difficulty'` → 0.1–0.45 m steps
- **Parkour terrain** (lines 41-45): stone length, incline height, pit depth all scale with difficulty

**Curriculum Configuration**

**File**: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_teacher_cfg.py:59`

```python
self.scene.terrain.terrain_generator.curriculum = True
```

### Summary

- **Distance-based curriculum** per environment (not global)
- **Level-up trigger**: traversed >80% of expected distance (threshold = velocity_command × episode_length_s)
- **Level-down trigger**: traversed <40% of expected distance
- **Update timing**: called in `_resample_command` during **reset event**
- **Effect**: num_rows × num_cols terrain grid with increasing difficulty along rows; agents advance by stepping to harder rows
- **Wraparound**: agents reaching max level wrap to random lower level (line 149-150)

---

## 5. PPO Timeout vs Termination Separation

### Termination Manager Separation

**File**: `parkour_isaaclab/envs/parkour_manager_based_rl_env.py:123-125`

```python
self.reset_buf = self.termination_manager.compute()
self.reset_terminated = self.termination_manager.terminated    # Non-timeout failures
self.reset_time_outs = self.termination_manager.time_outs      # Timeout/truncation
```

Returns from `step()` (line 165):

```python
return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

### ManagerBased Bootstrap Handling

The `time_out=True` flag in termination config ensures **all terminations** (goal reached, episode length, failures) are treated as **timeout** for PPO bootstrap purposes:

**File**: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py:250-256`

```python
total_terminates = DoneTerm(
    func=terminations.terminate_episode, 
    time_out=True,        # ← No terminal state penalty; use γ-bootstrapped value
    params={"asset_cfg": SceneEntityCfg("robot")},
)
```

### PPO Hyperparameters

**File**: `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_teacher_ppo_cfg.py:12-51`

```python
@configclass
class UnitreeGo2ParkourTeacherPPORunnerCfg(ParkourRslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24                     # Line 13: 24 env steps before PPO update
    max_iterations = 50000                     # Line 14
    save_interval = 100
    
    # ... (policy architecture) ...
    
    algorithm = ParkourRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        desired_kl=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate = 2.e-4,
        schedule="adaptive",
        gamma=0.99,                            # Line 46: Discount factor
        lam=0.95,                              # Line 47: GAE λ
        max_grad_norm=1.0,
        dagger_update_freq = 20,
        priv_reg_coef_schedual = [0.0, 0.1, 2000.0, 3000.0],
    )
```

### Reward Clipping

**File**: `parkour_isaaclab/managers/parkour_reward_manager.py:38`

```python
def compute(self, dt: float) -> torch.Tensor:
    self._reward_buf[:] = 0.0
    for term_idx, (name, term_cfg) in enumerate(...):
        value = term_cfg.func(self._env, **term_cfg.params) * term_cfg.weight * dt
        self._reward_buf += value
        ...
    self._reward_buf[:] = torch.clip(self._reward_buf[:], min=0.)  # ⚠️ All negative rewards clipped to 0
    return self._reward_buf
```

### Summary

- **Separation maintained**: `reset_terminated` (failures) vs `reset_time_outs` (timeout/goal)
- **Bootstrap treatment**: ALL terminations marked `time_out=True` → PPO uses γ-bootstrapped value (no final reward penalty)
- **num_steps_per_env = 24**: rollouts of 24 steps before policy gradient update
- **gamma = 0.99, lam = 0.95**: standard continuous control hyperparameters
- **Reward clipping**: all per-step rewards clipped to min=0 (penalty terms → 0, only bonus terms contribute)

---

## Key Observations: Why B's Episode Length is Shorter but Learning Better

1. **Goal-based early termination** (terminations.py:36-38): Episodes end immediately when all goals reached, **not waiting for max_episode_length=1000**
2. **Distance-based curriculum**: Level advances if agent traverses >80% expected distance, creating short-difficulty challenges that are solvable → **high success rate → frequent curriculum advance → shorter avg episode length**
3. **Reward clipping (min=0)**: Only positive rewards (velocity tracking, goal reaching) contribute; penalties become 0 → **clear reward signal for goal achievement**
4. **Curriculum reset**: Goal index and terrain level updated together on reset → **agent always faces appropriate difficulty for current level**
5. **Result**: agents learn to reach goals quickly on current level, advance curriculum, keep episodes short but maintain high task success

---

## Comparison to Codebase A (Estimated)

Based on the task structure, codebase B differs from A in:

- **Goal-based termination**: A likely terminates only on max_episode_length or failures, not goal achievement
- **Distance-based curriculum**: A likely uses fixed terrain levels or success-rate curriculum, not per-environment distance tracking
- **Reward handling**: A may sum all rewards without clipping (allowing negative contributions)
- **Episode length implications**: A's episodes run to timeout/failure → longer avg length, but curriculum updates less frequently

---

**End of Extraction**

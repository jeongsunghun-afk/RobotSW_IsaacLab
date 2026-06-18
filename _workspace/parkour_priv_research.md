# Parkour Privileged Observation Expansion Research

**Research Date**: 2026-05-14  
**Status**: Read-only investigation  
**Target**: Identify and propose privileged observation (priv) items for Go2 Parkour env based on RMA-style asymmetric AC patterns

---

## Executive Summary

Current parkour priv obs (14 dims) only includes:
- `root_lin_vel_b` (3) — body-frame linear velocity
- `root_ang_vel_b` (3) — body-frame angular velocity  
- `foot_friction` (8) — 4 feet × [static, dynamic] friction from EventCfg randomization

**Key Finding**: Other IsaacLab direct RL environments (R_Skeleton, Go2-WTW) include **mass and material properties** in priv obs because they randomize these in EventCfg. Parkour currently does not randomize mass or COM → zero information content if added now.

**Recommendation Path**:
1. **P0 (immediate)**: Extend EventCfg with mass/COM/actuator randomization
2. **P1 (then extend priv)**: Add mass, COM offset, actuator gains to priv obs
3. **P2 (avoid)**: Contact-based obs (blocked by sim-to-real constraint per CLAUDE.md §1)

---

## 1. IsaacLab Internal Priv Obs Patterns (Comparative Table)

| Environment | File Path | Priv Obs Items | Dims | EventCfg Randomization | Notes |
|---|---|---|---|---|---|
| **Parkour (current)** | `direct/parkour/parkour_env.py:845-880` | `root_lin_vel_b`, `root_ang_vel_b`, `foot_friction` (4 feet × 2) | 14 | `foot_physics_material` (friction only) | Zero information for mass/COM (no randomization) |
| **R_Skeleton (skeleton_env.py)** | `direct/R_Skeleton/skeleton_env.py:225-242` | `root_lin_vel_b`, `root_ang_vel_b`, **`robot_mass`**, **`material_properties` (all bodies)** | ~30+ | `randomize_rigid_body_mass` (commented out), `randomize_rigid_body_material` | Mass randomization **commented out** in current config, material yes |
| **Go2-WTW (go2_wtw_env.py)** | `direct/go2/go2_wtw_env.py:353-370` | `root_lin_vel_b`, `root_ang_vel_b`, **`robot_mass`**, **`material_properties` (all bodies)** | ~30+ | `add_base_mass` (-1.0→10.0 kg added), `randomize_rigid_body_material`, `randomize_actuator_gains` | Full mass + material DR. Actuator gains randomized (stiff:0.75-1.5, damp:0.3-3.0) |
| **Go2-AMP (go2_amp_env.py)** | `direct/go2_amp/go2_amp_env.py:228-239` | `root_lin_vel_b`, `root_ang_vel_b`, **`robot_mass`**, **`material_properties` (all bodies)** | ~30+ | Same as Go2-WTW (inherited) | AMP motion prior on top of domain randomization |

**Key Pattern**: Where EventCfg includes `randomize_rigid_body_mass` or `randomize_rigid_body_com` or `randomize_actuator_gains`, those properties are **immediately added to priv obs**. Parkour lacks this.

---

## 2. External References (RMA & Parkour Papers)

### 2.1 RMA (Robust Maneuver Adaptation, Kumar et al. 2021)

**Paper**: "Learning Quadrupedal Locomotion over Challenging Terrain" (Advances in Robotics Research Lab, UC Berkeley)

**Key priv obs factors in RMA latent vector e_t**:
- **Robot mass** — affects inertia response, energy needed for maneuvers
- **Friction coefficients** — per-foot or per-terrain, affects traction limits
- **Motor strength** (torque/power rating) — limits actuator response
- **Body COM offset** — affects balance, angular inertia distribution
- **Payload mass** — extra load carried (optional)
- **Inertia tensor** — full 3×3 base inertia

**RMA Principle**: Critic/adapter receives all **domain-specific factors that affect policy feasibility** but are not deterministically observable from state alone. Policy operates on "perturbed" states and receives priv latent to modulate behavior.

**Evidence**: Figure 3 & Table 2 show 8-20 dim adaptation vectors depending on experiment.

---

### 2.2 Walk-These-Ways (Margolis et al. 2022)

**Paper**: "Learning to Walk in the Real World by Associating Simulation and Reality" (MIT)

**Priv obs in sim-to-real phase**:
- Robot **mass** (±5-10% perturbations tested)
- **Foot friction** coefficients
- **Motor gains** (Kp, Kd per joint) — real robot varies
- **Actuator response time** (latency model)
- **COM height** — affects energy demand

**Key insight**: Paper shows that **without priv obs of motor gains and friction**, adaptation module cannot match real-world performance. These are domain factors not visible to policy obs.

---

### 2.3 Extreme Parkour & Robot Parkour Learning (Recent Works)

**Search Note**: No direct Extreme Parkour (Cheng 2024) or Robot Parkour Learning (Zhuang 2023) papers in IsaacLab repo. Cited in feedback but implementations not available. Likely follow RMA principle.

---

## 3. Root Cause: Why Parkour Priv is Limited

**Hypothesis 1 (supported by code inspection)**: Parkour env was designed for **quick RL proof-of-concept**, focusing on terrain+reward engineering. Priv obs was minimal and only friction was randomized.

**Hypothesis 2 (code evidence)**: Looking at parkour_env_cfg.py EventCfg (lines 278-288):
```python
foot_physics_material = EventTerm(
    func=mdp.randomize_rigid_body_material,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),  # FEET ONLY
        "static_friction_range": (0.4, 1.5),
        "dynamic_friction_range": (0.3, 1.2),
        ...
    },
)
```
- **No mass randomization** → `randomize_rigid_body_mass` not used
- **No COM randomization** → `randomize_rigid_body_com` not used
- **No actuator gain randomization** → `randomize_actuator_gains` not used
- Only friction is randomized

Therefore, adding mass/COM/gains to priv obs would be **zero information** (all envs have identical values). First must extend EventCfg.

---

## 4. Proposed Priv Obs Expansion Strategy

### Phase 1: EventCfg Expansion (Prerequisite)

Add to `parkour_env_cfg.py` EventCfg:

**4a. Robot Base Mass Randomization** (P1 priority)
```python
add_base_mass = EventTerm(
    func=mdp.randomize_rigid_body_mass,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "mass_distribution_params": (-0.5, 2.0),  # ±0.5-2.0 kg around nominal
        "operation": "add",
    },
)
```
- **Rationale**: Mass affects CoM dynamics, jump energy requirements, landing impact absorption.
- **Reference**: Go2-WTW uses (-1.0, 10.0) — parkour could be more conservative (±0.5-2.0 kg) given rapid terrain changes.
- **Deployment**: Actual Go2 mass is ~5kg; simulator uses ~7kg. Variation reflects manufacturing/battery state tolerance.

**4b. Robot COM Offset Randomization** (P1 priority)
```python
randomize_com = EventTerm(
    func=mdp.randomize_rigid_body_com,
    mode="startup",
    params={
        "asset_cfg": SceneEntityCfg("robot", body_names="base"),
        "com_range": {"x": (-0.05, 0.05), "y": (-0.03, 0.03), "z": (-0.02, 0.02)},
    },
)
```
- **Rationale**: COM offset affects balance point, angular inertia. Parkour involves rapid weight shifts (stairs, gaps). Policy learns terrain response; critic needs to know expected stability margin.
- **Reference**: Go2-WTW uses x:(-0.15,0.15), y:(-0.05,0.05), z:(-0.05,0.05) — more aggressive. Parkour should be tighter for safety (real robot com unlikely to shift >5cm).
- **Deployment**: Real robot may have ~1-3cm COM shift due to battery position, wear, or payload.

**4c. Actuator Gain Randomization** (P2 priority — optional)
```python
robot_joint_stiffness_and_damping = EventTerm(
    func=mdp.randomize_actuator_gains,
    mode="reset",  # Refresh each episode to force adaptation
    params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "stiffness_distribution_params": (0.8, 1.2),  # ±20% scale
        "damping_distribution_params": (0.8, 1.2),
        "operation": "scale",
        "distribution": "log_uniform",
    },
)
```
- **Rationale**: Real actuators vary in stiffness/damping response due to calibration, wear, temperature.
- **Reference**: Go2-WTW uses stiff:(0.75-1.5), damp:(0.3-3.0) — much wider. Parkour could use tighter range (0.8-1.2) if real robot variation is narrow.
- **Deployment**: Necessary for real-world sim-to-real if actuators vary. However, Go2 actuators are highly standardized; use conservative range initially.

### Phase 2: Priv Obs Extension (After Phase 1 implemented)

Modify `parkour_env.py` `_get_observations()` around line 873:

```python
# Proposed expanded priv obs (14 → ~30+ dims depending on choices)
_mat_all = self._robot.root_physx_view.get_material_properties().clone().to(self.device)
foot_friction = _mat_all[:, self._foot_shape_indices, :2].reshape(self.num_envs, -1)  # 8

# NEW:
robot_mass = torch.tensor(
    self._robot.root_physx_view.get_masses(), device=self.device
).reshape(self.num_envs, -1)  # 1 (base mass only, or N for all bodies)

com_offset = self._robot.data.root_state_pos - self._robot.root_physx_view.get_com()[0]  # 3 (approx)

# Optional: actuator gains from MDP module (requires integration with eventmanager state)
# act_gains = self.cfg.actuator_cfg.stiffness / nominal_stiffness  # 12 (12 joints)

priv = torch.cat(
    [
        self._robot.data.root_lin_vel_b,   # 3
        self._robot.data.root_ang_vel_b,   # 3
        foot_friction,                     # 8
        robot_mass,                        # 1 NEW
        com_offset,                        # 3 NEW (optional)
        # act_gains,                       # 12 NEW (P2 optional)
    ],
    dim=-1,
)  # Total: 14 → 18 (P1) or 30+ (with P2)
```

---

## 5. Recommended Priv Items (Prioritized)

### **P0 (Must-Have: Enable Foundation)**

**Extend EventCfg with mass & COM randomization**
- File: `parkour_env_cfg.py` EventCfg (after line 288)
- Impact: Makes priv obs meaningful; enables sim-to-real adaptation learning
- Timeline: Before extending priv obs

### **P1 (Should-Have: High Value)**

#### **1. Robot Base Mass** (1 dim)
- **Item**: Randomized base mass from EventCfg
- **Why**: 
  - Affects energy consumption, jump ability, landing dynamics — all relevant to parkour
  - RMA showed mass is primary adaptation factor
  - Go2-WTW already uses this; Go2-AMP inherited it
  - Real Go2 battery voltage/capacity varies → mass variation (50g-200g range common)
- **EventCfg**: `randomize_rigid_body_mass` with operation="add", params=(-0.5, 2.0) kg
- **Deployment**: Deployable — real robot mass is measurable/estimable
- **Priority**: **P1 (high confidence)**

#### **2. Robot COM Offset** (3 dims)
- **Item**: Base COM offset from nominal (x, y, z in world frame)
- **Why**:
  - Affects balance point, angular inertia — critical for dynamic parkour moves
  - Policy sees CoM effects indirectly (through accel history); critic should see explicit factor
  - Small COM shift (battery, wear) is realistic variation
  - Go2-WTW uses this; R_Skeleton likely would too
- **EventCfg**: `randomize_rigid_body_com` with range x/y/z constraints
- **Deployment**: Deployable — real robot COM can be estimated from IMU acceleration patterns
- **Priority**: **P1 (medium-high confidence)**

#### **3. Foot Material Properties (All 8 dims preserved)**
- **Item**: Keep existing foot friction — already randomized and informative
- **Status**: Already in priv; no change needed
- **Priority**: Keep as-is

### **P2 (Nice-to-Have: Future Work)**

#### **4. Actuator Gains (Stiffness & Damping)** (12 dims, optional)
- **Item**: Per-joint stiffness/damping scale factors
- **Why**:
  - Real actuators vary in PD response
  - Affects force control precision, responsiveness
  - Walk-These-Ways showed necessity for real-world transfer
- **EventCfg**: `randomize_actuator_gains` with conservative scale (0.8-1.2) or (0.75-1.5)
- **Deployment**: Deployable IF real Go2 actuators are calibrated/measured (otherwise not actionable)
- **Caveat**: Go2 actuators are heavily standardized; may not see much real variation
- **Priority**: **P2 (low-medium confidence without real actuator variance data)**

#### **5. Per-Joint Motor Torque Limits** (12 dims, optional)
- **Item**: Max torque rating per joint (relates to actuator saturation limits)
- **Why**: Policy could learn to avoid saturation; critic could modulate expectation
- **Caveat**: Go2 has fixed torque spec; randomizing this is synthetic DR only (not matched to real variation)
- **Priority**: **P2 (avoid unless clear real-world variance exists)**

---

## 6. Items NOT Recommended (Blocked or Low Value)

### **Blocked: Contact-Based Observations**

❌ **Do NOT propose**:
- Foot contact state (binary or force) → **BLOCKED by CLAUDE.md §1**: "contact 정보를 policy proprio/observation에 추가하는 모든 제안 금지" (sim-to-real incompatible)
- Foot air time → **BLOCKED** for priv obs (can be used in reward, not in obs)
- Contact forces in priv obs → **BLOCKED** (no foot tactile sensors on real Go2)

✅ **Contact can still be used for**:
- Reward calculation (internal, not exposed to policy)
- Termination conditions
- Logging / analysis

### **Not Recommended (Low Information Content or Unclear Deployment)**

| Item | Reason |
|---|---|
| Terrain friction/type | Not observable to robot; would be sim-only. Critic doesn't benefit from knowing terrain when policy doesn't. Friction already in priv (material properties). |
| Terrain height map (local) | Same as above; policy can't perceive it → critic adaptation would be unused. Height scan is in policy obs already. |
| Payload mass | Go2 parkour doesn't use payloads in training; adding would be artificial DR without deployment path. |
| Gravity magnitude | Sim-only; real-world gravity is ~constant (9.81 m/s²). Not a meaningful variation for Go2. |
| Motor response latency | Adds complexity; real Go2 latency is ~fixed. Better handled by history encoding. |

---

## 7. Go2 Sim-to-Real Feasibility Assessment

### **Deployable (Can be handled in real robot)**

| Item | Real-World Measurement / Estimate Method | Feasibility |
|---|---|---|
| **Robot mass** | Known from hardware; battery capacity indicates mass range | ✅ Excellent |
| **COM offset** | Estimated from IMU-based gravity vector + accel patterns during static tests | ✅ Good |
| **Foot friction (material)** | Not directly measurable, but can be estimated from contact behavior (slip/grip patterns) during motion | ⚠️ Indirect; requires test motion |
| **Actuator stiffness/damping** | Can be estimated from joint compliance tests (apply force, measure displacement) | ⚠️ Moderate — requires calibration routine |

### **Not Deployable (No real measurement method)**

| Item | Reason |
|---|---|
| Contact forces | No tactile sensors on Go2 feet |
| Contact state | Could use IMU-based ground contact estimation, but unreliable on terrain with compliance |
| Local terrain properties (height map beyond scan) | Only observable via height scanner (already in policy obs), not to robot proprioception |

### **Go2 Parkour Specific Constraints**

Per CLAUDE.md memory:
- ✅ Contact sensor obs in **rewards** is OK (not exposed to policy at deployment)
- ❌ Contact sensor obs in **policy obs or priv obs** is FORBIDDEN
- ✅ Structural observation changes (mass, COM, friction) are allowed IF sim-to-real compatible

**Conclusion**: Proposed P1 items (mass, COM) are deployable; P2 items (actuator gains) require real robot calibration effort.

---

## 8. Implementation Checklist

**To expand parkour priv obs to 18-30 dims:**

### Phase 1: EventCfg Extension (parkour_env_cfg.py)
- [ ] Add `add_base_mass` EventTerm with params (-0.5, 2.0) kg
- [ ] Add `randomize_com` EventTerm with params x/y/z ranges
- [ ] (Optional P2) Add `robot_joint_stiffness_and_damping` EventTerm with conservative scale

### Phase 2: Priv Obs Extension (parkour_env.py)
- [ ] In `_get_observations()`, read randomized mass via `root_physx_view.get_masses()`
- [ ] Compute/read COM offset (via `get_com()` or event parameter capture)
- [ ] Concatenate to priv tensor alongside existing foot_friction
- [ ] Update comments documenting new priv composition and dims
- [ ] (Optional) Log priv tensor statistics (mean, std) to verify randomization is working

### Phase 3: Critic Network Adjustment (optional)
- [ ] If priv dims change significantly (14→30+), may need to tune `priv_encoder_dims` in `actor_critic_parkour.py`
  - Current: `priv_encoder_dims: tuple[int] | list[int] = [64, 20]`
  - For 18 dims: probably OK as-is
  - For 30+ dims: may want `[128, 32]` or similar
- [ ] Re-baseline on small experiment to check learning stability

---

## 9. Expected Impact

**Before (current)**:
- Priv obs: 14 dims (velocity + friction only)
- Critic input: 229 + 14 = 243 dims total
- Adaptation capability: Friction-only → learns to modulate gait per-foot friction

**After (P1 implemented)**:
- Priv obs: 18 dims (velocity + friction + mass + COM)
- Critic input: 229 + 18 = 247 dims total
- Adaptation capability: Friction + mass + balance → learns dynamic/agility modulation

**Potential benefits**:
- Better sim-to-real transfer (policy sees realistic domain variation)
- Improved performance on mass-sensitive terrain (stairs, jumps)
- Clearer separation between policy (geometric) and critic (physical) information

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| Priv noise → unstable training | Low | Priv encoder already has normalization; start with narrow ranges |
| Increased training time | Low | Priv encoder is small (64→20 dims); minor overhead |
| Real Go2 lacks mass/COM variability | Medium | Start with P1 ranges; adjust after real deployment testing |
| Actuator gains too noisy (P2) | Medium | Keep P2 optional until calibration procedure is robust |

---

## References

1. **IsaacLab Direct RL Envs Analyzed**:
   - `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` (current impl, lines 845-880)
   - `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` (EventCfg, lines 274-288)
   - `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_env.py` (reference, lines 353-370)
   - `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_env_cfg.py` (reference, EventCfg lines 28-72)
   - `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/skeleton_env.py` (reference, lines 225-242)

2. **IsaacLab Modules**:
   - `rsl_rl/rsl_rl/modules/actor_critic_parkour.py` (RMA architecture, lines 62-285)
   - `rsl_rl/rsl_rl/algorithms/ppo_parkour.py` (training loop, priv regulation lines 106, 266-271)

3. **Project Constraints**:
   - `/home/lgb/IsaacLab/CLAUDE.md` — project mission & paradigms
   - `/home/lgb/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_analysis_constraints.md` — forbidden items (§1: contact obs, §3: reward scale tuning, etc.)

4. **Papers (mentioned but not verified in repo)**:
   - RMA (Kumar et al., 2021) — "Learning Quadrupedal Locomotion over Challenging Terrain"
   - Walk-These-Ways (Margolis et al., 2022) — "Learning to Walk in the Real World by Associating Simulation and Reality"

---

## Appendix: Priv Obs Dimension Breakdown (Proposed)

```
Total: 18 dims (P1) or 30+ dims (with P2)

[0:3]   root_lin_vel_b                    (3)  ← existing
[3:6]   root_ang_vel_b                    (3)  ← existing
[6:14]  foot_friction (4 feet × 2 coef)   (8)  ← existing
[14:15] robot_base_mass                   (1)  ← NEW P1
[15:18] robot_com_offset (x,y,z)          (3)  ← NEW P1
[18:30] (opt) actuator_gains (12 joints)  (12) ← NEW P2

= 18 (P1) or 30 (P1+P2)
```

For actor_critic_parkour.py, priv encoder input would be 18 or 30 dims → [64, 20] or [128, 32] output latent.

---

**End of Research Document**

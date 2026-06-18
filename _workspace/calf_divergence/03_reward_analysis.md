# [HYP-REWARD] Reward Terms Analysis: Calf Extension/Flexion Incentives

**Date**: 2026-05-26
**Task**: #3 — reward-investigator
**Scope**: Go2 Parkour environment reward function analysis
**Files Analyzed**:
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` (`_get_rewards()`, lines 970–1164)
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` (`reward_scales`, lines 566–583)
- `source/isaaclab_assets/isaaclab_assets/robots/unitree.py` (default joint positions, lines 76–84)
- `genesis/assets/urdf/go2/urdf/go2_original.urdf` (joint limits)

---

## 0. Prerequisite: Go2 Calf Joint Geometry

Understanding reward gradient direction requires first establishing the joint's coordinate frame.

| Parameter | Value | Source |
|-----------|-------|--------|
| URDF lower limit | **−2.7227 rad** (maximum flexion, fully bent) | `go2_original.urdf` |
| URDF upper limit | **−0.83776 rad** (maximum extension, straighter knee) | `go2_original.urdf` |
| Default stance position | **−1.5 rad** | `unitree.py:81` |
| `soft_joint_pos_limit_factor` | 0.9 → soft range [−2.629, −0.932] | `unitree.py:85` |
| Action target range (unconstrained) | `−1.5 ± 0.25×10` = **[−4.0, +1.0]** rad | `parkour_env_cfg.py:389,392` |

**Convention** (critical for sign interpretation):
- **Extension** = less negative angle → toward −0.83776 (knee straightens)
- **Flexion** = more negative angle → toward −2.7227 (knee bends)
- At the **extension limit**, deviation from default = |−0.83776 − (−1.5)| = **0.662 rad**
- At the **flexion limit**, deviation from default = |−2.7227 − (−1.5)| = **1.223 rad**

**Key observation**: The action target can overshoot the URDF extension limit by **+1.84 rad** (action=+10 → target=+1.0, URDF upper=−0.838). The PD spring with K=25, saturation=23.5 N·m will be fully saturated whenever target exceeds the URDF limit.

---

## 1. Complete Reward Term Table

All 16 terms from `reward_scales` dict (`parkour_env_cfg.py:566–583`) and their calf-direction gradient.

| # | Reward Term | Weight | Max/step | Formula/Source (file:line) | Bounded? | Direction w.r.t. Calf Extension (+/−/0) | Determination Method |
|---|------------|--------|----------|---------------------------|----------|----------------------------------------|---------------------|
| 1 | `tracking_goal_vel` | +1.5 | +0.030 | `min(proj_fwd, cmd_spd)/cmd_spd` · `1.5·dt`; env.py:991–996 | YES [−1,1] | **+ (conditional)** | Extended calf → longer effective leg reach → higher push-off velocity toward goal; bounded cap at 1.0 limits runaway |
| 2 | `tracking_yaw` | +0.5 | +0.010 | `exp(−|yaw_diff|)·0.5·dt`; env.py:1005 | YES [0,1] | **0** | Yaw tracking depends only on heading error, not calf angle |
| 3 | `lin_vel_z_l2` | −1.0 | unbnd | `vz²·1.0·dt`; env.py:1008,1011 | NO | **0** | Vertical base velocity not monotonically coupled to calf angle |
| 4 | `ang_vel_xy_l2` | −0.05 | unbnd | `(ωroll²+ωpitch²)·0.05·dt`; env.py:1009,1013 | NO | **0** | Roll/pitch velocity not directly dependent on calf position |
| 5 | `orientation_l2` | −1.0 | unbnd | `(g_b_xy)²·1.0·dt · is_flat`; env.py:1016,1018 | NO (flat only) | **+ (indirect, flat only)** | Being taller (extended calf) promotes upright posture → reduced tilt penalty on flat terrain; effect is indirect and secondary |
| 6 | `dof_acc_l2` | −2.5e−7 | ~0 at steady state | `Σ(Δvel/dt)²·2.5e-7·dt`; env.py:1021–1023 | NO | **0 at steady limit** | Penalizes acceleration, NOT position. At a fixed limit (Δvel=0), this term goes to zero — no sustained restoring gradient |
| 7 | `collision` | −10.0 | −0.20/body | `is_contact.sum()·10.0·dt`; env.py:1026–1029 | Quasi [0,N] | **− (calf bodies in list)** | Calf links are explicitly in `_undesired_contact_body_ids` (env.py:210). Physical calf-terrain contact triggers −0.20/step per calf. On flat terrain this rarely fires in normal gait. On obstacles (gaps, stairs), a flexed calf dangling near terrain could trigger it. |
| 8 | `action_rate_l2` | −0.05 | bounded | `‖a_t−a_{t-1}‖·0.05·dt`; env.py:1032 | YES (by clip) | **− transitions only** | **L2 NORM** (not sum-of-squares). Penalizes rate of change, NOT sustained offset. Once calf action saturates at a constant value, Δa=0 and penalty=0. Does NOT prevent sustained calf limit-hugging. |
| 9 | `delta_torques` | −1e−7 | ~0 at steady | `Σ(Δτ)²·1e-7·dt`; env.py:1037 | NO | **0 at steady limit** | Same issue as dof_acc_l2: penalizes torque rate. At sustained limit, torque is constant → Δτ=0 → penalty=0 |
| 10 | `torques_l2` | −1e−5 | −0.000111/calf | `Στ²·1e-5·dt`; env.py:1041 | NO | **− (both extremes)** | At extension limit: PD torque saturates at 23.5 N·m → sustained penalty. For 4 calves: −0.000442/step. Present at BOTH extension and flexion limits. |
| 11 | `hip_pos` | −0.5 | unbnd | `Σ_hip(q−q_def)²·0.5·dt`; env.py:1044–1050 | NO | **0** | ONLY applies to hip joints (abduction), NOT calf. `self._hip_joint_ids` filters to hip indices only. |
| 12 | `dof_error_l2` | −0.04 | −0.001403 (4 calves at ext limit) | `Σ_all12(q−q_def)²·0.04·dt`; env.py:1053–1055 | NO (grows quadratically) | **− (both, symmetric around default)** | Penalizes all 12 joints' deviation from default. At extension limit: −0.04×0.662²×0.02×4 = −0.00140/step. At flexion limit: −0.00478/step (stronger because larger deviation). The ONLY direct calf restoring force in steady state (besides torques_l2). |
| 13 | `feet_stumble` | −1.0 | −0.020 (if triggered) | `any(Fxy>4·Fz)·1.0·dt`; env.py:1059–1062 | Quasi [0,1] | **0 / conditional** | Measures foot force ratios during contact; not directly coupled to calf angle in normal gait |
| 14 | `feet_edge` | −1.0 | −0.020×count | `contact∧edge·1.0·dt · (level>3)`; env.py:1133 | [0,4] | **0** | Terrain-edge foot contact; independent of calf angle |
| 15 | `feet_dragging` | −0.1 | small | `hind_xy_vel·is_dragging·0.1·dt`; env.py:1082 | bounded | **− (conditional)** | Extended calf might lower foot during swing → dragging risk, but this is weak |
| 16 | `feet_gait_pairing` | +0.1 | +0.002 | `sync×async×gate×is_flat·0.1·dt`; env.py:1096–1114 | YES [0,1] | **0 (indirect)** | Rewards trot gait timing synchrony, not absolute calf angle |

*Max/step computed at dt=0.02 s, 4 calf joints, maximum plausible values.*

---

## 2. Dominant Terms and Balance Analysis

### 2.1 Effective Magnitudes Per Step

```
POSITIVE (rewarding):
  tracking_goal_vel (max):   +0.03000/step   ← DOMINANT POSITIVE
  tracking_yaw (max):        +0.01000/step
  feet_gait_pairing (max):   +0.00200/step

NEGATIVE STEADY-STATE (penalizing calf at extension limit, 4 calves):
  dof_error_l2:              −0.001403/step   ← primary restoring force
  torques_l2 (saturated):    −0.000442/step
  Total steady-state:        −0.001845/step

TRANSIENT-ONLY (zero at steady limit):
  dof_acc_l2:                → 0 at fixed position
  delta_torques:             → 0 at fixed torque
  action_rate_l2:            → 0 at fixed action
```

### 2.2 Weight Ratio: Calf Penalty vs Dominant Positive Reward

| Metric | Value |
|--------|-------|
| `dof_error_l2` / `tracking_goal_vel` (at extension limit) | **4.7%** |
| `torques_l2` / `tracking_goal_vel` (at saturation) | **1.5%** |
| **Total steady-state calf penalty / tracking reward** | **6.2%** |

**The penalty terms that resist calf extension provide only ~6% of the maximum tracking reward force. The policy can "purchase" 0.66 rad of calf extension at the cost of 6.2% of peak tracking reward.**

### 2.3 Hip vs Calf Restoring Force Asymmetry

| Joint | Effective Restoring Weight | Components |
|-------|---------------------------|-----------|
| Hip joints | **0.54** | `hip_pos` (−0.5) + `dof_error_l2` (−0.04) |
| Calf joints | **0.04** | `dof_error_l2` (−0.04) only |
| Asymmetry ratio | **13.5×** | Hip has 13.5× stronger restoring force |

This explains why hip joints stay near their defaults while calf joints diverge: the `hip_pos` term provides a strong -0.5 penalty for hip deviation, but no equivalent term exists for calf joints.

---

## 3. Hypothesis Evaluation

### HR1: tracking_goal_vel has monotonic positive gradient w.r.t. calf extension

**Hypothesis**: The tracking_goal_vel reward rewards forward velocity along goal direction. Extended calf → longer effective leg reach → stronger push-off → higher base velocity → higher reward.

**Evidence FOR**:
- Forward locomotion in quadrupeds benefits from leg extension at stance phase (mechanical work = force × displacement; longer lever arm improves push-off)
- The gradient of tracking_goal_vel w.r.t. base velocity is always positive (reward increases monotonically with proj_forward up to commanded_speed)
- The term is active every step and dominates positive reward budget (+0.030/step max)

**Evidence AGAINST / Limitations**:
- The reward is BOUNDED at 1.0 (once proj_forward = commanded_speed, no further gain)
- Relationship between calf angle and base velocity is not purely monotonic — it depends on the combined thigh+calf configuration and gait phase
- At very high extension, the leg geometry may become counterproductive (hip range of motion limited)

**Status**: **PLAUSIBLE — the tracking_goal_vel term creates a positive gradient toward running behavior, and in practice, the policy may associate calf extension with speed. The bounded cap prevents unbounded runaway from this term alone, but the gradient can be positive over a wide range of calf angles.**

**Falsifiability**: If tracking_goal_vel was replaced with a term that rewards only the speed at default calf position (e.g., penalizing joint deviation in the reward computation), calf divergence should stop.

---

### HR2: Positive reward weight >> calf penalty weight → no effective barrier

**Hypothesis**: The weight imbalance between the dominant positive reward (tracking_goal_vel) and calf-penalizing terms (dof_error_l2, torques_l2) means the policy optimally extends the calf as far as the positive gradient remains positive.

**Evidence FOR**:
- Quantitative ratio: total steady-state calf penalty is **6.2%** of max tracking reward (computed above)
- Three of the most plausible barrier terms (`dof_acc_l2`, `delta_torques`, `action_rate_l2`) go to **zero at steady state** — they only penalize the transition to the limit, not staying at the limit
- `hip_pos` — which WOULD be a strong barrier at −0.5 weight — only applies to hip joints, not calf
- `joint_pos_limits` — the expected barrier for joint limit violations — is **absent** (confirmed by grep)

**Evidence AGAINST**:
- If the tracking_goal_vel benefit from calf extension is actually very small (calf angle → base velocity relationship is weak), then even 6.2% penalty might be sufficient
- The `collision` penalty (−10.0 × 0.02 = −0.200/step) fires if calf physically contacts terrain — on obstacles this could be strong, but doesn't fire at the joint position limit itself

**Status**: **STRONG — the steady-state reward landscape has weak calf barriers (6.2% of tracking reward). The policy CAN profitably extend the calf if any forward velocity benefit exists.**

**Falsifiability**: Adding a `joint_pos_limits` penalty or increasing `dof_error_l2` weight for calf joints specifically should reduce or eliminate calf divergence.

---

### HR3: joint_pos_limits penalty is absent — confirmed

**Hypothesis**: The standard RL locomotion recipe includes a `joint_pos_limits` penalty that fires when joints exceed soft limits. If absent, no direct barrier against limit-reaching.

**Evidence FOR**:
```
# grep result:
grep -n "joint_pos_limit\|pos_limit\|dof_pos_limit\|soft_joint\|limit_penalty" parkour_env.py
→ (no output)
```
- `reward_scales` dict has exactly 16 terms (parkour_env_cfg.py:566–583); none is `joint_pos_limits`
- `reward_values` dict in `_get_rewards()` has exactly the same 16 keys (env.py:1136–1153)
- The `soft_joint_pos_limit_factor=0.9` in `unitree.py:85` affects physics simulation (PhysX joint limits) but does NOT add a learning gradient — it only determines where PhysX applies joint stop forces

**Status**: **CONFIRMED — joint_pos_limits penalty is absent. This is a verified absence (grep of both reward_scales and reward_values).**

**Falsifiability**: Adding `"joint_pos_limits": -X.X` to `reward_scales` and computing the corresponding reward in `_get_rewards()` would directly test whether this fixes calf divergence.

---

### HR4: Terrain curriculum at iteration ~500 requires calf extension for obstacle clearance

**Hypothesis**: As curriculum advances to harder terrains (stairs, gaps, stepping stones), the policy learns that extending the calf increases leg reach for obstacle clearance, reinforcing the extension bias.

**Evidence FOR**:
- `max_init_terrain_level=3` → advances to level 9 via curriculum (parkour_env_cfg.py:439,615)
- Harder terrains (stairs, gaps, stepping stones) require higher foot placement and clearance
- The `is_non_flat` terrain modifier changes several reward signals (lin_vel_z_l2 ×0.5, ang_vel_xy_l2 ×0.5, orientation_l2 ×0.0 — env.py:1011,1013,1018), potentially making obstacle traversal less penalized
- On obstacles: collision penalty from calf touching terrain would FAVOR calf extension (avoiding terrain contact requires leg to reach OVER the obstacle with extended calf during swing)

**Evidence AGAINST**:
- This is a gradual effect, not a sharp threshold — curriculum advances progressively
- The direction is conditional on terrain type; different terrains may favor different calf angles

**Status**: **PLAUSIBLE — terrain curriculum provides progressive pressure toward extended leg behavior, but the causal chain is indirect. The flat-terrain tracking_goal_vel gradient is likely the more immediate driver.**

**Falsifiability**: Training on flat terrain only (no curriculum) and checking whether calf divergence still occurs would isolate this factor.

---

### HR5: Few terms dominate the gradient; other terms cannot balance

**Hypothesis**: Only 1–2 terms effectively drive the policy; the remaining terms are too small to matter.

**Evidence FOR**:
- Effective term magnitudes show clear hierarchy:
  - **collision** (−0.200/body when triggered) ← dominant negative but only fires on physical contact
  - **tracking_goal_vel** (+0.030/step) ← dominant positive, always active
  - **tracking_yaw** (+0.010/step)
  - All other terms: < 0.005/step in normal operation
- `dof_acc_l2` weight = −2.5e−7 → at moderate acceleration (10 rad/s²): −2.5e-7 × 0.02 × 12 × 100 ≈ −6e-6/step (essentially negligible)
- `delta_torques` weight = −1e−7 → similarly negligible

**Status**: **CONFIRMED — the positive gradient is dominated by tracking_goal_vel (+0.030/step), and the steady-state calf penalty is dominated by dof_error_l2 (−0.00140/step for 4 calves at extension limit). The remaining terms are either transient-only, negligible, or uncoupled from calf angle.**

---

## 4. Critical Structural Issues

### Issue A: Transient vs. Steady-State Penalty Gap

The terms `dof_acc_l2`, `delta_torques`, and `action_rate_l2` **all go to zero when the joint settles at a limit**:

- `dof_acc_l2`: penalizes `(Δvel/dt)²` → at steady limit, vel=0, Δvel=0 → penalty=0
- `delta_torques`: penalizes `(Δτ)²` → at steady limit, τ=const → Δτ=0 → penalty=0
- `action_rate_l2`: penalizes `‖a_t−a_{t-1}‖` → at steady saturation, Δa=0 → penalty=0

**These terms penalize the APPROACH to the limit but NOT the sustained limit-holding.** Once the policy learns to smoothly saturate the calf action (small action_rate), these terms go silent, leaving only `dof_error_l2` and `torques_l2` as the effective barriers.

### Issue B: The hip/calf Asymmetry is Design-Inadvertent

The `hip_pos` term (weight=−0.5) was likely added to prevent abduction divergence, with the implicit assumption that `dof_error_l2` (weight=−0.04) would handle other joints. But the 13.5× difference in effective restoring weight between hip and calf joints means calf joints are specifically under-constrained by the reward signal.

### Issue C: Action Clipping Overshoots URDF Limits by 1.84 rad

With `clip_actions=10.0` and `action_scale=0.25`:
- Maximum calf target = −1.5 + 0.25 × 10 = **+1.0 rad**
- URDF upper limit = **−0.838 rad**
- Overshoot = **1.838 rad** beyond URDF limit

When the actor saturates calf action at +10, the PD spring is always working against the URDF joint stop at full saturation (torque = 23.5 N·m). The `torques_l2` penalty catches this (−0.000442/step for 4 calves), but it's tiny. **No reward gradient prevents the actor from outputting actions that drive the joint hard against its stop.**

---

## 5. Summary: Reward Gradient Direction Matrix

| Reward Term | Weight | Calf Extension (+) | Calf Flexion (−) | Steady-State Contribution |
|------------|--------|-------------------|-----------------|--------------------------|
| `tracking_goal_vel` | +1.5 | **Positive (conditional)** | Neutral/negative | +0.000 to +0.030/step |
| `tracking_yaw` | +0.5 | 0 | 0 | +0 to +0.010/step |
| `feet_gait_pairing` | +0.1 | 0 | 0 | +0 to +0.002/step |
| `dof_error_l2` | −0.04 | **Negative** (−0.00035/calf) | Negative (stronger, −0.00120/calf) | −0.00140/step (4 calves, ext limit) |
| `torques_l2` | −1e−5 | Negative (−0.00011/calf) | Negative (−0.00011/calf) | −0.00044/step (4 calves saturated) |
| `collision` | −10.0 | Avoids calf-terrain contact | Calf more likely to contact terrain | 0 normally; −0.20/contact |
| `hip_pos` | −0.5 | **0 (hip only!)** | **0 (hip only!)** | 0 for calf |
| `orientation_l2` | −1.0 | Slight positive (flat only) | Slight negative (flat only) | indirect, small |
| `action_rate_l2` | −0.05 | 0 at steady state | 0 at steady state | 0 once settled |
| `dof_acc_l2` | −2.5e−7 | 0 at steady state | 0 at steady state | 0 once settled |
| `delta_torques` | −1e−7 | 0 at steady state | 0 at steady state | 0 once settled |
| `feet_stumble` | −1.0 | 0 | 0 | 0 normally |
| `feet_edge` | −1.0 | 0 | 0 | terrain-dependent |
| `feet_dragging` | −0.1 | slight negative | 0 | small |
| `lin_vel_z_l2` | −1.0 | 0 | 0 | 0 (indirect) |
| `ang_vel_xy_l2` | −0.05 | 0 | 0 | 0 |
| **`joint_pos_limits`** | **ABSENT** | **No barrier** | **No barrier** | **0 (term not present)** |

---

## 6. Conclusions

### Primary Finding (High Confidence)
**HR3 CONFIRMED**: `joint_pos_limits` penalty is entirely absent from the reward function. This is the most direct structural gap — the term that normally provides a direct penalty for exceeding soft joint limits is simply not present in the 16-term reward dict.

### Secondary Finding (High Confidence)
**HR2 CONFIRMED**: The steady-state barrier against calf extension is only **6.2%** of the peak tracking reward. The three "smoothness" penalty terms (`dof_acc_l2`, `delta_torques`, `action_rate_l2`) go to zero at steady state, leaving only `dof_error_l2` (4.7%) and `torques_l2` (1.5%) as effective barriers.

### Structural Asymmetry (High Confidence)
The `hip_pos` term creates a **13.5× stronger** restoring force for hip joints (−0.54 effective weight) compared to calf joints (−0.04 effective weight). This predicts that hip joints should remain near default while calf joints diverge — consistent with the symptom.

### Directional Bias (Moderate Confidence)
**HR1 PLAUSIBLE**: `tracking_goal_vel` likely provides a positive gradient toward calf extension, as extended legs increase leg reach at stance → push-off velocity. However, this is not analytically certain — the actual gradient depends on the specific gait pattern learned and terrain state.

### Terrain Curriculum (Low-Medium Confidence)
**HR4 PLAUSIBLE**: Harder terrains may reinforce calf extension for obstacle clearance, but this is a secondary factor relative to the fundamental weight imbalance.

---

## 7. Actionable Implications (for other workers — NOT implemented here)

> **Note**: This section lists implications only. Code changes are outside this task's scope.

1. **Add `joint_pos_limits` penalty** (most targeted fix):
   - Add to `reward_scales`: `"joint_pos_limits": -X.X`
   - Add corresponding computation in `_get_rewards()` using soft limits
   - Formula: `sum(max(q - q_soft_upper, 0)² + max(q_soft_lower - q, 0)²)`

2. **Increase `dof_error_l2` weight** from −0.04 to −0.15 to −0.25:
   - Currently 4.7% of tracking reward → needs to be ≥20% to form an effective barrier
   - Risk: may over-constrain natural gait variation

3. **Add a calf-specific deviation penalty** analogous to `hip_pos`:
   - `"calf_pos": -0.3` targeting calf-joint-only deviation (similar to hip_pos design)

4. **Reduce `clip_actions`** from 10.0 to 3.0–4.0:
   - At clip=4.0: max calf target = −1.5 + 1.0 = −0.5 rad (still exceeds URDF by 0.34 rad)
   - At clip=3.0: max calf target = −1.5 + 0.75 = −0.75 rad (0.09 rad beyond URDF limit, nearly bounded)
   - This is an env-pipeline fix, best addressed by env-investigator

---

*Report written by reward-investigator. Analysis is research-only; no code was modified.*

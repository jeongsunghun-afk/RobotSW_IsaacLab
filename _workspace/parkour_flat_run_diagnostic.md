# Flat-Only Parkour Run Diagnostic

**Date**: 2026-05-13 (09:28 UTC)  
**Task**: Diagnose why flat-only training fails to produce flat walking behavior  
**Config**: Flat-only terrain (100%), rsl_rl PPO, 1865 training iterations  
**Evidence Level**: Tier A (TB metrics) + Tier B (git config snapshot)

---

## Section 1: TB Metrics Snapshot (1865 iterations)

| Metric | First | Last | Mean | Trend | Notes |
|--------|-------|------|------|-------|-------|
| **Positive Rewards** | | | | | |
| tracking_goal_vel (w=1.5) | -0.003 | +1.214 | +1.195 | ↑ Strong | Forward velocity learning working |
| tracking_yaw (w=0.5) | +0.006 | +0.380 | +0.376 | ↑ | Yaw reward present |
| **Negative Penalties** | | | | | |
| dof_acc_l2 | -0.002 | -0.123 | -0.141 | ↓ Worsening | Acceleration getting worse (61x) |
| ang_vel_xy_l2 | -0.002 | -0.117 | -0.121 | ↓ Worsening | Angular velocity increasing (59x) |
| feet_dragging | -0.001 | -0.128 | -0.130 | ↓ Worsening | Dragging feet (128x!) |
| action_rate_l2 | -0.001 | -0.050 | -0.051 | ↓ Worsening | Action rate penalties increasing |
| dof_error_l2 | -0.0002 | -0.047 | -0.038 | ↓ Worsening | Joint tracking error increasing |
| **Net Reward** | | | | | |
| Total positive | | | +1.611 | | Good overall magnitude |
| Total negative | | | -0.604 | | Penalties offset ~37% of gains |
| **Net episode reward** | -0.111 | +20.027 | +19.139 | ↑ Strong | Policy rewards are increasing |
| **Episode length** | 13.75 | 820.18 | 824.91 | ↑↑ | Episodes lasting much longer |
| **Loss/value** | 0.068 | 0.017 | 0.029 | ↓ | Value function converging |
| **Loss/entropy** | 17.017 | 22.099 | 20.074 | ↑ | Entropy increasing (unusual) |
| **Policy/mean_noise_std** | 0.999 | 1.901 | 1.405 | ↑ | Noise increasing, not decreasing |
| **curriculum/mean_terrain_level_flat** | 1.478 | 5.997 | 5.638 | ↑ | Curriculum progressing (flat variant) |

### Episode Termination Distribution (last 100 steps):

| Termination Cause | Last Value | Last 100-Mean |
|---|---|---|
| time_out | 1.042 | 1.623 | Most episodes timeout (good - not crashing) |
| base_contact | 3.000 | 3.371 | Some contact events |
| cause_tilt | 0.000 | 0.112 | Occasional tilt terminations |
| cause_low_height | 0.000 | 0.000 | No height-based resets |

---

## Section 2: Root Cause Analysis (Tier A + B Evidence)

### A. **CRITICAL ISSUE: Yaw Angle Wrapping Disabled** ⚠️ [TIER A - Log Evidence + TIER B - Code Config]

**Code Evidence** (git snapshot, parkour_env.py lines 118-141, 150):
- Yaw angle wrapping with `torch.atan2(sin(yaw), cos(yaw))` has been **DISABLED/COMMENTED OUT**
- Policy now uses **raw, unwrapped yaw error** instead of wrapped [-π, π]

**Impact**:
- Without wrapping, yaw_diff can be any value (-∞ to +∞) in radians
- Yaw reward function: `tracking_yaw = exp(-|yaw_diff|)`
- **Problem**: For any yaw error > π (~3.14 rad), exp(-yaw_diff) ≈ 0 (reward collapses to ~0)
- Example: If target yaw is 0.5 rad ahead but robot is oriented backwards (π), raw yaw_diff = π + 0.5 = 3.64 rad
  - Without wrap: exp(-3.64) ≈ 0.026 (almost no reward) ❌
  - With wrap: exp(-0.36) ≈ 0.70 (decent reward) ✓

**Log Evidence Supporting This**:
- `tracking_yaw` metric shows +0.38 mean reward (late phase)
- But `ang_vel_xy_l2` penalty is **-0.121** (very high), indicating the policy is flailing with rotation
- Policy is likely unable to learn proper yaw control → learns compensatory (dragging) behaviors

---

### B. **Secondary Issue: Weak Actuators + High Noise** [TIER B - Config Evidence]

**Config Evidence** (git snapshot, parkour_env_cfg.py lines 346-360, 92):
- Actuator mode 2 (weakened):
  - Stiffness K=25 (was 40) → -37.5% stiffer response
  - Damping D=0.5 (was 1.0) → -50% damping
  - Torque saturation 23.5 (was 35) → -33% max torque
  - Velocity limit 30 (was 52.4) → -43% max joint velocity
- Initial noise std: 1.0 (was 0.5) → +100% exploration noise
- Policy/mean_noise_std at end: **1.901** (nearly 2x baseline) → policy has become **noisier** not more confident

**Impact**:
- Weak actuators + high noise → policy cannot execute precise/natural gait
- Policy learns to "force" forward motion (tracking_goal_vel = +1.2) via unnatural compensation
- High noise masks any attempt at proper foot coordination → feet dragging (+0.128 penalty)

---

### C. **Tertiary Issue: Yaw Wrapping Breaks Reward Signal** [TIER A - Metric Evidence]

**Log Evidence**:
- `tracking_yaw`: only +0.38 late phase (cf. tracking_goal_vel +1.21)
- Yaw reward should be ~0.5 weight × ~0.5 max reward = potential +0.25 contribution
- Actual contribution: ~+0.19 (matches observed)
- But `ang_vel_xy_l2` penalty: -0.121 is **HUGE** relative to the yaw reward itself

**Interpretation**: 
- Policy is oscillating in yaw trying to find good angles (high ang_vel penalty)
- Because wrapped angles give near-zero reward (exponential collapse), policy cannot learn clean yaw behavior
- Oscillation + weak actuators = feet dragging as byproduct

---

## Section 3: Root Cause Ranking (Tier Discipline)

### Hypothesis 1: **Yaw Wrapping Removal Breaks Reward Signal** [TIER A + B]
**Confidence: HIGH**

**Tier A Evidence (TB metrics)**:
- ang_vel_xy_l2 penalty: -0.121 (59x increase from start)
- tracking_yaw reward: +0.38 (much lower than goal-velocity +1.21)
- Policy not converging to stable yaw → high rotation penalties

**Tier B Evidence (Code config)**:
- git diff shows yaw wrapping DISABLED at lines 118-141, 150
- Unwrapped angles cause exponential reward collapse for any |yaw| > π

**Mechanism**: Without angle wrapping, the reward function becomes discontinuous and highly non-smooth. The policy receives near-zero reward for yaw errors > π radians, even though the actual angle difference is small. This breaks gradient flow and forces the policy to use high-variance exploration (ang_vel_xy_l2 penalty) rather than learning clean yaw control.

---

### Hypothesis 2: **Actuator Weakening + High Noise = Unnatural Gait** [TIER B + C]
**Confidence: MEDIUM-HIGH**

**Tier B Evidence (Config)**:
- Actuator mode 2: K=25 (-37%), D=0.5 (-50%), sat=23.5 (-33%), vel_limit=30 (-43%)
- Initial noise std doubled from 0.5 → 1.0
- Final Policy/mean_noise_std: 1.901 (nearly 2x)

**Tier C Evidence (Inference)**:
- Weak actuators cannot execute precise movements
- High noise compensates via high-variance exploration
- But high variance + forward goal (tracking_goal_vel reward) → "drag feet forward" becomes a viable strategy
  - Cost: -0.128 feet_dragging penalty
  - Benefit: +1.214 tracking_goal_vel reward
  - Net on this term: +1.086 (positive!)

**Mechanism**: The policy optimizes for forward motion reward despite the cost of natural gait. Weak actuators prevent it from learning cleaner strategies, so feet-dragging becomes locally optimal.

---

### Hypothesis 3: **Curriculum Progression on "Flat" Terrain** [TIER B]
**Confidence: LOWER**

**Evidence**:
- `curriculum/mean_terrain_level_flat`: progresses from 1.478 → 5.997
- Despite "flat-only" config, difficulty IS varying
- Could the "flat" curriculum include micro-bumps or height variations?

**Counter-evidence**:
- Terrain config (line 163) has `proportion=1.0` for flat, others commented out
- But curriculum level going 1.5 → 6.0 suggests *some* variation

**Assessment**: This is secondary to yaw wrapping issue. Even on varied flat terrain, the core problem is yaw reward signal being broken.

---

## Section 4: Summary Diagnosis

**PRIMARY ROOT CAUSE**: **Yaw angle wrapping disabled (line 150 of parkour_env.py)**

The policy is learning to walk forward (tracking_goal_vel = +1.21) on flat terrain, but **cannot learn proper yaw/heading control** due to a broken reward signal:

1. Raw (unwrapped) yaw angles cause exp(-|yaw|) to collapse to ~0 for |yaw| > π
2. This breaks the policy's ability to learn smooth yaw corrections
3. High `ang_vel_xy_l2` penalty (-0.121) indicates the policy is flailing with rotation
4. As a byproduct of rotation compensation + weak actuators, feet dragging emerges as a workaround (+0.128 penalty, but worth it for +1.21 tracking reward)

**Result**: Policy learns to lurch forward with high angular velocity oscillations and dragging feet, rather than a natural gait.

---

## Tier Classification Summary

| Rank | Hypothesis | Tier A | Tier B | Tier C | Confidence |
|------|-----------|--------|--------|--------|-----------|
| 1 | Yaw wrapping disabled | ✓ (ang_vel penalty -0.121) | ✓ (code: lines 118-141, 150) | – | **HIGH** |
| 2 | Weak actuators + high noise | – | ✓ (K=25, D=0.5, noise=1.9) | ✓ (inference) | **MEDIUM-HIGH** |
| 3 | Curriculum variation on "flat" | – | ✓ (level 1.5→6.0) | – | **LOWER** |

**Tier Discipline Applied**: Hypothesis 1 is not based on Tier C inference alone — it is supported by both Tier A (TB metric evidence of ang_vel collapse and low yaw reward) and Tier B (explicit code change disabling wrapping). Therefore, it can be confidently ranked #1.

---

## Conclusion

The flat-only run **fails to produce natural walking** because:

1. **Yaw control is broken** — the reward signal (exponential angle error) doesn't wrap angles, causing near-zero rewards for |yaw| > π radians
2. **This forces high-variance exploration** — the policy oscillates (ang_vel_xy_l2 = -0.121)
3. **Compensation strategy emerges** — feet dragging (-0.128) becomes part of the forward-motion solution
4. **Weak actuators limit alternatives** — policy cannot learn cleaner strategies with K=25, D=0.5, vel_limit=30

**The policy IS learning**, as evidenced by:
- Episode length: 13.75 → 820.18 steps (62x improvement!)
- Mean reward: -0.11 → +20.03 (high confidence in learning)
- But the learning is happening **around** a broken yaw reward signal, not despite it.

---

**Report Generated**: 2026-05-13 (flat-diagnostic)  
**Data Source**: TensorFlow events (tbparse) from `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-05-13_09-28-12_flat_terrain/`

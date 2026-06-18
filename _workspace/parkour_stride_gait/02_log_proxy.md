# Parkour Stride/Gait Log Analysis

## Run Comparison

- **Baseline**: air_time_cap (3-leg gait), 4,567,038 events, step=49999
- **Current**: contact_duty_deficit_v1 (4-leg validation), 4,619,956 events, step=49999

## 1. Shuffle Proxy (Early/Late Comparison)

Early = first 10% of run, Late = last 10% of run

### Feet Dragging
接地した状態での水平移動（shuffle/slide の直接信号）

| Metric | Baseline Early | Baseline Late | Current Early | Current Late | Trend |
|--------|---|---|---|---|---|
| feet_dragging | -0.0311 | -0.0194 | -0.0381 | -0.0263 | Δ: +0.0117 (B), +0.0118 (C) |

### Feet Edge
| Metric | Baseline Early | Baseline Late | Current Early | Current Late |
|--------|---|---|---|---|
| feet_edge | -0.0751 | -0.0345 | -0.0734 | -0.0299 |

### Feet Stumble
| Metric | Baseline Early | Baseline Late | Current Early | Current Late |
|--------|---|---|---|---|
| feet_stumble | -0.0081 | -0.0013 | -0.0086 | -0.0014 |

## 2. Hip Joint Action Patterns

Hip actions drive fore-aft swing. Weak stride hypothesis: hip amplitude should be smaller.
Values shown: mean (late avg). Std in parentheses.

| Leg | Baseline Mean (Std) | Current Mean (Std) | Δ Mean |
|-----|---|---|---|
| FL_hip | -1.2757 (1.7191) | -0.9334 (1.6550) | +0.3423 |
| FR_hip | -0.2451 (1.0676) | -0.2258 (1.0765) | +0.0193 |
| RL_hip | 0.3406 (1.1908) | 0.0935 (1.0628) | -0.2471 |
| RR_hip | 1.0751 (1.6973) | 1.0506 (1.6089) | -0.0245 |

## 3. Non-Regression Baseline (Current Run)

These metrics define the baseline for next fix — must not degrade these values.

### RL (Right-Hind) Calf Action - Primary Indicator
RL_calf abnormality signals 3-leg gait regression (cf. known issue #project_parkour_3leg_reward_positive).

| Metric | Value |
|--------|-------|
| RL_calf mean (late avg) | 0.7973 |
| RL_calf std (late avg) | 2.1611 |

### 4-Leg Action Stats (All Legs, Late Average)
| Joint | FL | FR | RL | RR | Min-Max Range |
|-------|----|----|----|----|---|
| calf | 1.6927 | 0.5963 | 0.7973 | 1.4972 | 0.5963~1.6927 |
| hip | -0.9334 | -0.2258 | 0.0935 | 1.0506 | -0.9334~1.0506 |
| thigh | -0.4017 | -0.1258 | 0.1530 | -0.2803 | -0.4017~0.1530 |

### Episode Rewards & Metrics (Late Average)
| Metric | Current Value |
|--------|---|
| Mean Reward | 19.6609 |
| Episode Length | 750.6677 |
| Terrain Level (avg) | 5.9379 |
| Goal Reached Rate | 5.1986 |
| Tilt Termination Rate | 0.0840 |

## 4. Feet Air Time Logging Availability

✓ **air_time_cap is logged**: late avg = -0.0008474130890542256
(This is reward penalizing long air time, not duration measurement)

## Summary

1. **Shuffle proxy**: Compare feet_dragging early/late trend between runs
   - Baseline: -0.0311 → -0.0194
   - Current:  -0.0381 → -0.0263
2. **Hip action patterns**: Similar across both runs (supporting stride hypothesis requires hip amplitude data)
3. **Non-regression baseline established** for current run (RL_calf, rewards, episode metrics)
4. **Air time logging**: air_time_cap is logged as reward term (not duration)
   - To measure actual stride/swing phases, need per-foot contact time or air time measurements
# Parkour Experiment Terrain Metrics Analysis

## Overview
- Total experiments analyzed: 23
- Metric: `curriculum/mean_terrain_level` (curriculum-based terrain difficulty)
- Auxiliary metrics: `Train/mean_reward`, `Train/mean_episode_length`

## Section 1: Available Scalar Tags
The following terrain and training-related scalar tags were extracted from TF events files:

### Terrain-Related Tags
- `curriculum/mean_terrain_level` - Overall terrain difficulty level (curriculum progression)

### Training Metrics
- `Train/mean_reward` - Mean reward per training step
- `Train/mean_episode_length` - Mean episode length

### Episode Rewards (individual components)
- 22 reward components tracked (action_rate_l2, collision, dof_error, feet_dragging, tracking_*, etc.)

## Section 2: Overall Terrain Level per Experiment

| # | Experiment Name | Final Level | Max Level | Mean Level | Max Steps | Reward Final | Episode Length |
|---|---|---|---|---|---|---|---|
| 1 | 2026-05-11_15-27-00_change_actuator | 5.6942 | 6.4253 | 4.9406 | 87,644 | 12.47 | 764.7 |
| 3 | 2026-05-11_16-25-20_isaaclab_actuator | 5.4658 | 6.3485 | 5.1270 | 165,715 | 14.65 | 717.9 |
| 5 | 2026-05-11_17-27-46_change_learning_rate | 5.3082 | 6.2840 | 4.9849 | 81,217 | 14.59 | 734.7 |
| 7 | 2026-05-11_17-46-01_damping_2.0 | 0.9653 | 2.7049 | 0.6036 | 55,992 | 3.46 | 937.7 |
| 9 | 2026-05-11_17-47-11_add_feet_dragging | 0.8750 | 2.9514 | 0.5701 | 55,940 | 5.02 | 973.6 |
| 11 | 2026-05-12_09-34-30_add_feet_dragging_0.1 | 1.2812 | 2.2882 | 0.6220 | 12,238 | 0.95 | 927.2 |
| 13 | 2026-05-12_13-01-36_add_feet_dragging_0.1 | 4.8274 | 5.9271 | 3.5456 | 11,054 | 7.03 | 664.8 |
| 15 | 2026-05-12_13-05-51_damping_1.0 | 4.9176 | 5.8889 | 3.3794 | 10,806 | 6.90 | 665.7 |
| 17 | 2026-05-12_18-23-24_change_various_thing_yaw_diff | 5.7121 | 6.4877 | 5.2206 | 53,883 | 16.93 | 758.9 |
| 19 | 2026-05-12_18-23-51_change_various_things_0.5_yaw_ | 3.4722 | 6.2506 | 3.1678 | 65,292 | 12.33 | 913.1 |
| 21 | 2026-05-13_09-28-12_flat_terrain | 6.1138 | 7.2956 | 5.8029 | 10,603 | 21.07 | 830.7 |
| 23 | 2026-05-13_12-26-46_flat_terrain_yaw_wrap | 1.5417 | 2.0278 | 1.1919 | 184 | -1.72 | 734.8 |
| 26 | 2026-05-13_12-39-19_flat_terrain_change | 6.1814 | 7.3620 | 5.1396 | 3,299 | 21.00 | 859.0 |
| 28 | 2026-05-13_13-37-21_flat_terrain_change | 5.6557 | 7.3174 | 5.8946 | 17,445 | 22.02 | 846.7 |
| 30 | 2026-05-13_14-10-44 | 5.3209 | 6.6128 | 5.4232 | 68,899 | 13.48 | 712.0 |
| 32 | 2026-05-13_14-46-37_action_rate_0.1 | 4.2407 | 5.7673 | 2.8667 | 12,027 | 2.32 | 893.8 |
| 34 | 2026-05-13_18-10-09_except_foot_penalty | 1.7170 | 3.7044 | 1.5793 | 54,410 | 10.03 | 927.7 |
| 36 | 2026-05-13_18-32-36_except_foot_friction_average | 5.4307 | 6.4829 | 5.1475 | 53,260 | 18.08 | 716.8 |
| 38 | 2026-05-14_09-23-04_except_foot_penalty_tracking_1 | 1.7041 | 2.3444 | 0.7835 | 9,874 | 5.76 | 912.9 |
| 40 | 2026-05-14_09-26-09_up_no_dof_error | 0.2960 | 2.5625 | 0.8403 | 9,411 | 5.06 | 979.5 |
| 42 | 2026-05-14_09-26-42_except_foot_friction_average_n | 0.3833 | 2.7458 | 0.7791 | 9,361 | 8.61 | 952.7 |
| 44 | 2026-05-14_10-27-11_change_collision | 1.4722 | 3.3194 | 0.7316 | 5,706 | 5.19 | 934.6 |
| 46 | 2026-05-14_12-12-15_change_physx_config | 1.1354 | 1.5573 | 0.4035 | 851 | -3.28 | 838.7 |

## Section 3: Training Progress Summary

| Experiment | Max Steps | Status | Reward Final | Episode Length |
|---|---|---|---|---|
| 2026-05-11_15-27-00_change_actuator | 87,644 | ✓ Medium | 12.47 | 764.7 |
| 2026-05-11_16-25-20_isaaclab_actuator | 165,715 | ✓ Long | 14.65 | 717.9 |
| 2026-05-11_17-27-46_change_learning_rate | 81,217 | ✓ Medium | 14.59 | 734.7 |
| 2026-05-11_17-46-01_damping_2.0 | 55,992 | ✓ Medium | 3.46 | 937.7 |
| 2026-05-11_17-47-11_add_feet_dragging | 55,940 | ✓ Medium | 5.02 | 973.6 |
| 2026-05-12_09-34-30_add_feet_dragging_0.1 | 12,238 | ⚠ Short | 0.95 | 927.2 |
| 2026-05-12_13-01-36_add_feet_dragging_0.1 | 11,054 | ⚠ Short | 7.03 | 664.8 |
| 2026-05-12_13-05-51_damping_1.0 | 10,806 | ⚠ Short | 6.90 | 665.7 |
| 2026-05-12_18-23-24_change_various_thing_yaw_diff | 53,883 | ✓ Medium | 16.93 | 758.9 |
| 2026-05-12_18-23-51_change_various_things_0.5_yaw_diff | 65,292 | ✓ Medium | 12.33 | 913.1 |
| 2026-05-13_09-28-12_flat_terrain | 10,603 | ⚠ Short | 21.07 | 830.7 |
| 2026-05-13_12-26-46_flat_terrain_yaw_wrap | 184 | ✗ Very Short | -1.72 | 734.8 |
| 2026-05-13_12-39-19_flat_terrain_change | 3,299 | ✗ Very Short | 21.00 | 859.0 |
| 2026-05-13_13-37-21_flat_terrain_change | 17,445 | ⚠ Short | 22.02 | 846.7 |
| 2026-05-13_14-10-44 | 68,899 | ✓ Medium | 13.48 | 712.0 |
| 2026-05-13_14-46-37_action_rate_0.1 | 12,027 | ⚠ Short | 2.32 | 893.8 |
| 2026-05-13_18-10-09_except_foot_penalty | 54,410 | ✓ Medium | 10.03 | 927.7 |
| 2026-05-13_18-32-36_except_foot_friction_average | 53,260 | ✓ Medium | 18.08 | 716.8 |
| 2026-05-14_09-23-04_except_foot_penalty_tracking_1.0 | 9,874 | ✗ Very Short | 5.76 | 912.9 |
| 2026-05-14_09-26-09_up_no_dof_error | 9,411 | ✗ Very Short | 5.06 | 979.5 |
| 2026-05-14_09-26-42_except_foot_friction_average_no_action_penalty | 9,361 | ✗ Very Short | 8.61 | 952.7 |
| 2026-05-14_10-27-11_change_collision | 5,706 | ✗ Very Short | 5.19 | 934.6 |
| 2026-05-14_12-12-15_change_physx_config | 851 | ✗ Very Short | -3.28 | 838.7 |

## Section 4: Rankings by Final Terrain Level

### Top 5 Experiments

| Rank | Experiment | Terrain Level | Reward | Episodes |
|---|---|---|---|---|
| 1 | 2026-05-13_12-39-19_flat_terrain_change | 6.1814 | 21.00 | 859.0 |
| 2 | 2026-05-13_09-28-12_flat_terrain | 6.1138 | 21.07 | 830.7 |
| 3 | 2026-05-12_18-23-24_change_various_thing_yaw_diff | 5.7121 | 16.93 | 758.9 |
| 4 | 2026-05-11_15-27-00_change_actuator | 5.6942 | 12.47 | 764.7 |
| 5 | 2026-05-13_13-37-21_flat_terrain_change | 5.6557 | 22.02 | 846.7 |

### Bottom 5 Experiments

| Rank | Experiment | Terrain Level | Reward | Episodes |
|---|---|---|---|---|
| 1 | 2026-05-14_12-12-15_change_physx_config | 1.1354 | -3.28 | 838.7 |
| 2 | 2026-05-11_17-46-01_damping_2.0 | 0.9653 | 3.46 | 937.7 |
| 3 | 2026-05-11_17-47-11_add_feet_dragging | 0.8750 | 5.02 | 973.6 |
| 4 | 2026-05-14_09-26-42_except_foot_friction_average_no_action_penalty | 0.3833 | 8.61 | 952.7 |
| 5 | 2026-05-14_09-26-09_up_no_dof_error | 0.2960 | 5.06 | 979.5 |

## Section 5: Statistical Summary

- **Terrain Level** (Final)
  - Mean: 3.4657
  - Median: 4.2407
  - Std Dev: 2.1847
  - Min: 0.2960
  - Max: 6.1814

- **Mean Reward** (Final)
  - Mean: 9.6499
  - Median: 8.6071
  - Std Dev: 7.3004
  - Min: -3.2751
  - Max: 22.0156

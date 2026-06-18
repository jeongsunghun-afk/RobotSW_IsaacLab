# Go2 Parkour Kinematic Diagnostic — Summary (v2)

- Checkpoint: `2026-06-08_17-39-34_only_flat_duty_4.2/model_15000.pt`
- Terrain: pure flat (parkour_flat proportion=1.0)
- num_envs=32, total_steps=1200, warmup_discarded=200 (4.0s = 4.0×tau)
- step_dt=0.0200s | EMA tau≈1s≈50 steps | contact debounce=3 steps
- Feet in `_feet_ids` order (env assertion): ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
- Foot→leg: [('FL_foot', 'FL'), ('FR_foot', 'FR'), ('RL_foot', 'RL'), ('RR_foot', 'RR')]
- Joint indices [hip, thigh, calf] per foot:
  - FL: [0, 4, 8] = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint']
  - FR: [1, 5, 9] = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint']
  - RL: [2, 6, 10] = ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
  - RR: [3, 7, 11] = ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

## Sanity Check: contact_duty + air_time lens
_User observation: RL 비정상 (나머지 3발은 앞으로 뻗음)_
_Lens: (a) RL contact_duty ≈ 0.30 floor vs RR > 0.40  (b) RL air_time distribution vs RR_

Post-warmup contact_duty means:
| Foot | duty_mean | pinned_at_floor? |
|------|-----------|-----------------|
| FL | 0.7475 | no |
| FR | 0.6205 | no |
| RL | 0.4481 | no |
| RR | 0.6152 | no |

Air_time at touchdown (seconds), per foot:
| Foot | N | median | mean | p90  |
|------|---|--------|------|------|
| FL | 2003 | 0.0850 | 0.0658 | 0.1000 |
| FR | 2013 | 0.1400 | 0.1301 | 0.1700 |
| RL | 1965 | 0.2050 | 0.1443 | 0.2300 |
| RR | 1997 | 0.1350 | 0.0841 | 0.1550 |

**Sanity check result**: REPRODUCED (2/3 signals): duty_floor=False, duty_asym(RR−RL=0.167)=True, at_asym=True — RL 비정상 재현됨

## Metric 1: contact_duty (EMA, post-warmup)
_EMA tau=1s, warmup=200 steps (4.0s) discarded_

| Foot | mean | std  | min  | max  |
|------|------|------|------|------|
| FL | 0.7475 | 0.0698 | 0.3796 | 0.9399 |
| FR | 0.6205 | 0.0636 | 0.3530 | 0.8582 |
| RL | 0.4481 | 0.0361 | 0.2941 | 0.5636 |
| RR | 0.6152 | 0.0457 | 0.3953 | 0.7338 |

## Metric 2a: joint_pos — hip (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | 0.0775 | -0.1424 | 0.4156 | 0.5580 | 0.1366 |
| FR | -0.0560 | -0.2387 | 0.0768 | 0.3156 | -0.0809 |
| RL | 0.1183 | -0.0628 | 0.2890 | 0.3518 | 0.1131 |
| RR | -0.1003 | -0.2601 | 0.0814 | 0.3415 | -0.0893 |

## Metric 2b: joint_pos — thigh (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | 0.7910 | 0.2417 | 1.1468 | 0.9051 | 0.6943 |
| FR | 0.9954 | 0.6662 | 1.2595 | 0.5934 | 0.9629 |
| RL | 1.0157 | 0.6186 | 1.3294 | 0.7107 | 0.9740 |
| RR | 0.9104 | 0.3881 | 1.2395 | 0.8514 | 0.8138 |

## Metric 2c: joint_pos — calf (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | -1.3921 | -1.7068 | -0.8357 | 0.8711 | -1.2713 |
| FR | -1.3694 | -1.7131 | -0.8369 | 0.8762 | -1.2750 |
| RL | -1.3953 | -1.6676 | -0.8377 | 0.8299 | -1.2527 |
| RR | -1.4132 | -1.6964 | -0.8376 | 0.8588 | -1.2670 |

## Metric 3: stride_length (world-frame displacement · liftoff-heading, meters)
_v2 fix: world-frame liftoff/touchdown positions projected onto heading at liftoff._
_Positive = forward, negative = backward._

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2003 | 0.1290 | 0.0244 | 0.0104 | 0.2313 |
| FR | 2013 | 0.1950 | 0.0372 | 0.0049 | 0.3057 |
| RL | 1965 | 0.3171 | 0.0443 | 0.0054 | 0.8268 |
| RR | 1997 | 0.3829 | 0.0681 | -0.0356 | 0.5149 |

## Metric 4: foot_clearance (max z − liftoff_z per swing, meters)
_Per-foot liftoff_z baseline. NOT global terrain reference._

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2003 | 0.0002 | 0.0007 | 0.0000 | 0.0108 |
| FR | 2013 | 0.0053 | 0.0047 | 0.0000 | 0.0551 |
| RL | 1965 | 0.0206 | 0.0058 | 0.0000 | 0.0589 |
| RR | 1997 | 0.0146 | 0.0031 | 0.0000 | 0.0260 |

## Metric 5: air_time at touchdown (seconds)

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2003 | 0.0658 | 0.0380 | 0.0100 | 0.1550 |
| FR | 2013 | 0.1301 | 0.0450 | 0.0000 | 0.2600 |
| RL | 1965 | 0.1443 | 0.0969 | 0.0000 | 0.2700 |
| RR | 1997 | 0.0841 | 0.0695 | 0.0000 | 0.1700 |

## Verdict: 못 한다 vs 안 한다 (1차 판정)

**핵심 비교: RL(index=2) vs RR(index=3) — 같은 뒷다리 역할, 반대쪽**

- **hip**: RL range=0.3518 rad, RR range=0.3415 rad, ratio=1.030 | RL center=0.1131, RR center=-0.0893
- **thigh**: RL range=0.7107 rad, RR range=0.8514 rad, ratio=0.835 | RL center=0.9740, RR center=0.8138
- **calf**: RL range=0.8299 rad, RR range=0.8588 rad, ratio=0.966 | RL center=-1.2527, RR center=-1.2670
- **contact_duty**: RL=0.4481, RR=0.6152, diff(RR−RL)=0.1670

**판정 기준** (thigh range ratio = RL/RR):
- ratio < 0.7 → RL range 30%+ 잘림 → **못 한다** (구조적 제약)
- ratio ≥ 0.7 → range 유사, center만 다름 → **안 한다** (reward 교정 가능)

### 1차 판정: **안 한다** (thigh ratio=0.835 ≥ 0.7; range 유사, center 차이=0.1602 rad → policy가 RL을 다른 각도 영역 사용)

> 주의: 가설은 가설로 표기. actuator blame 금지(4발 대칭 → RL-only 설명 불가).

## Notes
- v2 stride fix: world-frame liftoff→touchdown XY dot heading_at_liftoff.
- contact debounce: 3 consecutive contact steps before touchdown.
- warmup=200 steps=4.0s=4.0×tau discarded from EMA stats.
- foot_xy_w dump in raw_kinematics.npz for cross-check of stride pipeline.

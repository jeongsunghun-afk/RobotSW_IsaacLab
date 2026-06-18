# Go2 Parkour Kinematic Diagnostic — Summary (v2)

- Checkpoint: `2026-06-05_16-02-33_contact_duty_deficit_v1/model_49999.pt`
- num_envs=32, total_steps=1200, warmup_discarded=200 (4.0s = 4.0×tau)
- step_dt=0.0200s | EMA tau≈1s≈50 steps | contact debounce=3 steps
- Feet in `_feet_ids` order (env assertion): ['FL_foot', 'FR_foot', 'RL_foot', 'RR_foot']
- Foot→leg: [('FL_foot', 'FL'), ('FR_foot', 'FR'), ('RL_foot', 'RL'), ('RR_foot', 'RR')]
- Joint indices [hip, thigh, calf] per foot:
  - FL: [0, 4, 8] = ['FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint']
  - FR: [1, 5, 9] = ['FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint']
  - RL: [2, 6, 10] = ['RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint']
  - RR: [3, 7, 11] = ['RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint']

## Sanity Check: swing event count per foot
_User observation: 'RL만 정상적으로 안 움직임' (나머지 3발은 앞으로 뻗음)_
_Expected if observation is reproduced: RL swing count << FL/FR/RR_

| Foot | swing_count | ratio_vs_FL |
|------|-------------|-------------|
| FL | 2075 | 1.00 |
| FR | 2371 | 1.14 |
| RL | 1947 | 0.94 |
| RR | 2067 | 1.00 |

**Sanity check (worker, swing-count lens)**: swing count RL(1947)≈others → "symmetric" → NOT_REPRODUCED.

> **⚠️ LEAD 정정 (2026-06-08, advisor 검토 후)**: 위 swing-count sanity check는 **잘못된 렌즈**다. 사용자 관찰은 RL이 "정상적으로 안 움직인다"이지 "덜 자주 움직인다"가 아니다. RL은 발을 자주 떼지만(swing count 정상) **딛지 않고 뜬다**: contact_duty 0.315(vs 0.61~0.70), air_time 최대(0.147/std0.110), clearance 최대(0.032). swing count는 "얼마나 자주 떼나"만 보고 "제대로 딛나/규칙적인가"를 못 본다. → **사용자 관찰은 재현됨**(metric을 contact_duty/air_time으로 보면). worker가 "moves properly"를 "moves often"으로 잘못 조작화한 것. 아래 verdict는 이 정정 기준으로 유효.

## Metric 1: contact_duty (EMA, post-warmup)
_EMA tau=1s, warmup=200 steps (4.0s) discarded_

| Foot | mean | std  | min  | max  |
|------|------|------|------|------|
| FL | 0.6958 | 0.0761 | 0.2881 | 0.8172 |
| FR | 0.7028 | 0.1029 | 0.2261 | 0.9072 |
| RL | 0.3152 | 0.0667 | 0.0854 | 0.4854 |
| RR | 0.6068 | 0.0595 | 0.3000 | 0.7237 |

## Metric 2a: joint_pos — hip (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | 0.0433 | -0.2846 | 0.2910 | 0.5756 | 0.0032 |
| FR | -0.0485 | -0.2340 | 0.1036 | 0.3376 | -0.0652 |
| RL | 0.0992 | -0.0630 | 0.3426 | 0.4056 | 0.1398 |
| RR | 0.0031 | -0.2631 | 0.1843 | 0.4474 | -0.0394 |

## Metric 2b: joint_pos — thigh (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | 0.7737 | 0.1888 | 1.1842 | 0.9954 | 0.6865 |
| FR | 1.0519 | 0.5732 | 1.2830 | 0.7098 | 0.9281 |
| RL | 1.3181 | 0.8118 | 1.6497 | 0.8380 | 1.2308 |
| RR | 1.1942 | 0.5131 | 1.4621 | 0.9490 | 0.9876 |

## Metric 2c: joint_pos — calf (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | -1.4497 | -1.6679 | -0.8352 | 0.8327 | -1.2515 |
| FR | -1.3366 | -1.7494 | -0.8360 | 0.9134 | -1.2927 |
| RL | -1.2953 | -1.7652 | -0.8375 | 0.9277 | -1.3013 |
| RR | -1.5272 | -1.9179 | -0.8366 | 1.0813 | -1.3772 |

## Metric 3: stride_length (world-frame displacement · liftoff-heading, meters)
_v2 fix: world-frame liftoff/touchdown positions projected onto heading at liftoff._
_Positive = forward, negative = backward._

> **⚠️ stride_length 크기(magnitude)는 신뢰하지 말 것 (advisor)**: RL max=2.17m는 물리적 불가능 = missed-touchdown이 여러 cycle을 한 stride로 묶은 파이프라인 오염. RL의 **stride std 폭발(0.150)·max 이상치**는 "불규칙성"의 정성 신호로만 읽고 **stride 크기는 판정 근거에서 제외**. 1차 신호는 contact_duty(EMA+debounce로 가장 견고).

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2075 | 0.1425 | 0.0218 | 0.0127 | 0.2347 |
| FR | 2371 | 0.1377 | 0.0451 | 0.0166 | 0.3081 |
| RL | 1947 | 0.3423 | 0.1498 | -0.0010 | 2.1715 |
| RR | 2067 | 0.3558 | 0.0428 | -0.0537 | 0.4860 |

## Metric 4: foot_clearance (max z − liftoff_z per swing, meters)
_Per-foot liftoff_z baseline. NOT global terrain reference._

> **front feet clearance≈0 (FL 0.0008) 설명**: 앞발은 낮게 전진(stride 0.14 양수, forward)하는 정상 패턴 → "안 든다 ≠ 비정상". RL의 clearance 최대(0.032)는 "RL이 발을 크게 들고 뜬다"는 RL-단독 비정상의 일부. 즉 비정상은 RL이지 front feet가 아님.

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2075 | 0.0008 | 0.0019 | 0.0000 | 0.0264 |
| FR | 2371 | 0.0074 | 0.0109 | 0.0000 | 0.1060 |
| RL | 1947 | 0.0323 | 0.0115 | 0.0000 | 0.0768 |
| RR | 2067 | 0.0228 | 0.0052 | 0.0000 | 0.0536 |

## Metric 5: air_time at touchdown (seconds)

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2075 | 0.0952 | 0.0349 | 0.0100 | 0.1750 |
| FR | 2371 | 0.0806 | 0.0456 | 0.0000 | 0.2450 |
| RL | 1947 | 0.1473 | 0.1100 | 0.0000 | 0.3450 |
| RR | 2067 | 0.1012 | 0.0643 | 0.0000 | 0.1800 |

## Verdict: 못 한다 vs 안 한다 (1차 판정)

**핵심 비교: RL(index=2) vs RR(index=3) — 같은 뒷다리 역할, 반대쪽**

- **hip**: RL range=0.4056 rad, RR range=0.4474 rad, ratio=0.907 | RL center=0.1398, RR center=-0.0394
- **thigh**: RL range=0.8380 rad, RR range=0.9490 rad, ratio=0.883 | RL center=1.2308, RR center=0.9876
- **calf**: RL range=0.9277 rad, RR range=1.0813 rad, ratio=0.858 | RL center=-1.3013, RR center=-1.3772
- **contact_duty**: RL=0.3152, RR=0.6068, diff(RR−RL)=0.2916

**판정 기준** (thigh range ratio = RL/RR):
- ratio < 0.7 → RL range 30%+ 잘림 → **못 한다** (구조적 제약)
- ratio ≥ 0.7 → range 유사, center만 다름 → **안 한다** (reward 교정 가능)

### 최종 판정 (LEAD, advisor 검토): **사용자 관찰 재현됨 + "안 한다"(local optimum)**

**단일 현상**(독립 증거로 3중 계산 금지): RL은 발을 크게 들고(clearance 최대) 오래·불규칙하게 떠 있으며(air_time 최대/std 최대) 최소한만 딛는다(duty 0.315 = floor 정박) = "pawing/floating". hind pair duty diff(RR−RL)=0.29 vs front pair diff=0.007 → **좌우 뒷다리 비대칭이 핵심**.

**왜 "안 한다"인가**: 3관절 range ratio 모두 ≥0.85(hip 0.907 / thigh 0.883 / calf 0.858) = RL range 안 잘림 → 구조적 "못 한다" 아님. RL은 충분한 관절 가동범위를 갖지만 그 범위를 stance가 아닌 swing/공중에 쓴다(thigh center 1.23 = 가장 굽힘). = **reward 교정 가능**.

**메커니즘 (advisor)**: `contact_duty_deficit = clamp(target−duty, min=0)`이라 **duty≥0.30이면 gradient=0**. RL은 floor(0.30)에 정박할 유인은 충분하고 초과할 유인은 없다. 나머지 3발은 gait 역학으로 자연히 0.30을 초과(duty 0.6-0.7) → floor에 안 걸림. RL만 직전 3-leg 병리의 **kinematic 잔재**로 floor에 정박. (RL_calf action 발산은 이미 정상화됐으나 RL movement 자체는 미해결 — action_stats proxy의 한계.)

> 주의: actuator blame 금지(4발 대칭 → RL-only 설명 불가, §5/§6). 순수 L/R asset 비대칭이면 FL도 망가져야 하나 FL duty 0.70 정상 → L/R asset bias 반증. 단 **RL link 단독 결함은 asset(USD/URDF) 직접 읽기로만 마감** → 별도 확인 중.
> fix 후보(다음 턴 결정): (a) L/R 비대칭 contact-duty penalty (b) deficit의 floor-only 모양 수정/target 상향. 둘 다 reward-side(="안 한다"와 정합). 03 제안서 2순위(L/R 대칭 신호)를 이 메커니즘으로 재평가 예정.

## Notes
- v2 stride fix: world-frame liftoff→touchdown XY dot heading_at_liftoff.
- contact debounce: 3 consecutive contact steps before touchdown.
- warmup=200 steps=4.0s=4.0×tau discarded from EMA stats.
- foot_xy_w dump in raw_kinematics.npz for cross-check of stride pipeline.

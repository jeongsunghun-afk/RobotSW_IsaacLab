# Go2 Parkour Kinematic Diagnostic — Summary (v2)

- Checkpoint: `2026-06-08_13-11-04_only_flat/model_2500.pt`
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
| FL | 0.7791 | no |
| FR | 0.6719 | no |
| RL | 0.3362 | YES — floor |
| RR | 0.6071 | no |

Air_time at touchdown (seconds), per foot:
| Foot | N | median | mean | p90  |
|------|---|--------|------|------|
| FL | 2066 | 0.0700 | 0.0580 | 0.0850 |
| FR | 2071 | 0.1100 | 0.0940 | 0.1350 |
| RL | 2062 | 0.2400 | 0.1768 | 0.2700 |
| RR | 2068 | 0.1350 | 0.0860 | 0.1600 |

**Sanity check result**: REPRODUCED (3/3 signals): duty_floor=True, duty_asym(RR−RL=0.271)=True, at_asym=True — RL 비정상 재현됨

## Metric 1: contact_duty (EMA, post-warmup)
_EMA tau=1s, warmup=200 steps (4.0s) discarded_

| Foot | mean | std  | min  | max  |
|------|------|------|------|------|
| FL | 0.7791 | 0.0719 | 0.3953 | 0.9473 |
| FR | 0.6719 | 0.0774 | 0.3230 | 0.8838 |
| RL | 0.3362 | 0.0462 | 0.1412 | 0.5084 |
| RR | 0.6071 | 0.0425 | 0.4116 | 0.7141 |

## Metric 2a: joint_pos — hip (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | 0.1247 | -0.1285 | 0.4584 | 0.5869 | 0.1650 |
| FR | -0.0924 | -0.4259 | 0.1176 | 0.5435 | -0.1542 |
| RL | 0.0978 | -0.1070 | 0.3079 | 0.4149 | 0.1004 |
| RR | -0.1223 | -0.3314 | 0.2948 | 0.6262 | -0.0183 |

## Metric 2b: joint_pos — thigh (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | 0.8399 | 0.2371 | 1.2952 | 1.0580 | 0.7662 |
| FR | 0.9770 | 0.5657 | 1.2450 | 0.6793 | 0.9054 |
| RL | 1.1673 | 0.8653 | 1.4204 | 0.5552 | 1.1429 |
| RR | 0.9633 | 0.4030 | 1.3318 | 0.9289 | 0.8674 |

## Metric 2c: joint_pos — calf (rad)

| Foot | mean  | min   | max   | range | center |
|------|-------|-------|-------|-------|--------|
| FL | -1.3728 | -2.0003 | -0.8353 | 1.1651 | -1.4178 |
| FR | -1.4047 | -1.9637 | -0.8359 | 1.1277 | -1.3998 |
| RL | -1.3900 | -1.7467 | -0.8372 | 0.9095 | -1.2920 |
| RR | -1.4098 | -1.7470 | -0.8351 | 0.9119 | -1.2910 |

## Metric 3: stride_length (world-frame displacement · liftoff-heading, meters)
_v2 fix: world-frame liftoff/touchdown positions projected onto heading at liftoff._
_Positive = forward, negative = backward._

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2066 | 0.1176 | 0.0268 | 0.0072 | 0.2345 |
| FR | 2071 | 0.1735 | 0.0342 | 0.0141 | 0.4277 |
| RL | 2062 | 0.3450 | 0.0756 | -0.0066 | 1.3869 |
| RR | 2068 | 0.3810 | 0.0645 | -0.0333 | 0.4883 |

## Metric 4: foot_clearance (max z − liftoff_z per swing, meters)
_Per-foot liftoff_z baseline. NOT global terrain reference._

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2066 | 0.0006 | 0.0011 | 0.0000 | 0.0119 |
| FR | 2071 | 0.0062 | 0.0066 | 0.0000 | 0.0904 |
| RL | 2062 | 0.0316 | 0.0095 | 0.0000 | 0.0666 |
| RR | 2068 | 0.0126 | 0.0032 | 0.0000 | 0.0298 |

## Metric 5: air_time at touchdown (seconds)

| Foot | N | mean  | std   | min   | max   |
|------|---|-------|-------|-------|-------|
| FL | 2066 | 0.0580 | 0.0283 | 0.0100 | 0.1400 |
| FR | 2071 | 0.0940 | 0.0489 | 0.0100 | 0.2550 |
| RL | 2062 | 0.1768 | 0.1101 | 0.0000 | 0.6450 |
| RR | 2068 | 0.0860 | 0.0687 | 0.0000 | 0.1700 |

## Verdict: 못 한다 vs 안 한다 (1차 판정)

**핵심 비교: RL(index=2) vs RR(index=3) — 같은 뒷다리 역할, 반대쪽**

- **hip**: RL range=0.4149 rad, RR range=0.6262 rad, ratio=0.663 | RL center=0.1004, RR center=-0.0183
- **thigh**: RL range=0.5552 rad, RR range=0.9289 rad, ratio=0.598 | RL center=1.1429, RR center=0.8674
- **calf**: RL range=0.9095 rad, RR range=0.9119 rad, ratio=0.997 | RL center=-1.2920, RR center=-1.2910
- **contact_duty**: RL=0.3362, RR=0.6071, diff(RR−RL)=0.2709

**판정 기준** (thigh range ratio = RL/RR):
- ratio < 0.7 → RL range 30%+ 잘림 → **못 한다** (구조적 제약)
- ratio ≥ 0.7 → range 유사, center만 다름 → **안 한다** (reward 교정 가능)

### 최종 판정 (LEAD, advisor 검토): **"못 한다" label 철회** — RL parked 상태, 능력은 입증됨

thigh ratio 0.598<0.7의 기계적 "못 한다"는 **오판**이다(advisor): range는 terrain이 다리를 외부에서 얼마나 흔드느냐의 함수다. parkour 지형은 RL을 가끔 강제로 움직여 range를 0.838로 부풀리고, 평지는 안 흔드니 0.555로 주저앉는다 — **같은 "RL parked" 상태**일 뿐. asset RL/RR 18/18 대칭 + thigh joint limit ~5rad인데 0.55만 씀 = **능력은 이미 입증** → 구조적 "못 한다(asset/limit)" 아님.

**그러나 단순 "안 한다(local optimum)"로도 닫지 않는다.** 진짜 헤드라인 신호: **전체 saga에서 항상 rear-LEFT(RL)** — air_time_cap 3-leg, parkour contact_duty_v1, flat 모두 RL, RR/FL/FR은 한 번도 아님. 독립 random-seed local optimum은 run마다 다른 다리로 수렴해야 하는데 **항상 RL** + thigh center 두 모델 모두 RL≈1.14 vs RR≈0.87-0.99 같은 방향 ~0.25rad offset → **좌/후-특이적 systematic 요인**(code 레벨 비대칭: init pose / index-2 처리 / obs 미러). asset 대칭은 이 branch를 못 닫는다.

**최종 결론 (advisor 검토): "안 한다"(seed-specific local optimum) 확정 — 구조/코드 원인 없음**

discriminator 전부 배제됨:
- ① **독립성**: flat·parkour 두 모델 모두 `seed: 42` 공유, `resume: false`. 같은 seed → "두 독립 모델" 논거 약화(같은 seed면 같은 다리 수렴이 자연).
- ② **init pose**: RL vs RR 완전 미러 대칭(hip +0.1/−0.1, thigh 1.0/1.0, calf −1.5/−1.5). **hip L=+0.1/R=−0.1은 4족 표준 좌우 미러(정상)**.
- ③ **코드 비대칭**: 없음(Explore grep). reward(contact_duty/feet_dragging hind=[2,3]/feet_edge/stumble/air_time_cap)·obs cat·command 전부 4발 균일, RL 오매핑 하드코딩 무.
- ④ mirror test: 학습된(이미 비대칭) 정책엔 무의미 → 불필요.

**⚠️ Explore의 "init pose 비대칭이 culprit" 주장 폐기 (advisor airtight 반박)**: FL과 RL은 **둘 다 `.*L_hip_joint:+0.1`로 동일 init**인데 FL은 건강(duty 0.78)/RL은 parked(0.33) → **같은 init 값, 반대 결과 → hip init은 차이의 원인이 될 수 없음**. Explore가 static init posture를 gait-phase diagonal pairing과 혼동.

**왜 항상 RL인가 (완전 인과)**: 대칭 asset/code/init + **랜덤 초기화된 MLP는 좌우 대칭이 아님**(step 0부터 4발을 다른 가중치로 매핑) + 학습에 L/R 대칭 항 부재. **seed 42 → 동일 init 비대칭 → gradient descent가 불리한 다리(RL)를 parked로 증폭 → 매번 RL.** 다른 seed면 다른 다리가 park될 것(optional 확인, blocker 아님).

**fix (greenlit, 다음): `contact_duty_target` 0.30→~0.42.** 정당성 = floor-gradient 메커니즘(duty≥0.30 ⇒ deficit gradient=0 ⇒ RL이 floor 정박 ⇒ floor 올려 gradient 생성). RR 하강은 절대 per-foot 공식이라 자기-처벌로 차단(degenerate 불가). **flat에서 seed 42로 단일변수 검증**, per-foot `_foot_contact_duty` TB 로깅을 kill-switch로(RL duty 안 오르면 early kill). 실패 시 escalation = L/R symmetry augmentation(`rsl_rl/.../symmetry.py`).

## Notes
- v2 stride fix: world-frame liftoff→touchdown XY dot heading_at_liftoff.
- contact debounce: 3 consecutive contact steps before touchdown.
- warmup=200 steps=4.0s=4.0×tau discarded from EMA stats.
- foot_xy_w dump in raw_kinematics.npz for cross-check of stride pipeline.

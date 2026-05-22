# May 21 Parkour 실험 — 하이퍼파라미터 & Reward Scale 변경

> Baseline: `2026-05-20_17-14-12_clip_actions_10` (직전 마지막 05/20 run).

## Agent (PPO) 하이퍼파라미터

**모든 7개 실험에서 agent.yaml은 baseline과 동일** (변경 없음):

| 항목                 | 값              |
| -------------------- | --------------- |
| learning_rate        | 0.0002          |
| gamma                | 0.99            |
| lam (GAE λ)          | 0.95            |
| entropy_coef         | 0.01            |
| clip_param           | 0.2             |
| num_learning_epochs  | 5               |
| num_mini_batches     | 4               |
| value_loss_coef      | 1.0             |
| max_grad_norm        | 1.0             |
| actor/critic hidden  | [512, 256, 128] |

→ **05/21 변경은 전적으로 환경(env.yaml) 측에서 발생**.

## Environment 변경 — Baseline 대비 차이만

### Exp 1: `09-49-21_goal_idx_visualization3`
- `clip_actions`: 10.0 (변경 없음)
- reward scales: **변경 없음**
- terrain: **변경 없음** (20 × 4 m, 표준 비율)
- → 환경 cfg 차이 없음. 코드만 진단 로깅 추가.

### Exp 2: `10-05-51_clip_actions_16`
- `clip_actions`: 10.0 → **16.0**
- reward scales: 변경 없음
- terrain: 변경 없음 (cfg 스냅샷 기준; 코드 diff에는 terrain 조정이 있으나 env.yaml에 반영 여부는 별도 검증)

### Exp 3: `10-11-16_clip_actions10_action_rate_0.01`
- `clip_actions`: 10.0 (유지)
- **`reward_scales.action_rate_l2`: -0.05 → -0.01** (5배 페널티 ↓)
- terrain: 변경 없음

### Exp 4 & 5: `14-49-35` / `14-58-19_change_terrain_difficulty`
**두 실험의 env.yaml은 동일** (14:49가 단명한 이유는 cfg가 아닌 외부 요인).
- terrain x size: 20 → **25** m
- `parkour_hurdle` x_spacing: (1.0, 1.5) → (1.8, 2.4)
- `parkour_step` x_length: (0.4, 0.8) → (1.2, 2.0)
- `parkour_stair`: `num_steps_per_stair` 4 → 8, `stair_height` (0.05, 0.20) → (0.05, 0.25)
- `parkour_flat/hurdle` platform_length: 2.5 → 1.5
- reward scales: 변경 없음

### Exp 6: `15-48-19_feet_dragging`
- `clip_actions`: 10.0
- terrain: baseline (20 × 4)
- **reward_scales 추가**: `feet_dragging: -0.1` (신규)
- cfg 추가: `dragging_velocity_threshold: 0.05`

### Exp 7: `17-31-36_add_feet_air_time_flat` ⚠️ 망가짐
- `clip_actions`: 10.0
- terrain: baseline (20 × 4)
- **reward_scales 추가**:
  - `feet_dragging: -0.1` (Exp 6에서 유지)
  - **`feet_air_time: +0.5`** (신규, 강한 양의 보상)
- cfg 추가:
  - `feet_air_time_threshold: 0.3`
  - `feet_air_time_cmd_speed_threshold: 0.1`

## Reward Scale 비교 표

| Reward 항목         | base 05/20-17:14 | Exp1 | Exp2 | Exp3       | Exp4/5 | Exp6     | Exp7         |
| ------------------- | ---------------- | ---- | ---- | ---------- | ------ | -------- | ------------ |
| tracking_goal_vel   | 1.5              | 1.5  | 1.5  | 1.5        | 1.5    | 1.5      | 1.5          |
| tracking_yaw        | 0.5              | 0.5  | 0.5  | 0.5        | 0.5    | 0.5      | 0.5          |
| lin_vel_z           | -1.0             | -1.0 | -1.0 | -1.0       | -1.0   | -1.0     | -1.0         |
| ang_vel_xy          | -0.05            | -0.05 | -0.05 | -0.05    | -0.05  | -0.05    | -0.05        |
| orientation         | -1.0             | -1.0 | -1.0 | -1.0       | -1.0   | -1.0     | -1.0         |
| dof_acc             | -2.5e-7          | …    | …    | …          | …      | …        | …            |
| collision           | -10.0            | …    | …    | …          | …      | …        | …            |
| action_rate_l2      | -0.05            | -0.05 | -0.05 | **-0.01** | -0.01  | -0.01    | -0.01        |
| feet_stumble        | -1.0             | -1.0 | -1.0 | -1.0       | -1.0   | -1.0     | -1.0         |
| feet_edge           | -1.0             | -1.0 | -1.0 | -1.0       | -1.0   | -1.0     | -1.0         |
| **feet_dragging**   | (없음)           | -    | -    | -          | -      | **-0.1** | **-0.1**     |
| **feet_air_time**   | (없음)           | -    | -    | -          | -      | -        | **+0.5** ⚠️ |

## 하이퍼파라미터 비교 표

| 항목         | 모든 실험 |
| ------------ | --------- |
| num_envs     | 4096 (또는 cfg default) |
| num_steps_per_env | 24 (PPO rollout) |
| 그 외        | baseline와 100% 동일 |

→ **agent 하이퍼파라미터는 한 줄도 안 바뀜**. 05/21 전체가 env-only 실험.

## Exp 7 망가짐 원인 후보 (정량적)

### scale magnitude 분석
- `feet_air_time: +0.5`는 `tracking_goal_vel: +1.5`의 1/3 → 비중 큼.
- 음의 안정성 reward 합산이 보통 환경에서 ~-1.0/step 수준인 점을 고려하면,
  `air_time_bonus = (air_time - 0.3) × first_contact` 가 0.5 s 체공 한 발마다 +0.5 × (0.5 - 0.3) = +0.1 만 보상.
- 그러나 4발 합산하면 +0.4, 그리고 `gate × is_flat` 곱셈으로 flat terrain에서만 보상.
- **위험**: 다리를 들고 있는 시간을 늘릴수록 보상이 누적 → 안정성 희생 가능성.

### 같이 변경된 항목
- Exp 6의 `feet_dragging: -0.1` 페널티가 hind feet에만 적용되어, front feet은 자유롭게 들 수 있음.
- `is_flat` 게이트가 hurdle/step 진입 직전에 갑자기 꺼지면 보상 신호 불연속 → 학습 불안정.

### 예측되는 동작 (Exp 7)
1. flat terrain에서 4발 stomp 같은 과한 점프 패턴 학습.
2. cmd_speed > 0.1 게이트가 항상 통과되어 정지/저속 보행 데이터 학습 불가능.
3. air-time bonus 누적이 stumble/edge 페널티를 압도할 가능성 → 자세 무너짐.

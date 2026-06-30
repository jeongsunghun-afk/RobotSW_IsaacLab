# Go2 Fall-Recovery 환경 컨텍스트

## 현재 상태: M1 (env 골격 + 임의자세 fall 초기화)

M1에서 구현된 것:
- `Go2RecoveryEnv(DirectRLEnv)` 골격
- 42-dim 고유감각 obs (lin_vel 제거 — sim-to-real)
- Fall initialization 80/10/10 (fallen/standing/sitting)
- M1 placeholder reward (upright + action_rate)
- DebugViewer 3줄 패턴

M2에서 구현할 것:
- 본격 복구 reward (`_reward_reset`: roll + stand + height)
- Tier-0 안전: actuator vel limit ↓, smoothness penalty 강화
- Success 명시판정 (upright & near-default 유지 N step)

## 등록 ID & 파일

| ID | 클래스 | 파일 |
|----|--------|------|
| Go2Recovery-v0 | Go2RecoveryEnv | `go2_recovery_env.py` |

- 설정: `go2_recovery_env_cfg.py` → `Go2RecoveryEnvCfg`
- PPO: `agents/rsl_rl_ppo_cfg.py` → `Go2RecoveryPPORunnerCfg`

## 로봇 스펙 (Go2)
- DOF: 12 (action_space = 12)
- Default pose: z=0.27m, hip ±0.1, thigh FR/FL=0.8/RL/RR=1.0, calf=-1.5
- Actuator (DCMotor): effort 23.5Nm, vel_limit 30 rad/s, Kp=25, Kd=0.5
- episode_length_s = 10.0s (복구 목표 6~8초 + 여유)
- decimation = 4 (200Hz physics / 50Hz policy)

## Observation 구성 (observation_space = 42)

```
root_ang_vel_b      [3]   × ang_vel_scale=0.25
projected_gravity_b [3]   (scale 없음)
joint_pos_error     [12]  × dof_pos_scale=1.0
joint_vel           [12]  × dof_vel_scale=0.05
previous_actions    [12]  (scale 없음)
─────────────────────────
TOTAL               42
```

**주의**: linear velocity 없음 (sim-to-real 대응 — HumanUP 방식). command/goal 없음 (목표=default pose 고정).

## Fall 초기화 (80/10/10)

| 그룹 | 비율 | 방식 |
|------|------|------|
| fallen | 80% | random roll±135°/pitch±45°/yaw±180° + random joint (cubic lerp) + z+0.45m 공중낙하 |
| standing | 10% | default pose, z=0.27m |
| sitting | 10% | calf 추가 굽힘(-0.5 내), z-0.10m, yaw 랜덤 |

fall init 파라미터는 cfg 필드로 노출: `fall_height`, `fall_standing_ratio`, `fall_sitting_ratio`, `fall_roll_range`, `fall_pitch_range`, `fall_yaw_range`

## Reward (M1 placeholder)

| 항 | 수식 | scale |
|----|------|-------|
| upright | `(-gravity_b_z - 1.0).clamp(0,1)` | +1.0 |
| action_rate | `sum((a - a_prev)^2)` | -0.01 |

## 종료 조건

- `terminated`: base_z < -0.1m (파국, 넘어짐 자체는 종료 아님)
- `time_out`: episode_length_buf >= max_episode_length - 1

## API 근거

- 쿼터니언 변환: `isaaclab.utils.math.quat_from_euler_xyz(roll, pitch, yaw)` → `(w, x, y, z)` 순서
  - `quat_from_axis_angle`는 이 코드베이스에 존재하지 않음 (확인됨)
- root state write: `write_root_pose_to_sim(pos_quat_7, env_ids)` / `write_root_velocity_to_sim(vel_6, env_ids)` / `write_joint_state_to_sim(pos, vel, None, env_ids)`

## 수정 시 주의사항

- obs 순서 변경 시 `observation_space=42` cfg 동기화 필수
- 새 buffer 추가 시 반드시 `_reset_idx`에서 초기화
- `_previous_actions`는 `_get_observations` 진입 시 갱신 (go2 베이스 패턴 동일)
- DebugViewer: `__init__` 활성화 → `_get_observations` 끝 update → `__del__` close

## 학습 명령

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Recovery-v0 --num_envs 4096

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2Recovery-v0 --num_envs 32
```

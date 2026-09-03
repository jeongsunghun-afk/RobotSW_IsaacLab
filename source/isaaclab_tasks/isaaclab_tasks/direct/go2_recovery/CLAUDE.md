# Go2 Fall-Recovery 환경 컨텍스트

## 구현된 것 (`Go2RecoveryEnv`, `go2_recovery_env.py`)
- 42-dim 고유감각 obs (lin_vel 제거 — sim-to-real)
- Fall initialization 80/10/10 (fallen/standing/sitting)
- Reward: `reward_reset`(r_roll upright 정렬 + r_stand height/pose/vel), `r_roll_progress`(progress shaping), success 명시판정(`_success_region_mask`, `_update_success`) — placeholder 아닌 완성된 구조
- DebugViewer 3줄 패턴 (`__init__` 활성화 → `_get_observations` 끝 update → `__del__` close)

## 등록 ID & 파일 (4종, `__init__.py`)

| ID | 클래스/cfg | base 대비 추가된 것 |
|----|-----------|---------------------|
| Go2Recovery-v0 | `Go2RecoveryEnv` / `Go2RecoveryEnvCfg` | 기본 복구 reward + success 판정 |
| Go2Recovery-RisePacing-v0 | `Go2RecoveryRisePacingEnv` / `Go2RecoveryRisePacingEnvCfg`(상속: base) | smoothness state-gate: cos_dist 기반으로 `dof_acc_l2`/`action_smoothness_1,2` 페널티를 게이팅(flip 구간 최소 30% 유지) |
| Go2Recovery-RiseSlow-v0 | `Go2RecoveryRiseSlowEnv` / `Go2RecoveryRiseSlowEnvCfg`(상속: RisePacing) | `r_rise_pace`: uprightness 상승 속도가 목표(0.5/s)를 넘으면 페널티(cos<0.5 구간은 면제) |
| Go2Recovery-FlipVel-v0 | `Go2RecoveryFlipVelEnv` / `Go2RecoveryFlipVelEnvCfg`(상속: RiseSlow) | `r_joint_vel`: joint 속도가 8.0 rad/s 초과 시 curriculum ramp(warmup 48000 + ramp 48000 step) 페널티로 whip 억제 |

PPO 설정: `agents/rsl_rl_ppo_cfg.py`, `rsl_rl_ppo_rise_pacing_cfg.py`, `rsl_rl_ppo_rise_slow_cfg.py`, `rsl_rl_ppo_flip_vel_cfg.py`.

## 로봇 스펙 (Go2)
- DOF: 12 (action_space = 12)
- Default pose: z=0.27m, hip ±0.1, thigh FR/FL=0.8/RL/RR=1.0, calf=-1.5
- Actuator (DCMotor): effort 23.5Nm, vel_limit 18 rad/s, Kp=25, Kd=1.0 (Tier-0 안전 패치로 30→18, 0.5→1.0; `go2_recovery_env_cfg.py:130-132`)
- episode_length_s = 10.0s
- decimation = 4 (200Hz physics / 50Hz policy)

## Observation 구성 (observation_space = 42, `_get_observations`)
```
root_ang_vel_b      [3]   × ang_vel_scale=0.25
projected_gravity_b [3]   (scale 없음)
joint_pos_error     [12]  × dof_pos_scale=1.0
joint_vel           [12]  × dof_vel_scale=0.05
previous_actions    [12]  (scale 없음)
```
linear velocity 없음(sim-to-real 대응). command/goal 없음(목표=default pose 고정).

## Fall 초기화 (80/10/10, `go2_recovery_env_cfg.py`)
| 그룹 | 비율 | 방식 |
|------|------|------|
| fallen | 80% | roll±135°(2.356rad)/pitch±45°(0.785rad)/yaw±180°(3.1416rad) + random joint + z+0.45m 공중낙하 |
| standing | 10% | default pose, z=0.27m |
| sitting | 10% | calf 추가 굽힘, z-0.10m, yaw 랜덤 |

cfg 필드: `fall_height`, `fall_standing_ratio`, `fall_sitting_ratio`, `fall_roll_range`, `fall_pitch_range`, `fall_yaw_range`

## Reward 항 (base, `_get_rewards`)
항 이름 목록(가중치는 각 `*_env_cfg.py`의 `*_scale` 참조): `reward_reset`, `r_roll_progress`, `action_rate_l2`, `action_smoothness_1`, `action_smoothness_2`, `delta_torques`, `dof_acc_l2`, `dof_vel_l2`, `dof_torques_l2`, `dof_pos_limits`, `success_bonus`, `r_success_region`. 파생 task는 위 표의 추가 항(`r_rise_pace`, `r_joint_vel`)을 더함.

## 종료 조건
- `terminated`: base_z < -0.1m (파국), 또는 `terminate_on_success=True`이고 성공 판정 시
- `time_out`: episode_length_buf >= max_episode_length - 1
- settle 중(`settle_max_steps>0`)에는 terminated=False 강제(낙하 transient 방지, time_out은 마스킹 안 함)

## API 근거 (IsaacLab 3.0 / 6.0 기준)
- 쿼터니언: `isaaclab.utils.math.quat_from_euler_xyz(roll, pitch, yaw)` → **(x, y, z, w)** 순서. 6.0에서 wxyz → xyzw 전면 변경됨.
- `quat_apply`/`quat_apply_inverse`도 xyzw 입력 기대.
- `root_quat_w`는 `ProxyArray`(warp-first) → torch 접근 시 `.torch` 사용.
- `quat_from_axis_angle`는 이 코드베이스에 없음.
- root state write: `write_root_pose_to_sim_index`/`write_root_velocity_to_sim_index`/`write_joint_state_to_sim_index`

## 수정 시 주의사항
- obs 순서 변경 시 `observation_space=42` cfg 동기화 필수
- 새 buffer 추가 시 반드시 `_reset_idx`에서 초기화
- `_previous_actions`는 `_get_observations` 진입 시 갱신
- 파생 cfg(RisePacing/RiseSlow/FlipVel)는 base 파일을 수정하지 않고 상속으로만 확장(설계 원칙)

## 학습·평가
학습·렌더 실행 방법은 `.claude/rules/training.md` 참조.

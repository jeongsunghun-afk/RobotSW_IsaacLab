# Go2 WTW (Walk-The-Walk) 환경 컨텍스트

## 환경 개요
Go2 강건 보행 환경. 속도 명령 추종 + 자연스러운 동작 구현.
AMP 없이 순수 reward shaping으로 자연스러움을 유도.

## 등록 ID & 파일
Task ID 목록·러너 변종은 `README.md` 참조.

- 메인 환경: `WTWEnv` (`go2_wtw_env.py`)
- 설정: `go2_env_cfg.py` → `Go2FlatEnvCfg`, `Go2RoughEnvCfg`
- Neck 확장: `go2_neck_env.py`, `go2_neck_interaction_cfg.py`
- PPO 설정: `agents/rsl_rl_ppo_cfg.py`

## 로봇 스펙 (Go2)
- DOF: 12 (action_space = 12)
- Base 기본 높이: 0.34m (`base_height_target`)
- 종료 조건(`_get_dones`): base 접촉 센서 힘 > 1.0N (`net_contact_forces` on base body). base height 기준 종료 아님.
- 에피소드: 20초, decimation: 4 (200Hz physics / 50Hz policy)
- action_scale: 0.25

## Observation 구성 (Go2FlatEnvCfg 기준)
`observation_space`는 다음 합으로 계산됨(`go2_env_cfg.py:99-113`):
```
num_prio_obs = 3 + 14 + action_space*3 (+4 clock_inputs) = 57
num_priv_latent = 4 + num_friction(31) = 35   (priv_latent=True)
history_len = 20
observation_space = num_prio_obs + num_priv_latent + num_prio_obs * history_len = 1232
```
`Go2RoughEnvCfg`는 `observation_space = 235`로 별도 지정. 42-dim 단순 구성(중력+명령+joint 3항)은 이 코드베이스에 없음.

## Command 구성
`num_commands = 14`. 항목: `lin_vel_x_range`, `lin_vel_y_range`, `ang_vel_range`, `body_height_cmd_range`, `gait_frequency_cmd_range`, `gait_phase_cmd_range`, `gait_offset_cmd_range`, `gait_bound_cmd_range`, `gait_duration_cmd_range`, `footswing_height_range`, `body_pitch_range`, `body_roll_range`, `stance_width_range`, `stance_length_range`.
유효값(파일 내 마지막 정의, `go2_env_cfg.py:222-237`): `lin_vel_x_range=[-1.0, 2.0]`, `lin_vel_y_range=[-0.5, 0.5]`, `ang_vel_range=[-1.0, 1.0]`.
**주의**: 동일 클래스에 `command_cfg`가 두 번 재정의되어 있어(205행, 222행) 뒤쪽 값이 유효함.

## 현재 보상 구조 (`_get_rewards`, `go2_wtw_env.py:776-800`)
항 이름 목록(가중치는 `go2_env_cfg.py`의 `*_reward_scale` 참조):
`track_lin_vel_xy_exp`, `track_ang_vel_z_exp`, `lin_vel_z_l2`, `ang_vel_xy_l2`, `dof_torques_l2`, `dof_acc_l2`, `action_rate_l2`, `undesired_contacts`, `feet_clearance_cmd_linear`, `feet_clearance_bezier`, `feet_clearance_bezier_5th`, `orientation_control`, `raibert_heuristic`, `tracking_contacts_shaped_force`, `tracking_contacts_shaped_vel`, `dof_vel_l2`, `jump`, `action_smoothness1`, `action_smoothness2`, `foot_landing_vel`, `foot_landing_vel_xy`, `landing_impact`, `feet_vel_5th_late`.
`bezier_curve_reward`라는 단일 항은 없음(3차/5차로 분리됨).

## 알려진 이슈
- `bezier` 계열 발궤적 latch가 world frame 기준이라 회전 중 방향이 틀어질 수 있음 (2026-09-03 코드 대조 시 미확인 — 최신 구현에서 재현 여부 확인 안 함)

## 수정 시 주의사항
- 새 reward 추가 시 `go2_env_cfg.py`에 weight 파라미터 먼저 선언
- `_reset_idx`에서 bezier 관련 버퍼(phase, foot_targets 등) 초기화 확인
- Neck Module 확장 시 `go2_neck_env.py` 참조 (action_space=19로 변경됨)

## 학습·평가
학습·렌더 실행 방법은 `.claude/rules/training.md` 참조.

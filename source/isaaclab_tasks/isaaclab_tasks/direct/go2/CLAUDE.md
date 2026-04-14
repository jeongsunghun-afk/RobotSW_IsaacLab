# Go2 WTW (Walk-The-Walk) 환경 컨텍스트

## 환경 개요
Go2 강건 보행 환경. 속도 명령 추종 + 자연스러운 동작 구현.
AMP 없이 순수 reward shaping으로 자연스러움을 유도.

## 등록 ID & 파일

| ID | 클래스 | 파일 |
|----|--------|------|
| Go2WTW | WTWEnv | `go2_wtw_env.py` |

- 설정: `go2_env_cfg.py` → `Go2FlatEnvCfg`, `Go2RoughEnvCfg`
- Neck 확장: `go2_neck_env.py`, `go2_neck_interaction_cfg.py`
- PPO 설정: `agents/rsl_rl_ppo_cfg.py`

## 로봇 스펙 (Go2)
- DOF: 12 (action_space = 12)
- Base 기본 높이: ~0.34m
- 종료 조건: base height < 기준값
- 에피소드: 20초, decimation: 4 (200Hz physics / 50Hz policy)
- action_scale: 0.25

## Observation 구성 (observation_space = 42 기준, 변경 가능)
```
projected_gravity_b(3) + commands(3) +
joint_pos_error(12) + joint_vel(12) + actions(12)
```
**주의**: history_observation=True 시 크기가 배수로 증가함.

## Command 구성
```python
command_cfg = {
    "lin_vel_x_range": [0.0, 2.0],   # 전진 속도 (m/s)
    "lin_vel_y_range": [-0.0, 0.0],  # 횡방향
    "ang_vel_range":   [-0.5, 0.5],  # 회전 (rad/s)
}
```

## 현재 보상 구조
- `track_lin_vel_xy_exp`: 선속도 추종 (메인 보상)
- `track_ang_vel_z_exp`: 각속도 추종
- `raibert_heuristic`: 발 배치 패턴 (자연스러움)
- `bezier_curve_reward`: Bezier 곡선 기반 발 궤적 보상
- 페널티: action_rate, joint_vel, dof_pos_limits 등

## 알려진 이슈
- `bezier_curve_reward`는 swing 시작 시 world frame으로 c3를 latch → 회전 중 방향 틀어짐
- `track_ang_vel_z_exp`가 `raibert_heuristic`보다 낮게 수렴하는 경향 (world-frame latch 문제)

## 수정 시 주의사항
- 새 reward 추가 시 `go2_env_cfg.py`에 weight 파라미터 먼저 선언
- `_reset_idx`에서 bezier 관련 버퍼(phase, foot_targets 등) 초기화 확인
- Neck Module 확장 시 `go2_neck_env.py` 참조 (DOF 수 변경됨)

# R_Skeleton AMP 환경 컨텍스트

## 환경 개요
R_Skeleton에 AMP 모방학습 적용. Go2 AMP 구조를 34-DOF로 확장.
**현재 프로젝트 주력 환경** (Go2 AMP → R_Skeleton AMP 확장 단계).

## 등록 ID & 파일

| ID | 클래스 | 파일 |
|----|--------|------|
| R_Skeleton-AMP-v0 | SkeletonAmpEnv | `skeleton_amp_env.py` |

- 설정: `skeleton_amp_env_cfg.py` → `SkeletonAmpEnvCfg`
- 모션 로더: `motion_loader.py`
- PPO 설정: `agents/rsl_rl_ppo_cfg.py`

## 로봇 스펙
- DOF: 34 (action_space = 34)
- 발 링크: `FL_link7_toe`, `FR_link7_toe`, `HL_link7_toe`, `HR_link7_toe`
- termination_height: 0.3m
- action_scale: 0.25

## Observation 구성

### Policy obs (observation_space = 108)
```
projected_gravity_b(3) + commands(3) +
joint_pos_error(34) + joint_vel(34) + actions(34)
```

### AMP obs (amp_observation_space = 99)
```
dof_pos(34) + dof_vel(34) + root_height(1) +
lin_vel(3) + ang_vel(3) +
key_body_pos(12) + key_body_lin_vel(12)
```
**중요**: motion_loader.py의 obs 추출 순서와 반드시 일치.

## Motion Data 구조
```
R_Skeleton_amp/imitation/
└── new_dataset/   ← 현재 사용 중
    └── *.txt
```

## AMP 핵심 파라미터 (agents/rsl_rl_ppo_cfg.py)
```python
amp_cfg = {
    "task_reward_lerp": 0.5,     # 0.3~0.7 권장
    "reward_coef": 2.0 * 0.02,
    "reset_strategy": "random",  # RSI 반드시 사용
}
```

## Command 구성 (skeleton_amp_env_cfg.py)
```python
command_cfg = {
    "lin_vel_x_range": [0.0, 2.0],
    "lin_vel_y_range": [-0.0, 0.0],
    "ang_vel_range":   [-0.5, 0.5],
}
```

## Go2 AMP와의 차이점
| 항목 | Go2 AMP | R_Skeleton AMP |
|------|---------|----------------|
| DOF | 12 | 34 |
| amp_obs | 55 | 99 |
| policy_obs | 42 | 108 |
| termination_height | 0.15m | 0.3m |

## 수정 시 주의사항
- Go2 코드를 참조할 때 shape 전환에 주의 (12 → 34)
- `amp_observation_space = 99` 변경 시 motion_loader와 discriminator 동시 수정 필수
- `key_body_pos` 계산 시 링크 이름이 Go2와 다름 (발 링크 이름 확인)
- reset_strategy = "random" 유지 (RSI 없으면 학습 불안정)

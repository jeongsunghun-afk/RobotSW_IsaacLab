# Go2 AMP (Adversarial Motion Prior) 환경 컨텍스트

## 환경 개요
Go2에 AMP 모방학습 적용. Discriminator가 dog motion 데이터를 판별하여
자연스러운 보행을 유도. task reward와 AMP reward를 lerp로 결합.

## 등록 ID & 파일

| ID | 클래스 | 파일 |
|----|--------|------|
| Go2-AMP-v0 | Go2AmpEnv | `go2_amp_env.py` |

- 설정: `go2_amp_env_cfg.py` → `Go2AmpEnvCfg`
- 모션 로더: `go2_motion_loader.py`
- PPO 설정: `agents/rsl_rl_ppo_cfg.py`

## 로봇 스펙
- DOF: 12 (action_space = 12)
- termination_height: 0.15m
- action_scale: 0.25

## Observation 구성

### Policy obs (observation_space = 42)
```
projected_gravity_b(3) + commands(3) +
joint_pos_error(12) + joint_vel(12) + actions(12)
```

### AMP obs (amp_observation_space = 55)
```
dof_pos(12) + dof_vel(12) + root_height(1) +
lin_vel(3) + ang_vel(3) +
key_body_pos(12) + key_body_lin_vel(12)
```
**중요**: motion loader의 obs 추출 순서와 반드시 일치해야 함.

## Motion Data 구조
```
go2_amp/imitation/
└── new_dataset_walk/   ← 현재 사용 중 (저속 walk 데이터)
    └── D1_XXX.txt
```
**알려진 제약**: 현재 데이터 최대 속도 ~1.9 m/s. command 범위와 맞춰야 함.

## Command 구성
```python
command_cfg = {
    "lin_vel_x_range": [0.0, 1.5],   # reference data 최대 속도와 맞출 것
    "lin_vel_y_range": [-0.0, 0.0],
    "ang_vel_range":   [-0.5, 0.5],
}
```

## AMP 핵심 파라미터 (agents/rsl_rl_ppo_cfg.py)
```python
amp_cfg = {
    "task_reward_lerp": 0.5,     # 0.3~0.7 권장. 높을수록 task 편향
    "reward_coef": 2.0 * 0.02,  # AMP reward 스케일
    "reset_strategy": "random",  # RSI 사용 권장
}
```

## 알려진 이슈 (이전 분석 결과)
1. **속도 범위 미스매치**: command 최대 속도 > reference 최대 속도이면 고속에서 AMP 무효
2. **단일 gait 수렴**: 모든 reference가 같은 gait type이면 다양한 gait 불가
3. **normalizer 분포 불일치**: expert 분포만으로 normalizer 학습 → 고속 policy obs 비정규화

## 수정 시 주의사항
- AMP obs 크기 변경 시: `amp_discriminator.py`의 `input_dim`과 motion loader 동시 수정
- reset_strategy → "random" 또는 "random-start" 권장 (RSI로 학습 안정화)
- motion file 경로 변경 시: `go2_amp_env_cfg.py`의 `MOTION_FILES_DIR` 수정

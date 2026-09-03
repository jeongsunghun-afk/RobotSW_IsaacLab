# Go2 AMP (Adversarial Motion Prior) 환경 컨텍스트

## 환경 개요
Go2에 AMP 모방학습 적용. Discriminator가 dog motion 데이터를 판별하여
자연스러운 보행을 유도. task reward와 AMP reward를 lerp로 결합.

## 등록 ID & 파일
Task ID는 `Go2AMP`, `Go2AMP-Simple` (환경 클래스 `Go2AmpEnv`, `__init__.py`). 이 디렉토리에 README.md는 없다.

- 환경: `go2_amp_env.py`, 모션 라이브러리: `motion_lib.py`, 모션 데이터: `imitation/`
- agent cfg: `agents/rsl_rl_ppo_cfg.py` (rl_games·skrl용 yaml도 있으나 현재 학습 경로는 rsl_rl)

- 설정: `go2_amp_env_cfg.py` → `Go2AmpEnvCfg`, `Go2AmpSimpleEnvCfg`
- 모션 로더: `go2_motion_loader.py`
- Discriminator 구현: `rsl_rl/rsl_rl/modules/amp_discriminator.py`의 `AMPDiscriminator`(저장소 최상위 `rsl_rl` 패키지, 이 디렉토리 밖)
- PPO 설정: `agents/rsl_rl_ppo_cfg.py`

## 로봇 스펙
- DOF: 12 (action_space = 12)
- termination_height: 0.1m (Go2 기본 높이 0.34m 대비)
- action_scale: 0.25

## Observation 구성

### Policy obs (observation_space = 42)
```
projected_gravity_b(3) + commands(3) +
joint_pos_error(12) + joint_vel(12) + actions(12)
```

### AMP obs (amp_observation_space = 43, 단일 프레임)
```
dof_pos(12) + dof_vel(12) + root_height(1) +
lin_vel(3) + ang_vel(3) + key_body_pos(12)
```
`key_body_lin_vel` 항은 포함되지 않음.

## Motion Data 구조
- 기본 경로(`MOTION_FILES_DIR`): `imitation/go2/` — pkl 형식, 다중 gait(`go2_walk0~3.pkl`, `go2_trot.pkl`, `go2_pace.pkl`, `go2_run.pkl`)
- 대안 경로(주석 처리됨): `imitation/new_dataset*` (txt 형식, 현재 미사용)

## Command 구성
```python
command_cfg = {
    "lin_vel_x_range": [0.0, 3.0],
    "lin_vel_y_range": [-0.0, 0.0],
    "ang_vel_range":   [-0.5, 0.5],
}
```

## AMP 핵심 파라미터 (agents/rsl_rl_ppo_cfg.py, amp_cfg)
2단계 annealing 구조:
```python
task_reward_lerp = 1.0             # Stage 2 최종값
task_reward_lerp_start = 1.0       # Stage 1 초기값
task_reward_lerp_anneal_iters = 10000
reward_coef = 2.0
discriminator_learning_rate = 1e-4
gradient_penalty_coef = 5.0
```

## 알려진 이슈 (2026-09-03 코드 대조 시 미확인 — 다중 gait pkl 도입 이후 재현 여부 확인 안 됨)
- 속도 범위 미스매치: command 최대 속도가 reference 최대 속도를 넘으면 고속 구간에서 AMP reward 무효화 가능
- normalizer가 expert 분포로만 학습되면 고속 policy obs 정규화가 어긋날 수 있음

## 수정 시 주의사항
- AMP obs 크기 변경 시: `amp_discriminator.py`의 `input_dim`과 `go2_motion_loader.py` 동시 수정
- motion file 경로 변경 시: `go2_amp_env_cfg.py`의 `MOTION_FILES_DIR` 수정

## 학습·평가
학습·렌더 실행 방법은 `.claude/rules/training.md` 참조.

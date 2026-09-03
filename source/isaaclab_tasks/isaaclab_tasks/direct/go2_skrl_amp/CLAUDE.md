# go2_skrl_amp 환경 컨텍스트

## 개요

Go2 로봇에 skrl의 AMP(Adversarial Motion Prior) 알고리즘을 적용한 모방학습 환경.
rsl_rl 기반 go2_amp와 달리 **skrl 프레임워크**를 사용하며, 모션 데이터는 NPZ 형식으로 관리된다.

## 등록 정보
환경 ID: `Isaac-Go2-AMP-Direct-v0` (`__init__.py`). 이 디렉토리에 README.md는 없다.

- 환경: `go2_skrl_amp_env.py`, cfg: `go2_skrl_amp_env_cfg.py`, 모션 데이터: `motions/smr_mirror_npz/`
- agent cfg: `agents/skrl_amp_cfg.yaml`. 학습은 `scripts/reinforcement_learning/train.py --rl_library skrl` (실행 규칙은 `.claude/rules/training.md`).

## 관측 공간

### Policy obs (42-dim)
```
projected_gravity_b(3) + commands(3) +
joint_pos_error(12) + joint_vel(12) + actions(12)
```

### AMP obs (43-dim, 단일 프레임)
```
dof_pos(12) + dof_vel(12) + root_height(1) +
lin_vel_b(3) + ang_vel_b(3) + key_body_pos_local(12)
```
- `num_amp_observations = 2` → Discriminator 입력 = 86-dim (2프레임 연결)
- key bodies: FL_foot, FR_foot, RL_foot, RR_foot (각 3-dim = 12)

## 액션 / 시뮬레이션

| 항목 | 값 |
|------|-----|
| action_space | 12 (DOF) |
| action_scale | 0.25 |
| Physics freq | 200 Hz |
| Policy freq | 50 Hz (decimation=4) |
| Episode length | 20 s |

## 조기 종료 조건

- `root_height < termination_height`(0.1m)
- `|roll| > roll_termination_deg`(70°) 또는 `|pitch| > pitch_termination_deg`(70°)

## 커맨드 범위

```python
lin_vel_x_range: [0.0, 5.0]  # m/s
lin_vel_y_range: [0.0, 0.0]
ang_vel_range:   [-1.0, 1.0] # rad/s
```

## 모션 데이터

- 기본 경로(`MOTIONS_DIR`): `motions/smr_mirror_npz/` (NPZ 형식) — 이 디렉토리만 실제 존재함
- `mimickit_npz/`, `npz_dataset/` 등 다른 경로는 주석 처리되어 있고 디렉토리 자체가 없음
- NPZ body 규약: `["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]`

## 보상 구조

`_get_rewards`의 항 이름: `lin_vel tracking`, `yaw_rate tracking` (가중치는 `go2_skrl_amp_env_cfg.py`의 `lin_vel_reward_scale`/`yaw_rate_reward_scale` 참조). AMP style reward는 Discriminator 출력.

- AMP task/style 비율: `skrl_amp_cfg.yaml`의 `task_reward_scale` / `style_reward_scale`

## 불변 규칙

- `observation_space = 42`, `amp_observation_space = 43` — 변경 시 yaml 모델 input도 동기화
- 새 버퍼 추가 시 `_reset_idx`에서 초기화 필수
- config 변경 → `go2_skrl_amp_env_cfg.py`, 로직 변경 → `go2_skrl_amp_env.py`

## 학습·평가
학습·렌더 실행 방법은 `.claude/rules/training.md` 참조.

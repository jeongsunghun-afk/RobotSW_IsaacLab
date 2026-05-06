# go2_skrl_amp 환경 컨텍스트

## 개요

Go2 로봇에 skrl의 AMP(Adversarial Motion Prior) 알고리즘을 적용한 모방학습 환경.
rsl_rl 기반 go2_amp와 달리 **skrl 프레임워크**를 사용하며, 모션 데이터는 NPZ 형식으로 관리된다.

## 등록 정보

| 항목 | 값 |
|------|-----|
| 환경 ID | `Isaac-Go2-AMP-Direct-v0` |
| 환경 클래스 | `Go2SkrlAmpEnv` |
| 설정 클래스 | `Go2SkrlAmpEnvCfg` |
| 등록 파일 | `__init__.py` |
| Agent 설정 | `agents/skrl_amp_cfg.yaml` |
| Motion loader | `motions/motion_loader.py` |

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

- `root_height < 0.1 m`
- `|roll| > 70°` 또는 `|pitch| > 70°`

## 커맨드 범위

```python
lin_vel_x_range: [0.0, 4.0]  # m/s
lin_vel_y_range: [0.0, 0.0]
ang_vel_range:   [-1.0, 1.0] # rad/s
```

## 모션 데이터

- **현재 사용**: `motions/mimickit_npz/` (NPZ 형식)
- 대안 경로: `motions/npz_dataset/`, `motions/smr_mirror_small_npz/`
- NPZ body 규약: `["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]`
- TXT → NPZ 변환:
  ```bash
  ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation/imitation/convert_smr_to_npz.py
  ```

## 보상 구조

| 보상 항목 | 스케일 | 설명 |
|----------|--------|------|
| lin_vel tracking | 1.0/0.02 | x 속도 추종 |
| yaw_rate tracking | 0.5/0.02 | yaw 속도 추종 |
| AMP style reward | skrl_amp_cfg.yaml에서 조정 | Discriminator 출력 |

- AMP task/style 비율: `skrl_amp_cfg.yaml`의 `task_reward_scale` / `style_reward_scale`

## 학습 명령어

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task Isaac-Go2-AMP-Direct-v0 --num_envs 4096

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py \
  --task Isaac-Go2-AMP-Direct-v0 --num_envs 32
```

## Worker 매핑

| 작업 | 담당 Worker |
|------|------------|
| 관측/보상/환경 로직 변경 | `obs-worker`, `reward-worker` |
| 설정값 변경 (`Go2SkrlAmpEnvCfg`) | `cfg-worker` |
| Actor/Critic/Discriminator 네트워크 | `network-worker` |
| `skrl_amp_cfg.yaml` 하이퍼파라미터 | `hyperparam-worker` |
| 디버그 (AMP/IL 계열) | `il-debug-worker` |
| 모션 데이터 통계 분석 | `motion-analyzer` |

## 불변 규칙

- `observation_space = 42`, `amp_observation_space = 43` — 변경 시 yaml 모델 input도 동기화
- 새 버퍼 추가 시 `_reset_idx`에서 초기화 필수
- config 변경 → `go2_skrl_amp_env_cfg.py`, 로직 변경 → `go2_skrl_amp_env.py`

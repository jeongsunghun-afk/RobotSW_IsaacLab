# Go2 Imitation Tracking 환경

`go2_imitation`(AMP 모방학습)에서 파생된 환경으로, command 구조를 **steering(tar_dir·tar_speed·face_dir)**에서
**body-frame 속도추종(vx, vy=0, yaw_rate)**으로 교체한 버전.
AMP discriminator / motion / reset 로직은 `go2_imitation`과 동일하게 유지된다.

---

## 등록 Task 목록

| Task ID | 러너 / 알고리즘 | 설명 |
|---------|----------------|------|
| `Go2-Imitation-Tracking-v0` | `OnPolicyRunnerAMPBase` + `ActorCritic` + `PPOAMPBase` | AMP 모방학습 + body-frame 속도추종 |

---

## 실행 명령어

> **conda 환경**: `isaac-5.1` (기본)
>
> `OnPolicyRunnerAMPBase` 러너는 agent cfg의 `class_name`으로 자동 선택 — 표준 `train.py`로 실행 가능.

### 학습

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --logger wandb --wandb-project IsaacLab-locomotion
```

### 평가

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2-Imitation-Tracking-v0 --num_envs 32
```

### 학습 결과 시각화 (커스텀 plot 스크립트)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/go2_imitation_tracking_plot.py \
  --task Go2-Imitation-Tracking-v0
```

---

## 환경 스펙

| 항목 | 값 |
|------|-----|
| 로봇 | Unitree Go2 (12-DOF) |
| Physics Hz | 200 Hz |
| Policy Hz | 50 Hz (decimation=4) |
| Episode 길이 | 10 s |
| `observation_space` | **48** |
| `action_space` | 12 |
| AMP obs (per-step) | 49-dim |
| AMP history depth | 10 step |
| `amp_observation_space` | **490** (= 49 × 10) |

---

## Observation Space (48-dim)

| idx | 성분 | dim |
|-----|------|-----|
| 0–2 | `root_lin_vel_b` | 3 |
| 3–5 | `root_ang_vel_b` | 3 |
| 6–8 | `projected_gravity_b` | 3 |
| 9–10 | `lin_vel_cmd` (vx, vy) | 2 |
| 11 | `yaw_vel_cmd` | 1 |
| 12–23 | `joint_pos - default` | 12 |
| 24–35 | `joint_vel` | 12 |
| 36–47 | `actions` | 12 |

> AMP observation (490-dim) = dof_pos(12)+dof_vel(12)+root_height(1)+root_lin_vel(3)+root_ang_vel(3)+foot_pos_local(12)+root_rot_tan_norm(6) × 10 step. **절대 수정 금지.**

---

## Command 범위

| 명령 | 범위 | 단위 |
|------|------|------|
| `vx` (전진 속도) | [0.0, 4.0] | m/s |
| `vy` (횡방향) | 0.0 (고정) | m/s |
| `yaw_rate` (회전속도) | [−1.5, 1.5] | rad/s |
| 재샘플링 주기 | [4.0, 7.0] | s |

---

## 보상 함수

```
reward = 0.7 × exp(-vel_err_scale × ‖lin_vel_cmd − lin_vel_b‖²)
       + 0.3 × exp(-yaw_vel_err_scale × (yaw_vel_cmd − yaw_vel_b)²)
```

AMP discriminator reward는 Stage 1(0~5000 iter: 순수 task)→Stage 2(5000~ iter: task 50% + AMP 50%) 스케줄로 혼합.

---

## go2_imitation 대비 변경점

| 항목 | go2_imitation | go2_imitation_tracking |
|------|--------------|------------------------|
| command 타입 | steering (tar_dir + tar_speed + face_dir) | body-frame 속도 (vx, vy=0, yaw_rate) |
| obs command block | `local_tar_dir(2)+tar_speed(1)+local_face_dir(2)` = 5 | `lin_vel_cmd(2)+yaw_vel_cmd(1)` = 3 |
| obs 총 차원 | 50 | **48** |
| reward | tar_reward + face_reward | lin_vel_reward + yaw_vel_reward |
| AMP obs / disc 구조 | 동일 | 동일 (불변) |

---

## 파일 구조

```
go2_imitation_tracking/
├── CLAUDE.md
├── __init__.py                         ← task 등록 (Go2-Imitation-Tracking-v0)
├── go2_imitation_tracking_env.py       ← 환경 본체
├── go2_imitation_tracking_env_cfg.py   ← 환경 config
├── motion_lib.py                       ← go2_imitation/motion_lib.py byte-identical 복사
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py              ← Go2ImitationTrackingPPORunnerCfg
└── imitation/
    └── smr_mirror_pkl -> ../../go2_imitation/imitation/smr_mirror_pkl  (심링크)
```

---

## 수정 가이드

| 수정 항목 | 파일 | 주의사항 |
|-----------|------|---------|
| reward 가중치 | `go2_imitation_tracking_env_cfg.py` | `lin_vel_reward_w + yaw_vel_reward_w` 합계 권장 = 1.0 |
| 명령 범위 | `go2_imitation_tracking_env_cfg.py` | `lin_vel_x_min/max`, `yaw_vel_min/max` |
| AMP anneal 일정 | `agents/rsl_rl_ppo_cfg.py` | `task_reward_lerp_anneal_iters` (현재 5000 iter) |
| 새 텐서 추가 | `go2_imitation_tracking_env.py` | `_reset_idx`에서 반드시 초기화 |
| motion 데이터 | `imitation/smr_mirror_pkl/*.pkl` | 심링크 — go2_imitation과 공유 |

### 절대 수정 금지

- `motion_lib.py` — go2_imitation byte-identical 유지
- AMP observation 관련 코드 — 490-dim (49×10) 불변
- `imitation/smr_mirror_pkl` 심링크 — 경로 구조 유지

# Go2 Imitation Tracking 환경 컨텍스트

## 개요

`go2_imitation`에서 파생된 환경으로, command 구조를 **steering(tar_dir·tar_speed·face_dir)**에서
**body-frame 속도추종(vx, vy=0, yaw_rate)**으로 교체한 버전이다.
AMP/discriminator/motion/reset 로직은 go2_imitation과 byte-identical 유지.

**Task ID**: `Go2-Imitation-Tracking-v0`

---

## 파일 구조

```
go2_imitation_tracking/
├── __init__.py                         ← task 등록 (Go2-Imitation-Tracking-v0)
├── go2_imitation_tracking_env.py       ← 환경 본체
├── go2_imitation_tracking_env_cfg.py   ← 환경 config
├── motion_lib.py                       ← go2_imitation/motion_lib.py byte-identical 복사
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py              ← PPO/AMP 러너 config
└── imitation/
    └── smr_mirror_pkl -> ../../go2_imitation/imitation/smr_mirror_pkl  (심링크)
```

---

## go2_imitation과의 차이점

| 항목 | go2_imitation | go2_imitation_tracking |
|------|---------------|------------------------|
| command 타입 | steering (tar_dir + tar_speed + face_dir) | body-frame 속도 (vx, vy=0, yaw_rate) |
| obs command block | `local_tar_dir(2) + tar_speed(1) + local_face_dir(2)` = 5 | `lin_vel_cmd(2) + yaw_vel_cmd(1)` = 3 |
| obs 총 차원 | 50 | **48** |
| command 변환 | `quat_apply` (heading relative) | 변환 없음 (이미 body-relative) |
| reward | tar_reward + face_reward | lin_vel_reward + yaw_vel_reward |
| AMP 경로 | 동일 | 동일 (불변) |

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
| **합계** | | **48** |

- command 변환(quat_apply) 제거. 버퍼를 그대로 concat.
- AMP observation (490-dim = 43×10 history) 은 go2_imitation과 동일, 절대 수정 금지.

---

## Command 버퍼

```python
self._lin_vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, vy) — vy 항상 0
self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)     # yaw rate [rad/s]
self._tar_timer   = torch.zeros(self.num_envs, device=self.device)     # 재샘플링 타이머
```

범위 (cfg 참조):
- `lin_vel_x`: [-1.0, 3.0] m/s
- `lin_vel_y`: [0.0, 0.0] (항상 0)
- `yaw_vel`: [-1.5, 1.5] rad/s
- `tar_timer`: [4.0, 7.0] s

---

## Reward 함수

```python
# lin_vel tracking (body frame 직접 비교)
lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]
lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
lin_vel_reward = torch.exp(-cfg.vel_err_scale * lin_vel_err)

# yaw_vel tracking
yaw_vel_b = self._robot.data.root_ang_vel_b[:, 2]
yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
yaw_vel_reward = torch.exp(-cfg.yaw_vel_err_scale * yaw_vel_err)

reward = cfg.lin_vel_reward_w * lin_vel_reward + cfg.yaw_vel_reward_w * yaw_vel_reward
# 기본: 0.7 * lin_vel_reward + 0.3 * yaw_vel_reward
```

---

## 알고리즘 설정

go2_imitation과 동일 (AMP dict / 네트워크 / 하이퍼파라미터 값 동일).
- `experiment_name = "go2_imitation_tracking"`
- WASABI cfg 없음

---

## 학습 실행

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --logger wandb --wandb-project IsaacLab-locomotion

# 플레이
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2-Imitation-Tracking-v0 --num_envs 32
```

---

## 수정 가이드

| 수정 항목 | 파일 | 주의사항 |
|-----------|------|---------|
| reward 가중치 | `go2_imitation_tracking_env_cfg.py` | `lin_vel_reward_w + yaw_vel_reward_w` 합계 권장 = 1.0 |
| 명령 범위 | `go2_imitation_tracking_env_cfg.py` | `lin_vel_x_min/max`, `yaw_vel_min/max` |
| Termination 임계값 | `go2_imitation_tracking_env_cfg.py` | go2_imitation과 동일 구조 |
| AMP anneal 일정 | `agents/rsl_rl_ppo_cfg.py` | go2_imitation과 동일 |
| 새 텐서 추가 | `go2_imitation_tracking_env.py` | `_reset_idx`에서 반드시 초기화 |
| motion 데이터 | `imitation/smr_mirror_pkl/*.pkl` | 심링크 통해 자동 반영 |

## 절대 수정 금지

- `motion_lib.py` — go2_imitation byte-identical 유지
- AMP observation 관련 코드 — 490-dim (43×10) 불변
- `imitation/smr_mirror_pkl` 심링크 — 경로 구조 유지

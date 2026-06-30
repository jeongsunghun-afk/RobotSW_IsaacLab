# Go2 Imitation 환경 컨텍스트

## 개요

MimicKit의 `TaskSteeringEnv` 구조를 IsaacLab `DirectRLEnv`에 맞게 재구현한 환경.
기존 `go2_amp`에서 학습이 되지 않는 문제를 해결하기 위해 처음부터 재구현했다.

**Task ID**: `Go2-Imitation-v0`

---

## 파일 구조

```
go2_imitation/
├── __init__.py                  ← task 등록 (Go2-Imitation-v0)
├── go2_imitation_env.py         ← 환경 본체
├── go2_imitation_env_cfg.py     ← 환경 config
├── motion_lib.py                ← PKL 모션 라이브러리 (MimicKit-style)
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py       ← PPO/AMP 러너 config
└── imitation/go2/               ← 참조 모션 PKL 파일 (7개)
    ├── go2_pace.pkl
    ├── go2_run.pkl
    ├── go2_trot.pkl
    └── go2_walk0~3.pkl
```

---

## 기존 go2_amp와의 차이점

| 항목 | go2_amp | go2_imitation |
|------|---------|---------------|
| Task reward | lin/ang_vel MSE tracking | `exp(-scale‖Δv‖²)` + face direction |
| AMP 비중 | `task_reward_lerp=1.0` (AMP 0%) | Stage 1→2 anneal (1.0→0.5) |
| Reset 전략 | default + random 혼합 | 항상 RSI (Reference State Init) |
| motion_lib | Go2MotionLoader (전체 concat) | Go2MotionLib (per-motion 분리) |
| Steering | X/Y vel + yaw command | tar_dir + tar_speed + face_dir |

---

## Observation Space

### Policy Observation (44-dim)

| 요소 | 차원 | 설명 |
|------|------|------|
| `projected_gravity_b` | 3 | body frame 중력 벡터 |
| `local_tar_dir` | 2 | heading-relative 목표 방향 (단위 벡터) |
| `tar_speed` | 1 | 목표 속도 (m/s) |
| `local_face_dir` | 2 | heading-relative 얼굴 방향 |
| `joint_pos - default` | 12 | 관절 각도 오프셋 |
| `joint_vel` | 12 | 관절 속도 |
| `actions` | 12 | 이전 액션 |
| **합계** | **44** | |

### AMP Discriminator Observation (43-dim × 10 history = 430-dim)

| 요소 | 차원 | 설명 |
|------|------|------|
| `dof_pos` | 12 | 관절 각도 |
| `dof_vel` | 12 | 관절 속도 |
| `root_height` | 1 | root z 위치 |
| `root_lin_vel` | 3 | body frame 선속도 |
| `root_ang_vel` | 3 | body frame 각속도 |
| `foot_pos_local` | 12 | 발 위치 (base-local, [FL,FR,RL,RR]×3) |
| **합계** | **43** | per step |

---

## Reward 함수

```
reward = tar_reward_w * tar_reward + face_reward_w * face_reward
       = 0.7 * tar_reward + 0.3 * face_reward
```

### tar_reward (목표 속도 추종)

```python
tar_vel = tar_speed * tar_dir                         # [N,2] 목표 속도 벡터
root_vel_xy = (pos_now - pos_prev)[:,:2] / step_dt   # 실제 xy 속도
tar_vel_err = sum((tar_vel - root_vel_xy)^2, dim=-1)
tar_reward  = exp(-vel_err_scale * tar_vel_err)       # vel_err_scale=0.5
# 목표 방향 반대로 가면 0
tar_reward[dot(tar_dir, root_vel_xy) < 0] = 0
```

### face_reward (방향 정렬)

```python
heading_rot = yaw_only_quat(root_quat)
char_fwd    = quat_apply(heading_rot, [1,0,0])[:,:2]  # 캐릭터 전진 방향
face_reward = max(0, dot(face_dir, char_fwd))          # [0,1]
```

**AMP Discriminator reward** (Runner가 자동 계산):
```
amp_reward = clamp(1 - 0.25*(logit-1)^2, min=0) * reward_coef(2.0)
total_reward = lerp * task_reward + (1-lerp) * amp_reward
             → lerp: 1.0(Stage1) → 0.5(Stage2, 5000 iter anneal)
```

---

## Steering Task

- **목표 방향** (`tar_dir`): 랜덤 각도 [-π, π], 단위 벡터
- **목표 속도** (`tar_speed`): 균등 분포 [0.5, 3.0] m/s
- **얼굴 방향** (`face_dir`): 목표 방향과 동일 (rand_face_dir=False)
- **방향 변경 주기**: [4.0, 7.0] s마다 자동 재샘플링

---

## Reset 전략 (항상 RSI)

```
motion_ids = sample_motions(n)         # 가중치 비례 모션 선택
times      = sample_times(motion_ids)  # 균등 [0, motion_length]
root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos
         = calc_motion_frame(motion_ids, times)
→ root_state = pos + env_origin, quat, world-frame vel
→ joint_pos/vel 초기화
→ amp_observation_buffer = RSI 시점의 reference motion obs
```

---

## Termination 조건

| 조건 | 임계값 |
|------|--------|
| base 높이 | < 0.15 m |
| projected gravity z | > 0 (뒤집힘) |
| roll | > 70° |
| pitch | > 70° |
| base contact force | > 500 N |
| timeout | 10 s (500 steps @ 50 Hz) |

*첫 1 step은 physics 정착을 위해 termination 제외*

---

## 알고리즘 설정 (rsl_rl_ppo_cfg.py)

```
Runner: OnPolicyRunnerAMPBase
Policy: ActorCritic (MLP 512→256→128)
Algo:   PPOAMPBase

AMP 파라미터:
  task_reward_lerp_start = 1.0   (Stage 1: 100% task)
  task_reward_lerp       = 0.5   (Stage 2: 50% task + 50% AMP)
  anneal_iters           = 5000  (= 120,000 steps)
  disc_lr                = 2.5e-4
  gradient_penalty_coef  = 5.0
  reward_coef            = 2.0
  disc_num_epochs        = 2
  disc_logit_reg         = 0.01
  replay_buffer_size     = 200,000
```

---

## 학습 실행

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-v0 --num_envs 4096 --headless \
  --logger wandb --wandb-project IsaacLab-locomotion

# 플레이
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2-Imitation-v0 --num_envs 32
```

---

## motion_lib.py 인터페이스

```python
lib = Go2MotionLib(motion_files="imitation/go2", device="cuda:0")

# 샘플링
motion_ids = lib.sample_motions(n)          # [n] int64
times      = lib.sample_times(motion_ids)   # [n] float32

# 프레임 계산
root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos = \
    lib.calc_motion_frame(motion_ids, times)
# root_pos      [N,3]    world frame
# root_quat     [N,4]    wxyz
# lin_vel/ang_vel [N,3]  body frame
# dof_pos/vel   [N,12]
# foot_pos      [N,4,3]  base-local ([FL,FR,RL,RR])
```

**PKL 프레임 레이아웃** (18개 값):
- `[0:3]` root_pos, `[3:6]` root_euler (rpy), `[6:18]` dof_pos (12 joints)
- 속도/발 위치: finite difference + FK 자동 계산

---

## 수정 가이드

| 수정 항목 | 파일 | 주의사항 |
|-----------|------|---------|
| reward 가중치 | `go2_imitation_env_cfg.py` | `tar_reward_w + face_reward_w` 합계 권장 = 1.0 |
| 목표 속도 범위 | `go2_imitation_env_cfg.py` | `tar_speed_min/max` |
| Termination 임계값 | `go2_imitation_env_cfg.py` | height/roll/pitch/contact |
| AMP anneal 일정 | `agents/rsl_rl_ppo_cfg.py` | `task_reward_lerp_anneal_iters` |
| Disc 구조 | `agents/rsl_rl_ppo_cfg.py` | `discriminator_hidden_dims` |
| 새 텐서 추가 | `go2_imitation_env.py` | `_reset_idx`에서 반드시 초기화 |
| motion 데이터 | `imitation/go2/*.pkl` | 18-col frames 포맷 |

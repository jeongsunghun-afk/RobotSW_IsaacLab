# Go2 Fall-Recovery (go2_recovery)

Go2 사족보행 로봇이 **임의 자세로 넘어진 상태에서 일어나는 복구 동작**을 학습하는 환경.
HumanUP 방식과 같이 선속도 없는 고유감각(proprioception) 전용 obs로 설계되어 sim-to-real 전이를 고려한다.

## 주요 특징

| 항목 | 내용 |
|------|------|
| 로봇 | Unitree Go2 (12 DOF) |
| 알고리즘 | PPO (OnPolicyRunner, 표준) |
| Observation | 42 dim (선속도 제외 — sim-to-real) |
| Action | 12 dim joint position target |
| Episode 길이 | 10 s (복구 목표 6~8 s + 여유) |
| 물리/정책 주기 | 200 Hz physics / 50 Hz policy (decimation=4) |
| 지형 | Flat plane |
| Fall 초기화 | 80% fallen / 10% standing / 10% sitting |

### Observation 구성 (42 dim)

```
root_ang_vel_b      [3]   × ang_vel_scale=0.25
projected_gravity_b [3]   (no scale)
joint_pos_error     [12]  (joint_pos - default) × dof_pos_scale=1.0
joint_vel           [12]  × dof_vel_scale=0.05
previous_actions    [12]  (no scale)
─────────────────────────
TOTAL               42
```

### Fall 초기화 분포

| 그룹 | 비율 | 초기화 방식 |
|------|------|-------------|
| fallen | 80% | random roll ±135°, pitch ±45°, yaw ±180° + random joint (cubic lerp) + z+0.45 m 공중낙하 |
| standing | 10% | default pose, z=0.27 m |
| sitting | 10% | calf 추가 굽힘, z-0.10 m, yaw 랜덤 |

### Actuator (Tier-0 안전 설정)

```
effort_limit  23.5 Nm
velocity_limit 18.0 rad/s  (30→18, 60% 억제)
stiffness     25.0
damping        1.0          (0.5→1.0, 2× 강화)
```

### 종료 조건

- `terminated`: base_z < -0.1 m
- `time_out`: episode 길이 초과
- `success` (terminate_on_success=True): upright 자세를 20 step 연속 유지 → 조기 종료 + success_bonus

---

## 등록된 Task

| Task ID | 러너 클래스 | CFG 클래스 |
|---------|------------|-----------|
| `Go2Recovery-v0` | `OnPolicyRunner` (표준) | `Go2RecoveryPPORunnerCfg` |

---

## 실행 명령어

> conda 환경: `isaac-5.1`

### 학습 (train)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Recovery-v0 --num_envs 4096
```

WandB 로깅 포함:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Recovery-v0 --num_envs 4096 \
  --logger wandb --wandb-project IsaacLab-Go2Recovery
```

최대 iteration 지정:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Recovery-v0 --num_envs 4096 --max_iterations 10000
```

### 평가 (play)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2Recovery-v0 --num_envs 32
```

체크포인트 직접 지정:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2Recovery-v0 --num_envs 32 \
  --load_run <experiment_name> --checkpoint model_<iter>.pt
```

> 로그/체크포인트 저장 위치: `logs/rsl_rl/go2_recovery_direct/`

---

## BLOCKING 검증 게이트

| 확인 항목 | 결과 |
|-----------|------|
| `rsl_rl_cfg_entry_point` 존재 | ✅ `agents/rsl_rl_ppo_cfg:Go2RecoveryPPORunnerCfg` |
| runner `class_name` | ✅ 기본값 `"OnPolicyRunner"` — 표준 train.py/play.py 바로 사용 가능 |
| 커스텀 play 스크립트 필요 여부 | ✅ 불필요 — 표준 `play.py` 사용 |

---

## 주요 파일

```
go2_recovery/
├── go2_recovery_env.py         # 환경 구현 (Go2RecoveryEnv)
├── go2_recovery_env_cfg.py     # 환경/보상 파라미터 (Go2RecoveryEnvCfg)
├── agents/
│   └── rsl_rl_ppo_cfg.py       # PPO 러너 cfg (Go2RecoveryPPORunnerCfg)
└── CLAUDE.md                   # 개발 컨텍스트 (이정표 / 스펙)
```

---

## 개발 로드맵

| 마일스톤 | 상태 | 내용 |
|---------|------|------|
| M1 | ✅ 완료 | env 골격 + fall 초기화 (80/10/10) + placeholder reward |
| M2 | ✅ 완료 | 복구 보상 (`r_roll + r_stand`) + smoothness penalty + success 판정 |
| M3 | 예정 | penalty curriculum + 안전 지표 + 장기 학습 |

# Go2 보행 환경

Unitree Go2 사족보행 로봇의 **강건 보행 + WTW(Walk-The-Walk) 자연보행 + 상호작용** 학습 환경 모음.
순수 Reward Shaping(PPO) 기반이며, AMP 연동 목적으로 등록된 task도 포함한다.

---

## 등록 Task 목록

| Task ID | 환경 클래스 | 러너 / 알고리즘 | 설명 |
|---------|------------|----------------|------|
| `Go2` | `Go2Env` | `RslRlOnPolicyRunner` (표준 PPO) | 평지 속도추종 기본 보행 |
| `Go2WTW` | `WTWEnv` | `OnPolicyRunnerParkour` + `ActorCriticRMA` + `PPOParkour` | Bezier 발궤적 + Raibert 휴리스틱 자연보행 |
| `Go2Rough` | `Go2Env` | `RslRlOnPolicyRunner` (표준 PPO) | 험지(rough terrain) 속도추종 보행 |
| `Go2Neck` | `Go2NeckEnv` | `OnPolicyRunnerParkour` + `ActorCriticRMA` + `PPOParkour` | 목 모듈 부착 Go2의 WTW 보행 |
| `Go2Interaction` | `Go2InteractionEnv` | `OnPolicyRunnerParkour` + `ActorCriticRMA` + `PPOParkour` | 사물 상호작용 (RMA 구조, 에피소드 4s) |
| `Go2NeckInteraction` | `Go2NeckInteractionEnv` | `OnPolicyRunnerParkour` + `ActorCriticRMA` + `PPOParkour` | 목 모듈 부착 + 사물 상호작용 |
| `Go2-AMP-v0` | `Go2AmpEnv` | `OnPolicyRunnerAMP` + `PPOAMP` | ⚠️ **실행 불가 — 구현 미완 (하단 참조)** |

---

## 실행 명령어

> **conda 환경**: `isaac-5.1` (기본)

### Go2 (평지 기본 보행)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2 --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2 --num_envs 32
```

### Go2WTW (자연보행 — Bezier + Raibert)

```bash
# 학습 (OnPolicyRunnerParkour + PPOParkour, ActorCriticRMA 자동 선택)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2WTW --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2WTW --num_envs 32
```

### Go2Rough (험지 보행)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Rough --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2Rough --num_envs 32
```

### Go2Neck (목 모듈 부착 WTW)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Neck --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2Neck --num_envs 32
```

### Go2Interaction (사물 상호작용)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2Interaction --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2Interaction --num_envs 32
```

### Go2NeckInteraction (목 모듈 + 상호작용)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2NeckInteraction --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Go2NeckInteraction --num_envs 32
```

---

## ⚠️ Go2-AMP-v0 — 실행 불가 (구현 미완)

`__init__.py`에 등록은 되어 있으나 구현 파일이 존재하지 않아 **import 시 즉시 오류 발생**한다.

- 참조하는 파일: `go2_amp_env.py` (클래스 `Go2AmpEnv`), `go2_amp_env_cfg.py` (클래스 `Go2AmpEnvCfg`)
- **현황**: 두 파일 모두 디렉토리에 미존재
- **실행 시도 시**: `ModuleNotFoundError` 또는 `AttributeError`

AMP 기반 Go2 모방학습은 별도 환경인 [`go2_imitation_tracking`](../go2_imitation_tracking/README.md)를 사용할 것.

---

## 환경 스펙 요약

| 항목 | 값 |
|------|-----|
| 로봇 | Unitree Go2 (12-DOF) |
| Physics Hz | 200 Hz |
| Policy Hz | 50 Hz (decimation=4) |
| Episode 길이 | 20s (Interaction: 4s) |
| action_scale | 0.25 |
| Obs (Go2/Rough) | 42+ (history 포함 시 증가) |

### 보상 구조 (Go2 / Go2WTW 공통)

| 항목 | 방향 | 설명 |
|------|------|------|
| `track_lin_vel_xy_exp` | 양수 | 선속도 추종 (메인 보상) |
| `track_ang_vel_z_exp` | 양수 | 각속도 추종 |
| `raibert_heuristic` | 페널티 | 발 배치 패턴 유도 |
| `bezier_curve_reward` | 페널티 | 발 궤적 (WTW) |
| `action_rate`, `joint_vel` 등 | 페널티 | 부드러움 유지 |

---

## 파일 구조

```
go2/
├── CLAUDE.md
├── __init__.py                     ← task 등록 (7개)
├── go2_env.py                      ← Go2Env (Go2 / Go2Rough)
├── go2_env_cfg.py                  ← Go2FlatEnvCfg / Go2RoughEnvCfg
├── go2_wtw_env.py                  ← WTWEnv (Go2WTW)
├── go2_neck_env.py                 ← Go2NeckEnv (Go2Neck)
├── go2_neck_interaction_env.py     ← Go2NeckInteractionEnv
├── go2_neck_interaction_cfg.py
├── go2_interaction_env.py          ← Go2InteractionEnv
├── go2_interaction_cfg.py
├── go2_motion_loader.py
├── agents/
│   └── rsl_rl_ppo_cfg.py           ← 모든 runner cfg 정의
└── imitation/
    └── txt_dataset_go2_stmr/       ← 모션 데이터 (참고용)
```

---

## 수정 가이드

- **보상 추가**: `go2_env_cfg.py`에 weight 파라미터 선언 → `go2_env.py`에 로직 추가
- **새 buffer**: `_reset_idx`에서 반드시 초기화
- **Neck 확장**: `go2_neck_env.py` 참조 (DOF 수 변경됨)
- **알려진 이슈**: `bezier_curve_reward`의 world-frame latch 문제 — 회전 중 방향 틀어짐 (`CLAUDE.md` 참조)

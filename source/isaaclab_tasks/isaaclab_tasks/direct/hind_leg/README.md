# Hind-Leg Biped Locomotion (hind_leg)

RGA Inc 자체 제작 **2족(biped) 8-DOF 로봇**의 보행 제어를 학습하는 환경.  
Parkour와 별개로 존재하는 biped 전용 환경이며, **RMA(Rapid Motor Adaptation)** 아키텍처를 기본으로 채택한다.  
Gait clock 신호로 교대 보행을 유도하며, 속도 명령(vx, yaw_rate) 추종을 목표로 한다.

> ⚠️ parkour 환경의 제약(sim-to-real obs 제한, contact sensor 금지 등)은 이 환경에 적용하지 않는다.

## 주요 특징

| 항목 | 내용 |
|------|------|
| 로봇 | HIND_LEG (RGA Inc 커스텀 biped, 8 DOF) |
| 알고리즘 | PPOParkour + ActorCriticRMA (RMA 아키텍처) |
| 러너 | `OnPolicyRunnerParkour` (표준 train.py/play.py 지원) |
| Policy obs | 34 dim (기본 + gait clock 4 dim) |
| Action | 8 dim joint position target |
| Episode 길이 | 20 s |
| 물리/정책 주기 | 200 Hz physics / 50 Hz policy (decimation=4) |
| 지형 | Flat plane |
| 명령 | vx [-0.5, 2.0] m/s, vy=0, yaw [-0.5, 0.5] rad/s |

### RMA 아키텍처 (HindLegHistoryEnvCfg 기본)

| Obs 그룹 | 차원 | 내용 |
|----------|------|------|
| `policy` | 34 | 기본 고유감각 + gait clock 4 dim (actor 입력) |
| `priv_explicit` | 6 | 명시적 특권 정보 (critic만 사용) |
| `priv_latent` | 5 | latent 특권 정보 (critic + estimator 타겟) |
| `history` | 34×10 | 10-step 고유감각 history (estimator 입력) |

**Estimator**: history(340) → priv_latent(5) 추정. 배포 시 실제 특권 정보 없이 추정값으로 동작.

### Policy Obs 구성 (34 dim)

```
ang_vel_b           [3]   base angular velocity
projected_gravity   [3]   gravity in body frame
joint_pos_error     [8]   (joint_pos - default) * scale
joint_vel           [8]   joint velocities * scale
previous_actions    [8]   previous actions
gait_clock          [4]   sin/cos of gait phase (FL/FR)
─────────────────────
TOTAL               34
```

### Gait Clock 설정

```
gait_period         0.6 s       (전체 보행 주기)
gait_swing_height   0.07 m      (스윙 최대 발 높이)
gait_phase_sharpness 4.0        (stance/swing 전환 급격도)
rel_standing_envs   0.1         (10% env를 cmd=0으로 고정 — standing 학습)
```

### Domain Randomization (startup + reset)

| 항목 | 범위 | 모드 |
|------|------|------|
| 마찰 계수 | static [0.4, 1.5], dynamic [0.3, 1.2] | startup |
| 기저부 질량 | ±[-1.0, 3.0] kg | startup |
| Actuator 강성/감쇠 | ×[0.75, 1.5] (log-uniform) | reset |
| CoM 위치 | x±0.08, y±0.04, z±0.02 m | startup |
| 외부 충격(push) | vx/vy ±0.5 m/s, 4~8 s 간격 | interval |

---

## 등록된 Task

| Task ID | 러너 클래스 | CFG 클래스 | Runner Cfg |
|---------|------------|-----------|------------|
| `HindLeg-Direct-v0` | `OnPolicyRunnerParkour` | `HindLegHistoryEnvCfg` | `HindLegParkourPPORunnerCfg` |

> `__init__.py`에 rl_games / skrl 백엔드 entry_point도 함께 등록되어 있다.  
> 기본 rsl_rl 러너는 `HindLegParkourPPORunnerCfg` (RMA 아키텍처).

---

## 실행 명령어

> conda 환경: `isaac-5.1`

### 학습 (train)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task HindLeg-Direct-v0 --num_envs 4096
```

WandB 로깅 포함:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task HindLeg-Direct-v0 --num_envs 4096 \
  --logger wandb --wandb-project IsaacLab-HindLeg
```

최대 iteration 지정:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task HindLeg-Direct-v0 --num_envs 4096 --max_iterations 50000
```

### 평가 — 권장: `play_hind_leg.py` (카메라 추적 + 관절 플롯)

`play_hind_leg.py`는 HindLeg 전용 스크립트로 카메라 base 추적 + 관절 위치/속도/토크 시각화를 제공한다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_hind_leg.py \
  --task HindLeg-Direct-v0 --num_envs 1
```

동영상 녹화 포함:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_hind_leg.py \
  --task HindLeg-Direct-v0 --num_envs 1 \
  --video --video_length 400 --steps 500
```

체크포인트 직접 지정:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_hind_leg.py \
  --task HindLeg-Direct-v0 --num_envs 1 \
  --load_run <experiment_name> --checkpoint model_<iter>.pt
```

### 평가 — 대안: 표준 `play.py`

기본 추론만 필요하면 표준 play.py도 동작한다 (`OnPolicyRunnerParkour` 지원 확인됨).

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task HindLeg-Direct-v0 --num_envs 32
```

> 로그/체크포인트 저장 위치: `logs/rsl_rl/hindLeg_history_direct/`

---

## BLOCKING 검증 게이트

| 확인 항목 | 결과 |
|-----------|------|
| `rsl_rl_cfg_entry_point` 존재 | ✅ `agents/rsl_rl_ppo_cfg:HindLegParkourPPORunnerCfg` |
| runner `class_name` | ✅ `"OnPolicyRunnerParkour"` — `train.py`(line 220)·`play.py`(line 276) 명시 지원 |
| 커스텀 play 스크립트 | ✅ `play_hind_leg.py` 존재 — 카메라 추적 + 관절 플롯 제공 (권장) |
| 표준 `play.py` 사용 가능 | ✅ 가능 (기본 추론만 필요 시) |
| parkour 전용 conda env 필요 | ✅ 불필요 — 표준 `isaac-5.1` 사용 |

---

## 주요 파일

```
hind_leg/
├── hind_leg_env.py             # 환경 구현 (HindLegEnv)
├── hind_leg_env_cfg.py         # 환경/보상 파라미터
│                               #   HindLegFlatEnvCfg    (단순 flat, 26 DOF 버전)
│                               #   HindLegHistoryEnvCfg (기본, RMA + history, 8 DOF)
│                               #   HindLegRoughEnvCfg   (rough terrain 확장)
└── agents/
    ├── rsl_rl_ppo_cfg.py       # PPO 러너 cfg (Flat / Parkour / Rough 3종)
    ├── rl_games_flat_ppo_cfg.yaml
    ├── rl_games_rough_ppo_cfg.yaml
    ├── skrl_flat_ppo_cfg.yaml
    └── skrl_rough_ppo_cfg.yaml
```

---

## 주의사항

- `HindLegFlatEnvCfg`는 26 DOF(전신 모형), `HindLegHistoryEnvCfg`(기본 등록)는 8 DOF biped. 혼동 주의.
- `play_hind_leg.py`에는 `--task` 기본값이 `HindLeg-Direct-v0`로 설정되어 있어 `--task` 생략 가능.
- 명령 cmd=0 standing 학습을 위해 `rel_standing_envs=0.1` (10% env 강제 standing) 설정됨.

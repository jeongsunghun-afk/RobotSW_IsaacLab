# Go2 Parkour 환경

Go2 로봇이 계단·갭·경사면·허들·발판돌 등 다양한 지형을 자율 주행하도록 PPO 계열 강화학습으로 훈련하는 환경. [Extreme Parkour](https://extreme-parkour.github.io/) 아키텍처 기반 fork.

## 환경 구조

| 파일 | 설명 |
|------|------|
| `parkour_env.py` | 환경 구현 (`Go2ParkourEnv`) |
| `parkour_env_cfg.py` | 환경 설정 (`ParkourEnvCfg`) |
| `agents/rsl_rl_ppo_cfg.py` | RSL-RL 러너 설정 (5가지 변종) |
| `mdp/symmetry.py` | L/R 대칭 데이터 증강 함수 |

## 등록된 Task ID

5개 task 모두 **동일한 환경(`Go2ParkourEnv`) + 동일한 env cfg(`ParkourEnvCfg`)** 를 사용하며, 러너 설정(알고리즘/네트워크 변종)만 다릅니다.

| Task ID | 러너 설정 클래스 | 알고리즘 변종 | 설명 |
|---------|-----------------|---------------|------|
| `Go2-Parkour-Direct-v0` | `Go2ParkourPPORunnerCfg` | PPOParkour (기본) | 기준선 — ActorCriticRMA + PPOParkour |
| `Go2-Parkour-Direct-SPO` | `Go2ParkourSPOPPORunnerCfg` | PPOParkour + SPO | 클립 대신 이차 비율 패널티 대리함수 (ε=0.4) |
| `Go2-Parkour-Direct-LCP` | `Go2ParkourLCPPPORunnerCfg` | PPOParkour + LCP | Lipschitz 제약 그래디언트 패널티 (현재 PPOParkour에서 미활성) |
| `Go2-Parkour-Direct-MoE` | `Go2ParkourMoEPPORunnerCfg` | PPOParkour + MoE | 액터를 6-expert MoE-MLP로 교체 (ActorCriticRMAMoE) |
| `Go2-Parkour-Symmetry` | `Go2ParkourSymmetryPPORunnerCfg` | PPOParkour + 대칭 증강 | L/R 미러 데이터 증강으로 3-leg local optimum 방지 |

## 알고리즘 / 네트워크 구조

- **Runner**: `OnPolicyRunnerParkour` — DAGGER 스타일 히스토리 인코더 업데이트 포함
- **Policy network**: `ActorCriticRMA` — scan_encoder + priv_encoder + state_history_encoder
- **Algorithm**: `PPOParkour` — priv_reg loss 스케줄링 + dagger 업데이트
- **Estimator**: 고유감각 히스토리에서 `priv_explicit`(lin_vel, 6-dim) 예측

### Observation 그룹 (`obs_groups`)

| 그룹 | 차원 | 내용 |
|------|------|------|
| `policy` | 42 | projected_gravity(3)+commands(3)+joint_pos(12)+joint_vel(12)+prev_actions(12) |
| `scan` | 187 | height scan |
| `priv_explicit` | 6 | root_lin_vel_b(3) + root_ang_vel_b(3) |
| `priv_latent` | **33** | base_friction(1)+foot_friction(**4**)+base_mass(1)+base_com(3)+stiffness_ratio(12)+damping_ratio(12) — foot_friction은 static-only 4-dim. cfg 주석(37/8)과 상이하며 런타임 실제값은 33 |
| `history` | 10×42 | proprio 히스토리 (StateHistoryEncoder 입력) |
| critic 총합 | **268** | policy(42)+scan(187)+priv_explicit(6)+priv_latent(**33**) |

## 실행 명령어

> **참고**: 모든 5개 task는 표준 `train.py` / `play.py` 로 실행 가능합니다.
> `train.py`가 agent cfg의 `class_name="OnPolicyRunnerParkour"` 를 읽어 자동으로 올바른 러너를 선택합니다.

### 기본 (Go2-Parkour-Direct-v0)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Direct-v0 --num_envs 4096

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-Parkour-Direct-v0 --num_envs 32
```

### SPO 변종 (Go2-Parkour-Direct-SPO)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Direct-SPO --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-Parkour-Direct-SPO --num_envs 32
```

### LCP 변종 (Go2-Parkour-Direct-LCP)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Direct-LCP --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-Parkour-Direct-LCP --num_envs 32
```

> **주의**: LCP 패널티(`lambda_gp=0.002`)는 cfg에 등록되어 있으나, 현재 `PPOParkour`에서 RMA 인코더 입력 경로의 복잡성으로 인해 실제 적용되지 않습니다 (`rsl_rl_ppo_cfg.py` 주석 참조).

### MoE 변종 (Go2-Parkour-Direct-MoE)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Direct-MoE --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-Parkour-Direct-MoE --num_envs 32
```

### 대칭 증강 변종 (Go2-Parkour-Symmetry)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Symmetry --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-Parkour-Symmetry --num_envs 32
```

### 공통 옵션

```bash
# WandB 로깅
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Direct-v0 --num_envs 4096 \
    --logger wandb --wandb-project IsaacLab-Parkour

# 체크포인트에서 재개
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-Parkour-Direct-v0 --num_envs 4096 --resume

# 특정 체크포인트로 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-Parkour-Direct-v0 --num_envs 32 \
    --load_run <run_dir> --checkpoint model_<N>.pt
```

## 지형 커리큘럼

- **구성**: 11 난이도 행 × 40 열 (5종 서브지형 균등 배분)
- **서브지형**: `parkour_flat`, `parkour_hurdle`, `parkour_step`, `parkour_gap`, `parkour_stair`
- **커리큘럼 규칙**: 이동 거리 > 기대값 80% → 레벨 상승, < 40% → 레벨 하락
- **초기 최대 레벨**: `max_init_terrain_level = 9`

## 환경 기본 파라미터

| 파라미터 | 값 |
|---------|-----|
| DOF | 12 (액션 공간 = 12) |
| 에피소드 길이 | 20.0 s |
| Decimation | 4 (200 Hz 물리 / 50 Hz 정책) |
| Action scale | 0.25 |
| Termination height | 0.1 m |
| 최대 학습 이터레이션 | 50,000 |

## 로그 경로

| Task | 실험 폴더 이름 |
|------|---------------|
| Go2-Parkour-Direct-v0 | `logs/rsl_rl/go2_parkour/` |
| Go2-Parkour-Direct-SPO | `logs/rsl_rl/go2_parkour_spo/` |
| Go2-Parkour-Direct-LCP | `logs/rsl_rl/go2_parkour_lcp/` |
| Go2-Parkour-Direct-MoE | `logs/rsl_rl/go2_parkour_moe/` |
| Go2-Parkour-Symmetry | `logs/rsl_rl/go2_parkour_symmetry/` |

![Isaac Lab](docs/source/_static/isaaclab.jpg)

---

# IsaacLab — RGA-Inc Fork

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.1.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-BSD--3-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![License](https://img.shields.io/badge/license-Apache--2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![Upstream](https://img.shields.io/badge/upstream-NVIDIA%20Isaac%20Lab-76b900.svg)](https://github.com/isaac-sim/IsaacLab)

> 이 저장소는 **[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)** 을 fork하여 RGA-Inc의 커스텀 로봇 태스크(Go2 보행·파쿠르·모방학습, HindLeg biped, Real2Sim 등)와 연구용 RL/IL 알고리즘을 추가한 저장소입니다.

---

## 커스텀 환경 목록

| 환경 디렉토리 | 한 줄 설명 | 대표 Task ID | 알고리즘 |
|---|---|---|---|
| [**go2**](source/isaaclab_tasks/isaaclab_tasks/direct/go2/README.md) | Unitree Go2 평지/험지 보행, Bezier 발궤적 자연보행, 사물 상호작용 (7개 task) | `Go2`, `Go2WTW`, `Go2Rough` 등 | PPO (RMA 포함) |
| [**go2_imitation_tracking**](source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_tracking/README.md) | Go2 AMP 모방학습 + body-frame 속도추종 (vx, yaw_rate) | `Go2-Imitation-Tracking-v0` | AMP (PPOAMPBase) |
| [**go2_recovery**](source/isaaclab_tasks/isaaclab_tasks/direct/go2_recovery/README.md) | 임의 자세로 넘어진 Go2가 일어서는 복구 동작 학습 | `Go2Recovery-v0` | PPO |
| [**hind_leg**](source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/README.md) | RGA Inc 자체 제작 2족(biped) 8-DOF 로봇 보행 | `HindLeg-Direct-v0` | PPO (ActorCriticRMA) |
| [**parkour**](source/isaaclab_tasks/isaaclab_tasks/direct/parkour/README.md) | Go2 계단·갭·경사 등 다중 지형 파쿠르, Extreme Parkour 기반 (5개 변종) | `Go2-Parkour-Direct-v0` | PPOParkour (ActorCriticRMA) |
| [**parkour_imitation**](source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/README.md) | Parkour + AMP 모방학습 하이브리드. Symmetry·RandomGoal·TerrainStyle 변종 및 인터랙티브 데모 포함 (6개 task) | `Go2-ParkourImitation-v0` | PPOParkour + AMP |
| [**r2s_hind_leg**](source/isaaclab_tasks/isaaclab_tasks/direct/r2s_hind_leg/README.md) | ⚠️ **RL 아님** — R_Skeleton HindLeg의 Real2Sim Phase 1. PyQt5 GUI로 관절 실시간 제어 | `Isaac-R2S-HindLeg-v0` | Real2Sim (포지션 제어) |

> **⚠️ 주의사항**
> - `Go2-AMP-v0` (`go2/` 디렉토리): `__init__.py`에 등록되어 있으나 구현 파일이 없어 **실행 불가**. AMP 기반 Go2 모방학습은 `go2_imitation_tracking`을 사용할 것.
> - `r2s_hind_leg`: 표준 `train.py` / `play.py` 불가. 전용 스크립트(`scripts/real2sim/`) 사용 필수.

---

## 빠른 실행 (`run/`)

`run/` 폴더에 통합 런처와 환경별 커스텀 스크립트가 있습니다. 자세한 내용은 **[run/README.md](run/README.md)** 참조.

### 통합 런처 (`run/launch`)

표준 train/play를 한 줄로 실행합니다.

```bash
./run/launch <env_alias> <train|play> [추가 인자]
```

주요 alias 목록:

| alias | Task ID |
|---|---|
| `go2` | `Go2` |
| `go2wtw` | `Go2WTW` |
| `go2rough` | `Go2Rough` |
| `go2_recovery` | `Go2Recovery-v0` |
| `hind_leg` | `HindLeg-Direct-v0` |
| `go2_imitation_tracking` | `Go2-Imitation-Tracking-v0` |
| `parkour` | `Go2-Parkour-Direct-v0` |
| `parkour_sym` | `Go2-Parkour-Symmetry` |
| `parkour_spo` | `Go2-Parkour-Direct-SPO` |
| `parkour_moe` | `Go2-Parkour-Direct-MoE` |
| `parkour_imitation` | `Go2-ParkourImitation-v0` |
| `parkour_imitation_sym` | `Go2-ParkourImitation-Symmetry-v0` |
| `parkour_imitation_rg` | `Go2-ParkourImitation-Symmetry-RandomGoal-v0` |
| `parkour_imitation_terrain` | `Go2-ParkourImitation-TerrainStyle-v0` |

```bash
# 예시: 학습
./run/launch go2_recovery train --headless
./run/launch parkour_imitation_sym train --num_envs 2048 --headless

# 예시: 평가
./run/launch go2_recovery play --checkpoint logs/rsl_rl/go2_recovery_direct/.../model_1000.pt
./run/launch hind_leg play --num_envs 1
```

> parkour 계열을 `isaac-parkour` conda env로 실행해야 할 경우:
> ```bash
> CONDA_ENV=isaac-parkour ./run/launch parkour train
> ```

### 환경별 커스텀 스크립트

| 스크립트 | 용도 |
|---|---|
| `./run/parkour_demo` | Parkour headless 녹화 sweep (test 지형 5종×11난이도) |
| `./run/parkour_demo_test` | Parkour GUI/WebRTC — 패널로 지형·난이도 선택 |
| `./run/parkour_demo_playground` | Parkour GUI/WebRTC — omni.ui 조이스틱 자유 조종 |
| `./run/play_interactive` | 터미널 실시간 커맨드 입력 추론 |
| `./run/play_hind_leg` | HindLeg 전용 추론 + 관절 position/velocity/torque 플롯 |
| `./run/report` | 다중 커맨드 일괄 평가 → `results/` 저장 |
| `./run/amp_report` | AMP discriminator obs 검증 (Sim obs vs Ref obs 차원 비교) |
| `./run/go2_imitation_plot` | vx sweep 속도 추종 플롯 |
| `./run/go2_imitation_tracking_plot` | Imitation Tracking vx sweep 플롯 |

---

## 표준 실행 (`./isaaclab.sh -p`)

통합 런처 없이 직접 실행할 때:

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task <Task-ID> --num_envs 4096 --headless

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task <Task-ID> --num_envs 32

# WandB 로깅 포함
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096 \
  --logger wandb --wandb-project IsaacLab-ParkourImitation
```

> **conda 환경**: 대부분의 task는 `isaac-5.1`. parkour 계열은 `isaac-parkour` 사용 가능 (환경별 README 참조).

---

## rsl_rl Fork

이 저장소는 [leggedrobotics/rsl_rl](https://github.com/leggedrobotics/rsl_rl)도 fork하여 다음을 추가했습니다.

- **알고리즘**: `PPOAMPBase`, `PPOAMP`, `PPOParkour` (SPO·RND·symmetry aug·estimator 공동학습 등 연구 옵션 포함)
- **네트워크**: `ActorCriticRMA` (Rapid Motor Adaptation), `ActorCriticMoE`, `ActorCriticCNN`
- **러너**: `OnPolicyRunnerParkour`, `OnPolicyRunnerAMP`, `OnPolicyRunnerParkourAMP`
- **보조 모듈**: `AMPDiscriminator`, `Estimator`, `RecurrentDepthBackbone`

자세한 내용은 **[rsl_rl/README.md](rsl_rl/README.md)** 의 "Fork 추가 알고리즘" 섹션 참조.

---

## 설치

upstream Isaac Lab 설치 방법은 **[공식 문서](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation)** 를 참조하세요.

설치 후 커스텀 환경 등록:

```bash
# 개발 모드 설치 (커스텀 task 포함)
./isaaclab.sh -i

# 코드 포맷
./isaaclab.sh -f

# 테스트
./isaaclab.sh -t
```

---

## Upstream Attribution & License

이 저장소는 **[NVIDIA Isaac Lab](https://github.com/isaac-sim/IsaacLab)** 을 기반으로 합니다.

> Isaac Lab framework is released under [BSD-3 License](LICENSE).  
> The `isaaclab_mimic` extension is released under [Apache 2.0](LICENSE-mimic).  
> Isaac Lab requires Isaac Sim, which includes components under proprietary licensing terms.  
> See [`docs/licenses/`](docs/licenses/) for full dependency license information.

원본 프로젝트의 상세 문서, 튜토리얼, API 레퍼런스:
- 🔗 [Isaac Lab 공식 문서](https://isaac-sim.github.io/IsaacLab)
- 🔗 [Isaac Lab GitHub](https://github.com/isaac-sim/IsaacLab)
- 🔗 [arXiv 논문](https://arxiv.org/abs/2511.04831)

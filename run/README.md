# run/ — 실행 스크립트 모음

IsaacLab fork 커스텀 환경의 학습·평가·시각화 실행 진입점.
모든 스크립트는 저장소 루트(`IsaacLab/`)에서 호출한다.

---

## 1. 통합 런처 (`run/launch`)

표준 `train.py` / `play.py` 기반 환경을 **한 줄**로 실행.

```bash
./run/launch <env_alias> <train|play> [추가 인자]
```

### 지원 env_alias 목록

| alias | task id | conda env |
|-------|---------|-----------|
| `go2` | `Go2` | `isaac-5.1` |
| `go2wtw` | `Go2WTW` | `isaac-5.1` |
| `go2rough` | `Go2Rough` | `isaac-5.1` |
| `go2_recovery` | `Go2Recovery-v0` | `isaac-5.1` |
| `hind_leg` | `HindLeg-Direct-v0` | `isaac-5.1` |
| `go2_imitation` | `Go2-Imitation-v0` | `isaac-5.1` |
| `go2_imitation_tracking` | `Go2-Imitation-Tracking-v0` | `isaac-5.1` |
| `parkour` | `Go2-Parkour-Direct-v0` | `isaac-5.1`\* |
| `parkour_sym` | `Go2-Parkour-Symmetry` | `isaac-5.1`\* |
| `parkour_spo` | `Go2-Parkour-Direct-SPO` | `isaac-5.1`\* |
| `parkour_lcp` | `Go2-Parkour-Direct-LCP` | `isaac-5.1`\* |
| `parkour_moe` | `Go2-Parkour-Direct-MoE` | `isaac-5.1`\* |
| `parkour_imitation` | `Go2-ParkourImitation-v0` | `isaac-5.1`\* |
| `parkour_imitation_sym` | `Go2-ParkourImitation-Symmetry-v0` | `isaac-5.1`\* |
| `parkour_imitation_rg` | `Go2-ParkourImitation-Symmetry-RandomGoal-v0` | `isaac-5.1`\* |
| `parkour_imitation_terrain` | `Go2-ParkourImitation-TerrainStyle-v0` | `isaac-5.1`\* |

> \* parkour/parkour_imitation 계열을 `isaac-parkour` env로 실행해야 하는 경우:
> `CONDA_ENV=isaac-parkour ./run/launch parkour train`

> **r2s_hind_leg (`Isaac-R2S-HindLeg-v0`)**: RL 환경이 아님. 표준 train/play 불가.
> `scripts/real2sim/sim_runner.py` + `scripts/real2sim/controller.py` 직접 사용.

### 기본값

- train: `--num_envs 4096`
- play: `--num_envs 32`
- `--num_envs`를 직접 주면 기본값을 덮어씀

### 예시

```bash
# 학습
./run/launch go2_recovery train --headless
./run/launch parkour_imitation_sym train --num_envs 2048 --headless

# 평가
./run/launch go2_recovery play --checkpoint logs/rsl_rl/go2_recovery/.../model_1000.pt
./run/launch hind_leg play --num_envs 1
```

---

## 2. 환경별 개별 스크립트

### Parkour Demo (커스텀 `play_parkour_demo.py` 사용)

| 스크립트 | 용도 |
|---------|------|
| `./run/parkour_demo` | headless sweep 녹화 (test 지형 5종×11난이도) |
| `./run/parkour_demo_test` | GUI/WebRTC — test 지형, goal 기반 자동 수행, 패널로 종류/난이도 선택 |
| `./run/parkour_demo_playground` | GUI/WebRTC — 자유 조종(omni.ui 조이스틱) |

기본 체크포인트는 `PARKOUR_CKPT` 환경변수로 덮어쓸 수 있음:
```bash
PARKOUR_CKPT=/path/to/model.pt ./run/parkour_demo
./run/parkour_demo_test --livestream 2 --enable_cameras
```

### 인터랙티브 추론

```bash
# 터미널에서 실시간 커맨드 입력 (Go2WTW, 일반 3-cmd 모두 지원)
./run/play_interactive --task Go2WTW --num_envs 1
./run/play_interactive --task Go2WTW --num_envs 1 --checkpoint logs/.../model.pt --save_data
```

### HindLeg 전용 play + 관절 플롯

```bash
# inference 후 관절 position/velocity/torque 플롯을 <run>/plots/ 에 저장
./run/play_hind_leg
./run/play_hind_leg --checkpoint logs/rsl_rl/hind_leg/.../model_1000.pt --steps 1000 --video
```

### 다중 커맨드 일괄 평가

```bash
# commands.yaml에 정의된 커맨드 수만큼 parallel env → 500스텝 → results/ 저장
./run/report --task Go2WTW --checkpoint logs/.../model.pt
./run/report --task Go2-Imitation-v0 --checkpoint logs/.../model.pt

# AMP Discriminator obs 검증 (Sim obs vs Ref obs 차원 비교)
./run/amp_report --task Go2-Imitation-v0 --checkpoint logs/.../model.pt
```

### 속도 추종 플롯

```bash
# vx 0→max→0 sweep → velocity_comparison.png + joints.png
./run/go2_imitation_plot --checkpoint logs/.../model.pt
./run/go2_imitation_plot --task Go2-Imitation-v0 --max_speed 3.0 --headless

./run/go2_imitation_tracking_plot --checkpoint logs/.../model.pt
./run/go2_imitation_tracking_plot --task Go2-Imitation-Tracking-v0 --max_speed 3.5 --video --headless
```

---

## 주의: 실행 불가 task

| task id | 이유 |
|---------|------|
| `Go2-AMP-v0` | `__init__.py`에 등록됐으나 구현 파일(`go2_amp_env.py`, `go2_amp_env_cfg.py`)이 없어 ModuleNotFoundError 발생. 통합 런처 매핑에서 제외. |

---

## 미지원 (래퍼 없음)

아래 스크립트는 인자 구조가 특수하거나 독립 실행용으로, 래퍼를 만들지 않았다.
직접 `./isaaclab.sh -p <script>` 로 실행:

| 스크립트 | 비고 |
|---------|------|
| `scripts/real2sim/sim_runner.py` | r2s_hind_leg 시뮬레이션 |
| `scripts/real2sim/controller.py` | r2s_hind_leg 컨트롤러 (별도 터미널) |
| `scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py` | pronk cost 측정 분석 |
| `scripts/reinforcement_learning/rsl_rl/verify_3leg_contact.py` | 3발 contact 검증 |

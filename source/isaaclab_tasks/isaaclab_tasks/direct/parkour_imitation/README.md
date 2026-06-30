# Go2 ParkourImitation 환경

Go2 로봇의 다중 지형 파쿠르 주행에 **AMP(Adversarial Motion Prior)** 모방학습을 결합한 하이브리드 환경. 평지 서브지형에서는 AMP 보상(trot 참조 모션 모방)이 추가되어 자연스러운 보행 스타일을 강제하며, 나머지 지형에서는 parkour 보상만 적용됩니다.

> 기반: `parkour_env.py` / `parkour_env_cfg.py` (`ParkourEnvCfg`) 상속.
> AMP 연산은 `parkour_imitation_env.py` (`Go2ParkourImitationEnv`)에서 처리.

## 환경 구조

| 파일 | 설명 |
|------|------|
| `parkour_imitation_env.py` | AMP 하이브리드 환경 (`Go2ParkourImitationEnv`) |
| `parkour_imitation_env_cfg.py` | AMP + 지형 설정 (`ParkourImitationEnvCfg`, `ParkourImitationTerrainStyleEnvCfg`) |
| `parkour_imitation_random_goal_env.py` | 360° 랜덤 목표 변종 (`Go2ParkourImitationRandomGoalEnv`) |
| `parkour_imitation_random_goal_env_cfg.py` | 랜덤 목표 환경 설정 |
| `parkour_imitation_terrain_style_env.py` | 지형 불변 AMP obs 변종 (`ParkourImitationTerrainStyleEnv`) |
| `parkour_demo_env.py` | 인터랙티브 데모 환경 (`Go2ParkourDemoEnv`) |
| `parkour_demo_env_cfg.py` | 데모 환경 설정 (`ParkourDemoEnvCfg`, `ParkourPlaygroundEnvCfg`) |
| `agents/rsl_rl_amp_cfg.py` | RSL-RL AMP 러너 설정 (6가지 변종) |
| `motion_lib.py` | 참조 모션 로더 (`Go2MotionLib`) |
| `imitation/` | 참조 모션 `.pkl` 파일 디렉토리 |

## AMP 설계 요약

- **Runner**: `OnPolicyRunnerParkourAMP` (PPOParkour + AMP discriminator 통합)
- **Policy**: `ActorCriticRMA` (scan_encoder + priv_encoder + state_history_encoder)
- **Algorithm**: `PPOAMP` — PPOParkour에 AMP 판별자 loss를 결합
- **AMP 보상 fusion**: `total = task_reward + amp_weight(0.3) × flat_env_mask × disc_reward`
- **AMP obs 기본**: 49-dim/프레임 × 10 프레임 = **490-dim** (dof_pos/vel + root_height + lin_vel + ang_vel + foot_pos_local + root_rot_tan_norm)
- **참조 모션**: `ParkourImitationEnvCfg.amp_motion_pkl = "imitation/go2_jump"` (디렉토리 → 복수 .pkl 자동 탐색)

## 등록된 Task ID

| Task ID | 환경 클래스 | 러너 설정 클래스 | 설명 |
|---------|------------|-----------------|------|
| `Go2-ParkourImitation-v0` | `Go2ParkourImitationEnv` | `Go2ParkourImitationPPOAMPRunnerCfg` | 기준선 — 파쿠르+AMP (평지 마스크 적용) |
| `Go2-ParkourImitation-Symmetry-v0` | `Go2ParkourImitationEnv` | `Go2ParkourImitationSymmetryPPOAMPRunnerCfg` | 위 + L/R 미러 데이터 증강 |
| `Go2-ParkourImitation-Symmetry-RandomGoal-v0` | `Go2ParkourImitationRandomGoalEnv` | `Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg` | Symmetry + 360° 랜덤 목표 커리큘럼 |
| `Go2-ParkourImitation-TerrainStyle-v0` | `ParkourImitationTerrainStyleEnv` | `Go2ParkourImitationTerrainStylePPOAMPRunnerCfg` | Symmetry + 지형 불변 AMP obs (37-dim/프레임 = 370-dim) |
| `Go2-ParkourDemo-v0` | `Go2ParkourDemoEnv` | `Go2ParkourImitationSymmetryPPOAMPRunnerCfg` | 데모 전용 — test 지형 (5종×11난이도, num_envs=1) |
| `Go2-ParkourDemo-Playground-v0` | `Go2ParkourDemoEnv` | `Go2ParkourImitationSymmetryPPOAMPRunnerCfg` | 데모 전용 — 자유 scatter 지형 (num_envs=1) |

### 변종별 차이 요약

| 변종 | 핵심 차이 |
|------|-----------|
| `v0` (기준) | AMP 보상을 평지(`flat_env_mask`) 환경에만 적용 |
| `Symmetry` | L/R 미러 데이터 증강 추가 (`symmetry_cfg`, `num_aug=2`) — 3-leg local optimum 방지 |
| `Symmetry-RandomGoal` | 각 리셋 시 로봇 주변 360°·[1.5, 3.0]m 범위에서 랜덤 목표 재추첨 |
| `TerrainStyle` | AMP obs에서 지형 의존 차원 제거 (root_height, lin_vel_z, foot_z, rot_tan_norm 삭제 → 37-dim) |
| `Demo/Playground` | num_envs=1, 팔로우 카메라 추가, **`play_parkour_demo.py`로만 실행 가능** |

## Observation 그룹 (AMP 학습 task 공통)

| 그룹 | 차원 | 내용 |
|------|------|------|
| `policy` | 42 | projected_gravity(3)+commands(3)+joint_pos(12)+joint_vel(12)+prev_actions(12) |
| `scan` | 187 | height scan |
| `priv_explicit` | 6 | root_lin_vel_b(3) + root_ang_vel_b(3) |
| `priv_latent` | **33** | base_friction(1)+foot_friction(**4**)+base_mass(1)+base_com(3)+stiffness_ratio(12)+damping_ratio(12) — foot_friction은 static-only 4-dim. cfg 주석(37/8)과 상이하며 런타임 실제값은 33 |
| `history` | 10×42 | proprio 히스토리 (StateHistoryEncoder 입력) |
| critic 총합 | **268** | policy(42)+scan(187)+priv_explicit(6)+priv_latent(**33**) |

AMP obs는 `env.extras["amp_obs"]` 별도 경로로 판별자에 전달됩니다 (obs 그룹과 독립).

---

## 실행 명령어

### ⚠️ AMP train.py 호환성 확인 결과

`scripts/reinforcement_learning/rsl_rl/train.py`는 `OnPolicyRunnerParkourAMP`를 명시적으로 지원합니다 (train.py:226-227). agent cfg의 `class_name = "OnPolicyRunnerParkourAMP"` 를 읽어 자동으로 올바른 러너를 선택하므로, **AMP 학습 task 4개는 모두 vanilla `train.py`로 실행 가능합니다**. 별도 스크립트 불필요.

### AMP 학습 Task (Go2-ParkourImitation-v0)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-v0 --num_envs 4096

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-v0 --num_envs 32
```

### Symmetry 변종 (Go2-ParkourImitation-Symmetry-v0)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 32
```

### RandomGoal 변종 (Go2-ParkourImitation-Symmetry-RandomGoal-v0)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-RandomGoal-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-Symmetry-RandomGoal-v0 --num_envs 32
```

### TerrainStyle 변종 (Go2-ParkourImitation-TerrainStyle-v0)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-TerrainStyle-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-TerrainStyle-v0 --num_envs 32
```

### 공통 학습 옵션

```bash
# WandB 로깅
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096 \
    --logger wandb --wandb-project IsaacLab-ParkourImitation

# 체크포인트에서 재개
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096 --resume
```

---

## 데모 실행 (Go2-ParkourDemo-v0 / Go2-ParkourDemo-Playground-v0)

> **⚠️ 데모 task는 vanilla `play.py`로 실행 불가** — 커스텀 진입점 `play_parkour_demo.py`를 사용해야 합니다.

데모 실행 시 반드시 `--checkpoint` 인자로 학습된 체크포인트 경로를 지정하세요.

### 루트 bash 래퍼 (권장)

```bash
# 헤드리스 녹화 sweep (test 지형 전체)
./parkour_demo

# 특정 지형/난이도만 녹화
./parkour_demo --record_classes stair,gap --record_levels 0,5,10

# 다른 체크포인트 사용
./parkour_demo --checkpoint logs/rsl_rl/<run>/model_<N>.pt
```

```bash
# test 지형 인터랙티브 (GUI, 패널로 지형/난이도 선택)
./parkour_demo_test

# WebRTC 원격 스트리밍
./parkour_demo_test --livestream 2 --enable_cameras
```

```bash
# playground 인터랙티브 (GUI, omni.ui 조이스틱)
./parkour_demo_playground

# WebRTC 원격 스트리밍
./parkour_demo_playground --livestream 2 --enable_cameras
```

### 직접 실행 (play_parkour_demo.py)

```bash
# test 모드 — goal 기반 자동 주행, test 지형
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \
    --task Go2-ParkourDemo-v0 \
    --checkpoint logs/rsl_rl/<run>/model_<N>.pt \
    --mode test

# playground 모드 — 자유 조종, scatter 지형
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \
    --task Go2-ParkourDemo-Playground-v0 \
    --checkpoint logs/rsl_rl/<run>/model_<N>.pt \
    --mode playground

# 헤드리스 녹화 sweep
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \
    --task Go2-ParkourDemo-v0 \
    --checkpoint logs/rsl_rl/<run>/model_<N>.pt \
    --record --headless --enable_cameras \
    --record_classes stair,gap \
    --record_levels 0,5,10 \
    --video_dir logs/demo_videos
```

> `num_envs`는 항상 1로 강제됩니다 (사용자 인자 무시).

### 데모 conda env

루트 bash 래퍼 3개(`parkour_demo`, `parkour_demo_test`, `parkour_demo_playground`)는 모두 **`conda activate isaac-5.1`** 을 사용합니다.
> **주의**: 프로젝트 메모리에 parkour 서브프로젝트가 `isaac-parkour` env를 사용한다는 노트가 있으나, 실제 래퍼 코드는 `isaac-5.1`을 activate합니다. 직접 실행 시에는 현재 활성 conda env 혹은 `./isaaclab.sh`를 사용하세요.

---

## 로그 경로

| Task | 실험 폴더 이름 |
|------|---------------|
| Go2-ParkourImitation-v0 | `logs/rsl_rl/parkour_imitation_go2/` |
| Go2-ParkourImitation-Symmetry-v0 | `logs/rsl_rl/parkour_imitation_go2_symmetry/` |
| Go2-ParkourImitation-Symmetry-RandomGoal-v0 | `logs/rsl_rl/parkour_imitation_go2_symmetry_random_goal/` |
| Go2-ParkourImitation-TerrainStyle-v0 | `logs/rsl_rl/parkour_imitation_go2_terrain_style/` |

## AMP 하이퍼파라미터 요약 (기본값)

| 파라미터 | 값 |
|---------|-----|
| `amp_weight` | 0.3 |
| `disc_loss_type` | `bce` |
| `discriminator_learning_rate` | 2.5e-4 |
| `gradient_penalty_coef` | 5.0 |
| `replay_buffer_size` | 200,000 |
| `amp_observation_space` | 490 (49-dim × 10 frames, TerrainStyle: 370) |
| 참조 모션 경로 | `ParkourImitationEnvCfg.amp_motion_pkl` (env cfg가 소유) |

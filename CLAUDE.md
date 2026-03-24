# IsaacLab Project Guide

## 실행 환경
- Isaac Sim 기반 GPU 가속 RL 프레임워크
- Python 3.10+ (Isaac Sim 4.5: 3.10 / 5.0+: 3.11)
- PyTorch 2.7+ (CUDA 12.8)
- 가상환경: conda 또는 uv

## 주요 명령어

```bash
# Python 실행 (항상 이 방식 사용)
./isaaclab.sh -p <script.py>

# 학습 실행
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task R_Skeleton-v0 --num_envs 4096

# 시뮬레이션 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task R_Skeleton-v0 --num_envs 32

# 코드 포맷
./isaaclab.sh -f

# 테스트 실행
./isaaclab.sh -t

# 패키지 설치 (개발 모드)
./isaaclab.sh -i
```

## 프로젝트 구조

```
source/
├── isaaclab/            # 핵심 프레임워크 (건드리지 말 것)
├── isaaclab_tasks/      # RL 환경 태스크 (주요 작업 영역)
│   └── isaaclab_tasks/direct/
│       └── R_Skeleton/  # 커스텀 환경
├── isaaclab_assets/     # 로봇/에셋 정의
└── isaaclab_rl/         # RL 프레임워크 연동
```

## 태스크 파일 구조 패턴

새 환경 추가 시 `source/isaaclab_tasks/isaaclab_tasks/direct/<name>/` 에:
```
<name>_env.py          # 환경 클래스 (DirectRLEnv 상속)
<name>_env_cfg.py      # 설정 클래스 (@configclass, DirectRLEnvCfg 상속)
__init__.py            # Gym 등록
agents/
└── rsl_rl_ppo_cfg.py  # RSL RL 학습 설정
```

## 코드 스타일

- 라인 길이: 120자
- Linting: ruff (E, W, F, I, UP, C90, SIM, RET 규칙)
- Type check: pyright (basic mode)
- `@configclass` 데코레이터를 모든 config 클래스에 사용

## R_Skeleton 환경

**파일 위치:** `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/`

| 등록 ID | 클래스 | 설명 |
|---------|--------|------|
| R_Skeleton-v0 | SkeletonEnv | 기본 (history 포함) |
| R_Skeleton-v1 | SkeletonEnv | fixed variant |
| R_Skeleton-AMP-v0 | SkeletonAmpEnv | Adversarial Motion Prior |
| R_Skeleton-WTW-v0 | SkeletonWTWEnv | Walk-the-walk |

**로봇 스펙:**
- DOF: 34 (action_space = 34)
- 발 링크: FL/FR/HL/HR_link7_toe
- 종료 조건: base height < 0.3m
- 에피소드: 20초, decimation: 4 (50Hz 정책)

## 환경 구현 필수 메서드

```python
class MyEnv(DirectRLEnv):
    def _setup_scene(self): ...       # 씬 초기화
    def _pre_physics_step(self): ...  # 물리 스텝 전 액션 처리
    def _apply_action(self): ...      # 액션 적용
    def _get_observations(self) -> dict: ...
    def _get_rewards(self) -> torch.Tensor: ...
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def _reset_idx(self, env_ids): ...
```

## 핵심 Import 패턴

```python
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate_inverse, yaw_quat
```

## 학습 출력

- 로그/체크포인트: `outputs/` 디렉토리
- RSL RL 설정: `agents/rsl_rl_ppo_cfg.py`
- WandB 연동 가능 (선택적)

## 주의사항

- Isaac Sim은 반드시 `./isaaclab.sh -p` 를 통해 실행 (직접 `python` 사용 금지)
- GPU 메모리: 4096 환경 기준 약 8-16GB VRAM 필요
- `source/isaaclab/` 내부 코어 파일 직접 수정 비권장
- 시뮬레이션 설정은 `SimulationCfg`의 `dt`, `render_interval` 조정으로 속도 제어

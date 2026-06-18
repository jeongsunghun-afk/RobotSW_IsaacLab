# MimicKit 분석 및 비교 자료 (Reference for Imitation Learning)

**경로**: `/home/lgb/MimicKit`

MimicKit은 강화학습(RL) 기반의 모션 모방 및 제어를 위한 경량화된 프레임워크입니다. IsaacLab, Isaac Gym 등 다양한 시뮬레이터 백엔드를 지원하며 최신 모방학습(Imitation Learning) 알고리즘들이 구현되어 있습니다.

## 1. 디렉토리 및 파일 구조

- **`mimickit/`**: 핵심 프레임워크 코드
  - **`envs/`**: 시뮬레이션 환경 정의 (다양한 알고리즘에 맞춘 환경 클래스)
    - `base_env.py`, `sim_env.py`, `char_env.py` (기본 물리 및 캐릭터 제어 환경)
    - `deepmimic_env.py`: DeepMimic 알고리즘용 환경 (직접적인 tracking reward)
    - `amp_env.py`: AMP 알고리즘용 환경 (Adversarial reward 연동)
    - `ase_env.py`: ASE 알고리즘용 환경 (Skill embedding)
    - `add_env.py`: ADD (Adversarial Differential Discriminator)용 환경
  - **`learning/`**: 강화학습 및 모방학습 에이전트(알고리즘) 구현
    - `base_agent.py`, `ppo_agent.py` (PPO 기본 구현)
    - `amp_agent.py`: AMP 기반 Adversarial 모방학습 에이전트
    - `ase_agent.py`: ASE 기반 스킬 임베딩 에이전트
    - `awr_agent.py`: AWR (Advantage-Weighted Regression) 에이전트
    - `lcp_agent.py`: LCP (Lipschitz-Constrained Policies) 에이전트
    - `add_agent.py`: ADD 기반 에이전트
  - **`engines/`**: 시뮬레이터 백엔드 연동 (Isaac Gym, Isaac Lab, Newton 등)
  - **`anim/`**: 모션 데이터 파싱 및 처리 (`motion.py` 등)

- **`data/`**: 에셋, 모션 데이터, 설정 파일
  - `envs/`: 환경 설정 yaml 파일 (`deepmimic_humanoid_env.yaml` 등)
  - `agents/`: 에이전트 하이퍼파라미터 yaml 파일
  - `motions/`, `datasets/`: 학습에 사용되는 레퍼런스 모션 데이터(.pkl)

- **`args/`**: 학습 및 테스트 실행을 위한 커맨드라인 인자 프리셋 (예: `amp_go2_args.txt`)

## 2. 지원 알고리즘 (Imitation Learning)

MimicKit은 다음과 같은 최신 모방학습 알고리즘을 지원하며, 이를 IsaacLab에 이식하거나 아이디어를 차용할 수 있습니다.

1. **DeepMimic (2018)**: 레퍼런스 모션의 관절 각도/속도를 직접적으로 추종하는 Reward 함수 사용.
2. **AMP (Adversarial Motion Priors, 2021)**: Discriminator를 사용하여 모션 클립으로부터 스타일 보상을 자동 학습.
3. **AWR (Advantage-Weighted Regression, 2019)**: 오프폴리시 데이터를 쉽게 통합할 수 있는 지도학습 기반 최적화 기법.
4. **ASE (Adversarial Skill Embeddings, 2022)**: AMP를 확장하여 재사용 가능한 다양한 스킬 임베딩 학습.
5. **LCP (Lipschitz-Constrained Policies, 2025)**: Actor 네트워크에 Lipschitz 제약을 걸어 동작의 부드러움(Smoothness) 보장. 발 충격 감소에 유효.
6. **ADD (Adversarial Differential Discriminator, 2025)**: 다목적 최적화(MOO)를 통해 단일 모션 샘플만으로도 효과적으로 모방학습을 수행하고 Reward 튜닝 부담을 최소화.

## 3. 시뮬레이션 환경 (Simulation Engines)

MimicKit은 엔진(Engine) 추상화를 통해 다중 시뮬레이터를 지원합니다.
- **Isaac Lab**: `--engine_config data/engines/isaac_lab_engine.yaml`
- **Isaac Gym**: `--engine_config data/engines/isaac_gym_engine.yaml`
- **Newton**: `--engine_config data/engines/newton_engine.yaml`

IsaacLab 연동이 이미 구현되어 있으므로, IsaacLab에서의 AMP 또는 LCP, ADD 알고리즘 구현 방식을 참고할 때 `mimickit/engines/` 및 `mimickit/envs/` 구조를 직접 참고할 수 있습니다.

## 4. 모션 데이터 표현 방식

- **포맷**: `.pkl` 형식으로 저장됨.
- **상태 벡터 (State Vector)**:
  `[root position (3D), root rotation (3D), joint rotations]`
  - 3D 회전은 **3D Exponential Maps**를 사용하여 표현.
  - 관절(Joint) 회전은 XML(또는 URDF)에서 정의된 순서(깊이 우선 탐색, DFS)대로 기록됨.
  - 예: Humanoid의 경우 Root(3D/3D) + Abdomen(3D) + Neck(3D) + 1D/3D 관절들의 조합.

## 5. 향후 활용 방안 (IsaacLab 프로젝트 적용)

본 문서는 R_Skeleton 및 Go2 로봇의 보행 학습 고도화를 위해 다음 작업 시 참고 자료로 활용됩니다:
- **부드러운 보행 보장**: `lcp_agent.py`의 Spectral Normalization 및 Gradient Penalty 구현 방식을 IsaacLab 코드베이스(`rsl_rl`)로 포팅.
- **보상 함수 최적화**: `add_agent.py`와 `add_env.py`를 분석하여 수동 Reward 튜닝을 줄이는 구조 적용 검토.
- **RSI (Reference State Initialization) 개선**: `amp_env.py` 및 `deepmimic_env.py`의 초기화 로직(Pose Divergence Termination 등)을 참조하여 안정적인 학습 환경 구성.
- **모션 좌표계 매핑**: `motion.py`와 `sim_env.py`의 관절 각도 추출/변환 로직을 분석하여 IsaacLab과 MimicKit 간의 좌표계 통일.

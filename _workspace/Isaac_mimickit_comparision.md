# IsaacLab vs MimicKit 코드베이스 비교 분석 보고서

본 보고서는 IsaacLab(`go2_amp_env.py`, `ppo_amp.py`, `go2_motion_loader.py`)과 MimicKit(`amp_env.py`, `amp_agent.py`, `motion_lib.py`)의 AMP(Adversarial Motion Prior) 기반 강화학습 파이프라인 코드를 비교 분석한 결과입니다.

---

## 1. 학습 환경 (Learning Environment) 비교
**대상 파일**: `go2_amp_env.py` (IsaacLab) vs `amp_env.py` (MimicKit)

### 프레임워크 및 구조의 범용성
*   **IsaacLab**: `DirectRLEnv` 기반으로 구축되었으며 특정 로봇(Unitree Go2, 12 DoF)에 고도로 종속된 최적화 구조를 가집니다. 발의 개수, 관절 이름, 링크의 기하학적 특성이 환경 내에 하드코딩되어 있어 연산 효율성이 높습니다.
*   **MimicKit**: `DeepMimicEnv` 기반이며 `_kin_char_model`을 주입받아 사용하는 범용적인 환경입니다. 로봇의 종류에 구애받지 않고 다양한 캐릭터 형태에 재사용이 가능하도록 설계되어 있습니다.

### AMP 관측 (Observation) 구성
*   **IsaacLab**: 43차원의 조밀한 1D 벡터를 사용합니다. 
    *   구성: `dof_pos(12)`, `dof_vel(12)`, `root_height(1)`, `lin_vel(3)`, `ang_vel(3)`, `key_body_pos(12)`(발 4개의 로컬 위치)
    *   `num_amp_observations` 변수를 통해 이전 프레임의 상태를 스택으로 누적하여 PyTorch Tensor 버퍼 내에서 직접 History를 관리합니다.
*   **MimicKit**: `compute_disc_obs` 함수를 통해 동적으로 상태를 추출합니다.
    *   로봇의 Heading Frame(진행 방향)을 기준으로 변환된 Root 속도/각속도와 Joint Rotation 차이를 입력으로 활용합니다.
    *   `CircularBuffer` 유틸리티를 활용해 각 속성별(위치, 회전, 속도 등)로 History를 구조적으로 저장합니다.

### 종료 조건 (Termination)
*   **공통점**: RSI(Random State Initialization)를 통한 초기화 방식을 지원하며, Reference 모션 데이터와 현재 시뮬레이션 로봇 간의 관절 또는 Key Body 위치 차이가 일정 거리 이상 벌어지면 에피소드를 종료(Pose Termination)합니다.
*   **IsaacLab 추가 조건**: 베이스(Base) 높이 저하, 전복(Flipped), Roll/Pitch 각도 한계 초과 및 발이 아닌 몸체 충돌(Contact Force) 시에도 조기 종료되도록 물리적 제약이 상세하게 구현되어 있습니다.

---

## 2. 학습 알고리즘 (Learning Algorithm) 비교
**대상 파일**: `ppo_amp.py` (IsaacLab) vs `amp_agent.py` (MimicKit)

### Discriminator 손실 함수 (Loss Function) 형태
*   **IsaacLab**: **LS-GAN (Least Squares GAN)** 방식을 사용합니다. 
    *   Expert 데이터에 대한 목표값을 `+1`로, Policy 데이터에 대한 목표값을 `-1`로 설정하고 `MSELoss`를 적용하여 수렴 속도와 안정성을 높였습니다.
*   **MimicKit**: **Standard GAN** 방식을 사용합니다. 
    *   `BCEWithLogitsLoss`를 통해 Expert는 `1`, Policy는 `0`의 라벨을 갖도록 이진 분류 모델로 학습합니다.

### 정규화 (Regularization) 및 Gradient Penalty
*   **공통점**: Expert 및 Policy 배치 양쪽의 입력 관측치에 대해 Gradient를 계산하고, 이 Norm의 제곱합으로 Gradient Penalty를 부여하여 Discriminator의 과적합과 발산을 막습니다.
*   **Logit 정규화 차이**:
    *   **IsaacLab**: Discriminator가 출력하는 *Logit 값 자체*의 제곱 평균(`expert_logits^2 + policy_logits^2`)에 L2 페널티를 곱하여 출력 스케일을 억제합니다.
    *   **MimicKit**: 네트워크 모델의 마지막 레이어 *가중치(Weights)* 제곱합(`logit_weights^2`)에 L2 페널티(Weight Decay)를 주어 정규화를 수행합니다.

### Catastrophic Forgetting (망각 현상) 방지 전략
*   두 알고리즘 모두 Policy가 생성한 이전 상태(AMP Obs)를 저장하는 **Replay Buffer**(`_replay_buffer`, `_disc_buffer`)를 운용합니다. Discriminator 업데이트 시 현재의 Policy 데이터뿐만 아니라 버퍼에서 샘플링한 과거의 Policy 데이터를 섞어 판별하게 함으로써 학습 초기의 동작을 잊어버리는 현상을 방지합니다.

---

## 3. Motion Loader 및 데이터 (Motion Loader & Data) 비교
**대상 파일**: `go2_motion_loader.py` (IsaacLab) vs `motion_lib.py` (MimicKit)

### 지원 데이터 형식 및 처리 방식
*   **IsaacLab**:
    *   DeepMimic 계열의 `.txt`(JSON 구조, 61개 값: 속도/위치 사전 계산됨) 및 MuJoCo의 `.pkl`(18개 값: 각도 위주) 포맷을 모두 지원합니다.
    *   `.pkl` 파일 로드 시에는 프레임 간의 시간에 따른 수치 미분(Finite Difference)을 통해 속도(Velocity)를 자동 연산합니다.
*   **MimicKit**:
    *   YAML 설정 파일을 읽어 복수의 모션 파일과 각 모션별 샘플링 비율(Weight)을 로드합니다.
    *   순수하게 각도/위치 위주의 Pose 데이터만을 추출한 뒤, `CircularBuffer` 및 수치 미분을 통해 모든 Root/DoF 속도를 런타임 전에 미리 전처리하여 들고 있습니다.

### Forward Kinematics (FK) 접근법
*   **IsaacLab**: `_go2_fk_toe_pos` 함수에 Go2 로봇의 다리 길이(`_THIGH_LEN`, `_CALF_LEN`) 및 Hip 오프셋 행렬이 완전히 **하드코딩**되어 있습니다. 이는 특정 로봇에 대해 속도를 극대화할 수 있는 장점이 있습니다.
*   **MimicKit**: `_kin_char_model.forward_kinematics()`를 호출합니다. 구조가 파라미터화된 일반화 모델이므로 다른 로봇에 적용 시 코드를 뜯어고칠 필요가 없습니다.

### 샘플링 (Sampling) 전략 및 모션 연속성 (Looping)
*   **IsaacLab (Velocity-balanced Sampling)**: 
    *   보행/달리기 등 속도 변화가 큰 로봇의 특성을 반영하여, 전체 프레임을 전진 속도(Speed Bins) 단위로 나눕니다. 빠른 속도의 모션 프레임 개수가 적더라도, 느린 속도 구간과 동일한 확률로 샘플링되게 가중치를 조정(`_build_velocity_sample_weights`)하여 고속 이동 모션의 모방 성능을 끌어올립니다.
*   **MimicKit (Spatial Wrap & YAML Weights)**:
    *   사용자가 YAML에 명시한 `weight`에 따라 모션을 선택합니다.
    *   `LoopMode.WRAP`이 활성화된 모션의 경우 `_calc_loop_offset`를 통해 모션의 시작과 끝 프레임의 위치 차이(Delta)를 누적 계산합니다. 이를 통해 원점 공간에 한정되지 않고 캐릭터가 연속해서 무한히 전진할 수 있도록 공간적인 이어붙이기를 수행합니다. IsaacLab은 이러한 루핑 대신 에피소드 리셋(RSI)에 의존합니다.
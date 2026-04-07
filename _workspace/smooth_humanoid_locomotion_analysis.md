# Smooth Humanoid Locomotion (LCP) 분석 및 비교 자료

**참고 경로**: `/home/lgb/smooth-humanoid-locomotion`

이 문서는 "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies (LCP)" 논문 기반의 코드베이스 분석 결과를 정리한 것입니다. Claude가 향후 LCP나 PPO-RMA(RMA 기반 시뮬레이션-실제 환경 전이) 기반 알고리즘을 IsaacLab 프로젝트에 적용할 때 유용한 참고 자료가 됩니다.

## 1. 개요 및 핵심 기여
해당 프로젝트는 **Isaac Gym 기반 환경**에서 5개의 휴먼노이드 로봇(Fourier GR1T1/T2, Unitree H1/G1, Berkeley Humanoid)을 제어하기 위해 구현되었습니다.
가장 큰 특징은 부드럽고 떨림(Jitter)이 없는 자연스러운 보행 모션을 얻기 위해, 기존의 Reward Shaping 대신 **Actor 네트워크에 LCP(Lipschitz-Constrained Policies)** 제약을 걸어 동작의 연속성을 보장한 점입니다. 또한, 실기체 전이(Sim-to-Real)를 위한 **PPO-RMA (Rapid Motor Adaptation) 기반 구조**가 도입되었습니다.

## 2. 코드 구조 및 주요 모듈

### A. Environment (Isaac Gym 기반 `legged_gym`)
- **`legged_gym/envs/base/humanoid.py`**: 휴먼노이드 보행을 위한 Base Environment 클래스. 
  - `compute_ref_state`: 발자국 위상의 Sine/Cosine 곡선을 이용해 목표 관절 궤적(Reference State)을 생성하는 방식 사용. (AMP와 다르게 직접 Heuristic Trajectory를 계산함)
  - `_reward_tracking_lin_vel_exp`, `_reward_feet_clearance`, `_reward_feet_air_time` 등 Dense Reward 체계 적용.
  - 다양한 **Terrain Curriculum**과 **Domain Randomization**(질량, 마찰, 모터 강도, 모터 딜레이 등) 구현.
- **로봇별 특화 환경 (e.g., `g1_walk_phase.py`, `h1_walk_phase.py`)**: `Humanoid` 클래스를 상속받아 로봇별 Joint 제어 인덱스, Reward Scale, 모터 강성(Stiffness/Damping) 등 구체적인 설정을 `*_config.py` 에 정의함.

### B. RL Algorithm & Network (`rsl_rl`)
- **`rsl_rl/algorithms/ppo_rma.py`**:
  - 기존 PPO에 **Privileged Information(RMA)** 및 **Gradient Penalty (LCP 관련)** 로직이 추가된 버전.
  - **Dagger Update**: `update_dagger` 메서드를 통해 `History Encoder` 가 `Privileged Encoder`의 출력을 모방하도록 지도학습(Student-Teacher) 방식으로 업데이트함.
  - **Gradient Penalty (`_calc_grad_penalty`)**: 관측값(Observation)의 변화에 따른 Action Log Prob의 변화량(Gradient)을 계산하여 Loss에 추가. 이는 Actor 네트워크의 Lipschitz 상수를 간접적으로 제어하여 **Smoothness(부드러움)**를 향상시킴.
- **`rsl_rl/modules/actor_critic_rma.py`**:
  - `ActorCriticRMA`: `priv_encoder_dims` 를 통해 물리 환경의 숨겨진 정보(마찰, 질량 등)를 압축하는 층과, `StateHistoryEncoder` 를 통해 과거 Observation History를 인코딩하는 층을 분리함.
  - `StateHistoryEncoder`: 1D Convolution(Conv1d)을 이용해 Time-series Observation 히스토리를 효율적으로 처리함.

### C. Deployment & Sim2Sim
- **`sim2sim.py`**: 학습된 JIT 스크립트 모델을 **MuJoCo** 환경에 배포하여 실제 로봇 환경과 유사한 물리 시뮬레이션에서 검증하는 스크립트.
- **`deployment/`**: 실기체 로봇(GR1 등) 구동용 코드 모음.
  - 모터 통신 딜레이(Delay)를 모델링한 필터 적용 (`motor_delay_fft.py` 참조).
  - Isaac Gym의 PD Gain과 실제 로봇의 하드웨어 PD Gain을 매핑하는 유틸리티(`convert_gains.py`) 포함.

## 3. IsaacLab 이식을 위한 주요 아이디어 (LCP & RMA 적용 방안)

### 아이디어 1: LCP 적용을 위한 Gradient Penalty 도입
AMP의 Adversarial 학습이 불안정하거나 로봇의 Action 떨림 현상(Jitter)이 심할 때, Loss 함수에 Gradient Penalty를 추가하여 부드러운 제어 신호를 유도할 수 있습니다.
* **참고 파일**: `rsl_rl/algorithms/ppo_rma.py` 의 `_calc_grad_penalty()`
```python
# LCP 적용 예시
grad_log_prob = torch.autograd.grad(actions_log_prob_batch.sum(), obs_est_batch, create_graph=True)[0]
gradient_penalty_loss = torch.sum(torch.square(grad_log_prob), dim=-1).mean()
```

**LCP 관련 하이퍼파라미터 세팅 (`grad_penalty_coef_schedule`)**:
코드베이스의 각 로봇 환경 설정 파일(`g1_walk_phase_config.py`, `h1_walk_phase_config.py`, `berkeley_walk_phase_config.py` 등)을 보면, 공통적으로 PPO RMA 알고리즘에 다음과 같은 하이퍼파라미터 스케줄을 사용하고 있습니다.
```python
class algorithm(HumanoidCfgPPO.algorithm):
    grad_penalty_coef_schedule = [0.002, 0.002, 700, 1000]
```
- `ppo_rma.py`의 적용 로직: `[start_coef, end_coef, delay_iterations, transition_iterations]` 형태의 스케줄링 구조를 따릅니다.
- **적용 계수(Coefficient)**: 현재 설정상 시작 값(0.002)과 종료 값(0.002)이 동일하므로, 사실상 **모든 로봇에 대해 0.002의 고정된 Gradient Penalty 계수**가 Total Loss 산출에 반영됩니다 (`loss = surrogate_loss + ... + gradient_penalty_coef * gradient_penalty_loss`).
- 설계 의도상으로는 초기 700 이터레이션(`delay`) 동안 학습을 진행한 후 1000 이터레이션에 걸쳐 점진적으로 Penalty 가중치를 변화시킬 수 있게 만들어 두었으나, 최종적으로는 **`0.002`라는 일정한 LCP 제약을 처음부터 가하여 Policy의 Smoothness를 확보**한 것으로 분석됩니다.


### 아이디어 2: PPO-RMA (Rapid Motor Adaptation) 구조
단순 관측값뿐만 아니라 마찰 계수, 질량 등 "Privileged Information"을 Teacher 네트워크가 먼저 학습하고, Student 네트워크(`HistoryEncoder`)가 이를 추론하도록 분리된 2-Phase 학습(또는 병렬 Dagger 학습)을 구성할 수 있습니다. 외란에 대한 강건성(Robustness)이 부족할 때 도입을 고려합니다.
* **참고 파일**: `rsl_rl/modules/actor_critic_rma.py`

### 아이디어 3: History 인코딩용 1D Conv 네트워크
현재 RNN/LSTM 기반의 History 처리 방식 외에 1D Convolution 레이어(`nn.Conv1d`)를 사용해 과거 `obs_history`의 특징을 더 명확하고 빠르게 추출하는 구조를 사용할 수 있습니다.
* **참고 파일**: `rsl_rl/modules/actor_critic_rma.py` 의 `StateHistoryEncoder`

## 4. 참고 사항 (현 프로젝트와 비교)
- **모션 생성 방식**: 본 LCP 코드는 MimicKit과 달리 모방할 `.bvh`나 `.pkl` 파일(Reference Motion)을 사용하지 않고, Sine 곡선 기반의 휴리스틱 보행 궤적(Walk Phase)을 계산하여 목표 관절각(`ref_dof_pos`)을 설정합니다. 우리는 현재 AMP 기반 모션 모방을 사용하므로, LCP의 "Smoothness 제약" 아이디어만 취사 선택하여 AMP Loss와 함께 사용하는 것이 유리합니다.
- **네트워크 정규화**: Observation 정규화에 `RunningMeanStd`를 사용하며, Actor 출력 시에도 분산(std)을 동적으로 조정하거나 고정(`fix_action_std`)하는 기능이 포함되어 있습니다.

---
**작성일**: 2026-04-03
**목적**: Claude가 LCP 모델 및 RMA 관련 디버깅/구현을 진행할 때 즉각 참조할 수 있는 기술 문서.
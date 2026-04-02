# IsaacLab 사족보행 로봇 제어 고도화를 위한 연구 적용 제안서 (최종안)

**작성자:** Research Surveyor & Research Advisor
**참조 프레임워크:** IsaacLab, RSL-RL

## 1. 배경 및 목적
현재 사족보행 로봇(Go2, R_Skeleton)의 보행 학습을 위해 PPO와 AMP(Adversarial Motion Prior)를 사용 중이나, 다양한 보행 스타일의 전환과 제어의 유연성에 한계가 존재합니다. 이를 극복하기 위해 최신(2024-2025) 연구를 바탕으로 IsaacLab 및 RSL-RL 프레임워크에 안정적으로 적용 가능한 새로운 방법론을 제안합니다.

## 2. 조사된 핵심 방법론 (Surveyor 제안)

### 2.1. BCAMP (Behavior-Controllable Adversarial Motion Priors)
- **개요:** 기존 Multi-AMP의 복잡한 다중 판별자 구조 대신, 단일 판별자에 **사용자 명령(User Commands)**을 상태 정보와 함께 조건부(Condition)로 입력하는 방식. (2025년 최신 연구)
- **장점:** 여러 동작(걷기, 뛰기, 제자리 걷기 등)을 하나의 네트워크로 효율적으로 학습하고 실시간으로 매끄럽게 전환할 수 있습니다.

### 2.2. Hierarchical AMP + Residual Control
- **개요:** 저수준(Low-level) 정책은 AMP로 자연스러운 모션을 생성하고, 고수준(High-level) 정책은 험지나 장애물 극복을 위한 잔차(Residual)를 학습하는 계층적 제어 구조.
- **장점:** 동물의 생체 모방적 움직임을 유지하면서 복잡한 지형에서의 강건성을 획기적으로 향상시킬 수 있습니다.

## 3. 적용 가능성 비판적 검토 (Advisor 검증)

Research Advisor의 코드베이스(IsaacLab, RSL-RL) 기반 검토 결과는 다음과 같습니다.

### Hierarchical AMP 모델의 한계
- **문제점:** RSL-RL의 기본 Actor-Critic 아키텍처를 계층적 구조(High-level / Low-level)로 분리하고 제어 주기(update frequency)를 다르게 가져가는 것은 RSL-RL 핵심 코어(PPO 로직)를 대대적으로 수정해야 합니다. 
- **결론:** 단기 적용에 부적합(난이도 '하', 500줄 이상 수정 및 구조 재설계 필요).

### BCAMP의 탁월한 적용 가능성
- **타당성:** RSL-RL의 AMP Discriminator 입력(Observation)에 사용자 명령(예: 목표 속도, 동작 스타일 원핫 인코딩 등) 차원만 추가하면 되므로 기존 프레임워크 구조를 거의 변경하지 않고 적용할 수 있습니다.
- **결론:** 호환성이 매우 높으며 빠른 구현이 가능함 (적용 가능성 '상', 200줄 이내의 ALGO 및 ENV 수정 예상).

## 4. 최종 제안안 (Final Proposal)

Research Surveyor와 Advisor의 치열한 토론 결과, 구현 가능성과 효과를 모두 만족하는 **BCAMP (Behavior-Controllable AMP)** 구조를 IsaacLab 프로젝트에 최우선적으로 도입할 것을 제안합니다.

| 우선순위 | 방법론 | 적용 근거 | 적용 가능성 | 예상 효과 | 구현 범위 (코드 수정) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | **BCAMP** (Command-Conditioned Discriminator) | RSL-RL 구조 변경 최소화 (단일 판별자 유지), 실시간 동작 전환 가능 | 상 | 높음 | ALGO (rsl_rl 내 AMP discriminator 입력부 수정), ENV (명령 스페이스 추가) |
| 2 | wAMP (Wasserstein GAN 기반) | AMP 모드 붕괴(Mode Collapse) 방지 및 다양한 모션 학습 안정성 증대 | 중 | 보통 | ALGO (rsl_rl 내 Discriminator 손실 함수 계산부 WGAN-div 방식으로 수정) |

### 4.1. 다음 실행 단계 (Action Items)
1. **환경(ENV) 수정:** IsaacLab 환경 파일(`source/isaaclab_tasks/...`)에서 로봇의 Observation Space에 보행 스타일(Style Command) 관련 변수를 추가합니다.
2. **알고리즘(ALGO) 수정:** `rsl_rl`의 AMP Discriminator 네트워크가 로봇의 상태뿐만 아니라 Style Command도 함께 입력받을 수 있도록 차원을 변경합니다.
3. **학습 및 검증:** Go2 로봇 모델을 대상으로 여러 모션 데이터셋(걷기, 트롯 등)을 활용해 단일 모델로 다양한 보행 방식을 학습할 수 있는지 실증 테스트를 진행합니다.

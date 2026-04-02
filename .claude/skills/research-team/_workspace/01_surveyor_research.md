# 모방학습 방법론 비교 조사 보고서

**조사일**: 2026-04-02
**조사자**: research-surveyor
**프로젝트 컨텍스트**: IsaacLab (Isaac Sim GPU 가속 RL) + RSL-RL, Go2/R_Skeleton 사족보행 로봇

---

## 프로젝트 현재 상태 요약

현재 프로젝트는 **PPO + AMP (Adversarial Motion Priors)** 방식을 사용하고 있으며, 구현 상태는 다음과 같다:

- **ppo_amp.py**: `PPOParkour`를 상속한 `PPOAMP` 클래스. LS-GAN 기반 discriminator 학습, gradient penalty (expert 데이터에만 적용), task_reward_lerp으로 task/style 보상 혼합
- **amp_discriminator.py**: `AMPDiscriminator` 클래스. EmpiricalNormalization + MLP(1024, 512) → 1 구조. LS-GAN reward: `clamp(1 - 0.25*(d-1)^2, min=0) * coef`
- **go2_amp_env.py**: AMP 관측 벡터 55차원 (dof_pos 12 + dof_vel 12 + root_height 1 + lin_vel 3 + ang_vel 3 + key_body_pos 12 + key_body_lin_vel 12). Go2MotionLoader로 DeepMimic JSON/pkl 모션 데이터 로드

---

## 방법론 비교 분석

### 1. DeepMimic (Peng et al., 2018)

| 항목 | 내용 |
|------|------|
| **논문** | "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills" (ACM TOG 2018) |
| **검증** | 확인됨 - [arXiv:1804.02717](https://arxiv.org/abs/1804.02717) |
| **핵심 아이디어** | 참조 모션 클립과의 직접적인 tracking reward를 설계하여 물리 시뮬레이션 캐릭터가 모션을 모방하도록 학습. 모방 목표와 태스크 목표를 결합하여 다양한 스킬 학습 가능. Discriminator를 사용하지 않고 수동 설계된 reward function 사용. |
| **현재 AMP 대비 장점** | - 모션 tracking 정확도가 높음 (직접 tracking이므로)<br>- 구현이 상대적으로 단순 (discriminator 불필요)<br>- 학습 안정성이 높음 (GAN 학습 불안정성 없음) |
| **현재 AMP 대비 단점** | - reward function 수동 설계 필요 (각 관절별 가중치 튜닝)<br>- 다수 모션 클립 처리 시 모션 선택 메커니즘 필요<br>- 스타일 일반화 어려움 |
| **사족보행 적용 사례** | humanoid 중심이나, 사족보행으로 확장한 다수 후속 연구 존재. MPC 궤적이나 동물 gait 직접 클로닝의 기본 베이스라인으로 널리 사용됨 |
| **rsl_rl 구현 난이도** | **하** - discriminator 제거하고 tracking reward만 추가하면 됨 |
| **예상 효과** | 특정 모션에 대한 정밀 tracking은 우수하나, 다양한 모션 전환이나 스타일 일반화에는 AMP보다 불리 |
| **구현 범위** | ENV만 (reward function 변경) |

---

### 2. AMP - Adversarial Motion Priors (Peng et al., 2021) [현재 사용 중]

| 항목 | 내용 |
|------|------|
| **논문** | "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control" (ACM TOG 2021) |
| **검증** | 확인됨 - [arXiv:2104.02180](https://arxiv.org/abs/2104.02180) |
| **핵심 아이디어** | Discriminator를 사용하여 expert 모션과 policy 모션을 구분하는 adversarial 학습. 모션 클립에서 자동으로 스타일 보상을 학습하여 수동 reward 설계 불필요. 비구조화된 대규모 모션 데이터셋 처리 가능. |
| **현재 구현 상태** | LS-GAN 기반, gradient penalty expert-only, task_reward_lerp=0.5로 task/style 보상 혼합. Discriminator: MLP(1024,512)→1 with EmpiricalNormalization |
| **강점** | - 수동 reward 설계 불필요<br>- 다수 모션 클립 자동 처리<br>- 실제 사족보행 로봇 (German Shepherd 모션) 전이 성공 사례 있음<br>- Isaac Gym에서 6분 내 학습 가능 (A100 기준) |
| **약점** | - GAN 학습 불안정성<br>- mode collapse 가능성<br>- discriminator와 policy 간 학습 속도 균형 필요<br>- 스킬 재사용/조합 메커니즘 없음 |
| **구현 범위** | 현재 구현 완료 (ENV + ALGO) |

---

### 3. AWR - Advantage-Weighted Regression (Peng et al., 2019)

| 항목 | 내용 |
|------|------|
| **논문** | "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning" (2019) |
| **검증** | 확인됨 - [OpenReview](https://openreview.net/pdf/ec69fdc5cafd6a55f98afb0ffea7d424eaee6034.pdf), [GitHub: xbpeng/awr](https://github.com/xbpeng/awr) |
| **핵심 아이디어** | Policy 최적화를 supervised regression 문제로 변환. 각 이터레이션에서 (1) value function 회귀 학습, (2) advantage-weighted 정책 회귀 학습의 두 단계만 수행. 온/오프폴리시 데이터 모두 활용 가능한 단순한 알고리즘. |
| **현재 AMP 대비 장점** | - 구현 극도로 단순 (수 줄 코드)<br>- Off-policy 데이터 활용 가능 (데모 데이터 직접 통합)<br>- GAN 학습 불안정성 없음<br>- 데모 데이터와 RL 자연스럽게 결합 |
| **현재 AMP 대비 단점** | - 스타일 보상 자동 학습 불가 (별도 reward 필요)<br>- 모션 품질이 AMP보다 낮을 수 있음 (adversarial 세분화 없음)<br>- 주로 manipulation task에서 검증됨 |
| **사족보행 적용 사례** | MuJoCo locomotion 벤치마크에서 검증. AW-Opt 등 후속 연구에서 로봇 manipulation에 활용. 사족보행 직접 적용 사례는 제한적. |
| **rsl_rl 구현 난이도** | **하** - PPO 대체 알고리즘으로 구현, supervised regression만 필요 |
| **예상 효과** | 보행 품질 개선보다는 오프라인 데모 데이터 활용 효율성 향상에 유리. AMP와 조합하기보다는 대안적 접근. |
| **구현 범위** | ALGO만 (PPO 대체) |

---

### 4. ASE - Adversarial Skill Embeddings (Peng et al., 2022)

| 항목 | 내용 |
|------|------|
| **논문** | "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters" (ACM TOG / SIGGRAPH 2022) |
| **검증** | 확인됨 - [arXiv:2205.01906](https://arxiv.org/abs/2205.01906), [GitHub: nv-tlabs/ASE](https://github.com/nv-tlabs/ASE) |
| **핵심 아이디어** | AMP를 확장하여 latent skill embedding space를 학습. Adversarial imitation learning + unsupervised RL을 결합하여 재사용 가능한 스킬 임베딩 생성. 저수준 스킬 컨트롤러를 먼저 학습한 후, 고수준 태스크 정책이 스킬 공간에서 동작을 선택. |
| **현재 AMP 대비 장점** | - 스킬 재사용 및 조합 가능 (AMP는 불가)<br>- 대규모 모션 데이터셋에서 다양한 스킬 자동 추출<br>- 새 태스크에 빠른 전이 (저수준 스킬 재학습 불필요)<br>- 더 자연스러운 모션 전환 |
| **현재 AMP 대비 단점** | - 구현 복잡도 크게 증가 (2단계 학습)<br>- 학습 시간 대폭 증가<br>- latent space 크기/구조 튜닝 필요<br>- NVIDIA Isaac Gym 전용 구현 (rsl_rl 포팅 필요) |
| **사족보행 적용 사례** | humanoid 중심이나, quadrupedal loco-manipulation에 확장 적용된 사례 있음. Dog motion retargeting과 결합 가능. |
| **rsl_rl 구현 난이도** | **상** - 2단계 학습 파이프라인, latent encoder/decoder 추가, 저수준/고수준 정책 분리 필요 |
| **예상 효과** | 다양한 보행 패턴 (걷기, 트로팅, 갤로핑) 간 자연스러운 전환. 장기적으로 가장 큰 품질 향상 기대. 그러나 구현 비용 높음. |
| **구현 범위** | 둘 다 (ENV + ALGO 대폭 변경) |

---

### 5. LCP - Lipschitz-Constrained Policies (Zhu et al., 2024)

| 항목 | 내용 |
|------|------|
| **논문** | "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies" (ICRA 2025) |
| **검증** | 확인됨 - [arXiv:2410.11825](https://arxiv.org/abs/2410.11825) |
| **후속 논문** | "Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion" (2025) - [arXiv:2504.08246](https://arxiv.org/abs/2504.08246) |
| **핵심 아이디어** | 정책 네트워크에 Lipschitz 연속성 제약을 부과하여 부드러운 행동 출력 보장. 입력의 작은 변화가 출력의 작은 변화로 이어지도록 하여 액추에이터 진동(jitter) 제거. Gradient penalty 또는 spectral normalization으로 구현. |
| **현재 AMP 대비 장점** | - smoothness reward 또는 low-pass filter 불필요<br>- 구현 단순 (네트워크 레이어에 제약 추가만)<br>- AMP와 **직교적으로 결합 가능** (상호 배타적이지 않음)<br>- sim-to-real 전이 성능 향상<br>- GPU 메모리 효율적 (spectral norm 버전) |
| **현재 AMP 대비 단점** | - 모션 스타일 학습 기능 없음 (별도 목적)<br>- 지나친 제약 시 성능 저하 가능<br>- humanoid 중심 검증 (사족보행 직접 검증은 제한적) |
| **사족보행 적용 사례** | 주로 humanoid locomotion에서 검증. Isaac Gym 4096 병렬 환경에서 학습. 사족보행 직접 적용은 아직 제한적이나 원리적으로 완전히 호환됨. |
| **rsl_rl 구현 난이도** | **하** - actor network에 spectral normalization 또는 gradient penalty 추가만 필요 |
| **예상 효과** | **발 충격 최소화, 부드러운 동작**이라는 프로젝트 목표에 직접 기여. AMP와 결합 시 자연스러움(AMP) + 부드러움(LCP) 동시 달성 가능. |
| **구현 범위** | ALGO만 (actor network 수정) |

---

### 6. ADD - Adversarial Differential Discriminator (Yang, Zhang et al., 2025)

| 항목 | 내용 |
|------|------|
| **논문** | "ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators" (SIGGRAPH Asia 2025) |
| **검증** | 확인됨 - [arXiv:2505.04961](https://arxiv.org/abs/2505.04961), [GitHub: xbpeng/MimicKit](https://github.com/xbpeng/MimicKit) |
| **핵심 아이디어** | 다목적 최적화(MOO)에 adversarial differential discriminator를 적용. 단일 positive sample만으로 효과적 학습 가능. Discriminator가 학습 중 동적으로 목표 간 가중치를 조정하여 수동 reward 설계 및 하이퍼파라미터 튜닝 최소화. |
| **현재 AMP 대비 장점** | - **단일 positive sample**만으로 학습 가능 (AMP는 다수 모션 클립 필요)<br>- reward function 수동 설계 완전 제거<br>- 태스크 전환 시 하이퍼파라미터 재튜닝 불필요<br>- DeepMimic 수준의 tracking 품질을 adversarial 방식으로 달성 |
| **현재 AMP 대비 단점** | - 2025년 최신 논문으로 커뮤니티 검증 아직 부족<br>- MimicKit 프레임워크 기반 (rsl_rl 포팅 필요)<br>- 다목적 최적화 개념 이해 필요 |
| **사족보행 적용 사례** | **Unitree Go1 사족보행 태스크에서 직접 검증됨**. 더 자연스러운 gait (큰 발 들어올림, 긴 보폭) 달성. 2D Walker 태스크에서도 검증. |
| **rsl_rl 구현 난이도** | **중** - discriminator 구조 변경 필요 (differential discriminator), MOO 학습 루프 추가 |
| **예상 효과** | 모션 데이터가 제한적인 상황에서 높은 품질 기대. Go1에서 검증되었으므로 Go2 적용 유망. reward 튜닝 부담 대폭 감소. |
| **구현 범위** | ALGO 주로 (discriminator 교체 + 학습 루프 수정) |

---

### 7. GMP - Generative Motion Prior (Zhang et al., 2025)

| 항목 | 내용 |
|------|------|
| **논문** | "Natural Humanoid Robot Locomotion with Generative Motion Prior" (2025) |
| **검증** | 확인됨 - [arXiv:2503.09015](https://arxiv.org/abs/2503.09015), [프로젝트 페이지](https://sites.google.com/view/humanoid-gmp) |
| **핵심 아이디어** | Conditional VAE (CVAE)로 생성 모델을 오프라인 학습하여 미래 자연스러운 참조 모션을 예측. 학습 시 frozen motion generator로 사용하여 trajectory-level supervision (관절 각도 + keypoint 위치) 제공. AMP의 불안정한 스타일 보상 대신 직접적이고 세밀한 모션 가이던스 제공. |
| **현재 AMP 대비 장점** | - AMP보다 안정적인 학습 (GAN 학습 불안정성 없음)<br>- 더 세밀한 모션 supervision (trajectory-level)<br>- 학습 해석성 향상 (어떤 참조 모션을 따르는지 명확)<br>- 모션 자연스러움에서 기존 방법 대비 우수한 실험 결과 |
| **현재 AMP 대비 단점** | - CVAE 사전 학습 단계 추가 필요<br>- humanoid 중심 설계 (사족보행 적용 시 retargeting 필요)<br>- 2025년 최신 논문으로 코드 공개 상태 불확실<br>- 모션 데이터 품질에 크게 의존 |
| **사족보행 적용 사례** | humanoid locomotion에서 시뮬레이션 + 실기체 검증. 사족보행 직접 적용은 없으나, dog motion 데이터를 CVAE로 학습하여 적용하는 것은 원리적으로 가능. |
| **rsl_rl 구현 난이도** | **상** - CVAE 모델 별도 구현/학습, motion retargeting 파이프라인, frozen generator 통합 필요 |
| **예상 효과** | 자연스러움 측면에서 가장 높은 품질 기대. 그러나 구현 비용과 사족보행 적응 비용이 높음. |
| **구현 범위** | 둘 다 (ENV: 참조 모션 생성 + ALGO: CVAE 학습/추론 통합) |

---

## 종합 비교표

| # | 방법 | 논문 (venue/연도) | 핵심 기여 | 검증 | IsaacLab 적용 가능성 | 구현 범위 | 구현 난이도 |
|---|------|-------------------|----------|------|---------------------|----------|-----------
| 1 | DeepMimic | ACM TOG 2018 | 참조 모션 직접 tracking + RL | 확인 | **중** - AMP 대비 후퇴이나 tracking 정확도 높음 | ENV | 하 |
| 2 | AMP | ACM TOG 2021 | Adversarial style reward 자동 학습 | 확인 | **상** - 현재 구현 완료 | - | - |
| 3 | AWR | 2019 | Supervised regression으로 policy 최적화 | 확인 | **하** - 사족보행 스타일 학습에 부적합 | ALGO | 하 |
| 4 | ASE | SIGGRAPH 2022 | 재사용 가능한 스킬 임베딩 | 확인 | **중** - 장기적 가치 높으나 구현 비용 큼 | 둘 다 | 상 |
| 5 | LCP | ICRA 2025 | Lipschitz 제약으로 부드러운 동작 | 확인 | **상** - AMP와 직교적 결합, 즉시 적용 가능 | ALGO | 하 |
| 6 | ADD | SIGGRAPH Asia 2025 | Differential discriminator로 MOO 최적화 | 확인 | **상** - Go1에서 검증, AMP discriminator 교체 | ALGO | 중 |
| 7 | GMP | arXiv 2025 | CVAE 기반 trajectory-level motion prior | 확인 | **중** - 높은 품질이나 구현 비용 높음 | 둘 다 | 상 |

---

## 권장 구현 순서 (초안)

### 1순위: LCP (Lipschitz-Constrained Policies) + 현재 AMP
- **이유**: 구현 난이도 최저, AMP와 직교적 결합 가능, 프로젝트 목표(발 충격 최소화, 부드러운 동작)에 직접 기여
- **예상 작업**: actor network에 spectral normalization 적용 (수 시간 내 구현 가능)
- **기대 효과**: 보행 부드러움 즉시 개선, sim-to-real 전이 성능 향상

### 2순위: ADD (Adversarial Differential Discriminator)
- **이유**: Go1(Go2와 유사)에서 직접 검증, 기존 AMP discriminator를 교체하는 형태로 구현 가능, reward 튜닝 부담 감소
- **예상 작업**: AMPDiscriminator를 ADDDiscriminator로 교체, MOO 학습 루프 수정 (1-2주)
- **기대 효과**: 모션 품질 향상, 하이퍼파라미터 튜닝 부담 감소

### 3순위: ASE (Adversarial Skill Embeddings)
- **이유**: 다양한 보행 패턴 전환에 장기적 가치. Isaac Gym 구현이 이미 존재하므로 참조 가능
- **예상 작업**: 2단계 학습 파이프라인 구축, latent space 설계 (3-4주)
- **기대 효과**: 걷기/트로팅/갤로핑 간 자연스러운 전환

### 비권장: AWR (사족보행 스타일 학습에 부적합), GMP (구현 비용 대비 불확실한 이점)

---

## 참고 문헌

1. Peng et al., "DeepMimic: Example-Guided Deep Reinforcement Learning of Physics-Based Character Skills", ACM TOG 2018 - [arXiv:1804.02717](https://arxiv.org/abs/1804.02717)
2. Peng et al., "AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control", ACM TOG 2021 - [arXiv:2104.02180](https://arxiv.org/abs/2104.02180)
3. Peng et al., "Advantage-Weighted Regression: Simple and Scalable Off-Policy Reinforcement Learning", 2019 - [GitHub](https://github.com/xbpeng/awr)
4. Peng et al., "ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters", SIGGRAPH 2022 - [arXiv:2205.01906](https://arxiv.org/abs/2205.01906)
5. Zhu et al., "Learning Smooth Humanoid Locomotion through Lipschitz-Constrained Policies", ICRA 2025 - [arXiv:2410.11825](https://arxiv.org/abs/2410.11825)
6. Yang, Zhang et al., "ADD: Physics-Based Motion Imitation with Adversarial Differential Discriminators", SIGGRAPH Asia 2025 - [arXiv:2505.04961](https://arxiv.org/abs/2505.04961)
7. Zhang et al., "Natural Humanoid Robot Locomotion with Generative Motion Prior", 2025 - [arXiv:2503.09015](https://arxiv.org/abs/2503.09015)
8. "Spectral Normalization for Lipschitz-Constrained Policies on Learning Humanoid Locomotion", 2025 - [arXiv:2504.08246](https://arxiv.org/abs/2504.08246)

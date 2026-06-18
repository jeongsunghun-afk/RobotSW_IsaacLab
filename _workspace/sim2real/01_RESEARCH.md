# Actuator Modeling & System Identification 논문 20편

> **초점**: legged 로봇 sim-to-real에서 **actuator 동역학 모델링**과 **system identification(질량/관성/마찰/latency/모터 파라미터 식별)** 자체가 핵심 기여인 논문. 강복님 마스터 플랜의 **C3(System Identification = 무게중심)** 와 직결.
>
> **접근법 분류**
> - **NN-Actuator**: 신경망 actuator 모델(black-box)
> - **Analytic/Parametric**: 해석적·파라메트릭 모터/마찰 모델
> - **SysID-Measurement**: 토크/전류 실측 기반 식별
> - **SysID-Sampling**: 샘플링·active exploration 기반(미분 불필요)
> - **SysID-Diff**: differentiable simulation 기반 gradient 식별
> - **Real2Sim**: 실기 rollout로 sim 파라미터 보정 루프
>
> 강복님 강점(토크/전류 실측 + encoder)은 **SysID-Measurement**를 완전 supervised 회귀로 만들고, 동시에 **SysID-Sampling/Diff**(torque-sensorless) 계열도 교차검증용으로 활용 가능.

---

## A. Actuator Net 계보 — 신경망 모터 모델 (NN-Actuator)

| # | 논문 | 출처/연도 | 핵심 기여 | 접근 | 링크 |
|---|------|----------|----------|------|------|
| 1 | **Learning Agile and Dynamic Motor Skills for Legged Robots** (Hwangbo et al., ETH) | Science Robotics 2019 | **actuator net** 원전 — SEA 토크 실측으로 NN이 (위치·속도 오차 history)→토크 학습, sim에 삽입. C3의 정점 | NN-Actuator + Measurement | https://arxiv.org/abs/1901.08652 |
| 2 | **Learning Quadrupedal Locomotion over Challenging Terrain** (Lee et al., ETH/KAIST) | Science Robotics 2020 | actuator net + privileged TS로 야외 zero-shot. actuator 모델을 지형 일반화와 결합 | NN-Actuator | https://arxiv.org/abs/2010.11251 |
| 3 | **Learning Robust Autonomous Navigation for Wheeled-Legged Robots** (Lee et al., ETH) | Science Robotics 2024 | SEA는 actuator net, 휠은 토크센서 없어 (속도명령+history)→**모터 전류** NN 매핑 후 τ=Kτ·GR·I. 센서 없는 구동부 식별 확장 | NN-Actuator | https://arxiv.org/abs/2405.01792 |
| 4 | **Bridging the Sim-to-Real Gap (torque-sensorless actuator net)** (Fey et al.) | 2025 | 토크센싱 없이 actuator net 학습 — 소비자급 하드웨어로 actuator net 일반화 | NN-Actuator | https://arxiv.org/abs/2502.xxxx (Fey 2025, 원문 확인 필요) |

---

## B. 해석적 / 파라메트릭 모터·마찰 모델 (Analytic/Parametric)

| # | 논문 | 출처/연도 | 핵심 기여 | 접근 | 링크 |
|---|------|----------|----------|------|------|
| 5 | **Sim-to-Real: Learning Agile Locomotion for Quadruped Robots** (Tan et al., Google) | RSS 2018 | **해석적 actuator 모델 + latency 모델링 + 모터마찰 측정 실험** + URDF 질량/관성 식별. sim-to-real SysID 원전 | Analytic + Measurement | https://arxiv.org/abs/1804.10332 |
| 6 | **Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model** | arXiv 2026 | 300kg+ 유압 4족용 **해석적 actuator 모델**, 토크 1μs 예측, 데이터 적은 상황서 NN 대비 우위 | Analytic | https://arxiv.org/abs/2601.11143 |
| 7 | **Impedance Matching: RL-Based Running Jump** (KAIST) | arXiv 2024 | **주파수영역 임피던스 매칭**으로 sim/real actuator 정렬 + DR 범위 선정 가이드 | Parametric | https://arxiv.org/abs/2404.15096 |
| 8 | **Torque Clipping in Realistic Motor Operating Region** (Shin et al., KAIST) | IEEE RAM 2024 | 모터 동작영역(MOR) 제약을 모델에 반영해 모터모델 불일치 완화 | Parametric | https://arxiv.org/abs/2403.03212 |
| 9 | **Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less Gear-Driven Humanoid** (Preferred Networks) | IROS 2022 | DTE 포함 actuator 모델 + **실패한 전이의 보상으로 재식별(re-identification)**. 기어구동·센서리스 | Parametric + Real2Sim | https://arxiv.org/abs/2204.03897 |

---

## C. 전신 System Identification — 측정·샘플링·미분 기반

actuator뿐 아니라 질량·관성·마찰·latency를 **시스템 레벨**로 식별하는 그룹. 강복님 C3의 핵심.

| # | 논문 | 출처/연도 | 핵심 기여 | 접근 | 링크 |
|---|------|----------|----------|------|------|
| 10 | **PACE: Systematic Sim-to-Real for Diverse Legged Robots** | arXiv 2025 | per-joint armature·viscous damping·Coulomb 마찰·joint bias·global delay를 직접 fit → **DR 없이 zero-shot**. actuator net 충실도를 더 적은 데이터로 매칭. *(강복님 기존 리뷰)* | SysID-Sampling/Param | https://arxiv.org/abs/2509.06342 |
| 11 | **SPI-Active: Sampling-based Parameter ID with Active Exploration** (Sobanbabu, He, Shi et al., CMU) | arXiv 2025 | **미분불가·토크센서 없이** 대규모 병렬 샘플링으로 관성·actuator 식별 + Fisher Information 기반 active excitation 설계 | SysID-Sampling | https://arxiv.org/abs/2505.14266 |
| 12 | **Trajectory-based Actuator Identification via Differentiable Simulation** | arXiv 2026 | MJX로 **토크/전류센서 없이** encoder 궤적만으로 actuator(armature·damping) 식별. 구조형·NN·엔진 파라미터 통합 최적화. 보행 거리 +46% | SysID-Diff | https://arxiv.org/abs/2604.10351 |
| 13 | **Achieving Precise Locomotion with Differentiable Simulation-Based SysID** (Kovalev et al.) | arXiv 2025 | MuJoCo-XLA로 **SysID를 RL 학습 루프에 통합**, 궤적(위치·속도)만으로 파라미터 추정(토크 불필요) | SysID-Diff | https://arxiv.org/abs/2508.04696 |
| 14 | **HALO: Closing Sim-to-Real Gap for Heavy-loaded Humanoid via Differentiable Sim** | arXiv 2026 | 2단계 gradient SysID(MuJoCo XLA): ①nominal 모델 보정 ②payload 조건 추가 식별. 중량부하 보행 | SysID-Diff + Real2Sim | https://arxiv.org/abs/2603.15084 |
| 15 | **Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation** | arXiv 2025 | 관성 파라미터 식별 + actuator 모델링 결합으로 athletic 동작 전이. SysID 한계(센서리스) 논의 | SysID-Measurement | https://arxiv.org/abs/2502.10894 |

---

## D. Real2Sim 보정 루프 — 실기로 sim 파라미터 역추적 (Real2Sim)

학습 후가 아니라 sim 자체를 실기에 맞춰 보정. 강복님 플랜의 **real2sim 닫힌 루프(C4)** 와 직접 정렬.

| # | 논문 | 출처/연도 | 핵심 기여 | 접근 | 링크 |
|---|------|----------|----------|------|------|
| 16 | **Closing the Sim-to-Real Loop (SimOpt)** (Chebotar et al., NVIDIA) | ICRA 2019 | 실기 rollout 몇 회로 **DR 분포 자체를 적응** — sim/real 행동 매칭. real2sim 보정 원전 | Real2Sim | https://arxiv.org/abs/1810.05687 |
| 17 | **Simulator Adaptation via Proprioceptive Distribution Matching** | arXiv 2026 | HW/sim rollout을 **분포로 비교**(시간정렬·외부센서 불필요), 5분 데이터로 파라미터·action-delta·residual actuator 식별 | Real2Sim + SysID | https://arxiv.org/abs/2604.11090 |
| 18 | **Real-time MPC and SysID Using Differentiable Physics** (Jin et al.) | RA-L 2023 | differentiable sim 그래디언트로 **control과 SysID를 온라인 동시** 수행 — 배포 후 지속 적응 | SysID-Diff + Real2Sim | https://arxiv.org/abs/2202.09834 |

---

## E. 기반·방법론 (Foundational / Method)

actuator·SysID 기법의 이론적 토대.

| # | 논문 | 출처/연도 | 핵심 기여 | 접근 | 링크 |
|---|------|----------|----------|------|------|
| 19 | **Differentiable Simulation for Physical System Identification** (Le Lidec et al.) | RA-L 2021 | differentiable sim의 **contact 모델이 gradient 거동을 좌우** — well-behaved gradient 위한 접촉모델 제시. SysID-Diff의 이론 기반 | SysID-Diff (이론) | https://arxiv.org/abs/2103.xxxxx (원문 확인 필요) |
| 20 | **Concurrent Training of Control Policy & State Estimator** (Ji, Mun, Kim, Hwangbo, KAIST) | RA-L 2022 | policy와 **base-state estimator 동시 학습** — 식별된 동역학을 실기 상태추정으로 연결(C3→EST 다리) | Estimator (SysID 인접) | https://arxiv.org/abs/2202.05481 |

---

## 부록 — 강복님 플랜(C3) 적용 매핑

| 접근법 | 강복님 자원/플랜과의 정합 | 대표 논문 |
|--------|--------------------------|-----------|
| **NN-Actuator** | 토크/전류 실측 → actuator net을 *완전 supervised 회귀*로 (가장 강력) | #1, #2, #3 |
| **Analytic/Parametric** | PMSM 에너지 reward·모터모델 nominal 주입(`rga.py`) | #5, #6, #7, #8 |
| **SysID-Sampling** | 토크센서 의존 줄이는 교차검증, excitation 설계(T5) | #10, #11 |
| **SysID-Diff** | IsaacLab/MJX 환경서 gradient 식별, 토크 불필요 경로 | #12, #13, #14 |
| **Real2Sim** | 플랜의 real2sim 닫힌 루프(divergence→역추적→재보정) | #16, #17, #18 |

### 권장 리뷰 순서 (C3 깊이 우선)
1. **#1 (Actuator Net, Hwangbo 2019)** — 강복님 기준점, 토크 실측 강점과 정확히 일치
2. **#11 (SPI-Active)** — excitation 설계(Fisher Info) + 샘플링 식별, T5 excitation 단계 직접 참고
3. **#12 (Trajectory-based Diff ID)** — 토크센서 없는 교차검증 경로, IsaacLab/MJX 정합
4. **#10 (PACE)** — 이미 리뷰, measured-centered·zero-shot 사상과 직결(재확인용)
5. **#17 (Distribution Matching)** — real2sim 정량 메트릭(분포비교) 설계

### 주의
- #4, #19의 arXiv 번호는 검색 스니펫 기반이라 부정확할 수 있음 — 원문 검색 후 확정 권장.
- #10(PACE)은 강복님이 이미 5-섹션 리뷰 완료한 논문. C3 맥락에서 actuator 식별 부분만 재참조 추천.
- 원하는 번호 지정 시 강복님 5-섹션 템플릿((1)요약 (2)motivation (3)method+수식/파이프라인 (4)문제점·개선 (5)프로젝트 적용)으로 심화 리뷰 진행.

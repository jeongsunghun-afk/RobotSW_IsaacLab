# 카테고리 A 심화 리뷰 — NN-Actuator 계보 (#1~#4)

> 리뷰어: Sim2Real 논문 심화 리뷰어 (원문 직접 확인 기반, 스니펫 추론 배제)
> 맥락: 강복님은 **토크/전류 실측 + encoder 보유** → actuator net을 *완전 supervised 회귀*로 만들 수 있는 강점. 환경은 **IsaacLab / MJX**.
> 4편 모두 NN-Actuator 계보(ETH 라인 #1→#2→#3, MIT 라인 #4). #1이 원전이자 1순위.

---

## #1. Learning Agile and Dynamic Motor Skills for Legged Robots (Hwangbo et al., ETH, Science Robotics 2019)
- **확정 링크**: https://arxiv.org/abs/1901.08652
- **접근 분류**: NN-Actuator (+ SysID-Measurement)
- **우선순위**: **1순위** (01_RESEARCH.md 권장순서 #1 — 강복님 토크 실측 강점과 정확히 일치하는 기준점)

### (1) 요약
Actuator network(액추에이터 망)의 **원전 논문**. ANYmal의 Series-Elastic Actuator(SEA)는 직렬 스프링 + 케이블 구동 + 모터 제어기 캐스케이드로 동역학이 비선형/지연이 커서 해석적 모델링이 어렵다. 저자들은 실 로봇 SEA에서 (위치오차·속도 history)→(실제 출력 토크) 데이터를 수집해 작은 MLP로 supervised 회귀하고, 이 학습된 actuator net을 강체(rigid-body) 시뮬레이터에 끼워 넣어 **하이브리드 시뮬레이터**를 만든다. 이 위에서 RL로 학습한 정책을 실 ANYmal에 zero-shot 전이하여, (a) 명령 추종 보행(이전 모델기반 대비 정확·고효율), (b) 1.2→1.5 m/s 고속주행(25% 향상), (c) 임의 자세 낙하복구(9/9 성공)를 달성했다.

### (2) Motivation
당시 legged RL은 시뮬레이션에 갇혀 있었다. 동적 균형 시스템은 실기 학습이 위험·고비용이고, sim-to-real 격차의 최대 원인이 **액추에이터 동역학**(특히 SEA의 토크 지연·대역폭 한계·케이블 마찰)이었다. 이상적 토크 소스 가정은 실패한다(이상적 모델 오차 3.55~5.74 Nm). 핵심 통찰: 액추에이터를 해석적으로 모델링하지 말고, 실측 데이터로 *그 입출력 사상 자체를 학습*하면 된다.

### (3) Method + 수식/파이프라인 + 데이터/학습/실험
**Actuator net 구조 (핵심)**
- MLP, hidden 3층 × 32 units, activation = **Softsign**(tanh 대비 빠름: 12 관절 12.2 μs vs 31.6 μs — 실시간 sim 속도 목적).
- 입력: **history 3 timestep** — 현재 t, t−0.01s, t−0.02s. 각 step에 [관절 위치오차(=명령−실제), 관절 속도]. (관절당 6차원)
- 출력: 관절 토크(12관절 각각).

**데이터 수집 (강복님 강점과 직결)**
- 실 ANYmal SEA에서 수집. IK로 만든 **파라메트릭 sine 발 궤적**: 진폭 5~10 cm, 주파수 sweep 1~25 Hz, + 수동 외란으로 주파수 커버리지 강화.
- 12개 동일 액추에이터 병렬 로깅 → 수집시간 **<4분**, 400 Hz → **>100만 샘플**, train/val ≈ 90/10.

**Loss / 검증**
- supervised 회귀(MSE). 검증오차 **0.740 Nm RMS**, 정책 실행 궤적상 테스트오차 **0.966 Nm RMS** (이상적 모델 3.55~5.74 Nm 대비 압도적).

**RL 파이프라인**
- 알고리즘 **TRPO**(기본 파라미터). 정책망 MLP 256-128, tanh(실기 과격행동 억제 목적), 출력 = 관절 **위치 목표**(토크 직접 X) → 고정 게인 PD(kp=50, kd=0.1)로 변환.
- 보상(예): 선/각속도 추종(logistic kernel `K(x)=−1/(e^x+2+e^{−x})`), 토크 penalty 0.005, 관절속도 0.03, foot clearance 0.1, slip 2.0. 복구는 orientation cost 6.0 + 목표자세.
- **커리큘럼**: cost 계수 `k_c`를 0.3→1.0으로 지수 증가(`k_{c,j+1}=(k_{c,j})^{k_d}`, k_d=0.997) — 조기 standing 수렴 방지.

**실험·메트릭**
1. 명령추종: lin 0.143 m/s·yaw 0.174 rad/s 오차(모델기반 0.231/0.278), 기계동력 78.1 W(vs 97.3), 토크 8.23 Nm(vs 11.7).
2. 고속주행: 실기 **1.50 m/s**(sim 1.58), 종전 1.2 m/s 대비 +25%, 토크/속도 한계 포화.
3. 낙하복구: 9개 임의 방향 100% 성공, <3초, 41 충돌체.

### (4) 문제점·한계·개선 여지
- **액추에이터 독립 가정**: 관절 간 결합(공유 유압 어큐뮬레이터 등)에는 부적합.
- **토크 센싱 인프라 필요**(SEA가 스프링 변형으로 토크를 측정 가능했기에 가능) — 일반 소비자급 PMSM엔 직접 적용 어려움(→ #4가 이 지점 공략).
- 단일 태스크당 단일 망. 비용함수/초기분포 수동 튜닝(~2일).
- 강체+CAD 관성 가정(±20%는 randomization으로 흡수).

### (5) 우리 적용 (강복님 자원 기준)
**가장 직접적인 정합.** 강복님은 토크/전류 실측이 가능하므로 본 논문의 *완전 supervised 회귀*를 그대로 재현할 수 있다(센서 없는 #4보다 훨씬 단순·정확).
- 데이터: Go2 등 대상 로봇에서 sine sweep(1~25 Hz, 진폭 sweep) + Gaussian 외란 궤적으로 PD 명령을 주고 [위치오차·속도] history와 실측 토크 로깅. 4분 분량이면 충분(병렬 관절 로깅).
- 망: hidden 3×32, history 3-step, Softsign(또는 ELU). loss=MSE. 검증오차 <1 Nm 목표.
- IsaacLab 통합: `ImplicitActuator` 대신 커스텀 actuator 모델 클래스에서 net을 호출하도록 삽입(아래 인프라 참조). 정책 출력=위치목표 유지하면 본 구조 그대로 호환.
- 주의: 강복님 actuator가 SEA가 아니라 quasi-direct-drive면 위치오차→토크 사상은 더 단순할 수 있으나, **전류→토크 비선형(하모닉 감속기)**이 있으면 #4식 보정이 보완책.

### sim 구현 난이도: **중**
- 필요 인프라: (a) 실측 리그 — PD 명령 가능 + 관절 토크/전류 동기 로깅(encoder는 이미 보유), (b) IsaacLab `actuator` 모듈에 학습된 MLP를 삽입하는 커스텀 ActuatorNet 클래스(forward에서 위치오차·속도 history 버퍼 유지 → 토크 반환), (c) 회귀 학습 스크립트(PyTorch). 코어 수정 없이 task별 actuator cfg로 주입 가능. 데이터 수집이 짧아(<4분) 진입장벽 낮음.

---

## #2. Learning Quadrupedal Locomotion over Challenging Terrain (Lee et al., ETH/KAIST, Science Robotics 2020)
- **확정 링크**: https://arxiv.org/abs/2010.11251
- **접근 분류**: NN-Actuator (#1 actuator net 재사용)
- **우선순위**: 2순위 (actuator net을 지형 일반화·privileged TS와 결합한 응용 정점)

### (1) 요약
#1의 actuator net을 **그대로 재사용**하고, 그 위에 **teacher-student privileged learning** + **적응적 지형 커리큘럼**을 얹어, 진흙·눈·물·초목·자갈 등 *훈련에서 본 적 없는* 자연지형을 proprioception(고유수용)만으로 zero-shot 보행. Teacher는 특권정보(지형·접촉)로 RL 학습, Student는 센서만으로 TCN(시간합성곱)이 과거 proprioception history를 인코딩해 teacher를 모방. DARPA SubT에서 4회 60분 미션 무실패.

### (2) Motivation
실 로봇엔 정확한 지형 정보가 없다. exteroception(카메라/라이다)은 진흙·풀에서 신뢰 못 한다. 동물처럼 **발에 닿는 감각(proprioception)만으로** 지형을 *추론*해 강건하게 걷고 싶다. 격차의 핵심은 여전히 액추에이터(→#1로 해결됨)와 지형 분포(→커리큘럼)이다.

### (3) Method + 수식/파이프라인 + 데이터/학습/실험
**Actuator 모델**: #1의 SEA actuator net **무수정 재사용** (입력 6차원: 위치오차·속도 × 현재+t−0.01+t−0.02). sim-to-real 전이의 토대.

**구조**
- **Teacher**: 2-MLP. 인코더가 특권정보 x_t(지형 프로파일·접촉상태/힘·마찰·외란)를 잠재 `l̄_t`로 압축 → o_t와 함께 16차원 action(다리 주파수 f_i + 발위치 잔차).
- **Student(TCN)**: dilated causal conv 3층 + strided, history **N=100(2초@50Hz)**. 출력은 teacher와 동일 포맷(action + latent). PMTG(Trajectory Generator 변조) 구조: `r_{f_i}=F(ϕ_i)+Δr_{f_i}`.

**학습**
- Teacher: **TRPO**. 보상 가중: lin 0.05, ang 0.05, base stability 0.04, foot clearance 0.01, body collision 0.02, smoothness 0.025, torque 2e-5. + domain randomization(마찰·외란·관측노이즈).
- Student: **behavioral cloning + DAgger**. loss = action 모방 + **latent 재구성**:
  `L = (ā_t − a_t)² + (l̄_t − l_t)²` (특권 잠재를 history로부터 복원하도록 강제 → belief 학습).
- **적응적 지형 커리큘럼**: particle filter로 traversability `Tr∈[0.5,0.9]`(적당한 난이도) 영역의 지형 파라미터 분포 유지. `Td=Pr(Tr∈[0.5,0.9])`. 지형 3종(Perlin hills/steps/stairs).

**실험·메트릭**
- 야외(산길·개울·진흙·초목·눈). 속도(vs baseline): moss 0.452 vs 0.199, mud 0.338 vs 0.197, vegetation 0.248 vs 통과불가. 기계 COT `Σ[τθ̇]⁺/(mgv)`로 효율 우위.
- 16.8 cm step 통과(평소 clearance 12.9cm 초과) — **암시적 foot-trapping 반사** 창발. 10kg 페이로드(미학습)에도 13.4cm step 통과(baseline 실패).
- DARPA SubT: 4×60분 무실패.

### (4) 문제점·한계·개선 여지
- **단일 trot gait**만 창발(다양한 보행 분포 부재).
- proprioception only → 절벽 등 위험 회피 불가(exteroception 없음).
- rigid terrain만 학습(변형지형은 일반화로 커버되나 분포엔 미포함).

### (5) 우리 적용 (강복님 자원 기준)
- actuator net 부분은 #1과 동일하게 강복님 실측으로 재현 가능. 본 논문의 추가가치는 **actuator net을 받쳐주는 상위 학습 레시피**(teacher-student + latent reconstruction + 적응 커리큘럼).
- 메모리상 우리 parkour 코드는 privileged latent(33) + DAGGER 패턴을 이미 쓰므로 본 논문의 student belief 구조와 **직접 호환**. actuator net을 sim에 넣은 뒤, 지형 일반화가 필요하면 이 TS 레시피로 확장.
- 강점 활용 포인트: actuator net 정확도(↑)가 teacher 보상의 토크/접촉 신호 신뢰도를 높여 student belief 학습 품질을 끌어올린다.

### sim 구현 난이도: **중** (actuator net 자체) / **상** (TS+커리큘럼 전체 재현)
- 필요 인프라: #1의 actuator 리그 + 삽입 인프라 동일. 추가로 teacher-student 2단계 학습 파이프라인(우리 rsl_rl에 일부 존재), particle-filter 지형 커리큘럼, PMTG. actuator net만 떼어 쓰면 난이도 중.

---

## #3. Learning Robust Autonomous Navigation and Locomotion for Wheeled-Legged Robots (Lee et al., ETH, Science Robotics 2024)
- **확정 링크**: https://arxiv.org/abs/2405.01792
- **접근 분류**: NN-Actuator (SEA=actuator net, **휠=전류 NN 매핑**으로 확장)
- **우선순위**: 3순위 (센서 없는 구동부로 actuator net 계보를 확장한 최신 ETH 사례 — 강복님 *전류* 실측과 직결)

### (1) 요약
휠-다리 로봇의 km급 도심 자율주행(취리히 8.3km, 세비야). 계층 RL: 글로벌 경로계획(Dijkstra) + HLC(navigation, 10Hz, 속도명령 출력) + LLC(locomotion, 50Hz, GRU RNN). 핵심은 **mobility-aware** — HLC가 LLC 능력을 존중. actuator 계보 관점의 핵심 기여: SEA 관절은 #1식 actuator net, **휠 모터는 토크센서가 없어 (속도명령+속도 history)→모터 전류 NN 매핑** 후 `τ=Kτ·GR·I + τ_friction`으로 토크 산출.

### (2) Motivation
도심 자율주행은 다양 지형·동적장애물을 고속으로 처리해야 한다. 전통 계획(샘플링)은 LLC 한계를 무시해 충돌·과속을 일으킨다. 또한 **휠 구동부는 토크 센싱이 없어** #1식 actuator net을 그대로 못 쓴다 → 측정 가능한 **전류**로 우회.

### (3) Method + 수식/파이프라인 + 데이터/학습/실험
**Actuator 모델 (핵심 기여)**
- 관절(SEA): #1식 actuator net.
- **휠(pseudo-direct-drive)**: 토크센서 부재 → NN이 `I_t = f(φ̇_target | φ̇_{t−1}, φ̇_{t−2}, …)` (속도명령+속도 history → 모터 전류) 학습.
  토크 환산: `τ_t = K_τ · GR · I_t + τ_friction`, `τ_friction = −C_1 φ̇ − C_2 sgn(φ̇)` (Coulomb+stick, 상수 randomization).

**LLC**: teacher(3-MLP, privileged) → student(GRU RNN, 노이즈 센서). action 16차원 = 관절위치 12 + 휠속도 4. 입력=원시 IMU+encoder(별도 state estimator 없음) + 발 주변 height sample + 속도명령(vx,vy,ωz). 보행/주행/creep gait가 핸드크래프트 없이 창발.

**HLC**: 입력 = elevation map(t,t−0.1,t−0.2) + LLC RNN belief + position buffer(최근 20위치) + waypoints. 출력 = Beta 분포로 bounded 속도(vx∈[−1,2], vy∈[−0.75,0.75], ωz∈[−1.25,1.25]). 망 = height map 2D-CNN + position 1D-CNN(PointNet식) + MLP. dense reward `r_{h,dense}`(waypoint 근접/속도투영) + exploration bonus + mobility-aware 정규화.

**학습**: 둘 다 **PPO**. 순차(LLC→HLC), LLC는 teacher→student imitation. 지형은 **Wave Function Collapse**로 절차생성(지형+navigation graph 동시).

**실험·메트릭**
- 취리히 Glattpark 8.3km, 13 goal, 30분+, 인간개입 3회. 평균속도 **1.68 m/s**(ANYmal 0.5), 기계 COT **0.16**(vs 0.34, −53%), peak 5.0 m/s.
- step: 상승 ~30cm/하강 ~50cm. HLC tracking error 0.24m(baseline 0.45). 구조환경 충돌실험: 제안 10/10 무충돌, baseline 30% 성공.

### (4) 문제점·한계·개선 여지
- 지각 병목(elevation map 3.5m/1.5m, 메모리·지연) → 하드웨어 최고속(6.3m/s) 미발휘.
- 시맨틱 이해 부족(기하 위주). position buffer 포화 시 stuck.
- **휠 actuator를 전류 proxy로 모델링** → 직접 토크 측정 대비 잔여 sim-to-real 격차 가능(저자 명시 한계).
- 맵 제작 수작업(245×345m 핸드스캔 ~90분).

### (5) 우리 적용 (강복님 자원 기준)
- **강복님은 전류 실측이 가능**하므로 본 논문의 휠 전류 NN(`I=f(φ̇_target|history)`)을 *완전 supervised*로 더 정확히 만들 수 있다. 본 논문은 센서 제약 우회였지만, 우리는 전류 실측으로 net을 직접 회귀 → 본 논문보다 낮은 잔여 격차 기대.
- 또한 `τ=Kτ·GR·I+τ_friction`에서 `Kτ`(토크상수)는 강복님 토크 실측으로 직접 캘리브 가능 → friction 상수도 데이터로 fit(randomization 불필요).
- 우리 환경에 휠 로봇이 없으면 actuator 부분만 차용 — 즉 "전류→토크" 회귀를 Go2급 PMSM에 적용하여 #1의 SEA 망보다 일반화. 하모닉 감속기 비선형이 있으면 본 매핑이 #4의 보정과 상보적.

### sim 구현 난이도: **상**
- 필요 인프라: 휠-다리 로봇 자체(우리 코드엔 부재) + elevation mapping + 계층 PPO(LLC RNN/HLC CNN) + WFC 지형생성 + km급 실험 인프라(SLAM/LiDAR/GPS). **단, actuator/전류 NN 부분만** 떼어 IsaacLab에 삽입하는 것은 **중**(전류→토크 회귀 + cfg 주입).

---

## #4. Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation (Fey, Margolis, Peticco, Agrawal — MIT, 2025)
- **확정 링크**: https://arxiv.org/abs/2502.10894 (**교정 완료** — 가제 "torque-sensorless actuator net"의 실제 제목/번호)
- **접근 분류**: NN-Actuator (torque-sensorless: **Unsupervised Actuator Net, UAN**)
- **우선순위**: 4순위 (토크센서가 *없을 때*의 대안 — 강복님은 토크 실측 가능하므로 우선순위는 낮지만, 하모닉/비선형 보정 아이디어로 가치)

### (1) 요약
**토크 센싱 없이** actuator net을 학습하는 방법 **UAN(Unsupervised Actuator Net)**. Hwangbo(#1)는 실측 토크 라벨이 필요했지만, UAN은 토크 라벨 대신 **실/시뮬 transition dynamics(상태 천이)의 불일치를 최소화하는 보정 토크 δτ**를 학습한다. 즉 "출력토크를 예측"하는 게 아니라 "시뮬레이터가 실제처럼 움직이도록 끼얹는 보정 토크"를 비지도로 찾는다. Unitree B2 4족 + 개조 Z1 Pro 팔(loco-manipulation)로 공 던지기(~20m), 덤벨 스내치(5/10 lb), 썰매 끌기를 실기 전이. + pre-training/fine-tuning(참조궤적을 힌트로) 전략.

### (2) Motivation
#1식 actuator net은 **토크 센싱 인프라**가 필요한데, 소비자급 하드웨어(Unitree 등)엔 토크센서가 없다. 특히 팔의 **하모닉 감속기**는 강한 비선형/lag을 만든다. 토크 라벨 없이도 actuator 동역학을 잡아 reward hacking(시뮬 비현실 동역학 악용)을 막고 강건 전이를 하고 싶다.

### (3) Method + 수식/파이프라인 + 데이터/학습/실험
**UAN 구조**
- 2층 MLP [128,128], ELU. 5ms마다 실행. 모든 팔 액추에이터 공유(동일 가정), 관절별 독립 처리.
- 입력: 과거 20 step(100ms)의 위치·속도 **오차** e (= q_real − q_sim 등).
- 출력: **보정 토크** δτ = π_UAN(e).

**핵심 목적함수 (라벨 불필요)**
`min_{π_UAN} || f_real(s, τ) − f_sim(s, τ + π_UAN(e)) ||`
(실제 천이 = 보정토크를 더한 시뮬 천이가 되도록). 학습 보상 `r^UAN = r^{sim-to-real} + r^{smoothness}` (실/시뮬 관절위치차 최소 + 점진적 보정).

**데이터 수집 (토크 불필요!)**
- 정책 데이터 불필요. 상태공간 커버용 3종 여기신호: square wave(관절당 진폭/주파수 12조합), sine(동일 12조합), Gaussian noise(5~400ms 샘플, ~5분). 액추에이터당 deterministic 신호 ~50초. 데이터셋 `{(s_t, τ_t, s_{t+1})}` — **토크 라벨 대신 천이(s_t→s_{t+1})만** 필요.

**RL loco-manipulation**
- Isaac Sim 4096 env, PPO. WBC pre-training(랜덤 base 속도+EE pose 추종, gait/regularization 보상) → task fine-tuning(lr 1e-5 낮게, task one-hot setup/execute/settle 임베딩, task 보상으로 참조궤적 *이탈* 허용). 토크 클리핑(속도의존), 모터파워 제약 `|τ|·|q̇| ≤ P_max`.

**실험·메트릭**
- B2(60kg) + Z1 Pro(6.8kg), 19 actuated joint. 던지기 100g 공 ~20m(시뮬 예측 초과), 덤벨 5/10 lb 들고 5초+ 유지, 썰매 113N/10m. sim-to-real gap(실측 vs 시뮬), peak leg power, task 성능, 외란복구.

### (4) 문제점·한계·개선 여지
- fine-tuning에 **task 참조궤적 필요**(모든 morphology엔 없음).
- task별 환경/보상/객체 시뮬 엔지니어링 필요.
- UAN을 **팔 액추에이터에만** 적용(전신 확장 미완).
- 구조 강성 미모델(개발 중 링크 파손 → 알루미늄 보강). 핸드 참조가 최적전략 미포착 가능.

### (5) 우리 적용 (강복님 자원 기준)
- **강복님은 토크 실측이 가능**하므로 UAN의 *비지도* 동기(=토크센서 회피)는 우리에게 핵심 동기가 아니다 → 우선 #1식 supervised가 더 정확/단순.
- 단, **가치 있는 아이디어 2개**: (a) "보정 토크 δτ로 sim 천이를 실제에 맞추는" residual 관점은 강복님 supervised net 위에 **잔차 보정층**으로 얹으면 하모닉 감속기 비선형/lag을 추가로 잡을 수 있다(supervised base + UAN-style residual). (b) **transition-matching 목적함수**는 토크 라벨이 노이즈가 큰 관절(전류→토크 환산 오차)에서 supervised보다 강건할 수 있으므로 교차검증 신호로 활용.
- pre-training(WBC)→fine-tuning(참조궤적 힌트) 레시피는 우리 AMP/imitation 트랙과 철학이 같아 loco-manipulation 확장 시 직접 참고.

### sim 구현 난이도: **상**
- 필요 인프라: **미분/전이매칭 학습 루프**(실 로봇 transition `(s_t,τ_t,s_{t+1})` 로깅 + sim rollout과의 천이오차 최소화 — IsaacLab에서 sim을 inner loop로 도는 비지도 최적화). + UAN 잔차 토크를 actuator 단계에 삽입. + loco-manipulation 환경(팔 부착 로봇, task 보상, 참조궤적). supervised(#1) 대비 학습 루프가 복잡(실/시뮬 정렬). **단, "residual 보정층" 아이디어만 차용**하면 중.

---

## 종합 비교표

| # | actuator 학습 방식 | 토크센서 | 입력 | 출력 | 우리 정합도 | sim 난이도 |
|---|---|---|---|---|---|---|
| 1 | supervised 회귀 | **필요** | 위치오차·속도 history(3) | 토크 | **최상**(토크 실측 강점 정합) | 중 |
| 2 | #1 재사용 | 필요 | 동일 6D | 토크 | 상(TS 레시피 보너스) | 중(net)/상(전체) |
| 3 | SEA=#1, 휠=전류 NN | 휠은 불필요(전류) | 속도명령+속도 history | 전류→τ | 상(**전류 실측** 정합) | 상(휠로봇)/중(net만) |
| 4 | 비지도 전이매칭(UAN) | **불필요** | 위치·속도 오차 history(20) | 보정 토크 δτ | 중(residual 아이디어) | 상 |

**권장 적용 순서(강복님 강점 기준)**: #1(supervised base, 즉시) → #3 전류 NN(휠/PMSM 일반화) → #4 residual 보정층(하모닉 비선형 추가 보정) → #2 TS+커리큘럼(지형 일반화 필요 시).

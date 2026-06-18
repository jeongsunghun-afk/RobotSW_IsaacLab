# 컴포넌트 3: 학습 전 일치 — System Identification (무게중심)

> **이 컴포넌트가 전체 마스터 플랜의 무게중심이다.** sim-to-real gap의 근원은 학습 후 보정(DR)이 아니라
> 학습을 시작하기 **전에** 시뮬레이터가 실제 로봇의 동역학을 얼마나 충실히 재현하느냐에 있다.
> DR(컴포넌트 4)은 *남은 불확실성을 흡수*하는 보험이지, 잘못 식별된 모델을 고쳐주지 않는다.
> 모델 중심이 틀려 있으면 DR 범위만 키우게 되고, 그러면 정책이 보수적·둔감해진다(over-conservative).
> **목표: a priori system ID로 모델 중심(nominal)을 실기에 정렬한 뒤, DR은 잔차(residual)에만 쓴다.**

## 0. 우리 자원이 만드는 결정적 강점

이 플랜의 실기 측정 자원은 **관절 pos/vel(encoder) + 관절 토크/전류 실측**이다. 이는 일반적인 sim-to-real
연구가 갖지 못하는 강점이다. 대부분의 quadruped 연구(예: Minitaur)는 토크 센서가 없어 전류→토크 상수를
가정하거나 모터 모델을 간접 추정해야 한다. 우리는 **토크/전류를 직접 측정**하므로:

- Actuator system ID를 **완전 supervised 회귀 문제**로 만들 수 있다 (입력=(pos error, vel) 측정, 타깃=토크 측정).
- Actuator network(Hwangbo 2019) 또는 파라메트릭 모터 모델을 **ground-truth 토크에 직접 fit** 한다.
- 검증 메트릭이 명확하다: **sim 토크 vs real 토크 RMSE**(Nm). 이건 추정이 아니라 측정값 대 측정값 비교다.

이 강점은 **Stage 1(고정 베이스 leg 리그)** 에서 최대로 활용된다. 베이스가 고정되어 있으면 base dynamics가
관성·중력·접촉으로 오염되지 않고, **순수 관절 동역학만** excitation으로 깨끗하게 식별할 수 있다. 식별된
actuator/관성/마찰 파라미터를 **Stage 2(full robot 자유보행)** 의 asset에 그대로 이식한다.

### 2-stage 매핑 (이 컴포넌트 전체에 적용)

| Stage | 환경 | 식별 대상 | 자원 | 우리 인프라 |
|-------|------|-----------|------|-------------|
| **Stage 1** | 고정 베이스 leg 리그 (R_Skeleton Hind Leg / MotionJig) | actuator net, motor friction, torque-speed envelope, PD gain, 링크 관성(부분), joint friction, latency | encoder pos/vel + **토크/전류** | `scripts/real2sim/` ZMQ 브리지로 excitation 주입·기록 |
| **Stage 2** | full robot 자유보행 | base mass/CoM/inertia, foot-ground friction, 전신 latency, 잔차 DR | + base IMU + estimator module | Stage 1 식별값 이식 + EventCfg DR |

---

## 1. Actuator / 모터 모델링 (가장 중요)

**왜 가장 중요한가.** sim-to-real gap의 가장 큰 단일 원인은 actuator 동역학의 모델링 오류다. URDF/USD의 기본
actuator는 *이상적 PD*(torque = Kp·(q_des − q) − Kd·q̇)로 가정한다. 실제 직렬 탄성/유성기어/BLDC 모터는
비선형 compliance, gear friction, torque-speed saturation, 전류 제어 루프 지연을 갖는다. Tan et al. 2018은
Minitaur에서 **선형 torque-current 가정으로 학습한 정책은 실기 전이가 나빴고, 비선형 actuator 모델을
도입하자 전이가 크게 개선됨**을 보였다 — 즉 actuator 충실도가 전이 성공의 1순위 결정 변수다.

우리는 토크 실측이 가능하므로 두 갈래 중 하나(또는 둘 다)를 쓸 수 있다: **(A) Actuator network**(비파라메트릭),
**(B) 파라메트릭 모터 모델**. 둘 다 Stage 1 고정 리그에서 식별한다.

### 1-A. Actuator Network (Hwangbo et al. 2019)

- **측정:** Stage 1 고정 리그에서 각 관절에 excitation trajectory(§5)를 주고, 시간 동기화된 시퀀스를 기록한다 —
  입력 = (관절 위치 오차 history `q_des − q`, 관절 속도 history `q̇`), 타깃 = **실측 토크 τ**. Hwangbo는 단일
  시점이 아니라 **위치 오차·속도의 시간 history(예: 현재 + 과거 2~3 시점)** 를 입력으로 써 직렬 탄성 액추에이터의
  속도-의존 compliance와 damping을 포착했다. 우리는 토크를 직접 재므로 타깃이 깨끗하다(Hwangbo도 토크 실측 사용).
  데이터는 actuator의 동작 영역 전체(저속/고속, 소/대 오차)를 커버하도록 광대역 excitation으로 수집한다.
- **근거:** Hwangbo et al., *"Learning agile and dynamic motor skills for legged robots"*, Science Robotics 2019
  (arXiv:1901.08652). 4단계 파이프라인 — (1) stochastic rigid-body 모델링, (2) **실데이터로 actuator net 학습**,
  (3) sim에서 RL, (4) 실기 배포. Actuator net이 "(action) → (torque)" 매핑을 end-to-end로 학습해 해석적 모델이
  접근 못 하는 **비선형 compliance·damping**을 포착. 이 actuator net 덕분에 ANYmal이 flying-trot 1.2 m/s(이전
  대비 +50%)와 fall recovery를 달성. 후속작 Lee et al. 2020(*"Learning quadrupedal locomotion over challenging
  terrain"*, Science Robotics, arXiv:2010.11251)도 **동일 actuator net을 그대로 재사용**해 teacher-student 구조로
  눈·진흙·잔해 등 OOD 지형까지 robust 전이 — actuator net이 재사용 가능한 핵심 자산임을 보임.
- **sim 반영:** IsaacLab의 기본 actuator(`ImplicitActuatorCfg` / `DCMotorCfg`, `source/isaaclab_assets/.../rga.py`의
  `stiffness`/`damping`/`effort_limit`)는 PD+토크클립을 PhysX 솔버 내부에서 계산한다. Actuator net을 쓰려면
  **커스텀 `ActuatorBase` 서브클래스**를 만들어(`isaaclab.actuators` 패턴) `compute()`에서 PD 식 대신 학습된 MLP로
  토크를 산출하고 그 토크를 관절에 적용한다(`set_joint_effort_target`류). 즉 PhysX는 *effort 모드*로 두고 토크는
  파이썬 측 actuator net이 공급. 이는 IsaacLab의 actuator 추상화(`ImplicitActuator`=솔버 PD, `DCMotor`=토크-속도
  saturation 모델)와 같은 계층에 들어가는 자연스러운 확장이다. 새 로봇마다 actuator net 가중치만 교체하면 범용.
- **검증:** **held-out excitation set에서 sim 토크 vs real 토크 RMSE / NRMSE(Nm, % of effort_limit).** Hwangbo급
  목표는 관절 토크 예측 오차를 모터 잡음 수준까지 낮추는 것. 추가로 Stage 1 고정 리그에서 동일 PD setpoint
  궤적(스텝/사인)을 sim·real 양쪽에 주고 **관절 위치 추종 오차(tracking RMSE, rad)** 와 **위상 지연(ms)** 비교.

### 1-B. 파라메트릭 모터 모델 (actuator net 대안/보완)

비파라메트릭 net이 과한 경우(또는 해석성이 필요할 때), 토크 실측으로 다음 파라미터를 직접 fit 한다. 각각이
IsaacLab actuator cfg 필드에 1:1 대응한다.

1. **모터 마찰 (Coulomb + viscous).**
   - **측정:** 무부하/저관성 조건에서 관절을 정속 회전(여러 속도)시키며 `τ_friction(q̇)` 측정. 모델
     `τ_f = τ_c·sign(q̇) + b·q̇` (Coulomb τ_c + viscous b). 저속에서 Stribeck 효과까지 보려면 마이크로 사인 가진.
   - **근거:** Tan et al. 2018(arXiv:1804.10332)이 Minitaur URDF 제작 시 **모터 마찰 측정 실험을 별도 설계**.
     로봇 동역학 식별의 표준(Recursive Newton-Euler + 회귀; "Optimal robot excitation and identification", Swevers).
   - **sim 반영:** `DCMotorCfg`/`ImplicitActuatorCfg`의 `friction` 필드(현재 Go2 cfg는 `friction=0.0`, rga.py:316).
     viscous 항은 `damping`에 합산되거나 actuator 모델에 직접. IsaacLab actuator는 Coulomb friction을 joint
     friction으로 받을 수 있음(§3과 통합).
   - **검증:** 정속 회전 시 sim τ vs real τ 곡선의 기울기(viscous)·절편(Coulomb) 일치. RMSE(Nm).

2. **Torque-speed envelope (saturation).**
   - **측정:** 각 관절을 다양한 속도에서 최대 토크 요구로 구동, 속도별 가용 토크 상한 곡선을 그림. BLDC는
     고속에서 back-EMF로 토크가 떨어지는 사다리꼴/삼각 envelope.
   - **근거:** Tan 2018이 비선형 torque-current 관계의 중요성을 강조. IsaacLab `DCMotorCfg`가 바로 이 모델
     (`saturation_effort`, `velocity_limit`, `effort_limit`로 속도-토크 한계를 표현).
   - **sim 반영:** `DCMotorCfg(effort_limit, saturation_effort, velocity_limit)` — Go2 cfg는 effort 23.5 /
     saturation 23.5 / velocity 30(rga.py:309-317). 실측 envelope으로 이 3값을 교정. R_Skeleton류는
     `ImplicitActuatorCfg`의 `effort_limit_sim`/`velocity_limit_sim`을 관절별 dict로 채움(이미 rga.py에서 관절별
     세분화되어 있음 — 실측치로 갱신만 하면 됨).
   - **검증:** 고속 구간 sim 토크 saturation 발생 시점과 real이 일치하는지. 포화 토크 RMSE.

3. **전류→토크 상수 (Kt) + gear ratio.**
   - **측정:** 전류 i와 실측 토크 τ를 동시 기록해 `τ = Kt·i·N`(N=gear ratio) 회귀. 우리는 전류·토크 둘 다 재므로
     Kt를 직접 식별 가능(대부분 연구는 못 함).
   - **근거:** Tan 2018 — 선형 가정 실패, 비선형(부하·온도 의존) Kt가 전이 개선. Hwangbo의 actuator net이
     사실상 이 비선형성까지 흡수.
   - **sim 반영:** 토크 단위로 작업하는 IsaacLab에선 Kt가 effort_limit 스케일/단위 일관성 점검용. actuator
     net을 쓰면 자동 반영. 파라메트릭이면 effort_limit를 실토크 단위로 정렬.
   - **검증:** i→τ 예측 RMSE. 단위 일관성(Nm) 교차 확인.

4. **PD gain (Kp/Kd) 실측.**
   - **측정:** 실기 컨트롤러가 사용하는 실제 PD gain을 firmware에서 확인하거나, 스텝 응답(상승시간·오버슈트)에서
     역산. 고정 리그에서 setpoint 스텝을 주고 응답의 ω_n, ζ를 fit → Kp = I_eff·ω_n², Kd = 2ζ·ω_n·I_eff.
   - **근거:** sim PD gain은 **실기 PD와 반드시 동일**해야 함(정책이 PD 위에서 작동). DR로 ±랜덤화는 하되
     중심은 실측. (HIND_LEG_CFG가 이미 측정 I_eff 기반 inertia-scaled gain을 적용한 선례 — rga.py:477-493,
     `_workspace/hind_leg_kp_kd_tuning_guide.md`. hip 0.163 / thigh 0.133 / calf 0.030 / foot 0.0019 kg·m²로
     ω_n=20, ζ=0.9 설계.)
   - **sim 반영:** actuator cfg `stiffness`(Kp) / `damping`(Kd). 관절별 dict로 정렬.
   - **검증:** 스텝 응답 sim vs real의 상승시간/오버슈트/정착시간 일치. 위치 추종 RMSE(rad).

5. **Gear backlash (선택, 정밀도 요구 시).**
   - **측정:** 방향 반전 시 위치-토크 hysteresis 폭(dead-zone, rad) 측정.
   - **근거:** 직렬 탄성/유성기어의 backlash는 정밀 추종·접촉 안정성에 영향. 다수 연구는 무시 가능 수준이면
     생략하고 DR로 흡수.
   - **sim 반영:** IsaacLab 기본 actuator엔 backlash 항이 없음 → actuator net이 흡수하거나, 무시 후 컴포넌트 4의
     DR(action noise/dead-zone)로 잔차 처리.
   - **검증:** hysteresis 폭이 모터 잡음·encoder 분해능 이하면 무시 정당화.

> **권고:** 우리 자원(토크 실측)에서는 **1-A actuator net을 1순위**로 두되, 해석성·디버깅을 위해 1-B
> 파라메트릭(특히 friction·torque-speed envelope·PD)을 **병행 식별**해 actuator net의 sanity check 및
> cfg 기본값 정렬에 쓴다. 둘의 RMSE를 비교하면 net이 잡아낸 비선형성의 크기를 정량화할 수 있다.

---

## 2. 링크 관성 / 질량 파라미터

**왜 중요한가.** 관성/질량 오차는 토크 요구·접촉력·동역학 응답을 직접 왜곡한다. base mass·CoM 오차는 균형
제어에 치명적. CAD 명목값은 케이블·배터리·실제 가공 오차로 실기와 수 % 벗어난다.

- **측정:**
  1. **CAD 명목값**을 1차 baseline으로 USD에 반영(질량, CoM, inertia tensor).
  2. **저울**로 각 링크/전체 실측 질량 → CAD 대비 보정 계수.
  3. **CoM**: 링크를 두 점에서 매달아 평형(균형) 측정, 또는 알려진 기준점 대비 무게중심 위치.
  4. **링크 관성**: **bifilar(이중사) pendulum**으로 진동 주기 → 관성 모멘트 산출 (`I = (m·g·d²·T²)/(4π²·L)`).
     소형 링크는 CAD inertia를 질량 보정 계수로 스케일하는 근사로 대체 가능.
  5. **Stage 1 토크 식별과의 결합**: 고정 리그 excitation에서 측정된 토크-가속 관계로 **유효 관성 I_eff를
     역식별**(Recursive Newton-Euler 회귀). 우리는 토크 실측이 있으므로 base-excited regression이 가능.
- **근거:** Tan et al. 2018 — **로봇을 분해해 치수·질량·각 링크 CoM을 측정**해 정확한 URDF 제작(전이의 전제조건).
  Swevers et al. *"Optimal robot excitation and identification"* — excitation trajectory + ML 추정으로 동적
  파라미터(관성 포함) 식별. HIND_LEG_CFG의 inertia-scaled gain은 우리 코드에서 I_eff 실측이 실제로 게인 설계를
  바꾼 선례.
- **sim 반영:** 로봇 asset USD(`ArticulationCfg.spawn=UsdFileCfg`, rga.py)의 rigid body mass / CoM / inertia
  properties. USD 작성 단계에서 CAD→실측 보정값 반영. 추가로 base mass·CoM은 **컴포넌트 4 DR의 randomization
  중심(nominal)** 으로도 사용(Go2/Hind Leg EventCfg가 base mass·CoM term 보유).
- **검증:** (a) Stage 1: 동일 excitation에 대한 sim vs real **토크 RMSE**(관성 오차는 가속 구간 토크에 나타남).
  (b) Stage 2: free-fall/pendulum drop 같은 수동 동역학에서 sim vs real **궤적/주기 오차**. (c) 전체 질량·CoM의
  CAD 대비 보정 후 잔차 %.

---

## 3. 마찰 (joint friction + foot-ground contact friction)

두 종류를 분리한다: **관절 마찰**(Stage 1에서 식별, §1-B와 통합)과 **발-지면 접촉 마찰**(Stage 2에서 식별).

### 3-A. 관절 마찰 (joint friction/damping)
- **측정:** §1-B-1과 동일 — 무부하 정속 회전으로 Coulomb τ_c + viscous b 식별. 토크 실측이 있어 직접 회귀.
- **근거:** Tan 2018의 모터 마찰 측정. 표준 로봇 식별(Coulomb+viscous 모델).
- **sim 반영:** actuator/joint의 `friction`(Coulomb) + `damping`(viscous) 필드. IsaacLab joint friction 또는
  actuator friction에 반영.
- **검증:** 정속 토크 곡선 sim vs real RMSE.

### 3-B. 발-지면 마찰계수 (foot-ground friction)
- **측정:** **force gauge로 정적 마찰 측정** — 알려진 normal force(발에 질량 부하) 아래 미끄러지기 시작하는
  수평 견인력 측정, `μ = F_friction / F_normal`. 대상 지면(실험실 바닥·고무매트 등)별로 측정. 동마찰은
  미끄러짐 중 견인력으로. 일반 실내 지면 μ≈0.4~0.8 범위.
- **근거:** 발-지면 μ는 접촉 안정성·미끄러짐의 1차 결정 변수. legged 로봇 마찰 식별은 정적 force-gauge
  측정이 표준(off-line); 온라인 추정 연구도 존재(예: arXiv:2502.16843 "Online Friction Coefficient
  Identification … Smoothed Contact Gradients"). DR에서 friction을 넓게 랜덤화하는 것은 정적 측정의 불확실성을
  흡수하기 위함.
- **sim 반영:** 지면/발 material의 PhysX friction(`static_friction`, `dynamic_friction`). IsaacLab DR EventCfg의
  `randomize_rigid_body_material`(friction) term의 **중심값**을 실측 μ로 설정하고, 측정 불확실성·지형 다양성을
  범위로(컴포넌트 4). 우리 Go2/Hind Leg EventCfg에 이미 friction(material) DR term 존재.
- **검증:** Stage 2에서 경사면 정적 슬립 임계각(static slip angle) sim vs real 비교, 또는 push 후 발 슬립
  거리 비교. friction이 맞으면 미끄러짐 onset 일치.

---

## 4. 지연 (Latency / Delay)

**왜 중요한가.** 센서→연산→통신→액추에이터의 지연은 시스템을 **non-Markovian**으로 만든다. sim이 지연 0이면
정책이 비현실적으로 즉응 가능한 dynamics에 의존하게 되어 실기에서 진동·불안정(특히 고게인·고속 동작).

- **측정:** Stage 1 고정 리그에서 **command timestamp → 관절 응답(첫 위치 변화) timestamp** 차이로 actuation
  latency(ms) 측정. 센서 지연은 알려진 외란(탭/임펄스)을 주고 encoder 보고 시점과 비교. 통신 지연은 우리
  real2sim 브리지에 이미 있는 **`utils/benchmark.py`(50Hz 주파수/레이턴시/지터 측정)** 로 정량화 — 이 인프라를
  지연 측정에 직접 재사용한다.
- **근거:** Tan et al. 2018이 latency를 핵심 DR 파라미터로 포함. 다수 quadruped sim-to-real 연구가
  observation/action delay를 **obs history를 지연시켜 과거 관측을 주입**하는 방식으로 모델링(지연으로 인한
  non-Markovian 성질 명시). Hwangbo의 actuator net history 입력도 부분적으로 지연을 흡수.
- **sim 반영:** (a) **action delay** — 정책 출력을 N 스텝 버퍼링 후 적용(IsaacLab에서 action buffer/지연 래퍼).
  (b) **observation delay** — obs를 지연시켜 과거 시점 주입. (c) 측정 latency를 중심으로 컴포넌트 4 DR에서
  ±지터 랜덤화. 새 buffer는 `_reset_idx`에서 초기화(프로젝트 규약).
- **검증:** 동일 setpoint 스텝에 대한 sim vs real **위상 지연(ms)** 일치. closed-loop에서 고게인 진동 발생
  임계 게인이 sim·real 간 일치하는지(지연이 안정성 한계를 결정).

---

## 5. Excitation Trajectory 설계 (system ID의 입력 신호)

**왜 중요한가.** §1~4의 식별 품질은 **입력 신호가 동역학을 얼마나 잘 자극(persistent excitation)** 하느냐에 좌우
된다. 한 동작 영역만 자극하면 그 영역에서만 맞는 모델이 나온다. Stage 1 고정 리그에서 실행하며, 우리 real2sim
ZMQ 브리지(`controller.py` → `sim_runner.py`)로 setpoint를 주입하고 토크/pos/vel을 50Hz로 기록한다.

- **측정(=가진 신호 설계):**
  - **Chirp / sine sweep**: 저→고 주파수로 천천히 sweep해 actuator/관절의 주파수 응답 전대역을 자극.
    대역폭(bandwidth)·resonance·위상 지연 식별에 최적. (단점: 각 주파수가 짧게 존재해 잡음 민감.)
  - **Multi-sine (sum of sinusoids)**: 선택한 대역의 여러 주파수를 동시 가진 → 주파수영역 식별에 적합,
    측정시간 효율적.
  - **PRBS (pseudo-random binary sequence)**: ±진폭 스위칭, 광대역 자극, 파라미터화 쉬움 → 토크-속도 envelope·
    비선형 마찰 영역까지 폭넓게 커버.
  - **Finite Fourier series 궤적**(Swevers): **주기적**이라 시간영역 평균으로 잡음 억제 + 측정잡음 특성 추정
    가능 → 관성/마찰 동적 파라미터 회귀의 표준 가진. 계수를 최적화해 regressor의 condition number를 낮춤
    (optimal excitation).
  - actuator net(§1-A)용은 **(pos error, vel) 공간을 고르게 덮도록** 소/대 진폭 × 저/고속 조합으로 설계.
- **근거:** Swevers et al. *"Optimal robot excitation and identification"* — optimal periodic excitation +
  maximum-likelihood 추정. Brunton/표준 식별 문헌의 chirp/PRBS/multi-sine 신호 특성. Hwangbo 2019가 actuator
  net 학습 데이터를 광대역 가진으로 수집(동작 영역 전체 커버). 가진 궤적 최적화는 활발한 연구 주제
  (arXiv:2401.16566 virtual-constraint excitation, arXiv:2003.01190 adversarial informative trajectories).
- **sim 반영:** 가진 신호 자체는 sim 파라미터가 아니라 **측정 프로토콜**이다. 단, 식별 검증을 위해 **sim에서도
  동일 가진 신호를 재생**해 sim·real 토크/궤적을 정렬 비교한다(아래 검증). 우리 `sim_runner.py`의 slew-rate
  제한·50Hz 루프가 sim측 재생 인프라를 제공.
- **검증:** 가진 신호의 **persistent excitation 충분성** = 식별 regressor의 condition number / 파라미터 추정
  공분산이 작은지. held-out 가진 신호에서 모델 일반화(§1~4의 RMSE)가 유지되면 가진이 충분했던 것.

---

## 6. 식별 → 검증 → 반복 루프 (컴포넌트 3 통합 워크플로우)

```
[Stage 1: 고정 베이스 leg 리그]
  1. CAD 명목 USD + 기본 actuator cfg 작성 (baseline)
  2. Excitation trajectory(§5) 설계 → real2sim 브리지로 실기 주입, (pos,vel,torque,current) 50Hz 기록
  3. 식별:
       actuator net (§1-A)  +  파라메트릭 모터모델 (§1-B: friction/envelope/Kt/PD)
       joint friction (§3-A),  유효 관성 I_eff (§2),  latency (§4)
  4. sim에 반영: actuator cfg / USD mass·CoM·inertia / friction / delay 래퍼
  5. 검증: 동일 가진을 sim에서 재생 → sim torque vs real torque RMSE,
            step 응답 tracking RMSE, 위상 지연(ms)
  6. RMSE가 목표 이하가 될 때까지 3~5 반복 (필요시 actuator net 재학습/파라미터 재fit)
        ↓ (식별값 이식)
[Stage 2: full robot 자유보행]
  7. Stage 1 actuator/joint 파라미터를 full-robot asset에 이식
  8. base mass/CoM/inertia 실측 반영 (§2), foot-ground μ 실측 반영 (§3-B), 전신 latency (§4)
  9. 검증: 수동 동역학(free-fall/drop/pendulum) sim vs real 궤적, 정적 슬립 임계각
 10. 남은 잔차 → 컴포넌트 4(DR + real2sim 루프)로 인계 (중심=식별값, 범위=잔차 불확실성)
```

### 검증 메트릭 요약 (acceptance 기준)

| 식별 대상 | 1차 메트릭 | 보조 메트릭 |
|-----------|-----------|-------------|
| Actuator (net/파라메트릭) | **sim torque vs real torque RMSE/NRMSE (Nm, %)** | step 응답 tracking RMSE(rad), 위상지연(ms) |
| 링크 관성/질량 | 가속구간 torque RMSE | free-fall/drop 궤적 오차, 질량·CoM CAD 대비 잔차% |
| joint friction | 정속 torque 곡선 RMSE | Coulomb 절편·viscous 기울기 일치 |
| foot-ground μ | 정적 슬립 임계각 오차(deg) | push 후 슬립 거리 |
| latency | 위상지연 ms 일치 | closed-loop 진동 임계게인 일치 |
| excitation 충분성 | regressor condition number / 추정 공분산 | held-out 가진 일반화 RMSE 유지 |

---

## 7. 우리 코드/인프라 재사용·일반화 정리

| 우리 자산 | 컴포넌트 3에서의 역할 |
|-----------|----------------------|
| `scripts/real2sim/` ZMQ 브리지(`controller.py`/`sim_runner.py`) | Stage 1 excitation 주입 + (pos,vel,torque) 기록 + sim 재생 |
| `scripts/real2sim/utils/benchmark.py` (50Hz 주파수/레이턴시/지터) | latency(§4) 측정 인프라 그대로 재사용 |
| `source/isaaclab_assets/.../rga.py` actuator cfg (KP/KD/effort/vel limit, `ImplicitActuator` vs `DCMotor`) | §1-B 파라메트릭 식별값 주입 지점 (관절별 dict 이미 존재) |
| `DCMotorCfg(saturation_effort, velocity_limit, effort_limit)` | §1-B-2 torque-speed envelope 직접 표현 |
| HIND_LEG_CFG inertia-scaled gain (rga.py:477-493) | §1-B-4/§2 I_eff 실측→게인 설계 **선례** |
| ArticulationCfg `UsdFileCfg` (mass/CoM/inertia) | §2 링크 관성 반영 지점 |
| EventCfg DR (friction/mass/CoM/actuator gain) | §2·§3-B·§4 식별값을 **DR 중심**으로 사용 → 컴포넌트 4로 인계 |
| 커스텀 `ActuatorBase` 서브클래스 (신규, 로드맵) | §1-A actuator net 주입 — 새 로봇마다 가중치 교체로 범용화 |

**범용성(legged 2족/4족 공통):** 위 절차는 특정 로봇에 독립적이다 — 관절 수·구조와 무관하게 (1) 고정 리그에서
관절별 actuator/friction/관성 식별, (2) full-robot에서 base·접촉 식별, (3) sim torque RMSE로 검증하는 동일 루프를
따른다. 새 로봇은 *USD + actuator cfg + actuator net 가중치*만 교체하면 같은 플랜으로 진행된다.

---

## 출처 (EvidenceBase)

- Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science Robotics 2019 — [arXiv:1901.08652](https://arxiv.org/abs/1901.08652) / [Science Robotics](https://www.science.org/doi/10.1126/scirobotics.aau5872) — **actuator network**.
- Lee et al., *Learning quadrupedal locomotion over challenging terrain*, Science Robotics 2020 — [arXiv:2010.11251](https://arxiv.org/abs/2010.11251) — actuator net 재사용 + teacher-student robust 전이.
- Tan et al., *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots*, RSS 2018 — [arXiv:1804.10332](https://arxiv.org/pdf/1804.10332) — URDF 실측(질량/CoM/치수), 모터 마찰 측정, 비선형 actuator 모델 + dynamics randomization(25 파라미터).
- Peng et al., *Sim-to-Real Transfer of Robotic Control with Dynamics Randomization*, ICRA 2018 — [paper](https://xbpeng.github.io/projects/SimToReal/SimToReal_2018.pdf) — DR 보완(컴포넌트 4 연계).
- Swevers et al., *Optimal robot excitation and identification* — [ResearchGate](https://www.researchgate.net/publication/3298758_Optimal_robot_excitation_and_identification) — **excitation trajectory 설계** + ML 동적 파라미터 식별.
- *Excitation Signals for Identification of Dynamic Systems* (chirp/PRBS/multi-sine 특성) — [SciEngineer](https://sciengineer.com/excitation-signals-for-identification-of-dynamic-systems/).
- *Excitation Trajectory Optimization … Virtual Constraints* — [arXiv:2401.16566](https://arxiv.org/pdf/2401.16566); *Adversarial Generation of Informative Trajectories for Dynamics System Identification* — [arXiv:2003.01190](https://arxiv.org/pdf/2003.01190).
- *Online Friction Coefficient Identification for Legged Robots … Smoothed Contact Gradients* — [arXiv:2502.16843](https://arxiv.org/pdf/2502.16843) — foot-ground μ 온라인 식별 보완.
- *Autotuning Bipedal Locomotion MPC with GRFM-Net (DiffTune)* — [arXiv:2409.15710](https://arxiv.org/pdf/2409.15710); *Auto-Tuned Sim-to-Real Transfer* — [arXiv:2104.07662](https://arxiv.org/pdf/2104.07662) — differentiable/auto-tuning 식별(고급, 잔차 자동보정 옵션).

> **내부 참조:** `_workspace/hind_leg_kp_kd_tuning_guide.md`(I_eff 실측→게인), `scripts/real2sim/INTEGRATION_GUIDE.md`(ZMQ 브리지), `_workspace/parkour_dr_research.md`(DR 카탈로그, 컴포넌트 4 연계).

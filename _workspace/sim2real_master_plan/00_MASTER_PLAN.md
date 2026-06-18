# 새 legged 로봇 Sim-to-Real 마스터 플랜 (통합 표지)

> **이 문서의 역할 = 지도 + 표지.** 4개 컴포넌트(01~04)의 상세를 복붙하지 않고 요약·링크한다. 디테일은 각 파일이 보유한다.
> 상세: [01 배포 파이프라인](01_deployment_pipeline.md) · [02 실기 검증 테스트](02_real_robot_tests.md) · [**03 System Identification (무게중심)**](03_system_identification.md) · [04 DR + real2sim 루프](04_dr_and_real2sim_loop.md)
> 스펙(맥락/제약): [`.omc/specs/deep-interview-new-robot-sim-to-real-master-plan.md`](../../.omc/specs/deep-interview-new-robot-sim-to-real-master-plan.md)

## Executive Summary

새 legged 로봇(2족/4족 범용)을 IsaacLab에서 학습시켜 실기에 배포하기까지의 **재사용 가능한 마스터 플랜**이다. 어떤 다리 달린 로봇이 생겨도 *USD + actuator cfg + actuator net 가중치*만 교체하면 같은 절차를 따른다. **무게중심은 학습 전 system ID** — sim-to-real gap의 근원은 학습 후 보정(DR)이 아니라, 학습을 시작하기 전에 시뮬레이터가 실기 동역학을 얼마나 충실히 재현하느냐다. DR은 잘못된 모델을 고쳐주지 않고 잔차(residual)만 흡수한다. 전체는 **2-stage scope**로 위험을 단계화한다: **Stage 1**(고정 베이스 leg 리그 — actuator/관절 추종 충실도 확립 + system ID 데이터 수집), **Stage 2**(full robot 자유보행 — base IMU + estimator module로 보행 정책 학습→배포). 우리 실기 자원(관절 pos/vel encoder + **토크/전류 실측** + 자유보행 IMU)은 actuator system ID를 *완전 supervised 회귀*로 만드는 결정적 강점이다. 산출물 깊이 기준 = 각 기법마다 **근거(논문/출처) + sim-to-real 효과 검증 방법(메트릭/실험)**.

---

## 1. 핵심 원칙 4가지

| # | 원칙 | 의미 |
|---|------|------|
| **P1** | **legged 범용** | 2족·4족 어떤 다리 로봇이든 따라갈 수 있는 플랜. 특정 로봇 하드코딩 금지. 관절 수·구조와 무관하게 같은 루프(고정 리그 식별 → full robot 식별 → torque RMSE 검증). |
| **P2** | **2-stage scope** | Stage 1(고정 리그: actuator system ID + 관절 추종 검증) → Stage 2(full robot: IMU + estimator 자유보행 학습→배포). Stage 1 게이트를 못 넘으면 Stage 2 진입 금지. |
| **P3** | **estimator (parkour RMA 패턴)** | 자유보행에서 직접 측정 난해한 base lin_vel은 estimator module이 추정. **설계 원칙: policy=추정값 입력, critic/discriminator=실제값(privileged).** 배포 시 actor가 추정값으로 자립. |
| **P4** | **measured-centered** | system ID(C3)가 측정한 값을 분포의 **중심(nominal)**으로 두고, DR(C4)은 측정 불확실성만큼 ±폭만 준다. system ID 없는 넓은 DR = over-conservative 정책, 좁은 DR = OOD 전이 실패. **DR은 system ID를 대체하지 않고 보완한다.** |

---

## 2. 전체 흐름도 (2-Stage 타임라인)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — 고정 베이스 leg 리그 (base 미동, 토크/전류 직접 측정)               │
│   목적: actuator 모델·관절 추종 충실도 확립 + system ID 데이터 수집            │
│                                                                               │
│   [C1] asset 준비(URDF→USD, ArticulationCfg, actuator 모델 선택)              │
│        → [C2] T1 통신벤치 → T2 step/sine → T3 정지토크/마찰 → T5 excitation   │
│             │  (검증 게이트 = 동시에 C3 데이터 수집 패스)                     │
│             └─→ [C3] actuator net / 파라메트릭 모터모델 / 관성 / 마찰 / latency│
│                      식별 → sim 반영 → sim torque vs real torque RMSE 검증     │
│        ↘ [C4] Stage1 DR: latency·actuator gain·obs noise (base 항 무의미)     │
│                                                                               │
│   ══ GATE: 관절 추종 RMSE + 토크/전류 매칭 임계 이하 (못 넘으면 Stage 2 금지) ══│
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓ (식별값 이식)
┌─────────────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — full robot 자유보행 (floating base, IMU + estimator)               │
│   목적: 보행 정책 RL 학습 → 실기 배포                                          │
│                                                                               │
│   [C3] base mass/CoM/inertia 실측, foot-ground μ 실측, 전신 latency           │
│   [C1] floating-base RL env → 학습(PPO/AMP + estimator: actor=추정,critic=실제)│
│        → export(estimator 번들 + normalizer 내장, ⚠action_scale 수동) → 배포   │
│   [C4] 6항 DR 전부 활성(mass/CoM/마찰/push/gain/latency/noise)                 │
│        + RMA history encoder online 적응 + AMP 동작정규화                      │
│   [C2] T4 무부하 정책(공중) → T6a IMU캘리 → T6b 정적균형 → T6c 점진보행(자유)  │
│        │ fail → C4(DR강화·real2sim) / C3(재식별) 회귀                          │
│   [C4] real2sim 비교 루프: 실기 rollout vs sim replay → 파라미터 역추적 → 보정 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**컴포넌트가 타임라인에 들어가는 위치 (요약)**

| 컴포넌트 | Stage 1 (고정 리그) | Stage 2 (자유보행) |
|----------|--------------------|--------------------|
| C1 배포 파이프라인 | asset cfg 확정, setpoint 재생 env | RL env→학습→export→배포 |
| C2 실기 검증 테스트 | T1·T2·T3·T5 (검증=C3 데이터 수집) | T4 무부하 → T6a/b/c 자유보행 |
| **C3 system ID (무게중심)** | **actuator/관성/마찰/latency 식별 (핵심)** | base mass/CoM·foot μ·전신 latency |
| C4 DR + real2sim | latency·gain·noise DR + 고정리그 replay | 6항 DR + RMA + AMP + real2sim 루프 |

---

## 3. 4개 컴포넌트 요약 + 링크

| # | 컴포넌트 | 깊이 | 1-2줄 요약 | 상세 |
|---|----------|------|-----------|------|
| C1 | **배포 파이프라인** | 표준 절차 | 새 로봇을 IsaacLab에 등록→학습→export→실기 배포까지 end-to-end **골격**. system ID/DR 디테일은 C3/C4에 위임하고, 그 단계를 **언제·어디서 호출하는지**만 정의. asset(URDF→USD, ArticulationCfg)·env(Stage1 재생 / Stage2 floating-base)·학습(+estimator)·export(JIT/ONNX, action_scale 함정)·배포(ZMQ 50Hz). | [01](01_deployment_pipeline.md) |
| C2 | **실기 검증 테스트** | 표준 절차 | **배포 가능성을 사다리식으로 올리는 게이트 시퀀스.** 안전(통신)→고정 actuator→무부하 정책→자유보행 순, 각 단계 pass 없이 다음 진입 금지. T1~T6. T2·T3·T5는 검증인 동시에 C3 데이터 수집 패스. | [02](02_real_robot_tests.md) |
| **C3** | **System Identification (무게중심)** | **가장 깊게** | **gap의 근원.** 토크/전류 실측으로 actuator를 *완전 supervised 회귀*로 식별 — actuator net(Hwangbo 2019) + 파라메트릭 모터모델(friction/torque-speed envelope/Kt/PD) + 링크 관성 + foot-ground μ + latency + excitation trajectory 설계. 측정값을 sim nominal로 정렬, DR은 잔차에만. | [**03**](03_system_identification.md) |
| C4 | **DR + real2sim 루프** | 표준 절차 | **measured-centered DR**(C3 측정값 중심, ±측정σ 폭) + adaptation(RMA history encoder·estimator·AMP) + **real2sim 비교 루프**(실기 rollout vs sim replay → divergence로 파라미터 역추적 → 재보정). | [04](04_dr_and_real2sim_loop.md) |

---

## 4. 컴포넌트 간 의존 / 실행 순서

```
C3 system ID (측정)  ──nominal 값──→  C1 asset cfg / USD
       │                                   │
       │ 측정 불확실성(σ)                  │ env·학습
       ↓                                   ↓
C4 DR 범위 설정 (measured-centered) ──→ 학습 (RL + estimator)
       │                                   │ export
       │                                   ↓
       │                              C2 실기 게이트 (T1→T6, 사다리)
       │                                   │ 통과 시 배포 / 실패 시 회귀
       └───────── real2sim 루프 ←──────────┘
              (실기 rollout으로 C3 asset·C4 DR 재보정, 닫힌 루프)
```

1. **C3 system ID 먼저** — 측정값이 모든 것의 출발점(nominal). 측정 없이는 C4 DR 중심을 정할 수 없다.
2. **C3 → C1 asset** — 측정한 actuator/관성/마찰을 `rga.py` ArticulationCfg·USD에 nominal로 반영.
3. **C3 σ → C4 DR 범위** — 측정 불확실성만큼만 ±폭(measured-centered). 마찰만 예외적으로 넓게.
4. **C1 학습** — RL(PPO/AMP) + estimator. C4의 RMA/AMP/DR이 학습 분포를 형성.
5. **C1 export → C2 실기 테스트 게이트** — 사다리(T1→T6) 각 단계 pass 게이트. 실패 시 C3/C4로 회귀.
6. **C4 real2sim 루프** — 배포 후 실기-sim divergence 측정 → C3 asset / C4 DR 범위 재보정 → 재학습. **닫힌 루프.**

> **Stage 1이 Stage 2를 게이트한다**: C3의 actuator 식별(Stage 1)이 수렴하고 C2의 관절 추종 RMSE 게이트를 넘어야만 Stage 2 자유보행으로 진입한다.

---

## 5. 통합 검증 메트릭 대시보드

각 컴포넌트의 핵심 검증 메트릭을 한 곳에. (상세 정의·임계는 각 파일.)

| 메트릭 | 컴포넌트 | Stage | pass 기준 (요약) |
|--------|----------|-------|------------------|
| 통신 주파수/지터/레이턴시 | C2 (T1) | 1 | hz≥49.5, mean latency<5ms, max<15ms, jitter<1ms, 60초 무드롭 |
| 관절 step/sine 추종 RMSE | C2(T2)·C3 | 1 | dead time 추가 ≤1 step(20ms), overshoot Δ≤5%p, 위치 RMSE ≤ ROM 5%, 대역폭비 ≥0.7 |
| 정지 토크·마찰 잔차 | C2(T3)·C3 | 1 | 마찰 보정 후 자세별 정지토크 sim/real 잔차 RMSE ≤ effort limit 5% |
| **sim torque vs real torque RMSE** | **C3 (actuator)** | 1 | held-out excitation에서 토크 NRMSE ≤ 10–15%, 파라미터 trajectory 간 일관 |
| excitation 충분성 | C3 (§5) | 1 | regressor condition number 낮음, held-out 일반화 RMSE 유지 |
| latency / 위상지연 | C2·C3·C4 | 1·2 | sim vs real 위상지연 ms 일치, closed-loop 진동 임계게인 일치 |
| 무부하 정책 안정성 | C2 (T4) | 1.5 | 토크 saturation ≤2%, 발산/chatter 없음, sim 무부하 rollout과 정성 일치, NaN 0 |
| export parity | C1 | 2 | 학습 그래프 vs JIT/ONNX 출력 ±1e-4 일치, estimator 추정값 일치 |
| estimator RMSE | C1·C4 | 2 | 추정 base lin_vel vs ground-truth MSE 수렴; 정지 시 \|추정 lin_vel\|≤0.1 m/s |
| IMU 캘리 / 정적 균형 | C2 (T6a/b) | 2 | roll/pitch bias ≤1°, 정적 stand roll/pitch σ ≤2–3°, 무발산 |
| 명령 추종 / 자유보행 | C2 (T6c) | 2 | tether 없이 60초+ 안정, 속도 추종 RMSE ≤0.15 m/s, 낙상 0 |
| per-joint trajectory/torque divergence | C4 (real2sim) | 1·2 | 동일 action replay 후 RMSE 단조 감소, 고정리그 near-overlap |
| return gap | C4 | 2 | sim return − 실기 성과 차이 감소 |
| DR coverage (in-dist rate) | C4 | 1·2 | 실기 파라미터가 DR 분포 안에 드는 비율 ↑ |
| history-latent probe / 동작품질(CoT·jerk) / push recovery | C4 | 2 | RMA 적응·AMP 정규화·강건성 작동 |

---

## 6. 기존 인프라 재사용 맵

우리 코드의 어느 파일/모듈을 어떻게 재사용·일반화하는가.

| 우리 자산 | 어떻게 쓰나 | 컴포넌트 |
|-----------|------------|----------|
| `scripts/real2sim/` ZMQ 브리지 (`sim_runner.py`/`controller.py`) | Stage1 setpoint 재생·excitation 주입·(pos,vel,torque) 기록; Stage2 배포 시 setpoint 소스를 GUI→policy 추론으로 교체 | C1·C2·C3·C4 |
| `scripts/real2sim/utils/benchmark.py` | 50Hz 주파수/레이턴시/지터 측정 → latency system ID + DR 범위 + 배포 게이트 | C2·C3·C4 |
| `scripts/real2sim/utils/data_logger.py` | 실기/sim rollout CSV 로깅(real2sim 비교 입력) | C4 |
| `scripts/real2sim/utils/robot_interface.py` | `ROS2RobotInterface` skeleton 채워 임의 로봇 일반화(Phase 2) | C2·C4 |
| `source/isaaclab_assets/.../rga.py` (ArticulationCfg, KP/KD/effort/vel, Implicit vs DCMotor) | 새 로봇 cfg 패턴 재사용; C3 파라메트릭 식별값 주입 지점(관절별 dict 존재) | C1·C3 |
| `DCMotorCfg(saturation_effort, velocity_limit, effort_limit)` | torque-speed envelope 직접 표현 | C1·C3 |
| `HIND_LEG_CFG` inertia-scaled gain (rga.py:477-493) | I_eff 실측→게인 설계 **선례** | C3 |
| `EventCfg` DR (friction/mass/CoM/actuator gain/push) | 패턴 복제 + system ID 측정값으로 범위만 교체(measured-centered) | C4 |
| `rsl_rl/.../modules/estimator.py` + `actor_critic_parkour.py`(ActorCriticRMA, StateHistoryEncoder) | estimator(base lin_vel) + RMA history encoder. policy=추정, critic=실제 | C1·C4 |
| `source/isaaclab_rl/.../rsl_rl/exporter_parkour.py` | ActorCriticRMA + estimator + normalizer 번들 export(배포 그래프가 raw proprio만 입력) | C1 |
| `exporter.py` | 단순 ActorCritic export(proprio만) | C1 |
| 커스텀 `ActuatorBase` 서브클래스 (신규, 로드맵) | actuator net 주입 — 새 로봇마다 가중치 교체로 범용 | C3 |

---

## 7. 알려진 함정

| 함정 | 내용 | 출처 / 대응 |
|------|------|------------|
| **ONNX action_scale 미반영** | ONNX export는 `action_scale`을 반영하지 않음. env는 `joint_target = action_scale·policy_output + default_joint_pos`인데 export 그래프는 raw output만 냄. | GitHub #2636. **배포 코드에서 action_scale 수동 곱·default 더함.** C2 T4에서 "최종 토크에 action_scale 정확히 한 번" 육안·수치 확인. |
| **estimator 빌드 = 러너 책임** | AMP 러너가 estimator 빌드를 누락(상속 함정)하면 actor가 실제 lin_vel을 직접 입력받아 **배포 불가** 정책 생성. | 메모리 "Parkour AMP 러너 estimator 누락 버그 fix". 새 로봇 Stage 2 학습 시 러너가 estimator를 빌드·학습하는지 **반드시 확인**, 누락 시 재학습. |
| **DR 과보수성** | system ID 없는 넓은 DR = "평균을 위한" 보수적·둔감 정책(nominal 성능 저하). 좁은 DR + system ID 없음 = 실기 OOD 전이 실패. | measured-centered(C3 측정값 중심 ±σ). 좁게 시작→점진 확대(curriculum). SPI-Active / IsaacLab #2813. |
| **normalizer 누락 export** | 학습 normalizer를 export에 안 넣으면 정규화 안 된 obs가 actor로 → 정책 붕괴. | 두 exporter 모두 `EmpiricalNormalization` deepcopy 내장. 학습에 쓴 normalizer 반드시 같이 export. |
| **전류=토크 가정 부정확** | actuator 마찰·hysteresis·lag 비선형 → motor current를 토크 proxy로 쓰면 부정확. | **토크·전류 동시 실측**(우리 자원) + actuator net이 비선형 흡수. |

---

## 8. 다음 액션 (로드맵)

실제 새 legged 로봇이 정해지면 우선순위:

1. **[C3·C1] Stage 1 고정 리그 구축** — URDF→USD + CAD 명목 ArticulationCfg(baseline) 작성. `scripts/real2sim/` ZMQ 브리지를 새 로봇 관절 수로 일반화(`robot_interface.py` 추상화).
2. **[C3] actuator system ID (최우선·무게중심)** — excitation trajectory(chirp/multi-sine/PRBS/Fourier) 설계 → 토크/전류 실측 → actuator net + 파라메트릭(friction/envelope/Kt/PD) 병행 식별 → sim torque RMSE 검증. **여기가 gap의 근원, 가장 깊게.**
3. **[C2] Stage 1 검증 사다리** — T1(통신)→T2(step/sine)→T3(정지토크)→T5(excitation). 각 게이트 통과. (T2·T3·T5 = C3 데이터 수집 겸함.)
4. **[C4] measured-centered DR 설정** — C3 측정값을 EventCfg 중심으로, ±측정σ 폭. latency·obs noise 항 추가(로드맵 항목).
5. **[C1·C4] Stage 2 학습** — floating-base RL env + estimator(러너가 빌드하는지 확인). RMA history encoder + (선택)AMP.
6. **[C1] export + parity 테스트** — estimator·normalizer 번들, action_scale 매핑 확인.
7. **[C2] Stage 2 실기 게이트** — T4 무부하 → T6a IMU캘리 → T6b 정적균형 → T6c 점진보행(tether→자유). 실패 시 C3/C4 회귀.
8. **[C4] real2sim 루프** — 배포 후 divergence 측정 → 파라미터 역추적 → 재보정 닫힌 루프. 자동화(replay·divergence 자동산출·파라미터 sweep)는 로드맵.
9. **[로드맵·비목표] ROS2 실배포** — `ROS2RobotInterface`/`DualTransportBridge` skeleton 구현(Phase 2). 이번 산출물은 플랜만, 구현 안 함.

> **우선순위 한 줄**: **고정 리그 + actuator system ID(2~3번)가 가장 먼저이자 가장 깊게.** 여기서 sim torque RMSE를 닫지 못하면 그 위 모든 단계의 측정·학습·배포가 오염된다.

---

## 정합성 노트 (통합 시 발견)

4개 컴포넌트는 전반적으로 일관됐다. 핵심 정렬 사항·미세 차이만 기록한다.

- **estimator 설계 원칙 일관**: C1·C2·C4 모두 "policy=추정값, critic/discriminator=실제값(privileged)" + "estimator 빌드=러너 책임(AMP 누락 함정)"을 동일하게 기술. 모순 없음.
- **measured-centered 합의**: C3(측정=nominal, DR은 잔차)와 C4(measured-centered DR)가 같은 철학으로 짝. C4가 명시적으로 C3를 무게중심으로 인정. 일관.
- **마찰의 예외 처리**: C3는 foot-ground μ를 force-gauge로 측정하라 하고, C4는 "마찰만은 상대적으로 넓게 두는 것이 합리적(measured-centered의 예외)"이라 한다. **모순 아님** — C3가 nominal(중심)을 측정하고 C4가 그 중심 위에 넓은 ±폭을 얹는 역할 분담. 통합 관점: 마찰은 "측정된 중심 + 넓은 잔차"로 읽으면 일관.
- **action_scale 함정 cross-reference**: C1(§4.3)이 원 출처(#2636)를 정의하고 C2(T4 안전가드)가 실기 점검 항목으로 재인용. 일관, 중복 아님(정의 vs 적용).
- **테스트 번호 표기**: C2의 사다리 표/본문은 T1·T2·T3·T5(Stage1)·T4(Stage1.5)·T6a/b/c(Stage2)로, **T4가 T5보다 뒤에 배치**(데이터 수집을 먼저 끝내고 무부하 정책을 마지막 안전 게이트로). 번호 순서≠실행 순서임을 본 통합 문서 §2 흐름도에 반영(T5까지 끝낸 뒤 T4). 누락된 번호 없음.
- **C2 사다리 표의 줄바꿈**: 원문 02 파일 T6 행에 "te\nther" 줄바꿈 오타가 있으나 의미("tether")는 명확. 내용 영향 없음.

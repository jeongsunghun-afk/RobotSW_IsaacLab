# 컴포넌트 1: 배포 파이프라인

> 마스터 플랜 4개 컴포넌트 중 **표준 절차 깊이** 담당. 새 legged 로봇(2족/4족 범용)을 IsaacLab에 등록 → 학습 → export → 실기 배포까지 end-to-end 순서를 기술한다. 무게중심(system ID 디테일)은 컴포넌트 3, gap 흡수(DR/real2sim)는 컴포넌트 4가 다루므로 여기서는 그 단계들을 **언제·어디서 호출하는지의 골격**만 제시하고 디테일은 위임한다.
>
> 표기 규약: 각 단계마다 `2-stage 매핑`(Stage 1 고정 리그 / Stage 2 자유보행), `근거:`(논문/출처), `검증:`(sim-to-real 효과를 확인하는 메트릭/실험)을 명시한다.

---

## 0. 2-Stage Scope 개요 (전 단계 공통 골격)

| Stage | 대상 | obs/state | actuator 자원 | 목적 |
|-------|------|-----------|----------------|------|
| **Stage 1 — 고정 베이스 leg 리그** | 다리(들)만 고정 jig에 장착 (base 미동) | 관절 pos/vel(encoder) + 관절 토크/전류 실측 | system ID 가능 (토크/전류 직접 측정) | actuator 모델·관절 추종 충실도 확립. base state 추정 불필요 (고정) |
| **Stage 2 — full robot 자유보행** | 전체 로봇 자유 이동 | 관절 pos/vel + base IMU(자세/각속도) + **estimator로 추정한 base lin_vel** | 동일 + IMU | 보행 정책 학습 → 실기 배포. base lin_vel은 직접 측정 난해 → estimator module |

이 분리는 sim-to-real 위험을 단계화한다. Stage 1에서 actuator/관절 추종 gap을 먼저 닫고(컴포넌트 3 system ID의 실험 베드), 그 위에서만 Stage 2 자유보행을 올린다. R_Skeleton Hind Leg가 현재 Stage 1을 점유한 레퍼런스(`scripts/real2sim/`, `R_SKELETON_HIND_LEG_CFG`).

근거: Hwangbo et al. 2019 (Science Robotics, ANYmal)는 actuator 모델을 먼저 실측 데이터로 학습해 sim-to-real gap의 가장 큰 단일 원인(actuator 비선형성)을 분리 처리했다 ([arXiv:1901.08652](https://arxiv.org/abs/1901.08652)). Stage 1 = 이 actuator-우선 철학의 우리 구현 슬롯.
검증: Stage 1 통과 게이트 = 고정 리그에서 sim 관절 추종 trajectory와 실기 trajectory의 RMSE(컴포넌트 2 실기 검증 테스트로 측정). 이 게이트를 못 넘으면 Stage 2 진입 금지.

---

## 1. 로봇 asset 준비 (URDF/USD 변환 + ArticulationCfg)

### 1.1 개념 + 순서
1. **URDF 확보**: 새 legged 로봇의 링크/관절/관성/충돌 메시 정의. 제조사 제공 또는 CAD에서 export.
2. **USD 변환**: IsaacLab은 USD asset을 사용. URDF → USD converter로 변환 후 `source/isaaclab_assets/data/Robots/<RobotName>/` 에 배치 (기존 `R_Skeleton_*`, `Hind_Leg`, `Go2Neck2` 와 동일 레이아웃).
3. **`ArticulationCfg` 작성**: `source/isaaclab_assets/isaaclab_assets/robots/<robot>.py` 에 cfg 정의. `rga.py` 패턴 재사용:
   - `spawn=UsdFileCfg(...)` (usd_path, `activate_contact_sensors=True`, rigid/articulation props)
   - `init_state` (기본 자세 joint_pos — 2족/4족마다 다름)
   - `soft_joint_pos_limit_factor=0.9`
   - `actuators={...}` (관절 그룹별 KP/KD/effort/vel limit)
4. **관절 그룹 분할**: `rga.py`는 `legs`/`feet`/`neck`/`waist` 식으로 관절을 그룹화하고 그룹별 게인을 준다. 범용 원칙 = 물리적으로 유사한 관절(같은 모터 모델)끼리 묶고, 측정된 effective inertia가 크게 다른 관절은 분리(예: `HIND_LEG_CFG`는 hip 0.163 / foot 0.0019 kg·m² 84배 차이 → inertia-scaled 게인).

### 1.2 actuator 모델 선택: Implicit vs DC Motor
| 모델 | IsaacLab cfg | 언제 | 근거 |
|------|--------------|------|------|
| **ImplicitActuatorCfg** | PD 게인 + effort/vel limit이 PhysX 솔버에 직접 들어감 | 고정밀 위치 추종 리그(Stage 1), 토크 saturation이 지배적이지 않은 관절 | `R_SKELETON_*`, `HIND_LEG_CFG` 가 사용. 솔버 내 PD라 안정적 |
| **DCMotorCfg** | `saturation_effort` + `velocity_limit` 으로 torque-speed 곡선(모터 포화) 모델 | torque-speed 한계가 보행에 본질적인 자유보행 로봇(Stage 2), 다이내믹 모션 | `RGA_GO2_CFG`의 `base_legs`가 사용 (effort 23.5 / saturation 23.5 / vel 30). Go2 실배포 파이프라인 표준 |
| **학습된 actuator network** (로드맵) | 커스텀 actuator 모듈 | 위 둘로도 실기 추종 RMSE가 안 닫힐 때 | Hwangbo 2019: 실측 (q_err, q̇)→τ MLP로 비선형 compliance/damping 포착 |

**왜 중요한가**: actuator 모델은 sim-to-real gap의 가장 큰 단일 기여원이다. 위치제어 PD 게인/포화/지연이 sim과 실기에서 다르면 같은 policy action이 다른 토크 → 다른 모션을 낸다. 따라서 이 cfg 값들은 컴포넌트 3 system ID가 실측으로 채워야 하는 1차 타깃 (Stage 1에서 측정).

2-stage 매핑: asset cfg 자체는 두 Stage 공용. 단 actuator 모델 **선택**은 Stage 1 system ID 결과로 확정 → Stage 2가 그대로 상속.
근거: Hwangbo et al. 2019 actuator network ([arXiv:1901.08652](https://arxiv.org/abs/1901.08652)); IsaacLab `DCMotorCfg`/`ImplicitActuatorCfg` API; unitree_rl_lab Go2 (DCMotor 포화 모델 + 200Hz PD) ([deepwiki/unitree_rl_lab](https://deepwiki.com/unitreerobotics/unitree_rl_lab)).
검증: 고정 리그에서 step/chirp setpoint 입력 후 sim vs 실기의 (a) 관절 위치 추종 RMSE, (b) 토크/전류 곡선 일치도. cfg를 바꿔가며 RMSE 최소화 (컴포넌트 3의 system ID 루프).

### 1.3 DR(domain randomization) hook 자리 예약
asset cfg와 함께 `EventCfg`(friction / base mass / CoM / actuator stiffness·damping randomization)를 둘 자리를 예약한다. Go2/Hind Leg는 이미 4-term DR + push_robot 보유. 실제 분포 폭 설정은 컴포넌트 4 위임 — 여기서는 "env cfg에 DR term을 두는 규약"만 명시.
근거: `_workspace/parkour_dr_research.md` (DR 함수 카탈로그).
검증: 컴포넌트 4에서 다룸.

---

## 2. 환경 / 보상 설계 (Stage 1 먼저)

### 2.1 환경 추가 규약 (모든 direct RL env 공통)
- `_common/DebugViewer` 패턴 사용 (CLAUDE.md 필수 3줄: `__init__` 활성화 → `_get_observations` 끝 `update` → `__del__` cleanup).
- 새 텐서 buffer는 반드시 `_reset_idx`에서 초기화.
- `obs_space` 차원 = `_get_observations`의 cat 결과 크기 (불일치 시 런타임 assert).

### 2.2 Stage 1 — 고정 리그 환경 (관절 추종 + system ID 베드)
**목적**: 정책 학습이 아니라 **관절 추종 충실도 확보 + actuator system ID 데이터 수집**. 현 `r2s_hind_leg` 환경이 정확히 이 역할 (`sim_runner.py`가 setpoint 주입, slew-rate 제한, state 발행).

순서:
1. 고정 base(jig)에 다리를 붙인 env (`R_SKELETON_HIND_LEG_CFG`처럼 base를 고정 pos로). `init_state.pos` 고정, base는 articulation root이되 floating 아님.
2. obs: 관절 pos/vel (학습 안 하므로 최소). reward: 정책 학습용이 아니라 추종 오차 로깅용(또는 단순 PD setpoint 재생). 실제로 R2S 리그는 **policy 없이 외부 setpoint 재생**(`sim_runner.py` line 134 `zero_action`)으로 운영 → 관절 추종/토크 측정.
3. system ID 입력 신호(step/chirp/sweep) 재생 → 관절 pos/vel/torque 로깅(`data_logger.py`).

2-stage 매핑: **Stage 1 전용**. 이 env에서 estimator 불필요(base 고정).
근거: Stage 1은 컴포넌트 3 system ID의 데이터 수집 플랫폼. actuator-우선(Hwangbo 2019) + privileged 학습 전 충실도 확보(Lee et al. 2020) ([arXiv:2010.11251](https://arxiv.org/abs/2010.11251)).
검증: 관절 추종 RMSE(sim setpoint vs 실기), 토크/전류 매칭. 컴포넌트 2의 무부하/추종 테스트로 측정.

### 2.3 Stage 2 — 자유보행 환경 (보행 정책 학습)
**목적**: full robot floating-base 보행 정책 RL 학습.

순서:
1. floating base env (Go2/parkour 류). obs = 관절 pos/vel + base IMU(중력투영/각속도) + command(vx,vy,yaw rate) + 이전 action + (선택)height scan. **base lin_vel은 obs에서 직접 넣지 않음** — 실기에서 측정 난해. 대신:
   - **policy(actor)**: estimator가 추정한 lin_vel을 받음.
   - **critic / discriminator**: 시뮬레이터의 실제 lin_vel을 받음 (학습 시에만 가용한 privileged 정보).
2. reward: 명령 추종(lin_vel/yaw) + 자연스러움/안전 penalty (smoothness, 접촉, 토크). 보상 디테일은 task별.
3. 새 buffer(estimator target 등)는 `_reset_idx` 초기화.

2-stage 매핑: **Stage 2 전용**. estimator module이 여기서 처음 등장(§3.2).
근거: privileged/teacher-student 비대칭(critic은 privileged, actor는 추정) = Lee et al. 2020 ([arXiv:2010.11251](https://arxiv.org/abs/2010.11251)) + RMA extrinsics 추정 ([arXiv:2107.04034](https://arxiv.org/pdf/2107.04034)). 우리 저장소 parkour RMA estimator 패턴과 동일 (메모리: "Parkour AMP 러너 estimator 누락 버그 fix" — estimator는 actor만 추정값, disc/critic은 실제값).
검증: 학습 후 play에서 명령 추종 오차, 보행 안정성. sim-to-real 효과는 컴포넌트 4(real2sim 비교)로 측정.

---

## 3. 학습 (rsl_rl / skrl, Stage 2 estimator 통합)

### 3.1 기본 학습 루프
```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <Task> --num_envs 4096
# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task <Task> --num_envs 32
```
- rsl_rl(PPO 계열) 또는 skrl(PPO/AMP). 환경 내부 구조는 동일, 알고리즘 라이브러리만 차이.
- WandB 로깅 권장 (`--logger wandb`).

2-stage 매핑: Stage 1은 보통 **학습 없음**(setpoint 재생). 학습 루프는 Stage 2 자유보행이 본체.

### 3.2 estimator module 통합 (Stage 2 핵심)
**설계 원칙 (parkour RMA 패턴)**:
- estimator는 raw proprio(관절 pos/vel + IMU + 이전 action history)로부터 **base lin_vel(및 기타 priv_explicit)** 를 추정하는 별도 head.
- **actor 입력** = `[proprio_normed, estimator(proprio), ...]` — 추정값 사용 (실기에서 그대로 동작).
- **critic / discriminator 입력** = 시뮬레이터 실제 lin_vel (privileged, 학습 시만).
- estimator 손실 = 추정 lin_vel vs 실제 lin_vel의 supervised(MSE). PPO 손실과 분리.

**구현 함정 (저장소 실증)**: estimator 빌드는 **러너의 책임**이다. AMP 러너가 estimator 빌드를 빠뜨려(상속 함정) actor가 실제 lin_vel을 직접 받는 배포 불가 정책이 나온 사례 있음 (메모리: "Parkour AMP 러너 estimator 누락 버그 fix"). 새 로봇 Stage 2 학습 시 러너가 estimator를 빌드·학습하는지 반드시 확인.

2-stage 매핑: **Stage 2 전용**. Stage 1은 base 고정이라 estimator 불필요.
근거: RMA adaptation module(state history → extrinsics 추정) ([arXiv:2107.04034](https://arxiv.org/pdf/2107.04034)); Lee et al. 2020 teacher-student privileged 비대칭 ([arXiv:2010.11251](https://arxiv.org/abs/2010.11251)). unitree_rl_lab도 policy + state estimator를 함께 ONNX export ([deepwiki/unitree_rl_lab](https://deepwiki.com/unitreerobotics/unitree_rl_lab)).
검증:
- (sim) estimator 추정 lin_vel vs 실제 lin_vel의 MSE 수렴 곡선.
- (배포 전) export된 그래프가 actor에 **추정값**을 먹이는지 확인(아래 §4).
- (실기) 추정 lin_vel과 외부 모캡/optical-flow 기준치 비교(가능 시).

---

## 4. Export (JIT / ONNX, action_scale 함정, normalizer 포함)

### 4.1 export 인프라 재사용
| 정책 구조 | exporter | 함수 |
|-----------|----------|------|
| 단순 ActorCritic (proprio만) | `rsl_rl/exporter.py` | `export_policy_as_jit` / `export_policy_as_onnx` |
| ActorCriticRMA + estimator (Stage 2 자유보행) | `rsl_rl/exporter_parkour.py` | `export_policy_as_jit_parkour` / `export_policy_as_onnx_parkour` (`estimator=` 인자) |

`exporter_parkour.py`는 다중 인코더(proprio / priv_explicit / history / scan)와 **estimator 번들링**을 동적 감지한다. `estimator`를 넘기고 policy에 `priv_explicit` 슬롯이 있으면, export된 그래프가 내부에서 `priv_explicit = estimator(proprio)`를 계산해 actor에 먹인다(line 244–250). 즉 **배포 그래프는 raw proprio만 입력받고 lin_vel 추정을 내부에서 수행** → 실기에서 base lin_vel 측정 불필요. 이것이 우리 estimator 설계의 배포 페이오프.

### 4.2 normalizer 포함 (필수)
두 exporter 모두 학습 시 쓴 `EmpiricalNormalization`(obs 정규화)을 그래프에 **deepcopy로 내장**한다(`exporter.py` line 80, `exporter_parkour.py` `actor_obs_normalizer`). 따라서 실기 측에서 obs를 별도 정규화할 필요 없음 — 단, **학습에 쓴 normalizer를 반드시 같이 export**해야 한다(누락 시 정규화 안 된 obs가 actor로 들어가 정책 붕괴).

### 4.3 ⚠️ action_scale 함정 (GitHub #2636)
**ONNX export는 `action_scale`을 반영하지 않는다.** env는 보통 `applied = action_scale * policy_output + default_joint_pos` 로 관절 setpoint을 만드는데, export된 그래프는 raw `policy_output`만 낸다. 따라서 **실기 배포 코드에서 출력에 `action_scale`을 수동 곱하고 `default_joint_pos`를 더해야 한다**. 이를 빠뜨리면 액션이 잘못된 스케일로 들어가 즉시 실패. (관련: parkour exporter에서 `priv_explicit_dim=-1` fallback fix도 구버전 .pt play 호환을 위해 존재.)

배포 시 반드시 확인할 매핑:
```
joint_target = action_scale * onnx_output + default_joint_pos   # action_scale은 코드에서 수동 적용
```

2-stage 매핑: export는 **Stage 2 정책 배포 직전** 단계. Stage 1은 export할 정책이 없음(setpoint 재생).
근거: IsaacLab `exporter.py`/`exporter_parkour.py`; GitHub issue #2636 (ONNX action_scale 미반영); unitree_rl_lab도 policy+estimator ONNX export ([deepwiki/unitree_rl_lab](https://deepwiki.com/unitreerobotics/unitree_rl_lab)).
검증: export 직후 **parity 테스트** — 동일 obs 배치에 대해 (a) 학습 그래프 actor 출력 vs (b) export된 JIT/ONNX 출력이 일치하는지(±1e-4). estimator 번들 그래프는 추정 lin_vel이 학습 시 estimator와 같은 값을 내는지 확인.

---

## 5. 실기 배포 (ZMQ / ROS2 인터페이스, 50Hz 제어 루프)

### 5.1 인터페이스 재사용: ZMQ 브리지 (Phase 1 완성)
`scripts/real2sim/`의 ZMQ 브리지를 일반화한다:
- `sim_runner.py`: IsaacLab 백엔드. ZMQ PULL로 setpoint 수신(별도 폴링 스레드, line 66–73), PUB로 state(pos/vel/torque + timestamp) 발행, slew-rate 제한.
- `controller.py`: PyQt5 GUI(슬라이더 + 모션 클립 재생). Python 3.10 별도 conda 환경(Isaac Sim 3.11과 분리).
- `utils/zmq_bridge.py`: ZMQ IPC. `utils/benchmark.py`: 50Hz 주파수/레이턴시/지터 측정. `utils/robot_interface.py`: ROS2 미구현 skeleton(`DualTransportBridge` 자리).

배포 시 이 구조의 setpoint 소스를 **GUI 슬라이더 → export된 policy 추론**으로 교체:
```
[real robot/sim] --state(ZMQ PUB)--> [policy node: obs 조립 → ONNX 추론 → action_scale 적용] --setpoint(ZMQ PUSH)--> [robot]
```

2-stage 매핑:
- **Stage 1**: ZMQ로 setpoint 재생(현 R2S 흐름 그대로) → 관절 추종/system ID. policy 없음.
- **Stage 2**: 동일 ZMQ 골격이되 setpoint 소스 = export된 보행 policy. base IMU + estimator(그래프 내장)로 obs 조립.

### 5.2 50Hz 제어 루프 (이중 rate)
- **policy 추론 50Hz** (period 20ms): 한 step의 전체 추론+IPC가 20ms 안에 끝나야 함.
- **PD 제어 200Hz**: 관절 PD는 더 빠르게(unitree_rl_lab은 policy 50Hz / PD 200Hz on Jetson Orin NX). IsaacLab의 `step_dt`(decimation)와 일치시켜 sim-real timing 정합.
- 합격 기준(INTEGRATION_GUIDE.md): `current_hz ≥ 49.5`, `mean_latency < 5ms`, `max_latency < 15ms`, `jitter < 1ms`.

### 5.3 ROS2 (Phase 2, 로드맵 — 이번 미구현)
`utils/robot_interface.py`의 `DualTransportBridge` skeleton을 채워 ZMQ ↔ ROS2 듀얼 트랜스포트로 확장. 실배포는 ROS2 토픽(joint cmd/state, IMU)으로. **이번 산출물은 플랜에만 포함, 구현 안 함**(스펙 Non-Goal).

2-stage 매핑: Stage 2 실배포의 최종 형태. ROS2는 Phase 2.
근거: `scripts/real2sim/INTEGRATION_GUIDE.md`(ZMQ 아키텍처, 50Hz 벤치 합격 기준); unitree_rl_lab(50Hz policy / 200Hz PD, inference <20ms, C++ + Unitree SDK2) ([deepwiki/unitree_rl_lab](https://deepwiki.com/unitreerobotics/unitree_rl_lab)).
검증: `utils/benchmark.py`로 50Hz 주파수/레이턴시/지터 측정 → 위 합격 기준 충족. 실기 무부하(다리 들고) policy 실행으로 발산 없이 안정 출력 확인(컴포넌트 2).

---

## 6. End-to-End 요약 표 (단계 ↔ 2-Stage ↔ 인프라)

| # | 단계 | Stage 1 (고정 리그) | Stage 2 (자유보행) | 재사용 인프라 |
|---|------|---------------------|--------------------|----------------|
| 1 | asset 준비 (URDF→USD, ArticulationCfg, actuator 모델) | actuator cfg 확정(system ID) | cfg 상속 | `rga.py` ArticulationCfg 패턴 |
| 2 | 환경/보상 설계 | setpoint 재생 env(추종/ID 베드) | floating-base 보행 RL env | `_common/DebugViewer`, `r2s_hind_leg` |
| 3 | 학습 (+estimator) | 보통 학습 없음 | PPO/AMP + estimator(actor=추정, critic=실제) | rsl_rl/skrl, parkour RMA 패턴 |
| 4 | export (JIT/ONNX) | — | estimator 번들 + normalizer 내장, **action_scale 수동** | `exporter.py`, `exporter_parkour.py` |
| 5 | 실기 배포 (ZMQ/ROS2, 50Hz) | setpoint 재생 + system ID 로깅 | policy 추론 → setpoint | `scripts/real2sim/`(ZMQ, benchmark) |

---

## 7. 핵심 검증 게이트 (단계 전이 조건)

1. **Stage 1 → Stage 2 게이트**: 고정 리그 관절 추종 RMSE + 토크/전류 매칭이 임계 이하(컴포넌트 3 system ID 수렴). 못 넘으면 자유보행 진입 금지.
2. **학습 → export 게이트**: estimator 추정 lin_vel MSE 수렴 + play 명령 추종 안정.
3. **export → 배포 게이트**: parity 테스트(학습 그래프 vs export 그래프 출력 일치), action_scale 매핑 확인.
4. **배포 게이트**: 50Hz 벤치(`benchmark.py`) 합격 + 실기 무부하 policy 안정 실행.

각 게이트의 **메트릭 디테일과 sim-real gap 정량화**는 컴포넌트 2(실기 검증 테스트)·3(system ID)·4(DR/real2sim)에 위임한다. 본 컴포넌트는 그 게이트들을 파이프라인의 어느 순서에 끼우는지를 정의한다.

---

## 참고 문헌
- Hwangbo et al., "Learning agile and dynamic motor skills for legged robots," Science Robotics 2019 (actuator network) — [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
- Lee et al., "Learning quadrupedal locomotion over challenging terrain," Science Robotics 2020 (teacher-student privileged learning) — [arXiv:2010.11251](https://arxiv.org/abs/2010.11251)
- Kumar et al., "RMA: Rapid Motor Adaptation for Legged Robots," RSS 2021 (extrinsics estimator / adaptation module) — [arXiv:2107.04034](https://arxiv.org/pdf/2107.04034)
- unitree_rl_lab (Isaac Lab → ONNX → Go2, 50Hz policy / 200Hz PD, policy+estimator export) — [deepwiki/unitree_rl_lab](https://deepwiki.com/unitreerobotics/unitree_rl_lab)
- 저장소 인프라: `source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py`, `exporter_parkour.py` (GitHub #2636 action_scale 함정); `source/isaaclab_assets/isaaclab_assets/robots/rga.py`; `scripts/real2sim/` (`sim_runner.py`, `INTEGRATION_GUIDE.md`, `utils/`)

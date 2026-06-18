# hind_leg 환경 방법론 타당성 검증 보고서

**검증 대상**: `HindLegHistoryEnvCfg` (실제 학습되는 유일 cfg)
**로봇**: HIND_LEG_CFG — 8관절 2족 보행기 (HL/HR × Hip/Thigh/Calf/Foot)
**검증 일시**: 2026-06-02
**검증자**: validate-method (방법론 read-only 리뷰어)

---

## 1. 총평 한 줄

> **RED: "가만히 서서 명령 무시" 정적 attractor가 reward 구조 안에 존재한다 — feet_air_time(0.5s threshold)·exp tracking(/0.1)·base_height(-10) 세 항의 상호작용이 동적 보행을 적극적으로 억제하며, 이 동일 메커니즘의 낮은 임계값(0.3s) 버전이 이 코드베이스에서 catastrophic failure를 유발한 전례가 있다.**

---

## 2. 최우선 위험: "정적 attractor" — 세 항의 상호작용

이것이 단일 항목의 문제가 아니라 설계 수준의 문제다. 아래 세 항이 동시에 같은 방향을 가리킨다.

### 구성 항목

**항목 A: `feet_air_time` — threshold 0.5s, net negative**

```python
air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (||cmd_xy|| > 0.1)
```

- `first_contact`가 발화할 때만 reward가 적용된다. 로봇이 한 발도 들지 않으면 (`first_contact=0` 상시) 이 항은 **0**이다 — 패널티가 없다.
- 발을 들었다 내디딜 때, air_time < 0.5s이면 **(last_air_time - 0.5) < 0** → 음수 reward.
- 50Hz 제어에서 0.5s = 25 스텝. 2족 보행의 일반적 air_time(0.2–0.4s)은 이 임계값보다 **짧다**.
- 수치: air_time=0.3s → (0.3-0.5)×0.5×0.02 = **-0.002/foot/step**, air_time=0.2s → **-0.003/foot/step**.
- **policy gradient 방향**: 발을 들지 않으면 0, 발을 들면 음수 → "발을 들지 말라"는 지속적 편향.

[이론적 근거] 이 코드베이스 parkour env에서 threshold=0.3s 버전이 catastrophic failure를 유발했다 (commit `fc1b0a874dd`, 2026-05-22 분석). `Episode_Reward/feet_air_time`이 학습 전 구간에서 음수였고, 발걸음 자체가 억제됐다. 현재 hind_leg에서는 0.3 → **0.5s로 더 높이** 설정되어, 동일 메커니즘이 더 강하게 작동한다.

**항목 B: `lin_vel tracking` exp 커널 — `/0.1` sigma 과도하게 좁음**

```python
lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.1)
```

- sigma=0.1 (m/s)^2 에서: 0.1m/s 추적 오차 → exp(-0.1/0.1)=exp(-1)=0.37, reward=0.0074
- 1m/s 오차 → exp(-10) ≈ 1e-5, reward≈0 (사실상 없음)
- command_range [-2.0, 1.0]에서 랜덤 초기 정책이 2m/s 후진을 즉시 추종할 확률 = 0.
- **학습 초기 전체 구간에서 lin_vel reward ≈ 0** → 발을 움직일 양의 동기가 거의 없다.

[이론적 근거] legged_gym 표준 (Rudin et al. 2022, "Learning to Walk in Minutes"): sigma=0.25. IsaacLab velocity tracking 예시: sigma=0.25. /0.1은 4배 더 좁아 학습 초기 gradient 신호가 4배 약해진다. [가설] — 비교 실험 없이 확정 불가하나 표준에서 명확히 이탈.

**항목 C: `base_height` penalty — scale=-10, 2족 CoM 역학과 충돌**

```python
base_height = torch.square(root_link_pos_w[:, 2] - default_root_state[:, 2])
```

- scale=-10.0 (History cfg). step_dt=0.02s 곱 후 dz=0.1m → -0.002/step, dz=0.3m → -0.018/step.
- 2족 보행은 매 보폭마다 inverted-pendulum 역학에 따라 CoM이 z방향으로 진동한다. 걷는 동안 CoM은 target height에서 지속적으로 벗어난다.
- **결과**: 동적 보행이 유발하는 CoM 진동 자체가 패널티를 발생시킨다. 정적 double-support stance는 CoM을 0.6m에 고정 → 페널티 최소.

[이론적 근거] Li et al. 2021 (Berkeley humanoid), Zhuang et al. 2023 (bipedal RL): base_height 항은 일반적으로 collapse 방지용 소규모 항(-1~-2 scale)으로 사용. -10은 주행 tracking reward scale(1.0)의 **10배**. CoM excursion을 허용하지 않으면 bipedal gait가 학습되지 않는다.

### attractor 종합

세 항이 공동으로 가리키는 local optimum:

| 행동 | feet_air_time | lin_vel track | base_height | 합산 |
|------|--------------|--------------|------------|------|
| 정적 double-support (발 안 듦) | 0 | ≈0 | ≈0 | 0 |
| 동적 보행 (발 들어 이동) | 음수 (-0.002~-0.006) | 작은 양수 | 음수 (CoM 진동) | 음수 가능 |

**결론**: 정책 초기화 이후 정적 attractor에 수렴할 구조적 유인이 존재한다. 이는 reward hacking 시나리오 (Ng et al. 1999 — trivial behavior maximizing reward)에 해당한다.

**판정: RED**

---

## 3. 항목별 평가 표

| 항목 | 판정 | 근거 요약 |
|------|------|-----------|
| feet_air_time threshold=0.5s | **RED** | 코드베이스 내 0.3s 버전 catastrophic failure 전례; 0.5s는 2족 정상 air_time(0.2-0.4s)보다 높아 매 스텝 음수 reward 발생 |
| lin_vel exp tracking sigma=0.1 | **YELLOW** | legged_gym 표준 0.25에서 이탈, 학습 초기 gradient 신호 과도하게 약함; ablation 필요 |
| base_height scale=-10 (History cfg) | **RED** | 2족 CoM 진동을 패널티화, lin_vel track(scale=1.0)의 10배. 정적 double-support attractor 강화 |
| flat_orientation scale=-0.0 (History cfg) | **YELLOW** | orientation 정규화가 완전히 OFF. 토르소 자세 유지를 termination + base_height에만 의존. 2족에서 자세 제어 신호 없음 |
| command lin_vel_x [-2.0, 1.0] 비대칭 | **YELLOW** | 후진 2m/s는 전진 1m/s의 2배. command_curriculum=False라 처음부터 전 범위 랜덤 샘플. 학습 초기 2m/s 후진 추종 불가 → 대부분 episode에서 large tracking error |
| lin_vel_y=[0,0] 고정 + reward는 y 포함 | **GREEN** | y=0 고정이면 y error=0 항상 → reward에 미포함과 동일. 불필요하나 무해 |
| similar_to_default scale=-0.1 | **GREEN** | 표준 regularlization 항. 2족에도 무난히 적용 가능. scale 적절 |
| ang_vel_xy penalty scale=-0.01 | **GREEN** | 표준 self-righting 항. scale 낮아 과도한 억제 없음 |
| z_vel penalty scale=-2.0 | **GREEN** | 과도한 bouncing 방지. 표준 범위 |
| dof_torques/accel penalty | **GREEN** | 표준 smooth locomotion 항 |
| action_rate scale=-0.001 | **GREEN** | 낮은 scale, 표준 |
| undesired_contacts scale=-1.0 | **GREEN** | 표준 |
| termination scale=-100 → 실효 -2.0/event | **YELLOW** | scale=-100에 step_dt=0.02 곱해 실제 한 번의 종료 = -2.0. 크기 자체는 적당하나, 2족 불안정 초기에 과도한 조기 종료가 학습 신호를 차단할 수 있음 |
| similar_to_default (2족 적용) | **GREEN** | 좌우 부호 반전(HL_Thigh=-0.2618, HR_Thigh=+0.2618)이 정상 거울 대칭이고 similar_to_default와 정합 |
| friction DR (0.4-1.5 / 0.3-1.2) | **GREEN** | RMA (Kumar et al. 2021) 표준 범위 내 |
| actuator gain DR (scale 0.75-1.5 / 0.3-3.0) | **YELLOW** | damping scale 0.3-3.0 → 10x 범위. 과도한 범위는 학습 불안정 유발 가능. stiffness 0.75-1.5는 표준 |
| add_base_mass (-1, +3 kg) | **YELLOW** | 실제 base mass 불명. -1kg이 base mass의 20% 이상이면 동역학 급변. [가설] — USD 미확인 |
| COM randomization | **GREEN** | 범위 소 (±0.08/0.04/0.02m), 표준 |
| push velocity ±0.5 m/s | **GREEN** | 4-8s 간격, 2족에 mild 수준 |
| ImplicitActuator kp=25 / kd=0.5 | **YELLOW** | Go2(4족) 동일 수치. 2족 full-weight 지지 시 kp=25가 충분한지 [가설]. max PD 토크 = 25×0.25=6.25Nm, Hip/Thigh 한계 28Nm의 22% — 여유는 있으나 gravity sag 하에 효과 불명 |
| action_scale=0.25 | **GREEN** | 표준. joint limit saturation 위험 없음 |
| control frequency 50Hz (dt=0.005, decimation=4) | **GREEN** | 표준 범위 (50-200Hz). 실 하드웨어 통신 한계(100-200Hz) 내 |
| observation noise 모델링 | **GREEN** | Gaussian+bias 양쪽 있음. sim-to-real 표준 |

---

## 4. catastrophic 위험 후보 우선순위

### #1 — RED: feet_air_time threshold=0.5s (즉시 수정 필요)

**메커니즘**: 매 발걸음마다 음수 reward 발생 → policy gradient가 "발을 들지 말라"는 방향으로 지속적으로 업데이트됨.

**코드베이스 전례**: commit `fc1b0a874dd` (2026-05-22) — parkour 환경에서 threshold=0.3s 버전이 `Episode_Reward/feet_air_time`을 학습 전 구간에서 음수로 유지하며 catastrophic failure 유발. 현재 hind_leg는 threshold=**0.5s로 더 높아** 동일 메커니즘이 더 강하게 작동한다.

**수정 방향** (설계자 판단 영역, 이하는 검증 근거에 기반한 옵션):
- Option A: threshold를 0.15–0.25s로 낮추거나, 완전히 제거 (cmd gate만 유지)
- Option B: `(last_air_time - threshold)` 대신 air_time > threshold 조건부 양수만 추출: `torch.clamp(last_air_time - threshold, min=0)`
- Option C: 2족에 맞는 gait phase 기반 reward로 교체 (Siekmann et al. 2021 — Periodic Reward Composition)

### #2 — RED: base_height scale=-10과 lin_vel scale=1.0의 불균형

**메커니즘**: dz=0.316m 이상이면 base_height 패널티가 lin_vel 보상을 완전히 압도. 2족 보행 중 자연스러운 CoM 진동이 이 범위에 들어가면 "걷는 것이 서 있는 것보다 불리"한 상황 발생.

**수정 방향**: scale을 -1.0 ~ -2.0으로 낮추고, 실제 보행 중 CoM 진동 폭을 empirically 측정 후 재조정.

### #3 — YELLOW (atractor 기여): exp tracking sigma=0.1

**메커니즘**: 학습 초기 정책이 command를 추종 못하면 reward ≈ 0 → "걸으려는 동기" 자체가 학습 초기에 없음. 특히 [-2.0, 1.0] 범위의 random command에서 초기 오차가 크면 gradient 신호가 극히 약하다.

**수정 방향**: sigma=0.25 (legged_gym 표준)으로 완화. Rudin et al. 2022 참조.

---

## 5. sim-to-real 우려점

### 5-1. flat_orientation=0.0 OFF (History cfg) — 실 로봇 토르소 제어 불안

[이론적 근거] 2족 로봇의 실 배포 시 가장 흔한 sim-to-real gap 원인 중 하나는 pitch/roll 불안정. flat_orientation 항 없이 base_height와 termination만으로 자세를 유지할 경우, 시뮬레이터에서는 다소 tilted 상태로 locomotion을 학습할 수 있으나 실 로봇에서는 불안정 발산 위험이 있다.

**권장**: flat_orientation_reward_scale을 -0.5 ~ -1.0으로 복원 (History cfg에서 의도적으로 0으로 설정된 이유가 있다면 문서화 필요).

### 5-2. kp=25/kd=0.5 — 2족 sim-to-real 적합성 불명

Go2(4족, 12kg, 4-leg support)와 동일한 PD gain이 8관절 2족 로봇에 그대로 이식됐다. 2족은 2개의 다리로 전체 무게를 지탱하므로 단일 다리 stance 시 요구 torque가 4족보다 크다. ImplicitActuator는 실제 모터 특성을 반영하지 않으므로 (IsaacLab 문서 — ImplicitActuator는 PD control의 ideal 버전), sim에서 작동해도 실 로봇에서 kd=0.5는 underdamped 진동 유발 가능성 있다.

[가설] — USD의 실제 질량/관성 정보 없이 확정 불가. 실 로봇 스펙과 대조 필요.

### 5-3. contact_sensor update_period=0.005s vs step_dt=0.02s

ContactSensorCfg `update_period=0.005s` (200Hz)이고 제어 step_dt=0.02s (50Hz). 센서 업데이트가 제어 주기보다 4배 빠름 — 정상이며 `last_air_time` 정밀도 확보에 유리. sim-to-real 문제 없음.

### 5-4. DR damping scale 0.3-3.0 (10x 범위) — 과도한 범위

stiffness scale 0.75-1.5는 표준이지만 damping scale 0.3-3.0은 10배 범위다. 실 로봇의 damping 변동이 이 정도는 아닐 것이며, 과도한 DR은 학습 불안정을 유발할 수 있다. Tobin et al. 2017의 권장은 "너무 넓은 DR은 policy 학습 자체를 어렵게 한다"는 경고를 포함한다. scale 0.5-2.0으로 좁히는 것을 권장.

### 5-5. action noise std=0.05 (매 스텝) + bias std=0.015

observation noise는 적절히 설정됨. action noise도 표준 수준. sim-to-real 측면 별도 우려 없음.

---

## 6. 요약 위험 등급

| 등급 | 항목 | 우선순위 |
|------|------|---------|
| RED | feet_air_time threshold=0.5s → net negative, attractor 형성 | 1위 |
| RED | base_height scale=-10 → lin_vel 압도, 2족 CoM 진동 패널티 | 2위 |
| YELLOW | exp tracking sigma=0.1 (너무 좁음, 초기 gradient 신호 약함) | 3위 |
| YELLOW | flat_orientation=0.0 OFF in History cfg | 4위 |
| YELLOW | command [-2.0, 1.0] 비대칭 + curriculum 없음 | 5위 |
| YELLOW | actuator damping DR 0.3-3.0 (10x 범위) | 6위 |
| YELLOW | add_base_mass -1~+3kg (base mass 불명) | 7위 |
| YELLOW | termination -100×step_dt 실효 -2.0 (초기 잦은 종료 시 신호 차단) | 8위 |
| GREEN | similar_to_default, ang_vel_xy, z_vel, action_scale, DR(friction/COM/push), 제어 주파수, obs noise | — |

**Yellow 5개 이상 누적 (7개): 논문/보고서 제출 전 ablation 및 수치 근거 보강 필요.**

---

## 참고 논문

- Rudin et al. 2022. "Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning." CoRL. (sigma=0.25 tracking standard)
- Siekmann et al. 2021. "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition." ICRA. (gait phase reward alternative to air_time)
- Kumar et al. 2021. "RMA: Rapid Motor Adaptation for Legged Robots." RSS. (DR range standard)
- Tobin et al. 2017. "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World." IROS.
- Ng et al. 1999. "Policy Invariance under Reward Transformations: Theory and Application to Reward Shaping." ICML. (trivial behavior / reward hacking)
- Rudin et al. 2022 (legged_gym): https://github.com/leggedrobotics/legged_gym (reward weight 표준 참조)

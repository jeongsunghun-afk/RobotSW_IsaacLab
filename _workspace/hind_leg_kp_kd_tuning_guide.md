# hind_leg actuator Kp/Kd 튜닝 가이드

> 목적: 직접 만든 8-DOF 2족 로봇 `hind_leg`의 actuator stiffness(Kp)/damping(Kd)를
> 어떻게 구하는지 — 이론·관행 조사 + Isaac Sim 실용 절차 + 현재 cfg 진단.
> 작성: 2026-06-08. 상태: **방법론 가이드 + 가설(테스트 대상) bracket**. 확정값 아님.

---

## 0. 결론부터 (TL;DR)

1. **게인은 "값"이 아니라 "절차"로 구한다.** 핵심 공식은
   `Kp = ω_n²·I_eff`, `Kd = 2·ζ·√(Kp·I_eff)` (ζ=1이면 `Kd = 2·√(Kp·I_eff)`).
   → 관절별 유효관성 `I_eff` 추정 → 목표 고유진동수 ω_n·감쇠비 ζ 선택 → 게인 계산 →
   토크/physics-dt 두 상한으로 sanity check → Gain Tuner·drop·step으로 검증 → sweep → DR.

2. **현재 cfg는 Go2/A1 4족 기본값과 동일하다**: `Kp=25, Kd=0.5` (전 관절 균일).
   이건 ~12kg 4족 quadruped의 값이다. hind_leg는 **7.528kg 2족**이므로 절대값을
   그대로 쓸 근거가 없다 — 가설로서 "현재 Kd는 under-damped일 가능성"이 크다(§3).

3. **두 개의 상한이 Kp를 "수백"이 아니라 "수십"으로 캡한다**(§3.2):
   - effort 한계: hip/thigh effort=28, action_scale=0.25 → `Kp ≲ 28/0.25 = 112`.
   - physics dt 한계: 40Hz 변형은 physics dt=0.025s라 `ω_n ≲ 40` → `Kp ≲ ~50–80`.
   → **stiff 게인은 200Hz-physics(decimation=4) 변형에서만 안정적으로 가능.**

4. **관절별 유효관성 `I_eff`를 USD에서 실측 완료**(§6). foot 0.0019 ~ hip 0.163 kg·m²로
   **84배 차이.** → 균일 Kp/Kd가 관절마다 완전히 다른 ω_n/ζ를 만든다(proximal ζ≈0.12 심한
   under-damped, foot ω_n≈113 과도). **inertia-scaled 게인**으로 교체 권장(§6.2 확정 표).

---

## 1. Kp/Kd는 실제로 어떻게 튜닝되나 (이론 + 관행)

### 1.1 position-target PD의 의미

policy가 `q_target`만 출력하고 actuator가 내는 토크:
```
τ = Kp·(q_target − q) − Kd·q̇        (q̇_target = 0, τ_ff = 0)
```
- **Kp (stiffness, N·m/rad)**: 위치 오차에 비례하는 복원 토크 = 가상 비틀림 스프링. 추종 강도.
- **Kd (damping, N·m·s/rad)**: 관절 속도에 비례하는 제동 토크 = 가상 댐퍼. 진동 억제·정착.

RL에서는 q̇_target·τ_ff가 0이라 사실상 비례(P) 제어처럼 동작한다.

### 1.2 2차 시스템 · 임계 감쇠 (핵심 공식)

관절을 유효관성 `I_eff`의 1자유도 스프링-댐퍼로 보면
`I_eff·q̈ + Kd·q̇ + Kp·q = Kp·q_target`, 표준형 `q̈ + 2ζω_n q̇ + ω_n² q = …` 와 매칭:
```
ω_n = √(Kp / I_eff)          # 고유진동수 (rad/s)
ζ   = Kd / (2·√(Kp·I_eff))    # 감쇠비
```
역산:
```
Kp = ω_n²·I_eff
Kd = 2·ζ·√(Kp·I_eff)         # ζ=1(임계감쇠): Kd = 2·√(Kp·I_eff)
```
- ζ<1 under-damped → overshoot·진동(다리 떨림). ζ=1 → 진동 없이 최단 정착.
  ζ>1 over-damped → 둔함(sluggish)·추종 지연.
- **중요(advisor 교정): Kp는 I_eff에 비례한다.** 무거운 로봇일수록 같은 ω_n에 더 큰 Kp.
  Cassie(~30kg)·H1(~50kg)의 Kp 100~200을 7.5kg 로봇에 그대로 쓰면 안 되는 이유.

### 1.3 안정성 제약: physics dt × ω_n

NVIDIA Omni Physics: **`Δt·ω_n`를 1보다 한참 크게 두지 말 것.** Kp를 키워 ω_n이
오르면 작은 physics dt가 강제되거나 발산한다. → physics dt가 클수록 Kp 상한이 낮다.

### 1.4 reflected inertia / armature

기어드 모터는 로터 관성이 기어비 제곱만큼 출력축에 반사된다: `armature = J·G²`.
이 항이 §1.2의 `I_eff`에 더해져 고주파 안정화 + 실제 모터 관성감을 재현한다.
(예: IsaacLab Go2 armature=0.01, GO-M8010-6 기어비 6.33). **hind_leg는 현재 armature 미설정** — 기어드 모터라면 추가 권장.

### 1.5 실제 하드웨어 튜닝 관행

1. 모터 스펙(rated/peak torque, 기어비, no-load speed)에서 Kp 상한 산정 —
   `Kp·err`가 `τ_max`를 자주 넘으면 saturation → bang-bang → 진동.
2. 목표 ω_n(보행 대역 ~2–5Hz)·I_eff로 `Kp = ω_n²·I_eff` 출발점.
3. Kd는 ζ≈0.7–1.0에서 시작. QDD/proprioceptive 액추에이터는 낮은 Kp(20–60)로도 부드러운 접촉.
4. step/sine 응답으로 overshoot·정착시간 보고 미세조정.
5. **sim-to-real**: 실 로봇 저수준 컨트롤러 PD 게인과 sim Kp/Kd를 **같은 단위**로 일치,
   제어 주파수도 일치, effort/velocity_limit으로 토크-속도 곡선 모델링, DR로 편차 강인화.

---

## 2. 레퍼런스 게인 값 (IsaacLab 로컬 코드 — 검증됨)

> 아래 IsaacLab 로컬 cfg 값은 디스크에서 직접 읽은 값이라 신뢰 가능.
> 외부 문서/논문 URL(NVIDIA docs, arxiv 등)은 이번 세션에서 재-fetch하지 않았으므로
> "표준 참고문헌, 미검증"으로 취급. (특히 미래 날짜 arxiv 인용은 폐기.)

### 2.1 IsaacLab의 두 actuator 모델 — hind_leg는 ImplicitActuator

| 모델 | 적용 | PD 위치 | 토크 필드 | 일반 Kp |
|---|---|---|---|---|
| `DCMotorCfg` | Go2/A1/ANYmal (quadruped) | Python 명시 PD + 토크-속도 saturation | `effort_limit`+`saturation_effort` | 25–40 |
| `ImplicitActuatorCfg` | Cassie/H1/G1 (biped), **hind_leg** | PD를 PhysX solver 내장(안정) | `effort_limit_sim` | 40–200 |

→ hind_leg는 ImplicitActuator라 고 stiffness가 수치적으로 가능은 하다(단 §3 상한 내).

### 2.2 quadruped (균일 게인)

| 로봇 | Kp | Kd | effort | vel | armature |
|---|---|---|---|---|---|
| Go2 | 25 | 0.5 | 23.5 | 30 | 0.01 |
| A1 | 25 | 0.5 | 33.5 | 21 | 0.01 |
| ANYmal | 40 | 5.0 | 80 | 7.5 | — |

### 2.3 biped/humanoid (관절 그룹별 차등 — 패턴이 중요)

**Cassie**(~30kg): hip_abduction/rotation 100, hip_flexion/thigh/ankle 200, toe 20–40. Kd 3/6/1.
**H1**(~50kg): hip_yaw/roll 150, hip_pitch/knee/torso 200, ankle 20. Kd 5/4.
**패턴**: proximal(hip/knee) 강하게, distal(ankle/toe) 약하게(부드러운 접촉).
**그러나 절대값은 질량 비례** → 7.5kg hind_leg는 이 패턴을 따르되 크기는 1/4~1/7로.

---

## 3. hind_leg 현재 cfg 진단

### 3.1 확인된 스펙 (코드 직접 확인)

출처: `source/isaaclab_assets/isaaclab_assets/robots/rga.py:436` (`HIND_LEG_CFG`),
`source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env_cfg.py`

| 항목 | 값 |
|---|---|
| 구조 | 2족 8-DOF: `{L,R} × {hip, thigh, calf, foot}` |
| actuator | `ImplicitActuatorCfg`, 전 관절 그룹 `legs` 하나 |
| **stiffness (Kp)** | **25.0** (전 관절 균일) |
| **damping (Kd)** | **0.5** (전 관절 균일) |
| effort_limit_sim | hip/thigh **28**, calf **42**, foot **56** N·m |
| velocity_limit_sim | hip/thigh 29.6, calf 19.7, foot 14.8 rad/s |
| armature | **미설정** |
| 총 질량 | ~7.528 kg (11 link) |
| action | position target, `action_scale=0.25`, `q_target=q_default+0.25·a` |
| 제어 — 변형 A | `dt=1/40, decimation=1` → **physics 40Hz / control 40Hz** |
| 제어 — 변형 B | `dt=1/200, decimation=4` → **physics 200Hz / control 50Hz** |
| DR | stiffness ×(0.75,1.5), damping ×(0.3,3.0) |

> ⚠️ **확인 요청**: effort 한계가 distal로 갈수록 큼(foot 56 > calf 42 > hip/thigh 28).
> 보통 proximal이 더 강한데 반대다. 의도된 설계인지(예: foot에 큰 토크 필요) 아니면
> cfg 실수인지 확인 필요.

### 3.2 두 상한 — Kp는 "수십" 영역에 갇힌다 (advisor 교정 핵심)

- **effort 상한**: action_scale=0.25에서 full action 시 명령 오차 0.25 rad.
  saturation 안 나려면 `Kp·0.25 ≲ effort` →
  hip/thigh `Kp ≲ 112`, calf `Kp ≲ 168`, foot `Kp ≲ 224`.
- **physics dt 상한** (`ω_n·Δt ≲ 1`):
  - 변형 A(physics dt=0.025): `ω_n ≲ 40` → `Kp ≲ 1600·I_eff`.
    I_eff 0.03~0.05면 **Kp ≲ 50~80**. → A 환경에선 stiff 게인 자체가 불안정.
  - 변형 B(physics dt=0.005): `ω_n ≲ 200` → effort 상한(112)이 먼저 작용.
- **질량 논리**: Cassie 100~200은 30kg 기준. 7.5kg·동일 ω_n이면 1/4~1/7 → **수십대**.
  세 근거(질량·effort·dt)가 모두 "수십"으로 수렴 → **수백 Kp는 부적절.**

### 3.3 현재 Kd=0.5 진단 (가설, I_eff 가정 하)

plausible `I_eff ≈ 0.02~0.05 kg·m²` 가정 시, 현재 Kp=25/Kd=0.5의 감쇠비:
```
ζ = 0.5 / (2·√(25·I_eff)) ≈ 0.22 ~ 0.35   (I_eff 0.02~0.05)
```
→ **under-damped 추정**(진동·overshoot 경향). 단 I_eff 미확정이므로 **가설**.
실제 step response로 overshoot·진동을 보고 확정해야 한다(§4 verify).

---

## 4. Isaac Sim에서 적절한 Kp/Kd를 구하는 절차 ★ (질문의 핵심 답)

> 이 절차가 "방법"이다. 아래 숫자는 절차의 산출 예시이지 정답이 아니다.

**Step 1 — 관절별 유효관성 `I_eff` 추정.**
home pose에서 각 관절이 본 원위 링크 관성. Gain Tuner는 "관절 양쪽 질량 기반 관성"을 사용.
기어드면 `armature = J·G²`를 더한다. → **이 입력이 없으면 모든 절대 Kp는 추정**(§5에서 추출).

**Step 2 — 목표 ω_n, ζ 선택.**
보행: `ω_n ≈ 15~25 rad/s`(≈2.5~4Hz), `ζ ≈ 0.8~1.0`.
**physics dt 상한 확인**: 변형 A는 `ω_n ≲ 40`, 변형 B는 `≲ 200`.

**Step 3 — 게인 계산.**
```
Kp = ω_n²·I_eff
Kd = 2·ζ·√(Kp·I_eff)
```
proximal(hip/thigh) 높게, distal(foot) 낮게 — biped 패턴. armature도 함께.

**Step 4 — 두 상한 sanity check.**
`Kp·0.25 ≲ effort_limit` (hip/thigh≤112…) **그리고** `ω_n·physics_dt ≲ 1`.
둘 중 작은 쪽이 Kp 천장. 초과 시 Kp↓ 또는 action_scale·effort·physics_rate 조정.

**Step 5 — Gain Tuner로 검증.**
Isaac Sim Gain Tuner extension의 step/sinusoidal 궤적으로 관절별 추종 확인.
Natural-Frequency 모드면 ω_n·ζ 입력 → Kp/Kd 자동 산출.

**Step 6 — drop test / step response.**
home pose로 떨어뜨려 자세 유지·진동·튐 확인. step 입력으로 overshoot·정착시간 측정.
overshoot/진동 → Kd↑ (또는 Kp↓). sluggish → Kp↑.

**Step 7 — RL sweep.**
`Kp ∈ {0.5×,1×,2×} × Kd ∈ {0.5×,1×,2×}` grid로 짧은 학습 후
추종 보상·진동/action-rate 페널티·접촉 품질 비교. action_scale도 함께(Kp×action_scale=실효 권한).

**Step 8 — domain randomization.**
최종 게인에 ±수% DR로 실 모터 편차 강인화(현재 cfg는 이미 ×0.75–1.5 / ×0.3–3.0 적용 중).

### 4.1 → §6 참조

I_eff를 실측했으므로 가정 기반 bracket을 폐기하고 **§6의 계산 기반 표**로 대체한다.

---

## 6. 실측 I_eff 기반 게인 (확정 계산값)

> 방법: kit 앱으로 pxr만 확보 → `hind_leg.usd`를 stage로 열어 link별 mass·diagonalInertia·
> CoM·principalAxes·world transform + revolute joint 축·위치를 읽고, **각 관절축 기준
> distal subtree의 합성관성**(composite rigid body + 평행축 정리, = mass matrix 대각 H[j,j])을
> USD rest pose에서 해석적으로 계산. 스크립트: `_workspace/compute_hind_leg_ieff.py`.
> (GPU 경합으로 physics-step 기반 mass-matrix 경로는 행 → USD 파싱 경로로 우회.)

### 6.1 실측 link 질량 + 관절별 I_eff

총 질량 **7.528 kg** (11 link). L/R 거의 대칭(asset 대칭 확인).

| 관절 | I_eff (kg·m²) | distal subtree | 비고 |
|---|---|---|---|
| hip (X, 외전/roll) | **0.163** | 다리 전체(hip+thigh+calf+foot) | 최대 |
| thigh (Y, pitch) | **0.133** | thigh+calf+foot | |
| calf (Y, pitch) | **0.030** | calf+foot | |
| foot (Y, pitch) | **0.00194** | foot+contact | 최소(hip의 1/84) |

주요 link 질량: base 1.43, thigh **1.64**(최대), hip 0.57/0.69, calf 0.53~0.55, foot 0.20.

### 6.2 ★ 권장 게인 — inertia-scaled (consistent ω_n 설계)

설계 원칙: **모든 관절이 같은 응답 대역 ω_n·감쇠비 ζ를 갖도록 Kp/Kd를 I_eff에 비례**시킨다.
`Kp = ω_n²·I_eff`, `Kd = 2ζ·ω_n·I_eff`. 목표 **ω_n=20 rad/s(≈3.2Hz), ζ=0.9**.

| 관절 | I_eff | **Kp** | **Kd** | full-action 토크(Kp·0.25) | effort 한계 | dt·ω_n (A 40Hz / B 200Hz) |
|---|---|---|---|---|---|---|
| hip   | 0.163 | **65** | **6.0** | 16.3 N·m | 28 ✓ | 0.50 / 0.10 |
| thigh | 0.133 | **53** | **4.8** | 13.3 N·m | 28 ✓ | 0.50 / 0.10 |
| calf  | 0.030 | **12** | **1.1** | 3.0 N·m | 42 ✓ | 0.50 / 0.10 |
| foot  | 0.0019| **8~12ⓐ / 15~25ⓑ** | **0.5~1.0** | ~5 N·m | 56 ✓ | ⓐ~1.1 / ⓑ~2.5(A 불안정) |

**foot은 특수 케이스 — inertia만으로 못 정한다(정직하게 조건부):**
- inertia-scaling만 따르면 Kp≈0.8인데, 이는 너무 약해 **지면 반력 토크**에 못 버틴다.
  ankle/foot은 자기 관성이 아니라 contact에 버티는 게 핵심(Cassie toe 20~40, H1 ankle 20).
- 그런데 Kp를 20으로 올리면 free-swing 관성(0.0019)에선 ω_n≈101 → **40Hz 변형(A)에서
  dt·ω_n≈2.5로 불안정.** 200Hz 변형(B)에선 dt·ω_n≈0.5로 OK.
- **권장**: ⓐ **변형 A(40Hz)면 foot Kp 8~12**, ⓑ **변형 B(200Hz)면 15~25.**
  Kd 0.5~1.0. foot은 contact-state 의존이라 **반드시 step/contact test로 검증**.
  (stance에서 발이 지면에 잡히면 실효관성↑로 ω_n이 떨어져 더 안정해진다.)

- **ω_n=20 설계의 핵심 이점**: 모든 관절이 ω_n=20 → `dt·ω_n`이 변형 A(40Hz, dt=0.025)에서도
  0.5로 안정. 현재 균일 Kp가 foot을 ω_n=113(dt·ω_n=2.8, A에서 발산권)으로 만든 문제를 해소.
- effort 한계(의도된 distal-strong: foot 56>calf 42>hip/thigh 28) 모두 만족.
- 더 민첩하게: ω_n=25면 Kp=hip 102/thigh 84… 단 hip 토크 25.5≈effort 28로 saturation 임박 →
  **ω_n 18~20을 1차 권장**, 25는 상한.

### 6.3 현재 균일 게인(25/0.5)의 문제 — 실측으로 확정

I_eff 대입 시 관절별 응답이 제각각 (균일 게인의 근본 한계):

| 관절 | 현재 ω_n=√(25/I) | 현재 ζ=0.5/(2√(25·I)) | 진단 |
|---|---|---|---|
| hip   | 12.4 rad/s | **0.12** | proximal 심한 under-damped → 진동 |
| thigh | 13.7 | **0.14** | 〃 |
| calf  | 28.9 | 0.29 | under-damped |
| foot  | **113** | 1.14 | 과도 stiff + over-damped |

→ proximal(hip/thigh)은 Kp 2배 이상·Kd 약 10배 올려야 하고, foot은 Kp 낮추고 Kd 올려야 함.
**균일 게인을 관절 그룹별 차등 게인으로 바꾸는 것이 1순위.**

### 6.4 적용 방법 (cfg 변경 — 아직 미적용, 검증 후 worker에게 위임)

`HIND_LEG_CFG`의 `ImplicitActuatorCfg`를 dict 형태 그룹별 게인으로 (foot은 사용 중인 env의
physics-dt에 맞춰; 아래는 **변형 A 40Hz** 기준, B면 foot Kp 15~25):
```python
stiffness={".*_hip_joint": 65, ".*_thigh_joint": 53, ".*_calf_joint": 12, ".*_foot_joint": 10},
damping  ={".*_hip_joint": 6.0, ".*_thigh_joint": 4.8, ".*_calf_joint": 1.1, ".*_foot_joint": 0.8},
```
(`source/isaaclab_assets/isaaclab_assets/robots/rga.py:467` actuators 블록. 현재 DR이
stiffness ×0.75~1.5 / damping ×0.3~3.0이라 학습 중 실효 게인은 이보다 넓게 흔들린다 —
명목값 확정 후 DR 범위도 재검토 권장.)

### 6.5 한계·검증 (정직한 caveat)

1. **rest pose 기준**: I_eff는 전관절 0 자세에서 계산. 다리 굽힘(stance)에서 thigh/calf I_eff는
   ±~30% 변함(특히 hip roll축은 자세 민감). → 게인은 nominal, ±30% 튜닝 여지.
2. **armature 미모델**: USD/cfg에 armature 없음. 실모터가 기어드면 반사관성 `J·G²`가 더해져
   특히 distal(foot) I_eff↑ → foot Kp가 자연히 올라감. 기어비 알면 armature 추가 권장.
3. **foot Kp는 contact 기준**(공학적 판단) — step/contact test로 검증.
4. **이 값은 계산 기반 "출발점"** — §4 Step 5~8(Gain Tuner·drop·step response·RL sweep)로
   반드시 검증 후 확정. action_scale(0.25)도 Kp와 함께 sweep.
5. ω_n=20, ζ=0.9는 보행용 1차 선택. 더 부드러운 접촉 원하면 ζ=1.0, 더 민첩하면 ω_n↑(상한 주의).

---

## 부록 — 출처

**로컬 코드(검증됨, 절대경로)**
- `source/isaaclab_assets/isaaclab_assets/robots/rga.py:436` — HIND_LEG_CFG
- `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env_cfg.py`
- `source/isaaclab_assets/isaaclab_assets/robots/{unitree,anymal,cassie}.py` — 레퍼런스 게인

**외부 표준 참고문헌(이번 세션 미-fetch, 표준 지식으로 인용)**
- NVIDIA Omni Physics — Articulation & Robot Simulation Stability Guide (Kp/Kd↔ω_n/ζ, armature=JG², dt·ω_n)
- Isaac Sim Gain Tuner Extension (Direct / Natural-Frequency 모드)
- IsaacLab Discussion #4121 (DCMotor 비례제어 함정)
- legged_gym / unitree_rl_gym (A1·Cassie·G1 게인 컨벤션)
- Unitree GO-M8010-6 모터 스펙 (23.7 N·m, 30 rad/s, 1:6.33)

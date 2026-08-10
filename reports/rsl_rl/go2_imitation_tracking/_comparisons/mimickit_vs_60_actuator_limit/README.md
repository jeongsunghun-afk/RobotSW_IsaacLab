# MimicKit 은 4 m/s 를 내는데 6.0 은 1.5 m/s — 원인은 **액추에이터 모델**

**질문**: MimicKit(`amp_adaptation_module_20260512_160750`)은 DR 을 켠 채로도 `cmd 4.0` 까지
추종한다. 같은 Go2, 같은 AMP 계열, 심지어 **같은 Isaac Lab 엔진**인데 6.0 은 왜 못 하나?

**답**: 6.0 GO2 는 `DCMotorCfg(effort 23.5, saturation 23.5, velocity_limit 30)` 이고
MimicKit 은 `ImplicitActuatorCfg(effort_limit=None)` + MJCF `actuatorfrcrange`
(hip/thigh 23.7, **calf 35.5**) 다. **속도-토크 곡선이 없고 calf 토크가 51% 크다.**

## 비교 대상

| | MimicKit | IsaacLab 6.0 |
|---|---|---|
| 결과 | `results/amp_adaptation_module_20260512_160750` | `clip10_scratch/model_16000` |
| 엔진 | `data/engines/isaac_lab_engine.yaml` (`engine_name: isaac_lab`) | IsaacLab 6.0 |
| 로봇 | `data/assets/go2/go2.xml` (MJCF) | `go2.usd` + `UNITREE_GO2_CFG` |
| 최고 속도 | **4.0 m/s** (명령 추종) | **1.527 m/s** |

MimicKit 학습 cfg 사본: `/home/lgb/MimicKit/logs/amp_adaptation_module_20260512_160750/`
(`env_config.yaml`, `agent_config.yaml`, `engine_config.yaml`).

## 1. 액추에이터 — 결정적 차이

| | MimicKit | 6.0 |
|---|---|---|
| actuator cfg | `ImplicitActuatorCfg(stiffness=25, damping=1.0, **effort_limit=None**)`<br>(`mimickit/engines/isaac_lab_engine.py:803`) | `DCMotorCfg(effort_limit=23.5, saturation_effort=23.5, velocity_limit=30, stiffness=25, damping=0.5)`<br>(`isaaclab_assets/robots/unitree.py:172-180`) |
| 실효 토크 한계 | **USD `maxForce` 그대로**: hip/thigh **23.7**, calf **35.5** | 전 관절 **23.5** (USD 값을 덮어씀) |
| 속도-토크 곡선 | **없음** — 속도와 무관하게 상수 | **있음**: `τ_max = 23.5 · (1 − q̇/30)` |

`effort_limit=None` 은 "USD drive 값을 덮어쓰지 않는다" 는 뜻이므로 **무제한이 아니다.**
`data/assets/go2/go2.usd` 의 `UsdPhysics.DriveAPI.maxForce` 를 직접 읽어 확인했다:

```
FL/FR/RL/RR_hip_joint    maxForce = 23.7
FL/FR/RL/RR_thigh_joint  maxForce = 23.7
FL/FR/RL/RR_calf_joint   maxForce = 35.5     ← 6.0 은 여기를 23.5 로 자른다
```

(`_parse_usd_path` 가 `go2.xml` → `go2.usd` 로 바꾸므로 MJCF 가 아니라 이 USD 가 쓰인다.)
`joints.png` 실측 calf 토크 30~35 N·m 가 이 35.5 한계와 일치한다.

### 두 메커니즘은 분리해서 봐야 한다

**지금 6.0 을 막는 것은 평평한 23.5 이지 속도-토크 곡선이 아니다.** 6.0 의 실측
`|q̇|p95` 는 5~9 rad/s 라 곡선에 의한 한계 감소가 **8.4%** 에 그친다
([`metrics/`](metrics/), `logs/torque_sat.py`). 곡선은 **한계를 올린 뒤에 다음으로 걸리는 제약**이다.

| | MimicKit | 6.0 |
|---|---:|---:|
| 현재 작동점(`q̇` 5~9)에서 calf 유효 한계 | 35.5 | **≈ 21.5** |
| MimicKit 작동점(`q̇` 15~20)에서 calf 유효 한계 | 35.5 | 7.8 ~ 11.8 |

앞 행이 **오늘 측정된 병목**이고, 뒷 행은 "6.0 이 MimicKit 처럼 달리려 할 때" 의 반사실이다.
`clip10` 은 여기에 PACE viscous(2.4)까지 더 깎인다.

### ★★ 참조모션이 6.0 플랜트에서 물리적으로 도달 불가능하다

기록된 참조모션 실측 관절 속도 요구치는 walk 4.3~5.4, trot 6.65, **run 7.85 / 8.30 / 12.09 rad/s**
(memory `project_go2_tracking_pace_dr_and_ppolegacy_fix`). 6.0 은 토크가 포화한 상태에서
**5~9 rad/s** 까지밖에 못 간다. 즉 **run 클립은 6.0 플랜트에서 재현 불가능하다** —
AMP discriminator 가 아무리 오래 학습해도 그 gait 를 만들어낼 수 없다. 학습량·하이퍼파라미터
문제가 아니다.

## 2. 실측 — MimicKit 은 그 여유를 실제로 쓴다

`results/.../joints.png` 에서:

- **calf 토크가 30~35 N·m** 까지 올라간다(RR_calf 는 35 근처에서 평탄 = 포화).
- **관절 속도가 ±15~20 rad/s**.

즉 4 m/s 주행은 "관절을 15~20 rad/s 로 돌리면서 calf 에 30+ N·m" 를 요구하는데, 6.0 에서는
그 속도대의 유효 토크가 7.8~11.8 N·m 라 **애초에 불가능하다.**

## 3. 실측 — 6.0 쪽은 이미 한계에 붙어 있다

각 정책이 **실제로 가장 빠르게 달린 구간**의 관절별 토크
([`metrics/torque_by_joint_peak.txt`](metrics/torque_by_joint_peak.txt),
[`logs/torque_by_joint.py`](logs/torque_by_joint.py)):

| joint | 6.0 `clip10@16k` (vx 1.527) | 5.1 @50k (vx 2.058) |
|---|---:|---:|
| FL_thigh | 21.28 / max **23.50** (105%) | 20.96 / max **23.42** (105%) |
| FR_thigh | 23.40 / max **23.50** (114%) | 21.29 / max **23.50** (107%) |
| FL_calf | 16.43 / 19.06 (78%) | 23.44 / max **23.50** (118%) |
| FR_calf | 9.54 / 12.49 (45%) | 23.50 / **23.50** (115%) |
| RL_calf | **23.50 / 23.50** (107%) | 23.38 / **23.50** (107%) |
| RR_calf | **23.50 / 23.50** (107%) | 21.53 / 23.15 (100%) |

사용률 = `|τ|p95 / mean(속도의존 유효한계)`. **p95 를 평균과 나눈 값이라 100% 초과가
"한계를 넘었다"는 뜻은 아니다** — 주장을 지탱하는 것은 `|τ|max = 23.50` 이 여러 관절에서
**정확히** 찍힌다는 사실이다(클램프에 닿았다는 직접 증거).

**두 정책 모두 thigh·calf 가 23.5 에 붙는다.** 6.0 환경에서는 **5.1 정책조차** 토크를 다 쓰고
있고, 그 상태의 상한이 2.058 m/s 다.

> ⚠ 측정 함정: "명령이 가장 큰 stage"(cmd 4.0)를 고르면 **둘 다 이미 넘어져 정지한 구간**이
> 잡혀 `|q̇| ≈ 0.02~0.5`, 포화율 0% 가 나온다. 반드시 **실측 vx 가 최대인 stage** 를 골라야 한다.

## 4. ★★ **IsaacLab 6.0 자신의 Go2 USD 도 calf 는 45.43 이다 — cfg 가 23.5 로 덮고 있다**

`UNITREE_GO2_CFG` 가 실제로 로드하는 `Go2_noninstanceable/go2.usd` 의 drive `maxForce` 를
직접 읽으면:

```
FL/FR/RL/RR_hip_joint    maxForce = 23.7
FL/FR/RL/RR_thigh_joint  maxForce = 23.7
FL/FR/RL/RR_calf_joint   maxForce = 45.43     ← 자산이 정의한 값
```

(`Go2Neck2/go2_neck.usd` 도 동일하게 calf 45.43 이다.)

그런데 `unitree.py:172-180` 의 `DCMotorCfg` 는 `joint_names_expr` 를 hip·thigh·calf **전부**로
잡고 `effort_limit=23.5` 단일값을 준다. **즉 자산이 45.43 이라고 말하는 calf 를 cfg 가 23.5 로
48% 깎는다.** 정황도 일치한다:

1. **Unitree 공식 사양은 관절 모터 peak torque 45 N·m** — USD 의 45.43 과 맞는다.
2. **같은 파일의 A1 이 33.5 다**(`unitree.py:91`). Go2 는 A1 의 후속·상위 모델인데 23.5 로 더 낮다.
3. `unitree.py:42` 의 23.7 주석이 `# taken from spec sheet` — 23.7 은 **hip/thigh 스펙**이고,
   6.0 은 그것을 calf 에도 그대로 적용한 것으로 보인다.

세 자산(6.0 USD 45.43 / MimicKit USD 35.5 / MJCF 35.5)이 전부 **calf > hip·thigh** 라고 말하는데
6.0 cfg 만 전 관절 동일값이다.

**여전히 미확인**: 23.5 가 "연속 정격"이고 35.5/45.43 이 "peak" 일 수 있다. 그렇다면 6.0 이
보수적인 것이지 틀린 것은 아니다. 다만 **관절별 차이를 없앤 것**은 그 논리로도 설명되지 않는다
(연속 정격이라면 calf 쪽이 더 커야 한다). 어느 쪽을 쓸지는 sim2real 목표에 달린 판단이다.

## 5. 부차 요인 (같이 다르지만 이만큼 크지 않다)

| | MimicKit | 6.0 |
|---|---|---|
| DR | 마찰 0.4~1.2 / base mass −0.5~+5.0 kg / CoM / PD gain 스케일 | 위 + **push_robot(5 s 마다 1 m/s 킥)**, action delay, obs noise, encoder bias |
| reward | AMP disc **+ imitation reward 직접**(pose 0.5, vel 0.1, root_pose 0.15, root_vel 0.1, key_pos 0.15) | AMP disc + task reward만 |
| 속도 추종 민감도 | `reward_tracking_lin_vel_scale = **1.0**` | `vel_err_scale = **0.5**` (2 배 둔감) |
| episode | 10 s | 20 s |
| yaw 명령 | ±1.0 | ±1.5 |
| action std | `FIXED 0.1` + `action_bound_weight 10.0` | 학습 std (41k 에서 0.116) + `clip_actions 10` |
| kd | 1.0 | 0.5 |

DR 쪽은 방향이 명확하다 — **6.0 의 `push_robot` 은 고속에서 특히 불리하다**(램프 stage 4.5 s 에
5 s 주기 킥). reward 쪽은 `vel_err_scale` 이 2 배 둔감해 고속 학습 압력이 약하다
([`../lin_vel_ceiling_51_vs_60/`](../lin_vel_ceiling_51_vs_60/)).

## 6. 미확인 사항

- **MimicKit `velocity_comparison.png` 를 만든 평가 실행의 엔진을 확인하지 못했다.**
  학습이 `isaac_lab` 이었다는 것은 로그 디렉터리의 `engine_config.yaml` 로 확인했지만
  (`engine_name: "isaac_lab"`), 결과 PNG 를 만든 `run_eval.py` 호출은 기록이 남아 있지 않다
  (`run_eval.py:473` 이 `engine_config` 를 인자로 받는다). `env_config.yaml` 에
  `ground_contact_height: 0.15 # this is needed for IsaacGym` 주석이 남아 있어 Isaac Gym 실행
  이력도 있다. **"같은 엔진에서 4 m/s" 라는 문장은 학습 기준으로만 확인된 것이다.**
- `joints.png` 의 30~35 N·m 는 렌더된 PNG 를 눈으로 읽은 값이다(원 데이터 없음).
  단 USD `maxForce = 35.5` 를 직접 읽어 확인했으므로 상한 자체는 확정이다.
- 참조 데이터셋 비교 미실시: MimicKit `dataset_go2_locomotion_smr_mirror3.yaml` vs
  6.0 `smr_mirror_pkl` — 두 세트가 담은 속도 분포가 다를 수 있다.

## 처방 후보 (미실행)

1. **`effort_limit`/`saturation_effort` 를 관절별로 분리** — calf 를 USD 값(45.43) 또는
   MimicKit 값(35.5)으로. **이것이 측정으로 뒷받침되는 유일한 처방이다**(현재 병목 = 평평한 23.5).
2. `velocity_limit` 상향 또는 `DCMotorCfg` → `ImplicitActuatorCfg`. **1 을 적용한 뒤에야 의미가
   있다** — 지금 작동점(`q̇` 5~9)에서 곡선의 기여는 8.4% 뿐이다. 그리고 DCMotor 곡선은 실기
   모터의 실제 특성이므로 없애면 sim2real gap 이 커진다.
3. `vel_err_scale` 0.5 → 1.0 (MimicKit 과 동일). 고속 학습 압력을 키운다
   ([`../lin_vel_ceiling_51_vs_60/`](../lin_vel_ceiling_51_vs_60/)).
4. 램프/평가 시 `push_robot` 제외 여부 재검토.

★ 1·2 는 **로봇 능력 자체를 바꾸는 변경**이라 기존 모든 런과 비교 불가능해진다.
바꾸려면 from-scratch A/B 가 필요하다.

## 7. 적용 (2026-08-05) — calf 분리 + 공식 URDF `velocity_limit`

처방 1 을 적용하고 from-scratch 2 런을 시작했다. 변경은
`go2_imitation_tracking_env_cfg.py` 안에서만 이뤄져 **다른 Go2 task 는 영향받지 않는다.**

| 그룹 | 관절 | `effort_limit` | `saturation_effort` | `velocity_limit` |
|---|---|---:|---:|---:|
| `base_legs` | hip 4 + thigh 4 | 23.5 | 23.5 | **30.1** |
| `calf` | calf 4 | **35.5** | **35.5** | **15.70** |

`velocity_limit` 은 Unitree 공식 URDF
([`unitree_ros/robots/go2_description/urdf/go2_description.urdf`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/go2_description/urdf/go2_description.urdf))
의 `<limit>` 값이다 — hip/thigh `effort 23.7 / velocity 30.1`, calf `effort 45.43 / velocity 15.70`.

### 왜 그룹을 나눴나

`saturation_effort` 는 dict 를 못 받는다(`actuator_pd_cfg.py:50` 이 `float = MISSING`).
그룹을 유지한 채 `effort_limit` 만 관절별 dict 로 주면
`τ_max = clip(saturation·(1 − q̇/v), −∞, effort_limit)` 에서 **saturation 이 공유**되므로
hip·thigh 의 토크-속도 곡선까지 완만해진다(q̇=8 에서 17.2 → 23.5, **+37%**).
calf 만 바꾸는 변경이 아니게 된다.

⚠ 그룹 분리의 부작용: `go2_imitation_tracking_env.py:594` 의
`self._act = self._robot.actuators["base_legs"]` 하드코딩이 **calf 의 kp/kd DR 을 놓친다.**
`_acts` 리스트로 바꿔 두 그룹을 모두 스케일하도록 함께 고쳤다.

### ★ calf `velocity_limit` 15.7 은 양날이다

URDF 의 calf 는 토크가 큰 대신 **속도 한계가 절반**이다(기어비). 곡선이 가팔라져
저속에서 강해지고 고속에서 약해진다:

| calf `q̇` [rad/s] | 기존 (23.5 / 30) | 새 (35.5 / 15.7) |
|---:|---:|---:|
| 5 | 19.6 | **24.2** |
| 8 | 17.2 | 17.4 |
| 10 | 15.7 | **12.9** |
| 12 | 14.1 | **8.4** |

실측 calf `|q̇|p95` 가 4.5~9.6 이라 대체로 이득이지만 **고속 대역은 오히려 불리**하다.
즉 이 변경은 "강화" 가 아니라 **실제 로봇 특성에 맞추는 것**이다. 속도가 오히려 떨어질 수도
있으며, 그 경우에도 sim2real 정확도 쪽은 개선된 것으로 읽어야 한다.

### 시작한 런

| run | `use_pace_params` | GPU | 비고 |
|---|---|---|---|
| `calf355_pace` | **true** (기본) | 3 | |
| `calf355_stock` | **false** | 2 | |

둘 다 4096 env / 60000 iter / `calf 35.5 · 15.7` 공유 → **`use_pace_params` 단일변수 A/B**.
액추에이터 표에 `FL_calf_joint … Velocity Limits 15.700` 이 찍혀 sim 반영을 확인했다.

중단한 이전 런: `clip10_scratch` **49.3k**, `stockplant_scratch` **20.5k** (체크포인트 보존,
램프 측정 가능).

## 8. 결과 (진행 중) — **6.0 최고 기록 갱신**, 단 고속은 예측대로 나빠졌다

램프 워치 산출: [`metrics/ramp_progress.csv`](metrics/ramp_progress.csv),
[`metrics/ramp_16k_matched.txt`](metrics/ramp_16k_matched.txt), 원자료 [`metrics/ramp/`](metrics/ramp/).

| iter | `calf355_pace` | `calf355_stock` |
|---:|---|---|
| 8000 | **1.751** (70%, @3.0) | 1.252 (50%, @2.5) |
| 16000 | 1.684 (91%, @2.5) | 1.269 (61%, @2.5) |
| 24000 | 1.688 (91%, @2.5) | 1.323 (80%, @2.5) |
| 32000 | 1.498 (**55%**, @2.5) ↓ | 1.339 (88%, @2.5) |

**이전 6.0 최고는 `clip10@16k` 의 1.527 이었다.** `calf355_pace` 는 8k 에 **1.751**(+14.7%),
안정적인 값(달성률 91%)으로 봐도 16k/24k 의 **1.684~1.688**(+10.3%)로 그것을 넘는다.
5.1 의 2.058 대비로는 74% → **85%** 로 좁혀졌다.

### ★ 예측대로 `cmd 3.0` 은 오히려 나빠졌다

16k matched 로 갈라 보면:

| cmd | `calf355_pace@16k` | `clip10@16k` (구, calf 23.5/30) |
|---:|---|---|
| 1.00 | **0.782 (88%)** | 0.684 (69%) |
| 2.00 | **1.449 (94%)** | 1.346 (97%) |
| 2.50 | **1.684 (91%)** | 1.527 (97%) |
| **3.00** | 0.831 (34%) | **1.494 (50%)** |

**중속(1.0~2.5)은 전부 개선인데 `cmd 3.0` 에서 역전된다**(1.684 → 0.831 붕괴).
7 절에서 예고한 calf `velocity_limit` 15.7 의 효과가 그대로 나타났다 — 저속·중속에서
토크가 커지고(q̇=5: 19.6 → 24.2) 고속에서 작아진다(q̇=12: 14.1 → 8.4).

`pace@8k` 가 `cmd 3.0` 에서 1.751 을 냈다가 16k 에 0.831 로 떨어진 것도 같은 방향이다 —
학습이 진행되며 **고속을 포기하고 중속 안정성**(달성률 70% → 91%)을 택했다.

### PACE vs 스톡 격차가 더 벌어졌다

| | 구 세대 (calf 23.5) | 신 세대 (calf 35.5/15.7) |
|---|---:|---:|
| PACE | 1.442 (@8k) | **1.688** (@24k) |
| 스톡 | 1.340 (@8k) | 1.339 (@32k) |
| 격차 | +7.6% | **+26.1%** |

**스톡 플랜트는 calf 변경의 이득을 거의 못 받았다**(1.340 → 1.339). 즉 이번 개선은
PACE 플랜트와의 상호작용에서 나온다. `use_pace_params` 단일변수 A/B 이므로 해석은 명확하다.

### 32k 하락 — 붕괴는 아니지만 추세를 봐야 한다

`pace` 가 24k 1.688(91%) → 32k **1.498(55%)** 로 내렸다. 구간별로 보면 **저속만 오르고
중·고속이 전부 내렸다**:

| cmd | `pace@24k` | `pace@32k` |
|---:|---|---|
| 0.50 | 0.006 (9%) | **0.237 (45%)** |
| 1.50 | 1.132 (89%) | 1.123 (78%) |
| 2.00 | 1.425 (89%) | 1.372 (73%) |
| 2.50 | **1.688 (91%)** | 1.498 (**55%**) |
| 3.00 | 0.809 (39%) | 0.380 (14%) |

**`newpace` 의 52k 붕괴와는 다르다** — 그건 top 0.858 → 0.015 로 완전히 죽은 것이었고,
지금은 −11% 하락에 이전 세대 최고(1.527)와 비슷한 수준이다. 그리고 이 저속↔중속 트레이드오프
진동은 `clip10` 에서도 관찰됐다(32k 에 저속 0.010/28% 로 죽었다가 40k 에 0.316/59% 로 회복되며
top 도 1.459 → 1.498 로 되돌아왔다).

⚠ 아직 진행 중이다. 40k 에서 회복하는지가 판단 지점이고, `newpace` 전례 때문에 48k/56k 까지
봐야 확정이다.

---

## 9. 적용 (2026-08-10) — DCMotor → **ImplicitActuator** (토크-속도 곡선 제거)

DCMotor 계보(23.5 → calf 35.5 → `velocity_limit` 30.0)는 **`cmd 3.5` 를 단 한 번도 넘지
못했다** — 두 세대 × 7 개 체크포인트 × 양 플랜트에서 예외 없이 0%. 그래서 곡선 자체를 뺐다.

```python
"base_legs": ImplicitActuatorCfg(
    joint_names_expr=[".*"],
    effort_limit=None,   # USD drive maxForce: hip/thigh 23.70, calf 45.43
    stiffness=25.0, damping=0.5, friction=0.0, armature=0.01,
)
```

근거는 `_comparisons/go2_highspeed_literature/` 다. 요점만:

- `DCMotorCfg.velocity_limit` 은 **무부하 속도(x 절편)** 인데 넣어 둔 30.1/30.0 은 공식 URDF 의
  **정격 최대 속도**였다. "정격 속도에서 토크 0" 이라는 과도하게 가파른 곡선이 된다.
- 데이터시트의 "30 rad/s" 는 **24 V** 기준이고 Go2 배터리는 **29.6 V 정격**이다.
- `saturation_effort == effort_limit` 이라 실제 PMSM 이 갖는 **정전류 평탄 구간이 없다**.
- 선행연구: MoE(arXiv:2602.00678)는 토크-속도 모델 **없이** 실기 4.01 m/s, MimicKit 도
  `ImplicitActuator(effort_limit=None)`.

**바뀐 것은 액추에이터 모델 하나다** — yaw ±1.0 · `joint_vel_noise` 0.15 는 직전 세대와 같다.

런: `implicit_pace` / `implicit_stock`(`use_pace_params=false`), 4096 env, 60k, 각각 GPU3/GPU1.

### 결과 (16k/60k, **진행 중**)

수치는 `metrics/implicit_vs_dcmotor_ramp.md`, 그림은 `figures/dcmotor_vs_implicit_ramp.png`.

| cmd | stock DC@16k | stock **IMP@16k** | pace DC@16k | pace **IMP@8k** |
|---:|---:|---:|---:|---:|
| 0.5 | 0.376 (67%) | 0.371 (**83%**) | 0.381 (97%) | 0.348 (92%) |
| 1.5 | 1.061 (100%) | 1.102 (94%) | 1.110 (97%) | 1.020 (**100%**) |
| 2.5 | **1.554 (95%)** | 1.524 (**56%**) | **1.452 (97%)** | 1.336 (81%) |
| 3.0 | 0.743 (23%) | **1.347 (38%)** | 1.213 (20%) | 0.886 (3%) |
| **3.5** | 0.147 (**0%**) | 0.194 (**0%**) | 0.140 (**0%**) | 0.132 (**0%**) |

**판정 기준(`cmd 3.5` 가 0 을 벗어나는가)은 아직 미달이다.** 그 외에는 두 점(stock 8k·16k,
pace 8k)뿐이고 달성률 재현 변동이 ±8~15%p 라 **추세로 서술하지 않는다** —
직전 세대에서 이 실수를 다섯 번 했다.

### ★ 포화가 곡선에서 **평탄 캡**으로 옮겨갔다

`stock IMP@16k` 의 최고 구간(cmd 2.5 → 1.524 m/s, 달린 env 37/64)에서:

| joint | 하드캡 | \|τ\|p95 | 캡도달% | \|q̇\|p95 |
|---|---:|---:|---:|---:|
| thigh ×4 | 23.7 | **23.70 (4개 전부)** | 9.8~14.8% | 5.7~7.7 |
| calf RL/RR | 45.4 | 34.9 / 35.9 | 2.5% | 4.5 |
| calf FL/FR | 45.4 | 16.2 / 15.1 | ~0% | 6.2~6.6 |
| hip ×4 | 23.7 | 8.6~10.6 | 0% | 1.3~2.6 |

**네 thigh 가 전부 p95 에서 캡에 붙어 있다.** 곡선을 없앤 대가로 이제 벽은 평탄한
`thigh maxForce = 23.7 N·m` 이고, 이 값은 Unitree 공식 peak 라 **올릴 근거가 없다**.

⚠ 미리 경고해 둔 calf USD 하드 클립 **15.70 rad/s 는 아직 안 걸린다**(`|q̇|p95` 4.5~6.6).
지금 벽은 USD 속도 클립이 아니라 thigh 토크다.

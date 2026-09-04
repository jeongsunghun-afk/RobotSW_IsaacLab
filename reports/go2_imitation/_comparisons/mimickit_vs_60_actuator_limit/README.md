# MimicKit 은 4 m/s 를 내는데 6.0 은 1.5 m/s — 원인은 **액추에이터 모델**

> 작성 일시: 2026-08-05 17:37 (소급 · 근거: 디렉터리 내 최초 산출물 mtime)
> 비교 대상: MimicKit(`amp_adaptation_module_20260512_160750`) vs IsaacLab 6.0(`clip10_scratch/model_16000`) — 액추에이터 모델 차이

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

DCMotor 계보(23.5 → calf 35.5 → `velocity_limit` 30.0)는 **`cmd 3.5` 를 사실상 넘지 못했다** —
16k 이후로는 양 플랜트 전부 0% 이고, 0 이 아닌 유일한 점이 `stock@8k` 의 8%(중앙값 0.229)다.
그래서 곡선 자체를 뺐다.

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

### 결과 — 전 궤적 (stock 56k, pace 48k)

원자료 `metrics/implicit_full_trajectory.md`, 그림 `figures/implicit_full_trajectory.png`,
생성 `logs/implicit_full_trajectory.py`.

#### ★★★ 결론: 액추에이터 모델 효과가 **플랜트마다 반대**다

| | stock | pace |
|---|---|---|
| `cmd 3.0` 달성률 Δ (16k→56k) | **+14 / +16 / +22 / +28 / +19 / +30** | −39 / −42 / −38 / −36 (24k→48k) |
| IMP 최고 | **1.686 @ cmd 3.0 (58%) @48k** | 1.497 @ cmd 3.0 (47%) @48k |
| DC 최고 (같은 구간) | 1.553 @ cmd 2.5 · 1.387 @ cmd 3.0 | **1.657 @ cmd 3.0 (83%)** |
| 판정 | **Implicit 이 확실히 낫다** | **DCMotor 가 확실히 낫다** |

stock 은 6 개 연속 체크포인트에서 `cmd 3.0` 이 앞서고 **처음으로 top 이 `cmd 2.5` 에서
`cmd 3.0` 으로 올라갔다**(48k, 1.686). pace 는 24k 이후 4 개 연속 −36~−42%p 로 뒤진다.

⚠ 과거에 8k 한 점으로 "플랜트별 부호 반전"을 주장했다가 철회한 적이 있다. 이번엔 각 방향이
**4~6 개 연속 체크포인트**로 지지되고 폭도 20~40%p 라 그때와 근거의 질이 다르다.

#### ★★★ `cmd 3.5` 는 **13 개 implicit 체크포인트 전부 0%**

stock 7 점 + pace 6 점, 예외 없음. **이 세대를 시작한 이유였던 판정 기준은 명확히 미달**이고,
곡선을 통째로 제거해도 이 선은 움직이지 않는다. → 원 가설("토크-속도 곡선이 `cmd 3.5` 를
막는다")은 **반증됐다.** 남은 벽은 아래의 평탄한 thigh 캡이다.

#### ⚠⚠ 8k~24k 로 내린 판정은 전부 틀렸다 — 철회

`cmd 2.5` 달성률 Δ (stock): **−27 / −39 / −41** → **−14 / +2 / +0 / +6**.

2026-08-10 에 "6 개 매치드 점 전부 음수 → stock 확정 · 중앙값까지 벌어짐" 이라고 적었는데,
그건 **학습 중반 일시적 현상**이었다. 40k 이후 stock IMP 는 `cmd 2.5` 에서 DC 와 동률
(97% vs 97%/91%)이고 중앙값은 오히려 앞선다(1.598/1.569 vs 1.553/1.495).

★ 이번 실수는 앞선 다섯 번보다 **더 강한 근거를 갖고도** 틀렸다는 점이 중요하다 —
3 개 연속 점 · 두 플랜트 · 부호 6/6 일치 · 폭 20~41%p 였다. **이 task 에서는 24k 이하의
차이가 48k 이후를 예측하지 못한다.** 앞으로 이 세대군의 판정은 **40k 이후 점으로만** 한다.
(`cmd 2.5` 는 pace 에서만 전 구간 음수로 남았다: −5/−20/−22/−17/−3/−9.)

### ★ 포화가 곡선에서 **평탄 캡**으로 옮겨갔다

최고 체크포인트 48k, `cmd 3.0` 구간(stock 1.686 m/s · 달린 env 37/64, pace 1.497 · 31/64):

| joint | 하드캡 | stock \|τ\|p95 (캡%) | pace \|τ\|p95 (캡%) | \|q̇\|p95 |
|---|---:|---:|---:|---:|
| thigh 앞 ×2 | 23.7 | **23.70 (13.5·14.4%)** | 4.8 / 5.5 (0%) | 7.1~7.3 |
| thigh 뒤 ×2 | 23.7 | **23.70 (19.5·21.8%)** | **23.70 (12.3·16.5%)** | 5.7~6.6 |
| calf 뒤 ×2 | 45.4 | 29.8 / 32.5 (3~4%) | 40.1 / 41.2 (2~3%) | 4.0~5.4 |
| calf 앞 ×2 | 45.4 | 16.5 / 18.0 (0%) | 11.8 (0%) | 6.1~6.4 |
| hip ×4 | 23.7 | 9.2~10.2 (0%) | 4.8~9.3 (0%) | 1.3~2.8 |

**벽은 평탄한 `thigh maxForce = 23.7 N·m` 이고 포화가 더 깊어졌다**(stock 4 개 전부 캡 도달
13.5~21.8%). 이 값은 Unitree 공식 peak 라 **올릴 근거가 없다**.

★ **앞다리 thigh 사용이 두 플랜트에서 다르다** — stock 은 4 개 다 캡에 붙는데 pace 는 앞다리가
p95 4.8/5.5 로 논다. 더 빠른 쪽(stock 1.686 vs pace 1.497)이 네 다리를 다 쓰는 쪽이다.
pace 열세의 후보 설명이지만 **인과는 미검증**이다.

⚠ 미리 경고해 둔 calf USD 하드 클립 **15.70 rad/s 는 최고 체크포인트에서도 안 걸린다**
(`|q̇|p95` 4.0~6.4). 지금 벽은 USD 속도 클립이 아니라 thigh 토크다.

## 10. 신규 실험 2 건 (2026-08-11) — AMP 제외 · MimicKit sim 스택 정렬

### 10-a. `sim dt` / `decimation` 은 **이미 MimicKit 과 같다** (실측)

`/home/lgb/MimicKit/data/engines/isaac_lab_engine.yaml`:

```yaml
control_mode: "pos"
# control_freq: 30
# sim_freq: 120
control_freq: 50
sim_freq: 200
```

`isaac_lab_engine.py:68-75` 가 `_sim_steps = sim_freq / control_freq = 4` 로 쓰고,
`SimulationCfg(dt=1/200, render_interval=4)` 를 만든다. 우리 쪽
(`go2_imitation_tracking_env_cfg.py:160-163`) 은 `sim_dt_hz 200` / `policy_dt_hz 50` /
`decimation 4` — **완전히 동일**하다. 주석 처리된 `30/120` 은 폐기된 옛 설정이다.
→ **"dt/decimation 을 MimicKit 과 맞춘다"는 실험은 바꿀 것이 없다.**

### 10-b. 대신 실제로 다른 sim/플랜트 항목 (전부 소스 확인)

| 항목 | 6.0 (ours) | MimicKit | 출처 |
|---|---|---|---|
| sim / control freq | 200 / 50 (dec 4) | 200 / 50 (dec 4) | 동일 ✓ |
| actuator | `ImplicitActuatorCfg(effort_limit=None)` | `ImplicitActuatorCfg(effort_limit=None)` | 동일 ✓ (2026-08-10 이후) |
| kp / kd | 25.0 / **0.5** | 25.0 / **1.0** | `isaac_lab_engine.yaml` `pd_gains` |
| calf 토크 캡 | **45.43** (go2.usd) | **35.5** (`actuatorfrcrange`) | `data/assets/go2/go2.xml:60` |
| hip/thigh 캡 | 23.70 | 23.7 | 동일 ✓ |
| armature | 0.01 | 0.01 | 동일 ✓ |
| `enabled_self_collisions` | True | True (`create_obj` 기본값) | `isaac_lab_engine.py:183` |
| `angular_damping` | 0.0 | **0.01** | `isaac_lab_engine.py:941,976` |
| `max_depenetration_velocity` | 1.0 | **10.0** | 같은 곳 |
| `physx.bounce_threshold_velocity` | IsaacLab 기본값 | **0.2** | `isaac_lab_engine.py:770` |
| solver iter (pos/vel) | 4 / 0 (articulation) | 4 / 0 (physx 전역) | 실질 동일 |
| asset | `go2.usd` | `go2.xml` (MJCF) | 파일 자체가 다르다 |

★ 이 중 물리적으로 유의미한 후보는 **kd 0.5→1.0** 과 **calf 캡 45.43→35.5** 다.
`angular_damping`·`max_depenetration_velocity`·`bounce_threshold_velocity` 는 접촉 수치안정용
파라미터라 속도 천장에 직접 붙일 근거가 약하다. **묶어서 한 번에 바꾸지 않는다** — leg 런에서
4 개를 동시에 바꿔 분리 불가능해진 전례가 있다.

⚠ 확인 못 한 것: MJCF joint 의 `stiffness=15 damping=2` 가 IsaacLab MJCF 변환에서 어떻게
반영되는지, `actuatorfrcrange` 가 USD `drive maxForce` 로 매핑되는지.

### 10-c. AMP 제외 학습 — 시작함

`2026-08-11_11-14-27_noamp_stock` (GPU1, 60k). 러너에 AMP off 스위치가 없어
`task_reward_lerp = 1.0` 고정으로 style term 계수를 0 으로 만든다. 설정값(`params/agent.yaml`)과
런타임값(`Loss/amp_task_reward_lerp = 1.0 @ iter 0`) 둘 다 확인했다.
상세: [`../2026-08-11_11-14-27_noamp_stock/`](../../go2_imitation_tracking/2026-08-11_11-14-27_noamp_stock/)

### 10-d. 세 접촉 노브의 **기본값** (설치된 스키마에서 직접 읽음)

`.../isaacsim/extscache/omni.usd.schema.physx-110.1.13/plugins/PhysxSchema/resources/generatedSchema.usda`

| 속성 | 의미 | 스키마 기본 | 6.0 (Go2) | MimicKit |
|---|---|---:|---:|---:|
| `physxRigidBody:angularDamping` | 각속도 인공 감쇠 (수치안정용, 물리적 마찰 아님) | **0.05** | 0.0 (`unitree.py:150`) | 0.01 |
| `physxRigidBody:maxDepenetrationVelocity` | 관통 해소 속도 상한 [m/s] | **3.0** | 1.0 (`unitree.py:153`) | 10.0 |
| `physxScene:bounceThreshold` | 이 값 미만의 상대속도 접촉은 반발 계산 안 함 [m/s] | 0 (스키마) / **0.5** (IsaacLab 기본, `isaaclab_physx/physics/physx_manager_cfg.py:172`) | 0.5 (설정 안 함) | 0.2 |

`RigidBodyPropertiesCfg` 의 필드 기본값은 전부 `None` 이며, 이는 "USD 에 적힌 값을 건드리지
않는다"는 뜻이다. 즉 위 6.0 열의 값은 **우리가 명시적으로 쓴 값**이다.

⚠ MimicKit 은 IsaacLab **5.1** 에서 돈다(`env_isaaclab` conda env → `/home/lgb/IsaacLab`).
`sim_cfg.physx.bounce_threshold_velocity` 라는 코드 자체가 6.0 에서는 동작하지 않는다 —
6.0 의 `SimulationCfg` 에는 `physx` 가 없고 `physics: PhysicsCfg | None` 로 바뀌었다.

### 10-e. `max_depenetration_velocity` 1.0 → 3.0 arm — 시작함

`2026-08-11_13-58-01_depen3_stock` (GPU3, 60k, stock). `implicit_stock` 대비 **변수 1 개**.
세 노브 중 유일하게 고속-특이적 기전이 그려진다 — 빠를수록 스텝당 발 관통이 깊어지는데
상한 1.0 은 그걸 천천히만 뽑아내므로 **고속에서만 지면이 물렁해지는** 방향이다. 검증 가능한
예측: `cmd 2.5` 는 변화 없고 `cmd 3.0` 이상만 움직인다.

설정 반영은 `params/env.yaml` 과 **USD 프림 되읽기**(`physxRigidBody:maxDepenetrationVelocity`
= 3.0) 두 단계로 확인했다. 상세: [`../2026-08-11_13-58-01_depen3_stock/`](../../go2_imitation_tracking/2026-08-11_13-58-01_depen3_stock/)

## 현재 돌아가는 arm (2026-08-11)

| run | 플랜트 | `implicit_stock` 대비 변수 | GPU |
|---|---|---|---|
| `2026-08-11_11-14-27_noamp_stock` | stock | AMP 혼합 off (`task_reward_lerp` 1.0) | 1 |
| `2026-08-11_13-58-01_depen3_stock` | stock | `max_depenetration_velocity` 1.0→3.0 | 3 |

둘 다 60k, 판정은 40k 이후 램프 점으로만.

## 11. 최종 (2026-08-18) — 속도 천장의 정체는 **AMP style 가중치**였다

세 세대(액추에이터 캡 분리 → ImplicitActuator → 접촉 파라미터)를 뒤진 끝에, 실제로 천장을
쥐고 있던 것은 `agent.amp.task_reward_lerp` 하나였다.

```
total_reward = task_reward_lerp · task + (1 − task_reward_lerp) · style
```
`lerp` 는 **task 쪽 계수**다 — 올릴수록 style 이 약해진다.

### 판정 구간(40k 이후) 점들의 평균 · 범위

`cmd 4.0` (명령 범위 상한):

| arm | n | vx 평균 | vx 범위 | 달성률 | `base_h` 평균 | `base_h` 범위 | h std 평균 |
|---|---:|---:|---|---:|---:|---|---:|
| stock 0.5 (baseline) | 4 | 0.032 | 0.020~0.049 | **0** | — | — | — |
| stock 0.6 (scratch) | 4 | **3.812** | 3.790~3.836 | 95 | 0.376 | 0.374~0.378 | 0.066 |
| stock 0.8→0.6 (커리큘럼) | 3 | 3.746 | 3.724~3.756 | 97 | **0.392** | 0.384~0.398 | 0.062 |
| stock 0.8 (고정) | 4 | **3.870** | 3.798~3.937 | 98 | **0.319** | **0.279~0.365** | 0.071 |
| PACE 0.5 (baseline) | 3 | 0.087 | 0.074~0.101 | **0** | — | — | — |
| **PACE 0.8** | 4 | **3.924** | 3.893~3.969 | 98 | 0.358 | 0.335~0.369 | 0.073 |

`cmd 3.5`:

| arm | vx 평균 | 달성률 | `base_h` 평균 | h std 평균 |
|---|---:|---:|---:|---:|
| stock 0.5 | 0.347 | **0** | 0.259 | 0.048 |
| stock 0.6 | 3.275 | 95 | 0.381 | 0.066 |
| stock 0.8→0.6 | 3.291 | 97 | **0.403** | 0.053 |
| stock 0.8 | 3.348 | 98 | 0.389 | 0.047 |
| PACE 0.8 | **3.423** | 99 | 0.387 | **0.033** |

### 확정된 것

1. **0.5 → 0.6 사이에 문턱이 있다.** baseline(0.5)은 `cmd 3.5`·`4.0` 에서 **전 점 0%** 인데
   0.6 은 95% 다. 한 칸 차이로 천장이 통째로 열린다.
2. **두 플랜트가 같은 방향**이다. 액추에이터 변경 때는 stock/pace 부호가 반대였는데
   (§9) style 가중치는 반전이 없다. 오히려 PACE 0.8 이 **3.924 m/s** 로 전체 최고다.
3. **6.0 종전 최고 1.724 → 3.92 m/s.** 5.1 최고 2.058 의 1.9 배.
4. **style 을 0 으로 만들면(lerp 1.0) 무너진다** — 3.589 m/s 를 내지만 34 Hz 진동 · 몸통
   0.18 m · 토크 캡 상시 54~90% 도달의 실기 불가 모드(§10-c, `noamp_stock`, 29.6k 중단).
   즉 **적정 구간은 0.6~0.8 이고, 양쪽 끝은 다 나쁘다.**

### ★ 커리큘럼은 scratch 0.6 을 못 이긴다

`0.8 → 0.6` 커리큘럼과 **처음부터 0.6** 이 사실상 같은 곳에 도착한다 —
`cmd 4.0` 에서 vx 3.746 vs 3.812, `base_h` 0.392 vs 0.376. 속도는 scratch 가 조금 빠르고
자세는 커리큘럼이 조금 낫다. **"속도를 먼저 얻고 나서 조인다"는 절차가 주는 추가 이득은
측정되지 않았다.** 40k 짜리 사전 학습을 재활용할 수 있다는 점은 실무상 장점이지만,
결과 품질로는 근거가 없다.

### ⚠ 철회 — "커리큘럼이 몸통 진동을 절반으로 줄인다"

48k **한 점**으로 h std 0.074 → 0.051 을 보고 그렇게 적었으나, 56k·59999 를 더 재니
0.070 / 0.065 로 고정 0.8(0.072 / 0.072)과 거의 같아진다. 판정 구간 평균으로는
**0.062 vs 0.071** — 개선은 있으나 "절반"이 아니다. `cmd 3.5` 에서는 오히려 커리큘럼이
근소하게 나쁘다(0.053 vs 0.047).

**대신 견고하게 남는 차이는 `base_h` 다.** 고정 0.8 은 `cmd 4.0` 에서 몸통이
0.279~0.365 로 주저앉고 체크포인트마다 흔들리는데, 0.6 으로 끝나는 두 arm 은
0.374~0.398 로 일관되게 서 있다. 사용자가 영상에서 "속도가 빨라질수록 이상해진다"고
지적한 것이 이 항목이다.

### 남은 것

- **`cmd 4.0` 은 명령 범위 상한**(`lin_vel_x` 0~4.0)이다. 3.92 m/s 는 "범위를 다 채웠다"는
  뜻이지 정책의 최고 속도가 아니다. 더 위를 보려면 범위를 넓혀야 하고, 그러면 기존 램프와
  상단 비교가 깨진다.
- 실기 이관 미검증. sim 지표가 baseline 대역이라는 것까지만 확인됐다.
- 0.6 과 0.8 사이(0.7)는 미측정. 0.5 와 0.6 사이의 문턱 위치도 미측정.

표·그림: `metrics/style_weight_summary.md` · `figures/style_weight_summary.png`
(생성기 `logs/style_weight_summary.py`)

## 12. ★★ 정정 (2026-08-18) — 4 m/s 의 걸음은 참조와 다르고, `cmd 4.0` 에서 **trot 으로 되돌아간다**

사용자가 영상을 보고 두 가지를 지적했다: (1) run 자세가 참조 데이터와 달라 어색하다,
(2) 더 빨라지면 빠른 run 이 아니라 **trot 으로 바뀐다**. 위상차로 재 보니 **둘 다 맞다.**

지금까지 걸음 품질을 `base_h`·부호반전 Hz·토크 캡으로만 봤는데, 이 지표들은 "무너졌는가"는
잡아도 **"어떤 걸음인가"는 구분하지 못한다.** 넓적다리 관절의 Hilbert 위상으로 다리 쌍의
위상차를 내면 보행 종류가 나온다(`logs/gait_phase.py`, 원자료 `metrics/gait_phase.md`).

| 보행 | FL–FR | FL–RL | FL–RR |
|---|---|---|---|
| trot | 0.5 | 0.5 | **0** |
| bound | **0** | 0.5 | 0.5 |
| pace | 0.5 | **0** | 0.5 |

### 참조 모션이 실제로 무엇인가

| clip | v [m/s] | FL–FR | FL–RL | FL–RR | stride Hz | 보행 |
|---|---:|---:|---:|---:|---:|---|
| walk / walk1 / walk2 / walk_turn | 0.09~0.77 | ~0.5 | 0.71~0.98 | 0.17~0.42 | 0.6~1.6 | walk/pace 계열 |
| trot0 · trot1 | 1.72 | 0.50 | 0.56 | 0.06 | 2.02 | **trot** |
| run0 | 2.41 | 0.75 | 0.40 | 0.59 | 2.05 | 비대칭(canter/rotary 계열) |
| run1 | 2.70 | 0.77 | 0.46 | 0.67 | 2.37 | 비대칭(같은 계열) |
| **run2** | **3.96** | **0.12** | 0.58 | 0.44 | **2.54** | **bound** |

★ **데이터셋에 2.70 ~ 3.96 m/s 구간이 비어 있다.** 그리고 3.96 대역은 `run2` **하나뿐**이고
71 프레임(1.18 s)으로 전체 2646 프레임의 **5.4%** 다. leg 프로젝트에서 겪은 속도 공백과 같은 모양이다
([[project_leg_multidisc_amp_design_review]]).

### 정책이 실제로 무엇을 하는가 (`model_59999`, stock, lerp 0.8 고정)

| cmd | vx | FL–FR | FL–RL | FL–RR | stride Hz | 보행 |
|---|---:|---:|---:|---:|---:|---|
| 1.5 | 1.27 | 0.50 | 0.54 | 0.03 | 2.00 | trot |
| 2.0 | 1.73 | 0.48 | 0.51 | 0.00 | 2.67 | trot |
| 2.5 | 2.34 | 0.66 | 0.47 | 0.21 (R **0.10**) | 2.67 | 전이 — env 마다 제각각 |
| 3.0 | 2.86 | 0.88 | 0.50 | 0.24 | 2.67 | 비대칭 |
| 3.5 | 3.37 | 0.87 | 0.47 | 0.22 | 2.67 | 비대칭 |
| **4.0** | **3.94** | 0.59 | 0.46 | **0.06** | **4.00** | **trot (되돌아감)** |

- **`cmd 4.0` 에서 대각 위상차가 0.06 으로 떨어지고 stride 주파수가 2.67 → 4.00 Hz 로 뛴다.**
  같은 속도의 참조(`run2`)는 **bound · 2.54 Hz** 다. 보행 종류도, 주파수도 다르다.
- 빠른 run 은 stride **길이**를 늘려야 하는데, 이 정책은 **주파수**를 올려서 속도를 낸다.
  4.00 Hz trot 은 실기 Go2 가 낼 수 있는 보행이 아니다.
- 2.5~3.5 구간의 비대칭 위상(0.87~0.88 / 0.47 / 0.22)도 참조 run0·run1(0.75~0.77 / 0.40~0.46 /
  0.59~0.67)과 **FL–RR 이 완전히 다르다**. "참조와 달라 어색하다"는 지적이 여기에 해당한다.

### ★ 이게 §11 의 `base_h` 하락을 설명한다

고정 0.8 이 `cmd 4.0` 에서만 몸통이 0.279 로 주저앉고 h std 가 0.072 로 뛰는 것을 §11 에서
"자세를 희생했다"고 적었는데, 실제 원인은 **보행 종류가 trot 으로 바뀐 것**이다. 3.9 m/s 를
trot 으로 내려니 몸통이 낮아지고 상하로 튄다.

### arm 마다 다르다 — 되돌아가는 것은 **고정 0.8(stock)뿐**

| arm | cmd 3.0 | cmd 3.5 | cmd 4.0 |
|---|---|---|---|
| 0.8 고정 (stock) | 비대칭 | 비대칭 | **trot 회귀 · 4.00 Hz** |
| 0.6 scratch | 0.11/0.13/0.66 | gallop 0.14/0.15/0.62 | 0.18/0.19/0.60 · 3.00 Hz |
| 0.8→0.6 커리큘럼 | 0.83/0.45/0.25 | 0.94/0.45/0.25 | 0.89/0.40/0.24 · 3.00 Hz |
| **PACE 0.8** | **bound** 0.11/0.39/0.59 | **bound** 0.12/0.39/0.60 | 0.18/0.39/0.67 · 3.00 Hz |

- **0.6 으로 끝나는 두 arm 과 PACE 0.8 은 고속에서 보행을 유지**한다(회귀 없음).
- **PACE 0.8 이 참조 `run2`(bound 0.12/0.58/0.44)에 가장 가깝다** — `cmd 3.0`·`3.5` 에서
  FL–FR 0.11~0.12 로 일치한다(FL–RL 은 0.39 vs 0.58 로 다름).
- 즉 §11 이 "속도로는 고정 0.8 이 최고(3.87)"라고 적은 것은 맞지만, **그 최고점이 참조에 없는
  4 Hz trot 으로 얻은 값**이다. 보행 종류까지 보면 **고정 0.8 은 채택 후보가 아니다.**

### §11 에서 수정되는 것

- "적정 구간 0.6~0.8" → **0.6 쪽**이다. 0.8 고정은 `cmd 4.0` 에서 참조에 없는 보행으로 새고,
  0.6 으로 끝나는 arm 들은 새지 않는다.
- "커리큘럼은 scratch 0.6 을 못 이긴다"는 속도·`base_h` 기준으로는 유지되지만, **보행 종류
  기준으로는 둘 다 고정 0.8 보다 낫다** — 이게 셋을 가르는 실질적 차이다.
- 3.92 m/s(PACE 0.8)라는 최고 기록 자체는 유지된다. 다만 **`cmd 4.0` 구간은 참조가 5.4% 뿐인
  외삽 영역**이라는 단서가 붙는다.

### 다음에 할 일 (미실행)

1. **참조 데이터셋의 2.7~4.0 공백을 메우는 것**이 가장 직접적이다. `run2` 가 71 프레임뿐이라
   discriminator 가 그 대역을 거의 못 배운다.
2. `cmd 4.0` 을 학습 명령 범위에서 빼고 3.5 까지만 두면(참조가 있는 대역) 회귀가 사라지는지.
3. 보행 종류를 램프 정규 지표로 승격 — 지금까지 `base_h` 만 보다가 이걸 놓쳤다.

## 13. ★★★ (2026-08-18) MimicKit 은 **같은 데이터 · 같은 50/50 혼합**으로 4 m/s 를 낸다

"MimicKit 은 다른 데이터 없이 그 데이터셋만으로 됐다는 거냐"는 질문에 확인한 결과 — **그렇다.**
그리고 이것이 §11~12 의 해석을 바꾼다.

### 13-a. 데이터셋: 동일하다

`/home/lgb/MimicKit/data/datasets/dataset_go2_locomotion_smr_mirror3.yaml`

| 그룹 | weight | 우리 `smr_mirror_pkl` |
|---|---|---|
| pace ×6 | **0.0** (비활성) | 파일 자체가 없음 |
| run0/1/2 (+mirror) ×6 | 1.0 | 있음 |
| trot0/1 (+mirror) ×4 | 1.0 | 있음 |
| walk/walk1/walk2/walk_turn (+mirror) ×8 | 1.0 | 있음 |

**실사용 클립 18 개가 우리와 완전히 같다.** 2.70~3.96 m/s 공백도, `run2` 가 71 프레임뿐인 것도
MimicKit 쪽 사정도 똑같다. → **데이터 공백은 우리만의 문제가 아니다.**

### 13-b. style 혼합: **0.5 / 0.5** — 우리 baseline 과 같은 값

```yaml
task_reward_weight: 0.5
disc_reward_weight: 0.5
```
`mimickit/learning/amp_agent.py:124` → `r = w_task · task_r + w_disc · disc_r`
우리 `on_policy_runner_amp.py:174` 의 `task_reward_lerp = 0.5` 와 **같은 식·같은 값**이다.

그리고 `lin_x_vel_max: 4.0` 으로 명령 범위도 같은데, `results/.../velocity_comparison.png` 는
계단 명령 0→4 m/s 를 **끝까지 추종**한다.

★★ **즉 우리가 `lerp` 를 0.8 로 올려서 얻은 것은 MimicKit 의 레시피가 아니라 우회로다.**
같은 조건(0.5)에서 우리 baseline 은 `cmd 4.0` 에서 0.03 m/s · 0% 다.

### 13-c. 그러면 무엇이 다른가 — 보상 **스케일**과 외란

속도 추종 보상 **식은 완전히 동일**하다 (`exp(-scale · err²)`,
`task_tracking_mixin.py:153-178` vs `go2_imitation_tracking_env.py:415-426`). 값이 다르다:

| | 우리 | MimicKit | 효과 |
|---|---:|---:|---|
| `vel_err_scale` | **0.5** | **1.0** | 1 m/s 오차에 0.61 vs **0.37** — 우리가 2 배 관대 |
| `yaw_vel_err_scale` | 0.5 | 0.5 | 동일 |
| lin : yaw 가중치 | 0.7 : 0.3 | 0.5 : 0.5 | 우리가 lin 을 더 봄 (불리 아님) |
| `push_robot` | **1 m/s 킥 / 5 s** | **없음** | 램프 각 stage 가 거의 매번 외란을 맞음 |
| episode | 20 s | 10 s | |
| action std | 학습 (≈0.15) | **FIXED 0.1** + `action_bound_weight 10` | |

`exp(-scale·err²)` 에서 오차 2 m/s 면 우리 0.135 vs MimicKit 0.018 — **고속 오차를 줄일 압력이
7 배 차이**난다. 우리 정책이 `cmd 4.0` 에서 1.4 m/s 로 만족하고 마는 것과 방향이 맞는다.

### 13-d. ⚠ §5 정정 — MimicKit 은 pose imitation reward 를 쓰지 않는다

§5 에 "MimicKit = AMP disc + imitation reward 직접(pose 0.5, vel 0.1, root_pose 0.15, ...)"
이라고 적었는데 **틀렸다.** `env_config.yaml` 에 `reward_pose_w` 등이 있는 것은 사실이지만,
`TaskTrackingMixin._update_reward` 가 `self._reward_buf[:]` 를 **속도 추종 보상으로 통째로
덮어쓴다**(`super()` 호출 없음). 상속된 DeepMimic 설정이 남아 있을 뿐 이 env 는 읽지 않는다.
→ **보상 구조는 우리와 같은 모양이다**(task 속도추종 + AMP style, 50/50).

### 13-e. MimicKit 의 4 m/s 도 곱진 않다

같은 결과 폴더의 그림을 보면 `cmd 4.0` 부근에서 roll rate ±1.5~2.5 rad/s, pitch rate
±3~4 rad/s, vz ±0.5 m/s 로 몸통이 크게 흔들리고, `joints.png` 의 thigh 토크는 ±20~25 N·m
(캡 23.7)로 포화, thigh `|q̇|` 는 ~20 rad/s 다. **우리 lerp 0.8 과 같은 대역이다.**
MimicKit 쪽 보행 **종류**는 원자료(joint 시계열)가 없어 §12 와 같은 위상 판정을 못 했다.

⚠ §6 에 적어 둔 단서가 그대로 유효하다 — 이 PNG 를 만든 평가 실행의 엔진 설정은 기록이 없다.
"같은 엔진에서 4 m/s"는 **학습 기준으로만** 확인된 것이다.

### 13-f. 다음 실험 (권장 순서)

1. **`vel_err_scale` 0.5 → 1.0, `lerp` 는 0.5 유지** — MimicKit 의 정확한 설정. 이것만으로
   baseline 이 열리면 §11 의 "style 가중치가 천장"이라는 진단 자체가 바뀐다.
2. `push_robot=False` 추가 (MimicKit 에 없는 외란).
3. 둘 다 켠 조합.

★ 1 번이 열리면 **lerp 0.8/0.6 세대는 전부 우회로였던 것**이 되고, §12 에서 본
`cmd 4.0` trot 회귀도 "약해진 style 로 고속을 짜낸 부작용"으로 설명된다.

## 14. (2026-08-22) MimicKit 설정 정렬 3 arm — **천장은 안 열린다.** 원인은 아직 미상

§13 의 처방(`vel_err_scale` 0.5→1.0, `push_robot` off)을 lerp 0.5 그대로 두고 셋 다 60k 완주.
판정 구간(40k 이후) 4 점 평균:

| arm | cmd 2.5 vx / % | cmd 3.0 vx / % | cmd 3.5 % | cmd 4.0 % | top |
|---|---:|---:|---:|---:|---:|
| baseline (0.5 / push) | 1.499 / 96 | 1.309 / 48 | **0** | **0** | 1.686 @3.0 |
| **vel_scale 1.0** | 1.866 / 90 | 1.329 / 46 | **0** | **0** | 1.896 @2.5 |
| **push off** | 1.474 / 96 | 1.265 / 48 | **0** | **0** | 1.688 @3.0 |
| **둘 다** | 1.880 / 93 | 1.608 / 53 | **0** | **0** | 1.914 @3.0 |
| *[참고] lerp 0.8* | *2.370 / 98* | *2.886 / 98* | ***98*** | ***98*** | ***3.937 @4.0*** |

### ★ §13 의 가설은 기각된다

- **`cmd 3.5`·`4.0` 은 세 arm 전부 0%.** baseline 과 같다. `vel_err_scale` 을 MimicKit 값으로
  맞춰도 고속 천장은 그대로다.
- `vel_err_scale` 1.0 의 효과는 **중속에 국한**된다 — `cmd 2.5` 평균 1.499 → 1.866 (+25%).
  다만 달성률은 96 → 90% 로 **떨어진다**(더 엄격한 보상이라 절반 기준 통과가 빡세짐).
- **`push_robot` off 는 단독으로 아무 효과가 없다** (1.499 → 1.474). 조합에서도 `vel_scale`
  단독과 사실상 같다. → 외란 가설은 기각.
- 최고 기록: 1.914 m/s. lerp 0.8 의 **3.937** 과 두 배 차이다.

→ **"MimicKit 이 0.5 로 4 m/s 를 내는데 우리는 못 내는" 이유는 여전히 설명되지 않는다.**
우리가 4 m/s 에 닿는 유일한 경로는 style 을 약화(lerp 0.8)하는 것이고, 그 결과물은
§12 에서 본 대로 참조에 없는 4 Hz trot 이다.

### ★★ 왜 안 열리는지에 대한 관측 — style 0.5 는 **trot 에 잠근다**

위상차로 보면 세 arm 모두 전 구간 **trot 을 벗어나지 못한다**:

| arm | cmd 2.0 | cmd 2.5 | cmd 3.0 | cmd 3.5 |
|---|---|---|---|---|
| vel_scale 1.0 | trot (FL–RR 0.00) | trot (0.01) | trot (0.11) | 붕괴 (vx 0.13) |
| push off | trot (0.02) | trot (0.01) | trot (0.05) | 붕괴 (vx 0.20) |
| 둘 다 | trot (0.02) | trot (0.01) | trot (0.12) | 붕괴 |
| *lerp 0.8* | *trot* | *전이 시작* | *비대칭* | *비대칭 3.37 m/s* |

lerp 0.8 은 `cmd 2.5` 부근에서 **trot → 비대칭 보행으로 전이**하는데, 0.5 짜리 arm 들은
전이하지 못하고 trot 을 더 빠르게 돌리다가 3.5 에서 무너진다. `vel_scale` 1.0 은 그 trot 의
상한을 1.50 → 1.87 m/s 로 올릴 뿐 **전이를 만들지 못한다.**

가설(미검증): 데이터셋에서 trot·walk 가 프레임의 대부분이고 `run2`(bound, 3.96 m/s)는 5.4%
뿐이라, disc 가중치 0.5 에서는 **trot 이 discriminator 가 보상하는 안전 영역**이고 정책이
거기서 못 벗어난다. style 을 0.2 로 낮춰야 비로소 벗어난다.

### 아직 안 본 MimicKit 차이 (다음 후보)

§13-c 에서 목록만 만들고 검증하지 않은 것들:

| 항목 | 우리 | MimicKit |
|---|---|---|
| action std | 학습 (≈0.15) | **FIXED 0.1** + `action_bound_weight 10` |
| optimizer | Adam + adaptive KL | **SGD** (actor 2e-4 / critic 1e-4) |
| episode | 20 s | 10 s |
| lin : yaw 보상 가중 | 0.7 : 0.3 | 0.5 : 0.5 |
| DR | 마찰·질량·CoM·게인 + **obs noise · action delay** | 마찰·질량·CoM·게인만 |

★ `FIXED 0.1` action std 가 가장 유력해 보인다 — 학습 std 는 탐색량이 정책 스스로 정해지는데,
고정 std 는 **끝까지 같은 크기로 탐색**하므로 trot 국소해에서 빠져나올 확률이 다르다.
`obs noise`·`action delay` 부재도 고속에서 유리한 방향이다.

원자료: `metrics/ramp_velscale1/`, `metrics/ramp_nopush/`, `metrics/ramp_velscale1_nopush/`
· 보행 위상 `metrics/gait_phase.md`

---

## 15. (2026-08-26) 고정 std · 상수 LR arm — 그리고 **실험 설계 전체 지도**

### 15-a. 결과 요약

두 arm 을 더 돌렸다. 둘 다 `task_reward_lerp` 0.5(baseline 혼합) 고정, 변수 하나씩.

| arm | 변경 | 결말 |
|---|---|---|
| `fixedstd_stock` / `fixedstd_vel1_stock` | `noise_std_type=fixed`, `init_noise_std=0.1` | **붕괴** — 48k/51k 에서 중단 |
| `schedfixed_stock` | `agent.algorithm.schedule=fixed` (상수 LR 2e-4) | 60k 완주, 붕괴 없음 |

**고정 std 붕괴의 원인은 std 가 아니라 스케줄러와의 상호작용이었다.** 고정 σ 에서 KL 은
`Δμ²/(2σ²)` 라, baseline 이 초기에 0.42 까지 올렸다 0.15 로 내리는 std 를 0.1 로 못 박으면 같은
평균 변화가 훨씬 큰 KL 로 찍힌다 → 스케줄러가 LR 을 바닥 1e-5 에 고정 → surrogate 발산(0.29).
MimicKit 은 SGD 상수 LR 이라 이 경로가 없다. **상수 LR 단독 arm 이 멀쩡히 완주한 것으로 이 진단이
뒷받침된다** — Adam + 상수 2e-4 가 터질 거라던 사전 우려는 빗나갔다.

추가로 **단위도 안 맞았다**: MimicKit `pos` 모드는 action 이 곧 관절 목표각인데(`char_env.py:472`)
우리는 `action_scale=0.25` 를 곱한다. 그쪽 std 0.1 = 0.1 rad, 우리 std 0.1 = 0.025 rad 로 **4 배 작다.**
등가값은 0.4. 고정 std 를 다시 하려면 **상수 LR + 0.4** 를 같이 가야 한다.

### 15-b. 상수 LR 은 **속도가 아니라 달성률**을 바꿨다 — 그러나 전이는 없다

학습 지표는 전 구간 baseline 위다(60k: ep_len 996 vs 980, reward 810 vs 750,
`lin_vel_reward` 33.4 vs 28.9, `amp_reward` 43.3 vs 40.4 — **두 항이 동시에** 올랐다).

램프(40k 이후 4 점, `metrics/mimickit_align_summary.md`):

```
                       arm |   cmd 2.5    cmd 3.0    cmd 3.5    cmd 4.0
    baseline (adaptive LR) | 1.598(97)  1.686(58)  0.592( 0)  0.049( 0)
 const LR (schedule=fixed) | 1.586(95)  1.673(95)  1.626( 9)  0.131( 0)
```

★ `cmd 3.0` 에서 **median vx 는 1.673 vs 1.686 으로 사실상 동일한데 달성률만 58% → 95%** 다.
더 빨라진 게 아니라 넘어지거나 멈추던 env 가 안 넘어진다. `cmd 3.5` 도 style 0.5 계열에서 처음으로
0 이 아닌 값(0.592 → 1.626)이 나왔다. 이것은 §14 의 세 arm(`velscale1`·`nopush`·`vel1+nopush`)과
**반대 축**이다 — 저쪽은 속도를 올리고 달성률은 58~64% 그대로였다.

★★ 그러나 **위상차로 보면 전 구간 trot 이고 전이가 없다**(`metrics/gait_phase.md`).
`cmd 3.5` 의 1.626 도 2.67 Hz trot 이다. 같은 표에서 PACE lerp 0.8 은 `cmd 3.0` 부터 bound
(FL-FR 0.11)로 넘어간다. **결론은 §14 와 같다: style 0.5 는 trot 에 잠근다.** 상수 LR 은 그 trot 을
더 튼튼하게 만들 뿐이다.

`cmd 4.0` 걸음 품질은 붕괴 모드가 아니다(`base_h` 0.271, flip 8.4 Hz) — 단지 안 달릴 뿐이다.

### 15-c. ★ 실험 설계 지도 — 왜 이 조합들인가

지금까지의 arm 은 **세 축**으로 나뉜다. 축을 섞지 않는 것이 이 시리즈의 설계 원칙이다.

```
축 A. 플랜트/시뮬          "로봇이 물리적으로 못 내는 속도인가?"
축 B. AMP style 가중치     "보상 혼합이 고속을 막는가?"
축 C. MimicKit 레시피 정렬  "같은 데이터·같은 혼합인데 왜 저쪽만 되는가?"
```

| # | arm | 축 | 변경한 것 하나 | 결말 |
|---|---|---|---|---|
| 1 | `implicit_stock` | A | DCMotor → ImplicitActuator (토크-속도 곡선 제거) | baseline. `cmd 3.5` 0% — 곡선은 원인이 아님 |
| 2 | `depen3_stock` | A | `max_depenetration_velocity` 1.0 → 3.0 | 기각. `cmd 2.5` 손해(4/4 점), 최고점 못 넘김 |
| 3 | `noamp_stock` | B | `lerp` 1.0 (style 제거) | 3.589 m/s 나오지만 `base_h` 0.18 · 34 Hz 진동 = 실기 불가 |
| 4 | `lerp08_stock` / `lerp08_pace` | B | `lerp` 0.8 | **`cmd 4.0` 97~98% 달성.** 단 stock 은 고속에서 trot 회귀(§12) |
| 5 | `lerp06_stock` | B | `lerp` 0.6 | 0.8 과 대등. 임계는 0.5~0.6 사이 |
| 6 | `lerp08to06_curr` | B | 0.8 → 0.6 커리큘럼 | scratch 0.6 과 동률. 이점은 비용뿐 |
| 7 | `velscale1_stock` | C | `vel_err_scale` 0.5 → 1.0 | `cmd 2.5` 1.50→1.87 로 오르지만 `cmd 3.5+` 0% |
| 8 | `nopush_stock` | C | `dr.push_robot` off | 무효과 |
| 9 | `velscale1_nopush` | C | 7+8 동시 | 7 과 같음. 상호작용 없음 |
| 10 | `fixedstd_stock` / `_vel1` | C | action std 학습 → 고정 0.1 | **붕괴** (원인은 15-a) |
| 11 | `schedfixed_stock` | C | adaptive KL → 상수 LR | 완주. 달성률만 개선, 전이 없음 |

**왜 이 순서인가.**

1. **축 A 를 먼저 소거했다.** "액추에이터가 부족해서"라면 보상을 아무리 만져도 안 되므로, 토크-속도
   곡선을 통째로 제거하는 극단 실험(#1)으로 먼저 끝냈다. 13 개 체크포인트 전부 `cmd 3.5` 0% 라
   **물리 한계 가설은 반증**됐고, 벽은 평탄한 thigh 23.7 N·m 로 옮겨갔다.
2. **축 B 로 원인을 찾았다.** `lerp` 하나만 움직여 0.5 / 0.6 / 0.8 / 1.0 을 채웠다(#3~#6). 양 끝이 다
   나쁘고 중간이 좋은 모양이라, **단조 관계가 아니라 최적점이 있는 축**임을 알려면 끝점을 반드시
   찍어야 했다. 이 축에서 `cmd 4.0` 이 열렸다.
3. **축 C 는 §13 이후에 열렸다.** MimicKit 이 **우리와 같은 18 클립 · 같은 50/50 혼합**으로 4 m/s 를
   낸다는 것이 확인되면서, `lerp 0.8` 은 해법이 아니라 **우회로**가 됐다. 그래서 "50/50 을 고정한 채
   MimicKit 과 다른 항을 하나씩 되돌린다"가 #7~#11 의 규칙이다. `lerp` 를 같이 움직이면 개선이
   어느 쪽 덕인지 못 가린다.
4. **#9 처럼 조합 arm 을 넣는 이유**는 상호작용 때문이다. 두 요인이 따로는 안 되는데 같이 걸면
   열리는 경우가 이 task 에서 드물지 않아, 단독 두 개가 실패하면 2×2 격자를 한 칸 채워 확인한다.

**판정 규칙(이 시리즈 전체에 적용).**

- ★ **40k 이후 점으로만 판정.** 24k 이하에서 "3 점 연속·두 플랜트·부호 6/6" 으로 낸 판정이 40k 에서
  뒤집힌 적이 있다(−41 → +6). 점을 더 모으는 게 해법이 아니라 **늦은 점만** 쓰는 것이 해법이다.
- 재현 산포는 달성률 ±8~15%p / median ±0.02 m/s. 이보다 작은 차이는 차이가 아니다.
- ★★ **속도와 보행 종류를 같이 본다.** `base_h`·부호반전 Hz·토크 캡은 "무너졌는가"는 잡아도
  "어떤 걸음인가"는 못 잡는다. §12 의 trot 회귀와 §14~15 의 "전이 없음"은 둘 다 위상차로만 보였다.
- 램프는 run params 가 아니라 **현재 소스 cfg** 로 env 를 만든다. 세대가 다른 체크포인트를 잴 때
  `--no_pace` / `--joint_vel_noise` / `--sync_spawn_props` 로 학습 조건을 명시 고정할 것.

### 15-d. 남은 미검증 (축 C)

| 항목 | 우리 | MimicKit | 비고 |
|---|---|---|---|
| optimizer | **Adam** | **SGD** | 상수 LR 은 맞췄으나 optimizer 는 아직 |
| action std | 학습 ≈0.15 | FIXED 0.1 (= 우리 **0.4**) | 상수 LR + 0.4 로 재시도 가능해짐 |
| `action_bound_weight` | 없음 (하드 클립만) | 10.0 | FIXED std 와 세트로 쓰임 |
| episode | 20 s | 10 s | |
| lin : yaw 보상 가중 | 0.7 : 0.3 | 0.5 : 0.5 | |
| DR | + obs noise · action delay | 없음 | 고속에 불리한 방향 |

축 C 를 다 소거해도 안 열리면, 남는 설명은 **참조 데이터의 2.70~3.96 m/s 공백**(§13-e)이다.
그 구간을 채우거나 학습 명령 상한을 참조가 존재하는 3.5 로 낮추는 것이 그다음 갈래다.

원자료: `metrics/ramp_schedfixed/` · `metrics/ramp_fixedstd{,_vel1}/`
· 요약 `metrics/mimickit_align_summary.md` (생성기 `logs/mimickit_align_summary.py`)
· 보행 위상 `metrics/gait_phase.md`

---

## 16. (2026-08-26) 관측 입력 대조 — **의미는 같아도 표현이 다르다**

§13~15 는 보상·DR·optimizer 만 비교했고 **관측 구성은 한 번도 대조하지 않았다.** 여기서 잰다.

### 16-a. 검산 — 차원은 전부 체크포인트 가중치로 확인했다

소스만 읽으면 틀리기 쉬워서 두 쪽 다 실제 가중치 shape 으로 닫았다.

| | 우리 (`schedfixed model_59999`) | MimicKit (`amp_adaptation_module_20260512_160750`) |
|---|---|---|
| actor 1층 | `(512, 68)` = 42 + 6 + 20 | `(1024, 137)` = 117 + 20 |
| critic 1층 | `(512, 67)` = 42 + 6 + 19 | `(1024, 1296)` = 117 + 9 + 10×117 |
| priv_encoder 입력 | **19** (DR 파라미터) | **9** (root_vel 3 + root_ang_vel 3 + mass 1 + friction 2) |
| history_encoder step 폭 | **42** | **117** |
| discriminator 입력 | `(1024, 490)` = 10 × **49** | `(1024, 1340)` = 10 × **134** |

네 식이 모두 정확히 닫힌다.

### 16-b. proprio 42 vs 117 — 차이는 **관절 각도 표현**이 대부분이다

| 항목 | 우리 | MimicKit |
|---|---|---|
| 몸통 자세 | `projected_gravity_b` **3** | `quat_to_tan_norm(heading⁻¹·root_rot)` **6** |
| **관절 위치** | `joint_pos − default_joint_pos` **12** (raw rad) | 관절별 quaternion → `quat_to_tan_norm`, 16 × 6 = **96** |
| 관절 속도 | **12** | **12** |
| 명령 (vx, vy, yaw) | **3** | **3** |
| **이전 action** | **12** | **없음** |
| 합 | **42** | **117** |

96 의 출처: `go2.xml` 은 body 17 개 · `<joint>` 12 개인데 `kin_char_model.dof_to_rot` 는
`num_joints − 1 = 16` 개의 관절 회전을 낸다(`kin_char_model.py:146`). 16 × 6 = 96.

★ **의미는 같지만 성질이 다르다.** 우리 관절 입력은 기본자세 기준 **raw 라디안 12 개** — 무한
정의역의 선형값이다. MimicKit 은 관절마다 회전을 tan-norm 으로 펴서 **[−1,1] 유계 · θ 에 대해
주기적**인 96 개다. 몸통 자세도 3(중력벡터, roll/pitch 만) vs 6(tan-norm) 이다.

### 16-c. priv 는 담는 내용 자체가 다르다

| | 우리 | MimicKit |
|---|---|---|
| priv_encoder 입력 | DR 파라미터 19 (마찰·질량·게인·CoM …) | **root 선속도·각속도** + 질량 + 마찰 = 9 |
| actor 가 직접 받는 것 | proprio 42 + **추정 root 속도 6** + latent 20 | proprio 117 + latent 20 (**속도는 latent 안에만**) |
| critic 이 보는 것 | 42 + 6 + 19 = 67 (**history 안 봄**) | 117 + 9 + **history 전체 1170** = 1296 |

우리 actor 는 root 선속도를 **6 차원 슬롯으로 직접** 받는다(`train_with_estimated_states: true`
라 학습·램프 모두 estimator 추정치). MimicKit actor 는 속도를 명시적으로 못 받고 latent 로만 받는다.
critic 쪽은 반대로 MimicKit 이 압도적으로 많이 본다.

### 16-d. ✅ estimator 오차는 원인이 아니다 (추가 실행 0 으로 확인)

"actor 가 받는 추정 속도가 고속에서 틀려서 못 달리는 것 아닌가"는 램프 npz 에 이미 저장된
`est_lin`/`gt_lin`(env 0)으로 즉시 확인된다:

```
             cmd 2.5   cmd 3.0   cmd 3.5   cmd 4.0    (|추정−실제| / 실제)
baseline        0.03      0.02      0.09      0.29
schedfixed      0.02      0.03      0.07      0.05
lerp 0.8        0.02      0.05      0.02      0.01   <- 실제 3.9 m/s 로 달리는 중
```

절대오차는 어디서도 0.15 m/s 를 넘지 않는다. **실제로 4 m/s 를 내는 arm 에서 오차가 가장 작다**
(1%). 고속 실패 구간의 큰 상대오차는 속도가 0 근처라 분모가 작아서 나온 값이다.
→ **estimator 가설 기각.** `--gt_priv` A/B 를 돌릴 근거가 없다.

### 16-e. DR — MimicKit 쪽을 config 로 확정했다

§13-c 에서 목록만 적었던 것을 `engine_config.yaml` 로 확인했다. MimicKit DR 은
**마찰 · base 질량 · CoM · PD 게인 4 개뿐**이고, **obs noise · action delay · push 는 없다.**
우리는 여기에 obs noise 와 action delay 가 더 있다(push 는 §14 에서 무효과로 확인).

### 16-f. 판정 — 후보이지 원인이 아니다

지금까지 관측 축은 **한 번도 실험하지 않았다.** 아래는 후보 목록이며, 어느 것도 천장의 원인으로
확인되지 않았다. §15-d 의 미검증 항목에 이 셋을 더한다.

| 후보 | 왜 후보인가 |
|---|---|
| **관절 각도 표현** (12 raw rad vs 96 tan-norm) | 무계 선형 입력은 학습 분포 **가장자리**에서 성질이 나빠진다(정규화 통계 이동·활성 포화). 천장이 정확히 그 가장자리에 있다 |
| **discriminator 관측 폭** (49 vs 134/step) | §14~15 의 결론이 "style 0.5 는 trot 에 잠근다" 인데, 그 style 신호를 만드는 판별기가 2.7 배 좁게 본다 |
| **이전 action** (우리 12, 저쪽 0) | 실재하는 차이. 기전은 아직 붙이지 않는다 |

원자료: 두 쪽 체크포인트 가중치 shape · `logs/amp_adaptation_module_20260512_160750/{env,engine,agent}_config.yaml`
· estimator 오차는 `metrics/ramp_*/[..]/ramp_data.npz` 의 `est_lin`/`gt_lin`

---

## 17. (2026-08-27) 관측 표현 arm — 정책 쪽 결과, disc 쪽 착수, 그리고 참조 데이터 결함 1 건

### 17-a. 정책 tan-norm (`tannorm_stock`) — 40k 에서 중단

§16-f 의 후보 1 순위였다. `cmd 3.5`·`4.0` **0%**, 위상차 전 구간 **trot**. 40k 한 점만 재고
중단했다(사용자 판정). 같은 40000 체크포인트끼리 맞춰 비교하면:

```
  arm @40000 |    cmd 2.0     cmd 2.5     cmd 3.0     cmd 3.5     cmd 4.0
    baseline | 1.315(98)   1.389(94)   0.686(41)   0.180( 0)   0.037( 0)
    const LR | 1.406(94)   1.567(94)   1.672(94)   0.594( 3)   0.051( 0)
    tan-norm | 1.324(86)   1.379(86)   1.306( 0)   0.395( 0)   0.125( 0)
```

⚠ `cmd 3.0` 에서 **두 지표가 반대 방향**이다 — tan-norm median 이 baseline 의 거의 2 배인데
(1.306 vs 0.686) 달성률은 0% vs 41%. 모순이 아니라 분포 모양이다: 달성 문턱이 1.5 m/s 인데
baseline 은 **양봉**(41% 가 1.7 에서 달리고 나머지는 주저앉아 median 이 끌려내려감), tan-norm 은
**뭉쳐 있는데 문턱 바로 아래**다. 어느 쪽이 낫다고 말할 수 없다.

한 점이라 우열은 판정하지 않는다 — baseline 자신의 `cmd 3.0` 이 40k 이후 점마다
0.686(41%) / 1.686(58%) / 1.623(53%) / 1.240(42%) 로 vx 가 2.5 배 범위에서 흔들린다.
**판정 가능한 것은 "전이 없음" 하나**이고, 그건 한 점으로도 말할 수 있다(대각 위상차 0.02~0.04).

### 17-b. ★ 정책 쪽 노브는 6/6 실패했다 — 벽은 style 신호에 있다

| arm | 바꾼 것 | `cmd 3.5` | 보행 |
|---|---|---:|---|
| baseline | — | 0% | trot |
| vel_err_scale 1.0 | 보상 스케일 | 0% | trot |
| push off | 외란 | 0% | trot |
| vel1 + nopush | 둘 다 | 0% | trot |
| const LR | 옵티마이저 스케줄 | 9% | trot |
| tan-norm (policy) | 관측 표현 | 0% | trot |

여섯 arm 이 서로 다른 축을 건드렸는데 전부 같은 자리로 수렴한다. 산포가 아니라 구조다.
반면 `lerp` 를 올려 **style 을 약화**하면 한 번에 열린다(`cmd 4.0` 97%, bound 로 전이).

### 17-c. disc 관측 분해 — 차이 85 중 **84 가 관절 각도 표현**이다

§16-a 에서 disc 폭만 비교했던 것(49 vs 134)을 성분까지 분해했다
(`amp_env.compute_disc_obs` = `deepmimic_env.compute_tar_obs` + `compute_disc_vel_obs`):

| 성분 | 우리 | MimicKit |
|---|---:|---:|
| **관절 각도** | **12** (raw rad) | **96** (tan-norm, 16 관절) |
| root 위치 | 높이 1 | 참조 기준 상대 xy 2 |
| root 회전 | tan-norm 6 | tan-norm 6 |
| key body(발) | 12 | 12 |
| root 선속도 / 각속도 | 3 / 3 | 3 / 3 |
| 관절 속도 | 12 | 12 |
| 합 | **49** | **134** |

`49 + 84 + 1 = 134` 로 정확히 닫힌다. → **`amp_joint_tan_norm` arm 착수**
(`2026-08-27_10-54-18_ampTanNorm_stock`, per-step 49 → **109**).

### 17-d. ⚠ 참조 모션 회전 파라미터화 결함 — **전 AMP run 에 들어가 있었다**

disc 스모크에서 expert 값 범위가 `[−353, 41]` 로 나와 추적했다.

★★ **[2026-08-28 정정] 처음 적은 원인 규정("yaw 를 unwrap 하지 않는다")은 틀렸다.**
pkl 의 `frames[:, 3:6]` 은 roll/pitch/yaw 가 **아니라 exponential map**(axis × angle)이다
(변환기 `quat_to_exp_map` 이 `w >= 0` 을 강제해 최단호로 만든다). 구 코드는 이를 **euler 로
해석**하고 있었다 — unwrap 누락이 아니라 **회전 파라미터화 자체가 틀린 것**이고, 그래서 영향이
각속도에 그치지 않는다.

```
구 코드   root_quat = _euler_to_quat_wxyz(root_exp_map)          # exp-map 을 euler 로 읽음
          ang_vel   = _euler_rates_to_body_angvel(...)           # 그걸 다시 euler-rate 로 미분
현재 코드 root_quat = _exp_map_to_quat_wxyz(root_exp_map)
          ang_vel   = _quat_body_ang_vel(root_quat, dt)          # 쿼터니언 차분 + double-cover 처리
```

회전각이 작으면 exp-map ≈ euler 라 17 개 클립에서는 오차가 묻혔고, **크게 도는 `go2_walk_turn`
에서만** 크게 벌어졌다 — 그래서 오래 안 보였다:

```
                  |회전각| max     root_quat 오차 p50     p99      max   [deg]
go2_walk_turn           179.7             1.45          11.94    12.08
go2_walk2                34.6             0.58           2.62     2.79
go2_walk                  7.7             0.07           0.24     0.24
전체                                       0.32           9.25    12.1
```

각도가 π 를 넘는 지점에서 `w >= 0` 강제 때문에 표현이 **반대축으로 점프**하고, 그 ~2π 불연속을
그대로 미분해 각속도가 **373.4 rad/s** 로 튄다(실제 2.26).

정규화가 이를 전 구간에 퍼뜨린다 — `ppo_amp.py:187` 이
`update_normalization(cat([expert, policy]))` 라 expert 가 통계에 들어간다:

```
                  std        wx       wy        wz     |max|
현재 코드                 0.526    1.572    10.280     373.4
np.unwrap 적용            0.586    1.006     0.544      13.6
팽창률                    0.90x    1.56x    18.9x
```

→ **disc 가 보는 expert yaw rate 가 실제의 5% 로 눌린다.** 2628 프레임 중 2 개(0.08%)의 결과다.
pitch 도 64% 로 눌린다.

★ 수정된 코드로 다시 재면 스파이크가 사라진다(제안했던 `np.unwrap` 보다 나은 수정이다):

```
                       std       wx      wy       wz     |max|
구 코드(euler-rate)          0.526   1.572   10.280     373.4
현재 코드(quat 차분)          0.446   1.003    0.544       4.3
```

⚠ **4 m/s 천장의 원인일 가능성은 여전히 낮다** — 눌리던 것이 주로 yaw 인데 고속 전이는 pitch 축
문제이고, 같은 데이터로 lerp 0.8 은 잘 열렸다. §18 의 실측도 이 예상과 맞는다.

원자료: `metrics/ramp_tannorm/stock_40000/` · 요약 `metrics/mimickit_align_summary.md`
· 위상 `metrics/gait_phase.md`

---

## 18. (2026-08-28) disc tan-norm + 회전 수정 (`ampTanNorm_rotfix_stock`) — 40k 에서 천장 안 열림

### 18-a. 이 run 은 **단일 변수가 아니다**

`ampTanNorm_stock`(§17-c 로 띄운 것)은 iteration **7426** 에서 멈췄고, 08-27 14:52 에
`ampTanNorm_rotfix_stock` 이 새로 시작됐다. `motion_lib.py` 가 그 10 분 전(14:42)에 수정됐고
이 run 의 `git/IsaacLab-6.0.diff` 에 그 파일이 들어 있다. 즉 **두 가지가 동시에 바뀌었다**:

1. disc 관절 각도 tan-norm (per-step 49 → **109**)
2. 참조 모션 **회전 파라미터화 수정** (§17-d) — expert 분포가 이전 arm 전부와 다르다

★ 이기든 지든 **어느 쪽 덕/탓인지 이 run 만으로는 못 가른다.** 가르려면
`amp_joint_tan_norm=false` + rotfix 만 켠 arm 이 하나 필요하다.

설정 확인: `amp_joint_tan_norm: true` / `amp_observation_space: 109` /
`joint_pos_tan_norm: false` / `observation_space: 42` / `task_reward_lerp: 0.5` /
`schedule: adaptive` / `use_pace_params: false`.

### 18-b. 40k 램프 — 같은 체크포인트끼리

```
       arm @40000 |     cmd 1.5     cmd 2.0     cmd 2.5     cmd 3.0     cmd 3.5     cmd 4.0
         baseline |  1.109( 98)  1.315( 98)  1.389( 94)  0.686( 41)  0.180(  0)  0.037(  0)
         const LR |  1.139( 92)  1.406( 94)  1.567( 94)  1.672( 94)  0.594(  3)  0.051(  0)
  policy tan-norm |  1.101( 91)  1.324( 86)  1.379( 86)  1.306(  0)  0.395(  0)  0.125(  0)
 disc tn + rotfix |  1.105( 91)  1.331( 91)  1.449( 88)  1.379( 28)  0.248(  0)  0.033(  0)
```

`cmd 3.5`·`4.0` **0%**. `cmd 3.0` 은 median 1.379 로 baseline(0.686)보다 높지만 달성률 28% 로
낮다 — §17-a 의 policy tan-norm 과 같은 **양봉 vs 뭉침** 구도이고, 우열은 한 점으로 판정하지 않는다.
`cmd 4.0` 걸음 품질은 붕괴 모드가 아니다(`base_h` 0.265, flip 10.1 Hz, cap 8%) — 안 달릴 뿐이다.

### 18-c. ★★ 전이 없음 — style 0.5 는 이제 **7/7** 이다

```
[disc tn+rotfix 40k]  cmd    vx      FL-FR    FL-RL    FL-RR     Hz   gait
                      2.0  1.331     0.50     0.53     0.03    2.33  trot
                      2.5  1.449     0.50     0.52     0.03    2.33  trot
                      3.0  1.379     0.50     0.54     0.03    2.33  trot
```

대각 위상차가 0.03 으로 끝까지 붙어 있다. (`cmd 3.5`·`4.0` 의 "pace" 판정은 R 0.27~0.57 로
env 간 위상이 흩어진 상태 — 걷지 못하고 있는 것이지 보행 종류가 아니다.)

**§17-d 수정이 천장을 열지 못했다.** 예상대로다 — 그 결함이 누르던 것은 주로 yaw 채널인데
고속 전이는 pitch 축 문제다.

### 18-d. disc 는 포화하지 않았다 (같은 iter 끼리)

입력이 490 → 1090 이라 disc 하이퍼파라미터의 실효 강도가 움직였을 수 있어 확인했다:

```
분리도 (expert−policy)     1000     5000    15000    25000    35000    43000
baseline                +0.5710  +0.4163  +0.3648  +0.3582  +0.3469  +0.3450
rotfix                  +0.5356  +0.4328  +0.3818  +0.3708  +0.3691  +0.3576
```

두 곡선이 붙어 있다 → `lerp` 0.5 의 의미가 유지된다. 이 arm 은 조용히 `noamp` 쪽으로 흘러간
다른 실험이 아니다.

⚠ 다만 **LR 은 다르다** — baseline 은 바닥에서 벗어나는데(15k 3.4e-5, 43k 5.1e-5) rotfix 는
전 구간 **1e-5 에 고정**이다. 붕괴는 없지만(ep_len 972) 고정 std arm 을 죽인 것과 같은 패턴이므로
기록해 둔다.

### 18-e. 정리 — style 0.5 에서 시도한 것

| arm | 바꾼 것 | `cmd 3.5` | 보행 |
|---|---|---:|---|
| baseline | — | 0% | trot |
| vel_err_scale 1.0 | 보상 스케일 | 0% | trot |
| push off | 외란 | 0% | trot |
| vel1 + nopush | 둘 다 | 0% | trot |
| const LR | 옵티마이저 스케줄 | 9% | trot |
| policy tan-norm | 정책 관측 표현 | 0% | trot |
| **disc tan-norm + rotfix** | **판별기 관측 표현 + 참조 회전** | **0%** | **trot** |

일곱 arm 이 서로 다른 축을 건드렸는데 전부 같은 자리다. `lerp` 를 올리는 것만이 열린다.

원자료: `metrics/ramp_amptn_rotfix/stock_40000/` · `metrics/mimickit_align_summary.md`
· `metrics/gait_phase.md`

### 18-f. 램프는 낙상 리셋을 끄고 잰다 — 그 대가를 확인했다

`speed_ramp_record.py:198` 이 `env_cfg.early_termination = False` 로 **의도적으로 끈다**(학습은
`True`, `termination_height` 0.15 m, roll/pitch 70°). 명령 프로파일이 0 → 4.0 → 0 인 하나의 연속
시퀀스라, 중간에 리셋되면 참조 모션 프레임으로 재스폰돼 "처음부터 달려온 개체"와 "방금 되살아난
개체"가 같은 stage 통계에 섞이기 때문이다.

대가는 **넘어진 env 가 이후 전 구간에 그대로 남는다**는 것이다. leg 쪽에서 같은 구조가 실제로
오독을 낳았다(넘어져 누운 롤아웃이 duty 0.00 = "정지 자세 유지"로 집계). go2 쪽 분석기
(`mimickit_align_summary.py`)에는 그 가드가 없으므로 직접 셌다 — 명령 단계별로 hold 구간 중앙
`base_h` 가 0.15 m 미만인 env 를 "넘어짐"으로 본다:

```
                       cmd 2.0        cmd 2.5        cmd 3.0        cmd 3.5        cmd 4.0
                    넘어짐 / 달성  넘어짐 / 달성  넘어짐 / 달성  넘어짐 / 달성  넘어짐 / 달성
baseline              1.6 / 98.4    1.6 / 93.8    1.6 / 40.6    4.7 /  0.0    6.2 /  0.0
const LR              6.2 / 93.8    6.2 / 93.8    6.2 / 93.8    6.2 /  3.1    7.8 /  0.0
policy tan-norm       9.4 / 85.9   10.9 / 85.9   10.9 /  0.0   10.9 /  0.0   10.9 /  0.0
disc tn + rotfix      9.4 / 90.6   10.9 / 87.5   10.9 / 28.1   10.9 /  0.0   12.5 /  0.0
```

★ **판정은 바뀌지 않는다.** 낙상 비율이 5~12% 로 낮고 명령에 따라 거의 평평한 반면, `cmd 3.5`·`4.0`
실패의 **87~95% 는 "서 있는데 명령을 못 내는" env** 다. §18-b 에서 걸음 품질(`base_h` 0.265,
flip 10.1 Hz)로 "무너진 게 아니라 안 달린다"고 한 것이 개체 수로도 확인된다.

⚠ 다만 **저속 낙상률에는 실제 차이가 있다** — baseline 1.6% 대 두 tan-norm arm 9.4~10.9%
(`cmd 2.0`~`2.5`). 재현 산포(±8~15%p)와 겹치는 크기라 단정하지 않지만, tan-norm arm 이
저속에서 덜 튼튼하다는 방향은 두 arm 에서 같다. 다음 arm 에서 같이 볼 것.

---

## 19. (2026-09-01) 참조 데이터 속도 분포 실측 — **"공백" 이 아니라 "표본이 없다시피 하다"**

### 19-a. 앞서 쓴 "2.70~3.96 m/s 공백" 은 **틀렸다**

`smr_mirror_pkl` 18 클립 2646 프레임(60 fps, 44.1 s)의 평면 속력을 직접 쟀다.
5 프레임(83 ms) 이동평균 후 분포는 이렇다.

```
구간 [m/s]        비율     프레임
[0.00, 0.50)    25.43%      650
[0.50, 1.00)    19.41%      496
[1.00, 1.50)     6.81%      174
[1.50, 2.00)    22.54%      576
[2.00, 2.50)     8.29%      212
[2.50, 2.70)     3.91%      100
[2.70, 3.00)     6.34%      162      <- "공백" 이라고 썼던 구간에 162 프레임이 있다
[3.00, 3.50)     2.82%       72
[3.50, 3.96)     1.96%       50
[3.96, 4.50)     2.50%       64
전체 최대 속력 = 4.440 m/s
```

**속도는 있다.** 최대 4.44 m/s 까지 존재한다. 문제는 존재 여부가 아니라 **양**이다 —
`>= 3.0` 이 7.28 %(186 fr = 3.1 s), `>= 3.5` 가 4.46 %(114 fr = 1.9 s).

### 19-b. 3.2 m/s 위쪽은 **클립 하나, 1.02 초**가 전부다

```
클립               3.2 m/s 이상 프레임      최대 속력
go2_run2                61 fr (1.02 s)      4.440 m/s
go2_run2_mirror         61 fr (1.02 s)      4.440 m/s
(그 외 16 클립)             0 fr              run0 3.151 · run1 3.160
```

`run0`·`run1` 은 3.16 m/s 에서 멈춘다. 그러니 **Go2 가 3.2 m/s 위에서 어떻게 움직이는지에 대한
증거 전체가 `go2_run2` 71 프레임(1.18 s)** 이고, mirror 는 좌우만 바꾼 것이라 속력 분포가
완전히 동일해 **속도 다양성을 하나도 더하지 않는다**.

### 19-c. ★ `go2_trot0` 과 `go2_trot1` 은 **바이트 단위로 같은 파일이다**

```
go2_trot0        == go2_trot1          shape (208,18)  최대 절대차 0.000e+00
go2_trot0_mirror == go2_trot1_mirror   shape (208,18)  최대 절대차 0.000e+00
```

이름만 다른 같은 모션이다. 넷을 합치면 832 프레임 = **전체의 31.4 %** 이고,
그중 절반(416 fr = **15.7 %**)은 순수 중복이다.

즉 데이터셋은 예산의 **15.7 % 를 trot 복사본**에 쓰고,
3.2 m/s 이상 유일한 증거에는 **2.7 %**(71 fr)를 쓴다.

### 19-d. 가설 — 우리가 매번 닿는 1.7 m/s 는 **데이터셋의 최빈 모드**다

| | 값 |
|---|---|
| `go2_trot0` 평균 속력 | **1.686 m/s** |
| `go2_trot0` 중앙 속력 | **1.758 m/s** |
| style 0.5 arm 7 개가 수렴한 median vx | **~1.7 m/s** |
| trot 계열이 차지하는 프레임 비율 | **31.4 %** |

style 항이 데이터셋 최빈 모드로 끌어당긴다면 **"항상 trot" 과 "항상 ~1.7 m/s" 가 한 번에**
설명된다. §11~§12 에서 style 을 약화(`lerp 0.8`)했을 때만 4 m/s 가 열린 것과도 부합한다.

★ **이건 아직 가설이다.** 확인하려면 최소 두 가지가 필요하다:
1. `go2_trot1`(중복) 제거 후 재학습 — trot 비중 31.4 % → 18.3 % 로 떨어진다. 단일 변수다.
2. 속도 조건부 샘플링 또는 고속 클립 가중 — 지금은 `run2` 가 프레임 비율대로면 2.7 % 만 뽑힌다.

★ 반례 후보: 우리 학습 명령 상한은 4.0 m/s 인데 데이터 최대가 4.44 m/s 라 **명령 범위 자체는
데이터에 덮여 있다**. 따라서 "고속 참조가 아예 없어서 못 한다" 는 성립하지 않는다.

---

## 20. (2026-09-01) ★★★ 차이는 데이터가 아니라 **샘플링 가중치**다 — 파일은 md5 까지 같다

### 20-a. 먼저 §19-c·§19-d 를 정정한다

`go2_trot0` 과 `go2_trot1` 이 같은 파일인 것은 맞지만 **우리 결함이 아니다.**
MimicKit 원본에도 같은 중복이 있고, 파일은 바이트 단위로 동일하다.

```
             ours md5      MimicKit md5
go2_trot0    ef6128c6c910  ef6128c6c910
go2_trot1    ef6128c6c910  ef6128c6c910   <- trot0 과 같은 해시
go2_run2     ee026d0409d5  ee026d0409d5
```

MimicKit 은 이 중복을 안은 채로 4 m/s 를 낸다. 따라서 **중복 자체는 우리와 MimicKit 을
가르는 변수가 아니다.** §19-d 의 "1.686 m/s = 최빈 모드" 가설도 이 가중치 정정 없이는
성립하지 않는다(§20-c 에서 다시 계산한다).

### 20-b. 진짜 차이 — 한쪽은 **클립 균등**, 한쪽은 **길이 비례**

두 구현 모두 `torch.multinomial(self._motion_weights, ...)` 로 뽑는다. 가중치 정의가 다르다.

| | 가중치 출처 | 결과 |
|---|---|---|
| MimicKit | `dataset_go2_locomotion_smr_mirror3.yaml` 의 클립별 `weight: 1.0`<br>(`mimickit/anim/motion_lib.py:204`, 정규화는 `:158`) | **클립 균등** 1/18 = 5.56 % |
| 6.0 | `weights=None` → `w = motion_lengths`<br>(`motion_lib.py:307-311`) | **길이 비례** |

★ 원인은 호출부다 — `go2_imitation_tracking_env.py:144` 가
`Go2MotionLib(motion_files=..., device=...)` 로 **`weights` 를 아예 안 넘긴다.**
`MotionLib` 에는 `weights` 인자가 이미 있다(`motion_lib.py:259`). 배선만 빠졌다.

양쪽 모두 이 샘플링이 **AMP expert 배치와 RSI 리셋 둘 다**에 쓰인다
(우리 `env:864`·`env:981` / MimicKit `amp_env.py:30`·`deepmimic_env.py:278`).

MimicKit YAML 은 pace 6 클립을 `weight: 0.0` 으로 제외하는데, 우리 `smr_mirror_pkl` 에는
pace 파일 자체가 없다. **활성 클립 18 개가 정확히 일치한다.**

### 20-c. 정량 — 고속 클립이 **절반으로 깎인다**

```
그룹                      우리(길이비례)   MimicKit(균등)    배율
run2 (유일한 4.44 m/s)         5.33%          11.11%      2.09x
run 계열 전체                 25.34%          33.33%      1.32x
trot 계열 전체                31.51%          22.22%      0.71x
walk 계열 전체                43.15%          44.44%      1.03x
```

`run2` 는 1.167 s 로 가장 짧은 축인데 가장 빠르다. 길이 비례는 **짧을수록 벌한다** —
그래서 하필 유일한 고속 증거가 절반으로 깎인다. `run2 : trot` 상대비로 보면
**0.169 → 0.500 으로 약 3 배** 차이다.

★ 다만 과대평가하지 말 것: 클립 평균속력의 가중평균은
**1.416 → 1.572 m/s (+11 %)** 밖에 안 움직인다. 이건 평균의 이동이 아니라 **꼬리의 이동**이고,
discriminator 가 고속 구간을 몇 번이나 보느냐의 문제다.

### 20-d. 다음 arm — 단일 변수, 한 줄

`env:144` 에 클립 균등 가중치를 넘긴다. cfg 플래그(`motion_uniform_weights: bool = False`)로
기본값을 바꾸지 않고 arm 만 켠다. §19-d 가 맞다면 trot 쏠림이 31.5 → 22.2 % 로 줄면서
`cmd 3.0` 이상이 움직여야 하고, 틀렸다면 아무 변화가 없다 — 어느 쪽이든 판정이 된다.

---

## 21. (2026-09-02) 클립 균등 샘플링 arm — **iter 26k 중간 관측, 판정 아님**

`motion_uniform_weights` 플래그를 만들어(§20-d) 대조·처치 두 run 을 09-01 15:10 에 동시 착수했다.
09-02 09:00 기준 대조 29.8k / 처치 26.0k 진행 중이라, **둘 다 가진 26000 으로 맞춰** 램프를 걸었다.

| | run | `motion_uniform_weights` |
|---|---|---|
| 대조 | `2026-09-01_15-10-54_rotfixonly_stock` | false |
| 처치 | `2026-09-01_15-10-59_uniformw_stock` | **true** |

나머지 전부 동일(`use_pace_params=false`, tan-norm 둘 다 off, 4096 env). 램프는 64 env · `--no_pace`.

★ **대조군이 따로 필요한 이유**: 기존 baseline(`implicit_stock`, 08-10) 이후 `motion_lib` 회전
수정이 소스에 들어가, baseline 과 직접 비교하면 변수가 둘이 된다.

### 21-a. 결과 — 처치군이 **고속에서 오히려 뒤진다**

```
                           arm |    0.5         1.0         1.5         2.0         2.5         3.0         3.5      4.0
  control: length-proportional | 0.353(91%) 0.671(91%) 1.112(91%) 1.398(91%) 1.619(83%) 1.697(52%) 0.259(3%) 0.028(0%)
   treatment: uniform per clip | 0.389(88%) 0.624(94%) 1.071(94%) 1.365(94%) 1.468(83%) 0.657(25%) 0.295(0%) 0.087(0%)
```

낙상 비율(`base_h < 0.18 m`)은 반대 방향이다 — 처치가 **전 구간 4.7 % 로 평탄**,
대조는 6.2 % 에서 `cmd 4.0` 10.9 % 까지 오른다.

보행 종류는 **둘 다 전 구간 trot**. 전이는 없다.

### 21-b. 셋업 검증 — 대조군이 baseline 을 재현한다

`cmd 3.0` 에서 대조군 **1.697(52 %)** 는 §15-b 의 baseline **1.686(58 %)** 와 사실상 같다
(달성률 차 6 %p 는 재현 산포 ±8~15 %p 안). 회전 수정이 들어간 현재 소스가 baseline 을 그대로
재현한다는 뜻이고, §18 의 null 결과가 **disc tan-norm 이 아니라 rotfix 쪽에서 온 게 아님**을
26k 시점에서 뒷받침한다(40k 에서 재확인 필요).

### 21-c. 읽는 법 — **판정이 아니다**

- iter 26000 이다. 이 시리즈에서 24k 이하 차이가 48k 이후를 예측한 적이 없다.
- 학습 지표도 같은 방향을 가리킨다(25k): 처치가 `ep_len` 961 vs 946, `mean_noise_std`
  0.171 vs 0.196 으로 **더 안 넘어지고 더 확신**하는데, `amp_reward` 는 34.2 vs 38.9 다.
  ★ `amp_reward` 하락은 성능 저하가 아니다 — **처치군은 expert 분포 자체를 바꿨으므로**
  두 run 의 style 보상은 같은 자로 잰 값이 아니다.
- 지금까지의 그림은 "고속을 얻는" 게 아니라 **"안정을 얻고 고속을 내준"** 쪽이다.
  §19-d 가설(최빈 모드가 속도를 정한다)이 맞다면 방향이 반대다.

산출물: `figures/uniformw_arm_26000.png` · `metrics/uniformw_arm_26000.md` ·
`videos/rampvid_{rotfixonly,uniformw}_26k_chase/`

---

## 22. (2026-09-02) ★★★ 명령 재샘플이 **죽어 있다** — 램프는 학습에 없던 조건을 재고 있었다

§21 을 정리하다 `episode_length_s` 와 명령 재샘플 주기를 MimicKit 과 맞춰 보면서 찾았다.

### 22-a. 코드 사실

`_post_physics_step` 은 `go2_imitation_tracking_env.py:303` 에 정의돼 있고 안에서 `_tar_timer`
를 깎아 `_resample_steering` 을 부른다. 그런데 **이 메서드를 호출하는 곳이 저장소 어디에도 없다.**
`DirectRLEnv` 에 그런 훅이 없기 때문이다.

`grep -rn "_post_physics_step" --include=*.py .` 의 호출부는 둘뿐이고 둘 다 자기 자신을 부르는
다른 env 다:

- `R_Skeleton/skeleton_wtw_env.py:276` — 주석: "6.0 에서 `DirectRLEnv.step()` 이 더 이상 부르지 않음"
- `go2_pedipulation/go2_pedipulation_env.py:817` — 같은 우회

`leg_imitation_tracking` 은 이름을 바꿔 살렸다. ★ **`go2_imitation_tracking` 만 안 고쳐졌다.**

따라서 `tar_change_time_min/max` = 2.0~7.0 s 는 **dead config** 이고, 각 env 는 리셋 때 뽑은
명령 하나를 **20 초 에피소드 내내** 유지한다.

### 22-b. MimicKit 은 살아 있다 — 대조

`sim_env.py:180` `self._update_misc()` → `task_tracking_mixin.py:100` `_update_task()` →
`_reset_task()` 경로가 정상 동작한다.

| | 6.0 (우리) | MimicKit |
|---|---|---|
| 에피소드 길이 | **20.0 s** | 10.0 s |
| 재샘플 주기 설정 | 2.0 ~ 7.0 s | 4.0 ~ 7.0 s |
| 재샘플이 실제로 도는가 | **아니오 (dead hook)** | 예 |
| 에피소드당 명령 변경 횟수 | **0** | 약 1.2 |
| 에피소드당 서로 다른 명령 수 | **1** | 약 2.2 |

MimicKit 기댓값: 첫 변경은 항상 들어온다(최대 7 s < 10 s). 두 번째가 들어올 확률은 두 균등변수
합이 10 s 이하일 확률 = 2/9 = 0.22.

### 22-c. 왜 중요한가 — 그리고 **단정하면 안 되는 이유**

`speed_ramp_record.py` 는 명령을 0 → 4 → 0 으로 **계속 바꾸며** 잰다. 그런데 우리 정책은 학습
중 명령이 바뀌는 상황을 **한 번도 겪지 않았다.** 램프는 우리에게 **분포 밖**, MimicKit 에게는
분포 안 조건이다.

`cmd 2.5` 까지 83~94 % 를 달성하다 `cmd 3.0` 에서 급락하는 패턴이, "그 속도를 못 내는 것"인지
"속도가 바뀌는 것을 못 따라가는 것"인지 **지금 데이터로는 구분되지 않는다.**

★ 이것이 속도 천장의 원인이라고 단정할 근거는 아직 없다. 분리에 두 가지가 필요하다.

1. 훅 복구 재학습 — `go2_pedipulation_env.py:817` 과 같은 한 줄. 단일 변수다.
2. 램프를 **명령 고정** 조건으로도 재기 — 지금 수치 중 얼마가 전이 실패인지 분리한다.
   (2 는 학습 없이 지금 당장 할 수 있고, 1 보다 먼저 해야 한다.)

★★ **§1~§21 의 판정이 전부 이 조건에서 이뤄졌다.** 훅을 살리면 축적된 arm 비교를 재측정해야
한다 — 틀린 게 아니라 **잰 조건이 좁다**([[project_leg_tar_change_time_dead_config]] 와 같은 교훈).

---

## 23. (2026-09-03) ★★★ 명령 재샘플 복구 → **`cmd 3.5` 가 열렸다** · 그리고 램프가 재현되지 않았다

§22 의 dead hook 을 `_get_dones` 에서 부르도록 고치고 두 arm 을 재학습했다
(`cmdlive_stock` / `cmdlive_uniformw_stock`, 09-02 09:46 착수, `push_robot=false` 로 고정해
**명령 재샘플 하나만** 바꿨다). iter 30000 램프 결과다.

### 23-a. 수정이 먹었다는 실측

| | 수정 전 | 수정 후 |
|---|---|---|
| 8 s 안에 명령이 바뀐 env (512 env) | 0 | **511 / 512 (99.8 %)** |
| `_push_robots` 호출 (128 env, 12 s) | 0 회 | **97 회 / 201 env-push** |

### 23-b. 천장이 열렸다

```
세대 / 대조군              cmd 2.5      cmd 3.0      cmd 3.5      cmd 4.0
cmdfixed  @26k          1.619(83%)   1.697(52%)   0.259( 3%)   0.028( 0%)
cmdlive   @30k (seed 0) 1.943(64%)   2.816(62%)   3.488(61%)   1.953(48%)

낙상 %                    0.5    1.0    1.5    2.0    2.5    3.0    3.5    4.0
cmdfixed  @26k          6.2    7.8    7.8    7.8    7.8    7.8    7.8   10.9
cmdlive   @30k          7.8    9.4   28.1   32.8   35.9   35.9   39.1   39.1
```

`cmd 3.5` 가 **0.259 → 3.488 m/s**(13 배). 대가는 **낙상 7.8 → 39.1 %**(5 배)와
저속 달성률 91 → 52~70 % 하락이다. iter 는 26k vs 30k 로 다르지만 13 배를 4k 로 설명할 수 없다.

### 23-c. ★★ 램프가 재현되지 않는다 — 시드 부재

같은 체크포인트를 세 번 쟀다(A·B 무시드, C seed 0).

```
대조군              cmd 3.0      cmd 3.5      cmd 4.0     낙상@3.5
A (무시드)        2.849(69%)   3.488(66%)   2.429(52%)    29.7%
B (무시드)        2.769(59%)   3.455(59%)   0.587(36%)    37.5%
C (seed 0)       2.816(62%)   3.488(61%)   1.953(48%)    39.1%

처치군              cmd 3.0      cmd 3.5      cmd 4.0     낙상@3.5
A (무시드)        2.622(55%)   3.196(55%)   2.116(50%)    45.3%
B (무시드)        2.685(66%)   3.359(64%)   3.800(62%)    34.4%
C (seed 0)       2.644(58%)   3.179(58%)   3.704(58%)    42.2%
```

★ `cmd 3.5` 까지는 median 이 ±0.16 으로 안정적이지만 **`cmd 4.0` 은 무의미하다**
(대조군 0.587~2.429, 1.84 m/s 폭). 낙상률도 같은 arm 에서 29.7~39.1 %p 로 흔들린다.
env 별 궤적은 아예 다른 개체가 된다 — `env25` 가 한 실행에선 낙상, 다른 실행에선 4.037 m/s 완주.

`speed_ramp_record.py` 에 `--seed` 를 추가했다. 1 차·2 차 npz 의 `vx` 전체 최대 절대차가
**정확히 0.0** 이라 완전히 재현된다.

★★ **앞으로 arm 비교는 시드 고정 + 최소 2 회 반복.** §1~§22 의 판정도 단일 실행 기반이라
`cmd 4.0` 근처 수치는 이 산포를 감안해 다시 읽어야 한다.

### 23-d. `motion_uniform_weights` 는 아직 판정 불가

3 실행을 모으면 `cmd 3.5` 는 대조가 일관되게 ~0.2 m/s 높지만(3.488/3.455/3.488 vs
3.196/3.359/3.179), `cmd 4.0` 은 순서가 뒤집히고(A 대조 우세, B·C 처치 우세) 낙상도
일관성이 없다. **arm 간 차이가 실행 간 산포와 같은 크기다.**

### 23-e. 램프 자체의 결함 하나를 더 고쳤다

훅이 살아나면서 램프가 `env.step()` 직전에 써넣는 `_lin_vel_cmd` 를 **step 안에서 재샘플이
덮어쓰게** 됐다. 정책은 램프의 의도가 아닌 랜덤 명령을 보고, npz 에는 의도값이 기록돼
조용히 틀린 측정이 된다. `--free_cmd` 가 아니면 `tar_change_time` 을 1e9 로 밀어 끄고 로그를 찍는다.
★ 이 보고서의 모든 램프는 그 수정 이후 값이다.

또 `--cam_env` 를 추가했다 — 낙상률이 39 % 대라 env0 이 초반에 넘어져 영상이 "누워 있는
로봇"만 보여주는 일이 잦다. 개체 선택 기준은 **모집단 중앙 프로파일과의 L1 거리 최소**이고,
시드를 고정해야 표의 개체와 영상의 개체가 같은 로봇이 된다.

산출물: `figures/uniformw_arm_cmdlive_30000.png` · `metrics/uniformw_arm_cmdlive_30000.md`(seed 0) ·
`metrics/uniformw_arm_cmdlive_30000_runA.md`(무시드 A) · `videos/rampvid_cmdlive{,_uniformw}_30k_chase/`

---

## 24. (2026-09-03) ★★★ **`cmd 4.0` 이 style 0.5 에서 열렸다** — MimicKit 조건 재현

40k 에서 학습을 끊고(근거 §23-c 및 40k 이후 이득 실측) **시드 0·1 두 번씩** 램프를 걸었다.

### 24-a. 결과

```
                        arm / seed |    2.5         3.0         3.5         4.0
 control (length-proportional) / 0 | 2.074(72%) 2.836(72%) 3.551(66%) 0.290(19%)
 control (length-proportional) / 1 | 2.035(70%) 2.815(70%) 3.551(66%) 0.291(17%)
  treatment (uniform per clip) / 0 | 1.868(58%) 2.642(58%) 3.286(56%) 3.703(56%)
  treatment (uniform per clip) / 1 | 1.973(69%) 2.716(69%) 3.385(67%) 3.793(62%)
```

**`cmd 4.0` 에서 대조 0.29 vs 처치 3.75.** 차이 3.457 m/s 인데 시드 폭은 대조 0.001 · 처치 0.091 로,
**산포의 38 배**다. 두 시드가 같은 방향을 가리키므로 잡음이 아니다.

대신 `cmd 0.5~3.5` 는 대조가 일관되게 0.04~0.22 m/s 앞선다. **고속을 얻고 저·중속을 조금 내준 것**이고,
§19-d 가설(고속 클립 노출을 늘리면 고속이 열린다)과 방향이 맞는다.

### 24-b. 이게 왜 중요한가 — style 0.5 에서 `cmd 4.0` 이 열린 것은 처음이다

지금까지 `task_reward_lerp` 0.5(baseline 혼합) arm 은 **전부** `cmd 4.0` 에서 0 % 였다.

```
                       arm |      vx    %
    baseline (adaptive LR) |   0.049    0
         vel_err_scale 1.0 |   0.213    0
                  push off |   0.058    0
             vel1 + nopush |   0.214    0
 const LR (schedule=fixed) |   0.131    0
        tan-norm joint obs |   0.125    0
    disc tan-norm + rotfix |   0.033    0
       lerp 0.8 (style 약화) |   3.937   97   <- 유일한 성공, 단 style 을 약화한 조건
```

`lerp 0.8` 은 style 을 약화해 얻은 것이라 MimicKit 조건이 아니었다. **처치군은 style 0.5 를
유지한 채 3.70~3.79 m/s** 를 낸다 — MimicKit 이 4 m/s 를 내는 그 혼합이다.

### 24-c. 실기 불가 모드가 아니다 — 걸음 품질 확인

과거 `noamp` arm 이 3.589 m/s 를 냈지만 `base_h` 0.18 · 34 Hz 진동 포복이었다. 그래서 속도만으로
성공을 선언하지 않는다.

```
   arm/seed   cmd      vx    %  base_h  h std  flipHz  cap%
  control/0   4.0   0.290   19   0.232  0.105    11.4     2
treatment/0   4.0   3.703   56   0.267  0.115    11.2     6
treatment/1   4.0   3.793   62   0.273  0.108    11.5     6
(참고) lerp08  4.0   3.937   97   0.279  0.072    12.7     7
```

`base_h` 0.267~0.273 은 **기립 범위(0.27~0.32) 하단**이고, `flipHz` 11 대는 정상이다(병리 기준 30+).
토크 포화도 6 % 로 낮다. `lerp 0.8` 의 품질과 사실상 같은 급이다.

### 24-d. 정리 — 두 변경이 **함께** 필요했다

| 변경 | 근거 | 연 것 |
|---|---|---|
| 명령 재샘플 복구 (§22·§23) | `_post_physics_step` 이 죽은 훅이었다 | `cmd 3.5` (0.259 → 3.55) |
| 클립 균등 샘플링 (§20) | MimicKit 은 dataset YAML 로 클립 균등, 우리는 길이 비례 | `cmd 4.0` (0.29 → 3.75) |

대조군도 명령 재샘플은 살아 있지만 `cmd 4.0` 에서 무너진다. 둘 다 있어야 4 m/s 가 나온다.
**두 변경 모두 MimicKit 과의 코드 대조로 찾은 정렬 결함**이지 새 알고리즘이 아니다.

### 24-e. 남은 것

- 달성률이 56~62 % 다. `lerp 0.8` 의 97 % 와는 거리가 있다 — 모드는 열렸지만 신뢰도는 낮다.
- 낙상이 여전히 31~44 % 다(§23-b 의 대가). 대조 27~34 % 보다 높다.
- `cmd 0.5~3.5` 손실(0.04~0.22 m/s)이 실기에서 문제인지 판단 필요.
- 시드 2 개는 **잠정**이다. 이 결론을 확정하려면 시드를 늘려야 한다 — 다만 `cmd 4.0` 의
  3.457 m/s 차이는 산포 38 배라 뒤집힐 여지가 거의 없다.

산출물: `metrics/seeded_ramps_40000.md` · `metrics/ramp_cmdlive{,_uniformw}_40000/seed{0,1}/`

---

## 25. (2026-09-03) 걸음 종류 — **참조에 gallop 이 없다.** 최고속 클립은 bound 다

§24 를 보고 "3.7 m/s 인데 gallop 이 안 보인다" 는 지적이 나와 위상을 제대로 쟀다.
`logs/gait_classify.py` · `logs/gait_stride_vs_reference.py` 가 재현한다.

### 25-a. 먼저, §24 의 위상 패널이 비었던 이유

`plot_uniformw_arm.py` 는 env 간 위상 일치도 `R > 0.75` 를 요구한다. 낙상이 27~44 % 라
대부분의 점이 그 필터에 걸렸다. **낙상하지 않고 명령을 따라가는 env 만 골라 per-env 로
분류하면** 멀쩡히 나온다 — 걸음이 없었던 게 아니라 **집계 방식이 가렸다.**

### 25-b. 참조 모션의 걸음 (thigh 위상차)

```
클립              vmean   FL-RR   FL-FR   FL-RL   분류
go2_run2 (최고속)  3.81   0.442   0.121   0.580   bound
go2_run1           2.54   0.671   0.771   0.457   gallop/other
go2_run0           2.27   0.588   0.749   0.404   gallop/other
go2_trot0          1.67   0.061   0.498   0.555   trot
go2_walk2          0.56   0.275   0.497   0.811   gallop/other
```

★ **3.2 m/s 위를 담은 유일한 클립 `go2_run2` 가 bound 다.** 데이터셋에 gallop 이 없으므로
정책이 gallop 을 낼 이유가 없다. §19-b 에서 "고속 증거가 이 클립 하나" 라고 한 것의 걸음이 이것이다.

⚠️ 처음엔 `run2` 가 pace 로 나왔다 — pkl `DOF_NAMES` 가 다리별 `[hip, thigh, calf]` 순서인데
(`motion_lib.py:46`) hip 을 thigh 로 읽었기 때문이다. 위 표는 정정한 값이다.
**참조 pkl 을 열 때 관절 순서를 IsaacLab 순서(`hip×4, thigh×4, calf×4`)로 가정하지 말 것.**

### 25-c. 정책의 걸음 — 대조는 trot 에 갇혔고, 처치는 bound 를 배웠다

40k · seed 0 · 낙상 안 하고 명령의 절반을 넘는 env 만:

```
       arm  cmd   적격 env   분류
   control  2.5   46/64   trot 42(91%)  gallop/other 4(9%)
   control  3.5   42/64   trot 42(100%)
   control  4.0   12/64   trot 12(100%)
 treatment  2.5   37/64   bound 15(41%)  trot 15(41%)  other 7(19%)
 treatment  3.5   36/64   bound 21(58%)  trot 15(42%)
 treatment  4.0   36/64   bound 21(58%)  trot 15(42%)
```

**대조군이 `cmd 3.5` 에서 막힌 이유가 이것이다 — 100 % trot 이라 거기가 trot 의 상한이다.**
처치군은 클립 균등 샘플링으로 `run2` 노출이 2 배가 되면서 그 bound 를 배웠고, 그래서 4.0 이 열렸다.

### 25-d. 위상은 맞는데 **템포가 25 % 빠르다** — "어색함" 의 정체

```
                    FL-RR   FL-FR   FL-RL   보폭 주파수   보폭 길이(vx/f)
참조 go2_run2       0.442   0.121   0.580     2.41 Hz      1.58 m
정책 bound (n=21)   0.478   0.127   0.586     3.02 Hz      1.23 m
정책 trot  (n=15)   0.997   0.520   0.533     3.54 Hz      1.05 m
```

위상 세 개가 참조와 거의 일치한다(최대 차 0.036). **그런데 보폭 주파수가 3.02 vs 2.41 Hz 로
25 % 빠르고, 보폭 길이는 22 % 짧다.** 같은 속도를 더 잘게 쪼개 내고 있어서 종종거리는 인상을 준다.

두 번째 요인은 **모집단이 갈려 있다**는 것이다 — `cmd 4.0` 에서 42 % 가 아직 trot 이고,
3.7 m/s 를 trot 으로 내는 것은 무리한 걸음이다(보폭 3.54 Hz · 1.05 m).

### 25-e. 함의

속도는 열렸지만 **걸음은 아직 참조와 다르다.** style 가중치 0.5 가 이미 보폭 정보를 담고 있는데도
템포가 25 % 어긋나 있다 — disc 가 보폭 주파수를 못 잡고 있을 가능성이 있다. 별도 조사가 필요하다.
낙상 31~44 % 도 이 두 가지(짧고 빠른 보폭 · 걸음 분열)와 무관해 보이지 않는다.

영상: `videos/rampvid_cmdlive_40k_chase/`(대조 env32, 100 % trot 쪽) ·
`videos/rampvid_cmdlive_uniformw_40k_chase/`(처치 env52, bound 쪽) — 둘 다 seed 0.

---

## 26. (2026-09-03) ★★★ 정정 — **낙상 31~44 % 는 정책이 아니라 램프의 push 였다**

§25 를 쓰다 `pace08_vs_lerp06` README 와 대조하던 중, 그쪽은 **런타임 외란을 끄고** 쟀는데
우리 40k 램프는 **켠 채**로 쟀다는 것을 발견했다. 같은 체크포인트를 `--no_push` 로 다시 쟀다.

```
       arm  push |        cmd 2.5            cmd 3.0            cmd 3.5            cmd 4.0
   control    ON | 2.074( 72%) 낙27%  2.836( 72%) 낙27%  3.551( 66%) 낙27%  0.290( 19%) 낙33%
   control   OFF | 2.121(100%) 낙 0%  2.889(100%) 낙 0%  3.584( 91%) 낙 0%  0.886( 39%) 낙 2%
 treatment    ON | 1.868( 58%) 낙42%  2.642( 58%) 낙42%  3.286( 56%) 낙44%  3.703( 56%) 낙44%
 treatment   OFF | 1.995( 94%) 낙 3%  2.757( 95%) 낙 5%  3.461( 95%) 낙 5%  3.914( 95%) 낙 5%
```

**낙상 31~44 % → 3~5 %. 달성률 56~58 % → 94~95 %. `cmd 4.0` 은 3.703 → 3.914 m/s (95 %).**

### 26-a. 왜 이런 일이 생겼나

램프는 `parse_env_cfg` 로 **현재 소스 cfg** 를 쓰고 `dr.push_robot` 기본값은 `True` 다.
두 arm 은 `push_robot=false` 로 학습했으므로 **학습에 없던 외란**을 맞은 것이다.

★ 그런데 이게 지금까지 드러나지 않은 이유가 있다 — 2026-09-02 에 `_post_physics_step` 을
살리기 전에는 **램프에서도 push 가 발사되지 않았다**(같은 죽은 훅 안에 있었다).
즉 훅을 되살린 순간, 학습뿐 아니라 **모든 램프 측정 조건이 같이 바뀌었다.**

### 26-b. 따라서 §23-b·§24-e 의 "대가" 서술을 정정한다

§23-b 는 명령 재샘플 복구의 대가를 "낙상 7.8 % → 39 %(5 배)" 로 적었다. **틀렸다.**
`cmdfixed` 세대 램프(26k, 09-02 09:07)는 훅 수정 **전**이라 push 가 없었고, `cmdlive` 세대
램프는 수정 **후**라 push 가 있었다. 두 세대의 낙상 차이는 학습 변경이 아니라 **측정 조건 차이**다.

`--no_push` 로 맞춰 보면 낙상은 3~5 % 로, `cmdfixed` 의 7.8 % 보다 오히려 **낮다.**
명령 재샘플 복구는 낙상을 늘리지 않았다.

### 26-c. 그러면 처치군은 `lerp 0.6` 급이다 — style 0.5 에서

```
축                        lerp 0.6 (08-14, 60k)   처치 uniformw (40k, no_push)
cmd 4.0 vx / 달성률          3.911 / 98 %            3.914 / 95 %
cmd 3.5 달성률               96 %                    95 %
cmd 2.5 달성률               72 %                    94 %
낙상                        1 / 64                  3~5 %
cmd 4.0 보행 · stride        bound · 3.00 Hz         bound · 3.02 Hz
task_reward_lerp            0.6 (style 약화)         **0.5 (baseline 혼합)**
```

★ 측정 조건은 여전히 우리 쪽이 **더 가혹하다** — `lerp 0.6` 은 런타임 외란 4 종(push·obs noise·
action delay·encoder bias)을 모두 끄고 명령당 10 s 를 줬고, 우리는 push 만 끄고 나머지는 켠 채 3 s 다.
그 조건에서 동급이 나온다.

### 26-d. 남은 것 — 걸음은 여전히 참조와 다르다

낙상이 해소돼도 §25-d 의 두 가지는 그대로다.

- 보폭 3.02 Hz vs 참조 2.41~2.54 Hz (**단, `lerp 0.6` 도 3.00 Hz 라 이건 이 정책군의 공통 성질이다**).
- `cmd 4.0` 걸음 분열 — push OFF 재측정에서 재분류가 필요하다(§25-c 는 push ON 데이터였다).

### 26-e. 램프 규약 — 앞으로

**학습 cfg 와 램프 cfg 의 `dr.*` 를 맞춰야 한다.** `push_robot=false` 로 학습했으면 램프도
`--no_push` 가 기본이다. push 를 켠 측정은 "학습에 없던 외란에 대한 강건성" 이라는 **별도 질문**이고,
그렇게 라벨해야 한다. `--no_pace` 와 같은 부류의 함정이다([[project_ramp_uses_source_cfg_not_run_params]]).

데이터: `metrics/ramp_cmdlive{,_uniformw}_40000_nopush/seed0/`

---

## 27. (2026-09-03) push OFF 로 걸음 재분류 — **분열은 진짜지만 손해가 아니고, 보폭은 MimicKit 보다 우리가 참조에 가깝다**

§26 에서 낙상이 램프의 push 였음이 드러났으므로, §25-c 의 걸음 분류를 `--no_push` 데이터로 다시 했다.

### 27-a. 분열은 push 아티팩트가 아니다

```
       arm  cmd   적격 env   분류
   control  3.5   58/64   trot 58(100%)
   control  4.0   25/64   trot 24(96%)  other 1(4%)
 treatment  3.5   61/64   bound 33(54%)  trot 28(46%)
 treatment  4.0   61/64   bound 33(54%)  trot 28(46%)
```

적격 env 가 36 → 61 로 늘었는데 비율은 58/42 → 54/46 으로 유지된다. **분열은 실재한다.**

### 27-b. 그런데 분열이 성능 손해가 아니다 — trot 쪽이 오히려 빠르다

```
  cmd 4.0 (push OFF)
    bound  n=33  vx 3.855  달성 100%  보폭 3.02 Hz  길이 1.28 m  base_h 0.301
    trot   n=28  vx 3.985  달성 100%  보폭 3.55 Hz  길이 1.12 m  base_h 0.282
    참조 go2_run2  vx 3.805              보폭 2.41 Hz  길이 1.58 m
```

**두 그룹 다 달성률 100 % 이고 trot 쪽이 더 빠르다.** "42 % 가 무리한 trot" 이라던 §25-d 서술은
틀렸다 — 속도로는 손해가 없다. 차이는 **참조 충실도**다.

정책은 속도를 **보폭 길이가 아니라 스텝 빈도로** 만든다 — 참조는 1.58 m 를 2.41 Hz 로 밟는데
정책은 1.12~1.28 m 를 3.02~3.55 Hz 로 밟는다. "종종거린다" 는 인상의 정체가 이것이다.

### 27-c. ★ 그런데 MimicKit 정책이 **더 심하다**

`_comparisons/mimickit_vs_pace08/metrics/mimickit/ramp_data.npz` 로 MimicKit 자체 정책을 쟀다.

```
  cmd 2.5  적격 64/64  trot 45(70%)  other 19(30%)   trot  보폭 2.82 Hz
  cmd 3.5  적격 64/64  bound 46(72%) other 18(28%)   bound 보폭 3.29 Hz
  cmd 4.0  적격 63/64  bound 51(81%) other 12(19%)   bound 보폭 3.68 Hz
```

```
보폭 @cmd 4.0     참조 2.41 Hz   우리 3.02 Hz   MimicKit 3.68 Hz
```

**우리 보폭이 MimicKit 보다 참조에 가깝다.** MimicKit 도 걸음이 갈린다(81/19). 즉 "짧고 빠른
보폭" 은 우리 정렬 결함이 아니라 **이 setup(AMP · 50 Hz 제어 · 0.2 s disc 창) 의 공통 성질**이다.

⚠️ MimicKit npz 에는 `stage` 키가 없어 단계를 **균등 분할로 근사**했다. 명령별 값이 단조 증가하는
것으로 보아 대체로 맞지만, 경계가 정확하지는 않다.

### 27-d. disc 창 길이는 정렬 결함이 아니다

```
                    ours              MimicKit
num_disc_obs_steps   10                10
control_freq         50 Hz             50 Hz
→ disc 창            0.20 s            0.20 s
참조 보폭 주기 0.415 s → 창이 담는 비율 48 % (양쪽 동일)
```

disc 가 참조 한 주기를 못 본다는 것은 사실이지만 **MimicKit 도 같다.** §25-e 에서 "disc 가 보폭
주파수를 못 잡는 우리 문제" 라고 쓴 것은 **정렬 축의 문제가 아니다** — 고치려면 MimicKit 정렬이
아니라 setup 자체(창 길이 ≥ 21 step, 또는 보폭 항 추가)를 바꿔야 하고, 그건 별개 연구다.

### 27-e. 우리가 MimicKit 에 뒤지는 축은 하나 — **걸음 일관성**

```
cmd 4.0 bound 비율     우리 54 %      MimicKit 81 %
```

속도·보폭·낙상에서는 동급이거나 우리가 낫다. 남은 격차는 정책이 한 걸음으로 수렴하지 않는다는 것이다.

---

## 28. (2026-09-03) ★ 좌우 비대칭 — **처치군이 가장 나쁘고, 참조 클립이 원인일 가능성**

"Hz 만 문제가 아니라 좌우 균형도 이상하다" 는 지적에서 출발해 쟀다.
램프는 `vy = 0` · `yaw = 0` 을 명령하므로 남는 측방·요 운동과 좌우 관절 사용 차이는 전부 비대칭이다.
`logs/gait_symmetry.py` 가 재현한다.

### 28-a. 실측 — 처치군이 대조·MimicKit 의 2 배

`cmd 3.5` / `cmd 4.0`, push OFF, 서 있는 env 만. ROM 좌우차 = `|L−R| / ((L+R)/2)` 의 중앙값.

```
                       cmd   |vy|    |yaw|   thigh ROM 좌우차   calf 좌우차   토크 좌우차
우리 처치 (uniformw)    3.5   0.132   0.239        36.5%          20.8%       26.5%
                        4.0   0.173   0.285        28.8%          16.4%       23.7%
우리 대조 (control)     3.5   0.144   0.240         7.2%          13.3%       13.0%
                        4.0   0.124   0.288        14.8%          16.3%       16.7%
MimicKit 정책           3.5   0.184   0.235        16.5%          14.6%       15.9%
                        4.0   0.188   0.258        16.8%          15.1%       14.3%
```

★ `|vy|`·`|yaw|` 는 셋이 비슷하다 — **몸통이 더 흐르는 게 아니라 다리 사용이 한쪽으로 쏠린다.**
`cmd 3.5` thigh 기준 처치 36.5 % vs 대조 7.2 % · MimicKit 16.5 % 로 **2 배 이상**이다.
§27 에서 "속도·보폭은 MimicKit 급" 이라 했지만 **이 축에서는 우리가 확실히 나쁘다.**

### 28-b. 참조 클립 자체가 비대칭이다

```
clip           길이    thigh 좌우차   calf 좌우차
go2_run2      1.18 s      14.0%         4.6%    <- 3.2 m/s 위 유일 클립
go2_run1      2.95 s      20.5%        31.8%
go2_run0      1.47 s      29.8%        15.9%
go2_trot0     3.47 s       7.1%        12.6%
go2_walk2     2.13 s       2.1%         8.0%
```

**빠른 클립일수록 비대칭이 크고, 느린 클립은 깨끗하다.** `go2_run2` 는 1.18 s — 보폭 2.41 Hz 기준
**약 2.8 주기**라 반주기 단위로 끊기지 않는다. 좌우가 불균등하게 샘플링된 셈이다.

`_mirror` 클립이 있어 **데이터셋 총합은 좌우 균형**이 맞지만, **개별 클립은 비대칭**이고
정책은 그 개별 패턴을 배운다.

### 28-c. 해석 — 정책이 참조의 비대칭을 **증폭**한다

- 대조군은 trot 에 머무르고 `go2_trot0`(7.1 %)을 주로 배운다 → 7~15 %. 참조 수준.
- 처치군은 클립 균등으로 `go2_run2`(14.0 %) 노출이 2 배 → **28~36 %**. 참조의 약 2 배.

즉 클립 균등 샘플링이 `cmd 4.0` 을 연 대가로 **가장 비대칭한 클립의 비중을 늘렸다.**
§24-a 의 "저·중속 0.04~0.22 m/s 손실" 외에 **이 축의 비용이 추가로 있다.**

★ 단 이것은 상관에서 온 추론이다. 확인하려면 `go2_run2` 를 정수 주기로 잘라 다시 학습하거나,
좌우 대칭 손실([[project_leg_imitation_symmetry_fix]] 의 mirror 접근)을 켜서 재야 한다.

### 28-d. 그래서 "보기에 안 좋다" 의 정체는 두 가지다

1. **짧고 빠른 보폭** — 1.12~1.28 m / 3.02~3.55 Hz vs 참조 1.58 m / 2.41 Hz.
   ★ 단 MimicKit(3.68 Hz)이 더 심하므로 우리 고유 결함이 아니다(§27-c).
2. **좌우 쏠림** — thigh ROM 좌우차 28~36 %, 대조·MimicKit 의 2 배. **이건 우리 고유 문제다.**

---

## 29. (2026-09-04) push 외란 fine-tune — **arm #8 의 첫 실측**, 외란 낙상이 사라진다

§26 에서 push 가 죽은 훅 안에 있어 한 번도 발사된 적이 없음이 드러났고, 그래서 기존 arm #8
`nopush_stock` 의 "무효과" 판정은 push-off 를 **push-없음**과 비교한 null-by-construction 이었다.
`cmdlive_stock` 40k 에서 이어받아 `dr.push_robot=true` 로 20k 를 더 돈 것이 이번 실측이다
(`push_ft_from_cmdlive`, 40000 → 60000).

### 29-a. 결과 (64 env · `--no_pace` · 시드 0·1)

```
                   run  sd |     cmd 2.5           cmd 3.0           cmd 3.5           cmd 4.0
 push_ft@60k  push ON    0 | 1.915( 98%) 낙 0%  2.631( 97%) 낙 0%  3.252( 86%) 낙 0%  0.985(38%) 낙5%
 push_ft@60k  push ON    1 | 1.921(100%) 낙 0%  2.645(100%) 낙 0%  3.251( 98%) 낙 0%  0.708(33%) 낙6%
 push_ft@60k  push OFF   0 | 1.905(100%) 낙 0%  2.585(100%) 낙 0%  3.154( 95%) 낙 0%  2.147(52%) 낙2%
 push_ft@60k  push OFF   1 | 1.940(100%) 낙 0%  2.610(100%) 낙 0%  3.215(100%) 낙 0%  0.889(38%) 낙0%
   부모@40k   push ON    0 | 2.074( 72%) 낙27%  2.836( 72%) 낙27%  3.551( 66%) 낙27%  0.290(19%) 낙33%
   부모@40k   push ON    1 | 2.035( 70%) 낙30%  2.815( 70%) 낙30%  3.551( 66%) 낙30%  0.291(17%) 낙34%
   부모@40k   push OFF   0 | 2.121(100%) 낙 0%  2.889(100%) 낙 0%  3.584( 91%) 낙 0%  0.886(39%) 낙 2%
```

### 29-b. push 학습은 **외란 낙상을 없앤다**

같은 push ON 조건에서:

```
낙상 @cmd 2.5~3.5      부모 27~30 %  →  push_ft  0 %
달성률 @cmd 3.5        부모 66 %     →  push_ft  86~98 %
```

두 시드가 같은 방향이고 크기가 크다. **arm #8 의 "무효과" 판정은 대체된다** — push DR 은
무효과가 아니라 애초에 **발사되지 않았을 뿐**이고, 실제로 켜면 큰 효과가 있다.

### 29-c. 대가는 속도 5~10 %

push OFF 조건(부모의 판정 조건)에서 비교하면:

```
cmd 2.5   부모 2.121  →  push_ft 1.905~1.940   (−9 %)
cmd 3.0   부모 2.889  →  push_ft 2.585~2.610   (−10 %)
cmd 3.5   부모 3.584  →  push_ft 3.154~3.215   (−11 %)
```

외란 없는 조건에서도 낙상은 0 % 로 같으므로 **잃은 것은 속도뿐**이다.

★ `cmd 4.0` 은 읽지 말 것 — 부모가 `cmdlive_stock`(길이 비례 대조군)이라 애초에 4.0 을 못 낸다
(§24). push_ft 도 그 성질을 물려받아 0.708~2.147 로 시드 간 3 배 흔들린다.

### 29-d. 혼입 — 남아 있다

`push_ft` 는 60k, 부모는 40k 라 **push 효과와 추가 20k iter 가 섞인다.** 다만:

- **낙상 27~30 % → 0 % 는 안전하게 귀속된다** — 부모의 외란 낙상은 학습에 없던 조건에서 온
  train/test 불일치이고, push 없는 데이터로 20k 를 더 돈다고 없어질 성질이 아니다.
- **속도 −9~11 % 는 덜 안전하다** — 40k 이후 이득이 작다는 실측(§23-c, 수렴 arm +0.01~0.03)을
  감안해도, 같은 40k 에서 push 없이 20k 더 도는 **대조 fine-tune** 이 있어야 확정된다. 미실시.

데이터: `metrics/ramp_pushft_59999_push{on,off}/seed{0,1}/`

# hindLeg_history_direct — 2족 8-DOF 보행 (foot↔calf 커플링 세대)

`HindLeg-Direct-v0` / `HindLegHistoryEnvCfg`. RGA 2족 8-DOF 다리의 평지 속도추종 보행.
2026-08-12~13 세션에서 **r2s_bridge 검증 플랜트(URDF3_SignFix)로 env를 개편**하고 학습을 돌린
기록이다. 아래 4개 런은 모두 같은 계보이며, 마지막 런만 유효한 baseline 후보다.

## 런 계보

| run | iter | 결과 | 비고 |
|---|---|---|---|
| `2026-08-12_18-17-46_signfix_coupled_realgains` | 0~10,004 | **NaN 파국** | value loss 발산 |
| `2026-08-12_23-14-14_signfix_coupled_v2_resume9700` | 9,700~10,378 | **NaN 파국** | grad clip 0.5로도 재발 |
| `2026-08-12_23-33-41_signfix_coupled_v3_clip` | 10,300~46,732 | 안정 완주했으나 **보행 실패** | 사용자 판단으로 중단 |
| `(중단) colmesh_restz_fix` | 0~662 | 12조각 충돌로 처리량 1/7 → 중단 | 학습 자체는 정상이었음 |
| `2026-08-13_10-55-36_colmesh_v2_restz_fix` | 0~462 | GPU/CPU 경합으로 중단 | model_400 에서 이어감 |
| `2026-08-13_12-04-06_colmesh_v2_gpu2` | 400~**50,399 완주** | ★**성공 — 구 플랜트 최고 기록 경신** | **현행 baseline** |

## ★ 최종 결과 (`model_50399`, 50,400 iter / 15.1 시간 완주)

`_sole_rest_z` 를 고친 뒤 학습은 구 플랜트 성공 run 을 **4배 빠른 iter 로 따라잡고**(5,962 iter 에
구 23,000 iter 수준) 끝까지 그 위를 유지했다.

| 학습 지표 | 구 플랜트 성공 run @ 23,115 iter | 현재 @ 5,962 iter | 현재 @ **50,399 iter** |
|---|---|---|---|
| `track_lin_vel_xy_exp` | 0.5971 | 0.5793 | **0.7616** (+28%) |
| `gait_swing` | 0.7530 | 0.7381 | 0.7223 |
| `gait_stance` | 0.9582 | 0.9517 | 0.9551 |
| episode length | 986.5 | 974.5 | 981.9 |
| mean reward | 35.18 | 42.73 | **47.63** (+35%) |

### 보행 실측 (`hindleg_gait_probe.py`, 64 env, 250 step)

| 명령 vx | 실제 vx | 달성률 | 양발 접지 | 좌우 교대 (HL/HR) | 유격 | swing 리프트 | \|τ\| p99 | \|q̇\| p99 |
|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.487 | **97%** | 10.1% | 44.8 / 44.7 | 45.1% | 89.5 mm | 19.4 | 10.5 |
| 1.0 | 0.966 | **97%** | 7.8% | 45.5 / 46.2 | 46.4% | 93.1 mm | 23.0 | 11.6 |
| 1.5 | 1.426 | **95%** | 5.9% | 45.7 / 47.8 | 47.4% | 91.4 mm | 28.7 | 11.7 |
| 2.0 | 1.810 | **90%** | 3.5% | 47.3 / 48.0 | 48.9% | 96.6 mm | 33.8 | 13.0 |

- **좌우 지지 시간이 44.7~48.0% 로 균등** — 대칭 보행 확보(iter 900 에서는 14% vs 51% 비대칭이었다)
- 양발 동시 접지 3.5~10.1%, 공중 유격 45~49% — 실패 런(99.3% 양발 접지, 미끄러짐)과 정반대
- 명령이 빠를수록 유격이 늘고 base 가 낮아진다(0.575 → 0.533 m) — 속도에 맞춰 자세를 바꾼다

### 실기 이식 관점

토크 p99 19~34 N·m 로 effort limit(84/84/126/100.8)에 크게 못 미치고, 관절속도 p99 10.5~13.0 rad/s 도
velocity limit(29.6/29.6/19.7/24.6) 안이다. 다만 **실기 런타임 보호(보고토크 15 N·m 50 ms → limp 래치,
200 dps 속도 트립)는 이보다 훨씬 낮으므로** cmd 1.5 이상 구간은 실기에서 트립할 수 있다.
배포 전 이 구간의 토크 프로파일을 별도로 검토할 것.

**★판정 시점 주의**: iter 900 스냅샷에서는 전진 3%(제자리걸음)라 "학습 실패"로 오판하기 쉽다.
구 플랜트 run 도 step 900 에서 `track_lin` 0.2972 였다. **반드시 같은 iter 시점끼리 비교할 것.**

### 영상

`videos/` 아래 (renderer: `scripts/tools/report_video.py`)

| 파일 | 시점 | 내용 |
|---|---|---|
| `model_800__iter800_colmesh_restz__*.mp4` | iter 800 | 웅크린 자세, 발 들기 시작 |
| `model_900__latest_walking__*.mp4` | iter 900 | 제자리걸음(전진 3%) |
| `model_6000__iter5900_walking__*.mp4` | iter 5,900 | 걷기 시작(cmd 1.0 → 95%) |
| **`model_50399__FINAL_50399__*.mp4`** | **iter 50,399** | **최종 — 직립 보행, cmd 2.0 → 90%** |

## env 개편 내역 (미커밋)

### 1. foot↔calf 전달기구 커플링을 학습 액추에이션에 이식

실기 foot 모터는 관절각이 아니라 **raw각 `q_raw = q_foot + q_calf`** 를 구동한다
(`RL_INTERFACE` coef=+1, 비가역 전달기구). `r2s_biped_leg` live 모드에서 검증된 모델을
`hind_leg_env.py::_apply_action` (200 Hz) 에 그대로 이식했다.

- 정책은 종전대로 8관절 **관절각** 목표를 내고, env 가 foot 의 raw 목표를 `q_f_des + q_c_des` 로 합성
- foot 위치목표 `raw_t − q_calf`, 속도목표 `−q̇_calf` 로 치환 → actuator PD 가 raw 공간 오차를 계산
- 전치 토크 `τ_calf += τ_foot_motor` (effort limit 클램프)
- 게인/토크한계는 DR 이 스케일한 `actuator.stiffness` 텐서를 실시간 참조 (상수 캐시 금지)

**배포 시 대응 필요**: real_runner 가 foot 모터 목표를 같은 식으로 합성해야 하고,
실기 TELEM 의 foot 은 raw 각이므로 정책 obs 에 넣기 전 `q_foot = raw − q_calf` 로 되돌려야 한다.

### 2. PD 게인을 실기 운용값으로 정합

hip 100/5, thigh 50/5, calf 50/5, foot 20/5 (`motions.py::DEFAULT_KP/KD`,
`real_runner/calib_bipedleg.hpp` 와 동일). 종전 `rga.py` 값(hip kd 6.0 등)은 실기 드라이버의
kd 클램프 `[0, 5]` 를 넘어 그대로 배포할 수 없었다. `rga.py` 원본은 두고 env cfg 에서 override.

### 3. 새 플랜트 호환 수정

URDF3 부터 `base` → `base_collision` 으로 리네임됐다. `find_bodies` 는 `re.fullmatch` 라
`"base"` 가 매치되지 않아 termination/penalty/DR 이 모두 깨진다. 3곳을 `base.*` 로 교정.

### 4. ★ 중첩 링크 prim 의 contact sensor 전멸 (신규 발견)

새 UrdfConverter 자산은 링크 prim 이 운동학 트리 그대로 **중첩**된다
(`Robot/Geometry/base_collision/HL_hip_link_collision/...`). 코어
`sim/schemas/schemas.py::activate_contact_sensors` 는 "nested rigid bodies 는 SDK 가 금지"라는
가정으로 **첫 rigid body 에서 하강을 멈춰**, base 에만 `PhysxContactReportAPI` 가 붙는다.
결과: contact sensor 가 `Available strings: ('base_collision',)` 로 죽는다.

fix = env `_setup_scene` 에서 스폰 직후(클론 전) 서브트리 전체를 걸어 rigid body 마다
`activate_contact_sensors` 재호출 (`HindLegEnv._activate_nested_contact_report`).
**새 컨버터 자산에 contact sensor 를 쓰는 다른 env 도 같은 조치가 필요하다.**

## ★★ value loss 파국 — 두 번 재현 후 처방

`signfix_coupled_realgains`(iter 10,004)와 `v2_resume9700`(iter 10,378)이 같은 방식으로 죽었다.

**시그니처**: iter ~9,748 부터 산발 스파이크(수백~5e4, 1 iter 만에 자가 회복) → 어느 순간
**매 iter ×300 단조 증가로 전환** → `inf` → `NaN`.
★결정적: **발산하는 동안 reward·episode length 는 완전 정상**(34~35 / 999). advantage 정규화가
actor 를 보호하는 사이 critic 만 죽는다.

**기전**: 희귀 물리 폭주 이벤트(iter 10,276 에서 mean reward −333.92 실측)의 극단 obs/reward 를
critic 이 학습 → 오염된 V 가 **GAE bootstrap(`returns = A + V`)을 타고 다음 rollout 의 target 으로
전파** → 지수 발산. `max_grad_norm` 0.5 는 지연만 시켰다(v2 에서 재발).

**처방** (v3 에서 유효):
- `hind_leg_env.py`: obs clamp ±100 (policy + priv_explicit), step reward clamp ±10 (cfg `obs_clip` / `reward_clip`)
- `agents/rsl_rl_ppo_cfg.py`: `max_grad_norm` 1.0 → 0.5

v3 는 이 조합으로 36,000 iter 를 무발산 완주했다.

**모니터링 기준**: value loss 가 수백이어도 1 iter 만에 회복하면 관망,
**연속 2 iter 이상 단조 증가면 즉시 개입**(체크포인트가 100 iter 간격이라 빠른 kill 이 손실 최소).

## ★★★ v3 는 안정적이었지만 **걷지 않았다** — 원인 규명

36k iter 를 완주하고 reward 35 / episode length 995 로 모든 학습 지표가 건강했다. 그런데
`Episode_Reward/gait_swing` 이 전 구간 **0.0002~0.0006** 으로 사실상 0 이었다.

`model_46000` 을 직접 굴려 계측한 결과(`_workspace/hindleg_gait_probe.py`, 64 env):

| 명령 vx | 실제 vx | 달성률 | 양발 접지 | 좌우 교대 접지 |
|---|---|---|---|---|
| 0.3 m/s | 0.222 | 74% | 99.3% | 0.3% |
| 0.5 m/s | 0.426 | 85% | 96.8% | 2.1% |
| 1.0 m/s | 0.159 | 16% | 95.1% | 1.7% |
| 2.0 m/s | −0.047 | −2% | 99.5% | 0.2% |

두 발을 붙인 채 저속에서만 밀고 나가는 **미끄러짐**이지 보행이 아니다.

### 원인: `_sole_rest_z` 가 스폰 공중값으로 캡처됨

swing clearance 보상의 기준선을 첫 `_get_rewards` 호출에서 전 env 평균 sole z 로 캡처하는데,
그 시점은 리셋 직후로 로봇이 스폰 높이(base z=0.6, 관절 전부 0)에서 **아직 공중에 있다**.
코드 주석은 "flat-ground constant, reset-invariant" 라고 단언하고 있었다.

| | HL | HR |
|---|---|---|
| 캡처된 기준선 | 109.0 mm | 105.5 mm |
| 실측 접지 sole z | 34.0 mm | 33.8 mm |
| **오차** | **+74.9 mm** | **+71.7 mm** |

`clearance = clamp((sole_z − rest) / gait_swing_height, 0, 1)` 이므로 보상을 받으려면 실제로
**145 mm** 를 들어야 했다(설계값 70 mm 의 2배, 다리 길이상 사실상 불가). 정책은 발을 드는 행동에서
**비용만 있고 보상이 0** 이었고, 대신 `gait_stance` 는 양발을 붙이면 만점(1.10)이라
"서서 버티며 저속으로 미는" 것이 최적해가 됐다. reward 35 는 대부분 여기서 나온 값이다.

### fix

접지 중인 발의 sole z 만 표본해 발별 평균으로 추정한다(`sole_rest_min_samples = 20000`).
검증: 학습 로그에 `sole rest-z 확정 [mm]: [34.5, 34.7]` — 프로브 실측(34.0/33.8)과 일치.
효과: `gait_swing` 이 **iter 2 에서 0.0076**, 662 iter 구간 평균이 0.0092 → 0.0175 로 단조 상승
(종전 36k iter 고정값 0.0002 대비 90배).

### ★ 교훈

**reward 총합 · episode length · value loss 가 모두 건강해도 보행 여부는 판정할 수 없다.**
접지 듀티와 좌우 교대율 실측이 유일한 판정선이다. 도구: `_workspace/hindleg_gait_probe.py`
(체크포인트 로드 → cmd 별 접지 듀티·좌우 교대·리프트·달성률 + 기준선 오차 출력).

## 충돌 모델 교체 (2026-08-13)

### 문제

URDF3 계열의 `*_collision.STL` 은 이름과 달리 **정밀 visual 메시**다(base 183,352 삼각형,
thigh 263,287). `collision_type: Convex Hull` 로 변환되므로 실제 충돌체는 브래킷·모터 하우징
돌출부까지 감싼 볼록 덩어리가 되어 **실측 부피의 2.2~4.9배**로 부푼다.

### 참조 자산과의 관계

구 리그 URDF `04_Hind_Leg_URDF/urdf/03_Leg_UFDF_260617_02.urdf` 는 링크당 1~2개 단순 볼록체
(104~1,780 삼각형, 부피비 1.0~1.8)로 이 문제를 풀어 두었다. 그러나 **다리 CAD 세대가 달라
메시를 재사용할 수 없다**:

| joint origin | 04 (구 리그) | 현재 SignFix |
|---|---|---|
| `HL_calf_joint` | `0.078665, 0, -0.21613` | `0.13192, 0, -0.1884` |
| `HL_foot_joint` | `-0.1225, 0, -0.21218` | `-0.18768, 0, -0.15748` |

base 와 hip 만 bbox·파일 크기가 완전히 일치한다(계보상 05 병합본이 base 를 03 에서 가져왔기 때문).
그래서 **03 의 설계 방식을 현재 CAD 에 재현**하는 쪽으로 구현했다.

### 결과

`_workspace/make_hindleg_collision_meshes.py` + `make_hindleg_collision_urdf.py`
→ `Hind_Leg_SignFix_Col.urdf` → 자산 `Hind_Leg_URDF3_SignFix_Col`

- visual 은 정밀 메시 유지, collision 은 `meshes/collision_simple/` 의 전용 볼록 메시로 분리
- 충돌 메시 총량 **50 MB / 104만 삼각형 → 0.5 MB / 수천**
- 접지 극점 z 오차 **0.000 mm** (물리 종전과 동일)

### package 경로

URDF 의 `package://` 이름은 **실제 폴더명 `Hind_Leg_URDF3`** 를 쓴다. SolidWorks 익스포터가
넣어준 robot name(`03_Leg_UFDF_Colision_260617_2`)은 디렉터리와 달라서, IsaacLab UrdfConverter
는 URDF 기준 상대 경로로 풀어 문제가 없지만(실측: 구 경로로도 조각 메시가 USD 에 정상 반영됨)
다른 ROS 도구에서는 해석되지 않는다. 2026-08-13 사용자 지적으로 생성 스크립트에 반영.

### ⚠ 조각화는 되돌렸다 — 처리량 7.3배 저하

발 계열을 12조각으로 쪼갠 판(충돌 shape 11 → 59개)은 부피비를 개선했으나
(발 2.20→1.48, 발바닥 1.65→1.46) 학습 처리량을 **69,500 → 9,500 steps/s** 로 떨어뜨렸다
(ETA 13시간 → 14시간 52분). 4096 env × shape 수만큼 broad/narrow phase 가 늘기 때문이다.

그런데 **평지 보행에서는 그 이득이 실현되지 않는다**: convex hull 은 부피가 부풀어도
극점(z_min)이 원본과 같아 지면에 처음 닿는 순간이 동일하고, `self_collision: false` 라
다리끼리 접촉은 계산조차 되지 않는다. 조각화가 실제로 이득인 것은 발 측면이 지면에 닿는
거친 지형뿐이므로 그때 다시 올린다. → 전 링크 단일 hull 로 되돌림.

## 자산 재생성 시 필수 수동 패치 (URDF3 계열 공통)

1. `payloads/Physics/physics.usda` 의 base 링크 `apiSchemas` 에서 `NewtonArticulationRootAPI` 제거
   (없으면 `fix_root_link=True` 에서 "Expected exactly one ArticulationRootAPI ... found 2")
2. `payloads/Physics/physx.usda` 의 foot 2관절 `physxJoint:maxJointVelocity`
   `847.97754` → `1415.2058` deg/s (URDF3 CAD 익스포터 감속비 퇴행 보정)

## 실행 방법

```bash
conda activate isaac-6.0          # ★ base python 에는 gymnasium 이 없다
cd /home/lgb/IsaacLab-6.0

# 학습 (--video 없이)
CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task HindLeg-Direct-v0 --num_envs 4096 --headless --run_name <name>

# 보행 실측 (학습 지표로는 판정 불가)
CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p _workspace/hindleg_gait_probe.py \
    --checkpoint logs/rsl_rl/hindLeg_history_direct/<run>/model_XXXX.pt \
    --num_envs 64 --steps 400 --cmd_x 0.3 0.5 1.0 2.0
```

## 변경 파일 (전부 미커밋)

- `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env_cfg.py`
- `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/agents/rsl_rl_ppo_cfg.py`
- `source/isaaclab_assets/isaaclab_assets/robots/rga.py`
- `_workspace/{make_hindleg_collision_meshes,make_hindleg_collision_urdf,hindleg_gait_probe}.py`
- 신규 자산 `source/isaaclab_assets/data/Robots/Hind_Leg_URDF3_SignFix_Col/`
- 신규 URDF `/home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg_SignFix_Col.urdf`

pre-commit(`./isaaclab.sh -f` 상당) 전체 통과 확인.

# A 제어기 MJCF vs 회사 Isaac 모델 — 대조 (2026-09-09)

- A 제어기 모델: `simulation/quad/mjcf/quad_real_17dof_waist_sphere.mjcf` (MuJoCo, MPC+WBIC)
- 회사 Isaac 모델: `LEG_CFG` (rga.py) → `Robots/Leg/Leg_gen/.../Leg.usda`, `Leg_URDF2` 임포트
- 교차검증에 쓴 제3의 자료: `data/Robots/Hind_Leg_URDF3_SignFix/urdf/Hind_Leg.urdf`
  (회사가 2026-08-12 에 **실기 방향 실측**으로 갱신한 8-DOF URDF. repo 에 실물이 있는 유일한 URDF)

## BLUF

**같은 로봇이다.** 관절명·질량·hip/thigh/calf 토크·URDF 속도한계가 전부 일치한다.
차이는 **모델링 철학**(MuJoCo 는 제어기가 물리를 보완, Isaac 은 모델에 담음)과 **값 3건**이다.
그중 발목 토크는 **회사 17-DOF 쪽이 stale** 이고, 회사의 최신 8-DOF URDF 가 우리 값을 뒷받침한다.

## 1. 일치하는 것

| 항목 | A 제어기 MJCF | 회사 `LEG_CFG` / URDF |
|---|---|---|
| 관절 구성 | 4 다리 × (hip, thigh, calf, foot) + `FB_waist_joint` = 17 | 동일 |
| 관절 이름 | `{FL,FR,HL,HR}_{hip,thigh,calf,foot}_joint` | **동일** |
| root body | `Base` | `Base` |
| 총 질량 | **38.02 kg** (mass 태그 22개 합) | **38.0 kg** (rga.py 주석) |
| hip / thigh / calf 토크 | 84 / 84 / 126 | 84 / 84 / 126 (URDF effort) |
| `foot_contact_link` | 존재 (`_foot_link` 의 자식) | 존재 (URDF 에 `_foot_contact_joint` type=fixed) |

## 2. ★ 차이 3건

### 2-1. 발목 토크 — 회사 17-DOF 가 stale (우리 값이 맞다, 사용자 확정)

| 출처 | foot effort |
|---|---|
| `Leg_URDF2` `<limit effort>` → `LEG_CFG` (×85%) | 168 → **142.8** |
| A 제어기 MJCF `actuatorfrcrange` | **100.8** |
| **회사 자신의 2026-08-12 8-DOF URDF** | **100.8** |

발목 8.4:1 재기어 실측값이 100.8 이다. 168 은 재기어 이전 값이다.
**회사의 최신 8-DOF URDF 도 100.8 을 쓴다** — 즉 17-DOF `LEG_CFG` 만 갱신이 안 된 것이지
우리와 회사의 견해차가 아니다. 142.8 은 실기의 **1.42 배**다.

### 2-2. 관절 속도한계 — `LEG_CFG` 가 평탄화했다

| | hip | thigh | calf | foot |
|---|---|---|---|---|
| URDF `<limit velocity>` | 29.6 | 29.6 | **19.7** | **14.8** |
| `LEG_CFG.velocity_limit_sim` | 30.0 | 30.0 | **30.0** | **30.0** |

calf **1.5 배**, foot **2.0 배** 과대. 이산지형은 스윙 각속도가 평지보다 크므로 실제로 걸린다.
(→ `LEG_DTC_CFG` 에서 URDF 실값으로 되돌림)

### 2-3. 게인 — 회사가 더 원칙적이다

| | hip | thigh | calf | foot | waist |
|---|---|---|---|---|---|
| 회사 `LEG_CFG` Kp | 71 | 53 | **134** | **59** | 112 |
| 내 구 `QUAD_17DOF_CFG` Kp | 65 | 53 | **12** | **20** | 40 |

회사 값은 `compute_leg_ieff.py` 로 USD 에서 유효관성 I_eff 를 실측해 `Kd = 2ζ√(Kp·I_tot)`, ζ=0.9 로 뽑았고,
접지 관절(calf/foot)은 관성 기반 Kp 가 스탠스 하중에 붕괴하므로 **정적토크/허용처짐 0.25 rad 로 floor** 를 건다.
→ `LEG_DTC_CFG` 는 이 값을 **채택**한다. 단 평지 AMP 기준이라 지형 학습 후 재확인 대상.

## 3. 모델링 철학 차이 — 값이 아니라 "어디서 처리하는가"

| 물리 | A 제어기 (MuJoCo) | 회사 Isaac |
|---|---|---|
| **반사관성** | MJCF 에 `armature` **없음** → 제어기 `GEARBOX` 항이 담당(기본 ON) | `LEG_ARMATURE = 0.01` 전 관절 균일 |
| **calf–foot 기구 커플링** | MJCF 에 `equality`/`tendon` **없음** → 제어기 `couple_clamp`(평행사변형 토크집합)가 담당 | **없음**(17-DOF 쪽엔 아무 처리도 없다) |
| **발 접촉** | `{LEG}_sphere` **구 geom r=0.025**, mesh 는 `contype=0`(시각 전용) | URDF collision mesh 그대로 |
| **접촉 파라미터** | `friction=1.3`, `solref="0.03 1"`(soft, 착지 완충) | PhysX 기본 |
| **관절 감쇠/마찰** | MJCF 에 `damping`/`frictionloss` **없음** | `friction=0.0`, `armature=0.01` |

★ 이 중 **이산지형에 직접 영향을 주는 것은 발 형상**이다. 우리 17-DOF 는 sphere 발이 사실상 필수였는데
(→ `sim2real-checklist-17dof`) `Leg_gen` 은 URDF mesh 발이다. stepping-stone/gap 에서 접촉 거동이
달라질 수 있어 **USD 도착 후 가장 먼저 볼 항목**이다.

★★ **calf–foot 커플링이 회사 17-DOF 모델에 없다.** A 제어기는 `|τ_foot| ≤ 100.8 ∧ |τ_calf − τ_foot| ≤ 126`
평행사변형 집합을 QP 에 넣는데(PACE c=1, → `02leg-motor-spec`), `LEG_CFG` 는 두 축을 독립으로 본다.
즉 **회사 정책은 실기가 못 내는 토크 조합을 낼 수 있다.** 8-DOF 쪽(`r2s_biped_leg`)은 이 커플링을
다루고 있으므로 17-DOF 만 빠진 것으로 보인다. 배포 단계에서 반드시 다시 봐야 한다.

## 4. ⚠ 미해결 — 관절 한계와 축 부호

회사 8-DOF URDF(SignFix)와 내 MJCF 의 뒷다리 관절 한계가 다르다:

| 관절 | 회사 URDF (SignFix, 8-DOF) | 내 MJCF (17-DOF) |
|---|---|---|
| `HL_hip` | −0.26 ~ 0.26 | −0.349 ~ 0.349 |
| `HL_thigh` | −1.13 ~ **2.36** | **−2.356** ~ 1.134 |
| `HL_calf` | −0.87 ~ 1.04 | −0.959 ~ 1.134 |
| `HL_foot` | −0.44 ~ 1.48 | −1.396 ~ 0.698 |

`HL_thigh` 는 부호가 뒤집혀 있고(±범위가 거울), 나머지는 폭 자체가 다르다.
회사 URDF 헤더에 사유가 적혀 있다 — **"실기 방향 실측에 맞춰 URDF3 에서 5개 관절의 회전축·한계를 반전"**.

**이게 왜 중요한가:** AMP 데이터셋 README 가 같은 계열 문제를 기록하고 있다 —
`hrfix` = "HR_foot_joint 축 반전(0,−1,0) 미정합으로 인한 **우측 뒷발목 41.6° 과굴곡·지면관통**" 을
데이터 쪽에서 수술로 때운 것이다. 즉 **17-DOF 자산에도 같은 축 부호 문제가 남아 있을 가능성**이 있고,
회사는 모델이 아니라 모션 데이터를 고쳐서 우회했다.

단, 두 URDF 의 패키지명이 다르다(`03_Leg_UFDF_Colision_260617_2` vs 우리 02_Leg 계보)므로
**하드웨어 리비전 차이일 수도 있다. 지금 자료로는 판정 불가** — `Leg_URDF2` 원본을 받아야 확정된다.

## 5. 결론

| 질문 | 답 |
|---|---|
| 많이 차이나나? | **아니다.** 같은 로봇, 값 3건 차이 |
| 어느 쪽을 믿나 | 발목 토크·속도한계 = **우리**(회사 최신 8-DOF URDF 가 뒷받침) / 게인 = **회사**(I_eff 실측) |
| 이식에 걸림돌 | 없음. 이름 기반 코드라 순서 차이 자동 흡수 |
| USD 도착 후 볼 것 | ① 발 형상(sphere vs mesh) ② 관절 축 부호/한계 ③ prim 중첩 구조 ④ spawn 높이 |
| 배포 전 볼 것 | **calf–foot 커플링이 회사 모델에 없다** |

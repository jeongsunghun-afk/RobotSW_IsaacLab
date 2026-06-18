# 02 — go2_parkour_symmetry 3-Run Reward Scales 비교

**작성**: worker-2 (Task #2) · **대상**: `logs/rsl_rl/go2_parkour_symmetry/` 3개 런의 `params/env.yaml`
**소스**: 각 런 `params/env.yaml` `reward_scales:` 블록 (line 1014–1033)

## 런 식별

| ID | 디렉토리 | 이름이 시사하는 변경 |
|----|----------|--------------------|
| R1 | `2026-06-11_12-13-02_positive_work`       | positive_work 항 활성 |
| R2 | `2026-06-11_16-21-24_no_duty_time_cap`    | contact_duty_deficit + air_time_cap 제거 |
| R3 | `2026-06-11_17-28-23_positive_work_0.01`  | positive_work weight = 0.01 |

---

## 전체 reward_scales 비교표

3개 런의 `reward_scales`는 **`positive_work` 단 한 항만 다르고 나머지 17항은 완전 동일**하다.

| 항 | R1 (`_positive_work`) | R2 (`_no_duty_time_cap`) | R3 (`_positive_work_0.01`) |
|----|----:|----:|----:|
| tracking_goal_vel    | **1.5**     | **1.5**     | **1.5**     |
| tracking_yaw         | **0.5**     | **0.5**     | **0.5**     |
| lin_vel_z_l2         | -1.0        | -1.0        | -1.0        |
| ang_vel_xy_l2        | -0.05       | -0.05       | -0.05       |
| orientation_l2       | -1.0        | -1.0        | -1.0        |
| dof_acc_l2           | -2.5e-07    | -2.5e-07    | -2.5e-07    |
| collision            | -10.0       | -10.0       | -10.0       |
| action_rate_l2       | -0.1        | -0.1        | -0.1        |
| delta_torques        | -1.0e-07    | -1.0e-07    | -1.0e-07    |
| **torques_l2**       | **-1.0e-05**| **-1.0e-05**| **-1.0e-05**|
| hip_pos              | -0.5        | -0.5        | -0.5        |
| dof_error_l2         | -0.04       | -0.04       | -0.04       |
| feet_stumble         | -1.0        | -1.0        | -1.0        |
| feet_edge            | -1.0        | -1.0        | -1.0        |
| feet_dragging        | -0.1        | -0.1        | -0.1        |
| **feet_gait_pairing**| **0.0**     | **0.0**     | **0.0**     |
| **air_time_cap**     | **-0.0**    | **-0.0**    | **-0.0**    |
| **contact_duty_deficit** | **-0.0**| **-0.0**    | **-0.0**    |
| **positive_work**    | **-3e-4**   | **0.0**     | **-1e-2** ⚠ |

> 부수 파라미터(`tracking_sigma=0.2`, `contact_duty_target=0.5`, `air_time_cap_max_s=1.0`, `feet_gait_*`, `_actuator_mode=2` 등)도 3런 전부 동일.

---

## 런 이름 ↔ 실제 yaml 일치성 검증

| 런 | 이름 주장 | 실제 yaml | 판정 |
|----|----------|-----------|------|
| R1 | positive_work 활성 | positive_work = **-3e-4** (그 외 항 baseline) | ✅ **일치** |
| R2 | duty + time_cap 제거 | contact_duty_deficit=0, air_time_cap=0 | ⚠ **부분일치 (아래 주의)** |
| R3 | positive_work = 0.01 | positive_work = **-1e-2** | ✅ **일치** (부호 음수, 크기 0.01) |

### ⚠ R2 이름의 함정 — "no_duty_time_cap"은 R2 고유 변경이 아니다
`contact_duty_deficit`와 `air_time_cap`은 **R1·R2·R3 세 런 모두 `-0.0`** 이다.
즉 "duty/time_cap 제거"는 R2만의 차별점이 아니라 **세 런 공통 상태**이다.
이 세트 안에서 R2를 실질적으로 구분짓는 유일한 차이는 **positive_work = 0** (R1/R3와 달리 positive_work도 OFF) 이라는 점.
→ **R2 = 세 항(duty·time_cap·positive_work) 모두 OFF인 순수 baseline**, R1/R3는 거기에 positive_work만 각각 -3e-4 / -1e-2로 켠 ablation으로 읽어야 한다. R2 이름은 (이 세트 외부의 더 이전 런 대비) 상태 서술일 뿐, 세 런 간 대조축으로 오해 금지.

---

## ⭐ R3 positive_work weight = -1e-2 → clip floor 위험 (요청 핵심)

- **확인**: R3 이름의 "0.01"은 실제로 `positive_work: -0.01` = **-1e-2**. (부호 음수, penalty 방향 정상)
- **calibration 권장값 대비**: 권장 -3e-4 의 **약 33.3배** (-1e-2 / -3e-4 = 33.3×).
- **clip floor 위험**: env 코드 `_get_rewards()`는 `total_reward = clip(Σ scale·step_dt·value, **min=0.0**)` 로 음수 보상을 0에서 잘라낸다 (parkour_env.py:1287, A/B 공통 의도된 설계).
  - calibration 기준선: positive_work **-3e-4 = stair p99(857W)에서 tracking 기여의 24%** (안전), **-1e-3 = 82%** (경계).
  - 선형 외삽: **-1e-2 = stair p99에서 tracking의 ~800%** → positive_work penalty 단독으로 tracking_goal_vel(1.5)+tracking_yaw(0.5) 등 양수 항을 압도.
  - 결과: 장애물 구간(고출력 구간)에서 `total_reward`가 **clip(min=0) floor에 정박** → gradient 소실, 학습 신호가 "통과하되 멈춤(=일 적게)"로 붕괴 가능. **-1e-2는 권장 안전대(-3e-4) 밖, 경계(-1e-3)도 10배 초과.**

> 단위 주의: 위 % 비교는 **episode-normalized(step_dt 이미 반영)** 측정값 기준. env 코드가 scale 단계에서 `step_dt`를 곱하므로(`scaled = scale·step_dt·value`), weight 산정/비교 시 step_dt 재곱 금지.

---

## 참조: 비교 대상 reward 항 수식 (parkour_env.py `_get_rewards()`)

- **positive_work** (line 1126): `Σ_j max(0, τ_j · q̇_j)` over 12 joints, 단위 W, 항상 ≥0.
  motoring(양의 기계적 일)만 카운트, clamp(min=0)로 회생 구간 제외. scale 음수 → efficiency penalty.
- **air_time_cap** (line 1225): `Σ_foot max(0, feet_air_time − air_time_cap_max_s(=1.0))`, ≥0. 3런 모두 scale 0 → 비활성.
- **contact_duty_deficit** (line 1235): `Σ_foot max(0, contact_duty_target(=0.5) − foot_contact_duty_EMA) · is_flat`, ≥0. flat 지형에서만. 3런 모두 scale 0 → 비활성.
- **tracking_goal_vel** (line ~1066): goal 방향 속도 추종 양수 보상, weight 1.5 (3런 동일, 비교 분모).
- **feet_gait_pairing** (line 1175): diagonal 쌍 air/contact time 매칭. 3런 모두 scale 0 → 비활성.
- **torques_l2**: `-1e-5 · Σ τ²`, 3런 동일.

---

## 결론

1. **3런의 유일한 실질 차이 = `positive_work` weight**: R1 **-3e-4** / R2 **0** / R3 **-1e-2**. 나머지 17항·부수 파라미터 전부 동일 → 이 세트는 깨끗한 positive_work weight ablation (0 → 권장 → 33×과대).
2. **이름 검증**: R1·R3 이름은 yaml과 정확히 일치. R2 "no_duty_time_cap"은 R2 고유 변경이 아님(세 런 공통) — 실 차별점은 positive_work=0, 주의 요함.
3. **R3 = -1e-2는 calibration 권장(-3e-4)의 33배·경계(-1e-3)의 10배** → 고출력 장애물 구간에서 total_reward clip(min=0) floor 정박 위험 명시. 학습곡선 분석 시 R3의 tracking/return 붕괴·정체 여부를 우선 확인 권장.

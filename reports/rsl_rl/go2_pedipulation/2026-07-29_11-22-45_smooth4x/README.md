# 2026-07-29_11-22-45_smooth4x

gym task id: `Go2-Pedipulation-v0`

S1 첫 산출(`2026-07-27_18-44-01/model_10600.pt`)에서 **action 규제를 4배**로 올려 3000 iteration
finetune (10600 → 13599), 2048 env, `trajectory_mode=static`.

```
env.w_action_rate=-0.02      # 기존 -0.005 의 4배
env.w_action_smooth=-0.008   # 기존 -0.002 의 4배
```

## ⚠ 이 run 의 `params/env.yaml` 은 오해를 부른다

`env.yaml` 에는 이렇게 적혀 있다.

```yaml
dr:
  push_robot: true
  push_interval_s: 3.0
  max_push_vel_xy: 0.6
command:
  resample_time_min: 3.0
  resample_time_max: 5.0
```

**둘 다 실제로는 한 번도 실행되지 않았다.** 이 기능들은 `_post_physics_step()` 안에 있는데,
그 메서드는 `DirectRLEnv` 의 훅이 아니어서 **호출되지 않았다** (`890a3174dc4` 에서
`_get_dones` 첫머리 명시 호출로 수정). 따라서 이 시점(2026-07-29)까지 학습된 모든 정책
— S1 첫 산출 · `smooth4x` · `smooth10x` — 은 실제로

- base push DR 을 **받은 적이 없고**
- 에피소드 중 명령 변경을 **겪은 적이 없다** (명령은 에피소드마다 한 번만 정해진다)

`890a3174dc4` **이전** run 의 `params/env.yaml` 에서 `dr.push_robot` 과 `command.resample_time_*`
을 읽을 때는 이 점을 반드시 감안할 것. 평가 지표는 영향이 없다 (평가는 `resample_time` 을
1e6 으로 고정하고 push 를 직접 주입하므로 버그 유무와 무관하게 같은 조건이다) — 영향은
**학습 분포**에만 있다.

이것이 `smooth4x` 의 push 열세(150 N 3족 53.0%)의 유력한 원인이며, `_pushdr` finetune 의 전제다.

## 평가 결과

`model_13599.pt`. 상세 비교는
[`../_comparisons/hold_tremor_objective/`](../_comparisons/hold_tremor_objective/).

| 항목 | 값 | 계보 내 순위 |
|---|---:|---|
| hold 오차 | **10.18 mm** | **1위** |
| hold std | 3.85 mm | 2위 |
| 떨림 (12관절) | 1.29 °/step | 2위 |
| 원 RMSE ω=0.25/0.5/1.0 | 13.70 / 17.24 / 26.62 mm | 2위 (zero-shot) |
| step 재수렴 p95 | **0.68 s** | **1위** |
| step 정착 오차 | 10.64 mm | 2위 |
| S4 하중 전이 위반 | **0.00%** | **1위** |
| S4 정상상태 하중 | 14.5 N (자중 9.9%) | 2위 |
| 몸통 표류 | 6.28 cm | 4위 |
| push 150 N 3족 | **53.0%** | **5위** ← 유일한 약점 |

**사용자가 명확히 한 우선순위(hold 정확도 · 무진동 · 명령 일반화)에서 계보 1~2위이고,
약점은 push 하나뿐이다.**

## 산출물

| 경로 | 내용 |
|---|---|
| `metrics/hold_s4x.json` | hold (3072 시행) |
| `metrics/push_s4x.json` | push 스윕 |
| `metrics/g2g3_hold_1024x6.json` | ⚠ 파일명과 달리 **S1 첫 산출(`model_10600`)** 측정치다 |
| `metrics/drift_dr_on.json` | 몸통 표류 (2026-08-03 측정) |
| `metrics/g5_push.json`, `g7_drift.json` | ⚠ 이것들도 `model_10600` 측정치 |
| `videos/model_13599__showcase_s1__20260729-133600.mp4` | S1 시연 29.8 s |

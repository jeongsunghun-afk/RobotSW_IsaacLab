# 2026-08-04_09-09-20_hipoff_4x

gym task id: `Go2-Pedipulation-v0`

**`smooth4x` 의 정정판이자, `hipON 4배` 학습 실패의 원인 판별용 대조군.**

| key | 값 |
|---|---|
| `hip_scale_reduction` | **false** |
| `w_action_rate` / `w_action_smooth` | −0.02 / −0.008 (4배, cfg 기본값) |
| `w_base_drift` / `w_joint_target_rate` | 0.0 / 0.0 |
| `jvel_filter_alpha` | 1.0 (끔) |
| from scratch / seed / env / iter | resume 없음 / 42 / 2048 / 20000 |

## 왜 필요했나 ① — `smooth4x` 는 비교 기준으로 쓸 수 없었다

`smooth4x` 는 `_post_physics_step` 이 `DirectRLEnv` 의 훅이 아닌데 훅처럼 정의만 되어 있어
**호출되지 않은 채** 학습됐다. `_cmd_timer` 가 줄지 않아 **에피소드 중 명령이 한 번도 바뀌지
않았다.** `params/env.yaml` 은 `resample_time 3.0/5.0` 을 기록하지만 그 코드는 실행되지
않았다 — 설정 파일만 보면 알 수 없는 실패다. 버그는 `890a3174dc4` 에서 고쳐졌고
(`go2_pedipulation_env.py:761`), 이 run 은 고쳐진 env 로 돈다.

### 결과 — 버그는 `smooth4x` 에 이득이 아니라 손해였다

| | `smooth4x` (버그) | **이 run** (정정) |
|---|---:|---:|
| hold 오차 | 10.18 mm | **9.23 mm** |
| 떨림 | 1.29 °/step | **1.173** |
| step 재수렴 p95 | 0.68 s | **0.34 s** |
| 원 RMSE ω=0.5 | 17.24 mm | **13.19 mm** |
| push 150 N 3족 | 53.0% | **82.2%** |
| 몸통 표류 | **6.28 cm** | 6.98 |

명령 분포가 더 어려워졌는데도 표류 하나 빼고 전부 개선됐다. **"명령이 안 바뀌어서 hold 가
좋아 보였다"는 가설은 기각된다** — 오히려 명령 변화를 겪지 못해 step 응답과 push 가 나빴다.

⚠ `smooth4x` 는 13599 iteration, 이 run 은 20000 이다. 개선폭 중 얼마가 iteration 덕인지는
이 쌍으로 가를 수 없다.

## 왜 필요했나 ② — `hipON 4배` 실패의 원인 판별

| | hipON 4배 | **이 run** (hip OFF, 나머지 동일) |
|---|---:|---:|
| 커리큘럼 승급 | **0 회** / 9891 평가 | **7726 회** |
| 최종 `cmd_box_x` | 0.060 = 초기값 | **0.200** = 최대 |
| 최종 `Policy/mean_std` | **2.826** (단조 발산) | 1.317 |
| hold 오차 | 159.78 mm | **9.23 mm** |

**hip 축소가 4배 규제에서 학습을 무너뜨린 것이 맞다.** 다만 칸당 n=1 이므로 "4배+hip ON 은
반드시 실패한다"가 아니라 여기까지가 말할 수 있는 범위다.

## 전체 성능

hold 9.23 mm · 떨림 1.173 °/step · hold 기준 충족 100% · G2 100% · 생존 100% ·
step 재수렴 96.8% (p95 **0.34 s** — 2×2 최고) · 원 RMSE 13.19 mm · 완주 100% ·
push 150 N **82.2%** (2×2 최고) · S4 위반 0.00% · 접촉 하중 12.0% (2×2 최저) · 표류 6.98 cm.

떨림만 10배 계열(0.823~0.878)보다 높다.

## 상태

**완료** (2026-08-04 09:09 ~ 17:50, GPU 0, `model_19999.pt`). 6종 평가 완료.
2×2 전체 해석은 [`../../_comparisons/hipscale_2x2/`](../../_comparisons/hipscale_2x2/).

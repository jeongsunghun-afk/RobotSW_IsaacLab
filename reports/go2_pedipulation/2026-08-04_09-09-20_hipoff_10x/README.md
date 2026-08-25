# 2026-08-04_09-09-20_hipoff_10x

gym task id: `Go2-Pedipulation-v0`

[`2026-08-03_15-53-29_hipscale_scratch_s10x`](../2026-08-03_15-53-29_hipscale_scratch_s10x/)
의 **순수 단일 변수 대조군.** `params/env.yaml` 대조로 `hip_scale_reduction` **한 키만**
다름을 확인했다 — 이 쌍의 차이가 곧 hip 축소가 성능에 매기는 값이다.

| key | 값 |
|---|---|
| `hip_scale_reduction` | **false** (대조 arm 은 true) |
| `w_action_rate` / `w_action_smooth` | −0.05 / −0.02 (10배) |
| `w_base_drift` / `w_joint_target_rate` | 0.0 / 0.0 |
| `jvel_filter_alpha` | 1.0 (끔) |
| from scratch / seed / env / iter | resume 없음 / 42 / 2048 / 20000 |

## 결과 — hip 축소의 대가는 측정 한계 안에 있다

| | 이 run (hip OFF) | hipON 10배 |
|---|---:|---:|
| hold 오차 | **8.65 mm** | 9.32 |
| 떨림 | 0.878 °/step | **0.823** |
| step 재수렴 p95 | 0.40 s | **0.38** |
| 원 RMSE ω=0.5 | **12.32 mm** | 13.44 |
| push 150 N 3족 | 79.5% [73.6, 84.3] | 78.1% [72.1, 83.1] |
| push 200 N 3족 | 51.1% [44.6, 57.7] | 45.2% [38.8, 51.8] |
| 접촉 하중 (자중 대비) | 17.6% | **14.9%** |
| S4 지지다각형 위반 | 0.00% | 0.00% |
| 측정 타당성 4족 / 3족 | 0.893 / 0.998 | **0.994 / 0.994** |
| 몸통 표류 | **7.44 cm** | 7.88 |

**push 차이는 유의하지 않다** — Wilson 95% 구간이 크게 겹친다 (셀당 n=219). hold 는 이 run 이
0.67 mm 낮고, 떨림·접촉 하중·측정 타당성은 hip ON 이 낫다.

"hip 은 좌우 균형의 주 액추에이터이므로 축소하면 push 가 떨어질 것"이라는 사전 예측은
**틀렸다.** 학습이 성립하기만 하면 정책이 나머지 자유도로 보상한다.

`hold` 오차 8.65 mm 는 이 프로젝트 전체 최저값이다 (이전 최저 `smooth4x` 10.18 mm).

## 상태

**완료** (2026-08-04 09:09 ~ 17:50, GPU 2, `model_19999.pt`). 6종 평가 완료.
2×2 전체 해석은 [`../../_comparisons/hipscale_2x2/`](../../_comparisons/hipscale_2x2/).

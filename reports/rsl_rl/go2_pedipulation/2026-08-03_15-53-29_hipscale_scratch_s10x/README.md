# 2026-08-03_15-53-29_hipscale_scratch_s10x

gym task id: `Go2-Pedipulation-v0`

[`2026-08-03_13-13-09_hipscale_scratch`](../2026-08-03_13-13-09_hipscale_scratch/) 의
**action 규제 10배 대조군.** 둘 다 `hip_scale_reduction=True` 로 처음부터 학습한다.

## 단일 변수 A/B

| key | 4배 (`hipscale_scratch`) | **10배 (이 run)** |
|---|---:|---:|
| `w_action_rate` | −0.02 | **−0.05** |
| `w_action_smooth` | −0.008 | **−0.02** |
| `hip_scale_reduction` | True | True |
| `w_base_drift` | 0.0 | 0.0 |
| `w_joint_target_rate` | 0.0 | 0.0 |
| `jvel_filter_alpha` | 1.0 (끔) | 1.0 (끔) |
| `resume` | False | False |
| `command.box_init` | (0.06, 0.05, 0.08) | (0.06, 0.05, 0.08) |
| `command.resample_time` | 3.0 / 5.0 | 3.0 / 5.0 |
| `num_envs` / `max_iterations` | 2048 / 20000 | 2048 / 20000 |

**두 가중치 외에는 전부 같다.** `params/env.yaml` 대조로 확인했다.

## 왜 이 A/B 인가

`hip_scale_reduction` **이전** 계보에서, hold 오차와 떨림을 동시에 개선한 방법은
**action 규제 상향 하나뿐**이었다. 떨림을 직접 겨냥한 장치들(관절 목표 rate 페널티, 출력 EMA)은
전부 hold 를 팔았다.

| 배율 | 정책 | hold 오차 | 떨림 | push 150 N 3족 |
|---|---|---:|---:|---:|
| 1배 (−0.005 / −0.002) | S1 첫 산출 | 12.44 mm | 2.56 | 52.1% |
| 4배 (−0.02 / −0.008) | `smooth4x` | **10.18 mm** | 1.29 | 53.0% |
| 10배 (−0.05 / −0.02) | `smooth10x` | 10.22 mm | **0.91** | **30.6%** |

4배는 push 를 **전혀 팔지 않았고**(52.1 → 53.0, 오차범위), 10배는 **절반으로 팔았다.**
hold 는 4배·10배가 사실상 동률이다.

## 관측 대상 — 두 감쇠가 겹칠 때

`hip_scale_reduction` 자체가 hip 액션 권한을 절반으로 줄이는 조치다. hip 은 좌우 균형의 주
액추에이터이므로, **규제 10배와 겹쳤을 때 push 가 어디까지 떨어지는지**가 이 run 의 핵심
질문이다. 구 계보의 10배 단독이 30.6% 였으므로 그보다 낮아질 수 있다.

반대 가능성도 있다 — hip 축소가 이미 떨림을 줄여 준다면 10배까지 갈 필요가 없을 수 있고,
그 경우 4배가 그대로 답이다.

## 판정 기준 — **결과를 보기 전에 정한다**

| 판정 | 조건 |
|---|---|
| **10배 채택** | 떨림이 4배 대비 유의하게 낮고, push 150 N 3족이 **40% 이상** 유지 |
| **4배 채택** | 위 push 조건 미달, 또는 떨림 이득이 미미 |
| **둘 다 부족** | 양쪽 다 push 가 40% 미만 → `hip_scale_reduction` 이 지지 안정성에 주는 대가가 크다는 뜻. 그때는 규제 배율이 아니라 **hip 축소 계수 0.5 자체**를 재검토한다 |

⚠ push 40% 기준은 구 계보의 4배(53.0%)와 10배(30.6%) 사이에서 잡았다. hip 축소가 push 를
얼마나 깎을지 사전 정보가 없어 **절대 기준이 아니라 판단 보조선**이다. 실제 판정 때는
4배 run 의 실측값을 기준선으로 다시 읽는다.

## 기동 시 확인한 것

```
Learning iteration 6/20000                       ← resume 아님
Metric/cmd_box_x: 0.0600  y: 0.0500  z: 0.0800   ← 커리큘럼 작게 시작
Episode_Reward/action_rate:   -8.3599            ← 4배 run 보다 크게 음수
Episode_Reward/action_smooth: -10.0006
Episode_Reward/base_drift:      0.0000
Episode_Reward/joint_target_rate: 0.0000
```

## 상태

**진행 중** (시작 2026-08-03 15:53, GPU 3, ETA ~9 h 30 m). 4배 run 은 GPU 1 에서 병렬로
진행 중이며 약 6300 iteration 앞서 있다. 둘 다 20000 iteration 이므로 최종 비교는 공정하다.

완료 후 hold / push / circle / step / contact / drift 6종 평가 예정.
평가 시 `--hip_scale_reduction auto` 가 `params/env.yaml` 에서 `True` 를 읽어 자동으로 맞춘다.

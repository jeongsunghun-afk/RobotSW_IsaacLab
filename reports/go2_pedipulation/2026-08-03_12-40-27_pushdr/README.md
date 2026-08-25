# 2026-08-03_12-40-27_pushdr

gym task id: `Go2-Pedipulation-v0`

**가설: `smooth4x` 의 push 열세는 학습 중 push DR 이 한 번도 실행되지 않았기 때문이다.**

`2026-07-29_11-22-45_smooth4x/model_13599.pt` 에서 3000 iteration finetune (13599 → 16599),
2048 env. `_post_physics_step` 수정(`890a3174dc4`) 이후이므로 **이번에는 push DR 이 실제로
실행된다** — 이것이 유일한 새 변수다.

## 왜 hold·떨림을 지키려 하는가

사용자가 2026-08-03 에 목표를 명확히 했다. 우선순위는 **① 특정 목표 유지 ② 떨리지 않을 것**
③ 명령 일반화 ④ push. `smooth4x` 는 ①②③ 에서 계보 1~2위이고 약점은 ④ 하나뿐이므로,
`s2_circle` 경로를 다시 타는 대신 **약점만 직접 고친다.**

## 변수 격리 — 조용히 섞일 뻔한 것 3개

`smooth4x` 이후 cfg 가 바뀌어, 기본값을 그냥 쓰면 변수가 4개가 된다. 셋을 명시로 되돌렸다.

| 항목 | 현재 기본값 | 이 run | 이유 |
|---|---|---|---|
| `w_joint_target_rate` | −0.4 | **0.0** | `smooth4x` 에 없던 보상 항 |
| `w_base_drift` | −50.0 | **0.0** | `smooth4x` 에 없던 항. `drift50` 에서 S4 위반을 19.19% 로 만든 바로 그 항 |
| `command.box_init` | (0.06, 0.05, 0.08) | **(0.20, 0.14, 0.26)** | `smooth4x` 는 `box_init = box_max`, 즉 **커리큘럼 없이 전체 작업공간**에서 학습했다 |
| `command.resample_time_*` | 3.0 / 5.0 | **1e6** | 수정으로 에피소드 중 명령 재샘플도 함께 살아난다. push DR 만 격리하려면 막아야 한다 |

⚠ **`box_init` 은 첫 시도에서 놓쳤다.** resume 은 커리큘럼 상태를 복원하지 않는데
(checkpoint 에 없다) 기본값이 작아 커리큘럼이 처음부터 다시 자라고 있었다.
`Metric/cmd_box_x` 가 0.12 로 찍혀 6 초 만에 발견하고 재시작했다. 재시작 후 확인:

```
Metric/cmd_box_x: 0.2000   cmd_box_y: 0.1400   cmd_box_z: 0.2600
Episode_Reward/base_drift: 0.0000
Episode_Reward/joint_target_rate: 0.0000
```

## 성공 기준 — **결과를 보기 전에 정한다**

| 판정 | 조건 |
|---|---|
| **성공** | push 150 N 3족이 **53.0% 에서 유의하게 상승**하고, **동시에** hold 오차 ≤ 11 mm · 떨림 ≤ 1.4 °/step 유지 |
| **부분** | push 는 오르지만 hold/떨림이 기준을 넘음 |
| **실패** | push 가 오르지 않음 → 열세의 원인이 push DR 부재가 아니다 |

**부분** 판정이 나오면 그것은 반복할 실험이 아니라 **결론**이다 — 이 env 에서 push 강건성과
hold 정확도가 실제로 결합돼 있다는 뜻이고, 그대로 보고한다. 이번 세션 내내 문서화한 교환
(우선순위 1·2·3 을 팔아 4를 사는 것)이 회피 가능한 실수가 아니라 구조라는 증거가 된다.

## 비교 기준선

| | `smooth4x` (출발점) | `jtr40_jvf` (직전 채택) |
|---|---:|---:|
| hold 오차 | 10.18 mm | 15.60 mm |
| 떨림 | 1.29 °/step | 1.42 |
| 원 RMSE ω=0.5 | 17.24 mm | 19.72 |
| step 재수렴 p95 | 0.68 s | 4.37 |
| S4 위반 / 하중 | 0.00% / 9.9% | 0.05% / 24.5% |
| 몸통 표류 | 6.28 cm | 5.52 |
| **push 150 N 3족** | **53.0%** | **76.5%** |

## 실행

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Pedipulation-v0 --num_envs 2048 --headless \
  --run_name pushdr --max_iterations 3000 \
  --resume --load_run 2026-07-29_11-22-45_smooth4x --checkpoint model_13599.pt \
  env.w_joint_target_rate=0.0 env.w_base_drift=0.0 \
  env.command.trajectory_mode=static \
  env.command.resample_time_min=1000000.0 env.command.resample_time_max=1000000.0 \
  'env.command.box_init=[0.20,0.14,0.26]'
```

학습 중 영상 녹화 없음 (이 저장소에서 hang 2회).

## 결과 — 사전 정의대로 **실패** 판정. push 가 오르지 않았다

| 지표 | `smooth4x` (출발점) | **이 run** | 판정 |
|---|---:|---:|---|
| **push 150 N 3족** | **53.0%** | **48.9%** | 오르지 않음 (오히려 하락) |
| push 200 N 3족 | 27.9 | 26.9 | 동일 |
| push 100 N 3족 | 90.9 | 68.0 | **하락** |
| push 0 N 3족 | 100.0 | **65.8** | **하락** |
| hold 오차 | 10.18 mm | **29.07 mm** | 기준(≤11) 초과 |
| 떨림 | 1.29 °/step | **5.56 °/step** | 기준(≤1.4) 초과 |
| 생존율 | — | **75.1%** | — |
| 원 RMSE ω=0.5 | 17.24 mm | 44.79 mm | 하락 |
| step 재수렴 p95 | 0.68 s | 4.98 s | 하락 |
| 몸통 표류 | 6.28 cm | 16.86 cm | 하락 |

**`0 N` 에서조차 3족 생존율이 65.8% 다.** 외력이 없는데 3분의 1이 넘어진다는 뜻이므로,
이것은 push 강건성 실험의 결과가 아니라 **정책이 망가졌다는 신호**다. 3000 iteration
fine-tune 이 `smooth4x` 를 회복 불가하게 무너뜨렸다.

### 결론 — 가설 기각

사전에 기록한 판정표의 **실패** 항목 그대로다: "push 가 오르지 않음 → 열세의 원인이 push DR
부재가 아니다." `smooth4x` 의 push 열세는 학습 중 push DR 이 실행되지 않았기 때문이 **아니다.**
push DR 을 실제로 켠 채 이어 학습하자 push 는 그대로이고 나머지가 전부 무너졌다.

이 방향(기존 정책 fine-tune 으로 push 만 보강)은 여기서 종료한다. 반복할 실험이 아니다.

### 이 결과가 이후 판단에 미친 영향

같은 시각 진행한 `hipscale_scratch_s10x` (from scratch, 규제 10배)는 **push 150 N 3족
78.1% 를 hold 9.32 mm · 떨림 0.823 °/step 과 동시에** 달성했다. 즉 push 와 hold 의 교환은
이 env 의 구조가 아니라 **fine-tune 경로의 산물**이었다. 처음부터 다시 학습하면 둘 다
얻을 수 있다. 상세는
[`../2026-08-03_15-53-29_hipscale_scratch_s10x/`](../2026-08-03_15-53-29_hipscale_scratch_s10x/).

## 상태

**완료** (2026-08-03 12:40 ~ 13:49, `model_16598.pt`). 5종 평가 완료, 가설 기각.

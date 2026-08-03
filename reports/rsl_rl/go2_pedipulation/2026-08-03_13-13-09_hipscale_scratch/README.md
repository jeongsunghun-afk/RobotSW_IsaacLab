# 2026-08-03_13-13-09_hipscale_scratch

gym task id: `Go2-Pedipulation-v0`

**sim2real 용 `hip_scale_reduction` 을 켠 채로 처음부터 학습** (resume 없음),
20000 iteration, 2048 env.

## `hip_scale_reduction` 이 하는 일

지지 다리 슬롯(`a_loc`)의 hip(abduction) 액션만 **0.5 배**로 줄인다
(`go2_pedipulation_env.py:399-402`). 조작 다리 슬롯(`a_man`)은 건드리지 않는다 — `a_man` 은
적분형이라 같은 계수를 곱하면 "범위 축소"가 아니라 "hip 이동 속도 감쇠"가 되고, 조작 다리의
abduction 권한은 이 태스크의 목적 자체이기 때문이다.

hip 은 관절 이름(`"hip" in n`)으로 고른다. 이 env 의 관절 순서는 **type-major** 라
`3k` 산술은 틀리고 이름 기반이 맞다.

## 보상 구성 — `smooth4x` 레시피 재현

`smooth4x` 가 사용자 우선순위(① hold 정확도 ② 무진동 ③ 명령 일반화)에서 계보 1~2위였으므로
그 보상 구성을 그대로 쓴다. 상세는
[`../_comparisons/hold_tremor_objective/`](../_comparisons/hold_tremor_objective/).

| 항목 | cfg 기본값 | 이 run | 이유 |
|---|---|---|---|
| `w_base_drift` | −50.0 | **0.0** | 2026-08-03 측정에서 이 항이 **S4 하중 전이 위반을 0.05% → 19.19%** 로 만든 원인. 유일한 hard constraint 다 |
| `w_joint_target_rate` | −0.4 | **0.0** | `smooth4x` 에 없던 항. `smooth4x` 는 이 항 없이 떨림 1.29, 이 항을 넣은 `jtr40_jvf` 는 1.42 였다 |
| `w_action_rate` / `_smooth` | −0.02 / −0.008 | 그대로 | 기본값이 이미 `smooth4x` 수준(원래의 4배)이다 |

## from-scratch 라서 fine-tune 과 반대로 둔 것 2개

| 항목 | 값 | 이유 |
|---|---|---|
| `command.box_init` | **(0.06, 0.05, 0.08)** 기본값 | S1 from-scratch 도 이 값으로 커리큘럼을 작게 시작했다. `_pushdr` fine-tune 에서는 반대로 `box_max` 로 **고정해야** 했다 (resume 이 커리큘럼 상태를 복원하지 않으므로) |
| `command.resample_time_*` | **3.0 / 5.0** 기본값 | `smooth4x` 의 *실효* 조건은 `_post_physics_step` 버그로 이게 죽어 있었으나, **버그의 부작용을 일부러 재현하지는 않는다.** 이번은 변수 격리가 아니라 새 계보이므로 고쳐진 env 거동을 쓴다 |

⚠ 따라서 이 run 은 `smooth4x` 와 **보상은 같고 명령 분포는 다르다** (에피소드 중 명령이
3~5 s 마다 바뀐다). `smooth4x` 대비 차이는 `hip_scale_reduction` 하나가 아니라 둘이다.
순수 A/B 가 아니며, 그렇게 읽지 말 것.

## 기동 시 확인한 것

```
Learning iteration 6/20000              ← resume 아님
Metric/cmd_box_x: 0.0600  y: 0.0500  z: 0.0800   ← 커리큘럼 작게 시작
Episode_Reward/base_drift: 0.0000
Episode_Reward/joint_target_rate: 0.0000
Metric/track_err_m: 0.2548              ← 미학습 정책
```

## 비교 기준선 (전부 `hip_scale_reduction` **없이** 학습된 값)

| | `smooth4x` | `jtr40_jvf` |
|---|---:|---:|
| hold 오차 | 10.18 mm | 15.60 mm |
| 떨림 | 1.29 °/step | 1.42 |
| 원 RMSE ω=0.5 | 17.24 mm | 19.72 |
| step 재수렴 p95 | 0.68 s | 4.37 |
| S4 위반 / 하중 | 0.00% / 9.9% | 0.05% / 24.5% |
| 몸통 표류 | 6.28 cm | 5.52 |
| push 150 N 3족 | 53.0% | 76.5% |

**미지수: hip abduction 권한을 절반으로 줄인 것이 지지 안정성(push·표류)에 주는 영향.**
hip 은 좌우 균형의 주 액추에이터이므로 push 강건성이 떨어질 수 있다. 반대로 액션 범위가
줄어 떨림은 유리할 수 있다.

## 평가 시 반드시

`play_pedipulation_gates.py` 의 `--hip_scale_reduction` 은 기본 `auto` 로, 체크포인트 옆
`params/env.yaml` 에서 학습 값을 읽어 맞춘다. 키가 없는 구 run 은 `False` 로 처리된다.
**이 자동 판별이 없으면 cfg 기본값 `True` 가 구 정책에도 걸려 조용히 분포 밖 평가가 된다.**

## 대조군

[`../2026-08-03_15-53-29_hipscale_scratch_s10x/`](../2026-08-03_15-53-29_hipscale_scratch_s10x/)
— 이 run 과 **action 규제 배율만** 다르다 (4배 → 10배). 나머지 9개 항목은 `params/env.yaml`
대조로 동일함을 확인했으므로 단일 변수 A/B 다.

구 계보에서 10배는 떨림을 계보 최저(0.91 °/step)로 낮췄지만 push 를 53.0 → 30.6% 로 절반
팔았다. `hip_scale_reduction` 도 액션 권한을 줄이는 조치라, 둘이 겹쳤을 때의 push 가 관측
대상이다.

## 상태

**진행 중** (시작 2026-08-03 13:13, GPU 1, ETA ~9 h 20 m). 완료 후 hold / push / circle /
step / contact / drift 6종 평가 예정.

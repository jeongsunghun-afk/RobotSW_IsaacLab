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

## 결과 — **학습 실패.** 커리큘럼이 한 번도 승급하지 못했다

20000 iteration 을 모두 돌았으나(`model_19999.pt`), **명령 박스가 초기값에서 전혀 자라지
않았다.**

| | 값 |
|---|---|
| 커리큘럼 평가 횟수 | **9891 회** |
| 커리큘럼 승급 횟수 | **0 회** |
| 최종 `cmd_box` | **(0.06, 0.05, 0.08)** = `box_init` 그대로 |
| `track_err_m` (tail-100) | 0.0860 m |
| `hold_rate` (tail-100) | 0.366 |
| `Policy/mean_std` | 0.352 (iter 250 최저) → **2.826** (최종) |

승급 조건은 `mean_err < curriculum_err_threshold = 0.06 m`
(`go2_pedipulation_env.py:899`). `Metric/track_err_m` 이 전체 iteration 의 17.0% 에서 0.06
아래로 내려갔지만, 이는 그 스텝에 리셋된 env 배치의 순간값이다. 승급 판정은 **≥200 에피소드
누적 평균**이라 순간 딥이 평활화되고, 그 평균은 9891 회 전부 0.06 을 넘었다.

**따라서 이 run 의 지표를 10배 run 과 "정책 품질"로 나란히 읽으면 안 된다.** 4배는 가장
**쉬운** 박스(0.06)에서 측정된 값이고 10배는 가장 **어려운** 박스(0.20)에서 측정된 값인데,
그런데도 4배가 더 나쁘다.

### 실패 신호 — std 단조 발산

```
iter   250 :  std 0.352  ← 최저
iter  5600 :  std 0.787
iter 12000 :  std 1.424
iter 19999 :  std 2.826  ← 끝까지 단조 증가
```

이 저장소에서 이미 관측된 실패 서명이다
(`project_isaac60_entropy_coef_std_divergence`). 10배 run 은 같은 구간에서 0.50 → **0.23**
으로 떨어진 뒤 최종 1.04 에 머물렀다.

### 해석과 한계

`hip_scale_reduction` 이 좌우 균형의 주 액추에이터 권한을 절반으로 줄인 상태에서, 4배 규제는
안정적인 해를 찾기에 부족했던 것으로 보인다. 10배의 강한 action 페널티는 고분산 액션을 직접
억제하므로 std 발산에 반대로 작용한다 — 기전은 있다.

⚠ **그러나 arm 당 n=1 이다.** std 발산은 확률적 실패 모드이므로 이 한 쌍만으로 "4배는 구조적으로
불가능"이라고 단정할 수 없다. 시드가 같아도(둘 다 42) 보상이 다르면 궤적은 첫 업데이트부터
갈라지므로 시드 통제는 도움이 되지 않는다. 단정하려면 4배를 다른 시드로 재실행해야 한다.

## 상태

**완료** (2026-08-03 13:13 ~ 2026-08-04, GPU 1, 20000 iteration). 6종 평가 진행 중이나,
위 사유로 이 체크포인트의 평가값은 **박스 0.06 조건의 값**임을 감안해 읽어야 한다.

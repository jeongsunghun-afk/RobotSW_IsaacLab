# May 21 Parkour 실험 — 학습 결과 분석 (tfevents)

> 데이터 소스: 각 실험의 `events.out.tfevents.*`를 `tensorboard.event_accumulator`로 파싱.
> 통계 기준: `last100` = 마지막 100 iter 평균 (수렴 시점).

## 요약 표

| #   | 실험                                          | iters     | last_mean_reward | last_terrain_level | last_ep_len | 판정          |
| --- | --------------------------------------------- | --------- | ---------------- | ------------------ | ----------- | ------------- |
| 1   | 09-49-21 goal_idx_visualization3              | **21282** | 18.46            | 5.45               | 654         | 🟢 Green      |
| 2   | 10-05-51 clip_actions_16                      | 17574     | 18.67            | 5.40               | 663         | 🟢 Green      |
| 3   | 10-11-16 clip_actions10_action_rate_0.01      | 3078      | 7.91             | 4.57               | 707         | 🟡 Yellow (중도 중단) |
| 4   | 14-49-35 change_terrain_difficulty (1st try)  | **111**   | 3.74             | 0.20               | 565         | 🔴 Red (abort) |
| 5   | 14-58-19 change_terrain_difficulty (2nd try)  | 11864     | **14.30**        | **6.19**           | **860**     | 🟢 Green      |
| 6   | 15-48-19 feet_dragging                        | 16009     | **24.88** ⭐      | 6.00               | **890** ⭐  | 🟢 Green (Best) |
| 7   | 17-31-36 add_feet_air_time_flat               | 9521      | **0.004** ❌     | **0.03** ❌        | **131** ❌   | 🔴 Red (collapse) |

## Terrain Curriculum Level (5종 sub-terrain별, last100 평균)

| #   | 실험                                  | flat | hurdle | step    | gap  | stair |
| --- | ------------------------------------- | ---- | ------ | ------- | ---- | ----- |
| 1   | goal_idx_visualization3               | 6.02 | 6.10   | **3.86** | 5.94 | 6.00  |
| 2   | clip_actions_16                       | 6.02 | 5.97   | **3.84** | 5.98 | 6.03  |
| 3   | clip_actions10_action_rate_0.01       | 6.06 | 6.59   | **0.40** | 6.40 | 6.24  |
| 4   | change_terrain_difficulty (abort)     | 0.65 | 0.23   | 0.38     | 0.29 | 0.24  |
| 5   | change_terrain_difficulty (retry)     | 6.10 | 6.19   | **6.29** | 6.09 | 6.15  |
| 6   | feet_dragging                         | 6.01 | 6.10   | 5.96     | 5.96 | 6.08  |
| 7   | feet_air_time                         | 0.05 | 0.06   | 0.07     | 0.00 | 0.04  |

**관찰**:
- Exp 1-2의 step terrain은 **3.85에 정체** — 다른 4개 (flat/hurdle/gap/stair)는 6.0 도달했으나 step만 뒤처짐.
- Exp 5의 terrain_difficulty 변경(x_length 0.4-0.8 → 1.2-2.0, num_steps 4→8)으로 step level 0.40 → **6.29**로 해결.
- Exp 6 feet_dragging은 step level이 5.96으로 안정 유지 (terrain cfg는 baseline).
- Exp 4는 fresh start로 첫 100 iter에 abort, Exp 5에서 동일 cfg로 재시작해 성공 — **즉 cfg 자체는 정상, 단지 첫 시도가 short crash**.

## 개별 Reward Component (last100 평균)

| # | 실험                       | tracking_goal_vel | tracking_yaw | collision  | action_rate | feet_stumble | feet_edge | feet_dragging | feet_air_time |
| - | -------------------------- | ----------------- | ------------ | ---------- | ----------- | ------------ | --------- | ------------- | ------------- |
| 1 | goal_idx_vis              | 0.944             | 0.297        | -0.082     | -0.121      | -0.002       | -0.045    | -             | -             |
| 2 | clip_actions_16            | 0.958             | 0.302        | -0.069     | -0.122      | -0.003       | -0.047    | -             | -             |
| 3 | clip10_action_rate_0.01    | 0.761             | 0.287        | **-1.735** | -0.067      | -0.024       | -0.119    | -             | -             |
| 4 | change_terrain (abort)     | 0.258             | 0.208        | **-3.366** | -0.157      | -0.075       | 0.000     | -             | -             |
| 5 | change_terrain (retry)     | **1.170**         | 0.379        | -0.621     | -0.271      | -0.015       | -0.139    | -             | -             |
| 6 | feet_dragging              | **1.293** ⭐       | **0.407**    | -0.146     | -0.165      | -0.003       | -0.046    | **-0.023**    | -             |
| 7 | feet_air_time              | **0.011** ❌      | 0.042        | -0.362     | -0.105      | -0.003       | 0.000     | -0.014        | **-0.0007** ⚠️ |

**핵심 관찰**:
- Exp 3 (action_rate 5배 완화): **collision -1.74**로 폭증 — penalty 줄이자 충돌이 늘어남. action_rate가 안정성에 기여하던 것.
- Exp 6: tracking_goal_vel 1.293 (모든 실험 중 최고), feet_dragging도 음수로 활성 → 페널티 작용 중이지만 학습은 잘 진행.
- Exp 7: **tracking_goal_vel 0.011** (정상의 1/100), **feet_air_time이 음수 (-0.0007)** — 보상이 아니라 페널티로 작동!

## Termination 분석 (last100 평균 — episode당 횟수)

| # | 실험                       | base_contact | tilt        | low_height | goal_reached | time_out |
| - | -------------------------- | ------------ | ----------- | ---------- | ------------ | -------- |
| 1 | goal_idx_vis              | 0.016        | 0.053       | 0.005      | 0.000        | 6.26     |
| 2 | clip_actions_16            | 0.012        | 0.073       | 0.002      | 0.000        | 6.18     |
| 3 | clip10_action_rate_0.01    | 0.204        | **1.267**   | 0.030      | 0.000        | 5.79     |
| 4 | change_terrain (abort)     | 0.055        | **7.078**   | 0.008      | 0.308        | 9.53     |
| 5 | change_terrain (retry)     | 0.045        | 0.446       | 0.027      | **2.891**    | 4.83     |
| 6 | feet_dragging              | 0.011        | 0.105       | 0.008      | **3.198** ⭐  | 4.66     |
| 7 | feet_air_time              | 0.091        | **23.106** ❌ | **8.490** ❌ | 0.000        | **31.29** ❌ |

**핵심**: Exp 7은 **tilt termination이 23.1**(정상 0.05-0.1 대비 200-400배), **low_height 8.49**, **time_out 31.3** — 로봇이 거의 매 step 넘어지고 있음.

## 변화된 Reward 집중 분석

### Exp 3: action_rate_l2 scale -0.05 → -0.01 (5배 완화)
- 의도: action rate penalty가 학습 진행 억제 → 완화로 더 빠른 제어 학습 기대.
- **실제**: 단 3078 iter에서 중단. terrain_level 4.57로 Exp1/2 (5.4) 대비 후퇴.
- **부작용**: collision penalty 평균 -1.74 (Exp 1/2 대비 25배 ↑). action_rate가 충돌 방지에 기여했음을 시사.
- **결론**: 단순 5배 완화는 단기적으로 충돌 증가를 야기. 0.01보다는 0.02-0.03 정도 절충 추천.

### Exp 4 vs 5: terrain_difficulty 변경 (동일 cfg)
- Exp 4 (111 iter abort): terrain_level 첫 100 iter에서 1.54 → 0.20으로 추락. tilt 7.08.
- Exp 5 (11864 iter): 동일 cfg로 재시작, 최종 level 6.19 도달, step 6.29로 가장 높음.
- **의미**: cfg는 정상. Exp 4는 PPO 초기 변동성 + 어려운 terrain의 우연한 조합으로 조기 발산 (시드 영향 추정). 두 번째 시도에서 정상 학습.
- **단, Exp 5의 mean_reward 14.30은 Exp 1/2 (18.5)보다 낮음** → 어려운 terrain에서 절대 reward 감소는 자연스러움.

### Exp 6: feet_dragging -0.1 추가
- **최고 성능**: mean_reward 24.88, ep_len 890. tracking_goal_vel도 1.29로 최고.
- terrain_level 6.00 안정 유지.
- feet_dragging 값 자체는 -0.023 (작은 활성) — 페널티가 부드럽게 작동.
- **결론**: hind feet dragging 페널티는 학습 안정성 + 동작 품질 모두에 긍정적.

### Exp 7: feet_air_time +0.5 추가 — **로봇 망가짐**

**시계열 추이 (mean_reward & terrain_level)**:

```
iter        mean_reward    terrain_level
   0          0.006          1.54
 680         12.84           6.53   ← 정상 학습 진행
1360         10.69           6.27
2040          8.97           6.33
2720         10.26           6.00
3400         12.63           5.36
4080         13.00           5.37
4760         10.29           6.48
5440          8.43           5.55
6120          0.003 ❌       0.00 ❌   ← 약 6000 iter에서 catastrophic collapse
6800          0.012          0.00
9520          0.004          0.03   ← 회복 불가능, 학습 종료까지 무너진 상태
```

**무엇이 무너졌는가**:
- mean_reward: 13.0 (4080 iter) → **0.003** (9520 iter)
- terrain_level: 6.48 → **0.034**
- tracking_goal_vel: 1.18 → **0.01** (목표 추종 완전 상실)
- tilt termination: 정상 0.07 → **23.1** (매 episode 마다 거의 매 step 전복)
- ep_len: 정상 660 → **131**

**근본 원인 (정량 증거)**:
- `feet_air_time` 보상의 시계열 값: 학습 내내 **음수** (-0.0002 ~ -0.0121).
- 계산식: `(last_air_time - threshold(=0.3s)) × first_contact × cmd_gate × is_flat`
- Go2의 자연 보행 cycle은 ~0.25 s/step → `last_air_time < 0.3`이 대부분 → **(0.25 - 0.3) = -0.05** 형태의 음수 보상이 누적.
- scale +0.5를 곱하면 발 4개 × first_contact 1개 × 음수 → 매 step **마이너스 보상**으로 작용.
- 결과: 정책은 발의 접지 자체를 회피하도록 학습 → 불안정한 다리 동작 → tilt 폭증 → 전복.

**즉, "+0.5 air-time bonus를 추가"라는 의도와 달리, 실제로는 짧은 보행 사이클을 패널티로 학습시켰음.**

`is_flat` 게이트도 의심:
- 만약 게이트가 hurdle/step terrain에서 갑자기 꺼지면 보상 신호가 불연속 → 정책 가치 추정 불안정.
- 게이트가 항상 True이면 (모든 terrain에서) ANYmal식 보행을 강요 → parkour task와 충돌.

## 실험 간 비교 결론

| 변경            | 효과                                                       |
| --------------- | ---------------------------------------------------------- |
| 진단 로깅 추가  | 영향 없음 (Exp 1)                                          |
| clip_actions 16 | Exp 1과 거의 동일 — 보수적 한계로 묶인 학습은 클리핑 16에 거의 도달하지 않음 |
| action_rate -0.01 | 충돌 25배 ↑, 학습 후퇴. **추천하지 않음**                 |
| terrain 난이도 ↑ | 1차 abort(시드 영향), 2차 성공. step terrain 정체 해결    |
| feet_dragging   | **순효과 +**. mean_reward 24.9로 최고 + 안정 유지          |
| feet_air_time   | **catastrophic failure** — 음수 보상으로 잘못 작동           |

## 사용자 보고와의 일치

> "feet_air_time reward를 추가했을 때에는 로봇이 완전히 망가지는 경우가 발생함"

✅ 정량 확인:
- iter 5440 이후 약 6000 iter에서 mean_reward 8.4 → 0.003으로 폭락.
- terrain_level 5.55 → 0.00으로 curriculum 초기화 (자동 demotion 반복).
- 9520 iter까지 회복 흔적 없음 (3500 iter 동안 reward ≈ 0 유지).
- 원인은 **feet_air_time 보상이 실제로는 음수**로 작동했기 때문 (임계값 0.3 s가 Go2의 자연 사이클보다 큼).

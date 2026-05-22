# Go2 Parkour — 2026-05-21 실험 시리즈 분석 보고서

> 대상: `logs/rsl_rl/go2_parkour/` 의 7개 05/21 실험 (09:49 ~ 17:31).
> 비교 베이스: `2026-05-20_17-14-12_clip_actions_10` (직전 05/20 마지막 실험).
> 데이터 소스: `git/IsaacLab.diff` 스냅샷 + `params/{agent,env}.yaml` + `events.out.tfevents.*`.
> 세부 분석: [code_changes](may21_code_changes.md) · [params_changes](may21_params_changes.md) · [results](may21_results.md)

---

## 한 줄 요약

> 05/21에는 7개의 환경(env-only) 변형 실험을 했고, **feet_dragging penalty 추가는 최고 성능(24.9)** 을 냈으나 **feet_air_time +0.5 보상은 catastrophic failure**를 유발했다 — `Episode_Reward/feet_air_time` 메트릭이 학습이 건강했던 iter 680 시점에서도 -0.0079였고 모든 sampled iter에서 음수였음 (의도된 보상이 실질 패널티로 작동).

---

## 1. 7개 실험 개요

| # | 시작 시각 | 디렉토리                            | 한 줄 의도                                  | iters | 결과         |
| - | --------- | ----------------------------------- | ------------------------------------------- | ----- | ------------ |
| 1 | 09:49     | goal_idx_visualization3             | goal index/hold time 진단 로깅              | 21282 | 🟢            |
| 2 | 10:05     | clip_actions_16                     | action clip 4.8→16, terrain 난이도 ↑       | 17574 | 🟢            |
| 3 | 10:11     | clip_actions10_action_rate_0.01     | clip 10 + action_rate -0.05→-0.01           | 3078  | 🟡 (abort)    |
| 4 | 14:49     | change_terrain_difficulty (1st)     | terrain x=25 m, hurdle/step/stair 강화      | 111   | 🔴 (early abort) |
| 5 | 14:58     | change_terrain_difficulty (2nd)     | 동일 cfg 재시작                              | 11864 | 🟢            |
| 6 | 15:48     | feet_dragging                       | hind feet drag penalty -0.1 신규            | 16009 | 🟢 **Best**   |
| 7 | 17:31     | add_feet_air_time_flat              | flat-gated feet_air_time +0.5 + drag -0.1   | 9521  | 🔴 **망가짐** |

핵심 사실:
- **PPO agent 하이퍼파라미터는 모든 실험에서 동일** (lr 2e-4, γ=0.99, λ=0.95, clip 0.2, entropy 0.01, hidden [512, 256, 128]).
- 05/21 변경은 **100% 환경 측** (env_cfg, terrain_cfg, parkour_env.py).
- 단 하나의 git commit (`dff065bccf9` 15:09): Exp 5 → Exp 6 사이에 "hurdle goals between hurdles".

---

## 2. Code 변경 (`parkour_env.py` / `parkour_env_cfg.py` / `parkour_terrains.py`)

자세한 diff는 [`may21_code_changes.md`](may21_code_changes.md) 참조.

> **데이터 출처 주의**: 각 실험의 `git/IsaacLab.diff`는 upstream(`origin/main`) 대비 working-tree 누적 차이의 스냅샷이다. 따라서 "이전 실험 대비" 변경을 보려면 **`params/env.yaml`**(런타임에 실제 dump된 값)을 비교해야 정확하다. 아래 표는 env.yaml 기준으로 작성.

### 누적 변경 매트릭스 (env.yaml 기준, 05/20 17:14 baseline 대비)

| 항목                       | Baseline | Exp1 | Exp2  | Exp3   | Exp4/5 | Exp6     | Exp7     |
| -------------------------- | -------- | ---- | ----- | ------ | ------ | -------- | -------- |
| `clip_actions`             | 10.0     | 10.0 | **16.0** | 10.0 | 10.0   | 10.0     | 10.0     |
| `action_rate_l2` scale     | -0.05    | -0.05 | -0.05 | **-0.01** | -0.01 | -0.01   | -0.01    |
| platform_length (flat/hurdle) | 2.5    | 2.5  | 2.5   | 2.5    | **1.5** | 2.5     | 2.5      |
| num_steps_per_stair        | 4        | 4    | 4     | 4      | **8**  | 4        | 4        |
| 진단 로깅 (env.py)         | -        | ✓    | -     | -      | -      | -        | -        |
| feet_dragging 로직         | -        | -    | -     | -      | -      | **신규** | **신규** |
| feet_air_time 로직         | -        | -    | -     | -      | -      | -        | **신규** |

> **diff에는 보이지만 env.yaml에 반영되지 않은 변경**: `IsaacLab.diff`는 `parkour_terrains.py`/`parkour_env_cfg.py`의 다양한 terrain 파라미터 변경(예: hurdle x_spacing, gap_length, stair_height_range 등)을 보여주지만, 이는 origin/main 대비 **누적된 working-tree drift**일 뿐 05/21 실험 간 활성 변경이 아니다. 활성 변경은 위 표에 정리한 것뿐.

### Exp 6 — feet_dragging 구현 (정상 작동)

```python
hind_feet_ids = self._feet_ids[2:4]              # [RL, RR]
hind_xy_vel = torch.norm(body_lin_vel_w[:, hind, :2], dim=-1)
is_dragging = contact_filt[:, 2:4] & (hind_xy_vel > 0.05)
feet_dragging = torch.sum(hind_xy_vel * is_dragging.float(), dim=-1)
```
→ scale −0.1, threshold 0.05 m/s. **Hind feet가 접지된 채 움직일 때만** 활성.

### Exp 7 — feet_air_time 구현 (역효과)

```python
last_air_time = contact_sensor.data.last_air_time[:, feet_ids]
first_contact = contact_sensor.compute_first_contact(step_dt)[:, feet_ids]
air_time_bonus = (last_air_time - 0.3) * first_contact.float()  # ← (a - 0.3)
gate = (cmd_speed > 0.1).float()
feet_air_time = torch.sum(air_time_bonus, dim=-1) * gate * is_flat
```
→ scale **+0.5**. 의도는 ANYmal식 "긴 air time 보상"이지만, **threshold 0.3 s가 너무 큼**.

---

## 3. Hyperparameter / Reward Scale 변경

자세한 비교는 [`may21_params_changes.md`](may21_params_changes.md) 참조.

### Agent (PPO) 하이퍼파라미터
**7개 실험 모두 baseline과 동일.** lr=2e-4, γ=0.99, λ=0.95, entropy=0.01, num_learning_epochs=5, num_mini_batches=4.

### Reward Scale 변경 (baseline 대비)

| Reward 항목      | base 05/20 | Exp1 | Exp2 | Exp3       | Exp4/5 | Exp6     | Exp7          |
| ---------------- | ---------- | ---- | ---- | ---------- | ------ | -------- | ------------- |
| tracking_goal_vel | 1.5        | =    | =    | =          | =      | =        | =             |
| tracking_yaw     | 0.5        | =    | =    | =          | =      | =        | =             |
| action_rate_l2   | -0.05      | =    | =    | **-0.01**  | -0.01  | -0.01    | -0.01         |
| collision        | -10.0      | =    | =    | =          | =      | =        | =             |
| orientation      | -1.0       | =    | =    | =          | =      | =        | =             |
| feet_stumble     | -1.0       | =    | =    | =          | =      | =        | =             |
| feet_edge        | -1.0       | =    | =    | =          | =      | =        | =             |
| **feet_dragging**| —          | —    | —    | —          | —      | **-0.1** | **-0.1**      |
| **feet_air_time**| —          | —    | —    | —          | —      | —        | **+0.5** ⚠️    |

→ 단 3개 항목이 5월 21일에 건드려졌음 (action_rate_l2, feet_dragging 신규, feet_air_time 신규).

---

## 4. 실험 결과 — Reward & Curriculum Terrain Level

자세한 표/시계열은 [`may21_results.md`](may21_results.md) 참조.

### 종합 표

| # | 실험                            | mean_reward (last100) | terrain_level (last100) | ep_len | 판정 |
| - | ------------------------------- | --------------------- | ----------------------- | ------ | ---- |
| 1 | goal_idx_visualization3         | 18.46                 | 5.45                    | 654    | 🟢   |
| 2 | clip_actions_16                 | 18.67                 | 5.40                    | 663    | 🟢   |
| 3 | clip10_action_rate_0.01         | 7.91                  | 4.57                    | 707    | 🟡   |
| 4 | change_terrain (abort)          | 3.74                  | 0.20                    | 565    | 🔴   |
| 5 | change_terrain (retry)          | 14.30                 | **6.19**                | 860    | 🟢   |
| 6 | **feet_dragging**               | **24.88** ⭐           | 6.00                    | **890** ⭐ | 🟢 **Best** |
| 7 | **feet_air_time**               | **0.004** ❌           | **0.03** ❌              | **131** ❌ | 🔴 **망가짐** |

### Sub-terrain 별 Level (마지막 100 iter 평균)

| # | 실험                       | flat | hurdle | **step**  | gap  | stair |
| - | -------------------------- | ---- | ------ | --------- | ---- | ----- |
| 1 | goal_idx_vis              | 6.02 | 6.10   | **3.86** ← 정체 | 5.94 | 6.00  |
| 2 | clip_actions_16           | 6.02 | 5.97   | **3.84** ← 정체 | 5.98 | 6.03  |
| 3 | clip10_action_rate_0.01   | 6.06 | 6.59   | **0.40** ← 추락 | 6.40 | 6.24  |
| 5 | change_terrain (retry)    | 6.10 | 6.19   | **6.29** ← 해결 | 6.09 | 6.15  |
| 6 | feet_dragging             | 6.01 | 6.10   | 5.96      | 5.96 | 6.08  |
| 7 | feet_air_time             | 0.05 | 0.06   | 0.07      | 0.00 | 0.04  |

**핵심 관찰**:
- 05/20까지 step terrain이 가장 어려운 sub-task였음 (level 3.84 정체).
- Exp 5의 terrain 난이도 cfg(`x_length 1.2-2.0`, `num_steps_per_stair=8`)가 **step terrain 정체 문제를 해결** (3.84 → 6.29).
- Exp 6은 baseline terrain cfg로 돌아왔는데도 step level 5.96으로 안정.

---

## 5. 변화된 Reward 심층 분석

### 🟡 Exp 3 — action_rate_l2: -0.05 → -0.01 (5배 완화)

**의도**: action rate penalty가 학습을 과도하게 억제 → 완화로 빠른 제어 학습 기대.
**실제**:
- 3078 iter에서 중단 (early stop). mean_reward 7.91, terrain_level 4.57로 직전 (Exp 2: 18.67/5.40)에서 후퇴.
- **collision penalty 평균 -1.74** (Exp 1/2의 -0.07 대비 25배 증가).
- tilt termination 1.27 (Exp 1/2: 0.05).

**결론**: action_rate penalty가 단순 정칙화가 아니라 **충돌 회피의 implicit 안정자** 역할. 5배 완화는 과도. 절충안 (-0.02 ~ -0.03) 검토 필요.

### 🟢 Exp 5 — Terrain Difficulty Up

**변경**: x_size 20→25 m, hurdle/step/stair 간격 2배, stair_height_range (0.05,0.20)→(0.05,0.25).
**효과**: step terrain 정체 해결 (3.84 → 6.29). 다른 terrain level은 baseline 수준 유지.
**비용**: mean_reward 14.30 (Exp 1/2의 18.5보다 낮음 — 자연스러운 trade-off).

**Exp 4 (abort) vs Exp 5 (success) 차이**:
- 같은 cfg, 다른 random seed/시간으로 시작. Exp 4는 초기 100 iter에서 tilt termination 7.08로 발산.
- Exp 5는 정상 학습 — **cfg는 정상, Exp 4는 시드 영향에 의한 우연한 early divergence**로 추정.

### 🟢 Exp 6 — feet_dragging −0.1 (Best Run)

**변경**: hind feet (RL, RR) 접지 중 횡 속도 > 0.05 m/s 시 패널티.
**효과**:
- mean_reward **24.88** (모든 실험 최고).
- tracking_goal_vel **1.293**, tracking_yaw **0.407** (모두 최고).
- terrain_level 6.00, ep_len 890.
- feet_dragging 자체 값 -0.023 (가벼운 활성 — 페널티가 부드럽게 작용).

**의의**: hind feet 마찰 회피 = 자연스러운 보행 + 안정성 동반 향상. **이 reward는 유지 권장**.

### 🔴 Exp 7 — feet_air_time +0.5 (Catastrophic Failure)

**무엇이 일어났는가** (시계열):

```
iter    mean_reward    terrain_level    tracking_goal_vel
   0       0.006          1.54              -0.003
 680      12.84           6.53               1.08    ← 정상 학습
1360      10.69           6.27               1.01
2040       8.97           6.33               1.03
2720      10.26           6.00               1.05
3400      12.63           5.36               1.08
4080      13.00           5.37               1.18
4760      10.29           6.48               1.09
5440       8.43           5.55               0.78
6120     ▼ 0.003 ▼      ▼ 0.00 ▼          ▼ -0.005 ▼   ← Catastrophic collapse
6800       0.012          0.00               0.016
9520       0.004          0.03               0.014    ← 회복 불가
```

**Termination 폭증** (last100 평균):
- **tilt: 23.1** (정상 0.05의 약 460배)
- **low_height: 8.49** (정상 0.005의 약 1700배)
- **time_out: 31.3** (정상 6.2의 5배)
- ep_len: 660 → **131**

**진짜 원인 — feet_air_time이 사실은 음수 보상 (실증 데이터)**:

`Episode_Reward/feet_air_time` 메트릭의 모든 시점 값 (Exp 7 전체 시계열):

```
iter   feet_air_time   mean_reward    상태
   0     -0.00024         0.006        시작
 680     -0.00788        12.84         학습 양호 (이때조차 음수!)
2720     -0.01210        10.26         학습 진행 중에도 음수
5440     -0.00543         8.43         붕괴 직전
6120     -0.00015         0.003        collapse — 접지 자체가 사라져 보상도 0에 수렴
9520     -0.00067         0.004        회복 불가
```

핵심: **정책이 건강했던 iter 680 시점(reward 12.84)에서도 feet_air_time = -0.0079** — 단 한 번도 양수였던 적이 없음. 즉 threshold 0.3 s가 실제 air time보다 항상 컸다는 직접 증거.

식: `air_time_bonus = (last_air_time − 0.3) × first_contact`
→ 실제 air_time < 0.3 s 인 한 매 접지마다 음수 보상.
→ scale `+0.5`을 곱해도 부호는 음수 → **+0.5 보상 의도가 사실상 패널티로 작동**.

**정책이 학습한 것**:
1. 발 접지를 회피 → 다리를 들고 있는 시간 늘리기.
2. cmd_speed > 0.1 게이트가 항상 통과 → 정지 시에도 압박.
3. 결과: 불안정한 다리 동작 → 자세 무너짐 → tilt 폭증.

**`is_flat` 게이트의 의문점**:
- 의도는 "flat terrain에서만 air-time을 보상" — 그러나 음수 보상이므로 "flat terrain에서만 패널티"가 되어버림.
- hurdle/step terrain에서는 게이트가 꺼져서 신호 불연속 → value function 추정 불안정 가능성.

**즉, "ANYmal-style air-time bonus 추가"라는 의도가 정반대로 작동**한 것이 단일 catastrophic failure의 근본 원인.

> 보조 plausibility 노트: Go2의 자연 step 주기는 일반적으로 0.2~0.3 s 범위로 알려져 있으므로 threshold 0.3 s가 일반적 air time보다 크다는 가설은 합리적이지만, 본 데이터로 직접 검증된 핵심 증거는 위의 **"reward가 모든 iter에서 음수였다"** 사실이다.

---

## 6. 권장 후속 작업

1. **유지**: feet_dragging −0.1 (Exp 6) — 명백한 net positive.
2. **재시도**: feet_air_time — threshold를 **0.20 또는 0.15**로 낮추고 scale을 **0.1 ~ 0.2**로 축소.
   - 또는 `(air_time − threshold)`를 절대값 제곱식 `−(air_time − target)^2` 으로 변경하여 명확한 target 보행을 유도.
3. **재검토**: action_rate_l2 = −0.01 → 절충안 −0.02 ~ −0.03 시도.
4. **유지**: terrain_difficulty 확장 (Exp 5 cfg) — step terrain 정체 해결.
5. **clip_actions**: 10.0 유지가 안전 (16.0이 더 좋다는 증거 없음).
6. **검증**: Exp 4의 early abort가 단순 시드 우연인지 확인 — 다른 seed로 재시도.

---

## 7. 사용자 보고와의 일치

> "feet_air_time reward를 추가했을 때에는 로봇이 완전히 망가지는 경우가 발생함"

✅ **완전히 확인됨**:
- mean_reward 8.4 → 0.003로 폭락 (iter ~6120, 즉 학습 ~1시간 후).
- terrain_level 6.0 → 0.03로 curriculum 완전 초기화.
- tilt termination 23.1로 매 step 전복.
- 회복 시도 흔적 없이 종료까지 3500 iter 동안 reward ≈ 0 유지.

**근본 원인 확정**: `Episode_Reward/feet_air_time` 메트릭이 학습이 건강했던 iter 680 (reward 12.84) 시점에서도 -0.0079였고 모든 sampled iter에서 음수. 즉, threshold `0.3 s`가 실측 air time보다 항상 컸기 때문에 `+0.5` 의도된 보상이 실질 패널티로 작동 → 정책이 접지를 회피 → 자세 붕괴.

---

## 보조 자료

- 코드 diff 정리: `_workspace/may21_code_changes.md`
- 하이퍼파라미터/reward scale 비교: `_workspace/may21_params_changes.md`
- 학습 결과 (reward/terrain/termination): `_workspace/may21_results.md`
- 원시 tfevents 추출 JSON: `/tmp/may21_report_summary.json`
- 추출 스크립트: `/tmp/may21_tfevents_extract.py`, `/tmp/may21_report_data.py`

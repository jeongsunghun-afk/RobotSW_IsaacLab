# 00 — 종합 보고서: go2_parkour_symmetry 3-Run 분석 (로봇이 왜 망가졌나)

**작성**: worker-1 (Task #4 종합) · **소스**: 01_code_diff.md(코드), 02_reward_scales.md(설정), 03_training_curves.md(학습로그)
**대상 런**: `logs/rsl_rl/go2_parkour_symmetry/` 3개
- 런1 = `2026-06-11_12-13-02_positive_work`
- 런2 = `2026-06-11_16-21-24_no_duty_time_cap`
- 런3 = `2026-06-11_17-28-23_positive_work_0.01`

---

## ⭐ 한 줄 결론

> **3런은 `positive_work` reward weight 하나만 다른 단일변수 실험이다 (런1 -3e-4 / 런2 0 / 런3 -1e-2).
> 런3의 -1e-2는 권장값의 33배 과대로, iter 1100~1580에 비가역 정책 붕괴를 일으켰다.
> 효율 penalty가 tracking 보상을 압도 → `total_reward=clip(min=0)` floor 정박 → advantage 소멸 →
> entropy/std runaway → critic 붕괴. positive_work *항 자체는 무죄*, weight 과대가 단일 원인.**

---

# 사용자 3대 질문 — 직접 답변

## Q1. "코드가 어떻게 바뀌었나?"

**답: 3런 사이의 코드/설정 차이는 `positive_work` 단 한 항뿐이다.** 다른 17개 reward 항과 모든 부수 파라미터는 완전 동일. [Tier 1, diff+yaml 직접 인용]

- **positive_work 항 추가**: 런1에서 `parkour_env.py::_get_rewards()`에 신규 추가 후 커밋(`50b0da2`).
  수식 = `Σ_j max(0, τ_j · q̇_j)` over 12 joints (단위 W, 항상 ≥0; motoring work만 카운트). 세 런 코드 동일.
- **세 런이 다른 것은 오직 weight**:

| reward term | 런1 | 런2 | 런3 |
|---|---|---|---|
| **positive_work** | **-3e-4** | **0 (OFF)** | **-1e-2** ⚠ |
| air_time_cap | 0 | 0 | 0 |
| contact_duty_deficit | 0 | 0 | 0 |
| feet_gait_pairing | 0 | 0 | 0 |
| tracking_goal_vel / tracking_yaw | 1.5 / 0.5 | = | = |
| (나머지 13항: collision -10, feet_stumble/edge -1.0, feet_dragging -0.1, torques_l2 -1e-5 …) | 전부 동일 | = | = |

- **타임라인**: 직전 commit `b0c21f9`에는 air_time_cap=-0.1·contact_duty_deficit=-0.5가 켜져 있었으나,
  **런1이 이 둘을 0으로 끄면서 positive_work=-3e-4를 도입**. 이후 `50b0da2`로 커밋 → 런2/런3의 베이스.
  런2는 거기서 positive_work만 0으로, 런3는 -1e-2로 변경.

## Q2. "reward scale 비교"

위 표 참조. **핵심: 이 3런은 깨끗한 positive_work weight ablation (0 → 권장 -3e-4 → 33×과대 -1e-2).**
- 런3 -1e-2 = 권장 안전값 -3e-4의 **33배**, 경계 위험값 -1e-3의 **10배**. (런1 대비 episodic 기여로는 ~38배)
- cfg 자체 calibration 주석 기준: -1e-3조차 stair p99(857W)에서 tracking 기여의 82%까지 → floor 위험 경고.
  -1e-2는 그 10배 → 고출력 구간에서 tracking을 **수배로 압도**. [Tier 1 설정 + Tier 2 외삽]

## Q3. "로봇이 왜 망가졌나?" (= 런3)

**답: 런3의 과대한 positive_work penalty(-1e-2)가 tracking을 압도 → clip(min=0) floor 정박 →
gradient 소멸 → entropy/std 폭주로 정책이 비가역 붕괴했다.** [Tier 1, tfevents 직접 수치 + 통제변인 분리]

정량 붕괴 (런3, iter 1100→1580 개시·정박, 이후 영구):

| 지표 | 정상(런1 late) | 런3 붕괴 |
|---|---|---|
| Train/mean_reward | 18.99 | **0.000** |
| tracking_goal_vel | 1.100 | **0.003** (정지) |
| Policy/mean_noise_std | 0.570 | **25.8** (단조 폭발) |
| Loss/entropy | 10.17 | **55.85** (단조↑) |
| Loss/value | 6.4e-3 | **5e-10** (critic 붕괴) |
| curriculum/mean_terrain_level | 5.93 | **0.0** (진전 없음) |
| collision | -0.064 | **-48.1** (붕괴자세 지속접촉) |
| 종료 원인 | goal_reached | **time_out**(eplen 999 고착), goal_reached=0 |

**death-spiral 인과 사슬** (모든 변수 단조, 회복 없음):
1. 커리큘럼이 지형↑ → 요구 일량↑ → positive_work penalty -0.79 → -1.5 → -3.0 단조 증가, tracking 상한(+1.1) 초과.
2. "전진" 한계보상이 net 음수화 → 정책이 움직임을 줄임(trkvel 0.65→0.06).
3. total_reward net 음수 → `clip(min=0)` → mean_reward iter~1580에 0.000 고착.
4. advantage 신호 소멸 → 남은 gradient는 entropy bonus뿐 → std 0.55→25.8, entropy→55.85 무한 증폭.
5. 모든 target≈0 → critic V≈0 → value_loss→5e-10.
6. 물리 붕괴: 비종료 붕괴자세 발견 → eplen 999 timeout이지만 collision -48 누적. **영구**.

> 정정: 흔한 실패 모드인 "entropy collapse(탐색 죽음)"가 **아니라 반대 — entropy explosion**. clip floor가 advantage를 없애 entropy bonus만 남긴 결과.

---

# Root Cause (확정)

**단일 root cause = `positive_work` weight = -1e-2 (과대)**. [Tier 1]

- **메커니즘**: 효율 penalty 과대 → tracking 압도 → clip(min=0) floor 정박 → advantage 소멸 → entropy/std runaway → critic 붕괴. (tracking 압도가 floor를 유발하는 **단일 root, 순차 결합** 메커니즘)
- "positive_work 과대 vs contact항 제거 vs 둘다" 중 → **positive_work 과대 단독.** contact 항(air_time_cap/contact_duty_deficit) "제거"는 망가짐과 무관 (아래 함정 정정 참조).

### 통제변인이 단일 원인을 증명
- **런2 (positive_work=0)**: 전 구간 안정(reward 13→19.3 단조, std 0.58–0.76, terrain~6). → 항이 사라지면 붕괴 없음.
- **런1 (같은 항, weight 33배 작음)**: iter~7–9.5k 일시 dip 있으나 std가 0.83에서 멈추고 iter 10248에 reward 16.3 **완전 회복**. → 작은 weight는 floor에 잠깐 닿아도 tracking gradient 생존 → bounded std → 탈출.
- 결정적 차이 = **std/entropy runaway 발생 여부**. 런3만 penalty가 floor를 깊고 영구히 만들어 비가역.

---

# 두 가지 해석 함정 정정 (lead 지시 반영)

### 1. 이름 함정 — "no_duty_time_cap"은 런2 고유 변경이 아니다
`air_time_cap`·`contact_duty_deficit`는 **런1·런2·런3 세 런 모두 0(OFF)**. (코드·yaml·로그 3중 확인: 01 타임라인, 02 표, 03 §1 비활성 태그)
→ "duty/time_cap 제거"를 런2만의 차별점·붕괴 변인으로 오해 금지. **실제 대조축은 positive_work weight 하나.**
런2를 세 런 안에서 실질 구분짓는 유일한 점은 positive_work=0 (= 모든 gait-shaping OFF인 순수 baseline).

### 2. positive_work 항 자체는 무죄 — weight가 과대했을 뿐
- 망가진 건 **항의 존재가 아니라 weight -1e-2의 크기**. 런1(-3e-4)은 같은 항을 켜고도 **회복**했다(붕괴 아님).
- 이전 post-hoc 분석(「positive_work는 pronk 조장 아님 — takeoff가 비쌈」)과 **일관**: 항의 설계 의도는 유효, 단지 calibration 실패.
- `total_reward = clip(min=0)`은 A/B 공통 **의도된 설계**(메모리 등록 사실) → bug 아님, 제거 권고 아님.

---

# 다음 런 권고

1. **[critical] positive_work weight 환원**: `-1e-2 → -3e-4 이하`로 되돌림. cfg calibration 안전대(-3e-4) 준수, 최대 -1e-3 미만(경계값). 효율 penalty의 episodic 기여가 커리큘럼 max terrain에서도 tracking_goal_vel(+1.1)을 **절대 초과하지 않도록** (예: episodic |pw| ≤ 0.3·tracking). **단일 변경**으로 재학습 후 재현/해소 확인.
   - 모니터: 재학습 시 `Policy/mean_noise_std`가 bounded(<1.0) 유지되는지, terrain_level이 floor 안 닿고 상승하는지.
2. **워킹트리 잔재 복원**: 현재 워킹트리에 `positive_work=-1e-2`(런3 값)가 **미복원 상태로 남아 있음** → 다음 런 전 반드시 -3e-4(또는 0)로 되돌릴 것. (현재 그대로 학습 돌리면 런3 붕괴 재현)
3. **clip(min=0) 유지**: 의도된 설계이므로 제거하지 말 것. floor는 증상 전파 경로일 뿐 root cause 아님.
4. **[별도 트랙, 동시 변경 금지] 구조 강건화(선택)**: floor에 닿아도 entropy runaway를 막는 std 상한/entropy_coef 적응은 **1번 단독 검증 이후** 별개로 검토. 원인 분리 위해 절대 1번과 동시 변경 금지.
5. **noise 무시 확인**: 런2/런3 간 `rel_standing_envs`(0.2 vs 0.05)는 `hind_leg` env 소속 → parkour 무관, 변인 아님.

---

# Evidence Tier 요약

- **Tier 1 (확정·직접 증거)**: 3런 코드 차이=positive_work weight (diff 인용) · reward_scales 표 (yaml 인용) · 런3 붕괴 메트릭 및 단조 인과 사슬 (tfevents 직접 수치) · 통제변인 분리(런1 회복/런2 안정/런3 붕괴) · 이름 함정(3런 공통 OFF, 3중 확인).
- **Tier 2 (정량 추론)**: -1e-2가 stair p99에서 tracking을 ~수배 압도한다는 정도(런1 calibration 주석에서 선형 외삽). 단, 압도→floor→붕괴라는 **결과**는 Tier 1 로그로 직접 확증됨.
- 추측/미검증 단정 없음.

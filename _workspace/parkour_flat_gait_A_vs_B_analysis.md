# A vs B — 평지 보행 품질 차이 분석 (B는 trot, A는 어색함)

> 비교 대상
> - **A. `parkour` 환경** — `IsaacLab/.../direct/parkour/` · `DirectRLEnv` 기반 · **Genesis Go2 parkour 포팅**
> - **B. `Isaaclab_Parkour`** — `parkour_isaaclab/envs/...` · `ManagerBasedRLEnv` 기반 · **Extreme Parkour 논문(arXiv 2309.14341) 포팅**
>
> 증상: B는 평지에서 trot 느낌으로 자연스럽게 보행 / A는 평지 보행 자체가 어색 (drag-forward 경향)
> 작성일 2026-05-19 · read-only 분석

---

## Q1. Manager-based vs Direct-based 차이가 원인인가?

### 결론: **아니다.** 프레임워크 추상화 자체는 보행 품질의 원인이 아니다.

`ManagerBasedRLEnv`와 `DirectRLEnv`는 **코드 조직 방식의 차이**일 뿐, 그 아래의 물리 시뮬레이션·PD 제어·step 루프는 동일하다. 동일한 reward/action/control/obs config라면 두 프레임워크는 *물리적으로 같은 정책*을 학습한다.

**증거 — 두 환경의 reward 설계가 거의 동일하다:**

| Reward term | A (Genesis 포팅) | B (extreme_parkour 포팅) |
|---|---|---|
| tracking_goal_vel | **+1.5** | **+1.5** |
| tracking_yaw | **+0.5** | **+0.5** |
| collision | **-10.0** | **-10.0** |
| dof_acc | **-2.5e-7** | **-2.5e-7** |
| hip_pos | **-0.5** | **-0.5** |
| ang_vel_xy | **-0.05** | **-0.05** |
| lin_vel_z | **-1.0** | **-1.0** |
| feet_stumble / feet_edge | -1.0 / -1.0 | -1.0 / -1.0 |
| action / sim 설정 | scale 0.25, decimation 4, sim dt 0.005 | scale 0.25, decimation 4, sim dt 0.005 |

term 집합도 weight도 거의 일치한다. A와 B는 서로를 베낀 게 아니라, **각자 가까운 친척 reference(Genesis Go2 parkour ↔ Extreme Parkour 논문)를 포팅**했고 두 reference가 같은 뿌리를 공유하기에 수렴한 것이다. 프레임워크가 보행 품질을 결정한다면 *같은 reward에서 다른 결과*가 나올 수 없다 → **프레임워크는 원인이 아님.**

### 단, 프레임워크는 "원인이 자라는 토양"이다 (간접 효과)

- **Manager-based(B)**: reward term을 선언적으로 등록/조립. 각 term이 독립 테스트 가능하고, 안 쓰는 term은 *등록 자체를 안 함*.
- **Direct(A)**: `_get_rewards()`가 200줄짜리 손수 작성 메서드(22개 항). divergence가 끼어들기 쉽고, **비활성 term이 `scale=0.0`으로 매달린 채 남는다** — 실제 A에는 `feet_air_time=0.0`, `base_height=0.0`, `tracking_lin_vel_xy_exp=0.0`, `tracking_ang_vel_z_exp=0.0` 4개가 dangling 상태.

→ 프레임워크는 *cause*가 아니라, A 쪽에서 config divergence가 **누적·은폐되기 쉬운 환경**을 제공했을 뿐이다.

---

## Q2. 그러면 실제 원인은 무엇인가 — config divergence 분석

A와 B의 reward 세트는 대부분 일치한다. **차이는 A가 평지(flat) 전용으로 얹은 몇 개의 후처리 항에 집중**되어 있다. A의 어색한 보행은 "B와 다른 설계"가 아니라 **"공통 베이스에 얹은 flat 전용 묶음의 부작용"**으로 보는 것이 정확하다.

### 🔴 1순위 (강한 가설) — `dof_error ×10 on flat` ↔ `feet_air_time 비활성`의 상호작용

두 코드가 **정확히 맞물려 drag-forward를 유발**한다. 표면 추론이 아니라 코드 + git 커밋 메시지의 1차 증거가 있다.

**(a) `dof_error_l2`가 평지에서만 10배 증폭** — `parkour_env.py:1009-1011` (직접 확인)
```python
dof_error_l2 = torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
# Genesis conditional: dof_error penalized 10x more on flat
dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat * 0)
```
- base scale `-0.04` × 10 = **평지 effective weight -0.4**.
- 의미: "관절을 default 자세에서 벗어나게 하지 마라"를 평지에서 강하게 강제.
- **문제**: 보행은 정의상 다리를 default에서 크게 swing해야 한다. -0.4 weight의 "관절 deviation 제곱합" 페널티는 *걷는 행위 자체에 음의 gradient*를 준다.
- B에는 이 증폭이 없다 — B의 `reward_dof_error`는 평지에서도 `-0.04` 그대로(`mdp/rewards.py`). **이것이 A·B의 핵심 divergence.**
- ⚠️ 단, 이 ×10은 주석상 Genesis를 충실히 포팅한 *의도된* 코드다. "버그"가 아니라 "자세 안정화 의도가 보행 학습과 충돌하는 부작용"으로 보는 것이 정확.

**(b) 이를 상쇄할 `feet_air_time`이 구현돼 있으나 꺼져 있음** — `parkour_env_cfg.py:565` (직접 확인)
```python
"feet_air_time": 0.0,  # anymal_c/R_Skeleton 표준 패턴; trot 발현 유도 (NEW 2026-05-13)
```
- git 커밋 `e81b71c8` 메시지: **"Add feet_air_time reward to fix Parkour flat-terrain drag-forward lock-in"**.
- 즉 개발자가 **이미 drag-forward 증상을 인지**하고 counter-reward를 *구현*했으나, **`scale=0.0`으로 비활성**인 채 남겼다 (dormant fix).
- `feet_air_time`은 발이 0.5초 이상 공중에 떠야 보상 → 제대로 된 swing/stepping을 유도하는 항.

**종합**: 평지에서 A의 reward 지형 = **"다리 움직이지 마라(-0.4 dof_error)" + "다리 들어도 보상 없음(air_time off)"**. 정책이 도달하는 최적해는 *다리를 최소로 움직이며 몸을 앞으로 끌기(drag-forward shuffle)* — 관측된 어색한 보행과 정확히 일치한다. B는 (a)가 없으므로 `tracking_goal_vel +1.5`와 자연 동역학이 trot을 만든다.

> **권장(검증 동반)**: ① `feet_air_time` scale을 표준값(예: anymal 계열 0.25~1.0 수준)으로 활성화하거나, ② 평지 `dof_error` 증폭 계수를 ×10 → ×1~3으로 완화. 둘 중 하나를 **단독으로** 적용해 원인 분리. 두 변경을 동시에 하지 말 것.

### 🟠 2순위 — 액션 평활화 항 과다 (A 3개 vs B 1개)

| | A | B |
|---|---|---|
| action_rate | -0.05 | -0.1 |
| action_smoothness_1 | **-0.01** | 없음 |
| action_smoothness_2 | **-0.01** | 없음 |
| delta_torques | -1e-7 | -1e-7 |

A는 액션 평활화 항을 3개(action_rate + smoothness 1·2차) 쌓았다. action_rate 자체는 B보다 가볍지만(-0.05 vs -0.1), 1·2차 smoothness가 추가로 액션의 시간 변화를 억제 → 다리 swing이 crisp하지 못하고 뭉개질 수 있다. 1순위만큼 결정적이진 않으나 어색함을 가중하는 보조 요인.

### 🟡 3순위 — `termination` 페널티 -100

A는 종료 시 `-100` 페널티. B는 별도 termination 페널티 없이 `collision -10`으로만 접촉을 다룬다. -100은 정책을 **과도하게 보수적**으로 만들 수 있고, 1순위의 "다리 움직이지 마라"와 합쳐지면 "안 넘어지게 거의 안 움직인다"는 freeze/shuffle 해를 강화한다. 단독 원인은 아니나 1순위와 같은 방향으로 작용.

### ⚪ 사실로만 기재 — root cause / lever 아님

- **Actuator gain 차이**: A는 K=25 / D=0.5 / saturation 23.5 (`_actuator_mode=2`), B는 K=40 / D=1.0. **단일 raw config로는 가장 큰 물리적 차이**이며 stiffness가 낮으면 위치 추종이 느슨해 보행이 무를 수 있다. 그러나 사용자가 *이 actuator 설정(mode 2)으로 이전에 학습 성공을 검증*했으므로 본 분석의 lever가 아니다 — 사실로만 기재하고 변경 제안하지 않음.
- **Observation — foot contact**: B는 policy obs에 `contact_fill(4)`(발 접촉 상태)을 노출해 gait phase 타이밍에 유리. A는 sim-to-real 제약상 contact obs를 추가할 수 없음(deploy 시 Go2에 신뢰 가능한 contact 센서 없음). → A가 바꿀 수 없는 항목, 맥락 정보로만 기재.
- **Command resampling**: B는 6초마다 재샘플, A는 에피소드 시작·goal 도달 시에만. 영향 경미.
- **Flat 노출 비율**: A 10% vs B 5% — A가 오히려 평지를 더 본다 → A의 결손 요인 아님.
- action latency / height_scanner: 본 분석에서 원인으로 거론하지 않음(별도 검증 완료 영역).

---

## 종합 결론

1. **Q1 — Manager vs Direct 프레임워크 차이는 원인이 아니다.** A·B의 reward·action·control 설정이 거의 동일하다는 사실이 이를 증명한다. 프레임워크는 단지 Direct(A) 쪽에서 config divergence가 누적·은폐되기 쉬운 토양을 제공했을 뿐이다.

2. **Q2 — 1순위 원인은 A의 평지 전용 reward 후처리 2개의 상호작용**:
   - `dof_error_l2`의 평지 ×10 증폭(effective -0.4) → 다리 움직임에 음의 gradient
   - 이를 상쇄할 `feet_air_time`이 구현됐으나 `scale=0.0`으로 비활성
   - → drag-forward shuffle. git 커밋 `e81b71c8`("fix drag-forward lock-in")이 증상을 직접 뒷받침.
   - 보조 요인: 액션 평활화 항 과다(2순위), termination -100(3순위).

3. **다음 액션(검증 동반, 단독 적용)**: `feet_air_time` 활성화 **또는** 평지 `dof_error` 증폭 완화 중 하나를 단독으로 적용해 평지 보행을 재학습하고 효과를 분리 측정.

### 부록 — 코드 검증 (직접 확인)

| 주장 | 위치 | 확인 |
|---|---|---|
| `dof_error_l2` 평지 ×10 증폭 | `parkour_env.py:1009-1011` | ✓ `* (10.0 * is_flat + is_non_flat * 0)` |
| `feet_air_time` 비활성 | `parkour_env_cfg.py:565` | ✓ `"feet_air_time": 0.0` |
| drag-forward 인지 + counter-fix 구현 | git `e81b71c8` | ✓ 커밋 메시지에 명시 |
| A는 Genesis 포팅 | `parkour_env.py` reward 주석 다수 | ✓ "Genesis original" / "Genesis line ..." |
| B는 extreme_parkour 포팅 | `extreme_parkour_terrians.py:13` | ✓ "Reference from arxiv 2309.14341" |

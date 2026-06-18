# HindLeg 발 궤적 추종 보상 설계 — Research + 구현 방법

> 작성 2026-06-09. 대상 env: `HindLeg-Direct-v0` (active cfg `HindLegHistoryEnvCfg`, 2족 8-DOF).
> 선행 기록: `hind_leg_foot_trajectory_reward_design.md`(v2/v3), 메모리 `project_hind_leg_env_identity`.
> 본 문서는 **research + 구현 방법(plan)** 이다. 코드는 수정하지 않았다.

---

## 0. TL;DR

- **측정 확정**: v3(model_2900)은 **16/16 env에서 한 다리(HR) duty≈0.00로 영구 거상, 다른 다리(HL) duty 0.60으로 셔플** 하는 한-다리 보행으로 수렴. 사용자 관찰 정확.
- **Root cause는 weight가 아니라 reward의 class**: `foot_height`가 phase-free·per-foot라 "발 하나를 계속 들고 있기"로 max된다. v1→v2→v3 세 번의 tuning이 모두 실패한 이유 = 같은 gameable class를 계속 튜닝했기 때문. **네 번째 band-aid 금지.**
- **해결의 척추 = phase-periodic contact schedule (Siekmann/Cassie)**, bezier/raibert가 아니다. clock이 각 발의 stance/swing 위상을 지정하고, **stance 위상에서 접촉을, swing 위상에서 clearance를 보상** → 발을 영구히 들면 매 cycle stance 위상을 위반 → 구조적으로 음수. 이것이 현재 보상에 없는 성질.
- bezier/cycloid와 raibert는 schedule 위에 얹는 **직교 레이어**(arc 모양 / 착지 위치)이지 anti-gaming 핵심이 아니다.
- **불변식 2개**: (1) phase clock을 **obs에 넣어야** 정책이 reference에 동기화 가능(없으면 추종 보상이 noise). (2) 두 다리 **anti-phase(0.5 offset) 강제**. 둘 다 obs 변경 → from-scratch 재학습(사용자 동의함).
- **단 하나의 합격 판정 테스트**: *"정지 상태로 한 발을 든 자세가 교대 보행보다 반드시 더 낮은 점수를 받아야 한다."* 어떤 후보 보상이든 이 테스트를 **수식으로** 통과하는지 확인.

---

## 1. Kinematic 검증 결과 (model_2900, cmd_x=1.0, 16 env × 400 step)

측정 스크립트 `/tmp/hindleg_foot_traj_v2.py`, 데이터 `/tmp/hindleg_traj_v3_2900.npz`. sole = `*_foot_contact_link`.

| foot | duty (접촉 비율) | lift mean | lift p95 / max | 해석 |
|------|------|-----------|----------------|------|
| HL (foot0) | **0.599** | 0.56 cm | 1.68 / 2.34 cm | 지지발. 거의 안 듦 → 셔플/드래그 |
| HR (foot1) | **0.003** | 2.75 cm | 7.83 / **15.9 cm** | **영구 거상**(거의 착지 안 함) |

- **16/16 env 전부** HR duty < 0.01 → 개별 env 우연이 아닌 정책의 결정적 수렴 모드.
- (정직성 메모: never-landing 발은 5th-pct ground proxy 자체가 들려 있어 HR lift는 **과소** 추정. 결론을 강화할 뿐.)
- 교차검증: TF 로그 `Episode_Reward/foot_height=+1.04`(track_lin_vel +0.83 다음 2위 양항), `air_time_cap=-0.0002`(거의 0).

### 1.1 reward 코드 레벨 root cause (`hind_leg_env.py`)

```python
# L266-271  foot_height (phase-free, per-foot)
in_swing = ~contact_filt                       # 든 발 = 항상 swing
lift = sole_z - self._sole_rest_z
height_rew = Σ_feet in_swing * clamp(lift / offset, 0, 1)   # 든 발 = 매 step +1.0

# L284-285  air_time_cap (의도: one-foot farming 방지 — 그러나 실패)
air_excess = clamp(last_air_time - cap, min=0) # last_air_time = "마지막 *완료된* 공중 구간"
air_time_cap_pen = Σ_feet air_excess           # 착지를 안 하면 갱신 안 됨 → 0
```

- `foot_height`: 든 발은 영구히 `in_swing=True`, `lift≥offset` → 매 step 발당 +1.0을 farming. → `Episode_Reward/foot_height=+1.04`와 정확히 일치.
- `air_time_cap`: `last_air_time`은 **착지 순간에만** 갱신되는 "직전 완료 공중구간" 길이. HR이 영원히 안 내려오면 갱신이 안 돼 `air_excess≈0` → penalty 회피. → `-0.0002`와 일치. **의도는 맞으나 키(key)가 틀렸다.**

**결론**: 한-다리 거상은 reward-positive(+1.04) + penalty-free 평형. phase-free per-foot height 보상의 교과서적 reward-hacking.

---

## 2. 왜 trajectory "추종"이 아니라 contact "schedule"이 핵심인가

"bezier 만들어서 따라가게 보상" 만으로는 또 다른 gameable/untrackable 보상이 되기 쉽다. **gaming을 by-construction으로 불가능하게** 만드는 것은 곡선이 아니라 **위상 기반 접촉 스케줄**이다.

사용자가 언급한 3가지는 서로 **직교하는 레이어**다. 섞으면 설계가 흐려진다:

| 레이어 | 답하는 질문 | 역할 | anti-gaming? |
|--------|------------|------|--------------|
| **A. Phase clock + contact schedule** | 각 발이 *언제* 땅에 있어야/떠야 하나 (two-leg anti-phase) | **척추(structural fix)** | **예 — 핵심** |
| **B. Bezier/cycloid** | swing *arc*의 모양 (얼마나 높이, 어떤 곡선) | shaping | 아니오 |
| **C. Raibert heuristic** | swing 발이 *어디에* 착지하나 (속도 적응) | placement | 아니오 |

- **A (Siekmann et al. 2021, "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition", Cassie)**: 위상 φ∈[0,1)와 다리별 offset으로 각 발을 stance/swing으로 라벨. **stance 위상엔 접촉(또는 발속도 0)을, swing 위상엔 무접촉/clearance를 보상.** 발을 영구히 들면 그 발의 stance 위상을 매 cycle 위반 → strictly negative. 이게 현재 보상에 **없는** 성질이고, 이것만으로 한-다리 모드가 죽는다.
- **B**: A가 swing을 "떠 있어라"로만 강제하면 발끝 궤적 모양이 거칠 수 있다. 0→H→0의 매끄러운 z 곡선(cycloid/bezier)을 swing 위상 동안의 reference로 추가해 자연스러운 호를 만든다. **A 없이 B만 쓰면 again gameable**(어디서 떠있든 곡선 한 점에 맞출 수 있음).
- **C**: base-frame 고정 bezier는 명령 속도가 바뀌면 착지점이 안 맞아 미끄러진다. Raibert: `x_foot* = x_hip + v·T_stance/2 + k(v − v_cmd)`. swing의 **착지 x 목표**를 속도에 적응시켜 stance 슬립을 줄인다. 가변속 보행에서만 필요.

**권고 순서: A → (수렴 확인) → B → C.** 사용자가 곡선을 원했지만, research 단계의 결론은 **"곡선 자체는 gaming을 못 막는다 — 스케줄이 막는다"**.

---

## 3. 합격 판정 테스트 (모든 후보 보상이 통과해야 함)

> **정지 상태 한-발-거상 자세의 보상 < 교대 보행의 보상** — strictly.

현재 v3가 이걸 위반한다(한-발-거상 = +1.04 farming). 새 보상은 **수식으로** 이걸 만족해야 한다:

- contact-schedule 항에서, 한 발을 영구히 들면 그 발의 stance 위상 구간(φ가 stance일 때) 동안 `I_stance(φ)·(contact reward)`를 매 cycle 0으로 비우고 추가로 swing을 강제받는 다른 발과 동시에 양발이 동시 swing이 되는 순간 base가 무너져 termination(-100) penalty. 즉 한-발-거상은 (a) stance 보상 상실 + (b) 균형 상실로 이중 음수.
- 직관이 아니라 **위상 적분으로** 한 cycle 점수를 손으로 계산해 부등호를 확인할 것.

---

## 4. 구현 방법 (staged plan — 코드 아님)

현재 인프라 사실(확인 완료):
- **Action**: `_processed_actions = cfg.action_scale(0.25) * actions + default_joint_pos` (L116-121) → **residual position control**. PMTG에 이상적(nominal trajectory를 default_joint_pos 대신/위에 더하면 됨).
- **Obs (History cfg, dim 30)**: `projected_gravity(3) + commands(3) + (q−q_def)(8) + q̇(8) + actions(8)`. clock 미포함.
- **`clock_inputs` 플래그**: cfg(L100/108/254/262)에 dim 자리만 있고 **env에 실제 구현 없음**(phase state·sin/cos·obs 주입 전무). → 새로 구현해야 함.
- 활성 cfg `HindLegHistoryEnvCfg`: `history_len=10`, foot ids = `_sole_body_ids`(`*_foot_contact_link`), `_feet_ids`(contact sensor).

### Stage A — Phase clock + contact schedule (최소 구조 수정, 필수)

**A-1. Phase state (env)**
- `self._gait_phase` buffer `(num_envs,)` ∈ [0,1) 추가. `__init__`에 0 초기화, **`_reset_idx`에서 0 초기화(필수)**.
- 매 step `_gait_phase = (_gait_phase + step_dt / gait_period) % 1.0` (cfg `gait_period`≈0.5–0.7s, 초기엔 고정).
- 다리별 위상: `phase_HL = _gait_phase`, `phase_HR = (_gait_phase + 0.5) % 1.0` (**anti-phase 강제**).

**A-2. Obs 주입 (불변식 1)**
- 각 다리 위상을 `[sin(2π φ_leg), cos(2π φ_leg)]`로 obs에 cat → **+4 dim**(2 legs × 2). `clock_inputs=True` 경로를 이 4-dim으로 실제 구현하고 `num_prio_obs`를 +4.
- obs 차원 변경 → `observation_space` 동기화 + **from-scratch 재학습**.

**A-3. Contact-schedule 보상 (Siekmann 형)**
- swing/stance 지시 계수(부드러운 사각파): `I_swing(φ_leg)`, `I_stance(φ_leg)`, 합=1. von Mises 또는 `0.5(1+cos)` smoothing.
- 두 항:
  - `r_stance = Σ_feet I_stance(φ_leg) · contact_indicator` (땅에 있어야 할 때 접촉 보상; 또는 발속도 0 보상)
  - `r_swing  = Σ_feet I_swing(φ_leg) · (1 − contact_indicator) · clamp(lift/H, 0, 1)` (뜰 때 clearance 보상)
- **`foot_height`(phase-free)·`air_time_cap`은 제거** — `r_swing`이 대체하고, schedule이 air_time_cap의 목적을 구조적으로 달성.
- `foot_slip`(contact-gated `‖v_xy‖²`)은 **유지**(stance 슬립 억제, 직교).

**A-4. 합격 테스트 통과 확인** (§3) 후 재학습 → 재측정(`/tmp/hindleg_foot_traj_v2.py`). 기대: 양발 duty≈0.5–0.6, HR duty가 더 이상 0이 아님.

### Stage B — Swing arc 곡선 (A 수렴 후)

- swing 위상 동안 reference z: cycloid `z_ref(φ_sw) = H · (1 − cos(2π φ_sw))/2` 또는 3차 bezier(0→H→0).
- 보상: `r_arc = Σ I_swing · exp(−(sole_z − z_ref)² / σ²)` 또는 `clamp` 기반. clearance를 모양까지 가이드.
- 선택: base-frame x도 reference에 포함하면 전후 stride 모양까지 규정.

### Stage C — Raibert 착지 + PMTG (가변속에서만)

- swing 발 착지 x 목표: `x_foot* = x_hip + v·T_stance/2 + k(v − v_cmd)`, base-frame. 보상 또는 IK로 nominal에 반영.
- **PMTG (Iscen et al. 2018)**: clock-driven nominal joint trajectory `q_TG(φ)`를 만들고 `_processed_actions = action_scale·a + q_TG(φ)`로 교체(현재 `default_joint_pos` 자리). 정책은 residual만 학습 → swing lift가 **구조적으로 보장**(보상이 약해도 발이 물리적으로 든다). action이 이미 residual이라 **최소 변경으로 가능**.

### cfg에 추가될 파라미터 (예시, 값은 튜닝)
`gait_period=0.6`, `swing_height_H=0.07`, `phase_offset_HR=0.5`, `clock_inputs=True`, schedule smoothing κ, `r_stance/r_swing/r_arc` scale. **한 번에 1–2개만 조정**(메모리 규칙).

---

## 5. 불변식 / 금지

- **불변식 1**: phase clock을 obs에 넣는다(없으면 추종 보상 = noise). clock(time-function)→obs는 sim-to-real 허용(메모리 `project_hind_leg_env_identity`).
- **불변식 2**: 두 다리 anti-phase(0.5) 강제. 안 하면 both-up/both-down 재발.
- **금지**: contact sensor 자체를 obs에 추가 금지(sim-to-real firewall). contact은 **보상 계산에만** 사용. phase clock과 `contact_indicator(보상용)`는 별개.
- **금지**: phase-free per-foot height 류의 band-aid 4차 시도.
- 새 buffer(`_gait_phase` 등)는 반드시 `_reset_idx` 초기화.

---

## 6. 참고문헌

- Siekmann et al. 2021, *Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition* (Cassie) — contact-schedule periodic reward, 본 설계의 척추.
- Iscen et al. 2018, *Policies Modulating Trajectory Generators (PMTG)* — residual + TG 구조(Stage C).
- Raibert 1986, *Legged Robots That Balance* — 착지 heuristic(Stage C).
- 곡선: cycloid/bezier swing(MIT Cheetah, ANYmal 계열) — Stage B.

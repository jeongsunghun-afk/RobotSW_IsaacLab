# hind_leg foot-dragging 진단 + reward 제안서

작성일: 2026-06-08 · 환경: `HindLeg-Direct-v0` (활성 cfg `HindLegHistoryEnvCfg`)

> 요약: hind_leg은 parkour Go2와 **별개인 2족(biped) 8-DOF 로봇** 환경이다. 발을 끄는 이유는
> 버그가 아니라 **"발을 들어올리는 것에 대한 직접적 양의 보상이 0개"이고, swing을 처벌하는 항이
> 더 강해서 생긴 reward-positive drag 평형**이다 (TF 로그로 확정). 해법은
> **velocity-gated foot-clearance 보상(IsaacLab Spot 정형)을 추가**하는 것이 1순위다.

---

## 1. 환경 식별

| 역할 | 경로 |
|---|---|
| env 로직 | `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env.py` (`HindLegEnv`) |
| cfg (활성) | `.../hind_leg/hind_leg_env_cfg.py` → **`HindLegHistoryEnvCfg`** |
| agent/runner | `.../hind_leg/agents/rsl_rl_ppo_cfg.py` → `HindLegParkourPPORunnerCfg` |
| robot asset | `source/isaaclab_assets/isaaclab_assets/robots/rga.py:436` (`HIND_LEG_CFG`) |

**parkour와의 관계: 별개 환경.** `hind_leg_env.py`는 `direct/parkour/`를 import하지 않고
자체 `_get_rewards`/`_get_observations`를 가진 standalone flat-locomotion env다. 러너 cfg에서
`OnPolicyRunnerParkour`/`ActorCriticRMA`/`PPOParkour`(`rsl_rl_ppo_cfg.py:47,64,74`)라는
**알고리즘(RMA/estimator) 골격만** parkour에서 차용했을 뿐, 보상·관측·물리·로봇은 전부 로컬이다.
로봇은 Go2가 아니라 **2족 로봇**(`action_space=8` = 2 leg × 4 joint: hip/thigh/calf/foot,
`rga.py:469-473`). → 메모리의 "Go2 Parkour 왼쪽 뒷다리 3-leg" 문제와 무관하다. parkour 전용
제약(total_reward clip, actuator_mode=2 등)은 이 환경에 적용되지 않는다. 단 sim-to-real 원칙
(**contact sensor를 observation에 넣지 않음**)은 그대로 유지한다.

---

## 2. 현재 reward 구조 (활성 cfg)

reward dict: `hind_leg_env.py:234-248` · scale: `hind_leg_env_cfg.py:309-321` · 모든 항 `×step_dt`.

| 항목 | 수식 | scale | 부호 |
|---|---|---|---|
| track_lin_vel_xy_exp | `exp(-‖cmd_xy-v_xy‖²/0.1)` | +1.0 | 보상(지배적) |
| track_ang_vel_z_exp | `exp(-(cmd_yaw-ω_z)²/0.1)` | +0.5 | 보상 |
| feet_air_time | `Σ(last_air_time-0.2)·first_contact·𝟙(‖cmd_xy‖>0.1)` | +0.5 | **들기 유일 신호(간접)** |
| lin_vel_z_l2 | `v_z²` | -2.0 | penalty |
| ang_vel_xy_l2 | `‖ω_xy‖²` | -0.01 | penalty |
| dof_torques_l2 | `Στ²` | -2e-4 | penalty |
| dof_acc_l2 | `Σq̈²` | -2.5e-7 | **swing 억제** |
| action_rate_l2 | `Σ(aₜ-aₜ₋₁)²` | -1e-3 | swing 억제 |
| undesired_contacts | hip/thigh/calf 접촉 수 | -1.0 | penalty |
| flat_orientation_l2 | `Σg_proj_xy²` | **-0.0(비활성)** | — |
| similar_to_default | `Σ\|q-q_default\|` | -0.1 | **swing 최대 억제** |
| base_height | `(z_base-0.6)²` | -10.0 | penalty |
| termination | base contact | -100.0 | penalty |

- total_reward에 **clip 없음** (parkour와 다름, 정상 설계).
- **height_scan 미사용** (History cfg는 RayCaster 미생성, Rough cfg에만 존재).
- contact sensor는 termination/undesired/air_time **계산에만** 사용, obs에는 없음(유지 권장).
- **clearance/foot-height/swing-height/no-fly/gait-schedule 항: 13개 중 0개.**

---

## 3. 근본 원인 (TF 로그로 확정 — 표면 추론 아님)

증거: `logs/rsl_rl/hindLeg_history_direct/2026-06-08_13-20-32_airtime_thr0.2_phase1`, step 8364.

| Episode_Reward 항 | 값 | 해석 |
|---|---|---|
| feet_air_time | **-0.046** | 음수! 평균 touchdown air_time < 0.2s, 학습 내내 악화 |
| similar_to_default | **-0.181** | 전 항목 중 최대 penalty, 단조 악화(-0.001→-0.181) |
| dof_acc_l2 | -0.088 | swing 가속 억제 |
| track_lin_vel_xy_exp | **+0.577** | 지배적 보상 — 끌어도 전진하면 받음 |
| (total) mean_reward | +5.85 | 정책은 안정 수렴 (episode_length 797) |

**메커니즘:**

1. **`feet_air_time`가 보상이 아니라 페널티로 작동.** `(last_air_time-0.2)`는 발이 0.2s 미만으로
   떴다 닿으면 음수. 끄는 gait(짧은 air_time)는 이 항이 음수이며, threshold 0.2s를 넘기려면 큰
   swing 투자가 선행돼야 하는데 그 투자가 아래 2번에서 더 크게 처벌됨 → **로컬 미니멈에 갇힘.**
2. **swing을 직접 처벌하는 항이 더 강함.** `similar_to_default`(`Σ|q-q_default|`, -0.1)는 관절을
   기본자세에서 멀리 움직일수록 선형 페널티 → clearing swing(calf/foot 큰 excursion)을 직접 억제.
   로그상 최대 음수 항. `dof_acc_l2`·`action_rate_l2`도 같은 방향.
3. **들어올림을 직접 보상하는 항이 0개.** 양의 lift 신호가 air_time(간접·threshold형) 하나뿐이고
   그조차 1·2 때문에 음수에 갇힘. 결과적으로 "전진 만족 + 관절 최소 이동(=끌기)"이 **net 양의 보상
   최적해**가 된다.
4. **자세 유지 압력 약함.** `flat_orientation_l2` = -0.0(비활성)이라 직립 동적 swing 유도 보조항 부재.

**판정: dead-gradient가 아니라 reward-positive drag 평형.** 정책이 실패한 게 아니라 *잘못 설계된
보상의 최적해(끌기)로 정상 수렴*했다. 핵심 결손 = **발의 수직 변위(높이)를 직접 보상하는 항의 부재.**

---

## 4. 해법: reward 제안

리서치(legged_gym / IsaacLab Spot/ANYmal / Walk-these-ways / humanoid RL 논문)에서 후보 5종을 비교한
결과, biped hind_leg의 "height 신호 부재 + drag 평형"에 가장 정확히 맞는 것은 아래다.

### ⭐ 제안 1 (1순위, 단독 우선 적용) — Velocity-gated Foot Clearance

IsaacLab-native(Spot) 정형. **드래그의 정의와 gating이 정렬**되는 유일 후보.

```
foot_z      = body_pos_w[:, foot_body_ids, 2]          # (N, n_feet) 발 링크 월드 높이
foot_v_xy   = ‖body_lin_vel_w[:, foot_body_ids, :2]‖   # (N, n_feet) 발 월드 수평속도
z_err       = (foot_z - target_height)**2
clearance_p = Σ_feet  z_err · tanh(tanh_mult · foot_v_xy)   # (N,)  penalty (scale < 0)
```

$$ r_{clear} = -\sum_{f}(z_f - z^{*})^2 \cdot \tanh\!\big(\kappa\,\|\dot p_{xy,f}\|\big) $$

**왜 이게 정답인가:** 끌리는 발 = *낮은 z(큰 z-error) × 큰 수평속도(tanh→1)* → **곱이 최대 = 페널티
최대**. 반대로 planted stance 발은 월드 수평속도≈0 → tanh→0 → 페널티 없음(접지 자체는 처벌 안 함).
즉 정확히 "끌 때만" 페널티가 켜지고, air_time처럼 height를 무시하지 않으며(micro-tap 함정 차단),
gait-schedule처럼 stance-drag를 통과시키지도 않는다(stance/swing 식별 불필요).

**시작 파라미터 (Spot `flat_env_cfg.py:209` 기준 → biped 조정):**
- `target_height`: **절대값(0.05–0.08)을 그대로 쓰지 말 것.** Spot의 0.1은 Spot 기하학 값이다.
  대신 **측정된 rest 높이 상대값**으로 정의: `target_height = (default state에서 foot link world-z)
  + 0.05~0.08`. base_height target이 0.6이므로 foot rest-z는 작을 것(~0.02–0.05)이다 →
  구현 시 default joint pos로 한 번 forward-kinematics를 찍어 foot rest-z를 **측정**한 뒤 offset을
  더하라(추측 금지). link origin이 발바닥이 아니라 ankle이면 그 offset도 target에 반영.
- `tanh_mult = 2.0`, `weight ≈ -0.5 ~ -1.0` (exp 래핑 없이 **합산형 penalty** 권장; Go2/biped에서
  exp는 큰 오차 구간에서 saturate되어 gradient 둔화)
- 적용 대상: 2족이므로 **양 발(foot link 2개) per-foot 합산**. per-foot로 적용하면 한쪽 발이 더 끌어도
  그 발에 비례해 페널티가 커져 **자동 교정**된다(별도 비대칭 가중 불필요). ※ 본 문제가 "대칭 평형"인지
  "한쪽 우세"인지는 per-foot 로그 없이 단정하지 않는다 — per-foot 구조면 어느 쪽이든 커버된다.
- **foot body 식별 주의 (구현 블로커):** position용 body는 contact-sensor body(`.*foot_contact.*`,
  `hind_leg_env.py:73`)와 **다를 수 있다.** `body_pos_w`/`body_lin_vel_w`는 articulation rigid-body
  인덱스를 쓰므로 `self._robot.find_bodies(<실제 foot link명, 예: ".*_foot">)`로 별도 확인하고,
  **정확히 2개(2족)가 매칭되는지 검증**하라.
- 신호는 **reward 계산에만** 사용(`body_pos_w`/`body_lin_vel_w`) → obs 미변경 → sim-to-real 무관.
- **History cfg는 height_scan 미사용**이므로 평지 기준 `foot_z`(월드 z) 직접 사용. Rough cfg로 확장
  시에만 발 밑 ray z를 빼 terrain-relative로 전환.

> 구현 위치: `hind_leg_env.py` `__init__`에서 `self._foot_body_ids, _ = self._robot.find_bodies(<foot link 패턴>)`
> 추가(+rest-z 측정), `_get_rewards()`(234-248)에 항 추가, cfg(309-321)에 `foot_clearance_reward_scale` 추가.
> 동일 reward dict를 공유하는 `HindLegFlatEnvCfg`(88-224)·`HindLegRoughEnvCfg`(350)에도 일관 반영.

### 제안 2 (상보적, 비접촉 활공 escape 봉쇄) — Foot Slip penalty 강화형

제안 1만으로 정책이 "발을 살짝 띄워(F_z<1N) 낮게 활공"하는 escape를 쓸 수 있으므로, **접촉 중 수평
이동**을 직접 처벌하는 항을 함께 둔다(IsaacLab `feet_slide` 계열).

```
contact_filt = (F_z > 1.0) ∨ last_contact
slip_p       = Σ_feet  contact_filt · ‖foot_v_xy‖²    # 제곱형, penalty
```
- `weight ≈ -0.1 ~ -0.25`. 제곱형이 큰 slip을 강하게 누름.
- 역할 분담: **제안 1 = 비접촉 저공활공 차단(height), 제안 2 = 접촉 끌림 차단(slip).** 두 escape를 양면
  봉쇄. slip은 끌수록 단조 증가 → gradient 항상 생존.
- hind_leg엔 이 항이 아직 없음(net-new). 단독보다 제안 1과 묶을 때 가치.

### 제안 3 (제안 1·2 적용 후에도 끌기 지속 시) — swing 억제항 완화

진단상 `similar_to_default`(-0.1)가 swing을 직접 처벌하는 최대 음수 항이다. clearance 보상과
정면 충돌하므로, 제안 1·2 적용 후에도 효과가 약하면 `hind_leg_env_cfg.py:319`의
`similar_to_default_reward_scale`를 **-0.1 → -0.05**로 완화한다. (단독 변경 금지 — 한 번에 한 가설.)
선택적으로 `flat_orientation_l2`(-0.0, `cfg:318`)를 소폭 활성(-0.5~-1.0)해 직립 동적 swing 유도.

---

## 5. 후보 비교 요약 (왜 다른 것이 아닌가)

| 후보 | gating | drag 함정 회피 | height 신호 | 비고 |
|---|---|:---:|:---:|---|
| **2A velocity-gated clearance** ⭐ | foot xy-velocity | ✅ drag=최대penalty | **있음** | **1순위.** IsaacLab-native |
| feet_air_time (현존) | first-contact | ❌ micro-tap | 없음 | 현재 음수 정박의 원흉 |
| 2B schedule-gated clearance (WTW) | swing schedule | ❌ stance-drag 통과 | 있음 | gait clock 필요, 함정 자체 |
| 3 contact-schedule shaped (Raibert) | gait clock | 부분 | 없음 | over-engineering |
| 4 foot slip | stance contact | ❌ 비접촉 활공 통과 | 없음 | **2A와 묶어 보완(제안 2)** |
| 5 bipedal max-clearance (phase-free) | swing/vel | ✅(vel형) | 있음 | per-foot, 대안 |

**핵심:** air_time / schedule-gated clearance / stance-only slip은 모두 "끌어도 보상 만족" 함정을
가진다. **velocity-gated + height-aware**인 2A가 유일하게 height·micro-tap·stance-drag 세 함정을
동시에 회피한다. 이족(humanoid) 순수 RL에서 swing clearance를 만드는 정형(후보 5)도 per-leg 구조라
2족 hind_leg에 직접 전이되며, 그중 **velocity-gated 변형이 2A와 본질적으로 동일**하다.

---

## 6. 권장 실행 순서

1. **제안 1 단독 적용** → 200~500 iter 학습 → `Episode_Reward/foot_clearance`가 음수에서 0 근처로
   수렴하고 영상에서 발이 들리는지 확인. (한 번에 한 가설 원칙)
   - ⚠️ **false-negative 주의:** 진단상 `similar_to_default`(-0.181)가 swing을 억제하는 *최대 음수
     항*으로, clearance와 정면 충돌한다. 제안 1에서 **부분 개선만** 나오면 "실패"로 판단해 폐기하지
     말고, 곧바로 **제안 1+3을 한 묶음으로 재시도**하라(지배적 반대항이 이미 식별된 경우의 예외).
     완전 무반응일 때만 제안 1 자체를 재검토.
2. 비접촉 활공(F_z<1N 저공 활주)이 보이면 **제안 2 추가**.
3. 위에서 묶지 않았다면, 그래도 부족할 때 **제안 3**(similar_to_default 완화 / flat_orientation 활성).
4. air_time threshold 0.2s는 biped 보폭에 비해 클 수 있음 — 1~3 후에도 air_time 음수 정박이면 재검토
   (run 이름 `airtime_thr0.2_phase1`로 보아 이미 튜닝 중).

## 출처 (리서치 검증)
- IsaacLab Spot `foot_clearance_reward`: `.../manager_based/locomotion/velocity/config/spot/mdp/rewards.py:182`,
  실측 cfg `.../spot/flat_env_cfg.py:209`
- IsaacLab `feet_air_time`/`feet_slide`: `.../velocity/mdp/rewards.py:27,71`
- legged_gym `_reward_feet_air_time`: https://github.com/leggedrobotics/legged_gym
- ANYmal clearance 원형 (Lee et al. 2020): https://arxiv.org/pdf/2010.11251
- Walk-these-ways (Margolis & Agrawal 2024): https://github.com/Improbable-AI/walk-these-ways ,
  https://journals.sagepub.com/doi/10.1177/02783649231224053
- Humanoid 순수-RL swing clearance: https://arxiv.org/html/2404.19173v1 , https://arxiv.org/pdf/2506.08416

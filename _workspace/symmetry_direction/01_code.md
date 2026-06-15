# 01 — Go2-Parkour-Symmetry: 8개 런 코드 변경 타임라인

> 작성: worker-1 (Task #1) · 분석 대상: `logs/rsl_rl/go2_parkour_symmetry/` 8개 런의 `git/IsaacLab.diff` 코드 스냅샷
> 방법: 각 런의 `git/IsaacLab.diff`는 학습 시점 작업트리의 `git diff HEAD`. 관련 파일(`parkour_env_cfg.py`, `rsl_rl_ppo_cfg.py`, `symmetry.py`, `ppo_parkour.py`, `actor_critic_parkour.py`)의 hunk를 런 간 비교해 재구성. **코드 수정 없음, 분석만.**

---

## 0. 한눈에 보는 변경 타임라인

| # | 런 (시각) | iter | 의도(런 이름) | **이번 런에서 바뀐 것** | 비교 기준 |
|---|-----------|------|--------------|------------------------|----------|
| 1 | 2026-06-09_17-53-09 | 30000 | baseline | (기준점) 쉬운 지형 · torques_l2=-1e-5 · air_time_cap=-0.1 · contact_duty_deficit=-0.5 · positive_work=0 | — |
| 2 | 06-10_17-00-32_gap_step_more_difficult | 21700 | 지형 난이도↑ | **terrain**: step_height (0.10,**0.45→0.6**), gap_length (0.05,**0.6→0.8**) | vs #1 |
| 3 | 06-10_17-38-06_torque_penalty_0.0001 | 14000 | 토크 패널티↑ | **torques_l2 -1e-5 → -1e-4** (10×) | vs #2 |
| 4 | 06-11_10-08-37_no_duty_time_cap | 5000 | duty/air cap OFF | **air_time_cap -0.1 → 0** , **contact_duty_deficit -0.5 → 0** | vs #3 |
| 5 | 06-11_12-13-02_positive_work | 49999(완주) | 효율항 도입 | **positive_work 0 → -3e-4** (신규 활성) | vs #4 |
| 6 | 06-11_16-21-24_no_duty_time_cap | 18900 | 효율 OFF 대조군 | **positive_work -3e-4 → 0** (= #5에서 효율만 제거) | vs #5 |
| 7 | 06-11_17-28-23_positive_work_0.01 | 14300(붕괴) | 효율 과대 | **positive_work 0 → -1e-2** | vs #6 |
| 8 | 06-12_09-53-06_positive_work_0.001 | 49999(완주) | 효율 중간 | **positive_work -1e-2 → -1e-3** | vs #7 |

핵심: **변경은 전부 reward weight + terrain config 수준의 1~2 인자 변경.** 알고리즘/네트워크 코드(`symmetry.py`, `ppo_parkour.py`, `actor_critic_parkour.py`)는 8개 런 전부 **동일**(내용 변경 0, 해시·공백만 차이).

---

## 1. 불변 요소 (8개 런 전체 공통)

### 1-1. 알고리즘 = mirror data-augmentation symmetry (run1부터 이미 적용)
- `Go2ParkourSymmetryPPORunnerCfg` (← `Go2ParkourPPORunnerCfg` 상속), `experiment_name="go2_parkour_symmetry"`.
- `algorithm.symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True, use_mirror_loss=False, mirror_loss_coeff=0.0, data_augmentation_func="…parkour.mdp.symmetry:compute_parkour_symmetric_states")`.
- L/R 미러만 적용(전방-only task). num_aug=2로 유효 미니배치 2배, PPO 하이퍼/네트워크/encoder는 base에서 그대로 상속.
- **baseline 런1도 이미 이 symmetry 설정 사용** (run1 diff에 uncommitted로 존재 → run1↔run2 사이에 그 코드가 그냥 commit만 됨, 내용 동일).
- `symmetry.py`, `ppo_parkour.py`, `actor_critic_parkour.py`: 8개 런 전체에서 **내용 diff 없음** → 알고리즘/네트워크 변경 0.

### 1-2. positive_work reward **함수**는 처음부터 코드에 존재 (opt-in, weight로만 on/off)
- 구현 위치: `parkour_env.py:1126-1128`
  ```python
  positive_work = torch.sum(
      torch.clamp(self._robot.data.applied_torque * self._robot.data.joint_vel, min=0.0), dim=1
  )  # (N,), units: W, 항상 >= 0
  ```
- = `Σ_j max(0, τ_j · q̇_j)` (12관절 양의 기계적 일률). weight<0 → penalty. 음의 일(braking)은 clamp(min=0)로 제외.
- `_get_rewards()` 끝: `total_reward = torch.clip(total_reward, min=0.0)` (`parkour_env.py:1272`) — **total reward floor=0** (A/B 공통 의도된 설계).
- → positive_work는 런5에서 처음 **활성화(weight≠0)** 됐을 뿐, 함수 자체는 그 전부터 committed 코드에 있었음. (런5 이전엔 weight=0이라 gradient 0.)
- 참고: `air_time_cap`(env:1225), `contact_duty_deficit`(env:1235) 함수도 동일하게 코드에 상주, weight로만 on/off.

> **`parkour_env.py` / `parkour_terrains.py`는 8개 런 모두 작업트리 diff에 안 잡힘** = 학습 기간 내내 committed 상태로 불변. reward 로직 코드 변경 없음. 변경은 전부 `parkour_env_cfg.py`(weight/terrain 인자)에서 발생.

---

## 2. 런별 상세 (변경 인자 + 정확한 위치)

모든 reward/terrain 변경은 `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py`
(reward_scales dict 약 L614~660, terrain PARKOUR_TERRAINS_CFG L97~111).

### 런 1 — baseline (2026-06-09_17-53-09, 30000 iter)
작업트리 reward_scales (핵심):
```
torques_l2            = -1e-5     # (committed baseline)
feet_dragging         = -0.1      # 뒷발(RL,RR)만
feet_gait_pairing     =  0.0      # 비활성(Spot trot sync, 양수 only)
air_time_cap          = -0.1      # 활성 (과도 체공 graded penalty)
contact_duty_deficit  = -0.5      # 활성 (EMA 접촉 duty 부족 penalty)
positive_work         =  0        # (weight 미설정 → OFF)
```
terrain: `step_height_range=(0.10, 0.45)`, `gap_length_range=(0.05, 0.6)` ← **상대적으로 쉬움**

### 런 2 — gap_step_more_difficult (06-10_17-00-32, 21700)
- **terrain만 변경** (`parkour_terrains.py`가 아니라 `parkour_env_cfg.py` 내 `PARKOUR_TERRAINS_CFG`):
  - `step_height_range=(0.10, 0.45) → (0.10, 0.6)` (계단 최대 +0.15 m)
  - `gap_length_range=(0.05, 0.6) → (0.05, 0.8)` (틈 최대 +0.2 m, 주석상 overflow-fix로 줄였던 걸 원복)
- reward 동일(런1과 같음). **런2 이후 모든 런은 이 hard terrain 유지.**

### 런 3 — torque_penalty_0.0001 (06-10_17-38-06, 14000)
- **torques_l2: -1e-5 → -1e-4** (10배 강화). 런 이름 0.0001 = 1e-4 일치.
- 나머지 reward/terrain = 런2와 동일(air_time_cap=-0.1, contact_duty_deficit=-0.5 유지).
- torques_l2=-1e-4는 **런3~8 내내 유지**.

### 런 4 — no_duty_time_cap (06-11_10-08-37, 5000)
- **air_time_cap: -0.1 → 0** (체공 cap OFF)
- **contact_duty_deficit: -0.5 → 0** (접촉 duty 강제 OFF)
- → 런 이름대로 "duty + time cap 제거". 명시적 접촉/체공 강제 신호를 모두 끔.
- positive_work 아직 0. (5000 iter에서 중단)

### 런 5 — positive_work (06-11_12-13-02, 49999 = **완주**, pw -3e-4)
- **positive_work: 0 → -3e-4** (효율항 첫 활성화). cfg 주석에 calibration 기록:
  - stair 평균 기여 0.0006(2.9%), p90 0.0014(6.7%) → conservative/safe.
- air_time_cap=0, contact_duty_deficit=0 유지(런4 상태). → **명시적 접촉강제 없이 효율항만으로 자연 걸음 유도** 실험의 첫 완주 런.
- 이 시점 작업트리(`parkour_env_cfg.py` blob `0d7261ba1c6`)가 **런6 직전에 commit** 됨 → 런6~8의 HEAD 기준이 됨.

### 런 6 — no_duty_time_cap (06-11_16-21-24, 18900, pw 0)
- 기준이 런5 commit으로 바뀌어 diff엔 **positive_work만** 나타남.
- **positive_work: -3e-4 → 0** (= 런5에서 효율항만 제거).
- 즉 air_time_cap=0 · contact_duty_deficit=0 · positive_work=0 → **gait-shaping penalty 전부 OFF인 대조군**(hard terrain + 기본 parkour reward + torque -1e-4 + symmetry만). 효율항의 순효과를 분리하는 control.

### 런 7 — positive_work_0.01 (06-11_17-28-23, 14300 = **붕괴**)
- **positive_work: 0 → -1e-2** (런5 대비 33배). cfg 주석 경고대로 stair p99에서 total_reward floor(clip min=0) 위험 → 학습 붕괴 확정.

### 런 8 — positive_work_0.001 (06-12_09-53-06, 49999 = **완주**, pw -1e-3)
- **positive_work: -1e-2 → -1e-3** (중간값). 두 번째 완주 런.
- 나머지 동일(air/duty OFF, torques_l2=-1e-4, hard terrain, symmetry).

---

## 3. reward weight 변천 매트릭스 (검증값)

| reward 항 | 런1 | 런2 | 런3 | 런4 | 런5 | 런6 | 런7 | 런8 |
|-----------|----|----|----|----|----|----|----|----|
| torques_l2 | -1e-5 | -1e-5 | **-1e-4** | -1e-4 | -1e-4 | -1e-4 | -1e-4 | -1e-4 |
| air_time_cap | -0.1 | -0.1 | -0.1 | **0** | 0 | 0 | 0 | 0 |
| contact_duty_deficit | -0.5 | -0.5 | -0.5 | **0** | 0 | 0 | 0 | 0 |
| positive_work | 0 | 0 | 0 | 0 | **-3e-4** | **0** | **-1e-2** | **-1e-3** |
| feet_dragging | -0.1 | -0.1 | -0.1 | -0.1 | -0.1 | -0.1 | -0.1 | -0.1 |
| feet_gait_pairing | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| **terrain** step_h max | 0.45 | **0.6** | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 | 0.6 |
| **terrain** gap_len max | 0.6 | **0.8** | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 | 0.8 |

(굵게 = 직전 런 대비 변경. 정합성 검증: cfg blob 해시 8개 런 모두 distinct.)

---

## 4. 효율(positive_work) 캘리브레이션 정리 (cfg 주석 근거)

| weight | stair mean 기여 | stair p90 | stair p99 | 판정 |
|--------|----------------|-----------|-----------|------|
| -3e-4 (런5) | 0.0006 (2.9%) | 0.0014 (6.7%) | — | 보수적/안전 → **완주** |
| -1e-3 (런8) | 0.0020 (9.6%) | 0.0047 (22%) | 0.017 (82%) | 중간(경계) → **완주** |
| -1e-2 (런7) | ~10× 위 | — | total_reward floor(clip min=0) 직격 | **붕괴 확정** |

> 출처: `pronk_cost_result.npz` (256 envs × 3000 steps, dt=0.02 s) 기반 cfg 주석. tracking_goal_vel 1-step 기여 ≈0.021 대비 비율.

---

## 5. 추측 / 한계 표기

- (검증됨) reward/terrain weight 값 — 8개 diff의 `+`/context 라인 직접 비교, cfg blob 해시 distinct 확인.
- (검증됨) algorithm/network 코드 불변 — `symmetry.py`/`ppo_parkour.py`/`actor_critic_parkour.py` 내용 diff 0.
- (검증됨) positive_work 함수 위치/수식 — `parkour_env.py:1126-1128` 실파일 확인.
- (추정) positive_work·air_time_cap·contact_duty_deficit **함수 코드가 정확히 언제 parkour_env.py에 추가됐는지**는 diff로 알 수 없음(해당 파일은 학습 기간 내내 committed·불변). 다만 런1 시점에 이미 cfg에 weight 키가 존재 → 함수도 런1 이전 commit. weight 활성화 시점(=실제 학습 영향 시점)은 위 표대로 검증됨.
- (한계) iteration 수는 Task #1 명세(체크포인트 존재) 기준. 완주/붕괴 판정은 런 이름·iter 수 기반 1차 해석이며, 실제 수렴/붕괴 곡선은 Task #3(로그)에서 확정.

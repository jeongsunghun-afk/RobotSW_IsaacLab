# Parkour Flat-Only Trot 학습 — 변경 권장서

**Date**: 2026-05-13
**Run analyzed**: `logs/rsl_rl/go2_parkour/2026-05-13_09-28-12_flat_terrain` (1865 iter, 15h+)
**Team**: `parkour-flat-trot` (lead = team-lead, workers = flat-diagnostic + reward-validator)
**Input reports**:
- `_workspace/parkour_flat_run_diagnostic.md` (flat-diagnostic)
- `_workspace/parkour_feet_air_time_validation.md` (reward-validator)

**Hard constraints (영구 메모리: `project_parkour_analysis_constraints.md`)**:
- 금지: contact sensor를 **policy obs**에 추가, `_actuator_mode=2` blame, 토크 envelope Tier-2, action latency, height_scan
- 해제 (2026-05-13): 새 reward 추가 (검증 절차 거친 경우)
- Tier discipline: A (log) / B (code) / C (inference). 1순위 가설은 Tier C 단독 금지.

---

## 0. TL;DR

| 변경 | 위치 | 타입 | 효과 | Tier |
|------|------|------|------|------|
| **(1) feet_air_time reward 추가** | `parkour_env.py` `_get_rewards()` + `parkour_env_cfg.py:474` | 신규 reward (scale=0.5) | swing-stance 분리 신호 → trot 발현, drag-forward 안정점 탈출 | **B** (anymal_c/R_Skeleton 표준 패턴 + drag 128× 악화 Tier A 정황) |
| **(2) `dof_error_l2` flat ×10 강화** | `parkour_env.py:1003` — 한 줄 uncomment | Genesis-faithful 활성화 | 평지에서 default pos 유지 압력 → 자세 정상화 | **B** (코드 이미 존재, 주석만 처리됨) |
| **(3) yaw wrap 복구** | `parkour_env.py:629, 631` — 두 줄 uncomment | regression fix | yaw_diff를 [-π, π]로 wrap → reward 신호 정상화 | **A+B** (Tier A: ang_vel 59× 악화 + Tier B: 코드 주석) |

**3 변경 모두 한 batch로 적용 가능** (서로 독립, 동반 위험 없음). obs dim 불변 → 기존 checkpoint resume 가능하나 reward 구조가 변하므로 from-scratch 권장.

---

## 1. 두 worker 발견 종합

### 1.1 flat-diagnostic (TB metrics, Tier A) — 채택 항목

> **참고**: flat-diagnostic의 Hypothesis #2 ("weak actuators K=25, D=0.5, vel_limit=30")는 **프로젝트 메모리 §5 (`_actuator_mode=2` blame 금지) + §6 (토크 envelope Tier-2 금지) 위반**. 이번 보고서 결론에서 제외. 단, 학습 데이터(메트릭) 자체는 valid.

#### 학습 진행 (Tier A, 채택):
- `tracking_goal_vel`: -0.003 → **+1.214** (forward motion 학습 ✓)
- `tracking_yaw`: +0.006 → **+0.380** (yaw 학습 약함)
- Episode length: 13.75 → **820.18 steps** (62× ↑, 종료가 timeout)
- Net episode reward: -0.11 → **+20.03**

→ **정책은 학습 중**. 그러나 trot 보행 발현 X.

#### Drag-forward 안정점 신호 (Tier A, 채택):
- `feet_dragging`: -0.001 → **-0.128** (**128× 악화** — 발 끌기 패널티 누적)
- `dof_acc_l2`: -0.002 → **-0.123** (61× 악화 — 관절 가속도 누적)
- `ang_vel_xy_l2`: -0.002 → **-0.117** (59× 악화 — 좌우/피치 진동)
- `dof_error_l2`: -0.0002 → **-0.047** (default pos에서 멀어짐)
- `action_rate_l2`: -0.001 → -0.050 (action 진동)

→ 정책이 **자세를 무너뜨리고 발을 끌면서 전진하는 안정점**에 락. tracking_goal_vel +1.21 양수가 dragging 페널티 -0.128을 압도해 net positive ⇒ 학습 신호상 stable.

#### Entropy/noise 비정상 (Tier A, 채택):
- `Loss/entropy`: 17.0 → **22.1** (PPO에서 보통 감소, 증가는 비정상)
- `Policy/mean_noise_std`: 0.999 → **1.901** (exploration noise 1.9배, 정책이 확신 못함)

→ 보상 landscape가 부드럽지 못함. **yaw 신호 결함 (Hypothesis #1)의 정황 증거**.

#### Termination 분포 (Tier A, 채택):
- timeout dominant (정상)
- base_contact 가끔 있음
- low_height 없음

→ 빈번한 fall은 root cause 아님. 학습 진행을 막는 건 termination이 아니라 reward landscape의 trot-blind 안정점.

### 1.2 flat-diagnostic Hypothesis #1 (yaw wrap disabled) — 라인 corrected, 채택

flat-diagnostic 주장 라인 (118-141, 150)은 **잘못된 위치** — 그건 goal buffer init. lead가 직접 verify한 진짜 위치:

`parkour_env.py:628-631`:
```python
yaw_raw = self._target_yaw - self._robot.data.heading_w           # [N] relative yaw error
# yaw_diff_new = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))  # wrap to [-π, π]   ← 주석
next_yaw_raw = self._next_target_yaw - self._robot.data.heading_w
# next_yaw_diff_new = torch.atan2(torch.sin(next_yaw_raw), torch.cos(next_yaw_raw))   ← 주석
```

→ **yaw wrap이 비활성화되어 `yaw_raw`가 unwrapped 상태로 policy obs와 reward에 들어감**. |yaw_raw| > π이면 `tracking_yaw = exp(-|yaw_raw|)` ≈ 0 → reward 신호 붕괴.

이전 RCA v2의 fix-auditor도 동일 finding을 짚었으나 라인을 617-618/625-626로 잘못 지목. **진짜 라인은 629, 631**.

### 1.3 reward-validator (코드 + git 분석) — 전체 채택

#### A. 사용자 제안 코드는 표준 패턴
- `anymal_c_env.py:128-132` + `R_Skeleton/skeleton_env.py:264-268`과 **bit-identical**
- 두 env 모두 `feet_air_time_reward_scale = 0.5` 사용
- parkour `_feet_ids`(`parkour_env.py:181`) + `track_air_time=True`(`parkour_env_cfg.py:454`) 이미 구비 → drop-in 가능

#### B. magnitude 충돌 없음
- 새 항 contrib: -0.0005 ~ +0.002 per step
- vs `tracking_goal_vel` +0.021/step (10~30× 차이) → dominance 없음
- vs `feet_dragging` -0.001~-0.004/step (같은 자릿수, 상보적)
- termination -100 one-shot에 가리지 않음 (다른 dense reward들이 termination에 살아남아 학습 중)

#### C. feet_dragging/stumble/edge와 상보적
- feet_dragging: stance phase (in-contact + sliding) 페널티
- feet_air_time: touchdown 시점 (air→contact) 양성 보상
- **이벤트 시점 다름 → 직접 cancellation 없음, 정보 보완**

#### D. go2_wtw 주석 처리는 "교체"이지 "실패" 아님
- commit `f3240fd4b37f` (2026-02-20)에서 feet_air_time 주석화와 **동시에** `raibert_heuristic` + `tracking_contacts_shaped_force/vel` + `bezier_target_pos` (phase-based gait reward 스택) 도입
- 즉 "더 구조적인 phase reward로 교체"이지 "도움 안 돼서 제거" 아님
- **parkour엔 phase reward 스택 없음 → feet_air_time 재도입 정당화**

#### E. "default pos에 가깝게" — 한 줄 uncomment로 해결
- 현재 `dof_error_l2 = -0.04`, `hip_pos = -0.5`은 이미 default pos 페널티
- `parkour_env.py:1003`에 **Genesis-faithful 코드가 이미 작성되어 주석 처리됨**:
  ```python
  # Genesis conditional: dof_error penalized 10x more on flat (nominal posture expected there)
  # dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)
  ```
- 이 한 줄 uncomment하면 flat에서 effective scale = `-0.04 × 10 = -0.4` (hip_pos와 같은 자릿수)
- cfg `reward_scales`를 손대지 않으므로 사용자 §3 제약 준수
- non-flat은 기존 -0.04 유지

---

## 2. 진단 종합

### 2.1 진짜 root cause (메모리 위반 항목 제거 후)

**평지 보행 실패의 두 메커니즘** (Tier A+B):

#### (M1) Drag-forward 안정점 락
- 정책이 학습은 했으나 **자세 무너뜨리고 발 끌면서 전진**하는 모드에 수렴
- 증거: feet_dragging 128× ↑, dof_acc_l2 61× ↑, dof_error_l2 117× ↑, ang_vel_xy_l2 59× ↑, net reward +20 양수
- 메커니즘: tracking_goal_vel +1.21 양수 dominance가 dragging 페널티 -0.128을 압도. 보상 landscape가 trot을 차별화하지 못함

#### (M2) Yaw 신호 결함
- yaw wrap이 비활성 → unwrapped yaw_diff가 reward/obs에 흘러감
- 증거: tracking_yaw +0.38만 (max 0.5 의 76%), ang_vel_xy_l2 59× 악화, entropy 비정상 증가
- 메커니즘: |yaw_diff| > π에서 reward collapse → 정책이 부드러운 yaw 학습 불가 → 진동 + 회전 페널티 누적

### 2.2 두 메커니즘은 서로를 강화

yaw 신호 결함(M2) → 정책이 yaw 학습 못함 → 회전 보상 신호 차단 → 정책이 forward velocity만 의지 → drag-forward 안정점(M1)으로 더 강하게 수렴.

→ 두 결함 모두 fix해야 trot 발현 기대 가능.

---

## 3. 권장 변경 — 단일 batch

### Change 1: `feet_air_time` reward 신설

**파일 1**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`

`_get_rewards()` 내부, `feet_dragging` 블록 직후 (현재 ~L1056) 추가:
```python
# === Feet air time (anymal_c / R_Skeleton 표준 패턴) ===
# 발이 일정 시간 air 후 contact한 순간 양성 보상. command 크기 > 0.1 m/s일 때만 활성.
first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
    torch.norm(self._commands[:, :2], dim=1) > 0.1
)
```

`reward_values` dict에 추가:
```python
"feet_air_time": feet_air_time,
```

**파일 2**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py`

`reward_scales` dict (L474)에 추가:
```python
"feet_air_time": 0.5,  # anymal_c/R_Skeleton 표준 scale; trot 유도
```

**자동 처리**: `_episode_sums`는 `reward_scales.keys()`를 순회하므로 추가 작업 불필요.

**Worker**: `reward-worker` (parkour_env.py + cfg 동시 수정).

### Change 2: `dof_error_l2` flat ×10 활성화

**파일**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py:1003`

```python
# BEFORE (현재 주석)
# dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)

# AFTER (uncomment)
dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)
```

cfg `reward_scales`는 변경 없음. 사용자 §3 제약 준수.

**Worker**: `reward-worker` (1 line).

### Change 3: Yaw wrap 복구

**파일**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py:629, 631`

```python
# BEFORE (현재 주석)
yaw_raw = self._target_yaw - self._robot.data.heading_w
# yaw_diff_new = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))  # wrap to [-π, π]
next_yaw_raw = self._next_target_yaw - self._robot.data.heading_w
# next_yaw_diff_new = torch.atan2(torch.sin(next_yaw_raw), torch.cos(next_yaw_raw))

# AFTER (uncomment + 변수 교체)
yaw_raw = self._target_yaw - self._robot.data.heading_w
yaw_diff_new = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))
next_yaw_raw = self._next_target_yaw - self._robot.data.heading_w
next_yaw_diff_new = torch.atan2(torch.sin(next_yaw_raw), torch.cos(next_yaw_raw))
```

이후 line 633 이후 `do_global_refresh` / `recently_reset` 분기에서 `yaw_raw`/`next_yaw_raw` 대신 `yaw_diff_new`/`next_yaw_diff_new`를 사용하도록 확인. (현재 코드 흐름을 worker가 verify 후 일관 적용.)

**Worker**: `obs-worker` (parkour_env.py:629-635 영역).

---

## 4. 학습 + 판정

### 4.1 학습 명령

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Parkour-Direct-v0 --num_envs 4096 --max_iterations 2000 \
  --logger wandb --wandb-project IsaacLab-parkour --run_name flat_trot_3changes \
  --headless
```

- flat-only로 학습 (terrain cfg 그대로 유지하면 됨 — `proportion=1.0` for flat)
- 3개 변경이 한 batch → 효과는 종합 평가 (개별 ablation은 다음 이터레이션)
- from-scratch 학습 (기존 checkpoint resume도 가능하나 reward 구조 변화로 unstable 가능)

### 4.2 판정 메트릭 (TB)

**Trot 발현 (Change 1 효과)**:
- `Episode_Reward/feet_air_time` — 초기 음수 (학습 전 air time 짧음) → 양수 전환 (200-500 iter 내)
- `Episode_Reward/feet_dragging` — 현재 -0.128에서 절대값 감소 (드래그 줄어들면 페널티 누적 적음)
- `dof_acc_l2` — 61× 악화 추세 반전 또는 안정

**자세 정상화 (Change 2 효과)**:
- `Episode_Reward/dof_error_l2` — flat에서 강화되므로 처음 페널티 크게 박힘 → 학습 후 감소
- `Episode_Reward/hip_pos` — 함께 회복

**Yaw 신호 정상화 (Change 3 효과)**:
- `Episode_Reward/tracking_yaw` — 현재 +0.380 → +0.45+ 기대 (max 0.5의 90%)
- `ang_vel_xy_l2` — 59× 악화 반전
- `Loss/entropy` — PPO 정상 거동 (감소 추세)
- `Policy/mean_noise_std` — 1.9에서 감소 추세

**보행 자체**:
- `Episode/episode_length` — 824 → 1000+ (full episode)
- `curriculum/mean_terrain_level_flat` — 6 → 9+
- play.py rollout: **4-leg trot 가시화**, 발 dragging 없음

### 4.3 If 부분 효과만

이터레이션 N+1 dispatch 계획:
- Change 효과 분리 평가 — 각 1개씩 비활성화한 ablation
- yaw wrap이 효과 없으면 obs 표현 추가 점검 (yaw_diff가 history에 마스킹되는 Fix-2 잔존 영향)
- trot 발현 약하면 scale 0.5 → 1.0으로 한 단계 강화 (단, 다른 변경 없이)
- drag mode 잔존하면 feet_dragging weight 강화 검토 (현재 -0.1, scale 변경은 신중)

---

## 5. 사용자 답변 정리

**질문 1**: 제안 코드(`first_contact`, `last_air_time`, `air_time`)가 보행 학습에 효과적인가?
→ **YES**. IsaacLab 표준 패턴(anymal_c, R_Skeleton)과 bit-identical. parkour drop-in 가능. magnitude 안전, 기존 reward와 상보적. **trot 발현 + drag-forward 안정점 탈출 기대**.

**질문 2**: "평지의 경우 default pos에 가깝게 움직이고" — 어떻게 구현?
→ **별도 신규 reward 불필요**. `parkour_env.py:1003`의 Genesis-faithful 코드가 이미 작성되어 주석 처리됨. **한 줄 uncomment**로 flat에서 `dof_error_l2` 효과 ×10. cfg 변경 없음.

**추가 발견**: 평지 학습 실패의 다른 root cause로 **yaw wrap regression** (`parkour_env.py:629, 631` 주석 처리) 확인. 별도로 복구 권장.

---

## 6. References

- 입력 보고서:
  - `_workspace/parkour_flat_run_diagnostic.md` (flat-diagnostic, **Hypothesis #2 제외**)
  - `_workspace/parkour_feet_air_time_validation.md` (reward-validator, 전체 채택)
- 코드 evidence:
  - `parkour_env.py:181, 629-631, 1003, 1056` (lead 직접 verify)
  - `parkour_env_cfg.py:454, 474` (lead 직접 verify)
  - `anymal_c_env.py:128-132`, `R_Skeleton/skeleton_env.py:264-268` (표준 패턴)
  - `go2_wtw_env.py:423-424` + commit `f3240fd4b37f` (주석 처리 이력)
- 영구 메모리:
  - `~/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_analysis_constraints.md`

*End of action plan.*

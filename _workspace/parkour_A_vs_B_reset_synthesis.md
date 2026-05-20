# Parkour A(direct) vs B(Isaaclab_Parkour) — Reset/Termination/Curriculum 비교 종합

**작성일**: 2026-05-20
**프레임**: B는 학습이 잘 되는 환경(known-good), A는 학습 성공 이력 없음. "A의 버그를 찾는다"가 아니라 **B에 있고 A에 없거나 다른 구조**를 사실 기반으로 비교.
**입력 보고서**:
- `_workspace/parkour_A_reset_extraction.md` (worker-A, direct/parkour)
- `_workspace/parkour_B_reset_extraction.md` (worker-B, Isaaclab_Parkour)

---

## 핵심 결론 (TL;DR)

두 환경의 **reset/termination/curriculum 메커니즘은 구조적으로 매우 유사**하다. 단 하나, **PPO bootstrap 시 어떤 종료 사유를 `time_out`(truncation)으로 분류하는가**에서 결정적 차이가 있다.

- **A**: `terminated`(failure+goal_success) vs `time_out`(episode timeout만) — RL 표준대로 분리
- **B**: **모든 termination이 `time_out=True`** — failure(roll/pitch/height)에도 bootstrap value 적용

이 차이가 사용자가 관찰한 "B가 학습은 더 좋은데 ep_len은 더 짧다"의 **유일한 인과 원인이라는 보장은 없다**. reset/term/curriculum 외 영역(reward 항 구성, network, observation, hyperparam)은 본 비교 범위 밖이다.

ep_len 차이의 원인은 본 코드 비교로는 결정 불가 — **termination cause breakdown 로그**(goal_reached vs tilt vs low_height vs timeout 비율)로 구별해야 한다.

---

## 1. 5개 항목 비교 표

| 항목 | A (direct/parkour) | B (Isaaclab_Parkour) |
|------|---|---|
| **Framework** | DirectRLEnv (단일 파일) | ManagerBasedRLEnv (이벤트 기반) |
| **episode_length_s** | **20.0 s** (parkour_env_cfg.py:383) | **20.0 s** (parkour_teacher_cfg.py:50) |
| **decimation / sim.dt** | 4 / 1/200 = 0.005 s | 4 / 0.005 s |
| **max_episode_length** | **1000 steps** (timeout @ buf>=999) | **1000 steps** (timeout @ buf>=1000) |
| **Grace period (failure 무시)** | 5 steps (failure만, goal에는 미적용) — `parkour_env.py:1132` | 없음 (모든 termination 즉시) |
| **Failure 조건** | tilt(grav_xy²>0.99), low_height(z<-0.2m) — `parkour_env.py:1120, 1123` | roll/pitch(>1.5 rad), height(z<-0.25m) — `terminations.py:23-28` |
| **Goal 도달 → 즉시 reset** | ✅ 있음. 0.1 s hold 후 — `parkour_env.py:594, 1133` | ✅ 있음. cur_goal_idx >= num_goals — `terminations.py:36-38` |
| **Termination 분류 → PPO** | `terminated` = failure ∪ goal_success ∪ ~grace; `time_out` = episode timeout만 — `parkour_env.py:1110-1134` | **모든 reset_buf → `time_out=True`** flag — `parkour_mdp_cfg.py:250-256` |
| **PPO bootstrap on failure** | ❌ NO (value=0, terminal) | **✅ YES (γ × V(s'))** |
| **PPO bootstrap on goal_success** | ❌ NO (value=0, terminal) | ✅ YES |
| **PPO bootstrap on timeout** | ✅ YES | ✅ YES |
| **Curriculum metric** | XY distance / (v_cmd × ep_len_s) — `parkour_env.py:1266-1273` | 동일 — `parkour_event.py:143-144` |
| **Level-up threshold** | dis > **0.8** × expected | dis > **0.8** × expected |
| **Level-down threshold** | dis < **0.4** × expected | dis < **0.4** × expected |
| **Curriculum 호출 위치** | `_reset_idx()` Phase 3 — `parkour_env.py:1173-1176` | `curriculum_manager.compute()` + `_resample_command` — `parkour_manager_based_rl_env.py:253`, `parkour_event.py:133-176` |
| **Reward clip(min=0)** | 있음 (사용자 메모 + B와 동일 설계) — A는 의도된 설계 | 있음 — `parkour_reward_manager.py:38` |
| **PPO 하이퍼파라미터** | (별도 cfg 비교 필요, `Go2ParkourPPORunnerCfg`) | gamma=0.99, lam=0.95, num_steps_per_env=24 — `rsl_teacher_ppo_cfg.py:46-47, 13` |
| **PPO/Runner 클래스** | `OnPolicyRunnerParkour` + `PPOParkour` — `agents/rsl_rl_ppo_cfg.py:30,69` | `ParkourRslRlOnPolicyRunnerCfg` — `rsl_teacher_ppo_cfg.py` |

### 1-1. 동일한 부분 (B에 있고 A에도 있음)

워커-B 보고서의 잘못된 추측(보고서 408–415줄)을 정정한다:

| worker-B의 추측 | 실제 (worker-A 보고서로 검증) | 정정 |
|---|---|---|
| "A는 goal achievement로 종료 안 할 것" | A `parkour_env.py:594, 1133`: 0.1 s hold 후 `_term_goal_reached` 플래그로 즉시 reset | **A에도 goal-based 종료 있음** |
| "A는 fixed terrain level/success-rate curriculum 사용할 것" | A `parkour_env.py:1271-1273`: 정확히 동일한 distance-based 0.8/0.4 임계 | **A에도 distance-based per-env curriculum 있음** |
| "A는 negative reward를 clip 안 할 것" | 메모리: A에도 reward total clip(min=0) 의도된 설계(`project_parkour_reward_clip_intentional.md`) | **A에도 reward clip(min=0) 있음** |

이 세 영역은 **두 환경이 동일**하다. ep_len 차이의 원인으로 거론 금지.

---

## 2. 가장 큰 구조적 차이: Termination → PPO Bootstrap 분류 정책

### 2-1. A: 표준 RL 분리

**`parkour_env.py:1110-1134`** (`_get_dones`):
```python
time_out = self.episode_length_buf >= self.max_episode_length - 1
terminated = self._term_tilt | self._term_low_height
grace = self.episode_length_buf < self.cfg.termination_grace_steps
terminated = (terminated & ~grace) | self._term_goal_reached
return terminated, time_out
```

→ DirectRLEnv가 5-tuple로 반환:
**`source/isaaclab/isaaclab/envs/direct_rl_env.py:420`**:
```python
return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

→ RslRlVecEnvWrapper가 분리:
**`source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:168, 172`**:
```python
dones = (terminated | truncated).to(dtype=torch.long)
extras["time_outs"] = truncated  # ← 오직 episode-length timeout만
```

→ PPO bootstrap (A는 `PPOParkour` 사용, `ppo_parkour.py:188-190`):
```python
if "time_outs" in extras:
    self.transition.rewards += self.gamma * torch.squeeze(
        self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
    )
```

**효과**:
- Failure(tilt/low_height) → `terminated=True, time_out=False` → bootstrap **없음** → value=0 (terminal로 학습)
- Goal success → `terminated=True, time_out=False` → bootstrap **없음** → value=0
- Episode timeout(999) → `terminated=False, time_out=True` → bootstrap **있음** → r + γV(s')

> **참고 (정정)**: worker-A 보고서 5절은 `rsl_rl/algorithms/ppo.py:144-172`를 인용했지만, A가 실제 사용하는 알고리즘은 `Go2ParkourPPORunnerCfg.class_name="PPOParkour"`로 명시된 `ppo_parkour.py`다. bootstrap 로직 자체는 두 파일이 동일 (`ppo_parkour.py:188-190` ↔ `ppo.py:165-167`)이므로 worker-A의 결론은 유효. git status의 `*_parkour_original.py`는 사용하지 않는 legacy 파일.

### 2-2. B: 모두 time_out으로 통합

**`parkour_isaaclab/envs/mdp/terminations.py:25-43`**:
```python
def terminate_episode(env, asset_cfg):
    reset_buf = torch.zeros(...)
    roll_cutoff = abs(wrap_to_pi(roll)) > 1.5
    pitch_cutoff = abs(wrap_to_pi(pitch)) > 1.5
    time_out_buf = env.episode_length_buf >= env.max_episode_length
    reach_goal_cutoff = parkour_event.cur_goal_idx >= num_goals
    height_cutoff = root_state_w[:, 2] < -0.25
    time_out_buf |= reach_goal_cutoff  # goal도 timeout으로
    reset_buf |= time_out_buf
    reset_buf |= roll_cutoff
    reset_buf |= pitch_cutoff
    reset_buf |= height_cutoff
    return reset_buf  # ← 모든 사유 통합 단일 반환
```

**`parkour_mdp_cfg.py:250-256`**:
```python
total_terminates = DoneTerm(
    func=terminations.terminate_episode,
    time_out=True,   # ← 단일 DoneTerm, 모두 time_out으로 분류
    ...
)
```

→ ManagerBased termination_manager가 분리:
**`parkour_manager_based_rl_env.py:123-125`**:
```python
self.reset_terminated = self.termination_manager.terminated   # time_out=False인 term만
self.reset_time_outs  = self.termination_manager.time_outs    # time_out=True인 term
```

B에는 `time_out=True` flag를 가진 단일 DoneTerm만 등록되어 있으므로 → `reset_terminated`는 **항상 비어 있고**, **모든 reset 사유가 `reset_time_outs`로 PPO에 전달**된다.

**효과**:
- Failure(roll/pitch/height) → bootstrap **있음** (r + γV(s'))
- Goal reached → bootstrap **있음**
- Episode timeout → bootstrap **있음**

### 2-3. 인과 해석 — 과해석 금지

**구조적 사실**: A는 failure/success에 terminal(V=0), B는 모두에 V(s') bootstrap.

**RL 이론 표준**: failure state에 V(s') bootstrap을 적용하면 일반적으로 **over-optimistic value** → value loss 증가 → 학습 악화가 예상되는 방향. 따라서 "B가 모든 걸 time_out으로 처리해서 학습이 잘된다"는 **단일 인과 설명은 약하다**.

**왜 B에서 이 패턴이 학습을 망가뜨리지 않는가** (가설, 미검증):
- B는 reward total clip(min=0)으로 페널티 신호가 0에 가까움 → failure 직전 V(s')가 작게 학습됨 → bootstrap 오류 영향 작음
- γ=0.99 + GAE(λ=0.95)에서 단일 step bootstrap 오류는 GAE 평균으로 희석
- failure 직전 state의 V(s')가 자연스럽게 작은 값으로 수렴

**진짜 학습 우위 원인**은 본 비교 범위(reset/term/curriculum) **밖**에 있을 가능성도 있다:
- reward 항 구성/weight (penalty 항목, gait reward 등)
- network 구조 (ActorCriticRMA in A — `agents/rsl_rl_ppo_cfg.py:58` vs B의 architecture)
- observation 차이
- domain randomization 차이
- PPO 하이퍼파라미터 차이

**결론 강도**:
- ✅ 확인된 구조 차이: termination → PPO bootstrap 분류 정책
- ⚠️ 인과: B 학습 우위의 단일 원인이라는 보장 없음

---

## 3. 사용자 질문 직접 답변

### Q1. "왜 B가 학습 결과가 더 좋은가?"

본 비교 범위(reset/term/curriculum)에서 확인된 **유일한 구조 차이**는 termination → PPO bootstrap 분류 정책이다. 다만 위 §2-3에서 짚었듯 단일 인과 설명으로 부족.

**다음 단계 권장**:
1. **Reward 항 구성 비교** — A의 reward_scales (parkour_env_cfg.py) vs B의 reward terms (parkour_mdp_cfg.py)
2. **Network 비교** — A: `ActorCriticRMA`(RMA estimator 포함), B: `parkour_teacher_cfg.py`의 policy class
3. **PPO 하이퍼파라미터 diff** — A `Go2ParkourPPORunnerCfg` vs B `UnitreeGo2ParkourTeacherPPORunnerCfg`
4. **Observation/DR 비교** — privileged obs, height_scan, domain randomization 설정

### Q2. "왜 B의 mean episode length가 더 짧은가?"

**코드 비교로는 결정 불가**. 두 환경 모두:
- 동일한 max_episode_length (1000 steps)
- 동일한 goal-based immediate reset
- 동일한 failure-based immediate reset (A는 grace 5 steps 차이)
- 동일한 distance-based curriculum

따라서 ep_len 차이는 **학습 상태에 의한 결과**이지 reset 메커니즘의 직접 영향이 아니다.

**두 가지 가설**:
1. **Curriculum 가속 가설**: B가 학습이 잘 되어 → distance 진행률 빨라 → level up 빠름 → 더 어려운 terrain → 자주 fall → ep_len 짧음
2. **Goal 도달 빈도 가설**: B가 학습이 잘 되어 → goal에 더 자주 도달 → 0.1 s hold 후 즉시 reset → ep_len 짧음

**검증 방법**: 두 환경 모두 termination cause를 로깅한다 — A는 `parkour_env.py:1206-1224`(base_contact/tilt/low_height/goal_reached counter), B는 termination_manager.episode_sums. **각 사유 비율**을 비교하면 어느 가설이 우세한지 즉시 판별 가능.
- goal_reached 비율이 B가 훨씬 높으면 → 가설 2
- tilt/height(failure) 비율이 B가 높으면 → 가설 1

### Q3. "reset과 관련된 문제일 것으로 추정되는데?"

**부분적으로 맞음**. 정확히는 "reset 로직"이 아니라 **"reset을 trigger한 사유를 PPO에 어떻게 알리는가"**의 차이. reset_idx 흐름, curriculum, init state 분포는 두 환경이 거의 같다. 차이는 termination classifier 단계에 집중되어 있다.

---

## 4. Worker 보고서 신뢰성 정정

| 항목 | 정정 |
|---|---|
| worker-A: `rsl_rl/algorithms/ppo.py:165-172` 인용 | **사실은 `ppo_parkour.py:188-190`** (A는 `class_name="PPOParkour"` 사용). bootstrap 로직은 동일하므로 결론 유효. |
| worker-B 408-415줄 "A에는 goal termination 없을 것" | **틀림**. A `parkour_env.py:1133`에 있음. |
| worker-B 408-415줄 "A는 fixed level/success-rate curriculum일 것" | **틀림**. A도 distance-based 0.8/0.4 동일. |
| worker-B 408-415줄 "A는 reward clip 없을 것" | **틀림**. A에도 reward clip(min=0) 있음 (메모리 `project_parkour_reward_clip_intentional.md`). |

이 정정은 cross-codebase에서 "X 없다"고 단정하기 전 반드시 grep 검증해야 한다는 규칙(`feedback_worker_absence_claim_must_verify.md`)을 따른 결과.

---

## 5. 검증된 부재 / 미검증 영역

**검증된 부재 (A에 명시적 없음)**:
- (없음 — worker-A가 검증한 모든 항목이 코드에 존재)

**미검증 영역 (본 비교 범위 밖, 학습 우위 원인 가능성)**:
- reward 항 구성/weight diff
- network architecture diff (ActorCriticRMA vs B의 teacher policy)
- observation 구성 (privileged obs, exteroceptive)
- domain randomization 설정
- PPO 하이퍼파라미터 (clip_param, learning_rate, num_learning_epochs 등)

---

## 6. 한 줄 요약

> **두 환경의 reset/curriculum 메커니즘은 거의 동일하다. 유일한 결정적 구조 차이는 "B는 모든 termination을 `time_out=True`로 분류해 PPO bootstrap에 V(s')를 적용한다"는 점이다. 다만 이것이 B의 학습 우위와 ep_len 단축의 단일 인과라고 단정할 수는 없다 — 비교 범위 밖(reward/network/hyperparam) 차이가 더 클 수 있고, ep_len 차이는 termination cause breakdown 로그로 직접 판별해야 한다.**

---

**파일 산출물**:
- `_workspace/parkour_A_reset_extraction.md` (worker-A)
- `_workspace/parkour_B_reset_extraction.md` (worker-B)
- `_workspace/parkour_A_vs_B_reset_synthesis.md` (본 문서, team-lead)

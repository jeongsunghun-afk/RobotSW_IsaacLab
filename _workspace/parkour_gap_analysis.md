# Parkour Environment Gap Analysis

작성일: 2026-04-30
대상: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`

## 비교 대상 (3 references)

| 항목 | 경로 | 시뮬레이터 |
|------|------|-----------|
| 현재 (Target) | `IsaacLab/source/.../direct/parkour/` | Isaac Sim / IsaacLab |
| Ref-A | `RobotSW_Parkour/legged_gym/` | Isaac Gym (legged_gym) |
| Ref-B | `RobotSW_Genesis/parkour/` | Genesis |
| Ref-C (자산) | `IsaacLab/rsl_rl/rsl_rl/algorithms/ppo_parkour.py` 외 | (알고리즘 코드만) |

> **이식 대상은 아키텍처 컨셉이지 구현 코드가 아니다** — 시뮬레이터 API/joint indexing/terrain generation/scene managers 모두 다름. Ref-A·B의 코드를 통째로 복붙하지 말 것.

---

## 결정적 발견 — IsaacLab rsl_rl에 PPOParkour 자산이 이미 존재

`/home/lgb/IsaacLab/rsl_rl/rsl_rl/`:
- `algorithms/ppo_parkour.py` — `PPOParkour` 클래스 (`update_dagger`, priv_reg loss, `priv_reg_coef_schedual=[0,0.1,2000,3000]`)
- `algorithms/distillation.py` — student/teacher 분리
- `modules/actor_critic_parkour.py` — `ActorCriticRMA` + `StateHistoryEncoder`
- `modules/estimator.py` — proprio→priv estimator
- `modules/depth_backbone.py` — depth CNN

**현재 환경은 이걸 일절 사용하지 않음.** `agents/rsl_rl_ppo_cfg.py` → 표준 `RslRlPpoActorCriticCfg` + `RslRlPpoAlgorithmCfg`.

---

## Gap 매트릭스

### 1. Goal Waypoint System

| 요소 | Ref-B (Genesis) | Ref-A (Isaac Gym) | 현재 IsaacLab | 상태 |
|------|------------------|---------------------|------|------|
| `terrain_goals[L,T,K,3]` (terrain별 goal 좌표) | terrain metadata에 `"goals"` 키로 저장 | terrain_utils 별도 sample | **없음** | ❌ MISSING |
| `env_goals[N, K+F, 3]` per-env goal sequence | `_get_env_origins`에서 구성 | 동일 | **없음** | ❌ MISSING |
| `cur_goal_idx[N]` | 사용 (`_update_goals`로 advance) | 사용 | 버퍼만 선언 (`_current_goal_idx`), 사용 안 됨 | ⚠️ DEAD CODE |
| `cur_goals = env_goals.gather(...)` | `_gather_cur_goals(future=0)` | 동일 | **없음** | ❌ MISSING |
| `target_pos_rel`, `target_yaw` | `_update_goals`에서 매 step 갱신 | 동일 | **없음** | ❌ MISSING |
| Goal reach 검출 | `norm(base-goal) < threshold` + `reach_goal_timer` delay | 동일 | **없음** | ❌ MISSING |
| `reach_goal_delay` (goal 도달 후 다음으로 넘어가는 hold time) | cfg에 있음 | 있음 | 없음 | ❌ MISSING |
| `next_goal_threshold` | cfg에 있음 (보통 0.3~0.5m) | 있음 | `goal_reach_threshold=0.5`만 선언 | ⚠️ ORPHAN CFG |
| `_resample_commands` semantics | random vel — 단, parkour에선 vel command가 약하게 작동, 실제 학습은 goal 추종으로 | random vel | 단순 random vel | ⚠️ NOT GOAL-DRIVEN |

**구현 방향:**
- IsaacLab `TerrainGeneratorCfg`의 각 sub-terrain에 `flat_patch_sampling` dict로 `init_positions` 외에 `goal_positions` 패치 정의를 추가하거나, post-hoc으로 goal 좌표를 환경 origin 기준 일정 간격 직선상에 sampling.
- 또는 **terrain importer가 노출하는 `terrain_origins`를 기준으로 +x 방향으로 (goal_distance/num_goals) 간격으로 K개 일직선 goal 생성** — 이게 가장 단순하고 작동 보장. 향후 정교한 path는 다음 iter.
- `env_goals[N, num_goals+num_future, 3]` 텐서, `cur_goal_idx[N]`, `_update_goals()`, `_gather_cur_goals(future=0/1)` 메서드를 `parkour_env.py`에 추가.
- `_get_observations`와 `_get_rewards`에서 `target_pos_rel`, `target_yaw` 사용.

### 2. Reward — Goal Tracking

| Reward | Ref-B (Genesis) | 현재 | 상태 |
|--------|------------------|------|------|
| `tracking_goal_vel` | `cos(theta)` × forward speed projection on goal direction (line 1451-1459 추정) | **없음** | ❌ MISSING — 가장 핵심 |
| `tracking_yaw` | `exp(-|target_yaw - heading|)` | **없음** | ❌ MISSING |
| `tracking_lin_vel` | (baseline용으로만, parkour scale 작음) | 1.5 | OK 단 weight 재조정 필요 |
| `feet_edge`, `feet_stumble` | terrain `x_edge_mask` 사용 정밀 | contact-force proxy로 단순화 | ⚠️ DEGRADED |
| `total_reward.clamp(min=0)` | **없음** (penalty 그대로 음수) | clamp 적용됨 | 🐞 BUG — 페널티 무력화 |

**구현 방향:**
- `tracking_goal_vel`: `forward = quat_apply(base_quat_yaw_only, [1,0,0])`, `cur_vel = root_lin_vel_w[:, :2]`, `goal_dir = normalize(target_pos_rel[:, :2])`. reward = `min(dot(cur_vel, goal_dir), commanded_speed) / commanded_speed`. 즉 명령 속도까지만 보상.
- `tracking_yaw`: `exp(-|target_yaw - rpy_z|)` 또는 `cos(target_yaw - heading)` 양수.
- `total_reward.clamp(min=0)` 제거 — penalty가 의도대로 작동해야 함.

### 3. Network — ActorCriticRMA 미사용

| 요소 | `ActorCriticRMA` 요구 | 현재 `RslRlPpoActorCriticCfg` | Gap |
|------|---------------------|----------------------------|-----|
| `obs_groups["policy"]` | proprio (joint, vel, gravity, prev_actions, goal_vec, command) | 단일 229-d concat | ❌ |
| `obs_groups["critic"]` | proprio + scan + priv (full info) | 동일 단일 텐서 | ❌ |
| `obs_groups["scan"]` | height_scan (별도) → `scandot_encoder` MLP | concat 안에 포함 | ❌ |
| `obs_groups["priv"]` | domain rand params (mass, friction, motor_strength 등) → `priv_encoder` MLP | 없음 | ❌ |
| `obs_groups["history"]` | proprio T-step buffer (T=10/20/50) → `StateHistoryEncoder` Conv1D | 없음 | ❌ |
| `actor_hidden_dims` | [512,256,128] | [512,256,128] | OK |
| `priv_encoder_dims` | [64, 20] | n/a | ❌ |
| `scan_encoder_dims` | [128, 64, 32] | n/a | ❌ |
| `history len (tsteps)` | 10/20/50 | n/a | ❌ |

**구현 방향:**
- env가 dict observation을 반환해야 함: `{"policy": proprio, "critic": full, "scan": height_scan_187, "history": history_buf, "priv": priv_obs}`
- Cfg에 `obs_groups`와 위 차원들을 명시. `RslRlPpoActorCriticCfg`로 표현 안 되면 dict-based config로 직접 전달.
- `DirectRLEnvCfg.observation_space`를 단일 int(229) 대신 dict 형태로 변경 (Gymnasium spaces.Dict). 또는 `rsl_rl` runner 측이 이미 obs_groups를 처리하면 그쪽 규약 따르기.

### 4. Algorithm — PPOParkour 미연결

| 항목 | PPOParkour | 현재 PPO | Gap |
|------|-----------|----------|-----|
| `priv_reg_coef_schedual=[0,0.1,2000,3000]` | counter 기반 점진적 priv-reg loss 가중치 ramp | 없음 | ❌ |
| `update_dagger()` (history encoder가 priv encoder 모방) | 별도 hook으로 호출됨 | 없음 | ❌ |
| `hist_encoder_optimizer` | 분리 optimizer | 없음 | ❌ |
| KL adaptive lr | 동일 | 동일 | OK |
| RND/Symmetry | 옵션 | 옵션 | OK |

**구현 방향:**
- `agents/rsl_rl_ppo_cfg.py`에서 algorithm/policy class를 `PPOParkour` / `ActorCriticRMA`로 변경. RslRl* configclass가 이걸 명시적으로 지원 안 하면 두 가지 경로:
  1. 새 RslRl* configclass 정의 (priv/scan/history dim 추가)
  2. 또는 `agents/rsl_rl_ppo_cfg.py`를 dict-based로 작성하고 train.py 라우팅에 맞춤
- `runner.algorithm_class_name="PPOParkour"`, `policy_class_name="ActorCriticRMA"` 인식 여부를 `IsaacLab/rsl_rl/rsl_rl/runners/on_policy_runner.py`에서 확인 (이미 있을 가능성 높음).
- `update_dagger()` 호출 시점: PPOParkour는 일정 stage 이후 매 N iter마다 dagger 호출. runner가 이걸 인식하는지 확인.

### 5. Domain Randomization — 주석만 있음

| 항목 | Ref-B | 현재 | 상태 |
|------|--------|------|------|
| `_randomize_friction` | 구현됨 | 주석만 | ❌ |
| `_randomize_base_mass` | 구현됨 | 주석만 | ❌ |
| `_randomize_com_displacement` | 구현됨 | 없음 | ❌ |
| `_randomize_motor_strength` | 구현됨 | 주석만 | ❌ |
| `_randomize_motor_offset` | 구현됨 | 없음 | ❌ |

**구현 방향:**
- IsaacLab 자산 활용: `isaaclab.envs.mdp.events`에 `randomize_rigid_body_material`, `randomize_actuator_gains` 등 헬퍼가 있음. Direct RL env에서는 `_reset_idx`에서 직접 호출.
- 또는 `DirectRLEnvCfg.events: EventCfg`로 선언 (event term).
- 결과(현재 friction, mass, kp scale 등) → `priv_obs[N, num_priv]` 텐서에 저장 → ActorCriticRMA의 `priv_encoder` 입력으로 공급.

### 6. 기타 차이

| 항목 | Ref | 현재 | 영향 |
|------|-----|------|------|
| `obs_history_buf[N, T, num_proprio]` ring buffer | 있음 | 없음 | StateHistoryEncoder 입력 누락 |
| `last_root_vel`, `last_torques` | 있음 (smoothness reward용) | 없음 | smoothness reward 못 만듦 |
| `simulate_action_latency` | 옵션 | 없음 | sim2real 불리 |
| `reset_buf` 명시 관리 | 있음 | DirectRLEnv 내장 | OK |
| `extras["episode"]` per-terrain level logging | 6종 terrain별 분리 | 단일 `mean_terrain_level` | 학습 분석 빈약 |

---

## 수정 우선순위 (smoke test 통과를 위한 최소 변경)

500 iter smoke test에서 reward 상승을 보려면 **모든 항목을 한 번에 바꾸는 것보다** 단계적이 안전. 다만 ActorCriticRMA를 쓰려면 obs dict 분리가 동시에 필요함 — 이건 한 번에 묶어야 함.

### Phase A (Critical — 한 PR에 묶음)
1. **Goal waypoint 시스템** (Task #2): `env_goals`, `cur_goal_idx`, `_update_goals`, `_gather_cur_goals`, `target_pos_rel`, `target_yaw`.
2. **Goal-tracking reward** (Task #4): `tracking_goal_vel`, `tracking_yaw`, `clamp(min=0)` 제거.
3. **Obs dict 분리** (Task #3): `{policy, critic, scan, history, priv}`. priv는 일단 zero tensor로 시작 (도메인 랜덤화는 나중).
4. **Cfg dict observation_space** (Task #6): num_actor_obs / num_priv_obs / num_scan_obs / history_len 정의.
5. **agents cfg → PPOParkour/ActorCriticRMA** (Task #7).

### Phase B (Quality)
6. Domain randomization 실구현 → priv 채움 (Task #5).
7. validate-code, validate-method (Task #9, #10).

### Phase C (Verify)
8. **500 iter smoke test** — `tracking_goal_vel` mean reward가 0보다 유의하게 상승하면 PASS.

---

## 위험 요소 (워커가 빠지기 쉬운 함정)

1. **Genesis x/y 좌표가 IsaacLab과 다를 수 있음** — Genesis는 +x 전진을 기본, IsaacLab도 동일. 단 quat convention 주의 (`wxyz` vs `xyzw`).
2. **`flat_patch_sampling`은 init_positions용** — goal에 그대로 쓰려면 별도 키로 추가하거나 직접 sampling 함수 작성. 가장 단순한 방법은 terrain origin + 등간격 직선이지만, 계단/갭 위로 좌표가 떨어질 수 있어 **height_scan 데이터로 z 보정 필요**.
3. **DirectRLEnv의 observation_space dict 지원 여부** — `Gymnasium.spaces.Dict`로 선언 시 runner가 이를 obs_groups로 자동 처리하지 않을 수 있음. 필요하면 `_get_observations` 반환만 dict로 하고 `observation_space`는 총합 int로 두되, runner 측에서 obs_groups 따로 명시.
4. **PPOParkour가 요구하는 storage type** — `RolloutStorage.Transition`이 `obs: TensorDict`를 기대. 현재 표준 PPO storage가 dict를 처리하는지 확인 필요.
5. **`update_dagger` 호출 hook** — `OnPolicyRunner`가 알고리즘에 `update_dagger`가 있으면 호출하는지 확인. 없으면 runner를 fork하지 말고 **stage 진입 후에는 일반 PPO update에서 `hist_encoding=True`로 전환**하는 단순화 가능.

---

## 합격 기준 (User 컨펌됨)

- **A옵션 / 500 iter smoke test**
- 합격 조건:
  - 학습이 NaN/divergence 없이 500 iter 완주
  - `Episode_Reward/tracking_goal_vel` (또는 동등 reward) mean이 100 iter 평균 < 400 iter 평균 (의미 있는 상승)
  - `mean_terrain_level` 또는 curriculum metric이 정체/하락하지 않음
  - policy gradient norm 폭발 없음

---

## 다음 단계

1. ENV 작업: obs-worker에 Task #2/#3/#5 묶음, reward-worker에 Task #4 위임 (병렬)
2. CFG 작업: cfg-worker에 Task #6, #7 위임 (Phase A 완료 후)
3. 라우팅 확인: Explore에 Task #8 위임 (Phase A와 병렬)
4. validate → train smoke test → 결과 보고

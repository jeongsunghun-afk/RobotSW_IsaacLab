# A(direct/parkour) Reset/Termination/Curriculum Flow Extraction

**목표**: Codebase A의 termination, max_episode_length, reset flow, curriculum, PPO bootstrap을 사실(file:line)로 추출하기.  
**프레임**: A는 학습 성공 이력이 없다. 버그 찾는 것 아니라 **구조 사실 추출**. "X 없음"은 grep 재검증 후 "검증된 부재" 표기.

---

## 1. Termination 조건 (failure vs success vs timeout 분리)

### _get_dones() 메서드

**파일:라인**: `parkour_env.py:1110-1134`

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    time_out = self.episode_length_buf >= self.max_episode_length - 1

    net_contact_forces = self._contact_sensor.data.net_forces_w_history
    # Base contact with ground
    self._term_base_contact = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 5.0,
        dim=1,
    )
    # Excessive tilt: |projected_gravity_xy|^2 > sin^2(1.5 rad) ≈ 0.997
    self._term_tilt = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) > 0.99

    # Robot too low (fallen into terrain gap or flipped)
    self._term_low_height = self._robot.data.root_link_pos_w[:, 2] < self.cfg.termination_height

    # Goal-reached termination is set in _update_goals() (called from _get_observations()).
    # It fires when the robot holds position at the last waypoint long enough — a success event,
    # NOT a failure.  Use terminated=True (not time_out) so the value bootstrap is zero (episode
    # truly ends) rather than using the next-state value estimate.
    # Grace period applies only to failure conditions, not to goal success.
    # terminated = self._term_base_contact | self._term_tilt | self._term_low_height
    terminated = self._term_tilt | self._term_low_height
    grace = self.episode_length_buf < self.cfg.termination_grace_steps
    terminated = (terminated & ~grace) | self._term_goal_reached
    return terminated, time_out
```

### Termination 분류

| 조건 | 타입 | 설명 | 파일:라인 |
|------|------|------|---------|
| **Failure (terminated)** |  |  |  |
| Excessive tilt | `terminated` | projected_gravity_xy² > 0.99 | `parkour_env.py:1120` |
| Low height (fall) | `terminated` | root_pos_z < termination_height (-0.2m) | `parkour_env.py:1123`, `parkour_env_cfg.py:600` |
| Goal reached (success!) | `terminated` | 마지막 waypoint 도달 후 hold_time 경과 | `parkour_env.py:1133`, `parkour_env.py:594` |
| **Timeout (truncated/time_out)** |  |  |  |
| Episode length timeout | `time_out` | `episode_length_buf >= max_episode_length - 1` | `parkour_env.py:1111` |
| **Grace period** |  |  |  |
| Early termination skip | Apply to failure only | `episode_length_buf < termination_grace_steps (5)` | `parkour_env.py:1132`, `parkour_env_cfg.py:601` |

### 상세 분석

**1. Base contact 조건 (현재 비활성화)**  
- 코드: `# terminated = self._term_base_contact | ... ` (주석 처리됨) — `parkour_env.py:1130`
- 이유: 앞부분에서 계산되지만 최종 terminated에 포함되지 않음
- 검증: `grep -n "_term_base_contact" parkour_env.py` 확인 결과, base contact는 logging에만 사용 (라인 1213-1214)

**2. Goal-reached 조건 (즉시 success reset)**  
- `_term_goal_reached`는 `_update_goals()` 매 스텝마다 계산 — `parkour_env.py:580-594`
- 조건: `self._reach_goal_timer > hold_time_steps AND current_goal_idx >= num_goals - 1`
- hold_time_steps = int(cfg.reach_goal_delay / step_dt) = int(0.1 / 0.02) = 5 steps — `parkour_env_cfg.py:598`
- 동작: 마지막 goal 도달 후 0.1초 동안 hold하면 `_term_goal_reached = True` → 다음 스텝에서 즉시 reset
- **예시**: env 0가 마지막 goal 근처 도달 → timer 증가 → 5 스텝 후 _term_goal_reached 플래그 → _get_dones()에서 terminated에 포함 → reset_idx() 호출

**3. Grace period (early termination 방지)**  
- Failure conditions (tilt, low_height)에만 적용
- Goal success는 grace 영향 없음 (항상 종료)
- 계산: `terminated = (terminated & ~grace) | self._term_goal_reached` — `parkour_env.py:1133`

**4. Return 값**  
- `terminated`: failure + success (goal reached) + time_out이 아님
- `time_out`: timeout only (truncated)
- 합치면: `dones = (terminated | time_out)`

---

## 2. max_episode_length / episode_length_s 수치

### 설정값

**Config파일**: `parkour_env_cfg.py:383-384`

```python
episode_length_s: float = 20.0
decimation: int = 4
```

**Simulation parameters**: `parkour_env_cfg.py:409`

```python
dt: float = 1 / 200  # 200 Hz physics
```

### 계산

**공식**: `max_episode_length = ceil(episode_length_s / (sim.dt * decimation))`  
**파일:라인**: `source/isaaclab/isaaclab/envs/direct_rl_env.py:286`

```python
def max_episode_length(self):
    return math.ceil(self.max_episode_length_s / (self.cfg.sim.dt * self.cfg.decimation))
```

**계산식**:
- step_dt = sim.dt × decimation = (1/200) × 4 = 0.02 s
- max_episode_length = ceil(20.0 / 0.02) = ceil(1000) = **1000 steps**
- max_episode_length_s = **20.0 seconds**

### Timeout 조건

**파일:라인**: `parkour_env.py:1111`

```python
time_out = self.episode_length_buf >= self.max_episode_length - 1
```

즉, `episode_length_buf >= 999`일 때 timeout 발생 → 실제 에피소드는 최대 1000 스텝(정확히 20.0초)

---

## 3. _reset_idx() Flow Trace

**파일:라인**: `parkour_env.py:1136-1237`

### Reset Flow (순서대로)

| 단계 | 동작 | 파일:라인 |
|------|------|---------|
| **Phase 1: Robot Reset** | `self._robot.reset(env_ids)` + `super()._reset_idx(env_ids)` | 1140-1141 |
| **Phase 1.5: Episode Offset** | 모든 env reset 시 episode_length_buf 랜덤화 (spike 방지) | 1142-1144 |
| **Phase 2: State Buffers** | actions, previous_actions, processed_actions, applied_torque, goal tracking 초기화 | 1146-1167 |
| **Phase 3: Curriculum Update** | `_update_terrain_curriculum(env_ids)` 호출 (if cfg.terrain_curriculum) | 1173-1176 |
| **Phase 4: Goal Reinit** | `_init_env_goals(env_ids)` 호출 (terrain level 변경 후) | 1179 |
| **Phase 5: Robot Repositioning** | root state, joint pos/vel, +0.05m physics settling | 1181-1191 |
| **Phase 6: Command Resample** | `_resample_commands(env_ids)` | 1195-1196 |
| **Phase 7: Logging** | Episode reward sum, termination cause count, curriculum level 기록 | 1199-1237 |

### 상세 흐름

**Step 2.1-2.7**: Buffer 초기화 (총 7개 buffer 초기화)  
- `_actions[env_ids] = 0.0` — `1147`
- `_previous_actions[env_ids] = 0.0` — `1148`
- `_current_goal_idx[env_ids] = 0` — `1150`
- `_last_contacts[env_ids] = False` — `1151`
- `_term_goal_reached[env_ids] = False` — `1152`
- `_processed_actions[env_ids] = 0.0` — `1155`
- `_reach_goal_timer[env_ids] = 0` — `1163`
- (기타 goal tracking buffers 초기화) — `1164-1167`

**Step 3**: Curriculum 호출

```python
if self.cfg.terrain_curriculum:
    self._update_terrain_curriculum(env_ids)
    self._env_class[env_ids] = self._col_to_class[self._terrain_types[env_ids]]
```

**파일:라인**: `1173-1176`

**Step 5**: Robot 재배치

```python
joint_pos = self._robot.data.default_joint_pos[env_ids]
joint_pos += (torch.rand_like(joint_pos) * 0.1 - 0.05)  # ±0.05 random
joint_vel = self._robot.data.default_joint_vel[env_ids]
default_root_state = self._robot.data.default_root_state[env_ids]
default_root_state[:, :3] += self._terrain.env_origins[env_ids]
default_root_state[:, 2] += 0.05  # physics settling buffer (NOT free-fall)
```

**파일:라인**: `1182-1188`

- Root position: terrain origin + [0, 0, 0.05m]
- Root velocity: default (보통 0)
- Joint position: default ± 0.05 (randomization)
- Joint velocity: default (보통 0)

**Step 6**: Command resample

```python
self._time_since_command_resample[env_ids] = 0
self._resample_commands(env_ids)
```

**파일:라인**: `1195-1196`

**Step 7**: Logging

- Episode reward sums (모든 reward scale에 대해) — `1200-1203`
- Termination causes (base_contact, tilt, low_height, goal_reached) — `1206-1224`
- Mean episode length at reset — `1226`
- Mean terrain curriculum level — `1227`
- Per-class terrain level — `1228-1237`

---

## 4. Curriculum Advance Trigger

### 함수

**파일:라인**: `parkour_env.py:1239-1286`

```python
def _update_terrain_curriculum(self, env_ids: torch.Tensor):
    """Game-inspired terrain curriculum: advance on success, regress on failure."""
```

### 핵심 로직 (라인 1265-1282)

```python
# Distance traveled from spawn origin during episode
dis_to_origin = torch.norm(
    self._robot.data.root_link_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2],
    dim=1,
)
# Expected travel based on commanded velocity and episode length
expected_dist = self._commands[env_ids, 0].abs() * self.max_episode_length_s
move_up = dis_to_origin > 0.8 * expected_dist
move_down = dis_to_origin < 0.4 * expected_dist

max_level = self.cfg.terrain.terrain_generator.num_rows - 1
self._terrain_levels[env_ids] += move_up.long() - move_down.long()
# Wrap top level to random (inclusive of max_level), clip bottom at 0
self._terrain_levels[env_ids] = torch.where(
    self._terrain_levels[env_ids] >= max_level,
    torch.randint_like(self._terrain_levels[env_ids], max_level + 1),
    torch.clamp(self._terrain_levels[env_ids], 0),
)
```

### Curriculum Advance 규칙

| 조건 | 트리거 | 설명 | 파일:라인 |
|------|--------|------|---------|
| **Level Up** | `dis_to_origin > 0.8 × expected_dist` | 거리 > 80% of expected | 1272 |
| **Level Down** | `dis_to_origin < 0.4 × expected_dist` | 거리 < 40% of expected | 1273 |
| **No Change** | otherwise | 진행도 40~80% range | - |

### 상세 분석

**메트릭**: Distance traveled (진짜 거리, NOT episode_length 기반)

- `dis_to_origin`: 스폰 지점으로부터 현재 XY 거리
- `expected_dist = commanded_velocity × max_episode_length_s`
  - 예: v_cmd = 1.0 m/s → expected_dist = 1.0 × 20.0 = 20.0 m
- **Threshold**:
  - Move up: > 16.0 m (80% of 20m)
  - Move down: < 8.0 m (40% of 20m)
  - Dead zone: 8~16 m (no change)

**Wraparound**:
- 최대 레벨에 도달하면 0으로 리셋 (난수 선택) — `1278-1280`
- 최소는 0으로 클램프

**호출 타이밍**:
- `_reset_idx()` 내에서 호출 (환경 리셋 전) — `1173-1176`
- 스킵 조건: `init_done = False` (첫 리셋 시) — `1242-1243`

### 리셋 생략 흐름

**검증된 부재**: keyboard override 메커니즘이 있음. `_change_terrain_for_viewer`가 `_skip_curriculum` 플래그 설정 → curriculum 스킵 — `parkour_env.py:1248-1258`

---

## 5. PPO Timeout vs Termination 분리 (Bootstrap Value 처리)

### _get_dones() 반환값 분리

**파일:라인**: `parkour_env.py:1110, 1134`

```python
def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
    ...
    time_out = self.episode_length_buf >= self.max_episode_length - 1
    ...
    terminated = (terminated & ~grace) | self._term_goal_reached
    return terminated, time_out
```

**반환**:
1. **terminated** (Tuple[0]): failure + goal success conditions
2. **time_out** (Tuple[1]): timeout only

### DirectRLEnv → RslRlVecEnvWrapper 변환

**DirectRLEnv.step() return**: `parkour_env.py`가 상속받는 부모의 step() 실행  
**파일:라인**: `source/isaaclab/isaaclab/envs/direct_rl_env.py:420`

```python
return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras
```

**5개 반환값**:
1. obs_buf
2. reward_buf
3. **reset_terminated** (← _get_dones()[0])
4. **reset_time_outs** (← _get_dones()[1])
5. extras

### RslRlVecEnvWrapper 변환

**파일:라인**: `source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py:166-172`

```python
obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
# compute dones for compatibility with RSL-RL
dones = (terminated | truncated).to(dtype=torch.long)
# move time out information to the extras dict
# this is only needed for infinite horizon tasks
if not self.unwrapped.cfg.is_finite_horizon:
    extras["time_outs"] = truncated
# return the step information
return TensorDict(obs_dict, batch_size=[self.num_envs]), rew, dones, extras
```

**변환 로직**:
- DirectRLEnv 반환 5값 받음
- `terminated = reset_terminated` (failure + goal success)
- `truncated = reset_time_outs` (timeout only)
- Combined `dones = terminated | truncated` (all done conditions)
- **Key**: `extras["time_outs"] = truncated` (runner에서 bootstrap에 사용)

### PPO 알고리즘의 Bootstrap 처리

**파일:라인**: `rsl_rl/rsl_rl/algorithms/ppo.py:144-172`

```python
def process_env_step(
    self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
) -> None:
    ...
    self.transition.rewards = rewards.clone()
    self.transition.dones = dones
    ...
    # Bootstrapping on time outs
    if "time_outs" in extras:
        self.transition.rewards += self.gamma * torch.squeeze(
            self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device), 1
        )
    ...
```

**Bootstrap 로직**:

| 시나리오 | dones | time_outs in extras | Bootstrap 적용? |
|---------|-------|-------------------|----------------|
| Failure/Goal success | 1 | 0 | ❌ NO (reward 그대로) |
| Timeout (truncated) | 1 | 1 | ✅ YES (reward += γ × V(s') × 1) |
| Continue | 0 | 0 | ❌ NO (not done) |

**효과**:
- **Failure/Success**: `time_outs = 0` → bootstrap 없음 → value = 0 (에피소드 종료)
- **Timeout**: `time_outs = 1` → bootstrap 적용 → value = V(next_state) (계속 진행으로 봄)
- PPO의 가치 함수 학습이 정확한 구분에 의존

### Parkour A의 on_policy_runner_parkour 통합

**파일:라인**: `rsl_rl/rsl_rl/runners/on_policy_runner_parkour.py:100-104`

```python
obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
self.alg.process_env_step(obs, rewards, dones, extras)
```

**흐름 정리**:
1. env.step() → (obs, rewards, dones=terminated|time_outs, extras={..., time_outs=time_outs})
2. process_env_step() 호출 → extras["time_outs"] 체크 → bootstrap 적용

---

## 요약: A의 Reset/Term/Curriculum 구조

| 항목 | 설정값 | 파일:라인 |
|------|-------|---------|
| **Episode Length** | 20.0 s / 1000 steps | cfg:383-384, direct_rl_env.py:286 |
| **Failure Conditions** | tilt (0.99), low_height (-0.2m) | parkour_env.py:1120, 1123 |
| **Goal Success** | hold 0.1s at last waypoint → immediate reset | parkour_env.py:594 |
| **Timeout** | ≥ 999 steps | parkour_env.py:1111 |
| **Grace Period** | 5 steps (failure only, not goal) | parkour_env_cfg.py:601 |
| **Curriculum Metric** | Distance traveled % of expected | parkour_env.py:1271-1273 |
| **Level Up Trigger** | > 80% expected distance | parkour_env.py:1272 |
| **Level Down Trigger** | < 40% expected distance | parkour_env.py:1273 |
| **Bootstrap Separation** | ✅ YES — time_outs in extras → PPO bootstrap | vecenv_wrapper.py:172, ppo.py:165-168 |

### 중요 설계 결정

1. **Base contact 조건이 주석 처리됨** — 현재 failure가 아님 (`parkour_env.py:1130`)
2. **Goal success는 즉시 reset** — grace period 미적용, value bootstrap 0 (episode truly ends)
3. **Curriculum은 거리 기반, 에피소드 길이 기반 아님** — 빠른 학습이 자동으로 난이도 상승
4. **Timeout → bootstrap 활성화** — 20초 제한은 유연한 학습 경계 (value 추정값 전파)
5. **Parkour A는 PPO bootstrap 메커니즘 완전 지원** — time_outs 분리 구현됨

---

**작성일**: 2026-05-20  
**검증 방식**: grep -rn, Read 도구로 파일:라인 직접 인용  
**부재 검증**: base_contact condition 제외 모든 항목 코드에서 확인

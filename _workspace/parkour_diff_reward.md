# Parkour Reward 비교: A (Direct RL) vs B (Manager-based)

- **A** = `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/` — Direct RL, 학습 초반 안 됨
- **B** = `Isaaclab_Parkour/` — Manager-based, 학습 잘 됨 (extreme_parkour 포팅 원본)

대상 파일:
- A: `parkour_env.py::_get_rewards()` (L960–1107), `parkour_env_cfg.py::reward_scales` (L560–576)
- B: `parkour_isaaclab/envs/mdp/rewards.py`, `parkour_isaaclab/managers/parkour_reward_manager.py`, `extreme_parkour_task/config/go2/parkour_mdp_cfg.py::TeacherRewardsCfg`

---

## 1. Reward 항 매핑 (14항 — 항 집합 자체는 일치)

| Reward 항 | A 이름 | B 이름 | A scale | B weight | scale 일치? |
|---|---|---|---|---|---|
| 목표 속도 추종 | `tracking_goal_vel` | `reward_tracking_goal_vel` | **+1.5** | +1.5 | ✅ |
| 목표 yaw 추종 | `tracking_yaw` | `reward_tracking_yaw` | **+0.5** | +0.5 | ✅ |
| 수직 속도 penalty | `lin_vel_z_l2` | `reward_lin_vel_z` | **-1.0** | -1.0 | ✅ (단 비평지 계수 다름, §3) |
| 수평 각속도 penalty | `ang_vel_xy_l2` | `reward_ang_vel_xy` | **-0.05** | -0.05 | ✅ (단 비평지 계수 추가, §3) |
| 자세(기울기) penalty | `orientation_l2` | `reward_orientation` | **-1.0** | -1.0 | ✅ |
| 관절 가속 penalty | `dof_acc_l2` | `reward_dof_acc` | **-2.5e-7** | -2.5e-7 | ✅ |
| 충돌 penalty | `collision` | `reward_collision` | **-10.0** | -10.0 | ✅ (단 body set/history 다름, §2) |
| action rate penalty | `action_rate_l2` | `reward_action_rate` | **-0.05** | **-0.1** | ❌ **A가 B의 1/2** |
| torque 변화 penalty | `delta_torques` | `reward_delta_torques` | **-1.0e-7** | -1.0e-7 | ✅ |
| torque penalty | `torques_l2` | `reward_torques` | **-1e-5** | -1e-5 | ✅ |
| hip 이탈 penalty | `hip_pos` | `reward_hip_pos` | **-0.5** | -0.5 | ✅ |
| 관절 위치 오차 penalty | `dof_error_l2` | `reward_dof_error` | **-0.04** | -0.04 | ✅ |
| 발 stumble penalty | `feet_stumble` | `reward_feet_stumble` | **-1.0** | -1.0 | ✅ |
| 발 edge penalty | `feet_edge` | `reward_feet_edge` | **-1.0** | -1.0 | ✅ |

- A에만 있는 항 / B에만 있는 항: **없음** (14항 1:1 대응).
- B `StudentRewardsCfg`는 collision(weight 0)만 있는 distillation용 — Teacher 학습과 무관.
- scale 차이는 `action_rate`(-0.05 vs -0.1) 단 1개. (A 주석은 -0.1을 "10x error"라 표기했으나 B 원본은 -0.1.)

### 합산 방식 (둘 다 `weight * dt * value`)
- A `parkour_env.py:1102`: `scaled = reward_scales[key] * self.step_dt * value`
- B `parkour_reward_manager.py:31`: `value = func(...) * term_cfg.weight * dt`
- dt 곱 여부 동일. → scale 값은 직접 비교 가능.

---

## 2. ★ 최상위 divergence: total reward 음수 clip 유무

| 항목 | A | B |
|---|---|---|
| total reward 합산 후 처리 | **clip 없음** — `total_reward` 그대로 return (`parkour_env.py:1100–1107`) | **`torch.clip(reward_buf, min=0.)`** (`parkour_reward_manager.py:38`) |

- **B는 매 step 합산 총 reward를 0 이하로 못 내려가게 clip** (legged_gym `only_positive_rewards` 관습).
  → B에서는 서 있거나 비틀거리는 로봇이라도 step reward 최솟값이 0. penalty들은 positive 항을 0까지 깎을 뿐 부호를 못 바꿈.
- **A는 clip이 전혀 없음.** iter 0 평지 정지 로봇은 positive 항(tracking_goal_vel≈0, tracking_yaw만 일부)이 작고 penalty 합(dof_acc/torque/action_rate/collision/hip/dof_error/orientation/lin_vel_z/ang_vel_xy/stumble/delta_torques)이 커서 **total reward가 강하게 음수**가 됨.

이것은 reward scale 튜닝 이슈가 아니라 **reward 집계 구조의 차이**다 (사실 기록).

---

## 3. 동일 reward 항의 수식 line-by-line divergence

### 3-1. `collision` (scale -10.0) — **A가 더 자주/넓게 발화**
| | A (`parkour_env.py:1015–1019`) | B (`rewards.py:215–221`) |
|---|---|---|
| body set | `["base",".*thigh",".*calf",".*hip","Head_upper","Head_lower"]` (`L199–201`) | `["base",".*_calf",".*_thigh"]` (`parkour_mdp_cfg.py:145`) |
| history 사용 | `net_forces_w_history[:, :, ids]` 전체 history에 **`max(dim=1)`** | `net_forces_w_history[:, 0, ids]` — **최신 substep 1개만** |
| 임계 | norm > 0.1 N | norm > 0.1 N |

→ A는 (a) **hip·Head 2종 body 추가 포함**, (b) **history 전 substep 중 max** 사용 → 동일 상황에서 B보다 collision 카운트가 구조적으로 큼. scale -10이라 음수 reward 기여가 B보다 큼.

### 3-2. `lin_vel_z` 비평지 계수 — **계수 값 자체 불일치**
- A (`parkour_env.py:1001`): `lin_vel_z_l2 *= (is_flat + is_non_flat * 0.1)` → 비평지에서 **0.1배**
- B (`rewards.py:144`): `rew[terrain!='parkour_flat'] *= 0.5` → 비평지에서 **0.5배**
→ 비평지 env에서 A의 lin_vel_z penalty가 B의 1/5. (평지에서는 둘 다 1.0배, 동일.)

### 3-3. `ang_vel_xy` 비평지 계수 — **A에만 존재하는 conditional**
- A (`parkour_env.py:1003`): `ang_vel_xy_l2 *= (is_flat + is_non_flat * 0.5)` → 비평지 0.5배
- B (`rewards.py:88–93`): conditional **없음** — `sum(square(root_ang_vel_b[:,:2]))` 단순값
→ A는 비평지에서 ang_vel_xy penalty를 절반으로 깎지만 B는 그대로 적용. (평지 동일.)

### 3-4. `orientation` 비평지 처리 — **일치**
- A (`parkour_env.py:1008`): `orientation_l2 *= is_flat` (비평지 0)
- B (`rewards.py:156`): `rew[terrain!='parkour_flat'] = 0.` (비평지 0) → 동일.

### 3-5. `tracking_goal_vel` — 거의 동등 (미세 divergence)
| | A (`parkour_env.py:977–987`) | B (`rewards.py:169–182`) |
|---|---|---|
| 현재 속도 | `root_lin_vel_w[:, :2]` | `root_vel_w[:, :2]` (= 동일 선속도 xy) |
| command | `abs(self._commands[:,0])` (**abs 추가**) | `command_manager.get_command('base_velocity')[:,0]` (signed) |
| 식 | `min(proj, cmd)/(cmd+1e-5)` | `min(proj, cmd)/(cmd+1e-5)` |
| gating | `where(cmd>1e-3, val, 0)` (**A 추가**) | 없음 |
| clamp(min=0) | 주석 처리됨 (비활성) | 없음 |
→ command range가 항상 양수(lin_vel_x 0.3~0.8)라 `abs`/`where` 차이는 실질 영향 미미. **사실상 동등.**

### 3-6. `tracking_yaw` — 거의 동등
- A (`parkour_env.py:992–995`): `heading_w` 사용, `yaw_diff = target_yaw - heading`, wrap 주석처리, `exp(-abs(yaw_diff))`
- B (`rewards.py:184–194`): quat에서 yaw 직접 atan2, `exp(-abs(target_yaw - yaw))`
→ 둘 다 wrap_to_pi 미적용. yaw 산출 경로만 다름. 식 동등.
→ 참고: A cfg의 `tracking_sigma=0.2`, `yaw_reward_speed_lower/upper`는 **현재 코드에서 미사용 dead config** (yaw 식이 sigma·speed-gating 안 씀).

### 3-7. `feet_stumble` — 동등
- A `L1049–1052` / B `rewards.py:159–167`: 둘 다 `norm(F_xy) > 4.0*abs(F_z)`, 최신 substep 사용. 동등.

### 3-8. `feet_edge` — 동등 + last_contacts 의미 차이
- 식·grid 변환·`terrain_levels>3` gating 동일.
- `contact_filt`의 OR 대상:
  - A (`parkour_env.py:1059–1062`): 최신 substep `contact` OR **직전 env-step** `self._last_contacts` (step 간 버퍼)
  - B (`rewards.py:51–55`): history `[:,0]`(최신) OR `[:,-1]`(같은 step 내 가장 오래된 substep)
→ "이전 접촉"의 시간 범위가 다름. 단 feet_edge는 `terrain_level>3`에서만 발화 → 학습 초반 평지/저레벨에선 양쪽 0, 영향 거의 없음.

---

## 4. 학습 초반 신호 관점 (iter 0 평지 정지 로봇)

- positive gradient를 만드는 항: `tracking_goal_vel`, `tracking_yaw` 2개뿐. 정지 시 `tracking_goal_vel`≈0 → 실질 `tracking_yaw`(weight 0.5, `exp(-|yaw err|)`)만 살아있음.
- A에서 positive 항이 "죽어있지"는 않음 — `tracking_yaw`는 정지해도 값 발생. 단 weight 0.5로 작음.
- 나머지 12항은 전부 penalty(음수). 평지에서 A/B penalty 식은 §3 기준 거의 동일하되:
  - A `collision`: hip·Head body 추가 + history-max → **A가 더 큰 음수 기여** (§3-1)
  - A `action_rate` scale -0.05로 B(-0.1)보다 절반 → 이 항만 A가 덜 아픔
- **결정적 차이**: §2 — B는 step reward를 `clip(min=0)` 하므로 정지 로봇의 net reward 하한이 0. A는 clip 없어 net reward가 음수.

---

## 5. 학습 초반 영향 가설 (우선순위)

> 아래는 "사실로 기록된 A/B 차이"에 근거한 **가설**이며 단정이 아님. RCA ranking이 아니라 영향 가능성 순 정리.

**[가설 1 — 가장 유력] B의 `clip(min=0)` 부재 (§2).**
B는 매 step total reward를 0 이하로 못 가게 막아, 정지/탐색 단계 로봇이 항상 reward ≥ 0을 받음 → 에피소드를 길게 끌수록 손해가 없고 positive 항(전진/yaw)만이 차이를 만듦. A는 clip이 없어 iter 0 정지 로봇의 step reward가 음수 → "에피소드를 빨리 끝내는(넘어지기·충돌) 것이 누적 음수 reward를 줄인다"는 early-termination local optimum이 형성될 수 있음. 학습 초반 정체와 직접적으로 연결되는 구조적 차이. (검증 필요: A의 termination penalty 유무 — worker-4 영역.)

**[가설 2] A `collision` penalty의 과발화 (§3-1).**
A는 history 전 substep을 `max`로 집계 → 동일 step에서 B(최신 substep 1개)보다 collision이 구조적으로 더 자주 카운트됨 (이쪽이 더 견고한 근거). 추가로 hip·Head_upper·Head_lower body를 포함하나, Go2 `*_hip` 링크는 정상 보행 시 지면 접촉이 드물고 로봇이 넘어질 때 주로 접촉 → 실제 음수 기여 크기는 접촉 패턴에 의존(가정 아닌 실측 필요). scale -10과 결합 시 학습 초반 음수 reward를 키워 가설 1의 음수 누적을 가속할 수 있음.

**[가설 3 — 영향 작음] `action_rate` scale 불일치 (-0.05 vs -0.1).**
A가 B의 절반. penalty가 약한 방향이라 "학습 안 됨"의 원인 방향과 반대(오히려 A가 유리). 기록 목적.

**[가설 4 — 비평지 한정] `lin_vel_z`(0.1배 vs 0.5배), `ang_vel_xy`(A만 0.5배 conditional) 비평지 계수 불일치 (§3-2, §3-3).**
평지/iter 0에는 영향 없음. 커리큘럼 진행 후 비평지 env에서 A의 자세 안정화 penalty가 B보다 약해짐 → 중반 이후 동작 품질 차이 가능. 초반 정체의 직접 원인은 아님.

**영향 없음으로 분류**: `tracking_goal_vel`/`tracking_yaw`/`feet_stumble`/`feet_edge`/`orientation` 식 차이는 실질 동등하거나 학습 초반(평지·저레벨)엔 비활성.

---

## 부록: 미사용 dead config (A)
- `parkour_env_cfg.py` `tracking_sigma=0.2`, `yaw_reward_speed_lower/upper`, `dragging_velocity_threshold`, `base_height_target` — 현 `_get_rewards()` 코드에서 참조 안 함 (혼동 주의, reward 동작엔 무영향).

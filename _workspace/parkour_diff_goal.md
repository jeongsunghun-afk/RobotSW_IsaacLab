# Parkour A vs B — Goal / Command 시스템 비교

- **A** = `parkour` (Direct RL) — `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`
- **B** = `Isaaclab_Parkour` (Manager-based, extreme_parkour 포팅, 학습 성공) — `Isaaclab_Parkour/parkour_isaaclab/`
- 범위: goal waypoint / velocity command 의 **생성·갱신·스케일 로직**. observation 자체 구성(차원/순서)은 제외.
- 모든 검증되지 않은 인과 추론은 "가설"로 명시. RCA ranking 아님.

---

## 0. 시스템 구조 개요

| 항목 | A (Direct RL) | B (Manager-based) |
|---|---|---|
| Goal waypoint 로직 | `parkour_env.py` 내부 메서드 (`_init_env_goals`, `_update_goals`, `_gather_cur_goals`) | 별도 term `ParkourEvent` (`mdp/parkours/parkour_event.py`) |
| Velocity command 로직 | `parkour_env.py::_resample_commands` (`self._commands`, 3-vec) | 별도 term `UniformParkourCommand` (`mdp/parkour_commands/uniform_parkour_command.py`) |
| Goal vs velocity command | 두 시스템 분리 (goal=`_yaw_diff`, command=`_commands`) | 두 시스템 분리 (goal=`ParkourEvent.target_yaw`, command=`base_velocity`) |

두 코드 모두 "goal waypoint(방향 정보) + velocity command(전진 속도)" 의 **2-트랙 구조**로 동일. 차이는 각 트랙의 갱신 주기·wrapping·command 생성 방식에 있다.

---

## 1. Goal / Waypoint 정의 방식

| 항목 | A | B |
|---|---|---|
| Goal 출처 | terrain 함수가 `PARKOUR_GOALS_REGISTRY` 에 append → `_build_terrain_goals_map`이 **world-frame** 맵 생성 (`parkour_env.py:368-424`) | `terrain_generator.goals` numpy 배열 직접 로드 → **terrain-local frame** (`parkour_event.py:55-56`) |
| Goal 개수 | `num_goals=8`, `num_future_goal_obs=2` (`parkour_env_cfg.py:594-595`) | `num_goals=8` (terrain cfg), `num_future_goal_obs=2` (`parkour_events_cfg.py:50`) → 동일 |
| env별 goal 배정 | `_env_goals[env_ids,:num_goals] = terrain_goals_world[levels, types]` (`parkour_env.py:542`) | `env_goals = terrain_goals[terrain_levels, terrain_types]` (`parkour_event.py:60-62`) |
| future goal 패딩 | 마지막 goal 복제 (`parkour_env.py:557-560`) | 마지막 goal 복제 (`parkour_event.py:61-62`) → 동일 |
| **좌표 frame** | **world frame** (goal·robot 모두 world) | **terrain-local frame** (goal·robot 모두 `- env_origins`) |
| Fallback 경로 | registry size mismatch 시 `_terrain_goals_world=None` → **직선 goal(+x, goal_distance=1.0 간격)로 silent fallback** (`parkour_env.py:544-554`, 경고 출력) | 없음 (terrain_generator.goals 직접 사용) |

**핵심 차이 (1):** A는 goal을 별도 registry → world-frame 맵으로 재구성하는 **추가 변환 단계**가 있고, 크기 불일치 시 **직선 goal로 조용히 fallback**한다. 이 fallback이 발동하면 모든 terrain이 단순 직선 경로로 바뀌어 학습 신호가 무의미해진다. B는 terrain generator 산출물을 직접 쓰므로 이 위험이 없다. (좌표 frame 자체는 A/B 모두 내부적으로 일관 — goal·robot 같은 frame.)

---

## 2. Goal 도달 판정 / Goal index 전진 조건

| 항목 | A (`_update_goals`, `parkour_env.py:580-620`) | B (`_update_command`, `parkour_event.py:106-135`) |
|---|---|---|
| 도달 거리 임계값 | `next_goal_threshold = 0.2` m (`cfg:597`) | `next_goal_threshold = 0.2` m (`parkour_events_cfg.py:54`) → 동일 |
| 도달 시 동작 | `dist < threshold` → `_reach_goal_timer += 1` | `dist < threshold` → `reach_goal_timer += 1` → 동일 |
| 전진 hold 조건 | `_reach_goal_timer > int(reach_goal_delay/step_dt)` = `> 5` step (delay 0.1s/0.02s) | `reach_goal_timer > reach_goal_delay/step_dt` = `> 5.0` (delay 0.1s) → 동일 |
| index 전진 | `_current_goal_idx += 1`, **`max=num_goals-1`로 clamp** (`:596-599`) | `cur_goal_idx += 1`, **clamp 없음** (`:111`) |
| 마지막 goal 처리 | `_term_goal_reached` 플래그 → episode 성공 종료 (`:594`) | clamp 없이 증가, `num_future_goal_obs` 패딩이 OOB 방어, 종료는 termination term이 담당 |
| 거리 계산 frame | world: `base_xy - cur_goals_xy` (`:608`) | local: `(root_pos_w - env_origins) - cur_goals_xy` (`:113`) |

**핵심 차이 (2):** 도달 판정·전진 조건의 수치(0.2 m, 5 step)는 **완전 동일**. 차이는 index 전진 후 처리뿐 — A는 clamp + 성공종료 플래그, B는 unclamped + 패딩 의존. 학습 초반 영향은 작다고 판단.

---

## 3. Goal 방향(yaw) 계산 — wrapping 검증

이전 보고서 `parkour_flat_run_diagnostic.md` 가 "A에서 yaw wrapping이 비활성/주석처리" 의심을 제기 → **현재 코드 기준 재검증 결과:**

### A — `parkour_env.py`
- `_target_yaw = atan2(target_vec_norm[:,1], target_vec_norm[:,0])` (`:620`) → goal 절대방향, `[-π,π]`.
- obs용 delta: `yaw_raw = _target_yaw - heading_w` → `_yaw_diff = atan2(sin(yaw_raw), cos(yaw_raw))` (`:655-656`) → **`[-π,π]`로 wrapping 적용됨 (활성)**.
- 바로 아래 unwrapped 버전 `# self._yaw_diff = yaw_raw` 은 **주석처리(비활성)** (`:662`).
- → **이전 보고서 의심은 현재 코드에서 사실이 아님. A는 delta_yaw를 wrapping 한다.**

### B — `observations.py:64`
- `delta_yaw = parkour_event.target_yaw - wrap_to_pi(yaw)`.
- `target_yaw` 은 atan2 결과(`[-π,π]`), `wrap_to_pi(yaw)` 는 robot heading wrapping(`[-π,π]`).
- **둘의 차(差)에는 wrapping 미적용** → `delta_yaw` 범위 `[-2π, 2π]`.

**핵심 차이 (3) — wrapping 방향이 이전 의심과 반대:**
- **A: delta_yaw를 `[-π,π]`로 wrapping 함.**
- **B: delta_yaw를 wrapping 하지 않음 (`[-2π,2π]`).**
- extreme_parkour 원본도 B와 동일(delta 미wrapping). 즉 A쪽이 원본에서 벗어나 있다.
- 가설: 정상 주행(heading≈goal방향)에서는 `|delta|<π`라 차이 미미. 그러나 학습 **초반** robot이 goal 반대로 향하거나 크게 회전할 때 `|delta|>π` 영역에서 A(wrapped)와 B(unwrapped)는 **다른 부호/크기**의 신호를 정책에 준다. reward나 obs clip 범위가 원본(unwrapped) 기준으로 튜닝돼 있다면 A의 wrapped 신호와 불일치 가능 — **가설, 미검증**.

---

## 4. delta_yaw / scan 갱신 주기 (스케일 전 처리)

| 항목 | A | B |
|---|---|---|
| goal 위치/`_target_yaw` 갱신 | `_update_goals()` 매 step 호출 (`parkour_env.py:627`) | `_update_command()` 매 step 호출 (`parkour_manager.compute`, rl_env:117) |
| **정책에 들어가는 `_yaw_diff` 갱신** | **5 step마다만 refresh** (`do_global_refresh = common_step_counter % 5 == 0`, `parkour_env.py:642`); reset 직후 env만 예외 patch | **매 step refresh** (`ExtremeParkourObservations` 매 step 계산) |

**핵심 차이 (4):** A는 goal의 절대방향(`_target_yaw`)은 매 step 갱신하지만, **정책이 실제로 보는 `_yaw_diff`(delta_yaw obs)는 5 step(=100 ms @ 50 Hz)마다만 갱신**하고 나머지 4 step은 캐시값을 재사용한다 ("10 Hz hardware constraint" 의도적 설계). B는 매 step 신선한 delta_yaw를 정책에 준다.
- 가설: 학습 초반 빠른 자세/방향 변화 구간에서 A 정책은 최대 ~80 ms 지연된 heading-error 신호로 행동을 결정 → goal 방향 추종 신호의 시간 정합성이 B보다 낮음. obs 자체 구성이 아니라 "goal 신호가 정책에 전달되는 갱신 주기" 차이이므로 본 비교 항목에 해당. **가설, 학습영향 미검증.**

---

## 5. Target velocity / command 생성·스케일 로직

### A — `_resample_commands` (`parkour_env.py:1284-1294`)
- `_commands` 3-vec를 **순수 uniform 랜덤 샘플**:
  - `vx ∈ [0.3, 1.0]` (`command_cfg.lin_vel_x_range`)
  - `vy ∈ [0.0, 0.0]` (항상 0)
  - `wz ∈ [0.0, 0.0]` (**항상 0 — ang_vel command 없음**)
- goal과 **무관**하게 샘플링. resample 주기 `resampling_time_s=6.0` s → 300 step (`parkour_env.py:111`).
- 정책 obs에는 `_commands[:,0:1]`(전진속도)만 투입 (`parkour_env.py:857`).
- dead-zone(소량 command 0 처리) **없음**.

### B — `UniformParkourCommand` (`uniform_parkour_command.py` + `parkour_mdp_cfg.py:22-34`)
- `vel_command_b[:,0]` = `lin_vel_x ∈ [0.3, 0.8]` uniform.
- `vel_command_b[:,2]` = **heading PD로 계산**: `clip(wrap_to_pi(heading_target - heading_w) * heading_control_stiffness, -1, 1)` (`heading_control_stiffness=0.8`).
  - `heading_target` 은 uniform `[-1.6, 1.6]` 샘플 (goal과 무관, 랜덤 heading).
- `small_commands_to_zero=True`: `|vx| ≤ lin_vel_clip(0.2)` 이면 lin_vel command 0 처리; `|wz| ≤ ang_vel_clip(0.4)` 이면 ang_vel command 0 처리 (dead-zone 존재).
- resample 주기 `resampling_time_range=(6.0,6.0)` → 300 step → **A와 동일**.
- 정책 obs에는 `commands[:,0:1]`(전진속도)만 투입 (`observations.py:75`); `vel_command_b[:,2]`(ang_vel)는 reward tracking 용.

**핵심 차이 (5):**
| 세부 | A | B |
|---|---|---|
| 전진속도 범위 | `[0.3, 1.0]` | `[0.3, 0.8]` |
| ang_vel command | **항상 0** (command term 없음) | heading-PD 계산값 (stiffness 0.8, clip ±1) |
| command dead-zone | 없음 | `lin_vel_clip=0.2`, `ang_vel_clip=0.4` |
| resample 주기 | 6.0 s (300 step) | 6.0 s (300 step) — 동일 |

- A는 yaw 회전 command(`wz`)가 **존재하지 않으며**, 회전 유도는 전적으로 `_yaw_diff` obs + yaw reward(`tracking_yaw=0.5`)에만 의존한다. B는 heading-PD `ang_vel` command를 추가로 가지고 있어 회전에 대한 명시적 velocity-tracking 신호가 reward 쪽에 존재한다.
- 가설: A에 yaw command 트랙이 없으므로, goal 방향 추종이 깨지면(특히 wrapping 차이가 겹치는 학습 초반) 정책이 회전을 학습할 신호원이 `_yaw_diff` 단일 채널 + yaw reward로 제한됨. B는 동일 상황에서 ang_vel command-tracking 신호가 추가로 작동. **가설 — 단, reward/hyperparam scale 단독 튜닝을 수정안으로 제시하지 않음.**

---

## 6. Goal resampling 주기 / episode 시작 시 초기화

| 항목 | A | B |
|---|---|---|
| episode 시작 goal 초기화 | `_reset_idx` → `_init_env_goals(env_ids)` (`parkour_env.py:285`, reset 시 호출) | `parkour_manager.reset` → `ParkourEvent._resample_command(env_ids)` (`parkour_event.py:137-185`) |
| goal index reset | `_current_goal_idx`, `_reach_goal_timer` reset 시 0으로 (reset_idx 내) | `cur_goal_idx[env_ids]=0`, `reach_goal_timer[env_ids]=0` (`parkour_event.py:181-182`) |
| reset 시 env_goals 재배정 | terrain level/type 기반 재조회 | terrain level/type 기반 재조회 + **terrain curriculum(level±1)도 동시 수행** (worker-2 영역) |
| velocity command resample | 시간기반 6 s 주기 + reset 시 (`_resample_commands`, `parkour_env.py:1192`) | 시간기반 6 s 주기 + reset 시 (CommandManager 표준) |

**판정:** goal/command 초기화·resampling 주기는 A/B **실질 동등**. (B는 reset 시 terrain curriculum을 같은 함수에서 처리하지만 이는 worker-2 범위.)

---

## 7. 요약 — A/B 차이 정리

| # | 차이 | A | B | 동일/차이 |
|---|---|---|---|---|
| 1 | goal 좌표 | world-frame 재구성 + 직선 silent fallback | terrain-local 직접 로드 | **차이** |
| 2 | 도달 판정(0.2 m)·hold(5 step) | 동일 | 동일 | 동일 |
| 3 | delta_yaw wrapping | **wrapping 함 `[-π,π]`** | **wrapping 안 함 `[-2π,2π]`** | **차이 (원본=B)** |
| 4 | delta_yaw obs 갱신 주기 | **5 step마다** | 매 step | **차이** |
| 5 | velocity command 전진범위 | `[0.3,1.0]` | `[0.3,0.8]` | **차이** |
| 6 | ang_vel command 트랙 | **없음(항상 0)** | heading-PD 계산 | **차이** |
| 7 | command dead-zone | 없음 | `lin_vel_clip/ang_vel_clip` | **차이** |
| 8 | command resample 주기 | 6 s | 6 s | 동일 |
| 9 | goal 개수/future | 8 / 2 | 8 / 2 | 동일 |

---

## 8. 학습 초반 영향 가설 (우선순위)

> 아래는 모두 **가설**이며 RCA 단정·ranking 아님. reward/hyperparam scale 단독 튜닝을 수정안으로 제시하지 않음.

**[가설 H1 — 우선 검토] delta_yaw obs 갱신 주기 (차이 #4)**
A는 정책이 보는 `_yaw_diff`(goal heading-error)를 5 step(~100 ms)마다만 갱신한다. 학습 초반 robot이 빠르게 방향을 트는 구간에서 정책은 최대 ~80 ms 지연된 goal 방향 신호로 행동을 정한다. B는 매 step 신선한 신호 제공. goal 추종 신호의 시간 정합성 저하 → 초반 방향 학습 지연 가능. **검증법:** A의 `_yaw_diff` 갱신을 매 step으로 바꾼 실험과 현행 비교.

**[가설 H2 — 우선 검토] delta_yaw wrapping 불일치 (차이 #3)**
A는 delta_yaw를 `[-π,π]`로 wrapping, B(=extreme_parkour 원본)는 미wrapping(`[-2π,2π]`). 학습 초반 robot 방향이 goal과 크게 어긋날 때(`|delta|>π`) 두 코드가 부호·크기가 다른 신호를 정책에 준다. 원본 기준으로 튜닝된 reward/obs 처리와 A의 wrapped 신호가 불일치할 가능성. **검증법:** A의 wrapping을 원본과 동일(미wrapping)하게 두고 비교.

**[가설 H3 — 보조] goal world-frame 변환 / silent fallback (차이 #1)**
A는 `_build_terrain_goals_map`의 registry size mismatch 시 경고만 출력하고 직선 goal로 fallback한다. fallback이 실제 발동하면 모든 terrain이 직선 경로가 되어 학습 신호가 무의미. **검증법:** 학습 로그에서 `[Parkour] Goals map ready` 정상 출력 vs fallback 경고 확인 (간단·우선 확인 권장).

**[가설 H4 — 보조] ang_vel command 트랙 부재 (차이 #6)**
A에는 yaw 회전 velocity command가 없어 회전 유도가 `_yaw_diff` obs + yaw reward 단일 경로에 의존. B는 heading-PD ang_vel command-tracking 신호가 추가로 존재. H1/H2로 goal 방향 신호가 약해진 학습 초반에 A는 회전 학습 신호원이 더 적다. (구조적 차이 기록 — 단독 수정안 제시 아님.)

**참고:** 도달 판정 임계값(0.2 m)·hold(5 step)·goal 개수·resample 주기(6 s)는 A/B 동일하므로 학습 초반 원인 후보에서 제외.

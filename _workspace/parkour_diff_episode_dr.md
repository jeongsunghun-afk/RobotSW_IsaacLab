# Parkour A vs B — Episode / Termination / Reset / Events(DR) 비교

- **A** = `parkour` (Direct RL, 학습 초반 학습 안 됨)
  `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`
- **B** = `Isaaclab_Parkour` (Manager-based, 학습 잘 됨, extreme_parkour 포팅 원본)
  `Isaaclab_Parkour/`
- 본 문서는 worker-4 담당 범위(종료/리셋/이벤트)만 다룬다. goal/command 세부, terrain/curriculum 세부, reward weight, PPO 하이퍼파라미터는 각각 worker-3 / worker-2 / worker-1 / worker-5 담당.
- 검증 안 된 추론은 "가설"로 표기. 원인 단정/RCA ranking 없음.

---

## 1. 종료 조건 (Termination) line-by-line

| 항목 | A (`parkour_env.py:_get_dones` 1109-1132) | B (`mdp/terminations.py:terminate_episode` 25-43) | 차이 |
|------|-------------------------------------------|---------------------------------------------------|------|
| **base contact** | `_term_base_contact`: base body net contact force history max `> 5.0 N` → terminate (1114-1117) | **없음** | **A에만 존재.** B는 base/calf/thigh 접촉을 `reward_collision`(weight −10, `parkour_mdp_cfg.py:141-147`)로 *벌점만* 부여, 종료시키지 않음 |
| **tilt / orientation** | `_term_tilt`: `sum(projected_gravity_b[:, :2]²) > 0.99` (1119) — 두 축 결합, ≈84° | `roll/pitch`: `|wrap_to_pi(roll)| > 1.5` **OR** `|pitch| > 1.5` (rad, ≈85.9°), 축 독립 (31-33) | A는 roll·pitch 결합 → 두 축 동시 기울임 시 B보다 먼저 발동. 임계 자체도 A가 약간 타이트 |
| **low height** | `_term_low_height`: `root_link_pos_w[:,2] < cfg.termination_height` = **−0.2** (1122, cfg:600) | `root_state_w[:,2] < −0.25` (37) | A가 0.05m 더 보수적(일찍 종료) |
| **goal reached** | `_term_goal_reached` → **terminated=True** (1131) | `reach_goal_cutoff = cur_goal_idx >= num_goals` → **time_out_buf 에 OR** (36-38) | **bootstrap 처리 정반대** (§5 참조) |
| **time out** | `episode_length_buf >= max_episode_length − 1` (1110) | `episode_length_buf >= max_episode_length` (34) | 1스텝 차이(무시 가능) |
| **grace period** | `episode_length_buf < cfg.termination_grace_steps`(=5) 동안 실패 종료 무시 (1130-1131) | **없음** | A가 spawn 직후 5 policy-step 동안 base_contact/tilt/low_height 종료를 면제. goal_reached 는 grace 적용 안 함 |

> A 종료 = `(base_contact | tilt | low_height) & ~grace | goal_reached`
> B 종료 = `time_out | reach_goal | roll | pitch | height` (모두 1개 `DoneTerm`)

**핵심 사실**: A는 B에 없는 **base contact 종료 조건**을 추가로 가진다. B에서 동일한 접촉 상황은 종료가 아니라 보상 벌점으로만 처리된다.

---

## 2. Episode length / step 수

| 항목 | A | B | 차이 |
|------|---|---|------|
| `episode_length_s` | 20.0 (`parkour_env_cfg.py:383`) | 20.0 (`parkour_teacher_cfg.py:50`) | 동일 |
| `decimation` | 4 (cfg:384) | 4 (teacher_cfg:49) | 동일 |
| sim `dt` | 1/200 = 0.005 (cfg:409) | 0.005 (teacher_cfg:52) | 동일 |
| policy step dt | 0.02 s | 0.02 s | 동일 |
| **max policy steps/episode** | 1000 | 1000 | 동일 |

→ Episode 길이 구조 자체는 동일. 차이는 "episode가 **얼마나 일찍 끝나는가**"(§1, §3)에 있음.

---

## 3. 학습 초반 reset 분포 / episode 조기 종료 구조

- A는 `_get_dones`에서 **base_contact(>5N)** 를 추가 종료로 둔다. 학습 초반 미숙한 policy는 base/몸통이 지면·장애물에 빈번히 접촉 → A에서는 episode가 수~수십 step 만에 종료될 수 있음. B는 같은 접촉을 종료시키지 않으므로(보상 벌점만) episode가 더 오래 유지됨.
- A의 grace period(5 step)는 spawn 직후 물리 안정화 구간만 면제하므로, grace 종료 후의 빈번한 base 접촉 종료는 그대로 발생.
- **가설(미검증)**: A의 base_contact 종료 + 더 보수적인 low_height(−0.2) 조합으로 학습 초반 episode가 B 대비 구조적으로 짧게 끝나, 한 rollout당 유효 transition·goal 진행 경험이 적어질 수 있음. 정량 확인은 `Episode_Length/mean_at_reset`, `Episode_Termination/cause_base_contact` 로그(이미 A에 기록됨, 1209-1222)로 가능.

---

## 4. Reset 시 초기 자세 / 위치 / 속도 분포

| 항목 | A (`_reset_idx` 1134-1192) | B (`mdp/events.py` reset-mode terms) | 차이 |
|------|-----------------------------|---------------------------------------|------|
| root 위치 | `default_root_state[:,:3] + terrain.env_origins`, `z += 0.05` (1182-1184) | `reset_root_state`: `default + env_origins − (size[1]+offset, 0, 0)`, offset=3.0 (`events.py:47-61`) | A는 terrain origin 그대로, B는 x축으로 뒤로 밀어 spawn (goal-0 대비 시작점 차이 — 세부는 worker-3 영역). z는 A만 +0.05m settling 버퍼 |
| root 자세(쿼터니언) | `default_root_state[:,3:7]` (노이즈 없음) | `root_states[:,3:7]` (노이즈 없음) | 동일 — 둘 다 root orientation 노이즈 없음 |
| root 속도 | `default_root_state[:,7:]` 그대로 (0) (1186) | `root_states[:,7:13]` 그대로 (0) (`events.py:61`) | 동일 — 초기 속도 0, RSI 없음 |
| **joint position** | `default_joint_pos` **그대로**, 노이즈 없음 (1180,1187) | `reset_robot_joints` = `reset_joints_by_scale`, **position_range (0.95, 1.05)** = ±5% 스케일 (`parkour_mdp_cfg.py:267-274`) | **A에 joint pos 랜덤화 없음**, B에는 있음 |
| joint velocity | `default_joint_vel` 그대로 (0) | velocity_range (0.0, 0.0) → 0 | 동일 |
| RSI(Reference State Init) | 없음 | 없음 | 동일 — 둘 다 default 자세에서만 시작 |
| `base_external_force_torque` | 없음 | reset-mode 항 존재하나 force/torque range 모두 (0,0) → **no-op** (`parkour_mdp_cfg.py:328-336`) | 실효 동일(둘 다 외력 0) |

**핵심 사실**: B는 reset 시 joint 위치를 default의 ±5%로 스케일 랜덤화한다. A는 매 reset마다 정확히 동일한 default 자세에서 시작 → reset 시점의 상태 다양성(탐색 노이즈 한 축)이 A에 없음.

추가: A는 전체 env 동시 reset 시 `episode_length_buf` 를 `randint(0, max)` 로 랜덤화해 reset을 분산(1140-1142). B(manager-based 기본)는 이런 stagger가 없어 ~1000 step마다 전 env가 동기 reset됨. (학습 초반 blocker라기보다 reset wave 차이.)

---

## 5. terminated vs truncated(time_out) — value bootstrap 처리

이 항목이 본 worker 범위에서 가장 구조적인 차이다.

| 항목 | A | B |
|------|---|---|
| 종료 함수 반환 | `_get_dones` → `(terminated, time_out)` 2개 분리 텐서 (1132) | `terminate_episode` → `reset_buf` **단일** 텐서 (43) |
| 실패(넘어짐 등) 분류 | `terminated` (base_contact/tilt/low_height) | — |
| time_out 분류 | `time_out` (episode_length 만) | — |
| goal 도달 | `terminated` 에 OR (1131) → **bootstrap = 0** | `time_out_buf` 에 OR (38) |
| cfg 종료항 플래그 | DirectRLEnv가 terminated/time_out 분리 사용 | `total_terminates = DoneTerm(func=terminate_episode, **time_out=True**)` (`parkour_mdp_cfg.py:250-256`) |

**B의 구조적 귀결 (가설, IsaacLab `TerminationManager` 동작 전제)**:
B에는 종료항이 `total_terminates` **단 1개**뿐이고 그 항이 `time_out=True` 다. Manager 규약상 `time_outs = OR(time_out=True 인 항들)`, `terminated = dones & ~time_outs` 이므로 — B에서는 **모든 종료 원인(roll/pitch/height/goal/time)이 time_out 으로 플래그**되고 `terminated` 는 항상 비어 있게 된다. 즉 **B는 넘어짐을 포함한 모든 reset에서 value bootstrap(다음 상태 가치 추정)을 수행**한다.
반면 **A는 실패(base_contact/tilt/low_height)와 goal 도달을 `terminated`로 두어 bootstrap=0** (episode 가치가 그 자리에서 끊김). A 코드 주석(1126-1128)도 goal 도달에 zero-bootstrap을 의도했다고 명시.

> ※ 위 B 동작은 IsaacLab `TerminationManager.compute()`의 `time_outs / terminated` 분해 규약에 기반한 추론이다. B의 `DoneTerm`이 1개·`time_out=True`라는 사실은 코드로 확인됨(`parkour_mdp_cfg.py:250-256`). PPO runner단의 실제 bootstrap 적용 여부는 worker-5(알고리즘) 교차 확인 권장.

**학습 초반 함의(가설)**: 학습 초반 A의 policy는 자주 넘어짐 → A는 매 실패마다 value target 0(harsh) 을 받음. B는 같은 실패에서도 next-state value 로 bootstrap → value target 이 덜 가혹. 동일한 조기 실패율 하에서 A의 critic 학습 신호가 B보다 비관적/불안정해질 수 있음.

---

## 6. 도메인 랜덤화 (Events / DR) 비교

| DR 항목 | A (`parkour_env_cfg.py:EventCfg` 274-352) | B (`parkour_mdp_cfg.py:EventCfg` 258-336) | 차이 |
|---------|--------------------------------------------|---------------------------------------------|------|
| **friction** | `foot_physics_material`(startup, `.*foot`): static (0.4,1.5), dynamic (0.3,1.2), restitution 0, buckets 64 + `body_physics_material`(startup, `base`): static (0.6,1.2), dynamic (0.5,1.0) | `physics_material`(startup, **`.*` 전 body**): friction_range (0.6,2.0), buckets 64. 커스텀 함수가 dynamic=static 강제(`events.py:243`), restitution 0 | A는 foot/base 분리·static·dynamic 별도 범위; B는 전 body 단일 범위, static=dynamic. 범위도 다름(B 상한 2.0 더 넓음) |
| **base mass** | `add_base_mass`(startup, `base`): add (−1.0, +3.0) kg | `randomize_rigid_body_mass`(startup, `base`): add (−1.0, +3.0) | 동일 |
| **base CoM** | `randomize_com`(startup, `base`): x(−0.08,0.08), y(−0.04,0.04), z(−0.02,0.02) | `randomize_rigid_body_com`(startup, `base`): x/y/z 모두 (−0.02,0.02) | A의 CoM x/y 범위가 더 넓음 |
| **push_robot** | `push_by_setting_velocity`(interval, **8.0–8.0 s**): vel x/y (−0.5,0.5) | `push_by_setting_velocity`(interval, **8.0–8.0 s**, `is_global_time=True`): vel x/y (−0.5,0.5) | 강도·주기 동일. B만 `is_global_time=True`(전 env 동기 push), A는 기본값(per-env timer) — 타이밍 분산만 다름 |
| **actuator gain 랜덤화** | 비활성(주석: weakened actuator와 충돌) | 비활성(코드 주석 처리, "bad result") | 둘 다 미사용 |
| **external force/torque** | 없음 | reset-mode 항 있으나 range (0,0) → no-op | 실효 동일 |
| **camera position** | 해당 없음(A는 depth camera 미사용) | `random_camera_position`(startup) 정의되나 teacher cfg에서 `None`으로 비활성(teacher_cfg:62) | teacher 학습에는 둘 다 무영향 |

**핵심 사실**: DR은 startup 항(friction/mass/com)과 interval push 모두 **대체로 정합**하며, A가 오히려 friction을 foot/base로 분리하고 CoM x/y 범위가 더 넓다. push 강도·주기 동일. → DR 강도/타이밍 자체에서 "A가 더 가혹"하다고 볼 근거는 약함.

---

## 7. Terrain curriculum 갱신 타이밍

- **A**: `_reset_idx` 안에서 robot 재배치 **이전에** `_update_terrain_curriculum(env_ids)` 호출(`parkour_env.py:1170-1177`). episode 종료마다 해당 env의 terrain level 을 이동거리 기준으로 상향/하향. `init_done` False 인 최초 reset은 skip.
- **B**: terrain curriculum 은 `EventCfg`/`TerminationsCfg`에 없음 → 별도 `CurriculumTerm`(manager) 으로 처리됨. 본 worker 범위 밖, 세부는 worker-2 담당.
- 사실만 기록: A는 curriculum 갱신을 reset 경로 안에 인라인으로 둔다. B의 curriculum 경로/기준 비교는 worker-2 결과 참조.

---

## 학습 초반 영향 가설 (우선순위)

> 아래는 원인 단정이 아닌, 본 worker 범위(종료/리셋/이벤트)에서 관찰된 차이를 "학습 초반에 영향 줄 가능성" 순으로 정렬한 **가설 목록**이다. 정량 검증 필요.

**(가설 1) A에만 있는 `base_contact > 5N` 종료 — episode 조기 사망 구조**
A는 B에 없는 base 접촉 종료를 가진다(§1). 미숙한 초반 policy가 몸통을 자주 지면/장애물에 닿게 하므로, B에서는 보상 벌점으로 끝날 상황이 A에서는 episode 종료로 이어질 수 있다. rollout당 유효 경험·goal 진행이 줄어드는 구조. → A 로그 `Episode_Termination/cause_base_contact`, `Episode_Length/mean_at_reset` 로 즉시 정량 확인 가능.

**(가설 2) terminated vs time_out bootstrap 처리 비대칭**
B는 단일 `DoneTerm(time_out=True)` 구조라 (Manager 규약상) 넘어짐 포함 모든 reset에서 value bootstrap을 수행하는 반면, A는 실패·goal을 `terminated`로 두어 bootstrap=0(§5). 초반 잦은 실패 하에서 A의 critic 신호가 더 비관적/harsh. 가설 1과 함께 작동하면(잦은 조기 종료 × harsh value target) 학습 초반 신호 품질에 복합 영향 가능. ※ B의 time_out 일괄 처리 동작은 Manager 규약 기반 추론이며 runner단 적용은 worker-5 교차 확인 필요.

**(가설 3) reset 시 joint 자세 다양성 부재**
A는 매 reset마다 정확히 동일한 default joint pos에서 시작(§4). B는 ±5% 스케일 랜덤화. 초반 상태 분포 다양성(탐색 노이즈 한 축)이 A에 없음 — 단독으로 학습을 죽일 요인은 아니나 초기 탐색 폭을 좁힐 수 있음.

**(낮음) low_height 임계(−0.2 vs −0.25), tilt 결합 방식, episode_length stagger, friction/CoM DR 세부**
종료 임계의 미세 차이와 DR 세부는 방향성은 있으나 단독 초반 blocker로 보기 어려움. DR 강도는 오히려 정합~A가 약간 넓음.

> 금지 규칙 준수: 위 가설은 RCA ranking 이 아니라 "영향 가능성" 정렬이며, reward/hyperparam scale 단독 튜닝·actuator 약화·contact obs 추가·토크 정량계산은 수정안으로 제시하지 않음.

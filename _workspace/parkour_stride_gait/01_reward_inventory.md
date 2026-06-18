# Go2 Parkour — Reward Inventory & Forward-Stride Signal Audit

**Task #1 / team parkour-stride-gait / worker-1**
**Scope:** inventory only — reward 구조 표 + stride/swing positive 신호 부재 grep 확정 + Genesis(B) 대조 + tracking_goal_vel shuffle 충족 가능성. Root-cause 단정·fix 제안은 task #3 소관(본 문서 범위 밖).
**Files read (현재 파일 직접 인용; 03_fix_proposal.md line 번호는 STALE이라 미사용):**
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` (A, cfg)
- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` (A, `_get_rewards`)
- `Isaaclab_Parkour/parkour_isaaclab/envs/mdp/rewards.py` + `.../config/go2/parkour_mdp_cfg.py` (B = extreme_parkour port; 코드 주석이 "Genesis original"로 지칭하는 upstream)

---

## 0. 핵심 결론 (2-part headline)

1. **A에 전진 stride/swing/foot-clearance/step-length를 양(+)으로 보상하는 항은 존재하지 않는다 — grep 검증됨.**
   활성 양수 항은 `tracking_goal_vel`(+1.5)·`tracking_yaw`(+0.5) 둘뿐. 유일한 gait-shaping 항 `feet_gait_pairing`은 **scale 0.0 = 비활성**(게다가 활성돼도 *위상 동기*이지 전진 변위 보상이 아님). 나머지는 전부 penalty(음수) 또는 air_time **penalty**.

2. **B(extreme_parkour / 주석상 "Genesis original")에도 전진 stride 항은 없다 — 동일 14항 세트, grep 검증됨.**
   B의 `reward_tracking_goal_vel`도 **root COM 속도**(`root_vel_w`) 투영 방식으로 A와 식이 동일. stride/swing/clearance/positive-air_time 항은 B 어디에도 없음(grep hit는 전부 CNN `stride=` 커널 인자 + sensor `track_air_time=True` 플래그였고 reward가 아님).

> **task #3에 대한 함의(중요):** 메모리 §2 "Genesis에 있으면 그대로 도입 우선" 지침은 **여기서 산출물이 없다 — 도입할 Genesis stride reward 자체가 존재하지 않음.** 따라서 "전진 stride 항 추가"는 A·B 양쪽을 넘어서는 **신규 설계**이지 port가 아니다.
> **범위 가드(advisor 교정):** B의 gait *품질*에 대한 증거는 본 조사에 없음(메모리는 A가 원하는 결과 미도달만 명시, B의 실제 gait는 미상). 따라서 "B는 깨끗하게 학습된다"거나 "부재 = root cause"라고 단정하지 않는다. 본 문서는 **구조적 사실**(A·B 공히 stride 항 부재, 식 동일)만 확정한다.

---

## 1. 현재 reward_scales 전체 표 (A)

출처: `parkour_env_cfg.py:607-638` (scale) + `parkour_env.py:1040-1255` (계산식). 18개 항.

| # | term | scale | 부호/상태 | 계산식 (env.py line) | 비고 |
|---|------|-------|----------|----------------------|------|
| 1 | `tracking_goal_vel` | **+1.5** | **positive (active)** | 1057-1066: COM 수평속도 `root_lin_vel_w[:, :2]`를 goal 방향에 투영, `min(proj, cmd)/cmd` | **전진 보상의 사실상 유일 원천. COM 기반 → shuffle 충족 가능(§4)** |
| 2 | `tracking_yaw` | **+0.5** | positive (active) | 1072-1075: `exp(-|target_yaw − heading|)` | heading 정렬만, 다리 동작 무관 |
| 3 | `lin_vel_z_l2` | -1.0 | penalty | 1078-1081: `root_lin_vel_b[z]²`, non-flat 0.5× | |
| 4 | `ang_vel_xy_l2` | -0.05 | penalty | 1079-1083: `Σ root_ang_vel_b[:2]²`, non-flat 0.5× | |
| 5 | `orientation_l2` | -1.0 | penalty | 1086-1088: `Σ proj_grav_b[:2]²`, non-flat에서 0 | |
| 6 | `dof_acc_l2` | -2.5e-7 | penalty | 1091-1093 | |
| 7 | `collision` | -10.0 | penalty | 1096-1099: undesired body 접촉 수 | |
| 8 | `action_rate_l2` | -0.1 | penalty | 1102: `‖a_t − a_{t-1}‖` | |
| 9 | `delta_torques` | -1.0e-7 | penalty | 1105-1108 | |
| 10 | `torques_l2` | -1e-5 | penalty | 1111 | |
| 11 | `hip_pos` | -0.5 | penalty | 1114-1120: hip joint default 편차² | **hip 외전(다리 벌림/뻗기)을 억제하는 방향 — stride와 길항 가능(task #3 검토거리)** |
| 12 | `dof_error_l2` | -0.04 | penalty | 1123-1125: 전 관절 default 편차² | **default 자세 유지 압력 → 큰 swing 억제 방향** |
| 13 | `feet_stumble` | -1.0 | penalty | 1128-1132: 수평력 > 4×수직력 | |
| 14 | `feet_edge` | -1.0 | penalty | 1134-1203: 발이 edge grid에 접지(levels>3) | |
| 15 | `feet_dragging` | -0.1 | penalty | 1143-1152: **뒷발(RL,RR)만** 접촉中 xy속도>0.05 | swing은 직접 보상 안 함, 끌기만 처벌 |
| 16 | `feet_gait_pairing` | **0.0** | **DISABLED** | 1154-1184: 대각쌍 air/contact 시간 **위상 동기** × flat gate × cmd gate | scale 0 → 기여 0. **활성돼도 위상 동기이지 전진 변위 아님** |
| 17 | `air_time_cap` | -0.1 | penalty | 1205-1214: `Σ clamp(air_time − 1.0s, min=0)` | **air time 초과 처벌**(positive 아님). 실측 inactive(폐기 가설) |
| 18 | `contact_duty_deficit` | -0.5 | penalty | 1216-1224: per-foot 접촉 EMA가 target 0.30 미만이면 처벌 | 직전 3-leg fix. 접지 유도이지 전진 보상 아님 |

**positive(활성): #1, #2 둘뿐. #16은 disabled. stride/swing/clearance/step-length/positive-air_time 항: 0개.**

---

## 2. stride/swing positive 신호 — 존재/부재 확정 (grep 증거)

검색: `grep -n "air_time\|stride\|step_length\|clearance\|foot.*pos\|swing\|gait\|feet_air\|displacement"` on `parkour_env.py` + `parkour_env_cfg.py`.

| 신호 유형 | 상태 | 증거 (file:line) |
|-----------|------|------------------|
| positive `feet_air_time`(swing duration을 +로 보상, IsaacLab 표준 `reward_feet_air_time`류) | **검증된 부재** | `air_time` 사용처는 env.py:1155(gait sync), 1162/1169-1170(sync 제곱오차), 1211-1213(air_time_cap **penalty**)뿐. 양수 보상 용례 없음 |
| step length / stride / foot displacement | **검증된 부재** | grep 결과 `stride`/`step_length`/`displacement` 매치 0건 (A env+cfg 양쪽) |
| foot clearance / swing height | **검증된 부재** | grep `clearance`/`feet_height`/`swing_height` 매치 0건 (env.py:1155 줄 출력 확인) |
| gait phase/pairing | **존재하나 비활성** | env.py:1154-1184 정의 + cfg `feet_gait_pairing: 0.0`(cfg:623). scale 0 → 무기여. flat-gate(`* is_flat`, 1184)로 parkour 지형에선 추가로 0 |
| `foot.*pos` (발 위치) | 사용되나 reward용 아님 | env.py:1188-1200 `feet_pos_w`/`feet_xy`는 **feet_edge 마스크 grid 조회**용일 뿐, 전진 변위 보상 아님 |

> "추정 부재" 없음 — 위 5행 전부 grep/Read로 **검증된 부재 또는 검증된 비활성**.

---

## 3. Genesis(B = extreme_parkour) 대조

repo에 실행 가능한 별도 Genesis 소스 없음(`legged_env_parkour.py`는 주석/주석처리 import만: env.py:1138, on_policy_runner_parkour_original.py:48). **권위 있는 reference = B(Isaaclab_Parkour, extreme_parkour port)**, A 주석이 upstream을 "Genesis original"로 지칭.

**B의 reward 세트** (`parkour_mdp_cfg.py` `TeacherRewardsCfg`):
collision(-10) · feet_edge(-1) · torques(-1e-5) · dof_error(-0.04) · hip_pos(-0.5) · ang_vel_xy(-0.05) · action_rate(-0.1) · dof_acc(-2.5e-7) · lin_vel_z(-1) · orientation(-1) · feet_stumble(-1) · **tracking_goal_vel(+1.5)** · **tracking_yaw(+0.5)** · delta_torques(-1e-7).
→ **A의 B-aligned 14항과 동일. 양수 항도 tracking_goal_vel/tracking_yaw 둘뿐.**

**B에 stride/swing/clearance/positive-air_time 있나? — 없음 (grep 검증):**
`grep -rn "air_time\|stride\|clearance\|swing\|step_length\|feet_air\|gait" Isaaclab_Parkour --include="*.py"` 의 매치는 전부:
- CNN 커널 `stride=` 인자 (`depth_backbone.py`, `state_encoder.py`) — reward 아님
- `track_air_time=True` (`parkour_teacher_cfg.py:25`) — **contact sensor 설정 플래그**이지 reward 아님

**B `reward_tracking_goal_vel` 식** (`rewards.py:169-182`):
```python
target_vel = target_pos_rel / (‖target_pos_rel‖ + 1e-5)
cur_vel    = asset.data.root_vel_w[:, :2]          # ← root COM 속도
proj_vel   = Σ(target_vel * cur_vel)
command_vel = command_manager['base_velocity'][:, 0]
rew_move   = min(proj_vel, command_vel) / (command_vel + 1e-5)
```
→ **A(env.py:1057-1063)와 식 동일.** 둘 다 COM 속도 투영, per-foot 항 없음.

> **결론:** "Genesis에 있으면 도입" 경로는 산출물 없음 — B/Genesis에 도입할 stride reward가 부재. task #3의 stride 항은 **A·B를 넘어서는 신규 설계**.

---

## 4. tracking_goal_vel이 shuffle로 충족 가능한가 — 코드 근거

`parkour_env.py:1057-1066`:
```python
cur_vel_w   = self._robot.data.root_lin_vel_w[:, :2]   # body COM 수평속도
proj_forward = Σ(cur_vel_w * goal_dir)                 # goal 방향 투영
commanded_speed = |self._commands[:, 0]|
tracking_goal_vel = min(proj_forward, commanded_speed) / (commanded_speed + 1e-5)
```

- 보상 입력이 **root COM 속도 단 하나** — 발이 어떻게 움직였는지(전진 reach swing인지, 제자리 shuffle/끌기인지)에 **무관**.
- COM이 goal 방향으로 전진하기만 하면(짧은 shuffle/제자리 step으로도) `proj_forward > 0` → 보상 충족.
- per-foot swing/stride/clearance 항이 reward 세트에 **부재(§2)** → "깨끗한 stride"로 유도하는 gradient 부재. shuffle과 stride는 `tracking_goal_vel` 관점에서 **동일 보상**(=구분 불가).
- B도 동일(`root_vel_w` COM 기반, §3) → 이 shuffle-충족 가능성은 A 고유 회귀가 아니라 reference 공유 구조.

> **검증된 사실:** tracking_goal_vel은 COM-velocity-only 구조이므로 shuffle로 충족 가능. (단, 이것이 관측된 shuffle의 *원인*이라는 단정은 task #3 소관 — 본 문서는 구조적 가능성만 확정.)

---

## 부록: 폐기 가설 재확인 (다시 꺼내지 말 것)
- "air_time_cap이 swing 처벌" — air_time_cap은 air time>1.0s **초과분**만 처벌(#17), 실측 inactive. swing 자체 처벌 아님.
- "contact_duty가 tapping 유도" — target 0.30 < 정상 duty 0.5-0.7 → stride/tap 중립.
- actuator/torque/height_scan/action_latency: 본 문서 미거론(메모리 제약).

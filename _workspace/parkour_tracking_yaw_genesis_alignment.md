# Parkour tracking_yaw — Genesis 정렬 분석

> **Scope**: READ-ONLY 분석. 코드 수정 없음. 단일 patch 권고만 제시.
> **작성**: debug-worker (Claude opus 4.7 1M)
> **증상**: step/stair/gap 지형에서 robot이 goal 방향만 응시한 채 정지(standstill trap)
> **사용자 관찰**: c3a7c4395af commit (`moving_mask` 게이트 제거) 직후부터 재발

---

## 0. Genesis 원전 위치 (Tier 1)

| 자원 | 절대경로 |
|------|---------|
| Genesis env 본체 | `/home/lgb/RobotSW_Genesis/parkour/legged_env_parkour.py` |
| Genesis 학습 스크립트 (cfg) | `/home/lgb/RobotSW_Genesis/parkour/train_parkour.py` |
| IsaacLab parkour env | `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py` |
| IsaacLab parkour cfg | `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py` |
| 직전 commit diff 캐시 | `/tmp/parkour_diff.txt` (c3a7c4395af^ → c3a7c4395af) |

---

## 1. Genesis `tracking_yaw` 정확 구현 (Tier 1)

### 1-1. Reward 함수 본체

`legged_env_parkour.py:1462-1465`

```python
def _reward_tracking_yaw(self):
    # print(self.target_yaw, self.yaw)
    rew = torch.exp(-torch.abs(self.target_yaw - self.yaw))
    return rew
```

**관찰**:
- 입력: `self.target_yaw`(절대 world yaw, 다음 goal 방향), `self.yaw`(robot base world yaw)
- 수식: `exp(-|target_yaw - yaw|)` — **L1, σ=1.0 등가**
- **wrap 없음**: `atan2(sin, cos)` 처리 없음. ±π 경계에서 보상 점프 발생 가능 (Genesis 원본 자체의 minor 결함이지만 학습은 이대로 수렴함)
- **gating 없음**: 속도 조건, alive 조건, command 조건 모두 **없음**

### 1-2. `target_yaw` 정의

`legged_env_parkour.py:746-764` (`_update_goals`)

```python
self.target_pos_rel = self.cur_goals[:, :2] - self.base_pos[:, :2]
norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
target_vec_norm = self.target_pos_rel / (norm + 1e-5)
self.target_yaw = torch.atan2(target_vec_norm[:, 1], target_vec_norm[:, 0])
```

`self.yaw` 정의: `legged_env_parkour.py:925`
```python
self.yaw = torch.deg2rad(self.base_euler[:, 2]).to(gs.device)
```

### 1-3. Reward scale

`train_parkour.py:191-215`

```python
"tracking_goal_vel": 1.5,
"tracking_yaw": 0.5,
...
```

`tracking_sigma: 0.2` (line 187), `positive_rewards: False` (line 189 — 보상 음수 누적 허용).

### 1-4. 호출 빈도 — Genesis는 매 step (decimation 후) 계산. IsaacLab도 동일.

### 1-5. **Genesis 전체 코드 grep 결과** (Tier 1)

```
grep -n "moving_mask\|smoothstep\|horizontal_speed" legged_env_parkour.py
→ (Bash completed with no output)
```

**결론**: Genesis 원전에 `moving_mask` 류 게이팅 메커니즘은 **존재하지 않는다**.

---

## 2. IsaacLab 현재 구현 (post c3a7c43, Tier 1)

`parkour_env.py:912-918`

```python
# tracking_yaw: exponential decay from heading error (Genesis line 1451-1454)
# No speed gating — stand-still local optimum is prevented by tracking_goal_vel (weight=1.5)
# which only rewards forward velocity along the goal direction.
heading = self._robot.data.heading_w  # [N] world frame yaw
yaw_diff = self._target_yaw - heading  # [N]
yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))   # ← Genesis 대비 추가됨
tracking_yaw = torch.exp(-torch.abs(yaw_diff))
```

`parkour_env_cfg.py:475-476` (scale): `tracking_goal_vel = 1.5, tracking_yaw = 0.5` — Genesis 일치.

### 2-1. IsaacLab 직전 구현 (c3a7c4395af^, with `moving_mask`) — Tier 1

`/tmp/parkour_diff.txt:381-392` (diff 캡처):

```python
# (직전 버전, c3a7c43에서 제거됨)
yaw_diff = self._target_yaw - self._robot.data.heading_w
yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
tracking_yaw = torch.exp(-torch.abs(yaw_diff))

# Movement gating: robot이 정지 상태일 때 yaw reward를 차단
horizontal_speed = torch.norm(self._robot.data.root_lin_vel_b[:, :2], dim=1)
speed_lo = self.cfg.yaw_reward_speed_lower
speed_hi = self.cfg.yaw_reward_speed_upper
moving_mask = ((horizontal_speed - speed_lo) / (speed_hi - speed_lo)).clamp(min=0.0, max=1.0)
tracking_yaw = tracking_yaw * moving_mask
```

c3a7c43에서 함께 제거된 또 다른 변경: `tracking_goal_vel.clamp(min=0.0)` 도 주석 처리 (`/tmp/parkour_diff.txt:376-377`)
→ 현재는 음수 proj_forward (= 후진) 가 그대로 음수 보상으로 흐름 (Genesis와 동일 의도).

---

## 3. 차이점 매트릭스

| 항목 | Genesis 원전 | IsaacLab 현재 (post c3a7c43) | IsaacLab 직전 (with moving_mask) | 정렬 필요? |
|------|--------------|------------------------------|-----------------------------------|----------|
| heading source | `self.yaw` (base_euler[2], rad) | `self._robot.data.heading_w` | 동일 | OK |
| target_yaw | `atan2(target_pos_rel)` | `atan2(target_pos_rel)` | 동일 | OK |
| yaw_diff wrap | **wrap 없음** | `atan2(sin, cos)` | atan2 wrap | **차이 (사소)** |
| 수식 | `exp(-\|yaw_diff\|)` | `exp(-\|yaw_diff\|)` | 동일 | OK |
| 게이팅 | **없음** | 없음 | smoothstep on \|root_lin_vel_b xy\| | **OK (제거가 Genesis 정렬)** |
| weight (scale) | 0.5 | 0.5 | 0.5 | OK |
| `tracking_goal_vel` clamp(min=0) | **없음** (signed proj) | 없음 (주석 처리) | 있었음 (c3a7c43에서 제거) | OK |
| `positive_rewards` clip | False (음수 누적 허용) | 없음 = 음수 허용 | 동일 | OK |
| `next_goal_threshold` | 0.2 m | 0.2 m | 동일 | OK |
| `reach_goal_delay` | 0.1 s | 0.1 s | 동일 | OK |
| `num_goals` | 8 | 8 | 동일 | OK |
| termination tilt | roll/pitch > **70°** (1.22 rad) | `sin²(tilt) > 0.99` ≈ **84°** | 동일 | **차이 (Tier 2, defer)** |
| termination height | `base_z < -0.15` (Genesis line 971) | `base_z < termination_height = -0.2` | 동일 | OK |
| **`push_robots` event** | **존재 (line 1414-1424)**: 매 `push_interval_s=10s` ±1 m/s xy push | **부재** (EventCfg에는 `foot_physics_material` 1개만) | 부재 | **★ 결정적 차이** |

---

## 4. Root cause 재정의 (advisor 점검 후)

### 4-1. Standstill trap의 보상 산수

per-step (step_dt=0.02s) 기준, robot이 정확히 goal을 응시한 채 정지:

- `tracking_yaw` = 1.0 × scale 0.5 × step_dt 0.02 = **+0.01/step**
- `tracking_goal_vel` = `min(proj=0, cmd=0.65)/cmd` = 0 × 1.5 × 0.02 = **0**
- default pose 유지: `hip_pos≈0, dof_error_l2≈0, action_rate≈0, torques≈small` → penalty 거의 0
- net: **양수**

**즉 standstill-facing-goal은 Genesis에서도 IsaacLab에서도 보상 산수로는 local optimum이다.**

→ `moving_mask` 제거가 *원인*이 아니다. Trap을 깨는 **외부 메커니즘**이 Genesis에는 있고 IsaacLab에는 없는 것이 진짜 원인.

### 4-2. Genesis가 trap을 깨는 메커니즘 (Tier 1)

`legged_env_parkour.py:1414-1424` `_push_robots()`:

```python
def _push_robots(self):
    if self.push_interval_s > 0:
        max_push_vel_xy = self.env_cfg["domain_rand"]["max_push_vel_xy"]  # = 1.0 m/s
        dofs_vel = self.robot.get_dofs_velocity()
        push_vel = gs_rand_float(-max_push_vel_xy, max_push_vel_xy, (self.num_envs, 2), self.device)
        push_vel[((self.common_step_counter + self.env_identities) % int(self.push_interval_s / self.dt) != 0)] = 0
        dofs_vel[:, :2] += push_vel
        self.robot.set_dofs_velocity(dofs_vel)
```

`train_parkour.py` 의 `domain_rand`: `push_robots=True, push_interval_s=10, max_push_vel_xy=1.0`.

**핵심 메커니즘**:

매 10초마다 robot base에 **±1 m/s 랜덤 xy 충격** 부여.
- 뒤로 푸시되면 → `proj_forward = base_vel · goal_dir < 0` → `tracking_goal_vel < 0` (signed, no clamp)
- weight 1.5 × 음수 → 즉시 **-1.5/step × 음수 크기** 의 큰 페널티
- 정지 상태로는 보상 누적에서 빠르게 손해 → 정지가 **불안정한 균형**으로 강제됨

`positive_rewards: False` (train_parkour.py:189) 가 이 음수 신호를 그대로 살려둔다.

### 4-3. IsaacLab parkour `EventCfg` (Tier 1)

`parkour_env_cfg.py:274-288`:

```python
@configclass
class EventCfg:
    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        ...
    )
```

**push_robots 이벤트 부재**. `parkour_env.py` 본체에도 `_push_robots`, `set_root_velocity_to_sim` 류 외란 코드 없음 (`grep -n "push" parkour_env.py` → no output).

→ **IsaacLab parkour는 외란 자체가 없다**. Standstill 균형이 영원히 안정적.

---

## 5. 권고 patch (단일 변경, env-worker 위임)

### 5-1. Patch 설명

**의도**: Genesis `_push_robots()` 를 IsaacLab의 표준 EventCfg idiom으로 포팅. `set_dofs_velocity` 직접 호출 X. IsaacLab의 `mdp.push_by_setting_velocity` 사용.

**범위**: `parkour_env_cfg.py` 한 파일, `EventCfg` 클래스에 EventTerm 1개 추가.

### 5-2. Patch text — **`parkour_env_cfg.py:274-288` `EventCfg` 확장**

```python
@configclass
class EventCfg:
    """Configuration for environment randomization events."""

    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # === Genesis 정렬: _push_robots (legged_env_parkour.py:1414-1424) ===
    # Genesis 원전: push_interval_s=10, max_push_vel_xy=1.0 — base xy velocity 에 ±1 m/s 랜덤 충격.
    # standstill-facing-goal local optimum (tracking_yaw +0.5 weight + tracking_goal_vel proj=0) 을
    # 깨는 핵심 외란. 이 event 없으면 정지가 안정 균형이 되어 trap 발생 (commit c3a7c43 이후 재현).
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 10.0),  # Genesis push_interval_s=10
        params={
            "velocity_range": {
                "x": (-1.0, 1.0),   # Genesis max_push_vel_xy=1.0
                "y": (-1.0, 1.0),
            },
        },
    )
```

**참고 reference 구현** (IsaacLab 표준 패턴):
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/velocity_env_cfg.py` — `push_robot` EventTerm 같은 형식
- `isaaclab.envs.mdp.events.push_by_setting_velocity` — IsaacLab native

`mdp` import 확인: `parkour_env_cfg.py` 상단에 `from isaaclab.envs import mdp` (또는 동등) 가 이미 import 되어 있어야 함. 안 되어 있으면 env-worker 가 추가.

### 5-3. Patch 적용 후 검증 메트릭

- `Episode_Reward/tracking_goal_vel` : trap 시 0에 고착 → 양수로 회복 (≥ 0.1/step 평균)
- `Episode_Reward/tracking_yaw` : 1.0 근처 saturate 였던 것이 0.5~0.8 분포로 내려옴 (움직이는 동안 yaw 오차 자연 발생)
- `Episode_Termination/cause_base_contact` : push로 미세 증가 (OK, 학습 신호 회복 신호)
- `curriculum/mean_terrain_level_{step,stair,gap}` : 정체했던 것이 0→3 이상으로 상승 (1k iter 내)
- `Episode_Length/mean_at_reset` : trap 상태에서 max_episode_length 고착이었던 것이 다양화

권고 검증 길이: **500–1000 iters**, 4096 envs. 1k iter까지 `curriculum/mean_terrain_level_step` 가 1 이상 못 오르면 다음 가설로.

### 5-4. **NOT** 권고 (advisor 점검 결과)

- `moving_mask` 재도입 → Genesis 원전에 없음. 본 작업의 정렬 의도와 정반대.
- termination tilt 70°로 타이트닝 → Tier 2 차이, 단일 변경 원칙 위반. push_robots 단독 효과 검증 후로 deferred.
- tracking_yaw 수식 변경 (wrap 제거) → 사소한 차이, sin/cos atan2 wrap 은 ±π 경계 외에는 동일. 위험 대비 이득 없음.

---

## 6. 사용자 질문 답변: "왜 IsaacLab parkour만 trap?"

> **"다른 IsaacLab/Genesis 프로젝트에서는 moving_mask 같은 게이트 없이도 standstill trap이 안 생기는데 왜 parkour만 생기는가?"**

**답**: Genesis 원전 parkour 도 게이트는 없지만 **`push_robots` 외란 event** 가 정지 균형을 강제로 깬다 (legged_env_parkour.py:1414-1424). 일반 IsaacLab velocity locomotion 환경 (예: `Isaac-Velocity-Flat-Anymal-C-v0`, `Isaac-Velocity-Rough-Anymal-D-v0`) 도 EventCfg에 `push_robot` EventTerm 이 표준으로 포함되어 있어 같은 메커니즘으로 trap이 방지된다.

IsaacLab `direct/parkour` 환경은 Genesis 의 reward 와 obs 는 정밀하게 포팅했지만 **`_push_robots` 이벤트 포팅이 누락**되어 있다 (parkour_env_cfg.py:274-288 의 `EventCfg`에는 `foot_physics_material` 1개만 존재). 이것이 parkour 환경에서만 trap 이 재현되는 단일 원인.

c3a7c4395af 의 `moving_mask` 는 이 누락된 prong 을 보완하기 위한 **workaround** 였다. 사용자 의도("Genesis 방식 그대로")는 workaround 제거 + 누락된 prong 포팅이 정합적.

---

## 7. Evidence tier 정리

| 항목 | Tier | 근거 |
|------|------|------|
| Genesis `_reward_tracking_yaw` 구현 | 1 | `legged_env_parkour.py:1462-1465` 직접 read |
| Genesis `_push_robots` 구현 | 1 | `legged_env_parkour.py:1414-1424` 직접 read |
| Genesis reward_scales | 1 | `train_parkour.py:191-215` 직접 read |
| Genesis push_robots cfg (interval_s=10, vel=1.0) | 1 | `train_parkour.py:168-172` 직접 read |
| Genesis `moving_mask` 부재 | 1 | grep no-match |
| IsaacLab 현재 tracking_yaw | 1 | `parkour_env.py:912-918` |
| IsaacLab 현재 `EventCfg` (push 부재) | 1 | `parkour_env_cfg.py:274-288` |
| IsaacLab 직전 (with moving_mask) | 1 | `/tmp/parkour_diff.txt:381-392` |
| standstill 보상 산수 (+0.01/step) | 2 | 코드 인용 기반 추론, runtime 측정 없음 |
| termination tilt 차이 (70° vs 84°) | 2 | sin² 변환 추론, 동작 검증 없음 |

## 8. 후속 worker 위임 권고

| Worker | 작업 | 파일:라인 | 권고 우선순위 |
|--------|------|-----------|---------------|
| env-coordinator → reward-worker 또는 cfg-worker | `EventCfg.push_robot` 추가 | `parkour_env_cfg.py:274-288` | **critical** |
| (보류) debug-worker | push_robots 단독 효과 1k iter 검증 후 tilt termination 70° 타이트닝 검토 | `parkour_env.py:1089-1090`, `parkour_env_cfg.py:522` | warning, deferred |

---

## 9. 변경 적용 전 확인 사항 (env-worker)

1. `parkour_env_cfg.py` 상단 import에 `from isaaclab.envs.mdp.events import push_by_setting_velocity` 또는 `from isaaclab.envs import mdp` 존재 확인. 없으면 추가.
2. `push_by_setting_velocity` signature 검증: `params.velocity_range` 가 `{"x": (lo, hi), "y": (lo, hi)}` 형식인지 (manager_based velocity_env_cfg 와 일치하는지). IsaacLab 0.4+ 표준.
3. push 가 학습 초반 (curriculum level=0) 에 너무 강할 우려 → `interval_range_s=(10.0, 10.0)` 유지하고 `velocity_range` 만 `(-0.5, 0.5)` 로 보수 시작 후, 학습 안정화 후 Genesis 값 `(-1.0, 1.0)` 복원도 옵션. 단, 본 patch 의 단일 변경 원칙은 Genesis 값으로 시작.

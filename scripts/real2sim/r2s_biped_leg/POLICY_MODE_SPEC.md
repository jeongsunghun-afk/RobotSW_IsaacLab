# R2S-BipedLeg Policy Mode 구현 SPEC

학습된 hind_leg history 정책을 gui_controller에서 policy mode로 구동.
초기 요구(사용자): ①policy mode ②x_vel/yaw command UI ③action을 sim+real 동시 전송.

> **모델 선택 (2026-08-26 갱신)** — 특정 체크포인트를 박아 두지 않는다. GUI 의 Plant 그룹에
> **Policy model 콤보**가 붙어 `logs/rsl_rl/hindLeg_history_direct/*/exported/deployable_policy.pt`
> 를 훑고, **Plant 라디오와 같은 플랜트로 학습된 최신 정책**을 자동 선택한다.
> 플랜트 판정은 run 의 `params/env.yaml` 에 기록된 actuator 블록으로 한다(run 이름이 아니다):
> foot viscous `0.09`=stock · `0.0337`=PACE(현재 `rga.py`) · 없음=legacy.
> 학습 계약(obs 34 · action_scale · gait_period · step_dt · command range)이 GUI 상수와
> 어긋나는 후보는 **사유를 달아 선택 불가**로 표시한다. `--model` 로 명시하면 목록 맨 위에 얹힌다.
>
> 왜 바뀌었나: 종전에는 `_DEFAULT_MODEL_PATH` 로 `2026-07-22` run 을 박아 뒀는데, `rga.py` 플랜트가
> 바뀌어도 그 상수는 안 따라와서 **없어진 로봇용 정책**이 조용히 기본값이 됐다.

## 확정 아키텍처 (검증 완료 2026-07-23)

```
gui_controller (시스템 python, PyQt5, torch-free)
  │  policy command: mode, input_source(sim/real), x_vel, yaw   ── UDP POLICY_CMD_PORT(9884)
  ▼
policy_runner_bipedleg.py (conda isaac-6.0, torch, Isaac 앱 없음 ← 검증됨)
  │  obs 구성(bit-parity) → estimator로 priv_explicit → act_inference → action(8 target q)
  ├── action ── UDP CMD_PORT(9881) ──▶ sim_runner_bipedleg.py (Isaac, actuator)
  │                                         └─ rich state(joint + gravity) ── UDP STATE_PORT(9882) ──▶ policy_runner
  └── action ── UDP REAL_ACTION_PORT ──▶ real endpoint (seam; 미설정 시 no-op)
                                              └─ real state(joint + IMU gravity) ──▶ policy_runner (input_source=real 시)
```

- **정책은 gui 밖 conda 프로세스**(gui=PyQt5 시스템 python엔 torch 없음, conda엔 PyQt5 없음 → 한 프로세스 불가).
- gui는 통제(mode/source 토글 + x_vel/yaw)만. 사용자 "gui가 통제" 의도 충족.
- 두 input case 대칭: policy_runner가 sim state 또는 real state 중 하나로 obs 구성. action은 항상 양쪽 fan-out.
- **real은 seam**: endpoint(IP/port) 설정 가능, 미설정 시 no-op. real state 수신(real-primary)은 real 프로토콜 미정이라 stub. **경고: real이 자기 IMU 폐루프 없이 미러 target만 받으면 자유베이스 2족은 넘어짐** — 사용자에게 고지됨.

## 정책 로드 전략 (verify_policy_load.py 로 검증 PASS)

```python
import inspect, yaml, torch
from rsl_rl.algorithms.ppo_parkour import PPOParkour
from rsl_rl.runners import OnPolicyRunnerParkour
train_cfg = yaml.safe_load(open("<run>/params/agent.yaml"))
train_cfg["algorithm"].setdefault("rnd_cfg", None)
train_cfg["algorithm"].setdefault("symmetry_cfg", None)
_acc = set(inspect.signature(PPOParkour.__init__).parameters)
train_cfg["algorithm"] = {k:v for k,v in train_cfg["algorithm"].items() if k in _acc or k=="class_name"}
runner = OnPolicyRunnerParkour(MockEnv(), train_cfg, log_dir=None, device=DEVICE)  # MockEnv= get_observations/num_envs/num_actions/device/cfg/_robot.data.joint_names/unwrapped
runner.load("<run>/model_23100.pt", load_optimizer=False)
policy = runner.get_inference_policy(device=DEVICE)   # = act_inference
estimator = runner.alg.estimator
# 추론: obs["priv_explicit"] = estimator(obs["policy"]); action = policy(obs)
```
- jit/onnx export 불가(RMA=estimator+history, dict 입력 — 표준 exporter는 단일 tensor만).
- priv_latent 차원은 ckpt `priv_encoder.0.weight.shape[1]` 로 역산(mock obs에 반영). act_inference는 priv_latent 미사용(history_latent 대체)이나 runner 초기화/critic 사이징에 필요.

## obs bit-parity (hind_leg_env._get_observations, history_observation=True 경로)

policy obs (34) = cat 순서 **정확히**:
1. `projected_gravity_b` (3)  ← state의 gravity
2. `commands` (3) = [x_vel, y_vel=0, yaw]  ← gui
3. `joint_pos - default_joint_pos` (8)  ← state joint (default 전부 0)
4. `joint_vel` (8)  ← state joint
5. `actions` (8) = 직전 스텝의 raw action(정책 출력, action_scale 적용 전)
6. `clock_obs` (4) = [sin2πφ_HL, cos2πφ_HL, sin2πφ_HR, cos2πφ_HR], φ_HR=(φ+0.5)%1

관절 순서 = JOINT_NAME_PATTERNS leg-major: HL{hip,thigh,calf,foot}→HR{...}. 학습 env의
`_robot.joint_names`(articulation 순서)와 다를 수 있음 — **정책은 학습 시 articulation 순서로 obs/action**을
봤으므로, policy_runner는 학습 articulation 순서를 재현해야 함. (주의: 학습 env는 find_bodies/joint 순서 그대로
joint_pos를 썼다 → r2s의 JOINT_NAME_PATTERNS leg-major와 다르면 재매핑 필요. 구현 시 학습 env joint_names 확인.)

stateful (policy_runner 내부 관리, per control step 50Hz):
- **clock**: φ += step_dt/gait_period = 0.02/0.6 = 0.0333/step, 매 step 무조건 진행(standing 게이트는 reward만 건드림, clock obs 아님).
- **history buffer** (10, 34): 첫 스텝 obs 10× 복제, 이후 roll(oldest 버리고 최신 append). 학습 `episode_length_buf<=1` 분기 재현.
- **prev action**: 이번 obs의 action = 직전에 정책이 낸 raw action. 초기 0.
- 순서: state 수신 → obs 구성(현재 clock, prev_action) → act_inference → clock 진행 → prev_action=action → action 송신.

action → joint target: `target = 0.25*action + default_joint_pos(0)`. **slew limiter 우회**(학습엔 없음).
게인은 cfg DCMotor(65/53/12/20) 그대로. **faithful_pd/write_joint_stiffness 경로 금지**(q≈target/2 버그).

## command 범위 (학습값)
x_vel∈[-0.5, 2.0], y_vel≡0(미학습, 슬라이더 금지), yaw∈[-0.5, 0.5]. **yaw 추종 약함**(학습 track_ang_vel 0.145) — 회전 굼뜸 예상.

## 파일별 변경
1. `r2s_udp.py`: POLICY_CMD 패킷(gui→policy_runner: mode,source,x_vel,yaw + magic), rich STATE에 gravity(3) 추가 or 신규 패킷, action은 기존 CMD 패킷 재사용(q=target).
2. `r2s_biped_leg_env.py`: get_lowstate에 gravity 추가; policy-mode apply 경로(slew 우회, action→target). fix_base=False.
3. `sim_runner_bipedleg.py`: rich state(gravity) 송신; policy action 수신 적용.
4. `policy_runner_bipedleg.py` (신규): 위 로드+obs+fan-out.
5. `gui_controller.py`: mode 토글 + policy UI(x_vel/yaw 슬라이더, source 토글) + POLICY_CMD 송신.

## joint 순서 (실측 2026-07-23 check_joint_order.py) — 확정
- **articulation 순서**(정책 obs/action): `[HL_hip, HR_hip, HL_thigh, HR_thigh, HL_calf, HR_calf, HL_foot, HR_foot]` (type-major).
- r2s leg-major(_joint_ids)=`[0,2,4,6,1,3,5,7]` — position-control 전용.
- **결정: policy mode의 action/state UDP는 articulation 순서로 통일**(재매핑 회피). sim_runner의 policy 경로는
  `set_joint_position_target(target)`(전체 articulation, joint_ids 없이), state는 `robot.data.joint_pos[0]`(articulation) + gravity.
  → policy_runner는 재매핑 없이 정책과 직결. gui position mode(leg-major)와는 별개 데이터 경로/포트.

## 미해결/구현 시 확인
- sim-primary: fix_base=False면 정책 없이 안 서지만 정책이 폐루프로 균형. spawn z=0.6(학습값).
- 50Hz UDP 왕복(sim_runner↔policy_runner) 지연이 한 step(20ms) 안에 드는지.
- estimator로 priv_explicit 채움: `runner.alg.estimator(obs["policy"])`. base velocity state 불필요.


---

## 액션 배선 (2026-08-27) — deploy 시험용 두 갈래 동시 추론

배포를 검증하려면 같은 정책을 **두 입력으로 동시에** 돌려 비교해야 한다:

```
real obs (모터 엔코더) ─▶ 정책 ─▶ action_real
sim  obs               ─▶ 정책 ─▶ action_sim
```

그리고 그 둘을 **어느 쪽으로 보낼지 목적지마다 따로** 고를 수 있어야 한다(교차 배선 포함).

### 프로토콜 변경 — `source` 하나 → `src_for_sim` / `src_for_real`

```
POLICY_CMD  magic(I) seq(I) mode(i) src_for_sim(i) src_for_real(i) x_vel(f) yaw(f)   28 B
  src_*   -1 = POLICY_SRC_OFF   중립(default) 목표 송신
           0 = POLICY_SRC_SIM   sim obs 로 계산한 액션
           1 = POLICY_SRC_REAL  real obs 로 계산한 액션
기본값     src_for_sim = 0, src_for_real = 1   — 각자 자기 세계
```

**왜 목적지 중심인가.** 소스 중심(각 액션 → 목적지 비트마스크)으로 두면 두 액션이 같은 목적지를
노리는 **표현 가능한 충돌**이 생기고, 규칙으로 해소해야 한다. 목적지 중심이면 그 상태가
원천적으로 없다. 표현력은 같다 — 9 가지 조합이 전부 가능하다.

⚠ 구조체가 24 → 28 B 다. `unpack_policy_cmd` 가 크기 불일치를 `None` 으로 거절하므로 구버전 gui 는
**조용히 오해석하지 않고 명확히 실패**한다 (자체검정에 24 B 거부 assert 있음).

### 구현 불변식

| 항목 | 규칙 | 안 지키면 |
|---|---|---|
| `prev_action`·history | **갈래마다 분리** | 한쪽이 다른 쪽의 다음 obs 를 덮어써서 장부 버그가 플랜트 불일치처럼 보인다 |
| 안 나가는 갈래 | **그래도 자기 액션으로 전진** | 배선을 도중에 바꾸면 history 가 튄다 |
| gait clock | **공유** (벽시계) | 두 갈래가 비교 불가능해진다 |
| `OFF` | 미송신이 아니라 **중립 송신** | lockstep 상대가 step 안 함 → state 안 옴 → 그 state 쓰는 갈래가 교착 |
| state 대기 | **소비되는 출처만** | 안 쓰는 real 이 조용하면 sim 까지 멈춘다 |

### ★ `sim obs → real` 은 **제공하지 않는다** (2026-08-27 결정)

처음에는 교차 배선을 다 열어 두고 확인 대화상자 + 클램프로 막으려 했는데, **그 조합의 쓰임이
없다**는 것이 정리됐다.

실기 obs 가 정책의 **학습 분포 안에 있으려면 실기가 자기 액션으로 폐루프를 돌아야 한다.**
손으로 잡고 있거나 다른 것이 움직이는 상태의 `q`·`dq`·gravity 는 정책이 본 적 없는 분포다.
그래서 "실기는 안 움직이면서 실기 obs 만 빌려 쓰는" 안전한 중간 단계가 성립하지 않는다.

⇒ 검증 계단은 두 개뿐이다:

```
1. sim  obs → sim     정책이 sim 에서 도는지          (real ← off/hold)
2. real obs → real    실기 폐루프 배포                (sim ← off/hold, 또는 미러링)
```

`real obs → sim` 은 **남겨 뒀다** — 실기가 폐루프를 도는 동안 같은 obs 로 sim 몸이 어떻게
반응하는지 보는 **미러링**이라, 실기로 나가는 것은 아무것도 바뀌지 않는다.
반대 방향(`sim obs → real`)만 콤보에서 뺐다. 두 세계의 자세 차가 그대로 실기 명령이 되는
위험만 남고 얻는 것이 없기 때문이다.

### 비교 지표

두 갈래가 다 살아 있으면 1 초마다 `|Δaction| mean/max [rad]` 를 로그에 낸다.

⚠ **해석 주의.** sim 과 real 이 각자 폐루프를 돌면 두 세계는 몇 스텝 만에 서로 다른 상태로
갈라지므로, 그 뒤의 `|Δaction|` 은 "플랜트 갭"이 아니라 **서로 다른 상태에 대한 응답 차**다.
같은 상태에 대한 비교가 되려면 두 갈래가 **같은 obs** 를 봐야 한다 — 즉 미러링
(`sim ← realobs`, `real ← realobs`) 구성에서 sim 몸의 응답을 실기와 비교하는 쪽이 맞다.

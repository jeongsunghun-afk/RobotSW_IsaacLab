# R2S-BipedLeg — Real2Sim 2족 다리 실험 가이드

8-DOF 2족 다리 로봇(HL/HR 각 4관절)을 Isaac Sim에 올리고, PyQt5 GUI로 관절을 실시간 제어하며,
Monitor 창으로 명령(action) vs 시뮬(sim) 응답을 실시간 비교하는 도구 모음
(`r2s_go2` / `r2s_hind_leg` 패턴 기반).

```
GUI 버튼 ─cmd UDP(9881)─▶ sim_runner_bipedleg(Isaac) ─state UDP(9882)─▶ GUI
                                        GUI ─relay UDP(9883)─▶ monitor(별도 프로세스)
```

전송은 **순수 UDP**다. ROS2는 향후 사용 예정이나 메시지 형식이 미정이라 지금은 UDP seam으로 둔다.
(go2의 ROS2 unitree 계층은 GO2 전용이라 이 커스텀 다리에 재사용 불가.)

> **기존 리그와 공존한다.** `r2s_hind_leg`(R_Skeleton 5-DOF, 9873/9874/9875)와 `r2s_go2`
> (9871/9872, tuner 9875/9876)는 전부 다른 포트/다른 magic을 쓰므로 **동시 실행 가능**하다.
> 패킷 magic도 서로 달라(`R2BC/R2BS/R2BM` vs `R2HC/R2HS/R2MN`) 잘못 보낸 패킷이 오파싱될 일이 없다.

---

## 1. 로봇 (8-DOF 2족)

관절 순서는 **leg-major** — HL 4개(hip, thigh, calf, foot) → HR 4개. 게인/한계는 실측값
(2026-07-21 probe)이며 `motions.py`와 env cfg 양쪽에 동일하게 들어 있다.

| idx | label | kp | kd | soft limit [rad] |
|---|---|---|---|---|
| 0 | HL_hip | 65.0 | 6.0 | (-0.5498, 0.5498) |
| 1 | HL_thigh | 53.0 | 4.8 | (-1.9024, 1.5533) |
| 2 | HL_calf | 12.0 | 1.1 | (-1.3836, 0.4236) |
| 3 | HL_foot | 20.0 | 1.0 | (-0.4192, 1.4662) |
| 4 | HR_hip | 65.0 | 6.0 | (-0.5498, 0.5498) |
| 5 | HR_thigh | 53.0 | 4.8 | (-1.9024, 1.5533) |
| 6 | HR_calf | 12.0 | 1.1 | (-1.3836, 0.4236) |
| 7 | HR_foot | 20.0 | 1.0 | (-1.4662, 0.4192) |

- 기본 자세 `DEFAULT_POSE = [0.0] × 8` (sim `default_joint_pos`와 일치).
- **soft limit은 좌우 비대칭이다.** `HL_foot`과 `HR_foot`은 서로 **미러**이며 오타가 아니다.
  대칭이라고 가정해 "고치지" 말 것.
- 게인이 5-DOF 리그(300/5)보다 한 자릿수 낮으므로(12~65 / 1.0~6.0) GUI 게인 스핀박스 범위도
  kp 0~150, kd 0~15 로 낮춰 놓았다.

---

## 2. 실행 환경

| 프로세스 | Python | 역할 |
|---|---|---|
| `sim_runner_bipedleg.py` | Isaac conda `isaac-6.0` | Isaac에서 다리 구동, UDP 명령/상태 교환 |
| `gui_controller.py` | 시스템 `/usr/bin/python3` (PyQt5) | UDP 발행 GUI (ROS 비의존) |
| `monitor.py` | 시스템 3.10 (matplotlib) | GUI가 spawn하는 실시간 plot 창 |

---

## 3. 빠른 시작 (별칭)

```bash
source scripts/real2sim/r2s_biped_leg/r2s_commands.sh
```

| 별칭 | 터미널 | 스크립트 |
|---|---|---|
| `r2s_bl_sim` | 1 (Isaac) | `run_sim_runner.sh` — conda `isaac-6.0` + sim_runner |
| `r2s_bl_gui` | 2 (시스템 py) | `run_gui_controller.sh` — GUI |

---

## 4. 실행 순서

```bash
# 터미널 1 — Isaac sim_runner (라이브스트림 ON, base 고정)
r2s_bl_sim
#   또는 수동:
#     LIVESTREAM=2 CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p \
#       scripts/real2sim/sim_runner_bipedleg.py --num_envs 1 --fix_base --viz kit
#
#   헤드리스로 돌리려면:  VIZ= r2s_bl_sim        (= --viz 미전달)
#   자유베이스로 돌리려면: FIX_BASE=0 r2s_bl_sim  (넘어짐 — 아래 참고)

# 터미널 2 — PyQt GUI (디스플레이 필요)
r2s_bl_gui
```

다리는 중립(0) 자세로 시작한다(sim `default_joint_pos`와 일치).

### `FIX_BASE`가 기본 1인 이유

이 로봇은 자유베이스 2족이라 base를 고정하지 않으면 GUI로 관절을 스텝하는 순간 균형을 잃고
**즉시 넘어진다**. GUI 관절 조작/추종 확인이 목적이므로 런처 기본값은 `FIX_BASE=1`
(`--fix_base`, base 공중고정)이다. 지면 접촉/낙하 거동을 보려면 `FIX_BASE=0`으로 끈다.

---

## 5. GUI 버튼

| 그룹 | 동작 |
|---|---|
| **Home (default)** | 중립(0) 자세로 보간 이동 |
| **Joint Step** | 선택 관절만 delta[rad] 스텝 (soft limit 클램프) |
| **Sine Sweep** | 선택 관절에 사인 궤적 주입(Start/Stop, 실행 중 amp/freq/관절 라이브 반영) |
| **Motion Playback** | 리타게팅된 SMR 모션 클립 재생 (§5.1) |
| **Gains (faithful PD)** | 선택 관절 kp/kd 실시간 변경 → sim drive 게인에 즉시 반영 |
| **Computed Gains** | sim이 보낸 관절별 유효 관성 I_eff로 `kp=I·(2πf_n)²`, `kd=2ζ·I·(2πf_n)` 계산 → 전 관절 **임시** 적용(Restore defaults로 §1 실측값 복귀) |
| **Monitor** | action/sim 실시간 plot 창(별도 프로세스) |

관절 콤보에는 8개 레이블(HL 4 + HR 4)이 모두 나온다. 명령은 관절별 `kp/kd`(§1 표의 실측 기본값)와
함께 50Hz로 연속 발행되고, soft limit을 벗어나는 명령은 자동 클램프된다.

### 5.1 Motion Playback — 리타게팅 SMR 모션 클립 재생

개 mocap을 이 리그로 리타게팅한 관절각 클립(`motion_data/*_joints.npz`)을 재생한다.
클립 출처·재생성 커맨드·사전 스캔 결과는 [`motion_data/SOURCES.md`](motion_data/SOURCES.md),
로더는 [`motion_clips.py`](motion_clips.py)(순수 numpy, `/usr/bin/python3 motion_clips.py`로 자가 테스트).

**사용법**

1. **Clip** 콤보에서 클립 선택 (`motion_data/` 를 기동 시 스캔 — npz를 넣으면 코드 수정 없이 뜬다).
   패널 아래 한 줄에 그 클립의 클램프 통계·주기성·최대 관절속도가 바로 표시된다.
2. **Speed** 0.25~1.0 (기본 1.0 = 원속). **실기에서는 0.25부터 올리며 확인할 것** —
   최대 관절속도가 배속에 그대로 비례하고, 표시줄의 `peak N rad/s` 가 그 값이다.
   Speed/Loop/Clip은 **Play 시점에** 프레임으로 굳는다(Sine과 달리 재생 중 라이브 반영 아님) —
   재생 중에 바꾸면 표시줄만 갱신되고, 적용하려면 Play를 다시 누른다.
3. **Loop** 체크 시 반복 재생 (아래 ⚠).
4. **Play** — 현재 자세에서 클립 첫 프레임까지 1.5 s(`SEQUENCE_DURATION_S`) 보간으로 진입한 뒤
   클립을 재생한다. RELAX 상태에서 눌러도 되며(Home과 같은 동작) 그 순간 모터가 물린다.
5. **Stop** — 그 시점 자세에서 그대로 정지(HOLD). 스냅백 없음. 비반복 재생은 마지막 프레임에서
   자동으로 HOLD 로 넘어간다.

재생 중 슬라이더·Home·Relax·Sine·Chirp 를 조작하면 **그 조작이 이긴다** — 재생은 즉시 중단되고
패널 버튼이 복구된다(chirp와 같은 폴링 방식). Joint Sliders 의 **Send to robot** 체크박스가 켜져
있으면 모션 프레임도 같은 경로로 실기(9887)에 나간다 — 별도 특례 없음.

**프레임 타이밍** — 소스는 59.988 fps, 발행은 50 Hz 다. 소스 프레임을 그대로 두고 인덱싱만
60 Hz로 하면 6프레임에 1개씩 건너뛰어 불규칙해지므로, UI가 **미리 50 Hz 그리드로 선형 리샘플**해
공유 버퍼에 넣는다(배속도 여기서 반영 — 0.25배속이면 프레임 수가 4배). publisher는 경과 시간으로
프레임 위치를 실수로 계산하고 **인접 두 프레임을 보간**한다: 그리드와 발행 루프가 같은 50 Hz라도
위상이 자유롭게 떠다녀 정수 인덱싱하면 경계에서 한 프레임을 건너뛰고 그 틱의 명령 변화량이 2배로
튄다(trot0 1.0배속 13.2 → 26 rad/s, 실기 무부하 한계 29.6 rad/s 근처).

**⚠ soft limit 클램프** — `motions.SOFT_LIMITS_RAD` 는 URDF limit의 soft(0.9배) 버전이라
두 클립 모두 일부 프레임이 범위를 넘는다. 발행 프레임은 **모델각 기준으로 미리 클램프**된다.

| clip | 위반 프레임 | 최대 초과 |
|---|---|---|
| trot0 | 60/209 (28.7%) | HR_foot 0.0960 rad(5.50°) · HL_foot 0.0644 · HR_hip 0.0211 |
| walk1 | 13/38 (34.2%) | HR_foot 0.0960 rad(5.50°) · HL_foot 0.0651 |

**⚠ Loop — 두 클립 모두 주기적이지 않다.** 마지막↔첫 프레임 차이가 1.10 rad(약 63°)라 그대로
이으면 한 틱에 55 rad/s 다. 그래서 Loop 재생은 사이에 **합성 복귀 구간(bridge)** 을 넣는다
(관절속도 2.0 rad/s 상한 → 두 클립 모두 약 0.56 s). walk1은 본 모션이 0.63 s 라 한 주기의 절반
가까이가 합성 구간이다 — "루프"보다 모션과 복귀 스트로크의 왕복에 가깝다는 점을 감안할 것.

**⚠ foot 채널은 raw각** — CMD/ACT/STATE의 foot 값은 관절각이 아니라 `q_raw = q_foot + q_calf`
(RL_INTERFACE.md §2, coef=+1, `r2s_biped_leg_env_cfg.foot_coupling=True`). npz는 모델각이므로
로더가 발행 직전에 `foot += calf` 를 적용한다. 클램프는 **모델각에서** 하고 변환은 그 뒤다.

**검증** — [`test_motion_playback.py`](test_motion_playback.py) 가 offscreen GUI를 띄워 재생을
프로그램적으로 구동한다: `QT_QPA_PLATFORM=offscreen /usr/bin/python3 test_motion_playback.py`.
프레임 순서·값, foot raw 변환, soft limit 위반 0, 루프 wrap/이음매, Stop/자연종료, 가로채기,
기존 Home/슬라이더/Sine/Relax 무회귀, 실제 UDP(CMD 9881) 경로까지 확인한다.

### 프로세스 구조 (r2s_go2 gui에서 이식, 2026-08-06)

- **publisher 별도 프로세스**: cmd 발행 + 목표 생성(보간/사인)을 `multiprocessing.Process`로 격리.
  UI는 공유메모리(mp.Array)에 모션 스펙만 쓰고, publisher가 경과 시간 기반으로 50Hz 균일 발행한다
  — Qt event loop가 바빠도(스핀박스 드래그) 발행/목표가 굶지 않는다. sim state 수신·monitor 중계·
  I_eff 수신도 publisher 담당.
- **startup 실측 latch**: 첫 sim state를 받을 때까지 발행 보류(CMD_VALID=0), 받으면 실측 자세를
  hold latch — 로봇/sim이 다른 자세일 때 기동 즉시 스냅하는 것을 막는다(2s 타임아웃 시 default).
  이를 위해 sim_runner는 state를 cmd 수신과 무관하게 `(127.0.0.1, 9882)`로 무조건 발행한다.
- **I_eff 채널**: sim_runner가 기동 시 generalized mass matrix 대각(관절별 유효 관성)을 계산해
  1Hz로 state 포트에 흘린다(`R2BI` 패킷). Computed Gains 그룹이 이를 사용.
  실측(2026-08-06, default 자세): hip 0.164 / thigh 0.134 / calf 0.030 / foot **0.002** kg·m².
  ⚠ 단일 f_n이면 게인이 I_eff에 비례하므로 foot처럼 관성이 작은 관절은 매우 낮은 kp가 나온다
  (§1 실측 kp 20은 f_n≈16Hz에 해당) — 계산값은 **시작점**이고 관절별 미세조정은 Gains 그룹으로.

---

## 6. Monitor — action vs sim 실시간 비교

GUI **Monitor** 버튼 → 선택 관절의 3개 plot:

- **q**: action(점선=명령) / sim(엔코더)
- **Joint torque**: sim / real
- **dq**: sim / real

별도 프로세스로 렌더를 GUI의 50Hz UDP 발행에서 격리한다(in-process matplotlib은 발행을 굶긴다).
robot 시리즈는 ROS2 실로봇 연동 시 추가된다.

> ⚠ **토크 축은 실기 값을 관절 좌표로 올려서 그린다.** 실기가 보고하는 토크는 **채널기준**이고
> sim 은 **관절기준**이라 그대로 겹치면 calf 1.5배 · foot 1.2배 어긋난다(hip/thigh는 k=1이라
> 맞아떨어져서 오래 안 보였다). 환산은 감속비 + 커플링 전치 두 조각이고, 감속비 배율은 아직
> **미판정인 게인 지수 n**에 달려 있어 축 라벨에 `n=2`로 표시된다.
> 수식·근거·순서 함정은 **[`TORQUE_COORDINATES.md`](TORQUE_COORDINATES.md)** 참고.

---

## 7. 옵션 / 환경변수

| 변수 / 플래그 | 기본 | 의미 |
|---|---|---|
| `--viz` | `kit` (런처 기본) | 비주얼라이저. **라이브스트림엔 `kit` 필수**, 생략하면 헤드리스 (`--headless`는 6.0에서 deprecated) |
| `VIZ` (env) | `kit` | 런처가 붙일 `--viz` 값. `VIZ=` 로 비우면 헤드리스 |
| `FIX_BASE` (env) | `1` | 1 → `--fix_base`(base 공중고정), 0 → 자유베이스 |
| `LIVESTREAM` (env) | 2 | Isaac 라이브스트림 모드 |
| `GPU` (env) | 2 | sim_runner CUDA 디바이스 |
| `SIM_ENV` (env) | `isaac-6.0` | 활성화할 conda env |
| `--num_envs` | 1 | 시뮬 환경 수(1만 지원) |
| `faithful_pd` (cfg) | True | GUI kp/kd를 실제 sim 게인에 반영 |

---

## 8. 트러블슈팅

- **sim_runner가 `ModuleNotFoundError`로 죽음**: `./isaaclab.sh -p`가 base conda를 가리키는 경우.
  `conda activate isaac-6.0` 후 실행하거나 `r2s_bl_sim` 사용.
- **GUI에서 명령해도 sim이 안 움직임**: sim_runner가 떠 있는지, 포트 9881/9882가 열려 있는지 확인
  (`ss -uln | grep 988`). 다른 R2S 리그와는 포트가 다르므로 동시 실행 가능.
- **로봇이 바로 넘어짐**: `FIX_BASE=0`으로 돌린 경우다. 자유베이스 2족은 GUI 조작에 버티지 못한다.
  `FIX_BASE=1`(기본)로 되돌린다.
- **Monitor 창이 비어있음**: GUI가 sim 상태를 받아야 sim 라인이 그려진다(sim_runner 먼저 기동).
- **라이브스트림 화면이 안 뜸**: `--viz kit`이 빠졌는지 확인(`LIVESTREAM=2`만으로는 부족).
  `r2s_bl_sim`은 자동으로 붙인다. 흰 화면이면 학습이 점유한 GPU와 겹친 것 —
  `GPU=<유휴 GPU> r2s_bl_sim`으로 분리한다.
- **관절이 명령보다 덜 감**: soft limit 클램프(§1 표). hip은 ±0.55 rad로 특히 좁다.

---

## 9. Policy 모드 — 학습된 hind_leg 정책 구동

GUI의 **Mode: Policy**로 전환하면, 학습된 hind_leg history 정책(`model_23100.pt`)이 로봇을 자유베이스에서
폐루프로 구동한다. GUI는 x_vel(앞뒤)·yaw(좌우) 속도 명령과 input source(Sim/Real)만 통제하고, 정책 추론은
별도 `policy_runner_bipedleg.py`(conda torch, Isaac 앱 없음)가 수행하며 action을 **sim + real 양쪽에 전송**한다.

```
gui ──POLICY_CMD(9884)──▶ policy_runner ──POLICY_ACT(9886)──▶ sim_runner(--policy_mode)
                                        ◀──POLICY_STATE(9885)──┘  (요청-응답 lockstep: 1 action = 1 sim step)
policy_runner ──REAL_ACT(9887)──▶ real ──REAL_STATE(9888)──▶ policy_runner   (seam; --real_host 지정 시)
```

**3-터미널 실행 (별칭 — 가장 쉬움):**

```bash
# 한 번만 source (각 터미널에서). ~/.bashrc 에 넣어두면 매번 안 쳐도 된다.
source /home/lgb/IsaacLab-6.0/scripts/real2sim/r2s_biped_leg/r2s_commands.sh

r2s_bl_psim    # 터미널1: sim_runner --policy_mode (라이브스트림 kit). 헤드리스는 VIZ= r2s_bl_psim
r2s_bl_prun    # 터미널2: policy_runner (학습 정책 로드·추론)
r2s_bl_gui     # 터미널3: GUI → Mode를 Policy로, Run 클릭, x_vel/yaw 조절
```

환경변수로 조정: `GPU=1 r2s_bl_psim`, `CKPT=model_10000.pt r2s_bl_prun`, `REAL_HOST=192.168.x.y r2s_bl_prun`.

<details><summary>별칭 없이 직접 실행</summary>

```bash
# 터미널 1: sim (라이브스트림으로 로봇을 보려면 --viz kit, 생략 시 headless)
bash scripts/real2sim/r2s_biped_leg/run_policy_sim.sh
# 터미널 2: 정책 추론 (Isaac 앱 없이 순수 torch)
bash scripts/real2sim/r2s_biped_leg/run_policy_runner.sh
# 터미널 3: GUI
bash scripts/real2sim/r2s_biped_leg/run_gui_controller.sh
```
</details>

**핵심 계약 (POLICY_MODE_SPEC.md):**
- **lockstep 필수**: sim_runner는 POLICY_ACT 하나당 정확히 1 env.step만 진행(학습 `1 action = 1 step` 불변식).
  free-running이면 부팅 중 로봇이 넘어지고 gait clock이 비동기가 된다.
- obs(34) = gravity3 + cmd3 + (q−default)8 + dq8 + prev_action8 + clock4, **articulation 순서**(재매핑 없음).
  priv_explicit(base vel)는 estimator로 추정하므로 state엔 gravity+관절만 필요.
- action→target = `0.25·action + default`, **slew limiter 우회**(학습엔 없음), 게인은 cfg DCMotor 고정.
- command 학습 범위: x_vel∈[-0.5, 2.0], yaw∈[-0.5, 0.5], y_vel≡0(미학습). **yaw 추종은 약함**(학습 track_ang_vel 0.145).
- **real 주의**: real은 자기 IMU로 폐루프를 돌지 않으면(미러 target만 받으면) 자유베이스 2족이 넘어진다.

### 9.1 Real 엔드포인트 — `real_runner/` (라즈베리파이 브리지)

REAL_ACT(9887)/REAL_STATE(9888) seam을 받아주는 파이 쪽 브리지가 `real_runner/`에 있다.
실기 스택(RobotEmbedded ─EtherCAT→ MCU ─CAN-FD→ MD80×8)과는 `RobotTestGait` 예제와 동일하게
Shared Memory(`libRobotSharedMem`)로 교환하며, 단위(deg↔rad)·관절 순서(articulation↔leg-major)·
게인 상한(kd≤5) 변환을 담당한다. **STATE 회신을 20ms로 페이싱해 policy를 50Hz로 묶는 것이 핵심**
(sim과 달리 실기는 lockstep이 아니므로). 빌드·실행·캘리브레이션 체크리스트는
`real_runner/README.md` 참조. ⚠ sign/zero 캘리브레이션 전 TRACK 금지.

## 10. 파일

| 파일 | 역할 |
|---|---|
| `sim_runner_bipedleg.py` (상위 폴더) | Isaac 런처 + UDP 루프 (position / `--policy_mode` lockstep) |
| `policy_runner_bipedleg.py` (상위 폴더) | 학습 정책 추론 + UDP fan-out (conda torch, Isaac 앱 없음) |
| `gui_controller.py` | PyQt5 GUI, position 발행 + policy 명령(Mode 토글) |
| `monitor.py` | 실시간 plot(별도 프로세스) |
| `motions.py` | 자세/게인/한계 상수 + 보간/스텝/사인/클램프 유틸(순수 함수) |
| `motion_clips.py` | 리타게팅 모션 클립 로더/리샘플/클램프 스캔(순수 numpy, 자가 테스트 포함) |
| `motion_data/` | 재생용 SMR 클립 npz + 출처·재생성 커맨드([`SOURCES.md`](motion_data/SOURCES.md)) |
| `test_motion_playback.py` | Motion Playback 통합 테스트(offscreen GUI + mock UDP 수신) |
| `r2s_udp.py` | UDP 패킷 스키마(공유, stdlib만) — 재구현 금지 |
| `POLICY_MODE_SPEC.md` | policy 모드 아키텍처/계약/obs parity 명세 |
| `run_*.sh` / `r2s_commands.sh` | 원커맨드 런처 + 별칭 |
| `real_runner/` | **실기(파이) UDP↔SharedMem 브리지** (§9.1, C++/CMake, 파이에서 빌드) |
| `RobotTestGait/` | 실기 모터 제어 예제 원본(RGA 제공) + SOEM/통신 설명 PDF — 브리지의 참조 사양 |

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
| **Sine Sweep** | 선택 관절에 사인 궤적 주입(Start/Stop, soft limit 클램프) |
| **Gains (faithful PD)** | 선택 관절 kp/kd 실시간 변경 → sim drive 게인에 즉시 반영 |
| **Monitor** | action/sim 실시간 plot 창(별도 프로세스) |

관절 콤보에는 8개 레이블(HL 4 + HR 4)이 모두 나온다. 명령은 관절별 `kp/kd`(§1 표의 실측 기본값)와
함께 50Hz로 연속 발행되고, soft limit을 벗어나는 명령은 GUI에서 자동 클램프된다.

---

## 6. Monitor — action vs sim 실시간 비교

GUI **Monitor** 버튼 → 선택 관절의 3개 plot:

- **q**: action(점선=명령) / sim(엔코더)
- **tau_est**: sim
- **dq**: sim

별도 프로세스로 렌더를 GUI의 50Hz UDP 발행에서 격리한다(in-process matplotlib은 발행을 굶긴다).
robot 시리즈는 ROS2 실로봇 연동 시 추가된다.

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

## 10. 파일

| 파일 | 역할 |
|---|---|
| `sim_runner_bipedleg.py` (상위 폴더) | Isaac 런처 + UDP 루프 (position / `--policy_mode` lockstep) |
| `policy_runner_bipedleg.py` (상위 폴더) | 학습 정책 추론 + UDP fan-out (conda torch, Isaac 앱 없음) |
| `gui_controller.py` | PyQt5 GUI, position 발행 + policy 명령(Mode 토글) |
| `monitor.py` | 실시간 plot(별도 프로세스) |
| `motions.py` | 자세/게인/한계 상수 + 보간/스텝/사인/클램프 유틸(순수 함수) |
| `r2s_udp.py` | UDP 패킷 스키마(공유, stdlib만) — 재구현 금지 |
| `POLICY_MODE_SPEC.md` | policy 모드 아키텍처/계약/obs parity 명세 |
| `run_*.sh` / `r2s_commands.sh` | 원커맨드 런처 + 별칭 |

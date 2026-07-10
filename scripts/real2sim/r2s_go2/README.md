# R2S-GO2 — Real2Sim GO2 실험 가이드

Unitree GO2를 Isaac Sim에 올리고, ROS2 GUI로 관절을 실시간 제어하며,
필요하면 **시뮬레이터와 실로봇을 동시에** 같은 명령으로 구동/비교하는 도구 모음.

```
GUI 버튼 ─▶ ROS2 /lowcmd ─▶ sim_bridge ─▶ UDP:9871 ─▶ sim_runner(Isaac GO2)
   ▲                                                        │
   │            ROS2 /lowstate ◀─ sim_bridge ◀─ UDP:9872 ◀──┘
 Monitor(별도 프로세스): /lowcmd(action) + /lowstate(sim/robot) 실시간 plot
```

고정 계약(토픽/패킷/조인트 순서/게인)은
[`CONTRACT.md`](../../../source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md) 참고.

---

## 1. 실행 환경 (Python 두 세계, 반드시 구분)

| 프로세스 | Python | 역할 |
|---|---|---|
| `sim_runner_go2.py` | **Isaac conda `isaac-6.0` (3.12)** | Isaac Sim에서 GO2 구동, UDP로 명령/상태 교환. rclpy 미사용. |
| `sim_bridge.py` | **시스템 `/usr/bin/python3` (3.10) + ROS2 humble** | `/lowcmd`↔UDP↔`/lowstate` 브릿지 |
| `gui_controller.py` | 시스템 3.10 + humble | PyQt5 GUI, `/lowcmd` 50Hz 발행 |
| `monitor.py` | 시스템 3.10 + humble | (GUI가 spawn) 실시간 plot 창 |

> **함정**: conda가 켜진 셸에서 `python3`는 conda(3.12)를 가리켜 rclpy 임포트가 깨진다.
> `run_*.sh` 런처들은 conda를 **완전히 비활성화**한 뒤 humble + unitree_go overlay를 소싱하고
> `/usr/bin/python3`로 실행하므로, conda가 켜진 터미널에서 그냥 실행해도 된다.

---

## 2. 빠른 시작 (별칭 권장)

한 번만 별칭을 등록하면 세 터미널에서 `cd`처럼 바로 쓸 수 있다:

```bash
source scripts/real2sim/r2s_go2/r2s_commands.sh
```

| 별칭 | 터미널 | 실제 스크립트 |
|---|---|---|
| `r2s_sim` | 1 (Isaac) | `run_sim_runner.sh` — conda `isaac-6.0` 활성화 + livestream + sim_runner |
| `r2s_udp` | 2 (ROS2) | `run_sim_bridge.sh` — 브릿지 |
| `r2s_gui` | 3 (ROS2) | `run_gui_controller.sh` — GUI |
| `r2s_all` | (선택) | `run_all.sh` — tmux 3-pane 동시 기동 (`r2s_all stop`으로 종료) |

각 별칭은 인자를 그대로 전달한다: `r2s_sim --headless`, `GPU=0 r2s_sim`, `FIX_BASE=0 r2s_sim`.

---

## 3. 실행 순서 (sim 전용)

```bash
# 터미널 1 — Isaac sim_runner
r2s_sim
#   또는 수동:
#   conda activate isaac-6.0
#   python scripts/real2sim/sim_runner_go2.py --num_envs 1 [--fix_base] [--headless]

# 터미널 2 — ROS2 <-> UDP 브릿지
r2s_udp                                    # = bash .../run_sim_bridge.sh

# 터미널 3 — PyQt GUI (디스플레이 필요)
r2s_gui                                     # = bash .../run_gui_controller.sh
```

GUI가 뜨면 로봇은 **엎드린(prone) 자세**로 시작한다(실기 GO2와 동일).

---

## 4. GUI 버튼

| 버튼 | 동작 |
|---|---|
| **Stand Up** | prone → folded → stand 궤적(go2_stand_example 방식)으로 기립 |
| **Default (stand)** | `DEFAULT_POSE`(선 자세)로 보간 이동 |
| **Sit** | `SIT_POSE`(웅크림)로 보간 이동 |
| **Joint Step** | 선택 관절만 delta[rad]만큼 스텝 |
| **Sine Sweep** | 선택 관절에 사인 궤적 주입(Start/Stop) |
| **Monitor** | 실시간 plot 창 열기(아래 §5) |

모든 명령은 `kp=25, kd=0.5`(전 관절), `mode=0x01`, `dq=0`, `tau=0`으로 50Hz 연속 발행된다
(실로봇 watchdog heartbeat + CRC/헤더 포함, `CONTRACT.md` §6·§8).

---

## 5. Monitor — 실시간 plot

GUI의 **Monitor** 버튼 → 선택 모터의 3개 plot 창이 **별도 프로세스**로 뜬다:

- **q** (관절각): action(점선=명령) / sim / robot
- **tau_est** (토크): sim / robot
- **dq** (각속도): sim / robot

- **action** = `/lowcmd`의 목표각(절대값). residual 아님 — 이 시스템은 직접 위치 제어.
- **sim** = `/sim/lowstate`(shared) 또는 `/lowstate`(sim 전용)의 엔코더.
- **robot** = `/lowstate`(실로봇)의 엔코더.

> 왜 별도 프로세스인가: matplotlib 렌더가 GUI와 같은 event loop에 있으면 50Hz `/lowcmd`
> heartbeat를 굶겨 실로봇이 protection fault로 빠질 수 있다. 프로세스 격리로 GUI는 발행만,
> monitor는 렌더만 한다(`CONTRACT.md` §11).

모터 선택 콤보 + Pause 버튼. sim 전용 모드에서는 robot 라인이 sim과 겹치지 않도록 자동으로
"sim only"로 표시된다.

---

## 6. 동시 구동 (sim + 실로봇, ⚠ 실로봇이 움직인다)

GUI 버튼 하나의 `/lowcmd`를 **sim과 실로봇 양쪽**이 받게 하는 모드.

> ⚠ **안전**: `R2S_SHARED=1 r2s_gui`를 켜는 순간 실로봇도 즉시 `/lowcmd`를 받는다.
> **실로봇을 안전 위치(엎드림/매달기) + sport release 상태**로 둔 뒤 켤 것.
> 실로봇 이더넷 연결 + `192.168.123.x` static IP 세팅이 선행되어야 한다.

```bash
# 터미널 1 — sim (지면, 실로봇과 정합하려면 fix_base 끄기)
FIX_BASE=0 r2s_sim
# 터미널 2 — 브릿지 (shared: cyclonedds + /sim/lowstate remap)
R2S_SHARED=1 r2s_udp
# 터미널 3 — GUI (shared)
R2S_SHARED=1 r2s_gui
```

shared 모드에서 sim 상태는 `/sim/lowstate`로, 실로봇은 `/lowstate`로 분리 발행되어
Monitor가 sim vs robot을 나란히 비교한다.

---

## 7. 실로봇 통신 점검 (M2, sim 불필요)

로봇 이더넷 연결 후 서버↔GO2 ROS2 통신을 점검:

```bash
bash scripts/real2sim/r2s_go2/check_go2_comms.sh [iface]
#   iface 생략 시 192.168.123.x 보유 iface 자동탐지(예: ens10f1).
#   ping + 기대 토픽(/lowstate,/lowcmd,...) + read_lowstate.py 파싱까지 PASS/FAIL.
```

- `read_lowstate.py` — `/lowstate` 구독해 motor q/dq, imu quat, foot_force 출력(실측 ~500Hz).
- `safe_joint_test.py` — 엎드린 상태에서 한 관절만 조심스럽게 흔드는 안전 테스트(`--dry-run` 지원).

> daemon이 stale 토픽 캐시를 들면 로봇 토픽이 안 보인다 — `check_go2_comms.sh`는 discovery 전에
> `ros2 daemon`을 재시작해 이 함정을 피한다.

---

## 8. 옵션 / 환경변수

| 변수 / 플래그 | 기본 | 의미 |
|---|---|---|
| `--fix_base` / `FIX_BASE` | 지면(0) | base를 공중(z=0.5)에 고정 — 접촉/균형 없이 관절 추종만 볼 때 |
| `--headless` | off | 렌더 창 없이 실행(원격/자동화) |
| `--num_envs` | 1 | 시뮬 환경 수(브릿지는 env0만 사용) |
| `R2S_SHARED` | 0 | 1이면 브릿지/GUI를 cyclonedds로 전환 → 실로봇과 같은 도메인 |
| `R2S_ROBOT_IFACE` | 자동탐지 | 로봇 네트워크 인터페이스(예: `ens10f1`) |
| `R2S_SIM_STATE_TOPIC` | 모드별 자동 | Monitor의 sim 소스 토픽(shared=`/sim/lowstate`, sim전용=`/lowstate`) |
| `R2S_ROBOT_STATE_TOPIC` | `/lowstate` | Monitor의 robot 소스 토픽 |
| `ROS_DOMAIN_ID` | 0 | 모든 ROS2 터미널이 동일해야 discovery됨 |
| `GPU` | 2 | sim_runner CUDA 디바이스 |
| `R2S_LOG_BASEZ` | off | 1이면 sim_runner가 base_z + mean thigh 디버그 로그 출력 |

---

## 9. 트러블슈팅

- **ros2 노드가 서로 안 보임 / `/lowstate` 안 옴**: 모든 터미널의 `ROS_DOMAIN_ID`와
  `RMW_IMPLEMENTATION`이 일치해야 한다. `echo $ROS_DOMAIN_ID $RMW_IMPLEMENTATION`로 확인.
  sim 전용은 기본 FastDDS, shared는 CycloneDDS — 섞이면 안 보인다.
- **sim_runner가 `ModuleNotFoundError`로 죽음**: `./isaaclab.sh -p`가 base conda를 가리키는 경우.
  `conda activate isaac-6.0` 후 실행하거나 `r2s_sim` 사용.
- **`/lowstate`는 오는데 로봇이 안 움직임**: 지면에 떨어졌거나 관절 순서 어긋남 → `--fix_base`로 격리 재확인.
- **Monitor 창이 안 뜸/에러**: 디스플레이가 있는 환경인지, 이미 실행 중인지 확인(중복 실행 방지됨).
- **엎드린 자세에서 sim≠robot, 목표 미도달**: 버그 아님 — kp=25가 약해 뒷 hip이 벌어지고 calf는
  기계 한계에 박힌다(접촉). Real2Sim gap의 실측 신호. 자세한 분석은 `CONTRACT.md` 참고.

---

## 10. 파일

| 파일 | 역할 |
|---|---|
| `sim_runner_go2.py` (상위 폴더) | Isaac 런처 + UDP 루프 |
| `sim_bridge.py` | `/lowcmd`↔UDP↔`/lowstate` 브릿지(rclpy) |
| `gui_controller.py` | PyQt5 GUI, `/lowcmd` 50Hz 발행 |
| `monitor.py` | 실시간 plot(별도 프로세스) |
| `motions.py` | 자세 상수 + 보간/스텝/사인 유틸(순수 함수) |
| `lowcmd_crc.py` | LowCmd CRC/헤더(실기 deploy) |
| `r2s_udp.py` | UDP 패킷 스키마(공유, stdlib만) — 재구현 금지 |
| `read_lowstate.py` / `safe_joint_test.py` | 실로봇 상태 읽기 / 안전 관절 테스트 |
| `check_go2_comms.sh` | 실로봇 통신 점검 |
| `run_*.sh` / `r2s_commands.sh` | 원커맨드 런처 + 별칭 |
| `robot_env.sh` / `example_env.sh` | ROS2/로봇 env 소싱 헬퍼 |

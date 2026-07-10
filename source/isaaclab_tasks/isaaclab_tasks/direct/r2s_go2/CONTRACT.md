# R2S-GO2 인터페이스 계약 (Milestone 1)

> **이 문서는 모든 컴포넌트가 반드시 따라야 하는 고정 계약이다.**
> 병렬 구현 시 조인트 순서 / 토픽명 / 패킷 스키마가 어긋나면 통합이 전부 깨진다.
> 변경은 lead 승인 후 이 문서를 먼저 갱신한다.

## 0. 목표 (Milestone 1)

GUI 버튼 → ROS2 토픽(`/lowcmd`) → sim_bridge → UDP → Isaac Sim GO2 로봇이 움직이고,
sim 상태가 역방향(UDP → sim_bridge → `/lowstate` 토픽)으로 발행되는 **한 바퀴 통신 검증**.

- **범위 포함**: sim 환경, sim_runner(Isaac), sim_bridge(ROS2), GUI(ROS2)
- **범위 제외 (M2)**: 실제 로봇 노드, sim↔real 비교/플롯 노드, CycloneDDS RMW 전환

## 1. 실행 환경 (반드시 구분)

| 프로세스 | Python | 소싱 | 실행 |
|---|---|---|---|
| `sim_runner_go2.py` | Isaac conda **3.12** | 없음 | `./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py` |
| `sim_bridge.py` | 시스템 **3.10** | humble + unitree_go overlay | `/usr/bin/python3 scripts/real2sim/r2s_go2/sim_bridge.py` |
| `gui_controller.py` | 시스템 **3.10** | humble + unitree_go overlay | `/usr/bin/python3 scripts/real2sim/r2s_go2/gui_controller.py` |

> **함정**: conda가 활성화된 셸에서는 `python3`가 conda(3.12)를 가리켜 rclpy 임포트가 깨진다.
> ROS2 노드는 반드시 `/usr/bin/python3`(3.10) + `source /opt/ros/humble/setup.bash` +
> `source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash` 로 실행한다.
> Isaac(3.12)에서는 rclpy 임포트 불가 → sim은 UDP만 사용(rclpy/zmq 불필요, stdlib socket).

## 2. 조인트 순서 (Unitree GO2 표준, 불변)

`unitree_go/MotorCmd[20]`의 인덱스 0~11이 GO2 12개 다리 관절이다. **이 순서가 유일한 진실이다.**

| idx | 관절 (Isaac USD 이름) | idx | 관절 |
|---|---|---|---|
| 0 | `FR_hip_joint`   | 6  | `RR_hip_joint`   |
| 1 | `FR_thigh_joint` | 7  | `RR_thigh_joint` |
| 2 | `FR_calf_joint`  | 8  | `RR_calf_joint`  |
| 3 | `FL_hip_joint`   | 9  | `RL_hip_joint`   |
| 4 | `FL_thigh_joint` | 10 | `RL_thigh_joint` |
| 5 | `FL_calf_joint`  | 11 | `RL_calf_joint`  |

인덱스 12~19는 미사용(0으로 채움). sim은 위 12개 이름을
`find_joints(JOINT_ORDER, preserve_order=True)`로 매핑하고 개수/이름을 assert 검증한다.

## 3. ROS2 토픽 계약

| 토픽 | 타입 | 방향 | 발행자 → 구독자 |
|---|---|---|---|
| `/lowcmd`   | `unitree_go/msg/LowCmd`   | 명령 | GUI → sim_bridge (M2: + 실로봇) |
| `/lowstate` | `unitree_go/msg/LowState` | 상태 | sim_bridge → (M2: 비교 노드) |

- `LowCmd.motor_cmd[i]` (i=0..11): `q`(목표각 rad), `dq`(목표각속도 rad/s), `tau`(피드포워드 Nm), `kp`, `kd`, `mode=0x01`.
- `LowState.motor_state[i]` (i=0..11): `q`, `dq`, `tau_est`, `ddq`.
- `LowState.imu_state`: `quaternion`(w,x,y,z), `gyroscope`, `accelerometer`, `rpy`.
- 미사용 필드(sn/version/bms 등)는 0. **M1 sim 경로는 CRC/head를 검증하지 않는다**(sim_bridge가 무시).
  단, gui_controller는 실기 호환을 위해 head/level_flag/crc를 채워 발행한다 — §8 참고. sim에는 inert.

## 4. UDP 패킷 스키마 (sim_bridge ↔ sim_runner, 두 Python 세계의 경계)

- 전송: **UDP, localhost**. 구조체 리틀엔디안. 공유 정의는 `scripts/real2sim/r2s_go2/r2s_udp.py`
  (순수 stdlib — Isaac 3.12 와 시스템 3.10 양쪽에서 임포트 가능, ros/isaac 의존성 없음).
- **최신값 우선(latest-wins)**: sim_runner의 recv는 **non-blocking**, 루프마다 큐를 비우고 마지막 패킷만 사용.
  sim step 루프는 절대 recv로 블로킹되지 않는다.

### 4.1 명령 패킷 (sim_bridge → sim_runner), 포트 `9871`

```
magic   : uint32  = 0x52324743  ("R2GC")
seq     : uint32             # 증가 시퀀스
per motor i=0..11 (총 12 × 5 float32 = 60):
    q_i, dq_i, kp_i, kd_i, tau_i   (float32)
```
`struct` 포맷: `"<II" + "5f"*12` → 총 8 + 240 = 248 bytes.

### 4.2 상태 패킷 (sim_runner → sim_bridge), 포트 `9872`

```
magic   : uint32  = 0x52324753  ("R2GS")
seq     : uint32
sim_time: float32            # 초
per motor i=0..11 (총 12 × 4 float32 = 48):
    q_i, dq_i, ddq_i, tau_est_i  (float32)
imu (10 float32):
    quat_w, quat_x, quat_y, quat_z,
    gyro_x, gyro_y, gyro_z,
    acc_x, acc_y, acc_z
```
`struct` 포맷: `"<IIf" + "4f"*12 + "10f"` → 12 + 192 + 40 = 244 bytes. <!-- codespell:ignore -->


## 5. sim 환경 계약 (`Isaac-R2S-Go2-v0`)

- RL 없음. `DirectRLEnv` 기반, reward/done stub (r2s_hind_leg 패턴).
- `set_setpoint(q, dq, kp, kd, tau)`: 12-벡터 주입. **slew rate limiter** 적용(토크 스파이크 방지).
  - M1 기본: `q`를 position target으로 cfg PD(kp=25,kd=0.5) 적용. `kp/kd/tau`는 버퍼에 저장(향후 faithful 모드 seam).
- `get_lowstate()`: 12관절 `(q,dq,ddq,tau_est)` + base IMU(quat,gyro,acc,rpy) numpy 반환. `num_envs==1`.
- **초기 자세 = prone(엎드림)**: 실기 GO2가 엎드려 시작하므로 sim도 `PRONE_JOINT_POS`(=go2_stand_example
  target_pos_1 = `motions.STAND_FOLDED`)로 spawn. base 스폰 높이 `PRONE_HEIGHT_M=0.10`(실측 정착 base_z≈0.088).
  `_reset_idx`가 이 prone을 default로 홀드 → 버튼 전까지 엎드림 유지.
- **GUI "Stand Up" 버튼**: `go2_stand_example` 궤적(현재→FOLDED→UP, 마지막 splay 생략)으로 기립.
  `motions.stand_up_sequence()`. sim 실측: kp=25로 base_z 0.088→0.270 기립 확인(최종 thigh는 명령 0.67 vs 실측 0.463 — Real2Sim steady-state gap, 튜닝 대상).
- **`--fix_base` 토글**: 공중 고정 spawn. cfg에서 base를 고정(fixed root)하거나 자유낙하 선택.
  - fix_base=True: 로봇 base를 공중 고정 높이(z=`FIXED_BASE_HEIGHT_M`=0.5)에 fixed articulation root로 spawn(관절은 prone).
  - fix_base=False: 지면 위 prone spawn(z=`PRONE_HEIGHT_M`=0.10, 기본).
- 새 버퍼는 `_reset_idx`에서 초기화 (전역 DO 규칙).
- `sim.dt=1/200`, `decimation=4` → 제어 50Hz. `episode_length_s` 길게(조기종료 없음).

## 6. GO2 물성/액추에이터 (튜닝 대상 — UNITREE_GO2_CFG 기준)

- 액추에이터: `DCMotorCfg` base_legs. `stiffness=25, damping=0.5, effort_limit=23.5,
  saturation_effort=23.5, velocity_limit=30, armature=0.01`.
- **real2sim 튜닝 대상**: 이 PD gain, effort/velocity limit, armature, mass/inertia, friction.
  M1에서는 손대지 않고 통신만 검증. 튜닝은 후속.

## 7. 검증 기준 (M1 완료 게이트)

1. `sim_runner_go2.py` 기동 → GO2 로드, UDP 9871 수신 대기.
2. `sim_bridge.py` 기동 → `/lowcmd` 구독, `/lowstate` 발행.
3. `gui_controller.py` 기동 → 버튼 클릭 시 `/lowcmd` 발행.
4. `ros2 topic echo /lowcmd` 로 명령 확인, `ros2 topic echo /lowstate` 로 상태 확인.
5. GUI "기본자세" 버튼 → sim GO2가 default pose로 이동(육안/상태값 확인).
6. GUI "관절 스텝" 버튼 → 해당 관절이 움직이고 `/lowstate`의 q가 추종.

## 8. 실기 Deploy 준비 (M2 — `/lowcmd` 실로봇 호환)

M1은 sim 전용(§0)이라 `/lowcmd`의 CRC/헤더를 검증하지 않았다. **동일한 `/lowcmd`를 실기 GO2로
그대로 보내려면** 아래가 필요하다. sim 경로(sim_bridge → UDP)는 `head`/`crc`/`level_flag`/`mode`를
무시하므로 아래 메시지 필드 항목은 **sim에 inert**하다(회귀 없음).

| 항목 | 요구 | 상태 |
|---|---|---|
| `crc` | `crc32(LowCmd C struct)` (poly 0x04C11DB7, init 0xFFFFFFFF, 202 워드) | ✅ `lowcmd_crc.py` |
| `head` | `[0xFE, 0xEF]` | ✅ gui |
| `level_flag` | `0xFF` (LOWLEVEL) | ✅ gui |
| `motor_cmd[i].mode` | `0x01` (servo/PMSM) | ✅ gui |
| 연속 heartbeat | 명령 스트림 끊김 시 실기 fault → 상시 재발행 필요 | ✅ gui `PUBLISH_HZ=50` 연속 발행 |
| 명령 주파수 | **RL 배포 표준 = 50Hz 연속** (policy 50Hz = sim control 50Hz, decimation=4). 로봇 펌웨어가 내부 고주파 PD 수행. C++ example의 500Hz(2ms)는 스크립트 궤적용이라 RL엔 부적합 | ✅ 50Hz (sim control과 일치) |
| slew limiter | sim 전용 → 실기와 과도응답 다름 | ⚠️ 미정렬 (fidelity seam) |
| sport/motion 서비스 release | 실기 저수준 제어 진입 전제 | ☐ 운영 절차 (M2 노드) |
| RMW = CycloneDDS + 로봇 DDS 도메인 | 실기 `/lowcmd` 연결 | ☐ M2 |

- **CRC 검증(하드웨어 불필요)**: `lowcmd_crc.py`는 unitree_ros2 example의 `motor_crc.cpp`/`.h`를
  바이트단위로 재현한다. 공식 C 코드를 컴파일한 oracle에서 뽑은 정답 벡터(V0/V1/V2)와 self-test가
  일치한다. `python3 scripts/real2sim/r2s_go2/lowcmd_crc.py`.
- **joint index 순서**는 §2가 이미 Unitree 표준(FR-first, idx0=FR_hip)과 일치 — 실기/ sim 공통. 변경 불필요.

## 9. 실기 GO2 ↔ 서버 ROS2 통신 (M2 — **검증 완료 2026-07-10**)

순수 ROS2(rclpy + CycloneDDS)로 실기와 통신한다. **unitree_sdk2(C++) 설치 불필요** — Unitree
DDS가 ROS2 메시지와 직접 호환되므로 `unitree_ros2`의 `unitree_go`/`unitree_api` 메시지만으로
발행/구독한다(이미 humble 빌드됨).

### 9.1 네트워크 (물리 세팅)
- GO2 ↔ 서버 이더넷: 인터페이스 **`ens10f1`**, 서버 static IP **192.168.123.222/24**, 로봇 **192.168.123.161**.
- ping 192.168.123.161 OK 확인.

### 9.2 RMW 환경
- **`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`** + `CYCLONEDDS_URI`에 인터페이스명(`ens10f1`) 지정 필수.
- 소싱 방법(둘 중 하나):
  - `source scripts/real2sim/r2s_go2/robot_env.sh` — humble+overlay+cyclonedds, iface **자동탐지**(192.168.123.x) + conda 완전비활성화.
  - `source /home/lgb/unitree_ros2/setup_local.sh` — 사용자가 humble+ens10f1로 수정해 둠(동일 효과).

### 9.3 검증 도구
- `bash scripts/real2sim/r2s_go2/check_go2_comms.sh` — ping + 기대 토픽(`/lowstate`,`/lowcmd`,`/sportmodestate`,`/lf/lowstate`,`/wirelesscontroller`,`/utlidar/cloud`) + `read_lowstate.py` 파싱까지 PASS/FAIL.
- `read_lowstate.py` — `/lowstate`를 rclpy로 구독해 motor q/dq, imu quat(wxyz), foot_force 파싱. 실측: **~500Hz**, head=0xFE 0xEF.

### 9.4 함정 / M2 주의
- **⚠ ros2 daemon stale 캐시**: daemon이 이전 토픽 캐시를 들고 있으면 로봇 토픽이 안 보인다(실측). 디스커버리 전 `ros2 daemon stop` 필요 — `check_go2_comms.sh`에 내장됨.
- **⚠ RMW 불일치**: 현재 sim 스택(sim_bridge/gui)은 기본 **fastdds**, 로봇은 **cyclonedds**. 서로 못 본다. M2 sim↔real 동시비교 시 **전 노드 cyclonedds 통일** 필요 + `/lowstate` 토픽명 충돌(sim vs real) → 한쪽 remap(예: sim은 `/sim/lowstate`).
- **⚠ rate 차이**: real `/lowstate` ~500Hz vs sim state 50Hz(control) → 비교 시 다운샘플/정렬.
- **☐ 저수준 제어 전제(TODO)**: `/lowcmd`로 모터 제어하려면 GO2 sport(motion) 서비스 release 필요(`/api/motion_switcher/request`, `/api/robot_state/request` — 전부 ROS2 토픽, sdk2 불필요). 제어 직전 **스트레이 `/lowcmd` 발행자 없는지** 확인(안전).

## 10. 동시 구동 (sim + real 한 버튼) — **배선 검증 완료 2026-07-10 (격리 도메인)**

GUI 버튼 한 번의 `/lowcmd`가 **sim과 실로봇 양쪽**에 도달하게 하는 모드.

### 10.1 원리
- `sim_runner`는 **순수 UDP(ROS 없음)** → RMW 무관. RMW 전환은 `sim_bridge`·`gui`만.
- **`R2S_SHARED=1`** → 두 런처가 `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` + `CYCLONEDDS_URI`(ens10f1)로 전환 → 실로봇과 같은 RMW/도메인.
- gui가 `/lowcmd` 발행 → **sim_bridge와 실로봇이 동시에 구독** → 둘 다 같은 명령 수신.
- **`/lowstate` 충돌 회피(remap)**: sim_bridge는 `R2S_STATE_TOPIC=/sim/lowstate`로 발행(shared 시 자동). 실로봇은 `/lowstate` 그대로. `/lowcmd`는 remap 안 함(공유가 목적). 비교 노드는 `/lowstate`(real)+`/sim/lowstate`(sim) 둘 다 구독.

### 10.2 실행 (도메인 0 = 로봇 도메인)
| 터미널 | 명령 | 비고 |
|---|---|---|
| 1 sim | `FIX_BASE=0 r2s_sim` | 지면. UDP라 SHARED 불필요 |
| 2 bridge | `R2S_SHARED=1 r2s_udp` | cyclonedds 도메인0 + `/sim/lowstate` |
| 3 gui | `R2S_SHARED=1 r2s_gui` | cyclonedds 도메인0 ⚠ **실로봇도 /lowcmd 수신** |

### 10.3 ⚠ 안전 / 한계
- **gui가 t=0부터 STAND_FOLDED(prone) kp=25를 연속 발행** → shared gui 시작 즉시 실로봇도 명령 받음. **실로봇을 안전 위치(prone/매달기) + sport release 상태로 둔 뒤** shared gui를 켤 것.
- **게인 한계**: gui는 kp=25 발행. sim은 cfg PD(25)로 기립하지만 **실로봇은 `/lowcmd`의 kp=25를 직접** 써서 go2_stand_example의 kp=60보다 약함 → **실로봇은 kp=25로 못 일어설 수 있음**. 실로봇 기립엔 per-motion 게인(stand=60) 필요(후속).
- 검증(2026-07-10): 격리 도메인 42에서 cyclonedds same-host 디스커버리 + remap + sim 기립 PASS. 실로봇 도메인0 미개입.

## 11. Monitor (선택 모터 action/sim/robot 실시간 plot) — 검증 완료 2026-07-10

gui "Monitor" 버튼 → `monitor.py`를 **별도 프로세스**로 spawn. 선택 모터의 **q / tau_est / dq** 세 plot(matplotlib Qt5Agg 임베드).

### 11.1 데이터 소스 (모터 i)
- **action**: `/lowcmd` `motor_cmd[i].q` — gui가 발행 중인 목표각(q plot에 점선 오버레이).
- **sim**: `R2S_SIM_STATE_TOPIC` `motor_state[i]` (q/dq/tau_est).
- **robot**: `R2S_ROBOT_STATE_TOPIC`(기본 `/lowstate`) `motor_state[i]`.
- 토픽 해석: shared → sim=`/sim/lowstate`, robot=`/lowstate`(분리). 비-shared(sim 전용) → 둘 다 `/lowstate` → monitor가 **robot 구독 dedup**(sim only 표시). `run_gui_controller.sh`가 `R2S_SIM_STATE_TOPIC`을 모드별로 export.

### 11.2 왜 별도 프로세스인가 (heartbeat 보호, 핵심)
- monitor를 gui와 **같은 Qt event loop**에 두면 실제 디스플레이의 무거운 matplotlib draw가 50Hz `/lowcmd` 발행 타이머(§8 heartbeat)를 막아 gap이 생긴다. matplotlib/Qt가 GIL을 쥐므로 **background thread로도 못 푼다** → 진짜 격리는 **별도 프로세스**뿐.
- gui 버튼은 `subprocess.Popen([sys.executable, monitor.py])`로 spawn만. env(RMW/`ROS_DOMAIN_ID`/`CYCLONEDDS_URI`) 상속 → 같은 토픽 관측. gui 종료 시 자식 프로세스 terminate.
- 프로세스 내부: 수집/렌더(15Hz) 분리 + **decimate**(실로봇 500Hz → `DECIM_HZ`=60Hz, QoS depth=1 latest-wins). 렌더는 `draw_idle()` full redraw — blit 은 쓰지 않는다(별도 프로세스라 heartbeat 무관이고, blit on-screen 합성이 일부 X 환경에서 창을 까맣게 남기는 문제가 있었음).

### 11.3 검증 (2026-07-10, sim-free · 실로봇 미개입)
판별 프로브(실제 X=Xvfb, dual-topic 500Hz)로 in-process vs 별도-프로세스 대조:

| 지표 | in-process (구) | **별도 프로세스 (현)** |
|---|---|---|
| `/lowcmd` rate | 48.8Hz | **50.1Hz** |
| gaps >30ms | 12 | **0** |
| maxgap | 54.9ms | **21.3ms** |

- decimation: robot 500Hz 유입 → 버퍼 ~51Hz(≤60Hz), 데이터 정상. sim 50Hz 그대로.
- 결론: heartbeat가 렌더에서 완전 격리됨(gui 프로세스는 발행만, 렌더 0).

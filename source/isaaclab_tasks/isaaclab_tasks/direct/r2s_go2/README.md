# R2S-GO2 — Real2Sim Milestone 1

> **⚠️ 이 환경은 강화학습(RL)이 아닙니다.**
> `agents/` 디렉토리가 없으며 표준 `train.py` / `play.py`로 실행할 수 없습니다.
> 전용 실행 스크립트(`scripts/real2sim/sim_runner_go2.py`)를 사용해야 합니다.

## 개요

Unitree GO2 로봇을 IsaacLab 시뮬레이터에 올리고, ROS2 GUI 컨트롤러(`/lowcmd`) →
sim_bridge → UDP 경로로 관절 목표를 실시간 주입하는 **Real2Sim Milestone 1** 환경입니다.

- 목적: GUI 버튼 → ROS2 → UDP → sim → UDP → ROS2 한 바퀴 통신 검증
- 제어 방식: PD 포지션 제어 + slew rate 리미터 (토크 스파이크 방지)
- 통신: **UDP** (stdlib socket, ROS2/rclpy는 sim 프로세스에 임포트되지 않음)
- RL 없음 · reward 없음 · gym 학습 루프 없음

자세한 계약(조인트 순서, 토픽 스키마, UDP 패킷 포맷)은
`source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md` 참고.

## 등록 정보

| 항목 | 값 |
|------|----|
| Task ID | `Isaac-R2S-Go2-v0` |
| 환경 클래스 | `R2SGo2Env` |
| 설정 클래스 | `R2SGo2EnvCfg` |

## 실행 방법

세 프로세스가 각각 다른 Python 인터프리터로 실행됩니다 (CONTRACT §1).

```bash
# 터미널 1 — 시뮬레이션 (Isaac conda 3.12)
./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py --num_envs 1
# 공중 고정(base fixed) 모드로 기동하려면:
./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py --num_envs 1 --fix_base

# 터미널 2 — ROS2 sim_bridge (시스템 3.10 + humble)
/usr/bin/python3 scripts/real2sim/r2s_go2/sim_bridge.py

# 터미널 3 — GUI 컨트롤러 (시스템 3.10 + humble)
/usr/bin/python3 scripts/real2sim/r2s_go2/gui_controller.py
```

> **함정**: conda 활성화 셸에서 `python3`는 conda(3.12)를 가리켜 rclpy 임포트가 깨진다.
> ROS2 노드는 `/usr/bin/python3` + humble 소싱으로 실행할 것 (CONTRACT §1).

## 시스템 구조

```
[gui_controller.py]  /lowcmd  [sim_bridge.py]  UDP:9871  [sim_runner_go2.py]
   (ROS2, 3.10)     ────────→   (ROS2, 3.10)   ────────→   (Isaac, 3.12)
                     /lowstate                  UDP:9872
                    ←────────                  ←────────
```

- UDP 9871: sim_bridge → sim_runner, 명령 패킷 (q, dq, kp, kd, tau × 12)
- UDP 9872: sim_runner → sim_bridge, 상태 패킷 (q, dq, ddq, tau_est × 12 + imu 10)
- **latest-wins**: sim_runner의 recv는 non-blocking, 매 루프 큐를 비우고 마지막 패킷만 사용

## 조인트 순서 (CONTRACT §2, 불변)

| idx | 관절 | idx | 관절 |
|---|---|---|---|
| 0 | `FR_hip_joint`   | 6  | `RR_hip_joint`   |
| 1 | `FR_thigh_joint` | 7  | `RR_thigh_joint` |
| 2 | `FR_calf_joint`  | 8  | `RR_calf_joint`  |
| 3 | `FL_hip_joint`   | 9  | `RL_hip_joint`   |
| 4 | `FL_thigh_joint` | 10 | `RL_thigh_joint` |
| 5 | `FL_calf_joint`  | 11 | `RL_calf_joint`  |

`find_joints(JOINT_ORDER, preserve_order=True)`로 매핑하며 개수/순서를 assert 검증합니다.

## 시뮬레이션 파라미터

| 항목 | 값 |
|------|-----|
| Physics 주파수 | 200 Hz |
| 제어 주파수 | 50 Hz (decimation = 4) |
| 에피소드 길이 | 600 s (10분, 조기 종료 없음) |
| Slew rate | `V_MAX_RAD[i] / 50` rad/step (velocity_limit=30 rad/s 기준, 전 관절 동일) |
| PD 게인 | kp=25, kd=0.5 (`UNITREE_GO2_CFG` DCMotorCfg 그대로, CONTRACT §6) |

## `--fix_base` 토글

`R2SGo2EnvCfg(fix_base=True)`를 넘기면 base가 world에 `FixedJoint`로 고정되어
z=0.5에 공중 부양한 채로 다리만 움직입니다. 관절 파라미터 튜닝/단일 다리 검증에 유용합니다.

- 구현: `ArticulationRootBaseCfg.fix_root_link=True` (solver-common, `isaaclab.sim.schemas`)
- `fix_base=False`(기본): 지면 위 자유 spawn, z=0.27

## 외부 인터페이스

```python
env = gym.make("Isaac-R2S-Go2-v0", cfg=R2SGo2EnvCfg(fix_base=False)).unwrapped

# PD 목표 주입 — q만 slew limiter 통과 후 실제 적용, dq/kp/kd/tau는 버퍼 저장(seam)
env.set_setpoint(q=[...12], dq=[...12], kp=[...12], kd=[...12], tau=[...12])

env.step(...)  # actions 인자는 무시됨

state = env.get_lowstate()
# {"q": (12,), "dq": (12,), "ddq": (12,), "tau_est": (12,), "imu": (10,)}  numpy, num_envs==1
```

`imu` 순서: `quat_w, quat_x, quat_y, quat_z, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z`.

## 관련 파일

```
scripts/real2sim/
├── sim_runner_go2.py       # Isaac Sim 진입점 (터미널 1)
└── r2s_go2/
    ├── sim_bridge.py       # ROS2 ↔ UDP 브릿지 (터미널 2)
    ├── gui_controller.py   # ROS2 GUI 컨트롤러 (터미널 3)
    └── r2s_udp.py          # UDP 패킷 스키마 (stdlib, ROS2/Isaac 양쪽에서 임포트)
```

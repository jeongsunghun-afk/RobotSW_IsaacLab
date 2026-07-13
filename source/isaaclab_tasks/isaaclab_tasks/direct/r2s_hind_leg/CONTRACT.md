# R2S-HindLeg 계약 (CONTRACT)

R_Skeleton 뒷다리(고정베이스 5-DOF)를 Isaac Sim에 올리고 순수 UDP로 GUI/monitor와
통신하는 Real2Sim 환경의 고정 계약. r2s_go2 패턴 기반.

## 0. 목표

GUI(PyQt5) → UDP → sim_runner(Isaac GO2 대신 hind leg) → UDP → GUI → monitor 로 이어지는
**위치/PD 제어 + 실시간 시각화** 파이프라인. RL 없음.

## 1. 실행 환경 (반드시 구분)

| 프로세스 | Python | 비고 |
|---|---|---|
| `sim_runner_hindleg.py` | Isaac conda `isaac-6.0` (3.12) | UDP만 사용, rclpy/ROS 금지 |
| `gui_controller.py` | 시스템 `/usr/bin/python3` (3.10, PyQt5) | 순수 UDP, ROS 비의존 |
| `monitor.py` | 시스템 3.10 (matplotlib+PyQt5) | GUI가 별도 프로세스로 spawn |

## 2. 조인트 순서 (불변, 실측 2026-07-13)

`find_joints(preserve_order=True)` + `JOINT_NAME_PATTERNS`로 고정. USD 로드 순서 독립.

| idx | 실측 관절명 | 레이블 | soft limit [rad] | KP | KD | v_max | τ_max |
|---|---|---|---|---|---|---|---|
| 0 | HL_joint2_thigh_r | thigh_r | ±2.826 | 300 | 5 | 41.0 | 22.0 |
| 1 | HL_joint3_thigh_p | thigh_p | ±1.413 | 300 | 5 | 25.0 | 53.0 |
| 2 | HL_joint4_knee_p | knee_p | ±1.413 | 300 | 5 | 25.0 | 53.0 |
| 3 | HL_joint5_ankle_p | ankle_p | ±1.413 | 100 | 5 | 51.0 | 48.0 |
| 4 | HL_joint6_toe_p | toe_p | ±1.413 | 100 | 5 | 51.0 | 48.0 |

- 고정베이스(USD `R_Skeleton_Hind_Leg_Fixed_Filpped`, init z=0.6, 실측 drift 0.000) → IMU/base 상태 없음.
- default_joint_pos 전부 0.0 (중립 자세).

## 3. 전송 (현재: 순수 UDP)

```
gui ──cmd(9873)──▶ sim_runner ──state(9874)──▶ gui ──relay(9875)──▶ monitor
```

- **ROS2는 향후 사용 예정이나 메시지 형식 미정** → 지금은 순수 UDP로 두고, ROS2 브릿지는 seam으로 남긴다.
  ROS2 도입 시: sim_runner의 UDP 경계는 유지하고, gui↔(bridge) 사이만 ROS2로 교체하거나 monitor를
  직접 구독으로 전환한다. `set_setpoint`/`get_lowstate` env 인터페이스는 불변.
- go2의 unitree_go `LowCmd/LowState`(20모터 GO2 전용 + CRC)는 이 5-DOF 커스텀 다리에 **재사용 불가**.

## 4. UDP 패킷 스키마 (`scripts/real2sim/r2s_hind_leg/r2s_udp.py`, stdlib만)

| 패킷 | 포트 | magic | 구조 | 크기 |
|---|---|---|---|---|
| cmd (gui→sim) | 9873 | R2HC | magic,seq + 5×(q,dq,kp,kd,tau) | 108 B |
| state (sim→gui) | 9874 | R2HS | magic,seq,sim_time + 5×(q,dq,ddq,tau_est) | 92 B |
| monitor (gui→mon) | 9875 | R2MN | magic,seq + 5×action_q + 5×(sim_q,sim_dq,sim_tau) | 88 B |

- 모두 little-endian float32. IMU 없음(고정베이스).
- latest-wins: sim_runner recv는 non-blocking, 매 스텝 큐를 비우고 마지막 명령만 반영.
- sim_runner는 state를 **마지막 cmd 발신자**의 IP:9874로 회신(gui가 9874 bind).

## 5. sim 환경 계약 (`Isaac-R2S-HindLeg-v0`)

- `set_setpoint(q, dq, kp, kd, tau)`: PD 목표 주입. q는 slew rate limiter 통과 후 position target.
- `get_lowstate()`: 5관절 `(q, dq, ddq, tau_est)` numpy 반환. num_envs==1. IMU 없음.
- **faithful PD** (`cfg.faithful_pd=True`, 기본): set_setpoint의 kp/kd를 `write_joint_stiffness/damping_to_sim`으로
  실제 sim drive 게인에 반영 → GUI의 kp/kd 슬라이더가 살아있음. False면 cfg 액추에이터 PD 고정(go2 seam 방식).
- slew rate: `max_step[i] = V_MAX_RAD[i] / 50`. 전 관절 개별.
- 새 버퍼는 `_reset_idx`에서 초기화 필수 (default_pos, kp/kd=DEFAULT).

## 6. 게인/한계 중복 정의 (일치 필수)

관절 게인·soft limit은 두 패키지에 중복 정의된다 — 값 일치 필수:

- `source/.../r2s_hind_leg/r2s_hind_leg_env_cfg.py`: DEFAULT_KP/KD, SOFT_LIMITS_RAD, DEFAULT_POSE
- `scripts/real2sim/r2s_hind_leg/motions.py`: 동일 상수

GUI는 명령을 SOFT_LIMITS_RAD로 클램프한다 — sim이 `soft_joint_pos_limit_factor=0.9`로 position
target을 silently 클램프하는 것을 방지(추종 혼란 차단).

## 7. 검증 기준 (완료 2026-07-13, sim-free 게이트 + 실제 Isaac)

- env smoke: `get_lowstate` shape (5,), faithful PD write 반영(kp=200→joint_stiffness 200), q 추종 오차 <0.002.
- monitor 렌더: Xvfb + `screen.grabWindow` 실픽셀 캡처 PASS(action/sim/tau/dq 정상).
- GUI 배선: publish/recv/relay 50Hz PASS(cmd 123, relay 123, 상태 수신).
- 실제 Isaac UDP E2E: sim_runner 기동 → 드라이버 step 명령 → sim q 추종 오차 <0.002, state 466 수신.

## 11. Monitor (별도 프로세스)

GUI "Monitor" 버튼 → `monitor.py`를 별도 프로세스로 spawn. 선택 관절의 q(action/sim) / tau / dq plot.

- **별도 프로세스 이유**: matplotlib 렌더를 GUI event loop에 두면 50Hz UDP 발행을 굶긴다(r2s_go2에서 확인).
  렌더는 `draw_idle()` full redraw — blit은 쓰지 않음(일부 X 환경에서 창을 까맣게 남기는 문제).
- 데이터: GUI가 중계하는 monitor UDP 패킷(action+sim, time-aligned). robot 시리즈는 ROS2 실로봇 연동 시 추가.

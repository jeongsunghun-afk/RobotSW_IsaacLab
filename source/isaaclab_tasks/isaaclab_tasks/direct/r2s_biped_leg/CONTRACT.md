# R2S-BipedLeg 계약 (CONTRACT)

8-DOF 2족 다리(`HIND_LEG_CFG`, `direct/hind_leg`가 쓰는 것과 동일 로봇)를 Isaac Sim에 올리고
순수 UDP로 GUI/monitor와 통신하는 Real2Sim 환경의 고정 계약. `r2s_hind_leg` 패턴 기반.

`r2s_hind_leg`(R_Skeleton 5-DOF)는 **별개 패키지로 무손상 유지**된다 — 이 패키지는 추가일 뿐이다.

## 0. 목표

GUI(PyQt5) → UDP → sim_runner_bipedleg → UDP → GUI → monitor 로 이어지는
**위치/PD 제어 + 실시간 시각화** 파이프라인. RL 없음.

## 1. 실행 환경 (반드시 구분)

| 프로세스 | Python | 비고 |
|---|---|---|
| `sim_runner_bipedleg.py` | Isaac conda `isaac-6.0` (3.12) | UDP만 사용, rclpy/ROS 금지 |
| `gui_controller.py` | 시스템 `/usr/bin/python3` (3.10, PyQt5) | 순수 UDP, ROS 비의존 |
| `monitor.py` | 시스템 3.10 (matplotlib+PyQt5) | GUI가 별도 프로세스로 spawn |

## 2. 조인트 순서 (불변, 실측 2026-07-21 probe)

`find_joints(preserve_order=True)` + `JOINT_NAME_PATTERNS`로 고정. USD 로드 순서 독립.
순서는 **leg-major**(HL 4개 → HR 4개). 정규식이 아니라 **정확한 관절명**을 쓴다 —
`.*_hip_joint` 류는 HL/HR 양쪽에 매칭되어 좌우 순서가 USD 로드 순서에 의존하게 된다.

| idx | 실측 관절명 | 레이블 | KP | KD | v_max | τ_max | soft limit [rad] |
|---|---|---|---|---|---|---|---|
| 0 | HL_hip_joint   | HL_hip   | 65.0 | 6.0 | 29.6 | 28.0 | (-0.5498, 0.5498) |
| 1 | HL_thigh_joint | HL_thigh | 53.0 | 4.8 | 29.6 | 28.0 | (-1.9024, 1.5533) |
| 2 | HL_calf_joint  | HL_calf  | 12.0 | 1.1 | 19.7 | 42.0 | (-1.3836, 0.4236) |
| 3 | HL_foot_joint  | HL_foot  | 20.0 | 1.0 | 14.8 | 56.0 | (-0.4192, 1.4662) |
| 4 | HR_hip_joint   | HR_hip   | 65.0 | 6.0 | 29.6 | 28.0 | (-0.5498, 0.5498) |
| 5 | HR_thigh_joint | HR_thigh | 53.0 | 4.8 | 29.6 | 28.0 | (-1.9024, 1.5533) |
| 6 | HR_calf_joint  | HR_calf  | 12.0 | 1.1 | 19.7 | 42.0 | (-1.3836, 0.4236) |
| 7 | HR_foot_joint  | HR_foot  | 20.0 | 1.0 | 14.8 | 56.0 | (-1.4662, 0.4192) |

- soft limit은 **좌우 비대칭**(foot은 HL/HR이 서로 미러). 대칭이라 가정하고 "고치지" 말 것.
- default_joint_pos 전부 0.0 (중립 자세).
- body_names: base, {HL,HR}_{hip,thigh,calf,foot}_link, {HL,HR}_foot_contact_link.
- 게인이 `r2s_hind_leg`(300/5)보다 훨씬 낮다(12~65) → GUI 슬라이더 상한도 여기에 맞춰야 한다.

## 3. 베이스 (fix_base)

`HIND_LEG_CFG`는 **자유베이스**다. 2족이라 `fix_base=False`면 GUI로 관절을 스텝하는 순간 넘어진다.

- `cfg.fix_base=True` → `robot.spawn.articulation_props.fix_root_link=True` + spawn z=0.8 (`FIXED_BASE_HEIGHT_M`).
- 적용 지점은 `R2SBipedLegEnv._setup_scene` — `sim_runner_bipedleg.py`가 cfg 생성 **후** `--fix_base`로
  덮어쓴 값도 반영되어야 하므로 `__post_init__`이 아니라 env 빌드 시점에 읽는다.
- 런처 기본값은 `FIX_BASE=1`.
- 자유 spawn(`fix_base=False`)은 cfg 자체의 z=0.6을 그대로 쓴다.

## 4. 전송 (현재: 순수 UDP)

```
gui ──cmd(9881)──▶ sim_runner ──state(9882)──▶ gui ──relay(9883)──▶ monitor
```

포트는 `r2s_go2` live/tuner(9871~9876)와 `r2s_hind_leg`(9873~9875)를 피해 9881~9883을 쓴다.

- **ROS2는 향후 사용 예정이나 메시지 형식 미정** → 지금은 순수 UDP로 두고, ROS2 브릿지는 seam으로 남긴다.
  `set_setpoint`/`get_lowstate` env 인터페이스는 불변.
- go2의 unitree_go `LowCmd/LowState`(20모터 GO2 전용 + CRC)는 이 8-DOF 커스텀 다리에 **재사용 불가**.

## 5. UDP 패킷 스키마 (`scripts/real2sim/r2s_biped_leg/r2s_udp.py`, stdlib만)

`r2s_hind_leg/r2s_udp.py`와 동일 구조, `NUM_JOINTS=8`. 모두 little-endian float32, IMU 없음.

- cmd (gui→sim, 9881): magic,seq + 8×(q,dq,kp,kd,tau)
- state (sim→gui, 9882): magic,seq,sim_time + 8×(q,dq,ddq,tau_est)
- monitor (gui→mon, 9883): magic,seq + 8×action_q + 8×(sim_q,sim_dq,sim_tau)

latest-wins: sim_runner recv는 non-blocking, 매 스텝 큐를 비우고 마지막 명령만 반영.
sim_runner는 state를 **마지막 cmd 발신자**의 IP:9882로 회신(gui가 9882 bind).

## 6. sim 환경 계약 (`Isaac-R2S-BipedLeg-v0`)

- `set_setpoint(q, dq, kp, kd, tau)`: PD 목표 주입. q는 slew rate limiter 통과 후 position target.
- `get_lowstate()`: 8관절 `(q, dq, ddq, tau_est)` numpy 반환. num_envs==1. IMU 없음.
- **faithful PD** (`cfg.faithful_pd=True`, 기본): set_setpoint의 kp/kd를 `write_joint_stiffness/damping_to_sim`으로
  실제 sim drive 게인에 반영 → GUI의 kp/kd 슬라이더가 살아있음.
- slew rate: `max_step[i] = V_MAX_RAD[i] / 50`. 전 관절 개별.
- observation 24-dim(pos8+vel8+torque8), action 8-dim(미사용), 200Hz physics / 50Hz control, episode 600 s.
- 새 버퍼는 `_reset_idx`에서 초기화 필수 (default_pos, kp/kd=DEFAULT).

## 7. 게인/한계 중복 정의 (일치 필수)

관절 게인·soft limit은 두 패키지에 중복 정의된다 — 값 일치 필수:

- `source/.../r2s_biped_leg/r2s_biped_leg_env_cfg.py`: DEFAULT_KP/KD, V_MAX_RAD, SOFT_LIMITS_RAD, DEFAULT_POSE
- `scripts/real2sim/r2s_biped_leg/motions.py`: 동일 상수

GUI는 명령을 SOFT_LIMITS_RAD로 클램프한다 — sim이 `soft_joint_pos_limit_factor=0.9`로 position
target을 silently 클램프하는 것을 방지(추종 혼란 차단).

## 8. 검증 상태

- 정적: `py_compile` + `ruff check` PASS (2026-07-21).
- **런타임 게이트 미완**: sim_runner 기동 → GUI로 관절 스텝 → 추종 오차 확인이 남아있다.

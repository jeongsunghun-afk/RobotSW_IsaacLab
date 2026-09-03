# r2s_hind_leg 환경 컨텍스트

## 개요

R_Skeleton 뒷다리(고정베이스 5-DOF)를 Isaac Sim에 올리고 순수 UDP로 PyQt5 GUI/monitor와
통신하는 **Real2Sim** 환경. r2s_go2 패턴 기반. RL 없음 — 순수 위치/PD 제어 테스트.

고정 계약은 이 디렉토리의 `CONTRACT.md` 참고.

## 등록 정보

| 항목 | 값 |
|------|-----|
| 환경 ID | `Isaac-R2S-HindLeg-v0` |
| 환경 클래스 | `R2SHindLegEnv` |
| 설정 클래스 | `R2SHindLegEnvCfg` |

## 조인트 파라미터 (실측 2026-07-13)

5관절, `find_joints(JOINT_NAME_PATTERNS, preserve_order=True)`로 순서 고정.

| idx | 실측명 | 레이블 | KP | KD | v_max | τ_max | soft limit [rad] |
|---|---|---|----|----|-------|-------|---|
| 0 | HL_joint2_thigh_r | thigh_r | 300 | 5 | 41.0 | 22.0 | ±2.826 |
| 1 | HL_joint3_thigh_p | thigh_p | 300 | 5 | 25.0 | 53.0 | ±1.413 |
| 2 | HL_joint4_knee_p | knee_p | 300 | 5 | 25.0 | 53.0 | ±1.413 |
| 3 | HL_joint5_ankle_p | ankle_p | 100 | 5 | 51.0 | 48.0 | ±1.413 |
| 4 | HL_joint6_toe_p | toe_p | 100 | 5 | 51.0 | 48.0 | ±1.413 |

- 고정베이스(USD `R_Skeleton_Hind_Leg_Fixed_Filpped`, init z=0.6) → IMU/base 상태 없음.
- default_joint_pos 전부 0.0.
- soft limit은 실측 USD 값(`JOINT_LIMITS_DEG` 구 표는 설계의도라 실제와 다름 — 실측 사용).

## 시뮬레이션 파라미터

| 항목 | 값 |
|------|-----|
| Physics freq | 200 Hz |
| Control freq | 50 Hz (decimation=4) |
| Episode length | 600 s (조기 종료 없음) |
| Slew rate | `V_MAX_RAD[i] / 50` rad/step (관절별) |

## 통신 구조 (순수 UDP)

```
gui_controller.py ─cmd(9873)─▶ sim_runner_hindleg.py ─state(9874)─▶ gui ─relay(9875)─▶ monitor.py
```

- **ROS2는 향후 사용 예정이나 메시지 형식 미정** → 지금은 순수 UDP, ROS2 브릿지는 CONTRACT §3 seam.
- go2의 unitree_go `LowCmd/LowState`(GO2 전용)는 이 5-DOF 커스텀 다리에 재사용 불가.
- sim 쪽(Isaac 3.12)은 UDP만. rclpy/ros 임포트 금지.
- 패킷 스키마는 `scripts/real2sim/r2s_hind_leg/r2s_udp.py` (stdlib만) + `CONTRACT.md` §4.

## env 인터페이스

- `set_setpoint(q, dq, kp, kd, tau)` — q는 slew 통과 후 position target. faithful_pd=True면 kp/kd 반영.
- `get_lowstate()` — 5관절 `(q, dq, ddq, tau_est)` numpy. IMU 없음.
- **faithful PD**: `write_joint_stiffness/damping_to_sim`으로 GUI kp/kd를 실제 sim 게인에 반영
  (kp/kd 변경 시에만 write). False면 cfg 액추에이터 PD 고정.

## 실행 방법

```bash
# 터미널 1: 시뮬레이션 (Isaac conda)
./isaaclab.sh -p scripts/real2sim/sim_runner_hindleg.py --num_envs 1 [--headless]
#   또는: bash scripts/real2sim/r2s_hind_leg/run_sim_runner.sh

# 터미널 2: GUI (시스템 python)
bash scripts/real2sim/r2s_hind_leg/run_gui_controller.sh
```

운영 상세는 `scripts/real2sim/r2s_hind_leg/README.md`.

## 불변 규칙

공통 불변 규칙: `.claude/rules/r2s.md`

- 게인/soft limit은 `scripts/real2sim/r2s_hind_leg/motions.py`와 값 일치 필수(중복 정의)

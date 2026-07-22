# r2s_biped_leg 환경 컨텍스트

## 개요

8-DOF 2족 다리(`HIND_LEG_CFG`, `direct/hind_leg`와 동일 로봇)를 Isaac Sim에 올리고 순수 UDP로
PyQt5 GUI/monitor와 통신하는 **Real2Sim** 환경. `r2s_hind_leg` 패턴 기반. RL 없음 — 순수 위치/PD 제어 테스트.

`r2s_hind_leg`(R_Skeleton 고정베이스 5-DOF)와는 **별개 패키지**다. 둘 다 유지된다.

고정 계약은 이 디렉토리의 `CONTRACT.md` 참고.

## 등록 정보

| 항목 | 값 |
|------|-----|
| 환경 ID | `Isaac-R2S-BipedLeg-v0` |
| 환경 클래스 | `R2SBipedLegEnv` |
| 설정 클래스 | `R2SBipedLegEnvCfg` |

## 조인트 파라미터 (실측 2026-07-21 probe, 동결)

8관절, `find_joints(JOINT_NAME_PATTERNS, preserve_order=True)`로 순서 고정. leg-major(HL→HR).

| idx | 실측명 | 레이블 | KP | KD | v_max | τ_max | soft limit [rad] |
|---|---|---|----|----|-------|-------|---|
| 0 | HL_hip_joint | HL_hip | 65.0 | 6.0 | 29.6 | 28.0 | (-0.5498, 0.5498) |
| 1 | HL_thigh_joint | HL_thigh | 53.0 | 4.8 | 29.6 | 28.0 | (-1.9024, 1.5533) |
| 2 | HL_calf_joint | HL_calf | 12.0 | 1.1 | 19.7 | 42.0 | (-1.3836, 0.4236) |
| 3 | HL_foot_joint | HL_foot | 20.0 | 1.0 | 14.8 | 56.0 | (-0.4192, 1.4662) |
| 4 | HR_hip_joint | HR_hip | 65.0 | 6.0 | 29.6 | 28.0 | (-0.5498, 0.5498) |
| 5 | HR_thigh_joint | HR_thigh | 53.0 | 4.8 | 29.6 | 28.0 | (-1.9024, 1.5533) |
| 6 | HR_calf_joint | HR_calf | 12.0 | 1.1 | 19.7 | 42.0 | (-1.3836, 0.4236) |
| 7 | HR_foot_joint | HR_foot | 20.0 | 1.0 | 14.8 | 56.0 | (-1.4662, 0.4192) |

- soft limit은 **좌우 비대칭**(foot은 미러) — 실측값이므로 대칭으로 "고치지" 말 것.
- default_joint_pos 전부 0.0.
- 게인이 `r2s_hind_leg`(300/5)보다 훨씬 낮다 → GUI 슬라이더 상한을 낮춰야 조작 가능.
- `JOINT_NAME_PATTERNS`는 정규식이 아니라 정확한 관절명 — `.*_hip_joint`는 HL/HR 양쪽에 매칭된다.

## 베이스 (fix_base)

`HIND_LEG_CFG`는 자유베이스라 2족이 GUI 조작 시 즉시 넘어진다. `cfg.fix_base=True`면
`fix_root_link=True` + spawn z=0.8(`FIXED_BASE_HEIGHT_M`). 적용은 `_setup_scene`에서 —
`sim_runner_bipedleg.py --fix_base`가 cfg 생성 후 덮어쓴 값도 반영해야 하기 때문(r2s_go2 패턴).

## 시뮬레이션 파라미터

| 항목 | 값 |
|------|-----|
| Physics freq | 200 Hz |
| Control freq | 50 Hz (decimation=4) |
| Episode length | 600 s (조기 종료 없음) |
| Slew rate | `V_MAX_RAD[i] / 50` rad/step (관절별) |
| Observation | 24-dim (pos8 + vel8 + torque8) |

## 통신 구조 (순수 UDP)

```
gui_controller.py ─cmd(9881)─▶ sim_runner_bipedleg.py ─state(9882)─▶ gui ─relay(9883)─▶ monitor.py
```

- 포트 9881~9883 — go2 live/tuner(9871~9876)와 hind_leg(9873~9875) 회피.
- **ROS2는 향후 사용 예정이나 메시지 형식 미정** → 지금은 순수 UDP, ROS2 브릿지는 CONTRACT §4 seam.
- sim 쪽(Isaac 3.12)은 UDP만. rclpy/ros 임포트 금지.
- 패킷 스키마는 `scripts/real2sim/r2s_biped_leg/r2s_udp.py` (stdlib만) + `CONTRACT.md` §5.

## env 인터페이스

- `set_setpoint(q, dq, kp, kd, tau)` — q는 slew 통과 후 position target. faithful_pd=True면 kp/kd 반영.
- `get_lowstate()` — 8관절 `(q, dq, ddq, tau_est)` numpy. IMU 없음.
- **faithful PD**: `write_joint_stiffness/damping_to_sim`으로 GUI kp/kd를 실제 sim 게인에 반영
  (kp/kd 변경 시에만 write). False면 cfg 액추에이터 PD 고정.

## 실행 방법

```bash
# 터미널 1: 시뮬레이션 (Isaac conda) — 라이브스트림
LIVESTREAM=2 CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p scripts/real2sim/sim_runner_bipedleg.py \
    --num_envs 1 --fix_base --viz kit
#   --viz 생략 시 헤드리스 (--headless는 deprecated)

# 터미널 2: GUI (시스템 python)
bash scripts/real2sim/r2s_biped_leg/run_gui_controller.sh
```

## 불변 규칙

- 새 버퍼 추가 시 `_reset_idx`에서 초기화 필수 (CLAUDE.md 전역 DO)
- Slew rate limiter 필수 — setpoint 점프 시 토크 스파이크 방지
- `JOINT_NAME_PATTERNS`로 `find_joints(preserve_order=True)` 사용 — USD 로드 순서 독립
- 게인/soft limit은 `scripts/real2sim/r2s_biped_leg/motions.py`와 값 일치 필수(중복 정의)
- `from isaaclab.utils.configclass import configclass` 사용 — `from isaaclab.utils import configclass`는
  6.0에서 `TypeError: 'module' object is not callable`
- `r2s_hind_leg` / `r2s_go2` 및 코어 파일(`source/isaaclab/`) 수정 금지

## Worker 매핑

| 작업 | 담당 |
|------|------|
| 관절 파라미터, 환경 로직 | `obs-worker` |
| 설정값 변경 | `cfg-worker` |
| UDP 브릿지/GUI | 직접 수정 (`scripts/real2sim/r2s_biped_leg/`) |

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

## 조인트 파라미터 (2026-08-11 신규 CAD 리비전 Hind_Leg_URDF2 → 같은 날 URDF3로 재교체)

8관절, `find_joints(JOINT_NAME_PATTERNS, preserve_order=True)`로 순서 고정. leg-major(HL→HR).
asset=`data/Robots/Hind_Leg_URDF3/Hind_Leg/Hind_Leg.usda` (rga.py `HIND_LEG_CFG`).
URDF2는 링크 프레임/축(+Y)이 biped MJCF 모델각 규약과 동일, **좌우 동일 규약(미러 아님)** — URDF3도 동일.
τ_max/v_max는 실기값(RL_INTERFACE.md §6-c), 관절한계는 URDF2/URDF3 공통 자체값(신규 리비전 기준).

- URDF3(원본 `/home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg.urdf`)는 관절명·타입·순서·
  effort/lower/upper가 URDF2와 전부 동일 — 아래 표 값 무수정. base 링크만 `base`→`base_collision`
  리네임(질량 5.617→2.8kg), 다리 링크 10개도 전부 `_link`→`_link_collision` 리네임(코드에서 링크명을
  직접 참조하는 곳 없어 무해). robot 이름이 숫자로 시작(`03_Leg_UFDF_Colision_260617_2`)해 USD prim은
  `tn__03_Leg_UFDF_Colision_260617_2_`로 sanitize됨.

| idx | 실측명 | 레이블 | KP | KD | v_max | τ_max | soft limit [rad] |
|---|---|---|----|----|-------|-------|---|
| 0 | HL_hip_joint | HL_hip | 65.0 | 6.0 | 29.6 | 84.0 | (-0.2340, 0.2340) |
| 1 | HL_thigh_joint | HL_thigh | 53.0 | 4.8 | 29.6 | 84.0 | (-0.9555, 2.1855) |
| 2 | HL_calf_joint | HL_calf | 12.0 | 1.1 | 19.7 | 126.0 | (-0.9445, 0.7745) |
| 3 | HL_foot_joint | HL_foot | 20.0 | 1.0 | 24.6 | 100.8 | (-1.3840, 0.3440) |
| 4 | HR_hip_joint | HR_hip | 65.0 | 6.0 | 29.6 | 84.0 | (-0.2340, 0.2340) |
| 5 | HR_thigh_joint | HR_thigh | 53.0 | 4.8 | 29.6 | 84.0 | (-0.9555, 2.1855) |
| 6 | HR_calf_joint | HR_calf | 12.0 | 1.1 | 19.7 | 126.0 | (-0.9445, 0.7745) |
| 7 | HR_foot_joint | HR_foot | 20.0 | 1.0 | 24.6 | 100.8 | (-1.3840, 0.3440) |

- ⚠ URDF2/URDF3 공통 관절한계는 RL_INTERFACE.md §3 실측표와 **다르다**(예: hip ±14.9° vs ±35°) —
  실기팀 신규 리비전 값으로 그대로 사용. 질량은 URDF2 총 ~16.6 kg → URDF3는 base만 5.617→2.8kg으로
  줄어 총 ~13.8 kg(다리 링크 질량은 URDF2와 사실상 동일). KP/KD(구 모델 I_eff 기반)는 재검토 대상.
- ⚠ 실기 런타임 보호는 τ_max보다 훨씬 낮다: 보고토크 15 N·m 50 ms → limp 래치,
  200 dps(≈3.5 rad/s) 속도 트립 (RL_INTERFACE.md §6-i). 정책이 τ_max까지 쓰면 실기에서 죽는다.
- default_joint_pos 전부 0.0.
- 게인이 `r2s_hind_leg`(300/5)보다 훨씬 낮다 → GUI 슬라이더 상한을 낮춰야 조작 가능.
- `JOINT_NAME_PATTERNS`는 정규식이 아니라 정확한 관절명 — `.*_hip_joint`는 HL/HR 양쪽에 매칭된다.

## 베이스 (fix_base)

`HIND_LEG_CFG`는 자유베이스라 2족이 GUI 조작 시 즉시 넘어진다. `cfg.fix_base=True`면
`fix_root_link=True` + spawn z=0.8(`FIXED_BASE_HEIGHT_M`). 적용은 `_setup_scene`에서 —
`sim_runner_bipedleg.py --fix_base`가 cfg 생성 후 덮어쓴 값도 반영해야 하기 때문(r2s_go2 패턴).

- ⚠ **URDF2/URDF3 asset 재생성 시 주의** (2026-08-11): 신형 URDF USD Converter는 base(URDF3는
  `base_collision`)에 `NewtonArticulationRootAPI`를 같이 붙이는데, `fix_root_link=True`가 RootAPI를
  부모로 옮기며 `UsdPhysics.ArticulationRootAPI`만 제거하므로 Newton API가 `HasAPI`를 계속 참으로 만들어
  "Expected exactly one ArticulationRootAPI ... found 2" 오류가 난다.
  fix = `payloads/Physics/physics.usda`의 base(base_collision) `apiSchemas`에서 `NewtonArticulationRootAPI`
  제거(PhysX 경로에선 미사용). asset을 다시 임포트하면 이 제거를 재적용해야 한다.
- ⚠ **URDF3 foot 관절 velocity limit CAD 퇴행 — asset 재생성 시 재패치 필요** (2026-08-11): URDF3
  원본의 `HL/HR_foot_joint <limit velocity>`가 24.7(URDF2)에서 14.8로 바뀌어 있다. 이 14.8은
  `rga.py` `HIND_LEG_CFG` 주석("구값 foot 14.8은 감속비 오인")에 이미 실기 재조사로 폐기됐다고 기록된
  값과 정확히 일치 — CAD 익스포터 쪽 gear ratio 가정이 미갱신인 것으로 추정(actuator 쪽 실측 확정값은
  24.6이며 DCMotorCfg에 별도 하드코딩돼 URDF와 무관, 영향 없음). 문제는 USD 변환 시
  `payloads/Physics/physx.usda`의 `physxJoint:maxJointVelocity`가 foot 2관절만 847.97754 deg/s
  (14.8 rad/s)로 baked-in되는 것 — asset을 다시 임포트하면 이 두 값을 URDF2 생성본과 동일한
  `1415.2058`(=24.7 rad/s)로 수동 패치해야 한다(hip/thigh/calf는 원래 URDF2와 동일해 무수정).

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
- `get_joint_ieff()` — 관절별 유효 관성 [kg·m²] (generalized mass matrix 대각, leg-major).
  sim_runner가 기동 시 1회 계산해 1Hz로 GUI에 전송(`R2BI` 패킷) — GUI Computed Gains용.
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

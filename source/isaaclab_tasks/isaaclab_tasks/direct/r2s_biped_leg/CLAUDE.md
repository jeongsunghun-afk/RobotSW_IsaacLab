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

## 조인트 파라미터 (2026-08-11 URDF2 → URDF3 → **2026-08-12 URDF3_SignFix**)

8관절, `find_joints(JOINT_NAME_PATTERNS, preserve_order=True)`로 순서 고정. leg-major(HL→HR).
asset=`data/Robots/Hind_Leg_URDF3_SignFix/Hind_Leg_SignFix/Hind_Leg_SignFix.usda` (rga.py `HIND_LEG_CFG`).
τ_max/v_max는 실기값(`scripts/real2sim/r2s_biped_leg/RL_INTERFACE.md` §6-c; 이하 RL_INTERFACE.md).

**SignFix (2026-08-12)**: 실기 통신 실측에서 같은 목표에 **반대로 도는 관절**이 확인돼
(HL: hip/calf/foot, HR: hip/thigh), sim을 실기에 맞추기 위해 URDF3에서 이 5개 관절의
`<axis>` 부호와 `<limit>`([lo,hi]→[-hi,-lo])을 반전한 파생 URDF
(`/home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg_SignFix.urdf`,
저장소 사본 `data/Robots/Hind_Leg_URDF3_SignFix/urdf/Hind_Leg.urdf`)에서 재변환했다.
- 결과: **thigh/calf/foot이 좌우 미러 규약**(soft limit 좌우 다름), hip은 대칭이라 값 동일.
- real_runner `calib_bipedleg.hpp`의 sign은 전 관절 **+1 유지가 정답**(sim이 실기에 맞춰졌으므로).
  min/max_rad 클램프만 반전 관절에서 갱신 → **파이 re-scp+rebuild 필요**.
- ⚠ 구 규약(URDF2/URDF3)으로 학습된 정책·수집 데이터(chirp 포함)는 반전 관절 부호가 달라
  호환되지 않는다.
- 검증: 구·신 자산 동시 스폰, 반전 관절 q_new=−q_old ↔ 발끝 상대위치 일치(케이스1) +
  동일 q에서 불일치(케이스2)로 물리 방향 반전 실증.

- URDF3(원본 `/home/lgb/Dog_Motion_data_3D/robots/Hind_Leg_URDF3/urdf/Hind_Leg.urdf`)는 관절명·타입·순서·
  effort/lower/upper가 URDF2와 전부 동일 — 아래 표 값 무수정. base 링크만 `base`→`base_collision`
  리네임(질량 5.617→2.8kg), 다리 링크 10개도 전부 `_link`→`_link_collision` 리네임(코드에서 링크명을
  직접 참조하는 곳 없어 무해). robot 이름이 숫자로 시작(`03_Leg_UFDF_Colision_260617_2`)해 USD prim은
  `tn__03_Leg_UFDF_Colision_260617_2_`로 sanitize됨.

| idx | 실측명 | 레이블 | KP | KD | v_max | τ_max | soft limit [rad] |
|---|---|---|----|----|-------|-------|---|
| 0 | HL_hip_joint | HL_hip | 100.0 | 5.0 | 29.6 | 84.0 | (-0.2340, 0.2340) |
| 1 | HL_thigh_joint | HL_thigh | 50.0 | 5.0 | 29.6 | 84.0 | (-0.9555, 2.1855) |
| 2 | HL_calf_joint | HL_calf | 112.5 | 11.25 | 19.7 | 126.0 | (-0.9445, 0.7745) |
| 3 | HL_foot_joint | HL_foot | 28.8 | 7.2 | 24.6 | 100.8 | (-1.3840, 0.3440) |
| 4 | HR_hip_joint | HR_hip | 100.0 | 5.0 | 29.6 | 84.0 | (-0.2340, 0.2340) |
| 5 | HR_thigh_joint | HR_thigh | 50.0 | 5.0 | 29.6 | 84.0 | (-0.9555, 2.1855) |
| 6 | HR_calf_joint | HR_calf | 112.5 | 11.25 | 19.7 | 126.0 | (-0.9445, 0.7745) |
| 7 | HR_foot_joint | HR_foot | 28.8 | 7.2 | 24.6 | 100.8 | (-1.3840, 0.3440) |

(KP/KD는 2026-08-18 실기 드라이버 게인의 관절 공간 환산값으로 갱신됨 — 근거:
`r2s_biped_leg_env_cfg.py:78-79`, `rga.py:614-627`.)

- ⚠ URDF2/URDF3 공통 관절한계는 RL_INTERFACE.md §3 실측표와 **다르다**(예: hip ±14.9° vs ±35°) —
  실기팀 신규 리비전 값으로 그대로 사용. 질량은 URDF2 총 ~16.6 kg → URDF3는 base만 5.617→2.8kg으로
  줄어 총 ~13.8 kg(다리 링크 질량은 URDF2와 사실상 동일).
- ⚠ 실기 런타임 보호는 τ_max보다 훨씬 낮다: 보고토크 15 N·m 50 ms → limp 래치,
  200 dps(≈3.5 rad/s) 속도 트립 (RL_INTERFACE.md §6-i). 정책이 τ_max까지 쓰면 실기에서 죽는다.
- default_joint_pos 전부 0.0.
- 게인이 `r2s_hind_leg`(300/5)보다 낮다(100/50/112.5/28.8) → GUI 슬라이더 상한을 낮춰야 조작 가능.
- `JOINT_NAME_PATTERNS`는 정규식이 아니라 정확한 관절명 — `.*_hip_joint`는 HL/HR 양쪽에 매칭된다.

## foot↔calf 전달기구 커플링 (2026-08-12, `cfg.foot_coupling=True` 기본)

실기 실측: foot 모터는 관절각이 아니라 **raw각 `q_raw = q_foot + q_calf`**(RL_INTERFACE coef=+1)를
구동한다 — calf가 +10° 돌면 foot 관절각이 −10° 따라가고(역방향 없음), **무토크에서도** 기어 마찰이
raw를 잠가 커플링이 유지된다(비가역 전달기구).

sim 재현(live/policy 모드, `_apply_foot_coupling()`에서 physics step 200Hz마다):
- foot 위치목표 = `raw_target − q_calf`, 속도목표 = `−q̇_calf` → actuator PD가 raw 공간 오차로 계산
  (위치 강제 쓰기 아님 — 전달기구 강성으로 미는 방식이라 접촉/동역학 무손상)
- **전치 토크**: `τ_calf += τ_foot_motor` (모터좌표 r=(q_c, q_f+q_c) ⇒ τ_joint=Tᵀτ_motor.
  ⚠RL_INTERFACE의 `τ_raw_src −= …`는 역방향(관절토크→모터명령) 식이라 부호가 반대)
- relax(foot kp≈0): 진입 순간 raw 래치 + `coupling_hold_kp/kd`(기본 200/2)로 잠금 — 실기의
  비가역 마찰 재현. 해제 시 GUI 게인 복귀.

### ★좌표 규약 이관 (2026-08-14, `CONVENTION_VERSION` 0 → 1)

워크스테이션 전체가 **관절(모델) 좌표 하나**로 통일됐다. raw↔관절 변환은 브리지(`real_runner`)가
전담하므로 실기 TELEM/ACT도 관절각이고, **sim의 env 경계도 전부 관절각**이다.

| 항목 | 구 규약 0 (~08-13) | **현 규약 1** |
|---|---|---|
| CMD/ACT의 foot 목표 | raw 목표 `q_f+q_c` | **관절 목표 `q_f`** |
| `get_lowstate()` foot q/dq/ddq | raw 합산 보고 | **관절각 그대로** |
| policy 모드 커플링 | **미적용**(학습과 불일치) | **적용** |
| sysid 모드 | raw (변경 없음) | raw (**이관 대상 아님**) |

- raw는 env 안에서 **목표값끼리** 합성한다: `raw_t = live_target[foot] + live_target[calf]`
  (측정 calf가 아니다). 학습 env `hind_leg_env._apply_action`의
  `raw_t = processed_actions[foot] + processed_actions[calf]` 와 **같은 식**이다.
- 관측 가능한 귀결: foot 관절 목표를 고정해도 **calf 추종오차만큼** foot 관절각이 밀린다
  (`e_f = q_f − foot_t` 가 `e_c = calf_t − q_c` 를 따라감). 커플링을 끄면 `e_f ≡ 0`.
- **relax의 `_raw_latch`만 예외로 여전히 raw 공간**이다 — 기어 마찰이 잠그는 것은 모터축이지
  관절각이 아니므로 그게 물리적으로 맞다. 규약 변경과 무관.
- ⚠ **sysid는 이관 대상이 아니다.** 엔코더가 raw만 재므로 데이터·재생·채점을 전부 raw로 일관시키는
  것이 맞다. `R2SBipedLegSysidEnvCfg.foot_coupling`은 **True**다(과거 문서의 "sysid cfg는
  `foot_coupling=False` 명시"는 2026-08-13 이후 틀린 서술 — 정정).
- `convention_version`은 `R2SBipedLegEnvCfg` 필드로 노출돼 `sim_runner_bipedleg.py`가 STATE 패킷에
  싣는다. ⚠ **스위치가 아니라 코드가 말하는 규약의 서술**이다 — 값만 바꿔도 거동은 안 바뀐다.
- soft limit 클램프는 GUI publisher(`_clamp_target`)와 real_runner 담당(다른 워커 소관).
- 검증 실측(`fix_base`, calf 1.5 Hz sine + foot 관절목표 고정): live-ON `slope(e_f~e_c)=+0.750
  corr=+0.946`, **coupling-OFF `slope=+0.000 corr=+0.062 pk-pk(q_f)=0.0000`**,
  policy-ON은 live-ON과 소수 4자리까지 동일.

### raw 좌표 foot 마찰 (2026-08-14, `cfg.foot_raw_friction=True` 기본)

실기 구조 확인(2026-08-14): calf/foot 모터가 **둘 다 허벅지**에 있고, foot 모터는 **무릎을 건너는
1:1 벨트**로 발목을 돈다 ⇒ 모터 출력각 `θ_f = q_foot + q_calf`(coef=+1 커플링의 기구적 정체).

foot 쪽 감속기·벨트 마찰은 관절축이 아니라 **모터축**에 앉아 있으므로 좌표를 옮긴다:

```
w_raw   = q̇_foot + q̇_calf
τ_fric  = −(b_raw·w_raw + c_raw·tanh(w_raw / eps))      # eps = foot_raw_friction_vel_eps (0.2 rad/s)
τ_foot += τ_fric,  τ_calf += τ_fric                      # 일률 보존 — 전치와 같은 규칙
```

- foot 관절의 **PhysX 마찰(static/dynamic/viscous)은 0으로 눌러 둔다**(`_clear_foot_joint_friction`,
  제어 스텝마다 1회 sim write). hip/thigh/calf는 종전대로 관절 좌표 PhysX 마찰.
- ★ **파라미터 재해석(개수 불변)**: `b_raw`/`c_raw`는 **foot 관절의 기존 viscous/Coulomb 슬롯**을
  그대로 읽는다. PACE 33개는 그대로이고 bounds도 그대로. 다만 식별 후 `viscous[*_foot]` /
  `coulomb[*_foot]`의 **의미가 관절 좌표가 아니라 모터축 좌표**다 — 배포 시 관절 마찰로 되쓰면 안 된다.
- ⚠ **저장소는 `data.default_joint_*` 캐시**다(sim은 0으로 눌려 있으므로). 이 캐시는 **최초 접근
  시점에 sim 값을 복제하는 lazy clone**이라 env `__init__`의 stock 캡처가 순서상 load-bearing이다.
  외부에서 마찰을 sim에만 쓰고 캐시를 안 쓰면 foot 마찰이 통째로 0이 된다
  (`validate_bipedleg.py`가 이 함정에 있었고 2026-08-14 수정).
- 안정성 캡: 명시적 feedforward라 `b·dt/I > 2`면 발산 → `|τ_fric| ≤ (armature+0.0019)·|w_raw|/dt`로
  한 스텝 내 축 속도 역전을 금지한다. 없으면 CMA-ES 후보가 크래시 없이 조용히 발산해 적합이 편향된다.
- A/B: `fit_bipedleg.py --foot_raw_friction on|off`, `--foot_transpose on|off`.
- ⚠ live relax(foot kp≈0)의 raw 래치(`coupling_hold_kp/kd` 200/2)는 **같은 기어 마찰을 다른 방식으로**
  모델링한 것이라 raw 마찰과 중복된다 — 실기 마찰이 식별되면 hold 게인 재검토 필요.
- 검증(구 규약 0 기준 서술, 참고용): calf sine+foot **raw** 고정 → `q_foot = raw_t − q_calf` 추종,
  foot 단독 시 calf 부동, relax에서 calf 강제 이동 시 foot이 raw 래치 유지하며 역추종.
  → 규약 1의 관절 프레임 재작성은 위 "좌표 규약 이관" 절의 검증 실측 참고.

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

공통 불변 규칙: `.claude/rules/r2s.md`

- soft limit은 `scripts/real2sim/r2s_biped_leg/motions.py`와 값 일치 필수(중복 정의). KP/KD는 예외 —
  motions.py는 실기 채널좌표, env_cfg.py는 관절좌표(calf×1.5², foot×1.2² 기어환산)라 값이 달라야
  정상이다(env_cfg.py:70-77, 미해결 seam 기록됨).
- `r2s_hind_leg` / `r2s_go2` 수정 금지

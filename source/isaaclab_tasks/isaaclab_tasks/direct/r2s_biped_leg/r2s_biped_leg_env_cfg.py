# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Biped Leg(8-DOF 2족) Real2Sim 환경 설정 (r2s_hind_leg 패턴 기반).

RL 없음. 외부(sim_runner_bipedleg.py)가 UDP로 받은 PD 목표를 set_setpoint()로 주입하면
slew rate limiter를 통해 안전하게 적용한다. `direct/hind_leg`와 동일한 8-DOF 2족 로봇
(`HIND_LEG_CFG`)을 사용하며, 자유베이스라 `fix_base` 토글을 지원한다.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_biped_leg/CONTRACT.md
"""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.rga import HIND_LEG_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 관절 계약 (CONTRACT.md §2) — USD 로드 순서 독립, find_joints(preserve_order=True)로 고정.
# 실측 관절명(2026-07-21 probe): leg-major 순서(HL 4개 → HR 4개).
# 패턴이 아니라 **정확한 관절명**을 쓴다 — 정규식(.*_hip_joint 등)은 HL/HR 양쪽에 매칭되어
# 좌우 순서가 USD 로드 순서에 의존하게 되기 때문.
# ---------------------------------------------------------------------------

NUM_JOINTS: int = 8

# 좌표 규약 버전 — STATE 패킷에 실어 브리지/GUI와 규약 불일치를 조기에 잡는다.
#   0 : (구) foot 명령·보고가 **raw각** q_foot + q_calf. 2026-08-14 이전.
#   1 : (현) 워크스테이션 전체가 **관절(모델) 좌표** 하나로 통일. raw↔관절 변환은 브리지
#       (`real_runner`)가 전담하고, sim은 실기와 같은 관절각을 말한다. foot 전달기구 커플링은
#       env 내부에서 raw를 합성해 재현할 뿐, **밖으로 나가는 값은 전부 관절각**이다.
# ⚠ 이 상수는 **코드가 말하는 규약의 서술**이지 스위치가 아니다 — 값만 바꿔도 거동은 안 바뀐다.
#   좌표 규약을 실제로 되돌리려면 코드를 되돌려야 하고, 그때 이 값도 같이 내려야 한다.
CONVENTION_VERSION: int = 1

JOINT_NAME_PATTERNS: list[str] = [
    "HL_hip_joint",
    "HL_thigh_joint",
    "HL_calf_joint",
    "HL_foot_joint",
    "HR_hip_joint",
    "HR_thigh_joint",
    "HR_calf_joint",
    "HR_foot_joint",
]

# GUI/로그 표시용 짧은 레이블 (JOINT_NAME_PATTERNS와 동일 순서).
JOINT_LABELS: list[str] = [
    "HL_hip",
    "HL_thigh",
    "HL_calf",
    "HL_foot",
    "HR_hip",
    "HR_thigh",
    "HR_calf",
    "HR_foot",
]

# 관절별 PD 게인 — HIND_LEG_CFG(rga.py)의 legs 액추에이터와 일치. **관절(모델) 좌표**다.
# 2026-08-18: 실기 드라이버 게인의 관절 공간 환산(채널값 × gear_k², 실기팀 2026-08-12 지정)으로
# 갱신. 구값 65/53/12/20 + 6/4.8/1.1/1.0 은 구 모델 I_eff 기반 설계값이라 실기와 무관했다.
#
# ⚠⚠ **게인 경로에 좌표 seam 이 남아 있다 (미해결).**
#   `motions.DEFAULT_KP = [100, 50, 50, 20]` 은 **채널** 게인이고, GUI 가 그 값을 그대로
#   ① 실기 드라이버(`real_runner_bipedleg.cpp:481`, 변환 없음)와 ② sim(`set_setpoint` →
#   faithful_pd → 관절 게인) **양쪽에 같은 숫자로** 보낸다. 두 좌표가 k^n 만큼 다르므로 한쪽은
#   반드시 틀린다 — 토크 모니터에서 찾았던 것과 **같은 종류의 버그**다(README §8).
#   여기 값은 sim 플랜트 기준(관절)이라 맞지만, GUI 가 이 값을 실기로 보내면 calf 가 2.25 배
#   과도해진다. GUI/real_runner 중 한쪽에 변환을 넣어야 한다.
#   근거: reports/_comparisons/pace_bipedleg_foot_coupling_probe/README.md §13
DEFAULT_KP: list[float] = [100.0, 50.0, 112.5, 28.8, 100.0, 50.0, 112.5, 28.8]
DEFAULT_KD: list[float] = [5.0, 5.0, 11.25, 7.2, 5.0, 5.0, 11.25, 7.2]

# 관절 최대 속도 [rad/s] — slew rate limiter용 (실기 무부하 한계, RL_INTERFACE.md §6-c).
V_MAX_RAD: list[float] = [29.6, 29.6, 19.7, 24.6, 29.6, 29.6, 19.7, 24.6]

# soft joint position limit [rad] (soft_joint_pos_limit_factor=0.9 반영).
# 2026-08-12 SignFix: 실기 방향 실측(같은 목표에 반대로 도는 관절 발견)에 맞춰
#   HL_hip/HL_calf/HL_foot/HR_hip/HR_thigh 축을 반전한 Hind_Leg_URDF3_SignFix 자산 기준.
#   반전 관절은 limit이 [lo,hi]→[-hi,-lo]로 뒤집힌다(hip은 대칭이라 값 동일).
#   좌우가 thigh/calf/foot에서 미러 관계가 됐다.
# ⚠ scripts/real2sim/r2s_biped_leg/motions.py 의 SOFT_LIMITS_RAD 와 값 일치 필수(다른 패키지라 중복).
#   sim이 position target을 이 범위로 silently 클램프하므로 GUI 범위도 여기에 맞춘다.
SOFT_LIMITS_RAD: list[tuple[float, float]] = [
    (-0.2340, 0.2340),  # HL_hip   (raw ±0.26 rad = ±14.9°, 축 반전 — 대칭이라 값 동일)
    (-0.9555, 2.1855),  # HL_thigh (raw −1.13~+2.36 rad)
    (-0.7745, 0.9445),  # HL_calf  (축 반전, raw −0.87~+1.04 rad)
    (-0.3440, 1.3840),  # HL_foot  (축 반전, raw −0.44~+1.48 rad)
    (-0.2340, 0.2340),  # HR_hip   (축 반전 — 대칭이라 값 동일)
    (-2.1855, 0.9555),  # HR_thigh (축 반전, raw −2.36~+1.13 rad)
    (-0.9445, 0.7745),  # HR_calf
    (-1.3840, 0.3440),  # HR_foot
]

# 기본(중립) 자세 — 실측 default_joint_pos 전부 0.0.
DEFAULT_POSE: list[float] = [0.0] * NUM_JOINTS

# 공중 고정(fix_base) spawn 높이 [m] — 지면 자유 spawn(0.6)보다 높여 다리 스윙 여유를 둔다.
FIXED_BASE_HEIGHT_M: float = 0.8

# foot 링크의 관절축 기준 유효 관성 [kg·m²] — armature(로터 반사관성)를 뺀 링크분만.
# rga.py `HIND_LEG_CFG` kp/kd 주석의 실측 I_eff(foot 0.0019)에서 왔다.
# raw 좌표 마찰(`foot_raw_friction`)의 explicit 적분 안정 캡을 잡는 데만 쓴다 — armature가
# PACE 식별 대상(하한 1e-5)이라 armature만으로는 안전 하한이 0에 가까워지기 때문.
FOOT_LINK_INERTIA_KGM2: float = 0.0019


@configclass
class R2SBipedLegEnvCfg(DirectRLEnvCfg):
    """Biped Leg Real2Sim 테스트 환경 (8-DOF 2족).

    Observation (24-dim): joint_pos(8) + joint_vel(8) + applied_torque(8).
    Action (8-dim): 사용 안 함(setpoint은 set_setpoint()로 주입, action은 no-op).
    """

    # 좌표 규약 버전 — sim_runner_bipedleg.py가 STATE 패킷에 실어 브리지/GUI와 대조한다.
    # ⚠ 스위치가 아니라 **코드가 말하는 규약의 서술**이다 (:data:`CONVENTION_VERSION` 주석 참고).
    convention_version: int = CONVENTION_VERSION

    # 에피소드 — 10분 (조기 종료 없음)
    episode_length_s: float = 600.0
    decimation: int = 4  # 200Hz physics / 50Hz control

    observation_space: int = 3 * NUM_JOINTS  # 8 × (pos + vel + torque)
    action_space: int = NUM_JOINTS
    state_space: int = 0

    # faithful PD: True면 set_setpoint의 kp/kd를 write_joint_stiffness/damping_to_sim으로
    # 실제 반영(GUI 슬라이더가 살아있음). False면 cfg 액추에이터 PD 고정(r2s_go2 seam 방식).
    faithful_pd: bool = True

    # 공중 고정(fix_base) 토글 — True면 base를 fixed articulation root로 공중 스폰 (CONTRACT §5).
    # HIND_LEG_CFG는 자유베이스 2족이라 fix_base=False면 GUI로 관절을 스텝하는 순간 넘어진다.
    # 런처(run_sim_runner.sh) 기본값은 fix_base=True.
    fix_base: bool = False

    # 시스템 식별(PACE) 모드 — True면 _pre_physics_step이 action을 절대 관절 목표각으로 직접 쓰고
    # slew limiter/faithful PD를 우회한다 (chirp 고주파 왜곡 방지). R2SBipedLegSysidEnvCfg가 켠다.
    sysid: bool = False

    # foot↔calf 커플링 (2026-08-12 실기 실측, RL_INTERFACE.md §0~§1의 "foot만 커플링" 항목).
    # foot 모터는 관절각이 아니라 **raw각 q_raw = q_foot + q_calf**(coef=+1)를 구동한다 — 벨트가
    # 무릎을 건너기 때문이다. True면 live/policy 모드에서 foot PD를 raw 공간으로 계산하고,
    # foot kp≈0(relax)이면 raw를 래치해 coupling_hold_kp/kd로 잠근다.
    #
    # ★ 2026-08-14 좌표 규약 이관 (:data:`CONVENTION_VERSION` 0 → 1): **env 경계는 전부 관절각**이다.
    #   - CMD/GUI의 foot 목표 = **관절 목표** (구 규약에선 raw 목표였다). raw는 env 안에서
    #     `raw_t = foot_target + calf_target`으로 합성한다 — 학습 env `hind_leg_env._apply_action`과 동일 식.
    #   - `get_lowstate()`의 foot q/dq/ddq = **관절각** (구 규약의 raw 합산 제거).
    #   - raw↔관절 변환은 브리지(`real_runner`)가 전담하므로 실기 TELEM/ACT도 관절각이다.
    #   - **policy 모드에도 적용**(구: 미적용). 학습 env가 커플링을 적용하므로, 안 하면 배포
    #     리허설이 학습과 다른 플랜트가 된다.
    #   - relax의 raw 래치만 예외로 여전히 raw 공간이다 — 기어 마찰이 잠그는 것이 모터축이므로.
    # ⚠ sysid 모드는 이관 대상이 **아니다**: 엔코더가 raw만 재므로 데이터·재생·채점을 전부 raw로
    #   일관시키는 것이 맞다 (`R2SBipedLegSysidEnvCfg.foot_coupling=True`, 아래 sysid 분기 참고).
    foot_coupling: bool = True
    coupling_hold_kp: float = 200.0  # relax 시 raw 잠금(모터 마찰 등가) 강성 — 실기 실측 후 조정
    coupling_hold_kd: float = 2.0

    # 전치 토크 τ_calf += τ_foot_motor — **기본 True**.
    #
    # 실기 구조 (2026-08-14 실기팀 확인): calf/foot 모터가 **둘 다 허벅지에 모여** 있다.
    #   calf 모터 → 기어 → 무릎,  foot 모터 → 기어 → **무릎을 건너는 1:1 벨트** → 발목.
    # 벨트가 무릎을 건너므로 foot 모터 출력각은 발의 **대퇴 기준** 각도, 즉 θ_f = q_foot + q_calf 다
    # (coef=+1 커플링의 기구적 정체). 이건 기구 구속이므로 일률 보존에서 토크 관계가 **강제된다**:
    #   δW = τ_θf·δθ_f = τ_θf·(δq_c + δq_f)  ⇒  Q_calf += τ_θf,  Q_foot = τ_θf
    # 별개의 모델링 선택이 아니다 — 위치 커플링을 넣으면 이 항도 넣어야 한다.
    #
    # ⚠ 2026-08-13 에 실측 반증이라 판단해 False 로 바꿨다가 **되돌렸다**. 근거였던 두 관측
    #   (foot 단독 chirp 에서 무릎 무반응 / calf 무여자에서 foot 구동 시 calf 부동)은 전치를
    #   반증하지 못한다 — **무릎 드라이브 정지마찰이 외란을 그대로 먹으면 같은 결과**가 나오고,
    #   그러려면 4.87 / 4.80 N·m 만 있으면 된다(무릎 peak 126 N·m 의 **3.9%**, 10.5:1 감속기에서
    #   지극히 평범한 값). 즉 그 검정은 반증이 아니라 **판정 불능**이었다.
    #   (당시 마찰 추정 1.1 N·m 은 gear_k 오염 적합에서 나온 값인 데다, chirp 는 calf 가 움직이는
    #    구간이라 **운동마찰**을 잰 것 — foot 단독 chirp 의 정지 상태에 적용할 값이 아니었다.)
    #   기록: reports/_comparisons/pace_bipedleg_foot_coupling_probe/GEAR_K_INVALIDATION.md §6
    #
    # 판정 가능한 검정: **calf 를 천천히 등속으로 움직이면서**(이미 미끄러지는 중 = 정지마찰 없음)
    #   foot 을 chirp 한다. 그러면 외란이 무릎 추종오차/토크에 반드시 드러난다.
    #
    # ★ 2026-08-18 구현 변경: 전치 토크로 foot 게인을 재계산한 예측치가 아니라 foot 액추에이터가
    #   **실제로 낸** 직전 스텝 토크(`data.applied_torque`)를 싣는다. 옛 식은 정적 effort_limit
    #   (100.8 N·m)로 클램프했는데 DCMotor 는 4사분면 속도 곡선으로 자르므로, 옛 식은 낼 수 없는
    #   토크를 calf 에 실을 수 있었다. 두 식은 교차점 q̇ ≈ 4.92 rad/s **아래에서 완전히 일치**한다.
    #   실측(§11, `fix_base`·무중력·마찰 OFF): kp_foot=20 이면 foot 이 교차점에 도달조차 못 해
    #   **차이 없음**. 게인 200(hold)에서 스텝의 17.2%, 600 에서 40.0% 가 옛 식으로는 곡선 밖이었다.
    #   신식은 세 조건 모두 곡선 안.
    #   ⚠ 이건 **고정베이스 조건**의 결과다. 접촉·중력·마찰이 붙는 학습 env 에서는 kp_foot=20 에서도
    #   foot q̇ 가 24.70 rad/s 까지 가고 구식이 0.012% 의 스텝에서 곡선 밖이었다(§11 후반).
    foot_transpose: bool = True

    # foot 마찰을 raw(모터축) 좌표로 옮긴다 — **기본 True**.
    #
    # foot 쪽 감속기·벨트 마찰은 물리적으로 **모터축**에 앉아 있고, 그 축의 속도는
    # θ̇_f = q̇_foot + q̇_calf 다. 그리고 벨트가 무릎을 건너므로 `foot_transpose`와 **같은 일률 보존
    # 규칙**에 의해 그 마찰 토크는 foot·calf 양쪽 관절에 **같은 부호·같은 크기**로 실린다:
    #     w_raw   = q̇_f + q̇_c
    #     τ_fric  = −(b_raw·w_raw + c_raw·sign(w_raw))
    #     τ_foot += τ_fric,   τ_calf += τ_fric
    # PhysX의 관절 마찰은 q̇_f 에만 걸리므로 좌표가 틀렸다 — foot 관절의 PhysX 마찰(static/dynamic/
    # viscous)을 0으로 눌러 두고 위 식을 feedforward 토크로 직접 넣는다.
    #
    # ★ 파라미터 재해석 (파라미터 수 불변): b_raw / c_raw 는 **foot 관절의 기존 viscous / Coulomb
    #   슬롯**을 그대로 읽어 쓴다. PACE 33개(armature8+viscous8+coulomb8+bias8+delay1)는 그대로다.
    #   foot 관절의 마찰은 곧 감속기·벨트 마찰이므로 물리적으로도 이 재해석이 맞다. 다만 식별 후
    #   `viscous[HL/HR_foot]`·`coulomb[HL/HR_foot]`의 의미는 "관절 q̇_f 에 대한 마찰"이 아니라
    #   **"모터축 θ̇_f 에 대한 마찰"**이다 — 배포 시 관절 마찰로 되쓰면 안 된다.
    #   (짝: `r2s_biped_leg_sysid_cfg.BipedLegPaceCfg` bounds 주석)
    #
    # False면 기존(관절 좌표 PhysX 마찰) 동작 그대로 — 되돌릴 수 있다.
    # 반사관성 off-diagonal — 학습 env `hind_leg_env_cfg.foot_reflected_inertia` 와 같은 스위치.
    # 벨트가 무릎을 건너 foot 로터가 θ_f = N_f·(q_f + q_c) 로 돌기 때문에 반사관성이 대각이 아니다:
    #     M_refl = I_r·[[N_c² + N_f²,  N_f²],  =  [[0.1338, 0.0522], [0.0522, 0.0522]]
    # 대각은 `rga.py` armature 로 들어가고, PhysX 가 표현 못 하는 off-diagonal 만 명시적 토크로:
    #     τ_calf += −I_off·q̈_foot,  τ_foot += −I_off·q̈_calf   (I_off = foot armature, DR 자동 일관)
    # 실측상 발산하지 않는다(§12) — 보정을 effort target 에 더해 모터 곡선이 다시 자르기 때문.
    foot_reflected_inertia: bool = True

    foot_raw_friction: bool = True

    # sign(w_raw) 완화 폭 [rad/s]. 0 근처에서 부호가 매 스텝 뒤집히는 채터링을 막기 위해
    # sign 대신 tanh(w_raw / eps)를 쓴다. 값이 작을수록 실제 Coulomb에 가깝지만 채터링에 취약하다.
    foot_raw_friction_vel_eps: float = 0.2

    # policy 모드 — True면 _pre_physics_step이 외부(policy_runner_bipedleg.py)가 set_policy_target()으로
    # 주입한 **articulation 순서** 목표각을 set_joint_position_target()으로 직접 적용한다.
    # slew limiter/faithful PD를 우회하고(학습 파이프라인엔 slew 없음), 게인은 cfg DCMotor 고정
    # (write_joint_stiffness 경로는 explicit actuator에서 q≈target/2 버그를 유발하므로 절대 사용 안 함).
    # 자유베이스로 정책이 균형을 잡으므로 fix_base=False 여야 한다.
    policy_mode: bool = False

    # 로봇 — HIND_LEG_CFG는 자유베이스 8-DOF 2족, init z=0.6.
    robot: ArticulationCfg = HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)

    # 참고: fix_base 적용(fix_root_link=True + spawn 높이)은 R2SBipedLegEnv._setup_scene에서 수행한다.
    # cfg 생성 후 sim_runner_bipedleg.py가 --fix_base로 이 필드를 뒤늦게 덮어써도 반영되도록,
    # __post_init__(생성 시점)이 아니라 env 빌드 시점에 self.cfg.fix_base를 읽어 적용한다.

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

# 관절별 PD 게인 — HIND_LEG_CFG(rga.py)의 legs 액추에이터 실측값과 일치.
# GUI가 이 값을 기본으로 발행하고, faithful_pd=True면 슬라이더로 런타임 변경 가능.
# ⚠ r2s_hind_leg(300/5)보다 훨씬 낮다(12~65) — GUI 슬라이더 상한도 여기에 맞춰야 조작 가능.
DEFAULT_KP: list[float] = [65.0, 53.0, 12.0, 20.0, 65.0, 53.0, 12.0, 20.0]
DEFAULT_KD: list[float] = [6.0, 4.8, 1.1, 1.0, 6.0, 4.8, 1.1, 1.0]

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


@configclass
class R2SBipedLegEnvCfg(DirectRLEnvCfg):
    """Biped Leg Real2Sim 테스트 환경 (8-DOF 2족).

    Observation (24-dim): joint_pos(8) + joint_vel(8) + applied_torque(8).
    Action (8-dim): 사용 안 함(setpoint은 set_setpoint()로 주입, action은 no-op).
    """

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

    # foot↔calf 전달기구 커플링 (2026-08-12 실기 실측, RL_INTERFACE.md §0~§1의 "foot만 커플링" 항목).
    # 실기 foot 모터는 관절각이 아니라 **raw각 q_raw = q_foot + q_calf**(coef=+1)를 구동한다 —
    # calf가 +10° 돌면 foot 관절각이 −10° 따라가고(역방향 없음), 무토크에서도 기어 마찰이 raw를
    # 잠가 커플링이 유지된다(비가역 전달기구, 실기 관찰). True면 live(position) 모드에서 foot PD를
    # raw 공간으로 계산(+ 전치 토크 τ_calf += τ_foot_motor)하고, foot kp≈0(relax)이면 raw를 래치해
    # coupling_hold_kp/kd로 잠근다. GUI/CMD의 foot 목표 의미는 **raw 목표**가 되며(실기 해석과 동일),
    # get_lowstate의 foot q/dq도 raw로 보고한다(실기 TELEM과 정합). policy/sysid 모드에는 미적용.
    foot_coupling: bool = True
    coupling_hold_kp: float = 200.0  # relax 시 raw 잠금(기어 마찰 등가) 강성 — 실기 실측 후 조정
    coupling_hold_kd: float = 2.0

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

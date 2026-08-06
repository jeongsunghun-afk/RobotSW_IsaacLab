# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 Real2Sim 환경 설정."""

from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# ---------------------------------------------------------------------------
# 조인트 순서 (CONTRACT.md §2 고정 — unitree_go/MotorCmd[0:12]의 유일한 진실)
# ---------------------------------------------------------------------------

JOINT_ORDER: list[str] = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]

NUM_JOINTS: int = 12

# 관절 최대 속도 (rad/s) — slew rate limiter용. UNITREE_GO2_CFG의 velocity_limit=30 기준 (CONTRACT §6).
V_MAX_RAD: list[float] = [30.0] * NUM_JOINTS

# 기본 PD 게인 (CONTRACT §6, DCMotorCfg base_legs) — set_setpoint()로 주입된 kp/kd는 M1에서
# 미적용(버퍼 저장만)이므로 reset 시 이 값으로 채워 실제 적용 게인과 표시값을 맞춘다.
DEFAULT_KP: float = 25.0
DEFAULT_KD: float = 0.5

# 공중 고정(fix_base) spawn 높이 [m] — 지면 자유 spawn보다 높여 다리 스윙 여유를 둔다.
FIXED_BASE_HEIGHT_M: float = 0.5

# 엎드린(prone) 초기 자세 — 실기 GO2가 엎드려 시작하므로 sim도 동일하게 초기화(Real2Sim 정합).
# GUI "Stand Up" 버튼이 여기서 go2_stand_example 궤적으로 일어선다.
# ⚠ scripts/real2sim/r2s_go2/motions.py 의 STAND_FOLDED 와 값 일치 필수(다른 패키지라 중복 정의).
#   값 출처: unitree_ros2 go2_stand_example target_pos_1. 단 calf 는 -2.65→-2.6 완화 —
#   -2.65 는 GO2 calf soft limit(soft_lo≈-2.628, soft_joint_pos_limit_factor=0.9)를 벗어나
#   sim 이 position target 을 silently 클램프하므로, 명령이 soft limit 안에 들도록 -2.6 사용.
PRONE_HEIGHT_M: float = 0.10  # 엎드림 base 스폰 높이 [m] — sim 실측 정착 base_z≈0.088 위 소폭 마진(정착 확인됨).
_STAND_FOLDED: list[float] = [0.0, 1.36, -2.6, 0.0, 1.36, -2.6, -0.2, 1.36, -2.6, 0.2, 1.36, -2.6]
PRONE_JOINT_POS: dict[str, float] = {JOINT_ORDER[i]: _STAND_FOLDED[i] for i in range(NUM_JOINTS)}


@configclass
class R2SGo2EnvCfg(DirectRLEnvCfg):
    """R2S-GO2 Real2Sim 테스트 환경 (Milestone 1).

    RL 없음. sim_runner_go2.py가 set_setpoint()로 주입한 PD 목표를 그대로 추종하는
    순수 포지션 제어 stub (r2s_hind_leg 패턴).

    Observation (36-dim):
        joint_pos(12) + joint_vel(12) + applied_torque(12)

    Action (12-dim):
        미사용. actions 인자는 무시되며 setpoint은 set_setpoint()로 외부 주입된다.
    """

    # 에피소드 — 10분 (조기 종료 없음)
    episode_length_s: float = 600.0
    decimation: int = 4  # 200Hz physics / 50Hz control

    # 공간
    observation_space: int = 3 * NUM_JOINTS
    action_space: int = NUM_JOINTS
    state_space: int = 0

    # 공중 고정(fix_base) 토글 — True면 base를 fixed articulation root로 공중 스폰 (CONTRACT §5).
    fix_base: bool = False

    # 시스템 식별(sysid) 모드 토글 — opt-in. live(UDP) 경로는 기본값 False로 완전히 불변이다.
    #   True면 `actions`(관절 목표각, articulation 관절 순서)를 slew limiter 없이 그대로 적용한다.
    #   PACE CMA-ES는 chirp 명령을 매 스텝 재생하므로 slew(0.6 rad/step)가 고주파를 왜곡하면 안 되고,
    #   setpoint은 UDP가 아니라 env.step(actions)로 들어온다.
    # 설정은 r2s_go2_sysid_cfg.R2SGo2SysidEnvCfg 참고.
    sysid: bool = False

    # PACE 식별 관절 물성(armature / viscous / coulomb) 적용 토글.
    #
    # **기본 True.** r2s 의 목적은 sim 이 실기와 같아지는 것이고 PACE 는 바로 그 실기 GO2 를
    # 식별한 값이다. 예전엔 이 env 가 nominal `UNITREE_GO2_CFG`(armature 0.01, 마찰 0)로 돌아
    # 학습 env(`Go2-Imitation-Tracking-v0`, armature 0.17~0.20 / viscous 2.3~2.5)와 플랜트가
    # 전혀 달랐고, 그래서 학습 정책을 GUI Policy 모드로 돌리면 관절이 초당 ~22회 진동했다
    # (`reports/rsl_rl/go2_imitation_tracking/_comparisons/r2s_sim_plant_gap/`).
    #
    # ⚠ kp/kd 는 25/0.5 로 **유지한다** — viscous 가 kd 오차를 흡수하도록 함께 식별된 조합이라
    #   게인을 따로 바꾸면 식별 결과가 깨진다(`PACE_KP`/`PACE_KD` 주석 참고).
    # False 로 두면 이전 nominal 거동으로 정확히 되돌아간다(PACE 이전 수집 데이터 재현용).
    use_pace_params: bool = True

    # 로봇 (fix_base=False 기본: 지면 자유 spawn, CONTRACT §6 물성 그대로)
    # 초기 자세를 prone(엎드림)으로 오버라이드 — 실기 시작 자세와 정합.
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=UNITREE_GO2_CFG.init_state.replace(pos=(0.0, 0.0, PRONE_HEIGHT_M), joint_pos=PRONE_JOINT_POS),
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=4.0, replicate_physics=True)

    # 뷰포트 카메라 — 기본은 **로봇 추적**. `origin_type="asset_root"` 이면 IsaacLab 이 매 렌더
    # 스텝마다 eye/lookat 을 로봇 base 기준 상대좌표로 다시 잡아준다(별도 콜백 불필요).
    # 런타임 전환은 `R2SGo2Env.set_camera_follow()` — GUI 의 Sim 그룹이 UDP 로 호출한다.
    # headless 에서는 `viewport_camera_controller` 자체가 None 이라 전부 no-op 이 된다.
    viewer: ViewerCfg = ViewerCfg(
        eye=(-2.2, -1.6, 0.9),  # 로봇 base 기준 상대 위치 [m] — 뒤 왼쪽 위에서 내려다봄
        lookat=(0.0, 0.0, 0.15),
        origin_type="asset_root",
        asset_name="robot",
    )

    # 시뮬레이션
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 200.0, render_interval=decimation)

    # 참고: fix_base 적용(fix_root_link=True + spawn 높이)은 R2SGo2Env._setup_scene에서 수행한다.
    # cfg 생성 후 sim_runner_go2.py가 --fix_base로 이 필드를 뒤늦게 덮어써도 반영되도록,
    # __post_init__(생성 시점)이 아니라 env 빌드 시점에 self.cfg.fix_base를 읽어 적용한다.

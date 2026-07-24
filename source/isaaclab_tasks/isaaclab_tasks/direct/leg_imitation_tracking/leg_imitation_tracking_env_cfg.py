# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg Imitation Tracking (AMP + Body-frame Velocity Tracking) 환경 설정.

`go2_imitation_tracking` 의 Leg_URDF2(17-DOF 4족 + 허리) 이식 버전.
Go2 실기에서 식별한 PACE 액추에이터 파라미터와 그에 딸린 Domain Randomization 은
이 로봇에 적용할 근거가 없으므로 제거했다.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets.robots.rga import LEG_CFG  # isort: skip

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "smr_leg_pkl")

NUM_JOINTS: int = 17
"""다리 4개 × (hip, thigh, calf, foot) + 허리 1개."""

NUM_FEET: int = 4


@configclass
class LegImitationTrackingEnvCfg(DirectRLEnvCfg):
    """Leg Imitation Tracking 환경 설정 — body-frame 속도추종 버전.

    Policy 관측 (observation_space = 63):
        root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3) +
        lin_vel_cmd(2) + yaw_vel_cmd(1) +
        joint_pos_offset(17) + joint_vel(17) + actions(17)

    AMP Discriminator 관측 (amp_observation_space = 59, per step):
        dof_pos(17) + dof_vel(17) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [heading-relative 6D rotation, MimicKit compute_tar_obs 방식]

    AMP History (num_amp_observations = 10):
        amp_observation_size = 59 × 10 = 590
    """

    # ── 에피소드 ────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # sim: 200 Hz physics, 50 Hz policy (decimation=4)
    sim_dt_hz: int = 200
    policy_dt_hz: int = 50
    decimation: int = sim_dt_hz // policy_dt_hz  # 4

    # ── 공간 ────────────────────────────────────────────────────
    observation_space: int = 3 + 3 + 3 + 2 + 1 + 3 * NUM_JOINTS  # = 63
    action_space: int = NUM_JOINTS  # = 17
    state_space: int = 0

    num_amp_observations: int = 10  # disc history depth
    # per-step disc obs = dof_pos + dof_vel + root_height + lin_vel + ang_vel + foot_pos + rot_tan_norm
    amp_observation_space: int = 2 * NUM_JOINTS + 1 + 3 + 3 + 3 * NUM_FEET + 6  # = 59
    include_rel_track_obs: bool = False  # 상대적 2D 궤적 포함 여부 토글

    # ── 모션 데이터 ─────────────────────────────────────────────
    motion_file: str = MOTION_FILES_DIR
    reference_body: str = "Base"

    # 항상 RSI (Reference State Initialization) 사용
    reset_strategy: str = "random"  # "random" | "random_start"

    # ── 속도추종 command 범위 ────────────────────────────────────
    # 레퍼런스 모션 분포: vx 0.0~5.0 m/s, yaw rate -1.5~2.3 rad/s
    lin_vel_x_min: float = 0.0  # vx 최소 [m/s]
    lin_vel_x_max: float = 4.0  # vx 최대 [m/s]
    lin_vel_y_min: float = 0.0  # vy 항상 0 [m/s]
    lin_vel_y_max: float = 0.0  # vy 항상 0 [m/s]
    yaw_vel_min: float = -1.5  # yaw rate 최소 [rad/s]
    yaw_vel_max: float = 1.5  # yaw rate 최대 [rad/s]
    tar_change_time_min: float = 4.0  # 목표 명령 변경 최소 주기 [s]
    tar_change_time_max: float = 7.0  # 목표 명령 변경 최대 주기 [s]

    # ── 보상 가중치 ─────────────────────────────────────────────
    # Task reward = lin_vel_reward_w * lin_vel_reward + yaw_vel_reward_w * yaw_vel_reward
    lin_vel_reward_w: float = 0.7  # 선속도 추종 가중치
    yaw_vel_reward_w: float = 0.3  # yaw 속도 추종 가중치
    vel_err_scale: float = 0.5  # lin_vel reward 지수 스케일
    yaw_vel_err_scale: float = 0.5  # yaw_vel reward 지수 스케일

    # ── 조기 종료 ───────────────────────────────────────────────
    early_termination: bool = True
    # 레퍼런스 base 높이 0.43~0.65 m (최소 0.432). 0.35 는 그보다 낮아 정상 보행을 자르지 않으면서
    # 반쯤 누운 자세(넘어짐)는 잡는다.
    termination_height: float = 0.35  # base 높이 임계값 [m]
    contact_force_threshold: float = 500.0  # base 접촉 판정 [N] (contact sensor 결함으로 현재 사실상 무력)
    # base 총 기울기 임계 [deg]. projected_gravity_b[2](=gz) 로 판정한다.
    # 축별 roll/pitch 방식은 roll·pitch 가 각각 임계 미만이어도 총 tilt 가 큰 대각 사각지대가 있어
    # gz 기반 total-tilt 로 대체했다. 정상 보행 gz≈-0.95(약 18°), reference 최악 -0.952 라 여유가 크다.
    max_tilt_deg: float = 45.0

    # ── 액션 ────────────────────────────────────────────────────
    action_scale: float = 0.25

    # ── 시뮬레이션 ──────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / sim_dt_hz,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    robot: ArticulationCfg = LEG_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

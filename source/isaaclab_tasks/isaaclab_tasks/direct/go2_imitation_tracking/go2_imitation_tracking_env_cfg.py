# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Imitation Tracking (AMP + Body-frame Velocity Tracking) 환경 설정."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "smr_mirror_pkl")


@configclass
class Go2ImitationTrackingEnvCfg(DirectRLEnvCfg):
    """Go2 Imitation Tracking 환경 설정 — body-frame 속도추종 버전.

    Policy 관측 (observation_space = 48):
        root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3) +
        lin_vel_cmd(2) + yaw_vel_cmd(1) +
        joint_pos_offset(12) + joint_vel(12) + actions(12)

    AMP Discriminator 관측 (amp_observation_space = 49, per step):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) +
        root_rot_tan_norm(6)  [R4: heading-relative 6D rotation, MimicKit compute_tar_obs 방식]

    AMP History (num_amp_observations = 10):
        amp_observation_size = 49 × 10 = 490
    """

    # ── 에피소드 ────────────────────────────────────────────────
    episode_length_s: float = 10.0

    # sim: 200 Hz physics, 50 Hz policy (decimation=4)
    sim_dt_hz: int = 200
    policy_dt_hz: int = 50
    decimation: int = sim_dt_hz // policy_dt_hz  # 4

    # ── 공간 ────────────────────────────────────────────────────
    observation_space: int = 3 + 3 + 3 + 2 + 1 + 12 + 12 + 12  # = 48
    action_space: int = 12
    state_space: int = 0

    num_amp_observations: int = 10  # disc hist depth (ablation: 2→10, MimicKit 방향)
    amp_observation_space: int = 49  # per-step disc obs (R4: +6 root_rot_tan_norm)
    include_rel_track_obs: bool = False  # 상대적 2D 궤적 포함 여부 토글

    # ── 모션 데이터 ─────────────────────────────────────────────
    motion_file: str = MOTION_FILES_DIR
    reference_body: str = "base"

    # 항상 RSI (Reference State Initialization) 사용
    reset_strategy: str = "random"  # "random" | "random_start"

    # ── 속도추종 command 범위 ────────────────────────────────────
    lin_vel_x_min: float = -1.0  # vx 최소 (m/s)
    lin_vel_x_max: float = 3.0   # vx 최대 (m/s)
    lin_vel_y_min: float = 0.0   # vy 항상 0
    lin_vel_y_max: float = 0.0   # vy 항상 0
    yaw_vel_min: float = -1.5    # yaw rate 최소 (rad/s)
    yaw_vel_max: float = 1.5     # yaw rate 최대 (rad/s)
    tar_change_time_min: float = 4.0  # 목표 명령 변경 최소 주기 (s)
    tar_change_time_max: float = 7.0  # 목표 명령 변경 최대 주기 (s)

    # ── 보상 가중치 ─────────────────────────────────────────────
    # Task reward = lin_vel_reward_w * lin_vel_reward + yaw_vel_reward_w * yaw_vel_reward
    lin_vel_reward_w: float = 0.7   # 선속도 추종 가중치
    yaw_vel_reward_w: float = 0.3   # yaw 속도 추종 가중치
    vel_err_scale: float = 0.5      # lin_vel reward 지수 스케일
    yaw_vel_err_scale: float = 0.5  # yaw_vel reward 지수 스케일

    # ── 조기 종료 ───────────────────────────────────────────────
    early_termination: bool = True
    termination_height: float = 0.15  # base 높이 임계값 (m)
    contact_force_threshold: float = 500.0  # base 접촉 판정 (N)
    roll_termination_deg: float = 70.0
    pitch_termination_deg: float = 70.0

    # ── 액션 ────────────────────────────────────────────────────
    action_scale: float = 0.25

    # ── 시뮬레이션 ──────────────────────────────────────────────
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / sim_dt_hz,
        render_interval=decimation,  # decimation
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

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

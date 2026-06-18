# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""R_Skeleton AMP 환경 설정."""

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.rga import R_SKELETON_CFG  # isort: skip

# 모션 파일 디렉토리 (이 파일 기준으로 상대 경로)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "txt_dataset_skeleton_sample")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "txt_dataset_skeleton_stmr")
MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "txt_dataset_skeleton_sample2")


@configclass
class SkeletonAmpEnvCfg(DirectRLEnvCfg):
    """R_Skeleton AMP imitation 학습 환경 설정.

    AMP 관측 벡터 구성 (amp_observation_space = 99):
        dof_pos(34) + dof_vel(34) + root_height(1) +
        lin_vel(3) + ang_vel(3) +
        key_body_pos(12) + key_body_lin_vel(12)
    """

    # 에피소드
    episode_length_s = 20.0
    decimation = 4

    # 공간
    observation_space = 108
    action_space = 34  # R_Skeleton DOF 수
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 99

    # History & Privileged
    history_observation = True
    history_len = 50
    priv_latent = True
    num_priv_obs = 114  # lin_vel(3)+ang_vel(3)+projected_gravity(3)+commands(3)+joint_pos(34)+joint_vel(34)+actions(34)+mass(1) + material_props = 적절히 분배됨 (아래 obs_groups에서 114 예상)

    # 조기 종료
    early_termination = True
    termination_height = 0.3  # base 높이가 이 값 이하이면 넘어짐으로 판정
    action_scale = 0.25

    # 모션
    motion_file: str = MOTION_FILES_DIR
    reference_body = "base"
    reset_strategy = "random"  # {"default", "random", "random-start"}

    # 시뮬레이션 (200 Hz physics, 50 Hz policy)
    # 시뮬레이션 (240 Hz physics, 60 Hz policy)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=5.0, replicate_physics=True)

    # 로봇
    robot: ArticulationCfg = R_SKELETON_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )
    # 자기 충돌 활성화 (상대적으로 계산량이 많으므로 환경 클래스에서 필터링 예정)
    # robot.spawn.articulation_props.enabled_self_collisions = True

    # 접촉 센서
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [0.0, 2.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }

    # Tracking 보상
    tracking_sigma = 0.25
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    # lin_vel_reward_scale = 1.0 * 1. / (.02 * 6)
    # yaw_rate_reward_scale = 0.5 * 1. / (.02 * 6)

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 AMP 환경 설정."""

from __future__ import annotations

import os

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

# 모션 파일 디렉토리 (stmr_go2.py 출력 경로와 일치해야 함)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset")
MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset_50")
# MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "new_dataset_single")


@configclass
class Go2AmpEnvCfg(DirectRLEnvCfg):
    """Go2 AMP imitation 학습 환경 설정.

    AMP 관측 벡터 구성 (amp_observation_space = 55):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        lin_vel(3) + ang_vel(3) +
        key_body_pos(12) + key_body_lin_vel(12)

    Policy 관측 벡터 구성 (observation_space = 42):
        projected_gravity_b(3) + commands(3) +
        joint_pos - default(12) + joint_vel(12) + actions(12)
    """

    # 에피소드
    episode_length_s = 20.0
    decimation = 4

    # 공간
    observation_space = 42
    action_space = 12            # Go2 DOF 수
    state_space = 0
    num_amp_observations = 2
    amp_observation_space = 55

    # History & Privileged
    history_observation = True
    history_len = 50
    priv_latent = True
    num_priv_obs = 74            # lin_vel(3)+ang_vel(3)+masses(17)+material(51) ≈ 74

    # 조기 종료
    early_termination = True
    termination_height = 0.15    # Go2 기본 높이 0.34m, 절반 이하이면 넘어짐으로 판정
    action_scale = 0.25

    # 모션
    motion_file: str = MOTION_FILES_DIR
    reference_body = "base"
    reset_strategy = "random"    # {"default", "random", "random-start"}

    # 시뮬레이션 (200 Hz physics, 50 Hz policy)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True
    )

    # 로봇
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

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
        "lin_vel_x_range": [0.0, 4.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }

    # Tracking 보상
    tracking_sigma = 0.25
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    # lin_vel_reward_scale = 1.0 * 1.0 / (0.02 * 6)
    # yaw_rate_reward_scale = 0.5 * 1.0 / (0.02 * 6)

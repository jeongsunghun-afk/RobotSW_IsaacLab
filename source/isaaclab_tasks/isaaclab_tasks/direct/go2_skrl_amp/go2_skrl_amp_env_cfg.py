# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 skrl AMP 환경 설정.

데이터: new_dataset_smr TXT → NPZ 변환 파일 사용
알고리즘: skrl AMP (humanoid_amp 스타일)
AMP obs: dof_pos(12) + dof_vel(12) + root_height(1) + lin_vel(3) + ang_vel(3) + key_body_pos(12) = 43
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from isaaclab.actuators import DCMotorCfg

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# convert_smr_to_npz.py 실행 후 생성되는 NPZ 파일 디렉토리
# MOTIONS_DIR = os.path.join(_THIS_DIR, "motions", "npz_dataset")
MOTIONS_DIR = os.path.join(_THIS_DIR, "motions", "smr_mirror_npz")
# MOTIONS_DIR = os.path.join(_THIS_DIR, "motions", "mimickit_npz")


@configclass
class Go2SkrlAmpEnvCfg(DirectRLEnvCfg):
    """Go2 skrl AMP 환경 기본 설정.

    AMP 관측 벡터 (amp_observation_space = 43):
        dof_pos(12) + dof_vel(12) + root_height(1) +
        lin_vel_b(3) + ang_vel_b(3) + key_body_pos_local(12)

    Policy 관측 벡터 (observation_space = 42):
        projected_gravity_b(3) + commands(3) +
        joint_pos_error(12) + joint_vel(12) + actions(12)
    """

    # 에피소드
    episode_length_s = 20.0
    decimation = 4  # 200Hz physics / 50Hz policy

    # 공간
    observation_space = 42
    action_space = 12
    state_space = 0
    num_amp_observations = 2      # skrl AMP 히스토리 길이 (humanoid_amp 스타일)
    amp_observation_space = 43    # 단일 프레임 AMP obs 크기

    # 조기 종료
    early_termination = True
    termination_height = 0.1
    roll_termination_deg = 70.0
    pitch_termination_deg = 70.0

    # 액션
    action_scale = 0.25

    # 모션 파일 (NPZ 디렉토리 또는 단일 파일)
    motion_file: str = MOTIONS_DIR
    reference_body = "base"
    reset_strategy = "random"   # {"default", "random", "random-start"}

    # 커맨드
    command_cfg = {
        "lin_vel_x_range": [0.0, 5.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-1.0, 1.0],
    }

    # 보상
    tracking_sigma = 0.5
    lin_vel_reward_scale = 1.0 / 0.02
    yaw_rate_reward_scale = 0.5 / 0.02

    # 시뮬레이션 (200Hz physics, 50Hz policy)
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200.0,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # 씬
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True
    )

    # 로봇
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=1.0,
                friction=0.0,
                armature=0.01
            ),
        },
    )

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

from isaaclab_assets.robots.rga import R_SKELETON_CFG


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (5.0, 5.0),
            "dynamic_friction_range": (2.0, 2.0),
            "restitution_range": (0.1, 0.1),
            "num_buckets": 64,
        },
    )


@configclass
class SkeletonWtwEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    resampling_time = 5.0
    dt = 1 / 200
    decimation = 4
    action_scale = 0.25
    action_space = 34
    clip_actions = 6.28
    hip_scale_reduction = False  # Set to false for skeleton unless requested otherwise

    priv_explicit = True
    priv_latent = True
    ang_vel = False
    friction_terrain = False
    timing_parameter = False
    clock_inputs = True
    prev_actions = False
    history_observation = True
    debug_vis = True

    # 3 (lin vel) + 14 (projected grav + commands) + action_space*3 (joint pos, joint vel, actions)
    num_prio_obs = 3 + 14 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4

    num_heights = 0

    num_priv = 3 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 10

    observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len

    state_space = 0

    penalized_body_names = [
        ".*_shoulder_y",
        ".*_shoulder_r",
        ".*_shoulder_p",
        ".*_thigh_y",
        ".*_thigh_r",
        ".*_thigh_p",
        ".*_elbow_p",
        ".*_knee_p",
        ".*_ankle_p",
        ".*_ankle_r",
        ".*_wrist_p",
        ".*_wrist_r",
        ".*_neck_p",
        ".*_neck_r",
        ".*_neck_y",
        ".*_waist_p",
        ".*_waist_r",
        ".*_waist_y",
    ]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=dt,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = R_SKELETON_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    gait_force_sigma = 100.0
    gait_vel_sigma = 10.0
    tracking_sigma = 0.25
    base_height_target = 0.608  # Updated base height for skeleton
    sigma_rew_neg = 0.02

    # reward scales
    lin_vel_reward_scale = 1.5
    yaw_rate_reward_scale = 0.75
    z_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.00001
    joint_accel_reward_scale = -1e-8
    action_rate_reward_scale = -0.01
    undesired_contact_reward_scale = -10.0
    jump_reward_scale = 5.0
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_cmd_linear_reward_scale = -30
    orientation_control_reward_scale = -5
    tracking_contacts_shaped_force_reward_scale = 1.0
    tracking_contacts_shaped_vel_reward_scale = 1.0
    dof_vel_reward_scale = -1e-5
    action_smoothness1_reward_scale = -0.1
    action_smoothness2_reward_scale = -0.1

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="add"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="add"),
    )

    # Command Definition
    num_commands = 14
    command_curriculum = True
    curriculum_threshold = 5.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [-0.5, 1.5],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
        "body_height_cmd_range": [-0.3, 0.15],
        "gait_frequency_cmd_range": [1.5, 3.0],
        "gait_phase_cmd_range": [0.0, 1.0],
        "gait_offset_cmd_range": [0.0, 1.0],
        "gait_bound_cmd_range": [0.0, 1.0],
        "gait_duration_cmd_range": [0.3, 0.7],
        "footswing_height_range": [0.03, 0.2],
        "body_pitch_range": [-0.0, 0.0],
        "body_roll_range": [-0.0, 0.0],
        "stance_width_range": [0.19, 0.20],  # 0.195
        "stance_length_range": [0.75, 0.85],
    }


@configclass
class SkeletonWtwRoughEnvCfg(SkeletonWtwEnvCfg):
    # env
    observation_space = 235

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=9,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

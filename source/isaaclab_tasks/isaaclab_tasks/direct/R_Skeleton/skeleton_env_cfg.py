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
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg, RayCasterCfg, patterns
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.rga import R_SKELETON_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


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

    # add_base_mass = EventTerm(
    #     func=mdp.randomize_rigid_body_mass,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="base"),
    #         "mass_distribution_params": (-1.0, 3.0),
    #         "operation": "add",
    #     },
    # )

    # robot_joint_stiffness_and_damping = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #       "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #       "stiffness_distribution_params": (0.75, 1.5),
    #       "damping_distribution_params": (0.3, 3.0),
    #       "operation": "scale",
    #       "distribution": "log_uniform",
    #     },
    # )


@configclass
class SkeletonEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 26
    clip_actions = 6.28

    priv_explicit = False
    priv_latent = False
    ang_vel = False
    friction_terrain = False
    timing_parameter = False
    clock_inputs = False
    prev_actions = False
    history_observation = False

    num_prio_obs = 3 + 3 + 3 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4
    num_heights = 0

    num_priv = 3 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 0

    observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len

    state_space = 0

    penalzied_body_names = [
        "base",
        "FL_link_1",
        "FL_link_2",
        "FL_link_3",
        "FL_link_4",
        "FL_link_5",
        "FL_link_6",
        "FR_link_1",
        "FR_link_2",
        "FR_link_3",
        "FR_link_4",
        "FR_link_5",
        "FR_link_6",
        "HL_link_1",
        "HL_link_2",
        "HL_link_3",
        "HL_link_4",
        "HL_link_5",
        "HR_link_1",
        "HR_link_2",
        "HR_link_3",
        "HR_link_4",
        "HR_link_5",
    ]

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -1.0
    similar_to_default_reward_scale = -1.0
    base_height_reward_scale = -10.0

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # Command Definition
    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [0.0, 1.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [0.0, 1.0]
    lin_vel_y_range = [-0.0, 0.0]
    ang_vel_range = [-0.5, 0.5]


@configclass
class SkeletonHistoryEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 34
    clip_actions = 6.28

    priv_explicit = True
    priv_latent = True
    ang_vel = False
    friction_terrain = False
    timing_parameter = False
    clock_inputs = False
    prev_actions = False
    history_observation = True

    num_prio_obs = 3 + 3 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4

    num_heights = 0

    num_priv = 3 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 10

    # observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len
    observation_space = num_prio_obs

    state_space = 0

    penalized_contact_link_names = [
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
        dt=1 / 200,
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

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -10.0
    flat_orientation_reward_scale = -1.0
    similar_to_default_reward_scale = -0.01
    base_height_reward_scale = -10.0

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # Command Definition
    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [0.0, 0.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.0, 0.0],
    }


@configclass
class SkeletonHistoryFixedEnvCfg(SkeletonHistoryEnvCfg):
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 34
    clip_actions = 6.28

    priv_explicit = True
    priv_latent = True
    ang_vel = False
    friction_terrain = False
    timing_parameter = False
    clock_inputs = False
    prev_actions = False
    history_observation = True

    num_prio_obs = 3 + 3 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4

    num_heights = 0

    num_priv = 3 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 10

    # observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len
    observation_space = num_prio_obs

    state_space = 0

    penalized_contact_link_names = [
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
        dt=1 / 200,
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
    # FrameTransformer: Fixed joint으로 병합된 발끝(toe) 위치를 각 발의 마지막 링크 기준으로 추적합니다.
    # 오프셋은 원본 USD에서 시뮬레이션 측정한 값입니다.
    # 앞다리(FL/FR): wrist_r → toe, 뒷다리(HL/HR): ankle_r → toe
    foot_frame: FrameTransformerCfg = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/FL_link6_wrist_r",
                name="FL_toe",
                offset=OffsetCfg(pos=(-0.0382, 0.0, -0.0993)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/FR_link6_wrist_r",
                name="FR_toe",
                offset=OffsetCfg(pos=(0.0382, 0.0, -0.0993)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/HL_link6_ankle_r",
                name="HL_toe",
                offset=OffsetCfg(pos=(-0.0142, 0.0, -0.1488)),
            ),
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/HR_link6_ankle_r",
                name="HR_toe",
                offset=OffsetCfg(pos=(0.0127, 0.0, -0.1489)),
            ),
        ],
    )

    # reward scales
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -0.5
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-8
    action_rate_reward_scale = -0.01
    # feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -10.0
    flat_orientation_reward_scale = -1.0
    similar_to_default_reward_scale = -0.01
    # base_height_reward_scale = -10.0

    sigma_rew_neg = 0.02

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )

    # at every time-step add gaussian noise + bias. The bias is a gaussian sampled at reset
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # Command Definition
    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [0.0, 2.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }


@configclass
class SkeletonRoughEnvCfg(SkeletonEnvCfg):
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

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0

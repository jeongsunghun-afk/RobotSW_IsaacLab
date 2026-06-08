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
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.rga import HIND_LEG_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {
                "x": (-0.08, 0.08),
                "y": (-0.04, 0.04),
                "z": (-0.02, 0.02),
            },
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

@configclass
class HindLegFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 1
    action_scale = 0.25
    action_space = 26

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
        dt=1 / 40,
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
    robot: ArticulationCfg = HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
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
    action_rate_reward_scale = -0.001
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -1.0
    similar_to_default_reward_scale = -0.1
    base_height_reward_scale = -10.0
    termination_reward_scale = -100.0
    # velocity-gated foot clearance: penalizes low foot height when foot has horizontal velocity
    # (foot_z - target_height)^2 * tanh(tanh_mult * foot_v_xy); scale < 0 → penalty
    foot_clearance_reward_scale = -0.5
    foot_clearance_tanh_mult = 2.0
    foot_clearance_offset = 0.05  # metres above measured rest-z to set as target clearance

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
class HindLegHistoryEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 8

    priv_explicit = True
    priv_latent = True
    ang_vel = False
    friction_terrain = True
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

    num_priv = 6 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 10

    # observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len
    observation_space = num_prio_obs

    state_space = 0

    penalzied_body_names = [
        "base",
        ".*hip.*",
        ".*thigh.*",
        ".*calf.*"
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
    robot: ArticulationCfg = HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")
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
    action_rate_reward_scale = -0.001
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -0.0
    similar_to_default_reward_scale = -0.1
    base_height_reward_scale = -10.0
    termination_reward_scale = -100.
    # velocity-gated foot clearance: penalizes low foot height when foot has horizontal velocity
    # (foot_z - target_height)^2 * tanh(tanh_mult * foot_v_xy); scale < 0 → penalty
    foot_clearance_reward_scale = -0.5
    foot_clearance_tanh_mult = 2.0
    foot_clearance_offset = 0.05  # metres above measured rest-z to set as target clearance

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
        "lin_vel_x_range": [-0.5, 2.0],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [-0.5, 2.0]
    lin_vel_y_range = [-0.0, 0.0]
    ang_vel_range = [-0.5, 0.5]


@configclass
class HindLegRoughEnvCfg(HindLegFlatEnvCfg):
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

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
# from isaaclab_assets.robots.rga import MOTION_JIG_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.rga import RGA_GO2_CFG


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 10.0),
            "operation": "add",
        },
    )

    randomize_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.15, 0.15), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
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


@configclass
class Go2FlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 20.0
    resampling_time = 5.0
    dt = 1 / 200
    decimation = 4
    action_scale = 0.25
    action_space = 12
    clip_actions = 10.0
    hip_scale_reduction = True

    priv_explicit = False
    priv_latent = True
    ang_vel = False
    friction_terrain = False
    timing_parameter = False
    clock_inputs = True
    prev_actions = False
    history_observation = True
    debug_vis = True

    num_prio_obs = 3 + 14 + action_space * 3

    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4

    num_heights = 0

    num_priv = 3 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 20

    observation_space = num_prio_obs + num_heights + num_priv + num_priv_latent + num_prio_obs * history_len

    state_space = 0

    penalized_body_names = ["base", ".*thigh", ".*calf", ".*hip"]

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
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    gait_force_sigma = 100.0
    gait_vel_sigma = 10.0
    tracking_sigma = 0.25  # 0.25, 0.125 사용함
    base_height_target = 0.34
    sigma_rew_neg = 0.02
    # reward scales
    lin_vel_reward_scale = 1.2
    yaw_rate_reward_scale = 0.7
    z_vel_reward_scale = -0.02
    ang_vel_reward_scale = -0.001
    # joint_torque와 action_rate 페널티를 절반 정도로 줄여 다리가 힘을 더 잘 내게 함
    joint_torque_reward_scale = -0.0001
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.005
    undesired_contact_reward_scale = -10.0
    jump_reward_scale = 10.0
    raibert_heuristic_reward_scale = -10.0
    feet_clearance_cmd_linear_reward_scale = -1.0  # 목표 발 높이 추종 (swing phase)
    feet_clearance_bezier_reward_scale = 0.0  # 3차 베지에 추종 (ablation: 3차 실험 시 -5.0으로 활성화)
    feet_clearance_bezier_5th_reward_scale = -0.0  # 비활성화 (ablation: 5차 실험 시 -5.0으로 활성화)
    orientation_control_reward_scale = -5
    tracking_contacts_shaped_force_reward_scale = 1.0
    tracking_contacts_shaped_vel_reward_scale = 1.0
    dof_vel_reward_scale = -1e-5
    action_smoothness1_reward_scale = -0.1
    action_smoothness2_reward_scale = -0.1
    foot_landing_vel_reward_scale = -3.0
    foot_landing_vel_xy_reward_scale = -2.0  # 착지 순간 수평 속도 패널티 (발 구르기 방지)
    landing_impact_reward_scale = 0.0  # Force 노이즈 기반 → 비활성화 (foot_landing_vel로 대체)
    feet_vel_5th_late_reward_scale = -0.0  # 5차 베지에 Z velocity target 추종 (swing 후반 s>0.7), 비교용

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
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
        "body_height_cmd_range": [-0.25, 0.1],
        "gait_frequency_cmd_range": [2.0, 4.0],
        "gait_phase_cmd_range": [0.0, 1.0],
        "gait_offset_cmd_range": [0.0, 1.0],
        "gait_bound_cmd_range": [0.0, 1.0],
        "gait_duration_cmd_range": [0.5, 0.5],
        "footswing_height_range": [0.05, 0.35],
        "body_pitch_range": [-0.4, 0.4],
        "body_roll_range": [-0.0, 0.0],
        "stance_width_range": [0.10, 0.45],
        "stance_length_range": [0.35, 0.45],
    }

    command_cfg = {
        "lin_vel_x_range": [-1.0, 2.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-1.0, 1.0],
        "body_height_cmd_range": [-0.2, 0.1],
        "gait_frequency_cmd_range": [1.5, 4.0],
        "gait_phase_cmd_range": [0.0, 1.0],
        "gait_offset_cmd_range": [0.0, 1.0],
        "gait_bound_cmd_range": [0.0, 1.0],
        "gait_duration_cmd_range": [0.3, 0.7],
        "footswing_height_range": [0.05, 0.35],
        "body_pitch_range": [-0.3, 0.3],
        "body_roll_range": [-0.0, 0.0],
        "stance_width_range": [0.1, 0.45],
        "stance_length_range": [0.35, 0.45],
    }


@configclass
class Go2RoughEnvCfg(Go2FlatEnvCfg):
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


@configclass
class Go2NeckFlatEnvCfg(Go2FlatEnvCfg):
    # env
    whole_body_control: bool = False

    action_space = 19  # 12 (legs) + 7 (neck)

    clock_inputs = True
    # observation
    # num_prio_obs = 3 (lin_vel) + 14 (projected_gravity + commands) + 19 (joint_pos) + 19 (joint_vel) + 19 (actions)
    num_prio_obs = 3 + 14 + action_space * 3  # 74

    if clock_inputs:
        num_prio_obs += 4

    history_len = 20

    observation_space = num_prio_obs + (num_prio_obs * history_len)  # 74 + 74 * 20 = 1554

    # robot
    robot = RGA_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # update penalized body names to include neck if necessary
    penalized_body_names = ["base", ".*thigh", ".*calf", ".*hip", ".*neck_p", ".*neck_r", ".*neck_y"]
    # neck action smoothness
    neck_alpha = 0.1

    def __post_init__(self):
        super().__post_init__()
        # Adjust dimensions based on control mode
        self.action_space = 19 if self.whole_body_control else 12
        # obs is composed of: gravity(3) + commands(14) + joint_pos(19) + joint_vel(19) + actions(action_space)
        self.num_prio_obs = 3 + 14 + 19 * 2 + self.action_space
        if self.clock_inputs:
            self.num_prio_obs += 4
        self.observation_space = self.num_prio_obs + (self.num_prio_obs * self.history_len)

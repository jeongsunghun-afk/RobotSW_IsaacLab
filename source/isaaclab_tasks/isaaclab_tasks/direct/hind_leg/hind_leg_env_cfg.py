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
from isaaclab_assets.robots.rga import FLAT_HIND_LEG_CFG, HIND_LEG_CFG  # isort: skip
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
            "damping_distribution_params": (0.75, 1.5),
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
    similar_to_default_reward_scale = -0.01
    base_height_reward_scale = -10.0
    termination_reward_scale = -100.0
    # (A) Phase-scheduled stance reward: bonus when scheduled-stance foot is in contact (scale > 0)
    # (B) Phase-scheduled swing clearance reward: bonus when scheduled-swing foot is lifted (scale > 0)
    # (C) Contact-gated anti-slip penalty: stance foot horizontal velocity → penalty (scale < 0)
    foot_slip_reward_scale = -0.15  # < 0: penalty; at 0.93 m/s → ~0.13 per foot per step
    # Gait clock parameters
    gait_period = 0.6  # seconds; full gait cycle duration
    gait_swing_height = 0.07  # metres; clearance ramp saturates at this lift above sole_rest_z
    gait_phase_sharpness = 4.0  # tanh sharpness; higher = more square-wave swing/stance boundary
    gait_stance_reward_scale = 1.0  # > 0: bonus
    gait_swing_reward_scale = 1.0  # > 0: bonus
    # Standing detection thresholds: when command magnitude is below these values, gait is suppressed
    standing_vel_threshold = 0.1  # ‖cmd_xy‖ below this → standing candidate (matches feet_air_time gate)
    standing_yaw_threshold = 0.1  # |yaw command| below this → standing confirmed
    rel_standing_envs: float = 0.05  # fraction of resampled envs forced to cmd=0 (standing) each resample step

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
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [0.0, 1.0]
    lin_vel_y_range = [-0.5, 0.5]
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
    clock_inputs = True
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

    penalzied_body_names = ["base", ".*hip.*", ".*thigh.*", ".*calf.*"]

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
    termination_reward_scale = -100.0
    # (A) Phase-scheduled stance reward: bonus when scheduled-stance foot is in contact (scale > 0)
    # (B) Phase-scheduled swing clearance reward: bonus when scheduled-swing foot is lifted (scale > 0)
    # (C) Contact-gated anti-slip penalty: stance foot horizontal velocity → penalty (scale < 0)
    foot_slip_reward_scale = -0.15  # < 0: penalty; at 0.93 m/s → ~0.13 per foot per step
    # Gait clock parameters
    gait_period = 0.6  # seconds; full gait cycle duration
    gait_swing_height = 0.07  # metres; clearance ramp saturates at this lift above sole_rest_z
    gait_phase_sharpness = 4.0  # tanh sharpness; higher = more square-wave swing/stance boundary
    gait_stance_reward_scale = 1.0  # > 0: bonus
    gait_swing_reward_scale = 1.0  # > 0: bonus
    # Standing detection thresholds: when command magnitude is below these values, gait is suppressed
    standing_vel_threshold = 0.1  # ‖cmd_xy‖ below this → standing candidate (matches feet_air_time gate)
    standing_yaw_threshold = 0.1  # |yaw command| below this → standing confirmed
    rel_standing_envs: float = 0.1  # fraction of resampled envs forced to cmd=0 (standing) each resample step

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
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [-0.5, 2.0]
    lin_vel_y_range = [-0.5, 0.5]
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


@configclass
class HindLegFlatFootEnvCfg(HindLegHistoryEnvCfg):
    """평발(2점 접촉) 변형 — 로봇만 FLAT_HIND_LEG_CFG로 교체, 나머지(obs/action/reward/command)는 동일."""

    robot: ArticulationCfg = FLAT_HIND_LEG_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # --- Reward rebalance #1 (plateau fix, 2026-07-24): make WALKING pay more than STANDING ---
    # Diagnosis: gait_stance (~1.02) dominated the return; robot stood on 2 feet for the bonus.
    # --- Reward rebalance #2 (clean-gait fix, 2026-07-27): keep walking, add rhythmic stepping ---
    # Diagnosis: rebalance #1 broke the plateau but the gait is shuffly/irregular (feet drag, no
    #   clean alternating swing). Fix: strengthen the phase-scheduled swing-clearance term and the
    #   feet-air-time term (the two drivers of clean stepping) + a moderate smoothness bump, while
    #   KEEPING lin_vel high (3.0) so it does not revert to standing. All are attribute overrides in
    #   this FlatFoot subclass only -- the point-foot task (HindLegHistoryEnvCfg) is untouched, and
    #   every term already exists in hind_leg_env.py (no env.py edit, fully backward-compatible).
    lin_vel_reward_scale = 3.0        # base 1.0 -> 3.0: 3x boost on velocity tracking            [kept from #1]
    lin_vel_tracking_sigma = 0.25     # base 0.1 -> 0.25: wider exp kernel -> gradient toward moving [kept from #1]
    rel_standing_envs: float = 0.05   # base 0.1 -> 0.05: fewer forced-standing envs               [kept from #1]
    # Gait-shaping (rhythmic-stepping drivers) -- strengthened in #2:
    gait_stance_reward_scale = 0.8    # #1 set 0.5 (from base 1.0); #2 -> 0.8 (anchor stance foot on schedule)
    gait_swing_reward_scale = 1.5     # base 1.0 -> 1.5: reward lifting the SWING foot to clearance (kills drag/shuffle)
    feet_air_time_reward_scale = 1.0  # base 0.5 -> 1.0: classic clean-stepping term (air-time > 0.2s, cmd-gated)
    # Smoothness -- strengthened in #2 (kept moderate so it does not suppress stepping into a shuffle):
    action_rate_reward_scale = -0.005 # base -0.001 -> -0.005: penalise jittery action changes
    # foot_slip (-0.15), joint_accel (-2.5e-7) and self-collision (FLAT_HIND_LEG_CFG) all kept as-is.
    # --- Reward rebalance #3 (limp fix, 2026-07-27): symmetric, flat-footed gait ---
    # Measured limp @ vx~0.2: one foot (HR) plants FLAT and hogs stance (~79%), the other (HL)
    # TOE-WALKS (heel raised 0.10-0.15 m) with only ~52% stance. Root cause: gait_stance rewards
    # stance-contact per foot but NOTHING penalises a foot for staying in contact through its own
    # SWING phase, and no term constrains foot pitch → one foot can park in stance while the other
    # toe-walks, both locally optimal. Two new penalties (both instantaneous, no reset buffer; gated
    # behind cfg attrs via getattr in env.py so the point-foot task HindLegHistoryEnvCfg is unaffected):
    gait_swing_contact_reward_scale = -1.0  # penalise contact during scheduled swing → forces BOTH feet
    #                                         to lift on their 50/50 antiphase schedule (fixes lopsided duty)
    flat_foot_reward_scale = -15.0          # penalise heel-above-toe gap (m) while loaded → heel+toe both
    #                                         down = flat foot; kills the toe-walking foot
    # Commands: forward, achievable, no ambiguous near-zero walk speed (explicit stop handled by rel_standing_envs)
    command_cfg = {
        "lin_vel_x_range": [0.1, 0.3],   # was [-0.5, 2.0]: min 0.3 (no ambiguous ~0), max 1.0 (reachable)
        "lin_vel_y_range": [-0.3, 0.3],  # was [-0.5, 0.5]
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [0.1, 0.3]
    lin_vel_y_range = [-0.3, 0.3]
    ang_vel_range = [-0.5, 0.5]

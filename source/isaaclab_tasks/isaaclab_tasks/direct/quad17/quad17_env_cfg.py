# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the DTC Phase-1 17-DOF quadruped velocity-tracking task (flat terrain).

Adapted from ``direct/hind_leg`` (2-leg biped, ActorCriticRMA + estimator + RMA obs groups),
extended 2 -> 4 legs + waist. Per-leg DOF (hip/thigh/calf/foot) matches the hind_leg base
exactly, so the observation/reward machinery is reused almost verbatim; only the foot-body
count (2 -> 4), the trot gait clock (diagonal pairs), and the base body name ("Base") differ.
"""

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.rga import QUAD_17DOF_CFG  # isort: skip


# ------------------------------------------------------------------------------------------------
# Bent-knee crouch default pose (17 DOF).
# Front vs rear thigh/calf axis signs are OPPOSITE (see MJCF), so a symmetric crouch uses opposite
# signs front vs rear. All values are inside the joint ranges:
#   rear (HL/HR): thigh [-2.356, 1.134]  calf [-0.959, 1.134]  foot [-1.396, 0.698]
#   front (FL/FR): thigh [-2.705, 1.309]  calf [-1.221, 0.785]  foot [-0.785, 1.571]
# Hips and waist stay at 0. Mild crouch (keeps base near the baked ~0.52 spawn height).
# ------------------------------------------------------------------------------------------------
CROUCH_JOINT_POS = {
    ".*_hip_joint": 0.0,
    "FB_waist_joint": 0.0,
    # rear legs
    "HL_thigh_joint": 0.6,
    "HR_thigh_joint": 0.6,
    "HL_calf_joint": -0.7,
    "HR_calf_joint": -0.7,
    "HL_foot_joint": 0.2,
    "HR_foot_joint": 0.2,
    # front legs (opposite sign)
    "FL_thigh_joint": -0.6,
    "FR_thigh_joint": -0.6,
    "FL_calf_joint": 0.7,
    "FR_calf_joint": 0.7,
    "FL_foot_joint": -0.2,
    "FR_foot_joint": -0.2,
}


@configclass
class EventCfg:
    """Domain randomization (reused from hind_leg; body name 'Base')."""

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
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
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
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
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
class Quad17VelocityEnvCfg(DirectRLEnvCfg):
    """RMA velocity-tracking on flat ground (mirrors HindLegHistoryEnvCfg, 2->4 legs + waist)."""

    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 17  # 16 leg joints + 1 waist joint (waist is actuated; see report)

    # RMA / privileged obs switches (same layout hind_leg's ActorCriticRMA expects)
    priv_explicit = True
    priv_latent = True
    ang_vel = False
    friction_terrain = True
    timing_parameter = False
    clock_inputs = True
    prev_actions = False
    history_observation = True

    # -- DTC Phase-2: foothold-tracking (procedural randomized footholds) --
    # Turns the P1 velocity walker into a foothold tracker WITHOUT a new network: the reference
    # footholds are stacked onto the policy obs (+28) and a touchdown-tracking reward rewards
    # landing the sole on the target. Reference = per-foot Raibert placement + decorrelated jitter
    # (the jitter is what makes the channel non-redundant with cmd/clock — see quad17_env).
    foothold_obs = True  # append the 28-dim foothold reference block to the policy obs
    num_foothold_targets = 2  # current target + one-step look-ahead (obs stacks both)
    # DTC Fig-7D enabler: also append a desired-joint-position (IK) block — per-leg first-order
    # Jacobian-inverse joint delta toward the current foothold target (4 legs x hip/thigh/calf = +12).
    # Ablation proved the policy ignores the Cartesian foothold alone; the IK signal is what lets it
    # map a Cartesian target to joint actions. Blinded together with the Cartesian block (see env).
    foothold_ik_obs = True
    foothold_ik_joints_per_leg = 3  # hip/thigh/calf per leg -> 4 * 3 = +12 obs dims
    foothold_track_reward_scale = 1.0
    foothold_track_eps = 0.01  # -log(err2 + eps): eps caps the peak reward at -log(eps)=4.6
    foothold_err2_max = 1.0  # clamp err2 so a single far-miss can't dominate the unbounded-below log
    foothold_T_stance = 0.5  # Raibert stance duration for the placement offset (= gait_period)
    foothold_max_age = 0.75  # s; force one target advance if a scheduled touchdown never fires
    foothold_jitter_xy = 0.05  # m; decorrelated horizontal jitter (MANDATORY, breaks redundancy)
    foothold_jitter_z = 0.02  # m; vertical jitter

    # -- DTC offline / APT-RL: offline TAMOLS foothold cache (A/B against the Raibert path above) --
    # When True, ``_regen_footholds`` looks up precomputed TAMOLS-planned footholds from ``./tamols_cache``
    # (indexed by the commanded forward vx + the local x-distance to the next gap near-edge) and re-anchors
    # them to the robot's live base pose, giving terrain-derived gap-straddling targets instead of the
    # redundant base-relative Raibert ones. The cache is a FORWARD-vx-only plan, so the command
    # distribution is restricted to forward motion (vy=yaw=0, vx clamped to the cached grid) while it is
    # active (see Quad17Env.__init__). Set False to fall back to the intact Raibert path for an A/B run.
    # If the cache dir is missing at runtime it silently falls back to Raibert (see _load_tamols_cache).
    use_tamols_cache: bool = True

    # -- DTC Phase-2 gap-test terrain (physical "must read obs" pressure) --
    # On FLAT ground the touchdown reward has no physical consequence for missing a foothold, so the
    # policy ignores the foothold obs (proven by the ablation gate: none≈scramble). This toggle adds
    # randomized periodic TRANSVERSE trenches (gaps along forward-x, running full-width in y): the
    # robot must step on solid strips or a foot physically drops into a trench and it destabilizes.
    # Foothold targets are SNAPPED to the nearest solid-strip center (see _regen_footholds), so the
    # ONLY way to know where solid ground is = read the (snapped) foothold obs. Because the policy obs
    # carries NO absolute world position, the static-but-non-uniform strip pattern is unpredictable to
    # the policy -> the foothold channel becomes necessary. Toggle off (env-var QUAD17_GAP_TERRAIN=0)
    # to fall back to the flat plane. Trench geometry is built in Quad17Env._build_gap_terrain.
    gap_terrain = True  # True = transverse trenches; False = flat plane (original P1/P2 behaviour)
    gap_strip_width = 0.35  # m; solid strip width along x (foothold snap target = strip center)
    gap_width = 0.18  # m; nominal trench width (kept for the spawn_dx bound; actual width is the ramp below)
    gap_depth = 0.30  # m; strip height above the trench floor = drop depth when a foot misses (>=0.25)
    gap_spacing_jitter = 0.06  # m; +/- random jitter on each gap width (breaks a memorizable period)
    gap_forward_corridor = 12.0  # m; how far forward of the spawn rows the trench field extends
    # -- PER-BAND spatial gap-difficulty curriculum (gaps widen with forward distance; see _build_gap_terrain) --
    # gap_w(x) = gap_width_min + (gap_width_max - gap_width_min) * clip((x - edge - curric_x_start)/curric_x_ramp, 0, 1)
    # where edge = the leading edge of the platform band immediately BEHIND the strip (the ramp resets at
    # every platform), so x-edge = distance into that corridor. Easy just past every platform (all rows
    # start easy), hard toward the corridor's far end. strip_width fixed; only the gap ramps. Build-time
    # env-var overrides: GAP_WIDTH_MIN / GAP_WIDTH_MAX / CURRIC_X_START / CURRIC_X_RAMP.
    gap_width_min = 0.05  # m; trivial gap right past each platform (robot first learns to walk forward)
    gap_width_max = 0.20  # m; target gap toward the corridor far end (~the pre-curriculum uniform 0.18 m)
    curric_x_start = 0.3  # m; first ~0.3 m past each platform stays easy before that corridor's ramp begins
    curric_x_ramp = 0.0  # m; 0 = auto per-corridor = max(2.0, 0.7*corridor_len); >0 pins a fixed ramp span
    foot_in_gap_reward_scale = -1.0  # mild penalty: a stance foot dropped below the strip surface
    # forward-only command for the gap test (must cross trenches); applied in Quad17Env.__init__.
    gap_lin_vel_x_range = [0.4, 0.8]

    # -- DTC Eq1 base-pose (position) tracking reward (anti-hesitation) --
    # Fixes the "stand still on the spawn platform to avoid gaps" local optimum: velocity tracking
    # alone is too weak to beat fall-avoidance, so the robot hesitates (track_lin_vel≈0.13, epLen 999,
    # never reaches the trenches). DTC Eq 1 tracks a reference base trajectory that ADVANCES forward at
    # the commanded velocity; standing still accumulates position error -> a strong, dense penalty that
    # forces forward motion into the trenches (where the foothold obs + gap drop already teach tracking).
    # exp(-||base_xy - ref_xy||^2 / sigma) * scale * dt. Env-var overrides: BASE_POSE_SCALE / BASE_POSE_SIGMA.
    base_pose_track_reward_scale = 3.0  # strong — must beat the standing local optimum
    base_pose_track_sigma = 0.25  # m^2 error scale for the exp kernel
    # v3 anti-hesitation additions (exp alone has ~zero gradient once the robot falls behind):
    #  - progress_fail_dist: terminate if the base falls this far behind the advancing reference, so
    #    standing still = falling behind = death (removes the "stand still is safe" local optimum).
    #    The reference is anchored so err=0 at reset, so this does NOT fire at step 0.
    #  - progress_reward_scale: LINEAR reward on clipped forward base velocity — a constant forward
    #    gradient that pulls a hesitant robot forward where exp() is flat. Env-vars PROGRESS_FAIL_DIST,
    #    PROGRESS_SCALE.
    # Loosened 0.8 -> 1.5 for the curriculum: give the policy room to learn to walk in the easy near-spawn
    # region before it is killed for lagging behind the advancing reference.
    progress_fail_dist = 1.5  # m behind the reference before the episode is terminated
    progress_reward_scale = 1.0  # linear forward-velocity progress reward (bounded, ungameable)

    # policy obs width: projected_gravity(3) + commands(3) + [joint_pos-def, joint_vel, actions] (3*action_space)
    num_prio_obs = 3 + 3 + action_space * 3
    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4  # trot clock: (sin,cos) for each of the 2 diagonal phase groups
    if foothold_obs:
        # 2 targets x 4 feet x 3 (body-frame xyz) = 24, plus 4 per-foot target ages = 28
        num_prio_obs += num_foothold_targets * 4 * 3 + 4
        if foothold_ik_obs:
            num_prio_obs += 4 * foothold_ik_joints_per_leg  # +12 IK desired-joint block

    num_heights = 0
    num_priv = 6 if priv_explicit else 0
    num_friction = 1 if friction_terrain else 31
    num_priv_latent = 4 + num_friction if priv_latent else 0
    history_len = 10

    # DirectRLEnv "policy" gym space = policy obs width. The privileged/history groups are returned
    # as extra dict keys and their dims are inferred at runtime by the RMA runner (obs_groups).
    observation_space = num_prio_obs
    state_space = 0

    # bodies that should not touch the ground (undesired-contact penalty). "Base" = capital (MJCF root).
    penalzied_body_names = ["Base", ".*hip.*", ".*thigh.*", ".*calf.*"]

    # simulation (200 Hz physics, decimation 4 -> 50 Hz control)
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

    # robot — QUAD_17DOF_CFG with a bent-knee crouch default pose
    # Spawn height MUST be set explicitly: Isaac Lab derives ``default_root_state`` from
    # ``init_state.pos`` (NOT the USD's baked ~0.5235 Base transform), and ``_reset_idx`` writes that
    # world pose every episode — overriding the baked transform. With pos.z=0 the base was planted on
    # the ground (base contact force ~272 N > 1 N death threshold) and the crouched feet spawned ~0.55 m
    # underground, so every episode died on step 1 (epLen pinned at 1.0). z=0.57 lifts the base clear of
    # the ground and lands the crouched feet at ground level (feet hang ~0.55 m below the base).
    robot: ArticulationCfg = QUAD_17DOF_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=QUAD_17DOF_CFG.init_state.replace(joint_pos=CROUCH_JOINT_POS, pos=(0.0, 0.0, 0.57)),
    )
    # NOTE: the quad USD nests all rigid bodies under an extra "Base" Xform (articulation root),
    # so links are at /Robot/Base/<link> (one level deeper than hind_leg's /Robot/<link>).
    # find_matching_prims matches a single path segment per ".*", so the pattern must reach that depth.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Base/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # reward scales
    # P2: lin_vel 1.0 -> 0.5 so the foothold-track reward is not drowned out, while keeping a dense
    # move signal (do NOT zero it — zeroing collapses to the standing local optimum).
    lin_vel_reward_scale = 0.5
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.01
    joint_torque_reward_scale = -0.0002
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.005
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0
    flat_orientation_reward_scale = -1.0
    similar_to_default_reward_scale = -0.05
    base_height_reward_scale = -10.0
    base_height_target = 0.52  # explicit target (baked spawn height); avoids default_root_state ambiguity
    termination_reward_scale = -100.0
    # phase-scheduled gait terms (trot: 4 feet, diagonal pairs anti-phase)
    foot_slip_reward_scale = -0.15
    gait_period = 0.5
    gait_swing_height = 0.08
    gait_phase_sharpness = 4.0
    gait_stance_reward_scale = 0.5
    gait_swing_reward_scale = 1.0
    lin_vel_tracking_sigma = 0.25
    # standing detection (suppress gait when command ~0)
    standing_vel_threshold = 0.1
    standing_yaw_threshold = 0.1
    rel_standing_envs: float = 0.05

    # per-step action noise + reset bias
    action_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.05, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.015, operation="abs"),
    )
    observation_noise_model: NoiseModelWithAdditiveBiasCfg = NoiseModelWithAdditiveBiasCfg(
        noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.002, operation="add"),
        bias_noise_cfg=GaussianNoiseCfg(mean=0.0, std=0.0001, operation="abs"),
    )

    # command definition (vx, vy, yaw-rate)
    num_commands = 3
    command_curriculum = False
    curriculum_threshold = 10.0
    curriculum_step = 0.05
    command_cfg = {
        "lin_vel_x_range": [-0.5, 1.0],
        "lin_vel_y_range": [-0.5, 0.5],
        "ang_vel_range": [-0.5, 0.5],
    }
    lin_vel_x_range = [-0.5, 1.0]
    lin_vel_y_range = [-0.5, 0.5]
    ang_vel_range = [-0.5, 0.5]

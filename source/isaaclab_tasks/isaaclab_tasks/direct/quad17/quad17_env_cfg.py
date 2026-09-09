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
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg, NoiseModelWithAdditiveBiasCfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.rga import LEG_DTC_CFG  # isort: skip


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
    # -- DTC adaptive terrain-LEVEL curriculum (IsaacLab game-inspired; OFF => the spatial ramp above) --
    # When True, _build_gap_terrain lays out ``num_terrain_levels`` corridors as spatial ROWS stacked along
    # +y (each a lane running along +x). Level 0 = FLAT (fully paved, no gaps); level L carves gaps whose
    # width scales L/(num_terrain_levels-1) from ~0 up to ``gap_width_max``. Each env carries a persistent
    # ``_terrain_level`` (init 0 => everyone starts flat) and spawns in its level's lane; at episode end it
    # is PROMOTED a level if its forward progress this episode exceeded ``promote_frac`` of the corridor
    # length, DEMOTED if below ``demote_frac`` (clamped [0, num_terrain_levels-1]). The offline TAMOLS cache
    # lookup (_next_gap_dist) is made PER-ENV by its level's gap edges/widths so the foothold plan matches
    # the lane the robot is actually on. Default OFF => byte-for-byte the per-band spatial-ramp behaviour
    # above (the currently-running training is unaffected). Only takes effect together with gap_terrain=True
    # (it extends the gap builder). Build-time knobs are cfg-only (no env-var overrides).
    terrain_curriculum: bool = False
    num_terrain_levels: int = 10  # difficulty rows; level 0 = flat, level (n-1) = gap_width_max
    terrain_curriculum_corridor_len = 8.0  # m; forward length of each level's gap corridor (past the platform)
    terrain_curriculum_lane_pitch = 4.0  # m; +y spacing between difficulty lanes (keeps a robot inside its lane)
    promote_frac = 0.8  # advance a level if forward progress this episode > promote_frac * corridor_len
    demote_frac = 0.4  # regress a level if forward progress this episode < demote_frac * corridor_len
    foot_in_gap_reward_scale = -1.0  # mild penalty: a stance foot dropped below the strip surface
    # forward-only command for the gap test (must cross trenches); applied in Quad17Env.__init__.
    gap_lin_vel_x_range = [0.4, 0.8]

    # -- Terrain KIND selector: gap trenches vs ascending stairs (3D-terrain DTC test) --
    # "gap"   => the transverse-trench curriculum above (DEFAULT; byte-for-byte the current behaviour).
    # "stair" => _build_stair_terrain_curriculum: the SAME num_terrain_levels +y lanes / per-env
    #            _terrain_level / promote-demote / spawn-in-lane machinery, but every level L is an
    #            ASCENDING STAIRCASE instead of a trench field — level 0 = FLAT, level L step rise =
    #            L/(num_terrain_levels-1) * stair_step_height_max at a fixed tread depth stair_step_depth.
    #            Stairs reuse the curriculum lanes, so they take effect only with terrain_curriculum=True;
    #            env-var QUAD17_TERRAIN_KIND=stair force-enables the curriculum (exactly as
    #            QUAD17_TERRAIN_CURRICULUM=1 force-enables gap_terrain). Default "gap" => obs/behaviour unchanged.
    terrain_kind: str = "gap"  # "gap" | "stair"
    stair_step_height_max = 0.15  # m; step rise at the hardest lane (level n-1); level 0 = flat (0 rise)
    stair_step_depth = 0.35  # m; fixed tread depth (run) along +x for every stair level

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

    # -- Heightmap terrain perception (RayCaster height scanner; mirrors direct/parkour) --
    # When ON, a GridPattern RayCaster reads the terrain under/ahead of the base and its per-ray
    # relative heights (base_z - hit_z, clipped) are appended to the policy obs. Default OFF => the
    # sensor is NOT even created (see Quad17Env._setup_scene) and the obs is byte-for-byte unchanged
    # (obs=101). Env-var QUAD17_HEIGHTMAP=1 flips it on early in __init__ (before the scene / obs-dim
    # are built). See Quad17Env._height_scan_obs and the height_scanner field below.
    use_heightmap: bool = False
    height_scan_resolution = 0.1  # m; GridPattern cell size (parkour value)
    height_scan_size = [1.6, 1.0]  # m; [x, y] extent -> (1.6/0.1+1) x (1.0/0.1+1) = 17 x 11 = 187 rays
    height_scan_offset = (0.375, 0.0, 20.0)  # base-frame sensor offset (x fwd; z high so rays cast down)
    height_scan_clip = 1.0  # relative height clipped to [-clip, +clip] (parkour uses 1.0)
    # ray count from the GridPattern arange (x in [-0.8, 0.8] step 0.1 -> 17, y in [-0.5, 0.5] -> 11).
    num_height_rays = (int(round(height_scan_size[0] / height_scan_resolution)) + 1) * (
        int(round(height_scan_size[1] / height_scan_resolution)) + 1
    )
    num_heights = num_height_rays if use_heightmap else 0  # obs dims added when the scanner is on

    # policy obs width: projected_gravity(3) + commands(3) + [joint_pos-def, joint_vel, actions] (3*action_space)
    num_prio_obs = 3 + 3 + action_space * 3
    if timing_parameter:
        num_prio_obs += 1
    if clock_inputs:
        num_prio_obs += 4  # trot clock: (sin,cos) for each of the 2 diagonal phase groups
    if use_heightmap:
        num_prio_obs += num_heights  # RayCaster height-scan block (187 rays), appended after the clock
    if foothold_obs:
        # 2 targets x 4 feet x 3 (body-frame xyz) = 24, plus 4 per-foot target ages = 28
        num_prio_obs += num_foothold_targets * 4 * 3 + 4
        if foothold_ik_obs:
            num_prio_obs += 4 * foothold_ik_joints_per_leg  # +12 IK desired-joint block

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

    # robot — LEG_DTC_CFG with a bent-knee crouch default pose.
    # LEG_DTC_CFG = the company's LEG_CFG (same USD/joint order as the AMP dataset) with the foot
    # torque and per-joint velocity limits restored to measured values (see rga.py).
    # ⚠ ``pos.z=0.57`` and CROUCH_JOINT_POS were tuned against the old JSH/quad_17dof USD, whose Base
    # carries a baked ~0.5235 transform. LEG_CFG spawns at z=0.50 with no bake, so this height MUST be
    # re-measured once Leg_gen is available — see the spawn-height reasoning below.
    # Spawn height MUST be set explicitly: Isaac Lab derives ``default_root_state`` from
    # ``init_state.pos`` (NOT the USD's baked ~0.5235 Base transform), and ``_reset_idx`` writes that
    # world pose every episode — overriding the baked transform. With pos.z=0 the base was planted on
    # the ground (base contact force ~272 N > 1 N death threshold) and the crouched feet spawned ~0.55 m
    # underground, so every episode died on step 1 (epLen pinned at 1.0). z=0.57 lifts the base clear of
    # the ground and lands the crouched feet at ground level (feet hang ~0.55 m below the base).
    robot: ArticulationCfg = LEG_DTC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=LEG_DTC_CFG.init_state.replace(joint_pos=CROUCH_JOINT_POS, pos=(0.0, 0.0, 0.57)),
    )
    # ⚠ USD-STRUCTURE DEPENDENCY — verify against Leg_gen before the first run.
    # NOTE: the quad USD nests all rigid bodies under an extra "Base" Xform (articulation root),
    # so links are at /Robot/Base/<link> (one level deeper than hind_leg's /Robot/<link>).
    # find_matching_prims matches a single path segment per ".*", so the pattern must reach that depth.
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Base/.*", history_length=3, update_period=0.005, track_air_time=True
    )

    # -- Heightmap RayCaster (terrain perception) — instantiated ONLY when use_heightmap=True (see env) --
    # ★CRITICAL prim_path: attach to the rigid BODY /Robot/Base/Base, NOT the grouping Xform /Robot/Base.
    # The quad USD nests every link under an extra "Base" Xform, so the base RIGID body is /Robot/Base/Base
    # (same depth the contact sensor's /Robot/Base/.* reaches). Attaching a RayCaster to the /Robot/Base
    # Xform (no rigid body) forces a per-step CPU physics readback that measured 41 s/iter vs 2.4 s/iter on
    # the GPU physics-view rigid-body path. GridPattern 0.1 m over [1.6, 1.0] m => 17 x 11 = 187 vertical
    # rays; ray_alignment="yaw" so the map rotates with heading only. mesh = /World/ground (plane + any
    # gap strips imported under it). The field is harmless when use_heightmap is off (never instantiated).
    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Base/Base",
        offset=RayCasterCfg.OffsetCfg(pos=height_scan_offset),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=height_scan_resolution, size=height_scan_size),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
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

    # -- Always-on numerical robustness (pure safety; NO behaviour change on finite, in-range data) --
    # A physics blow-up on one env (e.g. a wide gap at a high terrain level) can emit NaN/Inf/huge values
    # that propagate obs -> reward -> gradient -> NaN policy std (crash: "normal expects all elements of
    # std >= 0.0"). These guards sanitize obs (_get_observations) and reward (_get_rewards) and terminate a
    # detonated robot early (_get_dones). They do NOT change obs/action dims, so checkpoints stay loadable.
    obs_clip = 100.0  # obs sanitized to nan_to_num then clamped to +/-obs_clip (generous; real obs << 100)
    reward_clip = 50.0  # per-step summed reward clamped to +/-reward_clip after nan_to_num
    # blow-up termination thresholds (well above real locomotion so gap-level fast motion is NOT killed):
    blowup_lin_vel = 25.0  # m/s; base speed above this = explosion (real top speed ~2 m/s)
    blowup_ang_vel = 50.0  # rad/s; base angular speed above this = explosion
    blowup_height_max = 5.0  # m; base world-z above this = launched (normal ~0.82 m on the strip surface)
    blowup_height_min = -5.0  # m; base world-z below this = fell through the world / exploded downward

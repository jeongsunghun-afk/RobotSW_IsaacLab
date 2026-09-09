# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import FlatPatchSamplingCfg, TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

##
# Pre-defined configs
##
from isaaclab_assets.robots.rga import LEG_DTC_CFG  # isort: skip


# ------------------------------------------------------------------------------------------------
# Bent-knee crouch default pose (17 DOF) — copied from direct/quad17 (validated spawn pose).
# Front vs rear thigh/calf axis signs are OPPOSITE (see MJCF), so a symmetric crouch uses opposite
# signs front vs rear. Hips and waist stay at 0. This is the PD-target neutral + the reference used
# by the dof_error_l2 / hip_pos / similar-pose reward terms (so the waist is held near 0 emergently).
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

from .parkour_terrains import (
    MeshParkourBalanceBeamTerrainCfg,
    MeshParkourCrawlTerrainCfg,
    MeshParkourRoughBlocksTerrainCfg,
    MeshParkourSlopeTerrainCfg,
    MeshParkourSteppingStonesTerrainCfg,
    MeshParkourZigzagHurdlesTerrainCfg,
)
from .parkour_terrains import parkour_jump_hurdle_terrain as _parkour_jump_hurdle_terrain

##
# Parkour terrain configuration
##

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(25.0, 4.0),
    border_width=20.0,
    num_rows=11,
    num_cols=40,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # row 0 = easiest (difficulty≈0), row 9 = hardest (difficulty≈1)
    sub_terrains={
        # Proportions match Genesis reference (train_parkour.py terrain_dict, active config):
        # parkour_flat=0.1, parkour_hurdle=0.2, parkour_step=0.2, parkour_gap=0.3, parkour_stair=0.2 → sum=1.0
        # Note: Genesis "parkour" (0.0) has no IsaacLab equivalent — omitted.
        # Terrain class indices follow dict insertion order (TERRAIN_CLASS_* constants below).
        # num_goals=8: explicit (matches default added by W1 to MeshParkour*TerrainCfg).
        # cfg.terrain_goals is populated by the terrain function at build time (used by W3 env wiring).
        "parkour_flat": terrain_gen.MeshParkourHurdleTerrainCfg(
            function=_parkour_jump_hurdle_terrain,
            proportion=0.2,
            flat=True,
            num_hurdles=8,
            num_goals=8,
            hurdle_height_range=(0.0, 0.0),
            x_spacing_range=(1.0, 2.4),  # explicit: 2.5 + 8*(1.5+0.3)=16.9m ≤ 20m
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_hurdle": terrain_gen.MeshParkourHurdleTerrainCfg(
            function=_parkour_jump_hurdle_terrain,
            proportion=0.2,
            num_hurdles=8,
            num_goals=8,
            hurdle_height_range=(0.05, 0.30),
            x_spacing_range=(1.8, 2.4),
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_step": terrain_gen.MeshParkourStepTerrainCfg(
            proportion=0.2,
            num_steps=8,
            num_goals=8,
            # x_length_range=(0.4, 0.8),
            x_length_range=(1.2, 2.0),
            half_valid_width_range=(0.8, 1.0),
            # y_offset_range=(1.0, 1.0),
            step_height_range=(0.10, 0.6),
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_gap": terrain_gen.MeshParkourGapTerrainCfg(
            proportion=0.2,
            num_gaps=8,
            num_goals=8,
            # gap_length_range=(0.05, 0.6),  # reduced: overflow fix (was 0.3, 0.8)
            gap_length_range=(0.05, 0.8),
            platform_length_range=(1.2, 2.0),  # explicit: 2.0+8*1.6+8*0.4=18.0m ≤ 20m
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_stair": terrain_gen.MeshParkourStairTerrainCfg(
            proportion=0.2,
            stair_width_range=(0.25, 0.40),
            stair_height_range=(0.05, 0.25),
            num_goals=8,
            num_steps_per_stair=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        # -----------------------------------------------------------------------
        # New terrain types — proportion=0.0 (inactive, registered for future use).
        # Existing 5-terrain proportions (sum=1.0) are preserved unchanged.
        # To activate: set proportion>0 and ensure all proportions still sum to 1.0.
        # Class IDs follow dict insertion order: stones=5, beam=6, crawl=7,
        # slope=8, zigzag=9, rough=10  (see TERRAIN_CLASS_* constants below).
        # NOTE: With proportion=0.0 these terrains are never spawned; the _col_to_class
        # LUT (parkour_env.py) will correctly assign class IDs 5-10 only once any of
        # them has proportion > 0. The sum=1.0 assert still passes.
        # -----------------------------------------------------------------------
        "parkour_stepping_stones": MeshParkourSteppingStonesTerrainCfg(
            proportion=0.0,
            platform_length=2.5,
            num_stones=8,
            stone_size_xy_range=(0.20, 0.40),
            stone_height_range=(0.05, 0.15),
            gap_length_range=(0.20, 0.45),
            lateral_jitter_range=(0.0, 0.30),
            num_goals=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_balance_beam": MeshParkourBalanceBeamTerrainCfg(
            proportion=0.0,
            platform_length=2.5,
            platform_height=0.15,
            beam_width_range=(0.20, 0.50),
            beam_height=0.15,
            max_segments=3,
            y_shift_per_segment_range=(0.0, 0.40),
            num_goals=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_crawl": MeshParkourCrawlTerrainCfg(
            proportion=0.0,
            platform_length=2.5,
            num_crawls=3,
            ceiling_height_range=(0.28, 0.50),
            ceiling_length_x=1.2,
            ceiling_thickness=0.10,
            ceiling_top_extra=0.50,
            corridor_width=1.2,
            side_wall_height=1.0,
            side_walls=True,
            x_spacing_range=(1.0, 2.0),
            num_goals=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_slope": MeshParkourSlopeTerrainCfg(
            proportion=0.0,
            platform_length=2.5,
            slope_angle_deg_range=(5.0, 25.0),
            slope_length=4.0,
            flat_top_length=1.5,
            num_goals=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_zigzag_hurdles": MeshParkourZigzagHurdlesTerrainCfg(
            proportion=0.0,
            platform_length=2.5,
            num_hurdles=6,
            hurdle_thickness=0.30,
            hurdle_height_range=(0.10, 0.25),
            corridor_width=2.0,
            x_spacing_range=(1.5, 2.4),
            num_goals=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_rough_blocks": MeshParkourRoughBlocksTerrainCfg(
            proportion=0.0,
            platform_length=2.5,
            block_size=0.30,
            block_height_range=(0.0, 0.10),
            block_density_range=(0.6, 0.9),
            num_goals=8,
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
    },
)
"""Parkour terrain configuration with 5 parkour-specific obstacle types for curriculum learning."""

##
# Terrain class index constants (match sub_terrains dict insertion order above).
# Used by Task #4 reward worker to map env_class → terrain type for conditional reward multipliers.
##
TERRAIN_CLASS_FLAT = 0  # parkour_flat   — hurdle terrain with flat=True (no boxes)
TERRAIN_CLASS_HURDLE = 1  # parkour_hurdle — hurdle boxes the robot must jump over
TERRAIN_CLASS_STEP = 2  # parkour_step   — ascending/descending step pyramid
TERRAIN_CLASS_GAP = 3  # parkour_gap    — platforms separated by gaps
TERRAIN_CLASS_STAIR = 4  # parkour_stair  — ascending/descending stair cycles
# New terrain classes — registered inactive (proportion=0.0).
# Class IDs follow dict insertion order in PARKOUR_TERRAINS_CFG.sub_terrains above.
# To activate a terrain: set its proportion > 0 (ensure all proportions still sum to 1.0).
TERRAIN_CLASS_STEPPING_STONES = 5  # parkour_stepping_stones — discrete stones with lateral jitter
TERRAIN_CLASS_BALANCE_BEAM = 6  # parkour_balance_beam    — narrow raised beam, fall-off risk
TERRAIN_CLASS_CRAWL = 7  # parkour_crawl           — low-ceiling corridor (requires height_scan masking)
TERRAIN_CLASS_SLOPE = 8  # parkour_slope           — wedge-primitive incline
TERRAIN_CLASS_ZIGZAG_HURDLES = 9  # parkour_zigzag_hurdles  — left/right alternating hurdles
TERRAIN_CLASS_ROUGH_BLOCKS = 10  # parkour_rough_blocks    — random-height block grid


##
# Event (randomization) configuration
##


@configclass
class EventCfg:
    """Configuration for environment randomization events.

    Standard sim-to-real DR set (RMA-style):
      - body_physics_material: all-body static/dynamic friction
      - add_base_mass: base link mass perturbation
      - randomize_com: base CoM xyz offset
      - push_robot: periodic velocity impulse for robustness
      - randomize_actuator_gains: joint stiffness/damping ±2.5% scale (conservative)

    Note on randomize_actuator_gains: enabled at ±2.5% (uniform scale) to provide
    signal for joint_stiffness_ratio / joint_damping_ratio in priv_latent. Without
    this term those dims are constant zero. Range is deliberately conservative given
    actuator_mode=2 (stiffness=25, damping=0.5) — B extreme_parkour marks this as
    potentially destabilizing at wider ranges; monitor training stability.
    """

    # foot_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
    #         "static_friction_range": (0.4, 1.5),
    #         "dynamic_friction_range": (0.3, 1.2),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 64,
    #     },
    # )

    # NEW — base body friction (conservative range to preserve balance on narrow obstacles)
    # Reference: Go2 manager-based env (legged_robots/velocity/config/unitree_go2)
    body_physics_material = EventTerm(
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

    # NEW — base mass perturbation (battery/payload variation; affects jump energy + landing)
    # Reference: Anymal-C / Go2 standard (-1.0, +3.0 kg, additive)
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # NEW — base CoM offset (battery position drift / wear; affects balance + angular inertia)
    # Reference: Go2 direct env (parkour-friendly conservative ranges vs. stock ±15/±5/±5 cm)
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

    # NEW — push DR: periodic random velocity impulse on the robot base (sim-to-real robustness)
    # Reference: B parkour_mdp_cfg.py:321-327 (same interval / velocity ranges)
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 8.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
        },
    )

    # NEW — actuator gains DR: ±2.5% scale on joint stiffness and damping (startup, conservative).
    # Provides non-zero signal for joint_stiffness_ratio / joint_damping_ratio in priv_latent.
    # Without this term those 24 dims are constant zero (zero information for the priv_encoder).
    # Range (0.975, 1.025) is very conservative relative to typical ±20%; monitor training stability
    # given actuator_mode=2 (already-weakened stiffness=25 / damping=0.5 / saturation=23.5).
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.975, 1.025),
            "damping_distribution_params": (0.975, 1.025),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


##
# Environment configuration
##


@configclass
class Quad17ParkourEnvCfg(DirectRLEnvCfg):
    """Parkour locomotion for OUR 17-DOF quadruped (LEG_DTC_CFG).

    This is the Go2 parkour recipe (PARKOUR_TERRAINS_CFG curriculum + height_scanner perception +
    gait-free rewards) with ONLY the robot swapped to our 17-DOF machine and the dims/body-names
    adapted. See the module header of ``parkour_env.py`` for the full list of adaptations.
    """

    # viewer — absolute world coordinates; system tracking callback skips when origin_type="world"
    viewer: ViewerCfg = ViewerCfg(
        origin_type="world",
        asset_name="robot",
        env_index=0,
        eye=(0.0, -2.5, 0.8),
        lookat=(0.0, 0.0, 0.3),
    )

    # visualization toggles
    debug_vis: bool = False
    enable_keyboard_view_switch: bool = True
    debug_print_contacts: bool = (
        False  # if True, print active contact bodies + forces to console every step (P key toggles at runtime)
    )
    debug_vis_edge_mask: bool = False  # if True, visualize edge mask cells around env 0 as green spheres (default OFF for training perf; play.py forces True)
    debug_vis_edge_mask_radius_m: float = 5.0  # radius (m) around env 0 base position to visualize

    # Clearance 3D scanner (teacher privileged GT).
    # Set True only for validation or teacher-training runs. Default OFF to avoid
    # overhead on all parkour variants (symmetry, imitation, random-goal, lidar).
    enable_clearance_scanner: bool = False

    # Voxel occupancy GT (teacher privileged, ablation counterpart to clearance_vec).
    # Reuses clearance scanner ray hits — enable_clearance_scanner is auto-forced True
    # when this flag is True (guarded in parkour_env._setup_scene).
    # Default OFF; enable alongside enable_clearance_scanner for CrawlTest or teacher runs.
    enable_voxel_scanner: bool = False

    # Teacher privileged 3D scan mode (R2).
    # When True: the "scan" obs group returns self._clearance_vec (294-dim, normalized to [-1,1])
    # instead of the 2D height-scan (187-dim).  Requires enable_clearance_scanner=True.
    # Default False → all base parkour variants are unaffected (scan=187 unchanged).
    # Normalization: (clearance_m - 2.0) / 2.0  maps [0.2, 4.0] → [-0.9, 1.0].
    clearance_as_scan: bool = False

    # env
    episode_length_s: float = 20.0
    decimation: int = 4
    action_scale: float = 0.25
    action_space: int = 17  # 16 leg joints (4 legs x hip/thigh/calf/foot) + 1 waist joint
    clip_actions: float = 10.0

    # Observation space (dict-based; the ActorCriticRMA runner infers every group dim from the
    # returned obs TensorDict at runtime, so only "policy"/proprio must match observation_space):
    # policy (proprio): yaw_diff(1)+next_yaw_diff(1)+proj_grav(3)+cmd_x(1)+jpos-def(17)+jvel(17)
    #                   +actions(17)+contact_filt(4) = 61
    # scan:          height_scan (GridPattern 1.6x1.0 @0.1) = 187  (robot-independent)
    # priv_explicit: root_lin_vel_b(3) + root_ang_vel_b(3) = 6
    # priv_latent:   base_friction(1)+foot_friction(4)+base_mass(1)+base_com(3)
    #                +joint_stiffness_ratio(17)+joint_damping_ratio(17) = 43  (17 joints)
    # history:       history_len * num_proprio = 10 * 61 = 610
    observation_space: int = 61  # policy obs dim (runner overrides with dict obs_groups)
    state_space: int = 0

    # Observation dimensions
    num_proprio: int = 61  # 1+1+3+1+17+17+17+4
    num_scan_obs: int = 187
    num_priv_obs: int = 49  # priv_explicit(6) + priv_latent(43) — informational; dims inferred at runtime
    history_len: int = 10

    # simulation
    dt: float = 1 / 200
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            # friction_combine_mode="average",
            # restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**23,
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # terrain
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            # friction_combine_mode="average",
            # restitution_combine_mode="average",
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

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    # events
    events: EventCfg = EventCfg()

    # robot — OUR 17-DOF quadruped with a bent-knee crouch spawn pose.
    # Spawn height MUST be set explicitly (Isaac Lab derives default_root_state from init_state.pos,
    # NOT the USD's baked ~0.5235 Base transform, and _reset_idx writes that world pose every reset).
    # z=0.57 lifts the base clear and lands the crouched feet at ground level (validated in direct/quad17).
    # Actuators (PD gains, effort/velocity limits) come straight from LEG_DTC_CFG — no Go2 DCMotor
    # override: the parkour action is joint-position targets applied through these ImplicitActuator PD gains.
    # SPEED (task-local override — the shared LEG_DTC_CFG is NOT modified): disable self-collision
    # for THIS task only. Self-collision across 17 links x 4096 envs on the parkour mesh explodes the
    # collision-pair / solver cost (~18x slower than Go2, whose parkour robot also runs self-collision
    # OFF, unitree.py:71). Nested .replace() builds FRESH cfg objects (dataclasses.replace is shallow),
    # so LEG_DTC_CFG.spawn.articulation_props is never mutated; the solver iteration counts
    # (position=8, velocity=4) are preserved by only overriding enabled_self_collisions.
    robot: ArticulationCfg = LEG_DTC_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=LEG_DTC_CFG.init_state.replace(joint_pos=CROUCH_JOINT_POS, pos=(0.0, 0.0, 0.57)),
        spawn=LEG_DTC_CFG.spawn.replace(
            articulation_props=LEG_DTC_CFG.spawn.articulation_props.replace(
                enabled_self_collisions=False,
            ),
        ),
    )
    # NOTE (perf investigation): self-collision OFF and solver iters 8/4->4/0 were each probed at
    # 4096 envs and changed the 41 s/iter NOT AT ALL — so neither was the bottleneck. Solver iters are
    # therefore left at the QUAD_17DOF default (8/4) to preserve physics fidelity; self-collision is
    # kept OFF (matches the Go2 parkour robot, harmless). The real cost was the height_scanner below
    # attaching to the /Robot/Base *Xform* (RayCaster "not a physics prim -> XFormPrim" CPU-readback
    # fallback), fixed by pointing it at the real base RIGID BODY /Robot/Base/Base.

    # sensors
    # The quad USD nests all rigid bodies under an extra "Base" Xform (articulation root), so links
    # live at /Robot/Base/<link> — one level deeper than Go2's /Robot/<link>. The base RIGID BODY is
    # at /Robot/Base/Base (the /Robot/Base level is only a grouping Xform). The contact-sensor pattern
    # matches that depth; the height-scanner MUST attach to the rigid body /Robot/Base/Base — attaching
    # to the /Robot/Base Xform makes the RayCaster fall back to a per-step XFormPrim CPU pose readback,
    # which at 4096 envs was the ENTIRE ~18x slowdown (41 s/iter -> low single digits once fixed).
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/Base/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/Base/Base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.375, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # command configuration (Genesis original: forward-only parkour)
    command_cfg: dict = {
        "lin_vel_x_range": [0.3, 1.5],  # forward only (Genesis original)
        "lin_vel_y_range": [0.0, 0.0],  # no lateral movement
        "ang_vel_range": [0.0, 0.0],  # no yaw turning
    }

    # Time-based velocity command resampling interval (seconds).
    # Converted to policy steps in env __init__: 6.0s / 0.02s = 300 steps.
    # Policy step dt = decimation(4) / physics_rate(200 Hz) = 0.02 s.
    resampling_time_s: float = 4.0

    # reward scales (B-aligned: Isaaclab_Parkour / extreme_parkour 14-term set)
    reward_scales: dict = {
        "tracking_goal_vel": 1.5,  # was 1.2, Genesis original = 1.5; forward-velocity signal 강화
        "tracking_yaw": 0.5,  # was 0.7, Genesis original = 0.5; tracking_goal_vel 우선시
        "lin_vel_z_l2": -1.0,  # Genesis original
        "ang_vel_xy_l2": -0.05,  # Genesis original (= ang_vel_xy2)
        "orientation_l2": -1.0,  # Genesis original
        "dof_acc_l2": -2.5e-7,  # Genesis original (joint-space; magnitude ~= Go2 per-joint, kept)
        "collision": -10.0,  # Genesis original (= collision2)
        "action_rate_l2": -0.1,  # Genesis original (WAS -0.1, 10x error)
        # --- 17-DOF adaptation: our peak joint torques are ~4-5x Go2's (effort limits up to 126 Nm
        # vs 23.5 Nm), so torque² and Δtorque² are ~20x larger. Scaled DOWN so these penalties do
        # not dominate early learning / swamp the forward-progress signal. (Only the torque-family
        # weights are retuned; all other reward weights are the Go2 parkour recipe unchanged.)
        "delta_torques": -1.0e-8,  # 17-DOF: Go2 -1e-7 / ~10 (larger torques)
        "torques_l2": -1e-6,  # 17-DOF: Go2 -1e-5 / 10 (larger torques)
        "hip_pos": -0.5,  # Genesis original
        "dof_error_l2": -0.04,  # Genesis original
        "feet_stumble": -1.0,  # Genesis original (= feet_stumble2)
        "feet_edge": -1.0,  # Genesis original (= feet_edge2)
        "feet_dragging": -0.1,  # hind feet only (HL, HR); threshold = dragging_velocity_threshold
        "feet_gait_pairing": 0.0,  # Spot GaitReward sync-only: trot 대각 쌍 phase 동기 (양수 only)
        # air_time_cap: per-foot graded penalty for excessive continuous air time.
        # Rationale: 정상 발 p99=0.36s/max=1.70s(gap 도약), RL 병리 p90=1.24s/max=5.60s.
        # max_air=1.0s anchor → 정상 도약 0.008%만 걸림, RL 병리꼬리 13.3% 처벌.
        # weight=-0.1 선택 근거: typical-bad(0.5s 초과) → 0.001/step(ceiling의 3%, clip-safe),
        # extreme-tail(4.6s 초과) → 0.0092/step(ceiling의 31%, clip-safe). 효과 부족 시
        # weight→-0.25 또는 air_time_cap_max_s→0.6s로 조임.
        "air_time_cap": -0.0,  # graded per-foot penalty (value≥0, weight<0 → contribution≤0)
        # contact_duty_deficit: escape-불가 EMA 기반 per-foot 접촉비율 부족분 penalty.
        # weight 단위: Episode_Reward 측정 단위(= Σ scaled / episode_length_s, step_dt 이미 반영).
        # 이 단위에서 deficit value≈0.28(들린 발 1개, full episode) → 기여 = |weight| × 0.28.
        # 회피 유인 0.054를 상회하려면 |weight| > 0.19. -0.5: 기여 0.14 ≈ 2.6× 회피 유인(net escape gradient 유의미).
        # ⚠️ step_dt 재곱 금지: env line 1251 scale×step_dt×value의 step_dt는 측정값에 이미 포함됨.
        #    이전 -15는 step_dt 이중 곱 오류(50× 과대) → clip(min=0)(env:1259)으로 학습 gradient 소멸.
        "contact_duty_deficit": -0.0,  # 1순위 headline fix. weight는 측정 단위(Episode_Reward, step_dt 이미 반영)에서 회피유인 0.054 대비 설정 — step_dt 재곱 금지(이전 -15는 50x 과대 오류). 검증 run 후 -0.3~-1.0 범위 조정
        # positive_work: Fu et al. 2021 positive mechanical work efficiency penalty.
        # Formula: Σ_j max(0, τ_j · q̇_j) over 12 joints  [units: W, always >= 0].
        # Weight < 0 → penalty; default 0.0 (opt-in, causes no gradient until enabled).
        #
        # Calibration (pronk_cost_result.npz, 256 envs × 3000 steps, dt=0.02 s):
        #   arr_mech_pow cross-check: flat 194,672 J / stair 307,586 J matches raw_md to <1 J.
        #   Per-step power: flat mean=62 W p90=133 W | stair mean=101 W p90=235 W p99=857 W
        #   tracking_goal_vel typical per-step contribution ≈ 1.5 × 0.02 × 0.70 = 0.021
        #
        #   weight=-3e-4:  stair mean=0.0006 (2.9%), stair p90=0.0014 (6.7%) — conservative, safe
        #   weight=-1e-3:  stair mean=0.0020 (9.6%), stair p90=0.0047 (22%)  — moderate
        #                  BUT stair p99=0.017 (82%), stair max=0.028 (132%) → risks total_reward
        #                  flooring at extreme steps (clip min=0 at env.py:1261).
        #
        #   Recommended experiment weight: -3e-4  (conservative first run; step up to -1e-3 if
        #   efficiency pressure appears too weak after 10k+ steps of observation).
        "positive_work": -3e-4,  # 17-DOF: our mechanical power is larger; calibrated conservative value
    }

    # tracking reward parameters (Genesis original)
    tracking_sigma: float = 0.2  # exp(-error / sigma) for tracking rewards

    # === Yaw reward gating thresholds ===
    # Robot이 정지 상태일 때 tracking_yaw가 max 값을 줘서 "정지+정면" local optimum이 형성되는
    # 것을 방지. horizontal speed가 lower 이하면 reward=0, upper 이상이면 full, 사이는 smooth.
    yaw_reward_speed_lower: float = 0.05  # m/s — 이 속도 이하에선 yaw reward 0
    yaw_reward_speed_upper: float = 0.15  # m/s — 이 속도 이상에선 yaw reward full

    # feet dragging detection threshold (Genesis original)
    dragging_velocity_threshold: float = 0.05  # m/s — feet considered dragging if in contact + moving

    # air_time_cap penalty parameter
    # max_air=1.0s: 정상 발 max=1.70s(gap 도약)보다 낮게, RL 병리 p90=1.24s보다 낮게 잡아
    # 정상 도약 0.008%만 걸림, 병리꼬리 13.3% 처벌. 효과 부족 시 0.6s로 조임.
    air_time_cap_max_s: float = 1.0  # seconds; 초과분에 graded penalty

    # contact_duty_deficit: escape-불가 per-foot 접촉비율(EMA) 기반 penalty 파라미터
    # EMA 업데이트: duty ← α·contact + (1−α)·duty,  α = step_dt / contact_duty_tau
    # graded deficit = Σ_feet clamp(contact_duty_target − duty, min=0)
    contact_duty_tau: float = 1.0  # s, EMA 시상수 (≈2~3 gait cycle). alpha = step_dt/tau
    contact_duty_target: float = 0.5  # 각 발이 평균 30% 이상 접지 요구 (trot 지지발 ~0.5-0.7 / 들린 발 ~0 분리)
    contact_duty_force_thr: float = 2.0  # N, 접촉 판정 임계 (parkour_env.py line 1133 feet contact threshold와 일관)

    # Gait pairing reward parameters (Spot GaitReward style, sync-only)
    feet_gait_std: float = 0.2  # error tolerance (Spot=0.1; parkour 완화)
    feet_gait_max_err: float = 0.3  # max squared-error cap (Spot=0.2; parkour 완화)
    feet_gait_velocity_threshold: float = 0.3  # m/s; cmd_speed gate (parkour cmd range [0.3, 1.0] lower bound)

    # base height reward target (used only on flat terrain; scale=0.0 by default → disabled)
    # nominal 17-DOF stance height above terrain (m); currently unused (no base_height reward term).
    base_height_target: float = 0.52

    # parkour-specific
    num_goals: int = 8
    num_future_goal_obs: int = 2  # Task #2: lookahead goals
    goal_distance: float = 1.0  # Task #2: spacing between waypoints (m)
    next_goal_threshold: float = 0.2  # Task #2: distance to trigger goal reached
    reach_goal_delay: float = 0.1  # Task #2: hold time after reaching goal (s)
    goal_z: float = 0.3  # Task #2: placeholder z coordinate for goals (m)
    termination_height: float = -0.2
    termination_grace_steps: int = 5  # skip termination during first N policy steps after spawn (for physics settling)
    max_tilt: float = 1.309
    terrain_curriculum: bool = True

    # domain randomization (ranges) — Task #5 future work
    # friction_range: [0.6, 2.0]
    # added_mass_range: [0.0, 5.0]
    # motor_strength_range: [0.8, 1.2]

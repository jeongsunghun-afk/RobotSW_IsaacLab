# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
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
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip

from .parkour_terrains import parkour_jump_hurdle_terrain as _parkour_jump_hurdle_terrain
from .parkour_terrains import (
    MeshParkourSteppingStonesTerrainCfg,
    MeshParkourBalanceBeamTerrainCfg,
    MeshParkourCrawlTerrainCfg,
    MeshParkourSlopeTerrainCfg,
    MeshParkourZigzagHurdlesTerrainCfg,
    MeshParkourRoughBlocksTerrainCfg,
)

##
# Parkour terrain configuration
##

PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(20.0, 4.0),
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
            proportion=0.1,
            flat=True,
            num_hurdles=8,
            num_goals=8,
            hurdle_height_range=(0.0, 0.0),
            x_spacing_range=(1.0, 1.5),  # explicit: 2.5 + 8*(1.5+0.3)=16.9m ≤ 20m
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
            x_spacing_range=(1.0, 1.5),  # explicit: 2.5 + 8*(1.5+0.3)=16.9m ≤ 20m
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
            x_length_range=(0.4, 0.8),
            step_height_range=(0.10, 0.45),
            flat_patch_sampling={
                "init_positions": FlatPatchSamplingCfg(
                    num_patches=2,
                    patch_radius=0.5,
                    max_height_diff=0.05,
                )
            },
        ),
        "parkour_gap": terrain_gen.MeshParkourGapTerrainCfg(
            proportion=0.3,
            num_gaps=8,
            num_goals=8,
            gap_length_range=(0.05, 0.5),          # reduced: overflow fix (was 0.3, 0.8)
            platform_length_range=(1.2, 1.6),     # explicit: 2.0+8*1.6+8*0.4=18.0m ≤ 20m
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
            stair_height_range=(0.05, 0.20),
            num_goals=8,
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
TERRAIN_CLASS_FLAT = 0    # parkour_flat   — hurdle terrain with flat=True (no boxes)
TERRAIN_CLASS_HURDLE = 1  # parkour_hurdle — hurdle boxes the robot must jump over
TERRAIN_CLASS_STEP = 2    # parkour_step   — ascending/descending step pyramid
TERRAIN_CLASS_GAP = 3     # parkour_gap    — platforms separated by gaps
TERRAIN_CLASS_STAIR = 4   # parkour_stair  — ascending/descending stair cycles
# New terrain classes — registered inactive (proportion=0.0).
# Class IDs follow dict insertion order in PARKOUR_TERRAINS_CFG.sub_terrains above.
# To activate a terrain: set its proportion > 0 (ensure all proportions still sum to 1.0).
TERRAIN_CLASS_STEPPING_STONES = 5   # parkour_stepping_stones — discrete stones with lateral jitter
TERRAIN_CLASS_BALANCE_BEAM    = 6   # parkour_balance_beam    — narrow raised beam, fall-off risk
TERRAIN_CLASS_CRAWL           = 7   # parkour_crawl           — low-ceiling corridor (requires height_scan masking)
TERRAIN_CLASS_SLOPE           = 8   # parkour_slope           — wedge-primitive incline
TERRAIN_CLASS_ZIGZAG_HURDLES  = 9   # parkour_zigzag_hurdles  — left/right alternating hurdles
TERRAIN_CLASS_ROUGH_BLOCKS    = 10  # parkour_rough_blocks    — random-height block grid


##
# Event (randomization) configuration
##


@configclass
class EventCfg:
    """Configuration for environment randomization events."""

    foot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*foot"),
            "static_friction_range": (0.4, 1.5),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


##
# Environment configuration
##


@configclass
class ParkourEnvCfg(DirectRLEnvCfg):
    """Configuration for the Go2 parkour locomotion environment."""

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
    debug_print_contacts: bool = False  # if True, print active contact bodies + forces to console every step (P key toggles at runtime)
    debug_vis_edge_mask: bool = True  # if True, visualize edge mask cells around env 0 as green spheres
    debug_vis_edge_mask_radius_m: float = 5.0  # radius (m) around env 0 base position to visualize

    # env
    episode_length_s: float = 20.0
    decimation: int = 4
    action_scale: float = 0.25
    action_space: int = 12
    clip_actions: float = 10.0

    # Observation space (Task #3, #6 — dict-based, kept for backward compat):
    # Deprecated: observation_space is now a dict with keys {policy, critic, scan, history, priv}
    # policy: 3+3+2+2+12+12+12 = 46
    # scan: 187
    # priv: 4 (placeholder for domain randomization)
    # history: history_len * num_proprio = 10 * 46 = 460
    # critic = policy + scan + priv + history = 46 + 187 + 4 + 460 = 697
    # Note: DirectRLEnv may still use observation_space for Space creation, but runners use dict obs.
    observation_space: int = 42  # policy obs dim (runner overrides with dict obs_groups)
    state_space: int = 0

    # Observation dimensions (Task #3)
    num_proprio: int = 42  # 3+1+1+1+12+12+12
    num_scan_obs: int = 187
    num_priv_obs: int = 14  # lin_vel_b(3) + ang_vel_b(3) + foot_friction_4feet×2(8); mass excluded until randomized
    history_len: int = 10

    # simulation
    dt: float = 1 / 200
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=4,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=2**20,  # 1048576, 에러 요구치 536321의 약 2배 안전마진
            gpu_found_lost_pairs_capacity=2**22,
            gpu_total_aggregate_pairs_capacity=2**22,
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
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
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

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # sensors
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["/World/ground"],
    )

    # command configuration (Genesis original: forward-only parkour)
    command_cfg: dict = {
        "lin_vel_x_range": [0.3, 1.0],   # forward only (Genesis original)
        "lin_vel_y_range": [0.0, 0.0],   # no lateral movement
        "ang_vel_range": [0.0, 0.0],     # no yaw turning
    }

    # reward scales (Genesis original: train_parkour.py line 184-214)
    reward_scales: dict = {
        "tracking_goal_vel": 1.2,        # Genesis original
        "tracking_yaw": 0.7,             # Genesis original
        "tracking_lin_vel_xy_exp": 0.0,  # parkour-specific weakening
        "tracking_ang_vel_z_exp": 0.0,   # parkour-specific weakening
        "lin_vel_z_l2": -1.0,            # Genesis original
        "ang_vel_xy_l2": -0.05,          # Genesis original (= ang_vel_xy2)
        "orientation_l2": -1.0,          # Genesis original
        "dof_acc_l2": -1.0e-7,           # Genesis original
        "collision": -10.0,              # Genesis original (= collision2)
        "action_rate_l2": -0.1,         # Genesis original (WAS -0.1, 10x error)
        "delta_torques": -1.0e-7,        # Genesis original (NEW)
        "torques_l2": -1e-5,             # Genesis original
        "hip_pos": -0.5,                 # Genesis original
        "dof_error_l2": -0.04,           # Genesis original
        "feet_stumble": -1.0,            # Genesis original (= feet_stumble2)
        "feet_edge": -1.0,               # Genesis original (= feet_edge2)
        "termination": -0.0,           # Genesis original (NEW, suicide penalty)
        "feet_dragging": -0.0,           # Genesis original (NEW)
        "action_smoothness_1": -0.01,    # Genesis original (NEW)
        "action_smoothness_2": -0.01,    # Genesis original (NEW)
        "base_height": 0.0,              # flat-only: penalize base height deviation (disabled by default)
    }

    # tracking reward parameters (Genesis original)
    tracking_sigma: float = 0.2          # exp(-error / sigma) for tracking rewards

    # feet dragging detection threshold (Genesis original)
    dragging_velocity_threshold: float = 0.05  # m/s — feet considered dragging if in contact + moving

    # base height reward target (used only on flat terrain; scale=0.0 by default → disabled)
    base_height_target: float = 0.34  # nominal Go2 stance height above terrain (m)

    # parkour-specific
    num_goals: int = 8
    num_future_goal_obs: int = 2  # Task #2: lookahead goals
    goal_distance: float = 1.0  # Task #2: spacing between waypoints (m)
    next_goal_threshold: float = 0.05  # Task #2: distance to trigger goal reached
    reach_goal_delay: float = 0.1  # Task #2: hold time after reaching goal (s)
    goal_z: float = 0.3  # Task #2: placeholder z coordinate for goals (m)
    goal_reach_threshold: float = 0.5  # (deprecated, use next_goal_threshold)
    termination_height: float = 0.1
    termination_grace_steps: int = 5  # skip termination during first N policy steps after spawn (for physics settling)
    max_tilt: float = 1.5
    terrain_curriculum: bool = True

    # domain randomization (ranges) — Task #5 future work
    # friction_range: [0.6, 2.0]
    # added_mass_range: [0.0, 5.0]
    # motor_strength_range: [0.8, 1.2]

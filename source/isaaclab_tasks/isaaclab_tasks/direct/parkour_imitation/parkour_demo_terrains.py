# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Playground scatter terrain for the Go2 Parkour Demo.

Provides a single wide flat field with ONE representative of each of the 5 training
obstacle types (flat/hurdle/step/gap/stair) placed at distinct (x, y) offsets and
orientations so the robot can roam 360° among them (AC-T2).

Design constraints:
  - All obstacle params stay INSIDE the training distribution (reuses PARKOUR_TERRAINS_CFG
    param ranges; no OOD variation — AC-T4).
  - BINDING CONTRACT (critic MAJOR-1): this function appends EXACTLY
    num_rows * num_cols entries to PARKOUR_GOALS_REGISTRY during its generation pass.
    For the playground cfg (num_rows=1, num_cols=1) this is exactly 1 entry.
    Mismatch → _build_terrain_goals_map (parkour_env.py:435) sets
    _terrain_goals_world=None → _init_env_goals prints "no" and returns without
    populating _env_goals → uninitialized buffer (silent bug). The num_goals stub
    waypoints are dummies — playground ignores them functionally (goal bypass holds
    _current_goal_idx=0).
  - Do NOT edit parkour/parkour_terrains.py (P1: existing terrain assets frozen).

Registry entry format (parkour_env.py:448):
    (goals_local: np.ndarray[num_goals, 3], origin_local: np.ndarray[3])
  Both in the pre-centering local frame (same frame used by all terrain functions).

Ordering (parkour_env.py:449-456):
  curriculum=False (playground) → row-major: entry k → row=k//num_cols, col=k%num_cols.
  With num_rows=1, num_cols=1 → k=0 → row=0, col=0. Only one entry needed.

Usage:
    Register ParkourPlaygroundScatterTerrainCfg as the sole sub-terrain in a
    TerrainGeneratorCfg with num_rows=1, num_cols=1, curriculum=False.

    The terrain function is called exactly once by TerrainGenerator and appends
    exactly 1 entry. cfg.num_goals in the demo env cfg must match the stub count
    (NUM_PLAYGROUND_GOALS constant below).
"""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.mesh_terrains import PARKOUR_GOALS_REGISTRY
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.utils import configclass

# Number of stub goal waypoints registered per playground tile.
# Must match cfg.num_goals in ParkourPlaygroundEnvCfg (cfg-worker Step 2b).
# Set to 8 to match the training sub-terrain default (PARKOUR_TERRAINS_CFG num_goals=8).
NUM_PLAYGROUND_GOALS: int = 8


def parkour_playground_scatter_terrain(
    difficulty: float,
    cfg: ParkourPlaygroundScatterTerrainCfg,
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a playground scatter terrain with 5 obstacle types on a single wide flat field.

    Places ONE representative of each of the 5 training obstacle classes (flat/hurdle/step/
    gap/stair) at distinct (x, y) offsets so the robot can roam among them (AC-T2).
    All obstacle params are at mid-range training distribution values (AC-T4).

    BINDING CONTRACT (MAJOR-1): appends EXACTLY 1 entry to PARKOUR_GOALS_REGISTRY.
    Entry contains NUM_PLAYGROUND_GOALS stub waypoints in pre-centering local frame.

    Args:
        difficulty: Terrain difficulty in [0, 1]. Ignored for playground (fixed mid-range params).
        cfg: ParkourPlaygroundScatterTerrainCfg instance.

    Returns:
        Tuple of (list of trimesh meshes, origin ndarray shape (3,)).
    """
    # suppress unused-arg warning — playground uses fixed params, not difficulty
    _ = difficulty

    tile_x, tile_y = cfg.size
    mid_y = tile_y / 2.0

    meshes_list: list[trimesh.Trimesh] = []

    # ---- Ground plane covers the full tile ----
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    # ---- Border walls (R3: contain the robot, prevent scan OOD) ----
    if cfg.border_width > 0.0:
        inner_size = (tile_x - 2 * cfg.border_width, tile_y - 2 * cfg.border_width)
        border_center = (tile_x / 2.0, tile_y / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # ---- Stub goal list (NUM_PLAYGROUND_GOALS waypoints) ----
    # Placed at even x intervals along the tile center — the robot never needs to
    # reach them in playground mode (goal bypass holds _current_goal_idx=0), but
    # _init_env_goals (parkour_env.py:592) requires _env_goals to be populated.
    goals_list: list[list[float]] = []
    goal_x_step = tile_x / (NUM_PLAYGROUND_GOALS + 1)
    for gi in range(NUM_PLAYGROUND_GOALS):
        goal_x = (gi + 1) * goal_x_step
        goals_list.append([goal_x, mid_y, 0.0])

    # ---- Obstacle cluster positions ----
    # Each obstacle cluster is centered at a distinct (obstacle_x_base, y_center).
    # x_base is fixed at cfg.platform_length + cfg.obstacle_x_spacing * cluster_idx.
    # y_center cycles through cfg.obstacle_y_centers (5 values spread across tile width).
    # Obstacle order: 0=flat(none), 1=hurdle, 2=step, 3=gap, 4=stair.
    # Class 0 (flat) has no mesh — the ground plane already provides it.

    x_base = cfg.platform_length + cfg.obstacle_x_spacing
    y_centers = list(cfg.obstacle_y_centers)

    # Ensure we have 5 y_centers (one per obstacle class); pad with mid_y if fewer.
    while len(y_centers) < 5:
        y_centers.append(mid_y)

    # ---- Class 1: Hurdle ----
    # Central raised box (Genesis-style: robot jumps over or goes around sides).
    # Params: mid-range from training hurdle_height_range=(0.05, 0.30).
    hurdle_x = x_base + 0.0 * cfg.obstacle_x_spacing  # first cluster offset
    hurdle_y_center = y_centers[1]
    hurdle_h = cfg.hurdle_height
    hurdle_half_w = cfg.hurdle_width / 2.0
    hurdle_lo_y = hurdle_y_center - hurdle_half_w
    hurdle_hi_y = hurdle_y_center + hurdle_half_w
    hurdle_w = hurdle_hi_y - hurdle_lo_y
    if hurdle_w > 1e-3 and hurdle_h > 1e-3:
        dim = (cfg.hurdle_thickness, hurdle_w, hurdle_h)
        pos = (
            hurdle_x + cfg.hurdle_thickness / 2.0,
            hurdle_lo_y + hurdle_w / 2.0,
            hurdle_h / 2.0,
        )
        meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    # ---- Class 2: Step (ascending staircase) ----
    # Multiple ascending steps. Params: mid-range from step_height_range=(0.10, 0.60).
    step_x = x_base + 1.0 * cfg.obstacle_x_spacing
    step_y_center = y_centers[2]
    step_half_w = 1.0  # 1 m half-width — stays within training half_valid_width_range=(0.8, 1.0)
    step_w = 2.0 * step_half_w
    step_lo_y = step_y_center - step_half_w
    step_h = cfg.step_height
    step_x_len = cfg.step_x_length
    cur_step_x = step_x
    cur_step_z = 0.0
    for _ in range(cfg.num_steps):
        cur_step_z += step_h
        dim = (step_x_len, step_w, cur_step_z)
        pos = (cur_step_x + step_x_len / 2.0, step_lo_y + step_half_w, cur_step_z / 2.0)
        meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
        cur_step_x += step_x_len

    # ---- Class 3: Gap ----
    # Two platforms separated by a gap. Params: mid-range gap_length from training (~0.3–0.6 m).
    gap_x = x_base + 2.0 * cfg.obstacle_x_spacing
    gap_y_center = y_centers[3]
    gap_half_w = 1.0  # platform half-width in y
    gap_w = 2.0 * gap_half_w
    gap_lo_y = gap_y_center - gap_half_w
    gap_platform_w = cfg.gap_platform_width  # platform depth in x
    gap_len = cfg.gap_length
    gap_platform_h = 0.05  # slight elevation to mark the platforms visually
    # Platform before gap
    dim_a = (gap_platform_w, gap_w, gap_platform_h)
    pos_a = (gap_x + gap_platform_w / 2.0, gap_lo_y + gap_half_w, gap_platform_h / 2.0)
    meshes_list.append(trimesh.creation.box(dim_a, trimesh.transformations.translation_matrix(pos_a)))
    # Platform after gap
    gap_after_x = gap_x + gap_platform_w + gap_len
    dim_b = (gap_platform_w, gap_w, gap_platform_h)
    pos_b = (gap_after_x + gap_platform_w / 2.0, gap_lo_y + gap_half_w, gap_platform_h / 2.0)
    meshes_list.append(trimesh.creation.box(dim_b, trimesh.transformations.translation_matrix(pos_b)))

    # ---- Class 4: Stair (ascending steps, tighter pitch than class 2) ----
    # Params: mid-range from training stair step height [0.05, 0.25], step depth ~0.3–0.4 m.
    stair_x = x_base + 3.0 * cfg.obstacle_x_spacing
    stair_y_center = y_centers[4]
    stair_half_w = 1.0
    stair_w = 2.0 * stair_half_w
    stair_lo_y = stair_y_center - stair_half_w
    stair_step_h = cfg.stair_step_height
    stair_step_d = cfg.stair_step_depth
    cur_stair_x = stair_x
    cur_stair_z = 0.0
    for _ in range(cfg.stair_num_steps):
        cur_stair_z += stair_step_h
        # Each stair step is a box of increasing height, placed flush with the prior step
        dim = (stair_step_d, stair_w, cur_stair_z)
        pos = (cur_stair_x + stair_step_d / 2.0, stair_lo_y + stair_half_w, cur_stair_z / 2.0)
        meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
        cur_stair_x += stair_step_d

    # ---- Origin: center of start platform at ground level ----
    # Convention matches all existing parkour terrain functions (parkour_terrains.py:165, :406).
    origin = np.array([cfg.platform_length / 2.0, mid_y, 0.0], dtype=np.float64)

    # ---- Build goals array (pad/truncate to NUM_PLAYGROUND_GOALS) ----
    _raw = np.array(goals_list, dtype=np.float32)
    if len(_raw) < NUM_PLAYGROUND_GOALS:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (NUM_PLAYGROUND_GOALS - len(_raw), 1))])
    goals = _raw[:NUM_PLAYGROUND_GOALS]  # shape (NUM_PLAYGROUND_GOALS, 3)

    # ---- BINDING CONTRACT (MAJOR-1): append EXACTLY 1 entry ----
    # playground cfg: num_rows=1, num_cols=1 → _build_terrain_goals_map expects
    # len(PARKOUR_GOALS_REGISTRY)==1 after this function runs.
    # curriculum=False, row-major: k=0 → row=0, col=0. Exactly 1 entry required.
    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))

    return meshes_list, origin


@configclass
class ParkourPlaygroundScatterTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the playground scatter terrain.

    One wide flat field with 5 obstacle types placed at fixed offsets/orientations.
    All params are drawn from PARKOUR_TERRAINS_CFG training distribution (AC-T4).

    Fields:
        size:               (x_length [m], y_width [m]) of the tile. Wide field so the
                            robot can roam between all 5 obstacle clusters (AC-T3).
        platform_length:    Flat spawn area length in front of the robot spawn point [m].
        border_width:       Width of the border walls [m] (R3: bound the field edge).
        border_height:      Height of the border walls [m].
        num_goals:          Number of stub goal waypoints to register (must == NUM_PLAYGROUND_GOALS).
        hurdle_height:      Hurdle obstacle height [m]. Training range: [0.05, 0.30].
        step_height:        Step obstacle height [m]. Training range: [0.10, 0.60].
        gap_length:         Gap obstacle length [m]. Training range: [0.20, 1.00] (approx).
        stair_height:       Stair step height [m]. Training range: [0.05, 0.25] (approx).
        stair_num_steps:    Number of stair steps.
        obstacle_x_spacing: X-spacing between obstacle cluster centers [m].
    """

    function = parkour_playground_scatter_terrain

    # Tile dimensions: wide and long to accommodate 5 obstacle clusters with free roam space.
    # Training tile size: (25.0, 4.0). Playground uses a square wide field.
    size: tuple[float, float] = (30.0, 20.0)

    # Flat spawn platform length at x=0
    platform_length: float = 1.5

    # Border walls to bound the field and prevent scan OOD (R3)
    border_width: float = 0.5
    border_height: float = 0.5

    # Stub goal count — MUST match NUM_PLAYGROUND_GOALS and cfg.num_goals in ParkourPlaygroundEnvCfg
    num_goals: int = NUM_PLAYGROUND_GOALS

    # ---- Obstacle parameters (in training distribution, AC-T4) ----
    # Hurdle (class 1): mid-range height from training [0.05, 0.30]
    hurdle_height: float = 0.15
    hurdle_thickness: float = 0.2
    hurdle_width: float = 1.2  # half_valid_width mid-range

    # Step (class 2): mid-range height from training [0.10, 0.60]
    step_height: float = 0.25
    step_x_length: float = 1.5
    num_steps: int = 3

    # Gap (class 3): mid-range gap from training — approx 0.3–0.6 m
    gap_length: float = 0.40
    gap_platform_width: float = 1.0

    # Stair (class 4): mid-range from training [0.05, 0.25]
    stair_step_height: float = 0.12
    stair_step_depth: float = 0.35
    stair_num_steps: int = 4

    # X spacing between obstacle cluster centers [m]
    obstacle_x_spacing: float = 5.0

    # Y center positions for each obstacle cluster (5 clusters on a wide tile)
    # Spread across the tile width (20 m) with 3.5 m between each
    obstacle_y_centers: tuple[float, ...] = (2.5, 6.0, 10.0, 14.0, 17.5)

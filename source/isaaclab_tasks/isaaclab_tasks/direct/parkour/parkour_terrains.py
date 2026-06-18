# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom parkour terrain functions for the Go2 parkour environment.

This module provides a Genesis-style hurdle terrain where a single raised box is placed
in the *center* of the corridor, forcing the robot to jump over it or pass through the
unobstructed side passages.

Contrast with the IsaacLab core ``parkour_hurdle_terrain`` (mesh_terrains.py:966), which
places two side walls with a walkable *corridor* in the middle — a robot can traverse
without jumping. Here, the single central hurdle makes jump/vault the primary strategy.

Note: side passages (y < hurdle_center_y - half_valid_w and y > hurdle_center_y + half_valid_w)
are fully open ground (~1.2–1.6 m wide each on a 4 m terrain). The forward-only command
(lin_vel_y_range=[0,0]) discourages lateral detours but does not hard-block them; this is
intentional per the Genesis design.

New terrain functions (proportion=0.0 — inactive until user activates):
  parkour_stepping_stones_terrain  — discrete stepping stones with lateral jitter
  parkour_balance_beam_terrain     — narrow raised beam, fall-off → termination
  parkour_crawl_terrain            — low-ceiling corridor, robot must crouch
  parkour_slope_terrain            — wedge-primitive incline (ascend + flat + descend)
  parkour_zigzag_hurdles_terrain   — left/right alternating hurdles
  parkour_rough_blocks_terrain     — random-height block grid
"""

from __future__ import annotations

import time

import numpy as np
import torch
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.trimesh.mesh_terrains import PARKOUR_GOALS_REGISTRY
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshParkourHurdleTerrainCfg, MeshParkourStairTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane
from isaaclab.utils import configclass
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


def parkour_jump_hurdle_terrain(
    difficulty: float, cfg: MeshParkourHurdleTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a Genesis-style parkour hurdle terrain with a single raised central hurdle.

    Unlike the IsaacLab core ``parkour_hurdle_terrain`` which creates two side walls with a
    walkable corridor, this function places a single raised box in the *center* of the path.
    The robot must jump over the hurdle or navigate through the unobstructed side passages.

    The terrain layout per hurdle:
    - Left passage:  y in [0, hurdle_center_y - half_valid_w]   — open ground
    - Central hurdle: y in [hurdle_center_y - half_valid_w, hurdle_center_y + half_valid_w]
                      raised box of height ``hurdle_h``
    - Right passage: y in [hurdle_center_y + half_valid_w, cfg.size[1]] — open ground

    The goal waypoint is placed just past the hurdle face at ground level (z=0), matching
    the IsaacLab goal convention (consumed by PARKOUR_GOALS_REGISTRY).

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain. Uses all fields from
            :class:`~isaaclab.terrains.trimesh.mesh_terrains_cfg.MeshParkourHurdleTerrainCfg`.
            ``half_valid_width_range`` is reinterpreted as "half-width of the raised hurdle"
            (Genesis semantics) rather than "half-width of the passage corridor" (IsaacLab
            core semantics).

    Returns:
        A tuple containing the tri-mesh list of the terrain and the origin of the terrain (in m).
    """
    # ------------------------------------------------------------------
    # Resolve difficulty-dependent parameters (same interpolation as core)
    # ------------------------------------------------------------------
    hurdle_h = cfg.hurdle_height_range[0] + difficulty * (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0])
    # half_valid_w: in Genesis semantics this is the *half-width of the raised hurdle*,
    # not the half-width of the walkable corridor.
    half_valid_w = float(np.random.uniform(cfg.half_valid_width_range[0], cfg.half_valid_width_range[1]))

    mid_y = cfg.size[1] / 2.0

    meshes_list: list[trimesh.Trimesh] = []

    # Ground plane covers the full terrain (always present)
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []

    # ------------------------------------------------------------------
    # Pass 1: sample all hurdle positions up-front so that goal_x[i] can
    # be placed at the midpoint between hurdle[i]'s back face and
    # hurdle[i+1]'s front face (rather than 0.3 m past the current hurdle).
    # ------------------------------------------------------------------
    hurdle_spacings: list[float] = []  # rand_x[i] — spacing *before* hurdle i
    hurdle_front_xs: list[float] = []  # x of hurdle i front face (= dis_x after += rand_x)
    hurdle_center_ys: list[float] = []

    dis_x = cfg.platform_length
    for _ in range(cfg.num_hurdles):
        rand_x = float(np.random.uniform(cfg.x_spacing_range[0], cfg.x_spacing_range[1]))
        rand_y = float(np.random.uniform(cfg.y_offset_range[0], cfg.y_offset_range[1]))
        dis_x += rand_x
        hurdle_spacings.append(rand_x)
        hurdle_front_xs.append(dis_x)
        hurdle_center_ys.append(mid_y + rand_y)

    # Pass 2: build goals — midpoint between current hurdle back face and next hurdle front face.
    # For the last hurdle there is no next hurdle; reuse the last spacing as an estimate.
    for i in range(cfg.num_hurdles):
        hurdle_back_x = hurdle_front_xs[i] + cfg.hurdle_thickness
        if i + 1 < cfg.num_hurdles:
            next_hurdle_front_x = hurdle_front_xs[i + 1]
        else:
            # Extrapolate: assume the gap after the last hurdle equals the last sampled spacing.
            next_hurdle_front_x = hurdle_back_x + hurdle_spacings[i]
        goal_x = (hurdle_back_x + next_hurdle_front_x) / 2.0  # midpoint between hurdles
        goals_list.append([goal_x, hurdle_center_ys[i], 0.0])

    # Pass 3: build meshes using the pre-sampled positions.
    for i in range(cfg.num_hurdles):
        dis_x = hurdle_front_xs[i]
        hurdle_center_y = hurdle_center_ys[i]

        if cfg.flat or hurdle_h <= 0.0:
            # flat=True: skip obstacle creation but keep goal emission and registry append
            continue

        # ------------------------------------------------------------------
        # Central raised hurdle box (Genesis-style)
        # Width = 2 * half_valid_w (the hurdle spans [center-hw, center+hw] in y)
        # Left and right passages are open ground — no mesh added there.
        # ------------------------------------------------------------------
        hurdle_lo_y = hurdle_center_y - half_valid_w
        hurdle_hi_y = hurdle_center_y + half_valid_w
        hurdle_w = hurdle_hi_y - hurdle_lo_y  # = 2 * half_valid_w

        if hurdle_w > 1e-3 and hurdle_h > 1e-3:
            dim = (cfg.hurdle_thickness, hurdle_w, hurdle_h)
            pos = (
                dis_x + cfg.hurdle_thickness / 2.0,
                hurdle_lo_y + hurdle_w / 2.0,  # = hurdle_center_y
                hurdle_h / 2.0,
            )
            meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    # ------------------------------------------------------------------
    # Border walls (optional — cfg.border_width defaults to 0.0)
    # ------------------------------------------------------------------
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # ------------------------------------------------------------------
    # Origin: center of start platform at ground level (same as core)
    # ------------------------------------------------------------------
    origin = np.array([cfg.platform_length / 2.0, cfg.size[1] / 2.0, 0.0])

    # ------------------------------------------------------------------
    # Build goals array: pad with last goal if fewer than num_goals, truncate to num_goals
    # (identical logic to core parkour_hurdle_terrain)
    # ------------------------------------------------------------------
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    # Publish to module-level registry (consumed by parkour_env after terrain build)
    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))

    return meshes_list, origin


def compute_edge_mask_from_terrain_mesh(
    terrain_mesh: trimesh.Trimesh,
    terrain_origin: np.ndarray,
    terrain_size_x: float,
    terrain_size_y: float,
    horizontal_scale: float = 0.05,
    edge_threshold: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a height field from the terrain mesh via raycast and detect edges via Sobel x/y OR.

    Uses trimesh raycast (downward rays from above each grid cell) rather than cKDTree
    nearest-neighbour, which was the root cause of false-positive edge detection on flat
    terrain: sparse obstacle vertices near flat grid cells caused the KDTree to return
    an obstacle vertex z-value, polluting the height field and triggering spurious Laplacian
    responses.

    Raycast fires one -Z ray per grid cell from well above the mesh surface and takes the
    first intersection, which is always the correct surface height for that (x, y) cell.
    Rays that miss (open gaps in the mesh) default to z=0.

    Edge detection applies Sobel filters along x and y axes separately, then combines them
    with OR so edges along both axes are captured without cancellation.  This replaces the
    previous Laplacian (∂²f/∂x² + ∂²f/∂y²) approach which could cancel opposite-sign edges.

    ``terrain_origin`` is the **lower-left corner** of the full terrain in world frame
    (i.e. the minimum-x, minimum-y corner), not the centre.  This is the ``transform[:2, -1]``
    applied by TerrainGenerator after centering:
        origin = [-size_x * num_rows / 2, -size_y * num_cols / 2, 0]

    Grid cell (i, j) represents world position:
        x = terrain_origin[0] + (i + 0.5) * horizontal_scale
        y = terrain_origin[1] + (j + 0.5) * horizontal_scale

    The reverse lookup (world -> cell index) in the env is:
        idx = round((world - terrain_origin) * inv_scale - 0.5)
    which is equivalent to floor-dividing into cell bins, matching Genesis semantics.

    Performance note: trimesh.ray (numpy-only backend, no pyembree) processes roughly
    500K–1M rays/s.  For a 14M-cell grid (4400×3200 at 0.05 m) this takes ~15–30 s at
    build time.  The function batches rays in chunks of 500K and prints progress so the
    console does not appear frozen.

    Args:
        terrain_mesh: The full concatenated trimesh of the terrain (already centered).
        terrain_origin: Lower-left corner of the terrain bounding box, shape (3,) or (2,).
                        Typically ``[-terrain_size_x/2, -terrain_size_y/2, 0]``.
        terrain_size_x: Total terrain extent along x (num_rows * tile_x), in metres.
        terrain_size_y: Total terrain extent along y (num_cols * tile_y), in metres.
        horizontal_scale: Grid resolution in metres. Defaults to 0.05.
        edge_threshold: |Sobel| threshold for edge detection on the normalised 0–255 uint8
                        height field. Sobel is a first-derivative filter so magnitudes are
                        larger than Laplacian; 30.0 is a reasonable starting point (was 20.0
                        for Laplacian). Tune upward to suppress noise, downward to catch
                        subtle edges.

    Returns:
        A tuple ``(edge_mask, height_field)`` where:
        - ``edge_mask``: bool ndarray of shape (n_x, n_y), True indicates a terrain edge cell.
        - ``height_field``: float32 ndarray of shape (n_x, n_y), terrain height per grid cell (m).
        n_x = round(terrain_size_x / horizontal_scale)
        n_y = round(terrain_size_y / horizontal_scale)
    """
    import cv2  # local import: keep cv2 as a soft dependency

    n_x = round(terrain_size_x / horizontal_scale)
    n_y = round(terrain_size_y / horizontal_scale)

    origin_x = float(terrain_origin[0])
    origin_y = float(terrain_origin[1])

    # ---- Early exit: empty mesh ----
    if len(terrain_mesh.vertices) == 0:
        empty_hf = np.zeros((n_x, n_y), dtype=np.float32)
        return np.zeros((n_x, n_y), dtype=bool), empty_hf

    # ---- WARP GPU raycast (replaces numpy face rasterization) ----
    # Fires one downward ray per grid cell from well above the mesh and takes the first
    # intersection, which is always the correct surface height for that (x, y) cell.
    # This is exact for any geometry (axis-aligned boxes, slopes, steps) — the previous
    # face-rasterization approach was a conservative over-estimate for non-axis-aligned faces.
    # Missed rays (open gaps) return float('inf') and are substituted with z=0.

    t0 = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Build warp mesh from trimesh vertices/faces
    wp_mesh = convert_to_warp_mesh(
        np.asarray(terrain_mesh.vertices, dtype=np.float32),
        np.asarray(terrain_mesh.faces, dtype=np.int32),
        device=device,
    )

    # 2. Build grid ray starts — one ray per grid cell, centred in cell
    xs = torch.tensor(
        [origin_x + (i + 0.5) * horizontal_scale for i in range(n_x)],
        dtype=torch.float32,
    )
    ys = torch.tensor(
        [origin_y + (j + 0.5) * horizontal_scale for j in range(n_y)],
        dtype=torch.float32,
    )
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")  # each (n_x, n_y)

    z_high = float(terrain_mesh.bounds[1, 2]) + 10.0  # well above terrain top

    # raycast_mesh expects (N, 3) — no batch dimension
    ray_starts = torch.stack(
        [grid_x.flatten(), grid_y.flatten(), torch.full((n_x * n_y,), z_high)],
        dim=-1,
    ).to(device)  # (N, 3)

    ray_dirs = torch.zeros_like(ray_starts)
    ray_dirs[:, 2] = -1.0  # downward

    # 3. GPU raycast — returns (ray_hits, None, None, None) with default flags
    ray_hits, *_ = raycast_mesh(ray_starts, ray_dirs, wp_mesh)  # ray_hits: (N, 3)

    elapsed = time.time() - t0
    print(f"[edge_mask] WARP raycast: {n_x * n_y:,} rays in {elapsed:.2f}s on {device} (grid {n_x}×{n_y})")

    # 4. Extract z values; missed rays return inf — substitute with 0 (open gap = ground)
    hit_z = ray_hits[:, 2]  # (N,)
    hit_z = torch.where(torch.isfinite(hit_z), hit_z, torch.zeros_like(hit_z))

    height_field = hit_z.view(n_x, n_y).detach().cpu().numpy().astype(np.float32)

    # ---- Sobel x/y edge detection (OR-combined) ----
    hf_min = height_field.min()
    hf_max = height_field.max()
    if hf_max - hf_min < 1e-3:
        return np.zeros((n_x, n_y), dtype=bool), height_field  # flat terrain, no edges

    hf_norm = ((height_field - hf_min) / (hf_max - hf_min) * 255.0).astype(np.uint8)
    hf_blur = cv2.GaussianBlur(hf_norm, (5, 5), 0)

    # Detect edges along x and y axes separately, then OR to avoid cancellation
    sobel_x = cv2.Sobel(hf_blur, cv2.CV_64F, 1, 0, ksize=3)  # ∂f/∂x
    sobel_y = cv2.Sobel(hf_blur, cv2.CV_64F, 0, 1, ksize=3)  # ∂f/∂y

    edge_x = np.abs(sobel_x) > edge_threshold
    edge_y = np.abs(sobel_y) > edge_threshold
    edge_mask = (edge_x | edge_y).astype(bool)

    print(
        f"[edge_mask] sobel_x edges: {int(edge_x.sum())}, "
        f"sobel_y edges: {int(edge_y.sum())}, "
        f"combined: {int(edge_mask.sum())}"
    )

    return edge_mask, height_field


# =============================================================================
# New terrain functions — proportion=0.0 (inactive) until user activates.
# Terrain class IDs (dict-insertion order in parkour_env_cfg.py sub_terrains):
#   5 = stepping_stones, 6 = balance_beam, 7 = crawl,
#   8 = slope, 9 = zigzag_hurdles, 10 = rough_blocks
# IsaacLab difficulty standard: difficulty=0 → easiest, difficulty=1 → hardest.
# =============================================================================


def parkour_stepping_stones_terrain(
    difficulty: float, cfg: MeshParkourSteppingStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour stepping-stones terrain with discrete raised stones and lateral jitter.

    A flat ground plane is placed first. Then ``num_stones`` small box-shaped stones are arranged
    along the x-axis, each with a random y-jitter. The robot must hop from stone to stone without
    falling to ground level.

    Difficulty mapping (IsaacLab standard — higher = harder):
        stone_size      : easy=``stone_size_xy_range[1]``  → hard=``stone_size_xy_range[0]``
        gap between stones: easy=``gap_length_range[0]``   → hard=``gap_length_range[1]``
        lateral jitter  : easy=0.0                         → hard=``lateral_jitter_range[1]``

    Args:
        difficulty: Terrain difficulty in [0, 1]. 0 = easiest, 1 = hardest.
        cfg: Terrain configuration. See :class:`MeshParkourSteppingStonesTerrainCfg`.

    Returns:
        Tuple of (list of trimesh objects, origin ndarray of shape (3,)).
    """
    # -- resolve difficulty-dependent parameters (IsaacLab standard: larger difficulty = harder) --
    stone_size = cfg.stone_size_xy_range[1] - difficulty * (cfg.stone_size_xy_range[1] - cfg.stone_size_xy_range[0])
    gap_len = cfg.gap_length_range[0] + difficulty * (cfg.gap_length_range[1] - cfg.gap_length_range[0])
    max_jitter = difficulty * cfg.lateral_jitter_range[1]
    stone_h = cfg.stone_height_range[0] + difficulty * (cfg.stone_height_range[1] - cfg.stone_height_range[0])

    mid_y = cfg.size[1] / 2.0

    meshes_list: list[trimesh.Trimesh] = []

    # Ground plane covers the full terrain (robot falls here if it misses a stone)
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []

    # Start: spawn platform — implicit (ground plane is flat from x=0 to x=platform_length)
    dis_x = cfg.platform_length

    for _ in range(cfg.num_stones):
        # advance by the gap first, then place the stone
        dis_x += gap_len
        # lateral jitter, clamped so stone stays inside terrain y bounds with half_size margin
        raw_jitter = float(np.random.uniform(-max_jitter, max_jitter))
        stone_center_y = float(np.clip(mid_y + raw_jitter, stone_size / 2.0, cfg.size[1] - stone_size / 2.0))

        if stone_size > 1e-3 and stone_h > 1e-3:
            dim = (stone_size, stone_size, stone_h)
            pos = (dis_x + stone_size / 2.0, stone_center_y, stone_h / 2.0)
            meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

        # goal: top surface center of this stone
        goals_list.append([dis_x + stone_size / 2.0, stone_center_y, stone_h])
        dis_x += stone_size

    # border walls (optional)
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin: center of start platform at ground level
    origin = np.array([cfg.platform_length / 2.0, mid_y, 0.0])

    # build goals: pad or truncate to cfg.num_goals
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))
    return meshes_list, origin


def parkour_balance_beam_terrain(
    difficulty: float, cfg: MeshParkourBalanceBeamTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour balance-beam terrain: a narrow raised beam the robot must traverse.

    The terrain consists of a start platform, one or more beam segments (raised narrow boxes), and
    an end platform. The sides of each beam segment drop to nothing — the robot will fall off if it
    strays laterally. Each successive segment can shift laterally by a small random y-offset.

    Note on fall-off: start and end platforms rest on ground (z=0). Between them only the beam
    segments exist — there is no ground plane under the beam. A robot that falls off the beam
    will drop to the global ground below (z=−border or z=−platform_height if present), which
    triggers the environment's ``termination_height`` check.

    Difficulty mapping (IsaacLab standard):
        beam_width: easy=``beam_width_range[1]``  → hard=``beam_width_range[0]``
        num_segments: increases with difficulty (1 at easy, ``max_segments`` at hard)
        y_shift per segment: easy=0.0 → hard=``y_shift_per_segment_range[1]``

    Args:
        difficulty: Terrain difficulty in [0, 1]. 0 = easiest, 1 = hardest.
        cfg: Terrain configuration. See :class:`MeshParkourBalanceBeamTerrainCfg`.

    Returns:
        Tuple of (list of trimesh objects, origin ndarray of shape (3,)).
    """
    # -- difficulty-dependent parameters --
    beam_w = cfg.beam_width_range[1] - difficulty * (cfg.beam_width_range[1] - cfg.beam_width_range[0])
    num_segments = max(1, round(1 + difficulty * (cfg.max_segments - 1)))
    max_y_shift = difficulty * cfg.y_shift_per_segment_range[1]

    mid_y = cfg.size[1] / 2.0

    meshes_list: list[trimesh.Trimesh] = []
    goals_list: list[list[float]] = []

    # -- start platform --
    plat_h = cfg.platform_height
    dim = (cfg.platform_length, cfg.size[1], plat_h)
    pos = (cfg.platform_length / 2.0, cfg.size[1] / 2.0, plat_h / 2.0)
    meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    # -- beam segments --
    # available x-length for all segments
    # reserve end platform at the far side
    total_beam_x = cfg.size[0] - cfg.platform_length - cfg.platform_length
    seg_len = total_beam_x / num_segments if num_segments > 0 else total_beam_x

    current_x = cfg.platform_length
    current_y = mid_y  # beam center y tracks lateral shift

    for seg_idx in range(num_segments):
        # lateral shift (except first segment starts at mid)
        if seg_idx > 0:
            shift = float(np.random.uniform(-max_y_shift, max_y_shift))
            current_y = float(np.clip(current_y + shift, beam_w / 2.0, cfg.size[1] - beam_w / 2.0))

        beam_top = cfg.beam_height
        dim = (seg_len, beam_w, beam_top)
        pos = (current_x + seg_len / 2.0, current_y, beam_top / 2.0)
        meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

        # goal: mid-point of this beam segment, top surface
        goals_list.append([current_x + seg_len / 2.0, current_y, beam_top])
        current_x += seg_len

    # -- end platform --
    end_plat_x = cfg.size[0] - current_x
    if end_plat_x > 1e-3:
        dim = (end_plat_x, cfg.size[1], plat_h)
        pos = (current_x + end_plat_x / 2.0, cfg.size[1] / 2.0, plat_h / 2.0)
        meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
        goals_list.append([current_x + end_plat_x / 2.0, cfg.size[1] / 2.0, plat_h])

    # border walls (optional)
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin: center of start platform top surface
    origin = np.array([cfg.platform_length / 2.0, mid_y, plat_h])

    # build goals
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))
    return meshes_list, origin


def parkour_crawl_terrain(
    difficulty: float, cfg: MeshParkourCrawlTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour crawl terrain: a low-ceiling corridor the robot must crouch to pass.

    Layout: ground plane + repeated (side walls + ceiling box) sections separated by open sections.
    The ceiling forces the robot to lower its body. Side walls prevent lateral bypass.

    The ceiling section also includes an extra top-blocker box (``ceiling_top_extra``) that rises
    above the ceiling to prevent jump-bypass (Go2 jump apex ≈ 0.5 m, ceiling top blocker reaches
    ceiling_z_low + ceiling_thickness + ceiling_top_extra ≥ 1.0 m).

    Difficulty mapping (IsaacLab standard):
        ceiling_z_low: easy=``ceiling_height_range[1]`` → hard=``ceiling_height_range[0]``

    WARNING (activation caveat): When proportion > 0 the height_scanner RayCaster fires downward
    rays that hit the ceiling top surface instead of the floor, polluting height_scan observations.
    Activate only after adding ceiling-hit masking in parkour_env.py (obs-worker scope).

    Args:
        difficulty: Terrain difficulty in [0, 1]. 0 = easiest (high ceiling), 1 = hardest (low ceiling).
        cfg: Terrain configuration. See :class:`MeshParkourCrawlTerrainCfg`.

    Returns:
        Tuple of (list of trimesh objects, origin ndarray of shape (3,)).
    """
    # -- difficulty-dependent parameters --
    # IsaacLab standard: difficulty=0 → ceiling HIGH (easy), difficulty=1 → ceiling LOW (hard)
    ceiling_z_low = cfg.ceiling_height_range[1] - difficulty * (
        cfg.ceiling_height_range[1] - cfg.ceiling_height_range[0]
    )

    mid_y = cfg.size[1] / 2.0

    meshes_list: list[trimesh.Trimesh] = []

    # Ground plane covers the full terrain
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []

    # dis_x: current x position of the crawl section start
    dis_x = cfg.platform_length

    for _ in range(cfg.num_crawls):
        # optional x-spacing before this crawl section
        rand_x = float(np.random.uniform(cfg.x_spacing_range[0], cfg.x_spacing_range[1]))
        dis_x += rand_x

        clen = cfg.ceiling_length_x
        cthick = cfg.ceiling_thickness
        wall_h = cfg.side_wall_height
        corr_w = cfg.corridor_width
        side_w = (cfg.size[1] - corr_w) / 2.0  # width of each side wall

        # -- ceiling box --
        # The ceiling bottom face is at ceiling_z_low; top face at ceiling_z_low + cthick
        ceil_center_z = ceiling_z_low + cthick / 2.0
        dim_ceil = (clen, corr_w, cthick)
        pos_ceil = (dis_x + clen / 2.0, mid_y, ceil_center_z)
        meshes_list.append(trimesh.creation.box(dim_ceil, trimesh.transformations.translation_matrix(pos_ceil)))

        # -- ceiling top-blocker: prevents jump bypass --
        # Rises from ceiling_z_low + cthick upward by ceiling_top_extra
        top_extra = cfg.ceiling_top_extra
        if top_extra > 1e-3:
            dim_top = (clen, corr_w, top_extra)
            pos_top = (dis_x + clen / 2.0, mid_y, ceiling_z_low + cthick + top_extra / 2.0)
            meshes_list.append(trimesh.creation.box(dim_top, trimesh.transformations.translation_matrix(pos_top)))

        # -- side walls: block lateral bypass --
        if cfg.side_walls and side_w > 1e-3:
            # left wall: y in [0, side_w]
            dim_wall = (clen, side_w, wall_h)
            pos_left = (dis_x + clen / 2.0, side_w / 2.0, wall_h / 2.0)
            meshes_list.append(trimesh.creation.box(dim_wall, trimesh.transformations.translation_matrix(pos_left)))
            # right wall: y in [side_w + corr_w, size_y]
            right_wall_center_y = side_w + corr_w + side_w / 2.0
            pos_right = (dis_x + clen / 2.0, right_wall_center_y, wall_h / 2.0)
            meshes_list.append(trimesh.creation.box(dim_wall, trimesh.transformations.translation_matrix(pos_right)))

        # goal: just after the ceiling exit (robot has cleared the ceiling), ground level, corridor center
        goals_list.append([dis_x + clen + 0.3, mid_y, 0.0])
        dis_x += clen

    # border walls (optional)
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin: center of start platform at ground level
    origin = np.array([cfg.platform_length / 2.0, mid_y, 0.0])

    # build goals
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))
    return meshes_list, origin


def parkour_slope_terrain(
    difficulty: float, cfg: MeshParkourSlopeTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour slope terrain using wedge primitives.

    Layout: start platform → ascending wedge → flat top → descending wedge → end.
    Each wedge is built as a trimesh with 6 vertices (triangular prism / wedge primitive).
    This is more physically accurate than staircase approximation for smooth inclines.

    Difficulty mapping (IsaacLab standard):
        slope_angle_deg: easy=``slope_angle_deg_range[0]`` → hard=``slope_angle_deg_range[1]``

    Note: The WARP-based edge_mask raycast handles wedge geometry correctly (no axis-aligned
    restriction), so ``feet_edge`` reward will fire at the slope edges. If feet_edge penalties
    on slope terrain degrade learning, add a reward-worker terrain_class=8 mask (obs-worker scope).

    Args:
        difficulty: Terrain difficulty in [0, 1]. 0 = gentlest slope, 1 = steepest slope.
        cfg: Terrain configuration. See :class:`MeshParkourSlopeTerrainCfg`.

    Returns:
        Tuple of (list of trimesh objects, origin ndarray of shape (3,)).
    """
    # -- difficulty-dependent parameters --
    angle_deg = cfg.slope_angle_deg_range[0] + difficulty * (
        cfg.slope_angle_deg_range[1] - cfg.slope_angle_deg_range[0]
    )
    angle_rad = np.deg2rad(angle_deg)

    terrain_w = cfg.size[1]
    mid_y = terrain_w / 2.0

    slope_len = cfg.slope_length
    # height gained by ascending wedge
    slope_h = slope_len * np.tan(angle_rad)

    meshes_list: list[trimesh.Trimesh] = []

    # Ground plane covers the full terrain (base level z=0)
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []
    current_x = cfg.platform_length

    # -- ascending wedge --
    # Wedge: 6 vertices of a triangular prism lying along x-axis.
    # Bottom face at z=0, top-front edge at z=slope_h, top-back edge at z=0.
    # Vertices (local, before translation to current_x):
    #   0: (0, 0, 0)   bottom-back-left
    #   1: (0, W, 0)   bottom-back-right
    #   2: (L, 0, 0)   bottom-front-left
    #   3: (L, W, 0)   bottom-front-right
    #   4: (L, 0, H)   top-front-left
    #   5: (L, W, H)   top-front-right
    # where L=slope_len, W=terrain_w, H=slope_h
    L, W, H = float(slope_len), float(terrain_w), float(slope_h)
    ox = float(current_x)
    v_asc = np.array(
        [
            [ox, 0.0, 0.0],  # 0
            [ox, W, 0.0],  # 1
            [ox + L, 0.0, 0.0],  # 2
            [ox + L, W, 0.0],  # 3
            [ox + L, 0.0, H],  # 4
            [ox + L, W, H],  # 5
        ],
        dtype=np.float32,
    )
    # Faces (outward normals — CCW when viewed from outside):
    # Bottom: 0,1,3,2
    # Back:   0,1,5,4   (the slope face at front)
    # Left:   0,2,4
    # Right:  1,3,5
    # Top-front (degenerate — no face needed, it's an edge)
    # We split quads into triangles:
    f_asc = np.array(
        [
            [0, 3, 1],  # bottom tri 1
            [0, 2, 3],  # bottom tri 2
            [0, 1, 5],  # slope face tri 1
            [0, 5, 4],  # slope face tri 2
            [0, 4, 2],  # left triangular face
            [1, 3, 5],  # right triangular face
            [2, 5, 3],  # front vertical face tri 1  (outward normal: +x)
            [2, 4, 5],  # front vertical face tri 2
        ],
        dtype=np.int32,
    )
    asc_wedge = trimesh.Trimesh(vertices=v_asc, faces=f_asc, process=False)
    meshes_list.append(asc_wedge)
    current_x += slope_len

    # -- flat top section --
    if cfg.flat_top_length > 1e-3 and slope_h > 1e-3:
        dim_top = (cfg.flat_top_length, terrain_w, slope_h)
        pos_top = (current_x + cfg.flat_top_length / 2.0, mid_y, slope_h / 2.0)
        meshes_list.append(trimesh.creation.box(dim_top, trimesh.transformations.translation_matrix(pos_top)))
    # goal: center of flat top surface
    goals_list.append([current_x + cfg.flat_top_length / 2.0, mid_y, slope_h])
    current_x += cfg.flat_top_length

    # -- descending wedge (mirror of ascending) --
    # Vertices:
    #   0: (0, 0, H)   top-back-left
    #   1: (0, W, H)   top-back-right
    #   2: (L, 0, H)   top-front-left   (at start of descend, still at H)
    #   3: (L, W, H)   top-front-right
    #   4: (L, 0, 0)   bottom-front-left (descends to 0)
    #   5: (L, W, 0)   bottom-front-right
    # Actually descend: high at back (current_x), low at front (current_x+L)
    ox2 = float(current_x)
    v_desc = np.array(
        [
            [ox2, 0.0, H],  # 0 top-back-left
            [ox2, W, H],  # 1 top-back-right
            [ox2 + L, 0.0, H],  # 2 -- unused, descend goes to 0
            [ox2 + L, W, H],  # 3 -- unused
            [ox2 + L, 0.0, 0.0],  # 4 bottom-front-left
            [ox2 + L, W, 0.0],  # 5 bottom-front-right
            [ox2, 0.0, 0.0],  # 6 bottom-back-left  (base of wedge)
            [ox2, W, 0.0],  # 7 bottom-back-right
        ],
        dtype=np.float32,
    )
    # For descending wedge we only need 6 meaningful vertices (box-minus-triangle):
    # Use a simpler form: back face is at height H, front face descends to 0.
    # 6 vertices: 0,1 (top-back), 4,5 (bottom-front), 6,7 (bottom-back)
    v_desc = np.array(
        [
            [ox2, 0.0, H],  # 0 top-back-left
            [ox2, W, H],  # 1 top-back-right
            [ox2 + L, 0.0, 0.0],  # 2 bottom-front-left
            [ox2 + L, W, 0.0],  # 3 bottom-front-right
            [ox2, 0.0, 0.0],  # 4 bottom-back-left
            [ox2, W, 0.0],  # 5 bottom-back-right
        ],
        dtype=np.float32,
    )
    f_desc = np.array(
        [
            [0, 1, 5],  # bottom tri 1 (back bottom)
            [0, 5, 4],  # bottom tri 2
            [4, 5, 3],  # bottom tri 3 (front bottom)
            [4, 3, 2],  # bottom tri 4
            [0, 4, 2],  # left triangular face
            [0, 2, 1],  # slope face tri 1
            [1, 2, 3],  # slope face tri 2
            [1, 3, 5],  # right triangular face
        ],
        dtype=np.int32,
    )
    desc_wedge = trimesh.Trimesh(vertices=v_desc, faces=f_desc, process=False)
    meshes_list.append(desc_wedge)

    # goal: just past the descend, back at ground level
    goals_list.append([current_x + slope_len + 0.5, mid_y, 0.0])

    # border walls (optional)
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin: center of start platform at ground level
    origin = np.array([cfg.platform_length / 2.0, mid_y, 0.0])

    # build goals
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))
    return meshes_list, origin


def parkour_zigzag_hurdles_terrain(
    difficulty: float, cfg: MeshParkourZigzagHurdlesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour zigzag-hurdles terrain with left/right alternating passage corridors.

    Each hurdle is a full-width wall with one half left open (corridor side alternates left→right).
    - Even hurdles: right half open (left half blocked) → robot must pass on the right
    - Odd hurdles:  left half open (right half blocked) → robot must pass on the left
    This forces lateral steering without requiring an explicit lateral velocity command.

    Difficulty mapping (IsaacLab standard):
        hurdle_height: easy=``hurdle_height_range[0]`` → hard=``hurdle_height_range[1]``
        x_spacing: easy=``x_spacing_range[1]`` → hard=``x_spacing_range[0]`` (tighter spacing)

    Args:
        difficulty: Terrain difficulty in [0, 1]. 0 = easiest, 1 = hardest.
        cfg: Terrain configuration. See :class:`MeshParkourZigzagHurdlesTerrainCfg`.

    Returns:
        Tuple of (list of trimesh objects, origin ndarray of shape (3,)).
    """
    # -- difficulty-dependent parameters --
    hurdle_h = cfg.hurdle_height_range[0] + difficulty * (cfg.hurdle_height_range[1] - cfg.hurdle_height_range[0])
    # tighter spacing = harder (smaller x_spacing)
    x_spacing = cfg.x_spacing_range[1] - difficulty * (cfg.x_spacing_range[1] - cfg.x_spacing_range[0])

    terrain_w = cfg.size[1]
    mid_y = terrain_w / 2.0
    corr_w = cfg.corridor_width  # width of the open passage

    meshes_list: list[trimesh.Trimesh] = []

    # Ground plane covers full terrain
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []

    dis_x = cfg.platform_length
    for k in range(cfg.num_hurdles):
        dis_x += x_spacing

        if hurdle_h <= 1e-3:
            goals_list.append([dis_x + cfg.hurdle_thickness + 0.3, mid_y, 0.0])
            continue

        if k % 2 == 0:
            # Even: open corridor on the RIGHT side, wall on the LEFT
            # Wall covers y in [0, terrain_w - corr_w]
            wall_w = terrain_w - corr_w
            corridor_center_y = terrain_w - corr_w / 2.0
            if wall_w > 1e-3:
                dim = (cfg.hurdle_thickness, wall_w, hurdle_h)
                pos = (dis_x + cfg.hurdle_thickness / 2.0, wall_w / 2.0, hurdle_h / 2.0)
                meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
        else:
            # Odd: open corridor on the LEFT side, wall on the RIGHT
            # Wall covers y in [corr_w, terrain_w]
            wall_w = terrain_w - corr_w
            corridor_center_y = corr_w / 2.0
            if wall_w > 1e-3:
                dim = (cfg.hurdle_thickness, wall_w, hurdle_h)
                pos = (dis_x + cfg.hurdle_thickness / 2.0, corr_w + wall_w / 2.0, hurdle_h / 2.0)
                meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

        # goal: just past hurdle, at corridor center
        goals_list.append([dis_x + cfg.hurdle_thickness + 0.3, corridor_center_y, 0.0])
        dis_x += cfg.hurdle_thickness

    # border walls (optional)
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin: center of start platform at ground level
    origin = np.array([cfg.platform_length / 2.0, mid_y, 0.0])

    # build goals
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))
    return meshes_list, origin


def parkour_rough_blocks_terrain(
    difficulty: float, cfg: MeshParkourRoughBlocksTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour rough-blocks terrain: a grid of small randomly-height blocks.

    A start and end platform flank a middle section filled with small box-shaped blocks
    arranged on a regular grid. Each block has an independent random height drawn from
    [0, block_height_max(difficulty)]. Blocks may be omitted with probability (1-density).

    Difficulty mapping (IsaacLab standard):
        block_height_max: easy=``block_height_range[0]`` → hard=``block_height_range[1]``
        block_density: easy=``block_density_range[0]`` → hard=``block_density_range[1]``

    Note: feet_edge reward will fire on block edges. If this degrades learning on this terrain
    class, consider applying a terrain_class=10 reward mask (reward-worker scope).

    Args:
        difficulty: Terrain difficulty in [0, 1]. 0 = mostly flat, 1 = maximum height variation.
        cfg: Terrain configuration. See :class:`MeshParkourRoughBlocksTerrainCfg`.

    Returns:
        Tuple of (list of trimesh objects, origin ndarray of shape (3,)).
    """
    # -- difficulty-dependent parameters --
    block_h_max = cfg.block_height_range[0] + difficulty * (cfg.block_height_range[1] - cfg.block_height_range[0])
    density = cfg.block_density_range[0] + difficulty * (cfg.block_density_range[1] - cfg.block_density_range[0])

    terrain_w = cfg.size[1]
    mid_y = terrain_w / 2.0

    meshes_list: list[trimesh.Trimesh] = []

    # Ground plane covers full terrain (base reference)
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []

    # -- start platform --
    # The platform is implicit (ground plane flat from 0 to platform_length)
    rough_start_x = cfg.platform_length
    rough_end_x = cfg.size[0] - cfg.platform_length  # symmetric end platform
    rough_len = rough_end_x - rough_start_x

    # -- block grid --
    # Use block_size as the grid cell size; blocks fill each cell stochastically
    bs = cfg.block_size

    n_cols_x = max(1, int(rough_len / bs))
    n_cols_y = max(1, int(terrain_w / bs))

    for ix in range(n_cols_x):
        for iy in range(n_cols_y):
            # stochastic density: skip with probability (1-density)
            if float(np.random.rand()) > density:
                continue
            if block_h_max < 1e-4:
                continue

            bh = float(np.random.uniform(0.0, block_h_max))
            if bh < 1e-4:
                continue

            bx_center = rough_start_x + (ix + 0.5) * bs
            by_center = (iy + 0.5) * bs

            dim = (bs, bs, bh)
            pos = (bx_center, by_center, bh / 2.0)
            meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))

    # -- goals: evenly spaced through the rough section --
    n_goals = cfg.num_goals
    for g in range(n_goals):
        gx = rough_start_x + (g + 0.5) * rough_len / n_goals
        goals_list.append([gx, mid_y, block_h_max / 2.0])  # approximate surface level

    # border walls (optional)
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin: center of start platform at ground level
    origin = np.array([cfg.platform_length / 2.0, mid_y, 0.0])

    # build goals
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))
    return meshes_list, origin


# =============================================================================
# Cfg dataclasses for new terrain functions.
# All inherit SubTerrainBaseCfg; @configclass is required.
# proportion=0.0 by default — inactive until user explicitly sets > 0.
# =============================================================================


@configclass
class MeshParkourSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the stepping-stones parkour terrain.

    Discrete raised stones arranged along x with lateral jitter.
    Difficulty controls stone size (↓), gap length (↑), and jitter amplitude (↑).

    Activation caveat: proportion=0.0 by default. Set proportion>0 to activate.
    """

    function = parkour_stepping_stones_terrain

    proportion: float = 0.0
    """Inactive by default. Set > 0 to activate. Defaults to 0.0."""

    platform_length: float = 2.5
    """Length of the flat start platform along x before the first stone (m). Defaults to 2.5."""

    num_stones: int = 8
    """Number of stepping stones. Defaults to 8."""

    stone_size_xy_range: tuple[float, float] = (0.20, 0.40)
    """Min/max side length of each square stone (m). Interpolated by difficulty:
    difficulty=0 → 0.40 m (large, easy), difficulty=1 → 0.20 m (small, hard). Defaults to (0.20, 0.40)."""

    stone_height_range: tuple[float, float] = (0.05, 0.15)
    """Min/max height of each stone above ground (m). Interpolated by difficulty. Defaults to (0.05, 0.15)."""

    gap_length_range: tuple[float, float] = (0.20, 0.45)
    """Min/max gap between consecutive stones along x (m). Interpolated by difficulty:
    difficulty=0 → 0.20 m (short gap), difficulty=1 → 0.45 m (long gap). Defaults to (0.20, 0.45)."""

    lateral_jitter_range: tuple[float, float] = (0.0, 0.30)
    """Max lateral (y) jitter of each stone center from terrain midline (m).
    Interpolated by difficulty: difficulty=0 → 0 jitter, difficulty=1 → 0.30 m. Defaults to (0.0, 0.30)."""

    border_width: float = 0.0
    """Width of optional border walls around terrain (m). Defaults to 0.0."""

    border_height: float = 0.5
    """Height of optional border walls (m). Defaults to 0.5."""

    num_goals: int = 8
    """Number of goal waypoints per terrain tile. Defaults to 8."""


@configclass
class MeshParkourBalanceBeamTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the balance-beam parkour terrain.

    A narrow raised beam (or segmented beams with lateral shifts) flanked by start/end platforms.
    The robot must traverse the beam without falling off.

    Activation caveat: proportion=0.0 by default. At hard difficulty (beam_width=0.20 m) only
    2 height_scan cells span the beam — the policy may not resolve it clearly. Start with
    proportion>0 at easy rows first (beam_width=0.50 m = 5 cells).
    """

    function = parkour_balance_beam_terrain

    proportion: float = 0.0
    """Inactive by default. Set > 0 to activate. Defaults to 0.0."""

    platform_length: float = 2.5
    """Length of each start/end platform along x (m). Defaults to 2.5."""

    platform_height: float = 0.15
    """Height of start/end platforms and the beam above z=0 (m). Defaults to 0.15."""

    beam_width_range: tuple[float, float] = (0.20, 0.50)
    """Min/max beam width along y (m). Interpolated by difficulty:
    difficulty=0 → 0.50 m (wide), difficulty=1 → 0.20 m (narrow). Defaults to (0.20, 0.50)."""

    beam_height: float = 0.15
    """Height of the beam box above z=0 (m). Should equal platform_height for smooth entry.
    Defaults to 0.15."""

    max_segments: int = 3
    """Maximum number of beam segments. Actual count = max(1, round(1 + difficulty*(max_segments-1))).
    Defaults to 3."""

    y_shift_per_segment_range: tuple[float, float] = (0.0, 0.40)
    """Max lateral shift between consecutive beam segments (m).
    Interpolated by difficulty: difficulty=0 → 0 shift, difficulty=1 → 0.40 m. Defaults to (0.0, 0.40)."""

    border_width: float = 0.0
    """Width of optional border walls (m). Defaults to 0.0."""

    border_height: float = 0.5
    """Height of optional border walls (m). Defaults to 0.5."""

    num_goals: int = 8
    """Number of goal waypoints per terrain tile. Defaults to 8."""


@configclass
class MeshParkourCrawlTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the crawl (low-ceiling corridor) parkour terrain.

    The robot must lower its body to pass under the ceiling boxes. Side walls block lateral bypass.
    A ceiling top-blocker prevents jump bypass.

    Activation caveat: proportion=0.0 by default.
    IMPORTANT — activating crawl without height_scan masking will cause the downward RayCaster
    to hit the ceiling top surface instead of the floor, injecting false height values into
    height_scan observations. Add ceiling-hit masking in parkour_env.py before activating.
    """

    function = parkour_crawl_terrain

    proportion: float = 0.0
    """Inactive by default. Set > 0 to activate (after adding height_scan masking). Defaults to 0.0."""

    platform_length: float = 2.5
    """Length of flat start platform before the first crawl section (m). Defaults to 2.5."""

    num_crawls: int = 3
    """Number of ceiling sections per terrain tile. Defaults to 3."""

    ceiling_height_range: tuple[float, float] = (0.28, 0.50)
    """Min/max ceiling clearance (bottom face of ceiling box above z=0), in m.
    Interpolated by difficulty (IsaacLab standard): difficulty=0 → 0.50 m (high, easy),
    difficulty=1 → 0.28 m (low, hard). Go2 standing hip ~0.30 m; 0.28 m forces full crouch.
    Lower bound ≥ 0.17 m (Go2 kinematic minimum). Defaults to (0.28, 0.50)."""

    ceiling_length_x: float = 1.2
    """x-length of each ceiling section (m). ≥ 0.8 m prevents single-jump bypass. Defaults to 1.2."""

    ceiling_thickness: float = 0.10
    """Thickness of the ceiling box along z (m). Defaults to 0.10."""

    ceiling_top_extra: float = 0.50
    """Extra box height above the ceiling box to block jump bypass (m).
    ceiling_z_low + ceiling_thickness + ceiling_top_extra ≥ 1.0 m recommended
    (Go2 jump apex ≈ 0.5 m). Defaults to 0.50."""

    corridor_width: float = 1.2
    """Width of the clear passage along y (m). Go2 body width ~0.3 m. Defaults to 1.2."""

    side_wall_height: float = 1.0
    """Height of side walls along y (m). Prevents lateral bypass of ceiling. Defaults to 1.0."""

    side_walls: bool = True
    """If True, side walls are placed to block lateral bypass. Defaults to True."""

    x_spacing_range: tuple[float, float] = (1.0, 2.0)
    """Min/max spacing between consecutive crawl sections along x (m). Defaults to (1.0, 2.0)."""

    border_width: float = 0.0
    """Width of optional border walls (m). Defaults to 0.0."""

    border_height: float = 0.5
    """Height of optional border walls (m). Defaults to 0.5."""

    num_goals: int = 8
    """Number of goal waypoints per terrain tile. Defaults to 8."""


@configclass
class MeshParkourSlopeTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the slope parkour terrain (wedge primitive).

    Layout: flat start platform → ascending wedge → flat top → descending wedge.
    Wedges are built as trimesh triangular prisms (6 vertices), not staircase approximations.
    The WARP-based edge_mask raycast handles wedge geometry correctly.

    Activation caveat: proportion=0.0 by default. feet_edge reward will fire at the wedge
    lateral edges; if this harms learning add a terrain_class=8 mask in reward-worker.
    """

    function = parkour_slope_terrain

    proportion: float = 0.0
    """Inactive by default. Set > 0 to activate. Defaults to 0.0."""

    platform_length: float = 2.5
    """Length of flat start platform (m). Defaults to 2.5."""

    slope_angle_deg_range: tuple[float, float] = (5.0, 25.0)
    """Min/max slope angle in degrees. Interpolated by difficulty:
    difficulty=0 → 5° (gentle), difficulty=1 → 25° (steep). Defaults to (5.0, 25.0)."""

    slope_length: float = 4.0
    """Horizontal length of each slope section (m). Defaults to 4.0."""

    flat_top_length: float = 1.5
    """Length of the flat section between ascending and descending slopes (m). Defaults to 1.5."""

    border_width: float = 0.0
    """Width of optional border walls (m). Defaults to 0.0."""

    border_height: float = 0.5
    """Height of optional border walls (m). Defaults to 0.5."""

    num_goals: int = 8
    """Number of goal waypoints per terrain tile. Defaults to 8."""


@configclass
class MeshParkourZigzagHurdlesTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the zigzag-hurdles parkour terrain.

    Left/right alternating hurdles force lateral steering without explicit lateral commands.
    Even hurdles: corridor on the right. Odd hurdles: corridor on the left.

    Activation caveat: proportion=0.0 by default. lin_vel_y_range=[0,0] discourages but does
    not hard-block lateral movement; the robot will learn to swing left/right via goal tracking.
    """

    function = parkour_zigzag_hurdles_terrain

    proportion: float = 0.0
    """Inactive by default. Set > 0 to activate. Defaults to 0.0."""

    platform_length: float = 2.5
    """Length of flat start platform (m). Defaults to 2.5."""

    num_hurdles: int = 6
    """Number of hurdles per terrain tile. Defaults to 6."""

    hurdle_thickness: float = 0.30
    """Thickness of each hurdle wall along x (m). Defaults to 0.30."""

    hurdle_height_range: tuple[float, float] = (0.10, 0.25)
    """Min/max hurdle height (m). Interpolated by difficulty:
    difficulty=0 → 0.10 m, difficulty=1 → 0.25 m. Defaults to (0.10, 0.25)."""

    corridor_width: float = 2.0
    """Width of the open passage in each hurdle along y (m). Defaults to 2.0."""

    x_spacing_range: tuple[float, float] = (1.5, 2.4)
    """Min/max spacing between hurdles along x (m). Interpolated by difficulty:
    difficulty=0 → 2.4 m (relaxed), difficulty=1 → 1.5 m (tight). Defaults to (1.5, 2.4)."""

    border_width: float = 0.0
    """Width of optional border walls (m). Defaults to 0.0."""

    border_height: float = 0.5
    """Height of optional border walls (m). Defaults to 0.5."""

    num_goals: int = 8
    """Number of goal waypoints per terrain tile. Defaults to 8."""


@configclass
class MeshParkourRoughBlocksTerrainCfg(SubTerrainBaseCfg):
    """Configuration for the rough-blocks parkour terrain.

    A field of randomly-height small blocks on a regular grid. Tests unstructured ground
    navigation and foot placement on non-uniform surfaces.

    Activation caveat: proportion=0.0 by default. feet_edge reward will trigger on block
    edges frequently; consider terrain_class=10 reward mask in reward-worker if needed.
    """

    function = parkour_rough_blocks_terrain

    proportion: float = 0.0
    """Inactive by default. Set > 0 to activate. Defaults to 0.0."""

    platform_length: float = 2.5
    """Length of flat start and end platforms (m). Defaults to 2.5."""

    block_size: float = 0.30
    """Grid cell size (= block footprint side length) in m. Defaults to 0.30."""

    block_height_range: tuple[float, float] = (0.0, 0.10)
    """Min/max block height above z=0 (m). Interpolated by difficulty:
    difficulty=0 → max 0.02 m (near-flat), difficulty=1 → max 0.10 m. Defaults to (0.0, 0.10)."""

    block_density_range: tuple[float, float] = (0.6, 0.9)
    """Min/max probability of placing a block in each grid cell. Interpolated by difficulty:
    difficulty=0 → 0.6 (sparse), difficulty=1 → 0.9 (dense). Defaults to (0.6, 0.9)."""

    border_width: float = 0.0
    """Width of optional border walls (m). Defaults to 0.0."""

    border_height: float = 0.5
    """Height of optional border walls (m). Defaults to 0.5."""

    num_goals: int = 8
    """Number of goal waypoints per terrain tile. Defaults to 8."""


def parkour_stair_with_midgoals_terrain(
    difficulty: float, cfg: MeshParkourStairWithMidGoalsTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a parkour stair terrain with intermediate ascend/descend goals and a run-out last goal.

    Geometry is identical to the core ``parkour_stair_terrain`` (mesh_terrains.py:1043):
    a start platform followed by ``num_stairs`` cycles of
    ``ascending steps -> flat top -> descending steps -> flat bottom``.

    Goal-list changes vs core (mesh geometry is unchanged):

    1. Intermediate goals — each cycle emits 4 goals instead of 2, in the order
       ``[ascend-mid, top, descend-mid, bottom]``:

       * ``ascend-mid``: centre of the mid ascending step (index ``num_steps_per_stair // 2``),
         placed on that step's *top surface* (z = ``(m + 1) * sh``).
       * ``top``: centre of the flat top section (z = ``num_steps_per_stair * sh``) — same as core.
       * ``descend-mid``: centre of the mid descending step (index ``num_steps_per_stair // 2``),
         placed on that step's *top surface* (z = ``(num_steps_per_stair - m - 1) * sh``).
       * ``bottom``: centre of the flat bottom section (z = 0) — same as core, except the
         final cycle (see #2).

       With ``num_stairs = 2`` this yields exactly ``4 * 2 = 8`` goals, matching
       ``num_goals = 8`` so the core padding branch is never taken.

    2. Run-out last goal — the final bottom goal is pushed away from the stairs by
       ``last_goal_runout`` metres so the robot has room to reach it instead of u-turning
       on top of it.  The full tile ground plane already covers this run-out region
       (bottom flats are at z = 0), so no extra mesh box is required.  Only the final
       cycle's bottom goal is moved; intermediate cycles keep the flat-centre bottom goal.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain. Uses all fields from
            :class:`MeshParkourStairWithMidGoalsTerrainCfg` (a subclass of the core
            ``MeshParkourStairTerrainCfg`` with the extra ``last_goal_runout`` field).

    Returns:
        A tuple containing the tri-mesh list of the terrain and the origin of the terrain (in m).
    """
    # resolve difficulty-dependent parameters (identical to core)
    sw = float(np.random.uniform(cfg.stair_width_range[0], cfg.stair_width_range[1]))
    sh = cfg.stair_height_range[0] + difficulty * (cfg.stair_height_range[1] - cfg.stair_height_range[0])

    terrain_w = cfg.size[1]
    mid_y = terrain_w / 2.0

    meshes_list: list[trimesh.Trimesh] = []

    # ground plane (covers the full tile at z=0 — also provides the run-out floor)
    meshes_list.append(make_plane(cfg.size, height=0.0, center_zero=False))

    goals_list: list[list[float]] = []

    # mid-step index (same for ascending and descending sections)
    m = cfg.num_steps_per_stair // 2

    current_x = cfg.platform_length

    for cycle in range(cfg.num_stairs):
        # ── ascending steps ───────────────────────────────────────────────────
        asc_start_x = current_x
        for k in range(cfg.num_steps_per_stair):
            step_top = (k + 1) * sh
            if step_top > 1e-3 and sw > 1e-3:
                dim = (sw, terrain_w, step_top)
                pos = (current_x + sw / 2.0, mid_y, step_top / 2.0)
                meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
            current_x += sw

        # goal (ascend-mid): centre of the mid ascending step, on its top surface
        ascend_mid_x = asc_start_x + m * sw + sw / 2.0
        ascend_mid_z = (m + 1) * sh
        goals_list.append([ascend_mid_x, mid_y, ascend_mid_z])

        # ── flat top section ──────────────────────────────────────────────────
        top_h = cfg.num_steps_per_stair * sh
        # goal (top): centre of the flat top (landing after ascent) — same as core
        goals_list.append([current_x + cfg.flat_section_length / 2.0, mid_y, top_h])
        if top_h > 1e-3 and cfg.flat_section_length > 1e-3:
            dim = (cfg.flat_section_length, terrain_w, top_h)
            pos = (current_x + cfg.flat_section_length / 2.0, mid_y, top_h / 2.0)
            meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
        current_x += cfg.flat_section_length

        # ── descending steps ──────────────────────────────────────────────────
        desc_start_x = current_x
        for k in range(cfg.num_steps_per_stair):
            step_top = (cfg.num_steps_per_stair - k - 1) * sh
            if step_top > 1e-3 and sw > 1e-3:
                dim = (sw, terrain_w, step_top)
                pos = (current_x + sw / 2.0, mid_y, step_top / 2.0)
                meshes_list.append(trimesh.creation.box(dim, trimesh.transformations.translation_matrix(pos)))
            current_x += sw

        # goal (descend-mid): centre of the mid descending step, on its top surface
        descend_mid_x = desc_start_x + m * sw + sw / 2.0
        descend_mid_z = (cfg.num_steps_per_stair - m - 1) * sh
        goals_list.append([descend_mid_x, mid_y, descend_mid_z])

        # ── flat bottom section (ground level — no box needed, advance x only) ─
        last_step_end_x = current_x
        if cycle == cfg.num_stairs - 1:
            # run-out: push the final bottom goal away from the stairs so the robot
            # has room to reach it instead of u-turning on top of it.
            bottom_x = last_step_end_x + cfg.last_goal_runout
        else:
            # intermediate cycles keep the flat-centre bottom goal (core behaviour)
            bottom_x = last_step_end_x + cfg.flat_section_length / 2.0
        goals_list.append([bottom_x, mid_y, 0.0])
        current_x += cfg.flat_section_length

    # border walls
    if cfg.border_width > 0.0:
        inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
        border_center = (cfg.size[0] / 2.0, cfg.size[1] / 2.0, cfg.border_height / 2.0)
        meshes_list += make_border(cfg.size, inner_size, cfg.border_height, border_center)

    # origin at center of start platform (ground level)
    origin = np.array([cfg.platform_length / 2.0, cfg.size[1] / 2.0, 0.0])

    # build goals array: pad with last goal if needed, truncate to num_goals.
    # With num_stairs cycles × 4 goals == num_goals (8 == 8) the padding branch is
    # never taken; it is retained for robustness against cfg changes.
    _raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
    if len(_raw) < cfg.num_goals:
        _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
    goals = _raw[: cfg.num_goals]

    # publish to module-level registry (consumed by parkour_env after terrain build)
    PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))

    return meshes_list, origin


@configclass
class MeshParkourStairWithMidGoalsTerrainCfg(MeshParkourStairTerrainCfg):
    """Configuration for :func:`parkour_stair_with_midgoals_terrain`.

    Inherits every field from the core :class:`MeshParkourStairTerrainCfg`
    (platform_length, stair_width_range, stair_height_range, flat_section_length,
    num_steps_per_stair, num_stairs, border_width, border_height, num_goals) and only

    1. swaps ``function`` to the task-level mid-goal stair generator, and
    2. adds the ``last_goal_runout`` parameter (run-out distance for the final goal).
    """

    function = parkour_stair_with_midgoals_terrain

    last_goal_runout: float = 1.0
    """Distance (m) past the final descending step at which the final bottom goal is placed.

    Pushes the last goal away from the stairs so the robot has room to reach it instead of
    u-turning on top of it.  The full tile ground plane already covers this run-out region
    (bottom flats are at z = 0), so no extra mesh is added.  Must keep the final goal x inside
    the tile ``size[0]``.  Defaults to 1.0.
    """

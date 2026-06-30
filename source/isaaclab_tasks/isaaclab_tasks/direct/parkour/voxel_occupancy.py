# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Voxel occupancy GT builder for parkour teacher privileged observations.

Converts the existing clearance-scanner ray hits into a 3D occupancy grid
using the same sensing model (sensor-observable, occluded, first-surface only).

Coordinate convention (base yaw-aligned frame, sensor origin at base+0.05 m):
    x = forward, y = left, z = up

Occupancy values:
    -1 : unknown (behind a hit surface — ray never reached; or outside grid)
     0 : free    (ray passed through without obstruction)
     1 : occupied (first surface hit by at least one ray)

Key design choice — no world-frame transform required:
    Because the clearance scanner uses ``ray_alignment="yaw"``, hit positions in
    the *local* yaw-aligned frame are simply ``dir_local * dist`` where
    ``dir_local`` are the unit directions from ``clearance_3d_pattern`` and
    ``dist`` is the pre-clamp Euclidean distance from the same scanner.
    This avoids any explicit rotation and makes the voxel grid trivially
    consistent with the clearance vector.

Sensor offset: the scanner sits at base+0.05 m (z). The grid is defined
relative to the sensor origin, so no z correction is needed inside this
module. The caller passes grid bounds in sensor-local coords and the 0.05
offset from base is transparent.

Filling algorithm: fixed-step sampling along each ray at step = resolution/2.
3D-DDA would be more exact but is harder to vectorise; oversampling is
idempotent for the free-marking pass, and hit voxels are always set last
(occupied wins conflicts).

Usage::

    cfg = VoxelOccupancyCfg()
    dirs = build_ray_dirs(cfg, device)           # cache once
    grid = fill_voxel_grid(dirs, dist, is_hit, cfg, num_envs, device)
    # grid: (num_envs, X, Y, Z) int8 in {-1, 0, 1}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .clearance_3d_pattern import Clearance3DPatternCfg


@configclass
class VoxelOccupancyCfg:
    """Configuration for the sensor-observable voxel occupancy GT grid.

    Grid bounds are in the sensor-local yaw-aligned frame (x=fwd, y=left, z=up)
    with origin at the clearance-scanner mount point (base + 0.05 m z).

    Default bounds cover the robot's immediate traversable volume:
        x : [-0.6, 2.0] m  (rearward buffer + forward reach)
        y : [-1.0, 1.0] m  (lateral clearance)
        z : [-0.6, 0.6] m  (floor below feet ... ceiling above back)

    At resolution 0.1 m → 27 × 21 × 13 = 7371 voxels per env.
    At resolution 0.15 m → 18 × 14 × 9 = 2268 voxels per env.
    """

    x_range: tuple[float, float] = (-0.6, 2.0)
    """Grid x bounds in sensor-local frame (m). Forward = positive."""

    y_range: tuple[float, float] = (-1.0, 1.0)
    """Grid y bounds in sensor-local frame (m). Left = positive."""

    z_range: tuple[float, float] = (-0.6, 0.6)
    """Grid z bounds in sensor-local frame (m). Up = positive."""

    resolution: float = 0.1
    """Voxel side length (m). Filling step = resolution / 2."""

    max_distance: float = 4.0
    """Ray max distance (m). Must match the clearance scanner's max_distance."""

    fill_chunk_size: int = 1024
    """Number of environments processed per chunk inside ``fill_voxel_grid``.

    Chunking reduces peak transient CUDA memory by processing the env dimension
    in slices rather than all at once.  With ``num_envs=4096`` and the default
    ``chunk_size=1024`` (4 chunks):

    * Free-marking pass:  peak per iteration ≈ (1024, K_r) × float32 + 2×int64
                          down from (4096, K_r) — ~4× reduction.
    * Occupied-marking pass: peak ≈ (1024, 294, 3) float32 + 4×(1024, 294) int64
                              down from (4096, …) — ~4× reduction.

    Lower values reduce fragmentation pressure further at the cost of more Python
    loop overhead.  ``fill_chunk_size=num_envs`` disables chunking (original
    behaviour).  Must be ≥ 1.
    """


def build_ray_dirs(cfg_clearance: Clearance3DPatternCfg, device: str) -> torch.Tensor:
    """Return cached unit-vector ray directions in sensor-local frame.

    Calls the same ``clearance_3d_pattern`` function used by the clearance
    scanner so directions are guaranteed identical.

    Args:
        cfg_clearance: A ``Clearance3DPatternCfg`` instance with the same
            parameters used to create the ``_clearance_scanner`` in the env.
        device: PyTorch device string.

    Returns:
        Tensor of shape ``(N_rays, 3)`` — unit vectors x=fwd, y=left, z=up.
    """
    from .clearance_3d_pattern import clearance_3d_pattern

    _, ray_dirs = clearance_3d_pattern(cfg_clearance, device)  # (N, 3)
    return ray_dirs


def _fill_voxel_grid_loop(
    ray_dirs: torch.Tensor,
    dist_pre_clamp: torch.Tensor,
    is_hit: torch.Tensor,
    cfg: VoxelOccupancyCfg,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Reference implementation using a per-env Python loop.

    Semantically identical to ``fill_voxel_grid`` but O(num_envs) Python
    iterations.  **Do not call in production** — use for correctness
    verification only (e.g. ``torch.equal(fill_voxel_grid(...), _fill_voxel_grid_loop(...))``)
    at small *num_envs*.
    """
    res = cfg.resolution
    step = res * 0.5

    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range

    nx = int(round((x_max - x_min) / res)) + 1
    ny = int(round((y_max - y_min) / res)) + 1
    nz = int(round((z_max - z_min) / res)) + 1

    grid = torch.full((num_envs, nx, ny, nz), fill_value=-1, dtype=torch.int8, device=device)

    dist_clamped = dist_pre_clamp.clamp(max=cfg.max_distance)
    n_steps = int(cfg.max_distance / step) + 1
    t_vals = torch.arange(n_steps, dtype=torch.float32, device=device) * step

    pts = t_vals[:, None, None] * ray_dirs[None, :, :]
    ix = ((pts[..., 0] - x_min) / res).round().long()
    iy = ((pts[..., 1] - y_min) / res).round().long()
    iz = ((pts[..., 2] - z_min) / res).round().long()
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
    flat_idx = ix * (ny * nz) + iy * nz + iz
    flat_idx_safe = flat_idx.clamp(0, nx * ny * nz - 1)

    for e in range(num_envs):
        dist_e = dist_clamped[e]
        t_valid = (t_vals[:, None] < dist_e[None, :]) & in_bounds
        free_flat = flat_idx_safe[t_valid]
        grid_e_flat = grid[e].view(-1)
        is_unknown = grid_e_flat[free_flat] == -1
        grid_e_flat[free_flat[is_unknown]] = 0

        hit_mask = is_hit[e]
        if hit_mask.any():
            hit_pts = dist_e[hit_mask, None] * ray_dirs[hit_mask]
            hix = ((hit_pts[:, 0] - x_min) / res).round().long()
            hiy = ((hit_pts[:, 1] - y_min) / res).round().long()
            hiz = ((hit_pts[:, 2] - z_min) / res).round().long()
            hib = (hix >= 0) & (hix < nx) & (hiy >= 0) & (hiy < ny) & (hiz >= 0) & (hiz < nz)
            if hib.any():
                grid[e, hix[hib], hiy[hib], hiz[hib]] = 1

    return grid


def fill_voxel_grid(
    ray_dirs: torch.Tensor,
    dist_pre_clamp: torch.Tensor,
    is_hit: torch.Tensor,
    cfg: VoxelOccupancyCfg,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Fill a (num_envs, X, Y, Z) int8 voxel occupancy grid.

    Algorithm (fixed-step sampling):

    1. For each ray sample points at ``t = 0, step, 2*step, ...`` up to
       ``min(dist_pre_clamp, max_distance)``.
    2. Points that land inside the grid → mark free (0).
    3. The hit voxel (point at ``t = dist_pre_clamp``) for true hits only →
       mark occupied (1), overwriting free.
    4. All other voxels stay unknown (-1).

    Vectorisation strategy (chunked):
        The former O(num_envs=4096) per-env Python loop caused physics stalls.
        This implementation replaces it with:

        *  **Free-marking pass**: O(n_steps × n_chunks) Python iterations.
           Each iteration dispatches a ``(chunk_size, K_r)`` bool comparison.
           Peak transient per iteration ≈ ``(chunk_size, K_r)`` × float32 +
           2 × ``(M,)`` int64.  At chunk_size=1024 and n_steps=81: 324 iters,
           peak ~6 MB (vs ~25 MB without chunking).
        *  **Occupied-marking pass**: O(n_chunks) iterations — batched
           ``(chunk_size, R)`` index computation per chunk.  Peak ~18 MB
           (vs ~74 MB without chunking at 4096 envs).

        Both passes are semantically identical to ``_fill_voxel_grid_loop``
        (reference).  Verify with
        ``torch.equal(fill_voxel_grid(...), _fill_voxel_grid_loop(...))``
        at small num_envs.  ``cfg.fill_chunk_size`` controls the trade-off
        between memory peak and Python loop overhead.

    Args:
        ray_dirs:       ``(N_rays, 3)`` unit vectors in sensor-local frame.
                        Origin = sensor mount = base + 0.05 m.
        dist_pre_clamp: ``(num_envs, N_rays)`` Euclidean distances,
                        **pre-clamp** (inf on miss).
        is_hit:         ``(num_envs, N_rays)`` bool mask: True = real geometry
                        hit within max_distance.
        cfg:            ``VoxelOccupancyCfg`` instance.
        num_envs:       Number of parallel environments.
        device:         PyTorch device string.

    Returns:
        ``(num_envs, X, Y, Z)`` int8 tensor with values -1 / 0 / 1.
    """
    res = cfg.resolution
    step = res * 0.5  # sampling step ≤ resolution → no skipped voxels

    # ---- grid shape ----
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range

    nx = int(round((x_max - x_min) / res)) + 1
    ny = int(round((y_max - y_min) / res)) + 1
    nz = int(round((z_max - z_min) / res)) + 1
    n_voxels = nx * ny * nz

    # ---- output grid (start all unknown) ----
    grid = torch.full((num_envs, nx, ny, nz), fill_value=-1, dtype=torch.int8, device=device)
    # flat view for scatter writes — shares storage with grid
    grid_flat = grid.view(num_envs, n_voxels)  # (E, V)

    # ---- clamp dist to max_distance for traversal ----
    dist_clamped = dist_pre_clamp.clamp(max=cfg.max_distance)  # (E, R)

    # ---- env-independent geometry (computed once) ----
    n_steps = int(cfg.max_distance / step) + 1
    t_vals = torch.arange(n_steps, dtype=torch.float32, device=device) * step  # (S,)

    # Sample points: (S, R, 3)
    pts = t_vals[:, None, None] * ray_dirs[None, :, :]  # (S, R, 3)

    # Voxel indices for all sample points: (S, R)
    ix = ((pts[..., 0] - x_min) / res).round().long()
    iy = ((pts[..., 1] - y_min) / res).round().long()
    iz = ((pts[..., 2] - z_min) / res).round().long()

    # In-bounds mask (env-independent): (S, R)
    in_bounds = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)

    # Flat voxel index C-order: flat = ix*(ny*nz) + iy*nz + iz  (S, R)
    # .clamp so OOB indices never corrupt memory; OOB entries are filtered by in_bounds below.
    flat_idx_safe = (ix * (ny * nz) + iy * nz + iz).clamp(0, n_voxels - 1)  # (S, R)

    # ---- free-marking pass: O(n_steps × n_chunks) Python iterations ----
    # For step s, ray r: voxel at flat_idx_safe[s,r] is free for env e iff
    #   t_vals[s] < dist_clamped[e, r]  (ray hasn't hit yet)  AND  in_bounds[s, r]
    # Chunking the env dimension (chunk_size envs at a time) reduces peak transient
    # from (E, K_r) down to (chunk_size, K_r) — ~4× reduction at chunk_size=1024.
    chunk_size = max(1, cfg.fill_chunk_size)
    for s_idx in range(n_steps):
        t_s = t_vals[s_idx]
        valid_r = in_bounds[s_idx]  # (R,) — env-independent
        if not valid_r.any():
            continue
        r_valid = valid_r.nonzero(as_tuple=True)[0]  # (K_r,) ray indices in-bounds at step s
        v_valid = flat_idx_safe[s_idx, r_valid]       # (K_r,) flat voxel indices

        for e_start in range(0, num_envs, chunk_size):
            e_end = min(e_start + chunk_size, num_envs)
            # Which (env, ray) pairs in this chunk satisfy the free condition? (C, K_r) bool
            free_cond = dist_clamped[e_start:e_end, r_valid] > t_s
            if not free_cond.any():
                continue
            e_local, k_idx = free_cond.nonzero(as_tuple=True)  # each (M,) long
            # Scatter: grid_flat[e, v] = 0 (unknown → free; occupied wins — overwritten below)
            grid_flat[e_start + e_local, v_valid[k_idx]] = 0

    # ---- occupied-marking pass: chunked by env, zero Python loops per chunk ----
    # hit_pts[e, r] = dist_clamped[e, r] * ray_dirs[r] — sensor-local hit position
    # Only valid for (e, r) pairs where is_hit[e, r] is True.
    # Chunking reduces peak from (E, R, 3)+(3×(E,R) int64) down to the chunk equivalent.
    for e_start in range(0, num_envs, chunk_size):
        e_end = min(e_start + chunk_size, num_envs)
        hit_pts_c = dist_clamped[e_start:e_end].unsqueeze(-1) * ray_dirs[None, :, :]  # (C, R, 3)

        hix_c = ((hit_pts_c[..., 0] - x_min) / res).round().long()  # (C, R)
        hiy_c = ((hit_pts_c[..., 1] - y_min) / res).round().long()  # (C, R)
        hiz_c = ((hit_pts_c[..., 2] - z_min) / res).round().long()  # (C, R)

        hib_c = (hix_c >= 0) & (hix_c < nx) & (hiy_c >= 0) & (hiy_c < ny) & (hiz_c >= 0) & (hiz_c < nz)
        valid_hit_c = is_hit[e_start:e_end] & hib_c  # (C, R) — true hit AND inside grid

        occ_flat_c = (hix_c * (ny * nz) + hiy_c * nz + hiz_c).clamp(0, n_voxels - 1)  # (C, R)
        e_occ, r_occ = valid_hit_c.nonzero(as_tuple=True)  # (H_c,) each
        if e_occ.numel() > 0:
            grid_flat[e_start + e_occ, occ_flat_c[e_occ, r_occ]] = 1  # occupied wins all conflicts

    return grid


def voxel_grid_shape(cfg: VoxelOccupancyCfg) -> tuple[int, int, int]:
    """Return (nx, ny, nz) for a given cfg — useful for pre-allocating buffers.

    Args:
        cfg: ``VoxelOccupancyCfg`` instance.

    Returns:
        Tuple ``(nx, ny, nz)`` — number of voxels along each axis.
    """
    res = cfg.resolution
    nx = int(round((cfg.x_range[1] - cfg.x_range[0]) / res)) + 1
    ny = int(round((cfg.y_range[1] - cfg.y_range[0]) / res)) + 1
    nz = int(round((cfg.z_range[1] - cfg.z_range[0]) / res)) + 1
    return nx, ny, nz

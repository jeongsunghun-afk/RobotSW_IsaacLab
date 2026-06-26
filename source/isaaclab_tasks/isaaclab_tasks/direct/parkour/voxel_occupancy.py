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

import torch

from isaaclab.utils import configclass


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


def build_ray_dirs(cfg_clearance: "Clearance3DPatternCfg", device: str) -> torch.Tensor:  # noqa: F821
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


def fill_voxel_grid(
    ray_dirs: torch.Tensor,
    dist_pre_clamp: torch.Tensor,
    is_hit: torch.Tensor,
    cfg: VoxelOccupancyCfg,
    num_envs: int,
    device: str,
) -> torch.Tensor:
    """Fill a (num_envs, X, Y, Z) int8 voxel occupancy grid.

    Algorithm (fixed-step sampling, fully vectorised):

    1. For each ray sample points at ``t = 0, step, 2*step, ...`` up to
       ``min(dist_pre_clamp, max_distance)``.
    2. Points that land inside the grid → mark free (0) — unless already
       occupied (occupied wins).
    3. The hit voxel (point at ``t = dist_pre_clamp``) for true hits only →
       mark occupied (1), overwriting free.
    4. All other voxels stay unknown (-1).

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
    step = res * 0.5  # sampling step ≤ resolution guarantees no skipped voxels

    # ---- grid shape ----
    x_min, x_max = cfg.x_range
    y_min, y_max = cfg.y_range
    z_min, z_max = cfg.z_range

    nx = int(round((x_max - x_min) / res)) + 1
    ny = int(round((y_max - y_min) / res)) + 1
    nz = int(round((z_max - z_min) / res)) + 1

    # ---- output grid (start all unknown) ----
    grid = torch.full((num_envs, nx, ny, nz), fill_value=-1, dtype=torch.int8, device=device)

    n_rays = ray_dirs.shape[0]  # e.g. 294

    # ---- clamp dist to max_distance for traversal ----
    dist_clamped = dist_pre_clamp.clamp(max=cfg.max_distance)  # (E, R)

    # Number of samples per ray (uniform across envs — use max dist for step count)
    n_steps = int(cfg.max_distance / step) + 1

    # t values: (n_steps,)
    t_vals = torch.arange(n_steps, dtype=torch.float32, device=device) * step  # (S,)

    # Sample points: (S, R, 3) broadcast with (R, 3)
    # pts[s, r] = t_vals[s] * ray_dirs[r]  — sensor-local frame, origin = (0,0,0)
    pts = t_vals[:, None, None] * ray_dirs[None, :, :]  # (S, R, 3)

    # Voxel indices for all sample points
    ix = ((pts[..., 0] - x_min) / res).round().long()  # (S, R)
    iy = ((pts[..., 1] - y_min) / res).round().long()  # (S, R)
    iz = ((pts[..., 2] - z_min) / res).round().long()  # (S, R)

    # In-bounds mask for spatial dimensions: (S, R)
    in_bounds = (
        (ix >= 0) & (ix < nx)
        & (iy >= 0) & (iy < ny)
        & (iz >= 0) & (iz < nz)
    )

    # ---- free-marking pass ----
    # For each env, mark voxels free where:
    #   - the sample point is inside the grid
    #   - the sample t < dist_clamped for that (env, ray)  → ray hasn't hit yet
    # We iterate over envs using scatter, fully vectorised over (S, R).

    # Flat voxel index: (S, R)
    flat_idx = ix * (ny * nz) + iy * nz + iz  # (S, R)
    flat_idx_safe = flat_idx.clamp(0, nx * ny * nz - 1)  # safe for OOB indices (masked below)

    for e in range(num_envs):
        # dist_e: (R,) — travel distance for this env
        dist_e = dist_clamped[e]  # (R,)

        # t_valid[s, r] = True if sample s is strictly before the hit on ray r
        # and the point is inside the grid
        t_valid = (t_vals[:, None] < dist_e[None, :]) & in_bounds  # (S, R)

        free_flat = flat_idx_safe[t_valid]  # (K,)
        grid_e_flat = grid[e].view(-1)  # view into grid[e] — shape (nx*ny*nz,)
        # Only mark if currently unknown (-1); occupied stays occupied
        is_unknown = grid_e_flat[free_flat] == -1
        grid_e_flat[free_flat[is_unknown]] = 0

        # ---- occupied-marking pass (hits only) ----
        # hit point: t = dist_e for rays that actually hit
        hit_mask = is_hit[e]  # (R,) bool
        if hit_mask.any():
            hit_pts = dist_e[hit_mask, None] * ray_dirs[hit_mask]  # (H, 3)
            hix = ((hit_pts[:, 0] - x_min) / res).round().long()
            hiy = ((hit_pts[:, 1] - y_min) / res).round().long()
            hiz = ((hit_pts[:, 2] - z_min) / res).round().long()
            hib = (
                (hix >= 0) & (hix < nx)
                & (hiy >= 0) & (hiy < ny)
                & (hiz >= 0) & (hiz < nz)
            )
            if hib.any():
                hix_v = hix[hib]
                hiy_v = hiy[hib]
                hiz_v = hiz[hib]
                grid[e, hix_v, hiy_v, hiz_v] = 1  # occupied wins all conflicts

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

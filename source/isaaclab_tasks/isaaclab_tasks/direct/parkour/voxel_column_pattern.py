# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Per-column ray pattern for a ground-truth voxel occupancy grid.

Why this exists
---------------
The clearance-scanner fill marks a voxel occupied only where a ray's *hit point* lands
(``voxel_occupancy.fill_voxel_grid``: ``grid_flat[e, occ_flat] = 1``, one cell per ray). With 294
diverging rays that yields roughly 70-80 occupied cells out of 7371, so:

* **False negatives** — solid geometry no ray happened to strike reads ``0``, which after
  binarisation is indistinguishable from open air.
* **Surface, not volume** — the space *below* a ground surface is never marked, so a gap in the
  floor and a solid floor differ by only a thin shell of cells.
* **Pose-dependent flicker** — which cells are marked shifts as the hit points move between
  neighbouring cells while the robot walks, even over static terrain.

This pattern replaces sampling with enumeration: one ray per grid *column*, so no column can be
missed, and the fill becomes a comparison against a per-column surface height rather than a
scatter of hit points.

Geometry
--------
Three rays per column, all purely vertical:

* first ``n_col`` rays start at the sensor origin and point **down** → the surface the robot
  stands above,
* next ``n_col`` start at the origin and point **up** → the first surface overhead,
* last ``n_col`` start at ``top_start_z`` above the origin and point **down** → the column's
  *topmost* surface.

Column ``c`` maps to grid indices as ``c = ix * ny + iy`` — x outer, y inner — matching the
``ix * (ny * nz) + iy * nz + iz`` flattening used by the occupancy grid.

Why the third ray exists
------------------------
The first two rays alone cannot tell a ceiling from a pillar. The upward ray is *assumed* to hit
a ceiling, but when the column's origin lies inside solid geometry — the robot walking beside a
step block or a stair riser taller than the sensor mount — the ray instead exits through that
solid's own **top** face, and the two-interval fill then marks the entire volume above it as
occupied. Auditing the shipped 2-ray build showed that on pinned stair this phantom roof covered
**236 of 567 columns on average** (more than the 212 genuine ceiling columns crawl produces), and
step and gap failed too, while flat and hurdle — the terrains with no geometry above the mount —
were clean.

The top-down ray resolves the ambiguity by geometry alone, with no dependence on hit normals:

* **pillar** — the upward hit and the topmost surface are the *same face*, so they agree to within
  float error. The column is solid up to that face and has no ceiling.
* **genuine ceiling** — the topmost surface lies strictly above the upward hit. In
  ``parkour_crawl_terrain`` the ceiling slab (``ceiling_thickness`` 0.10 m) carries a top-blocker
  (``ceiling_top_extra`` 0.50 m) of identical footprint, so every column under the ceiling sees a
  separation of **0.60 m** — set by construction, not by difficulty.

That two-order-of-magnitude gap between "same face" and "0.60 m apart" is what makes
``fill_voxel_grid_gt``'s ``pillar_eps`` safe to pick.

Remaining assumption
--------------------
Each column is still modelled as at most two solid intervals, so a slab with free space both above
*and* below it cannot be expressed. Every generator in ``parkour_terrains.py`` stays inside that
model: all are solid from the ground up, and crawl's overhang is handled by the pillar test above.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils.configclass import configclass


def voxel_column_pattern(cfg: VoxelColumnPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a downward, an upward and a top-down ray per voxel-grid column.

    Args:
        cfg: Pattern configuration; its ranges and resolution must match the voxel grid's.
        device: PyTorch device string.

    Returns:
        Tuple of ``ray_starts`` and ``ray_directions``, both ``(3 * n_col, 3)``. The three
        ``n_col``-sized blocks are, in order, origin-down, origin-up and top-down, all in the same
        column order.
    """
    xs = torch.arange(cfg.nx, device=device, dtype=torch.float32) * cfg.resolution + cfg.x_range[0]
    ys = torch.arange(cfg.ny, device=device, dtype=torch.float32) * cfg.resolution + cfg.y_range[0]
    # x outer, y inner → column index c = ix * ny + iy.
    grid_x = xs.repeat_interleave(cfg.ny)  # (n_col,)
    grid_y = ys.repeat(cfg.nx)  # (n_col,)
    starts = torch.stack([grid_x, grid_y, torch.zeros_like(grid_x)], dim=-1)  # (n_col, 3)
    starts_top = starts.clone()
    starts_top[:, 2] = cfg.top_start_z

    n_col = starts.shape[0]
    down = torch.zeros(n_col, 3, device=device)
    down[:, 2] = -1.0
    up = torch.zeros(n_col, 3, device=device)
    up[:, 2] = 1.0

    return (
        torch.cat([starts, starts, starts_top], dim=0),
        torch.cat([down, up, down], dim=0),
    )


@configclass
class VoxelColumnPatternCfg(PatternBaseCfg):
    """Configuration for :func:`voxel_column_pattern`.

    The ranges and resolution must equal the paired
    :class:`~isaaclab_tasks.direct.parkour.voxel_occupancy.VoxelOccupancyCfg`'s, otherwise columns
    do not line up with grid cells and the fill silently shifts geometry.
    """

    func: Callable = voxel_column_pattern

    x_range: tuple[float, float] = (-0.6, 2.0)
    """Grid x bounds in the sensor-local yaw-aligned frame [m]. Forward is positive."""

    y_range: tuple[float, float] = (-1.0, 1.0)
    """Grid y bounds in the sensor-local yaw-aligned frame [m]. Left is positive."""

    resolution: float = 0.1
    """Column spacing [m]; equals the voxel side length."""

    top_start_z: float = 3.0
    """Height [m] above the sensor origin the top-down ray starts from.

    Must clear the tallest geometry a column can contain, or the ray starts *inside* it and reports
    that solid's underside as the topmost surface. The paired ``RayCasterCfg.max_distance`` has to
    exceed this plus the deepest ground the column may reach, otherwise the ray stops short and the
    miss reads as "no surface".
    """

    @property
    def nx(self) -> int:
        """Number of columns along x."""
        return int(round((self.x_range[1] - self.x_range[0]) / self.resolution)) + 1

    @property
    def ny(self) -> int:
        """Number of columns along y."""
        return int(round((self.y_range[1] - self.y_range[0]) / self.resolution)) + 1

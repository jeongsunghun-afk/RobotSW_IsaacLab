# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the ground-truth voxel column pattern and its volume fill.

The cases mirror the terrain shapes the pattern has to survive: open ground, a hole in the floor,
a crawl tunnel's ceiling, and a solid taller than the sensor mount standing beside the robot. That
last one is the case the two-ray build got wrong — it roofed the column with phantom occupancy.
"""

import pytest
import torch

from isaaclab_tasks.direct.parkour.voxel_column_pattern import VoxelColumnPatternCfg, voxel_column_pattern
from isaaclab_tasks.direct.parkour.voxel_occupancy import VoxelOccupancyCfg, fill_voxel_grid_gt, voxel_grid_shape

INF = float("inf")


@pytest.fixture
def cfg() -> VoxelOccupancyCfg:
    return VoxelOccupancyCfg()


def _fill(
    cfg: VoxelOccupancyCfg,
    down: float,
    up: float,
    top: float | None,
    down_nz: float | None = None,
    up_nz: float | None = None,
):
    """Fill a single-env, single-column grid from scalar surface readings.

    ``down_nz``/``up_nz`` are the hit normals' z components. Both must be given to exercise the
    normal-based rule; omitting them falls back to the ``top_z``-only rule the env used before
    normals were available.
    """
    nx, ny, _ = voxel_grid_shape(cfg)
    n_col = nx * ny
    d = torch.full((1, n_col), down)
    u = torch.full((1, n_col), up)
    t = None if top is None else torch.full((1, n_col), top)
    dn = None if down_nz is None else torch.full((1, n_col), down_nz)
    un = None if up_nz is None else torch.full((1, n_col), up_nz)
    grid, diag = fill_voxel_grid_gt(d, u, cfg, top_z=t, down_normal_z=dn, up_normal_z=un)
    return grid, diag["n_pillar"]


def _z_levels(cfg: VoxelOccupancyCfg) -> torch.Tensor:
    _, _, nz = voxel_grid_shape(cfg)
    return torch.arange(nz, dtype=torch.float32) * cfg.resolution + cfg.z_range[0]


# --------------------------------------------------------------------------------------------
# pattern
# --------------------------------------------------------------------------------------------


def test_pattern_emits_three_ray_sets_per_column():
    pc = VoxelColumnPatternCfg()
    starts, dirs = voxel_column_pattern(pc, device="cpu")
    n_col = pc.nx * pc.ny
    assert starts.shape == (3 * n_col, 3)
    assert dirs.shape == (3 * n_col, 3)


def test_pattern_directions_are_down_up_down():
    pc = VoxelColumnPatternCfg()
    _, dirs = voxel_column_pattern(pc, device="cpu")
    n = pc.nx * pc.ny
    assert torch.all(dirs[:n, 2] == -1.0)
    assert torch.all(dirs[n : 2 * n, 2] == 1.0)
    assert torch.all(dirs[2 * n :, 2] == -1.0)


def test_top_ray_starts_above_origin_over_the_same_columns():
    pc = VoxelColumnPatternCfg()
    starts, _ = voxel_column_pattern(pc, device="cpu")
    n = pc.nx * pc.ny
    assert torch.all(starts[:n, 2] == 0.0)
    assert torch.all(starts[n : 2 * n, 2] == 0.0)
    assert torch.all(starts[2 * n :, 2] == pc.top_start_z)
    # xy of the top-down set must match the origin sets, or it samples a different column.
    torch.testing.assert_close(starts[2 * n :, :2], starts[:n, :2])


def test_column_index_is_x_outer_y_inner():
    pc = VoxelColumnPatternCfg()
    starts, _ = voxel_column_pattern(pc, device="cpu")
    n = pc.nx * pc.ny
    xy = starts[:n, :2]
    for ix in (0, 1, pc.nx - 1):
        for iy in (0, 1, pc.ny - 1):
            c = ix * pc.ny + iy
            assert xy[c, 0] == pytest.approx(pc.x_range[0] + ix * pc.resolution)
            assert xy[c, 1] == pytest.approx(pc.y_range[0] + iy * pc.resolution)


def test_pattern_and_grid_column_counts_agree():
    pc, vc = VoxelColumnPatternCfg(), VoxelOccupancyCfg()
    nx, ny, _ = voxel_grid_shape(vc)
    assert (pc.nx, pc.ny) == (nx, ny)


# --------------------------------------------------------------------------------------------
# fill — open terrain
# --------------------------------------------------------------------------------------------


def test_flat_ground_fills_volume_not_a_shell(cfg):
    grid, _ = _fill(cfg, down=0.32, up=INF, top=-0.32)
    z = _z_levels(cfg)
    col = grid[0, 0, 0]
    assert torch.all(col[z <= -0.32] == 1), "everything below the surface must be solid"
    assert torch.all(col[z > -0.32] == 0)
    assert int(col.sum()) > 1, "a volume fill, not a single surface cell"


def test_hole_in_the_floor_empties_the_column(cfg):
    grid, _ = _fill(cfg, down=INF, up=INF, top=-INF)
    assert int(grid.sum()) == 0


def test_ground_below_the_volume_leaves_the_column_empty(cfg):
    # A gap deep enough that its floor is outside the grid must not fill the bottom layer.
    grid, _ = _fill(cfg, down=5.0, up=INF, top=-5.0)
    assert int(grid.sum()) == 0


# --------------------------------------------------------------------------------------------
# fill — ceiling vs pillar, the case the 2-ray build got wrong
# --------------------------------------------------------------------------------------------


def test_genuine_ceiling_fills_above_it(cfg):
    # Crawl: floor 0.32 m below, ceiling 0.20 m above, its top-blocker 0.60 m higher still.
    grid, n_pillar = _fill(cfg, down=0.32, up=0.20, top=0.80)
    z = _z_levels(cfg)
    col = grid[0, 0, 0]
    assert n_pillar == 0, "a slab 0.60 m thick is not the same face"
    assert torch.all(col[z >= 0.20] == 1)
    assert torch.all(col[(z > -0.32) & (z < 0.20)] == 0), "the tunnel itself must stay free"


def test_pillar_is_not_roofed(cfg):
    # Origin inside a block whose top is 0.25 m up: the upward ray exits through the same face the
    # top-down ray enters, so nothing above it may be marked.
    grid, n_pillar = _fill(cfg, down=0.32, up=0.25, top=0.25)
    z = _z_levels(cfg)
    col = grid[0, 0, 0]
    assert n_pillar == grid.shape[1] * grid.shape[2]
    assert torch.all(col[z > 0.25] == 0), "phantom ceiling above a pillar"
    assert torch.all(col[z <= 0.25] == 1), "the pillar itself must be solid up to its top face"


def test_pillar_and_ceiling_differ_only_by_the_top_reading(cfg):
    """Same down/up readings, different top → opposite classifications. Isolates the new signal.

    The two are not ordered by cell count — a pillar is solid from the grid floor to its top face
    and can occupy *more* cells than a roofed column whose tunnel stays free. What separates them
    is where the solid sits relative to the upward hit.
    """
    roofed = _fill(cfg, down=0.32, up=0.25, top=0.85)[0][0, 0, 0]
    pillared = _fill(cfg, down=0.32, up=0.25, top=0.25)[0][0, 0, 0]
    z = _z_levels(cfg)
    above, between = z > 0.25, (z > -0.32) & (z < 0.25)
    assert torch.all(roofed[above] == 1) and torch.all(pillared[above] == 0)
    assert torch.all(roofed[between] == 0) and torch.all(pillared[between] == 1)


def test_two_ray_call_reproduces_the_phantom_roof(cfg):
    """Omitting ``top_z`` must keep the old behaviour, so the regression is attributable."""
    grid, n_pillar = _fill(cfg, down=0.32, up=0.25, top=None)
    z = _z_levels(cfg)
    assert n_pillar == 0
    assert torch.all(grid[0, 0, 0][z >= 0.25] == 1)


def test_pillar_test_tolerates_float_error_but_not_a_thin_slab(cfg):
    _, n_close = _fill(cfg, down=0.32, up=0.25, top=0.25 + 1e-6)
    assert n_close > 0, "the same face read twice must classify as a pillar"
    # crawl's minimum separation is ceiling_thickness = 0.10 m even with no top-blocker.
    _, n_thin = _fill(cfg, down=0.32, up=0.25, top=0.35)
    assert n_thin == 0, "a 0.10 m slab is a ceiling, not a pillar"


def test_top_ray_miss_does_not_roof_the_column(cfg):
    """A top-down miss arrives as -inf and must mark nothing, never as an overhead surface."""
    grid, n_pillar = _fill(cfg, down=0.32, up=INF, top=-INF)
    z = _z_levels(cfg)
    assert n_pillar == 0
    assert torch.all(grid[0, 0, 0][z > -0.32] == 0)


# --------------------------------------------------------------------------------------------
# fill — normals. The origin is free space only for the column under the robot; every other
# column can have geometry at mount height, and then both rays *exit* that solid.
# --------------------------------------------------------------------------------------------


def test_normals_flat_ground_unchanged(cfg):
    # Ray enters the ground from above → normal points up.
    grid, n_pillar = _fill(cfg, down=0.32, up=INF, top=-0.32, down_nz=1.0, up_nz=0.0)
    z = _z_levels(cfg)
    assert n_pillar == 0
    assert torch.all(grid[0, 0, 0][z <= -0.32] == 1)
    assert torch.all(grid[0, 0, 0][z > -0.32] == 0)


def test_normals_ducked_under_ceiling_keeps_corridor_free(cfg):
    # Robot below the slab: down ray enters the floor (nz>0), up ray hits the slab underside (nz<0).
    grid, n_pillar = _fill(cfg, down=0.20, up=0.13, top=0.73, down_nz=1.0, up_nz=-1.0)
    z = _z_levels(cfg)
    col = grid[0, 0, 0]
    assert n_pillar == 0
    assert torch.all(col[z <= -0.20] == 1), "floor"
    assert torch.all(col[(z > -0.20) & (z < 0.13)] == 0), "corridor must stay free"
    assert torch.all(col[z >= 0.13] == 1), "ceiling"


def test_normals_origin_inside_ceiling_slab_does_not_plug_the_corridor(cfg):
    """The bug this rule exists for: standing tall, a forward column's origin is in the slab.

    Both rays exit the slab — down through its underside (nz<0), up through its top (nz>0) — and
    the slab's own top-blocker is the topmost surface. Reading the downward exit as ground filled
    everything beneath it, turning the tunnel into a wall.
    """
    grid, n_pillar = _fill(cfg, down=0.04, up=0.06, top=0.56, down_nz=-1.0, up_nz=1.0)
    z = _z_levels(cfg)
    col = grid[0, 0, 0]
    assert n_pillar == 0, "a slab carrying a blocker is not a pillar"
    assert torch.all(col[z < -0.04] == 0), "corridor below the slab must be free"
    assert torch.all(col[z >= -0.04] == 1), "the slab underside is the ceiling"


def test_two_ray_normalless_call_reproduces_the_plugged_corridor(cfg):
    """Without normals the same readings plug the column, so the regression stays attributable."""
    grid, _ = _fill(cfg, down=0.04, up=0.06, top=0.56)
    z = _z_levels(cfg)
    assert torch.all(grid[0, 0, 0][z <= -0.04] == 1), "old rule fills below the slab underside"


def test_normals_pillar_beside_a_block(cfg):
    # Origin inside a ground-standing block: both rays exit it and its top face is the topmost
    # surface, so the column is solid up to that face and carries no ceiling.
    grid, n_pillar = _fill(cfg, down=0.32, up=0.25, top=0.25, down_nz=-1.0, up_nz=1.0)
    z = _z_levels(cfg)
    col = grid[0, 0, 0]
    assert n_pillar == grid.shape[1] * grid.shape[2]
    assert torch.all(col[z <= 0.25] == 1)
    assert torch.all(col[z > 0.25] == 0)


def test_normals_miss_carries_no_classification(cfg):
    # A miss must have normal 0.0 so it fails both sign tests rather than asserting a surface.
    grid, _ = _fill(cfg, down=INF, up=INF, top=-INF, down_nz=0.0, up_nz=0.0)
    assert int(grid.sum()) == 0


def test_shape_is_unchanged_by_the_third_ray(cfg):
    nx, ny, nz = voxel_grid_shape(cfg)
    grid, _ = _fill(cfg, down=0.32, up=INF, top=-0.32)
    assert grid.shape == (1, nx, ny, nz)
    assert grid.dtype == torch.int8

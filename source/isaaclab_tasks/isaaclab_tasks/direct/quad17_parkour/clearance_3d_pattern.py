# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clearance 3D ray-cast pattern for parkour teacher privileged observations.

This module implements a forward-hemisphere spherical grid pattern for measuring
clearance distances in all relevant directions: down (floor), up (ceiling), and
lateral/forward (walls, obstacles).

Coordinate convention (matches IsaacLab RayCaster):
    x = forward, y = left, z = up

Azimuth (yaw around z): 0 = forward, positive = left.
Elevation (pitch from horizontal plane): positive = up, negative = down.

Ray directions formula:
    x = cos(azimuth) * cos(elevation)   # forward component
    y = sin(azimuth) * cos(elevation)   # left component
    z = sin(elevation)                  # up component  ← +elev → ceiling, -elev → floor

R2 note: azimuth index ordering (left-to-right, outer loop) is preserved here so
symmetry augmentation can apply a sign-flip on y-component + azimuth index reversal.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    pass


def clearance_3d_pattern(cfg: Clearance3DPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a forward-hemisphere spherical grid pattern for 3D clearance sensing.

    Produces a fixed-dimension grid of rays spanning:
        azimuth   : [azimuth_range[0], azimuth_range[1]]  (degrees, num_azimuth steps)
        elevation : [elevation_range[0], elevation_range[1]]  (degrees, num_elevation steps)

    All rays originate at (0, 0, 0) in sensor-local coordinates. The RayCasterCfg
    offset and ray_alignment transform them to world frame at runtime.

    Args:
        cfg: Configuration instance for the pattern.
        device: PyTorch device string.

    Returns:
        ray_starts:     (N, 3) tensor, all zeros (rays start at sensor origin).
        ray_directions: (N, 3) unit-vector tensor, x=forward y=left z=up convention.
                        N = num_azimuth * num_elevation.
    """
    # linspace gives deterministic count regardless of floating-point step precision
    azimuth_deg = torch.linspace(cfg.azimuth_range[0], cfg.azimuth_range[1], cfg.num_azimuth, device=device)
    elevation_deg = torch.linspace(cfg.elevation_range[0], cfg.elevation_range[1], cfg.num_elevation, device=device)

    # Outer loop = azimuth, inner = elevation  → index layout: [az0*ne+el0, az0*ne+el1, ..., az1*ne+el0, ...]
    # R2 symmetry: flip y + reverse az-axis to mirror left/right.
    az_rad = torch.deg2rad(azimuth_deg)  # (num_azimuth,)
    el_rad = torch.deg2rad(elevation_deg)  # (num_elevation,)

    # Meshgrid: az as outer dim, el as inner dim
    az_grid, el_grid = torch.meshgrid(az_rad, el_rad, indexing="ij")  # both (num_azimuth, num_elevation)
    az_flat = az_grid.reshape(-1)  # (N,)
    el_flat = el_grid.reshape(-1)  # (N,)

    cos_az = torch.cos(az_flat)
    sin_az = torch.sin(az_flat)
    cos_el = torch.cos(el_flat)
    sin_el = torch.sin(el_flat)

    # x=forward, y=left, z=up  (same convention as lidar_pattern in IsaacLab core)
    x = cos_az * cos_el  # forward
    y = sin_az * cos_el  # left
    z = sin_el  # up (+elevation → ceiling, -elevation → floor)

    ray_directions = torch.stack([x, y, z], dim=-1)  # (N, 3)
    # Normalize (should be unit vectors already, but numerical safety)
    ray_directions = ray_directions / torch.norm(ray_directions, dim=-1, keepdim=True).clamp(min=1e-8)

    ray_starts = torch.zeros_like(ray_directions)  # (N, 3), all origin
    return ray_starts, ray_directions


@configclass
class Clearance3DPatternCfg(PatternBaseCfg):
    """Configuration for the forward-hemisphere 3D clearance pattern.

    Produces a fixed-dimension spherical grid of rays for measuring clearance
    distances to geometry (floor, ceiling, walls, obstacles).

    Default FOV:
        Azimuth   [-100°, +100°] × 21 steps  ≈ 10° resolution
        Elevation [ -75°,  +60°] × 14 steps  ≈ 10.38° resolution
        Total: 21 × 14 = **294 rays**

    Azimuth convention: 0° = forward (+x), positive = left (+y).
    Elevation convention: 0° = horizontal, positive = up (+z), negative = down (-z).

    Note on R2 symmetry augmentation:
        Left/right mirror = sign-flip on azimuth (negate az indices from center) +
        negate y-component of ray_directions. Index layout is az-outer/el-inner,
        so mirrored ray i corresponds to index (num_azimuth-1-az_idx)*num_elevation + el_idx.
    """

    func: Callable = clearance_3d_pattern

    azimuth_range: tuple[float, float] = (-100.0, 100.0)
    """Azimuth range (degrees). Defaults to [-100, +100] covering forward hemisphere."""

    elevation_range: tuple[float, float] = (-75.0, 60.0)
    """Elevation range (degrees). Defaults to [-75, +60]: down to near-vertical floor, up to above-horizontal ceiling."""

    num_azimuth: int = 21
    """Number of azimuth steps. Default 21 → ~10° resolution over [-100, +100]."""

    num_elevation: int = 14
    """Number of elevation steps. Default 14 → ~10.38° resolution over [-75, +60]."""

    @property
    def num_rays(self) -> int:
        """Total ray count = num_azimuth * num_elevation."""
        return self.num_azimuth * self.num_elevation

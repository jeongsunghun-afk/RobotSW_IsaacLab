# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ray-caster pattern for the Livox Mid-360 LiDAR.

This module implements the Mid-360 beam pattern as a ray-caster pattern usable by IsaacLab's
``RayCaster`` sensor. It follows the core pattern function contract without touching any core
file: we subclass :class:`PatternBaseCfg` from the core and point its ``func`` field at our
custom function.

The Mid-360 uses a non-repetitive scanning pattern. We ship a pre-sampled ``(800000, 2)`` table
of ``(azimuth theta, elevation phi)`` angles (radians) accumulated over a time sequence, and
subsample a fixed beam set from it (single-frame fixed rays).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from .livox_patterns import LivoxPatternCfg

# Default path to the bundled Mid-360 angle table, resolved relative to this module.
DEFAULT_MID360_PATTERN_FILE = os.path.join(os.path.dirname(__file__), "data", "mid360.npy")


def livox_pattern(cfg: LivoxPatternCfg, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Livox Mid-360 LiDAR pattern for ray casting.

    Loads the bundled ``(N, 2)`` table of ``(theta, phi)`` angles (radians) and subsamples
    ``cfg.num_rays`` beams from it. The angles are converted to unit ray directions in the
    sensor's local Z-up frame via the spherical mapping
    ``x = cos(phi) cos(theta), y = cos(phi) sin(theta), z = sin(phi)``.

    Args:
        cfg: The configuration instance for the pattern.
        device: The device to create the pattern on.

    Returns:
        The starting positions (all zeros) and unit directions of the rays. Both have shape
        ``(num_rays, 3)`` and dtype ``float32``.

    Raises:
        FileNotFoundError: If the pattern file does not exist.
        ValueError: If ``cfg.sampling`` is not ``"stride"`` or ``"window"``.
    """
    if not os.path.isfile(cfg.pattern_file):
        raise FileNotFoundError(
            f"Livox Mid-360 pattern file not found: '{cfg.pattern_file}'. Expected a (N, 2) float32 .npy "
            "table of (azimuth theta, elevation phi) angles in radians."
        )

    # load (N, 2) angle table: col0 = theta (azimuth), col1 = phi (elevation)
    angles = np.load(cfg.pattern_file)
    num_available = angles.shape[0]
    num_rays = min(cfg.num_rays, num_available)

    # subsample fixed beam indices
    if cfg.sampling == "stride":
        # even stride across the full table -> maximizes angular coverage
        idx = torch.linspace(0, num_available - 1, num_rays).long()
    elif cfg.sampling == "window":
        # leading contiguous block -> approximates a sparse real single-frame scan
        idx = torch.arange(num_rays)
    else:
        raise ValueError(f"Sampling must be 'stride' or 'window'. Received: '{cfg.sampling}'.")

    angles_t = torch.from_numpy(angles).to(device=device, dtype=torch.float32)
    theta = angles_t[idx, 0]
    phi = angles_t[idx, 1]

    # spherical (Z-up) to Cartesian unit directions
    cos_phi = torch.cos(phi)
    x = cos_phi * torch.cos(theta)
    y = cos_phi * torch.sin(theta)
    z = torch.sin(phi)
    ray_directions = torch.stack([x, y, z], dim=-1).to(device=device, dtype=torch.float32)

    # all rays originate from the sensor origin
    ray_starts = torch.zeros_like(ray_directions)

    return ray_starts, ray_directions


@configclass
class LivoxPatternCfg(PatternBaseCfg):
    """Configuration for the Livox Mid-360 LiDAR pattern for ray-casting."""

    func: Callable = livox_pattern

    pattern_file: str = DEFAULT_MID360_PATTERN_FILE
    """Path to the ``(N, 2)`` float32 ``.npy`` table of ``(theta, phi)`` angles (radians).

    Defaults to the bundled Mid-360 table next to this module (``data/mid360.npy``).
    """

    num_rays: int = 20000
    """Number of rays to subsample from the angle table.

    Defaults to 20000 = Livox Mid-360 per-frame samples
    (OmniPerception genera_lidar_scan_pattern.py reference).
    """

    sampling: Literal["stride", "window"] = "window"
    """Subsampling strategy. Defaults to ``"window"``.

    * ``"window"``: leading contiguous block -> matches OmniPerception single-frame convention
      (20000 rays = one real Mid-360 frame).
    * ``"stride"``: even stride across the full table -> maximum angular coverage.
    """

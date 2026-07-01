# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Data class for the LiDAR sensor."""

from __future__ import annotations

import torch

from isaaclab.sensors.ray_caster.ray_caster_data import RayCasterData
from isaaclab.utils.math import convert_quat


class LidarSensorData(RayCasterData):
    """Data container for the LiDAR sensor.

    This extends :class:`~isaaclab.sensors.ray_caster.ray_caster_data.RayCasterData`
    with LiDAR-specific measurements (``distances`` and ``pointcloud``).

    Convention note (Sim 6.0 migration):
        The native :class:`RayCasterData` returns ``ProxyArray`` wrappers and uses an
        **xyzw** quaternion convention for ``quat_w``. This LiDAR container restores the
        Sim 5.1 consumer contract by exposing ``pos_w`` / ``quat_w`` / ``ray_hits_w`` as
        plain ``torch.Tensor`` views, and ``quat_w`` is converted to the Isaac Lab
        **wxyz** convention (``.data.quat_w`` is documented as ``(N, 4) wxyz``).
    """

    def __init__(self) -> None:
        super().__init__()
        self.distances: torch.Tensor | None = None
        """Distance measurements for each ray in meters. Shape is (num_instances, num_rays)."""
        self.pointcloud: torch.Tensor | None = None
        """Point cloud data in either world or sensor coordinates. Shape is (num_instances, num_rays, 3).

        Only populated when ``return_pointcloud`` is True in the sensor configuration. The coordinate
        frame depends on the ``pointcloud_in_world_frame`` setting.
        """

    @property
    def pos_w(self) -> torch.Tensor:  # type: ignore[override]
        """Position of the sensor origin in world frame [m]. Shape is (N, 3)."""
        return self._pos_w_ta.torch

    @property
    def quat_w(self) -> torch.Tensor:  # type: ignore[override]
        """Orientation of the sensor origin in world frame, quaternion (w, x, y, z). Shape is (N, 4).

        The native :class:`RayCasterData` stores the quaternion as xyzw; it is converted to the
        Isaac Lab wxyz convention here so downstream consumers see ``(N, 4) wxyz``.
        """
        return convert_quat(self._quat_w_ta.torch, to="wxyz")

    @property
    def ray_hits_w(self) -> torch.Tensor:  # type: ignore[override]
        """The ray hit positions in world frame [m]. Shape is (N, B, 3). Misses contain ``inf``."""
        return self._ray_hits_w_ta.torch

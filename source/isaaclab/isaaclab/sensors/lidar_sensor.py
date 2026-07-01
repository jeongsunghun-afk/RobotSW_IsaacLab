# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""LiDAR sensor implementation extending the native ray-caster (Sim 6.0)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import warp as wp

# Subclass the *concrete* single-mesh ray-caster.
#
# The public ``isaaclab.sensors.ray_caster.RayCaster`` is a ``FactoryBase`` whose
# ``__new__`` dispatches to a backend implementation; subclassing it would make
# ``LidarSensor(cfg)`` silently return a plain physx ``RayCaster`` (FactoryBase
# returns ``impl(*args)``, not an instance of the subclass). The actual concrete
# implementation lives in ``isaaclab_physx`` (``_PhysXRayCasterMixin + BaseRayCaster``,
# no FactoryBase in its MRO), so subclassing it gives normal ``object.__new__``
# behaviour and inherits real pose-tracking + ray-casting. This module is lazily
# imported, so backend-only (e.g. newton) setups that never import ``LidarSensor``
# do not pull in physx.
from isaaclab_physx.sensors.ray_caster.ray_caster import RayCaster

from .lidar_sensor_data import LidarSensorData

if TYPE_CHECKING:
    from .lidar_sensor_cfg import LidarSensorCfg


class LidarSensor(RayCaster):
    """A LiDAR sensor implementation based on ray-casting.

    Extends the concrete single-mesh ray-caster with LiDAR-specific post-processing:
    per-ray ``distances`` and an optional ``pointcloud``. Supports Livox scan patterns
    (e.g. Mid-360) via :class:`~isaaclab.sensors.ray_caster.patterns.patterns_cfg.LivoxPatternCfg`.
    """

    cfg: LidarSensorCfg
    """The configuration parameters."""

    def __init__(self, cfg: LidarSensorCfg):
        """Initializes the LiDAR sensor.

        Args:
            cfg: The configuration parameters.
        """
        # Initialize base class (sets a base ``RayCasterData``); replace with LiDAR container.
        super().__init__(cfg)
        self._data = LidarSensorData()

        # LiDAR-specific timing parameters.
        self.update_frequency = cfg.update_frequency
        self.update_dt = 1.0 / self.update_frequency
        self.sensor_t = 0.0

    def __str__(self) -> str:
        """Returns: A string containing information about the instance."""
        return (
            f"LiDAR Sensor @ '{self.cfg.prim_path}': \n"
            f"\tupdate period (s)    : {self.cfg.update_period}\n"
            f"\tupdate frequency (Hz): {self.cfg.update_frequency}\n"
            f"\tnumber of sensors    : {self._view_count}\n"
            f"\tnumber of rays/sensor: {self.num_rays}\n"
            f"\tmax range (m)        : {self.cfg.max_distance}\n"
            f"\tnoise enabled        : {self.cfg.enable_sensor_noise}"
        )

    @property
    def data(self) -> LidarSensorData:
        """The sensor data object."""
        # Update buffers if needed.
        self._update_outdated_buffers()
        return self._data

    """
    Implementation.
    """

    def _initialize_rays_impl(self):
        """Initialize ray patterns and LiDAR-specific buffers."""
        # Base implementation creates ray buffers and the RayCasterData warp buffers.
        super()._initialize_rays_impl()

        # LiDAR-specific torch buffers (full-env sized).
        self._data.distances = torch.zeros(self._view_count, self.num_rays, device=self._device)
        if self.cfg.return_pointcloud:
            self._data.pointcloud = torch.zeros(self._view_count, self.num_rays, 3, device=self._device)

    def _update_buffers_impl(self, env_mask: wp.array):
        """Update sensor buffers with LiDAR-specific post-processing.

        Sim 6.0 changed the buffer-update signature from torch ``env_ids`` to a warp
        ``env_mask`` (see :meth:`SensorBase._update_outdated_buffers`). After delegating
        the masked ray-cast to the base class, ``distances`` / ``pointcloud`` are recomputed
        for **all** environments. The recompute is idempotent (non-masked envs keep their
        previous, mutually-consistent ``ray_hits_w`` and ``ray_starts_w``), which sidesteps
        partial-mask indexing bugs.

        Note (6.0 behaviour change): the Sim 5.1 ``_update_dynamic_rays`` in-place Mid-360
        rotation is removed. Ray directions are now warp ``ProxyArray`` buffers (no per-step
        torch mutation), and the old rotation was already disabled (quadratic-in-time bug).
        The static sampled Mid-360 pattern is used instead.
        """
        # Update sensor time (kept for parity with the 5.1 API).
        self.sensor_t += self.cfg.update_period

        # Base class performs the masked ray-cast and updates pos_w / quat_w / ray_hits_w.
        super()._update_buffers_impl(env_mask)

        # Full-env recompute from warp buffers (read via ``.torch`` views).
        ray_hits = self._data.ray_hits_w  # (N, B, 3) torch view
        ray_starts_w = wp.to_torch(self._ray_starts_w)  # (N, B, 3) per-ray world origin
        distances = torch.norm(ray_hits - ray_starts_w, dim=2)

        # Misses are stored as inf; clamp them to max range.
        inf_mask = torch.isinf(ray_hits).any(dim=2)
        distances = torch.where(inf_mask, torch.full_like(distances, self.cfg.max_distance), distances)

        # Apply noise if enabled.
        if self.cfg.enable_sensor_noise:
            distances = self._apply_noise(distances)

        self._data.distances = distances

        # Generate point cloud if requested.
        if self.cfg.return_pointcloud:
            self._generate_pointcloud(distances, inf_mask)

    def _apply_noise(self, distances: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian range noise + dropout to distance measurements."""
        noise = torch.randn_like(distances) * self.cfg.random_distance_noise
        distances_noisy = distances + noise

        # Dropout: rays return max range.
        dropout_mask = torch.rand_like(distances) < self.cfg.pixel_dropout_prob
        distances_noisy[dropout_mask] = self.cfg.max_distance

        # Clamp to valid range.
        return torch.clamp(distances_noisy, self.cfg.min_range, self.cfg.max_distance)

    def _generate_pointcloud(self, distances: torch.Tensor, inf_mask: torch.Tensor) -> None:
        """Generate point cloud from ray hits (world or sensor frame)."""
        if self.cfg.pointcloud_in_world_frame:
            # World-frame hits directly.
            self._data.pointcloud = self._data.ray_hits_w.clone()
        else:
            # Sensor-frame: local ray directions scaled by range.
            ray_directions_sensor = self.ray_directions.torch  # (N, B, 3)
            pointcloud_sensor = ray_directions_sensor * distances.unsqueeze(-1)
            pointcloud_sensor[inf_mask] = float("inf")
            self._data.pointcloud = pointcloud_sensor

    """
    Convenience accessors.
    """

    def get_distances(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get distance measurements (num_envs, num_rays). If ``env_ids`` is None, returns all."""
        if env_ids is None:
            return self.data.distances
        return self.data.distances[env_ids]

    def get_pointcloud(self, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        """Get point cloud data (num_envs, num_rays, 3). If ``env_ids`` is None, returns all."""
        if not self.cfg.return_pointcloud:
            raise ValueError("Point cloud generation is disabled. Set 'return_pointcloud=True' in config.")
        if env_ids is None:
            return self.data.pointcloud
        return self.data.pointcloud[env_ids]

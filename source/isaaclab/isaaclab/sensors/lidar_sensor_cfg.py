# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the LiDAR sensor."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from isaaclab.sensors.ray_caster.ray_caster_cfg import RayCasterCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .lidar_sensor import LidarSensor


@configclass
class LidarSensorCfg(RayCasterCfg):
    """Configuration for the LiDAR sensor.

    Extends the native :class:`~isaaclab.sensors.ray_caster.ray_caster_cfg.RayCasterCfg`
    with LiDAR-specific parameters (range limit, noise simulation, point cloud generation).

    Note (Sim 6.0 migration):
        The Sim 5.1 fork-only field ``dynamic_env_mesh_prim_paths`` (multi-mesh
        self-occlusion) is intentionally **dropped** — the native ray-caster is single-mesh.
        Multi-mesh self-occlusion is deferred.
    """

    class_type: type[LidarSensor] | str = "{DIR}.lidar_sensor:LidarSensor"

    ray_alignment: Literal["base", "yaw", "world"] = "yaw"

    # LiDAR-specific timing parameters.
    update_frequency: float = 50.0
    """LiDAR update frequency in Hz. Defaults to 50.0 Hz."""

    # Range settings.
    min_range: float = 0.2
    """Minimum sensing range in meters. Defaults to 0.2m."""

    # Output settings.
    return_pointcloud: bool = True
    """Whether to generate point cloud data. Defaults to True."""

    pointcloud_in_world_frame: bool = False
    """Whether to return point cloud in world frame or sensor frame. Defaults to False (sensor frame)."""

    # Noise settings.
    enable_sensor_noise: bool = False
    """Whether to enable sensor noise simulation. Defaults to False."""

    random_distance_noise: float = 0.03
    """Standard deviation of Gaussian noise added to distances. Defaults to 0.03m."""

    random_angle_noise: float = 0.15 * 3.14159 / 180
    """Standard deviation of angular noise in radians. Defaults to 0.15 degrees.

    Note: declared for cfg compatibility but not applied by the current ``_apply_noise`` implementation.
    """

    pixel_dropout_prob: float = 0.01
    """Probability of pixel dropout (no return). Defaults to 0.01."""

    pixel_std_dev_multiplier: float = 0.01
    """Multiplier for pixel-wise noise standard deviation. Defaults to 0.01.

    Note: declared for cfg compatibility but not applied by the current ``_apply_noise`` implementation.
    """

    # Data normalization.
    normalize_range: bool = False
    """Whether to normalize range values. Defaults to False."""

    far_out_of_range_value: float = -1.0
    """Value to assign to far out-of-range measurements. Defaults to -1.0."""

    near_out_of_range_value: float = -1.0
    """Value to assign to near out-of-range measurements. Defaults to -1.0."""

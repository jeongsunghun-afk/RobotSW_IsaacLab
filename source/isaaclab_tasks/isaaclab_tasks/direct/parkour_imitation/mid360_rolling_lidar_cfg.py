# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for :class:`~isaaclab_tasks.direct.parkour_imitation.mid360_rolling_lidar.Mid360RollingLidarSensor`.

Kept in its own module, like the core ``lidar_sensor_cfg.py`` / ``lidar_sensor.py`` pair, because
``train.py`` imports the env cfg module **before** the simulation app is launched.  The sensor class
imports ``isaaclab_physx`` (``omni.physics``), which only exists after launch, so the cfg must not
import the class.  ``class_type`` is therefore the same ``"module:Class"`` string form the core
cfg uses; the env resolves it lazily in ``_setup_scene``.
"""

from __future__ import annotations

from isaaclab.sensors.lidar_sensor_cfg import LidarSensorCfg
from isaaclab.utils.configclass import configclass


@configclass
class Mid360RollingLidarSensorCfg(LidarSensorCfg):
    """Configuration for :class:`Mid360RollingLidarSensor`.

    Every field inherited from :class:`~isaaclab.sensors.lidar_sensor_cfg.LidarSensorCfg` keeps
    its meaning. ``pattern_cfg.samples`` sets the rays per frame (``R``) and therefore how far
    the window advances per update.
    """

    class_type: str = "isaaclab_tasks.direct.parkour_imitation.mid360_rolling_lidar:Mid360RollingLidarSensor"
    """Resolved lazily by the env (``module:Class``), see the module docstring."""

    rolling_scan: bool = True
    """Advance the scan window before every cast. ``False`` reproduces the static core sensor."""

    random_scan_phase: bool = True
    """Draw each environment's starting row uniformly at init and on every reset.

    ``False`` starts every environment at row 0, which makes a run reproducible but also makes
    the beam pattern a deterministic function of the step index.
    """

    mount_pitch_range_deg: tuple[float, float] | None = None
    """Per-env mount-pitch domain randomisation range ``(lo, hi)`` [deg], or ``None``.

    When set, every environment draws its own boom pitch uniformly in ``[lo, hi]`` at
    initialisation and again whenever it is reset, instead of sharing the single rotation baked
    into ``offset.rot``. The drawn value is readable as
    :attr:`~isaaclab_tasks.direct.parkour_imitation.mid360_rolling_lidar.Mid360RollingLidarSensor.mount_pitch_deg`.

    ``None`` keeps the single-rotation path bit-identical: the pattern stays pre-rotated by
    ``offset.rot`` and the per-env quaternion is not applied at all.

    Requires ``rolling_scan``, since the rotation is applied while the scan window is written.
    """

    scan_pattern_file: str | None = None
    """Absolute path to the ``(P, 2)`` ``[theta, phi]`` pattern ``.npy``.

    ``None`` uses the packaged ``mid360.npy`` that ``livox_pattern`` itself reads, so the rolling
    sensor and the static one draw from the same pattern.
    """

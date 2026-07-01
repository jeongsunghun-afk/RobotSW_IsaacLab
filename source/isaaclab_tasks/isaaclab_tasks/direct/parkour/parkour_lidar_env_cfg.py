# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour + Livox Mid-360 LiDAR inspection environment configuration (Sim 6.0).

Uses the core ``LidarSensorCfg`` with the core ``LivoxPatternCfg(sensor_type="mid360")``.

The existing height_scanner and ALL policy observations are UNCHANGED — the trained
parkour checkpoint loads without modification.

Sim 6.0 migration notes:
    * ``from isaaclab.utils import configclass`` -> ``from isaaclab.utils.configclass import configclass``.
    * The fork-only field ``dynamic_env_mesh_prim_paths`` (multi-mesh self-occlusion) is removed —
      the native single-mesh ray-caster does not support it (multi-mesh self-occlusion deferred).
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.lidar_sensor_cfg import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LivoxPatternCfg
from isaaclab.utils.configclass import configclass

from .parkour_env_cfg import ParkourEnvCfg


@configclass
class ParkourLidarEnvCfg(ParkourEnvCfg):
    """Go2 Parkour env cfg with an additional Mid-360 LiDAR (inspection-only).

    All policy-facing fields (observation_space, height_scanner, obs_groups, etc.)
    are inherited UNCHANGED from ParkourEnvCfg.  Only ``mid360_lidar`` is added.

    Uses core ``LidarSensorCfg`` with ``LivoxPatternCfg(sensor_type="mid360")``.
    """

    # Mid-360 LiDAR — inspection-only, NEVER concatenated into observations.
    # Mount: SLAM_local_window/mount_transform.py (runtime-faithful values).
    # mesh_prim_paths = terrain ground mesh, same as height_scanner.
    mid360_lidar: LidarSensorCfg = LidarSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            # SLAM_local_window mount_transform.py: MOUNT_OFFSET_BASE_LINK_M
            # = _stl_to_go2((-0.000485, -0.333644, +0.050079)) = (+0.333644, -0.000485, +0.050079)
            # Mid-360 mounted on a forward boom from the Go2 head (~14 cm ahead of body).
            pos=(0.333644, -0.000485, 0.050079),
            # SLAM_local_window mount_transform.py: MOUNT_ROT_WXYZ
            # = (0.0, cos(15°), 0.0, sin(15°)) ≈ (0, 0.96593, 0, 0.25882)
            # dome-down 180° flip ⊗ pitch-up 30° (w=0 -> flip included).
            rot=(0.0, 0.96593, 0.0, 0.25882),
        ),
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=24000,  # SLAM publisher default=20000; keeping 24000 (pending user decision)
            use_simple_grid=False,
            downsample=1,
        ),
        ray_alignment="base",
        mesh_prim_paths=["/World/ground"],
        max_distance=40.0,
        min_range=0.2,
        debug_vis=True,  # GUI point-cloud overlay in viewport (no perf cost on play-only tasks)
        return_pointcloud=False,  # not needed; use ray_hits_w directly
        pointcloud_in_world_frame=False,
        update_frequency=10.0,  # 10 Hz is sufficient for inspection
        # ── Mid-360 noise fidelity ──────────────────────────────────────────────
        # distance += randn() * random_distance_noise  (absolute Gaussian, σ in metres)
        # ray dropped (distance=max_distance) when rand() < pixel_dropout_prob  ([0,1] prob)
        # random_angle_noise / pixel_std_dev_multiplier are declared but NOT implemented.
        enable_sensor_noise=True,
        random_distance_noise=0.02,  # σ=2 cm — Mid-360 range accuracy (1σ, absolute m)
        pixel_dropout_prob=0.10,  # 10% ray dropout — OmniPerception DR baseline
        random_angle_noise=0.0,  # declared but not implemented; 0 = no silent effect
        pixel_std_dev_multiplier=0.0,  # declared but not implemented; 0 = no silent effect
    )

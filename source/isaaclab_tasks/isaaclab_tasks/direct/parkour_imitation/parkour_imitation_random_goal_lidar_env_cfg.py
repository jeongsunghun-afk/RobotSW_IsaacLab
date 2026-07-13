# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Go2 ParkourImitation-Symmetry-RandomGoal + Mid-360 LiDAR environment.

Policy obs (policy/scan/priv_explicit/priv_latent/history) and amp_obs are UNCHANGED
relative to ParkourImitationRandomGoalEnvCfg.  The Mid-360 LiDAR is attached as an
additional sensor but is NEVER concatenated into any policy obs group.  Its range-image
projection is exposed only as ``obs["lidar"]`` (R2 side-channel for the SL runner arm).

height_scan (obs["scan"]) is kept intact as the critic supervision target.

Sim 6.0 migration notes
-----------------------
* Uses the core ``LidarSensorCfg`` (single-mesh native ray-caster) with the core
  ``LivoxPatternCfg(sensor_type="mid360")``.
* The Sim 5.1 fork-only field ``dynamic_env_mesh_prim_paths`` (multi-mesh body
  self-occlusion) is dropped — the native ray-caster is single-mesh.  Body
  self-occlusion is handled at the env level via the precomputed static az/el grid
  (``body_occ_azel_grid.npy``); see ``parkour_imitation_random_goal_lidar_env.py``.
* Mount ``offset`` (pos/rot) matches the already-ported 6.0 ``parkour_lidar_env_cfg.py``
  reference for the identical Mid-360 boom mount.
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.lidar_sensor_cfg import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LivoxPatternCfg
from isaaclab.utils.configclass import configclass

from .parkour_imitation_random_goal_env_cfg import ParkourImitationRandomGoalEnvCfg


@configclass
class ParkourImitationRandomGoalLidarEnvCfg(ParkourImitationRandomGoalEnvCfg):
    """Go2 ParkourImitation-RandomGoal cfg with an additional Livox Mid-360 LiDAR.

    All policy-facing fields (observation_space, height_scanner, obs_groups, amp_obs, etc.)
    are inherited UNCHANGED from ParkourImitationRandomGoalEnvCfg.  Only ``mid360_lidar``
    and the range-image projection parameters (lidar_image_h/w, lidar_max_range,
    lidar_frame_stack) are added.

    Mount parameters (pos / rot) and noise settings are identical to those in
    ``parkour_lidar_env_cfg.ParkourLidarEnvCfg`` (6.0 reference), which was verified with
    the Go2 SLAM local-window reference implementation (2026-06-25).

    ``debug_vis=False`` is intentional — this cfg is intended for large-scale training
    where viewport point-cloud rendering would add unnecessary overhead.

    Range-image contract (R2)
    -------------------------
    ``obs["lidar"]`` shape = ``(N, lidar_frame_stack * 2 * lidar_image_h * lidar_image_w)``
    produced by flattening a ``(N, K, C=2, H, W)`` buffer via ``.reshape(N, -1)``.

    - ch0 = range_norm: ``clamp(d, 0, lidar_max_range) / lidar_max_range`` for a hit;
      ``1.0`` for miss/occluded.
    - ch1 = hit_mask: ``1.0`` for valid hit, ``0.0`` for miss/occluded/dropout.
    - Frame stack: newest frame at index k=0, older frames at k=1..K-1.
    """

    # ------------------------------------------------------------------
    # Range-image projection parameters (R2 contract)
    # ------------------------------------------------------------------
    # Elevation bins for the range image (H).
    lidar_image_h: int = 24
    # Azimuth bins for the range image (W).  Full 360° coverage.
    lidar_image_w: int = 96
    # Maximum range for range normalisation (m).  Hits beyond this clamp to 1.0
    # in range_norm but still carry hit_mask=1 (distinct from empty bins).
    lidar_max_range: float = 20.0
    # Number of temporal frames stacked in the obs["lidar"] buffer (K).
    lidar_frame_stack: int = 3

    # Temporal ring-buffer push cadence.
    #
    # False (default): push a new range-image frame into the ring buffer on EVERY env
    #     control step.  With step_dt=0.02 s and K=3 this spans only 0.06 s — the three
    #     slots carry near-identical content (near-duplicate frames).
    #
    # True: push to the ring buffer only once every
    #     push_every = round(1 / (update_frequency * step_dt))
    #   env steps — matching the Mid-360 measurement rate (10 Hz).  At step_dt=0.02 s
    #   this gives push_every=5, so K=3 slots span 3x0.1 s = 0.3 s of real temporal
    #   context.  Non-push steps hold the previous ring-buffer content unchanged
    #   (obs["lidar"] is constant between pushes), matching real 10 Hz deployment.
    #
    # Note: LidarSensorCfg inherits update_period=0.0 from SensorBaseCfg, so the
    #   sensor's _is_outdated gate fires on every physics step regardless of
    #   update_frequency; push_every is therefore derived from update_frequency
    #   directly, not from update_period.
    lidar_stack_at_sensor_rate: bool = False

    # Mid-360 LiDAR — range-image side-channel only, NEVER concatenated into policy obs.
    # Exposed as: obs["lidar"] (N, K*2*H*W).
    mid360_lidar: LidarSensorCfg = LidarSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            # Mount position confirmed from SLAM_local_window mount_transform.py
            # (MOUNT_OFFSET_BASE_LINK_M): x=0.3336 uses the runtime-faithful boom mount.
            pos=(0.333644, -0.000485, 0.050079),
            # dome-down 180° flip ⊗ pitch-up 30° — matches 6.0 parkour_lidar_env_cfg reference.
            rot=(0.0, 0.96593, 0.0, 0.25882),
        ),
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=24000,
            use_simple_grid=False,
            downsample=1,
        ),
        ray_alignment="base",
        mesh_prim_paths=["/World/ground"],
        max_distance=40.0,
        min_range=0.2,
        debug_vis=False,  # disabled for training — no viewport overhead
        return_pointcloud=False,
        pointcloud_in_world_frame=False,
        update_frequency=10.0,
        # Noise: Mid-360 datasheet range accuracy ±2 cm (1σ); 10% dropout for sim-to-real DR.
        enable_sensor_noise=True,
        random_distance_noise=0.02,
        pixel_dropout_prob=0.10,
        random_angle_noise=0.0,
        pixel_std_dev_multiplier=0.0,
    )


@configclass
class ParkourImitationRandomGoalLidarSLEnvCfg(ParkourImitationRandomGoalLidarEnvCfg):
    """SL (R2 student-learning) arm cfg for ``Go2-ParkourImitation-Symmetry-RandomGoal-Lidar-SL-v0``.

    Identical to :class:`ParkourImitationRandomGoalLidarEnvCfg` (policy obs, amp_obs, LiDAR
    range-image side-channel — all unchanged) except the LiDAR temporal frame-stack is pushed
    at the sensor rate instead of the control rate.

    With the default control-rate cadence, K=3 frames span only step_dt*3 = 0.06 s of
    near-duplicate content.  Setting ``lidar_stack_at_sensor_rate = True`` pushes frames every
    ``push_every`` control steps so K=3 spans ~0.3 s (matching the 10 Hz sensor cadence).
    Obs shape is unchanged (``obs["lidar"]`` stays (N, 3*2*24*96=13824)).
    """

    lidar_stack_at_sensor_rate: bool = True

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Go2 ParkourImitation-Symmetry-RandomGoal + Mid-360 LiDAR environment.

Policy obs (policy/scan/priv_explicit/priv_latent/history) and amp_obs are UNCHANGED
relative to ParkourImitationRandomGoalEnvCfg.  The Mid-360 LiDAR is attached as an
additional sensor but is NEVER concatenated into any obs group.

LiDAR data is exposed exclusively via three extras keys — a read-only side-channel for
downstream modules such as a height-scan estimator that uses GT scan (obs["scan"], 187-dim)
as supervision and LiDAR geometry as raw input:

    extras["lidar_hits_w"]  shape (num_envs, num_rays, 3)  world-frame 3-D hit points, miss=inf
    extras["lidar_pos_w"]   shape (num_envs, 3)            sensor origin in world frame (m)
    extras["lidar_quat_w"]  shape (num_envs, 4)            sensor orientation (wxyz)

height_scan (obs["scan"]) is kept intact as the estimator supervision target.
Replacement of height_scan by LiDAR is strictly out-of-scope for this env.
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.lidar_sensor_cfg import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LivoxPatternCfg
from isaaclab.utils import configclass

from .parkour_imitation_random_goal_env_cfg import ParkourImitationRandomGoalEnvCfg

# ---------------------------------------------------------------------------
# Go2 self-occlusion mesh prims (per-env robot body + leg link visuals).
# Path structure: /World/envs/env_{N}/Robot/<link>/visuals/mesh_0
# {ENV_REGEX_NS} expands to the per-env root at runtime.
# Copied verbatim from parkour_lidar_env_cfg.py (runtime-verified, 17 prims).
# ---------------------------------------------------------------------------
_GO2_SELF_OCCLUSION_PRIMS: list[str] = [
    # robot body
    "{ENV_REGEX_NS}/Robot/base/visuals/mesh_0",
    # front-left leg
    "{ENV_REGEX_NS}/Robot/FL_hip/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/FL_thigh/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/FL_calf/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/FL_foot/visuals/mesh_0",
    # front-right leg
    "{ENV_REGEX_NS}/Robot/FR_hip/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/FR_thigh/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/FR_calf/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/FR_foot/visuals/mesh_0",
    # rear-left leg
    "{ENV_REGEX_NS}/Robot/RL_hip/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/RL_thigh/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/RL_calf/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/RL_foot/visuals/mesh_0",
    # rear-right leg
    "{ENV_REGEX_NS}/Robot/RR_hip/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/RR_thigh/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/RR_calf/visuals/mesh_0",
    "{ENV_REGEX_NS}/Robot/RR_foot/visuals/mesh_0",
]


@configclass
class ParkourImitationRandomGoalLidarEnvCfg(ParkourImitationRandomGoalEnvCfg):
    """Go2 ParkourImitation-RandomGoal cfg with an additional Livox Mid-360 LiDAR.

    All policy-facing fields (observation_space, height_scanner, obs_groups, amp_obs, etc.)
    are inherited UNCHANGED from ParkourImitationRandomGoalEnvCfg.  Only ``mid360_lidar``
    is added.

    Mount parameters (pos / rot) and noise settings are identical to those in
    ``parkour_lidar_env_cfg.ParkourLidarEnvCfg``, which was verified with the Go2
    SLAM local-window reference implementation (2026-06-25).

    ``debug_vis=False`` is intentional — this cfg is intended for large-scale training
    where viewport point-cloud rendering would add unnecessary overhead.
    """

    # Mid-360 LiDAR — side-channel only, NEVER concatenated into obs.
    # Exposed as: extras["lidar_hits_w"] (N,R,3), extras["lidar_pos_w"] (N,3), extras["lidar_quat_w"] (N,4).
    mid360_lidar: LidarSensorCfg = LidarSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            # Mount position confirmed from SLAM_local_window mount_transform.py
            # (MOUNT_OFFSET_BASE_LINK_M): x=0.3336 uses the runtime-faithful boom mount.
            pos=(0.333644, -0.000485, 0.050079),
            # dome-down 180° flip ⊗ pitch-up 30° (wxyz).
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
        dynamic_env_mesh_prim_paths=_GO2_SELF_OCCLUSION_PRIMS,
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

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour + Livox Mid-360 LiDAR inspection environment configuration.

Uses the core ``LidarSensorCfg`` (OmniPerception-extended) with the core
``LivoxPatternCfg(sensor_type="mid360")`` — no custom task-level pattern needed.

The existing height_scanner and ALL policy observations are UNCHANGED — the trained
parkour checkpoint loads without modification.
"""

from __future__ import annotations

from isaaclab.sensors import RayCasterCfg
from isaaclab.sensors.lidar_sensor_cfg import LidarSensorCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import LivoxPatternCfg
from isaaclab.utils import configclass

from .parkour_env_cfg import ParkourEnvCfg

# Go2 self-occlusion mesh prims — runtime-verified (17 prims, all use "mesh_0").
# Path structure confirmed by runtime prim dump:
#   /World/envs/env_0/Robot/<link>/visuals/mesh_0
# env index is replaced by {ENV_REGEX_NS} for per-env instancing.
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
class ParkourLidarEnvCfg(ParkourEnvCfg):
    """Go2 Parkour env cfg with an additional Mid-360 LiDAR (inspection-only).

    All policy-facing fields (observation_space, height_scanner, obs_groups, etc.)
    are inherited UNCHANGED from ParkourEnvCfg.  Only ``mid360_lidar`` is added.

    Uses core ``LidarSensorCfg`` with ``LivoxPatternCfg(sensor_type="mid360")``
    verified working in Sim 5.1.  Self-occlusion is enabled via
    ``dynamic_env_mesh_prim_paths`` (Go2 body + leg meshes).
    """

    # Mid-360 LiDAR — inspection-only, NEVER concatenated into observations.
    # Mount: SLAM_local_window/mount_transform.py (runtime-faithful values).
    # mesh_prim_paths = terrain ground mesh, same as height_scanner.
    # dynamic_env_mesh_prim_paths = per-env robot body/leg meshes (self-occlusion).
    mid360_lidar: LidarSensorCfg = LidarSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(
            # SLAM_local_window mount_transform.py: MOUNT_OFFSET_BASE_LINK_M
            # = _stl_to_go2((-0.000485, -0.333644, +0.050079)) = (+0.333644, -0.000485, +0.050079)
            # NOTE: SLAM 코드 활성값=0.3336 채택(runtime-faithful). 실물 사진으로 확정(2026-06-25):
            # Mid-360이 Go2 머리에서 전방으로 뻗은 브래킷(boom) 위에 장착 → 몸통보다 ~14 cm 앞이 맞음.
            # (docstring 유도값 0.1336은 boom 없는 head-mount 가정이라 실물과 다름.)
            pos=(0.333644, -0.000485, 0.050079),
            
            # SLAM_local_window mount_transform.py: MOUNT_ROT_WXYZ
            # = (0.0, cos(15°), 0.0, sin(15°)) ≈ (0, 0.96593, 0, 0.25882)
            # dome-down 180° flip ⊗ pitch-up 30° (w=0 → flip 포함).
            # 기존 _tilt_quaternion(30.0)은 (cos15°,0,sin15°,0)=(0.966,0,0.259,0) — 순수 pitch, flip 없음.
            rot=(0.0, 0.96593, 0.0, 0.25882),
        ),
        pattern_cfg=LivoxPatternCfg(
            sensor_type="mid360",
            samples=24000,           # SLAM publisher 기본=20000; 기존값 24000 유지 (사용자 결정 대기)
            use_simple_grid=False,   # SLAM과 동일 (LivoxPatternCfg 기본값도 False — 명시만)
            downsample=1,            # SLAM과 동일 (LivoxPatternCfg 기본값도 1 — 명시만)
        ),
        ray_alignment="base",
        mesh_prim_paths=["/World/ground"],
        dynamic_env_mesh_prim_paths=_GO2_SELF_OCCLUSION_PRIMS,
        # dynamic_env_mesh_prim_paths=[],
        max_distance=40.0,
        min_range=0.2,
        debug_vis=True,  # GUI point-cloud overlay in viewport (no perf cost on play-only tasks)
        return_pointcloud=False,  # not needed; use ray_hits_w directly
        pointcloud_in_world_frame=False,
        update_frequency=10.0,  # 10 Hz is sufficient for inspection
        # ── Mid-360 noise fidelity ──────────────────────────────────────────────
        # Field semantics confirmed from _apply_noise() in lidar_sensor.py:
        #   distance += randn() * random_distance_noise  (absolute Gaussian, σ in metres)
        #   ray dropped (distance=max_distance) when rand() < pixel_dropout_prob  ([0,1] prob)
        #   random_angle_noise / pixel_std_dev_multiplier are declared but NOT implemented
        #   in the current _apply_noise; they have no runtime effect.
        #
        # Mid-360 datasheet: range accuracy ±2 cm (1σ) at typical ranges.
        # → random_distance_noise = 0.02 m  (σ = 2 cm, absolute metres)
        #
        # OmniPerception noise DR: ~10% point masking for sim-to-real robustness.
        # → pixel_dropout_prob = 0.10  (10% of rays return max_distance)
        enable_sensor_noise=True,
        random_distance_noise=0.02,    # σ=2 cm — Mid-360 range accuracy (1σ, absolute m)
        pixel_dropout_prob=0.10,       # 10% ray dropout — OmniPerception DR baseline
        random_angle_noise=0.0,        # declared but not implemented; 0 = no silent effect
        pixel_std_dev_multiplier=0.0,  # declared but not implemented; 0 = no silent effect
    )

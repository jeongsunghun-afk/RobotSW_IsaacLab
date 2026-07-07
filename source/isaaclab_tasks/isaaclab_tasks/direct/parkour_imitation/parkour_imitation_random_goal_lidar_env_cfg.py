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
    and the range-image projection parameters (lidar_image_h/w, lidar_max_range,
    lidar_frame_stack) are added.

    Mount parameters (pos / rot) and noise settings are identical to those in
    ``parkour_lidar_env_cfg.ParkourLidarEnvCfg``, which was verified with the Go2
    SLAM local-window reference implementation (2026-06-25).

    ``debug_vis=False`` is intentional — this cfg is intended for large-scale training
    where viewport point-cloud rendering would add unnecessary overhead.

    Range-image contract (R2)
    -------------------------
    ``obs["lidar"]`` shape = ``(N, lidar_frame_stack * 2 * lidar_image_h * lidar_image_w)``
    produced by flattening a ``(N, K, C=2, H, W)`` buffer via ``.reshape(N, -1)``.

    - ch0 = range encoding (mode selected by ``lidar_range_encoding``):
        - "inverse" (default): ``closeness = 1 / (1 + d)`` for a hit; ``0.0`` for miss/occluded.
          near→high (0.67@0.5m), far→low (0.05@20m), empty/miss→0.0.
        - "linear": ``clamp(d, 0, lidar_max_range) / lidar_max_range`` for a hit; ``1.0`` for miss/occluded.
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
    # Maximum range used for distance clamping in linear mode (m).
    # In inverse-depth mode far values are auto-compressed; this field is kept for
    # reproducibility and linear-mode backward compat.  Value unchanged: 20.0.
    lidar_max_range: float = 20.0
    # Number of temporal frames stacked in the obs["lidar"] buffer (K).
    lidar_frame_stack: int = 3
    # ch0 encoding mode for the range image.
    #   "inverse" (default): closeness = 1 / (1 + d).  near→high, far→low, miss/occluded→0.0.
    #              d=0.5→0.67  d=1→0.50  d=2→0.33  d=5→0.17  d=20→0.05  miss→0.0
    #              Empty/miss bins: 0.0.  Scatter: amax (closest hit = highest closeness).
    #   "linear":  range_norm = clamp(d, 0, lidar_max_range) / lidar_max_range.
    #              miss/occluded→1.0.  Empty bins: 1.0.  Scatter: amin (closest = lowest norm).
    #              Preserved for A/B comparison and checkpoint reproduction.
    lidar_range_encoding: str = "inverse"

    # Temporal ring-buffer push cadence.
    #
    # False (default): push a new range-image frame into the ring buffer on EVERY env
    #     control step.  With step_dt=0.02 s and K=3 this spans only 0.06 s — the three
    #     slots carry nearly-identical content because the scene barely changes in 0.06 s.
    #     This is the current (default) behaviour; the running GPU3 experiment uses it.
    #
    # True: push to the ring buffer only once every
    #     push_every = round(1 / (update_frequency * step_dt))
    #   env steps — the same cadence as the intended Mid-360 measurement rate (10 Hz).
    #   At step_dt=0.02 s this gives push_every=5, so K=3 slots span 3×0.1 s = 0.3 s of
    #   real temporal context.  Non-push steps hold the previous ring-buffer content
    #   unchanged (obs["lidar"] is constant between pushes), which also matches the
    #   real 10 Hz deployment: the on-robot policy receives the same scan for 5 control
    #   cycles.
    #
    # Note on sensor update_period: LidarSensorCfg inherits update_period=0.0 from
    #   SensorBaseCfg, so the sensor's _is_outdated gate fires on every physics step
    #   regardless of update_frequency.  push_every is therefore derived from
    #   update_frequency directly (not from update_period) to achieve the intended rate.
    #
    # Activation: set this to True in the next experiment after the current GPU3
    #   inverse-depth run completes (attribution isolation).
    lidar_stack_at_sensor_rate: bool = False

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
        # R1 throughput fix (2026-06-30): dynamic body-mesh self-occlusion removed.
        # The 17-prim dynamic mesh path (see _GO2_SELF_OCCLUSION_PRIMS above) caused a
        # second full raycast_mesh + per-fire warp.refit() of 17 prims × N_env, inflating
        # step-time from ~21 ms to ~45 s/step and causing 16 GB GPU OOM at 512 envs.
        #
        # Static self-occlusion mask is DEFERRED pending two blockers:
        #   1. Sensor-inside-body-mesh artifact: all 24000 rays hit the inner mesh surface
        #      at 0.07–0.16 m (_bc_ON.npy probe) — body-only cast returns all-blocked.
        #   2. sensor_t quadratic rotation: _update_dynamic_rays applies an incremental
        #      rotation of sensor_t*0.1 each call (not absolute from initial), so the total
        #      rotation = Σ(k*0.01) — quadratic in update count — making a per-index mask
        #      meaningless within ~100 sensor updates (~10 s).
        # Correct approach: _workspace/parkour_imitation_lidar/r1_throughput/compute_body_mask.py
        # uses body-mesh-only raycast (_GO2_SELF_OCCLUSION_PRIMS as static mesh_prim_paths)
        # with ray starts offset by min_range (0.2 m) to clear the inside-mesh artifact.
        # Requires GPU run + sensor_t fix decision before wiring into env.
        # Residual sim-to-real gap is covered by the existing 10% pixel_dropout_prob DR.
        dynamic_env_mesh_prim_paths=[],
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

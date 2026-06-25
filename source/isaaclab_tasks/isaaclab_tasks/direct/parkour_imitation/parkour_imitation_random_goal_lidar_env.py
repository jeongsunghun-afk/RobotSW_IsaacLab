# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 ParkourImitation-RandomGoal + Livox Mid-360 LiDAR environment.

Attaches a Livox Mid-360 LiDAR to ``Go2ParkourImitationRandomGoalEnv`` and exposes
the raw 3D hit geometry as an extras side-channel.  The policy obs pipeline is
byte-identical to the base RandomGoal env — no obs group is added or modified.

Side-channel contract
---------------------
After each ``_get_observations()`` call:

    self.extras["lidar_hits_w"]  — torch.Tensor, shape (num_envs, num_rays, 3), float32
                                   World-frame 3-D hit points.  Missed / dropped rays
                                   carry inf values.  Cloned so callers own the data.

    self.extras["lidar_pos_w"]   — torch.Tensor, shape (num_envs, 3), float32
                                   Sensor origin in world frame (metres).

    self.extras["lidar_quat_w"]  — torch.Tensor, shape (num_envs, 4), float32
                                   Sensor orientation in world frame (wxyz).

Why ray_hits_w instead of distances
-------------------------------------
LidarSensor._update_dynamic_rays() rotates the ray-direction array in-place each
update (Z-axis spin for Livox pattern).  The per-ray direction is an internal
attribute not exposed in LidarSensorData, so a scalar distance[i] alone cannot be
back-projected to a 3-D point.  ray_hits_w is self-sufficient — it already encodes
the cast geometry — and is therefore the correct estimator input.

Estimator usage
---------------
    base_hits = quat_apply_inverse(lidar_quat_w, lidar_hits_w - lidar_pos_w)
    mask      = torch.isfinite(base_hits).all(-1)   # valid-ray boolean mask
    # apply point-level domain-randomisation on base_hits before feeding estimator
    # supervision target: obs["scan"]  (187-dim GT heightmap, base frame, metres)

GT height_scan (obs["scan"], 187-dim) is preserved as the estimator supervision
target.  Replacing it with LiDAR data is out of scope for this env.

No new persistent buffers are added.  The sensor data is read live each step, so no
``_reset_idx`` override is needed.

Inheritance chain:
    Go2ParkourImitationRandomGoalLidarEnv
        → Go2ParkourImitationRandomGoalEnv
            → Go2ParkourImitationEnv
                → Go2ParkourEnv
                    → DirectRLEnv
"""

from __future__ import annotations

from isaaclab.sensors.lidar_sensor import LidarSensor

from .parkour_imitation_random_goal_env import Go2ParkourImitationRandomGoalEnv
from .parkour_imitation_random_goal_lidar_env_cfg import ParkourImitationRandomGoalLidarEnvCfg


class Go2ParkourImitationRandomGoalLidarEnv(Go2ParkourImitationRandomGoalEnv):
    """Go2 ParkourImitation-RandomGoal env with attached Livox Mid-360 LiDAR.

    The only additions over the parent env are:

    1. ``self._mid360`` (``LidarSensor``) — created in ``_setup_scene`` after all base
       sensors and terrain, appended to ``self.scene.sensors``.

    2. Three extras keys — populated in ``_get_observations`` each step:
       - ``extras["lidar_hits_w"]``  shape (num_envs, num_rays, 3)  world-frame hits, miss=inf
       - ``extras["lidar_pos_w"]``   shape (num_envs, 3)            sensor world position
       - ``extras["lidar_quat_w"]``  shape (num_envs, 4)            sensor orientation (wxyz)

       The obs dict returned is the unmodified parent result.

    Policy obs dimensions (policy/scan/priv_explicit/priv_latent/history) and amp_obs
    are byte-identical to ``Go2ParkourImitationRandomGoalEnv``.
    """

    cfg: ParkourImitationRandomGoalLidarEnvCfg

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Create all base sensors / terrain, then append the Mid-360 LidarSensor."""
        super()._setup_scene()

        # LidarSensor resolves its prim_path regex lazily at sim-start, so appending
        # after clone_environments() is safe (same pattern as parkour height_scanner).
        self._mid360 = LidarSensor(self.cfg.mid360_lidar)
        self.scene.sensors["mid360_lidar"] = self._mid360

    # ------------------------------------------------------------------
    # Observation hook — obs untouched; lidar exposed via extras
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Return the parent obs dict unchanged; expose LiDAR geometry in extras.

        The parent obs dict (policy / scan / priv_explicit / priv_latent / history /
        amp) is returned with zero modifications.  Three extras keys are written with
        cloned copies of the sensor's geometry tensors.

        Note: self.extras is NOT reassigned — only individual keys are set — to prevent
        silently dropping other keys (amp_obs, terminal_amp_obs, log) written earlier
        in the step by the parent chain.

        ray_hits_w rationale: LidarSensor rotates the ray-direction array in-place each
        update (_update_dynamic_rays), so the per-ray direction is not exposed in
        LidarSensorData.  A scalar distance alone cannot be back-projected to a 3-D
        point.  ray_hits_w encodes the full cast geometry and is therefore the correct
        estimator input.
        """
        obs = super()._get_observations()

        # Expose raw LiDAR geometry as top-level extras keys.  Never nested under
        # extras["observations"] to avoid being picked up by symmetry / obs groups.
        self.extras["lidar_hits_w"] = self._mid360.data.ray_hits_w.clone()  # (N, R, 3) world, miss=inf
        self.extras["lidar_pos_w"] = self._mid360.data.pos_w.clone()        # (N, 3) sensor world pos
        self.extras["lidar_quat_w"] = self._mid360.data.quat_w.clone()      # (N, 4) wxyz

        return obs

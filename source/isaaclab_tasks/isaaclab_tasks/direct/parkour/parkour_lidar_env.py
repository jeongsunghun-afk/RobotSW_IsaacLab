# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Parkour + Livox Mid-360 LiDAR inspection environment.

Uses the core ``LidarSensor`` (OmniPerception-extended) instead of the custom
task-level ``RayCaster`` + ``LivoxPatternCfg``.  Self-occlusion (Go2 body/leg
meshes) is enabled via ``dynamic_env_mesh_prim_paths`` in the cfg.

The existing ``height_scanner`` and the ENTIRE obs pipeline are UNCHANGED —
the trained parkour checkpoint loads as-is.

Usage (headless smoke / train):

    conda run --no-capture-output -n isaac-5.1 python \\
        scripts/reinforcement_learning/rsl_rl/train.py \\
        --task Go2-Parkour-Lidar-v0 --num_envs 16 --max_iterations 2 --headless

Usage (play with existing checkpoint, GUI with real-time point cloud):

    conda run --no-capture-output -n isaac-5.1 python \\
        scripts/reinforcement_learning/rsl_rl/play.py \\
        --task Go2-Parkour-Lidar-v0 --num_envs 4
"""

from __future__ import annotations

import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, VisualizationMarkers
from isaaclab.sensors.lidar_sensor import LidarSensor
from isaaclab.utils.math import quat_apply, quat_mul

from .parkour_env import Go2ParkourEnv
from .parkour_lidar_env_cfg import ParkourLidarEnvCfg

# Auto-dump fires at this common_step_counter value (headless inspection convenience).
# num_steps_per_env=24 → step 20 fires well within the first training iteration.
_AUTO_DUMP_STEP: int = 20


class Go2ParkourLidarEnv(Go2ParkourEnv):
    """Go2 Parkour env with an attached Livox Mid-360 LiDAR (inspection-only).

    Policy observations are byte-identical to ``Go2ParkourEnv``.  The lidar sensor
    (``self._mid360``, type ``LidarSensor``) is created after the base sensors and
    terrain.  It uses core ``LidarSensorCfg`` with ``LivoxPatternCfg(sensor_type="mid360")``
    and includes Go2 self-occlusion meshes.

    Data API (same as base ``RayCaster`` — ``LidarSensorData`` extends ``RayCasterData``):
    - ``self._mid360.data.ray_hits_w``  shape (num_envs, num_rays, 3), world-frame hits
    - ``self._mid360.data.pos_w``       shape (num_envs, 3), sensor world position
    - ``self._mid360.data.distances``   shape (num_envs, num_rays), range in metres

    Auto-dump: fires once at step ``_AUTO_DUMP_STEP`` (headless-safe).
    Interactive: press ``M`` in the GUI (DebugViewer) to dump on demand.
    """

    cfg: ParkourLidarEnvCfg

    def __init__(self, cfg: ParkourLidarEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg, render_mode, **kwargs)

        # One-time auto-dump flag (headless-safe).
        self._lidar_auto_dumped: bool = False

        # Register interactive lidar dump key ("M" — base parkour uses P/J/K/L/G).
        self._debug_viewer.register_key(
            "M",
            on_press=lambda: self.dump_lidar_cloud(env_idx=0, tag="parkour_interactive"),
        )

        # FRAME marker showing Mid-360 sensor pose (position + orientation).
        # Created only when GUI is active; headless runs skip marker creation entirely.
        if self._debug_viewer.has_gui:
            _frame_cfg = FRAME_MARKER_CFG.copy()
            _frame_cfg.prim_path = "/Visuals/Debug/mid360_pose"
            _frame_cfg.markers["frame"].scale = (0.25, 0.25, 0.25)
            self._lidar_pose_marker: VisualizationMarkers | None = VisualizationMarkers(_frame_cfg)
            # Yellow arrow pointing along sensor local +z (the actual LiDAR view-cone axis).
            # arrow_x.usd points along local +x; a fixed y-axis −90° rotation in _draw_lidar_pose()
            # remaps +x → +z so the arrow tracks the true scan direction (forward-down).
            # Yellow distinguishes it from the FRAME's blue +z axis.
            _arrow_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
            _arrow_cfg.prim_path = "/Visuals/Debug/mid360_view"
            _arrow_cfg.markers["arrow"].scale = (0.4, 0.1, 0.1)
            _arrow_cfg.markers["arrow"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))
            self._lidar_view_marker: VisualizationMarkers | None = VisualizationMarkers(_arrow_cfg)
            self._debug_viewer.register_debug_vis(
                "lidar_pose",
                lambda _dt: self._draw_lidar_pose(),
                default_on=True,
            )
            self._debug_viewer.register_key("N", on_press=self._toggle_lidar_pose)
        else:
            self._lidar_pose_marker = None
            self._lidar_view_marker = None

    # ------------------------------------------------------------------
    # Scene setup — append LidarSensor after base sensors and terrain
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        """Create all base sensors/terrain, then append the Mid-360 LidarSensor."""
        super()._setup_scene()

        # LidarSensor resolves its prim_path regex lazily at sim-start, so appending
        # after clone_environments() is safe (same pattern as parkour height_scanner).
        self._mid360 = LidarSensor(self.cfg.mid360_lidar)
        self.scene.sensors["mid360_lidar"] = self._mid360

    # ------------------------------------------------------------------
    # Observation hook — policy obs untouched; side-effect dump only
    # ------------------------------------------------------------------

    def _get_observations(self) -> dict:
        """Return obs dict unchanged; fire one-time lidar dump for inspection."""
        obs = super()._get_observations()

        if not self._lidar_auto_dumped and self.common_step_counter >= _AUTO_DUMP_STEP:
            self._lidar_auto_dumped = True
            self._report_lidar_stats(env_idx=0, step=self.common_step_counter)
            self.dump_lidar_cloud(env_idx=0, tag="parkour")

        return obs

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def dump_lidar_cloud(self, env_idx: int = 0, tag: str = "parkour") -> None:
        """Save the current Mid-360 point cloud for ``env_idx`` to ``_workspace/sim2real/``.

        Files written:

        * ``mid360_hits_world_<tag>.npy``  — (N_valid, 3) world-frame finite hits
        * ``mid360_sensor_pos_<tag>.npy``  — (3,) sensor origin in world frame

        Hit data comes from ``LidarSensorData.ray_hits_w`` (inherited from
        ``RayCasterData``) — inf/nan entries are filtered as misses.
        """
        _d = os.path.dirname(os.path.abspath(__file__))
        for _ in range(8):
            if os.path.isfile(os.path.join(_d, "isaaclab.sh")):
                break
            _d = os.path.dirname(_d)
        out_dir = os.path.join(_d, "_workspace", "sim2real")
        os.makedirs(out_dir, exist_ok=True)

        # ray_hits_w: (num_envs, num_rays, 3) — finite entries are valid terrain hits
        hits = self._mid360.data.ray_hits_w[env_idx]  # (num_rays, 3)
        finite_mask = torch.isfinite(hits).all(dim=-1)
        hits_world = hits[finite_mask].detach().cpu().numpy().astype(np.float32)

        # true sensor origin: base_pos + rotate(base_quat, offset_pos) — offset-corrected
        pos_w, _ = self._true_sensor_pose()
        sensor_pos = pos_w[env_idx].detach().cpu().numpy().astype(np.float32)

        path_hits = os.path.join(out_dir, f"mid360_hits_world_{tag}.npy")
        path_pos = os.path.join(out_dir, f"mid360_sensor_pos_{tag}.npy")
        np.save(path_hits, hits_world)
        np.save(path_pos, sensor_pos)

        n_valid = hits_world.shape[0]
        n_total = hits.shape[0]
        print(
            f"[Mid360] dump @ env_idx={env_idx}: valid={n_valid}/{n_total} "
            f"({100 * n_valid / max(n_total, 1):.1f}%)  ->  {out_dir}"
        )

    def _true_sensor_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the true Mid-360 sensor world pose for all envs.

        Composes the base prim pose with the sensor mount offset so that the
        result is independent of whether ``data.pos_w`` is offset-aware or not.

        Returns:
            sensor_pos  (num_envs, 3) — world-frame sensor origin
            sensor_quat (num_envs, 4) — world-frame sensor orientation (wxyz)
        """
        base_pos = self._mid360.data.pos_w    # (N, 3)
        base_quat = self._mid360.data.quat_w  # (N, 4) wxyz
        N = base_quat.shape[0]
        dev, dtype = base_quat.device, base_quat.dtype

        # true sensor position: base_pos + rotate(base_quat, offset_pos)
        offset_pos = (
            torch.tensor(self._mid360.cfg.offset.pos, device=dev, dtype=dtype)
            .unsqueeze(0)
            .expand(N, 3)
        )  # (N, 3)
        sensor_pos = base_pos + quat_apply(base_quat, offset_pos)  # (N, 3)

        # true sensor orientation: base_quat ⊗ offset_rot
        offset_quat = (
            torch.tensor(self._mid360.cfg.offset.rot, device=dev, dtype=dtype)
            .unsqueeze(0)
            .expand(N, 4)
        )  # (N, 4) wxyz
        sensor_quat = quat_mul(base_quat, offset_quat)  # (N, 4)

        return sensor_pos, sensor_quat

    def _draw_lidar_pose(self) -> None:
        """Visualize the Mid-360 sensor pose as a FRAME marker for all envs.

        Computes the true sensor world pose by composing the base prim pose with the
        sensor's mount offset (cfg.offset.pos / cfg.offset.rot).  RayCaster stores
        only the base prim pose in ``data.pos_w`` / ``data.quat_w``; the offset is
        baked into ray patterns only, so raw ``data.*`` values point at the robot base,
        not the physical sensor location.

        No-ops when the marker was not created (headless mode).
        """
        if self._lidar_pose_marker is None:
            return

        sensor_pos, sensor_quat = self._true_sensor_pose()  # (N, 3), (N, 4)

        self._lidar_pose_marker.visualize(
            translations=sensor_pos,
            orientations=sensor_quat,
        )

        if self._lidar_view_marker is None:
            return
        # q_fixed: y-axis rotation of −90° → sends arrow local +x to sensor local +z.
        # wxyz = (cos(−45°), 0, sin(−45°), 0) = (0.70711, 0.0, −0.70711, 0.0)
        N = sensor_quat.shape[0]
        dev, dtype = sensor_quat.device, sensor_quat.dtype
        q_fixed = (
            torch.tensor([0.70711, 0.0, -0.70711, 0.0], device=dev, dtype=dtype)
            .unsqueeze(0)
            .expand(N, 4)
        )  # (N, 4)
        # quat_mul(parent, child): true sensor orientation ⊗ q_fixed → arrow along sensor +z.
        arrow_quat = quat_mul(sensor_quat, q_fixed)
        self._lidar_view_marker.visualize(
            translations=sensor_pos,
            orientations=arrow_quat,
        )

    def _toggle_lidar_pose(self) -> None:
        """Toggle the Mid-360 pose FRAME marker on/off (bound to the "N" key)."""
        self._debug_viewer.set_debug_vis("lidar_pose", not self._debug_viewer.vis_is_enabled("lidar_pose"))

    def _report_lidar_stats(self, env_idx: int = 0, step: int = 0) -> None:
        """Print lidar hit statistics for ``env_idx``."""
        hits = self._mid360.data.ray_hits_w[env_idx]  # (num_rays, 3)
        total = hits.shape[0]
        finite_mask = torch.isfinite(hits).all(dim=-1)
        n_valid = int(finite_mask.sum().item())
        n_inf = int(torch.isinf(hits).any(dim=-1).sum().item())
        n_nan = int(torch.isnan(hits).any(dim=-1).sum().item())
        print(f"[Mid360] step={step} total={total} valid={n_valid} inf={n_inf} nan={n_nan}")
        if n_valid > 0:
            valid = hits[finite_mask]
            mn = valid.min(dim=0).values
            mx = valid.max(dim=0).values
            print(
                f"[Mid360]   x[{mn[0]:.3f},{mx[0]:.3f}] "
                f"y[{mn[1]:.3f},{mx[1]:.3f}] "
                f"z[{mn[2]:.3f},{mx[2]:.3f}]"
            )
        else:
            print("[Mid360]   (no finite hits — check tilt_deg and mesh_prim_paths)")

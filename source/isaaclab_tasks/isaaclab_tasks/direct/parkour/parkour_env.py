# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import weakref

import numpy as np
import torch

import omni.kit.app
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, SPHERE_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.trimesh import mesh_terrains as _parkour_mesh_terrains
from isaaclab.utils import math as math_utils

from isaaclab_tasks.direct._common import DebugKeyBindingCfg, DebugViewer, DebugViewerCfg

from .parkour_env_cfg import (
    TERRAIN_CLASS_FLAT,
    TERRAIN_CLASS_HURDLE,
    TERRAIN_CLASS_STEP,
    TERRAIN_CLASS_GAP,
    TERRAIN_CLASS_STAIR,
    TERRAIN_CLASS_STEPPING_STONES,
    TERRAIN_CLASS_BALANCE_BEAM,
    TERRAIN_CLASS_CRAWL,
    TERRAIN_CLASS_SLOPE,
    TERRAIN_CLASS_ZIGZAG_HURDLES,
    TERRAIN_CLASS_ROUGH_BLOCKS,
    ParkourEnvCfg,
)

# Mapping from terrain class ID → short name used as WandB metric suffix.
# Only classes that are active (proportion > 0) will actually appear in logs
# because the mask check skips empty classes.
_TERRAIN_CLASS_NAMES: dict[int, str] = {
    TERRAIN_CLASS_FLAT: "flat",
    TERRAIN_CLASS_HURDLE: "hurdle",
    TERRAIN_CLASS_STEP: "step",
    TERRAIN_CLASS_GAP: "gap",
    TERRAIN_CLASS_STAIR: "stair",
    TERRAIN_CLASS_STEPPING_STONES: "stepping_stones",
    TERRAIN_CLASS_BALANCE_BEAM: "balance_beam",
    TERRAIN_CLASS_CRAWL: "crawl",
    TERRAIN_CLASS_SLOPE: "slope",
    TERRAIN_CLASS_ZIGZAG_HURDLES: "zigzag_hurdles",
    TERRAIN_CLASS_ROUGH_BLOCKS: "rough_blocks",
}

from .parkour_terrains import compute_edge_mask_from_terrain_mesh as _compute_edge_mask_from_terrain_mesh


class _CapturingTerrainImporter(TerrainImporter):
    """TerrainImporter subclass that captures the concatenated trimesh before it is discarded.

    ``TerrainImporter.__init__`` calls ``self.import_mesh("terrain", terrain_generator.terrain_mesh)``
    and then lets the generator (and its mesh) go out of scope.  By overriding ``import_mesh`` we
    grab the trimesh the first time it is called so the env can use it for edge-mask computation.

    The captured mesh is stored in ``self._captured_trimesh`` (None until first import_mesh call).
    """

    def __init__(self, cfg, *args, **kwargs):
        self._captured_trimesh = None
        super().__init__(cfg, *args, **kwargs)

    def import_mesh(self, name: str, mesh, **kwargs):
        # Capture the very first mesh (which is the full terrain mesh from TerrainGenerator)
        if self._captured_trimesh is None:
            self._captured_trimesh = mesh
        super().import_mesh(name, mesh, **kwargs)


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class Go2ParkourEnv(DirectRLEnv):
    cfg: ParkourEnvCfg

    def __init__(self, cfg: ParkourEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Actions
        self._actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Processed actions (for action_smoothness rewards) — Genesis original
        self._processed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._last_processed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self._last_last_processed_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Applied torques (for delta_torques reward) — Genesis original
        self._last_applied_torque = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Commands [lin_vel_x, lin_vel_y, ang_vel_z]
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Per-env step counter for time-based velocity command resampling.
        # Counts policy steps since the last resample; reset to 0 on episode start and after each resample.
        # Interval: cfg.resampling_time_s / step_dt  (6.0 s / 0.02 s = 300 steps).
        self._time_since_command_resample = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._command_resample_interval: int = max(1, int(round(self.cfg.resampling_time_s / self.step_dt)))

        # Cached sensor observations refreshed at 10 Hz (every 5 policy steps).
        # Allocated here as zero tensors so _get_observations() never references an undefined
        # attribute on the very first call (before common_step_counter reaches a multiple of 5).
        # _reset_idx does NOT need to clear these: the elementwise recently_reset gate in
        # _get_observations handles stale carry-over for newly reset envs.
        _num_scan_rays = self._height_scanner.data.ray_hits_w.shape[1]  # C (e.g. 187)
        self._scan = torch.zeros(self.num_envs, _num_scan_rays, device=self.device)
        self._yaw_diff = torch.zeros(self.num_envs, device=self.device)
        self._next_yaw_diff = torch.zeros(self.num_envs, device=self.device)

        # Goal tracking: which waypoint each env is targeting
        self._current_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Goal waypoint system (Task #2)
        num_total_goals = self.cfg.num_goals + self.cfg.num_future_goal_obs
        self._env_goals = torch.zeros(self.num_envs, num_total_goals, 3, device=self.device)
        self._reach_goal_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self._next_target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._next_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._cur_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self._next_goals = torch.zeros(self.num_envs, 3, device=self.device)

        # Terrain curriculum — alias TerrainImporter's tensors so spawn (env_origins) and
        # goal lookup (_init_env_goals) always reference the same (row, col) cell.
        # Previously these were independent torch.randint / torch.div tensors, causing
        # the robot to spawn on row=A while goals were computed for row=B.
        num_cols = self.cfg.terrain.terrain_generator.num_cols
        self._terrain_levels = self._terrain.terrain_levels  # shape (num_envs,), int64
        self._terrain_types = self._terrain.terrain_types    # shape (num_envs,), int64

        # Per-env terrain class index (0=flat, 1=hurdle, 2=step, 3=gap, 4=stair).
        # _terrain_types is a column index (0..num_cols-1); map to class via proportion-based LUT.
        # LUT formula mirrors TerrainGenerator._generate_curriculum_terrain() exactly:
        #   sub_index = min(where(col/num_cols + 0.001 < cumsum(proportions)))
        # RAW cumsum (no normalization) — must match TG or reward masks diverge.
        _props = [sc.proportion for sc in self.cfg.terrain.terrain_generator.sub_terrains.values()]
        assert abs(sum(_props) - 1.0) < 1e-6, (
            f"sub_terrain proportions must sum to 1.0 (got {sum(_props):.4f}); "
            "env_class LUT would diverge from TerrainGenerator's column assignment."
        )
        _cumprops = torch.tensor(
            [sum(_props[: i + 1]) for i in range(len(_props))], device=self.device
        )
        _col_vals = torch.arange(num_cols, device=self.device).float() / num_cols + 1e-3
        # self._col_to_class[c] = terrain class for column c
        self._col_to_class = (_col_vals.unsqueeze(1) >= _cumprops.unsqueeze(0)).sum(dim=1).long()
        self._env_class = self._col_to_class[self._terrain_types]  # [num_envs]

        # Per-env flag: when True, _update_terrain_curriculum skips curriculum logic for that env
        # so the keyboard-forced level/type is preserved.  Cleared inside _update_terrain_curriculum
        # (not _reset_idx) because the curriculum block is the only consumer that would overwrite it.
        self._skip_curriculum = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Contact history for edge detection
        self._last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)

        # Previous joint velocity for acceleration computation
        self._prev_joint_vel = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)

        # Proprioceptive history ring buffer (Task #3)
        self._proprio_history = torch.zeros(
            self.num_envs, self.cfg.history_len, self.cfg.num_proprio, device=self.device
        )

        # Episode sums for logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in self.cfg.reward_scales.keys()
        }

        # Per-step scaled reward contribution for env 0 — read by external UDP debug publisher.
        # Updated every _get_rewards() call; never affects reward computation.
        self._last_reward_breakdown_env0: dict[str, float] = {
            key: 0.0 for key in self.cfg.reward_scales.keys()
        }

        # Get body indices for contact sensing
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(
            ["base", ".*thigh", ".*calf", ".*hip", "Head_upper", "Head_lower"]
        )

        # Foot shape indices for privileged friction observation (Task #5).
        # PhysX ArticulationView.get_material_properties() returns (num_envs, num_shapes, 3)
        # where num_shapes is the TOTAL number of collision shapes across all links — NOT num_bodies.
        # Go2 has 27 shapes across 19 bodies (some links own multiple collision spheres/capsules).
        #
        # We compute num_shapes_per_body using the same pattern as isaaclab/envs/mdp/events.py
        # (randomize_rigid_body_material, lines 208-220): iterate link_paths[0] and query each
        # link's RigidBodyView.max_shapes.  The shape index range for body i is:
        #   [sum(num_shapes_per_body[:i]),  sum(num_shapes_per_body[:i]) + num_shapes_per_body[i])
        #
        # Strategy: use the FIRST shape of each foot link as representative (Option a).
        # Note: EventManager samples bucket_ids per-shape (not per-body), so in principle each
        # shape on a foot link can receive different friction values.  Using only the first shape
        # is therefore a representative (slightly lossy) sample of foot friction — acceptable for
        # priv obs, and keeps the priv dimension fixed at 14.
        _foot_body_ids, _foot_body_names = self._robot.find_bodies(".*foot")
        assert len(_foot_body_ids) == 4, (
            f"Expected 4 foot bodies (FR/FL/RR/RL_foot), got {len(_foot_body_ids)}: {_foot_body_names}"
        )

        # Build per-body shape counts via the link_paths API (identical to events.py pattern).
        _num_shapes_per_body: list[int] = []
        for _link_path in self._robot.root_physx_view.link_paths[0]:
            _link_view = self._robot._physics_sim_view.create_rigid_body_view(_link_path)
            _num_shapes_per_body.append(_link_view.max_shapes)

        # Sanity check: sum of per-body shapes must match total shapes in articulation.
        _total_shapes = self._robot.root_physx_view.max_shapes
        assert sum(_num_shapes_per_body) == _total_shapes, (
            f"Shape-count mismatch: per-body sum={sum(_num_shapes_per_body)} "
            f"vs articulation max_shapes={_total_shapes}. link_paths parsing may be wrong."
        )
        # Each foot body must have at least one shape.
        assert all(_num_shapes_per_body[bid] >= 1 for bid in _foot_body_ids), (
            f"One or more foot bodies have 0 collision shapes. "
            f"foot_body_ids={_foot_body_ids}, shapes={[_num_shapes_per_body[b] for b in _foot_body_ids]}"
        )

        # Compute cumulative shape offset for each body (i.e. the starting shape index).
        _shape_offsets = [sum(_num_shapes_per_body[:i]) for i in range(len(_num_shapes_per_body))]
        # First shape index for each foot body.
        _foot_first_shape_indices = [_shape_offsets[bid] for bid in _foot_body_ids]
        self._foot_shape_indices = torch.tensor(
            _foot_first_shape_indices, dtype=torch.long, device=self.device
        )  # shape: (4,)

        # One-time INFO log: body names, per-body shape counts, and selected shape indices.
        print(
            f"[Parkour] Foot body→shape mapping:\n"
            f"  foot body names : {_foot_body_names}\n"
            f"  foot body ids   : {_foot_body_ids}\n"
            f"  shapes per foot : {[_num_shapes_per_body[b] for b in _foot_body_ids]}\n"
            f"  foot shape idx  : {_foot_first_shape_indices}  (first shape per foot)\n"
            f"  total shapes    : {_total_shapes} across {len(_num_shapes_per_body)} bodies"
        )
        # get_material_properties() returns (N, num_shapes, 3).
        # Slice: (N, 4, 3)[..., :2] → (N, 8) static+dynamic friction per foot, restitution excluded.

        # Hip joint indices for hip_pos reward [0, 3, 6, 9]
        all_joint_names = self._robot.data.joint_names
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if "hip" in n],
            dtype=torch.long,
            device=self.device,
        )

        # Termination mask buffers (set by _get_dones each step, used in _reset_idx for logging)
        self._term_base_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_tilt = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._term_low_height = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        # Successful episode termination: robot reached the last goal waypoint
        self._term_goal_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # DEBUG: height-scan multi-step diagnostic.
        # Fires at step ∈ {5, 50, 100, 200, 500}; each step fires at most once per process.
        # Not reset in _reset_idx — print-once-per-step-set semantics.
        self._scan_debug_print_steps: set = {5, 50, 100, 200, 500}
        self._scan_debug_already_printed: set = set()
        # Legacy single-shot flag kept for backward compat (unused by new logic below).
        self._scan_debug_printed = False

        # Initialize goal waypoints (must happen after terrain is set up)
        self._init_env_goals(torch.arange(self.num_envs, device=self.device))

        # DebugViewer: free-fly camera, WASD/arrow movement, [ / ] env-index switching.
        # enable_keyboard_view_switch gates the helper in the same way it gated _setup_keyboard().
        _dbg_enabled = getattr(self.cfg, "enable_keyboard_view_switch", True)
        self._debug_viewer = DebugViewer(
            self,
            cfg=DebugViewerCfg(
                enabled=_dbg_enabled,
                free_fly_speed_mps=2.0,
                free_fly_speed_boost_mps=6.0,
                keys=DebugKeyBindingCfg(toggle_free_fly="G"),
            ),
        )

        # Parkour-specific P key: toggle contact debug print.
        self._contact_print_handle = self._debug_viewer.register_debug_vis(
            "contact_print",
            lambda dt: self._print_contact_debug(),
            default_on=bool(self.cfg.debug_print_contacts),
        )
        self._debug_viewer.register_key(
            "P",
            on_press=lambda: self._toggle_contact_print(),
        )
        # Interactive terrain navigation: J/K change difficulty level, L cycles terrain type.
        # Only the env currently tracked by the viewport camera is affected; all other envs
        # continue training without interruption.
        self._debug_viewer.register_key(
            "K",
            on_press=lambda: self._change_terrain_for_viewer(level_delta=+1),
        )
        self._debug_viewer.register_key(
            "J",
            on_press=lambda: self._change_terrain_for_viewer(level_delta=-1),
        )
        self._debug_viewer.register_key(
            "L",
            on_press=lambda: self._change_terrain_for_viewer(type_delta=+1),
        )

        # Activate goal marker visualisation (respects cfg.debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # Always subscribe camera follow callback, independent of debug_vis toggle.
        # Uses weakref.proxy to avoid extending the env's lifetime via the closure.
        self._camera_follow_handle = (
            omni.kit.app.get_app_interface()
            .get_post_update_event_stream()
            .create_subscription_to_pop(
                lambda event, obj=weakref.proxy(self): obj._camera_follow_callback(event)
            )
        )

        self.init_done = True

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # Clear goals registry before terrain generation so we get a clean set for this env
        _parkour_mesh_terrains.PARKOUR_GOALS_REGISTRY.clear()
        # Use capturing subclass to intercept the trimesh for edge-mask computation.
        # The original class_type is restored after construction so cfg is not permanently mutated.
        _original_class_type = self.cfg.terrain.class_type
        self.cfg.terrain.class_type = _CapturingTerrainImporter
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.cfg.terrain.class_type = _original_class_type
        # Build world-frame goals map from registry populated by terrain functions
        self._build_terrain_goals_map(_parkour_mesh_terrains.PARKOUR_GOALS_REGISTRY)
        # ---- Build Genesis-style edge mask from captured terrain mesh ----
        self._build_edge_mask()
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _build_terrain_goals_map(self, registry: list) -> None:
        """Build a world-frame goals map ``(num_rows, num_cols, num_goals, 3)`` from the registry.

        Each parkour terrain function appends ``(goals_local, origin_local)`` to
        ``_parkour_mesh_terrains.PARKOUR_GOALS_REGISTRY`` during terrain generation.
        Both arrays are in the pre-centering local frame (same coordinate system).

        World-frame conversion:
          ``world_goal = terrain_origins[row, col] + (local_goal - local_origin)``
        The subtraction is invariant under all rigid translations applied by TerrainGenerator,
        so this formula holds regardless of curriculum or cell-placement offsets.

        Entry order depends on curriculum flag in TerrainGeneratorCfg:
          curriculum=True  → col-major: call k → (row=k % num_rows, col=k // num_rows)
          curriculum=False → row-major: call k → (row=k // num_cols, col=k % num_cols)

        On success sets ``self._terrain_goals_world`` as Tensor[num_rows, num_cols, num_goals, 3].
        On failure (size mismatch) sets ``self._terrain_goals_world = None`` → straight-line fallback.
        """
        num_rows = self.cfg.terrain.terrain_generator.num_rows
        num_cols = self.cfg.terrain.terrain_generator.num_cols
        num_goals = self.cfg.num_goals
        curriculum = getattr(self.cfg.terrain.terrain_generator, "curriculum", False)

        if len(registry) != num_rows * num_cols:
            print(
                f"[Parkour] WARN: PARKOUR_GOALS_REGISTRY size {len(registry)} ≠ "
                f"{num_rows * num_cols} (expected num_rows*num_cols). "
                "Falling back to straight-line goals."
            )
            self._terrain_goals_world = None
            return

        # terrain_origins shape: (num_rows, num_cols, 3), already in world frame (torch.Tensor)
        t_origins = self._terrain.terrain_origins.cpu().numpy()  # [num_rows, num_cols, 3]
        goals_map = np.zeros((num_rows, num_cols, num_goals, 3), dtype=np.float32)

        for k, (local_goals, local_origin) in enumerate(registry):
            if curriculum:
                # col-major: outer=col loop, inner=row loop
                col = k // num_rows
                row = k % num_rows
            else:
                # row-major: np.unravel_index(k, (num_rows, num_cols))
                row = k // num_cols
                col = k % num_cols

            world_origin = t_origins[row, col]  # [3], world frame
            # world_goal = world_origin + (local_goal - local_origin)
            delta = local_goals[:num_goals] - local_origin[np.newaxis, :]  # [num_goals, 3]
            n = min(num_goals, len(local_goals))
            goals_map[row, col, :n] = world_origin + delta[:n]
            if n < num_goals:
                # Pad remaining with last valid goal
                goals_map[row, col, n:] = goals_map[row, col, n - 1]

        self._terrain_goals_world = torch.tensor(goals_map, dtype=torch.float32, device=self.device)
        sample = goals_map[0, 0, :2]  # first two goals of cell (0,0) for logging
        print(f"[Parkour] Goals map ready: shape={goals_map.shape}, cell(0,0) goals[:2]={sample}")

    def _build_edge_mask(self) -> None:
        """Build Genesis-style x_edge_mask from the captured terrain trimesh.

        Called once from ``_setup_scene`` after terrain construction.  The resulting
        ``self.x_edge_mask`` is a bool tensor of shape (n_x, n_y) on ``self.device``.
        ``self._edge_mask_origin`` holds the world-frame lower-left corner (x, y) of the
        grid, and ``self._edge_mask_inv_scale`` is 1 / horizontal_scale.

        World → grid index conversion (used in ``_get_rewards`` feet_edge):
            idx_x = round((feet_x - origin_x) / scale - 0.5) = floor((feet_x - origin_x) / scale)
            idx_y = round((feet_y - origin_y) / scale - 0.5) = floor((feet_y - origin_y) / scale)

        If the terrain mesh was not captured (e.g. mesh import failed), falls back to an
        all-False mask so training continues but feet_edge reward is zero.
        """
        import time

        captured_mesh = getattr(self._terrain, "_captured_trimesh", None)
        if captured_mesh is None:
            print("[Parkour] WARN: _build_edge_mask: no captured trimesh, x_edge_mask all-False.")
            self.x_edge_mask = torch.zeros(1, 1, dtype=torch.bool, device=self.device)
            self._edge_mask_height_field = torch.zeros(1, 1, dtype=torch.float32, device=self.device)
            self._edge_mask_origin = torch.zeros(2, dtype=torch.float32, device=self.device)
            h_scale_fallback = self.cfg.terrain.terrain_generator.horizontal_scale
            self._edge_mask_inv_scale = 1.0 / h_scale_fallback
            self._edge_mask_scale = h_scale_fallback
            return

        h_scale = self.cfg.terrain.terrain_generator.horizontal_scale
        num_rows = self.cfg.terrain.terrain_generator.num_rows
        num_cols = self.cfg.terrain.terrain_generator.num_cols
        tile_x = self.cfg.terrain.terrain_generator.size[0]
        tile_y = self.cfg.terrain.terrain_generator.size[1]
        terrain_size_x = tile_x * num_rows
        terrain_size_y = tile_y * num_cols

        # Lower-left corner (world frame) — same offset TerrainGenerator applies when centering.
        # transform[:2, -1] = [-size_x * num_rows / 2, -size_y * num_cols / 2]
        origin = np.array([-terrain_size_x / 2.0, -terrain_size_y / 2.0, 0.0])

        t0 = time.time()
        n_x = round(terrain_size_x / h_scale)
        n_y = round(terrain_size_y / h_scale)
        print(f"[Parkour] Building edge mask: {n_x}×{n_y} = {n_x * n_y:,} grid cells ...")
        edge_mask_np, height_field_np = _compute_edge_mask_from_terrain_mesh(
            terrain_mesh=captured_mesh,
            terrain_origin=origin,
            terrain_size_x=terrain_size_x,
            terrain_size_y=terrain_size_y,
            horizontal_scale=h_scale,
        )
        elapsed = time.time() - t0
        n_edges = int(edge_mask_np.sum())
        print(f"[Parkour] Edge mask built in {elapsed:.2f}s — {n_edges:,} / {n_x * n_y:,} edge cells "
              f"({100.0 * n_edges / (n_x * n_y):.1f}%)")

        self.x_edge_mask = torch.from_numpy(edge_mask_np).to(self.device)  # (n_x, n_y) bool
        self._edge_mask_height_field = torch.from_numpy(height_field_np).to(self.device, dtype=torch.float32)
        self._edge_mask_origin = torch.tensor(origin[:2], dtype=torch.float32, device=self.device)
        self._edge_mask_inv_scale = 1.0 / h_scale
        self._edge_mask_scale = h_scale
        _em_total = self.x_edge_mask.numel()
        _em_true = int(self.x_edge_mask.sum().item())
        _em_ratio = 100.0 * _em_true / max(_em_total, 1)
        print("\n========== [EDGE-MASK DEBUG] ==========")
        print(f"x_edge_mask shape: {tuple(self.x_edge_mask.shape)}")
        print(f"  total cells = {_em_total:,}, true (edge) cells = {_em_true:,}, "
              f"ratio = {_em_ratio:.3f}%  (healthy: 5–15%)")
        print(f"  origin (world LL corner): {self._edge_mask_origin.cpu().numpy()}")
        print(f"  inv_scale (cells/m): {self._edge_mask_inv_scale:.4f}  "
              f"(= 1 / h_scale = 1 / {self._edge_mask_scale:.4f})")
        print(f"  height_field: min={float(self._edge_mask_height_field.min()):.4f}, "
              f"max={float(self._edge_mask_height_field.max()):.4f}")
        print("=======================================\n")

    def _pre_physics_step(self, actions: torch.Tensor):
        # Time-based velocity command resampling (every resampling_time_s = 300 steps @ 0.02 s/step).
        # Each env is resampled independently when its per-env counter reaches the interval.
        self._time_since_command_resample += 1
        _resample_mask = self._time_since_command_resample >= self._command_resample_interval
        if _resample_mask.any():
            _resample_ids = _resample_mask.nonzero(as_tuple=False).squeeze(-1)
            self._resample_commands(_resample_ids)
            self._time_since_command_resample[_resample_ids] = 0

        self._previous_actions = self._actions.clone()  # Save previous BEFORE updating current
        self._actions = torch.clip(actions.clone(), -self.cfg.clip_actions, self.cfg.clip_actions)
        scaled = self._actions.clone()
        # Hip scale reduction: reduces hip abduction range
        scaled[:, self._hip_joint_ids] *= 0.5

        # Save last_last and last for action_smoothness rewards (Genesis original)
        self._last_last_processed_actions = self._last_processed_actions.clone()
        self._last_processed_actions = self._processed_actions.clone()

        # Compute current processed actions
        self._processed_actions = self.cfg.action_scale * scaled + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _init_env_goals(self, env_ids: torch.Tensor):
        """Initialize goal waypoints for specified envs.

        If ``_terrain_goals_world`` was built from the parkour terrain registry (W1 side-effect),
        goals are looked up by (terrain_level, terrain_type) → (row, col) in the goals map.
        Otherwise falls back to a straight-line of goals along +x from the terrain origin.
        """
        terrain_goals_world: torch.Tensor | None = getattr(self, "_terrain_goals_world", None)
        if terrain_goals_world is not None:
            # Use actual per-obstacle terrain goals (world frame)
            levels = self._terrain_levels[env_ids]   # [n] — row in terrain grid
            types = self._terrain_types[env_ids]     # [n] — col in terrain grid
            # terrain_goals_world: [num_rows, num_cols, num_goals, 3]
            self._env_goals[env_ids, : self.cfg.num_goals] = terrain_goals_world[levels, types]
        else:
            # Fallback: straight-line goals along +x from terrain origin
            print('no')
            return
            origins = self._terrain.env_origins[env_ids]  # [n, 3]
            goal_offsets = (
                torch.arange(1, self.cfg.num_goals + 1, device=self.device) * self.cfg.goal_distance
            )
            goal_x = origins[:, 0:1] + goal_offsets[None, :]
            goal_y = origins[:, 1:2].expand(-1, self.cfg.num_goals)
            goal_z = torch.full_like(goal_x, self.cfg.goal_z)
            self._env_goals[env_ids, : self.cfg.num_goals] = torch.stack([goal_x, goal_y, goal_z], dim=-1)

        # Repeat last goal for future goal observations (lookahead)
        if self.cfg.num_future_goal_obs > 0:
            last = self._env_goals[env_ids, self.cfg.num_goals - 1 : self.cfg.num_goals]  # [n, 1, 3]
            self._env_goals[env_ids, self.cfg.num_goals :] = last.expand(
                -1, self.cfg.num_future_goal_obs, -1
            )

    def _gather_cur_goals(self, future: int = 0) -> torch.Tensor:
        """Gather current (or future) goal for each env.

        Args:
            future: offset into future goals (0 = current, 1 = next, etc.)

        Returns:
            Goal positions [num_envs, 3]
        """
        # Create index tensor for gather: [num_envs, 1, 1] with cur_goal_idx + future
        goal_indices = (self._current_goal_idx[:, None, None] + future).expand(-1, -1, 3)
        # Clamp to valid range
        goal_indices = torch.clamp(goal_indices, max=self._env_goals.shape[1] - 1)
        # Gather goals
        goals = self._env_goals.gather(1, goal_indices).squeeze(1)  # [num_envs, 3]
        return goals

    def _update_goals(self):
        """Update goal tracking state every step.

        1. Check if current goal is reached (distance < threshold)
        2. If reached, increment reach_goal_timer
        3. After hold time, advance to next goal
        4. Compute relative goal position and yaw for obs
        """
        # Advance to next goal if hold time expired
        hold_time_steps = int(self.cfg.reach_goal_delay / self.step_dt)
        next_goal_mask = self._reach_goal_timer > hold_time_steps
        # Detect success: currently at the LAST goal and hold time just expired → episode complete.
        # We record this BEFORE advancing so _current_goal_idx stays at num_goals-1 (preserves gather
        # and viz semantics; _get_dones() consumes this flag and triggers reset).
        self._term_goal_reached = next_goal_mask & (self._current_goal_idx >= self.cfg.num_goals - 1)
        # Advance index for non-terminal goals; clamp keeps index valid for remaining steps.
        self._current_goal_idx[next_goal_mask] = torch.clamp(
            self._current_goal_idx[next_goal_mask] + 1,
            max=self.cfg.num_goals - 1,
        )
        self._reach_goal_timer[next_goal_mask] = 0

        # Check if current goal is reached
        base_xy = self._robot.data.root_link_pos_w[:, :2]
        self._cur_goals = self._gather_cur_goals(future=0)
        self._next_goals = self._gather_cur_goals(future=1)

        cur_goals_xy = self._cur_goals[:, :2]
        dist_to_cur_goal = torch.norm(base_xy - cur_goals_xy, dim=1)
        reached = dist_to_cur_goal < self.cfg.next_goal_threshold
        self._reach_goal_timer[reached] += 1

        # Compute relative target position (goal - robot)
        self._target_pos_rel = cur_goals_xy - base_xy
        next_goals_xy = self._next_goals[:, :2]
        self._next_target_pos_rel = next_goals_xy - base_xy

        # Compute target yaw (atan2 of normalized direction vector)
        norm_cur = torch.norm(self._target_pos_rel, dim=-1, keepdim=True)
        target_vec_cur_norm = self._target_pos_rel / (norm_cur + 1e-5)
        self._target_yaw = torch.atan2(target_vec_cur_norm[:, 1], target_vec_cur_norm[:, 0])

        norm_next = torch.norm(self._next_target_pos_rel, dim=-1, keepdim=True)
        target_vec_next_norm = self._next_target_pos_rel / (norm_next + 1e-5)
        self._next_target_yaw = torch.atan2(target_vec_next_norm[:, 1], target_vec_next_norm[:, 0])

    def _get_observations(self) -> dict:
        # Update goal waypoints every step (Task #2)
        self._update_goals()

        # Edge mask debug visualization (env 0 only; no-op when flag is False)
        if self.cfg.debug_vis_edge_mask:
            self._update_edge_mask_visualization()

        # Height scan: relative height difference (robot_z - ray_hit_z - 0.5)
        # WARNING (Change K): The comment below is STALE. The code still uses pos_w[:, 2]
        # which includes the RayCasterCfg offset=(0,0,20). This means the formula is
        # (base_z + 20) - terrain_z - 0.5 ≈ 19.84 → always clipped to 1.0 (dead channel).
        # The DEBUG block below will confirm whether this saturation is occurring.
        # Fix: replace pos_w[:, 2] with root_pos_w[:, 2] once the diagnostic confirms the bug.
        recently_reset = (self.episode_length_buf <= 1)           # [N] bool — env just reset this step
        do_global_refresh = (self.common_step_counter % 5 == 0)  # 10 Hz cadence gate (hardware constraint)

        if do_global_refresh or recently_reset.any():
            # Compute fresh values for ALL envs in one pass (sensor data is fresh post sim-step).
            scan_new = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self._height_scanner.data.ray_hits_w[..., 2] - 0.3
            ).clip(-1.0, 1.0)
            # Change E: compute relative yaw error (target_yaw - robot_heading), wrapped to [-π, π].
            # Genesis uses delta_yaw = target_yaw - self.yaw (robot heading), giving the policy direct
            # information about how much it needs to rotate. The previous absolute world-frame yaw gave
            # no heading reference — the policy could not compute turn direction/magnitude.
            # NOTE: self._target_yaw (absolute) is preserved unchanged for reward use at line ~632.
            yaw_raw = self._target_yaw - self._robot.data.heading_w           # [N] relative yaw error
            yaw_diff_new = torch.atan2(torch.sin(yaw_raw), torch.cos(yaw_raw))  # wrap to [-π, π]
            next_yaw_raw = self._next_target_yaw - self._robot.data.heading_w
            next_yaw_diff_new = torch.atan2(torch.sin(next_yaw_raw), torch.cos(next_yaw_raw))

            if do_global_refresh:
                # Normal 10 Hz cadence — update all envs at once (preserves hardware constraint).
                self._scan = scan_new.clone()
                self._yaw_diff = yaw_diff_new
                self._next_yaw_diff = next_yaw_diff_new
                # self._yaw_diff = yaw_raw
                # self._next_yaw_diff = next_yaw_raw
            else:
                # One-shot per-env patch: only recently-reset envs get their stale cache cleared.
                # Envs whose episode_length_buf > 1 keep their previous cached values unchanged.
                self._scan[recently_reset] = scan_new[recently_reset]
                self._yaw_diff[recently_reset] = yaw_diff_new[recently_reset]
                self._next_yaw_diff[recently_reset] = next_yaw_diff_new[recently_reset]
                # self._yaw_diff[recently_reset] = yaw_raw[recently_reset]
                # self._next_yaw_diff[recently_reset] = next_yaw_raw[recently_reset]

        # DEBUG: Height-scan multi-step diagnostic.
        # Fires at each step in _scan_debug_print_steps (at most once per step value per process).
        _step = int(self.common_step_counter)
        _step = 50000
        if _step in self._scan_debug_print_steps and _step not in self._scan_debug_already_printed:
            self._scan_debug_already_printed.add(_step)
            with torch.no_grad():
                # ── raw tensors ──────────────────────────────────────────────
                sensor_pos_z = self._height_scanner.data.pos_w[:, 2]       # (N,)   base_z + offset_z
                robot_base_z = self._robot.data.root_pos_w[:, 2]           # (N,)   true base_z
                ray_hits_w   = self._height_scanner.data.ray_hits_w        # (N, C, 3)  or (N, C) — defensive
                if ray_hits_w.dim() == 3:
                    ray_hit_z = ray_hits_w[..., 2]                         # (N, C)
                    # ray_starts_w is not exposed by RayCasterData.
                    # For vertical rays (attach_yaw_only=True + downward pattern),
                    # hit point xy == ray start xy, so we use hit xy as a proxy.
                    ray_starts_xy = ray_hits_w[..., :2]                    # (N, C, 2)
                else:
                    ray_hit_z     = ray_hits_w                             # (N, C) already
                    ray_starts_xy = None

                num_cells = ray_hit_z.shape[1]                             # C (e.g. 187)

                scan_raw = sensor_pos_z.unsqueeze(1) - ray_hit_z - 0.5    # (N, C) current formula pre-clip
                scan_alt = robot_base_z.unsqueeze(1) - ray_hit_z - 0.5    # (N, C) alt formula pre-clip
                scan_alt_clipped = scan_alt.clip(-1.0, 1.0)

                # ── inf / nan masks ──────────────────────────────────────────
                inf_mask = torch.isinf(ray_hit_z)                          # (N, C)
                nan_mask = torch.isnan(ray_hit_z)                          # (N, C)
                inf_count_per_env = inf_mask.sum(dim=1)                    # (N,)
                nan_total         = nan_mask.sum().item()
                inf_envs          = (inf_count_per_env > 0).sum().item()   # # envs with at least 1 inf
                n_show = min(4, self.num_envs)

                # ── safe finite statistics for ray_hit_z ─────────────────────
                finite_mask  = ~inf_mask & ~nan_mask
                if finite_mask.any():
                    rh_mean = ray_hit_z[finite_mask].mean().item()
                    rh_min  = ray_hit_z[finite_mask].min().item()
                    rh_max  = ray_hit_z[finite_mask].max().item()
                else:
                    rh_mean = rh_min = rh_max = float("nan")

                # ── robot status (env 0) ─────────────────────────────────────
                base_xy_0  = self._robot.data.root_pos_w[0, :2].cpu()
                goal_idx_0 = int(self._current_goal_idx[0].item())
                goal_xy_0  = self._cur_goals[0, :2].cpu()
                dist_0     = torch.norm(goal_xy_0 - base_xy_0).item()

                print(f"\n========== [HEIGHT-SCAN DEBUG @ step={_step}] ==========")
                print(f"num_envs={self.num_envs}, cells_per_env={num_cells}")

                # Robot status
                print(f"\n[Robot status — env 0]")
                print(f"  base xy: ({base_xy_0[0]:.3f}, {base_xy_0[1]:.3f}), "
                      f"goal idx: {goal_idx_0}, "
                      f"goal xy: ({goal_xy_0[0]:.3f}, {goal_xy_0[1]:.3f}), "
                      f"dist={dist_0:.3f}m")

                # Sensor z
                print(f"\n[Sensor z]")
                print(f"  sensor pos_w[:, 2] (base_z + offset_z): "
                      f"mean={sensor_pos_z.mean().item():.4f}, "
                      f"min={sensor_pos_z.min().item():.4f}, "
                      f"max={sensor_pos_z.max().item():.4f}")
                print(f"  robot root_pos_w[:, 2] (true base_z):   "
                      f"mean={robot_base_z.mean().item():.4f}, "
                      f"min={robot_base_z.min().item():.4f}, "
                      f"max={robot_base_z.max().item():.4f}")
                print(f"  implied offset (sensor - base):          "
                      f"{(sensor_pos_z - robot_base_z).mean().item():.4f}")

                # Scanner geometry (env 0)
                # ray_starts_xy = ray_hits_w[..., :2]: valid because rays are vertical,
                # so hit-point xy == ray-start xy regardless of whether the ray hit or not.
                print(f"\n[Scanner geometry — env 0]")
                if ray_starts_xy is not None:
                    rs0_xy = ray_starts_xy[0].cpu()                         # (C, 2)
                    base_xy_t = base_xy_0                                    # already cpu
                    c0   = rs0_xy[0]
                    cmid = rs0_xy[num_cells // 2]
                    cend = rs0_xy[num_cells - 1]
                    rel_cx = cmid[0].item() - base_xy_t[0].item()
                    rel_cy = cmid[1].item() - base_xy_t[1].item()
                    x_ext_min = (rs0_xy[:, 0] - base_xy_t[0]).min().item()
                    x_ext_max = (rs0_xy[:, 0] - base_xy_t[0]).max().item()
                    y_ext_min = (rs0_xy[:, 1] - base_xy_t[1]).min().item()
                    y_ext_max = (rs0_xy[:, 1] - base_xy_t[1]).max().item()
                    print(f"  ray start xy [env 0]: shape={tuple(rs0_xy.shape)} "
                          f"(derived from ray_hits_w xy — vertical rays)")
                    print(f"    cell[0]           xy = ({c0[0]:.3f}, {c0[1]:.3f})")
                    print(f"    cell[{num_cells // 2}]         xy = ({cmid[0]:.3f}, {cmid[1]:.3f})  <- approx center")
                    print(f"    cell[{num_cells - 1}]       xy = ({cend[0]:.3f}, {cend[1]:.3f})")
                    print(f"  scan center offset from base: (dx={rel_cx:+.3f}, dy={rel_cy:+.3f})")
                    print(f"  scan extent relative to base: x=[{x_ext_min:+.3f}, {x_ext_max:+.3f}], "
                          f"y=[{y_ext_min:+.3f}, {y_ext_max:+.3f}]")
                else:
                    print("  ray_starts_xy not available (ray_hits_w dim mismatch)")

                # Ray hits — inf/nan stats
                print(f"\n[Ray hits — inf/nan stats]")
                print(f"  ray_hits_w[..., 2] (finite only): "
                      f"mean={rh_mean:.4f}, min={rh_min:.4f}, max={rh_max:.4f}")
                inf_per = inf_count_per_env[:n_show].cpu().tolist()
                inf_per_str = ", ".join(f"env{i}={int(v)}/{num_cells}" for i, v in enumerate(inf_per))
                print(f"  inf count: {inf_per_str}  "
                      f"(all-env mean={inf_count_per_env.float().mean().item():.1f}, "
                      f"max={inf_count_per_env.max().item()})")
                print(f"  inf-containing env ratio: {inf_envs}/{self.num_envs} "
                      f"({100.0 * inf_envs / self.num_envs:.0f}%)")
                print(f"  nan count (total across all envs+cells): {nan_total}")

                # inf cell positions (env 0, first 10)
                print(f"\n[inf cell positions — env 0 (first 10)]")
                inf_idx_0 = inf_mask[0].nonzero(as_tuple=False).squeeze(-1)  # (K,)
                if inf_idx_0.numel() == 0:
                    print("  none — all rays hit mesh")
                else:
                    show_k = min(10, inf_idx_0.numel())
                    print(f"  total inf cells in env 0: {inf_idx_0.numel()}")
                    print(f"  inf cell indices (first {show_k}): {inf_idx_0[:show_k].cpu().tolist()}")
                    if ray_starts_xy is not None:
                        rs0_xy = ray_starts_xy[0].cpu()
                        base_xy_t = base_xy_0
                        for k in range(show_k):
                            ci = int(inf_idx_0[k].item())
                            cx, cy = rs0_xy[ci, 0].item(), rs0_xy[ci, 1].item()
                            dx = cx - base_xy_t[0].item()
                            dy = cy - base_xy_t[1].item()
                            print(f"    cell[{ci:3d}]: xy=({cx:.3f}, {cy:.3f})  "
                                  f"offset from base (dx={dx:+.3f}, dy={dy:+.3f})")

                # Scan statistics
                print(f"\n[Scan statistics]")
                # safe mean for scan_raw (may contain ±inf)
                scan_raw_fin = scan_raw[~torch.isinf(scan_raw) & ~torch.isnan(scan_raw)]
                scan_raw_mean = scan_raw_fin.mean().item() if scan_raw_fin.numel() > 0 else float("nan")
                print(f"  scan_raw pre-clip (current formula, finite only): "
                      f"mean={scan_raw_mean:.4f}, "
                      f"min={scan_raw.min().item():.4f}, max={scan_raw.max().item():.4f}")
                print(f"  scan post-clip:    "
                      f"mean={self._scan.mean().item():.4f}, "
                      f"min={self._scan.min().item():.4f}, "
                      f"max={self._scan.max().item():.4f}, "
                      f"unique={self._scan.unique().numel()}")
                sat_count = (self._scan == -1.0).sum(dim=1)                # (N,)
                sat_per = sat_count[:n_show].cpu().tolist()
                sat_per_str = ", ".join(f"env{i}={int(v)}" for i, v in enumerate(sat_per))
                print(f"  cells saturated at -1 (post-clip): {sat_per_str}")

                scan_alt_fin = scan_alt[~torch.isinf(scan_alt) & ~torch.isnan(scan_alt)]
                scan_alt_mean = scan_alt_fin.mean().item() if scan_alt_fin.numel() > 0 else float("nan")
                print(f"  scan_alt pre-clip  (alt formula, finite only):   "
                      f"mean={scan_alt_mean:.4f}, "
                      f"min={scan_alt.min().item():.4f}, max={scan_alt.max().item():.4f}")
                print(f"  scan_alt post-clip: "
                      f"mean={scan_alt_clipped.mean().item():.4f}, "
                      f"min={scan_alt_clipped.min().item():.4f}, "
                      f"max={scan_alt_clipped.max().item():.4f}, "
                      f"unique={scan_alt_clipped.unique().numel()}")

                # First n_show envs sample values
                print(f"\n[Sample cell values — first {n_show} envs, cells 0..9 post-clip (current)]")
                for i in range(n_show):
                    print(f"  env[{i}] = {self._scan[i, -50:].cpu().numpy()}")
                print(f"\n[Sample cell values — first {n_show} envs, cells 0..9 post-clip (alt)]")
                for i in range(n_show):
                    print(f"  env[{i}] = {scan_alt_clipped[i, -50:].cpu().numpy()}")

                print(f"=====================================================\n")
            # keep legacy flag consistent
            self._scan_debug_printed = True



        # Proprioceptive observations (Task #3)
        proprio = torch.cat(
            [
                self._yaw_diff[:, None],                                                   # 1 (delta_yaw: target - robot heading, wrapped)
                self._next_yaw_diff[:, None],                                              # 1 (delta_next_yaw)
                self._robot.data.projected_gravity_b,                                # 3
                self._commands[:, 0:1],                                              # 1
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,    # 12
                self._robot.data.joint_vel * 0.05,                                  # 12 (Change A: Genesis obs_scales["dof_vel"]=0.05; raw rad/s is ~20x too large)
                self._actions,                                                       # 12 (current actions taken this step)
            ],
            dim=-1,
        )
        # proprio dim = 3 + 1 + 1 + 1 + 12 + 12 + 12 = 42

        # Privileged observations: domain-randomized physical properties visible to the critic /
        # adaptation module but NOT the policy (RMA-style asymmetric AC).
        #
        # Composition (Option 1 — friction-only, 14 dims total):
        #   root_lin_vel_b  : (N, 3)  — world-lin-vel expressed in robot body frame
        #   root_ang_vel_b  : (N, 3)  — angular velocity in body frame
        #   foot_friction   : (N, 8)  — static+dynamic friction for 4 feet × 2 coefficients
        #                               sourced from EventManager-randomized robot material properties
        #
        # Mass is excluded: randomize_rigid_body_mass not yet wired in EventCfg → all envs share
        # the same default mass → zero information content. Extend here once mass randomization lands.
        #
        # EventManager mode note: if EventCfg uses mode="startup", get_material_properties() returns
        # the randomized values from startup; if mode="reset", values are refreshed each episode.
        # Either way, the tensor read here reflects the current per-env material state.
        #
        # get_material_properties() returns a CPU tensor (num_envs, num_shapes, 3):
        #   dim 2 → [static_friction, dynamic_friction, restitution]
        # _foot_shape_indices holds the FIRST shape index for each of the 4 foot links, computed
        # in __init__ via per-link RigidBodyView.max_shapes (events.py pattern). Go2 has 27 shapes
        # across 19 bodies; indexing by body_id directly would be wrong.
        # Restitution (dim 2) excluded — EventCfg randomize range is 0 → constant, zero info.
        _mat_all = (
            self._robot.root_physx_view.get_material_properties()
            .clone()
            .to(self.device)
        )  # (N, num_shapes, 3)
        foot_friction = _mat_all[:, self._foot_shape_indices, :2].reshape(self.num_envs, -1)  # (N, 8)
        priv = torch.cat(
            [
                self._robot.data.root_lin_vel_b,   # 3
                self._robot.data.root_ang_vel_b,   # 3
                foot_friction,                     # 8  (4 feet × [static, dynamic])
            ],
            dim=-1,
        )  # total: 14

        # Full critic observation: proprio + scan + priv
        critic_obs = torch.cat(
            [
                proprio,
                self._scan,
                priv,
            ],
            dim=-1,
        )

        # Update proprioceptive history ring buffer (shift and append new)
        proprio_for_history = proprio.clone()
        proprio_for_history[:, :2] = 0
        self._proprio_history = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([proprio_for_history] * self.cfg.history_len, dim=1),
                torch.cat([self._proprio_history[:, 1:], proprio_for_history.unsqueeze(1)], dim=1),
        )

        # Tick DebugViewer free-fly camera translation (no-op in headless or tracking mode).
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
            self._debug_viewer.update(self.step_dt)

        return {
            "policy": proprio,
            # "critic": critic_obs,
            "scan": self._scan,
            "priv": priv,
            "history": self._proprio_history,
        }

    def _get_rewards(self) -> torch.Tensor:
        # Save and update previous joint velocity for acceleration computation
        prev_joint_vel = self._prev_joint_vel.clone()
        self._prev_joint_vel = self._robot.data.joint_vel.clone()

        # Contact forces (Genesis original)
        net_contact_forces = self._contact_sensor.data.net_forces_w_history  # [N, hist, bodies, 3]

        # === Terrain-class mask for Genesis-style conditional reward multipliers ===
        # is_flat=1.0 for envs on flat terrain (TERRAIN_CLASS_FLAT=0), 0.0 otherwise.
        is_flat = (self._env_class == TERRAIN_CLASS_FLAT).float()
        is_non_flat = 1.0 - is_flat

        # === Goal-tracking rewards (Task #4: parkour-specific) ===
        # tracking_goal_vel: forward velocity projection onto goal direction (Genesis line 1519-1527)
        # NOTE: abs(commanded_speed)+clamp(min=0) variant — safer than Genesis raw signed proj which
        # penalizes the robot for exceeding commanded speed. This keeps gradient positive on overshoot.
        goal_dir_norm = torch.norm(self._target_pos_rel, dim=-1, keepdim=True)
        goal_dir = self._target_pos_rel / (goal_dir_norm + 1e-5)  # [N, 2]
        cur_vel_w = self._robot.data.root_lin_vel_w[:, :2]  # [N, 2] world frame velocity
        # cur_vel_w = self._robot.data.root_vel_w[:, :2]  # [N, 2] world frame velocity
        proj_forward = torch.sum(cur_vel_w * goal_dir, dim=-1)  # [N] velocity toward goal
        commanded_speed = torch.abs(self._commands[:, 0])  # [N] forward command magnitude
        tracking_goal_vel = torch.minimum(proj_forward, commanded_speed) / (commanded_speed + 1e-5)
        tracking_goal_vel = torch.where(
            commanded_speed > 1e-3, tracking_goal_vel, torch.zeros_like(tracking_goal_vel)
        )
        # tracking_goal_vel = tracking_goal_vel.clamp(min=0.0)

        # tracking_yaw: exponential decay from heading error (Genesis line 1451-1454)
        # No speed gating — stand-still local optimum is prevented by tracking_goal_vel (weight=1.5)
        # which only rewards forward velocity along the goal direction.
        heading = self._robot.data.heading_w  # [N] world frame yaw
        yaw_diff = self._target_yaw - heading  # [N]
        # yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        tracking_yaw = torch.exp(-torch.abs(yaw_diff))

        # === Velocity penalties (Genesis line 1437-1445) ===
        lin_vel_z_l2 = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # Genesis conditional: lin_vel_z penalized 10x less on non-flat (robot needs to move vertically)
        lin_vel_z_l2 = lin_vel_z_l2 * (is_flat + is_non_flat * 0.1)
        # Genesis conditional: ang_vel_xy penalized 2x less on flat (upright posture less critical there)
        ang_vel_xy_l2 = ang_vel_xy_l2 * (is_flat + is_non_flat * 0.5)

        # === Orientation penalty (Genesis line 1447-1449) ===
        orientation_l2 = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)
        # Genesis conditional: orientation not penalized on non-flat (robot must lean for obstacles)
        orientation_l2 = orientation_l2 * is_flat

        # === Joint acceleration penalty (Genesis line 1465-1470) ===
        dof_acc_l2 = torch.sum(
            torch.square((self._robot.data.joint_vel - prev_joint_vel) / self.step_dt), dim=1
        )

        # === Collision penalty (Genesis line 1540-1548) ===
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 0.1
        )
        collision = torch.sum(is_contact, dim=1).float()

        # === Action rate penalty (Genesis line 1506-1507) — L2 NORM (NOT sum of squares) ===
        action_rate_l2 = torch.norm(self._actions - self._previous_actions, dim=1)

        # === Delta torques (Genesis line 1509-1510) — NEW ===
        current_applied_torque = self._robot.data.applied_torque.clone()
        # print(current_applied_torque)
        delta_torques = torch.sum(torch.square(current_applied_torque - self._last_applied_torque), dim=1)
        self._last_applied_torque = current_applied_torque  # Update for next step

        # === Torque penalty (Genesis line 1511-1517) ===
        torques_l2 = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)

        # === Hip deviation penalty (Genesis line 1550-1558) ===
        hip_pos = torch.sum(
            torch.square(
                self._robot.data.joint_pos[:, self._hip_joint_ids]
                - self._robot.data.default_joint_pos[:, self._hip_joint_ids]
            ),
            dim=1,
        )

        # === Overall joint error penalty (Genesis line 1472-1486) ===
        dof_error_l2 = torch.sum(
            torch.square(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1
        )

        # === Feet stumble (Genesis line 1559-1573) ===
        feet_forces = net_contact_forces[:, 0, self._feet_ids]  # [N, 4, 3]
        feet_stumble = torch.any(
            torch.norm(feet_forces[..., :2], dim=-1) > 4.0 * torch.abs(feet_forces[..., 2]),
            dim=1,
        ).float()

        # === Feet edge — Genesis x_edge_mask port (Option B) ===
        # contact: current-step foot contact boolean (N, 4), threshold=2.0 N
        # Use only the most-recent physics substep (history index 0) to match Genesis semantics.
        # net_forces_w_history[:, 0] is the latest substep (contact_sensor_data.py:90).
        # Genesis ref: legged_env_parkour.py:937,943 — single get_links_net_contact_force() call.
        contact = torch.norm(net_contact_forces[:, 0, self._feet_ids], dim=-1) > 2.0
        # contact_filt: OR with last step to suppress 50 Hz false-negatives (matches Genesis two-sample OR).
        contact_filt = torch.logical_or(contact, self._last_contacts)
        self._last_contacts = contact
        # feet world XY positions: (N, 4, 2)
        feet_pos_w = self._robot.data.body_pos_w[:, self._feet_ids, :]  # (N, 4, 3)
        feet_xy = feet_pos_w[..., :2]  # (N, 4, 2)
        # World XY → grid index.
        # Grid cell (i, j) covers [origin + i*scale, origin + (i+1)*scale).
        # floor((x - origin) / scale) = round((x - origin) / scale - 0.5)
        # Using round() with the - 0.5 correction is equivalent to floor and is Genesis-faithful.
        feet_grid_xy = (
            (feet_xy - self._edge_mask_origin) * self._edge_mask_inv_scale - 0.5
        ).round().long()  # (N, 4, 2)
        feet_grid_xy[..., 0] = feet_grid_xy[..., 0].clamp(0, self.x_edge_mask.shape[0] - 1)
        feet_grid_xy[..., 1] = feet_grid_xy[..., 1].clamp(0, self.x_edge_mask.shape[1] - 1)
        # Lookup edge mask: (N, 4) bool
        feet_at_edge = self.x_edge_mask[feet_grid_xy[..., 0], feet_grid_xy[..., 1]]
        # Gate by contact and terrain level (Genesis: rew=0 on levels ≤ 3)
        feet_at_edge = contact_filt & feet_at_edge
        feet_edge = (self._terrain_levels > 3).float() * torch.sum(feet_at_edge.float(), dim=-1)

        # === Assemble all reward terms (B-aligned 14-term set) ===
        reward_values = {
            "tracking_goal_vel": tracking_goal_vel,
            "tracking_yaw": tracking_yaw,
            "lin_vel_z_l2": lin_vel_z_l2,        # *= 0.1 on non-flat (applied above)
            "ang_vel_xy_l2": ang_vel_xy_l2,       # *= 0.5 on non-flat (applied above)
            "orientation_l2": orientation_l2,      # zero on non-flat (applied above)
            "dof_acc_l2": dof_acc_l2,
            "collision": collision,
            "action_rate_l2": action_rate_l2,
            "delta_torques": delta_torques,
            "torques_l2": torques_l2,
            "hip_pos": hip_pos,
            "dof_error_l2": dof_error_l2,
            "feet_stumble": feet_stumble,
            "feet_edge": feet_edge,
        }

        # === Accumulate and scale rewards ===
        total_reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in reward_values.items():
            scaled = self.cfg.reward_scales[key] * self.step_dt * value
            self._episode_sums[key] += scaled
            total_reward += scaled
            self._last_reward_breakdown_env0[key] = float(scaled[0].detach().item())

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        # Base contact with ground
        self._term_base_contact = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 5.0,
            dim=1,
        )
        # Excessive tilt: |projected_gravity_xy|^2 > sin^2(1.5 rad) ≈ 0.997
        self._term_tilt = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1) > 0.99

        # Robot too low (fallen into terrain gap or flipped)
        self._term_low_height = self._robot.data.root_link_pos_w[:, 2] < self.cfg.termination_height

        # Goal-reached termination is set in _update_goals() (called from _get_observations()).
        # It fires when the robot holds position at the last waypoint long enough — a success event,
        # NOT a failure.  Use terminated=True (not time_out) so the value bootstrap is zero (episode
        # truly ends) rather than using the next-state value estimate.
        # Grace period applies only to failure conditions, not to goal success.
        terminated = self._term_base_contact | self._term_tilt | self._term_low_height
        grace = self.episode_length_buf < self.cfg.termination_grace_steps
        terminated = (terminated & ~grace) | self._term_goal_reached
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        all_envs_reset = env_ids is None or len(env_ids) == self.num_envs
        if all_envs_reset:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if all_envs_reset:
            # Spread resets to avoid training spikes
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset buffers
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._prev_joint_vel[env_ids] = 0.0
        self._current_goal_idx[env_ids] = 0
        self._last_contacts[env_ids] = False
        self._term_goal_reached[env_ids] = False

        # Reset processed actions (Genesis original)
        self._processed_actions[env_ids] = 0.0
        self._last_processed_actions[env_ids] = 0.0
        self._last_last_processed_actions[env_ids] = 0.0

        # Reset applied torques (Genesis original)
        self._last_applied_torque[env_ids] = 0.0

        # Reset goal tracking buffers (Task #2)
        self._reach_goal_timer[env_ids] = 0
        self._target_pos_rel[env_ids] = 0.0
        self._next_target_pos_rel[env_ids] = 0.0
        self._target_yaw[env_ids] = 0.0
        self._next_target_yaw[env_ids] = 0.0

        # Reset proprioceptive history (Task #3)
        self._proprio_history[env_ids] = 0.0

        # Terrain curriculum update before repositioning
        if self.cfg.terrain_curriculum:
            self._update_terrain_curriculum(env_ids)
            # Refresh env_class after curriculum update (future-proof; _terrain_types fixed in current impl)
            self._env_class[env_ids] = self._col_to_class[self._terrain_types[env_ids]]

        # Reinitialize goals after terrain curriculum changes (since terrain_origins may change)
        self._init_env_goals(env_ids)

        # Reset robot state at terrain origin
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        default_root_state[:, 2] += 0.05  # 0.05m physics-settling buffer only — NOT a free-fall drop
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # Resample velocity commands and reset per-env timer so the next time-based
        # resample fires exactly resampling_time_s after episode start.
        self._time_since_command_resample[env_ids] = 0
        self._resample_commands(env_ids)

        # Logging
        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
        # Decomposed termination causes (only env_ids that are being reset)
        self.extras["log"]["Episode_Termination/cause_base_contact"] = torch.count_nonzero(
            self._term_base_contact[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/cause_tilt"] = torch.count_nonzero(
            self._term_tilt[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/cause_low_height"] = torch.count_nonzero(
            self._term_low_height[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/cause_goal_reached"] = torch.count_nonzero(
            self._term_goal_reached[env_ids]
        ).item()
        # Mean episode length at reset
        self.extras["log"]["Episode_Length/mean_at_reset"] = self.episode_length_buf[env_ids].float().mean().item()
        self.extras["log"]["curriculum/mean_terrain_level"] = self._terrain_levels[env_ids].float().mean().item()
        # Per-terrain-type mean difficulty level — only logged for types that have at least one
        # env being reset in this batch (avoids NaN for inactive / zero-proportion types).
        _levels_reset: torch.Tensor = self._terrain_levels[env_ids].float()
        _class_reset: torch.Tensor = self._env_class[env_ids]
        for _class_id, _class_name in _TERRAIN_CLASS_NAMES.items():
            _mask: torch.Tensor = _class_reset == _class_id
            if _mask.any():
                self.extras["log"][f"curriculum/mean_terrain_level_{_class_name}"] = (
                    _levels_reset[_mask].mean().item()
                )

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Game-inspired terrain curriculum: advance on success, regress on failure."""
        # Skip on first reset (robot not yet initialized)
        if not getattr(self, "init_done", False):
            return

        # Keyboard-override bypass: envs flagged by _change_terrain_for_viewer keep their
        # forced level/type.  Clear the flag unconditionally first (even if env_ids is a
        # subset) so it never carries over into subsequent normal resets.
        skip_mask = self._skip_curriculum[env_ids]          # bool [n], True → bypass curriculum
        self._skip_curriculum[env_ids] = False              # always clear before returning

        # Refresh env_origins for bypassed envs so the robot spawns on the forced patch.
        # (env_origins was already set by _change_terrain_for_viewer, but we also update
        # here to keep the update path symmetric with the normal curriculum branch below.)
        skip_ids = env_ids[skip_mask]
        if skip_ids.numel() > 0:
            self._terrain.env_origins[skip_ids] = self._terrain.terrain_origins[
                self._terrain_levels[skip_ids], self._terrain_types[skip_ids]
            ]

        # Only run curriculum logic for envs that were NOT keyboard-overridden.
        env_ids = env_ids[~skip_mask]
        if env_ids.numel() == 0:
            return

        # Distance traveled from spawn origin during episode
        dis_to_origin = torch.norm(
            self._robot.data.root_link_pos_w[env_ids, :2] - self._terrain.env_origins[env_ids, :2],
            dim=1,
        )
        # Expected travel based on commanded velocity and episode length
        expected_dist = self._commands[env_ids, 0].abs() * self.max_episode_length_s
        move_up = dis_to_origin > 0.8 * expected_dist
        move_down = dis_to_origin < 0.4 * expected_dist

        max_level = self.cfg.terrain.terrain_generator.num_rows - 1
        self._terrain_levels[env_ids] += move_up.long() - move_down.long()
        # Wrap top level to random (inclusive of max_level), clip bottom at 0
        self._terrain_levels[env_ids] = torch.where(
            self._terrain_levels[env_ids] >= max_level,
            torch.randint_like(self._terrain_levels[env_ids], max_level + 1),
            torch.clamp(self._terrain_levels[env_ids], 0),
        )
        # Refresh env_origins so the robot spawns on the updated terrain patch
        self._terrain.env_origins[env_ids] = self._terrain.terrain_origins[
            self._terrain_levels[env_ids], self._terrain_types[env_ids]
        ]

    def _resample_commands(self, env_ids: torch.Tensor):
        n = len(env_ids)
        vx_lo: float = self.cfg.command_cfg["lin_vel_x_range"][0]
        vx_hi: float = self.cfg.command_cfg["lin_vel_x_range"][1]
        vy_lo: float = self.cfg.command_cfg["lin_vel_y_range"][0]
        vy_hi: float = self.cfg.command_cfg["lin_vel_y_range"][1]
        wz_lo: float = self.cfg.command_cfg["ang_vel_range"][0]
        wz_hi: float = self.cfg.command_cfg["ang_vel_range"][1]
        self._commands[env_ids, 0] = torch_rand_float(vx_lo, vx_hi, (n,), self.device)
        self._commands[env_ids, 1] = torch_rand_float(vy_lo, vy_hi, (n,), self.device)
        self._commands[env_ids, 2] = torch_rand_float(wz_lo, wz_hi, (n,), self.device)

    # ------------------------------------------------------------------
    # Debug visualisation
    # ------------------------------------------------------------------

    def _set_debug_vis_impl(self, debug_vis: bool):
        """Create / toggle goal-waypoint sphere markers.

        Two marker sets are created lazily on first activation:
          - ``cur_goal_visualizer``  : single large red sphere  (radius 0.15)
          - ``future_goal_visualizer``: up to 7 small orange spheres (radius 0.08)

        Both prim paths live under ``/Visuals/Parkour/`` to avoid conflicts.
        """
        if debug_vis:
            if not hasattr(self, "cur_goal_visualizer"):
                # Current goal — large red sphere
                cur_cfg = SPHERE_MARKER_CFG.copy()
                cur_cfg.prim_path = "/Visuals/Parkour/cur_goal"
                cur_cfg.markers["sphere"].radius = 0.15
                cur_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                )
                self.cur_goal_visualizer = VisualizationMarkers(cur_cfg)

            if not hasattr(self, "future_goal_visualizer"):
                # Future goals — small orange spheres
                fut_cfg = SPHERE_MARKER_CFG.copy()
                fut_cfg.prim_path = "/Visuals/Parkour/future_goals"
                fut_cfg.markers["sphere"].radius = 0.08
                fut_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.5, 0.0)
                )
                self.future_goal_visualizer = VisualizationMarkers(fut_cfg)

            if not hasattr(self, "_heading_arrow_visualizer"):
                # Heading direction — cyan sphere dots (A-project parkour_event pattern)
                heading_cfg = SPHERE_MARKER_CFG.copy()
                heading_cfg.prim_path = "/Visuals/Parkour/HeadingDots"
                heading_cfg.markers["sphere"].radius = 0.05
                heading_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 1.0)  # cyan
                )
                self._heading_arrow_visualizer = VisualizationMarkers(heading_cfg)

            if not hasattr(self, "_target_yaw_arrow_visualizer"):
                # Target yaw direction — red sphere dots (A-project parkour_event pattern)
                target_cfg = SPHERE_MARKER_CFG.copy()
                target_cfg.prim_path = "/Visuals/Parkour/TargetDots"
                target_cfg.markers["sphere"].radius = 0.05
                target_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)  # red
                )
                self._target_yaw_arrow_visualizer = VisualizationMarkers(target_cfg)

            self.cur_goal_visualizer.set_visibility(True)
            self.future_goal_visualizer.set_visibility(True)
            self._heading_arrow_visualizer.set_visibility(True)
            self._target_yaw_arrow_visualizer.set_visibility(True)
        else:
            if hasattr(self, "cur_goal_visualizer"):
                self.cur_goal_visualizer.set_visibility(False)
            if hasattr(self, "future_goal_visualizer"):
                self.future_goal_visualizer.set_visibility(False)
            if hasattr(self, "_heading_arrow_visualizer"):
                self._heading_arrow_visualizer.set_visibility(False)
            if hasattr(self, "_target_yaw_arrow_visualizer"):
                self._target_yaw_arrow_visualizer.set_visibility(False)

    # ------------------------------------------------------------------
    # Edge mask debug visualisation
    # ------------------------------------------------------------------

    # Small z offset to lift markers above terrain surface to avoid z-fighting with mesh
    _EDGE_VIS_Z_OFFSET: float = 0.01

    def _setup_edge_mask_visualizer(self):
        """Lazily create green sphere markers for edge mask debug visualization.

        Mirrors the ``_set_debug_vis_impl`` pattern: create once on first call, guarded by
        ``hasattr``.  Prim path lives under ``/Visuals/Parkour/`` alongside goal markers.
        """
        if hasattr(self, "_edge_mask_visualizer"):
            return
        edge_cfg = SPHERE_MARKER_CFG.copy()
        edge_cfg.prim_path = "/Visuals/Parkour/edge_mask"
        edge_cfg.markers["sphere"].radius = 0.02
        edge_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0)  # green
        )
        self._edge_mask_visualizer = VisualizationMarkers(edge_cfg)

    def _get_active_viewer_env_id(self) -> int:
        """Return the env id currently tracked by the viewport camera controller.

        Reads ``viewport_camera_controller.cfg.env_index``, which is the live value
        updated by the UI/keyboard env-switching handler via ``set_view_env_index()``.
        Falls back to env 0 when running headless (controller is None) or when the
        index is out of range.
        """
        vcc = getattr(self, "viewport_camera_controller", None)
        if vcc is None:
            return 0
        idx = int(vcc.cfg.env_index)
        if 0 <= idx < self.num_envs:
            return idx
        return 0

    def _update_edge_mask_visualization(self):
        """Update green sphere markers at edge cells within radius of the active viewer env.

        Algorithm:
        1. Determine the env id currently tracked by the viewport camera (dynamic).
        2. Convert that env's base XY to grid indices.
        3. Clamp a window of ±radius_cells around that index to grid bounds.
        4. Slice ``x_edge_mask`` within the window (no full-mask scan).
        5. Extract nonzero cell indices, convert to world XY + terrain-height Z.
        6. Call ``visualize()`` on the marker set.

        Called every policy step from ``_get_observations`` when
        ``cfg.debug_vis_edge_mask`` is True.  Uses only vectorised torch/numpy ops —
        no Python-level loops over cells.
        """
        self._setup_edge_mask_visualizer()

        # --- active viewer env (dynamically tracks keyboard/UI env switching) ---
        active_env_id = self._get_active_viewer_env_id()

        # --- grid metadata ---
        origin_x = self._edge_mask_origin[0].item()
        origin_y = self._edge_mask_origin[1].item()
        inv_scale = self._edge_mask_inv_scale
        h_scale = self._edge_mask_scale
        n_x, n_y = self.x_edge_mask.shape

        # --- active env base position → grid index ---
        base_xy = self._robot.data.root_pos_w[active_env_id, :2]  # (2,) world frame
        base_x = base_xy[0].item()
        base_y = base_xy[1].item()
        base_ix = int(round((base_x - origin_x) * inv_scale - 0.5))
        base_iy = int(round((base_y - origin_y) * inv_scale - 0.5))

        # --- window bounds (clamped to grid) ---
        radius_cells = int(round(self.cfg.debug_vis_edge_mask_radius_m * inv_scale))
        ix_lo = max(0, base_ix - radius_cells)
        ix_hi = min(n_x, base_ix + radius_cells + 1)
        iy_lo = max(0, base_iy - radius_cells)
        iy_hi = min(n_y, base_iy + radius_cells + 1)

        if ix_lo >= ix_hi or iy_lo >= iy_hi:
            # Window entirely outside terrain — hide markers
            self._edge_mask_visualizer.set_visibility(False)
            return

        # --- slice mask and height field within window ---
        window_mask = self.x_edge_mask[ix_lo:ix_hi, iy_lo:iy_hi]  # (wx, wy) bool
        window_hf = self._edge_mask_height_field[ix_lo:ix_hi, iy_lo:iy_hi]  # (wx, wy) float32

        # --- extract edge cell indices within window ---
        edge_local = window_mask.nonzero(as_tuple=False)  # (E, 2): local (ix, iy) offsets

        # --- print on env transition (replaces once-only _edge_vis_logged guard) ---
        last_active = getattr(self, "_last_active_env_id", None)
        if last_active != active_env_id:
            # window world-range: helps diagnose x/y swap if window extends in wrong direction
            win_x_min = origin_x + ix_lo * h_scale
            win_x_max = origin_x + ix_hi * h_scale
            win_y_min = origin_y + iy_lo * h_scale
            win_y_max = origin_y + iy_hi * h_scale
            # robot yaw: if forward is +x, yaw≈0; if forward is +y, yaw≈±π/2
            robot_quat = self._robot.data.root_quat_w[active_env_id]
            _, _, yaw_tensor = math_utils.euler_xyz_from_quat(robot_quat.unsqueeze(0))
            yaw_deg = float(yaw_tensor[0]) * 57.2958
            print(
                f"[edge_mask viz] active_env={active_env_id}, "
                f"base_xy=[{base_x:.2f}, {base_y:.2f}], "
                f"robot_yaw_deg={yaw_deg:.1f}, "
                f"window_x=[{win_x_min:.2f}, {win_x_max:.2f}], "
                f"window_y=[{win_y_min:.2f}, {win_y_max:.2f}], "
                f"edge_count_in_window={int(edge_local.shape[0])}"
            )
            self._last_active_env_id = active_env_id

        if edge_local.shape[0] == 0:
            # No edge cells in window — hide markers
            self._edge_mask_visualizer.set_visibility(False)
            return

        # --- convert local offsets → global grid indices ---
        global_ix = edge_local[:, 0] + ix_lo  # (E,)
        global_iy = edge_local[:, 1] + iy_lo  # (E,)

        # --- grid indices → world XY (cell centres) ---
        world_x = origin_x + (global_ix.float() + 0.5) * h_scale  # (E,)
        world_y = origin_y + (global_iy.float() + 0.5) * h_scale  # (E,)

        # --- terrain height at each edge cell ---
        world_z = window_hf[edge_local[:, 0], edge_local[:, 1]] + self._EDGE_VIS_Z_OFFSET  # (E,)

        # --- stack into (E, 3) translations tensor ---
        translations = torch.stack([world_x, world_y, world_z.to(world_x.dtype)], dim=-1)

        self._edge_mask_visualizer.set_visibility(True)
        self._edge_mask_visualizer.visualize(translations=translations)

    def _camera_follow_callback(self, _event):
        """Update side camera every render frame, regardless of debug_vis state."""
        del _event
        if self.viewport_camera_controller is None:
            return
        # When DebugViewer free-fly is active, the helper drives camera translation via
        # update(dt); skip parkour's robot-yaw tracking to avoid conflicting writes.
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None and self._debug_viewer.is_free_fly_camera:
            return
        # ---- camera follow ------------------------------------------------
        viewer_idx = self.cfg.viewer.env_index

        robot_pos = self._robot.data.root_pos_w[viewer_idx]       # (3,)
        robot_quat = self._robot.data.root_quat_w[viewer_idx]     # (4,) wxyz

        # Extract yaw from quaternion (euler_xyz_from_quat returns roll, pitch, yaw)
        _, _, yaw = math_utils.euler_xyz_from_quat(robot_quat.unsqueeze(0))
        yaw = yaw[0]  # scalar tensor

        # Configured side-view offset (local robot frame): (0, -2.5, 0.8)
        eye_cfg = self.cfg.viewer.eye    # tuple (x, y, z)
        look_cfg = self.cfg.viewer.lookat  # tuple (x, y, z)

        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)

        # Rotate XY component of eye offset by yaw; Z unchanged
        eye_x = cos_y * eye_cfg[0] - sin_y * eye_cfg[1]
        eye_y = sin_y * eye_cfg[0] + cos_y * eye_cfg[1]
        eye_z = eye_cfg[2]

        # lookat (0, 0, 0.3) — XY are zero so rotation is a no-op, but applied anyway
        look_x = cos_y * look_cfg[0] - sin_y * look_cfg[1]
        look_y = sin_y * look_cfg[0] + cos_y * look_cfg[1]
        look_z = look_cfg[2]

        robot_pos_np = robot_pos.detach().cpu().numpy()
        eye_world = robot_pos_np + np.array([eye_x.item(), eye_y.item(), eye_z], dtype=float)
        look_world = robot_pos_np + np.array([look_x.item(), look_y.item(), look_z], dtype=float)

        # update_view_location stores new offsets as default_cam_eye/lookat so the
        # controller's own tick (fired in the same post-update stream) also uses them.
        self.viewport_camera_controller.update_view_location(eye=eye_world, lookat=look_world)

    def _debug_vis_callback(self, _event):
        """Update goal-waypoint sphere markers and yaw arrows every render frame (only when debug_vis is True)."""
        del _event
        # ---- goal markers -------------------------------------------------
        if not hasattr(self, "cur_goal_visualizer"):
            return

        # Current goal for ALL envs — shape (num_envs, 3)
        cur_goals = self._gather_cur_goals(future=0)
        self.cur_goal_visualizer.visualize(translations=cur_goals)

        # Future goals for ALL envs — flatten (num_envs, num_total, 3) -> (num_envs * num_total, 3)
        all_goals = self._env_goals.reshape(-1, 3)
        self.future_goal_visualizer.visualize(translations=all_goals)

        # ---- yaw direction sphere dots (A parkour_event pattern) ----------
        if not hasattr(self, "_heading_arrow_visualizer"):
            return

        arrow_num = 8        # dots per direction
        start_offset = 0.3   # m — distance from robot to first dot
        spacing = 0.15       # m — spacing between consecutive dots; total length = 0.3 + 7*0.15 = 1.35m

        base_xy = self._robot.data.root_pos_w[:, :2]   # (N, 2)
        base_z = self._robot.data.root_pos_w[:, 2:3]   # (N, 1)

        # Heading direction: unit vector from robot's current world-frame yaw
        heading_yaw = self._robot.data.heading_w       # (N,) rad
        heading_dir = torch.stack(
            [torch.cos(heading_yaw), torch.sin(heading_yaw)], dim=-1
        )  # (N, 2)

        # Target direction: normalized target_pos_rel toward current goal
        target_norm = torch.norm(self._target_pos_rel, dim=-1, keepdim=True)
        target_dir = self._target_pos_rel / (target_norm + 1e-5)  # (N, 2)

        heading_dots = []
        target_dots = []
        for i in range(arrow_num):
            dist = start_offset + i * spacing
            h_xy = base_xy + dist * heading_dir   # (N, 2)
            t_xy = base_xy + dist * target_dir    # (N, 2)
            heading_dots.append(torch.cat([h_xy, base_z], dim=-1))
            target_dots.append(torch.cat([t_xy, base_z], dim=-1))

        heading_positions = torch.cat(heading_dots, dim=0)   # (N*arrow_num, 3)
        target_positions = torch.cat(target_dots, dim=0)
        self._heading_arrow_visualizer.visualize(translations=heading_positions)
        self._target_yaw_arrow_visualizer.visualize(translations=target_positions)

    # ------------------------------------------------------------------
    # DebugViewer callbacks
    # ------------------------------------------------------------------

    def __del__(self):
        """Clean up DebugViewer and camera-follow subscription."""
        try:
            if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
                self._debug_viewer.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_camera_follow_handle"):
                self._camera_follow_handle = None
        except Exception:
            pass

    def _toggle_contact_print(self):
        """Toggle the contact-debug visualisation registered with DebugViewer."""
        new_state = not self._debug_viewer.vis_is_enabled("contact_print")
        self._debug_viewer.set_debug_vis("contact_print", new_state)
        print(f"[parkour] Contact debug: {'ON' if new_state else 'OFF'}")

    def _change_terrain_for_viewer(self, level_delta: int = 0, type_delta: int = 0) -> None:
        """Force a terrain level/type change on the currently-viewed env and reset it.

        Only the single env tracked by the viewport camera is affected; all other envs
        continue running without interruption.

        Args:
            level_delta: +1 to increase difficulty row, -1 to decrease.  Clamped to
                         [0, num_rows - 1].
            type_delta:  Column offset; wraps around via mod num_cols.
        """
        if not hasattr(self, "_terrain") or self._terrain is None:
            print("[parkour] _change_terrain_for_viewer: terrain not yet initialized, skipping.")
            return

        env_id = self._get_active_viewer_env_id()

        # terrain_origins shape: (num_rows, num_cols, 3) — derive grid size from tensor directly
        # TerrainImporterCfg does not expose num_rows/num_cols as top-level attributes.
        origins = getattr(self._terrain, "terrain_origins", None)
        if origins is None or origins.ndim != 3:
            print("[parkour] terrain_origins not available or unexpected shape — skip terrain change.")
            return
        num_rows = int(origins.shape[0])
        num_cols = int(origins.shape[1])

        cur_level = int(self._terrain_levels[env_id].item())
        cur_type = int(self._terrain_types[env_id].item())

        new_level = max(0, min(cur_level + level_delta, num_rows - 1))
        new_type = (cur_type + type_delta) % num_cols

        # Write forced values into the TerrainImporter tensors (which _terrain_levels /
        # _terrain_types alias directly — no copy needed).
        self._terrain_levels[env_id] = new_level
        self._terrain_types[env_id] = new_type

        # Update env_origin so the robot spawns on the correct patch.
        # _update_terrain_curriculum would normally do this, but it is bypassed below.
        self._terrain.env_origins[env_id] = self._terrain.terrain_origins[new_level, new_type]

        # Flag this env so _update_terrain_curriculum does NOT overwrite the forced values.
        self._skip_curriculum[env_id] = True

        # Reset only the one env.  curriculum flag is cleared inside _update_terrain_curriculum.
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)

        terrain_name = _TERRAIN_CLASS_NAMES.get(int(self._env_class[env_id].item()), "unknown")
        print(
            f"[parkour] env {env_id} → terrain level={new_level}/{num_rows - 1}, "
            f"type={new_type} ({terrain_name})"
        )

    # ------------------------------------------------------------------
    # Contact debug helper
    # ------------------------------------------------------------------

    def _print_contact_debug(self):
        """Print contact bodies and forces for the viewer env to stdout.

        Invoked by the DebugViewer post-update callback when the "contact_print"
        debug-vis is enabled (toggled by the P key).  The cfg.debug_print_contacts
        early-return guard has been removed — call gating is now handled by
        DebugViewer.vis_is_enabled("contact_print").

        Output format (per active contact body, |F| >= 1 N):
            [parkour env=<idx> step=<N>] contacts:
              [contact_all] <body_name>   F=(fx, fy, fz)  |F|=<mag> N
        """
        env_idx = int(getattr(self.cfg.viewer, "env_index", 0))
        env_idx = max(0, min(env_idx, self.num_envs - 1))

        body_names = self._contact_sensor.body_names
        if not body_names:
            return

        # net_forces_w shape: (num_envs, num_bodies, 3) — current step forces.
        forces = self._contact_sensor.data.net_forces_w[env_idx]  # (B, 3)
        magnitudes = forces.norm(dim=-1)                           # (B,)

        threshold = 1.0  # N — ignore simulation noise below this magnitude
        step = int(self.episode_length_buf[env_idx].item())
        lines = [f"[parkour env={env_idx} step={step}] contacts:"]
        any_contact = False

        for b_idx, bname in enumerate(body_names):
            mag = float(magnitudes[b_idx].item())
            if mag < threshold or bname[-4:] == 'foot':
                continue
            
            fx = float(forces[b_idx, 0].item())
            fy = float(forces[b_idx, 1].item())
            fz = float(forces[b_idx, 2].item())
            lines.append(
                f"  [contact_all] {bname:<24s} F=({fx:+7.2f},{fy:+7.2f},{fz:+7.2f}) |F|={mag:7.2f} N"
            )
            any_contact = True

        if any_contact:
            print("\n".join(lines))

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
from isaaclab.markers.config import SPHERE_MARKER_CFG
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.terrains import TerrainImporter
from isaaclab.terrains.trimesh import mesh_terrains as _parkour_mesh_terrains
from isaaclab.utils import math as math_utils

from isaaclab_tasks.direct._common import DebugViewer, DebugViewerCfg

from .parkour_env_cfg import TERRAIN_CLASS_FLAT, ParkourEnvCfg
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
        print(f"[edge_mask] origin={self._edge_mask_origin.tolist()}, "
              f"mask_shape={tuple(self.x_edge_mask.shape)}, "
              f"true_count={int(self.x_edge_mask.sum().item())}, "
              f"hf_min={float(self._edge_mask_height_field.min()):.3f}, "
              f"hf_max={float(self._edge_mask_height_field.max()):.3f}")

    def _pre_physics_step(self, actions: torch.Tensor):
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
        # BUGFIX (Change K): pos_w[:, 2] includes the RayCasterCfg offset=(0,0,20), so using it
        # directly would yield ~(base_z + 20) - terrain_z - 0.5 ≈ 19.84 → always clipped to 1.0.
        # Use root_pos_w[:, 2] (robot base z, no sensor offset) instead.
        scan = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)

        # Change E: compute relative yaw error (target_yaw - robot_heading), wrapped to [-π, π].
        # Genesis uses delta_yaw = target_yaw - self.yaw (robot heading), giving the policy direct
        # information about how much it needs to rotate. The previous absolute world-frame yaw gave
        # no heading reference — the policy could not compute turn direction/magnitude.
        # NOTE: self._target_yaw (absolute) is preserved unchanged for reward use at line ~632.
        yaw_diff = self._target_yaw - self._robot.data.heading_w          # [N] relative yaw error
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))  # wrap to [-π, π]
        next_yaw_diff = self._next_target_yaw - self._robot.data.heading_w
        next_yaw_diff = torch.atan2(torch.sin(next_yaw_diff), torch.cos(next_yaw_diff))

        # Proprioceptive observations (Task #3)
        proprio = torch.cat(
            [
                self._robot.data.projected_gravity_b,                                # 3
                self._commands[:, 0:1],                                              # 1
                yaw_diff[:, None],                                                   # 1 (delta_yaw: target - robot heading, wrapped)
                next_yaw_diff[:, None],                                              # 1 (delta_next_yaw)
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
                scan,
                priv,
            ],
            dim=-1,
        )

        # Update proprioceptive history ring buffer (shift and append new)
        self._proprio_history = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([proprio] * self.cfg.history_len, dim=1),
                torch.cat([self._proprio_history[:, 1:], proprio.unsqueeze(1)], dim=1),
        )

        # Tick DebugViewer free-fly camera translation (no-op in headless or tracking mode).
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
            self._debug_viewer.update(self.step_dt)

        return {
            "policy": proprio,
            "critic": critic_obs,
            "scan": scan,
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
        proj_forward = torch.sum(cur_vel_w * goal_dir, dim=-1)  # [N] velocity toward goal
        commanded_speed = torch.abs(self._commands[:, 0])  # [N] forward command magnitude
        tracking_goal_vel = torch.minimum(proj_forward, commanded_speed) / (commanded_speed + 1e-5)
        tracking_goal_vel = torch.where(
            commanded_speed > 1e-3, tracking_goal_vel, torch.zeros_like(tracking_goal_vel)
        )
        tracking_goal_vel = tracking_goal_vel.clamp(min=0.0)

        # tracking_yaw: exponential decay from heading error (Genesis line 1451-1454)
        heading = self._robot.data.heading_w  # [N] world frame yaw
        yaw_diff = self._target_yaw - heading  # [N]
        yaw_diff = torch.atan2(torch.sin(yaw_diff), torch.cos(yaw_diff))
        tracking_yaw = torch.exp(-torch.abs(yaw_diff))

        # === Velocity tracking (exponential) (Genesis line 1426-1435) ===
        lin_vel_error = torch.sum(
            torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1
        )
        tracking_lin_vel_xy_exp = torch.exp(-lin_vel_error / self.cfg.tracking_sigma)

        ang_vel_z_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        tracking_ang_vel_z_exp = torch.exp(-ang_vel_z_error / self.cfg.tracking_sigma)

        # === Velocity penalties (Genesis line 1437-1445) ===
        lin_vel_z_l2 = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # Genesis conditional: lin_vel_z penalized 10x less on non-flat (robot needs to move vertically)
        lin_vel_z_l2 = lin_vel_z_l2 * (is_flat + 0.1 * is_non_flat)
        # Genesis conditional: ang_vel_xy penalized 2x less on flat (upright posture less critical there)
        ang_vel_xy_l2 = ang_vel_xy_l2 * (0.5 * is_flat + is_non_flat)

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
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        collision = torch.sum(is_contact, dim=1).float()

        # === Action rate penalty (Genesis line 1506-1507) — L2 NORM (NOT sum of squares) ===
        action_rate_l2 = torch.norm(self._actions - self._previous_actions, dim=1)

        # === Delta torques (Genesis line 1509-1510) — NEW ===
        current_applied_torque = self._robot.data.applied_torque.clone()
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
        # Genesis conditional: dof_error penalized 10x more on flat (nominal posture expected there)
        dof_error_l2 = dof_error_l2 * (10.0 * is_flat + is_non_flat)

        # === Base height penalty (flat terrain only) (Genesis: base_height reward) ===
        # Penalizes deviation from nominal stance height; zeroed on non-flat where height varies.
        base_height = torch.square(
            self._robot.data.root_link_pos_w[:, 2] - self._terrain.env_origins[:, 2] - self.cfg.base_height_target
        ) * is_flat

        # === Feet stumble (Genesis line 1559-1573) ===
        feet_forces = net_contact_forces[:, 0, self._feet_ids]  # [N, 4, 3]
        feet_stumble = torch.any(
            torch.norm(feet_forces[..., :2], dim=-1) > 4.0 * torch.abs(feet_forces[..., 2]),
            dim=1,
        ).float()

        # === Feet edge — Genesis x_edge_mask port (Option B) ===
        # contact: current-step foot contact boolean (N, 4), threshold=2.0 N
        contact = torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 2.0
        # contact_filt: OR with last step to suppress 50 Hz false-negatives
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

        # === Termination penalty (Genesis line 1588-1594) — NEW ===
        # Penalize early termination: base_contact | tilt | low_height
        termination = (self._term_base_contact | self._term_tilt | self._term_low_height).float()

        # === Feet dragging (Genesis line 1596-1608) — NEW ===
        # Penalize feet sliding horizontally while in contact
        feet_lin_vel = self._robot.data.body_link_lin_vel_w[:, self._feet_ids, :2]  # [N, 4, 2]
        feet_speed = torch.norm(feet_lin_vel, dim=-1)  # [N, 4]
        feet_dragging = torch.sum(
            feet_speed * contact_filt.float() * (feet_speed > self.cfg.dragging_velocity_threshold).float(),
            dim=1,
        )

        # === Action smoothness 1 (Genesis line 1575-1579) — NEW ===
        # Penalize position target changes when last action was nonzero
        last_action_mask = (self._last_processed_actions != 0).float()
        diff_1 = torch.square(self._processed_actions - self._last_processed_actions) * last_action_mask
        action_smoothness_1 = torch.sum(diff_1, dim=1)

        # === Action smoothness 2 (Genesis line 1581-1586) — NEW ===
        # Penalize 2nd-order action changes
        last_last_action_mask = (self._last_last_processed_actions != 0).float()
        diff_2 = torch.square(self._processed_actions - 2 * self._last_processed_actions + self._last_last_processed_actions)
        diff_2 = diff_2 * last_action_mask * last_last_action_mask
        action_smoothness_2 = torch.sum(diff_2, dim=1)

        # === Assemble all reward terms (Genesis original + parkour extensions) ===
        reward_values = {
            "tracking_goal_vel": tracking_goal_vel,
            "tracking_yaw": tracking_yaw,
            "tracking_lin_vel_xy_exp": tracking_lin_vel_xy_exp,
            "tracking_ang_vel_z_exp": tracking_ang_vel_z_exp,
            "lin_vel_z_l2": lin_vel_z_l2,        # *= 0.1 on non-flat (applied above)
            "ang_vel_xy_l2": ang_vel_xy_l2,       # *= 0.5 on flat (applied above)
            "orientation_l2": orientation_l2,      # zero on non-flat (applied above)
            "dof_acc_l2": dof_acc_l2,
            "collision": collision,
            "action_rate_l2": action_rate_l2,
            "delta_torques": delta_torques,
            "torques_l2": torques_l2,
            "hip_pos": hip_pos,
            "dof_error_l2": dof_error_l2,          # *= 10.0 on flat (applied above)
            "feet_stumble": feet_stumble,
            "feet_edge": feet_edge,
            "termination": termination,
            "feet_dragging": feet_dragging,
            "action_smoothness_1": action_smoothness_1,
            "action_smoothness_2": action_smoothness_2,
            "base_height": base_height,            # flat-only; scale=0.0 by default (disabled)
        }

        # === Accumulate and scale rewards ===
        total_reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in reward_values.items():
            scaled = self.cfg.reward_scales[key] * self.step_dt * value
            self._episode_sums[key] += scaled
            total_reward += scaled

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

        # Resample velocity commands
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

    def _update_terrain_curriculum(self, env_ids: torch.Tensor):
        """Game-inspired terrain curriculum: advance on success, regress on failure."""
        # Skip on first reset (robot not yet initialized)
        if not getattr(self, "init_done", False):
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

            self.cur_goal_visualizer.set_visibility(True)
            self.future_goal_visualizer.set_visibility(True)
        else:
            if hasattr(self, "cur_goal_visualizer"):
                self.cur_goal_visualizer.set_visibility(False)
            if hasattr(self, "future_goal_visualizer"):
                self.future_goal_visualizer.set_visibility(False)

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
        """Update goal-waypoint sphere markers every render frame (only when debug_vis is True)."""
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

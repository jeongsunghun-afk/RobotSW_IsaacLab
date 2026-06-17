# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Parkour Demo environment — thin subclass of Go2ParkourImitationEnv.

Adds two live-injection hooks for interactive demo (Option B, v4 plan):
  - Hook 1 (vx): _pre_physics_step override — after super(), in playground mode
    writes _commands[:,0]=_demo_vx BEFORE the proprio cat so vx (col 5) is
    consistent in BOTH obs["policy"] and _proprio_history (col 5 is NOT zeroed
    at parkour_env.py:1031; a post-super patch would desync them).
  - Hook 2 (yaw): _get_observations override — after super(), in playground mode
    overwrites _yaw_diff/_next_yaw_diff and patches obs["policy"][...,0:2].
    Safe for history because _proprio_history zeroes cols [:2] (parkour_env.py:1031).

Goal bypass (AC-C3):
  In playground mode _current_goal_idx is clamped to < num_goals-1 every step
  so _term_goal_reached (parkour_env.py:640) never fires. Reward logs are
  polluted by the frozen waypoint but the demo runs inference only — no gradients.

set_demo_tile(class_id, level):
  Maps class→representative column via inverse of _col_to_class LUT, level→row,
  then calls _change_terrain_for_viewer-style forced assignment + _reset_idx.
  Uses env index [0] (num_envs=1 fixed for demo). Do NOT use type_delta cycling
  (R9: cycles columns, not classes).

AMP extras (amp_obs / terminal_amp_obs) pass through super() unmodified.

Obs invariant: assert obs["policy"].shape[-1] == 42 at __init__ (AC-X2).
"""

from __future__ import annotations

import math

import torch

from .parkour_imitation_env import Go2ParkourImitationEnv
from .parkour_imitation_env_cfg import ParkourImitationEnvCfg

try:
    from isaaclab.sensors import Camera  # type: ignore[import]

    _CAMERA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CAMERA_AVAILABLE = False


class Go2ParkourDemoEnv(Go2ParkourImitationEnv):
    """Thin demo subclass of Go2ParkourImitationEnv.

    Adds _demo_mode / _demo_vx / _demo_yaw state and two injection hooks.
    All training-time buffers and obs layout are preserved byte-identical.

    Modes:
        "test"       — identical to normal play; resampler runs, goals advance.
        "playground" — vx clamped to slider, yaw overridden, goal advance blocked.
    """

    cfg: ParkourImitationEnvCfg

    # ------------------------------------------------------------------ #
    # Scene setup — add follow camera after parent sensors are registered
    # ------------------------------------------------------------------ #

    def _setup_scene(self) -> None:
        """Call parent _setup_scene (robot + contact_sensor + height_scanner + terrain),
        then register the follow camera sensor if cfg.follow_camera is defined.

        Registration order mirrors the parent pattern:
            self.<attr> = SensorClass(self.cfg.<field>)
            self.scene.sensors["<key>"] = self.<attr>
        This must happen BEFORE scene.clone_environments() — the parent already
        calls clone_environments() as its last step, so super() is called first
        and the camera is appended after (clone has already run by then).
        Because Camera is a prim-level sensor that attaches to an existing prim
        rather than requiring pre-clone registration, appending post-super is safe
        (same pattern used by CameraSensor in other IsaacLab direct envs).
        # 확인 필요: 만약 특정 빌드에서 Camera가 clone 전 등록을 요구하면
        # super() 호출을 분리해야 한다. 현재는 post-super 등록으로 처리.
        """
        super()._setup_scene()

        follow_cam_cfg = getattr(self.cfg, "follow_camera", None)
        if follow_cam_cfg is not None and _CAMERA_AVAILABLE:
            self.follow_camera = Camera(follow_cam_cfg)
            self.scene.sensors["follow_camera"] = self.follow_camera

    # ------------------------------------------------------------------ #
    # Initialisation
    # ------------------------------------------------------------------ #

    def __init__(self, cfg: ParkourImitationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # headless: parent _camera_follow_callback (parkour_env.py:1671) checks
        # `self.viewport_camera_controller is None` before returning early.
        # DirectRLEnv only assigns this attribute in GUI mode; headless leaves it absent,
        # causing AttributeError at teardown when the app-event callback fires.
        # Guard: guarantee the attribute exists so the None-check always succeeds.
        # Timing is safe — the subscription (parkour_env.py:376-379) posts to the Omniverse
        # post-update event stream, which only ticks after __init__ returns; the callback
        # cannot fire during super().__init__(), so this assignment precedes the first tick.
        if not hasattr(self, "viewport_camera_controller"):
            self.viewport_camera_controller = None

        # Demo state — plain Python scalars, not tensors (num_envs=1 fixed for demo).
        # No buffer shape change → no _reset_idx initialisation required.
        self._demo_mode: str = "test"  # "test" | "playground"
        self._demo_vx: float = 0.5  # forward speed command [m/s], clamped by slider
        self._demo_yaw: float = 0.0  # yaw_diff command [rad], wrapped to [-π, π]

        # Build class→representative-column inverse LUT from _col_to_class (parkour_env.py:159).
        # _col_to_class: (num_cols,) int64 — _col_to_class[col] = class_id
        # _class_to_col[class_id] = first column whose class == class_id
        self._class_to_col: dict[int, int] = {}
        col_to_class_cpu = self._col_to_class.cpu().tolist()
        for col, cls in enumerate(col_to_class_cpu):
            if cls not in self._class_to_col:
                self._class_to_col[cls] = col  # first representative column for this class

        # AC-X2: assert obs invariance at startup.
        # single_observation_space["policy"] may be a gymnasium.spaces.Box; check .shape[-1].
        # Expected layout: 1 yaw_diff + 1 next_yaw_diff + 3 gravity + 1 vx + 12 joint_pos
        #                  + 12 joint_vel + 12 actions + 4 contact = 46
        _EXPECTED_POLICY_DIM = 46
        policy_dim = self.single_observation_space["policy"].shape[-1]
        assert policy_dim == _EXPECTED_POLICY_DIM, (
            f"Go2ParkourDemoEnv: obs invariance FAIL — policy obs dim={policy_dim}, expected {_EXPECTED_POLICY_DIM}. "
            "A parent change altered the policy observation. Investigate before running the demo."
        )

    # ------------------------------------------------------------------ #
    # Hook 1 — vx injection (MUST run before proprio cat in _get_observations)
    # ------------------------------------------------------------------ #

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        """Call super (which runs _resample_commands at parkour_env.py:553-561),
        then in playground mode override _commands[:,0] with the slider value.

        This runs BEFORE _get_observations/_proprio_history build, so vx appears
        consistently in obs["policy"] col 5 AND _proprio_history col 5 (un-zeroed,
        parkour_env.py:1030-1031). A post-super patch of obs["policy"] would desync
        them — this is the architect must-fix (v2 plan).
        """
        super()._pre_physics_step(actions)
        if self._demo_mode == "playground":
            # Override resampler write: clamp to training vx range to avoid OOD (R4).
            vx_lo: float = self.cfg.command_cfg["lin_vel_x_range"][0]
            vx_hi: float = self.cfg.command_cfg["lin_vel_x_range"][1]
            clamped_vx = float(max(vx_lo, min(vx_hi, self._demo_vx)))
            self._commands[:, 0] = clamped_vx

    # ------------------------------------------------------------------ #
    # Hook 2 — yaw injection + goal bypass (runs AFTER super's 10 Hz refresh)
    # ------------------------------------------------------------------ #

    def _get_observations(self) -> dict:
        """Call super() (runs _update_goals + 10 Hz yaw refresh at :704-708),
        then in playground mode:
          1. Overwrite _yaw_diff / _next_yaw_diff with the slider value (wrapped).
          2. Patch obs["policy"][..., 0:2] to match.
          3. Hold _current_goal_idx < num_goals-1 to prevent goal termination.

        Yaw-via-post-patch is safe because _proprio_history zeroes cols [:2]
        (parkour_env.py:1031), so the un-injected yaw never enters history.
        Do NOT patch col 5 (vx) here — that is Hook 1's responsibility.
        """
        obs = super()._get_observations()

        if self._demo_mode == "playground":
            # ---- yaw injection ----
            raw_yaw = float(self._demo_yaw)
            # wrap to [-π, π]
            yw = math.atan2(math.sin(raw_yaw), math.cos(raw_yaw))
            yw_tensor = torch.tensor(yw, dtype=torch.float32, device=self.device)

            self._yaw_diff[:] = yw_tensor
            self._next_yaw_diff[:] = yw_tensor
            # patch obs vector: cols 0 and 1 are yaw_diff and next_yaw_diff (parkour_env.py:923-924)
            obs["policy"][:, 0] = yw_tensor
            obs["policy"][:, 1] = yw_tensor

            # ---- goal bypass (AC-C3) ----
            # _term_goal_reached fires when _current_goal_idx >= num_goals-1 (parkour_env.py:640).
            # Hold at 0 so goal-termination never triggers in playground mode.
            # _update_goals (called in super) may have incremented the index; clamp it back.
            self._current_goal_idx[:] = 0

        return obs

    # ------------------------------------------------------------------ #
    # _reset_idx — document playground goal-bypass state at reset time
    # ------------------------------------------------------------------ #

    def _camera_follow_callback(self, _event) -> None:
        """Override parent to guard against teardown AttributeError.

        At app close the Omniverse event callback fires after __dict__ is partially torn down.
        The parent's ``if self.viewport_camera_controller is None`` check raises AttributeError
        because the attribute itself may no longer exist at that point.  Using getattr with a
        default makes the guard robust even when the object is mid-teardown.
        """
        if getattr(self, "viewport_camera_controller", None) is None:
            return
        try:
            super()._camera_follow_callback(_event)
        except ReferenceError:
            # robot/physx view torn down at app close — callback fired during teardown
            return

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """Call super()._reset_idx then ensure playground goal-bypass state is consistent.

        _demo_mode, _demo_vx, _demo_yaw are plain Python scalars (not tensors) —
        num_envs=1 fixed for demo — so no per-env reset is needed for them.

        _current_goal_idx is a tensor owned by the parent env. In playground mode
        _get_observations clamps it to 0 every step (goal-bypass, AC-C3). We mirror
        that intent here at reset time so freshly-reset envs start with idx=0
        regardless of what _update_goals may have written inside super()._reset_idx.
        """
        super()._reset_idx(env_ids)
        if self._demo_mode == "playground" and hasattr(self, "_current_goal_idx"):
            # Clamp only the resetting envs (env_ids) to index 0, matching the
            # per-step clamp in _get_observations (self._current_goal_idx[:] = 0).
            self._current_goal_idx[env_ids] = 0

    # ------------------------------------------------------------------ #
    # set_demo_tile — force env [0] onto a specific (class, level) tile
    # ------------------------------------------------------------------ #

    def set_demo_tile(self, class_id: int, level: int) -> None:
        """Force env index [0] (num_envs=1) onto the specified obstacle class and difficulty level.

        Maps class_id → representative column via the inverse of _col_to_class (R9: do NOT
        use type_delta cycling — it walks columns, not classes). Maps level → row, clamped to
        valid range. Then applies forced assignment + _reset_idx, mirroring the logic of
        _change_terrain_for_viewer (parkour_env.py:1783-1832).

        After this call _reset_idx re-reads _terrain_goals_world (already built in _setup_scene)
        via _init_env_goals — no re-registration of PARKOUR_GOALS_REGISTRY is needed (gap note,
        v4 plan: _init_env_goals only performs a lookup on the already-built map).

        Args:
            class_id: Terrain class index (0=flat, 1=hurdle, 2=step, 3=gap, 4=stair).
            level:    Difficulty row index, clamped to [0, num_rows-1].
        """
        if not hasattr(self, "_terrain") or self._terrain is None:
            print("[demo] set_demo_tile: terrain not yet initialised, skipping.")
            return

        origins = getattr(self._terrain, "terrain_origins", None)
        if origins is None or origins.ndim != 3:
            print("[demo] set_demo_tile: terrain_origins unavailable — skip.")
            return

        num_rows = int(origins.shape[0])
        num_cols = int(origins.shape[1])

        # class → column (R9 mitigation: use precomputed LUT, not type_delta cycling)
        if class_id not in self._class_to_col:
            print(f"[demo] set_demo_tile: class_id={class_id} not found in LUT. "
                  f"Available classes: {sorted(self._class_to_col.keys())}. Skipping.")
            return
        new_col = self._class_to_col[class_id]

        # level → row, clamped
        new_row = max(0, min(level, num_rows - 1))

        env_id = 0  # num_envs=1 fixed for demo (gap note, v4 plan)

        # Force assignment into TerrainImporter tensors (aliased by _terrain_levels/_terrain_types)
        self._terrain_levels[env_id] = new_row
        self._terrain_types[env_id] = new_col

        # Update env_origin so the robot spawns on the correct patch
        self._terrain.env_origins[env_id] = self._terrain.terrain_origins[new_row, new_col]

        # Flag env so _update_terrain_curriculum does NOT overwrite forced values
        self._skip_curriculum[env_id] = True

        # Reset only env [0]
        env_ids = torch.tensor([env_id], device=self.device, dtype=torch.long)
        self._reset_idx(env_ids)

        from isaaclab_tasks.direct.parkour.parkour_env_cfg import (
            TERRAIN_CLASS_FLAT,
            TERRAIN_CLASS_GAP,
            TERRAIN_CLASS_HURDLE,
            TERRAIN_CLASS_STAIR,
            TERRAIN_CLASS_STEP,
        )
        _names = {
            TERRAIN_CLASS_FLAT: "flat",
            TERRAIN_CLASS_HURDLE: "hurdle",
            TERRAIN_CLASS_STEP: "step",
            TERRAIN_CLASS_GAP: "gap",
            TERRAIN_CLASS_STAIR: "stair",
        }
        terrain_name = _names.get(class_id, f"class_{class_id}")
        print(f"[demo] env {env_id} → class={class_id} ({terrain_name}), "
              f"level={new_row}/{num_rows - 1}, col={new_col}/{num_cols - 1}")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Demo-only environment configurations for Go2 Parkour interactive demo.

Two cfg classes are provided:

ParkourDemoEnvCfg
    "test" mode — standard 5-type × 11-difficulty terrain grid, num_envs=1.
    Inherits PARKOUR_TERRAINS_CFG unchanged.  Attach a follow camera for rendering.

ParkourPlaygroundEnvCfg
    "playground" mode — single scatter tile (num_rows=1, num_cols=1) built by
    parkour_demo_terrains.parkour_playground_scatter_terrain.  num_envs=1.
    num_goals=NUM_PLAYGROUND_GOALS (8) must match the stub count the scatter
    function registers (BINDING CONTRACT MAJOR-1 in parkour_demo_terrains.py).

Camera note
-----------
Both cfgs add a follow camera (CameraCfg) attached to the robot base with a
fixed rear-above offset.  This is demo-only; training cfgs are unmodified.
The sensor is declared as ``follow_camera`` so env code can find it by name.
If ``CameraCfg`` is not available in the target IsaacLab build, remove the
``follow_camera`` field and drive the viewport via DebugViewer instead.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils.configclass import configclass

from .parkour_demo_terrains import NUM_PLAYGROUND_GOALS, ParkourPlaygroundScatterTerrainCfg
from .parkour_imitation_env_cfg import ParkourImitationEnvCfg

# ---------------------------------------------------------------------------
# Try to import CameraCfg — graceful fallback if sensor module unavailable.
# ---------------------------------------------------------------------------
try:
    from isaaclab.sensors import CameraCfg  # type: ignore[import]

    _CAMERA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CAMERA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Try to import TerrainGeneratorCfg for playground terrain replacement.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ParkourDemoEnvCfg — "test" mode, full 5×11 training terrain grid
# ---------------------------------------------------------------------------


@configclass
class ParkourDemoEnvCfg(ParkourImitationEnvCfg):
    """Demo env cfg for "test" mode: identical terrain to training, single env.

    Inherits all fields from ParkourImitationEnvCfg unchanged.
    Overrides:
        num_envs = 1  (demo/visualisation only)

    Adds:
        follow_camera (CameraCfg) — rear-above camera attached to the robot base.
    """

    # Single env for interactive demo
    num_envs: int = 1

    if _CAMERA_AVAILABLE:
        # Follow camera: 0.5 m behind, 0.3 m above base, looking forward.
        # ``prim_path`` uses {ENV_REGEX_NS} — resolved to env_0 at runtime.
        # offset=(−0.5, 0, 0.3) in base frame (x-forward convention).
        # 확인 필요: prim_path 내 base 링크 이름은 로봇 URDF에 따라 다를 수 있음.
        follow_camera: CameraCfg = CameraCfg(  # type: ignore[name-defined]
            prim_path="/World/envs/env_.*/FollowCam",
            update_period=0.0,  # every physics step
            data_types=["rgb"],
            width=640,
            height=480,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(  # type: ignore[name-defined]
                # OffsetCfg retained as required field; ignored at runtime — pose set via set_world_poses each frame.
                pos=(-1.8, 0.0, 0.8),
                rot=(0.8192, 0.0, -0.5736, 0.0),
                convention="ros",
            ),
        )


# ---------------------------------------------------------------------------
# ParkourPlaygroundEnvCfg — "playground" mode, single scatter tile
# ---------------------------------------------------------------------------


@configclass
class ParkourPlaygroundEnvCfg(ParkourImitationEnvCfg):
    """Demo env cfg for "playground" mode: one scatter tile with 5 obstacle types.

    Replaces the training terrain generator with a single-entry scatter terrain
    (ParkourPlaygroundScatterTerrainCfg, num_rows=1, num_cols=1, curriculum=False).

    BINDING CONTRACT (MAJOR-1): num_goals must equal NUM_PLAYGROUND_GOALS (8) so
    that _build_terrain_goals_map (parkour_env.py:435) finds exactly 1 registry entry
    with 8 stub waypoints. Mismatch → _env_goals uninitialised → silent bug.

    Overrides:
        num_envs = 1
        terrain_num_rows = 1
        terrain_num_cols = 1
        terrain_curriculum = False
        terrain_size = (30.0, 20.0)  — wide scatter tile (matches scatter cfg default)
        num_goals = NUM_PLAYGROUND_GOALS (8)

    __post_init__ replaces tg.sub_terrains with the single scatter sub-terrain
    after calling super().__post_init__() which applies the base terrain overrides.
    """

    # Single env for interactive demo
    num_envs: int = 1

    # --- Terrain grid: single 1×1 tile, no curriculum ---
    terrain_num_rows: int = 1
    terrain_num_cols: int = 1
    terrain_curriculum: bool = False
    terrain_difficulty_range: tuple[float, float] = (0.5, 0.5)  # fixed mid-range for playground
    terrain_max_init_level: int = 0  # only one row

    # Wide scatter tile to match ParkourPlaygroundScatterTerrainCfg.size default
    terrain_size: tuple[float, float] = (30.0, 20.0)

    # Border around the single tile — smaller is fine for a demo
    terrain_border_width: float = 5.0

    # num_goals must equal NUM_PLAYGROUND_GOALS so goals map is populated correctly.
    # This field is consumed by parkour_env.py when building _env_goals.
    num_goals: int = NUM_PLAYGROUND_GOALS  # = 8

    if _CAMERA_AVAILABLE:
        # Same follow camera as ParkourDemoEnvCfg — chase view rear-above.
        follow_camera: CameraCfg = CameraCfg(  # type: ignore[name-defined]
            prim_path="/World/envs/env_.*/FollowCam",
            update_period=0.0,
            data_types=["rgb"],
            width=640,
            height=480,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.1, 1.0e5),
            ),
            offset=CameraCfg.OffsetCfg(  # type: ignore[name-defined]
                # OffsetCfg retained as required field; ignored at runtime — pose set via set_world_poses each frame.
                pos=(-1.8, 0.0, 0.8),
                rot=(0.8192, 0.0, -0.5736, 0.0),
                convention="ros",
            ),
        )

    def __post_init__(self):
        # 1. Run ParkourImitationEnvCfg.__post_init__ — applies size/rows/cols/border/
        #    horizontal_scale/vertical_scale/curriculum/difficulty_range overrides to tg,
        #    then applies sub_terrain proportions (which are irrelevant here but harmless).
        super().__post_init__()

        # 2. Replace sub_terrains with a single scatter sub-terrain entry.
        #    TerrainGenerator will call parkour_playground_scatter_terrain exactly once
        #    (num_rows=1, num_cols=1 → 1 tile) and append exactly 1 registry entry.
        tg = self.terrain.terrain_generator  # TerrainGeneratorCfg instance (already patched)

        scatter_cfg = ParkourPlaygroundScatterTerrainCfg(
            proportion=1.0,  # 100 % of tiles are the scatter terrain
            size=self.terrain_size,
            num_goals=self.num_goals,
        )

        # Replace sub_terrains entirely — discard the 5 training sub-terrains.
        tg.sub_terrains = {"parkour_playground": scatter_cfg}

        # curriculum=False already applied by super(); reinforce here for safety.
        tg.curriculum = False
        tg.num_rows = self.terrain_num_rows
        tg.num_cols = self.terrain_num_cols

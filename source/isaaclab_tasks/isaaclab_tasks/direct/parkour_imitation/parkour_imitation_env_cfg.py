# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Go2 Parkour-Imitation hybrid environment.

Inherits all fields from ParkourEnvCfg (terrain, robot, sensors, rewards, etc.)
and adds AMP integration parameters.

AMP is applied only to flat sub-terrain envs (TERRAIN_CLASS_FLAT = 0).
Spawn distribution uses parkour's default uniform terrain assignment unchanged.

Terrain override fields
-----------------------
The following fields are exposed at ParkourImitationEnvCfg level so terrain geometry
can be adjusted without touching parkour_env_cfg.py or the global PARKOUR_TERRAINS_CFG
constant.  All values default to the current PARKOUR_TERRAINS_CFG values so the base
behaviour is unchanged.  __post_init__ applies them to self.terrain.terrain_generator
after the parent actuator setup has run.

Quick-change examples::

    cfg = ParkourImitationEnvCfg()
    # Smaller grid for fast smoke runs
    cfg.terrain_num_rows = 5
    cfg.terrain_num_cols = 20
    # Shift AMP flat ratio higher (more flat envs for motion imitation)
    cfg.terrain_sub_terrain_proportions = {
        "parkour_flat": 0.4,
        "parkour_hurdle": 0.15,
        "parkour_step": 0.15,
        "parkour_gap": 0.15,
        "parkour_stair": 0.15,
    }
"""

from isaaclab.utils import configclass

from isaaclab_tasks.direct.parkour.parkour_env_cfg import ParkourEnvCfg


@configclass
class ParkourImitationEnvCfg(ParkourEnvCfg):
    """Configuration for the Go2 Parkour-Imitation hybrid environment.

    Extends ParkourEnvCfg with AMP discriminator parameters. All parkour fields
    (obs_space=42, terrain_cfg, reward_scales, sensors, actuators) are inherited unchanged.

    AMP discriminator observation layout (49-dim per step × 10 history = 490-dim):
        dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel(3) +
        root_ang_vel(3) + foot_pos_local(12) + root_rot_tan_norm(6) = 49
        root_rot_tan_norm(6): heading-relative 6D rotation feature (MimicKit 방식).

    Spawn distribution: parkour's default uniform terrain assignment is used unchanged.
    AMP applies only to envs spawned on parkour_flat (flat_env_mask) — both reward and
    discriminator gradient are masked. See parkour_imitation_env.py:_flat_env_mask.
    """

    # ── AMP integration ──────────────────────────────────────────────────────
    # Path to reference motion(s), relative to the parkour_imitation package directory.
    # May point to a single .pkl file OR a directory containing multiple .pkl files.
    # If a directory, Go2MotionLib auto-discovers all *.pkl entries (sorted) and
    # samples from them with uniform weighting by default. See motion_lib.py:243-247.
    # Backward-compat: set to "imitation/go2/go2_trot0.pkl" to use a single file.
    amp_motion_pkl: str = "imitation/go2"

    # amp_weight is the runner's reward fusion coefficient and is configured exclusively
    # in agents/rsl_rl_amp_cfg.py::Go2ParkourImitationPPOAMPRunnerCfg.amp["amp_weight"].
    # This env cfg intentionally does not expose it — runner cfg is the single source of truth.

    # Number of past observation frames fed to the AMP discriminator (history window).
    amp_history_length: int = 10

    # Dimension of a single AMP discriminator observation frame (per policy step).
    # Layout: dof_pos(12)+dof_vel(12)+root_height(1)+root_lin_vel(3)+root_ang_vel(3)+
    #         foot_pos_local(12)+root_rot_tan_norm(6) = 49
    # root_rot_tan_norm(6): heading-relative 6D rotation feature (MimicKit compute_tar_obs 방식).
    # Internal base buffer stores 43-dim; tan_norm 6D is computed at consumption time.
    amp_obs_dim: int = 49

    # Spawn distribution: parkour's default uniform terrain assignment is used unchanged.
    # AMP applies only to envs spawned on parkour_flat (flat_env_mask) — both reward and
    # discriminator gradient are masked. See parkour_imitation_env.py:_flat_env_mask.

    # Inherits parkour reward_scales entirely — regularization penalties (torque, action_rate,
    # joint_acc, collision, stumble, edge, ...) all retained for sim-to-real safety.
    # AMP reward fusion is handled by OnPolicyRunnerParkourAMP (additive + flat_env_mask).

    # ── Terrain override parameters ───────────────────────────────────────────
    # All defaults mirror PARKOUR_TERRAINS_CFG in parkour_env_cfg.py so behaviour is
    # unchanged unless the user explicitly modifies them.
    #
    # Applied in __post_init__ → self.terrain.terrain_generator.<field> = value
    # after super().__post_init__() (which handles actuators only).

    # Tile dimensions [m].  Each sub-terrain tile is size[0] long × size[1] wide.
    # PARKOUR_TERRAINS_CFG default: (25.0, 4.0).
    terrain_size: tuple[float, float] = (25.0, 4.0)

    # Number of curriculum rows (difficulty axis: row 0 = easiest, row N-1 = hardest).
    # Decrease for smoke / ablation runs (e.g. 5).  PARKOUR_TERRAINS_CFG default: 11.
    terrain_num_rows: int = 11

    # Number of terrain columns = total number of terrain tiles per row.
    # Must be divisible by the number of active sub-terrains (5 active, 6 inactive).
    # Decrease proportionally with num_rows for small grids.  Default: 40.
    terrain_num_cols: int = 40

    # Border width [m] around the entire terrain grid.  Default: 20.0.
    terrain_border_width: float = 20.0

    # Horizontal mesh resolution [m/cell].  Smaller = finer, heavier to build.  Default: 0.05.
    terrain_horizontal_scale: float = 0.05

    # Vertical height resolution [m/cell].  Default: 0.005.
    terrain_vertical_scale: float = 0.005

    # Whether to use curriculum (row → difficulty).  Set False for uniform random sampling.
    terrain_curriculum: bool = True

    # Difficulty range for terrain generator [min, max].  Default: (0.0, 1.0).
    terrain_difficulty_range: tuple[float, float] = (0.0, 1.0)

    # Maximum curriculum level at env initialisation.  Caps the spawn row for new envs.
    # Maps to TerrainImporterCfg.max_init_terrain_level.  Default: 3.
    terrain_max_init_level: int = 3

    # Sub-terrain proportion overrides for the 5 active terrain types.
    # Values must sum to 1.0.  Keys must match sub_terrains dict in PARKOUR_TERRAINS_CFG.
    # AMP note: increasing "parkour_flat" raises the flat_env_mask fraction, which
    # directly increases the number of envs receiving AMP reward.
    # Default: all five active types at 0.2 each (uniform, Genesis reference).
    terrain_sub_terrain_proportions: dict = {
        "parkour_flat": 0.2,
        "parkour_hurdle": 0.2,
        "parkour_step": 0.2,
        "parkour_gap": 0.2,
        "parkour_stair": 0.2,
    }

    def __post_init__(self):
        # Run parent __post_init__ first (sets up actuators on self.robot).
        super().__post_init__()

        # ── Apply terrain generator overrides ─────────────────────────────────
        tg = self.terrain.terrain_generator  # TerrainGeneratorCfg instance

        tg.size = self.terrain_size
        tg.num_rows = self.terrain_num_rows
        tg.num_cols = self.terrain_num_cols
        tg.border_width = self.terrain_border_width
        tg.horizontal_scale = self.terrain_horizontal_scale
        tg.vertical_scale = self.terrain_vertical_scale
        tg.curriculum = self.terrain_curriculum
        tg.difficulty_range = self.terrain_difficulty_range

        # Apply sub-terrain proportion overrides (only for keys present in the dict).
        # Keys not listed are left unchanged (inactive terrains remain at proportion=0.0).
        for name, proportion in self.terrain_sub_terrain_proportions.items():
            if name in tg.sub_terrains:
                tg.sub_terrains[name].proportion = proportion

        # Apply max_init_terrain_level to the TerrainImporterCfg wrapper.
        self.terrain.max_init_terrain_level = self.terrain_max_init_level

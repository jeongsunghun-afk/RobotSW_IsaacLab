# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Visualization-only cfg: all 4 perception modalities active in one running instance.

Purpose
-------
This cfg is for interactive/offline **inspection** of the four perception modalities
side by side — height_scan (2D heightfield), clearance (3D ray-cast, 294-dim), voxel
(occupancy grid, 7371-dim), and Mid-360 LiDAR (range-image + raw hit points) — on the
terrain types where the 3D-vs-2D distinction actually matters (crawl/hurdle/gap/stair).
It is **NOT** a training cfg: no algorithm depends on the extra buffers this cfg
activates, and it is not wired into any rsl_rl reward/obs consumption path beyond what
the inherited Lidar env already does.

Inherits ``ParkourImitationRandomGoalLidarEnvCfg`` (→ ``ParkourImitationRandomGoalEnvCfg``
→ ``ParkourImitationEnvCfg`` → ``ParkourEnvCfg``), so ``Go2ParkourImitationRandomGoalLidarEnv``
(entry_point reused unchanged) already gives every instance:
    - ``self._scan``          height_scan buffer   (always built by ``Go2ParkourEnv``)
    - ``self._mid360``        Mid-360 LiDAR sensor  (added by the Lidar env cfg)
This cfg additionally activates (via inherited ``Go2ParkourEnv`` flags):
    - ``self._clearance_vec`` clearance 3D ray-cast buffer (294-dim)  [enable_clearance_scanner]
    - ``self._voxel_grid``    voxel occupancy grid buffer (7371-dim) [enable_voxel_scanner]
      (voxel filling reuses the clearance scanner ray hits, so enabling voxel implicitly
      requires clearance — both flags are set True here; see parkour_env.py:451.)

Obs routing is UNCHANGED — do not confuse "buffer activation" with "obs concatenation":
    - ``clearance_as_scan`` is left at its inherited default (False) → obs["scan"] stays
      the 187-dim height_scan exactly as in ``ParkourImitationRandomGoalLidarEnvCfg``.
    - ``enable_voxel_scanner=True`` adds a NEW ``obs["voxel"]`` dict key
      (parkour_env.py:1346), but does not touch ``observation_space`` (46, "policy" dim)
      or any existing obs group size — dict keys not consumed by a given runner/policy
      network are simply ignored at rollout time.
    - Mid-360 LiDAR obs["lidar"] (range-image) and extras (lidar_hits_w/pos_w/quat_w) are
      produced exactly as in the parent Lidar env cfg — untouched here.

Random-goal is disabled for this cfg
-------------------------------------
``Go2ParkourImitationRandomGoalEnv.__init__`` (parkour_imitation_random_goal_env.py:88-93)
unconditionally asserts at least one ``parkour_flat``-class terrain column exists
(``assert self._flat_cols.numel() > 0``), independent of ``enable_random_goal``. This cfg
therefore MUST keep a nonzero (if minimal) ``parkour_flat`` proportion below — but with
``enable_random_goal=False`` the random-goal / force-flat curriculum-override machinery
never activates, so the robot always follows the terrain's normal fixed goal course
(no confound from 360° random-goal spawning while inspecting sensor data).

Terrain mix: crawl / hurdle / gap / stair only (+ minimal flat)
-----------------------------------------------------------------
Restricted to the 4 terrain types where 3D perception (clearance/voxel/LiDAR) actually
diverges from the 2D height_scan baseline: parkour_crawl (ceiling), parkour_hurdle,
parkour_gap, parkour_stair. ``parkour_step`` and all other inactive terrain classes stay
at proportion=0.0. ``parkour_flat`` is kept at the minimal 0.05 required by the hard
assert above; the remaining 0.95 is split evenly across the 4 target terrains
(0.2375 each). Sum = 0.05 + 4*0.2375 = 1.0 (required by the proportion-sum assert in
``Go2ParkourEnv.__init__``, parkour_env.py:184).

Terrain curriculum frozen
--------------------------
``terrain_curriculum=False`` (a ``ParkourImitationEnvCfg`` field, applied in its
``__post_init__``) does two things simultaneously:
    1. ``TerrainGeneratorCfg.curriculum = False`` — sub-terrain difficulty is randomized
       per tile instead of increasing monotonically by row, so a broad difficulty range
       is visible immediately without needing curriculum progression.
    2. The per-episode ``_update_terrain_curriculum`` call is skipped entirely
       (``Go2ParkourEnv._reset_idx``, parkour_env.py:1650), so envs never drift off their
       initially spawned terrain column/level while being inspected — stable, reproducible
       layout across a viewing session. This also matches the RandomGoal env's advice: with
       no trained policy driving rollout (e.g. zero/random actions), reward-driven
       advance/regress logic would otherwise converge envs toward the easiest terrain level.
"""

from isaaclab.utils import configclass

from .parkour_imitation_random_goal_lidar_env_cfg import ParkourImitationRandomGoalLidarEnvCfg


@configclass
class ParkourImitationRandomGoalLidarVizAllModalityEnvCfg(ParkourImitationRandomGoalLidarEnvCfg):
    """Visualization-only cfg with height_scan + clearance + voxel + Mid-360 LiDAR all active.

    See module docstring for the full rationale. All policy-facing obs fields
    (observation_space, obs_groups, clearance_as_scan, amp_obs, lidar range-image
    contract) are inherited UNCHANGED from ``ParkourImitationRandomGoalLidarEnvCfg``.
    Only buffer-activation flags, random-goal, terrain curriculum, and terrain mix
    proportions are overridden.
    """

    # ── Activate the 3D privileged buffers (clearance ray-cast + voxel occupancy). ──
    # height_scan (self._scan) and Mid-360 LiDAR (self._mid360) are already active
    # unconditionally / via the parent Lidar cfg respectively — no flag needed for those.
    enable_clearance_scanner: bool = True
    enable_voxel_scanner: bool = True

    # ── Disable random-goal (viz simplicity; see module docstring). ─────────────────
    # NOTE: parkour_flat must still be > 0 below — Go2ParkourImitationRandomGoalEnv
    # hard-asserts this at construction time regardless of enable_random_goal.
    enable_random_goal: bool = False

    # ── Freeze terrain curriculum (stable, reproducible layout for inspection). ─────
    terrain_curriculum: bool = False

    # ── Terrain mix: crawl + hurdle + gap + stair (minimal flat for the hard assert). ─
    terrain_sub_terrain_proportions: dict = {
        "parkour_flat": 0.05,
        "parkour_hurdle": 0.2375,
        "parkour_step": 0.0,
        "parkour_gap": 0.2375,
        "parkour_stair": 0.2375,
        "parkour_crawl": 0.2375,
    }

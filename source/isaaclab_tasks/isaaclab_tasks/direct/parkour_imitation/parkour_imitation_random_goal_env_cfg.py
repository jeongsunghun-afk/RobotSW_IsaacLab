# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Go2 ParkourImitation-Symmetry-RandomGoal environment.

Inherits all fields from ParkourImitationEnvCfg (AMP, terrain, reward, sensors, actuators)
and adds random-goal parameters for 360° yaw generalisation.

Random-goal mode activates for a subset of curriculum-graduated envs
(``random_goal_graduated_ratio``).  Those envs receive a goal sampled uniformly in
``random_goal_dist_range`` metres around the robot at a uniformly random yaw, instead
of the fixed forward goal used by the base parkour task.  When
``random_goal_force_flat`` is True, random-goal envs are additionally forced to spawn
on flat sub-terrain so that yaw learning is not confounded by terrain difficulty.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_tasks.direct.parkour_imitation.parkour_imitation_env_cfg import ParkourImitationEnvCfg


@configclass
class ParkourImitationRandomGoalEnvCfg(ParkourImitationEnvCfg):
    """Configuration for the Go2 ParkourImitation-Symmetry-RandomGoal environment.

    Extends ParkourImitationEnvCfg with parameters that control the random-goal
    curriculum mode.  All AMP, terrain, reward, sensor, and actuator fields are
    inherited unchanged.

    Random-goal behaviour
    ---------------------
    ``enable_random_goal``
        Master switch.  When False, env behaves identically to ParkourImitationEnvCfg
        (no random-goal logic is activated).

    ``random_goal_graduated_ratio``
        Fraction of curriculum-graduated envs (those at the highest terrain level)
        that are assigned random-goal mode at each episode reset.  0.2 = 20 %.

    ``random_goal_dist_range``
        Uniform sampling range (min, max) in metres for the distance from the robot
        base to the sampled goal position.

    ``num_random_goals``
        Number of consecutive goal-reaches required to count as a successful
        random-goal episode (mirrors the semantics of the base ``num_goals`` field).

    ``random_goal_force_flat``
        When True, envs assigned random-goal mode are additionally restricted to
        flat sub-terrain (terrain class 0) so yaw learning is not confounded by
        terrain obstacles.

    ``random_goal_z_offset``
        Goal z-coordinate = terrain surface height at goal XY + this offset.
        Default 0.3 matches the base task's ``goal_z`` convention.

    ``random_goal_max_height_diff``
        Maximum allowed height difference (m) between the terrain at the candidate
        goal XY and the terrain directly under the robot.  Candidates that exceed
        this threshold are rejected (avoids placing goals inside gaps/voids or on
        large step edges).

    ``random_goal_max_sample_tries``
        Maximum number of rejection-sampling attempts before falling back to a
        relaxed (no height-diff check) goal placement.
    """

    # ── Stair mid-goal override ─────────────────────────────────────────────
    # RandomGoal trains fine with stock stair but collapses with mid-goals (unexplained
    # mid-goal × random-goal interaction).  Disable mid-goals so the stair sub-terrain falls
    # back to the core ``MeshParkourStairTerrainCfg``.  The parent ``__post_init__`` reads
    # this flag (this subclass has no own __post_init__) and skips the stair replacement.
    enable_stair_midgoals: bool = False

    # ── Random-goal mode ────────────────────────────────────────────────────
    # Master switch: set False to disable all random-goal logic.
    enable_random_goal: bool = True

    # Fraction of curriculum-graduated envs that receive random-goal mode.
    random_goal_graduated_ratio: float = 0.2

    # Distance range [m] from robot base for random goal sampling (min, max).
    random_goal_dist_range: tuple[float, float] = (1.0, 3.0)

    # Number of successive goal-reaches to count as episode success.
    num_random_goals: int = 6

    # Force random-goal episodes to flat terrain (terrain class 0).
    random_goal_force_flat: bool = True

    # Goal z = terrain surface height at goal XY + this offset [m].
    random_goal_z_offset: float = 0.3

    # Max height diff [m] between robot foot terrain and candidate goal terrain.
    # Candidates exceeding this are rejected to avoid gap/void/step placement.
    random_goal_max_height_diff: float = 0.3

    # Max rejection-sampling tries before falling back to unconstrained placement.
    random_goal_max_sample_tries: int = 20

    # Forward cone total angle [degrees] for random goal direction sampling.
    # Goals are sampled within ±(cone/2) of the robot's current heading.
    # 120 = ±60°.  Set to 360 to restore the original omnidirectional behaviour.
    random_goal_forward_cone_deg: float = 120.0

    def __post_init__(self):
        super().__post_init__()
        self.reward_scales["collision"] = -10.0


@configclass
class ParkourImitationRandomGoalEasyEntryEnvCfg(ParkourImitationRandomGoalEnvCfg):
    """RandomGoal with a lowered level-0 obstacle floor ("re-entry rung").

    Motivation
    ----------
    Both Isaac Sim 5.1 and 6.0 follow the same early curriculum trajectory: the policy
    first learns flat walking and every obstacle sub-terrain is demoted to level 0 by
    roughly iteration 300.  Recovery is where they diverge.  On 5.1 the flat walker
    starts clearing level-0 obstacles around iteration 800 and climbs the ladder
    (hurdle 0.05 → 7.20 by iteration 1600); on 6.0 the same terrains stay pinned at
    0.00 indefinitely, because promotion requires travelling 80 % of the expected
    distance (see :meth:`~isaaclab_tasks.direct.parkour.parkour_env.Go2ParkourEnv.
    _update_terrain_curriculum`) and the 6.0 walker does not clear even the smallest
    obstacle.  With no rung below level 0 the curriculum has no gradient to re-enter.

    This variant lowers only the *easiest* end of each obstacle range so a flat walker
    can be promoted at all.  The hardest end is unchanged, so the final skill target is
    identical to the parent task.

    Everything else — rewards, AMP, sensors, actuators, random-goal behaviour — is
    inherited unchanged, so this is a controlled single-variable change against
    :class:`ParkourImitationRandomGoalEnvCfg`.
    """

    # Lowered lower-bounds for the difficulty-0 end of each obstacle sub-terrain [m].
    # Upper bounds are left at the parent values.
    easy_entry_hurdle_height: float = 0.01
    easy_entry_gap_length: float = 0.02
    easy_entry_step_height: float = 0.02
    easy_entry_stair_height: float = 0.02

    def __post_init__(self):
        super().__post_init__()
        # ``tg`` is instance-local: @configclass deep-copies every mutable member during
        # super().__post_init__(), so mutating sub-terrain ranges here cannot leak into
        # the module-level PARKOUR_TERRAINS_CFG shared by the other parkour tasks.
        sub = self.terrain.terrain_generator.sub_terrains

        # parkour_flat also uses MeshParkourHurdleTerrainCfg but with a (0.0, 0.0) height
        # range — it must stay perfectly flat, so it is deliberately not touched here.
        hurdle = sub["parkour_hurdle"]
        hurdle.hurdle_height_range = (self.easy_entry_hurdle_height, hurdle.hurdle_height_range[1])

        gap = sub["parkour_gap"]
        gap.gap_length_range = (self.easy_entry_gap_length, gap.gap_length_range[1])

        step = sub["parkour_step"]
        step.step_height_range = (self.easy_entry_step_height, step.step_height_range[1])

        stair = sub["parkour_stair"]
        stair.stair_height_range = (self.easy_entry_stair_height, stair.stair_height_range[1])


@configclass
class ParkourImitationRandomGoalTeacher3DEnvCfg(ParkourImitationRandomGoalEnvCfg):
    """Teacher env: RandomGoal imitation training with 3D clearance scan (294-dim) privileged obs.

    Extends ParkourImitationRandomGoalEnvCfg with:
    - enable_clearance_scanner=True  : activates the Clearance3D ray-caster
    - clearance_as_scan=True         : routes clearance_vec (normalized) into the "scan" obs slot
    - Terrain mix: all 5 base parkour terrains retained + parkour_crawl added,
      so the teacher sees both open and ceiling-constrained environments.

    Normalization used: (clearance_m - 2.0) / 2.0 → maps [0.2 m, 4.0 m] to [-0.9, 1.0].
    self._clearance_vec stays raw (0–4.0) — debug blocks and voxel logic unaffected.

    floating_ring terrain was requested but is not registered in PARKOUR_TERRAINS_CFG;
    parkour_crawl is used as the sole ceiling terrain (0.25 allocation).

    Terrain proportions (must sum to 1.0):
        parkour_flat   = 0.15
        parkour_hurdle = 0.15
        parkour_step   = 0.15
        parkour_gap    = 0.15
        parkour_stair  = 0.15
        parkour_crawl  = 0.25   (ceiling terrain; teacher uniquely benefits from 3D scan here)
        all others     = 0.0
    """

    # Activate clearance scanner and route it into the scan slot.
    enable_clearance_scanner: bool = True
    clearance_as_scan: bool = True

    # Keep random-goal enabled (inherited from parent).
    # All other random-goal / AMP / reward / actuator / sensor parameters unchanged.

    def __post_init__(self):
        super().__post_init__()
        # Rebalance terrain proportions to include parkour_crawl.
        # All 5 original base terrains are kept at 0.15 each (= 0.75).
        # parkour_crawl gets 0.25 so ceiling experience is well-represented.
        # First, zero all proportions, then set known keys.
        for key in self.terrain.terrain_generator.sub_terrains:
            self.terrain.terrain_generator.sub_terrains[key].proportion = 0.0
        self.terrain.terrain_generator.sub_terrains["parkour_flat"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_hurdle"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_step"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_gap"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_stair"].proportion = 0.15
        self.terrain.terrain_generator.sub_terrains["parkour_crawl"].proportion = 0.25


@configclass
class ParkourImitationRandomGoalTeacher3DVoxelEnvCfg(ParkourImitationRandomGoalTeacher3DEnvCfg):
    """Voxel-teacher arm: RandomGoal + 3D clearance scan + voxel occupancy grid.

    Extends ParkourImitationRandomGoalTeacher3DEnvCfg with:
    - enable_voxel_scanner=True : activates voxel occupancy grid fill (27×21×13=7371)
      and routes a flat (N,7371) ``"voxel"`` tensor into the obs dict.

    Controlled variables (D8 in voxel_teacher_arm_plan.md):
    - enable_clearance_scanner=True and clearance_as_scan=True are **inherited unchanged**
      so the critic still consumes raw clearance-294.
    - Terrain mix is **identical** to Teacher3D (parkour_crawl=0.25) for a fair comparison.

    Intended difference vs Teacher3D:
        Actor terrain encoder: clearance scandot-MLP(294→32) → voxel CNN(7371→32, VoxelEncoder).
        Critic: raw clearance-294 (same in both arms — controlled variable).
    """

    # Activate voxel grid builder.  clearance scanner is inherited (enable_clearance_scanner=True)
    # because voxel filling reuses the clearance ray hits (see parkour_env.py logic).
    enable_voxel_scanner: bool = True

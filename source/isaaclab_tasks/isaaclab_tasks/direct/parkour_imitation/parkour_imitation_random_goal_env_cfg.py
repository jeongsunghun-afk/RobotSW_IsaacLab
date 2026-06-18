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

from isaaclab.utils import configclass

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

    # ── Random-goal mode ────────────────────────────────────────────────────
    # Master switch: set False to disable all random-goal logic.
    enable_random_goal: bool = True

    # Fraction of curriculum-graduated envs that receive random-goal mode.
    random_goal_graduated_ratio: float = 0.2

    # Distance range [m] from robot base for random goal sampling (min, max).
    random_goal_dist_range: tuple[float, float] = (1.5, 3.0)

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

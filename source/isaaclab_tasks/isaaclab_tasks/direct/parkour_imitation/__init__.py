# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Parkour-Imitation hybrid locomotion environment.

Combines multi-terrain parkour with flat-terrain AMP trot imitation.
AMP reward (weight=0.3) is applied only to flat sub-terrain envs;
all other terrains use the parkour reward unchanged.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-ParkourImitation-v0",
    entry_point=f"{__name__}.parkour_imitation_env:Go2ParkourImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_imitation_env_cfg:ParkourImitationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationPPOAMPRunnerCfg",
    },
)

gym.register(
    id="Go2-ParkourImitation-Symmetry-v0",
    entry_point=f"{__name__}.parkour_imitation_env:Go2ParkourImitationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_imitation_env_cfg:ParkourImitationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryPPOAMPRunnerCfg",
    },
)

gym.register(
    id="Go2-ParkourImitation-Symmetry-RandomGoal-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_imitation_random_goal_env_cfg:ParkourImitationRandomGoalEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg",
    },
)

gym.register(
    id="Go2-ParkourImitation-Symmetry-RandomGoal-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_env_cfg:ParkourImitationRandomGoalEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalEasyEntryPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Teacher3D-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_env_cfg:ParkourImitationRandomGoalTeacher3DEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Teacher3DVoxel-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_env_cfg:ParkourImitationRandomGoalTeacher3DVoxelEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalTeacher3DVoxelPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Symmetry-RandomGoal-Lidar-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:ParkourImitationRandomGoalLidarEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalLidarPPOAMPRunnerCfg",
    },
)

# R2 student-learning arm: range-image obs["lidar"] + ActorCriticRMALidar / OnPolicyRunnerParkourAMPLidar.
# Same env/cfg as Lidar-v0 (obs["lidar"] is produced there); runner switches to the SL arm.
gym.register(
    id="Go2-ParkourImitation-Symmetry-RandomGoal-Lidar-SL-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:ParkourImitationRandomGoalLidarSLEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalLidarSLPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Teacher3D-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_env_cfg:ParkourImitationRandomGoalTeacher3DEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Teacher3DVoxel-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_env_cfg:ParkourImitationRandomGoalTeacher3DVoxelEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalTeacher3DVoxelPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Symmetry-RandomGoal-Lidar-SL-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:ParkourImitationRandomGoalLidarSLEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryRandomGoalLidarSLPPOAMPRunnerCfg"
        ),
    },
)

# Teacher-student distillation arm: frozen voxel teacher labels actions, raw-LiDAR student
# imitates them (on-policy DAgger, action MSE).  Same env class as the LiDAR arms — the cfg
# additionally turns on the clearance/voxel producers and matches the teacher's terrain mix.
# Launch WITHOUT --video (the replicator render path is what crashes 6.0 LiDAR runs).
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillRunnerCfg"),
    },
)

# Same distillation arm with ch0 normalised over 4 m instead of 20 m (matches the teacher's
# clearance max_distance).  Separate task + log dir so the two range scalings can be compared at
# matched iterations; the obs scaling differs, so checkpoints are NOT interchangeable.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-R4-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillR4EasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillR4RunnerCfg"),
    },
)

# Same distillation arm with the temporal window widened from 0.3 s to 1.0 s (K=3 -> 10).
# Rung A0 of the voxel-reconstruction ladder: window length is the only change, so later rungs
# (metric grid, ego-motion registration, auxiliary occupancy loss) stay attributable.
# Separate task + log dir; obs width differs (13824 -> 46080), so checkpoints are NOT
# interchangeable with the K=3 arm.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-K10-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillK10EasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillK10RunnerCfg"),
    },
)

# Lower bracket of the same rung: 0.5 s window (K=5, the value DreamWaQ++ uses).  Run alongside
# the K=3 baseline and the K=10 arm so window length is measured as a dose-response curve
# (0.3 / 0.5 / 1.0 s) rather than a single comparison.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-K5-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillK5EasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillK5RunnerCfg"),
    },
)

# Upper bracket of the same rung: 1.5 s window (K=15), the upper end of the measured 1.0-1.5 s
# lookback requirement.  Answers whether 1.0 s was still short, or whether the window has
# saturated and the remaining deficit is not a matter of window length.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-K15-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillK15EasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillK15RunnerCfg"),
    },
)

# K=10 with the post-reset ring-buffer warm-up removed.  The window sweep showed a penalty that
# grows with K, which fits both "unregistered stacking costs encoder capacity" and "the zeroed
# buffer asserts a false obstacle for K*push_every steps after every reset".  Both predict
# K=15-worst, so only removing the warm-up separates them.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-K10Fix-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillK10FixEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillK10FixRunnerCfg"),
    },
)

# Rung A1-0: the student reads a single-frame metric occupancy grid in the teacher's voxel frame
# instead of the stacked angular range image, through the same voxel CNN the teacher uses.  This
# isolates representation from accumulation — A1-1 adds the pose-registered accumulator on top.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-Grid-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillGridRunnerCfg"),
    },
)

# Rung A1-1: the metric grid plus pose-registered accumulation, so cells that go blind keep the
# value they had while visible.  This is the first arm that can actually fill the measured frontal
# blind band at 0.45-1.0 m; A1-0 changed only the encoding of a single scan.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-Acc-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillAccEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillAccRunnerCfg"),
    },
)

# Ceiling control: the student is handed the teacher's own privileged grid, so its terrain input
# has zero reconstruction error.  Any LiDAR history/reconstruction module can at best recover that
# grid, which makes this arm an upper bound on the entire programme — run it before designing one.
gym.register(
    id="Go2-ParkourImitation-Lidar-Distill-Ceiling-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillCeilingEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarDistillCeilingRunnerCfg"
        ),
    },
)

# From-scratch PPO+AMP on the metric grid — no teacher.  The earlier lidar_sl arm failed through
# the angular range image, which was later measured to be the binding defect, so that failure does
# not carry over.  Shares the A1-0 env cfg exactly: only the learning objective differs.
gym.register(
    id="Go2-ParkourImitation-Lidar-SL-Grid-Crawl-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarSLGridCrawlEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarSLGridCrawlPPOAMPRunnerCfg"
        ),
    },
)

# SL-Grid + crawl with the L/R mirror augmentation restored (obs["lidar"] y-flip now covered by
# mdp/symmetry.py).  Same env cfg as the symmetry-OFF arm — the runner cfg is the only change —
# so the pair isolates the augmentation.
gym.register(
    id="Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarSLGridCrawlEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarSLGridCrawlSymPPOAMPRunnerCfg"
        ),
    },
)

# Real-sensor arm: same objective and same runner as the Sym arm above, with the Mid-360
# modelled from the deployed sensor's rosbag (non-repetitive rolling scan, 20k rays at 38%
# no-return, 23 deg mount pitch, 0.8 m blind range, rear 120 deg cropped, body-occlusion grid
# off).  obs["lidar"] is unchanged at 7371, so the two are directly comparable.
gym.register(
    id="Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarSLGridCrawlRealSensorEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarSLGridCrawlSymPPOAMPRunnerCfg"
        ),
    },
)

gym.register(
    id="Go2-ParkourImitation-Lidar-SL-Grid-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_lidar_env:Go2ParkourImitationRandomGoalLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_lidar_env_cfg:"
            "ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationLidarSLGridPPOAMPRunnerCfg"),
    },
)

gym.register(
    id="Go2-ParkourImitation-TerrainStyle-v0",
    entry_point=f"{__name__}.parkour_imitation_terrain_style_env:ParkourImitationTerrainStyleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_imitation_env_cfg:ParkourImitationTerrainStyleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationTerrainStylePPOAMPRunnerCfg",
    },
)

gym.register(
    id="Go2-ParkourDemo-v0",
    entry_point=f"{__name__}.parkour_demo_env:Go2ParkourDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_demo_env_cfg:ParkourDemoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryPPOAMPRunnerCfg",
    },
)

gym.register(
    id="Go2-ParkourDemo-Playground-v0",
    entry_point=f"{__name__}.parkour_demo_env:Go2ParkourDemoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_demo_env_cfg:ParkourPlaygroundEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationSymmetryPPOAMPRunnerCfg",
    },
)


# Ground-truth voxel teacher: the grid is filled by enumerating every column (one ray down, one up)
# instead of scattering 294 clearance hit points, so solid geometry is never missed and the volume
# below a surface is filled.  Separate baseline family — the teacher's observation changes.
gym.register(
    id="Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0",
    entry_point=f"{__name__}.parkour_imitation_random_goal_env:Go2ParkourImitationRandomGoalEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.parkour_imitation_random_goal_env_cfg:"
            "ParkourImitationRandomGoalTeacher3DVoxelGTEasyEntryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_amp_cfg:Go2ParkourImitationTeacher3DVoxelGTPPOAMPRunnerCfg"
        ),
    },
)

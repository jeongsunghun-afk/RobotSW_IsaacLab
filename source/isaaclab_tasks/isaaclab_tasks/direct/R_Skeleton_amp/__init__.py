# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Ant locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="R_Skeleton-AMP-v0",
    entry_point=f"{__name__}.skeleton_amp_env:SkeletonAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.skeleton_amp_env_cfg:SkeletonAmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SkeletonAmpPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

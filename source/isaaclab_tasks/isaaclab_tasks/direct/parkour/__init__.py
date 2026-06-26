# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Go2 Parkour locomotion environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-Parkour-Direct-v0",
    entry_point=f"{__name__}.parkour_env:Go2ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Parkour-CrawlTest-v0",
    entry_point="isaaclab_tasks.direct.parkour.parkour_env:Go2ParkourEnv",
    kwargs={
        "env_cfg_entry_point": (
            "isaaclab_tasks.direct.parkour.parkour_env_cfg:ParkourCrawlTestEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "isaaclab_tasks.direct.parkour.agents.rsl_rl_ppo_cfg:Go2ParkourPPORunnerCfg"
        ),
    },
    disable_env_checker=True,
)

gym.register(
    id="Go2-Parkour-Direct-SPO",
    entry_point=f"{__name__}.parkour_env:Go2ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourSPOPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Parkour-Direct-LCP",
    entry_point=f"{__name__}.parkour_env:Go2ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourLCPPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Parkour-Direct-MoE",
    entry_point=f"{__name__}.parkour_env:Go2ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourMoEPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Parkour-Symmetry",
    entry_point=f"{__name__}.parkour_env:Go2ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourSymmetryPPORunnerCfg",
    },
)

gym.register(
    id="Go2-Parkour-Lidar-v0",
    entry_point=f"{__name__}.parkour_lidar_env:Go2ParkourLidarEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_lidar_env_cfg:ParkourLidarEnvCfg",
        # Reuse the SAME runner cfg as Go2-Parkour-Direct-v0 so the existing checkpoint
        # (experiment_name="go2_parkour") loads without any architecture change.
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourPPORunnerCfg",
    },
)

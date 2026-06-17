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

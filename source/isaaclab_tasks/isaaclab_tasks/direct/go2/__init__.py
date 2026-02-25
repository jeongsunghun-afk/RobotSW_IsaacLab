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
    id="Go2",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2FlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Go2WTW",
    entry_point=f"{__name__}.go2_wtw_env:WTWEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2FlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2WTWPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Go2Rough",
    entry_point=f"{__name__}.go2_env:Go2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2RoughEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_rough_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_rough_ppo_cfg.yaml",
    },
)

gym.register(
    id="Go2Neck",
    entry_point=f"{__name__}.go2_neck_env:Go2NeckEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_env_cfg:Go2NeckFlatEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_flat_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2NeckWTWPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_flat_ppo_cfg.yaml",
    },
)

gym.register(
    id="Go2Interaction",
    entry_point=f"{__name__}.go2_interaction_env:Go2InteractionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_interaction_cfg:Go2InteractionCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2InteractionPPORunnerCfg",
    },
)

gym.register(
    id="Go2NeckInteraction",
    entry_point=f"{__name__}.go2_neck_interaction_env:Go2NeckInteractionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_neck_interaction_cfg:Go2NeckInteractionCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2NeckInteractionPPORunnerCfg",
    },
)

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
    id="Go2AMP",
    entry_point=f"{__name__}.go2_amp_env:Go2AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_amp_env_cfg:Go2AmpEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AmpPPORunnerCfg",
    },
)

gym.register(
    id="Go2AMP-Simple",
    entry_point=f"{__name__}.go2_amp_env:Go2AmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_amp_env_cfg:Go2AmpSimpleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2AmpSimplePPORunnerCfg",
    },
)

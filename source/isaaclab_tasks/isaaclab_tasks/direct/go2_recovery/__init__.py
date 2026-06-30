# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 fall-recovery environment."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2Recovery-v0",
    entry_point=f"{__name__}.go2_recovery_env:Go2RecoveryEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_recovery_env_cfg:Go2RecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Go2Recovery-RisePacing-v0",
    entry_point=f"{__name__}.go2_recovery_rise_pacing_env:Go2RecoveryRisePacingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_recovery_rise_pacing_env_cfg:Go2RecoveryRisePacingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_rise_pacing_cfg:Go2RecoveryRisePacingPPORunnerCfg",
    },
)

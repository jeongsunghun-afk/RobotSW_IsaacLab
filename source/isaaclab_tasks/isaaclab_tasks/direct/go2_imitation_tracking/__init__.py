# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Imitation Tracking (AMP + body-frame velocity command) 환경."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Go2-Imitation-Tracking-v0",
    entry_point=f"{__name__}.go2_imitation_tracking_env:Go2ImitationTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_imitation_tracking_env_cfg:Go2ImitationTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ImitationTrackingPPORunnerCfg",
    },
)

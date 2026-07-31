# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg(17-DOF 4족 + 허리) Imitation Tracking (AMP + body-frame 속도추종) 환경."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Leg-Imitation-Tracking-v0",
    entry_point=f"{__name__}.leg_imitation_tracking_env:LegImitationTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leg_imitation_tracking_env_cfg:LegImitationTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LegImitationTrackingPPORunnerCfg",
    },
)

gym.register(
    id="Leg-Imitation-Tracking-RMA-v0",
    entry_point=f"{__name__}.leg_imitation_tracking_rma_env:LegImitationTrackingRMAEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.leg_imitation_tracking_rma_env_cfg:LegImitationTrackingRMAEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LegImitationTrackingRMAPPORunnerCfg",
    },
)

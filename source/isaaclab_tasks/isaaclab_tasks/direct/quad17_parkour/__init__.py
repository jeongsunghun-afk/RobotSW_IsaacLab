# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""17-DOF quadruped parkour: the PROVEN Go2 parkour recipe applied to OUR 17-DOF robot.

Same recipe as ``direct/parkour`` (PARKOUR_TERRAINS_CFG curriculum + height_scanner perception +
gait-free rewards + ActorCriticRMA), with ONLY the robot swapped to ``LEG_DTC_CFG`` and the
dims / body names adapted. Used to isolate "our robot vs our approach": our custom
``Quad17-Velocity-Direct-v0`` foothold-tracker failed at gap crossing; this task tests whether the
same robot learns terrain crossing under the validated recipe.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Quad17-Parkour-Direct-v0",
    entry_point=f"{__name__}.parkour_env:Quad17ParkourEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:Quad17ParkourEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Quad17ParkourPPORunnerCfg",
    },
)

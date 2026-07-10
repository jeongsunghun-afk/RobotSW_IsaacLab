# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 Real2Sim 환경.

실행:
  ./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py --num_envs 1
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-R2S-Go2-v0",
    entry_point=f"{__name__}.r2s_go2_env:R2SGo2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2s_go2_env_cfg:R2SGo2EnvCfg",
    },
)

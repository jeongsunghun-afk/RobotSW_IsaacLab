# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""R_Skeleton Hind Leg Real2Sim 환경.

실행:
  # 시뮬레이션
  ./isaaclab.sh -p scripts/real2sim/sim_runner.py --num_envs 1

  # 컨트롤러 (별도 터미널)
  python scripts/real2sim/controller.py
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-R2S-HindLeg-v0",
    entry_point=f"{__name__}.r2s_hind_leg_env:R2SHindLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2s_hind_leg_env_cfg:R2SHindLegEnvCfg",
    },
)

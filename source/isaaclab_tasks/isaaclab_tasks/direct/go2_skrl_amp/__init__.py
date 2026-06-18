# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 skrl AMP (Adversarial Motion Prior) 환경.

new_dataset_smr TXT → NPZ 변환 후 skrl AMP로 학습.

변환 스크립트:
  ./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation/imitation/convert_smr_to_npz.py

학습:
  ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \\
    --task Isaac-Go2-AMP-Direct-v0 --num_envs 4096
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Go2-AMP-Direct-v0",
    entry_point=f"{__name__}.go2_skrl_amp_env:Go2SkrlAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_skrl_amp_env_cfg:Go2SkrlAmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_amp_cfg.yaml",
    },
)

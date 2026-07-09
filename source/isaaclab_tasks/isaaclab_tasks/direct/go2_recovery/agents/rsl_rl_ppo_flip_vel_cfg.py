# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_rise_slow_cfg import Go2RecoveryRiseSlowPPORunnerCfg


@configclass
class Go2RecoveryFlipVelPPORunnerCfg(Go2RecoveryRiseSlowPPORunnerCfg):
    """PPO runner config for Go2Recovery-FlipVel-v0.

    Go2RecoveryRiseSlowPPORunnerCfg를 상속. experiment_name과 max_iterations만 override.
    네트워크/알고리즘 하이퍼파라미터는 RiseSlow와 동일.
    """

    experiment_name: str = "go2_recovery_flip_vel"
    max_iterations: int = 10000

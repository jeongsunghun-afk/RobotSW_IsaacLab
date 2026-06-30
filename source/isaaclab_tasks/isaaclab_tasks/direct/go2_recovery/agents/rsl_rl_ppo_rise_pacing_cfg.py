# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import Go2RecoveryPPORunnerCfg


@configclass
class Go2RecoveryRisePacingPPORunnerCfg(Go2RecoveryPPORunnerCfg):
    """PPO runner config for Go2Recovery-RisePacing-v0.

    base(Go2RecoveryPPORunnerCfg)를 상속. experiment_name만 override.
    네트워크/알고리즘 하이퍼파라미터는 base와 동일.
    """

    experiment_name: str = "go2_recovery_rise_pacing"

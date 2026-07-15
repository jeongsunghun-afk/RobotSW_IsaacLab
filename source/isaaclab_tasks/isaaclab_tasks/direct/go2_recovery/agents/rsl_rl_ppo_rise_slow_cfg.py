# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_rise_pacing_cfg import Go2RecoveryRisePacingPPORunnerCfg


@configclass
class Go2RecoveryRiseSlowPPORunnerCfg(Go2RecoveryRisePacingPPORunnerCfg):
    """PPO runner config for Go2Recovery-RiseSlow-v0.

    base(Go2RecoveryRisePacingPPORunnerCfg)를 상속. experiment_name만 override.
    네트워크/알고리즘 하이퍼파라미터는 RisePacing과 동일.
    """

    experiment_name: str = "go2_recovery_rise_slow"

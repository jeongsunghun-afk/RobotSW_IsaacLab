# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .rsl_rl_ppo_rise_slow_cfg import Go2RecoveryRiseSlowPPORunnerCfg


@configclass
class Go2RecoveryFlipVelPPORunnerCfg(Go2RecoveryRiseSlowPPORunnerCfg):
    """PPO runner config for Go2Recovery-FlipVel-v0.

    Go2RecoveryRiseSlowPPORunnerCfg를 상속. experiment_name / max_iterations /
    entropy_coef만 override. 네트워크 하이퍼파라미터는 RiseSlow와 동일.
    """

    experiment_name: str = "go2_recovery_flip_vel"
    max_iterations: int = 10000

    def __post_init__(self):
        super().__post_init__()

        # entropy_coef: 0.01 → 0.002
        #
        # IsaacLab-5.1(rsl_rl ActorCritic)에서는 0.01로 std가 1.0→0.34로 수축하며
        # 정상 학습됐으나, 6.0(rsl_rl 5.0.1: MLPModel + GaussianDistribution)에서는
        # 같은 값으로 entropy 보너스가 policy gradient를 압도해 std가 1.0→18로 발산한다.
        # 정책이 랜덤 노이즈가 되어 success_rate 0%, reward -157로 학습이 붕괴한다.
        # PPO 수식/advantage 정규화/std 파라미터화는 두 버전이 동일하므로 코드 버그가
        # 아니라, 0.01이 수축↔발산 임계값 근처였고 6.0에서 발산 쪽으로 넘어간 것이다.
        #
        # 400-iter A/B (num_envs=4096, seed=42), iter 399 시점:
        #   0.01  → std 2.48(발산 중), reward -21.2, success  0.0%
        #   0.002 → std 0.173,         reward  +7.1, success 15.4%   ← 채택
        #   0.0   → std 0.080(과소),   reward +6.77, success 11.4%
        # 0.0은 탐색이 죽어 success가 낮다. 0.002가 5.1의 수렴 std(0.34)에 가장 가깝다.
        self.algorithm.entropy_coef = 0.002

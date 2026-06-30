# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2RecoveryPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for Go2 fall-recovery.

    Hyperparameters chosen for recovery task:
      - Larger hidden dims [512, 256, 128] for complex fall-recovery dynamics
      - entropy_coef 0.01 for sufficient exploration of diverse fall postures
      - learning_rate 1e-4 (conservative for stable recovery learning)
      - max_iterations 1000 for M1 skeleton validation; increase in M2+
    """

    num_steps_per_env: int = 24
    max_iterations: int = 10000
    save_interval: int = 100
    experiment_name: str = "go2_recovery_direct"

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

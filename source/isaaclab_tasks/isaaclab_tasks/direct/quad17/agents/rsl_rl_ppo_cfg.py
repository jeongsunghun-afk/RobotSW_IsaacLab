# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Quad17PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """RMA PPO runner (ActorCriticRMA + estimator), same structure as HindLegParkourPPORunnerCfg.

    obs_groups maps the env's returned observation dict keys to the RMA network's inputs:
      policy    -> proprio only
      critic    -> proprio + privileged (asymmetric critic)
      history   -> stacked proprio history (RMA adaptation module)
      priv*     -> privileged groups estimated / regressed by the estimator
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "quad17_velocity_direct"
    class_name = "OnPolicyRunnerParkour"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "priv_explicit", "priv_latent"],
        "history": ["history"],
        "priv": ["priv_latent"],
        "priv_explicit": ["priv_explicit"],
    }

    estimator = {
        "hidden_dims": [128, 64],
        "learning_rate": 1.0e-3,
        "train_with_estimated_states": True,
    }

    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        noise_std_type="log",
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOParkour",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

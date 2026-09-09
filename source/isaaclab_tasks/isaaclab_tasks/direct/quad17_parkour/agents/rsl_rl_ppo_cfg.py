# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Quad17ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPOParkour + ActorCriticRMA runner for the 17-DOF parkour env.

    Identical to ``Go2ParkourPPORunnerCfg`` (the proven Go2 parkour recipe) — the ActorCriticRMA
    infers EVERY group dimension from the returned obs TensorDict at runtime, so no dims change when
    the robot goes 12->17 DOF; the obs_groups keys (policy / scan / priv_explicit / priv_latent /
    history) are exactly the keys ``Quad17ParkourEnv._get_observations`` returns. Only the class
    name and ``experiment_name`` differ so checkpoints land in a separate folder.

    - ActorCriticRMA: policy + asymmetric critic + scan_encoder + priv_encoder + state_history_encoder
    - PPOParkour: priv_reg loss scheduling + update_dagger() for the history encoder
    - OnPolicyRunnerParkour: orchestrates the dagger / encoder updates
    - estimator: predicts priv_explicit (base lin/ang vel, 6D) from the proprio history
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "quad17_parkour"
    empirical_normalization = False
    clip_actions = 10.0

    # Specialized parkour runner with dagger support.
    class_name = "OnPolicyRunnerParkour"

    # Route the env's dict observations to actor / critic / encoders. Group DIMS are inferred at
    # runtime from the obs tensors, so these are robot-agnostic (identical to the Go2 recipe):
    #   policy (proprio) = 61, scan = 187, priv_explicit = 6, priv_latent = 43, history = 10 x 61.
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "scan", "priv_explicit", "priv_latent"],
        "scan": ["scan"],
        "history": ["history"],
        "priv": ["priv_latent"],
        "priv_explicit": ["priv_explicit"],
    }

    # Estimator: predicts priv_explicit (base lin/ang vel) from proprioceptive history.
    estimator = {
        "hidden_dims": [128, 64],
        "learning_rate": 1.0e-3,
        "train_with_estimated_states": True,
    }

    # ActorCriticRMA: policy net + critic net + scan_encoder + priv_encoder + state_history_encoder.
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        noise_std_type="log",
        activation="elu",
    )

    # PPOParkour: extends PPO with priv_reg loss scheduling and dagger updates.
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOParkour",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

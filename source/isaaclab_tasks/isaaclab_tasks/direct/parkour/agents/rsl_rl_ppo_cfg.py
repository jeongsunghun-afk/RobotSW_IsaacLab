# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2ParkourPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 Parkour 환경용 PPOParkour + ActorCriticRMA 러너 설정.

    Task #7: Replaces standard PPO with goal-directed parkour-specific algorithm.
    - ActorCriticRMA: supports obs_groups {policy, critic, scan, history, priv} with encoder networks
    - PPOParkour: includes priv_reg loss scheduling, update_dagger() for history encoder
    - OnPolicyRunnerParkour: orchestrates dagger updates and encoder training
    - obs_groups: maps environment dict observations to actor/critic/encoder inputs
    """

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_parkour"
    empirical_normalization = True
    clip_actions = 10.0

    # Use specialized parkour runner with dagger support
    class_name = "OnPolicyRunnerParkour"

    # obs_groups: route env's dict observations to actor/critic/encoders
    # env returns: {policy, critic, scan, history, priv}
    # policy: proprio [N, 46]
    # critic: full info [N, 46+187+4] (will be expanded with history in runner)
    # scan: height_scan [N, 187]
    # history: obs history [N, 10, 46] for StateHistoryEncoder
    # priv: domain rand params [N, 4]
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "scan", "priv"],
        "scan": ["scan"],
        "history": ["history"],
        "priv": ["priv"],
    }

    # ActorCriticRMA: policy network + critic network + scan_encoder + priv_encoder + state_history_encoder
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPOParkour: extends PPO with priv_reg loss scheduling and dagger updates
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOParkour",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,  # CHANGED: 2.0e-4 → 1.0e-3 (for parkour adaptation)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

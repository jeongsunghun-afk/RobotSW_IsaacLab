# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 AMP 환경용 PPO 러너 설정."""

    num_steps_per_env = 24
    max_iterations = 50000
    save_interval = 100
    experiment_name = "go2_amp_direct"
    clip_actions = 1.0
    class_name = "OnPolicyRunnerAMP"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "priv"],
        "history": ["history"],
        "priv": ["priv"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="PPOAMP",
    )
    amp = dict(
        task_reward_lerp=1.0,                   # Stage 2 최종값 (annealing 완료 후)
        task_reward_lerp_start=1.0,            # Stage 1 초기값 (pure task 위주)
        task_reward_lerp_anneal_iters=10000,    # Stage 1→2 전환 iteration 수
        discriminator_learning_rate=1e-4,
        gradient_penalty_coef=5.0,
        reward_coef=2.0,
        discriminator_hidden_dims=[1024, 512],
        enable_replay_buffer=True,
        replay_buffer_size=100000,
        disc_num_epochs=2,
        enable_lerp_schedule=True,              # False로 바꾸면 lerp annealing 비활성화
        disc_logit_reg=0.01,                    # discriminator 출력 L2 정규화 (MimicKit 방식)
    )


@configclass
class Go2AmpSimplePPORunnerCfg(Go2AmpPPORunnerCfg):
    """Go2 AMP 단순 버전 러너 설정.

    ActorCritic(RMA 없음) + PPOAMPBase(PPO 기반) + OnPolicyRunnerAMPBase 사용.
    history obs 없음, priv obs는 critic에서만 사용 (비대칭 AC 유지).
    """

    experiment_name = "go2_amp_simple"
    class_name = "OnPolicyRunnerAMPBase"
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "priv"],
        "priv": ["priv"],
    }
    policy = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="PPOAMPBase",
    )
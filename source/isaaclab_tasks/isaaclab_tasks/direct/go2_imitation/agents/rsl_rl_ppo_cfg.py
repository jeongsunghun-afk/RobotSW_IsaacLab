# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Imitation 환경용 PPO + AMP 러너 설정 (Simple 버전).

PPOAMPBase + ActorCritic + OnPolicyRunnerAMPBase 조합.
MimicKit amp_go2_task_agent.yaml 하이퍼파라미터 기반.

AMP reward mixing:
  - Stage 1 (0~5000 iter):  task_reward_lerp=1.0 → 순수 task reward
  - Stage 2 (5000~ iter):   task_reward_lerp→0.5 → 50% task + 50% AMP
  - annealing 5000 iter (= 5000×24=120000 steps)
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2ImitationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 Imitation 환경용 PPO 러너 설정 (AMP Simple 버전)."""

    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = "go2_imitation"
    clip_actions: float = 4.0

    # OnPolicyRunnerAMPBase 사용
    class_name: str = "OnPolicyRunnerAMPBase"

    # policy obs만 사용 (history/priv 없음)
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    estimator = None

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.007,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2e-4,  # MimicKit actor_optimizer lr
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="PPOAMPBase",
    )

    amp: dict = dict(
        # ── AMP reward 혼합 ────────────────────────────────────────
        task_reward_lerp=0.5,  # Stage 2 최종값
        task_reward_lerp_start=0.5,  # Stage 1 초기값 (pure task → disc warm-up)
        task_reward_lerp_anneal_iters=5000,  # 전환 iter 수 (×24 steps = 120000 steps)
        enable_lerp_schedule=True,
        # ── Discriminator 학습 ────────────────────────────────────
        discriminator_learning_rate=2.5e-4,  # MimicKit disc_optimizer lr
        gradient_penalty_coef=5.0,  # MimicKit disc_grad_penalty
        reward_coef=2.0,  # MimicKit disc_reward_scale
        discriminator_hidden_dims=[1024, 512],
        disc_num_epochs=2,  # epoch 수 (mini-batch 반복 횟수)
        disc_mini_batch_size=4096,  # mini-batch 크기 (MimicKit 방식)
        disc_logit_reg=0.01,  # logit L2 정규화 (MimicKit 방식)
        # ── Replay buffer ─────────────────────────────────────────
        enable_replay_buffer=True,
        replay_buffer_size=200000,  # MimicKit disc_buffer_size
        # ── AMP obs 차원 (env와 일치해야 함) ──────────────────────
        # amp_observation_space = 49 (per-step) × num_amp_observations(=2) = 98
        # R4: per-step 43→49 (+6 root_rot_tan_norm), env_cfg.amp_observation_space=49
        amp_observation_space=98,
        # ── Loss / Reward / Normalizer 방식 선택 ─────────────────
        # MimicKit 방식: disc_loss_type="bce", disc_reward_type="bce",
        #                disc_logit_reg_type="weight", disc_norm_clip=10.0
        # 기존 LS-GAN:   disc_loss_type="ls_gan", disc_reward_type="ls_gan",
        #                disc_logit_reg_type="logit", disc_norm_clip=None
        # disc_loss_type="bce",          # "ls_gan" | "bce" (MimicKit 방식)
        # disc_reward_type="bce",        # "ls_gan" | "bce" (MimicKit 방식)
        # disc_logit_reg_type="weight",  # "logit" | "weight" (MimicKit 방식)
        # disc_norm_clip=10.0,           # None | 10.0 (MimicKit 방식)
        disc_loss_type="ls_gan",  # "ls_gan" | "bce" (MimicKit 방식)
        disc_reward_type="ls_gan",  # "ls_gan" | "bce" (MimicKit 방식)
        disc_logit_reg_type="logit",  # "logit" | "weight" (MimicKit 방식)
        disc_norm_clip=None,  # None | 10.0 (MimicKit 방식)
    )

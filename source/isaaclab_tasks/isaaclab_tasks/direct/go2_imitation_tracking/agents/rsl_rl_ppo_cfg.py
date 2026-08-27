# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Imitation Tracking 환경용 PPO + AMP + RMA 러너 설정.

Runner:    OnPolicyRunnerAMP
Policy:    ActorCriticRMA  (history encoder + priv encoder, scan 없음)
Algorithm: PPOAMP  (PPO + AMP discriminator)
Estimator: priv_explicit(root_lin_vel_b + root_ang_vel_b) 예측 — 둘 다 policy obs에 없으므로
           proprio(42)+history로부터의 **미관측 상태 추정** 과제다

MimicKit amp_go2_task_agent.yaml 하이퍼파라미터 기반 (AMP dict 값은 불변).

AMP reward mixing:
  - Stage 1 (0~5000 iter):  task_reward_lerp=1.0 → 순수 task reward
  - Stage 2 (5000~ iter):   task_reward_lerp→0.5 → 50% task + 50% AMP
  - annealing 5000 iter (= 5000×24=120000 steps)

obs_groups: env가 반환하는 dict obs {policy, priv_explicit, priv_latent, history} 라우팅.
  policy:        proprio                [N, 42]
  priv_explicit: root_lin_vel_b*scale + root_ang_vel_b*scale  [N, 6]
  priv_latent:   armature/friction/...  [N, 19]
  history:       policy proprio history [N, 10, 42]
  critic total:  42+6+19 = 67
scan 그룹 없음 (이 env는 height_scan/clearance 미사용 — ActorCriticRMA는 scan이 obs_groups에
없으면 scan encoder를 생략한다).
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2ImitationTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 Imitation Tracking 환경용 PPO 러너 설정 (RMA + estimator + AMP)."""

    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = "go2_imitation_tracking"
    clip_actions: float = 10.0

    # OnPolicyRunnerAMP 사용 (RMA + estimator)
    class_name: str = "OnPolicyRunnerAMP"

    # dict obs: policy/priv_explicit/priv_latent/history (scan 없음)
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy", "priv_explicit", "priv_latent"],
        "priv_explicit": ["priv_explicit"],
        "priv": ["priv_latent"],
        "history": ["history"],
    }

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # priv_explicit(root 선속도/각속도) 예측 — 배포 시 lin_vel 실측 대체 + ang_vel denoising.
    # loss_blocks는 진단 전용 분해(최적화 대상 loss는 불변): 단일 스칼라로는 ang 블록이
    # 실제로 학습되는지, lin 대비 수치적으로 묻히는지 구분할 수 없다.
    estimator: dict = {
        "hidden_dims": [128, 64],
        "learning_rate": 1.0e-3,
        "train_with_estimated_states": True,
        "loss_blocks": {"lin": [0, 3], "ang": [3, 6]},
    }

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,  # std 발산 억제 (0.007→0.005)
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2e-4,  # MimicKit actor_optimizer lr
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        class_name="PPOAMP",
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
        # amp_observation_space = 49 (per-step) × num_amp_observations(=10) = 490
        # R4: per-step 43→49 (+6 root_rot_tan_norm), env_cfg.amp_observation_space=49
        # NOTE: 이 값은 fallback default일 뿐. OnPolicyRunnerAMP가 런타임에
        #       env.amp_observation_space.shape[0] (=490)로 덮어쓰므로 disc input_dim은 자동 파생.
        amp_observation_space=490,
        # ── Loss / Reward / Normalizer 방식 선택 ─────────────────
        # MimicKit 방식: disc_loss_type="bce", disc_reward_type="bce",
        #                disc_logit_reg_type="weight", disc_norm_clip=10.0
        # 기존 LS-GAN:   disc_loss_type="ls_gan", disc_reward_type="ls_gan",
        #                disc_logit_reg_type="logit", disc_norm_clip=None
        # disc_loss_type="bce",          # "ls_gan" | "bce" (MimicKit 방식)
        # disc_reward_type="bce",        # "ls_gan" | "bce" (MimicKit 방식)
        # disc_logit_reg_type="weight",  # "logit" | "weight" (MimicKit 방식)
        # disc_norm_clip=10.0,           # None | 10.0 (MimicKit 방식)
        disc_loss_type="bce",  # "ls_gan" | "bce" (MimicKit 방식) — ablation: ls_gan→bce
        disc_reward_type="bce",  # "ls_gan" | "bce" (MimicKit 방식) — ablation: ls_gan→bce
        disc_logit_reg_type="weight",  # "logit" | "weight" (MimicKit 방식) — ablation: logit→weight
        disc_norm_clip=10.0,  # None | 10.0 (MimicKit 방식) — ablation: None→10.0
    )

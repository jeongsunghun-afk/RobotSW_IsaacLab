# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Latent Imitation 환경용 PPO + latent style + RMA 러너 설정.

Runner:    OnPolicyRunnerLatent
Policy:    ActorCriticRMA  (history encoder + priv encoder, scan 없음)
Algorithm: PPOLatent  (PPO + RMA/estimator, discriminator 없음)
Estimator: priv_explicit(root_lin_vel_b + root_ang_vel_b) 예측 — 둘 다 policy obs에 없으므로
           proprio(42)+history로부터의 **미관측 상태 추정** 과제다

PPO 하이퍼파라미터는 AMP baseline(`go2_imitation_tracking`)과 동일하게 두어 스타일 신호
교체 하나만 달라지게 한다.

Style reward mixing (AMP 의 `task_reward_lerp` 스케줄을 그대로 승계):
  - Stage 1 (0~5000 iter):  task_reward_lerp=1.0 → 순수 task reward
  - Stage 2 (5000~ iter):   task_reward_lerp→0.5 → 50% task + 50% latent style
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
class Go2ImitationLatentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 Latent Imitation 환경용 PPO 러너 설정 (RMA + estimator + latent style)."""

    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = "go2_imitation_latent"
    clip_actions: float = 10.0

    # OnPolicyRunnerLatent 사용 (RMA + estimator, discriminator 없음)
    class_name: str = "OnPolicyRunnerLatent"

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
        class_name="PPOLatent",
    )

    #: 스타일 보상 혼합 스케줄. 보상 자체는 env 가 동결 인코더로 계산하므로 여기에는
    #: discriminator 하이퍼파라미터가 없다 (학습되는 판별자가 없다).
    #: 과거 실측에서 이 축(스타일 대 과제 비중)이 속도 달성률을 크게 좌우했다.
    style: dict = dict(
        task_reward_lerp=0.5,  # Stage 2 최종값
        task_reward_lerp_start=1.0,  # Stage 1 초기값 (순수 task → 스타일 도입 전 warm-up)
        task_reward_lerp_anneal_iters=5000,  # 전환 iter 수 (×24 steps = 120000 steps)
        enable_lerp_schedule=True,
    )

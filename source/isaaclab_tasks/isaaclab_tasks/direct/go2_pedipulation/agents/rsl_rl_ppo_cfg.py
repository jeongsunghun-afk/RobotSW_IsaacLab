# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Pedipulation 환경용 PPO 러너 설정 (asymmetric actor-critic).

Runner:    OnPolicyRunner
Policy:    ActorCritic
Algorithm: PPO

estimator 를 쓰지 않는다. ``ActorCriticRMA`` 는 ``priv_explicit`` 를 actor 입력에 **무조건**
concat 하는 구조라(actor_critic_parkour.py:276-282), estimator 없이 쓰면 actor 가 GT base
선속도를 직접 보게 되어 실기 배포가 막힌다. 이 저장소가 이미 두 번 겪은 실패 모드이므로
표준 ``ActorCritic`` + obs_groups 로 비대칭 구조를 만든다:

  actor  ← policy(83) + history(830)          — 실기에서 얻을 수 있는 신호만
  critic ← policy(83) + history(830) + priv(30) — base 선/각속도, 접촉력, DR 파라미터

sim2real 강건성은 estimator 대신 **domain randomization + proprio history** 로 확보한다.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2PedipulationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 Pedipulation PPO 러너 설정."""

    num_steps_per_env: int = 24
    max_iterations: int = 20000
    save_interval: int = 200
    experiment_name: str = "go2_pedipulation"
    clip_actions: float = 4.0

    class_name: str = "OnPolicyRunner"

    obs_groups: dict = {
        "policy": ["policy", "history"],
        "critic": ["policy", "history", "priv"],
    }

    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCritic",
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # 6.0 에서 entropy_coef 0.01 은 std 발산 이력이 있다 (FlipVel 진단).
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

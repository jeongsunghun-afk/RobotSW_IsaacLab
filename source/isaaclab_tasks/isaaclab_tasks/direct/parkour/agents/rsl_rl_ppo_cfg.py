# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlLcpCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoActorCriticMoECfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


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
    empirical_normalization = False
    clip_actions = 10.0

    # Use specialized parkour runner with dagger support
    class_name = "OnPolicyRunnerParkour"

    # obs_groups: route env's dict observations to actor/critic/encoders
    # env returns: {policy, scan, priv_explicit, priv_latent, history}
    # policy:        proprio           [N, 42]   (3+1+1+1+12+12+12)
    # scan:          height_scan       [N, 187]
    # priv_explicit: lin_vel+ang_vel   [N, 6]    (root_lin_vel_b*2.0(3) + root_ang_vel_b*0.25(3))
    # priv_latent:   B-style extended  [N, 37]   (base_friction(1) + foot_friction(8) +
    #                                             base_mass(1) + base_com(3) +
    #                                             joint_stiffness_ratio(12) + joint_damping_ratio(12))
    # history:       proprio history   [N, 10, 42] for StateHistoryEncoder
    # critic total:  42+187+6+37 = 272
    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy", "scan", "priv_explicit", "priv_latent"],
        "scan": ["scan"],
        "history": ["history"],
        "priv": ["priv_latent"],
        "priv_explicit": ["priv_explicit"],
    }

    # Estimator: predicts priv_explicit (base_lin_vel, 3D) from proprioceptive history
    estimator = {
        "hidden_dims": [128, 64],
        "learning_rate": 1.0e-3,
        "train_with_estimated_states": True,  # estimator 수렴 전 게이트 off; 수렴 후 True 전환
    }

    # ActorCriticRMA: policy network + critic network + scan_encoder + priv_encoder + state_history_encoder
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

    # PPOParkour: extends PPO with priv_reg loss scheduling and dagger updates
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPOParkour",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=2.0e-4,  # CHANGED: 2.0e-4 → 1.0e-3 -> 1.0e-4(for parkour adaptation)
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2ParkourSPOPPORunnerCfg(Go2ParkourPPORunnerCfg):
    """Go2 Parkour with SPO (Simple Policy Optimization) surrogate.

    Replaces PPO clip surrogate with a quadratic ratio penalty:
        f_spo = r·A − (|A|/2ε)(r−1)²
    Reference: Xie et al., arXiv:2401.16025, Eq. 16.

    Only the surrogate function changes; env, architecture, and all other
    hyperparameters are inherited from Go2ParkourPPORunnerCfg unchanged.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.surrogate_type = "spo"
        # FINAL combined stabilization screen (lead-driven): ε↑ + epochs↓ both attack the
        # SPO surrogate spike (|A|/ε)(r-1)² at its source. ② adv-norm FAILED full-env;
        # ① epochs=2 alone only DELAYED divergence (~2636→~3100). ε=0.4 halves the (1/ε)
        # amplification; epochs=2 bounds intra-update ratio drift. Judged by REWARD recovery.
        self.algorithm.spo_epsilon = 0.4
        self.algorithm.schedule = "fixed"
        self.algorithm.num_learning_epochs = 2
        self.experiment_name = "go2_parkour_spo"


@configclass
class Go2ParkourLCPPPORunnerCfg(Go2ParkourPPORunnerCfg):
    """Go2 Parkour with LCP (Lipschitz-Constrained Policy) gradient penalty.

    Adds λ·‖∇_obs log π‖² to the policy loss to reduce action jitter and
    improve sim-to-real transfer.
    Reference: arXiv:2410.11825.

    Note: PPOParkour currently accepts lcp_cfg but does not apply the penalty
    (RMA encoder inputs make the gradient path non-trivial). This cfg registers
    the variant so the penalty can be activated once ppo_parkour is extended.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.lcp_cfg = RslRlLcpCfg(lambda_gp=0.002, penalize="log_prob")
        self.experiment_name = "go2_parkour_lcp"


@configclass
class Go2ParkourMoEPPORunnerCfg(Go2ParkourPPORunnerCfg):
    """Go2 Parkour with MoE-RMA actor (Mixture-of-Experts).

    Replaces the single-MLP actor with a dense-softmax MoE: 6 expert MLPs
    weighted by a learned gating network. The critic and all encoders are
    inherited unchanged from ActorCriticRMA.
    Reference: Huang*, Zhu* et al., arXiv:2503.08564 — MoE-Loco.

    ActorCriticRMAMoE is implemented in a separate worker; this cfg wires the
    class_name so OnPolicyRunnerParkour will instantiate it.
    """

    def __post_init__(self):
        super().__post_init__()
        self.policy = RslRlPpoActorCriticMoECfg(
            class_name="ActorCriticRMAMoE",
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[512, 256, 128],
            critic_hidden_dims=[512, 256, 128],
            noise_std_type="log",
            activation="elu",
            num_experts=6,
            gating_hidden_dims=[128],
            expert_hidden_dims=[512, 256, 128],
            gating_temperature=1.0,
        )
        self.experiment_name = "go2_parkour_moe"


@configclass
class Go2ParkourSymmetryPPORunnerCfg(Go2ParkourPPORunnerCfg):
    """Go2 Parkour with L/R symmetry data-augmentation (soft equivariance enforcement).

    Activates ``PPOParkour``'s data-augmentation path with the parkour-specific left/right
    mirror function.  The augmented batch doubles the effective minibatch size (``num_aug=2``),
    so KL adaptive scheduling absorbs the change without requiring lr tuning.

    Design rationale: RL (left hind leg) parks as a reward-positive local equilibrium when
    policy equivariance is not enforced.  Mirror augmentation softly penalises L/R asymmetry
    by including both original and mirrored (obs, action) pairs in every PPO update.

    Only L/R symmetry is applied — the task is forward-only (forward command, scan offset
    +0.375 m forward), so front-back / diagonal symmetries do NOT hold.

    All PPO hyperparameters, network architecture, and encoder structure are inherited
    unchanged from ``Go2ParkourPPORunnerCfg``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=("isaaclab_tasks.direct.parkour.mdp.symmetry:compute_parkour_symmetric_states"),
        )
        self.experiment_name = "go2_parkour_symmetry"

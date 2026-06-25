# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 ParkourImitation 환경용 PPO + AMP + RMA 러너 설정.

Runner:    OnPolicyRunnerParkourAMP
Policy:    ActorCriticRMA  (history encoder + priv encoder + scan encoder)
Algorithm: PPOAMP  (PPOParkour + AMP discriminator)

AMP reward fusion (additive + flat-env mask):
    total = task_reward + amp_weight * flat_env_mask.float() * disc_reward
    amp_weight = 0.3 (train_cfg["amp"]["amp_weight"])
    task_reward_lerp / enable_lerp_schedule are IGNORED by OnPolicyRunnerParkourAMP.

AMP observation space:
    amp_observation_space = 43 per-step × 10 history frames = 430
    This value is overwritten at runner construction from env.unwrapped.amp_observation_space.
    The literal 430 here serves as a documentation cross-check.

motion_files:
    None — ParkourImitationEnvCfg.amp_motion_pkl is the single source of truth.
    The runner never reads train_cfg["amp"]["motion_files"].
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg, RslRlSymmetryCfg


@configclass
class Go2ParkourImitationPPOAMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Go2 ParkourImitation 환경용 OnPolicyRunnerParkourAMP 러너 설정.

    ActorCriticRMA + PPOAMP 조합.
    OnPolicyRunnerParkourAMP (ParkourAMP → AMP → Parkour → object) MRO 사용.
    """

    # ── Runner ────────────────────────────────────────────────────────────────
    class_name: str = "OnPolicyRunnerParkourAMP"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 200
    experiment_name: str = "parkour_imitation_go2"
    empirical_normalization: bool = True
    seed: int = 1

    # clip_actions: parkour 기준(10.0) 사용 — ActorCriticRMA + init_noise_std=1.0 기반
    clip_actions: float = 10.0

    # ── obs_groups: env dict obs → actor/critic/encoder 라우팅 ────────────────
    # env returns: {policy, scan, priv_explicit, priv_latent, history}
    # policy:        proprio           [N, 42]
    # scan:          height_scan       [N, 187]
    # priv_explicit: lin_vel+ang_vel   [N, 6]
    # priv_latent:   B-style extended  [N, 37]  (base_friction(1)+foot_friction(8)+
    #                                            base_mass(1)+base_com(3)+
    #                                            joint_stiffness_ratio(12)+joint_damping_ratio(12))
    # history:       proprio history   [N, 10, 42]
    # critic total:  42+187+6+37 = 272
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy", "scan", "priv_explicit", "priv_latent"],
        "scan": ["scan"],
        "history": ["history"],
        "priv": ["priv_latent"],
        "priv_explicit": ["priv_explicit"],
    }

    # ── Estimator: priv_explicit(lin/ang vel) 예측 ───────────────────────────
    estimator: dict = {
        "hidden_dims": [128, 64],
        "learning_rate": 1.0e-3,
        "train_with_estimated_states": True,
    }

    # ── Policy: ActorCriticRMA ────────────────────────────────────────────────
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # ── Algorithm: PPOAMP ────────────────────────────────────────────────────
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPOAMP",
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

    # ── AMP ──────────────────────────────────────────────────────────────────
    # amp_weight: 평탄 지형 환경에서 AMP reward 기여 가중치 (additive fusion)
    #   total = task_reward + amp_weight * flat_mask * disc_reward
    #   task_reward_lerp 계열 파라미터는 OnPolicyRunnerParkourAMP에서 무시됨
    #
    # amp_observation_space: 런너 초기화 시 env.unwrapped.amp_observation_space.shape[0]으로 덮어씀
    #   43(per-step) × 10(history) = 430 — 문서화/fallback 용도
    #
    # motion_files: None — ParkourImitationEnvCfg.amp_motion_pkl이 단일 소스.
    #   PPOAMP.__init__ 및 러너 모두 train_cfg["amp"]["motion_files"]를 소비하지 않음.
    amp: dict = dict(
        amp_weight=0.3,
        # ── Discriminator 학습 ────────────────────────────────────────────
        discriminator_learning_rate=2.5e-4,
        gradient_penalty_coef=5.0,
        reward_coef=4.0 * 0.02,  # 2 vs 4
        discriminator_hidden_dims=[1024, 512],
        disc_num_epochs=2,
        disc_mini_batch_size=4096,
        disc_logit_reg=0.01,
        # ── Loss / Reward / Regularizer 방식 ─────────────────────────────
        disc_loss_type="bce",  # "ls_gan" | "bce"  (go2_imitation 방식)
        disc_reward_type="bce",  # "ls_gan" | "bce"  (go2_imitation 방식)
        disc_logit_reg_type="weight",  # "logit" | "weight"  (MimicKit 방식)
        disc_norm_clip=10.0,  # None | float  (go2_imitation 방식)
        # ── Replay buffer ─────────────────────────────────────────────────
        enable_replay_buffer=True,
        replay_buffer_size=200000,
        # ── AMP obs 차원 (env와 일치; 런너 초기화 시 자동 덮어씀) ─────────
        amp_observation_space=490,  # 49 per-step × 10 history frames
        # ── Motion 파일 (env_cfg가 소유; 러너 미사용) ─────────────────────
        motion_files=None,
    )


@configclass
class Go2ParkourImitationSymmetryPPOAMPRunnerCfg(Go2ParkourImitationPPOAMPRunnerCfg):
    """Go2 ParkourImitation with L/R symmetry data-augmentation (soft equivariance enforcement).

    Activates ``PPOParkourAMP``'s data-augmentation path with the parkour-imitation-specific
    left/right mirror function.  The augmented batch doubles the effective minibatch size
    (``num_aug=2``), so KL adaptive scheduling absorbs the change without requiring lr tuning.

    AMP note:
        AMP observations travel through a separate discriminator pipeline (``env.extras["amp_obs"]``)
        and are NOT part of the obs TensorDict passed to ``data_augmentation_func``.  Therefore
        amp_obs mirroring is not performed and not required here.

    All AMP, PPO hyperparameters, network architecture, and encoder structure are inherited
    unchanged from ``Go2ParkourImitationPPOAMPRunnerCfg``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=(
                "isaaclab_tasks.direct.parkour_imitation.mdp.symmetry:compute_parkour_imitation_symmetric_states"
            ),
        )
        self.experiment_name = "parkour_imitation_go2_symmetry"


@configclass
class Go2ParkourImitationTerrainStylePPOAMPRunnerCfg(Go2ParkourImitationPPOAMPRunnerCfg):
    """Go2 ParkourImitation with terrain-invariant AMP discriminator observations.

    AMP obs per frame is 37-dim (dof_pos/vel + lin_vel_xy + ang_vel + foot_xy).
    History length 10 → flat AMP dim = 370.

    amp_observation_space is overwritten at runner construction from
    env.unwrapped.amp_observation_space (370). The literal 370 here is documentation only.
    """

    def __post_init__(self):
        super().__post_init__()
        # Mirror data-augmentation: same symmetry_cfg as Go2ParkourImitationSymmetryPPOAMPRunnerCfg.
        # AMP obs travels via env.extras["amp_obs"] (separate path) and is NOT mirrored.
        # Policy obs/action mirror is therefore safe and fully compatible with this env.
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            mirror_loss_coeff=0.0,
            data_augmentation_func=(
                "isaaclab_tasks.direct.parkour_imitation.mdp.symmetry:compute_parkour_imitation_symmetric_states"
            ),
        )
        # Override AMP obs dim to match ParkourImitationTerrainStyleEnv (37-dim/frame × 10).
        self.amp["amp_observation_space"] = 370  # 37 per-step × 10 history frames
        self.experiment_name = "parkour_imitation_go2_terrain_style"


@configclass
class Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg(Go2ParkourImitationSymmetryPPOAMPRunnerCfg):
    """Go2 ParkourImitation + L/R symmetry data-aug + 360° random-goal curriculum.

    Inherits ``Go2ParkourImitationSymmetryPPOAMPRunnerCfg`` in full:
    - PPO + AMP discriminator (BCE loss, WGAN disabled)
    - L/R mirror data-augmentation (``symmetry_cfg``, ``num_aug=2``)
    - AMP obs 49-dim/frame × 10 history = 490-dim (unchanged)

    Only ``experiment_name`` is changed so that WandB/TensorBoard logs are
    written to a separate run directory from the plain-symmetry task.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_symmetry_random_goal"


@configclass
class Go2ParkourImitationSymmetryRandomGoalLidarPPOAMPRunnerCfg(Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg):
    """Go2 ParkourImitation + symmetry + random-goal + Mid-360 LiDAR side-channel.

    Inherits ``Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg`` in full:
    - PPO + AMP discriminator (BCE loss, WGAN disabled)
    - L/R mirror data-augmentation (``symmetry_cfg``, ``num_aug=2``)
    - AMP obs 49-dim/frame × 10 history = 490-dim (unchanged)
    - 360° random-goal curriculum

    Only ``experiment_name`` is changed so logs are separated from the base
    random-goal run.  The LiDAR data appears only in ``extras["lidar_range"]``
    and has no effect on the policy/critic/discriminator network shapes.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_random_goal_lidar"

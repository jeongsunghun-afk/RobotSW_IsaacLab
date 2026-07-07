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
class Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg(
    Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg
):
    """Go2 ParkourImitation + symmetry + random-goal + Teacher 3D clearance scan (294-dim).

    Inherits ``Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg`` in full:
    - PPO + AMP discriminator (BCE loss)
    - L/R mirror data-augmentation (symmetry_cfg, num_aug=2)
    - AMP obs 49-dim/frame × 10 history = 490-dim (unchanged; scan group is separate from amp_obs)
    - 360° random-goal curriculum (±60° forward cone)

    scan obs group changes 187 → 294 (clearance_3d).
    scandot_encoder input is read from runtime obs shape (ActorCriticRMA line 121),
    so no architecture override is needed — encoder adapts automatically.

    critic total: policy(46) + scan(294) + priv_explicit(6) + priv_latent(33) = 379
    (vs. baseline 272; wider scan feeds a larger critic — acceptable for teacher training).

    Only experiment_name and (obs_groups comment) differ from parent.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_teacher3d"


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


@configclass
class Go2ParkourImitationSymmetryRandomGoalTeacher3DNoCrawlPPOAMPRunnerCfg(
    Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg
):
    """Run-B ablation runner: clearance-294 scan + original no-crawl terrain.

    Inherits ``Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg`` in full:
    - ``OnPolicyRunnerParkourAMP`` reward fusion, spawn-scheduler, save/load
    - PPO + AMP discriminator (BCE loss)
    - L/R mirror data-augmentation (symmetry_cfg, num_aug=2)
    - AMP obs 49-dim/frame × 10 history = 490-dim (unchanged)
    - scan group = clearance-294 (via clearance_as_scan=True in env cfg)
    - num_steps_per_env=24, num_mini_batches=4 (identical to baseline and Teacher3D)

    Only ``experiment_name`` differs from Teacher3D runner — logs written to
    ``parkour_imitation_go2_teacher3d_nocrawl`` for clean ablation separation.

    Ablation triangle:
        baseline (RandomGoal-v0)     : scan=187,  terrain=5×0.2 crawl=0
        Run B (Teacher3DNoCrawl-v0)  : scan=294,  terrain=5×0.2 crawl=0  ← this runner
        Teacher3D (Teacher3D-v0)     : scan=294,  terrain=5×0.15 crawl=0.25
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_teacher3d_nocrawl"


@configclass
class Go2ParkourImitationSymmetryRandomGoalTeacher3DVoxelPPOAMPRunnerCfg(
    Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg
):
    """Go2 ParkourImitation + symmetry + random-goal + Teacher 3D voxel-occupancy encoder.

    Voxel teacher arm for the clearance ablation experiment
    (design doc ``_workspace/parkour_imitation_lidar/voxel_teacher_arm_plan.md``, D1–D8).

    Inherits ``Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg`` in full:
    - ``OnPolicyRunnerParkourAMP`` reward fusion, spawn-scheduler, save/load
    - PPO + AMP discriminator (BCE loss)
    - L/R mirror data-augmentation (symmetry_cfg, num_aug=2)
    - AMP obs 49-dim/frame × 10 history = 490-dim (unchanged)
    - scan group = clearance-294 (critic controlled variable, D8)
    - Terrain mix identical to Teacher3D (parkour_crawl=0.25)

    Differences vs Teacher3D runner cfg:
    - class_name → ``OnPolicyRunnerParkourAMPVoxel`` (selects VoxelEncoder-based actor-critic)
    - policy.class_name → ``ActorCriticRMAVoxel``
    - obs_groups gains ``"voxel": ["voxel"]`` (routes flat 7371-dim grid to actor terrain encoder)
    - experiment_name → ``parkour_imitation_go2_teacher3d_voxel``

    critic obs: policy(46) + scan(294) + priv_explicit(6) + priv_latent(33) = 379 (same as Teacher3D).
    """

    def __post_init__(self):
        super().__post_init__()
        # Switch runner to the voxel-arm variant (selects ActorCriticRMAVoxel in _construct_algorithm).
        self.class_name = "OnPolicyRunnerParkourAMPVoxel"
        # Inform policy cfg of the new actor-critic class (informational; actual selection is via runner).
        self.policy.class_name = "ActorCriticRMAVoxel"
        # Add voxel obs group — actor terrain encoder reads this, critic is unaffected.
        # A fresh dict is created to avoid mutating the parent class-level default.
        self.obs_groups = {**self.obs_groups, "voxel": ["voxel"]}
        self.experiment_name = "parkour_imitation_go2_teacher3d_voxel"
        # ---- VRAM budget adjustment for voxel obs (7371-dim) ----
        # The voxel rollout buffer is 4096 × num_steps × 7371 × 4 bytes.
        # At num_steps=24: ~2.9 GB vs clearance scan (294-dim): ~116 MB.
        # Reducing to 16 steps saves ~967 MB and reducing num_mini_batches from 4→8
        # halves per-minibatch activation memory during the PPO update, together
        # keeping peak PyTorch allocation below the 23.53 GiB RTX-4090 budget
        # when running with --video (Isaac Sim non-PyTorch overhead ≈12.5 GB).
        self.num_steps_per_env = 16
        self.algorithm.num_mini_batches = 8


@configclass
class Go2ParkourImitationSymmetryRandomGoalLidarSLPPOAMPRunnerCfg(
    Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg
):
    """Go2 ParkourImitation + symmetry + random-goal + LiDAR student-only (SL) arm (R2).

    R2 LiDAR SL arm — end-to-end PPO+AMP where the actor's terrain encoder is a
    range-image CNN (``LidarEncoder``) instead of the clearance scandot MLP.
    The critic is unchanged from the Teacher3D arm (raw clearance-294, D8 controlled variable).

    Inherits ``Go2ParkourImitationSymmetryRandomGoalTeacher3DPPOAMPRunnerCfg`` in full:
    - PPO + AMP discriminator (BCE loss)
    - L/R mirror data-augmentation (symmetry_cfg, num_aug=2)
    - AMP obs 49-dim/frame × 10 history = 490-dim (unchanged)
    - 360° random-goal curriculum (±60° forward cone)
    - scan group = clearance-294 (critic controlled variable, D8)

    Differences vs Teacher3D runner cfg:
    - class_name → ``OnPolicyRunnerParkourAMPLidar``
    - policy.class_name → ``ActorCriticRMALidar``
    - obs_groups gains ``"lidar": ["lidar"]``
    - experiment_name → ``parkour_imitation_go2_lidar_sl``

    LidarEncoder defaults (no overrides needed):
        lidar_image_shape   = (24, 96)    H×W
        lidar_num_channels  = 2           ch0=range_norm, ch1=hit_mask
        lidar_frame_stack   = 3           K temporal frames
        → in_channels = 6, flat lidar dim = 6*24*96 = 13824

    actor input (computed at runtime):
        proprio(46) + priv_explicit(6) + priv_latent(20) + lidar_latent(32) = 104
    critic input:
        policy(46) + scan(294) + priv_explicit(6) + priv_latent(33) = 379

    Note: gym task registration is handled by the r2-env worker (obs["lidar"] is produced
    there).  Smoke test via direct instantiation (no full env required).
    """

    def __post_init__(self):
        super().__post_init__()
        # Switch runner to the LiDAR-arm variant (selects ActorCriticRMALidar in _construct_algorithm).
        self.class_name = "OnPolicyRunnerParkourAMPLidar"
        # Inform policy cfg (informational; actual selection is via runner._get_actor_critic_class).
        self.policy.class_name = "ActorCriticRMALidar"
        # Add lidar obs group — actor terrain encoder reads this, critic is unaffected.
        # A fresh dict is created to avoid mutating the parent class-level default.
        self.obs_groups = {**self.obs_groups, "lidar": ["lidar"]}
        self.experiment_name = "parkour_imitation_go2_lidar_sl"
        # Disable symmetry entirely for this first LiDAR SL run.
        # Even with use_data_augmentation=False and use_mirror_loss=False, PPO update still calls
        # data_augmentation_func + act_inference on a 2× mini-batch for logging — runs LidarEncoder
        # over a doubled CNN batch without no_grad, causing OOM at any practical num_envs.
        # The lidar range-image mirror (H-flip) is also not wired yet; re-enable symmetry once
        # _mirror_range_image is integrated into compute_parkour_imitation_symmetric_states.
        self.algorithm.symmetry_cfg = None

        # ── Entropy coefficient decay schedule (opt-in, PPOParkour) ──────────────────
        # debug-worker RCA: fixed entropy_coef=0.01 becomes an amplifier once the surrogate
        # gradient weakens late in training (~iter 8000+), driving action_std 0.59->1.42
        # in a runaway that ends in irreversible policy collapse. Decaying entropy_coef in
        # the back half of training removes that amplifier. This is opt-in via 4 cfg fields
        # (all-or-nothing) on PPOParkour/PPOAMP — every other runner cfg in this file leaves
        # them unset, so those tasks keep the original fixed entropy_coef behavior unchanged.
        #   it < 5000            : entropy_coef = 0.01   (unchanged from other tasks)
        #   5000 <= it <= 15000   : linear decay 0.01 -> 0.002
        #   it > 15000            : entropy_coef = 0.002
        self.algorithm.entropy_coef_start = 0.01
        self.algorithm.entropy_coef_end = 0.002
        self.algorithm.entropy_decay_start_iter = 5000
        self.algorithm.entropy_decay_end_iter = 15000

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

from isaaclab.utils.configclass import configclass

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
        entropy_coef=0.01,  # 0.01
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

    Note: the runner classes ``OnPolicyRunnerParkourAMPLidar`` / ``ActorCriticRMALidar`` are
    owned by the rsl_rl SL session — referenced here by class-name string only.
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


@configclass
class Go2ParkourImitationLidarDistillRunnerCfg(Go2ParkourImitationSymmetryRandomGoalLidarSLPPOAMPRunnerCfg):
    """Teacher-student distillation: frozen voxel teacher labels actions, LiDAR student imitates.

    Used by ``Go2-ParkourImitation-Lidar-Distill-EasyEntry-v0``.  The student rolls out the
    environment (on-policy DAgger, so the state distribution is the student's), the teacher
    scores the *same* observations deterministically, and the loss is the action MSE between
    them.  PPO's value/advantage/surrogate terms, the AMP discriminator, symmetry and RND are
    all unused on this path.

    Differences vs the LiDAR SL runner cfg:
    - ``class_name`` → ``OnPolicyRunnerParkourDistill``
    - ``obs_groups`` gains ``"voxel": ["voxel"]`` so the teacher can read its terrain input
      (the student still reads only ``"lidar"``)
    - ``teacher_checkpoint`` points at the trained voxel teacher
    - ``experiment_name`` → ``parkour_imitation_go2_lidar_distill``

    The paired env cfg
    (:class:`~isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_lidar_env_cfg.ParkourImitationRandomGoalLidarDistillEasyEntryEnvCfg`)
    emits ``obs["voxel"]`` and ``obs["lidar"]`` in the same step and matches the teacher's
    terrain mix — without that, teacher labels would be generated off-distribution.

    Note: this arm must be launched **without** ``--video``.  The replicator render path is
    what crashes the 6.0 LiDAR pipeline (rc=245); no-video runs are healthy.
    """

    teacher_checkpoint: str = (
        "/home/lgb/IsaacLab-6.0/logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel/"
        "2026-07-27_10-34-04_teacher3d_voxel_binary_50k/model_49999.pt"
    )
    """Absolute path to the frozen voxel-teacher checkpoint.

    The actor weights live under ``model_state_dict`` (there is no ``actor_state_dict`` key in
    this save format), so the runner loads them explicitly rather than through rsl-rl's
    auto-detecting distillation loader.
    """

    def __post_init__(self):
        super().__post_init__()
        self.class_name = "OnPolicyRunnerParkourDistill"
        self.policy.class_name = "ActorCriticRMALidar"
        # Teacher reads voxel; student reads lidar. Both groups must be present in one env.
        self.obs_groups = {**self.obs_groups, "voxel": ["voxel"]}
        self.experiment_name = "parkour_imitation_go2_lidar_distill"
        # Inherited from the SL arm and still required here (LidarEncoder over a doubled batch).
        self.algorithm.symmetry_cfg = None
        # Storage holds the student obs sets only — lidar(13824) is stored, voxel(7371) is NOT.
        # ``OnPolicyRunnerParkourDistill.Storage`` deliberately excludes "voxel" because the
        # teacher consumes it during the rollout and never needs it again.  Keep the trim from
        # 24 anyway: lidar alone is 13,824 floats per env-step.
        self.num_steps_per_env = 16


@configclass
class ParkourPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """``RslRlPpoAlgorithmCfg`` + adaptive LR 스케줄의 하한/상한 노출.

    ``PPOParkour`` 의 adaptive 스케줄은 KL 이 ``desired_kl`` 의 2배를 넘으면 LR 을 1/1.5 배씩
    낮추는데, 그 하한이 기존에는 1e-5 로 하드코딩돼 있었다. 하한이 binding 이 되면 KL 초과를
    감지해도 LR 을 더 낮출 수 없어 정책이 계속 크게 움직인다. 두 값을 cfg 로 노출해
    task 별로 조정할 수 있게 한다. 기본값은 기존 하드코딩과 동일하다.
    """

    learning_rate_min: float = 1.0e-5
    """adaptive 스케줄이 LR 을 낮출 수 있는 하한."""

    learning_rate_max: float = 1.0e-2
    """adaptive 스케줄이 LR 을 올릴 수 있는 상한."""


@configclass
class Go2ParkourImitationSymmetryRandomGoalEasyEntryPPOAMPRunnerCfg(
    Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg
):
    """EasyEntry 지형 전용 — 파국 붕괴 대응 안정화 설정.

    run ``2026-07-27_09-36-56`` (33800→48279) 이 iter 47255 에서 파국 붕괴했다. 진단 결과:

    - 얕은 자력회복 붕괴 10회의 ``std_max`` 는 0.469~0.548 로 좁게 모여 있었고, 파국만
      0.681 → 0.930 으로 이 범위를 넘어선 채 복귀하지 않았다. std 미복귀가 유일한 판별 지표였다.
    - ``use_clipped_value_loss`` 는 이미 True 였으므로 value clipping 은 처방이 되지 못한다.
    - LR 은 파국 전 구간에서 1e-5 하한에 100% 정박해 있었다. adaptive 스케줄이 포화되어
      KL 초과를 감지하고도 더 낮출 수단이 없는 상태였다.

    조치는 두 가지로 제한한다. 원인 귀속이 가능해야 하므로 ``entropy_coef`` / ``desired_kl`` /
    ``use_clipped_value_loss`` 는 건드리지 않는다.

    1. ``max_grad_norm`` 1.0 → 0.5 — per-update 정책 이동량을 직접 제한. 저위험.
    2. ``learning_rate_min`` — LR 하한이 실제 병목인지 확인한 뒤 조정한다. 기본값(1e-5)으로
       두고 새로 추가된 ``Loss/kl`` 태그로 KL 이 ``desired_kl`` 을 근소하게 넘는지 자릿수로
       넘는지 관측한 다음 결정한다.

    ``std`` 상한 clamp 는 검토 후 기각했다. 이 런에서 std 상승(0.488→0.530 @36900)이 국소해
    탈출을 이끌어 자력 회복을 만든 사례가 두 번 있었으므로, 판별 지표를 눌러버리면 회복 기제까지
    함께 없애고 붕괴가 조용히 진행된다.
    """

    def __post_init__(self):
        super().__post_init__()
        base = self.algorithm
        # algorithm 을 교체하되 부모 __post_init__ 이 설정한 symmetry_cfg 를 반드시 보존한다.
        self.algorithm = ParkourPpoAlgorithmCfg(
            class_name=base.class_name,
            num_learning_epochs=base.num_learning_epochs,
            num_mini_batches=base.num_mini_batches,
            learning_rate=base.learning_rate,
            schedule=base.schedule,
            gamma=base.gamma,
            lam=base.lam,
            entropy_coef=base.entropy_coef,
            desired_kl=base.desired_kl,
            optimizer=base.optimizer,
            value_loss_coef=base.value_loss_coef,
            use_clipped_value_loss=base.use_clipped_value_loss,
            clip_param=base.clip_param,
            normalize_advantage_per_mini_batch=base.normalize_advantage_per_mini_batch,
            share_cnn_encoders=base.share_cnn_encoders,
            rnd_cfg=base.rnd_cfg,
            symmetry_cfg=base.symmetry_cfg,
            # ── 안정화 조치 ──
            max_grad_norm=0.5,
            learning_rate_min=1.0e-5,
        )
        self.experiment_name = "parkour_imitation_go2_symmetry_random_goal"


@configclass
class Go2ParkourImitationLidarDistillK10RunnerCfg(Go2ParkourImitationLidarDistillRunnerCfg):
    """Distillation runner for the widened-window arm (K=10, 1.0 s) — rung A0.

    Used by ``Go2-ParkourImitation-Lidar-Distill-K10-EasyEntry-v0``.  Paired with
    :class:`~isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_lidar_env_cfg.ParkourImitationRandomGoalLidarDistillK10EasyEntryEnvCfg`.

    No policy-side change is needed: ``ActorCriticRMALidar`` infers K from the width of
    ``obs["lidar"]``, so the env cfg is the single source of truth.

    VRAM.  The stored lidar observation grows 3.33x (13,824 -> 46,080 floats per env-step).
    At ``num_steps_per_env=16`` and 1024 envs the rollout storage for the lidar group alone
    goes from ~0.91 GB to ~3.02 GB (float32), a ~2.1 GB increase.  ``num_steps_per_env`` is
    deliberately left at 16 so the batch size matches the K=3 baseline and the A/B stays
    clean — if this does not fit, prefer a card with more free memory over changing the
    batch, since changing it would confound the comparison.

    Compare against ``parkour_imitation_go2_lidar_distill`` at matched iterations
    (``curriculum/mean_terrain_level_gap``: 2k 1.94 / 5k 2.13 / 10k 2.28).

    Launch **without** ``--video`` — the replicator render path crashes the 6.0 LiDAR
    pipeline (rc=245).
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_k10"


@configclass
class Go2ParkourImitationLidarDistillK5RunnerCfg(Go2ParkourImitationLidarDistillRunnerCfg):
    """Distillation runner for the 0.5 s window arm (K=5) — lower bracket of rung A0.

    Used by ``Go2-ParkourImitation-Lidar-Distill-K5-EasyEntry-v0``.  Together with the K=3
    baseline and the K=10 arm this gives a window-length dose-response curve
    (0.3 / 0.5 / 1.0 s) instead of a single comparison.

    VRAM: stored lidar grows 1.67x vs K=3 (13,824 -> 23,040 floats per env-step), about
    +0.6 GB of rollout storage at ``num_steps_per_env=16`` and 1024 envs.  ``num_steps_per_env``
    stays 16 so the batch matches every other arm.

    Launch **without** ``--video``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_k5"


@configclass
class Go2ParkourImitationLidarDistillK15RunnerCfg(Go2ParkourImitationLidarDistillRunnerCfg):
    """Distillation runner for the 1.5 s window arm (K=15) — upper bracket of rung A0.

    Used by ``Go2-ParkourImitation-Lidar-Distill-K15-EasyEntry-v0``.  Completes the
    window-length dose-response curve (0.3 / 0.5 / 1.0 / 1.5 s).

    VRAM: stored lidar is 5x the K=3 baseline (13,824 -> 69,120 floats per env-step), about
    +3.6 GB of rollout storage at ``num_steps_per_env=16`` and 1024 envs, putting the process
    near 16.1 GB.  ``num_steps_per_env`` stays 16 so the batch matches every other arm — if it
    does not fit, wait for a freer card rather than trimming the batch.

    Launch **without** ``--video``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_k15"


@configclass
class Go2ParkourImitationLidarDistillK10FixRunnerCfg(Go2ParkourImitationLidarDistillK10RunnerCfg):
    """K=10 with the post-reset stack warm-up removed — discriminator for the A0 trend.

    Used by ``Go2-ParkourImitation-Lidar-Distill-K10Fix-EasyEntry-v0``.  Identical to the K=10
    arm except that the env cfg sets ``lidar_fill_stack_on_reset``, so the only difference is
    whether the ring buffer starts an episode self-consistent or zero-filled.  Compare against
    both the K=3 baseline and the plain K=10 arm at matched iterations.

    Same VRAM as K=10 (~14.6 GB at 1024 envs).  Launch **without** ``--video``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_k10fix"


@configclass
class Go2ParkourImitationLidarDistillGridRunnerCfg(Go2ParkourImitationLidarDistillRunnerCfg):
    """Rung A1-0: the student reads a metric occupancy grid through the voxel CNN.

    Used by ``Go2-ParkourImitation-Lidar-Distill-Grid-EasyEntry-v0``, paired with
    :class:`~isaaclab_tasks.direct.parkour_imitation.parkour_imitation_random_goal_lidar_env_cfg.ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg`,
    which makes ``obs["lidar"]`` a 27x21x13 = 7371 binary occupancy grid in the teacher's frame.

    The student's encoder becomes :class:`ActorCriticRMAVoxel` — the same class as the teacher —
    so the two are architecturally identical and differ only in the grid's provenance:

    * student ``obs_groups["voxel"] = ["lidar"]``  -> grid scattered from its own Mid-360 hits
    * teacher ``teacher_obs_groups["voxel"] = ["voxel"]`` -> grid from privileged clearance rays

    Both groups stay declared so the runner's storage whitelist (which iterates the observation
    *set* names, ``["policy", "priv_explicit", "history", "lidar"]``) still resolves to the group
    ``"lidar"`` and keeps it in the rollout buffer.

    ``teacher_policy`` is intentionally left unset: the teacher and student now take identical
    policy kwargs, so the runner's fallback to the student's ``policy`` block is correct here
    rather than merely tolerated.
    """

    teacher_obs_groups: dict | None = None
    """Observation-set mapping used to build the teacher.

    Required whenever the student's mapping is not also valid for the teacher — here the student
    resolves ``"voxel"`` to the LiDAR-derived grid, so without this the teacher would be handed
    the student's grid and its frozen weights would be scoring the wrong input.
    """

    def __post_init__(self):
        super().__post_init__()
        # Authoritative since OnPolicyRunnerParkourDistill._get_actor_critic_class now dispatches
        # on this name (it used to be informational while the runner hard-coded the LiDAR class).
        self.policy.class_name = "ActorCriticRMAVoxel"
        # Student: the "voxel" set the voxel CNN reads is fed by the LiDAR-derived grid.
        self.obs_groups = {**self.obs_groups, "lidar": ["lidar"], "voxel": ["lidar"]}
        # Teacher: unchanged privileged grid.
        self.teacher_obs_groups = {**self.obs_groups, "voxel": ["voxel"]}
        self.experiment_name = "parkour_imitation_go2_lidar_distill_grid"


@configclass
class Go2ParkourImitationLidarDistillAccRunnerCfg(Go2ParkourImitationLidarDistillGridRunnerCfg):
    """Rung A1-1: metric grid plus pose-registered accumulation.

    Used by ``Go2-ParkourImitation-Lidar-Distill-Acc-EasyEntry-v0``.  Identical wiring to the
    A1-0 grid arm — same voxel-CNN student, same observation key and width — so the only
    difference from that arm is whether the grid carries memory.  Compare against A1-0 to isolate
    accumulation, and against the K=3 baseline for the end-to-end effect.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_acc"


@configclass
class Go2ParkourImitationLidarDistillCeilingRunnerCfg(Go2ParkourImitationLidarDistillGridRunnerCfg):
    """Ceiling control for the whole LiDAR-reconstruction programme.

    Used by ``Go2-ParkourImitation-Lidar-Distill-Ceiling-EasyEntry-v0``.  The runner wiring is
    byte-identical to A1-0 — same voxel-CNN student, same ``obs_groups["voxel"] = ["lidar"]``,
    same storage whitelist.  The paired env cfg is what differs: it fills ``obs["lidar"]`` with
    the teacher's privileged grid, so the student's terrain input has zero reconstruction error.

    Any LiDAR history/reconstruction module can at best recover that grid, so this arm's result
    is an upper bound on what such a module could ever buy.  Run it *before* designing one.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_ceiling"


@configclass
class Go2ParkourImitationLidarSLGridPPOAMPRunnerCfg(Go2ParkourImitationSymmetryRandomGoalTeacher3DVoxelPPOAMPRunnerCfg):
    """From-scratch PPO+AMP on the metric grid — no teacher, no distillation.

    Used by ``Go2-ParkourImitation-Lidar-SL-Grid-EasyEntry-v0``.  This re-opens the question the
    earlier ``parkour_imitation_go2_lidar_sl`` arm appeared to close.  That arm learned from
    scratch through the **angular range image**, the representation later measured to be the
    binding defect (swapping it for the metric grid moved pinned completion 0.078 -> 0.238 under
    distillation).  Its failure therefore does not carry over, and from-scratch RL deserves a
    second measurement on the representation that works.

    Why it matters beyond curiosity: distillation inherits a conditional-mean ceiling, because
    the student's grid comes from a real sensor while the teacher's comes from privileged
    clearance rays, making ``obs_student -> action_teacher`` a distribution rather than a
    function.  RL against the actual reward has no such ceiling — the policy optimises what is
    achievable under its *own* observability.  If this arm reaches teacher-level, the entire
    teacher-then-distill-then-finetune pipeline becomes unnecessary.

    Wiring vs the voxel teacher arm (the reference this must be read against):
    - ``obs_groups["voxel"] -> ["lidar"]``: the actor's voxel CNN reads the grid scattered from
      the robot's own Mid-360 hits instead of the privileged clearance grid.
    - critic is untouched (clearance-294 in the ``"scan"`` slot) — asymmetric actor-critic, which
      is what makes from-scratch tractable at all here.
    - ``symmetry_cfg = None``.  ``mdp/symmetry.py`` mirrors only policy/scan/priv/history, so the
      terrain grid would be left unmirrored while the proprioception is flipped, i.e. the
      augmented sample would carry an inconsistent label.  This deviates from the teacher's own
      recipe (which ran with symmetry on) and removes an augmentation, so a shortfall here is a
      candidate confound to check before concluding from-scratch fails.

    Pair with ``ParkourImitationRandomGoalLidarDistillGridEasyEntryEnvCfg`` — deliberately the
    *same* env cfg as the A1-0 distillation arm, so env, terrain mix and representation are all
    held fixed and the only difference is the learning objective.
    """

    def __post_init__(self):
        super().__post_init__()
        # Actor terrain encoder reads the LiDAR-derived grid; critic keeps privileged clearance.
        self.obs_groups = {**self.obs_groups, "lidar": ["lidar"], "voxel": ["lidar"]}
        self.experiment_name = "parkour_imitation_go2_lidar_sl_grid"
        # See the class docstring: the mirror function does not cover the terrain grid.
        self.algorithm.symmetry_cfg = None


@configclass
class Go2ParkourImitationLidarDistillR4RunnerCfg(Go2ParkourImitationLidarDistillRunnerCfg):
    """Distillation runner for the 4 m range-normalisation arm.

    Identical to :class:`Go2ParkourImitationLidarDistillRunnerCfg` except for the log directory,
    so the two arms never write into the same run folder and can be compared at matched iterations.
    Used by ``Go2-ParkourImitation-Lidar-Distill-R4-EasyEntry-v0``.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_distill_r4"


@configclass
class Go2ParkourImitationTeacher3DVoxelGTPPOAMPRunnerCfg(
    Go2ParkourImitationSymmetryRandomGoalTeacher3DVoxelPPOAMPRunnerCfg
):
    """Voxel teacher trained on a ground-truth volume instead of a ray-hit scatter.

    Used by ``Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0``.  Runner wiring is unchanged;
    only the paired env cfg differs (``voxel_gt_columns=True``), so this isolates the observation.

    The resulting checkpoint is **not** interchangeable with
    ``parkour_imitation_go2_teacher3d_voxel``: the actor's terrain input has different semantics
    and roughly 20x more occupied cells, so distillation arms measured against the older teacher
    form a separate baseline family.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_teacher3d_voxel_gt"


@configclass
class Go2ParkourImitationLidarSLGridCrawlPPOAMPRunnerCfg(Go2ParkourImitationLidarSLGridPPOAMPRunnerCfg):
    """SL-Grid trained on a terrain mix that includes ``parkour_crawl``.

    Used by ``Go2-ParkourImitation-Lidar-SL-Grid-Crawl-EasyEntry-v0``. Runner wiring is
    identical to the crawl-free SL-Grid arm — only the paired env cfg's terrain proportions
    differ — so any difference is attributable to the terrain distribution.

    Logged separately because the two are **not** the same baseline: a checkpoint trained with
    crawl at 0.16 cannot be placed in the same table as one that never saw the terrain.
    """

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "parkour_imitation_go2_lidar_sl_grid_crawl"

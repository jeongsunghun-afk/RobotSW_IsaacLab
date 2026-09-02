# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Leg(17-DOF) Imitation Tracking 환경용 PPO + AMP 러너 설정 (Simple 버전).

PPOAMPBase + ActorCritic + OnPolicyRunnerAMPBase 조합.
MimicKit amp_go2_task_agent.yaml 하이퍼파라미터 기반.

AMP reward mixing:
  - Stage 1 (0~5000 iter):  task_reward_lerp=1.0 → 순수 task reward
  - Stage 2 (5000~ iter):   task_reward_lerp→0.5 → 50% task + 50% AMP
  - annealing 5000 iter (= 5000×24=120000 steps)
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


@configclass
class LegImitationTrackingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Leg Imitation Tracking 환경용 PPO 러너 설정 (AMP Simple 버전)."""

    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = "leg_imitation_tracking"
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
        # std 발산 억제 (go2 버전 관측: reward 평탄한데 noise_std 가 0.25→4.25 단조발산)
        entropy_coef=0.005,
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
        # amp_observation_space = 59 (per-step) × num_amp_observations(=10) = 590
        # per-step 53 + 6 root_rot_tan_norm = 59, env_cfg.amp_observation_space=59
        # NOTE: 이 값은 fallback default일 뿐. OnPolicyRunnerAMPBase가 런타임에
        #       env.amp_observation_space.shape[0] (=590)로 덮어쓰므로 disc input_dim은 자동 파생.
        amp_observation_space=590,
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
        # ── 조건부 discriminator (env `amp_cond_mode` 와 짝) ──────
        # amp_cond_dim 은 러너가 env.amp_cond_dim 에서 읽어 넣는다 (여기 적지 않는다).
        amp_cond_dropout=0.1,  # disc 학습 시 조건을 지우는 행 비율 (무조건부 경로 동시 학습)
        amp_cond_reward_blend=0.0,  # 보상에 무조건부 평가를 섞는 비율 (0=조건부만)
        # ── Discriminator 구조 ──────────────────────────────────
        # "mlp"   : 기존 AMPDiscriminator (BCE / LS-GAN / WGAN + GP + logit reg)
        # "drail" : 확산 discriminator (LaCoLoco/DRAIL). bce/bce 전용, GP·logit reg 없음.
        disc_arch="mlp",  # "mlp" | "drail"
        drail_hidden_dims=[256, 256, 256, 256],
        drail_activation="elu",
        drail_diffusion_steps=1000,
        drail_label_dim=10,
        drail_sample_strategy="antithetic",  # "antithetic" | "uniform" | "constant"
    )


@configclass
class LegImitationTrackingRMAPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Leg Imitation Tracking + RMA Estimator 러너 설정.

    학습 스택은 ``parkour_imitation``의 RMA 계열을 참고:
        Runner:    OnPolicyRunnerParkourAMP  (estimator 자동 구성 + AMP reward fusion)
        Policy:    ActorCriticRMA            (history encoder + priv encoder + estimator)
        Algorithm: PPOAMP                    (PPOParkour + AMP discriminator)

    핵심:
      - Estimator 는 러너가 obs 로부터 자동 구성한다
        (input = obs["policy"](57), output = priv_explicit(6)).
        ``train_with_estimated_states=True`` 이므로 actor 는 추정 속도를 소비하고
        critic 은 ground-truth 속도를 본다.
      - AMP fusion: ``total = task_reward + amp_weight * flat_mask * disc_reward``.
        env 가 전부 평지라 ``_flat_env_mask`` 는 전부 True → AMP 전 env 적용.
        (부모 leg 러너의 ``task_reward_lerp`` 스케줄은 이 러너에서 **무시**된다.)
      - symmetry data-aug **적용**(``__post_init__``): 17-DOF leg L/R mirror 함수
        (``mdp.symmetry:compute_leg_symmetric_states``)로 좌우 대칭을 소프트하게 강제한다.
        num_aug=2 로 미니배치가 2배가 되며 KL adaptive 스케줄이 흡수한다. mirror 부호는
        SMR reference(대략 좌우대칭)로 실측 검증: hip·뒷발·waist=FLIP, thigh·calf·앞발=NO-FLIP.

    priv_latent 은 상수 placeholder(zeros)다 — DR 미적용이라 adaptation module 은 no-op.
    Estimator(속도 추정)는 이와 무관하게 학습된다.
    """

    # ── Runner ────────────────────────────────────────────────────────────────
    class_name: str = "OnPolicyRunnerParkourAMP"
    num_steps_per_env: int = 24
    max_iterations: int = 50000
    save_interval: int = 100
    experiment_name: str = "leg_imitation_tracking_rma"
    empirical_normalization: bool = True
    seed: int = 1

    # leg action scale(0.25) 기준 clip. 부모 leg 러너와 동일하게 4.0 유지(torque85 와 정합).
    clip_actions: float = 4.0

    # ── obs_groups: env dict obs → actor/critic/encoder 라우팅 ────────────────
    # env returns: {policy(57), priv_explicit(6), priv(placeholder), history(10×57)}
    #   scan 그룹 없음 → ActorCriticRMA 는 scandot_encoder 를 만들지 않는다(num_scan_obs=0).
    #   critic = policy + priv_explicit + priv (concat)
    obs_groups: dict = {
        "policy": ["policy"],
        "critic": ["policy", "priv_explicit", "priv"],
        "history": ["history"],
        "priv": ["priv"],
        "priv_explicit": ["priv_explicit"],
    }

    # ── Estimator: priv_explicit(base 선/각속도) 예측 ─────────────────────────
    estimator: dict = {
        "hidden_dims": [128, 64],
        "learning_rate": 1.0e-3,
        "train_with_estimated_states": True,
    }

    # ── Policy: ActorCriticRMA ────────────────────────────────────────────────
    # actor/critic_obs_normalization=True — baseline(비-RMA)과 동일하게 empirical normalization 사용.
    # 이유: 이 env 는 obs 에 큰 raw 값이 섞여 있다(base_mass ~40kg in priv_latent, joint_vel ~30rad/s,
    #   base 속도 in priv_explicit). parkour 는 이를 legged_gym 수동 스케일로 O(1) 화하지만 우리는 그 스케일을
    #   두지 않았다. EmpiricalNormalization 은 (x-mean)/std 로 **센터링+스케일**을 데이터에서 학습하므로
    #   수동 스케일보다 완전하고(예: base_mass 40 offset 을 평균으로 제거), baseline 과 전처리가 일치해
    #   RMA+DR 만 순수 변수로 격리된다. DR 런은 priv_latent 이 상수가 아니라 variance-0 문제도 없다.
    #   (estimator MLP 자체는 raw proprio→raw priv_explicit 회귀로 별개; 이미 raw 에서 수렴 확인됨.)
    # init_noise_std=0.25: leg 는 1.0 에서 std 가 단조 발산(→9.98)해 mean 정책이 거칠어진다.
    # baseline(비-RMA) 도 0.25 사용. (첫 RMA 런은 parkour 기본 1.0 을 써서 발산했다.)
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
        class_name="ActorCriticRMA",
        init_noise_std=0.25,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # ── Algorithm: PPOAMP ────────────────────────────────────────────────────
    # entropy_coef 는 leg 의 std 발산 억제값(0.005)을 유지 (parkour 기본 0.01 대비 보수적).
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        class_name="PPOAMP",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
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
    # amp_weight: additive fusion 가중치 (total = task + amp_weight * flat_mask * disc_reward).
    # amp_observation_space: 런너 초기화 시 env.amp_observation_space.shape[0](=590)로 덮어씀.
    # AMP 계수/fusion 을 baseline(LegImitationTrackingPPORunnerCfg)과 완전히 동일하게 맞춘다 — RMA+DR 효과만 격리.
    # fusion="lerp" + task_reward_lerp=0.5: baseline(OnPolicyRunnerAMPBase)의 fusion 과 동일한
    #   total = 0.5*task + 0.5*amp. (parkour 러너 기본 additive 를 lerp 모드로 전환 — on_policy_runner_parkour_amp
    #   에 추가한 플래그. amp_weight 는 lerp 모드에서 미사용.)
    # reward_coef=2.0 (baseline과 동일; parkour 기본 0.08 은 25배 약해 첫 런에서 모방이 붕괴했다).
    # 나머지 disc 하이퍼파라미터(lr/gp/hidden/epochs/logit_reg/loss/reward/norm_clip/replay)는 baseline 과 동일.
    amp: dict = dict(
        fusion="lerp",
        task_reward_lerp=0.5,
        amp_weight=1.0,
        discriminator_learning_rate=2.5e-4,
        gradient_penalty_coef=5.0,
        reward_coef=2.0,
        discriminator_hidden_dims=[1024, 512],
        disc_num_epochs=2,
        disc_mini_batch_size=4096,
        disc_logit_reg=0.01,
        disc_loss_type="bce",
        disc_reward_type="bce",
        disc_logit_reg_type="weight",
        disc_norm_clip=10.0,
        amp_cond_dropout=0.1,
        amp_cond_reward_blend=0.0,
        disc_arch="mlp",  # "mlp" | "drail"
        enable_replay_buffer=True,
        replay_buffer_size=200000,
        amp_observation_space=590,
        motion_files=None,
    )

    def __post_init__(self):
        # L/R mirror data-augmentation (num_aug=2). AMP obs travel via extras["amp_obs"] (separate
        # discriminator path), so they are NOT mirrored here.
        # use_mirror_loss defaults OFF (pure data-aug). Opt-in per-run via env vars:
        #   LEG_MIRROR_LOSS=1 LEG_MIRROR_LOSS_COEFF=1.0  → add explicit mirror-consistency loss.
        # (Rationale: data-aug alone let Loss/symmetry drift up unchecked → high-speed symmetry
        #  break; the mirror loss actively penalizes it. See reports symmetry_retrain README.)
        import os

        # AMP reward 비중 조절: LEG_TASK_REWARD_LERP 로 task 비중(=lerp) override (기본 0.5 = 50% task + 50% amp).
        # 값↑ → task 비중↑, AMP 비중↓ → 참조 밖 gait(예: 정지 출발) 탐색 여지↑. lerp fusion 전용.
        # (symmetry 조기 return 앞에 두어 LEG_SYMMETRY_AUG 값과 무관하게 항상 적용.)
        _lerp = os.environ.get("LEG_TASK_REWARD_LERP")
        if _lerp is not None:
            self.amp["task_reward_lerp"] = float(_lerp)

        # Full symmetry opt-out (default ON). Set LEG_SYMMETRY_AUG=0 to disable ALL symmetry
        # (no data-aug, no mirror loss) — e.g. when the reference dataset already contains
        # mirrored clips, so L/R symmetry is enforced at the data level via the AMP discriminator.
        # symmetry_cfg=None makes rsl_rl skip the entire symmetry path (ppo.py guard).
        if os.environ.get("LEG_SYMMETRY_AUG", "1") == "0":
            self.algorithm.symmetry_cfg = None
            return

        _mirror_loss = os.environ.get("LEG_MIRROR_LOSS", "0") == "1"
        _mirror_coeff = float(os.environ.get("LEG_MIRROR_LOSS_COEFF", "0.0"))
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=_mirror_loss,
            mirror_loss_coeff=_mirror_coeff,
            data_augmentation_func=(
                "isaaclab_tasks.direct.leg_imitation_tracking.mdp.symmetry:compute_leg_symmetric_states"
            ),
        )

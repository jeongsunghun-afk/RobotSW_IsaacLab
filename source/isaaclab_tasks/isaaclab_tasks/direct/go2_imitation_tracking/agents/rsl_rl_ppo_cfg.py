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

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlSymmetryCfg,
)


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

    def __post_init__(self):  # noqa: D105
        # ── 좌우 대칭 (기본 OFF) ────────────────────────────────────────────
        # 2026-09-03 실측: `cmd 3.5` thigh ROM 좌우차가 처치군 36.5 % (대조 7.2 % · MimicKit 16.5 %).
        # `|vy|`·`|yaw|` 는 셋이 비슷하므로 몸통 드리프트가 아니라 **다리 사용 쏠림**이다.
        #
        # ★ 데이터셋에는 이미 `*_mirror.pkl` 이 있고 클립 균등으로 뽑히는데도 쏠린다 — AMP 는
        #   0.2 s 창의 **분포**만 맞추고, expert 분포는 좌/우 편향 창의 합집합이라 한쪽으로
        #   쏠린 정책도 그 안에 들어간다. 미러 **데이터**로는 개별 롤아웃 대칭을 못 만든다.
        #   leg 계열에서 같은 실험이 실패했다(데이터셋 대칭오차는 줄었는데 정책 비대칭은
        #   9.4k −0.024 → 50k +0.646 단조 발산). 거기서 통한 것은 **mirror loss** 였다.
        #
        # 기존 run 재현을 위해 **기본은 완전 OFF**(symmetry_cfg=None → rsl_rl 이 경로 전체를 건너뜀).
        # run 단위로 환경변수로 켠다:
        #   GO2_SYMMETRY_AUG=1                      → data augmentation (mirror 샘플 추가)
        #   GO2_MIRROR_LOSS=1 GO2_MIRROR_LOSS_COEFF=1.0  → mirror-consistency loss 추가
        #
        # ★★★ 2026-09-06 정정 — 아래 예측은 **틀렸다.** 원래 여기에는 "data-aug 단독은 함수만
        #   equivariant 라 closed-loop 자발적 대칭붕괴를 못 막으니 mirror loss 를 같이 켜라" 고
        #   적혀 있었다. go2 에서 40k A/B 로 재 보니 **data-aug 단독으로 잡힌다**:
        #     · `cmd 3.5` thigh ROM 좌우차 33.0 % → 6.2 % (5.3 배). 대조군 7.2 % · MimicKit 16.5 %
        #       보다도 낮다. 시드 0·1 구간이 겹치지 않는다(부모 최저 19.2 > symaug 최고 9.4).
        #     · `Loss/symmetry` 는 6k 정점 0.0021 후 39k 0.0003 까지 **단조 감소**(7 배).
        #     · 대가는 총보상 동률(+0.1 %) · 낙상 2 % → 0~1 % · 최고속만 −0.09 m/s.
        #   leg 에서 미러 **데이터셋**이 실패한 것과 혼동하지 말 것 — 그건 expert 분포를 바꾼
        #   것이고, 이건 **정책 업데이트**를 대칭쌍으로 증강한 것이라 기전이 다르다.
        #   ⇒ go2 권장은 `GO2_SYMMETRY_AUG=1` 단독이고 **mirror loss 는 끈 채로 둔다.**
        #   (보고서 `_comparisons/mimickit_vs_60_actuator_limit/README.md` §34)
        import os

        _aug = os.environ.get("GO2_SYMMETRY_AUG", "0") == "1"
        _mirror = os.environ.get("GO2_MIRROR_LOSS", "0") == "1"
        if not (_aug or _mirror):
            self.algorithm.symmetry_cfg = None
            return

        from isaaclab_tasks.direct.go2_imitation_tracking.mdp.symmetry import compute_go2_symmetric_states

        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=_aug,
            use_mirror_loss=_mirror,
            mirror_loss_coeff=float(os.environ.get("GO2_MIRROR_LOSS_COEFF", "1.0")),
            data_augmentation_func=compute_go2_symmetric_states,
        )

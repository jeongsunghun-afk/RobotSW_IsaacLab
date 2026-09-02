# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""PPO + AMP discriminator.

두 클래스가 있다 — :class:`PPOAMPBase` (기본 ``ActorCritic`` + :class:`~rsl_rl.algorithms.ppo_legacy.PPO`)
와 :class:`PPOAMP` (``ActorCriticRMA`` + :class:`~rsl_rl.algorithms.ppo_parkour.PPOParkour`).
discriminator 관련 코드는 전부 :class:`_AMPDiscriminatorMixin` 하나에 있고 두 클래스는 그것을 섞는다.

discriminator 입력 레이아웃은 ``[kinematics | condition(cond_dim)]`` 이다. env 가 ``amp_cond_dim`` 을
내면 러너가 ``amp_cfg["amp_cond_dim"]`` 로 넘겨 주고, 조건 열은 정규화·gradient penalty 에서 제외된다.
"""

from __future__ import annotations

import warnings

import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms.ppo_legacy import PPO
from rsl_rl.algorithms.ppo_parkour import PPOParkour
from rsl_rl.modules.amp_diffusion_discriminator import AMPDiffusionDiscriminator
from rsl_rl.modules.amp_discriminator import AMPDiscriminator


class _AMPDiscriminatorMixin:
    """AMP discriminator 구성·갱신·replay buffer. ``self.device`` 와 multi-GPU 속성은 PPO 가 준다."""

    # PPO / PPOParkour 가 제공하는 속성 (타입 검사용 선언)
    device: str
    is_multi_gpu: bool
    gpu_world_size: int

    def _init_amp(self, amp_cfg: dict | None) -> None:
        if amp_cfg is None:
            amp_cfg = {}

        self.amp_task_reward_lerp = amp_cfg.get("task_reward_lerp", 0.5)
        self.amp_discriminator_lr = amp_cfg.get("discriminator_learning_rate", 1e-4)
        self.amp_gradient_penalty_coef = amp_cfg.get("gradient_penalty_coef", 10.0)
        self.amp_reward_coef = amp_cfg.get("reward_coef", 2.0)

        amp_obs_dim = amp_cfg.get("amp_observation_space", 105)
        # env 가 AMP obs 끝에 붙인 조건 열 수 (유효 비트 포함). 러너가 env 에서 읽어 넣는다.
        self.amp_cond_dim = int(amp_cfg.get("amp_cond_dim", 0))
        # disc 학습 시 조건을 지우는 비율 (classifier-free guidance 식). 조건부 D 에서만 의미.
        self.amp_cond_dropout = float(amp_cfg.get("amp_cond_dropout", 0.0))

        # Loss / Reward / Regularization 방식 선택
        self.disc_arch = amp_cfg.get("disc_arch", "mlp")  # "mlp" | "drail"
        self.disc_loss_type = amp_cfg.get("disc_loss_type", "ls_gan")  # "ls_gan" | "bce" | "wgan"
        self.disc_logit_reg_type = amp_cfg.get("disc_logit_reg_type", "logit")  # "logit" | "weight"
        _disc_reward_type = amp_cfg.get("disc_reward_type", "ls_gan")

        # WGAN 불일치 가드: loss_type=wgan인데 reward_type이 다르면 critic 출력에 잘못된 변환 적용됨.
        if self.disc_loss_type == "wgan" and _disc_reward_type != "wgan":
            raise AssertionError(
                f"disc_loss_type='wgan'이지만 disc_reward_type='{_disc_reward_type}'입니다. "
                "WGAN 사용 시 disc_reward_type도 'wgan'으로 설정해야 합니다."
            )
        if _disc_reward_type == "wgan" and self.disc_loss_type != "wgan":
            warnings.warn(
                f"disc_reward_type='wgan'이지만 disc_loss_type='{self.disc_loss_type}'입니다. "
                "reward 변환과 loss 종류가 불일치합니다. disc_loss_type도 'wgan'으로 설정을 권장합니다.",
                stacklevel=2,
            )

        if self.disc_arch == "drail":
            # DRAIL 은 logit = L_pi − L_M 을 내므로 BCE 손실 + bce 보상이 정확히 원본 정의다.
            if self.disc_loss_type != "bce" or _disc_reward_type != "bce":
                raise AssertionError(
                    f"disc_arch='drail' 은 disc_loss_type='bce', disc_reward_type='bce' 만 지원한다 "
                    f"(현재 {self.disc_loss_type}/{_disc_reward_type})."
                )
            self.discriminator = AMPDiffusionDiscriminator(
                input_dim=amp_obs_dim,
                hidden_dims=amp_cfg.get("drail_hidden_dims", [256, 256, 256, 256]),
                activation=amp_cfg.get("drail_activation", "elu"),
                device=self.device,
                disc_reward_type=_disc_reward_type,
                norm_clip=amp_cfg.get("disc_norm_clip"),
                cond_dim=self.amp_cond_dim,
                label_dim=amp_cfg.get("drail_label_dim", 10),
                diffusion_steps=amp_cfg.get("drail_diffusion_steps", 1000),
                sample_strategy=amp_cfg.get("drail_sample_strategy", "antithetic"),
                sample_strategy_value=amp_cfg.get("drail_sample_strategy_value", 0),
                paired_noise=amp_cfg.get("drail_paired_noise", True),
            )
            if self.amp_gradient_penalty_coef != 0.0 or amp_cfg.get("disc_logit_reg", 0.0) != 0.0:
                warnings.warn(
                    "disc_arch='drail' 은 gradient penalty 와 logit reg 를 쓰지 않는다 (LaCoLoco 원본과 동일). "
                    "설정값은 무시된다.",
                    stacklevel=2,
                )
        elif self.disc_arch == "mlp":
            self.discriminator = AMPDiscriminator(
                input_dim=amp_obs_dim,
                hidden_dims=amp_cfg.get("discriminator_hidden_dims", [1024, 512]),
                device=self.device,
                disc_reward_type=_disc_reward_type,
                norm_clip=amp_cfg.get("disc_norm_clip"),
                cond_dim=self.amp_cond_dim,
            )
        else:
            raise ValueError(f"disc_arch 는 'mlp' | 'drail' 중 하나여야 한다: {self.disc_arch}")
        self.discriminator.amp_reward_coef = self.amp_reward_coef
        self.discriminator.cond_reward_blend = float(amp_cfg.get("amp_cond_reward_blend", 0.0))

        # AdamW with decoupled weight decay — CASSI/WASABI-faithful critic anchor.
        # disc_weight_decay=0.0 (default) → AdamW(wd=0) ≡ Adam → 기존 bce/ls_gan run 수치 불변.
        # WGAN 사용 시 ~5e-4 (CASSI default) 또는 1e-3 (논문 Table S5)로 설정 권장.
        self.disc_optimizer = optim.AdamW(
            self.discriminator.parameters(),
            lr=self.amp_discriminator_lr,
            weight_decay=amp_cfg.get("disc_weight_decay", 0.0),
        )

        # Replay Buffer (catastrophic forgetting 방지)
        self.enable_replay_buffer = amp_cfg.get("enable_replay_buffer", True)
        self.replay_buffer_size = amp_cfg.get("replay_buffer_size", 100000)
        self._replay_buffer: torch.Tensor | None = None
        self._replay_buffer_ptr: int = 0
        self._replay_buffer_full: bool = False

        # disc_num_epochs: discriminator를 iteration당 반복 업데이트 횟수
        self.disc_num_epochs = amp_cfg.get("disc_num_epochs", 2)
        # enable_lerp_schedule: task_reward_lerp annealing 활성화 여부
        self.enable_lerp_schedule = amp_cfg.get("enable_lerp_schedule", True)
        # disc_logit_reg: discriminator 출력 레이어 L2 정규화 (MimicKit 방식)
        self.disc_logit_reg = amp_cfg.get("disc_logit_reg", 0.0)

    # ── Replay buffer ─────────────────────────────────────────
    def add_to_replay_buffer(self, policy_obs: torch.Tensor) -> None:
        """Policy AMP obs를 replay buffer에 추가 (circular)."""
        if not self.enable_replay_buffer:
            return
        obs = policy_obs.detach()
        n = obs.shape[0]
        if self._replay_buffer is None:
            obs_dim = obs.shape[-1]
            self._replay_buffer = torch.zeros(self.replay_buffer_size, obs_dim, device=self.device)
        buf_size = self._replay_buffer.shape[0]
        if self._replay_buffer_ptr + n <= buf_size:
            self._replay_buffer[self._replay_buffer_ptr : self._replay_buffer_ptr + n] = obs
        else:
            first = buf_size - self._replay_buffer_ptr
            self._replay_buffer[self._replay_buffer_ptr :] = obs[:first]
            self._replay_buffer[: n - first] = obs[first:]
            self._replay_buffer_full = True
        self._replay_buffer_ptr = (self._replay_buffer_ptr + n) % buf_size

    def sample_replay_buffer(self, num_samples: int) -> torch.Tensor | None:
        """Replay buffer에서 num_samples개 샘플링. 데이터 부족 시 None 반환."""
        if self._replay_buffer is None:
            return None
        valid_size = self._replay_buffer.shape[0] if self._replay_buffer_full else self._replay_buffer_ptr
        if valid_size < num_samples:
            return None
        idx = torch.randint(0, valid_size, (num_samples,), device=self.device)
        return self._replay_buffer[idx]

    # ── Discriminator update ──────────────────────────────────
    def _apply_cond_dropout(self, batch: torch.Tensor) -> torch.Tensor:
        """배치의 일부 행에서 조건 열(유효 비트 포함)을 0 으로 지운다. 무조건부 경로를 함께 학습시킨다."""
        if self.amp_cond_dim == 0 or self.amp_cond_dropout <= 0.0:
            return batch
        out = batch.clone()
        mask = torch.rand(out.shape[0], device=out.device) < self.amp_cond_dropout
        out[mask, self.discriminator.kin_dim :] = 0.0
        return out

    def _gradient_penalty(self, batch: torch.Tensor) -> torch.Tensor:
        """R1 gradient penalty. 조건 열은 제외하고 kinematics 열의 기울기만 벌한다."""
        data = batch.detach().requires_grad_(True)
        logits = self.discriminator.get_logits(data)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=data,
            grad_outputs=torch.ones_like(logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grads = grads[:, : self.discriminator.kin_dim]
        return grads.norm(2, dim=1).pow(2).mean()

    def update_amp(self, expert_batch, policy_batch):
        """판별자(Discriminator) 업데이트.

        disc_loss_type에 따라 LS-GAN / BCE(MimicKit) / WGAN 방식 선택.
        disc_logit_reg_type에 따라 logit값 정규화 또는 출력 레이어 가중치 정규화 선택.
        ``disc_arch="drail"`` 이면 gradient penalty 와 logit reg 를 건너뛴다.
        """
        self.disc_optimizer.zero_grad()

        expert_batch = self._apply_cond_dropout(expert_batch)
        policy_batch = self._apply_cond_dropout(policy_batch)

        expert_logits = self.discriminator.get_logits(expert_batch)
        policy_logits = self.discriminator.get_logits(policy_batch)

        # ── Disc Loss ─────────────────────────────────────────────
        if self.disc_loss_type == "bce":
            # MimicKit 방식: BCE (expert→1, policy→0). DRAIL 도 같은 식 (logit = L_pi − L_M).
            bce = nn.BCEWithLogitsLoss()
            expert_loss = bce(expert_logits, torch.ones_like(expert_logits))
            policy_loss = bce(policy_logits, torch.zeros_like(policy_logits))
        elif self.disc_loss_type == "wgan":
            # WASABI(WGAN) critic loss — arXiv:2206.11693 Eq 2 / CASSI WassersteinLoss verbatim.
            # expert critic ↑ (minimize -E[D_exp]), policy critic ↓ (minimize E[D_pol]).
            # 0.5 스케일은 total_loss 결합식에서 흡수되어 수학적으로 일관.
            expert_loss = -expert_logits.mean()
            policy_loss = policy_logits.mean()
        else:
            # LS-GAN (기존): expert→+1, policy→-1
            expert_loss = nn.MSELoss()(expert_logits, torch.ones_like(expert_logits))
            policy_loss = nn.MSELoss()(policy_logits, -1 * torch.ones_like(policy_logits))

        use_regularizers = self.disc_arch != "drail"

        # ── Gradient Penalty (expert + policy 모두) ───────────────
        # WGAN: 논문은 expert-only coef 5.0이지만 양방향 zero-centered R1 GP 인프라를 그대로 재사용.
        # both-sided이므로 effective penalty가 크다 — wgan 사용 시 gradient_penalty_coef=5.0 권장 (기본 10.0).
        if use_regularizers:
            grad_penalty = (
                0.5
                * self.amp_gradient_penalty_coef
                * (self._gradient_penalty(expert_batch) + self._gradient_penalty(policy_batch))
            )
        else:
            grad_penalty = torch.zeros((), device=self.device)

        # ── Logit Regularization ──────────────────────────────────
        if not use_regularizers:
            logit_reg = torch.zeros((), device=self.device)
        elif self.disc_logit_reg_type == "weight":
            # MimicKit 방식: 출력 레이어 가중치 L2 정규화
            w = self.discriminator.get_output_layer_weights()
            logit_reg = self.disc_logit_reg * torch.sum(w**2)
        else:
            # 기존 방식: 출력 logit 값 L2 정규화
            logit_reg = self.disc_logit_reg * (expert_logits.pow(2).mean() + policy_logits.pow(2).mean())

        total_loss = 0.5 * (expert_loss + policy_loss) + grad_penalty + logit_reg
        total_loss.backward()
        self.disc_optimizer.step()

        # update obs normalization (expert + policy 혼합으로 정규화)
        self.discriminator.update_normalization(torch.cat([expert_batch, policy_batch], dim=0))

        expert_loss_val = expert_loss.item()
        policy_loss_val = policy_loss.item()
        grad_penalty_val = grad_penalty.item()
        total_loss_val = total_loss.item()

        # ── Output Monitoring ─────────────────────────────────────────────
        # wgan: raw critic값 로깅 (sigmoid 적용 금지 — unbounded critic 값이 의미 있음).
        # bce만 sigmoid, 나머지(ls_gan/wgan)는 raw logit.
        if self.disc_loss_type == "bce":
            # sigmoid 적용하여 확률값 (0~1): expert→1, policy→0 → 수렴 시 둘 다 ~0.5
            expert_output_mean = torch.sigmoid(expert_logits).mean().item()
            policy_output_mean = torch.sigmoid(policy_logits).mean().item()
        else:
            # LS-GAN: raw logit (+1/-1 범위) / WGAN: raw critic값 (unbounded)
            expert_output_mean = expert_logits.mean().item()
            policy_output_mean = policy_logits.mean().item()

        del expert_logits, policy_logits, expert_loss, policy_loss, grad_penalty, logit_reg, total_loss

        return {
            "disc_total_loss": total_loss_val,
            "disc_expert_loss": expert_loss_val,
            "disc_policy_loss": policy_loss_val,
            "disc_grad_penalty": grad_penalty_val,
            "disc_expert_output": expert_output_mean,
            "disc_policy_output": policy_output_mean,
        }

    # ── multi-GPU ─────────────────────────────────────────────
    def broadcast_parameters(self) -> None:
        super().broadcast_parameters()
        # Broadcast discriminator as well
        if self.is_multi_gpu:
            disc_params = [self.discriminator.state_dict()]
            torch.distributed.broadcast_object_list(disc_params, src=0)
            self.discriminator.load_state_dict(disc_params[0])

    def reduce_parameters(self) -> None:
        super().reduce_parameters()
        # Reduce discriminator gradients
        if self.is_multi_gpu:
            disc_grads = [param.grad.view(-1) for param in self.discriminator.parameters() if param.grad is not None]
            if len(disc_grads) > 0:
                all_disc_grads = torch.cat(disc_grads)
                torch.distributed.all_reduce(all_disc_grads, op=torch.distributed.ReduceOp.SUM)
                all_disc_grads /= self.gpu_world_size

                offset = 0
                for param in self.discriminator.parameters():
                    if param.grad is not None:
                        numel = param.numel()
                        param.grad.data.copy_(all_disc_grads[offset : offset + numel].view_as(param.grad.data))
                        offset += numel


class PPOAMPBase(_AMPDiscriminatorMixin, PPO):
    """PPO 기반 AMP 알고리즘. history/priv encoder 없는 단순 버전.

    PPOAMP와 동일한 AMP discriminator 로직을 갖지만 PPO(ppo.py)를 상속하여
    ActorCriticRMA/PPOParkour 없이 기본 ActorCritic과 함께 사용할 수 있다.
    """

    def __init__(self, *args, amp_cfg=None, **kwargs):
        kwargs.pop("class_name", None)
        super().__init__(*args, **kwargs)
        self._init_amp(amp_cfg)

    def update_dagger(self) -> float:
        """PPOAMPBase는 history encoder가 없으므로 no-op stub."""
        return 0.0


class PPOAMP(_AMPDiscriminatorMixin, PPOParkour):
    """PPOParkour(ActorCriticRMA + estimator/dagger) 기반 AMP 알고리즘."""

    def __init__(self, *args, amp_cfg=None, **kwargs):
        kwargs.pop("class_name", None)
        super().__init__(*args, **kwargs)
        self._init_amp(amp_cfg)

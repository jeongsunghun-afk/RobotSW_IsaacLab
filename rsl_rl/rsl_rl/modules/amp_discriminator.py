import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[1024, 512],
        activation="relu",
        device="cuda",
        disc_reward_type="ls_gan",  # "ls_gan" | "bce" (MimicKit: -log(1-sigmoid(logit)))
        norm_clip=None,             # None=clip없음, 10.0=MimicKit 스타일
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.disc_reward_type = disc_reward_type
        self.norm_clip = norm_clip

        # AMP loss parameter
        self.amp_reward_coef = 1.5 # AMP style 보상 스케일

        # Empirical Normalizer (input_dim 수치만큼 정규화)
        self.amp_obs_normalizer = EmpiricalNormalization(input_dim).to(self.device)

        # discriminator network (input -> hidden_dims -> 1)
        self.trunk = MLP(input_dim, 1, hidden_dims, activation)
        self.trunk.to(self.device)

    def _normalize(self, amp_obs: torch.Tensor) -> torch.Tensor:
        """정규화 + 선택적 clip."""
        norm = self.amp_obs_normalizer(amp_obs)
        if self.norm_clip is not None:
            norm = torch.clamp(norm, -self.norm_clip, self.norm_clip)
        return norm

    def update_normalization(self, amp_obs: torch.Tensor) -> None:
        """Update the empirical normalizer statistics using real simulation observations."""
        self.amp_obs_normalizer.update(amp_obs)

    def compute_amp_reward(self, amp_obs):
        """판별자(Discriminator)를 통해 AMP 보상을 계산합니다.

        Args:
            amp_obs (torch.Tensor): agent 또는 expert의 모션 관측치. [Batch, amp_observation_size]
        Returns:
            torch.Tensor: 에이전트의 모션이 전문가 모션과 얼마나 유사한지에 대한 스칼라 보상
        """
        amp_obs = amp_obs.to(self.device)
        norm_obs = self._normalize(amp_obs)
        disc_logits = self.trunk(norm_obs)

        if self.disc_reward_type == "bce":
            # MimicKit 방식: -log(1 - sigmoid(logit))
            # logit → +∞ (expert-like): reward → +∞
            # logit → -∞ (fake): reward → 0
            prob = torch.sigmoid(disc_logits)
            reward = -torch.log(torch.clamp(1.0 - prob, min=1e-4))
        else:
            # LS-GAN reward: clamp(1 - (1/4)*(d-1)^2, min=0) · coef
            reward = torch.clamp(1 - 0.25 * torch.square(disc_logits - 1), min=0)

        return reward.squeeze(-1) * self.amp_reward_coef

    def get_logits(self, amp_obs):
        amp_obs = amp_obs.to(self.device)
        norm_obs = self._normalize(amp_obs)
        return self.trunk(norm_obs)

    def get_output_layer_weights(self) -> torch.Tensor:
        """MimicKit 방식 logit regularization용 — trunk 마지막 Linear 레이어 가중치 반환."""
        for m in reversed(list(self.trunk.modules())):
            if isinstance(m, nn.Linear):
                return m.weight.flatten()
        raise RuntimeError("Linear layer not found in trunk")

import torch
import torch.nn as nn
from rsl_rl.networks import MLP, EmpiricalNormalization


class AMPDiscriminator(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dims=[1024, 512],
        activation="relu",
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.input_dim = input_dim

        # AMP loss parameter
        self.amp_reward_coef = 1.5 # AMP style 보상 스케일
        
        # Empirical Normalizer (input_dim 수치만큼 정규화)
        self.amp_obs_normalizer = EmpiricalNormalization(input_dim).to(self.device)
        
        # discriminator network (input -> hidden_dims -> 1)
        self.trunk = MLP(input_dim, 1, hidden_dims, activation)
        self.trunk.to(self.device)

    def update_normalization(self, amp_obs: torch.Tensor) -> None:
        """Update the empirical normalizer statistics using real simulation observations."""
        self.amp_obs_normalizer.update(amp_obs)

    def compute_amp_reward(self, amp_obs):
        """판별자(Discriminator)를 통해 AMP 보상을 계산합니다.
        
        Args:
            amp_obs (torch.Tensor): agent 또는 expert의 모션 관측치. [Batch, amp_observation_size]
        Returns:
            torch.Tensor: 에이전트의 모션이 전문가 모션과 얼마나 유사한지에 대한 스칼라 보상 (0 ~ 1.0)
        """
        amp_obs = amp_obs.to(self.device)
        
        # 정규화 거친 뒤 판별자 통과
        norm_obs = self.amp_obs_normalizer(amp_obs)
        disc_logits = self.trunk(norm_obs)
        
        # 논문 수식 및 일반적 구현에 따른 보상 계산: r_s = exp(-c * max(0, 1 - D)) 또는 -log(1 - sigmoid(D))
        # 간단히 skrl 구현체와 유사하게 최소-최대 clip 등의 연산을 사용할 수 있습니다.
        # 여기서는 기본적으로 -log(1 - D(x)) 에 비례하는 형태로 단순화할 수도 있습니다.
        
        # LS-GAN reward: clamp(1 - (1/4)*(d-1)^2, min=0) · coef  (Genesis 원본 방식)
        reward = torch.clamp(1 - 0.25 * torch.square(disc_logits - 1), min=0)

        return reward.squeeze(-1) * self.amp_reward_coef

    def get_logits(self, amp_obs):
        amp_obs = amp_obs.to(self.device)
        norm_obs = self.amp_obs_normalizer(amp_obs)
        return self.trunk(norm_obs)

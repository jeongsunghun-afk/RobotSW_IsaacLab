import torch
import torch.nn as nn
import torch.optim as optim

from rsl_rl.algorithms import PPOParkour
from rsl_rl.modules.amp_discriminator import AMPDiscriminator
from tensordict import TensorDict

class PPOAMP(PPOParkour):
    def __init__(self, *args, amp_cfg=None, **kwargs):
        kwargs.pop("class_name", None)
        super().__init__(*args, **kwargs)
        
        # AMP Configuration
        if amp_cfg is None:
            amp_cfg = {}
            
        self.amp_task_reward_lerp = amp_cfg.get("task_reward_lerp", 0.5)
        self.amp_discriminator_lr = amp_cfg.get("discriminator_learning_rate", 1e-4)
        self.amp_gradient_penalty_coef = amp_cfg.get("gradient_penalty_coef", 10.0)
        self.amp_reward_coef = amp_cfg.get("reward_coef", 2.0)
        
        # Determine AMP observation size (fallback to a typical flat size if not provided)
        amp_obs_dim = amp_cfg.get("amp_observation_space", 105) 
        
        self.discriminator = AMPDiscriminator(
            input_dim=amp_obs_dim,
            hidden_dims=amp_cfg.get("discriminator_hidden_dims", [1024, 512]),
            device=self.device
        )
        self.discriminator.amp_reward_coef = self.amp_reward_coef

        # Discriminator Optimizer
        self.disc_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.amp_discriminator_lr
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

    def update_amp(self, expert_batch, policy_batch):
        """판별자(Discriminator) 업데이트
        
        Args:
            expert_batch (torch.Tensor): 모션 모방을 위한 정답 데이터 (Reference motions)
            policy_batch (torch.Tensor): 에이전트가 생성한 AMP 모션 로그 
        """
        self.disc_optimizer.zero_grad()

        # 모델 예측
        expert_logits = self.discriminator.get_logits(expert_batch)
        policy_logits = self.discriminator.get_logits(policy_batch)

        # LS-GAN: expert target=+1, policy target=-1
        expert_loss = nn.MSELoss()(expert_logits, torch.ones_like(expert_logits))
        policy_loss = nn.MSELoss()(policy_logits, -1 * torch.ones_like(policy_logits))

        # Gradient Penalty: expert + policy 양쪽에 적용
        expert_data = expert_batch.detach().requires_grad_(True)
        expert_logits_gp = self.discriminator.get_logits(expert_data)
        gradients_expert = torch.autograd.grad(
            outputs=expert_logits_gp,
            inputs=expert_data,
            grad_outputs=torch.ones_like(expert_logits_gp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty_expert = gradients_expert.norm(2, dim=1).pow(2).mean()

        policy_data = policy_batch.detach().requires_grad_(True)
        policy_logits_gp = self.discriminator.get_logits(policy_data)
        gradients_policy = torch.autograd.grad(
            outputs=policy_logits_gp,
            inputs=policy_data,
            grad_outputs=torch.ones_like(policy_logits_gp),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_penalty_policy = gradients_policy.norm(2, dim=1).pow(2).mean()

        grad_penalty = 0.5 * self.amp_gradient_penalty_coef * (grad_penalty_expert + grad_penalty_policy)

        total_loss = 0.5 * (expert_loss + policy_loss) + grad_penalty

        total_loss.backward()
        self.disc_optimizer.step()

        # update obs normalization (expert + policy 혼합으로 정규화)
        self.discriminator.update_normalization(torch.cat([expert_batch, policy_batch], dim=0))

        # 명시적 메모리 해제 (VRAM 누수 방지)
        expert_loss_val = expert_loss.item()
        policy_loss_val = policy_loss.item()
        grad_penalty_val = grad_penalty.item()
        total_loss_val = total_loss.item()

        del expert_data, expert_logits_gp, gradients_expert
        del policy_data, policy_logits_gp, gradients_policy
        del expert_logits, policy_logits, expert_loss, policy_loss, grad_penalty, total_loss
        
        return {
            "disc_total_loss": total_loss_val,
            "disc_expert_loss": expert_loss_val,
            "disc_policy_loss": policy_loss_val,
            "disc_grad_penalty": grad_penalty_val
        }
        
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

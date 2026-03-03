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

        # Discriminator 손실 계산 (Least Squares GAN 방식 튜닝 또는 기본 로그 로스)
        expert_loss = nn.BCEWithLogitsLoss()(expert_logits, torch.ones_like(expert_logits))
        policy_loss = nn.BCEWithLogitsLoss()(policy_logits, torch.zeros_like(policy_logits))

        # Gradient Penalty 연산 (WGAN-GP 변형 기법: expert 동작 부근에서 판별자 굴곡을 완만하게)
        expert_batch_gp = expert_batch.detach().requires_grad_(True)
        expert_logits_gp = self.discriminator.get_logits(expert_batch_gp)
        grad_outputs = torch.ones_like(expert_logits_gp)
        gradients = torch.autograd.grad(
            outputs=expert_logits_gp,
            inputs=expert_batch_gp,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,  # backward 후에도 penalty grad 계산 위해 필요
            only_inputs=True,
        )[0]
        grad_penalty = torch.sum(torch.square(gradients), dim=-1).mean()

        total_loss = 0.5 * (expert_loss + policy_loss) + (self.amp_gradient_penalty_coef * 0.5) * grad_penalty

        total_loss.backward()
        self.disc_optimizer.step()
        
        # 명시적 메모리 해제 (VRAM 누수 방지)
        expert_loss_val = expert_loss.item()
        policy_loss_val = policy_loss.item()
        grad_penalty_val = grad_penalty.item()
        total_loss_val = total_loss.item()
        
        del expert_batch_gp, expert_logits_gp, grad_outputs, gradients
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

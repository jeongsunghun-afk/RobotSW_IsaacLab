import time
import os
import torch

from rsl_rl.runners.on_policy_runner_parkour import OnPolicyRunnerParkour
from rsl_rl.algorithms import PPOAMP
from rsl_rl.modules import ActorCriticRMA
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils.logger import Logger
from tensordict import TensorDict

class OnPolicyRunnerAMP(OnPolicyRunnerParkour):
    """AMP 학습 기능을 지원하는 OnPolicyRunner."""
    
    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir, device)
        # AMP용 파라미터 체크 및 로더 연결 (환경에서 모션 추출)
        self.amp_cfg = train_cfg.get("amp", {})
        
        # AMP Reward 로깅용 버퍼
        self.amp_reward_sums = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def _construct_algorithm(self, obs: TensorDict) -> PPOAMP:
        """PPOAMP 알고리즘 생성"""
        # AMP 특화 알고리즘
        actor_critic_class = ActorCriticRMA
        actor_critic = actor_critic_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        amp_cfg = self.cfg.get("amp", {})
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "amp_observation_space"):
            amp_cfg["amp_observation_space"] = self.env.unwrapped.amp_observation_space.shape[0]
        elif hasattr(self.env, "amp_observation_space"):
            amp_cfg["amp_observation_space"] = self.env.amp_observation_space.shape[0]

        alg = PPOAMP(
            actor_critic, storage, device=self.device, amp_cfg=amp_cfg,
            **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )
        return alg

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # Initializations
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
            
        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            self.alg.broadcast_parameters()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        
        amp_obs_buffer = []

        for it in range(start_it, total_it):
            start = time.time()
            hist_encoding = it % 20 == 0 # dagger_update_freq
            amp_obs_buffer.clear()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # Sample actions
                    actions = self.alg.act(obs, hist_encoding=hist_encoding)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    
                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)
                    
                    # AMP: Extract discriminator rewards & append buffer
                    if "amp_obs" in extras:
                        agent_amp_obs = extras["amp_obs"].to(self.device)
                        amp_obs_buffer.append(agent_amp_obs.detach())
                        
                        amp_reward = self.alg.discriminator.compute_amp_reward(agent_amp_obs)
                        # 로깅 버퍼 합산
                        self.amp_reward_sums += amp_reward
                        
                        # Reward fusion
                        task_reward_lerp = self.alg.amp_task_reward_lerp
                        total_reward = task_reward_lerp * rewards + (1.0 - task_reward_lerp) * amp_reward
                    else:
                        total_reward = rewards

                    # 에피소드 종료 환경 식별 및 AMP Reward 평균 기록
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        if "log" not in extras:
                            extras["log"] = dict()
                        ep_lens = getattr(self.env, "max_episode_length_s", getattr(self.env, "episode_length_s", 20.0))
                        avg_amp_rew = torch.mean(self.amp_reward_sums[done_ids]) / ep_lens
                        extras["log"]["Episode_Reward/amp_reward"] = avg_amp_rew
                        self.amp_reward_sums[done_ids] = 0.0

                    # Process the step
                    self.alg.process_env_step(obs, total_reward, dones, extras)
                    
                    self.logger.process_env_step(total_reward, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            # --- AMP Discriminator Update ---
            if len(amp_obs_buffer) > 0 and hasattr(self.env, "get_amp_observations"):
                # Concatenate accumulated batched policy motions
                policy_amp_obs_batch = torch.cat(amp_obs_buffer, dim=0)
                
                # Fetch expert reference motion samples from env (or dataset)
                # Env should provide a method to sample experts -> e.g., get_amp_observations(num_samples)
                num_samples = policy_amp_obs_batch.shape[0]

                expert_amp_obs_batch = self.env.get_amp_observations(num_samples).to(self.device)
                
                # Unfrozen/Update discriminator
                self.train_mode() 
                for param in self.alg.discriminator.parameters():
                    param.requires_grad = True
                
                amp_loss_dict = self.alg.update_amp(expert_amp_obs_batch, policy_amp_obs_batch)
                
                # Freeze Discriminator for Actor-Critic Update
                for param in self.alg.discriminator.parameters():
                    param.requires_grad = False
            else:
                amp_loss_dict = {}
            

            # Update Policy & Critic
            loss_dict = self.alg.update()
            loss_dict.update(amp_loss_dict)
            loss_dict["hist_latent_loss"] = self.alg.update_dagger()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=None,
            )

            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "disc_optimizer_state_dict": self.alg.disc_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, map_location=map_location)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", {})

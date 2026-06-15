# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPOAMP, PPOAMPBase
from rsl_rl.modules import ActorCritic, ActorCriticRMA, resolve_symmetry_config
from rsl_rl.runners.on_policy_runner_parkour import OnPolicyRunnerParkour
from rsl_rl.storage import RolloutStorage


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
        # Resolve symmetry config if used (inject env into symmetry_cfg["_env"]).
        # NOTE: this override previously dropped the parent's resolve call, causing
        # KeyError '_env' in PPOParkour.update() when symmetry_cfg was set.
        # resolve_symmetry_config leaves symmetry_cfg as None when it is None/absent,
        # so the non-symmetry path (Go2-ParkourImitation-v0) is unaffected.
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        # AMP 특화 알고리즘
        actor_critic_class = ActorCriticRMA
        actor_critic = actor_critic_class(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(
            self.device
        )

        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        amp_cfg = self.cfg.get("amp", {})
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "amp_observation_space"):
            amp_cfg["amp_observation_space"] = self.env.unwrapped.amp_observation_space.shape[0]
        elif hasattr(self.env, "amp_observation_space"):
            amp_cfg["amp_observation_space"] = self.env.amp_observation_space.shape[0]

        alg = PPOAMP(
            actor_critic, storage, device=self.device, amp_cfg=amp_cfg, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
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

        # Lerp annealing 파라미터 (Stage 1: task 위주 → Stage 2: AMP 도입)
        _amp_cfg = self.cfg.get("amp", {})
        _lerp_end = _amp_cfg.get("task_reward_lerp", self.alg.amp_task_reward_lerp)
        _lerp_start = _amp_cfg.get("task_reward_lerp_start", _lerp_end)
        _anneal_iters = _amp_cfg.get("task_reward_lerp_anneal_iters", 0)
        _disc_mini_batch_size = _amp_cfg.get("disc_mini_batch_size", 4096)

        amp_obs_buffer = []

        for it in range(start_it, total_it):
            # task_reward_lerp 스케줄 업데이트 (Stage 1 → Stage 2)
            if self.alg.enable_lerp_schedule and _anneal_iters > 0:
                progress = min(1.0, (it - start_it) / _anneal_iters)
                self.alg.amp_task_reward_lerp = _lerp_start + (_lerp_end - _lerp_start) * progress

            start = time.time()
            hist_encoding = it % 20 == 0  # dagger_update_freq
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

                        # terminal step: post-reset RSI obs → pre-reset terminal obs로 교체.
                        # extras["amp_obs"]는 _get_observations() 이후 값이므로
                        # done env는 RSI 상태(새 에피소드) 기반 obs임.
                        # _terminal_amp_obs는 _reset_idx() 진입 시점(실패 에피소드 마지막) obs.
                        if "terminal_amp_obs" in extras and dones.any():
                            terminal_amp_obs = extras["terminal_amp_obs"].to(self.device)
                            corrected_amp_obs = agent_amp_obs.clone()
                            corrected_amp_obs[dones.bool()] = terminal_amp_obs[dones.bool()]
                        else:
                            corrected_amp_obs = agent_amp_obs

                        amp_obs_buffer.append(corrected_amp_obs.detach())

                        amp_reward = self.alg.discriminator.compute_amp_reward(corrected_amp_obs).detach()
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
                        # 실제 에피소드 길이(step 수)로 평균 계산
                        ep_lens = getattr(self.env, "max_episode_length_s", getattr(self.env, "episode_length_s", 20.0))
                        avg_amp_rew = torch.mean(self.amp_reward_sums[done_ids]) / ep_lens
                        # step_dt = getattr(self.env, "step_dt", 1.0 / 30.0)
                        # actual_ep_lens = self.env.episode_length_buf[done_ids].float() * step_dt
                        # actual_ep_lens = torch.clamp(actual_ep_lens, min=step_dt)
                        # avg_amp_rew = torch.mean(self.amp_reward_sums[done_ids] / actual_ep_lens)
                        extras["log"]["Episode_Reward/amp_reward"] = avg_amp_rew
                        self.amp_reward_sums[done_ids] = 0.0

                    # Process the step
                    self.alg.process_env_step(obs, total_reward, dones, extras)

                    self.logger.process_env_step(total_reward, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            # --- AMP Discriminator Update (MimicKit mini-batch 방식) ---
            if len(amp_obs_buffer) > 0 and hasattr(self.env, "get_amp_observations"):
                policy_amp_obs_batch = torch.cat(amp_obs_buffer, dim=0)
                num_samples = policy_amp_obs_batch.shape[0]

                self.alg.add_to_replay_buffer(policy_amp_obs_batch)

                for param in self.alg.discriminator.parameters():
                    param.requires_grad = True

                # Policy pool: current iteration + replay (한 번만 샘플링)
                replayed = self.alg.sample_replay_buffer(num_samples)
                if replayed is not None:
                    policy_pool = torch.cat([policy_amp_obs_batch, replayed], dim=0)
                else:
                    policy_pool = policy_amp_obs_batch
                num_pool = policy_pool.shape[0]

                # Expert pool: 한 번만 fetch (GPU 효율)
                expert_pool = self.env.get_amp_observations(num_pool).to(self.device)

                # Mini-batch disc 학습 (MimicKit 방식: epoch × mini-batch)
                amp_loss_dict = {}
                for _ in range(self.alg.disc_num_epochs):
                    perm = torch.randperm(num_pool, device=self.device)
                    for batch_start in range(0, num_pool, _disc_mini_batch_size):
                        batch_end = min(batch_start + _disc_mini_batch_size, num_pool)
                        batch_ids = perm[batch_start:batch_end]
                        amp_loss_dict = self.alg.update_amp(expert_pool[batch_ids], policy_pool[batch_ids])
            else:
                amp_loss_dict = {}

            # Update Policy & Critic
            loss_dict = self.alg.update()
            loss_dict.update(amp_loss_dict)
            loss_dict["hist_latent_loss"] = self.alg.update_dagger()
            loss_dict["amp_task_reward_lerp"] = self.alg.amp_task_reward_lerp  # type: ignore[attr-defined]

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
        disc_missing, disc_unexpected = self.alg.discriminator.load_state_dict(
            loaded_dict["discriminator_state_dict"], strict=False
        )
        non_normalizer_missing = [k for k in disc_missing if "reward_normalizer" not in k]
        if non_normalizer_missing:
            print(f"[WARNING] discriminator load: unexpected missing keys: {non_normalizer_missing}")
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", {})

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        self.alg.discriminator.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        self.alg.discriminator.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()


class OnPolicyRunnerAMPBase(OnPolicyRunnerAMP):
    """AMP 학습 러너 단순 버전 — history encoder / dagger 없이 기본 ActorCritic + PPOAMPBase 사용."""

    def _construct_algorithm(self, obs) -> PPOAMPBase:
        """ActorCritic + PPOAMPBase 조합으로 알고리즘 생성."""
        actor_critic = ActorCritic(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(self.device)

        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        amp_cfg = self.cfg.get("amp", {})
        if hasattr(self.env, "unwrapped") and hasattr(self.env.unwrapped, "amp_observation_space"):
            amp_cfg["amp_observation_space"] = self.env.unwrapped.amp_observation_space.shape[0]
        elif hasattr(self.env, "amp_observation_space"):
            amp_cfg["amp_observation_space"] = self.env.amp_observation_space.shape[0]

        alg = PPOAMPBase(
            actor_critic, storage, device=self.device, amp_cfg=amp_cfg, **self.alg_cfg, multi_gpu_cfg=self.multi_gpu_cfg
        )
        return alg

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        """hist_encoding / update_dagger 없는 단순 AMP 학습 루프."""
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

        _amp_cfg = self.cfg.get("amp", {})
        _lerp_end = _amp_cfg.get("task_reward_lerp", self.alg.amp_task_reward_lerp)
        _lerp_start = _amp_cfg.get("task_reward_lerp_start", _lerp_end)
        _anneal_iters = _amp_cfg.get("task_reward_lerp_anneal_iters", 0)
        _disc_mini_batch_size = _amp_cfg.get("disc_mini_batch_size", 4096)

        amp_obs_buffer = []

        for it in range(start_it, total_it):
            if self.alg.enable_lerp_schedule and _anneal_iters > 0:
                progress = min(1.0, (it - start_it) / _anneal_iters)
                self.alg.amp_task_reward_lerp = _lerp_start + (_lerp_end - _lerp_start) * progress

            start = time.time()
            amp_obs_buffer.clear()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    if "amp_obs" in extras:
                        agent_amp_obs = extras["amp_obs"].to(self.device)

                        # terminal step: post-reset RSI obs → pre-reset terminal obs로 교체.
                        # extras["amp_obs"]는 _get_observations() 이후 값이므로
                        # done env는 RSI 상태(새 에피소드) 기반 obs임.
                        # _terminal_amp_obs는 _reset_idx() 진입 시점(실패 에피소드 마지막) obs.
                        if "terminal_amp_obs" in extras and dones.any():
                            terminal_amp_obs = extras["terminal_amp_obs"].to(self.device)
                            corrected_amp_obs = agent_amp_obs.clone()
                            corrected_amp_obs[dones.bool()] = terminal_amp_obs[dones.bool()]
                        else:
                            corrected_amp_obs = agent_amp_obs

                        amp_obs_buffer.append(corrected_amp_obs.detach())

                        amp_reward = self.alg.discriminator.compute_amp_reward(corrected_amp_obs).detach()
                        self.amp_reward_sums += amp_reward

                        task_reward_lerp = self.alg.amp_task_reward_lerp
                        total_reward = task_reward_lerp * rewards + (1.0 - task_reward_lerp) * amp_reward
                    else:
                        total_reward = rewards

                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        if "log" not in extras:
                            extras["log"] = dict()

                        ep_lens = getattr(self.env, "max_episode_length_s", getattr(self.env, "episode_length_s", 20.0))
                        avg_amp_rew = torch.mean(self.amp_reward_sums[done_ids]) / ep_lens
                        # step_dt = getattr(self.env, "step_dt", 1.0 / 30.0)
                        # actual_ep_lens = self.env.episode_length_buf[done_ids].float() * step_dt
                        # actual_ep_lens = torch.clamp(actual_ep_lens, min=step_dt)
                        # avg_amp_rew = torch.mean(self.amp_reward_sums[done_ids] / actual_ep_lens)
                        extras["log"]["Episode_Reward/amp_reward"] = avg_amp_rew
                        self.amp_reward_sums[done_ids] = 0.0

                    self.alg.process_env_step(obs, total_reward, dones, extras)
                    self.logger.process_env_step(total_reward, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            # --- AMP Discriminator Update (MimicKit mini-batch 방식) ---
            if len(amp_obs_buffer) > 0 and hasattr(self.env, "get_amp_observations"):
                policy_amp_obs_batch = torch.cat(amp_obs_buffer, dim=0)
                num_samples = policy_amp_obs_batch.shape[0]

                self.alg.add_to_replay_buffer(policy_amp_obs_batch)

                self.train_mode()
                for param in self.alg.discriminator.parameters():
                    param.requires_grad = True

                # Policy pool: current iteration + replay (한 번만 샘플링)
                replayed = self.alg.sample_replay_buffer(num_samples)
                if replayed is not None:
                    policy_pool = torch.cat([policy_amp_obs_batch, replayed], dim=0)
                else:
                    policy_pool = policy_amp_obs_batch
                num_pool = policy_pool.shape[0]

                # Expert pool: 한 번만 fetch (GPU 효율)
                expert_pool = self.env.get_amp_observations(num_pool).to(self.device)

                # Mini-batch disc 학습 (MimicKit 방식: epoch × mini-batch)
                amp_loss_dict = {}
                for _ in range(self.alg.disc_num_epochs):
                    perm = torch.randperm(num_pool, device=self.device)
                    for batch_start in range(0, num_pool, _disc_mini_batch_size):
                        batch_end = min(batch_start + _disc_mini_batch_size, num_pool)
                        batch_ids = perm[batch_start:batch_end]
                        amp_loss_dict = self.alg.update_amp(expert_pool[batch_ids], policy_pool[batch_ids])
            else:
                amp_loss_dict = {}

            loss_dict = self.alg.update()
            loss_dict.update(amp_loss_dict)
            loss_dict["amp_task_reward_lerp"] = self.alg.amp_task_reward_lerp

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

    def train_mode(self) -> None:
        # PPO
        self.alg.policy.train()
        self.alg.discriminator.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        # PPO
        self.alg.policy.eval()
        self.alg.discriminator.train()
        # RND
        if self.alg_cfg["rnd_cfg"]:
            self.alg.rnd.eval()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Hybrid runner: OnPolicyRunnerParkour + AMP discriminator.

Inherits from OnPolicyRunnerAMP (which already extends OnPolicyRunnerParkour) and overrides
three behaviours specific to parkour_imitation:

1. Reward fusion — additive + masked:
       total = task_reward + amp_weight * flat_env_mask.float() * disc_reward
   (instead of the lerp-based fusion in OnPolicyRunnerAMP)
   NOTE: task_reward_lerp in amp cfg is IGNORED for this runner. Set amp_weight in
   train_cfg["amp"]["amp_weight"] (default 0.3) to control AMP contribution.

2. Spawn scheduler callback — calls env.update_iteration(it) after every policy
   iteration so the obs-worker's flat_bias curriculum can advance.

3. Save / load completeness — saves discriminator + estimator + their optimisers so
   checkpoints can be resumed after either PPO-only or AMP-augmented runs.

flat_env_mask interface (contract with obs-worker)
──────────────────────────────────────────────────
The env MUST expose:
    env.unwrapped._flat_env_mask : torch.BoolTensor  shape [num_envs]

This tensor is True for every env currently spawned on a flat sub-terrain and is
updated inside env._reset_idx() each time an episode ends. The runner reads it once
per rollout step (inside torch.inference_mode) and uses it to gate the AMP reward
contribution. If the attribute is missing at the start of learn(), an informative
RuntimeError is raised immediately — the runner does NOT silently fall back to
treating all envs as flat.

AMP observation interface (contract with obs-worker)
─────────────────────────────────────────────────────
env MUST expose:
    env.unwrapped.amp_observation_space : gym.Space  (shape[0] == amp_obs_dim × history)
    env.get_amp_observations(n) : torch.Tensor  shape [n, amp_obs_dim × history]
extras from env.step() MUST include:
    extras["amp_obs"]         : torch.Tensor  shape [num_envs, amp_obs_dim × history]
    extras["terminal_amp_obs"]: torch.Tensor  shape [num_envs, amp_obs_dim × history]  (optional)
"""

from __future__ import annotations

import os
import time
import torch

from rsl_rl.modules.actor_critic_parkour import ActorCriticRMALidar, ActorCriticRMAVoxel
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP
from rsl_rl.runners.on_policy_runner_parkour import _rename_joint_keys


class OnPolicyRunnerParkourAMP(OnPolicyRunnerAMP):
    """Parkour-AMP hybrid runner.

    Extends OnPolicyRunnerAMP with additive+masked reward fusion, a spawn-scheduler
    callback, and complete checkpoint save/load (disc + estimator + optimisers).
    """

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir, device)

        # AMP weight for flat-env reward gating (additive fusion).
        # Read from train_cfg["amp"]["amp_weight"]; falls back to env cfg if set there.
        amp_cfg = train_cfg.get("amp", {})
        self.amp_weight: float = float(amp_cfg.get("amp_weight", 0.3))

    # ──────────────────────────────────────────────────────────────────────────
    # learn() — full override with additive+masked fusion & spawn callback
    # ──────────────────────────────────────────────────────────────────────────

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        # Validate flat_env_mask interface before spending time on rollouts.
        _unwrapped = self.env.unwrapped
        if not hasattr(_unwrapped, "_flat_env_mask"):
            raise RuntimeError(
                "OnPolicyRunnerParkourAMP: env.unwrapped._flat_env_mask not found. "
                "obs-worker must expose a BoolTensor [num_envs] named _flat_env_mask."
            )

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
        _disc_mini_batch_size = _amp_cfg.get("disc_mini_batch_size", 4096)

        amp_obs_buffer: list[torch.Tensor] = []

        for it in range(start_it, total_it):
            start = time.time()
            hist_encoding = it % 20 == 0  # dagger_update_freq
            amp_obs_buffer.clear()

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, hist_encoding=hist_encoding)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    if "amp_obs" in extras:
                        agent_amp_obs = extras["amp_obs"].to(self.device)
                        # Read flat_mask once; reused for both disc-buffer filter and reward fusion.
                        flat_mask = _unwrapped._flat_env_mask.to(self.device)  # bool [num_envs]

                        # Terminal step correction: replace post-reset RSI obs with
                        # the pre-reset terminal obs so the disc sees real failure data.
                        # Must run before flat-env filtering so terminal obs from flat envs
                        # are also correctly sourced.
                        if "terminal_amp_obs" in extras and dones.any():
                            terminal_amp_obs = extras["terminal_amp_obs"].to(self.device)
                            corrected_amp_obs = agent_amp_obs.clone()
                            corrected_amp_obs[dones.bool()] = terminal_amp_obs[dones.bool()]
                        else:
                            corrected_amp_obs = agent_amp_obs

                        # ── Disc training: flat envs only (Option B) ─────────
                        # Filtering here means policy_amp_obs_batch is flat-only by
                        # construction; add_to_replay_buffer (post-rollout) needs no change.
                        if flat_mask.any():
                            amp_obs_buffer.append(corrected_amp_obs[flat_mask].detach())

                        # ── Reward fusion: additive + flat-env mask ──────────
                        disc_reward = self.alg.discriminator.compute_amp_reward(corrected_amp_obs).detach()
                        amp_contribution = self.amp_weight * flat_mask.float() * disc_reward

                        # Logging: raw disc_reward over all envs (unmasked) so the
                        # diagnostic visibility of non-flat disc behaviour is preserved.
                        # NOTE: logged metric includes non-flat envs where disc is OOD
                        # after this change — if per-flat-env logging is needed, replace
                        # disc_reward below with (disc_reward * flat_mask.float()).
                        self.amp_reward_sums += disc_reward

                        total_reward = rewards + amp_contribution
                    else:
                        total_reward = rewards

                    # Episode-end logging
                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        if "log" not in extras:
                            extras["log"] = {}
                        ep_lens = getattr(self.env, "max_episode_length_s", getattr(self.env, "episode_length_s", 20.0))
                        avg_amp_rew = torch.mean(self.amp_reward_sums[done_ids]) / ep_lens
                        extras["log"]["Episode_Reward/amp_reward"] = avg_amp_rew
                        self.amp_reward_sums[done_ids] = 0.0

                    self.alg.process_env_step(obs, total_reward, dones, extras)
                    self.logger.process_env_step(total_reward, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            # ── AMP Discriminator Update ──────────────────────────────────────
            if len(amp_obs_buffer) > 0 and hasattr(self.env, "get_amp_observations"):
                policy_amp_obs_batch = torch.cat(amp_obs_buffer, dim=0)
                num_samples = policy_amp_obs_batch.shape[0]

                self.alg.add_to_replay_buffer(policy_amp_obs_batch)

                for param in self.alg.discriminator.parameters():
                    param.requires_grad = True

                # Policy pool: current + replay
                replayed = self.alg.sample_replay_buffer(num_samples)
                if replayed is not None:
                    policy_pool = torch.cat([policy_amp_obs_batch, replayed], dim=0)
                else:
                    policy_pool = policy_amp_obs_batch
                num_pool = policy_pool.shape[0]

                expert_pool = self.env.get_amp_observations(num_pool).to(self.device)

                amp_loss_dict: dict = {}
                for _ in range(self.alg.disc_num_epochs):
                    perm = torch.randperm(num_pool, device=self.device)
                    for batch_start in range(0, num_pool, _disc_mini_batch_size):
                        batch_end = min(batch_start + _disc_mini_batch_size, num_pool)
                        batch_ids = perm[batch_start:batch_end]
                        amp_loss_dict = self.alg.update_amp(expert_pool[batch_ids], policy_pool[batch_ids])
            else:
                amp_loss_dict = {}

            # ── Policy + Estimator Update ────────────────────────────────────
            # Wire current iteration into the algorithm for PPOParkour's opt-in entropy_coef
            # decay schedule (only active when the algorithm cfg sets the 4 schedule params;
            # a no-op attribute-set for every other task, see ppo_parkour.py `current_iteration`).
            self.alg.current_iteration = it
            loss_dict = self.alg.update()
            loss_dict.update(amp_loss_dict)
            loss_dict["hist_latent_loss"] = self.alg.update_dagger()
            loss_dict["amp_weight"] = self.amp_weight  # logged for reference

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # ── Spawn scheduler callback ──────────────────────────────────────
            # Notifies the env's flat_bias curriculum to advance. If the env does
            # not implement update_iteration(), the call is silently skipped (hasattr guard).
            if hasattr(self.env.unwrapped, "update_iteration"):
                self.env.unwrapped.update_iteration(it)

            # ── Logging ───────────────────────────────────────────────────────
            # Separate per-joint stats (keys containing "/") from scalar losses.
            grouped_stats = {k: v for k, v in loss_dict.items() if "/" in k}
            scalar_loss_dict = {k: v for k, v in loss_dict.items() if "/" not in k}

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=scalar_loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.policy.action_std,
                rnd_weight=None,
            )

            if self.logger.writer is not None and not self.logger.disable_logs:
                named_stats = _rename_joint_keys(grouped_stats, self.joint_names)
                for key, value in named_stats.items():
                    self.logger.writer.add_scalar(key, value, it)

            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    # ──────────────────────────────────────────────────────────────────────────
    # save / load — disc + estimator + all optimisers
    # ──────────────────────────────────────────────────────────────────────────

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict: dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "discriminator_state_dict": self.alg.discriminator.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "disc_optimizer_state_dict": self.alg.disc_optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        # Estimator (present in PPOAMP/PPOParkour lineage)
        if hasattr(self.alg, "estimator") and self.alg.estimator is not None:
            saved_dict["estimator_state_dict"] = self.alg.estimator.state_dict()
        if hasattr(self.alg, "estimator_optimizer") and self.alg.estimator_optimizer is not None:
            saved_dict["estimator_optimizer_state_dict"] = self.alg.estimator_optimizer.state_dict()
        # RND (optional)
        if self.alg_cfg.get("rnd_cfg") and hasattr(self.alg, "rnd"):
            saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
            if hasattr(self.alg, "rnd_optimizer") and self.alg.rnd_optimizer is not None:
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        resumed = self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        self.alg.discriminator.load_state_dict(loaded_dict["discriminator_state_dict"])
        if "estimator_state_dict" in loaded_dict and hasattr(self.alg, "estimator"):
            self.alg.estimator.load_state_dict(loaded_dict["estimator_state_dict"])
        if load_optimizer and resumed:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.alg.disc_optimizer.load_state_dict(loaded_dict["disc_optimizer_state_dict"])
            if "estimator_optimizer_state_dict" in loaded_dict and hasattr(self.alg, "estimator_optimizer"):
                self.alg.estimator_optimizer.load_state_dict(loaded_dict["estimator_optimizer_state_dict"])
            if self.alg_cfg.get("rnd_cfg") and "rnd_optimizer_state_dict" in loaded_dict:
                self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if self.alg_cfg.get("rnd_cfg") and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)
        return loaded_dict.get("infos", {})

    # ──────────────────────────────────────────────────────────────────────────
    # train_mode / eval_mode — fix inherited eval_mode typo
    # ──────────────────────────────────────────────────────────────────────────

    def train_mode(self) -> None:
        self.alg.policy.train()
        self.alg.discriminator.train()
        if self.alg_cfg.get("rnd_cfg") and hasattr(self.alg, "rnd"):
            self.alg.rnd.train()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        self.alg.discriminator.eval()  # fixed: parent had .train() here by mistake
        if self.alg_cfg.get("rnd_cfg") and hasattr(self.alg, "rnd"):
            self.alg.rnd.eval()


class OnPolicyRunnerParkourAMPVoxel(OnPolicyRunnerParkourAMP):
    """Variant of OnPolicyRunnerParkourAMP that uses ActorCriticRMAVoxel as the policy.

    Voxel teacher arm for the clearance ablation experiment (design doc D1–D8 in
    ``_workspace/parkour_imitation_lidar/voxel_teacher_arm_plan.md``).

    The only change vs the parent is the actor-critic class: ``ActorCriticRMAVoxel``
    replaces ``ActorCriticRMA`` so that the actor's terrain encoder is a voxel-occupancy
    CNN (``VoxelEncoder``) instead of the clearance scandot MLP.  The critic is identical
    to the clearance arm (raw clearance-294, D8) — critic capacity is the controlled variable.

    Everything else (reward fusion, flat_env_mask, spawn scheduler, save/load, disc/estimator)
    is inherited unchanged from ``OnPolicyRunnerParkourAMP``.
    """

    def _get_actor_critic_class(self):
        return ActorCriticRMAVoxel


class OnPolicyRunnerParkourAMPLidar(OnPolicyRunnerParkourAMP):
    """Variant of OnPolicyRunnerParkourAMP that uses ActorCriticRMALidar as the policy.

    R2 LiDAR student-only (SL) arm.

    The only change vs the parent is the actor-critic class: ``ActorCriticRMALidar``
    replaces ``ActorCriticRMA`` so that the actor's terrain encoder is a range-image CNN
    (``LidarEncoder``) instead of the clearance scandot MLP.  The critic is identical
    to the clearance arm (raw clearance-294, D8) — critic capacity is the controlled variable.

    Everything else (reward fusion, flat_env_mask, spawn scheduler, save/load, disc/estimator)
    is inherited unchanged from ``OnPolicyRunnerParkourAMP``.
    """

    def _get_actor_critic_class(self):
        return ActorCriticRMALidar

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-space imitation 러너 — AMP disc reward 자리에 env 의 스타일 보상을 넣는다.

:class:`~rsl_rl.runners.on_policy_runner_amp.OnPolicyRunnerAMP` 와 같은 보상 융합 구조를
쓰되, discriminator 학습·replay buffer·expert 샘플링이 전부 빠진다. 스타일 보상은 env 가
동결 인코더로 계산해 ``extras["style_reward"]`` 로 내보낸다 (env 마다 다른 값이다 —
배치 상수는 PPO advantage 에서 value baseline 에 흡수되어 gradient 기여가 0 이다)::

    total = lerp * task + (1 - lerp) * (w * style + (1 - w) * substitute)

``w`` = ``extras["style_weight"]`` (없으면 1), ``substitute`` = ``extras["style_substitute"]``
(없으면 0) — 정지 명령 구간에서 스타일을 다른 신호로 갈아끼우는 기존 경로를 그대로 승계한다.
"""

from __future__ import annotations

import os
import time
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.ppo_latent import PPOLatent
from rsl_rl.env import VecEnv
from rsl_rl.modules import ActorCriticRMA, resolve_symmetry_config
from rsl_rl.modules.estimator import Estimator
from rsl_rl.runners.on_policy_runner_parkour import OnPolicyRunnerParkour
from rsl_rl.storage.rollout_storage_legacy import RolloutStorage


class OnPolicyRunnerLatent(OnPolicyRunnerParkour):
    """PPOLatent 러너. save/load 는 부모(estimator 포함)를 그대로 쓴다."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """``style`` 블록을 읽고 스타일 보상 로깅 버퍼를 만든다."""
        super().__init__(env, train_cfg, log_dir, device)
        self.style_cfg = train_cfg.get("style", {})
        self.style_reward_sums = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

    def _construct_algorithm(self, obs: TensorDict) -> PPOLatent:
        """PPOLatent 생성 (RMA actor-critic + estimator, discriminator 없음)."""
        self.alg_cfg = resolve_symmetry_config(self.alg_cfg, self.env)
        self.alg_cfg.pop("class_name", None)

        actor_critic = ActorCriticRMA(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(
            self.device
        )
        storage = RolloutStorage(
            "rl", self.env.num_envs, self.cfg["num_steps_per_env"], obs, [self.env.num_actions], self.device
        )

        estimator_cfg = getattr(self, "estimator_cfg", None)
        if estimator_cfg is None:
            raise ValueError("OnPolicyRunnerLatent 는 estimator cfg 를 요구한다 (RMA 배포 경로 승계).")
        estimator = Estimator(
            input_dim=obs["policy"].shape[-1],
            output_dim=sum(obs[k].shape[-1] for k in self.cfg["obs_groups"]["priv_explicit"]),
            hidden_dims=estimator_cfg["hidden_dims"],
            activation=self.policy_cfg.get("activation", "elu"),
        ).to(self.device)

        return PPOLatent(
            actor_critic,
            storage,
            device=self.device,
            style_cfg=self.cfg.get("style", {}),
            estimator=estimator,
            estimator_cfg=estimator_cfg,
            **self.alg_cfg,
            multi_gpu_cfg=self.multi_gpu_cfg,
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """롤아웃 → 보상 융합 → PPO 갱신."""
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

        for it in range(start_it, total_it):
            self.alg.update_lerp_schedule(it)

            start = time.time()
            hist_encoding = it % 20 == 0  # dagger_update_freq
            # ★ 롤아웃 **전체** 평균으로 낸다. 마지막 스텝 하나만 쓰면 iter 대표값으로 읽혀
            #   오해를 부른다. `latent_kl` 은 창이 찬 env 만 센다 (안 찬 env 는 0 이라 평균이
            #   아래로 편향된다).
            st_sum, st_sq, st_n = 0.0, 0.0, 0
            kl_sum, kl_n = 0.0, 0

            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs, hist_encoding=hist_encoding)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))

                    obs = obs.to(self.device)
                    rewards = rewards.to(self.device)
                    dones = dones.to(self.device)

                    if "style_reward" in extras:
                        style_reward = extras["style_reward"].to(self.device)
                        self.style_reward_sums += style_reward

                        # env 가 특정 구간에서 스타일을 다른 보상으로 대체할 수 있다(선택).
                        style_term = style_reward
                        if "style_weight" in extras:
                            style_w = extras["style_weight"].to(self.device)
                            substitute = extras.get("style_substitute")
                            substitute = (
                                torch.zeros_like(style_reward) if substitute is None else substitute.to(self.device)
                            )
                            style_term = style_w * style_reward + (1.0 - style_w) * substitute

                        lerp = self.alg.task_reward_lerp
                        total_reward = lerp * rewards + (1.0 - lerp) * style_term

                        # 진단: env 간 산포가 0 이면 PPO advantage 가 이 항을 못 본다.
                        st_sum += float(style_reward.mean())
                        st_sq += float(style_reward.std())
                        st_n += 1
                        if "latent_kl" in extras and "style_ready" in extras:
                            ready = extras["style_ready"].to(self.device).bool()
                            if ready.any():
                                kl_sum += float(extras["latent_kl"].to(self.device)[ready].mean())
                                kl_n += 1
                    else:
                        total_reward = rewards

                    done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    if len(done_ids) > 0:
                        if "log" not in extras:
                            extras["log"] = dict()
                        ep_lens = getattr(self.env, "max_episode_length_s", getattr(self.env, "episode_length_s", 20.0))
                        extras["log"]["Episode_Reward/style_reward"] = (
                            torch.mean(self.style_reward_sums[done_ids]) / ep_lens
                        )
                        self.style_reward_sums[done_ids] = 0.0

                    self.alg.process_env_step(obs, total_reward, dones, extras)
                    self.logger.process_env_step(total_reward, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop

                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            loss_dict["hist_latent_loss"] = self.alg.update_dagger()
            loss_dict["task_reward_lerp"] = self.alg.task_reward_lerp
            if st_n > 0:
                loss_dict["style_reward_mean"] = st_sum / st_n
                loss_dict["style_reward_std"] = st_sq / st_n
            if kl_n > 0:
                loss_dict["latent_kl_mean"] = kl_sum / kl_n

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

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
                for key, value in grouped_stats.items():
                    self.logger.writer.add_scalar(key, value, it)

            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""go2_imitation_tracking RMA+estimator+AMP 정책을 **단일 deployable jit**로 export.

표준 exporter(export_policy_as_jit)는 actor MLP만 뽑아 estimator/history_encoder가 빠진다.
이 스크립트는 세 모듈을 하나로 묶어 self-contained jit를 만든다 (export_deployable_bipedleg.py의
구조를 그대로 따르되, runner=OnPolicyRunnerAMP / algorithm=PPOAMP로 교체):

    입력: proprio (B, 45), history (B, 10, 45)
    내부: estimator(proprio)→priv_explicit(3), history_encoder(history)→history_latent(20),
          actor([norm(proprio), norm(priv_explicit), history_latent])→action(12)
    출력: action (B, 12)

priv_explicit(root_lin_vel_b)는 실측 불가하므로 estimator가 policy(45)로부터 추정한다
(학습 시 train_with_estimated_states=True 와 동일 경로). act_inference는 obs["priv_explicit"]를
그대로 읽는 gap이 있으므로(GT 필요), 배포용은 이를 estimator(proprio)로 대체한 것이 핵심 차이다.

Isaac 앱 없이 순수 torch + rsl_rl 로 로드한다 (mock env). conda isaac-6.0 python.

실행:
    CUDA_VISIBLE_DEVICES=0 ./isaaclab.sh -p scripts/real2sim/export_deployable_go2.py \
        --run_dir logs/rsl_rl/go2_imitation_tracking/2026-07-24_13-03-00 \
        --checkpoint model_19.pt
    # → <run_dir>/exported/deployable_policy.pt  (torch.jit)
"""

from __future__ import annotations

import argparse
import inspect
import os

import torch
import torch.nn as nn
import yaml

POLICY_DIM, HISTORY_LEN, NUM_JOINTS = 45, 10, 12
PRIV_EXPLICIT_DIM, PRIV_LATENT_DIM, AMP_OBS_DIM = 3, 19, 490


class DeployablePolicy(nn.Module):
    """actor + estimator + history_encoder 를 묶은 self-contained 정책.

    forward(proprio(B,45), history(B,10,45)) → action(B,12). priv_explicit 는 estimator(proprio)로
    추정한다 (학습 train_with_estimated_states 와 동일; base 선속도 실측 불필요 — 실배포 핵심).
    """

    def __init__(self, policy, estimator):
        super().__init__()
        self.actor = policy.actor
        self.history_encoder = policy.history_encoder
        self.estimator = estimator
        self.actor_obs_normalizer = policy.actor_obs_normalizer
        self.priv_explicit_obs_normalizer = policy.priv_explicit_obs_normalizer
        self.history_obs_normalizer = policy.history_obs_normalizer

    def forward(self, proprio: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        obs_actor = self.actor_obs_normalizer(proprio)
        priv_explicit = self.priv_explicit_obs_normalizer(self.estimator(proprio))
        hist_flat = history.reshape(history.shape[0], -1)
        history_latent = self.history_encoder(self.history_obs_normalizer(hist_flat))
        x = torch.cat([obs_actor, priv_explicit, history_latent], dim=-1)
        return self.actor(x)


def _load_runner(run_dir: str, ckpt: str, device: str):
    from rsl_rl.algorithms.ppo_parkour import PPOParkour
    from rsl_rl.runners import OnPolicyRunnerAMP
    from tensordict import TensorDict

    ckpt_path = os.path.join(run_dir, ckpt)
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        tc = yaml.safe_load(f)
    tc["algorithm"].setdefault("rnd_cfg", None)
    tc["algorithm"].setdefault("symmetry_cfg", None)
    # PPOAMP.__init__ forwards **kwargs (minus amp_cfg) straight to PPOParkour.__init__, so filter
    # tc["algorithm"] against PPOParkour's signature (amp_cfg is injected separately below).
    acc = set(inspect.signature(PPOParkour.__init__).parameters)
    tc["algorithm"] = {k: v for k, v in tc["algorithm"].items() if k in acc or k == "class_name"}

    class MockEnv:
        def __init__(self):
            self.num_envs, self.num_actions, self.device = 1, NUM_JOINTS, device
            self.cfg = type("C", (), {})()
            self._robot = type(
                "R", (), {"data": type("D", (), {"joint_names": [f"joint_{i}" for i in range(NUM_JOINTS)]})()}
            )()
            self.max_episode_length = 1000
            # OnPolicyRunnerAMP._construct_algorithm reads env.unwrapped.amp_observation_space.shape[0]
            # to size the discriminator input — only .shape is accessed, no gym dependency needed.
            self.amp_observation_space = type("Space", (), {"shape": (AMP_OBS_DIM,)})()

        @property
        def unwrapped(self):
            return self

        def get_observations(self):
            return TensorDict(
                {
                    "policy": torch.zeros(1, POLICY_DIM, device=device),
                    "priv_explicit": torch.zeros(1, PRIV_EXPLICIT_DIM, device=device),
                    "priv_latent": torch.zeros(1, PRIV_LATENT_DIM, device=device),
                    "history": torch.zeros(1, HISTORY_LEN, POLICY_DIM, device=device),
                },
                batch_size=[1],
            )

    runner = OnPolicyRunnerAMP(MockEnv(), tc, log_dir=None, device=device)
    runner.load(ckpt_path, load_optimizer=False)
    return runner


def main() -> None:
    parser = argparse.ArgumentParser(description="go2_imitation_tracking RMA+AMP 정책을 단일 deployable jit로 export.")
    parser.add_argument("--run_dir", default="logs/rsl_rl/go2_imitation_tracking/2026-07-24_13-03-00")
    parser.add_argument("--checkpoint", default="model_19.pt")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="출력 경로 (기본: <run_dir>/exported/deployable_policy.pt)")
    args = parser.parse_args()

    runner = _load_runner(args.run_dir, args.checkpoint, args.device)
    policy = runner.alg.policy.eval()
    estimator = runner.alg.estimator.eval()
    assert estimator is not None, "estimator가 빌드되지 않았다 — agent.yaml의 estimator dict를 확인하라."
    deploy = DeployablePolicy(policy, estimator).to(args.device).eval()

    # 정합성 대조: wrapper vs act_inference (priv_explicit=estimator)
    torch.manual_seed(0)
    proprio = torch.randn(4, POLICY_DIM, device=args.device)
    history = torch.randn(4, HISTORY_LEN, POLICY_DIM, device=args.device)
    with torch.inference_mode():
        ref = policy.act_inference({"policy": proprio, "priv_explicit": estimator(proprio), "history": history})
        out = deploy(proprio, history)
    diff = (ref - out).abs().max().item()
    assert out.shape == (4, NUM_JOINTS) and diff < 1e-5, f"wrapper 불일치 shape={out.shape} diff={diff}"

    out_path = args.out or os.path.join(args.run_dir, "exported", "deployable_policy.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    scripted = torch.jit.script(deploy)
    scripted.save(out_path)

    # 저장본 재로드 대조
    reloaded = torch.jit.load(out_path, map_location=args.device).eval()
    with torch.inference_mode():
        diff2 = (ref - reloaded(proprio, history)).abs().max().item()
    assert diff2 < 1e-5, f"jit 재로드 불일치 {diff2}"

    print(f"[export] OK — {out_path}")
    print(f"[export] 입력: proprio(B,{POLICY_DIM}), history(B,{HISTORY_LEN},{POLICY_DIM}) → action(B,{NUM_JOINTS})")
    print(f"[export] wrapper diff={diff:.1e}, jit reload diff={diff2:.1e} (both < 1e-5)")


if __name__ == "__main__":
    main()

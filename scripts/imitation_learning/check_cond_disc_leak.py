# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""조건부 AMP discriminator 의 "조건 누설" 게이트.

학습된 disc 를 체크포인트에서 읽어 expert 배치에 대해 다음 네 가지 조건 라벨로 D(expert) 를 잰다.

.. code-block:: text

    clip_mean : 클립 평균 속도 라벨 (학습 때 쓴 것)              → 높으면 정상
    shuffled  : 라벨을 배치 안에서 뒤섞음 (kinematics-조건 불일치)  → clip_mean 보다 낮아야 조건을 "쓰는" 것
    uniform   : 라벨을 정책 명령처럼 U[0,1] 로 교체              → clip_mean 대비 크게 떨어지면 조건 **값 분포**로
                                                                 판별한 것 = 누설
    matched   : command_matched 샘플링 (조건 먼저 뽑고 근접 클립)  → 새 방식에서 D 가 얼마나 보는지
    dropped   : cond=0, valid=0 (무조건부 경로)

.. code-block:: bash

    CUDA_VISIBLE_DEVICES=3 python scripts/imitation_learning/check_cond_disc_leak.py --headless \\
        --task Leg-Imitation-Tracking-RMA-v0 --checkpoint logs/.../model_10000.pt \\
        --set motion_file=... motion_weight_mode=command_uniform lin_vel_x_max=3.2 amp_cond_mode=speed
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--n", type=int, default=4096, help="expert 샘플 수")
parser.add_argument("--set", nargs="*", default=[], help="env cfg 오버라이드 key=value (학습 run 과 같게)")
parser.add_argument("--disc_hidden", type=int, nargs="*", default=[1024, 512])
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.modules import AMPDiscriminator  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _cast(v: str):
    for f in (int, float):
        try:
            return f(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def main() -> None:
    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
    for kv in args.set:
        k, v = kv.split("=", 1)
        assert hasattr(env_cfg, k), f"env cfg 에 {k} 없음"
        setattr(env_cfg, k, _cast(v))
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    assert env.amp_cond_dim > 0, "amp_cond_mode 가 꺼져 있다"
    kin = env.amp_observation_size - env.amp_cond_dim

    ck = torch.load(args.checkpoint, map_location="cuda:0", weights_only=False)
    disc = AMPDiscriminator(
        input_dim=env.amp_observation_size,
        hidden_dims=args.disc_hidden,
        device="cuda:0",
        disc_reward_type="bce",
        norm_clip=10.0,
        cond_dim=env.amp_cond_dim,
    )
    disc.load_state_dict(ck["discriminator_state_dict"])
    disc.eval()

    def d_mean(x: torch.Tensor) -> float:
        with torch.no_grad():
            return torch.sigmoid(disc.get_logits(x)).mean().item()

    env.cfg.amp_cond_expert_sampling = "clip_mean"
    e_clip = env.get_amp_observations(args.n)
    env.cfg.amp_cond_expert_sampling = "command_matched"
    e_match = env.get_amp_observations(args.n)

    e_shuf = e_clip.clone()
    e_shuf[:, kin:] = e_clip[torch.randperm(args.n, device=e_clip.device), kin:]
    e_unif = e_clip.clone()
    e_unif[:, kin : kin + env._amp_cond_values_dim] = torch.rand(args.n, env._amp_cond_values_dim, device=e_clip.device)
    e_drop = disc.drop_condition(e_clip)

    rows = [
        ("clip_mean (학습 라벨)", d_mean(e_clip)),
        ("shuffled  (라벨 뒤섞음)", d_mean(e_shuf)),
        ("uniform   (라벨 U[0,1])", d_mean(e_unif)),
        ("matched   (command_matched)", d_mean(e_match)),
        ("dropped   (무조건부)", d_mean(e_drop)),
    ]
    print(f"\n== D(expert) by condition label  (checkpoint: {args.checkpoint}, n={args.n})")
    for name, v in rows:
        print(f"  {name:30s} {v:.3f}")
    vals = e_clip[:, kin].unique()
    print(f"  clip_mean 라벨 고유값 수: {vals.numel()}  (예: {[round(x, 3) for x in vals[:6].tolist()]})")
    env.close()


if __name__ == "__main__":
    main()
    app.close()

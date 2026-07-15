# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""자유 스폰 GO2를 표준 instanceable vs non-instanceable 에셋으로 각각 렌더해 프레임 비교.

live sim_runner_go2는 play.py와 달리 non-instanceable swap을 하지 않는다. 표준 go2.usd가
Isaac Sim 6.0 렌더 조각 버그(#2925)로 num_envs=1에서 깨져 보이는지 확인한다.
"""

"""Launch first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--noninst", action="store_true", help="non-instanceable 에셋으로 swap해서 렌더.")
parser.add_argument("--out", type=str, default="logs/r2s_chirp_preview/render_check.png")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

import gymnasium as gym
import imageio
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR  # isort: skip

TASK = "Isaac-R2S-Go2-v0"


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.fix_base = False
    # 로봇을 가까이 보는 카메라 (기본 뷰어는 너무 멀다).
    env_cfg.viewer.eye = (1.6, 1.6, 0.9)
    env_cfg.viewer.lookat = (0.0, 0.0, 0.15)
    if args_cli.noninst:
        p = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Go2_noninstanceable", "go2.usd")
        if os.path.isfile(p):
            env_cfg.robot.spawn.usd_path = p
            print(f"[render_check] non-instanceable: {p}", flush=True)

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    frame = None
    for _ in range(150):  # 정착까지 스텝
        with torch.inference_mode():
            env.step(action)
        frame = env.render()
    out = args_cli.out if os.path.isabs(args_cli.out) else os.path.join(os.getcwd(), args_cli.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.imwrite(out, frame)
    print(f"[render_check] saved {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

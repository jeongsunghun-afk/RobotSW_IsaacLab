# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""FIX_BASE=0 자유 스폰 GO2가 명령 없이 어떻게 정착하는지 진단 (뒤집힘 여부).

live 환경(Isaac-R2S-Go2-v0)을 fix_base=False로 띄우고, /lowcmd 없이(=gui 미실행 상황) 그냥
prone setpoint만 홀드하며 스텝을 돌린 뒤 base 높이/자세(quat, 중력 투영)를 로깅한다.
중력 투영 z(projected_gravity_b[2])가 -1이면 정상(등이 위), +1이면 뒤집힘(배가 위).
"""

"""Launch first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=400, help="진단 스텝 수 (50Hz 기준 400=8s).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.math import quat_apply_inverse

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK = "Isaac-R2S-Go2-v0"


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    env_cfg.fix_base = False  # 자유 스폰 (FIX_BASE=0 r2s_sim 과 동일)
    env = gym.make(TASK, cfg=env_cfg)
    env.reset()
    robot = env.unwrapped.scene["robot"]
    action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    grav_w = torch.tensor([[0.0, 0.0, -1.0]], device=env.unwrapped.device)

    print(f"{'step':>5} {'base_z':>8} {'quat(wxyz)':>34} {'grav_b_z':>9}  판정", flush=True)
    for i in range(args_cli.steps):
        with torch.inference_mode():
            env.step(action)  # live 모드: setpoint(=prone default) 홀드, 명령 없음
        if i % 40 == 0 or i == args_cli.steps - 1:
            z = float(robot.data.root_pos_w[0, 2].item())
            quat_xyzw = robot.data.root_quat_w[0]
            gb = quat_apply_inverse(quat_xyzw.unsqueeze(0), grav_w).squeeze(0)
            gbz = float(gb[2].item())
            w, x, y, zc = (float(quat_xyzw[3]), float(quat_xyzw[0]), float(quat_xyzw[1]), float(quat_xyzw[2]))
            verdict = "정상(등 위)" if gbz < -0.5 else ("뒤집힘(배 위)" if gbz > 0.5 else "옆으로/기울어짐")
            print(f"{i:5d} {z:8.3f}  ({w:+.2f},{x:+.2f},{y:+.2f},{zc:+.2f})   {gbz:+9.3f}  {verdict}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

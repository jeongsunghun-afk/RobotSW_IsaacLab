# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-HindLeg Isaac Sim 런처 겸 UDP 통신 루프.

gui_controller.py(시스템/conda Python 무관, 순수 UDP)와 UDP로 명령/상태를 주고받는
Isaac Sim(conda Python 3.12) 쪽 프로세스. rclpy는 사용하지 않는다.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_hind_leg/CONTRACT.md §1, §4, §5

실행:
    ./isaaclab.sh -p scripts/real2sim/sim_runner_hindleg.py [--headless]
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# r2s_udp는 순수 stdlib이라 AppLauncher 기동 전에 임포트해도 안전하다.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_hind_leg"))
from r2s_udp import CMD_PORT, STATE_PORT, pack_state, unpack_cmd  # isort: skip

parser = argparse.ArgumentParser(description="R2S-HindLeg sim runner (UDP bridge to gui_controller.py).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--cmd_port", type=int, default=CMD_PORT, help="UDP port to receive motor commands on.")
parser.add_argument("--state_port", type=int, default=STATE_PORT, help="UDP port to send sim state to.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import socket

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_NAME = "Isaac-R2S-HindLeg-v0"
HOST = "127.0.0.1"


def main() -> None:
    """Run the UDP <-> Isaac Sim hind-leg bridge loop (latest-wins, non-blocking recv)."""
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env.reset()

    # UDP sockets: recv is non-blocking so the sim loop never stalls (CONTRACT §4 latest-wins)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, args_cli.cmd_port))
    recv_sock.setblocking(False)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    gui_addr: tuple[str, int] | None = None  # 마지막 cmd 발신자에게 state 회신

    print(
        f"[sim_runner_hindleg] UDP listening on {HOST}:{args_cli.cmd_port}, "
        f"sending state to sender:{args_cli.state_port}",
        flush=True,
    )

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    seq = 0
    try:
        while simulation_app.is_running():
            # drain the recv queue and keep only the latest command
            last = None
            while True:
                try:
                    data, src = recv_sock.recvfrom(4096)
                except BlockingIOError:
                    break
                cmd = unpack_cmd(data)
                if cmd is not None:
                    last = cmd
                    gui_addr = (src[0], args_cli.state_port)
            if last is not None:
                env.unwrapped.set_setpoint(last["q"], last["dq"], last["kp"], last["kd"], last["tau"])

            with torch.inference_mode():
                env.step(zero_action)

            # publish the resulting state to whoever last sent a command
            if gui_addr is not None:
                st = env.unwrapped.get_lowstate()
                sim_time = float(env.unwrapped.episode_length_buf[0].item()) * env.unwrapped.step_dt
                packet = pack_state(seq, sim_time, st["q"], st["dq"], st["ddq"], st["tau_est"])
                send_sock.sendto(packet, gui_addr)
            seq += 1
    finally:
        recv_sock.close()
        send_sock.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

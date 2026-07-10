# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 Isaac Sim 런처 겸 UDP 통신 루프.

sim_bridge.py(ROS2, 시스템 Python 3.10)와 UDP로 명령/상태를 주고받는
Isaac Sim(conda Python 3.12) 쪽 프로세스. rclpy는 사용하지 않는다.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_go2/CONTRACT.md §1, §4, §5

실행:
    ./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py [--fix_base] [--headless]
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# r2s_udp는 순수 stdlib이라 AppLauncher 기동 전에 임포트해도 안전하다.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_go2"))
from r2s_udp import CMD_PORT, STATE_PORT, pack_state, unpack_cmd  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="R2S-GO2 sim runner (UDP bridge to sim_bridge.py).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--fix_base", action="store_true", default=False, help="Fix the robot base in mid-air instead of free spawn."
)
parser.add_argument("--cmd_port", type=int, default=CMD_PORT, help="UDP port to receive motor commands on.")
parser.add_argument("--state_port", type=int, default=STATE_PORT, help="UDP port to send sim state to.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import socket

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_NAME = "Isaac-R2S-Go2-v0"
HOST = "127.0.0.1"


def main() -> None:
    """Run the UDP <-> Isaac Sim GO2 bridge loop (latest-wins, non-blocking recv)."""
    # parse and override the environment configuration (CONTRACT §5: fix_base toggle)
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.fix_base = args_cli.fix_base

    # create environment
    env = gym.make(TASK_NAME, cfg=env_cfg)
    env.reset()

    # UDP sockets: recv is non-blocking so the sim loop never stalls (CONTRACT §4 latest-wins)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, args_cli.cmd_port))
    recv_sock.setblocking(False)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(
        f"[sim_runner_go2] UDP listening on {HOST}:{args_cli.cmd_port}, sending state to {HOST}:{args_cli.state_port}",
        flush=True,
    )

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # Optional base-height debug log (R2S_LOG_BASEZ=1) — used to verify prone spawn +
    # stand-up (base z rises). Purely additive; off by default.
    log_basez = os.environ.get("R2S_LOG_BASEZ") == "1"
    seq = 0
    try:
        while simulation_app.is_running():
            # drain the recv queue and keep only the latest command
            last = None
            while True:
                try:
                    data, _ = recv_sock.recvfrom(4096)
                except BlockingIOError:
                    break
                cmd = unpack_cmd(data)
                if cmd is not None:
                    last = cmd
            if last is not None:
                env.unwrapped.set_setpoint(last["q"], last["dq"], last["kp"], last["kd"], last["tau"])

            # step the sim (setpoint is applied from the internal buffer, action content is a no-op)
            with torch.inference_mode():
                env.step(zero_action)

            # publish the resulting state
            st = env.unwrapped.get_lowstate()
            sim_time = float(env.unwrapped.episode_length_buf[0].item()) * env.unwrapped.step_dt
            packet = pack_state(seq, sim_time, st["q"], st["dq"], st["ddq"], st["tau_est"], st["imu"])
            send_sock.sendto(packet, (HOST, args_cli.state_port))

            if log_basez and seq % 50 == 0:
                base_z = float(env.unwrapped.robot.data.root_pos_w[0, 2].item())
                q = st["q"]
                thigh = (q[1] + q[4] + q[7] + q[10]) / 4.0  # mean thigh angle (1.36 prone → 0.67 stand)
                print(f"[basez] seq={seq} base_z={base_z:.4f} mean_thigh_q={thigh:.3f}", flush=True)
            seq += 1
    finally:
        recv_sock.close()
        send_sock.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

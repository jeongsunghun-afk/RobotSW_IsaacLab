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
from r2s_udp import (  # isort: skip
    CAMERA_FOLLOW,
    CAMERA_FREE,
    CMD_PORT,
    CTRL_PORT,
    PLANT_NAMES,
    PLANT_NOMINAL,
    PLANT_SET2,
    PLANT_SET3,
    STATE_PORT,
    pack_state,
    unpack_cmd,
    unpack_ctrl,
    unpack_marker,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="R2S-GO2 sim runner (UDP bridge to sim_bridge.py).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--fix_base", action="store_true", default=False, help="Fix the robot base in mid-air instead of free spawn."
)
parser.add_argument("--cmd_port", type=int, default=CMD_PORT, help="UDP port to receive motor commands on.")
parser.add_argument("--state_port", type=int, default=STATE_PORT, help="UDP port to send sim state to.")
parser.add_argument(
    "--ctrl_port", type=int, default=CTRL_PORT, help="UDP port to receive sim control (camera/plant) on."
)
parser.add_argument(
    "--no_lockstep",
    action="store_false",
    dest="lockstep",
    default=True,
    help="Free-run the sim instead of stepping once per received /lowcmd packet. Free-running "
    "lets sim time and policy time drift apart (measured 62.5 Hz sim vs 50 Hz policy) and makes "
    "the feedback delay vary from tick to tick, while training only saw one control step of "
    "delay. Absolute-target policies (locomotion, recovery) overwrite that error every tick, but "
    "pedipulation carries its manipulation target over and accumulates it: foot tracking went "
    "264 mm free-running vs 10-14 mm locked "
    "(reports/rsl_rl/go2_pedipulation/_comparisons/r2s_deploy_tracking_gap/). Use this only to "
    "reproduce data collected before lockstep became the default.",
)
parser.add_argument(
    "--camera",
    choices=("follow", "free"),
    default="follow",
    help="Initial viewport camera mode. 'follow' tracks the robot base; 'free' leaves the viewport to you.",
)
parser.add_argument(
    "--plant",
    choices=("default", "set2", "set3", "nominal", "pace"),
    default=None,
    help="Initial joint-property preset. 'default' (alias 'nominal') = UNITREE_GO2_CFG values "
    "(armature 0.01, no friction); 'set2' (alias 'pace') = the live PACE_* identification; "
    "'set3' = the 2026-08-04 re-capture (not adopted for training — comparison only). "
    "Defaults to the env cfg (use_pace_params).",
)
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
# 마커 로그용 다리 이름. 인덱스는 GUI 쪽 `pedipulation_runtime.LEG_NAMES` 와 같은 순서다.
MARKER_LEG_NAMES = ("FL", "FR", "RL", "RR")

# --lockstep 에서 명령 하나를 기다리는 최대 시간 [s]. 넘으면 그냥 스텝한다 — publisher 가
# 아직 안 떴거나 죽었을 때 sim 이 멈춘 채로 보이지 않게 하기 위한 것뿐이다.
LOCKSTEP_TIMEOUT_S = 0.1


def main() -> None:
    """Run the UDP <-> Isaac Sim GO2 bridge loop (latest-wins, non-blocking recv)."""
    # parse and override the environment configuration (CONTRACT §5: fix_base toggle)
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.fix_base = args_cli.fix_base
    plant_override = {
        "default": PLANT_NOMINAL,
        "nominal": PLANT_NOMINAL,
        "set2": PLANT_SET2,
        "pace": PLANT_SET2,
        "set3": PLANT_SET3,
    }
    if args_cli.plant is not None:
        # cfg 는 bool 하나뿐이라 set3 를 담지 못한다. cfg 로는 default/set2 만 정하고,
        # set3 는 env 생성 뒤 set_joint_plant() 로 덮어쓴다(아래).
        env_cfg.use_pace_params = plant_override[args_cli.plant] != PLANT_NOMINAL

    # create environment
    env = gym.make(TASK_NAME, cfg=env_cfg)
    env.reset()

    # sim 제어 상태 (GUI 가 UDP 로 바꾼다). reset 이 cfg 프리셋을 이미 적용했으므로 여기선 초기값만 기록.
    camera_mode = CAMERA_FOLLOW if args_cli.camera == "follow" else CAMERA_FREE
    plant_mode = PLANT_SET2 if env_cfg.use_pace_params else PLANT_NOMINAL
    if args_cli.plant is not None:
        plant_mode = plant_override[args_cli.plant]
        env.unwrapped.set_joint_plant(plant_mode)
    env.unwrapped.set_camera_follow(camera_mode == CAMERA_FOLLOW)

    # UDP sockets: recv is non-blocking so the sim loop never stalls (CONTRACT §4 latest-wins)
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, args_cli.cmd_port))
    recv_sock.setblocking(False)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # sim 제어 채널 (GUI -> 여기). 명령 소켓과 분리 — 저빈도 one-shot 이라 50Hz 명령 스트림에
    # 필드를 얹지 않는다. GUI 가 안 떠 있어도 아무 일도 일어나지 않는다(수신만 함).
    ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ctrl_sock.bind((HOST, args_cli.ctrl_port))
    ctrl_sock.setblocking(False)

    print(
        f"[sim_runner_go2] UDP listening on {HOST}:{args_cli.cmd_port}, sending state to {HOST}:{args_cli.state_port}",
        flush=True,
    )
    print(
        f"[sim_runner_go2] ctrl on {HOST}:{args_cli.ctrl_port} — camera={args_cli.camera} "
        f"plant={PLANT_NAMES[plant_mode]} "
        f"stepping={'lockstep' if args_cli.lockstep else 'free-run (--no_lockstep)'}",
        flush=True,
    )

    # pedipulation 마커 표시 상태 — 전환 시에만 로그를 찍기 위해 들고 간다.
    marker_on = False

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    # Optional base-height debug log (R2S_LOG_BASEZ=1) — used to verify prone spawn +
    # stand-up (base z rises). Purely additive; off by default.
    log_basez = os.environ.get("R2S_LOG_BASEZ") == "1"
    seq = 0
    try:
        while simulation_app.is_running():
            # drain the recv queue and keep only the latest command
            last = None
            if args_cli.lockstep:
                # 기본값. 정책 tick 당 정확히 한 번만 스텝한다 — 자유 구동이면 sim 시계와
                # 정책 시계가 따로 흘러 지연이 tick 마다 달라지고, 목표를 이어받는 정책
                # (pedipulation)은 그 오차를 덮어쓰지 못하고 쌓는다.
                # 타임아웃이면 그냥 스텝해 sim 이 얼어붙지 않게 한다 (publisher 가 아직
                # 안 떴거나 죽은 경우).
                recv_sock.settimeout(LOCKSTEP_TIMEOUT_S)
                try:
                    cmd = unpack_cmd(recv_sock.recvfrom(4096)[0])
                    if cmd is not None:
                        last = cmd
                except OSError:  # TimeoutError 포함
                    pass
                recv_sock.setblocking(False)
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

            # sim 제어 패킷도 latest-wins 로 비운다. 값이 실제로 바뀔 때만 적용한다 —
            # 물성 write 는 PhysX 왕복이고 카메라 전환은 뷰를 스냅시키므로 매 tick 반복하면 안 된다.
            last_ctrl = None
            last_marker = None
            while True:
                try:
                    data, _ = ctrl_sock.recvfrom(256)
                except BlockingIOError:
                    break
                parsed = unpack_ctrl(data)
                if parsed is not None:
                    last_ctrl = parsed
                    continue
                # 같은 포트로 들어오는 pedipulation 마커 패킷 (magic·크기로 갈린다).
                parsed = unpack_marker(data)
                if parsed is not None:
                    last_marker = parsed
            if last_ctrl is not None:
                if last_ctrl["camera"] != camera_mode:
                    camera_mode = last_ctrl["camera"]
                    env.unwrapped.set_camera_follow(camera_mode == CAMERA_FOLLOW)
                    print(
                        f"[sim_runner_go2] camera -> {'follow' if camera_mode == CAMERA_FOLLOW else 'free'}", flush=True
                    )
                if last_ctrl["plant"] != plant_mode:
                    plant_mode = last_ctrl["plant"]
                    env.unwrapped.set_joint_plant(plant_mode)
                    print(f"[sim_runner_go2] plant -> {PLANT_NAMES.get(plant_mode, plant_mode)}", flush=True)

            # 마커는 camera/plant 와 달리 **매번 적용한다** — 목표가 그대로여도 발과 base 는
            # 매 tick 움직이므로 갱신을 건너뛰면 구가 뒤처져 보인다. 값 비교로 거를 게 없다.
            if last_marker is not None:
                env.unwrapped.set_pedi_markers(last_marker["leg"], last_marker["target_b"])
                # 로그는 켜짐/꺼짐이 바뀔 때만 — 50Hz 로 들어오므로 매번 찍으면 안 된다.
                if (last_marker["leg"] >= 0) != marker_on:
                    marker_on = last_marker["leg"] >= 0
                    leg_label = MARKER_LEG_NAMES[last_marker["leg"]] if marker_on else "off"
                    print(f"[sim_runner_go2] pedi marker -> {leg_label}", flush=True)

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
        ctrl_sock.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

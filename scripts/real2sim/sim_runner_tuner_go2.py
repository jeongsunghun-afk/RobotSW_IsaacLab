# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 tuner 모드 sim 런너 (Phase 2.5) — 물성 파라미터 실시간 튜닝.

`Isaac-R2S-Go2-Sysid-v0`(PaceDCMotor + slew 우회 + fix_base)를 **num_envs=1**로 구동하면서:

    1. chirp를 계속 재생하거나(``--replay`` 없을 때) 녹화된 ``.pt``의 명령을 재생한다.
    2. tuner_gui가 UDP(:9875)로 보낸 전역 물성(armature/viscous/coulomb/kp/kd/delay)을
       매 변경 시 ``write_joint_*_to_sim``으로 sim에 즉시 반영한다(cma_es.update_simulator와 동일 API).
    3. q_sim / q_cmd / q_real(replay 시)을 UDP(:9876)로 tuner_monitor에 흘려보낸다.

CMA-ES 배치 적합과 달리 이건 **사람이 눈으로 bounds/초기분포를 잡는 용도**다 — 전역 스칼라만 다룬다.
rclpy는 쓰지 않는다(순수 UDP). 계약: source/isaaclab_tasks/.../r2s_go2/CONTRACT.md §4.

실행::

    ./isaaclab.sh -p scripts/real2sim/sim_runner_tuner_go2.py [--replay data/go2_real/chirp_kp25.pt] [--headless]
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# r2s_udp / chirp 는 순수 stdlib이라 AppLauncher 기동 전에 임포트해도 안전하다.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_go2"))
import chirp  # isort: skip
from r2s_udp import (  # isort: skip
    TUNER_PARAM_PORT,
    TUNER_TELEM_PORT,
    pack_tuner_telem,
    unpack_tuner_params,
)

parser = argparse.ArgumentParser(description="R2S-GO2 tuner sim runner (realtime physical-parameter tuning).")
parser.add_argument(
    "--replay",
    type=str,
    default=None,
    help="녹화 .pt 경로(time/des_dof_pos/dof_pos). 없으면 chirp.py의 chirp를 생성해 재생.",
)
parser.add_argument("--param_port", type=int, default=TUNER_PARAM_PORT, help="물성 파라미터 수신 UDP 포트.")
parser.add_argument("--telem_port", type=int, default=TUNER_TELEM_PORT, help="텔레메트리 송신 UDP 포트.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import socket

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.r2s_go2.r2s_go2_env_cfg import JOINT_ORDER
from isaaclab_tasks.direct.r2s_go2.r2s_go2_sysid_cfg import SYSID_KD, SYSID_KP
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_NAME = "Isaac-R2S-Go2-Sysid-v0"
HOST = "127.0.0.1"


def load_command_source(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, bool, float, float]:
    """재생할 명령/실기 궤적을 준비한다.

    Returns:
        ``(q_cmd, q_real, has_real, kp0, kd0)`` — ``q_cmd``/``q_real``은 (T, 12) 텐서(JOINT_ORDER 순서),
        ``has_real``은 replay 데이터 유무, ``kp0``/``kd0``은 초기 게인(replay 메타 또는 sysid 기본값).
    """
    if args_cli.replay is None:
        _, targets = chirp.build_chirp(rate_hz=chirp.DEFAULT_RATE_HZ)
        q_cmd = torch.tensor(targets, dtype=torch.float32, device=device)
        print(f"[tuner] chirp 재생: {q_cmd.shape[0]} steps @ {chirp.DEFAULT_RATE_HZ:.0f} Hz", flush=True)
        return q_cmd, q_cmd, False, SYSID_KP, SYSID_KD

    data = torch.load(args_cli.replay)
    q_cmd = data["des_dof_pos"].to(device=device, dtype=torch.float32)
    q_real = data["dof_pos"].to(device=device, dtype=torch.float32)
    meta = data.get("meta", {})
    kp0 = float(meta.get("kp", SYSID_KP))
    kd0 = float(meta.get("kd", SYSID_KD))
    print(
        f"[tuner] replay 재생: {os.path.basename(args_cli.replay)} {q_cmd.shape[0]} steps, kp0={kp0} kd0={kd0}",
        flush=True,
    )
    return q_cmd, q_real, True, kp0, kd0


def apply_params(robot, actuator, sim_joint_ids: torch.Tensor, env_ids: torch.Tensor, params: dict) -> None:
    """전역 물성 스칼라를 sim에 반영한다 (cma_es.update_simulator와 동일 API, num_envs=1).

    Args:
        robot: articulation 핸들.
        actuator: PaceDCMotor 인스턴스(delay 반영용).
        sim_joint_ids: JOINT_ORDER를 articulation 인덱스로 매핑한 int32 텐서.
        env_ids: int32 env 인덱스(num_envs=1이므로 [0]).
        params: ``armature/viscous/coulomb/kp/kd/delay`` 키를 가진 dict.
    """
    n_joints = sim_joint_ids.shape[0]
    device = sim_joint_ids.device

    def full(value: float) -> torch.Tensor:
        return torch.full((1, n_joints), float(value), dtype=torch.float32, device=device)

    armature = full(params["armature"])
    viscous = full(params["viscous"])
    coulomb = full(params["coulomb"])

    robot.write_joint_armature_to_sim_index(armature=armature, joint_ids=sim_joint_ids, env_ids=env_ids)
    robot.data.default_joint_armature.torch[:, sim_joint_ids] = armature
    # IsaacLab 6.0: static/dynamic/viscous 마찰을 한 번에 쓴다. dynamic==static이 Coulomb 마찰을 모델링.
    robot.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=coulomb,
        joint_dynamic_friction_coeff=coulomb,
        joint_viscous_friction_coeff=viscous,
        joint_ids=sim_joint_ids,
        env_ids=env_ids,
    )
    robot.data.default_joint_friction_coeff.torch[:, sim_joint_ids] = coulomb
    robot.data.default_joint_viscous_friction_coeff.torch[:, sim_joint_ids] = viscous
    # kp/kd — PACE는 식별하지 않지만 tuner에서는 알려진 게인을 눈으로 맞추기 위해 write 한다.
    robot.write_joint_stiffness_to_sim(full(params["kp"]), joint_ids=sim_joint_ids)
    robot.write_joint_damping_to_sim(full(params["kd"]), joint_ids=sim_joint_ids)
    # delay — PaceDCMotor의 DelayBuffer time-lag(정수 sim step). reset으로 버퍼 히스토리 클리어.
    delay_int = int(round(params["delay"]))
    actuator.update_time_lags(torch.full((1,), delay_int, dtype=torch.int, device=device))
    actuator.reset(env_ids)


def main() -> None:
    """tuner 루프: chirp/replay 재생 + 실시간 파라미터 반영 + 텔레메트리 송신."""
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
    env = gym.make(TASK_NAME, cfg=env_cfg)
    env.reset()

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    actuator = robot.actuators["base_legs"]

    # JOINT_ORDER(CONTRACT §2) -> articulation 관절 인덱스. warp 커널이 int32를 요구.
    sim_joint_ids = torch.tensor(
        [robot.joint_names.index(name) for name in JOINT_ORDER],
        dtype=torch.int32,
        device=device,
    )
    env_ids = torch.zeros(1, dtype=torch.int32, device=device)

    q_cmd, q_real, has_real, kp0, kd0 = load_command_source(device)
    num_steps = q_cmd.shape[0]

    # 초기 파라미터 — sysid cfg 기본 물성 + replay 게인. gui가 값 보내기 전까지 이걸로 구동.
    current_params = {
        "armature": 0.01,
        "viscous": 0.0,
        "coulomb": 0.0,
        "kp": kp0,
        "kd": kd0,
        "delay": 0.0,
    }
    with torch.inference_mode():
        apply_params(robot, actuator, sim_joint_ids, env_ids, current_params)

    # UDP: recv(param)은 non-blocking(루프 정지 금지), send(telem)은 monitor로.
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, args_cli.param_port))
    recv_sock.setblocking(False)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(
        f"[tuner] param UDP :{args_cli.param_port} 수신, telem -> :{args_cli.telem_port}  "
        f"(has_real={has_real}, {num_steps} steps loop)",
        flush=True,
    )

    action = torch.zeros(env.action_space.shape, device=device)
    counter = 0
    seq = 0
    try:
        # 루프 전체를 inference_mode로 감싼다 (fit.py와 동일). apply_params가 write_joint_*_to_sim과
        # actuator.reset(DelayBuffer in-place)을 부르는데, inference_mode 밖이면 inference tensor
        # in-place 갱신이 RuntimeError로 막힌다.
        while simulation_app.is_running():
            with torch.inference_mode():
                # 파라미터 큐 비우고 마지막 값만 반영 (latest-wins). 값이 실제로 바뀌었을 때만 write.
                latest = None
                while True:
                    try:
                        data, _ = recv_sock.recvfrom(4096)
                    except BlockingIOError:
                        break
                    parsed = unpack_tuner_params(data)
                    if parsed is not None:
                        latest = parsed
                if latest is not None:
                    new_params = {k: latest[k] for k in current_params}
                    if new_params != current_params:
                        current_params = new_params
                        apply_params(robot, actuator, sim_joint_ids, env_ids, current_params)

                # 현재 명령 프레임을 articulation 인덱스에 채워 스텝 (sysid 경로: slew 우회 직접 타겟).
                cmd_row = q_cmd[counter]
                action[:, sim_joint_ids] = cmd_row.unsqueeze(0)
                env.step(action)

                # 텔레메트리 송신: q_sim(JOINT_ORDER 순서) / q_cmd / q_real.
                q_sim = robot.data.joint_pos[0, sim_joint_ids]
                sim_time = float(env.unwrapped.episode_length_buf[0].item()) * env.unwrapped.step_dt
                packet = pack_tuner_telem(
                    seq,
                    sim_time,
                    has_real,
                    q_sim.cpu().tolist(),
                    cmd_row.cpu().tolist(),
                    q_real[counter].cpu().tolist(),
                )
                send_sock.sendto(packet, (HOST, args_cli.telem_port))

                seq += 1
                counter += 1
                if counter >= num_steps:
                    counter = 0  # 궤적 끝 -> 처음부터 재생(연속 튜닝).
    finally:
        recv_sock.close()
        send_sock.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

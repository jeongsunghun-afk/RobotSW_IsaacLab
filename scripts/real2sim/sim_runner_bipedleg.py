# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg Isaac Sim 런처 겸 UDP 통신 루프.

gui_controller.py(시스템/conda Python 무관, 순수 UDP)와 UDP로 명령/상태를 주고받는
Isaac Sim(conda Python 3.12) 쪽 프로세스. rclpy는 사용하지 않는다.

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_biped_leg/CONTRACT.md §1, §4, §5

실행 (라이브스트림):
    LIVESTREAM=2 CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p scripts/real2sim/sim_runner_bipedleg.py \
        --num_envs 1 --fix_base --viz kit

``--viz kit``은 6.0 라이브스트림에 필수다. ``--viz``를 생략하면 헤드리스로 돌아간다
(``--headless``는 deprecated). 8-DOF 2족이라 ``--fix_base`` 없이는 GUI로 관절을 스텝하는
순간 넘어지므로, 관절 추종 확인 용도로는 ``--fix_base``를 켠다.

position 모드 루프는 step_dt(20ms) wall-clock 페이싱으로 실시간 1.0배를 유지한다. 렌더가
병목이면(kit 라이브스트림 ~25ms/frame) 페이싱만으론 부족하므로 ``--render_decimation``(기본 2)
으로 env step N회당 1회만 렌더해 한 바퀴를 20ms 아래로 끌어내린다 — 실측: 렌더 매 step이면
루프가 33Hz로 떨어지고 sim이 0.66배속 슬로모션이 돼 monitor plot이 계단(40~100ms hold)+
겉보기 지연 180ms로 보인다.
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# r2s_udp는 순수 stdlib이라 AppLauncher 기동 전에 임포트해도 안전하다.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_biped_leg"))
from r2s_udp import (  # isort: skip
    CMD_PORT,
    POLICY_ACT_PORT,
    POLICY_STATE_PORT,
    STATE_PORT,
    pack_ieff,
    pack_policy_state,
    pack_state,
    unpack_cmd,
    unpack_plant,
    unpack_policy_act,
)

parser = argparse.ArgumentParser(description="R2S-BipedLeg sim runner (UDP bridge to gui_controller.py).")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--fix_base", action="store_true", default=False, help="Fix the robot base in mid-air instead of free spawn."
)
parser.add_argument(
    "--policy_mode",
    action="store_true",
    default=False,
    help="Policy mode: receive articulation-order target from policy_runner_bipedleg.py (POLICY_ACT_PORT) "
    "and publish rich state (q,dq,gravity) on POLICY_STATE_PORT. Free base (fix_base forced False).",
)
parser.add_argument("--cmd_port", type=int, default=CMD_PORT, help="UDP port to receive motor commands on.")
parser.add_argument("--state_port", type=int, default=STATE_PORT, help="UDP port to send sim state to.")
parser.add_argument(
    "--render_decimation",
    type=int,
    default=2,
    help="Render once every N env steps (kit viz). 1 = every step (legacy, drops the loop below "
    "real time when render is the bottleneck). Headless runs are unaffected.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import socket
import time

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_NAME = "Isaac-R2S-BipedLeg-v0"
HOST = "127.0.0.1"


def main() -> None:
    """Run the UDP <-> Isaac Sim biped-leg bridge loop (latest-wins, non-blocking recv)."""
    # parse and override the environment configuration (CONTRACT §5: fix_base toggle)
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.policy_mode:
        # policy 모드: 정책이 자유베이스에서 균형을 잡으므로 fix_base는 강제 False.
        env_cfg.policy_mode = True
        env_cfg.fix_base = False
    else:
        env_cfg.fix_base = args_cli.fix_base

    # 렌더 데시메이션 — render_interval은 physics step 단위라 env step 단위 인자에 decimation을 곱한다.
    # 헤드리스(is_rendering=False)에선 무효과. DirectRLEnv가 render_interval < decimation이면 경고하므로
    # 1 미만은 받지 않는다.
    if args_cli.render_decimation < 1:
        raise ValueError(f"--render_decimation must be >= 1, got {args_cli.render_decimation}")
    env_cfg.sim.render_interval = env_cfg.decimation * args_cli.render_decimation

    env = gym.make(TASK_NAME, cfg=env_cfg)
    env.reset()

    if args_cli.policy_mode:
        _run_policy_loop(env)
    else:
        _run_position_loop(env)


def _run_position_loop(env) -> None:
    """Position-control 브릿지 (gui_controller.py <-> sim, CMD/STATE 포트).

    state는 cmd 수신 여부와 무관하게 매 step (HOST, state_port)로 발행한다(sim_runner_go2 패턴).
    GUI의 startup 실측 latch(첫 state로 현재 자세를 잡은 뒤에야 발행 시작)가 이를 전제한다 —
    cmd를 받아야만 회신하는 구조면 GUI가 영원히 실측을 못 받는 닭-달걀이 된다.
    """
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, args_cli.cmd_port))
    recv_sock.setblocking(False)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_addr = (HOST, args_cli.state_port)

    print(
        f"[sim_runner_bipedleg] position mode — UDP listening on {HOST}:{args_cli.cmd_port}, "
        f"sending state to {HOST}:{args_cli.state_port}",
        flush=True,
    )

    # 관절별 유효 관성 (default 자세 기준, 자세 의존이라 startup 1회만 계산). GUI의 계산 게인
    # (kp=I·ωn², kd=2ζ·I·ωn) 초기값용으로 1Hz로 state 포트에 흘린다.
    ieff = env.unwrapped.get_joint_ieff().numpy()
    print("[sim_runner_bipedleg] I_eff [kg·m²]: " + " ".join(f"{v:.4f}" for v in ieff), flush=True)

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    seq = 0
    # step_dt(20ms) wall-clock 페이싱 — 없으면 루프가 "도는 만큼만" 돌아 렌더/물리 부하에 따라
    # sim이 슬로모션 또는 fast-forward가 된다 (gui_controller publisher와 같은 sleep 패턴).
    period = float(env.unwrapped.step_dt)
    next_t = time.monotonic()
    rate_n, overrun_n = 0, 0
    rate_t0 = time.monotonic()
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
                    continue
                # PLANT(R2PP, one-shot): 플랜트 물성 선택 적용 — CMD와 크기·magic이 달라 오파싱 없음.
                pl = unpack_plant(data)
                if pl is not None:
                    env.unwrapped.set_plant_params(pl["mode"], pl["armature"], pl["viscous"], pl["coulomb"])
                    if pl["mode"] == 0:
                        print("[sim_runner_bipedleg] plant → stock cfg 복원", flush=True)
                    else:
                        arma = " ".join(f"{v:.4f}" for v in pl["armature"])
                        visc = " ".join(f"{v:.3f}" for v in pl["viscous"])
                        coulomb_s = " ".join(f"{v:.3f}" for v in pl["coulomb"])
                        print(
                            f"[sim_runner_bipedleg] plant → PACE 적용  armature=[{arma}] "
                            f"viscous=[{visc}] coulomb=[{coulomb_s}]",
                            flush=True,
                        )
            if last is not None:
                env.unwrapped.set_setpoint(last["q"], last["dq"], last["kp"], last["kd"], last["tau"])

            with torch.inference_mode():
                env.step(zero_action)

            # publish the resulting state (unconditional — GUI startup latch 전제)
            st = env.unwrapped.get_lowstate()
            sim_time = float(env.unwrapped.episode_length_buf[0].item()) * env.unwrapped.step_dt
            send_sock.sendto(pack_state(seq, sim_time, st["q"], st["dq"], st["ddq"], st["tau_est"]), state_addr)
            if seq % 50 == 0:  # 1Hz — I_eff는 상수라 저빈도면 충분
                send_sock.sendto(pack_ieff(seq, ieff), state_addr)
            seq += 1

            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()  # 밀렸으면 리싱크 (누적 드리프트 방지)
                overrun_n += 1
            rate_n += 1
            now = time.monotonic()
            if now - rate_t0 >= 10.0:
                rate = rate_n / (now - rate_t0)
                if rate < 0.9 / period:  # 지속 저속 = 페이싱으로 못 잡는 병목 — 슬로모션 상태
                    print(
                        f"[sim_runner_bipedleg] WARN: loop {rate:.1f}Hz < target {1.0 / period:.0f}Hz "
                        f"({overrun_n}/{rate_n} overruns) — sim is below real time; "
                        "raise --render_decimation",
                        flush=True,
                    )
                rate_n, overrun_n, rate_t0 = 0, 0, now
    finally:
        recv_sock.close()
        send_sock.close()
        env.close()


def _run_policy_loop(env) -> None:
    """Policy 브릿지 (policy_runner_bipedleg.py <-> sim).

    POLICY_ACT_PORT 에서 articulation-순서 목표각을 받아 set_policy_target 으로 적용하고,
    매 step 후 rich state(q,dq,gravity)를 POLICY_STATE_PORT 로 회신한다 (latest-wins).
    """
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, POLICY_ACT_PORT))
    recv_sock.settimeout(0.1)  # action 대기(blocking) — 없으면 step하지 않고 upright reset 유지
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(
        f"[sim_runner_bipedleg] policy mode (lockstep) — UDP listening on {HOST}:{POLICY_ACT_PORT}, "
        f"sending rich state to sender:{POLICY_STATE_PORT}",
        flush=True,
    )

    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    seq = 0
    try:
        while simulation_app.is_running():
            # Lockstep: POLICY_ACT 하나당 정확히 1 env.step (학습 1 action = 1 step 불변식 유지).
            # action이 없으면 step하지 않는다 — 부팅 중/idle에 로봇이 default 자세로 넘어지는 것을 막는다.
            try:
                data, src = recv_sock.recvfrom(4096)
            except (TimeoutError, OSError):
                continue  # 타임아웃: step 없이 대기 (upright 유지)
            act = unpack_policy_act(data)
            # 큐에 쌓인 나머지는 버리고 최신만 사용 (latest-wins, 지연 누적 방지)
            recv_sock.setblocking(False)
            while True:
                try:
                    data2, src2 = recv_sock.recvfrom(4096)
                except (BlockingIOError, OSError):
                    break
                a2 = unpack_policy_act(data2)
                if a2 is not None:
                    act, src = a2, src2
            recv_sock.settimeout(0.1)
            if act is None:
                continue

            env.unwrapped.set_policy_target(act["target_q"])
            with torch.inference_mode():
                env.step(zero_action)  # 정확히 1 step

            # 이번 step 결과 rich state를 action 발신자(policy_runner)에게 회신
            st = env.unwrapped.get_policy_state()
            packet = pack_policy_state(seq, st["q"], st["dq"], st["gravity"])
            send_sock.sendto(packet, (src[0], POLICY_STATE_PORT))
            seq += 1
    finally:
        recv_sock.close()
        send_sock.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

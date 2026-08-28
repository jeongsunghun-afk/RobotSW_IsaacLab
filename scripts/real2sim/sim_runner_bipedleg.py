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
parser.add_argument(
    "--unified",
    action="store_true",
    default=False,
    help="Unified mode: CMD(슬라이더)와 POLICY_ACT(정책)를 한 프로세스에서 받는다. 정책이 도착한 "
    "스텝은 lockstep + slew 우회(= --policy_mode 와 동일 경로), 아니면 슬라이더 free-run. "
    "--fix_base 와 함께 쓸 수 있다 — 공중 고정 정책 구동은 넘어질 수 없어 안전 점검의 첫 칸이다 "
    "(단 base 상태가 학습 분포 밖이고 접촉이 없어 보행 성능 판정에는 못 쓴다).",
)
parser.add_argument(
    "--convention_version",
    type=int,
    default=None,
    help="Override the coordinate-convention version stamped into POLICY_STATE. Default: derived "
    "from the env (see _policy_state_convention_version).",
)
parser.add_argument("--cmd_port", type=int, default=CMD_PORT, help="UDP port to receive motor commands on.")
parser.add_argument("--state_port", type=int, default=STATE_PORT, help="UDP port to send sim state to.")
# 정책 seam 포트도 인자로 뺀다 — 상수로 두면 이미 돌고 있는 세션 옆에 두 번째 sim 을 못 띄운다
# (bind 충돌로 즉사). 검정용으로도, sim 두 개를 나란히 비교할 때도 필요하다.
parser.add_argument("--policy_act_port", type=int, default=POLICY_ACT_PORT, help="UDP port to receive POLICY_ACT on.")
parser.add_argument(
    "--policy_state_port", type=int, default=POLICY_STATE_PORT, help="UDP port to send POLICY_STATE to."
)
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


def _policy_state_convention_version(env_cfg) -> int:
    """POLICY_STATE 에 실을 좌표 규약 버전을 env cfg 에서 읽는다 (:data:`r2s_udp.R2S_CONVENTION_VERSION`).

    **env cfg 가 자기 규약을 신고하고, 여기서는 그대로 옮긴다.** 하드코딩하지 않는 이유는
    "송신자는 자기가 실제로 하는 것을 신고한다"가 이 필드의 계약이기 때문이다 — env 가 규약을
    바꿨는데 여기가 옛 값을 계속 보내면, 수신측(policy_runner)은 **바이트를 잘못 해석하고도
    조용히 돈다.** 크기가 같아 거부도 안 된다.

    ⚠ **2026-08-14 정정**: 이 docstring 은 한때 "코드 세 곳으로 확인한 사실"이라며 근거를 나열했는데,
    그중 둘이 좌표 이관(:data:`r2s_biped_leg_env_cfg.CONVENTION_VERSION` 0 → 1)으로 거짓이 됐다 —
    ``_apply_action`` 이 policy 를 즉시 return 한다는 서술(지금은 policy 전용 커플링 분기가 있다)과
    ``get_lowstate`` 가 foot 을 raw 로 보고한다는 서술(지금은 관절각이다)이다. **결론(버전 1)은
    그때도 지금도 맞지만 근거가 낡았다** — 낡은 근거로 재유도하면 "커플링이 없다"는 틀린 결론이
    나온다. 결론이 맞다고 근거까지 최신인 것은 아니다.
    (같은 유형: ``COORDINATE_MIGRATION_CHECKLIST.md`` §F6)

    Args:
        env_cfg: :class:`R2SBipedLegEnvCfg` — ``convention_version`` 필드를 읽는다.

    Returns:
        POLICY_STATE 에 스탬프할 규약 버전.

    Raises:
        RuntimeError: cfg 에 ``convention_version`` 이 없을 때. 기본값으로 넘기지 않는 이유는
            규약 미상 패킷을 내보내는 것이 조용한 오독으로 이어지기 때문이다.
    """
    ver = getattr(env_cfg, "convention_version", None)
    if ver is None:
        raise RuntimeError(
            "env cfg 에 convention_version 이 없다 — POLICY_STATE 규약을 추측해서 내보낼 수 없다. "
            "r2s_biped_leg_env_cfg.CONVENTION_VERSION 을 확인할 것."
        )
    return int(ver)


def main() -> None:
    """Run the UDP <-> Isaac Sim biped-leg bridge loop (latest-wins, non-blocking recv)."""
    # parse and override the environment configuration (CONTRACT §5: fix_base toggle)
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=args_cli.num_envs)
    if args_cli.policy_mode or args_cli.unified:
        # unified 는 스텝마다 cfg.policy_mode 를 뒤집으므로 여기서는 False 로 시작한다
        # (슬라이더 경로가 기본, 정책 패킷이 오는 스텝만 True).
        env_cfg.policy_mode = args_cli.policy_mode
        # ★ fix_base 를 강제로 끄지 않는다 — **공중 고정 정책 구동이 안전 사다리의 첫 칸**이다.
        #   넘어질 수 없는 상태에서 관절 거동·부호·토크를 먼저 보고 자유베이스로 내려간다.
        env_cfg.fix_base = args_cli.fix_base
        if args_cli.fix_base:
            print(
                "[sim_runner_bipedleg] ⚠ 공중 고정 + 정책 — 안전 점검용이다.\n"
                "   정책이 보는 base 상태(중력방향·base 속도)가 **고정되어 학습 분포 밖**이고,\n"
                "   발이 지면에 안 닿아 접촉이 없다. 관절 거동·부호·토크 크기 확인에는 쓰되,\n"
                "   **보행 성능·트립 판정을 이 모드에서 내리지 말 것** (자유베이스로 다시 잰다).",
                flush=True,
            )
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

    if args_cli.policy_mode or args_cli.unified:
        conv_ver = (
            args_cli.convention_version
            if args_cli.convention_version is not None
            else _policy_state_convention_version(env_cfg)
        )
        if args_cli.unified:
            _run_unified_loop(env, conv_ver)
        else:
            _run_policy_loop(env, conv_ver)
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


def _drain_cmd(env, recv_sock) -> dict | None:
    """CMD 큐를 비우고 **최신 하나**만 돌려준다. PLANT(R2PP) 는 도착 즉시 적용한다."""
    last = None
    while True:
        try:
            data, _ = recv_sock.recvfrom(4096)
        except (BlockingIOError, OSError):
            break
        cmd = unpack_cmd(data)
        if cmd is not None:
            last = cmd
            continue
        pl = unpack_plant(data)
        if pl is not None:
            env.unwrapped.set_plant_params(pl["mode"], pl["armature"], pl["viscous"], pl["coulomb"])
            print(
                "[sim_runner_bipedleg] plant → stock cfg 복원"
                if pl["mode"] == 0
                else "[sim_runner_bipedleg] plant → PACE 적용",
                flush=True,
            )
    return last


def _run_unified_loop(env, conv_ver: int) -> None:
    """CMD(슬라이더)와 POLICY_ACT(정책)를 **한 프로세스**에서 받는다.

    두 모드를 합치되, 갈라져 있던 이유였던 두 성질은 **그대로 지킨다** —
    합쳐서 문제가 됐던 것은 "한 프로세스"가 아니라 이 둘을 잃는 것이었다.

    ① **lockstep** — POLICY_ACT 하나당 정확히 1 env.step. 학습의 "1 action = 1 step" 불변식이다.
       자유 구동으로 돌렸을 때 sim 62.5 Hz vs 정책 50 Hz 의 홉별 지연으로 **발 오차 264 mm** 가
       났고 lockstep 으로 10~14 mm 가 됐다 (project_r2s_sim_runner_free_run_desync).
    ② **slew/faithful PD 우회** — 정책 목표는 `set_policy_target` 으로 직접 들어간다.
       live 경로(slew limiter + GUI 게인 write)를 태우면 배포 리허설이 **학습과 다른 경로**가 된다.

    분기 규칙 — **정책이 흐르는 동안은 정책이 로봇을 소유한다**:

    ``POLICY_ACT`` 가 도착하면 그 스텝은 정책 경로(lockstep). 타임아웃이면 슬라이더 경로
    (free-run 페이싱 + live PD). 정책이 멈추면 자동으로 슬라이더로 넘어간다.
    ``cfg.policy_mode`` 를 스텝마다 뒤집어 env 의 두 경로를 그대로 재사용한다
    (``_policy_target`` 은 생성 시점에 zeros 로, reset 에 default 자세로 초기화되므로 항상 유효).

    **fix_base 는 인자를 그대로 따른다.**
      - 자유베이스(기본): 정책이 균형을 잡는다. 슬라이더로 크게 움직이면 넘어진다 — 정상이다.
      - ``--fix_base``: 넘어질 수 없다. 정책 구동의 **안전 점검 첫 칸**으로 쓴다 — 관절 거동·부호·
        토크 크기를 먼저 본다. ⚠ 다만 정책이 보는 base 상태(중력방향·base 속도)가 고정돼
        **학습 분포 밖**이고 발이 지면에 안 닿아 접촉이 없다.
        **보행 성능·트립 판정은 이 모드에서 내리지 않는다.**

    Args:
        env: gym 환경 (fix_base=False).
        conv_ver: POLICY_STATE 에 실을 좌표 규약 버전.
    """
    cmd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    cmd_sock.bind((HOST, args_cli.cmd_port))
    cmd_sock.setblocking(False)
    pol_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pol_sock.bind((HOST, args_cli.policy_act_port))
    # 짧은 타임아웃 — 정책이 없으면 슬라이더 경로가 제때 돌아야 한다(step_dt 20ms 보다 짧게).
    pol_sock.settimeout(0.005)
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state_addr = (HOST, args_cli.state_port)

    print(
        f"[sim_runner_bipedleg] unified mode — CMD {HOST}:{args_cli.cmd_port} + "
        f"POLICY_ACT {HOST}:{args_cli.policy_act_port} (정책 도착 시 lockstep, 아니면 슬라이더 free-run)  "
        f"convention_version={conv_ver}  "
        + (
            "base=공중고정 — 넘어지지 않는다(안전 점검용, 접촉 없음)"
            if args_cli.fix_base
            else "⚠ base=자유 — 슬라이더로 크게 움직이면 넘어진다"
        ),
        flush=True,
    )

    ieff = env.unwrapped.get_joint_ieff().numpy()
    zero_action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    seq = 0
    period = float(env.unwrapped.step_dt)
    next_t = time.monotonic()
    # ★ 정책 소유 창 — 마지막 POLICY_ACT 이후 이 시간 안에는 **타임아웃에도 step 하지 않는다**.
    #   이게 없으면 정책 50 Hz(20 ms) 사이에 5 ms 타임아웃이 3 번 터져 슬라이더 경로가 끼어들고,
    #   결과적으로 정책 1 action 당 4 step 이 돌아 **lockstep 불변식이 조용히 깨진다**.
    #   `_run_policy_loop` 가 타임아웃에 step 없이 continue 하는 것과 같은 취지다.
    #
    # ⚠ 값이 **1.0 s** 인 이유 — 비용이 비대칭이다.
    #   너무 길면: 정책이 멈춘 뒤 슬라이더가 최대 1 s 늦게 듣는다 (불편할 뿐).
    #   너무 짧으면: 정책이 조금만 느려져도 슬라이더 스텝이 끼어들어 **lockstep 이 깨진다** (치명적).
    #   처음엔 0.2 s 였는데 그게 하필 GUI 의 state 대기 타임아웃(`recv_state_blocking` 기본 0.2 s)과
    #   **같은 값**이라, GUI 가 한 틱이라도 늦으면 정확히 경계에서 소유권이 떨었다
    #   (증상: 로그에 "정책이 차지 / 슬라이더가 차지"가 반복). 두 임계를 붙여 두면 안 된다.
    policy_own_s = 1.0
    last_pol_t = float("-inf")
    was_policy = False
    try:
        while simulation_app.is_running():
            # --- 정책 패킷 우선 확인 (최대 5ms 대기 후 latest-wins 배수) ---
            act, src = None, None
            try:
                data, src = pol_sock.recvfrom(4096)
                act = unpack_policy_act(data)
            except (TimeoutError, OSError):
                pass
            if act is not None:
                pol_sock.setblocking(False)
                while True:
                    try:
                        d2, s2 = pol_sock.recvfrom(4096)
                    except (BlockingIOError, OSError):
                        break
                    a2 = unpack_policy_act(d2)
                    if a2 is not None:
                        act, src = a2, s2
                pol_sock.settimeout(0.005)

            # 슬라이더 CMD 는 정책이 돌든 말든 **항상 배수**한다 — 안 비우면 큐가 쌓이고,
            # 정책이 멈춘 순간 오래된 명령이 한꺼번에 적용된다.
            last_cmd = _drain_cmd(env, cmd_sock)

            now = time.monotonic()
            if act is not None and src is not None:
                # ── 정책 경로: lockstep, slew/faithful PD 우회 ──
                if not was_policy:
                    print("[sim_runner_bipedleg] → 정책이 로봇을 소유 (lockstep)", flush=True)
                    was_policy = True
                last_pol_t = now
                env.unwrapped.cfg.policy_mode = True
                env.unwrapped.set_policy_target(act["target_q"])
                with torch.inference_mode():
                    env.step(zero_action)  # 정확히 1 step
                st_p = env.unwrapped.get_policy_state()
                send_sock.sendto(
                    pack_policy_state(seq, st_p["q"], st_p["dq"], st_p["gravity"], convention_version=conv_ver),
                    (src[0], args_cli.policy_state_port),
                )
                next_t = time.monotonic()  # 정책이 페이싱을 소유한다 — 벽시계 예산을 리싱크
            elif now - last_pol_t < policy_own_s:
                # 정책이 아직 소유 중인데 이번 폴에는 안 왔다 — **step 하지 않는다**(lockstep 보존).
                continue
            else:
                if was_policy:
                    print("[sim_runner_bipedleg] → 정책 정지, 슬라이더로 전환 (free-run)", flush=True)
                    was_policy = False
                # ── 슬라이더 경로: live PD + free-run 페이싱 ──
                env.unwrapped.cfg.policy_mode = False
                if last_cmd is not None:
                    env.unwrapped.set_setpoint(
                        last_cmd["q"], last_cmd["dq"], last_cmd["kp"], last_cmd["kd"], last_cmd["tau"]
                    )
                with torch.inference_mode():
                    env.step(zero_action)
                next_t += period
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_t = time.monotonic()

            # GUI 표시는 어느 경로든 살아 있어야 한다 (startup latch 가 state 발행을 전제한다).
            st = env.unwrapped.get_lowstate()
            sim_time = float(env.unwrapped.episode_length_buf[0].item()) * env.unwrapped.step_dt
            send_sock.sendto(pack_state(seq, sim_time, st["q"], st["dq"], st["ddq"], st["tau_est"]), state_addr)
            if seq % 50 == 0:
                send_sock.sendto(pack_ieff(seq, ieff), state_addr)
            seq += 1
    finally:
        cmd_sock.close()
        pol_sock.close()
        send_sock.close()
        env.close()


def _run_policy_loop(env, conv_ver: int) -> None:
    """Policy 브릿지 (policy_runner_bipedleg.py <-> sim).

    POLICY_ACT_PORT 에서 articulation-순서 목표각을 받아 set_policy_target 으로 적용하고,
    매 step 후 rich state(q,dq,gravity)를 POLICY_STATE_PORT 로 회신한다 (latest-wins).

    Args:
        env: gym 환경 (policy_mode=True).
        conv_ver: STATE 에 실을 좌표 규약 버전 — :func:`_policy_state_convention_version` 참조.
    """
    recv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    recv_sock.bind((HOST, args_cli.policy_act_port))
    recv_sock.settimeout(0.1)  # action 대기(blocking) — 없으면 step하지 않고 upright reset 유지
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(
        f"[sim_runner_bipedleg] policy mode (lockstep) — UDP listening on {HOST}:{args_cli.policy_act_port}, "
        f"sending rich state to sender:{args_cli.policy_state_port}  (convention_version={conv_ver})",
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
            packet = pack_policy_state(seq, st["q"], st["dq"], st["gravity"], convention_version=conv_ver)
            send_sock.sendto(packet, (src[0], args_cli.policy_state_port))
            seq += 1
    finally:
        recv_sock.close()
        send_sock.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

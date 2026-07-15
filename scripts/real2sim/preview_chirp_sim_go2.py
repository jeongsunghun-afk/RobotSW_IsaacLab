# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 chirp 미리보기 — `r2s_chirp`가 내보내는 명령을 매달린 sim GO2에서 재생하고 mp4로 녹화.

실기에 chirp를 걸기 전에 **실제 로봇이 얼마나 크게/과격하게 움직이는지** 눈으로 확인하는 용도.
`Isaac-R2S-Go2-Sysid-v0`(fix_base=공중 고정, PaceDCMotor kp=25/kd=0.5)를 num_envs=1로 띄워:

    1. 스폰(prone)에서 chirp 중심으로 부드럽게 approach 램프(수집기와 동일한 선행 이동),
    2. `chirp.py`의 chirp(0.1→10 Hz, 20 s, 500 Hz)를 그대로 재생,
    3. 매 ``--capture_every`` 스텝마다 프레임을 잡아 실시간 배속 mp4로 저장.

명령(des)은 고주파에서 매우 빠르지만(calf 명령 각속도 ~31 rad/s), 실제 움직임은 kp=25 토크
한계로 억제된다 — 이 영상은 그 **실제 응답**을 보여준다. RL/UDP 없음.

실행::

    env -u DISPLAY ./isaaclab.sh -p scripts/real2sim/preview_chirp_sim_go2.py [--low_freq_only]
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_go2"))
import chirp  # isort: skip

parser = argparse.ArgumentParser(description="R2S-GO2 chirp preview video (suspended sim GO2).")
parser.add_argument("--out", type=str, default="logs/r2s_chirp_preview/chirp_preview.mp4", help="출력 mp4 경로.")
parser.add_argument("--approach", type=float, default=2.0, help="스폰→chirp 중심 램프 시간 [s].")
parser.add_argument("--hold", type=float, default=1.0, help="chirp 전 중심 정착 홀드 [s].")
parser.add_argument("--capture_every", type=int, default=10, help="몇 스텝마다 1프레임 (500Hz/10 = 50fps 실시간).")
parser.add_argument("--fps", type=int, default=50, help="출력 mp4 프레임레이트.")
parser.add_argument(
    "--low_freq_only",
    action="store_true",
    help="안전상 중요한 저주파(0.1→2 Hz, 전진폭 스윕)만 녹화. 고주파 버징 구간 생략.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# rgb_array 렌더는 카메라가 필요하다 (headless --video 경로와 동일).
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import imageio
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.direct.r2s_go2.r2s_go2_env_cfg import JOINT_ORDER
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR  # isort: skip

TASK_NAME = "Isaac-R2S-Go2-Sysid-v0"


def use_noninstanceable_go2_for_render(env_cfg) -> None:
    """스톡 instanceable ``go2.usd``를 flatten된 non-instanceable 사본으로 교체.

    표준 instanceable 에셋은 Isaac Sim 6.0 렌더 경로 버그(IsaacLab #2925)로 다리 일부만
    확장돼 조각난 로봇이 녹화된다. flatten본은 링크마다 비주얼을 갖고 있어 온전히 렌더된다.
    """
    robot_cfg = getattr(env_cfg, "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None) if robot_cfg is not None else None
    usd_path = getattr(spawn_cfg, "usd_path", None) if spawn_cfg is not None else None
    if spawn_cfg is None or not (isinstance(usd_path, str) and usd_path.endswith("/Robots/Unitree/Go2/go2.usd")):
        return
    noninst_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Go2_noninstanceable", "go2.usd")
    if not os.path.isfile(noninst_path):
        print(
            f"[WARN] non-instanceable Go2 에셋 없음: {noninst_path} — 조각 렌더 가능. "
            "생성: ./isaaclab.sh -p scripts/tools/make_go2_noninstanceable.py"
        )
        return
    spawn_cfg.usd_path = noninst_path
    print(f"[INFO] non-instanceable Go2로 렌더 (6.0 render fix): {noninst_path}", flush=True)


def main() -> None:
    """approach 램프 + chirp 재생을 매달린 GO2에서 돌리고 프레임을 모아 mp4로 저장."""
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
    # capture_every 스텝마다만 렌더 (500Hz 매 스텝 렌더는 과하다).
    env_cfg.sim.render_interval = args_cli.capture_every
    use_noninstanceable_go2_for_render(env_cfg)

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    sim_joint_ids = torch.tensor(
        [robot.joint_names.index(name) for name in JOINT_ORDER], dtype=torch.int32, device=device
    )

    # chirp 궤적 (JOINT_ORDER 순서, 500 Hz). sin(0)=0이라 첫 프레임이 곧 중심.
    f1 = 2.0 if args_cli.low_freq_only else chirp.DEFAULT_F1_HZ
    _, targets = chirp.build_chirp(rate_hz=chirp.DEFAULT_RATE_HZ, f1=f1)
    q_chirp = torch.tensor(targets, dtype=torch.float32, device=device)
    center = q_chirp[0].clone()

    action = torch.zeros(env.action_space.shape, device=device)
    frames: list = []
    k = 0

    def step(cmd_row: torch.Tensor) -> None:
        nonlocal k
        action[:, sim_joint_ids] = cmd_row.unsqueeze(0)
        with torch.inference_mode():
            env.step(action)
        if k % args_cli.capture_every == 0:
            frames.append(env.render())
        k += 1

    # 1) approach: 스폰(prone) → chirp 중심 선형 램프 (수집기의 선행 이동과 동일 취지).
    start = robot.data.joint_pos[0, sim_joint_ids].clone()
    n_appr = max(1, int(args_cli.approach * chirp.DEFAULT_RATE_HZ))
    for i in range(n_appr):
        a = (i + 1) / n_appr
        step(start * (1.0 - a) + center * a)
    # 2) hold: 중심에서 정착.
    for _ in range(max(0, int(args_cli.hold * chirp.DEFAULT_RATE_HZ))):
        step(center)
    # 3) chirp 재생.
    print(f"[preview] chirp 재생: {q_chirp.shape[0]} steps (f1={f1:.0f} Hz)", flush=True)
    for i in range(q_chirp.shape[0]):
        step(q_chirp[i])
        if i % 2000 == 0:
            print(f"[preview] chirp {i}/{q_chirp.shape[0]}  frames={len(frames)}", flush=True)

    out_path = args_cli.out if os.path.isabs(args_cli.out) else os.path.join(os.getcwd(), args_cli.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[preview] {len(frames)} 프레임 → {out_path} @ {args_cli.fps} fps", flush=True)
    imageio.mimwrite(out_path, frames, fps=args_cli.fps, macro_block_size=None)
    print(f"[preview] 저장 완료: {out_path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg chirp 미리보기 — `r2s_biped_leg/chirp.py`의 명령을 sim 2족 다리에서 재생하고 mp4로 녹화.

수집을 돌리기 전에 **다리가 얼마나 크게/과격하게 움직이는지** 눈으로 확인하는 용도.
`Isaac-R2S-BipedLeg-Sysid-v0`(fix_base=공중 고정, PaceDCMotor 관절별 게인)를 num_envs=1로 띄워:

    1. 스폰 자세에서 chirp 중심으로 부드럽게 approach 램프(수집기와 동일한 선행 이동),
    2. `chirp.py`의 chirp(0.1→10 Hz, 20 s, 500 Hz)를 그대로 재생,
    3. 매 ``--capture_every`` 스텝마다 프레임을 잡아 실시간 배속 mp4로 저장.

명령(des)은 고주파에서 매우 빠르지만 실제 움직임은 게인/토크 한계로 억제된다 — 이 영상은 그
**실제 응답**을 보여준다. RL/UDP 없음.

⚠ f1=10 Hz는 ``fix_base=True``(world에 용접) 전제다. 실기 매단 리그에서는 리그 공진 때문에
2 Hz 상한이 필요하다 — 그 경우 ``--low_freq_only``로 미리보기도 같은 대역만 확인할 것.

실행::

    env -u DISPLAY ./isaaclab.sh -p scripts/real2sim/preview_chirp_sim_bipedleg.py [--low_freq_only]

미리보기로 안전을 확인한 뒤 수집기가 실행하는 3종 (`collect_chirp_sim_bipedleg.py`)::

    --gain_scale 0.6 --amplitude_scale 1.0 --out chirp_g060.pt   # fit
    --gain_scale 1.6 --amplitude_scale 0.6 --out chirp_g160.pt   # fit
    --gain_scale 1.0 --amplitude_scale 0.8 --out chirp_g100.pt   # hold-out(미관측 게인)
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_biped_leg"))
import chirp  # isort: skip

parser = argparse.ArgumentParser(description="R2S-BipedLeg chirp preview video (fixed-base sim biped leg).")
parser.add_argument(
    "--out", type=str, default="logs/r2s_chirp_preview/chirp_preview_bipedleg.mp4", help="출력 mp4 경로."
)
parser.add_argument("--approach", type=float, default=2.0, help="스폰→chirp 중심 램프 시간 [s].")
parser.add_argument("--hold", type=float, default=1.0, help="chirp 전 중심 정착 홀드 [s].")
parser.add_argument("--capture_every", type=int, default=10, help="몇 스텝마다 1프레임 (500Hz/10 = 50fps 실시간).")
parser.add_argument("--fps", type=int, default=50, help="출력 mp4 프레임레이트.")
parser.add_argument(
    "--amplitude_scale", type=float, default=1.0, help="기본 chirp 진폭에 걸 배율 (수집기와 동일 의미)."
)
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
from isaaclab_tasks.direct.r2s_biped_leg.r2s_biped_leg_env_cfg import JOINT_LABELS, JOINT_NAME_PATTERNS
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

TASK_NAME = "Isaac-R2S-BipedLeg-Sysid-v0"


def report_trajectory_range(targets: list[list[float]]) -> None:
    """관절별 궤적 범위를 soft limit과 나란히 출력하고 위반이 있으면 경고한다.

    수집기(``collect_chirp_sim_bipedleg.py``)는 같은 검사에서 **중단**하지만, 미리보기는
    "얼마나 아슬아슬한지"를 보는 도구라 경고만 하고 재생을 계속한다.

    Args:
        targets: T×8 목표각 [rad], ``JOINT_NAME_PATTERNS`` 순서.
    """
    violations = {j for j, _, _ in chirp.check_within_soft_limits(targets)}
    for j, label in enumerate(JOINT_LABELS):
        lo = min(row[j] for row in targets)
        hi = max(row[j] for row in targets)
        soft = chirp.SOFT_LIMITS[j]
        flag = "  <-- soft limit 위반" if j in violations else ""
        print(f"[preview]   {label:9s} range=[{lo:+.3f}, {hi:+.3f}]  soft=({soft[0]:+.4f}, {soft[1]:+.4f}){flag}")
    if violations:
        print("[preview] ⚠ soft limit 위반 — sim이 목표각을 조용히 클램프한다. amplitude_scale을 낮출 것.", flush=True)


def main() -> None:
    """approach 램프 + chirp 재생을 공중 고정된 2족 다리에서 돌리고 프레임을 모아 mp4로 저장."""
    env_cfg = parse_env_cfg(TASK_NAME, device=args_cli.device, num_envs=1)
    # capture_every 스텝마다만 렌더 (500Hz 매 스텝 렌더는 과하다).
    env_cfg.sim.render_interval = args_cli.capture_every

    env = gym.make(TASK_NAME, cfg=env_cfg, render_mode="rgb_array")
    env.reset()

    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    # 여기서는 warp 커널을 부르지 않으므로 인덱싱용 long이면 충분하다 (수집기는 int32가 필요).
    sim_joint_ids = torch.tensor(
        [robot.joint_names.index(name) for name in JOINT_NAME_PATTERNS], dtype=torch.long, device=device
    )

    # chirp 궤적 (JOINT_NAME_PATTERNS 순서, 500 Hz). sin(0)=0이라 첫 프레임이 곧 중심.
    f1 = 2.0 if args_cli.low_freq_only else chirp.DEFAULT_F1_HZ
    amplitude = [a * args_cli.amplitude_scale for a in chirp.CHIRP_AMPLITUDE]
    _, targets = chirp.build_chirp(rate_hz=chirp.DEFAULT_RATE_HZ, f1=f1, amplitude=amplitude)
    report_trajectory_range(targets)
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

    # 1) approach: 스폰 자세 → chirp 중심 선형 램프 (수집기의 선행 이동과 동일 취지).
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

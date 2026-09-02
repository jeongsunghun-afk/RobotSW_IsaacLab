# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""리셋 초기 상태 노이즈가 실제로 걸리는지 검정한다 (2026-09-02).

왜 필요한가
-----------
종전 `_reset_idx` 는 `default_joint_pos` / `default_root_state` 를 그대로 써서 **모든
에피소드가 완전히 같은 자세**(전 관절 0, 정확히 직립)에서 시작했다. 노이즈를 넣었으니
"정말 걸렸는가"와 "한계를 넘지 않는가"를 실측으로 확인한다.

⚠ 이 노이즈는 EventCfg 가 아니라 `_reset_idx` 인라인이다 — reset 이벤트는
`DirectRLEnv._reset_idx` 안에서 적용되고 그 뒤 env 가 관절 상태를 덮어쓰기 때문이다.
그래서 "cfg 에 값이 있다"가 아니라 **sim 에 실제로 쓰인 상태**를 읽어야 검정이 성립한다.

실행::

    ./isaaclab.sh -p _workspace/hindleg_reset_noise_check.py --headless
    ./isaaclab.sh -p _workspace/hindleg_reset_noise_check.py --headless --joint_noise 0 --rp_noise 0
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="HindLeg-Direct-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--joint_noise", type=float, default=None, help="cfg 기본값 대신 이 값을 쓴다 [rad]")
parser.add_argument("--rp_noise", type=float, default=None, help="cfg 기본값 대신 이 값을 쓴다 [deg]")
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

OK = True


def chk(name: str, cond: bool, extra: str = "") -> None:
    global OK
    print(("[PASS] " if cond else "[FAIL] ") + name + ("  " + extra if extra else ""), flush=True)
    OK = OK and bool(cond)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device
    if args_cli.joint_noise is not None:
        env_cfg.reset_joint_pos_noise = args_cli.joint_noise
    if args_cli.rp_noise is not None:
        env_cfg.reset_base_rp_noise_deg = args_cli.rp_noise
    jn = float(env_cfg.reset_joint_pos_noise)
    rp = float(env_cfg.reset_base_rp_noise_deg)
    print(f"[cfg] reset_joint_pos_noise={jn} rad · reset_base_rp_noise_deg={rp} deg · envs={args_cli.num_envs}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    u.reset()

    def t(x):
        return x.torch if hasattr(x, "torch") else x

    q = t(u._robot.data.joint_pos).clone()
    quat = t(u._robot.data.root_quat_w).clone()  # (x, y, z, w)
    default_q = t(u._robot.data.default_joint_pos).clone()
    soft = t(u._robot.data.soft_joint_pos_limits).clone()
    names = u._robot.joint_names

    dev = q - default_q
    print("\n관절별 초기 자세 편차 [rad] (default 대비)")
    for j, nm in enumerate(names):
        print(
            f"  {nm:16s} std {dev[:, j].std():.4f}  |max| {dev[:, j].abs().max():.4f}"
            f"   soft [{soft[0, j, 0]:+.4f}, {soft[0, j, 1]:+.4f}]"
        )

    # roll/pitch 복원 — quat 은 (x, y, z, w)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    print(
        f"\nbase roll  |max| {math.degrees(roll.abs().max()):.2f} deg"
        f"  · pitch |max| {math.degrees(pitch.abs().max()):.2f} deg"
        f"  · yaw |max| {math.degrees(yaw.abs().max()):.4f} deg"
    )
    print()

    if jn > 0.0:
        chk("관절 자세가 env 마다 다르다", bool((dev.std(dim=0) > 1e-6).all()),
            f"최소 std {dev.std(dim=0).min():.5f}")
        chk("관절 편차가 노이즈 반폭 이내", bool(dev.abs().max() <= jn + 1e-5),
            f"|max| {dev.abs().max():.4f} <= {jn}")
        chk("hip 이 노이즈로 한계를 넘지 않는다", bool((q >= soft[..., 0] - 1e-5).all() and (q <= soft[..., 1] + 1e-5).all()))
    else:
        chk("노이즈 0 이면 전 env 동일 자세(종전 동작)", bool(dev.abs().max() < 1e-6),
            f"|max| {dev.abs().max():.2e}")

    if rp > 0.0:
        lim = math.radians(rp) + 1e-4
        chk("base roll/pitch 가 env 마다 다르다", bool(roll.std() > 1e-6 and pitch.std() > 1e-6))
        chk("base roll/pitch 가 노이즈 반폭 이내", bool(roll.abs().max() <= lim and pitch.abs().max() <= lim),
            f"{math.degrees(roll.abs().max()):.2f}/{math.degrees(pitch.abs().max()):.2f} <= {rp} deg")
        chk("yaw 는 안 흔든다", bool(yaw.abs().max() < 1e-4))
    else:
        chk("노이즈 0 이면 base 직립(종전 동작)", bool(roll.abs().max() < 1e-6 and pitch.abs().max() < 1e-6))

    env.close()
    print("\n전 항목 통과" if OK else "\n실패 있음")


main()
simulation_app.close()

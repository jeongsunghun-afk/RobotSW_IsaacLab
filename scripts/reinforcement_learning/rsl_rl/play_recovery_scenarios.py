# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2Recovery-FlipVel 학습 정책을 3가지 초기 자세로 강제해 영상으로 확인.

시나리오 (env를 3그룹으로 균등 분할):
  - flipped: 완전히 뒤집힘 (belly-up, roll=180°)
  - side:    옆으로 누움 (roll=90°)
  - sitting: 앉은 자세 (Genesis sitting pose)에서 기립

기존 랜덤 fall-init을 monkeypatch로 대체해 각 그룹의 자세를 결정론적으로 고정한다.
정책/네트워크/obs는 그대로 사용 (학습 시와 동일 추론 경로).
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play Go2Recovery-FlipVel with forced initial poses.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model_*.pt checkpoint.")
parser.add_argument(
    "--scenario",
    type=str,
    default="all",
    choices=["all", "flipped", "side", "sitting"],
    help="Force ALL envs into one initial pose (flipped/side/sitting), or 'all' to split into 3 groups.",
)
parser.add_argument("--num_envs", type=int, default=30, help="Total envs.")
parser.add_argument("--video_length", type=int, default=300, help="Recorded video length in steps (50 Hz).")
parser.add_argument("--video_folder", type=str, required=True, help="Output folder for the recorded video.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
from importlib import metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils.math import quat_from_euler_xyz

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg

TASK = "Go2Recovery-FlipVel-v0"


def use_noninstanceable_go2_for_render(env_cfg) -> None:
    """Swap the stock instanceable go2.usd for a non-instanceable copy (Isaac Sim 6.0 render fix).

    Stock instanceable Go2 renders fragmented in the play/camera path (IsaacLab #2925).
    """
    from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

    robot_cfg = getattr(env_cfg, "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None) if robot_cfg is not None else None
    usd_path = getattr(spawn_cfg, "usd_path", None) if spawn_cfg is not None else None
    if spawn_cfg is None or not (isinstance(usd_path, str) and usd_path.endswith("/Robots/Unitree/Go2/go2.usd")):
        return
    noninst_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Go2_noninstanceable", "go2.usd")
    if not os.path.isfile(noninst_path):
        print(f"[WARN] non-instanceable Go2 not found at {noninst_path}; robot may render fragmented.")
        return
    spawn_cfg.usd_path = noninst_path
    print(f"[INFO] Rendering with non-instanceable Go2 asset: {noninst_path}")


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    use_noninstanceable_go2_for_render(env_cfg)
    # 지형/씬 단순화 없이 학습과 동일 cfg 사용.
    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args_cli.video_folder,
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 정책 로드 (play.py와 동일: deprecated cfg → actor/critic 변환 + PPO 시그니처 필터) ──
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))

    import inspect

    from rsl_rl.algorithms.ppo import PPO as _VendoredStockPPO

    _accepted = set(inspect.signature(_VendoredStockPPO.__init__).parameters.keys()) - {"self"}
    _cfg_dict = agent_cfg.to_dict()
    _cfg_dict["algorithm"] = {k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted or k == "class_name"}
    runner = OnPolicyRunner(env, _cfg_dict, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── fall-init monkeypatch: 3그룹 결정론적 자세 강제 ──────────────────────────
    base = env.unwrapped
    n = base.num_envs
    if args_cli.scenario == "all":
        g = n // 3
        # env 인덱스 → 시나리오: [0:g)=flipped, [g:2g)=side, [2g:n)=sitting
        flipped_ids = torch.arange(0, g, device=base.device)
        side_ids = torch.arange(g, 2 * g, device=base.device)
        sitting_ids = torch.arange(2 * g, n, device=base.device)
    else:
        # 단일 시나리오: 모든 env를 같은 자세로.
        allids = torch.arange(0, n, device=base.device)
        empty = torch.arange(0, 0, device=base.device)
        flipped_ids = allids if args_cli.scenario == "flipped" else empty
        side_ids = allids if args_cli.scenario == "side" else empty
        sitting_ids = allids if args_cli.scenario == "sitting" else empty

    def forced_reset_idx(env_ids=None):
        if env_ids is None:
            env_ids = base._robot._ALL_INDICES
        # 원본 reset의 buffer 초기화 로직을 그대로 태운다 (success/settle 등).
        Go2Base_reset_idx(env_ids)
        # 그 직후 root pose/joint를 시나리오 자세로 덮어쓴다.
        ids = env_ids
        for group, roll_val, z_val, use_sitting in [
            (flipped_ids, torch.pi, 0.16, False),
            (side_ids, torch.pi / 2.0, 0.14, False),
            (sitting_ids, 0.0, None, True),
        ]:
            sel = ids[torch.isin(ids, group)]
            if len(sel) == 0:
                continue
            if use_sitting:
                base._reset_sitting(sel)
                base._started_fallen[sel] = True  # 기립 성공을 유효 카운트
                continue
            m = len(sel)
            roll = torch.full((m,), roll_val, device=base.device)
            pitch = torch.zeros(m, device=base.device)
            yaw = torch.zeros(m, device=base.device)
            quat = quat_from_euler_xyz(roll, pitch, yaw)
            joint_pos = base._robot.data.default_joint_pos[sel].clone()
            joint_vel = torch.zeros_like(joint_pos)
            root_state = base._robot.data.default_root_state[sel].clone()
            root_state[:, :3] += base._terrain.env_origins[sel]
            root_state[:, 2] = base._terrain.env_origins[sel][:, 2] + z_val
            root_state[:, 3:7] = quat
            root_state[:, 7:] = 0.0
            base._robot.write_root_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=sel)
            base._robot.write_root_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=sel)
            base._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=sel)
            base._started_fallen[sel] = True

    Go2Base_reset_idx = base._reset_idx
    base._reset_idx = forced_reset_idx

    # reset() → 강제 자세 적용 + RecordVideo 녹화 시작(step_trigger=0).
    # episode_length_s가 길어 video_length 동안 자동 리셋 없이 한 에피소드 촬영.
    obs, _ = env.reset()
    with torch.inference_mode():
        for _ in range(args_cli.video_length + 5):
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

    env.close()
    print(f"[DONE] video saved under: {args_cli.video_folder}")


if __name__ == "__main__":
    main()
    simulation_app.close()

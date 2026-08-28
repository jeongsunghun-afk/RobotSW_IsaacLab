# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""env 자체 명령 분포에서 **보행 종류를 한 번에 대량 표집**한다.

램프 평가는 직진 전용이라 명령 공간의 한 선(線)만 훑는다. 그래서 "램프에는 gallop 이 없는데
학습 중간 영상에는 보인다"는 관찰을 램프로는 못 푼다 — 단일 env 로 축을 하나씩(체크포인트,
명령 동역학, 선회) 움직여 봐도 전부 pace 였다.

여기서는 **명령을 아예 건드리지 않는다.** env 의 `_resample_steering` 이 학습 때와 같은 분포로
vx·yaw 를 뿌리게 두고, 수천 개 env 를 한 에피소드만 돌려 (명령, 보행) 쌍을 한꺼번에 얻는다.
`play.py` 와 같은 조건이면서 npz 가 남아 정량화된다(play 경로는 상태를 안 남긴다).

★ 리셋된 env 는 버린다 — 에피소드 경계를 걸치면 위상이 끊겨 아무 보행으로나 분류된다.
★ 넘어진 env 도 버린다 — 누워서 버둥거려도 순시 위상은 나온다.

Run:
    CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p _workspace/leg/gait_survey_multienv.py \
        --checkpoint <RUN>/model_49999.pt --run_params <RUN>/params/env.yaml \
        --n_envs 4096 --dur_s 8 --out_dir _workspace/leg/gait_survey
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--run_params", type=str, default=None, help="run 의 params/env.yaml (cfg 정합용)")
parser.add_argument("--n_envs", type=int, default=4096)
parser.add_argument("--dur_s", type=float, default=8.0, help="기록 길이 [s]")
parser.add_argument("--out_dir", type=str, default="_workspace/leg/gait_survey")
parser.add_argument(
    "--all_stand",
    action="store_true",
    help="모든 env 를 정지 자세에서 리셋(`rel_stand_envs=1.0`). 램프 평가의 `--force_stand` 와 같은 "
    "초기 조건이므로, 이걸 켜서 보행 분포가 달라지면 원인이 **명령이 아니라 초기 상태**다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import inspect  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPO  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerParkourAMP  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

TASK = "Leg-Imitation-Tracking-RMA-v0"
os.makedirs(args_cli.out_dir, exist_ok=True)

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.n_envs)

# run 의 cfg 를 그대로 얹는다 — 명령 범위·보상 스케일이 다르면 다른 정책을 재는 셈이 된다.
if args_cli.run_params:
    with open(args_cli.run_params) as f:
        saved = yaml.unsafe_load(f)
    for k in ("lin_vel_x_min", "lin_vel_x_max", "lin_vel_y_min", "lin_vel_y_max",
              "yaw_vel_min", "yaw_vel_max", "motion_file", "motion_weight_mode",
              "vel_err_scale", "reset_strategy", "rel_stand_envs", "cmd_deadzone"):
        if k in saved and hasattr(env_cfg, k):
            setattr(env_cfg, k, saved[k])

if args_cli.all_stand:
    env_cfg.reset_strategy = "random_stand"
    env_cfg.rel_stand_envs = 1.0

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
dt = 1.0 / env_cfg.policy_dt_hz
total_steps = int(round(args_cli.dur_s / dt))

env = gym.make(TASK, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

_accepted = set(inspect.signature(_VendoredPPO.__init__).parameters.keys()) - {"self"}
_cfg = agent_cfg.to_dict()
_cfg["algorithm"] = {k: v for k, v in _cfg["algorithm"].items() if k in _accepted or k == "class_name"}
runner = OnPolicyRunnerParkourAMP(env, _cfg, log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

base = env.unwrapped
# ★ 명령 샘플러를 **건드리지 않는다** — 이 스크립트의 존재 이유다.
print(f">>> {args_cli.n_envs} envs · {total_steps} steps · env 자체 명령 분포")
print(f">>> vx [{env_cfg.lin_vel_x_min}, {env_cfg.lin_vel_x_max}]  "
      f"vy [{env_cfg.lin_vel_y_min}, {env_cfg.lin_vel_y_max}]  "
      f"yaw [{env_cfg.yaw_vel_min}, {env_cfg.yaw_vel_max}]")

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

N = args_cli.n_envs
jpos = torch.zeros(total_steps, N, base._robot.data.joint_pos.shape[1], dtype=torch.float16, device=base.device)
vxc = torch.zeros(total_steps, N, device=base.device)
yawc = torch.zeros(total_steps, N, device=base.device)
hgt = torch.zeros(total_steps, N, device=base.device)
elen = torch.zeros(total_steps, N, device=base.device)

with torch.inference_mode():
    for step in range(total_steps):
        actions = policy(obs)
        obs = env.step(actions)[0]
        d = base._robot.data
        jpos[step] = d.joint_pos.to(torch.float16)
        vxc[step] = base._lin_vel_cmd[:, 0]
        yawc[step] = base._yaw_vel_cmd
        hgt[step] = d.body_pos_w[:, base.ref_body_index, 2]
        elen[step] = base.episode_length_buf

env.close()

out = os.path.join(args_cli.out_dir, "gait_survey.npz")
np.savez_compressed(
    out,
    jpos=jpos.cpu().numpy(),
    vx_cmd=vxc.cpu().numpy(),
    yaw_cmd=yawc.cpu().numpy(),
    height=hgt.cpu().numpy(),
    ep_len=elen.cpu().numpy(),
    joint_names=np.array(base._robot.data.joint_names),
    dt=dt,
    checkpoint=args_cli.checkpoint,
)
print(f">>> 저장: {out}  jpos={tuple(jpos.shape)}")
app.close()

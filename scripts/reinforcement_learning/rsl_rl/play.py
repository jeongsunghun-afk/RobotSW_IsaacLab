# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# use --load_run

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--use_pretrained_checkpoint_local",
    action="store_true",
    help="Use the pre-trained checkpoint from local directory",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import os as _os
import socket
import sys as _sys
import time

import gymnasium as gym

# PLACEHOLDER: Extension template (do not remove this comment)
import numpy as np
import pandas as pd
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP, OnPolicyRunnerAMPBase
from rsl_rl.runners.on_policy_runner_parkour_amp import (
    OnPolicyRunnerParkourAMP,
    OnPolicyRunnerParkourAMPLidar,
    OnPolicyRunnerParkourAMPVoxel,
)

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_jit_parkour,
    export_policy_as_onnx,
    export_policy_as_onnx_parkour,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _debug.reward_publisher import RewardUDPPublisher  # noqa: E402


def save_obs_data_to_csv(obs_history, save_path, num_obs):
    """
    Saves observation buffer history to a CSV file.

    Parameters:
        obs_history (list of np.array): List of 1D arrays, each representing observations at a time step.
        save_path (str): Full path to save the CSV file.
        num_obs (int): The number of observations.
    """
    if not obs_history:
        print("Warning: Observation history is empty. Skipping CSV generation.")
        return

    obs_data = np.array(obs_history)

    if obs_data.size == 0:
        print("Warning: Observation data is effectively empty. Skipping CSV generation.")
        return
    # print(obs_data)

    num_timesteps, num_observations = obs_data.shape

    if num_observations != num_obs:
        print(
            f"Warning: Mismatch in CSV generation between num_observations in data ({num_observations}) and expected num_obs ({num_obs})."
        )
        num_obs = num_observations

    column_names = [f"obs_{i}" for i in range(num_obs)]
    data_dict = {"Time_Step": np.arange(num_timesteps)}
    for i, col_name in enumerate(column_names):
        data_dict[col_name] = obs_data[:, i]

    df = pd.DataFrame(data_dict)
    df.to_csv(save_path, index=False)
    print(f"Saved observation data to {save_path}")


def save_actions_to_csv(actions, save_path):
    """
    Saves a single timestep of actions to CSV.

    Parameters:
        actions (np.array): Action array with shape (num_envs, action_dim) or (action_dim,).
        save_path (str): Full path to save the CSV file.
    """
    if actions is None:
        print("Warning: Actions are None. Skipping CSV generation.")
        return

    actions_np = np.asarray(actions)
    if actions_np.ndim == 1:
        actions_np = actions_np[None, :]
    if actions_np.ndim != 2:
        print(f"Warning: Unexpected action array shape {actions_np.shape}. Skipping CSV generation.")
        return

    num_envs, action_dim = actions_np.shape
    records = []
    for env_id in range(num_envs):
        row = {"env_id": env_id}
        for i in range(action_dim):
            row[f"action_{i}"] = actions_np[env_id, i]
        records.append(row)

    pd.DataFrame(records).to_csv(save_path, index=False)
    print(f"Saved action data to {save_path}")


def print_action_joint_mapping(env):
    """Prints action index -> joint name mapping for environments with an articulation robot."""
    base_env = getattr(env, "unwrapped", env)
    robot = getattr(base_env, "_robot", None)
    if robot is None or not hasattr(robot, "data"):
        print("[WARN] Could not find robot instance to infer action mapping.")
        return

    joint_names = getattr(robot.data, "joint_names", None)
    if joint_names is None:
        print("[WARN] Robot joint names are unavailable; skipping action mapping print.")
        return

    action_dim = None
    if hasattr(base_env, "single_action_space"):
        action_dim = gym.spaces.flatdim(base_env.single_action_space)

    print("[INFO] Action index -> joint name mapping (Articulation joint order):")
    for idx, name in enumerate(joint_names):
        print(f"  {idx:02d}: {name}")

    if action_dim is not None and action_dim != len(joint_names):
        print(
            f"[WARN] Action dimension ({action_dim}) does not match number of joints ({len(joint_names)}). "
            "Actions still follow the listed articulation joint order."
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # ★연속 재생(teleop 관찰용): EPISODE_LEN_S 설정 시 에피소드 길이 오버라이드 → 타임아웃 리셋 방지
    if os.environ.get("EPISODE_LEN_S") and hasattr(env_cfg, "episode_length_s"):
        env_cfg.episode_length_s = float(os.environ["EPISODE_LEN_S"])
        print(f"[연속모드] episode_length_s = {env_cfg.episode_length_s}s (타임아웃 리셋 사실상 없음)")

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    # elif args_cli.use_pretrained_checkpoint_local:
    #     resume_path = f"logs/motion_jig_flat_direct"
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env_cfg.debug_vis = True
    # Enable height scanner ray visualization in play mode (if the scene defines one)
    if hasattr(env_cfg.scene, "height_scanner") and hasattr(env_cfg.scene.height_scanner, "debug_vis"):
        env_cfg.scene.height_scanner.debug_vis = True
    # Enable parkour edge mask visualization (if the cfg supports it)
    if hasattr(env_cfg, "debug_vis_edge_mask"):
        env_cfg.debug_vis_edge_mask = True
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped.set_debug_vis(getattr(env_cfg, "debug_vis", True))

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Print action index -> joint name mapping for debugging (e.g., MotionJig).
    print_action_joint_mapping(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    print(agent_cfg.class_name)
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkour":
        runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMP":
        runner = OnPolicyRunnerAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerAMPBase":
        runner = OnPolicyRunnerAMPBase(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMP":
        runner = OnPolicyRunnerParkourAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPVoxel":
        runner = OnPolicyRunnerParkourAMPVoxel(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPLidar":
        runner = OnPolicyRunnerParkourAMPLidar(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(policy)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer

    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if agent_cfg.class_name in (
        "OnPolicyRunnerParkour",
        "OnPolicyRunnerParkourAMP",
        "OnPolicyRunnerParkourAMPVoxel",
        "OnPolicyRunnerParkourAMPLidar",
    ):
        # Bundle estimator into exported graph if the algorithm has one (predicts priv_explicit from proprio).
        estimator = getattr(runner.alg, "estimator", None)
        export_policy_as_jit_parkour(
            policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt", estimator=estimator
        )
        export_policy_as_onnx_parkour(
            policy_nn, path=export_model_dir, normalizer=normalizer, filename="policy.onnx", estimator=estimator
        )
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Go2Recovery eval: settle 비활성화 (학습 시 2s 안착 단계 → eval에서 불필요)
    # 다른 task는 settle_max_steps 속성 자체가 없으므로 hasattr 가드로 무영향 보장.
    if hasattr(env.unwrapped.cfg, "settle_max_steps"):
        env.unwrapped.cfg.settle_max_steps = 0

    # reset environment
    obs = env.get_observations()
    timestep = 0
    obs_history = []
    action_history = []

    gaits = {
        "pronking": [0, 0, 0],
        "trotting": [0.5, 0, 0],
        "bounding": [0, 0.5, 0],
        "pacing": [0, 0, 0.5],
        "galloping": [0.25, 0.0, 0.0],
        "walking": [0.0, 0.25, 0.5],
        "ambling": [0.0, 0.25, 0.5],
        "cantering": [0.0, 0.3, 0.3],
        "half-bounding": [0.0, 0.25, 0.0],
        "gallop_rot": [0.4646, 0.0, 0.7677],
    }

    if args_cli.task == "Go2WTW" or args_cli.task == "Go2Neck":
        x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 1.5, 0.0, 0.0
        step_frequency_cmd = 2.0
        body_height_cmd = 0.0
        gait = torch.tensor(gaits["trotting"])
        footswing_height_cmd = 0.15
        pitch_cmd = 0.0
        roll_cmd = 0.0
        duration = 0.5
        stance_width_cmd = 0.25
        stance_length_cmd = 0.45
    # --- live reward UDP publisher (parkour only) ---
    _reward_publisher = None
    _task_str = (args_cli.task or "").lower()
    if "parkour" in _task_str:
        try:
            _reward_publisher = RewardUDPPublisher(env.unwrapped, target_env_id=0)
        except Exception as _exc:
            print(f"[reward-publisher] init failed: {_exc}; continuing without publisher.")
            _reward_publisher = None

    # ★UDP 원격 teleop: TELEOP_PORT 설정 시 외부 GUI 명령 수신
    _teleop_sock = None
    _teleop_cmd = {"v": 0.0, "vy": 0.0, "w": 0.0}
    if os.environ.get("TELEOP_PORT"):
        _teleop_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        _teleop_sock.setblocking(False)
        _teleop_sock.bind(("0.0.0.0", int(os.environ["TELEOP_PORT"])))
        print(f"[teleop] UDP 수신 대기 :{os.environ['TELEOP_PORT']}")

    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            if args_cli.task == "Go2WTW" or args_cli.task == "Go2Neck":
                env.unwrapped._commands[:, 0] = x_vel_cmd
                env.unwrapped._commands[:, 1] = y_vel_cmd
                env.unwrapped._commands[:, 2] = yaw_vel_cmd
                env.unwrapped._commands[:, 3] = body_height_cmd
                env.unwrapped._commands[:, 4] = step_frequency_cmd
                env.unwrapped._commands[:, 5:8] = gait
                env.unwrapped._commands[:, 8] = duration
                env.unwrapped._commands[:, 9] = footswing_height_cmd
                env.unwrapped._commands[:, 10] = pitch_cmd
                env.unwrapped._commands[:, 11] = roll_cmd
                env.unwrapped._commands[:, 12] = stance_width_cmd
                env.unwrapped._commands[:, 13] = stance_length_cmd
            elif hasattr(env.unwrapped, "_commands"):
                if _teleop_sock is not None:
                    try:
                        while True:  # 최신 패킷까지 drain
                            _data, _ = _teleop_sock.recvfrom(2048)
                            _teleop_cmd = json.loads(_data.decode())
                    except BlockingIOError:
                        pass
                    except Exception:
                        pass
                    env.unwrapped._commands[:, 0] = float(_teleop_cmd.get("v", 0.0))
                    env.unwrapped._commands[:, 1] = float(_teleop_cmd.get("vy", 0.0))
                    env.unwrapped._commands[:, 2] = float(_teleop_cmd.get("w", 0.0))
                else:
                    # command 기반 task: 기본 전진 명령
                    env.unwrapped._commands[:, 0] = 1.0
                    env.unwrapped._commands[:, 1] = 0.0
                    env.unwrapped._commands[:, 2] = 0.0
            # Go2Recovery 등 _commands 없는 task는 위 블록 모두 스킵 (no-op)

            if args_cli.task[:10] == "R_Skeleton":
                actions = torch.zeros(env.action_space.shape, device=env.device)
                print("here?")
            else:
                actions = policy(obs)
            obs_history.append(obs["policy"].cpu().numpy().squeeze())
            action_history.append(actions.detach().cpu().numpy().squeeze())
            # print(env.unwrapped._robot.data.root_lin_vel_b)
            # print('FL', actions[:, [0, 4, 8]])
            # print('RL', actions[:, [2, 6, 10]])
            # print('FR', actions[:, [1, 5, 9]])
            # print('RR', actions[:, [3, 7, 11]])

            # Save policy actions at timestep 499
            if timestep == 499:
                save_obs_data_to_csv(obs_history, "sim_obs_data.csv", env.unwrapped.cfg.num_prio_obs)
                save_actions_to_csv(action_history, "sim_action_data.csv")

            # print(obs["policy"])
            # env stepping
            obs, _, dones, _ = env.step(actions)
            if _reward_publisher is not None:
                _reward_publisher.step(env.unwrapped)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if _reward_publisher is not None:
        _reward_publisher.close()
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

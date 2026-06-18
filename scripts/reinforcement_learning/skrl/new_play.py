# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.
Report mode (--report) 사용 시 commands.yaml 로부터 커맨드를 읽어
관절 데이터 / 속도 플롯 / 비디오를 자동 저장합니다.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")

# ── Report mode 인자 ──────────────────────────────────────────────────────────
parser.add_argument(
    "--report",
    action="store_true",
    default=False,
    help="보고서 모드: commands.yaml 커맨드별 관절 데이터·속도 플롯·비디오를 자동 저장합니다.",
)
parser.add_argument(
    "--commands_file",
    type=str,
    default="scripts/reinforcement_learning/skrl/commands.yaml",
    help="report 모드에서 사용할 커맨드 YAML 파일 경로.",
)
parser.add_argument(
    "--report_steps",
    type=int,
    default=500,
    help="report 모드에서 데이터를 수집할 스텝 수 (기본: 500).",
)
parser.add_argument(
    "--task_group",
    type=str,
    default=None,
    help="commands.yaml 내 태스크 그룹 키 (기본: 태스크 이름으로 자동 추론).",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video or args_cli.report:
    args_cli.enable_cameras = True

# ── Report 모드: YAML 로드 및 num_envs 결정 (AppLauncher 전) ─────────────────
_report_commands_dict: dict = {}
_report_command_keys: list = []

if args_cli.report:
    import yaml as _yaml

    with open(args_cli.commands_file, encoding="utf-8") as _f:
        _full_commands = _yaml.safe_load(_f)

    if not _full_commands:
        raise ValueError(f"commands_file '{args_cli.commands_file}' 이 비어 있거나 파싱 불가합니다.")

    # 태스크 그룹 키 자동 추론
    if args_cli.task_group:
        _group_key = args_cli.task_group
    else:
        # "Isaac-Go2-AMP-Direct-v0" → "Go2AMPDirect" 로 변환 후 yaml 키와 매칭
        _task_clean = args_cli.task.replace("Isaac-", "").replace("-", "").split("v")[0]
        _group_key = None
        for _k in _full_commands:
            if _k.lower() in _task_clean.lower():
                _group_key = _k
                break
        if _group_key is None:
            _group_key = list(_full_commands.keys())[0]
            print(f"[WARN] 태스크 그룹 자동 추론 실패 → fallback: '{_group_key}'")

    if _group_key in _full_commands and isinstance(_full_commands[_group_key], dict):
        _report_commands_dict = _full_commands[_group_key]
    else:
        _report_commands_dict = _full_commands

    _report_command_keys = list(_report_commands_dict.keys())
    print(f"[INFO] Report 모드: '{_group_key}' 에서 {len(_report_command_keys)}개 커맨드 로드.")

    # num_envs를 커맨드 수로 강제
    args_cli.num_envs = len(_report_command_keys)
    args_cli.headless = True  # report는 headless로 실행

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import datetime
import importlib.util as _ilu
import os
import pathlib
import random
import time

import gymnasium as gym
import torch

import skrl

# skrl version check disabled: local skrl is 2.0.0, which changed act() signature
# from act(obs, timestep, timesteps) → act(obs, states, *, timestep, timesteps)
# SKRL_VERSION = "1.4.3"
# if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
#     skrl.logger.error(...)
#     exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── play_utils 동적 임포트 (report 모드용) ─────────────────────────────────────
_PLAY_UTILS_DIR = pathlib.Path(__file__).resolve().parent.parent / "rsl_rl" / "play_utils"


def _load_play_util(name: str, filename: str):
    spec = _ilu.spec_from_file_location(name, _PLAY_UTILS_DIR / filename)
    if spec is None:
        raise ImportError(f"play_utils 모듈 로드 실패: {_PLAY_UTILS_DIR / filename}")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# PLACEHOLDER: Extension template (do not remove this comment)

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent. --report 플래그 시 커맨드별 보고서 자동 생성."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # ── 체크포인트 탐색 ───────────────────────────────────────────────────────
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))
    env_cfg.log_dir = log_dir

    # ── 환경 생성 ─────────────────────────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if (args_cli.video or args_cli.report) else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # ── gym_env 참조 저장 (skrl wrapping 전) ─────────────────────────────────
    # ReportMultiDataRecorder 가 env.unwrapped.로봇 속성에 접근하기 위해 필요
    gym_env = env

    # Go2SkrlAmpEnv 는 'robot' 속성을 사용하지만 recorder 는 '_robot'을 탐색 → alias 추가
    _base_env = gym_env.unwrapped
    if hasattr(_base_env, "robot") and not hasattr(_base_env, "_robot"):
        _base_env._robot = _base_env.robot

    # 일반 play 모드 비디오 래퍼
    if args_cli.video and not args_cli.report:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # ── Runner / Agent 초기화 ─────────────────────────────────────────────────
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"]["write_interval"] = 0
    experiment_cfg["agent"]["experiment"]["checkpoint_interval"] = 0
    runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    runner.agent.enable_training_mode(False)

    # ═══════════════════════════════════════════════════════════════════════════
    # Report 모드
    # ═══════════════════════════════════════════════════════════════════════════
    if args_cli.report:
        _run_report_mode(
            env=env,
            gym_env=gym_env,
            runner=runner,
            base_env=_base_env,
            log_dir=log_dir,
            task_name=task_name,
            dt=dt,
        )
        env.close()
        return

    # ═══════════════════════════════════════════════════════════════════════════
    # 일반 Play 모드
    # ═══════════════════════════════════════════════════════════════════════════
    obs, _ = env.reset()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            # skrl 2.0.0: act(observations, states, *, timestep, timesteps) — states 인수 추가됨
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            obs, _, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


def _run_report_mode(env, gym_env, runner, base_env, log_dir: str, task_name: str, dt: float):
    """커맨드별 보행 데이터를 수집하고 플롯/비디오를 저장합니다."""

    # ── play_utils 임포트 ─────────────────────────────────────────────────────
    _recorder_mod = _load_play_util("play_utils.report_data_recorder", "report_data_recorder.py")
    ReportMultiDataRecorder = _recorder_mod.ReportMultiDataRecorder

    num_envs = args_cli.num_envs
    report_steps = args_cli.report_steps

    # ── 커맨드 텐서 빌드 ─────────────────────────────────────────────────────
    command_tensor_list = []
    env_labels = []

    for key in _report_command_keys:
        cfg = _report_commands_dict[key]
        vx = float(cfg.get("lin_vel_x", 0.0))
        vy = float(cfg.get("lin_vel_y", 0.0))
        wz = float(cfg.get("yaw_vel", 0.0))
        command_tensor_list.append([vx, vy, wz])
        env_labels.append(f"vx{vx}_vy{vy}_wz{wz}")

    c_tensor = torch.tensor(command_tensor_list, dtype=torch.float32, device=base_env.device)
    n_cmd_dim = min(c_tensor.shape[1], base_env._commands.shape[1])
    c_aligned = c_tensor[:, :n_cmd_dim]

    # ── 커맨드 리샘플링 비활성화 ─────────────────────────────────────────────
    def _dummy_resample(env_ids):
        pass

    if hasattr(base_env, "_resample_commands"):
        base_env._resample_commands = _dummy_resample

    # ── Data Recorder 초기화 ─────────────────────────────────────────────────
    load_run_name = os.path.basename(os.path.normpath(log_dir))
    data_recorder = ReportMultiDataRecorder(
        results_root=os.path.join(os.getcwd(), "results"),
        task_name=task_name,
        load_run_name=load_run_name,
        num_envs=num_envs,
        max_steps=report_steps,
    )

    # ── 환경 초기화 및 커맨드 주입 ───────────────────────────────────────────
    base_env._commands[:, :n_cmd_dim] = c_aligned
    obs, _ = env.reset()
    base_env._commands[:, :n_cmd_dim] = c_aligned  # reset 후 재주입

    print(f"[INFO] Report 루프 시작 ({report_steps} 스텝). 커맨드:")
    for i, label in enumerate(env_labels):
        print(f"  env_{i}: {label}")

    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    video_frames: list = []
    timestep = 0

    while simulation_app.is_running():
        if timestep % 50 == 0:
            print(f"  Report Step: {timestep}/{report_steps}", flush=True)

        with torch.inference_mode():
            # 커맨드 매 스텝 재주입 (reset 등으로 덮어씌워지지 않도록)
            base_env._commands[:, :n_cmd_dim] = c_aligned

            # _processed_actions 업데이트 (recorder 가 저장하도록)
            base_env._processed_actions = (
                base_env.cfg.action_scale * base_env.actions + base_env.robot.data.default_joint_pos
            )

            # 에이전트 추론 (skrl 2.0.0 API)
            outputs = runner.agent.act(obs, env.state(), timestep=0, timesteps=0)
            if hasattr(env, "possible_agents"):
                actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])

            # 데이터 수집
            if not data_recorder.is_full:
                data_recorder.record(gym_env)

            # 비디오 프레임 캡처
            if len(video_frames) < report_steps:
                frame = gym_env.render()
                if frame is not None:
                    video_frames.append(frame[0] if isinstance(frame, list) else frame)

            # 환경 스텝
            obs, _, dones, _, _ = env.step(actions)

            # 에피소드 리셋 시 커맨드 재주입
            if dones.any():
                base_env._commands[:, :n_cmd_dim] = c_aligned

        timestep += 1

        # 데이터 수집 완료 시 저장 후 종료
        if data_recorder.is_full and not data_recorder.is_saved:
            joint_names = list(base_env.robot.data.joint_names)

            # 비디오 저장
            video_src = None
            if len(video_frames) > 0:
                video_dir = pathlib.Path(log_dir) / "videos" / "report"
                video_dir.mkdir(parents=True, exist_ok=True)
                video_src = video_dir / f"report_{timestamp_str}.mp4"
                try:
                    import imageio

                    imageio.mimsave(str(video_src), video_frames, fps=max(1, int(1.0 / dt)))
                    print(f"[INFO] 비디오 저장 완료: {video_src}")
                except Exception as _e:
                    print(f"[WARN] 비디오 저장 실패: {_e}")
                    video_src = None

            save_dir = data_recorder.save(
                command_labels=env_labels,
                timestamp_str=timestamp_str,
                joint_names=joint_names,
                video_src=video_src,
            )
            print(f"[INFO] Report 저장 완료: {save_dir}")
            break

    print("[INFO] Report 모드 종료.")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

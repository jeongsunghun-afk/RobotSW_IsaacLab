# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
play_interactive.py
===================
play.py 를 기반으로, **시뮬레이션 실행 중 터미널에서 보행 커맨드를 실시간으로 입력**할 수 있는
인터랙티브 Inference 스크립트.

주요 기능:
- 별도 스레드에서 터미널 입력을 받아 시뮬레이션 루프를 블로킹하지 않음
- Go2WTW (14개 커맨드): <gait> <x_vel> <y_vel> <yaw_vel> 형식
- R_Skeleton 등 (3개 커맨드): <x_vel> <y_vel> <yaw_vel> 형식
- 매 스텝마다 커맨드를 환경에 강제 적용하여 내부 resampling 방지
- 커맨드 변경 시 환경 전체 리셋

사용 예시:
    python play_interactive.py --task Go2WTW --num_envs 1
    python play_interactive.py --task R_Skeleton_AMP-Play --num_envs 1
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="RSL-RL 인터랙티브 Inference 스크립트.")
parser.add_argument("--video", action="store_true", default=False, help="영상 녹화 여부.")
parser.add_argument("--video_length", type=int, default=200, help="녹화 길이 (스텝).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Fabric 비활성화 (USD I/O 사용).",
)
parser.add_argument("--num_envs", type=int, default=None, help="환경 수.")
parser.add_argument("--task", type=str, default=None, help="태스크 이름.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point",
    help="RL 에이전트 설정 엔트리포인트 이름.",
)
parser.add_argument("--seed", type=int, default=None, help="환경 시드.")
parser.add_argument(
    "--use_pretrained_checkpoint", action="store_true",
    help="Nucleus에서 사전학습 체크포인트 사용.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="실시간 평가 모드.")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

# Isaac Sim 앱 실행
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner, OnPolicyRunnerParkour

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
    export_policy_as_onnx,
    export_policy_as_jit_parkour,
    export_policy_as_onnx_parkour,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# ── play_utils 임포트 (절대 파일 경로 기반 로드 — sys.path 무관) ─────────────
import importlib.util as _ilu
import inspect as _inspect
import pathlib as _pl

_THIS_FILE = _pl.Path(
    __file__ if (__file__ is not None and _pl.Path(__file__).exists())
    else _inspect.getfile(_inspect.currentframe())
).resolve()
_PLAY_UTILS_DIR = _THIS_FILE.parent / "play_utils"


def _load_module(name: str, filename: str):
    """절대 파일 경로에서 모듈을 로드 (sys.path 무관)."""
    abs_path = _PLAY_UTILS_DIR / filename
    spec = _ilu.spec_from_file_location(name, abs_path)
    if spec is None:
        raise ImportError(f"[play_interactive] 모듈 로드 실패: {abs_path}")
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_csv_mod      = _load_module("play_utils.csv_utils",      "csv_utils.py")
_env_mod      = _load_module("play_utils.env_utils",      "env_utils.py")
_gait_mod     = _load_module("play_utils.gait_commands",  "gait_commands.py")
_terminal_mod = _load_module("play_utils.terminal_input", "terminal_input.py")
_data_recorder_mod = _load_module("play_utils.data_recorder", "data_recorder.py")

save_obs_data_to_csv       = _csv_mod.save_obs_data_to_csv
save_actions_to_csv        = _csv_mod.save_actions_to_csv
print_action_joint_mapping = _env_mod.print_action_joint_mapping
get_env_command_info       = _env_mod.get_env_command_info
get_env_interaction_info   = _env_mod.get_env_interaction_info
build_go2wtw_command       = _gait_mod.build_go2wtw_command
TerminalCommandInput       = _terminal_mod.TerminalCommandInput
DataRecorder               = _data_recorder_mod.DataRecorder


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """인터랙티브 Inference 메인 함수."""

    # ── 설정 ─────────────────────────────────────────────────────────
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── 체크포인트 경로 ──────────────────────────────────────────────
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] 실험 디렉토리 로드: {log_root_path}")

    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] 이 태스크에 대한 사전학습 체크포인트가 없습니다.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    print(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ── 환경 생성 ────────────────────────────────────────────────────
    env_cfg.debug_vis = True
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    env.unwrapped.set_debug_vis(getattr(env_cfg, "debug_vis", True))

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] 영상 녹화 활성화.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # 액션 매핑 출력
    print_action_joint_mapping(env)

    # RSL-RL 래퍼
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 모델 로드 ────────────────────────────────────────────────────
    print(f"[INFO] 체크포인트 로드: {resume_path}")
    print(agent_cfg.class_name)

    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "OnPolicyRunnerParkour":
        runner = OnPolicyRunnerParkour(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"지원하지 않는 runner 클래스: {agent_cfg.class_name}")

    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ── 정규화기 추출 ────────────────────────────────────────────────
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # ── 모델 내보내기 ────────────────────────────────────────────────
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if agent_cfg.class_name == "OnPolicyRunnerParkour":
        export_policy_as_jit_parkour(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx_parkour(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    else:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    # ── 커맨드 설정 ──────────────────────────────────────────────────
    labels, ranges = get_env_command_info(env)
    num_commands = len(labels)

    # Interaction 커맨드 감지 (Go2Interaction 등)
    has_interaction, num_motions, motion_labels = get_env_interaction_info(env)

    # 기본 초기 커맨드 값 (환경 현재 커맨드 읽기)
    try:
        init_cmds = env.unwrapped._commands[0].tolist()
    except Exception:
        init_cmds = [0.0] * num_commands

    # 초기 interaction 커맨드 읽기
    try:
        init_interaction_cmd = int(env.unwrapped._interaction_command[0].item()) if has_interaction else 0
    except Exception:
        init_interaction_cmd = 0

    # thread-safe 공유 상태
    command_state: dict = {
        "values": list(init_cmds),
        "interaction_cmd": init_interaction_cmd,
        "reset_requested": False,
        "quit_requested": False,
    }

    if has_interaction:
        command_state["cmd_str"] = motion_labels[init_interaction_cmd] if init_interaction_cmd < len(motion_labels) else f"motion_{init_interaction_cmd}"
    elif num_commands == 3:
        command_state["cmd_str"] = f"vel_{init_cmds[0]:.2f}_{init_cmds[1]:.2f}_{init_cmds[2]:.2f}"
    elif num_commands > 3:
        command_state["cmd_str"] = f"gait_{init_cmds[0]:.2f}_{init_cmds[1]:.2f}_{init_cmds[2]:.2f}"
    else:
        command_state["cmd_str"] = "default"

    if has_interaction:
        print(f"[INFO] Interaction 커맨드 환경 감지: {num_motions}개 모션")
        for i, label in enumerate(motion_labels):
            print(f"  [{i}] {label}")
    else:
        print(f"[INFO] 커맨드 차원: {num_commands}개")
        for i, (lbl, rng) in enumerate(zip(labels, ranges)):
            print(f"  [{i:02d}] {lbl}: {rng}")

    # 터미널 입력 핸들러 시작
    TerminalCommandInput(
        command_state=command_state,
        num_commands=num_commands,
        gait_mod=_gait_mod,
        interaction_mode=has_interaction,
        num_motions=num_motions,
        motion_labels=motion_labels,
    )

    # ── 데이터 저장기 (DataRecorder) ─────────────────────────────────
    # log_dir 끝 두 폴더 (load_run) 추출 (예: logs/rsl_rl/skeleton_fixed_direct/2026-02-24_11-03-50 -> 2026-02-24_11-03-50)
    load_run_name = os.path.basename(os.path.normpath(log_dir))
    data_recorder = DataRecorder(
        results_root=os.path.join(os.getcwd(), "results"),
        task_name=args_cli.task,
        load_run_name=load_run_name,
        max_steps=500,
    )
    if args_cli.save_data:
        print("[INFO] --save_data 활성화. 매 500스텝마다 커맨드 단위 데이터를 results에 저장합니다.")

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────
    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    obs_history: list = []
    action_history: list = []

    while simulation_app.is_running():

        # quit 요청 시 루프 탈출
        if command_state.get("quit_requested", False):
            print("[INFO] 사용자 quit 요청으로 종료합니다.")
            break

        start_time = time.time()

        with torch.inference_mode():

            # ── 매 스텝 커맨드 강제 적용 (내부 resampling 방지) ────
            if num_commands > 0 and not has_interaction:
                current_values = command_state["values"]
                n = min(len(current_values), env.unwrapped._commands.shape[1])
                env.unwrapped._commands[:, :n] = torch.tensor(
                    current_values[:n],
                    dtype=torch.float32,
                    device=env.unwrapped.device,
                )

            # ── Interaction 커맨드 강제 적용 ─────────────────────────
            if has_interaction:
                cur_icmd = int(command_state["interaction_cmd"])
                env.unwrapped._interaction_command[:] = torch.tensor(
                    cur_icmd, dtype=torch.long, device=env.unwrapped.device
                )

            # ── 커맨드 변경 시 환경 리셋 ────────────────────────────
            if command_state["reset_requested"]:
                data_recorder.reset()
                all_env_ids = torch.arange(env.unwrapped.num_envs, device=env.unwrapped.device)
                env.unwrapped._reset_idx(all_env_ids)
                # 리셋 이후에도 커맨드 재적용 (resampling 덮어쓰기)
                if num_commands > 0 and not has_interaction:
                    env.unwrapped._commands[:, :n] = torch.tensor(
                        current_values[:n],
                        dtype=torch.float32,
                        device=env.unwrapped.device,
                    )
                if has_interaction:
                    env.unwrapped._interaction_command[:] = torch.tensor(
                        cur_icmd, dtype=torch.long, device=env.unwrapped.device
                    )
                obs = env.get_observations()
                policy_nn.reset(torch.ones(env.unwrapped.num_envs, dtype=torch.bool))
                command_state["reset_requested"] = False

            # ── 추론 ────────────────────────────────────────────────
            actions = policy(obs)

            obs_history.append(obs["policy"].cpu().numpy().squeeze())
            action_history.append(actions.detach().cpu().numpy().squeeze())

            # DataRecorder 수집 및 완료 시 저장
            if args_cli.save_data and not data_recorder.is_full:
                data_recorder.record(env)
                if data_recorder.is_full and not data_recorder.is_saved:
                    joint_names = env.unwrapped._robot.data.joint_names
                    video_src = None
                    if args_cli.video:
                        video_src = os.path.join(log_dir, "videos", "play", "rl-video-step-0.mp4")
                    data_recorder.save(
                        command_label=command_state.get("cmd_str", "default"),
                        joint_names=joint_names,
                        video_src=video_src,
                    )

            # 500 스텝 시 구형 CSV 저장 (이전 기능 대비)
            if timestep == 499:
                num_obs = getattr(env_cfg, "num_prio_obs", obs["policy"].shape[-1])
                save_obs_data_to_csv(obs_history, "sim_obs_data.csv", num_obs)
                save_actions_to_csv(action_history, "sim_action_data.csv")

            obs, _, dones, _ = env.step(actions)
            # 에피소드 종료 시 policy hidden state 리셋 + 커맨드 재적용
            if dones.any():
                policy_nn.reset(dones)
                if num_commands > 0 and not has_interaction:
                    env.unwrapped._commands[:, :n] = torch.tensor(
                        current_values[:n],
                        dtype=torch.float32,
                        device=env.unwrapped.device,
                    )
                if has_interaction:
                    env.unwrapped._interaction_command[:] = torch.tensor(
                        cur_icmd, dtype=torch.long, device=env.unwrapped.device
                    )

        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break

        # 실시간 모드 슬립
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

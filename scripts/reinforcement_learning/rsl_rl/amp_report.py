# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
amp_report.py
=========
Discriminator가 학습에 사용하는 실제 시뮬레이션의 관측치(Sim Obs)와 
모션 데이터셋에서 가져온 참조 변환 관측치(Ref Obs)의 
형태(Shape)와 각 차원(Dimension)별 값을 1:1로 직접 비교하기 위한 디버깅 스크립트.
"""

import argparse
import sys
import os
import datetime
import csv
import torch
import numpy as np
import yaml

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

# ── argparse ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="AMP Discriminator 관측치 검증용 스크립트.")
parser.add_argument(
    "--commands_file", type=str, default="scripts/reinforcement_learning/rsl_rl/commands.yaml",
    help="평가할 커맨드들을 정의한 YAML 파일의 위치."
)
parser.add_argument("--num_envs", type=int, default=None, help="환경 수.")
parser.add_argument("--task", type=str, default=None, help="태스크 이름.")
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point", help="RL 에이전트 설정.")
parser.add_argument("--seed", type=int, default=1234, help="환경 시드.")
parser.add_argument("--use_pretrained_checkpoint", action="store_true")
parser.add_argument("--wbc", action="store_true", default=False)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False,
    help="Fabric 비활성화 (USD I/O 사용).",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True
args_cli.enable_cameras = False

# YAML 파싱 (env 갯수 확보용)
with open(args_cli.commands_file, "r", encoding="utf-8") as f:
    full_commands_dict = yaml.safe_load(f)

if not full_commands_dict:
    raise ValueError(f"입력 파일 {args_cli.commands_file} 이 비어 있거나 파싱할 수 없습니다.")

task_group = args_cli.task.split("-")[0]
if task_group in full_commands_dict and isinstance(full_commands_dict[task_group], dict):
    commands_dict = full_commands_dict[task_group]
else:
    if "env_0" in full_commands_dict:
        commands_dict = full_commands_dict
    else:
        first_key = list(full_commands_dict.keys())[0]
        commands_dict = full_commands_dict[first_key]

args_cli.num_envs = len(commands_dict.keys())
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym

from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMP
from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    if hasattr(args_cli, "wbc") and hasattr(env_cfg, "whole_body_control"):
        env_cfg.whole_body_control = args_cli.wbc
        if hasattr(env_cfg, "__post_init__"):
            env_cfg.__post_init__()

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunnerAMP(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # 비교 시작 전, 항상 모션 시간 t=0에서 시작격으로 맞추기 위해 강제 리셋 (원할 경우)
    env.unwrapped.cfg.reset_strategy = "random_start"
    obs, _ = env.reset()
    
    ref_duration = env.unwrapped._motion_loader.duration
    step_dt = env.unwrapped.step_dt
    max_steps = int(ref_duration / step_dt)
    
    print("\n" + "="*80)
    print(f"[INFO] AMP 관측치 추출 시작... (최대 레퍼런스 모션 길이: {ref_duration:.3f}초)")
    print(f"[INFO] 시뮬레이션 dt: {step_dt:.3f}초 -> 총 {max_steps} 스텝 추출 예정")
    
    # buffer 형태가 [num_envs, num_amp_observations * amp_observation_space] 로 펼쳐져있음
    if hasattr(env.unwrapped.cfg, "amp_observation_space"):
        amp_dim = env.unwrapped.cfg.amp_observation_space
    else:
        amp_dim = 99
        
    # ── 관측 라벨링 생성 ──
    try:
        joint_names = list(env.unwrapped._robot.data.joint_names)
        key_body_names = env.unwrapped.KEY_BODY_NAMES
    except AttributeError:
        # Fallback for generic dicts/lists
        joint_names = [f"joint_{i}" for i in range(34)]
        key_body_names = ["FL", "FR", "HL", "HR"]
    
    labels = []
    for j in joint_names: labels.append(f"dof_pos_{j}")
    for j in joint_names: labels.append(f"dof_vel_{j}")
    labels.append("root_pos_z(height)")
    labels.extend(["root_lin_vel_x", "root_lin_vel_y", "root_lin_vel_z"])
    labels.extend(["root_ang_vel_x", "root_ang_vel_y", "root_ang_vel_z"])
    for kb in key_body_names:
        labels.extend([f"key_pos_{kb}_x", f"key_pos_{kb}_y", f"key_pos_{kb}_z"])
    for kb in key_body_names:
        labels.extend([f"key_vel_{kb}_x", f"key_vel_{kb}_y", f"key_vel_{kb}_z"])
        
    if len(labels) != amp_dim:
        print(f"[WARN] 생성된 라벨 개수({len(labels)})가 AMP 차원({amp_dim})과 다릅니다!")
        if amp_dim > len(labels):
            for i in range(len(labels), amp_dim):
                labels.append(f"unknown_dim_{i}")
        else:
            labels = labels[:amp_dim]
            
    # CSV 저장 준비
    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"amp_obs_comparison_{timestamp_str}.csv")
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Step", "CurrentTime(s)", "Index", "Feature_Label", "Sim_Obs_Value", "Ref_Obs_Value(Expert)", "Abs_Diff"])
        
        all_diffs = []

        for step in range(max_steps):
            current_time = step * step_dt
            
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, extras = env.step(actions)
                
            sim_amp_obs_full = extras.get("amp_obs")
            if sim_amp_obs_full is None:
                print("[ERROR] 환경에서 'amp_obs'를 반환하지 않습니다.")
                break
                
            # Expert 참조 모션 매칭 (현재 시간 기준 강제)
            current_times_array = np.array([current_time] * args_cli.num_envs)
            ref_amp_obs_full = env.unwrapped.collect_reference_motions(args_cli.num_envs, current_times=current_times_array)

            sim_amp_obs = sim_amp_obs_full[0, :amp_dim].cpu().numpy()
            ref_amp_obs = ref_amp_obs_full[0, :amp_dim].cpu().numpy()
            
            diffs = np.abs(sim_amp_obs - ref_amp_obs)
            all_diffs.append(diffs)
            
            for i in range(amp_dim):
                s_val = float(sim_amp_obs[i])
                r_val = float(ref_amp_obs[i])
                diff = diffs[i]
                writer.writerow([step, f"{current_time:.3f}", i, labels[i], f"{s_val:.6f}", f"{r_val:.6f}", f"{diff:.6f}"])
            
            if step % 50 == 0 or step == max_steps - 1:
                print(f"[INFO] 진행 중... {step+1}/{max_steps} (시간: {current_time:.3f}s)")

    print("="*80)
    print(f"\n[SUCCESS] 특성들을 매칭하여 CSV 분석 리포트로 저장했습니다.")
    print(f" 저장경로: {csv_path}\n")
    
    # 평균 오차 기준 가장 차이가 큰 특성 Top 10
    if len(all_diffs) > 0:
        mean_diffs = np.mean(all_diffs, axis=0)
        top_diff_indices = np.argsort(mean_diffs)[::-1][:10]
        print("\n[INFO] 가장 큰 평균 차이를 보이는 특성 Top 10 (시퀀스 전체 평균):")
        for idx in top_diff_indices:
            print(f"  - [{idx:2d}] {labels[idx]:<30} | Mean Diff: {mean_diffs[idx]:.4f}")

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
        env = None

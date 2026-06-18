# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
import pickle
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from datetime import datetime

import cv2
import genesis as gs
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from interaction.interaction_env import InteractionEnv
from rsl_rl.runners import OnPolicyRunnerParkour
from tqdm import tqdm


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    # xyz = a[:, :3]
    xyz = a[:, 1:4]
    t = xyz.cross(b, dim=-1) * 2
    # return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)
    return (b + a[:, :1] * t + xyz.cross(t, dim=-1)).view(shape)


def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)


def save_video(frames, save_path, fps=30):
    if not frames:
        return

    # numpy / uint8 정리
    processed = []
    for f in frames:
        arr = np.asarray(f)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        processed.append(arr)

    imageio.mimsave(
        save_path,
        processed,
        fps=fps,
        codec="libx264",  # H.264 (브라우저/VSCode 친화적)
        quality=8,  # 0~10 (10이 제일 좋음)
    )


def save_depth_video(frames, save_path, fps=30, colormap=True):
    if not frames:
        return

    # Depth 값의 최소/최대 범위를 가져옴
    min_depth = np.min([np.min(frame) for frame in frames])
    max_depth = np.max([np.max(frame) for frame in frames])

    height, width = frames[0].shape  # Depth 이미지는 단일 채널 (H, W)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    for frame in frames:
        # 1️⃣ Depth 값을 0~255 범위로 정규화
        depth_norm = (frame - min_depth) / (max_depth - min_depth)  # 0~1로 정규화
        depth_8bit = (depth_norm * 255).astype(np.uint8)  # 8비트 변환

        # 2️⃣ 컬러맵 적용 (선택 사항)
        if colormap:
            depth_8bit = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # 3️⃣ 비디오 파일에 저장
        out.write(depth_8bit)

    out.release()


def create_video_directory(base_dir="videos", cmd=None, args=None, terrain_dict=None):
    date_folder = datetime.now().strftime("%Y-%m-%d")
    exp_folder = args.exp_name if args and args.exp_name else "default_exp"
    time_folder = datetime.now().strftime("%H-%M-%S")

    if cmd is not None:
        time_folder += f"_{cmd}"

    if terrain_dict is not None:
        key_with_value_1 = [key for key, value in terrain_dict.items() if value == 1.0]
        time_folder += f"_{args.ckpt}_{key_with_value_1[0]}_{args.difficulty}"

    dir_path = os.path.join(base_dir, date_folder, exp_folder, time_folder)
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def save_dof_visualizations(pos_history, torque_history, joint_names, save_path):
    """
    Saves plots of DOF positions and torques over time.

    Parameters:
        pos_history (list of np.array): List of 1D arrays, each representing joint positions at a time step.
        torque_history (list of np.array): List of 1D arrays, each representing joint torques at a time step.
        joint_names (list of str): Names of the joints.
        save_path (str): Full path to save the combined plot image.
    """
    if not pos_history or not torque_history:
        print("Warning: DOF history is empty. Skipping plot generation.")
        return

    pos_data = np.array(pos_history)  # Shape: (num_timesteps, num_joints)
    torque_data = np.array(torque_history)  # Shape: (num_timesteps, num_joints)

    if pos_data.ndim == 1:  # Handle case of single DOF
        pos_data = pos_data.reshape(-1, 1)
    if torque_data.ndim == 1:
        torque_data = torque_data.reshape(-1, 1)

    num_timesteps, num_joints = pos_data.shape

    if num_joints != len(joint_names):
        print(
            f"Warning: Mismatch between number of joints in data ({num_joints}) and number of joint names ({len(joint_names)}). Using generic names for plotting."
        )
        joint_names = [f"DOF_{j}" for j in range(num_joints)]

    if num_joints == 0:
        print("Warning: No DOF data to plot.")
        return

    time_steps = np.arange(num_timesteps)

    # Determine the layout: ceil(num_joints / 2) rows, 4 columns
    # Each row: [Joint_A_Pos, Joint_A_Torque, Joint_B_Pos, Joint_B_Torque]
    num_plot_rows = (num_joints + 1) // 2  # Ceiling division

    # Adjust figsize. If one (pos,torque) pair was roughly (7.5w, 3h) per plot,
    # now 4 plots per row. (e.g., 8w per plot, 4h per plot row)
    plot_width_per_metric = 15  # Approximate width for one metric (pos or torque)
    plot_height_per_row = 4
    fig_width = plot_width_per_metric * 4  # 4 plots horizontally
    fig_height = plot_height_per_row * num_plot_rows

    fig, fig_axs = plt.subplots(num_plot_rows, 4, figsize=(fig_width, fig_height), sharex=True)

    # Ensure fig_axs is always 2D for consistent indexing, even if num_plot_rows is 1.
    # plt.subplots with ncols > 1 and nrows = 1 returns a 1D array.
    if num_plot_rows == 1:
        current_axs = fig_axs.reshape(1, 4)
    else:
        current_axs = fig_axs

    for i in range(num_joints):
        plot_row = i // 2
        # Determine if this is the first (left) or second (right) joint pair in the current row
        is_second_joint_in_row_pair = (i % 2) == 1

        pos_plot_col = 0 if not is_second_joint_in_row_pair else 2
        torque_plot_col = 1 if not is_second_joint_in_row_pair else 3

        # Position plot
        ax_pos = current_axs[plot_row, pos_plot_col]
        ax_pos.plot(time_steps, pos_data[:, i], color="blue")
        ax_pos.set_title(f"{joint_names[i]} - Position")
        ax_pos.set_ylabel("Position (rad)")
        ax_pos.grid(True)

        # Torque plot
        ax_torque = current_axs[plot_row, torque_plot_col]
        ax_torque.plot(time_steps, torque_data[:, i], color="red")
        ax_torque.set_title(f"{joint_names[i]} - Torque")
        ax_torque.set_ylabel("Torque (Nm)")
        ax_torque.grid(True)

        if plot_row == num_plot_rows - 1:  # Add x-label to plots in the last data-containing row
            ax_pos.set_xlabel("Time Step")
            ax_torque.set_xlabel("Time Step")

    # If the number of joints is odd, hide the unused subplots in the last row
    if num_joints % 2 == 1 and num_plot_rows > 0:
        current_axs[num_plot_rows - 1, 2].axis("off")  # Hide the placeholder for the second joint's position
        current_axs[num_plot_rows - 1, 3].axis("off")  # Hide the placeholder for the second joint's torque

    fig.suptitle("DOF Joint Positions and Torques Over Time", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    try:
        plt.savefig(save_path)
        print(f"Saved DOF visualization to {save_path}")
    except Exception as e:
        print(f"Error saving DOF visualization: {e}")
    finally:
        plt.close(fig)


def load_policy(logdir, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = torch.jit.load(logdir + "/actor.jit", map_location=device)
    history_module = torch.jit.load(logdir + "/history_module.jit", map_location=device)
    priv_module = torch.jit.load(logdir + "/priv_module.jit", map_location=device)
    # estimation_module = torch.jit.load(logdir + 'estimation_module.jit', map_location=device)

    def policy(obs):
        obs_prop_scan = obs[:, : cfg["num_prio_obs"] + cfg["num_heights"]]
        obs_priv_explicit = obs[
            :, cfg["num_prio_obs"] + cfg["num_heights"] : cfg["num_prio_obs"] + cfg["num_heights"] + cfg["num_priv"]
        ]
        if cfg["history_encoding"]:
            hist = obs[:, -cfg["history_len"] * cfg["num_prio_obs"] :]
            latent = history_module.forward(hist.view(-1, cfg["history_len"], cfg["num_prio_obs"]))
        else:
            priv = obs[
                :,
                cfg["num_prio_obs"] + cfg["num_heights"] + cfg["num_priv"] : cfg["num_prio_obs"]
                + cfg["num_heights"]
                + cfg["num_priv"]
                + cfg["num_priv_latent"],
            ]
            latent = priv_module.forward(priv)
        backbone_input = torch.cat([obs_prop_scan, obs_priv_explicit, latent], dim=1)
        action = actor.forward(backbone_input)
        return action

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="interaction")
    parser.add_argument("--ckpt", type=int, default=500)
    parser.add_argument("--task", type=int, default=7)
    args = parser.parse_args()

    gs.init(logging_level="warning")

    log_dir = f"logs/interaction/{args.exp_name}"
    configs = pickle.load(open(f"logs/interaction/{args.exp_name}/cfgs.pkl", "rb"))
    if len(configs) == 6:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg, _ = configs
    else:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = configs
    # reward_cfg["reward_scales"] = {}
    # env_cfg["base_init_pos"] = [0., 0., 0.55]
    env_cfg["left_dof_names"] = [
        "FL_joint1_withers_p",
        "FL_joint2_shoulder_p",
        "FL_joint3_shoulder_r",
        "FL_joint4_elbow_p",
        "FL_joint5_wrist_r",
        "FL_joint6_wrist_p",
        "FL_joint7_toe_p",
        "HL_joint1_thigh_r",
        "HL_joint2_thigh_p",
        "HL_joint3_knee_p",
        "HL_joint4_ankle_r",
        "HL_joint5_ankle_p",
        "HL_joint6_toe_p",
    ]
    env_cfg["right_dof_names"] = [
        "FR_joint1_withers_p",
        "FR_joint2_shoulder_p",
        "FR_joint3_shoulder_r",
        "FR_joint4_elbow_p",
        "FR_joint5_wrist_r",
        "FR_joint6_wrist_p",
        "FR_joint7_toe_p",
        "HR_joint1_thigh_r",
        "HR_joint2_thigh_p",
        "HR_joint3_knee_p",
        "HR_joint4_ankle_r",
        "HR_joint5_ankle_p",
        "HR_joint6_toe_p",
    ]
    env_cfg["hip_link_names"] = ["shoulder"]
    env = InteractionEnv(
        num_envs=2,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=False,
    )

    print(f"Camera status: env.cam = {env.cam}, env.view_cam = {env.view_cam}")

    runner = OnPolicyRunnerParkour(env, train_cfg, log_dir, device="cuda:0")
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device="cuda:0")

    # Get actuated joint names for plotting titles from env_cfg
    # This assumes the order in env_cfg["dof_names"] corresponds to the order of data
    # collected via env.motor_dofs, which is typical.
    actuated_joint_names = env_cfg.get("dof_names")

    # Validate and provide fallback if names are not suitable or not found
    if (
        not actuated_joint_names
        or not isinstance(actuated_joint_names, list)
        or len(actuated_joint_names) != env.num_actions
    ):
        print(
            f"Warning: 'dof_names' from env_cfg is not suitable (found: {actuated_joint_names}, expected list of {env.num_actions} names)."
        )
        # Fallback to generic names if env_cfg["dof_names"] is not appropriate
        print("Using generic joint names for plotting.")
        actuated_joint_names = [f"DOF_{j}" for j in range(env.num_actions)]
    else:
        print(f"Using joint names from env_cfg['dof_names'] for plots: {actuated_joint_names}")

    # The save_dof_visualizations function also has a fallback for name mismatches with data.

    interaction_cmds = [0, 1, 2, 3]

    # lin_vel_cmds = [0.5]
    # ang_vel_cmds = [0.5, 1.0]
    for interaction_cmd in interaction_cmds:
        obs, _ = env.reset()

        dof_pos_history = []
        dof_torque_history = []
        # fov_frames = []
        frames = []
        # depth_frames = []

        step_count = 0
        save_interval = 200  # Save every 1000 timesteps
        video_dir = create_video_directory(args=args, cmd=(interaction_cmd))
        if env.cam is not None:
            rgb, _, _, _ = env.view_cam.render()
            frames.append(rgb)
            rgb, depth, _, _ = env.cam.render(depth=True)
            # fov_frames.append(rgb)
            # depth_frames.append(depth)

        with torch.no_grad():
            for t in tqdm(range(save_interval), desc="Simulation Progress"):
                env.interaction_command[:, 0] = interaction_cmd
                actions = policy(obs)
                # env.scene.clear_debug_objects()
                # actions = torch.zeros_like(actions, device=env.device)
                env.scene.clear_debug_objects()
                contact_info = env.robot.get_contacts(env.terrain)
                env._draw_debug_vis()

                obs, _, rews, dones, infos = env.step(actions)

                # Collect DOF data
                current_dof_pos = env.robot.get_dofs_position(dofs_idx_local=env.motor_dofs).cpu().numpy()[0, :]
                current_dof_torque = (
                    env.robot.get_dofs_force(dofs_idx_local=env.motor_dofs).cpu().numpy()[0, :]
                )  # Applied motor torques
                dof_pos_history.append(current_dof_pos)
                dof_torque_history.append(current_dof_torque)

                if env.cam is not None:
                    rgb, _, _, _ = env.view_cam.render()
                    frames.append(rgb)
                    rgb, depth, _, _ = env.cam.render(depth=True)
                    # fov_frames.append(rgb)
                    # depth_frames.append(depth)

            # After the simulation loop for this command pair (one segment)
            # Save videos if cam is available and frames were collected
            if env.cam is not None and frames:  # Check fov_frames as it's tied to env.cam
                # Using simplified names as video_dir is unique per command
                video_path = os.path.join(video_dir, "video.mp4")
                save_video(frames, video_path)
                # fov_video_path = os.path.join(video_dir, "fov_video.mp4")
                # save_video(fov_frames, fov_video_path)
                # depth_video_path = os.path.join(video_dir, "depth_fov_video.mp4")
                # save_depth_video(depth_frames, depth_video_path)

            # Save DOF plots (unconditionally after the segment simulation, if data exists)
            if dof_pos_history:
                plot_save_path = os.path.join(video_dir, "dof_summary.png")
                save_dof_visualizations(dof_pos_history, dof_torque_history, actuated_joint_names, plot_save_path)
            # The original break inside the loop is implicitly handled by the loop range
            # and processing after the loop completes one segment.


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking -v --ckpt 100
"""

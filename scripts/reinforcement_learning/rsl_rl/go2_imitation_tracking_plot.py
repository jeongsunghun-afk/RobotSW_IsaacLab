# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2-Imitation-Tracking 정책의 속도 명령 추종 결과를 플롯하는 스크립트.

MimicKit의 ``run_control_tracking_plot.py``를 IsaacLab/Go2-Imitation-Tracking
(body-frame 속도 추종) 환경에 맞게 이식.

동작:
  - body-frame 속도 명령을 (vx, vy=0, yaw_rate=0)으로 고정하고 vx만 0→max→0으로 ramp+hold sweep.
  - 매 step 정책을 평가하며 실제 body-frame 속도 / joint torque·pos·vel을 수집.
  - 2종 플롯 저장:
      velocity_comparison.png — (vx,vy,vz)+(roll,pitch,yaw rate), 실제=실선/명령=점선.
      joints.png              — joint별 Torque / Position / Velocity (FL/FR/RL/RR×hip,thigh,calf).

정책 로딩 경로는 play.py를 그대로 미러링 (gym.make → RslRlVecEnvWrapper →
OnPolicyRunnerAMPBase.load → get_inference_policy). get_inference_policy가 반환하는
act_inference는 내부에서 actor_obs_normalizer를 적용하므로 obs 정규화가 보존된다.

실행 예시:
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/go2_imitation_tracking_plot.py \
    --task Go2-Imitation-Tracking-v0 --num_envs 1 --headless \
    --load_run 2026-05-22_11-36-30_no_pace --checkpoint model_500.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Plot Go2-Imitation-Tracking velocity command tracking.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Go2-Imitation-Tracking-v0", help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--out_dir", type=str, default=None, help="Output directory for PNGs (default: <run>/results).")
# velocity schedule shaping
parser.add_argument("--max_speed", type=float, default=3.5, help="Peak forward speed of the sweep (m/s).")
parser.add_argument("--speed_step", type=float, default=0.5, help="Speed increment between hold levels (m/s).")
parser.add_argument("--ramp_duration", type=float, default=1.5, help="Ramp duration between hold levels (s).")
parser.add_argument("--hold_duration", type=float, default=4.0, help="Hold duration at each speed level (s).")
parser.add_argument("--settle_duration", type=float, default=2.0, help="Initial speed-0 hold so heading settles (s).")
parser.add_argument("--debug_fast", action="store_true", help="Use a tiny schedule for a quick end-to-end smoke run.")
# video recording
parser.add_argument("--video", action="store_true", default=False, help="녹화 mp4 저장 (sweep 전체).")
parser.add_argument("--video_length", type=int, default=0, help="녹화 프레임 수. 0이면 sweep 전체 길이 자동 사용.")
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

import os

import gymnasium as gym
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from rsl_rl.runners.on_policy_runner_amp import OnPolicyRunnerAMPBase  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

# 플롯의 row 순서 (leg-grouped). IsaacLab 실제 articulation 순서(type-grouped:
# 모든 hip → 모든 thigh → 모든 calf)와 다르므로 런타임 joint_names로 remap한다.
PLOT_DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


# ─────────────────────────────────────────────────────────────────────────────
# Velocity schedule
# ─────────────────────────────────────────────────────────────────────────────


def build_velocity_schedule(dt, max_speed, speed_step, ramp_duration, hold_duration, settle_duration):
    """0→max→0 ramp+hold 형태의 per-step forward 속도 명령을 반환.

    시작에 settle_duration 만큼 speed=0 hold를 둬서 RSI heading이 +x로 정착할 시간을 준다.
    """
    ramp_steps = max(1, int(round(ramp_duration / dt)))
    hold_steps = max(1, int(round(hold_duration / dt)))
    settle_steps = max(1, int(round(settle_duration / dt)))

    levels = list(np.arange(speed_step, max_speed + 1e-6, speed_step))
    sequence = [0.0] + levels + levels[-2::-1] + [0.0] if levels else [0.0]

    commands = [0.0] * settle_steps  # initial settle hold at 0
    prev_speed = sequence[0]
    for target_speed in sequence[1:]:
        for r in range(ramp_steps):
            alpha = (r + 1) / ramp_steps
            commands.append(prev_speed + (target_speed - prev_speed) * alpha)
        commands.extend([target_speed] * hold_steps)
        prev_speed = target_speed

    return np.array(commands, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Data collection
# ─────────────────────────────────────────────────────────────────────────────


def freeze_command(base_env, speed):
    """body-frame 속도 명령을 (vx=speed, vy=0, yaw_rate=0)으로 고정하고 타이머를 무력화."""
    base_env._lin_vel_cmd[:, 0] = speed  # vx command (forward sweep)
    base_env._lin_vel_cmd[:, 1] = 0.0  # vy 항상 0
    base_env._yaw_vel_cmd[:] = 0.0  # yaw rate = 0 (직진/고정 heading)
    base_env._tar_timer[:] = float("inf")  # _post_physics_step resample 방지


def collect_tracking_data(env, base_env, policy, velocity_schedule):
    """sweep을 시뮬레이션하고 속도/관절 데이터를 수집."""
    n = len(velocity_schedule)
    robot = base_env._robot
    num_dofs = robot.data.joint_pos.shape[1]

    body_vel_xyz = np.zeros((n, 3))
    body_ang_xyz = np.zeros((n, 3))
    cmd_lin_x = np.zeros(n)
    dof_pos = np.zeros((n, num_dofs))
    dof_vel = np.zeros((n, num_dofs))
    dof_torque = np.zeros((n, num_dofs))
    root_pos_w = np.zeros((n, 3))

    obs = env.get_observations()

    with torch.inference_mode():
        for t in range(n):
            spd = float(velocity_schedule[t])
            # 명령 freeze (belt-and-suspenders: 매 step 재주입)
            freeze_command(base_env, spd)

            cmd_lin_x[t] = spd
            body_vel_xyz[t] = robot.data.root_lin_vel_b[0].cpu().numpy()
            body_ang_xyz[t] = robot.data.root_ang_vel_b[0].cpu().numpy()
            dof_pos[t] = robot.data.joint_pos[0].cpu().numpy()
            dof_vel[t] = robot.data.joint_vel[0].cpu().numpy()
            dof_torque[t] = robot.data.applied_torque[0].cpu().numpy()
            root_pos_w[t] = robot.data.body_pos_w[0, base_env.ref_body_index].cpu().numpy()

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            if t % 200 == 0:
                print(f"[tracking] step {t}/{n}  cmd_vx={spd:.2f} m/s  vx={body_vel_xyz[t, 0]:.2f}")

    return dict(
        body_vel_xyz=body_vel_xyz,
        body_ang_xyz=body_ang_xyz,
        cmd_lin_x=cmd_lin_x,
        dof_pos=dof_pos,
        dof_vel=dof_vel,
        dof_torque=dof_torque,
        root_pos_w=root_pos_w,
    )


def verify_continuity(root_pos_w, dt, max_speed):
    """텔레포트(리셋) 검출: per-step 위치 점프가 물리적으로 가능한 한계를 넘으면 실패."""
    deltas = np.linalg.norm(np.diff(root_pos_w, axis=0), axis=1)
    # 한 step 최대 이동 = (peak_speed + 여유) * dt. 텔레포트는 RSI 포즈로 수 m 점프.
    max_plausible = (max_speed + 5.0) * dt
    jumps = np.where(deltas > max_plausible)[0]
    if len(jumps) > 0:
        print(
            f"[verify] FAIL: {len(jumps)} teleport(s) detected (max step delta="
            f"{deltas.max():.3f} m > {max_plausible:.3f} m). Auto-reset likely fired."
        )
        return False
    print(f"[verify] OK: root_pos continuous (max step delta={deltas.max():.3f} m <= {max_plausible:.3f} m).")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_velocity_comparison(data, title, out_dir):
    """2행×3열: (vx,vy,vz) + (roll,pitch,yaw rate). 실제=실선, 명령=점선."""
    bv = data["body_vel_xyz"]
    bav = data["body_ang_xyz"]
    n = bv.shape[0]
    steps = np.arange(n)

    # +x 고정 sweep: 명령은 vx 패널에만 점선(lin_vel_cmd[:,0]). 나머지 패널 명령=0.
    cmd_vel = np.stack([data["cmd_lin_x"], np.zeros(n), np.zeros(n)], axis=1)
    cmd_ang = np.zeros((n, 3))

    vel_labels = ["vx (m/s)", "vy (m/s)", "vz (m/s)"]
    ang_labels = ["roll rate (rad/s)", "pitch rate (rad/s)", "yaw rate (rad/s)"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    fig.suptitle(f"Body Frame Velocity — {title}", fontsize=13)

    for col in range(3):
        ax = axes[0, col]
        ax.plot(steps, bv[:, col], color="tab:blue", linewidth=0.9, label="actual")
        ax.plot(steps, cmd_vel[:, col], color="black", linestyle="--", linewidth=0.9, label="command")
        ax.set_ylabel(vel_labels[col], fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        ax2 = axes[1, col]
        ax2.plot(steps, bav[:, col], color="tab:orange", linewidth=0.9, label="actual")
        ax2.plot(steps, cmd_ang[:, col], color="black", linestyle="--", linewidth=0.9, label="command")
        ax2.set_ylabel(ang_labels[col], fontsize=9)
        ax2.set_xlabel("Timestep", fontsize=8)
        ax2.legend(fontsize=7, loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(labelsize=8)

    plt.tight_layout()
    path = os.path.join(out_dir, "velocity_comparison.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Saved: {path}")


def plot_joint_data(data, title, out_dir, plot_to_sim_index, plot_dof_names):
    """12행×3열: joint별 torque / position / velocity (leg-grouped 순서)."""
    n_joints = len(plot_dof_names)
    col_labels = ["Torque (N·m)", "Position (rad)", "Velocity (rad/s)"]
    col_arrays = [data["dof_torque"], data["dof_pos"], data["dof_vel"]]
    col_colors = ["tab:red", "tab:blue", "tab:green"]

    steps = np.arange(col_arrays[0].shape[0])

    fig, axes = plt.subplots(n_joints, 3, figsize=(14, 2.2 * n_joints), sharex=True)
    fig.suptitle(f"Joint Data — {title}", fontsize=13, y=1.002)

    for row, jname in enumerate(plot_dof_names):
        sim_idx = plot_to_sim_index[row]
        for col, (arr, lbl, color) in enumerate(zip(col_arrays, col_labels, col_colors)):
            ax = axes[row, col]
            ax.plot(steps, arr[:, sim_idx], color=color, linewidth=0.8)
            if col == 0:
                ax.set_ylabel(jname, fontsize=7, rotation=0, labelpad=70, va="center")
            if row == 0:
                ax.set_title(lbl, fontsize=9)
            if row == n_joints - 1:
                ax.set_xlabel("Timestep", fontsize=8)
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=6)

    plt.tight_layout()
    path = os.path.join(out_dir, "joints.png")
    plt.savefig(path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run velocity-tracking sweep and save plots."""
    # ── play.py 미러링: cfg 오버라이드 ─────────────────────────────────────
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # ── 체크포인트 경로 해석 (play.py와 동일) ──────────────────────────────
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    # ◆ video 녹화 시 카메라가 robot root를 추적 (origin_type=asset_root → 매 render step 자동 follow)
    if args_cli.video:
        env_cfg.viewer = ViewerCfg(
            origin_type="asset_root",
            asset_name="robot",
            env_index=0,
            eye=(0.0, -2.5, 0.8),
            lookat=(0.0, 0.0, 0.3),
        )

    # ── 환경 생성 (play.py 미러링) ─────────────────────────────────────────
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording (raw env, before RslRlVecEnvWrapper). step_trigger 사용:
    # _get_dones override로 episode 경계가 없으므로 step 0부터 sweep 전체를 한 클립으로 녹화.
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length if args_cli.video_length > 0 else 100000,
            "disable_logger": True,
        }
        print("[INFO] Recording video of the sweep.")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    base_env = env.unwrapped

    # ── 러너/정책 로딩 (play.py 미러링; class_name=OnPolicyRunnerAMPBase) ──
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    print(f"[INFO]: Runner class: {agent_cfg.class_name}")
    if agent_cfg.class_name != "OnPolicyRunnerAMPBase":
        print(f"[WARN] Expected OnPolicyRunnerAMPBase but cfg says {agent_cfg.class_name}; instantiating it anyway.")
    runner = OnPolicyRunnerAMPBase(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)

    # act_inference가 내부에서 actor_obs_normalizer를 적용 (BLOCKER 3 해결).
    policy = runner.get_inference_policy(device=base_env.device)

    # ── BLOCKER 1: 자동 리셋 무력화 (died/time_out 모두 False) ─────────────
    def _no_reset_dones():
        false_t = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        return false_t, false_t.clone()

    base_env._get_dones = _no_reset_dones

    # ── joint 순서 매핑: 런타임 joint_names → leg-grouped 플롯 순서 ─────────
    joint_names = list(base_env._robot.data.joint_names)
    print(f"[INFO] IsaacLab joint order: {joint_names}")
    name_to_sim = {name: i for i, name in enumerate(joint_names)}
    if all(name in name_to_sim for name in PLOT_DOF_NAMES):
        plot_dof_names = PLOT_DOF_NAMES
        plot_to_sim_index = [name_to_sim[name] for name in PLOT_DOF_NAMES]
    else:
        print("[WARN] PLOT_DOF_NAMES not found in runtime joint_names; using raw articulation order.")
        plot_dof_names = joint_names
        plot_to_sim_index = list(range(len(joint_names)))

    # ── 속도 스케줄 ────────────────────────────────────────────────────────
    dt = base_env.step_dt
    if args_cli.debug_fast:
        velocity_schedule = build_velocity_schedule(
            dt, max_speed=2.0, speed_step=0.5, ramp_duration=0.2, hold_duration=0.4, settle_duration=0.4
        )
    else:
        velocity_schedule = build_velocity_schedule(
            dt,
            max_speed=args_cli.max_speed,
            speed_step=args_cli.speed_step,
            ramp_duration=args_cli.ramp_duration,
            hold_duration=args_cli.hold_duration,
            settle_duration=args_cli.settle_duration,
        )
    total_steps = len(velocity_schedule)
    print(f"[INFO] dt={dt:.4f}s  total_steps={total_steps}  total_time={total_steps * dt:.1f}s")

    # ── 출력 디렉토리 ──────────────────────────────────────────────────────
    out_dir = args_cli.out_dir if args_cli.out_dir is not None else os.path.join(log_dir, "results")
    os.makedirs(out_dir, exist_ok=True)

    run_name = os.path.basename(log_dir)
    ckpt_name = os.path.basename(resume_path)
    title = f"{run_name} ({ckpt_name})"

    # ── 수집 + 검증 + 플롯 ─────────────────────────────────────────────────
    data = collect_tracking_data(env, base_env, policy, velocity_schedule)

    continuous = verify_continuity(data["root_pos_w"], dt, args_cli.max_speed)

    plot_velocity_comparison(data, title, out_dir)
    plot_joint_data(data, title, out_dir, plot_to_sim_index, plot_dof_names)

    print(f"\n[INFO] All results saved to: {out_dir}/")
    print(f"[INFO] Continuity check: {'PASS' if continuous else 'FAIL (teleport detected)'}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

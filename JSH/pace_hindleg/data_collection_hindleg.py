# HindLeg PACE sim-to-sim data collection: inject KNOWN GT actuator params, drive a
# per-joint linear chirp, record (dof_pos, des_dof_pos) -> data/hindleg_sim/chirp_data.pt
# Also saves gt.pt (the injected ground truth) next to it for recovery comparison.
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--task", type=str, default="Isaac-Pace-HindLeg-v0")
parser.add_argument("--min_frequency", type=float, default=0.1)
parser.add_argument("--max_frequency", type=float, default=10.0)
parser.add_argument("--duration", type=float, default=20.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from torch import pi
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root


def jtype(name):
    n = name.lower()
    if "hip" in n: return "hip"
    if "thigh" in n: return "thigh"
    if "calf" in n: return "calf"
    if "foot" in n: return "foot"
    return "other"

# Ground-truth actuator params (small-joint scale) per joint type
GT_ARMATURE = {"hip": 0.05, "thigh": 0.04, "calf": 0.02, "foot": 0.01}
GT_VISCOUS  = {"hip": 0.10, "thigh": 0.10, "calf": 0.10, "foot": 0.10}
GT_COULOMB  = {"hip": 0.03, "thigh": 0.03, "calf": 0.03, "foot": 0.03}
GT_BIAS     = {"hip": 0.02, "thigh": 0.02, "calf": 0.02, "foot": 0.02}
GT_DELAY    = 5
# chirp profile per joint type: (center, amplitude) in rad
CHIRP = {"hip": (0.0, 0.15), "thigh": (-0.25, 0.35), "calf": (-0.6, 0.5), "foot": (-0.4, 0.4)}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    dev = env.unwrapped.device
    articulation = env.unwrapped.scene["robot"]
    joint_order = list(articulation.joint_names)
    N = len(joint_order)
    types = [jtype(n) for n in joint_order]
    print("[COLLECT] joint_order:", joint_order)
    print("[COLLECT] joint types:", types)
    joint_ids = torch.tensor([articulation.joint_names.index(n) for n in joint_order], device=dev)

    armature = torch.tensor([[GT_ARMATURE[t] for t in types]], device=dev)
    damping  = torch.tensor([[GT_VISCOUS[t]  for t in types]], device=dev)
    friction = torch.tensor([[GT_COULOMB[t]  for t in types]], device=dev)
    bias     = torch.tensor([[GT_BIAS[t]     for t in types]], device=dev)
    time_lag = torch.tensor([[GT_DELAY]], dtype=torch.int, device=dev)
    env.reset()

    articulation.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=torch.arange(1))
    articulation.data.default_joint_armature[:, joint_ids] = armature
    articulation.write_joint_viscous_friction_coefficient_to_sim(damping, joint_ids=joint_ids, env_ids=torch.arange(1))
    articulation.data.default_joint_viscous_friction_coeff[:, joint_ids] = damping
    articulation.write_joint_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_friction_coeff[:, joint_ids] = friction
    articulation.write_joint_dynamic_friction_coefficient_to_sim(friction, joint_ids=joint_ids, env_ids=torch.tensor([0]))
    articulation.data.default_joint_dynamic_friction_coeff[:, joint_ids] = friction
    for drive_type in articulation.actuators.keys():
        drive_indices = articulation.actuators[drive_type].joint_indices
        if isinstance(drive_indices, slice):
            drive_indices = torch.arange(joint_ids.shape[0], device=dev)[drive_indices]
        comp = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
        drive_joint_idx = torch.argmax(comp.int(), dim=0)
        articulation.actuators[drive_type].update_time_lags(time_lag)
        articulation.actuators[drive_type].update_encoder_bias(bias[:, drive_joint_idx])
        articulation.actuators[drive_type].reset(torch.arange(env.unwrapped.num_envs))

    duration = args_cli.duration
    sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
    num_steps = int(duration * sample_rate)
    t = torch.linspace(0, duration, steps=num_steps, device=dev)
    f0, f1 = args_cli.min_frequency, args_cli.max_frequency
    phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
    chirp_signal = torch.sin(phase)  # [T]
    center = torch.tensor([CHIRP[ty][0] for ty in types], device=dev)
    amp    = torch.tensor([CHIRP[ty][1] for ty in types], device=dev)
    trajectory = center.unsqueeze(0) + amp.unsqueeze(0) * chirp_signal.unsqueeze(-1)  # [T,N]

    articulation.write_joint_position_to_sim(trajectory[0, :].unsqueeze(0) + bias)
    articulation.write_joint_velocity_to_sim(torch.zeros((1, N), device=dev))

    dof_pos_buffer = torch.zeros(num_steps, N, device=dev)
    dof_target_pos_buffer = torch.zeros(num_steps, N, device=dev)
    counter = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            dof_pos_buffer[counter, :] = articulation.data.joint_pos[0, joint_ids] - bias[0]
            actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.step(actions)
            dof_target_pos_buffer[counter, :] = articulation._data.joint_pos_target[0, joint_ids]
            counter += 1
            if counter % 2000 == 0:
                print(f"[INFO]: {counter/sample_rate:.1f}s")
            if counter >= num_steps:
                break
    env.close()

    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    data_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"time": t.cpu(), "dof_pos": dof_pos_buffer.cpu(), "des_dof_pos": dof_target_pos_buffer.cpu()},
               data_dir / "chirp_data.pt")
    torch.save({"joint_order": joint_order, "types": types,
                "armature": armature.cpu(), "viscous": damping.cpu(), "coulomb": friction.cpu(),
                "bias": bias.cpu(), "delay": GT_DELAY}, data_dir / "gt.pt")
    print("[COLLECT] saved", data_dir / "chirp_data.pt")


if __name__ == "__main__":
    main()
    simulation_app.close()

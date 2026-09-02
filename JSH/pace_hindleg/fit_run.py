# Wrapper around PACE fit.py adding --max_iter override + final param dump.
import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0")
parser.add_argument("--max_iter", type=int, default=-1, help="override CMA-ES max_iteration")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg
import pace_sim2real.tasks  # noqa: F401
from pace_sim2real.utils import project_root
from pace_sim2real import CMAESOptimizer


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    articulation = env.unwrapped.scene["robot"]
    # Derive joint_order from the articulation sim order if cfg did not fix real names.
    joint_order = env_cfg.sim2real.joint_order
    auto = (joint_order is None) or (not isinstance(joint_order, list)) or (len(joint_order)==0) or (joint_order[0] not in articulation.joint_names)
    if auto:
        joint_order = list(articulation.joint_names)
    print("[FIT] joint_order:", joint_order)
    bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
    sim_joint_ids = torch.tensor([articulation.joint_names.index(n) for n in joint_order], device=env.unwrapped.device)

    data_file = project_root() / "data" / env_cfg.sim2real.data_dir
    log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name
    data = torch.load(data_file)
    target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
    measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)
    initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
    time_steps = data["time"].shape[0]
    sim_dt = env.unwrapped.sim.cfg.dt
    max_iter = args_cli.max_iter if args_cli.max_iter > 0 else env_cfg.sim2real.cmaes.max_iteration

    opt = CMAESOptimizer(bounds=bounds_params, population_size=env.unwrapped.num_envs, log_dir=log_dir,
                         joint_order=joint_order, max_iteration=max_iter, data=data, device=env.unwrapped.device,
                         epsilon=env_cfg.sim2real.cmaes.epsilon, sigma=env_cfg.sim2real.cmaes.sigma,
                         save_interval=env_cfg.sim2real.cmaes.save_interval,
                         save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process)
    env.reset()
    opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos)
    counter = 0
    while simulation_app.is_running():
        with torch.inference_mode():
            opt.tell(env.unwrapped.scene.articulations["robot"].data.joint_pos[:, sim_joint_ids],
                     measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1))
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            actions[:, sim_joint_ids] = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
            env.step(actions)
            counter += 1
            if counter % 2000 == 0:
                print(f"[INFO]: replay {counter*sim_dt:.1f}/{time_steps*sim_dt:.1f}s")
            if counter >= time_steps:
                counter = 0
                opt.evolve()
                if opt.finished():
                    break
                env.reset()
                opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos)
    best = opt.get_best_sim_params().tolist()
    N = len(joint_order)
    print("[RESULT] joint_order:", joint_order)
    print("[RESULT] armature:", best[0:N])
    print("[RESULT] viscous :", best[N:2*N])
    print("[RESULT] coulomb :", best[2*N:3*N])
    print("[RESULT] bias    :", best[3*N:4*N])
    print("[RESULT] delay   :", best[4*N])
    opt.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

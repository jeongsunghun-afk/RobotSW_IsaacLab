# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent from RSL-RL."""

import argparse
import contextlib
import importlib.metadata as metadata
import os
import sys
import time

import gymnasium as gym
import torch
from packaging import version
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg, ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.string import list_intersection, string_to_callable

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import (
    add_launcher_args,
    get_checkpoint_path,
    launch_simulation,
    setup_preset_cli,
)
from isaaclab_tasks.utils.hydra import hydra_task_config

# local imports
import cli_args  # isort: skip

# PLACEHOLDER: Extension template (do not remove this comment)
with contextlib.suppress(ImportError):
    import isaaclab_tasks_experimental  # noqa: F401

# -- argparse ----------------------------------------------------------------
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--external_callback", default=None, help="Fully qualified path to an externally defined callback.")
cli_args.add_rsl_rl_args(parser)
add_launcher_args(parser)
args_cli, remaining_args = setup_preset_cli(parser)

if args_cli.video:
    args_cli.enable_cameras = True


# Call an external callback if requested. This gives opportunity to external code to register the environments
# The function is expected to return a list of arguments that were not consumed by the callback.
remaining_args_env_registration = None
if args_cli.external_callback:
    external_callback_function = string_to_callable(args_cli.external_callback, separator=".")
    remaining_args_env_registration = external_callback_function()

# clear out sys.argv for Hydra
# The remaining arguments are the arguments that were not consumed by both this scripts
# argparser and (optionally) the external callback function.
remaining_args = list_intersection(remaining_args, remaining_args_env_registration)
sys.argv = [sys.argv[0]] + remaining_args

# Check for installed RSL-RL version
installed_version = metadata.version("rsl-rl-lib")


def use_noninstanceable_go2_for_render(env_cfg) -> None:
    """Swap the stock instanceable ``go2.usd`` for a flattened non-instanceable copy.

    The stock instanceable Unitree Go2 asset triggers an Isaac Sim 6.0 render-path issue
    (see IsaacLab issue 2925 / IsaacSim issue 227) where the visual meshes expand only
    partially in the play/viewer camera path, so recorded videos and the viewport show a
    fragmented robot (body and some legs missing). Flattening the asset and clearing the
    ``instanceable`` flag makes every link carry its own visual geometry, which renders
    reliably. This is applied to the visualization path (play) only; training keeps the
    instanceable asset, which is memory-efficient at large environment counts.
    """
    from isaaclab_assets import ISAACLAB_ASSETS_DATA_DIR

    # Locate the robot articulation cfg (Direct envs expose ``cfg.robot``; manager-based
    # envs expose ``cfg.scene.robot``).
    robot_cfg = getattr(env_cfg, "robot", None)
    if robot_cfg is None and hasattr(env_cfg, "scene"):
        robot_cfg = getattr(env_cfg.scene, "robot", None)
    spawn_cfg = getattr(robot_cfg, "spawn", None) if robot_cfg is not None else None
    usd_path = getattr(spawn_cfg, "usd_path", None) if spawn_cfg is not None else None
    if spawn_cfg is None or not (isinstance(usd_path, str) and usd_path.endswith("/Robots/Unitree/Go2/go2.usd")):
        return  # not the stock instanceable Go2 asset; nothing to swap

    noninst_path = os.path.join(ISAACLAB_ASSETS_DATA_DIR, "Robots", "Go2_noninstanceable", "go2.usd")
    if not os.path.isfile(noninst_path):
        print(
            f"[WARN] Non-instanceable Go2 asset not found at {noninst_path}; the robot may render fragmented. "
            "Generate it once with: ./isaaclab.sh -p scripts/tools/make_go2_noninstanceable.py"
        )
        return
    spawn_cfg.usd_path = noninst_path
    print(f"[INFO] Rendering with non-instanceable Go2 asset (Isaac Sim 6.0 render fix): {noninst_path}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    with launch_simulation(env_cfg, args_cli):
        # grab task name for checkpoint path
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")

        # override configurations with non-hydra CLI arguments
        agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

        # handle deprecated configurations
        # NOTE: The vendored custom runners (parkour / AMP; rsl-rl 3.2.0-style) consume the
        # deprecated `policy` config directly (including `policy.class_name` and its network
        # dims). With rsl-rl-lib >= 4.0.0 the deprecation handler infers `actor`/`critic` model
        # configs and clears `policy`, which breaks those runners. Only run the handler for the
        # stock runners that expect the new model-config format.
        if agent_cfg.class_name in ("OnPolicyRunner", "DistillationRunner"):
            agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

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
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        log_dir = os.path.dirname(resume_path)

        # set the log directory for the environment
        env_cfg.log_dir = log_dir

        # enable height-scan ray visualization in play mode.
        # Direct parkour env exposes cfg.height_scanner; manager-based envs use cfg.scene.height_scanner.
        _scanner_owners = [env_cfg]
        if hasattr(env_cfg, "scene"):
            _scanner_owners.append(env_cfg.scene)
        for _owner in _scanner_owners:
            _height_scanner_cfg = getattr(_owner, "height_scanner", None)
            if _height_scanner_cfg is not None and hasattr(_height_scanner_cfg, "debug_vis"):
                _height_scanner_cfg.debug_vis = True

        # work around the Isaac Sim 6.0 instanceable-render issue for Go2 in the play path
        use_noninstanceable_go2_for_render(env_cfg)

        # create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # convert to single-agent instance if required by the RL algorithm
        if isinstance(env.unwrapped.cfg, DirectMARLEnvCfg):
            from isaaclab.envs import multi_agent_to_single_agent

            env = multi_agent_to_single_agent(env)

        # wrap for video recording
        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # wrap around environment for rsl-rl
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunnerAMPBase":
            # Guarded (branch-local) import: only resolved when a vendored custom AMP runner is
            # requested. Stock isaac-6.0 (rsl-rl-lib 5.0.1, no custom classes) never enters here.
            import inspect

            from rsl_rl.algorithms.ppo import PPO as _VendoredPPO
            from rsl_rl.runners import OnPolicyRunnerAMPBase

            # Keep class_name (PPOAMPBase pops it before super().__init__); drop 5.0.1-only fields.
            _accepted_ppo_params = set(inspect.signature(_VendoredPPO.__init__).parameters.keys()) - {"self"}
            _cfg_dict = agent_cfg.to_dict()
            _cfg_dict["algorithm"] = {
                k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted_ppo_params or k == "class_name"
            }
            runner = OnPolicyRunnerAMPBase(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunnerParkour":
            # Guarded (branch-local) import for the vendored custom RMA/parkour runner.
            import inspect

            from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPOParkour
            from rsl_rl.runners import OnPolicyRunnerParkour

            _accepted_ppo_params = set(inspect.signature(_VendoredPPOParkour.__init__).parameters.keys()) - {"self"}
            _cfg_dict = agent_cfg.to_dict()
            _cfg_dict["algorithm"] = {
                k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted_ppo_params or k == "class_name"
            }
            runner = OnPolicyRunnerParkour(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunnerAMP":
            # Guarded (branch-local) import for the vendored custom AMP runner (go2_amp, R_Skeleton_amp).
            import inspect

            from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPOParkour
            from rsl_rl.runners import OnPolicyRunnerAMP

            # PPOAMP forwards the named algorithm cfg to PPOParkour.__init__, so filter target is PPOParkour.
            _accepted_ppo_params = set(inspect.signature(_VendoredPPOParkour.__init__).parameters.keys()) - {"self"}
            _cfg_dict = agent_cfg.to_dict()
            _cfg_dict["algorithm"] = {
                k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted_ppo_params or k == "class_name"
            }
            runner = OnPolicyRunnerAMP(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunnerParkourAMP":
            # Guarded (branch-local) import for the vendored custom parkour-AMP runner (parkour_imitation).
            import inspect

            from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPOParkour
            from rsl_rl.runners import OnPolicyRunnerParkourAMP

            _accepted_ppo_params = set(inspect.signature(_VendoredPPOParkour.__init__).parameters.keys()) - {"self"}
            _cfg_dict = agent_cfg.to_dict()
            _cfg_dict["algorithm"] = {
                k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted_ppo_params or k == "class_name"
            }
            runner = OnPolicyRunnerParkourAMP(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "OnPolicyRunnerParkourAMPVoxel":
            # Guarded (branch-local) import for the vendored custom voxel parkour-AMP runner.
            import inspect

            from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPOParkour
            from rsl_rl.runners import OnPolicyRunnerParkourAMPVoxel

            _accepted_ppo_params = set(inspect.signature(_VendoredPPOParkour.__init__).parameters.keys()) - {"self"}
            _cfg_dict = agent_cfg.to_dict()
            _cfg_dict["algorithm"] = {
                k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted_ppo_params or k == "class_name"
            }
            runner = OnPolicyRunnerParkourAMPVoxel(env, _cfg_dict, log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
        runner.load(resume_path)

        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # export the trained policy to JIT and ONNX formats
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

        # Vendored custom runners (parkour / AMP; rsl-rl 3.2.0-style) do not implement the 5.0.1
        # runner.export_policy_to_* methods; export their actor-critic via the standalone exporter.
        _custom_runner = agent_cfg.class_name in (
            "OnPolicyRunnerParkour",
            "OnPolicyRunnerAMP",
            "OnPolicyRunnerAMPBase",
            "OnPolicyRunnerParkourAMP",
            "OnPolicyRunnerParkourAMPVoxel",
        )
        if version.parse(installed_version) >= version.parse("4.0.0") and not _custom_runner:
            # use the new export functions for rsl-rl >= 4.0.0
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None  # Not needed for rsl-rl >= 4.0.0
        else:
            # extract the neural network for rsl-rl < 4.0.0
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            # export to JIT and ONNX
            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        dt = env.unwrapped.step_dt

        # reset environment
        obs = env.get_observations()
        timestep = 0
        # simulate environment
        try:
            while True:
                start_time = time.time()
                # run everything in inference mode
                with torch.inference_mode():
                    # agent stepping
                    actions = policy(obs)
                    # env stepping
                    obs, _, dones, _ = env.step(actions)
                    # reset recurrent states for episodes that have terminated
                    # policy_nn is set for the standalone-export path (rsl-rl < 4.0.0 and the
                    # vendored 3.2.0-style custom runners); reset the actor-critic directly there.
                    if policy_nn is None:
                        policy.reset(dones)
                    else:
                        policy_nn.reset(dones)
                if args_cli.video:
                    timestep += 1
                    if timestep == args_cli.video_length:
                        break

                sleep_time = dt - (time.time() - start_time)
                if args_cli.real_time and sleep_time > 0:
                    time.sleep(sleep_time)

            # close the simulator
            env.close()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

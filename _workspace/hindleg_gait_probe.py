# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HindLeg 정책의 실제 걸음을 계측한다 (학습 지표로는 판정 불가한 항목).

측정: 발 리프트(sole z − 접지 기준), 접지 듀티, 발 교대 여부, 명령 대비 실제 전진속도,
base 높이, 관절 토크/속도. 학습 로그의 ``gait_swing`` ≈ 0 이 진짜 "발을 안 든다"인지,
아니면 ``_sole_rest_z`` 기준선 오염(스폰 공중 캡처)에 의한 계측 아티팩트인지 가른다.
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg gait probe.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.3, 0.5, 1.0, 2.0], help="고정 전진 명령 [m/s].")
add_launcher_args(parser)
args_cli, _ = setup_preset_cli(parser)
args_cli.headless = True

import inspect  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # ★ 커플링 플래그는 run 마다 다르다(ablation). 소스 기본값으로 굴리면 정책이 학습된 것과
    #   **다른 플랜트**에서 측정된다 — `params/env.yaml` 에서 그 run 의 값을 복원한다.
    #   같은 함정: project_ramp_uses_source_cfg_not_run_params
    import re
    from pathlib import Path

    yml = Path(args_cli.checkpoint).resolve().parent / "params" / "env.yaml"
    keys = ("foot_coupling", "foot_transpose", "foot_raw_friction", "foot_reflected_inertia", "terminate_tilt_deg")
    if yml.exists():
        text = yml.read_text()
        for k in keys:
            m = re.search(rf"^{k}:\s*(true|false|null|[0-9.]+)\s*$", text, re.MULTILINE)
            if m is None:
                continue
            v = m.group(1)
            if v in ("true", "false"):
                parsed = v == "true"
            elif v == "null":
                parsed = None
            else:
                parsed = float(v)
            setattr(env_cfg, k, parsed)
    else:
        print(f"[probe] ⚠ {yml} 없음 — 소스 기본값으로 굴린다 (플랜트 불일치 가능)")
    print(f"[probe] 적용 cfg: {{{', '.join(f'{k}={getattr(env_cfg, k, None)}' for k in keys)}}}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        from rsl_rl.algorithms.ppo_parkour import PPOParkour
        from rsl_rl.runners import OnPolicyRunnerParkour

        accepted = set(inspect.signature(PPOParkour.__init__).parameters.keys()) - {"self"}
        cfg_dict = agent_cfg.to_dict()
        cfg_dict["algorithm"] = {
            k: v for k, v in cfg_dict["algorithm"].items() if k in accepted or k == "class_name"
        }
        runner = OnPolicyRunnerParkour(env, cfg_dict, log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        u = env.unwrapped
        robot = u._robot
        sole_ids = u._sole_body_ids
        feet_ids = u._feet_ids
        dev = u.device

        obs = env.get_observations()

        def rollout(cmd_x: float) -> dict:
            """고정 명령으로 굴리며 계측 (랜덤 명령은 정지 구간이 섞여 듀티가 흐려진다)."""
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            nonlocal obs
            rec = {k: [] for k in ("sole_z", "contact", "vx", "base_z", "tau", "qd", "obs_cmd")}
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    actions = policy(obs)
                    obs, _, _, _ = env.step(actions)
                    u._commands[:] = cmd
                    if i < 100:  # 명령 전환 과도 구간 제외
                        continue
                    # 정책이 실제로 본 명령 — policy obs 레이아웃상 [3:6] 이 commands 다
                    # (projected_gravity 3 + commands 3 + joint_pos 8 + joint_vel 8 + actions 8 + clock 4).
                    # 리셋된 env 는 _resample_commands 가 랜덤 명령을 넣으므로 그 비율도 함께 본다.
                    rec["obs_cmd"].append(obs["policy"][:, 3:6].clone())
                    rec["sole_z"].append(robot.data.body_pos_w[:, sole_ids, 2].clone())
                    f = torch.norm(u._contact_sensor.data.net_forces_w_history[:, :, feet_ids], dim=-1)
                    rec["contact"].append((torch.max(f, dim=1)[0] > 1.0).float())
                    rec["vx"].append(robot.data.root_lin_vel_b[:, 0].clone())
                    rec["base_z"].append(robot.data.root_link_pos_w[:, 2].clone())
                    rec["tau"].append(robot.data.applied_torque.abs().clone())
                    rec["qd"].append(robot.data.joint_vel.abs().clone())
            return {k: torch.stack(v) for k, v in rec.items()}

        print("\n================ HindLeg gait probe ================")
        print(f"checkpoint : {args_cli.checkpoint}   (envs={args_cli.num_envs}, 측정 {args_cli.steps - 100} step)")

        results = {}
        for cmd_x in args_cli.cmd_x:
            r = rollout(cmd_x)
            results[cmd_x] = r
            sole_z, contact = r["sole_z"], r["contact"]
            # 접지 기준선: env·발별 "접지 상태 sole z" 중앙값 (스폰 높이와 무관한 실측 지면)
            ground = torch.where(contact > 0.5, sole_z, torch.full_like(sole_z, float("nan")))
            base_line = torch.nanmedian(ground, dim=0).values  # (N,2)
            lift = sole_z - base_line.unsqueeze(0)
            swing_lift = torch.where(contact < 0.5, lift, torch.full_like(lift, float("nan")))
            vx, base_z = r["vx"], r["base_z"]
            both = (contact.sum(dim=-1) == 2).float().mean()
            alt = ((contact[..., 0] > 0.5) & (contact[..., 1] < 0.5)).float().mean()
            alt2 = ((contact[..., 1] > 0.5) & (contact[..., 0] < 0.5)).float().mean()
            sw = swing_lift[~torch.isnan(swing_lift)]
            print(f"\n[cmd vx = {cmd_x:.2f} m/s]")
            oc = r["obs_cmd"]  # (T, N, 3)
            match = (oc[..., 0] - cmd_x).abs() < 0.02
            print(f"  ★정책이 본 명령 vx: mean {oc[..., 0].mean():6.3f}  일치율 {match.float().mean() * 100:5.1f}%"
                  f"  (vy {oc[..., 1].abs().mean():.3f}, yaw {oc[..., 2].abs().mean():.3f})")
            print(f"  실제 vx  mean {vx.mean():6.3f}  p50 {vx.median():6.3f}  달성률 {vx.mean() / cmd_x * 100:5.0f}%")
            print(f"  base z   {base_z.mean():.3f} m   |tau| mean {r['tau'].mean():5.2f} p99 {r['tau'].flatten().quantile(0.99):6.2f}"
                  f"   |qd| mean {r['qd'].mean():5.2f} p99 {r['qd'].flatten().quantile(0.99):5.2f}")
            print(f"  양발 접지 {both * 100:5.1f}%   좌우 교대 HL만 {alt * 100:4.1f}% / HR만 {alt2 * 100:4.1f}%"
                  f"   유격 시간 {(contact < 0.5).float().mean() * 100:4.1f}%")
            if sw.numel():
                print(f"  swing 리프트 mean {sw.mean() * 1000:6.1f} mm  p95 {sw.quantile(0.95) * 1000:6.1f} mm  max {lift.max() * 1000:6.1f} mm  (보상 포화 70 mm)")
            else:
                print("  swing 구간 없음 — 한 번도 발이 떨어지지 않음")

        # --- 보상 계측 기준선 진단 -------------------------------------------------
        # env._sole_rest_z 는 첫 _get_rewards 호출(리셋 직후, 스폰 z=0.6)에 캡처된다.
        # 그때 로봇이 아직 공중이면 기준선이 높게 잡혀 lift 가 상시 과소평가되고
        # gait_swing 보상이 구조적으로 0에 묶인다 — 실측 지면과 비교해 확인한다.
        r = results[args_cli.cmd_x[-1]]
        ground = torch.where(r["contact"] > 0.5, r["sole_z"], torch.full_like(r["sole_z"], float("nan")))
        measured = torch.nanmedian(ground, dim=0).values.mean(dim=0)  # (2,) 발별 실측 접지 z
        rest = u._sole_rest_z
        print("\n--- gait_swing 보상 기준선 진단 ---")
        print(f"  env._sole_rest_z (첫 스텝 캡처) : {[f'{v * 1000:.1f}' for v in rest.tolist()]} mm")
        print(f"  실측 접지 sole z (정상 상태)    : {[f'{v * 1000:.1f}' for v in measured.tolist()]} mm")
        off = (rest - measured) * 1000.0
        print(f"  기준선 오차                     : {[f'{v:+.1f}' for v in off.tolist()]} mm"
              f"  → 보상상 필요한 실제 리프트 {[f'{70 + v:.0f}' for v in off.tolist()]} mm (설계값 70 mm)")
        print("====================================================\n")

        env.close()


if __name__ == "__main__":
    main()

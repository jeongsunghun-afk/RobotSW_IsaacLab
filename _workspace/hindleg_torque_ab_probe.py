# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""토크 페널티 2배 A/B — 고정 명령 0.0 / 0.5 / 1.0 m/s 에서 두 정책을 같은 env 로 잰다.

두 run 의 ``params/env.yaml`` 이 device·log_dir·`joint_torque_reward_scale` 을 빼면 **동일**함을
확인했으므로 (보상 스케일은 동역학에 영향 없음) env 인스턴스를 하나만 만들어 두 체크포인트를
번갈아 굴린다. 롤아웃마다 같은 시드로 리셋해 초기분포·이벤트 추첨을 맞춘다.

⚠ 시드를 맞춰도 **정책 간 초기상태가 비트 단위로 같지는 않다** (같은 함정:
project_leg_ramp_seed_and_render). 같은 명령·같은 분포에서 잰 짝지은 비교이지 "같은 외란" 이 아니다.
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg torque-penalty A/B at fixed commands.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--ckpt_a", type=str, required=True, help="이전 정책 (baseline).")
parser.add_argument("--ckpt_b", type=str, required=True, help="새 정책 (torque2x).")
parser.add_argument("--label_a", type=str, default="baseline")
parser.add_argument("--label_b", type=str, default="torque2x")
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup", type=int, default=100, help="명령 전환 과도 구간 (제외).")
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.0, 0.5, 1.0])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None, help="npz 저장 경로.")
add_launcher_args(parser)
args_cli, _ = setup_preset_cli(parser)
args_cli.headless = True

import copy  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

JOINTS = ["HL_hip", "HR_hip", "HL_thigh", "HR_thigh", "HL_calf", "HR_calf", "HL_foot", "HR_foot"]
# 실기 채널 토크로 환산할 때 쓰는 외부 기어단 비율 (articulation 순서).
GEAR = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.2, 1.2])

RESTORE_KEYS = (
    "foot_coupling", "foot_transpose", "foot_raw_friction", "foot_reflected_inertia",
    "foot_reflected_inertia_cap", "terminate_tilt_deg",
    "reset_joint_pos_noise", "reset_base_rp_noise_deg",
)


def restore_env_cfg(env_cfg, ckpt: str) -> None:
    """run 의 params/env.yaml 로 동역학 관련 cfg 를 되돌린다 (소스 기본값으로 굴리면 cross-plant)."""
    yml = Path(ckpt).resolve().parent / "params" / "env.yaml"
    if not yml.exists():
        print(f"[probe] ⚠ {yml} 없음 — 소스 기본값 사용")
        return
    text = yml.read_text()
    for k in RESTORE_KEYS:
        m = re.search(rf"^{k}:\s*(true|false|null|[0-9.]+)\s*$", text, re.MULTILINE)
        if m is None:
            if k in ("reset_joint_pos_noise", "reset_base_rp_noise_deg"):
                setattr(env_cfg, k, 0.0)
            continue
        v = m.group(1)
        setattr(env_cfg, k, (v == "true") if v in ("true", "false") else (None if v == "null" else float(v)))


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    restore_env_cfg(env_cfg, args_cli.ckpt_a)
    print(f"[probe] cfg: {{{', '.join(f'{k}={getattr(env_cfg, k, None)}' for k in RESTORE_KEYS)}}}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        from rsl_rl.algorithms.ppo_parkour import PPOParkour
        from rsl_rl.runners import OnPolicyRunnerParkour

        accepted = set(inspect.signature(PPOParkour.__init__).parameters.keys()) - {"self"}
        cfg_dict = agent_cfg.to_dict()
        cfg_dict["algorithm"] = {k: v for k, v in cfg_dict["algorithm"].items() if k in accepted or k == "class_name"}

        u = env.unwrapped
        robot = u._robot
        dev = u.device
        gear = GEAR.to(dev)
        soft = u._data_tensor(robot.data.soft_joint_pos_limits) if hasattr(u, "_data_tensor") else robot.data.soft_joint_pos_limits

        def load_policy(ckpt):
            # ⚠ 러너 생성자가 `policy_cfg.pop("class_name")` 으로 **넘긴 dict 를 파괴**한다.
            #    같은 dict 를 두 번 쓰면 두 번째가 KeyError 로 죽으므로 매번 깊은 복사를 준다.
            runner = OnPolicyRunnerParkour(env, copy.deepcopy(cfg_dict), log_dir=None, device=agent_cfg.device)
            runner.load(ckpt)
            return runner.get_inference_policy(device=dev)

        def rollout(policy, cmd_x: float) -> dict:
            torch.manual_seed(args_cli.seed)
            # ⚠ `inference_mode` 안에서 리셋해야 한다. 앞 롤아웃이 남긴 inference 텐서를
            #    바깥에서 in-place 갱신하면 RuntimeError 로 죽는다 (첫 롤아웃만 통과해
            #    파일이 완성된 것처럼 보이는 실패 양식).
            with torch.inference_mode():
                env.reset()
                obs = env.get_observations()
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            rec = {k: [] for k in ("vx", "vy", "base_z", "tau", "qd", "clamp", "contact", "sole_z", "tilt")}
            deaths = 0
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    actions = policy(obs)
                    obs, _, dones, _ = env.step(actions)
                    u._commands[:] = cmd
                    if i < args_cli.warmup:
                        continue
                    deaths += int(dones.sum().item())
                    pre = u.cfg.action_scale * u._actions + robot.data.default_joint_pos
                    rec["clamp"].append(((pre < soft[..., 0]) | (pre > soft[..., 1])).float())
                    rec["vx"].append(robot.data.root_lin_vel_b[:, 0].clone())
                    rec["vy"].append(robot.data.root_lin_vel_b[:, 1].clone())
                    rec["base_z"].append(robot.data.root_link_pos_w[:, 2].clone())
                    rec["tau"].append(robot.data.applied_torque.abs().clone())
                    rec["qd"].append(robot.data.joint_vel.abs().clone())
                    f = torch.norm(u._contact_sensor.data.net_forces_w_history[:, :, u._feet_ids], dim=-1)
                    rec["contact"].append((torch.max(f, dim=1)[0] > 1.0).float())
                    rec["sole_z"].append(robot.data.body_pos_w[:, u._sole_body_ids, 2].clone())
                    rec["tilt"].append(robot.data.projected_gravity_b[:, :2].norm(dim=1).clone())
            out = {k: torch.stack(v) for k, v in rec.items()}
            out["_deaths"] = deaths
            return out

        def summarize(r, cmd_x):
            tau, clamp = r["tau"], r["clamp"]
            contact, sole_z = r["contact"], r["sole_z"]
            ground = torch.where(contact > 0.5, sole_z, torch.full_like(sole_z, float("nan")))
            lift = sole_z - torch.nanmedian(ground, dim=0).values.unsqueeze(0)
            sw = lift[contact < 0.5]
            n_steps, n_env = r["vx"].shape
            return {
                "vx_mean": r["vx"].mean().item(), "vx_p50": r["vx"].median().item(),
                "vx_std": r["vx"].std().item(), "vy_abs": r["vy"].abs().mean().item(),
                "base_z": r["base_z"].mean().item(),
                "tilt_deg": torch.rad2deg(torch.asin(r["tilt"].clamp(max=1.0))).mean().item(),
                "tau_sumsq": (tau ** 2).sum(-1).mean().item(),
                "tau_mean": tau.mean().item(),
                "tau_per_joint": tau.mean(dim=(0, 1)).cpu().numpy(),
                "tau_p95_per_joint": tau.reshape(-1, 8).quantile(0.95, dim=0).cpu().numpy(),
                # 실기 트립 임계는 **채널** 15 N·m → 관절기준 15×gear 와 비교
                "trip_frac": ((tau / gear) > 15.0).float().mean(dim=(0, 1)).cpu().numpy(),
                "clamp_per_joint": clamp.mean(dim=(0, 1)).cpu().numpy(),
                "qd_mean": r["qd"].mean().item(), "qd_p99": r["qd"].reshape(-1).quantile(0.99).item(),
                "both_contact": (contact.sum(-1) == 2).float().mean().item(),
                "air_frac": (contact < 0.5).float().mean().item(),
                "swing_lift_mm": (sw.mean().item() * 1000) if sw.numel() else float("nan"),
                "deaths_per_1k_envstep": r["_deaths"] / (n_steps * n_env) * 1000,
            }

        results = {}
        for lbl, ckpt in ((args_cli.label_a, args_cli.ckpt_a), (args_cli.label_b, args_cli.ckpt_b)):
            pol = load_policy(ckpt)
            for cmd_x in args_cli.cmd_x:
                results[(lbl, cmd_x)] = summarize(rollout(pol, cmd_x), cmd_x)
                print(f"[probe] done {lbl} cmd={cmd_x}")

        A, B = args_cli.label_a, args_cli.label_b
        print("\n" + "=" * 92)
        print(f"HindLeg 토크 페널티 A/B  ({A} vs {B})   envs={args_cli.num_envs}  "
              f"측정 {args_cli.steps - args_cli.warmup} step  seed={args_cli.seed}")
        print("=" * 92)
        for cmd_x in args_cli.cmd_x:
            a, b = results[(A, cmd_x)], results[(B, cmd_x)]
            print(f"\n[cmd vx = {cmd_x:.1f} m/s]")
            print(f"  {'지표':<26} {A:>12} {B:>12} {'변화':>12}")
            rows = [
                ("실제 vx [m/s]", "vx_mean", 3), ("vx 산포 (std)", "vx_std", 3),
                ("|vy| [m/s]", "vy_abs", 3), ("base z [m]", "base_z", 3),
                ("몸통 기울기 [deg]", "tilt_deg", 2),
                ("sum(tau^2) [N2m2]", "tau_sumsq", 1), ("|tau| mean [N·m]", "tau_mean", 3),
                ("|qd| mean [rad/s]", "qd_mean", 3), ("|qd| p99", "qd_p99", 2),
                ("양발 접지 비율", "both_contact", 3), ("유격 비율", "air_frac", 3),
                ("swing 리프트 [mm]", "swing_lift_mm", 1),
                ("종료/1k env-step", "deaths_per_1k_envstep", 3),
            ]
            for nm, k, d in rows:
                va, vb = a[k], b[k]
                dv = vb - va
                pct = f"{100 * dv / abs(va):+7.1f}%" if abs(va) > 1e-9 else "     --"
                print(f"  {nm:<26} {va:>12.{d}f} {vb:>12.{d}f} {dv:>+9.{d}f} {pct}")
            print(f"  {'관절별 |tau| mean':<26}")
            for j, n in enumerate(JOINTS):
                va, vb = a["tau_per_joint"][j], b["tau_per_joint"][j]
                pct = f"{100 * (vb - va) / abs(va):+7.1f}%" if abs(va) > 1e-9 else "     --"
                print(f"    {n:<24} {va:>12.3f} {vb:>12.3f} {vb - va:>+9.3f} {pct}"
                      f"   p95 {a['tau_p95_per_joint'][j]:6.2f} -> {b['tau_p95_per_joint'][j]:6.2f}"
                      f"   trip% {100 * a['trip_frac'][j]:5.2f} -> {100 * b['trip_frac'][j]:5.2f}")
            print(f"  {'관절별 클램프 비율':<26}")
            for j, n in enumerate(JOINTS):
                print(f"    {n:<24} {a['clamp_per_joint'][j]:>12.3f} {b['clamp_per_joint'][j]:>12.3f}"
                      f" {b['clamp_per_joint'][j] - a['clamp_per_joint'][j]:>+9.3f}")
        print("=" * 92)

        if args_cli.out:
            flat = {}
            for (lbl, c), d in results.items():
                for k, v in d.items():
                    flat[f"{lbl}|{c}|{k}"] = np.asarray(v)
            np.savez(args_cli.out, **flat, meta=json.dumps({
                "ckpt_a": args_cli.ckpt_a, "ckpt_b": args_cli.ckpt_b, "seed": args_cli.seed,
                "num_envs": args_cli.num_envs, "steps": args_cli.steps, "warmup": args_cli.warmup}))
            print(f"[probe] saved {args_cli.out}")
        env.close()


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""hip 이 모터 한계(84 N·m)에 붙는 이유가 **목표각이 관절 한계 밖**이기 때문인지 확인한다.

가설
----
`_pre_physics_step` 은 `target = action_scale·action + default_joint_pos` 로 목표를 만들고
**관절 한계로 클램프하지 않는다**. hip 가동범위는 ±0.26 rad 로 좁은데 `action_scale`=0.25 라
`|action|` 이 2 를 넘으면 목표가 ±0.5 rad 로 범위 밖에 놓인다. 관절은 한계에서 멈추므로
추종오차가 **해소되지 않고 남고**, PD 가 그 오차에 비례해 계속 포화한다 — 트립이 요구하는
"넘은 채 50 ms" 가 정확히 이렇게 만들어진다.

무엇을 재나
-----------
관절 종류별로::

    ① |target| 이 관절 한계를 넘는 비율      한계 밖 목표가 실제로 나오는가
    ② 초과량 (target - limit) 분포           얼마나 밖인가
    ③ 포화(|tau| >= effort_limit·0.99) 비율   실제 포화율
    ④ 포화 스텝 중 한계밖 목표 비율          ★ 둘이 같이 가는가 (가설의 핵심)

실행::

    python _workspace/hindleg_hip_saturation.py --checkpoint <model.pt> --device cuda:0
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg hip 포화 기전 확인.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--cmd_x", type=float, nargs="+", default=[1.0])
parser.add_argument("--tilt_deg", type=float, default=50.0)
add_launcher_args(parser)
args_cli, _ = setup_preset_cli(parser)
args_cli.headless = True

import inspect  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

KINDS = ("hip", "thigh", "calf", "foot")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg)
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

        from rsl_rl.algorithms.ppo_parkour import PPOParkour
        from rsl_rl.runners import OnPolicyRunnerParkour

        accepted = set(inspect.signature(PPOParkour.__init__).parameters.keys()) - {"self"}
        cfg_dict = agent_cfg.to_dict()
        cfg_dict["algorithm"] = {k: v for k, v in cfg_dict["algorithm"].items() if k in accepted or k == "class_name"}
        runner = OnPolicyRunnerParkour(env, cfg_dict, log_dir=None, device=agent_cfg.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        u = env.unwrapped
        robot = u._robot
        act = robot.actuators["legs"]
        cos_thr = math.cos(math.radians(args_cli.tilt_deg))

        def dt_(x):
            return x.torch if hasattr(x, "torch") else x

        lim = dt_(robot.data.joint_pos_limits)  # [N, J, 2]
        cols = {k: robot.find_joints(f".*_{k}_joint", preserve_order=True)[0] for k in KINDS}
        print(f"[probe] action_scale {u.cfg.action_scale}  clip_actions {agent_cfg.clip_actions}")
        for k, ids in cols.items():
            lo, hi = float(lim[0, ids[0], 0]), float(lim[0, ids[0], 1])
            print(f"         {k:5s} 한계 [{lo:+.3f}, {hi:+.3f}] rad  폭 {hi - lo:.3f}"
                  f"  effort_limit {float(act.effort_limit[0, ids[0]]):6.1f} N·m")

        obs = env.get_observations()
        for cmd_x in args_cli.cmd_x:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            rec = {k: [] for k in ("tgt", "pos", "tau", "gz")}
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    obs, _, _, _ = env.step(policy(obs))
                    u._commands[:] = cmd
                    if i < 100:
                        continue
                    rec["tgt"].append(u._processed_actions.clone())
                    rec["pos"].append(robot.data.joint_pos.clone())
                    rec["tau"].append(robot.data.applied_torque.abs().clone())
                    rec["gz"].append(robot.data.projected_gravity_b[:, 2:3].clone())
            R = {k: torch.stack(v) for k, v in rec.items()}
            up = (R["gz"] <= -cos_thr).squeeze(-1)  # [T, N]

            print(f"\n{'=' * 96}\n[cmd vx = {cmd_x:.2f}]  기립 {float(up.float().mean()) * 100:.1f} %")
            print(f"  {'관절':6s} {'목표>한계':>9s} {'초과량 p99':>11s} {'최대':>8s}"
                  f" {'포화율':>8s} {'포화중 한계밖':>13s}")
            for k, ids in cols.items():
                tgt, tau = R["tgt"][..., ids], R["tau"][..., ids]
                lo, hi = lim[:, ids, 0].unsqueeze(0), lim[:, ids, 1].unsqueeze(0)
                # 한계 밖으로 나간 양 (양쪽 어느 쪽이든 0 이상)
                over = torch.clamp(tgt - hi, min=0.0) + torch.clamp(lo - tgt, min=0.0)
                elim = act.effort_limit[:, ids].unsqueeze(0)
                sat = tau >= elim * 0.99
                m = up.unsqueeze(-1).expand_as(over)
                ov_f, sat_f = over[m], sat[m]
                frac_over = float((ov_f > 1e-6).float().mean())
                frac_sat = float(sat_f.float().mean())
                both = float((ov_f[sat_f] > 1e-6).float().mean()) if sat_f.any() else float("nan")
                print(f"  {k:6s} {frac_over * 100:8.3f}% {float(ov_f.quantile(0.99)):10.3f}"
                      f" {float(ov_f.max()):8.3f} {frac_sat * 100:7.3f}% {both * 100:12.1f}%")
            print(f"\n  ★ 마지막 열이 100 % 에 가까우면 '포화 = 한계 밖 목표' 가설이 지지된다."
                  f"  (관절 위치는 한계에서 멈추므로 오차가 안 줄어든다)")

            # ── foot 은 벨트로 calf 와 묶여 있다 — 실제 모터가 도는 좌표는 raw = foot + calf ──
            # 관절별 클램프는 실기 브리지와 같은 좌표지만, raw 가 어디까지 가는지는 따로 봐야 한다.
            fi, ci = cols["foot"], cols["calf"]
            raw_t = R["tgt"][..., fi] + R["tgt"][..., ci]
            raw_q = R["pos"][..., fi] + R["pos"][..., ci]
            m2 = up.unsqueeze(-1).expand_as(raw_t)
            lo_r = (lim[:, fi, 0] + lim[:, ci, 0]).unsqueeze(0)
            hi_r = (lim[:, fi, 1] + lim[:, ci, 1]).unsqueeze(0)
            print(f"\n  ── foot raw 좌표 (q_foot + q_calf) — 벨트가 실제로 도는 축 ──")
            print(f"     관절별 한계의 합이 만드는 범위  [{float(lo_r.min()):+.3f}, {float(hi_r.max()):+.3f}]"
                  f"  폭 {float((hi_r - lo_r).max()):.3f} rad"
                  f"  (foot 단독 폭 {float((lim[0, fi[0], 1] - lim[0, fi[0], 0])):.3f})")
            for lab, v in (("목표 raw", raw_t), ("실제 raw", raw_q)):
                f_ = v[m2]
                print(f"     {lab}  p1 {float(f_.quantile(0.01)):+.3f}  p50 {float(f_.quantile(0.5)):+.3f}"
                      f"  p99 {float(f_.quantile(0.99)):+.3f}"
                      f"  min {float(f_.min()):+.3f}  max {float(f_.max()):+.3f}")
            print(f"     ⚠ 이 범위가 발목 모터의 실제 가동범위 안인지는 **사양이 없어 판단 불가**다."
                  f" 사양이 나오면 raw 기준 클램프를 추가할 자리다.")
        print(f"\n{'=' * 96}")
        env.close()


main()

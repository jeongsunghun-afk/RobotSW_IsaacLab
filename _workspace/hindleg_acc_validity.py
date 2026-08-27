# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""`corr = −I_off·q̈_foot` 이 커지는 것이 **물리인가 수치 아티팩트인가**를 가른다.

왜 묻나
-------
캡(25 N·m)을 넘는다는 건 `q̈_foot > 300 rad/s²` 라는 뜻이다. 그런데 발목 관절이 **모터만으로**
낼 수 있는 가속도에는 상한이 있다::

    q̈_max(모터) = effort_limit_foot / (armature_foot + I_link)

이걸 넘는 가속도는 모터가 만든 것이 아니라 **접촉 충격**이거나, `joint_acc` 가 속도의 유한차분
(`Δq̇/dt`)이라 **충격량을 dt 로 나눈 값**이 나온 것이다. 후자면 그 순간의 `corr` 은 물리량이
아니므로 **자르는 게 아니라 계산이 틀린 것**이다.

세 영역으로 나눠 본다::

    ① |q̈| ≤ cap/I_off                     캡 아래 — 정상
    ② cap/I_off < |q̈| ≤ q̈_max(모터)       모터가 낼 수 있는 진짜 가속도 — **캡이 이걸 자르면 과잉**
    ③ |q̈| > q̈_max(모터)                   모터 능력 밖 — 접촉 충격 또는 유한차분 아티팩트

그리고 ③ 이 **발 접촉 개시 스텝에 몰려 있는지** 본다. 몰려 있으면 접촉 충격이 원인이라는 직접 증거다.

실행::

    python _workspace/hindleg_acc_validity.py --device cuda:0 --checkpoint <model.pt>
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg joint_acc validity check.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--cmd_x", type=float, nargs="+", default=[1.0])
parser.add_argument("--tilt_deg", type=float, default=50.0)
parser.add_argument(
    "--dt_div", type=float, default=1.0,
    help="sim.dt 를 이 값으로 나누고 decimation 을 같은 배로 곱한다 (제어율 고정). "
    "★ 이산화 검정: 충격량은 물리량이라 dt 에 무관한데 `joint_acc = Δq̇/dt` 는 dt 에 반비례한다. "
    "dt 를 절반으로 했을 때 항의 최댓값이 약 2 배가 되면 그 크기는 **이산화 산물**이다.",
)
add_launcher_args(parser)
args_cli, _ = setup_preset_cli(parser)
args_cli.headless = True

import inspect  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

TERM_KEYS = ("foot_coupling", "foot_transpose", "foot_raw_friction", "foot_reflected_inertia")
FOOT_LINK_INERTIA = 0.0019


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device
    if args_cli.dt_div != 1.0:
        env_cfg.sim.dt = env_cfg.sim.dt / args_cli.dt_div
        env_cfg.decimation = int(round(env_cfg.decimation * args_cli.dt_div))
        print(f"[probe] ★ 이산화 검정: sim.dt {env_cfg.sim.dt:.6f} s, decimation {env_cfg.decimation}"
              f" (제어율 고정 {1.0 / (env_cfg.sim.dt * env_cfg.decimation):.1f} Hz)")
    yml = Path(args_cli.checkpoint).resolve().parent / "params" / "env.yaml"
    if yml.exists():
        text = yml.read_text()
        for k in TERM_KEYS:
            m = re.search(rf"^{k}:\s*(true|false)\s*$", text, re.MULTILINE)
            if m:
                setattr(env_cfg, k, m.group(1) == "true")
        # ★ 캡도 복원해야 한다 — 안 하면 캡 없이 학습된 run 을 소스 기본 캡(25.0)으로 재게 된다.
        m = re.search(r"^foot_reflected_inertia_cap:\s*(null|[0-9.]+)\s*$", text, re.MULTILINE)
        if m:
            env_cfg.foot_reflected_inertia_cap = None if m.group(1) == "null" else float(m.group(1))
    cap_cfg = getattr(env_cfg, "foot_reflected_inertia_cap", None)
    print(f"[probe] 적용: {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in TERM_KEYS)}}}  cap={cap_cfg}")

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
        foot_ids = u._foot_ids
        act = robot.actuators["legs"]
        cos_thr = math.cos(math.radians(args_cli.tilt_deg))

        def dt_(x):
            return x.torch if hasattr(x, "torch") else x

        i_off0 = float(dt_(robot.data.joint_armature)[0, foot_ids[0]])
        tau_max = float(act.effort_limit[0, u._foot_act_cols[0]])
        i_eff = i_off0 + FOOT_LINK_INERTIA
        acc_motor_max = tau_max / i_eff
        cap = float(cap_cfg) if cap_cfg is not None else float("inf")
        acc_cap = cap / max(i_off0, 1e-9)
        print(f"[probe] I_off {i_off0:.4f} kg·m²  I_link {FOOT_LINK_INERTIA}  τ_max(foot) {tau_max:.1f} N·m")
        print(f"[probe] 캡 {cap:.1f} N·m  ⇔ q̈ {acc_cap:7.1f} rad/s²")
        print(f"[probe] 모터 한계 q̈_max = τ_max/(I_off+I_link) = {acc_motor_max:7.1f} rad/s²"
              f"  ⇔ corr {acc_motor_max * i_off0:6.1f} N·m")

        obs = env.get_observations()
        for cmd_x in args_cli.cmd_x:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            rec = {k: [] for k in ("acc", "gz", "newc", "anyc")}
            prev_c = torch.zeros(u.num_envs, len(u._feet_ids), dtype=torch.bool, device=u.device)
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    obs, _, _, _ = env.step(policy(obs))
                    u._commands[:] = cmd
                    f = torch.norm(u._contact_sensor.data.net_forces_w_history[:, :, u._feet_ids], dim=-1)
                    c = torch.max(f, dim=1)[0] > 1.0
                    if i >= 100:
                        rec["acc"].append(robot.data.joint_acc[:, foot_ids].abs().clone())
                        rec["gz"].append(robot.data.projected_gravity_b[:, 2:3].expand(-1, 2).clone())
                        # 발 접촉 **개시** 스텝 (직전에 안 닿았다가 이번에 닿음) — 발별로 짝지어
                        rec["newc"].append((c & ~prev_c).clone())
                        rec["anyc"].append(c.clone())
                    prev_c = c
            R = {k: torch.stack(v) for k, v in rec.items()}
            up = (R["gz"] <= -cos_thr).flatten()
            a = R["acc"].flatten()[up]
            newc = R["newc"].flatten()[up]
            anyc = R["anyc"].flatten()[up]
            corr = a * i_off0

            print(f"\n{'=' * 92}\n[cmd vx = {cmd_x:.2f}]  기립 표본 {int(up.sum())}")
            print(f"  |q̈_foot| [rad/s²]  mean {float(a.mean()):8.1f}  p50 {float(a.quantile(0.5)):8.1f}"
                  f"  p95 {float(a.quantile(0.95)):8.1f}  p99 {float(a.quantile(0.99)):8.1f}"
                  f"  max {float(a.max()):9.1f}")
            r1 = a <= acc_cap
            r2 = (a > acc_cap) & (a <= acc_motor_max)
            r3 = a > acc_motor_max
            print("\n  영역 분해")
            print(f"    ① 캡 아래            |q̈| ≤ {acc_cap:7.1f}   {float(r1.float().mean()) * 100:6.3f} %"
                  f"   ← 정상, 캡 무관")
            print(f"    ② 캡~모터한계        ~{acc_motor_max:7.1f}   {float(r2.float().mean()) * 100:6.3f} %"
                  f"   ← **모터가 낼 수 있는 진짜 가속도인데 캡이 자른다**")
            print(f"    ③ 모터한계 초과      > {acc_motor_max:7.1f}   {float(r3.float().mean()) * 100:6.3f} %"
                  f"   ← 접촉 충격 / 유한차분 아티팩트")
            print(f"\n  ③ 구간의 corr 크기   mean {float(corr[r3].mean()) if r3.any() else float('nan'):8.1f}"
                  f"  max {float(corr[r3].max()) if r3.any() else float('nan'):9.1f} N·m"
                  f"   (calf effort limit 126)")

            print("\n  ★ ③ 이 발 접촉 개시 스텝에 몰려 있나 (몰려 있으면 접촉 충격이 원인)")
            base_new = float(newc.float().mean())
            print(f"    전체 스텝 중 접촉개시 비율        {base_new * 100:6.3f} %")
            for lab, m in (("② 구간", r2), ("③ 구간", r3)):
                if not m.any():
                    print(f"    {lab} 중 접촉개시 비율            표본 없음")
                    continue
                p = float(newc[m].float().mean())
                print(f"    {lab} 중 접촉개시 비율            {p * 100:6.3f} %"
                      f"   → 배율 {p / max(base_new, 1e-12):6.2f} ×")
            print(f"    (참고) 전체 접촉 중 비율 {float(anyc.float().mean()) * 100:5.1f} %"
                  f" · ③ 중 접촉 중 비율 {float(anyc[r3].float().mean()) * 100 if r3.any() else float('nan'):5.1f} %")
        print(f"\n{'=' * 92}")
        env.close()


main()

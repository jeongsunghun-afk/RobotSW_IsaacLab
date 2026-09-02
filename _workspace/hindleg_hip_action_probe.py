# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""hip 목표는 왜 soft limit 으로 가는가 — 정책 출력을 클램프 **이전**에서 본다 (2026-09-02).

왜
--
09-02 A/B 에서 hip 이 한계에 붙는 것이 **초기 자세와 무관**함이 드러났다(처짐 없이 출발해도
2 초면 같은 자리). 그러면 남는 것은 정책 자신이다. 그런데 지금까지 본 것은 전부 클램프 **이후**
값이라("한계에 붙어 있다") 정책이 얼마나 넘겼는지, 그게 편향인지 진동인지 알 수 없었다.

이 프로브는 `_pre_physics_step` 과 **같은 식**으로 목표를 클램프 전에 재구성한다::

    target_preclamp = action_scale * action + default_joint_pos
    target          = clamp(target_preclamp, soft_lo, soft_hi)

무엇을 재나
-----------
    ① 클램프율·방향   상단/하단 각각 (한쪽으로만 몰리면 **편향**, 양쪽이면 **진동 진폭 과다**)
    ② 초과량          한계를 넘은 정도 [rad] — 가동폭 대비 몇 배인지
    ③ action 편향     raw action 의 평균/표준편차. |mean| >> std 면 상수 오프셋이다
    ④ 위상 의존성     gait clock 과의 상관 — 보행 위상에 실린 진동인지, 자세 편향인지
    ⑤ 대조군          hip 이 아닌 축들의 같은 수치 (hip 만 특별한가)

실행::

    ./isaaclab.sh -p _workspace/hindleg_hip_action_probe.py --checkpoint <model.pt>
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg hip 목표 편향 프로브.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=600, help="명령당 스텝 (앞 100 은 과도 구간으로 버린다).")
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.0, 0.3, 1.0])
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
NOISE_KEYS = ("reset_joint_pos_noise", "reset_base_rp_noise_deg")


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    yml = Path(args_cli.checkpoint).resolve().parent / "params" / "env.yaml"
    if yml.exists():
        text = yml.read_text()
        for k in TERM_KEYS:
            m = re.search(rf"^{k}:\s*(true|false)\s*$", text, re.MULTILINE)
            if m:
                setattr(env_cfg, k, m.group(1) == "true")
        m = re.search(r"^foot_reflected_inertia_cap:\s*(null|[0-9.]+)\s*$", text, re.MULTILINE)
        if m:
            env_cfg.foot_reflected_inertia_cap = None if m.group(1) == "null" else float(m.group(1))
        for k in NOISE_KEYS:  # 부재 = 0 (노이즈 이전 run)
            m = re.search(rf"^{k}:\s*([0-9.]+)\s*$", text, re.MULTILINE)
            setattr(env_cfg, k, float(m.group(1)) if m else 0.0)
    print(f"[probe] 플랜트 {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in TERM_KEYS)}}}"
          f"  노이즈 {{{', '.join(f'{k}={getattr(env_cfg, k, None)}' for k in NOISE_KEYS)}}}")

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

        def t(x):
            return x.torch if hasattr(x, "torch") else x

        names = [n.replace("_joint", "") for n in robot.joint_names]
        soft = t(robot.data.soft_joint_pos_limits).clone()
        default = t(robot.data.default_joint_pos).clone()
        a_scale = float(u.cfg.action_scale)
        lo, hi = soft[0, :, 0], soft[0, :, 1]
        print(f"[probe] action_scale={a_scale}  default 전부 0={bool((default.abs() < 1e-9).all())}")
        print(f"[probe] target_preclamp = {a_scale} * action + default   (env `_pre_physics_step` 와 동일)")

        with torch.inference_mode():
            env.reset()
            obs = env.get_observations()

        for cmd_x in args_cli.cmd_x:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            with torch.inference_mode():
                env.reset()
                obs = env.get_observations()
            rec_a, rec_ph, rec_gz = [], [], []
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    act = policy(obs)
                    obs, _, _, _ = env.step(act)
                    u._commands[:] = cmd
                    if i < 100:
                        continue
                    rec_a.append(act.clone())
                    rec_ph.append(u._gait_phase.clone())
                    rec_gz.append(t(robot.data.projected_gravity_b)[:, 2].clone())
            A = torch.stack(rec_a)                       # [T, N, J]
            PH = torch.stack(rec_ph)                     # [T, N]
            up = (torch.stack(rec_gz) <= -math.cos(math.radians(50.0)))  # 걷는 스텝만
            T = a_scale * A + default.unsqueeze(0)       # 클램프 전 목표
            m = up.unsqueeze(-1).expand_as(A)

            print(f"\n{'=' * 104}\n[cmd vx = {cmd_x:.2f}]  기립 스텝 {float(up.float().mean()) * 100:.1f} %"
                  f"   표본 {int(m[..., 0].sum())} (스텝×env)")
            print(f"{'관절':>8} {'가동폭':>7} | {'action mean':>11} {'std':>6} {'|mean|/std':>10}"
                  f" | {'상단클램프':>9} {'하단클램프':>9} | {'초과량 p99':>10} {'최대':>8} | {'위상상관':>8}")
            for j in range(len(names)):
                span = float(hi[j] - lo[j])
                a = A[..., j][up]
                tt = T[..., j][up]
                over_hi = (tt > hi[j]).float().mean() * 100
                over_lo = (tt < lo[j]).float().mean() * 100
                exc = torch.where(tt > hi[j], tt - hi[j], torch.where(tt < lo[j], lo[j] - tt, torch.zeros_like(tt)))
                # gait clock 과의 상관 — 위상에 실린 진동이면 |r| 이 크다
                ph = PH[up]
                sph = torch.sin(2 * math.pi * ph)
                aa = a - a.mean()
                ss = sph - sph.mean()
                den = aa.std() * ss.std()
                r = float((aa * ss).mean() / den) if float(den) > 1e-12 else 0.0
                mark = "  ←" if names[j].endswith("hip") else ""
                print(f"{names[j]:>8} {span:7.3f} | {float(a.mean()):11.3f} {float(a.std()):6.3f}"
                      f" {abs(float(a.mean())) / max(float(a.std()), 1e-9):10.2f}"
                      f" | {float(over_hi):8.1f}% {float(over_lo):8.1f}%"
                      f" | {float(exc.quantile(0.99)):10.3f} {float(exc.max()):8.3f} | {r:+8.2f}{mark}")
        print(f"\n{'=' * 104}")
        env.close()


main()

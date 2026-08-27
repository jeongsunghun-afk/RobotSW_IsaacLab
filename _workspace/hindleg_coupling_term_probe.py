# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""커플링 3 항이 각 관절 토크에 **얼마나** 기여하는지 잰다 — 자세로 나눠서.

왜 자세로 나누나
----------------
2026-08-27 실측(`hindleg_fall_probe.py`)에서 `pace0819sym` 정책이 에피소드의 92~93 %를
**뒤로 누운 채** 보낸다는 것이 나왔다. 종전 §39-e 측정은 그 정책의 `model_18000` 으로 냈는데
그 체크포인트도 92.34 % 가 누운 자세였다 — 즉 그 숫자들은 "보행 중 로봇"이 아니라 "누워 있는
로봇"의 토크였다. 누운 자세는 다리 링크가 지면 반력 160 N 을 받는 완전히 다른 가속도 분포다.

그래서 이 프로브는 **기립/넘어짐을 나눠서** 보고하고, 기본 사용법은 *걷는* 정책
(`nocouple`)을 쓰되 세 항을 계측용으로 강제 ON 하는 것이다::

    # 걷는 정책 + 항 강제 ON — "이 항들이 정상 보행에 얼마를 넣는가"
    python _workspace/hindleg_coupling_term_probe.py --device cuda:1 \
        --checkpoint logs/.../pace0819sym_nocouple/model_34100.pt --force_terms on

    # 같은 정책, 학습된 플랜트 그대로(항 OFF) — 재계산값만 본다. 거동 오염 없음
    python _workspace/hindleg_coupling_term_probe.py --device cuda:1 \
        --checkpoint logs/.../pace0819sym_nocouple/model_34100.pt --force_terms off

``--force_terms on`` 은 정책이 학습한 것과 다른 플랜트라 거동이 바뀐다(그게 관측 대상이다).
``off`` 는 플랜트를 안 건드리고 **같은 식으로 재계산만** 하므로 "정상 보행에 넣었더라면 얼마"를
오염 없이 준다. 두 값을 나란히 보면 항이 거동을 얼마나 밀어내는지가 보인다.

재는 항 (``hind_leg_env._apply_action`` 과 같은 식)::

    corr_calf = −I_off·q̈_foot      반사관성 off-diagonal → calf   (안정성 캡 없음)
    corr_foot = −I_off·q̈_calf                            → foot   (안정성 캡 없음)
    tau_tr    = τ_applied[foot] − mrefl_prev   전치        → calf   (안정성 캡 없음)
    tau_fric  = −(b_raw·w + c_raw·tanh(w/eps)) raw 마찰    → foot·calf (캡 有)
                w = q̇_foot + q̇_calf,  |τ| ≤ (I+I_link)·|w|/dt 및 effort_limit
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg coupling-term contribution probe.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=600, help="명령당 스텝 수 (앞 100 은 과도 구간으로 버린다).")
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="고정 전진 명령 [m/s].")
parser.add_argument(
    "--force_terms",
    choices=["on", "off", "keep"],
    default="keep",
    help="세 항을 강제로 켜거나(on) 끄거나(off) run 값 그대로(keep). 'off' 여도 재계산값은 보고한다.",
)
parser.add_argument("--tilt_deg", type=float, default=50.0, help="넘어짐 판정 기울기 [deg].")
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


def summarize(x: torch.Tensor, mask: torch.Tensor) -> tuple[float, ...]:
    """(mean, p50, p95, p99, max) — mask 로 고른 스텝만. 표본이 없으면 nan."""
    if mask.sum() == 0:
        return (float("nan"),) * 5
    v = x[mask].flatten()
    return (float(v.mean()), *(float(v.quantile(q)) for q in (0.5, 0.95, 0.99)), float(v.max()))


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # run 의 커플링 플래그를 복원한 뒤 --force_terms 로 덮어쓴다 (소스 기본값으로 굴리면 정책이
    # 학습된 것과 다른 플랜트가 된다 — project_ramp_uses_source_cfg_not_run_params 와 같은 함정).
    yml = Path(args_cli.checkpoint).resolve().parent / "params" / "env.yaml"
    if yml.exists():
        text = yml.read_text()
        for k in TERM_KEYS:
            m = re.search(rf"^{k}:\s*(true|false)\s*$", text, re.MULTILINE)
            if m:
                setattr(env_cfg, k, m.group(1) == "true")
        print(f"[probe] run cfg 복원: {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in TERM_KEYS)}}}")
    if args_cli.force_terms != "keep":
        on = args_cli.force_terms == "on"
        for k in TERM_KEYS:
            setattr(env_cfg, k, True if k == "foot_coupling" else on)
        print(f"[probe] ★ --force_terms {args_cli.force_terms} → 세 항 {'ON' if on else 'OFF'} 로 강제")
    print(f"[probe] 적용: {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in TERM_KEYS)}}}")

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
        calf_ids, foot_ids = u._calf_ids, u._foot_ids
        act = robot.actuators["legs"]
        cos_thr = math.cos(math.radians(args_cli.tilt_deg))

        def dt_(x):
            return x.torch if hasattr(x, "torch") else x

        obs = env.get_observations()

        def rollout(cmd_x: float) -> dict:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            nonlocal obs
            keys = ("corr_calf", "corr_foot", "tau_fric", "tau_tr", "pd_calf", "pd_foot", "gz", "acc_foot", "acc_calf")
            rec: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    obs, _, _, _ = env.step(policy(obs))
                    u._commands[:] = cmd
                    if i < 100:
                        continue
                    acc = robot.data.joint_acc
                    i_off = dt_(robot.data.joint_armature)[:, foot_ids]
                    rec["corr_calf"].append((i_off * acc[:, foot_ids]).abs().clone())
                    rec["corr_foot"].append((i_off * acc[:, calf_ids]).abs().clone())
                    rec["acc_foot"].append(acc[:, foot_ids].abs().clone())
                    rec["acc_calf"].append(acc[:, calf_ids].abs().clone())
                    # raw 좌표 마찰 — env 와 같은 식 (안정성 캡 포함)
                    w = robot.data.joint_vel[:, foot_ids] + robot.data.joint_vel[:, calf_ids]
                    b = dt_(robot.data.default_joint_viscous_friction_coeff)[:, foot_ids]
                    c = dt_(robot.data.default_joint_friction_coeff)[:, foot_ids]
                    eps = max(float(getattr(u.cfg, "foot_raw_friction_vel_eps", 0.2)), 1e-6)
                    mag = (b * w + c * torch.tanh(w / eps)).abs()
                    cap = (dt_(robot.data.default_joint_armature)[:, foot_ids] + 0.0019) * w.abs() / u.cfg.sim.dt
                    rec["tau_fric"].append(torch.minimum(mag, cap).clone())
                    # 전치 — env 와 같은 식 (직전 스텝 반사관성 항을 뺀다)
                    prev = getattr(u, "_mrefl_foot_prev", None)
                    tr = robot.data.applied_torque[:, foot_ids]
                    rec["tau_tr"].append((tr - prev if prev is not None else tr).abs().clone())
                    for nm, ids in (("pd_calf", calf_ids), ("pd_foot", foot_ids)):
                        e = robot.data.joint_pos_target[:, ids] - robot.data.joint_pos[:, ids]
                        pd = act.stiffness[:, ids] * e - act.damping[:, ids] * robot.data.joint_vel[:, ids]
                        rec[nm].append(pd.abs().clone())
                    # 자세 (좌우 두 관절 열에 맞춰 브로드캐스트)
                    rec["gz"].append(robot.data.projected_gravity_b[:, 2:3].expand(-1, 2).clone())
            return {k: torch.stack(v) for k, v in rec.items()}

        print("\n================ coupling-term contribution probe ================")
        print(f"checkpoint : {args_cli.checkpoint}")
        print(f"envs {args_cli.num_envs} · 명령당 {args_cli.steps - 100} step · 넘어짐 판정 tilt>{args_cli.tilt_deg}°")
        print(f"구조항: transpose={u.cfg.foot_transpose} reflI={u.cfg.foot_reflected_inertia}"
              f" rawfric={u.cfg.foot_raw_friction}  (force_terms={args_cli.force_terms})")

        pairs = (("corr_calf", "pd_calf", "reflI→calf"), ("tau_tr", "pd_calf", "전치→calf"),
                 ("corr_foot", "pd_foot", "reflI→foot"), ("tau_fric", "pd_foot", "rawfric→foot"))
        for cmd_x in args_cli.cmd_x:
            R = rollout(cmd_x)
            up = R["gz"] <= -cos_thr
            frac_up = float(up.float().mean()) * 100
            print(f"\n{'=' * 92}\n[cmd vx = {cmd_x:.2f} m/s]  기립 {frac_up:5.1f}% / 넘어짐 {100 - frac_up:5.1f}%")
            for label, mask in (("기립(보행)", up), ("넘어짐", ~up)):
                if mask.sum() == 0:
                    print(f"\n  -- {label}: 표본 없음")
                    continue
                print(f"\n  -- {label}  ({int(mask.sum())} 표본) --")
                print(f"  {'항 [N·m]':22s}{'mean':>8s}{'p50':>8s}{'p95':>8s}{'p99':>8s}{'max':>9s}")
                for k in ("pd_calf", "corr_calf", "tau_tr", "pd_foot", "corr_foot", "tau_fric"):
                    s = summarize(R[k], mask)
                    print(f"  {k:22s}{s[0]:8.3f}{s[1]:8.3f}{s[2]:8.3f}{s[3]:8.3f}{s[4]:9.3f}")
                for k, nm in (("acc_foot", "q̈_foot [rad/s²]"), ("acc_calf", "q̈_calf [rad/s²]")):
                    s = summarize(R[k], mask)
                    print(f"  {nm:22s}{s[0]:8.1f}{s[1]:8.1f}{s[2]:8.1f}{s[3]:8.1f}{s[4]:9.1f}")
                print(f"  {'항 / 그 관절 PD':22s}{'mean비':>8s}{'p50비':>8s}{'>1 비율':>10s}{'항 평균':>9s}{'PD 평균':>9s}")
                for ff, pd, nm in pairs:
                    r = (R[ff] / R[pd].clamp_min(1e-6))[mask]
                    print(f"  {nm:22s}{float(r.mean()):8.2f}{float(r.quantile(0.5)):8.2f}"
                          f"{float((r > 1).float().mean()) * 100:9.1f}%"
                          f"{float(R[ff][mask].mean()):9.3f}{float(R[pd][mask].mean()):9.3f}")
        print(f"\n{'=' * 92}")
        print("⚠ 'off' 로 잰 값은 그 항이 실제로 적용되지 않은 궤적에서의 **재계산값**이다 —")
        print("  '정상 보행에 넣었더라면 얼마'를 오염 없이 준다. 'on' 은 거동이 바뀐 뒤의 실측값이다.")
        env.close()


main()

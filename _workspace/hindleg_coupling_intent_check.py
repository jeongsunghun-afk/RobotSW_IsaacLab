# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""커플링 구현이 **유도한 물리대로 동작하는지**를 학습 결과와 무관하게 검정한다.

무엇을 묻나
-----------
`hind_leg_env._apply_action` 은 foot PD 를 raw(모터축) 공간에서 계산하고, 그 모터 토크를
전치해 calf 에 싣는다::

    pos_t = raw_t − q_c,   vel_t = −q̇_c
    ⇒ foot actuator 가 내는 것 = kp_f·(raw_t − (q_f + q_c)) + kd_f·(−q̇_c − q̇_f)   [DCMotor 로 클립]
    ⇒ τ_calf += 그 토크 (전치)

즉 벨트는 무릎에 **강성과 감쇠를 동시에** 얹는다. 강성 항은 calf 를 목표 쪽으로 밀어 주고
(`raw_t` 가 `a_calf` 를 포함하므로), 감쇠 항 `−kd_f·q̇_c` 는 **무릎 운동을 항상 거스른다.**
둘 중 무엇이 지배하는지는 유도만으로는 안 나온다 — 재야 한다.

주 검정 (B): `tau_tr` 을 무릎 상태에 회귀
------------------------------------------
::

    tau_tr ≈ k_eff·(a_calf − q_c) + b_eff·(−q̇_c) + c

구현이 의도대로면 `k_eff ≈ kp_foot`, `b_eff ≈ kd_foot` (관절 공간 배포값 28.8 / 7.2) 가 나와야
한다. 부호가 뒤집히거나 크기가 크게 어긋나면 **구현 버그**이고, 맞으면 "벨트가 무릎 임피던스를
이만큼 바꾼다"는 **설계 귀결**이다 (kd_calf 11.25 대비 얼마인지 함께 낸다).

보조
----
- (A) raw 추종: `q_f + q_c` 가 `raw_t = a_foot + a_calf` 를 따라가는가 (위치 커플링 정상 동작).
- (C) 무릎에 들어가는 **순 기계일률** `(tau_tr + corr_calf)·q̇_c` — 음수면 벨트가 무릎을 제동한다.
  `corr_calf = −I_off·q̈_foot` 은 무릎 상태와 무상관이므로, 순일률이 크게 음수면 범인은 감쇠 항이다.

⚠ **걷는 정책으로 재야 한다.** `termsON` 정책은 `|q̇_calf|` 가 0 근처에 눌려 있어 회귀가
  ill-conditioned 다. `nocouple` 정책에 `--force_terms on` 을 걸어 무릎이 실제 범위를 쓰게 한다.
⚠ 일률 보존(`q̇ᵀQ = ṙᵀτ_m`)은 `Tᵀ` 구성상 **항등**이라 검정이 안 된다 — 재지 않는다.

실행::

    python _workspace/hindleg_coupling_intent_check.py --device cuda:3 \
        --checkpoint logs/.../termfix_nocouple/model_6000.pt --force_terms on
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg coupling intent check.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=700)
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.5, 1.0])
parser.add_argument("--force_terms", choices=["on", "off", "keep"], default="on")
parser.add_argument("--tilt_deg", type=float, default=50.0)
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


def lstsq_report(y: torch.Tensor, cols: dict[str, torch.Tensor]) -> tuple[dict[str, float], float, float]:
    """최소자승 + R² + 설계행렬 조건수. cols 는 {이름: 열}."""
    names = list(cols)
    A = torch.stack([cols[n] for n in names] + [torch.ones_like(y)], dim=1).double()
    b = y.double()
    sol = torch.linalg.lstsq(A, b.unsqueeze(1)).solution.squeeze(1)
    pred = A @ sol
    ss_res = float(((b - pred) ** 2).sum())
    ss_tot = float(((b - b.mean()) ** 2).sum())
    cond = float(torch.linalg.cond(A))
    return {**{n: float(sol[i]) for i, n in enumerate(names)}, "const": float(sol[-1])}, 1 - ss_res / ss_tot, cond


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    yml = Path(args_cli.checkpoint).resolve().parent / "params" / "env.yaml"
    if yml.exists():
        text = yml.read_text()
        for k in TERM_KEYS:
            m = re.search(rf"^{k}:\s*(true|false)\s*$", text, re.MULTILINE)
            if m:
                setattr(env_cfg, k, m.group(1) == "true")
    if args_cli.force_terms != "keep":
        on = args_cli.force_terms == "on"
        for k in TERM_KEYS:
            setattr(env_cfg, k, True if k == "foot_coupling" else on)
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

        hip_ids, _ = robot.find_joints(["HL_hip_joint", "HR_hip_joint"], preserve_order=True)
        thigh_ids, _ = robot.find_joints(["HL_thigh_joint", "HR_thigh_joint"], preserve_order=True)
        kp_c = float(act.stiffness[0, calf_ids[0]])
        kd_c = float(act.damping[0, calf_ids[0]])
        kp_f = float(act.stiffness[0, foot_ids[0]])
        kd_f = float(act.damping[0, foot_ids[0]])
        print(f"[probe] 배포 게인 (관절공간): kp_calf {kp_c:.4g}  kd_calf {kd_c:.4g}"
              f"   kp_foot {kp_f:.4g}  kd_foot {kd_f:.4g}")

        obs = env.get_observations()

        def rollout(cmd_x: float) -> dict:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            nonlocal obs
            keys = ("tau_tr", "corr_calf", "e_calf", "e_foot", "dq_calf", "dq_foot",
                    "raw_err", "raw", "raw_t", "gz", "e_hip", "e_thigh", "t_hip", "t_thigh",
                    "t_calf", "t_foot_raw")
            rec: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    obs, _, _, _ = env.step(policy(obs))
                    u._commands[:] = cmd
                    if i < 100:
                        continue
                    q, dq = robot.data.joint_pos, robot.data.joint_vel
                    a = u._processed_actions
                    raw = q[:, foot_ids] + q[:, calf_ids]
                    raw_t = a[:, foot_ids] + a[:, calf_ids]
                    prev = getattr(u, "_mrefl_foot_prev", None)
                    tr = robot.data.applied_torque[:, foot_ids]
                    rec["tau_tr"].append((tr - prev if prev is not None else tr).clone())
                    if u.cfg.foot_reflected_inertia:
                        i_off = dt_(robot.data.joint_armature)[:, foot_ids]
                        rec["corr_calf"].append((-i_off * robot.data.joint_acc[:, foot_ids]).clone())
                    else:
                        rec["corr_calf"].append(torch.zeros_like(raw))
                    rec["e_calf"].append((a[:, calf_ids] - q[:, calf_ids]).clone())
                    rec["e_foot"].append((a[:, foot_ids] - q[:, foot_ids]).clone())
                    rec["dq_calf"].append(dq[:, calf_ids].clone())
                    rec["dq_foot"].append(dq[:, foot_ids].clone())
                    # 다른 관절의 자기 추종 — "PD 가 원래 무른가"를 가르는 대조군
                    rec["e_hip"].append((a[:, hip_ids] - q[:, hip_ids]).clone())
                    rec["e_thigh"].append((a[:, thigh_ids] - q[:, thigh_ids]).clone())
                    rec["t_hip"].append(a[:, hip_ids].clone())
                    rec["t_thigh"].append(a[:, thigh_ids].clone())
                    rec["t_calf"].append(a[:, calf_ids].clone())
                    rec["t_foot_raw"].append(raw_t.clone())
                    rec["raw_err"].append((raw_t - raw).clone())
                    rec["raw"].append(raw.clone())
                    rec["raw_t"].append(raw_t.clone())
                    rec["gz"].append(robot.data.projected_gravity_b[:, 2:3].expand(-1, 2).clone())
            return {k: torch.stack(v) for k, v in rec.items()}

        print("\n================ coupling intent check ================")
        print(f"checkpoint : {args_cli.checkpoint}")
        for cmd_x in args_cli.cmd_x:
            R = rollout(cmd_x)
            up = (R["gz"] <= -cos_thr).flatten()
            n_up = int(up.sum())
            print(f"\n{'=' * 88}\n[cmd vx = {cmd_x:.2f}]  기립 표본 {n_up} / {up.numel()}")
            if n_up < 1000:
                print("  기립 표본 부족 — 회귀 생략")
                continue
            f = {k: v.flatten()[up] for k, v in R.items()}

            # --- (A) 추종 오차를 **관절끼리 비교** ------------------------------------------
            # raw 추종만 보면 "PD 가 원래 무른 것"과 "커플링이 안 먹는 것"을 못 가른다.
            # 같은 run 의 hip/thigh/calf 자기 추종과 나란히 놓아야 판별이 된다.
            print("\n  (A) 추종 오차 / 목표 std  — 관절 간 대조 (커플링이 특별히 나쁜가?)")
            print(f"      {'축':16s}{'|err| mean':>11s}{'err std':>9s}{'target std':>11s}{'비율':>7s}{'corr':>7s}")
            for lab, ek, tk in (("hip", "e_hip", "t_hip"), ("thigh", "e_thigh", "t_thigh"),
                                ("calf", "e_calf", "t_calf"), ("foot(raw)", "raw_err", "t_foot_raw")):
                e, t = f[ek], f[tk]
                meas = t - e
                c = float(torch.corrcoef(torch.stack([meas, t]))[0, 1])
                print(f"      {lab:16s}{float(e.abs().mean()):11.4f}{float(e.std()):9.4f}"
                      f"{float(t.std()):11.4f}{float(e.std()) / max(float(t.std()), 1e-9):7.3f}{c:7.3f}")
            ce = float(torch.corrcoef(torch.stack([f["e_foot"], f["e_calf"]]))[0, 1])
            print(f"      ★ corr(e_foot, e_calf) = {ce:+.4f}  — −1 에 가까우면 foot 이 raw 를 쫓느라"
                  " calf 오차를 상쇄한다(= 벨트 강성이 무릎에 안 남는다)")

            # --- (B) 전치 토크를 **actuator 가 실제로 보는 입력**에 회귀 -------------------
            # ⚠ e_calf 단독 회귀는 생략변수 편향으로 부호가 뒤집힌다: foot 이 raw 를 쫓으므로
            #   e_foot ≈ −e_calf 이고, kp_f·(e_foot + e_calf) 의 두 항이 서로 지운다.
            #   회귀식은 구현 그대로여야 한다:
            #       tau_tr ≈ kp_f·(raw_t − raw) + kd_f·(−q̇_calf − q̇_foot) + c
            coef, r2, cond = lstsq_report(
                f["tau_tr"], {"raw_err": f["raw_err"], "neg_dq_raw": -(f["dq_calf"] + f["dq_foot"])}
            )
            print("\n  (B) ★ tau_tr ≈ kp_f·(raw_t − raw) + kd_f·(−q̇_calf − q̇_foot) + c")
            print(f"      kp_f(적합) = {coef['raw_err']:8.3f}   설계 {kp_f:.4g}"
                  f"   →  {coef['raw_err'] / kp_f * 100:6.1f} %")
            print(f"      kd_f(적합) = {coef['neg_dq_raw']:8.3f}   설계 {kd_f:.4g}"
                  f"   →  {coef['neg_dq_raw'] / kd_f * 100:6.1f} %")
            print(f"      상수항 {coef['const']:+.3f} N·m   R² {r2:.4f}   조건수 {cond:.1f}")
            print(f"      적합 범위: raw_err std {float(f['raw_err'].std()):.4f} rad ·"
                  f" (q̇_c+q̇_f) std {float((f['dq_calf'] + f['dq_foot']).std()):.3f} rad/s")
            # 무릎만 떼어 본 유효 임피던스 (구현이 맞다는 전제에서의 **귀결**)
            print(f"      ⇒ 벨트가 무릎에 얹는 감쇠 = kd_f {kd_f:.3g} N·m·s/rad"
                  f"  (kd_calf {kd_c:.3g} 대비 {kd_f / kd_c * 100:+.0f} %)")

            # --- (C) 무릎에 들어가는 순 기계일률 -------------------------------------------
            p_tr = f["tau_tr"] * f["dq_calf"]
            p_ci = f["corr_calf"] * f["dq_calf"]
            print("\n  (C) 무릎 순 기계일률  (음수 = 벨트가 무릎을 제동)")
            print(f"      전치 τ·q̇_calf        mean {float(p_tr.mean()):+8.3f} W   음수 비율 {float((p_tr < 0).float().mean()) * 100:5.1f} %")
            print(f"      반사관성 τ·q̇_calf     mean {float(p_ci.mean()):+8.3f} W   음수 비율 {float((p_ci < 0).float().mean()) * 100:5.1f} %")
            print(f"      합계                  mean {float((p_tr + p_ci).mean()):+8.3f} W")
        print(f"\n{'=' * 88}")
        env.close()


main()

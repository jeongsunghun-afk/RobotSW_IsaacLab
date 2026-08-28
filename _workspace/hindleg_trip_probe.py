# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""배포 전 안전 게이트 — **이 정책을 실기에 올리면 토크 트립에 걸리는가**.

왜 묻나
-------
실기 펌웨어는 **보고 토크 15 N·m 가 50 ms 지속**되면 limp 래치를 건다. 보고값이 채널 기준이라
관절 기준으로는 `calf 22.5` · `foot 18.0` N·m 다 (`TORQUE_COORDINATES.md` §6-i).
래치가 걸리면 로봇이 힘을 잃으므로, 걸리는 정책을 올리면 **데이터가 아니라 사고**가 나온다.

무엇을 재나
-----------
분포가 아니라 **지속 시간**을 잰다. 트립 조건이 "넘는다"가 아니라 "넘은 채로 50 ms"이기 때문이다.
p99 가 임계를 넘어도 한 스텝씩 흩어져 있으면 안 걸리고, p95 만 넘어도 연속이면 걸린다::

    ① 초과율        임계를 넘는 스텝의 비율
    ② 연속 런 길이   넘은 채 이어진 스텝 수의 분포 (제어 1 스텝 = 20 ms @50 Hz)
    ③ 트립률        `ceil(50 ms / dt_ctrl)` 스텝 이상 이어진 런이 하나라도 있는 env 의 비율

`joint_pos_target` 기반 PD 재계산이 아니라 **`applied_torque` 실측**을 쓴다 — 커플링 항·
DCMotor 클립까지 반영된, 실기가 실제로 보고할 값에 가장 가까운 양이다.

실행::

    python _workspace/hindleg_trip_probe.py --checkpoint <model.pt> --device cuda:0
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg 실기 토크 트립 게이트.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=600, help="명령당 스텝 수 (앞 100 은 과도 구간으로 버린다).")
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.3, 0.5, 1.0])
parser.add_argument(
    "--trip_nm", type=float, nargs="+", default=[15.0],
    help="펌웨어 보고 토크 임계 [N·m, 채널 기준]. 여러 개 주면 **한 번의 롤아웃**으로 전부 평가한다 — "
    "'임계를 얼마로 풀면 트립이 사라지는가'를 재는 용도(같은 궤적이라 비교가 공정하다).",
)
parser.add_argument("--trip_ms", type=float, default=50.0, help="래치까지 필요한 지속 시간 [ms].")
parser.add_argument("--tilt_deg", type=float, default=50.0, help="넘어짐 판정 기울기 [deg] — 걷는 스텝만 센다.")
parser.add_argument(
    "--real_gains", action="store_true",
    help="★ 실기가 실제로 만들 관절 게인(`rga.py` HIND_LEG_CFG 의 gear² 환산값)으로 바꿔 잰다. "
    "학습 env(`hind_leg_env_cfg.py`)는 채널값을 관절 게인으로 그대로 써서 calf 50 / foot 20 인데, "
    "실기는 같은 채널값이 관절에서 calf 112.5 / foot 28.8 로 나타난다 — 즉 **정책이 실기에서 만날 무릎은 "
    "학습보다 2.25 배 단단하다**. 이 플래그가 그 배포 조건을 재현한다(정책엔 분포 밖).",
)
parser.add_argument(
    "--act_scale", type=float, default=1.0,
    help="정책 action 에 곱하는 배율 [0~1]. 배포 GUI 의 action scale 과 같은 자리다 — 트립을 "
    "피하려고 이 값을 낮췄을 때 **토크가 실제로 얼마나 내려가고 속도를 얼마나 잃는지** 잰다. "
    "⚠ 정책은 배율 1.0 에서 학습됐으므로 낮춘 값은 학습 분포 밖이다.",
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
# 채널→관절 변환비. 펌웨어 임계는 채널 기준이라 관절 기준 임계 = 15 × gear_k.
# ⚠ hip·thigh 의 k 는 1.0 이라 **관절 기준 임계가 가장 낮다**(15.0 < foot 18.0 < calf 22.5).
#   k≠1 축에만 눈이 가서 빠뜨리기 쉬운데, 트립은 네 종류 모두에서 걸릴 수 있다.
GEAR_K = {"hip": 1.0, "thigh": 1.0, "calf": 1.5, "foot": 1.2}


def run_lengths(mask: torch.Tensor) -> torch.Tensor:
    """`mask` [T, N] 의 시간축에서 True 가 연속된 런의 길이들을 1-D 로 모은다."""
    t, n = mask.shape
    pad = torch.zeros(1, n, dtype=torch.bool, device=mask.device)
    m = torch.cat([pad, mask, pad], dim=0).to(torch.int8)
    d = m[1:] - m[:-1]
    starts = (d == 1).nonzero(as_tuple=False)
    ends = (d == -1).nonzero(as_tuple=False)
    if starts.numel() == 0:
        return torch.zeros(0, dtype=torch.long, device=mask.device)
    # nonzero 는 (시간, env) 순 정렬이라 같은 env 안에서 시작·끝이 짝지어진다.
    order_s = starts[:, 1] * (t + 2) + starts[:, 0]
    order_e = ends[:, 1] * (t + 2) + ends[:, 0]
    return (order_e.sort().values - order_s.sort().values).to(torch.long)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    # run cfg 복원 — 소스 기본값으로 굴리면 정책이 학습된 것과 다른 플랜트다.
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
    cap = getattr(env_cfg, "foot_reflected_inertia_cap", None)
    print(f"[probe] 적용: {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in TERM_KEYS)}}}  cap={cap}")

    if args_cli.real_gains:
        # 실기 채널 게인 100/50/50/20 · kd 5 를 관절 공간으로 환산한 값 (kp_joint = k^2 * kp_ch).
        # 출처는 `rga.py` HIND_LEG_CFG (2026-08-18). 학습 env 는 이 환산 없이 채널값을 그대로 쓴다.
        env_cfg.robot.actuators["legs"] = env_cfg.robot.actuators["legs"].replace(
            stiffness={
                ".*_hip_joint": 100.0, ".*_thigh_joint": 50.0,
                ".*_calf_joint": 112.5, ".*_foot_joint": 28.8,
            },
            damping={
                ".*_hip_joint": 5.0, ".*_thigh_joint": 5.0,
                ".*_calf_joint": 11.25, ".*_foot_joint": 7.2,
            },
        )
        print("[probe] ★ --real_gains: calf kp 50→112.5 (kd 5→11.25) · foot kp 20→28.8 (kd 5→7.2)")
        print("[probe]   = 같은 채널 게인이 실기 관절에서 나타나는 값. 정책엔 **학습 분포 밖**이다.")

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
        # 네 종류 전부 본다 — 관절명으로 찾는다(env 가 들고 있는 것은 calf/foot 뿐이다).
        joint_ids = {}
        for kind in GEAR_K:
            ids, _ = robot.find_joints(f".*_{kind}_joint", preserve_order=True)
            joint_ids[kind] = ids
        cos_thr = math.cos(math.radians(args_cli.tilt_deg))
        dt_ctrl = float(u.cfg.sim.dt) * int(u.cfg.decimation)
        need = max(1, math.ceil(args_cli.trip_ms / 1000.0 / dt_ctrl))
        thresholds = sorted(args_cli.trip_nm)
        print(f"[probe] 제어 주기 {dt_ctrl * 1000:.1f} ms  ⇒ 트립까지 연속 **{need} 스텝** 필요"
              f"  ({args_cli.trip_ms:.0f} ms)")
        print(f"[probe] 평가할 채널 임계 {thresholds} N·m  → 관절 기준:")
        for nm, k in GEAR_K.items():
            js = "  ".join(f"{t * k:.1f}" for t in thresholds)
            print(f"         {nm:5s} ×{k}  =  {js}")

        obs = env.get_observations()
        for cmd_x in args_cli.cmd_x:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            rec: dict[str, list[torch.Tensor]] = {k: [] for k in (*GEAR_K, "gz", "vx")}
            with torch.inference_mode():
                for i in range(args_cli.steps):
                    u._commands[:] = cmd
                    obs, _, _, _ = env.step(policy(obs) * args_cli.act_scale)
                    u._commands[:] = cmd
                    if i < 100:
                        continue
                    for nm, ids in joint_ids.items():
                        rec[nm].append(robot.data.applied_torque[:, ids].abs().clone())
                    rec["gz"].append(robot.data.projected_gravity_b[:, 2:3].expand(-1, 2).clone())
                    rec["vx"].append(robot.data.root_lin_vel_b[:, 0:1].expand(-1, 2).clone())
            R = {k: torch.stack(v) for k, v in rec.items()}
            up = R["gz"] <= -cos_thr  # [T, N, 2] 아니라 [T, N*?]: 관절 열 수에 맞춰 이미 확장돼 있다

            vx = float(R["vx"][up].mean())
            print(f"\n{'=' * 92}\n[cmd vx = {cmd_x:.2f}  act_scale {args_cli.act_scale:.2f}]"
                  f"  기립 스텝 {float(up.float().mean()) * 100:.1f} %"
                  f"  실제 vx {vx:6.3f} m/s  달성률 {vx / cmd_x * 100 if cmd_x else float('nan'):5.0f} %")
            for nm in GEAR_K:
                tau = R[nm]
                flat = tau[up]  # 걷는 스텝만 — 넘어진 뒤의 토크는 실기에서 재현할 상황이 아니다
                t, n, j = tau.shape
                print(f"  {nm:5s} |tau| [N·m]  mean {float(flat.mean()):6.2f}  p95 {float(flat.quantile(0.95)):6.2f}"
                      f"  p99 {float(flat.quantile(0.99)):6.2f}  max {float(flat.max()):7.2f}"
                      f"   (관절-env 열 {n * j})")
                print(f"        {'채널':>6s} {'관절임계':>8s} {'초과율':>8s} {'런':>7s}"
                      f" {'최장':>6s}  {'≥' + str(need) + '스텝':>8s}  판정")
                for tnm in thresholds:
                    thr = tnm * GEAR_K[nm]
                    over = (tau > thr) & up
                    lens = run_lengths(over.reshape(t, n * j))
                    trips = lens[lens >= need]
                    longest = f"{int(lens.max()) * dt_ctrl * 1000:.0f}ms" if lens.numel() else "—"
                    verdict = f"★ 트립 {trips.numel()}" if trips.numel() else "통과"
                    print(f"        {tnm:6.1f} {thr:8.1f}"
                          f" {float(over.float().sum() / up.float().sum()) * 100:7.3f}%"
                          f" {lens.numel():7d} {longest:>6s}  {trips.numel():8d}  {verdict}")
        print(f"\n{'=' * 92}")
        env.close()


main()

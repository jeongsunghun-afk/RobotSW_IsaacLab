# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""처진 자세에서 보행을 개시할 수 있는가 — 리셋 노이즈 fine-tune 판정용 (2026-09-02).

왜 이 프로브가 따로 필요한가
----------------------------
`hindleg_gait_probe` · `hindleg_trip_probe` 는 둘 다 **앞 100 스텝을 과도 구간으로 버린다**.
정상 보행의 정상상태를 재는 것이 목적이기 때문이다. 그런데 이번 질문은 정확히 그 버려지는
구간에 있다 — "실기가 시작하는 자세에서 걷기 시작할 수 있는가".

실기가 어떤 자세에서 시작하는가는 실측돼 있다. 08-28 공중 고정 캡처에서 hip 이 중력으로 처져
HL 은 **+0.042 ~ +0.167 rad**, HR 은 **−0.235 ~ −0.061 rad** 에 앉아 있었다. 학습은 전 관절
정확히 0 에서만 시작했으므로 이 자세를 한 번도 본 적이 없다.
근거: reports/real2sim/_comparisons/bipedleg_policy_capture_20260828/README.md §3

무엇을 재나
-----------
과도 구간을 **버리지 않고** 리셋 이후 경과 시간(에피소드 나이)으로 정렬해서 본다::

    ① 개시 성공률   리셋 후 `--onset_s` 안에 vx ≥ `--onset_frac` × 명령 을 달성한 에피소드 비율
    ② 개시 지연     달성까지 걸린 시간의 분포
    ③ 낙상률        에피소드가 기울기 종료로 끝난 비율
    ④ 개시 구간 트립  첫 `--onset_s` 안에서 채널 토크 임계를 **연속 50 ms** 넘긴 런
    ⑤ hip 상태      개시 창 동안의 hip 각도 — 평균 · 창 끝 값 · soft limit 밀착 비율.
                    실기 증상이 "hip 이 처진 채 한계에 붙어 있다" 였으므로 좌우를 따로 본다.

★초기 자세는 `_reset_idx` 를 감싸 덮어쓴다. 리셋 **직후** 덮어써야 obs·history 가 그 자세에서
계산된다 — 리셋 뒤에 밖에서 덮어쓰면 첫 obs 만 옛 자세로 계산돼 조용히 어긋난다.
그리고 롤아웃 중 자동 리셋도 같은 자세에서 다시 시작하므로 **개시를 여러 번 표본**한다.

⚠ 이 프로브는 학습 조건이 아니라 **실기 조건**을 재현하는 것이므로, 리셋 노이즈는 run cfg 를
  따르지 않고 `--start_jitter` 로 명시한다(두 체크포인트에 같은 값을 준다).

실행::

    P=_workspace/hindleg_droop_start_probe.py
    B=logs/rsl_rl/hindLeg_history_direct/2026-08-28_09-39-20_gainclamp_ft/model_39799.pt
    N=logs/rsl_rl/hindLeg_history_direct/2026-09-02_09-07-14_resetnoise_ft/model_44799.pt
    for c in $B $N; do ./isaaclab.sh -p $P --checkpoint $c --start_pose real_hip; done
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg 처진 자세 기동 게이트.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=512)
parser.add_argument("--steps", type=int, default=1000, help="롤아웃 스텝 수 (자동 리셋으로 개시를 여러 번 표본한다).")
parser.add_argument("--cmd_x", type=float, nargs="+", default=[0.3, 0.5, 1.0])
parser.add_argument(
    "--start_pose", default="real_hip", choices=("default", "real_hip", "droop"),
    help="default=전 관절 0(종전 학습 조건) · real_hip=08-28 캡처 실측 hip 처짐 · "
    "droop=무토크 완전 처짐(RELAX_REST_POSE_SIM). 전부 soft limit 으로 자른다.",
)
parser.add_argument("--start_jitter", type=float, default=0.05, help="초기 자세에 더하는 균등노이즈 반폭 [rad].")
parser.add_argument("--onset_s", type=float, default=2.0, help="개시 판정 창 [s].")
parser.add_argument("--onset_frac", type=float, default=0.5, help="명령 대비 이 비율에 도달하면 개시 성공.")
parser.add_argument("--tilt_deg", type=float, default=50.0, help="낙상 판정 기울기 [deg].")
parser.add_argument("--trip_nm", type=float, nargs="+", default=[15.0, 25.0, 35.0], help="채널 기준 트립 임계.")
parser.add_argument("--trip_ms", type=float, default=50.0)
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
GEAR_K = {"hip": 1.0, "thigh": 1.0, "calf": 1.5, "foot": 1.2}

# 08-28 실기 캡처 hip 중앙값 근방 — HL 은 +0.16, HR 은 하한 −0.234 에 눌려 있었다.
# 나머지 관절은 그 캡처가 공중 고정이라 지면 보행의 시작 자세로 옮길 근거가 없어 0 으로 둔다.
REAL_HIP = {0: 0.160, 1: -0.234}
# 무토크로 늘어뜨린 실측 자세 (articulation 순서). `calib_bipedleg.hpp` RELAX_REST_POSE_SIM.
# ⚠ 그 상수는 zero_deg 미보정 프레임에서 캡처됐고 foot 은 공칭 soft limit 밖이다 — 여기서는
#   클램프해서 쓴다. 즉 "대략 이 방향"이지 정밀한 자세가 아니다.
DROOP = [0.205, -0.235, 0.621, -0.478, 0.856, -0.6267, -0.8285, 0.6000]


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
    order_s = starts[:, 1] * (t + 2) + starts[:, 0]
    order_e = ends[:, 1] * (t + 2) + ends[:, 0]
    return (order_e.sort().values - order_s.sort().values).to(torch.long)


def main() -> None:
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device

    # 플랜트 스위치는 run cfg 를 따른다 (소스 기본값으로 굴리면 다른 로봇이다).
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
    else:
        print(f"[probe] ⚠ {yml} 없음 — 플랜트 스위치를 소스 기본값으로 굴린다")
    # ★리셋 노이즈는 env 가 아니라 이 프로브가 건다 — 두 체크포인트에 **같은** 초기분포를 주기
    #   위해서다. run cfg 를 따르면 노이즈로 학습한 쪽만 더 넓은 분포에서 재게 된다.
    env_cfg.reset_joint_pos_noise = 0.0
    env_cfg.reset_base_rp_noise_deg = 0.0
    print(f"[probe] 플랜트: {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in TERM_KEYS)}}}"
          f"  cap={getattr(env_cfg, 'foot_reflected_inertia_cap', None)}")

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
        dev = u.device
        nj = robot.num_joints

        def t(x):
            return x.torch if hasattr(x, "torch") else x

        soft = t(robot.data.soft_joint_pos_limits).clone()
        base = torch.zeros(nj, device=dev)
        if args_cli.start_pose == "real_hip":
            for j, v in REAL_HIP.items():
                base[j] = v
        elif args_cli.start_pose == "droop":
            base = torch.tensor(DROOP, device=dev)
        names = robot.joint_names
        print(f"[probe] 초기 자세 '{args_cli.start_pose}' (jitter ±{args_cli.start_jitter} rad):")
        print("        " + "  ".join(f"{n.replace('_joint', '')}={float(base[j]):+.3f}" for j, n in enumerate(names)))

        # ★ _reset_idx 를 감싸 리셋 직후 초기 자세를 덮어쓴다. 리셋 밖에서 쓰면 첫 obs 가
        #   옛 자세로 계산돼 어긋난다. 자동 리셋에도 걸리므로 개시를 여러 번 표본한다.
        orig_reset_idx = u._reset_idx

        def reset_idx_drooped(env_ids):
            orig_reset_idx(env_ids)
            ids = env_ids
            if ids is None:
                ids = torch.arange(u.num_envs, device=dev)
            ids = torch.as_tensor(ids, device=dev).reshape(-1)
            n = ids.numel()
            q = base.unsqueeze(0).expand(n, nj).clone()
            if args_cli.start_jitter > 0.0:
                q = q + (torch.rand_like(q) * 2.0 - 1.0) * args_cli.start_jitter
            q = torch.clamp(q, soft[ids, :, 0], soft[ids, :, 1])
            robot.write_joint_state_to_sim_index(position=q, velocity=torch.zeros_like(q), env_ids=ids)

        u._reset_idx = reset_idx_drooped

        cos_thr = math.cos(math.radians(args_cli.tilt_deg))
        dt_ctrl = float(u.cfg.sim.dt) * int(u.cfg.decimation)
        onset_steps = max(1, int(round(args_cli.onset_s / dt_ctrl)))
        need = max(1, math.ceil(args_cli.trip_ms / 1000.0 / dt_ctrl))
        joint_ids = {k: robot.find_joints(f".*_{k}_joint", preserve_order=True)[0] for k in GEAR_K}
        hip_ids = joint_ids["hip"]

        print(f"[probe] 제어 {dt_ctrl * 1000:.1f} ms · 개시 창 {args_cli.onset_s:.1f} s = {onset_steps} 스텝"
              f" · 트립 연속 {need} 스텝")
        print(f"\n{'=' * 96}")
        print(f"checkpoint : {args_cli.checkpoint}")

        # ★ reset 은 반드시 `inference_mode` **안**에서 호출한다. 첫 롤아웃이
        #   `_action_noise_model._bias` 를 inference 텐서로 만들기 때문에, 밖에서 리셋하면
        #   그 텐서를 제자리 갱신하다 `RuntimeError: Inplace update to inference tensor` 로 죽는다.
        #   (2026-09-02 실측 — 이 때문에 첫 명령 하나만 측정되고 조용히 끝났다.)
        with torch.inference_mode():
            env.reset()
            obs = env.get_observations()

        for cmd_x in args_cli.cmd_x:
            cmd = torch.zeros_like(u._commands)
            cmd[:, 0] = cmd_x
            with torch.inference_mode():
                env.reset()
                obs = env.get_observations()
            age = torch.zeros(u.num_envs, dtype=torch.long, device=dev)
            # 에피소드별 집계 — 리셋될 때 확정하고 age 를 0 으로 되돌린다.
            onset_hit = torch.zeros(u.num_envs, dtype=torch.bool, device=dev)
            onset_step = torch.full((u.num_envs,), -1, dtype=torch.long, device=dev)
            ep_onset, ep_delay, ep_fell = [], [], []
            tau_win, up_win, hip_win, hip_end = [], [], [], []

            target = args_cli.onset_frac * cmd_x
            with torch.inference_mode():
                for _ in range(args_cli.steps):
                    u._commands[:] = cmd
                    obs, _, dones, _ = env.step(policy(obs))
                    u._commands[:] = cmd
                    vx = t(robot.data.root_lin_vel_b)[:, 0]
                    gz = t(robot.data.projected_gravity_b)[:, 2]
                    hipq = t(robot.data.joint_pos)[:, hip_ids]  # [N, 2] 좌우 따로

                    inwin = age < onset_steps
                    newly = inwin & ~onset_hit & (vx >= target)
                    onset_step = torch.where(newly, age, onset_step)
                    onset_hit |= newly
                    if bool(inwin.any()):
                        hip_win.append(torch.where(inwin.view(-1, 1), hipq, torch.full_like(hipq, float("nan"))))
                        hip_end.append(torch.where((age == onset_steps - 1).view(-1, 1), hipq,
                                                   torch.full_like(hipq, float("nan"))))

                    if bool(inwin.any()):
                        tau = torch.stack([t(robot.data.applied_torque)[:, joint_ids[k]].abs() for k in GEAR_K], 0)
                        tau_win.append(torch.where(inwin.view(1, -1, 1), tau, torch.zeros_like(tau)))
                        up_win.append(inwin)

                    d = dones.reshape(-1).bool()
                    if bool(d.any()):
                        ep_onset.append(onset_hit[d].clone())
                        ep_delay.append(onset_step[d].clone())
                        ep_fell.append((gz[d] > -cos_thr))
                        onset_hit[d] = False
                        onset_step[d] = -1
                        age[d] = -1
                    age += 1

            if not ep_onset:
                print(f"\n[cmd {cmd_x:.2f}] 종료된 에피소드가 없다 — steps 를 늘릴 것")
                continue
            oh = torch.cat(ep_onset)
            od = torch.cat(ep_delay).float()
            fell = torch.cat(ep_fell)
            ok_d = od[od >= 0] * dt_ctrl
            print(f"\n[cmd vx = {cmd_x:.2f} m/s]  종료 에피소드 {oh.numel()}")
            print(f"  ① 개시 성공률 (≥{target:.2f} m/s, {args_cli.onset_s:.0f}s 안)  {float(oh.float().mean()) * 100:5.1f} %")
            if ok_d.numel():
                print(f"  ② 개시 지연  중앙 {float(ok_d.median()):.2f} s  p90 {float(ok_d.quantile(0.9)):.2f} s")
            print(f"  ③ 낙상률 (종료 시 기울기 >{args_cli.tilt_deg:.0f}°)          {float(fell.float().mean()) * 100:5.1f} %")
            if hip_win:
                HW = torch.stack(hip_win)   # [T, N, 2]
                HE = torch.stack(hip_end)
                print(f"  ⑤ 개시 창 hip 상태 (시작 HL {float(base[hip_ids[0]]):+.3f} / HR {float(base[hip_ids[1]]):+.3f})")
                for c, jid in enumerate(hip_ids):
                    col = HW[..., c]
                    endc = HE[..., c]
                    lim = float(soft[0, jid, 1])
                    stick = (col.abs() >= 0.95 * lim).float()
                    ok = ~torch.isnan(col)
                    print(f"     {names[jid].replace('_joint', ''):8s} q mean {float(col[ok].mean()):+.4f}"
                          f"  창끝 {float(endc[~torch.isnan(endc)].mean()):+.4f}"
                          f"  한계(±{lim:.3f}) 밀착 {float(stick[ok].mean()) * 100:5.1f} %")
            if tau_win:
                T = torch.stack(tau_win)            # [T, kind, N, j]
                W = torch.stack(up_win)             # [T, N]
                print(f"  ④ 개시 창 토크 (채널 기준 임계, 연속 {need} 스텝 = {args_cli.trip_ms:.0f} ms)")
                for ki, nm in enumerate(GEAR_K):
                    tau = T[:, ki]                                   # [T, N, j]
                    m = W.unsqueeze(-1).expand_as(tau)
                    flat = tau[m]
                    row = f"     {nm:5s} |tau| mean {float(flat.mean()):5.2f} p99 {float(flat.quantile(0.99)):6.2f}  "
                    for thr_ch in sorted(args_cli.trip_nm):
                        over = (tau > thr_ch * GEAR_K[nm]) & m
                        lens = run_lengths(over.reshape(over.shape[0], -1))
                        row += f"| {thr_ch:.0f}N·m 트립 {int((lens >= need).sum()):4d} "
                    print(row)
        print(f"\n{'=' * 96}")
        env.close()


main()

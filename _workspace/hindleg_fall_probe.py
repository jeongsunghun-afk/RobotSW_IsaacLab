# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""HindLeg 넘어짐의 **전후 비대칭**과 그 자세에 머무는 시간을 계측한다.

`hind_leg_env._get_dones` 는 ``base.*`` 접촉(>1 N)만 종료 조건으로 쓴다. 앞으로 넘어지면
base 가 지면에 닿아 리셋되지만, 뒤로 넘어질 때 다리 링크가 몸통을 받치면 base 접촉이
안 잡혀 **에피소드가 안 끝난다**. 그러면 넘어진 자세가 학습 분포에 계속 남는다.

계측 (명령은 env 기본 랜덤 재샘플 그대로 — 학습 분포를 그대로 본다):
  · 넘어짐 판정: 기울기 tilt = arccos(−g_z) > ``--tilt_deg``
  · 전후 부호  : g_x > 0 = 앞으로(nose-down), g_x < 0 = 뒤로
  · 종료 여부  : ``reset_terminated`` (base 접촉) / ``reset_time_outs``
  · **에피소드당 넘어진 채 안 끝난 스텝 수** ← overfitting 주장의 핵심 수치
  · 넘어졌을 때 base / hip / thigh / calf 각 링크가 받는 접촉력 (기구적 원인 판별)

실행::

    python _workspace/hindleg_fall_probe.py --checkpoint <model.pt> --device cuda:1
"""

import argparse

from isaaclab_tasks.utils import add_launcher_args, launch_simulation, load_cfg_from_registry, setup_preset_cli

parser = argparse.ArgumentParser(description="HindLeg fall-asymmetry probe.")
parser.add_argument("--task", type=str, default="HindLeg-Direct-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=3000)
parser.add_argument("--warmup", type=int, default=200, help="초기 과도 구간 제외 [step].")
parser.add_argument("--tilt_deg", type=float, default=50.0, help="넘어짐 판정 기울기 [deg].")
add_launcher_args(parser)
args_cli, _ = setup_preset_cli(parser)
args_cli.headless = True

import inspect  # noqa: E402
import math  # noqa: E402

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
    #   **다른 플랜트**에서 측정된다 — `params/env.yaml` 에서 그 run 의 값을 그대로 복원한다.
    #   (같은 함정 기록: project_ramp_uses_source_cfg_not_run_params)
    import re
    from pathlib import Path

    ckpt = Path(args_cli.checkpoint).resolve()
    yml = ckpt.parent / "params" / "env.yaml"
    keys = ("foot_coupling", "foot_transpose", "foot_raw_friction", "foot_reflected_inertia")
    if yml.exists():
        text = yml.read_text()
        restored = {}
        for k in keys:
            m = re.search(rf"^{k}:\s*(true|false)\s*$", text, re.MULTILINE)
            if m:
                restored[k] = m.group(1) == "true"
                setattr(env_cfg, k, restored[k])
        print(f"[probe] run cfg 복원 ({yml}): {restored}")
    else:
        print(f"[probe] ⚠ {yml} 없음 — 소스 기본값으로 굴린다 (플랜트 불일치 가능)")
    print(f"[probe] 적용 플래그: {{{', '.join(f'{k}={getattr(env_cfg, k)}' for k in keys)}}}")

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
        cs = u._contact_sensor

        # --- 기구 확인: 종료 판정이 보는 링크 vs 페널티만 받는 링크 -------------------
        base_ids, base_names = cs.find_bodies("base.*")
        groups = {}
        for pat in ("base.*", ".*hip.*", ".*thigh.*", ".*calf.*", ".*foot.*"):
            ids, names = cs.find_bodies(pat)
            groups[pat] = (ids, names)
        print("\n================ HindLeg fall probe ================")
        print(f"checkpoint : {args_cli.checkpoint}")
        print(f"envs={args_cli.num_envs}  steps={args_cli.steps} (warmup {args_cli.warmup})  tilt>{args_cli.tilt_deg}°")
        print(f"\n★ 종료 판정 대상(_base_id, 'base.*') : {base_names}")
        for pat, (_, names) in groups.items():
            print(f"   {pat:12s} → {names}")
        print(f"   접촉 센서 전체 링크 {len(cs.body_names)}개: {cs.body_names}")

        default_z = float(robot.data.default_root_state[0, 2])
        cos_thr = math.cos(math.radians(args_cli.tilt_deg))
        obs = env.get_observations()

        rec = {k: [] for k in ("gx", "gz", "base_z", "term", "tout", "f_base", "f_leg")}
        leg_ids = sorted({i for pat in (".*hip.*", ".*thigh.*", ".*calf.*") for i in groups[pat][0]})
        # ⚠ 상태는 **step 직전**에 읽는다. ``DirectRLEnv.step`` 은 종료 판정 직후 그 안에서
        #   ``_reset_idx`` 를 부르므로, step 이후에 읽으면 종료된 env 의 자세·접촉력이 **리셋된
        #   스폰 자세**로 덮여 있다(넘어진 자세가 아니라). 그래서 pose[t] = 스텝 t 진입 상태,
        #   done[t] = 그 스텝의 종료 여부로 짝지어 기록한다 — 종료 시점보다 제어 1스텝(20 ms) 앞선다.
        with torch.inference_mode():
            for i in range(args_cli.steps):
                g = robot.data.projected_gravity_b
                pre = (
                    g[:, 0].clone(),
                    g[:, 2].clone(),
                    robot.data.root_link_pos_w[:, 2].clone(),
                )
                f = torch.norm(cs.data.net_forces_w_history, dim=-1).max(dim=1)[0]  # (N, B)
                pre_f = (f[:, base_ids].max(dim=-1)[0].clone(), f[:, leg_ids].max(dim=-1)[0].clone())
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
                if i < args_cli.warmup:
                    continue
                rec["gx"].append(pre[0])
                rec["gz"].append(pre[1])
                rec["base_z"].append(pre[2])
                rec["f_base"].append(pre_f[0])
                rec["f_leg"].append(pre_f[1])
                rec["term"].append(u.reset_terminated.clone())
                rec["tout"].append(u.reset_time_outs.clone())
        R = {k: torch.stack(v) for k, v in rec.items()}  # 각 (T, N)

        gx, gz, bz = R["gx"], R["gz"], R["base_z"]
        term, tout = R["term"], R["tout"]
        fallen = gz > -cos_thr  # tilt > 임계
        back, fwd = fallen & (gx < 0), fallen & (gx > 0)
        T, N = gx.shape

        print("\n--- 1. 자세 분포 (전 스텝) ---")
        print(f"  기립(tilt≤{args_cli.tilt_deg:.0f}°)  {(~fallen).float().mean() * 100:5.2f}%"
              f"   넘어짐 {fallen.float().mean() * 100:5.2f}%"
              f"  (뒤 {back.float().mean() * 100:5.2f}% / 앞 {fwd.float().mean() * 100:5.2f}%)")
        print(f"  base z  기립 {bz[~fallen].mean() if (~fallen).any() else float('nan'):.3f} m"
              f"   뒤로넘어짐 {bz[back].mean() if back.any() else float('nan'):.3f}"
              f"   앞으로넘어짐 {bz[fwd].mean() if fwd.any() else float('nan'):.3f}   (기본 스폰 {default_z:.3f})")

        # 기울기만으로는 "기울지 않고 주저앉은" 붕괴를 놓친다 — 높이 기준도 함께 본다.
        stand_z = bz[~fallen].median() if (~fallen).any() else torch.tensor(float(default_z))
        low = bz < 0.65 * stand_z
        lb, lf = low & (gx < 0), low & (gx > 0)
        print(f"  주저앉음(base z < {0.65 * stand_z:.3f} m = 기립 중앙값의 65%) {low.float().mean() * 100:5.2f}%"
              f"  (뒤 {lb.float().mean() * 100:5.2f}% / 앞 {lf.float().mean() * 100:5.2f}%)"
              f"   그중 종료율 뒤 {term[lb].float().mean() * 100 if lb.any() else float('nan'):6.3f}%"
              f" / 앞 {term[lf].float().mean() * 100 if lf.any() else float('nan'):6.3f}%")

        print("\n--- 2. 종료 사건의 자세 (★비대칭 판정) ---")
        n_term, n_tout = int(term.sum()), int(tout.sum())
        tb, tf = int((term & (gx < 0)).sum()), int((term & (gx > 0)).sum())
        print(f"  base 접촉 종료 {n_term:5d}건   그중 뒤로 {tb:5d} ({tb / max(n_term, 1) * 100:5.1f}%)"
              f" / 앞으로 {tf:5d} ({tf / max(n_term, 1) * 100:5.1f}%)")
        print(f"  time-out       {n_tout:5d}건   그중 넘어진 채 {int((tout & fallen).sum()):5d}"
              f" (뒤 {int((tout & back).sum()):4d} / 앞 {int((tout & fwd).sum()):4d})")
        for nm, m in (("뒤로 넘어짐", back), ("앞으로 넘어짐", fwd)):
            if not m.any():
                print(f"  {nm}: 표본 없음")
                continue
            print(f"  {nm} 중 종료율 {term[m].float().mean() * 100:6.3f}%"
                  f"   base 접촉력 mean {R['f_base'][m].mean():7.2f} N  p95 {R['f_base'][m].quantile(0.95):8.2f} N"
                  f"   다리링크 {R['f_leg'][m].mean():7.2f} N")

        print("\n--- 3. ★넘어진 채 안 끝난 시간 (overfitting 주장의 핵심) ---")
        # 리셋 경계로 나눠 에피소드별 누적. done 인 스텝에서 카운터를 비운다.
        done = term | tout
        cnt_b = torch.zeros(N, device=gx.device)
        cnt_f = torch.zeros(N, device=gx.device)
        ep_b, ep_f, ep_len, cur_len = [], [], [], torch.zeros(N, device=gx.device)
        for t in range(T):
            cnt_b += back[t].float()
            cnt_f += fwd[t].float()
            cur_len += 1
            d = done[t]
            if d.any():
                ep_b.append(cnt_b[d].clone())
                ep_f.append(cnt_f[d].clone())
                ep_len.append(cur_len[d].clone())
                cnt_b[d] = 0.0
                cnt_f[d] = 0.0
                cur_len[d] = 0.0
        if ep_b:
            eb, ef, el = torch.cat(ep_b), torch.cat(ep_f), torch.cat(ep_len)
            print(f"  완결 에피소드 {el.numel()}개, 평균 길이 {el.mean():7.1f} step ({el.mean() * u.step_dt:5.2f} s,"
                  f" 최대 {u.max_episode_length:.0f})")
            print(f"  에피소드당 뒤로 넘어진 채   mean {eb.mean():7.1f} step ({eb.mean() / el.mean() * 100:5.1f}% of ep)"
                  f"  p95 {eb.quantile(0.95):7.1f}  max {eb.max():7.1f}")
            print(f"  에피소드당 앞으로 넘어진 채 mean {ef.mean():7.1f} step ({ef.mean() / el.mean() * 100:5.1f}% of ep)"
                  f"  p95 {ef.quantile(0.95):7.1f}  max {ef.max():7.1f}")
            long_b = (eb > 100).float().mean() * 100
            print(f"  뒤로 100 step(2 s) 이상 방치된 에피소드 비율: {long_b:5.1f}%")
        else:
            print("  측정 구간에 완결된 에피소드 없음 — --steps 를 늘릴 것")

        print("\n--- 4. 기울기 분포 (g_x, 부호=전후) ---")
        for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99):
            print(f"   p{q * 100:4.0f}  g_x {gx.flatten().quantile(q):+7.4f}   g_z {gz.flatten().quantile(q):+7.4f}")
        print("====================================================\n")

        env.close()


if __name__ == "__main__":
    main()

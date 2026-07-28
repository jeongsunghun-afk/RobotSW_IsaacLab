# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Go2-Pedipulation-v0`` S1 게이트 평가 — G5(push 강건성) / G7(base 표류).

학습 로그가 아니라 **고정 시드·고정 명령**의 별도 패스에서 판정한다 (PLAN §5 공통 규약).

``--mode push`` (S1-G5)
    base 에 0.1 s 수평 외력을 가하고 낙상/추종 유지 여부를 본다. 힘 대역과 지속시간은
    Variable Stiffness 문헌 인용값(50~300 N, 0.1 s, 랜덤 시점)을 따른다.

    **4족 기준선의 정의는 PLAN 문언과 다르다.** ``leg_role`` 을 전부 1 로 두는 것이
    문자 그대로의 "4족" 이지만, 학습 중 조작 다리는 항상 1 개였으므로 all-ones 역할
    벡터는 분포 밖 입력이고 그렇게 얻은 곡선은 기준선 구실을 못 한다. 대신 **조작
    다리의 목표를 nominal 발 위치 근처로 주어 발이 지면에 남게** 한다. 명령 표현은
    학습 분포 안에 있으면서 지지 다각형만 4족으로 유지되므로, 3족 조건과의 차이가
    "지지 다각형 축소" 단독으로 분리된다. 조건이 실제로 성립했는지는 접촉 센서로
    측정한 지지 발 개수(``stance4_rate``)로 검증한다.

``--mode drift`` (S1-G7)
    외란 없이 3족 hold 를 유지하며 base xy 이동을 본다. PLAN §5 의 지시대로 **누적
    경로장(path length)과 순변위(net displacement)를 분리**해 로깅한다 — 자세를 한 번
    바꾸고 머무는 counterbalance 와 계속 밀려나는 진짜 표류는 순변위만으로 구분되지
    않는다.

사용 예::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_pedipulation_gates.py \
        --mode push --checkpoint logs/rsl_rl/go2_pedipulation/<run>/model_10600.pt \
        --num_envs 1024 --seeds 3 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2-Pedipulation-v0 S1 게이트(G5/G7) 평가.")
parser.add_argument("--checkpoint", type=str, required=True, help="model_*.pt 경로.")
parser.add_argument("--mode", type=str, default="push", choices=["push", "drift"], help="push=S1-G5, drift=S1-G7.")
parser.add_argument("--num_envs", type=int, default=1024, help="시드당 병렬 시행 수.")
parser.add_argument("--seeds", type=int, default=3, help="시드 개수 (PLAN §5: 최소 3).")
parser.add_argument("--seed0", type=int, default=1000, help="첫 시드 값. 시드는 seed0, seed0+1, ... 로 진행.")
parser.add_argument("--out", type=str, default=None, help="결과 JSON 저장 경로.")
parser.add_argument(
    "--stance4_z_jitter",
    type=float,
    default=0.01,
    help="4족 조건에서 목표 z 를 nominal 대비 흔드는 폭 [m]. stance4_rate 가 낮으면 --stance4_z_bias 로 눌러붙인다.",
)
parser.add_argument(
    "--stance4_z_bias", type=float, default=0.0, help="4족 조건 목표 z 의 nominal 대비 평균 오프셋 [m] (음수=지면 쪽)."
)
parser.add_argument("--no_domain_rand", action="store_true", help="DR 을 끄고 평가 (기본은 학습과 동일하게 ON).")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

TASK = "Go2-Pedipulation-v0"

# ── 시행 타임라인 (정책 50 Hz 기준 step) ─────────────────────────────────────
SETTLE_STEPS = 100  # 2.0 s — 발이 목표로 이동해 자리를 잡는 구간
PUSH_STEP = 125  # 2.5 s — 외력 인가 시점
PUSH_STEPS = 5  # 0.1 s — 문헌 인용 지속시간
PUSH_TRIAL_STEPS = 300  # 6.0 s — push 시행 총 길이 (인가 후 3.5 s 회복 관찰)
DRIFT_TRIAL_STEPS = 490  # 9.8 s — episode_length_s=10 의 타임아웃(499) 직전까지
DRIFT_CHECKPOINTS = (100, 200, 300, 400, 490)

# 힘 스윕 [N]. 0 은 대조군.
GATE_FORCES: tuple[float, ...] = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
CONDITIONS: tuple[str, ...] = ("stance4", "stance3")

# 성공 판정
SUCCESS_FOOT_ERR = 0.10  # [m] PLAN S1-G5: 발 오차 ≤ 0.1 m
FINAL_WINDOW = 25  # 마지막 0.5 s 평균으로 발 오차를 낸다


def build_env(seed: int):
    """평가용 env 를 만든다. 학습 cfg 에서 **평가에 해로운 동역학만** 끈다."""
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed

    # 명령이 시행 도중 재샘플되면 안 된다.
    # ⚠ min 만 키우면 (max-min) 이 음수가 되어 타이머가 즉시 만료된다. 반드시 둘 다 같은 값.
    env_cfg.command.resample_time_min = 1.0e6
    env_cfg.command.resample_time_max = 1.0e6
    # ⚠ 평가 중 커리큘럼이 승급하면 목표 분포가 런 도중 바뀐다. 승급 판정 자체를 봉쇄한다.
    env_cfg.command.curriculum_min_episodes = 10**9
    # 학습 커리큘럼이 도달한 최대 박스에서 평가한다.
    env_cfg.command.box_init = env_cfg.command.box_max

    if args_cli.no_domain_rand:
        env_cfg.domain_rand = False
    # 외란은 이 스크립트가 통제한다 — env 내장 랜덤 push/발 외력은 항상 끈다.
    env_cfg.dr.push_robot = False
    env_cfg.dr.push_foot_force = False

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")

    env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 정책 로드 (play.py 의 OnPolicyRunner 경로와 동일) ────────────────────
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    from rsl_rl.algorithms.ppo import PPO as _VendoredStockPPO

    _accepted = set(inspect.signature(_VendoredStockPPO.__init__).parameters.keys()) - {"self"}
    _cfg_dict = agent_cfg.to_dict()
    _cfg_dict["algorithm"] = {k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted or k == "class_name"}
    runner = OnPolicyRunner(env, _cfg_dict, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return env, policy


def assign_cells(n_env: int, device, generator: torch.Generator):
    """env 를 (다리 × 조건 × 힘) 셀에 균등 배정한다.

    셀을 라운드로빈으로 깔고 한 번 섞는다 — env 인덱스와 셀이 상관되면 (예: env 0 근처가
    항상 FL) DR 이나 초기 상태의 계통 편향이 특정 셀에 몰릴 수 있다.
    """
    n_leg, n_cond, n_force = 4, len(CONDITIONS), len(GATE_FORCES)
    n_cell = n_leg * n_cond * n_force
    cell = torch.arange(n_env, device=device) % n_cell
    cell = cell[torch.randperm(n_env, generator=generator, device=device)]
    leg_idx = cell // (n_cond * n_force)
    cond_idx = (cell // n_force) % n_cond
    force_idx = cell % n_force
    return leg_idx, cond_idx, force_idx


def build_targets(base_env, leg_idx, cond_idx, generator: torch.Generator):
    """조작 다리의 목표를 조건별로 만든다. 반환 shape [N, 4, 3] (base frame)."""
    device = base_env.device
    n = base_env.num_envs
    nominal = base_env._nominal_foot_pos_b.expand(n, -1, -1).clone()  # [N,4,3]
    box = torch.tensor(base_env.cfg.command.box_max, device=device)

    def u(lo, hi):
        return torch.rand(n, generator=generator, device=device) * (hi - lo) + lo

    # 3족: 학습 박스 안에서 뽑되 z 는 확실히 들어올리는 구간만 (z<0.10 이면 발이 안 뜬다).
    off3 = torch.stack([u(-box[0].item(), box[0].item()), u(-box[1].item(), box[1].item()), u(0.10, box[2].item())], -1)
    # 4족: nominal 근처 — 발이 지면에 남는다. z bias 로 지면 쪽으로 눌러붙일 수 있다.
    jit = args_cli.stance4_z_jitter
    off4 = torch.stack(
        [u(-0.05, 0.05), u(-0.05, 0.05), u(args_cli.stance4_z_bias - jit, args_cli.stance4_z_bias + jit)], -1
    )

    is3 = (cond_idx == CONDITIONS.index("stance3")).unsqueeze(-1).float()
    off = is3 * off3 + (1.0 - is3) * off4  # [N,3]

    target = nominal
    rows = torch.arange(n, device=device)
    target[rows, leg_idx] = nominal[rows, leg_idx] + off
    return target


def patch_command(base_env, leg_idx, target_b):
    """``_resample_command`` 를 고정 명령으로 대체한다.

    monkeypatch 로 명령 생성 자체를 고정하면, 조기 종료 후 자동 리셋된 env 도 **같은 셀의
    명령**을 다시 받는다 (그 env 는 집계에서 배제하지만, 명령이 재추첨되어 남은 시행의
    분포를 흔드는 일은 막아야 한다).
    """
    device = base_env.device
    n_leg = base_env.cfg.num_legs

    def fixed_resample(env_ids: torch.Tensor):
        if env_ids is None or int(env_ids.numel()) == 0:
            return
        ids = env_ids.to(torch.long)
        role = torch.ones(int(ids.numel()), n_leg, device=device)
        role[torch.arange(int(ids.numel()), device=device), leg_idx[ids]] = 0.0
        base_env._leg_role[ids] = role
        base_env._foot_target_b[ids] = target_b[ids]
        base_env._traj_center_b[ids] = target_b[ids]
        base_env._traj_phase[ids] = 0.0
        base_env._hold_counter[ids] = 0.0
        base_env._cmd_timer[ids] = 1.0e6

    base_env._resample_command = fixed_resample


def apply_push(base_env, force_mag: torch.Tensor, angle: torch.Tensor):
    """base 에 수평 외력을 인가한다 (world frame)."""
    n = base_env.num_envs
    forces = torch.zeros(n, 1, 3, device=base_env.device)
    forces[:, 0, 0] = force_mag * torch.cos(angle)
    forces[:, 0, 1] = force_mag * torch.sin(angle)
    body_ids = torch.tensor([base_env._base_body_id], dtype=torch.long, device=base_env.device)
    composer = base_env._robot.permanent_wrench_composer
    composer.reset()
    composer.add_forces_and_torques_index(forces=forces, body_ids=body_ids, is_global=True)


def clear_push(base_env):
    base_env._robot.permanent_wrench_composer.reset()


def foot_err_of_manip(base_env, leg_idx) -> torch.Tensor:
    """조작 다리의 목표-현재 거리 [m]. shape [N]."""
    pos = base_env._compute_foot_pos_b()
    rows = torch.arange(base_env.num_envs, device=base_env.device)
    return torch.norm(base_env._foot_target_b[rows, leg_idx] - pos[rows, leg_idx], dim=-1)


def stance_count(base_env) -> torch.Tensor:
    """접촉 중인 발 개수 [N]. 조건(4족/3족)이 실제로 성립했는지 검증하는 데 쓴다."""
    f = torch.norm(base_env.contact_sensor.data.net_forces_w[:, base_env._foot_sensor_ids], dim=-1)
    return (f > 1.0).float().sum(dim=-1)


def base_xy(base_env) -> torch.Tensor:
    return base_env._robot.data.body_pos_w[:, base_env._base_body_id, :2].clone()


def run_push_seed(env, policy, seed: int) -> dict:
    """push 시행 1 seed 를 돌리고 env 별 결과를 반환한다."""
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # 1) 최초 리셋으로 nominal 발 위치를 확정시킨다 (첫 관측에서 캡처된다).
    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패 — 목표 기준이 잡히지 않았다."

    # 2) 셀 배정 → 목표 생성 → 명령 고정 → 재리셋(고정 명령으로 관측까지 새로 만든다)
    leg_idx, cond_idx, force_idx = assign_cells(base_env.num_envs, device, gen)
    target_b = build_targets(base_env, leg_idx, cond_idx, gen)
    patch_command(base_env, leg_idx, target_b)
    obs, _ = env.reset()

    force_mag = torch.tensor(GATE_FORCES, device=device)[force_idx]
    angle = torch.rand(base_env.num_envs, generator=gen, device=device) * 2.0 * math.pi

    n = base_env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=device)
    stance4_hits = torch.zeros(n, device=device)
    stance3_hits = torch.zeros(n, device=device)
    settle_steps = torch.zeros(n, device=device)
    manip_dz = torch.zeros(n, device=device)  # 조작 발 z − nominal z [m]
    manip_force = torch.zeros(n, device=device)  # 조작 발 접촉력 [N]
    err_accum = torch.zeros(n, device=device)
    err_steps = torch.zeros(n, device=device)
    peak_speed = torch.zeros(n, device=device)  # push 인가 검증용

    for step in range(PUSH_TRIAL_STEPS):
        if step == PUSH_STEP:
            apply_push(base_env, force_mag, angle)
        elif step == PUSH_STEP + PUSH_STEPS:
            clear_push(base_env)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        died = base_env._died.clone()
        # 시행 도중 한 번이라도 죽으면 그 env 는 이후 상태가 무의미하다 (자동 리셋됨).
        alive = alive & (~died)

        # 조건 검증: settle 구간의 지지 발 개수
        if SETTLE_STEPS // 2 <= step < SETTLE_STEPS:
            sc = stance_count(base_env)
            stance4_hits += (sc >= 4).float() * alive.float()
            stance3_hits += (sc == 3).float() * alive.float()
            settle_steps += alive.float()
            # 진단: "발이 떠 있다"와 "닿았지만 하중이 없다"는 전혀 다른 상태다.
            rows = torch.arange(n, device=device)
            fp = base_env._compute_foot_pos_b()[rows, leg_idx, 2]
            manip_dz += (fp - base_env._nominal_foot_pos_b[0, leg_idx, 2]) * alive.float()
            cf = torch.norm(base_env.contact_sensor.data.net_forces_w[:, base_env._foot_sensor_ids], dim=-1)
            manip_force += cf[rows, leg_idx] * alive.float()

        # push 인가 검증: 인가 직후 base 속도 피크
        if PUSH_STEP <= step < PUSH_STEP + 3 * PUSH_STEPS:
            spd = torch.norm(base_env._robot.data.root_lin_vel_w[:, :2], dim=-1)
            peak_speed = torch.maximum(peak_speed, spd * alive.float())

        # 최종 발 오차: 마지막 0.5 s 평균
        if step >= PUSH_TRIAL_STEPS - FINAL_WINDOW:
            err_accum += foot_err_of_manip(base_env, leg_idx) * alive.float()
            err_steps += alive.float()

        del dones

    final_err = err_accum / err_steps.clamp(min=1.0)
    success = alive & (final_err <= SUCCESS_FOOT_ERR)
    return {
        "leg": leg_idx.cpu(),
        "cond": cond_idx.cpu(),
        "force": force_mag.cpu(),
        "alive": alive.cpu(),
        "final_err": final_err.cpu(),
        "success": success.cpu(),
        "stance4_rate": (stance4_hits / settle_steps.clamp(min=1.0)).cpu(),
        "stance3_rate": (stance3_hits / settle_steps.clamp(min=1.0)).cpu(),
        "peak_speed": peak_speed.cpu(),
        "manip_dz": (manip_dz / settle_steps.clamp(min=1.0)).cpu(),
        "manip_force": (manip_force / settle_steps.clamp(min=1.0)).cpu(),
    }


def run_drift_seed(env, policy, seed: int) -> dict:
    """외란 없는 3족 hold 에서 base 표류를 측정한다 (S1-G7)."""
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패."

    n = base_env.num_envs
    leg_idx = torch.arange(n, device=device) % 4
    leg_idx = leg_idx[torch.randperm(n, generator=gen, device=device)]
    cond_idx = torch.full((n,), CONDITIONS.index("stance3"), dtype=torch.long, device=device)
    target_b = build_targets(base_env, leg_idx, cond_idx, gen)
    patch_command(base_env, leg_idx, target_b)
    obs, _ = env.reset()

    alive = torch.ones(n, dtype=torch.bool, device=device)
    xy0 = base_xy(base_env)
    xy_prev = xy0.clone()
    path_len = torch.zeros(n, device=device)
    net_at = {}
    path_at = {}
    # settle 이후 기준점 — 다리를 들면서 생기는 **일회성 counterbalance 오프셋**과 그 뒤로도
    # 계속 밀려나는 **진짜 표류**를 분리한다. 중앙값 추이로 추론하지 않고 env 별로 직접 잰다.
    xy_ref = xy0.clone()
    path_ref = torch.zeros(n, device=device)
    net_late_at = {}
    path_late_at = {}
    err_accum = torch.zeros(n, device=device)
    err_steps = torch.zeros(n, device=device)

    for step in range(DRIFT_TRIAL_STEPS):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        alive = alive & (~base_env._died.clone())

        xy = base_xy(base_env)
        path_len += torch.norm(xy - xy_prev, dim=-1) * alive.float()
        xy_prev = xy

        if step == SETTLE_STEPS - 1:
            xy_ref = xy.clone()
            path_ref = path_len.clone()

        if step >= SETTLE_STEPS:
            err_accum += foot_err_of_manip(base_env, leg_idx) * alive.float()
            err_steps += alive.float()

        s = step + 1
        if s in DRIFT_CHECKPOINTS:
            net_at[s] = torch.norm(xy - xy0, dim=-1).cpu().clone()
            path_at[s] = path_len.cpu().clone()
            net_late_at[s] = torch.norm(xy - xy_ref, dim=-1).cpu().clone()
            path_late_at[s] = (path_len - path_ref).cpu().clone()

        del dones

    return {
        "leg": leg_idx.cpu(),
        "alive": alive.cpu(),
        "net": {k: v for k, v in net_at.items()},
        "path": {k: v for k, v in path_at.items()},
        "net_late": {k: v for k, v in net_late_at.items()},
        "path_late": {k: v for k, v in path_late_at.items()},
        "hold_err": (err_accum / err_steps.clamp(min=1.0)).cpu(),
        "com_b": base_env._com_pos_b().mean(dim=0).cpu(),
    }


def pct(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float(), q).item())


def report_push(results: list[dict]) -> dict:
    leg = torch.cat([r["leg"] for r in results])
    cond = torch.cat([r["cond"] for r in results])
    force = torch.cat([r["force"] for r in results])
    success = torch.cat([r["success"] for r in results])
    alive = torch.cat([r["alive"] for r in results])
    s4 = torch.cat([r["stance4_rate"] for r in results])
    s3 = torch.cat([r["stance3_rate"] for r in results])
    peak = torch.cat([r["peak_speed"] for r in results])
    dz = torch.cat([r["manip_dz"] for r in results])
    mf = torch.cat([r["manip_force"] for r in results])

    out: dict = {"n_trials": int(success.numel()), "cells": [], "validity": {}, "gate": {}}

    i4, i3 = CONDITIONS.index("stance4"), CONDITIONS.index("stance3")
    m4, m3 = cond == i4, cond == i3
    out["validity"] = {
        "stance4_cond_mean_4foot_rate": float(s4[m4].mean().item()),
        "stance3_cond_mean_3foot_rate": float(s3[m3].mean().item()),
        "peak_speed_by_force": {
            f"{f:.0f}N": float(peak[force == f].mean().item()) for f in GATE_FORCES if (force == f).any()
        },
        "stance4_cond_manip_dz_m": float(dz[m4].mean().item()),
        "stance4_cond_manip_force_N": float(mf[m4].mean().item()),
        "stance3_cond_manip_dz_m": float(dz[m3].mean().item()),
        "stance3_cond_manip_force_N": float(mf[m3].mean().item()),
    }

    print("\n" + "=" * 78)
    print("S1-G5  push 강건성")
    print("=" * 78)
    print(f"{'force[N]':>9} | {'4족 성공률':>12} {'n':>6} | {'3족 성공률':>12} {'n':>6} | {'열화[%p]':>9}")
    print("-" * 78)
    for f in GATE_FORCES:
        row = {"force_N": f}
        for label, mask in (("stance4", m4), ("stance3", m3)):
            sel = mask & (force == f)
            k = int(sel.sum().item())
            rate = float(success[sel].float().mean().item()) if k else float("nan")
            row[f"{label}_rate"] = rate
            row[f"{label}_n"] = k
            row[f"{label}_fall_rate"] = float((~alive[sel]).float().mean().item()) if k else float("nan")
        row["degradation_pp"] = (row["stance4_rate"] - row["stance3_rate"]) * 100.0
        out["cells"].append(row)
        print(
            f"{f:9.0f} | {row['stance4_rate'] * 100:11.1f}% {row['stance4_n']:6d} |"
            f" {row['stance3_rate'] * 100:11.1f}% {row['stance3_n']:6d} | {row['degradation_pp']:9.1f}"
        )

    # 게이트: <150 N 대역 (0/50/100 N) 집계
    band = force < 150.0
    r4 = float(success[m4 & band].float().mean().item())
    r3 = float(success[m3 & band].float().mean().item())
    deg = (r4 - r3) * 100.0
    out["gate"] = {
        "band": "<150 N",
        "stance4_rate": r4,
        "stance3_rate": r3,
        "degradation_pp": deg,
        "pass_stance4_ge_0.95": r4 >= 0.95,
        "pass_stance3_ge_0.70": r3 >= 0.70,
        "pass_degradation_le_25pp": deg <= 25.0,
    }
    print("-" * 78)
    print(
        f"<150 N 대역:  4족 {r4 * 100:.1f}% (기준 ≥95%)  3족 {r3 * 100:.1f}% (기준 ≥70%)  열화 {deg:.1f}%p (기준 ≤25%p)"
    )
    print(f"조건 검증:  4족 조건의 4-foot 접지 비율 {out['validity']['stance4_cond_mean_4foot_rate'] * 100:.1f}%")
    print(f"            3족 조건의 3-foot 접지 비율 {out['validity']['stance3_cond_mean_3foot_rate'] * 100:.1f}%")
    print(
        f"            4족 조건 조작발: Δz {out['validity']['stance4_cond_manip_dz_m'] * 100:+.1f} cm,"
        f" 접촉력 {out['validity']['stance4_cond_manip_force_N']:.2f} N"
    )
    print(
        f"            3족 조건 조작발: Δz {out['validity']['stance3_cond_manip_dz_m'] * 100:+.1f} cm,"
        f" 접촉력 {out['validity']['stance3_cond_manip_force_N']:.2f} N"
    )
    print(f"push 인가 검증 (base 속도 피크): {out['validity']['peak_speed_by_force']}")

    # 다리별 분해 — 앞/뒤 다리는 nominal x 가 비대칭이라 난이도가 다를 수 있다.
    leg_names = ("FL", "FR", "RL", "RR")
    out["per_leg"] = {}
    print(f"\n다리별 (<150 N 대역, 3족): {'':>4}", end="")
    for li, name in enumerate(leg_names):
        sel = m3 & band & (leg == li)
        rate = float(success[sel].float().mean().item()) if int(sel.sum().item()) else float("nan")
        out["per_leg"][name] = {"stance3_rate": rate, "n": int(sel.sum().item())}
        print(f"{name} {rate * 100:.1f}%  ", end="")
    print()
    return out


def report_drift(results: list[dict]) -> dict:
    alive = torch.cat([r["alive"] for r in results])
    hold_err = torch.cat([r["hold_err"] for r in results])
    out: dict = {"n_trials": int(alive.numel()), "alive_rate": float(alive.float().mean().item()), "steps": {}}

    print("\n" + "=" * 78)
    print("S1-G7  base 표류 (외란 없음, 3족 hold)")
    print("=" * 78)
    print(f"낙상 없이 완주: {out['alive_rate'] * 100:.1f}%   hold 중 발 오차 중앙값 {pct(hold_err[alive], 0.5):.4f} m")
    print(
        f"\n{'t[s]':>6} | {'순변위 중앙':>11} {'순변위 95p':>11} | {'경로장 중앙':>11} {'경로장 95p':>11}"
        f" | {'2s후 중앙':>11} {'2s후 95p':>11}"
    )
    print("-" * 78)
    for s in DRIFT_CHECKPOINTS:
        net = torch.cat([r["net"][s] for r in results])[alive]
        path = torch.cat([r["path"][s] for r in results])[alive]
        net_late = torch.cat([r["net_late"][s] for r in results])[alive]
        path_late = torch.cat([r["path_late"][s] for r in results])[alive]
        rec = {
            "net_median": pct(net, 0.5),
            "net_p95": pct(net, 0.95),
            "path_median": pct(path, 0.5),
            "path_p95": pct(path, 0.95),
            "net_late_median": pct(net_late, 0.5),
            "net_late_p95": pct(net_late, 0.95),
            "path_late_median": pct(path_late, 0.5),
        }
        out["steps"][str(s)] = rec
        print(
            f"{s / 50.0:6.1f} | {rec['net_median']:11.4f} {rec['net_p95']:11.4f} |"
            f" {rec['path_median']:11.4f} {rec['path_p95']:11.4f} |"
            f" {rec['net_late_median']:11.4f} {rec['net_late_p95']:11.4f}"
        )

    final = out["steps"][str(DRIFT_CHECKPOINTS[-1])]
    out["gate"] = {
        "net_median_m": final["net_median"],
        "net_p95_m": final["net_p95"],
        "pass_median_le_0.05": final["net_median"] <= 0.05,
        "pass_p95_le_0.12": final["net_p95"] <= 0.12,
        # settle(2.0 s) 이후로 추가로 밀려난 양 — 일회성 counterbalance 를 제외한 "진짜 표류".
        "net_late_median_m": final["net_late_median"],
        "net_late_p95_m": final["net_late_p95"],
    }
    print("-" * 78)
    print(f"게이트(원안): 중앙 {final['net_median']:.4f} m (기준 ≤0.05)  95p {final['net_p95']:.4f} m (기준 ≤0.12)")
    print(
        f"settle 이후 추가 이동: 중앙 {final['net_late_median']:.4f} m  95p {final['net_late_p95']:.4f} m"
        "   ← 일회성 counterbalance 를 제외한 순수 표류"
    )

    # ── counterbalance 가설의 검증 ────────────────────────────────────────────
    # nominal 발 배치에서 CoM 은 **뒷다리를 들 때는 이미 지지 삼각형 안**이고, 앞다리를 들 때만
    # 밖이다. 초기 변위가 counterbalance 라면 앞다리 거상의 변위가 뒷다리보다 커야 한다.
    # 두 값이 같다면 counterbalance 설명은 성립하지 않는다.
    leg = torch.cat([r["leg"] for r in results])[alive]
    net_final = torch.cat([r["net"][DRIFT_CHECKPOINTS[-1]] for r in results])[alive]
    leg_names = ("FL", "FR", "RL", "RR")
    out["per_leg_net"] = {}
    print("\n다리별 순변위 중앙값 [m] — counterbalance 가설 검증:")
    for li, name in enumerate(leg_names):
        sel = leg == li
        val = pct(net_final[sel], 0.5)
        out["per_leg_net"][name] = {"net_median": val, "n": int(sel.sum().item())}
        print(f"    {name}: {val:.4f}  (n={int(sel.sum().item())})")
    com = results[0]["com_b"]
    out["com_b"] = [float(v) for v in com]
    print(f"    CoM(base frame) 실측: ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}) m")
    return out


def main():
    results = []
    for i in range(args_cli.seeds):
        seed = args_cli.seed0 + i
        print(f"\n########## seed {seed} ({i + 1}/{args_cli.seeds}) ##########")
        env, policy = build_env(seed)
        try:
            if args_cli.mode == "push":
                results.append(run_push_seed(env, policy, seed))
            else:
                results.append(run_drift_seed(env, policy, seed))
        finally:
            env.close()

    summary = report_push(results) if args_cli.mode == "push" else report_drift(results)
    summary["config"] = {
        "checkpoint": args_cli.checkpoint,
        "mode": args_cli.mode,
        "num_envs": args_cli.num_envs,
        "seeds": args_cli.seeds,
        "seed0": args_cli.seed0,
        "domain_rand": not args_cli.no_domain_rand,
    }
    if args_cli.out:
        with open(args_cli.out, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()

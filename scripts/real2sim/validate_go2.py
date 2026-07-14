# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 PACE 식별 결과 검증 — hold-out 포함.

식별된 49개 파라미터를 시뮬레이터에 넣고 녹화 시퀀스를 재생해, 실측 궤적을 얼마나 재현하는지
관절별 RMSE로 잰다. **핵심은 hold-out** — 적합에 쓰지 않은 시퀀스(보지 않은 PD 게인 / 진폭 /
주파수)에서도 재현되어야 과적합이 아니다. 논문의 전신 단계 검증도 이 방식이다:

    "fitted simulators reproduce in-air trajectories at unseen PD gains and for unseen trajectories
     with higher fidelity than the actuator-network baseline, indicating a consistent physical model
     rather than an overfit."

환경을 **2개** 띄워 같은 명령을 동시에 먹인다:

* env 0 — 식별된 파라미터
* env 1 — nominal (현재 ``UNITREE_GO2_CFG``: armature 0.01, 마찰 0, 바이어스 0, 지연 0)

따라서 "얼마나 잘 맞나"뿐 아니라 **"지금 cfg 대비 얼마나 나아졌나"**가 한 번에 나온다.

실행::

    python scripts/real2sim/validate_go2.py --headless          # logs/pace/<robot>의 최신 결과 사용
    python scripts/real2sim/validate_go2.py --headless --params logs/pace/go2_sim/26_07_13_.../mean_120.pt
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="R2S-GO2 PACE identification validation (incl. hold-out).")
parser.add_argument("--task", type=str, default="Isaac-R2S-Go2-Sysid-v0", help="Name of the task.")
parser.add_argument("--params", type=str, default=None, help="Path to a mean_XXX.pt. Defaults to the latest run.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from pathlib import Path

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from isaaclab_tasks.direct.r2s_go2.r2s_go2_sysid_cfg import (  # isort: skip
    SYNTHETIC_GT_ARMATURE,
    SYNTHETIC_GT_BIAS,
    SYNTHETIC_GT_COULOMB,
    SYNTHETIC_GT_DELAY,
    SYNTHETIC_GT_VISCOUS,
)
from pace_sim2real.optim import MultiTrajectoryCMAES  # isort: skip
from pace_sim2real.utils import project_root, require_physx_backend  # isort: skip

FITTED, NOMINAL = 0, 1  # env 인덱스


def latest_params(log_root: Path) -> Path:
    """가장 최근 run 폴더에서 가장 마지막 mean_XXX.pt 경로를 찾는다.

    Args:
        log_root: ``logs/pace/<robot_name>`` 경로.

    Returns:
        체크포인트 경로.

    Raises:
        FileNotFoundError: run 폴더나 체크포인트가 없을 때.
    """
    runs = sorted(p for p in log_root.glob("*") if p.is_dir())
    if not runs:
        raise FileNotFoundError(f"적합 로그가 없다: {log_root}")
    checkpoints = sorted(runs[-1].glob("mean_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"체크포인트가 없다: {runs[-1]}")
    return checkpoints[-1]


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=2)
    env = gym.make(args_cli.task, cfg=env_cfg)
    require_physx_backend(env)

    device = env.unwrapped.device
    articulation = env.unwrapped.scene["robot"]
    sim2real = env_cfg.sim2real
    joint_order = sim2real.joint_order
    n = len(joint_order)

    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], dtype=torch.int32, device=device
    )
    joint_ids_long = joint_ids.to(torch.long)
    env_ids = torch.arange(2, dtype=torch.int32, device=device)

    # ------------------------------------------------------------------
    # 식별 파라미터 로드 (mean_XXX.pt = 물리 단위 49-벡터)
    # ------------------------------------------------------------------
    data_root = project_root() / "data"
    log_root = project_root() / "logs" / "pace" / sim2real.robot_name
    params_path = project_root() / args_cli.params if args_cli.params else latest_params(log_root)
    fitted = torch.load(params_path, map_location=device).to(torch.float32).reshape(-1)
    if fitted.numel() != 4 * n + 1:
        raise RuntimeError(f"{params_path.name}의 파라미터 수가 {fitted.numel()}개다 (기대 {4 * n + 1}).")
    print(f"[INFO]: 식별 파라미터: {params_path}")

    f_armature, f_viscous, f_coulomb, f_bias = (fitted[i * n : (i + 1) * n] for i in range(4))
    f_delay = int(round(float(fitted[4 * n])))

    # env 0 = 식별값, env 1 = nominal(UNITREE_GO2_CFG 그대로)
    armature = torch.stack([f_armature, torch.full((n,), 0.01, device=device)])
    viscous = torch.stack([f_viscous, torch.zeros(n, device=device)])
    coulomb = torch.stack([f_coulomb, torch.zeros(n, device=device)])
    bias = torch.stack([f_bias, torch.zeros(n, device=device)])
    delay = torch.tensor([f_delay, 0], dtype=torch.int, device=device)

    # 합성 데이터라면 GT와 직접 대조한다 (Phase 1 자기복원 판정).
    if sim2real.robot_name == "go2_sim":
        gt = {
            "armature": (SYNTHETIC_GT_ARMATURE, f_armature),
            "viscous": (SYNTHETIC_GT_VISCOUS, f_viscous),
            "Coulomb": (SYNTHETIC_GT_COULOMB, f_coulomb),
            "bias": (SYNTHETIC_GT_BIAS, f_bias),
        }
        print("\n=== 파라미터 복원 (GT 대비) ===")
        print(f"{'파라미터':10s} {'관절타입':8s} {'GT':>9s} {'식별':>9s} {'오차':>9s}")
        for name, (gt_vec, fit_vec) in gt.items():
            for k, jt in enumerate(["hip", "thigh", "calf"]):
                # 관절 타입별로 4개 다리 평균 (GT는 다리마다 동일)
                idx = [k + 3 * leg for leg in range(4)]
                g = gt_vec[k]
                f = float(fit_vec[idx].mean())
                print(f"{name:10s} {jt:8s} {g:9.4f} {f:9.4f} {f - g:+9.4f}")
        print(f"{'delay':10s} {'global':8s} {SYNTHETIC_GT_DELAY:9d} {f_delay:9d} {f_delay - SYNTHETIC_GT_DELAY:+9d}")

    # ------------------------------------------------------------------
    # 시퀀스별 재생 → RMSE
    # ------------------------------------------------------------------
    fit_sets = [(rel, "fit") for rel in sim2real.datasets]
    holdout_sets = [(rel, "HOLD-OUT") for rel in sim2real.holdout]
    if not holdout_sets:
        print("[WARN]: hold-out 시퀀스가 없다. 과적합 여부를 판정할 수 없다.")

    results = []
    for rel, kind in fit_sets + holdout_sets:
        raw = torch.load(data_root / rel)
        measured = raw["dof_pos"].to(device)  # 엔코더 읽음값 (실제각 − bias)
        target = raw["des_dof_pos"].to(device)
        kp = raw["kp"].to(device).reshape(n)
        kd = raw["kd"].to(device).reshape(n)
        num_steps = measured.shape[0]

        with torch.inference_mode():
            env.reset()
            MultiTrajectoryCMAES.apply_gains(articulation, joint_ids, kp, kd)

            articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=coulomb,
                joint_dynamic_friction_coeff=coulomb,
                joint_viscous_friction_coeff=viscous,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
            for actuator in articulation.actuators.values():
                drive_ids = actuator.joint_indices
                if isinstance(drive_ids, slice):
                    drive_ids = torch.arange(articulation.num_joints, device=device)[drive_ids]
                drive_ids = torch.as_tensor(drive_ids, device=device).to(torch.long)
                order_pos = torch.tensor(
                    [int((joint_ids_long == int(j)).nonzero()[0]) for j in drive_ids], dtype=torch.long, device=device
                )
                actuator.update_encoder_bias(bias[:, order_pos])
                actuator.update_time_lags(delay)
                actuator.reset(env_ids)

            # 두 env 모두 엔코더가 measured[0]을 읽도록 초기화 → 실제각 = measured[0] + 각자의 bias
            init_true = torch.zeros(2, articulation.num_joints, device=device)
            init_true[:, joint_ids_long] = measured[0].unsqueeze(0) + bias
            articulation.write_joint_position_to_sim_index(position=init_true)
            articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(init_true))

            actions = torch.zeros(2, articulation.num_joints, device=device)
            sim_enc = torch.zeros(num_steps, 2, n, device=device)
            for t in range(num_steps):
                # 엔코더 읽음값 = 실제 관절각 − 해당 env의 bias (실기 기록 규약과 동일)
                sim_enc[t] = articulation.data.joint_pos[:, joint_ids_long] - bias
                actions[:, joint_ids_long] = target[t].unsqueeze(0)
                env.step(actions)

        err = sim_enc - measured.unsqueeze(1)  # (T, 2, n)
        rmse = torch.sqrt((err**2).mean(dim=0))  # (2, n)
        results.append(
            {
                "name": rel.split("/")[-1],
                "kind": kind,
                "kp": float(kp[0]),
                "kd": float(kd[0]),
                "rmse_fitted": float(rmse[FITTED].mean()),
                "rmse_nominal": float(rmse[NOMINAL].mean()),
                "max_fitted": float(err[:, FITTED].abs().max()),
                "per_joint": rmse[FITTED],
            }
        )

    env.close()

    # ------------------------------------------------------------------
    # 보고
    # ------------------------------------------------------------------
    print("\n=== 궤적 재현 RMSE [rad] ===")
    print(
        f"{'시퀀스':20s} {'구분':9s} {'kp':>5s} {'kd':>5s} {'식별':>9s} {'nominal':>9s} {'개선':>8s} {'식별 max':>9s}"
    )
    for r in results:
        ratio = r["rmse_nominal"] / r["rmse_fitted"] if r["rmse_fitted"] > 0 else float("inf")
        print(
            f"{r['name']:20s} {r['kind']:9s} {r['kp']:5.0f} {r['kd']:5.2f} "
            f"{r['rmse_fitted']:9.5f} {r['rmse_nominal']:9.5f} {ratio:7.1f}x {r['max_fitted']:9.5f}"
        )

    print("\n=== hold-out 관절별 RMSE [rad] ===")
    for r in results:
        if r["kind"] != "HOLD-OUT":
            continue
        for i, name in enumerate(joint_order):
            print(f"  {name:18s} {float(r['per_joint'][i]):.5f}")

    hold = [r for r in results if r["kind"] == "HOLD-OUT"]
    fitr = [r for r in results if r["kind"] == "fit"]
    if hold and fitr:
        mean_fit = sum(r["rmse_fitted"] for r in fitr) / len(fitr)
        mean_hold = sum(r["rmse_fitted"] for r in hold) / len(hold)
        print(f"\n적합 시퀀스 평균 RMSE : {mean_fit:.5f} rad")
        print(f"hold-out 평균 RMSE    : {mean_hold:.5f} rad  (적합 대비 {mean_hold / mean_fit:.2f}배)")
        print("hold-out이 적합 시퀀스와 비슷한 수준이면 과적합이 아니라 일관된 물리 모델이다(논문 판정 기준).")


if __name__ == "__main__":
    main()
    simulation_app.close()

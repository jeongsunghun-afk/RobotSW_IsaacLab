# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg PACE 식별 결과 검증 — hold-out + 합성 GT 자기복원 판정.

식별된 33개 파라미터(armature 8 + viscous 8 + coulomb 8 + bias 8 + delay 1)를 시뮬레이터에 넣고
녹화 시퀀스를 재생해, 실측 궤적을 얼마나 재현하는지 관절별 RMSE로 잰다. **핵심은 hold-out** —
적합에 쓰지 않은 시퀀스(보지 않은 PD 게인 / 진폭)에서도 재현되어야 과적합이 아니다. 논문의 전신
단계 검증도 이 방식이다:

    "fitted simulators reproduce in-air trajectories at unseen PD gains and for unseen trajectories
     with higher fidelity than the actuator-network baseline, indicating a consistent physical model
     rather than an overfit."

환경을 **2개** 띄워 같은 명령을 동시에 먹인다:

* env 0 — 식별된 파라미터
* env 1 — nominal (``HIND_LEG_CFG`` 기본값: 마찰 0, 바이어스 0, 지연 0, armature는 cfg/USD 기본값)

따라서 "얼마나 잘 맞나"뿐 아니라 **"지금 cfg 대비 얼마나 나아졌나"**가 한 번에 나온다.

합성 데이터(``robot_name == "bipedleg_sim"``)에는 정답이 있으므로 파라미터별 GT 대조표와
**PASS/FAIL 판정**을 함께 출력한다. 마지막 줄은 ``VERDICT:``로 시작한다 (grep 용).

⚠ 모델 갭: 여기서 검증하는 플랜트는 :class:`PaceDCMotor`(토크-속도 포화)다. ``direct/hind_leg``
RL env는 :class:`ImplicitActuator`를 쓰므로, 식별값을 그대로 옮기면 갭이 남는다.

실행::

    python scripts/real2sim/validate_bipedleg.py --headless          # logs/pace/<robot>의 최신 결과 사용
    python scripts/real2sim/validate_bipedleg.py --headless --params logs/pace/bipedleg_sim/26_07_21_.../mean_120.pt
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="R2S-BipedLeg PACE identification validation (incl. hold-out).")
parser.add_argument("--task", type=str, default="Isaac-R2S-BipedLeg-Sysid-v0", help="Name of the task.")
parser.add_argument("--params", type=str, default=None, help="Path to a mean_XXX.pt. Defaults to the latest run.")
# ---- 합격 기준 (기본값 선정 근거는 아래 "합격 기준 근거" 주석 참고) ----
parser.add_argument("--min_improvement", type=float, default=2.0, help="hold-out RMSE의 nominal 대비 최소 개선 배율.")
parser.add_argument("--max_holdout_rmse", type=float, default=0.02, help="hold-out 평균 RMSE 상한 [rad].")
parser.add_argument(
    "--max_gt_rel_err", type=float, default=0.20, help="armature/viscous/coulomb GT 상대오차 상한 (0.20 = 20%%)."
)
parser.add_argument("--max_bias_abs_err", type=float, default=0.005, help="엔코더 바이어스 GT 절대오차 상한 [rad].")
parser.add_argument("--max_delay_err", type=int, default=1, help="지연 GT 오차 상한 [sim step].")
parser.add_argument(
    "--nominal_armature",
    type=float,
    default=None,
    help="nominal env(env 1)의 armature [kg·m²]. 미지정 시 로드된 cfg/USD 기본값을 그대로 쓴다.",
)
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

from isaaclab_tasks.direct.r2s_biped_leg.r2s_biped_leg_sysid_cfg import (  # isort: skip
    SYNTHETIC_GT_ARMATURE,
    SYNTHETIC_GT_BIAS,
    SYNTHETIC_GT_COULOMB,
    SYNTHETIC_GT_DELAY,
    SYNTHETIC_GT_VISCOUS,
)
from pace_sim2real.optim import MultiTrajectoryCMAES  # isort: skip
from pace_sim2real.utils import project_root, require_physx_backend  # isort: skip

FITTED, NOMINAL = 0, 1  # env 인덱스

# 표/로그용 짧은 관절 레이블 (cfg의 joint_order와 동일 순서, leg-major).
JOINT_LABELS = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]

# 합격 기준 근거:
#   * min_improvement=2.0 — 합성 GT 자기복원이라면 nominal 대비 한 자릿수 이상 좋아야 정상이다.
#     2배는 "파이프라인이 살아 있다"는 하한선이지 목표치가 아니다 (go2 합성 게이트는 수천 배 나왔다).
#   * max_holdout_rmse=0.02 rad ≈ 1.1° — chirp 진폭 0.45~0.85 rad의 ~2~4% 수준. 이보다 크면
#     플랜트 모델이 궤적을 설명하지 못한다는 뜻.
#   * max_gt_rel_err=0.20 — armature/viscous/coulomb는 서로 일부 흡수 관계라(특히 viscous↔kd,
#     viscous↔Coulomb 저속 구간) 정확히 일치할 수 없다. 20%면 "자릿수 복원 + 관절별 서열 보존" 판정.
#   * max_bias_abs_err=0.005 rad — bias GT가 0.01~0.04 rad 규모라 상대오차는 의미가 없다. 절대값으로 본다.
#   * max_delay_err=1 step (=2 ms @500 Hz) — delay는 정수 격자라 ±1 step 양자화 오차를 허용한다.


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

    # nominal armature. HIND_LEG_CFG는 armature를 명시하지 않으므로 "현재 cfg 값" = USD/actuator cfg
    # 기본값이다. 하드코딩하지 않고 아무것도 쓰기 전에 스냅샷해서 쓴다 (--nominal_armature로 덮어쓰기 가능).
    if args_cli.nominal_armature is not None:
        nominal_armature = torch.full((n,), args_cli.nominal_armature, device=device)
    else:
        nominal_armature = articulation.data.joint_armature[NOMINAL, joint_ids_long].clone().to(torch.float32)
    print(f"[INFO]: nominal armature [kg·m²]: {[round(float(v), 5) for v in nominal_armature]}")

    # ------------------------------------------------------------------
    # 식별 파라미터 로드 (mean_XXX.pt = 물리 단위 33-벡터)
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

    # env 0 = 식별값, env 1 = nominal(HIND_LEG_CFG 그대로: 마찰 0, 바이어스 0, 지연 0)
    armature = torch.stack([f_armature, nominal_armature])
    viscous = torch.stack([f_viscous, torch.zeros(n, device=device)])
    coulomb = torch.stack([f_coulomb, torch.zeros(n, device=device)])
    bias = torch.stack([f_bias, torch.zeros(n, device=device)])
    delay = torch.tensor([f_delay, 0], dtype=torch.int, device=device)

    # ------------------------------------------------------------------
    # 합성 데이터라면 GT와 직접 대조한다 (자기복원 게이트).
    # ------------------------------------------------------------------
    gt_pass = None
    if sim2real.robot_name == "bipedleg_sim":
        # (이름, GT 리스트, 식별 벡터, 상대오차로 판정할지 여부)
        gt_specs = [
            ("armature", SYNTHETIC_GT_ARMATURE, f_armature, True),
            ("viscous", SYNTHETIC_GT_VISCOUS, f_viscous, True),
            ("Coulomb", SYNTHETIC_GT_COULOMB, f_coulomb, True),
            ("bias", SYNTHETIC_GT_BIAS, f_bias, False),
        ]
        print("\n=== 파라미터 복원 (합성 GT 대비) ===")
        print(f"{'파라미터':10s} {'관절':10s} {'GT':>10s} {'식별':>10s} {'절대':>10s} {'상대':>10s} {'판정':>6s}")
        gt_pass = True
        for name, gt_vec, fit_vec, use_rel in gt_specs:
            for i, label in enumerate(JOINT_LABELS):
                g = float(gt_vec[i])
                f = float(fit_vec[i])
                abs_err = f - g
                rel_err = abs(abs_err) / abs(g) if abs(g) > 0.0 else float("inf")
                ok = abs(rel_err) <= args_cli.max_gt_rel_err if use_rel else abs(abs_err) <= args_cli.max_bias_abs_err
                gt_pass = gt_pass and ok
                print(
                    f"{name:10s} {label:10s} {g:10.4f} {f:10.4f} {abs_err:+10.4f} "
                    f"{rel_err * 100:9.1f}% {'OK' if ok else 'FAIL':>6s}"
                )
        delay_err = f_delay - SYNTHETIC_GT_DELAY
        delay_ok = abs(delay_err) <= args_cli.max_delay_err
        gt_pass = gt_pass and delay_ok
        print(
            f"{'delay':10s} {'global':10s} {SYNTHETIC_GT_DELAY:10d} {f_delay:10d} {delay_err:+10d} "
            f"{'':>10s} {'OK' if delay_ok else 'FAIL':>6s}"
        )
        print(
            f"[GATE]: GT 복원 = {'PASS' if gt_pass else 'FAIL'} "
            f"(rel≤{args_cli.max_gt_rel_err * 100:.0f}%, bias≤{args_cli.max_bias_abs_err} rad,"
            f" delay≤±{args_cli.max_delay_err} step)"
        )

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
            # explicit actuator는 PD를 파이썬에서 계산한다 → write_joint_stiffness_to_sim*은 no-op.
            MultiTrajectoryCMAES.apply_gains(articulation, joint_ids, kp, kd)

            articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=coulomb,
                joint_dynamic_friction_coeff=coulomb,
                joint_viscous_friction_coeff=viscous,
                joint_ids=joint_ids,
                env_ids=env_ids,
            )
            # ⚠ sim 뿐 아니라 data.default_* 캐시에도 반드시 써야 한다 (PACE update_simulator /
            # collect_chirp_sim_bipedleg.py와 동일). env의 raw 좌표 foot 마찰(cfg.foot_raw_friction)이
            # 그 캐시를 b_raw/c_raw의 저장소로 읽기 때문 — 캐시를 안 쓰면 foot 마찰이 통째로 0이 된 채
            # 검증이 돌아 적합 때와 다른 플랜트를 재는 결과가 된다.
            articulation.data.default_joint_armature.torch[:, joint_ids] = armature
            articulation.data.default_joint_friction_coeff.torch[:, joint_ids] = coulomb
            articulation.data.default_joint_viscous_friction_coeff.torch[:, joint_ids] = viscous
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
                # 게인은 관절별 벡터다. 표에는 데이터셋 구분이 되는 hip 게인만 싣는다.
                "kp": float(kp[0]),
                "kd": float(kd[0]),
                "rmse_fitted": float(rmse[FITTED].mean()),
                "rmse_nominal": float(rmse[NOMINAL].mean()),
                "max_fitted": float(err[:, FITTED].abs().max()),
                "per_joint": rmse[FITTED],
                "per_joint_nominal": rmse[NOMINAL],
            }
        )

    env.close()

    # ------------------------------------------------------------------
    # 보고
    # ------------------------------------------------------------------
    print("\n=== 궤적 재현 RMSE [rad] ===")
    print(
        f"{'시퀀스':22s} {'구분':9s} {'kp(hip)':>8s} {'kd(hip)':>8s} {'식별':>9s} {'nominal':>9s}"
        f" {'개선':>8s} {'식별 max':>9s}"
    )
    for r in results:
        ratio = r["rmse_nominal"] / r["rmse_fitted"] if r["rmse_fitted"] > 0 else float("inf")
        print(
            f"{r['name']:22s} {r['kind']:9s} {r['kp']:8.1f} {r['kd']:8.2f} "
            f"{r['rmse_fitted']:9.5f} {r['rmse_nominal']:9.5f} {ratio:7.1f}x {r['max_fitted']:9.5f}"
        )

    print("\n=== hold-out 관절별 RMSE [rad] ===")
    print(f"{'관절':10s} {'식별':>10s} {'nominal':>10s} {'개선':>8s}")
    for r in results:
        if r["kind"] != "HOLD-OUT":
            continue
        for i, label in enumerate(JOINT_LABELS):
            f_rmse = float(r["per_joint"][i])
            n_rmse = float(r["per_joint_nominal"][i])
            ratio = n_rmse / f_rmse if f_rmse > 0 else float("inf")
            print(f"  {label:10s} {f_rmse:10.5f} {n_rmse:10.5f} {ratio:7.1f}x")

    hold = [r for r in results if r["kind"] == "HOLD-OUT"]
    fitr = [r for r in results if r["kind"] == "fit"]
    holdout_pass = None
    if hold and fitr:
        mean_fit = sum(r["rmse_fitted"] for r in fitr) / len(fitr)
        mean_hold = sum(r["rmse_fitted"] for r in hold) / len(hold)
        mean_hold_nominal = sum(r["rmse_nominal"] for r in hold) / len(hold)
        improvement = mean_hold_nominal / mean_hold if mean_hold > 0 else float("inf")
        print(f"\n적합 시퀀스 평균 RMSE : {mean_fit:.5f} rad")
        print(f"hold-out 평균 RMSE    : {mean_hold:.5f} rad  (적합 대비 {mean_hold / mean_fit:.2f}배)")
        print(f"hold-out nominal 대비 : {improvement:.1f}배 개선 (nominal {mean_hold_nominal:.5f} rad)")
        print("hold-out이 적합 시퀀스와 비슷한 수준이면 과적합이 아니라 일관된 물리 모델이다(논문 판정 기준).")
        holdout_pass = improvement >= args_cli.min_improvement and mean_hold <= args_cli.max_holdout_rmse
        print(
            f"[GATE]: hold-out = {'PASS' if holdout_pass else 'FAIL'} "
            f"(개선≥{args_cli.min_improvement:.1f}x, RMSE≤{args_cli.max_holdout_rmse} rad)"
        )

    # ------------------------------------------------------------------
    # 최종 판정 (grep 대상: "VERDICT:")
    # ------------------------------------------------------------------
    checks = [c for c in (gt_pass, holdout_pass) if c is not None]
    if not checks:
        print("\nVERDICT: INCONCLUSIVE — hold-out도 합성 GT도 없어 판정할 수 없다.")
    elif all(checks):
        print("\nVERDICT: PASS — 합성 GT 복원과 hold-out 재현이 모두 기준을 만족한다.")
    else:
        failed = []
        if gt_pass is False:
            failed.append("GT 복원")
        if holdout_pass is False:
            failed.append("hold-out")
        print(f"\nVERDICT: FAIL — {', '.join(failed)} 기준 미달.")


if __name__ == "__main__":
    main()
    simulation_app.close()

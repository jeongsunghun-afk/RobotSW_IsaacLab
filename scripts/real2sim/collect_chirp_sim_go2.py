# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 합성 chirp 수집 — PACE 파이프라인 자기복원 테스트용 (Phase 1).

알려진 GT 물성(armature / viscous / Coulomb / encoder bias / delay)을 시뮬레이터에 주입하고
chirp을 재생해 ``data/<robot_name>/chirp_data.pt``를 만든다. 이후 ``scripts/pace/fit.py``가
그 GT를 되찾아오는지로 **식별 파이프라인 자체의 정확성**을 검증한다. 실로봇이 필요 없다.

실기 데이터를 쓸 때는 이 스크립트 대신 ``r2s_go2/chirp_collector.py``(실기 500 Hz 수집)가
같은 포맷의 ``chirp_data.pt``를 만든다. 여기신호는 양쪽 모두 ``r2s_go2/chirp.py``에서 나온다.

기록 규약 (PACE와 동일):
    dof_pos      = 엔코더가 읽는 값 = 실제 관절각 − encoder_bias
    des_dof_pos  = 시뮬레이터에 적용된 관절 목표각

실행::

    python scripts/real2sim/collect_chirp_sim_go2.py --headless
"""

"""Launch Omniverse Toolkit first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# chirp는 순수 stdlib이라 AppLauncher 기동 전에 임포트해도 안전하다 (r2s_udp와 동일 계약).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "r2s_go2"))
from chirp import DEFAULT_DURATION_S, DEFAULT_F0_HZ, DEFAULT_F1_HZ, build_chirp  # isort: skip

parser = argparse.ArgumentParser(description="R2S-GO2 synthetic chirp collection for PACE system identification.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-R2S-Go2-Sysid-v0", help="Name of the task.")
parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S, help="Chirp duration in seconds.")
parser.add_argument("--min_frequency", type=float, default=DEFAULT_F0_HZ, help="Chirp start frequency in Hz.")
parser.add_argument("--max_frequency", type=float, default=DEFAULT_F1_HZ, help="Chirp end frequency in Hz.")
parser.add_argument("--amplitude_scale", type=float, default=1.0, help="Scale applied to the default chirp amplitude.")
parser.add_argument("--kp", type=float, default=None, help="Position gain used for this capture. Defaults to SYSID_KP.")
parser.add_argument("--kd", type=float, default=None, help="Velocity gain used for this capture. Defaults to SYSID_KD.")
parser.add_argument("--out", type=str, default="chirp_data.pt", help="Output filename under data/<robot_name>/.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

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
    SYSID_KD,
    SYSID_KP,
)
from pace_sim2real.optim import MultiTrajectoryCMAES  # isort: skip
from pace_sim2real.utils import project_root, require_physx_backend  # isort: skip


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    require_physx_backend(env)

    device = env.unwrapped.device
    articulation = env.unwrapped.scene["robot"]
    joint_order = env_cfg.sim2real.joint_order

    # JOINT_ORDER(실기 인덱스) -> articulation 관절 인덱스. warp 커널이 int32를 요구한다.
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], dtype=torch.int32, device=device
    )
    env_ids = torch.arange(env.unwrapped.num_envs, dtype=torch.int32, device=device)

    # chirp 샘플률은 sim 제어율과 같아야 한다 (decimation=1이므로 물리 스텝 = 명령 스텝).
    sim_dt = env.unwrapped.sim.get_physics_dt()
    rate_hz = 1.0 / sim_dt
    if env_cfg.decimation != 1:
        raise RuntimeError(f"sysid 모드는 decimation=1이어야 한다 (현재 {env_cfg.decimation}).")

    from chirp import CHIRP_AMPLITUDE  # noqa: PLC0415

    amplitude = [a * args_cli.amplitude_scale for a in CHIRP_AMPLITUDE]
    time_list, target_list = build_chirp(
        duration=args_cli.duration,
        rate_hz=rate_hz,
        f0=args_cli.min_frequency,
        f1=args_cli.max_frequency,
        amplitude=amplitude,
    )

    # 이 캡처에 사용한 PD 게인. 데이터와 함께 저장해 재생 때 그대로 복원한다 —
    # PACE는 게인을 식별하지 않으므로(공통 스케일 축퇴), 다른 게인으로 재생하면 플랜트 파라미터가 편향된다.
    kp_value = SYSID_KP if args_cli.kp is None else args_cli.kp
    kd_value = SYSID_KD if args_cli.kd is None else args_cli.kd
    kp = torch.full((len(joint_order),), kp_value, dtype=torch.float32, device=device)
    kd = torch.full((len(joint_order),), kd_value, dtype=torch.float32, device=device)
    MultiTrajectoryCMAES.apply_gains(articulation, joint_ids, kp, kd)
    print(f"[INFO]: PD gains for this capture: kp={kp_value}, kd={kd_value}")
    num_steps = len(time_list)
    time_data = torch.tensor(time_list, dtype=torch.float32, device=device)
    # (T, 12) — JOINT_ORDER 순서
    targets = torch.tensor(target_list, dtype=torch.float32, device=device)

    print(f"[INFO]: Chirp {args_cli.min_frequency}->{args_cli.max_frequency} Hz, {args_cli.duration}s")
    print(f"[INFO]: {num_steps} steps @ {rate_hz:.0f} Hz (sim dt = {sim_dt * 1e3:.2f} ms)")

    # ------------------------------------------------------------------
    # GT 물성 주입 — CMA-ES가 되찾아와야 할 정답
    # ------------------------------------------------------------------
    def _gt(values: list[float]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.float32, device=device).unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    armature = _gt(SYNTHETIC_GT_ARMATURE)
    viscous = _gt(SYNTHETIC_GT_VISCOUS)
    coulomb = _gt(SYNTHETIC_GT_COULOMB)
    bias = _gt(SYNTHETIC_GT_BIAS)

    env.reset()

    articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=env_ids)
    articulation.data.default_joint_armature.torch[:, joint_ids] = armature
    # IsaacLab 6.0: static / dynamic / viscous를 한 번에 쓴다. dynamic == static이면 Coulomb 마찰.
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=coulomb,
        joint_dynamic_friction_coeff=coulomb,
        joint_viscous_friction_coeff=viscous,
        joint_ids=joint_ids,
        env_ids=env_ids,
    )
    articulation.data.default_joint_friction_coeff.torch[:, joint_ids] = coulomb
    articulation.data.default_joint_viscous_friction_coeff.torch[:, joint_ids] = viscous

    # 엔코더 바이어스 / 지연은 actuator 모델(PaceDCMotor) 안에 산다.
    # actuator의 joint 인덱스 순서는 articulation 순서이므로 JOINT_ORDER 값을 그 순서로 흩뿌린다.
    time_lag = torch.full((env.unwrapped.num_envs,), SYNTHETIC_GT_DELAY, dtype=torch.int, device=device)
    for drive_name, actuator in articulation.actuators.items():
        drive_ids = actuator.joint_indices
        if isinstance(drive_ids, slice):
            drive_ids = torch.arange(articulation.num_joints, device=device)[drive_ids]
        # drive_ids[i] = articulation 관절 인덱스 -> JOINT_ORDER에서의 위치
        order_pos = torch.tensor(
            [int((joint_ids == int(j)).nonzero()[0]) for j in drive_ids], dtype=torch.long, device=device
        )
        actuator.update_encoder_bias(bias[:, order_pos])
        actuator.update_time_lags(time_lag)
        actuator.reset(env_ids)
        print(f"[INFO]: actuator '{drive_name}': {len(drive_ids)} joints, bias/delay injected")

    # 초기 관절각을 chirp 시작점에 맞춘다. 엔코더가 chirp[0]을 읽어야 하므로 실제 관절각 = chirp[0] + bias.
    init_true = torch.zeros(env.unwrapped.num_envs, articulation.num_joints, device=device)
    init_true[:, joint_ids.long()] = targets[0].unsqueeze(0) + bias
    articulation.write_joint_position_to_sim_index(position=init_true)
    articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(init_true))

    # ------------------------------------------------------------------
    # chirp 재생
    # ------------------------------------------------------------------
    dof_pos_buffer = torch.zeros(num_steps, len(joint_order), device=device)
    dof_target_buffer = torch.zeros(num_steps, len(joint_order), device=device)

    actions = torch.zeros(env.unwrapped.num_envs, articulation.num_joints, device=device)
    for counter in range(num_steps):
        with torch.inference_mode():
            # 기록되는 값은 엔코더 읽음값 = 실제 관절각 − bias (PACE 규약).
            dof_pos_buffer[counter] = articulation.data.joint_pos[0, joint_ids.long()] - bias[0]
            # sysid 모드 action = articulation 관절 순서의 절대 목표각.
            actions[:, joint_ids.long()] = targets[counter].unsqueeze(0)
            env.step(actions)
            dof_target_buffer[counter] = articulation.data.joint_pos_target[0, joint_ids.long()]
            if (counter + 1) % int(rate_hz) == 0:
                print(f"[INFO]: {(counter + 1) / rate_hz:.0f} / {args_cli.duration:.0f} s")

    env.close()

    # ------------------------------------------------------------------
    # 저장
    # ------------------------------------------------------------------
    data_dir = project_root() / "data" / env_cfg.sim2real.robot_name
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / args_cli.out
    torch.save(
        {
            "time": time_data.cpu(),
            "dof_pos": dof_pos_buffer.cpu(),
            "des_dof_pos": dof_target_buffer.cpu(),
            # 재생 시 복원할 게인 (CONTRACT §12.4). 실기 collector도 같은 키로 저장한다.
            "kp": kp.cpu(),
            "kd": kd.cpu(),
            "joint_order": list(joint_order),
            "meta": {
                "source": "sim",
                "rate_hz": rate_hz,
                "f0_hz": args_cli.min_frequency,
                "f1_hz": args_cli.max_frequency,
                "duration_s": args_cli.duration,
                "amplitude_scale": args_cli.amplitude_scale,
            },
        },
        out_path,
    )
    tracking_err = (dof_pos_buffer - dof_target_buffer).abs()
    print(f"[INFO]: Saved {out_path}  ({num_steps} x {len(joint_order)})")
    print(f"[INFO]: |q - q_des| mean={tracking_err.mean():.4f} max={tracking_err.max():.4f} rad")
    print("[INFO]: 추종오차가 0에 가까우면 여기신호가 동역학을 자극하지 못한 것 — bounds/진폭을 재검토할 것.")
    print(
        "[INFO]: GT — armature="
        f"{SYNTHETIC_GT_ARMATURE[:3]}... viscous={SYNTHETIC_GT_VISCOUS[:3]}... "
        f"coulomb={SYNTHETIC_GT_COULOMB[:3]}... bias={SYNTHETIC_GT_BIAS[:3]}... delay={SYNTHETIC_GT_DELAY}"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

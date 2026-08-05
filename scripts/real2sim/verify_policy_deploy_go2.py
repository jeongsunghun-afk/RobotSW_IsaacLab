# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""r2s_go2 Policy 모드를 ROS/UDP 없이 r2s sim 안에서 그대로 돌려본다.

GUI publisher 가 하는 일을 그대로 재현한다 — `env.get_lowstate()`(DDS 순서 + wxyz quat)를 읽어
`policy_runtime` 으로 proprio 를 조립하고, deployable jit 로 추론하고, `art_to_dds` 한 목표를
`set_setpoint` 으로 넣는다. 즉 **관절 순서·quat 변환·obs 레이아웃·history 규약이 전부 이 경로에
포함된다** — GUI 를 띄우지 않고도 배포 입력단을 검증할 수 있다.

`--no_pace` 로 PACE 물성을 꺼서 A/B 를 만든다. 지표는 "허우적댐"을 수치화한 것:
관절 각속도 RMS 와 **부호 반전율**(진동과 단조 이동을 가른다).

실행:
    conda activate isaac-6.0
    CUDA_VISIBLE_DEVICES=1 env -u DISPLAY python scripts/real2sim/verify_policy_deploy_go2.py \
        --checkpoint logs/rsl_rl/go2_imitation_tracking/<RUN>/exported/deployable_policy.pt
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True, help="deployable_policy.pt 경로")
parser.add_argument("--x_vel", type=float, default=0.0, help="lin_vel_cmd x [m/s]")
parser.add_argument("--yaw_vel", type=float, default=0.0, help="yaw_vel_cmd [rad/s]")
parser.add_argument("--hold_s", type=float, default=8.0, help="engage 후 구동 시간 [s]")
parser.add_argument("--no_pace", action="store_true", help="PACE 물성을 끄고 nominal 로 (대조군)")
parser.add_argument(
    "--flip",
    action="store_true",
    help="`--recovery` 전용. 로봇을 belly-up(roll 180°)으로 뒤집어 안착시킨 뒤 engage 한다. "
    "PACE viscous 관절속도 천장이 실제로 물리는 구간을 보려면 이쪽이어야 한다.",
)
parser.add_argument(
    "--recovery",
    action="store_true",
    help="tracking(Policy) 대신 **Recovery** 정책을 검증한다. `--checkpoint` 는 recovery_policy.pt. "
    "GUI Recovery Start 와 같이 엎드린 spawn 자세에서 즉시 engage 하고, 기립 성공 여부를 본다. "
    "⚠ recovery 는 nominal 물성으로 학습됐다(`go2_recovery_env_cfg` armature 0.01 / 마찰 0)."
    " PACE 를 켠 r2s sim 은 recovery 에게는 반대 방향의 플랜트 격차다.",
)
parser.add_argument(
    "--init_stand",
    action="store_true",
    help="open-loop 기립 궤적을 건너뛰고 상태를 직접 기립 정지(z=0.34, 수평, default 관절)로 쓴다. "
    "정책 거동만 분리해 보기 위한 모드.",
)
parser.add_argument("--fold_s", type=float, default=1.5, help="기립 궤적: 현재 → STAND_FOLDED [s]")
parser.add_argument("--rise_s", type=float, default=2.0, help="기립 궤적: STAND_FOLDED → STAND_UP [s]")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app = AppLauncher(args_cli).app

import os  # noqa: E402
import sys  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "r2s_go2"))
import motions  # noqa: E402
import policy_runtime as pr  # noqa: E402
import recovery_runtime as rr  # noqa: E402

TASK = "Isaac-R2S-Go2-v0"
SETTLE_S = 1.0  # engage 직후 과도구간은 따로 집계


def main() -> None:
    cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=1)
    if args_cli.no_pace:
        cfg.use_pace_params = False
    env = gym.make(TASK, cfg=cfg).unwrapped
    ctrl_dt = cfg.sim.dt * cfg.decimation

    # ⚠ 이 스크립트는 GUI publisher 의 policy tick 을 그대로 재현하는 것이 목적이다. jit 을 직접
    #    로드하므로 `PolicyModel` 이 하는 스케일링 확정을 여기서 손으로 해줘야 한다 — 빠뜨리면
    #    `pr.action_to_target_art` 가 미확정 에러로 막는다(예전엔 배포와 다른 값으로 조용히
    #    검증됐다).
    pr.configure_from_policy(args_cli.checkpoint)
    model = torch.jit.load(args_cli.checkpoint, map_location=env.device).eval()
    hist = pr.ProprioHistory()
    zero = torch.zeros(env.num_envs, cfg.action_space, device=env.device)

    env.reset()
    arm = env.robot.data.joint_armature[0, 0].item()
    print(f"[plant] armature={arm:.4f}  use_pace_params={cfg.use_pace_params}")

    def play(frames):
        """프레임 시퀀스를 GUI publisher 와 같은 50 Hz 로 재생."""
        for q in frames:
            env.set_setpoint(q, [0.0] * 12, [motions.DEFAULT_KP] * 12, [motions.DEFAULT_KD] * 12, [0.0] * 12)
            env.step(zero)

    hz = 1.0 / ctrl_dt
    if args_cli.recovery:
        if args_cli.flip:
            # belly-up(roll 180°)으로 뒤집어 놓는다. **PACE viscous 천장이 물리는 유일한 구간이다** —
            # 엎드린 자세에서의 기립은 관절속도가 2 rad/s 대라 천장(23.5(1−q̇/30)=2.41q̇ → 7.36 rad/s)
            # 근처에도 못 가지만, flip 복구는 학습 기록상 8~22 rad/s 의 whip 을 쓴다.
            ids = torch.arange(env.num_envs, device=env.device)
            pose = env.robot.data.root_link_pose_w.clone()
            pose[:, 2] = 0.30  # 뒤집은 뒤 낙하시켜 자연스럽게 안착
            pose[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device)  # xyzw: roll 180°
            env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
            env.robot.write_root_com_velocity_to_sim_index(
                root_velocity=torch.zeros(env.num_envs, 6, device=env.device), env_ids=ids
            )
            # 안착까지 현재 자세를 유지 명령으로 잡아둔다(학습 env 의 settle_mode="passive" 와 같은 뜻)
            for _ in range(int(1.5 * hz)):
                q_now = env.robot.data.joint_pos[0, env._joint_ids].cpu().tolist()
                env.set_setpoint(q_now, [0.0] * 12, [rr.RECOVERY_KP] * 12, [rr.RECOVERY_KD] * 12, [0.0] * 12)
                env.step(zero)
            print(
                f"[flip] 안착 완료 — base_h={env.robot.data.root_pos_w[0, 2].item():.3f} "
                f"upright(-grav_z)={-env.robot.data.projected_gravity_b[0, 2].item():+.3f}"
            )

        # GUI Recovery Start 와 같다: 현재 자세에서 보간 없이 즉시 engage.
        prev = rr.PrevActionBuffer()
        kp, kd = rr.RECOVERY_KP, rr.RECOVERY_KD
        base_h0 = env.robot.data.root_pos_w[0, 2].item()
        up0 = -env.robot.data.projected_gravity_b[0, 2].item()
        print(f"[engage] recovery — base_h={base_h0:.3f} m, upright={up0:+.3f} 에서 즉시 시작, kd={kd}")
        jv_log, h_log = [], []
        for _ in range(int(args_cli.hold_s / ctrl_dt)):
            ls = env.get_lowstate()
            obs = rr.build_obs(
                tuple(float(v) for v in ls["imu"][4:7]),  # gyro (body frame 각속도)
                tuple(float(v) for v in ls["imu"][:4]),  # quat wxyz
                rr.dds_to_art([float(v) for v in ls["q"]]),
                rr.dds_to_art([float(v) for v in ls["dq"]]),
                prev.get(),
            )
            with torch.inference_mode():
                action = model(torch.tensor([obs], dtype=torch.float32, device=env.device))[0].tolist()
            clipped = rr.clip_action(action)
            prev.commit(clipped)
            env.set_setpoint(
                rr.art_to_dds(rr.action_to_target_art(clipped)), [0.0] * 12, [kp] * 12, [kd] * 12, [0.0] * 12
            )
            env.step(zero)
            jv_log.append(env.robot.data.joint_vel[0].cpu().numpy().copy())
            h_log.append(env.robot.data.root_pos_w[0, 2].item())
        jv, bh = np.array(jv_log), np.array(h_log)
        grav_z = env.robot.data.projected_gravity_b[0, 2].item()
        print(f"\n[결과] recovery / PACE {'OFF' if args_cli.no_pace else 'ON'}")
        print(f"  base_h  최대={bh.max():.3f}  최종={bh[-1]:.3f}  (기립 성공 기준 ≥ 0.28)")
        print(f"  upright(-grav_z) 최종={-grav_z:.3f}  (직립=+1, 뒤집힘=-1)")
        print(f"  jvel RMS={np.sqrt((jv**2).mean()):.3f}  p99={np.percentile(np.abs(jv), 99):.3f}")
        print(f"  판정: {'기립 성공' if bh[-1] >= 0.28 and -grav_z > 0.8 else '★ 기립 실패'}")
        env.close()
        app.close()
        return

    if args_cli.init_stand:
        # 기립 궤적을 건너뛰고 상태를 직접 기립 정지로 쓴다. 기립이 실패하는 조건에서도 정책 거동만
        # 떼어 보기 위한 모드다 — 기립에 실패한 arm 과 성공한 arm 의 수치를 나란히 놓으면 안 되므로.
        ids = torch.arange(env.num_envs, device=env.device)
        pose = env.robot.data.root_link_pose_w.clone()  # [N,7] pos(3)+quat(4, xyzw)
        pose[:, 2] = 0.34
        pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device)
        env.robot.write_root_link_pose_to_sim_index(root_pose=pose, env_ids=ids)
        env.robot.write_root_com_velocity_to_sim_index(
            root_velocity=torch.zeros(env.num_envs, 6, device=env.device), env_ids=ids
        )
        # ⚠ `default_joint_pos` 를 쓰면 안 된다 — 이 env 의 init_state 는 **prone**(STAND_FOLDED)이라
        # 기체만 z=0.34 로 올리고 다리는 접힌 자세가 된다. 기립 자세는 DEFAULT_POSE 다(DDS 순서라
        # articulation 순서로 remap 해서 쓴다).
        stand_art = torch.tensor([pr.dds_to_art(motions.DEFAULT_POSE)], dtype=torch.float32, device=env.device).repeat(
            env.num_envs, 1
        )
        env.robot.write_joint_state_to_sim_index(
            position=stand_art,
            velocity=torch.zeros_like(env.robot.data.joint_vel.torch),
            env_ids=ids,
        )
        env.set_setpoint(
            motions.DEFAULT_POSE, [0.0] * 12, [motions.DEFAULT_KP] * 12, [motions.DEFAULT_KD] * 12, [0.0] * 12
        )
        env.step(zero)
    else:
        # GUI 와 같은 순서: Stand Up(go2_stand_example 궤적) → Policy Start 의 안전 handover.
        cur = env._setpoint[0].cpu().tolist()
        play(motions.stand_up_sequence(cur, hz, fold_s=args_cli.fold_s, rise_s=args_cli.rise_s, hold_s=1.5))
        play(motions.interpolate_sequence(motions.STAND_UP, motions.DEFAULT_POSE, int(1.5 * hz)))
    base_h0 = env.robot.data.root_pos_w[0, 2].item()
    print(f"[engage] base_h={base_h0:.3f} m 에서 정책 시작")
    if base_h0 < 0.20:
        print("  ⚠ 기립 실패 — 이 상태의 측정치는 다른 arm 과 비교 불가다(정책이 바닥에서 시작).")

    jv_log, h_log = [], []
    for _ in range(int(args_cli.hold_s / ctrl_dt)):
        # ── GUI publisher 의 policy tick 을 그대로 ──
        ls = env.get_lowstate()
        quat_wxyz = tuple(float(v) for v in ls["imu"][:4])
        q_art = pr.dds_to_art([float(v) for v in ls["q"]])
        dq_art = pr.dds_to_art([float(v) for v in ls["dq"]])
        proprio = pr.build_proprio(quat_wxyz, (args_cli.x_vel, 0.0), args_cli.yaw_vel, q_art, dq_art)
        with torch.inference_mode():
            p = torch.tensor([proprio], dtype=torch.float32, device=env.device)
            h = torch.tensor([hist.push(proprio)], dtype=torch.float32, device=env.device)
            action = model(p, h)[0].tolist()
        pose_dds = pr.art_to_dds(pr.action_to_target_art(action))
        env.set_setpoint(pose_dds, [0.0] * 12, [motions.DEFAULT_KP] * 12, [motions.DEFAULT_KD] * 12, [0.0] * 12)
        env.step(zero)

        jv_log.append(env.robot.data.joint_vel[0].cpu().numpy().copy())
        h_log.append(env.robot.data.root_pos_w[0, 2].item())

    jv, bh = np.array(jv_log), np.array(h_log)
    n0 = int(SETTLE_S / ctrl_dt)
    print(f"\n[결과] PACE {'OFF' if args_cli.no_pace else 'ON'}  cmd=({args_cli.x_vel}, {args_cli.yaw_vel})")
    print(f"  {'구간':12s} {'jvel RMS':>10s} {'jvel p99':>10s} {'부호반전/step':>14s} {'base_h':>8s}")
    for label, sl in (("0~1s", slice(0, n0)), ("1s~", slice(n0, None))):
        v = jv[sl]
        flip = (np.sign(v[1:]) != np.sign(v[:-1])) & (np.abs(v[:-1]) > 0.5)
        print(
            f"  {label:12s} {np.sqrt((v**2).mean()):10.3f} {np.percentile(np.abs(v), 99):10.3f} "
            f"{flip.mean():14.4f} {bh[sl].mean():8.3f}"
        )

    env.close()
    app.close()


if __name__ == "__main__":
    main()

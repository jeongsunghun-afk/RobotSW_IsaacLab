# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
MPC(Model Predictive Control) 기반 4족 보행 테스트 스크립트

play.py 방식의 환경 로딩을 사용합니다:
  - isaaclab_tasks import → gym.register() 자동 등록
  - --robot 하나로 gymnasium task ID가 자동 결정됩니다

로봇 타입과 task ID 매핑:
  go2      → Go2
  go2_neck → Go2Neck
  skeleton → R_Skeleton-v1

사용법:
    # Go2 로봇 (GUI 시각화 ON)
    ./isaaclab.sh -p scripts/mpc/mpc_test.py --robot go2 --target_vx 0.5

    # 헤드리스 (시각화 OFF)
    ./isaaclab.sh -p scripts/mpc/mpc_test.py --robot go2 --headless

시각화 마커 색상:
    🟢 녹색 = stance 발의 현재 위치
    🔴 적색 = swing  발의 현재 위치
    🔵 청색 = 목표   발 착지 위치 (Bezier target)
"""

from __future__ import annotations

import argparse
import os
import sys

# ─── 1. Isaac Sim 부트스트랩 (반드시 가장 먼저) ──────────────────────────────
from isaaclab.app import AppLauncher

ROBOT_TO_TASK = {
    "go2": "Go2",
    "go2_neck": "Go2Neck",
    "skeleton": "R_Skeleton-v1",
}

parser = argparse.ArgumentParser(description="MPC 기반 4족 보행 테스트")
parser.add_argument(
    "--robot",
    type=str,
    default="go2",
    choices=list(ROBOT_TO_TASK.keys()),
    help=f"로봇 타입 (task ID 자동 결정): {ROBOT_TO_TASK}",
)
parser.add_argument("--target_vx", type=float, default=0.5, help="목표 전진 속도 (m/s)")
parser.add_argument("--target_vy", type=float, default=0.0, help="목표 측면 속도 (m/s)")
parser.add_argument("--target_wz", type=float, default=0.0, help="목표 요 각속도 (rad/s)")
parser.add_argument("--gait", type=str, default="trot", choices=["trot", "stand", "walk", "pace"], help="보행 패턴")
parser.add_argument("--num_envs", type=int, default=1, help="병렬 환경 수")
parser.add_argument("--num_steps", type=int, default=2000, help="총 실행 스텝 수")
parser.add_argument("--swing_height", type=float, default=0.08, help="발 스윙 최대 높이 (m)")
parser.add_argument("--gait_period", type=float, default=0.5, help="보행 주기 (초)")

# AppLauncher 표준 인자 (--headless, --device 등 포함)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# --robot으로 task ID 자동 결정
args_cli.task = ROBOT_TO_TASK[args_cli.robot]

# Isaac Sim 앱 시작
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ─── 2. Isaac Sim 초기화 이후 모듈 임포트 ────────────────────────────────────
import gymnasium as gym
import torch

sys.path.insert(0, os.path.dirname(__file__))
from mpc_locomotion import MPCLocomotionController

import isaaclab_tasks  # noqa: F401 — 환경 gym.register() 트리거
from isaaclab_tasks.utils import parse_env_cfg

# =============================================================================
# 환경 설정 조정 헬퍼
# =============================================================================


def patch_env_cfg(env_cfg, args_cli_ref):
    """play.py처럼 CLI 인자를 env_cfg에 적용합니다."""
    env_cfg.scene.num_envs = args_cli_ref.num_envs
    env_cfg.episode_length_s = 100.0  # MPC 테스트용 충분히 긴 에피소드
    if hasattr(env_cfg, "command_curriculum"):
        env_cfg.command_curriculum = False
    if hasattr(args_cli_ref, "device") and args_cli_ref.device:
        env_cfg.sim.device = args_cli_ref.device
    return env_cfg


# =============================================================================
# 시각화 마커 헬퍼
# =============================================================================


def make_foot_markers():
    """발 상태 시각화용 마커 생성.

    마커 색상:
        0 - 녹색 : stance 발 현재 위치
        1 - 적색 : swing  발 현재 위치
        2 - 청색 : 목표   발 위치 (Bezier landing)
    """
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

    cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/FootMarkers",
        markers={
            "stance": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 0.1),
                ),
            ),
            "swing": sim_utils.SphereCfg(
                radius=0.035,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.9, 0.15, 0.05),
                ),
            ),
            "target": sim_utils.SphereCfg(
                radius=0.025,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.05, 0.35, 0.95),
                ),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def update_foot_markers(markers, robot, controller):
    """발 현재 위치 & 목표 위치를 마커로 시각화합니다.

    마커 배치 (env 0 기준):
        인덱스 0~3 : 현재 발 위치  (stance=녹, swing=적)
        인덱스 4~7 : 목표 발 위치  (청색)
    """
    if controller._feet_body_ids is None:
        return

    # env 0만 시각화
    feet_pos = robot.data.body_pos_w[0, controller._feet_body_ids, :].detach()  # (4,3)
    target_pos = controller._target_pos_w[0].detach()  # (4,3)
    contact = controller._prev_contact[0].detach()  # (4,) bool

    # 현재 발 위치 마커 타입 (stance=0, swing=1)
    cur_idx = [0 if contact[i].item() else 1 for i in range(4)]
    tgt_idx = [2] * 4

    translations = torch.cat([feet_pos, target_pos], dim=0).cpu()  # (8,3)
    marker_indices = cur_idx + tgt_idx  # 길이 8

    markers.visualize(translations=translations, marker_indices=marker_indices)


# =============================================================================
# MPC 제어 루프
# =============================================================================


def run_mpc_loop(
    env,
    controller: MPCLocomotionController,
    num_steps: int,
    target_vx: float,
    target_vy: float,
    target_wz: float,
):
    """MPC 제어 루프.

    시각화 마커: 🟢=stance 현재 | 🔴=swing 현재 | 🔵=목표 착지점
    """
    obs, _ = env.reset()
    controller.initialize(env.unwrapped)
    controller.reset()

    robot = env.unwrapped._robot
    total_vel_err = 0.0
    fall_count = 0
    step_count = 0

    # 시각화 마커 생성 (GUI 모드에서만)
    foot_markers = None
    if not args_cli.headless:
        try:
            foot_markers = make_foot_markers()
            print("[MPC] 발 위치 시각화 마커 활성화")
            print("      🟢 녹색=stance 현재 위치 | 🔴 적색=swing 현재 위치 | 🔵 청색=목표 착지점")
        except Exception as e:
            print(f"[MPC] 마커 생성 실패 (무시): {e}")

    print(f"\n{'=' * 62}", flush=True)
    print("  🤖 MPC 보행 테스트 시작", flush=True)
    print(f"  task:  {args_cli.task}", flush=True)
    print(f"  robot: {args_cli.robot.upper():^12} | gait: {args_cli.gait.upper()}", flush=True)
    print(f"  목표:  vx={target_vx:+.2f} m/s | vy={target_vy:+.2f} m/s | wz={target_wz:+.2f} rad/s", flush=True)
    print(f"{'=' * 62}", flush=True)

    for step in range(num_steps):
        with torch.inference_mode():
            env.unwrapped._commands[:, 0] = target_vx
            env.unwrapped._commands[:, 1] = target_vy
            env.unwrapped._commands[:, 2] = target_wz

        # MPC 제어 수행 (observations 전달)
        # env.unwrapped를 전달하여 내부 속성에 접근 가능하게 함
        action = controller.compute_action(env.unwrapped, obs)

        # 환경 한 걸음 진행
        obs, rewards, terminated, truncated, info = env.step(action)

        # ── 시각화 업데이트 (10 스텝마다) ────────────────────────────────
        if foot_markers is not None and step % 10 == 0:
            try:
                update_foot_markers(foot_markers, robot, controller)
            except Exception:
                pass

        # ── 성능 측정 ────────────────────────────────────────────────────
        cur_vx = robot.data.root_lin_vel_b[:, 0].mean().item()
        cur_vy = robot.data.root_lin_vel_b[:, 1].mean().item()
        total_vel_err += abs(cur_vx - target_vx) + abs(cur_vy - target_vy)
        step_count += 1

        # ── 리셋 처리 ────────────────────────────────────────────────────
        done = terminated | truncated
        if done.any():
            reset_ids = done.nonzero(as_tuple=False).flatten()
            controller.reset(reset_ids)
            fall_count += int(terminated.sum().item())

        if step % 50 == 0:
            avg_err = total_vel_err / max(step_count, 1)
            base_height = robot.data.root_link_pos_w[:, 2].mean().item()
            print(
                f"  [Step {step:5d}] "
                f"vx={cur_vx:+.3f}/{target_vx:+.3f} | "
                f"vy={cur_vy:+.3f}/{target_vy:+.3f} | "
                f"높이={base_height:.3f}m | 오차={avg_err:.4f} | 낙하={fall_count}",
                flush=True,
            )

            # --- 자동 디버깅: 오차 과다 시 상세 정보 출력 ---
            if avg_err > 0.001 and step > 100:
                print(f"    [DEBUG] 속도 오차(= {avg_err:.4f}) 임계치 초과! 기구학 정렬 상태 점검 중...", flush=True)
                if hasattr(controller, "_ik_offsets"):
                    print(f"    [DEBUG] 현재 IK Offsets: \n{controller._ik_offsets}", flush=True)

            # ── 발 위치 디버그 출력 ────────────────────────────────────
            if controller._foot_indices is not None:
                p_foot_curr_w = env.unwrapped._robot.data.body_pos_w[0, controller._foot_indices, :]
                contact_mask = controller.gait._last_mask[0]  # env 0 기준
                labels = ["FL", "FR", "RL", "RR"]
                for fi in range(4):
                    st = "STC" if contact_mask[fi, 0].item() else "SWG"
                    cz = p_foot_curr_w[fi, 2].item()
                    tz = controller._target_pos_w[0, fi, 2].item()
                    cx = p_foot_curr_w[fi, 0].item()
                    cy = p_foot_curr_w[fi, 1].item()
                    print(f"    {labels[fi]}[{st}] cur=({cx:.2f},{cy:.2f},{cz:.3f}) tgt_z={tz:.3f}", flush=True)

    avg_vel_err = total_vel_err / max(step_count, 1)
    print(f"\n{'=' * 62}")
    print(f"  [결과]  총 스텝={step_count} | 평균오차={avg_vel_err:.4f} m/s | 낙하={fall_count}")
    print(f"{'=' * 62}\n")


# =============================================================================
# 메인
# =============================================================================


def main():
    device = args_cli.device if hasattr(args_cli, "device") and args_cli.device else "cuda:0"

    print(f"\n[MPC] task={args_cli.task} | robot={args_cli.robot} | envs={args_cli.num_envs}")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=device,
        num_envs=args_cli.num_envs,
    )
    env_cfg = patch_env_cfg(env_cfg, args_cli)
    env_cfg.events = None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)

    unwrapped = env.unwrapped
    print(f"[MPC] Joint names: {unwrapped._robot.joint_names}", flush=True)
    control_dt = unwrapped.step_dt
    action_scale = unwrapped.cfg.action_scale

    print(f"[MPC] 제어 dt={control_dt:.4f}s | action_scale={action_scale}")

    # MPC 제어기 초기화 (개선된 API 적용)
    controller = MPCLocomotionController(args_cli.robot, env.unwrapped.num_envs, device=env.unwrapped.device)
    controller.gait.set_gait(args_cli.gait)  # Gait 타입 명시적 설정

    # 제어 루프 실행
    try:
        run_mpc_loop(
            env,
            controller,
            target_vx=args_cli.target_vx,
            target_vy=args_cli.target_vy,
            target_wz=args_cli.target_wz,
            num_steps=args_cli.num_steps,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""new_dataset_smr TXT(JSON) 파일을 humanoid_amp MotionLoader 호환 NPZ 형식으로 변환.

입력 형식 (TXT/JSON, 61차원/프레임):
  [0:3]   root_pos          (world frame, xyz)
  [3:7]   root_quat         (xyzw 순서)
  [7:19]  joint_pos         (12개, FL→FR→RL→RR)
  [19:31] toes_local        (12개, 4발 × 3, FL→RL→FR→RR 순서, body-local frame)
  [31:34] lin_vel_local     (3개, body frame)
  [34:37] ang_vel_local     (3개, body frame)
  [37:49] joint_vel         (12개, FL→FR→RL→RR)
  [49:61] toe_vel_local     (12개, 4발 × 3, FL→RL→FR→RR 순서, body-local frame)

출력 형식 (NPZ):
  fps                       (int64, scalar)
  dof_names                 (unicode, [12])
  body_names                (unicode, [5]) — ["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]
  dof_positions             (float32, [N, 12])
  dof_velocities            (float32, [N, 12])
  body_positions            (float32, [N, 5, 3]) — toes: body-local, base: world
  body_rotations            (float32, [N, 5, 4]) — wxyz; toes=identity, base=root_quat
  body_linear_velocities    (float32, [N, 5, 3]) — body frame
  body_angular_velocities   (float32, [N, 5, 3]) — body frame; toes=zeros
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

# SMR 데이터셋 디렉토리 (스크립트 위치 기준)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(_THIS_DIR, "new_dataset_smr")
DEFAULT_OUTPUT_DIR = os.path.join(_THIS_DIR, "npz_dataset")

# Go2 관절 이름 (FL→FR→RL→RR, 3개/다리)
DOF_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]

# Body 이름: go2_motion_loader.BODY_NAMES 와 동일
BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]

# SMR toe 순서: [FL, RL, FR, RR] → 목표: [FL, FR, RL, RR]
_TOE_REORDER = [0, 2, 1, 3]


def convert_txt_to_npz(txt_path: str, output_path: str) -> bool:
    """단일 TXT 파일을 NPZ 파일로 변환.

    Args:
        txt_path: 입력 TXT(JSON) 파일 경로.
        output_path: 출력 NPZ 파일 경로 (확장자 없이 전달해도 자동 추가됨).

    Returns:
        변환 성공 여부.
    """
    try:
        with open(txt_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [ERROR] 파일 로드 실패 {txt_path}: {e}")
        return False

    frame_duration = float(data["FrameDuration"])
    fps = round(1.0 / frame_duration)
    frames = np.array(data["Frames"], dtype=np.float32)  # (N, 61)
    N = frames.shape[0]

    if frames.shape[1] != 61:
        print(f"  [ERROR] 예상치 못한 프레임 차원: {frames.shape[1]} (expected 61)")
        return False

    # ------------------------------------------------------------------ #
    # DOF (관절) 데이터 — 좌표 변환 불필요, 그대로 사용
    # ------------------------------------------------------------------ #
    dof_positions = frames[:, 7:19]  # (N, 12) FL→FR→RL→RR
    dof_velocities = frames[:, 37:49]  # (N, 12) FL→FR→RL→RR

    # ------------------------------------------------------------------ #
    # 쿼터니언 형식 변환: xyzw → wxyz
    # frames[:, 3:7] = (qx, qy, qz, qw)
    # ------------------------------------------------------------------ #
    quat_wxyz = np.concatenate([frames[:, 6:7], frames[:, 3:6]], axis=-1).astype(np.float32)  # (N, 4) (qw, qx, qy, qz)

    # ------------------------------------------------------------------ #
    # 발끝(toe) 위치 — body-local frame, 순서 재배열
    # SMR 순서: [FL, RL, FR, RR] → 목표: [FL, FR, RL, RR]
    # ------------------------------------------------------------------ #
    toes_raw = frames[:, 19:31].reshape(N, 4, 3)  # [FL, RL, FR, RR]
    toes_local = toes_raw[:, _TOE_REORDER, :]  # [FL, FR, RL, RR] (N, 4, 3)

    # ------------------------------------------------------------------ #
    # 발끝 속도 — body-local frame, 동일한 순서 재배열
    # ------------------------------------------------------------------ #
    toe_vel_raw = frames[:, 49:61].reshape(N, 4, 3)  # [FL, RL, FR, RR]
    toe_vel_local = toe_vel_raw[:, _TOE_REORDER, :]  # [FL, FR, RL, RR] (N, 4, 3)

    # ------------------------------------------------------------------ #
    # 베이스 데이터
    # ------------------------------------------------------------------ #
    root_pos = frames[:, 0:3]  # (N, 3) world frame
    lin_vel = frames[:, 31:34]  # (N, 3) body frame
    ang_vel = frames[:, 34:37]  # (N, 3) body frame

    # ------------------------------------------------------------------ #
    # body_positions: [FL, FR, RL, RR, base] (N, 5, 3)
    # toes: body-local frame (go2_amp 관례)
    # base: world frame
    # ------------------------------------------------------------------ #
    body_positions = np.concatenate([toes_local, root_pos[:, None, :]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # body_rotations: [FL, FR, RL, RR, base] (N, 5, 4) — wxyz
    # toes: 단위 쿼터니언 [1, 0, 0, 0]
    # base: root_quat_wxyz
    # ------------------------------------------------------------------ #
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 4, 1))
    body_rotations = np.concatenate([identity, quat_wxyz[:, None, :]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # body_linear_velocities: [FL, FR, RL, RR, base] (N, 5, 3)
    # toes: body-local frame
    # base: body frame
    # ------------------------------------------------------------------ #
    body_linear_velocities = np.concatenate([toe_vel_local, lin_vel[:, None, :]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # body_angular_velocities: [FL, FR, RL, RR, base] (N, 5, 3)
    # toes: 0 (발끝 각속도 데이터 없음)
    # base: body frame
    # ------------------------------------------------------------------ #
    toe_ang_vel = np.zeros((N, 4, 3), dtype=np.float32)
    body_angular_velocities = np.concatenate([toe_ang_vel, ang_vel[:, None, :]], axis=1).astype(np.float32)

    # ------------------------------------------------------------------ #
    # NPZ 저장
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    np.savez(
        output_path,
        fps=np.int64(fps),
        dof_names=np.array(DOF_NAMES),
        body_names=np.array(BODY_NAMES),
        dof_positions=dof_positions,
        dof_velocities=dof_velocities,
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )

    duration = frame_duration * (N - 1)
    print(
        f"  OK  {os.path.basename(txt_path)} → {os.path.basename(output_path)}.npz  (N={N}, fps={fps}, {duration:.2f}s)"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert new_dataset_smr TXT files to NPZ format")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=DEFAULT_INPUT_DIR,
        help=f"입력 TXT 파일 디렉토리 (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"출력 NPZ 파일 디렉토리 (default: {DEFAULT_OUTPUT_DIR})",
    )
    args, _ = parser.parse_known_args()

    txt_files = sorted(glob.glob(os.path.join(args.input_dir, "*.txt")))
    if not txt_files:
        print(f"[ERROR] TXT 파일 없음: {args.input_dir}")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"변환 시작: {len(txt_files)}개 파일  {args.input_dir} → {args.output_dir}")

    success = 0
    for txt_path in txt_files:
        stem = os.path.splitext(os.path.basename(txt_path))[0]
        output_path = os.path.join(args.output_dir, stem)
        if convert_txt_to_npz(txt_path, output_path):
            success += 1

    print(f"\n완료: {success}/{len(txt_files)}개 변환 성공")
    print(f"출력 디렉토리: {args.output_dir}")


if __name__ == "__main__":
    main()

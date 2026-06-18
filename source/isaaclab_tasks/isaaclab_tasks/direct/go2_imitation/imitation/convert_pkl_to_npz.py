#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 PKL 모션 데이터 → skrl AMP 호환 NPZ 변환 스크립트.

입력: go2_imitation/imitation/go2/*.pkl (18차원 frames)
  [0:3]  root_pos  (world frame, xyz)
  [3:6]  root_euler (roll, pitch, yaw) [rad]
  [6:18] dof_pos   (12관절: FL_hip/thigh/calf, FR_..., RL_..., RR_...)

출력: go2_skrl_amp/motions/npz_dataset/*.npz (humanoid_amp 호환 형식)
  fps             int64
  dof_names       (12,)  str
  body_names      (5,)   str  ["FL_foot","FR_foot","RL_foot","RR_foot","base"]
  dof_positions   (N,12) float32
  dof_velocities  (N,12) float32   ← finite difference
  body_positions  (N,5,3) float32  ← toe: body-local (FK), base: world
  body_rotations  (N,5,4) float32  ← toe: identity, base: wxyz
  body_linear_velocities  (N,5,3)  ← toe: 0, base: body frame
  body_angular_velocities (N,5,3)  ← toe: 0, base: body frame

사용법:
  python3 convert_pkl_to_npz.py [--pkl_dir ...] [--out_dir ...]
"""

from __future__ import annotations

import argparse
import os

# motion_lib.py의 Go2 FK / 변환 유틸리티 재사용
import pickle  # noqa: E402
import sys

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GO2_IMITATION_DIR = os.path.dirname(_SCRIPT_DIR)
_DIRECT_DIR = os.path.dirname(_GO2_IMITATION_DIR)
sys.path.insert(0, _GO2_IMITATION_DIR)
from motion_lib import (  # noqa: E402
    _euler_rates_to_body_angvel,
    _euler_to_quat_wxyz,
    _finite_diff,
    _go2_fk_foot_pos,
    _world_vel_to_body,
)

# ──────────────────────────────────────────────────────────────────────────
# 상수
# ──────────────────────────────────────────────────────────────────────────

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

BODY_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot", "base"]

_DEFAULT_PKL_DIR = os.path.join(_SCRIPT_DIR, "go2")
_DEFAULT_OUT_DIR = os.path.join(_DIRECT_DIR, "go2_skrl_amp", "motions", "npz_dataset")


# ──────────────────────────────────────────────────────────────────────────
# 변환 함수
# ──────────────────────────────────────────────────────────────────────────


def _load_pkl_data(pkl_path: str):
    """PKL 파일 로드 후 속도·FK 계산 (motion_lib._load_pkl 동등 구현)."""
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    fps = float(data["fps"])
    dt = 1.0 / fps
    frames = np.array(data["frames"], dtype=np.float32)  # (N, 18)
    assert frames.shape[1] == 18, f"프레임 크기 불일치: {frames.shape[1]} != 18 ({pkl_path})"

    root_pos = frames[:, 0:3]  # (N, 3)
    root_euler = frames[:, 3:6]  # (N, 3) roll/pitch/yaw
    dof_pos = frames[:, 6:18]  # (N, 12)

    root_quat = _euler_to_quat_wxyz(root_euler)
    lin_vel_world = _finite_diff(root_pos, dt)
    euler_rates = _finite_diff(root_euler, dt)
    dof_vel = _finite_diff(dof_pos, dt)

    lin_vel = _world_vel_to_body(lin_vel_world, root_quat)
    ang_vel = _euler_rates_to_body_angvel(root_euler, euler_rates)
    foot_pos = _go2_fk_foot_pos(dof_pos)

    return root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, fps


def convert_pkl_to_npz(pkl_path: str, out_dir: str) -> str:
    """PKL 파일 1개를 NPZ로 변환하여 out_dir에 저장합니다.

    Returns:
        저장된 NPZ 파일 경로
    """
    root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, fps = _load_pkl_data(pkl_path)
    N = root_pos.shape[0]

    # ── body_positions (N, 5, 3) ──────────────────────────────────────────
    # 인덱스 0~3: 발끝 위치 (body-local frame, FK 결과)
    # 인덱스 4  : 베이스 위치 (world frame)
    body_positions = np.concatenate(
        [foot_pos, root_pos[:, None, :]],  # (N,4,3) + (N,1,3)
        axis=1,
    )

    # ── body_rotations (N, 5, 4) wxyz ────────────────────────────────────
    # 발끝은 단위 쿼터니언(회전 없음), 베이스는 root_quat
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (N, 4, 1))
    body_rotations = np.concatenate(
        [identity, root_quat[:, None, :]],  # (N,4,4) + (N,1,4)
        axis=1,
    )

    # ── body_linear_velocities (N, 5, 3) — body frame ───────────────────
    # 발끝은 0, 베이스는 lin_vel (body frame)
    toe_lv = np.zeros((N, 4, 3), dtype=np.float32)
    body_linear_velocities = np.concatenate(
        [toe_lv, lin_vel[:, None, :]],  # (N,4,3) + (N,1,3)
        axis=1,
    )

    # ── body_angular_velocities (N, 5, 3) — body frame ──────────────────
    toe_av = np.zeros((N, 4, 3), dtype=np.float32)
    body_angular_velocities = np.concatenate(
        [toe_av, ang_vel[:, None, :]],  # (N,4,3) + (N,1,3)
        axis=1,
    )

    # ── 저장 ──────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pkl_path))[0]
    out_path = os.path.join(out_dir, f"{stem}.npz")
    np.savez(
        out_path,
        fps=np.int64(round(fps)),
        dof_names=np.array(DOF_NAMES),
        body_names=np.array(BODY_NAMES),
        dof_positions=dof_pos.astype(np.float32),
        dof_velocities=dof_vel.astype(np.float32),
        body_positions=body_positions,
        body_rotations=body_rotations,
        body_linear_velocities=body_linear_velocities,
        body_angular_velocities=body_angular_velocities,
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Go2 PKL → NPZ 변환")
    parser.add_argument(
        "--pkl_dir",
        default=_DEFAULT_PKL_DIR,
        help=f"PKL 파일 디렉토리 (기본값: {_DEFAULT_PKL_DIR})",
    )
    parser.add_argument(
        "--out_dir",
        default=_DEFAULT_OUT_DIR,
        help=f"출력 NPZ 디렉토리 (기본값: {_DEFAULT_OUT_DIR})",
    )
    args = parser.parse_args()

    pkl_files = sorted(f for f in os.listdir(args.pkl_dir) if f.endswith(".pkl"))
    if not pkl_files:
        print(f"[ERROR] PKL 파일 없음: {args.pkl_dir}")
        sys.exit(1)

    print(f"변환 대상: {len(pkl_files)}개 파일 ({args.pkl_dir})")
    print(f"출력 경로: {args.out_dir}")
    print()

    total_frames = 0
    for fname in pkl_files:
        pkl_path = os.path.join(args.pkl_dir, fname)
        out_path = convert_pkl_to_npz(pkl_path, args.out_dir)

        # 결과 검증 출력
        d = np.load(out_path)
        N = d["dof_positions"].shape[0]
        total_frames += N
        fps = int(d["fps"])
        dur = (N - 1) / fps
        print(f"  {fname} → {os.path.basename(out_path)}  frames={N}, fps={fps}, dur={dur:.2f}s")

    print(f"\n완료: 총 {total_frames}프레임 → {args.out_dir}")

    # NPZ 키 구조 검증 (첫 번째 파일)
    first_out = os.path.join(args.out_dir, os.path.splitext(pkl_files[0])[0] + ".npz")
    d = np.load(first_out)
    print("\n[구조 검증 — 첫 번째 NPZ]")
    for k, v in d.items():
        shape = getattr(v, "shape", "(scalar)")
        dtype = getattr(v, "dtype", "")
        print(f"  {k}: shape={shape}, dtype={dtype}")


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 모션 라이브러리 — MimicKit-style 인터페이스 + IsaacLab 전용 구현.

MimicKit의 MotionLib 인터페이스(sample_motions / sample_times / calc_motion_frame)를
외부 의존성 없이 순수 NumPy + PyTorch 로 재구현합니다.
go2_amp의 Go2MotionLoader에서 검증된 PKL 파싱·FK·속도 계산 코드를 재사용합니다.

PKL 프레임 레이아웃 (18개 값):
  [0:3]   root_pos  (x, y, z)
  [3:6]   root_euler  (roll, pitch, yaw) [rad]
  [6:18]  joint_pos (12 DOF — DOF_NAMES 순서)
  velocities / foot positions: finite difference 및 FK 자동 계산

calc_motion_frame 반환값:
  root_pos      [N, 3]   world frame 위치
  root_quat     [N, 4]   quaternion (w, x, y, z)
  root_lin_vel  [N, 3]   body frame 선속도
  root_ang_vel  [N, 3]   body frame 각속도
  dof_pos       [N, 12]  관절 각도
  dof_vel       [N, 12]  관절 속도
  foot_pos_local[N, 4, 3] 발 위치 (base-local frame, [FL, FR, RL, RR])
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import torch


# ──────────────────────────────────────────────────────────────
# Go2 FK / 변환 유틸리티 (Go2MotionLoader에서 검증된 코드)
# ──────────────────────────────────────────────────────────────

# 관절 이름 (Isaac Sim 순서 = MuJoCo go2.xml 순서)
DOF_NAMES = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
]

# Go2 운동학 파라미터 (표준 Unitree Go2 URDF 기준, 오차 < 0.5 mm)
_HIP_BASE = {
    "FL": np.array([+0.1934, +0.0465, 0.0], dtype=np.float32),
    "FR": np.array([+0.1934, -0.0465, 0.0], dtype=np.float32),
    "RL": np.array([-0.1934, +0.0465, 0.0], dtype=np.float32),
    "RR": np.array([-0.1934, -0.0465, 0.0], dtype=np.float32),
}
_THIGH_OFS_Y = {"FL": +0.0955, "FR": -0.0955, "RL": +0.0955, "RR": -0.0955}
_THIGH_LEN = 0.213
_CALF_LEN = 0.213


def _Rx(a: np.ndarray) -> np.ndarray:
    """Roll 회전 행렬 (N,) → (N, 3, 3)."""
    N = len(a)
    R = np.zeros((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = np.cos(a)
    R[:, 1, 2] = -np.sin(a)
    R[:, 2, 1] = np.sin(a)
    R[:, 2, 2] = np.cos(a)
    return R


def _Ry(a: np.ndarray) -> np.ndarray:
    """Pitch 회전 행렬 (N,) → (N, 3, 3)."""
    N = len(a)
    R = np.zeros((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = np.cos(a)
    R[:, 0, 2] = np.sin(a)
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -np.sin(a)
    R[:, 2, 2] = np.cos(a)
    return R


def _go2_fk_foot_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Go2 순운동학으로 발 위치(body-local frame)를 계산합니다.

    Args:
        joint_pos: (N, 12) — [FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...]

    Returns:
        foot_pos: (N, 4, 3) — [FL, FR, RL, RR] (body-local frame)
    """
    N = joint_pos.shape[0]
    leg_order = ["FL", "FR", "RL", "RR"]
    leg_start = [0, 3, 6, 9]
    foot_pos = np.zeros((N, 4, 3), dtype=np.float32)
    _down = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    for idx, (leg, start) in enumerate(zip(leg_order, leg_start)):
        hip_a = joint_pos[:, start]
        thigh_a = joint_pos[:, start + 1]
        calf_a = joint_pos[:, start + 2]

        hip_base = _HIP_BASE[leg]
        thigh_ofs = np.array([0.0, _THIGH_OFS_Y[leg], 0.0], dtype=np.float32)

        R_hip = _Rx(hip_a)    # (N, 3, 3)
        R_th = _Ry(thigh_a)   # (N, 3, 3)
        R_ca = _Ry(calf_a)    # (N, 3, 3)

        # thigh pivot = hip_base + R_hip @ thigh_ofs
        thigh_pivot = hip_base + np.einsum("nij,j->ni", R_hip, thigh_ofs)

        # calf pivot = thigh_pivot + R_hip @ R_th @ (0,0,-thigh_len)
        thigh_end_th = np.einsum("nij,j->ni", R_th, _down * _THIGH_LEN)
        calf_pivot = thigh_pivot + np.einsum("nij,nj->ni", R_hip, thigh_end_th)

        # foot = calf_pivot + R_hip @ R_th @ R_ca @ (0,0,-calf_len)
        calf_end_ca = np.einsum("nij,j->ni", R_ca, _down * _CALF_LEN)
        calf_end_th = np.einsum("nij,nj->ni", R_th, calf_end_ca)
        foot = calf_pivot + np.einsum("nij,nj->ni", R_hip, calf_end_th)

        foot_pos[:, idx, :] = foot

    return foot_pos


def _euler_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
    """Roll-Pitch-Yaw → quaternion (w, x, y, z) ZYX 내재적 회전."""
    r, p, y = rpy[:, 0] / 2, rpy[:, 1] / 2, rpy[:, 2] / 2
    cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
    sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    yq = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.stack([w, x, yq, z], axis=-1).astype(np.float32)


def _finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Forward finite difference (N, D) → (N, D), 마지막 프레임은 이전 복사."""
    vel = np.zeros_like(arr)
    vel[:-1] = (arr[1:] - arr[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def _world_vel_to_body(vel_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """World frame 속도 → body frame 속도 (quaternion inverse 적용)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    vx = (1 - 2*(y*y + z*z)) * vel_world[:, 0] + (2*(x*y + w*z)) * vel_world[:, 1] + (2*(x*z - w*y)) * vel_world[:, 2]
    vy = (2*(x*y - w*z))     * vel_world[:, 0] + (1 - 2*(x*x + z*z)) * vel_world[:, 1] + (2*(y*z + w*x)) * vel_world[:, 2]
    vz = (2*(x*z + w*y))     * vel_world[:, 0] + (2*(y*z - w*x)) * vel_world[:, 1] + (1 - 2*(x*x + y*y)) * vel_world[:, 2]
    return np.stack([vx, vy, vz], axis=-1).astype(np.float32)


def _euler_rates_to_body_angvel(euler: np.ndarray, euler_rates: np.ndarray) -> np.ndarray:
    """ZYX Euler rates → body frame 각속도."""
    roll, pitch = euler[:, 0], euler[:, 1]
    dr, dp, dy = euler_rates[:, 0], euler_rates[:, 1], euler_rates[:, 2]
    wx = dr - np.sin(pitch) * dy
    wy =  np.cos(roll) * dp + np.sin(roll) * np.cos(pitch) * dy
    wz = -np.sin(roll) * dp + np.cos(roll) * np.cos(pitch) * dy
    return np.stack([wx, wy, wz], axis=-1).astype(np.float32)


def _slerp_torch(q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation for quaternions. blend: (N,) or (N,1)."""
    if blend.dim() < q0.dim():
        blend = blend.unsqueeze(-1)
    cos_half = (q0 * q1).sum(dim=-1, keepdim=True)
    neg_mask = cos_half < 0
    q1 = q1.clone()
    q1[neg_mask.expand_as(q1)] = -q1[neg_mask.expand_as(q1)]
    cos_half = torch.abs(cos_half)
    half_theta = torch.acos(torch.clamp(cos_half, -1.0 + 1e-6, 1.0 - 1e-6))
    sin_half = torch.sqrt(torch.clamp(1.0 - cos_half * cos_half, min=1e-10))
    ra = torch.sin((1 - blend) * half_theta) / sin_half
    rb = torch.sin(blend * half_theta) / sin_half
    result = ra * q0 + rb * q1
    result = torch.where(torch.abs(sin_half) < 0.001, 0.5 * q0 + 0.5 * q1, result)
    result = torch.where(cos_half >= 1.0, q0, result)
    return result


# ──────────────────────────────────────────────────────────────
# Go2MotionLib 클래스
# ──────────────────────────────────────────────────────────────

class Go2MotionLib:
    """Go2 PKL 모션 라이브러리.

    MimicKit-style 인터페이스:
        sample_motions(n)                → motion_ids [n]
        sample_times(motion_ids, ...)    → times [n]
        calc_motion_frame(ids, times)    → 7-tuple (root_pos, root_quat, lin_vel, ang_vel,
                                                     dof_pos, dof_vel, foot_pos_local)

    Parameters
    ----------
    motion_files : str | list[str]
        PKL 파일 경로 또는 경로 목록. 디렉토리 지정 시 내부 *.pkl 자동 탐색.
    device : str
        PyTorch 디바이스 문자열 (예: "cuda:0", "cpu").
    weights : list[float] | None
        모션별 샘플링 가중치. None이면 프레임 수 비례 균등 가중치.
    """

    def __init__(
        self,
        motion_files: str | list[str],
        device: str,
        weights: list[float] | None = None,
    ) -> None:
        self._device = device

        # 파일 목록 결정
        if isinstance(motion_files, str):
            if os.path.isdir(motion_files):
                pkl_files = sorted(glob.glob(os.path.join(motion_files, "*.pkl")))
                assert pkl_files, f"디렉토리에 PKL 파일 없음: {motion_files}"
                motion_files = pkl_files
            else:
                motion_files = [motion_files]

        assert len(motion_files) > 0, "로드할 모션 파일이 없습니다."

        # ── PKL 파일 로드 ──────────────────────────────────────────
        all_root_pos: list[np.ndarray] = []
        all_root_quat: list[np.ndarray] = []
        all_lin_vel: list[np.ndarray] = []
        all_ang_vel: list[np.ndarray] = []
        all_dof_pos: list[np.ndarray] = []
        all_dof_vel: list[np.ndarray] = []
        all_foot_pos: list[np.ndarray] = []

        num_frames_list: list[int] = []
        fps_list: list[float | np.ndarray] = []

        for path in motion_files:
            assert os.path.isfile(path), f"파일이 존재하지 않습니다: {path}"
            rp, rq, lv, av, dp, dv, fp, fps = self._load_pkl(path)
            n = rp.shape[0]
            all_root_pos.append(rp)
            all_root_quat.append(rq)
            all_lin_vel.append(lv)
            all_ang_vel.append(av)
            all_dof_pos.append(dp)
            all_dof_vel.append(dv)
            all_foot_pos.append(fp)
            num_frames_list.append(n)
            fps_list.append(fps)
            print(f"[Go2MotionLib] 로드: {os.path.basename(path)} — {n} 프레임 ({(n-1)/fps:.2f}s @ {fps:.0f}fps)")

        # ── 모션별 메타데이터 ───────────────────────────────────────
        num_frames_arr = np.array(num_frames_list, dtype=np.int64)
        fps_arr = np.array(fps_list, dtype=np.float32)
        motion_lengths = (num_frames_arr - 1).astype(np.float32) / fps_arr  # 각 모션 길이 (초)

        # 가중치: None이면 모션 길이(총 시간) 비례
        if weights is None:
            w = motion_lengths.astype(np.float64)
        else:
            assert len(weights) == len(motion_files), "weights 길이가 파일 수와 불일치"
            w = np.array(weights, dtype=np.float64)
        w = w / w.sum()

        # ── 플랫 텐서 (모든 모션 프레임 연결) ──────────────────────
        self._frame_root_pos = torch.tensor(np.concatenate(all_root_pos), dtype=torch.float32, device=device)
        self._frame_root_quat = torch.tensor(np.concatenate(all_root_quat), dtype=torch.float32, device=device)
        self._frame_lin_vel = torch.tensor(np.concatenate(all_lin_vel), dtype=torch.float32, device=device)
        self._frame_ang_vel = torch.tensor(np.concatenate(all_ang_vel), dtype=torch.float32, device=device)
        self._frame_dof_pos = torch.tensor(np.concatenate(all_dof_pos), dtype=torch.float32, device=device)
        self._frame_dof_vel = torch.tensor(np.concatenate(all_dof_vel), dtype=torch.float32, device=device)
        self._frame_foot_pos = torch.tensor(np.concatenate(all_foot_pos), dtype=torch.float32, device=device)

        self._motion_num_frames = torch.tensor(num_frames_arr, dtype=torch.long, device=device)
        self._motion_lengths = torch.tensor(motion_lengths, dtype=torch.float32, device=device)
        self._motion_weights = torch.tensor(w, dtype=torch.float32, device=device)

        # 각 모션의 플랫 배열 내 시작 인덱스
        start_idx = np.zeros(len(motion_files), dtype=np.int64)
        start_idx[1:] = num_frames_arr[:-1].cumsum()
        self._motion_start_idx = torch.tensor(start_idx, dtype=torch.long, device=device)

        num_motions = len(motion_files)
        total_len = motion_lengths.sum()
        print(f"[Go2MotionLib] 총 {num_motions}개 모션, {total_len:.2f}s 로드 완료")

    # ── 공개 인터페이스 ────────────────────────────────────────────

    def sample_motions(self, n: int) -> torch.Tensor:
        """가중치 기반 모션 ID 샘플링.

        Returns:
            motion_ids: [n] int64 Tensor
        """
        return torch.multinomial(self._motion_weights, num_samples=n, replacement=True)

    def sample_times(self, motion_ids: torch.Tensor, truncate_time: float = 0.0) -> torch.Tensor:
        """모션 ID별 랜덤 타임스탬프 샘플링.

        Args:
            motion_ids: [N] int64 Tensor
            truncate_time: 모션 끝에서 이만큼 여유를 두고 샘플링

        Returns:
            times: [N] float32 Tensor (초)
        """
        motion_len = self._motion_lengths[motion_ids]
        if truncate_time > 0.0:
            motion_len = torch.clamp(motion_len - truncate_time, min=0.0)
        phase = torch.rand(motion_ids.shape, device=self._device)
        return phase * motion_len

    def calc_motion_frame(
        self, motion_ids: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        """지정 시간의 모션 프레임을 보간하여 반환합니다.

        Args:
            motion_ids: [N] int64
            times:      [N] float32 (초)

        Returns:
            root_pos      [N, 3]    world frame 위치
            root_quat     [N, 4]    quaternion (w, x, y, z)
            root_lin_vel  [N, 3]    body frame 선속도
            root_ang_vel  [N, 3]    body frame 각속도
            dof_pos       [N, 12]   관절 각도
            dof_vel       [N, 12]   관절 속도
            foot_pos_local[N, 4, 3] 발 위치 (base-local frame, [FL, FR, RL, RR])
        """
        idx0, idx1, blend = self._calc_frame_blend(motion_ids, times)  # blend: [N]
        b = blend.unsqueeze(-1)                 # [N, 1] for broadcasting

        root_pos = (1 - b) * self._frame_root_pos[idx0] + b * self._frame_root_pos[idx1]
        root_quat = _slerp_torch(self._frame_root_quat[idx0], self._frame_root_quat[idx1], blend)
        lin_vel = (1 - b) * self._frame_lin_vel[idx0] + b * self._frame_lin_vel[idx1]
        ang_vel = (1 - b) * self._frame_ang_vel[idx0] + b * self._frame_ang_vel[idx1]
        dof_pos = (1 - b) * self._frame_dof_pos[idx0] + b * self._frame_dof_pos[idx1]
        dof_vel = (1 - b) * self._frame_dof_vel[idx0] + b * self._frame_dof_vel[idx1]

        b3 = blend.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1] for foot_pos [N, 4, 3]
        foot_pos = (1 - b3) * self._frame_foot_pos[idx0] + b3 * self._frame_foot_pos[idx1]

        return root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos

    @property
    def num_motions(self) -> int:
        return int(self._motion_num_frames.shape[0])

    @property
    def total_length(self) -> float:
        return float(self._motion_lengths.sum().item())

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """IsaacLab joint 이름 목록(IsaacLab 순서)을 받아 motion data DOF 인덱스를 반환.

        go2_amp_env.py의 Go2MotionLoader.get_dof_index()와 동일한 패턴.

        Usage:
            motion_dof_indices = motion_lib.get_dof_index(list(robot.data.joint_names))
            # motion_data[motion_dof_indices] → IsaacLab 순서로 재배열된 joint 데이터
        """
        indexes = []
        for name in dof_names:
            assert name in DOF_NAMES, (
                f"DOF 이름 '{name}'이 motion data에 없습니다. "
                f"사용 가능: {DOF_NAMES}"
            )
            indexes.append(DOF_NAMES.index(name))
        return indexes

    # ── 내부 헬퍼 ──────────────────────────────────────────────────

    def _calc_frame_blend(
        self, motion_ids: torch.Tensor, times: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """모션 ID + 시간 → (idx0, idx1, blend)."""
        num_frames = self._motion_num_frames[motion_ids]
        start_idx = self._motion_start_idx[motion_ids]
        motion_len = self._motion_lengths[motion_ids]

        phase = torch.clamp(times / motion_len, 0.0, 1.0)
        frame_float = phase * (num_frames - 1).float()
        idx0 = torch.minimum(frame_float.long(), num_frames - 2)
        idx1 = idx0 + 1
        blend = frame_float - idx0.float()

        return idx0 + start_idx, idx1 + start_idx, blend

    @staticmethod
    def _load_pkl(path: str) -> tuple:
        """PKL 파일 로드 + 속도/FK 계산.

        Returns:
            root_pos   (N, 3)    world frame
            root_quat  (N, 4)    (w, x, y, z)
            lin_vel    (N, 3)    body frame
            ang_vel    (N, 3)    body frame
            dof_pos    (N, 12)
            dof_vel    (N, 12)
            foot_pos   (N, 4, 3) body-local frame
            fps        float
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        fps = float(data["fps"])
        dt = 1.0 / fps
        frames = np.array(data["frames"], dtype=np.float32)  # (N, 18)
        assert frames.shape[1] == 18, f"프레임 크기 불일치: {frames.shape[1]} != 18 ({path})"

        root_pos = frames[:, 0:3]    # (N, 3)
        root_euler = frames[:, 3:6]  # (N, 3) roll/pitch/yaw
        dof_pos = frames[:, 6:18]    # (N, 12)

        root_quat = _euler_to_quat_wxyz(root_euler)   # (N, 4) wxyz
        lin_vel_world = _finite_diff(root_pos, dt)    # (N, 3) world frame
        euler_rates = _finite_diff(root_euler, dt)    # (N, 3)
        dof_vel = _finite_diff(dof_pos, dt)           # (N, 12)

        lin_vel = _world_vel_to_body(lin_vel_world, root_quat)         # body frame
        ang_vel = _euler_rates_to_body_angvel(root_euler, euler_rates) # body frame

        foot_pos = _go2_fk_foot_pos(dof_pos)  # (N, 4, 3) body-local

        return root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, float(fps)

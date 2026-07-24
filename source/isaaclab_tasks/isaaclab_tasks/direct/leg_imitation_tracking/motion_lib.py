# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Leg(17-DOF 4족 + 허리) 모션 라이브러리 — MimicKit-style 인터페이스 + IsaacLab 전용 구현.

`go2_imitation_tracking/motion_lib.py` 의 Leg_URDF2 버전. MimicKit의 MotionLib 인터페이스
(sample_motions / sample_times / calc_motion_frame)를 외부 의존성 없이
순수 NumPy + PyTorch 로 재구현합니다.

PKL 프레임 레이아웃 (23개 값 — `mujoco_retarget_leg/convert_smr_to_pkl.py` 출력):
  [0:3]   root_pos  (x, y, z) [m]
  [3:6]   root_rot  (exponential map = axis * angle, 3D) [rad]
  [6:23]  joint_pos (17 DOF — DOF_NAMES 순서) [rad]
  velocities / foot positions: finite difference 및 FK 자동 계산

.. note::
    go2 버전은 [3:6] 을 ZYX Euler 로 해석했지만, 변환 스크립트는 exponential map 을
    기록한다. 이 파일은 변환 스크립트와 일치하도록 exponential map 으로 해석하며,
    각속도도 Euler rate 근사가 아닌 quaternion finite difference 로 계산한다.

calc_motion_frame 반환값:
  root_pos      [N, 3]   world frame 위치 [m]
  root_quat     [N, 4]   quaternion (w, x, y, z)
  root_lin_vel  [N, 3]   body frame 선속도 [m/s]
  root_ang_vel  [N, 3]   body frame 각속도 [rad/s]
  dof_pos       [N, 17]  관절 각도 [rad]
  dof_vel       [N, 17]  관절 속도 [rad/s]
  foot_pos_local[N, 4, 3] 발(foot_link 원점) 위치 [m] (base-local frame, [FL, FR, HL, HR])
"""

from __future__ import annotations

import glob
import os
import pickle

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────
# Leg FK / 변환 유틸리티
# ──────────────────────────────────────────────────────────────

# PKL 프레임의 관절 순서 (= MuJoCo Leg.xml qpos 순서 = 변환 스크립트 DOF_NAMES).
# IsaacLab articulation 의 관절 순서와 다를 수 있으므로 get_dof_index() 로 런타임 재매핑한다.
DOF_NAMES = [
    "HL_hip_joint",
    "HL_thigh_joint",
    "HL_calf_joint",
    "HL_foot_joint",
    "HR_hip_joint",
    "HR_thigh_joint",
    "HR_calf_joint",
    "HR_foot_joint",
    "FB_waist_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FL_foot_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FR_foot_joint",
]

# AMP foot 피처의 다리 순서 (env 의 KEY_BODY_NAMES 와 반드시 동일해야 함).
FOOT_ORDER = ["FL", "FR", "HL", "HR"]

# Leg_URDF2 운동학 체인 (robots/Leg_URDF2/urdf/Leg.xml 에서 그대로 발췌).
# 링크 오프셋에 회전 성분이 없어(quat 미지정) 순수 병진 + 관절 회전으로 정확히 재현된다.
#   offsets: (hip_link, thigh_link, calf_link, foot_link) 부모 기준 위치 [m]
#   signs:   (hip, thigh, calf) 관절 회전 부호. hip 은 항상 x축, thigh/calf 는 y축.
#   waist:   앞다리는 FB_waist_joint(z축) 아래에 매달려 있음
_LEG_CHAIN: dict[str, dict] = {
    "FL": {
        "offsets": [(0.299, 0.0225, 0.0), (0.0, 0.115, 0.0), (-0.13192, -0.275, -0.1884), (0.10354, 0.275, -0.22205)],
        "signs": (1.0, 1.0, 1.0),
        "waist": True,
    },
    "FR": {
        "offsets": [(0.299, -0.0225, 0.0), (0.0, -0.115, 0.0), (-0.13192, 0.0, -0.1884), (0.10354, 0.0, -0.22205)],
        "signs": (1.0, 1.0, 1.0),
        "waist": True,
    },
    "HL": {
        "offsets": [(-0.301, 0.0225, 0.0), (0.0, 0.115, 0.0), (0.13192, 0.0, -0.1884), (-0.18768, 0.0, -0.15748)],
        "signs": (1.0, -1.0, -1.0),
        "waist": False,
    },
    "HR": {
        "offsets": [
            (-0.301, -0.0225, 0.0),
            (0.0, -0.115, 0.0),
            (0.13192, 0.0, -0.1884),
            (-0.187680888564152, 0.0, -0.157482964373201),
        ],
        "signs": (1.0, -1.0, -1.0),
        "waist": False,
    },
}

# DOF_NAMES 기준, 다리별 (hip, thigh, calf) 인덱스
_LEG_DOF_IDX: dict[str, tuple[int, int, int]] = {
    "HL": (0, 1, 2),
    "HR": (4, 5, 6),
    "FL": (9, 10, 11),
    "FR": (13, 14, 15),
}
_WAIST_DOF_IDX = 8

NUM_DOF = len(DOF_NAMES)  # 17
FRAME_DIM = 6 + NUM_DOF  # 23 (root_pos 3 + exp_map 3 + dof 17)


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


def _Rz(a: np.ndarray) -> np.ndarray:
    """Yaw 회전 행렬 (N,) → (N, 3, 3)."""
    N = len(a)
    R = np.zeros((N, 3, 3), dtype=np.float32)
    R[:, 0, 0] = np.cos(a)
    R[:, 0, 1] = -np.sin(a)
    R[:, 1, 0] = np.sin(a)
    R[:, 1, 1] = np.cos(a)
    R[:, 2, 2] = 1.0
    return R


def _leg_fk_foot_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Leg_URDF2 순운동학으로 발(foot_link 원점) 위치를 base-local frame 에서 계산합니다.

    ``foot_link`` 원점(= 발목 관절 위치)을 사용한다. env 의 sim 측 피처가
    ``body_pos_w[FL_foot_link]`` 를 base frame 으로 옮긴 값이므로 정확히 대응된다.
    발목(foot) 관절 각도는 이 원점에 영향을 주지 않지만 ``dof_pos`` 에 이미 포함되어 있다.

    Args:
        joint_pos: (N, 17) — DOF_NAMES 순서의 관절 각도 [rad]

    Returns:
        foot_pos: (N, 4, 3) — FOOT_ORDER([FL, FR, HL, HR]) 순서, base-local frame [m]
    """
    N = joint_pos.shape[0]
    foot_pos = np.zeros((N, 4, 3), dtype=np.float32)

    R_waist = _Rz(joint_pos[:, _WAIST_DOF_IDX])  # (N, 3, 3)

    for idx, leg in enumerate(FOOT_ORDER):
        chain = _LEG_CHAIN[leg]
        i_hip, i_thigh, i_calf = _LEG_DOF_IDX[leg]
        s_hip, s_thigh, s_calf = chain["signs"]

        p_hip, p_thigh, p_calf, p_foot = (np.array(o, dtype=np.float32) for o in chain["offsets"])

        R_hip = _Rx(s_hip * joint_pos[:, i_hip])  # (N, 3, 3)
        R_th = _Ry(s_thigh * joint_pos[:, i_thigh])
        R_ca = _Ry(s_calf * joint_pos[:, i_calf])

        # 안쪽에서 바깥으로: foot = p_hip + R_hip @ (p_thigh + R_th @ (p_calf + R_ca @ p_foot))
        acc = p_calf + np.einsum("nij,j->ni", R_ca, p_foot)  # (N, 3)
        acc = p_thigh + np.einsum("nij,nj->ni", R_th, acc)
        acc = p_hip + np.einsum("nij,nj->ni", R_hip, acc)

        # 앞다리는 허리 관절 아래에 매달려 있다.
        if chain["waist"]:
            acc = np.einsum("nij,nj->ni", R_waist, acc)

        foot_pos[:, idx, :] = acc

    return foot_pos


def _exp_map_to_quat_wxyz(exp_map: np.ndarray) -> np.ndarray:
    """Exponential map (axis * angle) → quaternion (w, x, y, z).

    변환 스크립트 ``convert_smr_to_pkl.quat_to_exp_map`` 의 역변환.
    """
    angle = np.linalg.norm(exp_map, axis=-1)  # (N,)
    small = angle < 1e-8
    safe_angle = np.where(small, 1.0, angle)
    axis = exp_map / safe_angle[:, None]

    half = 0.5 * angle
    w = np.cos(half)
    xyz = axis * np.sin(half)[:, None]
    xyz = np.where(small[:, None], 0.0, xyz)
    return np.concatenate([w[:, None], xyz], axis=-1).astype(np.float32)


def _quat_body_ang_vel(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    """Quaternion 시계열 → body frame 각속도 [rad/s].

    w_body = 2 * vec(q_t^{-1} ⊗ q_{t+1}) / dt (finite difference).
    """
    q0 = quat_wxyz[:-1]
    q1 = quat_wxyz[1:].copy()

    # 이웃 프레임 간 부호 뒤집힘(double cover) 제거
    flip = np.sum(q0 * q1, axis=-1) < 0.0
    q1[flip] = -q1[flip]

    # q_rel = conj(q0) ⊗ q1
    w0, x0, y0, z0 = q0[:, 0], -q0[:, 1], -q0[:, 2], -q0[:, 3]
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    rx = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    ry = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    rz = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1

    ang_vel = np.zeros_like(quat_wxyz[:, :3])
    ang_vel[:-1] = 2.0 * np.stack([rx, ry, rz], axis=-1) / dt
    ang_vel[-1] = ang_vel[-2] if len(ang_vel) > 1 else 0.0
    return ang_vel.astype(np.float32)


def _finite_diff(arr: np.ndarray, dt: float) -> np.ndarray:
    """Forward finite difference (N, D) → (N, D), 마지막 프레임은 이전 복사."""
    vel = np.zeros_like(arr)
    vel[:-1] = (arr[1:] - arr[:-1]) / dt
    vel[-1] = vel[-2]
    return vel


def _world_vel_to_body(vel_world: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """World frame 속도 → body frame 속도 (quaternion inverse 적용)."""
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    vx = (
        (1 - 2 * (y * y + z * z)) * vel_world[:, 0]
        + (2 * (x * y + w * z)) * vel_world[:, 1]
        + (2 * (x * z - w * y)) * vel_world[:, 2]
    )
    vy = (
        (2 * (x * y - w * z)) * vel_world[:, 0]
        + (1 - 2 * (x * x + z * z)) * vel_world[:, 1]
        + (2 * (y * z + w * x)) * vel_world[:, 2]
    )
    vz = (
        (2 * (x * z + w * y)) * vel_world[:, 0]
        + (2 * (y * z - w * x)) * vel_world[:, 1]
        + (1 - 2 * (x * x + y * y)) * vel_world[:, 2]
    )
    return np.stack([vx, vy, vz], axis=-1).astype(np.float32)


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
# LegMotionLib 클래스
# ──────────────────────────────────────────────────────────────


class LegMotionLib:
    """Leg(17-DOF) PKL 모션 라이브러리.

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
            print(f"[LegMotionLib] 로드: {os.path.basename(path)} — {n} 프레임 ({(n - 1) / fps:.2f}s @ {fps:.0f}fps)")

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
        print(f"[LegMotionLib] 총 {num_motions}개 모션, {total_len:.2f}s 로드 완료")

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

    def calc_motion_frame(self, motion_ids: torch.Tensor, times: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """지정 시간의 모션 프레임을 보간하여 반환합니다.

        Args:
            motion_ids: [N] int64
            times:      [N] float32 (초)

        Returns:
            root_pos      [N, 3]    world frame 위치
            root_quat     [N, 4]    quaternion (w, x, y, z)
            root_lin_vel  [N, 3]    body frame 선속도
            root_ang_vel  [N, 3]    body frame 각속도
            dof_pos       [N, 17]   관절 각도
            dof_vel       [N, 17]   관절 속도
            foot_pos_local[N, 4, 3] 발 위치 (base-local frame, [FL, FR, HL, HR])
        """
        idx0, idx1, blend = self._calc_frame_blend(motion_ids, times)  # blend: [N]
        b = blend.unsqueeze(-1)  # [N, 1] for broadcasting

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

        IsaacLab articulation 순서 → PKL DOF 순서 인덱스.

        Usage:
            motion_dof_indices = motion_lib.get_dof_index(list(robot.data.joint_names))
            # motion_data[motion_dof_indices] → IsaacLab 순서로 재배열된 joint 데이터
        """
        indexes = []
        for name in dof_names:
            assert name in DOF_NAMES, f"DOF 이름 '{name}'이 motion data에 없습니다. 사용 가능: {DOF_NAMES}"
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
            dof_pos    (N, 17)
            dof_vel    (N, 17)
            foot_pos   (N, 4, 3) body-local frame
            fps        float
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        fps = float(data["fps"])
        dt = 1.0 / fps
        frames = np.array(data["frames"], dtype=np.float32)  # (N, 23)
        assert frames.shape[1] == FRAME_DIM, f"프레임 크기 불일치: {frames.shape[1]} != {FRAME_DIM} ({path})"

        # PKL 이 dof_names 를 담고 있으면 순서 일치를 강제한다 (silent 재배열 버그 방지).
        pkl_dof_names = data.get("dof_names")
        assert pkl_dof_names is None or list(pkl_dof_names) == DOF_NAMES, (
            f"PKL 관절 순서가 DOF_NAMES 와 다릅니다 ({path}): {pkl_dof_names}"
        )

        root_pos = frames[:, 0:3]  # (N, 3)
        root_exp_map = frames[:, 3:6]  # (N, 3) exponential map
        dof_pos = frames[:, 6:FRAME_DIM]  # (N, 17)

        root_quat = _exp_map_to_quat_wxyz(root_exp_map)  # (N, 4) wxyz
        lin_vel_world = _finite_diff(root_pos, dt)  # (N, 3) world frame
        dof_vel = _finite_diff(dof_pos, dt)  # (N, 17)

        lin_vel = _world_vel_to_body(lin_vel_world, root_quat)  # body frame
        ang_vel = _quat_body_ang_vel(root_quat, dt)  # body frame

        foot_pos = _leg_fk_foot_pos(dof_pos)  # (N, 4, 3) body-local

        return root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, float(fps)

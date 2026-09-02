# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 모션 라이브러리 — MimicKit-style 인터페이스 + IsaacLab 전용 구현.

MimicKit의 MotionLib 인터페이스(sample_motions / sample_times / calc_motion_frame)를
외부 의존성 없이 순수 NumPy + PyTorch 로 재구현합니다.
go2_amp의 Go2MotionLoader에서 검증된 PKL 파싱·FK·속도 계산 코드를 재사용합니다.

PKL 프레임 레이아웃 (18개 값):
  [0:3]   root_pos  (x, y, z)
  [3:6]   root_rot  (exponential map = axis * angle) [rad]
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

        R_hip = _Rx(hip_a)  # (N, 3, 3)
        R_th = _Ry(thigh_a)  # (N, 3, 3)
        R_ca = _Ry(calf_a)  # (N, 3, 3)

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

        self._motion_names = [os.path.splitext(os.path.basename(path))[0] for path in motion_files]
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
            print(f"[Go2MotionLib] 로드: {os.path.basename(path)} — {n} 프레임 ({(n - 1) / fps:.2f}s @ {fps:.0f}fps)")

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
        # `motion_mean_speeds` 는 모션마다 파이썬 루프를 도므로 리셋마다 부르면 비싸다. 첫 호출에 캐시.
        self._cached_mean_speeds: torch.Tensor | None = None

        # 각 모션의 플랫 배열 내 시작 인덱스
        start_idx = np.zeros(len(motion_files), dtype=np.int64)
        start_idx[1:] = num_frames_arr[:-1].cumsum()
        self._motion_start_idx = torch.tensor(start_idx, dtype=torch.long, device=device)

        num_motions = len(motion_files)
        total_len = motion_lengths.sum()
        print(f"[Go2MotionLib] 총 {num_motions}개 모션, {total_len:.2f}s 로드 완료")

    # ── 공개 인터페이스 ────────────────────────────────────────────

    @property
    def motion_names(self) -> list[str]:
        """모션 클립 이름 (확장자 제외). 인덱스는 가중치/속도 배열과 같은 순서."""
        return list(self._motion_names)

    @property
    def motion_mean_speeds(self) -> torch.Tensor:
        """모션별 평균 수평 속도 [m/s], shape [num_motions], float.

        root 위치의 프레임 간 차분으로 계산한다(world frame xy 평면). leg motion_lib 와 같은 정의라
        두 task 의 속도 라벨을 같은 자로 잰다.
        """
        speeds = []
        for i in range(len(self._motion_lengths)):
            s = int(self._motion_start_idx[i])
            n = int(self._motion_num_frames[i])
            pos_xy = self._frame_root_pos[s : s + n, :2]
            fps = (n - 1) / float(self._motion_lengths[i])
            step = torch.linalg.norm(pos_xy[1:] - pos_xy[:-1], dim=-1) * fps
            speeds.append(step.mean())
        return torch.stack(speeds)

    @property
    def motion_mean_yaw_rates(self) -> torch.Tensor:
        """모션별 평균 yaw 각속도 [rad/s] (body frame z, 부호 유지), shape [num_motions], float.

        미러 클립은 부호가 뒤집히므로 좌우 짝은 크기가 같고 부호가 반대다.
        """
        rates = []
        for i in range(len(self._motion_lengths)):
            s = int(self._motion_start_idx[i])
            n = int(self._motion_num_frames[i])
            rates.append(self._frame_ang_vel[s : s + n, 2].mean())
        return torch.stack(rates)

    def sample_motions_near_speed(self, target_speeds: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
        """명령 속도에 가까운 속도의 클립을 확률적으로 고른다 (leg motion_lib 와 동일한 정의).

        softmax(-|speed - target| / temperature) 에 기존 `_motion_weights` 를 곱해 뽑는다.

        Args:
            target_speeds: [N] 각 env 의 명령 속도 [m/s]
            temperature: 매칭 무름 [m/s]. 작을수록 최근접에 가깝다.

        Returns:
            motion_ids: [N] int64
        """
        if self._cached_mean_speeds is None:
            self._cached_mean_speeds = self.motion_mean_speeds.to(self._device)
        speeds: torch.Tensor = self._cached_mean_speeds  # [M]
        tau = max(float(temperature), 1e-6)
        logits = -(speeds[None, :] - target_speeds[:, None].to(speeds.device)).abs() / tau
        w = self._motion_weights[None, :] * torch.softmax(logits, dim=-1)
        return torch.multinomial(w, num_samples=1).squeeze(-1)

    def set_motion_weights_command_uniform(self, vel_max: float) -> torch.Tensor:
        """샘플링 가중치를 재설정해 expert 속도 분포가 U[0, vel_max] 에 근사하도록 만든다.

        각 모션에 속도축 상의 최근접 셀 폭을 가중치로 준다(leg motion_lib 와 동일). 같은 속도의
        모션(mirror 쌍)은 셀을 균등 분할한다.

        Args:
            vel_max: 명령 선속도 상한 [m/s]. 보통 env cfg 의 `lin_vel_x_max`.

        Returns:
            재설정된 정규화 가중치, shape [num_motions], float.
        """
        speeds = self.motion_mean_speeds
        order = torch.argsort(speeds)
        sorted_speeds = speeds[order].clamp(0.0, vel_max)

        mids = 0.5 * (sorted_speeds[1:] + sorted_speeds[:-1])
        lo = torch.cat([sorted_speeds.new_zeros(1), mids])
        hi = torch.cat([mids, sorted_speeds.new_full((1,), vel_max)])
        cell = (hi - lo).clamp(min=0.0)

        w_sorted = cell.clone()
        i = 0
        while i < len(sorted_speeds):
            j = i
            while j + 1 < len(sorted_speeds) and torch.isclose(sorted_speeds[j + 1], sorted_speeds[i], atol=1e-4):
                j += 1
            if j > i:
                w_sorted[i : j + 1] = w_sorted[i : j + 1].sum() / (j - i + 1)
            i = j + 1

        w = torch.empty_like(w_sorted)
        w[order] = w_sorted
        w = w.clamp(min=1e-6)
        self._motion_weights = (w / w.sum()).to(self._device)
        return self._motion_weights

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
            dof_pos       [N, 12]   관절 각도
            dof_vel       [N, 12]   관절 속도
            foot_pos_local[N, 4, 3] 발 위치 (base-local frame, [FL, FR, RL, RR])
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

        go2_amp_env.py의 Go2MotionLoader.get_dof_index()와 동일한 패턴.

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

        root_pos = frames[:, 0:3]  # (N, 3)
        root_exp_map = frames[:, 3:6]  # (N, 3) exponential map (axis * angle)
        dof_pos = frames[:, 6:18]  # (N, 12)

        root_quat = _exp_map_to_quat_wxyz(root_exp_map)  # (N, 4) wxyz
        lin_vel_world = _finite_diff(root_pos, dt)  # (N, 3) world frame
        dof_vel = _finite_diff(dof_pos, dt)  # (N, 12)

        lin_vel = _world_vel_to_body(lin_vel_world, root_quat)  # body frame
        ang_vel = _quat_body_ang_vel(root_quat, dt)  # body frame (쿼터니언 차분)

        foot_pos = _go2_fk_foot_pos(dof_pos)  # (N, 4, 3) body-local

        return root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, float(fps)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import glob
import os

import numpy as np
import torch


class MotionLoader:
    """Helper class to load and sample motion data from NumPy-file format.

    단일 NPZ 파일 또는 NPZ 파일들이 있는 디렉토리를 받아 모든 프레임을 이어붙여 로드한다.
    여러 파일의 경우 첫 번째 파일의 dof_names / body_names 를 기준으로 하며,
    모든 파일은 동일한 fps, dof 수, body 수를 가져야 한다.

    Args:
        motion_file: NPZ 파일 경로 또는 NPZ 파일들이 있는 디렉토리 경로.
        device: 텐서를 올릴 디바이스.
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        self.device = device

        # 파일 목록 결정
        if os.path.isdir(motion_file):
            npz_files = sorted(glob.glob(os.path.join(motion_file, "*.npz")))
            if not npz_files:
                raise FileNotFoundError(f"NPZ 파일 없음: {motion_file}")
        elif os.path.isfile(motion_file):
            npz_files = [motion_file]
        else:
            raise FileNotFoundError(f"경로가 존재하지 않음: {motion_file}")

        # 첫 번째 파일에서 메타데이터 읽기
        first = np.load(npz_files[0])
        self._dof_names: list[str] = first["dof_names"].tolist()
        self._body_names: list[str] = first["body_names"].tolist()
        self.dt: float = float(1.0 / first["fps"])

        # 모든 파일의 프레임 데이터를 리스트로 수집
        dof_pos_list, dof_vel_list = [], []
        body_pos_list, body_rot_list = [], []
        body_lv_list, body_av_list = [], []

        # per-clip 경계 인덱스 (uniform time sampling에 사용)
        self._clip_end_frames: list[int] = []  # 각 clip의 마지막 프레임 인덱스 (exclusive)
        total = 0

        for path in npz_files:
            d = np.load(path)
            n = d["dof_positions"].shape[0]
            dof_pos_list.append(d["dof_positions"])
            dof_vel_list.append(d["dof_velocities"])
            body_pos_list.append(d["body_positions"])
            body_rot_list.append(d["body_rotations"])
            body_lv_list.append(d["body_linear_velocities"])
            body_av_list.append(d["body_angular_velocities"])
            total += n
            self._clip_end_frames.append(total)
            print(f"  Motion loaded ({os.path.basename(path)}): frames={n}, dt={self.dt:.5f}s")

        # 전체 프레임 concat
        def _to_tensor(arrays: list) -> torch.Tensor:
            return torch.tensor(np.concatenate(arrays, axis=0), dtype=torch.float32, device=self.device)

        self.dof_positions = _to_tensor(dof_pos_list)
        self.dof_velocities = _to_tensor(dof_vel_list)
        self.body_positions = _to_tensor(body_pos_list)
        self.body_rotations = _to_tensor(body_rot_list)
        self.body_linear_velocities = _to_tensor(body_lv_list)
        self.body_angular_velocities = _to_tensor(body_av_list)

        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"MotionLoader: {len(npz_files)}개 파일 로드 완료 — 총 {self.num_frames}프레임, {self.duration:.2f}s")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        # floor 기반으로 frame index를 계산해야 blend가 항상 [0, 1) 범위에 들어온다.
        # round를 쓰면 index_0가 실제 시간보다 앞 프레임으로 올림될 때 blend가 음수(외삽)가 된다.
        frame_float = np.clip(times / self.dt, 0.0, self.num_frames - 1)
        index_0 = frame_float.astype(int)  # floor
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = frame_float - index_0  # 항상 [0, 1)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert duration <= self.duration, (
            f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        )
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            A tuple containing sampled motion data:
                - DOF positions (with shape (N, num_dofs))
                - DOF velocities (with shape (N, num_dofs))
                - Body positions (with shape (N, num_bodies, 3))
                - Body rotations (with shape (N, num_bodies, 4), as wxyz quaternion)
                - Body linear velocities (with shape (N, num_bodies, 3))
                - Body angular velocities (with shape (N, num_bodies, 3))
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)

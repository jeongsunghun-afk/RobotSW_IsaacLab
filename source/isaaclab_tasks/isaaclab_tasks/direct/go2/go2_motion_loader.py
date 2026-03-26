"""Go2 모션 파일 로더.

stmr_go2.py (compute_dataset_features)가 생성하는 DeepMimic JSON txt 파일을 파싱하여
Isaac Lab AMP 학습에 사용할 수 있는 인터페이스를 제공합니다.

각 프레임 레이아웃 (61개 값):
  [0:3]    root_pos  (x, y, z)
  [3:7]    root_rot  (quat xyzw)
  [7:19]   joint_pos (12개, MuJoCo go2.xml 순서: FL/FR/RL/RR × hip/thigh/calf)
  [19:31]  toe_pos_local (4개 × 3, 저장 순서: FL, RL, FR, RR)
  [31:34]  lin_vel   (base frame)
  [34:37]  ang_vel   (base frame)
  [37:49]  joint_vel (12개)
  [49:61]  toe_vel_local (4개 × 3, 저장 순서: FL, RL, FR, RR)
"""

import glob
import json
import os
from typing import Optional

import numpy as np
import torch


class Go2MotionLoader:
    """Go2 DeepMimic JSON txt 모션 파일 로더.

    SkeletonMotionLoader와 동일한 API를 제공합니다.
    """

    # txt 프레임 인덱스 상수
    ROOT_POS_START = 0
    ROOT_POS_END = 3
    ROOT_ROT_START = 3
    ROOT_ROT_END = 7
    JOINT_POS_START = 7
    JOINT_POS_END = 19   # 12개
    TOE_POS_START = 19
    TOE_POS_END = 31     # 4 × 3
    LIN_VEL_START = 31
    LIN_VEL_END = 34
    ANG_VEL_START = 34
    ANG_VEL_END = 37
    JOINT_VEL_START = 37
    JOINT_VEL_END = 49   # 12개
    TOE_VEL_START = 49
    TOE_VEL_END = 61     # 4 × 3

    NUM_JOINTS = 12
    NUM_TOES = 4

    # Go2 관절 이름 (MuJoCo go2.xml 순서와 동일)
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

    # body 이름 (발끝 4개 [FL,FR,RL,RR] + base)
    BODY_NAMES = [
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
        "base",
    ]

    def __init__(
        self,
        motion_files: str | list[str],
        device: torch.device | str,
    ) -> None:
        self.device = device

        if isinstance(motion_files, str):
            if os.path.isdir(motion_files):
                motion_files = sorted(glob.glob(os.path.join(motion_files, "*.txt")))
            else:
                motion_files = [motion_files]

        assert len(motion_files) > 0, "로드할 모션 파일이 없습니다."

        all_joint_pos: list[np.ndarray] = []
        all_joint_vel: list[np.ndarray] = []
        all_root_pos: list[np.ndarray] = []
        all_root_rot_wxyz: list[np.ndarray] = []
        all_lin_vel: list[np.ndarray] = []
        all_ang_vel: list[np.ndarray] = []
        all_toe_pos: list[np.ndarray] = []
        all_toe_vel: list[np.ndarray] = []

        total_dt = None

        for path in motion_files:
            assert os.path.isfile(path), f"파일이 존재하지 않습니다: {path}"
            with open(path) as f:
                data = json.load(f)

            frames = np.array(data["Frames"], dtype=np.float32)  # (N, 61)
            frame_duration = float(data["FrameDuration"])

            if total_dt is None:
                total_dt = frame_duration
            else:
                assert abs(total_dt - frame_duration) < 1e-6, (
                    f"모션 파일 간 FrameDuration 불일치: {total_dt} vs {frame_duration} ({path})"
                )

            root_pos = frames[:, self.ROOT_POS_START : self.ROOT_POS_END]    # (N,3)
            root_rot_xyzw = frames[:, self.ROOT_ROT_START : self.ROOT_ROT_END]  # (N,4) xyzw
            joint_pos = frames[:, self.JOINT_POS_START : self.JOINT_POS_END]  # (N,12)
            toe_pos_flat = frames[:, self.TOE_POS_START : self.TOE_POS_END]   # (N,12)
            lin_vel = frames[:, self.LIN_VEL_START : self.LIN_VEL_END]        # (N,3)
            ang_vel = frames[:, self.ANG_VEL_START : self.ANG_VEL_END]        # (N,3)
            joint_vel = frames[:, self.JOINT_VEL_START : self.JOINT_VEL_END]  # (N,12)
            toe_vel_flat = frames[:, self.TOE_VEL_START : self.TOE_VEL_END]   # (N,12)

            # xyzw → wxyz (Isaac Lab 쿼터니언 순서)
            root_rot_wxyz = np.concatenate(
                [root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1
            )  # (N,4)

            # 발끝 (N,12) → (N,4,3), 순서 재정렬
            # stmr_go2.py 저장 순서: [0=FL, 1=RL, 2=FR, 3=RR]
            # BODY_NAMES 기대 순서:  [0=FL, 1=FR, 2=RL, 3=RR]
            # 재정렬: [0, 2, 1, 3]
            toe_pos = toe_pos_flat.reshape(-1, self.NUM_TOES, 3)[:, [0, 2, 1, 3], :]
            toe_vel = toe_vel_flat.reshape(-1, self.NUM_TOES, 3)[:, [0, 2, 1, 3], :]

            all_root_pos.append(root_pos)
            all_root_rot_wxyz.append(root_rot_wxyz)
            all_joint_pos.append(joint_pos)
            all_joint_vel.append(joint_vel)
            all_lin_vel.append(lin_vel)
            all_ang_vel.append(ang_vel)
            all_toe_pos.append(toe_pos)
            all_toe_vel.append(toe_vel)

            print(
                f"모션 로드 ({os.path.basename(path)}): "
                f"{frames.shape[0]} 프레임 ({frames.shape[0] * frame_duration:.2f}s)"
            )

        joint_pos_all = np.concatenate(all_joint_pos, axis=0)
        joint_vel_all = np.concatenate(all_joint_vel, axis=0)
        root_pos_all = np.concatenate(all_root_pos, axis=0)
        root_rot_all = np.concatenate(all_root_rot_wxyz, axis=0)
        lin_vel_all = np.concatenate(all_lin_vel, axis=0)
        ang_vel_all = np.concatenate(all_ang_vel, axis=0)
        toe_pos_all = np.concatenate(all_toe_pos, axis=0)   # (N,4,3)
        toe_vel_all = np.concatenate(all_toe_vel, axis=0)   # (N,4,3)

        self.dof_positions = torch.tensor(joint_pos_all, dtype=torch.float32, device=device)
        self.dof_velocities = torch.tensor(joint_vel_all, dtype=torch.float32, device=device)

        # body_positions: (N, 5, 3) = toe 4개(local frame) + base(world frame)
        root_pos_t = torch.tensor(root_pos_all, dtype=torch.float32, device=device).unsqueeze(1)
        toe_pos_t = torch.tensor(toe_pos_all, dtype=torch.float32, device=device)
        self.body_positions = torch.cat([toe_pos_t, root_pos_t], dim=1)  # (N,5,3)

        # body_rotations: (N, 5, 4) — 발끝은 root rot 복사
        root_rot_t = torch.tensor(root_rot_all, dtype=torch.float32, device=device)
        toe_rot_t = root_rot_t.unsqueeze(1).expand(-1, self.NUM_TOES, -1)
        root_rot_body = root_rot_t.unsqueeze(1)
        self.body_rotations = torch.cat([toe_rot_t, root_rot_body], dim=1)  # (N,5,4)

        lin_vel_t = torch.tensor(lin_vel_all, dtype=torch.float32, device=device)
        ang_vel_t = torch.tensor(ang_vel_all, dtype=torch.float32, device=device)
        toe_vel_t = torch.tensor(toe_vel_all, dtype=torch.float32, device=device)

        root_lin = lin_vel_t.unsqueeze(1)
        root_ang = ang_vel_t.unsqueeze(1)
        self.body_linear_velocities = torch.cat([toe_vel_t, root_lin], dim=1)  # (N,5,3)
        self.body_angular_velocities = torch.cat(
            [torch.zeros_like(toe_vel_t), root_ang], dim=1
        )  # (N,5,3)

        self._dof_names = self.DOF_NAMES
        self._body_names = self.BODY_NAMES

        self.dt = total_dt
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)

        print(
            f"Go2MotionLoader: 총 {self.num_frames} 프레임 "
            f"({self.duration:.2f}s, dt={self.dt:.4f}s)"
        )

    @property
    def dof_names(self) -> list[str]:
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        return self._body_names

    @property
    def num_dofs(self) -> int:
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
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
        q1: Optional[torch.Tensor] = None,
        blend: Optional[torch.Tensor] = None,
        start: Optional[np.ndarray] = None,
        end: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        cos_half_theta = (q0 * q1).sum(dim=-1, keepdim=True)
        neg_mask = cos_half_theta[..., 0] < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)

        half_theta = torch.acos(torch.clamp(cos_half_theta, -1.0 + 1e-6, 1.0 - 1e-6))
        sin_half_theta = torch.sqrt(torch.clamp(1.0 - cos_half_theta * cos_half_theta, min=1e-10))

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q = ratio_a * q0 + ratio_b * q1
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(cos_half_theta >= 1.0, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray):
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: Optional[float] = None) -> np.ndarray:
        duration = self.duration if duration is None else duration
        assert duration <= self.duration
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self,
        num_samples: int,
        times: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        """Isaac Sim joint 이름으로 모션 로더 DOF 인덱스를 반환합니다.

        Go2는 MuJoCo와 Isaac Sim의 joint 이름이 동일하므로 직접 매핑합니다.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, (
                f"Joint 이름 '{name}'이 모션 로더에 없습니다: {self._dof_names}"
            )
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        indexes = []
        for name in body_names:
            assert name in self._body_names, (
                f"Body 이름 '{name}'이 존재하지 않습니다: {self._body_names}"
            )
            indexes.append(self._body_names.index(name))
        return indexes

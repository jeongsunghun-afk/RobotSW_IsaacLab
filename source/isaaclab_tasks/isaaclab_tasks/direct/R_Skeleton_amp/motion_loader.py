"""R_Skeleton 모션 파일 로더.

DeepMimic JSON 형식의 txt 파일을 파싱하여 Isaac Lab skrl AMP 학습에 사용할 수 있는
인터페이스를 제공합니다. humanoid_amp/motions/motion_loader.py 와 동일한 API를 제공합니다.

각 프레임 레이아웃 (113개 값):
  [0:3]    root_pos  (x, y, z)
  [3:7]    root_rot  (quat xyzw, PyBullet 순서)
  [7:45]   joint_pos (38개 관절)
  [45:57]  toe_pos_local (발끝 4개 × 3)
  [57:60]  lin_vel
  [60:63]  ang_vel
  [63:101] joint_vel (38개)
  [101:113] toe_vel_local (발끝 4개 × 3)
"""

import glob
import json
import os
from typing import Optional

import numpy as np
import torch


class SkeletonMotionLoader:
    """R_Skeleton DeepMimic JSON txt 모션 파일 로더.

    humanoid_amp의 MotionLoader와 동일한 API를 제공합니다.
    멀티 파일을 연결하여 하나의 긴 모션으로 처리합니다.
    """

    # txt 프레임 인덱스 상수
    # 실제 파일 포맷 (113개 값):
    #   root_pos(3) + root_rot(4) + joint_pos(38) + toe_pos(12) +
    #   lin_vel(3) + ang_vel(3) + joint_vel(38) + toe_vel(12)
    # rskeleton_retarget_motion.py가 curr_pose_full로 저장하는 순서
    ROOT_POS_START = 0
    ROOT_POS_END = 3
    ROOT_ROT_START = 3
    ROOT_ROT_END = 7
    JOINT_POS_START = 7
    JOINT_POS_END = 45   # 38개
    TOE_POS_START = 45
    TOE_POS_END = 57     # 4 × 3
    LIN_VEL_START = 57
    LIN_VEL_END = 60
    ANG_VEL_START = 60
    ANG_VEL_END = 63
    JOINT_VEL_START = 63
    JOINT_VEL_END = 101  # 38개
    TOE_VEL_START = 101
    TOE_VEL_END = 113    # 4 × 3

    NUM_JOINTS = 38
    NUM_TOES = 4

    # R_Skeleton 관절 이름 (AMPLoader 사용 순서와 동일)
    DOF_NAMES = [
        "FL_joint1_hip_yaw",
        "FL_joint2_hip_roll",
        "FL_joint3_hip_pitch",
        "FL_joint4_knee",
        "FL_joint5_ankle",
        "FL_joint6_wrist_p",
        "FR_joint1_hip_yaw",
        "FR_joint2_hip_roll",
        "FR_joint3_hip_pitch",
        "FR_joint4_knee",
        "FR_joint5_ankle",
        "FR_joint6_wrist_p",
        "RL_joint1_hip_yaw",
        "RL_joint2_hip_roll",
        "RL_joint3_hip_pitch",
        "RL_joint4_knee",
        "RL_joint5_ankle",
        "RL_joint6_wrist_p",
        "RR_joint1_hip_yaw",
        "RR_joint2_hip_roll",
        "RR_joint3_hip_pitch",
        "RR_joint4_knee",
        "RR_joint5_ankle",
        "RR_joint6_wrist_p",
        "waist_joint1_yaw",
        "waist_joint2_pitch",
        "arm_joint1_yaw",
        "arm_joint2_pitch",
        "arm_joint3_yaw",
        "arm_joint4_elbow",
        "arm_joint5_wrist_yaw",
        "arm_joint6_wrist_pitch",
        "arm_joint7_wrist_roll",
        "neck_joint1_yaw",
        "neck_joint2_pitch",
        "neck_joint3_roll",
        "neck_joint4_yaw",
        "neck_joint5_pitch",
    ]

    # 발끝 body 이름 (FL, FR, HL, HR 순서 → toe_pos_local 순서와 대응)
    BODY_NAMES = [
        "FL_link7_toe",
        "FR_link7_toe",
        "HL_link7_toe",
        "HR_link7_toe",
        "base",
    ]

    def __init__(
        self,
        motion_files: str | list[str],
        device: torch.device | str,
    ) -> None:
        """모션 파일(들)을 로드합니다.

        Args:
            motion_files: txt 파일 경로(문자열) 또는 경로 리스트.
                          디렉토리 경로를 주면 내부의 모든 *.txt 를 사용합니다.
            device: 텐서를 올릴 디바이스.
        """
        self.device = device

        # 파일 목록 수집
        if isinstance(motion_files, str):
            if os.path.isdir(motion_files):
                motion_files = sorted(glob.glob(os.path.join(motion_files, "*.txt")))
            else:
                motion_files = [motion_files]

        assert len(motion_files) > 0, "로드할 모션 파일이 없습니다."

        # 각 파일 파싱 후 리스트로 누적
        all_joint_pos: list[np.ndarray] = []
        all_joint_vel: list[np.ndarray] = []
        all_root_pos: list[np.ndarray] = []
        all_root_rot_wxyz: list[np.ndarray] = []
        all_lin_vel: list[np.ndarray] = []
        all_ang_vel: list[np.ndarray] = []
        all_toe_pos: list[np.ndarray] = []    # (N, 4, 3)
        all_toe_vel: list[np.ndarray] = []    # (N, 4, 3)

        total_dt = None

        for path in motion_files:
            assert os.path.isfile(path), f"파일이 존재하지 않습니다: {path}"
            with open(path) as f:
                data = json.load(f)

            frames = np.array(data["Frames"], dtype=np.float32)  # (N, 113)
            frame_duration = float(data["FrameDuration"])

            if total_dt is None:
                total_dt = frame_duration
            else:
                assert abs(total_dt - frame_duration) < 1e-6, (
                    f"모션 파일 간 FrameDuration 불일치: {total_dt} vs {frame_duration} ({path})"
                )

            # --- 슬라이싱 (113개 포맷) ---
            root_pos = frames[:, self.ROOT_POS_START : self.ROOT_POS_END]   # (N,3)
            root_rot_xyzw = frames[:, self.ROOT_ROT_START : self.ROOT_ROT_END]  # (N,4) xyzw
            joint_pos = frames[:, self.JOINT_POS_START : self.JOINT_POS_END]   # (N,38)
            toe_pos_flat = frames[:, self.TOE_POS_START : self.TOE_POS_END]     # (N,12)
            lin_vel = frames[:, self.LIN_VEL_START : self.LIN_VEL_END]         # (N,3)
            ang_vel = frames[:, self.ANG_VEL_START : self.ANG_VEL_END]         # (N,3)
            joint_vel = frames[:, self.JOINT_VEL_START : self.JOINT_VEL_END]   # (N,38)
            toe_vel_flat = frames[:, self.TOE_VEL_START : self.TOE_VEL_END]    # (N,12)

            # xyzw → wxyz (Isaac Lab 쿼터니언 순서)
            root_rot_wxyz = np.concatenate(
                [root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1
            )  # (N,4)

            # 발끝 (N,12) → (N,4,3)
            # rskeleton_retarget_motion.py의 SIM_TOE_JOINT_IDS = [20, 37, 13, 30]
            # 저장 순서: [FR(20), HR(37), FL(13), HL(30)]
            # BODY_NAMES 기대 순서: [FL, FR, HL, HR]
            # → [FR→idx1, HR→idx3, FL→idx0, HL→idx2]
            # Pybullet
            # 원본: [0=FR, 1=HR, 2=FL, 3=HL]
            # 재정렬 목표: [FL(2), FR(0), HL(3), HR(1)]
            # toe_pos = toe_pos_flat.reshape(-1, self.NUM_TOES, 3)[:, [2, 0, 3, 1], :]
            # toe_vel = toe_vel_flat.reshape(-1, self.NUM_TOES, 3)[:, [2, 0, 3, 1], :]
            # Mujoco
            # 원본: [0=FL, 1=HL, 2=FR, 3=HR]
            # 재정렬 목표: [FL(0), FR(2), HL(1), HR(3)]
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

        # 연결
        joint_pos_all = np.concatenate(all_joint_pos, axis=0)
        joint_vel_all = np.concatenate(all_joint_vel, axis=0)
        root_pos_all = np.concatenate(all_root_pos, axis=0)
        root_rot_all = np.concatenate(all_root_rot_wxyz, axis=0)
        lin_vel_all = np.concatenate(all_lin_vel, axis=0)
        ang_vel_all = np.concatenate(all_ang_vel, axis=0)
        toe_pos_all = np.concatenate(all_toe_pos, axis=0)   # (N,4,3)
        toe_vel_all = np.concatenate(all_toe_vel, axis=0)   # (N,4,3)

        # 텐서 변환 (dof_positions 등 humanoid_amp MotionLoader 와 동일한 attribute)
        self.dof_positions = torch.tensor(joint_pos_all, dtype=torch.float32, device=device)
        self.dof_velocities = torch.tensor(joint_vel_all, dtype=torch.float32, device=device)

        # body_positions: (N, num_bodies, 3)
        # toe_pos (N,4,3) + root_pos (N,1,3) 를 body axis로 스택
        root_pos_t = torch.tensor(root_pos_all, dtype=torch.float32, device=device).unsqueeze(1)
        toe_pos_t = torch.tensor(toe_pos_all, dtype=torch.float32, device=device)
        self.body_positions = torch.cat([toe_pos_t, root_pos_t], dim=1)  # (N,5,3)

        # body_rotations: (N, num_bodies, 4)  — 발끝은 root rot 복사, root는 실제값
        root_rot_t = torch.tensor(root_rot_all, dtype=torch.float32, device=device)
        toe_rot_t = root_rot_t.unsqueeze(1).expand(-1, self.NUM_TOES, -1)
        self.body_rotations = torch.cat([toe_rot_t, root_rot_t.unsqueeze(1)], dim=1)  # (N,5,4)

        # body linear/angular velocities
        lin_vel_t = torch.tensor(lin_vel_all, dtype=torch.float32, device=device)
        ang_vel_t = torch.tensor(ang_vel_all, dtype=torch.float32, device=device)
        toe_vel_t = torch.tensor(toe_vel_all, dtype=torch.float32, device=device)

        # (N, 5, 3): toe 4개 + root
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
            f"SkeletonMotionLoader: 총 {self.num_frames} 프레임 "
            f"({self.duration:.2f}s, dt={self.dt:.4f}s)"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dof_names(self) -> list[str]:
        """R_Skeleton DOF 이름 목록."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """R_Skeleton body 이름 목록."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """DOF 수."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Body 수."""
        return len(self._body_names)

    # ------------------------------------------------------------------
    # 내부 보간 헬퍼
    # ------------------------------------------------------------------

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
        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask.expand_as(q1)] = -q1[neg_mask.expand_as(q1)]
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
        blend = np.clip(blend, 0.0, 1.0)
        return index_0, index_1, blend

    # ------------------------------------------------------------------
    # 공개 API (humanoid_amp MotionLoader 와 동일)
    # ------------------------------------------------------------------

    def sample_times(self, num_samples: int, duration: Optional[float] = None) -> np.ndarray:
        """랜덤 시간 샘플링."""
        duration = self.duration if duration is None else duration
        assert duration <= self.duration, (
            f"요청 duration({duration}) > 모션 duration({self.duration})"
        )
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self,
        num_samples: int,
        times: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
    ) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        """모션 데이터 샘플링.

        Returns:
            (dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel)
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
        """DOF 이름으로 인덱스를 반환합니다.
        
        R_Skeleton 로봇의 실제 조인트 이름(예: FL_joint_1_shoulder_y)을 
        txt 파일의 38개 조인트 인덱스로 자동 매핑합니다.
        txt 파일에 존재하는 4개의 발가락(toe) 인덱스(13, 20, 30, 37)는 제외됩니다.
        """
        indexes = []
        for name in dof_names:
            if name.startswith('N_joint'):
                # N_joint1 ~ N_joint7 -> 0 ~ 6
                j = int(name.split('_')[1].replace('joint', '')) - 1
                idx = 0 + j
            elif name.startswith('FL_joint'):
                # FL_joint1 ~ FL_joint6 -> 7 ~ 12
                j = int(name.split('_')[1].replace('joint', '')) - 1
                idx = 7 + j
            elif name.startswith('FR_joint'):
                # FR_joint1 ~ FR_joint6 -> 14 ~ 19
                j = int(name.split('_')[1].replace('joint', '')) - 1
                idx = 14 + j
            elif name.startswith('W_joint'):
                # W_joint1 ~ W_joint3 -> 21 ~ 23
                j = int(name.split('_')[1].replace('joint', '')) - 1
                idx = 21 + j
            elif name.startswith('HL_joint') or name.startswith('RL_joint'):
                # HL_joint1 ~ HL_joint6 -> 24 ~ 29
                j = int(name.split('_')[1].replace('joint', '')) - 1
                idx = 24 + j
            elif name.startswith('HR_joint') or name.startswith('RR_joint'):
                # HR_joint1 ~ HR_joint6 -> 31 ~ 36
                j = int(name.split('_')[1].replace('joint', '')) - 1
                idx = 31 + j
            else:
                idx = 0
            indexes.append(idx)
        # [7, 14, 0, 21, 8, 15, 1, 22, 9, 16, 2, 23, 10, 17, 3, 24, 31, 11, 18, 4, 25, 32, 12, 19, 5, 26, 33, 6, 27, 34, 28, 35, 29, 36]
        # ['FL_joint1_shoulder_y', 'FR_joint1_shoulder_y', 'N_joint1_neck_y', 'W_joint1_waist_p', 'FL_joint2_shoulder_r', 'FR_joint2_shoulder_r', 'N_joint2_neck_p', 'W_joint2_waist_y', 'FL_joint3_shoulder_p', 'FR_joint3_shoulder_p', 'N_joint3_neck_y', 'W_joint3_waist_r', 'FL_joint4_elbow_p', 'FR_joint4_elbow_p', 'N_joint
        # 4_neck_r', 'HL_joint1_thigh_y', 'HR_joint1_thigh_y', 'FL_joint5_wrist_p', 'FR_joint5_wrist_p', 'N_joint5_neck_y', 'HL_joint2_thigh_r', 'HR_joint2_thigh_r', 'FL_joint6_wrist_r', 'FR_joint6_wrist_r', 'N_joint6_neck_p', 'HL_joint3_thigh_p', 'HR_joint3_thigh_p', 'N_joint7_neck_y', 'HL_joint4_knee_p', 'HR_joint4_kne
        # e_p', 'HL_joint5_ankle_p', 'HR_joint5_ankle_p', 'HL_joint6_ankle_r', 'HR_joint6_ankle_r'] 
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Body 이름으로 인덱스를 반환합니다."""
        indexes = []
        for name in body_names:
            assert name in self._body_names, (
                f"Body 이름 '{name}'이 존재하지 않습니다: {self._body_names}"
            )
            indexes.append(self._body_names.index(name))
        return indexes

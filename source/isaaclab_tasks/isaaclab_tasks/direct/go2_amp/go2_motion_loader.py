"""Go2 모션 파일 로더.

stmr_go2.py가 생성하는 DeepMimic JSON 형식 txt 파일과
MuJoCo pkl 형식 파일을 파싱하여 Isaac Lab AMP 학습에 사용할 수 있는 인터페이스를 제공합니다.

txt 각 프레임 레이아웃 (61개 값):
  [0:3]    root_pos  (x, y, z)
  [3:7]    root_rot  (quat xyzw)
  [7:19]   joint_pos (12개 관절, MuJoCo 순서)
  [19:31]  toe_pos_local (발 4개 × 3, [FL, RL, FR, RR] 순서)
  [31:34]  lin_vel   (base frame)
  [34:37]  ang_vel   (base frame)
  [37:49]  joint_vel (12개)
  [49:61]  toe_vel_local (발 4개 × 3, [FL, RL, FR, RR] 순서)

pkl 각 프레임 레이아웃 (18개 값):
  [0:3]    root_pos  (x, y, z)
  [3:6]    root_euler (roll, pitch, yaw) [rad]
  [6:18]   joint_pos (12개 관절, DOF_NAMES 순서)
  velocities / toe_pos: finite difference 및 FK로 자동 계산
"""

from __future__ import annotations

import glob
import json
import os
import pickle
from typing import Optional

import numpy as np
import torch


class Go2MotionLoader:
    """Go2 DeepMimic JSON txt 모션 파일 로더.

    SkeletonMotionLoader와 동일한 API를 제공합니다.
    """

    # txt 프레임 인덱스 상수 (61개 값)
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

    # body 이름: toe 4개 + base
    # stmr_go2.py TARGET_TOE_BODY_NAMES = ["FL_foot", "RL_foot", "FR_foot", "RR_foot"]
    # 저장 순서: [FL, RL, FR, RR] → 재정렬 [0,2,1,3] → [FL, FR, RL, RR]
    BODY_NAMES = [
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
        "base",
    ]

    # ------------------------------------------------------------------ #
    #  Go2 FK 파라미터 (표준 Unitree Go2 URDF 기준, 검증 오차 < 0.5mm)    #
    # ------------------------------------------------------------------ #
    # 각 다리의 hip joint 위치 (body frame, xyz)
    _HIP_BASE = {
        "FL": np.array([+0.1934, +0.0465, 0.0]),
        "FR": np.array([+0.1934, -0.0465, 0.0]),
        "RL": np.array([-0.1934, +0.0465, 0.0]),
        "RR": np.array([-0.1934, -0.0465, 0.0]),
    }
    # hip frame 내 thigh joint y-offset (좌측 +, 우측 -)
    _THIGH_OFS_Y = {"FL": +0.0955, "FR": -0.0955, "RL": +0.0955, "RR": -0.0955}
    _THIGH_LEN = 0.213
    _CALF_LEN = 0.213

    @staticmethod
    def _Rx(a: np.ndarray) -> np.ndarray:
        """Roll 회전 행렬 (N,) → (N,3,3)."""
        N = len(a)
        R = np.zeros((N, 3, 3))
        R[:, 0, 0] = 1.0
        R[:, 1, 1] = np.cos(a)
        R[:, 1, 2] = -np.sin(a)
        R[:, 2, 1] = np.sin(a)
        R[:, 2, 2] = np.cos(a)
        return R

    @staticmethod
    def _Ry(a: np.ndarray) -> np.ndarray:
        """Pitch 회전 행렬 (N,) → (N,3,3)."""
        N = len(a)
        R = np.zeros((N, 3, 3))
        R[:, 0, 0] = np.cos(a)
        R[:, 0, 2] = np.sin(a)
        R[:, 1, 1] = 1.0
        R[:, 2, 0] = -np.sin(a)
        R[:, 2, 2] = np.cos(a)
        return R

    @classmethod
    def _go2_fk_toe_pos(cls, joint_pos: np.ndarray) -> np.ndarray:
        """Go2 순운동학으로 발 위치(body frame local)를 계산합니다.

        Args:
            joint_pos: (N, 12) — [FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...]

        Returns:
            toe_pos: (N, 4, 3) — [FL, FR, RL, RR] 순서 (BODY_NAMES 순서)
        """
        N = joint_pos.shape[0]
        # 관절 인덱스: DOF_NAMES 순서 = [FL(0-2), FR(3-5), RL(6-8), RR(9-11)]
        leg_slices = {"FL": slice(0, 3), "FR": slice(3, 6), "RL": slice(6, 9), "RR": slice(9, 12)}
        leg_order = ["FL", "FR", "RL", "RR"]

        toe_pos = np.zeros((N, 4, 3))
        _down = np.array([0.0, 0.0, -1.0])

        for idx, leg in enumerate(leg_order):
            s = leg_slices[leg]
            hip_a = joint_pos[:, s.start]        # (N,)
            thigh_a = joint_pos[:, s.start + 1]  # (N,)
            calf_a = joint_pos[:, s.start + 2]   # (N,)

            hip_base = cls._HIP_BASE[leg]         # (3,)
            thigh_ofs = np.array([0.0, cls._THIGH_OFS_Y[leg], 0.0])

            R_hip = cls._Rx(hip_a)    # (N,3,3)
            R_th = cls._Ry(thigh_a)   # (N,3,3)
            R_ca = cls._Ry(calf_a)    # (N,3,3)

            # thigh pivot = hip_base + R_hip @ thigh_ofs
            thigh_pivot = hip_base + np.einsum("nij,j->ni", R_hip, thigh_ofs)  # (N,3)

            # calf pivot = thigh_pivot + R_hip @ R_th @ (0,0,-thigh_len)
            thigh_end_local = _down * cls._THIGH_LEN   # (3,)
            thigh_end_th = np.einsum("nij,j->ni", R_th, thigh_end_local)       # (N,3)
            calf_pivot = thigh_pivot + np.einsum("nij,nj->ni", R_hip, thigh_end_th)  # (N,3)

            # foot = calf_pivot + R_hip @ R_th @ R_ca @ (0,0,-calf_len)
            calf_end_local = _down * cls._CALF_LEN
            calf_end_ca = np.einsum("nij,j->ni", R_ca, calf_end_local)          # (N,3)
            calf_end_th = np.einsum("nij,nj->ni", R_th, calf_end_ca)            # (N,3)
            foot = calf_pivot + np.einsum("nij,nj->ni", R_hip, calf_end_th)     # (N,3)

            toe_pos[:, idx, :] = foot

        return toe_pos.astype(np.float32)

    @staticmethod
    def _euler_to_quat_wxyz(rpy: np.ndarray) -> np.ndarray:
        """Roll-Pitch-Yaw → 쿼터니언 wxyz (ZYX 내재적 회전, scipy 불필요).

        Args:
            rpy: (N, 3) — [roll, pitch, yaw] in radians

        Returns:
            quat: (N, 4) — [w, x, y, z]
        """
        r, p, y = rpy[:, 0] / 2, rpy[:, 1] / 2, rpy[:, 2] / 2
        cr, cp, cy = np.cos(r), np.cos(p), np.cos(y)
        sr, sp, sy = np.sin(r), np.sin(p), np.sin(y)
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        yq = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        return np.stack([w, x, yq, z], axis=-1).astype(np.float32)

    @staticmethod
    def _finite_diff(arr: np.ndarray, dt: float, loop: bool = False) -> np.ndarray:
        """Forward finite difference로 속도 계산 (N, D) → (N, D).

        마지막 프레임은 loop=True이면 wraparound, 아니면 이전 프레임 복사.
        """
        vel = np.zeros_like(arr)
        vel[:-1] = (arr[1:] - arr[:-1]) / dt
        if loop:
            vel[-1] = (arr[0] - arr[-1]) / dt
        else:
            vel[-1] = vel[-2]
        return vel

    def _load_pkl_file(self, path: str) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray,
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float
    ]:
        """pkl 파일을 로드하고 txt 로더와 동일한 출력을 반환합니다.

        Returns:
            (root_pos, root_rot_wxyz, joint_pos, joint_vel,
             lin_vel, ang_vel, toe_pos, toe_vel, frame_duration)
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        fps: int = int(data["fps"])
        dt = 1.0 / fps
        frames = np.array(data["frames"], dtype=np.float32)  # (N, 18)

        assert frames.shape[1] == 18, f"pkl 프레임 크기 불일치: {frames.shape[1]} != 18 ({path})"

        root_pos = frames[:, 0:3]    # (N, 3)
        root_euler = frames[:, 3:6]  # (N, 3) — roll, pitch, yaw
        joint_pos = frames[:, 6:18]  # (N, 12)

        # euler → quaternion wxyz
        root_rot_wxyz = self._euler_to_quat_wxyz(root_euler)  # (N, 4)

        # 속도: finite difference (loop=False — 위치는 절대 좌표라 wraparound 불가)
        lin_vel = self._finite_diff(root_pos, dt, loop=False)        # (N, 3)
        ang_vel = self._finite_diff(root_euler, dt, loop=False)      # (N, 3)
        joint_vel = self._finite_diff(joint_pos, dt, loop=False)     # (N, 12)

        # 발 위치: FK
        toe_pos = self._go2_fk_toe_pos(joint_pos)                    # (N, 4, 3)
        toe_vel = np.zeros_like(toe_pos)
        toe_vel_flat = self._finite_diff(toe_pos.reshape(len(toe_pos), -1), dt, loop=False)
        toe_vel = toe_vel_flat.reshape(len(toe_pos), 4, 3)           # (N, 4, 3)

        return root_pos, root_rot_wxyz, joint_pos, joint_vel, lin_vel, ang_vel, toe_pos, toe_vel, dt

    def __init__(
        self,
        motion_files: str | list[str],
        device: torch.device | str,
    ) -> None:
        self.device = device

        if isinstance(motion_files, str):
            if os.path.isdir(motion_files):
                txt_files = sorted(glob.glob(os.path.join(motion_files, "*.txt")))
                pkl_files = sorted(glob.glob(os.path.join(motion_files, "*.pkl")))
                motion_files = txt_files + pkl_files
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
            ext = os.path.splitext(path)[1].lower()

            if ext == ".pkl":
                # ---- pkl 로딩 경로 ----
                (root_pos, root_rot_wxyz, joint_pos, joint_vel,
                 lin_vel, ang_vel, toe_pos, toe_vel, frame_duration) = self._load_pkl_file(path)
                num_frames_file = root_pos.shape[0]
            else:
                # ---- txt (JSON) 로딩 경로 ----
                with open(path) as f:
                    data = json.load(f)

                frames = np.array(data["Frames"], dtype=np.float32)  # (N, 61)
                frame_duration = float(data["FrameDuration"])
                num_frames_file = frames.shape[0]

                root_pos = frames[:, self.ROOT_POS_START : self.ROOT_POS_END]
                root_rot_xyzw = frames[:, self.ROOT_ROT_START : self.ROOT_ROT_END]
                joint_pos = frames[:, self.JOINT_POS_START : self.JOINT_POS_END]
                toe_pos_flat = frames[:, self.TOE_POS_START : self.TOE_POS_END]
                lin_vel = frames[:, self.LIN_VEL_START : self.LIN_VEL_END]
                ang_vel = frames[:, self.ANG_VEL_START : self.ANG_VEL_END]
                joint_vel = frames[:, self.JOINT_VEL_START : self.JOINT_VEL_END]
                toe_vel_flat = frames[:, self.TOE_VEL_START : self.TOE_VEL_END]

                # xyzw → wxyz
                root_rot_wxyz = np.concatenate(
                    [root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1
                )
                # stmr_go2.py 저장 순서: [FL=0, RL=1, FR=2, RR=3] → [FL, FR, RL, RR]
                toe_pos = toe_pos_flat.reshape(-1, self.NUM_TOES, 3)[:, [0, 2, 1, 3], :]
                toe_vel = toe_vel_flat.reshape(-1, self.NUM_TOES, 3)[:, [0, 2, 1, 3], :]

            if total_dt is None:
                total_dt = frame_duration
            else:
                assert abs(total_dt - frame_duration) < 1e-3, (
                    f"FrameDuration 불일치: {total_dt} vs {frame_duration} ({path})"
                )

            all_root_pos.append(root_pos)
            all_root_rot_wxyz.append(root_rot_wxyz)
            all_joint_pos.append(joint_pos)
            all_joint_vel.append(joint_vel)
            all_lin_vel.append(lin_vel)
            all_ang_vel.append(ang_vel)
            all_toe_pos.append(toe_pos)
            all_toe_vel.append(toe_vel)

            print(
                f"모션 로드 ({os.path.basename(path)}, {ext}): "
                f"{num_frames_file} 프레임 ({num_frames_file * frame_duration:.2f}s)"
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

        # body_positions: toe 4개(local) + base(world) → (N, 5, 3)
        root_pos_t = torch.tensor(root_pos_all, dtype=torch.float32, device=device).unsqueeze(1)
        toe_pos_t = torch.tensor(toe_pos_all, dtype=torch.float32, device=device)
        self.body_positions = torch.cat([toe_pos_t, root_pos_t], dim=1)  # (N,5,3)

        # body_rotations: toe는 root rot 복사, base는 실제값 → (N, 5, 4)
        root_rot_t = torch.tensor(root_rot_all, dtype=torch.float32, device=device)
        toe_rot_t = root_rot_t.unsqueeze(1).expand(-1, self.NUM_TOES, -1)
        self.body_rotations = torch.cat([toe_rot_t, root_rot_t.unsqueeze(1)], dim=1)  # (N,5,4)

        # body velocities: (N, 5, 3)
        lin_vel_t = torch.tensor(lin_vel_all, dtype=torch.float32, device=device)
        ang_vel_t = torch.tensor(ang_vel_all, dtype=torch.float32, device=device)
        toe_vel_t = torch.tensor(toe_vel_all, dtype=torch.float32, device=device)

        self.body_linear_velocities = torch.cat([toe_vel_t, lin_vel_t.unsqueeze(1)], dim=1)
        self.body_angular_velocities = torch.cat(
            [torch.zeros_like(toe_vel_t), ang_vel_t.unsqueeze(1)], dim=1
        )

        self._dof_names = self.DOF_NAMES
        self._body_names = self.BODY_NAMES

        self.dt = total_dt
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)

        # per-frame forward velocity 저장 (RSI-command 매칭용)
        self._frame_lin_vel_x = lin_vel_all[:, 0].copy()

        # 속도 구간별 균등 샘플링 가중치 사전 계산
        self._vel_sample_weights = self._build_velocity_sample_weights(lin_vel_all)

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

    def _build_velocity_sample_weights(self, lin_vel_all: np.ndarray, num_bins: int = 5) -> np.ndarray:
        """속도 구간별 균등 샘플링 가중치를 계산합니다.

        전체 프레임을 num_bins개의 속도 구간으로 나누고, 각 구간에 동일한 총 가중치를
        부여합니다. 고속 프레임이 적더라도 저속 프레임과 동일한 비율로 샘플링됩니다.
        """
        speeds = np.abs(lin_vel_all[:, 0])  # forward (x) 속도 크기
        max_speed = speeds.max()
        bin_edges = np.linspace(0.0, max_speed + 1e-6, num_bins + 1)
        bin_ids = np.clip(np.digitize(speeds, bin_edges) - 1, 0, num_bins - 1)

        weights = np.zeros(len(speeds), dtype=np.float64)
        occupied = 0
        for b in range(num_bins):
            mask = bin_ids == b
            count = int(mask.sum())
            if count > 0:
                occupied += 1
                weights[mask] = 1.0 / count  # 구간 내 균등

        weights /= weights.sum()  # 전체 합 = 1

        # 로그: 구간별 프레임 수와 속도 범위 출력
        print("Go2MotionLoader: velocity-balanced sampling weights")
        for b in range(num_bins):
            mask = bin_ids == b
            count = int(mask.sum())
            lo, hi = bin_edges[b], bin_edges[b + 1]
            pct = 100.0 * count / len(speeds)
            w_pct = 100.0 * weights[mask].sum() if count > 0 else 0.0
            print(f"  bin {b} [{lo:.1f}, {hi:.1f}) m/s: {count:4d} frames ({pct:5.1f}%) → sample prob {w_pct:.1f}%")

        return weights.astype(np.float64)

    def sample_times(self, num_samples: int, duration: Optional[float] = None, velocity_balanced: bool = True) -> np.ndarray:
        if velocity_balanced:
            frame_indices = np.random.choice(self.num_frames, size=num_samples, p=self._vel_sample_weights)
            return (frame_indices * float(self.dt)).astype(np.float32)  # type: ignore[arg-type]
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
    ) -> tuple[torch.Tensor, ...]:
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
        """Isaac Sim joint 이름으로 motion data DOF 인덱스를 반환합니다."""
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, (
                f"DOF 이름 '{name}'이 존재하지 않습니다: {self._dof_names}"
            )
            indexes.append(self._dof_names.index(name))
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

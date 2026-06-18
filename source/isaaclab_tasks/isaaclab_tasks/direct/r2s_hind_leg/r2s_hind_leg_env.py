# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""R_Skeleton Hind Leg Real2Sim 환경 — 순수 포지션 제어 테스트."""

from __future__ import annotations

import math

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .r2s_hind_leg_env_cfg import JOINT_NAME_PATTERNS, NUM_JOINTS, V_MAX_RAD, R2SHindLegEnvCfg


class R2SHindLegEnv(DirectRLEnv):
    """R_Skeleton Hind Leg Real2Sim 환경.

    RL 없음. 순수 포지션 제어 테스트용.
    외부에서 set_setpoint()로 관절 목표 각도(rad)를 주입하면
    slew rate limiter를 통해 안전하게 적용됩니다.

    Slew rate: max_step[i] = V_MAX_RAD[i] / control_freq (1/50 s)
    """

    cfg: R2SHindLegEnvCfg

    def __init__(self, cfg: R2SHindLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스 매핑 (USD 로드 순서 독립)
        self._joint_ids, found_names = self.robot.find_joints(JOINT_NAME_PATTERNS, preserve_order=True)
        assert len(self._joint_ids) == NUM_JOINTS, (
            f"관절 {NUM_JOINTS}개 필요, {len(self._joint_ids)}개 발견.\n"
            f"찾은 이름: {found_names}\n"
            f"전체 관절: {list(self.robot.data.joint_names)}"
        )
        print(f"[R2SHindLegEnv] 관절 매핑: {list(zip(JOINT_NAME_PATTERNS, found_names))}")

        # slew rate 한계 (rad per control step = 1/50 s)
        control_freq = 1.0 / (self.cfg.sim.dt * self.cfg.decimation)
        self._max_step = torch.tensor([v / control_freq for v in V_MAX_RAD], device=self.device, dtype=torch.float32)

        # 새 버퍼 — _reset_idx에서 초기화 필수
        self._setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._prev_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

    # ------------------------------------------------------------------
    # 외부 인터페이스 (sim_runner.py에서 호출)
    # ------------------------------------------------------------------

    def set_setpoint(self, q_rad: torch.Tensor) -> None:
        """관절 목표 각도(rad) 주입. shape: (num_envs, 5) 또는 (5,)."""
        if q_rad.dim() == 1:
            q_rad = q_rad.unsqueeze(0).expand(self.num_envs, -1)
        self._setpoint.copy_(q_rad.to(self.device))  # in-place: 버퍼 참조 유지

    def get_state(self) -> dict:
        """현재 관절 state 반환 (numpy). num_envs==1 가정."""
        return {
            "pos": self.robot.data.joint_pos[0, self._joint_ids].cpu().numpy(),
            "vel": self.robot.data.joint_vel[0, self._joint_ids].cpu().numpy(),
            "torque": self.robot.data.applied_torque[0, self._joint_ids].cpu().numpy(),
            "timestamp": float(self.episode_length_buf[0].item()) * self.cfg.sim.dt,
        }

    def get_applied_setpoint(self) -> torch.Tensor:
        """slew 적용 후 실제 적용된 setpoint 반환."""
        return self._prev_setpoint.squeeze(0)

    # ------------------------------------------------------------------
    # DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        del actions  # setpoint은 self._setpoint에서 직접 주입
        # Slew rate limiter: 한 step에서 max_step 이상 이동 불가
        delta = (self._setpoint - self._prev_setpoint).clamp(-self._max_step, self._max_step)
        smoothed = self._prev_setpoint + delta
        self._prev_setpoint = smoothed.clone()
        self.robot.set_joint_position_target(smoothed, joint_ids=self._joint_ids)

    def _apply_action(self) -> None:
        pass  # _pre_physics_step에서 처리

    def _get_observations(self) -> dict:
        pos = self.robot.data.joint_pos[:, self._joint_ids]
        vel = self.robot.data.joint_vel[:, self._joint_ids]
        torque = self.robot.data.applied_torque[:, self._joint_ids]
        return {"policy": torch.cat([pos, vel, torque], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        # RL 없음 — stub
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        """새 버퍼 초기화 — CLAUDE.md 전역 DO 규칙."""
        super()._reset_idx(env_ids)
        # default_joint_pos에서 인덱싱하여 reset
        default_pos = self.robot.data.default_joint_pos[:, self._joint_ids]
        self._setpoint[env_ids] = default_pos[env_ids]
        self._prev_setpoint[env_ids] = default_pos[env_ids]

    @staticmethod
    def deg_to_rad(deg: float) -> float:
        return deg * math.pi / 180.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 Real2Sim 환경 — 순수 포지션 제어 테스트."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse

from .r2s_go2_env_cfg import (
    DEFAULT_KD,
    DEFAULT_KP,
    FIXED_BASE_HEIGHT_M,
    JOINT_ORDER,
    NUM_JOINTS,
    V_MAX_RAD,
    R2SGo2EnvCfg,
)


class R2SGo2Env(DirectRLEnv):
    """R2S-GO2 Real2Sim 환경.

    RL 없음. sim_runner_go2.py가 UDP로 받은 PD 목표를 set_setpoint()로 주입하면
    slew rate limiter를 통해 안전하게 적용하고, get_lowstate()로 현재 상태를
    Unitree LowState 스키마(CONTRACT §3, §4.2)와 맞춰 반환합니다.

    Slew rate: max_step[i] = V_MAX_RAD[i] / control_freq (1/50 s)
    """

    cfg: R2SGo2EnvCfg

    def __init__(self, cfg: R2SGo2EnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스 매핑 (CONTRACT §2 고정 순서, USD 로드 순서 독립)
        self._joint_ids, found_names = self.robot.find_joints(JOINT_ORDER, preserve_order=True)
        assert len(self._joint_ids) == NUM_JOINTS, (
            f"관절 {NUM_JOINTS}개 필요, {len(self._joint_ids)}개 발견.\n"
            f"찾은 이름: {found_names}\n"
            f"전체 관절: {list(self.robot.data.joint_names)}"
        )
        assert found_names == JOINT_ORDER, (
            f"관절 순서가 CONTRACT §2 고정 순서와 다름.\n찾은 순서: {found_names}\n기대 순서: {JOINT_ORDER}"
        )
        print(f"[R2SGo2Env] 관절 매핑: {list(zip(JOINT_ORDER, found_names))}")

        # base body 인덱스 (IMU 가속도 조회용)
        base_ids, base_names = self.robot.find_bodies("base")
        assert len(base_ids) == 1, f"base body 1개 필요, {len(base_ids)}개 발견: {base_names}"
        self._base_id = base_ids[0]

        # slew rate 한계 (rad per control step = 1/50 s)
        control_freq = 1.0 / (self.cfg.sim.dt * self.cfg.decimation)
        self._max_step = torch.tensor([v / control_freq for v in V_MAX_RAD], device=self.device, dtype=torch.float32)

        # 새 버퍼 — _reset_idx에서 초기화 필수 (CLAUDE.md 전역 DO 규칙)
        self._setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._prev_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        # dq/kp/kd/tau는 M1에서 시뮬레이션에 미적용(향후 faithful 모드 seam) — 버퍼 저장만 (CONTRACT §5).
        self._dq_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._kp = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._kd = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._tau_ff = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

    # ------------------------------------------------------------------
    # 외부 인터페이스 (sim_runner_go2.py에서 호출)
    # ------------------------------------------------------------------

    def _to_tensor(self, x: torch.Tensor | list[float]) -> torch.Tensor:
        """12-벡터를 (num_envs, 12) torch 텐서로 정규화."""
        t = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        return t

    def set_setpoint(
        self,
        q: torch.Tensor | list[float],
        dq: torch.Tensor | list[float],
        kp: torch.Tensor | list[float],
        kd: torch.Tensor | list[float],
        tau: torch.Tensor | list[float],
    ) -> None:
        """PD 목표 주입 (CONTRACT §5). shape: (num_envs, 12) 또는 (12,), JOINT_ORDER 순서.

        M1 기본 동작: q만 slew rate limiter를 통과해 position target으로 실제 적용된다.
        dq/kp/kd/tau는 버퍼에 저장만 되고 시뮬레이션에는 반영되지 않는다(향후 faithful PD 모드 seam).

        Args:
            q: 목표 관절각 [rad].
            dq: 목표 관절각속도 [rad/s]. 현재 미적용.
            kp: 위치 게인. 현재 미적용(cfg 액추에이터 PD 고정).
            kd: 속도 게인. 현재 미적용.
            tau: 피드포워드 토크 [Nm]. 현재 미적용.
        """
        self._setpoint.copy_(self._to_tensor(q))
        self._dq_setpoint.copy_(self._to_tensor(dq))
        self._kp.copy_(self._to_tensor(kp))
        self._kd.copy_(self._to_tensor(kd))
        self._tau_ff.copy_(self._to_tensor(tau))

    def get_lowstate(self) -> dict:
        """현재 lowstate 반환 (numpy). num_envs==1 가정 (CONTRACT §5, §4.2).

        Returns:
            dict with keys:
                q, dq, ddq, tau_est: 각 길이 12, JOINT_ORDER 순서.
                imu: 길이 10 — quat_w,quat_x,quat_y,quat_z, gyro_xyz, acc_xyz.
        """
        q = self.robot.data.joint_pos[0, self._joint_ids]
        dq = self.robot.data.joint_vel[0, self._joint_ids]
        ddq = self.robot.data.joint_acc[0, self._joint_ids]
        tau_est = self.robot.data.applied_torque[0, self._joint_ids]

        # root_quat_w는 IsaacLab 6.0부터 (x,y,z,w) 순서. Unitree LowState는 (w,x,y,z) 순이라 재배열.
        quat_xyzw = self.robot.data.root_quat_w[0]
        quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
        gyro = self.robot.data.root_ang_vel_b[0]
        # base link 선가속도: body_lin_acc_w(world frame, shape (num_envs, num_bodies, 3))를
        # base id로 인덱싱한 뒤 root 회전의 역으로 body frame으로 변환(accelerometer 관례).
        acc_w = self.robot.data.body_lin_acc_w[0, self._base_id]
        acc_b = quat_apply_inverse(quat_xyzw.unsqueeze(0), acc_w.unsqueeze(0)).squeeze(0)
        imu = torch.cat([quat_wxyz, gyro, acc_b])

        return {
            "q": q.cpu().numpy(),
            "dq": dq.cpu().numpy(),
            "ddq": ddq.cpu().numpy(),
            "tau_est": tau_est.cpu().numpy(),
            "imu": imu.cpu().numpy(),
        }

    # ------------------------------------------------------------------
    # DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        # fix_base 적용 (CONTRACT §5): env 빌드 시점에 cfg.fix_base를 읽어야
        # sim_runner_go2.py가 --fix_base로 cfg 생성 후 덮어쓴 값도 반영된다.
        # configclass가 robot cfg를 인스턴스마다 deepcopy하므로 전역 UNITREE_GO2_CFG 오염 없음.
        if self.cfg.fix_base:
            self.cfg.robot.spawn.articulation_props.fix_root_link = True
            self.cfg.robot.init_state.pos = (0.0, 0.0, FIXED_BASE_HEIGHT_M)
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
        if self.cfg.sysid:
            # sysid 모드: actions = 절대 관절 목표각 [rad], **articulation 관절 순서**.
            # PACE(fit.py / collect_chirp_sim.py)가 관절 인덱스로 직접 채워 넣는 규약이며,
            # slew limiter는 chirp 고주파를 왜곡하므로 우회한다.
            self.robot.set_joint_position_target_index(target=actions)
            return
        del actions  # live 모드: setpoint은 self._setpoint에서 직접 주입
        # Slew rate limiter: 한 step에서 max_step 이상 이동 불가
        delta = (self._setpoint - self._prev_setpoint).clamp(-self._max_step, self._max_step)
        smoothed = self._prev_setpoint + delta
        self._prev_setpoint = smoothed.clone()
        self.robot.set_joint_position_target_index(target=smoothed, joint_ids=self._joint_ids)

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
        default_pos = self.robot.data.default_joint_pos[:, self._joint_ids]
        self._setpoint[env_ids] = default_pos[env_ids]
        self._prev_setpoint[env_ids] = default_pos[env_ids]
        self._dq_setpoint[env_ids] = 0.0
        self._kp[env_ids] = DEFAULT_KP
        self._kd[env_ids] = DEFAULT_KD
        self._tau_ff[env_ids] = 0.0

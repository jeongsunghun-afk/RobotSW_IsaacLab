# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Biped Leg Real2Sim 환경 — 순수 포지션 제어 테스트 (r2s_hind_leg 패턴 기반).

RL 없음. sim_runner_bipedleg.py가 UDP로 받은 PD 목표를 set_setpoint()로 주입하면
slew rate limiter를 통해 안전하게 적용하고, get_lowstate()로 현재 상태를 반환한다.
8-DOF 2족(`HIND_LEG_CFG`)이며, 자유베이스이므로 `fix_base` 토글을 지원한다(IMU/base 상태는 미노출).

계약: source/isaaclab_tasks/isaaclab_tasks/direct/r2s_biped_leg/CONTRACT.md
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from .r2s_biped_leg_env_cfg import (
    DEFAULT_KD,
    DEFAULT_KP,
    FIXED_BASE_HEIGHT_M,
    JOINT_NAME_PATTERNS,
    NUM_JOINTS,
    V_MAX_RAD,
    R2SBipedLegEnvCfg,
)


class R2SBipedLegEnv(DirectRLEnv):
    """Biped Leg Real2Sim 환경 (8-DOF 2족).

    외부에서 :meth:`set_setpoint` 로 관절 PD 목표를 주입하면 slew rate limiter를 통해
    position target으로 적용한다. :attr:`R2SBipedLegEnvCfg.faithful_pd` 가 True면 kp/kd도
    실제 sim drive 게인으로 반영한다.

    Slew rate: ``max_step[i] = V_MAX_RAD[i] / control_freq (1/50 s)``
    """

    cfg: R2SBipedLegEnvCfg

    def __init__(self, cfg: R2SBipedLegEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # 관절 인덱스 매핑 (USD 로드 순서 독립, CONTRACT §2 leg-major 순서 고정)
        self._joint_ids, found_names = self.robot.find_joints(JOINT_NAME_PATTERNS, preserve_order=True)
        assert len(self._joint_ids) == NUM_JOINTS, (
            f"관절 {NUM_JOINTS}개 필요, {len(self._joint_ids)}개 발견.\n"
            f"찾은 이름: {found_names}\n전체 관절: {list(self.robot.data.joint_names)}"
        )

        # slew rate 한계 (rad per control step = 1/50 s)
        control_freq = 1.0 / (self.cfg.sim.dt * self.cfg.decimation)
        self._max_step = torch.tensor([v / control_freq for v in V_MAX_RAD], device=self.device, dtype=torch.float32)

        # 새 버퍼 — _reset_idx에서 초기화 필수 (CLAUDE.md 전역 DO)
        self._setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._prev_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._dq_setpoint = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._kp = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._kd = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._tau_ff = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        # faithful PD: kp/kd가 바뀔 때만 sim drive 게인을 다시 쓴다(매 step write 회피).
        self._applied_kp = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)
        self._applied_kd = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)

    # ------------------------------------------------------------------
    # 외부 인터페이스 (sim_runner_bipedleg.py에서 호출)
    # ------------------------------------------------------------------

    def set_setpoint(
        self,
        q: torch.Tensor | list[float],
        dq: torch.Tensor | list[float],
        kp: torch.Tensor | list[float],
        kd: torch.Tensor | list[float],
        tau: torch.Tensor | list[float],
    ) -> None:
        """PD 목표 주입 (CONTRACT §5). shape: (num_envs, 8) 또는 (8,), JOINT 순서.

        q는 slew rate limiter를 통과해 position target으로 적용된다. kp/kd는 :attr:`faithful_pd`
        가 True면 sim drive 게인으로 반영되고, dq/tau는 버퍼에 저장만 된다(향후 seam).

        Args:
            q: 목표 관절각 [rad].
            dq: 목표 관절각속도 [rad/s]. 현재 미적용.
            kp: 위치 게인. faithful_pd=True면 반영.
            kd: 속도 게인. faithful_pd=True면 반영.
            tau: 피드포워드 토크 [Nm]. 현재 미적용.
        """
        self._setpoint.copy_(self._to_tensor(q))
        self._dq_setpoint.copy_(self._to_tensor(dq))
        self._kp.copy_(self._to_tensor(kp))
        self._kd.copy_(self._to_tensor(kd))
        self._tau_ff.copy_(self._to_tensor(tau))

    def get_lowstate(self) -> dict:
        """현재 상태 반환 (numpy). num_envs==1 가정 (CONTRACT §5).

        Returns:
            dict — q, dq, ddq, tau_est (각 길이 8, JOINT 순서). IMU/base 상태는 미노출.
        """
        return {
            "q": self.robot.data.joint_pos[0, self._joint_ids].cpu().numpy(),
            "dq": self.robot.data.joint_vel[0, self._joint_ids].cpu().numpy(),
            "ddq": self.robot.data.joint_acc[0, self._joint_ids].cpu().numpy(),
            "tau_est": self.robot.data.applied_torque[0, self._joint_ids].cpu().numpy(),
        }

    def _to_tensor(self, x: torch.Tensor | list[float]) -> torch.Tensor:
        t = torch.as_tensor(x, device=self.device, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        return t

    # ------------------------------------------------------------------
    # DirectRLEnv 필수 메서드
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        # fix_base 적용 (CONTRACT §5): env 빌드 시점에 cfg.fix_base를 읽어야
        # sim_runner_bipedleg.py가 --fix_base로 cfg 생성 후 덮어쓴 값도 반영된다.
        # configclass가 robot cfg를 인스턴스마다 deepcopy하므로 전역 HIND_LEG_CFG 오염 없음.
        # 2족 자유베이스라 fix_base=False면 GUI로 관절을 스텝하는 순간 즉시 넘어진다 —
        # 관절 추종 확인이 목적이므로 런처 기본값은 fix_base=True.
        if self.cfg.fix_base:
            self.cfg.robot.spawn.articulation_props.fix_root_link = True
            self.cfg.robot.init_state.pos = (0.0, 0.0, FIXED_BASE_HEIGHT_M)
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
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
            # PACE(fit_bipedleg.py / collect_chirp_sim_bipedleg.py)가 관절 인덱스로 직접 채워 넣는
            # 규약이며, slew limiter는 chirp 고주파를 왜곡하므로 우회한다.
            # 게인도 여기서 건드리지 않는다 — CMA-ES가 actuator.stiffness 텐서를 직접 쓴다.
            self.robot.set_joint_position_target_index(target=actions)
            return
        del actions  # live 모드: setpoint은 self._setpoint에서 직접 주입
        # faithful PD: kp/kd가 바뀌었으면 sim drive 게인 갱신.
        if self.cfg.faithful_pd:
            if not torch.equal(self._kp, self._applied_kp) or not torch.equal(self._kd, self._applied_kd):
                self._write_pd_gains(self._kp, self._kd)
                self._applied_kp.copy_(self._kp)
                self._applied_kd.copy_(self._kd)
        # Slew rate limiter: 한 step에서 max_step 이상 이동 불가
        delta = (self._setpoint - self._prev_setpoint).clamp(-self._max_step, self._max_step)
        smoothed = self._prev_setpoint + delta
        self._prev_setpoint = smoothed.clone()
        self.robot.set_joint_position_target_index(target=smoothed, joint_ids=self._joint_ids)

    def _write_pd_gains(self, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """GUI kp/kd를 **실제** PD 게인에 반영한다 (액추에이터 모델별로 경로가 다르다).

        explicit actuator(:class:`~isaaclab.actuators.DCMotor` 계열 — 현 `HIND_LEG_CFG`)는 PD를
        PhysX가 아니라 파이썬에서 계산하므로 진짜 게인은 ``actuator.stiffness`` / ``actuator.damping``
        텐서다. 이때 :meth:`write_joint_stiffness_to_sim` 을 쓰면 게인이 반영되지 않을 뿐 아니라
        **PhysX 드라이브가 켜져서** explicit 토크와 동시에 작용한다. 드라이브 목표가 actuator가 쓰는
        목표와 어긋나 있으면 두 스프링이 서로 당겨 관절이 목표의 절반에서 평형을 이룬다
        (2026-07-22 실측: 8관절 전부 ``q ≈ target/2``).

        implicit actuator는 반대로 PhysX 쪽이 진짜 게인이므로 기존 경로를 그대로 쓴다.

        Args:
            kp: 위치 게인, shape (num_envs, NUM_JOINTS), JOINT 순서.
            kd: 속도 게인, shape (num_envs, NUM_JOINTS), JOINT 순서.
        """
        joint_ids_t = torch.as_tensor(self._joint_ids, device=self.device, dtype=torch.long)
        for actuator in self.robot.actuators.values():
            drive_ids = actuator.joint_indices
            if isinstance(drive_ids, slice):
                drive_ids = torch.arange(self.robot.num_joints, device=self.device)[drive_ids]
            drive_ids = torch.as_tensor(drive_ids, device=self.device, dtype=torch.long)
            # drive_ids[i] = articulation 관절 인덱스 -> JOINT 순서에서의 위치
            order_pos = torch.tensor(
                [int((joint_ids_t == int(j)).nonzero()[0]) for j in drive_ids],
                dtype=torch.long,
                device=self.device,
            )
            if actuator.is_implicit_model:
                self.robot.write_joint_stiffness_to_sim(kp[:, order_pos], joint_ids=drive_ids.tolist())
                self.robot.write_joint_damping_to_sim(kd[:, order_pos], joint_ids=drive_ids.tolist())
            else:
                actuator.stiffness[:] = kp[:, order_pos]
                actuator.damping[:] = kd[:, order_pos]

    def _apply_action(self) -> None:
        pass  # _pre_physics_step에서 처리

    def _get_observations(self) -> dict:
        pos = self.robot.data.joint_pos[:, self._joint_ids]
        vel = self.robot.data.joint_vel[:, self._joint_ids]
        torque = self.robot.data.applied_torque[:, self._joint_ids]
        return {"policy": torch.cat([pos, vel, torque], dim=-1)}

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)  # RL 없음 — stub

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
        self._kp[env_ids] = torch.tensor(DEFAULT_KP, device=self.device)
        self._kd[env_ids] = torch.tensor(DEFAULT_KD, device=self.device)
        self._tau_ff[env_ids] = 0.0

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
        # policy 모드 target — **articulation 순서 전체 관절** (leg-major _joint_ids 재매핑 없음).
        # 학습 정책은 articulation 순서로 obs/action을 봤으므로 그대로 apply해야 순서가 맞는다.
        self._policy_target = torch.zeros(self.num_envs, self.robot.num_joints, device=self.device)

        # foot↔calf 전달기구 커플링 (cfg.foot_coupling, RL_INTERFACE.md coef=+1) — live 모드 전용.
        # leg-major (calf, foot) 쌍: (2,3)=HL, (6,7)=HR.
        self._couple_calf_lm = [2, 6]
        self._couple_foot_lm = [3, 7]
        self._calf_art_ids = [self._joint_ids[i] for i in self._couple_calf_lm]
        self._foot_art_ids = [self._joint_ids[i] for i in self._couple_foot_lm]
        # 새 버퍼 — _reset_idx에서 초기화 필수 (CLAUDE.md 전역 DO)
        self._live_target = torch.zeros(self.num_envs, NUM_JOINTS, device=self.device)  # slew 통과 목표
        self._foot_hold = torch.zeros(self.num_envs, 2, dtype=torch.bool, device=self.device)
        self._raw_latch = torch.zeros(self.num_envs, 2, device=self.device)
        self._foot_kp_eff = torch.zeros(self.num_envs, 2, device=self.device)
        self._foot_kd_eff = torch.zeros(self.num_envs, 2, device=self.device)
        # foot 토크 한계 (전치 토크 예측 클램프용) — actuator 그룹에서 수집.
        tau_max = torch.full((2,), float("inf"), device=self.device)
        for actuator in self.robot.actuators.values():
            ids = actuator.joint_indices
            if isinstance(ids, slice):
                ids = list(range(self.robot.num_joints))
            ids = [int(i) for i in ids]
            for k, fid in enumerate(self._foot_art_ids):
                if int(fid) in ids:
                    tau_max[k] = float(actuator.effort_limit[0, ids.index(int(fid))])
        self._foot_tau_max = tau_max

        # 스톡 플랜트 물성 캡처 (set_plant_params mode=0 복원용) — cfg 적용 직후의 값 [leg-major].
        # 6.0 data의 default_* 컨테이너는 warp 프론트엔드일 수 있어 .torch 뷰로 접근한다.
        jid_long = torch.as_tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._plant_joint_ids_i32 = torch.as_tensor(self._joint_ids, dtype=torch.int32, device=self.device)
        self._stock_armature = self._data_tensor(self.robot.data.default_joint_armature)[:, jid_long].clone()
        self._stock_coulomb = self._data_tensor(self.robot.data.default_joint_friction_coeff)[:, jid_long].clone()
        self._stock_viscous = self._data_tensor(self.robot.data.default_joint_viscous_friction_coeff)[
            :, jid_long
        ].clone()

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
        out = {
            "q": self.robot.data.joint_pos[0, self._joint_ids].cpu().numpy(),
            "dq": self.robot.data.joint_vel[0, self._joint_ids].cpu().numpy(),
            "ddq": self.robot.data.joint_acc[0, self._joint_ids].cpu().numpy(),
            "tau_est": self.robot.data.applied_torque[0, self._joint_ids].cpu().numpy(),
        }
        if self.cfg.foot_coupling:
            # foot은 raw각(q_foot+q_calf)으로 보고 — 실기 TELEM(커플링 미해제 모터각)과 의미를 맞춰
            # GUI 슬라이더 래치/monitor 비교가 같은 좌표가 되게 한다. tau는 관절토크=모터토크라 그대로.
            for c, f in zip(self._couple_calf_lm, self._couple_foot_lm):
                out["q"][f] += out["q"][c]
                out["dq"][f] += out["dq"][c]
                out["ddq"][f] += out["ddq"][c]
        return out

    def get_joint_ieff(self) -> torch.Tensor:
        """관절별 유효 관성 [kg·m²]을 반환 (leg-major, 길이 8).

        현재 자세에서의 generalized mass matrix 대각 성분이다. GUI의 계산 게인
        (kp = I·ωn², kd = 2ζ·I·ωn) 초기값 산정용 — 자세 의존이므로 default 자세
        (reset 직후) 기준으로 쓰는 것을 권장한다.

        Returns:
            관절별 유효 관성 [kg·m²], shape [8], leg-major 순서, CPU 텐서.
        """
        # (count, N, N). fix_base면 N=num_joints, 자유베이스면 N=num_joints+6(root 6-DOF가 앞).
        # 6.0 physx view는 warp 프론트엔드라 wp.array를 반환한다 → torch로 변환.
        m_all = self.robot.root_physx_view.get_generalized_mass_matrices()
        if not isinstance(m_all, torch.Tensor):
            import warp as wp

            m_all = wp.to_torch(m_all)
        m = m_all[0]
        offset = m.shape[-1] - self.robot.num_joints
        diag = m.diagonal()[offset:]
        return diag[self._joint_ids].cpu()

    @staticmethod
    def _data_tensor(x) -> torch.Tensor:
        """6.0 data 컨테이너(warp 프론트엔드)의 torch 뷰를 얻는다 — torch 텐서면 그대로."""
        return x.torch if hasattr(x, "torch") else x

    def set_plant_params(
        self,
        mode: int,
        armature: list[float] | None = None,
        viscous: list[float] | None = None,
        coulomb: list[float] | None = None,
    ) -> None:
        """sim 플랜트 물성(armature/마찰)을 PACE 식별값으로 적용하거나 스톡 cfg로 복원한다.

        적용 경로는 PACE ``CMAESOptimizer.update_simulator``와 동일하다 — sim에 write하고
        ``data.default_*`` 캐시도 갱신해 이후 reset에서도 유지되게 한다. Isaac ≥5.0에서 Coulomb은
        계수가 아니라 effort이며(static=dynamic), 단일 호출로 static/dynamic/viscous를 함께 쓴다.
        PACE 33개 중 encoder bias(채점용 오프셋)와 delay(live DCMotor에 지연 버퍼 없음)는 적용
        대상이 아니다.

        Args:
            mode: 0이면 스톡 cfg 값 복원(나머지 인자 무시), 1이면 전달된 값 적용.
            armature: 관절별 armature [kg·m²], 길이 8, leg-major 순서.
            viscous: 관절별 점성 마찰 [N·m·s/rad], 길이 8, leg-major 순서.
            coulomb: 관절별 Coulomb 마찰 [N·m], 길이 8, leg-major 순서.
        """
        if mode == 0:
            arma_t = self._stock_armature.clone()
            visc_t = self._stock_viscous.clone()
            coul_t = self._stock_coulomb.clone()
        else:
            arma_t = self._to_tensor(armature)
            visc_t = self._to_tensor(viscous)
            coul_t = self._to_tensor(coulomb)
        env_ids = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        jid = self._plant_joint_ids_i32
        jid_long = jid.to(torch.long)
        self.robot.write_joint_armature_to_sim_index(armature=arma_t, joint_ids=jid, env_ids=env_ids)
        self._data_tensor(self.robot.data.default_joint_armature)[:, jid_long] = arma_t
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=coul_t,
            joint_dynamic_friction_coeff=coul_t,
            joint_viscous_friction_coeff=visc_t,
            joint_ids=jid,
            env_ids=env_ids,
        )
        self._data_tensor(self.robot.data.default_joint_friction_coeff)[:, jid_long] = coul_t
        self._data_tensor(self.robot.data.default_joint_viscous_friction_coeff)[:, jid_long] = visc_t

    def set_policy_target(self, target_q: torch.Tensor | list[float]) -> None:
        """policy_runner가 계산한 **articulation 순서** 목표각을 주입 (policy mode 전용).

        slew limiter를 우회한다 — 학습 파이프라인엔 slew가 없으므로, 넣으면 정책 동역학이 왜곡된다.
        target_q는 이미 ``action_scale * raw_action + default_joint_pos`` 로 계산된 절대 목표각이며
        articulation 순서(재매핑 없음)다.

        Args:
            target_q: 관절 목표각 [rad], 길이 = articulation 관절 수, articulation 순서.
        """
        t = torch.as_tensor(target_q, device=self.device, dtype=torch.float32)
        if t.dim() == 1:
            t = t.unsqueeze(0).expand(self.num_envs, -1)
        self._policy_target.copy_(t)

    def get_policy_state(self) -> dict:
        """policy_runner용 rich state (numpy, num_envs==1 가정).

        학습 정책 obs와 같은 **articulation 순서** 전체 관절 q/dq + base frame projected gravity.
        joint order 재매핑이 없도록 leg-major(_joint_ids)가 아니라 raw articulation 순서를 그대로 낸다.

        Returns:
            dict — ``q`` [rad], ``dq`` [rad/s] (각 길이 = articulation 관절 수), ``gravity`` (3, 단위벡터).
        """
        return {
            "q": self.robot.data.joint_pos[0].cpu().numpy(),
            "dq": self.robot.data.joint_vel[0].cpu().numpy(),
            "gravity": self.robot.data.projected_gravity_b[0].cpu().numpy(),
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
        if self.cfg.policy_mode:
            # policy 모드: 외부(policy_runner)가 set_policy_target으로 넣은 articulation 순서 절대
            # 목표각을 그대로 적용. slew/faithful PD 우회. joint_ids 생략 = 전체 관절 articulation 순서.
            del actions
            self.robot.set_joint_position_target(self._policy_target)
            return
        if self.cfg.sysid:
            # sysid 모드: actions = 절대 목표각 [rad], **articulation 관절 순서**.
            # PACE(fit_bipedleg.py / collect_chirp_sim_bipedleg.py)가 관절 인덱스로 직접 채워 넣는
            # 규약이며, slew limiter는 chirp 고주파를 왜곡하므로 우회한다.
            # 게인도 여기서 건드리지 않는다 — CMA-ES가 actuator.stiffness 텐서를 직접 쓴다.
            if self.cfg.foot_coupling:
                # 커플링 재생 (2026-08-13): actions의 foot 항은 관절각이 아니라 **raw 목표**
                # (실기 ACT와 동일 — 엔코더가 재는 것도 raw뿐이다). live 모드와 같은 목표 치환으로
                # foot PD를 raw 공간에서 계산하고 전치 토크를 calf에 싣는다. sysid는 decimation=1이라
                # 제어 스텝 = 물리 스텝 — 여기서 치환하면 매 물리 스텝 갱신과 동일하다.
                # ⚠근사: PaceDCMotor의 delay 버퍼가 (raw_t − q_calf) 합성 목표를 통째로 지연시키므로
                # calf 피드백 항도 delay만큼 늦는다(기계 전달은 즉시가 맞음). chirp ≤2.5 Hz에서
                # delay ~10 ms면 오차 미미.
                q = self.robot.data.joint_pos
                dq = self.robot.data.joint_vel
                q_c = q[:, self._calf_art_ids]
                dq_c = dq[:, self._calf_art_ids]
                raw_t = actions[:, self._foot_art_ids].clone()
                actions = actions.clone()
                actions[:, self._foot_art_ids] = raw_t - q_c
                self.robot.set_joint_velocity_target_index(target=-dq_c, joint_ids=self._foot_art_ids)
                kp_f, kd_f = self._actuator_gains_at(self._foot_art_ids)
                tau_pred = kp_f * ((raw_t - q_c) - q[:, self._foot_art_ids]) + kd_f * (
                    -dq_c - dq[:, self._foot_art_ids]
                )
                tau_pred = tau_pred.clamp(-self._foot_tau_max, self._foot_tau_max)
                self.robot.set_joint_effort_target_index(target=tau_pred, joint_ids=self._calf_art_ids)
            self.robot.set_joint_position_target_index(target=actions)
            return
        del actions  # live 모드: setpoint은 self._setpoint에서 직접 주입
        kp_cmd, kd_cmd = self._kp, self._kd
        if self.cfg.foot_coupling:
            # relax(무토크, foot kp≈0)에서도 실기는 기어 마찰이 raw(q_foot+q_calf)를 잠근다
            # (비가역 전달기구, 2026-08-12 실기 관찰). foot 게인을 hold 게인으로 치환하고,
            # hold 진입 순간의 raw를 래치한다. 해제 시 GUI 게인으로 복귀.
            hold_now = self._kp[:, self._couple_foot_lm] <= 1e-6
            newly_hold = hold_now & ~self._foot_hold
            if newly_hold.any():
                q = self.robot.data.joint_pos
                raw_now = q[:, self._foot_art_ids] + q[:, self._calf_art_ids]
                self._raw_latch = torch.where(newly_hold, raw_now, self._raw_latch)
            self._foot_hold = hold_now
            hold_kp = torch.full_like(self._foot_kp_eff, self.cfg.coupling_hold_kp)
            hold_kd = torch.full_like(self._foot_kd_eff, self.cfg.coupling_hold_kd)
            self._foot_kp_eff = torch.where(hold_now, hold_kp, self._kp[:, self._couple_foot_lm])
            self._foot_kd_eff = torch.where(hold_now, hold_kd, self._kd[:, self._couple_foot_lm])
            kp_cmd = self._kp.clone()
            kd_cmd = self._kd.clone()
            kp_cmd[:, self._couple_foot_lm] = self._foot_kp_eff
            kd_cmd[:, self._couple_foot_lm] = self._foot_kd_eff
        # faithful PD: kp/kd가 바뀌었으면 sim drive 게인 갱신.
        if self.cfg.faithful_pd:
            if not torch.equal(kp_cmd, self._applied_kp) or not torch.equal(kd_cmd, self._applied_kd):
                self._write_pd_gains(kp_cmd, kd_cmd)
                self._applied_kp.copy_(kp_cmd)
                self._applied_kd.copy_(kd_cmd)
        # Slew rate limiter: 한 step에서 max_step 이상 이동 불가
        delta = (self._setpoint - self._prev_setpoint).clamp(-self._max_step, self._max_step)
        smoothed = self._prev_setpoint + delta
        self._prev_setpoint = smoothed.clone()
        self._live_target.copy_(smoothed)
        self.robot.set_joint_position_target_index(target=smoothed, joint_ids=self._joint_ids)

    def _actuator_gains_at(self, art_ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        """지정 관절(articulation id)의 현재 액추에이터 kp/kd를 (num_envs, len) 텐서로 모은다.

        explicit actuator의 진짜 게인은 ``actuator.stiffness/damping`` 텐서다 — sysid 커플링
        재생의 전치 토크 예측에 쓴다 (PACE는 게인을 식별하지 않으므로 전 env 공통 값).
        """
        kp = torch.zeros(self.num_envs, len(art_ids), device=self.device)
        kd = torch.zeros_like(kp)
        for actuator in self.robot.actuators.values():
            ids = actuator.joint_indices
            if isinstance(ids, slice):
                ids = list(range(self.robot.num_joints))
            ids = [int(i) for i in ids]
            for k, a in enumerate(art_ids):
                if int(a) in ids:
                    j = ids.index(int(a))
                    kp[:, k] = actuator.stiffness[:, j]
                    kd[:, k] = actuator.damping[:, j]
        return kp, kd

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
        # live 모드 foot↔calf 커플링 — physics step(200Hz)마다 foot 목표를 raw 공간으로 재계산한다.
        # 나머지 모드/관절은 _pre_physics_step에서 처리 완료.
        if self.cfg.policy_mode or self.cfg.sysid or not self.cfg.foot_coupling:
            return
        q = self.robot.data.joint_pos
        dq = self.robot.data.joint_vel
        q_c = q[:, self._calf_art_ids]
        dq_c = dq[:, self._calf_art_ids]
        q_f = q[:, self._foot_art_ids]
        dq_f = dq[:, self._foot_art_ids]
        # foot 명령(CMD/GUI)은 raw 목표(실기 해석과 동일) — hold(relax) 중엔 진입 시 래치한 raw.
        raw_t = torch.where(self._foot_hold, self._raw_latch, self._live_target[:, self._couple_foot_lm])
        # raw 공간 PD를 관절 목표 치환으로 구현: pos_t = raw_t − q_calf, vel_t = −q̇_calf 를 주면
        # actuator PD가 kp·(raw_t − (q_f+q_c)) + kd·(−q̇_c − q̇_f) 를 계산한다 — 위치를 강제로 쓰는
        # kinematic 방식이 아니라 전달기구 강성으로 미는 방식이라 접촉/동역학이 깨지지 않는다.
        pos_t = raw_t - q_c
        vel_t = -dq_c
        self.robot.set_joint_position_target_index(target=pos_t, joint_ids=self._foot_art_ids)
        self.robot.set_joint_velocity_target_index(target=vel_t, joint_ids=self._foot_art_ids)
        # 전치 토크: 모터좌표 r = (q_c, q_f+q_c) ⇒ τ_joint = Tᵀ·τ_motor ⇒ τ_calf **+=** τ_foot_motor.
        # (RL_INTERFACE의 `τ_raw_src −= coef·τ_joint_dst`는 역방향 — 원하는 관절토크에서 모터 명령을
        # 구할 때의 식이다. sim은 모터토크→관절토크 방향이라 부호가 +다.)
        # foot 모터 토크 예측치를 calf feedforward로 넣는다. DCMotor의 토크-속도 클립과 정적 한계
        # 클램프가 완전히 같지는 않아 포화 구간에서만 근사 오차가 있다.
        tau_pred = self._foot_kp_eff * (pos_t - q_f) + self._foot_kd_eff * (vel_t - dq_f)
        tau_pred = tau_pred.clamp(-self._foot_tau_max, self._foot_tau_max)
        self.robot.set_joint_effort_target_index(target=tau_pred, joint_ids=self._calf_art_ids)

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
        # policy target도 default(articulation 순서 전체 관절)로 초기화 — 스트림 점프 방지.
        self._policy_target[env_ids] = self.robot.data.default_joint_pos[env_ids]
        # foot↔calf 커플링 버퍼 — raw 래치는 default 자세의 raw(q_f+q_c)로.
        self._live_target[env_ids] = default_pos[env_ids]
        self._foot_hold[env_ids] = False
        default_all = self.robot.data.default_joint_pos
        self._raw_latch[env_ids] = (
            default_all[env_ids][:, self._foot_art_ids] + default_all[env_ids][:, self._calf_art_ids]
        )
        self._foot_kp_eff[env_ids] = self._kp[env_ids][:, self._couple_foot_lm]
        self._foot_kd_eff[env_ids] = self._kd[env_ids][:, self._couple_foot_lm]

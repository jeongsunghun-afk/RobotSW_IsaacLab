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
    FOOT_LINK_INERTIA_KGM2,
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
        # foot 토크 한계 — raw 마찰 feedforward 의 안정 캡에 쓴다
        # (:meth:`_foot_raw_friction_torque`). 전치 토크는 액추에이터가 낸 값을 그대로 읽으므로
        # 여기서 게인이나 한계를 따로 캐시할 필요가 없다.
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

        # raw 좌표 foot 마찰 (cfg.foot_raw_friction) — foot 관절의 PhysX 마찰을 0으로 눌러 둘 때
        # 쓰는 인덱스/영텐서를 미리 만들어 둔다 (warp 커널은 int32 인덱스를 요구한다).
        # 직전 스텝에 foot 에 실은 반사관성 보정 — 전치가 `applied_torque` 에서 이 값을 빼야
        # 관성항이 벨트를 타고 calf 에 이중으로 들어가지 않는다 (_apply_foot_coupling 참고).
        self._mrefl_foot_prev = torch.zeros(self.num_envs, len(self._foot_art_ids), device=self.device)
        self._foot_art_ids_i32 = torch.as_tensor(self._foot_art_ids, dtype=torch.int32, device=self.device)
        self._all_env_ids_i32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
        self._foot_zero_fric = torch.zeros(self.num_envs, len(self._foot_art_ids), device=self.device)

        # 스톡 플랜트 물성 캡처 (set_plant_params mode=0 복원용) — cfg 적용 직후의 값 [leg-major].
        # 6.0 data의 default_* 컨테이너는 warp 프론트엔드일 수 있어 .torch 뷰로 접근한다.
        # ⚠ 이 읽기는 raw 좌표 foot 마찰에도 **load-bearing**이다: ``data.default_joint_*`` 는 최초
        #   접근 시점의 sim 값을 복제하는 lazy clone이라, _clear_foot_joint_friction()이 sim을 0으로
        #   만든 뒤에 처음 접근하면 0이 복제돼 b_raw/c_raw가 사라진다. 여기(어떤 zeroing보다 먼저)서
        #   읽어 캐시를 cfg 값으로 고정해 둔다 — 순서를 바꾸지 말 것.
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

        **전 관절 관절(모델) 좌표**로 보고한다 (:data:`CONVENTION_VERSION` = 1). foot도 예외가
        아니다 — 브리지(``real_runner``)가 raw↔관절 변환을 전담하게 되면서 실기 TELEM도 관절각을
        보내므로, sim이 foot만 raw로 보고하면 좌표가 어긋난다.

        ⚠ 2026-08-14 이전에는 ``foot_coupling=True`` 일 때 foot q/dq/ddq를 raw각
        ``q_foot + q_calf`` 로 합산해 보고했다(구 규약 0). 그 합산을 제거했다 — foot 전달기구
        커플링은 :meth:`_apply_foot_coupling` 안에서 raw를 합성해 재현할 뿐, **밖으로 나가는 값은
        전부 관절각**이다.

        Returns:
            dict — q, dq, ddq, tau_est (각 길이 8, JOINT 순서, 관절 좌표). IMU/base 상태는 미노출.
        """
        return {
            "q": self.robot.data.joint_pos[0, self._joint_ids].cpu().numpy(),
            "dq": self.robot.data.joint_vel[0, self._joint_ids].cpu().numpy(),
            "ddq": self.robot.data.joint_acc[0, self._joint_ids].cpu().numpy(),
            "tau_est": self.robot.data.applied_torque[0, self._joint_ids].cpu().numpy(),
        }

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
            # **관절** 목표각을 그대로 적용. slew/faithful PD 우회. joint_ids 생략 = 전체 관절 순서.
            del actions
            self.robot.set_joint_position_target(self._policy_target)
            # 2026-08-14: foot 커플링을 policy 모드에도 적용한다(_apply_action). 이전에는 통째로
            # 건너뛰어서 **sim 정책 테스트가 학습 env(`hind_leg_env._apply_action`)와 다르게**
            # 동작했다 — 학습은 커플링을 적용하는데 배포 리허설은 안 하는 상태였다.
            if self.cfg.foot_coupling and self.cfg.foot_raw_friction:
                # 관절 좌표 PhysX 마찰 제거는 제어 스텝마다 1회. 빠뜨리면 raw 마찰(feedforward)과
                # PhysX 관절 마찰이 **이중으로** 걸린다 (b_raw는 default_* 캐시에서 계속 읽히므로).
                self._clear_foot_joint_friction()
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
                # calf에 실을 모터축 토크. 전치(foot 모터 실토크)와 raw 마찰은 같은 일률 보존
                # 규칙에서 나오지만 A/B를 위해 플래그가 따로 있다 — 실기 구성은 둘 다 True다.
                # 반사관성 off-diagonal — live 경로와 **같은 플랜트**여야 한다. 여기만 빠지면
                # PACE 가 학습 env 와 다른 플랜트에 적합된다(조용한 divergence).
                corr_calf = corr_foot = None
                if self.cfg.foot_reflected_inertia:
                    acc = self.robot.data.joint_acc
                    i_off = self._data_tensor(self.robot.data.joint_armature)[:, self._foot_art_ids]
                    corr_calf = -i_off * acc[:, self._foot_art_ids]
                    corr_foot = -i_off * acc[:, self._calf_art_ids]
                    cap = getattr(self.cfg, "foot_reflected_inertia_cap", None)
                    if cap is not None:
                        # 학습 env(`hind_leg_env`)와 **같은 캡** — 안 맞추면 다른 플랜트가 된다.
                        corr_calf = corr_calf.clamp(-cap, cap)
                        corr_foot = corr_foot.clamp(-cap, cap)
                tau_fric = None
                if self.cfg.foot_raw_friction:
                    # 관절 좌표 PhysX 마찰을 끄고(매 제어 스텝 = 매 물리 스텝, decimation=1)
                    # raw 좌표 마찰을 foot 에 feedforward 로 넣는다.
                    self._clear_foot_joint_friction()
                    tau_fric = self._foot_raw_friction_torque()
                tau_foot_ff = tau_fric
                if corr_foot is not None:
                    tau_foot_ff = corr_foot if tau_foot_ff is None else tau_foot_ff + corr_foot
                if tau_foot_ff is not None:
                    self.robot.set_joint_effort_target_index(target=tau_foot_ff, joint_ids=self._foot_art_ids)
                tau_calf = None
                if self.cfg.foot_transpose:
                    # live 경로(:meth:`_apply_foot_coupling`)와 같은 규약 — foot 모터가 실제로 낸
                    # 직전 스텝 토크를 싣는다. 게인 재계산 + 정적 클램프는 고속에서 최대 3배 과대였다.
                    # ⚠ PaceDCMotor 의 `applied_effort` 는 지연 버퍼 **앞** 값이다(`compute` 가
                    #   super() 결과를 지연시킨 뒤 반환하므로). 옛 식도 지연 앞 예측치였으니 회귀는
                    #   아니지만, 기계 전달은 지연된 실토크를 따르는 것이 더 맞다 — chirp ≤2.5 Hz,
                    #   지연 ~10 ms 라 위상 오차는 미미하다.
                    # ⚠ **적합 데이터에서는 구·신이 항등이다**: `data/bipedleg_g2/` 실기 chirp 18개
                    #   전부 foot |q̇| ≤ 4.15 rad/s 로 교차점 4.92 아래다(교차점 초과 0.00%).
                    #   과거 PACE 적합값과 직접 비교 가능하다. 반면 **합성 chirp**
                    #   (`collect_chirp_sim_bipedleg.py`, 10 Hz 스윕)는 foot 이 속도 클립 24.70 까지
                    #   가서 5.42% 의 스텝이 달라진다 — 그걸로 적합/검증하면 안 된다.
                    #   근거: reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/README.md §11
                    # ⚠ 직전 스텝에 foot 에 넣은 반사관성 보정은 모터 토크가 아니므로 빼 준다
                    #   (안 빼면 off-diagonal 이 전치를 타고 calf 에 이중으로 들어간다).
                    tau_calf = self.robot.data.applied_torque[:, self._foot_art_ids] - self._mrefl_foot_prev
                elif tau_fric is not None:
                    tau_calf = tau_fric
                if corr_calf is not None:
                    tau_calf = corr_calf if tau_calf is None else tau_calf + corr_calf
                if corr_foot is not None:
                    self._mrefl_foot_prev.copy_(corr_foot)
                elif self.cfg.foot_transpose:
                    self._mrefl_foot_prev.zero_()
                if tau_calf is not None:
                    # calf 자신의 액추에이터가 자기 토크-속도 곡선으로 다시 자른다 — 여기서 foot
                    # 한계로 미리 자르지 않는다.
                    self.robot.set_joint_effort_target_index(target=tau_calf, joint_ids=self._calf_art_ids)
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
            if self.cfg.foot_raw_friction:
                # 관절 좌표 PhysX 마찰 제거는 제어 스텝(50 Hz)마다 1회면 충분하다 — 파라미터는
                # set_plant_params 호출 때만 바뀐다. 마찰 토크 자체는 _apply_action에서 물리
                # 스텝(200 Hz)마다 갱신한다.
                self._clear_foot_joint_friction()
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

    def _clear_foot_joint_friction(self) -> None:
        """foot 관절의 PhysX 마찰(static/dynamic/viscous)을 sim에서 0으로 지운다.

        PhysX 마찰은 관절 속도 ``q̇_f`` 에만 걸리는데 실제 마찰은 모터축 ``θ̇_f = q̇_f + q̇_c`` 에
        앉아 있다 (`cfg.foot_raw_friction` 주석). 그래서 관절 좌표 마찰은 끄고
        :meth:`_foot_raw_friction_torque` 가 raw 좌표에서 계산한 토크를 대신 넣는다.

        ⚠ ``data.default_joint_*_friction_coeff`` 캐시는 **건드리지 않는다** — 그 캐시가 b_raw/c_raw의
        저장소이기 때문이다. PACE ``CMAESOptimizer.update_simulator`` 와 :meth:`set_plant_params` 가
        후보값을 sim과 캐시 양쪽에 쓰므로, sim 쪽만 매 제어 스텝 0으로 눌러 두면 캐시에는 파라미터가
        남는다. (파라미터가 언제 바뀌었는지 알려면 GPU→CPU 동기화가 필요해 매번 쓰는 편이 싸다.)
        """
        self.robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self._foot_zero_fric,
            joint_dynamic_friction_coeff=self._foot_zero_fric,
            joint_viscous_friction_coeff=self._foot_zero_fric,
            joint_ids=self._foot_art_ids_i32,
            env_ids=self._all_env_ids_i32,
        )

    def _foot_raw_friction_torque(self) -> torch.Tensor:
        """foot 모터축(raw) 좌표의 감속기·벨트 마찰 토크 [N·m], shape (num_envs, 2), leg 순서.

        모터축 속도가 ``w_raw = q̇_foot + q̇_calf`` 이므로::

            τ_fric = −(b_raw·w_raw + c_raw·tanh(w_raw / eps))

        b_raw/c_raw 는 foot 관절의 viscous/Coulomb 슬롯을 재해석해 읽는다
        (`cfg.foot_raw_friction` 주석의 파라미터 재해석 항목).

        ``sign`` 대신 ``tanh(w_raw / eps)`` (eps = :attr:`R2SBipedLegEnvCfg.foot_raw_friction_vel_eps`)
        를 쓴다 — 0 근처에서 부호가 매 스텝 뒤집히는 채터링을 막는다.

        ⚠ **안정성 캡**: PhysX 관절 마찰은 구속 기반이라 무조건 안정하지만, 여기서는 마찰을
        명시적(explicit) feedforward 토크로 넣으므로 ``b_eff·dt / I > 2`` 면 발산한다. raw 축 관성은
        ``I ≈ armature_foot + FOOT_LINK_INERTIA_KGM2`` 인데 armature 가 PACE 식별 대상(하한 1e-5)이라
        I 가 0.0019 까지 내려갈 수 있고, 그러면 안정 한계가 ``2·0.0019/0.005 ≈ 0.76 N·m·s/rad`` 로
        viscous 상한(5.0)보다 훨씬 작아진다. 그래서 마찰이 한 스텝 안에 축 속도를 **역전시키지
        못하도록** ``|τ_fric| ≤ I·|w_raw| / dt`` 로 자른다. 이 캡이 없으면 CMA-ES 가 뽑은 큰 b/c
        후보가 크래시 없이 조용히 발산해 적합이 저-마찰 쪽으로 편향된다.

        ⚠ 위 안정 캡은 armature 가 크면 느슨해진다(실측: armature 0.4 후보에서 캡 ≈3600 N·m). 그래서
        모터 토크 한계 ``_foot_tau_max`` 로 **한 번 더** 자른다 — 실측 216 N·m 짜리 마찰
        feedforward 는 물리적으로 무의미하다.

        .. note::
            2026-08-18 이전에는 이 캡의 주된 근거가 "foot 은 DCMotor 총토크 클립, calf 는
            ``clamp(±_foot_tau_max)`` 로 서로 다르게 잘려 일률 보존이 깨진다" 였다. 이제 calf 는
            foot 의 ``applied_torque`` 를 **그대로** 받으므로 두 관절이 보는 전달 토크가 구조적으로
            같고(1 스텝 지연 제외), 그 근거는 해소됐다. 캡은 수치 안정 장치로만 남는다.
        """
        dq = self.robot.data.joint_vel
        w_raw = dq[:, self._foot_art_ids] + dq[:, self._calf_art_ids]
        b_raw = self._data_tensor(self.robot.data.default_joint_viscous_friction_coeff)[:, self._foot_art_ids]
        c_raw = self._data_tensor(self.robot.data.default_joint_friction_coeff)[:, self._foot_art_ids]
        eps = max(float(self.cfg.foot_raw_friction_vel_eps), 1e-6)
        mag = (b_raw * w_raw + c_raw * torch.tanh(w_raw / eps)).abs()
        i_raw = self._data_tensor(self.robot.data.default_joint_armature)[:, self._foot_art_ids]
        tau_cap = (i_raw + FOOT_LINK_INERTIA_KGM2) * w_raw.abs() / self.cfg.sim.dt
        tau_cap = torch.minimum(tau_cap, self._foot_tau_max)
        return -torch.sign(w_raw) * torch.minimum(mag, tau_cap)

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

    def _apply_foot_coupling(self, raw_t: torch.Tensor) -> None:
        """foot 목표를 raw 공간으로 치환하고 전치 토크 + raw 마찰을 싣는다 (live/policy 공통).

        raw 공간 PD를 **관절 목표 치환**으로 구현한다: ``pos_t = raw_t − q_calf``,
        ``vel_t = −q̇_calf`` 를 주면 actuator PD가 ``kp·(raw_t − (q_f+q_c)) + kd·(−q̇_c − q̇_f)``,
        즉 raw 공간 오차를 계산한다 — 위치를 강제로 쓰는 kinematic 방식이 아니라 전달기구 강성으로
        미는 방식이라 접촉/동역학이 깨지지 않는다.

        전치 토크는 foot 게인으로 재계산하지 않고 액추에이터가 실제로 낸 토크를 읽는다 — 아래
        ``applied_torque`` 주석 참고.

        Args:
            raw_t: foot 모터축(raw) 목표 ``q_foot_des + q_calf_des`` [rad], shape (num_envs, 2).
        """
        q = self.robot.data.joint_pos
        dq = self.robot.data.joint_vel
        q_c = q[:, self._calf_art_ids]
        dq_c = dq[:, self._calf_art_ids]
        pos_t = raw_t - q_c
        vel_t = -dq_c
        self.robot.set_joint_position_target_index(target=pos_t, joint_ids=self._foot_art_ids)
        self.robot.set_joint_velocity_target_index(target=vel_t, joint_ids=self._foot_art_ids)
        # calf에 실을 모터축 토크 = 전치(foot 모터 실토크) + raw 좌표 마찰. 둘 다 같은 일률 보존
        # 규칙(벨트가 무릎을 건넌다)에서 나오지만 A/B를 위해 플래그가 따로 있다 — 실기 구성은 둘 다 True.
        # ── 반사관성 off-diagonal (cfg.foot_reflected_inertia) ────────────────────────────
        # 대각은 armature 로 들어가 있고, PhysX 가 못 쓰는 off-diagonal 만 명시적 토크로 넣는다.
        # I_off = I_r·N_f² 는 **foot 대각 armature 와 같은 양**이라 그대로 읽는다(DR 자동 일관).
        corr_calf = corr_foot = None
        if self.cfg.foot_reflected_inertia:
            acc = self.robot.data.joint_acc
            i_off = self._data_tensor(self.robot.data.joint_armature)[:, self._foot_art_ids]
            corr_calf = -i_off * acc[:, self._foot_art_ids]
            corr_foot = -i_off * acc[:, self._calf_art_ids]
            cap = getattr(self.cfg, "foot_reflected_inertia_cap", None)
            if cap is not None:
                # 학습 env(`hind_leg_env`)와 **같은 캡** — 안 맞추면 다른 플랜트가 된다.
                corr_calf = corr_calf.clamp(-cap, cap)
                corr_foot = corr_foot.clamp(-cap, cap)
        tau_fric = None
        if self.cfg.foot_raw_friction:
            # 감속기·벨트 마찰은 모터축(raw)에 앉아 있다 — foot 자체 PD에 feedforward로 더한다
            # (PhysX 관절 마찰은 _pre_physics_step에서 제거).
            tau_fric = self._foot_raw_friction_torque()
        tau_foot_ff = tau_fric
        if corr_foot is not None:
            tau_foot_ff = corr_foot if tau_foot_ff is None else tau_foot_ff + corr_foot
        if tau_foot_ff is not None:
            self.robot.set_joint_effort_target_index(target=tau_foot_ff, joint_ids=self._foot_art_ids)
        if self.cfg.foot_transpose:
            # 전치 토크: 모터좌표 r = (q_c, q_f+q_c) ⇒ τ_joint = Tᵀ·τ_motor ⇒ τ_calf += τ_foot.
            # (RL_INTERFACE의 `τ_raw_src −= coef·τ_joint_dst`는 역방향 — 원하는 관절토크에서 모터
            # 명령을 구할 때의 식이라 부호가 반대다.)
            # ★ 2026-08-18: foot 모터가 **실제로 낸** 토크(직전 physics step의 `applied_torque`)를
            #   그대로 싣는다. 이전에는 kp·e + kd·ė 를 재계산하고 정적 `_foot_tau_max`(100.8 N·m)로
            #   클램프했는데, DCMotor는 4사분면 속도 의존 곡선으로 자르므로 옛 식은 물리적으로 낼 수
            #   없는 토크를 calf에 실을 수 있었다.
            #   `applied_torque`는 그 곡선으로 이미 잘려 있고 위의 마찰 feedforward도 포함하므로
            #   (`IdealPDActuator.compute`: kp·e + kd·ė + joint_efforts → _clip_effort),
            #   마찰을 다시 더하거나 다시 클램프하면 이중계상이다. 지연은 1 physics step.
            #   ⚠ 실측: 배포 게인(kp_foot=20)에서는 foot 이 교차점 q̇ ≈ 4.92 rad/s 에 도달조차 못 해
            #     두 식이 **같다**. 차이는 live hold(`coupling_hold_kp` 200)처럼 게인이 높은 경로에서만
            #     난다(스텝의 17.2%). 근거: reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe §11
            # ⚠ `applied_torque` 에는 직전 스텝에 foot 에 넣은 **반사관성 보정도 섞여 있다**. 그건
            #   모터 토크가 아니라 관성항이라 벨트로 전달되면 안 된다(그대로 두면 off-diagonal 이
            #   전치를 타고 calf 에 한 번 더 들어간다). 같은 1스텝 지연으로 캐시해 뺀다.
            tau_calf = self.robot.data.applied_torque[:, self._foot_art_ids] - self._mrefl_foot_prev
        elif tau_fric is not None:
            # 전치를 끈 A/B: 마찰만 calf에 같은 부호로 싣는다.
            tau_calf = tau_fric
        else:
            tau_calf = None
        if corr_calf is not None:
            tau_calf = corr_calf if tau_calf is None else tau_calf + corr_calf
        # 다음 스텝의 전치 보정용으로 이번에 foot 에 실은 반사관성 항을 기억한다.
        if corr_foot is not None:
            self._mrefl_foot_prev.copy_(corr_foot)
        elif self.cfg.foot_transpose:
            self._mrefl_foot_prev.zero_()
        if tau_calf is None:
            return
        # calf 자신의 액추에이터가 이 feedforward를 포함해 자기 토크-속도 곡선으로 다시 자른다 —
        # 여기서 foot 한계로 미리 자르면 한계를 이중으로 거는 셈이라 클램프하지 않는다.
        self.robot.set_joint_effort_target_index(target=tau_calf, joint_ids=self._calf_art_ids)

    def _apply_action(self) -> None:
        # live/policy 모드 foot↔calf 커플링 — physics step(200Hz)마다 foot 목표를 raw 공간으로
        # 재계산한다. sysid는 _pre_physics_step에서 자체 처리(raw 규약 유지), 나머지 관절도 거기서 완료.
        if self.cfg.sysid or not self.cfg.foot_coupling:
            return
        if self.cfg.policy_mode:
            # policy 목표는 **관절 목표**(학습 정책이 관절 공간으로 학습됐다) — raw는 목표값끼리
            # 합성한다. 학습 env `hind_leg_env._apply_action`의
            #   raw_t = processed_actions[foot] + processed_actions[calf]
            # 와 **같은 식**이라야 배포 리허설이 학습과 일치한다.
            raw_t = self._policy_target[:, self._foot_art_ids] + self._policy_target[:, self._calf_art_ids]
            self._apply_foot_coupling(raw_t)
            return
        # live 모드 — CMD/GUI의 foot 목표는 **관절 목표**다 (CONVENTION_VERSION 1).
        # 2026-08-14 이전에는 raw 목표로 해석했다(구 규약 0). 이제 raw는 여기서 **목표값끼리**
        # 합성한다(측정 calf가 아니라 calf 목표를 쓴다 — 학습 env와 동일).
        # ⚠ hold(relax)의 `_raw_latch`는 여전히 **raw 공간** 래치다: 기어 마찰이 잠그는 것은
        #   모터축(raw)이지 관절각이 아니므로 그게 물리적으로 맞다. 규약 변경과 무관하다.
        joint_raw_t = self._live_target[:, self._couple_foot_lm] + self._live_target[:, self._couple_calf_lm]
        raw_t = torch.where(self._foot_hold, self._raw_latch, joint_raw_t)
        self._apply_foot_coupling(raw_t)

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
        # 리셋 직후에 직전 스텝의 반사관성 보정이 남아 있으면 전치가 그만큼 잘못 뺀다.
        self._mrefl_foot_prev[env_ids] = 0.0

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import gymnasium as gym
import torch
import warp as wp

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster

from .hind_leg_env_cfg import FOOT_LINK_INERTIA_KGM2, HindLegFlatEnvCfg, HindLegRoughEnvCfg


def torch_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class HindLegEnv(DirectRLEnv):
    cfg: HindLegFlatEnvCfg | HindLegRoughEnvCfg

    def __init__(self, cfg: HindLegFlatEnvCfg | HindLegRoughEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._previous_actions = torch.zeros(
            self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        self.command_curriculum = self.cfg.command_curriculum
        self.curriculum_rew_buf = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float)
        if self.command_curriculum:
            self.curriculum_lin_vel_x = torch.full((self.num_envs,), 0.0, device=self.device)
            self.curriculum_ang_vel = torch.full((self.num_envs,), 0.0, device=self.device)
            self.curriculum_step = self.cfg.curriculum_step
            self.curriculum_threshold = self.cfg.curriculum_threshold
            self.max_lin_vel_x = self.cfg.command_cfg["lin_vel_x_range"][1]
            self.max_ang_vel = self.cfg.command_cfg["ang_vel_range"][1]

        if self.cfg.history_observation:
            self.obs_history_buf = torch.zeros(
                self.num_envs, self.cfg.history_len, self.cfg.num_prio_obs, device=self.device, dtype=torch.float
            )

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "similar_to_default",
                "base_height",
                "termination",
                "gait_stance",
                "gait_swing",
                "foot_slip",
            ]
        }
        # Get specific body indices
        # URDF3부터 base 링크가 base_collision으로 리네임됨 (find_bodies는 fullmatch) — 구·신 자산 겸용 패턴.
        self._base_id, _ = self._contact_sensor.find_bodies("base.*")
        # Contact sensor body ids for swing/stance gating (.*foot_contact.*)
        self._feet_ids, feet_contact_names = self._contact_sensor.find_bodies(".*foot_contact.*")
        # Robot articulation body ids for sole position and velocity (.*foot_contact.*)
        # These are in a different index space than _feet_ids (sensor vs. robot body indices).
        self._sole_body_ids, sole_body_names = self._robot.find_bodies(".*foot_contact.*")
        assert len(self._sole_body_ids) == 2, (
            f"Expected exactly 2 sole bodies matching '.*foot_contact.*', "
            f"got {len(self._sole_body_ids)}: {sole_body_names}"
        )
        # Verify foot ordering is consistent between contact sensor and robot body indices.
        # A mismatch would silently swap swing/stance gating between feet, corrupting training.
        assert [n.split("/")[-1] for n in feet_contact_names] == [n.split("/")[-1] for n in sole_body_names], (
            f"Foot name order mismatch — contact sensor: {feet_contact_names}, sole bodies: {sole_body_names}"
        )
        # 접지 기준선 [m] — swing clearance 보상의 원점. 평지 상수이자 env 공통이라
        # _reset_idx 에서 초기화하지 않는다(전역 상수, CLAUDE.md 버퍼 규칙의 예외).
        #
        # ⚠ 2026-08-13: 종전 구현은 첫 _get_rewards 호출에서 전 env 평균 sole z 를 그대로 캡처했다.
        # 그 시점은 리셋 직후로 로봇이 스폰 높이(base z=0.6, 관절 전부 0)에서 아직 공중에 있어,
        # 기준선이 109.0/105.5 mm 로 잡혔다(실측 접지 높이는 34.0/33.8 mm). clearance 는
        # (sole_z − rest)/gait_swing_height 이므로 보상을 받으려면 실제로 145 mm 를 들어야 했고,
        # gait_swing 은 36k iter 내내 0 에 묶여 정책이 발을 드는 유인을 전혀 받지 못했다.
        # 이제는 **접지 중인 발의 sole z 만 표본**해 추정한다 — 공중 자세가 섞이지 않는다.
        self._sole_rest_z: torch.Tensor | None = None
        self._sole_rest_sum = torch.zeros(len(self._sole_body_ids), device=self.device)
        self._sole_rest_cnt = torch.zeros(len(self._sole_body_ids), device=self.device)

        # Gait phase clock ∈ [0, 1) — one scalar per env, advances each step by step_dt / gait_period.
        # Initialized to zero here; _reset_idx randomizes it per-episode for decorrelation.
        self._gait_phase = torch.zeros(self.num_envs, device=self.device)

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalzied_body_names)

        # 학습 신호 클리핑 — 희귀 물리 폭주 이벤트의 극단 obs/reward가 critic 입력·value target을
        # 오염시키면 GAE bootstrap을 타고 지수 발산한다(2026-08-12 signfix_coupled v1/v2 파국:
        # advantage는 정규화돼 actor·reward 지표는 정상인 채 value loss만 ×300/iter 폭주 → NaN).
        # 정상 신호 범위(|joint_vel|≲30, |reward|≲1/step) 밖에서만 작동한다.
        self._obs_clip = float(getattr(self.cfg, "obs_clip", 100.0))
        self._reward_clip = float(getattr(self.cfg, "reward_clip", 10.0))

        # foot↔calf 전달기구 커플링 (r2s_biped_leg live 모드와 동일 모델, RL_INTERFACE coef=+1).
        # 실기 foot 모터는 관절각이 아니라 raw각(q_foot + q_calf)을 구동한다 — 정책의 관절 목표를
        # raw 목표(q_f_des + q_c_des)로 합성해 raw 공간 PD로 추종하고 전치 토크를 calf에 더한다.
        self._foot_coupling = bool(getattr(self.cfg, "foot_coupling", False))
        # 전치 토크 / raw 좌표 마찰 — r2s_biped_leg(sysid 플랜트)와 같은 스위치. 커플링이 꺼져 있으면
        # 둘 다 의미가 없으므로 False로 접는다(속성은 항상 존재해야 _pre_physics_step이 안전하다).
        self._foot_transpose = self._foot_coupling and bool(getattr(self.cfg, "foot_transpose", True))
        self._foot_raw_friction = self._foot_coupling and bool(getattr(self.cfg, "foot_raw_friction", True))
        # 반사관성 off-diagonal — 벨트가 만드는 항이므로 커플링이 꺼지면 같이 꺼진다.
        self._foot_reflected_inertia = self._foot_coupling and bool(getattr(self.cfg, "foot_reflected_inertia", True))
        # 반사관성 off-diagonal 의 안정성 캡 (cfg 주석에 근거). None = 무제한.
        cap = getattr(self.cfg, "foot_reflected_inertia_cap", None)
        self._refl_cap = float(cap) if cap is not None else None
        # 넘어짐 종료 임계 (:meth:`_get_dones` 참고). cos 는 한 번만 계산한다.
        tilt = getattr(self.cfg, "terminate_tilt_deg", None)
        self._tilt_cos_limit = math.cos(math.radians(tilt)) if tilt is not None else 1.0
        if self._foot_coupling:
            self._calf_ids, _ = self._robot.find_joints(["HL_calf_joint", "HR_calf_joint"], preserve_order=True)
            self._foot_ids, _ = self._robot.find_joints(["HL_foot_joint", "HR_foot_joint"], preserve_order=True)
            # foot 게인/토크한계는 DR(randomize_actuator_gains)이 스케일한 현재 값을 읽어야 하므로
            # 상수 캐시 대신 actuator 텐서의 열 인덱스만 미리 계산해 둔다.
            actuator = self._robot.actuators["legs"]
            act_ids = actuator.joint_indices
            if isinstance(act_ids, slice):
                act_ids = list(range(self._robot.num_joints))
            act_ids = [int(i) for i in act_ids]
            self._foot_act_cols = [act_ids.index(int(j)) for j in self._foot_ids]
            self._foot_fric_eps = max(float(getattr(self.cfg, "foot_raw_friction_vel_eps", 0.2)), 1e-6)
            # foot 관절의 PhysX 마찰을 0으로 눌러 둘 때 쓰는 인덱스/영텐서 (warp 커널은 int32 요구).
            self._foot_ids_i32 = torch.as_tensor(self._foot_ids, dtype=torch.int32, device=self.device)
            self._all_env_ids_i32 = torch.arange(self.num_envs, dtype=torch.int32, device=self.device)
            self._foot_zero_fric = torch.zeros(self.num_envs, len(self._foot_ids), device=self.device)
            # 직전 스텝에 foot 에 실은 반사관성 보정 — 전치가 `applied_torque` 에서 이 값을 빼야
            # 관성항이 벨트를 타고 calf 에 이중으로 들어가지 않는다 (_apply_action 참고).
            self._mrefl_foot_prev = torch.zeros(self.num_envs, len(self._foot_ids), device=self.device)
            # ⚠ ``data.default_joint_*`` 는 **최초 접근 시점의 sim 값을 복제**하는 lazy clone이다.
            # _clear_foot_joint_friction()이 foot 관절 마찰을 0으로 만든 뒤에 처음 접근하면 0이 복제돼
            # b_raw/c_raw가 통째로 사라진다(실측: 스모크에서 b_raw=c_raw=0). 여기서 먼저 한 번 읽어
            # HIND_LEG_CFG 값(friction 0.38 / viscous 0.09)으로 캐시를 고정한다.
            self._data_tensor(self._robot.data.default_joint_viscous_friction_coeff)
            self._data_tensor(self._robot.data.default_joint_friction_coeff)
            self._data_tensor(self._robot.data.default_joint_armature)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # 새 UrdfConverter 자산(URDF3 계열)은 링크 prim이 운동학 트리 그대로 중첩된다. 코어
        # activate_contact_sensors는 첫 rigid body(base_collision)에서 하강을 멈추므로 자식 링크에
        # PhysxContactReportAPI가 붙지 않는다 — 서브트리 전체를 걸어 남은 링크에도 붙인다(클론 전 1회).
        self._activate_nested_contact_report("/World/envs/env_0/Robot")
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, HindLegRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        # Env isolation: filter cross-env collisions unconditionally (GPU too). In IsaacLab 3.0
        # the auto-filter path (interactive_scene:218) is skipped when the scene cfg declares no
        # entities (has_scene_cfg_entities=False), so the old cpu-only guard left GPU runs
        # unfiltered — robots from different envs physically collide. Ref: IsaacLab #1918.
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _activate_nested_contact_report(root_path: str):
        """루트 아래 모든 rigid body에 PhysxContactReportAPI를 붙인다 (중첩 링크 포함)."""
        from pxr import UsdPhysics

        from isaaclab.sim.schemas.schemas import activate_contact_sensors
        from isaaclab.sim.utils.stage import get_current_stage

        stage = get_current_stage()
        frontier = [stage.GetPrimAtPath(root_path)]
        while frontier:
            prim = frontier.pop(0)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                activate_contact_sensors(prim.GetPath().pathString, stage=stage)
            frontier += prim.GetChildren()

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        # Advance gait phase by one env-step before reward/obs read this step's value.
        self._gait_phase = (self._gait_phase + self.step_dt / self.cfg.gait_period) % 1.0
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        if self._foot_coupling and self._foot_raw_friction:
            # 관절 좌표 PhysX 마찰 제거는 제어 스텝(50 Hz)마다 1회면 충분하다 — 마찰 계수는 그보다
            # 자주 바뀌지 않는다. 마찰 토크 자체는 _apply_action에서 물리 스텝마다 갱신한다.
            self._clear_foot_joint_friction()

    @staticmethod
    def _data_tensor(x) -> torch.Tensor:
        """6.0 data 컨테이너(warp 프론트엔드)의 torch 뷰를 얻는다 — torch 텐서면 그대로."""
        return x.torch if hasattr(x, "torch") else x

    def _clear_foot_joint_friction(self) -> None:
        """foot 관절의 PhysX 마찰(static/dynamic/viscous)을 sim에서 0으로 지운다.

        PhysX 마찰은 관절 속도 ``q̇_f`` 에만 걸리는데 실제 감속기·벨트 마찰은 모터축
        ``θ̇_f = q̇_f + q̇_c`` 에 앉아 있다. 관절 좌표 마찰은 끄고
        :meth:`_foot_raw_friction_torque` 가 raw 좌표에서 계산한 토크를 대신 넣는다.
        ``data.default_joint_*_friction_coeff`` 캐시는 건드리지 않는다 — 그 캐시가 b_raw/c_raw의
        저장소다 (`HIND_LEG_CFG`의 friction 0.38 / viscous 0.09, 또는 PACE 식별값을 이식한 값).
        """
        self._robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self._foot_zero_fric,
            joint_dynamic_friction_coeff=self._foot_zero_fric,
            joint_viscous_friction_coeff=self._foot_zero_fric,
            joint_ids=self._foot_ids_i32,
            env_ids=self._all_env_ids_i32,
        )

    def _foot_raw_friction_torque(self) -> torch.Tensor:
        """foot 모터축(raw) 좌표의 감속기·벨트 마찰 토크 [N·m], shape (num_envs, 2), leg 순서.

        모터축 속도가 ``w_raw = q̇_foot + q̇_calf`` 이므로
        ``τ_fric = −(b_raw·w_raw + c_raw·tanh(w_raw / eps))``. b_raw/c_raw는 foot 관절의
        viscous/Coulomb 슬롯을 raw 좌표 계수로 재해석해 읽는다. ``sign`` 대신 ``tanh(w/eps)``를
        써서 0 근처 채터링을 막는다 (eps = ``cfg.foot_raw_friction_vel_eps``).

        ⚠ **안정성 캡**: PhysX 관절 마찰은 구속 기반이라 무조건 안정하지만 여기서는 명시적
        feedforward 토크라 ``b_eff·dt / I > 2`` 면 발산한다. raw 축 관성
        ``I ≈ armature_foot + FOOT_LINK_INERTIA_KGM2`` 가 작을 수 있으므로, 마찰이 한 스텝 안에
        축 속도를 역전시키지 못하도록 ``|τ_fric| ≤ I·|w_raw| / dt`` 로 자른다. armature 가 크면 이
        캡이 느슨해지므로 모터 토크 한계로 **한 번 더** 자른다 — 수백 N·m 짜리 마찰 feedforward 는
        물리적으로 무의미하기 때문이다.

        .. note::
            2026-08-18 이전에는 이 캡의 근거가 "foot 은 DCMotor 총토크 클립, calf 는 clamp(±tau_max)
            로 서로 다르게 잘려 일률 보존이 깨진다" 였다. 이제 calf 는 foot 의 ``applied_torque`` 를
            그대로 받으므로 두 관절의 전달 토크가 구조적으로 같아 그 근거는 해소됐다.
        """
        dq = self._robot.data.joint_vel
        w_raw = dq[:, self._foot_ids] + dq[:, self._calf_ids]
        b_raw = self._data_tensor(self._robot.data.default_joint_viscous_friction_coeff)[:, self._foot_ids]
        c_raw = self._data_tensor(self._robot.data.default_joint_friction_coeff)[:, self._foot_ids]
        mag = (b_raw * w_raw + c_raw * torch.tanh(w_raw / self._foot_fric_eps)).abs()
        i_raw = self._data_tensor(self._robot.data.default_joint_armature)[:, self._foot_ids]
        tau_cap = (i_raw + FOOT_LINK_INERTIA_KGM2) * w_raw.abs() / self.cfg.sim.dt
        tau_cap = torch.minimum(tau_cap, self._robot.actuators["legs"].effort_limit[:, self._foot_act_cols])
        return -torch.sign(w_raw) * torch.minimum(mag, tau_cap)

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)
        if not self._foot_coupling:
            return
        # foot↔calf 커플링 — physics step(200Hz)마다 foot 목표를 raw 공간으로 재계산한다.
        # pos_t = raw_t − q_calf, vel_t = −q̇_calf 를 주면 actuator PD가
        # kp·(raw_t − (q_f+q_c)) + kd·(−q̇_c − q̇_f), 즉 raw 공간 오차를 계산한다 — 위치 강제 쓰기가
        # 아니라 전달기구 강성으로 미는 방식이라 접촉/동역학이 깨지지 않는다 (r2s_biped_leg 검증 완료).
        q = self._robot.data.joint_pos
        dq = self._robot.data.joint_vel
        q_c = q[:, self._calf_ids]
        dq_c = dq[:, self._calf_ids]
        # 정책 foot 목표(관절각)를 raw 목표로 합성 — 배포 시 real_runner가 같은 식으로 모터 목표를 만든다.
        raw_t = self._processed_actions[:, self._foot_ids] + self._processed_actions[:, self._calf_ids]
        pos_t = raw_t - q_c
        vel_t = -dq_c
        self._robot.set_joint_position_target_index(target=pos_t, joint_ids=self._foot_ids)
        self._robot.set_joint_velocity_target_index(target=vel_t, joint_ids=self._foot_ids)
        # calf에 실을 모터축 토크 = 전치(foot 모터 실토크) + raw 좌표 마찰. 둘 다 "벨트가 무릎을
        # 건넌다"는 같은 기구 구속에서 나오지만, A/B와 되돌림을 위해 플래그가 따로 있다
        # (실기 구성은 둘 다 True).
        # ── 반사관성 off-diagonal (cfg.foot_reflected_inertia) ────────────────────────────
        # M_refl 의 대각은 armature 로 이미 들어가 있고, PhysX 가 못 쓰는 off-diagonal 만 여기서
        # 명시적 토크로 넣는다: τ_calf += −I_off·q̈_foot, τ_foot += −I_off·q̈_calf.
        # I_off = I_r·N_f² 는 **foot 대각 armature 와 같은 양**이라 그대로 읽는다 — DR 이 걸려도
        # 자동으로 같은 배수를 받는다(별도 상수로 두면 DR 에서 조용히 어긋난다).
        corr_calf = corr_foot = None
        if self._foot_reflected_inertia:
            acc = self._robot.data.joint_acc
            i_off = self._data_tensor(self._robot.data.joint_armature)[:, self._foot_ids]
            corr_calf = -i_off * acc[:, self._foot_ids]
            corr_foot = -i_off * acc[:, self._calf_ids]
            if self._refl_cap is not None:
                # 명시적 1스텝 지연 항이라 발산을 스스로 못 막는다 — 상수로 자른다.
                # ⚠ 마찰의 속도의존 캡을 쓰면 안 된다(적합 구간 4.29 % 발동). cfg 주석 참고.
                corr_calf = corr_calf.clamp(-self._refl_cap, self._refl_cap)
                corr_foot = corr_foot.clamp(-self._refl_cap, self._refl_cap)
        tau_fric = None
        if self._foot_raw_friction:
            # 감속기·벨트 마찰은 모터축(raw)에 앉아 있다 — foot PD에 feedforward로 더한다
            # (PhysX 관절 마찰은 _pre_physics_step에서 제거).
            tau_fric = self._foot_raw_friction_torque()
        tau_foot_ff = tau_fric
        if corr_foot is not None:
            tau_foot_ff = corr_foot if tau_foot_ff is None else tau_foot_ff + corr_foot
        if tau_foot_ff is not None:
            self._robot.set_joint_effort_target_index(target=tau_foot_ff, joint_ids=self._foot_ids)
        if self._foot_transpose:
            # 전치 토크: 모터좌표 r=(q_c, q_f+q_c) ⇒ τ_joint = Tᵀ·τ_motor ⇒ τ_calf += τ_foot_motor.
            # ★ 2026-08-18: foot 모터가 **실제로 낸** 토크를 쓴다 — 직전 physics step의
            #   `applied_torque[foot]`. 이전에는 kp·e + kd·ė 를 직접 계산하고 정적 effort_limit
            #   (100.8 N·m)로 클램프했는데, DCMotor는 4사분면 속도 의존 곡선으로 자르므로 옛 식은
            #   물리적으로 낼 수 없는 토크를 calf에 실을 수 있었다.
            #   `applied_torque`는 (a) 그 곡선으로 이미 잘려 있고 (b) 위에서 넣은 마찰 feedforward를
            #   포함하므로(`IdealPDActuator.compute`: kp·e + kd·ė + joint_efforts → _clip_effort),
            #   마찰을 다시 더하거나 다시 클램프하면 이중계상이다.
            #   지연은 1 physics step(200 Hz = 5 ms) — actuator 모델이 `_apply_action` 뒤의
            #   `write_data_to_sim`에서 돌기 때문이다.
            #   ⚠ 학습 env 실측(표본 194.6만, 완주 정책 resume): 접촉이 붙으면 foot q̇ 가 24.70
            #     rad/s 까지 가고 19.67% 가 교차점 4.92 를 넘는다. 구식이 곡선을 실제로 벗어난 건
            #     **0.012%** 로 드물지만 그때 최대 **101 N·m** 를 calf 에 실었다 — 평균은 같아도
            #     희귀 폭주 이벤트가 사라지는 것이 이 변경의 실질이다.
            #     근거: reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/README.md §11
            # ⚠ `applied_torque` 에는 직전 스텝에 우리가 foot 에 넣은 **반사관성 보정도 섞여 있다**.
            #   그건 모터 토크가 아니라 관성항이라 벨트로 전달되면 안 된다(넣으면 off-diagonal 이
            #   전치를 타고 calf 에 한 번 더 들어가 이중계상). 같은 1스텝 지연으로 캐시해 뺀다.
            tau_calf = self._robot.data.applied_torque[:, self._foot_ids] - self._mrefl_foot_prev
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
        elif self._foot_transpose:
            self._mrefl_foot_prev.zero_()
        if tau_calf is None:
            return
        # calf 자신의 DCMotor가 이 feedforward를 포함해 자기 토크-속도 곡선으로 다시 자른다 —
        # 여기서 foot 한계로 미리 자르면 calf 모터 한계를 이중으로 거는 셈이라 클램프하지 않는다.
        self._robot.set_joint_effort_target_index(target=tau_calf, joint_ids=self._calf_ids)

    def _get_observations(self) -> dict:
        # print(self._robot.joint_names)
        self._previous_actions = self._actions.clone()

        # Phase clock obs (4-dim) — HL at phase φ, HR at anti-phase φ+0.5.
        # _sole_body_ids order is [HL, HR] (asserted in __init__); index 1 (HR) gets 0.5 offset.
        phi_hl = self._gait_phase  # (N,)
        phi_hr = (self._gait_phase + 0.5) % 1.0  # (N,) — anti-phase
        clock_obs = torch.stack(
            [
                torch.sin(2.0 * torch.pi * phi_hl),
                torch.cos(2.0 * torch.pi * phi_hl),
                torch.sin(2.0 * torch.pi * phi_hr),
                torch.cos(2.0 * torch.pi * phi_hr),
            ],
            dim=1,
        )  # (N, 4)

        if self.cfg.history_observation:
            obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        # self._robot.data.root_lin_vel_b,
                        # self._robot.data.root_ang_vel_b,
                        self._robot.data.projected_gravity_b,
                        self._commands,
                        self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                        self._robot.data.joint_vel,
                        # height_data,
                        self._actions,
                        clock_obs if self.cfg.clock_inputs else None,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        else:
            obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        self._robot.data.root_lin_vel_b,
                        self._robot.data.root_ang_vel_b,
                        self._robot.data.projected_gravity_b,
                        self._commands,
                        self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                        self._robot.data.joint_vel,
                        # height_data,
                        self._actions,
                        clock_obs if self.cfg.clock_inputs else None,
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
        obs = obs.clamp(-self._obs_clip, self._obs_clip)
        observations = {"policy": obs}

        height_data = None
        if isinstance(self.cfg, HindLegRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
            ).clip(-1.0, 1.0)

        if height_data is not None:
            observations["scan"] = height_data

        if self.cfg.history_observation:
            self.obs_history_buf = torch.where(
                (self.episode_length_buf <= 1)[:, None, None],
                torch.stack([obs] * self.cfg.history_len, dim=1),
                torch.cat([self.obs_history_buf[:, 1:], obs.unsqueeze(1)], dim=1),
            )
            observations["history"] = self.obs_history_buf

        if self.cfg.priv_explicit:
            priv_explicit = torch.cat(
                [
                    self._robot.data.root_lin_vel_b * 2.0,  # 3
                    self._robot.data.root_ang_vel_b * 0.25,  # 3
                ],
                dim=-1,
            )
            priv_explicit = priv_explicit.clamp(-self._obs_clip, self._obs_clip)
            observations["priv_explicit"] = priv_explicit
        if self.cfg.priv_latent:
            priv_obs = torch.cat(
                [
                    tensor
                    for tensor in (
                        torch.tensor(self._robot.root_physx_view.get_masses(), device=self.device),
                        torch.tensor(
                            self._robot.root_physx_view.get_material_properties(),
                            device=self.device,
                        ).reshape(self.num_envs, -1),
                    )
                    if tensor is not None
                ],
                dim=-1,
            )
            observations["priv_latent"] = priv_obs

        return observations

    def _get_rewards(self) -> torch.Tensor:
        # linear velocity tracking
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.1)
        # yaw rate tracking
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.1)
        # z velocity tracking
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        # angular velocity x/y
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        air_time = torch.sum((last_air_time - 0.2) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)
        # flat orientation
        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        # Similar to default
        similar_to_default = torch.sum(
            torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos), dim=1
        )

        # base height
        base_height = torch.square(self._robot.data.root_link_pos_w[:, 2] - self._robot.data.default_root_state[:, 2])

        # termination
        termination = torch.any(
            torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1
        ).float()

        # Phase clock — HL at gait_phase, HR at anti-phase (offset 0.5).
        # _sole_body_ids order is [HL, HR] (asserted in __init__).
        phi_hl = self._gait_phase  # (N,)
        phi_hr = (self._gait_phase + 0.5) % 1.0  # (N,)
        theta = 2.0 * torch.pi * torch.stack([phi_hl, phi_hr], dim=1)  # (N, 2)

        # Smooth swing/stance coefficients via tanh (50/50 duty, gait_phase_sharpness controls ramp steepness).
        # E_swing ≈ 1 during swing half-cycle, ≈ 0 during stance; E_stance is complement.
        E_swing = 0.5 * (1.0 + torch.tanh(self.cfg.gait_phase_sharpness * torch.sin(theta)))  # (N, 2)
        E_stance = 1.0 - E_swing  # (N, 2)

        # Command-gated standing override: when command is near-zero, zero out E_swing so the robot
        # stands still instead of marching in place. E_stance remains untouched (two feet in contact
        # scores ~2.0 via gait_stance, strictly beating any march pattern). yaw is included in the
        # gate so that a pure-yaw command still allows swing. Phase clock is NOT touched — obs/checkpoint
        # compatibility is preserved.
        cmd_xy = torch.norm(self._commands[:, :2], dim=1)  # (N,)
        standing = (cmd_xy < self.cfg.standing_vel_threshold) & (
            self._commands[:, 2].abs() < self.cfg.standing_yaw_threshold
        )  # (N,) bool
        E_swing = torch.where(standing.unsqueeze(1), torch.zeros_like(E_swing), E_swing)
        E_stance = 1.0 - E_swing

        # Contact gate: True = foot in contact (force > 1N).
        # net_contact_forces shape: (N, history, n_contact_bodies); _feet_ids selects sole contact bodies.
        contact_filt = (
            torch.max(torch.norm(net_contact_forces[:, :, self._feet_ids], dim=-1), dim=1)[0] > 1.0
        )  # (N, n_feet)
        sole_z = self._robot.data.body_pos_w[:, self._sole_body_ids, 2]  # (N, n_feet)

        # 접지 기준선 추정 — **접지 중인 발만** 표본해 발별 평균을 낸다. 표본이 모일 때까지는
        # clearance 를 0 으로 두는데, 리셋 직후 낙하-착지에 수십 step 이면 충분히 채워진다.
        if self._sole_rest_z is None:
            mask = contact_filt.float()
            self._sole_rest_sum += (sole_z * mask).sum(dim=0)
            self._sole_rest_cnt += mask.sum(dim=0)
            if bool((self._sole_rest_cnt >= self.cfg.sole_rest_min_samples).all()):
                rest_z = (self._sole_rest_sum / self._sole_rest_cnt).detach()
                self._sole_rest_z = rest_z
                print(f"[INFO] sole rest-z 확정 [mm]: {[round(v * 1000, 1) for v in rest_z.tolist()]}")
        # Sole world-z minus rest-z = lift above ground (flat-ground baseline).
        lift = sole_z - self._sole_rest_z if self._sole_rest_z is not None else torch.zeros_like(sole_z)

        # (A) Phase-scheduled stance reward — bonus when scheduled-stance foot is in contact (scale > 0).
        gait_stance = torch.sum(E_stance * contact_filt.float(), dim=1)  # (N,) ∈ [0, 2]

        # (B) Phase-scheduled swing clearance reward — bonus when scheduled-swing foot is lifted (scale > 0).
        # clearance ∈ [0, 1]: saturates at gait_swing_height; partial lift already earns partial reward.
        clearance = torch.clamp(lift / self.cfg.gait_swing_height, 0.0, 1.0)  # (N, n_feet)
        gait_swing = torch.sum(E_swing * clearance, dim=1)  # (N,) ∈ [0, 2]

        # (C) Contact-gated anti-slip penalty (scale < 0).
        # Penalises stance feet that are sliding; complementary to swing clearance above.
        sole_vel_xy = torch.norm(self._robot.data.body_lin_vel_w[:, self._sole_body_ids, :2], dim=-1)  # (N, n_feet)
        slip_pen = torch.sum(contact_filt.float() * sole_vel_xy**2, dim=1)  # (N,) ≥ 0; scale < 0 → penalty

        rewards = {
            "track_lin_vel_xy_exp": lin_vel_error_mapped * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": yaw_rate_error_mapped * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "similar_to_default": similar_to_default * self.cfg.similar_to_default_reward_scale * self.step_dt,
            "base_height": base_height * self.cfg.base_height_reward_scale * self.step_dt,
            "termination": termination * self.cfg.termination_reward_scale * self.step_dt,
            "gait_stance": gait_stance * self.cfg.gait_stance_reward_scale * self.step_dt,
            "gait_swing": gait_swing * self.cfg.gait_swing_reward_scale * self.step_dt,
            "foot_slip": slip_pen * self.cfg.foot_slip_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # value target 유계화 — per-term 로깅은 원값 유지, 학습에 쓰는 총합만 클램프.
        reward = reward.clamp(-self._reward_clip, self._reward_clip)
        self.curriculum_rew_buf += reward
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        # ★ 2026-08-27: base 접촉만으로는 **뒤로 넘어진 자세가 종료되지 않는다.**
        #   이 다리는 몸통 아래로 두 다리가 뻗어 있어, 뒤로 자빠지면 hip/thigh/calf 링크가 몸통을
        #   받쳐 `base_collision` 이 지면에 닿지 않는다. 실측(`_workspace/hindleg_fall_probe.py`,
        #   `pace0819sym` model_48500, 256 env × 2800 step):
        #       뒤로 넘어짐 = 전 스텝의 91.96 %, 그 상태의 **종료율 0.000 %**
        #       앞으로 넘어짐 = 0.35 %, 종료율 5.65 %  (base 가 닿으므로 잡힌다)
        #       에피소드당 뒤로 넘어진 채 658.1 step = **에피소드의 92.9 %**
        #       time-out 674 건 전부(674/674) 넘어진 채로 끝났다
        #   즉 학습 분포의 대부분이 "누운 자세"였다. 기울기 기준을 더해 전후 대칭으로 끊는다.
        #   ⚠ 이 항을 켜면 종료 조건이 바뀌므로 **2026-08-26 이전 run 과 학습 곡선을 직접 비교할 수
        #     없다** (같은 플랜트가 아니라 같은 task 가 아니게 된다).
        if self.cfg.terminate_tilt_deg is not None:
            # projected_gravity_b[:, 2] = −cos(tilt). 기립 −1 → 넘어질수록 0 에 접근.
            died = died | (self._robot.data.projected_gravity_b[:, 2] > -self._tilt_cos_limit)
        if self.cfg.terminate_base_height is not None:
            died = died | (self._robot.data.root_link_pos_w[:, 2] < self.cfg.terminate_base_height)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        if self._foot_coupling:
            # 리셋 직후에는 직전 스텝의 반사관성 보정이 남아 있으면 안 된다 (전치가 그만큼 잘못 뺀다).
            self._mrefl_foot_prev[env_ids] = 0.0
        # Randomize gait phase for reset envs to decorrelate episodes across parallel envs.
        self._gait_phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        if self.cfg.history_observation:
            self.obs_history_buf[env_ids, :, :] = 0.0
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim_index(root_pose=default_root_state[:, :7], env_ids=env_ids)
        self._robot.write_root_velocity_to_sim_index(root_velocity=default_root_state[:, 7:], env_ids=env_ids)
        self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids)
        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/base_contact"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()

        if self.command_curriculum:
            extras["mean_lin_vel_x"] = torch.mean(self._commands[:, 0])
            extras["max_lin_vel_x"] = torch.max(self._commands[:, 0])
            extras["min_lin_vel_x"] = torch.min(self._commands[:, 0])
            extras["mean_ang_vel"] = torch.mean(torch.abs(self._commands[:, 2]))
            extras["curriculum_rew"] = torch.mean(self.curriculum_rew_buf[env_ids])
            success_mask = self.curriculum_rew_buf[env_ids] > self.curriculum_threshold
            self.curriculum_lin_vel_x[env_ids] += success_mask * self.curriculum_step
            self.curriculum_ang_vel[env_ids] += success_mask * self.curriculum_step
            self.curriculum_rew_buf[env_ids] = 0.0

        self.extras["log"].update(extras)

        # Sample new commands
        self._resample_commands(env_ids)

    def _resample_commands(self, env_ids: torch.Tensor):
        if self.command_curriculum:
            command_keys_in_order = [
                "lin_vel_x_range",
                "lin_vel_y_range",
                "ang_vel_range",
            ]
            for i in range(self.cfg.num_commands):
                if i < len(command_keys_in_order):
                    key = command_keys_in_order[i]
                    if key in self.cfg.command_cfg:
                        lower, upper = self.cfg.command_cfg[key]
                        if i == 0:  # lin_vel_x에 curriculum 적용
                            curr = self.curriculum_lin_vel_x[env_ids]
                            use_curriculum = curr < upper
                            low = torch.where(use_curriculum, curr - self.curriculum_step, torch.full_like(curr, lower))
                            high = torch.where(use_curriculum, curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
                        elif i == 1 or i == 2:  # ang_vel에 curriculum 적용
                            curr = self.curriculum_ang_vel[env_ids]
                            use_curriculum = curr < upper
                            # random sign 선택
                            direction = torch.randint(0, 2, (len(env_ids),), device=self.device) * 2 - 1  # {-1, +1}
                            signed_curr = curr * direction.float()
                            low = torch.where(
                                use_curriculum, signed_curr - self.curriculum_step, torch.full_like(curr, lower)
                            )
                            high = torch.where(use_curriculum, signed_curr, torch.full_like(curr, upper))
                            self._commands[env_ids, i] = torch.lerp(
                                low, high, torch.rand(len(env_ids), device=self.device)
                            )
        else:
            self._commands[env_ids, 0] = torch_rand_float(
                *self.cfg.command_cfg["lin_vel_x_range"], (len(env_ids),), self.device
            )
            self._commands[env_ids, 1] = torch_rand_float(
                *self.cfg.command_cfg["lin_vel_y_range"], (len(env_ids),), self.device
            )
            self._commands[env_ids, 2] = torch_rand_float(
                *self.cfg.command_cfg["ang_vel_range"], (len(env_ids),), self.device
            )

        # 일부 env는 정지(standing) 명령으로 강제 → 정책이 cmd=0 평형을 학습하게 함
        standing = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_standing_envs
        self._commands[env_ids[standing], :] = 0.0

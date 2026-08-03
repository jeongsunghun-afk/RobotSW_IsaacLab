# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Pedipulation 환경 — 제자리에서 지정된 발을 base frame 목표로 옮기고 유지한다.

명령 표현은 ALaM(P5)의 leg-role 벡터를 따른다: 어느 다리를 조작할지는 정책이 고르는 것이
아니라 명령 ``leg_role`` 로 주어지고, action 은 지지/조작 브랜치로 라우팅되어 쓰이지 않는
채널의 gradient 가 차단된다.

주의 — IsaacLab GO2 articulation 의 joint 순서는 **type-major** 다
(``FL_hip, FR_hip, RL_hip, RR_hip, FL_thigh, ...``). 다리 k 의 DOF 가 ``3k, 3k+1, 3k+2`` 라는
leg-major 산술은 성립하지 않으므로, 발↔DOF 매핑은 전부 이름 기반으로 만든다.
"""

from __future__ import annotations

import math

import torch
import warp as wp  # IsaacLab 3.0: _ALL_INDICES is a warp array → wp.to_torch()

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse

from .go2_pedipulation_env_cfg import (
    LEG_NAMES,
    PACE_ARMATURE,
    PACE_COULOMB,
    PACE_ENCODER_BIAS_MAG,
    PACE_VISCOUS,
    Go2PedipulationEnvCfg,
)


class Go2PedipulationEnv(DirectRLEnv):
    """Go2 발 조작(pedipulation) 환경."""

    cfg: Go2PedipulationEnvCfg

    def __init__(self, cfg: Go2PedipulationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.cfg = cfg
        device = self.device
        n_env = self.num_envs
        n_leg = self.cfg.num_legs

        # ── body / joint 인덱스 (전부 이름 기반) ──────────────────
        body_names = list(self._robot.data.body_names)
        joint_names = list(self._robot.data.joint_names)

        self._base_body_id = body_names.index("base")
        self._foot_body_ids = torch.tensor(
            [body_names.index(f"{leg}_foot") for leg in LEG_NAMES], dtype=torch.long, device=device
        )

        # joint -> leg 매핑. type-major 레이아웃이므로 산술 인덱싱 금지.
        leg_of_joint = []
        for name in joint_names:
            leg_prefix = name.split("_")[0]
            assert leg_prefix in LEG_NAMES, f"예상치 못한 joint 이름: {name}"
            leg_of_joint.append(LEG_NAMES.index(leg_prefix))
        self._leg_of_joint = torch.tensor(leg_of_joint, dtype=torch.long, device=device)
        print(f"[Go2PedipulationEnv] joint 순서: {joint_names}")
        print(f"[Go2PedipulationEnv] leg_of_joint: {leg_of_joint}")

        # hip joint (지지 다리 기본 자세 규제에서 더 강하게 볼 수 있도록 별도 보관)
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(joint_names) if "hip" in n], dtype=torch.long, device=device
        )

        # ── ContactSensor 인덱스 — articulation 순서와 반드시 대조 ──
        foot_sensor_names, _ = self.contact_sensor.find_bodies(".*_foot")
        base_sensor_ids, base_sensor_names = self.contact_sensor.find_bodies("base")
        sensor_body_names = list(self.contact_sensor.body_names)
        self._foot_sensor_ids = torch.tensor(
            [sensor_body_names.index(f"{leg}_foot") for leg in LEG_NAMES], dtype=torch.long, device=device
        )
        assert len(foot_sensor_names) == n_leg, f"발 센서 개수 불일치: {foot_sensor_names}"
        assert len(base_sensor_ids) == 1, f"base 센서 해석 실패: {base_sensor_names}"
        self._base_sensor_id = int(base_sensor_ids[0])

        # ── 명령 버퍼 ─────────────────────────────────────────────
        # leg_role: 1 = 지지(stance), 0 = 조작(manipulation)
        self._leg_role = torch.ones(n_env, n_leg, device=device)
        self._foot_target_b = torch.zeros(n_env, n_leg, 3, device=device)  # base frame 목표 [m]
        self._cmd_timer = torch.zeros(n_env, device=device)
        self._traj_phase = torch.zeros(n_env, device=device)  # 궤적 모드 위상 [rad]
        self._traj_center_b = torch.zeros(n_env, n_leg, 3, device=device)
        # 궤적 파라미터는 env 별로 다르다 — 속도 sweep 평가를 하려면 정책이 여러 각속도를
        # 겪어야 하므로 cfg 스칼라를 그대로 쓸 수 없다.
        self._traj_radius = torch.full((n_env,), self.cfg.command.circle_radius, device=device)
        self._traj_omega = torch.full((n_env,), self.cfg.command.circle_omega, device=device)
        self._traj_dir = torch.ones(n_env, device=device)  # +1 반시계 / −1 시계
        self._hold_counter = torch.zeros(n_env, n_leg, device=device)

        # 조작 다리 목표 관절각 (증분형 action 의 적분 상태)
        self._manip_joint_target = self._robot.data.default_joint_pos.clone()

        # ── 관측 / 액션 버퍼 ──────────────────────────────────────
        self._prev_actions = torch.zeros(n_env, self.cfg.action_space, device=device)
        self._prev_prev_actions = torch.zeros(n_env, self.cfg.action_space, device=device)
        self._proprio_history = torch.zeros(n_env, self.cfg.history_len, self.cfg.observation_space, device=device)
        self._processed_actions = self._robot.data.default_joint_pos.clone()
        self._prev_joint_target = self._robot.data.default_joint_pos.clone()
        # 관절 속도 관측 EMA 상태 (cfg.jvel_filter_alpha < 1 일 때만 쓴다)
        self._jvel_filt = torch.zeros(n_env, 12, device=device)
        # 관절 목표 출력 EMA 상태 (cfg.action_filter_alpha < 1 일 때만 쓴다)
        self._target_filt = self._robot.data.default_joint_pos.clone()

        # ── 물리 상수 ─────────────────────────────────────────────
        self._total_mass = float(self._robot.data.body_mass.torch[0].sum().item())
        self._gravity_mag = 9.81

        # ── nominal 발 위치 (base frame) ──────────────────────────
        # 목표는 이 위치를 중심으로 샘플한다 (close-range 보장). 시뮬이 아직 한 번도 스텝되지
        # 않았다면 body_pos_w 가 유효하지 않을 수 있으므로, 값이 그럴듯할 때만 확정하고
        # 아니면 첫 관측 시점에 다시 시도한다.
        self._nominal_valid = False
        self._nominal_foot_pos_b = torch.tensor(
            # GO2 기본 자세(hip ±0.1, thigh 0.8/1.0, calf -1.5)에서의 대략적인 발 위치 fallback
            [[[0.19, 0.14, -0.32], [0.19, -0.14, -0.32], [-0.19, 0.14, -0.32], [-0.19, -0.14, -0.32]]],
            device=device,
        )
        self._try_capture_nominal_foot_pos()

        # ── command curriculum ────────────────────────────────────
        self._cmd_box = torch.tensor(self.cfg.command.box_init, device=device)
        self._curr_err_sum = torch.zeros((), device=device)
        self._curr_err_count = torch.zeros((), device=device)

        # ── DR ───────────────────────────────────────────────────
        self._init_domain_rand()

        # ── episode 로깅 ─────────────────────────────────────────
        self._episode_sums = {
            key: torch.zeros(n_env, device=device)
            for key in (
                "track",
                "hold",
                "grav_moment",
                "support_area",
                "flat_orientation",
                "base_height",
                "stance_default",
                "base_drift",
                "feet_slip",
                "action_rate",
                "joint_target_rate",
                "action_smooth",
                "joint_acc",
                "joint_vel",
                "torque",
                "contact_force",
                "foot_speed",
                "foot_speed_barrier",
                "collision",
                "termination",
                "joint_limit",
            )
        }
        self._metric_track_err = torch.zeros(n_env, device=device)
        self._metric_steps = torch.zeros(n_env, device=device)
        self._metric_within = torch.zeros(n_env, device=device)
        # S4-G4 ① — 조작 발이 접촉 중인데 CoM 이 지지 다각형 밖에 있는 step 수(하중 전이).
        # PLAN §0b 는 이 카운터를 S1 에 미리 넣으라고 했다. 학습 내내 켜두면 게이트 판정이 공짜다.
        self._metric_contact_steps = torch.zeros(n_env, device=device)
        self._metric_unsafe_steps = torch.zeros(n_env, device=device)
        self._metric_manip_load_peak = torch.zeros(n_env, device=device)
        self._metric_hold = torch.zeros(n_env, device=device)
        self._episode_base_xy0 = torch.zeros(n_env, 2, device=device)
        # 커리큘럼 진단 — 승급 판정이 실제로 몇 번 돌았고 몇 번 올라갔는지
        self._curr_eval_count = torch.zeros((), device=device)
        self._curr_promote_count = torch.zeros((), device=device)

    # ──────────────────────────────────────────────────────────
    # Scene
    # ──────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.contact_sensor = ContactSensor(self.cfg.contact_sensor)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0, dynamic_friction=1.0, restitution=0.0
                ),
            ),
        )

        self.scene.clone_environments(copy_from_source=False)
        # Env isolation: cross-env 충돌 필터는 무조건 호출한다(GPU 포함). Direct env 는 scene cfg 에
        # entity 를 선언하지 않아 자동 경로(interactive_scene:218)를 타지 않으므로, 이 호출이
        # 빠지면 GPU 학습에서 env 간 로봇이 실제로 충돌한다.
        # 주의: filter_collisions 는 두 번째 호출이 조용한 no-op 이므로 전역 prim 을 여기서 전부 넘긴다.
        self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ──────────────────────────────────────────────────────────
    # 기하 헬퍼
    # ──────────────────────────────────────────────────────────

    def _base_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """base link 의 world 위치와 자세(xyzw)."""
        root_pos_w = self._robot.data.body_pos_w[:, self._base_body_id]  # [N,3]
        root_quat_w = self._robot.data.body_quat_w[:, self._base_body_id]  # [N,4] xyzw
        return root_pos_w, root_quat_w

    def _compute_foot_pos_b(self) -> torch.Tensor:
        """발 4개의 위치를 base frame 으로 변환해 반환. shape [N, 4, 3] [m].

        env origin 이 포함된 ``body_pos_w`` 를 그대로 쓰면 env_spacing 때문에 env 1 이상에서
        목표와 어긋난다. 반드시 base frame 으로 내려서 비교한다.
        """
        root_pos_w, root_quat_w = self._base_pose()
        rel = self._robot.data.body_pos_w[:, self._foot_body_ids] - root_pos_w.unsqueeze(1)  # [N,4,3]
        n_env, n_leg = rel.shape[:2]
        return quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, n_leg, -1).reshape(-1, 4), rel.reshape(-1, 3)
        ).view(n_env, n_leg, 3)

    def _compute_foot_vel_b(self) -> torch.Tensor:
        """발 4개의 링크 원점 선속도를 base frame 으로 변환. shape [N, 4, 3] [m/s].

        ``body_lin_vel_w`` 는 COM 기준이므로 링크 원점 속도인 ``body_link_lin_vel_w`` 를 쓴다.
        """
        _, root_quat_w = self._base_pose()
        vel_w = self._robot.data.body_link_lin_vel_w[:, self._foot_body_ids]  # [N,4,3]
        n_env, n_leg = vel_w.shape[:2]
        return quat_apply_inverse(
            root_quat_w.unsqueeze(1).expand(-1, n_leg, -1).reshape(-1, 4), vel_w.reshape(-1, 3)
        ).view(n_env, n_leg, 3)

    def _try_capture_nominal_foot_pos(self):
        """기본 자세에서의 발 위치를 base frame 으로 한 번만 캡처한다.

        시뮬 초기화 순서에 따라 첫 호출 시점의 ``body_pos_w`` 는 관절이 아직 0 인 상태
        (다리가 완전히 펴진 자세, 발 깊이 = thigh+calf = 0.426 m)일 수 있다. 그 값을 nominal 로
        잡으면 목표가 지면 아래에 놓여 task 자체가 성립하지 않으므로, **관절이 실제로 기본
        자세 근처일 때만** 확정한다.
        """
        if self._nominal_valid:
            return
        q_dev = torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos).max()
        if float(q_dev.item()) > 0.15:  # 기본 자세와 충분히 가깝지 않으면 보류
            return
        measured = self._compute_foot_pos_b().mean(dim=0, keepdim=True)  # [1,4,3]
        depth = -measured[0, :, 2]
        if bool(((depth > 0.20) & (depth < 0.40)).all().item()):
            self._nominal_foot_pos_b = measured.clone()
            self._nominal_valid = True
            print(f"[Go2PedipulationEnv] nominal foot pos (base frame):\n{self._nominal_foot_pos_b[0]}")

    def _com_pos_b(self) -> torch.Tensor:
        """전신 CoM 을 base frame 으로 반환. shape [N, 3] [m]."""
        root_pos_w, root_quat_w = self._base_pose()
        mass = self._robot.data.body_mass.torch  # [N, B]
        com_w = self._robot.data.body_com_pos_w.torch  # [N, B, 3]
        com_mean_w = (com_w * mass.unsqueeze(-1)).sum(dim=1) / mass.sum(dim=1, keepdim=True)
        return quat_apply_inverse(root_quat_w, com_mean_w - root_pos_w)

    # ──────────────────────────────────────────────────────────
    # 명령 생성
    # ──────────────────────────────────────────────────────────

    def _resample_command(self, env_ids: torch.Tensor):
        """조작 다리와 목표 위치를 재샘플한다."""
        n = int(env_ids.numel())
        if n == 0:
            return
        cmd = self.cfg.command
        device = self.device

        # ── 조작 다리 선택 ────────────────────────────────────
        candidates = torch.tensor(cmd.manip_leg_candidates, dtype=torch.long, device=device)
        role = torch.ones(n, self.cfg.num_legs, device=device)
        k = min(cmd.num_manip_legs, candidates.numel())
        if k > 0:
            # 후보 중 k개를 중복 없이 뽑는다 (env 별 독립).
            pick = torch.rand(n, candidates.numel(), device=device).argsort(dim=-1)[:, :k]
            manip_legs = candidates[pick]  # [n, k]
            role.scatter_(1, manip_legs, 0.0)
        self._leg_role[env_ids] = role

        # ── 목표 위치: nominal 발 위치 + 커리큘럼 박스 오프셋 ──
        nominal = self._nominal_foot_pos_b.expand(n, -1, -1)  # [n,4,3]
        box = self._cmd_box  # [3]
        n_leg = self.cfg.num_legs

        # 궤적 모드에서는 **원 전체가 박스 안에 들어가야** 하므로 중심 샘플링 범위를 반경만큼
        # 좁힌다. 이걸 안 하면 목표가 학습된 워크스페이스 밖으로 나가 도달 자체가 불가능해진다.
        if cmd.trajectory_mode == "circle":
            if cmd.circle_randomize:
                r = self._sample(cmd.circle_radius_range, n, 1).squeeze(-1)
                om = self._sample(cmd.circle_omega_range, n, 1).squeeze(-1)
                sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
            else:
                r = torch.full((n,), cmd.circle_radius, device=device)
                om = torch.full((n,), cmd.circle_omega, device=device)
                sign = torch.ones(n, device=device)
            # 일부 env 는 반경·각속도를 0 으로 눌러 **정지 목표**로 만든다. circle 만으로
            # 학습하면 정책이 "목표는 늘 움직인다"를 전제로 삼아, 정지 목표를 유지할 때
            # 오히려 더 떤다 (측정: circle 학습 후 유지 중 떨림이 1.29→2.83°/step).
            if cmd.static_fraction > 0.0:
                is_static = torch.rand(n, device=device) < cmd.static_fraction
                r = torch.where(is_static, torch.zeros_like(r), r)
                om = torch.where(is_static, torch.zeros_like(om), om)
            m = cmd.circle_center_margin
            # 반경이 박스에 안 들어가면 반경을 줄인다 (박스를 넓히지 않는다).
            r = torch.minimum(r, torch.clamp((box[2] - 2.0 * m) * 0.5, min=1e-3))
            r = torch.minimum(r, torch.clamp(box[0] - m, min=1e-3))
            self._traj_radius[env_ids] = r
            self._traj_omega[env_ids] = om
            self._traj_dir[env_ids] = sign
            x_half = torch.clamp(box[0] - r - m, min=0.0)
            z_lo, z_hi = r + m, torch.clamp(box[2] - r - m, min=r + m)
            cx = (torch.rand(n, device=device) * 2.0 - 1.0) * x_half
            cz = torch.rand(n, device=device) * (z_hi - z_lo) + z_lo
            off = torch.empty(n, n_leg, 3, device=device)
            off[..., 0] = cx.unsqueeze(-1)
            off[..., 1] = (torch.rand(n, n_leg, device=device) * 2.0 - 1.0) * box[1]
            off[..., 2] = cz.unsqueeze(-1)
        else:
            off = torch.empty(n, n_leg, 3, device=device)
            off[..., 0] = (torch.rand(n, n_leg, device=device) * 2.0 - 1.0) * box[0]
            off[..., 1] = (torch.rand(n, n_leg, device=device) * 2.0 - 1.0) * box[1]
            # z 는 위로만 — 발이 바닥에서 시작하므로 들어올리는 방향
            off[..., 2] = torch.rand(n, n_leg, device=device) * box[2]
        target = nominal + off

        self._traj_center_b[env_ids] = target
        self._traj_phase[env_ids] = torch.rand(n, device=device) * 2.0 * math.pi
        self._hold_counter[env_ids] = 0.0
        if cmd.trajectory_mode == "circle":
            # 첫 step 부터 위상에 맞는 점을 가리키게 한다 — 중심을 목표로 두면 리셋 직후
            # 한 step 동안만 반경만큼 어긋난 목표가 관측된다.
            ph = self._traj_phase[env_ids]
            rr = self._traj_radius[env_ids]
            target = target.clone()
            target[..., 0] += (rr * torch.cos(ph)).unsqueeze(-1)
            target[..., 2] += (rr * torch.sin(ph)).unsqueeze(-1)
        self._foot_target_b[env_ids] = target

        self._cmd_timer[env_ids] = (
            torch.rand(n, device=device) * (cmd.resample_time_max - cmd.resample_time_min) + cmd.resample_time_min
        )

    def _advance_trajectory(self):
        """S2 궤적 모드에서 목표를 매 step 갱신한다 (static 모드는 no-op)."""
        cmd = self.cfg.command
        if cmd.trajectory_mode == "static":
            return
        if cmd.trajectory_mode == "circle":
            # 각속도·반경·회전방향은 env 별로 다르다 (cfg 스칼라가 아니라 버퍼).
            self._traj_phase += self._traj_dir * self._traj_omega * self.step_dt
            # base frame x-z 평면 원 (발을 앞뒤/위아래로 돌린다)
            dx = self._traj_radius * torch.cos(self._traj_phase)
            dz = self._traj_radius * torch.sin(self._traj_phase)
            # ⚠ 재할당하면 텐서 객체가 바뀐다. 평가 스크립트가 이 버퍼에 밖에서 쓰기 때문에
            #   (inference tensor 문제) 반드시 in-place 로 갱신한다.
            self._foot_target_b.copy_(self._traj_center_b)
            self._foot_target_b[..., 0] += dx.unsqueeze(-1)
            self._foot_target_b[..., 2] += dz.unsqueeze(-1)
        else:
            raise ValueError(f"알 수 없는 trajectory_mode: {cmd.trajectory_mode}")

    # ──────────────────────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        scale = self.cfg.action_scale
        default_q = self._robot.data.default_joint_pos

        a_loc = self._actions[:, 0:12]
        a_man = self._actions[:, 12:24]

        # leg_role(4) → joint(12) 확장. type-major 이므로 이름 기반 매핑을 쓴다
        # (repeat_interleave 는 leg-major 가정이라 여기서는 틀린다).
        role_j = self._leg_role[:, self._leg_of_joint]  # [N,12]

        # 지지 다리: 기본 자세 대비 절대 offset
        # hip(abduction) 액션 축소는 지지 다리 슬롯에만 적용한다. a_man 은 적분형이라
        # 같은 계수를 곱하면 "범위 축소"가 아니라 "hip 이동 속도 감쇠"가 되고, 조작 다리의
        # abduction 권한은 이 태스크의 목적 자체다.
        if self.cfg.hip_scale_reduction:
            a_loc = a_loc.clone()
            a_loc[:, self._hip_joint_ids] *= 0.5
        loc_target = a_loc * scale + default_q
        # 조작 다리: 직전 목표 대비 증분 (적분형) — 스텝당 변화량을 제한해 windup 방지
        delta = torch.clamp(a_man * scale, -self.cfg.manip_delta_clip, self.cfg.manip_delta_clip)
        man_target = self._manip_joint_target + delta

        target = role_j * loc_target + (1.0 - role_j) * man_target

        # 관절 한계로 clamp — 적분형 목표가 한계 밖으로 누적되는 것을 막는다
        limits = self._robot.data.joint_pos_limits.torch  # [N,12,2]
        target = torch.clamp(target, limits[..., 0], limits[..., 1])

        # 다음 스텝의 적분 기준. 지지 다리 슬롯은 현재 목표를 그대로 이어받아,
        # stance→manipulation 전환 시 목표가 튀지 않게 한다.
        #
        # ⚠ 아래 출력 필터를 **거치기 전** 값을 적분 기준으로 남긴다. 필터 출력을 되먹이면
        #   적분기가 leaky 해져 plant 자체가 달라진다 — 여기서 원하는 것은 "적분기 + 출력
        #   필터"이지 "누설 적분기"가 아니다.
        self._manip_joint_target = target.clone()

        # 관절 목표 출력 EMA — 정책의 이득은 두고 고주파 성분만 깎는다.
        if self.cfg.action_filter_alpha < 1.0:
            af = self.cfg.action_filter_alpha
            self._target_filt.mul_(1.0 - af).add_(target, alpha=af)
            target = self._target_filt.clone()

        if self.cfg.domain_rand and self.cfg.dr.randomize_action_delay:
            self._action_delay_buf = torch.cat([target.unsqueeze(1), self._action_delay_buf[:, :-1]], dim=1)
            rows = torch.arange(self.num_envs, device=self.device)
            self._processed_actions = self._action_delay_buf[rows, self._action_delay_steps]
        else:
            self._processed_actions = target

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _post_physics_step(self):
        """물리 step 이후 명령·외란을 갱신한다.

        ⚠ ``_post_physics_step`` 은 ``DirectRLEnv`` 의 훅이 **아니다.** 이 저장소의 여러 env 가
        같은 이름의 메서드를 정의해 두었지만 프레임워크는 호출하지 않으며, 실제로 부르는 것은
        ``skeleton_wtw_env.py`` 하나뿐이다. 그래서 이 메서드는 오랫동안 **죽어 있었고**
        궤적 진행·에피소드 중 명령 재샘플·DR base push 가 전부 실행되지 않았다.
        ``_get_dones`` 첫머리에서 명시적으로 호출한다 — 물리가 t+1 로 간 뒤, 보상과 관측이
        계산되기 전이 목표를 t+1 로 옮길 자리다.
        """
        self._advance_trajectory()

        self._cmd_timer -= self.step_dt
        change_ids = (self._cmd_timer <= 0.0).nonzero(as_tuple=False).flatten()
        if change_ids.numel() > 0:
            self._resample_command(change_ids)

        if self.cfg.domain_rand and self.cfg.dr.push_robot:
            self._push_timer -= self.step_dt
            push_ids = (self._push_timer <= 0.0).nonzero(as_tuple=False).flatten()
            if push_ids.numel() > 0:
                self._push_robots(push_ids)
                self._push_timer[push_ids] = self.cfg.dr.push_interval_s

    # ──────────────────────────────────────────────────────────
    # Observation
    # ──────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        self._try_capture_nominal_foot_pos()
        foot_pos_b = self._compute_foot_pos_b()  # [N,4,3]
        manip_mask = (1.0 - self._leg_role).unsqueeze(-1)  # [N,4,1]
        # 추종 오차는 매 step 재계산한다. 지지 다리 슬롯은 0 으로 마스킹.
        foot_err_b = (self._foot_target_b - foot_pos_b) * manip_mask

        proprio = torch.cat(
            [
                self._robot.data.projected_gravity_b,  # 3
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,  # 12
                self._robot.data.joint_vel,  # 12
                self._prev_actions,  # 28
                foot_pos_b.reshape(self.num_envs, -1),  # 12
                self._leg_role,  # 4
                foot_err_b.reshape(self.num_envs, -1),  # 12
            ],
            dim=-1,
        )
        policy_obs = self._apply_obs_dr(proprio)

        # 관절 속도 관측 EMA — 노이즈를 **주입한 뒤** 거른다. 실기의 필터가 보는 것도
        # (참값 + 센서 노이즈) 이므로 순서가 이래야 실기와 같은 양이 된다.
        if self.cfg.jvel_filter_alpha < 1.0:
            a = self.cfg.jvel_filter_alpha
            self._jvel_filt.mul_(1.0 - a).add_(policy_obs[:, 15:27], alpha=a)
            policy_obs = torch.cat([policy_obs[:, :15], self._jvel_filt, policy_obs[:, 27:]], dim=-1)

        # history 링버퍼 (노이즈 포함 policy obs 저장)
        self._proprio_history = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([policy_obs] * self.cfg.history_len, dim=1),
            torch.cat([self._proprio_history[:, 1:], policy_obs.unsqueeze(1)], dim=1),
        )

        # critic 전용 privileged obs — actor 는 절대 보지 않는다.
        root_pos_w, _ = self._base_pose()
        contact_mag = torch.norm(self.contact_sensor.data.net_forces_w[:, self._foot_sensor_ids], dim=-1)  # [N,4]
        priv = torch.cat(
            [
                self._robot.data.root_lin_vel_b.torch,  # 3
                self._robot.data.root_ang_vel_b.torch,  # 3
                root_pos_w[:, 2:3],  # 1
                contact_mag * 0.01,  # 4 (스케일링)
                self._get_priv_latent(),  # 19
            ],
            dim=-1,
        )

        return {
            "policy": policy_obs,
            "history": self._proprio_history.reshape(self.num_envs, -1),
            "priv": priv,
        }

    # ──────────────────────────────────────────────────────────
    # Reward
    # ──────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        cfg = self.cfg
        foot_pos_b = self._compute_foot_pos_b()
        foot_vel_b = self._compute_foot_vel_b()
        manip_mask = 1.0 - self._leg_role  # [N,4] 1 = 조작 다리
        n_manip = manip_mask.sum(dim=-1).clamp(min=1.0)

        # ── (1) 위치 추종 — P1/P3 지수 커널 ─────────────────────
        err = torch.norm((self._foot_target_b - foot_pos_b), dim=-1)  # [N,4]
        track_per_leg = torch.exp(-err / cfg.track_sigma)
        r_track = (track_per_leg * manip_mask).sum(dim=-1) / n_manip

        # ── (2) hold 보너스 — P4 (delta_ee 이내 최소 지속시간) ──
        within = (err < cfg.hold_delta_ee).float() * manip_mask
        self._hold_counter = (self._hold_counter + within) * within  # 이탈하면 0으로 리셋
        held = (self._hold_counter >= cfg.hold_min_steps).float() * manip_mask
        r_hold = held.sum(dim=-1) / n_manip
        # 진단: "임계 안에 들어온 적이 있는가"(within)와 "연속으로 머물렀는가"(hold)를 분리한다.
        # within 은 높은데 hold 만 0이면 목표 주변에서 진동해 카운터가 계속 깨지는 것이고,
        # 둘 다 낮으면 아직 도달 자체를 못 배운 것이다. 이 구분 없이는 S1-G3 판정이 불가능하다.
        self._metric_within += within.sum(dim=-1) / n_manip
        self._metric_hold += r_hold

        # ── (3) 지지 안정성 — P5 CoF 중력모멘트 ─────────────────
        stance = self._leg_role  # [N,4]
        n_stance = stance.sum(dim=-1).clamp(min=1.0)
        cof_xy = (foot_pos_b[..., :2] * stance.unsqueeze(-1)).sum(dim=1) / n_stance.unsqueeze(-1)
        com_b = self._com_pos_b()
        lever = torch.norm(com_b[:, :2] - cof_xy, dim=-1)  # [m]
        grav_moment = lever * self._total_mass * self._gravity_mag  # [N·m]
        r_grav = torch.exp(-grav_moment / cfg.grav_moment_sigma)

        # ── (4) 지지 다각형 면적 — P5 ───────────────────────────
        area = self._support_polygon_area(foot_pos_b, stance)
        r_area = torch.clamp(torch.exp(area / cfg.support_area_sigma) - 1.0, max=1.0)

        # ── (5) 자세 유지 ───────────────────────────────────────
        grav_b = self._robot.data.projected_gravity_b
        p_flat = torch.sum(grav_b[:, :2] ** 2, dim=-1)
        root_pos_w, _ = self._base_pose()
        p_height = (root_pos_w[:, 2] - cfg.base_height_target) ** 2

        # ── (6) 지지 다리 규제 ──────────────────────────────────
        role_j = self._leg_role[:, self._leg_of_joint]
        q_dev = torch.abs(self._robot.data.joint_pos - self._robot.data.default_joint_pos)
        p_stance_default = (q_dev * role_j).sum(dim=-1)

        # base 병진 — 에피소드 시작 위치 기준. 데드존(필수 counterbalance) 밖만 2차로 벌한다.
        # ⚠ `_episode_base_xy0` 는 리셋에서 갱신되고 있었지만 어떤 보상 항에도 쓰이지 않았다.
        base_xy = self._robot.data.body_pos_w[:, self._base_body_id, :2]
        base_drift = torch.norm(base_xy - self._episode_base_xy0, dim=-1)
        p_base_drift = torch.clamp(base_drift - cfg.base_drift_free, min=0.0) ** 2

        contact_force = torch.norm(self.contact_sensor.data.net_forces_w[:, self._foot_sensor_ids], dim=-1)  # [N,4]
        in_contact = (contact_force > 1.0).float()
        p_slip = (torch.norm(foot_vel_b[..., :2], dim=-1) ** 2 * in_contact * stance).sum(dim=-1)

        # ── (7) action 규제 — 절대 크기가 아니라 변화율/가속/토크 ──
        # ⚠ p_action_rate 는 **경로에 따라 물리적 의미가 다르다.** 지지 다리의 a_loc 는 위치이므로
        #   Δa 는 속도지만, 조작 다리의 a_man 은 이미 증분(=속도)이라 Δa_man 은 **가속도**다.
        #   그래서 이 항만으로는 조작 다리에 1차(속도) 페널티가 전혀 걸리지 않고, a_man 이 0 이
        #   아닌 상수로 유지되는 목표 표류에는 비용이 0 이다. 실측 떨림이 조작 다리 4.83°/step
        #   대 지지 다리 2.16°/step 로 갈린 것이 이 비대칭이다.
        #   아래 p_joint_target_rate 가 두 경로에 동일하게 걸리는 1차 항이다.
        p_action_rate = torch.sum((self._actions - self._prev_actions) ** 2, dim=-1)
        # 실제로 액추에이터에 보내는 관절 목표의 스텝당 변화량. 모터가 떠는 양이 바로 이것이다.
        p_joint_target_rate = torch.sum((self._processed_actions - self._prev_joint_target) ** 2, dim=-1)
        p_action_smooth = torch.sum((self._actions - 2.0 * self._prev_actions + self._prev_prev_actions) ** 2, dim=-1)
        joint_acc = (self._robot.data.joint_vel - self._prev_joint_vel) / self.step_dt
        p_joint_acc = torch.sum(joint_acc**2, dim=-1)
        p_joint_vel = torch.sum(self._robot.data.joint_vel.torch**2, dim=-1)
        p_torque = torch.sum(self._robot.data.applied_torque.torch**2, dim=-1)

        # ── (8) 접촉 / 안전 ─────────────────────────────────────
        # 조작 다리의 접촉력만 벌한다 (지지 다리는 당연히 접촉 중).
        p_contact = (contact_force * manip_mask).sum(dim=-1)
        foot_speed = torch.norm(foot_vel_b, dim=-1)  # [N,4]
        p_foot_speed = (torch.clamp(foot_speed - cfg.foot_speed_penalty_threshold, min=0.0) ** 2 * manip_mask).sum(
            dim=-1
        )
        p_foot_speed_barrier = (torch.clamp(foot_speed - cfg.max_foot_speed, min=0.0) ** 2 * manip_mask).sum(dim=-1)

        undesired = self._undesired_contacts()
        limits = self._robot.data.joint_pos_limits.torch
        over = (self._robot.data.joint_pos - limits[..., 1]).clamp(min=0.0) + (
            limits[..., 0] - self._robot.data.joint_pos
        ).clamp(min=0.0)
        p_joint_limit = over.sum(dim=-1)

        terms = {
            "track": cfg.w_track * r_track,
            "hold": cfg.w_hold * r_hold,
            "grav_moment": cfg.w_grav_moment * r_grav,
            "support_area": cfg.w_support_area * r_area,
            "flat_orientation": cfg.w_flat_orientation * p_flat,
            "base_height": cfg.w_base_height * p_height,
            "stance_default": cfg.w_stance_default * p_stance_default,
            "base_drift": cfg.w_base_drift * p_base_drift,
            "feet_slip": cfg.w_feet_slip * p_slip,
            "action_rate": cfg.w_action_rate * p_action_rate,
            "joint_target_rate": cfg.w_joint_target_rate * p_joint_target_rate,
            "action_smooth": cfg.w_action_smooth * p_action_smooth,
            "joint_acc": cfg.w_joint_acc * p_joint_acc,
            "joint_vel": cfg.w_joint_vel * p_joint_vel,
            "torque": cfg.w_torque * p_torque,
            "contact_force": cfg.w_contact_force * p_contact,
            "foot_speed": cfg.w_foot_speed * p_foot_speed,
            "foot_speed_barrier": cfg.w_foot_speed_barrier * p_foot_speed_barrier,
            "collision": cfg.w_collision * undesired,
            "termination": cfg.w_termination * self._died.float(),
            "joint_limit": cfg.w_joint_limit * p_joint_limit,
        }

        reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in terms.items():
            reward = reward + value
            self._episode_sums[key] += value

        # ── 커리큘럼용 지표 (조작 다리 추종 오차) ────────────────
        # ── S4-G4 ① 안전 계측 (보상에는 안 들어간다. 판정용 누적만) ──
        manip_cf = self._manip_contact_force()
        in_contact_manip = manip_cf > cfg.contact_detect_force
        margin = self._stance_support_margin()
        self._metric_contact_steps += in_contact_manip.float()
        self._metric_unsafe_steps += (in_contact_manip & (margin < cfg.support_margin_min)).float()
        self._metric_manip_load_peak = torch.maximum(self._metric_manip_load_peak, manip_cf)

        manip_err = (err * manip_mask).sum(dim=-1) / n_manip
        self._metric_track_err += manip_err
        self._metric_steps += 1.0

        # ── 다음 스텝용 상태 갱신 ────────────────────────────────
        self._prev_prev_actions = self._prev_actions.clone()
        self._prev_actions = self._actions.clone()
        self._prev_joint_vel = self._robot.data.joint_vel.torch.clone()
        self._prev_joint_target = self._processed_actions.clone()

        return reward

    def _support_polygon_area(self, foot_pos_b: torch.Tensor, stance: torch.Tensor) -> torch.Tensor:
        """지지 발들이 이루는 다각형의 xy 면적 [m²].

        발을 시계 방향(FL, FR, RR, RL)으로 정렬한 뒤, 지지가 아닌 정점을 직전 지지 정점으로
        forward-fill 한다. shoelace 에서 중복 정점이 만드는 변은 면적에 기여하지 않으므로,
        결과는 지지 발만으로 이루어진 다각형의 면적과 정확히 같다.
        """
        cyclic = torch.tensor([0, 1, 3, 2], dtype=torch.long, device=foot_pos_b.device)  # FL, FR, RR, RL
        pts = foot_pos_b[:, cyclic, :2]  # [N,4,2]
        mask = stance[:, cyclic]  # [N,4]

        # 순환 forward-fill: 4개 정점이므로 4회 순회하면 모든 결측이 채워진다.
        filled = pts.clone()
        valid = mask.clone()
        for _ in range(4):
            prev_filled = torch.roll(filled, shifts=1, dims=1)
            prev_valid = torch.roll(valid, shifts=1, dims=1)
            take = ((valid < 0.5) & (prev_valid > 0.5)).unsqueeze(-1)
            filled = torch.where(take, prev_filled, filled)
            valid = torch.where(take.squeeze(-1), torch.ones_like(valid), valid)

        nxt = torch.roll(filled, shifts=-1, dims=1)
        cross = filled[..., 0] * nxt[..., 1] - nxt[..., 0] * filled[..., 1]
        return 0.5 * torch.abs(cross.sum(dim=-1))

    def _stance_support_margin(self) -> torch.Tensor:
        """CoM 수평투영에서 **지지 발만으로 이루어진** 다각형 경계까지의 부호 있는 여유 [m].

        양수 = 안쪽(조작 발 없이도 균형이 성립) · 음수 = 바깥(조작 발에 의존).

        이것이 PLAN S4-G4 ①("접촉 시 조작 발이 지지 다각형에서 제외됨")의 판정량이다.
        힘 게이트(peak ≤40 N)만으로는 부족하다 — **자중 147.3 N > ISO 손 한계 140 N** 이라
        하중 전이가 일어나면 접촉력이 작아도 위험하기 때문이다.

        ⚠ world 수평면에서 잰다. base frame xy 는 몸통이 기울면 수평이 아니다.
        """
        foot_w = self._robot.data.body_pos_w[:, self._foot_body_ids, :2]  # [N,4,2]
        mass = self._robot.data.body_mass.torch
        com_w = self._robot.data.body_com_pos_w.torch
        com_xy = (com_w * mass.unsqueeze(-1)).sum(dim=1)[:, :2] / mass.sum(dim=1, keepdim=True)

        stance = self._leg_role  # 1 = 지지
        cyclic = torch.tensor([0, 1, 3, 2], dtype=torch.long, device=foot_w.device)  # FL, FR, RR, RL
        pts = foot_w[:, cyclic, :]
        valid = stance[:, cyclic].clone()

        # 비지지 정점을 직전 지지 정점으로 순환 forward-fill (면적 계산과 같은 처리).
        filled = pts.clone()
        for _ in range(4):
            prev_filled = torch.roll(filled, shifts=1, dims=1)
            prev_valid = torch.roll(valid, shifts=1, dims=1)
            take = ((valid < 0.5) & (prev_valid > 0.5)).unsqueeze(-1)
            filled = torch.where(take, prev_filled, filled)
            valid = torch.where(take.squeeze(-1), torch.ones_like(valid), valid)

        nxt = torch.roll(filled, shifts=-1, dims=1)
        edge = nxt - filled  # [N,4,2]
        rel = com_xy.unsqueeze(1) - filled
        cross = edge[..., 0] * rel[..., 1] - edge[..., 1] * rel[..., 0]
        edge_len = torch.norm(edge, dim=-1)

        # 중복 정점이 만든 길이 0 변은 판정에서 제외한다 (거리가 정의되지 않는다).
        real = edge_len > 1e-6
        dist = torch.where(real, cross / edge_len.clamp(min=1e-6), torch.full_like(cross, 1e3))

        # 다각형 정점 순서(시계/반시계)는 자세에 따라 뒤집힐 수 있으므로 부호를 고정하지 않고,
        # 모든 변에 대해 같은 쪽이면 안쪽으로 본다. 여유는 가장 가까운 변까지의 거리.
        pos_margin = torch.amin(torch.where(real, dist, torch.full_like(dist, 1e3)), dim=-1)
        neg_margin = torch.amin(torch.where(real, -dist, torch.full_like(dist, 1e3)), dim=-1)
        return torch.maximum(pos_margin, neg_margin)

    def _manip_contact_force(self) -> torch.Tensor:
        """조작 다리 발의 접촉력 크기 [N]. shape [N]. 조작 다리가 없으면 0."""
        cf = torch.norm(self.contact_sensor.data.net_forces_w[:, self._foot_sensor_ids], dim=-1)  # [N,4]
        manip = 1.0 - self._leg_role
        return (cf * manip).sum(dim=-1)

    def _manip_foot_speed(self) -> torch.Tensor:
        """조작 다리 발 끝 속도 크기 [m/s] (world). shape [N]."""
        v = self._robot.data.body_lin_vel_w[:, self._foot_body_ids, :]  # [N,4,3]
        manip = (1.0 - self._leg_role).unsqueeze(-1)
        return torch.norm((v * manip).sum(dim=1), dim=-1)

    def _undesired_contacts(self) -> torch.Tensor:
        """발 이외 부위의 접촉 개수."""
        forces = torch.norm(self.contact_sensor.data.net_forces_w, dim=-1)  # [N, B]
        mask = torch.ones(forces.shape[-1], device=self.device)
        mask[self._foot_sensor_ids] = 0.0
        return ((forces > 1.0).float() * mask).sum(dim=-1)

    # ──────────────────────────────────────────────────────────
    # Termination
    # ──────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # DirectRLEnv 가 물리 step 직후 처음 부르는 훅이다. 명령·외란 갱신을 여기에 건다
        # (이유는 _post_physics_step 의 docstring 참조).
        self._post_physics_step()

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        if self.cfg.early_termination:
            root_pos_w, _ = self._base_pose()
            died = root_pos_w[:, 2] < self.cfg.termination_height

            grav_b = self._robot.data.projected_gravity_b
            died = died | (grav_b[:, 2] > 0.0)

            roll = torch.atan2(grav_b[:, 1], -grav_b[:, 2])
            pitch = torch.atan2(-grav_b[:, 0], torch.sqrt(grav_b[:, 1] ** 2 + grav_b[:, 2] ** 2))
            roll_limit = self.cfg.roll_termination_deg * math.pi / 180.0
            pitch_limit = self.cfg.pitch_termination_deg * math.pi / 180.0
            died = died | (torch.abs(roll) > roll_limit) | (torch.abs(pitch) > pitch_limit)

            base_force = torch.norm(self.contact_sensor.data.net_forces_w[:, self._base_sensor_id], dim=-1)
            died = died | (base_force > self.cfg.contact_force_threshold)

            died = died & (self.episode_length_buf > 1)
        else:
            died = torch.zeros_like(time_out)

        self._died = died
        return died, time_out

    # ──────────────────────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = wp.to_torch(self._robot._ALL_INDICES)  # IsaacLab 3.0: _ALL_INDICES is wp.array
        assert env_ids is not None
        env_ids_long = env_ids.to(torch.long)
        env_ids_int = env_ids.to(torch.int32)

        # base 표류량은 **리셋으로 로봇을 되돌리기 전에** 측정해야 한다.
        base_drift = torch.norm(
            self._robot.data.body_pos_w[env_ids_long][:, self._base_body_id, :2] - self._episode_base_xy0[env_ids_long],
            dim=-1,
        )

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        # 기본 자세 + 관절 노이즈로 리셋 (RSI 아님 — 모션 데이터 미사용)
        root_state = self._robot.data.default_root_state[env_ids_long].clone()
        root_state[:, 0:3] = root_state[:, 0:3] + self.scene.env_origins[env_ids_long]
        joint_pos = self._robot.data.default_joint_pos[env_ids_long].clone()
        joint_pos = joint_pos + (torch.rand_like(joint_pos) * 2.0 - 1.0) * 0.1
        joint_vel = self._robot.data.default_joint_vel[env_ids_long].clone()

        self._robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids_int)
        self._robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids_int)
        self._robot.write_joint_state_to_sim_index(position=joint_pos, velocity=joint_vel, env_ids=env_ids_int)

        # 버퍼 초기화 — 새로 추가한 텐서는 전부 여기서 리셋한다.
        self._prev_actions[env_ids_long] = 0.0
        self._prev_prev_actions[env_ids_long] = 0.0
        self._proprio_history[env_ids_long] = 0.0
        self._manip_joint_target[env_ids_long] = self._robot.data.default_joint_pos[env_ids_long].clone()
        self._prev_joint_target[env_ids_long] = self._robot.data.default_joint_pos[env_ids_long].clone()
        self._prev_joint_vel[env_ids_long] = 0.0
        self._jvel_filt[env_ids_long] = 0.0
        self._target_filt[env_ids_long] = self._robot.data.default_joint_pos[env_ids_long].clone()
        self._hold_counter[env_ids_long] = 0.0
        self._episode_base_xy0[env_ids_long] = root_state[:, 0:2]

        self._resample_command(env_ids_long)

        # ── 커리큘럼 갱신 ────────────────────────────────────────
        # ⚠ 스텝을 한 번도 밟지 않은 env(최초 전체 리셋 등)는 오차 0 으로 잡혀 커리큘럼을
        #    부당하게 승급시킨다. 실제로 진행된 에피소드만 집계한다.
        # ⚠ 아래에서 버퍼를 0으로 밀기 때문에, 로깅에 쓸 값은 **여기서 스냅샷**을 떠야 한다.
        steps_done = self._metric_steps[env_ids_long]
        valid = steps_done > 0
        steps_safe = steps_done.clamp(min=1.0)
        mean_err = self._metric_track_err[env_ids_long] / steps_safe
        within_rate = self._metric_within[env_ids_long] / steps_safe
        hold_rate = self._metric_hold[env_ids_long] / steps_safe
        if bool(valid.any().item()):
            self._curr_err_sum += mean_err[valid].sum()
            self._curr_err_count += float(valid.sum().item())
            self._update_curriculum()
        self._metric_track_err[env_ids_long] = 0.0
        self._metric_steps[env_ids_long] = 0.0
        self._metric_within[env_ids_long] = 0.0
        self._metric_contact_steps[env_ids_long] = 0.0
        self._metric_unsafe_steps[env_ids_long] = 0.0
        self._metric_manip_load_peak[env_ids_long] = 0.0
        self._metric_hold[env_ids_long] = 0.0

        # ── 로깅 ─────────────────────────────────────────────────
        extras: dict = {}
        for key in self._episode_sums:
            extras[f"Episode_Reward/{key}"] = torch.mean(
                self._episode_sums[key][env_ids_long] / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids_long] = 0.0
        extras["Metric/track_err_m"] = (
            torch.mean(mean_err[valid]) if bool(valid.any().item()) else torch.zeros((), device=self.device)
        )
        _zero = torch.zeros((), device=self.device)
        has_valid = bool(valid.any().item())
        extras["Metric/within_rate"] = torch.mean(within_rate[valid]) if has_valid else _zero
        extras["Metric/hold_rate"] = torch.mean(hold_rate[valid]) if has_valid else _zero
        extras["Metric/curriculum_evals"] = self._curr_eval_count.clone()
        extras["Metric/curriculum_promotions"] = self._curr_promote_count.clone()
        extras["Metric/cmd_box_x"] = self._cmd_box[0].clone()
        extras["Metric/cmd_box_y"] = self._cmd_box[1].clone()
        extras["Metric/cmd_box_z"] = self._cmd_box[2].clone()
        extras["Metric/base_drift_m"] = torch.mean(base_drift)
        _cs = self._metric_contact_steps[env_ids_long]
        extras["Metric/manip_contact_rate"] = torch.mean(_cs / steps_safe)
        extras["Metric/support_violation_rate"] = torch.mean(
            self._metric_unsafe_steps[env_ids_long] / _cs.clamp(min=1.0)
        )
        extras["Metric/manip_load_peak_N"] = torch.mean(self._metric_manip_load_peak[env_ids_long])

        if not isinstance(self.extras, dict):
            self.extras = {}
        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"].update(extras)

        self._apply_domain_rand(env_ids)

    def _update_curriculum(self):
        """평균 추종 오차가 임계 이하이면 명령 박스를 확장한다 (P1 command-space curriculum)."""
        cmd = self.cfg.command
        if float(self._curr_err_count.item()) < cmd.curriculum_min_episodes:
            return
        mean_err = float((self._curr_err_sum / self._curr_err_count).item())
        self._curr_err_sum.zero_()
        self._curr_err_count.zero_()
        self._curr_eval_count += 1.0
        if mean_err < cmd.curriculum_err_threshold:
            step = torch.tensor(cmd.box_step, device=self.device)
            box_max = torch.tensor(cmd.box_max, device=self.device)
            self._cmd_box = torch.minimum(self._cmd_box + step, box_max)
            self._curr_promote_count += 1.0

    # ──────────────────────────────────────────────────────────
    # Domain Randomization (go2_imitation_tracking 구조 승계)
    # ──────────────────────────────────────────────────────────

    def _init_domain_rand(self):
        device = self.device
        names = list(self._robot.data.joint_names)
        n_j = self._robot.num_joints

        self._pace_armature = torch.full((n_j,), 0.01, device=device)
        self._pace_viscous = torch.zeros(n_j, device=device)
        self._pace_coulomb = torch.zeros(n_j, device=device)
        for key in ("hip", "thigh", "calf"):
            ids = [i for i, n in enumerate(names) if n.endswith(f"_{key}_joint")]
            self._pace_armature[ids] = PACE_ARMATURE[key]
            self._pace_viscous[ids] = PACE_VISCOUS[key]
            self._pace_coulomb[ids] = PACE_COULOMB[key]
        self._all_joint_ids = torch.arange(n_j, dtype=torch.int32, device=device)

        self._act = self._robot.actuators["base_legs"]
        self._base_kp = self._act.stiffness.clone()
        self._base_kd = self._act.damping.clone()

        self._encoder_bias = torch.zeros(self.num_envs, 12, device=device)
        self._action_delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        d_cap = max(1, int(self.cfg.dr.max_action_delay_steps))
        self._action_delay_buf = self._robot.data.default_joint_pos.unsqueeze(1).repeat(1, d_cap + 1, 1).clone()
        self._push_timer = torch.full((self.num_envs,), self.cfg.dr.push_interval_s, device=device)
        self._prev_joint_vel = torch.zeros(self.num_envs, n_j, device=device)

        _bid = self._base_body_id
        self._default_base_mass = self._robot.data.body_mass.torch[:, _bid : _bid + 1].clone()

        self._priv_armature_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_joint_friction_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_base_mass_offset = torch.zeros(self.num_envs, 1, device=device)
        self._priv_foot_friction_offset = torch.zeros(self.num_envs, 1, device=device)
        self._priv_kp_scale = torch.ones(self.num_envs, 1, device=device)
        self._priv_kd_scale = torch.ones(self.num_envs, 1, device=device)
        self._died = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

        self._dr_ready = False

    def _sample(self, rng: tuple[float, float], n: int, d: int) -> torch.Tensor:
        lo, hi = rng
        return torch.rand(n, d, device=self.device) * (hi - lo) + lo

    def _apply_domain_rand(self, env_ids: torch.Tensor):
        env_ids_long = env_ids.to(torch.long)
        env_ids_int = env_ids.to(torch.int32)
        n = int(env_ids_long.numel())
        dr = self.cfg.dr

        if self.cfg.use_pace_params:
            arm_s = (
                self._sample(dr.armature_scale_range, n, 1) if (self.cfg.domain_rand and dr.randomize_armature) else 1.0
            )
            jf_s = (
                self._sample(dr.joint_friction_scale_range, n, 1)
                if (self.cfg.domain_rand and dr.randomize_joint_friction)
                else 1.0
            )
            armature = self._pace_armature.unsqueeze(0).repeat(n, 1) * arm_s
            viscous = self._pace_viscous.unsqueeze(0).repeat(n, 1) * jf_s
            coulomb = self._pace_coulomb.unsqueeze(0).repeat(n, 1) * jf_s
            self._robot.write_joint_armature_to_sim_index(
                armature=armature, joint_ids=self._all_joint_ids, env_ids=env_ids_int
            )
            self._robot.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=coulomb,
                joint_dynamic_friction_coeff=coulomb,
                joint_viscous_friction_coeff=viscous,
                joint_ids=self._all_joint_ids,
                env_ids=env_ids_int,
            )
            if self.cfg.domain_rand and dr.randomize_armature:
                self._priv_armature_scale[env_ids_long] = arm_s
            if self.cfg.domain_rand and dr.randomize_joint_friction:
                self._priv_joint_friction_scale[env_ids_long] = jf_s

        if not self.cfg.domain_rand:
            return

        if not self._dr_ready:
            self._randomize_material_startup()
            self._dr_ready = True

        if dr.randomize_mass:
            add = self._sample(dr.added_base_mass_range, n, 1)
            new_mass = torch.clamp(self._default_base_mass[env_ids_long] + add, min=1e-6)
            body_ids = torch.tensor([self._base_body_id], dtype=torch.int32, device=self.device)
            self._robot.set_masses_index(masses=new_mass, body_ids=body_ids, env_ids=env_ids_int)
            self._priv_base_mass_offset[env_ids_long] = new_mass - self._default_base_mass[env_ids_long]

        if dr.randomize_gains:
            kp_scale = self._sample(dr.kp_scale_range, n, 1)
            kd_scale = self._sample(dr.kd_scale_range, n, 1)
            self._act.stiffness[env_ids_long] = self._base_kp[env_ids_long] * kp_scale
            self._act.damping[env_ids_long] = self._base_kd[env_ids_long] * kd_scale
            self._priv_kp_scale[env_ids_long] = kp_scale
            self._priv_kd_scale[env_ids_long] = kd_scale

        if dr.encoder_bias:
            rand_bias = torch.rand(n, 12, device=self.device) * 2.0 - 1.0
            self._encoder_bias[env_ids_long] = rand_bias * PACE_ENCODER_BIAS_MAG
        if dr.randomize_action_delay:
            self._action_delay_steps[env_ids_long] = torch.randint(
                0, dr.max_action_delay_steps + 1, (n,), device=self.device
            )
            self._action_delay_buf[env_ids_long] = self._robot.data.default_joint_pos[env_ids_long].unsqueeze(1)
        self._push_timer[env_ids_long] = dr.push_interval_s

    def _randomize_material_startup(self):
        dr = self.cfg.dr
        materials = wp.to_torch(self._robot.root_view.get_material_properties())
        lo, hi = dr.foot_friction_range
        fr = torch.rand(materials.shape[0], 1, device="cpu") * (hi - lo) + lo
        materials[..., 0] = fr
        materials[..., 1] = fr
        env_ids = torch.arange(materials.shape[0], device="cpu", dtype=torch.int32)
        self._robot.root_view.set_material_properties(
            wp.from_torch(materials.contiguous(), dtype=wp.float32), wp.from_torch(env_ids, dtype=wp.int32)
        )
        self._priv_foot_friction_offset[:, 0] = fr.to(self.device).squeeze(-1) - 1.0

    def _push_robots(self, env_ids: torch.Tensor):
        """base 에 랜덤 속도 킥 — P1 의 disturbance curriculum (선/각속도 동시)."""
        n = int(env_ids.numel())
        lin = self._robot.data.root_lin_vel_w[env_ids].clone()
        ang = self._robot.data.root_ang_vel_w[env_ids].clone()
        lin[:, 0:2] += (torch.rand(n, 2, device=self.device) * 2.0 - 1.0) * self.cfg.dr.max_push_vel_xy
        ang += (torch.rand(n, 3, device=self.device) * 2.0 - 1.0) * self.cfg.dr.max_push_ang_vel
        vel = torch.cat([lin, ang], dim=-1)
        self._robot.write_root_com_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids.to(torch.int32))

    def _apply_obs_dr(self, obs: torch.Tensor) -> torch.Tensor:
        """policy obs 에만 관측 노이즈 + encoder bias 를 더한다.

        레이아웃: projected_gravity[0:3] + (joint_pos-default)[3:15] + joint_vel[15:27] +
                  prev_actions[27:55] + foot_pos_b[55:67] + leg_role[67:71] + foot_err_b[71:83]
        명령 블록(leg_role)에는 노이즈를 싣지 않는다.
        """
        if not self.cfg.domain_rand:
            return obs
        dr = self.cfg.dr
        noise = torch.zeros_like(obs)
        n = self.num_envs
        if dr.obs_noise:
            noise[:, 0:3] = torch.randn(n, 3, device=self.device) * dr.gravity_noise
            noise[:, 3:15] = torch.randn(n, 12, device=self.device) * dr.joint_pos_noise
            noise[:, 15:27] = torch.randn(n, 12, device=self.device) * dr.joint_vel_noise
            # 발 위치·추종 오차는 FK 로 유도되므로 관절 노이즈와 같은 계열의 오차를 싣는다.
            noise[:, 55:67] = torch.randn(n, 12, device=self.device) * dr.foot_pos_noise
            noise[:, 71:83] = torch.randn(n, 12, device=self.device) * dr.foot_pos_noise
        if dr.encoder_bias:
            noise[:, 3:15] += self._encoder_bias
        return obs + noise

    def _get_priv_latent(self) -> torch.Tensor:
        return torch.cat(
            [
                self._priv_armature_scale,  # 1
                self._priv_joint_friction_scale,  # 1
                self._priv_base_mass_offset,  # 1
                self._priv_foot_friction_offset,  # 1
                self._priv_kp_scale,  # 1
                self._priv_kd_scale,  # 1
                (self._action_delay_steps.float() / max(1, self.cfg.dr.max_action_delay_steps)).unsqueeze(-1),  # 1
                self._encoder_bias / PACE_ENCODER_BIAS_MAG,  # 12
            ],
            dim=-1,
        )  # (N, 19)

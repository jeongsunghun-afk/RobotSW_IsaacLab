# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Fall-Recovery environment — M2: full recovery reward + Tier-0 safety + success judgment.

M1 (landed):
  - DirectRLEnv scaffold with 42-dim proprioceptive obs (no lin_vel, no commands)
  - Fall initialization: 80% fallen (random euler + random joint), 10% standing, 10% sitting
  - DebugViewer 3-line integration

M2 (this file):
  - reward_reset: r_roll (upright alignment) + r_stand (height + pose + vel)
  - Smoothness / regularization penalties: action_rate_l2, action_smoothness_2,
    dof_acc_l2, dof_vel_l2, dof_torques_l2, dof_pos_limits
  - Tier-0 safety: action clip ±cfg.action_clip, hip action scale 0.5×
  - Success explicit judgment: upright & near-default & low-vel for N consecutive steps
  - Success bonus reward + early termination on success

Deferred to M3:
  - Progressive penalty curriculum (weight schedule by training step)
  - Safety KPI logging (peak torque, jerk, recovery time)
"""

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz

from isaaclab_tasks.direct._common import DebugViewer

from .go2_recovery_env_cfg import Go2RecoveryEnvCfg


class Go2RecoveryEnv(DirectRLEnv):
    """Go2 fall-recovery environment (M2).

    Observation (42-dim, proprioceptive only — no lin_vel for sim-to-real):
        root_ang_vel_b        [3]
        projected_gravity_b   [3]
        joint_pos_error       [12]  (joint_pos - default_joint_pos)
        joint_vel             [12]
        previous_actions      [12]
        ─────────────────────
        TOTAL                 42

    Fall init (§5 확정: 임의 자세 전체 커버, 참조 비율 80/10/10):
        standing  10%  — default pose, z=0.27
        sitting   10%  — slightly bent joints, lower z
        fallen    80%  — random roll/pitch/yaw, random joints, fall_height 공중낙하

    Joint indexing: name-based (computed once in __init__ from joint_names).
    hip/thigh/calf ids are stored as long tensors on device.
    """

    cfg: Go2RecoveryEnvCfg

    def __init__(self, cfg: Go2RecoveryEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        num_actions = gym.spaces.flatdim(self.single_action_space)

        # ── name-based joint index buffers (computed once, order-independent) ─
        all_joint_names = self._robot.data.joint_names
        self._hip_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if "hip" in n],
            dtype=torch.long,
            device=self.device,
        )
        self._thigh_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if "thigh" in n],
            dtype=torch.long,
            device=self.device,
        )
        self._calf_joint_ids = torch.tensor(
            [i for i, n in enumerate(all_joint_names) if "calf" in n],
            dtype=torch.long,
            device=self.device,
        )

        # ── joint weights tensor (r_pose 계산용) ─────────────────────────────
        # hip=1.0, thigh=0.75, calf=0.5 — 원위부로 갈수록 가중치 감소
        # 이름 기반으로 할당: 순서 무관하게 항상 올바른 weight 매핑
        num_joints = len(all_joint_names)
        _weights = torch.zeros(num_joints, dtype=torch.float, device=self.device)
        _weights[self._hip_joint_ids] = 1.0
        _weights[self._thigh_joint_ids] = 0.75
        _weights[self._calf_joint_ids] = 0.5
        self._joint_weights = _weights.unsqueeze(0)  # (1, num_joints) → broadcast over (num_envs, num_joints)

        # ── action buffers ────────────────────────────────────────────────────
        self._actions = torch.zeros(self.num_envs, num_actions, device=self.device)
        self._previous_actions = torch.zeros(self.num_envs, num_actions, device=self.device)
        # 2차 smoothness용 t-2 action buffer
        self._last_last_actions = torch.zeros(self.num_envs, num_actions, device=self.device)

        # dof_acc 계산용: 이전 step의 joint_vel 저장
        self._previous_joint_vel = torch.zeros(self.num_envs, num_actions, device=self.device)

        # action_smoothness_1용: 이전 step의 joint_pos_target (processed_actions) 저장
        # 참조(legged_env_recovery.py:1286): last_joint_pos_target
        # last_actions != 0 마스크로 첫 step을 무시하는 로직을 위해 _previous_actions와 함께 사용
        self._last_joint_pos_target = torch.zeros(self.num_envs, num_actions, device=self.device)

        # delta_torques용: 이전 step의 applied_torque 저장
        # 참조(legged_env_recovery.py:1155): last_torques
        self._last_torque = torch.zeros(self.num_envs, num_actions, device=self.device)

        # ── success judgment buffer ───────────────────────────────────────────
        # 연속 성공 조건 유지 step 카운터 (초기화: _reset_idx)
        self._success_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── started_fallen 버퍼: reset 시 "fallen 그룹에서 시작했는가" bool ─────
        # standing/sitting으로 시작한 env의 success를 자동성공으로 무효화하는 데 사용.
        # True = fallen 그룹에서 시작 → success 유효
        # False = standing/sitting → success 무효 (survivorship bias 제거)
        self._started_fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # fallen-group별 success 분리 로깅용: fallen 그룹 success 누적
        # _reset_idx에서 fallen-only success_rate를 extras에 기록할 때 사용
        self._current_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # ── settle phase buffers ──────────────────────────────────────────────
        # _settle_steps[i]: env i의 settle step 수 (randint(0, settle_max_steps+1))
        # _settle_counter[i]: env i의 현재 settle 경과 step 수
        # settle_counter < settle_steps인 동안: action override / reward=0 / done=False
        self._settle_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._settle_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # ── episode reward logging buffers ────────────────────────────────────
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "reward_reset",
                "r_roll_progress",
                "action_rate_l2",
                "action_smoothness_1",
                "action_smoothness_2",
                "delta_torques",
                "dof_acc_l2",
                "dof_vel_l2",
                "dof_torques_l2",
                "dof_pos_limits",
                "success_bonus",
                "r_success_region",  # Lever A: 연속 in-region 보상 (Episode_Reward/r_success_region)
            ]
        }

        # ── r_roll_progress용 이전 cos_dist 버퍼 ──────────────────────────────
        # potential-based shaping: Δcos_dist 계산에 사용 (초기화: _reset_idx)
        self._prev_cos_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # ── sitting 목표 자세 텐서 (Genesis 원본값, __init__에서 1회 구성) ────────
        # Genesis default_sitting_joint_angles: hip FL/RL=+0.1, FR/RR=-0.1 (=IsaacLab default),
        #   thigh all=1.57 rad, calf all=-2.67 rad.
        # 이름 기반 인덱싱 — self._thigh_joint_ids / self._calf_joint_ids 재사용.
        # 상수 텐서이므로 _reset_idx 재초기화 불필요.
        self._sitting_joint_pos = self._robot.data.default_joint_pos[0].clone()  # (num_joints,)
        self._sitting_joint_pos[self._thigh_joint_ids] = 1.57  # Genesis sitting thigh
        self._sitting_joint_pos[self._calf_joint_ids] = -2.67  # Genesis sitting calf
        # Note: hip stays at default (FL/RL=+0.1, FR/RR=-0.1) — matches Genesis exactly.
        # Note: calf -2.67 is near Go2 hard lower limit (~-2.72); soft-limit clamp in
        #   _reset_sitting may raise it slightly. That value is still within hard limits
        #   so pose is physically valid.

        # ── contact sensor body indices ───────────────────────────────────────
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")

        # ── DebugViewer (필수 3줄 중 1번째) ──────────────────────────────────
        self._debug_viewer = DebugViewer(self, cfg=self.cfg.debug_viewer)

    # ── scene setup ───────────────────────────────────────────────────────────

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # ── action pipeline ───────────────────────────────────────────────────────

    def _pre_physics_step(self, actions: torch.Tensor):
        # Tier-0: raw action clip ±action_clip (cfg 노출)
        clipped = actions.clamp(-self.cfg.action_clip, self.cfg.action_clip)

        # Tier-0: hip joint action scale 0.5× (name-based self._hip_joint_ids)
        scaled = clipped.clone()
        scaled[:, self._hip_joint_ids] = scaled[:, self._hip_joint_ids] * self.cfg.hip_action_scale

        self._actions = clipped  # 로깅/smoothness 계산은 clip 전 원본 기준이 일반적이나
        # 참조(FR-Net)는 joint_pos_target 차이로 smoothness 계산 → clip 후 scaled 사용
        # 여기서는 policy가 출력한 clipped action을 기준으로 smoothness 측정
        self._processed_actions = self.cfg.action_scale * scaled + self._robot.data.default_joint_pos

        # settle override: settle-active env의 sim target을 mode에 따라 강제
        # policy action(_actions, _processed_actions) 자체는 보존 — sim에 반영되는 target만 override
        #   "hold"   : default_joint_pos → PD가 default 자세로 유인하며 안착
        #   "passive": 현재 측정 joint_pos → stiffness 오차≈0, 중력으로 자연스럽게 흩어진 채 안착
        if self.cfg.settle_max_steps > 0:
            settle_active = self._settle_counter < self._settle_steps  # (num_envs,) bool
            if settle_active.any():
                if self.cfg.settle_mode == "passive":
                    # 현재 측정 위치를 target으로 → stiffness 오차≈0, 관절이 default로 이주하지 않고
                    # 중력으로 자연 안착 (Genesis reset 방식에 가까움)
                    settle_target = self._robot.data.joint_pos
                else:  # "hold" (legacy — 기존 default-hold 동작과 byte-identical)
                    settle_target = self._robot.data.default_joint_pos
                self._processed_actions = torch.where(
                    settle_active.unsqueeze(1),  # (num_envs, 1) → broadcast over (num_envs, 12)
                    settle_target,
                    self._processed_actions,
                )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # ── observation ───────────────────────────────────────────────────────────

    def _get_observations(self) -> dict:
        # 2차 smoothness buffer: t-2 갱신 (t-1 → t-2)
        self._last_last_actions = self._previous_actions.clone()
        # t-1 갱신 (current → previous)
        self._previous_actions = self._actions.clone()

        # fmt: off
        obs = torch.cat(
            [
                self._robot.data.root_ang_vel_b * self.cfg.ang_vel_scale,        # [3] body-frame angular velocity
                self._robot.data.projected_gravity_b,                              # [3] gravity vector in body frame
                (self._robot.data.joint_pos - self._robot.data.default_joint_pos)
                * self.cfg.dof_pos_scale,                                          # [12] joint position error
                self._robot.data.joint_vel * self.cfg.dof_vel_scale,              # [12] joint velocity
                self._previous_actions,                                             # [12] previous actions
            ],
            dim=-1,
        )
        # fmt: on
        # obs 차원 확인: 3+3+12+12+12 = 42 (cfg.observation_space=42 불변)

        # DebugViewer update (필수 3줄 중 2번째)
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
            self._debug_viewer.update(self.step_dt)

        # settle counter 증가: _get_observations는 DirectRLEnv step 순서상
        # dones→rewards→reset→observations 중 마지막에 호출되므로,
        # 이 step의 reward=0/done=False 처리가 완료된 후 counter를 증가시킨다.
        # 따라서 counter 증가는 현재 step의 settle 마스킹에 영향을 주지 않으며,
        # 다음 step부터 새 counter 값이 반영된다.
        if self.cfg.settle_max_steps > 0:
            settle_active = self._settle_counter < self._settle_steps
            self._settle_counter = torch.where(
                settle_active,
                self._settle_counter + 1,
                self._settle_counter,
            )

        return {"policy": obs}

    # ── rewards ───────────────────────────────────────────────────────────────

    def _get_rewards(self) -> torch.Tensor:
        # ── cos_dist: upright 정렬도 ──────────────────────────────────────────
        # 참조(Genesis) 방식: world up=[0,0,1]을 root_quat(body→world, wxyz)으로
        # 회전시켜 body z축의 world 방향 벡터 root_up을 구하고,
        # world up과의 dot product로 정렬도를 계산.
        # cos_dist = dot([0,0,1], R·[0,0,1]) = R[2,2]
        # 직립=+1, 수평=0, 뒤집힘=-1. 범위 [-1, 1].
        # ※ 수학적 등가: -projected_gravity_b[:,2] = R[2,2] (동일 결과)
        #   그러나 참조 구현 명시성을 위해 quaternion 방식으로 표현.
        _up_world = torch.zeros(self.num_envs, 3, device=self.device)
        _up_world[:, 2] = 1.0  # world up vector [0, 0, 1]
        root_up = quat_apply(self._robot.data.root_quat_w, _up_world)  # body z축 → world frame
        cos_dist = (root_up * _up_world).sum(dim=-1)  # dot with [0,0,1] = root_up[:,2]

        # ── r_roll: upright 정렬 보상 (항상 [0,1]) ────────────────────────────
        # (0.5·cos_dist + 0.5)²: cos_dist=-1→0, cos_dist=0→0.25, cos_dist=+1→1
        r_roll = (0.5 * cos_dist + 0.5) ** 2

        # ── r_roll_progress: potential-based shaping (Δcos_dist × weight) ────
        # fallen→upright 진행 방향에 즉각 양의 신호를 주어 "정지=최적" local optimum 탈출.
        # cos_dist_prev는 이전 step의 cos_dist (buffer로 관리).
        # 부호 규칙: 진행(cos_dist 증가) → 양수, 후퇴 → 음수 (shaping 허용).
        # 발산 방지: clamp(-1, +1) 범위 내에서만 작동 (이미 cos_dist가 [-1,1]).
        delta_cos = cos_dist - self._prev_cos_dist  # (num_envs,)
        r_roll_progress = delta_cos * self.cfg.roll_progress_weight  # weight > 0 → 진행 보상
        self._prev_cos_dist = cos_dist.clone()

        # ── r_stand: 충분히 세워진 경우에만 (cos_dist > threshold) ───────────
        r_stand = self._calc_r_stand()
        stand_active = cos_dist > self.cfg.stand_cos_threshold  # (num_envs,) bool
        r_stand = torch.where(stand_active, r_stand, torch.zeros_like(r_stand))

        # ── reward_reset = roll_w·r_roll + stand_w·r_stand ───────────────────
        # 정량 근거 (step_dt=0.02):
        #   roll_reward_weight=2.0, reward_reset_scale=2.0 기준
        #   r_roll=0.25 구간: 2.0×0.25×2.0×0.02 = 0.020/step → smoothness penalty 상회
        #   r_roll=1.0 구간: 2.0×1.0×2.0×0.02 = 0.080/step → success_bonus(+5×0.02=0.10)와 균형
        # r_roll_progress는 별도 항으로 rewards dict에 추가됨 (step_dt 1회 적용 보장)
        reward_reset = (
            self.cfg.roll_reward_weight * r_roll + self.cfg.stand_reward_weight * r_stand
        ) * self.cfg.reward_reset_scale

        # ── smoothness / regularization penalties ─────────────────────────────
        # 1차 action 변화율 (raw action 기반)
        action_rate_l2 = (
            torch.sum(torch.square(self._actions - self._previous_actions), dim=1) * self.cfg.action_rate_l2_scale
        )

        # joint_pos_target 1차 차분 (target 기반 smoothness)
        # 참조(legged_env_recovery.py:1284-1288): sum((target_t - target_{t-1})²)
        # last_actions != 0 마스크: 첫 step(이전 target이 zeros)에서의 허위 penalty 제거
        first_step_mask = (self._previous_actions != 0).float()  # (num_envs, num_actions)
        action_smoothness_1 = (
            torch.sum(
                torch.square(self._processed_actions - self._last_joint_pos_target) * first_step_mask,
                dim=1,
            )
            * self.cfg.action_smoothness_1_scale
        )

        # 2차 action 변화율
        action_smoothness_2 = (
            torch.sum(
                torch.square(self._actions - 2.0 * self._previous_actions + self._last_last_actions),
                dim=1,
            )
            * self.cfg.action_smoothness_2_scale
        )

        # applied torque 변화율 제곱합
        # 참조(legged_env_recovery.py:1154-1155): sum((torque_t - torque_{t-1})²)
        current_torque = self._robot.data.applied_torque  # (num_envs, 12)
        delta_torques = (
            torch.sum(torch.square(current_torque - self._last_torque), dim=1) * self.cfg.delta_torques_scale
        )

        # joint 가속도 (Δjoint_vel / dt)²
        joint_vel = self._robot.data.joint_vel  # (num_envs, 12)
        dof_acc = (joint_vel - self._previous_joint_vel) / self.step_dt
        dof_acc_l2 = torch.sum(torch.square(dof_acc), dim=1) * self.cfg.dof_acc_l2_scale

        # joint 속도 제곱합
        dof_vel_l2 = torch.sum(torch.square(joint_vel), dim=1) * self.cfg.dof_vel_l2_scale

        # applied torque 제곱합
        dof_torques_l2 = torch.sum(torch.square(current_torque), dim=1) * self.cfg.dof_torques_l2_scale

        # joint position soft limit 위반 패널티
        joint_pos = self._robot.data.joint_pos  # (num_envs, 12)
        soft_lower = self._robot.data.soft_joint_pos_limits[:, :, 0]
        soft_upper = self._robot.data.soft_joint_pos_limits[:, :, 1]
        out_of_limits = (-(joint_pos - soft_lower)).clamp(min=0.0)  # lower 위반
        out_of_limits += (joint_pos - soft_upper).clamp(min=0.0)  # upper 위반
        dof_pos_limits = torch.sum(out_of_limits, dim=1) * self.cfg.dof_pos_limits_scale

        # ── buffer 갱신: 다음 step 계산에 사용할 이전 값 저장 ─────────────────
        self._previous_joint_vel = joint_vel.clone()
        self._last_joint_pos_target = self._processed_actions.clone()
        self._last_torque = current_torque.clone()

        # ── success judgment ──────────────────────────────────────────────────
        success_bonus = self._update_success(cos_dist)
        # Lever A: 연속 in-region 보상 — _update_success 호출 후 self._success_region_mask가 최신.
        # 순간 3조건(upright∧near-default∧low-vel) 동시충족 시 매 step 양의 보상.
        # 실효 ≈ 3.0×0.02=0.06/step → crouch r_stand(≈0.0196/step)를 ~3배 상회.
        # 20-step hold 불필요(dense gradient용). 기존 success_bonus(1회성)는 불변.
        r_success_region = self._success_region_mask.float() * self.cfg.success_region_reward_scale

        # ── 전체 reward 합산 ──────────────────────────────────────────────────
        rewards = {
            "reward_reset": reward_reset,
            "r_roll_progress": r_roll_progress,  # progress shaping (별도 항, step_dt 1회 곱)
            "action_rate_l2": action_rate_l2,
            "action_smoothness_1": action_smoothness_1,  # target 기반 1차 차분 (참조 추가)
            "action_smoothness_2": action_smoothness_2,
            "delta_torques": delta_torques,  # torque 변화율 (참조 추가)
            "dof_acc_l2": dof_acc_l2,
            "dof_vel_l2": dof_vel_l2,
            "dof_torques_l2": dof_torques_l2,
            "dof_pos_limits": dof_pos_limits,
            "success_bonus": success_bonus,
            "r_success_region": r_success_region,  # Lever A: in-region 연속 보상
        }

        # step_dt 곱: per-step raw → episode 스케일 정규화
        # 주의: Episode_Reward/* 로그는 step_dt 이미 반영된 값 → weight 산정 시 재곱 금지
        reward = torch.zeros(self.num_envs, device=self.device)
        for key, value in rewards.items():
            weighted = value * self.step_dt
            reward += weighted
            self._episode_sums[key] += weighted

        # settle 마스킹: settle-active env의 reward를 0으로 강제
        # settle 중 낙하/동작 transient에 의한 penalty/bonus가 학습 신호를 오염하지 않게 함
        # 주의: rsl_rl buffer에 reward=0 transition이 기록됨 (약간 비효율이나 학습에 무해)
        if self.cfg.settle_max_steps > 0:
            settle_active = self._settle_counter < self._settle_steps
            reward = torch.where(settle_active, torch.zeros_like(reward), reward)

        return reward

    def _calc_r_stand(self) -> torch.Tensor:
        """r_stand = 0.2·r_height + 1.0·r_pose + 0.2·r_vel.

        참조(Genesis legged_env_recovery.py:1273) 기반, pose 실효가중 상향:
          stand_height_weight=0.2, stand_pose_weight=1.0, stand_vel_weight=0.2
          (합=1.4, 재정규화 없음 — 실효계수 0.5×1.0=0.5로 0.3에서 상향이 목적)

        r_height 양방향 대칭: |h_root - target| 기반 → target 초과도 페널티.
        cos_dist 임계값 필터링은 호출자(_get_rewards)에서 수행.
        """
        # r_height: target_height 기준 양방향 대칭 페널티
        # 이전: clamp(tar_h - root_h, 0, tar_h) → h_root >= target이면 saturate(=1 만점)
        # 변경: abs(h_root - target) → 위/아래 모두 target에서 멀어지면 감소
        # target에서 정확히 r_height=1, 위아래 target_height 벗어날수록 0 방향으로 감소
        tar_h = self.cfg.target_height
        root_h = self._robot.data.root_link_pos_w[:, 2]
        h_err = ((root_h - tar_h).abs() / tar_h).clamp(0.0, 1.0)
        r_height = 1.0 - h_err

        # r_pose: default joint pos와의 가중 편차 (exp 형태)
        # joint_weights: [hip 1.0, thigh 0.75, calf 0.5] × 4 → (1, 12)
        pose_diff = self._robot.data.default_joint_pos - self._robot.data.joint_pos
        pose_err = torch.sum(self._joint_weights**2 * pose_diff**2, dim=1)
        r_pose = torch.exp(-self.cfg.pose_exp_scale * pose_err)

        # r_vel: joint velocity 크기 억제 (exp 형태)
        vel_err = torch.sum(self._robot.data.joint_vel**2, dim=1)
        r_vel = torch.exp(-self.cfg.vel_exp_scale * vel_err)

        return (
            self.cfg.stand_height_weight * r_height
            + self.cfg.stand_pose_weight * r_pose
            + self.cfg.stand_vel_weight * r_vel
        )

    def _update_success(self, cos_dist: torch.Tensor) -> torch.Tensor:
        """Success 조건 판정 및 연속 카운터 갱신. 성공 시 bonus reward 반환.

        Success 조건 (AND):
          1. upright: cos_dist > success_cos_threshold
          2. near-default-pose: weighted pose_err < success_pose_eps
          3. low joint vel: sqrt(mean(vel²)) < success_vel_eps

        연속 success_hold_steps step 유지 시 success 판정.

        자동성공 제거: _started_fallen=False(standing/sitting 시작)인 env는
        success 판정에서 제외 → survivorship bias 완전 제거.
        """
        # 조건 1: upright
        cond_upright = cos_dist > self.cfg.success_cos_threshold

        # 조건 2: near-default-pose
        pose_diff = self._robot.data.default_joint_pos - self._robot.data.joint_pos
        pose_err = torch.sum(self._joint_weights**2 * pose_diff**2, dim=1)
        cond_pose = pose_err < (self.cfg.success_pose_eps**2)

        # 조건 3: low joint vel (RMS)
        vel_rms = torch.sqrt(torch.mean(self._robot.data.joint_vel**2, dim=1))
        cond_vel = vel_rms < self.cfg.success_vel_eps

        all_conditions = cond_upright & cond_pose & cond_vel

        # settle 마스킹: settle-active env는 success 조건을 False로 강제
        # 낙하 transient 중 우연히 조건을 만족하더라도 success counter가 누적되지 않게 함
        if self.cfg.settle_max_steps > 0:
            settle_active = self._settle_counter < self._settle_steps
            all_conditions = torch.where(settle_active, torch.zeros_like(all_conditions), all_conditions)

        # Lever A: 연속 in-region 보상용 순간 mask 저장.
        # 캡처 시점: settle 마스킹 적용 후, _started_fallen 게이트 적용 전.
        # - settle 적용: settle 중 transient 발화로 인한 로그 오염 방지.
        # - _started_fallen 미적용: 3조건 기반 reward shaping이며, 게이트는
        #   success 판정 지표용(survivorship bias 제거)이지 reward 조건이 아님.
        #   standing 시작 env가 즉시 발화 → smoke에서 r_success_region > 0 확인 가능.
        self._success_region_mask = all_conditions.clone()

        # 자동성공 게이트: fallen 그룹에서 시작한 env만 success 유효
        # standing/sitting 시작 env는 복구 동작 없이도 success 조건 충족 → 무효화
        all_conditions = all_conditions & self._started_fallen

        # 연속 카운터 갱신: 조건 만족 시 +1, 아니면 0 리셋
        self._success_counter = torch.where(
            all_conditions,
            self._success_counter + 1,
            torch.zeros_like(self._success_counter),
        )

        # success 판정: 연속 N step 유지
        just_succeeded = self._success_counter >= self.cfg.success_hold_steps

        # success bonus: 1회성 (카운터가 정확히 threshold에 도달한 step만)
        bonus = torch.where(
            self._success_counter == self.cfg.success_hold_steps,
            torch.full_like(cos_dist, self.cfg.success_reward_scale),
            torch.zeros_like(cos_dist),
        )

        # success_rate 로깅: extras에 기록 (_reset_idx에서 읽힘)
        # just_succeeded는 이미 _started_fallen 게이트 적용됨 (자동성공 제외)
        self._current_success = just_succeeded

        return bonus

    # ── termination ───────────────────────────────────────────────────────────

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # time_out: 에피소드 길이 초과
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # terminated: 파국적 상황 (지면 아래로 꺼짐)
        base_z = self._robot.data.root_link_pos_w[:, 2]
        terminated = base_z < -0.1

        # success early termination (cfg toggle)
        if self.cfg.terminate_on_success and hasattr(self, "_current_success"):
            terminated = terminated | self._current_success

        # settle 마스킹: settle-active env는 terminated=False 강제
        # 낙하 transient(base_z 일시적 음수, success 미달)로 인한 조기 reset 방지
        # time_out은 마스킹 안 함 — episode_length는 settle 중에도 그대로 진행
        if self.cfg.settle_max_steps > 0:
            settle_active = self._settle_counter < self._settle_steps
            terminated = torch.where(settle_active, torch.zeros_like(terminated), terminated)

        return terminated, time_out

    # ── reset ─────────────────────────────────────────────────────────────────

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        # Pyright narrowing: 가드 이후 env_ids는 반드시 torch.Tensor
        ids: torch.Tensor = env_ids

        self._robot.reset(ids)
        super()._reset_idx(ids)

        # episode_length_buf 랜덤 초기화 (스파이크 방지 — go2 베이스 패턴)
        if len(ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # ── Episode logging (buffer 초기화 전에 먼저 수행) ───────────────────
        # _current_success, _started_fallen은 직전 에피소드 값 → 초기화 전 참조해야 정확
        extras: dict = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][ids] = 0.0

        # success rate 로깅 (직전 에피소드 기준)
        # _started_fallen: 직전 에피소드 시작 시 fallen 그룹 여부 (buffer 초기화 전)
        prev_fallen_mask = self._started_fallen[ids]  # (len(ids),) bool
        prev_non_fallen_mask = ~prev_fallen_mask

        extras["Episode/success_rate"] = self._current_success[ids].float().mean().item()

        # fallen-only success율: 직전 에피소드에서 fallen으로 시작한 env만 집계
        fallen_success_ids = ids[prev_fallen_mask]
        if len(fallen_success_ids) > 0:
            extras["Episode/success_rate_fallen"] = self._current_success[fallen_success_ids].float().mean().item()
        else:
            extras["Episode/success_rate_fallen"] = 0.0

        # non-fallen success율: standing/sitting 시작 env (_started_fallen 게이트로 항상 0, sanity check용)
        non_fallen_success_ids = ids[prev_non_fallen_mask]
        if len(non_fallen_success_ids) > 0:
            extras["Episode/success_rate_non_fallen"] = (
                self._current_success[non_fallen_success_ids].float().mean().item()
            )

        # ── buffer 초기화 ─────────────────────────────────────────────────────
        self._actions[ids] = 0.0
        self._previous_actions[ids] = 0.0
        self._last_last_actions[ids] = 0.0
        self._previous_joint_vel[ids] = 0.0
        self._last_joint_pos_target[ids] = 0.0  # action_smoothness_1용 (불변규칙: 신규 buffer → reset 초기화)
        self._last_torque[ids] = 0.0  # delta_torques용 (불변규칙)
        self._success_counter[ids] = 0
        # _started_fallen: 이 reset에서는 아직 그룹 분류 전이므로 False로 초기화.
        # 실제 값은 _reset_fallen/_reset_standing/_reset_sitting 호출 직후 설정.
        self._started_fallen[ids] = False
        # _prev_cos_dist: episode 경계에서 0으로 초기화 (진행 progress shaping 누설 방지)
        self._prev_cos_dist[ids] = 0.0

        # settle phase 버퍼 초기화 (불변규칙: 신규 버퍼는 _reset_idx에서 반드시 초기화)
        if self.cfg.settle_max_steps > 0:
            self._settle_steps[ids] = torch.randint(0, self.cfg.settle_max_steps + 1, (len(ids),), device=self.device)
        else:
            self._settle_steps[ids] = 0
        self._settle_counter[ids] = 0

        # ── Fall initialization (80/10/10) ────────────────────────────────────
        n = len(ids)
        n_standing = max(1, int(self.cfg.fall_standing_ratio * n))
        n_sitting = max(1, int(self.cfg.fall_sitting_ratio * n))
        # n이 매우 작을 때 합이 n을 넘지 않도록 조정
        n_standing = min(n_standing, n)
        n_sitting = min(n_sitting, n - n_standing)

        # 셔플로 env_ids를 세 그룹으로 나눔
        perm = torch.randperm(n, device=self.device)
        standing_ids = ids[perm[:n_standing]]
        sitting_ids = ids[perm[n_standing : n_standing + n_sitting]]
        fallen_ids = ids[perm[n_standing + n_sitting :]]

        # settle은 fallen 그룹에만 적용 — standing/sitting은 이미 지면 근처에서 시작하므로
        # 안착 대기 불필요. Genesis(init_step=0) 기준: 어느 그룹도 settling 없음.
        # 우리 환경은 fallen 공중낙하(0.72m) 때문에 settle이 필요하지만,
        # standing/sitting에 settle이 걸리면 default_joint_pos로 관절이 끌려가는 버그 발생.
        if self.cfg.settle_max_steps > 0:
            self._settle_steps[standing_ids] = 0
            self._settle_steps[sitting_ids] = 0
            # fallen_ids: _reset_idx 상단의 randint에서 이미 0~settle_max_steps 부여됨

        # ── (1) STANDING: default pose ────────────────────────────────────────
        if len(standing_ids) > 0:
            self._reset_standing(standing_ids)
            # standing 시작 → success 무효 (자동성공 방지)
            self._started_fallen[standing_ids] = False

        # ── (2) SITTING: slightly bent + lower z ─────────────────────────────
        if len(sitting_ids) > 0:
            self._reset_sitting(sitting_ids)
            # sitting 시작 → success 무효
            self._started_fallen[sitting_ids] = False

        # ── (3) FALLEN: random orientation + random joints + aerial drop ──────
        if len(fallen_ids) > 0:
            self._reset_fallen(fallen_ids)
            # fallen 시작 → success 유효 (실제 복구를 한 경우만 카운트)
            self._started_fallen[fallen_ids] = True

        self.extras["log"] = {}
        self.extras["log"].update(extras)
        self.extras["log"]["Episode_Termination/base_fall"] = torch.count_nonzero(self.reset_terminated[ids]).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[ids]).item()

    # ── fall init helpers ─────────────────────────────────────────────────────

    def _reset_standing(self, env_ids: torch.Tensor):
        """Default pose reset (10% standing)."""
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        # velocity zero
        root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_sitting(self, env_ids: torch.Tensor):
        """Sitting pose reset (10% sitting): Genesis 원본 앉은 자세 사용.

        Genesis default_sitting_joint_angles (train_recovery.py:102-117):
          hip FL/RL=+0.1, FR/RR=-0.1 (=IsaacLab default, no change)
          thigh all = 1.57 rad (~90° folded)
          calf  all = -2.67 rad (deeply tucked)
        앉은 자세이므로 hip jitter 없음 (Genesis 원본과 동일).
        """
        n = len(env_ids)

        # Genesis sitting 자세 텐서 broadcast: (num_joints,) → (n, num_joints)
        joint_pos = self._sitting_joint_pos.unsqueeze(0).expand(n, -1).clone()
        # joint limits clamp (soft limit 적용 — calf -2.67은 hard limit -2.72 이내)
        joint_pos = joint_pos.clamp(
            self._robot.data.soft_joint_pos_limits[env_ids, :, 0],
            self._robot.data.soft_joint_pos_limits[env_ids, :, 1],
        )
        joint_vel = torch.zeros_like(joint_pos)

        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        # z를 낮춤 (sitting 자세)
        root_state[:, 2] -= self.cfg.sit_height_offset
        root_state[:, 2] = root_state[:, 2].clamp(min=0.05)  # 지면 이하 방지
        # yaw만 랜덤 (sitting은 orientation 대부분 upright)
        yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * 3.1416
        roll = torch.zeros(n, device=self.device)
        pitch = torch.zeros(n, device=self.device)
        quat = quat_from_euler_xyz(roll, pitch, yaw)  # (w, x, y, z)
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_fallen(self, env_ids: torch.Tensor):
        """Fallen pose reset (80%): random orientation + random joints + aerial drop.

        공중(fall_height)에서 낙하 → 첫 스텝들에서 자연스럽게 넘어진 상태로 안착.
        참조: legged_env_recovery.py:923-1001 (Genesis Go1).
        """
        n = len(env_ids)

        # ── random orientation: roll±135°, pitch±45°, yaw±180° ───────────────
        roll = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.fall_roll_range
        pitch = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.fall_pitch_range
        yaw = (torch.rand(n, device=self.device) * 2.0 - 1.0) * self.cfg.fall_yaw_range
        quat = quat_from_euler_xyz(roll, pitch, yaw)  # (w, x, y, z)

        # ── random joint positions: lerp between default ↔ limit ────────────
        # pose_lerp = rand^fall_lerp_exponent (cfg 노출 지수)
        # exponent=3.0 → 평균 α=0.25 (default 근방 밀도 ↑)
        # exponent=1.5 → 평균 α=0.40, P(α<0.1) 46%→22% (더 펼쳐진 자세에서 시작)
        alpha = torch.rand(n, 12, device=self.device) ** self.cfg.fall_lerp_exponent  # (n, 12)
        lower = self._robot.data.soft_joint_pos_limits[env_ids, :, 0]  # (n, 12)
        upper = self._robot.data.soft_joint_pos_limits[env_ids, :, 1]  # (n, 12)
        default_pos = self._robot.data.default_joint_pos[env_ids]  # (n, 12)
        # alpha=0 → default, alpha=1 → limit (방향은 랜덤 부호)
        sign = torch.randint(0, 2, (n, 12), device=self.device).float() * 2.0 - 1.0
        limit_target = torch.where(sign > 0, upper, lower)
        joint_pos = default_pos + alpha * (limit_target - default_pos)
        joint_pos = joint_pos.clamp(lower, upper)
        joint_vel = torch.zeros_like(joint_pos)

        # ── root state: 공중 + random orientation ────────────────────────────
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        root_state[:, 2] += self.cfg.fall_height  # 공중에서 낙하 시작
        root_state[:, 3:7] = quat
        root_state[:, 7:] = 0.0  # 초기 velocity = 0

        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    # ── cleanup ───────────────────────────────────────────────────────────────

    def __del__(self):
        # DebugViewer cleanup (필수 3줄 중 3번째)
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
            self._debug_viewer.close()

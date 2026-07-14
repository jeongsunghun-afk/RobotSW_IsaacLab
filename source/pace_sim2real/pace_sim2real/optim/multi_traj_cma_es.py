# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# Local extension to the vendored PACE optimizer (not upstream). See VENDORING.md.

"""여러 궤적을 동시에 맞추는 CMA-ES.

PACE 상위 클래스는 궤적 하나를 재생하며 점수를 누적한다. 이 확장은 한 세대 안에서 **여러
녹화 시퀀스를 차례로 재생**하고 점수를 합산해, 하나의 파라미터 집합이 모든 시퀀스를 동시에
설명하도록 만든다. PACE 논문도 단일 드라이브 단계에서 30개 chirp(PD 게인 3종 × 부하 5종 ×
펌웨어 2종)를 결합 적합했고, 전신 단계에서도 진폭이 다른 여러 시퀀스를 쓴다.

**시퀀스마다 녹화 당시의 PD 게인을 되살려 재생한다.** 이것이 이 클래스의 존재 이유다:
PACE는 게인을 식별하지 않고 알려진 값으로 가정하므로(논문: ``{I_a, d, P_tau, D_tau}``의 공통
스케일이 폐루프 거동을 보존해 최적해가 축퇴한다), 잘못된 게인으로 재생하면 그 불일치가
플랜트 파라미터를 **편향**시킨다. 잡음이 아니라 편향이다.

Note:
    kd 오차는 viscous 마찰과 수학적으로 구분되지 않는다(둘 다 관절 속도에 곱해진다). 따라서
    식별된 viscous 값은 "실제 점성 마찰 + kd 오차"다. 이건 다중 게인 데이터로도 깨지지 않는
    구조적 축퇴이며, 위치 궤적 재현에는 무해하지만 토크 크기에는 영향을 준다.
"""

from __future__ import annotations

import torch

from .cma_es import CMAESOptimizer


class MultiTrajectoryCMAES(CMAESOptimizer):
    """여러 궤적(각자 자신의 PD 게인을 가진)에 대해 공유 파라미터 집합을 적합한다."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 현재 재생 중인 궤적 인덱스. sim_dof_pos_buffer(체크포인트/플롯용)는 0번 궤적만 담는다.
        self._active_idx: int = 0
        self._traj_counter: int = 0

    def begin_trajectory(self, idx: int) -> None:
        """새 궤적 재생을 시작한다. :meth:`tell` 호출 전에 반드시 부른다.

        Args:
            idx: 궤적 인덱스. 0번 궤적만 ``best_trajectory.pt``로 저장된다.
        """
        self._active_idx = idx
        self._traj_counter = 0

    def tell(self, sim_dof_pos: torch.Tensor, real_dof_pos: torch.Tensor) -> None:
        """한 스텝의 위치 오차를 점수에 누적한다.

        상위 클래스와 동일한 손실(엔코더 바이어스를 뺀 제곱오차)을 쓰되, 궤적 길이가 서로 다를 수
        있으므로 궤적 버퍼는 0번 궤적에 대해서만, 그 궤적의 길이 안에서만 채운다.

        Args:
            sim_dof_pos: 시뮬레이션 관절각 [rad], shape (population, num_joints).
            real_dof_pos: 실측(엔코더) 관절각 [rad], shape (population, num_joints).
        """
        self.scores += torch.sum(torch.square(sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx]), dim=1)
        if self._active_idx == 0 and self._traj_counter < self.sim_dof_pos_buffer.shape[1]:
            self.sim_dof_pos_buffer[:, self._traj_counter, :] = sim_dof_pos
        self._traj_counter += 1
        # evolve()가 scores를 이 카운터로 나눈다 → 전체 궤적·전체 스텝에 대한 평균 제곱오차.
        self.scores_counter += 1

    @staticmethod
    def apply_gains(articulation, joint_ids: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor) -> None:
        """녹화 당시의 PD 게인을 explicit actuator에 되살린다.

        explicit actuator(:class:`~isaaclab.actuators.DCMotor` 계열)는 PD를 PhysX가 아니라 파이썬에서
        계산한다 — 진짜 게인은 ``actuator.stiffness`` / ``actuator.damping`` 텐서다. PhysX 쪽 관절
        stiffness/damping은 0으로 남아 있으므로 ``write_joint_stiffness_to_sim*``을 써도 효과가 없다.
        (``write_actuator_stiffness_to_sim``은 Newton 액추에이터 전용이라 PhysX에서는 no-op이다.)

        Args:
            articulation: 대상 articulation.
            joint_ids: JOINT_ORDER 순서의 articulation 관절 인덱스, shape (num_joints,).
            kp: 관절별 위치 게인 [N·m/rad], JOINT_ORDER 순서, shape (num_joints,).
            kd: 관절별 속도 게인 [N·m·s/rad], JOINT_ORDER 순서, shape (num_joints,).
        """
        device = kp.device
        joint_ids_long = joint_ids.to(torch.long)
        for actuator in articulation.actuators.values():
            drive_ids = actuator.joint_indices
            if isinstance(drive_ids, slice):
                drive_ids = torch.arange(articulation.num_joints, device=device)[drive_ids]
            drive_ids = torch.as_tensor(drive_ids, device=device).to(torch.long)
            # actuator의 관절 순서(articulation 순서) -> JOINT_ORDER에서의 위치
            order_pos = torch.tensor(
                [int((joint_ids_long == int(j)).nonzero()[0]) for j in drive_ids], dtype=torch.long, device=device
            )
            actuator.stiffness[:] = kp[order_pos].unsqueeze(0)
            actuator.damping[:] = kd[order_pos].unsqueeze(0)

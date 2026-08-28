# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent-space imitation 용 PPO — AMP discriminator 없이 RMA/estimator 만 승계한다.

:class:`~rsl_rl.algorithms.ppo_amp.PPOAMP` 가 아니라
:class:`~rsl_rl.algorithms.ppo_parkour.PPOParkour` 를 상속한다. 스타일 신호는 학습되는
discriminator 가 아니라 **env 가 계산하는 동결 인코더 기반 보상**이므로, 알고리즘 쪽에
추가 네트워크·옵티마이저·replay buffer 가 전혀 필요 없다. 이 클래스가 더하는 것은
task/style 혼합 비율 ``task_reward_lerp`` 와 그 스케줄 상태뿐이다.

혼합식 (러너가 적용한다)::

    total = lerp * task + (1 - lerp) * style

기호:

=============================  ==========================================  ==========
기호                           의미                                        단위
=============================  ==========================================  ==========
``task_reward_lerp``           task 보상 비중 (Stage 2 목표값)              무차원
``task_reward_lerp_start``     Stage 1 초기값                               무차원
``task_reward_lerp_anneal``    두 값 사이 선형 전환에 쓰는 iter 수           iter
=============================  ==========================================  ==========
"""

from __future__ import annotations

from rsl_rl.algorithms.ppo_parkour import PPOParkour


class PPOLatent(PPOParkour):
    """PPO + RMA + estimator. 스타일 보상은 env 에서 오고 여기서는 혼합 비율만 관리한다."""

    def __init__(self, *args: object, style_cfg: dict | None = None, **kwargs: object) -> None:
        """``style_cfg`` 는 러너 cfg 의 ``style`` 블록이다 (없으면 순수 task 보상)."""
        kwargs.pop("class_name", None)
        super().__init__(*args, **kwargs)

        style_cfg = style_cfg or {}
        self.task_reward_lerp: float = float(style_cfg.get("task_reward_lerp", 0.5))
        self.task_reward_lerp_end: float = self.task_reward_lerp
        self.task_reward_lerp_start: float = float(style_cfg.get("task_reward_lerp_start", self.task_reward_lerp_end))
        self.task_reward_lerp_anneal_iters: int = int(style_cfg.get("task_reward_lerp_anneal_iters", 0))
        self.enable_lerp_schedule: bool = bool(style_cfg.get("enable_lerp_schedule", True))

    def update_lerp_schedule(self, it: int) -> float:
        """Stage 1(순수 task) → Stage 2(task+style) 선형 전환.

        ★ progress 를 **절대 iter** 로 잰다. AMP 러너는 ``it - start_it`` 을 쓰는데, 그러면
        ``--resume`` 으로 재개할 때마다 ``lerp`` 가 Stage 1(스타일 0)로 되감기고 anneal 을
        다시 올라간다. AMP baseline 은 start 와 end 가 같아(둘 다 0.5) 이 거동이 드러나지
        않았지만, 여기서는 start 1.0 → end 0.5 라 실제로 밟는다. 스케줄은 학습 진행도의
        함수여야 하고 ``learn`` 호출 경계의 함수가 아니다.

        Args:
            it: 현재 iter (절대값 — resume 이면 이어진 값).

        Returns:
            갱신된 ``task_reward_lerp``.
        """
        if self.enable_lerp_schedule and self.task_reward_lerp_anneal_iters > 0:
            progress = min(1.0, it / self.task_reward_lerp_anneal_iters)
            self.task_reward_lerp = (
                self.task_reward_lerp_start + (self.task_reward_lerp_end - self.task_reward_lerp_start) * progress
            )
        return self.task_reward_lerp

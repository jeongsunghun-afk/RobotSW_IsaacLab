# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab_assets.robots.rga import RGA_GO2_CFG

from .go2_interaction_cfg import Go2InteractionCfg


@configclass
class Go2NeckInteractionCfg(Go2InteractionCfg):
    """Go2 목이 부착된 모델의 상호작용 학습 환경 설정."""

    # ------------------------------------------------------------------ #
    # 기본 환경 파라미터
    # ------------------------------------------------------------------ #
    action_space: int = 19  # 12 (legs) + 7 (neck)

    # ------------------------------------------------------------------ #
    # 관측 공간 구성
    # - prio_obs  : gravity(3) + interaction_cmd(1)
    #               + dof_pos(19) + dof_vel(19) + actions(19) + roll_pitch(2)
    #               = 63
    # - num_priv  : base_euler xyz (3)
    # - num_priv_latent : height(1) + lin_vel(3) + ang_vel(3)
    #                     + dof_pos[:6](6) + dof_vel[:6](6) + torques[:3](3) = 22
    # - history   : prio_obs * history_len
    # ------------------------------------------------------------------ #
    num_prio_obs: int = 63
    history_len: int = 10
    observation_space: int = num_prio_obs + 3 + 22 + num_prio_obs * history_len
    # 63 + 3 + 22 + 630 = 718

    # ------------------------------------------------------------------ #
    # 로봇
    # ------------------------------------------------------------------ #
    robot = RGA_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )

    # 패널티 대상 바디 이름에 목 파트 추가
    penalized_body_names: list = ["base", ".*thigh", ".*calf", ".*hip", ".*neck_p", ".*neck_r", ".*neck_y"]

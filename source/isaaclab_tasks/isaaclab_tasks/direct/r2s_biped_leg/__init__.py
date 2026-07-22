# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Biped Leg Real2Sim 환경 (8-DOF 2족, 순수 UDP).

실행:
  # 터미널 1: 시뮬레이션 (Isaac conda)
  LIVESTREAM=2 CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p scripts/real2sim/sim_runner_bipedleg.py \
      --num_envs 1 --fix_base --viz kit

  # 터미널 2: GUI (시스템 python)
  bash scripts/real2sim/r2s_biped_leg/run_gui_controller.sh
"""

import gymnasium as gym

##
# Register Gym environments.
##

gym.register(
    id="Isaac-R2S-BipedLeg-v0",
    entry_point=f"{__name__}.r2s_biped_leg_env:R2SBipedLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2s_biped_leg_env_cfg:R2SBipedLegEnvCfg",
    },
)

# 시스템 식별(PACE CMA-ES) 모드 — live 환경과 asset/관절순서를 공유하는 배치 적합용.
#   ./isaaclab.sh -p scripts/real2sim/fit_bipedleg.py --headless --num_envs 4096 \
#       --task Isaac-R2S-BipedLeg-Sysid-v0
gym.register(
    id="Isaac-R2S-BipedLeg-Sysid-v0",
    entry_point=f"{__name__}.r2s_biped_leg_env:R2SBipedLegEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.r2s_biped_leg_sysid_cfg:R2SBipedLegSysidEnvCfg",
    },
)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-BipedLeg 편의 셸 명령: r2s_bl_sim / r2s_bl_gui.
# 이 파일을 source 하면 터미널에서 바로 쓸 수 있다:
#
#   source /home/lgb/IsaacLab-6.0/scripts/real2sim/r2s_biped_leg/r2s_commands.sh
#
#   r2s_bl_sim   # 터미널1: Isaac sim_runner (conda isaac-6.0)
#   r2s_bl_gui   # 터미널2: PyQt GUI (순수 UDP, 시스템 python)
#
# 각 명령은 추가 인자를 그대로 전달한다. 예: VIZ= r2s_bl_sim  (헤드리스)

_R2S_BL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

r2s_bl_sim() { bash "$_R2S_BL_DIR/run_sim_runner.sh" "$@"; }      # 터미널1: Isaac sim_runner
r2s_bl_gui() { bash "$_R2S_BL_DIR/run_gui_controller.sh" "$@"; }  # 터미널2: PyQt GUI

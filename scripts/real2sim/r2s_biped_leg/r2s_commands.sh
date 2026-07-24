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
#   position 모드 (수동 관절 제어):
#     r2s_bl_sim   # 터미널1: Isaac sim_runner (conda isaac-6.0)
#     r2s_bl_gui   # 터미널2: PyQt GUI (순수 UDP, 시스템 python)
#
#   policy 모드 (학습 정책 구동, GUI에서 Mode=Policy):
#     r2s_bl_psim  # 터미널1: sim_runner --policy_mode (conda) — 라이브스트림 VIZ=kit
#     r2s_bl_prun  # 터미널2: policy_runner (conda torch)
#     r2s_bl_gui   # 터미널3: PyQt GUI → Mode를 Policy로, Run 클릭
#
# 각 명령은 추가 인자를 그대로 전달한다. 예: VIZ= r2s_bl_psim  (헤드리스),  GPU=1 r2s_bl_prun

_R2S_BL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

r2s_bl_sim() { bash "$_R2S_BL_DIR/run_sim_runner.sh" "$@"; }        # position: Isaac sim_runner
r2s_bl_gui() { bash "$_R2S_BL_DIR/run_gui_controller.sh" "$@"; }    # PyQt GUI (position/policy 공용)
r2s_bl_psim() { bash "$_R2S_BL_DIR/run_policy_sim.sh" "$@"; }       # policy: sim_runner --policy_mode
r2s_bl_prun() { bash "$_R2S_BL_DIR/run_policy_runner.sh" "$@"; }    # policy: policy_runner (추론)

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
#   기본 (정책 + 슬라이더를 한 프로세스에서 — 2026-08-28 통합):
#     r2s_bl_sim   # 터미널1: sim_runner --unified (conda isaac-6.0)
#     r2s_bl_gui   # 터미널2: PyQt GUI (순수 UDP, 시스템 python) → Mode를 Policy로, Run 클릭
#
#   ★ 안전 점검 (공중 고정 — 넘어질 수 없다):
#     FIX_BASE=1 r2s_bl_sim   # 정책·슬라이더 모두 공중 고정 상태로 돈다. 실기 전에 관절 거동·
#                             # 부호·토크 크기를 여기서 먼저 본다. 슬라이더로 관절 추종을 볼
#                             # 때도 이쪽이다(자유베이스면 스텝하는 순간 넘어진다).
#                             # ⚠ 접촉이 없고 base 상태가 학습 분포 밖이라 **보행 성능·트립
#                             #   판정은 자유베이스에서 다시 재야 한다.**
#
#   정책만 단독 (구 경로, 과거 캡처 재현용):
#     r2s_bl_psim  # sim_runner --policy_mode
#     r2s_bl_prun  # policy_runner (추론을 별도 프로세스에서)
#
# 각 명령은 추가 인자를 그대로 전달한다. 예: VIZ= r2s_bl_sim  (헤드리스),  GPU=1 r2s_bl_prun

_R2S_BL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 항상 unified(정책 lockstep + 슬라이더 free-run 한 프로세스). `FIX_BASE=1` 은 그 안에서
# base 공중고정으로 전달된다 — 정책을 넘어뜨리지 않고 먼저 보는 **안전 점검 첫 칸**이다.
r2s_bl_sim() { bash "$_R2S_BL_DIR/run_unified_sim.sh" "$@"; }
r2s_bl_gui() { bash "$_R2S_BL_DIR/run_gui_controller.sh" "$@"; }    # PyQt GUI (position/policy 공용)
r2s_bl_psim() { bash "$_R2S_BL_DIR/run_policy_sim.sh" "$@"; }       # policy: sim_runner --policy_mode
r2s_bl_prun() { bash "$_R2S_BL_DIR/run_policy_runner.sh" "$@"; }    # policy: policy_runner (추론)

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
#   공중 고정 슬라이더 작업 (관절 추종 확인 등):
#     FIX_BASE=1 r2s_bl_sim   # position 모드로 폴백 (--fix_base). 자유베이스면 관절을 스텝하는
#                             # 순간 넘어지므로, 이 용도는 반드시 FIX_BASE=1 이어야 한다.
#
#   정책만 단독 (구 경로, 과거 캡처 재현용):
#     r2s_bl_psim  # sim_runner --policy_mode
#     r2s_bl_prun  # policy_runner (추론을 별도 프로세스에서)
#
# 각 명령은 추가 인자를 그대로 전달한다. 예: VIZ= r2s_bl_sim  (헤드리스),  GPU=1 r2s_bl_prun

_R2S_BL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 기본은 unified(정책+슬라이더 한 프로세스). `FIX_BASE=1` 이면 공중 고정 position 모드로 간다 —
# unified 는 정책이 균형을 잡아야 해서 fix_base 를 강제 False 로 두므로, 두 요구를 한 프로세스로는
# 못 만족시킨다(fix_root_link 는 씬 생성 시점 성질이라 런타임 토글이 불가능).
r2s_bl_sim() {
    if [ "${FIX_BASE:-0}" != "0" ]; then
        bash "$_R2S_BL_DIR/run_sim_runner.sh" "$@"      # position: 공중 고정 슬라이더
    else
        bash "$_R2S_BL_DIR/run_unified_sim.sh" "$@"     # unified: 정책 lockstep + 슬라이더 free-run
    fi
}
r2s_bl_gui() { bash "$_R2S_BL_DIR/run_gui_controller.sh" "$@"; }    # PyQt GUI (position/policy 공용)
r2s_bl_psim() { bash "$_R2S_BL_DIR/run_policy_sim.sh" "$@"; }       # policy: sim_runner --policy_mode
r2s_bl_prun() { bash "$_R2S_BL_DIR/run_policy_runner.sh" "$@"; }    # policy: policy_runner (추론)

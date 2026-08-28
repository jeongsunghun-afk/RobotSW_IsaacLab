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

# ─────────────────────────────────────────────────────────────────────────────
# r2s_bl_vnc — SSH X11 포워딩 대신 **VNC 세션**에서 GUI/monitor 를 띄운다.
#
# 왜: 이 워크스테이션에 붙는 클라이언트가 XQuartz(macOS)이고 `DISPLAY=localhost:N` 즉 SSH
#   X11 포워딩이다. 이 조합의 알려진 증상이 (a) 창이 까맣게만 보이고 (b) 렉이 심한 것이다.
#   포워딩이 untrusted(`ssh -X`)면 **20분 타임아웃** 뒤 X 요청이 거부돼 창이 죽는다.
#   VNC 는 렌더를 워크스테이션의 진짜 X 서버에서 하고 화면만 보내므로 둘 다 해결된다.
#   세션이 남아 있어 SSH 가 끊겨도 GUI 가 안 죽는다(학습·실기 구동 중엔 이게 크다).
#
# 쓰는 법:
#   r2s_bl_vnc start      # :78 에 세션 생성 (localhost 바인딩)
#   # 맥에서:  ssh -N -L 5978:localhost:5978 lgb@<이 머신>
#   #          그 뒤 VNC 뷰어로 localhost:5978 접속 (macOS 기본 "화면 공유" 앱도 됨)
#   DISPLAY=:78 r2s_bl_gui        # GUI 를 VNC 쪽에 띄운다
#   DISPLAY=:78 python3 "$_R2S_BL_DIR/monitor.py"
#   r2s_bl_vnc stop
#
# ⚠ `-SecurityTypes None` 이라 **반드시 -localhost yes** 여야 한다(SSH 터널 경유 전제).
_R2S_BL_VNC_DISPLAY="${R2S_VNC_DISPLAY:-:78}"
r2s_bl_vnc() {
    case "${1:-start}" in
        start)
            tigervncserver -localhost yes -geometry "${R2S_VNC_GEOM:-1600x1000}" \
                -SecurityTypes None "$_R2S_BL_VNC_DISPLAY" || return 1
            echo "VNC up on $_R2S_BL_VNC_DISPLAY  (port 59${_R2S_BL_VNC_DISPLAY#:})"
            echo "  맥에서: ssh -N -L 59${_R2S_BL_VNC_DISPLAY#:}:localhost:59${_R2S_BL_VNC_DISPLAY#:} \$USER@$(hostname -I | awk '{print $1}')"
            echo "  그 다음: DISPLAY=$_R2S_BL_VNC_DISPLAY r2s_bl_gui"
            ;;
        stop) tigervncserver -kill "$_R2S_BL_VNC_DISPLAY" ;;
        *) echo "usage: r2s_bl_vnc [start|stop]" >&2; return 2 ;;
    esac
}

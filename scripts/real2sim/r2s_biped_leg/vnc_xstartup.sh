#!/bin/sh
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# r2s_bl_vnc 전용 TigerVNC 세션 시작 스크립트.
#
# 왜 따로 두나: 기본 `~/.vnc/xstartup` 은 root 소유이고 다른 VNC 세션(5900~5902)과 공용인데,
# 모든 것을 백그라운드로 띄우고 `wait` 없이 끝나서 세션이 **3초 안에 스스로 종료**된다
# ("Session startup ... cleanly exited too early"). 마지막 줄을 foreground `exec` 로 두어야
# 세션이 유지된다. 남의 설정을 고치는 대신 이 파일을 `-xstartup` 으로 지정한다.

unset SESSION_MANAGER
unset DBUS_SESSION_BUS_ADDRESS

[ -r "$HOME/.Xresources" ] && xrdb "$HOME/.Xresources"
xsetroot -solid "#20222a"
vncconfig -iconic &

# ★ foreground exec — 이게 세션 수명을 쥔다.
exec startxfce4

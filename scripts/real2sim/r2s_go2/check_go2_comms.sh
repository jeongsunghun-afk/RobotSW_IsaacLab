#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# GO2 <-> 서버 ROS2 통신 점검 (M2). 로봇 이더넷 연결 + network(192.168.123.x static) 후 실행.
#   bash scripts/real2sim/r2s_go2/check_go2_comms.sh [iface]
#     iface: 생략 시 robot_env.sh가 192.168.123.x 보유 iface 자동탐지(예: ens10f1).
#
# 주의: ros2 daemon이 stale 토픽 캐시를 들고 있으면 로봇 토픽이 안 보인다(실측 확인됨).
#       이 스크립트는 디스커버리 전에 daemon을 재시작해 그 함정을 피한다.

set +u
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate 2>/dev/null || break; done
unset PYTHONPATH PYTHONHOME
# shellcheck disable=SC1091
source "$DIR/robot_env.sh" "${1:-}" >/dev/null 2>&1

_iface_shown="$(printf '%s' "$CYCLONEDDS_URI" | grep -o 'name="[^"]*"' || echo 'name=(unset)')"
echo "[check] RMW=$RMW_IMPLEMENTATION  DOMAIN=${ROS_DOMAIN_ID:-0}  iface $_iface_shown"

# stale 캐시 제거
ros2 daemon stop >/dev/null 2>&1
sleep 1

ok=1

echo "===== 1) 로봇 ping (192.168.123.161) ====="
if ping -c 2 -W 2 192.168.123.161 >/dev/null 2>&1; then echo "  ✅ ping OK"; else echo "  ❌ ping 실패 (cable/network/IP)"; ok=0; fi

echo "===== 2) 기대 토픽 확인 ====="
topics="$(timeout 15 ros2 topic list 2>/dev/null | sort)"
for t in /lowstate /lowcmd /sportmodestate /lf/lowstate /wirelesscontroller /utlidar/cloud; do
    if printf '%s\n' "$topics" | grep -qx "$t"; then echo "  ✅ $t"; else echo "  ❌ $t 없음"; ok=0; fi
done

echo "===== 3) /lowstate 파싱 수신 (3s) ====="
/usr/bin/python3 "$DIR/read_lowstate.py" 3 || ok=0

echo ""
if [ "$ok" = 1 ]; then echo "RESULT: PASS — GO2 ROS2 통신 정상"; else echo "RESULT: FAIL — 위 ❌ 항목 확인 (RMW/iface/daemon/로봇전원)"; fi
[ "$ok" = 1 ]

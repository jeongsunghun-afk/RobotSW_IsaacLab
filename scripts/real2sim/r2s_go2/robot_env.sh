# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 실기 통신 환경 설정 (M2). **source 해서 사용** — 현재 셸에 ROS2 humble +
# unitree_go overlay + CycloneDDS RMW 를 세팅한다. 실기 GO2 는 CycloneDDS 로 통신하므로
# 로봇과 대화하는 모든 노드가 이 RMW + 올바른 네트워크 인터페이스를 써야 한다.
#
# 사용:
#   source scripts/real2sim/r2s_go2/robot_env.sh [iface]
#     iface  : GO2 가 연결된 네트워크 인터페이스명. 생략 시
#              (1) 환경변수 R2S_ROBOT_IFACE, (2) 192.168.123.x 주소를 가진 iface 자동탐지 순.
#
# 전제(당신이 하는 물리 세팅):
#   1) GO2 <-> 서버 이더넷 연결.
#   2) 연결된 인터페이스를 static IP 192.168.123.99 / mask 255.255.255.0 (수동) 설정.
#      (GO2 기본 대역 192.168.123.x, 로봇 기본 192.168.123.161.)

# conda 완전 비활성화 — ros2 CLI/rclpy 는 시스템 python3.10 을 써야 함(conda 3.12 는 rclpy 깨짐).
set +u 2>/dev/null
while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate 2>/dev/null || break; done
unset PYTHONPATH PYTHONHOME 2>/dev/null

# ROS2 humble + 우리 unitree_go/unitree_api overlay (humble 빌드).
source /opt/ros/humble/setup.bash
source /home/lgb/unitree_ros2/cyclonedds_ws/install/setup.bash

export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

# 인터페이스 결정: 인자 > R2S_ROBOT_IFACE > 192.168.123.x 보유 iface 자동탐지.
_r2s_iface="${1:-${R2S_ROBOT_IFACE:-}}"
if [ -z "$_r2s_iface" ]; then
    _r2s_iface="$(ip -o -4 addr show 2>/dev/null | awk '/192\.168\.123\./{print $2; exit}')"
fi

if [ -z "$_r2s_iface" ]; then
    echo "[robot_env] ⚠ GO2 인터페이스 미확정 (192.168.123.x 보유 iface 없음)."
    echo "[robot_env]   → 랜 연결 + static IP(192.168.123.99/24) 설정 후 다시 source 하거나,"
    echo "[robot_env]     R2S_ROBOT_IFACE=<iface> 로 지정하세요."
    echo "[robot_env]   현재 UP 인터페이스: $(ip -br link 2>/dev/null | awk '$2=="UP"{print $1}' | tr '\n' ' ')"
    echo "[robot_env]   (RMW=cyclonedds, DOMAIN=$ROS_DOMAIN_ID 는 설정됨. CYCLONEDDS_URI 만 미설정.)"
else
    export CYCLONEDDS_URI="<CycloneDDS><Domain><General><Interfaces><NetworkInterface name=\"$_r2s_iface\" priority=\"default\" multicast=\"default\" /></Interfaces></General></Domain></CycloneDDS>"
    echo "[robot_env] ✅ RMW=rmw_cyclonedds_cpp  iface=$_r2s_iface  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
    echo "[robot_env]   확인:  ros2 topic list   |   ros2 topic echo --once /sportmodestate"
fi
unset _r2s_iface

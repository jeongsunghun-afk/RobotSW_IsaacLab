#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 게인 규약 검증 원커맨드 런처 (Unitree kp가 IsaacLab과 같은 N·m/rad인가?).
#
# ⚠ 로봇을 공중에 매달고(다리 자유) sport 서비스를 내린 뒤 실행할 것.
#
# 사용:
#   bash run_gain_check.sh --selftest                      # 추정기만 검증 (로봇 불필요)
#   bash run_gain_check.sh --dry-run                       # 무모션 계획 확인
#   bash run_gain_check.sh --suspended --joint 1 --kp 25   # 실측

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_real_py.sh" gain_check.py "$@"

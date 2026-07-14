#!/usr/bin/env bash
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
#
# R2S-GO2 실기 chirp 수집 원커맨드 런처 (PACE Phase 2).
#
# ⚠ 로봇을 공중에 매달고 sport 서비스를 내린 뒤 실행할 것 (CONTRACT §12.10).
#
# 사용:
#   bash run_chirp_collector.sh --dry-run                                   # 무모션 계획 확인
#   bash run_chirp_collector.sh --suspended --amplitude_scale 0.3 --out sweep_check.npz   # 리그 공진 확인
#   bash run_chirp_collector.sh --suspended --kp 25 --kd 0.5 --out chirp_kp25.npz         # 본 수집

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$HERE/run_real_py.sh" chirp_collector.py "$@"

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-GO2 여기신호(chirp) — sim/실기 공유 정의.

PACE 시스템 식별은 **sim에서 재생하는 명령과 실기에 보낸 명령이 정확히 같을 때만** 성립한다.
그래서 궤적 생성은 여기 한 곳에만 둔다:

- Isaac 쪽(`collect_chirp_sim.py`, conda 3.12)
- ROS2 쪽(실기 `chirp_collector.py`, 시스템 3.10 + humble)

양쪽에서 임포트되므로 **순수 stdlib만 사용한다**(numpy/torch/ros 의존 금지 — `r2s_udp.py`와 동일 계약).

관절 순서는 CONTRACT §2의 `JOINT_ORDER`(FR/FL/RR/RL × hip,thigh,calf)를 따른다.
"""

from __future__ import annotations

import math

NUM_MOTORS: int = 12

# ---------------------------------------------------------------------------
# 여기신호 파라미터 (다리당 [hip, thigh, calf], 4개 다리 반복)
# ---------------------------------------------------------------------------

# 진동 중심 [rad]. GO2 기립 자세 근처 — 다리가 서로/몸통과 닿지 않는 영역.
CHIRP_CENTER: list[float] = [0.0, 0.9, -1.5] * 4

# 진폭 [rad]. soft limit(soft_joint_pos_limit_factor=0.9) 안에 여유를 두고 잡았다:
#   hip   0.00 ± 0.25 ⊂ soft[-0.943, 0.943]
#   thigh 0.90 ± 0.45 ⊂ soft[-1.317, 3.237]
#   calf -1.50 ± 0.50 ⊂ soft[-2.628, -0.932]   ← 가장 빡빡함. 0.5를 넘기지 말 것.
CHIRP_AMPLITUDE: list[float] = [0.25, 0.45, 0.50] * 4

# 좌우 hip 부호 반전 — 베이스에 걸리는 롤 모멘트를 상쇄한다(매단 리그 흔들림 최소화).
# JOINT_ORDER = FR, FL, RR, RL 이므로 R(우)=+1, L(좌)=-1.
CHIRP_DIRECTION: list[float] = [
    +1.0,
    1.0,
    1.0,  # FR
    -1.0,
    1.0,
    1.0,  # FL
    +1.0,
    1.0,
    1.0,  # RR
    -1.0,
    1.0,
    1.0,  # RL
]

# 기본 chirp 스펙 (PACE ANYmal 예제와 동일한 형태)
DEFAULT_F0_HZ: float = 0.1
DEFAULT_F1_HZ: float = 10.0
DEFAULT_DURATION_S: float = 20.0
DEFAULT_RATE_HZ: float = 500.0  # 실기 /lowcmd 발행률 = sim 제어율 (CONTRACT §8 참고)

# GO2 soft joint limit [rad] — URDF 하드 리밋에 soft_joint_pos_limit_factor=0.9를 적용한 값.
# 이걸 넘는 목표각은 sim이 조용히 클램프하고, 실기에서는 관절이 기계 한계에 부딪친다.
# (JOINT_ORDER 순서: 다리당 hip, thigh, calf)
SOFT_LIMITS: list[tuple[float, float]] = [(-0.943, 0.943), (-1.317, 3.237), (-2.628, -0.932)] * 4


def check_within_soft_limits(targets: list[list[float]]) -> list[tuple[int, float, float]]:
    """궤적이 GO2 soft limit 안에 있는지 검사한다.

    sim 재생과 실기 발행 **양쪽에서 발행 전에** 반드시 부른다.

    Args:
        targets: T×12 목표각 [rad], JOINT_ORDER 순서.

    Returns:
        위반 목록 ``[(joint_index, min, max), ...]``. 비어 있으면 안전.
    """
    violations: list[tuple[int, float, float]] = []
    for j in range(NUM_MOTORS):
        lo = min(row[j] for row in targets)
        hi = max(row[j] for row in targets)
        if lo <= SOFT_LIMITS[j][0] or hi >= SOFT_LIMITS[j][1]:
            violations.append((j, lo, hi))
    return violations


def chirp_phase(t: float, f0: float, f1: float, duration: float) -> float:
    """선형 chirp의 순시 위상 [rad].

    ``phase(t) = 2*pi * (f0*t + (f1-f0)/(2*duration) * t^2)`` — 주파수가 f0에서 f1로 선형 증가한다.

    Args:
        t: 시각 [s].
        f0: 시작 주파수 [Hz].
        f1: 종료 주파수 [Hz].
        duration: chirp 전체 길이 [s].

    Returns:
        순시 위상 [rad].
    """
    return 2.0 * math.pi * (f0 * t + ((f1 - f0) / (2.0 * duration)) * t * t)


def build_chirp(
    duration: float = DEFAULT_DURATION_S,
    rate_hz: float = DEFAULT_RATE_HZ,
    f0: float = DEFAULT_F0_HZ,
    f1: float = DEFAULT_F1_HZ,
    center: list[float] | None = None,
    amplitude: list[float] | None = None,
    direction: list[float] | None = None,
) -> tuple[list[float], list[list[float]]]:
    """전 관절 동시 여기용 chirp 위치 궤적을 만든다.

    각 관절 j의 목표각은 ``center[j] + direction[j] * amplitude[j] * sin(phase(t))`` 이다.
    모든 관절이 같은 위상을 공유하므로 한 번의 스윕으로 12관절을 동시에 여기한다.

    Args:
        duration: chirp 길이 [s].
        rate_hz: 샘플링/명령 주파수 [Hz].
        f0: 시작 주파수 [Hz].
        f1: 종료 주파수 [Hz].
        center: 관절별 진동 중심 [rad], 길이 12. None이면 :attr:`CHIRP_CENTER`.
        amplitude: 관절별 진폭 [rad], 길이 12. None이면 :attr:`CHIRP_AMPLITUDE`.
        direction: 관절별 부호, 길이 12. None이면 :attr:`CHIRP_DIRECTION`.

    Returns:
        ``(time, targets)`` — ``time``은 길이 T의 시각 [s], ``targets``는 T×12 목표각 [rad]
        (JOINT_ORDER 순서).
    """
    center = list(CHIRP_CENTER if center is None else center)
    amplitude = list(CHIRP_AMPLITUDE if amplitude is None else amplitude)
    direction = list(CHIRP_DIRECTION if direction is None else direction)
    for name, vec in (("center", center), ("amplitude", amplitude), ("direction", direction)):
        if len(vec) != NUM_MOTORS:
            raise ValueError(f"{name}는 길이 {NUM_MOTORS}이어야 한다 (현재 {len(vec)}).")

    num_steps = int(round(duration * rate_hz))
    dt = 1.0 / rate_hz

    time: list[float] = []
    targets: list[list[float]] = []
    for k in range(num_steps):
        t = k * dt
        s = math.sin(chirp_phase(t, f0, f1, duration))
        time.append(t)
        targets.append([center[j] + direction[j] * amplitude[j] * s for j in range(NUM_MOTORS)])
    return time, targets


if __name__ == "__main__":
    # 자체 검증: 길이 + soft limit 여유 + 시작점이 중심과 일치(sin(0)=0)
    t, traj = build_chirp()
    assert len(t) == len(traj) == 10000, (len(t), len(traj))
    assert all(abs(traj[0][j] - CHIRP_CENTER[j]) < 1e-9 for j in range(NUM_MOTORS))
    violations = check_within_soft_limits(traj)
    assert not violations, f"soft limit 위반: {violations}"
    print(f"chirp OK  steps={len(t)}  span={t[-1]:.3f}s")
    for j, name in enumerate(["hip", "thigh", "calf"]):
        lo = min(row[j] for row in traj)
        hi = max(row[j] for row in traj)
        print(f"  {name:6s} range=[{lo:+.3f}, {hi:+.3f}]  soft={SOFT_LIMITS[j]}")

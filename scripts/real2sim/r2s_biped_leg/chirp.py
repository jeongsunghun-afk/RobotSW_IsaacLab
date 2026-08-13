# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg 여기신호(chirp) — sim/실기 공유 정의.

PACE 시스템 식별은 **sim에서 재생하는 명령과 실기에 보낸 명령이 정확히 같을 때만** 성립한다.
그래서 궤적 생성은 여기 한 곳에만 둔다:

- Isaac 쪽(`collect_chirp_sim_bipedleg.py`, conda 3.12)
- 실기 수집 쪽(시스템 3.10 + humble; ROS2 메시지 형식은 CONTRACT §4 seam으로 미정)

양쪽에서 임포트되므로 **순수 stdlib만 사용한다**(numpy/torch/ros 의존 금지 — `r2s_udp.py`와 동일 계약).

관절 순서는 CONTRACT §2의 leg-major 순서(HL 4개 → HR 4개 × hip, thigh, calf, foot)를 따르며,
`r2s_biped_leg_env_cfg.JOINT_NAME_PATTERNS`와 1:1 대응한다.
"""

from __future__ import annotations

import math

NUM_MOTORS: int = 8

# ---------------------------------------------------------------------------
# 여기신호 파라미터 (다리당 [hip, thigh, calf, foot], HL → HR)
# ---------------------------------------------------------------------------

# 진동 중심 [rad]. soft limit 구간의 중앙 근처로 잡아 진폭 여유를 최대화한다.
# 2026-08-11: 신규 CAD 리비전 Hind_Leg_URDF2 (좌우 동일 규약)로 soft limit이 바뀌어
#   중심·진폭을 재설계했다. 이 값으로 수집한 chirp은 구 플랜트 데이터셋과 비교 불가 —
#   새 플랜트에서 재수집할 것.
# 2026-08-12 SignFix: HL_calf/HL_foot/HR_thigh 축 반전(자산 Hind_Leg_URDF3_SignFix)에 맞춰
# 반전 관절의 중심 부호를 뒤집었다 — 물리적 자세는 반전 전과 동일하다.
CHIRP_CENTER: list[float] = [0.0, 0.62, 0.09, 0.52, 0.0, -0.62, -0.09, -0.52]

# 진폭 [rad]. soft limit(soft_joint_pos_limit_factor=0.9) 안쪽으로 최소 ~0.09 rad 여유를 둔다:
#   hip       0.00 ± 0.14 ⊂ soft[-0.2340, 0.2340]  ← hip 가동범위가 ±14.9°로 좁아져 진폭 대폭 축소
#   HL_thigh  0.62 ± 0.60 ⊂ soft[-0.9555, 2.1855]  ← 가장 여유 큼 (HR은 미러: -0.62 ⊂ [-2.1855, 0.9555])
#   HL_calf   0.09 ± 0.75 ⊂ soft[-0.7745, 0.9445]  (HR은 미러: -0.09 ⊂ [-0.9445, 0.7745])
#   HL_foot   0.52 ± 0.76 ⊂ soft[-0.3440, 1.3840]  (HR은 미러: -0.52 ⊂ [-1.3840, 0.3440])
CHIRP_AMPLITUDE: list[float] = [0.14, 0.60, 0.75, 0.76, 0.14, 0.60, 0.75, 0.76]

# 좌우 hip 부호 반전 — 베이스에 걸리는 롤 모멘트를 상쇄한다(매단 리그 흔들림 최소화).
# 관절 순서가 HL(좌) 4개 → HR(우) 4개이므로 HR_hip만 -1.
CHIRP_DIRECTION: list[float] = [
    +1.0,
    +1.0,
    +1.0,
    +1.0,  # HL: hip, thigh, calf, foot
    -1.0,
    +1.0,
    +1.0,
    +1.0,  # HR: hip 부호반전
]

# 기본 chirp 스펙 (PACE ANYmal 예제와 동일한 형태)
DEFAULT_F0_HZ: float = 0.1
DEFAULT_F1_HZ: float = 10.0
DEFAULT_DURATION_S: float = 20.0
DEFAULT_RATE_HZ: float = 500.0  # sysid 제어율 = 실기 명령 발행률 (CONTRACT §8 참고)

# ⚠ f1 = 10 Hz는 **fix_base(=베이스를 world에 용접)** 전제다. 실기 매단 리그에서는 리그 자체의
# 공진(PACE 논문의 매단 리그 공진 관측)이 2 Hz 부근부터 데이터를 오염시키므로, 실기 수집에서는
# f1을 2 Hz로 낮춰야 한다. sim 합성 게이트에서만 10 Hz를 쓴다.

# soft joint position limit [rad] (soft_joint_pos_limit_factor=0.9 반영).
# 2026-08-12 SignFix: 실기 방향 실측에 맞춘 Hind_Leg_URDF3_SignFix 자산 기준 —
# HL_hip/HL_calf/HL_foot/HR_hip/HR_thigh 축 반전으로 좌우가 thigh/calf/foot에서 미러 관계.
# 이걸 넘는 목표각은 sim이 조용히 클램프하고, 실기에서는 관절이 기계 한계에 부딪친다.
# ⚠ `r2s_biped_leg_env_cfg.SOFT_LIMITS_RAD` / `motions.py`의 값과 반드시 일치해야 한다(다른 패키지라 중복).
SOFT_LIMITS: list[tuple[float, float]] = [
    (-0.2340, 0.2340),  # HL_hip   (축 반전 — 대칭이라 값 동일)
    (-0.9555, 2.1855),  # HL_thigh
    (-0.7745, 0.9445),  # HL_calf  (축 반전)
    (-0.3440, 1.3840),  # HL_foot  (축 반전)
    (-0.2340, 0.2340),  # HR_hip   (축 반전 — 대칭이라 값 동일)
    (-2.1855, 0.9555),  # HR_thigh (축 반전)
    (-0.9445, 0.7745),  # HR_calf
    (-1.3840, 0.3440),  # HR_foot
]

# GUI/로그 표시용 짧은 레이블 (관절 순서 동일).
JOINT_LABELS: list[str] = [
    "HL_hip",
    "HL_thigh",
    "HL_calf",
    "HL_foot",
    "HR_hip",
    "HR_thigh",
    "HR_calf",
    "HR_foot",
]


def check_within_soft_limits(targets: list[list[float]]) -> list[tuple[int, float, float]]:
    """궤적이 biped leg soft limit 안에 있는지 검사한다.

    sim 재생과 실기 발행 **양쪽에서 발행 전에** 반드시 부른다.

    Args:
        targets: T×8 목표각 [rad], CONTRACT §2 관절 순서.

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
    모든 관절이 같은 위상을 공유하므로 한 번의 스윕으로 8관절을 동시에 여기한다.

    Args:
        duration: chirp 길이 [s].
        rate_hz: 샘플링/명령 주파수 [Hz].
        f0: 시작 주파수 [Hz].
        f1: 종료 주파수 [Hz].
        center: 관절별 진동 중심 [rad], 길이 8. None이면 :attr:`CHIRP_CENTER`.
        amplitude: 관절별 진폭 [rad], 길이 8. None이면 :attr:`CHIRP_AMPLITUDE`.
        direction: 관절별 부호, 길이 8. None이면 :attr:`CHIRP_DIRECTION`.

    Returns:
        ``(time, targets)`` — ``time``은 길이 T의 시각 [s], ``targets``는 T×8 목표각 [rad]
        (CONTRACT §2 관절 순서).
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
    for j, name in enumerate(JOINT_LABELS):
        lo = min(row[j] for row in traj)
        hi = max(row[j] for row in traj)
        margin = min(lo - SOFT_LIMITS[j][0], SOFT_LIMITS[j][1] - hi)
        print(f"  {name:9s} range=[{lo:+.3f}, {hi:+.3f}]  soft={SOFT_LIMITS[j]}  margin={margin:.4f}")

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""리타게팅된 SMR 모션 클립(`motion_data/*_joints.npz`) 로더 (순수 함수, numpy만).

`gui_controller.py` 의 Motion Playback 패널이 쓰는 재생 프레임 생성기다. ROS/torch/Qt 비의존이라
자가 테스트(`/usr/bin/python3 motion_clips.py`)로 단독 검증할 수 있다 (motions.py 스타일).

관절 순서는 motions.py 와 같은 **leg-major** — HL 4개(hip, thigh, calf, foot) → HR 4개.
npz 의 `joint_names` 는 `_joint` 접미사만 다르며 :func:`load_clip` 이 검증한다.

좌표계는 **모델각(관절각) 하나**다 (2026-08-14 좌표 이관, RL_INTERFACE.md §0). npz `joint_angles`
· soft limit · 발행 프레임이 전부 같은 공간이라 변환이 없다.

⚠ 이전에는 여기서 발행 직전 ``to_raw`` (``q_raw_foot = q_foot + q_calf``, coef=+1) 로 옮겼다.
그 변환은 이제 브리지(``real_runner``)가 전담하고 자기 규약을 ``convention_version=1`` 로 신고한다
(`r2s_udp.R2S_CONVENTION_VERSION`) — 워크스테이션은 관절 좌표 하나로 통일됐다.
"""

from __future__ import annotations

import os

import motions
import numpy as np

# 이 파일과 같은 디렉터리의 motion_data/ — 출처·재생성 커맨드는 motion_data/SOURCES.md.
MOTION_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motion_data")
CLIP_SUFFIX: str = "_joints.npz"

# npz `joint_names` 기대값 — motions.JOINT_NAMES + "_joint".
EXPECTED_JOINT_NAMES: list[str] = [f"{n}_joint" for n in motions.JOINT_NAMES]

# 현재 자세 → 클립 첫 프레임 진입 보간 길이 [s]. ⚠gui_controller.SEQUENCE_DURATION_S 와 일치.
DEFAULT_INTRO_S: float = 1.5

# Loop 재생 시 마지막→첫 프레임을 잇는 **합성 복귀 구간**의 설계 파라미터.
# trot0/walk1 은 주기적 클립이 아니라 seam 이 1.1 rad 나 된다 — 그대로 이으면 50Hz 한 틱에
# 55 rad/s 로, 실기 무부하 속도한계(29.6 rad/s, RL_INTERFACE §5)를 넘는 충격이 된다.
LOOP_BRIDGE_MAX_RAD_S: float = 2.0  # bridge 구간 관절속도 상한 — 이 값으로 길이를 정한다
LOOP_BRIDGE_MIN_S: float = 0.30  # seam 이 작아도 최소 이만큼은 준다
SEAM_WARN_RAD: float = 0.10  # 이 이상이면 "주기적 클립 아님"으로 UI 경고

# 재생 배속 범위 — 상한 1.0 (원속). 실기 안전상 느리게 시작할 수 있게 하한을 둔다.
SPEED_RANGE: tuple[float, float] = (0.25, 1.0)


# ---------------------------------------------------------------------------
# 디렉터리 스캔 / 로드
# ---------------------------------------------------------------------------


def list_clips(directory: str = MOTION_DIR) -> list[str]:
    """`motion_data/` 의 클립 이름(접미사 제거)을 정렬해 반환. 디렉터리가 없으면 빈 리스트."""
    if not os.path.isdir(directory):
        return []
    return sorted(f[: -len(CLIP_SUFFIX)] for f in os.listdir(directory) if f.endswith(CLIP_SUFFIX))


def clip_path(name: str, directory: str = MOTION_DIR) -> str:
    return os.path.join(directory, f"{name}{CLIP_SUFFIX}")


def load_clip(name: str, directory: str = MOTION_DIR) -> dict:
    """클립 하나를 로드하고 관절 순서를 검증한다.

    Returns:
        ``{name, joint_angles (T,8) float64 **모델각**, dt, fps, num_frames, duration_s,
        method, source_urdf}``.

    Raises:
        ValueError: `joint_names` 가 leg-major 기대 순서와 다르거나 shape 이 (T, 8) 이 아닐 때.
    """
    path = clip_path(name, directory)
    with np.load(path, allow_pickle=True) as data:
        names = [str(s) for s in data["joint_names"]]
        if names != EXPECTED_JOINT_NAMES:
            raise ValueError(
                f"{path}: joint_names 불일치 (leg-major 아님?)\n  got={names}\n  want={EXPECTED_JOINT_NAMES}"
            )
        ja = np.asarray(data["joint_angles"], dtype=np.float64)
        if ja.ndim != 2 or ja.shape[1] != motions.NUM_JOINTS:
            raise ValueError(f"{path}: joint_angles shape {ja.shape} — (T, {motions.NUM_JOINTS}) 이어야 한다")
        if ja.shape[0] < 2:
            raise ValueError(f"{path}: 프레임이 {ja.shape[0]}개 — 재생하려면 2개 이상 필요")
        dt = float(data["dt"])
        return {
            "name": name,
            "path": path,
            "joint_angles": ja,
            "dt": dt,
            "fps": float(data["fps"]) if "fps" in data.files else (1.0 / dt),
            "num_frames": int(ja.shape[0]),
            "duration_s": float((ja.shape[0] - 1) * dt),
            "method": str(data["method"]) if "method" in data.files else "",
            "source_urdf": str(data["source_urdf"]) if "source_urdf" in data.files else "",
        }


# ---------------------------------------------------------------------------
# 클램프 / 통계 (전부 모델각 = 발행 좌표)
# ---------------------------------------------------------------------------


def clamp_frames(frames: np.ndarray) -> np.ndarray:
    """모델각 프레임을 `motions.SOFT_LIMITS_RAD` 로 클램프 (`motions.clamp_to_soft` 의 벡터판)."""
    lo = np.array([s[0] for s in motions.SOFT_LIMITS_RAD], dtype=np.float64)
    hi = np.array([s[1] for s in motions.SOFT_LIMITS_RAD], dtype=np.float64)
    return np.clip(np.asarray(frames, dtype=np.float64), lo, hi)


def soft_limit_scan(frames: np.ndarray) -> dict:
    """모델각 프레임의 soft limit 초과 통계.

    Returns:
        ``{num_frames, num_violating, ratio, max_excess, max_excess_joint, per_joint}``.
        `per_joint` 는 초과가 있는 관절만 ``{name, index, frames, max_excess}`` 로 담는다.
    """
    ja = np.asarray(frames, dtype=np.float64)
    lo = np.array([s[0] for s in motions.SOFT_LIMITS_RAD], dtype=np.float64)
    hi = np.array([s[1] for s in motions.SOFT_LIMITS_RAD], dtype=np.float64)
    excess = np.maximum(lo - ja, 0.0) + np.maximum(ja - hi, 0.0)
    bad = excess > 1e-12
    per_joint = [
        {
            "name": motions.JOINT_NAMES[j],
            "index": j,
            "frames": int(bad[:, j].sum()),
            "max_excess": float(excess[:, j].max()),
        }
        for j in range(motions.NUM_JOINTS)
        if bad[:, j].any()
    ]
    per_joint.sort(key=lambda d: -d["max_excess"])
    n = int(ja.shape[0])
    n_bad = int(bad.any(axis=1).sum())
    j_max = int(excess.max(axis=0).argmax())
    return {
        "num_frames": n,
        "num_violating": n_bad,
        "ratio": (n_bad / n) if n else 0.0,
        "max_excess": float(excess.max()) if n else 0.0,
        "max_excess_joint": motions.JOINT_NAMES[j_max],
        "per_joint": per_joint,
    }


def loop_seam(frames: np.ndarray) -> dict:
    """마지막→첫 프레임 불연속(주기성) 측정. ``{max_rad, joint, cyclic}``."""
    ja = np.asarray(frames, dtype=np.float64)
    d = np.abs(ja[0] - ja[-1])
    j = int(d.argmax())
    return {"max_rad": float(d[j]), "joint": motions.JOINT_NAMES[j], "cyclic": bool(d[j] < SEAM_WARN_RAD)}


# foot↔calf 전달기구 커플링 쌍 (leg-major) — **속도 안전 판정에만** 쓴다.
# 발행 좌표는 관절각이고 raw 변환은 브리지가 전담하지만(모듈 docstring 참조), 모터 속도한계는
# raw 축에 걸리므로 판정용 합성이 필요하다.
_COUPLED_CALF_FOOT_FOR_RATE: tuple[tuple[int, int], ...] = ((2, 3), (6, 7))


def _raw_for_rate_check(frames: np.ndarray) -> np.ndarray:
    """속도 판정용 raw 프레임 임시 합성 (``foot += calf``). **발행 경로에는 쓰지 않는다.**"""
    out = np.array(frames, dtype=np.float64, copy=True)
    for c, f in _COUPLED_CALF_FOOT_FOR_RATE:
        out[..., f] += out[..., c]
    return out


def _max_rate(a: dict, b: dict) -> dict:
    """:func:`peak_joint_rate` 결과 둘 중 큰 쪽을 고른다 (관절 축 vs 모터 축)."""
    return a if a["max_rad_s"] >= b["max_rad_s"] else b


def peak_joint_rate(frames: np.ndarray, hz: float) -> dict:
    """프레임 간 최대 관절속도 [rad/s]. ``{max_rad_s, joint}`` — 실기 안전 판단용."""
    ja = np.asarray(frames, dtype=np.float64)
    if ja.shape[0] < 2:
        return {"max_rad_s": 0.0, "joint": motions.JOINT_NAMES[0]}
    v = np.abs(np.diff(ja, axis=0)) * hz
    j = int(v.max(axis=0).argmax())
    return {"max_rad_s": float(v[:, j].max()), "joint": motions.JOINT_NAMES[j]}


# ---------------------------------------------------------------------------
# 리샘플 / 재생 프레임 조립
# ---------------------------------------------------------------------------


def resample(frames: np.ndarray, dt: float, out_hz: float, speed: float = 1.0) -> np.ndarray:
    """소스 프레임을 `out_hz` 균일 그리드로 선형 리샘플 (배속 `speed` 반영).

    소스는 59.988 fps 인데 발행은 50 Hz 다. 소스 프레임을 그대로 두고 인덱싱 주파수만 60 으로
    주면 publisher 의 정수 인덱싱이 6프레임에 1개씩 건너뛰어 불규칙해진다 — 발행률 위의 성분은
    어차피 표현할 수 없으므로 **발행 그리드에 정확히 맞춰** 미리 보간해 둔다.

    Args:
        frames: (T, 8) 소스 프레임.
        dt: 소스 프레임 간격 [s].
        out_hz: 출력 그리드 주파수 [Hz] (= publisher 의 FRAME_HZ).
        speed: 재생 배속 (0.25=4배 느리게 … 1.0=원속). 출력 길이가 1/speed 배가 된다.

    Returns:
        (M, 8) — 첫 프레임은 소스 첫 프레임, 마지막은 소스 마지막 프레임과 정확히 같다.
    """
    ja = np.asarray(frames, dtype=np.float64)
    t_src = ja.shape[0] - 1  # 소스 인덱스 도메인 [0, T-1]
    out_n = max(2, int(round(t_src * dt / max(1e-6, speed) * out_hz)) + 1)
    idx = np.linspace(0.0, float(t_src), out_n)
    src = np.arange(ja.shape[0], dtype=np.float64)
    return np.stack([np.interp(idx, src, ja[:, j]) for j in range(ja.shape[1])], axis=1)


def _lerp_frames(start: np.ndarray, goal: np.ndarray, num: int, include_goal: bool) -> np.ndarray:
    """start→goal 선형 보간 num 프레임. `include_goal=False` 면 goal 직전까지만(중복 방지)."""
    if num <= 0:
        return np.zeros((0, motions.NUM_JOINTS), dtype=np.float64)
    denom = float(num) if include_goal else float(num + 1)
    a = (np.arange(1, num + 1, dtype=np.float64) / denom)[:, None]
    return start[None, :] * (1.0 - a) + goal[None, :] * a


def build_playback(
    clip: dict,
    start_pose: list[float] | np.ndarray,
    out_hz: float,
    speed: float = 1.0,
    loop: bool = False,
    intro_s: float = DEFAULT_INTRO_S,
) -> dict:
    """재생용 프레임 버퍼를 조립한다 — [진입 보간] + [클립] (+ loop 면 [복귀 bridge]).

    ``frames[loop_start:]`` 이 반복 구간이다. 진입 보간은 한 번만 재생된다.
    프레임은 전부 **관절각**(= 발행 규약)이고 soft limit 클램프가 끝난 상태다.

    Args:
        clip: :func:`load_clip` 결과.
        start_pose: 진입 보간 시작 자세 (publisher 의 현재 출력 목표, 관절각).
        out_hz: 프레임 그리드 [Hz] (publisher 의 FRAME_HZ).
        speed: 재생 배속 (`SPEED_RANGE`).
        loop: True 면 마지막→첫 프레임 사이에 합성 복귀 구간을 넣고 반복 재생한다.
        intro_s: 진입 보간 길이 [s].

    Returns:
        ``{frames (N,8) raw, loop_start, num_intro, num_clip, num_bridge, clip_s, bridge_s,
        total_s, scan, seam, peak_rate}``.
    """
    speed = float(min(SPEED_RANGE[1], max(SPEED_RANGE[0], speed)))
    ja = clip["joint_angles"]
    scan = soft_limit_scan(ja)
    seam = loop_seam(ja)

    # 리샘플 → 클램프 순서: 원본 그리드에서 클램프하면 보간이 클램프 경계를 다시 넘을 수 있다.
    body = clamp_frames(resample(ja, clip["dt"], out_hz, speed))
    # ★안전 지표는 **두 축을 모두** 덮어야 한다 (loop bridge 길이·UI 배속 경고가 여기 걸려 있다):
    #   · 관절 축 — 발행 프레임의 변화율. 틱 간 점프 상한이 이 값으로 판정된다.
    #   · 모터(raw) 축 — foot 모터는 q_foot + q_calf 를 돈다(브리지가 옮긴다). 관절각으로만 재면
    #     foot 이 과소평가된다.
    #   둘 중 **큰 쪽**을 쓴다. raw 가 항상 크지는 않다 — foot 과 calf 가 반대로 움직이면 raw 쪽이
    #   상쇄돼 오히려 작아지므로, 한쪽만 보면 다른 쪽을 놓친다.
    rate = _max_rate(peak_joint_rate(body, out_hz), peak_joint_rate(_raw_for_rate_check(body), out_hz))

    bridge = np.zeros((0, motions.NUM_JOINTS), dtype=np.float64)
    if loop:
        # 합성 복귀 구간: 마지막 프레임 → 첫 프레임. 길이는 관절속도 상한으로 정한다.
        # 양 끝점은 클립이 이미 갖고 있으므로 사이 프레임만 만든다(include_goal=False).
        gap = float(np.abs(body[0] - body[-1]).max())
        n_bridge = max(
            int(round(LOOP_BRIDGE_MIN_S * out_hz)),
            int(np.ceil(gap / LOOP_BRIDGE_MAX_RAD_S * out_hz)),
        )
        bridge = clamp_frames(_lerp_frames(body[-1], body[0], n_bridge, include_goal=False))

    loop_body = np.concatenate([body, bridge], axis=0)

    start = np.asarray(start_pose, dtype=np.float64)
    n_intro = max(1, int(round(intro_s * out_hz)))
    # goal 은 반복 구간의 첫 프레임이 담당하므로 보간은 그 직전까지만 만든다.
    intro = _lerp_frames(start, loop_body[0], n_intro - 1, include_goal=False)

    frames = np.concatenate([intro, loop_body], axis=0)
    return {
        "frames": frames,
        "loop_start": int(intro.shape[0]),
        "num_intro": int(intro.shape[0]),
        "num_clip": int(body.shape[0]),
        "num_bridge": int(bridge.shape[0]),
        "clip_s": body.shape[0] / out_hz,
        "bridge_s": bridge.shape[0] / out_hz,
        "total_s": frames.shape[0] / out_hz,
        "scan": scan,
        "seam": seam,
        "peak_rate": rate,
    }


def describe_scan(clip_name: str, scan: dict, seam: dict, rate: dict | None = None) -> str:
    """상태 표시/콘솔용 1줄 요약."""
    parts = [f"{clip_name}: {scan['num_frames']} frames"]
    if scan["num_violating"]:
        worst = scan["per_joint"][0]
        parts.append(
            f"clamped {scan['num_violating']}/{scan['num_frames']} ({100 * scan['ratio']:.0f}%), "
            f"max excess {worst['max_excess']:.3f} rad @{worst['name']}"
        )
    else:
        parts.append("no soft-limit clamping")
    if not seam["cyclic"]:
        parts.append(f"non-cyclic (seam {seam['max_rad']:.2f} rad @{seam['joint']})")
    if rate is not None:
        parts.append(f"peak {rate['max_rad_s']:.1f} rad/s @{rate['joint']}")
    return " | ".join(parts)


if __name__ == "__main__":
    OUT_HZ = 50.0

    # --- 합성 데이터로 순수 함수 검증 ---
    ramp = np.linspace(0.0, 1.0, 61)[:, None] * np.ones((1, motions.NUM_JOINTS))
    rs = resample(ramp, 1.0 / 60.0, OUT_HZ, 1.0)
    assert rs.shape[1] == motions.NUM_JOINTS
    assert abs(rs[0, 0] - 0.0) < 1e-12 and abs(rs[-1, 0] - 1.0) < 1e-12, "리샘플이 끝점을 보존해야 한다"
    assert np.allclose(rs[:, 0], np.linspace(0.0, 1.0, rs.shape[0])), "선형 램프는 리샘플 후에도 선형"
    assert resample(ramp, 1.0 / 60.0, OUT_HZ, 0.25).shape[0] > 3.9 * rs.shape[0], "0.25배속이면 프레임 4배"

    over = np.array([[9.0] * motions.NUM_JOINTS, [-9.0] * motions.NUM_JOINTS])
    cl = clamp_frames(over)
    assert cl[0].tolist() == [s[1] for s in motions.SOFT_LIMITS_RAD]
    assert cl[1].tolist() == [s[0] for s in motions.SOFT_LIMITS_RAD]
    assert np.allclose(cl[0], motions.clamp_to_soft([9.0] * motions.NUM_JOINTS)), "motions.clamp_to_soft 와 동일"

    sc = soft_limit_scan(over)
    assert sc["num_violating"] == 2 and sc["ratio"] == 1.0
    assert soft_limit_scan(np.zeros((5, motions.NUM_JOINTS)))["num_violating"] == 0

    assert loop_seam(np.zeros((4, motions.NUM_JOINTS)))["cyclic"] is True
    assert peak_joint_rate(np.array([[0.0] * 8, [0.1] * 8]), 50.0)["max_rad_s"] == 5.0

    # --- 실제 클립 ---
    clips = list_clips()
    assert clips, f"클립이 없다: {MOTION_DIR}"
    print(f"motion_data: {MOTION_DIR}")
    for name in clips:
        c = load_clip(name)
        assert c["joint_angles"].shape[1] == motions.NUM_JOINTS
        for speed, loop in ((1.0, False), (1.0, True), (0.25, True)):
            pb = build_playback(c, [0.0] * motions.NUM_JOINTS, OUT_HZ, speed=speed, loop=loop)
            fr = pb["frames"]
            assert fr.shape[1] == motions.NUM_JOINTS
            assert pb["loop_start"] == pb["num_intro"] and pb["num_intro"] > 0
            assert fr.shape[0] == pb["num_intro"] + pb["num_clip"] + pb["num_bridge"]
            # 반복 구간 첫 프레임 = 클립 첫 프레임(클램프만 적용 — 좌표 변환 없음)
            want0 = clamp_frames(c["joint_angles"][:1])[0]
            assert np.allclose(fr[pb["loop_start"]], want0, atol=1e-9), "진입 보간의 도착점이 클립 첫 프레임"
            # ★발행 프레임 == 관절각 (2026-08-14 이관): foot 채널에 calf 가 **더해지지 않았음**을
            #   명시적으로 확인한다. 이 검사만이 "커플링 변환이 남아 있는가"를 구별한다
            #   (클램프는 어느 쪽이든 통과시킨다). calf 가 0 이 아닌 프레임에서만 의미가 있다.
            cl0 = clamp_frames(c["joint_angles"][:1])[0]
            if abs(cl0[2]) > 1e-6:
                assert abs(fr[pb["loop_start"], 3] - cl0[3]) < 1e-9, "foot 에 calf 가 더해지면 안 된다"
            assert soft_limit_scan(fr)["num_violating"] == 0, "발행 프레임은 관절각 기준 위반 0"
            # 틱 간 불연속 없음 — loop 이면 반복 이음매(마지막→loop_start)까지 포함해 검사
            steps = np.abs(np.diff(fr, axis=0)).max()
            if loop:
                steps = max(steps, float(np.abs(fr[pb["loop_start"]] - fr[-1]).max()))
            assert steps * OUT_HZ <= 1.05 * max(LOOP_BRIDGE_MAX_RAD_S, pb["peak_rate"]["max_rad_s"]), (
                f"{name} speed={speed} loop={loop}: 틱 간 점프 {steps * OUT_HZ:.2f} rad/s"
            )
        one = build_playback(c, [0.0] * motions.NUM_JOINTS, OUT_HZ, speed=1.0, loop=False)
        slow = build_playback(c, [0.0] * motions.NUM_JOINTS, OUT_HZ, speed=0.25, loop=False)
        lp = build_playback(c, [0.0] * motions.NUM_JOINTS, OUT_HZ, speed=1.0, loop=True)
        print("  " + describe_scan(name, one["scan"], one["seam"], one["peak_rate"]))
        print(
            f"    1.0x: intro {one['num_intro']} + clip {one['num_clip']} = {one['frames'].shape[0]} "
            f"frames ({one['total_s']:.2f} s) | loop bridge {lp['num_bridge']} frames ({lp['bridge_s']:.2f} s)"
            f" | 0.25x peak {slow['peak_rate']['max_rad_s']:.1f} rad/s"
        )
    print("motion_clips OK")

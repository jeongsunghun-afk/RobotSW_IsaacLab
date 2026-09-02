# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""**에피소드 도중 명령이 바뀔 때 걸음이 따라 바뀌는가** — `gait_survey_multienv.py` npz 재분석.

격자 조사는 (명령, 보행)의 **연관**만 본다. 그런데 초기 자세가 걸음을 잠그는 정책에서도,
초기 자세가 명령과 상관되게 만들면 같은 연관이 나온다 — 전환 능력이 0 이어도 그렇다.
`rsi_match_command` 는 정확히 그 상관을 만들므로, 격자만으로는 처방의 성공과 초기화 artifact 를
가를 수 없다.

가르는 방법: **한 env 안에서 명령이 바뀌는 순간**을 골라 앞뒤를 각각 분류한다. `tar_change_time`
이 4~7 s 이고 조사 창이 8 s 라 대부분의 env 에 명령 변경이 정확히 한 번 들어 있다. 격자 분석은
이것들을 `cmd_moved` 로 버려 왔는데, 여기서는 그것만 골라 쓴다.

  걸음이 명령을 따라 바뀐다      → 전환이 실재한다
  걸음이 시작한 대로 잠겨 있다   → 격자의 상관은 초기화 artifact

사용:
    python _workspace/leg/gait_transition_probe.py <npz> [<npz> ...] --win_s 2.0
"""

import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gait_classify import LEGS, MIN_AMP_RAD, _circ_mean_frac, _classify, _inst_phase  # noqa: E402

FALL_H = 0.35
MIN_CYCLES = 3.0
LO_MAX, HI_MIN = 2.0, 2.5  # '저속' / '고속' 으로 볼 명령 경계 [m/s]


def _label(sig_slice: dict, fs: float) -> str:
    """thigh 신호 한 창을 보행 라벨로. 사이클이 모자라면 '-'."""
    if max(float(np.ptp(s)) for s in sig_slice.values()) < MIN_AMP_RAD:
        return "stand"
    ph = {lg: _inst_phase(sig_slice[lg], fs) for lg in LEGS}
    n = len(sig_slice["FL"])
    freq = float(np.polyfit(np.arange(n) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
    if abs(freq) * n / fs < MIN_CYCLES:
        return "-"
    return _classify({lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS})


def _is_gallop(lab: str) -> bool:
    return lab in ("gallop", "bound")


def probe(path: str, win_s: float) -> None:
    d = np.load(path, allow_pickle=True)
    jpos = np.asarray(d["jpos"], dtype=np.float32)
    vxc = np.asarray(d["vx_cmd"])
    hgt, elen = np.asarray(d["height"]), np.asarray(d["ep_len"])
    names = [str(s) for s in d["joint_names"]]
    fs = 1.0 / float(d["dt"])
    T, N, _ = jpos.shape
    w = int(win_s * fs)
    idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}

    up, down, kept = [], [], 0
    for e in range(N):
        c = vxc[:, e]
        # ★ env 하나에서 **변경마다** 표본을 만든다. 예전에는 "변경이 정확히 1회" 인 env 만 썼는데,
        # 재샘플 주기를 4 s 로 줄이면 한 창에 변경이 여러 번 들어와 표본이 전부 버려진다.
        # 각 변경의 앞뒤 창 안에서 명령이 또 움직이지 않는지는 아래 std 검사가 이미 보장한다.
        jumps = np.flatnonzero(np.abs(np.diff(c)) > 1e-3) + 1
        jumps = jumps[(jumps >= w) & (jumps <= T - w)]
        for k in jumps.tolist():
            k = int(k)
            pre_sl, post_sl = slice(k - w, k), slice(k, k + w)
            # 창 안에서 명령이 또 움직이거나, 리셋을 걸치거나, 넘어졌으면 버린다
            if c[pre_sl].std() > 1e-3 or c[post_sl].std() > 1e-3:
                continue
            if (np.diff(elen[k - w : k + w, e]) < 0).any():
                continue
            if (hgt[k - w : k + w, e] < FALL_H).any():
                continue
            lo, hi = float(c[k - 1]), float(c[k])
            if lo <= LO_MAX and hi >= HI_MIN:
                bucket = up
            elif lo >= HI_MIN and hi <= LO_MAX:
                bucket = down
            else:
                continue
            pre = _label({lg: jpos[pre_sl, e, idx[lg]] for lg in LEGS}, fs)
            post = _label({lg: jpos[post_sl, e, idx[lg]] for lg in LEGS}, fs)
            if pre == "-" or post == "-":
                continue
            kept += 1
            bucket.append((lo, hi, pre, post))

    print(f"\n===== {path}")
    print(f"앞뒤 {win_s}s 창이 온전한 명령 변경 표본: {kept}개 "
          f"(저속→고속 {len(up)} · 고속→저속 {len(down)})")

    for tag, rows, want in (("저속→고속", up, True), ("고속→저속", down, False)):
        if not rows:
            print(f"\n[{tag}] 표본 없음")
            continue
        pre_g = sum(_is_gallop(p) for _, _, p, _ in rows)
        post_g = sum(_is_gallop(q) for _, _, _, q in rows)
        # 관심 대상: 바뀌어야 하는 쪽에서 실제로 바뀐 비율
        elig = [r for r in rows if _is_gallop(r[2]) != want]
        moved = [r for r in elig if _is_gallop(r[3]) == want]
        locked = [r for r in rows if _is_gallop(r[2]) == _is_gallop(r[3])]
        print(f"\n[{tag}] n={len(rows)}")
        print(f"  변경 前 gallop {pre_g:4d} ({100 * pre_g / len(rows):5.1f}%)"
              f"  →  변경 後 gallop {post_g:4d} ({100 * post_g / len(rows):5.1f}%)")
        print(f"  걸음 유지(잠김)            {len(locked):4d} / {len(rows)} = {100 * len(locked) / len(rows):5.1f}%")
        if elig:
            arrow = "pace→gallop" if want else "gallop→pace"
            print(f"  바뀌어야 할 표본 중 실제 {arrow:11s} {len(moved):4d} / {len(elig)}"
                  f" = {100 * len(moved) / len(elig):5.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--win_s", type=float, default=2.0, help="변경 앞뒤 분류 창 [s]")
    args = ap.parse_args()
    print("명령 변경 전후로 걸음이 바뀌는지 — 격자의 상관이 전환 능력인지 초기화 artifact 인지 가른다")
    for p in args.npz:
        probe(p, args.win_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

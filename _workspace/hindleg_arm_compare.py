# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""HindLeg run 들을 **같은 iter 끼리** 비교한다. Isaac 불필요 — tfevents 만 읽는다.

★ ``Train/mean_reward`` 는 **에피소드 합**이다. 2026-08-27 종료 수정으로 에피소드가 708 → 70
  step 이 됐으므로 원시값 비교는 무의미하다 — ``reward/eplen`` (step 당) 이 정본이다.
★ iter 900 이전 수치로 판정 금지: `nocouple` 은 1000 에서 −1.13 이다가 1500 에서 +19.19 로
  도약했다 (project_hindleg_calf_feedforward_blocks_training).

실행::

    python _workspace/hindleg_arm_compare.py                     # 기본 arm 세트
    python _workspace/hindleg_arm_compare.py --at 1500 3000      # 특정 iter 만
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

RUNS = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl" / "hindLeg_history_direct"

# (표시명, 디렉터리 glob, 종료수정 여부, 커플링 3항)
ARMS = [
    ("pace0819sym      ", "*_pace0819sym", "전", "ON "),
    ("termfix_termsON  ", "*_termfix_termsON", "후", "ON "),
    ("pace0819sym_nocpl", "*_pace0819sym_nocouple", "전", "OFF"),
    ("termfix_nocouple ", "*_termfix_nocouple", "후", "OFF"),
    ("termfix_reflIcap ", "*_termfix_reflIcap", "후", "ON+캡"),
    ("termfix_only_tr  ", "*_termfix_only_transpose", "후", "전치"),
    ("only_reflI       ", "*_pace0819sym_only_reflI", "전", "reflI"),
    ("only_rawfric     ", "*_pace0819sym_only_rawfric", "전", "rawfr"),
    ("stock baseline   ", "2026-08-13_12-04-06_colmesh_v2_gpu2", "전", "―  "),
]

TAGS = ("Train/mean_reward", "Train/mean_episode_length", "Loss/value", "Policy/mean_noise_std")


def load(pattern: str) -> dict[str, tuple[np.ndarray, np.ndarray]] | None:
    hits = sorted(glob.glob(str(RUNS / pattern)))
    if not hits:
        return None
    ev = sorted(glob.glob(hits[-1] + "/events.out.tfevents*"))
    if not ev:
        return None
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    out = {}
    for t in TAGS:
        if t in have:
            s = ea.Scalars(t)
            out[t] = (np.array([e.step for e in s]), np.array([e.value for e in s]))
    return out or None


def at(series, it: int) -> float:
    """iter it 이하의 마지막 값. ⚠ run 이 아직 그 iter 에 **도달하지 않았으면 nan** —
    마지막 값을 채우면 짧은 run 이 전 열에 같은 수로 퍼져 다 온 것처럼 보인다."""
    if series is None:
        return float("nan")
    steps, vals = series
    if steps.max() < it:
        return float("nan")
    m = steps <= it
    return float(vals[m][-1]) if m.any() else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--at", type=int, nargs="+", default=[300, 500, 900, 1500, 3000, 6000, 12000, 25000, 48000])
    args = ap.parse_args()

    data = {name: load(pat) for name, pat, _, _ in ARMS}
    live = {name: (d[TAGS[0]][0].max() if d and TAGS[0] in d else -1) for name, d in data.items()}

    print(f"\n{'=' * 108}")
    print("HindLeg arm 비교 — ★ 판정은 reward/eplen (step당). Train/mean_reward 는 에피소드 합이다")
    print(f"{'=' * 108}")
    print(f"\n{'arm':19s}{'종료수정':>7s}{'3항':>7s}{'최신 iter':>10s}")
    for name, _, fix, terms in ARMS:
        print(f"{name:19s}{fix:>8s}{terms:>8s}{int(live[name]):>10d}")

    for metric, fmt, label in (
        ("perstep", "8.4f", "reward / episode_length  [★ 정본]"),
        ("Train/mean_reward", "8.2f", "Train/mean_reward  (에피소드 합 — 종료수정 전후 비교 불가)"),
        ("Train/mean_episode_length", "8.1f", "Train/mean_episode_length [step]"),
        ("Loss/value", "8.4f", "Loss/value"),
        ("Policy/mean_noise_std", "8.4f", "Policy/mean_noise_std"),
    ):
        print(f"\n--- {label} ---")
        print(f"{'arm':19s}" + "".join(f"{f'@{i}':>9s}" for i in args.at))
        for name, _, _, _ in ARMS:
            d = data[name]
            row = []
            for i in args.at:
                if d is None:
                    row.append(float("nan"))
                elif metric == "perstep":
                    r = at(d.get(TAGS[0]), i)
                    el = at(d.get(TAGS[1]), i)
                    row.append(r / el if el and el == el and el > 0 else float("nan"))
                else:
                    row.append(at(d.get(metric), i))
            cells = "".join("      ―  " if v != v else f"{v:{fmt}} " for v in row)
            print(f"{name:19s}{cells}")

    # 새 arm 은 구간 스캔 — 마지막 200 iter 의 min/max 를 함께 낸다 (스파이크는 스냅샷으로 안 잡힌다)
    print("\n--- 신규 arm 구간 스캔 (최근 200 iter, reward/eplen) ---")
    for name in ("termfix_reflIcap ", "termfix_only_tr  "):
        d = data[name]
        if d is None or TAGS[0] not in d:
            print(f"{name:19s}  (데이터 없음)")
            continue
        st, rv = d[TAGS[0]]
        _, el = d[TAGS[1]]
        n = min(len(rv), len(el))
        ps = rv[:n] / np.maximum(el[:n], 1e-9)
        m = st[:n] >= st[:n].max() - 200
        vl = d.get("Loss/value")
        vmax = float(vl[1][vl[0] >= vl[0].max() - 200].max()) if vl is not None else float("nan")
        print(f"{name:19s} iter {int(st[:n].max()):5d}  reward/step  min {ps[m].min():7.4f}"
              f"  max {ps[m].max():7.4f}  last {ps[-1]:7.4f}   |  value_loss 구간max {vmax:8.4f}")
    print()


main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보행 전환 판정 그림 — 속도축 선택성 · 초기자세 대조 · 전환 프로브.

`gait_survey_multienv.py` 가 남긴 npz 를 **직접 읽어** 그린다(수치를 손으로 옮기지 않는다 —
표를 베끼다 틀리면 그림과 원자료가 조용히 갈라진다).

세 패널이 각각 하나의 반증을 맡는다:

  왼쪽   : 명령 속도별 gallop 비율. 처방이 통했다면 참조 gallop 속도(2.69 m/s) 위에서만
           올라가는 **단조 증가**여야 한다. 총 gallop% 가 오르는 것은 신호가 아니다 —
           baseline 은 속도와 무관하게 20~25% 로 평평하다.
  가운데 : 같은 정책·같은 명령에서 **초기 자세만** 바꾼 대조(RSI 출발 vs 정지 출발).
           왼쪽 패널이 올라가도 여기서 정지 출발이 0 이면, 그 상관은 전환 능력이 아니라
           **초기화 artifact** 다. 이 패널 하나가 rsimatch 판정을 뒤집었다.
  오른쪽 : 에피소드 도중 명령이 저속→고속으로 바뀐 표본에서 걸음이 따라 바뀐 비율.
           `resample_command_in_episode` 가 켜진 run 에서만 표본이 존재한다.

라벨은 영어로 둔다 — 렌더 환경에 CJK 폰트가 없어 한글이 □ 로 깨진다.

    python _workspace/leg/plot_gait_verdict.py --out <png> \
        --selectivity "label=<npz>" ... --contrast "label:rsi=<npz>,stand=<npz>" ... \
        --transition "label=<npz>" ...
"""

from __future__ import annotations

import argparse
import pathlib
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gait_classify import LEGS, MIN_AMP_RAD, _circ_mean_frac, _classify, _inst_phase  # noqa: E402

FALL_H = 0.35
MIN_CYCLES = 3.0
REF_GALLOP_MIN = 2.69  # 참조 클립상 gallop 이 시작되는 평균 전진속도 [m/s]
VX_BINS = [0.0, 0.5, 1.5, 2.5, 3.5]
LO_MAX, HI_MIN = 2.0, 2.5


def _labels(path: str, win_s: float = 2.0):
    """npz 한 개 → (env별 보행 라벨, 창 끝 명령속도, 사용가능 마스크)."""
    d = np.load(path, allow_pickle=True)
    jpos = np.asarray(d["jpos"], dtype=np.float32)
    vxc, hgt, elen = np.asarray(d["vx_cmd"]), np.asarray(d["height"]), np.asarray(d["ep_len"])
    names = [str(s) for s in d["joint_names"]]
    fs = 1.0 / float(d["dt"])
    T, N, _ = jpos.shape
    w = int(win_s * fs)
    sl = slice(T - w, T)
    idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}

    keep = ~(
        (np.diff(elen[sl], axis=0) < 0).any(axis=0)
        | (hgt[sl] < FALL_H).any(axis=0)
        | (vxc[sl].std(axis=0) > 1e-3)
    )
    lab = np.array(["-"] * N, dtype=object)
    for e in np.flatnonzero(keep):
        sig = {lg: jpos[sl, e, idx[lg]] for lg in LEGS}
        if max(float(np.ptp(s)) for s in sig.values()) < MIN_AMP_RAD:
            lab[e] = "stand"
            continue
        ph = {lg: _inst_phase(sig[lg], fs) for lg in LEGS}
        freq = float(np.polyfit(np.arange(w) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
        if abs(freq) * win_s < MIN_CYCLES:
            continue
        lab[e] = _classify({lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS})
    return lab, vxc[T - 1], keep


def _gallop_by_speed(path: str):
    """명령 속도 구간별 gallop 비율 [%] 과 표본 수."""
    lab, vx, keep = _labels(path)
    pct, ns = [], []
    for lo, hi in zip(VX_BINS[:-1], VX_BINS[1:]):
        m = keep & (vx >= lo) & (vx < hi)
        labs = [x for x in lab[m] if x != "-"]
        if len(labs) < 5:
            pct.append(np.nan)
            ns.append(len(labs))
            continue
        c = Counter(labs)
        pct.append(100 * (c["gallop"] + c["bound"]) / len(labs))
        ns.append(len(labs))
    return np.array(pct), ns


GAIT_ORDER = ["pace", "trot", "gallop", "bound", "other", "stand"]
GAIT_COLOR = {"pace": "#2a9d8f", "trot": "#e9c46a", "gallop": "#d1495b",
              "bound": "#9d4edd", "other": "#adb5bd", "stand": "#495057"}


def _composition(path: str):
    """속도 구간별 보행 구성 [%] — trot 과 pace 를 절대 묶지 않는다.

    둘은 앞/뒤 위상차가 (0.5,0.5) 로 같고 대각이냐 동측이냐로만 갈린다. 묶어 버리면
    "pace 로 붕괴" 와 "trot 으로 이동" 이 같은 그림이 된다.
    """
    lab, vx, keep = _labels(path)
    rows = []
    for lo, hi in zip(VX_BINS[:-1], VX_BINS[1:]):
        m = keep & (vx >= lo) & (vx < hi)
        labs = [x for x in lab[m] if x != "-"]
        n = len(labs)
        c = Counter(labs)
        rows.append((f"{lo:.1f}-{hi:.1f}", n,
                     {g: (100 * c[g] / n if n else 0.0) for g in GAIT_ORDER}))
    return rows


HIGH_BIN = VX_BINS.index(2.5)  # [2.5, 3.5) 칸의 인덱스 — VX_BINS 를 고쳐도 따라간다


def _gallop_high(path: str) -> float:
    """고속 칸(2.5~3.5) 의 gallop 비율 [%] — 초기자세 대조에 쓰는 한 숫자."""
    return float(_gallop_by_speed(path)[0][HIGH_BIN])


def _transition(path: str, win_s: float = 2.0, up: bool = True):
    """명령 변경 표본에서 (전환한 수, 전환 가능했던 수).

    `up=True` 는 저속→고속(pace 가 gallop 이 되어야 하는 쪽 — 이 실험이 원하는 전환),
    `up=False` 는 고속→저속(gallop 이 pace 가 되어야 하는 쪽)이다. 아래쪽 방향은
    프로브가 실제로 전환을 감지할 수 있음을 보이는 양성 대조 역할을 한다.
    """
    d = np.load(path, allow_pickle=True)
    jpos = np.asarray(d["jpos"], dtype=np.float32)
    vxc, hgt, elen = np.asarray(d["vx_cmd"]), np.asarray(d["height"]), np.asarray(d["ep_len"])
    names = [str(s) for s in d["joint_names"]]
    fs = 1.0 / float(d["dt"])
    T, N, _ = jpos.shape
    w = int(win_s * fs)
    idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}

    def lab(sl, e):
        sig = {lg: jpos[sl, e, idx[lg]] for lg in LEGS}
        if max(float(np.ptp(s)) for s in sig.values()) < MIN_AMP_RAD:
            return "stand"
        ph = {lg: _inst_phase(sig[lg], fs) for lg in LEGS}
        n = sl.stop - sl.start
        freq = float(np.polyfit(np.arange(n) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
        if abs(freq) * n / fs < MIN_CYCLES:
            return "-"
        return _classify({lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS})

    moved = elig = 0
    for e in range(N):
        c = vxc[:, e]
        # 변경마다 표본을 만든다 — 재샘플 주기가 짧으면 한 창에 변경이 여러 번 들어오는데,
        # "정확히 1회" 로 거르면 그런 run 의 표본이 통째로 사라진다(4 s 주기에서 전멸).
        j = np.flatnonzero(np.abs(np.diff(c)) > 1e-3) + 1
        for k in j[(j >= w) & (j <= T - w)].tolist():
            k = int(k)
            pre_sl, post_sl = slice(k - w, k), slice(k, k + w)
            if c[pre_sl].std() > 1e-3 or c[post_sl].std() > 1e-3:
                continue
            if (np.diff(elen[k - w : k + w, e]) < 0).any() or (hgt[k - w : k + w, e] < FALL_H).any():
                continue
            lo_hi = float(c[k - 1]) <= LO_MAX and float(c[k]) >= HI_MIN
            hi_lo = float(c[k - 1]) >= HI_MIN and float(c[k]) <= LO_MAX
            if not (lo_hi if up else hi_lo):
                continue
            a, b = lab(pre_sl, e), lab(post_sl, e)
            if a == "-" or b == "-":
                continue
            # "바뀔 수 있었던" 표본만 분모로 센다 — 이미 목표 걸음이면 전환할 게 없다.
            want = up  # 저속→고속이면 gallop 이 되어야, 고속→저속이면 pace 가 되어야 한다
            if (a in ("gallop", "bound")) == want:
                continue
            elig += 1
            moved += (b in ("gallop", "bound")) == want
    return moved, elig


def _spec(s: str) -> tuple[str, str]:
    label, _, path = s.partition("=")
    return label, path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Gait transition verdict")
    ap.add_argument("--selectivity", nargs="*", default=[], help="라벨=npz (왼쪽 패널)")
    ap.add_argument("--contrast", nargs="*", default=[], help="라벨=rsi_npz,stand_npz (가운데 패널)")
    ap.add_argument("--transition", nargs="*", default=[], help="라벨=npz (오른쪽 패널)")
    ap.add_argument("--composition", nargs="*", default=[], help="라벨=npz — 속도구간별 보행 구성 누적막대")
    args = ap.parse_args()

    n_panels = sum(bool(x) for x in (args.selectivity, args.contrast, args.transition, args.composition))
    fig, axes = plt.subplots(1, max(n_panels, 1), figsize=(5.2 * max(n_panels, 1), 4.3))
    axes = np.atleast_1d(axes)
    ax_i = 0
    centers = [(a + b) / 2 for a, b in zip(VX_BINS[:-1], VX_BINS[1:])]
    colors = ["#888888", "#2a9d8f", "#d1495b", "#e9c46a", "#457b9d"]

    if args.selectivity:
        ax = axes[ax_i]
        ax_i += 1
        for i, s in enumerate(args.selectivity):
            label, path = _spec(s)
            pct, ns = _gallop_by_speed(path)
            ax.plot(centers, pct, "o-", color=colors[i % len(colors)], label=label, lw=2, ms=6)
            print(f"[selectivity] {label:28s} " + "  ".join(f"{p:5.1f}%(n={n})" for p, n in zip(pct, ns)))
        ax.axvline(REF_GALLOP_MIN, color="k", ls=":", lw=1.2)
        # 곡선이 오른쪽 위로 치솟으므로 주석은 아래쪽에 둔다 — 위에 두면 데이터와 겹친다.
        lo_y, hi_y = ax.get_ylim()
        ax.text(REF_GALLOP_MIN - 0.06, lo_y + 0.04 * (hi_y - lo_y), "reference\ngallop speed",
                fontsize=7.5, va="bottom", ha="right", color="#333333")
        ax.set_xlabel("commanded forward speed [m/s]")
        ax.set_ylabel("gallop / bound [%]")
        ax.set_title("Is gallop selective for speed?", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    if args.contrast:
        ax = axes[ax_i]
        ax_i += 1
        labels, rsi_v, stand_v = [], [], []
        for s in args.contrast:
            label, _, paths = s.partition("=")
            rsi_p, stand_p = paths.split(",")
            labels.append(label)
            rsi_v.append(_gallop_high(rsi_p))
            stand_v.append(_gallop_high(stand_p))
            print(f"[contrast] {label:28s} RSI {rsi_v[-1]:5.1f}%  stand {stand_v[-1]:5.1f}%")
        x = np.arange(len(labels))
        ax.bar(x - 0.19, rsi_v, 0.38, color="#2a9d8f", label="reference-frame start")
        ax.bar(x + 0.19, stand_v, 0.38, color="#888888", label="standing start")
        for xi, (r, st) in enumerate(zip(rsi_v, stand_v)):
            ax.text(xi - 0.19, r + 1.5, f"{r:.1f}", ha="center", fontsize=8.5)
            ax.text(xi + 0.19, st + 1.5, f"{st:.1f}", ha="center", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("gallop / bound at cmd 2.5-3.5 [%]")
        ax.set_title("Same policy, same commands:\nonly the initial pose differs", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    if args.transition:
        ax = axes[ax_i]
        ax_i += 1
        labels, up_pct, dn_pct, up_nt, dn_nt = [], [], [], [], []
        for s in args.transition:
            label, path = _spec(s)
            mu, eu = _transition(path, up=True)
            md, ed = _transition(path, up=False)
            labels.append(label)
            up_pct.append(100 * mu / eu if eu else 0.0)
            dn_pct.append(100 * md / ed if ed else 0.0)
            up_nt.append(f"{mu}/{eu}" if eu else "no samples")
            dn_nt.append(f"{md}/{ed}" if ed else "no samples")
            print(f"[transition] {label:24s} up {up_nt[-1]:>9s}   down {dn_nt[-1]:>9s}")
        x = np.arange(len(labels))
        ax.bar(x - 0.19, up_pct, 0.38, color="#d1495b", label="low -> high cmd:\nbecame gallop")
        ax.bar(x + 0.19, dn_pct, 0.38, color="#457b9d", label="high -> low cmd:\nbecame pace")
        for xi in range(len(labels)):
            ax.text(xi - 0.19, up_pct[xi] + 2, up_nt[xi], ha="center", fontsize=7.5)
            ax.text(xi + 0.19, dn_pct[xi] + 2, dn_nt[xi], ha="center", fontsize=7.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("gait followed the command [%]")
        ax.set_title("Does gait follow a mid-episode\ncommand change?", fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, axis="y")

    if args.composition:
        ax = axes[ax_i]
        ax_i += 1
        xs, bottoms, ticks = [], None, []
        blocks = []
        for si, spec in enumerate(args.composition):
            label, path = _spec(spec)
            rows = _composition(path)
            print(f"[composition] {label}")
            for name, n, pct in rows:
                print("    " + f"{name:>9s} n={n:5d} " + " ".join(f"{g}={pct[g]:5.1f}" for g in GAIT_ORDER))
                ticks.append(f"{name}\n{label}" if len(args.composition) > 1 else name)
                blocks.append(pct)
        x = np.arange(len(blocks))
        bottoms = np.zeros(len(blocks))
        for g in GAIT_ORDER:
            vals = np.array([b[g] for b in blocks])
            ax.bar(x, vals, 0.72, bottom=bottoms, color=GAIT_COLOR[g], label=g)
            bottoms += vals
        ax.set_xticks(x)
        ax.set_xticklabels(ticks, fontsize=7)
        ax.set_ylabel("share of rollouts [%]")
        ax.set_xlabel("commanded forward speed [m/s]")
        ax.set_title("What gait, exactly?\n(trot and pace kept apart)", fontsize=11)
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(0, 100)

    fig.suptitle(args.title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\n저장: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

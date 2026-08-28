# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""램프 npz 에서 **명령 단계별 보행 종류**(trot/pace/bound/gallop/pronk)를 분류한다.

달성률·duty 는 "얼마나 잘 따라갔나"만 보고 **무슨 걸음으로** 갔는지는 못 본다. 그런데 이 프로젝트는
같은 달성률에서 걸음 종류가 갈린 전례가 있다(go2 의 `cmd 4.0` 4Hz trot 회귀).

각 hold 구간에서 네 다리 thigh 신호의 **순시 위상**(Hilbert 해석신호)을 구하고, FL 기준 위상차를
사이클 분율 [0,1) 로 원형 평균한다. 다리 기호는 F/H(앞뒤) × L/R(좌우).

```
보행      phi_FR   phi_HL   phi_HR      특징
trot       0.5      0.5      0.0        대각선 쌍이 함께
pace       0.5      0.0      0.5        같은 쪽 앞뒤가 함께
bound      0.0      0.5      0.5        앞쌍 정확히 동시, 뒤쌍 정확히 동시
gallop     ~0.15    ~0.5     ~0.65      앞쌍·뒤쌍이 *거의* 동시(선행 다리가 조금 앞선다)
pronk      0.0      0.0      0.0        네 다리 동시
```

★ gallop 과 bound 의 차이는 **앞쌍 위상차가 0 이 아니라 작게 있다**는 것뿐이다. 그래서 앞쌍/뒤쌍
위상차를 따로 내고 둘 다 `GALLOP_LO~GALLOP_HI` 범위면 gallop 으로 본다.

★★ **FFT 한 빈의 위상을 쓰면 안 된다.** hold 1.8 s 면 빈 간격이 0.56 Hz 라 1.94~2.50 Hz 가 전부
같은 빈으로 뭉개지고, 실제로 세 팔·세 명령이 모두 "2.20 Hz"로 똑같이 나왔다(첫 판본의 오류).
Hilbert 순시 위상은 빈 격자에 묶이지 않으므로 이 문제가 없다. 주파수도 위상의 기울기로 낸다.

거르지 않으면 결과가 거짓말을 하는 것 두 가지:

1. **넘어진 롤아웃** — 누워서 버둥거려도 순시 위상은 나오므로 뭔가로 분류된다.
   판정은 `ramp_fall_probe` 에 위임한다.
2. **사이클 수 하한** — 구간이 짧으면(저속에서 특히) 위상차 평균이 무의미하다.
   `MIN_CYCLES` 미만은 분류하지 않고 `-` 로 남긴다.

사용:
    ./isaaclab.sh -p _workspace/leg/gait_classify.py "라벨=글롭" ["라벨=글롭" ...] [--phases]
"""

import argparse
import glob
import pathlib
import sys

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from ramp_fall_probe import _probe as _fall_probe  # noqa: E402
from ramp_fall_probe import _verdict as _fall_verdict  # noqa: E402

LEGS = ("FL", "FR", "HL", "HR")
MIN_CYCLES = 3.0  # 이보다 짧은 구간은 위상 추정 불가로 본다
MIN_AMP_RAD = 0.05  # thigh 진폭이 이보다 작으면 "서 있음"으로 본다 [rad]
GALLOP_LO, GALLOP_HI = 0.06, 0.30  # 앞/뒤 쌍 위상차가 이 범위면 gallop (0=bound, 0.5=trot/pace)
BAND_HZ = (0.7, 8.0)  # 보행 기본파가 들어 있는 대역 [Hz]


def _segments(cmd, target, tol=1e-3, min_len=20):
    idx = np.flatnonzero(np.abs(cmd - target) < tol)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [g for g in groups if g.size > min_len]


def _pick_segment(cmd, target, peak):
    """상승 구간의 hold. 정점 명령이면 정점을 품은 구간을 쓴다(상승/하강 구분이 없다)."""
    segs = _segments(cmd, target)
    if not segs:
        return None
    for g in segs:
        if g[0] <= peak <= g[-1]:
            return g
    up = [g for g in segs if g[0] < peak]
    return up[0] if up else None


def _inst_phase(sig: np.ndarray, fs: float) -> np.ndarray:
    """대역통과 후 Hilbert 순시 위상 [rad]."""
    b, a = butter(2, [BAND_HZ[0] / (fs / 2), BAND_HZ[1] / (fs / 2)], btype="band")
    return np.unwrap(np.angle(hilbert(filtfilt(b, a, sig - sig.mean()))))


def _circ_mean_frac(diff_rad: np.ndarray) -> float:
    """위상차 시계열 → 원형 평균 [cycle, 0~1)."""
    z = np.exp(1j * diff_rad).mean()
    return float(np.angle(z) / (2 * np.pi)) % 1.0


def _circ_dist(a: float, b: float) -> float:
    d = abs(a - b) % 1.0
    return min(d, 1.0 - d)


def _classify(phi: dict[str, float]) -> str:
    front = _circ_dist(phi["FL"], phi["FR"])
    hind = _circ_dist(phi["HL"], phi["HR"])
    diag = _circ_dist(phi["FL"], phi["HR"])
    ipsi = _circ_dist(phi["FL"], phi["HL"])

    if front < GALLOP_LO and hind < GALLOP_LO:
        return "pronk" if ipsi < 0.12 else "bound"
    if GALLOP_LO <= front <= GALLOP_HI and GALLOP_LO <= hind <= GALLOP_HI:
        return "gallop"
    if front > 0.35 and hind > 0.35:
        return "trot" if diag < 0.15 else ("pace" if ipsi < 0.15 else "other")
    return "other"


def probe(path: str, targets, tail: float):
    d = np.load(path)
    cmd = np.asarray(d["vx_cmd"], dtype=float).ravel()
    jpos = np.asarray(d["jpos"], dtype=float)
    names = [str(s) for s in d["joint_names"]]
    fs = 1.0 / float(d["dt"])
    peak = int(np.argmax(cmd))
    idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}

    out = {}
    for tgt in targets:
        g = _pick_segment(cmd, tgt, peak)
        if g is None:
            continue
        g = g[int(len(g) * (1.0 - tail)) :]
        sig = {lg: jpos[g, idx[lg]] for lg in LEGS}
        if max(float(np.ptp(s)) for s in sig.values()) < MIN_AMP_RAD:
            out[tgt] = ("stand", 0.0, 0.0, None)
            continue
        ph = {lg: _inst_phase(sig[lg], fs) for lg in LEGS}
        # 주파수 = FL 순시 위상의 평균 기울기
        freq = float(np.polyfit(np.arange(len(g)) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
        cycles = abs(freq) * len(g) / fs
        if cycles < MIN_CYCLES:
            out[tgt] = ("-", freq, cycles, None)
            continue
        phi = {lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS}
        out[tgt] = (_classify(phi), freq, cycles, phi)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="라벨=글롭")
    ap.add_argument("--cmds", type=float, nargs="+", default=[0.5, 1.0, 2.0, 3.0, 3.5, 4.0])
    ap.add_argument("--tail", type=float, default=0.6, help="hold 구간 뒤쪽 사용 비율")
    ap.add_argument("--keep_fallen", action="store_true")
    ap.add_argument("--phases", action="store_true", help="롤아웃별 위상차(FR/HL/HR)까지 출력")
    args = ap.parse_args()

    print(f"상승 hold 별 보행 분류 · 사이클 <{MIN_CYCLES} 이면 '-' · thigh 진폭 <{MIN_AMP_RAD} rad 이면 stand")
    print(f"gallop = 앞쌍·뒤쌍 위상차가 둘 다 {GALLOP_LO}~{GALLOP_HI} cycle (0=bound, 0.5=trot/pace)")
    print("아래 줄의 Hz 는 Hilbert 순시 위상 기울기 — FFT 빈에 묶이지 않는다\n")
    hdr = "run".ljust(24) + "".join(f"cmd{c}".rjust(12) for c in args.cmds)
    print(hdr)
    print("-" * len(hdr))
    for spec in args.specs:
        label, pattern = spec.split("=", 1)
        rows, dropped = [], 0
        for p in sorted(glob.glob(pattern)):
            if not args.keep_fallen and _fall_verdict(_fall_probe(p, 0.3, 300)) == "FALL":
                dropped += 1
                continue
            rows.append((p, probe(p, args.cmds, args.tail)))
        if not rows:
            print(f"{label:24s} (남은 npz 없음)")
            continue
        line = label.ljust(24)
        for c in args.cmds:
            vals = [r[c][0] for _, r in rows if c in r]
            if not vals:
                line += "-".rjust(12)
                continue
            top = max(set(vals), key=vals.count)
            line += f"{top} {vals.count(top)}/{len(vals)}".rjust(12)
        print(line + (f"   (붕괴 {dropped}개 제외)" if dropped else ""))
        fl = "".ljust(24)
        for c in args.cmds:
            fs_ = [r[c][1] for _, r in rows if c in r and r[c][0] not in ("-", "stand")]
            fl += (f"{np.mean(fs_):.2f}Hz" if fs_ else "-").rjust(12)
        print(fl)
        if args.phases:
            for p, r in rows:
                for c in args.cmds:
                    if c in r and r[c][3]:
                        phi = r[c][3]
                        print(
                            f"    {pathlib.Path(p).parent.name:22s} cmd{c}: "
                            f"FR {phi['FR']:.2f}  HL {phi['HL']:.2f}  HR {phi['HR']:.2f}"
                            f"  ({r[c][0]}, {r[c][1]:.2f}Hz, {r[c][2]:.1f}cyc)"
                        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

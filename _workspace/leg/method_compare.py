# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""imitation 방법별 램프 결과를 '능력 축을 분리해서' 비교한다.

한 개의 평균 달성률로 방법 순위를 매기면 안 된다. `ramp_summary.py` 의 평균은 상승 구간만
집계하므로 사실상 '정지에서 가속하는 능력'이고, 저속 한 점(cmd 0.5)이 값을 지배한다.
그래서 여기서는 세 축을 따로 낸다.

    steady : cmd 2.0~4.0 상승 구간 달성률   — 정상 주행 추종 능력
    start  : cmd 0.5 상승 구간 달성률       — 정지에서 보행으로 진입하는 능력
    skew   : hip 좌우 합의 절대값 (앞+뒤)   — 자세 대칭성 (0 이 대칭)

start 는 duty(보행 중인 스텝 비율)와 함께 낸다. duty 가 0 이면 '느리게 걷는 데 실패'가 아니라
'아예 걷지 않음'이라 원인이 다르다.
"""

import glob
import sys

import numpy as np

PAIRS = [("FL_hip_joint", "FR_hip_joint"), ("HL_hip_joint", "HR_hip_joint")]
UP_05 = 1  # vx_profile 인덱스: 상승 0.5
STEADY = [4, 5, 6, 7, 8]  # 2.0 2.5 3.0 3.5 4.0


def per_rollout(path):
    d = np.load(path)
    prof = d["vx_profile"]
    dt, hold, ramp = float(d["dt"]), float(d["hold_s"]), float(d["ramp_s"])
    span, rs = int(round((hold + ramp) / dt)), int(round(ramp / dt))
    names = [str(x) for x in d["joint_names"]]
    w = np.sqrt((d["jvel"] ** 2).mean(axis=1))
    n = len(d["vx"])

    def win(i):
        return min(i * span + rs, n), min((i + 1) * span, n)

    s, e = win(UP_05)
    start = float(np.median(d["vx"][s:e])) / 0.5
    duty = float((w[s:e] > 0.4).mean())

    ach, sk = [], []
    for i in range(1, (len(prof) + 1) // 2):
        s, e = win(i)
        if s >= e:
            continue
        ach.append((float(prof[i]), float(np.median(d["vx"][s:e])) / float(prof[i])))
        j = d["jpos"][s:e]
        sk.append(abs(j[:, names.index(PAIRS[0][0])].mean() + j[:, names.index(PAIRS[0][1])].mean())
                  + abs(j[:, names.index(PAIRS[1][0])].mean() + j[:, names.index(PAIRS[1][1])].mean()))
    lv = np.array([a[0] for a in ach])
    av = np.array([a[1] for a in ach])
    steady = float(np.mean(av[(lv >= 2.0) & (lv <= 4.0)]))
    return steady, start, duty, float(np.mean(sk)), float(d["px"][-1])


def _cond(path):
    """측정 조건 태그. priv_explicit 를 GT 로 줬는지 estimator 로 줬는지가 성능을 바꾼다.

    구 npz 에는 `use_estimator` 키가 없으므로 GT 로 간주한다.
    """
    d = np.load(path)
    try:
        return "EST" if bool(d["use_estimator"]) else "GT"
    except KeyError:
        return "GT"


def summarize(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    conds = {_cond(p) for p in paths}
    if len(conds) > 1:
        raise SystemExit(f"조건이 섞였다({conds}): {pattern} — GT 와 estimator 측정을 한 표에 넣으면 안 된다.")
    r = np.array([per_rollout(p) for p in paths])
    return dict(steady=np.median(r[:, 0]), start=np.median(r[:, 1]), duty=np.median(r[:, 2]),
                skew=np.median(r[:, 3]), disp=np.median(r[:, 4]), n=len(paths),
                cond=conds.pop(), start_all=sorted(r[:, 1]))


def main(items):
    hdr = (f"{'방법':22s} {'cond':>4s} {'steady':>7s} {'start':>7s} {'duty':>6s} {'skew':>6s} "
           f"{'변위':>7s} {'n':>2s}   상승0.5 롤아웃별")
    print(hdr)
    print("-" * (len(hdr) + 6))
    for it in items:
        lab, pat = it.split("=", 1)
        s = summarize(pat)
        if s is None:
            print(f"{lab:22s}  (데이터 없음: {pat})")
            continue
        allv = " ".join(f"{100 * v:3.0f}" for v in s["start_all"])
        print(f"{lab:22s} {s['cond']:>4s} {100 * s['steady']:6.1f}% {100 * s['start']:6.1f}% {s['duty']:6.2f} "
              f"{s['skew']:6.3f} {s['disp']:6.1f}m {s['n']:2d}   [{allv}]")
    print("\ncond: GT = env 의 ground-truth base 속도(특권 정보) / EST = estimator 추정 (실기 배포 조건)")


if __name__ == "__main__":
    main(sys.argv[1:])

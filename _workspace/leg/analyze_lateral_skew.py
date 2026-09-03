# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""보행의 좌우 쏠림(비대칭) 정량화.

달성률 지표는 body-frame vx 하나만 보므로 "앞다리·뒷다리가 각각 한쪽으로 쏠린" 자세를
전혀 잡아내지 못한다. 램프 npz의 관절각으로 다음 세 가지 DC 편향을 잰다.

  waist   : FB_waist_joint 평균 — 몸통이 상시 비틀려 있는가
  hip 쌍  : 앞/뒤 hip 좌우 합 — 다리 쌍이 통째로 한쪽으로 벌어져 있는가
  lateral : body-frame vy 평균과 y 변위 — 실제로 옆으로 흐르는가

부호 규약은 좌우 미러이므로, 대칭이면 L + R ≈ 0. 합이 0에서 벗어난 값이 곧 쏠림이다.
"""

import glob

import numpy as np

W = "_workspace/leg"
R = "reports/leg_imitation"

RUNS = [
    ("ds14(08-10, 사용자 선호)", f"{W}/ds14_0810_final_*/ramp_data.npz"),
    ("vmax32", f"{W}/ds14_vmax32_final_[0-9]/ramp_data.npz"),
    ("trot110150", f"{W}/trot110150_final_*/ramp_data.npz"),
    ("waistfix", f"{W}/waistfix_final_*/ramp_data.npz"),
    ("waistfix+wcmd", f"{W}/waistfix_wcmd_final_*/ramp_data.npz"),
    ("wcmd(보고서 승자)", f"{W}/wcmd_final_*/ramp_data.npz"),
]

PAIRS = [("앞 hip", "FL_hip_joint", "FR_hip_joint"), ("뒤 hip", "HL_hip_joint", "HR_hip_joint")]


# 롤아웃 프로토콜이 다르면 쏠림을 비교할 수 없다. heading_hold 가 꺼진 램프는 방향이 자유롭게
# 흘러 그 회전이 hip 좌우 편향으로 둔갑한다(실측: 같은 정책이 heading_hold 유무로 앞hip 0.033 vs
# 0.093). 그래서 비교 전에 이 키들이 모든 npz 에서 같은지 확인하고, 다르면 표를 내지 않는다.
PROTOCOL_KEYS = ("heading_hold", "hold_s", "ramp_s", "reset_strategy")


def protocol(path):
    d = np.load(path, allow_pickle=True)
    out = {}
    for k in PROTOCOL_KEYS:
        out[k] = d[k].item() if k in d.files and d[k].shape == () else (str(d[k]) if k in d.files else None)
    out["vx_max"] = round(float(np.max(d["vx_cmd"])), 2)
    return out


def stats(path):
    d = np.load(path)
    names = [str(x) for x in d["joint_names"]]
    j = d["jpos"]
    mv = np.abs(d["vx"]) > 0.3  # 실제로 이동 중인 구간만
    out = {"waist_dc": float(j[mv, names.index("FB_waist_joint")].mean())}
    for lbl, ln, rn in PAIRS:
        l, r = j[mv, names.index(ln)], j[mv, names.index(rn)]
        out[lbl] = float(l.mean() + r.mean())
        out[lbl + "_L"] = float(l.mean())
        out[lbl + "_R"] = float(r.mean())
    out["vy_dc"] = float(d["vy"][mv].mean())
    out["py_end"] = float(d["py"][-1])
    out["px_end"] = float(d["px"][-1])
    return out


print(f"{'run':24s} {'waist DC':>9s} {'앞hip L+R':>10s} {'뒤hip L+R':>10s} {'vy DC':>8s} {'py 끝':>8s} {'px 끝':>8s}  n")
print("-" * 92)
rows = {}
protocols = {}
for lbl, pat in RUNS:
    paths = sorted(glob.glob(pat))
    if not paths:
        print(f"{lbl:24s}  (데이터 없음: {pat})")
        continue
    protos = {tuple(sorted(protocol(p).items())) for p in paths}
    if len(protos) > 1:
        print(f"{lbl:24s}  ** 롤아웃끼리 프로토콜이 다르다 — 비교 불가 **")
        continue
    protocols[lbl] = dict(next(iter(protos)))
    s = [stats(p) for p in paths]
    med = {k: float(np.median([x[k] for x in s])) for k in s[0]}
    rows[lbl] = (med, s)
    print(
        f"{lbl:24s} {med['waist_dc']:+9.3f} {med['앞 hip']:+10.3f} {med['뒤 hip']:+10.3f}"
        f" {med['vy_dc']:+8.3f} {med['py_end']:+8.1f} {med['px_end']:8.1f}  {len(s)}"
    )

if len({tuple(sorted(v.items())) for v in protocols.values()}) > 1:
    print("\n** 경고: run 마다 램프 프로토콜이 다르다 — 위 표를 run 간 비교에 쓰면 안 된다 **")
    for lbl, v in protocols.items():
        print(f"   {lbl:24s} " + "  ".join(f"{k}={v[k]}" for k in PROTOCOL_KEYS))

print("\n[좌우 개별 hip 평균 — 미러 규약이면 L ≈ -R]")
for lbl, (med, _) in rows.items():
    print(
        f"  {lbl:24s} FL{med['앞 hip_L']:+.3f} FR{med['앞 hip_R']:+.3f} | "
        f"HL{med['뒤 hip_L']:+.3f} HR{med['뒤 hip_R']:+.3f}"
    )

print("\n[롤아웃별 편차 — 쏠림이 정책 고유인지 롤아웃 우연인지]")
for lbl, (_, s) in rows.items():
    if len(s) > 1:
        w = [x["waist_dc"] for x in s]
        f = [x["앞 hip"] for x in s]
        print(f"  {lbl:24s} waist {[f'{v:+.3f}' for v in w]}  앞hip {[f'{v:+.3f}' for v in f]}")

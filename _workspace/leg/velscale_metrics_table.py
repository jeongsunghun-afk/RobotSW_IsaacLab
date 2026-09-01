# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""velscale 50k 판정 수치표 — GT/estimator × 붕괴 포함/제외 를 한 파일로 낸다.

붕괴를 포함한 값이 배포 기대값이고, 제외한 값은 "안 넘어졌을 때의 능력"이다. 둘은 다른 질문에
답하므로 같이 낸다. 제외한 값은 생존 편향이 있어 단독으로 쓰면 안 된다.
"""

from __future__ import annotations

import glob
import pathlib

import numpy as np

W = "/home/lgb/IsaacLab-6.0/_workspace/leg"
OUT = pathlib.Path(
    "/home/lgb/IsaacLab-6.0/reports/leg_imitation/_comparisons/velscale_deploy_and_symmetry/metrics"
)
ARMS = [
    ("baseline 0.5", "base_it49999_*", "vsest_base_*"),
    ("vel_err_scale 1.0", "vs10_it49999_*", "vsest_vs10_*"),
    ("vel_err_scale 1.5", "vs15_it49999_*", "vsest_vs15_*"),
    ("1.0 + mirror", "mir_it49999_*", "vsest_mir_*"),
]
PAIRS = [("FL_hip_joint", "FR_hip_joint"), ("HL_hip_joint", "HR_hip_joint")]
UP_05, STEADY = 1, [4, 5, 6, 7, 8]
FOLD_MIN = 0.25   # 붕괴 "시작 시점"을 찾을 때만 쓴다 (판정 자체는 변위로 한다)
ADV_MIN = 110.0   # [m] 사다리꼴 전체 이동거리. 관측 분포가 63~86 / 131~154 로 갈려 임계가 둔감하다


VMAX_TRAIN = 3.2  # 학습 명령 상한. 3.5 / 4.0 구간은 세 arm 모두 외삽이다


def per_rollout(path):
    d = np.load(path)
    prof, names = d["vx_profile"], [str(x) for x in d["joint_names"]]
    dt, hold, ramp = float(d["dt"]), float(d["hold_s"]), float(d["ramp_s"])
    span, rs, n = int(round((hold + ramp) / dt)), int(round(ramp / dt)), len(d["vx"])
    w = np.sqrt((d["jvel"] ** 2).mean(axis=1))
    fold_idx = [i for i, x in enumerate(names) if "thigh" in x or "calf" in x]

    # 붕괴 판정 = 사다리꼴 전체 이동거리. 실측이 완벽히 이봉성이다(완주 131~154 m vs 붕괴 63~86 m).
    # ★ 접힘 비율(folded)로 재면 arm 마다 무너지는 자세가 달라 놓친다 — mirror arm 은 덜 접힌
    #   채로 무너져 folded 0.12 인데 변위는 71 m 다. 변위는 4 arm 전부에서 일관된다.
    adv = float(np.hypot(d["px"][-1] - d["px"][0], d["py"][-1] - d["py"][0]))
    fallen, fall_cmd = adv < ADV_MIN, np.nan
    if fallen:
        # 붕괴 시작 = 끝까지 이어지는 접힘 구간의 시작. 정상 보행 중에도 calf 가 일시적으로
        # -0.6 rad 를 넘으므로 첫 교차로 재면 t=14~18 s 로 잘못 나온다.
        folded = (d["jpos"][:, fold_idx] < -0.6).mean(axis=1) >= FOLD_MIN
        if folded[-1]:
            k = len(folded) - 1
            while k > 0 and folded[k - 1]:
                k -= 1
            fall_cmd = float(d["vx_cmd"][k])
        else:  # 접힘 없이 주저앉은 경우 — 전진이 멈춘 시점으로 잡는다
            moving = np.abs(d["vx"]) > 0.3
            k = len(moving) - 1
            while k > 0 and not moving[k - 1]:
                k -= 1
            fall_cmd = float(d["vx_cmd"][k])

    def win(i):
        return min(i * span + rs, n), min((i + 1) * span, n)

    s, e = win(UP_05)
    start = float(np.median(d["vx"][s:e])) / 0.5
    duty = float((w[s:e] > 0.4).mean())
    ach, sk = [], []
    for i in range(1, (len(prof) + 1) // 2):
        lo, hi = win(i)
        if lo >= hi:
            continue
        ach.append((float(prof[i]), float(np.median(d["vx"][lo:hi])) / float(prof[i])))
        if float((w[lo:hi] > 0.4).mean()) >= 0.5:  # 걷는 칸만
            j = d["jpos"][lo:hi]
            sk.append(
                abs(j[:, names.index(PAIRS[0][0])].mean() + j[:, names.index(PAIRS[0][1])].mean())
                + abs(j[:, names.index(PAIRS[1][0])].mean() + j[:, names.index(PAIRS[1][1])].mean())
            )
    lv = np.array([a[0] for a in ach])
    av = np.array([a[1] for a in ach])
    steady = float(np.mean(av[(lv >= 2.0) & (lv <= 4.0)]))
    in_range = float(np.mean(av[(lv >= 2.0) & (lv <= VMAX_TRAIN)]))  # 학습 범위 안만
    return steady, in_range, start, duty, (float(np.mean(sk)) if sk else np.nan), adv, fallen, fall_cmd


def table(tag, key):
    lines = [
        f"## {tag}",
        "",
        f"{'arm':18s} {'n':>3s} {'붕괴<=3.2':>9s} {'붕괴>3.2':>8s} {'2.0~3.2':>8s} "
        f"{'2.0~4.0':>8s} {'start':>7s} {'duty':>6s} {'skew':>7s} {'변위':>8s}",
        "-" * 92,
    ]
    for lab, gt, est in ARMS:
        paths = sorted(glob.glob(f"{W}/{gt if key == 'gt' else est}/ramp_data.npz"))
        if not paths:
            lines.append(f"{lab:18s}  (데이터 없음)")
            continue
        r = [per_rollout(p) for p in paths]
        cond = {bool(np.load(p)["use_estimator"]) if "use_estimator" in np.load(p).files else False
                for p in paths}
        assert len(cond) == 1, f"조건이 섞였다: {lab}"
        f_in = sum(1 for x in r if x[6] and x[7] <= VMAX_TRAIN)
        f_out = sum(1 for x in r if x[6] and x[7] > VMAX_TRAIN)
        a = np.array([x[:6] for x in r], dtype=float)
        lines.append(
            f"{lab:18s} {len(r):3d} {f'{f_in}/{len(r)}':>9s} {f'{f_out}/{len(r)}':>8s} "
            f"{100 * np.median(a[:, 1]):7.1f}% {100 * np.median(a[:, 0]):7.1f}% "
            f"{100 * np.median(a[:, 2]):6.1f}% {np.median(a[:, 3]):6.2f} "
            f"{np.nanmedian(a[:, 4]):7.3f} {np.median(a[:, 5]):7.1f}m"
        )
    lines.append("")
    return lines


def main():
    out = [
        "# velscale sweep 50k — model_49999 매칭, 동일 프로토콜",
        "",
        "2.0~3.2 = 학습 명령 범위 안 달성률 · 2.0~4.0 = 외삽 포함 · start = cmd 0.5 상승 달성률",
        "· duty = cmd 0.5 hold 에서 `|jvel| RMS > 0.4` 스텝 비율 · skew = 롤아웃별 |hip L+R|(앞+뒤)",
        "절대값 먼저, 걷는 칸만 · 변위 = 사다리꼴 전체 이동거리. 값은 롤아웃 중앙값(붕괴 포함).",
        "",
        "★ 학습 상한은 lin_vel_x_max=3.2 다. 램프의 3.5 / 4.0 구간은 세 arm 모두 외삽이라 붕괴를",
        "  범위 안/밖으로 나눠 센다. 붕괴 시작은 '끝까지 이어지는 접힘 구간의 시작'으로 잡는다 —",
        "  정상 보행 중에도 calf 가 일시적으로 -0.6 rad 를 넘으므로 첫 교차로 재면 안 된다.",
        "★ skew 는 롤아웃별 절대값을 먼저 취하므로 부호를 살려 평균하는 정의보다 항상 크거나 같다.",
        "",
    ]
    out += table("GT 조건 (env 의 ground-truth base 속도 = 특권 정보)", "gt")
    out += table("estimator 조건 (실기 배포 — base 속도를 추정으로 대체)", "est")
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / "velscale_50k_gt_vs_estimator.md"
    p.write_text("\n".join(out) + "\n")
    print("\n".join(out))
    print(f"saved: {p}")


if __name__ == "__main__":
    main()

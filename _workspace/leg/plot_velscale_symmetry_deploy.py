# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""velscale sweep 50k — 학습측 대칭 지표와 램프 실측을 잇고, 배포(estimator) 조건을 붙인다.

다른 세션이 만든 `velscale_verdict_50k.png` 는 onset / 달성률 / 붕괴 3면을 덮는다. 여기서는
그 그림에 없는 두 축만 그린다.

  (a) 학습측 `Mean symmetry loss` 궤적 — s=1.0 이 30~35k 에서 기준을 추월한다.
  (b) 램프 실측 hip 좌우 합 (명령속도별) — (a) 의 학습 지표가 실제 자세로 나타나는지.
  (c) GT vs estimator 달성률 — 실기는 base 속도를 못 재므로 배포 조건이 진짜 값이다.

(a) 는 학습 로그, (b)(c) 는 램프 npz 에서 읽는다. 화면 문자열은 영어로 쓴다(폰트 제약).
"""

from __future__ import annotations

import glob
import pathlib
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = "/home/lgb/IsaacLab-6.0/_workspace/leg"
OUT = pathlib.Path("/home/lgb/IsaacLab-6.0/reports/leg_imitation/_comparisons/velscale_deploy_and_symmetry")

# (라벨, 학습로그, GT 램프 glob, EST 램프 glob, 색)
ARMS = [
    ("baseline 0.5", "ds14_wcmd", "base_it49999_*", "vsest_base_*", "#7f7f7f"),
    ("vel_err_scale 1.0", "velscale10", "vs10_it49999_*", "vsest_vs10_*", "#d62728"),
    ("vel_err_scale 1.5", "velscale15", "vs15_it49999_*", "vsest_vs15_*", "#2ca02c"),
    ("1.0 + mirror", "velscale10_cumirror", "mir_it49999_*", "vsest_mir_*", "#1f77b4"),
]
PAIRS = [("FL_hip_joint", "FR_hip_joint"), ("HL_hip_joint", "HR_hip_joint")]
CMDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]


def sym_trace(stem, bin_w=2500):
    """학습 로그에서 `Mean symmetry loss` 를 구간 평균으로 읽는다."""
    path = f"{W}/train_logs/{stem}.log"
    it, rows = None, []
    for line in open(path, errors="ignore"):
        m = re.search(r"Learning iteration (\d+)/", line)
        if m:
            it = int(m.group(1))
        elif "Mean symmetry loss:" in line and it is not None:
            rows.append((it, float(line.split(":")[-1])))
    if not rows:
        return np.array([]), np.array([])
    last = rows[-1][0]
    edges = np.arange(0, last + bin_w, bin_w)
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        v = [y for x, y in rows if a <= x < b]
        if v:
            xs.append((a + b) / 2)
            ys.append(np.mean(v))
    return np.array(xs), np.array(ys)


ADV_MIN = 110.0  # [m] 사다리꼴 전체 이동거리. 관측 분포가 63~86 / 131~154 로 갈려 임계가 둔감하다


def _fallen(d):
    """넘어진 롤아웃인지 — 사다리꼴 전체 이동거리로 판정한다.

    ★ 접힘 비율(종료 시 thigh/calf < -0.6 rad)로 재면 arm 마다 무너지는 자세가 달라 놓친다.
    mirror arm 은 덜 접힌 채로 주저앉아 folded 0.12 인데 변위는 71 m 다. 변위 기준은 4 arm
    전부에서 일관되며, base 0/8 · s=1.0 4/8 · s=1.5 1/8 도 그대로 재현한다.
    """
    return bool(np.hypot(d["px"][-1] - d["px"][0], d["py"][-1] - d["py"][0]) < ADV_MIN)


DUTY_MIN = 0.5  # 이 아래는 '걷지 않음'이라 자세 쏠림을 보행 비대칭으로 읽으면 안 된다


def ramp_stats(pattern, drop_fallen=True):
    """(달성률[cmd], 쏠림[cmd], 걷는 롤아웃 수[cmd], 넘어진 수, 전체 수).

    쏠림은 **롤아웃마다 절대값을 먼저 취한 뒤** 평균한다. 반대 방향으로 쏠린 롤아웃끼리 상쇄되지
    않으므로, 롤아웃을 먼저 평균하는 정의보다 항상 크거나 같다. 다른 세션의 `쏠림` 열은 부호를
    살려 평균하므로 두 값은 일치하지 않는다 — 같은 run 을 다르게 요약한 것이다.

    ★ duty(= `|jvel| RMS > 0.4` 인 스텝 비율) < DUTY_MIN 인 (롤아웃, 명령) 칸은 뺀다. 서 있는
    로봇도 hip 에 정적 오프셋이 있어 쏠림처럼 보이는데, 그건 보행 비대칭이 아니다.
    """
    paths = sorted(glob.glob(f"{W}/{pattern}/ramp_data.npz"))
    ach, skew, nfall = [], [], 0
    for p in paths:
        d = np.load(p)
        if _fallen(d):
            nfall += 1
            if drop_fallen:
                continue
        prof = d["vx_profile"]
        dt, hold, ramp = float(d["dt"]), float(d["hold_s"]), float(d["ramp_s"])
        span, rs, n = int(round((hold + ramp) / dt)), int(round(ramp / dt)), len(d["vx"])
        names = [str(x) for x in d["joint_names"]]
        w = np.sqrt((d["jvel"] ** 2).mean(axis=1))
        a, s = [], []
        for i in range(1, (len(prof) + 1) // 2):  # 상승 구간만
            lo, hi = min(i * span + rs, n), min((i + 1) * span, n)
            if lo >= hi:
                a.append(np.nan)
                s.append(np.nan)
                continue
            a.append(float(np.median(d["vx"][lo:hi])) / float(prof[i]))
            if float((w[lo:hi] > 0.4).mean()) < DUTY_MIN:
                s.append(np.nan)  # 걷지 않는 칸
                continue
            j = d["jpos"][lo:hi]
            s.append(
                abs(j[:, names.index(PAIRS[0][0])].mean() + j[:, names.index(PAIRS[0][1])].mean())
                + abs(j[:, names.index(PAIRS[1][0])].mean() + j[:, names.index(PAIRS[1][1])].mean())
            )
        ach.append(a)
        skew.append(s)
    if not ach:
        return None, None, None, nfall, len(paths)
    sk = np.array(skew, dtype=float)
    n_walk = (~np.isnan(sk)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        sk_mean = np.where(n_walk > 0, np.nanmean(np.where(np.isnan(sk), np.nan, sk), axis=0), np.nan)
    return np.nanmean(ach, axis=0), sk_mean, n_walk, nfall, len(paths)


def main():
    fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.0))

    # (a) 학습측 대칭 지표
    for lab, stem, _, _, c in ARMS:
        x, y = sym_trace(stem)
        if len(x):
            ax[0].plot(x / 1000, y, color=c, lw=2, ls="--" if "mirror" in lab else "-", label=lab)
    ax[0].set_xlabel("training iteration [k]")
    ax[0].set_ylabel("Mean symmetry loss")
    ax[0].set_title("(a) Training-side asymmetry\n"
                    "same scale 1.0, mirrored expert set: 0.63 -> 0.065")
    ax[0].legend(fontsize=9)
    ax[0].grid(alpha=0.3)

    # (b) 램프 실측 쏠림 — 걷는 칸만
    for lab, _, gt, _, c in ARMS:
        _, sk, nw, nf, nt = ramp_stats(gt)
        if sk is None:
            continue
        x = np.array(CMDS[: len(sk)])
        ax[1].plot(x, sk, "o-", color=c, lw=2, label=f"{lab}  ({nt - nf}/{nt} upright)")
        for xi, si, ni in zip(x, sk, nw):
            if not np.isnan(si) and ni < nt - nf:  # 일부 롤아웃이 그 명령에서 안 걸었다
                ax[1].annotate(f"{ni}", (xi, si), textcoords="offset points", xytext=(0, 7),
                               ha="center", fontsize=7, color=c)
        if nf:  # 붕괴한 롤아웃이 빠진 자리는 생존 편향이라 표시한다
            ax[1].annotate(f"{nf}/{nt} fell at 4.0", (x[-1], sk[-1]), textcoords="offset points",
                           xytext=(-6, -16), ha="right", fontsize=8, color=c, fontweight="bold")
    # 저속대는 '서기→보행' 전이 구간이라 쏠림이 보행 비대칭이 아닌 전이 자세를 반영할 수 있다.
    ax[1].axvspan(0.4, 1.75, color="#fff3cd", zorder=0)
    ax[1].text(1.05, ax[1].get_ylim()[1] * 0.03, "gait-onset band", ha="center", fontsize=8, color="#8a6d3b")
    ax[1].set_xlabel("commanded speed [m/s]")
    ax[1].set_ylabel("|hip L+R| front + |hip L+R| hind  [rad]")
    ax[1].set_title("(b) Posture skew while actually walking\n"
                    "per-rollout |sum| then averaged (opposite-sign rollouts do not cancel)")
    ax[1].legend(fontsize=9)
    ax[1].grid(alpha=0.3)

    # (c) GT vs estimator — 달성률은 붕괴 롤아웃도 포함한다. 붕괴는 cmd 3.5~4.0 에서만 일어나므로
    #     저속 구간 값은 멀쩡하고, 넘어진 회차를 빼면 고속 실패가 통째로 사라져 낙관 편향이 된다.
    any_est = False
    for lab, _, gt, est, c in ARMS:
        a_gt, _, _, nf_g, nt_g = ramp_stats(gt, drop_fallen=False)
        a_es, _, _, nf_e, nt_e = ramp_stats(est, drop_fallen=False)
        if a_gt is not None:
            ax[2].plot(CMDS[: len(a_gt)], 100 * a_gt, "o-", color=c, lw=2,
                       label=f"{lab} — GT ({nf_g}/{nt_g} fell)")
        if a_es is not None:
            any_est = True
            ax[2].plot(CMDS[: len(a_es)], 100 * a_es, "s--", color=c, lw=1.6, alpha=0.8,
                       label=f"{lab} — estimator ({nf_e}/{nt_e} fell)")
    ax[2].axvspan(3.2, 4.05, color="0.9", zorder=0)
    ax[2].text(3.62, 58, "beyond training 3.2\nall falls start here", ha="center", va="center",
               fontsize=7.5, color="0.4")
    ax[2].set_xlabel("commanded speed [m/s]")
    ax[2].set_ylabel("achievement [%]  (fallen rollouts included)")
    ax[2].set_title("(c) Deploy condition\nsolid = ground-truth base velocity, dashed = estimator")
    ax[2].legend(fontsize=7, loc="lower right")
    ax[2].grid(alpha=0.3)
    if not any_est:
        ax[2].text(0.5, 0.5, "estimator ramps not finished", transform=ax[2].transAxes,
                   ha="center", color="#d62728")

    fig.suptitle("leg_imitation  vel_err_scale sweep at 50k — symmetry and deploy condition", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    p = OUT / "figures" / "symmetry_and_deploy.png"
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"saved: {p}")


if __name__ == "__main__":
    sys.exit(main())

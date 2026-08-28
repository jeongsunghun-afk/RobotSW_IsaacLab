# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`vel_err_scale` 50k 판정 그림 — 개시 이봉 · 명령별 달성률 · 붕괴율.

세 패널을 한 장에 담는다. 왼쪽이 이 실험이 겨냥한 것이고, 나머지 둘은 "저속을 사느라
고속이나 안정성을 팔지 않았는가" 를 확인하는 게이트다.

  왼쪽 : cmd 0.5 상승 duty 를 **롤아웃 점으로** 찍는다. 평균 막대만 그리면 이봉이 안 보인다 —
         0.00 과 1.00 이 섞인 평균 0.5 는 "항상 애매" 가 아니라 "반은 서고 반은 걷는다" 다.
  가운데: 명령별 달성률 곡선. 저속↔고속 트레이드오프가 있으면 여기서 교차한다.
  오른쪽: 8회 중 넘어진 횟수. 넘어진 롤아웃은 왼쪽·가운데 통계에서 빠져 있으므로
         이 패널이 없으면 "붕괴를 감수하고 얻은 점수" 가 공짜처럼 보인다.

라벨은 영어로 둔다 — 렌더 환경에 CJK 폰트가 없어 한글이 □ 로 깨진다.

    ./isaaclab.sh -p _workspace/leg/plot_velscale_verdict.py
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = pathlib.Path("reports/leg_imitation/_comparisons/arch_scaling_study/figures/velscale_verdict_50k.png")

# 측정값 출처: metrics/velscale_ramp_50k.md (model_49999 매칭 · 8 repeat · 동일 플래그)
ARMS = [
    # (라벨, 색, cmd 0.5 상승 duty 롤아웃별(붕괴 제외), 명령별 달성률 %, 붕괴/8)
    ("baseline 0.5", "#888888", [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 1.00, 1.00],
     [2, 67, 69, 96, 91, 88, 83, 80], 0),
    ("vel_err_scale 1.0", "#d1495b", [0.08, 0.65, 0.95, 1.00],
     [25, 77, 93, 100, 96, 93, 93, 93], 4),
    ("vel_err_scale 1.5", "#2a9d8f", [0.92, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00],
     [70, 78, 95, 100, 99, 98, 95, 91], 1),
]
CMDS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VMAX_TRAIN = 3.2  # 학습 명령 상한 [m/s] — 이보다 위는 외삽


def main() -> int:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    rng = np.random.default_rng(0)  # 점 흩뿌리기용(재현 가능하게 고정)

    ax = axes[0]
    for i, (label, color, duties, _, _) in enumerate(ARMS):
        x = i + (rng.random(len(duties)) - 0.5) * 0.28
        ax.scatter(x, duties, s=70, color=color, alpha=0.85, edgecolor="white", linewidth=1.2, zorder=3)
        ax.hlines(np.mean(duties), i - 0.3, i + 0.3, color=color, linewidth=2.5, zorder=4)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[0].replace("vel_err_scale ", "scale ") for a in ARMS])
    ax.set_ylim(-0.08, 1.08)
    ax.set_ylabel("gait duty at cmd 0.5, ascent")
    ax.set_title("Onset: does it start walking?\n(one dot = one rollout; bar = mean)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.02, 0.5, "baseline is binary:\n5 stand, 3 walk", transform=ax.transAxes,
            fontsize=9, color="#555", va="center")

    ax = axes[1]
    for label, color, _, ach, _ in ARMS:
        ax.plot(CMDS, ach, "o-", color=color, label=label, linewidth=2, markersize=5)
    ax.axvspan(VMAX_TRAIN, CMDS[-1], color="#000000", alpha=0.06)
    # 범례가 오른쪽 아래에 있으므로 주석은 음영대 위쪽에 둔다(가려지면 못 읽는다).
    ax.text(3.6, 40, "extrapolated\n(> training 3.2)", fontsize=8, color="#666", ha="center")
    ax.set_xlabel("commanded speed [m/s]")
    ax.set_ylabel("achievement [%]")
    ax.set_title("No speed traded away\n(fallen rollouts excluded)", fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[2]
    falls = [a[4] for a in ARMS]
    ax.bar(range(len(ARMS)), falls, color=[a[1] for a in ARMS], width=0.55)
    for i, f in enumerate(falls):
        ax.text(i, f + 0.12, f"{f}/8", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([a[0].replace("vel_err_scale ", "scale ") for a in ARMS])
    ax.set_ylim(0, 5)
    ax.set_ylabel("rollouts fallen (of 8)")
    ax.set_title("Cost: collapse at the 4.0 peak\n(scale 1.0 is the worst, not the middle)", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "leg_imitation  vel_err_scale sweep - matched model_49999, 8 repeats, identical flags",
        fontsize=12, y=1.02,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"saved: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Comparison plots: GT-voxel teacher vs SL-Grid+crawl (both from-scratch RL 50k, seed 1).

Three figures:
  fig1  physical outcomes  — comparable across runs (identical definitions)
  fig2  terrain curriculum — 5 shared terrains comparable; crawl is NOT (different course)
  fig3  optimization       — mean_reward is WITHIN-RUN TREND ONLY (per-run AMP discriminator)

GT voxel = orange, SL-Grid+crawl = aqua. The aqua matches the SL sensor overlay in the
rendered videos; the GT overlay there is yellow (the voxel marker's own color), so the plot
uses orange for GT rather than an exact match.
Palette validated with the dataviz skill validator (adjacent CVD dE 9.2, normal 27.6).
"""

import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

W = "/home/lgb/IsaacLab-6.0/_workspace/gt_vs_slgrid_crawl"
OUT = "/home/lgb/IsaacLab-6.0/reports/_comparisons/gt_voxel_vs_slgrid_crawl/figures"

GT, SL = "gtvoxel", "slgrid_crawl"
COLOR = {GT: "#eb6834", SL: "#1baf7a"}
LABEL = {GT: "GT voxel (privileged)", SL: "SL-Grid + crawl (deployable)"}

INK = "#1a1a19"
INK_2 = "#5c5b55"
GRID = "#e6e5df"


def load(name):
    series = defaultdict(lambda: ([], []))
    with open(f"{W}/curves_{name}.csv") as fh:
        for row in csv.DictReader(fh):
            it, val = series[row["tag"]]
            it.append(int(row["iteration"]))
            val.append(float(row["value"]))
    return {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in series.items()}


def smooth(it, val, window=1000):
    """Bin-average into fixed iteration windows — robust to the sparse terrain tags."""
    if len(it) == 0:
        return np.array([]), np.array([])
    edges = np.arange(0, 50001, window)
    idx = np.digitize(it, edges) - 1
    xs, ys = [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if m.sum() >= 3:
            xs.append(edges[b] + window / 2)
            ys.append(val[m].mean())
    return np.asarray(xs), np.asarray(ys)


def style(ax, title, ylabel):
    ax.set_title(title, fontsize=10.5, color=INK, pad=7, loc="left")
    ax.set_ylabel(ylabel, fontsize=8.5, color=INK_2)
    ax.set_xlabel("iteration", fontsize=8.5, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=3)
    ax.set_xlim(0, 50000)
    ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
    ax.set_xticklabels(["0", "10k", "20k", "30k", "40k", "50k"])


def draw(ax, data, tag, runs=(GT, SL), window=1000):
    """Plot one tag for the given runs; direct-label each line at its right end."""
    for r in runs:
        if tag not in data[r]:
            continue
        x, y = smooth(*data[r][tag], window=window)
        if len(x) == 0:
            continue
        ax.plot(x, y, color=COLOR[r], linewidth=2.0, zorder=3, solid_capstyle="round")
        ax.annotate(
            "GT" if r == GT else "SL",
            xy=(x[-1], y[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            color=COLOR[r],
            fontsize=8.5,
            fontweight="bold",
            va="center",
            zorder=4,
        )


def legend(fig, note=None, legend_y=0.036, note_y=0.010):
    """Legend above, italic caveat note below — keep the two y-anchors from colliding."""
    handles = [
        plt.Line2D([], [], color=COLOR[r], linewidth=2.4, label=LABEL[r]) for r in (GT, SL)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, legend_y),
        labelcolor=INK,
    )
    if note:
        fig.text(0.5, note_y, note, ha="center", fontsize=8.2, color=INK_2, style="italic")


def main():
    import os

    os.makedirs(OUT, exist_ok=True)
    data = {GT: load(GT), SL: load(SL)}

    # ---- fig1: physical outcomes (comparable) --------------------------------
    panels = [
        ("Episode_Termination/cause_goal_reached", "terminated: goal reached", "per episode"),
        ("Episode_Termination/cause_base_contact", "terminated: base contact (fall)", "per episode"),
        ("Episode_Termination/cause_tilt", "terminated: tilt", "per episode"),
        ("Train/mean_episode_length", "episode length", "steps"),
        ("Episode_Reward/collision", "collision penalty", "reward"),
        ("Episode_Reward/feet_stumble", "feet_stumble penalty", "reward"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor="#fcfcfb")
    for ax, (tag, title, ylab) in zip(axes.ravel(), panels):
        ax.set_facecolor("#fcfcfb")
        style(ax, title, ylab)
        draw(ax, data, tag)
    fig.suptitle(
        "Physical outcomes - identical definitions, directly comparable across runs",
        fontsize=13,
        color=INK,
        y=0.985,
    )
    legend(
        fig,
        "Terrain proportions differ:  GT = 5 terrains x 0.15 + crawl 0.25   |   SL-Grid = flat 0.20 + 5 terrains x 0.16",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    fig.savefig(f"{OUT}/fig1_physical_outcomes.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ---- fig2: terrain curriculum -------------------------------------------
    terr = [
        ("stair", "stair"),
        ("gap", "gap"),
        ("hurdle", "hurdle"),
        ("step", "step"),
        ("flat", "flat"),
        ("crawl", "crawl   [!] DIFFERENT COURSE - not comparable"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), facecolor="#fcfcfb")
    for ax, (t, title) in zip(axes.ravel(), terr):
        ax.set_facecolor("#fcfcfb")
        style(ax, title, "curriculum level")
        draw(ax, data, f"curriculum/mean_terrain_level_{t}")
        ax.set_ylim(0, 8.5)
        if t == "crawl":
            ax.set_title(title, fontsize=10.5, color="#b3421a", pad=7, loc="left")
    fig.suptitle(
        "Terrain curriculum level - a promotion-threshold result, NOT a completion rate",
        fontsize=13,
        color=INK,
        y=0.985,
    )
    legend(
        fig,
        "crawl alone is not comparable: GT trained on crawl-3 (course 10.95 m), SL-Grid on crawl-8 (17.79 m)",
    )
    fig.tight_layout(rect=(0, 0.065, 1, 0.96))
    fig.savefig(f"{OUT}/fig2_terrain_curriculum.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    # ---- fig3: optimization --------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), facecolor="#fcfcfb")
    for ax, (tag, title, ylab) in zip(
        axes.ravel(),
        [
            ("Train/mean_reward", "mean reward   [!] NOT comparable across runs", "reward"),
            ("Policy/mean_noise_std", "action noise std", "std"),
            ("Loss/value", "value loss", "loss"),
        ],
    ):
        ax.set_facecolor("#fcfcfb")
        style(ax, title, ylab)
        draw(ax, data, tag)
    axes[0].set_title(
        "mean reward   [!] NOT comparable across runs", fontsize=10.5, color="#b3421a", pad=7, loc="left"
    )
    axes[2].set_yscale("log")
    fig.suptitle("Optimization metrics", fontsize=13, color=INK, y=0.97)
    legend(
        fig,
        "mean reward embeds a per-run AMP discriminator - absolute values are NOT comparable; read within-run trend only",
        legend_y=0.075,
        note_y=0.022,
    )
    fig.tight_layout(rect=(0, 0.155, 1, 0.94))
    fig.savefig(f"{OUT}/fig3_optimization.png", dpi=150, facecolor="#fcfcfb")
    plt.close(fig)

    print("wrote 3 figures to", OUT)


if __name__ == "__main__":
    main()

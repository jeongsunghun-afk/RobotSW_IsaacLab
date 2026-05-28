# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Offline reward attribution plot helper for Go2 Parkour.

Reads a per-terrain ``.npz`` buffer produced by ``play_reward_attribution.py``
and renders a figure with:

    Top section:  4×4 grid of 16 individual reward-term subplots (one per term).
    Middle:       Per-foot contact mask (4 binary lanes) + hind-foot dragging indicator.
    Bottom:       Active-foot count (0-4) with 3-leg / 2-leg phase shading.

All 18 axes share the same step-index x-axis.

**No Isaac Sim / CUDA dependency** — importable from any Python env that has
numpy + matplotlib.

Usage (standalone):
    python reward_attribution_plot.py <npz_path> <out_png_path>

Usage (as module):
    from reward_attribution_plot import plot_terrain_attribution
    plot_terrain_attribution("flat.npz", "flat.png")

Design reference: _workspace/parkour_reward_attribution/DESIGN.md §8
                  _workspace/parkour_reward_attribution/SUBTASKS.md §S3
"""

from __future__ import annotations

import os
import sys

import numpy as np


# ── Tab20 color palette (matches viewer _TERM_COLORS for visual continuity) ────
# 16 entries, one per reward term index.  Same RGB values as viewer's _TERM_COLORS,
# converted to normalized float tuples for matplotlib.
_TAB20_COLORS: list[tuple[float, float, float]] = [
    (31 / 255,  119 / 255, 180 / 255),   #  0 tracking_goal_vel
    (174 / 255, 199 / 255, 232 / 255),   #  1 tracking_yaw
    (255 / 255, 127 / 255,  14 / 255),   #  2 lin_vel_z_l2
    (255 / 255, 187 / 255, 120 / 255),   #  3 ang_vel_xy_l2
    ( 44 / 255, 160 / 255,  44 / 255),   #  4 orientation_l2
    (152 / 255, 223 / 255, 138 / 255),   #  5 dof_acc_l2
    (214 / 255,  39 / 255,  40 / 255),   #  6 collision
    (255 / 255, 152 / 255, 150 / 255),   #  7 action_rate_l2
    (148 / 255, 103 / 255, 189 / 255),   #  8 delta_torques
    (197 / 255, 176 / 255, 213 / 255),   #  9 torques_l2
    (140 / 255,  86 / 255,  75 / 255),   # 10 hip_pos
    (196 / 255, 156 / 255, 148 / 255),   # 11 dof_error_l2
    (227 / 255, 119 / 255, 194 / 255),   # 12 feet_stumble
    (247 / 255, 182 / 255, 210 / 255),   # 13 feet_edge
    (127 / 255, 127 / 255, 127 / 255),   # 14 feet_dragging
    (188 / 255, 189 / 255, 220 / 255),   # 15 feet_gait_pairing
]


def _term_color(idx: int) -> tuple[float, float, float]:
    """Return the tab20 color for reward term at *idx* (wraps if idx >= 16)."""
    return _TAB20_COLORS[idx % len(_TAB20_COLORS)]


def plot_terrain_attribution(
    npz_path: str,
    out_png_path: str,
    figsize: tuple[float, float] = (18, 14),
    dpi: int = 120,
) -> None:
    """Render 4×4 reward grid + contact + active-feet figure from a per-terrain npz buffer.

    Layout (all axes share the step-index x-axis):
        Top (4×4 grid):  16 individual reward-term subplots, one per term.
                         Each has its own y-axis (auto-scaled), title = term name,
                         and a small legend annotation showing the term name.
        Middle:          Per-foot contact lanes (black=contact) + hind-foot
                         dragging indicator (red overlay on RL/RR lanes).
        Bottom:          Active-foot count line + 3-leg/2-leg phase shading.

    Args:
        npz_path:     Path to a per-terrain ``.npz`` file from
                      ``play_reward_attribution.py``.
        out_png_path: Output PNG path (parent dir is created if missing).
        figsize:      Figure size in inches (width, height). Default enlarged for 4×4 grid.
        dpi:          Output resolution (dots per inch).
    """
    # Deferred import so the module can be imported without a display server.
    import matplotlib
    # Only set backend if not already set; avoids conflict when called from sim context.
    try:
        matplotlib.use("Agg")
    except Exception:
        pass
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.patches as mpatches

    # ── Load data ──────────────────────────────────────────────────────────────
    data = np.load(npz_path, allow_pickle=True)

    rewards: np.ndarray = data["rewards"].astype(np.float32)         # (T, K)
    contacts: np.ndarray = data["foot_contact"].astype(bool)          # (T, 4)
    step_dt: float = float(data["step_dt"])
    T: int = int(data["episode_len"])
    terrain_name: str = str(data["terrain_name"])
    difficulty: int = int(data["difficulty"])
    term_names: list[str] = list(data["term_names"].tolist())
    foot_names: list[str] = list(data["foot_names"].tolist())
    dragging_threshold: float = float(data.get("dragging_velocity_threshold", 0.05))

    # Foot XY speed — may be absent in buffers from older script versions.
    if "foot_xy_speed" in data:
        foot_xy_speed: np.ndarray = data["foot_xy_speed"].astype(np.float32)  # (T, 4)
    else:
        foot_xy_speed = np.zeros((T, 4), dtype=np.float32)

    # Trim to actual episode length (buffer may be pre-allocated longer).
    rewards = rewards[:T]
    contacts = contacts[:T]
    foot_xy_speed = foot_xy_speed[:T]

    t_steps = np.arange(T, dtype=np.float32)
    t_secs = t_steps * step_dt

    # ── Derived signals ────────────────────────────────────────────────────────
    n_contact = contacts.sum(axis=-1).astype(np.int32)        # (T,) 0..4
    n_airborne = 4 - n_contact                                # (T,) 0..4
    is_3leg = n_airborne == 1                                  # exactly 1 foot airborne
    is_2leg = n_airborne == 2                                  # exactly 2 feet airborne

    # Hind-foot dragging (RL=idx 2, RR=idx 3): in contact AND XY speed > threshold.
    hind_contact = contacts[:, 2:4]                                         # (T, 2) bool
    hind_speed = foot_xy_speed[:, 2:4]                                      # (T, 2) float
    is_dragging_rl = hind_contact[:, 0] & (hind_speed[:, 0] > dragging_threshold)
    is_dragging_rr = hind_contact[:, 1] & (hind_speed[:, 1] > dragging_threshold)

    K = rewards.shape[-1]  # normally 16

    # ── Figure layout ──────────────────────────────────────────────────────────
    # Outer GridSpec: row 0 = 4×4 reward block, row 1 = contact, row 2 = active feet.
    # We use a nested GridSpec for the 4×4 block.
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.suptitle(
        f"Terrain: parkour_{terrain_name}  "
        f"(level={difficulty},  ep_len={T} steps / {T * step_dt:.1f} s)",
        fontsize=13,
        fontweight="bold",
        y=0.998,
    )

    outer_gs = gridspec.GridSpec(
        3, 1,
        height_ratios=[8, 2, 1.5],
        hspace=0.35,
        figure=fig,
    )

    # ── TOP: 4×4 reward grid ───────────────────────────────────────────────────
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        4, 4,
        subplot_spec=outer_gs[0],
        hspace=0.55,
        wspace=0.30,
    )

    ax_reward_grid: list[plt.Axes] = []
    ax_anchor: plt.Axes | None = None

    for idx in range(K):
        row = idx // 4
        col = idx % 4
        name = term_names[idx] if idx < len(term_names) else f"term_{idx}"
        color = _term_color(idx)

        if ax_anchor is None:
            ax = fig.add_subplot(inner_gs[row, col])
            ax_anchor = ax
        else:
            ax = fig.add_subplot(inner_gs[row, col], sharex=ax_anchor)

        ax.plot(t_steps, rewards[:, idx], color=color, linewidth=0.9, alpha=0.9)
        ax.axhline(0.0, color="gray", linewidth=0.4, linestyle=":")
        ax.set_title(name, fontsize=7, pad=2)

        # Small legend-style annotation in upper-left corner.
        ax.annotate(
            name,
            xy=(0.03, 0.93),
            xycoords="axes fraction",
            fontsize=5.5,
            color=color,
            va="top",
        )
        # Minimal legend (single entry) for visual consistency with live viewer.
        ax.legend(
            handles=[plt.Line2D([0], [0], color=color, linewidth=1.2, label=name)],
            loc="upper right",
            fontsize=5,
            framealpha=0.6,
            handlelength=1.0,
        )

        # Y auto-scale (per-subplot — different magnitudes per term).
        ax.autoscale(axis="y")

        # Hide x-tick labels on all but bottom row.
        if row < 3:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel("step", fontsize=6)

        ax.tick_params(axis="both", labelsize=5.5)
        ax_reward_grid.append(ax)

    # ── MIDDLE: Foot contact lanes + dragging overlay ──────────────────────────
    ax_contact = fig.add_subplot(outer_gs[1], sharex=ax_anchor)

    lane_h = 0.85        # filled band height per foot
    lane_gap = 1.2       # vertical spacing between lanes
    contact_color = "black"
    air_color = "#e8e8e8"
    drag_color = "#d62728"   # red — dragging on hind feet

    for fi, fname in enumerate(foot_names):
        ybase = fi * lane_gap

        # Air phase: faint grey background.
        ax_contact.fill_between(
            t_steps, ybase, ybase + lane_h,
            where=~contacts[:, fi],
            step="post", color=air_color, alpha=0.5, linewidth=0,
        )
        # Contact phase: solid dark fill.
        ax_contact.fill_between(
            t_steps, ybase, ybase + lane_h,
            where=contacts[:, fi],
            step="post", color=contact_color, alpha=0.78, linewidth=0,
        )
        # Foot label.
        ax_contact.text(
            -T * 0.012, ybase + lane_h * 0.5, fname,
            ha="right", va="center", fontsize=8, fontweight="bold",
        )

    # Dragging overlay — RL lane (fi=2) and RR lane (fi=3).
    for fi, is_drag in [(2, is_dragging_rl), (3, is_dragging_rr)]:
        ybase = fi * lane_gap
        ax_contact.fill_between(
            t_steps, ybase, ybase + lane_h,
            where=is_drag,
            step="post", color=drag_color, alpha=0.55, linewidth=0,
        )

    # Y-axis: hide ticks (visual lane layout is self-explanatory).
    total_h = 4 * lane_gap
    ax_contact.set_ylim(-lane_gap * 0.4, total_h)
    ax_contact.set_yticks([])
    ax_contact.set_ylabel("Foot contact", fontsize=9)
    ax_contact.tick_params(labelbottom=False)

    # Small legend for dragging.
    drag_patch = mpatches.Patch(color=drag_color, alpha=0.55, label=f"hind dragging (>{dragging_threshold} m/s)")
    contact_patch = mpatches.Patch(color=contact_color, alpha=0.78, label="in contact")
    ax_contact.legend(
        handles=[contact_patch, drag_patch],
        loc="upper right",
        fontsize=7,
        framealpha=0.85,
    )

    # ── BOTTOM: Active-foot count + phase shading ──────────────────────────────
    ax_feet = fig.add_subplot(outer_gs[2], sharex=ax_anchor)

    ax_feet.step(
        t_steps, n_contact,
        where="post", color="#1f77b4", linewidth=1.3, label="feet in contact",
    )

    # 3-leg phase: exactly 1 foot airborne — magenta @ 28% alpha.
    ax_feet.fill_between(
        t_steps, 0, 4,
        where=is_3leg,
        step="post", color="magenta", alpha=0.28,
        label="3-leg phase (1 airborne)",
    )
    # 2-leg phase: exactly 2 feet airborne — cyan @ 20% alpha.
    ax_feet.fill_between(
        t_steps, 0, 4,
        where=is_2leg,
        step="post", color="cyan", alpha=0.20,
        label="2-leg phase (2 airborne)",
    )

    ax_feet.set_ylim(-0.2, 4.5)
    ax_feet.set_yticks([0, 1, 2, 3, 4])
    ax_feet.set_ylabel("# feet\nin contact", fontsize=9)
    ax_feet.set_xlabel("Step", fontsize=9)
    ax_feet.legend(loc="upper right", fontsize=7, framealpha=0.85)

    # Secondary x-axis in seconds at the top of the bottom subplot.
    ax_sec = ax_feet.twiny()
    ax_sec.set_xlim(t_secs[0], t_secs[-1])
    ax_sec.set_xlabel("Time (s)", fontsize=8, labelpad=4)

    # Sync shared x-axis limits.
    if ax_anchor is not None:
        ax_anchor.set_xlim(0, T - 1)

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_png_path)), exist_ok=True)
    fig.savefig(out_png_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
    print(f"[plot] Saved → {out_png_path}")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: python {sys.argv[0]} <npz_path> <out_png_path>")
        sys.exit(1)
    plot_terrain_attribution(sys.argv[1], sys.argv[2])

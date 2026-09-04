# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""긴 속도 램프(0->3.5->0)로 두 정책(cond-mlp A' vs cond+DRAIL B')을 시드 3개씩 겹쳐 비교한다.

speed_ramp_record_rma.py 가 저장한 ramp_data.npz 여러 개(정책 x 시드)를 읽어
- 속도 추종 + 레벨별 달성률/오차(상승 vs 하강, 히스테리시스)
- 관절 그룹별 토크 RMS/peak + effort_limit 대비 포화율
- 관절 그룹별 각속도 RMS/peak
- 레벨별 보행 분류(gait_classify 재사용)
를 겹쳐 그린 4장의 png 로 낸다.

레벨/방향(상승·하강)은 cmd 값으로 되짚지 않고 **정확한 stage 인덱스**(vx_profile 순서 + ramp_s/hold_s
로 계산한 시간 구간)로 나눈다 — cmd 값이 상승/하강에서 중복되므로(예: 2.0 이 두 번 나옴) 값 매칭만으로는
방향을 구분할 수 없다. hold 구간의 후반 4 s 만 평균한다(팀 지시).

각 stage 는 thigh/calf 가 FOLDED_RAD(-0.6 rad) 아래로 접힌 비율이 tail 창에서 0.5 를 넘으면
"낙상 후 누움" 으로 보고 그 stage 는 집계에서 제외한다(early_termination=False 라 넘어져도 npz 길이가
줄지 않고 계속 기록되므로, 별도로 걸러야 한다).

Run:
    ./isaaclab.sh -p _workspace/leg/plot_long_ramp_compare.py \
        --glob "condmlp50k=reports/leg_imitation/_comparisons/conditional_discriminator/metrics/long_ramp/condmlp50k_s*/ramp_data.npz" \
        --glob "conddrail45k=reports/leg_imitation/_comparisons/conditional_discriminator/metrics/long_ramp/conddrail45k_s*/ramp_data.npz" \
        --out_dir reports/leg_imitation/_comparisons/conditional_discriminator/figures
"""

import argparse
import glob
import os
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gait_classify import LEGS, MIN_AMP_RAD, MIN_CYCLES, _circ_mean_frac, _classify, _inst_phase  # noqa: E402
from ramp_fall_probe import FOLDED_RAD  # noqa: E402
from ramp_fall_probe import _probe as fall_probe  # noqa: E402
from ramp_fall_probe import _verdict as fall_verdict  # noqa: E402

TRAIN_CAP = 3.2  # 학습 명령 상한 [m/s] (vmax32)
SAT_FRAC = 0.95  # |tau| > SAT_FRAC * effort_limit 를 포화로 본다
TAIL_S = 4.0  # 레벨별 달성률/오차/토크/관절속도 평균에 쓸 hold 구간 후반 길이 [s]

GROUPS = {
    "FL": ["FL_hip_joint", "FL_thigh_joint", "FL_calf_joint", "FL_foot_joint"],
    "FR": ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FR_foot_joint"],
    "HL": ["HL_hip_joint", "HL_thigh_joint", "HL_calf_joint", "HL_foot_joint"],
    "HR": ["HR_hip_joint", "HR_thigh_joint", "HR_calf_joint", "HR_foot_joint"],
    "waist": ["FB_waist_joint"],
}


def parse_specs(specs):
    """['label=glob', ...] -> [(label, [npz path, ...]), ...]"""
    out = []
    for spec in specs:
        label, pattern = spec.split("=", 1)
        paths = sorted(glob.glob(pattern))
        out.append((label, paths))
    return out


class SeedRun:
    """npz 하나(정책 x 시드) + stage 분해 결과."""

    def __init__(self, path):
        self.path = path
        self.seed_label = pathlib.Path(path).parent.name
        d = np.load(path, allow_pickle=True)
        self.t = np.asarray(d["t"], dtype=float)
        self.vx_cmd = np.asarray(d["vx_cmd"], dtype=float)
        self.vx = np.asarray(d["vx"], dtype=float)
        self.vy = np.asarray(d["vy"], dtype=float)
        self.jpos = np.asarray(d["jpos"], dtype=float)
        self.jtau = np.asarray(d["jtau"], dtype=float)
        self.jvel = np.asarray(d["jvel"], dtype=float)
        self.names = [str(n) for n in d["joint_names"]]
        self.limit = np.asarray(d["effort_limit"], dtype=float)
        self.hold_s = float(d["hold_s"])
        self.ramp_s = float(d["ramp_s"]) if "ramp_s" in d.files else 0.0
        self.profile = np.asarray(d["vx_profile"], dtype=float)
        self.dt = float(d["dt"])
        self.seed = int(d["seed"]) if "seed" in d.files else -1
        self.fall = fall_probe(path, 0.3, 300)
        self.fall_verdict = fall_verdict(self.fall)

        self.stage_s = self.ramp_s + self.hold_s
        self.peak_idx = int(np.argmax(self.profile))  # 상승의 마지막 stage(정점)
        self.n_stages = len(self.profile)
        self.stages = self._build_stages()

    def _idx(self, t_val):
        return int(np.clip(round(t_val / self.dt), 0, len(self.t)))

    def _build_stages(self):
        legs_idx = [i for i, n in enumerate(self.names) if n.endswith(("_thigh_joint", "_calf_joint"))]
        rows = []
        for i, level in enumerate(self.profile):
            t0 = i * self.stage_s
            t_hold0 = t0 + self.ramp_s
            t1 = (i + 1) * self.stage_s
            tail0 = max(t_hold0, t1 - TAIL_S)
            lo, hi = self._idx(tail0), self._idx(t1)
            hi = min(hi, len(self.t))
            if hi <= lo:
                continue
            direction = "up" if i <= self.peak_idx else "down"
            folded_frac = float(np.mean(self.jpos[lo:hi][:, legs_idx].mean(axis=1) < FOLDED_RAD)) if legs_idx else 0.0
            rows.append(
                dict(
                    stage=i,
                    level=float(level),
                    direction=direction,
                    lo=lo,
                    hi=hi,
                    hold_lo=self._idx(t_hold0),
                    fallen=folded_frac > 0.5,
                    folded_frac=folded_frac,
                )
            )
        return rows

    def level_metric(self, stage, key):
        lo, hi = stage["lo"], stage["hi"]
        if key == "vx":
            return self.vx[lo:hi]
        if key == "vx_cmd":
            return self.vx_cmd[lo:hi]
        raise KeyError(key)

    def joint_series(self, stage, arr, joint):
        lo, hi = stage["lo"], stage["hi"]
        j = self.names.index(joint)
        return arr[lo:hi, j]

    def gait_at(self, stage):
        """stage 의 hold 구간 전체(ramp 이후)로 보행 분류. 낙상 stage 는 호출 전에 걸러야 한다."""
        lo, hi = stage["hold_lo"], stage["hi"]
        idx = {lg: self.names.index(f"{lg}_thigh_joint") for lg in LEGS}
        sig = {lg: self.jpos[lo:hi, idx[lg]] for lg in LEGS}
        n = hi - lo
        if n < 8:
            return "-"
        if max(float(np.ptp(s)) for s in sig.values()) < MIN_AMP_RAD:
            return "stand"
        fs = 1.0 / self.dt
        try:
            ph = {lg: _inst_phase(sig[lg], fs) for lg in LEGS}
        except ValueError:
            return "-"
        freq = float(np.polyfit(np.arange(n) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
        cycles = abs(freq) * n / fs
        if cycles < MIN_CYCLES:
            return "-"
        phi = {lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS}
        return _classify(phi)


def load_policies(specs):
    policies = []
    for label, paths in specs:
        runs = [SeedRun(p) for p in paths]
        policies.append((label, runs))
    return policies


def _collect_rate_and_err(runs, levels):
    """레벨/방향별 달성률(vx/vx_cmd) 과 |오차|(|vx-vx_cmd|) 딕셔너리 두 개."""
    rate, err = {}, {}
    for r in runs:
        for st in r.stages:
            if st["fallen"]:
                continue
            vx_seg = r.level_metric(st, "vx")
            cmd_seg = r.level_metric(st, "vx_cmd")
            cmd_v = float(np.mean(cmd_seg))
            key = round(st["level"], 1)
            abs_err = float(np.mean(np.abs(vx_seg - cmd_seg)))
            err.setdefault(key, {"up": [], "down": []})[st["direction"]].append(abs_err)
            if abs(cmd_v) < 1e-6:
                continue
            rate.setdefault(key, {"up": [], "down": []})[st["direction"]].append(float(np.mean(vx_seg)) / cmd_v)
    return rate, err


# ── 1. 속도 추종 + 달성률/오차(상승·하강) ────────────────────────────
def plot_tracking(policies, colors, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 13), height_ratios=[1.1, 0.9, 0.9])
    ax0 = axes[0]

    ref = policies[0][1][0]
    dt = ref.dt
    t_full = np.arange(len(ref.vx_cmd)) * dt
    ax0.plot(t_full, ref.vx_cmd, "k--", lw=1.5, label="vx command", zorder=5)
    ax0.axhline(TRAIN_CAP, color="0.3", ls=":", lw=1.2, label=f"training cap {TRAIN_CAP} m/s")
    # 3.5 (OOD) 구간 음영: profile 에서 3.5 인 stage 구간
    stage_s = ref.stage_s
    for i, lvl in enumerate(ref.profile):
        if abs(lvl - 3.5) < 1e-6:
            ax0.axvspan(i * stage_s, (i + 1) * stage_s, color="tab:red", alpha=0.08, zorder=0)

    for (label, runs), color in zip(policies, colors):
        seed_vx = []
        for r in runs:
            t = np.arange(len(r.vx)) * r.dt
            ax0.plot(t, r.vx, color=color, lw=0.6, alpha=0.35)
            seed_vx.append(r.vx)
        if seed_vx:
            n = min(len(v) for v in seed_vx)
            mean_vx = np.mean(np.stack([v[:n] for v in seed_vx]), axis=0)
            ax0.plot(np.arange(n) * ref.dt, mean_vx, color=color, lw=2.2, label=f"{label} (seed mean)")
    ax0.set_ylabel("x velocity [m/s]")
    ax0.set_title("Long speed ramp (0->3.5->0 m/s, hold 6s / ramp 2s): tracking, seed 0/1/2 overlay")
    ax0.legend(loc="upper right", fontsize=8, ncol=2)
    ax0.set_xlim(0, t_full[-1])

    # ── 레벨별 달성률/오차 막대(상승 vs 하강) ──
    levels = sorted(set(round(v, 1) for v in ref.profile))
    rate_fn = lambda runs: _collect_rate_and_err(runs, levels)[0]  # noqa: E731
    err_fn = lambda runs: _collect_rate_and_err(runs, levels)[1]  # noqa: E731
    _bar_grouped(
        axes[1], levels, policies, colors, rate_fn, "achievement rate  vx/vx_cmd",
        "Per-level achievement rate, up (solid) vs down (hatched) — mean +/- std over seeds",
    )
    axes[1].axhline(1.0, color="k", ls="--", lw=1.0)
    axes[1].legend(loc="upper left", fontsize=7, ncol=2)
    _bar_grouped(
        axes[2], levels, policies, colors, err_fn, "|vx - vx_cmd| [m/s]",
        "Per-level absolute tracking error, up (solid) vs down (hatched) — mean +/- std over seeds",
    )
    axes[2].set_xlabel("commanded vx [m/s] (level)")

    fig.tight_layout()
    p = os.path.join(out_dir, "long_ramp_tracking.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── 공통: 레벨별 스칼라 집계(그룹별) ────────────────────────────────
def _collect_group_stat(runs, arr_attr, stat, levels, group_joints, exclude_fallen=True):
    """stat in {'rms','peak'} over |value|, per level (양방향 합쳐 순서는 profile 순서 유지)."""
    out = {}  # level(round) -> {"up": [...], "down": [...]}
    for r in runs:
        arr = getattr(r, arr_attr)
        idxs = [r.names.index(j) for j in group_joints if j in r.names]
        if not idxs:
            continue
        for st in r.stages:
            if exclude_fallen and st["fallen"]:
                continue
            lo, hi = st["lo"], st["hi"]
            seg = np.abs(arr[lo:hi][:, idxs])
            val = float(seg.max()) if stat == "peak" else float(np.sqrt(np.mean(seg**2)))
            key = round(st["level"], 1)
            out.setdefault(key, {"up": [], "down": []})[st["direction"]].append(val)
    return out


def _collect_sat_frac(runs, levels, group_joints):
    out = {}
    for r in runs:
        idxs = [r.names.index(j) for j in group_joints if j in r.names]
        if not idxs:
            continue
        lim = r.limit[idxs]
        for st in r.stages:
            if st["fallen"]:
                continue
            lo, hi = st["lo"], st["hi"]
            seg = np.abs(r.jtau[lo:hi][:, idxs])
            sat = seg > (SAT_FRAC * lim[None, :])
            val = float(sat.mean())
            key = round(st["level"], 1)
            out.setdefault(key, {"up": [], "down": []})[st["direction"]].append(val)
    return out


def _bar_grouped(ax, levels, policies, colors, get_data_fn, ylabel, title, extra_line=None):
    n_pol = len(policies)
    width = 0.8 / (n_pol * 2)
    x = np.arange(len(levels))
    for pi, ((label, runs), color) in enumerate(zip(policies, colors)):
        data = get_data_fn(runs)
        for di, direction in enumerate(("up", "down")):
            means, stds, xs = [], [], []
            for li, lvl in enumerate(levels):
                vals = data.get(lvl, {}).get(direction, [])
                if not vals:
                    continue
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(li)
            if not xs:
                continue
            xs = np.array(xs, dtype=float)
            offset = (pi * 2 + di - (n_pol * 2 - 1) / 2.0) * width
            hatch = None if direction == "up" else "//"
            ax.bar(
                xs + offset, means, width=width * 0.9, yerr=stds, color=color,
                alpha=0.85 if direction == "up" else 0.5, hatch=hatch, capsize=2,
                label=f"{label} {direction}",
            )
    if extra_line is not None:
        ax.axhline(extra_line, color="0.3", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lv:.1f}" for lv in levels])
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9)


# ── 2. 토크 RMS/peak + 포화율 ────────────────────────────────────
def plot_torque(policies, colors, out_dir):
    ref = policies[0][1][0]
    levels = sorted(set(round(v, 1) for v in ref.profile))
    groups = list(GROUPS.items())
    fig, axes = plt.subplots(len(groups) + 1, 2, figsize=(16, 4 * (len(groups) + 1)))
    for row, (grp, joints) in enumerate(groups):
        rms_fn = lambda runs, joints=joints: _collect_group_stat(runs, "jtau", "rms", levels, joints)
        peak_fn = lambda runs, joints=joints: _collect_group_stat(runs, "jtau", "peak", levels, joints)
        _bar_grouped(axes[row, 0], levels, policies, colors, rms_fn, "|tau| RMS [N.m]", f"{grp}: torque RMS")
        _bar_grouped(axes[row, 1], levels, policies, colors, peak_fn, "|tau| peak [N.m]", f"{grp}: torque peak")
    # 마지막 행: 포화율(그룹별로 겹쳐서 하나의 패널 x2 대신, 그룹 평균 하나로 요약)
    sat_ax_rms, sat_ax = axes[-1, 0], axes[-1, 1]
    all_joints = [j for js in GROUPS.values() for j in js]
    sat_fn = lambda runs: _collect_sat_frac(runs, levels, all_joints)
    _bar_grouped(sat_ax, levels, policies, colors, sat_fn, f"frac |tau|>{SAT_FRAC}*limit", "All joints: torque saturation ratio")
    sat_ax_rms.axis("off")
    axes[0, 0].legend(loc="upper left", fontsize=6, ncol=2)
    axes[-1, 1].set_xlabel("commanded vx [m/s] (level)")
    fig.suptitle("Long ramp: joint torque by level, up (solid) vs down (hatched) — mean +/- std over seeds", y=0.995)
    fig.tight_layout()
    p = os.path.join(out_dir, "long_ramp_torque.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── 3. 관절 각속도 RMS/peak ──────────────────────────────────────
def plot_jointvel(policies, colors, out_dir, vel_limit=None):
    ref = policies[0][1][0]
    levels = sorted(set(round(v, 1) for v in ref.profile))
    groups = list(GROUPS.items())
    fig, axes = plt.subplots(len(groups), 2, figsize=(16, 4 * len(groups)))
    for row, (grp, joints) in enumerate(groups):
        rms_fn = lambda runs, joints=joints: _collect_group_stat(runs, "jvel", "rms", levels, joints)
        peak_fn = lambda runs, joints=joints: _collect_group_stat(runs, "jvel", "peak", levels, joints)
        _bar_grouped(axes[row, 0], levels, policies, colors, rms_fn, "|jvel| RMS [rad/s]", f"{grp}: joint vel RMS")
        _bar_grouped(
            axes[row, 1], levels, policies, colors, peak_fn, "|jvel| peak [rad/s]", f"{grp}: joint vel peak",
            extra_line=vel_limit,
        )
    axes[0, 0].legend(loc="upper left", fontsize=6, ncol=2)
    axes[-1, 0].set_xlabel("commanded vx [m/s] (level)")
    axes[-1, 1].set_xlabel("commanded vx [m/s] (level)")
    title = "Long ramp: joint velocity by level, up (solid) vs down (hatched) — mean +/- std over seeds"
    if vel_limit is not None:
        title += f" (dotted = velocity_limit_sim {vel_limit:.1f} rad/s)"
    fig.suptitle(title, y=0.995)
    fig.tight_layout()
    p = os.path.join(out_dir, "long_ramp_jointvel.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── 4. 레벨별 보행 격자 ──────────────────────────────────────────
GAIT_COLOR = {
    "trot": "tab:blue", "pace": "tab:orange", "gallop": "tab:red", "bound": "tab:purple",
    "pronk": "tab:brown", "stand": "0.75", "other": "tab:pink", "-": "white",
}


def plot_gait(policies, out_dir):
    ref = policies[0][1][0]
    levels = sorted(set(round(v, 1) for v in ref.profile))
    rows = []  # (row_label, [gait per level-up, gait per level-down])
    for label, runs in policies:
        for r in runs:
            up_g = {}
            down_g = {}
            for st in r.stages:
                if st["fallen"]:
                    g = "FALL"
                else:
                    g = r.gait_at(st)
                lvl = round(st["level"], 1)
                (up_g if st["direction"] == "up" else down_g)[lvl] = g
            rows.append((f"{label} {r.seed_label}", up_g, down_g))

    n_rows = len(rows)
    n_cols = len(levels)
    fig, axes = plt.subplots(1, 2, figsize=(2.0 + 0.9 * n_cols * 2, 0.6 * n_rows + 1.5))
    for ax, direction, gdict_key in zip(axes, ("Ascending", "Descending"), (1, 2)):
        for ri, row in enumerate(rows):
            gd = row[gdict_key]
            for ci, lvl in enumerate(levels):
                g = gd.get(lvl, "-")
                color = GAIT_COLOR.get(g, "tab:gray") if g != "FALL" else "black"
                ax.add_patch(plt.Rectangle((ci, n_rows - 1 - ri), 1, 1, facecolor=color, edgecolor="0.5", lw=0.5))
                txt_color = "white" if g in ("FALL",) else "black"
                ax.text(ci + 0.5, n_rows - 1 - ri + 0.5, g[:4], ha="center", va="center", fontsize=6, color=txt_color)
        ax.set_xlim(0, n_cols)
        ax.set_ylim(0, n_rows)
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels([f"{lv:.1f}" for lv in levels], fontsize=7)
        ax.set_yticks(np.arange(n_rows) + 0.5)
        ax.set_yticklabels([row[0] for row in reversed(rows)], fontsize=7)
        ax.set_xlabel("commanded vx [m/s]")
        ax.set_title(f"{direction} hold gait")
    fig.suptitle("Long ramp gait classification per level/seed (FALL = folded after collapse)")
    fig.tight_layout()
    p = os.path.join(out_dir, "long_ramp_gait.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


# ── surveys.md 작성용 수치 표(터미널에 markdown 그대로 출력, 파일로는 안 남긴다) ──
def print_md_tables(policies, colors):
    ref = policies[0][1][0]
    levels = sorted(set(round(v, 1) for v in ref.profile))
    all_joints = [j for js in GROUPS.values() for j in js]

    def fmt_row(label, direction, get):
        cells = []
        for lvl in levels:
            vals = get(lvl, direction)
            cells.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}" if vals else "-")
        return f"| {label} | " + " | ".join(cells) + " |"

    header = "| policy/dir | " + " | ".join(f"{lv:.1f}" for lv in levels) + " |"
    sep = "|---" * (len(levels) + 1) + "|"

    print("\n#### 달성률 vx/vx_cmd (mean±std over seeds)")
    print(header)
    print(sep)
    for label, runs in policies:
        rate, _ = _collect_rate_and_err(runs, levels)
        for direction in ("up", "down"):
            print(fmt_row(f"{label} {direction}", direction, lambda lvl, d: rate.get(lvl, {}).get(d, [])))

    print("\n#### 절대 오차 |vx-vx_cmd| [m/s] (mean±std)")
    print(header)
    print(sep)
    for label, runs in policies:
        _, err = _collect_rate_and_err(runs, levels)
        for direction in ("up", "down"):
            print(fmt_row(f"{label} {direction}", direction, lambda lvl, d: err.get(lvl, {}).get(d, [])))

    print("\n#### 토크 포화율 frac(|tau|>0.95*limit), 전 관절 (mean±std)")
    print(header)
    print(sep)
    for label, runs in policies:
        sat = _collect_sat_frac(runs, levels, all_joints)
        for direction in ("up", "down"):
            print(fmt_row(f"{label} {direction}", direction, lambda lvl, d: sat.get(lvl, {}).get(d, [])))

    print("\n#### 관절 각속도 peak [rad/s], 그룹별 최댓값(전 그룹 중 최대) (mean±std)")
    print(header)
    print(sep)
    for label, runs in policies:
        per_group = {grp: _collect_group_stat(runs, "jvel", "peak", levels, joints) for grp, joints in GROUPS.items()}
        for direction in ("up", "down"):
            def get(lvl, d, per_group=per_group):
                best = []
                for grp_data in per_group.values():
                    vals = grp_data.get(lvl, {}).get(d, [])
                    if vals:
                        best.append(vals)
                if not best:
                    return []
                # 그룹별 seed-mean 이 가장 큰 그룹을 대표로
                means = [np.mean(v) for v in best]
                return best[int(np.argmax(means))]
            print(fmt_row(f"{label} {direction}", direction, get))

    print("\n#### 낙상/자세 판정(자체 per-stage folded 기준, ramp_fall_probe 는 참고용)")
    for label, runs in policies:
        for r in runs:
            n_fallen = sum(1 for st in r.stages if st["fallen"])
            print(
                f"- {label} {r.seed_label}: per-stage fallen={n_fallen}/{len(r.stages)}"
                f" | ramp_fall_probe verdict={r.fall_verdict} {r.fall}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", action="append", required=True, help="label=glob (e.g. condmlp50k=.../condmlp50k_s*/ramp_data.npz)")
    ap.add_argument("--colors", nargs="+", default=["tab:blue", "tab:orange"])
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vel_limit", type=float, default=None, help="velocity_limit_sim [rad/s] — 선을 그릴 값. 없으면 생략")
    ap.add_argument("--print_tables", action="store_true", help="surveys.md 에 옮겨 적을 markdown 표를 stdout 에 출력")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    specs = parse_specs(args.glob)
    for label, paths in specs:
        print(f"[{label}] {len(paths)} npz: {paths}")
    policies = load_policies(specs)
    colors = args.colors[: len(policies)]

    outs = []
    outs.append(plot_tracking(policies, colors, args.out_dir))
    outs.append(plot_torque(policies, colors, args.out_dir))
    outs.append(plot_jointvel(policies, colors, args.out_dir, vel_limit=args.vel_limit))
    outs.append(plot_gait(policies, args.out_dir))

    print("저장 완료:")
    for p in outs:
        print(f"  {p}")

    print("\n낙상 판정(ramp_fall_probe):")
    for label, runs in policies:
        for r in runs:
            print(f"  {label:16s} {r.seed_label:20s} verdict={r.fall_verdict:26s} {r.fall}")

    if args.print_tables:
        print_md_tables(policies, colors)


if __name__ == "__main__":
    raise SystemExit(main())

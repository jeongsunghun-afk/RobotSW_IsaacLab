# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Post-hoc validity checks on pronk_cost_result.npz (numpy only, no Isaac).

Confirms: (1) tilt/low_height fire anywhere across ALL steps (real vs capture artifact),
(2) mean forward speed per terrain (traversal gate, independent of goal detection),
(3) goal/failure cause-flag capture reliability.
"""
import numpy as np

d = np.load("/home/lgb/IsaacLab/_workspace/pronk_cost_result.npz", allow_pickle=True)
T = int(d["num_steps"]); N = int(d["num_envs"]); dt = float(d["dt"])
cls = d["arr_class"]; rootxy = d["arr_rootxy"]; dones = d["arr_dones"].astype(bool)
c_tilt = d["arr_c_tilt"].astype(bool); c_low = d["arr_c_low"].astype(bool)
c_base = d["arr_c_base"].astype(bool); c_goal = d["arr_c_goal"].astype(bool)
NAMES = {0: "flat", 1: "hurdle", 2: "step", 3: "gap", 4: "stair"}

print(f"T={T} N={N} dt={dt}")
print("\n=== (1) termination cause flags across ALL steps (not just done-steps) ===")
print(f"  tilt  fired on ANY step:        {int(c_tilt.sum())} step-env occurrences")
print(f"  low_height fired on ANY step:   {int(c_low.sum())} step-env occurrences")
print(f"  base_contact fired on ANY step: {int(c_base.sum())} step-env occurrences")
print(f"  goal_reached fired on ANY step: {int(c_goal.sum())} step-env occurrences")
print("  -> these are post-step reads; goal_reached is zeroed in _reset_idx before sampling (blind).")
print("  -> tilt/low_height are NOT zeroed in reset, so non-zero here = real falls; zero = policy rarely falls.")

print("\n=== (1b) cause flags AT done-steps only (what the script's episode classifier saw) ===")
for nm, arr in [("tilt", c_tilt), ("low", c_low), ("base", c_base), ("goal", c_goal)]:
    print(f"  {nm:<6} at done-steps: {int(arr[dones].sum())}")

print("\n=== (2) mean forward speed per terrain (traversal gate) ===")
# per-step displacement magnitude, drop teleport (>0.5m) jumps from resets
print(f"  {'terrain':<10}{'mean_speed_mps':>16}{'n_steps':>10}")
for c in sorted(set(int(x) for x in np.unique(cls))):
    m = cls == c  # (T,N)
    speeds = []
    for n in range(N):
        col = m[:, n]
        xy = rootxy[:, n, :]
        step_d = np.linalg.norm(np.diff(xy, axis=0), axis=1)  # (T-1,)
        pair = col[1:] & col[:-1]
        step_d = np.where(step_d > 0.5, np.nan, step_d)  # drop reset teleports
        sel = step_d[pair]
        sel = sel[np.isfinite(sel)]
        if sel.size:
            speeds.append(sel)
    if speeds:
        alls = np.concatenate(speeds)
        print(f"  {NAMES.get(c, c):<10}{alls.mean()/dt:>16.3f}{alls.size:>10}")

print("\n=== (3) episode count by terminating done-step cause ===")
n_done = int(dones.sum())
print(f"  total done events: {n_done}")
# classify each done by cause flags present at that done step
fail = (c_tilt | c_low | c_base) & dones
goal = c_goal & dones
neither = dones & ~(c_tilt | c_low | c_base | c_goal)
print(f"  done w/ failure flag: {int(fail.sum())}")
print(f"  done w/ goal flag:    {int(goal.sum())}  (KNOWN-BLIND: goal zeroed pre-read)")
print(f"  done w/ NEITHER flag: {int(neither.sum())}  (timeout OR blind-goal)")

print("\n=== (3b) max episode length check ===")
# episode length = steps between consecutive dones per env
lens = []
for n in range(N):
    ds = np.nonzero(dones[:, n])[0]
    prev = -1
    for x in ds:
        lens.append(x - prev); prev = x
lens = np.array(lens)
if lens.size:
    print(f"  episode lengths: mean={lens.mean():.1f} median={np.median(lens):.0f} "
          f"min={lens.min()} max={lens.max()} (steps)")
    print(f"  ~max_episode_length = {int(np.percentile(lens,95))} steps = {np.percentile(lens,95)*dt:.1f}s")

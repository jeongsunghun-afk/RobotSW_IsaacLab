# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`amp_obs_clip_probe.py` 가 남긴 npz 를 읽어 클립별 2×2 교차표를 markdown 으로 찍는다."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = "reports/leg_imitation/_comparisons/conditional_discriminator/metrics/clip_probe"
TAGS = [("condmlp50k", "A' mlp"), ("conddrail45k", "B' drail")]


def row(tag: str, clip: str, start: str) -> dict | None:
    p = os.path.join(ROOT, tag, f"{clip}_{start}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    n_kept = int(z["n_kept"])
    out = {
        "kept": n_kept,
        "dropped": int(z["n_dropped"]),
        "n_envs": int(z["n_envs"]),
        "speed": float(z["clip_mean_speed"]),
        "clip_frac": float(z["clip_frac"]),
        "vy_drop": float(z["vy_dropped_mean"]),
        "clip_len": float(z["clip_len"]),
    }
    cmd = z["cmd"]  # [T, 3]
    if n_kept > 0:
        meas = z["meas"]  # [K, T, 3]
        err = np.abs(meas - cmd[None]).mean(axis=(0, 1))
        out["err_vx"], out["err_vy"], out["err_yaw"] = (float(e) for e in err)
        cv = float(np.abs(cmd[:, 0]).mean())
        out["ach"] = float(meas[..., 0].mean() / cv) if cv > 1e-6 else float("nan")
        for a in ("mlp", "drail"):
            for side in ("expert", "policy"):
                k = f"d_{a}_{side}"
                out[k] = float(z[k].mean()) if k in z.files else float("nan")
    else:
        for k in ("err_vx", "err_vy", "err_yaw", "ach"):
            out[k] = float("nan")
        for a in ("mlp", "drail"):
            for side in ("expert", "policy"):
                out[f"d_{a}_{side}"] = float("nan")
    return out


def main() -> None:
    start_modes = sys.argv[1:] or ["rsi", "stand"]
    any_tag = TAGS[0][0]
    clips = sorted({f.rsplit("_", 1)[0] for f in os.listdir(os.path.join(ROOT, any_tag)) if f.endswith(".npz")})
    z0 = np.load(os.path.join(ROOT, any_tag, f"{clips[0]}_{start_modes[0]}.npz"), allow_pickle=True)
    print(f"layout = {json.loads(str(z0['layout']))}")
    print(f"drail_reps = {int(z0['drail_reps'])}   dt = {float(z0['dt']):.4f}\n")

    for start in start_modes:
        print(f"\n### start = {start}\n")
        print(
            "| clip | v̄ [m/s] | 정책 | 완주 | Δvx 오차 | Δyaw 오차 | 달성률 |"
            " D_mlp(e) | D_mlp(p) | D_drail(e) | D_drail(p) | clip_frac |"
        )
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for clip in clips:
            for tag, label in TAGS:
                r = row(tag, clip, start)
                if r is None:
                    print(f"| {clip} | | {label} | 미확인 | | | | | | | | |")
                    continue
                print(
                    f"| {clip} | {r['speed']:.2f} | {label} | {r['kept']}/{r['n_envs']} |"
                    f" {r['err_vx']:.3f} | {r['err_yaw']:.3f} | {r['ach']:.3f} |"
                    f" {r['d_mlp_expert']:.3f} | {r['d_mlp_policy']:.3f} |"
                    f" {r['d_drail_expert']:.3f} | {r['d_drail_policy']:.3f} | {r['clip_frac']:.2f} |"
                )


if __name__ == "__main__":
    main()

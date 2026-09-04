# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""확률적 행동 프로브 표 — 결정론 mean · 학습경로 mean · 학습경로 표본추출을 나란히 놓는다."""

from __future__ import annotations

import os

import numpy as np

ROOT = "reports/leg_imitation/_comparisons/conditional_discriminator/metrics/clip_probe"
TAGS = [("condmlp50k", "A' mlp"), ("conddrail45k", "B' drail")]
ARMS = [("", "결정론 mean (act_inference)"), ("_trainmean", "학습경로 mean"), ("_stoch", "학습경로 표본추출")]
CLIPS = ["leg_run1", "leg_trot0", "leg_walk1"]


def stats(tag: str, clip: str, suffix: str) -> dict | None:
    p = os.path.join(ROOT, tag, f"{clip}_rsi{suffix}.npz")
    if not os.path.exists(p):
        return None
    z = np.load(p, allow_pickle=True)
    if int(z["n_kept"]) == 0:
        return {"kept": 0}
    cmd, meas, jv = z["cmd"], z["meas"], z["policy_jvel"]
    return {
        "kept": int(z["n_kept"]),
        "n_envs": int(z["n_envs"]),
        "err_vx": float(np.abs(meas - cmd[None]).mean(axis=(0, 1))[0]),
        "jvel_rms": float(np.sqrt((jv.astype(np.float64) ** 2).mean())),
        "d_mlp": float(z["d_mlp_policy"].mean()),
        "d_drail": float(z["d_drail_policy"].mean()),
        "clip_frac": float(z["act_clip_frac"]) if "act_clip_frac" in z.files else float("nan"),
        "std": float(z["action_std"]) if "action_std" in z.files else float("nan"),
    }


def main() -> None:
    print("| clip | 정책 | 행동 경로 | 완주 | Δvx 오차 | 관절속도 RMS | D_mlp(p) | D_drail(p) | 행동 clip 비율 |")
    print("|---|---|---|---|---|---|---|---|---|")
    for clip in CLIPS:
        for tag, label in TAGS:
            for suffix, arm in ARMS:
                r = stats(tag, clip, suffix)
                if r is None:
                    print(f"| {clip} | {label} | {arm} | 미확인 | | | | | |")
                    continue
                if r["kept"] == 0:
                    print(f"| {clip} | {label} | {arm} | 0/256 | 전멸 | | | | |")
                    continue
                cf = "-" if np.isnan(r["clip_frac"]) else f"{r['clip_frac']:.3f}"
                print(
                    f"| {clip} | {label} | {arm} | {r['kept']}/{r['n_envs']} | {r['err_vx']:.3f} |"
                    f" {r['jvel_rms']:.2f} | {r['d_mlp']:.3f} | {r['d_drail']:.3f} | {cf} |"
                )


if __name__ == "__main__":
    main()

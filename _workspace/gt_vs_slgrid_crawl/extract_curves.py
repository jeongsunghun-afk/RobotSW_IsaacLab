"""Extract comparable training curves for GT-voxel teacher vs SL-Grid+crawl from tfevents.

Writes one tidy CSV per run: iteration, tag, value (downsampled to every DECIM-th point).

Comparability note (see report): Train/mean_reward is NOT comparable across the two runs
(per-run AMP discriminator + different terrain proportions). It is extracted for within-run
trend only. The termination-cause and non-AMP reward terms use identical definitions.
"""

import csv
import sys

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = {
    "gtvoxel": "/home/lgb/IsaacLab-6.0/logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k",
    "slgrid_crawl": "/home/lgb/IsaacLab-6.0/logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl/2026-08-05_12-57-11_slgrid_crawl_scratch_50k",
}

TAGS = [
    "Train/mean_reward",  # within-run trend ONLY — not comparable across runs
    "Train/mean_episode_length",
    "Policy/mean_noise_std",
    "Loss/value",
    "Episode_Termination/cause_goal_reached",
    "Episode_Termination/cause_base_contact",
    "Episode_Termination/cause_tilt",
    "Episode_Termination/time_out",
    "Episode_Reward/collision",
    "Episode_Reward/feet_stumble",
    "Episode_Reward/tracking_goal_vel",
]
# terrain-level tags are added dynamically (tag set differs between the two tasks)
TERRAIN_PREFIX = "curriculum/mean_terrain_level"

OUT_DIR = "/home/lgb/IsaacLab-6.0/_workspace/gt_vs_slgrid_crawl"


def main() -> None:
    for name, path in RUNS.items():
        print(f"[{name}] loading {path} ...", flush=True)
        ea = EventAccumulator(path, size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags()["scalars"])
        tags = [t for t in TAGS if t in available]
        tags += sorted(t for t in available if t.startswith(TERRAIN_PREFIX))
        missing = [t for t in TAGS if t not in available]
        if missing:
            print(f"[{name}] MISSING tags: {missing}", flush=True)

        out = f"{OUT_DIR}/curves_{name}.csv"
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["iteration", "tag", "value"])
            for t in tags:
                events = ea.Scalars(t)
                for e in events:
                    w.writerow([e.step, t, f"{e.value:.6g}"])
                print(f"[{name}] {t}: {len(events)} points", flush=True)
        print(f"[{name}] -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

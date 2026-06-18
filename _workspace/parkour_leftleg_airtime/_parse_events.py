"""Streaming TensorBoard event parser for a large (>300MB) event file.

Avoids EventAccumulator (loads everything into RAM). Uses summary_iterator to
stream sequentially, collecting (1) the full tag set and (2) downsampled
(step, value) series for tags of interest. Safe to run while training is still
writing to the file — we just read what exists.
"""
import json
import sys

PATH = sys.argv[1]
OUT = sys.argv[2]

# tags whose substring marks them interesting
INTEREST = ("air_time", "episode_length", "tracking_goal_vel", "mean_reward",
            "total_reward", "feet_dragging", "feet_edge", "feet_stumble",
            "rew_", "Reward", "length")

try:
    from tensorflow.python.summary.summary_iterator import summary_iterator
except Exception:
    from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

    def summary_iterator(path):
        for ev in EventFileLoader(path).Load():
            yield ev

all_tags = set()
series = {}  # tag -> list of (step, value)
n = 0
last_step = 0
for ev in summary_iterator(PATH):
    if not ev.summary.value:
        continue
    last_step = ev.step
    for v in ev.summary.value:
        all_tags.add(v.tag)
        if any(k.lower() in v.tag.lower() for k in INTEREST):
            val = v.simple_value
            series.setdefault(v.tag, []).append((int(ev.step), float(val)))
    n += 1
    if n % 200000 == 0:
        print(f"...scanned {n} events, last_step={last_step}", flush=True)

# downsample each series to <=200 points
def downsample(pairs, k=200):
    if len(pairs) <= k:
        return pairs
    stride = len(pairs) // k
    return pairs[::stride]

out = {
    "path": PATH,
    "n_events": n,
    "last_step": last_step,
    "all_tags": sorted(all_tags),
    "series": {t: downsample(p) for t, p in series.items()},
}
with open(OUT, "w") as f:
    json.dump(out, f, indent=1)
print(f"DONE n_events={n} last_step={last_step} n_tags={len(all_tags)} interest_series={len(series)}", flush=True)
print("OUT=", OUT, flush=True)

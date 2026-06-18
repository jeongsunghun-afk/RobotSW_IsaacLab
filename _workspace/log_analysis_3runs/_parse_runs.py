import os, sys, json
from tensorboard.backend.event_processing import event_accumulator

RUNS = {
    "run1_positive_work": "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_12-13-02_positive_work",
    "run2_no_duty_time_cap": "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_16-21-24_no_duty_time_cap",
    "run3_positive_work_0.01": "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_17-28-23_positive_work_0.01",
}

def load(path):
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS: 0})
    ea.Reload()
    return ea

def series(ea, tag):
    ev = ea.Scalars(tag)
    return [(e.step, e.value) for e in ev]

def stats_at(s, frac_lo, frac_hi):
    # mean over a fractional window of the steps
    if not s: return None
    n = len(s)
    lo = int(n*frac_lo); hi = max(lo+1, int(n*frac_hi))
    vals = [v for _,v in s[lo:hi]]
    return sum(vals)/len(vals)

def summarize(s):
    if not s: return None
    steps = [st for st,_ in s]
    vals = [v for _,v in s]
    return {
        "n": len(s),
        "step_min": steps[0], "step_max": steps[-1],
        "first": vals[0], "last": vals[-1],
        "min": min(vals), "max": max(vals),
        "mean_early_0-10%": stats_at(s,0.0,0.10),
        "mean_mid_45-55%": stats_at(s,0.45,0.55),
        "mean_late_90-100%": stats_at(s,0.90,1.0),
    }

out = {}
all_tags = {}
for name, path in RUNS.items():
    ea = load(path)
    tags = ea.Tags()["scalars"]
    all_tags[name] = sorted(tags)
    out[name] = {}
    for t in tags:
        try:
            s = series(ea, t)
            out[name][t] = summarize(s)
        except Exception as e:
            out[name][t] = {"err": str(e)}

# print tag list
print("===== SCALAR TAGS PER RUN =====")
for name in RUNS:
    print(f"\n--- {name} ({len(all_tags[name])} tags) ---")
    for t in all_tags[name]:
        print("  ", t)

# tag set diff
sets = {n:set(t) for n,t in all_tags.items()}
common = set.intersection(*sets.values())
print("\n===== TAGS NOT COMMON TO ALL 3 =====")
for name in RUNS:
    diff = sets[name]-common
    if diff:
        print(f"{name} extra:", sorted(diff))

with open("/home/lgb/IsaacLab/_workspace/log_analysis_3runs/_parsed_summary.json","w") as f:
    json.dump({"tags":all_tags,"summary":out}, f, indent=2)
print("\nWROTE _parsed_summary.json")

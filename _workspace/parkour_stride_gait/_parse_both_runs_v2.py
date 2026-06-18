#!/usr/bin/env python3
"""Parse both baseline and current TensorBoard event files using tf.compat.v1."""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging

import json
import sys
from collections import defaultdict

# Suppress TF warnings and use compat v1
import warnings
warnings.filterwarnings('ignore')

def get_summary_iterator(path):
    """Get summary iterator, trying compat.v1 first."""
    try:
        import tensorflow.compat.v1 as tf
        tf.logging.set_verbosity(tf.logging.ERROR)
        return tf.train.summary_iterator(path)
    except Exception as e1:
        try:
            from tensorflow.python.summary.summary_iterator import summary_iterator
            return summary_iterator(path)
        except Exception as e2:
            print(f"Failed both approaches: {e1}, {e2}")
            return None

def parse_run(event_path, run_name):
    """Parse a single run's events and return metrics."""
    print(f"\n[{run_name}] Parsing {event_path}...")

    all_tags = set()
    series = defaultdict(list)  # tag -> [(step, value)]
    n_events = 0
    last_step = 0

    summary_iter = get_summary_iterator(event_path)
    if summary_iter is None:
        print(f"  ✗ Failed to create summary iterator")
        return None

    try:
        for ev in summary_iter:
            if not hasattr(ev, 'summary') or not ev.summary.value:
                continue

            last_step = ev.step
            n_events += 1

            for v in ev.summary.value:
                all_tags.add(v.tag)
                try:
                    val = v.simple_value
                    series[v.tag].append((int(ev.step), float(val)))
                except Exception:
                    pass

            if n_events % 200000 == 0:
                print(f"  ...scanned {n_events} events, last_step={last_step}", flush=True)
    except Exception as e:
        print(f"  WARNING: Parsing stopped at event {n_events}: {e}")

    print(f"  ✓ Parsed {n_events} events, last_step={last_step}, {len(all_tags)} unique tags")

    return {
        "run_name": run_name,
        "event_path": event_path,
        "n_events": n_events,
        "last_step": last_step,
        "all_tags": sorted(all_tags),
        "series": dict(series),
    }

def extract_stats(series_dict):
    """Extract summary statistics from series data."""
    stats = {}

    if not series_dict:
        return stats

    for tag, data in series_dict.items():
        if not data:
            continue

        vals = [v for _, v in data]
        stats[tag] = {
            "count": len(vals),
            "mean": sum(vals) / len(vals),
            "min": min(vals),
            "max": max(vals),
        }

        # Early/late averages
        n = len(vals)
        n_early = max(10, n // 10)
        n_late = max(10, n // 10)

        stats[tag]["early_avg"] = sum(vals[:n_early]) / n_early if n_early > 0 else 0
        stats[tag]["late_avg"] = sum(vals[-n_late:]) / n_late if n_late > 0 else 0

        # Std dev
        avg = stats[tag]["mean"]
        variance = sum((x - avg) ** 2 for x in vals) / len(vals)
        stats[tag]["std"] = variance ** 0.5

    return stats

def main():
    baseline_path = "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-06-02_16-12-18_air_time_cap_reward/events.out.tfevents.1780384371.server.1323854.0"
    current_path = "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-06-05_16-02-33_contact_duty_deficit_v1/events.out.tfevents.1780642969.server.1784681.0"

    out_baseline = "/home/lgb/IsaacLab/_workspace/parkour_stride_gait/_parsed_baseline.json"
    out_current = "/home/lgb/IsaacLab/_workspace/parkour_stride_gait/_parsed_current.json"

    # Parse both runs
    baseline_data = parse_run(baseline_path, "BASELINE")
    if baseline_data is None:
        print("✗ Failed to parse baseline")
        sys.exit(1)

    current_data = parse_run(current_path, "CURRENT")
    if current_data is None:
        print("✗ Failed to parse current")
        sys.exit(1)

    # Extract stats
    baseline_stats = extract_stats(baseline_data["series"])
    current_stats = extract_stats(current_data["series"])

    # Save results
    result_baseline = {
        "run_name": "baseline",
        "n_events": baseline_data["n_events"],
        "last_step": baseline_data["last_step"],
        "all_tags": baseline_data["all_tags"],
        "stats": baseline_stats,
    }

    result_current = {
        "run_name": "current",
        "n_events": current_data["n_events"],
        "last_step": current_data["last_step"],
        "all_tags": current_data["all_tags"],
        "stats": current_stats,
    }

    with open(out_baseline, "w") as f:
        json.dump(result_baseline, f, indent=2)

    with open(out_current, "w") as f:
        json.dump(result_current, f, indent=2)

    print(f"\n✓ Baseline metrics saved to {out_baseline}")
    print(f"✓ Current metrics saved to {out_current}")

    print("\n" + "="*80)
    print("KEY TAGS FOUND")
    print("="*80)

    # Show interesting tags
    interest_keys = {"feet_dragging", "feet_edge", "feet_stumble", "mean_reward",
                     "hip", "calf", "thigh", "episode_length", "goal_reached", "tilt", "air_time"}

    baseline_tags = {t for t in baseline_data["all_tags"]
                     if any(k in t.lower() for k in interest_keys)}
    current_tags = {t for t in current_data["all_tags"]
                    if any(k in t.lower() for k in interest_keys)}

    all_interest_tags = sorted(baseline_tags | current_tags)

    print(f"\nFound {len(all_interest_tags)} interesting tags:")
    for tag in all_interest_tags[:30]:  # Show first 30
        has_b = "✓" if tag in baseline_tags else " "
        has_c = "✓" if tag in current_tags else " "
        print(f"  [{has_b}B] [{has_c}C] {tag}")

    if len(all_interest_tags) > 30:
        print(f"  ... and {len(all_interest_tags) - 30} more")

if __name__ == "__main__":
    main()

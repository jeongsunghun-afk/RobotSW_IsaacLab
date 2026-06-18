#!/usr/bin/env python3
"""Parse both baseline and current TensorBoard event files.

Extracts:
- feet_dragging, feet_edge, feet_stumble (shuffle proxy)
- hip joint actions (FL/FR/RL/RR_hip)
- calf/hip/thigh actions for all 4 legs
- reward metrics
- episode_length, terrain_level, goal_reached, tilt
- feet_air_time (if available)
"""
import json
import sys
from collections import defaultdict

def summary_iterator(path):
    """Try tensorflow first, fallback to tensorboard."""
    try:
        from tensorflow.python.summary.summary_iterator import summary_iterator as si
        return si(path)
    except Exception:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        for ev in EventFileLoader(path).Load():
            yield ev

def parse_run(event_path, run_name):
    """Parse a single run's events and return metrics."""
    print(f"\n[{run_name}] Parsing {event_path}...")

    all_tags = set()
    series = defaultdict(list)  # tag -> [(step, value)]
    n_events = 0
    last_step = 0

    try:
        for ev in summary_iterator(event_path):
            if not ev.summary.value:
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
        print(f"  WARNING: Parsing error at event {n_events}: {e}")

    print(f"  ✓ Parsed {n_events} events, last_step={last_step}, {len(all_tags)} unique tags")

    return {
        "run_name": run_name,
        "event_path": event_path,
        "n_events": n_events,
        "last_step": last_step,
        "all_tags": sorted(all_tags),
        "series": dict(series),
    }

def extract_metrics(series_data):
    """Extract and compute summary statistics."""
    result = {}

    # Helper to compute early/late averages
    def early_late_avg(tag, name=None):
        name = name or tag
        if tag not in series_data["series"]:
            return {f"{name}": "NOT_FOUND"}

        series = series_data["series"][tag]
        if not series:
            return {f"{name}": "EMPTY"}

        vals = [v for _, v in series]
        n = len(vals)

        # Early: first 10% or first 10 points
        n_early = max(10, n // 10)
        early_avg = sum(vals[:n_early]) / n_early if n_early > 0 else 0

        # Late: last 10% or last 10 points
        n_late = max(10, n // 10)
        late_avg = sum(vals[-n_late:]) / n_late if n_late > 0 else 0

        return {
            f"{name}_early": round(early_avg, 4),
            f"{name}_late": round(late_avg, 4),
            f"{name}_all_avg": round(sum(vals) / len(vals), 4),
            f"{name}_min": round(min(vals), 4),
            f"{name}_max": round(max(vals), 4),
        }

    # Helper to compute std dev
    def std_dev(tag, name=None):
        name = name or tag
        if tag not in series_data["series"]:
            return {}
        series = series_data["series"][tag]
        if len(series) < 2:
            return {}
        vals = [v for _, v in series]
        avg = sum(vals) / len(vals)
        variance = sum((x - avg) ** 2 for x in vals) / len(vals)
        return {f"{name}_std": round(variance ** 0.5, 4)}

    # Shuffle proxy metrics
    result.update(early_late_avg("Episode_Reward/feet_dragging", "feet_dragging"))
    result.update(std_dev("Episode_Reward/feet_dragging", "feet_dragging"))
    result.update(early_late_avg("Episode_Reward/feet_edge", "feet_edge"))
    result.update(early_late_avg("Episode_Reward/feet_stumble", "feet_stumble"))

    # Hip actions (4 legs)
    for leg in ["FL", "FR", "RL", "RR"]:
        tag = f"action_stats/action_{leg}_hip_mean"
        result.update(early_late_avg(tag, f"{leg}_hip_mean"))
        result.update(std_dev(tag, f"{leg}_hip_mean"))

        tag_std = f"action_stats/action_{leg}_hip_std"
        result.update(early_late_avg(tag_std, f"{leg}_hip_std"))

    # Calf actions (4 legs)
    for leg in ["FL", "FR", "RL", "RR"]:
        tag = f"action_stats/action_{leg}_calf_mean"
        result.update(early_late_avg(tag, f"{leg}_calf_mean"))
        result.update(std_dev(tag, f"{leg}_calf_mean"))

    # Thigh actions (4 legs) - optional
    for leg in ["FL", "FR", "RL", "RR"]:
        tag = f"action_stats/action_{leg}_thigh_mean"
        result.update(early_late_avg(tag, f"{leg}_thigh_mean"))

    # Reward metrics
    result.update(early_late_avg("Episode_Reward/mean_reward", "mean_reward"))
    result.update(early_late_avg("Episode_Reward/total_reward", "total_reward"))

    # Episode metrics
    result.update(early_late_avg("Episode_Reward/episode_length", "episode_length"))
    result.update(early_late_avg("Episode_Reward/terrain_level", "terrain_level"))
    result.update(early_late_avg("Episode_Reward/goal_reached", "goal_reached"))
    result.update(early_late_avg("Episode_Reward/tilt", "tilt"))

    # Air time metrics if available
    result.update(early_late_avg("Episode_Reward/feet_air_time", "feet_air_time"))
    for leg in ["FL", "FR", "RL", "RR"]:
        tag = f"Episode_Reward/air_time_{leg}"
        result.update(early_late_avg(tag, f"air_time_{leg}"))

    return result

def main():
    baseline_path = "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-06-02_16-12-18_air_time_cap_reward/events.out.tfevents.1780384371.server.1323854.0"
    current_path = "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-06-05_16-02-33_contact_duty_deficit_v1/events.out.tfevents.1780642969.server.1784681.0"

    out_baseline = "/home/lgb/IsaacLab/_workspace/parkour_stride_gait/_parsed_baseline.json"
    out_current = "/home/lgb/IsaacLab/_workspace/parkour_stride_gait/_parsed_current.json"

    # Parse both runs
    baseline_data = parse_run(baseline_path, "BASELINE")
    current_data = parse_run(current_path, "CURRENT")

    # Extract metrics
    baseline_metrics = extract_metrics(baseline_data)
    current_metrics = extract_metrics(current_data)

    # Save raw data
    with open(out_baseline, "w") as f:
        json.dump({
            "run_info": {k: v for k, v in baseline_data.items() if k != "series"},
            "metrics": baseline_metrics,
            "series_tags": list(baseline_data["series"].keys()),
        }, f, indent=2)

    with open(out_current, "w") as f:
        json.dump({
            "run_info": {k: v for k, v in current_data.items() if k != "series"},
            "metrics": current_metrics,
            "series_tags": list(current_data["series"].keys()),
        }, f, indent=2)

    print(f"\n✓ Baseline metrics saved to {out_baseline}")
    print(f"✓ Current metrics saved to {out_current}")

    # Print summary
    print("\n" + "="*80)
    print("BASELINE (air_time_cap, 3-leg)")
    print("="*80)
    for k, v in sorted(baseline_metrics.items()):
        if isinstance(v, str) or v == "NOT_FOUND" or v == "EMPTY":
            continue
        print(f"  {k:40s} = {v}")

    print("\n" + "="*80)
    print("CURRENT (contact_duty_v1, 4-leg)")
    print("="*80)
    for k, v in sorted(current_metrics.items()):
        if isinstance(v, str) or v == "NOT_FOUND" or v == "EMPTY":
            continue
        print(f"  {k:40s} = {v}")

    # Compare shuffle proxy
    print("\n" + "="*80)
    print("SHUFFLE PROXY COMPARISON")
    print("="*80)
    for metric in ["feet_dragging", "feet_edge", "feet_stumble"]:
        b_early = baseline_metrics.get(f"{metric}_early", "N/A")
        b_late = baseline_metrics.get(f"{metric}_late", "N/A")
        c_early = current_metrics.get(f"{metric}_early", "N/A")
        c_late = current_metrics.get(f"{metric}_late", "N/A")
        print(f"\n{metric}:")
        print(f"  Baseline early={b_early}, late={b_late}")
        print(f"  Current  early={c_early}, late={c_late}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate markdown report comparing baseline and current runs."""
import json
from typing import Dict, Any, Optional

def load_results(path: str) -> Dict[str, Any]:
    """Load parsed results from JSON."""
    with open(path) as f:
        return json.load(f)

def format_metric(value: Any) -> str:
    """Format a metric value."""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)

def get_stat(stats: Dict, tag: str, field: str, default: str = "—") -> str:
    """Get a statistic from the stats dict."""
    if tag not in stats:
        return default
    s = stats[tag]
    if field in s:
        return format_metric(s[field])
    return default

def main():
    baseline = load_results("/home/lgb/IsaacLab/_workspace/parkour_stride_gait/_parsed_baseline.json")
    current = load_results("/home/lgb/IsaacLab/_workspace/parkour_stride_gait/_parsed_current.json")

    b_stats = baseline["stats"]
    c_stats = current["stats"]

    print(f"Baseline: {baseline['n_events']} events, {baseline['last_step']} steps")
    print(f"Current:  {current['n_events']} events, {current['last_step']} steps")
    print()

    # Create markdown content
    md = []
    md.append("# Parkour Stride/Gait Log Analysis\n")
    md.append("## Run Comparison\n")
    md.append(f"- **Baseline**: air_time_cap (3-leg gait), {baseline['n_events']:,} events, step={baseline['last_step']}")
    md.append(f"- **Current**: contact_duty_deficit_v1 (4-leg validation), {current['n_events']:,} events, step={current['last_step']}")
    md.append("")

    # 1. Shuffle Proxy
    md.append("## 1. Shuffle Proxy (Early/Late Comparison)\n")
    md.append("Early = first 10% of run, Late = last 10% of run\n")

    # Feet dragging
    md.append("### Feet Dragging")
    md.append("接地した状態での水平移動（shuffle/slide の直接信号）\n")
    md.append("| Metric | Baseline Early | Baseline Late | Current Early | Current Late | Trend |")
    md.append("|--------|---|---|---|---|---|")

    b_drag_early = get_stat(b_stats, "Episode_Reward/feet_dragging", "early_avg")
    b_drag_late = get_stat(b_stats, "Episode_Reward/feet_dragging", "late_avg")
    c_drag_early = get_stat(c_stats, "Episode_Reward/feet_dragging", "early_avg")
    c_drag_late = get_stat(c_stats, "Episode_Reward/feet_dragging", "late_avg")

    try:
        b_d_e = float(b_drag_early)
        b_d_l = float(b_drag_late)
        c_d_e = float(c_drag_early)
        c_d_l = float(c_drag_late)
        b_delta = f"{b_d_l - b_d_e:+.4f}"
        c_delta = f"{c_d_l - c_d_e:+.4f}"
        b_trend = "↑" if b_d_l > b_d_e else "↓" if b_d_l < b_d_e else "−"
        c_trend = "↑" if c_d_l > c_d_e else "↓" if c_d_l < c_d_e else "−"
    except:
        b_delta = "—"
        c_delta = "—"
        b_trend = "—"
        c_trend = "—"

    md.append(f"| feet_dragging | {b_drag_early} | {b_drag_late} | {c_drag_early} | {c_drag_late} | Δ: {b_delta} (B), {c_delta} (C) |")
    md.append("")

    # Feet edge
    md.append("### Feet Edge")
    md.append("| Metric | Baseline Early | Baseline Late | Current Early | Current Late |")
    md.append("|--------|---|---|---|---|")

    b_edge_early = get_stat(b_stats, "Episode_Reward/feet_edge", "early_avg")
    b_edge_late = get_stat(b_stats, "Episode_Reward/feet_edge", "late_avg")
    c_edge_early = get_stat(c_stats, "Episode_Reward/feet_edge", "early_avg")
    c_edge_late = get_stat(c_stats, "Episode_Reward/feet_edge", "late_avg")

    md.append(f"| feet_edge | {b_edge_early} | {b_edge_late} | {c_edge_early} | {c_edge_late} |")
    md.append("")

    # Feet stumble
    md.append("### Feet Stumble")
    md.append("| Metric | Baseline Early | Baseline Late | Current Early | Current Late |")
    md.append("|--------|---|---|---|---|")

    b_stumble_early = get_stat(b_stats, "Episode_Reward/feet_stumble", "early_avg")
    b_stumble_late = get_stat(b_stats, "Episode_Reward/feet_stumble", "late_avg")
    c_stumble_early = get_stat(c_stats, "Episode_Reward/feet_stumble", "early_avg")
    c_stumble_late = get_stat(c_stats, "Episode_Reward/feet_stumble", "late_avg")

    md.append(f"| feet_stumble | {b_stumble_early} | {b_stumble_late} | {c_stumble_early} | {c_stumble_late} |")
    md.append("")

    # 2. Hip Joint Action Patterns
    md.append("## 2. Hip Joint Action Patterns\n")
    md.append("Hip actions drive fore-aft swing. Weak stride hypothesis: hip amplitude should be smaller.")
    md.append("Values shown: mean (late avg). Std in parentheses.\n")
    md.append("| Leg | Baseline Mean (Std) | Current Mean (Std) | Δ Mean |")
    md.append("|-----|---|---|---|")

    for leg in ["FL", "FR", "RL", "RR"]:
        hip_tag = f"action_stats/{leg}_hip_joint/mean"
        std_tag = f"action_stats/{leg}_hip_joint/sample_std"

        b_mean = get_stat(b_stats, hip_tag, "late_avg")
        b_std = get_stat(b_stats, std_tag, "late_avg")
        c_mean = get_stat(c_stats, hip_tag, "late_avg")
        c_std = get_stat(c_stats, std_tag, "late_avg")

        try:
            b_m = float(b_mean)
            c_m = float(c_mean)
            delta = f"{c_m - b_m:+.4f}"
        except:
            delta = "—"

        md.append(f"| {leg}_hip | {b_mean} ({b_std}) | {c_mean} ({c_std}) | {delta} |")

    md.append("")

    # 3. Non-Regression Baseline (Current Run)
    md.append("## 3. Non-Regression Baseline (Current Run)\n")
    md.append("These metrics define the baseline for next fix — must not degrade these values.\n")

    md.append("### RL (Right-Hind) Calf Action - Primary Indicator")
    md.append("RL_calf abnormality signals 3-leg gait regression (cf. known issue #project_parkour_3leg_reward_positive).\n")
    md.append("| Metric | Value |")
    md.append("|--------|-------|")

    rl_calf_mean = get_stat(c_stats, "action_stats/RL_calf_joint/mean", "late_avg")
    rl_calf_std = get_stat(c_stats, "action_stats/RL_calf_joint/sample_std", "late_avg")

    md.append(f"| RL_calf mean (late avg) | {rl_calf_mean} |")
    md.append(f"| RL_calf std (late avg) | {rl_calf_std} |")
    md.append("")

    md.append("### 4-Leg Action Stats (All Legs, Late Average)")
    md.append("| Joint | FL | FR | RL | RR | Min-Max Range |")
    md.append("|-------|----|----|----|----|---|")

    for joint_type in ["calf", "hip", "thigh"]:
        values = []
        for leg in ["FL", "FR", "RL", "RR"]:
            tag = f"action_stats/{leg}_{joint_type}_joint/mean"
            val = get_stat(c_stats, tag, "late_avg")
            values.append(val)

        try:
            nums = [float(v) for v in values if v != "—"]
            if nums:
                range_str = f"{min(nums):.4f}~{max(nums):.4f}"
            else:
                range_str = "—"
        except:
            range_str = "—"

        md.append(f"| {joint_type} | {values[0]} | {values[1]} | {values[2]} | {values[3]} | {range_str} |")

    md.append("")

    md.append("### Episode Rewards & Metrics (Late Average)")
    md.append("| Metric | Current Value |")
    md.append("|--------|---|")

    metrics_to_report = [
        ("Train/mean_reward", "Mean Reward"),
        ("Train/mean_episode_length", "Episode Length"),
        ("curriculum/mean_terrain_level", "Terrain Level (avg)"),
        ("Episode_Termination/cause_goal_reached", "Goal Reached Rate"),
        ("Episode_Termination/cause_tilt", "Tilt Termination Rate"),
    ]

    for tag, label in metrics_to_report:
        val = get_stat(c_stats, tag, "late_avg")
        md.append(f"| {label} | {val} |")

    md.append("")

    # 4. Feet Air Time Availability
    md.append("## 4. Feet Air Time Logging Availability\n")

    air_time_tag = "Episode_Reward/air_time_cap"
    if air_time_tag in c_stats:
        air_time_val = c_stats[air_time_tag].get("late_avg", "—")
        md.append(f"✓ **air_time_cap is logged**: late avg = {air_time_val}")
        md.append("(This is reward penalizing long air time, not duration measurement)")
    else:
        md.append("✗ **feet_air_time duration metrics not found in logs**")
        md.append("  → Per-foot air_time metrics (air_time_FL/FR/RL/RR) are not logged")
        md.append("  → Would require adding new logging to env or play.py to measure stride phases")

    md.append("")

    # Summary
    md.append("## Summary\n")
    md.append("1. **Shuffle proxy**: Compare feet_dragging early/late trend between runs")
    md.append(f"   - Baseline: {b_drag_early} → {b_drag_late}")
    md.append(f"   - Current:  {c_drag_early} → {c_drag_late}")
    md.append("2. **Hip action patterns**: Similar across both runs (supporting stride hypothesis requires hip amplitude data)")
    md.append("3. **Non-regression baseline established** for current run (RL_calf, rewards, episode metrics)")
    md.append("4. **Air time logging**: air_time_cap is logged as reward term (not duration)")
    md.append("   - To measure actual stride/swing phases, need per-foot contact time or air time measurements")

    # Write markdown
    md_content = "\n".join(md)
    out_path = "/home/lgb/IsaacLab/_workspace/parkour_stride_gait/02_log_proxy.md"
    with open(out_path, "w") as f:
        f.write(md_content)

    print(f"✓ Report written to {out_path}")
    print(f"  ({len(md)} lines)")

if __name__ == "__main__":
    main()

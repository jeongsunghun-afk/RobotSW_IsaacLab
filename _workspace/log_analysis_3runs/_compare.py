import json

d = json.load(open("/home/lgb/IsaacLab/_workspace/log_analysis_3runs/_parsed_summary.json"))
S = d["summary"]
runs = ["run1_positive_work","run2_no_duty_time_cap","run3_positive_work_0.01"]

KEY = [
 "Train/mean_reward","Train/mean_episode_length","Episode_Length/mean_at_reset",
 "Episode_Reward/positive_work","Episode_Reward/tracking_goal_vel","Episode_Reward/tracking_yaw",
 "Episode_Reward/torques_l2","Episode_Reward/contact_duty_deficit","Episode_Reward/air_time_cap",
 "Episode_Reward/feet_dragging","Episode_Reward/collision","Episode_Reward/feet_gait_pairing",
 "Episode_Reward/dof_acc_l2","Episode_Reward/action_rate_l2","Episode_Reward/orientation_l2",
 "Episode_Reward/lin_vel_z_l2","Episode_Reward/hip_pos","Episode_Reward/feet_edge","Episode_Reward/feet_stumble",
 "Loss/entropy","Loss/value","Loss/surrogate","Loss/learning_rate","Policy/mean_noise_std",
 "curriculum/mean_terrain_level","curriculum/mean_terrain_level_flat","curriculum/mean_terrain_level_stair",
 "curriculum/mean_terrain_level_gap","curriculum/mean_terrain_level_hurdle","curriculum/mean_terrain_level_step",
 "Episode_Termination/cause_base_contact","Episode_Termination/cause_goal_reached",
 "Episode_Termination/cause_low_height","Episode_Termination/cause_tilt","Episode_Termination/time_out",
]

def fmt(x):
    if x is None: return "  --  "
    if isinstance(x,(int,float)):
        if abs(x)>=1000 or (abs(x)<0.001 and x!=0): return f"{x:.3e}"
        return f"{x:.4f}"
    return str(x)

print("step ranges:")
for r in runs:
    mr = S[r].get("Train/mean_reward",{})
    print(f"  {r}: n={mr.get('n')} steps {mr.get('step_min')}..{mr.get('step_max')}")

print("\n=== EARLY (0-10%) / MID (45-55%) / LATE (90-100%) means; also last value ===\n")
hdr = f"{'metric':38} | {'run':24} | {'early':>11} | {'mid':>11} | {'late':>11} | {'last':>11} | {'min':>11} | {'max':>11}"
for k in KEY:
    print(k)
    for r in runs:
        s = S[r].get(k)
        if s is None or "err" in (s or {}):
            print(f"   {r:24} : MISSING"); continue
        print(f"   {r:24} : early={fmt(s['mean_early_0-10%']):>11} mid={fmt(s['mean_mid_45-55%']):>11} late={fmt(s['mean_late_90-100%']):>11} last={fmt(s['last']):>11} min={fmt(s['min']):>11} max={fmt(s['max']):>11}")
    print()

from tensorboard.backend.event_processing import event_accumulator

RUNS = {
 "run1": "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_12-13-02_positive_work",
 "run2": "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_16-21-24_no_duty_time_cap",
 "run3": "/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_17-28-23_positive_work_0.01",
}
TAGS = ["Train/mean_reward","Train/mean_episode_length","Policy/mean_noise_std","Loss/entropy",
        "Loss/value","Episode_Reward/tracking_goal_vel","Episode_Reward/positive_work",
        "Episode_Reward/collision","curriculum/mean_terrain_level"]

for name,path in RUNS.items():
    ea = event_accumulator.EventAccumulator(path, size_guidance={event_accumulator.SCALARS:0})
    ea.Reload()
    print(f"\n################ {name} ################")
    data = {t:{s.step:s.value for s in ea.Scalars(t)} for t in TAGS}
    steps = sorted(data["Train/mean_reward"].keys())
    N = len(steps)
    # sample 25 evenly
    idxs = [int(i*(N-1)/24) for i in range(25)]
    hdr = f"{'it':>6} {'rew':>8} {'eplen':>7} {'std':>7} {'entr':>7} {'vloss':>10} {'trkvel':>7} {'poswrk':>8} {'colli':>9} {'terr':>6}"
    print(hdr)
    for ix in idxs:
        st = steps[ix]
        def g(t):
            v=data[t].get(st); return v
        print(f"{st:>6} {g('Train/mean_reward'):>8.3f} {g('Train/mean_episode_length'):>7.0f} {g('Policy/mean_noise_std'):>7.3f} {g('Loss/entropy'):>7.2f} {g('Loss/value'):>10.2e} {g('Episode_Reward/tracking_goal_vel'):>7.3f} {g('Episode_Reward/positive_work'):>8.4f} {g('Episode_Reward/collision'):>9.3f} {g('curriculum/mean_terrain_level'):>6.2f}")

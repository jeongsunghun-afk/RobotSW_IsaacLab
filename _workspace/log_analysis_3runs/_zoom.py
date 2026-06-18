from tensorboard.backend.event_processing import event_accumulator
path="/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour_symmetry/2026-06-11_17-28-23_positive_work_0.01"
TAGS=["Train/mean_reward","Train/mean_episode_length","Policy/mean_noise_std","Loss/entropy",
 "Loss/value","Loss/learning_rate","Episode_Reward/tracking_goal_vel","Episode_Reward/positive_work",
 "Episode_Reward/collision","curriculum/mean_terrain_level"]
ea=event_accumulator.EventAccumulator(path,size_guidance={event_accumulator.SCALARS:0}); ea.Reload()
data={t:{s.step:s.value for s in ea.Scalars(t)} for t in TAGS}
print(f"{'it':>6} {'rew':>7} {'eplen':>6} {'std':>7} {'entr':>6} {'vloss':>9} {'lr':>9} {'trkvel':>7} {'poswrk':>8} {'colli':>8} {'terr':>5}")
for st in range(1100,2200,20):
    if st not in data["Train/mean_reward"]: continue
    def g(t): return data[t].get(st,float('nan'))
    print(f"{st:>6} {g('Train/mean_reward'):>7.3f} {g('Train/mean_episode_length'):>6.0f} {g('Policy/mean_noise_std'):>7.3f} {g('Loss/entropy'):>6.2f} {g('Loss/value'):>9.2e} {g('Loss/learning_rate'):>9.2e} {g('Episode_Reward/tracking_goal_vel'):>7.3f} {g('Episode_Reward/positive_work'):>8.4f} {g('Episode_Reward/collision'):>8.3f} {g('curriculum/mean_terrain_level'):>5.2f}")

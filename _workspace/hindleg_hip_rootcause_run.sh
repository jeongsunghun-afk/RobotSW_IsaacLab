#!/usr/bin/env bash
set -u; cd /home/lgb/IsaacLab-6.0
source /home/user/miniconda3/etc/profile.d/conda.sh; conda activate isaac-6.0
D=reports/hindleg_locomotion/hindLeg_history_direct/hip_saturation_rootcause/metrics
B=logs/rsl_rl/hindLeg_history_direct/2026-08-28_09-39-20_gainclamp_ft/model_39799.pt
N=logs/rsl_rl/hindLeg_history_direct/2026-09-02_09-07-14_resetnoise_ft/model_42000.pt
./isaaclab.sh -p _workspace/hindleg_hip_action_probe.py --checkpoint $B --num_envs 512 --device cuda:1 > $D/hip_action_base.txt 2>&1
echo "base rc=$?"
./isaaclab.sh -p _workspace/hindleg_hip_action_probe.py --checkpoint $N --num_envs 512 --device cuda:1 > $D/hip_action_new.txt 2>&1
echo "new rc=$?"
./isaaclab.sh -p _workspace/hindleg_hip_gravity_probe.py --device cuda:1 --hold_s 3.0 --meas_s 1.0 --n_angle 13 > $D/hip_gravity_sim.txt 2>&1
echo "grav rc=$?"
for f in $D/*.txt; do grep -vE "\[Warning\]|Failed to find|^\[INFO\]: [A-Z]" "$f" > "$f.t" && mv "$f.t" "$f"; done
touch $D/.done

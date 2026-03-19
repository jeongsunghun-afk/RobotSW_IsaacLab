import os
import torch
import numpy as np
import sys

# Adjust the path to import SkeletonMotionLoader
sys.path.append('/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton')

from motion_loader import SkeletonMotionLoader

# Get motion dataset path
motion_dir = '/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/imitation/txt_dataset_skeleton_sample'

device = "cpu"
loader = SkeletonMotionLoader(motion_files=motion_dir, device=device)

print(f"Total frames: {loader.num_frames}")
num_samples = 5
times = np.linspace(0, min(1.0, loader.duration), num_samples)

(
    dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel
) = loader.sample(num_samples=num_samples, times=times)

print("----- Sampled Data ------")
base_idx = loader.body_names.index("base")

for i in range(num_samples):
    print(f"\nTime: {times[i]:.2f}")
    base_p = body_pos[i, base_idx].numpy()
    base_q = body_rot[i, base_idx].numpy()
    print(f"Base Pos (x,y,z): {base_p}")
    print(f"Base Rot (wxyz):  {base_q}")
    
    # Check if the up vector implies upside down.
    # In Issac Sim, +Z is up. (0, 0, 1). 
    # Let's rotate (0,0,1) by base_q to see the body's up vector
    w, x, y, z = base_q
    # rotation matrix z-column
    up_x = 2 * (x * z + w * y)
    up_y = 2 * (y * z - w * x)
    up_z = 1 - 2 * (x * x + y * y)
    print(f"Body Up Vector (rotated Z): ({up_x:.2f}, {up_y:.2f}, {up_z:.2f})")
    
    for toe_name in loader.body_names[:-1]:
        toe_idx = loader.body_names.index(toe_name)
        toe_p = body_pos[i, toe_idx].numpy()
        rel_pos = toe_p - base_p
        print(f"  {toe_name} world pos: \t {np.round(toe_p, 3)}")
        print(f"  {toe_name} rel to base: \t {np.round(rel_pos, 3)}")


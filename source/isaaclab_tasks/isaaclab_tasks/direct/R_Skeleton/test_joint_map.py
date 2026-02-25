robot_joints = ['FL_joint1_shoulder_y', 'FR_joint1_shoulder_y', 'N_joint1_neck_y', 'W_joint1_waist_p', 'FL_joint2_shoulder_r', 'FR_joint2_shoulder_r', 'N_joint2_neck_p', 'W_joint2_waist_y', 'FL_joint3_shoulder_p', 'FR_joint3_shoulder_p', 'N_joint3_neck_y', 'W_joint3_waist_r', 'FL_joint4_elbow_p', 'FR_joint4_elbow_p', 'N_joint4_neck_r', 'HL_joint1_thigh_y', 'HR_joint1_thigh_y', 'FL_joint5_wrist_p', 'FR_joint5_wrist_p', 'N_joint5_neck_y', 'HL_joint2_thigh_r', 'HR_joint2_thigh_r', 'FL_joint6_wrist_r', 'FR_joint6_wrist_r', 'N_joint6_neck_p', 'HL_joint3_thigh_p', 'HR_joint3_thigh_p', 'N_joint7_neck_y', 'HL_joint4_knee_p', 'HR_joint4_knee_p', 'HL_joint5_ankle_p', 'HR_joint5_ankle_p', 'HL_joint6_ankle_r', 'HR_joint6_ankle_r']

# txt (38)
# FL: 0~5
# FR: 6~11
# RL(HL): 12~17
# RR(HR): 18~23
# Waist: 24~25
# Arm: 26~32
# Neck: 33~37

map_dict = {}
for name in robot_joints:
    if name.startswith('FL_joint'): idx = 0 + int(name[8]) - 1
    elif name.startswith('FR_joint'): idx = 6 + int(name[8]) - 1
    elif name.startswith('HL_joint'): idx = 12 + int(name[8]) - 1
    elif name.startswith('HR_joint'): idx = 18 + int(name[8]) - 1
    elif name.startswith('W_joint'):
        j = int(name[7]) - 1
        idx = 24 + min(j, 1) # 2 joints limit
    elif name.startswith('N_joint'):
        j = int(name[7]) - 1
        idx = 33 + min(j, 4) # 5 joints limit
    else:
        idx = 0
    map_dict[name] = idx

print("indexes =", [map_dict[n] for n in robot_joints])

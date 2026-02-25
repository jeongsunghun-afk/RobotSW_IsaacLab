def get_dof_index(dof_names):
    indexes = []
    for name in dof_names:
        if name.startswith('N_joint'):
            j = int(name.split('_')[1].replace('joint', '')) - 1
            idx = 0 + j
        elif name.startswith('FL_joint'):
            j = int(name.split('_')[1].replace('joint', '')) - 1
            idx = 7 + j
        elif name.startswith('FR_joint'):
            j = int(name.split('_')[1].replace('joint', '')) - 1
            idx = 14 + j
        elif name.startswith('W_joint'):
            j = int(name.split('_')[1].replace('joint', '')) - 1
            idx = 21 + j
        elif name.startswith('HL_joint') or name.startswith('RL_joint'):
            j = int(name.split('_')[1].replace('joint', '')) - 1
            idx = 24 + j
        elif name.startswith('HR_joint') or name.startswith('RR_joint'):
            j = int(name.split('_')[1].replace('joint', '')) - 1
            idx = 31 + j
        else:
            idx = 0
        indexes.append((name, idx))
    return indexes

with open("robot_joints.txt", "r") as f:
    lines = f.readlines()
names = [line.strip().split(": ")[1] for line in lines if line.strip()]

mapping = get_dof_index(names)
for item in mapping:
    print(item)

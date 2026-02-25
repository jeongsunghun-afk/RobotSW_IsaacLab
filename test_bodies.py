import sys

# SkeletonMotionLoader BODY_NAMES
BODY_NAMES = [
    "Pelvis", "N_link1", "N_link2", "N_link3", "N_link4", "N_link5", "N_link6", "N_link7",
    "FL_link1", "FL_link2", "FL_link3", "FL_link4", "FL_link5", "FL_link6", "FL_foot",
    "FR_link1", "FR_link2", "FR_link3", "FR_link4", "FR_link5", "FR_link6", "FR_foot",
    "W_link1", "W_link2", "W_link3",
    "HL_link1", "HL_link2", "HL_link3", "HL_link4", "HL_link5", "HL_link6", "HL_foot",
    "HR_link1", "HR_link2", "HR_link3", "HR_link4", "HR_link5", "HR_link6", "HR_foot"
]

def get_body_index(body_names):
    indexes = []
    for name in body_names:
        try:
            idx = BODY_NAMES.index(name)
            indexes.append(idx)
        except ValueError:
            print(f"FAILED TO FIND {name} IN BODY_NAMES")
            raise AssertionError
    return indexes

try:
    idxes = get_body_index(["FL_foot", "FR_foot", "HL_foot", "HR_foot"])
    print("Found key bodies:", idxes)
except:
    pass


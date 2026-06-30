# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import pandas as pd

# CSV 파일 읽기
df = pd.read_csv("stand_cap_v2.csv")

# 프레임 0인 데이터만 추출
frame_0 = df[df["frame"] == 0]

# 새로운 데이터프레임 생성
new_frames = []
for i in range(1, 71):  # 1부터 70까지
    temp_frame = frame_0.copy()
    temp_frame["frame"] = i
    new_frames.append(temp_frame)

# 모든 프레임 합치기
result = pd.concat(new_frames)

# CSV로 저장
result.to_csv("new_stand_cap_v2.csv", index=False)

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""tfevents 에서 조건부 AMP disc 관련 tag 만 스트리밍 추출해 npz 로 캐시한다.

Run: python _workspace/leg/tb_extract_cond_disc.py <events_file> <out.npz>
(EventAccumulator 는 50k iter 로그에서 3 분 타임아웃 — EventFileLoader 스트리밍 사용. 값은 tensor.float_val 에 있다.)
"""
import sys, time, numpy as np
from collections import defaultdict
from tensorboard.backend.event_processing.event_file_loader import EventFileLoader

path, out = sys.argv[1], sys.argv[2]
TAGS = {
 "Loss/disc_expert_output","Loss/disc_policy_output","Loss/disc_expert_loss","Loss/disc_policy_loss",
 "Loss/disc_total_loss","Loss/disc_grad_penalty","Loss/amp_weight",
 "Episode_Reward/amp_reward","Episode_Reward/lin_vel_reward","Episode_Reward/yaw_vel_reward",
 "Episode_Reward/torque_penalty","Policy/mean_noise_std","Loss/entropy","Loss/kl","Loss/surrogate",
 "Loss/value","Loss/learning_rate","Train/mean_reward","Train/mean_episode_length",
}
data = defaultdict(lambda: ([], []))
t0 = time.time(); n = 0; last = -1
try:
    for ev in EventFileLoader(path).Load():
        for v in ev.summary.value:
            if v.tag in TAGS:
                if v.HasField("tensor"):
                    t = v.tensor
                    if len(t.float_val):
                        val = float(t.float_val[0])
                    elif len(t.double_val):
                        val = float(t.double_val[0])
                    elif t.tensor_content:
                        val = float(np.frombuffer(t.tensor_content, dtype=np.float32)[0])
                    else:
                        val = float("nan")
                else:
                    val = float(v.simple_value)
                s, vv = data[v.tag]
                s.append(ev.step); vv.append(val)
        n += 1
        last = max(last, ev.step)
except Exception as e:
    print(f"WARN partial read: {type(e).__name__}: {e}", file=sys.stderr)
arrs = {}
for k, (s, vv) in data.items():
    arrs[k + "__step"] = np.asarray(s, dtype=np.int64)
    arrs[k + "__val"] = np.asarray(vv, dtype=np.float64)
np.savez_compressed(out, **arrs)
print(f"{path}\n  events={n} last_step={last} elapsed={time.time()-t0:.0f}s tags={len(data)}")
for k in sorted(data):
    print(f"  {k}: n={len(data[k][0])} step[{data[k][0][0]}..{data[k][0][-1]}]")

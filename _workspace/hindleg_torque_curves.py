# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""torque2x fine-tune: baseline 구간과 이어 붙여 스케일 변경 시점을 한 축에 놓는다."""
import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

def load(run):
    f = glob.glob(f'logs/rsl_rl/hindLeg_history_direct/{run}/events*')[0]
    a = ea.EventAccumulator(f, size_guidance={'scalars': 0}); a.Reload()
    S = {k: np.array([s.value for s in a.Scalars(k)]) for k in a.Tags()['scalars']}
    return S, np.array([s.step for s in a.Scalars(list(S)[0])])

B, bs = load('2026-09-02_09-07-14_resetnoise_ft')
T, ts = load('2026-09-07_09-18-24_torque2x_ft')
SW = ts[0]  # scale 변경 시점

def smooth(y, w=51):
    """가장자리 인공물 없는 이동평균 — mode='same' 은 양 끝에서 0 으로 끌려 내려가
    학습 막판이 붕괴한 것처럼 보인다. 끝값으로 패딩해 그 허상을 없앤다."""
    pad = w // 2
    yp = np.concatenate([np.full(pad, y[:pad].mean()), y, np.full(pad, y[-pad:].mean())])
    return np.convolve(yp, np.ones(w) / w, mode='valid')[: len(y)]

PANELS = [
    ('Episode_Reward/dof_torques_l2', 'dof_torques_l2', 'torque penalty term'),
    ('Episode_Reward/track_lin_vel_xy_exp', 'track_lin_vel_xy_exp', 'velocity tracking'),
    ('Episode_Termination/base_contact', 'base_contact terminations (not falls-per-episode)', 'terminations per iteration'),
    ('Train/mean_episode_length', 'mean_episode_length', 'steps'),
]
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for ax, (k, title, ylab) in zip(axes.ravel(), PANELS):
    ax.plot(bs, B[k], lw=.4, color='#8fa8c8', alpha=.5)
    ax.plot(bs, smooth(B[k]), lw=1.6, color='#1f4e79', label='baseline  scale −0.0002')
    ax.plot(ts, T[k], lw=.4, color='#e0a49a', alpha=.5)
    ax.plot(ts, smooth(T[k]), lw=1.6, color='#c0392b', label='torque2x  scale −0.0004')
    ax.axvline(SW, color='#555', ls='--', lw=1.0)
    if k.endswith('dof_torques_l2'):
        # 정책이 안 바뀌었다면 곧장 2배가 됐을 선
        ax.axhline(2 * B[k][-50:].mean(), color='#7f8c8d', ls=':', lw=1.2,
                   label='2x baseline (if policy unchanged)')
    ax.set_title(title, fontweight='bold', fontsize=11)
    ax.set_ylabel(ylab); ax.set_xlabel('iteration'); ax.grid(alpha=.25)
    ax.legend(fontsize=8)
fig.suptitle('hindLeg torque penalty 2x fine-tune  (dashed = scale change at iter 44798)',
             fontweight='bold')
fig.tight_layout()
out = 'reports/hindleg_locomotion/hindLeg_history_direct/torque_penalty_2x/plots/curves.png'
fig.savefig(out, dpi=130)
print('saved', out)

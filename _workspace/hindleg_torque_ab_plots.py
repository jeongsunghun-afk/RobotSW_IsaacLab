# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""명령별 A/B 요약 플롯 — cmd_ab_probe.py 가 저장한 npz 를 읽는다."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

D = 'reports/hindleg_locomotion/hindLeg_history_direct/torque_penalty_2x/'
d = np.load(D + 'cmd_ab.npz', allow_pickle=True)
CMDS = [0.0, 0.5, 1.0]
JOINTS = ["HL_hip", "HR_hip", "HL_thigh", "HR_thigh", "HL_calf", "HR_calf", "HL_foot", "HR_foot"]
CB, CT = '#1f4e79', '#c0392b'
g = lambda lbl, c, k: d[f'{lbl}|{c}|{k}']

fig, ax = plt.subplots(2, 2, figsize=(13, 8))

a = ax[0, 0]
for lbl, col in (('baseline', CB), ('torque2x', CT)):
    y = [float(g(lbl, c, 'tau_sumsq')) for c in CMDS]
    a.plot(CMDS, y, 'o-', color=col, lw=2, ms=7, label=lbl)
for i, c in enumerate(CMDS):
    b, t = float(g('baseline', c, 'tau_sumsq')), float(g('torque2x', c, 'tau_sumsq'))
    # 첫 점은 축 왼쪽 끝이라 가운데 정렬하면 잘린다 — 라벨을 오른쪽으로 밀어 둔다.
    a.annotate(f'{100 * (t - b) / b:+.1f}%', (c, t), textcoords='offset points',
               xytext=(12 if i == 0 else 0, -18), ha='left' if i == 0 else 'center',
               color=CT, fontsize=9, fontweight='bold')
a.set_title('sum(tau^2) — torque cost', fontweight='bold')
a.set_xlabel('commanded vx [m/s]'); a.set_ylabel('N^2 m^2'); a.grid(alpha=.25); a.legend()

a = ax[0, 1]
a.plot(CMDS, CMDS, ':', color='#7f8c8d', lw=1.4, label='command')
for lbl, col in (('baseline', CB), ('torque2x', CT)):
    a.plot(CMDS, [float(g(lbl, c, 'vx_mean')) for c in CMDS], 'o-', color=col, lw=2, ms=7, label=lbl)
a.set_title('achieved vx — tracking', fontweight='bold')
a.set_xlabel('commanded vx [m/s]'); a.set_ylabel('measured vx [m/s]'); a.grid(alpha=.25); a.legend()

x = np.arange(8); w = 0.38
a = ax[1, 0]
b = g('baseline', 1.0, 'tau_per_joint'); t = g('torque2x', 1.0, 'tau_per_joint')
a.bar(x - w / 2, b, w, color=CB, label='baseline'); a.bar(x + w / 2, t, w, color=CT, label='torque2x')
for i in range(8):
    a.text(i, max(b[i], t[i]) + .15, f'{100 * (t[i] - b[i]) / b[i]:+.1f}%', ha='center', fontsize=8,
           color=CT if t[i] < b[i] else '#7f8c8d')
a.set_xticks(x); a.set_xticklabels(JOINTS, rotation=35, ha='right', fontsize=9)
a.set_title('per-joint |tau| mean  @ cmd 1.0', fontweight='bold'); a.set_ylabel('N·m')
a.grid(axis='y', alpha=.25); a.legend(fontsize=9)

a = ax[1, 1]
b = 100 * g('baseline', 1.0, 'trip_frac'); t = 100 * g('torque2x', 1.0, 'trip_frac')
a.bar(x - w / 2, b, w, color=CB, label='baseline'); a.bar(x + w / 2, t, w, color=CT, label='torque2x')
a.set_xticks(x); a.set_xticklabels(JOINTS, rotation=35, ha='right', fontsize=9)
a.set_title('channel 15 N·m exceedance  @ cmd 1.0', fontweight='bold'); a.set_ylabel('% of steps')
a.grid(axis='y', alpha=.25); a.legend(fontsize=9)

fig.suptitle('HindLeg torque-penalty A/B at fixed commands (512 envs x 300 steps, seed 0)', fontweight='bold')
fig.tight_layout()
fig.savefig(D + 'plots/cmd_ab.png', dpi=130)
print('saved', D + 'plots/cmd_ab.png')

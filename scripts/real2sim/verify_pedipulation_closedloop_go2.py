# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""배포 런타임으로 IsaacLab pedipulation env 를 **직접 폐루프 구동**한다 (UDP/ROS 없음).

r2s 통합 구동에서 발이 목표를 크게 벗어나는데, 정적 대조(obs 5.6e-7 / 목표 1.9e-6)는
깨끗했다. 남은 차이는 전송 경로(자유구동 sim + 다중 홉 지연)뿐이므로, 그 경로만 걷어낸
동일 코드로 같은 명령을 돌려 어느 쪽에 원인이 있는지 가른다.

명령 대본은 GUI 데모와 같다: FL dz=0.15 → dx=0.12 추가 → 조작 다리 RR 로 교체.

배포 경로를 바꾼 뒤에는 이걸 먼저 돌릴 것 — 정적 대조(obs·관절목표·FK)가 전부 통과해도
폐루프에서만 무너지는 결함이 실재한다. 근거·측정은
``reports/rsl_rl/go2_pedipulation/_comparisons/r2s_deploy_tracking_gap/``.
"""

import argparse
import os
import statistics
import sys

from isaaclab.app import AppLauncher

_REAL2SIM_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_REAL2SIM_DIR))
_DEFAULT_POLICY = os.path.join(
    _REPO_ROOT,
    "logs/rsl_rl/go2_pedipulation/2026-08-03_15-53-29_hipscale_scratch_s10x/exported/pedipulation_policy.pt",
)

parser = argparse.ArgumentParser()
parser.add_argument("--phase_steps", type=int, default=300, help="구간당 스텝 수 (50 Hz).")
parser.add_argument("--settle_steps", type=int, default=100, help="구간 끝에서 정착 판정에 쓸 스텝 수.")
parser.add_argument("--policy", type=str, default=_DEFAULT_POLICY, help="export 된 pedipulation jit 정책 경로.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = False

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(_REAL2SIM_DIR, "r2s_go2"))
import pedipulation_runtime as pr  # noqa: E402

cfg = parse_env_cfg("Go2-Pedipulation-v0", device="cuda:0", num_envs=1)
cfg.domain_rand = False  # r2s sim 과 같은 조건 (노이즈·외란 없음)
cfg.scene.num_envs = 1
cfg.episode_length_s = 1.0e6  # 구간 도중 타임아웃 리셋 금지
cfg.command.resample_time_min = 1.0e6
cfg.command.resample_time_max = 1.0e6
cfg.command.curriculum_min_episodes = 10**9
cfg.command.box_init = cfg.command.box_max
env = gym.make("Go2-Pedipulation-v0", cfg=cfg).unwrapped
obs_dict, _ = env.reset()
robot = env._robot

model = pr.PedipulationModel(args.policy)
state = pr.PedipulationState()

print(f"[cfg] box_max={cfg.command.box_max} world_anchor={cfg.command.world_anchor}", flush=True)


def measured():
    q = robot.data.joint_pos[0].tolist()
    dq = robot.data.joint_vel[0].tolist()
    _q = robot.data.root_link_quat_w[0].tolist()  # 6.0 은 xyzw
    return q, dq, [_q[3], _q[0], _q[1], _q[2]]


# ── nominal latch (런타임과 동일한 절차) ──────────────────────────────────
q, dq, quat = measured()
assert state.try_latch_nominal(q), "nominal latch 실패 — 기립 자세가 아님"
env_nom = env._nominal_foot_pos_b[0].tolist()
print("[latch] 런타임 vs env nominal:", flush=True)
for nm, f, e in zip(pr.LEG_NAMES, state.nominal_foot_pos_b, env_nom):
    d = max(abs(f[k] - e[k]) for k in range(3))
    print(
        f"  {nm}: rt=({f[0]:+.4f},{f[1]:+.4f},{f[2]:+.4f}) env=({e[0]:+.4f},{e[1]:+.4f},{e[2]:+.4f}) Δmax={d * 1000:.2f}mm"
    )

PHASES = [
    ("FL dz=0.15", 0, (0.0, 0.0, 0.15)),
    ("FL dx=0.12 추가", 0, (0.12, 0.0, 0.15)),
    ("RR 로 교체", 3, (0.12, 0.0, 0.15)),
]

for label, manip_leg, offset in PHASES:
    leg_role = pr.leg_role_from_manip(manip_leg)
    target_b = pr.foot_target_b(state.nominal_foot_pos_b, manip_leg, offset)
    # env 쪽 역할·목표도 같은 값으로 고정한다 — env 의 액션 매핑이 `_leg_role` 를 쓰고,
    # 명령 재샘플이 돌면 대본이 깨진다.
    env._leg_role[:] = torch.tensor([leg_role], device=env.device)
    env._foot_target_b[:] = torch.tensor([target_b], device=env.device)
    # ⚠ world_anchor 로 학습한 정책에 이 스크립트를 재사용하려면 `_resample_command` 를
    #   그냥 죽이면 안 된다 — 게이트의 `patch_command()` 처럼 `_arm_world_anchor(ids)` 를
    #   함께 불러야 이전 에피소드의 world 목표가 남지 않는다. 이 체크포인트는
    #   world_anchor=False 라 여기서는 문제되지 않는다(위 [cfg] 출력으로 확인).
    env._resample_command = lambda env_ids: None

    errs = []
    taus = []
    for step in range(args.phase_steps):
        q, dq, quat = measured()
        obs = pr.build_obs(quat, q, dq, leg_role, target_b, state)
        clipped = pr.clip_action(model.infer(obs))
        pr.action_to_target_art(clipped, leg_role, state)
        state.prev_actions = clipped
        obs_dict, _, term, trunc, _ = env.step(torch.tensor([clipped], device=env.device))
        if bool(term[0]) or bool(trunc[0]):
            print(f"  ⚠ {label}: step {step} 에서 에피소드 종료 — 대본 중단", flush=True)
            break
        foot = pr.foot_pos_b(robot.data.joint_pos[0].tolist())[manip_leg]
        tgt = target_b[manip_leg]
        errs.append(sum((foot[k] - tgt[k]) ** 2 for k in range(3)) ** 0.5)
        taus.append(float(robot.data.applied_torque[0].abs().max().item()))

    tail = errs[-args.settle_steps :]
    foot = pr.foot_pos_b(robot.data.joint_pos[0].tolist())[manip_leg]
    tgt = target_b[manip_leg]
    print(
        f"[{label}] 정착오차 중앙={statistics.median(tail) * 1000:6.1f}mm "
        f"최종={errs[-1] * 1000:6.1f}mm | max|τ|={max(taus):5.2f} N·m\n"
        f"    목표=({tgt[0]:+.3f},{tgt[1]:+.3f},{tgt[2]:+.3f}) 발=({foot[0]:+.3f},{foot[1]:+.3f},{foot[2]:+.3f})",
        flush=True,
    )

env.close()
simulation_app.close()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""hip 을 각도별로 잡아 두고 **버티는 데 드는 토크**를 잰다 — 좌우 4.17 배 차의 판별 (2026-09-02).

왜
--
08-28 실기 캡처(공중 고정)에서 두 hip 이 거울 자세인데 유지 토크가 4.17 배 달랐다::

    HL_hip  q +0.164 에서 |tau| mean 6.245 N·m   (목표 +0.234 에 도달 못 함)
    HR_hip  q -0.234 에서 |tau| mean 1.497 N·m   (목표 도달, 그 자세 tau 0.003)

기구가 좌우 대칭이면 이럴 수 없다. **어느 쪽이 이상한지**를 sim 이 가른다 — sim URDF 는 정의상
좌우 대칭이므로, sim 이 그 자세에서 요구하는 토크가 곧 "정상값"이다:

    sim 이 ~7 N·m 를 요구한다  ⇒ HL 이 정상이고 **HR 이 이상하다**(부하가 빠져 있다)
    sim 이 ~1.5 N·m 를 요구한다 ⇒ HR 이 정상이고 **HL 이 이상하다**(마찰·간섭·영점)

방법
----
sysid env(`fix_base=True`, 실기 공중 고정과 같은 전제)에서 hip 목표를 각도별로 홀드하고
정착 후 `applied_torque` 를 읽는다. 나머지 관절은 0. 게인은 배포값(`rga.py` HIND_LEG_CFG).

⚠ 이 프로브는 **중력+게인이 요구하는 토크**를 잰다. 실기의 추가 마찰은 여기 없다 —
   그 차이가 곧 답이다.

실행::

    ./isaaclab.sh -p _workspace/hindleg_hip_gravity_probe.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="HindLeg hip 유지토크 곡선.")
parser.add_argument("--task", type=str, default="Isaac-R2S-BipedLeg-Sysid-v0")
parser.add_argument("--hold_s", type=float, default=1.5, help="각 각도에서 정착까지 홀드 [s].")
parser.add_argument("--meas_s", type=float, default=0.5, help="홀드 끝의 이 구간만 평균낸다 [s].")
parser.add_argument("--n_angle", type=int, default=13, help="hip 각도 격자 수 (-0.234 ~ +0.234).")
# ⚠ 이 env 의 cfg 모듈은 물리 백엔드를 최상단에서 import 한다 — `SimulationApp` **생성 전에**
#   cfg 를 해석하면 Kit 이 재적재하며 죽는다(이번에 실제로 abort 났다). 그래서 고전 패턴을 쓴다:
#   AppLauncher 를 먼저 띄우고, 그 다음에 isaaclab_tasks 를 import 한다.
#   같은 함정: reports/.../pace_bipedleg_foot_coupling_probe README §검증 함정
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = False
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# 배포 게인 (`rga.py` HIND_LEG_CFG, 관절 공간). hip 은 gear 1.0 이라 채널값과 같다.
KP = {"hip": 100.0, "thigh": 50.0, "calf": 112.5, "foot": 28.8}
KD = {"hip": 5.0, "thigh": 5.0, "calf": 11.25, "foot": 7.2}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    if True:
        env = gym.make(args_cli.task, cfg=env_cfg)
        u = env.unwrapped
        robot = u.robot if hasattr(u, "robot") else u._robot

        def t(x):
            return x.torch if hasattr(x, "torch") else x

        names = [n.replace("_joint", "") for n in robot.joint_names]
        hl = names.index("HL_hip")
        hr = names.index("HR_hip")
        soft = t(robot.data.soft_joint_pos_limits).clone()
        lim = float(soft[0, hl, 1])
        nj = robot.num_joints

        def kind(n: str) -> str:
            return n.split("_")[1]

        act = robot.actuators["legs"]
        st = t(act.stiffness)[0] if hasattr(act, "stiffness") else None
        print(f"[probe] env 게인(관절공간) stiffness hip {float(st[0]):.1f} calf {float(st[4]):.1f}"
              if st is not None else "[probe] 게인 조회 실패")
        dt = float(u.cfg.sim.dt) * int(u.cfg.decimation)
        n_hold = max(1, int(round(args_cli.hold_s / dt)))
        n_meas = max(1, int(round(args_cli.meas_s / dt)))
        zero = torch.zeros(1, nj, device=u.device)

        print(f"[probe] fix_base={getattr(u.cfg, 'fix_base', None)}  제어 {dt * 1000:.1f} ms"
              f"  홀드 {n_hold} 스텝 · 측정 마지막 {n_meas}")
        print(f"[probe] hip soft limit ±{lim:.4f} rad · kp_hip {KP['hip']} · kd_hip {KD['hip']}")
        print("\n★ 규약: HL 과 HR 은 **거울**이라 같은 물리 자세가 부호 반대다.")
        print("        아래 표의 'HR q' 는 −q_HL 로 잡았다 (실기 캡처의 부호 관계와 동일).\n")
        print(f"{'q_HL [rad]':>11} {'(deg)':>7} | {'HL tau':>9} {'HR tau':>9} | {'대칭비 HL/HR':>13}"
              f" | {'HL 오차':>8} {'HR 오차':>8}")

        env.reset()
        angles = torch.linspace(-lim, lim, args_cli.n_angle)
        rows = []
        with torch.inference_mode():
            for a in angles:
                q_des = zero.clone()
                q_des[0, hl] = float(a)
                q_des[0, hr] = -float(a)  # 거울
                # ★ sysid 모드에서는 **action 자체가 절대 관절 목표각**(articulation 순서)이다
                #   (`r2s_biped_leg_env.py:334-338`). `set_setpoint` 는 live 모드용이라 여기선
                #   무시된다 — 처음에 그걸 써서 hip 이 전혀 안 움직였다(추종오차가 목표와 같았다).
                tau_acc, q_acc = [], []
                for k in range(n_hold):
                    env.step(q_des)
                    if k >= n_hold - n_meas:
                        tau_acc.append(t(robot.data.applied_torque)[0].clone())
                        q_acc.append(t(robot.data.joint_pos)[0].clone())
                TAU = torch.stack(tau_acc).mean(0)
                Q = torch.stack(q_acc).mean(0)
                thl, thr = abs(float(TAU[hl])), abs(float(TAU[hr]))
                err_hl = float(a) - float(Q[hl])
                err_hr = -float(a) - float(Q[hr])
                ratio = thl / thr if thr > 1e-6 else float("inf")
                rows.append((float(a), thl, thr, ratio))
                print(f"{float(a):11.4f} {float(a) * 57.2958:7.2f} | {thl:9.3f} {thr:9.3f} | {ratio:13.3f}"
                      f" | {err_hl:+8.4f} {err_hr:+8.4f}")

        print(f"\n{'=' * 78}")
        print("실기 대조 — 08-28 캡처의 두 동작점에서 sim 이 요구하는 유지 토크:")
        for target_q, label, real_tau in ((0.164, "HL_hip q +0.164", 6.245), (-0.234, "HR_hip q -0.234", 1.497)):
            near = min(rows, key=lambda r: abs(r[0] - abs(target_q)))
            sim_tau = near[1] if target_q > 0 else near[2]
            print(f"  {label:18s}  sim {sim_tau:6.3f} N·m   실기 {real_tau:6.3f} N·m"
                  f"   실기/sim {real_tau / max(sim_tau, 1e-6):6.2f}배")
        print(f"{'=' * 78}")
        env.close()


main()
simulation_app.close()

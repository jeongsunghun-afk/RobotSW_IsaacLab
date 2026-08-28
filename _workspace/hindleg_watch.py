# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""`termfix_*` 두 arm 을 지속 감시하다가 **보고할 일이 생기면 종료**한다. Isaac 불필요.

호출자(에이전트)는 이 스크립트를 background 로 띄워 두고, 종료되면 그때 보고하면 된다.
종료 사유는 stdout 마지막 줄 ``EVENT: <사유>`` 로 나온다.

종료 조건
---------
``milestone``  두 arm 이 모두 다음 마일스톤 iter 에 도달        → 정기 보고
``crashed``    학습 프로세스가 사라졌다                          → 즉시 보고
``stalled``    ``--stall_min`` 분 동안 iter 가 안 늘었다         → 즉시 보고
``spike``      value loss 가 구간 max 대비 ``--spike`` 배 급등   → 즉시 보고
``collapse``   reward/step 구간 max 가 running max 대비 급락     → 즉시 보고
``done``       두 arm 이 모두 max_iterations 도달                → 종료 보고

★ 스냅샷 폴링이 아니라 **구간 스캔**이다 — 틱 사이 스파이크를 놓치지 않도록 직전 확인 이후
  구간 전체를 훑고, 악화 판정은 구간 min 이 아니라 **구간 max 대 running max** 로 한다
  (project_train_monitor_interval_scan).
★ warmup(기본 200 iter) 은 running max 에서 제외한다 — critic 초기 value loss 를 넣으면
  이후 어떤 악화도 못 잡는다.

실행::

    python _workspace/hindleg_watch.py --milestones 6000 12000 25000 50000
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import time
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator

RUNS = Path(__file__).resolve().parents[1] / "logs" / "rsl_rl" / "hindLeg_history_direct"
# 기본 감시 대상. `--arms` 로 바꾼다 — arm 이름이 바뀌었는데 여기가 그대로면 감시기는
# **죽은 구 run 을 보고 "프로세스 소멸"을 외친다**(실제로 한 번 그랬다).
ARMS: dict[str, str] = {}  # main() 에서 --arms 로 채운다
DEFAULT_ARMS = ("termfix_reflIcap", "termfix_only_transpose")
WARMUP = 200


def read(pattern: str):
    hits = sorted(glob.glob(str(RUNS / pattern)))
    if not hits:
        return None
    ev = sorted(glob.glob(hits[-1] + "/events.out.tfevents*"))
    if not ev:
        return None
    ea = event_accumulator.EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    have = ea.Tags()["scalars"]
    out = {}
    for t in ("Train/mean_reward", "Train/mean_episode_length", "Loss/value"):
        if t in have:
            s = ea.Scalars(t)
            out[t] = (np.array([e.step for e in s]), np.array([e.value for e in s]))
    return out or None


def alive(run_name: str) -> bool:
    """해당 run_name 으로 돌아가는 train.py 프로세스가 있는가."""
    r = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True)
    return any("train.py" in ln and f"--run_name {run_name}" in ln for ln in r.stdout.splitlines())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestones", type=int, nargs="+", default=[6000, 12000, 25000, 50000])
    ap.add_argument("--interval", type=int, default=180, help="확인 주기 [s].")
    ap.add_argument("--stall_min", type=float, default=25.0, help="iter 정체 판정 [min].")
    ap.add_argument("--spike", type=float, default=4.0, help="value loss 급등 배수.")
    ap.add_argument("--drop", type=float, default=0.35, help="reward/step 급락 비율 (running max 대비).")
    ap.add_argument(
        "--arms", type=str, nargs="+", default=list(DEFAULT_ARMS),
        help="감시할 `--run_name` 들. ★arm 을 새로 띄웠으면 **반드시 같이 갱신**한다 — "
        "기본값 그대로 두면 이미 끝난 구 run 을 보고 'crashed' 를 보고한다.",
    )
    args = ap.parse_args()
    global ARMS
    ARMS = {a: f"*_{a}" for a in args.arms}

    pending = sorted(args.milestones)
    last_iter: dict[str, int] = {}
    last_move = time.time()
    run_max_r: dict[str, float] = {}
    run_max_v: dict[str, float] = {}
    seen_upto: dict[str, int] = {a: 0 for a in ARMS}

    def emit(reason: str, detail: str) -> None:
        print(f"\n{detail}\nEVENT: {reason}", flush=True)

    while True:
        cur = {}
        for name, pat in ARMS.items():
            d = read(pat)
            if d is None or "Train/mean_reward" not in d:
                cur[name] = None
                continue
            st, rv = d["Train/mean_reward"]
            _, el = d["Train/mean_episode_length"]
            n = min(len(rv), len(el))
            ps = rv[:n] / np.maximum(el[:n], 1e-9)
            cur[name] = {"step": st[:n], "ps": ps, "val": d.get("Loss/value")}

        lines = []
        for name, c in cur.items():
            if c is None:
                lines.append(f"  {name:18s} (데이터 없음)")
                continue
            it = int(c["step"].max())
            lines.append(f"  {name:18s} iter {it:6d}  reward/step {c['ps'][-1]:8.4f}")
            if last_iter.get(name) != it:
                last_move = time.time()
            last_iter[name] = it

        # --- 크래시 ---
        dead = [n for n in ARMS if not alive(n)]
        if dead:
            emit("crashed", f"프로세스 소멸: {dead}\n" + "\n".join(lines))
            return

        # --- 정체 ---
        if (time.time() - last_move) / 60.0 > args.stall_min:
            emit("stalled", f"{args.stall_min:.0f} 분간 iter 증가 없음\n" + "\n".join(lines))
            return

        # --- 구간 스캔: 스파이크 / 급락 ---
        for name, c in cur.items():
            if c is None:
                continue
            new = c["step"] > seen_upto[name]
            post_warm = c["step"] >= WARMUP
            if new.any() and post_warm.any():
                seg_r_max = float(c["ps"][new & post_warm].max()) if (new & post_warm).any() else None
                prev_max = run_max_r.get(name)
                if seg_r_max is not None:
                    if prev_max is not None and seg_r_max < prev_max * (1 - args.drop) and prev_max > 0.01:
                        emit("collapse", f"{name}: reward/step 구간max {seg_r_max:.4f}"
                                         f" < running max {prev_max:.4f} × {1 - args.drop:.2f}\n" + "\n".join(lines))
                        return
                    run_max_r[name] = max(prev_max or -1e9, seg_r_max)
                v = c["val"]
                if v is not None:
                    vs, vv = v
                    vm = (vs > seen_upto[name]) & (vs >= WARMUP)
                    if vm.any():
                        seg_v = float(vv[vm].max())
                        pv = run_max_v.get(name)
                        if pv is not None and seg_v > pv * args.spike:
                            emit("spike", f"{name}: value loss 구간max {seg_v:.4f}"
                                          f" > running max {pv:.4f} × {args.spike:.1f}\n" + "\n".join(lines))
                            return
                        run_max_v[name] = max(pv or 0.0, seg_v)
            seen_upto[name] = int(c["step"].max())

        # --- 마일스톤 / 완료 ---
        its = [last_iter.get(n, 0) for n in ARMS]
        if pending and min(its) >= pending[0]:
            m = pending.pop(0)
            emit("milestone", f"마일스톤 iter {m} 도달\n" + "\n".join(lines))
            return
        if min(its) >= 49990:
            emit("done", "두 arm 모두 완주\n" + "\n".join(lines))
            return

        time.sleep(args.interval)


main()

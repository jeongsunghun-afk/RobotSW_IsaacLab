# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""`gait_survey_multienv.py --log_style --other_checkpoint` 산출물 → **명령 구간 × 걸음 × 판별기** 보상 표.

조건부 판별기의 존재 이유는 "고속 명령에서는 gallop 을, 저속에서는 pace 를 더 쳐준다"는 것인데,
그게 실제로 성립하는지는 잰 적이 없다. 여기서는 같은 롤아웃을 자기 D 와 상대 D 로 채점한 값을
env 별 걸음 라벨에 붙여 두 질문에 수치로 답한다.

  (i) cmd 2.5~3.5 에서 gallop env 가 pace env 보다 style 보상을 더 받는가
  (ii) cmd 0.5~1.5 에서 pace 가 gallop 보다 더 받는가

측정 단위는 **env** 다. 스텝 표본을 그대로 쓰면 같은 env 의 100 스텝이 독립 표본처럼 세어져
n 이 100 배로 부풀고 p 값이 전부 0 이 된다.

거르는 표본 (`gait_survey_analyze.py` 와 같은 규칙):
  리셋 걸침 · 낙상(base 높이 < termination_height) · 창 안 명령 변동
추가로 이 스크립트만의 규칙:
  ★ **명령 정착 시간**. 이 run 들은 4 s 마다 명령을 다시 뽑는다(`cmdchg4s`). 창 안에서 명령이
  안 바뀌었더라도 창 직전에 바뀌었다면 그 env 의 걸음은 **옛 명령**의 것이고 D 가 보는 조건 열은
  **새 명령**이다 — 재는 효과를 정확히 씻어내는 표본이라 `--settle_s` 미만은 버린다.

사용:
    ./isaaclab.sh -p _workspace/leg/gait_reward_by_class.py \
        condmlp50k_rsi=<dir>/gait_survey.npz ... --out_md <md> --out_png <png>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from gait_classify import LEGS, MIN_AMP_RAD, _circ_mean_frac, _classify, _inst_phase  # noqa: E402

FALL_H = 0.35  # termination_height [m] — 두 run 의 env.yaml 실측값
MIN_CYCLES = 3.0
CMD_BINS = [(0.5, 1.5), (1.5, 2.5), (2.5, 3.5)]


def _label_envs(d, win_s: float, settle_s: float):
    """env 별 (걸음 라벨, 유지 여부, 창 평균 명령/실속도, 창 슬라이스)."""
    jpos = np.asarray(d["jpos"], dtype=np.float32)
    vxc, yawc = np.asarray(d["vx_cmd"]), np.asarray(d["yaw_cmd"])
    hgt, elen = np.asarray(d["height"]), np.asarray(d["ep_len"])
    names = [str(s) for s in d["joint_names"]]
    dt = float(d["dt"])
    fs = 1.0 / dt
    T, N, _ = jpos.shape
    w = int(win_s * fs)
    sl = slice(T - w, T)

    idx = {lg: names.index(f"{lg}_thigh_joint") for lg in LEGS}
    reset = (np.diff(elen[sl], axis=0) < 0).any(axis=0)
    fell = (hgt[sl] < FALL_H).any(axis=0)
    cmd_moved = (vxc[sl].std(axis=0) > 1e-3) | (yawc[sl].std(axis=0) > 1e-3)

    # 창 시작 이전으로 거슬러 올라가 마지막 명령 변경 시점을 찾는다.
    chg = (np.abs(np.diff(vxc, axis=0)) > 1e-3) | (np.abs(np.diff(yawc, axis=0)) > 1e-3)  # [T-1, N]
    last_chg = np.full(N, -1, dtype=np.int64)
    for t in range(chg.shape[0]):
        last_chg[chg[t]] = t + 1
    settle = (T - w - last_chg) * dt  # 창 시작 시점까지의 정착 시간 [s]
    unsettled = settle < settle_s

    keep = ~(reset | fell | cmd_moved | unsettled)
    labels = np.array(["-"] * N, dtype=object)
    for e in np.flatnonzero(keep):
        sig = {lg: jpos[sl, e, idx[lg]] for lg in LEGS}
        if max(float(np.ptp(s)) for s in sig.values()) < MIN_AMP_RAD:
            labels[e] = "stand"
            continue
        ph = {lg: _inst_phase(sig[lg], fs) for lg in LEGS}
        freq = float(np.polyfit(np.arange(w) / fs, ph["FL"], 1)[0]) / (2 * np.pi)
        if abs(freq) * win_s < MIN_CYCLES:
            continue
        phi = {lg: _circ_mean_frac(ph[lg] - ph["FL"]) for lg in LEGS}
        labels[e] = _classify(phi)

    stats = dict(
        n=N, reset=int(reset.sum()), fell=int(fell.sum()), cmd_moved=int(cmd_moved.sum()),
        unsettled=int(unsettled.sum()), keep=int(keep.sum()),
        settle_median=float(np.median(settle)),
    )
    vx_cmd = vxc[T - 1]
    vx_act = np.asarray(d["vx_act"])[sl].mean(axis=0)
    return labels, keep, vx_cmd, vx_act, sl, stats


def _welch(a: np.ndarray, b: np.ndarray):
    """Welch t 와 Cohen's d (pooled sd). scipy 없이 — 자유도는 Welch-Satterthwaite."""
    from math import sqrt

    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan"), float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = sqrt(va / na + vb / nb)
    t = (ma - mb) / se if se > 0 else float("nan")
    dof = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)) if se > 0 else float("nan")
    sp = sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    dd = (ma - mb) / sp if sp > 0 else float("nan")
    try:
        from scipy import stats as _st

        p = float(2 * _st.t.sf(abs(t), dof))
    except Exception:  # noqa: BLE001
        p = float("nan")
    return float(t), float(dd), p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+", help="라벨=npz경로")
    ap.add_argument("--win_s", type=float, default=2.0)
    ap.add_argument("--settle_s", type=float, default=1.5, help="창 시작 전 명령이 유지돼야 하는 시간 [s]")
    ap.add_argument("--min_n", type=int, default=10, help="이보다 적은 칸은 d/t 를 내지 않는다")
    ap.add_argument("--out_md", type=str, default=None)
    ap.add_argument("--out_png", type=str, default=None)
    ap.add_argument("--out_json", type=str, default=None)
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s=""):
        print(s)
        lines.append(s)

    results = {}
    for spec in args.specs:
        label, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        labels, keep, vx_cmd, vx_act, sl, stats = _label_envs(d, args.win_s, args.settle_s)
        own = str(d["disc_arch"]); other = str(d["other_disc_arch"])
        cols = {}
        for key, arch in (("style", own), ("style_other", other)):
            arr = np.asarray(d[key])
            if arr.size == 0:
                continue
            cols[f"{arch}{'(자기)' if key == 'style' else '(상대)'}"] = arr[sl].mean(axis=0)
        results[label] = dict(labels=labels, keep=keep, vx_cmd=vx_cmd, vx_act=vx_act, cols=cols,
                              stats=stats, ckpt=str(d["checkpoint"]), other_ckpt=str(d["other_checkpoint"]),
                              drail_reps=int(d["drail_reps"]))

    emit("## 실행 조건")
    emit()
    emit(f"창 {args.win_s} s · 명령 정착 {args.settle_s} s 이상 · 낙상 기준 base_h < {FALL_H} m · "
         f"보상 단위 = env 당 창 평균")
    emit()
    emit("| 표본 | 체크포인트 | 상대 | n | 리셋 | 낙상 | 명령변동 | 미정착 | 사용 |")
    emit("|---|---|---|---|---|---|---|---|---|")
    for lb, r in results.items():
        s = r["stats"]
        emit(f"| {lb} | {pathlib.Path(r['ckpt']).parent.name[:34]}/{pathlib.Path(r['ckpt']).name} | "
             f"{pathlib.Path(r['other_ckpt']).name} | {s['n']} | {s['reset']} | {s['fell']} | "
             f"{s['cmd_moved']} | {s['unsettled']} | {s['keep']} |")
    emit()

    emit("## 명령 구간 × 걸음 × 판별기 — style 보상 (평균 ± 표준편차, n=env)")
    emit()
    gaits = ["gallop", "trot", "pace", "other", "stand"]
    for lb, r in results.items():
        emit(f"### {lb}")
        emit()
        hdr = "| cmd 구간 | 판별기 | " + " | ".join(gaits) + " |"
        emit(hdr)
        emit("|---" * (len(gaits) + 2) + "|")
        for lo, hi in CMD_BINS:
            m = r["keep"] & (r["vx_cmd"] >= lo) & (r["vx_cmd"] < hi)
            for cname, cvals in r["cols"].items():
                row = f"| [{lo},{hi}) | {cname} |"
                for g in gaits:
                    gm = m & (r["labels"] == g) if g != "gallop" else m & ((r["labels"] == "gallop") | (r["labels"] == "bound"))
                    v = cvals[gm]
                    row += f" {v.mean():.3f}±{v.std(ddof=1):.3f} (n={len(v)}) |" if len(v) >= 2 else f" n={len(v)} |"
                emit(row)
        emit()

    emit("## 핵심 질문")
    emit()
    emit("(i) cmd 2.5~3.5 에서 gallop 이 pace 보다 더 받는가 · (ii) cmd 0.5~1.5 에서 pace 가 gallop 보다 더 받는가")
    emit()
    emit("| 질문 | 표본 | 판별기 | A 평균 (n) | B 평균 (n) | 차 | Cohen d | Welch t | p |")
    emit("|---|---|---|---|---|---|---|---|---|")
    qjson = []
    for qi, (lo, hi, ga, gb) in enumerate(
        # (iii) 은 보조 질문이다 — 저속에서 gallop 표본이 아예 없으면 (ii) 는 답이 안 나오므로,
        # 그 구간에 실제로 존재하는 두 걸음(pace vs trot)으로 "조건에 맞는 걸음을 쳐주는가" 를 대신 본다.
        [(2.5, 3.5, "gallop", "pace"), (0.5, 1.5, "pace", "gallop"), (0.5, 1.5, "pace", "trot")], start=1
    ):
        for lb, r in results.items():
            m = r["keep"] & (r["vx_cmd"] >= lo) & (r["vx_cmd"] < hi)
            for cname, cvals in r["cols"].items():
                def sel(g):
                    gm = m & ((r["labels"] == "gallop") | (r["labels"] == "bound")) if g == "gallop" else m & (r["labels"] == g)
                    return cvals[gm]

                a, b = sel(ga), sel(gb)
                if len(a) < args.min_n or len(b) < args.min_n:
                    emit(f"| ({'i' * qi}) {ga}>{gb} @[{lo},{hi}) | {lb} | {cname} | "
                         f"{a.mean():.3f} (n={len(a)}) | {b.mean():.3f} (n={len(b)}) | — | 판정불가 (n<{args.min_n}) | — | — |"
                         if len(a) and len(b) else
                         f"| ({'i' * qi}) {ga}>{gb} @[{lo},{hi}) | {lb} | {cname} | n={len(a)} | n={len(b)} | — | 판정불가 | — | — |")
                    continue
                t, dd, p = _welch(a, b)
                emit(f"| ({'i' * qi}) {ga}>{gb} @[{lo},{hi}) | {lb} | {cname} | {a.mean():.3f} (n={len(a)}) | "
                     f"{b.mean():.3f} (n={len(b)}) | {a.mean() - b.mean():+.3f} | {dd:+.2f} | {t:+.2f} | {p:.3g} |")
                qjson.append(dict(q=qi, sample=lb, disc=cname, a=ga, b=gb, cmd=[lo, hi],
                                  a_mean=float(a.mean()), b_mean=float(b.mean()), n_a=len(a), n_b=len(b),
                                  d=dd, t=t, p=p))
    emit()

    emit("## 같은 걸음 안에서 실제 속도 vs 보상 상관 (Pearson r, n=env)")
    emit()
    emit("D 가 조건 라벨과 운동학의 일치를 보는지 — 같은 걸음·같은 명령 구간에서 실제로 빠른 env 가 더 받는가")
    emit()
    emit("★ 구간 안에서도 보상은 명령에 기울기가 있다(예: mlp 자기 D 의 gallop 0.794@[1.5,2.5) → 0.748@[2.5,3.5)).")
    emit("그래서 raw r 은 '명령이 높은 env 가 더 빠르고 덜 받는다' 는 기울기만으로도 음수가 된다. `vx_cmd` 를 선형")
    emit("제거한 편상관 r(부분) 과 추종 잔차 `vx_act − vx_cmd` 상관을 같이 낸다 — 질문에 답하는 것은 뒤의 둘이다.")
    emit()
    emit("| 표본 | 판별기 | cmd 구간 | 걸음 | n | r(raw) | r(부분, vx_cmd 제거) | r(잔차 vx_act−vx_cmd) |")
    emit("|---|---|---|---|---|---|---|---|")
    for lb, r in results.items():
        for cname, cvals in r["cols"].items():
            for lo, hi in CMD_BINS:
                m = r["keep"] & (r["vx_cmd"] >= lo) & (r["vx_cmd"] < hi)
                for g in ("gallop", "trot", "pace"):
                    gm = m & ((r["labels"] == "gallop") | (r["labels"] == "bound")) if g == "gallop" else m & (r["labels"] == g)
                    n = int(gm.sum())
                    if n < args.min_n:
                        continue
                    va, rw, vc = r["vx_act"][gm], cvals[gm], r["vx_cmd"][gm]
                    rr = float(np.corrcoef(va, rw)[0, 1])
                    if vc.std() > 1e-6:
                        # vx_cmd 를 선형 제거한 뒤의 상관 = 편상관
                        ra = va - np.polyval(np.polyfit(vc, va, 1), vc)
                        rb = rw - np.polyval(np.polyfit(vc, rw, 1), vc)
                        rp = float(np.corrcoef(ra, rb)[0, 1]) if ra.std() > 1e-9 and rb.std() > 1e-9 else float("nan")
                    else:
                        rp = float("nan")
                    res = va - vc
                    rres = float(np.corrcoef(res, rw)[0, 1]) if res.std() > 1e-9 else float("nan")
                    emit(f"| {lb} | {cname} | [{lo},{hi}) | {g} | {n} | {rr:+.3f} | {rp:+.3f} | {rres:+.3f} |")
    emit()

    emit("## 걸음 분포 (사용 표본)")
    emit()
    emit("| 표본 | cmd 구간 | " + " | ".join(gaits) + " | n |")
    emit("|---" * (len(gaits) + 3) + "|")
    for lb, r in results.items():
        for lo, hi in CMD_BINS:
            m = r["keep"] & (r["vx_cmd"] >= lo) & (r["vx_cmd"] < hi)
            labs = [x for x in r["labels"][m] if x != "-"]
            if not labs:
                continue
            c, n = Counter(labs), len(labs)
            row = f"| {lb} | [{lo},{hi}) |"
            for g in gaits:
                v = c["gallop"] + c["bound"] if g == "gallop" else c[g]
                row += f" {100 * v / n:.1f}% |"
            emit(row + f" {n} |")
    emit()

    if args.out_md:
        pathlib.Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.out_md).write_text("\n".join(lines) + "\n")
        print(f">>> md: {args.out_md}")
    if args.out_json:
        pathlib.Path(args.out_json).write_text(json.dumps(qjson, indent=2))

    if args.out_png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        panels = [(lb, cn) for lb, r in results.items() for cn in r["cols"]]
        ncol = 2
        nrow = int(np.ceil(len(panels) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.6 * nrow), squeeze=False)
        pg = ["gallop", "trot", "pace", "stand"]
        colors = {"gallop": "#d1495b", "trot": "#30638e", "pace": "#edae49", "stand": "#8d99ae"}
        for ax, (lb, cn) in zip(axes.ravel(), panels):
            r = results[lb]
            cvals = r["cols"][cn]
            xs = np.arange(len(CMD_BINS))
            wbar = 0.8 / len(pg)
            for k, g in enumerate(pg):
                mus, ses = [], []
                for lo, hi in CMD_BINS:
                    m = r["keep"] & (r["vx_cmd"] >= lo) & (r["vx_cmd"] < hi)
                    gm = m & ((r["labels"] == "gallop") | (r["labels"] == "bound")) if g == "gallop" else m & (r["labels"] == g)
                    v = cvals[gm]
                    mus.append(v.mean() if len(v) >= 2 else np.nan)
                    ses.append(v.std(ddof=1) / np.sqrt(len(v)) if len(v) >= 2 else np.nan)
                ax.bar(xs + k * wbar - 0.4 + wbar / 2, mus, wbar, yerr=ses, capsize=2, label=g, color=colors[g])
            ax.set_xticks(xs)
            ax.set_xticklabels([f"[{lo},{hi})" for lo, hi in CMD_BINS])
            # ★ matplotlib 기본 폰트에 CJK 가 없어 한글이 □ 로 깨진다 — 그림 안 문자열은 ASCII 로.
            ax.set_title(f"policy={lb}  disc={cn.replace('(자기)', ' [own]').replace('(상대)', ' [other]')}", fontsize=10)
            ax.set_xlabel("vx command [m/s]")
            ax.set_ylabel("style reward (env-mean)")
            ax.legend(fontsize=7)
            ax.grid(axis="y", alpha=0.3)
        for ax in axes.ravel()[len(panels):]:
            ax.axis("off")
        fig.suptitle("style reward by command bin x gait  (bars=mean, error=SEM; compare within a panel only)", fontsize=11)
        fig.tight_layout()
        pathlib.Path(args.out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out_png, dpi=130)
        print(f">>> png: {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

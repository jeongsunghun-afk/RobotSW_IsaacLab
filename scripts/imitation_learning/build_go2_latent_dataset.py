# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 latent-imitation 용 expert 모션 데이터셋 빌더 (Phase 0).

리타게팅 산출물(``smr_txt_retarget_dataset_go2_v0``, 61열 DeepMimic JSON)에서
보행 관련 클립을 선별해 ``frames(N, 18)`` PKL 로 변환하고, 세션 단위 통계와
train/val split 을 함께 낸다.

61열 txt 레이아웃::

    [0:3]   root_pos
    [3:7]   root_quat (qx, qy, qz, qw)
    [7:19]  joint_dof (12)
    [19:31] toes_local
    [31:34] lin_vel_local  (body frame)
    [34:37] ang_vel_local  (body frame)
    [37:49] joint_vel
    [49:61] toe_vel_local

PKL 은 기존 ``motion_lib`` 계약을 따른다 — ``[3:6]`` 은 **exponential map** 이다.
속도는 저장하지 않고 ``motion_lib`` 이 재계산하지만, 그 결과가 txt 의 ``[31:37]`` 과
소스 자체의 정합성 한계까지 일치함이 검증되어 있다
(``reports/go2_imitation/go2_imitation_tracking/2026-08-27_motion_lib_rotation_fix/``).

선별 기준은 ``--help`` 참고. 커버리지·세션 통계는 ``--report_dir`` 로 저장한다.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import pickle
import re
from collections import defaultdict

import numpy as np

# ── 카테고리 ─────────────────────────────────────────────────────────────
#: 핵심 보행. 정상 상태 보행이 클립 전체를 차지한다.
CORE_CATEGORIES = ["walk", "run", "trot"]

#: 보행이 포함된 전이 동작. 논문 "Walk Like Dogs" 가 말하는 mode transition 학습용.
TRANSITION_CATEGORIES = [
    "stand_walk",
    "sit_walk",
    "run_walk",
    "walk_run",
    "lie_walk",
    "sit_trot",
    "stand_trot",
    "lie_trot",
    "trot_sit",
    "trot_lie",
    "sit_run",
    "run_sit",
]

#: 제외. ``spin`` 계열은 제자리 선회가 |wz| 5~6 rad/s 라 로봇 추종 범위 밖이고,
#: 클립의 봉투 내부 비율 중앙값이 0.01~0.13 이라 부분 트리밍으로도 살릴 수 없다.
#: ``jump``/정적 자세(``stand``/``sit``/``lie``)는 보행과 이질적이다.
EXCLUDED_NOTE = "spin, spin_sit, sit_spin, jump*, stand, sit, lie 및 그 전이"

#: ``[M_]D<n>_<seq>_<subj>_<take>_F<start>_F<end>_<idx>``.
#: ``seq`` 는 순수 숫자 외에 ``ex3``, ``047z`` 같은 변형이 있다.
NAME_PATTERN = re.compile(r"^(M_)?(D\d+_[A-Za-z0-9]+_[A-Za-z0-9]+_\d+)_F(\d+)_F(\d+)_(\d+)$")


def quat_to_exp_map(qxyzw: np.ndarray) -> np.ndarray:
    """단위 쿼터니언 (N, 4) [x, y, z, w] → exponential map (N, 3).

    ``w >= 0`` 을 강제해 최단호를 택한다(``quat_pos``).
    """
    q = np.asarray(qxyzw, dtype=np.float64).copy()
    q[q[:, 3] < 0] *= -1.0

    length = np.linalg.norm(q[:, :3], axis=-1)
    angle = 2.0 * np.arctan2(length, q[:, 3])
    axis = np.where(
        (length > 1e-5)[:, None],
        q[:, :3] / np.maximum(length, 1e-12)[:, None],
        np.array([0.0, 0.0, 1.0])[None, :],
    )
    return axis * angle[:, None]


def parse_name(path: str) -> tuple[bool, str, int, int] | None:
    """파일명 → (mirror, session, frame_start, frame_end). 실패 시 None."""
    m = NAME_PATTERN.match(os.path.splitext(os.path.basename(path))[0])
    if m is None:
        return None
    return bool(m.group(1)), m.group(2), int(m.group(3)), int(m.group(4))


def _md5(path: str) -> str:
    """파일 md5. 컨텍스트 매니저로 열어 핸들 누수를 막는다."""
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def main() -> None:  # noqa: C901 - 데이터셋 빌드 CLI (단계별 분기 나열, 분해 이득 없음)
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--src_dir", default="/home/lgb/Dog_Motion_data_3D/mujoco_retarget_go2/smr_txt_retarget_dataset_go2_v0"
    )
    p.add_argument("--dst_dir", required=True, help="PKL 출력 디렉터리")
    p.add_argument("--report_dir", default=None, help="통계 CSV/JSON 출력 디렉터리")
    p.add_argument("--vx_min", type=float, default=-0.5, help="명령 봉투 전진속도 하한 [m/s]")
    p.add_argument("--vx_max", type=float, default=4.0, help="명령 봉투 전진속도 상한 [m/s]")
    p.add_argument("--wz_max", type=float, default=2.0, help="명령 봉투 yaw rate 절대 상한 [rad/s]")
    p.add_argument("--min_inside", type=float, default=0.90, help="봉투 내부 프레임 비율 하한 (클립 단위 유지 기준)")
    p.add_argument("--min_frames", type=int, default=30, help="클립 최소 프레임 수")
    p.add_argument("--val_sessions", type=int, default=6, help="검증용으로 뗄 세션 수")
    p.add_argument(
        "--split_mode",
        choices=["random", "stratified"],
        default="stratified",
        help="stratified: 각 카테고리가 val 에서 속도 간격을 갖도록 세션을 고른다. "
        "random 은 walk 편중이라 run/trot 게이트를 돌릴 수 없다.",
    )
    p.add_argument(
        "--strat_categories", nargs="*", default=["trot", "run", "walk"], help="층화 대상 카테고리 (희소한 것부터)"
    )
    p.add_argument(
        "--strat_min_gap", type=float, default=0.4, help="카테고리별로 val 안에서 확보할 최소 평균속도 차 [m/s]"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry_run", action="store_true", help="PKL 을 쓰지 않고 통계만 낸다")
    args = p.parse_args()

    categories = CORE_CATEGORIES + TRANSITION_CATEGORIES
    print(f"[선별] 카테고리 {len(categories)}개: {', '.join(categories)}")
    print(f"[제외] {EXCLUDED_NOTE}")
    print(
        f"[봉투] vx ∈ [{args.vx_min}, {args.vx_max}] m/s,  |wz| <= {args.wz_max} rad/s,"
        f"  클립 유지 조건 내부비율 >= {args.min_inside:.0%}"
    )

    # ── 1) 스캔 + 파싱 ──────────────────────────────────────────────
    clips, unparsed = [], []
    for cat in categories:
        for f in sorted(glob.glob(os.path.join(args.src_dir, cat, "*.txt"))):
            info = parse_name(f)
            if info is None:
                unparsed.append(f)
                continue
            try:
                with open(f) as fh:
                    raw = json.load(fh)
                frames = np.array(raw["Frames"], dtype=np.float64)
            except Exception as exc:  # noqa: BLE001
                unparsed.append(f"{f} ({exc})")
                continue
            if frames.ndim != 2 or frames.shape[1] < 37:
                unparsed.append(f"{f} (열 {frames.shape})")
                continue
            mirror, session, f0, f1 = info
            clips.append(
                {
                    "path": f,
                    "cat": cat,
                    "mirror": mirror,
                    "session": session,
                    "f0": f0,
                    "f1": f1,
                    "raw": raw,
                    "frames": frames,
                    "md5": _md5(f),
                }
            )
    print(f"\n[1] 스캔: {len(clips)} 파싱 성공, {len(unparsed)} 실패")
    for u in unparsed[:5]:
        print(f"      실패: {os.path.basename(u)}")
    if unparsed:
        raise SystemExit("파일명 스키마 파싱 실패가 있다 — 조용히 넘기지 말 것.")

    # ── 2) md5 dedup ────────────────────────────────────────────────
    seen: set[str] = set()
    deduped = []
    for c in clips:
        if c["md5"] in seen:
            continue
        seen.add(c["md5"])
        deduped.append(c)
    print(f"[2] md5 dedup: {len(clips)} -> {len(deduped)} (중복 {len(clips) - len(deduped)})")

    # ── 3) 봉투 필터 (클립 단위 — 시계열 연속성 보존) ────────────────
    kept, dropped = [], []
    for c in deduped:
        fr = c["frames"]
        n = max(len(fr) - 1, 1)  # 마지막 프레임 속도는 복사값
        vx, wz = fr[:n, 31], fr[:n, 36]
        inside = (vx >= args.vx_min) & (vx <= args.vx_max) & (np.abs(wz) <= args.wz_max)
        c["inside"] = float(inside.mean())
        c["vx"], c["wz"] = vx, wz
        if len(fr) < args.min_frames:
            c["drop_reason"] = f"프레임 {len(fr)} < {args.min_frames}"
            dropped.append(c)
        elif c["inside"] < args.min_inside:
            c["drop_reason"] = f"봉투 내부비율 {c['inside']:.2f} < {args.min_inside}"
            dropped.append(c)
        else:
            kept.append(c)
    print(f"[3] 봉투 필터: {len(deduped)} -> {len(kept)} (제외 {len(dropped)})")
    by_reason: dict[str, int] = defaultdict(int)
    for c in dropped:
        by_reason[c["drop_reason"].split()[0]] += 1
    for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
        print(f"      {k}: {v}개")

    nonmirror = [c for c in kept if not c["mirror"]]
    sessions = sorted({c["session"] for c in nonmirror})
    union: dict[str, set[int]] = defaultdict(set)
    for c in nonmirror:
        union[c["session"]].update(range(c["f0"], c["f1"]))
    union_frames = sum(len(v) for v in union.values())
    md5_frames = sum(len(c["frames"]) for c in kept)

    print("\n[4] 규모 — 두 수치를 반드시 병기한다:")
    print(f"      md5-dedup 프레임 (미러 포함): {md5_frames:,}")
    print(f"      세션 합집합 프레임 (미러 제외): {union_frames:,}  ({union_frames / 3600:.2f}분 @60fps)")
    print(f"      원 녹화 세션: {len(sessions)}개   클립: {len(kept)} (미러 제외 {len(nonmirror)})")
    print(f"      논문 1(13,076) 대비: {union_frames / 13076:.2f}배")

    # ── 5) bin × 세션 다양성 ────────────────────────────────────────
    vx_edges = np.arange(args.vx_min, args.vx_max + 1e-9, 0.5)
    wz_edges = np.linspace(-1.5, 1.5, 7)
    nb_v, nb_w = len(vx_edges) - 1, len(wz_edges) - 1
    grid = [[set() for _ in range(nb_w)] for _ in range(nb_v)]
    counts = np.zeros((nb_v, nb_w), dtype=int)
    for c in nonmirror:
        sel = (c["vx"] >= vx_edges[0]) & (c["vx"] < vx_edges[-1]) & (c["wz"] >= wz_edges[0]) & (c["wz"] < wz_edges[-1])
        iv = np.clip(np.digitize(c["vx"], vx_edges) - 1, 0, nb_v - 1)
        iw = np.clip(np.digitize(c["wz"], wz_edges) - 1, 0, nb_w - 1)
        for a, b in zip(iv[sel], iw[sel]):
            counts[a, b] += 1
        for a, b in set(zip(iv[sel].tolist(), iw[sel].tolist())):
            grid[a][b].add(c["session"])
    filled = sum(1 for a in range(nb_v) for b in range(nb_w) if grid[a][b])
    ge3 = sum(1 for a in range(nb_v) for b in range(nb_w) if len(grid[a][b]) >= 3)
    total_bins = nb_v * nb_w
    print(f"\n[5] bin × 세션 다양성 ({nb_v}×{nb_w} = {total_bins}):")
    print(
        f"      채움 {filled}/{total_bins} ({100 * filled / total_bins:.1f}%),"
        f"  세션>=3 {ge3}/{total_bins} ({100 * ge3 / total_bins:.1f}%)"
    )
    _hdr = "vx \\ wz"
    print(f"      {_hdr:>13s}" + "".join(f"{wz_edges[b]:+8.1f}" for b in range(nb_w)))
    for a in range(nb_v):
        row = "".join(f"{len(grid[a][b]):4d}({counts[a, b] // 1000:3d}k)" for b in range(nb_w))
        print(f"      [{vx_edges[a]:+4.1f},{vx_edges[a + 1]:+4.1f})" + row)

    # ── 6) 세션 단위 train/val split ─────────────────────────────────
    # 클립 단위 split 은 누수가 난다 — 같은 세션의 F 구간이 겹치기 때문이다.
    rng = np.random.default_rng(args.seed)
    if args.split_mode == "random":
        order = list(sessions)
        rng.shuffle(order)
        val_sessions = sorted(order[: args.val_sessions])
    else:
        # 카테고리별 (세션 -> 클립 평균속도 목록)
        cat_sess: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for c in nonmirror:
            cat_sess[c["cat"]][c["session"]].append(float(c["vx"].mean()))

        chosen: list[str] = []
        for cat in args.strat_categories:
            d = cat_sess.get(cat, {})
            if not d:
                print(f"      [층화] '{cat}' 없음 — 건너뜀")
                continue

            def span(sel: list[str]) -> float:
                vs = [v for k in sel for v in d.get(k, [])]
                return (max(vs) - min(vs)) if len(vs) >= 2 else 0.0

            cur = [k for k in chosen if k in d]
            # 이미 확보된 간격이 부족하면, 간격을 가장 크게 넓히는 세션을 추가한다
            while span(cur) < args.strat_min_gap and len(chosen) < args.val_sessions:
                cands = [k for k in d if k not in chosen]
                if not cands:
                    break
                best = max(cands, key=lambda k: span(cur + [k]))
                if span(cur + [best]) <= span(cur) and cur:
                    break
                chosen.append(best)
                cur.append(best)
            print(
                f"      [층화] {cat:6s} val 세션 {len(cur)}개, 속도 간격 {span(cur):.2f} m/s"
                f"  {'OK' if span(cur) >= args.strat_min_gap else '★ 부족'}"
            )
        # 남은 자리는 무작위로 채운다 (walk 다양성)
        rest = [k for k in sessions if k not in chosen]
        rng.shuffle(rest)
        chosen += rest[: max(0, args.val_sessions - len(chosen))]
        val_sessions = sorted(chosen)

    train_sessions = sorted([s_ for s_ in sessions if s_ not in set(val_sessions)])
    print(f"\n[6] 세션 단위 split ({args.split_mode}): train {len(train_sessions)} / val {len(val_sessions)}")
    print(f"      val: {', '.join(val_sessions)}")

    # val 안에서 카테고리별로 실제 확보된 속도 범위를 보고한다 (조용한 실패 방지)
    vs_set = set(val_sessions)
    print(f"      {'카테고리':12s} {'val 클립':>8s} {'vx 범위':>16s} {'간격':>7s}")
    for cat in sorted({c["cat"] for c in nonmirror}):
        vv = [float(c["vx"].mean()) for c in nonmirror if c["cat"] == cat and c["session"] in vs_set]
        if not vv:
            continue
        gap = max(vv) - min(vv)
        flag = "" if gap >= args.strat_min_gap or cat not in args.strat_categories else "  ★ G2b 불가"
        print(f"      {cat:12s} {len(vv):8d} {min(vv):7.2f} ~{max(vv):7.2f} {gap:7.2f}{flag}")

    # ── 7) PKL 변환 ─────────────────────────────────────────────────
    if args.dry_run:
        print("\n[7] --dry_run — PKL 을 쓰지 않는다.")
    else:
        os.makedirs(args.dst_dir, exist_ok=True)
        for c in kept:
            fr = c["frames"]
            out = np.concatenate([fr[:, 0:3], quat_to_exp_map(fr[:, 3:7]), fr[:, 7:19]], axis=-1)
            assert out.shape[1] == 18, out.shape
            fps = round(1.0 / float(c["raw"]["FrameDuration"]))
            loop = 1 if str(c["raw"]["LoopMode"]).strip().lower() == "wrap" else 0
            name = os.path.splitext(os.path.basename(c["path"]))[0]
            with open(os.path.join(args.dst_dir, f"{c['cat']}__{name}.pkl"), "wb") as fh:
                pickle.dump({"loop_mode": loop, "fps": fps, "frames": out.tolist()}, fh)
        print(f"\n[7] PKL {len(kept)}개 -> {args.dst_dir}")

    # ── 8) 리포트 ───────────────────────────────────────────────────
    if args.report_dir:
        os.makedirs(args.report_dir, exist_ok=True)
        with open(os.path.join(args.report_dir, "clip_inventory.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(
                [
                    "category",
                    "session",
                    "mirror",
                    "f0",
                    "f1",
                    "frames",
                    "inside_frac",
                    "vx_min",
                    "vx_mean",
                    "vx_max",
                    "wz_absmax",
                    "kept",
                    "drop_reason",
                    "split",
                ]
            )
            for c in kept + dropped:
                is_kept = "drop_reason" not in c
                split = ("val" if c["session"] in val_sessions else "train") if is_kept else ""
                w.writerow(
                    [
                        c["cat"],
                        c["session"],
                        int(c["mirror"]),
                        c["f0"],
                        c["f1"],
                        len(c["frames"]),
                        f"{c['inside']:.4f}",
                        f"{c['vx'].min():.3f}",
                        f"{c['vx'].mean():.3f}",
                        f"{c['vx'].max():.3f}",
                        f"{np.abs(c['wz']).max():.3f}",
                        int(is_kept),
                        c.get("drop_reason", ""),
                        split,
                    ]
                )
        with open(os.path.join(args.report_dir, "bin_session_diversity.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["vx_lo", "vx_hi", "wz_lo", "wz_hi", "num_sessions", "num_frames"])
            for a in range(nb_v):
                for b in range(nb_w):
                    w.writerow(
                        [
                            f"{vx_edges[a]:.2f}",
                            f"{vx_edges[a + 1]:.2f}",
                            f"{wz_edges[b]:.2f}",
                            f"{wz_edges[b + 1]:.2f}",
                            len(grid[a][b]),
                            int(counts[a, b]),
                        ]
                    )
        with open(os.path.join(args.report_dir, "session_split.json"), "w") as fh:
            json.dump(
                {
                    "seed": args.seed,
                    "envelope": {
                        "vx_min": args.vx_min,
                        "vx_max": args.vx_max,
                        "wz_max": args.wz_max,
                        "min_inside": args.min_inside,
                    },
                    "split_mode": args.split_mode,
                    "train_sessions": train_sessions,
                    "val_sessions": val_sessions,
                    "num_clips_kept": len(kept),
                    "num_clips_dropped": len(dropped),
                    "md5_dedup_frames": int(md5_frames),
                    "session_union_frames": int(union_frames),
                    "num_sessions": len(sessions),
                    "bins_filled": filled,
                    "bins_ge3_sessions": ge3,
                    "bins_total": total_bins,
                },
                fh,
                indent=2,
            )
        print(f"[8] 리포트 -> {args.report_dir}")


if __name__ == "__main__":
    main()

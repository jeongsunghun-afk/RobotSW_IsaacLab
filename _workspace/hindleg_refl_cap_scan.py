# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

r"""여러 PACE 파라미터 집합을 **한 세션에서** 재생 비교한다 (README §23).

왜 필요했나
-----------
0818 재적합(`logs/pace/bipedleg_0818/26_08_18_17-28-28`)이 완주했는데, 나온 armature 가
`rga.py` 의 파생 반사관성과 3~100× 어긋난다::

    joint   적합      파생(I_r·N²)   비
    hip     0.0028    0.0363         0.08     ← 관절토크 기여 2.6%, 애초에 식별 불가
    thigh   0.0030    0.0363         0.08     ← 기여 31%, 식별 가능한데도 0 으로 감
    calf    0.0412    0.1338         0.31     ← 기여 68%
    foot    0.0175    0.0522         0.34     ← 기여 40%

score 곡선(단조 하강)과 좌우 대칭(calf 0.04120 vs 0.04125)만 봐서는 이 모순이 안 보인다.
**어느 쪽이 실측 궤적을 더 잘 재현하는가**로 판정해야 한다.

방법
----
env 하나당 파라미터 집합 하나를 실어 population 처럼 동시에 재생한다. 채점은 PACE 와
**같은 손실**을 쓴다 (`cma_es.tell`)::

    score = mean_t  sum_j (q_sim - q_real - bias)^2      [rad²]
    관절당 RMS = sqrt(score / 8)

커플링 규약도 적합기와 동일하게 맞춘다 (`fit_bipedleg.py:301`):
데이터셋 foot 은 raw(엔코더)각이므로 ①초기자세만 관절각으로 환산해 넣고 ②채점 때 sim 을
가상 엔코더 `q_f + q_c` 로 되돌려 **엔코더끼리** 비교한다.

2026-08-19 갱신 — 기본 데이터셋이 **0819 세트**(적합 3 / hold-out 4)로 바뀌었다.
08-18 때는 hold-out 이 사실 이전 적합의 in-sample 이라 중립 비교가 아니었다(§23-b 정정).
이번 hold-out 4 개는 **어느 적합에도 안 들어갔고** amp 0.2~1.0 · f1 1.5~5.0 을 덮는다.

이번에 볼 것 셋:

1. **최적 shift 가 0 인가** — 0 이면 시간축 문제가 해소된 것이다(브리지가 명령을 에코하고
   틱으로 시간을 못 박은 효과). 0 이 아니면 아직 남아 있다.
2. **진폭에 따라 갈리는가** — 갈리면 원인이 캡처가 아니라 **플랜트**다(§23-h 남은 후보).
3. **집합 B(파생 armature)가 A 를 따라잡는가** — 따라잡으면 PACE 가 armature 를 깎은 건
   물리가 아니라 delay 하한(§23-i) 때문이었다는 확정이다.

실행::

    ./isaaclab.sh -p reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/logs/eval_param_sets.py \
        --headless --cmd_shift -2 -1 0 1 2
    # --new 는 기본이 bipedleg_0819 의 **최신 run** (LATEST 가 자동 해석된다)
"""

import argparse

from isaaclab.app import AppLauncher

# 비교 대상 파라미터 — `--new` / `--prev` 로 덮어쓸 수 있다.
NEW_DEFAULT = "logs/pace/bipedleg_0819/LATEST/mean_199.pt"
PREV_DEFAULT = "logs/pace/bipedleg_0818/26_08_18_17-28-28/mean_199.pt"

# 0819 캡처 세트 — 적합 3개 / hold-out 4개. hold-out 은 **어느 적합에도 안 들어간** 것이라
# 08-18 때(hold-out 이 이전 적합의 in-sample 이었다)와 달리 중립 비교가 성립한다.
DATASETS_DEFAULT = [
    ("in  amp0.2 f2.0", "bipedleg_0819/chirp_gui_all_20260819_152544.pt"),
    ("in  amp0.8 f2.5", "bipedleg_0819/chirp_gui_all_20260819_153032.pt"),
    ("in  amp0.6 f4.0", "bipedleg_0819/chirp_gui_all_20260819_153311.pt"),
    ("HOLD amp0.2 f5.0", "bipedleg_0819/chirp_gui_all_20260819_152610.pt"),
    ("HOLD amp0.5 f3.0", "bipedleg_0819/chirp_gui_all_20260819_153226.pt"),
    ("HOLD amp0.8 f2.0", "bipedleg_0819/chirp_gui_all_20260819_152957.pt"),
    ("HOLD amp1.0 f1.5", "bipedleg_0819/chirp_gui_all_20260819_153139.pt"),
]



parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Isaac-R2S-BipedLeg-Sysid-v0")
parser.add_argument("--new", default=NEW_DEFAULT, help="집합 A/B/C 의 기준 파라미터 (data/ 아닌 repo 루트 기준). 'LATEST' 사용 가능")
parser.add_argument("--prev", default=PREV_DEFAULT, help="비교용 이전 적합 파라미터")
parser.add_argument("--datasets", nargs="*", default=None, help="평가할 .pt (data/ 기준). 생략 시 0819 세트 전체")
parser.add_argument(
    "--params",
    nargs="*",
    default=None,
    help="비교할 파라미터 집합을 직접 지정한다: `이름=값` 을 나열. 주면 A~D 기본 4 종 대신 "
    "이 목록만 재생하고 env 수도 목록 길이에 맞춘다. 값은 mean_*.pt 경로이거나 합성 키워드다: "
    "`stock`(rga.py 가 지금 배포 중인 값 — 파생 armature + 마찰 0.38/0.09), "
    "`nominal`(파생 armature 만, 마찰 0), `derived@<경로>`(그 적합의 마찰은 쓰되 armature 만 파생값).",
)
parser.add_argument(
    "--cmd_shift",
    type=int,
    nargs="+",
    default=[0],
    help="명령을 몇 step 앞당겨(음수면 늦춰) 먹일지. 1 step = 5 ms. 최적 shift 가 0 이면 "
    "시간축이 맞은 것이고, 0 이 아니거나 캡처마다 다르면 아직 어긋남이 남아 있다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from pace_sim2real.optim import MultiTrajectoryCMAES  # noqa: E402
from pace_sim2real.utils import project_root, require_physx_backend  # noqa: E402

LABELS = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
CALF_LM, FOOT_LM = [2, 6], [3, 7]

# rga.py 파생 반사관성 대각 — I_r(=7.4e-4)·N², calf 는 I_r(N_c²+N_f²).
DERIVED_ARMATURE = [0.0363, 0.0363, 0.1338, 0.0522] * 2

def resolve_latest(rel: str) -> str:
    """경로에 ``LATEST`` 가 있으면 그 자리의 **가장 최근 run 디렉터리**로 바꾼다.

    run 이름이 타임스탬프라 매번 손으로 고쳐 넣게 되는데, 그러다 **옛 run 을 평가하고
    새 것으로 착각**하기 쉽다. 여기서 풀어 준다.
    """
    if "LATEST" not in rel:
        return rel
    head, _, tail = rel.partition("/LATEST/")
    runs = sorted((project_root() / head).glob("*"))
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        raise SystemExit(f"[eval] run 이 없다: {head}")
    return str((runs[-1] / tail).relative_to(project_root()))


def load_params(rel: str, device: str) -> torch.Tensor:
    return torch.load(project_root() / rel, map_location=device).to(torch.float32).reshape(-1).clone()



# `rga.py` 가 **지금 실제로 배포하고 있는** 플랜트 (2026-08-26 기준). PACE 값은 한 번도
# 반영된 적이 없다 — `rga.py:541` 이 "PACE 가 식별하면 덮어쓰면 된다"고 적어 둔 상태 그대로다.
# 이 상수들을 바꿀 때는 rga.py 와 함께 고칠 것. 한쪽만 고치면 비교가 조용히 거짓말을 한다.
STOCK_VISCOUS = 0.09  # rga.py `viscous_friction={".*": 0.09}`
STOCK_COULOMB = 0.38  # rga.py `friction` / `dynamic_friction` = 0.38 [N·m]


def resolve_param_spec(rel: str, device: str) -> torch.Tensor:
    """``--params`` 의 값 부분을 33 벡터로 푼다.

    경로 외에 두 개의 합성 키워드를 받는다 — 배포 플랜트를 파일 없이 지목하기 위해서다.

    Args:
        rel: ``mean_*.pt`` 경로, 또는 ``"stock"`` / ``"nominal"`` / ``"derived@<경로>"``.
        device: 텐서를 올릴 디바이스.

    Returns:
        33 = armature(8) + viscous(8) + coulomb(8) + bias(8) + delay(1).
    """
    n = len(DERIVED_ARMATURE)
    if rel in ("stock", "nominal"):
        p = torch.zeros(4 * n + 1, device=device)
        p[0:n] = torch.tensor(DERIVED_ARMATURE, device=device)
        if rel == "stock":
            # 마찰까지 포함한 **실제 배포값**. nominal 은 마찰 0 인 대조군이라 서로 다르다.
            p[n : 2 * n] = STOCK_VISCOUS
            p[2 * n : 3 * n] = STOCK_COULOMB
        return p
    if rel.startswith("sym@"):
        # 좌우를 평균해 대칭화한다. viscous 좌우비 3.2~3.5 배는 kd 축퇴 아티팩트로 밝혀졌는데
        # (§37-e), 그대로 배포하면 **학습 플랜트가 좌우로 다른 로봇**이 된다. 대칭화가 hold-out 을
        # 얼마나 깎는지 재서, 아티팩트를 굽는 값이 얼마인지 값으로 보고 결정한다.
        p = load_params(resolve_latest(rel.split("@", 1)[1]), device)
        half = n // 2
        for blk in range(4):  # armature / viscous / coulomb / bias
            lo = blk * n
            m = 0.5 * (p[lo : lo + half] + p[lo + half : lo + n])
            p[lo : lo + half] = m
            p[lo + half : lo + n] = m
        return p
    if rel.startswith("derived@"):
        # 마찰·bias·delay 는 적합값 그대로 두고 armature 만 파생값으로 되돌린다 —
        # "이득이 armature 에서 나오나 마찰에서 나오나"를 가르는 절단이다.
        p = load_params(resolve_latest(rel.split("@", 1)[1]), device)
        p[0:n] = torch.tensor(DERIVED_ARMATURE, device=device)
        return p
    return load_params(resolve_latest(rel), device)


def build_sets(device: str) -> list[tuple[str, torch.Tensor]]:
    """비교할 파라미터 집합. 33 = armature(8)+viscous(8)+coulomb(8)+bias(8)+delay(1)."""
    # ``--params 이름=경로`` 를 주면 그 목록만 쓴다. 파생/nominal 대조군은 이미 판정이 끝난
    # 질문(§23·§25)의 대조군이라, 적합끼리만 견주는 A/B 에서는 env 를 두 칸 낭비할 뿐이다.
    if args_cli.params:
        out = []
        for spec in args_cli.params:
            name, _, rel = spec.partition("=")
            if not rel:
                raise SystemExit(f"[eval] --params 형식은 `이름=경로` 다: {spec!r}")
            out.append((name, resolve_param_spec(rel, device)))
        return out

    new_rel = resolve_latest(args_cli.new)
    prev_rel = resolve_latest(args_cli.prev)
    new = load_params(new_rel, device)
    prev = load_params(prev_rel, device)

    derived_arm = new.clone()
    derived_arm[0:8] = torch.tensor(DERIVED_ARMATURE, device=device)

    # PACE 를 전혀 안 쓴 판 — 파생 armature 만, 마찰·바이어스 0, 지연 0.
    nominal = torch.zeros_like(new)
    nominal[0:8] = torch.tensor(DERIVED_ARMATURE, device=device)

    # ⚠ 라벨은 **실제 경로에서 만든다**. 예전엔 "C 이전적합(gearfix_k2)" 처럼 박아 뒀는데,
    #   기본 경로를 바꾸는 순간 표가 거짓말을 한다 — 적합 비교에서 그건 치명적이다.
    def tag(rel: str) -> str:
        q = Path(rel).parts
        return f"{q[-3]}/{q[-1].replace('.pt', '')}" if len(q) >= 3 else rel

    return [
        (f"A 적합 {tag(new_rel)}", new),
        ("B A + armature=파생", derived_arm),
        (f"C 이전 {tag(prev_rel)}", prev),
        ("D 파생만(마찰·바이어스 0)", nominal),
    ]


def main() -> None:
    sets = None
    # env 하나가 파라미터 집합 하나를 싣는다 — 개수를 먼저 정해야 env 를 만들 수 있다.
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=len(args_cli.params or [0] * 4))
    env = gym.make(args_cli.task, cfg=env_cfg)
    require_physx_backend(env)
    device = env.unwrapped.device
    art = env.unwrapped.scene["robot"]
    s2r = env_cfg.sim2real
    order = s2r.joint_order
    n = len(order)

    sets = build_sets(device)
    n_set = len(sets)
    assert env.unwrapped.num_envs == n_set, f"num_envs({env.unwrapped.num_envs}) != 파라미터 집합 수({n_set})"

    jid = torch.tensor([art.joint_names.index(x) for x in order], dtype=torch.int32, device=device)
    jl = jid.to(torch.long)
    eid = torch.arange(n_set, dtype=torch.int32, device=device)

    P = torch.stack([p for _, p in sets])  # (n_set, 33)
    arm, vis, cou, bias = (P[:, i * n : (i + 1) * n] for i in range(4))
    # ⚠ **절삭**이어야 한다. PACE 는 `cma_es.py:187` 에서 `.to(torch.int)` 로 넣는다 — 반올림이 아니다.
    #   반올림하면 delay 0.510 인 집합이 1 step(5 ms) 을 더 먹어 적합 당시와 다른 조건으로 재생된다.
    delay = P[:, 4 * n].to(torch.int)

    print(f"\n{'=' * 96}")
    print("파라미터 집합")
    for i, (name, _) in enumerate(sets):
        print(f"  [{i}] {name}   armature={[round(float(x), 4) for x in arm[i]]}  delay={int(delay[i])}")

    coupled = bool(getattr(env_cfg, "sysid", False)) and bool(getattr(env_cfg, "foot_coupling", False))
    print(f"\n커플링 재생 = {coupled}  (채점: sim foot 을 raw 로 환산해 엔코더끼리 비교)")

    results = []
    REFL_MAX = []
    root = project_root() / "data"
    ds = DATASETS_DEFAULT if not args_cli.datasets else [(Path(x).stem[-6:], x) for x in args_cli.datasets]
    runs = [(f"{t} shift{s:+d}", r, s) for s in args_cli.cmd_shift for t, r in ds]
    with torch.inference_mode():
        for tag, rel, shift in runs:
            raw = torch.load(root / rel, map_location=device, weights_only=False)
            meas, tgt = raw["dof_pos"].to(device), raw["des_dof_pos"].to(device)
            kp, kd = raw["kp"].to(device).reshape(n), raw["kd"].to(device).reshape(n)
            steps = meas.shape[0]

            env.reset()
            MultiTrajectoryCMAES.apply_gains(art, jid, kp, kd)
            art.write_joint_armature_to_sim_index(armature=arm, joint_ids=jid, env_ids=eid)
            art.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=cou,
                joint_dynamic_friction_coeff=cou,
                joint_viscous_friction_coeff=vis,
                joint_ids=jid,
                env_ids=eid,
            )
            # ⚠ default_* 캐시에도 써야 한다 — env 의 raw 좌표 foot 마찰이 그 캐시를 읽는다.
            art.data.default_joint_armature.torch[:, jl] = arm
            art.data.default_joint_friction_coeff.torch[:, jl] = cou
            art.data.default_joint_viscous_friction_coeff.torch[:, jl] = vis
            for a in art.actuators.values():
                d_ids = a.joint_indices
                if isinstance(d_ids, slice):
                    d_ids = torch.arange(art.num_joints, device=device)[d_ids]
                d_ids = torch.as_tensor(d_ids, device=device).to(torch.long)
                pos = torch.tensor([int((jl == int(j)).nonzero()[0]) for j in d_ids], dtype=torch.long, device=device)
                a.update_encoder_bias(bias[:, pos])
                a.update_time_lags(delay)
                a.reset(eid)

            init = torch.zeros(n_set, art.num_joints, device=device)
            init[:, jl] = meas[0].unsqueeze(0)
            if coupled:
                init[:, [jl[i] for i in FOOT_LM]] -= init[:, [jl[i] for i in CALF_LM]]
            art.write_joint_position_to_sim_index(position=init)
            art.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(init))

            act = torch.zeros(n_set, art.num_joints, device=device)
            sq = torch.zeros(n_set, steps, n, device=device)
            # ★ 반사관성 항의 실측 크기: |I_off · joint_acc[foot]|  (PhysX joint_acc 기준)
            #   실기 위치 2회미분과 **추정량이 다르므로** 캡 값은 이 쪽으로 잡아야 한다.
            refl_mag = torch.zeros(n_set, steps, len(FOOT_LM), device=device)
            foot_art = torch.tensor([int(jl[i]) for i in FOOT_LM], device=device, dtype=torch.long)
            for t in range(steps):
                i_off_t = arm[:, FOOT_LM]
                refl_mag[:, t] = (i_off_t * art.data.joint_acc[:, foot_art]).abs()
                sp = art.data.joint_pos[:, jl].clone()
                if coupled:
                    sp[:, FOOT_LM] += sp[:, CALF_LM]
                sq[:, t] = sp
                # shift>0 = 명령을 앞당겨, shift<0 = 늦춰 먹인다. 양끝은 clamp.
                act[:, jl] = tgt[min(max(t + shift, 0), steps - 1)].unsqueeze(0)
                env.step(act)

            err = sq - meas.unsqueeze(0) - bias.unsqueeze(1)  # PACE 손실과 동일 (bias 차감)
            score = err.pow(2).sum(-1).mean(-1)  # (n_set,)  [rad²]
            per_joint = err.pow(2).mean(1).sqrt()  # (n_set, 8) [rad]
            # chirp 은 주파수를 쓸어 올리므로 시간 4분할 ≈ 주파수 4대역.
            # 최적 shift 가 대역마다 일정하면 **전송 지연**, 고주파로 갈수록 커지면 **플랜트 위상오차**.
            q4 = torch.stack([err[:, i * steps // 4 : (i + 1) * steps // 4].pow(2).sum(-1).mean(-1) for i in range(4)], 1)
            results.append((tag, rel, score, per_joint, q4))
            rm = refl_mag[0]
            print(f"\n  ★ reflI 항 |I_off·q̈_foot| [N·m]  mean {float(rm.mean()):7.3f}"
                  f"  p95 {float(rm.flatten().quantile(0.95)):7.3f}"
                  f"  p99 {float(rm.flatten().quantile(0.99)):7.3f}"
                  f"  **max {float(rm.max()):8.3f}**")
            REFL_MAX.append((Path(rel).name, float(rm.mean()), float(rm.flatten().quantile(0.99)), float(rm.max())))

            print(f"\n{'=' * 96}")
            print(f"[{tag}]  {Path(rel).name}   amp={raw['meta'].get('amplitude_scale')}  steps={steps}")
            print(f"  {'집합':26s} {'score[rad²]':>12s} {'관절RMS[rad]':>12s} {'[deg]':>8s}")
            for i, (name, _) in enumerate(sets):
                s = float(score[i])
                print(f"  {name:26s} {s:12.6f} {(s / n) ** 0.5:12.5f} {(s / n) ** 0.5 * 57.2958:8.2f}")
            print(f"\n  관절별 RMS [rad]  {' '.join(f'{x:>9s}' for x in LABELS)}")
            for i, (name, _) in enumerate(sets):
                print(f"  {name:26s} {' '.join(f'{float(v):9.4f}' for v in per_joint[i])}")

    print(f"\n{'=' * 96}")
    print("★★ 반사관성 항 실측 (PhysX joint_acc 기준) — 상수 캡 하한을 정하는 표")
    print(f"  {'capture':40s}{'mean':>9s}{'p99':>9s}{'max':>10s}")
    for nm_, mn_, p99_, mx_ in REFL_MAX:
        print(f"  {nm_[:38]:40s}{mn_:9.3f}{p99_:9.3f}{mx_:10.3f}")
    if REFL_MAX:
        allmax = max(x[3] for x in REFL_MAX)
        print(f"  {'적합 구간 전체 최댓값':40s}{'':9s}{'':9s}{allmax:10.3f}  N·m")
        print(f"  → 상수 캡은 이 값보다 커야 적합된 플랜트를 안 건드린다 (여유 2배 권장: {allmax * 2:.1f} N·m)")
        print(f"  → sim 보행 폭주는 313~406 N·m 이므로 그 사이 어디든 유효하다")
    print(f"\n{'=' * 96}")
    print("요약 — score [rad²] (낮을수록 실측 재현이 좋다)")
    for tag, _, sc, _, _ in results:
        print(f"  {tag:34s}" + "".join(f"  [{i}]{float(sc[i]):9.6f}" for i in range(n_set)))

    # ── 대역별 최적 shift ──────────────────────────────────────────────────────────
    # chirp 시간 4분할 = 저→고 주파수. 각 (캡처, 집합, 대역) 에서 score 최소인 shift 를 찾는다.
    print(f"\n{'=' * 96}")
    print("★ 대역별 최적 shift [step, 1=5ms]  — chirp 4분할(저주파→고주파)")
    print("   대역마다 일정 → 전송 지연 / 고주파로 갈수록 증가 → 플랜트 위상오차")
    by_ds: dict[str, list] = {}
    for tag, rel, _, _, q4 in results:
        base = tag.rsplit(" shift", 1)[0]
        shift = int(tag.rsplit("shift", 1)[1])
        by_ds.setdefault(base, []).append((shift, q4))
    for base, entries in by_ds.items():
        entries.sort()
        shifts = torch.tensor([s for s, _ in entries])
        stack = torch.stack([q for _, q in entries])  # (n_shift, n_set, 4)
        print(f"\n  [{base}]")
        print(f"    {'집합':26s}" + "".join(f"{'대역'+str(b+1):>12s}" for b in range(4)))
        for i, (name, _) in enumerate(sets):
            best = [int(shifts[int(stack[:, i, b].argmin())]) for b in range(4)]
            vals = [float(stack[:, i, b].min()) for b in range(4)]
            print(f"    {name:26s}" + "".join(f"{best[b]:+6d}({vals[b]:.5f})".rjust(12) for b in range(4)))
    print(
        "\n판정 지침:\n"
        "  · B(파생 armature)가 A 와 비슷하면 → 위치 재현은 armature 를 거의 안 가린다.\n"
        "    그러면 rga.py 는 물리 유래 파생값을 그대로 두는 게 맞다 (적합값은 위치에만 최적).\n"
        "  · B 가 A 보다 크게 나쁘면 → sim 의 링크 관성이 이미 과대해 PACE 가 armature 를\n"
        "    깎아 상쇄한 것이다. 그때는 USD 링크 관성부터 검증해야 한다.\n"
        "  · hold-out 은 amp 0.1 이라 armature 판정에는 무력하다 — bias/마찰/지연만 본다."
    )
    simulation_app.close()


main()

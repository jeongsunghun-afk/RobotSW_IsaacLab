# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""R2S-BipedLeg PACE 적합 — 여러 녹화 시퀀스를 동시에 맞춘다.

8-DOF 2족 다리(``HIND_LEG_CFG``)용. ``fit_go2.py``의 다중 궤적 CMA-ES 루프를 그대로 이식했고,
파라미터 수만 49개(12관절)에서 **33개**(armature 8 + viscous 8 + coulomb 8 + bias 8 + delay 1)로 바뀐다.

upstream ``scripts/pace/fit.py``는 궤적 하나만 재생한다. 이 스크립트는 ``BipedLegPaceCfg.datasets``의
모든 시퀀스를 한 세대 안에서 차례로 재생하고 점수를 합산해, **하나의 파라미터 집합이 모든 시퀀스를
동시에 설명**하도록 만든다. 시퀀스마다 **녹화 당시의 PD 게인을 복원**해서 재생한다.

PACE는 PD 게인을 식별하지 않는다 — 논문이 밝히듯 ``{armature, damping, kp, kd}``의 공통 스케일이
폐루프 거동을 보존하므로 최적해가 축퇴한다. 따라서 게인은 **알려진 입력**이고, 잘못된 게인으로
재생하면 그 불일치가 플랜트 파라미터를 편향시킨다. go2와 달리 이 다리는 게인이 스칼라가 아니라
**관절별 벡터**(``NOMINAL_KP``/``NOMINAL_KD``)이며, 데이터셋마다 배율(``gain_scale``)만 다르다.

hold-out 검증(``BipedLegPaceCfg.holdout``)은 적합에 쓰지 않는다. 논문의 전신 단계 검증도 **보지 않은
PD 게인**에서 재현되는지를 본다.

⚠ 모델 갭: PACE 액추에이터는 :class:`PaceDCMotor`(토크-속도 포화 곡선이 있는 DC 모터)다. 반면
``HIND_LEG_CFG``와 ``direct/hind_leg`` RL env는 :class:`ImplicitActuator`를 쓴다. 여기서 식별된
파라미터는 **DC 모터 플랜트 기준**이므로, RL env로 옮기려면 그 env도 DC 모터로 바꾸거나 갭을 감수해야 한다.

실행::

    python scripts/real2sim/fit_bipedleg.py --headless --num_envs 4096
    tensorboard --logdir logs/pace/bipedleg_sim
"""

"""Launch Omniverse Toolkit first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="R2S-BipedLeg multi-trajectory PACE system identification.")
parser.add_argument("--num_envs", type=int, default=4096, help="CMA-ES population (one candidate per env).")
parser.add_argument("--task", type=str, default="Isaac-R2S-BipedLeg-Sysid-v0", help="Name of the task.")
parser.add_argument(
    "--datasets",
    nargs="*",
    default=None,
    help="적합용 .pt 목록 (<repo>/data/ 기준 상대경로) — BipedLegPaceCfg.datasets 오버라이드. "
    "실기 데이터(data/bipedleg_real/*.pt, convert_gui_chirp_bipedleg.py 산출물)를 줄 때 사용.",
)
parser.add_argument("--holdout", nargs="*", default=None, help="hold-out .pt 목록 — cfg 오버라이드 (적합 미사용).")
parser.add_argument("--robot_name", default=None, help="로그 디렉토리 이름 (logs/pace/<robot_name>) 오버라이드.")
parser.add_argument("--max_iteration", type=int, default=None, help="CMA-ES 세대 수 오버라이드 (기본 cfg=200).")
parser.add_argument(
    "--fit_joints",
    default=None,
    help="부분 적합: 이 관절들의 파라미터(armature/viscous/coulomb/bias)+delay만 탐색하고 나머지는 "
    "--freeze_from 값으로 고정. 콤마 구분 — 그룹(hip/thigh/calf/foot) 또는 관절명(HL_foot 등). "
    "예: 'foot' = foot 단독 chirp로 foot만 재적합 (커플링 잔차 판별용).",
)
parser.add_argument(
    "--freeze_from",
    default=None,
    help="--fit_joints에서 고정할 파라미터의 출처 — 이전 적합의 mean_*.pt (이미 물리 단위, 33개).",
)
parser.add_argument(
    "--freeze_bias",
    default=None,
    help="encoder bias 8개를 고정하고 나머지 25개만 탐색한다. 값은 'zero'(전부 0) 또는 mean_*.pt 경로. "
    "bias 가 잉여인지 재는 용도 — hold-out 이 거의 안 나빠지면 33→25 차원으로 줄일 수 있고, "
    "그만큼 나머지 파라미터의 조건수가 좋아진다 (README §29-e).",
)
parser.add_argument(
    "--freeze_bias_joints",
    default=None,
    help="--freeze_bias 를 이 관절들에만 적용한다 (콤마 구분, 그룹명 또는 관절명). 생략하면 8개 전부. "
    "예: 'hip,thigh,calf' = foot bias 만 자유. foot 은 raw 엔코더 채널이라 bias 가 영점이 아니라 "
    "커플링 계수 오차를 흡수하고 있을 수 있다 (README §30-b).",
)
parser.add_argument(
    "--bias_bound",
    type=float,
    default=None,
    help="encoder bias 탐색 범위를 ±이 값으로 덮어쓴다 [rad] (기본 0.1). "
    "현재 4개(HL/HR hip·foot)가 ±0.1 레일에 붙어 있어, 넓히면 어디서 멈추는지 보인다. "
    "hip 이 6~8°(0.105~0.14 rad)에서 멈추면 그게 참값이고, 새 레일까지 달리면 bias 가 "
    "영점이 아닌 다른 것을 흡수하고 있다는 뜻이다 (README §26-e).",
)
parser.add_argument(
    "--freeze_armature",
    default=None,
    help="armature 8개를 고정하고 나머지 25개(viscous/coulomb/bias/delay)만 탐색한다. 값은 "
    "'derived'(rga.py 파생 I_r·N²) 또는 mean_*.pt 경로. --fit_joints와 반대 방향의 절단으로, "
    "관절이 아니라 **블록**을 접는다. "
    "'파생 armature로도 마찰·지연이 벌충해 같은 score에 도달하는가'를 재는 용도 — 도달하면 "
    "armature 초과분은 식별 아티팩트이고, 못 하면 진짜 관성이다 (README §25).",
)
parser.add_argument(
    "--freeze_armature_joints",
    default=None,
    help="--freeze_armature 를 이 관절들에만 적용한다 (콤마 구분, 그룹명 또는 관절명). 생략하면 8개 전부. "
    "예: 'hip' = hip armature 만 고정하고 hip 마찰·bias 는 자유. 캡처가 그 관절의 공진대를 "
    "지나지 않으면(hip: 2.0 Hz 에서 멈춘 반면 공진은 2.80 Hz) 관성은 원리적으로 식별 불가인데, "
    "자유로 두면 저주파 잔차를 최소화하는 값으로 흘러가 **마찰 신호를 먹는다** (README §29-b).",
)
parser.add_argument(
    "--foot_transpose",
    choices=["on", "off"],
    default=None,
    help="전치 토크 τ_calf += τ_foot 강제 on/off (기본 = cfg 값, 2026-08-14 기준 on). "
    "'벨트가 무릎을 건너가는가'의 A/B 판정용.",
)
parser.add_argument(
    "--foot_raw_friction",
    choices=["on", "off"],
    default=None,
    help="foot 마찰을 raw(모터축) 좌표로 계산할지 on/off (기본 = cfg 값, 2026-08-14 기준 on). "
    "on이면 foot 관절의 PhysX 마찰을 끄고 viscous/coulomb 슬롯을 b_raw/c_raw로 재해석해 "
    "w_raw = q̇_foot + q̇_calf 에 걸며, 그 토크를 foot·calf 양쪽에 같은 부호로 싣는다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

from pace_sim2real.optim import MultiTrajectoryCMAES  # isort: skip
from pace_sim2real.utils import project_root, require_physx_backend  # isort: skip


def load_dataset(path, joint_order: list[str], device: str) -> dict:
    """녹화 시퀀스를 읽고 관절 순서 / 게인 존재를 검증한다.

    Args:
        path: ``chirp_*.pt`` 경로.
        joint_order: 기대하는 관절 순서 (8개, leg-major).
        device: 텐서를 올릴 디바이스.

    Returns:
        ``time``, ``dof_pos``, ``des_dof_pos``, ``kp``, ``kd``, ``name``을 담은 dict.

    Raises:
        FileNotFoundError: 파일이 없을 때.
        RuntimeError: 저장된 관절 순서가 다르거나 kp/kd가 없을 때.
    """
    if not path.exists():
        raise FileNotFoundError(f"데이터셋이 없다: {path}")
    raw = torch.load(path)

    stored_order = raw.get("joint_order")
    if stored_order is not None and list(stored_order) != list(joint_order):
        raise RuntimeError(
            f"{path.name}의 관절 순서가 cfg의 joint_order와 다르다.\n  저장: {stored_order}\n  기대: {joint_order}"
        )

    num_joints = len(joint_order)
    if "kp" not in raw or "kd" not in raw:
        raise RuntimeError(
            f"{path.name}에 kp/kd가 없다. 녹화 당시의 게인을 모르면 재생이 성립하지 않는다. "
            "PACE는 게인을 식별하지 않으므로 게인은 알려진 입력이어야 한다 — "
            "collector를 최신 스키마로 다시 돌릴 것."
        )

    return {
        "name": path.name,
        "time": raw["time"].to(device),
        "dof_pos": raw["dof_pos"].to(device),
        "des_dof_pos": raw["des_dof_pos"].to(device),
        "kp": raw["kp"].to(device).reshape(num_joints),
        "kd": raw["kd"].to(device).reshape(num_joints),
        "meta": raw.get("meta", {}),
    }


def gain_summary(kp: torch.Tensor, kd: torch.Tensor) -> str:
    """관절별 게인 벡터를 한 줄로 요약한다 (HL 다리 4개만, HR은 미러라 동일).

    Args:
        kp: 위치 게인 [N·m/rad], shape (8,).
        kd: 속도 게인 [N·m·s/rad], shape (8,).

    Returns:
        표에 넣을 요약 문자열.
    """
    kp_s = " ".join(f"{float(v):.1f}" for v in kp[:4])
    kd_s = " ".join(f"{float(v):.2f}" for v in kd[:4])
    return f"kp=[{kp_s}] kd=[{kd_s}]"


JOINT_GROUPS = {"hip": [0, 4], "thigh": [1, 5], "calf": [2, 6], "foot": [3, 7]}


def select_joints(spec: str | None, joint_order: list[str], flag: str) -> list[int]:
    """``"hip,HL_foot"`` 같은 토큰 목록을 leg-major 인덱스로 푼다.

    Args:
        spec: 콤마 구분 토큰. 그룹명(hip/thigh/calf/foot) 또는 관절명. ``None`` 이면 전체.
        joint_order: cfg 의 관절 순서 (8개, leg-major).
        flag: 오류 메시지에 쓸 CLI 플래그 이름.

    Returns:
        정렬된 관절 인덱스. ``spec`` 이 ``None`` 이면 전체 인덱스.
    """
    if spec is None:
        return list(range(len(joint_order)))
    sel: set[int] = set()
    for tok in spec.split(","):
        tok = tok.strip()
        if tok in JOINT_GROUPS:
            sel.update(JOINT_GROUPS[tok])
        else:
            full = tok if tok.endswith("_joint") else tok + "_joint"
            if full not in joint_order:
                raise RuntimeError(f"{flag} 토큰 인식 불가: {tok} (그룹 {list(JOINT_GROUPS)} 또는 관절명)")
            sel.add(joint_order.index(full))
    return sorted(sel)


def freeze_armature_bounds(bounds: torch.Tensor, num_joints: int, source: str, joints: list[int] | None = None) -> None:
    """armature 블록의 bounds 폭을 0 으로 접어 상수로 만든다 (``--freeze_armature``).

    ``--fit_joints`` 와 같은 수법이지만 접는 방향이 다르다 — 관절이 아니라 **블록**이다.
    나머지 25개(viscous/coulomb/bias/delay)는 자유롭게 탐색하므로, 파생 armature 의 부족분을
    마찰·지연이 벌충할 수 있는지가 그대로 score 에 드러난다 (README §25-g).

    Args:
        bounds: 탐색 범위 [하한, 상한], shape (4*num_joints + 1, 2). 제자리에서 수정된다.
        num_joints: 관절 수.
        source: ``"derived"`` 이면 rga.py 파생 반사관성, 아니면 mean_*.pt 경로.
        joints: 접을 관절 인덱스. ``None`` 이면 8개 전부. 일부만 주면 **그 관절의 armature 만**
            상수가 되고 같은 관절의 마찰·bias 는 자유롭게 남는다 — 여기가 값을 못 싣는 관절
            (hip: 공진대 캡처가 없어 관성이 식별 불가)이 마찰 신호를 먹어치우는 것을 막는다.
    """
    if source == "derived":
        # rga.py 의 파생 반사관성 대각 — I_r(=7.4e-4)·N², calf 는 I_r(N_c²+N_f²).
        # ⚠ r2s_biped_leg_sysid_cfg.py 의 armature 초기치와 같은 값이어야 한다.
        src = torch.tensor([0.0363, 0.0363, 0.1338, 0.0522] * 2, dtype=torch.float32)
    else:
        src = torch.load(source, map_location="cpu").to(torch.float32).reshape(-1)[:num_joints]
    if src.numel() != num_joints:
        raise RuntimeError(f"--freeze_armature 값 개수 불일치: {src.numel()} != {num_joints}")
    sel = list(range(num_joints)) if joints is None else joints
    for i in sel:
        bounds[i, 0] = src[i]
        bounds[i, 1] = src[i]
    print(
        f"[INFO]: armature 고정 ({source}) 관절 {sel} — {[round(float(src[i]), 4) for i in sel]}, "
        f"나머지 {bounds.shape[0] - len(sel)}개만 탐색",
        flush=True,
    )


def freeze_bias_bounds(bounds: torch.Tensor, num_joints: int, source: str, joints: list[int] | None = None) -> None:
    """encoder bias 블록의 bounds 폭을 0 으로 접어 상수로 만든다 (``--freeze_bias``).

    :func:`freeze_armature_bounds` 와 같은 수법이고 블록 위치만 다르다
    (bias 는 3*num_joints ~ 4*num_joints).

    Args:
        bounds: 탐색 범위 [하한, 상한], shape (4*num_joints + 1, 2). 제자리에서 수정된다.
        num_joints: 관절 수.
        source: ``"zero"`` 이면 전부 0, 아니면 mean_*.pt 경로.
        joints: 접을 관절 인덱스. ``None`` 이면 8개 전부. foot 만 남기는 용도가 있다 —
            foot 은 raw 엔코더 채널(``raw = q_foot + coef·q_calf``)이라 bias 가 영점이 아니라
            커플링 계수 오차를 흡수하고 있을 수 있고(§30-b 에서 새 레일까지 달렸다), 그 잔차를
            0 으로 접으면 foot 의 마찰·관성으로 밀려난다.
    """
    lo, hi = 3 * num_joints, 4 * num_joints
    if source == "zero":
        src = torch.zeros(num_joints, dtype=torch.float32)
    else:
        src = torch.load(source, map_location="cpu").to(torch.float32).reshape(-1)[lo:hi]
    if src.numel() != num_joints:
        raise RuntimeError(f"--freeze_bias 값 개수 불일치: {src.numel()} != {num_joints}")
    sel = list(range(num_joints)) if joints is None else joints
    for i in sel:
        bounds[lo + i, 0] = src[i]
        bounds[lo + i, 1] = src[i]
    print(
        f"[INFO]: bias 고정 ({source}) 관절 {sel} — {[round(float(src[i]), 4) for i in sel]}, "
        f"나머지 {bounds.shape[0] - len(sel)}개만 탐색",
        flush=True,
    )


def widen_bias_bounds(bounds: torch.Tensor, num_joints: int, half_width: float) -> None:
    """encoder bias 탐색 범위를 ±``half_width`` 로 덮어쓴다 (``--bias_bound``).

    Args:
        bounds: 탐색 범위, shape (4*num_joints + 1, 2). 제자리에서 수정된다.
        num_joints: 관절 수.
        half_width: 새 반폭 [rad].
    """
    lo, hi = 3 * num_joints, 4 * num_joints
    bounds[lo:hi, 0] = -half_width
    bounds[lo:hi, 1] = half_width
    print(f"[INFO]: bias bounds = ±{half_width} rad (±{half_width * 57.2958:.1f}°)", flush=True)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Newton 백엔드는 static 마찰만 지원해 Coulomb/viscous 식별이 성립하지 않는다.
    require_physx_backend(env)

    device = env.unwrapped.device
    articulation = env.unwrapped.scene["robot"]
    sim2real = env_cfg.sim2real
    # CLI 오버라이드 — 실기 데이터셋 적합 등 cfg 편집 없이 데이터/로그 대상을 바꾼다.
    if args_cli.datasets is not None:
        sim2real.datasets = args_cli.datasets
    if args_cli.holdout is not None:
        sim2real.holdout = args_cli.holdout
    if args_cli.robot_name is not None:
        sim2real.robot_name = args_cli.robot_name
    if args_cli.max_iteration is not None:
        sim2real.cmaes.max_iteration = args_cli.max_iteration
    if args_cli.foot_transpose is not None:
        # env.unwrapped.cfg에 직접 쓴다 — gym.make가 cfg를 복사해도 런타임이 보는 쪽이 여기다.
        env.unwrapped.cfg.foot_transpose = args_cli.foot_transpose == "on"
    if args_cli.foot_raw_friction is not None:
        env.unwrapped.cfg.foot_raw_friction = args_cli.foot_raw_friction == "on"
    print(f"[INFO]: 전치 토크(τ_calf += τ_foot) = {'ON' if env.unwrapped.cfg.foot_transpose else 'OFF'}")
    print(
        "[INFO]: foot 마찰 좌표 = "
        + (
            "raw(모터축) — viscous/coulomb[foot] 슬롯은 b_raw/c_raw 재해석"
            if env.unwrapped.cfg.foot_raw_friction
            else "관절(PhysX 기본)"
        )
    )
    joint_order = sim2real.joint_order

    # 부분 적합 (--fit_joints): 선택 관절의 4개 블록(armature/viscous/coulomb/bias)+delay만 탐색.
    # 나머지 파라미터는 bounds 상하한을 freeze 값으로 접어(폭 0) 상수로 만든다 — CMA-ES가 그 차원을
    # 탐색해도 sim에는 항상 고정값이 쓰인다. foot 단독 chirp로 foot 물성만 재적합해 "all-joint
    # 적합의 foot viscous가 커플링 잔차인지"를 판별하는 용도 (calf 정지 ⇒ raw ≡ 관절각).
    if args_cli.fit_joints is not None:
        if args_cli.freeze_from is None:
            raise RuntimeError("--fit_joints에는 --freeze_from(이전 적합 mean_*.pt)이 필요하다.")
        frozen = torch.load(args_cli.freeze_from, map_location="cpu").to(torch.float32)
        if frozen.numel() != sim2real.bounds_params.shape[0]:
            raise RuntimeError(f"freeze_from 파라미터 수 불일치: {frozen.numel()} != {sim2real.bounds_params.shape[0]}")
        lm_sel = select_joints(args_cli.fit_joints, joint_order, "--fit_joints")
        n = len(joint_order)
        free = {4 * n}  # delay는 항상 함께 식별
        for blk in range(4):
            free.update(blk * n + i for i in lm_sel)
        for k in range(sim2real.bounds_params.shape[0]):
            if k not in free:
                sim2real.bounds_params[k, 0] = frozen[k]
                sim2real.bounds_params[k, 1] = frozen[k]
        print(
            f"[INFO]: 부분 적합 — 탐색 {len(free)}개(관절 {lm_sel} × 4블록 + delay), "
            f"나머지 {sim2real.bounds_params.shape[0] - len(free)}개는 {args_cli.freeze_from} 값으로 고정"
        )

    if args_cli.freeze_armature is not None:
        freeze_armature_bounds(
            sim2real.bounds_params,
            len(joint_order),
            args_cli.freeze_armature,
            select_joints(args_cli.freeze_armature_joints, joint_order, "--freeze_armature_joints"),
        )

    # ⚠ 순서 주의 — 넓히기를 먼저, 고정을 나중에. 둘 다 주면 고정이 이긴다.
    if args_cli.bias_bound is not None:
        widen_bias_bounds(sim2real.bounds_params, len(joint_order), args_cli.bias_bound)
    if args_cli.freeze_bias is not None:
        freeze_bias_bounds(
            sim2real.bounds_params,
            len(joint_order),
            args_cli.freeze_bias,
            select_joints(args_cli.freeze_bias_joints, joint_order, "--freeze_bias_joints"),
        )

    # warp 커널은 관절 인덱스를 int32로 요구한다. 텐서 인덱싱에는 long 버전을 쓴다.
    joint_ids = torch.tensor(
        [articulation.joint_names.index(name) for name in joint_order], dtype=torch.int32, device=device
    )
    joint_ids_long = joint_ids.to(torch.long)

    data_root = project_root() / "data"
    datasets = [load_dataset(data_root / rel, joint_order, device) for rel in sim2real.datasets]
    if not datasets:
        raise RuntimeError("BipedLegPaceCfg.datasets가 비어 있다.")

    print("[INFO]: 파라미터 33개 = armature 8 + viscous 8 + coulomb 8 + bias 8 + delay 1")
    print(f"[INFO]: {len(datasets)}개 시퀀스로 적합한다:")
    for d in datasets:
        print(f"         {d['name']:28s} {d['dof_pos'].shape[0]:6d} steps  {gain_summary(d['kp'], d['kd'])}")
    if sim2real.holdout:
        print(f"[INFO]: hold-out (적합에 미사용): {sim2real.holdout}")

    log_dir = project_root() / "logs" / "pace" / sim2real.robot_name
    opt = MultiTrajectoryCMAES(
        bounds=sim2real.bounds_params.to(device),
        population_size=env.unwrapped.num_envs,
        log_dir=log_dir,
        joint_order=joint_order,
        max_iteration=sim2real.cmaes.max_iteration,
        # 상위 클래스는 config.pt 저장과 궤적 버퍼 크기 산정에 첫 시퀀스를 쓴다.
        data=datasets[0],
        device=device,
        epsilon=sim2real.cmaes.epsilon,
        sigma=sim2real.cmaes.sigma,
        save_interval=sim2real.cmaes.save_interval,
        save_optimization_process=sim2real.cmaes.save_optimization_process,
    )

    actions = torch.zeros(env.unwrapped.num_envs, articulation.num_joints, device=device)

    # foot↔calf 커플링 재생 (env_cfg.foot_coupling=True): 데이터셋의 foot 명령·실측은 raw각
    # (= 엔코더가 실제로 재는 유일한 양, q_foot+q_calf)이다. env가 raw 구동+전치를 재생하고,
    # 여기서는 ①초기 자세만 관절각으로 환산해 넣고(sim 상태는 관절각) ②채점 시 sim 쪽도
    # 가상 엔코더(q_f+q_c)로 환산해 **엔코더끼리** 비교한다.
    coupled = bool(getattr(env_cfg, "foot_coupling", False))
    calf_lm = [2, 6]  # leg-major (HL_calf, HR_calf)
    foot_lm = [3, 7]  # leg-major (HL_foot, HR_foot)
    if coupled:
        print("[INFO]: foot↔calf 커플링 재생 ON — 데이터셋 foot=raw(엔코더) 규약, 채점도 raw끼리")
        for d in datasets:
            if d.get("meta", {}).get("coupling_converted", None) is True:
                raise RuntimeError(
                    f"{d['name']}: 관절각으로 변환된 데이터셋이다 — 커플링 재생에는 raw 데이터가 필요하다. "
                    "convert_gui_chirp_bipedleg.py --keep_raw_foot 산출물을 쓸 것."
                )

    # 루프 전체를 inference_mode로 감싼다. env.step이 만든 버퍼는 inference tensor가 되므로,
    # 이후의 env.reset()/actuator write를 inference_mode 밖에서 하면 in-place 갱신이 거부된다
    # (upstream fit.py도 같은 이유로 전체를 감싼다).
    with torch.inference_mode():
        while simulation_app.is_running():
            for idx, dataset in enumerate(datasets):
                measured = dataset["dof_pos"]
                target = dataset["des_dof_pos"]
                num_steps = measured.shape[0]
                initial_pos = measured[0].unsqueeze(0).repeat(env.unwrapped.num_envs, 1).clone()
                if coupled:
                    # 실측 foot은 raw — sim 관절 상태 초기화는 관절각으로 (q_f = raw − q_c)
                    initial_pos[:, foot_lm] -= initial_pos[:, calf_lm]

                # 이 시퀀스가 녹화될 때의 게인을 복원한 뒤, 후보 파라미터를 다시 써 넣는다.
                # explicit actuator는 PD를 파이썬에서 계산하므로 진짜 게인은 actuator.stiffness /
                # actuator.damping 텐서다 — write_joint_stiffness_to_sim*은 여기서 no-op이다.
                env.reset()
                MultiTrajectoryCMAES.apply_gains(articulation, joint_ids, dataset["kp"], dataset["kd"])
                opt.update_simulator(articulation, joint_ids, initial_pos)
                opt.begin_trajectory(idx)

                for counter in range(num_steps):
                    sim_pos = articulation.data.joint_pos[:, joint_ids_long]
                    if coupled:
                        # 가상 엔코더: sim foot도 raw(q_f+q_c)로 환산해 실기 엔코더와 같은 양끼리 비교
                        sim_pos = sim_pos.clone()
                        sim_pos[:, foot_lm] += sim_pos[:, calf_lm]
                    opt.tell(
                        sim_pos,
                        measured[counter].unsqueeze(0).repeat(env.unwrapped.num_envs, 1),
                    )
                    actions[:, joint_ids_long] = target[counter].unsqueeze(0)
                    env.step(actions)

            opt.evolve()
            if opt.finished():
                break

    opt.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""``Go2-Pedipulation-v0`` S1 게이트 평가 — G5(push 강건성) / G7(base 표류).

학습 로그가 아니라 **고정 시드·고정 명령**의 별도 패스에서 판정한다 (PLAN §5 공통 규약).

``--mode push`` (S1-G5)
    base 에 0.1 s 수평 외력을 가하고 낙상/추종 유지 여부를 본다. 힘 대역과 지속시간은
    Variable Stiffness 문헌 인용값(50~300 N, 0.1 s, 랜덤 시점)을 따른다.

    **4족 기준선의 정의는 PLAN 문언과 다르다.** ``leg_role`` 을 전부 1 로 두는 것이
    문자 그대로의 "4족" 이지만, 학습 중 조작 다리는 항상 1 개였으므로 all-ones 역할
    벡터는 분포 밖 입력이고 그렇게 얻은 곡선은 기준선 구실을 못 한다. 대신 **조작
    다리의 목표를 nominal 발 위치 근처로 주어 발이 지면에 남게** 한다. 명령 표현은
    학습 분포 안에 있으면서 지지 다각형만 4족으로 유지되므로, 3족 조건과의 차이가
    "지지 다각형 축소" 단독으로 분리된다. 조건이 실제로 성립했는지는 접촉 센서로
    측정한 지지 발 개수(``stance4_rate``)로 검증한다.

``--mode drift`` (S1-G7)
    외란 없이 3족 hold 를 유지하며 base xy 이동을 본다. PLAN §5 의 지시대로 **누적
    경로장(path length)과 순변위(net displacement)를 분리**해 로깅한다 — 자세를 한 번
    바꾸고 머무는 counterbalance 와 계속 밀려나는 진짜 표류는 순변위만으로 구분되지
    않는다.

사용 예::

    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_pedipulation_gates.py \
        --mode push --checkpoint logs/rsl_rl/go2_pedipulation/<run>/model_10600.pt \
        --num_envs 1024 --seeds 3 --headless
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2-Pedipulation-v0 S1 게이트(G5/G7) 평가.")
parser.add_argument("--checkpoint", type=str, required=True, help="model_*.pt 경로.")
parser.add_argument(
    "--mode",
    type=str,
    default="push",
    choices=["push", "drift", "hold", "circle", "video", "showcase"],
    help="push=S1-G5, drift=S1-G7, hold=S1-G2/G3, circle=S2-G1/G2/G4/G5, video/showcase=영상.",
)
parser.add_argument(
    "--showcase_stage",
    type=str,
    default="s1",
    choices=["s1", "s2"],
    help="showcase 대본. s1=정지 목표 도달·유지, s2=연속 원 궤적 추종.",
)
parser.add_argument("--circle_omega", type=float, default=0.5, help="circle 모드 각속도 [rad/s].")
parser.add_argument("--circle_radius", type=float, default=0.10, help="circle 모드 반경 [m].")
parser.add_argument("--video_folder", type=str, default=None, help="video 모드 출력 폴더.")
parser.add_argument("--video_length", type=int, default=1000, help="video 모드 기록 길이 [step].")
parser.add_argument(
    "--video_leg",
    type=str,
    default="cycle",
    choices=["cycle", "FL", "FR", "RL", "RR"],
    help="video 모드에서 조작할 다리. 특정 다리를 강제하면 leg 매핑을 육안으로 검증할 수 있다.",
)
parser.add_argument("--num_envs", type=int, default=1024, help="시드당 병렬 시행 수.")
parser.add_argument("--seeds", type=int, default=3, help="시드 개수 (PLAN §5: 최소 3).")
parser.add_argument("--seed0", type=int, default=1000, help="첫 시드 값. 시드는 seed0, seed0+1, ... 로 진행.")
parser.add_argument("--out", type=str, default=None, help="결과 JSON 저장 경로.")
parser.add_argument(
    "--stance4_z_jitter",
    type=float,
    default=0.01,
    help="4족 조건에서 목표 z 를 nominal 대비 흔드는 폭 [m]. stance4_rate 가 낮으면 --stance4_z_bias 로 눌러붙인다.",
)
parser.add_argument(
    "--stance4_z_bias", type=float, default=0.0, help="4족 조건 목표 z 의 nominal 대비 평균 오프셋 [m] (음수=지면 쪽)."
)
parser.add_argument("--no_domain_rand", action="store_true", help="DR 을 끄고 평가 (기본은 학습과 동일하게 ON).")
parser.add_argument(
    "--no_obs_noise",
    action="store_true",
    help="물리 DR 은 두고 **관측 노이즈만** 끈다. 떨림의 원인이 obs 노이즈인지 분리할 때 쓴다.",
)
parser.add_argument(
    "--jvel_filter_alpha",
    type=float,
    default=None,
    help=(
        "관절 속도 관측 EMA 계수. **학습 때 쓴 값과 같아야 한다** — 다르면 분포 밖 평가다."
        " 미지정이면 env cfg 기본값(1.0=끔)을 쓴다."
    ),
)
parser.add_argument(
    "--zero_noise",
    type=str,
    default="",
    help=(
        "관측 노이즈를 채널별로 0 으로 만든다 (쉼표 구분: joint_pos, joint_vel, gravity, foot_pos, bias)."
        " 집계 3-way ablation 은 어느 채널이 떨림을 만드는지 못 가른다."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
RENDER_MODES = ("video", "showcase")
args_cli.enable_cameras = args_cli.mode in RENDER_MODES
args_cli.headless = True
if args_cli.mode in RENDER_MODES and not args_cli.video_folder:
    parser.error(f"--mode {args_cli.mode} 에는 --video_folder 가 필요합니다.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
from importlib import metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

TASK = "Go2-Pedipulation-v0"

# ── 시행 타임라인 (정책 50 Hz 기준 step) ─────────────────────────────────────
SETTLE_STEPS = 100  # 2.0 s — 발이 목표로 이동해 자리를 잡는 구간
PUSH_STEP = 125  # 2.5 s — 외력 인가 시점
PUSH_STEPS = 5  # 0.1 s — 문헌 인용 지속시간
PUSH_TRIAL_STEPS = 300  # 6.0 s — push 시행 총 길이 (인가 후 3.5 s 회복 관찰)
DRIFT_TRIAL_STEPS = 490  # 9.8 s — episode_length_s=10 의 타임아웃(499) 직전까지
DRIFT_CHECKPOINTS = (100, 200, 300, 400, 490)

# 힘 스윕 [N]. 0 은 대조군.
GATE_FORCES: tuple[float, ...] = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0)
CONDITIONS: tuple[str, ...] = ("stance4", "stance3")

# 성공 판정
SUCCESS_FOOT_ERR = 0.10  # [m] PLAN S1-G5: 발 오차 ≤ 0.1 m
FINAL_WINDOW = 25  # 마지막 0.5 s 평균으로 발 오차를 낸다

# ── hold 모드 (S1-G2 워크스페이스 커버리지 / S1-G3 hold drift) ────────────────
HOLD_WINDOW_STEPS = 250  # 5.0 s — PLAN S1-G3 이 명시한 hold 길이
HOLD_TRIAL_STEPS = SETTLE_STEPS + HOLD_WINDOW_STEPS  # 7.0 s (타임아웃 499 이내)
GRID_CELL_M = 0.10  # [m] PLAN S1-G2 의 격자 크기
G2_CELL_ERR = 0.08  # [m] 셀 평균 오차 기준
G2_CELL_FRAC = 0.90  # 기준을 만족해야 하는 셀 비율
G3_STD_M = 0.02  # [m] 5 s hold 오차 std 기준
G3_RANGE_M = 0.05  # [m] 5 s hold 오차 max−min 기준


def build_env(seed: int):
    """평가용 env 를 만든다. 학습 cfg 에서 **평가에 해로운 동역학만** 끈다."""
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.seed = seed

    # 명령이 시행 도중 재샘플되면 안 된다 (video 모드는 예외 — 여러 도달을 보여줘야 한다).
    # ⚠ min 만 키우면 (max-min) 이 음수가 되어 타이머가 즉시 만료된다. 반드시 둘 다 같은 값.
    hold_s = 3.0 if args_cli.mode == "video" else 1.0e6
    env_cfg.command.resample_time_min = hold_s
    env_cfg.command.resample_time_max = hold_s
    if args_cli.mode == "showcase":
        # 대본이 30 s 가까이 이어지므로 10 s 기본 에피소드로는 중간에 리셋된다.
        env_cfg.episode_length_s = 120.0
        # 궤적 모드를 켜 두고 정지 구간은 **반경 0** 으로 표현한다. 그러면 한 런에서 정지·원을
        # 함께 다룰 수 있다 (모드 전환에 env 재생성이 필요 없다).
        env_cfg.command.trajectory_mode = "circle"
        env_cfg.command.circle_randomize = False
    if args_cli.mode == "circle":
        # 1회전 = 2π/ω. ω=0.25 rad/s 면 25 s 라 10 s 에피소드로는 한 바퀴도 못 돈다.
        env_cfg.episode_length_s = 120.0
        env_cfg.command.trajectory_mode = "circle"
        env_cfg.command.circle_randomize = False  # 평가는 고정 반경·각속도
        env_cfg.command.circle_radius = args_cli.circle_radius
        env_cfg.command.circle_omega = args_cli.circle_omega
    # ⚠ 평가 중 커리큘럼이 승급하면 목표 분포가 런 도중 바뀐다. 승급 판정 자체를 봉쇄한다.
    env_cfg.command.curriculum_min_episodes = 10**9
    # 학습 커리큘럼이 도달한 최대 박스에서 평가한다.
    env_cfg.command.box_init = env_cfg.command.box_max

    if args_cli.jvel_filter_alpha is not None:
        env_cfg.jvel_filter_alpha = args_cli.jvel_filter_alpha
    if args_cli.no_domain_rand:
        env_cfg.domain_rand = False
    if args_cli.no_obs_noise:
        # 물리 DR(질량·마찰·게인·지연)은 그대로 두고 관측 노이즈만 끈다.
        env_cfg.dr.obs_noise = False
        env_cfg.dr.encoder_bias = False
    if args_cli.zero_noise:
        # 채널별 ablation. env 코드를 건드리지 않고 노이즈 크기만 0 으로 만든다.
        _ZERO = {
            "joint_pos": "joint_pos_noise",
            "joint_vel": "joint_vel_noise",
            "gravity": "gravity_noise",
            "foot_pos": "foot_pos_noise",
        }
        for name in (s.strip() for s in args_cli.zero_noise.split(",")):
            if not name:
                continue
            if name == "bias":
                env_cfg.dr.encoder_bias = False
            elif name in _ZERO:
                setattr(env_cfg.dr, _ZERO[name], 0.0)
            else:
                raise ValueError(f"--zero_noise 에 알 수 없는 채널: {name} (가능: {', '.join(_ZERO)}, bias)")
    # 외란은 이 스크립트가 통제한다 — env 내장 랜덤 push/발 외력은 항상 끈다.
    env_cfg.dr.push_robot = False
    env_cfg.dr.push_foot_force = False

    agent_cfg: RslRlOnPolicyRunnerCfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")

    if args_cli.mode in RENDER_MODES:
        # 기본 뷰(7.5, 7.5, 7.5)는 로봇이 수십 픽셀로 찍혀 자세 판단이 불가능하다.
        # 로봇은 제자리 task 라 env 원점 기준 고정 카메라로 충분하다 (표류 ~7 cm).
        env_cfg.viewer.origin_type = "env"
        env_cfg.viewer.env_index = 0
        # 눈높이를 낮게 두면 반대쪽 뒷발의 목표 마커가 몸통에 가린다. 3/4 부감으로 올려
        # 네 발이 모두 보이게 한다.
        env_cfg.viewer.eye = (1.35, 1.0, 0.92)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.18)
        if args_cli.mode == "showcase":
            # ⚠ 대본 길이로 잡아야 한다. 다른 대본의 길이를 쓰면 영상이 중간에 잘린다
            # (s2 대본 2020 step 을 s1 길이 1490 으로 녹화해 뒷부분이 통째로 날아갔다).
            _script = SHOWCASE_SCRIPT_S2 if args_cli.showcase_stage == "s2" else SHOWCASE_SCRIPT
            length = sum(s["steps"] for s in _script)
        else:
            length = args_cli.video_length
        env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args_cli.video_folder,
            step_trigger=lambda step: step == 0,
            video_length=length,
            disable_logger=True,
        )
    else:
        env = gym.make(TASK, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # ── 정책 로드 (play.py 의 OnPolicyRunner 경로와 동일) ────────────────────
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    from rsl_rl.algorithms.ppo import PPO as _VendoredStockPPO

    _accepted = set(inspect.signature(_VendoredStockPPO.__init__).parameters.keys()) - {"self"}
    _cfg_dict = agent_cfg.to_dict()
    _cfg_dict["algorithm"] = {k: v for k, v in _cfg_dict["algorithm"].items() if k in _accepted or k == "class_name"}
    runner = OnPolicyRunner(env, _cfg_dict, log_dir=None, device=args_cli.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    return env, policy


def assign_cells(n_env: int, device, generator: torch.Generator):
    """env 를 (다리 × 조건 × 힘) 셀에 균등 배정한다.

    셀을 라운드로빈으로 깔고 한 번 섞는다 — env 인덱스와 셀이 상관되면 (예: env 0 근처가
    항상 FL) DR 이나 초기 상태의 계통 편향이 특정 셀에 몰릴 수 있다.
    """
    n_leg, n_cond, n_force = 4, len(CONDITIONS), len(GATE_FORCES)
    n_cell = n_leg * n_cond * n_force
    cell = torch.arange(n_env, device=device) % n_cell
    cell = cell[torch.randperm(n_env, generator=generator, device=device)]
    leg_idx = cell // (n_cond * n_force)
    cond_idx = (cell // n_force) % n_cond
    force_idx = cell % n_force
    return leg_idx, cond_idx, force_idx


def build_targets(base_env, leg_idx, cond_idx, generator: torch.Generator):
    """조작 다리의 목표를 조건별로 만든다. 반환 shape [N, 4, 3] (base frame)."""
    device = base_env.device
    n = base_env.num_envs
    nominal = base_env._nominal_foot_pos_b.expand(n, -1, -1).clone()  # [N,4,3]
    box = torch.tensor(base_env.cfg.command.box_max, device=device)

    def u(lo, hi):
        return torch.rand(n, generator=generator, device=device) * (hi - lo) + lo

    # 3족: 학습 박스 안에서 뽑되 z 는 확실히 들어올리는 구간만 (z<0.10 이면 발이 안 뜬다).
    off3 = torch.stack([u(-box[0].item(), box[0].item()), u(-box[1].item(), box[1].item()), u(0.10, box[2].item())], -1)
    # 4족: nominal 근처 — 발이 지면에 남는다. z bias 로 지면 쪽으로 눌러붙일 수 있다.
    jit = args_cli.stance4_z_jitter
    off4 = torch.stack(
        [u(-0.05, 0.05), u(-0.05, 0.05), u(args_cli.stance4_z_bias - jit, args_cli.stance4_z_bias + jit)], -1
    )

    is3 = (cond_idx == CONDITIONS.index("stance3")).unsqueeze(-1).float()
    off = is3 * off3 + (1.0 - is3) * off4  # [N,3]

    target = nominal
    rows = torch.arange(n, device=device)
    target[rows, leg_idx] = nominal[rows, leg_idx] + off
    return target


def patch_command(base_env, leg_idx, target_b):
    """``_resample_command`` 를 고정 명령으로 대체한다.

    monkeypatch 로 명령 생성 자체를 고정하면, 조기 종료 후 자동 리셋된 env 도 **같은 셀의
    명령**을 다시 받는다 (그 env 는 집계에서 배제하지만, 명령이 재추첨되어 남은 시행의
    분포를 흔드는 일은 막아야 한다).
    """
    device = base_env.device
    n_leg = base_env.cfg.num_legs

    def fixed_resample(env_ids: torch.Tensor):
        if env_ids is None or int(env_ids.numel()) == 0:
            return
        ids = env_ids.to(torch.long)
        role = torch.ones(int(ids.numel()), n_leg, device=device)
        role[torch.arange(int(ids.numel()), device=device), leg_idx[ids]] = 0.0
        base_env._leg_role[ids] = role
        base_env._foot_target_b[ids] = target_b[ids]
        base_env._traj_center_b[ids] = target_b[ids]
        base_env._traj_phase[ids] = 0.0
        base_env._hold_counter[ids] = 0.0
        base_env._cmd_timer[ids] = 1.0e6

    base_env._resample_command = fixed_resample


def apply_push(base_env, force_mag: torch.Tensor, angle: torch.Tensor):
    """base 에 수평 외력을 인가한다 (world frame)."""
    n = base_env.num_envs
    forces = torch.zeros(n, 1, 3, device=base_env.device)
    forces[:, 0, 0] = force_mag * torch.cos(angle)
    forces[:, 0, 1] = force_mag * torch.sin(angle)
    body_ids = torch.tensor([base_env._base_body_id], dtype=torch.long, device=base_env.device)
    composer = base_env._robot.permanent_wrench_composer
    composer.reset()
    composer.add_forces_and_torques_index(forces=forces, body_ids=body_ids, is_global=True)


def clear_push(base_env):
    base_env._robot.permanent_wrench_composer.reset()


def foot_err_of_manip(base_env, leg_idx) -> torch.Tensor:
    """조작 다리의 목표-현재 거리 [m]. shape [N]."""
    pos = base_env._compute_foot_pos_b()
    rows = torch.arange(base_env.num_envs, device=base_env.device)
    return torch.norm(base_env._foot_target_b[rows, leg_idx] - pos[rows, leg_idx], dim=-1)


def stance_count(base_env) -> torch.Tensor:
    """접촉 중인 발 개수 [N]. 조건(4족/3족)이 실제로 성립했는지 검증하는 데 쓴다."""
    f = torch.norm(base_env.contact_sensor.data.net_forces_w[:, base_env._foot_sensor_ids], dim=-1)
    return (f > 1.0).float().sum(dim=-1)


def base_xy(base_env) -> torch.Tensor:
    return base_env._robot.data.body_pos_w[:, base_env._base_body_id, :2].clone()


def run_push_seed(env, policy, seed: int) -> dict:
    """push 시행 1 seed 를 돌리고 env 별 결과를 반환한다."""
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # 1) 최초 리셋으로 nominal 발 위치를 확정시킨다 (첫 관측에서 캡처된다).
    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패 — 목표 기준이 잡히지 않았다."

    # 2) 셀 배정 → 목표 생성 → 명령 고정 → 재리셋(고정 명령으로 관측까지 새로 만든다)
    leg_idx, cond_idx, force_idx = assign_cells(base_env.num_envs, device, gen)
    target_b = build_targets(base_env, leg_idx, cond_idx, gen)
    patch_command(base_env, leg_idx, target_b)
    obs, _ = env.reset()

    force_mag = torch.tensor(GATE_FORCES, device=device)[force_idx]
    angle = torch.rand(base_env.num_envs, generator=gen, device=device) * 2.0 * math.pi

    n = base_env.num_envs
    alive = torch.ones(n, dtype=torch.bool, device=device)
    stance4_hits = torch.zeros(n, device=device)
    stance3_hits = torch.zeros(n, device=device)
    settle_steps = torch.zeros(n, device=device)
    manip_dz = torch.zeros(n, device=device)  # 조작 발 z − nominal z [m]
    manip_force = torch.zeros(n, device=device)  # 조작 발 접촉력 [N]
    err_accum = torch.zeros(n, device=device)
    err_steps = torch.zeros(n, device=device)
    peak_speed = torch.zeros(n, device=device)  # push 인가 검증용

    for step in range(PUSH_TRIAL_STEPS):
        if step == PUSH_STEP:
            apply_push(base_env, force_mag, angle)
        elif step == PUSH_STEP + PUSH_STEPS:
            clear_push(base_env)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

        died = base_env._died.clone()
        # 시행 도중 한 번이라도 죽으면 그 env 는 이후 상태가 무의미하다 (자동 리셋됨).
        alive = alive & (~died)

        # 조건 검증: settle 구간의 지지 발 개수
        if SETTLE_STEPS // 2 <= step < SETTLE_STEPS:
            sc = stance_count(base_env)
            stance4_hits += (sc >= 4).float() * alive.float()
            stance3_hits += (sc == 3).float() * alive.float()
            settle_steps += alive.float()
            # 진단: "발이 떠 있다"와 "닿았지만 하중이 없다"는 전혀 다른 상태다.
            rows = torch.arange(n, device=device)
            fp = base_env._compute_foot_pos_b()[rows, leg_idx, 2]
            manip_dz += (fp - base_env._nominal_foot_pos_b[0, leg_idx, 2]) * alive.float()
            cf = torch.norm(base_env.contact_sensor.data.net_forces_w[:, base_env._foot_sensor_ids], dim=-1)
            manip_force += cf[rows, leg_idx] * alive.float()

        # push 인가 검증: 인가 직후 base 속도 피크
        if PUSH_STEP <= step < PUSH_STEP + 3 * PUSH_STEPS:
            spd = torch.norm(base_env._robot.data.root_lin_vel_w[:, :2], dim=-1)
            peak_speed = torch.maximum(peak_speed, spd * alive.float())

        # 최종 발 오차: 마지막 0.5 s 평균
        if step >= PUSH_TRIAL_STEPS - FINAL_WINDOW:
            err_accum += foot_err_of_manip(base_env, leg_idx) * alive.float()
            err_steps += alive.float()

        del dones

    final_err = err_accum / err_steps.clamp(min=1.0)
    success = alive & (final_err <= SUCCESS_FOOT_ERR)
    return {
        "leg": leg_idx.cpu(),
        "cond": cond_idx.cpu(),
        "force": force_mag.cpu(),
        "alive": alive.cpu(),
        "final_err": final_err.cpu(),
        "success": success.cpu(),
        "stance4_rate": (stance4_hits / settle_steps.clamp(min=1.0)).cpu(),
        "stance3_rate": (stance3_hits / settle_steps.clamp(min=1.0)).cpu(),
        "peak_speed": peak_speed.cpu(),
        "manip_dz": (manip_dz / settle_steps.clamp(min=1.0)).cpu(),
        "manip_force": (manip_force / settle_steps.clamp(min=1.0)).cpu(),
    }


def run_drift_seed(env, policy, seed: int) -> dict:
    """외란 없는 3족 hold 에서 base 표류를 측정한다 (S1-G7)."""
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패."

    n = base_env.num_envs
    leg_idx = torch.arange(n, device=device) % 4
    leg_idx = leg_idx[torch.randperm(n, generator=gen, device=device)]
    cond_idx = torch.full((n,), CONDITIONS.index("stance3"), dtype=torch.long, device=device)
    target_b = build_targets(base_env, leg_idx, cond_idx, gen)
    patch_command(base_env, leg_idx, target_b)
    obs, _ = env.reset()

    alive = torch.ones(n, dtype=torch.bool, device=device)
    xy0 = base_xy(base_env)
    xy_prev = xy0.clone()
    path_len = torch.zeros(n, device=device)
    net_at = {}
    path_at = {}
    # settle 이후 기준점 — 다리를 들면서 생기는 **일회성 counterbalance 오프셋**과 그 뒤로도
    # 계속 밀려나는 **진짜 표류**를 분리한다. 중앙값 추이로 추론하지 않고 env 별로 직접 잰다.
    xy_ref = xy0.clone()
    path_ref = torch.zeros(n, device=device)
    net_late_at = {}
    path_late_at = {}
    err_accum = torch.zeros(n, device=device)
    err_steps = torch.zeros(n, device=device)

    for step in range(DRIFT_TRIAL_STEPS):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        alive = alive & (~base_env._died.clone())

        xy = base_xy(base_env)
        path_len += torch.norm(xy - xy_prev, dim=-1) * alive.float()
        xy_prev = xy

        if step == SETTLE_STEPS - 1:
            xy_ref = xy.clone()
            path_ref = path_len.clone()

        if step >= SETTLE_STEPS:
            err_accum += foot_err_of_manip(base_env, leg_idx) * alive.float()
            err_steps += alive.float()

        s = step + 1
        if s in DRIFT_CHECKPOINTS:
            net_at[s] = torch.norm(xy - xy0, dim=-1).cpu().clone()
            path_at[s] = path_len.cpu().clone()
            net_late_at[s] = torch.norm(xy - xy_ref, dim=-1).cpu().clone()
            path_late_at[s] = (path_len - path_ref).cpu().clone()

        del dones

    return {
        "leg": leg_idx.cpu(),
        "alive": alive.cpu(),
        "net": {k: v for k, v in net_at.items()},
        "path": {k: v for k, v in path_at.items()},
        "net_late": {k: v for k, v in net_late_at.items()},
        "path_late": {k: v for k, v in path_late_at.items()},
        "hold_err": (err_accum / err_steps.clamp(min=1.0)).cpu(),
        "com_b": base_env._com_pos_b().mean(dim=0).cpu(),
    }


def grid_shape(base_env) -> tuple[int, int, int]:
    """명령 박스를 ``GRID_CELL_M`` 격자로 나눴을 때의 셀 개수 (x, y, z).

    x·y 는 ±box, z 는 0~box 이므로 축마다 범위가 다르다. 마지막 셀은 잘릴 수 있다.
    """
    bx, by, bz = base_env.cfg.command.box_max
    return (
        max(1, math.ceil(2.0 * bx / GRID_CELL_M)),
        max(1, math.ceil(2.0 * by / GRID_CELL_M)),
        max(1, math.ceil(bz / GRID_CELL_M)),
    )


def build_stratified_targets(base_env, leg_idx, cell_idx, generator: torch.Generator):
    """각 env 를 배정된 격자 셀 **안에서** 균등 샘플한 목표로 만든다.

    커버리지 게이트는 셀마다 표본이 있어야 성립한다. 목표를 박스 전체에서 무작위로 뽑으면
    모서리 셀(부피가 잘린 셀)의 표본이 체계적으로 적어져 셀 평균이 불안정해진다.
    """
    device = base_env.device
    n = base_env.num_envs
    bx, by, bz = base_env.cfg.command.box_max
    nx, ny, nz = grid_shape(base_env)

    xi = cell_idx // (ny * nz)
    yi = (cell_idx // nz) % ny
    zi = cell_idx % nz

    def within(i, lo0, hi_all, count):
        lo = lo0 + i.float() * GRID_CELL_M
        hi = torch.minimum(lo + GRID_CELL_M, torch.full_like(lo, hi_all))
        del count
        return lo + torch.rand(n, generator=generator, device=device) * (hi - lo)

    off = torch.stack(
        [within(xi, -bx, bx, nx), within(yi, -by, by, ny), within(zi, 0.0, bz, nz)],
        dim=-1,
    )

    nominal = base_env._nominal_foot_pos_b.expand(n, -1, -1).clone()
    rows = torch.arange(n, device=device)
    nominal[rows, leg_idx] = nominal[rows, leg_idx] + off
    return nominal


def run_hold_seed(env, policy, seed: int) -> dict:
    """정적 목표를 5 s 유지하며 S1-G2(커버리지)와 S1-G3(hold drift)를 동시에 잰다.

    두 게이트 모두 "목표를 잡고 유지" 조건에서 정의되므로 한 시행에서 함께 측정한다.
    오차는 관측 노이즈가 실린 obs 가 아니라 **실제 발 위치**로 계산한다.
    """
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패."

    n = base_env.num_envs
    nx, ny, nz = grid_shape(base_env)
    n_cell = nx * ny * nz
    n_slot = 4 * n_cell
    slot = torch.arange(n, device=device) % n_slot
    slot = slot[torch.randperm(n, generator=gen, device=device)]
    leg_idx = slot // n_cell
    cell_idx = slot % n_cell

    target_b = build_stratified_targets(base_env, leg_idx, cell_idx, gen)
    patch_command(base_env, leg_idx, target_b)
    obs, _ = env.reset()

    alive = torch.ones(n, dtype=torch.bool, device=device)
    err_sum = torch.zeros(n, device=device)
    err_sq = torch.zeros(n, device=device)
    err_min = torch.full((n,), float("inf"), device=device)
    err_max = torch.zeros(n, device=device)
    steps = torch.zeros(n, device=device)
    # 떨림(tremor) 지표 — 유지 중인데도 관절 목표가 매 step 얼마나 흔들리는지.
    jitter = torch.zeros(n, device=device)  # mean_j |Δ joint target| [rad/step]
    jvel_sq = torch.zeros(n, device=device)  # mean_j joint_vel² [(rad/s)²]
    # ⚠ 위 jitter 는 12 관절 평균이라 **구조가 다른 두 경로를 섞는다.** 조작 다리는 적분형
    #   (man_target += clamp(a·scale)) 이고 지지 다리는 절대 맵이다. 두 경로의 주파수 응답이
    #   달라(적분기는 고주파를 1/f 로 감쇠) 어느 쪽이 떨리는지가 곧 어디를 고쳐야 하는지다.
    jit_manip = torch.zeros(n, device=device)
    jit_supp = torch.zeros(n, device=device)
    # 고주파 chatter 와 저주파 표류 판별: Δtarget 의 부호 반전율.
    #   ~1.0 = 매 step 방향 반전(Nyquist chatter) · ~0.5 = 백색 · ~0.0 = 단조 표류
    flip = torch.zeros(n, device=device)
    flip_steps = torch.zeros(n, device=device)
    # 조작 다리의 3 관절 마스크 [N,12] — leg_idx 는 이 시행에서 조작하는 다리다.
    manip_j = (base_env._leg_of_joint.unsqueeze(0) == leg_idx.unsqueeze(-1)).float()
    supp_j = 1.0 - manip_j
    prev_target = base_env._processed_actions.clone()
    prev_delta: torch.Tensor | None = None
    # 원시 action 궤적 — 랜덤워크 판별용. obs 에 prev_actions(28) 가 들어 있어
    # a_t = π(…, a_{t-1}, …) 의 되먹임 이득이 1 에 가까우면 백색 노이즈가 적분된다.
    act_hist: list[torch.Tensor] = []

    for step in range(HOLD_TRIAL_STEPS):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        alive = alive & (~base_env._died.clone())

        d = base_env._processed_actions - prev_target
        if step >= SETTLE_STEPS:
            m = alive.float()
            jitter += d.abs().mean(dim=-1) * m
            jit_manip += (d.abs() * manip_j).sum(dim=-1) / 3.0 * m
            jit_supp += (d.abs() * supp_j).sum(dim=-1) / 9.0 * m
            jvel_sq += (base_env._robot.data.joint_vel.torch**2).mean(dim=-1) * m
            if prev_delta is not None:
                # 부호가 뒤집힌 관절 비율. 진폭이 아주 작은 관절은 부호가 의미 없으므로 제외.
                live = (d.abs() > 1e-6) & (prev_delta.abs() > 1e-6)
                flipped = ((d * prev_delta) < 0.0) & live
                cnt = live.sum(dim=-1).clamp(min=1)
                flip += flipped.sum(dim=-1).float() / cnt.float() * m
                flip_steps += m
            act_hist.append(base_env._actions[:, :24].clone())
        prev_delta = d
        prev_target = base_env._processed_actions.clone()

        if step >= SETTLE_STEPS:
            e = foot_err_of_manip(base_env, leg_idx)
            m = alive.float()
            err_sum += e * m
            err_sq += (e**2) * m
            err_min = torch.where(alive, torch.minimum(err_min, e), err_min)
            err_max = torch.where(alive, torch.maximum(err_max, e), err_max)
            steps += m

    # ── 랜덤워크 판별 ────────────────────────────────────────────
    # MSD(k) = E[(a_{t+k} − a_t)²] 를 lag 로 나눈 값. 랜덤워크면 lag 에 무관하게 일정(선형 증가),
    # 평균회귀(OU)면 lag 가 커질수록 감소해 포화한다.
    a_seq = torch.stack(act_hist, dim=1)  # [N,T,24]
    a_c = a_seq - a_seq.mean(dim=1, keepdim=True)
    lag1 = (a_c[:, 1:] * a_c[:, :-1]).mean(dim=(1, 2)) / a_c.pow(2).mean(dim=(1, 2)).clamp(min=1e-12)
    msd = {}
    for lag in (1, 2, 4, 8, 16, 32, 64):
        if lag < a_seq.shape[1]:
            msd[lag] = ((a_seq[:, lag:] - a_seq[:, :-lag]).pow(2).mean(dim=(1, 2)) / lag).cpu()

    k = steps.clamp(min=1.0)
    mean_err = err_sum / k
    var = (err_sq / k - mean_err**2).clamp(min=0.0)
    return {
        "leg": leg_idx.cpu(),
        "cell": cell_idx.cpu(),
        "alive": alive.cpu(),
        "mean_err": mean_err.cpu(),
        "std_err": torch.sqrt(var).cpu(),
        "range_err": (err_max - torch.where(torch.isinf(err_min), err_max, err_min)).cpu(),
        "grid": (nx, ny, nz),
        "box": tuple(float(v) for v in base_env.cfg.command.box_max),
        "jitter": (jitter / k).cpu(),
        "jitter_manip": (jit_manip / k).cpu(),
        "jitter_supp": (jit_supp / k).cpu(),
        "flip_rate": (flip / flip_steps.clamp(min=1.0)).cpu(),
        "jvel_rms": torch.sqrt(jvel_sq / k).cpu(),
        "act_lag1": lag1.cpu(),
        "act_msd": msd,
    }


# ── showcase 시나리오 ────────────────────────────────────────────────────────
# 학습 목적을 그대로 보여주는 대본. (지속 step, 다리, nominal 대비 오프셋 [m], 자막, push)
# 오프셋은 전부 학습 박스(±0.20, ±0.14, 0~0.26) 안이다 — 분포 밖 동작을 시연하지 않는다.
# ⚠ 다리 교체 구간은 **네 다리에 같은 오프셋**을 준다. 다리마다 다른 목표를 주면 난이도가
#   섞여 "어느 다리든 같은 정책이 수행한다"는 요점이 흐려진다 (실제로 RL 에만 극단적인 목표를
#   줬다가 그 구간만 오차가 5 cm 로 나왔다 — 정책이 아니라 대본 탓이었다).
SWITCH_OFF: tuple[float, float, float] = (0.08, 0.00, 0.18)
SHOWCASE_SCRIPT: tuple[dict, ...] = (
    {"steps": 100, "leg": "FL", "off": (0.0, 0.0, 0.0), "label": "Standing on four legs", "marker": False},
    {"steps": 200, "leg": "FL", "off": SWITCH_OFF, "label": "COMMAND: front-left foot -> target"},
    {"steps": 150, "leg": "FR", "off": SWITCH_OFF, "label": "SAME COMMAND, front-right foot"},
    {"steps": 150, "leg": "RL", "off": SWITCH_OFF, "label": "SAME COMMAND, rear-left foot"},
    {"steps": 150, "leg": "RR", "off": SWITCH_OFF, "label": "SAME COMMAND, rear-right foot"},
    {"steps": 150, "leg": "FL", "off": (-0.12, 0.06, 0.22), "label": "Targets across the workspace"},
    {"steps": 150, "leg": "FL", "off": (0.16, -0.08, 0.10), "label": "Targets across the workspace"},
    {
        # ⚠ 마지막 push 이후 회복까지 담아야 한다. base frame 기준 오차라 push 직후가 아니라
        #   자세를 되잡는 ~1 s 뒤에 정점이 오고, 복귀에 다시 ~1 s 가 걸린다. 꼬리가 짧으면
        #   영상이 회복 도중에 끊겨 실패처럼 보인다.
        "steps": 440,
        "leg": "FL",
        "off": (0.08, 0.00, 0.20),
        "label": "HOLDING under 100 / 150 / 200 N pushes",
        "pushes": ((110, 100.0), (200, 150.0), (290, 200.0)),
    },
)
SHOWCASE_TITLE = "Go2 Pedipulation - move a commanded foot to a commanded position and hold it"

# ── S2 대본: 연속 궤적 추종 ──────────────────────────────────────────────────
# 반경 0 은 정지 목표와 같다(원이 점으로 축퇴). 그래서 정지 구간과 원 구간을 한 런에서
# `trajectory_mode="circle"` 하나로 다룰 수 있다.
# 원은 base frame x-z 평면에 있으므로 **측면(y 축 방향) 시점**이 아니면 원으로 안 보인다.
_S2_FL_CENTER = (0.08, 0.00, 0.18)
_S2_RR_CENTER = (0.08, 0.00, 0.18)
_CAM_FL_SIDE = (0.26, 1.35, 0.34)
_LOOK_FL_SIDE = (0.26, 0.17, 0.20)
_CAM_RR_SIDE = (-0.19, -1.35, 0.34)
_LOOK_RR_SIDE = (-0.19, -0.17, 0.21)
SHOWCASE_SCRIPT_S2: tuple[dict, ...] = (
    {"steps": 100, "leg": "FL", "off": (0.0, 0.0, 0.0), "label": "Standing on four legs", "marker": False},
    {"steps": 160, "leg": "FL", "off": _S2_FL_CENTER, "label": "S1: reach a fixed target and hold"},
    {
        "steps": 660,
        "leg": "FL",
        "off": _S2_FL_CENTER,
        "radius": 0.10,
        "omega": 0.5,
        "label": "S2: trace a circle  (r 0.10 m, 0.5 rad/s)",
        "eye": _CAM_FL_SIDE,
        "lookat": _LOOK_FL_SIDE,
    },
    {
        "steps": 340,
        "leg": "FL",
        "off": _S2_FL_CENTER,
        "radius": 0.10,
        "omega": 1.0,
        "label": "Same circle at twice the rate  (1.0 rad/s)",
        "eye": _CAM_FL_SIDE,
        "lookat": _LOOK_FL_SIDE,
    },
    {
        "steps": 340,
        "leg": "RR",
        "off": _S2_RR_CENTER,
        "radius": 0.10,
        "omega": 1.0,
        "label": "Same command on the rear-right foot",
        "eye": _CAM_RR_SIDE,
        "lookat": _LOOK_RR_SIDE,
    },
    {
        "steps": 420,
        "leg": "FL",
        "off": _S2_FL_CENTER,
        "radius": 0.10,
        "omega": 0.5,
        "label": "Tracking under 100 / 150 N pushes",
        "pushes": ((120, 100.0), (260, 150.0)),
        "eye": _CAM_FL_SIDE,
        "lookat": _LOOK_FL_SIDE,
    },
)
SHOWCASE_TITLE_S2 = "Go2 Pedipulation - follow a continuous foot trajectory on a commanded leg"
PATH_MARKER_N = 48  # 명령된 원 경로를 표시할 정적 마커 개수

# 조작 다리별 카메라 위치 (env 원점 기준). 고정 카메라로는 반대쪽 뒷발의 목표 마커가 몸통에
# 완전히 가려 "발이 목표에 갔다"를 볼 수 없다. 활성 다리 쪽으로 돌린다.
SHOWCASE_CAM: dict[str, tuple[float, float, float]] = {
    "FL": (1.35, 1.00, 0.92),
    "FR": (1.35, -1.00, 0.92),
    "RL": (-1.25, 1.10, 0.92),
    "RR": (-1.25, -1.10, 0.92),
}
SHOWCASE_LOOKAT: tuple[float, float, float] = (0.0, 0.0, 0.18)


def set_camera(base_env, eye, lookat) -> bool:
    """env 0 원점 기준으로 뷰포트 카메라를 옮긴다. 성공 여부를 반환한다.

    ``viewport_camera_controller`` 는 GUI/visualizer 가 없는 headless 에서 ``None`` 이므로
    (``direct_rl_env.py:168-172``) 쓸 수 없다. 그 컨트롤러가 내부에서 호출하는 두 경로를
    직접 부른다 — rgb_array 렌더는 Kit 렌더러 카메라를 쓰므로 두 번째 호출이 실제로 화면을
    바꾼다.
    """
    origin = base_env.scene.env_origins[0].detach().cpu().numpy()
    e = origin + np.asarray(eye, dtype=float)
    t = origin + np.asarray(lookat, dtype=float)
    base_env.sim.set_camera_view(eye=tuple(float(v) for v in e), target=tuple(float(v) for v in t))
    try:
        from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

        set_kit_renderer_camera_view(eye=e, target=t, camera_prim_path=base_env.cfg.viewer.cam_prim_path)
        return True
    except (ImportError, AttributeError):
        return False


def run_showcase(env, policy, seed: int) -> dict:
    """학습 목적이 드러나는 시연 영상.

    게이트 측정과 달리 통계가 아니라 **가독성**이 목적이다. 목표를 구 마커로 띄우고,
    발이 그 안으로 들어가 머무는 것을 보인다. 다리를 바꿔 명령해도 같은 정책이 수행한다는
    점(ALaM leg-role 설계)과 외란 중에도 유지된다는 점을 한 영상에 담는다.

    ``seed`` 는 env 구성에만 쓰이고 여기서는 참조하지 않는다 — 시나리오가 전부 대본이라
    무작위 추출이 없다.
    """
    del seed
    import isaaclab.sim as sim_utils
    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import SPHERE_MARKER_CFG
    from isaaclab.utils.math import quat_apply

    base_env = env.unwrapped
    device = base_env.device
    leg_names = ("FL", "FR", "RL", "RR")

    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패."

    n = base_env.num_envs
    n_leg = base_env.cfg.num_legs
    state = {"leg": 0, "off": (0.0, 0.0, 0.0), "radius": 0.0, "omega": 0.0}

    def scripted_resample(env_ids: torch.Tensor):
        """대본이 지정한 다리·목표·궤적 파라미터를 그대로 쓴다 (무작위 재샘플 금지)."""
        if env_ids is None or int(env_ids.numel()) == 0:
            return
        ids = env_ids.to(torch.long)
        nominal = base_env._nominal_foot_pos_b.expand(n, -1, -1).clone()
        off = torch.tensor(state["off"], device=device)
        rows = torch.arange(n, device=device)
        nominal[rows, state["leg"]] = nominal[rows, state["leg"]] + off
        role = torch.ones(int(ids.numel()), n_leg, device=device)
        role[:, state["leg"]] = 0.0
        base_env._leg_role[ids] = role
        base_env._traj_center_b[ids] = nominal[ids]
        # 반경 0 이면 목표 = 중심 (정지 목표와 동일).
        base_env._traj_radius[ids] = state["radius"]
        base_env._traj_omega[ids] = state["omega"]
        base_env._traj_dir[ids] = 1.0
        base_env._traj_phase[ids] = 0.0
        tgt = nominal[ids].clone()
        tgt[..., 0] += state["radius"]  # phase 0 → +x 방향
        base_env._foot_target_b[ids] = tgt
        base_env._hold_counter[ids] = 0.0
        base_env._cmd_timer[ids] = 1.0e6

    base_env._resample_command = scripted_resample

    # 목표 마커 — 반지름은 hold 판정 임계(hold_delta_ee)의 대략 절반으로 두어, 발이 마커에
    # 닿는 것이 곧 "임계 안"임을 눈으로 알 수 있게 한다.
    marker_cfg = SPHERE_MARKER_CFG.copy()
    marker_cfg.prim_path = "/Visuals/Pedipulation/target"
    marker_cfg.markers["sphere"].radius = 0.035
    marker_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.85, 0.25))
    target_marker = VisualizationMarkers(marker_cfg)
    hidden = torch.tensor([[0.0, 0.0, -10.0]], device=device)

    # 명령된 원 **경로**를 작은 점으로 깔아 둔다. 이게 없으면 발이 원을 그리는지, 그냥 아무렇게
    # 움직이는지 영상만으로는 구분되지 않는다.
    path_cfg = SPHERE_MARKER_CFG.copy()
    path_cfg.prim_path = "/Visuals/Pedipulation/path"
    path_cfg.markers["sphere"].radius = 0.008
    path_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.15))
    path_marker = VisualizationMarkers(path_cfg)
    path_hidden = torch.tensor([[0.0, 0.0, -10.0]], device=device).expand(PATH_MARKER_N, -1)
    path_ang = torch.linspace(0.0, 2.0 * math.pi, PATH_MARKER_N + 1, device=device)[:-1]

    obs, _ = env.reset()

    script = SHOWCASE_SCRIPT_S2 if args_cli.showcase_stage == "s2" else SHOWCASE_SCRIPT
    title = SHOWCASE_TITLE_S2 if args_cli.showcase_stage == "s2" else SHOWCASE_TITLE

    timeline: list[dict] = []
    all_ids = torch.arange(n, device=device)
    step_global = 0
    cam = np.asarray(script[0].get("eye", SHOWCASE_CAM[script[0]["leg"]]), dtype=float)
    for seg in script:
        cam_to = np.asarray(seg.get("eye", SHOWCASE_CAM[seg["leg"]]), dtype=float)
        look_to = seg.get("lookat", SHOWCASE_LOOKAT)
        state["radius"] = float(seg.get("radius", 0.0))
        state["omega"] = float(seg.get("omega", 0.0))
        state["leg"] = leg_names.index(seg["leg"])
        state["off"] = seg["off"]
        # 새 목표를 즉시 적용 (에피소드 리셋 없이). ⚠ `_hold_counter` 는 `_get_rewards` 에서
        # inference mode 안에서 재할당되므로 inference tensor 다 — 밖에서 쓰면 RuntimeError.
        with torch.inference_mode():
            scripted_resample(all_ids)
        pushes = dict(seg.get("pushes", ()))
        show_marker = seg.get("marker", True)

        # ⚠ 카메라는 **구간 시작에 한 번만** 옮긴다. 매 step 옮기면 RTX 렌더러가 따라오지
        #   못해 프레임이 백지로 나온다 (매 step 호출 시 1490 중 257 프레임이 날아갔고, 최장
        #   165 프레임이 연속으로 비었다). 자막도 같은 시점에 바뀌므로 컷 전환으로 읽힌다.
        cam = cam_to
        if not set_camera(base_env, cam, look_to) and step_global == 0:
            print("[WARN] Kit 렌더러 카메라를 옮길 수 없다 — 고정 시점으로 촬영된다.")

        for local in range(seg["steps"]):
            if local in pushes:
                mag = torch.full((n,), pushes[local], device=device)
                ang = torch.full((n,), math.pi * 0.25, device=device)
                apply_push(base_env, mag, ang)
            elif (local - PUSH_STEPS) in pushes:
                clear_push(base_env)

            with torch.inference_mode():
                obs, _, _, _ = env.step(policy(obs))

            root_pos_w, root_quat_w = base_env._base_pose()
            tgt_b = base_env._foot_target_b[all_ids, state["leg"]]
            tgt_w = root_pos_w + quat_apply(root_quat_w, tgt_b)
            target_marker.visualize(translations=tgt_w if show_marker else hidden.expand(n, -1))

            # 원 경로 마커 — 반경 0(정지 구간)이면 숨긴다.
            if show_marker and state["radius"] > 1e-6:
                ctr_b = base_env._traj_center_b[0, state["leg"]]
                ring_b = ctr_b.unsqueeze(0).repeat(PATH_MARKER_N, 1)
                ring_b[:, 0] += state["radius"] * torch.cos(path_ang)
                ring_b[:, 2] += state["radius"] * torch.sin(path_ang)
                ring_w = root_pos_w[0] + quat_apply(root_quat_w[0].expand(PATH_MARKER_N, -1), ring_b)
                path_marker.visualize(translations=ring_w)
            else:
                path_marker.visualize(translations=path_hidden)

            err = float(foot_err_of_manip(base_env, torch.full((n,), state["leg"], device=device))[0].item())
            active_push = 0.0
            for s, m in pushes.items():
                if s <= local < s + PUSH_STEPS:
                    active_push = m
            timeline.append(
                {
                    "step": step_global,
                    "label": seg["label"],
                    "leg": seg["leg"],
                    "err_m": err,
                    "push_N": active_push,
                    "marker": show_marker,
                }
            )
            step_global += 1

    print(f"\n시연 {step_global} step ({step_global / 50.0:.1f} s) 기록 완료. (대본 {args_cli.showcase_stage})")
    return {"timeline": timeline, "total_steps": step_global, "title": title}


def annotate_video(src: str, dst: str, timeline: list[dict], title: str):
    """렌더된 mp4 에 제목·현재 단계·추종 오차를 얹는다.

    자막이 없으면 "개가 다리를 흔든다"와 "명령받은 점으로 발을 옮긴다"가 구분되지 않는다.
    한글 폰트가 시스템에 없어 영문으로 쓴다.
    """
    # imageio v2 API 를 쓴다 — 이 환경에는 `pyav` 플러그인이 없고 imageio-ffmpeg 만 있다.
    import imageio
    from PIL import Image, ImageDraw, ImageFont

    def load_font(size: int):
        for path in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ):
            if os.path.isfile(path):
                return ImageFont.truetype(path, size)
        return ImageFont.load_default()

    f_title, f_label, f_num = load_font(22), load_font(30), load_font(26)
    reader = imageio.get_reader(src)
    fps = float(reader.get_meta_data().get("fps", 50.0))
    writer = imageio.get_writer(dst, fps=fps, codec="libx264", quality=8, macro_block_size=1)

    count = 0
    for i, frame in enumerate(reader):
        rec = timeline[min(i, len(timeline) - 1)]
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img, "RGBA")
        w, h = img.size

        d.rectangle([0, 0, w, 46], fill=(0, 0, 0, 165))
        d.text((16, 12), title, font=f_title, fill=(255, 255, 255, 255))

        d.rectangle([0, h - 78, w, h], fill=(0, 0, 0, 165))
        d.text((16, h - 70), rec["label"], font=f_label, fill=(120, 235, 140, 255))

        if rec["marker"]:
            err_cm = rec["err_m"] * 100.0
            col = (120, 235, 140, 255) if err_cm <= 5.0 else (245, 200, 90, 255)
            d.text((16, h - 34), f"foot-to-target error: {err_cm:5.1f} cm", font=f_num, fill=col)
        if rec["push_N"] > 0.0:
            d.text((w - 250, h - 34), f"PUSH {rec['push_N']:.0f} N", font=f_num, fill=(255, 110, 110, 255))

        writer.append_data(np.asarray(img))
        count += 1

    reader.close()
    writer.close()
    print(f"자막 입힌 영상: {dst}  ({count} frame @ {fps:.0f} fps)")


def run_circle_seed(env, policy, seed: int) -> dict:
    """원 궤적 추종 — S2-G1(RMSE) / G2(최대편차) / G4(위상지연) / G5(완주율).

    한 바퀴를 온전히 도는 구간에서 잰다. 명령은 env 가 생성하되(``trajectory_mode="circle"``),
    조작 다리만 고정 배정해 다리별 분해가 가능하게 한다.
    """
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패."

    n = base_env.num_envs
    n_leg = base_env.cfg.num_legs
    leg_idx = (torch.arange(n, device=device) % 4)[torch.randperm(n, generator=gen, device=device)]

    # 명령 생성은 env 의 circle 경로를 그대로 쓰고, **조작 다리만** 고정한다.
    orig_resample = base_env._resample_command

    def leg_fixed_resample(env_ids: torch.Tensor):
        orig_resample(env_ids)
        if env_ids is None or int(env_ids.numel()) == 0:
            return
        ids = env_ids.to(torch.long)
        role = torch.ones(int(ids.numel()), n_leg, device=device)
        role[torch.arange(int(ids.numel()), device=device), leg_idx[ids]] = 0.0
        base_env._leg_role[ids] = role
        base_env._cmd_timer[ids] = 1.0e6

    base_env._resample_command = leg_fixed_resample
    obs, _ = env.reset()

    omega = float(args_cli.circle_omega)
    rev_steps = int(math.ceil(2.0 * math.pi / omega / base_env.step_dt))
    total = SETTLE_STEPS + rev_steps
    assert total < base_env.max_episode_length - 1, f"에피소드가 짧다: {total} vs {base_env.max_episode_length}"

    alive = torch.ones(n, dtype=torch.bool, device=device)
    err_sq = torch.zeros(n, device=device)
    err_max = torch.zeros(n, device=device)
    cnt = torch.zeros(n, device=device)
    # 위상지연용 — 목표와 실제의 x 성분 시계열(중심 제거)을 모은다.
    rad_hist: list[torch.Tensor] = []
    tx_hist: list[torch.Tensor] = []
    fx_hist: list[torch.Tensor] = []
    rows = torch.arange(n, device=device)

    for step in range(total):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        alive = alive & (~base_env._died.clone())
        if step < SETTLE_STEPS:
            continue
        m = alive.float()
        e = foot_err_of_manip(base_env, leg_idx)
        err_sq += (e**2) * m
        err_max = torch.where(alive, torch.maximum(err_max, e), err_max)
        cnt += m
        tgt = base_env._foot_target_b[rows, leg_idx]
        cur = base_env._compute_foot_pos_b()[rows, leg_idx]
        ctr = base_env._traj_center_b[rows, leg_idx]
        tx_hist.append((tgt - ctr)[:, 0].clone())
        fx_hist.append((cur - ctr)[:, 0].clone())
        # 반경 방향 편향 — 중심에서의 거리. 진폭을 (max−min)/2 로 재면 **극값 통계**라
        # 떨림에 부풀려진다(발 std 3.9 mm 면 진폭이 ~8 mm 커진다). 평균 반경거리는 그에 강건해서
        # "원 바깥으로 도는가"를 판정할 수 있는 지표다.
        rel = cur - ctr
        rad_hist.append(torch.norm(rel[:, [0, 2]], dim=-1))

    rmse = torch.sqrt(err_sq / cnt.clamp(min=1.0))
    tx = torch.stack(tx_hist, dim=1)  # [n, T]
    fx = torch.stack(fx_hist, dim=1)
    lag = _phase_lag_steps(tx, fx, max_lag=int(0.4 / base_env.step_dt))
    # 조건 검증: 목표가 실제로 반경만큼 돌았는가. 이게 없으면 "궤적 추종"이 아니라
    # 정지 목표 유지를 재고 있을 수 있다 (trajectory_mode 가 반영되지 않은 경우).
    tgt_amp = 0.5 * (tx.max(dim=1).values - tx.min(dim=1).values)
    foot_amp = 0.5 * (fx.max(dim=1).values - fx.min(dim=1).values)
    radial_mean = torch.stack(rad_hist, dim=1).mean(dim=1)

    return {
        "leg": leg_idx.cpu(),
        "alive": alive.cpu(),
        "rmse": rmse.cpu(),
        "max_dev": err_max.cpu(),
        "lag_s": (lag.float() * base_env.step_dt).cpu(),
        "omega": omega,
        "radius": float(args_cli.circle_radius),
        "rev_steps": rev_steps,
        "tgt_amp": tgt_amp.cpu(),
        "foot_amp": foot_amp.cpu(),
        "radial_mean": radial_mean.cpu(),
    }


def _phase_lag_steps(target: torch.Tensor, actual: torch.Tensor, max_lag: int) -> torch.Tensor:
    """실제가 목표보다 몇 step 뒤처지는지 — 상호상관 최대점. shape [n].

    위치 오차만 보면 지연이 숨는다(원 위를 일정하게 뒤따라가면 오차는 작게 유지된다).
    그래서 PLAN §5 가 S2-G4 를 따로 둔다.
    """
    t = target - target.mean(dim=1, keepdim=True)
    a = actual - actual.mean(dim=1, keepdim=True)
    t = t / t.norm(dim=1, keepdim=True).clamp(min=1e-8)
    a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-8)
    best = torch.zeros(t.shape[0], device=t.device)
    best_lag = torch.zeros(t.shape[0], dtype=torch.long, device=t.device)
    T = t.shape[1]
    for k in range(0, max_lag + 1):
        # actual(t) 를 k 만큼 앞당겨 target 과 겹친다 → k 가 지연량
        c = (t[:, : T - k] * a[:, k:]).sum(dim=1)
        upd = c > best
        best = torch.where(upd, c, best)
        best_lag = torch.where(upd, torch.full_like(best_lag, k), best_lag)
    return best_lag


def report_circle(results: list[dict]) -> dict:
    alive = torch.cat([r["alive"] for r in results])
    rmse = torch.cat([r["rmse"] for r in results])
    mx = torch.cat([r["max_dev"] for r in results])
    lag = torch.cat([r["lag_s"] for r in results])
    leg = torch.cat([r["leg"] for r in results])
    om, rad = results[0]["omega"], results[0]["radius"]
    v_tan = om * rad

    a = alive
    out = {
        "omega_rad_s": om,
        "radius_m": rad,
        "tangential_speed_m_s": v_tan,
        "n_trials": int(alive.numel()),
        "completion_rate": float(alive.float().mean().item()),
        "rmse_median_m": pct(rmse[a], 0.5),
        "rmse_p95_m": pct(rmse[a], 0.95),
        "max_dev_median_m": pct(mx[a], 0.5),
        "max_dev_p95_m": pct(mx[a], 0.95),
        "lag_median_s": pct(lag[a], 0.5),
        "lag_p95_s": pct(lag[a], 0.95),
    }
    print("\n" + "=" * 78)
    print(f"S2 원 궤적  (반경 {rad:.3f} m, ω {om:.2f} rad/s, 접선속도 {v_tan:.3f} m/s, 1회전)")
    print("=" * 78)
    print(f"완주율(낙상 없음): {out['completion_rate'] * 100:.1f}%   (S2-G5 기준 ≥90%)")
    print(f"RMSE      중앙 {out['rmse_median_m']:.4f} m   95p {out['rmse_p95_m']:.4f} m")
    print(f"최대 편차  중앙 {out['max_dev_median_m']:.4f} m   95p {out['max_dev_p95_m']:.4f} m   (S2-G2 절대 ≤0.12)")
    print(f"위상 지연  중앙 {out['lag_median_s']:.3f} s   95p {out['lag_p95_s']:.3f} s   (S2-G4 기준 ≤0.10 s)")
    ratio = out["max_dev_median_m"] / max(out["rmse_median_m"], 1e-9)
    out["max_dev_over_rmse"] = ratio
    print(f"최대편차/RMSE = {ratio:.2f}   (S2-G2 기준 ≤2.5)")

    tamp = torch.cat([r["tgt_amp"] for r in results])[a]
    famp = torch.cat([r["foot_amp"] for r in results])[a]
    out["target_amplitude_m"] = pct(tamp, 0.5)
    out["foot_amplitude_m"] = pct(famp, 0.5)
    rmean = torch.cat([r["radial_mean"] for r in results])[a]
    out["radial_mean_m"] = pct(rmean, 0.5)
    print(
        f"조건 검증: 목표 x 진폭 {out['target_amplitude_m']:.4f} m (반경 {rad:.3f} 이어야 함),"
        f" 실제 발 진폭 {out['foot_amplitude_m']:.4f} m"
    )
    print(
        f"반경 편향: 평균 반경거리 {out['radial_mean_m']:.4f} m vs 지령 {rad:.3f} m"
        f"  ({(out['radial_mean_m'] - rad) * 1000:+.1f} mm)   ← 진폭보다 떨림에 강건한 지표"
    )
    out["per_leg"] = {}
    print("\n다리별 RMSE 중앙값 [m]:")
    for li, name in enumerate(("FL", "FR", "RL", "RR")):
        sel = a & (leg == li)
        v = pct(rmse[sel], 0.5)
        out["per_leg"][name] = v
        print(f"    {name}: {v:.4f}")
    return out


def run_video(env, policy, seed: int) -> dict:
    """육안 확인용 — 3족 hold 를 3 s 마다 새 목표로 갱신하며 영상으로 남긴다.

    수치 게이트가 통과해도 gait 가 비정상인 사례가 이 저장소에 여러 건 있으므로, S1 종결
    판정에는 영상 확인이 필요하다.
    """
    base_env = env.unwrapped
    device = base_env.device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    env.reset()
    assert base_env._nominal_valid, "nominal 발 위치 캡처 실패."

    n = base_env.num_envs
    leg_names = ("FL", "FR", "RL", "RR")
    if args_cli.video_leg == "cycle":
        leg_idx = torch.arange(n, device=device) % 4
    else:
        leg_idx = torch.full((n,), leg_names.index(args_cli.video_leg), dtype=torch.long, device=device)
    cond_idx = torch.full((n,), CONDITIONS.index("stance3"), dtype=torch.long, device=device)
    n_leg = base_env.cfg.num_legs

    def fresh_resample(env_ids: torch.Tensor):
        """호출 때마다 **새 목표**를 뽑는다 — 한 영상에서 여러 도달 동작을 보기 위함."""
        if env_ids is None or int(env_ids.numel()) == 0:
            return
        ids = env_ids.to(torch.long)
        target = build_targets(base_env, leg_idx, cond_idx, gen)
        role = torch.ones(int(ids.numel()), n_leg, device=device)
        role[torch.arange(int(ids.numel()), device=device), leg_idx[ids]] = 0.0
        base_env._leg_role[ids] = role
        base_env._foot_target_b[ids] = target[ids]
        base_env._traj_center_b[ids] = target[ids]
        base_env._hold_counter[ids] = 0.0
        base_env._cmd_timer[ids] = base_env.cfg.command.resample_time_min

    base_env._resample_command = fresh_resample
    obs, _ = env.reset()

    err_sum = torch.zeros(n, device=device)
    steps = 0
    for _ in range(args_cli.video_length):
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        err_sum += foot_err_of_manip(base_env, leg_idx)
        steps += 1

    mean_err = float((err_sum / max(steps, 1)).mean().item())
    print(f"\n영상 저장: {args_cli.video_folder}   (평균 추종 오차 {mean_err:.4f} m, {steps} step)")
    # 카메라는 env 0 만 비추므로 env 0 의 발 높이를 함께 찍어, 영상에서 어느 다리가 들렸는지와
    # 인덱스 매핑이 일치하는지 대조할 수 있게 한다.
    foot_z = base_env._compute_foot_pos_b()[0, :, 2]
    nom_z = base_env._nominal_foot_pos_b[0, :, 2]
    print(f"env 0 조작 다리: {leg_names[int(leg_idx[0].item())]}")
    print(
        "env 0 발 높이 (nominal 대비) [cm]: "
        + ", ".join(f"{leg_names[i]} {(foot_z[i] - nom_z[i]) * 100:+.1f}" for i in range(4))
    )
    return {"mean_err": mean_err, "steps": steps, "manip_leg_env0": leg_names[int(leg_idx[0].item())]}


def pct(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return float("nan")
    return float(torch.quantile(x.float(), q).item())


def report_push(results: list[dict]) -> dict:
    leg = torch.cat([r["leg"] for r in results])
    cond = torch.cat([r["cond"] for r in results])
    force = torch.cat([r["force"] for r in results])
    success = torch.cat([r["success"] for r in results])
    alive = torch.cat([r["alive"] for r in results])
    s4 = torch.cat([r["stance4_rate"] for r in results])
    s3 = torch.cat([r["stance3_rate"] for r in results])
    peak = torch.cat([r["peak_speed"] for r in results])
    dz = torch.cat([r["manip_dz"] for r in results])
    mf = torch.cat([r["manip_force"] for r in results])

    out: dict = {"n_trials": int(success.numel()), "cells": [], "validity": {}, "gate": {}}

    i4, i3 = CONDITIONS.index("stance4"), CONDITIONS.index("stance3")
    m4, m3 = cond == i4, cond == i3
    out["validity"] = {
        "stance4_cond_mean_4foot_rate": float(s4[m4].mean().item()),
        "stance3_cond_mean_3foot_rate": float(s3[m3].mean().item()),
        "peak_speed_by_force": {
            f"{f:.0f}N": float(peak[force == f].mean().item()) for f in GATE_FORCES if (force == f).any()
        },
        "stance4_cond_manip_dz_m": float(dz[m4].mean().item()),
        "stance4_cond_manip_force_N": float(mf[m4].mean().item()),
        "stance3_cond_manip_dz_m": float(dz[m3].mean().item()),
        "stance3_cond_manip_force_N": float(mf[m3].mean().item()),
    }

    print("\n" + "=" * 78)
    print("S1-G5  push 강건성")
    print("=" * 78)
    print(f"{'force[N]':>9} | {'4족 성공률':>12} {'n':>6} | {'3족 성공률':>12} {'n':>6} | {'열화[%p]':>9}")
    print("-" * 78)
    for f in GATE_FORCES:
        row = {"force_N": f}
        for label, mask in (("stance4", m4), ("stance3", m3)):
            sel = mask & (force == f)
            k = int(sel.sum().item())
            rate = float(success[sel].float().mean().item()) if k else float("nan")
            row[f"{label}_rate"] = rate
            row[f"{label}_n"] = k
            row[f"{label}_fall_rate"] = float((~alive[sel]).float().mean().item()) if k else float("nan")
        row["degradation_pp"] = (row["stance4_rate"] - row["stance3_rate"]) * 100.0
        out["cells"].append(row)
        print(
            f"{f:9.0f} | {row['stance4_rate'] * 100:11.1f}% {row['stance4_n']:6d} |"
            f" {row['stance3_rate'] * 100:11.1f}% {row['stance3_n']:6d} | {row['degradation_pp']:9.1f}"
        )

    # 게이트: <150 N 대역 (0/50/100 N) 집계
    band = force < 150.0
    r4 = float(success[m4 & band].float().mean().item())
    r3 = float(success[m3 & band].float().mean().item())
    deg = (r4 - r3) * 100.0
    out["gate"] = {
        "band": "<150 N",
        "stance4_rate": r4,
        "stance3_rate": r3,
        "degradation_pp": deg,
        "pass_stance4_ge_0.95": r4 >= 0.95,
        "pass_stance3_ge_0.70": r3 >= 0.70,
        "pass_degradation_le_25pp": deg <= 25.0,
    }
    print("-" * 78)
    print(
        f"<150 N 대역:  4족 {r4 * 100:.1f}% (기준 ≥95%)  3족 {r3 * 100:.1f}% (기준 ≥70%)  열화 {deg:.1f}%p (기준 ≤25%p)"
    )
    print(f"조건 검증:  4족 조건의 4-foot 접지 비율 {out['validity']['stance4_cond_mean_4foot_rate'] * 100:.1f}%")
    print(f"            3족 조건의 3-foot 접지 비율 {out['validity']['stance3_cond_mean_3foot_rate'] * 100:.1f}%")
    print(
        f"            4족 조건 조작발: Δz {out['validity']['stance4_cond_manip_dz_m'] * 100:+.1f} cm,"
        f" 접촉력 {out['validity']['stance4_cond_manip_force_N']:.2f} N"
    )
    print(
        f"            3족 조건 조작발: Δz {out['validity']['stance3_cond_manip_dz_m'] * 100:+.1f} cm,"
        f" 접촉력 {out['validity']['stance3_cond_manip_force_N']:.2f} N"
    )
    print(f"push 인가 검증 (base 속도 피크): {out['validity']['peak_speed_by_force']}")

    # 다리별 분해 — 앞/뒤 다리는 nominal x 가 비대칭이라 난이도가 다를 수 있다.
    leg_names = ("FL", "FR", "RL", "RR")
    out["per_leg"] = {}
    print(f"\n다리별 (<150 N 대역, 3족): {'':>4}", end="")
    for li, name in enumerate(leg_names):
        sel = m3 & band & (leg == li)
        rate = float(success[sel].float().mean().item()) if int(sel.sum().item()) else float("nan")
        out["per_leg"][name] = {"stance3_rate": rate, "n": int(sel.sum().item())}
        print(f"{name} {rate * 100:.1f}%  ", end="")
    print()
    return out


def report_hold(results: list[dict]) -> dict:
    leg = torch.cat([r["leg"] for r in results])
    cell = torch.cat([r["cell"] for r in results])
    alive = torch.cat([r["alive"] for r in results])
    mean_err = torch.cat([r["mean_err"] for r in results])
    std_err = torch.cat([r["std_err"] for r in results])
    range_err = torch.cat([r["range_err"] for r in results])
    nx, ny, nz = results[0]["grid"]
    n_cell = nx * ny * nz
    leg_names = ("FL", "FR", "RL", "RR")

    out: dict = {"n_trials": int(alive.numel()), "alive_rate": float(alive.float().mean().item())}
    a = alive  # 낙상한 시행은 hold 지표가 무의미하므로 제외 (제외 비율도 보고한다)

    # ── S1-G2 워크스페이스 커버리지 ──────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"S1-G2  워크스페이스 커버리지 (격자 {GRID_CELL_M} m → {nx}×{ny}×{nz} = {n_cell} 셀 / 다리)")
    print("=" * 78)
    cell_rows = []
    for li in range(4):
        for ci in range(n_cell):
            sel = a & (leg == li) & (cell == ci)
            cnt = int(sel.sum().item())
            if cnt == 0:
                cell_rows.append({"leg": leg_names[li], "cell": ci, "n": 0, "mean_err": float("nan")})
                continue
            cell_rows.append(
                {"leg": leg_names[li], "cell": ci, "n": cnt, "mean_err": float(mean_err[sel].mean().item())}
            )
    filled = [c for c in cell_rows if c["n"] > 0]
    ok = [c for c in filled if c["mean_err"] <= G2_CELL_ERR]
    frac = len(ok) / max(len(filled), 1)
    out["g2"] = {
        "n_cells": len(cell_rows),
        "n_cells_with_samples": len(filled),
        "min_samples_per_cell": min((c["n"] for c in filled), default=0),
        "cell_pass_fraction": frac,
        "threshold_m": G2_CELL_ERR,
        "pass": frac >= G2_CELL_FRAC,
        "worst_cells": sorted(filled, key=lambda c: -c["mean_err"])[:8],
    }
    print(f"표본이 있는 셀 {len(filled)}/{len(cell_rows)} (셀당 최소 {out['g2']['min_samples_per_cell']} 시행)")
    print(
        f"셀 평균 오차 ≤ {G2_CELL_ERR} m 인 셀: {len(ok)}/{len(filled)} = {frac * 100:.1f}%"
        f"  (기준 ≥{G2_CELL_FRAC * 100:.0f}%)"
    )
    bx, by, _ = results[0]["box"]
    print("\n오차가 큰 셀 상위 8개 (셀 중심 오프셋 [m]):")
    for c in out["g2"]["worst_cells"]:
        ci = c["cell"]
        xi, yi, zi = ci // (ny * nz), (ci // nz) % ny, ci % nz
        cx = -bx + (xi + 0.5) * GRID_CELL_M
        cy = -by + (yi + 0.5) * GRID_CELL_M
        cz = (zi + 0.5) * GRID_CELL_M
        print(f"    {c['leg']} (x{cx:+.2f}, y{cy:+.2f}, z{cz:+.2f}) → {c['mean_err']:.4f} m  (n={c['n']})")

    # ── S1-G3 hold drift ────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print(f"S1-G3  hold drift ({HOLD_WINDOW_STEPS / 50.0:.0f} s hold)")
    print("=" * 78)
    pass_std = std_err[a] <= G3_STD_M
    pass_rng = range_err[a] <= G3_RANGE_M
    both = pass_std & pass_rng
    out["g3"] = {
        "std_median_m": pct(std_err[a], 0.5),
        "std_p95_m": pct(std_err[a], 0.95),
        "range_median_m": pct(range_err[a], 0.5),
        "range_p95_m": pct(range_err[a], 0.95),
        "frac_std_ok": float(pass_std.float().mean().item()),
        "frac_range_ok": float(pass_rng.float().mean().item()),
        "frac_both_ok": float(both.float().mean().item()),
        "mean_err_median_m": pct(mean_err[a], 0.5),
    }
    g3 = out["g3"]
    print(f"오차 std   : 중앙 {g3['std_median_m']:.4f}  95p {g3['std_p95_m']:.4f}   (기준 ≤{G3_STD_M})")
    print(f"오차 max−min: 중앙 {g3['range_median_m']:.4f}  95p {g3['range_p95_m']:.4f}   (기준 ≤{G3_RANGE_M})")
    print(
        f"기준 충족 시행 비율: std {g3['frac_std_ok'] * 100:.1f}%  range {g3['frac_range_ok'] * 100:.1f}%"
        f"  둘 다 {g3['frac_both_ok'] * 100:.1f}%"
    )
    print(f"hold 중 평균 오차 중앙값: {g3['mean_err_median_m']:.4f} m")
    print(f"낙상으로 제외된 시행: {(1.0 - out['alive_rate']) * 100:.2f}%")

    # ── 떨림(tremor) ────────────────────────────────────────────────────────
    jit = torch.cat([r["jitter"] for r in results])[a]
    jv = torch.cat([r["jvel_rms"] for r in results])[a]
    jit_m = torch.cat([r["jitter_manip"] for r in results])[a]
    jit_s = torch.cat([r["jitter_supp"] for r in results])[a]
    flip = torch.cat([r["flip_rate"] for r in results])[a]
    _deg = 180.0 / math.pi
    out["tremor"] = {
        "joint_target_jitter_rad_per_step": pct(jit, 0.5),
        "joint_target_jitter_deg_per_step": pct(jit, 0.5) * _deg,
        "jitter_manip_deg_per_step": pct(jit_m, 0.5) * _deg,
        "jitter_supp_deg_per_step": pct(jit_s, 0.5) * _deg,
        "delta_sign_flip_rate": pct(flip, 0.5),
        "joint_vel_rms_rad_s": pct(jv, 0.5),
        "foot_pos_std_m": g3["std_median_m"],
        "act_lag1_autocorr": pct(torch.cat([r["act_lag1"] for r in results])[a], 0.5),
        "act_msd_per_lag": {
            str(lag): pct(torch.cat([r["act_msd"][lag] for r in results])[a], 0.5) for lag in results[0]["act_msd"]
        },
    }
    tr = out["tremor"]
    print("\n떨림 지표 (hold 중, 중앙값):")
    print(
        f"    관절 목표 흔들림 {tr['joint_target_jitter_rad_per_step']:.5f} rad/step"
        f" ({tr['joint_target_jitter_deg_per_step']:.3f}°/step)"
    )
    print(
        f"      └ 조작 다리(적분 경로) {tr['jitter_manip_deg_per_step']:.3f}°/step"
        f"  ·  지지 다리(절대 경로) {tr['jitter_supp_deg_per_step']:.3f}°/step"
    )
    print(
        f"    Δtarget 부호 반전율 {tr['delta_sign_flip_rate']:.3f}"
        "   (~1.0 = 매 step 반전(고주파 chatter) · ~0.5 = 백색 · ~0.0 = 단조 표류)"
    )
    print(f"    관절 속도 RMS   {tr['joint_vel_rms_rad_s']:.4f} rad/s")
    print(f"    발 위치 std     {tr['foot_pos_std_m'] * 1000:.2f} mm")
    print(f"    action lag-1 자기상관 {tr['act_lag1_autocorr']:.3f}")
    msd = tr["act_msd_per_lag"]
    print(
        "    MSD/lag: " + "  ".join(f"k={lag}:{v:.4f}" for lag, v in msd.items()) + "   (일정=랜덤워크 · 감소=평균회귀)"
    )

    # 다리별 — 앞/뒤 워크스페이스가 다르므로 분해해 본다.
    print("\n다리별 (hold 평균 오차 중앙 / std 중앙):")
    out["per_leg"] = {}
    for li, name in enumerate(leg_names):
        sel = a & (leg == li)
        rec = {"mean_err_median": pct(mean_err[sel], 0.5), "std_median": pct(std_err[sel], 0.5)}
        out["per_leg"][name] = rec
        print(f"    {name}: {rec['mean_err_median']:.4f} / {rec['std_median']:.4f}")
    return out


def report_drift(results: list[dict]) -> dict:
    alive = torch.cat([r["alive"] for r in results])
    hold_err = torch.cat([r["hold_err"] for r in results])
    out: dict = {"n_trials": int(alive.numel()), "alive_rate": float(alive.float().mean().item()), "steps": {}}

    print("\n" + "=" * 78)
    print("S1-G7  base 표류 (외란 없음, 3족 hold)")
    print("=" * 78)
    print(f"낙상 없이 완주: {out['alive_rate'] * 100:.1f}%   hold 중 발 오차 중앙값 {pct(hold_err[alive], 0.5):.4f} m")
    print(
        f"\n{'t[s]':>6} | {'순변위 중앙':>11} {'순변위 95p':>11} | {'경로장 중앙':>11} {'경로장 95p':>11}"
        f" | {'2s후 중앙':>11} {'2s후 95p':>11}"
    )
    print("-" * 78)
    for s in DRIFT_CHECKPOINTS:
        net = torch.cat([r["net"][s] for r in results])[alive]
        path = torch.cat([r["path"][s] for r in results])[alive]
        net_late = torch.cat([r["net_late"][s] for r in results])[alive]
        path_late = torch.cat([r["path_late"][s] for r in results])[alive]
        rec = {
            "net_median": pct(net, 0.5),
            "net_p95": pct(net, 0.95),
            "path_median": pct(path, 0.5),
            "path_p95": pct(path, 0.95),
            "net_late_median": pct(net_late, 0.5),
            "net_late_p95": pct(net_late, 0.95),
            "path_late_median": pct(path_late, 0.5),
        }
        out["steps"][str(s)] = rec
        print(
            f"{s / 50.0:6.1f} | {rec['net_median']:11.4f} {rec['net_p95']:11.4f} |"
            f" {rec['path_median']:11.4f} {rec['path_p95']:11.4f} |"
            f" {rec['net_late_median']:11.4f} {rec['net_late_p95']:11.4f}"
        )

    final = out["steps"][str(DRIFT_CHECKPOINTS[-1])]
    out["gate"] = {
        "net_median_m": final["net_median"],
        "net_p95_m": final["net_p95"],
        "pass_median_le_0.05": final["net_median"] <= 0.05,
        "pass_p95_le_0.12": final["net_p95"] <= 0.12,
        # settle(2.0 s) 이후로 추가로 밀려난 양 — 일회성 counterbalance 를 제외한 "진짜 표류".
        "net_late_median_m": final["net_late_median"],
        "net_late_p95_m": final["net_late_p95"],
    }
    print("-" * 78)
    print(f"게이트(원안): 중앙 {final['net_median']:.4f} m (기준 ≤0.05)  95p {final['net_p95']:.4f} m (기준 ≤0.12)")
    print(
        f"settle 이후 추가 이동: 중앙 {final['net_late_median']:.4f} m  95p {final['net_late_p95']:.4f} m"
        "   ← 일회성 counterbalance 를 제외한 순수 표류"
    )

    # ── counterbalance 가설의 검증 ────────────────────────────────────────────
    # nominal 발 배치에서 CoM 은 **뒷다리를 들 때는 이미 지지 삼각형 안**이고, 앞다리를 들 때만
    # 밖이다. 초기 변위가 counterbalance 라면 앞다리 거상의 변위가 뒷다리보다 커야 한다.
    # 두 값이 같다면 counterbalance 설명은 성립하지 않는다.
    leg = torch.cat([r["leg"] for r in results])[alive]
    net_final = torch.cat([r["net"][DRIFT_CHECKPOINTS[-1]] for r in results])[alive]
    leg_names = ("FL", "FR", "RL", "RR")
    out["per_leg_net"] = {}
    print("\n다리별 순변위 중앙값 [m] — counterbalance 가설 검증:")
    for li, name in enumerate(leg_names):
        sel = leg == li
        val = pct(net_final[sel], 0.5)
        out["per_leg_net"][name] = {"net_median": val, "n": int(sel.sum().item())}
        print(f"    {name}: {val:.4f}  (n={int(sel.sum().item())})")
    com = results[0]["com_b"]
    out["com_b"] = [float(v) for v in com]
    print(f"    CoM(base frame) 실측: ({com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}) m")
    return out


def main():
    if args_cli.mode in RENDER_MODES:
        env, policy = build_env(args_cli.seed0)
        try:
            runner = run_showcase if args_cli.mode == "showcase" else run_video
            summary = runner(env, policy, args_cli.seed0)
        finally:
            env.close()  # RecordVideo 는 close 시점에 mp4 를 쓴다 — 자막은 그 뒤에 얹는다.
        if args_cli.mode == "showcase":
            src = os.path.join(args_cli.video_folder, "rl-video-step-0.mp4")
            dst = os.path.join(args_cli.video_folder, "pedipulation_showcase.mp4")
            annotate_video(src, dst, summary["timeline"], summary["title"])
            summary["video"] = dst
        if args_cli.out:
            with open(args_cli.out, "w") as fh:
                json.dump(summary, fh, indent=2, ensure_ascii=False)
        return

    results = []
    for i in range(args_cli.seeds):
        seed = args_cli.seed0 + i
        print(f"\n########## seed {seed} ({i + 1}/{args_cli.seeds}) ##########")
        env, policy = build_env(seed)
        try:
            if args_cli.mode == "push":
                results.append(run_push_seed(env, policy, seed))
            elif args_cli.mode == "hold":
                results.append(run_hold_seed(env, policy, seed))
            elif args_cli.mode == "circle":
                results.append(run_circle_seed(env, policy, seed))
            else:
                results.append(run_drift_seed(env, policy, seed))
        finally:
            env.close()

    reporters = {"push": report_push, "hold": report_hold, "drift": report_drift, "circle": report_circle}
    summary = reporters[args_cli.mode](results)
    summary["config"] = {
        "checkpoint": args_cli.checkpoint,
        "mode": args_cli.mode,
        "num_envs": args_cli.num_envs,
        "seeds": args_cli.seeds,
        "seed0": args_cli.seed0,
        "domain_rand": not args_cli.no_domain_rand,
    }
    if args_cli.out:
        with open(args_cli.out, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"\n결과 저장: {args_cli.out}")


if __name__ == "__main__":
    main()
    simulation_app.close()

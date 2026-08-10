# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Leg-Imitation-Tracking 정책에 vx 명령을 계단식으로 0→3→0 인가하며

로봇 추종 영상을 녹화하고, 속도/관절 상태를 npz 로 저장한다.
plot 은 별도 스크립트(plot_speed_ramp.py)에서 그린다.

Run:
    CUDA_VISIBLE_DEVICES=1 ./isaaclab.sh -p _workspace/leg/speed_ramp_record.py \
        --checkpoint logs/rsl_rl/leg_imitation_tracking/<RUN>/model_14200.pt \
        --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--hold_s", type=float, default=3.0, help="각 속도 단계 유지 시간 [s]")
parser.add_argument("--ramp_s", type=float, default=2.0, help="단계 사이 명령 선형 상승/하강 시간 [s] (0이면 계단식)")
parser.add_argument("--out_dir", type=str, default="_workspace/leg/selfcol_ramp_out")
parser.add_argument("--vx_max", type=float, default=4.0, help="(미사용) 램프 스크립트 호환용")
parser.add_argument("--const_vx", type=float, required=True, help="유지할 vx 명령 [m/s] — 참조 클립의 body-frame 평균속도")
parser.add_argument("--const_yaw", type=float, default=0.0, help="유지할 yaw rate 명령 [rad/s] — 참조 클립의 평균 yaw rate")
parser.add_argument("--duration_s", type=float, default=15.0, help="롤아웃 길이 [s]")
parser.add_argument("--cam", type=str, default="chase", help="카메라 프리셋: chase | top | side")
parser.add_argument(
    "--run_params",
    type=str,
    default=None,
    help="학습 run 의 params/env.yaml 경로. 지정하면 motion_file/reset_strategy 등 학습 설정을 복원한다. "
    "지정하지 않으면 **현재 소스 cfg 기본값**으로 env 가 만들어져 학습과 다른 플랜트를 측정할 수 있다.",
)
parser.add_argument(
    "--force_stand",
    action="store_true",
    help="정지 기립 자세에서 출발(rel_stand_envs=1.0). 지정하지 않으면 학습 설정(기본 10%)대로라 "
    "1-env 램프는 대개 RSI(참조 모션의 움직이는 프레임)에서 시작해 '0 m/s 출발' 전제가 깨진다.",
)
parser.add_argument(
    "--heading_hold",
    action="store_true",
    help="heading 을 목표 방향으로 유지하도록 yaw rate 명령을 P 제어로 생성. "
    "끄면 기존 동작(yaw rate 명령 항상 0)이라 방향이 드리프트해도 되돌리지 않는다.",
)
parser.add_argument("--heading_kp", type=float, default=3.0, help="heading 보정 P 게인 [1/s]")
parser.add_argument(
    "--heading_ki",
    type=float,
    default=1.0,
    help="heading 보정 I 게인 [1/s^2]. 정책이 작은 yaw rate 명령에 거의 반응하지 않아 P 항만으로는 "
    "정상상태 오차가 남는다(횡변위 누적). 0 이면 순수 P 제어.",
)
parser.add_argument(
    "--heading_target",
    type=str,
    default="init",
    help="유지할 heading. 'init'=첫 스텝의 heading, 또는 숫자(world yaw [rad])",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
args_cli.headless = True
args_cli.enable_cameras = True
app = AppLauncher(args_cli).app

import inspect  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPO  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerParkourAMP  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# RecordVideo 가 캡처하는 perspective 카메라(/OmniverseKit_Persp) prim 을 USD 로 직접 옮긴다.
# (sim.set_camera_view 는 headless+RecordVideo 에서 _visualizers 가 비어 no-op; isaacsim.core.utils
#  viewports 헬퍼는 6.0 에서 제거됨 → 버전-무관한 USD 경로 사용.)
from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
from omni.kit.viewport.utility.camera_state import ViewportCameraState  # noqa: E402
from pxr import Gf  # noqa: E402

_PERSP_PATH = "/OmniverseKit_Persp"

TASK = "Leg-Imitation-Tracking-RMA-v0"
# vx 명령 프로파일 [m/s]: 0→vx_max 상승 후 vx_max→0 하강 (0.5 단위)
VX_PROFILE = [args_cli.const_vx]

os.makedirs(args_cli.out_dir, exist_ok=True)

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=1)

# ── 학습 run 설정 복원 ────────────────────────────────────────────
# parse_env_cfg 는 **현재 소스 cfg 기본값**을 쓴다. 학습이 CLI 로 덮어쓴 값(motion_file,
# reset_strategy 등)은 여기 반영되지 않으므로, 명시하지 않으면 학습과 다른 조건을 재게 된다.
# (예: 소스 기본 motion_file=merged_leg_pkl / reset_strategy=random(RSI, 움직이는 상태에서 시작))
_RESTORE_KEYS = (
    "motion_file",
    "reset_strategy",
    "rel_stand_envs",
    "rel_rest_init",
    "stand_reset_joint_noise",
    "cmd_deadzone",
    "lin_vel_x_min",
    "lin_vel_x_max",
    "lin_vel_y_min",
    "lin_vel_y_max",
    "yaw_vel_min",
    "yaw_vel_max",
)
_restored = {}
if args_cli.run_params:
    import yaml

    class _LooseLoader(yaml.SafeLoader):
        """python/object 등 알 수 없는 태그를 None 으로 흘려보내는 로더(스칼라 필드만 필요)."""

    _LooseLoader.add_multi_constructor("", lambda loader, suffix, node: None)
    with open(args_cli.run_params) as f:
        _run_cfg = yaml.load(f, Loader=_LooseLoader)
    for _k in _RESTORE_KEYS:
        if _k in _run_cfg and _run_cfg[_k] is not None:
            _old = getattr(env_cfg, _k, None)
            if str(_old) != str(_run_cfg[_k]):
                setattr(env_cfg, _k, _run_cfg[_k])
                _restored[_k] = (_old, _run_cfg[_k])

if args_cli.force_stand:
    if "stand" not in str(env_cfg.reset_strategy):
        env_cfg.reset_strategy = "random_stand"
    env_cfg.rel_stand_envs = 1.0

env_cfg.early_termination = False  # 넘어져도 시퀀스 끝까지 연속 기록
# episode timeout(기본 10s=500 step) 으로 로봇이 순간이동하면 영상이 튄다. 시퀀스 전체보다
# 길게 잡아 리셋 없이 연속 추종한다.
env_cfg.episode_length_s = 1e6
# NOTE: gym RecordVideo 는 IsaacLab visualizer 가 아니라서 ViewportCameraController 도
# sim.set_camera_view() 도 동작하지 않는다(_visualizers 가 비어 no-op). 대신 매 스텝
# /OmniverseKit_Persp 카메라 prim 의 world transform 을 USD 로 직접 써서 추종시킨다.
# 로봇 진행 방향(+x) 기준 뒤·위에서 따라가는 체이스캠.
# 로봇이 화면 중앙에 여유있게 들어오도록 뒤로 더 물리고 살짝 높인다.
# 카메라 프리셋. top 은 좌우 발 배치(대각 쏠림)를 보기 위한 준-수직 시점.
_CAM_PRESETS = {
    "chase": ((-6.0, 0.0, 2.4), (0.0, 0.0, 0.3)),
    "top": ((-0.55, 0.0, 2.15), (0.0, 0.0, 0.25)),
    "side": ((0.0, -3.0, 1.0), (0.0, 0.0, 0.3)),
}
CAM_EYE_OFFSET, CAM_TGT_OFFSET = _CAM_PRESETS[args_cli.cam]

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")

dt = 1.0 / env_cfg.policy_dt_hz  # policy step 주기 [s]
total_steps = int(round(args_cli.duration_s / dt))
VX_CMD_TRAJ = np.full(total_steps, args_cli.const_vx, dtype=np.float32)

env = gym.make(TASK, cfg=env_cfg, render_mode="rgb_array")
env = gym.wrappers.RecordVideo(
    env,
    video_folder=args_cli.out_dir,
    step_trigger=lambda s: s == 0,
    video_length=total_steps,
    name_prefix="trot_match",
    disable_logger=True,
)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

_accepted = set(inspect.signature(_VendoredPPO.__init__).parameters.keys()) - {"self"}
_cfg = agent_cfg.to_dict()
_cfg["algorithm"] = {k: v for k, v in _cfg["algorithm"].items() if k in _accepted or k == "class_name"}
runner = OnPolicyRunnerParkourAMP(env, _cfg, log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print(f">>> checkpoint 로드: {args_cli.checkpoint}")
print(f">>> 고정 명령 vx={args_cli.const_vx} yaw={args_cli.const_yaw} rad/s, "
      f"{total_steps} steps ({total_steps * dt:.1f}s), cam={args_cli.cam}")

base = env.unwrapped
base._resample_steering = lambda env_ids: None  # 자동 재샘플 무력화

if _restored:
    print(">>> 학습 run 설정 복원:")
    for _k, (_o, _n) in _restored.items():
        print(f"      {_k}: {_o} -> {_n}")
elif args_cli.run_params:
    print(">>> 학습 run 설정: 소스 cfg 와 동일(복원할 차이 없음)")
else:
    print("[경고] --run_params 미지정 — 현재 소스 cfg 기본값으로 측정한다(학습 조건과 다를 수 있음).")
    print(f"        motion_file={env_cfg.motion_file}")
    print(f"        reset_strategy={env_cfg.reset_strategy}")

names = list(base._robot.data.joint_names)
limit = base._robot.data.joint_effort_limits.torch[0].cpu().numpy()


def current_yaw() -> float:
    """base heading(world yaw) [rad]. quat 은 xyzw."""
    q = base._robot.data.body_quat_w[0, base.ref_body_index]
    x, y, z, w = q[0], q[1], q[2], q[3]
    return float(torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def wrap_to_pi(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# heading 유지 목표. 'init' 이면 첫 스텝의 heading 을 목표로 삼는다.
if args_cli.heading_target == "init":
    HEADING_TARGET = current_yaw()
else:
    HEADING_TARGET = float(args_cli.heading_target)
YAW_CMD_MIN, YAW_CMD_MAX = float(env_cfg.yaw_vel_min), float(env_cfg.yaw_vel_max)
yaw_err_int = 0.0  # heading 오차 적분항
print(
    f">>> heading_hold={args_cli.heading_hold} target={HEADING_TARGET:+.3f} rad "
    f"kp={args_cli.heading_kp} ki={args_cli.heading_ki} yaw_cmd clip=[{YAW_CMD_MIN}, {YAW_CMD_MAX}]"
)

# 기록 버퍼
log = {
    k: []
    for k in ("t", "vx_cmd", "vx", "vy", "yaw", "jpos", "jtau", "jvel", "px", "py", "yaw_ang", "yaw_err", "yaw_cmd")
}

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]


# Kit viewport 는 카메라 pose 를 자체 매니퓰레이터 상태로 관리하며 매 프레임 USD 를 덮어쓴다.
# 따라서 USD xformOp 직접 수정이 아니라 ViewportCameraState API 로 제어해야 렌더에 반영된다.
_viewport = get_active_viewport()
_cam_state = ViewportCameraState(_PERSP_PATH, _viewport) if _viewport is not None else None
if _cam_state is None:
    print(f"[경고] active viewport 를 찾지 못해 카메라 추종을 건너뜁니다.")


# 카메라 오프셋을 목표 heading 으로 회전 — 로봇이 목표 방향을 유지하면 화면상 정면 주행으로 보이고,
# 드리프트하면 화면에서 옆으로 벗어나는 것이 그대로 드러난다(world 고정 방향 기준 관찰).
_ct, _st = np.cos(HEADING_TARGET), np.sin(HEADING_TARGET)


def _rot_offset(off):
    return (off[0] * _ct - off[1] * _st, off[0] * _st + off[1] * _ct, off[2])


CAM_EYE_OFFSET = _rot_offset(CAM_EYE_OFFSET)
CAM_TGT_OFFSET = _rot_offset(CAM_TGT_OFFSET)


def track_camera():
    """현재 로봇 base(world 좌표)로 perspective 카메라를 이동시켜 추종."""
    if _cam_state is None:
        return
    bp = base._robot.data.body_pos_w[0, base.ref_body_index].cpu().numpy()
    eye = Gf.Vec3d(float(bp[0] + CAM_EYE_OFFSET[0]), float(bp[1] + CAM_EYE_OFFSET[1]), float(bp[2] + CAM_EYE_OFFSET[2]))
    tgt = Gf.Vec3d(float(bp[0] + CAM_TGT_OFFSET[0]), float(bp[1] + CAM_TGT_OFFSET[1]), float(bp[2] + CAM_TGT_OFFSET[2]))
    _cam_state.set_position_world(eye, True)
    _cam_state.set_target_world(tgt, True)


with torch.inference_mode():
    for step in range(total_steps):
        vx_cmd = float(VX_CMD_TRAJ[step])
        base._lin_vel_cmd[:, 0] = vx_cmd
        base._lin_vel_cmd[:, 1] = 0.0

        # heading 보정: 목표 heading 과의 각도 오차에 비례하는 yaw rate 를 명령한다.
        # 명령은 학습 시 노출된 yaw rate 범위로 clip 해 정책을 OOD 로 밀지 않는다.
        yaw_ang = current_yaw()
        yaw_err = wrap_to_pi(HEADING_TARGET - yaw_ang)
        if not args_cli.heading_hold:
            yaw_cmd = float(args_cli.const_yaw)
        else:
            raw = args_cli.heading_kp * yaw_err + args_cli.heading_ki * yaw_err_int
            yaw_cmd = float(np.clip(raw, YAW_CMD_MIN, YAW_CMD_MAX))
            # anti-windup: 명령이 포화된 상태에서 오차가 같은 방향이면 적분을 멈춘다.
            if raw == yaw_cmd or (raw - yaw_cmd) * yaw_err < 0.0:
                yaw_err_int += yaw_err * dt
        base._yaw_vel_cmd[:] = yaw_cmd

        track_camera()  # env.step 내부 render 전에 카메라를 현재 base 로 옮김
        if step % 150 == 0:
            _bp = base._robot.data.body_pos_w[0, base.ref_body_index].cpu().numpy()
            print(f"[cam] step {step:4d} vx_cmd {vx_cmd:.1f}  base=({_bp[0]:+.2f},{_bp[1]:+.2f},{_bp[2]:.2f})", flush=True)
        actions = policy(obs)
        _res = env.step(actions)
        obs = _res[0]

        d = base._robot.data
        bp = d.body_pos_w[0, base.ref_body_index]
        log["t"].append(step * dt)
        log["vx_cmd"].append(vx_cmd)
        log["vx"].append(d.root_link_lin_vel_b[0, 0].item())
        log["vy"].append(d.root_link_lin_vel_b[0, 1].item())
        log["yaw"].append(d.root_link_ang_vel_b[0, 2].item())
        log["px"].append(bp[0].item())
        log["py"].append(bp[1].item())
        log["yaw_ang"].append(yaw_ang)
        log["yaw_err"].append(yaw_err)
        log["yaw_cmd"].append(yaw_cmd)
        log["jpos"].append(d.joint_pos.torch[0].cpu().numpy())
        log["jtau"].append(d.applied_torque.torch[0].cpu().numpy())
        log["jvel"].append(d.joint_vel.torch[0].cpu().numpy())

env.close()

out = os.path.join(args_cli.out_dir, "trot_match_data.npz")
np.savez(
    out,
    t=np.array(log["t"]),
    vx_cmd=np.array(log["vx_cmd"]),
    vx=np.array(log["vx"]),
    vy=np.array(log["vy"]),
    yaw=np.array(log["yaw"]),
    jpos=np.array(log["jpos"]),
    jtau=np.array(log["jtau"]),
    jvel=np.array(log["jvel"]),
    joint_names=np.array(names),
    effort_limit=limit,
    vx_profile=np.array(VX_PROFILE),
    hold_s=args_cli.hold_s,
    dt=dt,
    px=np.array(log["px"]),
    py=np.array(log["py"]),
    yaw_ang=np.array(log["yaw_ang"]),
    yaw_err=np.array(log["yaw_err"]),
    yaw_cmd=np.array(log["yaw_cmd"]),
    heading_hold=bool(args_cli.heading_hold),
    heading_kp=float(args_cli.heading_kp),
    heading_ki=float(args_cli.heading_ki),
    heading_target=float(HEADING_TARGET),
    ramp_s=args_cli.ramp_s,
    const_vx=float(args_cli.const_vx),
    const_yaw=float(args_cli.const_yaw),
    cam=str(args_cli.cam),
    checkpoint=args_cli.checkpoint,
    motion_file=str(env_cfg.motion_file),
    reset_strategy=str(env_cfg.reset_strategy),
)
print(f">>> 데이터 저장: {out}")
print(f">>> 영상 저장: {args_cli.out_dir}/trot_match-step-0.mp4")
app.close()

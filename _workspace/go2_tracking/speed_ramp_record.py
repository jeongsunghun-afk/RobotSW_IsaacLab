# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2-Imitation-Tracking 정책에 vx 명령을 0→4→0 으로 인가하며 기록한다.

`_workspace/leg/speed_ramp_record_rma.py` 의 Go2 판이다. 영상과 npz 를 남기고
plot 은 별도 스크립트(`plot_speed_ramp.py`)가 그린다.

**leg 판과 다른 핵심 한 가지 — estimator 주입.**
`runner.get_inference_policy()` 가 돌려주는 `act_inference` 는
`actor_critic_parkour.py:298` 에서 **환경의 ground-truth `priv_explicit`**
(root 선속도/각속도)을 그대로 actor 에 넣는다. 그런데 학습은
`ppo_parkour.py:187` 에서 `obs["priv_explicit"]` 를 `estimator(obs["policy"])`
로 **대체**한다(`train_with_estimated_states: True`). 실기에는 root 선속도를
측정할 수단이 없으므로 배포 경로도 estimator 를 쓴다.
따라서 그대로 쓰면 **실제로 배포할 수 없는 정책**을 재는 셈이라, 여기서는
학습·배포와 같은 경로로 estimator 출력을 주입한다.

Run:
    CUDA_VISIBLE_DEVICES=1 env -u DISPLAY ./isaaclab.sh -p \
        _workspace/go2_tracking/speed_ramp_record.py \
        --checkpoint <경로>/model_34300.pt --out_dir _workspace/go2_tracking/ramp_out
"""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument(
    "--task", type=str, default="Go2-Imitation-Tracking-v0",
    help="gym task id. 5.1 에서 학습한 obs-48 체크포인트를 재려면 "
         "`Go2-Imitation-Tracking-Legacy48-v0` 를 쓴다(그 task 는 `use_pace_params=False`, "
         "`domain_rand=False` 가 기본이라 5.1 학습 조건과 일치한다).",
)
parser.add_argument("--hold_s", type=float, default=3.0, help="각 속도 단계 유지 시간 [s]")
parser.add_argument("--ramp_s", type=float, default=1.5, help="단계 사이 명령 선형 상승/하강 시간 [s] (0이면 계단식)")
parser.add_argument("--out_dir", type=str, default="_workspace/go2_tracking/ramp_out")
parser.add_argument("--smoke", action="store_true", help="3 레벨 × 0.5s 짧은 확인용")
parser.add_argument("--profile", type=str, default="", help="쉼표 구분 vx 레벨로 기본 프로파일을 대체 (진단용)")
parser.add_argument(
    "--warmup_s", type=float, default=3.0,
    help="프로파일 시작 전 --warmup_vx 로 유지하는 시간 [s]. 이 env 는 참조 모션 프레임에서 "
         "초기화돼 스폰 순간 이미 ~2 m/s 로 달리고 있다. 곧바로 vx_cmd=0 을 주면 급제동으로 "
         "넘어지고 early_termination=False 라 그대로 기록이 끝난다.",
)
parser.add_argument(
    "--warmup_vx", type=str, default="2.0,1.0,0.5",
    help="warmup 의 vx 명령 [m/s]. 쉼표로 여러 값을 주면 warmup_s 를 균등 분할해 **점진적으로** "
         "낮춘다. 스폰 속도(~2 m/s)에서 프로파일 첫 레벨(0.0)로 한 번에 떨어뜨리면 급제동으로 "
         "넘어지기 때문이다 — 단계적으로 줄이면 정상 정지한다(진단으로 확인).",
)
parser.add_argument(
    "--num_envs", type=int, default=1,
    help="동시 기록 env 수. 1 이면 leg 시리즈와 같은 단발 기록. 2 이상이면 domain randomization "
         "draw 가 env 마다 달라 낙상 여부·속도 천장을 **통계로** 볼 수 있다. 관절 시계열은 "
         "env 0 만 저장한다.",
)
parser.add_argument("--gt_priv", action="store_true", help="estimator 대신 GT priv_explicit 사용 (대조군)")
parser.add_argument(
    "--stochastic", action="store_true",
    help="평균 행동 대신 정책 분포에서 **샘플링**한다. 학습은 std~2.9 인 확률 정책으로 굴렀으므로 "
         "학습이 실제로 경험한 제어기는 이쪽이다. 배포는 평균을 쓴다.",
)
parser.add_argument(
    "--priv_latent", action="store_true",
    help="actor 의 세 번째 입력 블록을 history_latent 대신 **priv_latent**(특권 DR 파라미터 인코더)로 "
         "쓴다. 학습 롤아웃은 `hist_encoding = it %% 20 == 0`(on_policy_runner_amp.py:125)이라 "
         "**95%% 가 이 경로**이고, 배포/`act_inference` 는 항상 history_latent 다.",
)
parser.add_argument("--no_video", action="store_true", help="영상 없이 npz 만")
parser.add_argument(
    "--debug_obs", type=int, default=0,
    help="N>0 이면 N step 마다 obs/행동 진단을 찍는다. 램프가 학습과 다른 관측을 주고 있는지 "
         "(명령 채널·history 버퍼·행동 크기) 가르는 용도.")
parser.add_argument(
    "--free_cmd", action="store_true",
    help="명령을 덮어쓰지 않고 env 의 `_resample_steering`(학습과 동일한 랜덤 재샘플)을 그대로 둔다. "
         "학습이 실제로 방문하는 상태 분포를 재기 위한 모드 — 램프가 아니다.",
)
parser.add_argument(
    "--init_rest", action="store_true",
    help="시작 직후 로봇을 **깨끗한 기립 정지 상태**(default joint pos, 수평, 속도 0)로 덮어쓴다. "
         "이 env 는 항상 RSI(참조 모션 프레임)로 초기화돼 스폰 순간 이미 달리고 있으므로, "
         "'정지에서 출발' 자체를 시험하려면 상태를 강제로 바꿔야 한다.",
)
parser.add_argument(
    "--wide_cam", action="store_true",
    help="env0 추종 대신 **전 env 를 한 화면에 담는 고정 카메라**로 찍는다. 성공률이 100% 가 아닌 "
         "정책은 1 env 영상이 대표성이 없으므로(어느 개체가 잡히느냐로 결론이 뒤집힌다) 모집단을 "
         "그대로 보여줄 때 쓴다. 16 env 정도가 읽기 좋다.",
)
parser.add_argument(
    "--yaw_profile", type=str, default="",
    help="쉼표 구분 yaw rate [rad/s] 목록을 주면 **선회 추종 램프**로 바뀐다. vx 는 `--yaw_at_vx` 로 "
         "고정하고 yaw 만 계단식으로 훑는다(각 레벨 `--hold_s` 유지, 사이 `--ramp_s` 선형 전이). "
         "학습 범위는 `yaw_vel_min/max` = ∓1.0 rad/s 이므로 그 밖은 외삽이다. "
         "npz 에 `yaw_profile`·`yaw_cmd` 가 추가로 저장된다.",
)
parser.add_argument(
    "--yaw_at_vx", type=float, default=1.5,
    help="`--yaw_profile` 사용 시 유지할 전진 속도 [m/s]. 선회는 속도에 크게 의존하므로 어떤 값에서 "
         "쟀는지 반드시 보고서에 적을 것.",
)
parser.add_argument(
    "--seed", type=int, default=None,
    help="env RNG 시드. **기본은 None(시드 없음)이라 실행마다 DR 추첨이 달라진다.** "
         "2026-09-03 실측: 같은 체크포인트를 두 번 재니 `cmd 4.0` median 이 2.429 ↔ 0.587 로 "
         "1.84 m/s 흔들렸고, env 별 궤적은 아예 다른 개체가 됐다(env25 가 한 번은 낙상, 한 번은 "
         "4.037 m/s 완주). ★ arm 간 비교나 `--cam_env` 로 개체를 고를 때는 **반드시 시드를 고정**할 것 "
         "— 안 그러면 npz 로 고른 개체와 영상 속 개체가 다른 로봇이다.",
)
parser.add_argument(
    "--cam_env", type=int, default=0,
    help="추종 카메라가 따라갈 env 인덱스 (기본 0). 낙상률이 높은 정책은 env0 이 초반에 넘어져 "
         "영상이 '누워 있는 로봇'만 보여주는 일이 잦다. **어느 개체를 골랐는지와 고른 기준을 "
         "반드시 캡션에 적을 것** — 안 적으면 체리피킹과 구분되지 않는다. "
         "관절 시계열(jpos 등)은 이 값과 무관하게 env0 만 저장한다.",
)
parser.add_argument(
    "--no_dr", action="store_true",
    help="domain randomization 을 끈다. 기본 cfg 는 `push_robot=True`(5 s 마다 최대 1.0 m/s 수평 킥) "
         "+ obs 노이즈 + 마찰 0.4~1.4 + payload −1~+2 kg 이라 램프 각 stage(4.5 s)가 거의 매번 "
         "외란을 맞는다. `Go2-Imitation-Tracking-Legacy48-v0`(5.1 정책)는 `domain_rand=False` 가 "
         "기본이므로, 세대 간 matched 비교를 하려면 6.0 쪽에 이 플래그를 줘야 한다.",
)
parser.add_argument(
    "--no_pace", action="store_true",
    help="PACE 식별 물성(armature 0.17~0.20, viscous 2.3~2.5, coulomb)을 끄고 nominal 물성"
         "(armature 0.01, 마찰 0)으로 돌린다. **r2s sim(`Isaac-R2S-Go2-v0`)의 플랜트가 정확히 이것**"
         "이므로, 학습 플랜트와의 격차가 거동에 얼마나 영향을 주는지 재는 A/B 대조군이다.",
)
parser.add_argument(
    "--legacy_pace", action="store_true",
    help="PACE 물성을 **구 식별값**(`PACE_*_LEGACY`, armature 0.17~0.20 / viscous 2.3~2.5 / "
         "coulomb 0.01~0.02)으로 되돌린다. 구값은 실기 캡처의 104 ms 시간축 어긋남을 armature·"
         "viscous 가 흡수해 나온 오식별값이지만, **그 플랜트에서 학습된 정책을 그 플랜트에서 재는** "
         "matched 비교에는 반드시 필요하다. 현재 소스는 재식별값이라 이 플래그 없이 구 체크포인트를 "
         "돌리면 정책과 플랜트가 어긋난 cross-plant 측정이 된다. 근거: "
         "`reports/_comparisons/pace_go2_sysid_excitation_audit/`.",
)
parser.add_argument(
    "--no_push",
    action="store_true",
    help="push_robot(5 s 마다 최대 1.0 m/s 킥)만 끈다. 나머지 DR 은 유지하므로 `--no_dr` 과 달리 "
    "외란 항 하나의 기여도를 분리할 수 있다(`--no_dr` 은 DR 학습 정책에 OOD 라 해석 불가).",
)
parser.add_argument(
    "--joint_vel_noise",
    type=float,
    default=-1.0,
    help="≥0 이면 `dr.joint_vel_noise` 를 이 값으로 고정한다. 램프는 run 의 params 가 아니라 "
    "**현재 소스 cfg** 로 env 를 만들기 때문에, cfg 기본값을 바꾼 뒤 옛 세대를 재면 학습 조건과 "
    "어긋난 cross-condition 측정이 된다. 그 세대의 학습값을 여기에 명시해 고정할 것.",
)
parser.add_argument(
    "--sync_spawn_props",
    action="store_true",
    help="run 의 `params/env.yaml` 에 적힌 `robot.spawn.rigid_props` 를 그대로 적용한다. hydra 로 "
    "spawn 속성을 덮어 학습한 run(예: `max_depenetration_velocity=3.0`)은 소스 cfg 값이 달라서, "
    "이 플래그 없이 재면 **학습 플랜트와 측정 플랜트가 다른** cross-plant 측정이 된다. 차이가 "
    "발견되면 이 플래그 없이는 실행을 거부한다.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _unknown = parser.parse_known_args()
# ★ 알 수 없는 인자를 조용히 삼키면 안 된다. `parse_known_args` 라서 오타나 **아직 정의되지 않은
#   플래그**(예: 추가에 실패한 `--no_push`)가 무시된 채 실행되고, 결과는 "그 조건으로 쟀다" 로
#   보고된다. 실제로 2026-08-06 에 `--no_push` 가 파서에 없는 상태로 실행돼 push 가 켜진 채
#   측정됐고 "push 가설 기각" 이라는 틀린 결론이 나왔다. 이제는 즉시 죽는다.
_KNOWN_EXTERNAL = ()  # 외부에서 주입되는 알려진 인자가 생기면 여기에만 예외로 추가할 것
_bad = [a for a in _unknown if a not in _KNOWN_EXTERNAL]
if _bad:
    parser.error(f"알 수 없는 인자: {_bad} — 오타이거나 아직 구현되지 않은 플래그다. 조용히 무시하지 않는다.")
args_cli.headless = True
# ★ headless + 영상일 때 `DISPLAY` 가 남아 있으면 Kit 이 GLX 를 잡으려다 죽거나(GLXBadFBConfig)
#   startup 에서 그대로 멈춘다. 실제로 2026-08-13 에 `--num_envs 1` 렌더가 이 상태로 **21 시간**
#   걸려 있었고, 그 뒤에 걸어 둔 램프 측정까지 하루 밀렸다. `report_video.py` 도 같은 이유로
#   서브프로세스 env 에서 DISPLAY 를 지운다. ssh X11 forwarding 세션에서 특히 잘 걸린다.
if not args_cli.no_video:
    os.environ.pop("DISPLAY", None)
args_cli.enable_cameras = not args_cli.no_video
app = AppLauncher(args_cli).app

import inspect  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.algorithms.ppo_parkour import PPOParkour as _ParentPPO  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerAMP  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

# RecordVideo 가 캡처하는 perspective 카메라(/OmniverseKit_Persp)를 매 스텝 로봇으로 옮긴다.
# (headless+RecordVideo 에서는 sim.set_camera_view 가 no-op 이다.)
# `--no_video` 로 실행하면 `enable_cameras=False` 라 Kit 이 viewport 확장을 아예 안 올린다
# → 최상위 import 는 ModuleNotFoundError 를 낸다. 영상을 찍을 때만 가져온다.
if not args_cli.no_video:
    from omni.kit.viewport.utility import get_active_viewport  # noqa: E402
    from omni.kit.viewport.utility.camera_state import ViewportCameraState  # noqa: E402
    from pxr import Gf  # noqa: E402

_PERSP_PATH = "/OmniverseKit_Persp"

TASK = args_cli.task
# `Go2-Imitation-Tracking-Legacy48-v0` 는 obs 48 / `ActorCritic` / estimator 없음 / priv_explicit
# 없음이라 현행 RMA 경로가 통째로 성립하지 않는다. 5.1 에서 학습한 체크포인트를 재는 전용 경로다.
LEGACY = "Legacy48" in TASK
#: latent-space imitation arm (`Go2-Imitation-Latent-v0`) 은 discriminator 가 없는
#: `PPOLatent` + `OnPolicyRunnerLatent` 다. AMP 러너로는 체크포인트가 안 붙는다.
LATENT = "Latent" in TASK
# vx 명령 [m/s]: 0→4 상승 후 4→0 하강. 4.0 은 이 task 의 학습 명령 범위 상한이다
# (`lin_vel_x` 0~4.0) — leg 와 달리 외삽이 아니라 범위 안이다.
VX_PROFILE = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0]
if args_cli.profile:
    VX_PROFILE = [float(x) for x in args_cli.profile.split(",")]
if args_cli.smoke:
    VX_PROFILE = [0.0, 0.5, 1.0, 0.5, 0.0]
    args_cli.hold_s, args_cli.ramp_s = 0.5, 0.25

os.makedirs(args_cli.out_dir, exist_ok=True)

env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.num_envs)
env_cfg.early_termination = False  # 넘어져도 시퀀스 끝까지 연속 기록
env_cfg.episode_length_s = 1e6  # 리셋(순간이동) 없이 연속 추종
# ★ 2026-09-03: env 의 명령 재샘플이 **살아났다**(`_get_dones` 가 `_post_physics_step` 을 부른다).
#   램프는 매 스텝 `base._lin_vel_cmd` 에 자기 프로파일을 써넣는데, 그 뒤 `env.step()` 안에서
#   타이머가 만료된 env 의 명령이 **랜덤 값으로 덮어써진다.** 정책은 램프가 의도한 명령이 아니라
#   재샘플된 명령을 보고, npz 에는 램프의 의도값이 기록돼 **조용히 틀린 측정**이 된다.
#   램프는 명령을 자기가 통제해야 하므로 재샘플 주기를 전체 시퀀스보다 길게 밀어 둔다.
#   `--free_cmd` 는 env 가 정한 명령을 그대로 따라가는 모드라 이 처리를 하지 않는다.
if not args_cli.free_cmd:
    env_cfg.tar_change_time_min = 1e9
    env_cfg.tar_change_time_max = 1e9
    print(">>> [cmd] 명령 재샘플 OFF (tar_change_time 1e9) — 램프가 명령을 통제한다")
if args_cli.seed is not None:
    env_cfg.seed = args_cli.seed
    print(f">>> [seed] env RNG 시드 {args_cli.seed} 고정 — DR 추첨이 실행 간 재현된다")
else:
    print(">>> [seed] 시드 없음 — 실행마다 DR 추첨이 달라진다(arm 비교/개체 선택 시 --seed 를 줄 것)")
if args_cli.joint_vel_noise >= 0.0:
    _prev = env_cfg.dr.joint_vel_noise
    env_cfg.dr.joint_vel_noise = args_cli.joint_vel_noise
    print(f">>> [joint_vel_noise] {_prev} → {args_cli.joint_vel_noise} rad/s 고정 (학습 조건 일치용)")
if args_cli.no_push:
    env_cfg.dr.push_robot = False
    print(">>> [no_push] push_robot 만 OFF (나머지 DR 은 그대로)")
if args_cli.no_dr:
    env_cfg.domain_rand = False
    print(">>> [no_dr] domain randomization OFF (push·obs noise·마찰/질량/게인 랜덤화 없음)")
if args_cli.no_pace:
    # PACE 물성 write 를 통째로 끈다(`_apply_dr`의 `if self.cfg.use_pace_params:` 블록). 그러면
    # UNITREE_GO2_CFG nominal(armature 0.01, viscous/coulomb 0)이 그대로 남아 r2s sim 과 같아진다.
    env_cfg.use_pace_params = False
    print(">>> [no_pace] PACE 물성 write OFF (스톡 플랜트)")  # 침묵 실패 방지: 반드시 로그를 남긴다
if args_cli.legacy_pace:
    # env 는 `PACE_ARMATURE` 등 **dict 객체를 이름으로 import** 해 두고(`..._env.py:64-67`)
    # `_apply_dr` 에서 그대로 읽는다. 따라서 같은 객체를 in-place 로 갱신하면 env 쪽에도 반영된다
    # (재대입은 env 모듈의 이름을 못 바꾸므로 효과가 없다).
    from isaaclab_tasks.direct.go2_imitation_tracking import go2_imitation_tracking_env_cfg as _pc

    for _live, _legacy in (
        (_pc.PACE_ARMATURE, _pc.PACE_ARMATURE_LEGACY),
        (_pc.PACE_VISCOUS, _pc.PACE_VISCOUS_LEGACY),
        (_pc.PACE_COULOMB, _pc.PACE_COULOMB_LEGACY),
    ):
        _live.update(_legacy)
    print(f">>> [legacy_pace] 구 식별값 적용: armature={_pc.PACE_ARMATURE} viscous={_pc.PACE_VISCOUS}")

# ── spawn 속성 동기화 ────────────────────────────────────────────────────────────
# 램프는 `parse_env_cfg` 로 **현재 소스 cfg** 에서 env 를 만든다. hydra 로
# `env.robot.spawn.rigid_props.*` 를 덮어 학습한 run 은 그 값이 `unitree.py` 에 없으므로,
# 아무 조치 없이 재면 정책은 A 플랜트에서 학습되고 측정은 B 플랜트에서 이뤄진다.
# `--no_pace` 누락으로 같은 사고를 이미 한 번 냈다(`project_ramp_uses_source_cfg_not_run_params`).
# 조용히 맞춰 주지도, 조용히 넘어가지도 않는다 — 다르면 멈추고 플래그를 요구한다.
def _run_env_yaml(ckpt_path: str) -> dict | None:
    """체크포인트가 속한 run 의 `params/env.yaml` 을 통째로 읽는다 (없으면 None)."""
    path = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), "params", "env.yaml")
    if not os.path.isfile(path):
        return None

    class _Loader(yaml.SafeLoader):
        pass

    # cfg 덤프에는 `!!python/tuple` 같은 태그가 섞여 있다. rigid_props 값은 전부 스칼라라
    # 알 수 없는 태그는 None 으로 흘려보내면 충분하다.
    _Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)
    _Loader.add_multi_constructor("!", lambda loader, suffix, node: None)
    with open(path) as f:
        return yaml.load(f, Loader=_Loader)  # noqa: S506 — 태그를 전부 무력화한 로더다


_run_env = _run_env_yaml(args_cli.checkpoint) or {}

# ── 관측 인코딩은 **맞춰 주는 쪽이 맞다** ────────────────────────────────────────────
# `joint_pos_tan_norm` 은 policy obs 폭을 42 ↔ 102 로, `amp_joint_tan_norm` 은 per-step disc obs 를
# 49 ↔ 109 로 바꾼다. 어긋나면 actor / discriminator 1 층 크기가 달라져 `load_state_dict` 가 그
# 자리에서 터지므로(`strict=False` 여도 shape 불일치는 raise 된다) rigid_props 처럼 **조용히 틀릴
# 위험이 없다.** 멈춰 세워 플래그를 요구할 이유가 없어 run 값으로 맞추고, 대신 크게 찍는다.
#
# ★ 새 관측 플래그를 만들면 **여기에 이름을 추가할 것.** 빠뜨리면 학습은 멀쩡히 끝나고 몇 시간 뒤
#   램프를 걸 때 처음 터진다(`joint_pos_tan_norm` 하나만 넣었다가 `amp_joint_tan_norm` 에서 그럴 뻔했다).
for _key in ("joint_pos_tan_norm", "amp_joint_tan_norm"):
    _run_val = _run_env.get(_key)
    if isinstance(_run_val, bool) and _run_val != getattr(env_cfg, _key):
        print(f">>> [obs] {_key}: {getattr(env_cfg, _key)} → {_run_val} (run 값으로 맞춤)")
        setattr(env_cfg, _key, _run_val)

try:
    _run_props = _run_env["robot"]["spawn"]["rigid_props"]
except (KeyError, TypeError):
    _run_props = None
_live = getattr(getattr(getattr(env_cfg, "robot", None), "spawn", None), "rigid_props", None)
if _run_props and _live is not None:
    # `_usd_field_exceptions` 같은 밑줄 키는 ClassVar 라 튜플 태그로 덤프되고, 위 로더가 None 으로
    # 흘려보내므로 항상 "다르다"고 잡힌다. 설정값이 아니니 비교에서 제외한다.
    _diff = {
        k: (getattr(_live, k), v)
        for k, v in _run_props.items()
        if not k.startswith("_") and hasattr(_live, k) and getattr(_live, k) != v
    }
    if _diff and not args_cli.sync_spawn_props:
        parser.error(
            f"run 의 params/env.yaml 과 소스 cfg 의 robot.spawn.rigid_props 가 다르다: {_diff} "
            "(소스값, run값). 이대로 재면 cross-plant 측정이다. run 값으로 맞추려면 "
            "--sync_spawn_props 를, 소스값으로 재는 게 의도라면 그 이유를 보고서에 적을 것."
        )
    for _k, (_src, _run) in _diff.items():
        setattr(_live, _k, _run)
        print(f">>> [sync_spawn_props] robot.spawn.rigid_props.{_k}: {_src} → {_run} (run 값으로 맞춤)")

CAM_EYE_OFFSET = (-3.5, 0.0, 1.2)  # base 기준 카메라 위치 [m] (Go2 는 leg 보다 작아 가까이)
CAM_TGT_OFFSET = (0.0, 0.0, 0.15)

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")

dt = 1.0 / env_cfg.policy_dt_hz
hold_steps = int(round(args_cli.hold_s / dt))
ramp_steps = int(round(args_cli.ramp_s / dt))
# warmup: 프로파일 전에 --warmup_vx 로 유지해 스폰 과도상태를 흘려보낸다.
warmup_steps = int(round(args_cli.warmup_s / dt))
WARMUP_VX = [float(x) for x in str(args_cli.warmup_vx).split(",")]
_per = max(1, warmup_steps // len(WARMUP_VX))
_warm = []
for _wv in WARMUP_VX:
    _warm.extend([_wv] * _per)
_warm = _warm[:warmup_steps] + [WARMUP_VX[-1]] * max(0, warmup_steps - len(_warm))
warmup_steps = len(_warm)
# 사다리꼴 명령: 레벨마다 [이전→현재 선형 상승] + [유지].
_cmd_traj, _prev = list(_warm), (WARMUP_VX[-1] if _warm else VX_PROFILE[0])
for _lvl in VX_PROFILE:
    if ramp_steps > 0:
        _cmd_traj.extend(list(np.linspace(_prev, _lvl, ramp_steps, endpoint=False)))
    _cmd_traj.extend([_lvl] * hold_steps)
    _prev = _lvl
VX_CMD_TRAJ = np.asarray(_cmd_traj, dtype=np.float32)

# ── 선회 램프 ────────────────────────────────────────────────────────────────
# `--yaw_profile` 이 있으면 vx 는 `--yaw_at_vx` 상수로 두고 yaw 를 같은 사다리꼴로 훑는다.
# ★ 이때 `vx_profile` 은 상수 배열이 되므로 **기존 분석의 `peak = argmax(vx_profile)` 관용구가
#   전 구간을 잘라낸다.** 선회 데이터는 `yaw_profile` 을 기준으로 별도 분석해야 한다.
YAW_PROFILE = [float(x) for x in args_cli.yaw_profile.split(",")] if args_cli.yaw_profile else []
if YAW_PROFILE:
    VX_PROFILE = [float(args_cli.yaw_at_vx)] * len(YAW_PROFILE)
    _yaw_traj, _yprev = [0.0] * warmup_steps, 0.0
    for _lvl in YAW_PROFILE:
        if ramp_steps > 0:
            _yaw_traj.extend(list(np.linspace(_yprev, _lvl, ramp_steps, endpoint=False)))
        _yaw_traj.extend([_lvl] * hold_steps)
        _yprev = _lvl
    YAW_CMD_TRAJ = np.asarray(_yaw_traj, dtype=np.float32)
    VX_CMD_TRAJ = np.concatenate([
        np.asarray(_warm, dtype=np.float32),
        np.full(len(YAW_CMD_TRAJ) - warmup_steps, float(args_cli.yaw_at_vx), dtype=np.float32),
    ])
    print(f">>> [yaw] 선회 램프: vx {args_cli.yaw_at_vx:.2f} m/s 고정, yaw {YAW_PROFILE} rad/s "
          f"(학습 범위 ∓1.0 밖은 외삽)")
else:
    YAW_CMD_TRAJ = np.zeros_like(VX_CMD_TRAJ)
total_steps = len(VX_CMD_TRAJ)
# 각 스텝이 속한 레벨 인덱스. warmup 구간은 -1 로 두어 집계에서 빠진다.
STAGE = np.concatenate([
    np.full(warmup_steps, -1, dtype=np.int32),
    np.repeat(np.arange(len(VX_PROFILE)), ramp_steps + hold_steps).astype(np.int32),
])[:total_steps]

env = gym.make(TASK, cfg=env_cfg, render_mode=None if args_cli.no_video else "rgb_array")
if not args_cli.no_video:
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=args_cli.out_dir,
        step_trigger=lambda s: s == 0,
        video_length=total_steps,
        name_prefix="speed_ramp",
        disable_logger=True,
    )
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

# cfg 키 필터의 기준은 **`PPOParkour`** 다. `PPOAMP.__init__` 은 `(amp_cfg, *args, **kwargs)`
# 라 자기 시그니처로 거르면 `amp_cfg` 만 남고 `rnd_cfg`(Logger 가 참조)·`estimator`(추론 경로의
# 핵심)까지 날아가며, 반대로 아예 안 거르면 6.0 cfg 의 `optimizer` 키가 PPOParkour 에서
# TypeError 를 낸다. 실제로 인자를 검증하는 것은 부모이므로 부모 기준 + `amp_cfg` 로 거른다.
if LEGACY:
    # 5.1 계열은 `OnPolicyRunnerAMPBase` + 벤더링 `PPO` 다. 필터 기준도 그 PPO 여야 한다
    # (`scripts/reinforcement_learning/rsl_rl/eval_tracking_cross_engine.py:75-85` 와 동일).
    from rsl_rl.algorithms.ppo import PPO as _LegacyPPO
    from rsl_rl.runners import OnPolicyRunnerAMPBase

    _accepted = set(inspect.signature(_LegacyPPO.__init__).parameters.keys()) - {"self"} | {"class_name"}
    _cfg = agent_cfg.to_dict()
    _dropped = sorted(k for k in _cfg["algorithm"] if k not in _accepted)
    _cfg["algorithm"] = {k: v for k, v in _cfg["algorithm"].items() if k in _accepted}
    print(f">>> [legacy48] algorithm cfg 에서 제외된 키: {_dropped}")
elif LATENT:
    from rsl_rl.algorithms.ppo_latent import PPOLatent as _LatentPPO

    _accepted = set(inspect.signature(_LatentPPO.__init__).parameters.keys()) - {"self"} | {
        "class_name", "style_cfg"
    }
    _accepted |= set(inspect.signature(_ParentPPO.__init__).parameters.keys()) - {"self"}
    _cfg = agent_cfg.to_dict()
    _dropped = sorted(k for k in _cfg["algorithm"] if k not in _accepted)
    _cfg["algorithm"] = {k: v for k, v in _cfg["algorithm"].items() if k in _accepted}
    print(f">>> [latent] algorithm cfg 에서 제외된 키: {_dropped}")
else:
    _accepted = set(inspect.signature(_ParentPPO.__init__).parameters.keys()) - {"self"} | {"amp_cfg", "class_name"}
    _cfg = agent_cfg.to_dict()
    _dropped = sorted(k for k in _cfg["algorithm"] if k not in _accepted)
    _cfg["algorithm"] = {k: v for k, v in _cfg["algorithm"].items() if k in _accepted}
    print(f">>> algorithm cfg 에서 제외된 키: {_dropped}")
    assert "rnd_cfg" in _cfg["algorithm"], "rnd_cfg 가 필터에서 빠졌다 — Logger 가 KeyError 를 낸다"
# `estimator` 는 algorithm 하위가 아니라 **러너 cfg 최상위** 키다
# (`on_policy_runner_parkour.py:76` 이 `train_cfg["estimator"]` 로 읽는다). 필터는 algorithm
# 하위에만 걸리므로 영향이 없지만, 없으면 estimator 가 빌드되지 않아 추론 경로가 GT 로 떨어진다.
if LEGACY:
    runner = OnPolicyRunnerAMPBase(env, _cfg, log_dir=None, device=agent_cfg.device)
    _loaded = torch.load(args_cli.checkpoint, map_location=agent_cfg.device, weights_only=False)
    # strict 로드: 키가 어긋나면 여기서 예외가 난다. 반쯤 로드된 정책으로 재놓고 속도 차이를
    # 세대 탓으로 오귀속하는 사고를 막는 지점이다.
    runner.alg.policy.load_state_dict(_loaded["model_state_dict"], strict=True)
    print(f">>> [legacy48] strict 로드 OK (iter={_loaded.get('iter')})")
else:
    assert "estimator" in _cfg, "러너 cfg 에 estimator 항목이 없다 — estimator 가 빌드되지 않는다"
    if LATENT:
        from rsl_rl.runners import OnPolicyRunnerLatent

        runner = OnPolicyRunnerLatent(env, _cfg, log_dir=None, device=agent_cfg.device)
    else:
        runner = OnPolicyRunnerAMP(env, _cfg, log_dir=None, device=agent_cfg.device)
    _loaded = torch.load(args_cli.checkpoint, map_location="cpu", weights_only=False)
    runner.load(args_cli.checkpoint, load_optimizer=False)

# ── estimator 가 실제로 살아 있는지 확인 (조용히 죽으면 GT 를 재게 된다) ──────────
# LEGACY(obs 48)는 estimator 자체가 없는 세대라 이 검증 대상이 아니다.
estimator = None if LEGACY else getattr(runner.alg, "estimator", None)
if not args_cli.gt_priv and not LEGACY:
    if estimator is None:
        raise RuntimeError(
            "runner.alg.estimator 가 None 이다. agent cfg 의 estimator 항목이 시그니처 필터에 "
            "걸려 빠졌을 수 있다. --gt_priv 로 GT 를 쓸 수는 있지만 그건 배포 불가 정책이다."
        )
    if "estimator_state_dict" not in _loaded:
        raise RuntimeError(f"체크포인트에 estimator_state_dict 가 없다: {args_cli.checkpoint}")
    estimator.eval()
del _loaded

# 벤더링된 PPOParkour 계열은 `get_policy()` 가 없다 — actor-critic 은 `alg.policy` 다
# (`runner.load` 도 `self.alg.policy.load_state_dict` 로 복원한다).
actor = runner.alg.policy.to(env.unwrapped.device)
runner.eval_mode()  # alg 가 아니라 runner 에 정의돼 있다
print(f">>> checkpoint 로드: {args_cli.checkpoint}")
print(f">>> priv_explicit 소스: {'GT (대조군)' if args_cli.gt_priv else 'estimator (학습·배포와 동일)'}")
print(
    f">>> 행동: {'샘플(학습과 동일)' if args_cli.stochastic else '평균(배포와 동일)'}"
    f" | latent: {'priv_latent (학습 롤아웃 95%)' if args_cli.priv_latent else 'history (배포)'}"
)
print(
    f">>> num_envs={args_cli.num_envs}  warmup {args_cli.warmup_s}s @ {WARMUP_VX} m/s"
    f" + {len(VX_PROFILE)} 레벨 × (상승 {args_cli.ramp_s}s + 유지 {args_cli.hold_s}s)"
    f" = {total_steps} steps ({total_steps * dt:.1f}s)"
)

base = env.unwrapped
if not args_cli.free_cmd:
    base._resample_steering = lambda env_ids: None  # 자동 명령 재샘플 무력화

names = list(base._robot.data.joint_names)


def _act_scalar(attr, default):
    """actuator 파라미터를 [12] float 배열로 뽑는다 — **actuator 그룹별로** 채운다.

    ⚠ 예전 구현은 `base._act`(= 첫 그룹) 하나만 읽어 12 관절 전부에 같은 값을 넣었다.
    calf 가 별도 그룹으로 분리된 뒤(effort 35.5 / hip·thigh 23.5) 이 방식은 calf 한계를
    23.5 로 잘못 기록해 npz 의 `유효한계`·`사용률` 열을 통째로 틀리게 만든다.
    `|τ|max` 자체는 실측이라 영향받지 않지만, 포화 판정을 그 열로 하면 안 된다.
    """
    out = np.full(len(names), float(default), dtype=np.float32)
    for _a in base._robot.actuators.values():
        v = getattr(_a, attr, None)
        if v is None:
            continue
        idx = _a.joint_indices
        if idx is None or isinstance(idx, slice):
            idx = list(range(len(names)))
        else:
            if torch.is_tensor(idx):
                idx = idx.detach().cpu().numpy()
            idx = [int(x) for x in np.atleast_1d(np.asarray(idx)).reshape(-1)]
        if torch.is_tensor(v):
            v = v.detach().cpu().numpy().reshape(-1)
            # env 축이 앞에 붙어 있을 수 있다 — 뒤에서 그룹 관절 수만큼 잘라 쓴다.
            v = np.broadcast_to(v[-len(idx) :], (len(idx),))
            for k, j in enumerate(idx):
                out[j] = float(v[k])
        else:
            for j in idx:
                out[j] = float(v)
    return out


# `data.joint_effort_limits` 는 Go2 에서 1e9 (sim solver 한계)라 포화 판정에 쓸 수 없다.
# 실제 한계는 DCMotor actuator 쪽이다 — `UNITREE_GO2_CFG` 기준 effort/saturation 23.5 N·m,
# velocity_limit 30 rad/s. DCMotor 는 토크-속도 곡선이 있어 유효 한계가 관절 속도에 따라
# 줄어들므로, 고정 한계와 속도 의존 한계를 **둘 다** 저장해 plot 에서 후자로 판정한다.
limit = _act_scalar("effort_limit", 23.5)
sat_effort = _act_scalar("saturation_effort", 23.5)
vel_limit = _act_scalar("velocity_limit", 30.0)
for _j, _n in enumerate(names):
    print(f">>> actuator 한계 {_n:>16s}: effort={limit[_j]:6.2f} saturation={sat_effort[_j]:6.2f} vel={vel_limit[_j]:6.2f}")

log = {k: [] for k in (
    "t", "vx_cmd", "vx", "vy", "yaw", "h", "jpos", "jtau", "jvel", "est_lin", "gt_lin",
    "vx_head", "pitch",
)}

# ── heading-frame vx 와 pitch ───────────────────────────────────────────────
# reward(`_get_rewards`)와 이 램프는 `root_link_lin_vel_b`, 즉 **roll·pitch 까지 포함한 body
# frame** 의 vx 를 쓴다. MimicKit 은 yaw 만 되돌린 **heading frame** 이다(`calc_heading_quat_inv`).
# base 가 pitch 하면 같은 지면속도라도 두 값이 갈리므로 차이를 함께 기록해 크기를 남긴다.
from isaaclab.utils.math import quat_apply  # noqa: E402

_FWD_B = torch.tensor([1.0, 0.0, 0.0], device=base.device).repeat(base.num_envs, 1)


def _heading_vx():
    """world 선속도를 로봇 heading(수평 투영)에 사영한 전진속도 [m/s]."""
    # 6.0 은 warp ProxyArray 를 돌려주므로 `.torch` 로 실제 텐서를 꺼내야 한다.
    q = base._robot.data.root_link_quat_w.torch  # xyzw (6.0)
    fwd_w = quat_apply(q, _FWD_B)
    h = fwd_w[:, :2]
    h = h / h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return (base._robot.data.root_link_lin_vel_w.torch[:, :2] * h).sum(dim=-1)


def _pitch():
    """projected gravity 로부터 pitch [rad] (env `_get_dones` 와 같은 식)."""
    g = base._robot.data.projected_gravity_b.torch
    return torch.atan2(-g[:, 0], torch.sqrt(g[:, 1] ** 2 + g[:, 2] ** 2))

obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]

if args_cli.init_rest:
    # RSI 로 들어온 "달리는 중" 상태를 지우고 깨끗한 기립 정지로 덮어쓴다.
    # (env 의 _reset_idx 와 같은 *_index API 를 쓴다.)
    _n = base.num_envs
    _ids = torch.arange(_n, device=base.device)
    _pose = base._robot.data.root_link_pose_w.clone()  # [N,7] pos(3)+quat(4)
    _pose[:, 2] = 0.34  # 기립 base 높이 [m] (실측 보행 시 ~0.32)
    _pose[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=base.device)  # 수평(xyzw)
    base._robot.write_root_link_pose_to_sim_index(root_pose=_pose, env_ids=_ids)
    base._robot.write_root_com_velocity_to_sim_index(
        root_velocity=torch.zeros(_n, 6, device=base.device), env_ids=_ids
    )
    base._robot.write_joint_state_to_sim_index(
        position=base._robot.data.default_joint_pos.clone(),
        velocity=torch.zeros_like(base._robot.data.joint_vel.torch),
        env_ids=_ids,
    )
    print(">>> init_rest: 기립 정지 상태로 초기화 (z=0.34, quat=identity, vel=0, joint=default)")

_viewport = get_active_viewport() if not args_cli.no_video else None
_cam_state = ViewportCameraState(_PERSP_PATH, _viewport) if _viewport is not None else None
if _cam_state is None and not args_cli.no_video:
    print("[경고] active viewport 를 찾지 못해 카메라 추종을 건너뜁니다.")


def track_camera():
    """현재 로봇 base(world 좌표)로 perspective 카메라를 이동시켜 추종."""
    if _cam_state is None:
        return
    if args_cli.wide_cam:
        # 전 env 를 한 화면에 담는 고정 카메라. **개체를 고르지 않으므로 모집단이 그대로 보인다** —
        # 성공률이 100% 가 아닌 정책을 1 env 로 찍으면 어느 개체가 잡히느냐로 결론이 뒤집힌다
        # (실제로 28% 정책을 찍었더니 성공 개체가 걸렸다). 뒤쪽 위에서 진행 방향(+x)을 내려다본다.
        o = base.scene.env_origins.cpu().numpy()
        cx, cy = float(o[:, 0].mean()), float(o[:, 1].mean())
        span = float(max(np.ptp(o[:, 0]), np.ptp(o[:, 1]))) + 1e-6  # NumPy 2.0: ndarray.ptp 제거됨
        # 너무 당기면 로봇이 점이 되어 전달이 안 된다 — 16 env(span 15) 기준으로 읽히게 잡았다.
        # env 수를 늘리면 span 이 커져 다시 작아지므로, 가독성이 필요하면 env 수를 줄일 것.
        _cam_state.set_position_world(Gf.Vec3d(cx - 0.5 * span - 8.0, cy, 0.35 * span + 5.0), True)
        _cam_state.set_target_world(Gf.Vec3d(cx + 0.5 * span, cy, 0.0), True)
        return
    bp = base._robot.data.body_pos_w[args_cli.cam_env, base.ref_body_index].cpu().numpy()
    eye = Gf.Vec3d(*(float(bp[i] + CAM_EYE_OFFSET[i]) for i in range(3)))
    tgt = Gf.Vec3d(*(float(bp[i] + CAM_TGT_OFFSET[i]) for i in range(3)))
    _cam_state.set_position_world(eye, True)
    _cam_state.set_target_world(tgt, True)


with torch.inference_mode():
    for step in range(total_steps):
        if args_cli.free_cmd:
            vx_cmd = float(base._lin_vel_cmd[0, 0].item())  # env 가 정한 값을 기록만 한다
        else:
            vx_cmd = float(VX_CMD_TRAJ[step])
            base._lin_vel_cmd[:, 0] = vx_cmd
            base._lin_vel_cmd[:, 1] = 0.0
            yaw_cmd = float(YAW_CMD_TRAJ[step])
            base._yaw_vel_cmd[:] = yaw_cmd

        track_camera()
        if step % 250 == 0:
            _bp = base._robot.data.body_pos_w[0, base.ref_body_index].cpu().numpy()
            print(f"[cam] step {step:5d} vx_cmd {vx_cmd:.2f}  base=({_bp[0]:+.2f},{_bp[1]:+.2f},{_bp[2]:.2f})", flush=True)

        if LEGACY:
            # obs 48 은 dict 가 아니라 단일 텐서고 priv_explicit·estimator 가 없다.
            # `ActorCritic.act_inference` 가 배포 경로(평균 행동)다.
            actions = actor.act(obs) if args_cli.stochastic else actor.act_inference(obs)
            _res = env.step(actions)
            obs = _res[0] if isinstance(_res, tuple) else _res
            d = base._robot.data
            log["t"].append(step * dt)
            log["vx_cmd"].append(
                base._lin_vel_cmd[:, 0].cpu().numpy().copy() if args_cli.free_cmd else vx_cmd
            )
            log["vx"].append(d.root_link_lin_vel_b[:, 0].cpu().numpy().copy())
            log["vx_head"].append(_heading_vx().cpu().numpy().copy())
            log["pitch"].append(_pitch().cpu().numpy().copy())
            log["vy"].append(d.root_link_lin_vel_b[:, 1].cpu().numpy().copy())
            log["yaw"].append(d.root_link_ang_vel_b[:, 2].cpu().numpy().copy())
            log["h"].append(d.body_pos_w[:, base.ref_body_index, 2].cpu().numpy().copy())
            log["jpos"].append(d.joint_pos.torch.cpu().numpy().copy())
            log["jtau"].append(d.applied_torque.torch.cpu().numpy().copy())
            log["jvel"].append(d.joint_vel.torch.cpu().numpy().copy())
            # estimator 가 없으므로 est/gt 모두 실제 body 선속도를 남긴다(스키마 유지용).
            _gt = d.root_link_lin_vel_b[0, :3].cpu().numpy().copy()
            log["est_lin"].append(_gt)
            log["gt_lin"].append(_gt)
            continue

        # ── 학습/배포와 동일한 경로: priv_explicit 를 estimator 추정치로 대체 ──
        gt_priv = obs["priv_explicit"].clone()
        if args_cli.gt_priv:
            est_priv = gt_priv
        else:
            assert estimator is not None  # 위 가드에서 이미 걸러진다 (타입 좁히기용)
            est_priv = estimator(obs["policy"])
            obs["priv_explicit"] = est_priv
        # 행동 생성 경로 2×2 — (평균/샘플) × (history_latent/priv_latent).
        # `act_inference` 는 (평균, history_latent) 한 조합만 준다. 나머지는 `act` 로 분포를
        # 만든 뒤 sample() 또는 action_mean 을 꺼낸다(`act` 의 반환값이 곧 sample 이다).
        if args_cli.priv_latent or args_cli.stochastic:
            sampled = actor.act(obs, hist_encoding=not args_cli.priv_latent)
            actions = sampled if args_cli.stochastic else actor.action_mean
        else:
            actions = actor.act_inference(obs)

        if args_cli.debug_obs and step % args_cli.debug_obs == 0:
            _po = obs["policy"]
            # proprio 레이아웃: gravity(3) + lin_vel_cmd(2) + yaw_vel_cmd(1) + ...
            _cmd_ch = _po[:, 3:6]
            print(
                f"[dbg] step {step:5d} cmd_want {vx_cmd:5.2f} | obs.policy {tuple(_po.shape)} "
                f"finite={bool(torch.isfinite(_po).all())} |obs| mean {_po.abs().mean():.4f} "
                f"max {_po.abs().max():.3f} | 명령채널 {_cmd_ch[0].tolist()} "
                f"| act mean {actions.abs().mean():.4f} max {actions.abs().max():.3f} "
                f"std {actions.std():.4f}",
                flush=True,
            )
            if step == 0:
                _n = getattr(actor, "actor_obs_normalizer", None)
                print(f"        [norm] {type(_n).__name__} "
                      + (f"count={float(getattr(_n,'count',float('nan'))):.0f} "
                         f"mean|.|={_n._mean.abs().mean():.4f} std min={_n._std.min():.5f} "
                         f"max={_n._std.max():.4f}" if hasattr(_n, "_mean") else "(Identity)"),
                      flush=True)
                if hasattr(_n, "_mean"):
                    _sd = _n._std.flatten()
                    _mn = _n._mean.flatten()
                    _bad = (_sd < 1e-3).nonzero().flatten().tolist()
                    _names = (["gravity"] * 3 + ["cmd_vx", "cmd_vy", "cmd_yaw"]
                              + [f"jpos{i}" for i in range(12)] + [f"jvel{i}" for i in range(12)]
                              + [f"act{i}" for i in range(12)])
                    print(f"        [norm] std<1e-3 채널 {len(_bad)}개: "
                          + ", ".join(f"{_names[i] if i < len(_names) else i}(idx{i}, "
                                      f"mean {_mn[i]:.4f}, std {_sd[i]:.2e}, "
                                      f"현재값 {_po[0, i].item():.4f})" for i in _bad[:8]),
                          flush=True)
                    _z = ((_po - _mn) / _sd.clamp(min=1e-12))
                    print(f"        [norm] 정규화 후 |z| mean {_z.abs().mean():.2f} "
                          f"max {_z.abs().max():.3e} (argmax idx {int(_z.abs().max(0).values.argmax())})",
                          flush=True)
            for _k, _v in obs.items():
                if _k != "policy":
                    print(f"        obs[{_k}] {tuple(_v.shape)} |.| mean {_v.abs().mean():.4f} "
                          f"finite={bool(torch.isfinite(_v).all())}", flush=True)
        _res = env.step(actions)
        obs = _res[0]

        d = base._robot.data
        log["t"].append(step * dt)
        log["vx_cmd"].append(
            base._lin_vel_cmd[:, 0].cpu().numpy().copy() if args_cli.free_cmd else vx_cmd
        )
        log["vx"].append(d.root_link_lin_vel_b[:, 0].cpu().numpy().copy())
        log["vx_head"].append(_heading_vx().cpu().numpy().copy())
        log["pitch"].append(_pitch().cpu().numpy().copy())
        log["vy"].append(d.root_link_lin_vel_b[:, 1].cpu().numpy().copy())
        log["yaw"].append(d.root_link_ang_vel_b[:, 2].cpu().numpy().copy())
        log["h"].append(d.body_pos_w[:, base.ref_body_index, 2].cpu().numpy().copy())
        # ⚠ env 0 만 저장하면 안 된다. 성공률이 100% 가 아닌 정책에서 env 0 이 넘어져 있으면
        # `|q̇|≈0` + `|τ|` 가 캡에 붙은 "쓰러진 채 버티는" 표가 나오고, 이걸 포화로 오독하게 된다
        # (실제로 pace@16k 에서 calf 3 개가 캡도달 100% / q̇ 0.01 로 찍혔다). 전 env 를 저장한다.
        log["jpos"].append(d.joint_pos.torch.cpu().numpy().copy())
        log["jtau"].append(d.applied_torque.torch.cpu().numpy().copy())
        log["jvel"].append(d.joint_vel.torch.cpu().numpy().copy())
        # priv_explicit 앞 3개 = root 선속도 × priv_explicit_lin_vel_scale. 스케일을 되돌려 [m/s] 로 남긴다.
        _s = env_cfg.priv_explicit_lin_vel_scale
        log["est_lin"].append(est_priv[0, :3].cpu().numpy() / _s)
        log["gt_lin"].append(gt_priv[0, :3].cpu().numpy() / _s)

env.close()

out = os.path.join(args_cli.out_dir, "ramp_data.npz")
np.savez(
    out,
    t=np.array(log["t"]),
    vx_cmd=np.array(log["vx_cmd"]),
    vx=np.array(log["vx"]),
    vy=np.array(log["vy"]),
    yaw=np.array(log["yaw"]),
    base_h=np.array(log["h"]),
    num_envs=args_cli.num_envs,
    jpos=np.array(log["jpos"]),
    jtau=np.array(log["jtau"]),
    jvel=np.array(log["jvel"]),
    vx_head=np.array(log["vx_head"]),
    pitch=np.array(log["pitch"]),
    est_lin=np.array(log["est_lin"]),
    gt_lin=np.array(log["gt_lin"]),
    stage=STAGE[:total_steps],
    joint_names=np.array(names),
    effort_limit=limit,
    saturation_effort=sat_effort,
    velocity_limit=vel_limit,
    vx_profile=np.array(VX_PROFILE),
    yaw_profile=np.array(YAW_PROFILE),          # 선회 램프가 아니면 빈 배열
    yaw_cmd=YAW_CMD_TRAJ[:total_steps],         # 스텝별 명령 yaw rate (실측 `yaw` 와 짝)
    yaw_at_vx=float(args_cli.yaw_at_vx),
    hold_s=args_cli.hold_s,
    ramp_s=args_cli.ramp_s,
    warmup_s=args_cli.warmup_s,
    warmup_vx=args_cli.warmup_vx,
    dt=dt,
    checkpoint=args_cli.checkpoint,
    priv_source="gt" if args_cli.gt_priv else "estimator",
    action_mode="stochastic" if args_cli.stochastic else "mean",
    latent_path="priv_latent" if args_cli.priv_latent else "history",
    init_rest=bool(args_cli.init_rest),
)
print(f">>> 데이터 저장: {out}")
if not args_cli.no_video:
    print(f">>> 영상 저장: {args_cli.out_dir}/speed_ramp-step-0.mp4")
app.close()

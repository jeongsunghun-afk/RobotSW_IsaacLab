# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""참조 클립의 속도를 그대로 명령으로 준 롤아웃에서 policy/expert AMP 관측을 모아 **두 판별기로 교차 채점**한다.

배경: DRAIL 판별기는 expert/policy 를 거의 완벽히 가르는데(D(policy)≈0.06) 영상에서는 DRAIL 정책이
오히려 더 잘 따라하는 것처럼 보인다. 두 정책을 **같은 조건**(같은 클립 속도 명령, 같은 초기화)에서
굴리고, 두 run 의 판별기(A' mlp · B' drail) **모두**로 채점해 그 괴리를 수치로 남긴다.

★ 두 판별기는 중립 심판이 **아니다**. mlp 는 A' 를 상대로, drail 은 B' 를 상대로 학습됐고 각자 자기
정책을 눌러 놓았다. 게다가 DRAIL logit 은 확산 MSE 차분이라 sigmoid 가 0.5 근처에 몰리고 mlp logit 은
무계다. **같은 disc 안에서 expert vs policy(열 방향)만** 비교하라. 서로 다른 disc 의 절대값을 나란히
읽으면 안 된다.

측정 시 지킨 것:
  * 두 disc 다 ``eval()``. (``EmpiricalNormalization.forward`` 는 통계를 갱신하지 않지만 공짜라 건다.)
  * expert 시각은 ``(n_hist-1)·dt`` 이상에서만 뽑는다 — ``_compute_reference_buffers`` 가 창을 t=0 으로
    clamp 하므로 그보다 이른 시각은 히스토리가 납작해진다. policy 쪽은 워밍업 뒤부터만 모으니 그런
    표본이 없다.
  * 명령의 vy 는 **0 으로 버린다**. 학습 범위가 ``lin_vel_y_min=max=0`` 이라 정책은 nonzero vy 를 본 적이
    없고, ``_policy_amp_cond`` 가 ``‖lin_vel_cmd‖`` 를 쓰므로 vy 를 살리면 조건 열까지 분포를 벗어난다.
    버린 크기는 ``vy_dropped_mean`` 으로 남긴다.
  * RSI 시각은 env 마다 랜덤이다(클립만 고정). 전 env 를 t=0 으로 두면 초기상태가 동일해져 표본 수가
    사실상 1 이 된다. 명령은 클립 t=0 부터 재생하므로 상태 위상과 명령 위상은 어긋나 있다 — 대부분
    클립이 등속이라 무해하지만 yaw 가 구간 내 변하는 turn 클립에서는 주의.
  * DRAIL 의 ``get_logits`` 는 t·eps 를 매번 새로 뽑는다(``eval()`` 로 막히지 않음). ``--drail_reps`` 회
    평균 확률을 저장한다.

Run:
    CUDA_VISIBLE_DEVICES=1 env -u DISPLAY ./isaaclab.sh -p _workspace/leg/amp_obs_clip_probe.py \
        --checkpoint <A'>/model_49999.pt --run_params <A'>/params/env.yaml --tag condmlp50k \
        --other_checkpoint <B'>/model_45000.pt --n_envs 256 --start rsi
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True, help="측정 대상 정책 체크포인트")
parser.add_argument("--run_params", type=str, required=True, help="그 run 의 params/env.yaml")
parser.add_argument("--tag", type=str, required=True, help="출력 하위 폴더 이름 (예: condmlp50k)")
parser.add_argument("--other_checkpoint", type=str, default=None, help="교차 채점용 상대 run 체크포인트")
parser.add_argument("--other_run_params", type=str, default=None, help="상대 run 의 params/env.yaml (기본: 체크포인트 옆)")
parser.add_argument("--n_envs", type=int, default=256)
parser.add_argument("--start", type=str, default="rsi", choices=["rsi", "stand"])
parser.add_argument("--dur_s", type=float, default=6.0, help="클립당 기록 길이 [s] (워밍업 제외)")
parser.add_argument("--clips", type=str, default="all", help="쉼표로 구분한 클립 이름, 또는 all")
parser.add_argument("--max_samples", type=int, default=5000, help="policy/expert 각각 저장할 표본 수 상한")
parser.add_argument("--drail_reps", type=int, default=32, help="DRAIL logit 평균 횟수")
parser.add_argument("--seed", type=int, default=1234, help="★ 두 정책을 같은 난수 스트림에서 재려면 같은 값을 준다")
parser.add_argument(
    "--action_mode",
    type=str,
    default="mean",
    choices=["mean", "stochastic", "train_mean"],
    help="mean=act_inference(배포 경로) · stochastic=학습 롤아웃과 같은 경로에서 표본 추출 · "
    "train_mean=같은 경로의 분포 평균(잡음만 뺀 대조군)",
)
parser.add_argument("--stochastic", action="store_true", help="--action_mode stochastic 의 별칭")
parser.add_argument(
    "--save_seq",
    action="store_true",
    help="서브샘플 대신 **전 env·전 스텝**의 live amp 관측과 상태를 별도 `{clip}_{start}_seq.npz` 로 남긴다. "
    "저역통과 반사실 채점처럼 시간축이 필요한 후속 분석용. 기존 npz 의 키는 건드리지 않는다.",
)
parser.add_argument(
    "--out_root",
    type=str,
    default="reports/leg_imitation/_comparisons/conditional_discriminator/metrics/clip_probe",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, _ = parser.parse_known_args()
if args_cli.stochastic:
    args_cli.action_mode = "stochastic"
args_cli.headless = True
app = AppLauncher(args_cli).app

import inspect  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from rsl_rl.algorithms.ppo_parkour import PPOParkour as _VendoredPPO  # noqa: E402
from rsl_rl.modules import AMPDiscriminator  # noqa: E402
from rsl_rl.modules.amp_diffusion_discriminator import AMPDiffusionDiscriminator  # noqa: E402
from rsl_rl.runners import OnPolicyRunnerParkourAMP  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

TASK = "Leg-Imitation-Tracking-RMA-v0"

# run 의 env cfg 에서 복원할 키. ★ `action_scale` 이 빠지면 **조용히** 다른 정책을 재게 된다.
_ENV_KEYS = (
    "lin_vel_x_min", "lin_vel_x_max", "lin_vel_y_min", "lin_vel_y_max",
    "yaw_vel_min", "yaw_vel_max", "motion_file", "motion_weight_mode",
    "vel_err_scale", "yaw_vel_err_scale", "reset_strategy", "rel_stand_envs", "cmd_deadzone",
    "rsi_match_command", "rsi_match_temperature", "resample_command_in_episode",
    "tar_change_time_min", "tar_change_time_max", "stand_reset_joint_noise",
    "action_scale", "episode_length_s", "early_termination", "termination_height",
    "include_rel_track_obs", "num_amp_observations", "amp_observation_space",
    "amp_cond_mode", "amp_cond_v_max", "amp_cond_yaw_max",
    "amp_cond_expert_sampling", "amp_cond_match_temperature",
    # dof_vel ablation. 빠지면 disc 입력 차원이 어긋나 체크포인트 로드가 실패한다.
    "amp_drop_dof_vel",
)
# env 가 런타임에 계산하는 값은 agent.yaml 에서 얹으면 안 된다(590 vs 592 로 체크포인트를 못 읽는다).
_AGENT_SKIP = {"amp_observation_space", "motion_files", "num_amp_observations"}


def _load_yaml(path):
    with open(path) as f:
        return yaml.unsafe_load(f)


env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args_cli.n_envs)
_saved_env = _load_yaml(args_cli.run_params)
for _k in _ENV_KEYS:
    if _k in _saved_env and hasattr(env_cfg, _k):
        setattr(env_cfg, _k, _saved_env[_k])

# 명령은 우리가 매 스텝 직접 쓴다 — env 의 재샘플이 끼어들면 다른 명령을 잰다.
env_cfg.seed = args_cli.seed
torch.manual_seed(args_cli.seed)
np.random.seed(args_cli.seed)
env_cfg.resample_command_in_episode = False
env_cfg.tar_change_time_min = 1.0e6
env_cfg.tar_change_time_max = 1.0e6
if args_cli.start == "stand":
    env_cfg.reset_strategy = "random_stand"
    env_cfg.rel_stand_envs = 1.0
else:
    env_cfg.reset_strategy = "random"  # "stand" 미포함 → 전 env RSI
    env_cfg.rel_stand_envs = 0.0

agent_cfg = load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point")
_own_agent_yaml = os.path.join(os.path.dirname(args_cli.run_params), "agent.yaml")
_saved_agent = _load_yaml(_own_agent_yaml)
for _sec in ("algorithm", "policy", "amp", "estimator"):
    _src = _saved_agent.get(_sec) if isinstance(_saved_agent, dict) else None
    _dst = getattr(agent_cfg, _sec, None)
    if not isinstance(_src, dict) or _dst is None:
        continue
    _is_map = isinstance(_dst, dict)
    for _k, _v in _src.items():
        if _k in _AGENT_SKIP:
            continue
        if not _is_map and not hasattr(_dst, _k):
            continue
        _cur = _dst.get(_k, None) if _is_map else getattr(_dst, _k, None)
        if _cur != _v:
            print(f">>> agent.{_sec}.{_k}: {_cur} -> {_v}")
            if _is_map:
                _dst[_k] = _v
            else:
                setattr(_dst, _k, _v)

dt = 1.0 / env_cfg.policy_dt_hz
rec_steps = int(round(args_cli.dur_s / dt))
n_hist = env_cfg.num_amp_observations
warmup = n_hist

env = gym.make(TASK, cfg=env_cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

_accepted = set(inspect.signature(_VendoredPPO.__init__).parameters.keys()) - {"self"}
_cfg = agent_cfg.to_dict()
_cfg["algorithm"] = {k: v for k, v in _cfg["algorithm"].items() if k in _accepted or k == "class_name"}
runner = OnPolicyRunnerParkourAMP(env, _cfg, log_dir=None, device=agent_cfg.device)
runner.load(args_cli.checkpoint)
policy = runner.get_inference_policy(device=env.unwrapped.device)

# ── 행동 경로 ────────────────────────────────────────────────────
# `act_inference` 는 **history latent** 를 쓰고 GT `priv_explicit` 를 그대로 넘긴다(배포 경로).
# 학습 롤아웃은 `alg.act(hist_encoding=False)` 로 **privileged latent** + estimator 추정치를 쓰고
# 분포에서 표본을 뽑는다(dagger 주기 20 iter 중 19 번). 학습 로그의 D(policy) 는 후자를 잰 값이라
# 둘의 차이는 잡음 하나가 아니다 — `train_mean` 은 같은 경로에서 표본 추출만 뺀 대조군이다.
_alg = runner.alg


def _pick_action(o):
    if args_cli.action_mode == "mean":
        return policy(o)
    a = _alg.act(o, hist_encoding=False)
    if args_cli.action_mode == "train_mean":
        return _alg.policy.action_mean
    return a


_std = _alg.policy.std.detach() if hasattr(_alg.policy, "std") else None
ACT_STD = float(_std.mean()) if _std is not None else float("nan")
print(f">>> action_mode={args_cli.action_mode}  policy std 평균={ACT_STD:.4f}  clip_actions={agent_cfg.clip_actions}")
_SUFFIX = {"mean": "", "stochastic": "_stoch", "train_mean": "_trainmean"}[args_cli.action_mode]

base = env.unwrapped
lib = base._motion_lib
dev = base.device
D_IN = base.amp_observation_size
COND_DIM = base.amp_cond_dim
print(f">>> amp_observation_size={D_IN}  amp_cond_dim={COND_DIM}  cond_values_dim={base._amp_cond_values_dim}")


# ── 판별기 두 개 ────────────────────────────────────────────────
def _build_disc(agent_yaml_path: str, ckpt_path: str):
    amp = _load_yaml(agent_yaml_path)["amp"]
    arch = amp.get("disc_arch", "mlp")
    if arch == "mlp":
        d = AMPDiscriminator(
            input_dim=D_IN,
            hidden_dims=amp.get("discriminator_hidden_dims", [1024, 512]),
            device=str(dev),
            disc_reward_type=amp.get("disc_reward_type", "ls_gan"),
            norm_clip=amp.get("disc_norm_clip"),
            cond_dim=COND_DIM,
        )
    elif arch == "drail":
        d = AMPDiffusionDiscriminator(
            input_dim=D_IN,
            hidden_dims=amp.get("drail_hidden_dims", [256, 256, 256, 256]),
            activation=amp.get("drail_activation", "elu"),
            device=str(dev),
            disc_reward_type=amp.get("disc_reward_type", "bce"),
            norm_clip=amp.get("disc_norm_clip"),
            cond_dim=COND_DIM,
            label_dim=amp.get("drail_label_dim", 10),
            diffusion_steps=amp.get("drail_diffusion_steps", 1000),
            sample_strategy=amp.get("drail_sample_strategy", "antithetic"),
            sample_strategy_value=amp.get("drail_sample_strategy_value", 0),
            paired_noise=amp.get("drail_paired_noise", True),
        )
    else:
        raise ValueError(f"모르는 disc_arch: {arch}")
    ck = torch.load(ckpt_path, map_location=str(dev), weights_only=False)
    d.load_state_dict(ck["discriminator_state_dict"])  # strict — 차원이 어긋나면 시끄럽게 죽는다
    d.to(dev).eval()
    return arch, d


_own_arch = _saved_agent["amp"].get("disc_arch", "mlp")
own_disc = runner.alg.discriminator
own_disc.eval()  # ★ runner.load 도 get_inference_policy 도 disc 를 eval 로 내리지 않는다
DISCS = {_own_arch: own_disc}
print(f">>> 자기 run 판별기: {_own_arch}")

if args_cli.other_checkpoint:
    _other_params = args_cli.other_run_params or os.path.join(
        os.path.dirname(args_cli.other_checkpoint), "params", "env.yaml"
    )
    _other_agent_yaml = os.path.join(os.path.dirname(_other_params), "agent.yaml")
    try:
        _o_arch, _o_disc = _build_disc(_other_agent_yaml, args_cli.other_checkpoint)
        DISCS[_o_arch] = _o_disc
        print(f">>> 상대 run 판별기 로드 성공: {_o_arch}  ({args_cli.other_checkpoint})")
    except Exception as exc:  # noqa: BLE001
        print(f"!!! 상대 판별기 로드 실패 — 자기 판별기만 진행: {type(exc).__name__}: {exc}")


def _score(disc, x: torch.Tensor) -> np.ndarray:
    reps = args_cli.drail_reps if isinstance(disc, AMPDiffusionDiscriminator) else 1
    out = torch.zeros(x.shape[0], device=dev)
    with torch.inference_mode():
        for _ in range(reps):
            probs = []
            for i in range(0, x.shape[0], 4096):
                probs.append(torch.sigmoid(disc.get_logits(x[i : i + 4096].to(dev))).view(-1))
            out += torch.cat(probs)
    return (out / reps).cpu().numpy().astype(np.float32)


# ── 클립 고정 패치 ──────────────────────────────────────────────
_orig_sample_motions = lib.sample_motions
_target = {"id": None}


def _patched_sample_motions(n):
    if _target["id"] is None:
        return _orig_sample_motions(n)
    return torch.full((n,), int(_target["id"]), dtype=torch.long, device=dev)


lib.sample_motions = _patched_sample_motions

names = lib.motion_names
if args_cli.clips == "all":
    clip_ids = list(range(len(names)))
else:
    want = [c.strip() for c in args_cli.clips.split(",") if c.strip()]
    clip_ids = [names.index(c) for c in want]
print(f">>> 클립 {len(clip_ids)}개: {[names[i] for i in clip_ids]}")

VX_MIN, VX_MAX = float(env_cfg.lin_vel_x_min), float(env_cfg.lin_vel_x_max)
YAW_MIN, YAW_MAX = float(env_cfg.yaw_vel_min), float(env_cfg.yaw_vel_max)
JN = list(base._robot.data.joint_names)
LAYOUT = {
    "n_hist": int(n_hist),
    # ★ `amp_observation_space` 는 dof_vel 제거 전 값(59)이라 amp_drop_dof_vel run 에서 실제와 다르다.
    #   아래 `fields` 합으로 채운다.
    "step_dim": None,  # fields 합으로 아래에서 채운다
    "fields": ([["root_pos_xy", 2]] if env_cfg.include_rel_track_obs else [])
    + [["dof_pos", 17]]
    + ([] if getattr(env_cfg, "amp_drop_dof_vel", False) else [["dof_vel", 17]])
    + [["root_h", 1], ["lin_vel", 3], ["ang_vel", 3], ["foot_pos", 12], ["rot_tan_norm", 6]],
    "cond_dim": int(COND_DIM),
    "cond_values_dim": int(base._amp_cond_values_dim),
    "hist_order": "index0=newest",
}
LAYOUT["step_dim"] = int(sum(sz for _, sz in LAYOUT["fields"]))
out_dir = os.path.join(args_cli.out_root, args_cli.tag)
os.makedirs(out_dir, exist_ok=True)
N = args_cli.n_envs

for mid in clip_ids:
    name = names[mid]
    _target["id"] = mid
    clip_len = float(lib._motion_lengths[mid].item())

    # ── 클립 전 구간의 body-frame 속도 궤적 → 명령 ──
    n_clip = max(2, int(round(clip_len / dt)) + 1)
    t_clip = torch.arange(n_clip, device=dev, dtype=torch.float32) * dt
    t_clip = torch.clamp(t_clip, max=clip_len)
    mids_clip = torch.full((n_clip,), mid, dtype=torch.long, device=dev)
    fr = lib.calc_motion_frame(mids_clip, t_clip)
    lin_b, ang_b, dofp, dofv = fr[2], fr[3], fr[4], fr[5]
    raw_vx, raw_vy, raw_yaw = lin_b[:, 0], lin_b[:, 1], ang_b[:, 2]
    vx = raw_vx.clamp(VX_MIN, VX_MAX)
    vy = torch.zeros_like(raw_vy)  # 학습 범위가 0 고정 — 살리면 조건 열까지 분포 밖으로 나간다
    yaw = raw_yaw.clamp(YAW_MIN, YAW_MAX)
    clip_frac = float(((raw_vx < VX_MIN) | (raw_vx > VX_MAX) | (raw_yaw < YAW_MIN) | (raw_yaw > YAW_MAX)).float().mean())
    vy_dropped = float(raw_vy.abs().mean())
    expert_jpos = dofp[:, base._motion_dof_indices].cpu().numpy().astype(np.float32)
    expert_jvel = dofv[:, base._motion_dof_indices].cpu().numpy().astype(np.float32)

    # ── 롤아웃 ──
    alive = torch.ones(N, dtype=torch.bool, device=dev)
    amp_rec, meas_rec, jp_rec, jv_rec, cmd_rec, act_clip_hits = [], [], [], [], [], []
    # `--save_seq` 전용. 기존 기록은 생존 env 만 남기지만(`keep`), 시퀀스 덤프는 전 env 를 남기고
    # 대신 스텝별 `done` 을 같이 남겨 소비자가 유효 구간을 직접 자르게 한다.
    lin_rec, ang_rec, done_rec = [], [], []
    with torch.inference_mode():
        # ★ reset 도 inference_mode 안에서 불러야 한다. 롤아웃 중 만들어진 env 내부 버퍼
        # (`_proprio_history` 등)가 inference tensor 가 돼, 밖에서 in-place 로 0 을 쓰면 죽는다.
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        for step in range(warmup + rec_steps):
            k = step % n_clip
            base._lin_vel_cmd[:, 0] = vx[k]
            base._lin_vel_cmd[:, 1] = vy[k]
            base._yaw_vel_cmd[:] = yaw[k]
            actions = _pick_action(obs)
            if step >= warmup:
                act_clip_hits.append(float((actions.abs() >= agent_cfg.clip_actions).float().mean()))
            obs, _, dones, extras = env.step(actions)
            alive &= dones.view(-1) == 0
            if step >= warmup:
                amp_rec.append(extras["amp_obs"].clone())
                d = base._robot.data
                meas_rec.append(
                    torch.stack([d.root_link_lin_vel_b[:, 0], d.root_link_lin_vel_b[:, 1],
                                 d.root_link_ang_vel_b[:, 2]], dim=-1)
                )
                jp_rec.append(d.joint_pos.torch.clone())
                jv_rec.append(d.joint_vel.torch.clone())
                cmd_rec.append([float(vx[k]), float(vy[k]), float(yaw[k])])
                if args_cli.save_seq:
                    lin_rec.append(d.root_link_lin_vel_b.clone())
                    ang_rec.append(d.root_link_ang_vel_b.clone())
                    done_rec.append(dones.view(-1).clone())

    keep = alive.nonzero(as_tuple=False).flatten()
    n_kept, n_dropped = int(keep.numel()), int(N - alive.sum().item())
    cmd_arr = np.asarray(cmd_rec, dtype=np.float32)  # [T, 3]

    if n_kept == 0:
        print(f"!!! {name} ({args_cli.start}): 생존 env 0 — 빈 npz 저장")
        payload = dict(
            policy_obs=np.zeros((0, D_IN), np.float32), expert_obs=np.zeros((0, D_IN), np.float32),
            policy_jvel=np.zeros((0, rec_steps, 17), np.float32),
            policy_jpos=np.zeros((0, rec_steps, 17), np.float32),
            meas=np.zeros((0, rec_steps, 3), np.float32),
        )
    else:
        amp_all = torch.stack(amp_rec, dim=1)[keep]  # [K, T, D]
        meas_all = torch.stack(meas_rec, dim=1)[keep]
        jp_all = torch.stack(jp_rec, dim=1)[keep]
        jv_all = torch.stack(jv_rec, dim=1)[keep]
        flat = amp_all.reshape(-1, D_IN)
        n_take = min(args_cli.max_samples, flat.shape[0])
        # ★ 등간격 추출은 stride 가 보행 주기와 맞물려 특정 위상만 골라낼 수 있다 → 무작위.
        sel = torch.randperm(flat.shape[0], device=dev)[:n_take]
        pol_obs = flat[sel].contiguous()

        # expert: 같은 클립, 시각은 히스토리 창이 다 차는 구간에서 균등
        t_lo = min((n_hist - 1) * dt, clip_len)
        e_times = np.linspace(t_lo, clip_len, n_take, dtype=np.float32)
        e_ids = torch.full((n_take,), mid, dtype=torch.long, device=dev)
        with torch.inference_mode():
            exp_obs = base.collect_reference_motions(n_take, e_times, e_ids).contiguous()
            # ★ expert 조건 열을 policy 와 **같은 자**로 맞춘다. `collect_reference_motions` 는
            # motion_ids 를 주면 `_expert_amp_cond` 를 타 **클립 평균 속도**(클립당 상수) 라벨을 붙이는데,
            # policy 쪽은 `_policy_amp_cond` 가 매 프레임 `‖lin_vel_cmd‖/v_max` 를 넣는다. 조건부 D 의
            # 존재 이유인 그 한 열에서 두 표본이 다른 조건에 놓이면 D 비교가 성립하지 않는다.
            # (학습 때 expert 도 `command_matched` 라 클립 평균이 아닌 뽑힌 명령값을 썼다.)
            kin = D_IN - COND_DIM
            k_e = np.clip(np.rint(e_times / dt).astype(np.int64), 0, n_clip - 1)
            exp_obs[:, kin] = vx[torch.as_tensor(k_e, device=dev)] / base._amp_cond_scale[0]
            exp_obs[:, kin + 1] = 1.0

        payload = dict(
            policy_obs=pol_obs.cpu().numpy().astype(np.float32),
            expert_obs=exp_obs.cpu().numpy().astype(np.float32),
            policy_jvel=jv_all.cpu().numpy().astype(np.float32),
            policy_jpos=jp_all.cpu().numpy().astype(np.float32),
            meas=meas_all.cpu().numpy().astype(np.float32),
        )
        for arch, disc in DISCS.items():
            payload[f"d_{arch}_policy"] = _score(disc, pol_obs)
            payload[f"d_{arch}_expert"] = _score(disc, exp_obs)

        err = (meas_all - torch.tensor(cmd_arr, device=dev).unsqueeze(0)).abs().mean(dim=(0, 1))
        cmd_vx_mean = float(np.abs(cmd_arr[:, 0]).mean())
        ach = float((meas_all[..., 0].mean() / cmd_vx_mean).item()) if cmd_vx_mean > 1e-6 else float("nan")
        d_txt = "  ".join(
            f"D_{a}(e/p)={payload[f'd_{a}_expert'].mean():.3f}/{payload[f'd_{a}_policy'].mean():.3f}" for a in DISCS
        )
        print(
            f">>> {name:26s} [{args_cli.start}] kept={n_kept}/{N} |err| vx={err[0]:.3f} vy={err[1]:.3f}"
            f" yaw={err[2]:.3f}  달성률={ach:.3f}  jvelRMS={float(jv_all.pow(2).mean().sqrt()):.3f}  {d_txt}"
        )

    payload.update(
        layout=json.dumps(LAYOUT), dt=np.float32(dt), joint_names=np.array(JN),
        expert_jvel=expert_jvel, expert_jpos=expert_jpos, cmd=cmd_arr,
        clip_name=name, clip_mean_speed=np.float32(lib.motion_mean_speeds[mid].item()),
        clip_len=np.float32(clip_len), start_mode=args_cli.start,
        n_dropped=np.int32(n_dropped), n_kept=np.int32(n_kept), n_envs=np.int32(N),
        clip_frac=np.float32(clip_frac), vy_dropped_mean=np.float32(vy_dropped),
        drail_reps=np.int32(args_cli.drail_reps), checkpoint=args_cli.checkpoint,
        seed=np.int32(args_cli.seed), expert_cond_source="per_frame_command",
        action_mode=args_cli.action_mode, action_std=np.float32(ACT_STD),
        act_clip_frac=np.float32(np.mean(act_clip_hits) if act_clip_hits else np.nan),
        other_checkpoint=str(args_cli.other_checkpoint),
    )
    if args_cli.save_seq and done_rec:
        # ★ `valid[n, t]` = 스텝 t 까지 그 env 가 한 번도 리셋되지 않았다. 누적 `alive` 마스크가
        # 아니라 스텝별 done 에서 만든다 — 누적 마스크는 "언제" 끊겼는지를 잃는다.
        _done = torch.stack(done_rec, dim=1).bool()  # [N, T]
        _valid = torch.cumsum(_done.to(torch.int32), dim=1) == 0
        _seq = dict(
            policy_obs_seq=torch.stack(amp_rec, dim=1).to(torch.float16).cpu().numpy(),
            policy_jpos_seq=torch.stack(jp_rec, dim=1).to(torch.float16).cpu().numpy(),
            policy_jvel_seq=torch.stack(jv_rec, dim=1).to(torch.float16).cpu().numpy(),
            policy_root_lin_vel_seq=torch.stack(lin_rec, dim=1).cpu().numpy().astype(np.float32),
            policy_root_ang_vel_seq=torch.stack(ang_rec, dim=1).cpu().numpy().astype(np.float32),
            valid=_valid.cpu().numpy(),
            done_seq=_done.cpu().numpy(),
        )
        _seq.update(
            layout=json.dumps(LAYOUT), dt=np.float32(dt), joint_names=np.array(JN), cmd=cmd_arr,
            clip_name=name, clip_len=np.float32(clip_len), start_mode=args_cli.start,
            n_envs=np.int32(N), checkpoint=args_cli.checkpoint, seed=np.int32(args_cli.seed),
            action_mode=args_cli.action_mode, expert_obs=payload["expert_obs"],
            expert_jpos=expert_jpos, expert_jvel=expert_jvel,
        )
        _seq_out = os.path.join(out_dir, f"{name}_{args_cli.start}{_SUFFIX}_seq.npz")
        np.savez_compressed(_seq_out, **_seq)
        print(f"    seq 저장: {_seq_out}  obs={_seq['policy_obs_seq'].shape} valid={int(_valid.sum())}/{_valid.numel()}")

    out = os.path.join(out_dir, f"{name}_{args_cli.start}{_SUFFIX}.npz")
    np.savez_compressed(out, **payload)
    print(f"    저장: {out}  keys={sorted(payload.keys())}" if mid == clip_ids[0] else f"    저장: {out}")

env.close()
app.close()

"""R.pet 험지 zero-shot — 교정판.

1차 스윕의 두 결함
-------------------
① `--yaw` 를 [-1.5, 1.5] 랜덤으로 둬서 로봇이 제자리 선회를 했을 수 있다.
② `play_in_isaaclab.py` 요약은 **속도만** 재고 위치·이동거리를 안 잡는다.
IsaacLab 의 gap/stepping 지형은 둘 다 **중앙에 평평한 플랫폼**이 있고(docstring 확인) 로봇은
그 중심에서 스폰한다. 따라서 ①②가 겹치면 "지형을 만나지 않은 결과"를 성능으로 오독한다.
(Go2 에서 '주차 구간 오염'으로 세 번 틀렸던 것과 같은 계열의 계측 누락이다.)

교정
----
- yaw 를 0 으로 고정해 직선 주행만 본다.
- base 변위를 env 별로 기록하고, **플랫폼 반폭을 넘어간 env 비율**을 함께 보고한다.
  이것이 "지형을 실제로 만났나"의 판정이다. 넘지 못했으면 그 셀의 추종률은 무의미하다.
- 낙상은 누적 termination 이 아니라 **env 당 1회 이상 종료 비율**로도 함께 낸다.
"""
import argparse, os, json

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Leg-Imitation-Tracking-RMA-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--warmup", type=int, default=10)
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--platform_half", type=float, default=1.0, help="중앙 평지 반폭 [m] — 이탈 판정 기준")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import inspect, torch, yaml, gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms.ppo_parkour import PPOParkour
from rsl_rl.runners import OnPolicyRunnerParkourAMP


# env.yaml 에는 python/tuple 뿐 아니라 builtins.slice 같은 객체 태그도 들어 있다.
# 체크포인트 자신의 params 파일(신뢰 가능)이므로 unsafe_load 로 그대로 읽는다.
run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
agent_cfg = yaml.safe_load(open(os.path.join(run_dir, "params", "agent.yaml")))
run_env = yaml.unsafe_load(open(os.path.join(run_dir, "params", "env.yaml")))

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
# 학습 조건 반영 (play 와 동일 규약)
for k in ("last_action_obs", "num_amp_observations", "amp_observation_space",
          "include_rel_track_obs", "amp_drop_dof_vel", "amp_cond_mode",
          "motion_file", "vel_err_scale", "lin_vel_x_max"):
    if k in run_env:
        setattr(cfg, k, run_env[k])
# ★명령 고정: vx 한 점 + yaw 0 (직선). 범위 자체를 좁혀야 4s 재샘플 타이머와 싸우지 않는다.
cfg.lin_vel_x_min = cfg.lin_vel_x_max = float(args.vx)
cfg.yaw_vel_min = cfg.yaw_vel_max = 0.0
cfg.episode_length_s = 1e6          # 종료는 낙상으로만 — time_out 으로 끊기지 않게

env = gym.make(args.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=float(agent_cfg.get("clip_actions", 4.0)))
u = env.unwrapped
dev = u.device

acfg = dict(agent_cfg)
acfg.setdefault("algorithm", {}).setdefault("rnd_cfg", None)
acfg["algorithm"].setdefault("symmetry_cfg", None)
ok = set(inspect.signature(PPOParkour.__init__).parameters)
acfg["algorithm"] = {k: v for k, v in acfg["algorithm"].items() if k in ok or k == "class_name"}
runner = OnPolicyRunnerParkourAMP(env, acfg, log_dir=None, device=str(dev))
runner.load(os.path.abspath(args.checkpoint), load_optimizer=False)
policy = runner.get_inference_policy(device=dev)

obs = env.get_observations().to(dev)
p0 = u._robot.data.root_link_pos_w.torch[:, :2].clone()
far = torch.zeros(args.num_envs, device=dev)        # env 별 최대 변위
died_any = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)
terms = 0
vx_sum = torch.zeros(1, device=dev)
n = 0
for t in range(args.steps):
    with torch.inference_mode():
        a = policy(obs)
    obs, _, dones, _ = env.step(a)
    obs = obs.to(dev)
    pos = u._robot.data.root_link_pos_w.torch[:, :2]
    far = torch.maximum(far, (pos - p0).norm(dim=-1))
    if t >= args.warmup:
        vx_sum += u._robot.data.root_link_lin_vel_b.torch[:, 0].mean()
        n += 1
    d = dones.bool().flatten()
    if d.any():
        terms += int(d.sum())
        died_any |= d
        p0 = torch.where(d[:, None], pos, p0)       # 리셋된 env 는 변위 기준점 재설정
        far = torch.where(d, torch.zeros_like(far), far)

out = dict(
    vx_cmd=args.vx,
    vx_gt=float(vx_sum.item() / max(1, n)),
    far_mean=float(far.mean()), far_max=float(far.max()),
    left_platform_frac=float((far > args.platform_half).float().mean()),
    terminations=terms,
    died_env_frac=float(died_any.float().mean()),
    num_envs=args.num_envs, steps=args.steps,
)
print("RESULT_JSON " + json.dumps(out))
env.close()
app.close()

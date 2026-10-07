"""R.pet 걸음(gait) 분류 — 험지에서 pace/gallop 이 유지되는지 측정.

왜 이것이 필요한가
------------------
TAMOLS 는 fixed-timing TO 라 **접촉 스케줄을 외부에서 받아야** QP 가 성립한다. Go2 에서는
WTW 고정 게이트 스케줄러가 그걸 공급했다. R.pet 정책은 게이트가 **창발**이고 문서 실측이
저속 pace 70~71% / 고속 gallop 81~87% 다(MODEL_SELECTION.md:57-59). pace 는 지지선이 몸통
한쪽의 전후선이라 공칭 CoM 투영을 지나지 않고, gallop 은 비행상이 있어 지지폴리곤이
공집합인 구간이 생긴다 — 둘 다 GIAC 전제를 깬다. 따라서 **험지에서 어떤 걸음이 나오는지**가
TAMOLS 결합의 제1 관문이다.

분류 방법
---------
인계본에 분류기 구현(`summarize.py`)이 없어 새로 만든다. 문서가 "thigh 위상 기울기 기반"이라
적었으므로 같은 신호(thigh 관절각)를 쓰되, 위상은 FFT 주파수 영역에서 뽑는다:
  1) 창(기본 2 s) 안 4 다리 thigh 각을 detrend
  2) 4 다리 평균 파워스펙트럼의 최대 bin = 공통 보행 주파수
  3) 그 bin 에서 각 다리 위상 φ_i → FL 기준 Δφ 를 [0,1) 로 래핑
  4) 템플릿 최근접 분류
        trot  : FL 0 · FR .5 · HL .5 · HR 0     (대각쌍 동위상)
        pace  : FL 0 · FR .5 · HL 0  · HR .5    (같은쪽 동위상)
        bound/gallop : FL 0 · FR 0 · HL .5 · HR .5  (앞쌍·뒷쌍 각각 동위상)
     진폭이 임계 미만이면 stand, 최근접 거리가 임계 초과면 기타.

★검증 규약: 평지에서 문서값(저속 pace 70~71%, 고속 gallop 81~87%)을 재현하는지 먼저 본다.
재현하지 못하면 이 분류기로 험지를 판정하지 않는다.
"""
import argparse, os, json

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--task", default="Leg-Imitation-Tracking-RMA-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--win_s", type=float, default=2.0, help="분류 창 [s] (마지막 구간)")
parser.add_argument("--vx", type=float, default=1.0)
parser.add_argument("--yaw", type=float, default=0.0)
parser.add_argument("--amp_thresh", type=float, default=0.05, help="stand 판정 thigh 진폭 [rad]")
parser.add_argument("--dist_thresh", type=float, default=0.18, help="템플릿 최근접 거리 상한(초과=기타)")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import inspect, torch, yaml, gymnasium as gym, numpy as np
import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.algorithms.ppo_parkour import PPOParkour
from rsl_rl.runners import OnPolicyRunnerParkourAMP

run_dir = os.path.dirname(os.path.abspath(args.checkpoint))
agent_cfg = yaml.safe_load(open(os.path.join(run_dir, "params", "agent.yaml")))
run_env = yaml.unsafe_load(open(os.path.join(run_dir, "params", "env.yaml")))

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
for k in ("last_action_obs", "num_amp_observations", "amp_observation_space",
          "include_rel_track_obs", "amp_drop_dof_vel", "amp_cond_mode",
          "motion_file", "vel_err_scale", "lin_vel_x_max"):
    if k in run_env:
        setattr(cfg, k, run_env[k])
cfg.lin_vel_x_min = cfg.lin_vel_x_max = float(args.vx)
cfg.yaw_vel_min = cfg.yaw_vel_max = float(args.yaw)
cfg.episode_length_s = 1e6

env = gym.make(args.task, cfg=cfg, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=float(agent_cfg.get("clip_actions", 4.0)))
u = env.unwrapped
dev = u.device

acfg = dict(agent_cfg)
acfg.setdefault("algorithm", {}).setdefault("rnd_cfg", None)
acfg["algorithm"].setdefault("symmetry_cfg", None)
okp = set(inspect.signature(PPOParkour.__init__).parameters)
acfg["algorithm"] = {k: v for k, v in acfg["algorithm"].items() if k in okp or k == "class_name"}
runner = OnPolicyRunnerParkourAMP(env, acfg, log_dir=None, device=str(dev))
runner.load(os.path.abspath(args.checkpoint), load_optimizer=False)
policy = runner.get_inference_policy(device=dev)

# thigh 관절 인덱스 (이름 기반 — URDF 순서와 무관하게 안전)
jn = list(u._robot.joint_names)
order = ["FL", "FR", "HL", "HR"]
idx = []
for leg in order:
    cand = [i for i, n in enumerate(jn) if n.startswith(leg + "_thigh")]
    assert len(cand) == 1, f"{leg}_thigh 관절을 찾지 못함: {jn}"
    idx.append(cand[0])
print(f"[gait] thigh 관절 인덱스 {dict(zip(order, idx))}")

dt = u.step_dt
nwin = int(args.win_s / dt)
buf = torch.zeros(args.steps, args.num_envs, 4, device=dev)
died = torch.zeros(args.num_envs, dtype=torch.bool, device=dev)

obs = env.get_observations().to(dev)
for t in range(args.steps):
    with torch.inference_mode():
        a = policy(obs)
    obs, _, dones, _ = env.step(a)
    obs = obs.to(dev)
    buf[t] = u._robot.data.joint_pos.torch[:, idx]
    d = dones.bool().flatten()
    if d.any():
        died |= d

# ── 분류 ──────────────────────────────────────────────────────────────────────
w = buf[-nwin:].cpu().numpy()                      # (T, N, 4)
w = w - w.mean(axis=0, keepdims=True)
F = np.fft.rfft(w, axis=0)
P = (np.abs(F) ** 2).mean(axis=2)                  # (f, N) 4다리 평균 파워
P[0] = 0.0                                         # DC 제외
kbest = P.argmax(axis=0)                           # env 별 지배 주파수 bin
N = w.shape[1]
amp = np.abs(w).max(axis=0).mean(axis=1)           # env 별 thigh 진폭 평균
ph = np.zeros((N, 4))
for i in range(N):
    ph[i] = np.angle(F[kbest[i], i, :])
dphi = ((ph - ph[:, [0]]) / (2 * np.pi)) % 1.0      # FL 기준 [0,1)

# ── 쌍 구조 기반 분류 (임의 임계 제거) ───────────────────────────────────────
# 템플릿 최근접은 gallop 을 놓친다: 실제 gallop 은 앞쌍/뒷쌍 **안에서도** 선행-후행 차가
# 있어(Δφ 0.1~0.2) 이상 템플릿 [0,0,.5,.5] 와 거리가 벌어진다. 평지 vx3.0 에서 문서의
# gallop 81~87% 가 "기타 87.5%" 로 빠진 것이 그 증상이다.
# 대신 **어느 쌍이 함께 움직이는가**로 판정한다 — 쌍 안의 lead-lag 에 둔감하다.
#   pace  : 같은쪽 쌍 (FL,HL) · (FR,HR)
#   trot  : 대각 쌍   (FL,HR) · (FR,HL)
#   bound/gallop : 앞쌍 (FL,FR) · 뒷쌍 (HL,HR)
def circ(a, b):
    d = np.abs(a - b) % 1.0
    return np.minimum(d, 1.0 - d)


PAIRS = {
    "pace": ((0, 2), (1, 3)),            # FL-HL, FR-HR
    "trot": ((0, 3), (1, 2)),            # FL-HR, FR-HL
    "bound/gallop": ((0, 1), (2, 3)),    # FL-FR, HL-HR
}

labels = []
scores = []
for i in range(N):
    if amp[i] < args.amp_thresh:
        labels.append("stand"); scores.append(0.0); continue
    best, bd = None, 9e9
    for k, (p1, p2) in PAIRS.items():
        # 쌍 내부는 가깝고(동위상), 쌍 사이는 멀어야(반위상) 한다
        intra = 0.5 * (circ(dphi[i, p1[0]], dphi[i, p1[1]]) + circ(dphi[i, p2[0]], dphi[i, p2[1]]))
        inter = circ(dphi[i, p1[0]], dphi[i, p2[0]])
        d = intra + (0.5 - inter)            # 작을수록 그 패턴
        if d < bd:
            best, bd = k, d
    labels.append(best); scores.append(float(bd))

from collections import Counter
c = Counter(labels)
freq = float(np.median(kbest) / (nwin * dt))
out = dict(vx=args.vx, num_envs=N, win_s=args.win_s,
           gait={k: round(100.0 * v / N, 1) for k, v in c.most_common()},
           gait_freq_hz=round(freq, 2),
           thigh_amp_rad=round(float(amp.mean()), 3),
           pair_score_med=round(float(np.median(scores)), 3),
           died_env_frac=round(float(died.float().mean()), 3))
print("GAIT_JSON " + json.dumps(out, ensure_ascii=False))
env.close()
app.close()

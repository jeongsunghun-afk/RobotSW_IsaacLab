"""R.pet 접촉 센서가 실제로 값을 읽는지 + 종료 조건이 발동하는지 검증.

왜: parity 실행에서 ContactSensor 가 '/World/envs/env_*/Robot/Geometry/Base/Base' 에서
articulation/rigid body/contact report API 를 못 찾는다는 경고가 다수 났다. prim_path 가
'/Robot/.*' 와일드카드라 강체 아닌 prim 까지 매칭한 과매칭 잡음일 수 있으나,
09-09 조사 §G-4 가 "종료 조건의 contact sensor 가 사실상 무력"이라고 경고한 바 있고
hind_leg 에서 같은 계열 결함이 "에피소드 92.9% 를 누운 채" 보내게 만든 전례가 있다.
평지에서는 낙상이 없어 드러나지 않지만 험지 판정은 접촉에 걸리므로 먼저 확인한다.
"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=120)
parser.add_argument("--task", default="Leg-Imitation-Tracking-RMA-v0")
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch, gymnasium as gym
import isaaclab_tasks  # noqa: F401  (태스크 등록)
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
cfg.episode_length_s = 1e6          # 리셋으로 통계가 끊기지 않게
env = gym.make(args.task, cfg=cfg, render_mode=None)
u = env.unwrapped

cs = u.scene["contact_sensor"]
bodies = cs.body_names
print(f"\n[접촉센서] 추적 body {len(bodies)}개")
feet = [i for i, n in enumerate(bodies) if "foot" in n.lower()]
print("  발 관련 body:", [bodies[i] for i in feet][:10])

obs, _ = env.reset()
act = torch.zeros((args.num_envs, u.action_space.shape[-1]), device=u.device)
fmax = torch.zeros(len(bodies), device=u.device)
nz_steps = 0
for t in range(args.steps):
    obs, rew, term, trunc, info = env.step(act)
    f = cs.data.net_forces_w.norm(dim=-1)          # (N, B)
    fmax = torch.maximum(fmax, f.max(dim=0).values)
    if f.max() > 1.0:
        nz_steps += 1

print(f"\n[결과] {args.steps} 스텝 (영-액션 = 무릎 꺾여 주저앉음 예상)")
print(f"  접촉력 > 1N 인 스텝 수 : {nz_steps}/{args.steps}")
top = torch.argsort(fmax, descending=True)[:8]
print("  body 별 최대 접촉력 |F| [N]:")
for i in top.tolist():
    print(f"     {bodies[i]:<28} {fmax[i].item():8.2f}")
print(f"\n  판정: {'✓ 접촉센서 작동' if nz_steps > 0 else '✗ 접촉력이 전 스텝 0 — 센서 무력'}")
print(f"  종료 발동 : term={int(term.sum())} trunc={int(trunc.sum())}")
env.close()
app.close()

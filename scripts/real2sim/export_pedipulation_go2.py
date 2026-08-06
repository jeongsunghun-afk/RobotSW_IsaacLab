# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""go2_pedipulation 정책을 r2s_go2 GUI Pedipulation 모드용 jit 로 export.

tracking 쪽(`export_deployable_go2.py`)과 달리 이 정책은 estimator/history_encoder가 없는
**plain MLPModel** 이라 rsl_rl 표준 export(`OnPolicyRunner.export_policy_to_jit`)를 그대로 쓴다
(`export_recovery_go2.py` 와 같은 구조). 직접 `actor(normalizer(obs))` 로 wrapper 를 짜면 안 된다
— rsl_rl 5.0.1 의 `MLPModel` 은 MLP 출력이 분포 파라미터라 결정론적 행동을 얻으려면
`distribution.deterministic_output` 을 거쳐야 하고, 표준 export(`_TorchMLPModel`)가
normalizer → mlp → deterministic_output 순서를 이미 담고 있다.

    입력: obs (B, 83)    ← policy 관측 한 벌
    출력: action (B, 28) ← a_loc(12) + a_man(12) + a_stiffness(4, 미사용)

★ **actor 는 history 를 보지 않는다 — `agent.yaml` 을 믿지 말 것.** 그 파일은
``obs_groups.policy = [policy, history]`` 라고 적혀 있지만, 채택본의 실제 가중치는
``actor.mlp.0.weight`` 가 **(512, 83)** 이라 policy 한 벌만 받는다(critic 은 943 =
83+830+30 으로 history·priv 를 다 본다). 이 스크립트는 체크포인트에서 actor 입력 차원을
읽어 판정하고, policy 만 쓰는 런이면 `obs_groups` 를 그에 맞게 덮어써서 빌드한다.
**가중치가 진실이고 yaml 은 참고다.**

obs 레이아웃과 액션 해석은 `scripts/real2sim/r2s_go2/pedipulation_runtime.py` 참고.
tracking(42)·recovery(42)와 달리 차원이 83/28 이라 파일을 바꿔 끼우면 런타임 warmup 이 잡는다.

Isaac 앱 없이 순수 torch + rsl_rl 로 로드한다(mock env). conda isaac-6.0 python.

실행:
    ./isaaclab.sh -p scripts/real2sim/export_pedipulation_go2.py \
        --run_dir logs/rsl_rl/go2_pedipulation/2026-08-03_15-53-29_hipscale_scratch_s10x \
        --checkpoint model_19999.pt --device cpu
    # → <run_dir>/exported/pedipulation_policy.pt  (torch.jit)
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile

import torch
import yaml

NUM_ACTIONS = 28
OBS_DIM_DEFAULT = 83
HISTORY_LEN_DEFAULT = 10


def _load_runner(run_dir: str, ckpt: str, device: str):
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict

    ckpt_path = os.path.join(run_dir, ckpt)
    with open(os.path.join(run_dir, "params", "agent.yaml")) as f:
        tc = yaml.safe_load(f)

    obs_dim, history_len, num_actions = OBS_DIM_DEFAULT, HISTORY_LEN_DEFAULT, NUM_ACTIONS
    env_yaml_path = os.path.join(run_dir, "params", "env.yaml")
    if os.path.isfile(env_yaml_path):
        with open(env_yaml_path) as f:
            _env = yaml.unsafe_load(f)
        obs_dim = _env.get("observation_space", OBS_DIM_DEFAULT)
        history_len = _env.get("history_len", HISTORY_LEN_DEFAULT)
        num_actions = _env.get("action_space", NUM_ACTIONS)
    history_dim = obs_dim * history_len
    print(f"[INFO] obs_dim={obs_dim} history_len={history_len} → 입력 {obs_dim + history_dim}, 액션 {num_actions}")

    tc["algorithm"].setdefault("rnd_cfg", None)
    tc["algorithm"].setdefault("symmetry_cfg", None)

    # priv 그룹 차원은 critic 만 쓰므로 체크포인트에서 역산한다 — env.yaml 에 없는 값이라
    # 하드코딩하면 런이 바뀔 때 조용히 어긋난다.
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    critic_in = sd["critic_state_dict"]["mlp.0.weight"].shape[1]
    actor_in = sd["actor_state_dict"]["mlp.0.weight"].shape[1]
    priv_dim = critic_in - obs_dim - history_dim
    print(f"[INFO] actor 입력={actor_in}, critic 입력={critic_in} → priv_dim={priv_dim}")

    # ⚠ actor 가 실제로 무엇을 받는지는 **가중치가 진실**이다. agent.yaml 의
    #   `obs_groups.policy = [policy, history]` 를 그대로 믿으면 안 된다 — 이 런의 actor 는
    #   83(=policy 만)이라 history 를 보지 않는다. 배포 런타임의 입력 계약이 여기서 갈린다.
    if actor_in == obs_dim:
        actor_uses_history = False
    elif actor_in == obs_dim + history_dim:
        actor_uses_history = True
    else:
        raise RuntimeError(
            f"actor 입력 {actor_in} 이 policy({obs_dim}) 도 policy+history({obs_dim + history_dim}) 도 아니다"
        )
    print(f"[INFO] actor_uses_history={actor_uses_history}")
    if not actor_uses_history:
        # yaml 대로 빌드하면 actor 가 913 입력으로 만들어져 load 에서 shape mismatch 로 죽는다.
        tc["obs_groups"]["policy"] = ["policy"]

    class MockEnv:
        """rsl_rl OnPolicyRunner 가 네트워크를 빌드하는 데 필요한 최소 표면만 흉내낸다."""

        def __init__(self):
            self.num_envs, self.num_actions, self.device = 1, num_actions, device
            self.cfg = type("C", (), {})()
            self.max_episode_length = 1000

        @property
        def unwrapped(self):
            return self

        def get_observations(self):
            return TensorDict(
                {
                    "policy": torch.zeros(1, obs_dim, device=device),
                    "history": torch.zeros(1, history_dim, device=device),
                    "priv": torch.zeros(1, priv_dim, device=device),
                },
                batch_size=[1],
            )

    runner = OnPolicyRunner(MockEnv(), tc, log_dir=None, device=device)
    # 체크포인트는 cuda 텐서로 저장돼 있다. map_location 없이는 GPU 없는 머신/`--device cpu` 에서
    # "Attempting to deserialize object on a CUDA device" 로 죽는다.
    runner.load(ckpt_path, map_location=device)
    return runner, obs_dim, history_dim, num_actions, actor_uses_history


def main() -> None:
    parser = argparse.ArgumentParser(description="go2_pedipulation 정책을 GUI Pedipulation 모드용 jit 로 export.")
    parser.add_argument("--run_dir", default="logs/rsl_rl/go2_pedipulation/2026-08-03_15-53-29_hipscale_scratch_s10x")
    parser.add_argument("--checkpoint", default="model_19999.pt")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="출력 경로 (기본: <run_dir>/exported/pedipulation_policy.pt)")
    args = parser.parse_args()

    from tensordict import TensorDict

    runner, obs_dim, history_dim, num_actions, uses_history = _load_runner(args.run_dir, args.checkpoint, args.device)

    out_path = args.out or os.path.join(args.run_dir, "exported", "pedipulation_policy.pt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # rsl_rl 표준 export 는 (디렉터리, 파일명)을 받고 항상 cpu 로 저장한다. 임시 디렉터리에 뽑아
    # 원하는 이름으로 옮긴다(표준 파일명 policy.pt 와 섞이지 않게).
    with tempfile.TemporaryDirectory() as tmp:
        runner.export_policy_to_jit(tmp, filename="pedipulation_policy.pt")
        shutil.move(os.path.join(tmp, "pedipulation_policy.pt"), out_path)

    # 정합성 대조: 저장된 jit vs 학습/평가와 같은 결정론적 추론 경로
    torch.manual_seed(0)
    obs = torch.randn(4, obs_dim)
    obs_td = {"policy": obs}
    jit_in = obs
    if uses_history:
        hist = torch.randn(4, history_dim)
        obs_td["history"] = hist
        jit_in = torch.cat([obs, hist], dim=-1)
    policy = runner.alg.get_policy().eval().to("cpu")
    reloaded = torch.jit.load(out_path, map_location="cpu").eval()
    with torch.inference_mode():
        ref = policy(TensorDict(obs_td, batch_size=[4]), stochastic_output=False)
        out = reloaded(jit_in)
    diff = (ref - out).abs().max().item()
    assert out.shape == (4, num_actions), f"출력 shape 이상: {tuple(out.shape)}"
    assert diff < 1e-5, f"jit 불일치 diff={diff}"

    print(f"[export] OK — {out_path}")
    print(f"[export] 입력: obs({jit_in.shape[-1]}) → action({num_actions}) / history 사용={uses_history}")
    print(f"[export] jit vs deterministic forward diff={diff:.1e} (< 1e-5)")


if __name__ == "__main__":
    main()

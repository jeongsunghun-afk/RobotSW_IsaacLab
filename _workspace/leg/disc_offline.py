# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""AMP 판별기를 Isaac(AppLauncher) 없이 체크포인트에서 직접 세우고 채점한다.

``_workspace/leg/amp_obs_clip_probe.py`` 의 ``_build_disc`` / ``_score`` 를 그대로 옮긴 것이다.
``rsl_rl`` 은 순수 torch 라 시뮬레이터 없이 import 된다. 차원(``input_dim``·``cond_dim``)은 env 가
런타임에 계산하는 값이라 agent.yaml 에 없다 — 체크포인트의 normalizer 버퍼와 첫 층 weight 에서
**역산**해 하드코딩을 피한다.

주의(측정 계약):
  * 두 판별기는 중립 심판이 아니다. 같은 disc 안에서 expert vs policy 만 비교하라.
  * DRAIL ``get_logits`` 는 매 호출 t·eps 를 새로 뽑는다 → ``reps`` 평균.
  * ``norm_clip=10.0`` 이 걸려 있어 클립된 열은 gradient 가 **정확히 0** 이다.
"""

from __future__ import annotations

import json
import numpy as np
import os
import torch
import yaml

from rsl_rl.modules import AMPDiscriminator
from rsl_rl.modules.amp_diffusion_discriminator import AMPDiffusionDiscriminator

FIELDS = [
    ("dof_pos", 17),
    ("dof_vel", 17),
    ("root_h", 1),
    ("lin_vel", 3),
    ("ang_vel", 3),
    ("foot_pos", 12),
    ("rot_tan_norm", 6),
]
STEP_DIM = 59


def load_yaml(path: str):
    with open(path) as f:
        return yaml.unsafe_load(f)


def field_slices(step_dim: int = STEP_DIM, n_hist: int = 10) -> dict[str, np.ndarray]:
    """필드 이름 → 590 차원 안의 열 인덱스(모든 히스토리 슬롯). 레이아웃은 [H, 59].view(-1), index0=newest."""
    out = {}
    off = 0
    for name, w in FIELDS:
        cols = np.concatenate([np.arange(h * step_dim + off, h * step_dim + off + w) for h in range(n_hist)])
        out[name] = cols
        off += w
    assert off == step_dim, f"필드 합 {off} != step_dim {step_dim}"
    return out


def hist_slices(step_dim: int = STEP_DIM, n_hist: int = 10) -> list[np.ndarray]:
    return [np.arange(h * step_dim, (h + 1) * step_dim) for h in range(n_hist)]


def build_disc(agent_yaml_path: str, ckpt_path: str, device: str = "cuda:0"):
    """(arch, disc) 를 돌려준다. ``input_dim``/``cond_dim`` 은 체크포인트에서 역산."""
    amp = load_yaml(agent_yaml_path)["amp"]
    arch = amp.get("disc_arch", "mlp")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["discriminator_state_dict"]

    kin_dim = int(sd["amp_obs_normalizer._mean"].shape[-1])
    if arch == "mlp":
        w0 = [v for k, v in sd.items() if k.startswith("trunk") and v.ndim == 2][0]
        input_dim = int(w0.shape[1])
    elif arch == "drail":
        # _CondDiffusionMLP 첫 층: in = kin_dim + (label_dim + cond_dim)
        w0 = sd["model._layers.0.weight"]
        label_dim = int(amp.get("drail_label_dim", 10))
        input_dim = kin_dim + (int(w0.shape[1]) - kin_dim - label_dim)
    else:
        raise ValueError(f"모르는 disc_arch: {arch}")
    cond_dim = input_dim - kin_dim

    if arch == "mlp":
        d = AMPDiscriminator(
            input_dim=input_dim,
            hidden_dims=amp.get("discriminator_hidden_dims", [1024, 512]),
            device=device,
            disc_reward_type=amp.get("disc_reward_type", "ls_gan"),
            norm_clip=amp.get("disc_norm_clip"),
            cond_dim=cond_dim,
        )
    else:
        d = AMPDiffusionDiscriminator(
            input_dim=input_dim,
            hidden_dims=amp.get("drail_hidden_dims", [256, 256, 256, 256]),
            activation=amp.get("drail_activation", "elu"),
            device=device,
            disc_reward_type=amp.get("disc_reward_type", "bce"),
            norm_clip=amp.get("disc_norm_clip"),
            cond_dim=cond_dim,
            label_dim=amp.get("drail_label_dim", 10),
            diffusion_steps=amp.get("drail_diffusion_steps", 1000),
            sample_strategy=amp.get("drail_sample_strategy", "antithetic"),
            sample_strategy_value=amp.get("drail_sample_strategy_value", 0),
            paired_noise=amp.get("drail_paired_noise", True),
        )
    d.load_state_dict(sd)  # strict
    d.to(device).eval()
    d._kin_dim = kin_dim
    d._input_dim = input_dim
    d._arch = arch
    return arch, d


def norm_sigma(disc) -> torch.Tensor:
    """정규화 분모 (std + eps). kinematics 열만, shape [kin_dim]."""
    n = disc.amp_obs_normalizer
    return (n._std.squeeze(0) + n.eps).detach()


@torch.no_grad()
def score(disc, x: torch.Tensor, reps: int = 32, chunk: int = 4096) -> np.ndarray:
    """D = sigmoid(logit) 평균 확률. DRAIL 만 reps 반복."""
    r = reps if isinstance(disc, AMPDiffusionDiscriminator) else 1
    out = torch.zeros(x.shape[0], device=disc.device)
    for _ in range(r):
        probs = []
        for i in range(0, x.shape[0], chunk):
            probs.append(torch.sigmoid(disc.get_logits(x[i : i + chunk].to(disc.device))).view(-1))
        out += torch.cat(probs)
    return (out / r).cpu().numpy().astype(np.float32)


@torch.no_grad()
def logits(disc, x: torch.Tensor, reps: int = 32, chunk: int = 4096) -> np.ndarray:
    r = reps if isinstance(disc, AMPDiffusionDiscriminator) else 1
    out = torch.zeros(x.shape[0], device=disc.device)
    for _ in range(r):
        ls = []
        for i in range(0, x.shape[0], chunk):
            ls.append(disc.get_logits(x[i : i + chunk].to(disc.device)).view(-1))
        out += torch.cat(ls)
    return (out / r).cpu().numpy().astype(np.float32)


def grad_logit(disc, x: torch.Tensor, reps: int = 32, chunk: int = 512) -> torch.Tensor:
    """|∂ mean_reps logit / ∂ x_raw| — 원 입력(정규화 이전) 기준. shape = x.shape."""
    r = reps if isinstance(disc, AMPDiffusionDiscriminator) else 1
    out = torch.zeros_like(x, device=disc.device)
    for i in range(0, x.shape[0], chunk):
        xb = x[i : i + chunk].to(disc.device).clone().requires_grad_(True)
        acc = torch.zeros_like(xb)
        for _ in range(r):
            lg = disc.get_logits(xb).sum()
            (g,) = torch.autograd.grad(lg, xb, retain_graph=False)
            acc += g
        out[i : i + chunk] = acc / r
    return out.detach()


def clip_hit_frac(disc, x: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """norm_clip 에 걸린 열 비율 [kin_dim]. norm_clip 이 None 이면 0."""
    kin = disc._kin_dim
    if disc.norm_clip is None:
        return torch.zeros(kin)
    n = disc.amp_obs_normalizer
    hits = torch.zeros(kin, device=disc.device)
    tot = 0
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk):
            xb = x[i : i + chunk, :kin].to(disc.device)
            z = (xb - n._mean) / (n._std + n.eps)
            hits += (z.abs() >= disc.norm_clip).float().sum(0)
            tot += xb.shape[0]
    return (hits / max(tot, 1)).cpu()


def load_npz(path: str):
    z = np.load(path, allow_pickle=True)
    lay = json.loads(str(z["layout"])) if "layout" in z else None
    return z, lay


RUNS = {
    "condmlp50k": (
        "logs/rsl_rl/leg_imitation_tracking_rma/"
        "2026-09-02_17-56-53_condmatch_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32",
        "model_49999.pt",
    ),
    "conddrail45k": (
        "logs/rsl_rl/leg_imitation_tracking_rma/"
        "2026-09-03_12-51-22_condmatch_drail_resume16k7_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32",
        "model_45000.pt",
    ),
}
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE_ROOT = os.path.join(
    REPO, "reports/leg_imitation/_comparisons/conditional_discriminator/metrics/clip_probe"
)


def build_both(device: str = "cuda:0"):
    """{'mlp': disc, 'drail': disc} — A' 의 mlp 와 B' 의 drail."""
    out = {}
    for tag, (rd, ck) in RUNS.items():
        rd = os.path.join(REPO, rd)
        arch, d = build_disc(os.path.join(rd, "params", "agent.yaml"), os.path.join(rd, ck), device)
        out[arch] = d
    return out


if __name__ == "__main__":
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    discs = build_both(dev)
    for a, d in discs.items():
        print(f"{a}: input_dim={d._input_dim} kin_dim={d._kin_dim} cond_dim={d.cond_dim} norm_clip={d.norm_clip}")

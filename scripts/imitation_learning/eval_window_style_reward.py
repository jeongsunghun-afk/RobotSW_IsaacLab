# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""창 인코더(`--window_w`) 체크포인트용 스타일 판별력 검증 하네스.

``validate_latent_style_reward.py`` 는 수정 금지 파일이고, 그 안에서 쓰는
``rsl_rl.modules.motion_encoder.MotionEncoder`` 는 인코더 입력이 **2 프레임**으로
하드코딩되어 있어 ``w > 2`` 체크포인트를 읽지 못한다 (``x_dim = in_features // 2``).

그래서 검증 스크립트를 **다시 쓰지 않고** 모듈로 로드한 뒤 두 심볼만 갈아끼우고
그 ``main()`` 을 그대로 호출한다::

    MotionEncoder  ->  WindowMotionEncoder   (인코더 입력 w 프레임)
    pair_stack     ->  window_pair_stack     (문맥 (w-1) 프레임 + 표적 1 프레임)

이렇게 하면 negative 생성·rng 소비 순서·통제군(expert_train)·raw baseline·클립 단위
marginal KL 표본추출이 **원본과 완전히 동일**하다. raw baseline 도 자동으로 같은 w 프레임
창을 보게 되어 "latent 가 raw 를 이겨야 한다" 는 판정 논리가 유지된다.

w=2 에서는 원본 경로와 수치가 일치해야 한다 (하네스 검증). 실제로 published 표와
소수 셋째 자리까지 일치함을 확인했다.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))

#: 문맥 프레임 수 (w-1). ``main`` 에서 체크포인트를 보고 채운다.
_CTX_FRAMES = 1
#: 통제용 최소 클립 길이 [프레임]. 창이 넓어지면 짧은 클립이 표본에서 빠지므로,
#: "창이 도움을 준다" 와 "긴 클립이 쉽다" 를 가르려면 좁은 창을 **같은 클립 집단**에서 재야 한다.
_MIN_CLIP_LEN = 0
X_DIM = 49


def load_validator(path: str):
    spec = importlib.util.spec_from_file_location("vsr", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class WindowMotionEncoder(nn.Module):
    """Motion VAE 의 창 인코더만 떼어낸 채점용 모듈.

    ``MotionEncoder`` 와 같은 표면(``x_dim``/``latent_dim``/``inference``/``from_checkpoint``)을
    갖되 입력이 ``[문맥 (w-1)*x_dim, 표적 x_dim]`` 이다.
    """

    def __init__(self, x_dim: int, latent_dim: int, hidden: int, window_w: int) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.window_w = window_w
        layers: list[nn.Module] = []
        prev = window_w * x_dim
        for h in (hidden, hidden):
            layers += [nn.Linear(prev, h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 2 * latent_dim))
        self.encoder = nn.Sequential(*layers)
        self.register_buffer("obs_mean", torch.zeros(x_dim))
        self.register_buffer("obs_std", torch.ones(x_dim))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """프레임 단위 정규화. 입력 폭이 ``x_dim`` 의 배수면 프레임마다 같은 통계를 쓴다."""
        n = x.shape[-1] // self.x_dim
        if n * self.x_dim != x.shape[-1]:
            raise ValueError(f"입력 폭 {x.shape[-1]} 이 x_dim {self.x_dim} 의 배수가 아니다")
        return (x - self.obs_mean.repeat(n)) / self.obs_std.repeat(n)

    def encode(self, x_ctx: torch.Tensor, x_curr: torch.Tensor, normalized: bool = False):
        if not normalized:
            x_ctx = self.normalize(x_ctx)
            x_curr = self.normalize(x_curr)
        mu, logvar = self.encoder(torch.cat([x_ctx, x_curr], dim=-1)).chunk(2, dim=-1)
        return mu, logvar.clamp(-8.0, 8.0)

    def forward(self, x_ctx: torch.Tensor, x_curr: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        return self.encode(x_ctx, x_curr, normalized)[0]

    def inference(self, x_ctx: torch.Tensor, x_curr: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        with torch.no_grad():
            return self.forward(x_ctx, x_curr, normalized)

    def freeze(self) -> None:
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path: str, device="cpu", freeze: bool = True) -> WindowMotionEncoder:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        a = ckpt.get("args", {})
        w = int(a.get("window_w", 2))
        enc = {k[len("encoder.") :]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
        if not enc:
            raise KeyError(f"체크포인트에 encoder 가중치가 없다: {path}")
        in_dim = enc["0.weight"].shape[1]
        if in_dim % w:
            raise ValueError(f"인코더 입력 {in_dim} 이 window_w {w} 로 나뉘지 않는다")
        m = cls(in_dim // w, int(a.get("latent_dim", 18)), int(a.get("hidden", 256)), w)
        m.encoder.load_state_dict(enc)
        mean = torch.as_tensor(ckpt["mean"], dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(ckpt["std"], dtype=torch.float32).reshape(-1).clone()
        std[std < 1e-4] = 1.0
        m.obs_mean.copy_(mean)
        m.obs_std.copy_(std)
        m.to(device)
        if freeze:
            m.freeze()
        return m


def window_pair_stack(clips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """클립 목록 → (문맥 [N, (w-1)*X], 표적 [N, X], clip_id).

    ``_CTX_FRAMES = 1`` (w=2) 이면 원본 ``pair_stack`` 과 **완전히 같다**.
    """
    c = _CTX_FRAMES
    prev, curr, cid = [], [], []
    for i, x in enumerate(clips):
        n = len(x) - c
        if n <= 0 or len(x) < _MIN_CLIP_LEN:
            continue
        prev.append(np.concatenate([x[j : j + n] for j in range(c)], axis=-1))
        curr.append(x[c:])
        cid.append(np.full(n, i, dtype=np.int64))
    if not prev:
        raise SystemExit("창 길이가 모든 클립보다 길다 — 표본이 하나도 없다")
    return np.concatenate(prev), np.concatenate(curr), np.concatenate(cid)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--validator", default=os.path.join(HERE, "validate_latent_style_reward.py"))
    p.add_argument(
        "--min_clip_len",
        type=int,
        default=0,
        help="이 프레임 수 미만의 (negative 생성 후) 클립을 표본에서 제외한다. 넓은 창의 "
        "클립 집단을 좁은 창에서 재현하는 통제용. w 의 margKL 표본 조건은 len >= 63 + w 다.",
    )
    known, rest = p.parse_known_args()

    w = int(torch.load(known.ckpt, map_location="cpu", weights_only=False).get("args", {}).get("window_w", 2))
    global _CTX_FRAMES, _MIN_CLIP_LEN
    _CTX_FRAMES = w - 1
    _MIN_CLIP_LEN = known.min_clip_len

    v = load_validator(known.validator)
    v.MotionEncoder = WindowMotionEncoder
    v.pair_stack = window_pair_stack
    print(
        f"[창 하네스] window_w = {w}  (문맥 {w - 1} 프레임 + 표적 1)"
        + (f"  · 최소 클립 길이 {_MIN_CLIP_LEN} 프레임 통제" if _MIN_CLIP_LEN else "")
    )
    sys.argv = [known.validator, "--ckpt", known.ckpt] + rest
    v.main()


if __name__ == "__main__":
    main()

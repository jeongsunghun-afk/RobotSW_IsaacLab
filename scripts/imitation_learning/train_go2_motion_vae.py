# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 Motion VAE 오프라인 학습 (Phase 1).

"Walk Like Dogs" (arXiv 2507.00677) §V 의 상태전이 VAE 를 재현한다.

    encoder : (x[t-1], x[t])        -> q(z | ·)          latent 18-D
    decoder : (x[t-1], z)           -> x_hat[t]          MoE, expert 6개 + gating
    loss    : ||x[t] - x_hat[t]||^2 + beta * KL(q || p),  beta = 0.05

``--window_w`` 로 **인코더 입력만** 창으로 넓힐 수 있다 (디코더는 한 스텝 예측 유지)::

    encoder : (x[t-w+2], ..., x[t-1], x[t])  -> q(z | ·)     입력 49*w
    decoder : (x[t], z)                      -> x_hat[t+k]   그대로

한 스텝 전이 쌍은 보행 **위상**을 볼 수 없어 좌우 다리 교환·위상 분리 같은 미세 gait 교란을
구분하지 못한다 (Phase 2 검증에서 AUROC 0.50~0.53). 창을 넓히면 z 가 위상을 담을 수 있다.
디코더까지 창을 주면 다른 모델이 되어 Phase 1 결과와 비교가 끊기므로 건드리지 않는다.
``--window_w 2`` 는 기존 경로와 **완전히 동일**하다.

학습 일정도 논문을 따른다 — 상태전이로 warm-up 후 autoregressive 로 전환한다
(decoder 예측을 다음 스텝의 조건으로 되먹여 시퀀스 예측 안정성을 높인다).

``x_vae`` 49-D 레이아웃::

    [0]      base height                       1
    [1:7]    heading-relative 6D 회전          6    (yaw 제거 → roll/pitch 성분만)
    [7:10]   base 선속도 (body frame)          3
    [10:13]  base 각속도 (body frame)          3
    [13:25]  발 위치 (base-local, 4x3)        12
    [25:37]  관절각                           12
    [37:49]  관절속도                         12

데이터는 Phase 0 산출물을 쓴다. train/val 은 **세션 단위**로 나눈다 — 클립 단위로 나누면
같은 세션의 슬라이딩 윈도우가 양쪽에 걸려 누수가 생기고, G1 게이트가 무의미해진다.

게이트:
  G1  held-out **세션** 재구성 오차 <= in-sample 오차 x 2
  G2  같은 mode, 다른 속도 두 클립의 z 를 보간하면 디코드 결과의 vx 가 단조 변화
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time

import numpy as np
import scipy.special
import torch
import torch.nn as nn
import torch.nn.functional as F

X_DIM = 49
FEATURE_SLICES = {
    "base_height": slice(0, 1),
    "rot6d": slice(1, 7),
    "lin_vel": slice(7, 10),
    "ang_vel": slice(10, 13),
    "foot_pos": slice(13, 25),
    "joint_pos": slice(25, 37),
    "joint_vel": slice(37, 49),
}
CLIP_NAME_RE = re.compile(r"^[a-z_]+__(M_)?(D\d+_[A-Za-z0-9]+_[A-Za-z0-9]+_\d+)_F\d+_F\d+_\d+$")


# ──────────────────────────────────────────────────────────────
# x_vae 구성
# ──────────────────────────────────────────────────────────────
def heading_relative_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """wxyz 쿼터니언 → heading(yaw) 을 제거한 6D 회전 표현 (N, 6).

    6D 표현은 회전행렬의 앞 두 열이다 (Zhou et al. 2019). yaw 를 빼면
    latent 가 진행 방향에 불변해진다.
    """
    w, x, y, z = quat_wxyz[:, 0], quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3]
    # yaw-only 쿼터니언의 켤레를 왼쪽에 곱해 heading 제거
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    hw, hz = np.cos(-0.5 * yaw), np.sin(-0.5 * yaw)
    rw = hw * w - hz * z
    rx = hw * x - hz * y
    ry = hw * y + hz * x
    rz = hw * z + hz * w
    rot = np.stack(
        [
            np.stack([1 - 2 * (ry * ry + rz * rz), 2 * (rx * ry - rw * rz), 2 * (rx * rz + rw * ry)], -1),
            np.stack([2 * (rx * ry + rw * rz), 1 - 2 * (rx * rx + rz * rz), 2 * (ry * rz - rw * rx)], -1),
            np.stack([2 * (rx * rz - rw * ry), 2 * (ry * rz + rw * rx), 1 - 2 * (rx * rx + ry * ry)], -1),
        ],
        1,
    )
    return np.concatenate([rot[:, :, 0], rot[:, :, 1]], axis=-1)


def build_x_vae(motion_lib_module, pkl_path: str) -> np.ndarray:
    """PKL 한 개 → x_vae (N, 49)."""
    root_pos, root_quat, lin_vel, ang_vel, dof_pos, dof_vel, foot_pos, _ = motion_lib_module.Go2MotionLib._load_pkl(
        pkl_path
    )
    return np.concatenate(
        [
            root_pos[:, 2:3],
            heading_relative_rot6d(root_quat),
            lin_vel,
            ang_vel,
            foot_pos.reshape(len(foot_pos), -1),
            dof_pos,
            dof_vel,
        ],
        axis=-1,
    ).astype(np.float32)


# ──────────────────────────────────────────────────────────────
# von Mises-Fisher 분포 (hypersphere latent)
# ──────────────────────────────────────────────────────────────
#
# 기호 (Davidson et al. 2018, arXiv 1804.00891 / Ulrich 1984):
#
#   d       latent 차원 (ambient). z 는 S^{d-1} 위의 단위벡터.
#   mu      평균 방향, ||mu|| = 1
#   kappa   집중도 (concentration). 클수록 mu 주변에 몰린다. 각도 퍼짐 ~ 1/sqrt(kappa)
#   m       = d - 1  (rejection sampler 의 Beta/로그항 차수. d 가 아니다)
#   A_d(k)  = I_{d/2}(k) / I_{d/2-1}(k)   평균 resultant length = E[z . mu]
#   I_v     제1종 변형 베셀 함수. 직접 쓰면 overflow 하므로 log 스케일로만 다룬다.
#
# 균등 prior 기준 KL 닫힌형:
#
#   KL( vMF(mu,k) || U(S^{d-1}) ) = k * A_d(k) + log C_d(k) + log S_{d-1}
#   log C_d(k) = (d/2 - 1) log k - (d/2) log(2 pi) - log I_{d/2-1}(k)
#   log S_{d-1} = log 2 + (d/2) log pi - lgamma(d/2)
#
# ★ kappa 를 상수로 고정하면 이 값 전체가 방향 mu 에 **무관한 상수**가 되어
#   gradient 가 0 이다. 그래서 --kappa_fixed 경로에서는 KL 항을 아예 계산하지 않는다.
#   이것이 "붕괴 압력 0" 가설을 나르는 변형이다.


class _LogIv(torch.autograd.Function):
    """log I_v(kappa). scipy 의 지수 스케일 ive 로 계산해 overflow 를 피한다.

    d/dk log I_v(k) = I_{v+1}(k)/I_v(k) + v/k
    """

    @staticmethod
    def forward(ctx, kappa: torch.Tensor, order: float) -> torch.Tensor:
        k = kappa.detach().double().cpu().numpy()
        ive_v = scipy.special.ive(order, k)
        ive_v1 = scipy.special.ive(order + 1.0, k)
        out = np.log(np.maximum(ive_v, 1e-300)) + k
        ratio = torch.from_numpy(ive_v1 / np.maximum(ive_v, 1e-300)).to(kappa.device, kappa.dtype)
        ctx.save_for_backward(kappa, ratio)
        ctx.order = order
        return torch.from_numpy(out).to(kappa.device, kappa.dtype)

    @staticmethod
    def backward(ctx, grad):
        kappa, ratio = ctx.saved_tensors
        return grad * (ratio + ctx.order / kappa), None


def log_iv(kappa: torch.Tensor, order: float) -> torch.Tensor:
    return _LogIv.apply(kappa, order)


def vmf_mean_resultant(kappa: torch.Tensor, d: int) -> torch.Tensor:
    """A_d(kappa) = I_{d/2}(k)/I_{d/2-1}(k) = E[z . mu]."""
    k = kappa.detach().double().cpu().numpy()
    r = scipy.special.ive(d / 2.0, k) / np.maximum(scipy.special.ive(d / 2.0 - 1.0, k), 1e-300)
    return torch.from_numpy(r).to(kappa.device, kappa.dtype)


def vmf_kl_uniform(kappa: torch.Tensor, d: int) -> torch.Tensor:
    """KL( vMF(mu,kappa) || U(S^{d-1}) ). mu 에 무관하다."""
    a = vmf_mean_resultant(kappa, d)
    log_c = (d / 2.0 - 1.0) * torch.log(kappa) - (d / 2.0) * np.log(2.0 * np.pi) - log_iv(kappa, d / 2.0 - 1.0)
    log_s = np.log(2.0) + (d / 2.0) * np.log(np.pi) - float(scipy.special.gammaln(d / 2.0))
    return kappa * a + log_c + log_s


def _sample_w(kappa: torch.Tensor, d: int, max_iter: int = 100) -> torch.Tensor:
    """Ulrich(1984) rejection sampling 으로 w = z . mu 를 뽑는다. kappa: [B]."""
    m = d - 1
    b = (-2.0 * kappa + torch.sqrt(4.0 * kappa**2 + m**2)) / m
    x0 = (1.0 - b) / (1.0 + b)
    c = kappa * x0 + m * torch.log(1.0 - x0**2)

    bd = b.detach()
    x0d, cd, kd = x0.detach(), c.detach(), kappa.detach()
    shape = kappa.shape
    eps_keep = torch.zeros(shape, device=kappa.device, dtype=kappa.dtype)
    done = torch.zeros(shape, device=kappa.device, dtype=torch.bool)
    beta = torch.distributions.Beta(
        torch.tensor(m / 2.0, device=kappa.device, dtype=kappa.dtype),
        torch.tensor(m / 2.0, device=kappa.device, dtype=kappa.dtype),
    )
    for _ in range(max_iter):
        e = beta.sample(shape)
        w = (1.0 - (1.0 + bd) * e) / (1.0 - (1.0 - bd) * e)
        u = torch.rand(shape, device=kappa.device, dtype=kappa.dtype)
        acc = (kd * w + m * torch.log(1.0 - x0d * w) - cd) >= torch.log(u + 1e-20)
        take = acc & (~done)
        eps_keep = torch.where(take, e, eps_keep)
        done = done | acc
        if bool(done.all()):
            break
    # 미채택분(수치적 극단)은 마지막 표본을 그대로 쓴다 — 실측 reject rate 는 18-D 에서 무시할 수준
    eps_keep = torch.where(done, eps_keep, torch.full_like(eps_keep, 0.5))
    e = eps_keep.detach()
    # kappa 로의 gradient 는 b(kappa) 를 통해 흐른다 (eps 는 detach)
    return (1.0 - (1.0 + b) * e) / (1.0 - (1.0 - b) * e)


def _householder(x: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """e1 을 mu 로 보내는 Householder 반사를 x 에 적용한다. 이 경로로 mu 의 gradient 가 흐른다."""
    e1 = torch.zeros_like(mu)
    e1[..., 0] = 1.0
    u = e1 - mu
    u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
    return x - 2.0 * (x * u).sum(-1, keepdim=True) * u


def vmf_rsample(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """vMF 재매개화 샘플. mu: [B, d] (단위벡터), kappa: [B] -> [B, d]."""
    d = mu.shape[-1]
    w = _sample_w(kappa, d).unsqueeze(-1)  # [B, 1]
    v = torch.randn(mu.shape[0], d - 1, device=mu.device, dtype=mu.dtype)
    v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
    z = torch.cat([w, torch.sqrt(torch.clamp(1.0 - w**2, min=1e-10)) * v], -1)
    return _householder(z, mu)


# ──────────────────────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────────────────────
def mlp(dims: list[int], out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = dims[0]
    for h in dims[1:]:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


class MotionVAE(nn.Module):
    """상태전이 VAE. decoder 는 Mixture of Experts."""

    def __init__(
        self,
        latent_dim: int = 18,
        hidden: int = 256,
        num_experts: int = 6,
        gate_hidden: int = 64,
        latent_dist: str = "gauss",
        kappa_fixed: float = 0.0,
        gauss_fixed_logvar: float | None = None,
        window_w: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        # 인코더 입력 프레임 수. 2 = 상태전이 쌍(기존). w 프레임 중 앞 w-1 개가 문맥,
        # 마지막 1 개가 표적 프레임 x[t+k] 다.
        self.window_w = max(2, int(window_w))
        self.num_experts = num_experts
        self.latent_dist = latent_dist
        self.kappa_fixed = kappa_fixed
        self.gauss_fixed_logvar = gauss_fixed_logvar
        # vmf 는 (방향 d, 집중도 1) 를 낸다. kappa 고정이면 집중도 head 는 쓰이지 않는다.
        enc_out = 2 * latent_dim if latent_dist == "gauss" else latent_dim + 1
        # ★ w=2 에서 입력 폭은 2*X_DIM 이라 기존과 파라미터 shape·초기화 RNG 소비가 동일하다.
        self.encoder = mlp([self.window_w * X_DIM, hidden, hidden], enc_out)
        dec_in = X_DIM + latent_dim
        # expert 를 하나의 배치 텐서로 묶어 한 번에 계산한다.
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, dec_in, hidden))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, hidden))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden, hidden))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, hidden))
        self.expert_w3 = nn.Parameter(torch.empty(num_experts, hidden, X_DIM))
        self.expert_b3 = nn.Parameter(torch.zeros(num_experts, X_DIM))
        for p in (self.expert_w1, self.expert_w2, self.expert_w3):
            for e in range(num_experts):
                nn.init.kaiming_uniform_(p.data[e], a=5**0.5)
        self.gate = mlp([dec_in, gate_hidden, gate_hidden], num_experts)
        # ★ 축 정렬용 보조 헤드. **전 카테고리가 하나의 (w, b) 를 공유**한다 —
        #   카테고리마다 속도축이 다르면 이 손실을 줄일 수 없다 (§6-3 의 전이 상관이 그대로 목적함수).
        #   반드시 __init__ 의 **맨 마지막**에 만든다. 앞에 두면 나머지 파라미터의 초기화 RNG 가
        #   밀려 --lambda_speed 0 이 기존 결과와 달라진다.
        self.speed_head = nn.Linear(latent_dim, 1)

    def encode(self, x_ctx: torch.Tensor, x_curr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(mu, aux) 를 낸다.

        Args:
            x_ctx: 문맥 프레임 묶음, shape [B, (w-1)*X_DIM]. w=2 이면 곧 ``x[t]`` 한 프레임이다.
            x_curr: 표적 프레임 ``x[t+k]``, shape [B, X_DIM].

        gauss : mu = 평균,          aux = logvar        [B, d]
        vmf   : mu = 단위 평균방향, aux = kappa         [B]
        """
        h = self.encoder(torch.cat([x_ctx, x_curr], -1))
        if self.latent_dist == "gauss":
            mu, logvar = h.chunk(2, -1)
            if self.gauss_fixed_logvar is not None:
                logvar = torch.full_like(logvar, self.gauss_fixed_logvar)
            return mu, logvar.clamp(-8.0, 8.0)
        mu = F.normalize(h[..., : self.latent_dim], dim=-1)
        if self.kappa_fixed > 0.0:
            kappa = torch.full(mu.shape[:-1], self.kappa_fixed, device=mu.device, dtype=mu.dtype)
        else:
            # softplus + 1 로 하한을 두어 rejection sampler 의 수치 극단을 피한다
            kappa = F.softplus(h[..., self.latent_dim]) + 1.0
            kappa = kappa.clamp(max=1000.0)
        return mu, kappa

    #: x_prev 중 속도 성분 (lin_vel, ang_vel, joint_vel). 이 블록이 있으면 decoder 가
    #: x[t] ~= x[t-1] + v*dt 로 외삽할 수 있어 z 를 읽을 이유가 없어진다.
    VEL_IDX = list(range(7, 13)) + list(range(37, 49))
    #: VEL_IDX 의 여집합 — base_height(0:1) + rot6d(1:7) + foot_pos(13:25) + joint_pos(25:37).
    POSE_IDX = list(range(0, 7)) + list(range(13, 37))

    def decode(
        self,
        x_prev: torch.Tensor,
        z: torch.Tensor,
        cond_dropout: float = 0.0,
        vel_dropout: float = 0.0,
        pose_dropout: float = 0.0,
    ) -> torch.Tensor:
        if cond_dropout > 0.0 and self.training:
            keep = (torch.rand(len(x_prev), 1, device=x_prev.device) >= cond_dropout).float()
            x_prev = x_prev * keep
        if vel_dropout > 0.0 and self.training:
            keep = (torch.rand(len(x_prev), 1, device=x_prev.device) >= vel_dropout).float()
            x_prev = x_prev.clone()
            x_prev[:, self.VEL_IDX] = x_prev[:, self.VEL_IDX] * keep
        if pose_dropout > 0.0 and self.training:
            keep = (torch.rand(len(x_prev), 1, device=x_prev.device) >= pose_dropout).float()
            x_prev = x_prev.clone()
            x_prev[:, self.POSE_IDX] = x_prev[:, self.POSE_IDX] * keep
        h0 = torch.cat([x_prev, z], -1)  # [B, dec_in]
        gate = F.softmax(self.gate(h0), dim=-1)  # [B, E]
        h = F.elu(torch.einsum("bi,eih->beh", h0, self.expert_w1) + self.expert_b1)
        h = F.elu(torch.einsum("beh,ehg->beg", h, self.expert_w2) + self.expert_b2)
        out = torch.einsum("beh,ehx->bex", h, self.expert_w3) + self.expert_b3  # [B, E, X]
        return (gate.unsqueeze(-1) * out).sum(1)

    def forward(
        self,
        x_ctx: torch.Tensor,
        x_curr: torch.Tensor,
        cond_dropout: float = 0.0,
        vel_dropout: float = 0.0,
        pose_dropout: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``x_ctx`` 는 [B, (w-1)*X_DIM]. 디코더는 그중 **마지막 프레임만** 조건으로 받는다."""
        mu, aux = self.encode(x_ctx, x_curr)
        x_prev = x_ctx[:, -X_DIM:]
        if self.latent_dist == "gauss":
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * aux) if self.training else mu
        else:
            # eval 에서는 샘플링하지 않고 평균방향 mu 를 그대로 쓴다 — gauss 경로가
            # eval 에서 mu 를 쓰는 것과 맞춰야 G0/G1 을 두 분포 사이에서 비교할 수 있다.
            z = vmf_rsample(mu, aux) if self.training else mu
        return self.decode(x_prev, z, cond_dropout, vel_dropout, pose_dropout), mu, aux


# ──────────────────────────────────────────────────────────────
def load_dataset(args) -> tuple[dict, dict]:
    import importlib.util

    spec = importlib.util.spec_from_file_location("ml", args.motion_lib)
    assert spec is not None and spec.loader is not None
    ml = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ml)

    split = json.load(open(args.split_json))
    val_sessions = set(split["val_sessions"])

    train_clips, val_clips = [], []
    unparsed = []
    for p in sorted(glob.glob(os.path.join(args.pkl_dir, "*.pkl"))):
        m = CLIP_NAME_RE.match(os.path.splitext(os.path.basename(p))[0])
        if m is None:
            unparsed.append(p)
            continue
        x = build_x_vae(ml, p)[:: args.stride]
        if len(x) < 2 or not np.isfinite(x).all():
            unparsed.append(p)
            continue
        # 카테고리는 파일명 접두사(`walk__`, `run__`, `sit_walk__`, ...). 미러 클립(`__M_`)은
        # 같은 카테고리로 센다 — 좌우 복제라 다양성 기여는 0 이지만 실제 학습 표본이기 때문이다.
        prefix = os.path.basename(p).split("__")[0]
        (val_clips if m.group(2) in val_sessions else train_clips).append(
            {"path": p, "session": m.group(2), "x": x, "prefix": prefix}
        )
    if unparsed:
        raise SystemExit(f"클립 {len(unparsed)}개를 처리하지 못했다 (조용히 넘기지 않는다): {unparsed[:3]}")
    return {"clips": train_clips}, {"clips": val_clips}


def clip_ctx_pairs(x: np.ndarray, k: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    """클립 하나 → (문맥 [n, (w-1)*X], 표적 [n, X]).

        문맥  = x[t-(w-2)k], ..., x[t-k], x[t]      (w-1 프레임, 간격 k)
        표적  = x[t+k]

    w=2 면 문맥이 x[t] 한 프레임이라 기존 ``(x[t], x[t+k])`` 쌍과 **완전히 같다**.
    유효 t 는 ``(w-2)k <= t <= len-1-k`` 이므로 클립당 쌍 수는 ``len - (w-1)k`` 다.
    """
    n = len(x) - (w - 1) * k
    if n <= 0:
        return np.empty((0, (w - 1) * x.shape[1]), x.dtype), np.empty((0, x.shape[1]), x.dtype)
    base = (w - 2) * k  # 첫 t
    ctx = np.concatenate([x[base - j * k : base - j * k + n] for j in range(w - 2, -1, -1)], axis=-1)
    return ctx, x[base + k : base + k + n]


def make_pairs(clips: list[dict], k: int = 1, key: str | None = None, w: int = 2) -> tuple[np.ndarray, ...]:
    """(문맥, x[t+k]) 쌍. k>1 이면 예측 지평이 넓어져 x[t] 만으로는 결정이 안 된다.

    ``w`` 는 인코더 창 길이(프레임 수). 기본 2 = 상태전이 쌍으로 기존 동작과 동일하다.
    ``key`` 를 주면 쌍마다 그 클립의 ``key`` 값(카테고리/세션)을 담은 배열을 함께 돌려준다.
    """
    use = [c for c in clips if len(c["x"]) > (w - 1) * k]
    made = [clip_ctx_pairs(c["x"], k, w) for c in use]
    prev = np.concatenate([m[0] for m in made])
    curr = np.concatenate([m[1] for m in made])
    if key is None:
        return prev, curr
    lab = np.concatenate([np.array([c[key]] * len(m[1])) for c, m in zip(use, made)])
    return prev, curr, lab


#: `--balance_groups main` 에서 쓰는 묶음. 나머지 접두사(전이 구간)는 "other" 로 합친다.
#: 12개 접두사를 그대로 균등화하면 `lie_trot`(118 프레임)이 64.7배로 뻥튀기되어
#: 소수 클립 과적합이 가설 검증을 오염시킨다.
MAIN_GROUPS = ("walk", "run", "trot")


def group_of(prefix: str, mode: str) -> str:
    if mode == "prefix":
        return prefix
    return prefix if prefix in MAIN_GROUPS else "other"


def balance_weights(labels: np.ndarray, scheme: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """그룹별 표본 수의 역수(또는 그 제곱근)로 표본 가중치를 만든다.

    가중치 합은 표본 수로 정규화해 epoch 당 기대 표본 수가 `none` 과 같게 유지한다::

        n_g      = 그룹 g 의 표본 수
        w_sample = (1 / n_g)          scheme = inverse   -> 그룹 균등
                 = (1 / n_g) ** 0.5   scheme = sqrt      -> 완화된 균형
    """
    uniq, inv, cnt = np.unique(labels, return_inverse=True, return_counts=True)
    per_group = 1.0 / cnt if scheme == "inverse" else 1.0 / np.sqrt(cnt)
    w = per_group[inv]
    return w / w.sum() * len(labels), uniq, cnt, per_group


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = "source/isaaclab_tasks/isaaclab_tasks/direct"
    p.add_argument("--pkl_dir", default=f"{root}/go2_imitation_latent/imitation/motion_pkl")
    p.add_argument("--motion_lib", default=f"{root}/go2_imitation_latent/motion_lib.py")
    p.add_argument(
        "--split_json",
        default="reports/go2_imitation/go2_imitation_tracking/"
        "2026-08-27_phase0_dataset_build/metrics/session_split.json",
    )
    p.add_argument("--out_dir", required=True)
    p.add_argument("--latent_dim", type=int, default=18)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--num_experts", type=int, default=6)
    p.add_argument(
        "--latent_dist",
        choices=["gauss", "vmf"],
        default="gauss",
        help="latent 분포. vmf 는 z 를 단위 hypersphere 위에 둔다 (논문 §V).",
    )
    p.add_argument(
        "--kappa_fixed",
        type=float,
        default=0.0,
        help="vmf 집중도를 이 값으로 고정한다 (>0). 고정하면 KL 이 방향에 무관한 "
        "상수가 되어 posterior collapse 압력이 0 이다 — 가설의 핵심 변형.",
    )
    p.add_argument(
        "--gauss_fixed_logvar",
        type=float,
        default=None,
        help="[대조군] gauss 의 logvar 를 이 상수로 고정한다. vmf 의 이득이 구(球) "
        "기하 때문인지 단순히 z 의 SNR 때문인지 가른다.",
    )
    p.add_argument(
        "--window_w",
        type=int,
        default=2,
        help="인코더가 보는 프레임 수 (창 길이). 2 = 상태전이 쌍(기존, 기본값). "
        "w>2 면 인코더 입력이 49*w 로 넓어져 z 가 보행 위상을 담을 수 있다. "
        "**디코더는 항상 한 프레임(x[t])만 조건으로 받는다** — 한 스텝 예측 유지.",
    )
    p.add_argument(
        "--predict_ahead",
        type=int,
        default=1,
        help="과제 2 — (x[t], x[t+k]) 를 예측한다. k 가 크면 x[t] 만으로 결정이 안 되어 z 가 필요해진다.",
    )
    p.add_argument(
        "--lambda_speed",
        type=float,
        default=0.0,
        help="축 정렬 보조 손실의 가중치. L = L_recon + beta*L_kl + lambda_speed*L_speed. "
        "0 이면 기존과 동일(헤드는 만들어지되 손실에 들어가지 않는다).",
    )
    p.add_argument(
        "--speed_target",
        choices=["step", "clip"],
        default="step",
        help="보조 손실의 회귀 표적. step=해당 전이의 순간 vx, clip=그 클립의 평균 vx.",
    )
    p.add_argument(
        "--balance_speed_only",
        action="store_true",
        help="--balance 가중치를 **샘플링이 아니라 보조 손실에만** 적용한다. "
        "walk 가 71.9%% 라 그냥 두면 공유 w 가 walk 에 맞춰진다.",
    )
    p.add_argument(
        "--balance",
        choices=["none", "inverse", "sqrt"],
        default="none",
        help="카테고리(또는 세션) 균형 샘플링. none=프레임 비례(현행), "
        "inverse=표본 수의 역수로 균등화, sqrt=역수의 제곱근(완화된 균형). "
        "상태전이 배치와 autoregressive 윈도우에 **같은 가중치**를 적용한다.",
    )
    p.add_argument(
        "--balance_by",
        choices=["category", "session"],
        default="category",
        help="균형의 단위. category=보행 종류, session=촬영 세션. trot 은 train 세션이 "
        "4개뿐이라 카테고리를 균등화해도 축은 소수 세션에 의존한다.",
    )
    p.add_argument(
        "--balance_groups",
        choices=["main", "prefix"],
        default="main",
        help="balance_by=category 일 때의 묶음. main=walk/run/trot/other 4군, "
        "prefix=파일명 접두사 12종 그대로. prefix 는 118 프레임짜리 전이 카테고리를 "
        "64배로 키우므로 과적합 위험이 크다.",
    )
    p.add_argument("--beta", type=float, default=0.05, help="KL 가중치 (논문 값)")
    p.add_argument("--free_bits", type=float, default=0.02, help="차원당 KL 하한 — posterior collapse 방어")
    p.add_argument("--epochs_tf", type=int, default=20, help="teacher forcing 구간 epoch")
    p.add_argument("--epochs_ar", type=int, default=60, help="autoregressive 전환 구간 epoch")
    p.add_argument("--ar_len", type=int, default=8, help="autoregressive 롤아웃 길이")
    p.add_argument("--ar_batch", type=int, default=512, help="autoregressive 롤아웃 배치 크기")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="모션 서브샘플 간격. 60fps 원본에서 2 면 30fps. x[t] 가 x[t-1] 로부터 "
        "덜 결정적이 되어 latent 이 쓰일 여지가 생긴다.",
    )
    p.add_argument(
        "--vel_dropout",
        type=float,
        default=0.0,
        help="decoder 조건에서 **속도 블록만** 확률적으로 0. 외삽 지름길을 끊어 "
        "z 를 읽게 만든다. cond_dropout 과 달리 자세 조건은 유지된다.",
    )
    p.add_argument(
        "--pose_dropout",
        type=float,
        default=0.0,
        help="decoder 조건에서 **자세 블록만** 확률적으로 0 (vel_dropout 의 여집합). "
        "디코더는 x_prev 에서 자세를 그대로 베껴 쓸 수 있어 z 가 자세를 담을 유인이 "
        "없다 — 그 지름길을 끊는다. cond_dropout 과 달리 속도 조건은 유지된다.",
    )
    p.add_argument(
        "--cond_dropout",
        type=float,
        default=0.0,
        help="decoder 조건 x_prev 를 확률적으로 0 으로 만든다. z 가 정보를 나르도록 강제.",
    )
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train, val = load_dataset(args)
    print(
        f"[데이터] train 클립 {len(train['clips'])} (세션 {len({c['session'] for c in train['clips']})}), "
        f"val 클립 {len(val['clips'])} (세션 {len({c['session'] for c in val['clips']})})"
    )

    k_ahead = max(1, args.predict_ahead)
    w = max(2, int(args.window_w))
    lab_key = "prefix" if args.balance_by == "category" else "session"
    tr_prev, tr_curr, tr_lab = make_pairs(train["clips"], k_ahead, lab_key, w)
    va_prev, va_curr = make_pairs(val["clips"], k_ahead, None, w)
    if args.balance_by == "category":
        tr_lab = np.array([group_of(v, args.balance_groups) for v in tr_lab])
    # ★ 창이 넓어지면 클립당 쌍이 (w-1)*k 만큼 줄고 짧은 클립은 통째로 빠진다.
    #   w 별 비교에서 이 수의 감소분을 반드시 함께 읽어야 한다.
    n_tr_clip = sum(1 for c in train["clips"] if len(c["x"]) > (w - 1) * k_ahead)
    n_va_clip = sum(1 for c in val["clips"] if len(c["x"]) > (w - 1) * k_ahead)
    print(
        f"[데이터] 창 w={w} — 전이 표본 train {len(tr_prev):,} (클립 {n_tr_clip}/{len(train['clips'])})  "
        f"val {len(va_prev):,} (클립 {n_va_clip}/{len(val['clips'])})"
    )

    # 정규화 통계는 **train 에서만** 뽑는다 (val 누수 방지)
    mean = tr_curr.mean(0)
    std = tr_curr.std(0)
    std[std < 1e-4] = 1.0
    np.savez(os.path.join(args.out_dir, "norm.npz"), mean=mean, std=std)

    def norm(a: np.ndarray) -> torch.Tensor:
        """프레임 단위 정규화. 문맥 묶음(폭이 X_DIM 의 배수)이면 프레임마다 같은 통계를 쓴다."""
        n = a.shape[-1] // X_DIM
        return torch.from_numpy((a - np.tile(mean, n)) / np.tile(std, n)).to(dev)

    # ── 축 정렬 보조 손실의 표적 vx (정규화 공간) ──────────────
    # 정규화는 선형 변환이라 (w, b) 가 흡수한다. 상관·전이 상관은 영향을 받지 않는다.
    vx_i = FEATURE_SLICES["lin_vel"].start
    if args.speed_target == "clip":
        tgt_np = np.concatenate(
            [
                np.full(len(c["x"]) - (w - 1) * k_ahead, c["x"][:, vx_i].mean(), np.float32)
                for c in train["clips"]
                if len(c["x"]) > (w - 1) * k_ahead
            ]
        )
    else:
        tgt_np = tr_curr[:, vx_i].copy()
    tr_v = torch.from_numpy((tgt_np - mean[vx_i]) / std[vx_i]).to(dev)

    tr_p, tr_c = norm(tr_prev), norm(tr_curr)
    va_p, va_c = norm(va_prev), norm(va_curr)

    # autoregressive 롤아웃용 — 길이 ar_len+1 윈도우를 미리 쌓아 배치로 돌린다.
    # (시퀀스 하나씩 돌리면 gradient 가 지나치게 noisy 하다)
    windows, win_lab = [], []
    # 창 인코더는 롤아웃 첫 스텝에서도 w-1 프레임이 필요하므로 앞쪽에 (w-2) 프레임을 더 담는다.
    # w=2 면 span 과 슬라이스가 기존과 동일하다.
    lead = (w - 2) * k_ahead
    span = args.ar_len * k_ahead
    for c in train["clips"]:
        xs = (c["x"] - mean) / std
        lab = group_of(c["prefix"], args.balance_groups) if args.balance_by == "category" else c["session"]
        for s in range(0, len(xs) - lead - span - 1, max(span // 2, 1)):
            windows.append(xs[s : s + lead + span + 1 : k_ahead])
            win_lab.append(lab)
    tr_win = torch.from_numpy(np.stack(windows)).to(dev) if windows else None
    print(
        f"[데이터] autoregressive 윈도우 {0 if tr_win is None else len(tr_win):,}개 "
        f"(길이 {(w - 2) + args.ar_len + 1} = 문맥 {w - 1} + 표적 {args.ar_len})"
    )

    # ── 균형 샘플링 가중치 ────────────────────────────────────────
    # 상태전이 배치와 autoregressive 윈도우 **양쪽**에 같은 가중치를 건다.
    # 한쪽만 균형 잡으면 나머지 절반이 여전히 walk 비례라 반쪽짜리가 된다.
    w_pair = w_win = None
    sw_pair = sw_win = None  # 보조 손실 전용 가중치 (--balance_speed_only)
    if args.balance != "none":
        w_np, uniq, cnt, per_g = balance_weights(tr_lab, args.balance)
        w_pair = torch.from_numpy(w_np).to(dev).double()
        # 윈도우 가중치는 **쌍과 같은 그룹별 배율**을 쓴다 (윈도우 수로 다시 세지 않는다).
        lut = dict(zip(uniq.tolist(), per_g.tolist()))
        ww = np.array([lut.get(v, 0.0) for v in win_lab])
        if ww.sum() <= 0:
            raise SystemExit("균형 가중치가 전부 0 이다 — 윈도우 라벨과 쌍 라벨의 그룹이 어긋난다.")
        w_win = torch.from_numpy(ww / ww.sum() * len(ww)).to(dev).double()
        if args.balance_speed_only:
            # 샘플링은 건드리지 않고 **보조 손실의 표본 가중치**로만 쓴다.
            sw_pair = w_pair.float()
            sw_win = w_win.float()
            w_pair = w_win = None
        share = np.array([w_np[tr_lab == u].sum() for u in uniq]) / len(tr_lab)
        print(
            f"[균형] {args.balance} / {args.balance_by}"
            + (f" / {args.balance_groups}" if args.balance_by == "category" else "")
            + f" — 그룹 {len(uniq)}개"
        )
        for i in np.argsort(-cnt)[:8]:
            frac = cnt[i] / len(tr_lab)
            print(
                f"        {str(uniq[i]):<12} 표본 {cnt[i]:>7,} ({frac:>5.1%})"
                f"  ->  추출 비중 {share[i]:>5.1%}  (배율 {share[i] / frac:>5.2f}x)"
            )
        if len(uniq) > 8:
            print(f"        ... 그룹 {len(uniq) - 8}개 생략")

    model = MotionVAE(
        args.latent_dim,
        args.hidden,
        args.num_experts,
        latent_dist=args.latent_dist,
        kappa_fixed=args.kappa_fixed,
        gauss_fixed_logvar=args.gauss_fixed_logvar,
        window_w=w,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_par = sum(q.numel() for q in model.parameters())
    print(
        f"[모델] latent {args.latent_dim}({args.latent_dist}), hidden {args.hidden}, "
        f"expert {args.num_experts}, 파라미터 {n_par:,}, predict_ahead {k_ahead}, "
        f"인코더 입력 {w * X_DIM} (창 {w} 프레임)"
    )
    if args.latent_dist == "vmf":
        if args.kappa_fixed > 0.0:
            kc = torch.tensor([args.kappa_fixed], dtype=torch.float64)
            print(
                f"[모델] kappa 고정 {args.kappa_fixed:g} — KL = {float(vmf_kl_uniform(kc, args.latent_dim)):.3f} nats "
                f"(방향 무관 상수 → gradient 0, 손실에서 제외), E[z.mu] = "
                f"{float(vmf_mean_resultant(kc, args.latent_dim)):.4f}"
            )
        else:
            print("[모델] kappa 학습 — KL 닫힌형 (Davidson et al. 2018) 사용")

    zero = torch.zeros((), device=dev)

    def speed_loss(mu: torch.Tensor, v_tgt: torch.Tensor, w: torch.Tensor | None = None) -> torch.Tensor:
        """공유 선형 헤드로 z 에서 vx 를 회귀한다. gradient 는 **인코더까지** 흐른다.

            v_hat   = w_head . z + b        (walk/run/trot 이 하나의 (w, b) 를 공유)
            L_speed = mean( (v_hat - vx_true)^2 )

        헤드만 학습하면 프로브를 하나 더 만드는 것에 불과하므로 mu 를 detach 하지 않는다.
        """
        v_hat = model.speed_head(mu).squeeze(-1)
        se = (v_hat - v_tgt) ** 2
        return se.mean() if w is None else (se * w).sum() / w.sum()

    def kl_term(mu: torch.Tensor, aux: torch.Tensor) -> torch.Tensor:
        if args.latent_dist == "gauss":
            per_dim = 0.5 * (mu.pow(2) + aux.exp() - 1.0 - aux)
            return per_dim.clamp(min=args.free_bits).sum(-1).mean()
        if args.kappa_fixed > 0.0:
            # 방향에 무관한 상수. gradient 가 0 이므로 손실에 넣지 않는다 (붕괴 압력 0).
            return zero
        return vmf_kl_uniform(aux, args.latent_dim).mean()

    @torch.no_grad()
    def evaluate(xp: torch.Tensor, xc: torch.Tensor) -> tuple[float, float]:
        model.eval()
        rec_sum = kl_sum = 0.0
        for i in range(0, len(xp), args.batch):
            a, b = xp[i : i + args.batch], xc[i : i + args.batch]
            xh, mu, lv = model(a, b)
            rec_sum += F.mse_loss(xh, b, reduction="sum").item()
            kl_sum += kl_term(mu, lv).item() * len(a)
        return rec_sum / (len(xp) * X_DIM), kl_sum / len(xp)

    history = []
    total_ep = args.epochs_tf + args.epochs_ar
    t0 = time.time()
    for ep in range(total_ep):
        model.train()
        # 논문 일정: teacher forcing 후 autoregressive 비중을 선형으로 올린다
        ar_ratio = 0.0 if ep < args.epochs_tf else min(1.0, (ep - args.epochs_tf + 1) / max(args.epochs_ar * 0.5, 1))
        # balance=none 은 기존과 **완전히 동일한** randperm 경로를 탄다 (bit-identical 보장).
        if w_pair is None:
            perm = torch.randperm(len(tr_p), device=dev)
        else:
            perm = torch.multinomial(w_pair, len(tr_p), replacement=True)
        rec_acc = kl_acc = sp_acc = 0.0
        nb = 0
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            a, b = tr_p[idx], tr_c[idx]
            xh, mu, lv = model(a, b, args.cond_dropout, args.vel_dropout, args.pose_dropout)
            rec = F.mse_loss(xh, b)
            kl = kl_term(mu, lv)
            loss = rec + args.beta * kl
            if args.lambda_speed > 0.0:
                sp = speed_loss(mu, tr_v[idx], None if sw_pair is None else sw_pair[idx])
                loss = loss + args.lambda_speed * sp
                sp_acc += sp.item()

            if ar_ratio > 0.0 and tr_win is not None:
                n_w = min(args.ar_batch, len(tr_win))
                wsel = (
                    torch.randint(0, len(tr_win), (n_w,), device=dev)
                    if w_win is None
                    else torch.multinomial(w_win, n_w, replacement=True)
                )
                win = tr_win[wsel]  # [W, (w-2)+ar_len+1, X]
                # 문맥 버퍼: 실제 프레임 w-1 개로 시작해, 매 스텝 자기 예측을 밀어 넣는다.
                # w=2 면 버퍼가 곧 직전 예측 한 프레임이라 기존 경로와 동일하다.
                ctx = win[:, : w - 1].reshape(len(win), -1)
                ar_rec = 0.0
                for k in range(args.ar_len):
                    tgt = win[:, w - 1 + k]
                    xh_k, mu_k, lv_k = model(ctx, tgt, args.cond_dropout, args.vel_dropout, args.pose_dropout)
                    ar_rec = ar_rec + F.mse_loss(xh_k, tgt) + args.beta * kl_term(mu_k, lv_k)
                    if args.lambda_speed > 0.0:
                        # AR 절반에도 같은 손실을 건다 — 한쪽만 걸면 반쪽짜리다.
                        ar_rec = ar_rec + args.lambda_speed * speed_loss(
                            mu_k, tgt[:, vx_i], None if sw_win is None else sw_win[wsel]
                        )
                    ctx = torch.cat([ctx[:, X_DIM:], xh_k], -1)  # 자기 예측을 다음 조건으로 되먹임
                loss = loss + ar_ratio * ar_rec / args.ar_len

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            rec_acc += rec.item()
            kl_acc += kl.item()
            nb += 1

        if ep % 5 == 0 or ep == total_ep - 1:
            tr_rec, tr_kl = evaluate(tr_p, tr_c)
            va_rec, va_kl = evaluate(va_p, va_c)
            history.append(
                {
                    "epoch": ep,
                    "ar_ratio": ar_ratio,
                    "train_rec": tr_rec,
                    "train_kl": tr_kl,
                    "val_rec": va_rec,
                    "val_kl": va_kl,
                }
            )
            if args.lambda_speed > 0.0:
                print(f"  ep {ep:3d}  L_speed {sp_acc / max(nb, 1):.5f}")
            print(
                f"  ep {ep:3d}  ar {ar_ratio:.2f}  train rec {tr_rec:.5f} kl {tr_kl:6.3f}   "
                f"val rec {va_rec:.5f} kl {va_kl:6.3f}   ratio {va_rec / max(tr_rec, 1e-12):.2f}x"
            )

    tr_rec, _ = evaluate(tr_p, tr_c)
    va_rec, _ = evaluate(va_p, va_c)
    ratio = va_rec / max(tr_rec, 1e-12)

    # G0 — latent 활용도. z 를 죽였을 때 재구성이 얼마나 나빠지는가.
    #      1.0x 에 가까우면 posterior collapse 이고, 그러면 synthesis 정책이 조종할 대상이 없다.
    model.eval()
    with torch.no_grad():
        mu_v, _ = model.encode(va_p, va_c)
        va_last = va_p[:, -X_DIM:]  # 디코더 조건은 문맥의 마지막 프레임
        r_enc = F.mse_loss(model.decode(va_last, mu_v), va_c).item()
        r_zero = F.mse_loss(model.decode(va_last, torch.zeros_like(mu_v)), va_c).item()
        r_shuf = F.mse_loss(model.decode(va_last, mu_v[torch.randperm(len(mu_v), device=dev)]), va_c).item()
        # ★ vmf 에서 z=0 은 **구 위에 없는 점**이다. 그 비율이 크게 나와도 latent 활용의
        #   증거가 아니다 (분포 밖 입력에 대한 반응일 뿐). 구 위의 균등 난수를 함께 잰다.
        r_rand = F.mse_loss(model.decode(va_last, F.normalize(torch.randn_like(mu_v), dim=-1)), va_c).item()
    # 판정 통계는 **셔플**이다. z=0 은 prior 평균이라 decoder 가 그 점에 강건해질 수 있어
    # collapse 여부를 재는 지표로 부적절하다. 다른 표본의 z 를 끼워 넣었을 때 재구성이
    # 무너지는지가 "z 가 표본 고유 정보를 나르는가" 를 직접 잰다.
    g0 = r_shuf / max(r_enc, 1e-12)
    # ★ G0 는 **게이트가 아니라 진단 지표**다. 스텝 단위 z 를 셔플해도 decoder 가 x_prev 로
    #   복구할 수 있어 값이 1.0x 근처로 나오지만, 그렇다고 latent 이 비어 있는 것은 아니다.
    #   latent 이 실제로 무엇을 담는지는 G2a(선형 프로브)/G2b(보간 단조성)가 잰다.
    print("\n[G0] latent 활용도 (val) — 진단용, 게이트 아님")
    print(
        f"     z=encoder {r_enc:.6f}   z=셔플 {r_shuf:.6f} ({g0:.2f}x)   "
        f"z=단위난수 {r_rand:.6f} ({r_rand / max(r_enc, 1e-12):.2f}x)"
        + (
            f"   [무효-구밖] z=0 {r_zero / max(r_enc, 1e-12):.2f}x"
            if args.latent_dist == "vmf"
            else f"   [참고] z=0 {r_zero:.6f} ({r_zero / max(r_enc, 1e-12):.2f}x)"
        )
    )
    print("     (참고: 이 값이 1.0x 에 가까워도 G2a/G2b 가 통과하면 latent 은 구조를 갖고 있다)")
    print("\n[G1] leave-one-session-out 재구성 (정규화 공간 MSE/차원)")
    print(
        f"     in-sample {tr_rec:.6f}   held-out 세션 {va_rec:.6f}   비율 {ratio:.3f}x   "
        f"기준 <= 2.0x  ->  {'PASS' if ratio <= 2.0 else 'FAIL'}"
    )
    print(f"[시간] {time.time() - t0:.1f}s")

    torch.save(
        {
            "model": model.state_dict(),
            "args": vars(args),
            "mean": mean,
            "std": std,
            "history": history,
            "g1": {"train_rec": tr_rec, "val_rec": va_rec, "ratio": ratio},
        },
        os.path.join(args.out_dir, "motion_vae.pt"),
    )
    json.dump(
        {
            "history": history,
            "g0_ratio": g0,
            "g0_pass": bool(g0 >= 2.0),
            "g0_r_enc": r_enc,
            "g0_r_zero": r_zero,
            "g0_r_shuf": r_shuf,
            "g0_r_rand": r_rand,
            "g0_rand_ratio": r_rand / max(r_enc, 1e-12),
            "lambda_speed": args.lambda_speed,
            "speed_target": args.speed_target,
            "balance_speed_only": bool(args.balance_speed_only),
            "balance": args.balance,
            "balance_by": args.balance_by,
            "balance_groups": args.balance_groups,
            "latent_dist": args.latent_dist,
            "kappa_fixed": args.kappa_fixed,
            "gauss_fixed_logvar": args.gauss_fixed_logvar,
            "predict_ahead": k_ahead,
            "window_w": w,
            "enc_input_dim": w * X_DIM,
            "num_ar_windows": 0 if tr_win is None else int(len(tr_win)),
            "g1_train_rec": tr_rec,
            "g1_val_rec": va_rec,
            "g1_ratio": ratio,
            "g1_pass": bool(ratio <= 2.0),
            "num_params": n_par,
            "train_pairs": int(len(tr_prev)),
            "val_pairs": int(len(va_prev)),
        },
        open(os.path.join(args.out_dir, "train_summary.json"), "w"),
        indent=2,
    )
    print(f"[저장] {args.out_dir}")


if __name__ == "__main__":
    main()

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Latent style scoring — 학습된 motion VAE 인코더를 스타일 채점기로 쓴다 (Phase 2, 안 B).

AMP discriminator 를 latent 공간의 발산으로 대체하는 경로다. 디코더는 쓰지 않으므로
Phase 1 의 posterior collapse 와 무관하다 (인코더는 G2a LOO R² 0.83~0.95 로 통과했다).

LatentMimic (arXiv 2604.16440) 의 스타일 보상::

    d_step[t] = -log N( z[t] ; mu_ref, Sigma_ref )     # 낮을수록 expert 분포 안쪽
    r_style   = exp( -c_kl * d_step )

``mu_ref`` / ``Sigma_ref`` 는 expert 배치의 latent 통계(대각 근사)다. 스텝별 신호가
필요하므로 배치통계끼리의 KL 대신 참조 분포에 대한 로그밀도를 쓴다. 배치 marginal KL
(:meth:`LatentStyleScorer.marginal_kl`) 은 보상이 아니라 모니터링 지표다.

기호:

===========  ==========================================  ==========
기호         의미                                        단위
===========  ==========================================  ==========
``x_vae``    상태 특징 벡터 (49-D)                        혼합
``z``        latent 벡터 (18-D), 인코더 사후분포 평균     무차원
``d_step``   참조 분포에 대한 음의 로그밀도               nat
``c_kl``     스타일 보상 민감도                           1/nat
===========  ==========================================  ==========

인코더 입력은 **상태전이 쌍** ``(x_vae[t-1], x_vae[t])`` 이고, 두 프레임 모두 학습 시점의
동일한 정규화 통계(``x[t]`` 에서 뽑은 mean/std)로 정규화된다. 통계를 새로 만들면 학습
시점과 어긋나므로 반드시 체크포인트의 값을 쓴다.

지금은 **동결(frozen) 추론 경로만** 구현한다. 계획서 §8 이 경고하는 "동결 인코더가
exploit 당함" 에 대비해 주기적 재적합 경로를 나중에 붙일 수 있도록, 참조 통계는
:meth:`LatentStyleScorer.fit_reference` 로 언제든 갱신 가능하게 분리해 두었다.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn

from rsl_rl.utils import resolve_nn_activation


class MotionEncoder(nn.Module):
    """Motion VAE 의 인코더만 떼어낸 스타일 채점용 모듈.

    Args:
        x_dim: 프레임당 상태 특징 차원.
        latent_dim: latent 차원.
        hidden_dims: 은닉층 크기.
        activation: 활성함수 이름.
        logvar_clamp: ``logvar`` 클램프 범위 [최소, 최대]. 학습 시점과 동일해야 한다.
    """

    def __init__(
        self,
        x_dim: int = 49,
        latent_dim: int = 18,
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
        logvar_clamp: tuple[float, float] = (-8.0, 8.0),
    ) -> None:
        """인코더 MLP 와 정규화 버퍼를 만든다."""
        super().__init__()

        self.x_dim = x_dim
        self.latent_dim = latent_dim
        self.logvar_clamp = logvar_clamp

        act = resolve_nn_activation(activation)
        layers: list[nn.Module] = []
        prev = 2 * x_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(act)
            prev = h
        layers.append(nn.Linear(prev, 2 * latent_dim))
        self.encoder = nn.Sequential(*layers)

        # 정규화 통계 — 체크포인트에서 덮어쓴다. state_dict 에 함께 실려 배포 시 유실되지 않는다.
        self.register_buffer("obs_mean", torch.zeros(x_dim))
        self.register_buffer("obs_std", torch.ones(x_dim))

    # ── 정규화 ────────────────────────────────────────────────────

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """정규화 통계를 설정한다. ``x_prev`` 와 ``x_curr`` 에 **동일하게** 적용된다.

        Args:
            mean: 채널별 평균, shape [x_dim].
            std: 채널별 표준편차, shape [x_dim]. 0 에 가까운 값은 1 로 대체한다.
        """
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(-1).clone()
        std[std < 1e-4] = 1.0
        self.obs_mean.copy_(mean.to(self.obs_mean.device))
        self.obs_std.copy_(std.to(self.obs_std.device))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """원 특징 → 정규화 특징."""
        return (x - self.obs_mean) / self.obs_std

    # ── 인코딩 ────────────────────────────────────────────────────

    def encode(
        self, x_prev: torch.Tensor, x_curr: torch.Tensor, normalized: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """상태전이 쌍 → 사후분포 파라미터.

        Args:
            x_prev: 이전 프레임 특징, shape [N, x_dim].
            x_curr: 현재 프레임 특징, shape [N, x_dim].
            normalized: 입력이 이미 정규화되어 있으면 True.

        Returns:
            ``(mu, logvar)`` 각각 shape [N, latent_dim].
        """
        if not normalized:
            x_prev = self.normalize(x_prev)
            x_curr = self.normalize(x_curr)
        mu, logvar = self.encoder(torch.cat([x_prev, x_curr], dim=-1)).chunk(2, dim=-1)
        return mu, logvar.clamp(*self.logvar_clamp)

    def forward(self, x_prev: torch.Tensor, x_curr: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """상태전이 쌍 → latent ``z``. 샘플링하지 않고 사후분포 평균을 쓴다."""
        mu, _ = self.encode(x_prev, x_curr, normalized)
        return mu

    def inference(self, x_prev: torch.Tensor, x_curr: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """Gradient 없이 latent ``z`` 를 낸다 (롤아웃 중 스타일 채점용)."""
        with torch.no_grad():
            return self.forward(x_prev, x_curr, normalized)

    # ── 동결 ──────────────────────────────────────────────────────

    def freeze(self) -> None:
        """추론 전용으로 고정한다 (eval 모드 + gradient 차단)."""
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    # ── 체크포인트 ────────────────────────────────────────────────

    @classmethod
    def from_checkpoint(cls, path: str, device: str | torch.device = "cpu", freeze: bool = True) -> MotionEncoder:
        """``train_go2_motion_vae.py`` 체크포인트에서 인코더만 로드한다.

        체크포인트는 ``{"model": ..., "args": ..., "mean": ..., "std": ...}`` 구조이고,
        디코더(MoE)와 gating 가중치는 무시한다.

        Args:
            path: ``motion_vae.pt`` 경로.
            device: 로드할 장치.
            freeze: True 면 :meth:`freeze` 를 호출한다.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {})
        hidden = int(args.get("hidden", 256))
        latent_dim = int(args.get("latent_dim", 18))

        enc_state = {k[len("encoder.") :]: v for k, v in ckpt["model"].items() if k.startswith("encoder.")}
        if not enc_state:
            raise KeyError(f"체크포인트에 encoder 가중치가 없다: {path}")
        x_dim = enc_state["0.weight"].shape[1] // 2

        model = cls(x_dim=x_dim, latent_dim=latent_dim, hidden_dims=[hidden, hidden])
        model.encoder.load_state_dict(enc_state)
        model.set_normalization(torch.as_tensor(ckpt["mean"]), torch.as_tensor(ckpt["std"]))
        model.to(device)
        if freeze:
            model.freeze()
        return model


class LatentStyleScorer:
    """expert latent 분포에 대한 발산 척도와 스타일 보상.

    참조 분포는 expert 배치의 latent 통계를 대각 가우시안으로 근사한 것이다.
    :meth:`fit_reference` 를 다시 호출하면 참조가 갱신되므로, 인코더 재적합이나
    expert 배치 교체를 지원한다.

    Args:
        c_kl: 스타일 보상 민감도 [1/nat].
        var_floor: 분산 하한 — 축소된 latent 차원에서 발산이 폭주하는 것을 막는다.
    """

    def __init__(self, c_kl: float = 0.01, var_floor: float = 1e-4) -> None:
        """참조 분포를 비운 채로 만든다 — :meth:`fit_reference` 를 먼저 호출해야 한다."""
        self.c_kl = c_kl
        self.var_floor = var_floor
        self.mu_ref: torch.Tensor | None = None
        self.var_ref: torch.Tensor | None = None
        self.z_ref: torch.Tensor | None = None

    # ── 참조 적합 ─────────────────────────────────────────────────

    def fit_reference(self, z_ref: torch.Tensor, keep_samples: bool = False) -> None:
        """Expert latent 배치로 참조 분포를 적합한다.

        Args:
            z_ref: expert latent, shape [N, latent_dim].
            keep_samples: True 면 k-NN 발산용으로 표본을 보관한다 (메모리 비용 있음).
        """
        self.mu_ref = z_ref.mean(dim=0)
        self.var_ref = z_ref.var(dim=0, unbiased=False).clamp(min=self.var_floor)
        self.z_ref = z_ref.detach().clone() if keep_samples else None

    def _check(self) -> None:
        if self.mu_ref is None or self.var_ref is None:
            raise RuntimeError("참조 분포가 적합되지 않았다 — fit_reference 를 먼저 호출할 것")

    # ── 발산 척도 ─────────────────────────────────────────────────

    def neg_log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """``d_step = -log N(z; mu_ref, Sigma_ref)`` [nat], shape [N].

        상수 오프셋 ``0.5 * sum(log(2*pi*var))`` 를 포함한 절대 로그밀도다. 보상 설계에서
        문제가 되는 것은 expert 대비 상대 격차이므로 오프셋을 따로 빼서 쓰는 것을 권한다.
        """
        self._check()
        d2 = (z - self.mu_ref).pow(2) / self.var_ref
        return 0.5 * (d2 + torch.log(2.0 * math.pi * self.var_ref)).sum(dim=-1)

    def mahalanobis(self, z: torch.Tensor) -> torch.Tensor:
        """대각 마할라노비스 거리 [무차원], shape [N]."""
        self._check()
        return ((z - self.mu_ref).pow(2) / self.var_ref).sum(dim=-1).sqrt()

    def knn_distance(self, z: torch.Tensor, k: int = 8, chunk: int = 4096) -> torch.Tensor:
        """Expert latent 집합에 대한 k-NN 평균 거리 [무차원], shape [N].

        ``fit_reference(..., keep_samples=True)`` 가 선행되어야 한다.
        """
        if self.z_ref is None:
            raise RuntimeError("k-NN 발산은 표본이 필요하다 — fit_reference(keep_samples=True)")
        out = []
        for i in range(0, len(z), chunk):
            d = torch.cdist(z[i : i + chunk], self.z_ref)
            out.append(d.topk(k, dim=-1, largest=False).values.mean(dim=-1))
        return torch.cat(out)

    def marginal_kl(self, z: torch.Tensor) -> torch.Tensor:
        """배치 marginal KL ``KL( N(mu_pi, Sigma_pi) || N(mu_ref, Sigma_ref) )`` [nat], 스칼라.

        스텝별 보상이 아니라 **모니터링 지표**다.
        """
        self._check()
        mu_pi = z.mean(dim=0)
        var_pi = z.var(dim=0, unbiased=False).clamp(min=self.var_floor)
        return (
            0.5
            * (torch.log(self.var_ref / var_pi) + (var_pi + (mu_pi - self.mu_ref).pow(2)) / self.var_ref - 1.0).sum()
        )

    # ── 보상 ──────────────────────────────────────────────────────

    def style_reward(self, z: torch.Tensor, offset: float = 0.0, c_kl: float | None = None) -> torch.Tensor:
        """``r_style = exp(-c_kl * (d_step - offset))`` [무차원], shape [N].

        Args:
            z: 정책 롤아웃 latent, shape [N, latent_dim].
            offset: ``d_step`` 오프셋 [nat]. expert 중앙값을 넣으면 보상이 expert 에서 1
                근처가 되어 ``c_kl`` 이 절대 로그밀도 상수에 지배되지 않는다.
            c_kl: 민감도 [1/nat]. None 이면 생성자 값을 쓴다.
        """
        c = self.c_kl if c_kl is None else c_kl
        return torch.exp(-c * (self.neg_log_prob(z) - offset))

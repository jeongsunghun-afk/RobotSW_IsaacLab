# 구현 완료 상태 — SPO / LCP / MoE-Loco / WASABI

**Date**: 2026-06-08 · **검증**: validate-code PASS(블로커 0) + 런타임 smoke test 8/8 PASS (isaac-5.1, CPU)
**원칙**: 4개 전부 **opt-in 플래그**. 미지정 시 기존 PPO/AMP/parkour 동작 **수치 불변**. 각자 cfg 1~몇 줄로 ON/롤백.

> 명세서: `deep/spo_spec.md`, `deep/smoothness_spec.md`, `deep/moe_spec.md`, `deep/wasabi_spec.md`
> 종합/랭킹: `SYNTHESIS_decision_gate.md`

---

## 1. SPO (Simple Policy Optimization)
- **무엇**: PPO surrogate를 quadratic ratio penalty로 교체 `f_spo = r·A − (|A|/2ε)(r−1)²` (arXiv:2401.16025 Eq.16).
- **변경 파일**: `ppo.py:321-329`, `ppo_parkour.py:351-359` (surrogate 분기), `rl_cfg.py`(필드 2개).
- **켜는 법** — agent cfg의 `algorithm`(RslRlPpoAlgorithmCfg):
  ```python
  algorithm = RslRlPpoAlgorithmCfg(
      ...,
      surrogate_type="spo",   # default "ppo"
      spo_epsilon=0.2,        # PPO clip ε와 동일 스케일
  )
  ```
- **A/B**: 동일 task/seed로 `ppo` vs `spo`. 비교: mean reward, KL(adaptive 경로), `|r−1|` 분포 폭, 안정성. ε∈{0.1,0.2,0.3} sweep. SPO 시 `schedule:fixed` 권장(논문 근접).
- 적용 범위: PPO/PPOParkour/PPOAMPBase/PPOAMP 전부(상속). deploy 영향 0.

## 2. LCP (Lipschitz-Constrained Policy, action smoothness)
- **무엇**: policy loss에 `λ·‖∇_obs log π‖²` gradient penalty (arXiv:2410.11825). sim-to-real jitter↓.
- **변경 파일**: `ppo.py`(penalty 블록 + ctor), `actor_critic.py`(`log_prob_from_actor_obs` helper), `rl_cfg.py`(`RslRlLcpCfg`).
- **켜는 법**:
  ```python
  from isaaclab_rl.rsl_rl import RslRlLcpCfg
  algorithm = RslRlPpoAlgorithmCfg(
      ...,
      lcp_cfg=RslRlLcpCfg(lambda_gp=0.002, penalize="log_prob"),  # default None=off; "mean"=cheaper
  )
  ```
- **A/B**: baseline vs LCP λ_gp∈{0.001,0.002,0.005}. 측정: action rate/jerk 95th-pct(목표 ≥30~50%↓), command-tracking error 유지. over-smoothing→agile 둔화 주의(λ schedulable 권고).
- **범위 주의**: **base feed-forward(PPO+ActorCritic)** 만. parkour recurrent(PPOParkour/RMA)는 hidden-state·encoder 입력 때문에 **별도 설계 필요**(현재 ppo_parkour는 lcp_cfg를 받되 무시). CAPS(temporal+spatial)는 이번에 미구현(temporal은 RolloutStorage next_obs 버퍼 필요).

## 3. MoE-Loco (Mixture-of-Experts actor)
- **무엇**: 단일 MLP actor → dense softmax gating[128] + 6 expert (arXiv:2503.08564). load-balancing loss 없음 → PPO 무수정.
- **변경 파일**: 신규 `actor_critic_moe.py`, `modules/__init__.py`, `on_policy_runner.py:21`(import), `rl_cfg.py`(subclass).
- **켜는 법** — agent cfg의 `policy`를 교체:
  ```python
  from isaaclab_rl.rsl_rl import RslRlPpoActorCriticMoECfg
  policy = RslRlPpoActorCriticMoECfg(
      num_experts=6, gating_hidden_dims=[128],
      expert_hidden_dims=[256,256,256], gating_temperature=1.0,
  )
  ```
- **A/B**: 단일 MLP vs MoE, **multi-terrain** task에서 per-terrain reward attribution(기존 도구), 수렴/return, gating entropy(collapse 모니터). N∈{2,4,6,8} sweep.
- **⚠ framing**: MoE는 **3-leg gait를 고치지 않음**(그건 단일 terrain reward 비대칭 local-opt). 이득은 parkour가 distinct regime을 실제 span할 때. 미검증 가설. `state_dependent_std=True`면 NotImplementedError(현재 미지원). 1차 actor-only(RMA 결합 후속).

## 4. WASABI (Wasserstein-AMP)
- **무엇**: AMP discriminator LS-GAN → WGAN critic `−E[D_exp]+E[D_pol]`, reward `(D−μ̂)/σ̂`(signed) (arXiv:2206.11693).
- **변경 파일**: `ppo_amp.py:126-131`(PPOAMPBase)+`344-349`(PPOAMP) loss 분기, `amp_discriminator.py`(reward 분기 + `reward_normalizer=EmpiricalNormalization(1)`).
- **켜는 법** — agent cfg의 `amp_cfg` dict:
  ```python
  amp_cfg = {
      ..., "disc_loss_type": "wgan", "disc_reward_type": "wgan",  # 둘 다 wgan 일치 필수
      "gradient_penalty_coef": 5.0,   # both-sided GP라 5.0 시작(기존 10 재사용 가능)
      "reward_coef": 0.5,             # unit-var reward → LS-GAN(1.5~2) 보다 작게
  }
  ```
- **A/B**: `ls_gan` vs `wgan` 동 task. discriminator collapse/발산 빈도, AMP reward 곡선, imitation 품질. GP coef·reward_coef sweep.
- **⚠ 최상위 리스크**: WGAN reward는 signed(절반 음수) → **parkour `total_reward.clip(min=0)`과 충돌**(신호 절반 소실). **비-parkour Go2-Imitation 경로 우선 검증.** parkour 적용은 clip 상호작용 사전 검토. reward scale 재튜닝 필요. deploy 영향 0(discriminator 학습 전용).

---

---

# [업데이트 2026-06-08] 등록된 새 환경 (RMA 확장 포함)

유저 요청: SPO/LCP/MoE는 parkour에, WASABI는 go2_imitation에 새 algorithm cfg + env 등록. LCP/MoE는 parkour의 RMA 스택에 **확장 구현**(유저 선택). 정적+런타임 검증 PASS (LCP penalty가 RMA에서 nonzero gradient 0.0889 실측 — no-op 아님).

## 새 task id + 학습 명령

| task id | 메소드 | cfg 클래스 | base env |
|---------|--------|-----------|----------|
| `Go2-Parkour-Direct-SPO` | SPO | `Go2ParkourSPOPPORunnerCfg` | parkour |
| `Go2-Parkour-Direct-LCP` | LCP(RMA 확장) | `Go2ParkourLCPPPORunnerCfg` | parkour |
| `Go2-Parkour-Direct-MoE` | MoE(ActorCriticRMAMoE) | `Go2ParkourMoEPPORunnerCfg` | parkour |
| `Go2-Imitation-WASABI-v0` | WASABI(WGAN-AMP) | `Go2ImitationWASABIPPORunnerCfg` | go2_imitation |

```bash
# 각 메소드 학습 (parkour는 isaac-parkour env, 학습은 ./isaaclab.sh -p)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Parkour-Direct-SPO --num_envs 4096 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Parkour-Direct-LCP --num_envs 4096 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Parkour-Direct-MoE --num_envs 4096 --headless
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Imitation-WASABI-v0 --num_envs 4096 --headless
# A/B baseline: Go2-Parkour-Direct-v0 / Go2-Imitation-v0 (기존, 불변)
```

## RMA 확장 구현 노트
- **LCP-RMA**: `ppo_parkour.py` PPOParkour.update()에 penalty 추가. grad 대상 = **actor MLP 입력(obs+encoder latent concat)** — encoder까지는 미분 안 함(의도된 단순화, deploy smoothness 목적). `lcp_cfg=None`이면 기존 동작 불변.
- **MoE-RMA**: 신규 `ActorCriticRMAMoE(ActorCriticRMA)` — `self.actor`만 MoEActor로 교체, gating 입력 = RMA actor 입력(MoE-Loco ĥ_t 대응). RMA encoder/adaptation/dagger/priv_reg 전부 부모 상속 보존. method override 0개. `state_dependent_std=True` 미지원(NotImplementedError).
- 새 등록은 전부 별도 task id → 기존 `Go2-Parkour-Direct-v0`/`Go2-Imitation-v0` 불변.

## A/B 측정 포인트 (재확인)
- **SPO**: vs `-v0`, KL/reward/안정성.
- **LCP**: action rate/jerk↓ vs tracking 유지. ⚠ parkour의 agile jump 둔화 주의(λ schedulable 고려).
- **MoE**: per-terrain reward attribution(기존 도구), gating entropy(collapse 모니터). ⚠ 3-leg gait 개선 기대 금지(multi-terrain regime 분화가 표적).
- **WASABI**: ls_gan/bce vs wgan, discriminator collapse 빈도. ⚠ go2_imitation은 `clip(min=0)` 없으니 signed reward 안전.

---

## 다음 단계 (권장)
1. **포맷**(선택): `./isaaclab.sh -f` (repo-wide ruff — git status 다수 수정파일 동시 영향 주의).
2. **학습 A/B**: 위 순서(SPO→LCP→MoE→WASABI)로 한 번에 하나씩 플래그 ON, baseline 대비 측정. 서로 독립 플래그라 조합도 가능하나 효과 분리는 개별 A/B.
3. **WASABI는 Go2-Imitation부터**, MoE는 multi-terrain부터, LCP는 base feed-forward부터.

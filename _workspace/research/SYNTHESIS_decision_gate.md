# Research Synthesis — 구현 후보 결정 게이트

**Date**: 2026-06-08
**목적**: rsl_rl에 추가 구현할 네트워크/알고리즘/IL 메소드 리서치 종합 → 무엇을 먼저 구현할지 결정.
**근거 문서**: `00_codebase_inventory.md`, `01_network_architectures.md`, `02_algorithms.md`, `03_imitation_parkour.md`

---

## 핵심 발견 (3가지)

1. **이 코드베이스는 이미 심하게 커스터마이즈됨.** PPO+RND+symmetry, AMP 전체 suite(LS-GAN/BCE+GP+replay+logit-reg), RMA(history encoder+DAGGER+estimator), student-teacher distillation, depth backbones(FC/recurrent/stacked), 적응형 KL-LR, advantage norm, GAE가 **이미 존재**. → 문헌의 단골 후보 상당수가 이미 있음.

2. **비어있는 영역이 명확.** Transformer/attention, MoE, off-policy(SAC/TD3), world-model, hierarchical, **action smoothness loss(CAPS/LCP)**, **SPO(KL-clip surrogate)**, **Wasserstein-AMP**, **skill latent(ASE/CALM)** 은 없음 → 진짜 delta.

3. **SAC/TD3는 (유저가 지명했지만) 정직하게 down-rank.** 수천 parallel env on-policy 환경에서 off-policy의 sample-efficiency 이점이 대부분 상쇄되는 반면, replay buffer+twin critic+새 runner로 core infra를 갈아엎어야 함. **실로봇 on-robot fine-tuning으로 전환할 때** 비로소 가치(그때도 CrossQ가 최저비용 진입점). 지금은 보류 권장.

---

## 가장 강한 교차 추천: "저비용 drop-in" 클러스터

세 트랙에서 공통적으로 **PPO loop를 거의 안 건드리거나 module/loss만 교체하는 저비용·고ROI 후보**가 모임. 서로 독립적이라 병렬 구현 가능:

| # | 후보 | 트랙 | 무엇을 바꾸나 | 통합 비용 | 핵심 가치 |
|---|------|------|--------------|-----------|-----------|
| **1** | **SPO** (Simple Policy Optimization) | 알고리즘 | PPO ratio-clip → KL-clip surrogate | **Low** (~15줄, `ppo.py`, 기존 KL 재사용) | 유저 명시 "PPO 변형", Marco Hutter 공저, 저위험 A/B |
| **2** | **CAPS** 또는 **LCP** (action smoothness) | 알고리즘/loss | policy loss에 temporal/spatial(또는 Lipschitz GP) 항 추가 | **Low** (drop-in loss, symmetry/AMP-GP 패턴 재사용) | **sim-to-real 최대 이득** (Go2 jitter/에너지↓) |
| **3** | **MoE-Loco** (Mixture-of-Experts actor) | 네트워크 | 단일 MLP actor → gating+N expert | **Low** (`ActorCriticMoE` 새 module, PPO 무수정) | 진짜 novel 구조, parkour multi-terrain + 3-leg local-opt(gradient conflict) 직격 |
| **4** | **WASABI** (Wasserstein-AMP) | IL | `amp_discriminator` loss → WGAN critic | **Low** (loss 교체, GP 인프라 재사용) | 사족 sim-to-real 검증, partial demo 학습, AMP reward 불안정 이력 직결 |

> 이 4개는 각각 다른 파일을 건드려 충돌이 적고, 모두 deploy-safe(추가 센서 없음). **첫 구현 배치로 최적.**

---

## 트랙별 전체 랭킹 (참고)

### 트랙 1 — 네트워크 구조
- **drop-in(싸다)**: MoE-Loco ★, AME(attention map encoder, 입력=기존 height-scan), Transformer/ULT(학습 튜닝 비용 있음)
- **구조변경(비싸다)**: HIM·DreamWaQ(둘 다 기존 estimator와 구조 중복 + ppo loop 수정 → novelty 낮음), CPG-RL(env action pipeline 수정, 단 deploy 최강·A1 실증)
- **Top3**: ① MoE-Loco ② AME ③ CPG-RL

### 트랙 2 — 알고리즘/Loss
- **Group A drop-in(싸다)**: CAPS/LCP ★(smoothness), SPO ★(trust region), PopArt(value 정규화), Dual-Clip(2줄 안정화), PPG(중간, shared trunk일 때만)
- **Group B 새 family(비싸다·보류)**: CrossQ < SAC/TD3 < TQC/REDQ/DroQ
- **Top**: CAPS/LCP + SPO + (보조: Dual-Clip, PopArt)

### 트랙 3 — IL/Parkour
- **저비용(AMP/RMA 인프라 재사용)**: WASABI ★(WGAN loss 교체), AMP 안정화 묶음(Selective-AMP+action LPF), ASE(skill latent), Extreme Parkour 부분채택(yaw-goal conditioning, env-side)
- **고비용(새 파이프라인)**: CALM(ASE 위 확장), NCP/ControlVAE(VQ-VAE/world-model)
- **Top3**: ① WASABI ② AMP 안정화 묶음 ③ ASE

---

## 주의사항
- **인용 검증 필요**: 일부 arXiv ID(2604.x, 2601.x 등 미래/신규)는 구현 착수 전 원문 재확인 권장. 핵심 후보(CAPS 2012.06644, LCP 2410.11825, SPO 2401.16025, MoE-Loco 2503.08564, ASE 2205.01906, WASABI 2206.11693, Extreme Parkour 2309.14341)는 검증된 실재 논문.
- **CAPS vs LCP는 택일**: 같은 목적(smoothness), 동시 적용 불필요. CAPS=투명/2knob, LCP=1knob/legged 검증.
- **한 번에 하나씩 검증**: CLAUDE.md 원칙대로 동시 다발 변경 금지. drop-in이라도 A/B로 각각 효과 측정.

---

## 결정 필요 사항
**어떤 후보를 먼저 구현할지** 유저 선택 → 선택된 것만 worker로 구현 dispatch (각각 validate-method/validate-code 게이트 통과 후).

---

# [업데이트 2026-06-08] Deep-research 결과 + 구현 로드맵

유저가 4개(SPO, Action smoothness, MoE-Loco, WASABI) 전부 선택 + "구현 전 깊은 리서치" 요청 → 후보별 구현 명세서 작성 완료 (`deep/*.md`). 핵심 정정/발견:

## 후보별 구현 명세 요약

### 1. SPO — `deep/spo_spec.md` · 난이도 **LOW** ★최저위험
- **중요 정정**: 초기 요약모델이 "KL-clip piecewise"로 오독 → **PDF 원문 판독 결과 quadratic ratio penalty**: `f_spo = r·A − (|A|/2ε)(r−1)²` (Eq.16). 공개 GitHub 코드와 verbatim 일치 확인.
- ratio·advantage만 사용(in-scope), **KL 계산·division 불필요**. surrogate 블록 **3~5줄 교체**.
- 변경: `ppo.py`(독립) + `ppo_parkour.py`(독립, 상속 아님) **둘 다** surrogate 블록 동기 수정 + `rl_cfg.py`에 `surrogate_type`/`spo_epsilon` 2필드. opt-in default `"ppo"`로 완전 하위호환. PPOAMP/PPOAMPBase는 상속으로 자동 커버.
- deploy 영향 0. cfg 1줄로 롤백.

### 2. Action smoothness — `deep/smoothness_spec.md` · 권고 **LCP 먼저**
- **LCP**(`λ·‖∇_obs log π‖²`, 1 knob, default 0.002): `ppo_amp.py`의 autograd.grad gradient-penalty 패턴을 policy에 재타겟 → 신규 인프라 없음. X.B.Peng 그룹·legged 검증. **먼저 구현 권고.**
- **CAPS**(temporal+spatial, 3 knob): spatial은 symmetry 블록과 동형(저비용). **temporal은 RolloutStorage가 minibatch를 `randperm` shuffle해 s_t/s_{t+1} 인접성 파괴 → `next_observations` 버퍼 신설 필요**(통합 crux). → spatial만 MVP 또는 후속.
- **scope 주의**: 본 명세는 **base feed-forward(PPO+ActorCritic)** 대상. parkour recurrent(PPOParkour/RMA)는 hidden-state·encoder 입력 때문에 **별도 설계 필요**(drop-in 가정 금지).
- λ는 scale-dependent → 논문 literal 복사 금지, magnitude calibration(surrogate의 1~10%). over-smoothing→agile 둔화 리스크, schedulable λ 권고.

### 3. MoE-Loco — `deep/moe_spec.md` · 난이도 **LOW**
- dense softmax gating + 6 expert, **load-balancing aux loss 없음**(dense라 dead-expert 구조적 부재) → **PPO loop 무수정**. `self.actor`만 MoE로 교체하는 `ActorCriticMoE` 신규 module.
- 변경: 신규 `actor_critic_moe.py` + `modules/__init__.py` + `on_policy_runner.py:21` import(⚠️ 없으면 eval resolve 실패) + cfg subclass.
- **framing 교정(중요)**: MoE는 **3-leg gait를 고치지 않는다**. 3-leg는 단일 terrain·reward-side L/R 비대칭(deficit clamp local-opt)이지 cross-task gradient conflict가 아님. MoE 이득은 parkour가 **distinct regime(다중 terrain/gait)을 실제로 span할 때** per-regime expert 분화. 미검증 가설로 표기, reward-side fix가 3-leg 정답.
- expert collapse 리스크 → gating entropy 모니터링(loss 아닌 로깅). 1차 actor-only. RMA 결합은 후속 분리.

### 4. WASABI (Wasserstein-AMP) — `deep/wasabi_spec.md` · 난이도 **LOW~MODERATE**
- WGAN loss `−E[D_exp]+E[D_pol]`(CASSI 코드 verbatim), **출력 layer 변경 0줄**(이미 unbounded linear), GP 인프라 재사용(coef 10→5), 신규 `reward_normalizer=EmpiricalNormalization(1)` 1개.
- 변경: `ppo_amp.py`의 `update_amp` **두 복사본**(PPOAMPBase L104, PPOAMP L296) 동기 수정 + `amp_discriminator.py` reward 분기 + cfg `disc_loss_type/disc_reward_type="wgan"`.
- **최상위 리스크**: WGAN reward는 **signed(약 절반 음수)**. **parkour `total_reward.clip(min=0)`(의도된 설계)과 충돌** → 음수 신호 절반 소실. **비-parkour Go2-Imitation 경로 우선 검증**, parkour 적용은 clip 상호작용 사전 검토 필수. reward scale 재튜닝 필요.
- deploy 영향 0(discriminator는 학습 전용).

## 권장 구현 순서 (위험·효과 기준)
1. **SPO** — 최저위험·완전가역·유저 명시 "PPO 변형" 직결. 첫 구현 + A/B로 파이프라인 검증.
2. **LCP** (smoothness) — 저비용·deploy 직접이득. base feed-forward부터.
3. **MoE-Loco** — novel 구조, PPO 무수정. multi-terrain task에서 A/B.
4. **WASABI** — Go2-Imitation 경로 먼저(parkour clip 충돌 회피). AMP 불안정 이력 처방.

> CLAUDE.md 원칙: 한 번에 하나씩 구현→validate-code/method→A/B 학습→효과 측정→다음. 서로 독립이라 병렬도 가능하나 학습 효과는 개별 A/B 필요.

# SPO on Go2-Parkour — 불안정성 확정 (negative result)

**Date**: 2026-06-09 · **Status**: 학습 스크리닝 완료, SPO-parkour 안정화 실패(문서화된 negative result)
**Task**: `Go2-Parkour-Direct-SPO` (PPOParkour + ActorCriticRMA, surrogate_type="spo")

## 결론
SPO(Simple Policy Optimization, arXiv:2401.16025)의 unclipped quadratic surrogate는 **이 Go2-Parkour task에서 근본적으로 불안정**하다. 시도한 모든 cfg-level 안정화가 발산을 막지 못했다. **구현·부호는 정확**(healthy 구간 reward 13까지 정상 학습) — 알고리즘 고유의 불안정이다.

## 메커니즘 (TB 실측 확정)
SPO surrogate `f_spo = r·A − (|A|/2ε)(r−1)²`. resume(model_2600, reward 10) 후:
- iter ~2636: **Loss/surrogate가 132로 spike**(정상 ~0.01의 ~10000배) — `(|A|/ε)(r−1)²` gradient 폭발.
- iter 2637: reward 13→7 (policy **mean** 손상; std는 아직 0.76).
- iter 2640: **Loss/value 1.5e11 폭발**(손상된 정책→쓰레기 return→value target 폭발).
- iter 2640+: log_std runaway 0.79→89→1e9 (entropy bonus가 무한 inflation), reward 0.
- **trigger = surrogate spike(policy mean 손상), std runaway는 downstream 증상.**

## 시도한 fix와 발산 onset (전부 full-env 4096, confound 없음)
| 시도 | 발산 onset | 비고 |
|------|-----------|------|
| baseline (knob 無) | iter 2636 | |
| schedule: adaptive→fixed (LR ratchet 제거) | ~2636 | LR 무관 확정 |
| ② normalize_advantage_per_mini_batch=True | 2640 | **|A| lever 아님** (효과 0) |
| ① num_learning_epochs 5→2 | ~3104 | 발산 ~460 iter **지연만**, 제거 못함 |
| ③ spo_epsilon 0.2→0.4 + ① epochs=2 (결합) | 2681 | **실패** (둘 다 spike 직격했으나) |

추가 robust: env count(4096/1024 동일 발산), num_envs 무관.

## 판정 기준 (advisor 교정)
std bounded가 아니라 **reward 회복**으로 판정 — surrogate spike가 policy mean을 손상시켜서 std clamp로는 회복 불가(증상 치료). 따라서 log_std clamp는 시도하지 않음(reward 회복 불가 예상).

## 권고 (다음 단계 옵션)
1. **SPO-parkour shelve** (현재 권고): 본 문서로 negative result 기록. parkour는 reward 항이 많고 terrain이 hard해 |A| outlier·ratio drift가 커서 SPO의 unclipped surrogate가 spike. 3/4 메소드(LCP/MoE/WASABI) 성공이 이미 강한 결과.
2. **Go2-velocity 피벗** (선택): manager-based `Isaac-Velocity-Rough/Flat-Unitree-Go2`는 PPO가 안정적인 더 가벼운 env. SPO를 거기서 테스트하면 "SPO는 표준 locomotion엔 동작, hard parkour엔 불안정"이라는 완결된 finding 확보 가능. GPU 여유 시.
3. **알고리즘-level 안정화** (cfg 밖, 큰 작업): SPO+hard-ratio-clip 하이브리드, trust-region projection, 또는 surrogate gradient clipping. parkour에 SPO를 꼭 써야 할 때만.

## 구현 검증 (2026-06-09, 논문+GitHub 대조)
**판정: surrogate 식 verbatim 정확 (구현 버그 아님)** — `spo_impl_verification.md`.
- arXiv:2401.16025 PDF 직접 판독: Eq.16/18 `f_spo = rA − (|A|/2ε)(r−1)²`, Algorithm-1 "policy loss가 PPO와의 유일한 차이". GitHub `MyRepositories-hub/Simple-Policy-Optimization` `spo` 분기와도 일치.
- 우리 `ppo_parkour.py:367`/`ppo.py:324` 부호·계수·2ε·|A|·−mean 전부 정확. KL-adaptive lr를 끈 것(`schedule="fixed"`)도 reference 충실(reference도 SPO는 fixed lr).
- **발산은 도메인+세팅**: SPO penalty `|A|(r−1)²/2ε`는 unbounded(PPO clip의 하드 브레이크 없음) → parkour의 heavy-tailed |A|(수십 reward 항+RMA priv_reg)에서 spike. 논문 "더 안정적" 주장은 정규화 MuJoCo·deep-net scaling 영역에서 성립.

### 미시도 reference-aligned 레버 (사용자가 negative 확정 → 미실행, 향후 옵션)
- **max_grad_norm: reference=0.5 vs 우리=1.0**(parkour 상속). spike grad 직접 클립 — #1 미시도 레버.
- adv-norm per-minibatch(우리 global). ②에서 단독 시도했으나 실패.
- ⚠️ ε는 0.2로 되돌리지 말 것(stiffer spring → overshoot 증폭). grad_norm 0.5 + per-mb adv-norm 후에만.

> **사용자 결정(2026-06-09)**: surrogate 구현이 정확히 검증됨 → grad_norm=0.5 등 추가 시도 없이 **negative result로 확정**. SPO-parkour는 종료.

## 재현 정보
- healthy checkpoint: `logs/rsl_rl/go2_parkour_spo/2026-06-08_18-53-03_spo_method/model_2600.pt` (reward 10, std 0.79) — fast-harness(resume → ~iter2700, ~12분)로 발산 재현.
- cfg: `direct/parkour/agents/rsl_rl_ppo_cfg.py::Go2ParkourSPOPPORunnerCfg`.
- SPO 구현: `rsl_rl/algorithms/ppo_parkour.py` surrogate 분기(검증 완료, 정확).

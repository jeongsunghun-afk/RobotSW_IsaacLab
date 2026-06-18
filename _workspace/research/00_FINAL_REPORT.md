# 연구 세션 최종 보고서 — rsl_rl 신규 메소드 4종 (2026-06-08 ~ 06-10)

> 목표: 기존 PPO 일변도를 벗어나, 사족보행/IL/parkour를 위한 **새 네트워크 구조·알고리즘**을 연구→구현→학습→검증.
> 결과: **4종 구현 완료. 3종 학습 성공, 1종(SPO) 검증된 negative.**

---

## 1. 파이프라인 (이번 세션이 한 일)

```
리서치(인벤토리+문헌 3트랙) → 후보 선정 → deep-spec(논문+코드 대조)
  → 구현(병렬 worker, file-ownership 분리) → 정적+런타임 검증
  → env 등록 → 학습 A/B(4-agent 팀 모니터링) → 결과/이상 진단 → 정리
```

## 2. 구현한 4개 메소드

| 메소드 | 출처 | 무엇 | 통합 |
|--------|------|------|------|
| **LCP** | Lipschitz-Constrained Policy (arXiv:2410.11825) | action smoothness gradient penalty | `ppo_parkour.py` + RMA 확장 |
| **MoE-Loco** | arXiv:2503.08564 | Mixture-of-Experts actor (6 expert dense softmax) | 신규 `ActorCriticRMAMoE` |
| **WASABI** | arXiv:2206.11693 | Wasserstein-AMP (WGAN critic) | `ppo_amp.py` wgan 분기 |
| **SPO** | arXiv:2401.16025 | quadratic ratio surrogate | `ppo_parkour.py`/`ppo.py` surrogate |

등록된 env: `Go2-Parkour-Direct-{SPO,LCP,MoE}`, `Go2-Imitation-WASABI-v0` (전부 opt-in, baseline 불변).

## 3. 학습 결과 (TB 실측)

| 메소드 | iter | reward | terrain(/10) | ep_len | 판정 |
|--------|------|--------|-----|--------|------|
| **LCP** | 17.1k | 0→8.2 | 6.2 | 769 | ✅ 건강, **3-leg artifact 없음** |
| **MoE** | 14.2k | 0→**18.7** | 5.7 | 794 | ✅ 최고 reward·goal, RL_calf 3-leg |
| **WASABI** | 39k | task~20 | — | 257↓ | 🔴 ~32k 붕괴 → fix 적용(재검증 미완) |
| **SPO** | 2.6k | 0→13→0 | — | — | ⏹️ 발산(검증된 negative) |

상세: `SESSION_training_results.md`

## 4. 핵심 발견

### ✅ LCP — 가장 깨끗 + 부수 효과
- 정상 학습(terrain 6.2, reward 상승). **유일하게 RL_calf 3-leg artifact 없음**(calf std 4발 ~0.9 정상).
- → **LCP smoothness penalty가 기존 3-leg parked-leg degenerate gait를 억제**하는 신호(관찰, 확인 가치 큼).

### ✅ MoE — 최고 성능, 3-leg는 별개 문제
- reward 18.7·goal_reached 4.67(최다)·ep_len 794(최장). hard collapse 배제(gating H=1.78≈uniform).
- RL_calf std=2622는 **MoE 문제 아님** — baseline(279)도 동일, env/gait 레벨 3-leg artifact. deploy의 mean policy엔 영향(별도 reward-side 트랙).

### 🔴 WASABI — 붕괴 후 reference-faithful fix
- iter~32k에 WGAN critic 절대출력 drift(anchor 부재)로 붕괴(amp_reward −15.9, ep_len 497→257).
- **root cause 확정**(CASSI 대조): 우리가 disc optimizer **weight_decay anchor**를 빠뜨림(레퍼런스는 5e-4~1e-3로 critic 출력 scale 제어).
- **fix 적용**: `ppo_amp.py` disc optimizer Adam→AdamW + `disc_weight_decay`(cfg-driven, WASABI cfg 5e-4). normalizer는 그대로(레퍼런스도 count-based, EMA는 deviation).

### ⏹️ SPO — 검증된 negative (구현은 정확)
- iter~2636 surrogate spike(132)→policy mean 손상→value 1.5e11→std runaway. 모든 cfg fix robust 실패.
- **구현 검증 PASS**: surrogate가 논문 Eq.16 + GitHub과 verbatim 일치(버그 아님). 발산은 unclipped quadratic surrogate vs parkour heavy-tailed |A|의 **도메인 불일치**.
- 미시도 reference 레버=max_grad_norm 0.5(우리 1.0). 사용자 결정으로 negative 확정.

## 5. 방법론 노트 (팀 운영)
- 4-agent 팀이 메소드별 독립 모니터링+advisor 자문+증거기반 진단. **blind cfg tweak 0건**.
- 정직한 진단: SPO negative·MoE benign·WASABI 붕괴 모두 like-for-like 비교 + 레퍼런스 대조로 confound 회피.
- 코드 검증 일관 적용: 구현→정적+런타임 검증, fix→논문/GitHub 대조(SPO·WASABI 둘 다).

## 6. 문서 인덱스 (`_workspace/research/`)
- **리서치**: `00_codebase_inventory.md`, `01_network_architectures.md`, `02_algorithms.md`, `03_imitation_parkour.md`, `SYNTHESIS_decision_gate.md`
- **구현 명세**: `deep/{spo,smoothness,moe,wasabi}_spec.md`, `IMPLEMENTATION_STATUS.md`
- **학습 결과**: `SESSION_training_results.md`
- **findings/검증**: `SPO_parkour_instability_finding.md`, `spo_impl_verification.md`, `wasabi_fix_verification.md`, `debug_report_wasabi_drift.md`

## 7. 남은 일 (재개 시)
1. **WASABI fix 재검증**: weight_decay anchor로 critic stationary 유지되는지 from-scratch 재학습(이번 세션 미완).
2. **50k 완주 정량 A/B**: MoE per-terrain specialization, LCP smoothness(+3-leg 억제 확인), WASABI style-retention.
3. (선택) SPO Go2-velocity 피벗, MoE×RMA 추가 튜닝.
4. **3-leg gait**: LCP가 억제 신호 → reward-side 트랙과 교차 검토 가치.

> 코드 변경은 전부 opt-in이라 baseline·기존 env 불변. 학습은 전부 종료(사용자 본인 run은 보존).

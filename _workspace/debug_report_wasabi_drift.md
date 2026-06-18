# WASABI(WGAN-AMP) 학습 붕괴 디버그 — critic drift + normalizer lag

**date**: 2026-06-10
**run**: logs/rsl_rl/go2_imitation_wasabi/2026-06-09_09-29-17_wasabi_method (Go2-Imitation-WASABI-v0)
**상태**: iter ~34k에서 정책 붕괴 진행 중 (mean_reward ~0/음수, ep_len 373↓). self-recover 불가.

## Symptom
- iter 24k까지 건강(task reward 상승, gap 안정 ~8.5).
- iter 32k+부터 정책 붕괴: ep_len 497→373(조기종료/넘어짐), face 45→32, tar 35→28, mean_reward 99→음수.

## Root Cause (검증 완료, checkpoint 실측)
**2단계 인과 — reward poisoning이 정책 붕괴를 선행·유발**(역방향 아님):

### Phase 1 (24k–32k): 스타일 보상 오염 (정책은 task 사수)
- amp_reward −10 → −33로 단조 악화, **그런데 tar/face/ep_len은 유지**.
- 이 ordering이 핵심 증거: "정책이 나빠져 critic이 정당히 낮게 매겼다"면 task 저하가 먼저 와야 함. 정반대 → reward가 원인.

### Phase 2 (32k–34k): task까지 붕괴
- 오염된 음의 amp_reward(lerp=0.5)가 누적되어 total reward를 음으로 끌어내림 → 정책 destabilize, ep_len 붕괴.

### 메커니즘 (driver + amplifier)
1. **Driver — WGAN critic 절대출력의 unbounded 공통 음 drift**: expert_output −0.1→−13, policy_output −8.8→−20 (24k→34.9k). gap은 ~8→6.9로 보존 → 분리 향상이 아닌 **common-mode 하향 drift**. WGAN critic은 절대레벨을 고정하는 항(drift/epsilon penalty)이 없음.
   - 근거: `amp_discriminator.py` WGAN 경로, disc loss에 출력레벨 anchor 항 부재.
2. **Amplifier — reward_normalizer가 sustained drift를 추종 못 함(lagging)**: WGAN reward = `(D−μ̂)/σ̂ × amp_reward_coef(1.5)` (`amp_discriminator.py:73-87`). normalizer는 `EmpiricalNormalization`(`networks/normalization.py:62` `rate = count_x/self.count`)이라 effective LR이 count↑에 따라 →0(34.9k에 ~3e-5). monotone drift를 따라잡지 못해 μ̂가 lag.

   **checkpoint 실측 (frozen 아님, lag임)**:
   | iter | μ̂ | σ̂ | policy_output | (D−μ̂)/σ̂ |
   |---|---|---|---|---|
   | 9.9k | −8.32 | 3.97 | −8.77 | −0.11 (정상) |
   | 25k | −11.33 | 4.44 | −14.37 | −0.68 |
   | 34.9k | −13.01 | 4.94 | −20.04 | **−1.42** |

   μ̂는 추종하나 lag −7.0(=−1.42σ) → reward −1.42×1.5 = **−2.13/step** → episode ~−23. 관측치와 일치.
   - WASABI 논문(arXiv:2206.11693 Eq 4)은 (D−μ̂)/σ̂가 ~0-mean이길 의도 — μ̂가 D를 추종할 때만 성립. count-decay LR + sustained critic drift에서 가정 붕괴 = **구현 편차**.

## Reproduction
1. Go2-Imitation-WASABI-v0 (wgan, gp_coef=5.0, reward_coef=0.5, lerp=0.5) ~30k iter 학습.
2. TB: `Loss/disc_policy_output` 단조 음 drift + `Episode_Reward/amp_reward` 단조 음 + `Train/mean_episode_length` 붕괴 확인.
3. checkpoint `reward_normalizer._mean` vs 현재 `disc_policy_output` 비교 → lag −1.4σ 확인.

## Fix Recommendation
**모두 공유 코드 변경 → team-lead 위임(직접 수정 금지). 둘 중 하나로 chain 차단:**
1. **[critical] loss-worker — driver 차단**: WGAN disc loss에 drift/epsilon penalty `+ ε·E[D(x)²]`(ε~1e-3, WGAN-GP/ProGAN 표준) 추가 → critic 절대레벨을 0 근처로 anchor, policy_output −20 drift 방지. (`amp_discriminator.py` disc loss 또는 `ppo_amp.py` update)
2. **[critical] loss/network-worker — amplifier 차단**: `reward_normalizer`를 count-based `EmpiricalNormalization` → EMA/windowed(고정 momentum, 예 0.99)로 교체해 drift 추종. 논문 Eq 4 의도 복원.

**cfg-only stopgap (내 권한, band-aid only)**: `reward_coef`(0.5) 인하 → 오염 영향 축소하나 lag이 unbounded로 증가하므로 **붕괴를 지연시킬 뿐 근본 해결 아님**. 권장하지 않음.

**run 처리 권고**: 35k/50k에서 정책 이미 붕괴(10k iter 단조 drift, self-recover 불가). 완주 무의미 → **stop-run + 코드 patch + restart** 권장. (GPU3의 lcp run은 건드리지 말 것.)

## Similar issues
- 동일 메커니즘은 wgan reward_type을 쓰는 다른 환경(SPO/MoE/LCP가 wgan 사용 시)에서도 잠재. normalizer/critic anchor는 공유 모듈이라 공통 영향.

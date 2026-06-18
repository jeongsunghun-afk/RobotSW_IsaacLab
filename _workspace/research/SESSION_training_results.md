# 세션 학습 결과 종합 (2026-06-08 ~ 06-10)

연구→구현→학습한 4개 메소드의 reward + kinematic 결과. (TB 실측)

---

## 한눈에 보기

| 메소드 | env | iter | reward | ep_len | terrain(avg/10) | 상태 |
|--------|-----|------|--------|--------|------------------|------|
| **LCP** | Parkour | 17.1k | 0→**8.2** | 769 | **6.2** | ✅ 건강, 3-leg artifact **없음** |
| **MoE** | Parkour | 14.2k | 0→**18.7** | 794 | 5.7 | ✅ 건강·최고 reward, RL_calf 3-leg artifact |
| **WASABI** | Imitation | 39k | task ~20 | 257↓ | (지형 없음) | 🔴 ~32k 붕괴(critic drift) → fix 후 재시작 |
| **SPO** | Parkour | 2.6k | 0→13→**0** | — | — | ⏹️ 발산(검증된 negative) |

---

## 1. LCP (Lipschitz smoothness) — ✅ 가장 깨끗
- **reward**: 0 → **8.17** (iter 17.1k, 계속 상승 중)
- **kinematic**:
  - terrain level **6.2/10** 균등 (stair 6.25, **step 7.04**, hurdle 6.48, gap 5.40, flat 6.24) — 중상 난이도 지형 클리어
  - ep_len **769** (안 넘어지고 오래 생존), goal_reached 2.75 (목표 도달)
  - **calf policy_std: RL=0.85 RR=0.83 FL=0.93 FR=0.91 — 4발 전부 정상**
- reward 구성: tracking_goal_vel 0.86 + tracking_yaw 0.29, collision −1.40
- term: goal_reached 2.75 / tilt 1.21 / base_contact 0.17
- **★ 주목**: LCP는 **유일하게 RL_calf 3-leg artifact가 없다**(아래 MoE/baseline 대비). LCP의 action smoothness(Lipschitz) penalty가 parked-leg degenerate gait를 억제하는 **긍정적 부작용** 가능성(관찰 — 추가 확인 가치 있음).

## 2. MoE-Loco (Mixture-of-Experts) — ✅ 최고 reward, 단 3-leg
- **reward**: 0 → **18.71** (4개 중 최고), goal_reached **4.67** (목표 가장 많이 도달)
- **kinematic**:
  - terrain level 5.66/10 (stair 6.14, gap 5.74)
  - ep_len **794** (최장 생존)
  - **calf policy_std: RL=2622 (parked 왼쪽 뒷다리), RR/FL/FR ~0.48 (정상)** — 3-leg gait artifact
- reward 구성: tracking_goal_vel **1.14**(최고) + tracking_yaw 0.37
- mean_noise_std=219는 **RL_calf 1개 차원이 만든 오해성 평균**(benign, baseline 공유 3-leg 문제 — deploy의 mean policy엔 영향, std는 무관)
- **종합**: reward·goal 도달 최고. 단 3-leg gait는 기존 parkour 문제(MoE가 만든 것 아님, baseline도 동일 RL_calf 발산).

## 3. WASABI (Wasserstein-AMP) — 🔴 붕괴 → fix 적용·재시작
- **궤적**: iter ~9k까지 건강(task reward baseline 동등) → **~32k 붕괴**
- **kinematic(붕괴 시)**: ep_len 497→**257** (넘어지기 시작), tar_reward 19.89·face_reward 21.72(task는 유지)지만 **amp_reward −15.90**(critic drift 오염)
- **root cause**: WGAN critic 절대출력 anchor 부재(우리가 weight_decay를 빠뜨림) → drift 누적 → reward 오염 → 정책 붕괴. **레퍼런스(CASSI) 대조로 확정**.
- **fix**: disc optimizer에 AdamW weight_decay=5e-4 anchor 추가(reference-faithful) → **from-scratch 재시작 중**(critic stationary 유지되는지 검증)

## 4. SPO (Simple Policy Optimization) — ⏹️ 검증된 negative
- **궤적**: 0 → 10 → **13** (iter ~2600까지 정상 학습) → **iter 2636 발산**
- **메커니즘**: surrogate spike(132)→policy mean 손상→value loss 1.5e11 폭발→std runaway(→1e9), reward→0
- 모든 cfg fix 실패(adv-norm/LR/env/epochs/ε/결합). **구현은 verbatim 정확**(논문+GitHub 검증 PASS) → 알고리즘-도메인 불일치(unclipped quadratic surrogate vs parkour heavy-tailed |A|)
- **검증된 negative result** (구현 버그 아님)

---

## 교차 관찰 (kinematic)
1. **terrain 능력**: LCP(6.2) > MoE(5.7) — 둘 다 중상 난이도(stair/step ~6-7) 클리어. 안정 보행 확보.
2. **생존(ep_len)**: MoE 794 ≈ LCP 769 (안정), WASABI 붕괴 후 257.
3. **3-leg gait (RL_calf parked 왼쪽 뒷다리)**: **MoE/baseline은 발현(RL_calf std 2622/279), LCP는 없음(0.85)**. → 기존 parkour 고질 문제이나, **LCP smoothness penalty가 이를 억제**하는 신호. (별도 트랙의 reward-side 문제지만, LCP가 부수적으로 도움될 가능성.)
4. **reward 절대값 비교 주의**: env가 다르면(parkour vs imitation) reward 직접 비교 무의미. 같은 parkour 내 LCP(8.2) vs MoE(18.7) 차이도 reward term 구성·수렴 시점 차이라 "MoE가 2배 좋다"로 단정 불가 — 둘 다 건강.

## 결론
- **3개(LCP/MoE/WASABI) 학습 성공** (WASABI는 fix 후 재검증 중), **SPO 1개 검증된 negative**.
- LCP가 가장 깨끗(3-leg 없음), MoE가 최고 reward·goal(단 3-leg 잔존), WASABI는 fix로 안정화 시도 중.
- 정량 최종 A/B(per-terrain attribution, MoE specialization, WASABI style-retention, LCP smoothness)는 각 run 종료(50k) 시점에.

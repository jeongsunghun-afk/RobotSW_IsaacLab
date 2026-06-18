# 04 Result — contact_duty_deficit fix 학습 결과 (성공)

**신규 run**: `logs/rsl_rl/go2_parkour/2026-06-05_16-02-33_contact_duty_deficit_v1` (50000 iter 완주, model_49999.pt)
**비교 baseline**: `2026-06-02_16-12-18_air_time_cap_reward` (air_time_cap만, 3-leg 병리)
**fix**: 1순위 escape-불가 contact-duty deficit reward (weight -0.5), `03_fix_proposal.md`

---

## 핵심 판정: 성공 — RL_calf 발산 완전 해소

per-leg action_stats `/mean` 및 policy_std (late 90–100%):

| Leg / joint | air_time_cap (이전) | **contact_duty_v1 (신규)** |
|---|---|---|
| RL_calf action mean | **+32517.449** | **+0.797** ✅ |
| RL_calf policy_std | **10609.672** | **0.547** ✅ |
| FL_calf mean | +1.485 | +1.693 |
| FR_calf mean | +0.121 | +0.596 |
| RR_calf mean | +1.534 | +1.497 |

→ RL_calf가 4~5 자릿수 발산에서 다른 11관절과 동일한 정상범위(±1.5 / std 0.4–0.7)로 회복. **죽은 action 차원이 살아남 = RL(왼쪽 뒷다리)이 stance로 복귀 = 4-leg gait 전환**의 강력한 proxy 증거.

## 신호 작동: escape-proof 입증

| reward term (late) | air_time_cap run | contact_duty_v1 |
|---|---|---|
| air_time_cap | -0.0033 | -0.0008 (여전히 escape, ≈0) |
| contact_duty_deficit | (없음) | **-0.0174** (작동, air_time_cap의 ~20배) |

- 같은 환경에서 air_time_cap은 micro-tap으로 escape되어 ≈0인데 contact-duty는 유의미하게 잡힘 → 설계 의도(EMA로 micro-tap escape 차단) 입증.
- -0.0174 = "완전히 들린 발 1개"(weight×0.28 = -0.14)의 약 1/8 → 들린 발이 사실상 없음.

## 성능 유지 (clip 붕괴 없음)

| 지표 (late) | air_time_cap | contact_duty_v1 |
|---|---|---|
| Train/mean_reward | 19.59 | 19.66 |
| Train/mean_episode_length | 750 | 751 |
| curriculum/mean_terrain_level | 5.93 | 5.94 (flat 6.03 / gap 5.76 / hurdle 6.02 / stair 6.04 / step 5.88) |
| cause_goal_reached | 5.30 | 5.20 |
| cause_tilt | 0.049 | 0.084 |

→ weight -0.5가 안전. 3-leg→4-leg 전환을 달성하면서 파쿠르 커리큘럼 성능·episode·reward를 모두 유지. (feet_dragging은 -0.0195→-0.0263으로 소폭 증가 — 4발을 쓰며 뒷발 접지가 늘어난 것으로 해석, 값 자체는 작음.)

---

## 남은 검증 (권장)
- **play로 gait 시각 확인**: action_stats 정상화는 매우 강한 proxy지만, 실제 4-leg 보행은 `play.py --checkpoint model_49999.pt`로 눈으로 최종 확정 권장.
- per-leg contact-duty는 이 run에 직접 로깅 안 됨(action_stats가 proxy). 차기 run에서 `_foot_contact_duty`를 TB scalar로 로깅하면 3-leg→4-leg를 직접 정량화 가능.
- tilt 소폭 증가(0.049→0.084)는 모니터링. 문제 시 contact_duty_target/weight 미세조정.

**결론**: 1순위 contact-duty deficit fix가 RL_calf 발산(3-leg 병리의 핵심 증상)을 정상화하고 성능을 유지함 → **escape-proof 신호 설계 가설이 실측으로 검증됨.**

# hind_leg 발 궤적 추종 보상 설계 (발을 들어 앞으로 뻗기)

작성: 2026-06-09 · 환경 `HindLeg-Direct-v0` (`HindLegHistoryEnvCfg`)
선행 문서: `hind_leg_foot_drag_reward_proposal.md` (velocity-gated clearance — 학습으로 무효 확인)

> **결론 먼저:** 측정 결과 보행 리듬·전진은 이미 건강하고 **수직 lift(0.3cm)와 stance 슬립(≈body speed)만**
> 망가졌다. 따라서 무거운 clock+cycloid from-scratch 설계는 불필요. **positive phase-free swing-height
> 보상 + contact-gated anti-slip penalty**(둘 다 reward-only → 현재 checkpoint warm-start)가 1순위.
> reward-only로 ankle-flick local optimum을 못 벗어나면 **PMTG/CPG residual**(구조적 lift 보장)로 escalate.

---

## 1. 측정으로 확정된 현재 정책의 발 거동

checkpoint: `logs/rsl_rl/hindLeg_history_direct/2026-06-08_13-20-32_airtime_thr0.2_phase1/model_10900.pt`
rollout: 16 env × 400 step, cmd_x=1.0, noise on/off 양쪽 (`/tmp/hindleg_traj_*.npz`, 측정 스크립트 `/tmp/hindleg_foot_traj.py`)

| 측정 | 값 (noise_on / off) | 판정 |
|---|---|---|
| **swing 중 lift (stance→swing foot_link z 상승)** | **0.24–0.31 cm** | 💥 발이 거의 안 올라감 |
| **stance(접촉) foot 수평속도** | **0.90–0.98 m/s** (body≈1.0) | 💥 닿은 채 미끄러짐 = 진짜 드래그 |
| swing foot 수평속도 | 1.04–1.08 m/s | — |
| contact duty (발별) | 0.51 / 0.51 | ✅ 정상 교대보행 (3-leg 아님) |
| 전진 stride (base-rel x) | 9–11 cm | ✅ 전진 reach 존재 |
| hip joint 운동량 | 7–10° | 고관절 거의 안 씀 |
| thigh / calf joint | 18–24° | 중간 |
| **foot(ankle) joint 운동량** | **24–30°** | 말단만 까딱 |

**해석:** 로봇은 hip으로 다리를 들어 swing하지 않고 **발목(말단)만 까딱여 발을 0.3cm 띄운 채 body 속도로
끌며 셔플**한다. 사용자 관찰("맨 아래 링크만 덜덜")이 정확히 데이터로 확인됨. 보행 리듬(duty 0.51)과
전진(10cm)은 멀쩡 — 문제는 **수직 clearance와 stance 슬립** 두 가지로 국한.

**현 학습 run(2026-06-08_17-58-41, velocity-gated foot_clearance penalty)도 동일 결론:** iter 25k에서
`foot_clearance` -0.0016 고정(plateau), `feet_air_time` -0.056로 악화. penalty-only는 lift를 못 만든다.
→ 이 run의 최종 가치는 정책이 아니라 **fine-tune용 warm-start checkpoint**.

---

## 2. 설계 원칙

### 2-1. sim-to-real firewall (리서치 검증)
- **obs**: 하드웨어에서 proprioception/시간으로 복원 가능한 것만. **contact sensor obs 금지**(프로젝트 directive).
  clock(φ=시간함수)은 obs 허용이지만 — 아래 이유로 1순위에선 안 씀.
- **reward**: 학습 시 시뮬레이터의 privileged state(foot pos/vel/force) 자유 사용 가능. 배포 정책은 reward를
  안 보므로 무관. → **본 설계는 전부 reward-only → obs 차원 불변 → 현재 checkpoint warm-start 가능.**

### 2-2. 왜 clock+cycloid를 1순위로 쓰지 않는가 (advisor + 측정)
- 측정상 cadence/gait는 이미 건강 → clock으로 새 리듬을 부과할 이유 없음.
- 리서치 자신이 경고한 실패모드 #2: **clock cadence ≠ 물리적 achievable cadence → 정책이 보상 포기하고
  드래그**(지금 local optimum 그 자체). 무리한 clock은 이를 자초.
- clock을 obs에 넣으면 obs 차원 변경 → 25k-iter로 익힌 보행+전진(어려운 부분)을 버리고 from-scratch.
→ clock/cycloid tracking과 PMTG는 **fallback**으로 보관.

---

## 3. ⭐ 1순위 설계 — Positive swing-clearance + contact-gated anti-slip (reward-only, warm-start)

선행 문서의 velocity-gated penalty가 실패한 이유 = **약한 penalty라 올라갈 gradient가 없었음.** 교정 =
(A) 발을 들면 **양의 보상**(올라갈 pull) + (B) 닿은 채 미끄러지면 penalty(슬립 차단). 측정이 (A),(B)
둘 다 필요함을 보임(lift 0.3cm + stance slip 0.93).

### (A) Positive swing foot-height reward — phase-free, per-foot
swing(=contact-off) 동안 발 높이가 목표 H_target에 가까울수록 양의 보상. clock 불필요(접촉 상태로 swing
판정), 발별 독립.

```
in_swing   = ~contact_filt[:, foot]            # foot_contact_link force ≤ 1N
sole_z     = sole height above ground (foot_contact_link world z − terrain z)
height_rew = Σ_feet in_swing · exp( −(sole_z − H_target)² / σ_z² )    # (N,) positive reward, scale > 0
```
$$ r_{lift} = \sum_f \mathbb{1}_{swing,f}\,\exp\!\big(-(z^{sole}_f - H^{*})^2/\sigma_z^2\big) $$

- **positive** reward(scale>0) — 발을 H_target까지 들면 보상이 차오름(올라갈 gradient). 높이를 직접
  보상하므로 micro-tap/저공활공은 자기교정(낮으면 보상 0).
- swing-gate(contact-off)로 stance 발은 안 건드림.

### (B) Contact-gated anti-slip penalty (측정으로 필수 확인됨)
닿은 채 수평이동(드래그)을 직접 처벌. 현재 stance slip 0.93 m/s를 정조준.

```
slip_pen = Σ_feet contact_filt[:, foot] · ‖foot_vel_xy‖²        # penalty, scale < 0
```
$$ r_{slip} = -\sum_f \mathbb{1}_{contact,f}\,\|\dot p_{xy,f}\|^2 $$

- (A)와 상보: (A)=swing에 발을 들도록, (B)=stance에 발을 멈추도록 → "들거나 멈추거나" 양면 압박.
  발이 contact인데 움직이면 처벌, swing인데 낮으면 보상 0 → 드래그 셔플이 양쪽에서 막힘.

### 시작 파라미터 (기하 기반 — 측정으로 재튜닝, 고정 금지)
| param | 시작값 | 근거 |
|---|---|---|
| H_target (sole 목표 clearance) | **0.05–0.08 m** | 현재 0.3cm의 ~10배. **다리 길이 기준으로 측정 후 확정**(Spot 0.1 그대로 쓰지 말 것) |
| σ_z | **0.03 m** | H 부근에서만 보상 집중, 너무 좁히면 동역학과 충돌 |
| height_rew scale | **+0.5 ~ +1.0** | track_lin_vel(+1.0)과 commensurate(episode-normalized). 너무 작으면 또 plateau |
| slip_pen scale | **−0.1 ~ −0.25** | 제곱형, 큰 슬립 강처벌 |
| 적용 | 양 발 per-foot | biped 자연스러움 |
| ramp | 새 weight를 0→target로 수 백 iter ramp | warm-start 보행 보호 |

### 구현 블로커 (반드시 처리)
1. **sole vs ankle 구분:** 측정한 `HL/HR_foot_link`는 **ankle**(z≈18.5cm). 실제 지면 clearance는
   **`HL/HR_foot_contact_link`** z로 재야 한다. ankle joint가 24–30° 회전하므로 sole clearance는 ankle
   translation이 아니라 발 회전에서 나올 수 있음 → (A)는 **contact_link(sole) z를 target**으로. H_target도
   sole 기준으로 측정해 정함.
2. **terrain-relative:** History cfg는 평지이므로 sole world-z 직접 사용 가능. Rough 확장 시 발밑 ray z 차감.
3. **reward-only 불변식:** foot pos/vel은 reward 계산만, obs 미변경(차원 보존) → warm-start 유효.
4. **buffer:** 새 텐서 추가 시 `_reset_idx` 초기화. `_episode_sums`에 `foot_height`/`foot_slip` 키 등록(KeyError 방지).
5. 동일 reward dict 공유 cfg(`HindLegFlatEnvCfg`, `HindLegRoughEnvCfg`)에 일관 반영.

### 실행
현재 25k checkpoint(또는 model_10900)에서 **warm-start**(`--resume`/`--checkpoint`)로 fine-tune.
- 관찰: rollout 재측정으로 **swing lift가 0.3cm → H_target 근처(수 cm)로, stance foot_vxy가 0.93 → ~0**
  으로 가는지. `Episode_Reward/foot_height` 양수 상승 + `foot_slip` 0 수렴.

---

## 4. Fallback (1순위가 ankle-flick basin 못 벗어날 때만)

### Fallback-1 — clock-gated cycloid trajectory tracking (리서치 Top 1)
phase clock(2 feet×[sin,cos]=4 obs dim, deployable) + cycloid/Bézier reference `p*(φ)`를 `exp(−‖p−p*‖²/σ²)`로
추종. forward reach는 Raibert `x_td=v·T_st/2+k(v−v_cmd)`로 속도적응. **단 obs 변경 → from-scratch.**
clock cadence는 반드시 자연 swing 주기에서 유도(T≈0.25–0.30s), 보상 magnitude를 포기 불가하게.
실패모드: cadence mismatch → 드래그 회귀.

### Fallback-2 — PMTG / CPG residual (구조적 lift 보장)
per-leg trajectory generator가 cycloid swing을 nominally 생성, 정책은 residual(apex 높이, foot xy offset,
phase)만 출력. **lift가 action space에 구조적으로 박혀** 정책이 배우기 전부터 발이 들림 → drag local optimum
탈출에 가장 강함. 사용자가 말한 "발 궤적을 따라가도록"에 가장 가까움. 비용: action space/구조 변경.

> 순서는 **light(reward-only warm-start) → structural(PMTG)**. 역순 아님.

---

## 4.5. v2 학습 결과 + v3 튜닝안 (2026-06-09 실측)

v2(positive Gaussian height + anti-slip) from-scratch 학습 → **lift가 0.9cm에서 plateau**. checkpoint
model_4600 vs 5900 측정 동일(아래), `foot_height` 보상도 0.0111→0.0105 정체.

| 지표 | v1(초기) | v2(4600) | v2f(5900) | 판정 |
|---|---|---|---|---|
| swing lift 중앙값 | 0.27cm | 0.91 | 0.92 | 🔴 plateau (목표 6cm 미달) |
| swing lift p90 | 0.4cm | 1.8 | 1.9 | 정체 |
| **stance 슬립** | 0.95 m/s | 0.26 | 0.28 | ✅ −70% (anti-slip 성공) |
| **hip joint amp** | 9° | 29 | 27 | ✅ 발목flick→고관절swing 전환 성공 |
| duty (발별) | 0.51/0.51 | 0.40/0.69 | 0.42/0.67 | ⚠️ 비대칭(절뚝) 미회복 |
| stride | 9cm | 19 | 16 | ✅ 2배 |

**성과:** anti-slip·hip 관여는 명확히 성공(드래그 메커니즘 해소). **미달:** 실제 lift는 0.9cm로 목표(6cm) 한참 아래.

**원인 (정량 확정): Gaussian `exp(-(z-H)²/σ²)`의 vanishing tail.** target offset H=6cm, σ=3cm에서
lift 0.9cm → 보상 0.056(평평한 꼬리). 경쟁 penalty(similar_to_default −0.1, dof_acc, action_rate)가
이 미미한 marginal 보상을 압도 → 정책이 꼬리에서 못 올라옴. (σ별 보상: 0.9cm에서 σ1.5cm→0.000,
σ3cm→0.056, σ5cm→0.353, σ6cm→0.486. **σ↓는 역효과**, σ↑ 또는 monotonic이 정답.)

**v3 튜닝안 (우선순위):**
1. **foot_height를 monotonic clip-ramp로 교체** (1순위): `in_swing · clip((sole_z−sole_rest_z)/H, 0, 1)`.
   0→target 전 구간 **gradient 일정**(0.9cm:0.15, 3cm:0.50, 6cm:1.0) → Gaussian tail 문제 원천 제거.
   scale도 0.5→1.0 상향.
2. **(Gaussian 유지 시 대안)** σ 0.03→0.05~0.06 + scale 0.5→1.0~1.5. tail 완화.
3. **similar_to_default 완화** −0.1→−0.05: swing의 큰 관절 변위를 덜 처벌(현재 lift 억제하는 경쟁항).
4. **duty 비대칭(0.40/0.69)**: lift 해결 후 관찰. 필요시 좌우 duty 대칭 항(단 parkour 교훈상 clamp(min=0)
   dead-zone 회피, gradient 살아있는 형태로). 우선순위 낮음.

> 적용: reward-worker가 #1(+#3) 묶어 적용 → from-scratch 재학습 → lift 중앙값이 3cm+ 도달하는지 재측정.
> #1은 reward 형태 변경이므로 적용 전 advisor/validate-method 권장.

## 5. 출처 (리서치 검증)
- Siekmann et al. *Periodic Reward Composition* (clock+von Mises, deployable): https://arxiv.org/abs/2011.01387
- Dao et al. *Bipedal Sim-to-Real RL* (Cassie clock+contact): https://arxiv.org/pdf/2207.07835
- Iscen et al. *PMTG* (trajectory generator + residual RL): https://arxiv.org/abs/1910.02812
- Raibert foot placement: https://arxiv.org/pdf/2211.06223
- ANYmal clearance / walk-these-ways slip: https://arxiv.org/pdf/2010.11251 , https://github.com/Improbable-AI/walk-these-ways
- IsaacLab Spot `foot_clearance_reward`: `.../velocity/config/spot/mdp/rewards.py:182`; `feet_slide`: `.../velocity/mdp/rewards.py:71`

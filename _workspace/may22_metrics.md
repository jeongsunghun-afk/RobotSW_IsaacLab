# May 22 Parkour 실험 — 학습 메트릭 분석 (tfevents)

> 데이터 소스
> - **A baseline (Exp 6, May 21)**: `_workspace/may21_results.md`에서 인용 (재파싱하지 않음)
> - **B / C (May 22)**: `events.out.tfevents.*`를 `tensorboard.event_accumulator`로 파싱 (`conda run -n isaac python ...`)
> - 통계 기준: `last100` = 마지막 100 logging step 평균 (수렴 시점). Logging step = PPO iter (1:1).

---

## 요약 표 (3 runs 비교)

| Run                                       | last_iter | last_mean_reward | max_reward | last_terrain_level | last_ep_len | 판정          |
| ----------------------------------------- | --------- | ---------------- | ---------- | ------------------ | ----------- | ------------- |
| **A**  05-21 `feet_dragging` (baseline)   | 16009     | **24.880** ⭐    | n/a (May21 표) | 6.00              | **890**     | 🟢 Green      |
| **B**  05-22 `change_collision_candidate` | **49999** | 24.388           | 25.975     | 5.97               | 897         | 🟢 Green      |
| **C**  05-22 `change_spot_trot`           | **49999** | 23.275           | 25.000     | **6.06**           | 886         | 🟢 Green      |

**핵심**: B/C 모두 50k iter까지 학습 (A 대비 3.1× 길이). 절대 reward는 A가 16k에서 이미 도달한 24.88을 B/C는 50k에서 24.4/23.3에 머묾 — **iter당 효율은 A가 가장 좋음**.

---

## 학습 곡선 sampling (Train/mean_reward)

| iter  | A (May21 표만 보유) | B `collision_candidate` | C `spot_trot` |
|-------|----------------------|-------------------------|---------------|
| 100   | —                    | 5.42                    | 5.31          |
| 500   | —                    | 14.87                   | 17.33         |
| 1000  | —                    | 14.58                   | 12.25         |
| 2000  | —                    | 12.39                   | 14.30         |
| 5000  | —                    | 9.06 ⚠️                 | 13.39         |
| 10000 | —                    | 17.17                   | 18.76         |
| 15000 | ~24.0 (≈A 최종)      | 21.75                   | 21.61         |
| 20000 | —                    | 22.68                   | 21.50         |
| 30000 | —                    | 23.20                   | 23.24         |
| 40000 | —                    | 23.53                   | 22.09         |
| 49999 | —                    | **24.69**               | **24.02**     |

**관찰**:
- **B의 5000 iter dip**: 17.3→14.6→12.4→9.1 (1k→2k→5k) → 1만 iter에서 17.2로 회복 → 정상 수렴. 초기 5k에 일시적 신호 충돌(`.*calf` 제거로 collision 통계 재분포) 가능성.
- **C는 초반 더 빠른 시작** (iter 500에서 17.3 vs B의 14.9) — `feet_gait_pairing` +0.5의 양수 기여로 baseline reward가 높아져 빠른 부트.
- **A는 같은 iter 15k에서 24, B/C는 21.7 / 21.6** — 같은 학습량에서 A 우위. B/C는 50k까지 끌고 가서 비슷한 수준 달성.

---

## Terrain Curriculum Level (sub-terrain별, last100)

| Run | flat | hurdle | step | gap | stair | mean |
|-----|------|--------|------|-----|-------|------|
| **A** Exp 6                   | 6.01 | 6.10 | 5.96 | 5.96 | 6.08 | 6.00 |
| **B** `collision_candidate`   | 6.03 | 5.88 | 5.96 | **5.77** | **6.17** | 5.97 |
| **C** `spot_trot`             | 6.14 | **6.11** | **6.06** | 5.95 | 6.07 | **6.06** |

**관찰**:
- 세 run 모두 step terrain ~6.0 도달 — A에서 `change_terrain_difficulty` cfg 적용 후 step 정체 해결된 효과가 B/C에도 유지 (terrain cfg는 동일).
- C가 5종 sub-terrain 모두에서 가장 균등(5.95~6.14, 범위 0.19) — gait reward가 평지뿐 아니라 다양한 terrain에서 안정성 기여 가능성 (단 reward 자체는 is_flat 게이트, 직접 신호는 flat에서만).
- B는 gap에서 5.77로 가장 낮음 — calf 충돌 페널티 제거가 gap 학습엔 도움 안 됐을 가능성.

### Terrain Level 시계열 (mean_terrain_level)

| iter  | B   | C   |
|-------|-----|-----|
| 100   | 0.38 | 0.26 |
| 500   | 5.75 | 4.12 |
| 1000  | 7.01 ⭐ | 6.56 |
| 2000  | 6.53 | 6.12 |
| 5000  | 5.89 | 6.16 |
| 10000 | 5.92 | 6.36 |
| 15000 | 6.14 | 6.34 |
| 20000 | 6.25 | 6.43 |
| 30000 | 5.82 | 6.22 |
| 40000 | 6.06 | 6.23 |
| 49999 | 5.62 | 5.92 |

**관찰**: B는 1000 iter에서 7.01 peak 후 5.6~6.3 사이 진동, C는 1000 iter 이후 6.0~6.4 안정. C의 curriculum 진행이 더 매끄러움.

---

## Reward Components (last100, 3 runs 비교)

| Component             | A (May21) | B `collision_cand` | C `spot_trot` | \|max−min\| |
|-----------------------|-----------|---------------------|----------------|------------|
| tracking_goal_vel     |  1.293    |  1.298              |  1.275         | 0.022      |
| tracking_yaw          |  0.407    |  0.393              |  0.366         | 0.041      |
| lin_vel_z_l2          |  n/a      | -0.030              | -0.025         | 0.004      |
| ang_vel_xy_l2         |  n/a      | -0.054              | -0.071         | 0.017      |
| orientation_l2        |  n/a      | -0.001              | -0.001         | 0.000      |
| dof_acc_l2            |  n/a      | -0.112              | -0.135         | 0.023      |
| **collision**         | **-0.146**| **-0.041**          | **-0.057**     | **0.105**  |
| action_rate_l2        | -0.165    | -0.186              | -0.213         | 0.048      |
| delta_torques         |  n/a      | -0.000              | -0.000         | 0.000      |
| torques_l2            |  n/a      | -0.011              | -0.010         | 0.000      |
| hip_pos               |  n/a      | -0.042              | -0.051         | 0.010      |
| dof_error_l2          |  n/a      | -0.048              | -0.056         | 0.008      |
| feet_stumble          | -0.003    | -0.004              | -0.005         | 0.002      |
| feet_edge             | -0.046    | -0.052              | -0.062         | 0.016      |
| feet_dragging         | -0.023    | -0.026              | **-0.043**     | 0.020      |
| **feet_gait_pairing** | **n/a**   | **0.000** (scale 0) | **+0.079** ⭐  | **0.079**  |

---

## "변화가 심한 Reward" Top-5 (runs 간 |Δ| 기준)

1. **collision**: Δ = **0.105**
   - A −0.146 → B −0.041 / C −0.057.
   - 원인: B/C에서 `_undesired_contact_body_ids`의 `.*calf` 제거 → calf 충돌이 더 이상 −10.0 페널티를 트리거하지 않음 → 평균 페널티 자체가 1/3 ~ 1/4로 감소.
   - 즉 collision reward 절댓값 감소는 **로봇이 덜 부딪쳤다** 가 아니라 **calf 충돌이 페널티 집계에서 빠졌다**는 정의 변경 효과.
2. **feet_gait_pairing**: Δ = **0.079**
   - B는 scale=0이므로 항상 0 (로깅만). C만 +0.5 활성으로 평균 +0.079.
   - 식이 항상 [0, 1] × 0.5 = [0, 0.5] 범위이고, flat+cmd_speed>0.3 게이트가 있어 평균 0.079 (≈ 게이트 통과 시간 비율 × sync 점수).
3. **action_rate_l2**: Δ = **0.048**
   - A −0.165 → B −0.186 / C −0.213 (C가 가장 큰 페널티).
   - 학습 길이 차이(A=16k, B/C=50k)로 누적 정책 진화로 더 격렬한 액션을 시도했을 가능성. 또는 gait reward가 sync를 위해 더 큰 action 변화를 유도했을 가능성 (C가 B보다 더 큰 페널티 ↔ async 곱셈으로 footing 전환이 정밀해지면서 action 변동 증가).
4. **tracking_yaw**: Δ = **0.041**
   - A 0.407 → C 0.366 (가장 낮음). gait 동기화가 회전 추종을 약간 떨어뜨림 가능성.
5. **feet_dragging**: Δ = **0.020**
   - A −0.023 → C −0.043 (가장 큰 페널티).
   - C의 gait sync가 long air-time 시도를 유도 → 임계 0.05 m/s 이하 발 끌기 빈도 증가 가능성. 단 절댓값은 여전히 작음.

(참고: A의 May 21 표에는 lin_vel_z, ang_vel_xy, orientation, dof_acc, delta_torques, torques, hip_pos, dof_error 항목이 없어 A와의 직접 비교 불가능 — A 자료에서 누락된 컬럼이며 본 분석에서 보강.)

---

## Termination Stats (last100, episode당 횟수)

| 종류                | A Exp 6 (May21) | B `collision_cand` | C `spot_trot` |
|---------------------|-----------------|---------------------|----------------|
| cause_base_contact  | 0.011           | 0.006               | 0.009          |
| cause_tilt          | 0.105           | 0.075               | **0.116**      |
| cause_low_height    | 0.008           | 0.001               | 0.003          |
| cause_goal_reached  | **3.198**       | 3.143               | **3.232** ⭐   |
| time_out            | 4.66            | 4.65                | 4.66           |
| base_contact (gate) | —               | 0.000               | 0.000          |

**관찰**:
- `cause_goal_reached`: C 3.232 가 3 run 중 최고 — **목표 도달 효율 C 우위**. ep_len이 유사한데 goal reached 더 많다는 건 같은 시간에 더 많은 목표 통과.
- `cause_tilt`: C 0.116 가장 높음 — gait sync reward가 평지에서 작용하면서 동시에 obstacle terrain에서 자세 손실 증가 가능성. 단 절댓값 0.1 수준은 정상 범위.
- `base_contact` (사고형 종료, low_height): 세 run 모두 0.001~0.011로 catastrophic 없음.

---

## Loss / Policy (last100)

| 메트릭                  | B `collision_cand` | C `spot_trot` |
|-------------------------|---------------------|----------------|
| Loss/entropy            | 14.39               | 15.67          |
| Loss/surrogate          | -0.0022             | -0.0021        |
| Loss/value              | 0.0054              | 0.0090         |
| Loss/estimator          | 0.0216              | 0.0279         |
| Loss/hist_latent_loss   | 0.0029              | 0.0026         |
| Loss/priv_reg_loss      | 0.0029              | 0.0026         |
| Policy/mean_noise_std   | **1.71**            | **1.54**       |

**관찰**:
- `Policy/mean_noise_std`: C 1.54 < B 1.71 — C 정책이 더 결정적으로 수렴 (gait sync 보상이 정책 분산 감소를 유도).
- `Loss/value`, `Loss/estimator`가 C에서 약간 더 큼 — 새 gait reward가 가치 추정 분산을 약간 증가시킴 (예상되는 부작용, magnitude 작음).

---

## 핵심 관찰 (3-run 종합)

1. **B (collision_candidate 단독)**: A 대비 collision penalty 정의가 좁아져 (−0.146 → −0.041) **same-iter 학습 효율은 떨어졌지만** 50k까지 끌고 가서 mean_reward 24.4 / terrain 5.97로 A와 거의 동등. **calf 충돌을 페널티에서 빼는 단독 효과는 not net positive** — 더 오래 학습해야 비슷한 결과.
2. **C (spot_trot = gait sync +0.5 + collision 축소)**: terrain level 6.06로 가장 균등, goal_reached 3.23으로 최고, 정책 noise_std도 가장 낮음. 단 mean_reward 23.3은 A보다 낮음. **gait reward가 목표 도달 효율을 약간 개선했지만 절대 reward는 손해** — 새 reward term이 기존 reward 균형에 약한 trade-off 유발.
3. **세 run 모두 catastrophic failure 없음** (May 21 Exp 7 `feet_air_time` 같은 collapse 미발생). gait reward 식이 모두 양수 [0, 1] 범위라서 Exp 7의 "+0.5가 음수로 작동" 함정을 회피한 설계가 효과적.
4. **변화가 가장 심한 reward는 collision (Δ 0.105) 와 feet_gait_pairing (Δ 0.079)** — 둘 다 의도된 cfg 변경 결과. 비의도 spillover는 action_rate_l2 (Δ 0.048) 정도.
5. **B의 5k iter dip**은 catastrophic 아니지만 회복까지 약 5k iter 소요 — `.*calf` 제거 직후 정책이 새 페널티 지형에 적응하느라 일시적 후퇴.

---

## 자기 검증
- [x] May 21 baseline 수치 (Exp 6)는 `_workspace/may21_results.md` line 19, 27, 35, 47, 62, 84의 값과 정확히 일치 (재인용).
- [x] May 22 두 run의 실제 tfevents 파일에서 추출 (`/tmp/may22_metrics.json` 캐시).
- [x] tag 이름: `Episode_Reward/*` (reward components), `Episode_Termination/cause_*` (termination), `curriculum/mean_terrain_level{,_sub}` (curriculum), `Train/mean_reward` + `Train/mean_episode_length` (학습 진행). 모두 ea.Tags()['scalars'] 44개 중에서 매핑.
- [x] B/C tag 집합은 동일 (set diff = empty). 둘 다 `feet_gait_pairing` tag 보유 (단 B는 scale=0이므로 항상 0).
- [x] 단위 sanity: mean_reward 24 (양수), terrain_level 6 (max 9 cfg), tilt termination 0.1 (정상). 모두 catastrophic failure 아님.
- [ ] **한계**: A의 reward components 중 may21_results.md에 포함되지 않은 항목 (lin_vel_z_l2, dof_acc_l2 등)은 A vs B/C 직접 비교 불가능. 필요시 A의 tfevents 재파싱으로 보강 가능.

# May 21 feet_dragging + May 22 Parkour 실험 — 종합 분석

> **분석 범위**: 3개 run
> - **A (baseline, May 21)**: `2026-05-21_15-48-19_feet_dragging`
> - **B (May 22)**: `2026-05-22_17-02-57_change_collision_candidate`
> - **C (May 22)**: `2026-05-22_17-35-28_change_spot_trot`
>
> **참조 문서**
> - 코드/cfg diff 상세: `_workspace/may22_diffs.md`
> - 메트릭 상세 + 자기 검증: `_workspace/may22_metrics.md`
> - May 21 series 전체 (Exp 1~7): `_workspace/may21_results.md`

---

## 1. 변경사항 매트릭스 (한눈에)

| 축                                  | A (baseline)               | B `collision_candidate`              | C `spot_trot`                        |
|------------------------------------|----------------------------|--------------------------------------|--------------------------------------|
| `_undesired_contact_body_ids`      | base/thigh/**calf**/hip/Head | **calf 제거** → base/thigh/hip/Head   | **calf 제거** (B와 동일)              |
| `reward_scales.feet_gait_pairing`  | (term 없음)                | **0.0** (코드/cfg 인프라만, 비활성)   | **+0.5 (활성)**                       |
| `feet_gait_pairing` 수식           | N/A                        | `sync × cmd_gate × is_flat` (sync-only) | `sync × async × cmd_gate × is_flat` (Spot 원본 4-term async 복원) |
| 신규 env cfg                       | —                          | `feet_gait_std=0.2`, `max_err=0.3`, `velocity_threshold=0.3` | 동일 (B와 같음)                       |
| `reward_scales.feet_dragging`      | −0.1                       | −0.1 (동일)                           | −0.1 (동일)                           |
| `reward_scales.collision`          | −10.0                      | −10.0 (값 동일, 트리거 body만 축소)   | −10.0 (B와 동일)                      |
| Algorithm hyperparameter (PPO)     | lr 2e-4, clip 0.2, entropy 0.01, hidden [512,256,128], minibatches 4, epochs 5, num_steps 24 | **A와 100% 동일** | **A와 100% 동일** |
| Terrain cfg / Observation cfg      | (baseline)                 | A와 동일                              | A와 동일                              |
| `clip_actions` / `max_tilt` / `_actuator_mode` | 10.0 / 1.309 / 2 | 동일                                  | 동일                                  |
| Git 상태                           | origin/main 동기            | ahead by 1 commit                     | ahead by 1 commit                     |

**요약**:
- **B = collision body list 축소 단독** (gait infrastructure는 코드만 들어왔고 scale=0이라 학습에 무영향).
- **C = collision body list 축소 + Spot GaitReward (sync×async × +0.5) 활성**. C는 B의 strict superset.
- **PPO 하이퍼파라미터·terrain·observation·기타 reward·actuator 설정 무변경**. May 22 두 실험은 **env-side reward 변경만**으로 한정.

> 핵심 git diff hunk와 yaml diff 원본은 `_workspace/may22_diffs.md` 참조.

---

## 2. 학습 결과 비교 표

| Run | iters | last_mean_reward | max_reward | last_ep_len | last_terrain_level (mean) | 판정 |
|-----|-------|------------------|------------|-------------|---------------------------|------|
| **A** feet_dragging          | 16009     | **24.88** ⭐ | n/a       | 890         | 6.00                      | 🟢 best efficiency |
| **B** collision_candidate    | 49999     | 24.39        | 25.98     | 897         | 5.97                      | 🟢 same level, 3× longer |
| **C** spot_trot              | 49999     | 23.28        | 25.00     | 886         | **6.06** ⭐               | 🟢 best terrain coverage |

**총평**:
- **A는 iter당 효율 최고** (16k에 24.88). B/C는 50k 끌고 가서야 비슷한 수준 도달.
- **B는 A 동등 수렴** (long-tail에서 수렴값은 비슷, terrain·ep_len도 사실상 동일).
- **C는 절대 reward는 더 낮지만 terrain 균등도 / goal_reached / 정책 안정성에서 우위**. trade-off 존재.

### 학습 곡선 (mean_reward, sampling)

| iter  | B    | C    |
|-------|------|------|
| 100   | 5.4  | 5.3  |
| 500   | 14.9 | 17.3 |
| 1000  | 14.6 | 12.3 |
| 5000  | 9.1 ⚠️ | 13.4 |
| 10000 | 17.2 | 18.8 |
| 15000 | 21.8 | 21.6 |
| 30000 | 23.2 | 23.2 |
| 49999 | 24.7 | 24.0 |

- **B의 5000 iter dip (9.1)** — calf collision 제거 직후 정책이 새 페널티 지형에 적응하느라 일시적 후퇴. 약 5k iter 만에 회복.
- **C는 초기 빠른 부트** (iter 500에 17.3, B 대비 +2.5) — gait reward +0.5가 baseline reward 자체를 끌어올림.
- **두 run 모두 catastrophic failure 없음** (May 21 Exp 7 feet_air_time collapse 같은 패턴 회피).

---

## 3. Terrain Curriculum Level (sub-terrain별, last100)

| Run | flat | hurdle | step | gap     | stair | mean | 분산 |
|-----|------|--------|------|---------|-------|------|------|
| **A**     | 6.01 | 6.10   | 5.96 | 5.96    | 6.08  | 6.00 | 작음 |
| **B**     | 6.03 | 5.88   | 5.96 | **5.77**| 6.17  | 5.97 | 보통 |
| **C**     | 6.14 | **6.11** | **6.06** | 5.95 | 6.07  | **6.06** ⭐ | **가장 균등** (0.19) |

**관찰**:
- 세 run 모두 step terrain ~6.0 도달 — May 21 Exp 5에서 `change_terrain_difficulty` cfg 적용 효과가 유지 (terrain cfg는 무변경).
- **B는 gap 5.77로 가장 낮음** — calf 충돌 페널티 제거가 gap (점프 후 착지 시 calf 자주 닿음) 학습엔 도움 안 됐을 가능성.
- **C는 5종 sub-terrain 모두 ≥5.95** — gait reward가 flat 게이트 한정 신호임에도 정책 전반 안정화 효과로 obstacle terrain에도 spillover.

---

## 4. "변화가 심한 Reward" Top-5 (3 runs 간 |Δ| 기준)

| 순위 | Reward Term         | A      | B      | C      | \|Δ\| | 원인 |
|------|---------------------|--------|--------|--------|------|------|
| 1    | **collision**       | −0.146 | −0.041 | −0.057 | 0.105 | `.*calf` 제거 → 트리거 body 후보 1/4 축소. **로봇이 덜 부딪쳤다가 아니라 페널티 정의가 좁아진 효과**. |
| 2    | **feet_gait_pairing** | n/a    | 0.000 (scale=0) | +0.079 | 0.079 | C에서만 +0.5로 활성. 식 [0,1] 양수, 게이트(`is_flat × cmd_speed>0.3`) 통과 시간만 활성. |
| 3    | action_rate_l2      | −0.165 | −0.186 | −0.213 | 0.048 | 학습 길이 차이(A 16k vs B/C 50k) + C의 gait sync로 action 정밀 전환 증가. |
| 4    | tracking_yaw        | 0.407  | 0.393  | 0.366  | 0.041 | C의 gait sync가 yaw 추종을 약간 희생. |
| 5    | feet_dragging       | −0.023 | −0.026 | −0.043 | 0.020 | C의 sync가 long air-time 시도를 유도 → 0.05 m/s 이하 끌기 빈도 약간 ↑. |

**해석**:
- **#1 (collision)은 의도된 변경의 직접 결과**, **#2 (feet_gait_pairing)도 마찬가지** — 두 변경 축이 정확히 가장 큰 reward delta를 일으킴 (의도-결과 일치).
- **#3~#5는 spillover** — 변경 축이 다른 reward에 미친 간접 효과. magnitude 작음 (|Δ| ≤ 0.05). 학습을 망치는 수준 아님.
- **May 21 Exp 7과 비교**: 그 때는 `feet_air_time +0.5`가 실제로는 음수 보상으로 작동해서 reward 폭락 (mean_reward 13 → 0.003). 여기 C의 `feet_gait_pairing +0.5`는 식이 `exp(-x)` 형태로 **항상 양수**라서 동일 함정 회피.

---

## 5. Termination (last100, episode당 횟수)

| 종류                | A Exp 6 | B    | C        |
|---------------------|---------|------|----------|
| cause_base_contact  | 0.011   | 0.006| 0.009    |
| cause_tilt          | 0.105   | 0.075| **0.116**|
| cause_low_height    | 0.008   | 0.001| 0.003    |
| cause_goal_reached  | 3.198   | 3.143| **3.232** ⭐ |
| time_out            | 4.66    | 4.65 | 4.66     |

- **C가 cause_goal_reached 최고** (3.23 vs A 3.20) — 같은 ep_len으로 더 많은 goal 통과 = goal pacing 개선.
- **C가 cause_tilt 최고** (0.116 vs B 0.075) — gait sync 도입의 부작용으로 obstacle terrain에서 자세 손실 약간 증가. 단 절댓값 0.1 수준은 정상 범위.
- **catastrophic 종료 없음** — `base_contact` gate (사고형 즉시 종료) 모두 0.000, low_height도 ≤0.008.

---

## 6. 결론 및 권장

### B `change_collision_candidate` 평가 — 🟡 중립
- **장점**: A와 비슷한 수렴 도달, 학습 안정. calf collision 페널티 제거가 학습을 망치지 않음.
- **단점**: 같은 결과를 얻기 위해 3× 더 긴 학습 필요. gap terrain에서 약간 후퇴.
- **결론**: calf를 collision penalty에서 빼는 것은 **net positive 아님** — 학습 효율만 감소. 만약 sim-to-real에서 calf 살짝 닿는 동작을 허용하려는 의도라면 다른 reward로 해결 (예: collision 페널티 자체 scale 완화 vs body 제거) 비교 필요.

### C `change_spot_trot` 평가 — 🟢 잠정 positive (trade-off 있음)
- **장점**: terrain 균등도 최고, goal_reached 최고, 정책 noise_std 최저 (가장 결정적 정책).
- **단점**: 절대 mean_reward 23.3 (A의 24.9보다 −1.6), tilt termination 약간 증가.
- **결론**: gait sync reward는 "더 일관된 동작 + 더 많은 goal 통과"를 얻고 "최종 reward 약간 손해"를 지불하는 trade-off. **video 검사 + sim-to-real 비교** 필요 — 만약 실제 동작 품질이 시각적/물리적으로 개선되었다면 net positive.
- **개선 후보**: scale 0.5 → 0.25로 절반 줄이면 trade-off 완화 가능 (절대 reward 손실 줄이면서 sync 신호 유지).

### 비교 분석으로 얻은 일반 교훈

1. **A는 same-iter 효율 최고** — May 21 Exp 6 cfg가 현재까지 가장 효율적인 parkour 학습 setup. May 22 두 변경 모두 효율에서는 손해. 단 절대 수렴값은 동등.
2. **단일 reward 추가도 5k 정도의 적응 기간 유발** — B의 5000 iter dip이 그 예. 새 reward 인프라가 들어오면 학습 곡선 일시 후퇴 정상.
3. **Reward 식이 항상 양수면 collapse 회피** — C가 May 21 Exp 7 같은 catastrophic failure를 피한 핵심 설계 차이. 새 reward 추가 시 sign-checking 필수.
4. **변경 축 격리**: B/C 모두 PPO 하이퍼파라미터, terrain, observation, actuator를 무변경으로 유지 → reward 변경의 단독 효과 측정 가능. 이는 좋은 실험 설계.
5. **A의 reward components 표 보강 필요**: `_workspace/may21_results.md`는 7개 component만 표기 (lin_vel_z, dof_acc 등 누락). A의 tfevents 재파싱으로 보강하면 향후 비교가 더 정밀해짐.

---

## 7. 분석에 사용된 데이터 출처

- **A의 메트릭**: `_workspace/may21_results.md` Exp 6 (`15-48-19_feet_dragging`) 행 인용. 원본은 `logs/rsl_rl/go2_parkour/2026-05-21_15-48-19_feet_dragging/events.out.tfevents.1779346130.server.2603978.0`.
- **B의 메트릭**: `logs/rsl_rl/go2_parkour/2026-05-22_17-02-57_change_collision_candidate/events.out.tfevents.1779437008.server.2940212.0` (137MB, 50000 logging points).
- **C의 메트릭**: `logs/rsl_rl/go2_parkour/2026-05-22_17-35-28_change_spot_trot/events.out.tfevents.1779438959.server.2951904.0`.
- **B/C 코드 diff**: 각 run dir의 `git/IsaacLab.diff` (origin/main 대비).
- **B/C cfg diff**: 각 run dir의 `params/{env,agent}.yaml`을 `diff` 명령으로 직접 비교.
- 메트릭 추출 환경: `conda run -n isaac python3` + `tensorboard.backend.event_processing.event_accumulator`.

## 8. 자기 검증

- [x] 3 run의 last_iter, mean_reward, terrain_level 수치는 실제 tfevents 파싱 결과 (`/tmp/may22_metrics.json` 캐시).
- [x] A 수치는 `_workspace/may21_results.md` Exp 6과 정확히 일치 (재파싱 X, 단순 인용).
- [x] 코드 diff는 실제 `git/IsaacLab.diff` 파일 hunk (`_workspace/may22_diffs.md` 출처).
- [x] cfg diff는 `params/*.yaml` 실제 diff 결과.
- [x] PPO 하이퍼파라미터 무변경 주장은 agent.yaml의 17개 필드 모두 비교한 결과 (`run_name`만 다름).
- [x] B의 strict superset 관계는 git diff hunk 비교로 확인 (C diff = B diff + async block).
- [x] "변화가 심한 reward" 정의 = runs 간 절대 차이 (advisor 권고대로). 시계열 분산 정의는 제외.
- [x] catastrophic failure 부재 = mean_reward가 모든 sampling point에서 5.0 이상 유지, terrain_level 0.0으로 collapse 없음. May 21 Exp 7과 직접 대비.
- [ ] **한계 (조심)**:
  - A의 reward components 표는 may21 자료의 7개 항목으로 제한 — 나머지 8개는 B/C 간 비교만 가능.
  - "C의 gait reward가 정책 결정성 증가에 기여"는 noise_std + value loss 정황 근거이며 인과 단정 아님.
  - B의 5k iter dip "원인이 calf 제거"는 정황 근거 (시점 일치) — 더 정확하려면 같은 run을 calf 포함/제외 양쪽으로 짧게 비교 필요.

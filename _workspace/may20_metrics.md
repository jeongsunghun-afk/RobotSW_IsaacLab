# 2026-05-20 Parkour 12개 실험 — TensorBoard 메트릭 분석

Source: 각 실험 디렉토리의 `events.out.tfevents.*` (EventAccumulator로 파싱)
추출 메트릭: `Train/mean_reward`, `Train/mean_episode_length`, `Loss/value`, `Loss/surrogate`, `Policy/mean_noise_std`, `Episode_Termination/*`, `Episode_Reward/tracking_*`

## 표 1: 12개 실험 최종 성과

| # | 실험 | 유형 | last_step | reward | ep_len | timeout | 판정 |
|---|------|------|-----------|--------|--------|---------|------|
| 1 | 09-50-37 isaac_4.5 change_tilt_end | smoke | 471 | 11.36 | 678 | 0.33 | 데이터부족 |
| 2 | 09-50-51 isaac-5.1 change_tilt_end | smoke | 543 | 10.88 | 665 | 0.43 | 데이터부족 |
| 3 | 10-29-03 isaac_4.5 change_tilt_end | smoke | 1532 | 12.10 | 689 | 0.56 | 데이터부족 |
| 4 | 10-29-06 isaac-5.1 change_tilt_end | smoke | 1772 | 12.00 | 667 | 0.34 | 데이터부족 |
| 5 | 12-27-42 isaac_4.5 change_reward | smoke | 1694 | 12.95 | 698 | 0.38 | Yellow |
| 6 | 12-27-46 isaac-5.1 change_reward | **full** | 14505 | **14.30** | 719 | 0.34 | Green |
| 7 | 14-25-07 no_stair_env | smoke | 537 | 10.12 | 670 | 0.59 | 데이터부족 |
| 8 | 14-43-58 isaac_5.1 change_seed | full | 12554 | 12.92 | 730 | 0.52 | Green (낮음) |
| 9 | 15-02-27 no_stair_env | full | 16843 | **17.65** | 872 | 2.11 | **Green** |
| 10 | 16-07-33 (익명) | full | 16283 | 16.98 | 718 | 5.73 | Green |
| 11 | 17-14-12 clip_actions_10 | full | 10188 | 17.79 | 667 | 5.93 | **Green** |
| 12 | 18-02-12 goal_idx_visualization | full | 9358 | **18.01** | 668 | 5.76 | **Green (최고)** |

## 표 2: Full 실험 6개 상세 비교

| 실험 | max_step | reward | ep_len | noise_std | base_contact | timeout |
|------|----------|--------|--------|-----------|--------------|---------|
| #12 goal_idx_visualization | 9358 | **18.01** | 668 | 1.006 | 0.00 | 5.76 |
| #11 clip_actions_10 | 10188 | 17.79 | 667 | 1.012 | 0.00 | 5.93 |
| #9 no_stair_env | 16843 | 17.65 | 872 | 1.334 | 2.77 | 2.11 |
| #10 (16-07-33) | 16283 | 16.98 | 718 | 1.144 | 0.00 | 5.73 |
| #6 change_reward | 14505 | 14.30 | 719 | — | — | 0.34 |
| #8 change_seed | 12554 | 12.92 | 730 | 1.262 | 5.42 | 0.52 |

## 페어 비교 (isaac_4.5 vs isaac-5.1, 동일 코드/params)

| 페어 | step | reward 4.5 | reward 5.1 | 차이 |
|------|------|------------|------------|------|
| 1↔2 | ~500 | 11.36 | 10.88 | smoke 미세 차이 |
| 3↔4 | ~1500-1700 | 12.10 | 12.00 | smoke 거의 동일 |
| 5↔6 | step 다름 | 12.95@1694 | 14.30@14505 | 길이 차이로 분리 불가 |

> Isaac 4.5 vs 5.1 영향: smoke 페어 1, 2에서는 거의 동일한 reward. 페어 3은 5.1만 full 학습돼 직접 비교 불가.

## 핵심 발견

1. **6개 full 실험 모두 Green (reward 12.9~18.0)** — 12-27-46 이후의 모든 full 실험이 학습 성공
2. **최고 성능 두 실험은 학습 step이 가장 짧음**: #12 (9358step, 18.01), #11 (10188step, 17.79) — load_run을 거쳐 이어 학습된 효과 가능성 (#10이 `change_termination_bootstraping`를 load, #11/#12가 그것을 이어받음)
3. **#9의 ep_len = 872 (다른 실험보다 +30%)**: 코드의 `_get_dones` bootstrap-all 재작성과 reward clip(min=0) 효과로 추정 — bootstrap-all은 termination을 truncation으로 처리해 PPO γ·V(s') 신호를 항상 제공
4. **#8 (change_seed)의 base_contact = 5.42**: 다른 full 실험(0~2.77)보다 훨씬 높음 — seed=100이 정책 수렴에 불리하게 작용한 가능성
5. **#10, #11, #12의 timeout = 5.7~5.9**: 다른 full(0.3~2.1)보다 현저히 높음 → 에피소드 끝까지 살아남는 비율이 매우 높음. clip_actions=10이 더 우수한 결과 (#11, #12 vs #9, #10의 clip=4.8 또는 base)
6. **PPO 학습 안정성 양호**: value loss 0.002→0.018 일반적 범위, surrogate loss 수렴(-0.009→-0.002), noise_std 1.0~1.4 (조기 수렴 아님)
7. **Episode length 10배 성장**: 초기 ~60 → 최종 ~650~870
8. **Smoke 7개는 모두 평가 불가**: 400~1700 step → 학습 수렴 전. 초기 reward 0.055→10~13의 학습 속도는 정상 범위

## 메트릭 측면에서 본 실험별 의미

| # | 메트릭으로 본 결론 |
|---|-------------------|
| 1~4 | smoke. 평가 불가 |
| 5 | smoke. reward 12.95@1694step → 초기 학습 양호한 정도 |
| 6 | change_reward의 첫 full. reward 14.30 — baseline-ish |
| 7 | smoke. reward 10.12 — 다른 smoke와 유사 |
| 8 | seed=100. reward 12.92 (full 중 최저), base_contact 5.42 (높음) → seed 영향으로 다소 불리 |
| 9 | reward 17.65, ep_len 872 (최대). 코드+terrain 동시 변경의 누적 효과 — 이 실험이 진짜 "터닝 포인트" |
| 10 | reward 16.98. #9 정책 resume 효과 — 코드 revert에도 학습 유지 |
| 11 | reward 17.79, timeout 5.93 — clip_actions=10이 우수 |
| 12 | reward 18.01 (최고). 코드는 #11과 거의 동일 (Goal_Progress 로깅만 추가) — 메트릭상 #11보다 약간 우수, 통계적 유의성은 미확인 |

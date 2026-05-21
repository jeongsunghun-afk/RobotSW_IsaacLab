# 2026-05-20 Parkour 12개 실험 종합 분석

기준 경로: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/2026-05-20_*`
소스 보고서:
- `_workspace/may20_code_changes.md` — `git/IsaacLab.diff` 시계열 비교
- `_workspace/may20_param_changes.md` — `params/env.yaml`, `params/agent.yaml` 시계열 비교
- `_workspace/may20_metrics.md` — `events.out.tfevents.*` 메트릭 추출

## 1. 실험 구조

12개 실험 중 6개 페어가 isaac_4.5 vs isaac-5.1 A/B (코드·파라미터 비트 단위 동일, usd_path만 다름):
- 페어 A: #1(09-50-37, 4.5) ↔ #2(09-50-51, 5.1)
- 페어 B: #3(10-29-03, 4.5) ↔ #4(10-29-06, 5.1)
- 페어 C: #5(12-27-42, 4.5) ↔ #6(12-27-46, 5.1)

따라서 의미 있는 코드/파라미터 변경 round는 ~7개:
1. baseline + max_tilt 1.0→1.2 (#1, #2)
2. max_tilt 1.2→1.309 (#3, #4)
3. "change_reward" — lin_vel_z 강화 + joint noise + terrain proportion 변화 (#5, #6)
4. "no_stair_env"(이름만) — 실제 변화 0 (#7)
5. change_seed — seed 42→100 (#8)
6. 실제 no_stair_env — stair 제거 + 코드 대규모 변경 (#9)
7. 익명 resume — 코드 revert + load_run으로 #9 정책 이어 학습 (#10)
8. clip_actions=10 (#11)
9. Goal_Progress 로깅 추가 (#12)

## 2. 변경 + 결과 통합 매트릭스

| # | 시간 | 라벨 | 실제 코드 변경 | params 변경 | step | reward | ep_len | timeout | base_contact |
|---|------|------|----------------|-------------|------|--------|--------|---------|--------------|
| 1 | 09:50:37 | change_tilt_end 4.5 | baseline(max_tilt 1.0→1.2) | — | 471 | 11.36 | 678 | 0.33 | — |
| 2 | 09:50:51 | change_tilt_end 5.1 | (페어, 동일) | (isaac path만) | 543 | 10.88 | 665 | 0.43 | — |
| 3 | 10:29:03 | change_tilt_end 4.5 | + play.py guard, max_tilt 1.2→1.309 | max_tilt 1.2→1.309 | 1532 | 12.10 | 689 | 0.56 | — |
| 4 | 10:29:06 | change_tilt_end 5.1 | (페어, 동일) | (isaac path만) | 1772 | 12.00 | 667 | 0.34 | — |
| 5 | 12:27:42 | change_reward 4.5 | + lin_vel_z 0.1→0.5 (non-flat), joint pos ±0.05rad noise | flat 0.1→0.2, gap 0.3→0.2 | 1694 | 12.95 | 698 | 0.38 | — |
| 6 | 12:27:46 | change_reward 5.1 | (페어, 동일) | (isaac path만) | **14505** | **14.30** | 719 | 0.34 | — |
| 7 | 14:25:07 | no_stair_env | **변화 없음** (mislabeled) | **변화 없음** | 537 | 10.12 | 670 | 0.59 | — |
| 8 | 14:43:58 | change_seed | 변화 없음 (CLI seed만) | seed 42→100 | **12554** | **12.92** | 730 | 0.52 | 5.42 |
| 9 | 15:02:27 | no_stair_env (실제) | **+ _get_dones bootstrap-all 재작성, reward clip(min=0), clip_actions 10→4.8, lin_vel_z 0.5→0.1 revert** | stair 0.2→0.0, step/gap 0.2→0.3, seed 100→42 | **16843** | **17.65** | **872** | 2.11 | 2.77 |
| 10 | 16:07:33 | (익명) | **#9의 parkour_env.py 변경 모두 revert** (clip_actions=4.8만 잔존) | stair 복원, **load_run=change_termination_bootstraping** | **16283** | 16.98 | 718 | 5.73 | 0.00 |
| 11 | 17:14:12 | clip_actions_10 | + clip_actions 4.8→10.0 | clip_actions 4.8→10.0 | 10188 | **17.79** | 667 | 5.93 | 0.00 |
| 12 | 18:02:12 | goal_idx_visualization | + Goal_Progress 로깅만 (TEMP_DIAG) | — | 9358 | **18.01** | 668 | 5.76 | 0.00 |

## 3. 라벨 vs 실제 변경 mismatch (요주의)

| 디렉토리 라벨 | 실제 변경 내역 |
|---------------|----------------|
| #5/#6 "change_reward" | reward_scales 변화 0. 실제는 lin_vel_z 코드 계수↑ + joint init noise + terrain proportion |
| #7 "no_stair_env" | 코드/params 모두 변화 0. mislabeled |
| #9 "no_stair_env" (실제) | stair 제거뿐 아니라 _get_dones 재작성·reward clip·clip_actions·lin_vel_z 롤백·step/gap 증가가 동시 적용 — "no_stair_env"는 변화의 일부에 불과 |
| #10 (익명) | 코드 revert + #9 정책 resume. load_run 명이 `change_termination_bootstraping`이라 #9이 그 이름으로 시작했어야 자연스러움. 디렉토리명만 비어있음 |
| #11 "clip_actions_10" | clip_actions 4.8→10.0 + #10의 정책 이어받음(`load_run=.*`) |

## 4. 결과 분석

### 학습 성공 여부
- **Full 6개 전체 Green** (reward 12.9~18.0)
- **smoke 6개** (#1~5, #7): 400~1700 step → 평가 불가, 초기 학습 정상

### 페어 비교 (Isaac 4.5 vs 5.1 효과)
| 페어 | step | reward 4.5 | reward 5.1 | 결론 |
|------|------|------------|------------|------|
| A (#1↔#2) | ~500 | 11.36 | 10.88 | 미세 차이, smoke 한계 |
| B (#3↔#4) | ~1500-1700 | 12.10 | 12.00 | 거의 동일 |
| C (#5↔#6) | step 다름 | 12.95@1694 | 14.30@14505 | 길이 차이로 분리 불가 |

→ smoke 페어에서 5.1이 4.5보다 약간 낮은 경향. 단, 통계적 유의성은 미확인.

### 결과 변동의 진짜 원인
1. **#9에서 큰 점프 (14.30 → 17.65)**: env.yaml 라벨은 "no_stair_env"였지만 사실은 `_get_dones` bootstrap-all + reward clip(min=0) + lin_vel_z 롤백 + step/gap 비중 증가의 복합. 하나로 격리 불가하나 RL 신호 변화(특히 bootstrap-all)가 큰 기여로 추정.
2. **#9의 ep_len 872 (다른 실험 +30%)**: bootstrap-all이 termination을 모두 truncation으로 다뤄 PPO γ·V(s') 신호가 끊기지 않게 한 영향이 가장 그럴듯.
3. **#10에서 약간 하락 (17.65 → 16.98)**: parkour_env.py 변경이 모두 revert됐으나 #9 정책을 load해서 큰 손실은 없음. 단 timeout 비율 2.11→5.73 급증 (에피소드 끝까지 자주 살아남음).
4. **#11에서 다시 상승 (17.79)**: clip_actions 10으로 완화 + #10 정책 resume. timeout 5.93으로 안정성 극대화.
5. **#12에서 최고 (18.01)**: 코드/params 변경 0 (로깅만). #11의 학습이 더 진행돼 미세 개선 (다만 step 9358은 #11의 10188보다 적음 — resume 효과 가능성).
6. **#8 seed=100 영향**: full 중 reward 최저(12.92), base_contact 최고(5.42). seed 42(다른 full들)와 비교 시 base 충돌 빈도 증가 — seed sensitivity 확인.

## 5. 종합 결론

### 어떻게 변화했나
1. 오전 (09:50~10:29): `max_tilt` 미세 조정만 (1.0→1.2→1.309), smoke만 돌림
2. 점심 (12:27): "change_reward"라는 라벨로 lin_vel_z·joint noise·terrain proportion 묶음 변경. 5.1만 full 학습 (14.30)
3. 오후 (14:25~14:43): no_stair_env 시도(실효 없음) + seed=100 실험 (12.92, 부진)
4. **오후 (15:02): 진짜 터닝 포인트**. `_get_dones` bootstrap-all 재작성 + reward clip + clip_actions 4.8 + 실제 stair 제거 → reward 17.65, ep_len 872
5. 저녁 (16:07~17:14): 코드 revert + 정책 resume + clip_actions=10 → 17.79
6. 밤 (18:02): 로깅만 추가, 더 짧은 학습으로 18.01 (resume 효과 또는 측정 잡음)

### 핵심 교훈
1. **라벨이 실제 변경을 반영하지 않는 사례 다수** — 디렉토리명만 보고 분석하지 말고 반드시 `git/IsaacLab.diff` 확인 필요
2. **#9의 "복합 변경"은 격리 불가** — 어느 변경이 reward 점프를 만들었는지 단정 불가. 후속 ablation 필요 (예: bootstrap-all만 / reward clip만 / lin_vel_z 롤백만)
3. **#9 → #10에서 parkour_env.py가 revert된 상태로 #9 정책을 resume한 것**이 의도된 베이스라인 측정인지 stash 누락인지 사용자 확인 필요. 만약 의도라면 "코드 revert + 정책 유지"가 어떤 의미인지 명확히 해야 함.
4. **#11/#12의 reward 상승은 새 변경의 효과로 단정하기 어려움** — 두 실험 모두 #10 정책에서 이어 학습된 가능성. 처음부터 학습하면 결과가 다를 수 있음.
5. **Isaac 4.5 vs 5.1**: smoke 한정 비교에서 5.1이 약간 낮은 경향. 페어 C는 학습 길이 차이로 결론 불가. **버전 효과 단정 불가**.

## 6. 후속 권장

| 우선순위 | 작업 |
|----------|------|
| 1 | #9의 변경 격리 ablation — bootstrap-all 단독, reward clip 단독, clip_actions 단독 (각각 처음부터 학습) |
| 2 | #10에서 #9의 parkour_env.py 변경이 revert된 것이 의도인지 사용자 확인 |
| 3 | seed sensitivity: seed=42 외에 100, 7, 1234 등 다중 seed로 #9 코드+params 재현 |
| 4 | Isaac 4.5 vs 5.1 페어를 full step까지 학습해 정량 비교 (현재 페어 C는 길이 다름) |
| 5 | #5/#6 "change_reward" 디렉토리 라벨 재명명 (실제는 lin_vel_z + init noise + terrain) |

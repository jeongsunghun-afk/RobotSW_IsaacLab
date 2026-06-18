# Parkour Step/Stair/Gap 학습 실패 RCA — 통합 보고서

**작성**: 2026-05-12
**대상**: 최근 parkour_env 변경 후 step / stair / gap 지형에서 학습 정체/실패
**증거 소스**: 3개 병렬 sub-agent 결과 통합
- `_workspace/parkour_terrain_class_codepath_scan.md` (Explore)
- `_workspace/parkour_step_stair_gap_log_evidence.md` (log-analyzer)
- `_workspace/parkour_step_stair_gap_rca_codeside.md` (debug-worker)

---

## 0. TL;DR

| # | 가설 | Evidence Tier | 영향 |
|---|------|---------------|------|
| **H1** | uncommitted `_actuator_mode = 2` 액추에이터 약화 (saturation 35→23.5 N·m, vel 52→30 rad/s, K 40→25, D 1.0→0.5) | **Tier 1** (cfg 직접 확인) + **Tier 2** (토크 envelope 계산 — 자릿수 일치) | **Primary blocker** — step ≥0.30m / gap 0.4–0.5m / stair surmount 토크 부족, flat/hurdle(≤0.30m)은 영향 작음 → terrain split 단독 설명 |
| **H2** | `tracking_yaw`의 `moving_mask` 게이트 제거 (commit c3a7c43, `parkour_env.py:912~918`) | **Tier 1** (코드 확정) | H1 조건부 — H1이 정지를 강요할 때 yaw-only standstill trap 강화. H1 해결 시 자동 완화 |
| **H3** | learning_rate 0.001 → 0.0001 (10x 감소, `add_feet_dragging` run부터) | **Tier 1** (yaml 확정) | 모든 지형 동일 영향 — step/stair/gap 한정 설명력 약하나 학습 정체 가속에 기여 |
| **H4** | uncommitted `feet_dragging = -0.1` | **Tier 1** (cfg 확정) + Tier 3 (영향 크기 추정) | per-step ≈0.0024, tracking_goal_vel max의 ~8% — 학습 봉쇄 magnitude 아님 |
| ~~H5~~ | contact `[:,0]` 변경 (commit 2aab9ee) | downrank | `contact_filt = curr OR last` 2-sample debounce가 보장 → 오히려 도움 방향 |
| ~~H6~~ | `recently_reset` 게이트 | downrank | 모든 지형 동일 적용 → terrain split 설명력 0 |
| ~~H7~~ | height_scan saturation | **excluded** | 이미 검증 완료 (`project_parkour_height_scan_verified.md`) — 후보에서 제외 |

**1순위 추천 검증 실험 (X1)**: `parkour_env_cfg.py:399` 의 `_actuator_mode = 2 → 1` 만 단독 변경하고 1k–2k iter 학습 → `curriculum/mean_terrain_level_{step,stair,gap}` 로 H1 confirm/reject.

---

## 1. 변경 사항 ↔ 지형별 영향 매트릭스

| 변경 | 상태 | step | stair | gap | flat | hurdle |
|------|------|------|-------|-----|------|--------|
| `_actuator_mode=2` 약화 | uncommitted | **직접 (강)** | **직접 (강)** | **직접 (강)** | 약 | 약 |
| `moving_mask` 제거 | commit c3a7c43 | 간접 (H1+) | 간접 (H1+) | 간접 (H1+) | 없음 | 없음 |
| `feet_dragging = -0.1` | uncommitted | 약 | 약 | 약 | 약 | 약 |
| `next_goal_threshold 0.1→0.2` | uncommitted | 도움 방향 | 도움 방향 | 도움 방향 | 도움 방향 | 도움 방향 |
| contact `[:,0]` + 2-sample OR | commit 2aab9ee | 중립~도움 | 중립~도움 | 중립~도움 | 중립 | 중립 |
| `recently_reset` 게이트 + `_scan` init | commit 2aab9ee | 중립~도움 | 중립~도움 | 중립~도움 | 중립 | 중립 |
| LR 0.001→0.0001 | yaml (add_feet_dragging부터) | 학습 속도↓ | 학습 속도↓ | 학습 속도↓ | 학습 속도↓ | 학습 속도↓ |
| num_envs 4096↔2048 진동 | yaml | 비교 어려움 | 비교 어려움 | 비교 어려움 | 비교 어려움 | 비교 어려움 |

---

## 2. terrain class별 코드 분기 (Explore 결과)

`parkour_env.py` 가 terrain class별 분기를 거의 두지 않음:
- **reward 분기 (있음)**: `is_flat = (self._env_class == TERRAIN_CLASS_FLAT)` 마스크
  - `lin_vel_z_l2`: flat=1.0, non-flat=0.1배
  - `ang_vel_xy_l2`: flat=1.0, non-flat=0.5배
  - `orientation_l2`: flat=1.0, non-flat=0
  - `base_height`: flat only
- **`feet_edge`**: `terrain_levels > 3` 일 때만 (난이도 게이트, 지형 무관)
- **reset / observation / curriculum**: terrain class 무관

→ **step/stair/gap에서만 실패하는 이유는 코드 분기가 아니라 지형 기하학(필요 토크/점프 에너지)과 보상 형태의 상호작용**.

→ **로깅 키 활용 가능**: `curriculum/mean_terrain_level_{flat,hurdle,step,gap,stair}` — H1 검증에 직접 사용.

---

## 3. 1순위 가설 (H1) — 정량 envelope

debug-worker 계산 (Tier 2):
- Go2 ≈15 kg, calf push-off 요구 ≈ 25–40 N·m
- uncommitted `saturation_effort = 23.5 N·m`, `velocity_limit = 30 rad/s` (이전 calf 30.1 동일하나 hip 52.4→30 감소가 크다)
- step ≥0.30 m / gap 0.4–0.5 m / stair 0.20 m 단차에서 peak torque 초과 → 등반 미완 → 자주 fall → 학습 신호 부족
- flat / hurdle(≤0.30 m)은 low-torque envelope 안에 들어와 학습 가능

H2 결합 메커니즘:
- H1으로 장애물 앞에서 등반 미완 → 정지
- `moving_mask` 제거로 `tracking_yaw`가 정지 상태에서도 full reward 지급 (이전 commit c3a7c43에서 제거됨, `parkour_env.py:912~918`)
- `tracking_goal_vel`(weight 1.5)이 막아준다는 가정이 H1 하에선 무너짐 (전진 자체가 불가) → standstill + yaw 정렬 local optimum 재발

---

## 4. log-analyzer 한계 명시 (중요)

log-analyzer는 **TensorBoard event 파일을 실제로 파싱하지 못함**(권한/도구 제한). 결과는 `params/*.yaml`만 비교한 것이고 시계열 메트릭은 보지 못함.

이 때문에:
- H1 확정의 정량 증거(예: step 지형 episode_length 단축, tracking_goal_vel 정체)는 **현재 보고서에 없음**
- log-analyzer가 도출한 H2/H3/H4 cfg-side 가설은 Tier 1 (값 확인), 시계열 영향은 Tier 3 (코드 추정)

→ **다음 단계로 TF event 파싱 dispatch가 필요** (별도 sub-agent + tbparse 또는 사용자 직접 TensorBoard 확인).

---

## 5. 추천 액션 (우선순위 순)

### A. 우선 확인 (사용자 응답 필요)

1. "최근 변경 후 학습 안된다"의 학습 run이 다음 중 어느 쪽인지:
   - (a) 위 5개 log run 중 하나 (uncommitted 미적용 — `_actuator_mode = 1`, `feet_dragging = -0.0`)
   - (b) uncommitted 변경(`_actuator_mode = 2`, `feet_dragging = -0.1`) 반영 후 새로 돌린 run (로그 디렉터리 경로 알려주면 추가 분석)

2. learning_rate 0.001 → 0.0001 변경이 의도된 것인지 회귀인지 (`agents/rsl_rl_ppo_cfg.py`는 uncommitted diff에 포함됨)

### B. 검증 실험 (X1, H1 confirm/reject)

조건: 1번에서 (b)임이 확인되면 즉시. (a)이면 (b) 시도 후 동일 증상 재현되는지 확인 먼저.

```
parkour_env_cfg.py:399
  _actuator_mode = 2  →  1
(다른 uncommitted 변경 — feet_dragging, next_goal_threshold — 유지)
```

학습: 1k–2k iter, num_envs 4096 권장 (배치 노이즈 컨트롤)
판정 메트릭:
- `curriculum/mean_terrain_level_{step,stair,gap}` 회복(상승) ↔ flat 대비 차이 축소 → **H1 confirm**
- 변화 없음 → **H1 reject**, H3 (LR) 또는 cfg 다른 부분 의심으로 전환

### C. 후속 (H1 confirm 시)

- `_actuator_mode = 2` 운영 의도가 있다면 reward/curriculum 재설계가 필요 (terrain 초기 level 축소, step/stair/gap proportion 단계적 증가, peak torque 보상 추가 등). reward-worker + cfg-worker 협업.

### D. 후속 (H1 reject 시)

- TF event 시계열 파싱 dispatch (별도 sub-agent — tbparse 또는 EventAccumulator)
- learning_rate 복구 시도 (0.0001 → 0.0005 또는 0.001)
- `feet_dragging` -0.1 → 0.0 단독 복원 (H4 검정)

### E. 되돌리지 말 것

- `recently_reset` 게이트 + `_scan` init (commit 2aab9ee) — bug fix
- contact `[:,0]` + 2-sample OR debounce — Genesis 정렬, 더 정확
- `next_goal_threshold 0.2` — 도움 방향
- height_scan 관련 — 이미 정상 검증됨

---

## 6. Evidence Tier 라벨링 (재확인)

| Tier | 항목 | 내용 |
|------|------|------|
| 1 | uncommitted cfg | `_actuator_mode=2`, `feet_dragging=-0.1`, `next_goal_threshold=0.2` |
| 1 | committed code | `moving_mask` 제거 (c3a7c43), contact `[:,0]` (2aab9ee), `recently_reset` (2aab9ee) |
| 1 | yaml 기록 | LR 0.001→0.0001 (add_feet_dragging부터), num_envs 4096↔2048 진동 |
| 1 | 코드 구조 | terrain class 분기는 `is_flat` mask뿐, `curriculum/mean_terrain_level_{class}` 로깅 키 존재 |
| 2 | H1 토크 envelope | Go2 15kg + saturation 23.5 N·m vs step ≥0.30m / gap 0.4–0.5m 추정 자릿수 일치 |
| 3 | H2 conditional mechanism | yaw-only standstill trap — H1 가정 위에 성립 |
| 3 | H4 magnitude 추정 | feet_dragging -0.1 per-step ≈0.0024 (보상 비율 기반) |
| 미검증 | TF event 시계열 | 누구도 안 봄 → 후속 dispatch 필요 |

---

## 7. References

- 통합본: `_workspace/parkour_step_stair_gap_RCA_summary.md` (this)
- debug-worker 본보고서: `_workspace/parkour_step_stair_gap_rca_codeside.md`
- log-analyzer 본보고서: `_workspace/parkour_step_stair_gap_log_evidence.md`
- Explore 본보고서: `_workspace/parkour_terrain_class_codepath_scan.md`
- 이전 RCA: `_workspace/parkour_gait_rca.md`
- 검증 완료 영역: `~/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_height_scan_verified.md`

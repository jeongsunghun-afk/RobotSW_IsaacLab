# May 20, 2026 — Parkour 12 실험 코드 변경 타임라인

기준: 각 실험 디렉토리의 `git/IsaacLab.diff`(main 대비 working tree 누적). 인접 페어 11번 비교.

## 코드 변경 타임라인

### 실험 1 → 실험 2 (09-50-37 → 09-50-51)
- **변경 없음** (페어, isaac 4.5 vs 5.1 동일 코드). diff 0줄.
- 두 실험 모두 baseline 대비 `max_tilt: 1.0 → 1.2` (`parkour_env_cfg.py`).

### 실험 2 → 실험 3 (09-50-51 → 10-29-03)
- 파일: `scripts/reinforcement_learning/rsl_rl/play.py`
- 변경: `env_cfg.debug_vis_edge_mask=True` 라인의 위치/가드 변경 (height_scanner 블록 → 별도 `hasattr` 가드 블록 분리).
- 파일: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py`
- 변경: `max_tilt: 1.2 → 1.309` (≈ 75° 허용, 약 12° 완화).
- 의미(추정): play 모드 debug viz 안전화 + tilt 허용각 추가 완화.

### 실험 3 → 실험 4 (10-29-03 → 10-29-06)
- **변경 없음** (페어, isaac 4.5 vs 5.1). diff 0줄.

### 실험 4 → 실험 5 (10-29-06 → 12-27-42 change_reward)
- 파일: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py:998`
  - `lin_vel_z_l2 *= (is_flat + is_non_flat * 0.1) → * 0.5` (non-flat 지형 수직 속도 패널티 5배 강화).
- 파일: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py:1182` (`_reset_idx`)
  - `joint_pos += (torch.rand_like(joint_pos)*0.1 - 0.05)` 초기 관절 ±0.05 rad 노이즈 추가.
- 파일: `parkour_terrains.py` (terrain cfg)
  - `parkour_flat` proportion `0.1 → 0.2`, `parkour_gap` proportion `0.3 → 0.2` (flat 비중 ↑, gap ↓).
- 의미(추정): non-flat에서 수직속도 억제 강화 + 시작 자세 다양화 + 평지 비중 증가.

### 실험 5 → 실험 6 (12-27-42 → 12-27-46)
- **변경 없음** (페어). diff 0줄. 다만 max_tilt가 이미 `1.0 → 1.309`로 적용된 상태.

### 실험 6 → 실험 7 (12-27-46 → 14-25-07 no_stair_env)
- **코드 변경 없음**. diff는 untracked 파일/디렉토리 추가만:
  - `.claude/feedback/sessions/2026-05-20-parkour.md`
  - `source/isaaclab_assets/data/Robots/Go2_4.5/`, `Go2_5.1/` 디렉토리 등장.
- 의미(추정): "no_stair_env"는 코드 diff에 나타나지 않음 → terrain cfg 변경이 누락됐거나 런타임 args로만 stair 제거. **주의 필요**.

### 실험 7 → 실험 8 (14-25-07 → 14-43-58 change_seed)
- **코드 변경 없음**. diff 0줄. seed는 런타임 CLI 인자(`--seed`)로만 변경된 것으로 추정.

### 실험 8 → 실험 9 (14-43-58 → 15-02-27 no_stair_env)
- 파일: `parkour_env.py:998` `lin_vel_z_l2` 계수 `0.5 → 0.1` (실험 4→5 변경의 **롤백**).
- 파일: `parkour_env.py:1107` `_get_rewards` 끝:
  - `return total_reward` → `return torch.clip(total_reward, min=0.)` (음수 보상 클리핑 추가).
- 파일: `parkour_env.py:_get_dones` (1109~1153):
  - "bootstrap-all" 전략으로 전면 재작성. `time_out = self.episode_length_buf >= max-1` → 변수명 `episode_timeout`, `terminated = self._term_tilt | self._term_low_height` 만, `reset_all = (failure & ~grace) | _term_goal_reached | episode_timeout`, `terminated = zeros_like`, `time_out = reset_all` (모든 reset을 truncation으로 처리해 PPO γ·V(s') bootstrap 항상 적용). B 레퍼런스와 align.
- 파일: `parkour_env.py:_reset_idx` joint_pos 랜덤 노이즈 추가 (실험 5에서 이미 들어갔던 변경이 재유지).
- 파일: `parkour_env_cfg.py:386` `clip_actions: 10.0 → 4.8` (action clip 더 빡빡하게).
- 의미(추정): reward 클리핑+bootstrap-all+action clip 좁힘이 같은 커밋 흐름에 묶임.

### 실험 9 → 실험 10 (15-02-27 → 16-07-33)
- `parkour_env.py` 차이 **사라짐**(실험 9의 코드 변경이 stash/checkout 됨).
- 결과: lin_vel_z 0.5 ⇒ 0.1 롤백/clip(min=0)/bootstrap-all/joint noise 모두 **revert**된 상태로 실험 10 실행 (단 `parkour_env_cfg.py`의 `clip_actions=4.8`은 유지).
- 의미(추정): 의도된 "베이스라인 재측정" 또는 변경 revert 누락.

### 실험 10 → 실험 11 (16-07-33 → 17-14-12 clip_actions_10)
- 파일: `parkour_env_cfg.py:384-388` `clip_actions: 4.8 → 10.0` (주석/실값 토글).
- 의미: 이름 그대로 action clip 다시 10.0으로 완화.

### 실험 11 → 실험 12 (17-14-12 → 18-02-12 goal_idx_visualization)
- 파일: `parkour_env.py:1154-1156` `_reset_idx`: `_pre_reset_goal_idx = self._current_goal_idx[env_ids].clone()` 캡처.
- 파일: `parkour_env.py:1231-1247` end-of-episode 로깅 추가:
  - `Goal_Progress/max_idx_at_reset`, `mean_idx_at_reset`, `median_idx_at_reset`
  - 히스토그램: `Goal_Progress/count_idx_eq_{k}` (`k = 0..num_goals-1`)
- 의미: 텔레메트리 전용 변경(TEMP_DIAG). 학습 로직 영향 없음.

## 페어 검증
- **페어 1 (1↔2, 09-50-37 ↔ 09-50-51)**: diff 0줄. 완전 동일.
- **페어 2 (3↔4, 10-29-03 ↔ 10-29-06)**: diff 0줄. 완전 동일.
- **페어 3 (5↔6, 12-27-42 ↔ 12-27-46)**: diff 0줄. 완전 동일.
- 결론: 같은 timestamp prefix 페어는 isaac 4.5 vs 5.1 빌드만 다르며 코드는 비트 단위로 동일.

## 핵심 발견
1. **실험 5 변경(reward+terrain+joint noise)이 페어를 넘나든다**: change_tilt_end(2→3)에서 tilt만 바뀐 게 아니라 play.py 가드도 정리됨. change_reward(4→5)에서는 reward·terrain·init 3종이 한 번에 들어감 — "change_reward"라는 이름과 달리 terrain proportion(`flat 0.1→0.2`, `gap 0.3→0.2`)·init joint noise도 함께 변경됨. 변경 격리 측면에서 주의.
2. **실험 9는 본 12개 중 가장 큰 단일 점프**: bootstrap-all _get_dones 재작성 + reward clip(min=0) + clip_actions 10→4.8 + lin_vel_z 0.5→0.1 롤백을 동시 적용. "no_stair_env"라는 이름과 무관하게 RL 신호의 핵심부가 바뀜.
3. **실험 10에서 9의 핵심 변경이 사라짐**: bootstrap-all/clip(min=0)/lin_vel_z 0.1/joint noise 가 모두 revert (단 `clip_actions=4.8`은 잔존). 디렉토리명이 익명(`16-07-33`)이라 의도가 불명확. 사용자가 의도한 revert인지 stash 누락인지 확인 필요(추정).
4. **실험 7 "no_stair_env"는 코드 diff에 stair 제거 흔적 없음** — 런타임 인자 또는 외부 cfg override일 가능성. 실제 stair 제거가 적용됐는지 검증 필요(추정).
5. **실험 8 "change_seed"는 코드 diff 0줄** — seed는 CLI 인자로만 변경됐고 working tree는 실험 7과 동일.
6. **실험 12는 학습 로직 무변화**: `Goal_Progress/*` 로깅만 추가 (TEMP_DIAG). 학습 성능 차이가 보이면 코드가 아닌 다른 요인.

## 변경량 요약 (인접 페어)
| 전이 | 핵심 변경 | 규모 |
|---|---|---|
| 1→2 | 없음 | 0 |
| 2→3 | play.py guard + max_tilt 1.2→1.309 | S |
| 3→4 | 없음 | 0 |
| 4→5 | lin_vel_z 0.1→0.5 + joint noise + terrain proportion | M |
| 5→6 | 없음 | 0 |
| 6→7 | untracked만 | 0 (코드) |
| 7→8 | 없음 | 0 |
| 8→9 | _get_dones 재작성 + reward clip + clip_actions 4.8 + lin_vel_z revert + joint noise | **L** |
| 9→10 | parkour_env.py 변경 revert | M |
| 10→11 | clip_actions 4.8→10.0 | S |
| 11→12 | Goal_Progress 로깅 추가 | S (텔레메트리) |

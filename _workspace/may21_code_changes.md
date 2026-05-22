# May 21 Parkour 실험 — 코드 변경 분석

> **출처**: 각 실험 디렉토리의 `git/IsaacLab.diff` 스냅샷 (학습 시점 작업 트리 vs **origin/main**).
> Baseline (실험 비교용): `2026-05-20_17-14-12_clip_actions_10` (`clip_actions=10.0`).
>
> ⚠️ **해석 주의**: `IsaacLab.diff`는 origin/main 대비 누적 working-tree drift다. **"이전 실험 대비" 실제 활성 변경**을 확인하려면 `params/env.yaml`(런타임 dump)을 비교해야 한다. 본 문서에서 "clip_actions 4.8 → 16.0"처럼 보이는 항목은 diff baseline의 차이일 뿐, 05/20→05/21 실험 간 활성 변경은 `_workspace/may21_params_changes.md`와 `may21_parkour_analysis.md` 표를 신뢰할 것.

## 실험별 핵심 변경

### 실험 1: `2026-05-21_09-49-21_goal_idx_visualization3`
- **수정 파일**: `parkour_env.py` 만
- **변경 성격**: 순수 진단/로깅 (행동 영향 없음)
- `_reset_env()`에 추가된 로깅:
  - `Goal_Progress/*` — max/mean/median goal index at reset
  - `Goal_Debug/*` — 첫 goal 위치 vs 리셋 위치 거리
  - `Hold_Debug/*` — 마지막 goal에서의 hold time 진단
- Git 상태: "ahead of origin/main by 1 commit"

### 실험 2: `2026-05-21_10-05-51_clip_actions_16`
- **수정 파일**: `parkour_env_cfg.py`, `parkour_terrains.py`
- **핵심 변경**:
  - `clip_actions`: 4.8 → **16.0** (3.3배 ↑, 공격적)
  - `max_tilt`: 1.0 → **1.309** (terminate 각도 완화)
  - **Terrain 난이도 대폭 상향**:
    - `parkour_flat` proportion 0.1→0.2, x_spacing (1.0,1.5)→(1.0,2.4)
    - `parkour_hurdle` x_spacing (1.0,1.5)→**(1.8,2.4)**
    - `parkour_stepping` x_length (0.4,0.8)→**(1.2,2.0)**
    - `parkour_gap` proportion 0.3→0.2, gap_length (0.05,0.5)→(0.05,0.6), platform_length (1.2,1.6)→(1.2,2.0)
    - `parkour_stair` stair_height (0.05,0.20)→**(0.05,0.25)**

### 실험 3: `2026-05-21_10-11-16_clip_actions10_action_rate_0.01`
- **수정 파일**: `parkour_env_cfg.py`, `parkour_terrains.py`
- **변경**:
  - `clip_actions`: 4.8 → **10.0** (16.0에서 롤백, 중간값)
  - `action_rate_l2`: -0.05 → **-0.01** (5배 ↓, 코멘트: "WAS -0.1, 10x error")
  - Terrain 설정은 Exp 2 동일

### 실험 4: `2026-05-21_14-49-35_change_terrain_difficulty`
- **수정 파일**: `parkour_env_cfg.py`, `parkour_terrains.py`
- **변경**:
  - Exp 3 동일 파라미터 +
  - **Terrain 크기 20.0 → 25.0 m (x 방향)**
  - `parkour_stair`: `num_steps_per_stair=8` 추가, `stair_height_range=(0.05, 0.25)`
- 짧게 종료(model_100까지). 이유 불명확.

### 실험 5: `2026-05-21_14-58-19_change_terrain_difficulty`
- **수정 파일**: 동일
- **변경**: Exp 4와 **동일** (사용자가 14:58에 재시작)

### 실험 6: `2026-05-21_15-48-19_feet_dragging`
- **수정 파일**: `parkour_env.py`, `parkour_env_cfg.py`
- **신규 reward (hind feet drag penalty)**:
  ```python
  hind_feet_ids = self._feet_ids[2:4]  # URDF: [FL, FR, RL, RR]
  hind_xy_vel_norm = torch.norm(self._robot.data.body_lin_vel_w[:, hind_feet_ids, :2], dim=-1)
  hind_contact = contact_filt[:, 2:4]
  is_dragging = hind_contact & (hind_xy_vel_norm > self.cfg.dragging_velocity_threshold)
  feet_dragging = torch.sum(hind_xy_vel_norm * is_dragging.float(), dim=-1)
  ```
- Hind feet (RL, RR) assertion 추가
- Cfg 추가: `dragging_velocity_threshold=0.05` (m/s)
- Reward scale 추가: `feet_dragging: -0.1`

### 실험 7: `2026-05-21_17-31-36_add_feet_air_time_flat` ⚠️ 망가짐 케이스
- **수정 파일**: `parkour_env.py`, `parkour_env_cfg.py`
- **신규 reward (ANYmal-style air-time bonus, flat-only gate)**:
  ```python
  last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
  first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
  air_time_bonus = (last_air_time - self.cfg.feet_air_time_threshold) * first_contact.float()
  cmd_speed = torch.norm(self._commands[:, :2], dim=-1)
  gate = (cmd_speed > self.cfg.feet_air_time_cmd_speed_threshold).float()
  feet_air_time = torch.sum(air_time_bonus, dim=-1) * gate * is_flat
  ```
- Cfg 추가:
  - `feet_air_time_threshold=0.3 s`
  - `feet_air_time_cmd_speed_threshold=0.1 m/s`
- Reward scale 추가:
  - `feet_dragging: -0.1` (Exp 6에서 유지)
  - `feet_air_time: +0.5` (ANYmal 스타일 보상)
- `is_flat` 게이트가 "flat terrain에서만" 적용된다는 주석.

## 실험 간 변경 누적 매트릭스

| 항목                | Exp1 | Exp2  | Exp3  | Exp4/5 | Exp6  | Exp7  |
| ------------------- | ---- | ----- | ----- | ------ | ----- | ----- |
| 로깅 확장           | ✓    | -     | -     | -      | -     | -     |
| clip_actions        | 4.8  | **16.0** | 10.0  | 10.0   | 10.0  | 10.0  |
| max_tilt            | 1.0  | **1.309** | 1.309 | 1.309  | 1.309 | 1.309 |
| action_rate_l2      | -0.05 | -0.05 | **-0.01** | -0.01 | -0.01 | -0.01 |
| terrain x_size      | 20   | 20    | 20    | **25** | 20    | 20    |
| feet_dragging       | -    | -     | -     | -      | **-0.1** | **-0.1** |
| feet_air_time       | -    | -     | -     | -      | -     | **+0.5** |

## 실제 git commit (05/20 18:00 ~ 05/22)

대부분의 실험에서 diff는 **로컬 working-tree** 변경 스냅샷이고 commit되지 않음 (Exp 2-7 모두 "up to date with origin/main").

`git log`로 확인된 commit (parkour 영역):
- `dff065bccf9` — 2026-05-21 15:09 — **"Place parkour_hurdle goals between hurdles instead of right after each"**
  - 시각상 Exp 5 (14:58) 종료 후 ~ Exp 6 (15:48) 시작 사이.
  - 즉, Exp 4/5의 terrain_difficulty 변경 결과를 commit 한 것으로 추정.
- 이전 commit (05/20):
  - `29afd7d3dfa` — Analyze May 20 series
  - `75ab6024aad` — Bootstrap-all termination policy
  - `c7261bd37f9` — RMA estimator + B-style DR
  - `25d2071196f` — Align reward set with B

## 변경의 영향 (가설)

### Action Clipping (Exp 2 ↔ Exp 3)
- Exp 2의 16.0은 너무 큰 액션 폭 → 발 제어 오버슈트 가능.
- Exp 3에서 10.0으로 조정 + action_rate 페널티 완화는 합리적.
- 코멘트 "10x error"는 -0.05가 원래 -0.1로 잘못 설정되어 5배 강했다는 의미일 가능성.

### Terrain Difficulty (Exp 2-5)
- 누적 효과: 장애물 간격/크기/높이 모두 2배+. parkour 난이도 급상승.
- Exp 4/5의 25 m 폭은 더 긴 경로 → curriculum 진행이 느려질 위험.

### Feet Dragging (Exp 6)
- Hind feet 마찰 억제 → genesis 호환성 + 자연스러운 보행.
- -0.1 스케일은 중간 강도(예: feet_stumble = -1.0의 1/10).

### Feet Air Time (Exp 7) — 망가짐 원인 후보
1. **스케일 과대**: +0.5는 다른 양수 reward 대비 매우 큰 값. 예시: tracking_goal_vel=1.5, feet_stumble=-1.0. air-time bonus가 stability를 압도할 수 있음.
2. **`is_flat` 게이트 동작 의문**: 게이트가 항상 False라면 학습 무의미. 일부만 True라면 불안정한 신호.
3. **임계값 부적합**: 0.3 s는 ANYmal(0.5 s)와 Go1(0.25 s) 사이값. Go2의 자연 스윙 사이클과 어긋날 수 있음.
4. **첫 접지 시점 bias**: `compute_first_contact`가 episode 시작 직후 0에서 reset되어 음수 bonus 발생 → 다리를 들고 있도록 학습할 유인.

→ 정량 검증은 tfevents 분석으로 추가 확인 필요 (별도 보고서).

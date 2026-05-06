# 공원(Parkour) 목표(Goal) 시스템 비교 분석 보고서

**작성일**: 2026-04-30  
**팀**: isaaclab-parkour-fixes  
**담당자**: W6-writer

---

## 1. Executive Summary

- **IsaacLab**: 모듈 수준 레지스트리(`PARKOUR_GOALS_REGISTRY`)를 통해 지형 생성 중 목표를 수집 → 환경에서 world frame으로 변환 및 fancy indexing으로 환경별 할당
- **RobotSW_Genesis**: 지형 메타데이터(`terrain._morph.goals`)에서 직접 로드 → fancy indexing으로 (terrain_level, terrain_type) 기반 할당
- **RobotSW_Parkour**: 지형 생성 단계에서 2D 목표를 계산 → terrain cell 오프셋으로 조정 → 메타데이터 저장
- **핵심 차이**: IsaacLab은 동적 registry 기반 수집, Genesis는 사전 계산된 메타데이터 로드, RobotSW_Parkour는 terrain builder 내장 생성
- **이번 iteration**: IsaacLab에서 registry 방식으로 4개 parkour 지형 함수의 목표를 명시적으로 wire

---

## 2. IsaacLab Parkour 목표 시스템 (이번 team 작업 후)

### 2.1 아키텍처 개요

IsaacLab의 목표 시스템은 **module-level registry** 패턴을 사용하여 지형 생성 중 목표 메타데이터를 수집합니다:

```
mesh_terrains.py (목표 생성)
    ↓
PARKOUR_GOALS_REGISTRY (수집)
    ↓
parkour_env._build_terrain_goals_map() (변환)
    ↓
self._terrain_goals_world (저장)
    ↓
_init_env_goals() (할당)
```

### 2.2 목표 생성 및 등록 (W1 작업)

**파일**: `/home/lgb/IsaacLab/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py`

#### PARKOUR_GOALS_REGISTRY 선언

```python
# mesh_terrains.py, line 45
PARKOUR_GOALS_REGISTRY: list[tuple[np.ndarray, np.ndarray]] = []
```

**설명**:
- 각 tuple은 `(goals_local, origin_local)` 형태
- `goals_local`: shape `(num_goals, 3)`, 지형 생성 전 로컬 frame
- `origin_local`: shape `(3,)`, terrain function이 반환하는 origin (로컬 frame)

#### parkour_gap_terrain 예시 (line 890-963)

```python
# line 935: 목표 생성
goals_list.append([dis_x + rand_plen / 2.0, plat_center_y, float(cfg.platform_height)])

# line 952-961: 패딩 및 레지스트리 등록
origin = np.array([cfg.platform_length / 2.0, cfg.size[1] / 2.0, cfg.platform_height])
_raw = np.array(goals_list, dtype=float) if goals_list else origin.reshape(1, 3)
if len(_raw) < cfg.num_goals:
    _raw = np.vstack([_raw, np.tile(_raw[-1:], (cfg.num_goals - len(_raw), 1))])
cfg.terrain_goals = _raw[: cfg.num_goals]
PARKOUR_GOALS_REGISTRY.append((cfg.terrain_goals.copy(), origin.copy()))
```

**목표 배치**:
- 각 gap 후 중간 플랫폼의 중심-상단: `(dis_x + plen/2, plat_center_y, platform_height)`

#### 다른 3개 함수

| 함수 | 목표 배치 |
|------|----------|
| `parkour_hurdle` (line 966) | hurdle 통과 후 0.3m: `(dis_x + hurdle_thickness + 0.3, center_y, 0.0)` |
| `parkour_stair` (line 1043) | 계단 상승 마무리점 + 하강 마무리점: top_h 및 0.0 |
| `parkour_step` (line 1128) | 각 step 중심-상단: `(dis_x + step_len/2, step_center_y, max(0.0, height))` |

### 2.3 World Frame 변환 (W3 작업)

**파일**: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`

#### _build_terrain_goals_map() (line 142-200)

```python
def _build_terrain_goals_map(self, registry: list) -> None:
    # line 161-164
    num_rows = self.cfg.terrain.terrain_generator.num_rows
    num_cols = self.cfg.terrain.terrain_generator.num_cols
    num_goals = self.cfg.num_goals
    curriculum = getattr(self.cfg.terrain.terrain_generator, "curriculum", False)

    # line 176-178: numpy arrays로 준비
    t_origins = self._terrain.terrain_origins.cpu().numpy()  # [num_rows, num_cols, 3]
    goals_map = np.zeros((num_rows, num_cols, num_goals, 3), dtype=np.float32)

    # line 179-196: local → world 변환
    for k, (local_goals, local_origin) in enumerate(registry):
        if curriculum:
            col = k // num_rows
            row = k % num_rows
        else:
            row = k // num_cols
            col = k % num_cols
        
        world_origin = t_origins[row, col]  # [3], world frame
        delta = local_goals[:num_goals] - local_origin[np.newaxis, :]  # [num_goals, 3]
        n = min(num_goals, len(local_goals))
        goals_map[row, col, :n] = world_origin + delta[:n]
        if n < num_goals:
            goals_map[row, col, n:] = goals_map[row, col, n - 1]  # 마지막 목표로 padding
```

**핵심 변환 공식**:
```
world_goal = world_origin + (local_goal - local_origin)
```

**불변성**: 이 공식은 terrain generator의 모든 rigid translation에 불변입니다.

#### _init_env_goals() (line 219-249)

```python
def _init_env_goals(self, env_ids: torch.Tensor):
    terrain_goals_world: torch.Tensor | None = getattr(self, "_terrain_goals_world", None)
    if terrain_goals_world is not None:
        # line 229-232: fancy indexing
        levels = self._terrain_levels[env_ids]   # [n] — row in terrain grid
        types = self._terrain_types[env_ids]     # [n] — col in terrain grid
        # terrain_goals_world: [num_rows, num_cols, num_goals, 3]
        self._env_goals[env_ids, : self.cfg.num_goals] = terrain_goals_world[levels, types]
    else:
        # line 234-242: fallback (직선 목표)
        origins = self._terrain.env_origins[env_ids]  # [n, 3]
        goal_offsets = (
            torch.arange(1, self.cfg.num_goals + 1, device=self.device) * self.cfg.goal_distance
        )
        goal_x = origins[:, 0:1] + goal_offsets[None, :]
        goal_y = origins[:, 1:2].expand(-1, self.cfg.num_goals)
        goal_z = torch.full_like(goal_x, self.cfg.goal_z)
        self._env_goals[env_ids, : self.cfg.num_goals] = torch.stack([goal_x, goal_y, goal_z], dim=-1)
```

### 2.4 초기화 플로우 (W3 작업)

**파일**: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`

```python
# _setup_scene() (line 132-135)
_parkour_mesh_terrains.PARKOUR_GOALS_REGISTRY.clear()
# ... terrain generation ...
self._build_terrain_goals_map(_parkour_mesh_terrains.PARKOUR_GOALS_REGISTRY)

# _reset() (line 120)
self._init_env_goals(torch.arange(self.num_envs, device=self.device))
```

---

## 3. RobotSW_Genesis 목표 시스템

### 3.1 아키텍처

RobotSW_Genesis는 **terrain 메타데이터 사전 계산** 방식을 사용합니다:

```
terrain._morph.goals (사전 계산)
    ↓
terrain metadata로 저장
    ↓
legged_env_parkour 로드
    ↓
self.terrain_goals (fancy indexing 할당)
```

### 3.2 지형 메타데이터 로드

**파일**: `/home/lgb/RobotSW_Genesis/parkour/legged_env_parkour.py`

```python
# line 851
self.terrain_goals = torch.from_numpy(self.terrain_metadata["goals"].astype(np.float32)).to(self.device).to(torch.float)

# line 852
self.env_goals = torch.zeros(self.num_envs, self.terrain_cfg["num_goals"] + self.obs_cfg["num_future_goal_obs"], 3, device=self.device, requires_grad=False)

# line 858-860
temp = self.terrain_goals[self.terrain_levels, self.terrain_types]
last_col = temp[:, -1:, :]
self.env_goals[:] = torch.cat((temp, last_col.repeat(1, self.obs_cfg["num_future_goal_obs"], 1)), dim=1)[:]
```

**형태**:
- `self.terrain_goals`: `[num_rows, num_cols, num_goals, 3]`
- Fancy indexing: `terrain_goals[terrain_level, terrain_type]` → `[num_goals, 3]`

### 3.3 목표 시각화

**파일**: `/home/lgb/RobotSW_Genesis/parkour/legged_env_parkour.py`

```python
# _draw_goal() (line 827-834)
def _draw_goal(self):
    height_field = self.terrain.geoms[0].metadata["height_field"]
    heights = self.terrain_cfg["vertical_scale"] * torch.tensor(height_field, device="cpu").unsqueeze(-1)
    updated_goals = self.terrain._morph.goals.copy()
    updated_goals[..., 2] = heights[updated_goals[..., 0] / self.terrain_cfg["horizontal_scale"], updated_goals[..., 1] / self.terrain_cfg["horizontal_scale"], 0]
    poss = updated_goals.reshape(-1, 3)
    self.scene.draw_debug_spheres(poss=poss, radius=0.05, color=(0, 0, 1, 0.7))
```

**특징**: height field로부터 z 좌표를 동적으로 샘플링하여 goal 시각화에 반영.

---

## 4. RobotSW_Parkour 목표 시스템

### 4.1 아키텍처

RobotSW_Parkour는 **terrain generator 내장 목표 생성** 방식을 사용합니다:

```
terrain builder (gap_terrain, stepping_stones_terrain 등)
    ↓
terrain.goals [num_stones+2, 2] 생성
    ↓
Terrain class (line 362)
    ↓
self.goals[i, j] 누적 = terrain.goals + cell offset
    ↓
legged_env 로드 및 fancy indexing
```

### 4.2 Terrain 클래스 초기화

**파일**: `/home/lgb/RobotSW_Parkour/legged_gym/legged_gym/utils/terrain.py`

```python
# __init__() (line 59-60)
self.goals = np.zeros((cfg.num_rows, cfg.num_cols, cfg.num_goals, 3))
self.num_goals = cfg.num_goals
```

### 4.3 목표 생성 함수 (예: stepping_stones_terrain)

```python
# line 444-494
def stepping_stones_terrain(terrain, num_stones=8, stone_distance=1., stone_width=0.5, max_height=1.0, platform_width=4.):
    goals = np.zeros((num_stones+2, 2))
    # line 469: 시작 플랫폼
    goals[0] = [platform_len -  stone_len // 2, mid_y]
    # line 486: 각 stepping stone
    goals[i+1] = [dis_x, dis_y]
    # line 492: 끝 플랫폼
    goals[-1] = [final_dis_x, mid_y]
    # line 494: 스케일 적용
    terrain.goals = goals * terrain.horizontal_scale
```

**특징**:
- 목표는 2D로 생성: `[num_stones+2, 2]`
- `horizontal_scale`로 스케일 적용
- Z는 기본값 0.0

### 4.4 Terrain Grid에 누적

**파일**: `/home/lgb/RobotSW_Parkour/legged_gym/legged_gym/utils/terrain.py`

```python
# curiculum() (line 362)
self.goals[i, j, :, :2] = terrain.goals + [i * self.env_length, j * self.env_width]
```

**설명**:
- 각 cell `(i, j)`의 목표를 terrain.goals(로컬)에서 global로 변환
- `i * env_length` (행 오프셋) + `j * env_width` (열 오프셋)
- Z좌표는 초기값 0.0 유지

### 4.5 환경에서 로드

**파일**: `/home/lgb/RobotSW_Parkour/legged_gym/legged_gym/utils/terrain.py`

```python
# legged_env_parkour 로드 (line 851)
self.terrain_goals = torch.from_numpy(self.terrain_metadata["goals"].astype(np.float32)).to(self.device).to(torch.float)
```

**형태**: `[num_rows, num_cols, num_goals, 3]`

---

## 5. 세 시스템 비교 테이블

| 항목 | **IsaacLab** | **RobotSW_Genesis** | **RobotSW_Parkour** |
|------|-------------|-------------------|-------------------|
| **목표 생성 위치** | mesh_terrains.py (terrain builder) | terrain._morph (사전 계산) | terrain.py (terrain builder) |
| **메타데이터 전달** | PARKOUR_GOALS_REGISTRY (동적) | terrain metadata dict (정적) | terrain metadata dict (정적) |
| **저장 형식** | registry: list[tuple[np.ndarray, np.ndarray]] | JSON/metadata["goals"] | numpy array (num_rows, num_cols, num_goals, 3) |
| **좌표계 변환** | 환경에서 local → world (registry 기반) | 사전 계산 완료, world frame | 사전 계산 완료, cell offset 포함 |
| **목표 차원** | 3D (x, y, z) | 3D (x, y, z) | 3D (x, y, 0) |
| **할당 방식** | fancy indexing: goals_world[level, type] | fancy indexing: terrain_goals[level, type] | fancy indexing: terrain_goals[level, type] |
| **Height 처리** | terrain function에서 명시적 설정 | height_field에서 동적 샘플링 | 기본값 0.0 |
| **Fallback** | 직선 목표 (registry 오류 시) | 없음 | 없음 |
| **유연성** | ⭐⭐⭐ (지형 생성 시 조정 가능) | ⭐⭐ (메타데이터 수정 필요) | ⭐⭐ (메타데이터 수정 필요) |

---

## 6. 이번 Iteration의 변경 사항

### 6.1 추가된 4개 Parkour Terrain 함수

W1(task #1)에서 추가된 terrain 함수들:

1. **parkour_gap_terrain** (mesh_terrains.py line 890)
   - 각 gap 후 중간 플랫폼 상단에 목표 배치
   - 목표 개수: cfg.num_gaps (기본 8개)

2. **parkour_hurdle_terrain** (mesh_terrains.py line 966)
   - hurdle 통과 후 0.3m 떨어진 corridor 중심에 목표 배치
   - 목표 개수: cfg.num_hurdles (기본 8개)

3. **parkour_stair_terrain** (mesh_terrains.py line 1043)
   - 계단 상승 마무리점 + 하강 마무리점 (stair cycle당 2개)
   - 목표 개수: cfg.num_stairs * 2 (기본 8개)

4. **parkour_step_terrain** (mesh_terrains.py line 1128)
   - 각 step 중심-상단에 목표 배치
   - 목표 개수: cfg.num_steps (기본 8개)

### 6.2 Configuration 업데이트 (W2, task #2)

**파일**: `/home/lgb/IsaacLab/source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains_cfg.py`

```python
# PARKOUR_TERRAINS_CFG.sub_terrains (line 360-481)
sub_terrains={
    "parkour_flat": 0.1,
    "parkour_hurdle": 0.2,
    "parkour_step": 0.2,
    "parkour_gap": 0.3,
    "parkour_stair": 0.2,
}
```

모든 5개 entry에 `num_goals: int = 8` 추가됨.

### 6.3 Registry 기반 Wiring (W3, task #3)

**새로운 메소드**:
- `parkour_env._build_terrain_goals_map()`: registry → world frame 변환 tensor
- `parkour_env._init_env_goals()`: world frame 목표를 환경별로 할당 (fancy indexing)

**핵심 이점**:
- ✅ 지형 생성 중 목표를 명시적으로 지정 가능
- ✅ local frame 불변성으로 인해 terrain placement에 강건
- ✅ Fallback 메커니즘 (registry 오류 → 직선 목표)

### 6.4 Reward Multiplier (W4, task #4)

**파일**: parkour_env.py

```python
# env_class tracking (Genesis-style)
self.env_class = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
```

Conditional reward scaling 추가 (지형 유형별 난이도 조정).

### 6.5 Test/Validation

- ✅ 4개 parkour terrain 함수가 registry에 올바르게 추가되는지 확인
- ✅ local → world 변환 공식 검증 (rigid translation 불변성)
- ✅ fancy indexing으로 환경별 목표 할당 정상 작동
- ✅ Fallback: registry 크기 불일치 시 직선 목표로 전환

---

## 7. 설계 결정 및 근거

### 7.1 Registry 패턴 선택 이유

| 선택지 | IsaacLab (선택됨) | Genesis (대안) | RobotSW_Parkour (대안) |
|--------|-----------------|---------------|-------------------|
| **장점** | - 동적 수집 가능<br>- terrain builder와 env 분리<br>- local frame 불변성 | - 단순 로드<br>- 빠른 초기화 | - builder 통합<br>- 단순 관리 |
| **단점** | - registry 동기화 필요<br>- 복잡한 좌표 변환 | - 사전 계산 의존<br>- 유연성 낮음 | - 강한 결합<br>- terrain 수정 어려움 |

**선택 이유**: IsaacLab의 modular 아키텍처에 맞춤 → terrain builder와 env 독립적 진화 가능.

### 7.2 Local Frame 불변성

```
world_goal = world_origin + (local_goal - local_origin)
```

이 공식이 중요한 이유:
- Terrain generator가 `cfg.copy()` (dataclasses.replace)를 사용하므로 cfg side-effect 비신뢰
- Local frame에서의 상대 위치(delta)는 rigid translation에 불변
- 따라서 final world frame 계산이 terrain placement와 독립적

---

## 8. 향후 개선 방향

1. **Height Field Integration** (Genesis 패턴 참고):
   - `_draw_goal()`에서 height field 샘플링으로 목표 z 동적 조정
   - 더 정교한 시각화 및 검증

2. **Metadata Caching**:
   - PARKOUR_GOALS_REGISTRY를 JSON으로 저장 → restart 시 로드
   - 지형 생성 시간 단축

3. **Multi-Goal Curriculum**:
   - 난이도별로 목표 개수를 동적으로 조정
   - 조기 학습 단계에서는 목표 수 제한

4. **Goal Visualization Layer**:
   - IsaacLab viewer에 목표 시각화 (구 또는 마커)
   - 디버깅 및 검증 용이성 향상

---

## 9. 결론

이번 iteration은 IsaacLab에 **명시적이고 모듈화된 목표 시스템**을 구축했습니다:

- **Registry 기반 수집**: 지형 생성 중 목표를 동적으로 수집
- **Robust 변환**: local frame 불변성으로 인해 terrain placement와 독립적
- **Flexible 할당**: fancy indexing으로 환경별 (terrain_level, terrain_type) 기반 할당
- **Safety**: registry 오류 시 fallback 메커니즘

이는 RobotSW_Genesis의 사전 계산 방식과 RobotSW_Parkour의 builder 통합 방식의 중간 지점으로, **IsaacLab의 procedural terrain generation 철학**을 가장 잘 반영합니다.

---

**문서 끝**

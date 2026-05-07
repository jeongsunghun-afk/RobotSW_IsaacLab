# Parkour 신규 지형 5종 검토 + 구현 계획

> Owner: `planner-newterrains` · Task #2 · Team `parkour-terrain-design`
> Codebase basis: `parkour_terrains.py`, `mesh_terrains.py:890-1203`, `mesh_terrains_cfg.py:323-483`, `parkour_env_cfg.py`, `parkour_env.py:580-779`
> 본 문서는 plan 전용 — 코드 변경 없음. crawl 지형은 Task #1에서 다루므로 본 후보에서 제외.

---

## 0. Executive Summary

기존 5종 (`parkour_flat`, `parkour_hurdle`, `parkour_step`, `parkour_gap`, `parkour_stair`)은 (1) 일정 높이 박스 점프, (2) 피라미드 상하강, (3) X 방향 갭 점프, (4) 다단 계단 — 즉 "수직 변화 + 직선 전진"에 강하게 편중되어 있다. Robot이 실세계 parkour에서 요구되는 **(a) 비대칭/지그재그 발판, (b) 좁은 통로 가로지르기, (c) 비평탄 요철, (d) 경사면 추진, (e) 정밀 발 배치** 같은 스킬은 노출되지 않는다.

후보 5종은 각 axis를 1개 이상 새로 커버하면서, 모두 IsaacLab 코어 컨벤션(trimesh + `box`)으로 구현 가능하다. **Top-1 추천**: `parkour_stepping_stones` (디딤돌). 가장 큰 학습 가치 / 가장 낮은 시스템 변경 / 가장 강한 sim-to-real 임팩트(Barkour weave & 야외 디딤돌과 직접 매핑).

선정 5개:
1. **parkour_stepping_stones** — 비연속 디딤돌, 정밀 foot placement
2. **parkour_balance_beam** — 좁은 외나무다리, 측면 균형 / lateral COM 제어
3. **parkour_slope** — 경사면(오르막→내리막), pitch 적응 + 슬립 마진
4. **parkour_zigzag_hurdles** — 좌/우 교차 hurdle, heading 전환 + 측면 회피
5. **parkour_rough_blocks** — 작은 무작위 요철 블록 패치, 다지점 균형 / unstructured ground

---

## A. 후보 5개 선정 + 학습 가치 정당화

| # | 이름 | 핵심 스킬 | 기존과 중복? | sim-to-real 가치 | 난이도 axis (difficulty 0→1) |
|---|------|----------|-------------|------------------|-----------------------------|
| 1 | `parkour_stepping_stones` | 정밀 발 배치, 비연속 지지면 위 자세 안정화 | gap의 1축 점프와 다름 — **2D 패턴 매칭** 필요 | 야외 돌징검다리, 콘크리트 블록 위 보행 | stone 간격 ↑, stone 크기 ↓ |
| 2 | `parkour_balance_beam` | 좁은 통로 직진, 측면 발 떨어짐 회피 | step의 corridor와 다름 — **양 측 fall-off** 위험 | 파이프/난간/보 위 이동 (Barkour weave 변형) | beam 폭 ↓, beam 길이 ↑ |
| 3 | `parkour_slope` | pitch 보상, 마찰 적응, 가속/감속 균형 | stair의 이산 step과 다름 — **연속 pitch** 학습 | 경사로/언덕/완만한 옥상 | 경사 각도 ↑ (5°→25°) |
| 4 | `parkour_zigzag_hurdles` | heading 변경, 측면 회피, command 추종 vs 회피 trade-off | hurdle은 직진 점프뿐 — **lateral steering** 추가 | 도시 장애물(자전거/물건) 회피 | y-offset 분산 ↑, 간격 ↓ |
| 5 | `parkour_rough_blocks` | 다지점 비균일 접촉, foot edge 회피, COM 흔들림 보상 | step/stair는 정렬된 box — **불규칙 격자** 필요 | 자갈/벽돌/잔디 위 비정형 보행 | block 높이 분산 ↑, 밀도 ↑ |

**선정 원칙**:
- **각 후보는 새 스킬 1개 이상**을 학습시킴 (단순 hurdle 변형 아님)
- **trimesh box primitive 합성만으로 구현 가능** — 기존 IsaacLab 컨벤션 유지
- **height_scan 187 차원**이 의미 있게 변하는 지형 (즉 perception 학습 신호 제공)
- **goal sequence**가 자연스럽게 정의 가능 — `PARKOUR_GOALS_REGISTRY` 패턴과 호환
- **sim-to-real 우선** — 학술 벤치마크(Barkour, ANYmal Parkour) 또는 야외 환경에 매핑

---

## B. 각 지형의 mesh 구성

모든 함수는 `parkour_terrains.py` 또는 `mesh_terrains.py`에 추가하며, 시그니처:
```python
def parkour_<name>_terrain(difficulty: float, cfg: MeshParkour<Name>TerrainCfg)
    -> tuple[list[trimesh.Trimesh], np.ndarray]
```
공통 패턴: `make_plane(cfg.size, height=0.0)` 으로 ground 깔고, `trimesh.creation.box(dim, translation_matrix)` 합성 → `goals_list` 작성 → `PARKOUR_GOALS_REGISTRY.append((goals, origin))`.

### B1. parkour_stepping_stones

| 파라미터 | 기본/범위 | 의미 |
|---------|----------|------|
| `platform_length` | 2.5 | 시작 평지 길이 |
| `num_stones` | 8–10 | 디딤돌 개수 |
| `stone_size_xy_range` | (0.20, 0.40) m | 정사각 디딤돌 한 변 |
| `stone_height_range` | (0.05, 0.20) m | 위로 올라온 높이 |
| `gap_length_range` | (0.20, 0.45) m | 인접 stone 간 빈 공간 |
| `lateral_jitter_range` | (-0.30, 0.30) m | y-방향 무작위 오프셋 |
| `num_goals` | 8 | row 진행마다 1 goal |

Mesh: ground plane + stone box per step. **gap_terrain과 다름** — gap_terrain은 통째 platform 사이 gap, 이쪽은 점진적 비연속 + lateral jitter.

Difficulty 매핑:
- 0 → stone_size 0.40, gap 0.20, jitter 0.0 (거의 직진)
- 1 → stone_size 0.20, gap 0.45, jitter ±0.30 (정밀 zig-zag)

### B2. parkour_balance_beam

| 파라미터 | 기본/범위 | 의미 |
|---------|----------|------|
| `platform_length` | 2.5 | 시작 평지 |
| `beam_length` | 6.0–10.0 m | 외나무다리 총 길이 |
| `beam_width_range` | (0.20, 0.50) m | y-방향 폭 (difficulty↑에 따라 ↓) |
| `beam_height` | 0.15 | 지면 위 띄움 |
| `num_segments` | 1–3 | 직선 vs 분절 (분절마다 작은 y-shift) |
| `y_shift_per_segment_range` | (0.0, 0.4) m | 분절 사이 측면 shift |
| `num_goals` | 8 | 분절 입구 + 종단 |

Mesh: 시작/종단 platform + 길쭉한 beam box들 + (선택) 분절 사이 작은 디딤판.
- `box((seg_len, beam_w, beam_h))` 으로 각 분절 생성. fall-off 시 ground=0이라 즉시 떨어짐. termination은 termination_height(=0.1)로 자연 처리.

Difficulty 매핑:
- 0 → beam_w 0.50, segments=1, y_shift 0
- 1 → beam_w 0.20, segments=3, y_shift ±0.40

### B3. parkour_slope

| 파라미터 | 기본/범위 | 의미 |
|---------|----------|------|
| `platform_length` | 2.5 | 시작 평지 |
| `slope_angle_deg_range` | (5°, 25°) | 경사 각도 |
| `slope_length` | 4.0 m | 경사 길이 |
| `flat_top_length` | 1.5 m | 정상 평지 |
| `num_segments` | 1 ascend + 1 descend | 대칭 |

Mesh 방식: **계단 근사 (staircase approximation)** — `mesh_terrains.py:1043` `parkour_stair_terrain`이 axis-aligned box의 누적이라는 사실을 활용.
- 한 box 두께 0.05m × N개 stack → 평균 경사 angle 근사. 경사 각도 = `atan(step_h / step_w)`.
- 매우 매끄러운 slope가 필요하면 trimesh `creation.extrude_polygon` 으로 **wedge** 생성도 가능하나, **face rasterization (parkour_terrains.py L210-260)이 axis-aligned에서만 정확** → wedge는 edge_mask가 conservative over-estimate 발생. **권장: 0.05m step staircase 근사** (논문 ANYmal slope class도 동일 처리).

Risk: slope를 step 누적으로 흉내내면 발에 미세 edge 페널티가 발생할 수 있음 → `feet_edge` reward는 `terrain_levels > 3`에서만 동작하므로 초기 curriculum 영향은 작지만, level 후반 페널티 누적 가능. → reward 조건 분기 필요(D 섹션).

Difficulty 매핑:
- 0 → 5° (step_h=0.05, step_w=0.57, 즉 평지 거의)
- 1 → 25° (step_h=0.05, step_w=0.107, 가파른 경사)

### B4. parkour_zigzag_hurdles

| 파라미터 | 기본/범위 | 의미 |
|---------|----------|------|
| `platform_length` | 2.5 | 시작 평지 |
| `num_hurdles` | 6–8 | hurdle 개수 |
| `hurdle_thickness` | 0.30 | x-두께 |
| `hurdle_height_range` | (0.10, 0.25) | y-방향 폭 0.6m, 통과 corridor 0.6m |
| `corridor_y_offset_pattern` | ±0.6 alternating | hurdle별 좌/우 교차 |
| `x_spacing_range` | (1.5, 2.4) | x 간격 |

Mesh: 기존 `parkour_hurdle_terrain` (mesh_terrains.py:966) 의 변형으로 시작. 차이:
- **hurdle 통과 corridor가 좌↔우 교대로 배치**: hurdle k는 corridor 좌측(y_low), hurdle k+1은 우측(y_high). 즉 좌측만 막힌 hurdle / 우측만 막힌 hurdle 교대.
- 좌측 막힘: `box(thickness, y in [mid_y, size_y], hurdle_h)` — 좌측 비움.
- 우측 막힘: `box(thickness, y in [0, mid_y], hurdle_h)` — 우측 비움.
- Goal은 hurdle 직후 통과 corridor 중앙에 배치 → robot은 lateral 이동 명령 없이도 visual goal에 끌려 좌↔우 swing.

Difficulty 매핑:
- 0 → hurdle_h 0.10, x_spacing 2.4 (충분한 회피 시간)
- 1 → hurdle_h 0.25, x_spacing 1.5 (빠른 lateral 전환)

### B5. parkour_rough_blocks

| 파라미터 | 기본/범위 | 의미 |
|---------|----------|------|
| `platform_length` | 2.5 | 시작/종단 평지 |
| `rough_section_length` | 8.0–12.0 m | 요철 구간 길이 |
| `block_size_range` | (0.20, 0.40) m | block xy 크기 |
| `block_height_range` | (0.0, 0.10) m | difficulty 따라 0–0.10 m |
| `block_density` | 0.6–0.9 | block grid 채움 비율 |
| `num_goals` | 8 | rough 구간 통과 중간 지점들 |

Mesh: ground plane + N×M grid 위에 무작위 sparsity로 box 배치. 각 block은 `box((bs, bs, bh), translate(xc, yc, bh/2))`. 인접 block 높이는 독립 sample → 요철 패턴.

Difficulty 매핑:
- 0 → bh 0.0–0.02, density 0.6
- 1 → bh 0.0–0.10, density 0.9

---

## C. Goal 설계

`PARKOUR_GOALS_REGISTRY.append((goals[N,3], origin[3]))` 패턴 유지. `num_goals=8` 고정 (현재 `MeshParkour*TerrainCfg.num_goals` 기본값). 부족하면 마지막 goal 복제, 초과하면 truncate (기존 패턴, mesh_terrains.py L1031-1035).

| 후보 | Goal 배치 |
|------|----------|
| stepping_stones | 매 stone 중심 위 (`x_stone, y_stone, stone_h`) |
| balance_beam | 분절마다 beam 중앙 + 종단 platform 중앙 |
| slope | 정상 평지 중앙 + 종단 평지 중앙 (ascend/descend 분절) |
| zigzag_hurdles | hurdle 직후 통과 corridor 중앙 (좌→우 교대) — heading 자연 유도 |
| rough_blocks | rough 구간 1/4, 1/2, 3/4, 종단 등 일정 간격 |

각 goal은 (x,y,z) 3D — z는 robot 발 닿는 surface 높이 (stone_h, beam_h, slope_top_h, 0, block_h_max 등). `goal_z=0.3`은 deprecated이므로 직접 surface z 사용.

---

## D. Observation/Reward 영향

### D1. height_scan (1.6×1.0 m grid, 0.1 m resolution → 187 cell)
- stepping_stones: ✓ 의미 있는 신호. stone vs gap 패턴이 187 cell에 나타남. 기존 raycaster 그대로 OK (천장 없음).
- balance_beam: ✓ 좁은 beam의 falloff edge가 scan에 보임. 단, beam이 매우 좁으면 (0.20m) y-방향 cell(11개) 중 1–2개만 잡힘 → 정책이 "한 점" 인식 어려움 → 이 자체가 학습 challenge.
- slope: ✓ 경사면 → scan 값이 monotonic gradient. 기존과 호환.
- zigzag_hurdles: ✓ hurdle 잔존 + 통과 corridor 분포. 기존 hurdle reward 파이프라인 재활용 가능.
- rough_blocks: ✓ 요철 패턴이 scan에 분명. **단**: face rasterization은 axis-aligned에서 정확 → block의 max-z만 잡혀 OK.

**천장 없음** — 5종 모두 bottom-up. crawl처럼 height_scan 마스킹 로직 불필요.

### D2. edge_mask (Laplacian 기반, parkour_terrains.py L142-304)
- stepping_stones: ✓ 각 stone 둘레에 edge 검출 → `feet_edge` reward (스케일 -1.0) 정확히 동작. stone에 발 떨어지는 거 자체를 페널티.
- balance_beam: ✓ beam 양 옆이 edge → fall-off 직전 페널티. 학습 신호 강력.
- slope: **⚠ 주의** — staircase 근사 시 매 step이 edge로 잡힘. 즉 slope 전체가 edge 셀로 가득 → `feet_edge` reward가 슬로프 상에서 매 스텝마다 페널티. **해결책 (D3-slope-feet-edge)**: 신규 `terrain_class=5` (slope) 도입 → reward에서 `is_slope` 마스크로 `feet_edge *= 0` 적용. 또는 staircase step_h를 0.025 m로 더 잘게 → laplacian threshold 20.0 미만으로 클램프되도록 step 차이 약화.
- zigzag_hurdles: 기존 hurdle과 동일 처리.
- rough_blocks: ⚠ 모든 block 가장자리가 edge → `feet_edge` 매 step 페널티. **해결책**: terrain_class=6 (rough_blocks) → `feet_edge` 가중치 ↓ (예: -0.2). 또는 block height가 작으므로 edge_mask laplacian threshold를 키워 작은 단차는 무시.

### D3. terrain_class 매핑 (parkour_env_cfg.py L130-134, parkour_env.py L601-602)

기존:
```python
TERRAIN_CLASS_FLAT = 0; TERRAIN_CLASS_HURDLE = 1; TERRAIN_CLASS_STEP = 2;
TERRAIN_CLASS_GAP = 3; TERRAIN_CLASS_STAIR = 4
```
신규 추가 (insertion-order 기반이므로 dict에 추가하는 순서대로 자동 부여 — 명시 상수 추가 필수):
```python
TERRAIN_CLASS_STEPPING_STONES = 5
TERRAIN_CLASS_BALANCE_BEAM    = 6
TERRAIN_CLASS_SLOPE           = 7
TERRAIN_CLASS_ZIGZAG          = 8
TERRAIN_CLASS_ROUGH_BLOCKS    = 9
```

reward 분기 (parkour_env.py L599-645) 확장 제안:
| reward 항 | flat | hurdle/zigzag | step/stair | gap | stones | beam | slope | rough |
|----------|------|---------------|------------|-----|--------|------|-------|-------|
| `lin_vel_z_l2` 가중 | 1.0 | 0.1 | 0.1 | 0.1 | 0.1 | 0.5 | 0.1 | 0.3 |
| `orientation_l2` | on | off | off | off | off | **on** (균형 필요) | partial(0.3) | off |
| `feet_edge` | normal | normal | normal | normal | **boost ×2** | **boost ×3** | **off (slope)** | reduce ×0.3 |
| `dof_error_l2` x10 | yes (flat-only) | no | no | no | no | no | no | no |

표는 plan 권장이며 reward-worker가 실제 튜닝 시 ablation 권장.

### D4. 신규 reward 후보 (선택 사항)

- `lateral_balance_l2` (beam, stones): root y 중심선 이탈 페널티. 단순 `(root_pos_y_local)^2` 형태.
- `slope_climb_bonus`: slope 상에서 forward 진행 시 보너스 (이미 `tracking_goal_vel`이 부분 커버하므로 우선순위 낮음).

신규 reward 도입은 **권장하지 않음** — 기존 21개 항으로 충분히 cover. Loss 시그널 분리 학습 부담 ↑.

---

## E. Curriculum 설계

`PARKOUR_TERRAINS_CFG.curriculum=True` 유지, `num_rows=11`. row 0 = easy, row 10 = hard. 각 신규 cfg 의 `*_range` tuple은 difficulty=0/1에서 평균 보간되어 row별로 점진 변화.

| 후보 | difficulty 0 | difficulty 1 | 핵심 보간 변수 |
|------|-------------|-------------|---------------|
| stepping_stones | size 0.40, gap 0.20, jitter 0 | size 0.20, gap 0.45, jitter 0.30 | size, gap, jitter |
| balance_beam | width 0.50, segs 1 | width 0.20, segs 3, shift 0.4 | width 주도 |
| slope | 5° | 25° | angle |
| zigzag | h 0.10, spacing 2.4 | h 0.25, spacing 1.5 | h, spacing |
| rough_blocks | bh max 0.02, density 0.6 | bh max 0.10, density 0.9 | bh, density |

ParkourEnvCfg `max_init_terrain_level=3` 유지 — 신규 지형도 동일 (low-level 부터 시작 권장).

**경고**: `flat_patch_sampling` 패치 (start spawn) 가 신규 지형에서 항상 평지를 찾을 수 있도록 `platform_length≥2.0`, `patch_radius=0.5`, `max_height_diff=0.05` 유지. balance_beam은 시작 platform 명시, slope는 시작 평지 명시 → 모두 OK.

---

## F. 기존 mix와의 균형 (proportion 재배분)

현재 (parkour_env_cfg.py L43-122):
| flat | hurdle | step | gap | stair | sum |
|------|--------|------|-----|-------|-----|
| 0.10 | 0.20   | 0.20 | 0.30| 0.20  | 1.00|

5종 추가 후 권장 분배 — **반드시 sum=1.0**:

### Phase A (top-1, top-2만 추가 — 안전 도입)
| flat | hurdle | step | gap | stair | stones | beam | sum |
|------|--------|------|-----|-------|--------|------|-----|
| 0.08 | 0.18   | 0.18 | 0.24| 0.18  | 0.08   | 0.06 | 1.00|

### Phase B (5종 모두 추가 — 풀 통합)
| flat | hurdle | step | gap | stair | stones | beam | slope | zigzag | rough | sum |
|------|--------|------|-----|-------|--------|------|-------|--------|-------|-----|
| 0.06 | 0.14   | 0.14 | 0.18| 0.14  | 0.08   | 0.06 | 0.08  | 0.06   | 0.06  | 1.00|

**원칙**:
- 기존 stair/step/hurdle/gap 합 0.90 → 0.60으로 감소 (각 ~30% 축소).
- gap (0.30 → 0.18)이 가장 높은 비중 유지 — 점프 스킬 핵심.
- stones, slope에 가장 많은 비중 (0.08) — 신규 핵심 스킬.
- flat은 0.06까지 축소 (orientation_l2 조건부가 학습 신호 의존하므로 0 금지).

**학습 안정성 리스크**: 신규 5종 추가 시 `num_cols=40` 고정이라면 col 당 sub-terrain 1개 할당 → 작은 proportion (0.06)은 cols≈2.4개 → reasonable. proportion×40 < 1.0이 되면 generator가 무시할 수 있으므로 **신규 proportion ≥ 0.05 유지**.

---

## G. 우선순위 매트릭스

각 항목 1(낮음)–5(높음). **권장 순위 = 학습가치×2 + sim2real - 구현난이도 - 시스템변경**.

| 후보 | 학습 가치 | 구현 난이도 | 시스템 변경 | sim-to-real | **종합 점수** | 권장도 |
|------|-----------|-------------|-------------|-------------|---------------|--------|
| 1. stepping_stones | 5 | 2 | 1 | 5 | **5×2+5−2−1 = 12** | **★★★★★ Top-1** |
| 2. balance_beam | 4 | 2 | 1 | 4 | **4×2+4−2−1 = 9** | ★★★★ |
| 3. slope | 4 | 3 | 3 (feet_edge 분기 필요) | 5 | **4×2+5−3−3 = 7** | ★★★ |
| 4. zigzag_hurdles | 3 | 2 (hurdle 변형) | 1 | 3 | **3×2+3−2−1 = 6** | ★★★ |
| 5. rough_blocks | 3 | 2 | 2 (edge_mask thr 조정) | 4 | **3×2+4−2−2 = 6** | ★★★ |

**Top-1**: `parkour_stepping_stones`
- 가장 높은 학습 가치(정밀 발 배치는 기존 5종이 못 가르침), 가장 낮은 시스템 변경(기존 reward 그대로 활용 가능), 가장 강한 sim-to-real (Barkour weave/야외 디딤돌과 직접 매핑).

**Top-2**: `parkour_balance_beam`
- 측면 균형은 새 스킬이며 trimesh 박스 1–3개로 구현 끝. 시스템 변경 거의 없음.

**Top-3**: `parkour_slope`
- 학습 가치 높지만 staircase 근사로 인한 `feet_edge` 페널티 충돌이 잠재 risk. terrain_class 분기로 해결 가능하지만 reward-worker 작업 1건 추가.

---

## H. 상세 phase 체크리스트

### H1. parkour_stepping_stones (Top-1) — 풀 phase

#### Phase 1: cfg + 함수 정의
- [ ] `mesh_terrains_cfg.py` 에 `MeshParkourSteppingStonesTerrainCfg` 추가 (B1 표 모든 필드)
- [ ] `mesh_terrains.py` 또는 `parkour_terrains.py` 에 `parkour_stepping_stones_terrain(difficulty, cfg)` 작성
- [ ] 함수 내부:
  1. `make_plane(cfg.size, height=0.0)` ground
  2. start platform box (`platform_length × size_y × 0.0`은 ground 평면이라 mesh 추가 불필요, 단 spawn flat patch 영역 보장)
  3. 루프 `for k in range(num_stones)`: stone 위치 = (`platform_length + cumsum(gap+size) + lateral_jitter`), `box(size, size, height)` 합성, goal 추가
  4. `goals = pad_or_truncate(goals_list, num_goals)`
  5. `PARKOUR_GOALS_REGISTRY.append((goals, origin))`

#### Phase 2: env cfg 등록
- [ ] `parkour_env_cfg.py` `sub_terrains` dict에 `"parkour_stepping_stones": MeshParkourSteppingStonesTerrainCfg(...)` 추가
- [ ] proportion 재분배 (Phase A 표)
- [ ] `TERRAIN_CLASS_STEPPING_STONES = 5` 상수 추가

#### Phase 3: env 로직 검증
- [ ] `parkour_env.py:601` `is_flat`/`is_non_flat` 분기에 stones 추가 영향 확인 (현재 분기 로직은 flat vs non-flat 이분이라 수정 불필요)
- [ ] `feet_edge` 잘 동작하는지 시각 확인 (`debug_vis_edge_mask=True` 활성화)
- [ ] height_scan 187 dim 정상 (정책 obs shape 일치 → validate-code 호출)

#### Phase 4: 학습 검증
- [ ] num_envs=512 단축 학습 200 iter
- [ ] tracking_goal_vel 평균 ≥ 0.5 도달 확인
- [ ] feet_edge 페널티가 첫 100 iter 내 감소
- [ ] termination ratio > 50% 시 size_range 완화

#### Phase 5: 통합 및 회귀
- [ ] 기존 5종 + stones 풀 셋으로 1500 iter 학습
- [ ] 기존 5종 success rate 회귀 < 5%

#### Phase 6: 시각화 및 문서
- [ ] play.py 로 실제 stone 도약 영상 캡처
- [ ] CLAUDE.md 또는 README 업데이트

### H2. parkour_balance_beam (Top-2) — 풀 phase

#### Phase 1: cfg + 함수
- [ ] `MeshParkourBalanceBeamTerrainCfg` 추가 (B2 필드)
- [ ] `parkour_balance_beam_terrain` 작성:
  1. ground plane
  2. start platform box
  3. 루프 segments: 각 분절 = `box((seg_len, beam_w, beam_h))`, 분절 사이 작은 transition stone 또는 직접 연결
  4. end platform box
  5. goals = 분절 입구 + 종단

#### Phase 2: env cfg 등록
- [ ] `TERRAIN_CLASS_BALANCE_BEAM = 6`
- [ ] proportion 재분배 (Phase A 표)

#### Phase 3: reward 분기
- [ ] reward-worker에 의뢰: `is_beam` 마스크 추가, `orientation_l2` 활성, `feet_edge` ×3 가중 (D3 표)

#### Phase 4: 학습 검증
- [ ] beam fall-off 비율을 episode 종료 ratio로 추적

#### Phase 5–6: H1과 동일 패턴

### H3. parkour_slope — 약식 outline

1. cfg `MeshParkourSlopeTerrainCfg(slope_angle_deg_range, slope_length, flat_top_length)` 추가
2. 함수: ascend stair (step_h=0.05, step_w 보간) + flat top + descend stair
3. `TERRAIN_CLASS_SLOPE = 7` 추가
4. **reward worker 의뢰 필수**: `feet_edge *= 0` for `is_slope`, `orientation_l2 *= 0.3` partial
5. edge_mask laplacian threshold tuning 검토 (parkour_terrains.py L148, default 20.0; slope에서 작은 step의 edge가 잡힐지 실험)
6. 학습 200 iter sanity 후 풀 통합

### H4. parkour_zigzag_hurdles — 약식 outline

1. cfg `MeshParkourZigzagHurdleTerrainCfg(corridor_y_offset_pattern, ...)` — 기존 hurdle cfg 상속 또는 복제
2. 함수: 기존 `parkour_hurdle_terrain` 변형 — corridor를 alternating pattern으로
3. `TERRAIN_CLASS_ZIGZAG = 8` (기존 hurdle reward 분기 그대로 적용 가능, terrain_class만 다름)
4. proportion 0.06
5. command은 forward only(0.3–1.0)이지만 lateral 이동은 hurdle 회피 자체로 학습 → goal lateral position이 자연 유도

### H5. parkour_rough_blocks — 약식 outline

1. cfg `MeshParkourRoughBlocksTerrainCfg(rough_section_length, block_size_range, block_height_range, block_density)`
2. 함수: ground + 루프 grid 위 random block 합성 + start/end platform
3. `TERRAIN_CLASS_ROUGH_BLOCKS = 9`
4. **reward worker 의뢰**: `feet_edge ×0.3`, edge_mask threshold 검토
5. proportion 0.06

---

## I. Risks & Open Questions

### I1. 시스템 영향 risk

| Risk | 영향 | 완화 |
|------|------|------|
| edge_mask false positive (slope, rough_blocks) | feet_edge 페널티 폭주 → 정책 학습 정체 | terrain_class 분기로 reward off / threshold 조정 |
| stepping_stones 의 lateral_jitter가 spawn flat patch와 충돌 | spawn 실패 → env 시작 못 함 | `platform_length=2.5` 보장, FlatPatchSampling은 시작 platform 영역에만 적용 |
| balance_beam 폭 0.20 m이 RayCaster grid resolution 0.1m와 비슷 → scan 에 1–2 cell만 잡힘 | 정책이 beam을 시각적으로 trace 못함 | 첫 row(난이도 0)는 폭 0.50으로 시작 → curriculum이 점진 narrow |
| proportion 재분배가 학습 안정성 저하 | 초반 학습 슬로우 다운 | Phase A (top-2만 도입) → Phase B (5종 풀) 의 점진 도입 |
| terrain_class 5–9 추가 후 reward 분기 누락 | 어떤 신규 지형에서 일관 행동 학습 안 됨 | 본 plan D3 표 reward-worker에 명시 전달 |
| `num_cols=40`, neue proportion < 1/40 = 0.025 | generator가 무시 | 본 plan에서 모든 신규 proportion ≥ 0.06 |

### I2. 학습 가설 risk

| Risk | 영향 | 완화 |
|------|------|------|
| height_scan(187)이 stones/rough의 미세 변화 다 못 잡음 (resolution 0.1m, stone 0.2m) | foot placement 정밀도 한계 | 차후 obs-worker가 height_scan resolution을 0.05m로 ↑ (단 obs dim이 187→748로 증가, network update 필요) |
| AMP / 모방학습 reference data가 신규 스킬(beam, slope) 포함 안 됨 | discriminator로부터 학습 신호 없음 → policy가 "회피" 학습 | reference data extension 별도 작업; 본 plan 범위 밖 |
| zigzag_hurdles 의 lateral 이동이 `lin_vel_y_range=[0,0]` 명령과 충돌 | 정책이 forward 명령 추종 vs 회피 trade-off에서 stuck | command_cfg lateral 허용 또는 zigzag 전용 lateral random sample → cfg-worker 작업 |
| slope를 staircase 근사하면 실제 robot policy가 micro-step 패턴을 학습 → real slope에 sim-to-real gap | sim-to-real 성능 저하 | step_h ↓ (0.025) 로 step 차이 작게, friction range 도메인 랜덤화 강화 |
| 5종 풀 통합 시 학습 시간 증가 (현재 1500 iter → 추정 2000+ iter 필요) | 시간 / GPU 자원 | 우선 Top-2만 도입 → Phase B는 후속 |

### I3. Open Questions (의사 결정 필요)

1. **slope 구현 방식**: staircase 근사 vs 진짜 wedge primitive (trimesh.creation.extrude_polygon)? 후자는 정확하지만 face rasterization edge_mask가 conservative하게 잡혀 부작용 위험. → **권장: staircase 근사 + step_h=0.025 m**.
2. **zigzag_hurdles는 별도 cfg 클래스 필요한가?** 기존 `MeshParkourHurdleTerrainCfg`에 `corridor_y_offset_pattern: str = "fixed"|"alternating"` 필드만 추가하는 안도 있음. → **권장: cfg 상속(extension), 함수만 분리**.
3. **rough_blocks의 edge_mask threshold tuning**: 현재 laplacian_threshold=20.0(uint8). 작은 block(<0.1m)에서도 잡힐지? 첫 build 후 visual 검증 필요. → **Phase 3에서 sanity check**.
4. **reward 분기는 reward-worker가 한 번에 처리 vs 후보별 단독 PR?** → **단독 PR 권장** (debug 시 회귀 추적 용이).
5. **`num_cols=40` 유지 vs 50으로 증가?** 5종 추가 시 col 부족 가능성 — 현재 5종 합 1.0 / 40 col → cell당 8 col 평균. 10종이면 cell당 4 col. → **유지 가능, 단 모니터링**.
6. **AMP reference 확장 여부**: 본 plan 범위 밖이지만, balance_beam/slope에 AMP가 적용 안 되면 학습 신호 약함. → **순수 PPO만 우선 검증, AMP 통합은 향후 task**.
7. **신규 reward 항(`lateral_balance_l2`) 도입 vs 기존 21개 항 reuse?** → **기존 reuse 권장** (학습 안정성, 디버깅 용이).

### I4. Verify-가능 항목 (critic 단계 체크리스트)

- [ ] 5개 후보 각각 trimesh box composition으로 구현 가능 (✓ B 섹션 모든 mesh 가 box 합성)
- [ ] `PARKOUR_GOALS_REGISTRY` 패턴 호환 (✓ C 섹션)
- [ ] proportion 합 = 1.0 (✓ F 섹션 두 phase 모두)
- [ ] 신규 terrain_class 5–9 매핑 명시 (✓ D3)
- [ ] reward 분기 영향 표 제시 (✓ D3 8열 표)
- [ ] curriculum difficulty 0/1 양 끝값 명시 (✓ E 섹션)
- [ ] 우선순위 매트릭스 정량 점수 (✓ G 섹션)
- [ ] Top-1, Top-2 phase 체크리스트 (✓ H1, H2)
- [ ] Risk 표 + open questions (✓ I 섹션)

---

## J. Sources

- [ANYmal Parkour: Learning Agile Navigation for Quadrupedal Robots (Hoeller et al., Science Robotics 2024)](https://www.science.org/doi/10.1126/scirobotics.adi7566) — climb/jump/crouch skill set
- [ANYmal Parkour arXiv (2306.14874)](https://arxiv.org/html/2306.14874v1) — slope/stepping stones train pipeline
- [Barkour: Benchmarking Animal-level Agility (Caluwaerts et al., Google DeepMind 2023)](https://research.google/blog/barkour-benchmarking-animal-level-agility-with-quadruped-robots/) — weave poles, A-frame, broad jump
- [Barkour benchmark IEEE Spectrum coverage](https://spectrum.ieee.org/quadruped-robot-benchmark-barkour) — 0.5m broad jump as agility metric
- [Acrobotics: A Generalist Approach to Quadrupedal Robots' Parkour (arXiv 2509.02727)](https://arxiv.org/html/2509.02727) — multi-skill distillation
- [Parkour in the Wild (arXiv 2505.11164)](https://arxiv.org/abs/2505.11164) — terrain diversity for sim-to-real
- 코드 근거: `source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py:890-1203`, `mesh_terrains_cfg.py:323-483`, `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_terrains.py:32-304`, `parkour_env_cfg.py:33-134`, `parkour_env.py:580-779`

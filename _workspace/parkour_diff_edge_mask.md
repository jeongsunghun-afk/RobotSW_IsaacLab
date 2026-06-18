# Edge-Mask 비교: A(Direct Parkour) vs B(Isaaclab_Parkour)

**작성일**: 2026-05-20
**범위**: `feet_edge` reward에서 사용하는 edge mask의 생성·저장·전달·사용 방식 비교
**상태**: 코드 검증 기반 (worker dispatch). 실험 검증 전.

## 0. 결론 한 줄

**구조는 같으나 알고리즘이 다르다(b).** A=global trimesh + Sobel x∪y(양축 검출), B=per-tile heightfield + `move_x≠0` only(x축만) + x-축 dilation. 같은 cell 크기·같은 reward 식·같은 gate(`terrain_level>3`)이지만 검출 방식·인덱스 변환·격자 크기가 달라 의미 있는 차이가 존재.

## 1. 저장 표현

| 항목 | A | B |
|------|---|---|
| dtype | `bool` (device tensor) | `np.int16` (bool처럼 쓰임) |
| shape | **글로벌 2D** `(n_x, n_y)` | **per-tile 4D** `(num_rows, num_cols, w_px, l_px)` → 첫 reward 호출 시 글로벌 2D로 flatten/permute |
| anchor | `self._edge_mask_origin` = world LL corner | `rows_offset/cols_offset` 재계산 (동등) |
| cell size | `horizontal_scale=0.05` | `horizontal_scale=0.05` |
| per-tile pixel | `n_x = round(total/h_scale)` | `w_px = int(size/h_scale) + 1` **(+1 off-by-one)** |

- A: `parkour_env.py:484-488`, B: `parkour_terrain_generator.py:11-22`, B reshape: `rewards.py:32-35`

## 2. 생성 시점

| | A | B |
|---|---|---|
| 단계 | **terrain build 직후 1회** (`_setup_scene` → `_build_edge_mask`) | **각 sub-tile 생성 함수가 호출될 때 inline 1회** (`parkour_field_to_mesh` 데코레이터) |
| reset마다 갱신 | 안 함 | 안 함 |

- A: `parkour_env.py:361,428-501` / B: `parkour_terrain_generator.py:45,79`, `utils.py:40-45`

## 3. 생성 알고리즘 (핵심 차이)

### A — `compute_edge_mask_from_terrain_mesh` (`parkour_terrains.py:159-309`)

1. 조립된 trimesh를 `_CapturingTerrainImporter`로 캡처
2. **GPU WARP raycast** — 각 cell에 위→아래 수직 ray 1개. hit z → heightfield
3. uint8 정규화 + `cv2.GaussianBlur(5,5)`
4. **Sobel x ∪ Sobel y** (threshold 30 each, OR 결합) → bidirectional edge
5. `(hf_max - hf_min) < 1e-3` 평지면 all-False (조기 종료)

### B — `parkour_field_to_mesh` 데코레이터 + `convert_height_field_to_mesh` (`utils.py:14-115`)

1. sub-terrain은 raw heightfield(int16) 반환
2. `convert_height_field_to_mesh`는 `slope_threshold` 넘는 수직면 vertex를 x/y로 이동시켜 mesh 생성 → **`move_x != 0`을 mask로 반환** (line 115)
3. `move_y`는 계산되어 vertex 이동에 쓰이지만 **mask로 반환되지 않음** → **x-edges만 검출**
4. `binary_dilation` 으로 x축 방향만 `edge_width_thresh/h_scale` (default ±1 cell) 두께 확장

### 의미 있는 알고리즘 차이

| 항목 | A | B |
|------|---|---|
| mesh source | concatenated trimesh (모든 primitive 지원) | per-tile heightfield only |
| edge 정의 | Sobel x **OR** y > 30 (양축) | `move_x ≠ 0` 후 x-dilation (x축만) |
| 대각/y-only edge | 검출됨 | **검출 안 됨 (설계상)** |
| dilation | Sobel 자연 폭(~1 cell) | x축 ±1 cell 명시적 |

> A의 변수명 `x_edge_mask`는 extreme_parkour에서 가져온 잔재 — 실제 알고리즘은 양축 검출. B의 `x_edge_mask`는 글자 그대로 x-only.

## 4. terrain → env 전달

### A
- `_CapturingTerrainImporter` 가 `import_mesh` 후크로 `_captured_trimesh` 저장 (`parkour_env.py:58-90`)
- `_build_edge_mask`가 env attrs에 write:
  - `self.x_edge_mask` (n_x, n_y) bool
  - `self._edge_mask_origin` (2,) world LL
  - `self._edge_mask_inv_scale` = 1/h_scale
  - `self._edge_mask_height_field` (height_scan 용도와 공유)

### B
- `ParkourTerrainGenerator.__init__`이 `self.x_edge_maskes` 4D 사전 할당 (`parkour_terrain_generator.py:20`)
- 각 sub-tile 함수가 `(meshes, origin, goals, goal_heights, x_edge_mask)` 반환 (`utils.py:57`)
- generator가 per-tile slice write (`parkour_terrain_generator.py:50,85`)
- reward term이 첫 호출 시 generator의 4D를 가져와 global 2D로 stitch (`rewards.py:32-35`)

## 5. world → grid 인덱스 변환 (reward use)

### A (`parkour_env.py:1064-1079`)
```
feet_grid_xy = ((feet_xy - origin) * inv_scale - 0.5).round().long()  # = floor
feet_grid_xy.clamp_(0, shape-1)
feet_at_edge = x_edge_mask[gx, gy]
contact_filt = (||F_now||>2) | last_contact   # 1-step-ago
feet_edge    = (terrain_level>3) * sum(contact_filt & feet_at_edge, -1)
```

### B (`rewards.py:44-57`)
```
feet_pos_x = ((pos_w_x + rows_offset) / h_scale).round().long()   # = nearest
feet_pos_y = ((pos_w_y + cols_offset) / h_scale).round().long()
feet_pos_*.clip_(0, shape-1)
feet_at_edge = x_edge_masks_tensor[fx, fy]
contact_filt = (||F_now||>2) | (||F_hist[-1]||>2)   # history[-1] = oldest
rew          = (terrain_level>3) * sum(contact_filt & feet_at_edge, -1)
```

### 변환 차이

| 항목 | A | B | 영향 |
|------|---|---|------|
| 인덱스 함수 | `round(x - 0.5)` = floor | `round(x)` = nearest | 셀 경계가 ½ cell(=2.5 cm) 어긋남 — 경계 근처 발에서 인덱스 1개 차이 가능 |
| last-contact 출처 | `self._last_contacts` (직전 step) | `net_forces_w_history[:, -1]` (버퍼 가장 오래된 샘플) | 시간 커널 다름. 둘 다 두-샘플 OR이지만 latency 다름 |
| body indexing | `_feet_ids` 직접 | `asset_cfg.body_ids` (cfg에서 feet) | 효과 동일 |
| gate | `terrain_level > 3` | `terrain_level > 3` | **동일** |

## 6. A에만/B에만 있는 처리

| 차이 | 어디서 | 효과 |
|------|--------|------|
| **A: 양축 edge 검출** (Sobel x∪y) | `parkour_terrains.py:296-301` | 임의 mesh primitive(wedge, balance beam, zigzag)의 양축 edge 모두 페널티 |
| **B: x-edge only** (`move_x!=0`) | `utils.py:115` | y-edge/대각 edge 페널티 미적용. heightfield-only 가정 |
| **B: x-dilation `edge_width_thresh`** (default 0.05m → ±1 cell) | `utils.py:43-45` | B의 x-edge 페널티 밴드가 A보다 2 cell 더 두꺼움 |
| **B: per-tile `+1 pixel` padding** | `parkour_terrain_generator.py:15-16` | tile 경계마다 누적 → 먼 corner에서 feet-XY→mask 인덱스가 누적 drift. h_scale=0.05, 10×20 tile에서 **최대 ~0.5m drift** (잠재 off-by-N) |
| **A: 명시적 world LL origin 저장** | `parkour_env.py:466,486` | 비표준 centering에도 강건 |
| **B: cfg에서 offset 재계산** | `rewards.py:30-31` | TerrainGenerator centring 가정. centring 바뀌면 조용히 어긋남 |
| **A: trimesh 캡처 실패 시 all-False + 경고** | `parkour_env.py:447-454` | soft-fail (feet_edge=0) |
| **B: fallback 없음** | `rewards.py:32` | terrain_generator_class.x_edge_maskes 없으면 raise |
| **A: debug viz 훅** (`debug_vis_edge_mask*`) | `parkour_env_cfg.py:379-380` | 런타임 시각화 가능 |

## 7. 평지/장애물 없는 tile 처리

- **A**: 전체 평지면 조기 종료 all-False (`parkour_terrains.py:290`). 일부만 평지: Sobel=0 → 그 영역 False, obstacle tile만 True. 정상.
- **B**: heightfield 0 → `move_x=0` 전부 → tile mask all-False, dilation 후도 False. 정상.

A 현재 cfg는 `parkour_slope`(wedge primitive)를 포함 — WARP raycast가 wedge의 ramp를 부드럽게 잡아 Sobel이 정상 발화. B는 모든 sub-terrain이 heightfield-native라 wedge 같은 비축정렬 primitive가 없음 → 검출 누락이 발생하지 않음.

## 8. 학습 신호 관점 (가설, 단정 아님)

| 차이 | A가 더 페널티 받는 상황 | B가 더 페널티 받는 상황 |
|------|------------------------|------------------------|
| 양축 검출 vs x-only | wedge/balance-beam/zigzag의 y-방향 edge — A가 더 많은 cell 페널티 | — |
| x-dilation 유무 | — | 모든 x-edge 양옆 ±1 cell 추가 페널티 |
| 인덱스 floor vs nearest | 경계 근처 발 인덱스 1개 차이 | 동상 |
| **per-tile +1 padding (B만)** | — | 먼 corner tile에서 feet→mask drift 가능 — 잠재 off-by-N 인덱싱 결함 |

**확정 사실**: 1~7의 코드 인용은 모두 직접 확인됨.

**추측 (사용자 금지영역 준수 — root cause로 ranking 안 함)**: B의 per-tile `+1 pixel` 누적으로 인한 feet→mask drift는 실제 reward 발화 위치를 obstacle에서 점점 멀어지게 만드는 잠재 indexing 결함. 단 B는 학습이 잘 되는 환경이므로 이게 실제 학습을 막지는 않는 것으로 보임. 검증 필요 시 런타임에 `(feet_pos_x, feet_pos_y)` vs obstacle x-cell 비교 print 필요.

## 9. 결론

A와 B의 edge mask 생성·사용 파이프라인은 **구조적으로 비슷하지만 교환 불가능(non-exchangeable)**:
- A: mesh-raycast + Sobel (post-hoc, **양축 대칭**)
- B: heightfield-vertex-move-x (in-line, **x-only**, x-dilation)

현재 sub-terrain mix(A: wedge/zigzag 포함 / B: gap·hurdle·step heightfield)에서 **A의 알고리즘은 strictly 더 많은 case를 커버**한다. B의 알고리즘은 y-edge를 설계상 놓치지만, B의 terrain set이 x-aligned 장애물 위주라 실용상 충분하다.

reward 신호의 양적 차이는 있으나, A vs B "학습 가능성" 차이를 직접 설명하는 mechanism은 아니다 — A가 더 strict한 페널티(양축 + 평지 wedge)를 적용하는 방향이라 학습 어려움을 *증가*시킬 수 있는 정도. 단 §5의 §3.3에서 본 것처럼 A의 최신 run은 정상 학습하므로 단독 killer는 아님.

---
### 파일 경로 요약

**A**:
- `parkour_terrains.py:159-309` (생성), `parkour_env.py:58-90` (캡처), `:341-361` (배선), `:428-501` (build), `:1054-1079` (reward 사용), `parkour_env_cfg.py:379-380, 575` (debug+scale)

**B**:
- `parkour_isaaclab/terrains/utils.py:14-57, 61-115` (데코레이터+mask), `parkour_terrain_generator.py:11-22, 45, 79, 85` (4D 버퍼), `parkour_terrain_generator_cfg.py:11-19` (cfg), `parkour_terrain_importer.py:50-65`, `envs/mdp/rewards.py:19-64` (reward), `extreme_parkour/extreme_parkour_terrians.py:65-200+` (sub-terrain), `parkour_tasks/.../config/go2/parkour_mdp_cfg.py:148-149`

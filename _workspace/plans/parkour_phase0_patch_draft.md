# Parkour Phase 0 — Patch Draft (골격 패치 초안)

> **작성일**: 2026-05-07
> **목적**: 신규 6종 지형 추가 사전 골격 패치 초안. **실제 코드 변경 금지.**
> **원칙**: "즉시 적용(Patch A·C)"과 "Phase 1B 이후 적용(Patch B)"을 분리.

---

## §1. 코드 현황 요약

### 1-1. `_col_to_class` LUT (`parkour_env.py` L99–115)

LUT는 `sub_terrains.values()`에서 proportion을 추출해 누적합으로 class id를 동적 생성한다. **class 번호가 코드에 명시되지 않고 dict 삽입 순서에 전적으로 의존**하는 구조다.

```python
# L104 — proportion 추출 (dict insertion order 기반)
_props = [sc.proportion for sc in self.cfg.terrain.terrain_generator.sub_terrains.values()]
# L105–108 — sum=1.0 assert (현재 0.1+0.2+0.2+0.3+0.2=1.0, 통과)
assert abs(sum(_props) - 1.0) < 1e-6, (
    f"sub_terrain proportions must sum to 1.0 (got {sum(_props):.4f}); ..."
)
# L113–114 — LUT 생성: class_id = dict 내 위치 인덱스
self._col_to_class = (_col_vals.unsqueeze(1) >= _cumprops.unsqueeze(0)).sum(dim=1).long()
self._env_class = self._col_to_class[self._terrain_types]  # [num_envs]
```

**위험**: sub_terrains dict 중간에 신규 terrain을 삽입하면 class id가 밀려 reward 마스크가 오작동한다. Patch A가 이 규칙을 주석으로 명문화한다.

### 1-2. 기존 5종 패턴 (`parkour_env_cfg.py` L43–134)

각 entry는 `MeshParkour*TerrainCfg(proportion=…, num_goals=8, flat_patch_sampling={…})` 패턴. 현재 비율: flat=0.1, hurdle=0.2, step=0.2, gap=0.3, stair=0.2.

모듈 수준 상수 블록(L130–134)이 이미 존재한다:

```python
TERRAIN_CLASS_FLAT=0  TERRAIN_CLASS_HURDLE=1  TERRAIN_CLASS_STEP=2
TERRAIN_CLASS_GAP=3   TERRAIN_CLASS_STAIR=4
```

Patch C는 이 블록 바로 뒤에 신규 6개 상수를 append한다.

---

## §2. Patch A — 즉시 적용 (`parkour_env.py` L99 comment 교체)

동작 변화 없음. insertion-order 의존성과 append 규칙을 주석으로 명문화한다.

```diff
-        # Per-env terrain class index (0=flat, 1=hurdle, 2=step, 3=gap, 4=stair).
-        # LUT formula mirrors TerrainGenerator._generate_curriculum_terrain() exactly:
-        #   sub_index = min(where(col/num_cols + 0.001 < cumsum(proportions)))
-        # RAW cumsum (no normalization) — must match TG or reward masks diverge.
+        # Per-env terrain class index — canonical mapping (TERRAIN_CLASS_* in parkour_env_cfg.py):
+        #   0=flat, 1=hurdle, 2=step, 3=gap, 4=stair
+        #   [Phase 1B] 5=stepping_stones, 6=balance_beam, 7=crawl
+        #   [Phase 2 ] 8=slope_wedge, 9=zigzag_hurdles, 10=rough_blocks
+        # class_id == position in sub_terrains dict insertion order.
+        # IMPORTANT: always APPEND new terrains; never insert in the middle.
+        # LUT: class_id = (col/num_cols + 0.001 >= cumsum(proportions)).sum()
+        # RAW cumsum (no normalization) — must match TG or reward masks diverge.
```

`_props` ~ `self._env_class` 로직 라인은 변경 없이 유지. Phase 1B에서 신규 entry를 append하면 LUT가 자동으로 8-class로 확장된다.

---

## §3. Patch B — Phase 1B 진입 시 적용 (`parkour_env_cfg.py`)

> **전제 조건**: `parkour_terrains.py`에 `parkour_stepping_stones_terrain`, `parkour_balance_beam_terrain`, `parkour_crawl_terrain` 함수 및 각 `MeshParkour*TerrainCfg` 정의 완료 후에만 적용 가능. 미완성 상태 적용 시 **import error** 발생.

### Phase 1 proportion 변경 (8종 합=1.00)

기존 5종 proportion 변경 후 신규 3 entry를 append한다.

```
변경: flat 0.1→0.05, hurdle 0.2→0.15, step 0.2→0.15, gap 0.3→0.20, stair 0.2→0.15
추가: stepping_stones=0.10, balance_beam=0.10, crawl=0.10  (합 +0.30)
결과: 0.05+0.15+0.15+0.20+0.15+0.10+0.10+0.10 = 1.00 ✓
```

신규 3 entry는 동일 패턴(`num_goals=8, FlatPatchSamplingCfg(num_patches=2, …)`)으로 추가.

### 임시 옵션 (mesh 미완성 시 5종 비율만 조정)

신규 함수 없이 기존 5종 proportion만 먼저 조정하는 경우:
`flat=0.07, hurdle=0.22, step=0.22, gap=0.27, stair=0.22` (합=1.00). 학습 곡선에 영향이 있으므로 실험 외 목적에는 권장하지 않는다.

### Phase 2 proportion (Phase 2 진입 시 별도 patch)

11종 합=1.00: 기존 8종에서 비율 조정 + `slope_wedge=0.07, zigzag_hurdles=0.07, rough_blocks=0.06` 추가.

---

## §4. Patch C — 신규 terrain_class 상수 (`parkour_env_cfg.py` L134 이후)

```diff
  TERRAIN_CLASS_STAIR  = 4  # parkour_stair
+ # Phase 1 신규 (Phase 1B cfg 등록 후 reward 함수에서 참조)
+ TERRAIN_CLASS_STEPPING_STONES = 5  # parkour_stepping_stones
+ TERRAIN_CLASS_BALANCE_BEAM    = 6  # parkour_balance_beam (width 0.2–0.5m)
+ TERRAIN_CLASS_CRAWL           = 7  # parkour_crawl (low-ceiling corridor)
+ # Phase 2 신규
+ TERRAIN_CLASS_SLOPE_WEDGE     = 8  # parkour_slope_wedge (wedge primitive)
+ TERRAIN_CLASS_ZIGZAG_HURDLES  = 9  # parkour_zigzag_hurdles
+ TERRAIN_CLASS_ROUGH_BLOCKS    = 10 # parkour_rough_blocks
```

즉시 적용 가능. 선언만 추가하며 런타임 로직 변경 없음. Phase 1A에서 reward-worker의 `class_weight_table` 구성 시 이 상수를 참조한다.

---

## §5. Branch / PR 안

**권장 branch**: `feat/parkour-terrain-stones-beam-crawl`

| PR | 내용 | 머지 시점 |
|----|------|----------|
| PR-0 | Patch A + C | Phase 0 완료 즉시 (동작 변화 없음) |
| PR-1 | Phase 1A reward 재구조화 | Phase 1A 완료 후 |
| PR-2 | Phase 1B mesh + Patch B + Phase 1C obs 패치 | Phase 1B·C 완료 후 |
| PR-3 | Phase 2 mesh 3종 + proportion 11종 | Phase 2 완료 후 |

Patch B는 mesh 함수 없이 단독 머지 불가 (import error). PR-0은 로직 변경 없으므로 즉시 안전하다.

---

## §6. 검증 방법

**Patch A + C 적용 후**: `train.py --task Go2-Parkour-Direct-v0 --num_envs 256 --max_iterations 5 --headless`
확인: assertion error 없음 / import 오류 없음 / 5 iteration 정상 완주.

**Patch B 적용 후 (Phase 1B)**: `--max_iterations 50`. `print(self._col_to_class[:20])`으로 LUT 8-class 확인, episode_sums 로그에서 신규 terrain goal 생성 확인.

---

## §7. Risks & Open Items

| 항목 | 현황/위험 | 비고 |
|------|----------|------|
| proportion sum 불일치 | 없음 (현재·Phase 1·2 모두 1.00) | assert가 안전망 |
| class id 순서 불일치 | 없음 (dict 순서 = §2 번호표 일치) | Patch A 명문화 |
| Patch B 단독 머지 | import error 유발 | PR-2 구조로 방지 |
| import 라인 누락 | Phase 1B에서 신규 함수 3개 import 추가 필요 | cfg-worker가 Patch B 시 처리 |
| balance_beam height_scan 해상도 | scan 0.1m vs beam 폭 0.2m (S4) | Phase 1B obs-worker 담당; row 0 beam_width 0.5m 완화 |

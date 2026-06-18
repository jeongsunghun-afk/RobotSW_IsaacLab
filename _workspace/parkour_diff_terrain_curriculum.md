# Parkour A vs B — Terrain + Curriculum 비교

- **A** = `parkour` (Direct RL, 학습 초반 학습 안 됨)
  `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/`
- **B** = `Isaaclab_Parkour` (Manager-based, 학습 잘 됨, extreme_parkour 포팅 원본)
  `Isaaclab_Parkour/`

> 모든 수치는 파일:line 인용. 검증 안 된 추론은 "가설"로 명시.

---

## 1. Terrain Generator 전역 파라미터

| 항목 | A | B | 비고 |
|------|---|---|------|
| 정의 위치 | `parkour_env_cfg.py:44` `PARKOUR_TERRAINS_CFG` | `extreme_parkour/config/parkour.py:4` `EXTREME_PARKOUR_TERRAINS_CFG` | |
| Generator 타입 | `TerrainGeneratorCfg` (코어) | `ParkourTerrainGeneratorCfg` (커스텀, `num_goals`/`terrain_names`/`random_difficulty` 필드 추가) | B는 코어 확장 |
| Terrain 표현 | **trimesh mesh** (box/plane 조립) | **height-field raw** (height map) | 근본적 표현 차이 |
| size | `(20.0, 4.0)` `:45` | `(16.0, 4.0)` `:5` | A가 x축 4m 더 김 |
| border_width | 20.0 | 20.0 | 동일 |
| num_rows | **11** `:48` | **10** `parkour.py:7` | A 11단계, B 10단계 |
| num_cols | 40 | 40 | 동일 |
| horizontal_scale | 0.05 `:49` | **0.08** `parkour.py:9` (주석: 0.05 원본, IsaacLab 연산 이슈로 0.08) | |
| vertical_scale | 0.005 | 0.005 | 동일 |
| slope_threshold | 0.75 `:51` | 1.5 `:11` | |
| curriculum | True `:53` | True `:14` (+ teacher cfg `:59` 재확인) | 동일 |
| use_cache | False | False | 동일 |
| difficulty_range | (미지정 → 코어 기본 (0,1)) | `(0.0, 1.0)` `:12` | 실질 동일 |
| 표면 roughness | **없음** (mesh 함수가 깨끗한 box/plane만 생성, `parkour_terrains.py:93` `make_plane`) | **있음** — 모든 sub-terrain `apply_roughness=True`, `noise_range=(0.02,0.06)` `extreme_parkour_terrains_cfg.py:10` | A 지면은 완전 평탄, B 지면은 노이즈 |

---

## 2. Sub-terrain 구성 비율 (proportion)

### A (활성 5종, 합 1.0) — `parkour_env_cfg.py:54-244`
| 이름 | class id | proportion | terrain 함수 |
|------|----------|-----------|--------------|
| parkour_flat | 0 | **0.1** | hurdle 함수 + `flat=True` (항상 평지) |
| parkour_hurdle | 1 | 0.2 | `parkour_jump_hurdle_terrain` |
| parkour_step | 2 | 0.2 | `MeshParkourStepTerrainCfg` |
| parkour_gap | 3 | **0.3** | `MeshParkourGapTerrainCfg` |
| parkour_stair | 4 | 0.2 | `MeshParkourStairTerrainCfg` |
| (stepping_stones/beam/crawl/slope/zigzag/rough_blocks) | 5–10 | **0.0** (비활성) | 미래용 등록만 |

### B (활성 5종, 합 1.0) — `extreme_parkour/config/parkour.py:15-59`
| 이름 | proportion | terrain 함수 |
|------|-----------|--------------|
| parkour_gap | 0.2 | `parkour_gap_terrain` |
| parkour_hurdle | 0.2 | `parkour_hurdle_terrain` |
| parkour_flat | 0.2 | hurdle 함수 + `apply_flat=True` |
| parkour_step | 0.2 | `parkour_step_terrain` |
| **parkour** | **0.2** | `parkour_terrain` — extreme_parkour 시그니처(경사 stepping-stone, `incline_height`) |
| parkour_demo | 0.0 | 비활성 |

**차이 (사실):**
- A는 `parkour_stair`(계단 사이클)를 추가, B의 시그니처 `parkour`(경사 stepping-stone) 지형은 **A에 없음** (A cfg 주석 `parkour_env_cfg.py:57` "Genesis 'parkour' has no IsaacLab equivalent — omitted").
- proportion: A는 `gap` 비중이 0.3으로 더 큼, `flat`은 0.1로 더 작음. B는 5종 균등 0.2.
- 즉 **두 환경의 장애물 mix 자체가 다름** — A는 stair 포함/parkour 제외, B는 parkour 포함/stair 제외.

---

## 3. Difficulty 곡선 (difficulty 0~1 → 파라미터)

A: 선형보간 `param = lo + difficulty*(hi-lo)` (`parkour_terrains.py:83` 등)
B: 문자열식을 `eval`로 difficulty 대입 (`extreme_parkour_terrians.py:75` `eval(cfg.gap_size,{"difficulty":difficulty})`)

| 장애물 | A: d=0 → d=1 | B: d=0 → d=1 |
|--------|--------------|--------------|
| Hurdle 높이 | `(0.05,0.30)` 선형 → 0.05 → 0.30 m (`:82`) | `'0.1+0.1*d, 0.15+0.25*d'` → (0.10,0.15) → (0.20,0.40) m (`cfg.py:28`) |
| Step 높이 | `(0.10,0.45)` → 0.10 → 0.45 m (`:121`) | `'0.1+0.35*d'` → 0.10 → 0.45 m (`cfg.py:33`) — **동일** |
| Gap 길이 | `(0.05,0.5)` → 0.05 → 0.50 m (`:110`, 주석 "reduced: overflow fix, was 0.3,0.8") | `'0.1+0.7*d'` → 0.10 → 0.80 m (`cfg.py:21`) |
| Stair | width `(0.25,0.40)`, height `(0.05,0.20)` | (A 전용) |
| parkour(경사석) | (A 없음) | `incline_height='0.25*d'`, `stone_len='0.9-0.3*d,1-0.2*d'` 등 (`cfg.py:41-45`) |

**차이 (사실):**
- **Hurdle**: B는 d=0에서도 높이 0.10~0.15 m (장애물 존재). A는 d=0에서 0.05 m (거의 없음) → **A의 최저 난이도가 B보다 쉬움**.
- **Gap**: A는 명시적으로 (0.3,0.8)→(0.05,0.5)로 축소("overflow fix") → **A gap이 B보다 쉬움**.
- **Step**: 동일.
- A 난이도 step = `level/(11-1)=level/10`, B = `level/(10-1)=level/9`.

---

## 4. Curriculum 시작 레벨 / 초기 분포 (iter 0)

| 항목 | A | B |
|------|---|---|
| max_init_terrain_level | **3** `parkour_env_cfg.py:434` | **2** `default_cfg.py` ParkourDefaultSceneCfg `terrain` |
| num_rows (총 레벨 수) | 11 (level 0–10) | 10 (level 0–9) |
| iter0 초기 레벨 분포 | level ∈ {0,1,2,3} (코어 `randint(0, max_init+1)`) | level ∈ {0,1,2} |
| iter0 최대 difficulty | 3/10 = **0.30** | 2/9 ≈ **0.22** |
| terrain_type(컬럼) 배정 | proportion 기반 컬럼 누적합 LUT (`parkour_env.py:150-161`) | 코어 컬럼 배정 |
| EVAL/PLAY 분리 | 없음 (train cfg 단일) | `_EVAL`/`_PLAY` 별도 (`difficulty_range`, `random_difficulty=True` 등 오버라이드, teacher_cfg `:64-109`) |

**차이 (사실):** A는 iter0에 약간 더 높은 레벨(0–3, max d≈0.30)에서 시작, B는 0–2(max d≈0.22). 단 §3에서 본 대로 A의 동일 difficulty당 hurdle/gap이 B보다 쉬워, 실효 난이도 차이는 작음. 가설: 두 효과(레벨↑ vs 파라미터↓)가 대략 상쇄 — A iter0 난이도는 B와 대체로 비슷한 수준으로 추정.

---

## 5. Curriculum 진행 규칙 (level up/down)

| 항목 | A `parkour_env.py:_update_terrain_curriculum` (1235–1282) | B `parkour_event.py:_resample_command` (133–176) |
|------|------|------|
| 호출 시점 | `_reset_idx` 안에서 (`:1171`) | env reset 시 (ParkourEvent) |
| 거리 측정 기준점 | `env_origins` (= 로봇 spawn 위치, A는 origin에 spawn) `:1263` | `start_pos = env_origins − (size[1]+offset, 0)` `:137-139` (= 로봇 spawn 위치) |
| 측정량 | `dis_to_origin = ‖root_xy − env_origin_xy‖` | `dis_to_start_pos = ‖root_xy − start_pos‖` |
| 기대거리 | `command[:,0].abs() * max_episode_length_s` `:1267` | `command[:,0] * episode_length_s` `:142` |
| move_up 조건 | `dis > 0.8 * expected` `:1268` | `dis > 0.8 * threshold` `:143` |
| move_down 조건 | `dis < 0.4 * expected` `:1269` | `dis < 0.4 * threshold` `:144` |
| level 갱신 | `level += move_up − move_down` `:1272` | 동일 `:147` |
| 최상위 도달 처리 | `level ≥ max_level(=num_rows−1=10)` → `randint(0, max_level+1)` (10 **포함**) `:1274-1277` | `level ≥ max_terrain_level` → `randint(0, max_terrain_level)` (max **미포함**) `:149-151` |
| 하한 | `clamp(·, 0)` | `clip(·, 0)` |

**차이 (사실):**
- 핵심 규칙(0.8/0.4 임계, ±1 step)은 **A·B 동일**.
- 측정 기준점이 다름: 둘 다 "각자의 spawn 위치"를 기준으로 이동거리를 재므로 내부적으론 일관. 단 A는 origin에 spawn(§6), B는 origin−7m에 spawn.
- 최상위 wrap: A는 최상위 레벨 포함하여 random 재배치, B는 미포함. 미미.

---

## 6. Spawn 위치 / origin 처리

| 항목 | A | B |
|------|---|---|
| terrain 함수가 반환하는 origin | `[platform_length/2, size_y/2, 0]` — **start 플랫폼 중앙** (`parkour_terrains.py:142`) | height-field 타일 기준 origin |
| robot spawn (reset) | `default_root_state[:,:3] += env_origins`, `z += 0.05` (`parkour_env.py:1183-1184`) → **env_origin(=플랫폼 중앙)에 직접 spawn** | `events.reset_root_state` `offset=3` → `pos = default + origin − (size[1]+offset, 0,0) = origin − (7,0,0)` (`events.py:58-59`) → **env_origin에서 −x로 7 m 뒤** |
| 좌표 컨벤션 | terrain origin 자체가 플랫폼 근처 → 오프셋 불필요 | env_origin이 타일 중앙 → −7 m 보정해 타일 시작부로 이동 (코드 주석 `parkour_event.py:134-135`) |
| z buffer | +0.05 m (settling) | 없음 (default_root_state 그대로) |

**차이 (사실):** spawn 컨벤션이 다르나 둘 다 "로봇을 start 플랫폼에 놓는" 동일 의도. A·B 모두 curriculum 거리측정은 각자 spawn 기준이라 내부 일관성 유지. 가설: A에서 `env_origins`가 실제로 플랫폼 중앙에 정확히 오는지(terrain 함수 origin이 importer까지 올바르게 전파되는지)는 별도 검증 권장 — 어긋나면 curriculum 거리·goal 좌표가 동시에 편향됨.

---

## 7. 기타 Scene 차이

| 항목 | A | B |
|------|---|---|
| num_envs | 4096 `parkour_env_cfg.py:453` | 6144 (teacher) `teacher_cfg.py:35` |
| env_spacing | 4.0 | 1.0 (terrain=generator이므로 origin은 terrain grid가 결정 → 영향 미미) |
| terrain importer class | 코어 `TerrainImporterCfg` | 커스텀 `ParkourTerrainImporter` (`default_cfg.py`) |
| height_scanner 패턴 | `resolution=0.1, size=[1.6,1.0]` (187점) | `resolution=0.15, size=[1.65,1.5]` (teacher_cfg `:19`) |
| flat_patch_sampling | 모든 sub-terrain에 `init_positions` 정의 (`num_patches=2`, `patch_radius=0.5`) | 명시적 flat_patch_sampling 없음 |

---

## 8. 학습 초반 영향 가설 (우선순위)

> 본 task 제약상 "원인 단정/RCA ranking"은 하지 않음. 아래는 **terrain/curriculum 관점에서 검토할 가설**이며, 우선순위는 "초반 학습에 직접 닿을 개연성" 순.

**[가설 1] terrain 표면 표현 차이 (mesh vs height-field + roughness)** — 우선 검토
- A는 trimesh box/plane만 생성하여 지면이 완전 평탄, B는 모든 sub-terrain에 `apply_roughness=True` 노이즈.
- 이는 A를 더 "쉽게" 만들지만, height_scan / 접촉 분포 / 발 배치가 B와 다른 분포를 가지게 함. A 정책이 B와 다른 obs 분포에서 학습됨. (학습 차단 요인이라기보단 환경 fidelity 차이 — 사실로 기록.)

**[가설 2] env_origin → spawn 좌표 전파 정합성** — 검증 권장
- A는 terrain 함수 origin(`platform_length/2, size_y/2, 0`)이 `TerrainImporter.env_origins`로 올바르게 전달되어 로봇이 실제 start 플랫폼 위에 spawn하는지가 curriculum 거리측정(`:1263`)·goal 좌표(`:542`)와 동시에 연동됨.
- 만약 origin 전파가 어긋나면 iter0부터 (a) 로봇이 플랫폼 밖/장애물 위에 spawn, (b) `dis_to_origin`이 항상 비정상 → curriculum 오작동, (c) goal이 엉뚱한 위치 — 동시 발생. §6 참조. (가설, 미검증.)

**[가설 3] 장애물 mix 불일치 (stair 추가 / parkour 제외)** — 사실 기록
- A는 B 시그니처 `parkour`(경사 stepping-stone) 대신 `parkour_stair`를 0.2 비중으로 사용. 두 환경이 학습하는 obstacle 분포가 근본적으로 다름. B에서 잘 되는 정책/하이퍼파라미터가 A 지형 mix에 그대로 최적이 아닐 수 있음.
- A는 `gap` 비중 0.3(B는 0.2), `flat` 0.1(B는 0.2) → A는 평지 노출이 절반, gap 노출이 1.5배. 초반 "쉬운 성공 경험" 비중이 B보다 작음.

**[가설 4] iter0 시작 난이도** — 영향 작을 것으로 추정
- A: max_init_level=3 (max d≈0.30), B: max_init_level=2 (max d≈0.22)로 A가 약간 높게 시작.
- 그러나 §3에서 A의 동일 difficulty당 hurdle(d=0:0.05 m)·gap(축소됨)이 B보다 쉬움. 두 효과가 대체로 상쇄 → 이 항목 단독으로 초반 학습 차단 가능성은 낮다고 판단. (단, gap 0.3 비중 + A gap 함수의 실제 통과 가능성은 별도 확인 가치 있음.)

**규칙 자체(0.8/0.4 임계, ±1 step, curriculum 진행식)는 A·B 동일** — curriculum 알고리즘 로직은 차이 없음.

---

## 부록 — 인용 파일 경로
- A: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py`, `parkour_terrains.py`, `parkour_env.py`
- B: `Isaaclab_Parkour/parkour_isaaclab/terrains/extreme_parkour/{config/parkour.py, extreme_parkour_terrains_cfg.py, extreme_parkour_terrians.py}`,
  `parkour_isaaclab/terrains/{parkour_terrain_generator_cfg.py, parkour_terrain_importer.py}`,
  `parkour_isaaclab/envs/mdp/parkours/parkour_event.py`, `parkour_isaaclab/envs/mdp/events.py`,
  `parkour_tasks/parkour_tasks/default_cfg.py`, `parkour_tasks/.../config/go2/parkour_teacher_cfg.py`

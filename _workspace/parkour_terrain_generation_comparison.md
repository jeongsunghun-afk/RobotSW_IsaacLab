# Parkour Terrain 생성 방식 비교 분석

> 비교 대상
> - **A. `parkour` 환경** — `IsaacLab/source/isaaclab_tasks/.../direct/parkour/parkour_terrains.py` (단일 1237줄)
> - **B. `Isaaclab_Parkour`** — `/home/lgb/Isaaclab_Parkour/parkour_isaaclab/terrains/` (모듈 패키지, 총 ~811줄)
>
> 작성일 2026-05-18 · read-only 분석

---

## 0. 한눈에 보는 결론

| 항목 | A. parkour 환경 | B. Isaaclab_Parkour |
|---|---|---|
| **Mesh 표현 방식** | **직접 trimesh box/wedge** (geometric primitive) | **height-field → mesh 변환** (heightmap rasterize) |
| **TerrainGenerator** | IsaacLab **stock 그대로** 사용 | `ParkourTerrainGenerator(TerrainGenerator)` **커스텀 상속** |
| **TerrainImporter** | stock + 경량 `_CapturingTerrainImporter` 서브클래스 | `ParkourTerrainImporter(TerrainImporter)` **커스텀 상속** |
| **코드 조직** | 단일 파일 통합 (1237줄) | 모듈 분리 + `extreme_parkour/` 논문 서브패키지 |
| **Goal 전달 경로** | 전역 mutable registry (`PARKOUR_GOALS_REGISTRY`) | generator 소유 배열 (`self.goals`) |
| **Goal 차원** | 3칸 배열 — z에 값은 채우나 env 로직(도달/보상)은 x,y만 사용 | 2칸 배열 — z 미사용 |
| **난이도 파라미터** | 숫자 `*_range` 튜플, Python 보간 | **문자열 표현식** (`'0.1 + 0.7*difficulty'`) eval |
| **IsaacLab core 영향** | core `trimesh/mesh_terrains.py`에 의존 (core 확장됨) | core **무수정** — 자체 패키지에 격리 |
| **계보 (lineage)** | Genesis 시뮬레이터 스타일 + IsaacLab core parkour | Extreme Parkour 논문(arXiv 2309.14341) 직접 포팅 |
| **타일/그리드** | 20m×4m 타일, 11행×40열 | 16m×4m 타일, 10행×40열 |

**핵심 차이 한 줄 요약**: A는 *기하 primitive(box)를 직접 조립*해 IsaacLab의 stock 파이프라인에 얹은 구조이고, B는 *height-field를 rasterize*하며 Generator/Importer까지 통째로 상속·확장한 *논문 충실 포팅*이다.

---

## 1. Sub-terrain Primitive 정의

### A. parkour 환경 — 직접 mesh 조립

- 단일 파일 `parkour_terrains.py`에 **8개 함수**: 커스텀 hurdle 1 + edge-mask 유틸 1 + 신규 지형 6 (stepping_stones, balance_beam, crawl, slope, zigzag_hurdles, rough_blocks).
- primitive는 `trimesh.creation.box()` 축정렬 박스, slope만 6~8 vertex 커스텀 trimesh, 바닥은 `make_plane()`/`make_border()`.
- 각 지형 = "바닥 평면 + 박스 장애물 N개"를 **명시적으로 concat**. 높이맵 개념 없음.

```python
# parkour_terrains.py:43-48
from isaaclab.terrains.trimesh.mesh_terrains import PARKOUR_GOALS_REGISTRY
from isaaclab.terrains.trimesh.mesh_terrains_cfg import MeshParkourHurdleTerrainCfg
from isaaclab.terrains.trimesh.utils import make_border, make_plane
```

### B. Isaaclab_Parkour — height-field rasterize

- `extreme_parkour/extreme_parkour_terrians.py`에 **6개 함수** (gap, hurdle, step, parkour, demo + random_uniform 헬퍼).
- 각 함수는 **numpy height_field(int16)** 를 직접 채운 뒤 `@parkour_field_to_mesh` 데코레이터가 `convert_height_field_to_mesh()`로 mesh 변환.

```python
# terrains/utils.py:14-59
@parkour_field_to_mesh
def wrapper(difficulty, cfg, num_goals):
    vertices, triangles, x_edge_mask = convert_height_field_to_mesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_threshold)
    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
    return [mesh], origin, goals, goal_heights, x_edge_mask
```

> **차이의 본질**: A는 깨끗한 박스 geometry라 모서리(edge) 정보가 없어서 *나중에* mesh를 캡처해 edge-mask를 계산해야 한다(§3 참조). B는 height-field를 변환할 때 `slope_threshold`로 `x_edge_mask`를 **네이티브로 즉시** 산출한다. mesh-vs-heightfield 선택이 다운스트림 전체를 가른다.

---

## 2. Terrain Generator / Curriculum

### A — stock generator 그대로

- `PARKOUR_TERRAINS_CFG = TerrainGeneratorCfg(...)` — IsaacLab **stock `TerrainGenerator` 무수정 사용**. 커스텀한 건 sub-terrain *함수*뿐, 오케스트레이션은 stock.
- curriculum=True. 난이도 = `(row_id + η)/num_rows`, η~U(0,1). 열(column)은 proportion cumsum으로 지형 타입 배정.
- 활성 구성: flat 0.1 / hurdle 0.2 / step 0.2 / gap 0.3 / stair 0.2 (합=1.0). 신규 6종은 전부 `proportion=0.0`으로 등록만 되고 비활성.

### B — generator 자체를 상속·재작성

- `ParkourTerrainGenerator(TerrainGenerator)` — `_generate_curriculum_terrains()` / `_generate_random_terrains()` **두 모드를 직접 구현**.

```python
# parkour_terrain_generator.py:9
class ParkourTerrainGenerator(TerrainGenerator):
# :52-66  난이도 진행 — random_difficulty 플래그로 분기
    if self.cfg.random_difficulty:
        difficulty = (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
    else:
        difficulty = sub_row / (self.cfg.num_rows - 1)
```

- 활성 구성: gap/hurdle/flat/step/parkour 각 0.2, demo 0.0. `random_difficulty` 플래그로 결정적/확률적 난이도 전환.

> A는 stock 파이프라인에 함수만 끼워넣어 **유지보수 비용이 낮지만 generator 거동을 바꿀 수 없다**. B는 generator를 소유해 `num_goals`, `random_difficulty`, goal 수집 로직 등 **거동 자체를 제어**하지만 IsaacLab 버전 변화에 더 취약하다.

---

## 3. Importer / Scene 통합

### A — stock importer + 경량 mesh 캡처

```python
# parkour_env.py:61-79
class _CapturingTerrainImporter(TerrainImporter):
    def import_mesh(self, name, mesh, **kwargs):
        if self._captured_trimesh is None:
            self._captured_trimesh = mesh   # 사라지기 전 trimesh 가로채기
        super().import_mesh(name, mesh, **kwargs)
```

- 목적은 단 하나 — stock importer가 버리는 concat trimesh를 보존해 **edge-mask를 사후 계산**(§1의 결과). origin 계산·env 배치는 전부 stock.
- world goal = `terrain_origins[row,col] + (local_goal − local_origin)`.

### B — importer가 generator를 소유

```python
# parkour_terrain_importer.py:44-57
self._terrain_generator_class = ParkourTerrainGenerator(cfg=..., device=self.device)
self.import_mesh("terrain", self._terrain_generator_class.terrain_mesh)
self.configure_env_origins(self._terrain_generator_class.terrain_origins)
self._terrain_flat_patches = self._terrain_generator_class.flat_patches
```

- `ParkourTerrainImporter`가 `ParkourTerrainGenerator`를 직접 인스턴스화하고, goal/flat_patch/edge_mask 등 확장 속성을 generator에서 끌어온다. mesh 변환 시 origin에 transform 적용.

> A의 서브클래스는 *데이터 보존용 hook 하나*에 불과하고, B의 서브클래스는 *generator-importer 결합을 새로 설계*했다. A는 "stock에 최소 침습", B는 "전용 서브시스템".

---

## 4. Config Schema

### A
- 신규 6 Cfg 클래스 모두 `SubTerrainBaseCfg` 상속, `@configclass`. 파라미터는 **숫자 `*_range` 튜플** (`slope_angle_deg_range`, `block_density_range` …) → 함수 안에서 `param = lo + difficulty*(hi-lo)` 선형 보간.
- 클래스: `MeshParkourSteppingStonesTerrainCfg`, `…BalanceBeam…`, `…Crawl…`, `…Slope…`, `…ZigzagHurdles…`, `…RoughBlocks…`Cfg. 전부 `proportion=0.0` 기본(비활성).

### B
- 계층: `HfTerrainBaseCfg → ParkourSubTerrainBaseCfg → ExtremeParkourRoughTerrainCfg → 5개 구체 Cfg`. Generator는 `ParkourTerrainGeneratorCfg(TerrainGeneratorCfg)`에 `num_goals`, `terrain_names`, `random_difficulty` 추가.
- 난이도 파라미터가 **문자열 표현식**:

```python
# extreme_parkour_terrains_cfg.py
gap_size: str = '0.1 + 0.7*difficulty'
hurdle_height_range: str = '0.1 + 0.1*difficulty, 0.15 + 0.15*difficulty'
step_height: str = '0.1 + 0.35*difficulty'
```

> A는 타입 안전한 숫자 튜플(IDE/타입체커 친화적), B는 문자열 eval(논문 수식을 그대로 옮겨 가독성↑·동적이지만 런타임 파싱 의존). 설계 철학이 정반대.

---

## 5. Goal / Waypoint 생성

| | A. parkour 환경 | B. Isaaclab_Parkour |
|---|---|---|
| 생성 위치 | terrain 함수 내부 | terrain 함수 내부 |
| 전달 메커니즘 | **전역 mutable list** `PARKOUR_GOALS_REGISTRY.append((goals, origin))` | 함수가 `goals` **반환** → generator가 `self.goals[row,col]`에 수집 |
| env 측 재구성 | 호출 순서(k)로 (row,col) 역산 → `_build_terrain_goals_map()` | generator 배열 직접 인덱싱 |
| 차원 | `(num_goals, 3)` 배열. z에 의미값 기록(stepping_stones=stone_h, balance_beam=beam_top, slope=slope_h) | `(num_goals, 2)`만 채움, z=0 |
| z 실사용 | **미사용** — `parkour_env.py:583-593`에서 도달 판정·reward 모두 `[:, :2]` 슬라이싱(2D) | 미사용 (애초에 없음) |
| 좌표계 | local→world: `origin + (goal−local_origin)` | `goals -= 0.5*size` 중심정규화 후 `*horizontal_scale` |

```python
# A: parkour_terrains.py:107-154 — 전역 registry에 부수효과로 append
PARKOUR_GOALS_REGISTRY.append((goals.copy(), origin.copy()))

# B: parkour_terrain_generator.py:150 — generator가 명시적으로 수집
self.goals[row, col, :, :2] = sub_terrain_goal
```

> **A의 약점**: 전역 registry는 생성 호출 순서에 의존해 env에서 (row,col)을 역산한다(curriculum 여부에 따라 col-major/row-major 분기). 순서가 어긋나면 goal이 엉뚱한 타일에 매핑된다 — fragile. **B의 장점**: generator가 `(row,col)`을 알고 직접 배열에 쓰므로 순서 의존성·전역 상태가 없다.
>
> **goal 차원 정정**: A의 goal 배열은 `(num_goals, 3)`로 z 칸이 있고 terrain 함수가 stone_h·beam_top·slope_h 같은 의미값을 z에 기록한다. **그러나 `parkour_env.py`의 goal 로직은 z를 전혀 쓰지 않는다** — 도달 판정(`:583-589` `cur_goals[:, :2]`)도 reward(`:593`, `:931` `_target_pos_rel`)도 x,y만 사용. 따라서 **기능적으로 A도 B와 동일하게 2D goal**이며, A의 z는 저장만 되는 비활성 메타데이터(시각화/로깅용 추정)다. 차원 자체는 실질적 차이가 아니다.

---

## 6. Origin / Lineage

### A — Genesis 스타일 + IsaacLab core 통합
- `parkour_jump_hurdle_terrain`은 주석상 Genesis 시뮬레이터(Unitree) parkour 스타일. 학술 논문 직접 포팅 아님.
- **IsaacLab core를 확장한 흔적**: `from isaaclab.terrains.trimesh.mesh_terrains import PARKOUR_GOALS_REGISTRY`, `MeshParkourHurdleTerrainCfg` — 즉 active 5종(flat/hurdle/step/gap/stair)은 IsaacLab **core 자체에 추가된** parkour 지원에 의존. 신규 6종만 환경 디렉토리의 단일 파일에 통합.
- 구조: 단일 파일 monolith. 신규 지형은 backward-compat 위해 `proportion=0.0` 비활성 상태로 등록.

### B — Extreme Parkour 논문 충실 포팅
- `"""Reference from https://arxiv.org/pdf/2309.14341"""` — *Extreme Parkour with Legged Robots* 논문의 terrain generator를 직접 재구현.
- `extreme_parkour/` **서브패키지로 명시 분리** — base generator(논문 독립)와 논문 구현을 계층 분리. 향후 다른 parkour 논문을 `extreme_parkour_v2/` 식으로 추가 가능하도록 설계된 확장 구조.
- IsaacLab core는 **건드리지 않음** — 모든 것이 `parkour_isaaclab` 패키지에 격리.

> **계보 차이**: A는 IsaacLab에 흡수·통합되는 방향(core까지 확장), B는 IsaacLab 위에 얹히는 독립 패키지로 논문 재현성을 우선. A는 "IsaacLab 네이티브 환경", B는 "IsaacLab 플러그인형 연구 코드".

---

## 7. 종합 — 어떤 차이가 본질인가

1. **표현 방식 (가장 근본)**: A=직접 mesh primitive 조립 / B=height-field rasterize. 이 한 가지가 edge-mask 계산(A는 사후 캡처, B는 네이티브)·config 형태까지 연쇄적으로 가른다. (단 goal 차원은 차이 아님 — 아래 참조)
2. **stock 재사용 vs 상속 확장**: A는 Generator/Importer를 stock 그대로 두고 함수만 주입 → 침습 최소·유지보수 쉬움, 거동 제어 불가. B는 둘 다 상속해 전용 서브시스템 구축 → 제어력↑, IsaacLab 버전 결합도↑.
3. **상태 관리**: A의 전역 `PARKOUR_GOALS_REGISTRY`는 호출 순서 역산 의존으로 fragile. B는 generator 소유 배열로 전역 상태·순서 의존 없음 — B가 더 견고.
4. **코드 조직**: A=단일 파일 통합(환경에 밀착), B=모듈+논문 서브패키지(재현·확장 지향).
5. **계보**: A=Genesis 스타일을 IsaacLab core까지 확장 통합 / B=Extreme Parkour 논문(arXiv 2309.14341)을 core 무수정으로 포팅.

### 권장 시사점
- **goal 매핑 견고성**이 중요하면 B 방식(generator 소유 배열)이 A의 전역 registry보다 안전. A를 계속 쓴다면 registry 순서 역산 로직(`curriculum` 분기)을 회귀 테스트로 고정할 것.
- **goal 차원은 둘 다 실질 2D** — A의 goal 배열에 z 칸이 있고 값도 채워지지만 `parkour_env.py`가 z를 안 쓴다(도달/보상 모두 x,y만). "A는 3D" 인상은 코드 표면일 뿐 기능 차이가 아니다. 3D goal(높이 있는 waypoint 도달 판정)이 필요하면 **A·B 모두 env/generator 측 추가 구현 필요**.
- **IsaacLab 버전 업그레이드 내성**은 A(stock 의존 최소)가 유리하나, A는 이미 core에 parkour 코드를 추가한 상태이므로 core 변경분 관리가 별도 부담.

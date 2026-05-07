# Parkour Terrain Extension — 통합 요약 (User-Approved)

> **Team**: `parkour-terrain-design` · **Lead**: team-lead · **Workers**: planner-crawl, planner-newterrains, critic-reviewer
> **Source plans**:
> - `_workspace/plans/parkour_crawl_plan.md` (Yellow, 13.1k chars)
> - `_workspace/plans/parkour_new_terrains_plan.md` (Yellow, 27.6k chars)
> - `_workspace/plans/parkour_terrain_critic_review.md` (Critical 4 / Should Fix 5 / Nice 4)
> **User decisions (4개)**: 본 문서 §1.

---

## §1. 사용자 확정 결정

| # | 결정 항목 | 선택 |
|---|-----------|------|
| 1 | Difficulty 매핑 | **IsaacLab 표준 통일** (row 0 = easy, row last = hard). crawl도 row 0 = ceiling 높음(쉬움), row last = ceiling 낮음(어려움). |
| 2 | First-cut 신규 지형 | **stepping_stones + balance_beam + crawl 셋 다 동시 도입** (공격적 통합) |
| 3 | Crawl 우선순위 | **동시 도입** (Phase 1에 stones/beam과 함께) |
| 4 | Slope mesh 방식 | **Wedge primitive** (현재 WARP raycast edge_mask가 정확히 처리 가능) |

> **결과 영향**: critic이 권장한 Phase 1=stones+beam, Phase 3=crawl 분할은 압축됨. Phase 1에 3종 동시 도입하므로 reward 함수 재구조화(Critical C4)가 Phase 1 진입 전 prerequisite가 됨.

---

## §2. terrain_class 번호표 (Critical C1 해결 — 사전 확정)

| Terrain | class id | 비고 |
|---------|----------|------|
| flat | 0 | 기존 |
| hurdle | 1 | 기존 |
| step | 2 | 기존 |
| gap | 3 | 기존 |
| stair | 4 | 기존 |
| **stepping_stones** | **5** | 신규 |
| **balance_beam** | **6** | 신규 |
| **crawl** | **7** | 신규 |
| slope_wedge | 8 | 신규 (Phase 2) |
| zigzag_hurdles | 9 | 신규 (Phase 2) |
| rough_blocks | 10 | 신규 (Phase 2) |

> `parkour_env.py:99-115`의 `_col_to_class` LUT를 위 번호표대로 명시적으로 작성. dict insertion order 의존 제거.

---

## §3. proportion 단일표 (Critical C3 해결)

> sum = 1.000000 (부동소수 정확). `parkour_env.py:105-108` assert 통과.

### Phase 1 (3종 신규 + 기존 5종)

| Terrain | proportion |
|---------|------------|
| flat | 0.05 |
| hurdle | 0.15 |
| step | 0.15 |
| gap | 0.20 |
| stair | 0.15 |
| stepping_stones | 0.10 |
| balance_beam | 0.10 |
| crawl | 0.10 |
| **합** | **1.00** |

### Phase 2 (slope/zigzag/rough 추가 후 11종)

| Terrain | proportion |
|---------|------------|
| flat | 0.05 |
| hurdle | 0.10 |
| step | 0.10 |
| gap | 0.15 |
| stair | 0.10 |
| stepping_stones | 0.10 |
| balance_beam | 0.10 |
| crawl | 0.10 |
| slope_wedge | 0.07 |
| zigzag_hurdles | 0.07 |
| rough_blocks | 0.06 |
| **합** | **1.00** |

> 위 비율은 기준선. 학습 안정성 보고 최대 ±2%p 조정 가능.

---

## §4. 통합 Phase 계획

### Phase 0 — Prerequisite (0.5일, READ-only 결정 + 골격 작성)
- terrain_class 번호표 §2 확정
- `_col_to_class` LUT 명시적 작성안 (`parkour_env.py:99-115` 패치 초안)
- proportion 단일표 §3 확정
- 단일 통합 PR 브랜치 생성: `feat/parkour-terrain-stones-beam-crawl`

### Phase 1A — Reward 함수 재구조화 (Critical C4 선행) (1일)
> stones+beam+crawl을 동시 도입하므로 reward 분기는 8-class 가중표 prerequisite.
- **Worker**: `reward-worker`
- 파일: `parkour_env.py:601-689` `_get_rewards()`
- 변경: `is_flat / is_non_flat` 이분 → `class_weight_table[term_name][env_class]` 텐서 다중 가중
- 회귀 테스트: 기존 5종에서 reward magnitude 변화율 < 1%
- 검증 명령: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task <Parkour-Task> --num_envs 256 --max_iterations 50` → 기존과 reward curve 비교

### Phase 1B — 신규 3종 mesh + cfg + goal (1.5일, 병렬 가능)
> reward 재구조화 완료 후 시작.

#### 1B-a. stepping_stones
- **Worker**: `obs-worker` (terrain 함수 + cfg) + `cfg-worker` (env_cfg 등록)
- 파일:
  - `parkour_terrains.py`: `parkour_stepping_stones_terrain()` 신규 함수 (trimesh.creation.box 격자 + lateral_jitter)
  - `mesh_terrains_cfg.py:323-483` 패턴 따라 `MeshParkourSteppingStonesTerrainCfg` 추가
  - `parkour_env_cfg.py:43-134`의 `PARKOUR_TERRAINS_CFG.sub_terrains`에 entry
- Goal: 마지막 디딤돌 + 출구 platform에 2개

#### 1B-b. balance_beam
- 동일 worker 구조
- Risk: **height_scan 0.1m 해상도 vs beam 폭 0.2m** (Should Fix S4)
  - 완화: row 0에서 beam_width=0.5m 시작, row last=0.2m
  - 옵션: priv obs에 beam centerline 1-D scalar 추가 (별도 결정 필요)

#### 1B-c. crawl
- **Worker**: `obs-worker` + `cfg-worker`
- 파일:
  - `parkour_terrains.py`: `parkour_crawl_terrain()` 신규 (corridor floor + 양옆 wall + ceiling box)
  - 핵심 파라미터:
    - `corridor_width=0.8m` (Go2 폭 0.3m × 2.7배; critic N1 의견 반영해 1.2m → 0.8m로 조정)
    - `ceiling_thickness=0.10m`
    - `ceiling_z_low_range=(0.30, 0.50)m` — row 0 = 0.50m (쉬움, hip ~0.30m라 살짝만 숙임), row last = 0.30m (어려움, 거의 배 깔고 통과)
    - `ceiling_top_extra=0.5m` (점프 통과 봉쇄, Go2 점프정점 고려; critic N3)
  - Goal: 입구 직전 + 출구 직후 2개

### Phase 1C — Observation 호환성 패치 (1일)
- **Worker**: `obs-worker`
- height_scan ceiling 회피 (Critical C2 + crawl plan §4):
  - 권장: ceiling을 별도 RigidObject prim으로 분리 + RayCaster `mesh_prim_paths`에서 제외
  - 또는: `parkour_env.py:510-512` height_scan 정규화 후 `clip(min=0)`로 ceiling hit 신호 죽임
- Edge mask z-filter (Critical C2 — **수정 위치 정정**):
  - **수정 지점**: `parkour_terrains.py:262-263` WARP raycast `hit_z` 처리부
  - 패치 형태: `hit_z = torch.where(hit_z > ceiling_z_threshold, torch.zeros_like(hit_z), hit_z)` (대략, 정확한 변수명은 코드 확인 필요)
  - kwarg 추가: `compute_edge_mask_from_terrain_mesh(..., ceiling_z_threshold: float | None = None)`

### Phase 1D — Curriculum + 학습 검증 (1일)
- 200 iter PPO sanity (`num_envs=2048`)
- Success rate per class 모니터: 모든 class에서 > 0.3 (낮은 난이도 row)
- WandB run 비교: 기존 5종 + 신규 3종, success rate / reward curve

### Phase 2 — slope_wedge + zigzag_hurdles + rough_blocks (2일)
> Phase 1 안정 후 진행.

#### 2-a. slope_wedge (사용자 선택 §1.4)
- trimesh wedge primitive (`trimesh.creation.box` + 한 면 변형 또는 `trimesh.Trimesh(vertices, faces)` 직접 빌드)
- terrain_class=8, edge_mask는 raycast가 axis-free라 그대로 OK
- reward feet_edge는 class=8에서 비활성 (Should Fix S1 권장 무관, wedge mesh는 Laplacian polluting 없음)

#### 2-b. zigzag_hurdles
- 좌/우 교대 배치된 hurdle. lateral steering 학습.
- terrain_class=9
- `lin_vel_y_range=[0,0]` 유지(critic N4) + 첫 200 iter 회피율 모니터

#### 2-c. rough_blocks
- 무작위 요철 grid. unstructured surface.
- terrain_class=10
- edge_mask threshold 조정 필요 (Should Fix S5: 20→80 시작값 권장)

### Phase 3 — AMP/모방학습 통합 (별도 PR)
- 본 plan 범위 밖. crawl pose는 motion reference에 없을 가능성 → AMP discriminator가 negative reward로 정책 억제할 수 있음.
- 후속 작업으로 `motion-analyzer`로 reference coverage 분석 + crawl-prior reward 추가 검토.

---

## §5. Open Issue (사용자 추가 결정 필요 — 비차단)

| # | 항목 | 권장 | 차단 여부 |
|---|------|------|-----------|
| O1 | balance_beam priv obs centerline 1-D scalar 추가? | 추가 권장 (Phase 1B-b 학습 실패 시 활성화) | 비차단 |
| O2 | crawl ceiling material/색상 (시각 디버깅용) | 별도 색상(파랑) 권장 | 비차단 |
| O3 | reward `class_weight_table` 초기값 (모든 class에서 어떤 항을 어떻게) | reward-worker가 1차 안 작성 후 review | Phase 1A 진입 시 결정 |
| O4 | num_envs 권장 (8종 → 2048 vs 4096) | 4096 권장 (지형 다양성↑) | Phase 1D 진입 시 결정 |
| O5 | rough_blocks edge_mask threshold 사전값 (20 vs 80) | 80에서 시작 후 sanity | Phase 2-c 진입 시 결정 |

---

## §6. 실행 시 호출할 worker 순서

```
[Phase 0]   lead 직접: terrain_class 번호표 + proportion 단일표 패치 초안
[Phase 1A]  reward-worker     → _get_rewards() 8-class 재구조화
[Phase 1B]  obs-worker × 3    → 3종 terrain 함수 + cfg + goal (병렬)
            cfg-worker        → env_cfg 통합 등록
[Phase 1C]  obs-worker        → height_scan / edge_mask 호환 패치
[Phase 1D]  log-analyzer      → 200 iter sanity 학습 결과 분석
[Phase 2]   동일 worker 묶음 반복 (slope/zigzag/rough)
[Phase 3]   motion-analyzer + reward-worker (별도 PR)
```

---

## §7. Critical Issues 해결 매핑 (검증)

| Critical | 해결 위치 | 상태 |
|----------|-----------|------|
| C1 — class=5 충돌 | §2 번호표 사전 확정 | ✅ |
| C2 — edge_mask 수정 위치 오류 | §4 Phase 1C에 정확한 라인(L262-263) 명시 | ✅ |
| C3 — proportion sum=1.0 assert | §3 단일표 (Phase 1: 8개 합=1.00, Phase 2: 11개 합=1.00) | ✅ |
| C4 — reward 분기 재구조화 | §4 Phase 1A를 Phase 1B 진입 전 prerequisite로 | ✅ |

---

## §8. 다음 단계 (Decision Point)

위 통합 plan을 검토하시고:
- (A) Phase 0(번호표 + proportion 패치) 즉시 진행
- (B) Phase 1A reward-worker 즉시 dispatch (Phase 0와 병렬)
- (C) plan을 더 다듬어야 한다고 판단되는 항목 지정
- (D) 본 plan 보관만 하고 실제 구현은 별도 세션에서

중 어느 것을 원하시는지 결정 부탁드립니다.

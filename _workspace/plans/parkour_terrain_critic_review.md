# Parkour Plan 비판 검토 — Task #3 결과

> Reviewer: `critic-reviewer` · 두 plan을 read-only로 spot-check 후 종합 분석.
> 코드 근거: `parkour_terrains.py:144-278` (edge_mask, **현재 WARP raycast 기반**), `parkour_env.py:99-115` (`_col_to_class` LUT + assert), `parkour_env.py:510-512` (height_scan), `parkour_env.py:599-689` (reward `is_flat`/`is_non_flat` 이분), `parkour_env_cfg.py:43-134` (sub_terrains + class 상수), `mesh_terrains_cfg.py:323-483` (cfg 정의).

---

## Section 1. 종합 판정 (Green/Yellow/Red)

| Plan | 판정 | 핵심 근거 |
|------|------|----------|
| `parkour_crawl_plan.md` | **Yellow** | Mesh 설계·goal·curriculum은 견고. 그러나 **§5 edge_mask 수정 위치가 코드 사실과 불일치**(현재 코드는 face rasterization이 아니라 WARP GPU raycast). 7장 difficulty 매핑이 사용자 요구("난이도↓=낮게 숙임")와 명시적으로 반대 방향이며 본인이 OQ로 인정. Phase 단위와 rollback 잘 정의됨. |
| `parkour_new_terrains_plan.md` | **Yellow** | Top-5 후보 다양성·sim2real 매핑은 우수, 우선순위 표·proportion 테이블·체크리스트도 충실. 그러나 **slope를 staircase 근사하면서 "ANYmal도 동일 처리" 주장**은 출처 불명(논문은 heightfield slope). reward 분기 8열 표가 현재 `is_flat`/`is_non_flat` 이분 코드 구조와 호환되도록 어떻게 코드를 바꿀지 구체화 부족. |

---

## Section 2. Critical Issues (반드시 수정)

### C1. **두 plan 간 `TERRAIN_CLASS = 5` 충돌**
- crawl plan: `TERRAIN_CLASS_CRAWL = 5` (§6, Phase 1)
- new_terrains plan: `TERRAIN_CLASS_STEPPING_STONES = 5` (D3)
- 두 plan을 둘 다 적용하면 `_col_to_class` LUT(`parkour_env.py:114`)는 dict insertion order로 자동 부여되므로 **먼저 dict에 추가된 쪽이 5, 뒤가 6**이 된다. 두 plan이 독립적으로 진행되면 reward 분기 매핑이 어긋남. **통합 plan에서 클래스 번호를 사전 확정**해야 함.

### C2. **crawl plan §5의 edge_mask 수정 위치가 코드와 어긋남**
plan은 `parkour_terrains.py:248-260 face rasterization 루프에서 z 상한 필터 추가`라고 명시하지만, 실제 코드(`parkour_terrains.py:212-265`)는 **WARP GPU raycast**로 이미 교체되어 있다(face rasterization 아님). 올바른 수정 지점은 L262-263의 `hit_z` 처리부 — 예: `hit_z = torch.where(hit_z > ceiling_z_threshold, torch.zeros_like(hit_z), hit_z)`. 그렇지 않으면 patch가 적용 안 됨 (혹은 사라진 코드를 수정).

### C3. **proportion sum 1.0 assert 위배 위험**
`parkour_env.py:105-108`에 `assert abs(sum(_props) - 1.0) < 1e-6`가 있다. 두 plan이 각자 독립으로 비율을 재배분하면(crawl: gap 0.30→0.25, new_terrains: gap 0.30→0.18) 통합 시 부동소수 오차 또는 상호 충돌로 startup assert 실패 가능. 통합 단계에서 **단일 비율표를 산출**하고 floats를 직접 명시해야 함.

### C4. **reward 분기는 단순 코드 변경이 아님**
현재 `_get_rewards`(`parkour_env.py:601-689`)는 모든 conditional이 `is_flat * a + is_non_flat * b` 이분 구조다. new_terrains plan의 D3 8-클래스 표를 그대로 적용하려면 **8개 mask + per-term 다중 가중 합산**으로 함수 구조 자체를 재작성해야 한다. 양 plan 모두 "분기 추가"라고만 표기 — reward-worker에 코드 골격(예: `class_weight_table[term_name][env_class]` 텐서) 형태의 구체 설계 부재.

---

## Section 3. Should Fix (개선 권장)

### S1. **slope = staircase 근사 정당화 약함**
new_terrains plan §B3은 "ANYmal slope class도 동일 처리"라고 주장하지만 ANYmal Parkour 논문/코드는 heightfield slope를 사용. 출처 부정확. 또한 `parkour_terrains.py:144`의 edge_mask는 raycast 기반이라 **wedge primitive도 정확히 처리됨**(WARP raycast는 axis-aligned 제약 없음). 따라서 trimesh wedge가 더 적합할 수 있음 — 재검토 필요.

### S2. **stepping_stones Top-1 추천이 과적합 위험**
사용자 가치 "강건성·자연스러움·sim-to-real" 관점에서 stepping_stones는 *discrete jump 정밀도*만 학습. 반면 **balance_beam은 연속 lateral COM 제어**를 강제하며 robust gait 학습에 더 직접적이다. 또한 stones는 기존 `parkour_gap`(이미 `y_offset_range=(-1.2,1.2)` lateral platform 보유, mesh_terrains_cfg.py:345-349)과 **부분 중첩** — plan은 "1축 jump" vs "2D 패턴"이라 차별화 주장하나 gap도 lateral offset을 가짐. **Top-1을 balance_beam으로 재검토 또는 stones와 beam 동시 도입**(둘 다 Phase A) 권장.

### S3. **crawl plan §7 difficulty 매핑이 사용자 요구와 반대**
사용자 요구: "난이도↓ = 더 낮게 숙임" → 매핑식: `ceiling_z = min + difficulty*(max-min)`이어야 일관. plan §7은 IsaacLab 표준(난이도↑=hard)에 맞춰 `ceiling_z = max - difficulty*(max-min)`로 작성. 본인도 OQ 10.1로 인정. **Section 6 결정 질문 1번**으로 사용자 컨펌 필수.

### S4. **balance_beam 폭 0.20m vs height_scan res 0.1m**
plan I1이 인지하나 완화책이 "row 0은 0.50m"뿐. **187 차원 height_scan에서 beam이 1-2 cell만 잡힘 → critic도 beam 못 봄**. resolution 0.05m 업그레이드(obs 187→748) 또는 priv obs에 beam centerline 1-d scalar 추가 옵션을 plan에 명시 필요.

### S5. **rough_blocks edge_mask threshold 실측 부재**
new_terrains §I3-3에서 "Phase 3에서 sanity check"이라고만 적음. block height 0.02-0.10m가 normalized 0-255 uint8에서 어떻게 amplify되는지 사전 추정 없음. **threshold=20 → 80** 같은 사전 권장값을 추가하거나 첫 build 후 즉시 시각 검증 단계를 Phase 1에 포함.

---

## Section 4. Nice to Have

- N1. `corridor_width=1.2 m`(crawl §2) vs Go2 폭 0.3m → 사선 통과 회피 봉쇄 강도 정량 시뮬(예: max heading angle).
- N2. stepping_stones에 `lateral_jitter=0.30m`은 spawn FlatPatchSampling 패치 반경 0.5m와 거의 같음 — start platform 영역만 jitter 제외임을 함수 내 assert로 박아두면 좋음.
- N3. crawl ceiling_thickness=0.10m 위로 점프통과 봉쇄 옵션(`ceiling_top_extra=0.5m`)은 계산식 명시: Go2 점프정점≈0.5m이므로 `ceiling_z_low + ceiling_thickness + ceiling_top_extra > 1.0m` 권장.
- N4. zigzag_hurdles에서 forward-only command와 lateral 회피 trade-off 학습 가설 — `tracking_goal_vel`이 lateral 유도하므로 `lin_vel_y_range=[0,0]` 유지해도 OK라는 본인 주장은 합당하나, 첫 200 iter에 회피율 모니터 권장.

---

## Section 5. 통합 진행 권장 순서 (Phase 0 → N)

> 두 plan을 시간순으로 합치되, **terrain_class 번호와 proportion table은 사전 통합 결정**.

- **Phase 0 — 통합 계약 확정 (0.5일)**: `TERRAIN_CLASS_*` 번호표 사전 결정(예: stones=5, beam=6, slope=7, zigzag=8, rough=9, crawl=10), proportion 단일표 작성(sum=1.0 부동소수 정확), `_col_to_class` assert 통과 확인.
- **Phase 1 — Top-2 신규 지형 도입(stones + beam)** + 기존 reward로 sanity (1.5일): mesh + cfg + class 등록 + 200 iter PPO. crawl/slope/zigzag/rough는 미도입.
- **Phase 2 — Reward 함수 재구조화 (1일)**: `is_flat`/`is_non_flat` 이분을 `class_mask` 다중으로 일반화. 기존 동작 회귀 0%.
- **Phase 3 — crawl 도입 + edge_mask z-threshold 일반화 (2일)**: crawl plan §2-§3 mesh, **C2 수정안 적용**(WARP raycast hit_z 클램프), height_scan ceiling 마스킹.
- **Phase 4 — slope + zigzag + rough 추가 + reward 분기 확장 (2일)**: D3 표 단계 적용.
- **Phase 5 — Curriculum 튜닝 + 시각 회귀 (1.5일)**: 7개 신규 지형 success rate 모니터, debug_vis_edge_mask 시각.
- **Phase 6 — AMP 통합 검토 (별도 작업)**: 본 plan 범위 밖, 후속 PR.

---

## Section 6. 사용자 결정 질문 (3-5개)

1. **(Difficulty 매핑)** crawl/slope에서 "난이도↓ = 낮게 숙임/완만" (사용자 직관) vs "IsaacLab 표준 난이도↑=hard" 중 어느 쪽으로 통일하시겠습니까?
2. **(Top-1 우선)** 신규 지형 first-cut을 (a) stepping_stones 단독, (b) balance_beam 단독, (c) 둘 다(stones+beam) 동시 도입 — 어느 옵션이 좋습니까? 사용자 가치 "강건성/자연스러움" 기준 (c) 권장.
3. **(Crawl 우선순위)** crawl 지형을 신규 5종(stones/beam/slope/zigzag/rough)과 **동시 도입** vs **선행** vs **후행** 중 어느 순서로 진행할까요?
4. **(Slope mesh 방식)** staircase 근사(step_h=0.025m) vs 진짜 wedge primitive — 후자는 현재 raycast edge_mask가 정확히 처리 가능합니다. 어느 쪽을 선택?
5. **(Class 분기 reward)** new_terrains plan D3의 8-class 가중 표를 reward-worker에게 한 번에 패스 vs 후보별 단독 PR로 점진 도입 — 디버깅 편의 vs 통합 효율 트레이드오프, 어느 쪽?

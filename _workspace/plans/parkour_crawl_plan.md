# Parkour Crawl 지형 — Feasibility 및 구현 계획

본 문서는 IsaacLab Go2 / R_Skeleton parkour 환경에 "천장(overhang) 아래로 자세를 낮춰 통과하는 crawl 지형"을 추가하기 위한 read-only 분석/설계 문서다. 코드 변경은 포함하지 않으며 핵심 파일 경로와 라인 번호를 명시한다.

---

## 1. Feasibility 결론

**결론: Yes — with caveats.**

근거:

- **물리/충돌**: parkour 지형은 모두 `trimesh.creation.box(...)`로 생성되어 `TerrainImporter`가 단일 prim(`/World/ground`)으로 import한다(`source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_terrains.py:73-112`, `source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains.py:884`). 박스의 translation z를 양수로 띄우면 천장(overhang)이 그대로 static collider로 만들어지며, PhysX 측에서는 floor와 ceiling을 구분하지 않으므로 robot 등이 ceiling에 닿으면 정상적으로 contact force가 발생한다. 즉 "위에서 내려오는 장애물"은 추가 인프라 없이 mesh 한 장 추가로 구현 가능하다.
- **Mesh-only 통일**: 기존 5개 sub-terrain(`parkour_flat/hurdle/step/gap/stair`)이 전부 trimesh box 합성이므로, crawl도 동일한 패턴으로 들어가면 height-field/Genesis 등 별도 시스템을 도입할 필요가 없다.

Caveats(아래 4·5·6·7장에서 상세 처리):

1. RayCaster height_scan은 천장이 있는 cell에서 천장에 ray가 먼저 맞는다 → 정책이 "지면이 갑자기 솟았다"고 오인한다.
2. `compute_edge_mask_from_terrain_mesh`(`parkour_terrains.py:142-304`)의 face rasterization은 z>=horizontal_scale*0.5 인 모든 face의 max-z를 취하므로 천장도 `feet_edge` mask에 false-positive로 들어간다.
3. 현재 `orientation_l2`는 flat에서만 가중되지만(`parkour_env.py:643-645`), 자세를 강하게 낮추는 crawl 동작에서는 `dof_error_l2`/`hip_pos`가 nominal 자세 가정을 깨뜨린다.
4. AMP/모방학습 reference에 crawl pose가 없을 가능성이 높다.

---

## 2. Mesh 설계

**구조**: ground plane + 양옆 벽(선택) + ceiling box. 좌우 우회 차단을 위해 corridor 양옆을 벽으로 막는 것을 기본으로 한다(jump_hurdle은 의도적으로 양옆을 비웠지만, crawl은 우회 시 정책이 자세 낮추기를 학습하지 않으므로 차단 필수).

| 파라미터 | 값(권장) | 비고 |
|---|---|---|
| terrain size | (20.0, 4.0) | 기존 `PARKOUR_TERRAINS_CFG.size`와 동일 |
| corridor 폭(`corridor_width`) | 1.2 m | Go2 폭 ~0.3 m + 여유. 우회 봉쇄 |
| ceiling 두께 z(`ceiling_thickness`) | 0.10 m | 너무 얇으면 ray 노이즈, 너무 두꺼우면 시각 혼란 |
| ceiling 길이 x(`ceiling_length_x`) | 1.0 m | 1 발자국에 못 끝내는 길이여야 학습이 됨 |
| ceiling 시작 x | `platform_length` 뒤 1.0 m 부터 (n개 반복) | 첫 평지 패치는 spawn용 보장 |
| ceiling z_low (윗면 아래 평면) — `ceiling_height_range` | Go2: (0.18 m, 0.30 m), R_Skeleton: (0.45 m, 0.85 m) | 7장 참조 |
| 양옆 벽 z_high | 1.0 m, 두께 0.1 m | 우회 차단 |
| `num_crawls` | 3 (장기에는 4~6) | terrain x ≤ 20m 안에서 |
| 평지 마진(crawl 사이) | 1.5 m | 회복 / 다음 자세 준비 |

좌표 식(local frame, `mid_y = size[1]/2`):

```
ceiling_box dim = (ceiling_length_x, corridor_width, ceiling_thickness)
ceiling_box pos = (dis_x + ceiling_length_x/2, mid_y, ceiling_z_low + ceiling_thickness/2)
left_wall  dim = (ceiling_length_x, (size[1]-corridor_width)/2, wall_height)
right_wall dim 동일, y 반대편
```

원점은 `parkour_hurdle_terrain`과 동일하게 `(platform_length/2, mid_y, 0.0)`. spawn은 기존 FlatPatchSampling(patch_radius=0.5) 활용 가능 — 입구 직전 1 m 평지가 cfg.platform_length로 보장된다.

---

## 3. Goal 배치

`PARKOUR_GOALS_REGISTRY`에는 `(goals_local[num_goals,3], origin_local[3])`을 append 한다(`mesh_terrains.py:1037`, `parkour_terrains.py:137`). crawl 지형은 다음 규칙으로 goal을 둔다:

- **각 ceiling 직후 +0.3 m 지점, z=0** 에 goal 배치 → 점프 우회 방지를 위해 z=0 으로 고정. (점프하면 multi-step 지속이 어렵게 ceiling_z_low를 robot standing height 미만으로 설정)
- corridor 중앙 y(`mid_y`)에 모든 goal 정렬 → `tracking_yaw` 학습 단순화.
- num_goals < num_ceilings 인 경우 마지막 goal 으로 padding(기존 hurdle 동일 로직).

회피 방지 보강: corridor_width 를 1.2 m로 좁히고 양옆에 벽을 두면 점프 + 측면 회피가 모두 봉쇄된다. ceiling_thickness 0.10 m 위로는 충분히 두꺼운 z=1.0 m 벽이 없으면 점프 통과가 가능하므로, ceiling 위쪽으로 50 cm 더 올려 "위로도 막힌" 느낌을 주는 옵션(`ceiling_top_extra=0.5 m`)을 cfg에 둘지 Open Question(10장).

---

## 4. Observation 호환성 — height_scan

`parkour_env.py:264-271`의 `height_scanner`는 robot base 위 20 m 에서 `-z`로 ray를 쏘는 GridPattern(1.6 × 1.0 m, res 0.1)이다. 천장이 있는 cell에서는 ray가 천장 윗면에 먼저 맞아 `ray_hits_w[..., 2] ≈ ceiling_z_high` 가 들어온다. 이로 인해 `parkour_env.py:510-512`의 `scan = robot_z - hit_z - 0.5` 값이 **-1.0 으로 clamp** 되며 정책은 "지면이 매우 가깝다"고 오해한다.

**옵션 비교:**

| 옵션 | 설명 | 장점 | 단점 | 권장도 |
|---|---|---|---|---|
| (a) 마스킹: scan 정규화 단계에서 hit_z > robot_z + ε 이면 floor z=0으로 대체 | env 코드만 1줄 수정, 다른 지형 영향 없음 | 가장 단순, low-risk | 정책이 "ceiling 존재"라는 정보를 못 받음 → crawl 행동 발견 어려움 | △ 임시 안전망 |
| (b) RayCaster를 위→아래가 아니라 robot 발 끝(hip 아래) 부근에서 위로 따로 한 set 더 → ceiling_distance scalar 추가 | crawl class에서 천장까지의 clearance를 직접 관측 | 정책이 자세 낮추기를 명시 신호로 학습 | obs dim 추가, num_priv_obs/proprio 변경 필요 | ◎ 권장(주력) |
| (c) ceiling을 별도 prim으로 분리 + height_scanner.mesh_prim_paths 에서 제외 | ceiling이 floor 신호를 오염시키지 않음 | 깔끔 | TerrainImporter가 단일 prim 으로 합치므로 mesh를 분리 import 하는 별도 경로 필요 — 인프라 변경 큼 | × |

**권장**: **(a)+(b) 결합** — (a)로 scan 오염 즉시 차단, (b)로 ceiling clearance를 별도 1-d scalar(또는 5점 forward column ray) priv/scan에 추가. (b)의 obs dim 변경은 `num_proprio` 또는 `num_priv_obs`에 흡수해 critic-only 로 둘 수도 있음(asymmetric AC).

검증 필요: `RayCasterCfg`에 두 번째 instance를 add 했을 때 `Scene.sensors` 등록 충돌이 없는지(이름만 다르면 OK일 가능성 높지만 검증 필요).

---

## 5. Edge mask 호환성

`compute_edge_mask_from_terrain_mesh`(`parkour_terrains.py:142-304`)는 face rasterization 으로 height_field를 만들고 Laplacian 을 적용한다. L256-260 에서 `z < horizontal_scale*0.5` 면 ground face로 판단하고 skip 하지만, **z>=0 이면 모두 max-z 로 scatter** 한다. 결과: ceiling 윗면(z≈0.30 m)이 height_field 에 그대로 박혀 거기서 강한 Laplacian 응답이 발생, `feet_edge` 페널티가 ceiling 아래 corridor 의 경계 cell 에서 잘못 트리거된다.

**수정 위치**: `parkour_terrains.py:248-260` 의 face rasterization 루프에서 z 상한 필터 추가.

```python
# 추가될 정확한 위치: L256 직후, "z < horizontal_scale * 0.5" 분기 다음
ceiling_z_threshold = ...  # 인자로 받거나 환경마다 robot stand height + margin
if z > ceiling_z_threshold:
    continue   # ceiling face는 edge mask 에 반영 안 함
```

권장 임계값: Go2 = 0.45 m, R_Skeleton = 1.20 m (robot standing height 보다 충분히 높은 값). 호출부 `parkour_env.py:366-372`에서 `compute_edge_mask_from_terrain_mesh(...)`에 `ceiling_z_threshold` kwarg 를 cfg 로 주입.

또한 **height_field 자체**를 `parkour_env.py:379`의 `_edge_mask_height_field` 로 저장하는데, 같은 이유로 ceiling_z 가 포함되면 이 텐서도 오염된다 → 위 필터로 함께 정리됨.

---

## 6. Reward 설계

`parkour_env_cfg.py:127-134`에 `TERRAIN_CLASS_CRAWL = 5`를 추가. `_col_to_class` LUT(`parkour_env.py:104-115`)는 sub_terrains 의 dict insertion order 를 따르므로, 새 entry 를 dict 끝에 추가하면 자동으로 class=5가 된다. proportion 합 1.0 유지를 위해 기존 비율 재조정(예: gap 0.30→0.25, crawl 0.05~0.10 시작).

**`_get_rewards` 분기 수정안(`parkour_env.py:599-769`):**

```
is_flat   = (env_class == 0).float()
is_crawl  = (env_class == 5).float()
is_obstacle = 1 - is_flat - is_crawl   # hurdle/step/gap/stair
```

| 항목 | 현재 | crawl 분기 | 이유 |
|---|---|---|---|
| `orientation_l2` (L643-645) | flat에서만 `*1`, 그 외 `*0` | crawl 에서도 `*0` (그대로 두거나 약화) | crawl 에서는 pitch 큼 — 페널티 부적절 |
| `dof_error_l2` (L679-683) | flat `*10`, 그 외 `*1` | crawl 은 `*0.2` ~ `*0.5` | nominal 자세 강제 X |
| `hip_pos` (L670-676) | 모든 terrain 동일 | crawl 에서 `*0.2` 또는 0 | 자세 낮출 때 hip abduction 변동 큼 |
| `lin_vel_z_l2` (L635-638) | flat `*1`, 그 외 `*0.1` | crawl 도 `*0.1` 유지 | OK |
| `base_height` (L687-689) | flat-only | crawl 은 0 (기본 scale=0.0이므로 무시) | OK |
| **새 항** `ceiling_clearance_bonus` | — | crawl class 에서만 활성 | "robot top z + margin < ceiling z" 일 때 +reward |
| **새 항** `ceiling_collision_penalty` | — | base/back 부위와 ceiling collision 시 추가 음수 | 기존 collision은 `_undesired_contact_body_ids = [base, thigh, calf, hip]` 이므로 base로 ceiling 박는 건 이미 잡힘. 추가 가중 옵션. |

`reward_scales` 추가:

```
"ceiling_clearance":  +0.5,   # crawl class only
"ceiling_collision":  -2.0,   # 기존 collision 위에 surcharge (선택)
```

`is_crawl` mask 는 위 개별 항에 직접 곱한다(은닉 reward bug 방지: 매 step `_episode_sums` 에 0이라도 등록).

---

## 7. Curriculum 설계 — ceiling height 매핑

**IsaacLab 표준**: row 0 = difficulty 0 = easiest, row max = difficulty 1 = hardest (`parkour_env_cfg.py:42`). 사용자 요구는 "난이도가 높을수록 더 낮게 숙여야 함" 으로 해석(handoff doc Risks 섹션의 표준 합치 권고에 따름; 반대 해석은 Open Question 10.1 참조).

**ceiling 높이 보간(권장 — IsaacLab 정합)**:

```
ceiling_z_low = ceiling_height_range[1] - difficulty * (ceiling_height_range[1] - ceiling_height_range[0])
# difficulty=0 → ceiling_z_low = max (높음, easy)
# difficulty=1 → ceiling_z_low = min (낮음, hard)
```

**Go2** (default standing 0.34 m, base_height_target=0.34, hip joint ~0.30 m): `ceiling_height_range = (0.18, 0.30)`. 0.30 은 그냥 직립 보행으로 통과 가능(진짜 easiest), 0.18 은 hip joint flex + base lower 필요. 0.15 m 미만은 Go2 기구학상 통과 불가 → lower bound 0.17 floor.

**R_Skeleton** (humanoid, hip 약 0.85 m, head 1.6 m): `ceiling_height_range = (0.85, 1.40)`. 0.85 는 deep squat 보행, 1.40 은 가벼운 lean. lower bound 0.80 미만은 모방학습 reference 부재 시 학습 거의 불가.

**점프 불가 보장 — ceiling_thickness × ceiling_length_x**:

- ceiling_length_x ≥ 0.8 m 이면 Go2 가 한 번의 점프로 통과 못 함(점프 비행거리 ≈ 0.5 m). R_Skeleton 은 ≥ 1.2 m.
- 위쪽으로 점프 통과를 막으려면 ceiling 위에 추가 박스(z 1.0 m 까지) 또는 robot 이 ceiling 윗면에 안 닿도록 ceiling_z_low 자체를 robot 점프 정점 미만으로 설정(Go2 점프 정점 ~ 0.5 m).

**열거**:

| difficulty (row) | Go2 ceiling_z_low | R_Skel ceiling_z_low |
|---|---|---|
| 0.0 (easiest) | 0.30 m | 1.40 m |
| 0.5 | 0.24 m | 1.12 m |
| 1.0 (hardest) | 0.18 m | 0.85 m |

R_Skeleton 수치는 hip/head 측정 검증 필요(`source/isaaclab_assets/...` 또는 USD inspect 후 확정).

---

## 8. AMP / 모방학습 호환성

리스크:

1. **Crawl pose 부재**: 현재 AMP reference motion(있다면 `parkour/motions/*.npz` 등 — 검증 필요)에 hurdle/jump 만 포함되고 crawl(deep squat + forward locomotion)이 빠진 경우, discriminator 가 crawl 자세를 "non-expert" 로 분류해 reward 가 음의 방향으로 작용한다.
2. **Style reward와 task reward 충돌**: AMP style reward 가 nominal stance 만 보상하면 7·6장에서 추가한 `ceiling_clearance_bonus` 와 정면 충돌.

**Mitigation**:

- (Mit-1) Reference motion 확장: Mocap/Genesis 기반 crawl trajectory 를 추가 `.npz` 로 동봉. `motion-analyzer` worker 로 expert dataset 의 hip pitch / base z 분포를 먼저 확인.
- (Mit-2) Discriminator state subset 조정: AMP observation 에서 base z 와 pitch 를 빼고, joint pos 만 사용하도록 축소(스타일은 보존, 자세 자유도 확보). loss-worker 검토 필요.
- (Mit-3) Crawl class 에서는 AMP style coefficient 를 0.3~0.5x 로 줄이는 terrain-conditional weighting.
- (Mit-4) Pure RL warm-up: crawl class 만 AMP off + task reward only로 N k iter 선학습 후 AMP 재활성.

추가 검증: `source/isaaclab_tasks/.../parkour/` 에 AMP cfg 가 실제로 wired 되어 있는지(`agents/rsl_rl_amp_cfg.py` 등) 검증 필요(현재 repo는 PPO 만 보임, `agents/rsl_rl_ppo_cfg.py`).

---

## 9. 단계별 구현 계획 (Phase 1~5 체크리스트)

각 phase는 **rollback 단위**다. phase 1·2 만으로도 PPO 학습은 돌아간다.

### Phase 1 — Mesh + cfg 골격 (≈ 1 일)
- [ ] `source/isaaclab/isaaclab/terrains/trimesh/mesh_terrains_cfg.py` 에 `MeshParkourCrawlTerrainCfg(MeshParkourBaseTerrainCfg)` 신규 dataclass. 필드: `num_crawls, ceiling_height_range, ceiling_length_x, ceiling_thickness, corridor_width, x_spacing_range, y_offset_range, side_walls: bool=True`.
- [ ] `parkour_terrains.py` 에 `parkour_crawl_terrain(difficulty, cfg)` 추가 — 2장 스펙대로 mesh 생성, 3장 goal 등록, registry append.
- [ ] `parkour_env_cfg.py:43-122` `sub_terrains` 끝에 `"parkour_crawl"` entry 추가, proportion 재배분(예: gap 0.30→0.25, crawl 0.05). `TERRAIN_CLASS_CRAWL = 5` 상수 추가(L127-134).
- 검증: `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Go2-Parkour-Direct-v0 --num_envs 4 --headless=False` 로 spawn 시 ceiling box 시각 확인 + assert proportion sum == 1.0 통과.
- Rollback: cfg sub_terrains entry 한 줄 제거 + class 상수 제거.

### Phase 2 — Edge mask & height_scan 안전망 (≈ 0.5 일)
- [ ] `parkour_terrains.py:142` `compute_edge_mask_from_terrain_mesh` 에 `ceiling_z_threshold: float = 1.0` kwarg 추가, L256 직후 z 상한 필터 삽입.
- [ ] `parkour_env.py:366-372` 호출부에 `ceiling_z_threshold = robot_stand_height + 0.1` 전달.
- [ ] `parkour_env.py:510-512` height_scan 정규화에 ceiling 마스킹 1-line 추가(옵션 a).
- 검증: `_edge_mask` print log 의 `hf_max` 가 ceiling_z 가 아닌 hurdle 최대 z 와 일치. crawl row 의 `feet_edge` reward 가 ceiling 아래에서 spike 안 침(WandB).
- Rollback: kwarg default fallback.

### Phase 3 — Reward 분기 (≈ 1 일)
- [ ] `parkour_env.py:599-770` `_get_rewards` 에 `is_crawl` mask + 6장 표대로 분기 추가. 신규 `ceiling_clearance` term 구현(robot top z = root z + 0.10, ceiling z 는 cfg/registry에서 lookup — `_terrain_goals_world` 와 같은 방식의 ceiling map 별도 등록 필요).
- [ ] `reward_scales` 에 `ceiling_clearance` 추가, `_episode_sums` 자동 키 등록 확인.
- 검증: crawl class env 에서 orientation_l2 = 0, ceiling_clearance > 0 평균 (WandB log_dict).
- Rollback: 새 reward scale 0 으로 토글.

### Phase 4 — Optional ceiling-distance 관측 (≈ 1 일, 권장)
- [ ] 별도 RayCasterCfg(name="ceiling_scanner") — robot base 에서 +z 방향 짧은 ray pattern. priv obs 에 8 점 추가, `num_priv_obs` 14→22.
- [ ] `parkour_env.py:557-564` priv 텐서 cat 부분 + cfg dim 동기화.
- 검증: critic obs shape 일치 (`validate-code` worker).
- Rollback: cfg flag `enable_ceiling_obs: bool=False`.

### Phase 5 — Curriculum tuning + AMP 호환 (≈ 2 일)
- [ ] 7장 ceiling_height_range 를 robot 별로 적용. R_Skeleton 의 hip 높이 USD에서 측정.
- [ ] 8장 Mit-1~4 중 채택안 wired in. AMP cfg 가 없으면 본 phase 는 PPO 로 한정.
- 검증: Go2 PPO 4096 envs × 1500 iter, crawl class success rate(goal_reached) curve 모니터링. fail rate>50% 시 lower bound 상향.

---

## 10. Open Questions (사용자 결정 필요)

1. **Difficulty 매핑 방향**: handoff doc Risks 의 사용자 요구 "난이도 ↓ = ceiling ↓" 그대로 갈지, IsaacLab 표준(난이도 ↑ = 어려움)에 정합시킬지. 본 plan 은 후자(7장)로 가정 — 확정 필요.
2. **점프 회피 봉쇄 강도**: ceiling 위로 추가 box(`ceiling_top_extra`)를 둘지, 아니면 ceiling_z_low 만으로 점프 정점을 차단할지(Go2 점프 정점≈0.5 m).
3. **R_Skeleton 동시 지원 범위**: 본 phase 1~5 를 Go2 only 로 우선 끝내고 R_Skeleton 은 별도 PR 로 분리할지.
4. **Reward `ceiling_clearance` 형태**: continuous bonus 대신 binary "head clear" 가 안전 — 어느 쪽?
5. **AMP 활성화 여부**: 현재 repo 에 AMP가 wired-in 되어 있는지 확인 후, 없다면 8장 mitigation 은 향후 work 로 deferred.
6. **corridor_width 1.2 m vs 1.5 m**: 1.5 m 로 살짝 넓히면 Go2 가 사선 진입으로 ceiling을 부분적으로 회피할 수 있음 → 학습 난이도 감소. 정책 발견성 vs 회피 봉쇄 trade-off 결정 필요.

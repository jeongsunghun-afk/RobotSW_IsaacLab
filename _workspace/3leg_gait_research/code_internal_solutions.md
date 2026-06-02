# Go2 Parkour — 3-leg Gait: 코드 내부 유발 요소 분석 + Deploy-Safe 해결책

**작성자**: worker-code (team go2-3leg-gait-research, task #3)
**대상 코드**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/` (`parkour_env.py`, `parkour_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py`)
**분석 방식**: 코드 정적 분석 + 기존 reward-attribution 데이터(`_workspace/parkour_reward_attribution/results/*.npz`, commit 798d9b5) 재활용. **학습 실행 안 함.**

> 프레이밍 주의(메모 준수): "지형 극복 시 3-leg gait"는 **gait-level** 문제로 reframe됨. calf-divergence synthesis 인용 안 함. 아래 메커니즘은 모두 **가설(데이터로 뒷받침)** 로 표기하며 "확정 버그"로 단정하지 않음.

---

## 0. 핵심 실증 증거 (attribution npz, 재활용)

### 0.1 이전(≤5/27) 체크포인트에서 RL(Rear-Left) 발이 5개 롤아웃 모두에서 저(低)접지

> **데이터 범위 (중요, 검증됨)**: attribution npz는 `num_envs=5`(지형 클래스당 env 1개)로 캡처된 **지형별 단일 env 단일 롤아웃** (`foot_contact` shape `(T,4)`, DESIGN.md §6/`play_reward_attribution.py`). **4096-env 모집단 통계가 아님.**
>
> **데이터 출처 (Q3, 검증됨 — 정정)**: npz 파일 mtime은 **2026-05-27 10:55**, 내부에 `run_name/load_run/checkpoint` 키 **없음**. lead가 지목한 run은 **2026-06-01_14-06-08_use_default_usd** → npz는 그 run보다 **5일 앞선 별개(≤5/27) 체크포인트**에서 나온 것으로 보임(현재 use_default_usd run 데이터로 단정 불가). 또한 파일명 `ep730` 등의 숫자는 **per-terrain episode 길이(step 수)** 이지 학습 iteration/checkpoint 번호가 아님. → 따라서 접지율 0.18~0.28은 "**이전(≤5/27) parkour 체크포인트**"로 라벨해야 하며 use_default_usd run의 수치로 단정 금지. (3-leg 현상 자체는 use_default_usd에서도 재현된다고 보고됨 — 단 *어느 다리/정확한 수치*는 run별로 다를 수 있음.)
>
> 따라서 아래는 "정책 모집단이 항상 RL을 든다"가 아니라 "**이전 체크포인트가, 5개 서로 다른 지형/env 인스턴스 전부에서 RL을 덜 썼다**"로 해석. 재학습/현 run에서 같은 RL로 깨질지(아니면 RR/FL)는 **미검증**.

> **결정적 사실 확인 (team-lead 요청 Q1~Q3, 코드 검증):**
> - **Q1 (symmetry 구현 리스크)**: parkour에 좌우 미러 매핑 함수(`data_augmentation_func`) **없음**(grep 0건; env/cfg의 "mirror"는 무관 주석). 알고리즘 인프라는 지원함 — `PPOParkour`가 `symmetry_cfg` 받음(ppo_parkour.py:54-102), `OnPolicyRunnerParkour`가 `resolve_symmetry_config` 호출(on_policy_runner_parkour.py:337). 그러나 `RslRlSymmetryCfg.data_augmentation_func`는 `MISSING`(필수) → **"플래그만 켜기"는 에러/무효**. parkour obs 레이아웃(proprio 42 + height_scan 187 + priv + history 10×42)·12-dim action에 맞는 미러 매핑(좌↔우 관절 permutation+부호, height_scan 187 격자 좌우 반전, lateral cmd/yaw 부호)을 **신규 구현해야 함** → 해결책 #3은 중간~높은 난이도/리스크.
> - **Q2 (contact 신호 신뢰도)**: (a) terrain collider = **trimesh 확정**(env_cfg:457 generator + parkour_terrains.py `list[trimesh.Trimesh]`). (b) gait/air_time/dragging/contact reward는 **ContactSensor 경유**(raw `net_contact_force_tensor` 직접 아님) — 단 ContactSensor 내부가 `contact_physx_view.get_net_contact_forces()`(core contact_sensor.py:373)로 **동일 PhysX contact tensor를 래핑** → edge-underreporting에 **면역 안 됨**. ⇒ trimesh+(PhysX-tensor 상속) 조합이므로 contact 신호가 비대칭 왜곡될 *가능성 존재* → air-time 상한/gait_pairing fix가 역효과 낼 위험 있음. 단 §후보E 반대증거(평지서도 RL저접지·해당 항 가중치 ≤2e-3)로 *주 원인일 확률은 낮음*. **권장: 해결책 #1/#2 적용 전 §E 검증법(운동학 vs contact 접지 비교)을 먼저 1회 수행해 게이팅.**
> - **Q3 (수치 라벨)**: 위 "데이터 출처" 참조 — 0.18~0.28은 **이전(≤5/27) 체크포인트** 수치. use_default_usd run의 "이 체크포인트" 수치로 라벨하면 **부정확**. 라벨 정정함.

per-foot 접지 비율(contact fraction, 해당 롤아웃 전 step 평균):

| terrain | FL | FR | **RL** | RR |
|---|---|---|---|---|
| flat   | 0.66 | 0.53 | **0.28** | 0.64 |
| gap    | 0.61 | 0.48 | **0.24** | 0.56 |
| hurdle | 0.66 | 0.57 | **0.27** | 0.64 |
| stair  | 0.55 | 0.36 | **0.18** | 0.53 |
| step   | 0.65 | 0.62 | **0.27** | 0.65 |

→ **이전(≤5/27) 정책에서 RL 발만 접지율이 절반 수준**, 5개 지형 인스턴스 모두에서 일관(평지 포함). → 지형 특이적 물리 문제가 아니라 **학습된 비대칭(3-leg gait)**. 어느 다리로 깨지는지는 학습 seed별로 임의일 수 있으므로 "RL 특이"는 이 체크포인트 한정 사실. **핵심은 한 다리로 깨진다는 것 자체** (→ 해결책 #3 symmetry-aug의 우선순위를 높임). "Asset(USD) 비대칭" 가설과는 불일치(평지에서도 동일, `use_default_usd` run에서도 3-leg 재현 → asset 가설 추가 약화).

### 0.2 reward landscape는 task-tracking이 압도, gait shaping 사실상 0
per-term 평균 scaled reward(step당, flat 예시 — 타 지형도 동일 패턴):

```
tracking_goal_vel  +0.0298   ← 지배적 양수
tracking_yaw       +0.0079   ← 두 번째
action_rate_l2     -0.0017
feet_gait_pairing  +0.0016   (flat에서만, is_flat gate)
dof_error_l2       -0.0009
dof_acc_l2         -0.0008
feet_dragging      -0.0007
...나머지 모두 |값| ≤ 5e-4...
```

→ 정책을 형성하는 신호의 **~80%가 "goal 방향 전진 + goal 바라보기"**. 4발 대칭 보행을 **요구하거나 보상하는 항이 실질적으로 없음**. 비대칭 3-leg gait도 goal-tracking 목적을 거의 동일하게 만족 → 정책이 비대칭 해(解)로 수렴해도 reward 손해가 거의 없음.

---

## 1. 식별된 코드 내부 유발 요소 (후보)

### [먼저: air_time 무한 적립 버그 — 본 env에는 **없음**] (검증됨, 가설 기각)
- task 브리프의 최우선 단서(fc1b0a `feet_air_time threshold bug`)를 코드로 검증.
- `parkour_env.py`에서 raw air-time을 **직접 보상하는 항은 존재하지 않음**. `air_time`은 `feet_gait_pairing`(env.py:1148-1178)에서 **대각쌍 air/contact 시간의 동기(sync) 정도**를 exp(-제곱오차)로 보상하는 데만 사용 → 든 다리가 보상을 "무한 적립"하는 구조가 **아님**.
- **결론**: "영구히 든 다리가 air_time 보상 무한 적립" 메커니즘은 **현재 parkour env에는 적용 안 됨**(가설 기각). 그 버그는 다른 경로/과거 버전 이슈였음. → 무게를 아래 메커니즘으로 이동.

### 후보 A — gait shaping이 장애물 지형에서 완전 비활성 (`is_flat` gate) — **가설(데이터 뒷받침), 중간 신뢰**
- **위치**: `parkour_env.py:1178`
  ```python
  feet_gait_pairing = sync_reward * async_reward * gate * is_flat   # ← is_flat
  ```
- **메커니즘**: 유일한 양(+)의 gait-대칭 shaping 항이 `* is_flat`로 곱해져 **평지 외 모든 지형(hurdle/step/gap/stair/...)에서 0**. 정작 3-leg gait가 문제되는 장애물 지형에서 보행 형태를 잡아줄 reward가 전무. attribution 데이터에서 flat 외 전 지형 `feet_gait_pairing = +0.00000` 으로 확인.
- **제약 self-check**: contact는 reward 내부 사용(OK §1). obs 변경 없음. ✅

### 후보 B — `feet_gait_pairing` scale=0.0 (평지에서도 비활성) — **검증됨(설정값)**
- **위치**: `parkour_env_cfg.py:623` → `"feet_gait_pairing": 0.0`
- **메커니즘**: gate를 통과하는 평지에서조차 scale=0.0이라 학습에 기여 0. (attribution npz의 flat +0.0016은 캡처 당시 임시 비-0 scale 흔적으로 보이며 현재 cfg=0.0.) → **현 설정상 gait shaping은 모든 지형에서 사실상 OFF**.

### 후보 C — `feet_dragging`의 비대칭성: hind-only → **든 뒷발을 막는 항이 없음(permissive)** — **가설, 보조 요인(A/B보다 낮은 신뢰)**
- **위치**: `parkour_env.py:1137-1146`, scale `parkour_env_cfg.py:622` (-0.1)
  ```python
  hind_feet_ids = self._feet_ids[2:4]            # RL, RR 만
  hind_contact  = contact_filt[:, 2:4]
  is_dragging   = hind_contact & (hind_xy_vel_norm > thr)
  feet_dragging = sum(hind_xy_vel_norm * is_dragging)   # 접지+이동 시에만 패널티
  ```
- **메커니즘(정정)**: 이 항은 **든 다리를 *적극적으로* 보상하지는 않음**. 단단히 디딘 발(vel≈0)도, 공중에 든 발도 똑같이 dragging 패널티=0 — 즉 *드는 것*과 *제대로 디디는 것*을 구분하지 않음. 그리고 절약되는 노력은 미미함(`torques_l2`≈-1e-4/step). 따라서 C는 "드는 게 최적"이라는 **능동적 인센티브가 아니라**, 든 뒷발에 **불이익을 주는 항이 아예 없다는 *허용적(permissive)* 조건**. 진짜 driver는 **4발 접지에 대한 양(+) 보상의 부재(후보 A/B)**이며, C는 그 위에서 비대칭을 *방치*하는 보조 요인.
- **제약 self-check**: contact는 reward 내부(OK §1). actuator/토크 envelope blame 아님(§4,§5). ✅

### 후보 D — symmetry augmentation 부재 → 비대칭이 임의로 깨진 뒤 복원 안 됨 — **검증됨(부재), 중간 신뢰**
- **위치**: `agents/rsl_rl_ppo_cfg.py` 전체 — `symmetry`/`mirror` 키 **없음**(grep 재검증: parkour 디렉토리 전체 0건). git status의 `rsl_rl/.../symmetry.py` 수정은 **다른 env용**이며 parkour 학습에는 연결 안 됨.
- **메커니즘**: 좌/우 대칭을 강제하는 augmentation(mirror loss/data aug)이 없으면, **한 다리로의 임의적 대칭 파괴**(이 체크포인트에선 RL)가 **자기강화**되어도 이를 되돌릴 압력이 없음. "왜 한 다리만 드는 비대칭이 고착되느냐"에 대한 답이며, 데이터가 단일-롤아웃이라 "RL 특이"를 모집단 차원에서 단정할 수 없는 만큼 **어느 다리로 깨지든 무관하게 작동하는 #3가 더 중요해짐**.
- **제약 self-check**: 학습-시점 정규화만 변경, obs/deploy 영향 없음. ✅

### 후보 E — trimesh terrain edge에서 PhysX net-contact-force 누락/축소 → contact 기반 gait 신호 다리별 왜곡 — **미검증 가설 (web 단서, worker-web-isaac)**
- **출처**: legged_gym 원조 이슈 + IsaacLab Discussion #2697 — GPU + triangle-mesh terrain에서 `net_contact_force_tensor`가 삼각형 edge 위 contact force를 누락/축소하는 알려진 이슈.
- **코드 검증 (이번에 확인)**:
  - **(a) parkour terrain은 trimesh 맞음** — `parkour_env_cfg.py:457-460` `terrain_type="generator"` + `PARKOUR_TERRAINS_CFG`. sub-terrain들은 `parkour_terrains.py`에서 `list[trimesh.Trimesh]`(box/plane 합성, :41,:53,:90,:152)로 생성 → **삼각형 메쉬 collider 확정**.
  - **(b) gait/air-time/contact reward의 contact 측정은 전부 ContactSensor 경유** — `parkour_env.py:1040,1122,1133,1149-1150`이 `self._contact_sensor.data.net_forces_w_history / current_air_time / current_contact_time` 사용. 그런데 ContactSensor 내부는 `contact_physx_view.get_net_contact_forces()`(`contact_sensor.py:373`)로 **동일한 PhysX GPU contact tensor를 래핑** → ContactSensor를 써도 edge-underreporting 이슈에 **면역되지 않음(상속함)**.
- **메커니즘(가설)**: 어떤 발이 삼각형 edge 위에 착지하면 force가 threshold(2.0N) 미만으로 축소→접지 미검출→해당 발 air_time 계속 누적→gait/dragging/feet_edge/collision 신호가 **다리별로 비대칭 왜곡**. 정책이 "신호가 깨끗한 다리"만 쓰는 비대칭 보행으로 수렴할 수 있음. **obs 문제 아님, reward 신호 신뢰도 문제** → 제약 §1과 무관.
- **반대 증거(가설 신뢰도 하향 요인, 정직성)**:
  1. **다리 특이성 vs 공간 랜덤성**: edge-underreporting은 *발이 어디 착지하느냐*(공간적 확률)의 함수지 *어느 다리냐*가 아님 → 공간적으로 산발적인 miss를 만들지, RL 한 다리에 일관된 편향을 만들기 어려움. 관측된 "5개 지형 모두 RL 저접지"와 잘 안 맞음.
  2. **평지에서도 비대칭**: parkour_flat의 평탄 영역은 삼각형 *내부*(edge 아님)에 착지 → edge-underreporting이 작동 안 할 텐데도 RL 저접지(0.28) 지속.
  3. **가중치 leverage 부족**: 설령 contact가 왜곡돼도, 그 영향을 받는 항(feet_gait_pairing=0.0, feet_dragging/feet_edge/collision)은 reward 기여가 ≤2e-3 — tracking(+0.038) 대비 무시 가능. 거의-0 가중 신호의 신뢰도가 3-leg gait의 *주(主)* driver가 되긴 어려움.
- **종합 판단**: 메커니즘은 trimesh+ContactSensor-PhysX 경유가 코드로 확인돼 *성립 가능*하나, 위 반대 증거 3개로 **A/B(가중치/구조)보다 후순위**. 그래도 contact 신호 신뢰도는 termination(base_contact>5N)에도 영향을 주므로 배제는 말 것.
- **검증 방법(저비용)**: ContactSensor 접지 bool vs 운동학적 접지(발 z-height가 height_scan 지역 지면에 근접) 비교를 발별로 로깅 → 한 다리에서 *체계적으로* 불일치하면 edge-underreporting 시사. 또는 force threshold를 낮추거나 `force_matrix`/긴 history로 바꿔 RL 접지율 변화 관찰.
- **연관 mitigation(검증 후)**: contact-force 대신 **운동학 기반 stance/clearance 신호**(발 z vs height_scan 지면, 이미 `body_pos_w`/`body_lin_vel_w` 사용 중) 사용 — trimesh edge에 강건, deploy-safe(reward 내부). 단 설계 변경이라 후보 E 확인 후 검토.

### 비-원인으로 배제 (검증됨)
- **reset 초기 자세**: `parkour_env.py:1312` `joint_pos = default_joint_pos[env_ids]` — jitter 없는 **대칭** 자세. 비대칭 init 아님 → 원인 아님.
- **action 처리**: `parkour_env.py:560` `scaled[:, hip_joint_ids] *= 0.5` — 전 hip에 **대칭** 적용 → 원인 아님.
- **total_reward clip(min=0)** (`env.py:1229`): 메모상 **의도된 설계** → 버그 아님, blame 안 함. (다만 penalty gradient를 약화시켜 위 후보들의 효과를 더 키우는 *배경 조건*임은 언급만.)
- height_scan / actuator_mode=2 / 토크 envelope / latency: 메모 금지 항목, 의심 안 함.

---

## 2. Deploy-Safe 코드 해결책 제안 (우선순위순)

> 공통 제약 self-check: **(1) contact를 reward 내부에서만 사용, obs/proprio에 contact 추가 안 함 ✅  (2) scale 단독 튜닝 아님 — 모두 구조 변경(항 추가/gate 제거/augmentation)에 동반된 scale 설정 ✅  (3) latency/actuator/토크-envelope 근거 미사용 ✅**

### 해결책 #1 (권장, 가장 직접적) — 발별 air-time 상한 패널티 추가 (신규 reward 항, 구조 변경)
- **타깃**: 후보 C(+A/B) — "한 발을 영구히 든다"를 직접 금지.
- **구현**: `_get_rewards()`에 신규 항. `track_air_time=True`(cfg:582)라 `current_air_time` 사용 가능.
  ```python
  # _get_rewards 내부
  air_time = self._contact_sensor.data.current_air_time  # (N,4), 이미 위에서 로드됨
  # 정상 swing은 허용, 비정상적으로 오래 든 발만 패널티 (4발 대칭, 전 지형 적용 — is_flat gate 없음)
  feet_air_excess = torch.sum(
      torch.clamp(air_time - self.cfg.max_single_air_time_s, min=0.0), dim=1
  )  # (N,)
  reward_values["feet_air_excess"] = feet_air_excess
  ```
  cfg 추가: `"feet_air_excess": <음수 scale>`, `max_single_air_time_s: float = 0.5`(보수적; 큰 장애물 클리어 swing은 통과, 영구 거상만 차단).
- **왜 안전/유효**: contact-파생 데이터지만 **reward 내부 전용**(obs 미추가) → sim-to-real OK. 4발 모두 대칭 처리 → 특정 다리 편향 안 만듦. threshold를 넉넉히 잡아 정상 보행 swing은 비침해. **3-leg gait를 메커니즘 수준에서 직접 봉쇄**.
- **검증 방법**: 학습 후 §0.1 attribution 재실행 → RL 접지율이 0.18→타 발(0.5+)에 수렴하는지. 신규 항 episode_sum 로그가 학습 진행에 따라 0에 수렴하는지.
- **리스크/주의**: scale 과대 시 정상 swing 억제 → 보행 경직. → max_single_air_time_s를 먼저 넉넉히(0.5~0.8s), scale은 작게(-0.5~-1.0 수준) 동반 설정. **단독 scale 튜닝 아님(신규 항 도입)**.

### 해결책 #2 — `feet_gait_pairing` 재활성 + `is_flat` gate 제거 (기존 자산 재활용, 구조 변경)
- **타깃**: 후보 A, B — gait-대칭 shaping을 장애물 지형까지 확장.
- **구현**:
  - `parkour_env.py:1178`: `* is_flat` 제거(또는 비-flat에서도 일부 weight). 장애물에서도 대각 trot 동기를 보상.
  - `parkour_env_cfg.py:623`: scale 0.0 → 작은 양수(예 0.5~1.0)로 **동반 설정**.
- **왜 유효**: RL이 영구 거상이면 `contact_time[RL]≈0` ↔ FR cycle → `_async_pair`/`_sync_pair`의 제곱오차 ↑ → bonus 미획득. 즉 3-leg gait는 이 bonus를 못 받음 → 4발 trot로 유도. (bonus형 양수항 + clip(min=0)이라 신호가 약할 수 있어 **해결책 #1과 병행 권장**.)
- **검증**: 동일 §0.1 재실행 + feet_gait_pairing episode_sum이 비-flat 지형에서 상승하는지.
- **리스크**: 장애물에서 엄격한 trot 동기 강제가 험지 적응을 제약할 수 있음 → scale 작게 시작, std(`feet_gait_std=0.2`)는 관대 유지.

### 해결책 #3 — PPO symmetry augmentation 활성화 (구조 변경, deploy 무관)
- **타깃**: 후보 D — 좌/우 대칭 강제로 "한 다리만 드는" 대칭 파괴 자체를 차단.
- **구현**: `rsl_rl`의 symmetry 지원(`rsl_rl/rsl_rl/modules/symmetry.py`) 사용. agent cfg `algorithm`에 symmetry_cfg 추가 + Go2용 mirror permutation(좌↔우 관절/발 인덱스, 관측의 height_scan 좌우 반전, action 좌우 swap) 정의.
- **왜 안전**: **학습 시점 data/loss augmentation만** 변경. 관측 차원·deploy 정책 입력 불변 → sim-to-real 무영향.
- **검증**: 학습 곡선 안정성 + §0.1에서 좌(FL/RL)·우(FR/RR) 접지율이 대칭화되는지.
- **리스크/주의**: mirror permutation 정의 오류 시 학습 붕괴 → Go2 관절 순서(URDF [FL,FR,RL,RR]) 기준 매핑을 신중히 작성하고, height_scan 187 패턴의 좌우 대칭축을 정확히 잡아야 함. 중간 난이도.

### (참고) 우선순위·조합
- **즉효·저위험**: #1 단독 도입이 가장 타깃 명확. → #2 병행으로 양(+)방향 gait 유도 보강.
- **근본(임의적 대칭 파괴 방지)**: #3. 데이터가 단일-롤아웃이라 "어느 다리로 깨질지"를 모집단 차원에서 단정 못 하는 점이 오히려 #3(다리 무관)의 가치를 높임. 단 구현 난이도/리스크가 가장 큼 → #1/#2 효과 확인 후 도입 권장.
- 세 해결책 모두 **§1 contact-obs 금지, scale-단독 금지, actuator/latency/토크-envelope blame 금지**를 위반하지 않음(self-check 통과).

---

## 3. 한계 / 미검증 사항 (정직성)
- 위 메커니즘은 **정적 코드 + 1회 attribution 스냅샷**에 근거한 **가설**. 인과 확정은 제안 fix 적용 후 §0.1 재측정으로만 가능.
- attribution npz는 특정 체크포인트(ep~700) 1회 캡처 — 학습 후반 동작과 다를 수 있음.
- `feet_gait_pairing` flat=+0.0016 vs cfg=0.0 불일치는 캡처 시점 cfg가 현재와 달랐던 것으로 추정(미확정).
- 해결책 효과의 정량 예측은 안 함(학습 미실행).
- 후보 E(trimesh edge contact-force 누락)는 **미검증 가설**. 메커니즘 성립 가능성은 코드로 확인(trimesh+ContactSensor가 PhysX contact tensor 상속)했으나, 인과는 §E 검증 방법(운동학 vs contact 접지 비교)으로만 확정 가능. 반대 증거 3개로 A/B보다 후순위로 표기.
```

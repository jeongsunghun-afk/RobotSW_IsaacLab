# hind_leg Body Name Matching 정합성 리뷰
**작성일**: 2026-06-02  
**검토 대상**: `hind_leg_env.py` (active 경로: `HindLegHistoryEnvCfg`)  
**방법**: 소스 URDF 직접 읽기 + isaaclab 코어 코드 직접 읽기

---

## 1. 결론 한 줄

`.*_Foot_Contact_Link` 패턴은 실제 body 2개(`HL_Foot_Contact_Link`, `HR_Foot_Contact_Link`)에 정확히 매칭되며, `resolve_matching_names`는 unmatched 패턴에 **ValueError를 raise** 하므로 매칭 실패 시 `__init__`에서 즉시 crash한다. `_feet_ids` 자체는 정상이다.  
`penalzied_body_names`의 `"HR_Foot_Link_Link"` (Link 중복 typo)는 실제 body 이름 `"HR_Foot_Link"`와 다르므로 **__init__ crash를 일으킨다**.

---

## 2. `resolve_matching_names` raise 여부

[검증됨: `source/isaaclab/isaaclab/utils/string.py:259-268`]

```python
# string.py:260
if not all(keys_match_found):
    ...
    raise ValueError(
        f"Not all regular expressions are matched! ..."
    )
```

`find_bodies`(contact_sensor.py:180)는 `resolve_matching_names(name_keys, self.body_names, preserve_order)`를 직접 호출한다.  
[검증됨: `source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py:180`]

**동작 규칙**: 패턴 리스트에서 단 하나라도 매칭되는 body가 없으면 `ValueError` raise. 빈 리스트를 조용히 반환하지 않는다.

---

## 3. 로봇 실제 body(link) 이름 목록

[검증됨: URDF 직접 읽기 `/home/lgb/RobotSW_Genesis/genesis/assets/urdf/Hind_Leg_URDF_v3/urdf/Hind_Leg.urdf`]  
[검증됨: USD 변환 설정 `config.yaml:9` — `merge_fixed_joints: false` → 고정 조인트 링크도 USD body로 보존]

### 전체 link 이름 (11개)

| # | Link 이름 | Joint 타입 | 비고 |
|---|-----------|-----------|------|
| 1 | `base` | — | root body |
| 2 | `HL_Hip_Link` | revolute | `HL_Hip_Joint` child |
| 3 | `HL_Thigh_Link` | revolute | `HL_Thigh_Joint` child |
| 4 | `HL_Calf_Link` | revolute | `HL_Calf_Joint` child |
| 5 | `HL_Foot_Link` | revolute | `HL_Foot_Joint` child |
| 6 | `HL_Foot_Contact_Link` | **fixed** | `HL_Foot_Contact_Joint` child |
| 7 | `HR_Hip_Link` | revolute | `HR_Hip_Joint` child |
| 8 | `HR_Thigh_Link` | revolute | `HR_Thigh_Joint` child |
| 9 | `HR_Calf_Link` | revolute | `HR_Calf_Joint` child |
| 10 | `HR_Foot_Link` | revolute | `HR_Foot_Joint` child |
| 11 | `HR_Foot_Contact_Link` | **fixed** | `HR_Foot_Contact_Joint` child |

관절 이름 (8개 actuated): `HL_Hip_Joint`, `HL_Thigh_Joint`, `HL_Calf_Joint`, `HL_Foot_Joint`, `HR_Hip_Joint`, `HR_Thigh_Joint`, `HR_Calf_Joint`, `HR_Foot_Joint` → `action_space=8` 정합.

`merge_fixed_joints: false`이므로 USD에 11개 body 모두 존재.  
ContactSensor `prim_path="/World/envs/env_.*/Robot/.*"` 는 모든 body를 감지 대상으로 등록.

---

## 4. r2s_hind_leg 비교 결과

r2s_hind_leg는 **완전히 다른 로봇**(`R_SKELETON_HIND_LEG_CFG`, 5관절)을 사용하며 ContactSensor 자체가 없다.  
body name oracle로 사용 불가 — 이 환경은 비교 대상에서 제외.

[검증됨: `r2s_hind_leg_env_cfg.py:20` — `from isaaclab_assets.robots.rga import R_SKELETON_HIND_LEG_CFG`]  
[검증됨: `r2s_hind_leg_env_cfg.py` 전체 — `ContactSensorCfg` 없음]

---

## 5. 발견 이슈 목록

### Issue 1: `"HR_Foot_Link_Link"` typo — __init__ crash
**심각도**: CRITICAL (환경 시작 불가)  
**라벨**: [검증됨]  

`HindLegHistoryEnvCfg.penalzied_body_names`(cfg:272):
```python
"HR_Foot_Link_Link"   # ← typo: "Link" 중복
```
실제 body 이름: `HR_Foot_Link` [검증됨: URDF]

`find_bodies(["base", ..., "HL_Foot_Link", "HR_Foot_Link_Link"])` 호출 시  
`HR_Foot_Link_Link`에 매칭되는 body가 없으므로 `resolve_matching_names`가 `ValueError` raise.  
`HindLegHistoryEnvCfg` (active 경로)에서만 발생. `HindLegFlatEnvCfg`는 다른 body 이름 목록 사용.

**수정**: `"HR_Foot_Link_Link"` → `"HR_Foot_Link"`

---

### Issue 2: `.*_Foot_Contact_Link` — 정상 매칭, 이슈 없음
**심각도**: 없음  
**라벨**: [검증됨]  

패턴 `.*_Foot_Contact_Link`는 `HL_Foot_Contact_Link`, `HR_Foot_Contact_Link` 2개에 매칭.  
`merge_fixed_joints: false`로 두 body 모두 USD에 보존됨.  
`_feet_ids`는 올바르게 2개 index를 반환한다.

---

### Issue 3: `find_bodies("base")` — 정상, 단일 매칭
**심각도**: 없음  
**라벨**: [검증됨]  

`base`는 URDF root link 이름. 정확히 1개 매칭.  
`_base_id`는 길이 1 리스트. `_get_dones()`/`_get_rewards()`에서 `[:, :, self._base_id]` 인덱싱 정상.

---

### Issue 4: `HindLegFlatEnvCfg.penalzied_body_names` — 잘못된 body 이름 목록
**심각도**: HIGH (FlatEnvCfg 사용 시 __init__ crash)  
**라벨**: [검증됨]  

`HindLegFlatEnvCfg.penalzied_body_names`(cfg:122-146):
```python
"FL_link_1", "FL_link_2", ..., "HR_link_5"  # 23개
```
이 이름들은 URDF에 존재하지 않는다. 실제 body 이름은 `HL_Hip_Link`, `HL_Thigh_Link` 등.  
`HindLegFlatEnvCfg`로 환경 시작 시 `find_bodies()` 호출에서 `ValueError` crash.  
현재 active 경로(`HindLegHistoryEnvCfg`)는 이 목록을 사용하지 않으므로 현재 학습에는 영향 없음.

---

### Issue 5: `HindLegEnv` class annotation이 Flat/Rough만 선언
**심각도**: LOW  
**라벨**: [검증됨: hind_leg_env.py:23]  

```python
class HindLegEnv(DirectRLEnv):
    cfg: HindLegFlatEnvCfg | HindLegRoughEnvCfg   # HindLegHistoryEnvCfg 누락
```
`HindLegHistoryEnvCfg`가 타입 annotation에 없어 IDE/pyright가 History cfg 속성을 미검출.  
기능상 런타임 오류는 아니나 `isinstance(self.cfg, HindLegRoughEnvCfg)` 분기가 History cfg에서 False로 떨어지는 것은 의도된 동작으로 보임(history cfg는 flat terrain 사용).

---

### Issue 6: `termination_reward_scale` — History cfg에만 존재
**심각도**: LOW (오탐 방지 메모)  
**라벨**: [검증됨]  

`HindLegHistoryEnvCfg`(cfg:327)에는 `termination_reward_scale = -100.0`이 있지만  
`HindLegFlatEnvCfg`에는 없다. `_get_rewards()`(env:247)에서 `self.cfg.termination_reward_scale`을 참조하므로  
`HindLegFlatEnvCfg`로 실행 시 AttributeError가 추가로 발생한다. (Issue 4와 동일 경로에서 먼저 crash하므로 독립 영향은 없음.)

---

## 6. 요약 테이블

| 패턴 / 이름 | 실제 매칭 | crash 여부 | 심각도 |
|------------|----------|-----------|--------|
| `"base"` | `base` (1개) | 없음 | — |
| `".*_Foot_Contact_Link"` | `HL_Foot_Contact_Link`, `HR_Foot_Contact_Link` (2개) | 없음 | — |
| `"HR_Foot_Link_Link"` (History cfg) | **매칭 없음** | **ValueError crash** | CRITICAL |
| `"FL_link_1"` 등 (Flat cfg) | **매칭 없음** | **ValueError crash** | HIGH |

---

## 7. 수정 권고 (read-only 리뷰어 메모 — 수정은 별도 worker)

1. `HindLegHistoryEnvCfg.penalzied_body_names`의 `"HR_Foot_Link_Link"` → `"HR_Foot_Link"` 로 수정 (cfg 파일 272번 줄)
2. `HindLegFlatEnvCfg.penalzied_body_names` 전체를 실제 URDF body 이름으로 교체: `["base", "HL_Hip_Link", "HL_Thigh_Link", "HL_Calf_Link", "HL_Foot_Link", "HR_Hip_Link", "HR_Thigh_Link", "HR_Calf_Link", "HR_Foot_Link"]`
3. `HindLegEnv.cfg` 타입 annotation에 `HindLegHistoryEnvCfg` 추가 (선택)

---

**참조 파일**:
- `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env.py`
- `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/hind_leg_env_cfg.py`
- `/home/lgb/IsaacLab/source/isaaclab/isaaclab/utils/string.py` (resolve_matching_names: L178-271)
- `/home/lgb/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py` (find_bodies: L170-180)
- `/home/lgb/RobotSW_Genesis/genesis/assets/urdf/Hind_Leg_URDF_v3/urdf/Hind_Leg.urdf` (ground truth)
- `/home/lgb/IsaacLab/source/isaaclab_assets/data/Robots/Hind_Leg/config.yaml` (merge_fixed_joints: false)

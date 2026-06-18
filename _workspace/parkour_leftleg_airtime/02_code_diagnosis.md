# Go2 Parkour — 3-leg gait (왼쪽 다리 들림) 코드 구조 진단

**작성**: diagnose-worker (Task #2) — 측정 독립, 코드 구조 분석 only
**대상 증상**: air_time_cap penalty 추가 후에도 정책이 한쪽(왼쪽) 다리를 든 채 보행(3-leg gait)
**범위**: 왜 다리 들림이 구조적으로 가능/유지되는가. 측정값은 Task #1 소관.
**파일**:
- env: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`
- cfg: `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py`
- 센서: `/home/lgb/IsaacLab/source/isaaclab/isaaclab/sensors/contact_sensor/contact_sensor.py`
- 로봇: `/home/lgb/IsaacLab/source/isaaclab/.../isaaclab_assets/.../robots/unitree.py`

표기: **[VERIFIED]** = 소스 라인 직접 확인 / **[UNVERIFIED]** = 가설(미검증)

---

## 요약 (Top findings)

1. **[VERIFIED] air_time_cap의 escape 메커니즘이 구조적으로 성립한다.**
   penalty 입력으로 쓰는 `current_air_time`은 발이 **단 1번이라도 force>1.0N으로 닿으면 즉시 0으로 리셋**된다(contact_sensor.py:432-433, force_threshold=1.0 cfg:45). 들린 다리가 1.0s 안에 한 번씩 살짝만 tap 해도 누적 air_time이 1.0s를 넘지 못해 penalty=0 → **회피 경로가 코드상 열려 있다.**

2. **[VERIFIED] 다리 들림은 단순 "처벌 부재"가 아니라 contact-penalty 회피로 능동적으로 보상된다.** `feet_dragging`·`feet_edge`·`feet_stumble` 3개 penalty는 모두 **접촉 게이트**라 발이 공중에 있으면 사라진다. 발을 드는 것 자체가 이 penalty들에서 그 발을 빼주는 **reward-positive 유인**이다. 특히 `feet_dragging`은 **hind-only(RL/RR)**, `feet_edge`는 **terrain_level>3** 게이트 → 병리가 나타나는 hind-leg·고난도 지형과 정확히 일치. air_time_cap은 4발 대칭이라 이 유인을 상쇄하지 못하고, 그나마도 1번으로 무력화된다.

3. **[VERIFIED] '뒷다리(hind)' 성향은 구조적으로 부분 설명된다 — L/R 방향만 emergent.** `feet_dragging`이 hind-only(RL/RR)라 뒷다리를 들면 hind drag penalty를 회피한다(Q2b). 기본 자세는 거울 대칭(L hip +0.1 / R hip −0.1), 인덱싱 버그 부재. 좌/우 중 어느 쪽인지는 **emergent symmetry-breaking**이며 코드가 강제하지 않는다. asset(USD) 질량/관성 비대칭은 **[UNVERIFIED]** L/R 편향 후보.

4. **[VERIFIED] feet_gait_pairing은 scale=0.0 + flat-terrain gated** 라 병리 발생 지형(비평지)에선 어차피 꺼져 있다. 좌우 동기화를 책임질 후보였으나 구조상 비활성.

---

## Q1. air_time 누적/리셋 — escape 메커니즘 성립 여부

### 결론: **[VERIFIED] 성립 (회피 경로 열림)**

**penalty 계산** (parkour_env.py:1205-1208):
```python
feet_air_time = air_time[:, self._feet_ids]            # (N, 4)
air_time_cap = torch.sum(
    torch.clamp(feet_air_time - self.cfg.air_time_cap_max_s, min=0.0), dim=-1
)   # max_air_s = 1.0
```
여기서 `air_time`은 line 1149에서 가져온 **`self._contact_sensor.data.current_air_time`** 그대로다 (line 1200 주석도 명시: "Reuses air_time already fetched ... above").

**센서의 current_air_time 갱신 로직** (contact_sensor.py:422-433):
```python
is_contact = torch.norm(net_forces_w, dim=-1) > self.cfg.force_threshold   # (422)
...
self._data.current_air_time[env_ids] = torch.where(
    ~is_contact,
    self._data.current_air_time[env_ids] + elapsed_time,   # 공중이면 누적
    0.0,                                                    # 닿으면 즉시 0   (432-433)
)
```
- `force_threshold` 기본값 = **1.0 N** (contact_sensor_cfg.py:45) [VERIFIED]
- 센서 `update_period=0.005`s(=200Hz), `track_air_time=True` (parkour_env_cfg.py:581-582) [VERIFIED]

**메커니즘 성립 근거**:
- `current_air_time`은 "현재 연속 공중 체류 시간"이다. **접촉 force가 1.0N을 한 순간이라도 넘으면 그 즉시 0으로 리셋**되고 다시 0부터 누적한다.
- air_time_cap은 `clamp(current_air_time − 1.0, min=0)` 이므로, **연속 공중 시간이 1.0s를 넘는 구간에서만** 양수가 된다.
- 따라서 정책이 들린 다리를 **1.0s 주기 안에 한 번씩 force>1N으로 톡 찍어주면** current_air_time이 1.0s를 넘기 직전에 0으로 리셋 → **air_time_cap이 영원히 0** → penalty 회피.
- 이는 debounce/필터와 무관하게 성립한다. (참고: `feet_dragging`(1144)·`feet_edge`(1196)는 2-step OR debounce인 `contact_filt`를 쓰지만, **air_time_cap은 debounce를 거치지 않은 raw `current_air_time`을 쓴다** — escape는 raw 센서 리셋 자체에서 발생.)

**부가 구조 약점 (penalty 자체가 약함)** [VERIFIED]:
- weight=−0.1, value 단위=초과 초(seconds). 들린 다리가 2.0s 연속 공중이어도 step 기여 = `−0.1 × step_dt(0.02) × (2.0−1.0)= −0.002/step`. cfg:627 주석도 "ceiling의 3%"로 인정.
- 최종 보상은 `torch.clip(total_reward, min=0.)` (parkour_env.py:1241). total이 양수 영역(전진 잘 될 때)에서 −0.002를 빼는 marginal 효과가 작고, total이 음수면 penalty 한계효과가 clip으로 소멸.
  - ⚠️ 메모리 제약 §3 준수: 이 clip은 **의도된 설계(A/B 공통)**로 root-cause/제거 대상이 아님. 여기서는 "penalty 한계효과 약화"라는 **구조 관찰**로만 기술하며, weight 단독 튜닝을 처방하지 않음.

> **요약**: escape는 (a) raw `current_air_time` 사용 + 1N 즉시 리셋이라는 센서 의미론, (b) 1.0s 임계 + 약한 graded weight + min=0 clip 이 결합해 구조적으로 열려 있다. **둘 중 어느 경로가 실제로 발동하는지는 Task #1 측정 필요** (왼발 air_time 분포가 1.0s 부근에서 잘리는가 vs 1.0s 초과 꼬리가 두꺼운데도 penalty가 작아 무시되는가).

---

## Q2. 좌우/균형을 강제하는 reward 존재 여부 + **다리 들기를 유인하는 reward 존재 여부**

### 결론(2a): **[VERIFIED] 좌우 균형/한 다리 들기 직접 penalize 항 부재.**
### 결론(2b): **[VERIFIED] contact-penalty 회피가 다리 들기를 능동적으로 보상한다 (가장 중요한 발견).**

reward_scales 전체 16항 (parkour_env_cfg.py:607-631) 및 계산부(parkour_env.py:1211-1229) 점검:

| term | 좌우 대칭성 | 한쪽 다리 들림 억제? |
|---|---|---|
| tracking_goal_vel / tracking_yaw | 전진/yaw만, 발 무관 | ✗ (3-leg로도 전진 가능하면 만족) |
| lin_vel_z_l2, ang_vel_xy_l2, orientation_l2 | body 기준, 좌우 무관 | 간접(자세). 비평지서 orientation=0 (1082) |
| dof_acc/action_rate/delta_torques/torques_l2 | 전관절 합, 대칭 | ✗ |
| hip_pos (−0.5) | default 거울대칭(±0.1) → L/R 대칭 penalty | 간접: 든 다리 hip 편차 처벌하나 thigh/calf 들림은 hip_pos로 안 잡힘 |
| dof_error_l2 (−0.04) | 전관절 합, 대칭 | 약하게 default 자세로 당김 |
| feet_stumble / feet_edge | 4발/접촉 기반 | ✗ |
| feet_dragging (−0.1) | **hind RL+RR sum, 좌우 동일가중**(1140-1146) | ✗ (들린 발은 접촉이 없어 dragging 자체가 안 잡힘) |
| feet_gait_pairing (0.0) | 비활성 | ✗ (Q4) |
| air_time_cap (−0.1) | **4발 대칭**(1204-1208) | △ (Q1에서 무력화 가능) |

**핵심 부재** [VERIFIED]:
- 좌우 stance/swing 대칭, 좌우 접촉 시간 균형, "모든 발이 주기적으로 접지" 같은 **좌우 균형 신호가 명시적으로 없다.**
- 들기를 직접 막을 항은 air_time_cap이 유일 → Q1로 무력화.
- ⚠️ 메모리 §1 준수: 좌우 균형 신호를 **policy observation/contact obs로 추가하는 제안은 하지 않음**(sim-to-real). reward 내부에서 `_contact_sensor` 사용은 허용 범위(Task #3에서 검토 가능).

### 2b. **다리 들기를 능동적으로 보상하는 경로 (contact-penalty avoidance)** [VERIFIED 경로]

질문은 "들기를 막는 신호가 있나"뿐 아니라 **"들기를 유인하는 신호가 있나"**를 포함한다. 답: **YES.** 3개 penalty가 **접촉 게이트**라 발이 공중이면 0이 되고, 따라서 발을 드는 것이 그 발을 penalty에서 빼주는 **reward-positive 행동**이 된다:

| penalty | 접촉 게이트 (parkour_env.py) | 들었을 때 효과 | 비대칭 게이트 |
|---|---|---|---|
| `feet_dragging` (−0.1) | `hind_contact = contact_filt[:,2:4]` 필요 (1144-1146) | 공중이면 그 발 dragging=0 | **hind-only (RL,RR)** |
| `feet_edge` (−1.0) | `feet_at_edge = contact_filt & feet_at_edge` (1196) | 공중이면 그 발 edge=0 | **terrain_levels>3 only** (1197) |
| `feet_stumble` (−1.0) | `feet_forces≈0` for 들린 발 (1122-1126) | 공중이면 그 발 stumble 미발생 | `any`-aggregate라 더 약함 |

**정량(부호/스케일만, 메모리 §3 준수 — 처방 아님)**:
- `feet_edge`: `−1.0 × step_dt(0.02) = −0.02/step` per edge-contacting foot. `tracking_goal_vel` max(`1.5×0.02=0.03/step`)의 **~2/3**. 한 발의 edge 접촉만 피해도 전진 보상의 큰 몫과 맞먹음 → **marginal 아님.**
- 이 회피 유인은 **air_time_cap(−0.002/step 수준, Q1)보다 크다.** 들기를 막는 약한 신호 vs 들기를 부추기는 강한 회피 유인의 불균형.

**구조적 함의**: 다리 들림은 "처벌이 없어서 방치"가 아니라 **회피로 적극 보상**된다. Task #3 fix 방향이 바뀐다 — air_time_cap만 패치하면 회피 유인이 그대로 남는다. **주의**: 회피 *경로*는 코드로 VERIFIED지만, 이것이 실제 operative driver인지는 **Task #1 측정**으로 확인(들린 발이 곧 dragging/edge penalty를 가장 많이 누적했을 발과 일치하는가).

---

## Q3. "왼쪽"이 같은 다리(RL=_feet_ids[2])로 일관되는가 — asset/indexing 비대칭

### 결론: **'hind(뒷다리)'성향은 코드로 부분 설명됨(hind-only dragging 회피 유인, Q2b). 'L vs R' 선택만 emergent.** indexing 버그 **[VERIFIED 부재]**, asset 비대칭 **[UNVERIFIED]**.

> **Q2b 반영 갱신**: 들린 다리가 rear-left(RL=hind)라면, `feet_dragging`이 **hind-only(RL/RR)**라는 점이 *왜 하필 뒷다리인지*를 설명한다 — 뒷다리를 들면 hind dragging penalty를 피한다. 따라서 "완전 emergent"가 아니라 **hind 성향 = 구조적 유인, L/R 선택 = emergent/asset 비대칭**으로 분리된다.

**인덱싱** (parkour_env.py:198-209) [VERIFIED]:
```python
self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")   # 순서 [FL,FR,RL,RR] 가정
hind_foot_names = [body_names[i] for i in self._feet_ids[2:4]]
assert all(("RL" in n) or ("RR" in n) for n in hind_foot_names)   # 런타임 검증 있음
self._gait_synced_pair_0 = (self._feet_ids[0], self._feet_ids[3])  # FL+RR
self._gait_synced_pair_1 = (self._feet_ids[1], self._feet_ids[2])  # FR+RL
```
- assert로 hind 인덱싱은 런타임 검증됨. drag penalty는 RL/RR을 **sum**(1146)하므로 특정 좌/우를 차별하지 않음 → **인덱싱이 왼쪽을 강제하지 않는다.**
- air_time_cap도 `air_time[:, self._feet_ids]` 전체 sum(1206) → 발 구분 없음.

**기본 자세 대칭** (unitree.py:161-166) [VERIFIED]:
```python
joint_pos = {
    ".*L_hip_joint": 0.1,    ".*R_hip_joint": -0.1,    # 좌우 거울 대칭
    "F[L,R]_thigh_joint": 0.8, "R[L,R]_thigh_joint": 1.0,  # 좌우 동일
    ".*_calf_joint": -1.5,                                  # 좌우 동일
}
```
- hip default가 +0.1/−0.1로 **거울 대칭** → hip_pos penalty도 L/R 대칭. **기본 자세에 좌측 편향 없음.**

**따라서**:
- 보상·기본자세·인덱싱 모두 좌우 대칭 → **어느 다리가 들릴지는 코드가 결정하지 않는다.** "항상 왼쪽"이 사실이라면 이는 학습 중 **대칭 깨짐(symmetry breaking)**이 한 방향으로 고착된 것이며, 보상 구조가 처벌하지 않기 때문에 그대로 유지된다(Q2).
- **[UNVERIFIED] 잠재 기여 요인 (코드로 확정 불가)**:
  1. asset USD/URDF의 좌우 **질량/관성/COM 비대칭** — 미세 비대칭이 초기 편향 시드로 작용 가능. (USD 바이너리 inspection 필요, 본 진단 범위 밖)
  2. 좌우 actuator gain/마찰 randomization seed의 우연적 편향.
  3. 초기 terrain/goal 배치의 비대칭.
  - 이들은 **가설**이며, "왼쪽 고착"의 **방향**을 설명할 뿐 **존재**(다리 들림 자체)를 설명하지 않는다. 다리 들림의 구조적 원인은 Q1+Q2다.
- ⚠️ 메모리 준수: asset 비대칭을 "확정 버그"로 push 하지 않음. §5(actuator 약화)·§6(토크 envelope) blame 금지 준수 — 위 가설들은 root-cause가 아닌 방향-편향 후보로만 제시.

---

## Q4. feet_gait_pairing 비활성 구조 확인

### 결론: **[VERIFIED] scale=0.0 AND flat-terrain gated — 병리 지형에서 이중으로 비활성.**

- scale: `"feet_gait_pairing": 0.0` (parkour_env_cfg.py:623) → reward 합산 시 기여 0 (parkour_env.py:1234).
- gating: `feet_gait_pairing = sync_reward * async_reward * gate * is_flat` (parkour_env.py:1178). `is_flat = (self._env_class == TERRAIN_CLASS_FLAT).float()` (1044).
- 추가 gate: `gate = (cmd_speed > 0.3)` (1177).

**구조 기술**:
- 비평지(step/stair/gap 등 병리 발생 지형)에선 `is_flat=0` → 항이 0. **scale을 켜더라도 비평지에선 작동하지 않는다.**
- 이 항은 trot 대각쌍 phase 동기를 유도해 좌우 균형에 기여**할 수도** 있었으나, 현재 구조상 (a) scale 0, (b) flat 한정 두 겹으로 죽어 있어 **3-leg gait 억제에 전혀 관여하지 않는다.**
- **silver bullet 단정 금지**: 단순히 scale을 켜는 것만으로 해결된다고 볼 근거 없음(비평지 gate 때문). 구조적 재설계(Task #3)에서 "비평지에도 적용되는 좌우 대칭/접지 신호"를 검토할 가치는 있으나, reward 내부 신호로 한정(obs 추가 금지, §1).

---

## 종합 인과 모델 (구조 관점)

```
[다리 들림이 보상되는 이유 — Q2b ★핵심★]
  feet_dragging/feet_edge/feet_stumble = 접촉 게이트.
  발을 들면 그 발이 이 penalty들에서 빠짐 → reward-positive 유인.
  feet_edge 회피 ≈ -0.02/step ≈ 전진보상 max의 2/3 (marginal 아님).
  + reward set에 좌우 균형/주기적 접지 강제 신호 부재 (Q2a).
        │
        ▼
[들림을 막지 못하는 이유 — Q1]
  air_time_cap = clamp(current_air_time − 1.0, min=0), 4발 대칭.
  current_air_time은 force>1N 접촉 시 즉시 0 리셋(센서 의미론).
  → 1.0s 내 micro-tap으로 회피 가능 (escape 경로 VERIFIED).
  + weight −0.1로 graded 기여 미미, total clip(min=0).
        │
        ▼
[하필 '뒷다리'인 이유 — Q3+Q2b]
  feet_dragging이 hind-only(RL/RR) → 뒷다리 들면 hind drag 회피.
  → 'hind 성향'은 구조적 유인으로 부분 설명됨.
  'L vs R' 선택만 emergent/asset 비대칭(UNVERIFIED).
        │
        ▼
[feet_gait_pairing이 못 구한 이유 — Q4]
  scale=0.0 + flat-gated → 병리 지형에서 이중 비활성.
```

**핵심**: 다리 들림의 **존재**는 Q2b(contact-penalty 회피 유인)+Q1(유일 억제 신호 air_time_cap의 escape)+Q2a(균형 신호 부재)로 구조적으로 설명되고 **경로는 [VERIFIED]**. **hind 성향**은 hind-only dragging으로 부분 설명, **L/R 방향**은 emergent. *경로 검증 ≠ operative driver 확정* — 어느 경로가 실제 작동하는지는 Task #1 측정으로 판별.

---

## Task #3(fix 제안서) 인계 메모 — 제약 준수 방향성

> 본 worker는 fix를 처방하지 않음. 아래는 구조 진단에서 도출된 **검토 후보**(메모리 제약 필터 통과분)이며, Task #1 측정으로 어느 경로가 실제 발동하는지 확정 후 채택할 것.

0. **★ contact-penalty 회피 유인 중화 (구조, 최우선)**: feet_dragging/feet_edge/feet_stumble의 "들면 0" 구조 자체가 들기를 보상한다. air_time_cap만 패치하면 이 유인이 남는다 → **per-foot 장기 비접촉을 escape-불가 방식으로 penalize** 하거나 **주기적 stance(접지)를 요구**하는 신호를 reward 내부에 신설 검토.
1. **air_time_cap escape 차단 (구조)**: penalty 입력을 raw `current_air_time` 대신 **debounce/누적 접지 이력 기반** 또는 **stride-level swing 시간**으로 바꿔 micro-tap 회피를 무력화. (reward 내부 _contact_sensor 사용 OK, obs 추가 금지)
2. **좌우 균형 신호 신설 (구조)**: 좌우 hind/전체 swing·contact 시간 대칭을 reward 내부에서 측정해 비대칭 penalize — **단, contact를 obs로 노출 금지(§1), reward 내부 only.**
3. **feet_gait_pairing 재설계**: 비평지에도 적용되는 형태로 gate 완화 + scale 활성 — silver bullet 아님, 보조 신호로만.
4. ⚠️ **금지**: weight/threshold 단독 튜닝만으로 종결(§3), actuator·토크·height_scan·action_latency blame(§4-7), contact obs 추가(§1).

---

### 인용 라인 색인
- parkour_env.py: 1149(air_time 취득), 1199-1208(air_time_cap), 1140-1146(feet_dragging hind), 1148-1178(gait pairing), 1177-1178(gate/is_flat), 1044(is_flat), 1108-1114(hip_pos), 1206(4발 sum), 1234/1241(scale·clip), 198-209(feet_ids·pairs·assert)
- parkour_env_cfg.py: 581-582(track_air_time), 607-631(reward_scales), 623(gait scale 0), 630(air_time_cap −0.1), 648(max_air 1.0)
- contact_sensor.py: 422(is_contact), 432-433(current_air_time 리셋 to 0)
- contact_sensor_cfg.py: 45(force_threshold=1.0)
- unitree.py(GO2_CFG): 161-166(init joint_pos 거울대칭 ±0.1)

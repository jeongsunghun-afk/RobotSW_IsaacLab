# Go2 Parkour — 전진 stride / 자연스러운 gait 새 해결안 제안서

**Task #3 / team parkour-stride-gait / worker-3**
**산출: 제안서 only.**

> ⚠️ **상태 헤더**
> - **사용자 confirm 전 코드 적용 금지** — 본 문서는 설계 제안이며 한 줄도 적용하지 않았다.
> - **신규 설계 (Genesis port 아님)** — worker-1이 grep으로 확정: stride/swing/clearance/positive-air_time 보상은 A(direct/parkour)·B(Isaaclab_Parkour/extreme_parkour, 주석상 "Genesis original") **양쪽 모두 부재**. 도입할 reference 구현이 없다. 따라서 메모리 §2의 "Genesis에 있으면 그대로 도입 우선" 경로는 **산출물 없음** → 본 제안은 A·B를 넘어서는 신규 항 설계다.
> - **가설은 "유력하나 미검증"** — B에도 stride 항이 없다는 사실은 "stride 보상 부재 = root cause" 단정을 약화시킨다(B의 실제 gait 품질은 unknown). 그래서 **검증 plan 0단계(play.py 시각 확인 + kinematic 로깅)에서 증상을 먼저 직접 확인**한 뒤에야 reward 항을 적용한다.

---

## 0. 전제 (worker-1·2 결과, 사실로 사용)

| 출처 | 확정 사실 |
|------|-----------|
| worker-1 | A 활성 양수 reward = `tracking_goal_vel`(+1.5)·`tracking_yaw`(+0.5) **둘뿐**. stride/swing/foot-clearance/step-length/positive-air_time 항 = **grep 검증된 부재**. `feet_gait_pairing`은 scale 0.0 비활성(+활성돼도 *위상 동기*이지 전진 변위 아님). `air_time`은 `air_time_cap` **penalty**로만 사용. |
| worker-1 | `tracking_goal_vel`(env.py:1057-1066)은 **root COM 수평속도 1개**만 입력 → 발이 shuffle/제자리 step이어도 COM이 goal로 전진하면 충족. stride와 shuffle을 보상상 **구분 불가**. |
| worker-1 | B(extreme_parkour)도 동일 14항 + 동일 `root_vel_w` COM 식 → shuffle 가능성은 A 고유 회귀가 아닌 reference 공유 구조. |
| worker-2 | shuffle proxy(`feet_dragging`)는 baseline(-0.0194)≈current(-0.0263) → fix 부작용 아닌 **원래 gait 특성**. stride/air-time **duration은 로그에 없음** → 신규 kinematic 로깅 필수. |
| worker-2 | **non-regression baseline (current contact_duty_v1 run, late avg)**: mean_reward **19.66** · episode_length **750.67** · terrain_level **5.94** · goal_reached **5.20** · tilt_term **0.084** · RL_calf mean **0.797** / **sampled-action** std **2.16**. |

> **지표 구분 (worker-2 주의 인용)**: RL_calf **std=2.16은 sampled-action std**(탐색 노이즈 포함)로 `04_result.md`의 **policy_std=0.547**(정책 출력 std)과 **다른 지표**다. 비-regression 판정 시 둘을 섞어 인용하지 말 것.

**Root-cause framing (advisor 교정 반영):** 증상은 locomotion 실패가 아니라 **gait shaping 영역**(terrain_level 5.94 / goal_reached 5.20 = 실제 장애물 넘는 중). `tracking_goal_vel`이 COM-velocity-only라 "깨끗한 stride로 유도하는 gradient"가 부재 → shuffle local optimum 충족 가능. **이것이 1순위 가설(유력·미검증).**

---

## 1. 후보 ranking 축 — **3-leg 회귀 메커니즘과의 중첩도** (advisor 핵심 지시)

직전 commit이 `contact_duty_deficit`로 **3-leg gait(들린 뒷다리) 회귀를 막 고친 상태**다(메모리 `project_parkour_3leg_reward_positive`). 따라서 후보 선정 기준은 "표준이냐"가 **아니라**, **방금 고친 회귀(per-foot 접촉/공중 비율)와 메커니즘이 겹치느냐**여야 한다.

| 후보 | 핵심 신호 | 3-leg 회귀(접촉 duty)와의 중첩 | regression 위험 |
|------|-----------|-------------------------------|-----------------|
| **A. foot-clearance (swing 높이)** | 발이 스윙 중 **얼마나 높이 드는가** | **직교** — 접촉 duty 비율을 건드리지 않음. shuffle="낮게 끄는 발"을 **직접** 겨냥 | **낮음** ✅ (단 지형-강건 정확 구현 시 버퍼 복잡도 ↑ — [빈틈 ①]) |
| **B. positive feet_air_time** | 발이 착지 전 **공중에 머문 시간** | **높은 중첩** — air↑ = contact↓. `contact_duty_deficit`(contact↑ 유도)와 **길항** | **중간** ⚠️ |
| **C. stride / fore-aft 변위** | 발이 base 기준 **앞뒤로 이동한 거리** | 직교(변위 측정) | 낮음, 단 **버퍼 기계 최다** → 구현 위험 |
| **D. feet_gait_pairing 활성화** | 대각쌍 phase **동기** | — | **dead-end (아래 §6)** |

→ **추천 1순위 = A (foot-clearance)**: 회귀와 직교 + shuffle=저-리프트 증상 직격. B는 "가장 표준"이나 방금 고친 회귀와 메커니즘이 겹쳐 **대안(2순위)**으로 강등하고 길항 분석을 명시한다.

---

## 2. 후보 A — foot-clearance (swing-height) 보상 ★추천 1순위

### 가설
shuffle = 발을 스윙 중 거의 들지 않고 끄는 동작. 스윙 발의 **지지면 대비 높이**를 target까지 **하한(lower-bound)으로 보상**하면, COM 전진을 충족시키되 발을 들어 옮기는 "깨끗한 step"으로 gradient가 형성된다. 접촉 duty 비율을 건드리지 않으므로 3-leg 회귀와 직교.

### 구현 위치 (현재 parkour_env.py 기준)
- 계산: `_get_rewards()` 내부, `feet_dragging` 블록(env.py:1143-1152) 직후 — `self._last_contacts` overwrite(1186) **이전**, `contact_filt`(1141)와 `body_pos_w`/`body_lin_vel_w`(이미 1148·1188에서 사용 중) 재사용.

> ### 🛑 [빈틈 ①, 헤드라인] "지형-강건 reference" 단언 **철회** — `ground_ref=접지 발 최저 z`는 계단/step에서 깨진다
> 초안의 "현재 접지 발들의 최저 높이를 지면 기준으로 사용 → 계단/경사에서도 일관"은 **틀렸다.** 이 기준은 평지에서만 맞고, **커리큘럼을 지배하는 비평지(stair/step/gap)에서 정반대로 취약**하다:
> - **계단 오름**: 앞발=높은 단, 뒷발=낮은 단 접지 → `ground_ref`=낮은 뒷발 z → 앞 스윙발의 측정 clearance가 **실제 스윙 lift와 무관하게** target clamp를 trivially 포화 → **gradient 죽음**(보상은 주는데 스윙 높이를 안 가르침).
> - **계단 내림**: 스윙발이 낮은 단으로 내려가는 **정상 step-down**인데, 높은 접지발 기준이라 clearance가 작게(혹은 음수→clamp 0) 측정 → **올바른 동작을 무보상/처벌**.
> → clearance는 반드시 **각 발 아래의 per-foot 로컬 지형 높이** 기준이어야 한다("로봇 전체 최저 접지발" 기준 **폐기**). 올바른 로컬 기준 두 경로:
>   - **(a) per-foot liftoff-z 버퍼**: 발이 직전에 떠난 단의 z를 기억(스윙 동안 그 단 대비 lift 측정). → **후보 C 수준의 신규 persistent 버퍼 + `_reset_idx` 초기화 필요.**
>   - **(b) height_scanner를 reward 내부 로컬 지형 참조로 사용**: 각 발 xy 아래 지형 z를 조회. (※ 이는 메모리 **§7 "height_scan blame 금지"와 무관** — §7은 height_scan을 *버그 원인으로 지목* 금지이고, 여기선 *로컬 지형 참조 용도*로 쓰는 **별개 사안**. 단 발 위치 직하 ray 샘플링 정확도 검증 필요.)
> → **따라서 §1에서 내세운 "A의 신규 버퍼 불필요 장점"은 정확한(지형-강건) 구현에선 사라질 수 있다.** 아래 코드 스니펫의 `ground_ref` 블록은 **평지 전용 초안**으로만 두고, 비평지 정확 구현은 (a)/(b) 중 택1로 **구현 단계에서 재설계**해야 한다. (이 재설계로 A의 구현 복잡도가 C에 근접 → ranking 재고 여지 있음. 단 회귀-직교성 장점은 유지.)

- **평지 전용 초안 스니펫(⚠️ 비평지에선 위 (a)/(b)로 교체 필수)**:
  ```python
  feet_z_w   = self._robot.data.body_pos_w[:, self._feet_ids, 2]      # (N,4)  [FL,FR,RL,RR]
  in_contact = contact_filt                                           # (N,4) bool (이미 1141)
  swing      = ~in_contact                                            # (N,4)
  # ⚠️ FLAWED on stair/step/gap — per-foot 로컬 지형(liftoff-z 버퍼 또는 height_scan)으로 교체할 것
  big = torch.where(in_contact, feet_z_w, feet_z_w.new_full((), 1e3))
  ground_ref = big.min(dim=1, keepdim=True).values                   # (N,1)  ← 평지에서만 유효
  ground_ref = torch.where(torch.isfinite(ground_ref) & (ground_ref < 1e2),
                           ground_ref, self._robot.data.root_pos_w[:, 2:3] - 0.30)
  clearance  = (feet_z_w - ground_ref).clamp(min=0.0)                 # (N,4) 지면 위 높이
  # 하한 보상: target까지만 가산(초과분 무보상 → 큰 도약 억압 안 함)
  rew_clear  = torch.clamp(clearance, max=self.cfg.foot_clearance_target)  # (N,4)
  # 스윙 + cmd 속도 gate (정지 시 발 드는 local optimum 차단)
  cmd_gate   = (torch.abs(self._commands[:, 0]) > self.cfg.foot_clearance_speed_gate).float().unsqueeze(-1)
  foot_clearance = torch.sum(rew_clear * swing.float() * cmd_gate, dim=-1)  # (N,)
  ```
- reward_values dict(env.py:1227-1246)에 `"foot_clearance": foot_clearance` 추가.
- cfg `reward_scales`(parkour_env_cfg.py:607-638)에 `"foot_clearance": <weight>` + 신규 파라미터 3개(`foot_clearance_target`, `foot_clearance_speed_gate`; speed gate는 cmd 하한 0.3과 일관되게 0.2 권장).
- **버퍼**: 평지 초안은 instantaneous(버퍼 무) 이나, **비평지 정확 구현 (a) 채택 시 per-foot liftoff-z 버퍼 신설 → `__init__` init + `_reset_idx`(env.py:1306-1320) 초기화 필수**(신규 buffer→reset 불변규칙).

> ### ⚠️ [빈틈 ②, A·B 공통] `_commands[:, 0]`이 전진속도라는 가정 — **구현 전 미검증, 확인 TODO**
> cmd_gate(`torch.abs(self._commands[:, 0]) > thr`)는 `[:, 0]`이 **전진속도**라고 가정한다. 그러나 worker-1은 `tracking_goal_vel`이 **goal 방향(`_target_pos_rel`) 기반**(env.py:1057-1066)이고 `commanded_speed=abs(self._commands[:, 0])`로 쓴다고 확정했다 — 즉 parkour command가 순수 velocity가 아니라 **goal-position/heading 혼합** 레이아웃일 수 있다. 인덱스 [0]이 전진속도가 아니면 gate가 엉뚱하게 걸려 보상이 무의미해진다.
> → **TODO(구현 전 필수)**: `_commands` 텐서 레이아웃(각 인덱스 의미)을 env 코드에서 직접 확인 → gate 인덱스 확정. 만약 [0]이 속도가 아니면, 속도 proxy로 `torch.norm(root_lin_vel_w[:, :2])` 또는 올바른 속도 인덱스로 교체. **이 검증 없이 적용 금지.** (A의 cmd_gate, B의 cmd_gate 양쪽 동일 적용.)

### 예상 학습 신호 변화
스윙 발 리프트↑ → shuffle 발이 보상 못 받음 → 발을 드는 step으로 분기. `feet_dragging`(접지+이동 처벌)과 **상보**: dragging은 끌기 처벌, clearance는 들기 보상 → 양측 신호.

### weight 산정 (단위 함정 주의)
- env.py:1251 `scaled = reward_scales[key] * step_dt * value`. **Episode_Reward/\*는 step_dt 이미 반영** → weight 산정 시 **재곱 금지** ([[feedback_parkour_reward_weight_units]]).
- `value` 추정: target=0.08m, 보통 1~2 발 스윙·평균 clearance가 target의 ~0.5 → value≈0.08×1.5×0.5≈0.06/step. `tracking_goal_vel` scaled≈1.5×0.02×0.8≈0.024/step. clearance를 그 ~1/4(0.006/step)로 두려면 weight≈0.006/(0.02×0.06)≈**5** 부근.
- **단, 위 value는 미측정 추정** → **검증 plan 0단계 로깅으로 실제 clearance 분포·스윙 발 수를 측정한 뒤 weight 확정**. 잠정 시작값 **+3 ~ +5**, 로깅 후 조정.
- clip(min=0)(env.py:1258)는 **양수 항 추가에 무해**(총합만 키움).

### 위험 / 부작용
- **parkour 도약 억압?** → ❌ 없음. `clamp(max=target)`로 **하한 보상**이라 target 초과 리프트(장애물 넘기)는 무보상일 뿐 **처벌 아님** → 큰 도약 자유.
- 지형 reference 오차: 접지 발 0개(전 발 공중, 도약 정점) 시 base-z fallback → 일시적이라 영향 미미.
- **3-leg 회귀**: 접촉 duty 미접촉 → `contact_duty_deficit`와 직교, 회귀 재유발 경로 없음. (들린 발은 스윙으로 보상받지만 contact_duty_deficit가 동시에 접지를 요구 → 발을 들었다 **내리는** 완전 step으로 수렴, micro-tap 아님.)

### 메모리 §1~§7 self-check
| § | 항목 | 본 후보 |
|---|------|---------|
| §1 | contact 정보 policy **obs** 추가 금지 | ✅ 준수 — `body_pos_w`/`contact_filt`는 **reward 내부**만 사용, obs cat 미변경 (deploy 미노출) |
| §2 | [해제됨] 새 reward는 가설+검증 동반 허용 | ✅ 가설+검증 plan 동반. Genesis 원본 부재 → 신규 설계 명시 |
| §3 | scale **단독** 튜닝 금지 | ✅ 신규 **구조 항** 추가(scale 단독 조정 아님). 기존 scale 불변, **`contact_duty_deficit` weight 인하 안 함** |
| §4 | action_latency blame | ✅ 미거론 |
| §5 | actuator_mode=2 blame | ✅ 미거론 |
| §6 | 토크 envelope Tier2 | ✅ 미거론 |
| §7 | height_scan 의심 | ✅ 미거론 (clearance는 body_pos 기반, height_scan 무관) |

---

## 3. 후보 B — positive feet_air_time (clamped) ☆ 대안 (2순위)

### 가설
발이 착지 직전까지 **target swing duration**을 공중에 머물면 보상 → 짧은 micro-tap/shuffle 대신 실제 스윙 주기를 갖는 step 유도. `track_air_time=True`(cfg:582)라 `last_air_time`·`compute_first_contact` 사용 가능. IsaacLab 표준 `feet_air_time` 계열.

### 구현 위치
- `_get_rewards()` 내 `air_time`(env.py:1155, `current_air_time`) 인근. 착지 이벤트 기반:
  ```python
  first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]  # (N,4)
  last_air      = self._contact_sensor.data.last_air_time[:, self._feet_ids]                    # (N,4)
  # ★ load-bearing clamp: per-foot 보상을 air_time_cap_max_s(1.0s) 아래에서 차단
  air_excess = torch.clamp(last_air - self.cfg.air_time_reward_thr,
                           min=0.0, max=self.cfg.air_time_reward_clamp)   # thr≈0.25, clamp≈0.35 (<1.0)
  cmd_gate   = (torch.abs(self._commands[:, 0]) > 0.2).float().unsqueeze(-1)
  feet_air_time = torch.sum(air_excess * first_contact.float() * cmd_gate, dim=-1)  # (N,)
  ```
- reward_values + cfg reward_scales(`"feet_air_time": <weight>`) + 파라미터 `air_time_reward_thr`, `air_time_reward_clamp`.

### ★ clamp이 footnote가 아니라 핵심 설계 (advisor 지시)
`clamp(max=air_time_reward_clamp ≈ 0.35s < air_time_cap_max_s=1.0s)`가 **load-bearing**이다:
- clamp 없으면 "발을 오래 들었다 톡 찍기(long-hold→tap)"가 보상을 **사는** 패턴이 됨 = `air_time_cap`을 무력화시킨 바로 그 **micro-tap escape**(메모리 `project_parkour_3leg_reward_positive`)를 **보상으로 부활**시킴.
- clamp으로 보상 상한을 cap 아래에 두면 긴 air-hold에 추가 이득이 없어 escape 유인을 차단.

### ★ contact_duty_deficit와의 길항(antagonism) — 정직한 평형 분석 (advisor 교정)
**feet_air_time과 contact_duty_deficit는 시너지가 아니라 길항이다.** 같은 per-foot 접촉/공중 비율을 **반대 방향**으로 당긴다:
- `contact_duty_deficit`: 접촉 EMA < 0.30이면 처벌 → duty **↑**. **0.30 초과 시 penalty 소멸**(clamp(min=0)).
- positive `feet_air_time`: air 보상 → contact↓ → duty **↓**.
- **0.30 위에서는 air-time 보상만 작동** → duty를 0.30 바닥으로 다시 끌어내림. 정상 trot 지지율 ~0.5–0.6 대비 **0.30 바닥에 발이 정박**하면 under-supported/bouncy gait = 방금 고친 3-leg 회귀를 **갉아먹을** 위험.
- 붕괴가 **보장**되진 않음(하중 물리·collision(-10)·termination이 독립적 접지 바닥을 제공) — 그러나 **실재 위험**이며, B를 쓰면 contact_duty와 air_time의 **평형점을 학습 후 반드시 측정**해야 한다(per-foot duty EMA 분포를 로깅).

### weight 산정
- value는 착지 step에만 nonzero(sparse). 착지 빈도×평균 excess 추정폭이 커서(per-step value ~0.005~0.02 추정 불확실) **0단계 로깅으로 착지빈도·last_air_time 분포 측정 후 확정**. 잠정 **+1 ~ +5**(sparse라 dense항보다 큰 weight 정상), 단위 재곱 금지.

### 위험 / 부작용
- **상기 길항 → 3-leg 회귀 갉음**(중간 위험). clamp으로 micro-tap escape는 막지만 평형 위험은 별개.
- hopping 유발 가능 → clamp + air_time_cap(상한) + contact_duty_deficit(하한)의 삼중 bound로 억제.

### 메모리 §1~§7 self-check
| § | 항목 | 본 후보 |
|---|------|---------|
| §1 | contact obs 추가 금지 | ✅ `_contact_sensor`는 reward 내부만, obs 미변경 |
| §2 | 새 reward 가설+검증 | ✅ 동반. 단 Genesis 원본 부재 → 신규 |
| §3 | scale 단독 튜닝 금지 | ✅ 신규 항. **contact_duty_deficit weight 불변(인하 금지)** |
| §4–§7 | latency/actuator/torque/height_scan | ✅ 전부 미거론 |

---

## 4. 후보 C — stride / fore-aft 변위 보상 (3순위)

### 가설
발이 **base 기준 앞뒤(fore-aft)로 이동한 변위**(liftoff→touchdown)를 보상 → 전진 step 직접 보상. 회귀와 직교.

### 구현 위치 / 비용
- **신규 persistent 버퍼 필요**: `self._foot_liftoff_xy_b`(N,4,2). liftoff 시점 발 위치(base frame) 저장, touchdown 시 변위 계산.
- `__init__`(env.py:169 인근)에 버퍼 init + **`_reset_idx`(env.py:1306-1320)에서 반드시 초기화**(신규 buffer→reset 불변규칙).
- base frame 변환·liftoff/touchdown edge 검출 로직 추가 → **세 후보 중 코드 기계 최다 = 구현 위험 최고**.

### 평가
직교적이고 "전진 step"을 가장 직접 겨냥하나, 버퍼·edge-detection 복잡도와 base-frame 변환 버그 표면이 커서 **A로 잔존 시에만** 고려. self-check는 A와 동일(§1: 발 위치 reward 내부 사용 OK / §3: 신규 항).

---

## 5. weight 단위 함정 — 전 후보 공통 가드

env.py:1250-1251:
```python
scaled = self.cfg.reward_scales[key] * self.step_dt * value
self._episode_sums[key] += scaled        # Episode_Reward/* 로깅 = step_dt 이미 반영
```
→ Episode_Reward/\* 측정값으로 weight를 산정할 때 **step_dt(0.02) 재곱 금지**. (메모리 `feedback_parkour_reward_weight_units`: 이전 `contact_duty_deficit` -15는 step_dt 이중곱 50× 과대 → clip(min=0)로 학습 붕괴.) 측정값끼리 직접 비교.

---

## 6. 후보 D — feet_gait_pairing 활성화 (dead-end, 문서화)

env.py:1184 `feet_gait_pairing = sync_reward * async_reward * gate * is_flat`. **`* is_flat` gate가 parkour(비평지)에서 항을 0으로 만든다** = 우리가 관심 있는 바로 그 지형에서 무기여. 게다가 위상 동기일 뿐 전진 변위 아님. → **scale만 올려선 무효**. is_flat gate 제거는 unknown-risk 설계 변경이라 본 라운드 제외. **재제안 금지(이 dead-end를 명시 기록).**

---

## 7. 비-regression 제약 (필수)

**성공 = (stride/자연스러움 ↑) AND (아래 baseline 유지):**

| 지표 | baseline | 허용 | 비고 |
|------|----------|------|------|
| **policy_std** (정책 출력 std) | **~0.547** | **~0.5 유지** | **★1차 가드** — 실제 3-leg 발산 시 무너진 지표(10609→0.547 history). std collapse/explosion = 회귀 신호 |
| mean_reward | 19.66 | 유지(±소폭) | |
| episode_length | 750.67 | 유지 | |
| terrain_level | 5.94 | 유지/상승 | |
| goal_reached | 5.20 | 유지/상승 | |
| tilt_term | 0.084 | 미악화 | |
| RL_calf (**sampled-action** std 2.16 / mean 0.797) | 3-leg 발산 재발 없음 | **보조 지표** | 탐색 노이즈 섞여 둔함 → policy_std 보조. policy_std와 **혼동 금지** |

> **★ [빈틈 ③] 1차 가드 = policy_std(~0.5)**, RL_calf sampled-action std(2.16)는 보조. 실제로 3-leg 회귀에서 발산한 것은 **policy_std**(정책 출력 std)이지 sampled-action std가 아니다. 두 지표는 다른 것(sampled std는 탐색 노이즈 포함)이므로 비-regression 판정의 primary는 policy_std 유지로 본다.

- **`contact_duty_deficit` weight 인하 제안 = §3 위반 + 3-leg 회귀 직접 위험 → standalone lever 금지.** 본 제안 어느 후보도 기존 scale을 낮추지 않는다(신규 항만 추가).

---

## 8. 검증 plan (필수 — stride 품질은 action_stats proxy로 해소 불가)

| 단계 | 내용 | 위험 |
|------|------|------|
| **0-a [최우선]** | **play.py**로 현재 `contact_duty_v1` 체크포인트 gait를 **시각 확인** → "shuffle/저-리프트" 증상을 **직접 검증**(가설 확정 전 필수). `./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Go2-Parkour-Direct-v0 --num_envs 32`. **★[빈틈 ④] B(reference) gait 분기점 동시 관찰**: 아래 callout 참조 | zero (reward 미변경) |
| **0-b [최우선]** | **신규 kinematic TB 로깅(보상 0, 관측만)**: ① per-foot **stride length**(base 기준 fore-aft 변위) ② **foot clearance** — **★per-foot 로컬 지형 기준**(각 발 직하 지형/직전 liftoff-z 대비, [빈틈 ①]의 flawed "최저 접지발" 기준 **금지**) ③ **last_air_time @ touchdown** ④ **착지 빈도**. → 다음 fix **전/후 정량 비교** + **weight 캘리브레이션 입력**(추정 weight 0.5~30 폭은 측정 필요의 신호). **TB scalar이지 obs 아님(§1 준수)**. **★검증은 flat이 아니라 stair/step/gap 지형에서 직접**(아래 callout) | zero |
| **1** | **후보 A(foot-clearance) 단독 적용** → 학습 → §7 gate + 0-b 지표로 검증 | 단일 변경(원인 분리) |
| **2** | 잔존 shuffle 시에만 B 또는 C. **B 적용 시 per-foot duty EMA 분포를 추가 로깅**해 contact_duty와의 평형점 확인(§3 길항) | — |
| 보조 | RL_calf action_stats(sampled std 2.16) + 4-leg stats = 3-leg 회귀 비-regression **보조** 지표 | — |

> ### 🛑 [빈틈 ④] 0-a에 B gait 분기점 — 가설을 무효화할 수 있는 결정적 관찰
> B(extreme_parkour reference)도 stride 항이 **없다**(worker-1 확정). 따라서:
> - **만약 B가 자연스러운 gait를 낸다면** → A와 reward가 동일한데 gait가 다른 것 → 원인은 **reward 부재가 아니라 non-reward 요인(obs 구성 / curriculum / spawn init / network)**. 이 경우 **본 제안의 "reward 추가" 접근 자체가 무효화**되며, 진단을 obs/curriculum/init/network divergence로 전환해야 한다.
> - **만약 B도 shuffle한다면** → "stride 보상 부재 = 공유 원인" 가설이 강화됨.
> → **0-a play.py에서 (가능하면) B 정책의 gait도 같이 관찰**해 이 분기점을 판정할 것. **B를 실행/관찰 불가하면** "B gait 품질 unknown = 가설 미확정"으로 명시하고, reward 추가를 **잠정**으로만 진행(0-a A 증상 + 0-b 정량 로깅으로 보강).

> ### 🛑 [빈틈 ① 연계] 0-b 로깅이 ground_ref 버그를 **숨길 수 있다** — 비평지 직접 검사
> clearance metric을 [빈틈 ①]의 flawed "최저 접지발" 기준으로 로깅하면, **계단에서도 metric이 "정상"으로 보여 버그를 그대로 reward에 박는다**(같은 flawed 기준이 로깅·보상에 공유되므로 self-consistent하게 거짓 정상). 따라서:
> - 0-b의 clearance metric은 반드시 **per-foot 로컬 지형 기준**(각 발 직하 지형 z 또는 직전 liftoff-z)으로 정의.
> - 검증은 **flat이 아니라 stair/step/gap 지형에서 직접** 수행 — 오름/내림 단에서 스윙 발의 측정 clearance가 실제 lift와 일치하는지 육안+scalar 교차 확인.

---

## 9. 적용 순서 (한 번에 1개 — 원인 분리)

```
0. play.py 시각 확인 + kinematic 로깅 추가 (zero-risk, 증상 확정 + weight 캘리브레이션)
        ↓  [사용자 confirm]
1. 후보 A (foot-clearance) 단독 적용 → 학습 → §7 gate 검증
        ↓  (잔존 시)
2. 후보 B (feet_air_time, clamp 필수 + 평형 로깅) 또는 C (stride 변위)
```

---

## 10. 핵심 추천 요약 (team-lead 보고용)

1. **1순위 = 후보 A (foot-clearance)** — 방금 고친 3-leg 회귀(접촉 duty)와 **직교**, shuffle=저-리프트 증상 직격, lower-bound clamp로 parkour 도약 억압 없음. 시작 weight +3~+5(로깅 후 확정). **⚠️ 단 [빈틈 ①]: `ground_ref=최저 접지발` 기준은 계단에서 깨짐 → per-foot 로컬 지형(liftoff-z 버퍼 또는 height_scan) 기준으로 구현 단계 재설계 필요 = "버퍼 불필요" 장점은 정확 구현 시 사라질 수 있음. 회귀-직교성 장점은 유지.**
2. **2순위 = 후보 B (positive feet_air_time)** — 가장 표준이나 air-time 메커니즘이 `contact_duty_deficit`와 **길항**(평형이 0.30 바닥에 정박 → 3-leg 갉을 위험). 채택 시 `clamp(max<1.0s)` **필수**(micro-tap escape 차단) + per-foot duty 평형 로깅.
3. **0단계(play.py + kinematic 로깅)는 어느 후보든 선행 필수** — 증상 직접 확인 + weight 캘리브레이션. action_stats만으론 stride 품질 판정 불가.

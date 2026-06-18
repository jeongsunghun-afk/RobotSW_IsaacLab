# 03 Fix Proposal — Go2 Parkour 3-leg gait (RL/왼쪽 뒷다리 들림)

**작성**: synthesis-worker (Task #3) — 제안서 only, 코드 수정 금지
**입력**: `01_measurement.md`(lead 실측, air_time_cap run 50k step) + `02_code_diagnosis.md`(diagnose-worker)
**전제**: 측정+advisor로 확정된 root cause(아래 §1)를 사실로 두고 작성. 모든 제안은 CLAUDE.md 메모리 §1~§7 self-check 통과분만 기재.
**상태**: **사용자 confirm 전 코드 적용 금지.** 각 fix는 가설+검증절차 동반(§2 준수).

---

## 1. Root Cause 확정 (재진술)

> **3-leg gait는 처벌 부재가 아니라 reward-positive 평형이다. air_time_cap은 escape 가능 + 회피 유인에 압도되어 무력하다.**

인과 체인 (측정+코드 VERIFIED):

1. **다리 들기가 능동적으로 보상된다 (driver).** `feet_edge`(-1.0)·`feet_dragging`(-0.1, hind-only)·`feet_stumble`(-1.0)는 모두 **접촉 게이트** penalty다(parkour_env.py:1133-1146, 1196). 발이 공중이면 그 발이 penalty에서 빠진다. 들린 RL 발의 회피 유인 합 ≈ **-0.054/step** 분량(feet_edge -0.034 + feet_dragging -0.020, late). tracking_goal_vel(+1.082)은 3-leg로도 유지 → **다리 들기 = 순(純) reward-positive**.
2. **유일한 억제 신호 air_time_cap이 escape 가능 + 미미하다.** penalty 입력이 raw `self._contact_sensor.data.current_air_time`(parkour_env.py:1149→1205)인데, 센서는 force>1.0N 접촉이 **단 1회**라도 발생하면 `current_air_time`을 즉시 0으로 리셋한다(contact_sensor.py:432-433, force_threshold=1.0). 들린 다리가 1.0s 주기 안에 한 번씩 micro-tap → air_time이 max_air=1.0s를 못 넘김 → penalty=0. 실측 기여 **-0.0033/step (mean_reward 19.6의 0.017%, 회피 유인의 1/16)**. episode 750 step으로 충분히 긴데도 안 걸림 → **weight/max_air가 약해서가 아니라 신호 자체가 escape 가능하고 회피 유인에 압도됨.**
3. **L/R 균형·주기적 접지 강제 신호 부재** (diagnose Q2a). feet_gait_pairing은 scale=0.0 + flat-gated 이중 비활성(Q4).

**결론**: fix는 **회피-escape가 불가능한 contact-duty 기반 신호**로 air_time_cap을 대체/보강하고(1순위), **회피 유인 자체를 중화**해야(2순위) 한다. weight/threshold 단독 조정(3순위)은 1·2순위 적용 후 미세조정으로만.

**calf 발산은 root cause가 아니라 downstream 증상** → §6에서 모니터링 지표로만 기술.

---

## 2. Fix 우선순위

각 fix 끝에 메모리 §1~§7 self-check 표 첨부.

---

### 1순위 — escape-불가 contact-duty 신호 (headline fix, 신호 설계)

> **⚠️ weight 단위 정정 (2026-06-05 구현 시):** 본 문서 초안의 `-0.3`과 회피 유인 `-0.054`를 "per-step"으로 표기한 것이 혼동을 유발했다. 실제 `Episode_Reward/*` 측정값은 `(Σ scaled)/max_episode_length_s`(env:1362)로 **step_dt가 이미 반영된 동일 단위**다. 측정 단위에서 contact_duty_deficit 기여 ≈ `|weight| × 0.28`(들린 발 1개, full episode), 회피 유인 0.054 초과 조건은 `|weight| > 0.19`. **구현 채택값 = -0.5** (기여 0.14 ≈ 2.6× 회피 유인, per-step penalty 0.0028 ≪ positive budget 0.022 → `clip(min=0)` 안전). **step_dt를 재차 곱하지 말 것** — 그 오류가 한때 -15(50×)를 도출했고, -15는 들린 발 env를 clip floor 0에 가둬 planting gradient를 소멸시킨다(학습 붕괴). 검증 run 후 -0.3~-1.0 범위 조정.

**가설**: air_time_cap이 무력한 근본 이유는 입력이 raw `current_air_time`이라 micro-tap 1회로 리셋된다는 **센서 의미론**이다. 입력을 **시간 평균된 per-foot 접촉비율(contact duty)** 로 바꾸면, micro-tap은 평균에 alpha(≈0.02)만 기여하므로 escape가 수학적으로 불가능하다. 한 발을 95% 들고 있으면 그 발의 duty는 tap 타이밍과 무관하게 ≈0.05로 남아 graded penalty를 받는다.

**왜 escape 불가능한가 (정량)**: 정책이 1.0s 주기로 1 step(0.02s) micro-tap → 50 step 중 1 step contact. EMA `duty ← α·contact + (1−α)·duty`, α = step_dt/τ. τ=1.0s면 α=0.02. 정상 평형 duty ≈ 0.02 (= tap 빈도). 이는 duty_target(예 0.30)에 크게 못 미쳐 deficit = 0.28 만큼 graded penalty. **micro-tap 빈도를 아무리 올려도 실제 접촉 시간 비율을 올리지 않으면 duty가 안 오른다 → 회피하려면 진짜 stance를 해야 함 = 원하는 행동.**

**구현 (어느 텐서/함수/cfg)**:
- **신규 buffer** `self._foot_contact_duty` shape `(num_envs, 4)`:
  - 선언/초기화: `parkour_env.py` __init__ 내 `self._last_contacts` 선언부(line 169 부근)에 인접 추가. 초기값은 **0이 아니라 `duty_target` (또는 그 이상, 예 0.5)** 로 둔다 → 에피소드 초반 warmup 구간에서 spurious penalty 방지(아래 위험 섹션).
  - reset: `_reset_idx`의 `self._last_contacts[env_ids] = False`(line 1291) 옆에 `self._foot_contact_duty[env_ids] = self.cfg.contact_duty_target`(또는 0.5) 추가. (메모리 규칙: 신규 buffer는 반드시 `_reset_idx` 초기화)
- **업데이트 + penalty 계산**: `_get_rewards` 내 contact 계산부(이미 `contact = norm(...) > 2.0` at line 1133, `air_time/contact_time` fetch at 1149-1150 존재) 직후, air_time_cap 블록(1199-1208) **대체 또는 병행** 위치에 삽입:
  ```python
  # per-foot raw contact boolean (force>thr). debounce 불필요 — EMA가 평균화.
  foot_contact = (torch.norm(net_contact_forces[:, 0, self._feet_ids], dim=-1)
                  > self.cfg.contact_duty_force_thr).float()          # (N, 4)
  alpha = self.step_dt / self.cfg.contact_duty_tau                    # τ→α
  self._foot_contact_duty = (alpha * foot_contact
                             + (1.0 - alpha) * self._foot_contact_duty)  # (N,4) EMA
  # graded deficit: 시간평균 접촉비율이 target 미만이면 부족분 처벌
  contact_duty_deficit = torch.sum(
      torch.clamp(self.cfg.contact_duty_target - self._foot_contact_duty, min=0.0),
      dim=-1)                                                          # (N,) ≥0
  ```
  - `net_contact_forces`는 이미 line 1122/1133에서 사용 중인 동일 텐서 재사용.
  - `self._foot_contact_duty` 업데이트는 in-place가 아니라 재대입이므로 detach 이슈 없음(reward 계산 경로, gradient 불필요한 buffer).
- **reward_values 등록**: line 1211 dict에 `"contact_duty_deficit": contact_duty_deficit` 추가.
  - ⚠️ **정합성 주의 (reward-worker 확인 필수)**: per-env breakdown 버퍼 `_last_reward_breakdown_per_env`는 `(num_envs, len(self.cfg.reward_scales))`로 할당된다(parkour_env.py:193-194). 따라서 **새 key를 `cfg.reward_scales`에 추가하면 버퍼 폭은 자동 확장되어 OOB crash는 없다.** 단 line 1238은 `reward_values.items()`의 enumerate `i`로 컬럼을 채우고 컬럼 라벨은 `_reward_term_names = list(cfg.reward_scales.keys())`(line 196)에서 온다 → **`reward_values`(env)와 `reward_scales`(cfg)의 key 삽입 순서가 동일해야** 라벨-데이터 정합. reward-worker는 두 곳에 **같은 위치/순서**로 추가하고, 1·2순위 두 term 모두 양쪽에 반영됐는지 validate-code로 확인할 것.
- **신규 cfg 파라미터** (`parkour_env_cfg.py`):
  - `reward_scales`(line 607)에 `"contact_duty_deficit": -15.0` (**평형을 뒤집도록 유도된 값 — 아래 판별식 참조**. 임의 초기값 아님).
  - `contact_duty_tau: float = 1.0` (s, EMA 시상수 ≈ 2~3 gait cycle)
  - `contact_duty_target: float = 0.30` (각 발이 1s 윈도우 평균 30% 이상 접지 요구. trot 지지발 duty≈0.5~0.7 / 3-leg 들린 발≈0 을 깔끔히 분리)
  - `contact_duty_force_thr: float = 2.0` (feet_edge와 동일 임계, line 1133과 일관)

**예상 학습 신호 변화 + weight 판별식 (★ escape뿐 아니라 "압도" 도 풀어야 함)**:

root cause는 escape **와** "회피 유인에 압도됨" 둘 다다(§1). EMA는 escape를 풀지만, weight가 작으면 air_time_cap의 *나머지* 실패(압도)를 그대로 반복한다. 따라서 weight는 **들린 발 상태를 net reward-negative로 뒤집는 최소값**으로부터 유도해야 한다(이 유도 자체가 신호 설계이며 §3 단독 튜닝이 아님).

판별식 (per-step scaled contribution, line 1234 `scale × step_dt × value` 단위 — 회피 유인 -0.054와 동일 단위):
```
|weight| × step_dt × (target − equilibrium_duty)  ≥  avoidance_incentive
|weight| × 0.02   × (0.30  − 0.02)                ≥  0.054
|weight| × 0.0056                                  ≥  0.054   →   |weight| ≥ 9.64
```
- `equilibrium_duty ≈ 0.02`: 1.0s당 1 step(0.02s) micro-tap 시 EMA 정상상태 = 1/50 = 0.02 (= α). 즉 들린 발이 escape를 위해 tap 해도 duty는 0.02 부근에 갇힌다.
- **선택 default `|weight| = 15` (margin 포함)**:
  - 완전 들린 발(duty 0.02): 기여 = `15 × 0.02 × 0.28 = 0.084/step`. 회피 유인 0.054와 합산 → **net = −0.084 + 0.054 = −0.030/step → reward-NEGATIVE** ✅ (들기가 손해가 됨 = 평형 전복).
  - 약한 미달(duty 0.25): 기여 = `15 × 0.02 × 0.05 = 0.015/step` (완만, 정상 발 과처벌 아님).
  - duty ≥ 0.30: 0.
- **주의(早期 학습)**: 회피 유인은 late 기준 -0.054지만 early feet_edge는 -0.076(01_measurement)이라 early 합 ≈ -0.106 > 0.084. early엔 들기가 아직 약간 net-positive일 수 있음 → terrain_level/penalty가 late 값으로 안정되며 -15이 우위. early까지 덮으려면 **-20** (`20×0.02×0.28=0.112 > 0.106`)로 상향. 검증 run에서 net-negative 여부를 직접 확인해 -15↔-20 결정.
- 대안 formulation(완만한 경계): 선형 deficit 대신 **deficit² (제곱 grading)** 사용 시 duty→0에서 급격·target 부근에서 완만 → gap 도약 등 일시적 저(低)duty 발의 false-positive 감소. 단 이 경우 net-negative를 맞추려면 weight ≈ -35 필요(`35×0.02×0.28²=0.055`). 경계 jumpiness가 문제되면 채택, 아니면 선형 -15/-20 유지.

**핵심 불변량**: 어떤 formulation을 쓰든 **선택한 default에서 "완전 들린 발이 net reward-negative"가 산술로 성립**해야 하며(위 판별식), 이를 "나중에 튜닝"으로 미루지 않는다. escape 차단(EMA) + 압도 해소(weight 유도)가 함께 충족되어야 1순위가 headline fix로 성립.

**검증 방법**:
- per-foot `_foot_contact_duty` 분포를 TB scalar로 로깅(env0 + per-leg 평균). 성공 = 4발 모두 duty ≥ ~0.25 수렴, 특정 발(RL) duty가 0 부근 고착이 사라짐.
- `contact_duty_deficit` term 기여가 air_time_cap과 달리 **유의미(mean_reward의 ≥0.5%)** 하게 잡히는지 확인 → escape 차단 입증.
- RL_calf action_stats(/mean, /std)가 정상범위(±1.5, std 0.4-0.7)로 회복하는지 = 들린 다리가 stance로 복귀했다는 downstream 확인(§6).
- per-leg contact-duty로 3-leg→4-leg 전환 정량 측정(가장 직접적인 성공 지표).

**위험/부작용**:
- **에피소드 초반 warmup**: duty EMA가 0에서 시작하면 첫 ~τ 구간 전부 deficit. → reset 초기값을 `target`(또는 0.5)로 두어 회피. 또는 `episode_length_buf > warmup_steps` 게이트 추가(보조).
- **정당한 긴 swing/도약 억압 우려**: gap 도약은 일시적 air지만 도약 전후 stance가 있어 1s 평균 duty는 유지됨. target=0.30은 도약 비율을 고려해 보수적으로 설정. 실측(정상 발 air p99=0.36s)상 정상 보행은 duty≫0.30 → false-positive 낮음. τ·target은 검증 run에서 조정.
- step_dt가 decimation 기반 0.02s 고정이므로 α 계산 안정적.

| self-check | §1 obs금지 | §2 가설+검증 | §3 구조설계 | §4 latency | §5 actuator | §6 torque | §7 height_scan |
|---|---|---|---|---|---|---|---|
| 1순위 | ✅ reward 내부 _contact_sensor만, policy obs 미노출 | ✅ 가설+4개 검증지표 | ✅ scale 단독 아님, 신규 escape-proof 신호 설계 | ✅ 무관 | ✅ 무관 | ✅ 무관 | ✅ 무관 |

---

### 2순위 — 회피 유인 중화: L/R contact-duty 대칭 신호 (구조)

**가설**: contact-gated penalty(feet_edge/dragging/stumble)가 "발을 들면 빠지는" 구조라, 들기 자체가 보상이다(diagnose Q2b). 1순위가 "절대 접촉량 부족"을 처벌한다면, 2순위는 **좌우 비대칭(한 발만 체계적으로 드는 것)** 을 직접 처벌해 3-leg 패턴을 정면으로 반대한다. 1순위 EMA duty 버퍼를 재사용하므로 escape-불가 성질을 그대로 상속.

**구현**:
- 1순위의 `self._foot_contact_duty` (N,4) 재사용. feet 순서 [FL(0),FR(1),RL(2),RR(3)].
  ```python
  # 좌우 쌍 contact-duty 비대칭 (front: FL vs FR, hind: RL vs RR)
  lr_front = torch.abs(self._foot_contact_duty[:, 0] - self._foot_contact_duty[:, 1])
  lr_hind  = torch.abs(self._foot_contact_duty[:, 2] - self._foot_contact_duty[:, 3])
  contact_duty_asym = lr_front + lr_hind   # (N,) ≥0, 정상 보행 대칭이면 ≈0
  ```
- reward_values에 `"contact_duty_asym": contact_duty_asym`, cfg reward_scales에 `"contact_duty_asym": -5.0` (아래 sizing 참조).
- **신규 buffer 없음** (1순위 buffer 재사용) → 추가 reset 불필요. (단 §reward_values 정합성 주의는 1순위와 동일하게 적용 — 두 term 모두 cfg+env 동일 순서 추가.)
- **weight sizing**: 3-leg에서 hind 비대칭 lr_hind ≈ 0.6(RL≈0, RR≈0.6). default `-5.0`이면 기여 = `5 × 0.02 × 0.6 = 0.06/step` — 회피 유인(0.054)과 동급으로 비대칭을 직접 처벌. **단 2순위는 1순위가 평형을 이미 뒤집은 뒤의 보조**이므로, 1순위 단독으로 3-leg가 해소되면 weight를 낮추거나 생략 가능. 1순위와 합산 시 과처벌(정상 보행의 순간 비대칭까지 억압) 여부를 검증 run에서 확인해 조정.

**예상 학습 신호 변화**: 정상 trot은 좌우 발 duty가 시간평균상 거의 같아 asym≈0 → 무처벌. RL만 드는 3-leg는 hind L/R duty 차이가 크게 벌어져(예 RL≈0, RR≈0.6 → lr_hind≈0.6) graded penalize. **air_time_cap의 4발 대칭과 달리, 비대칭을 직접 겨냥**하므로 "어느 발이든 하나를 체계적으로 드는" 행동을 정확히 반대.

**검증 방법**: lr_hind / lr_front scalar 로깅, 학습 진행 시 0으로 수렴하는지. 1순위와 함께 적용 시 3-leg→4-leg 전환 가속 여부.

**위험/부작용**:
- 의도적 비대칭이 필요한 지형(예: 한쪽으로 회전, 비대칭 stepping stone)에서 약간의 false penalty 가능. → weight를 1순위보다 작게(보조), target/τ로 평균 윈도우를 충분히 길게 잡아 순간 비대칭은 허용하고 **체계적·지속** 비대칭만 처벌.
- 1순위만으로 3-leg가 해소되면 2순위는 불필요할 수 있음 → **1순위 먼저 단독 검증 후, 잔존 시 2순위 추가**(원인 분리, 메모리 "한 번에 1개 변경" 정신).

| self-check | §1 obs금지 | §2 가설+검증 | §3 구조설계 | §4 latency | §5 actuator | §6 torque | §7 height_scan |
|---|---|---|---|---|---|---|---|
| 2순위 | ✅ reward 내부만 | ✅ 가설+검증지표 | ✅ 비대칭 구조 신호 신설 | ✅ 무관 | ✅ 무관 | ✅ 무관 | ✅ 무관 |

---

### 3순위 — 보조 lever (headline 금지, 1·2순위 적용 후 미세조정 only)

> 아래는 **단독 처방 금지**(메모리 §3, 그리고 lead 지시 "max_air 단순 scale 조정 headline 금지"). 1·2순위 신호 설계가 적용되어 escape가 막힌 **이후**에만, 효과 부족 시 조이는 lever로 언급한다.

1. **air_time_cap_max_s 1.0→0.6** (cfg:648) + **weight -0.1→-0.25** (cfg:630): cfg 주석(627-629)에 이미 "효과 부족 시" 명시된 경로. 단, **escape가 1순위로 막히기 전엔 무의미**(max_air를 낮춰도 micro-tap이 air_time을 리셋하면 여전히 0). 따라서 1순위 적용 후 보조로만.
2. **feet_gait_pairing 비평지 활성 검토** (cfg:623 scale 0.0, env:1178 `* is_flat`): 현재 scale=0 + flat-gated 이중 비활성(Q4). 대각 쌍 phase 동기를 유도하므로 3-leg를 **간접** 반대할 수 있다. 다만:
   - `is_flat` gate 제거 시 비평지(병리 발생 지형)에도 적용되나, gait pairing은 평지 trot 가정 신호라 거친 지형에서 **자연스러운 비주기 보행을 억압**할 위험.
   - **silver bullet 단정 금지**(diagnose Q4 결론과 일치). 1·2순위가 contact-duty를 직접 다루므로 이 항은 우선순위 낮음. 검토만, 채택 시 별도 가설+검증 필요.
3. **contact_duty target/τ/weight 튜닝**: 1순위 신호의 파라미터를 검증 run 결과로 조정 — 이는 신호 설계의 일부이므로 §3 위반 아님(단, 신호 도입 없이 이것만 만지는 건 금지).

| self-check | §1 obs금지 | §2 가설+검증 | §3 구조설계 | §4 latency | §5 actuator | §6 torque | §7 height_scan |
|---|---|---|---|---|---|---|---|
| 3순위 | ✅ | ✅ 1·2순위 후 보조로만 | ⚠️ **단독 금지** 명시, 신호 설계 후 미세조정 only | ✅ | ✅ | ✅ | ✅ |

---

## 3. 권장 적용 순서 (원인 분리)

1. **1순위만 단독 적용** → 검증 run. contact_duty_deficit 기여가 유의미해지고 RL_calf가 정상화되며 3-leg→4-leg 전환되는지 확인.
2. 잔존 시 **2순위 추가** (1순위 buffer 재사용, 저비용).
3. 그래도 부족하면 **3순위 lever**로 미세조정.

> 한 번에 1·2·3 동시 적용 금지(메모리: 원인 분리 불가). 각 단계 검증 후 다음.

담당 worker 매핑(제안):
- 1·2순위 reward 신호 신설 → **reward-worker** (`parkour_env.py` `_get_rewards` + buffer init/reset)
- cfg 파라미터(reward_scales, contact_duty_*, air_time_cap_*) → **cfg-worker** (`parkour_env_cfg.py`)
- 신호 도입 후 정합성(buffer shape, reset 초기화, reward_values dict 일치) → **validate-code**

---

## 4. calf 발산 — downstream 증상 (모니터링 지표, fix target 아님)

> **인과 확정(advisor)**: RL_calf action mean=+32517 / std=10609 (다른 11관절 ±1.5 / std 0.4-0.7 대비 4-5 자릿수)은 **applied action이 clip_actions=10.0(parkour_env.py:557)으로 항상 ±10 clip → reward에 무관 → gradient 0 → random walk + entropy로 std 폭발** 한 결과. 교과서적 "죽은 action 차원". 인과 방향 = **다리 들기(reward 평형) → calf 방치(증상)**. "calf를 못 써서 다리를 든다"의 부활 아님.

- **fix 대상 아님.** actuator 약화(§5)·calf joint limit·토크 처방(§6) **절대 금지**.
- **성공 지표로만 사용**: 1·2순위 fix 적용 후 RL이 stance로 복귀하면 RL_calf_joint가 다시 학습 신호를 받아 action mean/std가 정상범위(±1.5 / 0.4-0.7)로 회복되어야 한다. **RL_calf action_stats 정상화 = fix 성공의 강한 방증.** 회복 안 되면 fix가 RL을 실제로 쓰게 만들지 못한 것.

---

## 5. 메모리 제약 종합 self-check

| 메모리 항목 | 본 제안서 준수 |
|---|---|
| §1 contact를 policy obs/proprio 추가 금지 | ✅ 모든 신호 reward 내부 `_contact_sensor`만, policy 관측 미노출 |
| §2 신규/수정 reward = 가설+검증 동반 시 허용 | ✅ 1·2순위 각각 가설+검증지표 명시 |
| §3 reward_scale/hyperparam 단독 튜닝 금지 | ✅ 1·2순위는 구조 신호 신설, 3순위는 "단독 금지" 명시 |
| §4 action_latency blame 금지 | ✅ 언급 없음 |
| §5 actuator_mode2/약화 금지 | ✅ calf 섹션에서 명시적 금지 재확인 |
| §6 토크 envelope 정량추론 금지 | ✅ 토크 처방 없음 |
| §7 height_scan blame 금지 | ✅ 언급 없음 |
| max_air 단순 scale을 headline로 금지 | ✅ 3순위 보조 lever로만 강등 |
| 검증 안 된 추론 '확정 버그' 단정 금지 | ✅ escape 경로는 코드 VERIFIED, calf 인과는 advisor 확정분만 단정 |
| 사용자 confirm 전 코드 적용 금지 | ✅ 본 문서는 제안서, 적용은 confirm 후 |

---

## 부록 — 인용 라인 색인
- parkour_env.py: 169(_last_contacts init), 557(clip_actions=10.0), 1122/1133(net_contact_forces·contact), 1135(contact_filt debounce), 1140-1146(feet_dragging hind), 1149-1150(air/contact_time fetch), 1178(gait_pairing is_flat gate), 1196(feet_edge contact gate), 1199-1208(air_time_cap), 1205(air_time[:,_feet_ids]), 1211-1229(reward_values dict), 1233-1238(scale·breakdown), 1241(clip min=0), 1291(_last_contacts reset)
- parkour_env_cfg.py: 607-631(reward_scales), 623(gait scale 0), 630(air_time_cap -0.1), 648(air_time_cap_max_s 1.0), 650-653(gait params)
- contact_sensor.py: 422(is_contact), 432-433(current_air_time→0 리셋); contact_sensor_cfg.py:45(force_threshold=1.0)

# 00 — Gait-Improvement Reward Design for Go2-Parkour-Symmetry (SYNTHESIS)

> 작성: worker-1 (task #4 종합) · 2026-06-10
> 입력: `01_biomechanics.md` (worker-1) · `02_rl_gait_rewards.md` (worker-2) · `03_coordination_signals.md` (worker-3)
> 목표: 3개 조사를 **우리 Go2 parkour env의 구체적 reward 설계 + A/B 검증 계획**으로 변환.
>
> **증거 등급 표기**: **[FACT]** = 우리 세션에서 측정/코드로 확인 · **[LIT]** = 문헌의 검증된 주장 · **[HYP]** = 본 설계의 가설(미측정, 학습으로 검증 필요).
>
> **인용 정정 (lead 지시 1)**: arXiv **2403.10723** 의 정확한 저자는 **Ding, J., Chen, X., Katz, G.E., Gan, Z.** (*Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy*, v1 제목 *Leveraging Symmetries in Gaits for RL*). worker-2의 "Ding et al."이 맞고 worker-3의 "Su et al."은 **오기**. WebSearch로 확인(arxiv.org/abs/2403.10723 저자 목록). 이하 본 문서는 **Ding et al. (2024)** 로 통일.

---

## 1. 문제 정의 (확정 사실)

| 관찰 | 등급 | 출처 |
|------|------|------|
| 평지에서 bounding으로 수렴 | [FACT] | measure_pronk_cost.py |
| 장애물에서 4발 동시 pronk (stair 27.7% all-airborne) | [FACT] | 측정 |
| 분할점프: 앞발 착지 → 재이륙 → 뒷발 착지 (전후협응 부재) | [FACT] | 측정 (retk/flight stair 0.060 vs flat 0.002) |
| 비효율 확인: CoT stair 1.49× flat | [FACT] | 측정 |
| 안전(착지충격/saturation)은 임계 이하 → **safety penalty NO-GO** | [FACT] | 측정 (pre-registered X_sat=0.20, Y_impact=4.0 BW 미달) |
| baseline(Extreme Parkour/Robot Parkour Learning)에 gait/footfall reward **없음** | [FACT] | Cheng et al. 2023 §reward; worker-2 §4.1 코드확인 |
| 기존 `feet_gait_pairing` 항: weight=0 + is_flat-gated + clamp 0.3s saturated → 장애물에서 기여 0 | [FACT] | worker-2 §3.3 |

**Root cause** [FACT+LIT]: reward에 **footfall-pattern 신호가 없다**. Extreme Parkour는 "보상항이 보통 4발을 쓰는 gait로 이어진다"고만 가정하고 footfall을 강제하지 않음 → pronk/분할점프는 이 설계의 **예견된 실패모드** (worker-2 §4.1). mirror augmentation은 **L/R 대칭만** 강제하고 fore-hind는 강제하지 않음 [FACT]. 생체역학적으로 전후협응은 emergent가 아니라 **전용 메커니즘**(말 obstacle place-memory, 고양이 PPC Area5)이며, 명시 신호 없이는 나타나지 않음 [LIT, worker-1 §3].

**목표**: 안전이 아니라 **품질(효율 + 자연스러움 + 전후협응)**. fix는 baseline에 없는 footfall/coordination 신호를 **순수 additive**로 추가 (env 재작성 불필요, lead 지시 5) [FACT].

---

## 2. 두 조사의 독립 수렴 (설계 중심축)

worker-2(RL구현)와 worker-3(신호+sim2real)가 **독립적으로 동일한 1순위 2축**을 도출 (lead 지시 2):

- **축 A — Fore-hind 협응 reward**: phase-clock swing/stance (Siekmann/WTW) 또는 time-reversal/temporal symmetry (Ding et al. 2024, **Go2에서 검증됨**). 전후다리 phase 관계를 명시 → 분할점프 제거.
- **축 B — Anti-pronk (simultaneous-flight / min-stance)**: 전(全)공중 상태를 억제. 가장 직접적이고 clip-safe.

세 문헌(생체역학·RL·신호)이 모두 같은 결론을 가리킴:
- 생체역학[LIT]: gait는 속도별 CoT 최소화로 선택, pronk는 효율 gait가 아님; 전후협응은 명시 메커니즘.
- RL[LIT]: phase/symmetry reward가 gait를 *prescribe*; energy-min만으로도 trot 등 emergent(Fu et al. 2021)지만 험지엔 부족.
- 신호[LIT]: contact-state indicator penalty는 **bounded → clip(min=0) 헤드룸에 구조적으로 안전**.

---

## 3. 메모리/배포 제약 (모든 옵션이 통과해야 함)

| # | 제약 | 본 설계의 준수 방식 |
|---|------|---------------------|
| (a) | **contact-sensor를 observation에 넣지 말 것** (sim-to-real) | 모든 옵션은 contact/force를 **reward 내부에서만** 사용. policy obs에는 최대 **phase clock(sin φ, cos φ)** = 시간신호만 추가(접촉 아님). baseline도 이미 contact를 reward에서만 씀 → 동일 메커니즘 [FACT, worker-3 §0]. |
| (b) | **total_reward = clip(min=0)** → 과대 penalty는 step 합을 0으로 죽여 gradient 소멸 | penalty보다 **양수 보상 형태 우선**. anti-flight는 `+w·min(n_contact,2)/2` (∈[0,1], 양수)로 (lead 지시 4). 모든 신항은 bounded·小 weight. |
| (c) | **weight는 episode-normalized (step_dt 이미 반영)** | 신규 weight 산정 시 step_dt 재곱 금지. 측정값(episode-normalized)끼리 직접 비교 [메모]. |
| (d) | **Go2 deployable** (12-DOF, proprioception-only, 표준 actuator) | phase clock·symmetry·contact-in-reward 모두 proprioceptive 정책과 호환. Ding et al.은 **실제 Go2에서 검증** [LIT]. |

> **경계 명확화**: "contact 금지"는 *obs*에 한정. reward 계산 내부에서 contact force tensor를 읽는 것은 baseline(foot-jerk, feet-air-time, edge penalty)이 이미 하는 것과 동일하며 허용 [FACT, worker-3 §0].

---

## 4. Reward 설계 옵션 (3안)

> 모든 수식은 clip-safe(양수/bounded) 형태 우선. weight `w`는 **시작 제안값**(문헌 검증값 아님, 우리 clip 헤드룸에서 튜닝 필요) [HYP].

### 옵션 1 — **Anti-pronk min-stance (양수) + 기존 feet_gait_pairing 재활성화** [최소·최저위험]

목적: 가장 싸고 안전한 첫 실험. 새 buffer/네트워크 변경 없음.

1. **Min-stance 양수 보상** (lead 지시 4의 clip-safe 형태):
   ```
   n_contact = #{feet : F_z > thresh}      # reward 내부 contact, obs 아님
   r_stance  = w1 · min(n_contact, 2) / 2   # ∈ [0,1], 양수 → clip(min=0) 안전
   ```
   - pronk(n=0)은 0점, ≥2발 접지는 만점 → all-airborne 평형 제거.
   - **flight를 hard-ban 하지 않음**: gap 같은 정당한 도약은 순간적으로만 n=0이므로 episode 평균 보상 손실이 작음. (worker-3 §3c 경고: airborne threshold로 정당 점프 보존.)
2. **기존 `feet_gait_pairing` 재활성화** (worker-2 §3.3, §5.3):
   - `is_flat` gate 제거(장애물에서도 작동), clamp 0.3s 완화/상향(gradient 부활), 小 양수 weight 부여.
   - 이미 존재하는 항이라 **순수 튜닝** = 가장 작은 변경.

- 근거: [LIT] worker-2 §2.2/§5.2, worker-3 §3c (#1 추천). [FACT] 기존 항이 죽어있음.
- 장점: 코드 변경 최소, clip-safe, obs 불변, 즉시 A/B 가능.
- 한계: anti-pronk는 *제거*만 함. 전후협응을 *적극 형성*하진 못함 → 옵션 2 필요 가능성 [HYP].

### 옵션 2 — **Fore-hind 협응 reward (phase-clock 또는 symmetry)** [핵심·최고 적중]

목적: 전후협응을 **명시 신호로 형성** (root cause 직접 타격).

선택지 2-1 **Phase-clock swing/stance** (Siekmann 2021 / WTW = Margolis&Agrawal 2022) — ⛔ **BLOCKED, 메모 dealbreaker 위반**:
```
per-leg phase φ_i (공유 클럭에서; front vs hind 반주기 offset = trot diagonal θ1=0.5)
r_phase = w2 · Σ_foot [ (1−C_i^cmd)·exp(−‖F_i‖²/σ_f) + C_i^cmd·exp(−‖v_i‖²/σ_v) ]
```
- 이 방식은 **obs에 `sin φ, cos φ` 추가 = input팽창**. 메모리 제약은 **"Walk These Ways(input팽창) 거부됨"** 을 *이름으로 명시* → **이 옵션은 사용자가 input팽창을 명시적으로 허가하기 전까지 차단**.
- **두 제약을 혼동하지 말 것**: worker-2는 클럭을 "sim2real-safe(접촉 아닌 시간신호)"로 정당화 → 이는 제약 (a)만 통과. **input팽창 dealbreaker(별개·기결정)는 통과 못 함.** (a)를 통과한다고 input팽창이 면제되지 않는다.
- ⚠️ **divergence 경고 (lead의 "수렴" 프레임이 가린 부분)**: worker-2가 꼽은 *single most important lever*가 바로 이 phase clock(obs 확장)이고, worker-3는 input팽창을 명시 경고하며 fore-hind를 symmetry(obs 불변)로 라우팅. **이는 깨끗한 수렴이 아니라 실제 분기**. 독자는 worker-2 1순위(clock)를 메모 위반이므로 집지 말 것 → **2-2를 채택.**

선택지 2-2 **Time-reversal/temporal symmetry** (Ding et al. 2024, **Go2 검증**) — ✅ **이것이 fore-hind 레버**:
```
r_sym = w2 · exp(−‖ q_hind(t) − q_front(t−T/2) ‖² / σ)   # 뒷다리 = 앞다리 반주기 시프트
      (+ temporal periodicity: exp(−‖s(t)−s(t−T)‖²/σ) 로 비주기적 분할점프 버스트 억제)
```
- "front and back legs move in a similar fashion"이 **전후협응의 형식적 정의** [LIT, worker-2 §3.1 / worker-3 §3a]. 기존 L/R mirror 데이터증강과 자연 결합.
- obs 변경 불필요(history 사용) — phase var 있으면 보조.

- 근거: [LIT] worker-2 §1/§3, worker-3 §3a (#2 추천). Ding et al. Go2 실증.
- 장점: root cause 직접 해결, clip-safe(exp 형태), Go2 deploy 검증, **obs 변경 0 → input팽창 dealbreaker를 건드리지 않음**(history 기반, phase var 없이도 동작). 2-1 대비 결정적 우위.
- 한계: 특정 gait를 강하게 prescribe → **over-constraining 위험**(험지에선 비대칭 gait가 유리할 수 있음, Fu et al.; worker-1 §3.1 [HYP]). loose tolerance·작은 weight로 시작.

### 옵션 3 — **기존 energy/work penalty 상향 재튜닝** [효율 보조]

목적: 효율(CoT) 직접 개선 + 분할점프 비용 증가. **새 항 없음**(기존 absolute-work penalty 튜닝).

```
r_energy = −w3 · Σ_i max(0, τ_i·q̇_i)   # 양의 기계일만 (기존 항 weight 상향)
```
- 분할점프 = stride당 takeoff 2회 → 협응 도약(1회)보다 **항상 더 비쌈** → 협응 쪽으로 최적 편향 [LIT, worker-3 §1; Fu et al. 2021].
- energy-min 단독으로 평지에선 trot emergent하지만 **험지엔 불충분** → 옵션 1/2의 **보조**로만 [LIT].
- clip 주의: 장애물 push-off에서 power 스파이크 → 합이 음수로 가지 않도록 modest 상향만, 기존 항이라 새 헤드룸 위험 낮음 [HYP].

### 권장 스택 (옵션 조합)
> **옵션 1 (anti-pronk min-stance + gait_pairing 재활성화)** 을 1차로, **옵션 2-2 (symmetry, Go2 검증)** 를 핵심 협응 레버로, **옵션 3 (energy 재튜닝)** 을 효율 보조로.
> **결정적 강점: 이 스택(1 + 2-2 + 3)은 observation 변경이 0** → 메모 **input팽창 dealbreaker를 완전히 회피**하고 additive·clip-safe·Go2-deploy를 모두 통과. (2-1 phase-clock 경로는 obs 확장이라 제외.)

---

## 5. A/B 검증 계획 (measure_pronk_cost.py 메트릭 재사용)

> lead 지시: 기존 측정 스크립트 메트릭을 **그대로** 재사용. 스크립트 출력(코드 확인):
> `scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py` → `_workspace/pronk_cost_result.npz` + `pronk_cost_raw.md`.

### 5.1 비교군
- **A (baseline)**: 현재 Go2-Parkour-Symmetry 정책 (이미 측정된 pronk_cost 결과 = 기준선).
- **B**: 옵션 적용 후 재학습 정책. **한 번에 옵션 1개씩**(원인 분리, 메모 DON'T) — 예: B1=옵션1, B2=옵션1+2, B3=+옵션3.

### 5.2 1차 성공지표 (스크립트가 직접 산출)
| 메트릭 (스크립트 섹션) | 방향 | 합격 가설 [HYP] |
|------|------|------|
| **(4) retk/flight** (split-jump rate) per terrain | ↓ | gap/stair/step에서 baseline 대비 유의 감소 (예: stair 0.060→ ≤0.02) — **최우선 지표**(전후협응) |
| **(1) airborne_frac** (pronk frequency) per terrain | ↓ | stair 0.277 → 현저 감소; contact_mean_feet ↑ |
| **(3) CoT_vs_flat** | ↓ | stair 1.49× → 1에 근접 (효율 개선) |
| **(4) frontfirst/rearfirst, flight count/mean_len** | — | 협응된 단일 도약 = flights↓·mean_len 합리화, 전후 동시착지 경향 |
| **goal rate / failure rate** (스크립트 episode 통계) | **불변 또는 ↑** | **gate: traversal 성공률이 떨어지면 reject** (품질을 위해 과제수행 희생 금지) |
| (2a/2b) saturation, landing impact | 악화 없음 감시 | safety는 목표 아니나 회귀 감시(안전 penalty 미추가) |

### 5.3 절차
1. baseline 측정값 확보(있으면 재사용, 없으면 A 재측정).
2. 옵션 1 적용 → 학습 → 동일 명령으로 measure_pronk_cost.py 재실행 → 표 비교.
3. retk/flight·airborne_frac 개선 & goal rate 유지 시 채택, 아니면 옵션 2 추가.
4. **clip 헤드룸 실측 게이트** [worker-3 §7]: 학습 중 per-step penalty 합 vs 양수 보상 floor 로깅 → 신규 항이 clip(min=0)을 치는지 확인 후에만 weight 신뢰.
5. 변경은 `*_env_cfg.py`(weight/scale) + `*_env.py`(`_get_rewards`, 신규 buffer는 `_reset_idx` 초기화) [메모].

---

## 6. 절대 하지 말 것 (What NOT to do)

| 금지 | 이유 | 등급 |
|------|------|------|
| **contact/force booleans를 observation에 추가** | sim-to-real 파손 + 메모 명시 금지 | [FACT/메모] |
| **bare `feet_air_time` 만으로 anti-pronk 시도** | air_time은 **공중 시간을 보상** → pronk를 *악화* 가능. 반드시 phase/min-stance와 *짝*으로만, 上限 cap | [LIT, worker-2 §2.1/§5.2 (lead 지시 3)] |
| **큰 GRF/impact penalty 추가** | (1) impact 이미 임계 이하 → safety 근거 없음, (2) clip floor 위험(버스트 제곱항), (3) 정책을 timid하게 만들어 **더 airborne** 유도 = counterproductive | [LIT/FACT, worker-3 §2/§6 (lead 지시 3)] |
| **CoT를 dense per-step reward로 직접 사용** | ÷v 불안정. metric/curriculum gate로만 | [LIT, worker-3 §5] |
| **음수 penalty 위주 설계** | clip(min=0)에서 합을 죽임. 양수 보상 형태 우선(`+w·min(n_contact,2)/2`) | [메모/LIT (lead 지시 4)] |
| **obs에 phase clock / gait-command 추가 (input팽창)** | 메모: **"Walk These Ways(input팽창) 거부됨"** 을 이름으로 명시. `sin φ, cos φ` 추가(옵션 2-1)도 input팽창이라 동일 차단. "clock≠command, sim2real-safe"는 제약(a)만 통과하지 input팽창(별개·기결정)을 면제하지 않음 → **사용자 명시 허가 전까지 금지**. 전후협응은 obs 불변인 옵션 2-2(symmetry)로 라우팅 | [메모/LIT, worker-3 §3b] |
| **env 재작성 / baseline 구조 변경** | fix는 순수 additive로 충분(baseline에 gait reward 부재). 메모: env 재작성 dealbreaker | [FACT/메모, lead 지시 5] |
| **한 번에 여러 옵션 동시 적용** | 원인 분리 불가 | [메모 DON'T] |
| **특정 phase offset을 과하게 hard-prescribe** | 험지에서 비대칭 gait 필요할 수 있음 → over-constrain. loose tolerance·小 weight | [LIT/HYP, worker-1 §3, worker-3 §3] |

---

## 7. 권장 #1 (단일 추천)

> **옵션 1 (anti-pronk min-stance 양수 보상 + 죽어있는 `feet_gait_pairing` 재활성화)을 먼저 적용·A/B 하라.**
>
> 이유: (1) 세 문헌이 anti-pronk를 가장 직접적·clip-safe한 레버로 수렴 [LIT]; (2) `feet_gait_pairing`이 weight=0/gated로 **이미 죽어있음**이 확인됨 — 되살리는 것은 새 항 추가보다 작은 변경 [FACT]; (3) clip-safe 양수 형태로 헤드룸 안전 [LIT]; (4) obs·구조 불변 → sim2real·deploy·메모 제약 전부 통과 [FACT].
>
> retk/flight·airborne_frac이 충분히 안 내려가면 (전후협응 형성이 부족하면) **옵션 2-2 (Ding et al. Go2-검증 symmetry reward, obs 변경 0)** 를 추가하라 — 이것이 root cause(전후협응 부재)를 직접 형성하는 핵심 레버. 옵션 3(energy 재튜닝)은 효율 보조로 마지막에.
>
> **권장 스택(1+2-2+3)은 observation을 전혀 바꾸지 않아 메모 input팽창 dealbreaker를 완전히 회피한다 — 이것이 이 경로의 핵심 우위.** worker-2의 1순위(phase-clock, obs 확장)는 input팽창 위반이므로 사용자 명시 허가 전까지 집지 말 것.
>
> **검증은 항상 measure_pronk_cost.py의 retk/flight(전후협응) + airborne_frac(pronk) + CoT_vs_flat(효율)를 1차 지표로, goal rate 유지를 게이트로.**

---

## 8. 미해결 / 한계 (정직성)

- 모든 weight `w1,w2,w3` 절대값은 **시작 제안**일 뿐 [HYP] — clip 헤드룸 실측 후 튜닝 필요(worker-3 §7).
- "옵션이 retk/flight를 내린다"는 **가설** [HYP] — 학습 A/B로만 검증. 어느 옵션이 충분조건인지 미확정.
- phase clock obs 추가가 메모 "input팽창" 제약에 걸리는지는 **사용자 확인 권장**(본 문서 해석: gait command≠시간 클럭이므로 허용) [HYP].
- Ding et al. symmetry term의 정확한 수식/σ/weight는 원문 재확인 필요(worker-2 §6: WTW PDF 일부 미추출).
- 생체역학 수치(말·고양이)는 Go2 스케일과 Froude 다름 → 정성 원리로만, 절대 임계는 측정으로 [worker-1 §6].
- baseline A에 대한 measure_pronk_cost 결과 파일 존재 여부 미확인 — 없으면 A 재측정 선행.

---

## 출처 (통합)
- Hoyt & Taylor 1981 *Nature* 292:239 — gait별 CoT 최소화 (worker-1)
- Ding, J., Chen, X., Katz, G.E., Gan, Z. (2024) *Towards Dynamic Quadrupedal Gaits: A Symmetry-Guided RL Hierarchy* — arXiv:2403.10723 (Go2, **저자 정정 확인**)
- Siekmann et al. 2021 *Periodic Reward Composition* — arXiv:2011.01387
- Margolis & Agrawal 2022 *Walk These Ways* — arXiv:2212.03238 (입력팽창 형태는 우리에 금지, reward 개념만)
- Rudin et al. 2021 legged_gym *feet_air_time* — arXiv:2109.11978 (단독 anti-pronk 금지)
- Fu et al. 2021 *Minimizing Energy → Emergent Gaits* — arXiv:2111.01674
- Cheng et al. 2023 *Extreme Parkour* — arXiv:2309.14341 (baseline, gait reward 부재 확인)
- Zhuang et al. 2023 *Robot Parkour Learning* — arXiv:2309.05665
- Mock & Muknahallipatna *Hierarchical RL...* — arXiv:2506.20036 (impact penalty sim2real)
- *Learning-based legged locomotion SOTA* — arXiv:2406.01152 (simultaneous-flight penalty)
- Kim & Oh et al. *Not Only Rewards But Also Constraints* — arXiv:2308.12517 (clip 우회 대안, 범위 외)
- 측정 스크립트: `scripts/reinforcement_learning/rsl_rl/measure_pronk_cost.py` → `_workspace/pronk_cost_result.npz`

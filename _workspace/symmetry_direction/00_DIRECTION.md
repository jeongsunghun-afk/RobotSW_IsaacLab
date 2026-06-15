# 00 — Go2-Parkour-Symmetry 8런 종합 + 앞으로 방향성

> 작성: worker-1 (Task #5 종합) · 2026-06-15
> 입력: `01_code.md`(코드 타임라인) · `02_config.md`(cfg diff) · `03_logs.md`(tfevents) · `04_gait.md`(걸음걸이 측정) · `_workspace/gait_design/00_DESIGN.md`(reward 설계)
> 범위: **분석·방향 제시만. 코드 수정 없음.**
> 증거등급: **T1**=검증사실(로그/cfg/코드/측정 직접) · **T2**=강한 추론(데이터에서 직접 도출) · **T3**=가설(학습으로만 검증).

---

## 0. 한 줄 결론

**positive_work(효율) 효율항은 `-3e-4`가 최적이자 상한이다.** 8런 전부에서 R5(-3e-4)가 최고·최안정(reward 21.36, 무이벤트 수렴)이고, `-1e-3` 이상은 **학습 붕괴 + 걸음걸이 backfire 이중 사유**로 배제된다. 그러나 효율 단독으로는 **전신 pronk만 부분 억제**할 뿐 **trot/전후협응을 만들지 못한다**(자연 걸음의 핵심 미달). 다음 레버는 효율 추가 강화가 아니라 **전후협응을 직접 형성하는 신호**(obs 변경 0 옵션)다.

---

## 1. 사용자 4질문 직접 답변

### Q1. 코드(algo) — 무엇이 바뀌었나? → **알고리즘은 한 번도 안 바뀜**  (T1)
- 8런 전부 **algorithm/network 코드 동일**(`symmetry.py`, `ppo_parkour.py`, `actor_critic_parkour.py` 내용 diff 0). PPO 하이퍼·symmetry 설정(`use_data_augmentation=true`, mirror L/R, num_aug=2)·estimator도 전 런 고정.
- L/R **mirror data-augmentation은 baseline(R1)부터 이미 켜져 있었다.** "symmetry"는 이 8런의 변수가 아니라 상수다.
- reward **함수**(`positive_work = Σ_j max(0, τ_j·q̇_j)` @ `parkour_env.py:1126`, `air_time_cap`, `contact_duty_deficit`)도 처음부터 코드에 상주. weight로만 on/off.
- → **실제 변수는 reward weight 4개 + terrain 2필드뿐.** (코드 레벨 변경 0)

### Q2. config — 무엇이 바뀌었나? → **terrain 2필드 + reward weight 4항**  (T1)
누적 타임라인(02_config.md params/env.yaml 기준 = authoritative):

| 런 | terrain | torques_l2 | air_time_cap | contact_duty_deficit | positive_work | 판정 |
|----|---------|-----------|--------------|----------------------|---------------|------|
| R1 baseline | **easy**(step0.45/gap0.6) | -1e-5 | -0.1 | -0.5 | (없음=0) | ✅ |
| R2 terrain↑ | **HARD**(step0.6/gap0.8) | -1e-5 | -0.1 | -0.5 | 0 | ✅ |
| R3 torque×10 | HARD | **-1e-4** | -0.1 | -0.5 | 0 | ⚠️회복 |
| R4 air/dutyOFF | HARD | -1e-4 | **0** | **0** | 0 | ❌붕괴 |
| **R5 pw-3e-4** | HARD | **-1e-5(원복)** | 0 | 0 | **-3e-4** | ✅✅ 최고 |
| R6 pw0(control) | HARD | -1e-5 | 0 | 0 | **0** | ❌폭주 |
| R7 pw-1e-2 | HARD | -1e-5 | 0 | 0 | **-1e-2** | ❌즉사 |
| R8 pw-1e-3 | HARD | -1e-5 | 0 | 0 | **-1e-3** | ⚠️중반붕괴+부분회복 |

- **confound 2개**: (a) terrain은 R2부터 어려워져 끝까지 고정 → **R1(easy)만 지형이 다름**, baseline 직접비교는 보정 필요. (b) air_time_cap·contact_duty_deficit는 R4부터 전부 OFF → positive_work 효과는 "기존 gait 신호 부재" 상태에서 측정된 것.
- **깨끗한 단일변수 sweep = R5↔R6↔R7↔R8** (positive_work만 0/-3e-4/-1e-2/-1e-3, 나머지 전부 동일).
- **런 이름 함정**: "no_duty_time_cap"이 R4·R6 두 번 쓰였고 air/duty OFF는 R4~R8 공통이라 변별력 없음 — R6의 정체는 **positive_work=0 대조군**이다.
- **01_code.md 정정**: 내가 diff로 추론한 "R5~8 torques_l2=-1e-4"는 오류. params/env.yaml 실측(02)대로 **R5부터 -1e-5로 원복**. positive_work sweep(R5~8)은 torque penalty 측면에서 baseline과 동일하다. (diff의 committed-baseline 이동을 내가 놓침; 저장된 yaml이 authoritative.)

### Q3. 로그(학습 곡선) — 어느 게 성공인가? → **R5만 깨끗한 성공, R8은 성공 아님**  (T1)
- **R5(-3e-4)가 8런 명백한 승자**: final reward **21.36**(최고), std **0.518**(최저=가장 수렴), entropy 9.02, fall계열 termination 전부 0, 49999 step **무이벤트 단조 수렴**.
- **R8(-1e-3)은 "완주"가 아님**: iter만 50000 채웠을 뿐, **step~42500에서 reward·terrain 동시에 0으로 클립-바닥 붕괴**(std 0.53→1.05, entropy 9.3→17.5) 후 penalty가 약해 우연히 부분회복. 최종 reward **10.24 = R5의 절반**, std 0.80·entropy 14.3로 건강 basin(0.52/9.0) 미복귀.
- **붕괴 시그니처**(T2): penalty가 reward를 압도 → `total_reward = clip(min=0)`이 step 합을 0으로 죽임 → gradient 소실 → std/entropy 폭주 → curriculum(terrain_level) 0 리셋. R4·R6·R7·R8-중반이 전부 "reward=0 + terrain=0" 공유.
- calibration 정량 확정(T1+T2): **-3e-4 무사고 / -1e-3 경계(crash 1회+부분회복) / -1e-2 초반 즉사.** weight↑ → 붕괴위험 단조↑.
- 부수 발견(T2): `no_duty_time_cap`(duty/air cap 제거)은 **2/2 붕괴**(R4·R6). duty/cap 항이 안정화 regularizer였을 가능성 → 회귀 위험 항목.

### Q4. 걸음걸이 — 효율로 자연스러워졌나? → **부분 YES(pronk↓) / 핵심 NO(trot 안 생김)**  (T1)
측정(04_gait.md, R5↔R8 단일변수 비교):
- **전신 pronk(4발 동시비행): 단조 감소** — stair all-airborne 0.277(baseline)→0.237(R5)→**0.132**(R8). 효율 penalty가 직접 작동.
- **그러나 trot/교대보행 안 생김 → split-jump(앞뒤 비협응 다단 hop) 폭증** — stair retk/flight 0.060→0.078(R5)→**0.290**(R8, ×3.7). pronk 누른 대가로 "앞발 착지→뒷발 착지 전 재도약" **협응 붕괴**로 도피.
- **-1e-3은 자기목적마저 backfire** — 측정 CoT(=효율이 줄이려는 바로 그 양)가 stair 1.50×(R5)→**2.64×**(R8)로 오히려 악화, actuator 포화 평지 0.009→**0.96**, 실패율 0.7%→3.1%.
- **R5(-3e-4)가 전 지표 균형점**: pronk 억제하면서 CoT·포화·split-jump·실패율을 baseline 수준으로 유지.
- **핵심 사실(T1)**: "효율 단독으로는 trot이 emergent하지 않는다." pronk↓이 곧 자연보행↑이 아니다.

---

## 2. 종합 결론

1. **positive_work 최적·상한 = -3e-4** (T1). 위로 갈수록 (a)학습 붕괴(R8 중반/R7 즉사) + (b)걸음걸이 backfire(split-jump·CoT·포화 악화)의 **이중 배제**. -1e-3을 "성공 완주"로 분류 금지.
2. **효율항은 pronk 억제엔 성공, 자연 걸음(전후협응/trot) 형성엔 실패** (T1). 효율은 "전신 비행이 비싸다"만 가르치지 "앞뒤를 맞춰 한 번에 뛰어라"를 가르치지 못한다. 정책은 더 잘게 쪼갠 hop(split-jump)으로 도피.
3. **L/R symmetry(이미 ON)도 fore-hind를 강제 못 함** (T1/LIT). mirror는 좌우만 대칭화. 전후협응은 별도 명시 신호 필요(gait_design 00_DESIGN §1 root cause와 일치).
4. **clip(min=0)이 붕괴 증폭기** (T2, 설계는 의도된 것). 새 penalty도 헤드룸 잠식하면 동일 붕괴 → 신규 항은 **양수 보상·bounded·小 weight** 필수.

---

## 3. 앞으로 방향성 (사용자 핵심 요청)

### 3-1. 큰 그림: 레버를 "효율"에서 "전후협응 신호"로 전환
효율(positive_work) 트랙은 **-3e-4에서 천장에 도달**했다(더 키우면 붕괴+backfire). 남은 자연-걸음 격차의 핵심 = **전후협응 부재(split-jump)**이고, 이는 효율로 풀리지 않는다(측정 확정). 따라서 다음 레버는:
> **positive_work -3e-4를 유지한 채, 전후협응을 직접 형성하는 신호를 additive로 1개 추가한다.** (obs 변경 0, clip-safe 양수, Go2 deploy)

근거: `_workspace/gait_design/00_DESIGN.md` — 세 문헌(생체역학·RL·신호)이 독립적으로 "anti-pronk + fore-hind 협응"을 1순위 2축으로 수렴. **obs 변경 0**이라 input팽창 dealbreaker(메모 기결정)를 회피.

### 3-2. 구체적 다음 실험 (한 번에 1변수, 순서대로)

**실험 E1 — anti-pronk min-stance(양수) + 죽은 feet_gait_pairing 재활성** [최소·최저위험, 1순위]
- 변경(`parkour_env.py _get_rewards` + `parkour_env_cfg.py`):
  - `r_stance = w1 · min(n_contact, 2)/2` (∈[0,1] 양수, reward 내부 contact만 — obs 아님). 시작 `w1 ≈ +0.2~0.5` (clip-safe, 튜닝 대상).
  - 기존 `feet_gait_pairing`: `is_flat` gate 제거 + clamp 완화 + 小 양수 weight 부여(현재 0.0 → 죽어있음, T1).
- **유지**: positive_work -3e-4, terrain HARD, symmetry ON, air/duty는 OFF 유지(또는 안정화 위해 contact_duty 소폭 복원 검토 — §3-3 리스크 참조).
- 판정(measure_pronk_cost.py 재실행, R5를 A 기준):
  - **1순위**: split-jump retk/flight ↓ (stair 0.078 → 목표 ≤0.04)
  - pronk airborne_frac 유지/↓ (악화 금지)
  - CoT_vs_flat 비악화 (≤1.5×)
  - **게이트**: goal/traversal 실패율 비악화 (R5 0.7% 유지) — 떨어지면 reject
- 가설(T3): min-stance가 pronk를 양수보상으로 눌러 split-jump 도피로를 줄임. 단 anti-pronk는 "제거"만 하므로 전후협응 적극형성은 E2 필요할 수 있음.

**실험 E2 — Ding et al.(2024) time-reversal symmetry (fore-hind), E1으로 부족 시** [핵심 레버]
- `r_sym = w2 · exp(−‖q_hind(t) − q_front(t−T/2)‖²/σ)` (뒷다리=앞다리 반주기 시프트; history 기반, **obs 변경 0**). Ding et al. **Go2 실증**.
- E1에서 split-jump가 충분히 안 내려가면 추가. 기존 L/R mirror와 자연 결합. loose tolerance·小 weight로 시작(over-constrain 위험).
- 판정 동일(retk/flight 최우선). E1+E2 누적이면 그것도 1변수씩(E1 → E1+E2 순).

> **순서 고정**: E1 단독 → 측정 → 부족 시 E1+E2. 효율(positive_work)은 -3e-4 고정, 더 안 키운다(천장 확정).

### 3-3. 리스크 & 선결 조치
1. **워킹트리 잔재 복원(필수, 다음 런 전)** (T1): `parkour_env_cfg.py`의 positive_work가 **-1e-2(R7 잔재)** 상태일 수 있음. 다음 학습 전 반드시 **-3e-4로 복원** 확인. (cfg-worker 영역)
2. **clip(min=0) 헤드룸** (T2): 신규 항은 양수보상 우선·bounded·小 weight. 학습 중 per-step penalty합 vs 양수보상 floor 로깅으로 clip 충돌 감시 후에만 weight 신뢰.
3. **duty/cap 제거의 안정성 영향** (T2, T3): `no_duty_time_cap` 2/2 붕괴 관찰. air/duty OFF가 R4~R8 붕괴들의 공통 배경 → E1에서 contact_duty 소폭 복원(예: -0.2)이 안정화에 도움 될 수 있으나 **이것도 별도 1변수로** 분리 검증(E1과 섞지 말 것). 가설, 미검증.
4. **terrain confound**: 향후 비교는 HARD terrain 고정 유지(R5 기준선과 동일 조건). baseline(easy) 직접비교 금지.

### 3-4. 절대 하지 말 것 (메모/DESIGN §6 재확인)
- positive_work를 -3e-4 위로 키우기 (붕괴+backfire 확정).
- contact/force boolean을 **observation**에 추가 (sim-to-real, 메모 금지). reward 내부 contact는 허용(baseline도 함).
- obs에 phase-clock(sin φ, cos φ) 추가 = **input팽창 dealbreaker**(사용자 명시 허가 전 금지). 전후협응은 obs-불변 E2(symmetry)로 라우팅.
- bare `feet_air_time` 단독 anti-pronk (pronk 악화 가능), 큰 GRF/impact penalty (safety 근거 없음+timid화), env 재작성, 한 번에 다변수.

---

## 4. Evidence tier 요약
- **T1(검증)**: 8런 algo/network 코드 동일 / 변수=reward weight+terrain / R5 최고·최안정(21.36, std0.518, fall0) / R8 step42500 클립붕괴+부분회복(최종10.24) / calibration -3e-4·-1e-3·-1e-2 / pronk 단조↓(0.277→0.132) / split-jump 폭증(0.060→0.290) / CoT backfire(1.50→2.64×) / 포화·실패율 악화 / feet_gait_pairing weight=0 죽음 / params yaml authoritative(torques_l2 R5~8=-1e-5).
- **T2(강한추론)**: clip-바닥→gradient소실→폭주 붕괴 메커니즘 / weight↑→붕괴 단조 / no_duty 2/2 붕괴 패턴 / 효율↔전후협응 분리.
- **T3(가설, 학습 검증 필요)**: E1/E2가 split-jump를 내린다 / contact_duty 소폭 복원이 안정화 / 어느 옵션이 충분조건인지 / Ding symmetry 정확한 σ·weight.

---

## 5. 한 줄 요약 (사용자 보고용)
효율항은 **-3e-4가 천장**(R5 최고·최안정; -1e-3↑은 붕괴+걸음 backfire 이중배제). 효율은 **pronk는 눌렀지만 trot은 못 만든다**(split-jump로 도피). 다음 한 수 = positive_work -3e-4 고정한 채 **전후협응 신호(E1: min-stance 양수보상+gait_pairing 재활성 → 부족 시 E2: Ding Go2-symmetry, 둘 다 obs 변경 0)** 를 1변수씩 추가하고 measure_pronk_cost.py의 **retk/flight**를 1순위 지표·**goal 실패율**을 게이트로 판정.

# hind_leg "잰걸음(tiptoe shuffle)" 진단 & 자연 보행 개선 방향 제안

작성일: 2026-06-02
대상 환경: `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/` (활성 cfg = `HindLegHistoryEnvCfg`)

---

## 0. 한 줄 결론

> 지금 잰걸음은 reward를 **덜** 튜닝해서가 아니라, **`feet_air_time` 보상이 발 들기를 직접 처벌하고 있고(부호 역전) + 발 높이·교대 swing·phase clock을 보상하는 항이 하나도 없어서** 생긴 cheap optimum입니다. 해법은 검증된 순서로: **(a) air_time 보너스 제거/재설계 → (b) sin/cos gait clock을 obs에 추가 → (c) swing-phase foot-clearance + drag penalty 추가**. 모두 proprioception+내부 clock만 쓰므로 sim-to-real 배포 제약을 위반하지 않습니다.

---

## 1. 진단 요약 (코드·로그로 검증됨)

**정체:** `hind_leg`는 사족의 뒷다리 2개(HL/HR)만으로 서서 걷는 **2족 로봇**. 한 다리당 actuated 4관절(Hip/Thigh/Calf/Foot), 총 action 8. control 50Hz(sim 1/200, decimation 4), action_scale 0.25, joint-position 제어. obs(policy)=projected_gravity+commands+joint_pos+joint_vel+actions = 30. **gait clock 입력 없음**(`clock_inputs=False`). command lin_vel_x=[-2,1].
> (참고: `r2s_hind_leg`는 RL 없는 Real2Sim position-control 환경 — 학습 대상 아님.)

**Smoking gun — `feet_air_time` 부호 역전 (검증됨):**
- 식: `Σ((last_air_time − 0.5)·first_contact)`, scale +0.5, threshold **0.5s 고정**.
- 2족·키 ~0.6m 로봇의 현실 체공시간은 0.1–0.3s → `(air_time − 0.5) < 0` → **착지할 때마다 음의 보상** → "발을 들수록 손해". 발을 안 들면 기여 0.
- 로그 검증: 가장 잘 걷는 run(`termination_DR`, ep_len 740)에서 `Episode_Reward/feet_air_time = −0.060` (음수, 4번째로 큰 페널티). **전 run에서 음수.** 명목상 +보상 항이 실제로는 swing 억제 페널티로 작동 중.
- ⚠️ 동일 클래스 버그가 과거 Go2 parkour에서 "catastrophic failure"로 분석된 이력 있음(commit `fc1b0a874dd`).

**보조 원인 (검증됨):**
1. **gait/clearance 양성 유인의 총체적 부재** — foot clearance reward 없음, periodic/phase reward 없음, clock obs 없음. air_time을 고쳐도 "trot 교대 swing"은 별도 압력 없이는 안 나올 수 있음.
2. **penalty:tracking 비중** — 큰 swing을 만드는 모든 항(dof_acc, action_rate, lin_vel_z, similar_to_default, feet_air_time)이 동시에 음수. tracking 양수폭이 작아(평균 reward ~0.9) 미세진동 최적해가 경쟁력 있음.
3. **termination 과민** — base contact force > 1.0N에서 종료 + termination scale −100. 2족 불안정성과 결합해 single-support 회피 → 미세보행 강화 (2차 요인).
4. **base_height −10 / similar_to_default −0.1** — 정적 웅크림 자세 유인(Calf/Foot default ±0.8727 rad).

**버그성 별도 발견:** `undesired_contacts` penalized 목록에 `"HR_Foot_Link_Link"` 오타(실제 `HR_Foot_Link`) → 우측 발 페널티 누락(좌우 비대칭).

**과거 시도 이력 (재제안 회피):** 로그 디렉터리 = 시도 기록. 사용자는 이미 `noise_std_3.0`(탐색 noise↑), `add_termination_penalty`/`termination`, `termination_DR`(domain randomization), `no_orientation_penalty`를 시도함. **air_time/foot-clearance/gait/clock 관련 시도는 전무** → 본 제안 방향이 미탐색 영역.

---

## 2. 논문 조사 (2025 우선, venue 검증/미검증 명시)

### A. Reward shaping 경로 (가장 직접적)

| # | 논문 | 랩 | venue | URL | 핵심 | sim2real |
|---|------|----|----|-----|------|----------|
| 1 | Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition | Oregon State (Hurst, Cassie) | **ICRA 2021 (확인)** | https://arxiv.org/abs/2011.01387 | von Mises phase indicator로 swing 중 발 힘 처벌 / stance 중 발 속도 처벌. clock을 obs에 포함 | **배포 가능** (proprio+내부 clock, contact sensor 불필요) |
| 2 | Diverse Legged Locomotion using Barrier-Based Style Rewards | KAIST (Hae-Won Park) | **ICRA 2025 (확인)** | https://arxiv.org/abs/2409.15780 | relaxed log-barrier로 foot clearance(목표 0.15m)·gait를 swing phase에만 부과 | **배포 가능** (proprio, contact는 학습 시 estimator 추정) |
| 3 | CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion | EPFL (Ijspeert) | **RA-L 2022 (확인)** | https://arxiv.org/abs/2211.00458 | 결합 진동자 위상이 다리 리듬 강제, 정책은 amplitude/freq modulate. oscillator state를 obs에 | **배포 가능** (내부 oscillator state) — 단 action space 변경 필요(침습적) |
| 4 | Heuristic Step Planning for Dynamic Bipedal Locomotion | (Sovukluk 외, 소속 미검증) | arXiv 2025 (**venue 미검증**) | https://arxiv.org/abs/2511.00840 | Raibert-type controller가 몸통속도 오차로 foot placement 변조 | 대체로 배포 가능 (contact obs 사용 여부 미검증) |
| 5 | Gait-Conditioned RL with Multi-Phase Curriculum for Humanoid | UCL (Chengxu Zhou) | arXiv 2025 (**venue 미검증**) | https://arxiv.org/abs/2505.20619 | feet swing-height penalty(−15) + feet drag penalty(−0.5) + sin/cos phase + gait-conditioned reward mask | **배포 가능** (proprio only) |

### B. 모방학습/AMP 경로 (reward로 안 되면 데이터로 강제)

| # | 논문 | 랩 | venue | URL | 핵심 | sim2real |
|---|------|----|----|-----|------|----------|
| 6 | AMP: Adversarial Motion Priors | UC Berkeley (Peng) | **SIGGRAPH 2021 (확인)** | https://arxiv.org/abs/2104.02180 | task reward + motion clip discriminator로 style 강제. 원형 | 캐릭터(실로봇 미검증) |
| 7 | AMP Make Good Substitutes for Complex Reward Functions | UC Berkeley+Google | **IROS 2022 (확인, Best Paper Finalist)** | https://arxiv.org/abs/2203.15103 | 복잡한 hand reward를 수 초 mocap style reward로 대체, COT↓ | **있음(4족 실로봇)** |
| 8 | Advanced Skills through Multiple AMP (Multi-AMP) | ETH RSL (Hutter) | **ICRA 2023 (확인)** | https://arxiv.org/abs/2203.14912 | 다중 style discriminator, 4족↔2족 기립 전환 실증 | **있음(wheeled-legged)** |
| 9 | Robust and Agile Locomotion using AMP | Wu 외 | **RA-L 2023 (확인)** | https://ieeexplore.ieee.org/document/10167753/ | 평지 motion만으로 난지형 zero-shot, proprio only Go1 | **있음(Go1, 순수 proprio)** |
| 10 | StyleLoco: Generative Adversarial Distillation for Natural Humanoid Locomotion | — | arXiv 2025 (**미검증**) | https://arxiv.org/abs/2503.15082 | dual-discriminator(teacher RL agility + dataset 자연스러움). **2족 H1** | 있음(H1, 주장) |
| 11 | Walk in Costume: AMP for Aesthetically Constrained Humanoids | UCLA RoMeLa (Hong) | **Humanoids 2025 (확인)** | https://arxiv.org/abs/2509.05581 | 비표준 형상 2족(Cosmo)에 AMP+DR로 sim2real | **있음(2족)** |
| 12 | Symmetry-Guided RL for Dynamic Quadrupedal Gaits | Ding 외 | arXiv 2024 (**미검증**) | https://arxiv.org/abs/2403.10723 | time-reversal/morphological symmetry reward로 reference 없이 자연 gait. Go2 실증 | **있음(Go2)** — ⚠️후속본 2510.10455는 저자 철회, 인용 금지 |

> 종합 서베이: "Imitation learning for legged robot locomotion: a survey", Frontiers in Robotics and AI 2025 (확인) — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2025.1678567/full

**문헌 공통 진단(StyleLoco 등):** "RL + handcrafted reward = agile but unnatural gait." 정확히 현재 상황. AMP/imitation은 이 빈틈을 reference 분포 style reward로 메움.

---

## 3. 권장 시도 방향 (난이도·리스크 오름차순, 한 번에 하나씩)

CLAUDE.md 원칙(한 번에 1개만 변경, 원인 분리 가능)에 맞춰 단계화. 각 단계 후 재학습 → 로그/거동 확인 → 다음.

### 🟢 Phase 0 — 버그 수정 (즉시, 단독)
- `undesired_contacts`의 `"HR_Foot_Link_Link"` 오타 → `"HR_Foot_Link"`. 좌우 비대칭 페널티 제거. (담당: cfg-worker)

### 🟢 Phase 1 — air_time 보상 제거/재설계 ★최우선·최저비용·미탐색★
근본 원인 직접 제거. 두 옵션:
- **1a (간단):** `feet_air_time` 보너스 항을 **제거**하거나 threshold를 현실 체공시간(0.15–0.2s)으로 낮춤. 효과: 발 들기 처벌 해소.
- **1b (정공법, [1] Siekmann):** air_time 스칼라 보상을 버리고 **von Mises phase scheduling**으로 교체 — swing 구간 indicator `I_swing(φ)`로 (i) swing 중 발 접촉력 처벌, (ii) stance 중 발 속도 처벌. 좌/우 위상차 0.5로 교대 swing 보장.
- 권고: 먼저 **1a로 단독 검증**(air_time이 진짜 범인인지 깨끗이 확인) → 이후 1b로 격상.
- sim2real: 안전. (담당: reward-worker)

### 🟢 Phase 2 — gait phase clock을 obs에 추가 ([1],[5])
- obs에 `[sin(2πφ), cos(2πφ)]` 추가, `clock_inputs=True` 경로 활용. φ는 시뮬레이터 내부 생성 → **deploy 시 contact sensor 불필요, 시계만 돌리면 됨**.
- 정책이 시간축에서 좌/우 swing을 구조적으로 분리 가능 → 교대 gait의 전제 조건.
- 주의: obs 차원 변경 → obs_space, history buffer, normalization, network input shape 동기화 필수. (담당: obs-worker + cfg-worker, validate-code 필수)

### 🟡 Phase 3 — swing-phase foot clearance + drag penalty ([2] KAIST, [5] UCL)
- swing phase에만 활성: `r_clear = −w·I_swing(φ)·(p_foot_z − p_des)²`, 목표 `p_des ≈ 0.08–0.15m`(2족 스케일로 축소).
- **feet drag penalty**: 발이 지면 근처(`p_foot_z < ε`)에서 수평속도 크면 처벌 → 떨림형 끌림 직접 차단.
- [5] 비중 참고: swing-height penalty 크게, drag penalty 소폭으로 시작 후 튜닝.
- sim2real: foot z는 forward kinematics로 계산 가능(privileged 아님). (담당: reward-worker)

### 🟡 Phase 4 — penalty/termination 재균형 (위 단계로도 잔존 시)
- `base_height −10`, `similar_to_default −0.1` 완화 검토(웅크림 유인 제거).
- termination contact force 임계 1.0N → 상향 또는 termination scale 완화(single-support 허용). (담당: cfg-worker/reward-worker)

### 🔵 Phase 5 — 데이터 경로(AMP) 도입 (위 reward-shaping으로 충분치 않을 때)
reward-shaping으로 "기능적 보행"은 나오지만 "자연스러움"이 부족하면. 이 프로젝트는 AMP를 1급 지원(`rsl_rl/.../ppo_amp.py`, `amp_discriminator.py`).
- **5-0 (저비용 선조치):** `rsl_rl/.../symmetry.py`의 mirror/symmetry loss 활성화 — reference 없이 좌우/시간 대칭으로 교대 swing 유도([12]).
- **5-1 reference 확보 (난이도순):** ① Phase1~4의 기존 정책 rollout 또는 간단한 TO/MPC 궤적을 reference로(형상 일치, mocap 불필요 — Multi-AMP[8]·Escontrela[7] 방식) ② 인간 mocap(LaFAN1/AMASS) retarget(GMR/KDMR) ③ 생성 prior(CVAE).
- **5-2 AMP:** task reward 유지 + discriminator style reward 소가중. **discriminator 안정화 필수**(gradient penalty + 가중치 스윕). 불안정하면 StyleLoco[10]식 dual-discriminator.
- sim2real: 배포 정책은 **proprioceptive only** 유지([9]), discriminator/teacher는 학습 전용. privileged obs 절대 배포 obs에 넣지 않음.
- ⚠️ 2족은 4족보다 reference 확보가 어렵고(발 접촉 phase 정합), retarget 인간 모션은 발 접촉이 어긋나기 쉬움 → kinodynamic retarget 권장.

---

## 4. 핵심 메시지

1. **reward를 더 튜닝하는 게 아니라, 잘못된 항(air_time)을 먼저 끄는 것**이 1순위. 로그상 명백히 swing을 처벌 중.
2. **교대 swing은 "유인"이 있어야 학습됨** — clock obs + swing-clearance reward가 그 유인. 현재 둘 다 부재(미탐색).
3. 모든 1차 권고(Phase 1~4)는 **proprioception + 내부 clock만** 사용 → Go2/2족 sim-to-real 배포 제약 위반 없음.
4. reward-shaping이 한계에 부딪히면 **AMP/symmetry로 데이터 강제** — 단, 4족 검증 사례가 많아 2족 전이는 명시적 가정 필요(2족 직접 증거: AMP, StyleLoco-H1, Walk-in-Costume).

---

## 부록: 검증 메모
- **venue 확인됨:** Siekmann/ICRA2021, KAIST/ICRA2025, CPG-RL/RA-L2022, AMP/SIGGRAPH2021, Escontrela/IROS2022, Multi-AMP/ICRA2023, Robust&Agile/RA-L2023, Walk-in-Costume/Humanoids2025, C-GAIL/NeurIPS2024, Frontiers survey/2025.
- **arXiv only(venue 미검증):** StepPlanning(2511.00840), Gait-Conditioned UCL(2505.20619), StyleLoco(2503.15082), Symmetry-Guided(2403.10723), Natural-Generative-Prior(2503.09015).
- **철회 경고:** arXiv:2510.10455(대칭 gait 후속본) 저자 철회 — 원본 2403.10723만 사용.
- 진단의 air_time 부호 역전·clearance/gait 부재·termination 수치·오타는 모두 코드·로그로 검증. "air_time 수정이 잰걸음을 해소한다"는 재학습으로 확인할 가설.

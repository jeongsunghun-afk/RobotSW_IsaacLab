# Parkour 알고리즘·네트워크 개선 리서치 v2 (코드 anchored, 2026-06-10)

> v1(`parkour_research_survey_2026-06.md`)이 "약하다"고 거부된 뒤 재수행.
> **차이점**: (1) 우리 실제 코드(Extreme Parkour 파생)에 anchor, (2) 이미 다룬 논문 배제 후 2024-2026 신규 발굴, (3) **deploy-obs 무팽창 / env 무재작성** 하드 제약 강제, (4) **검증 등급**(peer-review + 공개코드 + Go2/근접 하드웨어)으로 티어링하고 preprint는 명시 격리.
> 원천 보고서: `parkour_code_anchor.md`, `parkour_research_v2_encoders.md`, `parkour_research_v2_distillation.md`, `parkour_research_v2_sota.md`, 그리고 3-leg 전용 스레드.

---

## 0. Anchor — 우리 코드의 실제 상태 (genuinely additive 판정 기준)

`parkour_code_anchor.md` 요약:
- **네트워크**: `ActorCriticRMA`. Actor MLP[512,256,128], input 104 = policy(46)+priv_explicit(6)+priv_encoder(20)+scan_latent(32). **Depth CNN 미사용**(height-scan 187→MLP 32). temporal encoder = **1D-Conv `StateHistoryEncoder`**(46×10→20), `is_recurrent=False`. **GRU/transformer 없음.**
- **알고리즘**: `PPOParkour`. RMA-style — priv_encoder(37→20)와 history_encoder를 DAgger로 distill(priv_reg + hist_latent_loss). 2-phase가 단일 루프에서 동시 진행. **MoE/SPO/LCP/Symmetry variant는 이미 코드에 존재**(default off).
- **observation**: deploy actor 46-dim. **contact_filt(4발 debounced boolean)이 이미 obs에 포함**. foot position은 obs 아님(reward `feet_edge`에만).
- **reward**: `contact_duty_deficit`(−0.5, EMA duty<0.5, **clip min=0**), `air_time_cap`(−0.1), `feet_dragging`(뒷발), `feet_edge`, `feet_stumble` 보유. `feet_gait_pairing` weight=0 비활성. **양의 air-time 보너스·활성 gait/phase prior·CaT·per-leg 비대칭 penalty 없음.** total reward도 clip(min=0).

**핵심 진단(메모리 + anchor 일치)**: 3-leg는 **reward-positive local optimum**. 보조 가설 — RL_calf가 **dead action dimension**(plasticity collapse)일 수 있음. → reward clipping에 **robust한 gradient 채널**이 필요하고, plasticity 복원이 선행일 수 있음.

**왜 v1이 약했나**: Extreme Parkour foot-clearance / Walk-These-Ways contact reward는 우리가 이미 `contact_duty_deficit`·`feet_dragging`로 **유사하게 보유**. 추천이 "이미 있는 것"이었음.

---

## Goal A — 3-leg local optimum 깨기 (최우선 · 사용자 pain)

### Tier A1 — peer-reviewed + 공개코드 (즉시 시도 권장)

**A1-1. CaT — Constraints as Terminations** ⭐ 가장 싼 검증
- LAAS-CNRS, **IROS 2024**. arXiv 2403.18765. 코드: https://constraints-as-terminations.github.io
- 메커니즘: 제약 위반을 **episode termination 확률 δ**로 변환. `δ = max_i p_i^max·clip(c_i⁺/c_i^max,0,1)`; PPO에 **3줄** — `rewards*=(1−δ); dones=δ`. contact-duty는 `c_i=max(0, target−actual)`.
- 왜 우리에게: gradient가 **reward magnitude가 아니라 termination 채널**로 흐름 → 우리 `clip(min=0)` reward floor와 **직교**해 무력화 불가. 실패 원인이 gradient-zero든 reward-positive-local-opt든 **양쪽 모두에 robust**.
- 제약: obs/env 무변경(PASS). 통합비용 **낮음**(3줄). Solo 하드웨어 검증.
- caveat: termination이 확률적 — trot의 자연스러운 contact 변동이 spurious termination 유발 가능, `p_i^max` 캘리브레이션 필요.

**A1-2. KAIST IPO — "Not Only Rewards But Also Constraints"** ⭐ 본격 해법
- KAIST RaiLab, **IEEE T-RO 2024**. arXiv 2308.12517. 코드: https://github.com/railabatkaist/legged-robot-constrained-rl
- 메커니즘: Interior-Point Policy Optimization. contact-duty/clearance/contact-velocity를 **log-barrier 제약**으로. `maximize J(π)+Σ log(d_k−J_{C_k})/t`. multi-head cost critic(11 제약, 오버헤드 작음). PPO·TRPO 백엔드 제공.
- 왜 우리에게: log-barrier는 **target 양쪽 모두 non-zero gradient**, main-reward와 구조적으로 분리 → reward clipping에 robust. CaT보다 원리적이고 제약을 세밀히 제어.
- 제약: obs/env 무변경(PASS). 통합비용 **중간**(PPO objective 교체 + cost critic). 다종 legged 하드웨어 + peer-reviewed.
- caveat: 제약을 너무 타이트하게 잡으면 전진속도 희생. duty target≈0.45 느슨하게 시작해 점진 강화.

> A1-1로 **빠르게 가설 검증**(3-leg가 clip-robust gradient로 깨지는지) → 효과 시 A1-2로 본격화. 이 순서 권장.

### Tier A2 — peer-reviewed + 공개코드 (선행/보조)

**A2-1. Shrink+Perturb + LayerNorm (on-policy plasticity 복원)**
- Juliani & Ash (MSR), **NeurIPS 2024**. arXiv 2405.19153. 코드: https://github.com/awjuliani/deep-rl-plasticity
- 메커니즘: K update마다 `W←(1−ε_s)W+ε_p·N(0,1)` + activation 뒤 LayerNorm. **on-policy PPO 전용으로 검증된 유일** 방법(ReDo는 off-policy용, on-policy 부적합).
- 왜 우리에게: RL_calf가 dead action dimension이면 **어떤 reward fix도 gradient 전파 안 됨**. shrink+perturb=weight magnitude 복원, LayerNorm=rank 복원 → gradient 경로 자체를 살림. **A1보다 선행일 수 있음.**
- 제약: obs/env 무변경(PASS). 통합비용 낮음(~50줄). ⚠️ **로봇 검증 없음**(gridworld/ProcGen/Atari sim만). dead-action 진단이 틀리면 효과 제한.

### Tier A3 — preprint / 코드 없음 (검증 미흡 · 격리)

**A3-1. GPO — Growing Policy Optimization** (basin 예방, new run용)
- NUS/A*STAR, **preprint 2026-01**(미peer-review, 코드 없음). arXiv 2601.20668
- `ã=β_t·tanh(a/β_t)`, β_t Gompertz 스케줄로 학습 초반 action을 0 근처로 압축 → 4-leg 확립 전 parked-leg basin 형성 방지. action-space wrapper(낮은 통합비용). quadruped/hexapod zero-shot 주장.
- ⚠️ preprint·코드 없음·기존 collapsed checkpoint엔 효과 제한(A2-1이 더 적합). 코드 공개 시 재평가.

**A3-2. R_mor — Morphological Symmetry Reward** ❌ 비권장(검증 결과 부적합)
- Ding et al., **preprint** arXiv 2403.10723(Go2 하드웨어, anon 코드).
- 스레드 3은 "phase clock 불필요 always-on penalty"로 #1 추천했으나 **원문 수식 검증 결과 반박됨**: `R_mor`의 게이트 `f(σ(i,j))=1`은 **두 다리의 commanded phase offset θ가 일치할 때만**(|θ_i−θ_j|≤0.01) 발화. 즉 **commanded gait phase 구조에 의존** → 우리 free-gait parkour엔 θ 구조가 없어 inert이거나 **gait command 입력 추가 필요**(사용자가 거부한 입력 팽창). 또한 순간적 L/R 관절차 penalty는 trot(뒷다리 antiphase)와 상충 소지. **mirror-invariant 3-leg를 못 깬다는 우리 기존 결론과 동류.** → 채택 안 함.

---

## Goal B — temporal encoder 업그레이드 (1D-Conv `StateHistoryEncoder` 교체)

> 우리 deploy temporal encoder = 1D-Conv StateHistoryEncoder. 깨끗한 drop-in = **1D-Conv → attention/SSM/GRU**. (depth CNN은 unused이므로 "depth CNN 교체"가 아니라 "history encoder 교체"로 frame.)

**B-1. MSTA — Masked Sensory-Temporal Attention** ⭐ 가장 깨끗한 fit
- Liu et al., **ICRA 2025**(peer-reviewed). arXiv 2409.03332. 프로젝트: https://johnliudk.github.io/msta/
- 각 sensor modality를 [sensor×time] 토큰으로, causal attention으로 modality-timestep 선택 가중 + 결측 마스킹. **proprioception-only, 추론 O(1)/step**, contact-sensor 불요 — 우리 deploy 제약에 정합. obs 재토큰화 + actor에 causal attention 층(통합 중간).
- caveat: 헤드라인 결과는 **sensor-dropout robustness**. 순수 agility 향상은 전문 확인 필요. 코드 공개 미확인.

**B-2. REAL — FiLM-modulated Mamba** (preprint 격리)
- HKUST(GZ), **preprint 2026-03**(코드 미공개). arXiv 2603.17653
- FiLM(proprio가 visual feature 변조)+Mamba SSM+EKF. **Jetson 50Hz 통과 유일**(13.1ms, transformer 23ms 탈락), O(1) 추론. Extreme Parkour 대비 SR 0.78 vs 0.16 주장.
- ⚠️ preprint·코드 없음·숫자 web-sourced 미검증. gap traversal은 SoloParkour보다 낮음(0.28 vs 0.36). **격리** — 코드 공개 시 history encoder를 Mamba로 교체 검토.

> 참고: **MoE는 이미 우리 코드에 존재**(`Go2ParkourMoEPPORunnerCfg`). MoE-Loco(IROS 2025, 공개코드 `hrh6666/MoE-Loco`)·UCL Vision-MoE(preprint)는 **capacity/agility**용이며 **3-leg 해법 아님** — MoE-Loco abstract가 "three-legged locomotion"을 emergent skill로 명시. 3-leg와 혼동 금지.

---

## Goal C — distillation / training 개선 (2-phase DAgger 대체·보강)

**C-1. RobotKeyframing — Multi-Critic PPO** ⭐ A1과 시너지
- ETH Zurich (Coros), **CoRL 2024**. arXiv 2407.11562
- reward 성분별 critic head(공유 backbone), advantage를 성분별 추정 후 합산. → **scale 오염 방지**. CaT/IPO로 contact-duty 제약 reward를 추가할 때 terrain reward와 크기 차로 advantage가 편향되는 문제를 해소. critic-only 변경(deploy 무관). 통합 낮음~중간.
- caveat: goal-conditioned keyframe용 — gait-vs-terrain 분해 이득은 추론적, reward 그룹 오배정 시 무효.

**C-2. CTS — Concurrent Teacher-Student RL**
- SUSTech, **RA-L 2024**. arXiv 2405.10830 (코드 없음, 프로젝트 페이지만)
- teacher/student 동시 학습(공유 πθ/critic, 다른 encoder), `L=L_PPO(Dᵗ)+L_PPO(Dˢ)+‖Eˢ−Eᵗ‖²`. 2-phase의 **frozen-teacher 분포 외삽 오차 제거**(속도추종 ~20%↓). 우리 RMA-style DAgger를 single-stage로 대체. 통합 중간(dual rollout). 로봇 검증 있음·코드 없음 → 재구현.

**C-3. Parkour in the Wild** (ideas portable)
- ETH RSL+NVIDIA, **IJRR 2025**. arXiv 2505.11164 (코드 없음, ANYmal)
- 9 expert→DAgger distill(action noise)→RL fine-tune. 직접 쓸 부분: **DAgger distillation curriculum + SOTA depth noise model(edge shuffle/Perlin holes/blur)** = sim-to-real 이식 가치. 9-expert 단계는 오버헤드.

**C-4. CritiQ/ReTRy** (reset-state curriculum)
- Cornell, **IROS 2025**. arXiv 2505.09546. 코드: https://github.com/portal-cornell
- ReTRy: student가 이탈하는 state에서 teacher rollout을 reset 분포에 추가 → 3-leg 발생 지점(중간 episode)을 recovery 문제로 반복 노출. reset-API 이식 필요. 벤치가 manipulation일 수 있어 scope 확인.

---

## Goal D — capability / agility (참고, 대부분 preprint·타 하드웨어)

- **PUMA** (ZJU, preprint 2601.15995): egocentric polar **foothold prior 4-float** + PAS annealing. Extreme Parkour 대비 wall-gap 96.9% vs 76.5% 주장. ⚠️ Lite3(비Go2)·코드 없음·obs 4-float 추가. 격리.
- **AME-2** (ETH, under review 2601.08485): **attention 기반 neural map encoder**로 height-scan encoder 대체 + uncertainty-aware mapping. ANYmal·코드 없음. 격리.
- **ABS/BAS** (CMU+ETH, **RSS 2024**/L4DC 2025): reach-avoid value network 안전층. ABS 코드 https://github.com/LeCAR-Lab/ABS (Go1 전용). parkour 아닌 고속 충돌회피 — 별도 safety layer로만.
- **Agile Continuous Jumping** (Google/CMU, **ICRA 2025**, 2409.10923): 연속 점프 hierarchical RL+model-based. Go1. 순수 PPO에 비정합.

---

## 권장 실행 순서

1. **CaT(A1-1) 3줄로 빠른 검증** — 3-leg가 reward-clip-robust gradient로 깨지는지 최소비용 확인.
2. 병행 **plasticity(A2-1)** 적용 — dead-action-dimension 가설 동시 검증(gradient 경로 복원).
3. 효과 확인 시 **KAIST IPO(A1-2)** 본격화 + **RobotKeyframing multi-critic(C-1)** 으로 제약 reward scale 분리.
4. 그 다음 **encoder(MSTA, B-1)** / **distillation(CTS, C-2 또는 Parkour-in-the-Wild noise model, C-3)** 업그레이드.
5. preprint(REAL/PUMA/GPO/AME-2/R_mor)는 **코드 공개 시 재평가** — 현재 숫자는 미검증.

## 검증 등급 / 주의
- **모두 검증 대상 가설** — A/B 학습 미실시. 적용 시 단독 변경+A/B로 원인 분리.
- preprint(2026 다수)는 web-sourced·미peer-review·코드 없음 → 우선순위 결정에서 격리. peer-reviewed+공개코드(CaT/IPO/plasticity/MSTA/RobotKeyframing/CritiQ)를 리드로.
- **R_mor 반례**가 보여주듯 worker의 "always-on, no-clock" 류 메커니즘 주장은 원문 수식 검증 전 headline 금지([[feedback_dont_assert_unverified_bugs]] 적용).

# 사족보행 Imitation Learning + Parkour 메소드 조사 — Ranked Shortlist

**Date**: 2026-06-08
**Scope**: IsaacLab / rsl_rl (on-policy PPO, 수천 parallel env)에 추가 구현할 가치가 있는 IL+parkour 메소드 후보.
**Ranking 기준**: 통합 비용(기존 AMP/RMA/distillation 인프라 재사용 가능성) × deploy 가능성(Go2 sim-to-real, deploy 불가 obs 의존 감점) × inventory 대비 명확한 delta. Paper hype 배제.

---

## 이미 구현됨 (novel 추천 대상 아님 — delta 기준선)

`_workspace/research/00_codebase_inventory.md` 기준 다음은 **이미 존재**하며, 이것만으로는 delta 없음:
- **AMP** (LS-GAN/BCE discriminator, gradient penalty, replay buffer, logit reg) — `algorithms/ppo_amp.py`, `modules/amp_discriminator.py`
- **RMA** (privileged training + Conv1D history encoder + DAGGER priv_reg + estimator) — `actor_critic_parkour.py`, `ppo_parkour.py`
- **Student-Teacher distillation** (BCE/MSE/Huber, recurrent 포함) — `distillation.py`, `student_teacher*.py`
- **Depth parkour pipeline** (depth backbone + depth actor distillation) — `depth_backbone.py`, `on_policy_runner_parkour*.py`
- **RND, symmetry loss** — `rnd.py`, `symmetry.py`

따라서 아래 후보는 모두 위 대비 **명확한 delta**를 갖는 것만 선별했다.

---

## 후보 (ranked, top 6)

### #1. ASE — Adversarial Skill Embeddings (skill-conditioned latent 위에 AMP)

- **Method**: AMP discriminator + latent skill space `z` + skill-discovery encoder (mutual-information/diversity objective). 사전학습된 low-level "skill" 정책을 다양한 downstream task가 재사용.
- **Paper**: *ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically Simulated Characters*, Xue Bin Peng 외, SIGGRAPH 2022 / TOG 41(4). [arxiv 2205.01906](https://arxiv.org/abs/2205.01906)
- **무엇을 바꾸나**: AMP가 "단일 motion 분포"를 흉내내는 것을 넘어, 정책을 latent `z`로 conditioning하고 discriminator+encoder가 `z`별로 구분 가능한 skill을 강제. 하나의 pre-trained latent로 여러 parkour goal(점프/기어가기/오르기)을 hierarchical하게 호출.
- **parkour/IL 적용성**: **High** — parkour의 본질이 "다양한 스킬을 terrain goal에 따라 전환"이라 skill-conditioned latent와 직결. 단 원논문은 humanoid 격투/이동.
- **rsl_rl 통합 비용**: **저비용~중간**. 기존 `AMPDiscriminator`에 `z`-conditioning input + encoder head 추가, `ActorCriticRMA`에 latent input concat. on-policy PPO와 호환(ASE 자체가 PPO+parallel sim 기반). 새 파이프라인 불필요.
- **이미 존재? (delta)**: AMP는 있으나 **skill latent + skill-discovery encoder + latent-conditioned policy는 없음** = delta 명확.
- **public 구현**: **있음** — [github.com/nv-tlabs/ASE](https://github.com/nv-tlabs/ASE) (IsaacGym 기반, PPO+AMP 구조라 rsl_rl 이식 reference로 직접 활용).
- **deploy 코멘트**: latent `z`는 정책 내부 입력이라 deploy 시 상위 정책이 생성/선택 → 추가 센서 불필요. Go2 deploy 무리 없음. 단 motion clip 데이터셋 품질이 skill 다양성 상한.

---

### #2. WASABI — Wasserstein Adversarial Motion Prior (사족 전용, partial/rough demo)

- **Method**: AMP의 LS-GAN을 **Wasserstein-GAN critic**으로 교체. 불완전·물리적으로 호환 안 되는(예: 손으로 들고 흔든) demonstration에서 reward 추출.
- **Paper**: *Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations* (WASABI), Li 외, CoRL 2022. [arxiv 2206.11693](https://arxiv.org/abs/2206.11693)
- **무엇을 바꾸나**: discriminator divergence를 JS/LS → Wasserstein으로. partial demo로도 backflip 같은 agile skill을 Solo-8 사족로봇에 sim-to-real.
- **parkour/IL 적용성**: **High** — 사족로봇 sim-to-real 검증 + **rough/partial·물리적으로 호환 안 되는 demo로부터** agile skill 학습(이 partial-demo 능력이 핵심 delta). 완전한 mocap 없이 학습 가능 → 데이터 부담 ↓. (주: 사족 AMP sim-to-real 자체는 #5의 2304.10888 등에도 존재하므로 "유일"은 아님 — WASABI의 차별점은 *partial demo*.)
- **rsl_rl 통합 비용**: **최저비용**. 기존 `amp_discriminator.py`의 loss 함수만 WGAN critic으로 교체(현재 `disc_loss_type: ls_gan|bce`에 `wgan` 옵션 추가). gradient penalty 인프라 이미 존재(WGAN-GP에 그대로 재사용). 새 모듈 불필요.
- **이미 존재? (delta)**: GP/replay/logit-reg는 있으나 **Wasserstein divergence 자체는 없음** (현재 LS-GAN/BCE만) = delta = loss 교체. 추가로 "partial demo로 reward 추출" 워크플로가 delta.
- **public 구현**: **있음** — [github.com/martius-lab/wasabi](https://github.com/martius-lab/wasabi) (Isaac Gym + legged_gym + PPO 기반 → rsl_rl 이식 reference로 직접 활용 가능). 동저자 후속 [CASSI (martius-lab/cassi, ICRA 2023)](https://github.com/martius-lab/cassi)도 사족 self-supervised adversarial imitation 코드 공개.
- **deploy 코멘트**: 사족 sim-to-real이 논문 핵심 결과라 deploy 위험 최저. 추가 obs 없음. **Go2 가장 안전한 선택지**.

---

### #3. Extreme Parkour — goal-conditioning + scandots→depth, RL regularization delta

- **Method**: 단일 front-facing depth로 extreme parkour. **방향 goal(yaw) + waypoint conditioning**, scandots oracle → depth distillation, RL-stage에 specific regularization(action smoothness, energy).
- **Paper**: *Extreme Parkour with Legged Robots*, Cheng, Shi, Agarwal, Pathak, ICRA 2024. [arxiv 2309.14341](https://arxiv.org/abs/2309.14341)
- **무엇을 바꾸나**: depth-distillation 골격은 기존 인프라와 동일하나, **(a) yaw/unit-vector goal representation, (b) scandots oracle 정책의 regularization·reward 설계, (c) two-stage RL→depth 전이 디테일**이 delta.
- **parkour/IL 적용성**: **High** — Go2급 저가 사족 + 카메라로 검증. 본 프로젝트 parkour env(`direct/parkour`)와 task가 거의 동일.
- **rsl_rl 통합 비용**: **저비용(부분 채택)**. depth pipeline은 이미 있으므로 **goal representation·regularization 항목만 선택 이식**. 전체 파이프라인 재구현은 불필요(이미 RMA+depth 보유).
- **이미 존재? (delta)**: depth distillation은 있음. **yaw-goal conditioning + scandots oracle regularization 레시피는 없거나 다름** = delta. (단 메모: contact sensor obs 추가 금지 — Extreme Parkour는 contact obs 비의존이라 안전.)
- **public 구현**: **있음** — [github.com/chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour) (legged_gym/rsl_rl 계열, 이식성 높음).
- **deploy 코멘트**: 저가 로봇+단일 depth deploy가 논문 핵심. Go2 직접 적용 가능. deploy-불가 obs 없음.

---

### #4. CALM — Conditional Adversarial Latent Models (motion-conditioned encoder + directable)

- **Method**: ASE 후속. policy + **motion encoder를 동시 학습**, encoder가 reference motion의 핵심 특성을 재구성(복제 아님). 학습 후 motion으로 directable, style-conditioned 상위 task 학습.
- **Paper**: *CALM: Conditional Adversarial Latent Models for Directable Virtual Characters*, Tessler, Kasten, Guo, Mannor, Chechik, Peng, SIGGRAPH 2023. [arxiv 2305.02195](https://arxiv.org/abs/2305.02195)
- **무엇을 바꾸나**: ASE의 unsupervised skill latent을 **motion-conditioned encoder**로 대체 → 특정 reference로 style 지정 가능. precision-task(parkour의 정확한 발 배치)에 ASE보다 controllable.
- **parkour/IL 적용성**: **Med-High** — directable이 parkour goal-following에 유리하나, 원논문이 humanoid 그래픽스라 사족/sim-to-real 미검증. ASE 대비 추가 encoder 복잡도.
- **rsl_rl 통합 비용**: **중간**. ASE와 유사하게 AMP 위에 encoder 추가하나 motion-conditioned 학습 루프가 ASE보다 복잡. ASE를 먼저 깔면 증분 비용 낮음(ASE의 후속이므로).
- **이미 존재? (delta)**: 없음. ASE/CALM 계열 latent 전부 미구현 = delta. ASE와 중복 추천이므로 **ASE를 먼저 채택 시 CALM은 그 위 확장**.
- **public 구현**: **있음** — [github.com/NVlabs/CALM](https://github.com/NVlabs/CALM) (ASE 코드베이스 파생).
- **deploy 코멘트**: latent/encoder 내부 입력만 → 추가 센서 불필요. 단 사족 sim-to-real 미검증이라 ASE 대비 deploy 리스크 약간 ↑.

---

### #5. AMP reward 안정화 기법 묶음 (Selective-AMP + action LPF + adaptive curriculum)

- **Method**: 단일 논문이 아닌, AMP의 discriminator reward 불안정(너무 강하면 붕괴/너무 약하면 발산)을 완화하는 검증된 엔지니어링 묶음 — (a) **Selective AMP**(periodic gait에만 AMP 적용, dynamic 구간 제외), (b) **action low-pass filter**(smoothness↑ sim-to-real↑), (c) **adaptive curriculum**.
- **Paper**: 대표 근거 — *Multi-Gait Learning for Humanoid Robots Using RL with Selective Adversarial Motion Prior* ([arxiv 2604.19102](https://arxiv.org/abs/2604.19102)); *Learning Robust, Agile, Natural Legged Locomotion Skills in the Wild* ([arxiv 2304.10888](https://arxiv.org/abs/2304.10888)).
- **무엇을 바꾸나**: AMP reward를 task별로 게이팅하고 action을 필터링해 학습 안정성·gait 품질·전이성을 개선. 본 프로젝트 메모(AMP/3-leg gait, reward 불안정 이력)와 직결.
- **parkour/IL 적용성**: **Med-High** — 이미 AMP를 쓰는 Go2-Imitation/Parkour에 즉시 적용 가능한 안정화 패치. 새 capability라기보다 **기존 AMP 품질 개선**.
- **rsl_rl 통합 비용**: **최저비용**. `ppo_amp.py`의 reward 계산에 task-mask 추가, env action 단계에 1차 LPF(또는 action-rate penalty 강화). 새 모듈 없음.
- **이미 존재? (delta)**: AMP 자체는 있으나 **selective gating·action LPF·AMP 전용 curriculum은 미구현** = delta(엔지니어링 레벨).
- **public 구현**: **부분** — 개별 기법은 legged_gym/Isaac 계열에 산재(LPF, action-rate는 흔함). selective-AMP은 논문 기술 기반 자체 구현.
- **deploy 코멘트**: action LPF는 sim-to-real에 **유익**(jitter↓). 추가 obs 전무. deploy 안전. 빠른 ROI.

---

### #6. NCP / ControlVAE — VQ-VAE discrete skill prior (고품질 latent, 단 고비용)

- **Method**: AMP-tracking으로 motion을 흉내내되 skill을 **VQ-VAE discrete code**(NCP) 또는 world-model 기반 VAE latent(ControlVAE)로 압축. "prior shifting"(curiosity RL)으로 다양성 확보 후 상위 task가 code 위에서 학습.
- **Paper**: *Neural Categorical Priors for Physics-Based Character Control*, Zhu 외, TOG 2023 ([arxiv 2308.07200](https://arxiv.org/abs/2308.07200)); *ControlVAE*, Yao, Song, Chen, Liu, SIGGRAPH Asia 2022 ([arxiv 2210.06063](https://arxiv.org/abs/2210.06063)).
- **무엇을 바꾸나**: 연속 latent(ASE/CALM) 대신 discrete code book으로 motion 품질·다양성 향상. ControlVAE는 **learnable world model**로 latent를 직접 supervise(model-based).
- **parkour/IL 적용성**: **Med** — motion 품질은 SOTA급이나 사족/parkour/sim-to-real 미검증, humanoid 그래픽스 도메인.
- **rsl_rl 통합 비용**: **고비용**. VQ-VAE codebook(NCP)·world model(ControlVAE)은 **on-policy PPO 인프라 밖**. ControlVAE는 model-based로 rsl_rl(model-free on-policy)에 없는 dynamics model 파이프라인 필요. 새 파이프라인.
- **이미 존재? (delta)**: 전부 미구현 = delta는 크나 통합 부담도 큼.
- **public 구현**: **있음** — [NCP github](https://github.com/Tencent-RoboticsX/NCP), [ControlVAE 페이지](https://heyuanyao-pku.github.io/Control-VAE/).
- **deploy 코멘트**: latent/code 내부 입력 → 센서 추가 없음. 단 world model(ControlVAE) 런타임 비용·복잡도가 sim-to-real 검증 부재와 겹쳐 리스크 ↑. 연구 후순위.

---

## 참고: 조사했으나 후순위/제외

| 메소드 | 근거 | 제외/후순위 이유 |
|--------|------|------------------|
| **DeepMimic** (phase-based tracking) | [arxiv 1804.02717](https://arxiv.org/abs/1804.02717) | 단일 clip phase-tracking. AMP가 사실상 상위 호환(phase·clip-selection 불요). novel delta 약함 |
| **Diffusion Policy** (로봇 제어) | (일반) | off-policy/BC + 무거운 inference. rsl_rl on-policy 밖 + **deploy latency 리스크** → 고비용. 본 프레임워크 부적합 |
| **Robot Parkour Learning** (Zhuang, soft-dynamics two-stage) | [arxiv 2309.05665](https://arxiv.org/abs/2309.05665) | two-stage RL+DAgger distillation = 기존 RMA+distillation과 구조 중복. delta는 "soft→hard dynamics curriculum"뿐(env-side, algo delta 적음) |
| **ANYmal Parkour** (Hoeller, skill+navigation 계층) | [arxiv 2306.14874](https://arxiv.org/abs/2306.14874) | 고수준 navigation policy가 개별 skill 선택. ANYmal(고가) 특화, 계층 구조 통합 부담 큼. 아이디어(skill 라이브러리+selector)는 ASE로 더 저렴히 달성 |
| **Parkour in the Wild** (Rudin, multi-expert distillation) | [arxiv 2505.11164](https://arxiv.org/abs/2505.11164) | 3-stage multi-expert→distill→RL-finetune. distillation 인프라 이미 보유, 추가 delta는 "real-scan terrain finetune"(env-side). algo 신규성 낮음 |

---

## 통합 비용 분류

### A. 기존 AMP/RMA 인프라에 얹는 **저비용** (권장 우선)
- **WASABI (#2)** — `amp_discriminator.py` loss만 WGAN으로 교체. GP 인프라 재사용. 사족 sim-to-real 검증.
- **AMP 안정화 묶음 (#5)** — `ppo_amp.py` reward gating + env action LPF. 새 모듈 없음.
- **ASE (#1)** — `AMPDiscriminator`+`ActorCriticRMA`에 latent/encoder head 추가. on-policy 호환. nv-tlabs 코드 reference.
- **Extreme Parkour 부분 채택 (#3)** — goal-conditioning + regularization 레시피만 이식. depth pipeline 재사용.

### B. **새 파이프라인 필요 고비용**
- **CALM (#4)** — ASE 위 motion-encoder 추가(ASE 선행 시 증분, 단독은 중간).
- **NCP / ControlVAE (#6)** — VQ-VAE codebook / world-model. on-policy PPO 밖, model-based 신규 인프라.

---

## TOP 3 추천

1. **WASABI (Wasserstein-AMP)** — 최저 통합비용(loss 교체) + 사족 sim-to-real 검증 + **partial/물리적 비호환 demo로부터 학습**이라는 고유 delta. martius-lab/wasabi 공개 코드(legged_gym+PPO)로 이식 reference 확보. 본 프로젝트의 AMP reward 불안정 이력에 직결. **deploy 리스크 최저**. 가장 먼저 시도할 것.
2. **AMP 안정화 묶음 (Selective-AMP + action LPF + curriculum)** — 거의 즉시 적용 가능한 ROI. 기존 AMP/3-leg gait·reward 붕괴 문제의 실질적 완화책. action LPF는 sim-to-real에 직접 이득.
3. **ASE (Adversarial Skill Embeddings)** — parkour의 "다양한 skill을 goal로 전환" 본질에 구조적으로 부합하는 가장 큰 capability delta. nv-tlabs 공개 코드(PPO+AMP)로 이식 reference 확보. latent는 내부 입력이라 deploy 안전.

> 보조: parkour env 개선이 목적이면 **Extreme Parkour의 yaw-goal conditioning + oracle regularization**을 #3와 병행 채택(env-side, 저비용).

---

## Sources

- [ASE — arxiv 2205.01906](https://arxiv.org/abs/2205.01906) / [github.com/nv-tlabs/ASE](https://github.com/nv-tlabs/ASE)
- [WASABI — arxiv 2206.11693](https://arxiv.org/abs/2206.11693) / [github.com/martius-lab/wasabi](https://github.com/martius-lab/wasabi) / [CASSI github.com/martius-lab/cassi](https://github.com/martius-lab/cassi)
- [Extreme Parkour — arxiv 2309.14341](https://arxiv.org/abs/2309.14341) / [github.com/chengxuxin/extreme-parkour](https://github.com/chengxuxin/extreme-parkour)
- [CALM — arxiv 2305.02195](https://arxiv.org/abs/2305.02195) / [github.com/NVlabs/CALM](https://github.com/NVlabs/CALM)
- [Selective AMP (multi-gait) — arxiv 2604.19102](https://arxiv.org/abs/2604.19102) / [Legged locomotion in the wild — arxiv 2304.10888](https://arxiv.org/abs/2304.10888)
- [NCP — arxiv 2308.07200](https://arxiv.org/abs/2308.07200) / [github.com/Tencent-RoboticsX/NCP](https://github.com/Tencent-RoboticsX/NCP) ; [ControlVAE — arxiv 2210.06063](https://arxiv.org/abs/2210.06063)
- 후순위: [DeepMimic 1804.02717](https://arxiv.org/abs/1804.02717), [Robot Parkour Learning 2309.05665](https://arxiv.org/abs/2309.05665), [ANYmal Parkour 2306.14874](https://arxiv.org/abs/2306.14874), [Parkour in the Wild 2505.11164](https://arxiv.org/abs/2505.11164)
</content>
</invoke>

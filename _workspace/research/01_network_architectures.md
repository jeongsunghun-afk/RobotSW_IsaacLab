# Quadruped/Parkour/IL Network Architectures — Integration Shortlist for IsaacLab/rsl_rl

조사일: 2026-06-08 · 대상 프레임워크: on-policy PPO + 수천 parallel env (Isaac Sim GPU), sim-to-real / Go2 deploy 상시 제약

## 평가 기준 (rubric)
- **drop-in 여부의 진짜 기준**: PPO loop(`algorithms/ppo.py`)을 건드리지 않고 `nn.Module`(ActorCritic)만 교체하면 **Low**. 보조 loss + 별도 optimizer step이 `update()`에 들어가면 **Med/High**. "그냥 encoder니까 싸다"는 함정 — encoder라도 training objective가 바뀌면 ppo.py를 건드린다.
- **이미 존재 판정**: 아래는 코드베이스에 **이미 구현됨**(novel 금지). 본 보고서는 언급만:
  - MLP / CNN2D / GRU·LSTM recurrent actor-critic (`networks/`, `actor_critic_recurrent.py`)
  - `ActorCriticRMA` + Conv1D `StateHistoryEncoder` (privileged + history matching/DAGGER) — `actor_critic_parkour.py`
  - `StudentTeacher` distillation (recurrent 포함) — `student_teacher*.py`
  - Depth backbones: FC / Recurrent(GRU) / Stacked(Conv1D) — `depth_backbone.py`
  - Empirical normalization, observation groups — `networks/normalization.py`
  - `Estimator`, AMP `Discriminator`, DIAYN/LSD skill discriminators — `estimator.py`, `amp_discriminator.py`

---

## 후보 (7개)

### 1. MoE Locomotion Policy (MoE-Loco)
- **Method**: Sparse/soft Mixture-of-Experts actor (gating MLP + N개 expert MLP).
- **Paper**: *MoE-Loco: Mixture of Experts for Multitask Locomotion* — Runhan Huang, Shaoting Zhu, Yilun Du, Hang Zhao (2025). https://arxiv.org/abs/2503.08564 · 관련: *Quadruped Parkour Learning: Sparsely Gated MoE with Visual Input* https://arxiv.org/html/2604.19344
- **무엇을 바꾸나**: 단일 MLP actor → gating(softmax, hidden [128]) + 6 expert MLP. expert가 terrain별(bars/pits/stairs/slopes) 행동에 자연 특화, multitask gradient conflict 완화.
- **Go2/parkour 적용성**: **High** — parkour의 step/gap/stair multi-terrain이 정확히 MoE-Loco의 타깃. 현 코드베이스 3-leg gait local-optimum도 multitask gradient conflict 성격.
- **rsl_rl 통합 비용**: **Low (drop-in module)** — `modules/`에 `ActorCriticMoE` 새 파일. forward만 gating·expert로 교체, PPO `update()` 무수정. cfg에 `num_experts` 추가.
- **이미 존재?**: 없음. MLP(`mlp.py`)를 building block으로 재사용하나 gating/expert 구조는 신규.
- **public 구현 있음?**: 논문 코드 부분 공개(project page moe-loco.github.io). MoE actor 자체는 PyTorch로 100줄 내 재현 가능.
- **deploy 가능성**: **High** — soft-MoE는 추론 시 전 expert forward지만 expert당 작은 MLP라 Go2 onboard 충분. sparse top-k면 더 저렴. 추가 센서 불필요.

### 2. Attention-Based Map Encoder (AME)
- **Method**: CNN local terrain feature + multi-head attention(proprio query → map feature) 으로 height-scan 인코딩.
- **Paper**: *Attention-Based Map Encoding for Learning Generalized Legged Locomotion* — Junzhe He, Chong Zhang, Fabian Jenelten, Ruben Grandia, Moritz Bächer, Marco Hutter (2025). https://arxiv.org/abs/2506.09588 · 후속 *AME-2* https://arxiv.org/abs/2601.08485
- **무엇을 바꾸나**: 현재 scandots를 flat MLP/Conv로 처리하는 부분을, proprio-conditioned cross-attention으로 교체해 traversable 영역에 집중. ANYmal-D/GR-1 parkour 100% 성공.
- **Go2/parkour 적용성**: **High** — 입력이 **기존 height-scan/scandots와 동일**(새 센서 X). parkour env가 이미 height_scan 보유 → drop-in으로 terrain 인코더만 교체.
- **rsl_rl 통합 비용**: **Low~Med** — `modules/`에 attention map encoder 새 모듈. ActorCriticRMA의 scandot encoder(`actor_critic_parkour.py:154`) 자리를 교체. PPO loop 무수정이면 Low, asymmetric critic 분리하면 Med.
- **이미 존재?**: scandot encoder는 존재(MLP). delta = **MLP → cross-attention**(구조적 차이, 입력 동일).
- **public 구현 있음?**: 논문/프로젝트 페이지 공개. MHA는 `torch.nn.MultiheadAttention` 표준.
- **deploy 가능성**: **High** — 입력 센서 동일, MHA 한 블록은 onboard 부담 작음. 해석가능성↑(어디 보는지).

### 3. CPG-conditioned Policy (CPG-RL)
- **Method**: 정책이 직접 토크가 아니라 **coupled oscillator(CPG)의 amplitude/frequency setpoint**를 출력. CPG가 rhythmic foot trajectory 생성.
- **Paper**: *CPG-RL: Learning Central Pattern Generators for Quadruped Locomotion* — Guillaume Bellegarda, Auke Ijspeert (2022). https://arxiv.org/abs/2211.00458 · *Visual CPG-RL* https://arxiv.org/abs/2212.14400
- **무엇을 바꾸나**: action space를 joint target → 4 oscillator(다리당 1)의 amp/freq/phase로 축소. limit-cycle 보장으로 외란 회복·gait 규칙성↑.
- **Go2/parkour 적용성**: **Med~High** — 평지/거친지형 robust gait엔 강력. 단 parkour의 비주기적 jump/climb엔 CPG 주기성이 제약이 될 수 있음(평지·계단 강점, 큰 obstacle 약점).
- **rsl_rl 통합 비용**: **Med** — 정책 output layer는 drop-in이지만 **CPG integrator(env step마다 oscillator ODE)** 가 필요 → `*_env.py`의 action 적용부 수정. 즉 module이 아니라 env action pipeline을 건드림.
- **이미 존재?**: 없음. (phase-conditioned 관련 구현 부재.)
- **public 구현 있음?**: 있음 (Bellegarda CPG-RL 코드, A1 deploy).
- **deploy 가능성**: **Very High** — oscillator 파라미터 극소, 이미 A1 실로봇 검증. Go2 직접 적용 용이.

### 4. Unified Locomotion Transformer (ULT) — Transformer policy
- **Method**: transformer가 knowledge transfer(imitation) + policy optimization을 단일 네트워크로 통합, teacher-student 단계 없이 직접 sim-to-real.
- **Paper**: *Unified Locomotion Transformer with Simultaneous Sim-to-Real Transfer for Quadrupeds* — Dikai Liu, Tianwei Zhang, Jianxiong Yin, Simon See (2025, **quadruped**). https://arxiv.org/abs/2503.08997 · 일반형 *Body Transformer* (morphology-aware, embodiment as attention mask)
- **무엇을 바꾸나**: MLP/GRU actor → self-attention. token = obs 그룹/관절 단위. teacher-student 우회.
- **Go2/parkour 적용성**: **Med** — 표현력↑이나 PPO+수천 env에서 transformer는 학습 wall-clock·VRAM 비용↑. parkour 효용은 검증되었으나 MLP 대비 ROI 불확실.
- **rsl_rl 통합 비용**: **Med** — `ActorCriticTransformer` 새 module은 drop-in이나, sequence/token 구성·positional encoding 설계 + recurrent 대비 학습 안정화 튜닝 필요. PPO loop 자체는 무수정.
- **이미 존재?**: 부분 — recurrent(GRU/LSTM) sequence 처리는 존재. delta = **recurrence → self-attention**(완전 신규 구조).
- **public 구현 있음?**: ULT 코드 부분 공개. Body Transformer 공개 구현 존재.
- **deploy 가능성**: **Med** — 작은 transformer면 Go2 가능하나 추론 latency가 MLP/GRU보다 큼. token 수 작게 유지 필수.

### 5. Hybrid Internal Model (HIM) — contrastive proprioceptive embedding
- **Method**: proprio history → explicit velocity + implicit stability embedding을 **contrastive(successor state)** 로 학습, PPO와 동시 최적화(HIO).
- **Paper**: *Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response* — Junfeng Long, Zirui Wang, Quanyi Li, Jiawei Gao, Liu Cao, Jiangmiao Pang (2023). https://arxiv.org/abs/2312.11460
- **무엇을 바꾸나**: privileged estimator/RMA adaptation을 **contrastive internal model**로 대체. teacher 없이 proprio만으로 외란·terrain을 implicit 추정.
- **Go2/parkour 적용성**: **High** — proprio-only, IMU+encoder만. sim-to-real 친화적, Aliengo/A1/Go1 deploy 검증.
- **rsl_rl 통합 비용**: **Med~High (algorithm-loop change)** — 핵심이 **contrastive loss + HIO step**이라 `algorithms/ppo.py update()`에 보조 손실/optimizer가 들어감. module만 추가하는 게 아님.
- **이미 존재?**: **부분 겹침** — `Estimator`(velocity 추정)와 RMA adaptation module이 이미 존재. **delta = network가 아니라 training objective**(contrastive successor-state HIO). 구조 자체는 estimator와 유사 → top3 제외 사유.
- **public 구현 있음?**: 있음 (InternRobotics/HIMLoco, legged_gym+rsl_rl fork).
- **deploy 가능성**: **Very High** — proprio-only, 50Hz policy, 실로봇 다수 검증.

### 6. DreamWaQ — CENet (β-VAE context-aided estimator)
- **Method**: single encoder + multi-head decoder β-VAE로 body velocity + latent terrain context를 implicit 추정(asymmetric actor-critic).
- **Paper**: *DreamWaQ: Learning Robust Quadrupedal Locomotion With Implicit Terrain Imagination* — I Made Aswin Nahrendra, Byeongho Yu, Hyun Myung (2023). https://arxiv.org/abs/2301.10602
- **무엇을 바꾸나**: explicit privileged scandot → **implicit terrain imagination**(VAE latent). proprio-only로 height/friction/obstacle 추론.
- **Go2/parkour 적용성**: **Med~High** — robust blind locomotion에 강함. parkour의 명시적 vision 활용과는 철학이 다름(implicit). robust fallback로 가치.
- **rsl_rl 통합 비용**: **Med~High (algorithm-loop change)** — β-VAE recon + KL loss가 별도 optimizer로 `update()`에 추가. CENet module 자체는 작음.
- **이미 존재?**: **부분 겹침** — `Estimator` + RMA adaptation과 역할 중복. **delta = β-VAE 생성/recon objective**(estimator는 supervised regression). 구조 유사 → top3 제외.
- **public 구현 있음?**: 있음 (Manaro-Alpha/DreamWaQ, curieuxjy/go2_dreamwaq — Go2 포팅).
- **deploy 가능성**: **Very High** — proprio-only, A1 야간 보행 deploy 검증. Go2 포팅 repo 존재.

### 7. Conditional / Wasserstein AMP (multi-skill latent imitation)
- **Method**: AMP discriminator를 WGAN(Wasserstein)으로 바꾸고 **latent skill embedding**으로 conditioning → multi-skill mode collapse 완화.
- **Paper**: *Learning Multi-Skill Legged Locomotion Using Conditional Adversarial Motion Priors* — Ning Huang, Zhentao Xie, Qinchuan Li (2025, **quadruped**). https://arxiv.org/abs/2509.21810 · Wasserstein 변형 근거 *HumanMimic: Wasserstein Adversarial Imitation* — Annan Tang 외 (2023, humanoid — WGAN loss formulation 참고용) https://arxiv.org/abs/2309.14225
- **무엇을 바꾸나**: 기존 AMP에 skill latent 조건 + WGAN-GP loss. 다양한 reference motion을 한 정책에 안정 imitation.
- **Go2/parkour 적용성**: **Med** — Go2-Imitation(현 mimickit 포팅 작업)과 직접 연결. parkour 자체보단 imitation 트랙에 가치.
- **rsl_rl 통합 비용**: **Med (algorithm-loop change)** — `amp_discriminator.py`는 존재, loss를 WGAN-GP로 교체 + latent conditioning이 `ppo_amp.py update()` 수정.
- **이미 존재?**: **부분 겹침** — `amp_discriminator.py`(AMP) + `estimator.py`의 DIAYN/LSD skill discriminator 이미 존재. **delta = WGAN-GP loss + conditional latent**(BCE→Wasserstein, skill conditioning). 구조 거의 동일, objective만 변경 → top3 제외.
- **public 구현 있음?**: HumanMimic/conditional-AMP 논문 코드 일부 공개.
- **deploy 가능성**: **High** — AMP는 deploy 시 discriminator 불필요(정책만 배포). 현 AMP 파이프라인과 동일.

---

## 요약: drop-in(싸다) vs 구조변경(비싸다)

### A. Drop-in 가능 (module 교체, PPO loop 무수정) — 싸다
- **MoE-Loco** — `ActorCriticMoE` 새 module. ★ 최저 비용
- **AME (attention map encoder)** — scandot encoder 자리 교체, 입력 동일
- **ULT/Transformer policy** — `ActorCriticTransformer` module (단 학습 튜닝 비용 있음)
- *(CPG는 output layer는 drop-in이나 env action pipeline 수정 필요 → 경계선)*

### B. 구조 변경 필요 (algorithms/ppo.py의 update()에 보조 loss·optimizer 추가) — 비싸다
- **HIM** — contrastive HIO step
- **DreamWaQ/CENet** — β-VAE recon+KL loss
- **Conditional/Wasserstein AMP** — WGAN-GP + latent conditioning (ppo_amp.py)
- **CPG** — env step마다 oscillator integrator

> 주의: B 그룹(HIM/DreamWaQ/CWAMP)은 "encoder 추가"처럼 보이지만 본질이 **training objective**라 `ppo.py`를 건드린다. 또한 셋 다 기존 `estimator.py`/`amp_discriminator.py`와 구조가 겹치고 **delta는 loss/objective**다(architecture 신규성 낮음) → top3에서 제외.

## Top 3 추천 (impact / effort 기준)

1. **MoE-Loco (Mixture-of-Experts actor)** — 진짜 novel 구조, drop-in module(Low), parkour multi-terrain + 현재 3-leg gait local-optimum(gradient conflict)에 직접 타격. deploy High(작은 expert MLP, 추가 센서 X). **최고 ROI.**
2. **AME (Attention-Based Map Encoder)** — 입력이 기존 height-scan과 동일(새 센서 X)이라 통합 마찰 최소, scandot encoder만 cross-attention으로 교체. parkour terrain 집중·해석가능성↑, ANYmal/biped parkour 100% 실증. deploy High.
3. **CPG-RL** — 구조 novelty + **최강 deploy(이미 A1 실로봇, oscillator 파라미터 극소)**. effort는 Med(env action pipeline)지만 robust gait·외란 회복으로 sim-to-real 가치가 큼. 평지·계단 강점/큰 obstacle 약점은 명시.

(Transformer/ULT는 drop-in이나 PPO+수천 env 학습비용·deploy latency로 4순위. HIM/DreamWaQ는 deploy 최강이나 기존 estimator와 구조 중복 + ppo loop 수정이라 "novel architecture" 관점에서 후순위.)

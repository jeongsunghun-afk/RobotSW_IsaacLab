# Parkour 알고리즘·네트워크 구조 개선 리서치 종합 (2026-06-10)

> 4개 병렬 리서치 스레드(코어 parkour 논문 / 인코더 아키텍처 / RL 알고리즘·gait 품질 / sim-to-real 배포)의 종합.
> 모든 GitHub URL은 각 스레드에서 WebFetch로 확인. 미확인 항목은 명시.
> **대상**: Unitree Go2 quadruped parkour, IsaacLab, PPO + teacher-student(DAgger) + AMP.

---

## 0. 핵심 수렴 진단 (4개 스레드 공통)

우리의 **3-leg gait local optimum**(한쪽 뒷다리 영구 들림)에 대해 네 갈래 리서치가 같은 결론에 도달:

1. **Symmetry는 못 고친다** — 3-leg gait는 L/R 미러 불변(왼발 들림 ↔ 오른발 들림 = 여전히 3-leg). data-aug / equivariant net 둘 다 bilateral 일관성만 강제. (스레드 C 분석, 우리 메모리 `project_parkour_symmetry_env` / `project_parkour_stride_reward_absent`와 일치)
2. **직접 해법 = 각 발에 강제 swing 의무 부여** — parked leg 평형을 직접 reward-negative로 만드는 per-foot 신호. 우리 메모리가 요구한 "escape-proof contact-duty signal"과 정확히 매칭.
3. 이를 구현하는 방식이 reward-side(Extreme Parkour foot-clearance, Walk These Ways per-foot contact), constraint-side(SoloParkour CaT), curriculum(soft→hard), 또는 구조적(CPG)으로 나뉨.

> ⚠️ 아래는 모두 **검증 대상 가설**이지 "확정 fix"가 아님. 적용 시 A/B 학습으로 검증 필요.

---

## TIER 1 — 즉시 적용 (저비용 · 오픈코드 · Go2/A1 매치)

### 1-A. Extreme Parkour — foot-clearance penalty ⭐ 최저비용·최고효율
- **출처**: Cheng, Shi, Agarwal, Pathak (CMU Pathak Lab), **ICRA 2024**. arXiv 2309.14341
- **코드**: https://github.com/chengxuxin/extreme-parkour (IsaacGym, A1)
- **IsaacLab 포트**: https://github.com/CAI23sbP/Isaaclab_Parkour (Go2 Teacher/Student, GPL-3.0)
- **메커니즘**: terrain edge 5cm 이내 발 접촉을 swing 중 penalize → `−Σcᵢ·M[pᵢ]`. 지형 스치기 대신 동적 발 들기 유도.
- **왜**: 우리 `_get_rewards()`에 reward term 1개 추가로 끝. **아키텍처 변경 0**. foot-drag / RL-leg 정박에 직접 작용. A1=Go2와 모터 토폴로지 동일.
- **주의**: foot position/contact는 sim privileged reward 신호 → obs 아님 → 우리 sim-to-real 제약(contact sensor obs 금지) 위반 아님.

### 1-B. Walk These Ways — per-foot swing-phase force penalty ⭐ 3-leg 직접 타격
- **출처**: Margolis, Agrawal (MIT Improbable-AI), **CoRL 2022**. arXiv 2212.03238
- **코드**: https://github.com/Improbable-AI/walk-these-ways (MIT 라이선스, **Go1 deploy 완비**, 1.4k★)
- **메커니즘**: 각 발에 독립 phase clock → commanded swing window 동안 ground-reaction force penalty. `r_c = Σ_foot[1−C_cmd]·exp(−|F|²/σ)`. parked leg는 매 cycle swing window에서 penalty 무한 누적 → micro-tap으로 game 불가.
- **왜**: 1-A와 동일 진단을 다른 각도로 구현. 둘 다 reward-side, 동시 적용 가능. gait command vector(8-dim)는 explicit gait 제어 보너스.
- **주의**: deploy 코드는 Go1용 `unitree_legged_sdk`(Go2 비호환) — 개념만 차용, deploy는 별도.

### 1-C. Robot Parkour Learning — soft→hard curriculum + Go2 deploy 레퍼런스
- **출처**: Zhuang, Fu, ... Finn, Zhao (ShanghaiTech/CMU/Stanford), **CoRL 2023 Oral** (Best Systems finalist). arXiv 2309.05665
- **코드**: https://github.com/ZiwenZhuang/parkour (MIT, **Go1+Go2 deploy 코드 포함**, 1.1k★)
- **메커니즘**: 3단계 — soft(투과 가능 장애물 penalty)로 자유 탐색 → hard 물리 제약 → DAgger로 5 skill을 단일 depth policy로 distill.
- **왜**: soft phase가 hard 제약 전 자유 탐색 허용 → gait floor-pinning(local optima) 방지. 우리 stack과 동일(IsaacGym+legged_gym+rsl_rl). **유일하게 Go2 deploy 코드 완비**(Jetson Orin + RealSense D435i + `unitree_sdk2`).

---

## TIER 2 — 중기 아키텍처 업그레이드 (Unified History Encoder 계획과 직결)

### 2-A. HIM (Hybrid Internal Model) — 코어 인코더 후보 ⭐
- **출처**: Long et al. (Shanghai AI Lab / Zhejiang / Tsinghua), **ICLR 2024**. arXiv 2312.11460
- **코드**: https://github.com/InternRobotics/HIMLoco (IsaacGym Prev4, rsl_rl `him_ppo.py`, **training only**, deploy 미공개. A1/Go1/Aliengo)
- **아키텍처**: 3-layer MLP(512→256→128), H=5 history(parkour는 H=10~20 권장), **dual-head** = velocity regression + 16-dim SwAV contrastive implicit latent. **단일 스테이지**(HIO+PPO 교대, 별도 distillation 불필요).
- **왜**: 우리 `project_parkour_unified_encoder_plan`(HIM 패턴 명시)과 정확히 일치. velocity는 sim에서 supervise 가능 / terrain·obstacle proximity는 implicit. asymmetric critic만 privileged. deploy 시 proprioception-only → 우리 제약 충족.

### 2-B. SoloParkour — CaT (Constraints as Terminations)
- **출처**: Chane-Sane et al. (LAAS-CNRS + NYU), **CoRL 2024**. arXiv 2409.13678
- **코드**: https://github.com/Gepetto/SoloParkour (IsaacGym, 3-stage 전체 공개. Solo-12, deploy 미포함)
- **메커니즘**: 안전 제약을 per-step termination 확률로 변환(hard pmax=1.0 / soft annealed). contact-duty·foot-clearance를 **fragile reward weight 대신 hard constraint로 강제**.
- **왜**: 우리 reward-tuning 취약성(floor-pinning)의 근본 해결책. CaT wrapper는 framework-agnostic → IsaacLab 이식 가능. Stage 2의 DDPG+demo는 DAgger보다 vision fine-tune sample-efficient.

### 2-C. 보조 인코더 옵션
- **DreamWaQ CENet** (KAIST, ICRA 2023): β-VAE 보조 head(next-obs 예측 + velocity regression). HIM 위에 ~50줄로 fuse, deploy 시 discard(추론 비용 0). 공식 코드 없음(community: Manaro-Alpha/DreamWaQ, go2 port: curieuxjy/go2_dreamwaq).
- **PIE** (Zhejiang, RA-L 2024, arXiv 2408.13740): Transformer cross-modal(depth CNN + proprio MLP) + GRU + **explicit foot-clearance regression head**. 단일 스테이지. 코드 미공개. → 아키텍처 영감(foot-clearance를 reward 아닌 auxiliary regression으로).
- **RMA** (Berkeley, RSS 2021): 1D-CNN adaptation module(k=50 history→8-dim latent). HIM 수렴 후 sim-to-real gap 발견 시 옵션. 부분 코드 antonilo/rl_locomotion.
- **Body Transformer** (Berkeley, CoRL 2024): kinematic graph attention. 공식 코드 https://github.com/carlosferrazza/BodyTransformer (A1 deploy 포함). HIM(시간) + BoT(공간) 보완 가능.

---

## TIER 3 — 배포 인프라 & AMP 대체 (참고)

### 배포 스택 (우리 IsaacLab 2.3.2와 직접 호환)
| 레포 | 매치도 | 비고 |
|------|--------|------|
| **unitreerobotics/unitree_rl_lab** | 공식 Unitree, IsaacLab 2.3, Apache-2.0 | Go2 obs/action 표준 + C++ LibTorch deploy `deploy/robots/go2/` |
| **fan-ziqi/robot_lab + rl_sar** | IsaacLab **2.3.2 정확 일치**, Apache-2.0 | rl_sar = ONNX+JIT 양쪽 지원, MuJoCo/Gazebo sim2sim, Ethernet Go2 |
| **ZiwenZhuang/parkour** | MIT | 유일 parkour Go2 deploy(Jetson+D435i+sdk2) |
| unitreerobotics/unitree_rl_gym | BSD-3 | Go2 obs/action convention + MuJoCo sim2sim 게이트 |

**배포 실무 디테일** (스레드 D):
- Go2는 `unitree_sdk2`/`unitree_sdk2_python` 필수 (구 `unitree_legged_sdk`는 Go2 비호환)
- 제어 주파수 50Hz 보편, 모터 내부 500Hz ZOH
- **IsaacLab ONNX export는 action_scale 미반영** (issue #2636) → 출력에 수동 곱셈 필요
- sim2sim(unitree_mujoco) 게이트 후 실로봇
- PD: Kp=30~40, Kd=0.65~1.0

### AMP 대체 (현재 AMP mode collapse 대안)
- **APEX** (MarmotLab, arXiv 2025): decaying BC prior(α 1.0→0) + multi-critic. **Unitree Go2에서 직접 deploy 검증**. https://github.com/marmotlab/APEX. AMP가 3-leg에 collapse하기 전 reference-guided warm-start 제공.
- **RSI (DeepMimic)**: 기존 AMP motion data로 reference state init → standing 근처 leg-disuse 패턴 회피. https://github.com/xbpeng/DeepMimic

### 구조적 옵션
- **CPG-RL / AllGaits** (EPFL, RA-L 2022): Hopf oscillator 4개로 action space 대체. inter-leg coupling이 leg parking을 물리적으로 어렵게(structural inductive bias). Go1 검증. https://github.com/MiladShafiee/DeepTransition. 통합 ~2-3일.

---

## 권장 적용 순서

1. **[Tier 1] Extreme Parkour foot-clearance + Walk These Ways per-foot contact reward** 동시 적용 (둘 다 reward term, 아키텍처 무변경) → 3-leg 직접 공략. A/B로 단독/병합 효과 분리.
2. **[Tier 1] soft→hard curriculum** (Robot Parkour Learning) 도입 → local optima 탈출 강화.
3. **[Tier 2] HIM 인코더**로 Unified History Encoder 계획 구현 (H=10~20). 단일 스테이지라 기존 PPO 루프에 통합 용이.
4. 1~3 후에도 잔존 시 **[Tier 2] CaT**로 contact-duty를 hard constraint화.
5. AMP 유지 시 **APEX decaying BC prior**로 교체, **RSI** warm-start 추가.
6. 배포는 **unitree_rl_lab + robot_lab/rl_sar** 기준 ONNX 파이프라인, parkour 레포로 Jetson+depth 스택 참고.

## 미적용 / 비권장
- ANYmal Parkour (ETH, Sci.Rob. 2024), DTC (ETH), PIE: 아키텍처는 우수하나 **코드 미공개** 또는 비-RL 의존(TAMOLS). symmetry-aug 개념만 차용.
- Pure symmetry(SymmLoco/Abdolhosseini): 3-leg 못 고침 — 보조 안정화로만.

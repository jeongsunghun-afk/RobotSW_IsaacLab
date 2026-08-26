# on-policy PPO 아키텍처 서베이 — leg_imitation_tracking 저속 이봉(hysteresis) 겨냥

**2026-08-26** · 작성: research-onpolicy-scaling agent
**전제**: `README.md`의 실패 모드 (A) — cmd 0.5 상승(1~64%) vs 하강(61~77%) 이봉 — 를 겨냥한 서베이.
`README.md` (A-1)의 "정지 근방에서 보행 위상이 관측 불가하다" 가설과 이 문서의 결론은 **독립적으로 도출됐고 일치한다**.

## TL;DR — 아키텍처가 맞는 레버인가

**부분적으로만 그렇다. 실패 모드를 반드시 둘로 쪼개서 판단해야 한다.**

- **"걷기 시작하는지 여부"(이봉/hysteresis, README (A))** → **아키텍처가 맞는 레버다.** 같은 관측에서
  서로 다른 정답 행동이 요구되는 것은 partial observability 신호이고, memory(recurrence) 또는
  명시적 phase 신호로 직접 겨냥 가능하다. 근거는 §2, §7.1~7.3, §7.5에 정리.
- **"일단 걸으면 저속을 못 내고 초과달성한다"(팀이 예비 관측한 125~142%)** → **아키텍처가 맞는
  레버라는 근거가 약하다.** 정지-국소최적점(reward penalty 항), 저속 학습의 일반적 어려움(고꾸라짐/
  국소최소), limit-cycle 보행의 물리적 최저속도 하한, AMP reference 데이터 커버리지 — 넷 다 문헌에
  기록된, 아키텍처와 **무관한** 대안 설명이며 어느 것도 배제되지 않았다. 근거는 §7.4.
- **결론**: 3-arm 예산은 첫 번째 하위 문제(이봉)에 쓰고, 두 번째 하위 문제(초과달성)는 별도의
  reward/커리큘럼/reference-data 조사 트랙으로 병행할 것을 권한다. 하나로 묶으면 한쪽의 null 결과가
  다른 쪽 탓으로 잘못 귀속될 위험이 있다.

**증거 등급 정의**: `on-policy-locomotion` = PPO/on-policy, 다리로봇(sim 또는 실기).
`on-policy-other` = PPO/on-policy이나 비-로코모션(게임, 조작, 애니메이션 캐릭터).
`off-policy` = SAC/TD3/value-based. `offline` = RL 루프 없는 지도학습/표현학습.
`extrapolated` = 직접 인용 가능한 1차 출처가 없고, 다른 맥락의 결과에서 유추한 추론.

---

## 0. 핵심 프레이밍

같은 명령(cmd 0.5)에서 서로 다른 정답 행동(정지 유지 / 보행 개시)이 요구된다는 것은 **partial
observability / history 의존** 신호이지, "매 순간의 행동 분포가 다봉이어야 한다"는 신호가 아니다.
Markovian 정책은 표현력을 아무리 올려도(mixture, diffusion, flow, VAE) 여전히 **현재 관측에만**
조건화된다 — 같은 관측에서 두 정답이 갈리는 문제라면, 그 관측만으로는 애초에 구분할 수 없다.
해소 경로는 둘뿐이다: **(a) 정책에 history를 주거나(recurrence), (b) 모호성을 없애는 명시적 입력을
추가한다(phase/mode 신호)**. "더 표현력 있는 단일-스텝 분포"는 이 실패 모드를 기계적으로 겨냥하지
못한다. 아래 순위는 이 기준으로 매겼다. (증거 등급 정의는 위 TL;DR 참조.)

---

## 1. 다봉/표현력 있는 정책 클래스 (on-policy PPO 하에서)

| 후보 | 메커니즘 | 증거 라벨 | log-prob/ratio 처리 | 추론비용/export | (A) 겨냥 여부 |
|---|---|---|---|---|---|
| **Diffusion policy + PPO** (DPPO, D2PPO arXiv 2508.02644) | 매 denoising step을 하나의 MDP 전이로 취급, K-step chain 전체의 log-likelihood 합으로 ratio 계산 | `on-policy-other` (로봇 조작 벤치마크, 로코모션 PPO 결과 없음) | tractable하지만 비쌈 — PPO update마다 K-step chain 재생 필요 | **높음** — 스텝당 1회가 아닌 K회 forward pass, TorchScript/ONNX로 iterative sampling loop export는 rsl_rl/IsaacLab 툴체인에서 검증된 바 없음 | **아니오** — 스텝별 표현력 추가일 뿐, history 없음 |
| **Flow-matching + PPO** (FPO, PolicyFlow arXiv 2602.01156) | flow-matching objective의 per-sample 변화량으로 importance sampling 재정식화, full-trajectory likelihood 회피 | `on-policy-other` (일반 continuous-control 벤치마크, 로코모션 아님. PolicyFlow는 2026-01 preprint, 재현 사례 얕음) | diffusion보다 저렴하나 여전히 multi-step sampling | **높음** — 동일 사유 | **아니오** |
| **Mixture-of-Gaussians 정책 head** | K-component GMM output | 문헌 자체가 "on-policy actor-critic에서 largely unexplored"라 명시, 로코모션 증거 없음 | log-sum-exp ratio가 수치 불안정하다고 지적됨 | **낮음**(forward pass는 1회) — 표현력 계열 중 가장 저렴 | **오진단 가능성** — 같은 관측에서 다른 정답이 요구되는데, GMM도 그 관측에만 조건화되므로 컴포넌트를 고를 신호가 없다. 경계에서 두 모드를 "섞는" 결과(mode-averaging)가 될 가능성이 높다 |
| **잠재변수/VAE 다중모드 정책** (CALM arXiv 2305.02195, ASE, "Multimodal Bipedal Locomotion and Implicit Transitions" arXiv 2303.05711) | 외부 latent/mode 변수가 행동을 선택하고 정책이 여기 조건화됨 | CALM/ASE: `on-policy-other` (PPO, 물리 기반 시뮬레이션 캐릭터, 실기 아님). 2303.05711은 walk/jump/leap/idle을 autoencoder-latent 조건화로 다루어 주제상 직접 관련되나, PPO 여부·정확한 아키텍처·결과를 원문에서 확인 못함 — **미확인, 원문 정독 필요** | 표준 단일 스텝 log-prob(잠재는 조건 입력일 뿐) — 저렴 | **낮음~중간** — 조건 입력 하나 추가, export 용이 | **간접적** — 이 프레임워크들은 "무언가"(planner, 사람, 스크립트 커리큘럼)가 latent/mode를 **선택**한다고 가정한다. "정책이 history로부터 자율적으로 정지/보행을 결정"하는 문제를 풀지 않고 한 단계 위로 떠넘길 뿐이다(그 선택 자체가 다시 memory 문제) |

**이 축 종합**: 실기 배포 사례가 있는 on-policy-locomotion 증거를 가진 후보가 없고, 가장 저렴한 둘
(GMM, latent 조건화)도 history 의존 모호성과 메커니즘이 안 맞는다. diffusion/flow는 50Hz 임베디드
제약에 실제 위험 요소다. **이 축에 3-arm 예산 중 하나를 쓰는 것은 권장하지 않는다.**

---

## 2. 순환/메모리 아키텍처 — 메커니즘이 일치하는 후보

| 후보 | 증거 라벨 | 세부 |
|---|---|---|
| Cassie 이족 LSTM 액터 (Siekmann et al., RSS 2020) | `on-policy-locomotion`, **실기** | 액터+크리틱 전체가 LSTM, PPO. 순간 관측이 non-Markovian이라는 동기에서 memory 도입 |
| ZSL-RPPO (arXiv 2403.01928) | `on-policy-locomotion` (Isaac Gym 4족, PPO+GRU) | "partial observability 환경에서 직접 recurrent net 학습" — (A)와 동일한 partial-observability 프레이밍 |
| DreamWaQ (arXiv 2301.10602) | `on-policy-locomotion`, 실기(Unitree A1) | GRU 기반 estimator(전체 액터 아님)를 MLP 본체 옆에 병렬 배치 — 더 가벼운 recurrence 패턴 |
| **Gait-Conditioned RL with Multi-Phase Curriculum for Humanoid Locomotion** (arXiv 2505.20619) | `on-policy-locomotion`, **실기(Unitree G1)** | **서베이 전체에서 가장 직접적으로 일치하는 논문.** 단일 recurrent 정책이 정지·보행·주행·**그 사이 전환**을 수행 — 정확히 우리 실패 모드(같거나 비슷한 명령에서 정지↔보행 전환). 메커니즘: recurrent 정책 + one-hot gait ID로 활성화하는 "compact reward routing" + 커맨드 공간을 단계적으로 넓히는 커리큘럼. 정확한 RNN 종류/크기, PPO 여부(생태계상 유력하나 원문에서 명시 확인 못함)는 **미확인 — 이 arm을 설계하기 전에 원문 정독 필요**, 방향성은 recurrence + 명시적 mode 조건화가 이 실패 계열의 실제 처방임을 뒷받침 |

**메커니즘 적합성**: recurrence는 "이 관측에 어떻게 도달했는가"(2초 전에 서 있었는지, 이미 보행
중이었는지)를 정책에 직접 제공한다 — 이는 상승/하강 비대칭이 시사하는, 누락된 정보와 정확히 일치한다.
서베이 전체에서 **실기 on-policy-locomotion 선례가 있으면서 진단된 근본 원인과 메커니즘이 일치하는
유일한 후보**다.

**검증된 구현 함정(추정 아님, GitHub에서 확인)**: rsl_rl의 `ActorCriticRecurrent`는 `lstm`/`gru`
둘 다 지원하지만, **IsaacLab exporter(`isaaclab_rl/rsl_rl/exporter.py`)가 LSTM의 (hidden, cell)
튜플 상태 전달을 하드코딩해서 GRU에서 깨진다** — isaac-sim/IsaacLab **issue #3008**, 미해결 확인.
**arm에는 `rnn_type="lstm"`을 쓸 것** (GRU를 쓰려면 exporter 수정 작업을 먼저 예산에 넣어야 함).

---

## 3. 주기/위상 조건화 아키텍처

| 후보 | 증거 라벨 | 비고 |
|---|---|---|
| CPG-RL / Visual CPG-RL (Bellegarda et al.) | `on-policy-locomotion` | 정책이 관절 목표 대신 CPG 파라미터(주파수/진폭)를 출력; oscillator 상태가 명시적이고 시간에 걸쳐 위상을 실어 나름 — 사실상 손설계된, 해석 가능한 recurrence |
| DeepPhase / Periodic Autoencoder (Starke et al., SIGGRAPH 2022) | `offline` (모션캡처 데이터의 비지도 표현학습, 그 자체는 RL 아님) | 모션 데이터에서 저차원 주기 latent(phase manifold) 학습. 후속 FLD가 이 latent를 다운스트림 RL에 투입하나, on-policy/PPO 로코모션 결과는 검색으로 확인 못함(미확인) |
| one-hot gait-ID / 명시적 duty-phase 관측 (위 2505.20619에서 사용) | `on-policy-locomotion`, 실기 | 이 축에서 가장 저렴한 버전: 관측 벡터에 명시적 "위상" 또는 "최근 정지/보행 여부" 스칼라·one-hot을 추가. 네트워크 아키텍처 변경이 전혀 아니고 관측 엔지니어링 변경 — 추론 비용 증가 0, export 위험 0 |

**메커니즘 적합성**: phase/mode 조건화도 **partial observability를 직접 공격**하는데, 암묵적 memory
대신 명시적 feature로 모호한 상태를 구분 가능하게 만든다. recurrence보다 **더 저렴하고 위험이 낮은
보완/대안**으로 강하게 고려할 가치가 있다 — 추론 비용이 0이고 export 위험도 0(recurrence는 배포
런타임에 hidden-state 관리 부담을 추가함).

---

## 4. AMP/GAIL mode collapse

우리 PPO+AMP 설정과 직접 관련되며, 문헌에 기록되어 있다:
- **ASE** — unconditional discriminator + diversity loss로 latent가 비슷한 행동으로 collapse하는
  것을 방지; "motion과 latent 간 대응 부재가 mode-collapse를 유발한다"고 명시.
- **CALM** (arXiv 2305.02195) — **conditional discriminator**로 AMP/ASE의 mode collapse를 해결:
  정책이 데이터셋의 **각** 모션을 재현하도록 강제(unconditional discriminator는 "이게 real motion
  *중 하나*처럼 보이는가"만 체크하므로, walking 모션이 수적으로 우세하면 standing-still 같은 희귀
  모션을 정책이 버려도 통과시킴).
- Terrain/skill-conditioned discriminator (T-GMP arXiv 2606.06944, "Multi-AMP")는 여러 prior가
  같은 task를 커버할 때 서로 다른 스타일이 하나로 collapse하는 것을 막기 위해 이를 확장.
- 전부 `on-policy-other`(시뮬레이션 물리 캐릭터 또는 일반 로코모션, 실기 미검증) 라벨.

**우리 버그에 대한 직접 가설**: reference dataset에 standing-still 클립보다 walking 클립이 압도적으로
많다면(로코모션 imitation 데이터셋에서 충분히 있을 법함), AMP discriminator가 "정지 유지"를 walking
위주 참조 분포에서 벗어난 것으로 암묵적으로 페널티를 줄 수 있다 — 이는 명령 개시 경계 근처에서 정책이
항상 보행을 시도하도록 편향시켜, 정확히 관측된 비대칭(일단 걷기 시작하면 유지=하강 성공,
정지에서 개시=상승 실패)으로 나타난다. **저렴하게 검증 가능**: reference 클립 구성에서 standing-still
프레임 비율을 확인하고/또는 discriminator를 phase/mode 라벨로 조건화(CALM식)해서 정당한 정지 행동에
대한 암묵적 페널티를 제거.

---

## 5. 3-arm + baseline 예산에 대한 순위 제안 (구버전 — §7.7~7.9가 최종본)

> **이 섹션은 RMA 구조 정밀 진단(§7) 이전에 작성된 1차 초안이다. 최종 순위·정확한 인용·`extrapolated`
> 라벨링은 §7.7(랭크드 숏리스트)과 §7.8(탈락 리스트)을 참조할 것.** 이 섹션은 추론 과정 기록으로만
> 보존한다.

**baseline (필수 대조군)**: 현재 `[512,256,128]` 단봉-가우시안 MLP 액터, 무변경 — 동시 진행되는
reward/curriculum/dataset 변경으로부터 아키텍처 효과를 분리하기 위해 필요.

### Arm 1 — Recurrent 액터 (LSTM), 최우선
- 변경: `ActorCriticRecurrent`, `rnn_type="lstm"`, 단일 LSTM layer(hidden size ≈128, 가장 작은
  MLP layer에 맞춤), MLP 본체 차원은 무변경.
- 1순위 이유: history 의존 모호성과 메커니즘이 일치하는, 실기 on-policy-locomotion 선례를 가진
  유일한 후보(Cassie, ZSL-RPPO, arXiv 2505.20619의 정지/보행/주행/전환 결과).
- 비용: 낮음(rsl_rl 네이티브 지원). Export: **LSTM 필수, GRU 금지** — IsaacLab exporter의 GRU
  export가 확인된 미해결 버그(issue #3008).
- 위험: 배포 런타임이 50Hz 제어 루프에서 hidden+cell state를 이월해야 함(작지만 실질적 복잡도 증가);
  스텝당 연산량이 MLP보다 다소 높으나 diffusion/flow/MoE/transformer 대비 훨씬 저렴.

### Arm 2 — 명시적 phase/mode 관측 feature
- 변경: 관측에 명시적 스칼라 또는 one-hot 입력(예: 지수평활된 최근 duty-cycle 추정치, 또는
  "최근 N ms 이내 stepping했는가" 지시자) 추가, 그 외 네트워크 아키텍처 무변경. 같은 신호를
  AMP discriminator 입력에도 반영하면 §4 가설을 덤으로 검증 가능.
- 2순위 이유: (A)의 메커니즘과 일치(recurrence 없이 partial observability 해소), 추론 비용상
  사실상 무료(입력 채널 1개 추가), export 위험 0. Arm 1과 결합 가능성도 있으나 예산상 단독 arm으로
  "명시적 해소만으로 충분한가"를 깨끗하게 테스트.
- 위험: 손설계된 phase/duty 신호 자체가 짧은 history 창에서 계산된 "빈약한 memory"라, 실제 의존
  윈도우가 필터의 시정수보다 길면 부분적으로만 해소될 수 있음.

### Arm 3 — 조건부/skill-aware AMP discriminator (CALM식, 또는 최소 duty-라벨 discriminator)
- 변경: discriminator를 Arm 2와 동일한 phase/mode 라벨(또는 클립-출처 라벨)로 조건화해서
  standing-still을 암묵적으로 off-manifold 취급하지 않도록 함. 액터 아키텍처는 baseline과 동일(
  plain MLP).
- 3순위 이유: §4의 자체 가설(팀 리드 질문 4)을 직접 검증, 액터 용량/memory와 직교적이라 "discriminator가
  편향 원인인가 vs 액터의 memory 부재가 원인인가"를 분리 가능, 비용 저렴(loss-side 변경만).
- 위험: reference dataset에 실제로 standing-still 클립이 부재하면 이 arm의 상한은 discriminator
  조건화가 아니라 데이터 커버리지에 의해 제한됨 — 실행 전 데이터셋 감사 권장.

### 명시적 탈락 사유
- **Diffusion/flow-matching PPO (DPPO/FPO/PolicyFlow)**: 탈락 — 로코모션 on-policy 증거 없음,
  50Hz 임베디드에서 검증 안 된 높은 추론 비용(multi-step sampling), 비표준 TorchScript/ONNX export,
  history 의존 모호성을 메커니즘상 겨냥하지 못함(단일 스텝 행동 분포만 풍부화).
- **Mixture-of-Gaussians 정책 head**: 탈락 — 저렴하지만 메커니즘 불일치(같은 관측·다른 정답 ⇒
  그 관측에만 조건화된 GMM은 컴포넌트를 고를 신호가 없어 mode-averaging이 예상됨); 문헌 자체가
  on-policy 학습에서 수치 불안정하다고 지적. Arm 1~3이 격차를 못 줄이고 모호성 창이 예상보다 짧다고
  의심될 때의 향후 저비용 ablation으로만 보류.
- **잠재조건화 다중모드 정책(CALM/ASE를 액터 latent selector로, VAE mode latent)**: 단독 arm으로는
  탈락 — 테스트 시점에 latent를 고를 외부 장치가 필요해 자율 의사결정 문제를 해소가 아니라 이전만
  시킴. (discriminator 조건화 아이디어는 유지되어 Arm 3로 재활용됨.)
- **DeepPhase/PAE 학습된 주기 latent**: 이번 예산에서 탈락 — Arm 2의 더 무거운 버전으로 흥미롭지만
  별도 pretraining 단계(모션 데이터 autoencoder)가 필요하고 다운스트림 on-policy RL 증거(FLD)가
  미확인. Arm 2의 손설계 phase 신호가 부분적 성공을 보이고 학습된 phase manifold가 더 나은지 보고
  싶을 때 재검토.
- **원 서베이의 스케일링 축 전체(BRO, SimBa, SimbaV2, HL-Gauss, spectral norm, MoE, transformer
  액터, network sparsity, periodic reset)**: 재확인 탈락 — 다중모드/hysteresis를 전혀 겨냥하지
  않음(용량/plasticity를 겨냥), 다수가 50Hz 임베디드 제약과 정면 충돌(transformer, MoE), 실기
  on-policy-locomotion 증거 없음. (섹션 6에 원 서베이 상세 보존.)

### 남은 정직한 공백
- arXiv 2505.20619의 정확한 RNN 종류/크기, 알고리즘(생태계상 PPO로 추정되나 원문에서 명시 확인 못함),
  정지/보행 전환 성공률 정량치를 확인 못했다 — **이것을 템플릿으로 arm을 설계하기 전에 원문을 정독할
  것**. 현재 유일한 최적 선례인데 abstract 수준 확인만 했다.
- arXiv 2303.05711의 학습 알고리즘(PPO 여부)과 "autoencoder-latent 조건화" 이상의 정확한 아키텍처를
  확인 못했다 — 동일한 유보.
- 4족/다리로봇 RL 문헌에서 **고정 명령에서의 상승/하강 비대칭 자체**를 측정하거나 고친 논문은
  찾지 못했다 — 우리의 실패 시그니처(속도 임계값에서의 "gait transition"과는 별개 문제 프레이밍)는
  문헌에 이름 붙은 기성 해법이 없는, 원본 관찰로 보인다. Arm 1~3은 가장 가까운 선례에 기반한 근거
  있는 가설로 취급하고, 이미 풀린 문제의 재현으로 취급하지 말 것.

---

## 6. 원 서베이 (일반 스케일링 축, 참고용 — 위 §5에서 전부 탈락 처리됨)

### 6.1 의도적으로 PPO/on-policy 네트워크를 스케일업하는 논문

**BRO (Bigger, Regularized, Optimistic)** — Nauman et al., NeurIPS 2024 spotlight.
https://arxiv.org/abs/2405.16158
- critic을 강한 정규화(LayerNorm + weight decay/L2) + optimistic exploration으로 확장.
- 증거: `off-policy` (SAC 계열, DMC/MetaWorld/MyoSuite 포함 Dog/Humanoid). PPO 결과 없음.
- rsl_rl 비용: 낮음 — critic에만 residual+LayerNorm 블록 적용.
- 위험: PPO의 짧은 horizon, on-policy 업데이트 체제로 전이될지 미검증.

**SimBa** — Lee et al., NeurIPS 2024. https://arxiv.org/abs/2410.09754
- Residual FF 블록 + running-stats 관측 정규화 + LayerNorm. **on-policy PPO 결과(Craftax)** 명시적
  주장.
- 증거: 이 목록에서 **최선의 on-policy PPO 1차 증거**이나, Craftax는 로코모션이 아님. 로코모션 PPO
  결과 없음.
- rsl_rl 비용: 낮음~중간 — drop-in residual+LayerNorm MLP 교체.
- 위험: 로코모션 특유 증거 부재, action space·reward scale이 크게 다름.

**SimbaV2 / Hyperspherical Normalization** — DAVIAN Robotics, 2025.
https://arxiv.org/abs/2502.15280, 코드: https://github.com/dojeon-ai/SimbaV2
- LayerNorm 대신 L2-정규화(가중치+feature를 hypersphere로), residual을 learnable LERP로 대체,
  MSE Bellman loss를 categorical/KL value loss로 대체.
- 증거: `off-policy`만(57개 continuous-control task). PPO/on-policy 결과 없음.
- rsl_rl 비용: 높음 — feature 정규화 방식과 critic loss(categorical)를 함께 재구성해야 함.
- 위험: hypersphere 제약과 categorical value loss가 결합되어 있어 부분 도입 미검증.

**"Stop Regressing" / HL-Gauss categorical value loss** — Farebrother et al. (Google DeepMind),
2024. https://arxiv.org/abs/2403.03950
- value regression을 classification으로 재정식화(HL-Gauss가 스케일업 시 two-hot/C51보다 우수).
  Atari(value-based, SoftMoE), 로봇 조작(Q-transformer), 체스, Wordle에서 테스트 — 논문 자체에는
  PPO/continuous-control 로코모션 결과 없음.
- 후속: "Start Classifying: Categorical Critics for LLM RL" (arXiv 2608.02181, 2026)이 **HL-Gauss
  PPO가 수학추론 RLHF에서 PPO/DAPO baseline보다 개선**된다고 보고 — HL-Gauss+PPO의 가장 근접한
  1차 증거지만 물리적 continuous control이 아닌 LLM RLHF.
- 증거: value-based 스케일링엔 강함; PPO-continuous-control엔 간접적/약함(`on-policy-other`).
- rsl_rl 비용: 중간 — critic output이 categorical head + cross-entropy loss로 바뀜, GAE advantage
  계산에 expected-value decode 단계 필요. 액터는 안 건드림.
- 위험: 다리로봇 task는 커리큘럼 단계별로 reward scale이 비정상적(Atari/DMC 고정 스코어와 달리) —
  bin range 재튜닝 또는 adaptive support 필요.

**Spectral Normalization for Deep RL** — Bjorck et al. (Cornell), NeurIPS 2021.
https://arxiv.org/abs/2105.05246 (+ 후속 "Stability-Certified RL via Spectral Normalization",
arXiv 2012.13744)
- 가중치 행렬을 1-Lipschitz로 정규화; 불안정 없이 더 깊은 critic 가능; hard continuous-control에서
  시연.
- 증거: 주로 `off-policy`(SAC 계열)로 알려짐; PPO 커버리지는 원문 추출 실패로 **미확인** — 단정하지
  않고 플래그만.
- rsl_rl 비용: 낮음 — `nn.Linear` 래퍼로 쉽게 부착.
- 위험: 주로 critic 대상; PPO의 clipped surrogate/액터 업데이트와의 상호작용 불명확.

**Network Sparsity Unlocks Scaling Potential** — Ma et al. (ICML 2025). https://arxiv.org/abs/2506.17204
- 학습 전 one-shot random pruning(static sparsity)이 dense SOTA 아키텍처보다 파라미터 효율과
  plasticity loss/gradient interference 저항성을 개선.
- 증거: streaming RL·visual RL 커버; on-policy/PPO 커버리지 검색 스니펫으로 미확인 — 1차 확인 필요.
- rsl_rl 비용: 낮음 — 초기화 시 고정 pruning mask 적용.
- 위험: 이 목록에서 가장 최신/재현 사례가 적은 결과.

**Mixture-of-Experts**
- MoE-Loco (arXiv 2503.08564) — **멀티태스크** 로코모션(4족+2족 gait 전환) 아키텍처, gradient
  conflict 감소가 동기, 단일 task 용량 확장이 아님. 실기 검증됨. locomotion RL 파이프라인이라
  on-policy MoE+실기 증거는 맞으나, 동기가 "단일 17-DOF tracking task에 더 큰 net"과 다름.
- CMoE (arXiv 2603.03067) — 유사한 멀티태스크 프레이밍.
- HL-Gauss 논문이 인용한 SoftMoE-on-Atari 결과는 `off-policy`(value-based).
- rsl_rl 비용: 중간~높음 — router + expert 선택 로직, load-balancing loss, 추론 시 분기(임베디드
  추론 비용 예산에 불리, 배포 시 expert pruning/merge 없이는).
- 위험: MoE의 입증된 이점은 멀티태스크 gradient-conflict 감소에 특정됨 — **단일** task(우리
  imitation-tracking)에는 적용 근거가 약함.

**Transformer/attention 기반 PPO 로코모션**
- LA-PPO — LSTM + multi-head attention을 actor-critic에 결합해 history를 비균등 가중.
- Unified Locomotion Transformer(ULT, arXiv 2503.08997) — PPO + transformer distillation loss로
  학습, 배포 시엔 proprioception만으로 zero-shot; transformer는 학습용, 배포 전 작은 net으로 distill.
- Attention 기반 지형/맵 인코딩(AME-2, arXiv 2601.08485) — 공간 맵 토큰에 대한 attention, 핵심
  proprioceptive 액터 MLP 자체가 아님.
- 증거: `on-policy-locomotion` 1차 증거 존재하나, attention은 일관되게 *시퀀스/공간 인코딩*(history,
  지형)용이지 핵심 액터 MLP의 전면 대체가 아님. 50Hz 임베디드에 전체 transformer 액터를 배포한
  사례는 없음 — ULT도 명시적으로 배포 전 distill.
- rsl_rl 비용: 중간(history encoder) ~ 높음(핵심 정책). 추론 지연이 MLP 대비 실질적으로 증가 —
  배포 목표와 직접 충돌하는, 이 서베이에서 가장 뚜렷한 위험.

### 6.2 SOTA 다리로봇 PPO가 실제로 쓰는 아키텍처

문헌 전반에서 지배적 패턴은 여전히 **얕은 2~3층 MLP, 128~512 hidden, ELU/tanh/CELU, 정규화 레이어
없음**, 간혹 작은 recurrent(GRU/LSTM) 유닛을 history/estimation용으로 병렬 배치:

| 시스템 | 액터 구조 | 비고 | 출처 |
|---|---|---|---|
| rsl_rl/IsaacLab 기본값 | `ActorCritic`, hidden `[256,256,256]` ELU(repo 기본); IsaacLab 예제는 `[512,256,128]` ELU도 흔함 | 정규화·residual 없음 | github.com/leggedrobotics/rsl_rl |
| ANYmal (Rudin et al., "Learning to Walk in Minutes") | 작은 MLP, massively parallel(수천 env)이 스케일 축 | 스케일 축이 **net 깊이가 아니라 env 수/batch** | arXiv 2109.11978 |
| Extreme/Humanoid Parkour (2406.10759) | 액터 MLP `[512,256,128]` CELU + GRU(256,1층); scandot 인코더 MLP `[128,64]` CELU | **우리 현재 `[512,256,128]`과 정확히 일치** | arXiv 2406.10759 |
| Walk These Ways | 단일 MLP 정책 + 별도 estimator MLP `[256,128]` ELU | transformer/MoE 없음 | arXiv 2212.03238 |
| Cassie 이족(Siekmann et al., RSS 2020) | **LSTM** 액터/크리틱(MLP 아님), PPO clipped objective, MuJoCo+DR | memory를 위해 깊이/너비 대신 recurrence 선택 | RSS 2020 |
| Unitree H1_2/G1 | 3층 MLP `[256,128,64]` tanh(H1_2, 공유 actor/critic) 또는 `[512,256,128]`(G1) | 50Hz ONNX 온보드 추론 | github.com/unitreerobotics/unitree_rl_gym |
| Booster T1 | PPO, asymmetric actor-critic | 온보드 CPU 50Hz JIT — 추론비용 우선 설계 | arXiv 2506.15132 |
| ASAP/HumanoidVerse | rsl_rl PPO 기반, 표준 actor-critic MLP + phase-based motion tracking | 특이 net-scale 주장 없음 | RSS 2025 |
| MimicKit | actor/critic MLP + AMP식 discriminator(우리 설정과 동일 패턴) | character-agnostic(2족+4족) | arXiv 2510.13794 |
| DreamWaQ | asymmetric actor-critic + 작은 **GRU 기반 context/estimator**(~36-dim hidden) + MLP 액터 | recurrence가 estimator에 국한, 액터 본체엔 없음 | arXiv 2301.10602 |

**시사점**: 실기 배포된 SOTA 다리로봇 PPO 시스템 중 넓거나 깊은 정규화 residual net, transformer
액터, MoE를 핵심 정책 본체로 쓰는 사례는 없다. plain MLP에서 일관되게 벗어나는 유일한 지점은
**작은 recurrent 유닛**(GRU/LSTM)이며, 이는 §2의 Arm 1 제안과 직결된다.

### 6.3 PPO 스케일업 시 실패 모드/주의점

- **PPO 특유 batch/scale 저하** [`extrapolated`]: "PPO 성능은 scale-up ratio가 커질수록 심하게
  저하되며, 감소된 LR(lr/scale-up-ratio 또는 lr/sqrt(scale-up-ratio))이 완화하나 해소하지는 못한다"
  — 특정 논문 하나로 못 박지 못함(검색 엔진 종합), **방향성으로만 취급, 인용 근거로 쓰지 말 것**.
  net 크기 축이 아니라 batch/parallelism 축이나, massively-parallel IsaacLab식 학습에서 상호작용함.
- **on-policy RL의 plasticity loss는 만연함** — "A Study of Plasticity Loss in On-Policy Deep RL"
  (NeurIPS 2024 spotlight, arXiv 2405.19153): 많은 off-policy plasticity 처방이 on-policy에서는
  **실패하거나 오히려 악화**시키며, continual regularizer(soft Shrink+Perturb + LayerNorm)가 최선
  이었음(gridworld/Montezuma/ProcGen, 로코모션 아님이나 on-policy 특유 결과).
- **Dormant neurons** — "The Dormant Neuron Phenomenon in Deep RL"(Sokar et al., ICML 2023,
  arXiv 2302.12902): 현상 확립 + ReDo 처방(post-activation이 0에 가까운 뉴런 주기적 재초기화).
  원래 value-based/off-policy이나 스케일된 RL net 전반의 용량 저활용 메커니즘으로 널리 인용됨.
- **일반 plasticity 서베이**: arXiv 2411.04832 — dormant-neuron 비율과 gradient-norm decay가
  가장 신뢰할 만한 조기 경고 지표.
- **Trust-region/KL 상호작용**: 직접 증거 못 찾음. on-policy plasticity 논문의 프레이밍상 합리적
  우려이나, 정량화한 1차 출처는 못 찾음 — 확립된 발견이 아니라 열린 질문으로 플래그.
- **Sim-to-real 추론비용**: 찾은 모든 실기 배포 SOTA 시스템(Unitree, Booster, ASAP)이 ONNX/JIT로
  50Hz 온보드 CPU 추론을 명시적으로 최적화 — 3층 MLP(+작은 GRU)보다 큰 것을 실기에 배포한 사례
  없음. 배포 목표를 가진 `leg_imitation_tracking` 정책에 대해 깊이/너비를 맹목적으로 늘리는 것에
  반대하는 가장 구체적인 증거.
- **시뮬레이터 과적합**은 검색된 어떤 논문도 직접 다루지 않음(서베이 공백 — 지어내지 않고 플래그).

### 6.4 큰 net을 작동시키는 정규화/아키텍처 재료

| 재료 | 출처 | on-policy 증거? | rsl_rl 비용 |
|---|---|---|---|
| LayerNorm + residual block | SimBa (2410.09754) | 있음(Craftax PPO), 로코모션 아님 | 낮음 |
| Hyperspherical norm + learnable LERP residual | SimbaV2 (2502.15280) | 없음(off-policy만) | 높음 |
| Categorical value loss(HL-Gauss) | Farebrother et al.(2403.03950); PPO 변형(2608.02181, LLM RLHF) | 간접적(LLM-RL, continuous control 아님) | 중간 |
| Spectral normalization | Bjorck et al.(2105.05246) | 미확인(off-policy 위주로 추정 — 확인 필요) | 낮음 |
| 초기화 시 one-shot sparsity/pruning | Ma et al.(2506.17204) | 미확인, PPO 확인 필요 | 낮음 |
| 주기적 reset — ReDo/Shrink+Perturb | Sokar et al.(2302.12902); on-policy 변형(2405.19153) | **있음**, on-policy 특유, 이 서베이에서 가장 근거 튼튼한 regenerative 처방 | 낮음~중간 |
| 작은 recurrent 유닛(GRU/LSTM)을 MLP 본체에 추가 | Cassie(RSS 2020), DreamWaQ(2301.10602), Extreme Parkour(2406.10759) | **있음**, 실기 배포 다수 | 낮음(rsl_rl이 `ActorCriticRecurrent`로 이미 지원) |
| 멀티태스크 gradient-conflict 감소용 MoE | MoE-Loco(2503.08564) | 있음, 단 다른 문제(멀티태스크)를 풀며 단일-task 용량 문제와는 다름 | 중간~높음 |

**주의**: ReDo/Shrink+Perturb(periodic reset)는 multimodality/hysteresis와 무관한 축이다 —
plasticity loss를 겨냥하지 다중모드 문제를 겨냥하지 않으므로, §5의 3-arm에는 포함하지 않았다.

---

## 7. 심화 질의응답 — RMA 구조 정밀 진단 + 저속 floor가 아키텍처 문제인가

**전제 갱신**: 실 코드 확인 결과 baseline은 `ActorCriticRMA`(teacher-student/DAGGER)다. PPO rollout 동안
액터 입력은 `[proprio(57), priv_explicit(6), enc(priv_latent)]`이고, `priv_latent`(38차원 = base_mass +
base_com + joint_stiffness_ratio + joint_damping_ratio)는 **에피소드 내내 상수인 물리 파라미터**다.
10-step history는 별도 `history_encoder`가 소비하며 이는 privileged encoder 출력을 모사하도록 DAGGER로
학습되고, runner는 `it % 20 == 0`(전체 iteration의 5%)에서만 이 경로로 rollout한다. 배포/평가
(`act_inference`)는 항상 history 경로를 쓴다 — 즉 **PPO가 최적화하는 액터는 gait phase에 대해
memoryless이고, 동시에 train/eval 입력 불일치가 있다.**

### 7.1 가장 중요한 발견: RMA 경로를 고쳐도 이봉은 안 풀린다

RMA 원 논문(Kumar et al., arXiv 2107.04034)에서 extrinsics vector `e_t`를 직접 확인했다: **질량+위치(3),
모터 강도(12), 마찰(1), 지형 높이(1) = 17차원, 전부 준정적(quasi-static) 환경 파라미터**이고 시간적/보행
상태는 전혀 없다. 우리 코드의 `priv_latent`(base_mass+base_com+joint_stiffness/damping ratio, 38차원,
"에피소드 내내 상수")는 **이 패턴과 정확히 일치한다** — 즉 우리 팀의 읽기는 오독이 아니라 RMA 계열의
표준 설계 그대로다. Extreme Parkour와 "Rapid Locomotion via RL"(Margolis et al., RSS 2022, arXiv
2205.02824)도 history→encoder를 **privileged 환경 정보의 system-ID 대체재**로만 쓰지, gait phase를
실어 나르는 용도로 쓴 적이 없다.

**결론**: 5% rollout mismatch는 domain-adaptation 품질(마찰·질량 추정 정확도)에는 실제 버그이자
고칠 가치가 있지만, **그것을 고쳐도 phase-aware 액터가 되지는 않는다** — distillation target 자체가
애초에 phase를 담도록 설계되지 않았기 때문이다. RMA 경로는 이 특정 버그(이봉)에 대해서는 mismatch
여부와 무관하게 구조적으로 막다른 길이다.

### 7.2 PPO 로코모션에서 gait-initiation/phase-observability의 표준 처방 (message 1 Q1)

세 갈래가 있고 서로 대체 불가능하다:

- **명시적 커맨드 clock (Walk These Ways)**: WTW의 gait timing 변수(주파수·다리 간 위상차·duty cycle)는
  **사용자/스크립트가 지정하는 입력**이지 정책이 history에서 추론하는 것이 아니다 — 커맨드 벡터의
  일부로서 구성상 항상 관측 가능하다. 이는 observability 문제를 **풀지 않고 우회**한다: WTW는 정책이
  모호한 정지 상태에서 자율적으로 보행 개시 시점을 결정하도록 요구한 적이 없다, 위상이 항상 외부에서
  주어지기 때문이다. **우리 설정에 직접 이식 불가**(유사한 외부구동 phase/duty 커맨드를 추가할 의향이
  없다면).
- **CPG/PAE 주기 latent**: CPG-RL은 oscillator(시간에 걸쳐 위상을 실어 나르는, 사실상 손설계된
  recurrence)를 행동 공간 자체에 내장한다. DeepPhase/PAE(`offline`, SIGGRAPH 2022)는 mocap 데이터에서
  비지도·비-RL 사전학습으로 주기 latent를 학습; 다운스트림 on-policy RL 적용(FLD)은 정량적으로 확인
  못함 — **folklore에 가까움, 로코모션 PPO에 대해 미확인으로 플래그**.
- **on-policy BPTT로 학습된 recurrent 정책**(Cassie/Siekmann RSS 2020, ZSL-RPPO arXiv 2403.01928, 특히
  **Gait-Conditioned RL with Multi-Phase Curriculum**, arXiv 2505.20619): 가장 직접적으로 들어맞는
  계열. 이번 정독으로 확인: **액터·크리틱 모두 명시적으로 LSTM+MLP**(막연한 "recurrent"가 아니라
  구체적으로 확인됨), one-hot gait-ID reward-routing + 단계적 커리큘럼과 결합. **Unitree G1 실기에서
  walk-to-stand 전환을 보고** — 이번 서베이 전체에서 우리 실패 모드와 가장 가까운 발표 사례.

**실증 vs folklore**: recurrence(세 번째 갈래)만이 stand/walk 전환류 문제에 대해 정량적·실기·
on-policy-locomotion 결과를 가진 유일한 갈래다. CPG 명시적 위상은 아키텍처적으로 타당하고 저렴하지만
gait *개시* 자체에 대한 효과를 분리한 논문은 못 찾았다(주로 정상상태 보행 품질/에너지용으로 쓰임).
WTW식 커맨드 clock은 다른 문제(이미 움직이는 보행의 스타일링이지 자율 개시가 아님)를 위한 다른 도구다.
PAE/FLD의 RL 적용은 원문을 직접 정독하기 전까지 folklore로 취급할 것.

### 7.3 history-in-observation vs recurrent architecture — 혼동되고 있는가? (message 1 Q2)

부분적으로만 답 가능하며, 출처 강도에 중요한 유보가 있다:
- 명시적 ablation을 하나 찾았으나 처음 생각보다 약하다: **HACL(arXiv 2505.18429)**은 "RNN/LSTM/GRU
  전반에서 성능 차이는 미미하며, recurrent cell 종류가 아니라 memory 자체가 개선을 이끈다"와 "관측
  history를 길게 줄수록 MLP 정책의 성능 향상은 saturate한다"고 보고한다. **그러나 원문 전체를 확인한
  결과, 이 ablation은 HACL의 *커리큘럼 샘플링 네트워크*에 관한 것이지 로코모션 정책 자체가 아니다.**
  그들의 로코모션 정책은 plain PPO로 학습되고, "history-aware vs non-history"는 (다음에 학습할 속도
  커맨드를 고르는) **커리큘럼 스케줄러** 간의 비교이지, history-stack MLP 액터 대 recurrent 액터의
  비교가 아니다. **이것을 history-stack-vs-GRU 액터 ablation으로 인용하지 말 것** — 처음에 범위를
  잘못 짚었고 바로잡는다.
- on-policy 로코모션에서 "N-step 관측 history-stack MLP 액터" vs "recurrent 액터, 동일 task"의
  통제된 1차 ablation은 찾지 못했다. 가장 가까운 간접 증거: DreamWaQ와 Extreme Parkour 둘 다
  history를 먹는 *인코더*(RNN이 아니라 stacked window 위의 feedforward MLP)를 **domain/terrain
  추정**에 성공적으로 쓴다 — 즉 stacked-history MLP가 그 작업에 유용한 시간 신호를 뽑아낼 수 있다는
  것은 확실하다. 이것이 **phase-onset**에 대해서도 GRU와 같은 역할을 하는지는 찾은 범위 내에서는
  미검증 영역이다.
- 실용적 판단: 고정길이 history stack은 더 저렴한 arm이다(rsl_rl 아키텍처 변경 없음, exporter 위험
  없음, 50Hz에서 LSTM hidden-state 관리 없음) — Arm 1을 대체하는 것으로서가 아니라 **함께 고려할
  가치**가 있다. 다만 "둘이 동등하다"거나 "GRU가 더 낫다"를 로코모션 특정 통제 인용으로 뒷받침할
  수는 없다. 문헌이 이미 결론낸 것이 아니라, 3-arm 예산이 실제로 답할 수 있는 열린 질문으로 취급할 것.

### 7.4 저속 floor는 아키텍처 문제인가 — 가장 중요한 질문 (message 1 Q3)

**내 판단: 주로 아니다 — 사용자에게 그렇게 말해야 한다.** "저속을 달성 못 한다"에 대해 RMA/recurrence
질문과 완전히 독립적인, 문헌에 잘 기록된 비-아키텍처 설명이 최소 넷 있다:

1. **정지 유지는 흔한 reward 국소최적점이다.** [`on-policy-locomotion`, 4족] "Controlling the Solo12
   Quadruped Robot with Deep Reinforcement Learning" (arXiv 2309.16683,
   https://arxiv.org/abs/2309.16683 / Nature Sci. Rep. https://www.nature.com/articles/s41598-023-38259-7)
   에서 직접 확인: "에이전트는 명령 속도를 따르라는 긍정 보상 신호를 무시하고 정지를 학습할 수
   있다, 이것이 여러 페널티 항을 최적화하기 때문이다"(에너지/행동-부드러움/관절속도 페널티는 전부
   무동작에서 자명하게 최소화됨). 표준 처방은 **reward 커리큘럼**(페널티 가중치를 0에서 시작해
   서서히 올림, 그리고/또는 초반 커맨드 범위를 0 근처로 제한하고 고정된 작은 정지-확률을 둠)이지
   **네트워크 변경이 아니다.** 저속 floor에 대한, 아키텍처와 직교하는, 선례가 풍부한 대안 설명이다.
2. **느린 보행 자체가 독립적으로 더 어렵다고 기록됨**, 정지-국소최적점 이슈와는 별개로: [`on-policy-locomotion`,
   4족] "Synthesizing the optimal gait of a quadruped robot with soft actuators using deep
   reinforcement learning" (ScienceDirect 2022,
   https://www.sciencedirect.com/science/article/pii/S0736584522000692)는 낮은 목표 속도에서
   에이전트가 "앞으로 고꾸라지는 경향"과 "국소최소에 갇힘"을 보고하며, 속도에 대한 커리큘럼 학습으로
   해결한다 — 역시 학습 스케줄 처방이지 아키텍처가 아니다.
3. **limit-cycle 보행에는 진짜 물리적 최저속도 floor가 있다.** [`offline`, 고전 역학 해석 — RL이 아닌
   해석적 limit-cycle walker 연구] "Controlling the Walking Speed in Limit Cycle Walking" (IJRR,
   https://dl.acm.org/doi/abs/10.1177/0278364908095005)에 따르면 안정 정상상태 속도는 **stability
   bifurcation**으로 하한이(임계 주파수/속도 미만에서는 limit cycle 자체가 불안정), actuation 한계로
   상한이 결정된다 — 해당 연구된 이족 사례(다리 길이 0.6m)에서 안정 속도는 대략 0.24~0.68 m/s
   범위에서만 얻어졌다. 이것은 학습이 아니라 **동역학** 문제다. 우리 로봇 morphology에 그대로
   적용된다는 근거는 없으므로 이 항목은 **`extrapolated`**로 라벨한다 — **만약 우리 morphology의
   안정 limit-cycle 하한이 0.5 m/s 근처라면, 어떤 정책 아키텍처도 이를 고치지 못한다**; 시스템은
   질적으로 다른 저속 전략(shuffling gait, 연장된 double support, stop-and-go)이 필요하며 이는
   reward/gait 설계 결정이지 network 용량/memory 결정이 아니다.
4. **AMP reference data 커버리지** — [`extrapolated`] 반대쪽 끝(고속)에서는 이 메커니즘이 실재한다는
   직접 증거가 있다(§4/§6.4 AMP 절 참조: "AMP는 sprint reference data 부족으로 고속 구간에서 학습
   불안정을 겪는다", reference data를 더하면 해결). 저속 끝에 대해 동일하게 명시한 논문은 **못 찾았다**
   — 이것은 **대칭성에 의한 외삽이지 직접 인용이 아니다** — 하지만 AMP가 이미 우리 루프에 있으므로,
   reference 클립 구성에서 저속/정지 근접 커버리지를 확인하는 것은 저렴하고 직접 검증 가능하며,
   125~142% 초과달성을 아키텍처 탓으로 돌리기 전에 배제해야 한다.

**솔직한 결론, 요청하신 대로 명료하게**: 팀의 예비 관측(10-step history run이 duty=1.00을 찍지만
125~142%로 초과달성)은 위 (1)~(4) 전부와도, "아키텍처가 답이다"와도 똑같이 정합적이다.
recurrence/phase 조건화(Arm 1/2)는 **이봉(개시 여부)** 질문에는 잘 겨냥되어 있지만, 같은 아키텍처
변경을 **일단 움직이면 초과달성하는** 질문까지 설명/해결한다고 확장하지는 않겠다 — 이것은 다른
실패이고 아마 다른(비-아키텍처 모양의) 처방(페널티 항 reward 커리큘럼, 커맨드 범위 커리큘럼,
reference-data 감사)이 필요해 보인다. **만약 3-arm 아키텍처 스윕을 "이봉도 고치고 저속 초과달성도
고칠 것"이라는 근거로 정당화하고 있다면, 그 프레이밍에는 반대한다** — 초과달성 문제는 아키텍처
arm과 묶지 말고 **별도의, 아마 reward/data 쪽 조사 트랙**으로 병행 진행할 것을 권한다. 그래야 한쪽의
null 결과가 다른 쪽 탓으로 잘못 귀속되지 않는다.

### 7.5 RMA 병목의 대안 — on-policy 액터에 시간 정보를 유지하는 법 (message 2 Q2)

(a) **PPO와 동시에(concurrent) 학습되는 estimation** (별도 DAGGER distillation이 아님): **Ji et al.,
"Concurrent Training of a Control Policy and a State Estimator for Dynamic and Robust Legged
Locomotion"**, IEEE RA-L 2022, arXiv 2202.05481. 실기·on-policy-locomotion 증거(평지 3.75 m/s,
마찰계수 0.22 미끄러운 판 위 3.54 m/s). estimator와 정책이 **동시에** 학습됨 — estimator는 privileged
target에 대한 지도학습 loss로, 정책은 PPO로, 매 iteration 함께 업데이트(사후 distillation 단계가
아님). 결정적으로 이들이 추정하는 대상은 **base 선속도, foot height, contact probability** —
이들은 준정적 domain 파라미터가 아니라 **동적인, 스텝별, gait-relevant** 양이다. 우리 버그에 대해
RMA-병목 대안 셋 중 아키텍처적으로 가장 유망하다 — contact probability/foot-height는 RMA의 `e_t`와
달리 phase 관련 정보를 실제로 실어 나를 수 있기 때문이다. **4번째 후보 또는 Arm 3의 수정판으로
고려할 가치가 있다**(기존 `history_encoder`를, 5%만 rollout되는 DAGGER target 대신, 매 iteration
학습되는 contact/foot-height estimator로 바꿔 on-policy 액터에 먹인다).
(b) **raw history-stack을 액터 입력에 concatenate**: 흔함(DreamWaQ, Extreme Parkour student)이나,
§7.3에서처럼 이것 단독(recurrence 없이)이 phase-onset을 해소하는지 분리한 통제 실험은 못 찾았다 —
domain/terrain 추정에는 입증됐고, gait-onset에는(찾은 범위 내에서는) 미검증.
(c) **PPO+BPTT로 직접 학습되는 recurrent 정책**: 이것이 이미 제안한 Arm 1이다(Cassie, ZSL-RPPO,
Gait-Conditioned RL 2505.20619) — 최선의 근거를 가진 옵션으로 재확인되었고, 2505.20619의 확인된
LSTM+MLP 아키텍처와 실기 walk-to-stand 결과로 추가 뒷받침됨.

**이 새 정보를 반영한 순위 갱신**: **(a) concurrent contact/phase estimation**을 Arm 1(LSTM
recurrence)과 거의 동급으로 올린다 — 둘 다 동적/시간적 상태를 직접 겨냥하고, (a)는 추론하기 더
쉽고(지도학습 보조 loss, BPTT-through-time 복잡도 없음, estimator 출력을 stateless 액터에
concatenate만 한다면 exporter의 hidden-state 관리도 불필요) 이 서베이에서 실기 속도 기록도 가장
높다. Arm 3(discriminator 조건화)를 (a)로 교체하고, AMP-discriminator 조건화 아이디어는 Arm 1/(a)
중 하나에 얹는 저렴한 부가 로깅/변경으로 격하할 것을 제안한다 — §4의 AMP mode-collapse 가설은 학습
run 전체를 커밋하기 전에 reference 클립 구성 감사만으로 저렴하게 반증/확인 가능하기 때문이다.

### 7.6 message 2 Q1, Q3 요약 답변

RMA 계열(Kumar RMA, Extreme Parkour, Rapid Locomotion via RL)과 direct-recurrence 계열(Cassie,
ZSL-RPPO, Gait-Conditioned RL)은 "temporal info가 도움된다"는 주장을 **서로 겹치지 않는 두 근거로**
정당화한다: RMA 계열은 오직 domain/terrain adaptation을 위해서만(gait phase를 위해서는 결코 아님),
direct-recurrence 계열은 phase/gait-transition 행동을 위해서. 우리 `ActorCriticRMA`는 구조적으로
첫 번째 계열이다. phase-onset 이득을 얻으려면 두 번째 계열로 옮기거나(Arm 1) §7.5의 방식으로 병목을
우회해야 한다(Arm 2 수정판) — 이는 RMA 원 설계자들이 애초에 의도하거나 측정한 적 없는 것이며,
mismatch 버그 여부와 무관하다.

### 7.7 최종 랭크드 숏리스트 (최대 3개, §5를 대체) — 50k-iteration arm용

**주의**: 아래 3개 전부 실패 모드 (A) "이봉/개시 여부"만 겨냥한다. §7.4에서 결론냈듯 "일단 움직이면
저속을 못 냄(초과달성)" 문제는 이 3개 arm으로 검증되지 않으며 §7.8의 별도 병행 트랙이 필요하다.

**#1 — LSTM recurrent 액터**
- 무엇을 바꾸나: rsl_rl `ActorCriticRecurrent`, `rnn_type="lstm"`(GRU 아님), 단일 LSTM layer(hidden
  ≈128), MLP 본체 차원 무변경. PPO+BPTT로 직접 학습(RMA/DAGGER 경로 폐기).
- 겨냥하는 실패 모드: (A) 이봉 — 관측에 "어떻게 도달했는가"(직전 정지 상태였는지, 이미 보행 중이었는지)를
  직접 제공.
- 증거 등급: `on-policy-locomotion`, 실기. Cassie(Siekmann et al., RSS 2020,
  https://www.roboticsproceedings.org/rss16/p031.pdf), ZSL-RPPO(arXiv 2403.01928,
  https://arxiv.org/abs/2403.01928), Gait-Conditioned RL(arXiv 2505.20619,
  https://arxiv.org/abs/2505.20619, 실기 Unitree G1 walk-to-stand 전환 보고 — 단, 정량 성공률과
  PPO 여부 원문 미정독, §7.2 유보 참조).
- export/추론 비용: 낮음~중간. rsl_rl 네이티브 지원. **GRU 아닌 LSTM 필수** — IsaacLab exporter가
  LSTM (hidden,cell) 튜플을 하드코딩해 GRU export가 깨짐(isaac-sim/IsaacLab issue #3008, 미해결,
  https://github.com/isaac-sim/IsaacLab/issues/3008, GitHub에서 직접 확인). 배포 런타임이 50Hz
  루프에서 hidden+cell state를 이월해야 하는 부담 추가.
- 기대 효과에 대한 솔직한 평가: 서베이 전체에서 메커니즘·실기 선례가 가장 강하지만, 2505.20619의
  정확한 정량 결과를 확인 못했으므로 "이 정도 효과를 기대하라"고 숫자로 약속할 근거는 없다. **가능성
  높으나 확정 아님.**

**#2 — 동시학습(concurrent) contact/gait-relevant estimator**
- 무엇을 바꾸나: 기존 `history_encoder`를 DAGGER-distill된 준정적 `priv_latent`(38차원, episode-constant)
  타깃 대신 foot contact probability / foot height / base linear velocity를 예측하도록 재정의하고,
  이 지도학습 loss를 **매 PPO iteration마다**(현재의 5% rollout이 아니라) 정책과 동시에 업데이트,
  출력을 stateless 액터에 concatenate.
- 겨냥하는 실패 모드: (A) 이봉 — contact probability/foot height는 RMA의 `e_t`(질량·마찰 등 준정적
  파라미터)와 달리 **동적·스텝별·gait-relevant** 양이라 phase 관련 정보를 실제로 실어 나를 수 있음.
- 증거 등급: `on-policy-locomotion`, 실기. Ji et al., "Concurrent Training of a Control Policy and a
  State Estimator for Dynamic and Robust Legged Locomotion", IEEE RA-L 2022, arXiv 2202.05481
  (https://arxiv.org/abs/2202.05481) — 평지 3.75 m/s, 마찰계수 0.22 미끄러운 판 위 3.54 m/s 실기 결과.
- export/추론 비용: 낮음. BPTT 없음, hidden-state export 관리 불필요(estimator 출력을 단순
  concatenate하면 stateless 액터로 export 가능) — 이 서베이 전체에서 가장 export-friendly한 옵션.
- 기대 효과에 대한 솔직한 평가: RMA 병목을 구조적으로 우회한다는 점에서 논리적으로 #1과 동급이나,
  "estimator가 실제로 phase를 학습하는지"는 우리 task에서 검증된 바 없다(Ji et al.은 gait-onset을
  보고하지 않음, 고속 로버스트 주행 맥락). **원 논문이 우리와 같은 실패 모드를 겨냥/보고한 적 없다는
  점에서 #1보다 한 단계 더 추론(extrapolation)이 들어간다.**
- 증거 등급 참고: 이 후보의 "phase-onset을 고칠 것"이라는 기대 자체는 `extrapolated` — 논문은
  contact/foot-height 추정이 강건성/속도에 기여함을 보였지 stand-to-walk 전환을 보이지 않았다.

**#3 — 명시적 phase/duty 관측 feature**
- 무엇을 바꾸나: 관측에 명시적 스칼라/one-hot 입력(지수평활된 최근 duty-cycle 추정치, 또는 "최근
  N ms 이내 stepping했는가" 지시자) 추가. 네트워크 아키텍처는 무변경(baseline과 동일 MLP).
- 겨냥하는 실패 모드: (A) 이봉 — recurrence 없이 partial observability를 명시적 feature로 해소.
- 증거 등급: `on-policy-locomotion`, 실기(간접) — Gait-Conditioned RL(arXiv 2505.20619)의 one-hot
  gait-ID reward-routing이 유사 메커니즘. 순수 관측-엔지니어링 버전 자체를 단독으로 검증한 논문은
  못 찾음 — **`extrapolated`**(가장 가까운 선례에서 추론).
- export/추론 비용: 사실상 0. 입력 채널 1개 추가, 네트워크 변경 없음, export 위험 없음 — 이 서베이
  전체에서 가장 저렴한 후보.
- 기대 효과에 대한 솔직한 평가: 손설계 신호가 실제 의존 window를 못 담으면(짧은 시정수) 부분적으로만
  효과가 있을 수 있음. **가장 저렴한 대조군으로서 가치가 있지만, 셋 중 성공 확률에 대한 근거가 가장
  얇다.**

### 7.8 탈락 리스트 (한 줄 사유)

- **Diffusion/flow-matching PPO (DPPO/D2PPO/FPO/PolicyFlow)** — 로코모션 on-policy 증거 없음(`on-policy-other`),
  50Hz 임베디드에 검증 안 된 높은 추론비용(K-step sampling), 이봉의 메커니즘(history 결핍)을 겨냥
  못함.
- **Mixture-of-Gaussians 정책 head** — 같은 관측에 조건화되므로 컴포넌트를 고를 신호가 없어 mode-averaging
  예상, on-policy 학습에서 수치 불안정 지적됨, 로코모션 증거 없음.
- **잠재조건화 다중모드 정책 (CALM/ASE를 액터 latent selector로, VAE mode latent)** — latent를 고를
  외부 장치가 필요해 자율 의사결정 문제를 해소가 아니라 이전만 시킴.
- **DeepPhase/PAE 학습된 주기 latent** — 별도 pretraining 단계 필요, 다운스트림 on-policy RL 증거
  (FLD) 미확인, 손설계 phase 신호(#3)보다 무거움.
- **WTW식 명시적 커맨드 clock** — observability 문제를 풀지 않고 외부 입력으로 우회하는 다른 도구,
  자율 개시 결정이라는 우리 문제 자체를 요구하지 않음.
- **raw history-stack MLP 액터** (recurrence 없이 관측에 N-step 이력만 concat) — domain/terrain
  추정에는 입증됐으나 phase-onset에 대한 통제 ablation 없음(§7.3); #1·#2보다 근거가 약해 3위 밖으로
  밀림.
- **AMP discriminator 조건화 (CALM식)만 단독 arm** — §4의 가설 자체는 유효하지만 학습 run 없이
  reference 클립 구성 감사만으로 저렴하게 먼저 검증 가능하므로, 전체 arm을 쓸 필요 없음 — arm이
  아니라 #1/#2에 얹는 부가 로깅으로 격하.
- **원 서베이의 순수 스케일링 축 전체** (BRO, SimBa, SimbaV2, HL-Gauss, spectral norm, MoE, transformer
  액터, network sparsity, periodic reset/ReDo) — 전부 이봉/hysteresis를 전혀 겨냥하지 않음(용량/
  plasticity를 겨냥), 다수가 50Hz 임베디드 제약과 충돌(transformer, MoE), 실기 on-policy-locomotion
  증거 없음. 상세는 §6.

### 7.9 4번째 arm이 아니라 병행 트랙 — "저속 초과달성" 조사

별개의 "일단 움직이면 저속을 못 냄(125~142% 초과달성)" 문제에 대한 reward/커리큘럼/reference-data
조사(§7.4 참조): reference 클립의 정지-근접 커버리지 확인, 페널티 항 커리큘럼 ramping 존재 여부 확인,
0.5 m/s가 우리 morphology의 예상 안정 limit-cycle 범위에서 어디에 위치하는지 확인. 학습 run이 필요
없어 저렴하며, 아키텍처 스윕과 함께 또는 그 전에 해소/특성화되어야 한다 — 그래야 이 하위 문제의 진짜
병목이 reward/data였는데 아키텍처의 부정적/모호한 결과를 "아키텍처는 안 통한다"로 잘못 읽는 일을
막을 수 있다.

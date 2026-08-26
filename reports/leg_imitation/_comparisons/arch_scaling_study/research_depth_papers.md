# 1000-layer 논문 + BC mystery 정독 — on-policy PPO 전이 가능성 비판

**2026-08-26** · 작성: research-depth-papers agent
**전제**: `README.md`의 실패 모드 (A) — cmd 0.5 상승(1~64%) vs 하강(61~77%) 이봉 — 및
"이게 capacity 문제냐 expressiveness 문제냐"는 team-lead 질문을 겨냥한 두 소스 정독.
`research_onpolicy_scaling.md`(독립적으로 도출)와 결론이 **일치**한다: 이 문서도 "Markovian 정책은
표현력을 아무리 올려도 같은 관측에서 갈리는 두 정답을 못 구분한다"는 진단에 도달했다.

**표기 규칙**: 본문에서 "[출처]"로 표시한 문장은 원문 인용 또는 원문 직접 요약이다. "[추론]"으로
표시한 문장은 이 agent가 두 출처 또는 리포지토리 컨텍스트를 조합해 도출한 것으로, 원문에 없다.
수식은 LaTeX 없이 ASCII 코드블록으로 쓰고, 기호표를 먼저 둔다.

---

## 0. 기호표

| 기호 | 의미 | 단위/범위 |
|---|---|---|
| `r_t(theta)` | PPO 확률비 = pi_theta(a\|s) / pi_theta_old(a\|s) | 무차원, 이상적으로 1 근방 |
| `A_t` | advantage 추정치 | 무차원 |
| `eps` | PPO 클리핑 폭 | 통상 0.1~0.2 |
| `L` | InfoNCE critic이 판별하는 embedding 간 L2 거리 | 무차원 |
| `N` | CRL 논문의 residual block 개수 | 정수, depth = 4N |
| `UTD` | update-to-data ratio (환경 스텝 1개당 gradient step 수) | 무차원 비율 |
| `Lip(f)` | 함수 f의 Lipschitz 상수 | 무차원, 작을수록 매끄러움 |

---

## 1. "1000 Layer Networks for Self-Supervised RL" (Wang et al., arXiv:2503.14858, NeurIPS 2025)

링크: [arXiv abstract](https://arxiv.org/abs/2503.14858) · [HTML](https://arxiv.org/html/2503.14858v1) · [project page](https://wang-kevin3290.github.io/scaling-crl/)

**신뢰도 주의**: 아래 아키텍처 세부는 arXiv HTML 본문을 fetch-요약 도구로 읽은 것이며, 원문 PDF를 직접
정독한 것이 아니다. 특히 init 방식·UTD 비율·LR 값은 단일 소스만 확보해 낮은 확신으로 표시한다.

### 1-1. 아키텍처 레시피 [출처, 일부 저확신]

- **깊이**: 4, 8, 16, 32, 64, 256, 최대 1024층까지 테스트. 항상 4의 배수(residual block 하나 = Dense 4개).
- **Residual block 구조**: `h_{i+1} = h_i + F_i(h_i)`, `F_i` = (Dense -> LayerNorm -> Swish) x 4.
  post-activation/post-norm 스타일 — LayerNorm이 block 내부에서 각 Dense 뒤에 온다.
- **정규화/활성화**: LayerNorm + Swish(ReLU 아님)가 "non-negotiable" — 빼면 스케일링 이득이 붕괴한다고
  원문이 명시.
- **폭**: baseline 256, ablation에서 최대 4096까지. **깊이 스케일링이 폭 스케일링보다 효과적**이라는
  직접 ablation이 있다고 확인됨(정확한 수치는 확보 못함).
- **init**: 원문에서 확인 못함 [저확신/미확인].
- **optimizer/LR**: 3e-4 (policy·critic 공통), 스케줄 불명 [저확신, 단일 소스].
- **batch size**: 주 실험 512, 스윕 128/256/512/1024/2048. 핵심 발견: **깊은 net만 큰 배치의 이득을
  받는다** — 얕은 net은 배치를 키워도 정체.
- **UTD**: ~1:40으로 보고됨 [저확신, 교차검증 실패].
- **정규화항**: InfoNCE critic에 logsumexp penalty 계수 0.1. dropout/weight decay 언급 없음.

### 1-2. 학습 설정 — PPO 전이 판단에 가장 중요한 사실

- **알고리즘**: Contrastive RL(CRL). **오프폴리시, 온라인** self-supervised 알고리즘. reward 없이
  InfoNCE 분류기가 "미래에 방문한 goal vs 무작위 goal"을 구분하는 것으로 value 역할을 대체.
- **데이터 체제**: 온라인 롤아웃이 **replay buffer**(size ~10,000, 최소 1,000)를 채우고 거기서
  오프폴리시로 샘플링 — on-policy 단일 패스가 아니라 SAC/TD3와 데이터 체제상 훨씬 가깝다.
- **[출처] SAC·TD3(TD 계열)에도 같은 레시피를 적용했지만 4층 이상에서 이득이 없거나 오히려 악화**됨.
  즉 이 논문 안에서도 "깊이 스케일링 = 오프폴리시/온라인 RL 전반의 성질"이 아니라 **InfoNCE
  분류형 목적함수에 국한된 현상**이라는 것이 자체 실험으로 확인된다.
- **[출처] 오프라인 설정에서는 명시적으로 부정적**: "we found no evidence that increasing network
  depth improves performance in this offline setting" / 이 이득은 "online self-supervised setting"에
  국한되며, 오프라인 RL 벤치마크에 적용 시 "mixed or negative results"였고 "the active interplay
  between exploration and deep representation learning is critical"라고 저자들이 직접 설명한다.

### 1-3. 정량적 결과 (depth 4 -> 64) [출처]

| Env | depth 4 | depth 64 | 배율 |
|---|---|---|---|
| Humanoid | 12.6±1.3 | 649±19 | 52x |
| Humanoid U-Maze | 3.2±1.2 | 159±33 | 50x |
| Ant U5-Maze | 0.97±0.7 | 61±18 | 63x |
| Ant U4-Maze | 11.4±4.1 | 286±36 | 25x |
| Ant Big Maze | 61±20 | 441±25 | 7.3x |
| Arm Binpick Hard | 38±4 | 219±15 | 5.7x |

특정 깊이(Ant Big Maze는 8층, Humanoid U-Maze는 64층)에서 "위상 전이"처럼 성능이 도약하며, 이는
정성적으로 다른 정책(더 긴 경로를 짧은 학습 경험 조각들을 이어붙이는 "stitching" 능력)의 출현과
연결된다고 저자들이 설명한다.

### 1-4. 무엇이 개선되지 않았는가 [출처]

- 샘플 효율/wall-clock — 이득은 고정 데이터에서의 점근 성능이지 더 빠른 학습이 아니다.
- SAC/TD3(TD 계열) — 깊이에 무관하게 평평하거나 악화.
- 오프라인 RL 성능 — 이득 없음, 때로는 악화.

### 1-5. 저자들의 메커니즘 설명 — 이 논문에서 가장 약한 부분 [출처 + 평가]

- 분류형(InfoNCE) 그래디언트가 스칼라 TD 회귀보다 "더 강건하고 풍부하다"는 것이 CRL은 스케일하고
  TD는 안 되는 이유의 "가설"이라고 명시 — **검증된 것이 아니라 가설로 제시됨**.
- 표현 품질 논증: 깊은 net이 학습한 value function은 유클리드 거리가 아니라 미로의 위상 구조를
  반영하는 "geodesic"한 형태(Q-value 히트맵으로 시각화)라고 주장.
- residual connection의 "gradient propagation 개선"은 표준 ResNet 논리를 그대로 인용한 것으로,
  RL에 특화된 설명이 아니다.
- **정식 이론 없음** — NTK/feature-learning regime 분석, fixed-point/deep-equilibrium 논증,
  깊이가 쌓일수록 TD 부트스트래핑 불안정이 왜 증폭되는지에 대한 설명이 없다. 이 논문은
  경험적/아키텍처 논문이지 이론 논문이 아니다.

---

## 2. Seohong Park, "The Behavioral Cloning Mystery" (블로그)

링크: https://seohong.me/blog/behavioral-cloning-mystery/

여러 개의 느슨하게 연결된 "미스터리" 모음이며, 모델 크기 미스터리는 그 중 하나다(다른 것: 데이터셋
크기 비단조성, feature scaling 민감도).

### 2-1. 아키텍처 레시피 — 매우 부실하게 명시됨 [출처]

- 태스크: 37차원 state 기반 pick-and-place, **flow matching**으로 25-step action chunk를 예측하는
  정책. 7-DoF joint-velocity 제어, 50 Hz.
- 필요했던 모델: **"[4096]x8" residual MLP**(8층, 폭 4096), 8192폭으로 더 개선. 기존 벤치마크는
  최대 [1024]x4를 썼다고 언급.
- **정규화/활성화/optimizer/LR/init/weight decay/dropout/gradient step 수 — 전부 원문에 없음.**
  세 차례 별도 fetch로 재확인했고, 이 블로그 포스트는 논문의 methods 섹션에 해당하는 내용을
  애초에 담고 있지 않다.
- **depth-vs-width ablation 없음.** "[4096]x8"은 깊이와 폭을 동시에 바꾼 것이라 어느 쪽이
  기여했는지 구분할 근거가 원문에 없다 — CRL 논문이 깊이/폭/배치를 명시적으로 분리한 것과 대비된다.
- 학습 데이터: GPU MuJoCo(MJWarp)로 스크립트 전문가(piecewise Hermite spline)가 계속 새 demo를
  생성하는 "infinite-data" 체제 — 고정 오프라인 데이터셋도 아니고 온라인 RL도 아니다.

### 2-2. 스케일 관련 정확한 주장 [출처]

- [512,512,512] MLP: 불충분.
- [4096]x8 residual MLP: 이 태스크를 제대로 풀기 위한 필요조건.
- [8192] 폭: 추가 개선.
- (별개 미스터리, 같은 글) **10K demo 데이터셋이 50K demo보다 같은 태스크에서 더 좋다** — 저자는
  이를 network 자체가 아니라 test-time distribution shift 탓으로 돌린다.

### 2-3. 저자 본인의 설명 — 눈에 띄게 솔직한 불확실성 [출처]

직접 인용: **"I don't have a great answer for this mystery... I still find it a bit hard to believe
that we need such a large model for this simple, fixed, state-based task."**
대안으로 "maybe behavioral cloning is just really hard"(capacity 프레이밍) 또는 "the current way of
doing flow behavioral cloning... is just inefficient"(optimization/구현 프레이밍)만 제시하고 어느
쪽도 확정하지 않는다.

### 2-4. Expressiveness/다중모드 관련 — team-lead의 후속 질문에 대한 답 [출처]

네 가지를 명시적으로 다시 조회했다: (1) Gaussian vs flow-matching 비교, (2) 데이터 다중모드성,
(3) expressiveness 프레이밍, (4) capacity/expressiveness/optimization 구분. **전부 "언급 없음"**.

- "For behavioral cloning, I used standard flow matching..."라고만 쓰고 Gaussian 대비 flow matching을
  선택한 이유를 정당화하지 않는다.
- 데이터를 "narrowly distributed, highly temporally correlated"라고 묘사 — 오히려 다중모드성과는
  반대 방향을 시사하는 표현이다. 스크립트 전문가가 고정된 태스크에 대해 생성하는 데모는 시작
  조건이 같으면 대체로 결정적(deterministic)이지, 서로 다른 여러 해법의 앙상블이 아니다.
- **"expressiveness/multimodality"는 저자가 나열한 후보 설명 목록에 아예 들어있지 않다.**
  capacity 프레이밍("hard"→더 큰 모델 필요)과 optimization 프레이밍("비효율적 구현") 두 가지만
  제시되고, 다중모드성 가설은 후보로도 등장하지 않는다.

**결론: 이 포스트 자체는 다중모드성/expressiveness 주장의 근거가 아니다.** 저자 본인이 이 태스크를
다중모드로 특징짓지 않는, 미해결 capacity-or-optimization 미스터리다.

---

## 3. Expressiveness/다중모드 논증이 실제로 있는 곳 — Seohong Park의 다른 저작

`seohong.me/blog`에는 이 포스트를 포함해 총 4개 글만 있다: behavioral-cloning-mystery,
rl-without-td-learning, dual-representations, q-learning-is-not-yet-scalable. **제목에 policy
expressiveness/Gaussian-vs-flow/다중모드를 다루는 글은 하나도 없다** — 그 스레드는 블로그가 아니라
그의 논문에 있다.

### 3-1. Flow Q-Learning (FQL) 프로젝트 페이지 [출처]

https://seohong.me/projects/fql/ (Park, Li, Levine)

> "FQL's expressive flow policy can handle data with arbitrarily complex, multimodal action
> distributions."
> "FQL achieves substantially better performance than Gaussian policy-based methods on manipulation
> tasks with highly multimodal action distributions."

이것이 그의 저작 중 실제로 expressiveness 주장을 명시하는 곳이다. 다만 확보하지 못한 것: 다중모드
행동 분포를 보여주는 그림, **더 깊은 Gaussian이 격차를 메우는지에 대한 depth/width ablation**,
파라미터 수를 맞춘 비교. 즉 이 주장은 정성적 근거만 확인했고, "expressiveness가 capacity와 독립적"임을
보이는 통제된 ablation은 확보하지 못했다.

### 3-2. "Q-learning is not yet scalable" [출처, 부수적이지만 유용한 교차검증]

https://seohong.me/blog/q-learning-is-not-yet-scalable/

깊이 스케일링 얘기가 아니라 **horizon 길이에 걸친 TD bias 누적**이 TD 학습의 스케일 한계라는
주장이다. 두 가지가 1번 섹션과 독립적으로 교차검증된다:

- "MC-based methods like contrastive RL[...] might eventually scale better than TD-based approaches"
  — 1000-layer 논문의 SAC/TD3-vs-CRL 결과와 같은 TD-vs-분류형-목적함수 구도를, 같은 저자가
  독립적으로 제기.
- **PPO의 온폴리시 제약을 스케일링 한계로 직접 논의**: PPO는 "requires fresh data and cannot
  efficiently reuse past experience like off-policy methods should" — 아래 4-(a)의 데이터 재사용
  비율 논증을 저자 본인이 별도로 확인해준 셈이다.

### 3-3. 가장 정밀한 근거 — 제3의 논문 (Seohong Park 저작 아님) [출처]

Mazza, Datres, Rodriguez, Bodenstedt, Kutyniok, Speidel, "Understanding Multimodal Failure in
Action-Chunking Behavioral Cloning" (arXiv:2605.22493, 2026). 초록 전문:

> "Behavioral cloning becomes difficult when the same observation admits several valid actions. We
> study this problem for action-chunking policies and show that different multimodal
> parameterizations fail in different ways. For latent-variable policies, posterior-prior
> regularization makes deployment-time sampling more reliable, but excessive regularization removes
> the action-conditioned information needed to distinguish demonstrated modes... For action-space
> generative policies, multimodality is constrained by the smoothness of the base-to-action
> transport: a map with small `Lip(f)` cannot assign substantial probability to many well-separated
> modes. Covering many modes therefore requires either sharp transitions in base space or
> off-support bridge regions in action space."

**이 논문이 team-lead 질문에 가장 직접적으로 답한다**: 다중모드 실패는 파라미터 수가 아니라 output
분포 계열/transport map의 매끄러움(`Lip(f)`)의 성질이라고 명시한다. 작은 `Lip(f)`를 갖는 map은
파라미터나 레이어를 아무리 늘려도 서로 멀리 떨어진 여러 모드에 유의미한 확률을 배분할 수 없다 —
단, 늘어난 깊이가 정확히 그 국소적 `Lip(f)`를 필요한 지점에서 날카롭게 만드는 데 쓰이지 않는 한.
이건 residual block을 쌓아 표현을 풍부하게 만드는 종류의 "깊이"와는 다른 개념이다.
**PDF 본문 렌더링에 실패해 초록만 확보했다 — 수치·실험 세부는 확인 못함 [저확신].**

---

## 4. 온폴리시 PPO에 결여된 조건 — team-lead의 (a)~(e) 항목별 분석

### (a) 데이터 재사용 (few-epoch 온폴리시 vs 고정 데이터 다회 gradient step)

**[추론, 단 3-2의 저자 진술로 뒷받침됨]** CRL의 replay buffer(UTD ~1:40, 확신 낮음)와 BC의
"무한 신선 데이터 + 사실상 정적인 분포에 대한 다회 epoch" 둘 다, 특정 데이터 영역이 매우 많은
gradient update를 거친다는 공통점이 있다 — 깊은 net이 계층적 표현을 조직할 "시간"이 있는 체제다.
PPO는 배치당 소수 epoch 후 그 데이터를 버리므로 이 비율이 구조적으로 낮다. Seohong Park 본인도
"Q-learning is not yet scalable"에서 PPO의 이 제약을 독립적으로 지적한다(3-2 참조).

### (b) trust region/KL 제약과 capacity의 상호작용 — 가장 강력하고 직접 실증된 항목

**[출처, 직접 증거]** "Simple Policy Optimization"(Xu et al., arXiv:2401.16025,
[HTML](https://arxiv.org/html/2401.16025))이 vanilla PPO를 MuJoCo(Ant-v4, HalfCheetah-v4,
Hopper-v4, Humanoid)에서 3층 vs 7층으로 직접 테스트했고 **7층에서 PPO가 붕괴**한다:

| Env | 3층 PPO | 7층 PPO | 7층 SPO |
|---|---|---|---|
| Ant-v4 | 5323.2 | 1002.8 | 4672.5 |
| HalfCheetah-v4 | 4550.2 | 2242.3 | 5307.3 |
| Hopper-v4 | 1119.4 | 975.9 | 1507.6 |

HalfCheetah 최대 확률비 편차는 3층 0.225 -> 7층 **1675.34**로 폭증.

원인 진단: PPO의 ratio 클리핑은 클리핑된 데이터포인트의 gradient를 0으로 만든다 — 클리핑되고 나면
그 지점을 신뢰영역 안으로 되돌릴 교정 gradient가 없다. 네트워크가 깊어져 표현력이 커질수록 이런
표류가 스텝당 더 자주/크게 일어난다. SPO는 클리핑을 연속적인 이차 페널티로 대체해 항상 0이 아닌
gradient를 유지한다:

```
목적함수 비교 (기호는 0장 참조)

PPO:  J(theta) = E[ min( r_t(theta) * A_t,  clip(r_t(theta), 1-eps, 1+eps) * A_t ) ]

SPO:  J(theta) = E[ r_t(theta) * A_t  -  (|A_t| / (2*eps)) * (r_t(theta) - 1)^2 ]
```

7층에서 SPO는 PPO를 회복시키거나 능가한다(Ant-v4 4672 vs 1003, HalfCheetah 5307로 3층 PPO보다도
높음). 보완적 메커니즘 논문 "No Representation, No Trust"(arXiv:2405.00662,
[HTML](https://arxiv.org/html/2405.00662))는 표준 크기 네트워크(3층 CNN + 512-unit MLP)에서도
표현 rank가 붕괴하면 서로 다른 state의 gradient가 공선(colinear)해지면서 한 state에서의 클리핑이
다른 state의 정책 변화를 더는 못 막는다는 상보적 메커니즘을 제시하고, 이를 완화하는 정규화(PFO)를
제안한다.

**결론: PPO가 깊은 net에서 붕괴하는 것은 가설이 아니라 2024년에 이미 실증되고 두 개의 독립적인
후속 논문이 그 위에서 고침을 시도한 사실이다.**

### (c) 부트스트래핑/TD vs 분류형-또는-BC 목적함수

**[출처, 1000-layer 논문 자체 실험]** CRL 논문 내부 baseline인 SAC/TD3(TD 계열)가 깊이에서 이득을
못 봤다는 사실 자체가 근거다. PPO의 value function도 GAE를 통한 TD/부트스트랩 회귀이므로, CRL
저자들이 제시한 가설(분류형 gradient가 스칼라 회귀보다 강건)이 맞다면 그건 PPO의 critic에도 SAC/TD3
critic만큼 적용될 것이다. **[추론]** 다만 PPO의 actor는 그 자체로 부트스트랩되지 않는 direct
policy-gradient 목적함수이므로, actor는 (c)보다는 (b)의 클리핑 실패에 더 노출되고, critic은 (c)의
TD 불안정에 더 노출된다 — 두 네트워크가 "깊이 스케일링에 대해 같은 리스크를 안고 있다"고 뭉뚱그리면
안 된다.

### (d) 정책 확률성 기반 탐색

**[출처 + 추론]** CRL 저자들은 오프라인에서 이득이 사라지는 이유를 "탐색과 깊은 표현학습의 능동적
상호작용"이라고 스스로 설명한다. PPO의 탐색은 온폴리시 엔트로피 기반으로 매 업데이트마다 국소적이고
trust region에 제약되는 반면, CRL은 replay buffer 전체 goal 공간에 걸친 오프폴리시 탐색이다.
**[추론]** 이게 정말 인과적 메커니즘이라면 PPO의 좁은 탐색은 CRL보다 더 안 맞는 조건인데, 흥미롭게도
SAC/TD3는 오프폴리시 탐색을 갖고도 이미 깊이 이득을 못 봤다 — 즉 탐색 방식 하나만으로 결정 요인을
돌리기엔 (c)의 목적함수 종류와 뒤섞여 있어(저자들 자신의 ablation 설계 안에서도) 분리가 안 된다.

### (e) 계산 예산: 롤아웃 bound vs gradient bound

**[순수 추론, 출처 없음]** IsaacLab처럼 대규모 병렬 GPU sim에서 locomotion PPO는 대개 시뮬레이터
스텝률에 bound된다. 그래서 깊은 net의 추가 FLOPs가 CRL/BC 같은 gradient-bound 체제(UTD ~1:40,
반복적인 flow-matching pass)보다 wall-clock 비용이 상대적으로 적게 든다 — "비싸서 못 해본다"는
반박은 rollout-bound locomotion에서는 약하다. 하지만 이건 실험 비용에 대한 얘기지, (b)·(c)를
해결하지 않고 깊이만 늘렸을 때 실제로 도움이 되는지에 대한 답은 아니다.

---

## 5. on-policy PPO로 깊이 스케일링을 실제 시도한 후속 연구

- **부정 -> 처방으로 고침**: Xu et al., "Simple Policy Optimization"(arXiv:2401.16025) — MuJoCo/
  ResNet-18-on-Atari에서 PPO가 7층에서 붕괴, SPO 변형(연속 페널티)이 이를 고침. team-lead 질문에
  가장 직접적으로 답하는 결과.
- **상보적 진단**: "No Representation, No Trust"(arXiv:2405.00662) — 표준/소형 아키텍처에서도
  rank collapse가 신뢰영역을 붕괴시킴, PFO 정규화 제안.
  이 두 논문은 1000-layer 논문(2025-03)보다 앞서며, representation 스케일링이 아니라 trust-region
  이론에서 독립적으로 깊이 문제에 접근한다.
- **긍정, 단 1000-layer 레시피와는 다른 계열**: SimBa(Lee et al., arXiv:2410.09754) — residual
  feedforward block + LayerNorm + observation normalization, PPO+Craftax에 적용해 "유의미하게
  향상"된다고 보고. **정확한 깊이 값·PPO 전용 ablation 수치는 여러 차례 시도했으나 확보 못함**
  [미확인] — abstract 수준의 "긍정적 결과가 존재한다"는 것만 확인.
- **2503.14858(1000-layer 논문)의 정확한 1024층 residual+LayerNorm+Swish 레시피를 on-policy PPO에
  그대로 적용한 직접 후속 연구는 찾지 못했다.** 가장 가까운 것은 SPO/PFO 계열(1000-layer 논문보다
  선행, trust-region 이론에서 독립적으로 접근)과 SimBa(동시대, off-policy 우선·PPO는 부차적 검증).

---

## 6. 판정 (Verdict)

### 6-1. 1000-layer 결과가 on-policy PPO locomotion으로 전이되는가

**부분적으로만(Partial), 사실상 기본값으로는 No에 가깝다.**

근거 체인:

1. [출처] CRL 논문 자체가 자기 실험 안에서 이미 "깊이 스케일링은 오프폴리시/온라인 RL 일반의
   성질이 아니다"를 보여준다 — SAC/TD3(PPO의 critic과 같은 TD 계열)는 이득을 못 본다.
2. [출처] 별도로, vanilla PPO를 실제로 깊게 만든 실험(SPO 논문)은 **이득이 없는 정도가 아니라
   명시적 붕괴**를 보고한다(Ant-v4 5323 -> 1003, ratio 편차 0.225 -> 1675) — 원인은 CRL 논문과
   무관하게 PPO 고유의 클리핑 메커니즘.
3. 결여된 조건 중 (b)가 가장 결정적이고 실증적이다: trust-region/클리핑이 capacity와 충돌하는 것은
   가설이 아니라 관측된 실패 모드다. (c)는 critic에 한해 CRL 논문 자체 증거로 뒷받침된다. (a),(d)는
   방향은 맞지만 CRL 논문의 ablation 설계 안에서도 서로 얽혀 있어 단독 인과로 못 뗀다. (e)는 실험
   비용에 대한 얘기일 뿐 결과를 예측하지 않는다.
4. **"Partial"인 이유**: SPO/PFO가 보여주듯 클리핑 메커니즘을 고치면(연속 페널티, rank-collapse
   정규화) PPO도 더 깊은 net에서 동작하고 심지어 얕은 baseline을 능가한다. 즉 깊이 자체가
   PPO와 원천적으로 상극인 것은 아니고, **PPO의 특정 구현(hard clipping)이 상극**이다. 이 구멍을
   막지 않고 1000-layer 논문의 residual+LayerNorm+Swish 레시피를 그대로 IsaacLab PPO actor/critic에
   붙이면, CRL 논문의 스케일링 성공보다 SPO 논문의 붕괴를 재현할 가능성이 실증 근거상 더 높다.

### 6-2. "1000 layers" 결과는 capacity 얘기인가, expressiveness/다중모드 얘기인가

**Capacity 얘기다. Expressiveness/다중모드와는 별개 축이며, 이 논문은 후자를 전혀 다루지 않는다.**

- [출처] 1000-layer 논문(1장)은 CRL의 (여전히 단일한, 대각 정규분포 또는 CRL의 표준 정책 클래스인)
  actor/value 표현을 더 정확하게 만드는 얘기다 — "같은 입력에서 여러 정답을 표현하는 능력"이 아니라
  "입력으로부터 하나의 출력(또는 그 출력의 파라미터)을 얼마나 잘 근사하는가"를 다룬다.
  다중모드/multimodality라는 단어 자체가 이 논문의 프레이밍에 등장하지 않는다.
- [출처] BC mystery 포스트(2장)도 저자 본인이 다중모드성을 후보 설명으로 올리지 않는다 — 순수하게
  미해결 capacity-or-optimization 미스터리다.
- [출처] Expressiveness/다중모드 주장은 별도 계보(FQL, 3장)와 제3자 논문(Mazza et al., 3-3)에서
  나오며, 거기서의 결론은 **"다중모드는 output 분포 계열/transport map 매끄러움의 성질이지 파라미터
  수의 함수가 아니다"**이다.

**Substitutes인가 orthogonal인가 — orthogonal이다.** [출처 3-3 + 일반적 사실로서의 추론]
대각 가우시안은 몇 층을 쌓든 정의상 한 관측에서 정확히 하나의 mode를 갖는다. 깊이는 "관측 ->
분포 파라미터" 매핑을 더 정확하게 만들 뿐, 그 분포 자체가 한 관측에서 여러 봉우리를 갖게
만들지는 못한다. Mazza et al.의 `Lip(f)` 논증도 같은 결론을 다른 각도(transport map의 매끄러움)에서
뒷받침한다.

### 6-3. 리포지토리의 실제 실패 모드(cmd 0.5 이봉)에 적용 — 두 문서가 도달한 동일 결론

**[추론, `README.md`의 A-1 가설 및 `research_onpolicy_scaling.md`와 독립적으로 일치]**
`README.md`의 A-1 가설은 "같은 관측이 정말로 두 행동을 다 허용한다"(FQL/Mazza가 다루는 진짜
다중모드 상황)가 아니라 **부분관측성**이다: 정지 시 `joint_vel ≈ 0`이라 보행 위상이 순간 관측에서
복원 불가능하고, 그래서 서로 다른 숨은 상태가 같은 관측으로 보인다. 이 경우:

- 다중모드 정책 헤드(GMM/flow/diffusion)를 **같은 63차원 memoryless 관측**에 조건화해봐야, 그
  관측을 다시 만날 때마다 정지/보행 사이를 무작위로 코인플립할 뿐 — 숨은 위상이 요구하는 올바른
  행동을 안정적으로 내지 못한다. 다중모드 표현력은 "한 관측에서 여러 정답 중 하나를 고르는" 문제를
  풀지, "관측 자체에 정답을 고를 정보가 없는" 문제는 풀지 못한다.
- 깊이(capacity)도 같은 이유로 이 실패 모드를 기계적으로 겨냥하지 못한다 — 관측에 없는 정보를
  더 깊은 매핑이 만들어낼 수는 없다.
- 두 축(capacity, expressiveness) 다 이 특정 실패 모드에는 구조적으로 안 맞고, 유일하게 메커니즘이
  일치하는 처방은 memory/phase 정보 추가다(`README.md`의 자체 결론과 일치, `research_onpolicy_scaling.md`
  2장의 recurrent 후보들과 일치).
- **저비용 선행 진단 제안**: 아키텍처 실험에 arm 하나를 쓰기 전에, 참조 모션 데이터셋 자체에서
  동일한 시작 프레임으로부터 갈라지는 서로 다른 두 궤적이 실제로 존재하는지부터 확인할 것
  (motion-analyzer 영역). 있다면 진짜 다중모드성(FQL/Mazza 프레임)이고 GMM 헤드가 후보가 되고,
  없다면 A-1(부분관측성)이 맞고 memory/phase가 유일한 처방이다.
- **[추론, 엔지니어링 참고]** GMM 헤드로 방향을 잡을 경우, PPO의 `r_t(theta)` 계산은 샘플링된
  action의 밀도를 정확하고 저렴하게 구해야 한다. 대각 가우시안은 closed form이지만 flow-matching/
  diffusion 정책의 정확한 likelihood는 probability-flow ODE 적분이 필요해 비싸고, FQL류 설정도
  보통 정확한 likelihood 계산을 피해 학습한다. GMM은 closed-form mixture density를 유지하면서
  PPO의 기존 ratio 계산 구조 안에 남는 가장 저비용 업그레이드 경로다.

---

## 7. 요약 표

| 질문 | 답 | 근거 종류 |
|---|---|---|
| 1000-layer 결과가 PPO locomotion으로 전이되는가 | **Partial (기본값은 No에 가까움)** | 출처(SAC/TD3 자체 실패) + 출처(SPO 붕괴 실증) |
| 어떤 조건이 결여됐는가 | (b) trust-region/클리핑 vs capacity가 가장 결정적, (c) TD critic도 근거 있음 | 출처 |
| 1000-layer는 capacity 얘기인가 expressiveness 얘기인가 | **Capacity.** 다중모드는 다루지 않음 | 출처 |
| capacity와 expressiveness는 substitute인가 orthogonal인가 | **Orthogonal** | 출처(3-3) + 일반 사실 |
| 리포지토리의 cmd 0.5 이봉에 적용하면 | 둘 다 안 맞음 — memory/phase만이 메커니즘상 일치하는 처방 | 추론(A-1과 정합) |

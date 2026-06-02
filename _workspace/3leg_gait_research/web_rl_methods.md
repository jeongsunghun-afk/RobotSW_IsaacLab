# 사족보행 RL의 3-leg / 비대칭(limping) gait — 원인과 해결책 (Web/논문/커뮤니티 조사)

> Task #2 (worker-web-rl). 시뮬레이터 무관, **RL 방법론 관점**.
> 조사일: 2026-06-02. 대상: 일반 quadruped RL (rough terrain 포함).
>
> **Go2 parkour deploy 제약 (양립성 평가 기준):**
> 실제 로봇에 **contact sensor를 observation으로 추가 불가**(sim-to-real). 단, **contact 정보를 reward 내부에서만 사용하는 것은 OK** (reward는 학습 시에만 쓰이고 deploy 시 policy 입력이 아님). phase/clock을 obs에 넣는 것은 결정론적 신호라 deploy 가능. symmetry augmentation은 학습 시에만 작동하므로 deploy 무영향.

---

## TL;DR — 3-leg/비대칭 gait이 생기는 근본 메커니즘

3-leg(한 다리 끌기) 또는 좌우 비대칭 gait은 거의 항상 **"전진 속도 추종 reward는 강한데, gait을 shaping하는 신호가 없거나 잘못 gating되어, 정책이 한두 다리만 쓰는 local optimum에 빠진" 결과**다. 핵심 원인 5가지와 대응 해결책:

| # | 원인 | 해결책 카테고리 | Go2 deploy 양립성 |
|---|------|----------------|------------------|
| 1 | velocity reward만 강하고 gait shaping 부재 → 3다리로도 전진하는 local optimum | Gait/periodic reward (1) | ✅ (reward 내부 contact / obs phase clock) |
| 2 | 좌우/전후 대칭 강제 부재 → 한쪽으로 치우친 정책 | Symmetry augmentation/loss (2) | ✅ (학습 전용, obs 불변) |
| 3 | `feet_air_time` 잘못된 구현/gating → "안 딛는" 정책으로 붕괴 | Reward shaping 정정 (3) | ✅ (reward 내부 contact) |
| 4 | terrain curriculum이 너무 빨리 어려워짐 → 정책이 "버티기"로 도피 | Curriculum 조정 (4) | ✅ |
| 5 | exploration 부족(entropy↓) / action 급변 penalty 과다 → 비대칭 local optimum 고착 | Network/exploration (5) | ✅ |

---

## 1. Gait / periodic reward — first-contact gating & contact schedule

### 메커니즘 (왜 3-leg를 막나)
velocity-tracking reward만 있으면 정책은 "어떤 방식으로든 전진"만 하면 보상을 받으므로, 다리 하나를 끌면서 3다리로 미는 것도 valid solution이다(local optimum). **periodic / clock-based reward**는 각 다리에 **언제 stance(접지)이고 언제 swing(공중)이어야 하는지를 시간 함수로 명시**하여, 모든 다리가 주기적으로 swing/stance를 번갈아 하도록 강제한다. 한 다리를 끌면 그 다리의 "이 구간엔 공중에 있어야 한다"는 reward를 못 받으므로 3-leg는 더 이상 최적이 아니게 된다.

- **Siekmann et al. 2021, "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition" (ICRA 2021):** 각 다리 force/velocity에 대해 **확률적 주기 비용(probabilistic periodic cost)**을 phase 함수로 부여. swing phase엔 "발에 힘이 들어가면 penalty", stance phase엔 "발이 빠르게 움직이면 penalty" 식으로 contact schedule을 reward로 인코딩. 가장 큰 기여: **"reward만으로 원하는 gait을 random seed/hyperparameter에 robust하게 재현"**. 출처: https://arxiv.org/abs/2011.01387
- **Quadruped 적용 (phase/clock obs):** PGTT "Phase-Guided Terrain Traversal"는 per-leg phase를 cubic Hermite spline으로 인코딩하고 **heightmap에 맞춰 swing height를 적응**시키며 **swing-phase contact penalty**를 추가. rough terrain에서 각 다리의 deterministic phase clock을 obs로 주는 패턴. 출처: https://arxiv.org/html/2510.18348
- **gait = foot contact schedule:** 각 gait은 한 주기 안의 고유한 foot contact schedule로 정의된다(barrier-style reward 논문). 출처: https://arxiv.org/html/2409.15780

### Go2 deploy 양립성
- ✅ **phase/clock 신호는 obs에 추가 가능** — sin/cos(2π·phase) 같은 결정론적 시간 신호는 deploy 시 정책이 자체 계산하므로 sensor 불필요. (단, Go2 parkour는 "고정 gait clock"이 task 다양성을 제약할 수 있어 trotting clock을 강제하면 점프/등반에 불리할 수 있음 — soft하게 쓰거나 stance/swing penalty만 reward 내부에서 쓰는 편이 안전.)
- ✅ **contact 기반 swing/stance penalty는 reward 내부 사용** → deploy 무영향.

---

## 2. Symmetry augmentation / mirror loss

### 메커니즘 (왜 3-leg를 막나)
Quadruped는 **sagittal(좌우) 대칭** 구조를 갖는다(몸통 질량 대칭 + 좌우 다리 복제). 비대칭/3-leg gait은 이 대칭을 깨는 정책이다. **symmetry를 강제**하면 "왼쪽으로 입력을 거울 반전하면 행동도 거울 반전돼야 한다"는 제약이 걸려, 한쪽 다리만 쓰는 비대칭 해는 reward/loss에서 불이익을 받거나 애초에 표현 불가능해진다.

두 가지 구현 방식:
1. **Data augmentation:** 매 transition마다 좌우(그리고 전후) 반사된 (state, action) 쌍을 추가로 만들어 학습 → 정책이 근사적으로 equivariant해짐.
2. **Mirror loss:** 정책 출력과 "거울 반전 입력에 대한 거울 반전 출력"의 차이를 loss로 penalize.

- **"Leveraging Symmetry in RL-based Legged Locomotion Control" (IROS 2024, Berkeley):** data augmentation vs mirror loss 두 방식을 비교, **front-back과 left-right 반사 + 그 조합**을 state/action에 적용. 보고된 효과: **더 빠른 수렴, 더 나은 gait quality, 높은 robustness, real-world zero-shot 배포 가능**. 또한 "하드웨어가 완벽히 대칭이 아니어도 transfer 잘 됨". 출처: https://arxiv.org/pdf/2403.17320 / https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf
- **"Symmetry Considerations for Learning Task Symmetric Robot Policies" (ICRA 2024):** equivariant network 구조 vs data augmentation의 trade-off 분석. 출처: https://arxiv.org/html/2403.04359v1
- **MS-PPO: Morphological-Symmetry-Equivariant Policy:** 네트워크 자체를 equivariant하게 설계. 출처: https://arxiv.org/pdf/2512.00727

### ★ IsaacLab/rsl_rl 즉시 적용 가능 (Go2 parkour 직접 관련)
IsaacLab은 `RslRlSymmetryCfg`로 symmetry를 **기본 지원**한다:
- `use_data_augmentation` (bool, default False): 좌우 반사 (obs, action) 쌍 추가 학습
- `use_mirror_loss` (bool, default False): mirror loss 항 추가
- `data_augmentation_func`: 로봇 형상에 맞는 mirror 함수를 **직접 구현해야 함** (obs/action의 어느 index가 좌/우 다리인지 매핑)
- `mirror_loss_coeff` (float, default 0.0): mirror loss 가중치
- 둘 다 False여도 func가 호출되면 **symmetry 메트릭만 로깅** 가능 → 현재 정책이 얼마나 비대칭인지 측정용으로 먼저 켜볼 수 있음.

출처: https://docs.robotsfan.com/isaaclab_official/main/_modules/isaaclab_rl/rsl_rl/symmetry_cfg.html , 사용 예시 논의 https://github.com/isaac-sim/IsaacLab/discussions/2568 , https://github.com/isaac-sim/IsaacLab/issues/2835

### Go2 deploy 양립성
- ✅✅ **완전 양립.** symmetry는 학습 시에만 작동(augmentation/loss), 정책 obs/action 차원 불변, deploy 시 아무 영향 없음. parkour의 좌우 비대칭 gait 의심 시 **가장 먼저 시도할 만한, deploy 리스크 0인 도구.**
- ⚠️ 구현 주의: `data_augmentation_func`에서 obs index ↔ 좌/우 다리 매핑을 정확히 해야 함. parkour obs에 height_scan/heightmap이 있으면 그 grid도 좌우 반전 필요.

---

## 3. Reward shaping — feet_air_time 올바른 구현 + foot clearance

### 메커니즘 & 흔한 버그
`feet_air_time` reward의 표준 구현 (legged_gym):
```python
# 발이 "처음 접지하는 순간"에만, 그동안 공중에 있던 시간(air_time)에서 threshold를 뺀 값을 보상
reward = sum((feet_air_time - 0.5) * first_contact)   # first_contact: 이번 step에 막 닿은 발만 1
reward *= (norm(cmd_vel) > 0.1)                         # 정지 명령 시엔 비활성
```
출처: https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py

**왜 3-leg/안-딛기로 붕괴하나 (핵심 함정):**
- **first_contact gating이 없으면** 발이 그냥 공중에 오래 떠 있는 것만으로 계속 보상 → 정책이 **다리를 들고 안 딛는** degenerate 해로 도망. (커뮤니티 다수 보고)
- **weight를 너무 키우면** "스텝을 아예 안 밟는" 정책으로 붕괴. IsaacLab Discussion #1977: weight 0.125 → 크게 올리니 "strange behaviors", 한 유저 *"Whenever I train with the feet air time reward the policy doesn't learn to take a step at all."* threshold를 낮추거나 weight를 올려도 해결 안 됨. 2025-11 기준 **공식 미해결**. 출처: https://github.com/isaac-sim/IsaacLab/discussions/1977
- threshold(0.5초)가 task에 안 맞으면(빠른 gait인데 0.5초 air time 요구) 비대칭으로 회피.

**올바른 사용 원칙:**
1. **반드시 first_contact에 gating** (공중에 떠 있는 것 자체가 아니라 "주기적으로 딛는 것"을 보상).
2. velocity-tracking weight 대비 air_time weight를 **작게**(예: legged_gym default 1.0 tracking vs 1.0 air_time이지만 scale·dt 곱 후 실효치 확인). 너무 크면 안-딛기 붕괴.
3. **단독으로 대칭을 보장하지 못함** — feet_air_time은 "각 발이 얼마나 오래 떠 있나"만 보지, 좌우 균형을 강제하지 않음. 비대칭 해결엔 symmetry(2) 또는 per-leg contact-balance reward 병행 필요.

### 보완 reward
- **foot clearance reward:** swing 중 발 높이를 목표치로 유도(barrier function). rough terrain에서 발 끌기/걸림 방지. 출처: https://arxiv.org/html/2409.15780
- **PGTT swing-phase contact penalty:** swing이어야 할 때 접지하면 penalty → 끌기 억제. 출처: https://arxiv.org/html/2510.18348
- **gait symmetry / contact-balance reward:** 좌우 또는 대각 다리쌍의 air_time/contact 시간이 비슷하도록 보상(여러 quadruped 논문에서 "narrow velocity range가 symmetric gait(pace/bound/pronk) 촉진"이라 보고). 출처: https://arxiv.org/pdf/2403.10723

### Go2 deploy 양립성
- ✅ feet_air_time, foot clearance, contact-balance 모두 **reward 내부에서 contact/foot pos 사용** → deploy 시 정책 입력 아님, 양립.

---

## 4. Curriculum / terrain difficulty 조정

### 메커니즘 (왜 3-leg를 유발하나)
terrain curriculum이 **너무 빨리 어려워지면**, 정책이 정상 보행 gait을 충분히 학습하기 전에 험지에 노출 → "전진"보다 "넘어지지 않기/버티기"가 유리해져 **3다리로 버티거나 한 다리 끌며 기는** 보수적 비대칭 해로 도피한다. 반대로 너무 쉬우면 gait이 안 다듬어짐. 핵심은 **정책의 현재 능력에 맞춰 난이도를 올리는 것**.

- **legged_gym/Rudin et al. "Learning to Walk in Minutes" (game-inspired curriculum):** 로봇이 명령 거리의 일정 비율 이상 전진하면 다음 난이도로 승급, 못 하면 강등. **성능 기반 promotion/demotion**이 핵심 — hand-tuned threshold가 너무 공격적이면 도피 gait 유발. 출처: https://github.com/leggedrobotics/legged_gym
- **LP-ACRL (Automatic Curriculum, 2026):** manual curriculum의 한계(난이도 순서 모호) 지적. **학습 진척도(learning progress)를 온라인 추정**해 task 샘플링 분포를 적응 조절 → "느린 계단 보행 vs 빠른 자갈 주행" 같은 난이도 비교 불가 상황 해결. 출처: https://arxiv.org/html/2601.17428v1
- **Terrain-specialized policies:** 각 환경의 locomotion 한계에 맞춘 terrain-specific curriculum. 출처: https://arxiv.org/pdf/2509.20635

### 권장 조치 (Go2 parkour에 적용)
1. **flat/easy terrain에서 대칭 gait이 충분히 수렴한 뒤** 난이도 상승 (gait warm-up).
2. promotion threshold를 완화 (너무 빠른 승급 = 도피 gait).
3. demotion 활성화 — 실패하면 쉬운 terrain으로 되돌려 gait 재학습 기회 부여.
4. velocity command range를 너무 넓게/빠르게 주지 말 것 (narrow range가 symmetric gait 촉진).

### Go2 deploy 양립성
- ✅ curriculum은 순수 학습 시 환경 설정 → deploy 무관.

---

## 5. Network / exploration (entropy, action smoothness)

### 메커니즘 (왜 3-leg에 고착되나)
3-leg gait은 전형적인 **local optimum**이다. PPO는 한번 비대칭 해에 수렴하면 entropy(탐험)가 낮아 그곳을 못 벗어난다. 또한 **action rate/smoothness penalty가 과도하면**, 정책이 "다리를 활발히 흔드는" 큰 action 변화를 회피하고 **최소 움직임(한 다리 고정)**으로 수렴 → 비대칭 고착.

- **ECIM "Entropy-Controlled Intrinsic Motivation" (2025):** 표준 PPO는 **복잡 terrain에서 suboptimal locomotion(local optima)에 자주 빠지고 deep exploration이 부족**하다고 명시. **adaptive entropy scheduling + action smoothness regularization + intrinsic reward**로 탐험/안정/적응 균형. 출처: https://arxiv.org/html/2512.06486
- **Quadruped locomotion RL review (2410.10438):** reward 설계가 local optima 회피의 핵심이며 non-terminal episode 종료 처리가 중요. 출처: https://arxiv.org/pdf/2410.10438
- action smoothness trade-off: smoothness 강화는 motion quality↑이지만 과하면 반응성↓·minimal-motion 도피. 출처: https://arxiv.org/html/2512.06486

### 권장 조치
1. **entropy coefficient를 일시적으로↑** (또는 scheduling) — 비대칭 local optimum 탈출 유도. parkour PPO에서 entropy가 너무 빨리 0으로 수렴하는지 확인.
2. **action rate/smoothness penalty가 과도하지 않은지 점검** — 너무 크면 "안 움직이는" 비대칭 해 유발. velocity tracking 대비 상대 크기 확인.
3. 초기 random scale / 탐험 noise를 충분히 확보.

### Go2 deploy 양립성
- ✅ entropy/smoothness는 학습 하이퍼파라미터 → deploy 무관. action smoothness penalty는 오히려 sim-to-real에 유리(부드러운 동작).

---

## 카테고리별 근거 확보 요약

| 카테고리 | 근거 | Go2 deploy 양립성 |
|---------|------|------------------|
| 1. Gait/periodic reward | ✅ 확보 (Siekmann 2021, PGTT, barrier-style) | ✅ (phase obs / contact reward) |
| 2. Symmetry aug/loss | ✅✅ 강력 확보 + **IsaacLab 기본 지원** | ✅✅ deploy 리스크 0 |
| 3. feet_air_time / foot clearance | ✅ 확보 (legged_gym 구현 + IsaacLab #1977 함정) | ✅ (reward 내부 contact) |
| 4. Curriculum 조정 | ✅ 확보 (Rudin, LP-ACRL) | ✅ |
| 5. Network/exploration | ✅ 확보 (ECIM, review) | ✅ |

> **"근거 못 찾음" 항목: 없음.** 5개 카테고리 모두 출처 확보.

---

## Go2 parkour 비대칭/3-leg 의심 시 우선순위 권고 (RL 방법론 관점)

1. **[리스크 0, 먼저] Symmetry 메트릭 로깅 → data augmentation/mirror loss 켜기** (IsaacLab `RslRlSymmetryCfg`). deploy 무영향, 비대칭 직접 타격.
2. **feet_air_time이 first_contact gating 되어 있는지, weight가 과도하지 않은지 점검** (#1977 붕괴 사례). contact-balance(좌우 air_time 균형) reward를 reward 내부에 추가 검토.
3. **terrain curriculum promotion이 너무 공격적이지 않은지** + flat warm-up 충분한지 확인.
4. **entropy coef / action smoothness penalty 균형** 점검 (local optimum 탈출 + minimal-motion 도피 방지).
5. parkour의 task 다양성(점프/등반) 때문에 **고정 gait clock 강제는 신중히** — soft phase 또는 swing/stance contact penalty(reward 내부)로 제한적 적용.

---

## Sources (URL)

- Siekmann et al. 2021, Periodic Reward Composition (ICRA): https://arxiv.org/abs/2011.01387
- PGTT Phase-Guided Terrain Traversal: https://arxiv.org/html/2510.18348
- Barrier-Based Style Rewards (diverse legged locomotion): https://arxiv.org/html/2409.15780
- Leveraging Symmetry in RL-based Legged Locomotion (IROS 2024): https://arxiv.org/pdf/2403.17320 , https://hybrid-robotics.berkeley.edu/publications/IROS2024_Symmetry_RL_LeggedLoco.pdf
- Symmetry Considerations for Task Symmetric Policies (ICRA 2024): https://arxiv.org/html/2403.04359v1
- MS-PPO Morphological-Symmetry-Equivariant Policy: https://arxiv.org/pdf/2512.00727
- Symmetry-Guided RL Hierarchy / Gaits (symmetric gait & velocity range): https://arxiv.org/pdf/2403.10723
- IsaacLab RslRlSymmetryCfg docs: https://docs.robotsfan.com/isaaclab_official/main/_modules/isaaclab_rl/rsl_rl/symmetry_cfg.html
- IsaacLab Discussion #2568 (Symmetry 사용 예): https://github.com/isaac-sim/IsaacLab/discussions/2568
- IsaacLab Issue #2835 (Symmetry 예시 요청): https://github.com/isaac-sim/IsaacLab/issues/2835
- legged_gym (feet_air_time 구현, curriculum): https://github.com/leggedrobotics/legged_gym , https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py
- IsaacLab Discussion #1977 (feet_air_time tuning 함정, 미해결): https://github.com/isaac-sim/IsaacLab/discussions/1977
- LP-ACRL Automatic Curriculum (2026): https://arxiv.org/html/2601.17428v1
- Terrain-Specialized Policies: https://arxiv.org/pdf/2509.20635
- ECIM Entropy-Controlled Intrinsic Motivation (2025): https://arxiv.org/html/2512.06486
- Quadruped Locomotion RL review: https://arxiv.org/pdf/2410.10438

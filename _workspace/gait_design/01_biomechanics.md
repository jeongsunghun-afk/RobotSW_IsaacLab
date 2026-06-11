# 01 — 생체역학: 동물 사족보행 Gait 패턴 / 에너지 / 협응

> 작성: worker-1 (task #1) · 2026-06-10
> 목적: 동물 사족보행의 gait 선택 에너지 원리, pronk/bound/trot 비교, 전후협응, 착지충격 완화를
> 문헌으로 정리하고 **우리 Go2 parkour 문제(평지 bounding + 장애물 4발 pronk + 분할점프)** 에 주는 함의를 도출.
>
> 표기 규칙: **[VERIFIED]** = 원문/2차 문헌에서 직접 확인한 수치/주장 · **[LIT]** = 문헌의 정성적 주장 · **[추정]** = 본 문서의 추론(문헌 미확인).
> 추측 금지 원칙에 따라, 못 찾은 수치는 명시적으로 "원문 미확보"라고 적었다.

---

## 0. 핵심 요약 (TL;DR)

1. **Gait는 속도별 에너지 최소화로 선택된다.** 각 gait(walk/trot/gallop)의 산소소모-속도 곡선은 U자형이고, 동물은 그 속도에서 CoT가 최소인 gait를 고른다 (Hoyt & Taylor 1981). → **특정 지형/속도에서 "올바른 gait"가 존재**한다는 것을 생체역학이 지지.
2. **Pronk(stotting)는 에너지적으로 비싼 특수목적 gait.** 4발 동시 도약 + 긴 flight phase로 무게중심 수직진동·포텐셜에너지 낭비가 크다. 동물에서는 효율 주행이 아니라 **포식자 신호(honest signal)** 나 험지 통과용으로만 쓰인다 (Caro 1986; Maynard Smith & Harper 2003). → **평지/계단에서 pronk는 정상 보행이 아니라 회피해야 할 패턴.**
3. **전후(앞-뒤다리) 협응은 능동적 신경 메커니즘이다.** 사족동물은 앞다리가 장애물을 넘은 위치를 **기억(working memory)** 해 뒷다리를 순차적으로 같은 자리에 딛는다 (McVea & Pearson; Wanchisen 류 horse 연구). 협응은 "공짜로 emergent"한 게 아니라 별도 신호로 유지된다. → **분할점프(앞발 착지→재이륙→뒷발 착지)는 이 협응 신호의 부재 증상.**
4. **착지충격은 시간적 분산으로 완화된다.** 다리 순차 착지 + 관절 compliance로 peak force를 낮추고 impulse를 늘려 받는다. 동시 4발 강착(stiff pronk landing)은 peak force를 한 시점에 집중시킨다. → 우리 case에서 안전 penalty는 NO-GO였지만, **자연스러움/효율 관점에서 동시강착은 여전히 나쁜 패턴**.

---

## 1. Gait 선택의 에너지 원리

### 1.1 Hoyt & Taylor 1981 — gait별 CoT 최소화

- **출처**: Hoyt, D.F. & Taylor, C.R. (1981). "Gait and the energetics of locomotion in horses." *Nature* **292**, 239–240. https://www.nature.com/articles/292239a0
- **방법** [VERIFIED]: 작은 말 3마리(약 110–170 kg)를 트레드밀에서 walk/trot/gallop 시키고, 명령으로 각 gait를 "자연 속도 범위 밖"까지 확장. 산소소모율을 에너지소모 대용으로 측정.
- **핵심 결과** [VERIFIED]:
  - 각 gait의 (산소소모 vs 속도) 관계는 **U자형**. 즉 gait마다 CoT(단위거리당 에너지)가 최소가 되는 고유 속도가 있다.
  - walk / trot / gallop이 각각 **저속 / 중속 / 고속**에서 에너지 효율적.
  - 동물이 자연스럽게 선택하는 gait는 그 속도에서 **가능한 최소 에너지**를 쓰는 gait와 일치 → 동물은 속도 증가 시 walk→trot→gallop으로 바꿔 CoT를 최소로 유지.
- **원문 미확보 수치**: walk/trot/gallop 각각의 최소-CoT 절대값(ml O₂·kg⁻¹·m⁻¹)과 정확한 전이속도(m/s)는 원문 PDF 미확보로 직접 인용 불가. (2차 문헌은 "U-shaped, 저/중/고속" 정성적 기술만 재현.)

> **함의 (우리 문제)**: "지형/속도마다 에너지 최적 gait가 존재한다"는 명제는 생체역학적으로 강하게 지지됨. 평지 정속 주행에서 **bounding/pronk가 trot보다 효율적일 이유가 없다** → 우리가 관찰한 평지 bounding은 에너지 최적이 아닌 **RL local optimum**일 가능성을 생체역학이 뒷받침.

### 1.2 Alexander의 dynamic similarity & Froude number

- **출처**: Alexander, R.McN. & Jayes, A.S. (1983). "A dynamic similarity hypothesis for the gaits of quadrupedal mammals." *J. Zoology* 201:135–152. (요약: https://www.researchgate.net/publication/230222551)
  - 보강: *Dynamically similar locomotion in horses*, J. Exp. Biol. 209:455 (2006). https://journals.biologists.com/jeb/article/209/3/455
- **핵심 주장** [LIT]: 서로 다른 동물도 **같은 Froude number**(Fr = v²/(g·L), L=다리길이/hip height)에서 움직이면 *동역학적으로 유사* — 같은 상대 stride length, 같은 **duty factor**, 같은 phase relationship(다리 timing), 체중배수로 같은 지면반력.
- **Gait 전이의 무차원 기준** [LIT]: 동물은 대략 Fr ≈ 0.5 근방에서 walk→trot/run으로 전이(보행 추정), gallop 전이는 더 높은 Fr. Alexander의 Fr 상관과 Hoyt&Taylor의 에너지 최소화는 **동일 현상의 두 관점**(에너지 최소 = 동역학 유사 속도에서 gait 전환).

#### duty factor 참고값

| Gait | Duty factor (대략) | 비고 |
|------|------|------|
| Walk | ~0.69 (>0.5) | 항상 ≥1발 접지, 정적 안정 가능 |
| Trot | ~0.61 | 대각 2발 |
| Bound | ~0.60 | flight phase 존재 |
| Pronk | (낮음, flight 큼) | 4발 동시, 긴 체공 |

> 주: 위 0.69/0.61/0.60 수치는 검색으로 확인된 2차 문헌의 보고값이며 **원 출처(종/속도 조건)는 미확정**. [LIT] 수준으로만 사용 권장. "walk는 duty factor가 높아(>0.5) 정적 안정, dynamic gait일수록 duty factor↓·flight↑"라는 **정성적 경향**은 일관되게 지지됨.

> **duty factor = 한 다리가 보폭 주기 중 접지해 있는 비율.** 우리 메모의 contact_duty 신호와 직접 대응되는 생체역학 양 — 즉 **footfall duty는 임의 reward hack이 아니라 gait를 규정하는 1차 생체역학 변수**다.

---

## 2. Pronk / Bound / Trot / Pace 비교 — 왜 pronk가 비효율인가

### 2.1 에너지 비교

- **출처**:
  - Stotting, Wikipedia (Caro 1986; Maynard Smith & Harper 2003 인용). https://en.wikipedia.org/wiki/Stotting
  - *Energetic Analysis on the Optimal Bounding Gaits of Quadrupedal Robots*, arXiv:2303.04861. https://arxiv.org/pdf/2303.04861
  - *Control of underactuated planar pronking ...*, Autonomous Robots (2010). https://link.springer.com/article/10.1007/s10514-010-9216-x
- **주장** [LIT]:
  - Pronk(stotting) = 4발을 **동시에·뻣뻣하게** 펴 수직 도약, 느린 전진속도 + **긴 flight phase + 큰 도약 높이**. "같은 노력으로 bound보다 더 높이 뜬다" (점프 높이↑ = 전진효율↓).
  - flight phase 동안 **무게중심을 능동 제어할 수 없다** → 수직진동에 들어간 운동/포텐셜 에너지는 전진에 기여 못 하고 착지에서 소실. 이것이 비효율의 물리적 핵심.
  - 로봇 에너지 최적화 연구: A1 로봇에서 **두 번의 flight를 갖는 bounding(B2)이 가장 에너지 효율적**이며, pronk류 단일 대형 flight는 비효율 (arXiv:2303.04861).
- **stotting의 실제 기능** [VERIFIED]: 효율 주행이 아님. (a) 포식자에 대한 **honest signal**(잡기 어렵다는 정직 신호) — 치타는 gazelle이 stot하면 사냥 포기율↑, 추격해도 성공률↓ (Caro 1986). (b) 부차적으로 장애물/큰 풀 위 통과·anti-ambush 가설(증거 제한적). → **pronk는 "효율"이 목적인 적이 없다.**

### 2.2 무게중심 수직진동 / 포텐셜에너지 낭비

- [LIT] walking = inverted pendulum(운동↔포텐셜 교환으로 에너지 회수), running/trot = spring-mass(SLIP, 건·탄성으로 에너지 저장·반환) — 둘 다 **에너지 재활용** 메커니즘 (Alexander; SLIP 모델 문헌, J. Exp. Biol. 208:1645).
- [추정/물리] pronk의 4발 동시 stiff 도약은 이 진자/스프링 위상교환을 활용하기보다 무게중심을 **한꺼번에 위로 던졌다 받는** 형태 → 수직 운동에너지 회수율이 낮고 착지 충격으로 소산. (정량 수치는 우리 case 미측정.)

> **함의 (우리 문제)**:
> - **평지 bounding**: trot 대비 flight phase가 있어 정속 효율이 낮을 개연성. 우리 측정(평지 CoT 기준)과 대조 필요(다른 worker 영역).
> - **장애물 4발 pronk (계단 27.7% all-airborne)**: 생체역학적으로 **명백한 비효율 패턴**. 동물은 계단/턱을 pronk로 넘지 않고, 앞다리→뒷다리 순차 stepping으로 넘는다(§3). 우리 정책의 4발 pronk는 "지형 극복은 하되 에너지·자연스러움을 희생한 RL 편법".
> - **단, 진짜 점프가 불가피한 지형(넓은 gap)** 에서는 도약 자체는 정당. 문제는 *gap이 아닌 곳*에서의 pronk와, gap에서도 *전후협응 없는 분할점프*.

---

## 3. 전후(앞다리-뒷다리) 협응 — 장애물 통과 전략

### 3.1 협응은 능동 신경 메커니즘 (emergent 아님)

- **출처**:
  - Wanchisen 외 / Whishaw 류 — "Hind limb stepping over obstacles in the horse guided by place-object memory" *Behav. Brain Res.* (2009). https://pubmed.ncbi.nlm.nih.gov/19071161/
  - McVea & Pearson — "Long-Lasting Working Memories of Obstacles ... Require Area 5 of the Posterior Parietal Cortex" (cat). https://pmc.ncbi.nlm.nih.gov/articles/PMC6665557/
- **핵심 주장** [VERIFIED/LIT]:
  - 사족동물은 앞다리가 장애물을 넘은 뒤, 그 위치의 **기억(working memory)** 으로 뒷다리를 같은 자리로 들어 넘긴다. 뒷다리는 머리·눈이 이미 지난 뒤 장애물에 접근하므로 시각이 아닌 기억에 의존.
  - **말**: 장애물 기억이 **최대 15분** 지속, 시각 차단/장애물 제거에도 뒷다리 들기 유지 (place-object memory).
  - **고양이**: 오른앞→왼앞→오른뒤→왼뒤 **순차적**으로 같은 위치를 넘음. 이 working memory는 **posterior parietal cortex Area 5** 손상 시 사라짐 → 별도 신경 회로가 협응을 담당.
- **시퀀스 정성** [LIT]: 정상 보행 장애물 통과는 **하나의 "동시 4발 도약"이 아니라**, 각 다리가 시간차로 같은 지점을 순차 통과하는 **협응된 sequence**. 앞다리 동작 정보가 뒷다리 동작을 *조건짓는다*.

> **함의 (우리 문제) — 핵심**:
> - 우리가 관찰한 **분할점프(front feet land → re-takeoff → rear feet land)** 는 생체역학적으로 본 "협응된 순차 stepping"과 **다르다**. 동물의 순차 stepping은 *몸통이 한 번의 연속 운동으로 장애물을 통과*하며 앞-뒤가 같은 궤적을 공유; 분할점프는 **앞몸을 먼저 올리고 다시 점프**하는 2단 동작으로, 앞다리 동작이 뒷다리를 안내하지 못함(협응 신호 부재).
> - 문헌은 **전후협응이 "공짜 emergent"가 아니라 전용 메커니즘(기억/Area5)** 임을 보임 → 우리 메모의 진단("reward에 footfall-pattern 신호 없음, mirror는 L/R만 강제, fore-hind는 강제 안 함")과 정확히 부합. **전후협응은 별도 reward 신호로 명시해야 emergent할 수 있다**는 것을 생체역학이 간접 지지.
> - **주의(검증 구분)**: "동물의 obstacle stepping = 우리가 원하는 fore-hind timing"은 **생체역학 유추[LIT]** 이지, 우리 Go2에서 그 timing이 최적이라는 **측정[VERIFIED]은 아님**. reward 설계 시 "특정 phase offset 강제"는 over-constraining 위험 — §종합 문서에서 다룰 것.

### 3.2 협응된 도약 vs 분할 동작 (정성 비교표)

| 항목 | 협응된 순차 stepping (동물 정상) | 분할점프 (우리 관찰) |
|------|------|------|
| 앞-뒤 timing | 앞다리 궤적이 뒷다리를 안내(기억 기반) | 앞다리 착지 후 독립적 재이륙 |
| 무게중심 | 한 번의 연속 통과 | 2단(앞올림→재점프) |
| 에너지 | 1회 통과, 탄성 활용 가능 | 재이륙 추가 비용(우리 측정: retakeoff/flight stair 0.060 vs flat 0.002) |
| 신경/제어 근거 | working memory, Area5 [VERIFIED] | reward에 fore-hind 신호 부재 [VERIFIED, 우리 메모] |

---

## 4. 착지 충격 완화 — 시간적 분산

- **출처**:
  - "Joints with angle dependent damping can help to reduce impact forces in robots" *Sci. Rep.* (2025). https://www.nature.com/articles/s41598-025-13055-7
  - Drop-landing GRF 리뷰 (human) — peak vGRF systematic review. https://pmc.ncbi.nlm.nih.gov/articles/PMC4160626/ , https://pmc.ncbi.nlm.nih.gov/articles/PMC8826326/
- **핵심 주장** [LIT/VERIFIED]:
  - **peak force vs impulse**: impulse(=∫F dt = 운동량 변화)는 착지 운동량으로 거의 고정. 같은 impulse를 **짧은 시간에 받으면 peak force↑(강착), 긴 시간에 받으면 peak force↓(연착)**. 자연 동물은 관절 굽힘·근육 음성일(eccentric)·건 compliance로 접촉시간을 늘려 peak를 낮춘다.
  - 인간 soft landing은 hard landing 대비 **근육이 최대 ~19% 더 많은 운동에너지 흡수** (peak GRF↓) — 출처: jump-landing 리뷰.
  - 로봇: **가변 stiffness/damping**이 soft landing을 만들고, 상수 stiffness는 충격성 force peak를 만든다(Sci. Rep. 2025; hopping robot 연구).
  - **순차 착지 효과**: 두 다리 착지의 peak가 *불완전하게* 동시일 때, 한 발의 peak는 두 발 합 peak의 절반보다 약간 큼 — 즉 **완전 동시 착지일수록 순간 합력이 최대**, 시간차 착지는 peak를 분산.
- [추정/물리] 4발 동시 stiff pronk landing = impulse를 한 시점에 집중 → 동물의 순차·compliant 착지보다 peak force가 높음. 다리 순차 착지(앞→뒤)는 충격을 시간축으로 펴서 peak를 낮춤.

> **함의 (우리 문제)**:
> - 우리 case에서 **안전 penalty(착지충격/saturation)는 NO-GO** 로 판정됨(임계 미달). 따라서 "peak force가 위험"이라는 프레임은 **재사용 금지**.
> - 그러나 **자연스러움/효율** 관점에선, 동시강착(4발 동시 착지)은 시간분산 착지보다 비자연스럽고, retakeoff/flight 증가와 같은 비효율과 동반된다. 즉 착지충격 신호는 **안전이 아니라 "패턴 품질"의 대용 지표**로만 의미 — 직접 reward로 쓰기보다 **footfall 시간분산(순차 착지)을 유도하는 신호**가 더 적절. (구체안은 종합 문서.)
> - **제약 재확인(메모)**: contact-sensor를 *observation*에 넣는 건 sim-to-real 때문에 금지. 단 **reward 계산 내부에서 contact/force를 쓰는 것은 obs와 별개** — 종합 문서에서 이 경계를 명확히 할 것.

---

## 5. 우리 문제에 대한 종합 함의 (문헌 → 가설)

> 아래는 §1–4 문헌을 우리 측정 사실에 매핑한 것. **[VERIFIED]=우리 세션 측정 또는 원문 수치, [LIT]=문헌 정성 주장, [추정]=본 문서 추론.**

1. **"교대보행(trot)이 평지/계단에서 효율·충격상 유리, pronk는 점프 불가피 지형만"** 명제:
   - 문헌 지지 **강함** [LIT]: gait별 에너지 최소화(Hoyt&Taylor), trot/bound가 pronk보다 효율(arXiv 2303.04861), stotting은 효율목적 아님(Caro). duty factor·SLIP 에너지 회수가 trot/walk의 효율을 설명.
   - 단 "계단을 trot으로"는 직접 문헌 없음 — 동물은 계단류를 *순차 stepping*(§3)으로 넘음. 정확히는 **"비점프 지형=순차 접지 교대보행, 점프 지형=협응된 도약"**.

2. **우리의 4발 pronk + 분할점프가 왜 나쁜가** (생체역학):
   - 4발 pronk(계단 27.7% all-airborne) = 무게중심 수직진동 에너지 낭비 + flight 중 무제어 [LIT]. 동물은 이렇게 계단을 넘지 않음.
   - 분할점프 = 전후협응 부재의 직접 증상. 동물의 협응은 working memory/Area5라는 **전용 메커니즘** [VERIFIED-동물]; 우리 reward엔 대응 신호가 없음 [VERIFIED-우리메모] → 분할점프는 예견된 결과.
   - 측정 일치 [VERIFIED-우리]: retakeoff/flight stair 0.060 vs flat 0.002, CoT stair 1.49× flat — 비효율 정량 확인. 문헌은 이 비효율의 *물리적 이유*(수직진동·재이륙비용·협응부재)를 제공.

3. **설계로 넘기는 권고(요지, 상세는 00_DESIGN.md)**:
   - footfall **duty factor**는 생체역학 1차 변수 → reward 신호로 정당 [LIT].
   - **fore-hind 협응은 명시 신호 필요** (emergent 기대 금지) [LIT].
   - 착지충격은 **안전이 아닌 품질 대용** — 시간분산(순차 접지) 유도가 정도 [LIT/추정].
   - 단, 특정 phase offset 강제는 over-constraining 위험 → **금지 패턴(동시 4발 airborne)을 줄이는 negative/soft 신호**가 특정 gait 모방보다 안전 [추정].

---

## 6. 못 찾은 것 / 한계 (정직성)

- **Hoyt & Taylor 1981 원문 PDF 미확보**: walk/trot/gallop 최소 CoT 절대값(ml O₂·kg⁻¹·m⁻¹), 정확한 전이 속도(m/s) 직접 인용 불가. 2차 문헌의 정성 기술(U자형, 저/중/고속)만 확인.
- **duty factor 0.69/0.61/0.60**: 2차 문헌 보고값, 원 출처·종·속도 조건 미확정 → [LIT]로만.
- **Go2 스케일에서의 정량 매핑 없음**: 동물 수치(말·고양이·gazelle)는 우리 Go2(소형 사족, hip height ~0.3 m)와 Froude 스케일이 다름. 본 문서의 동물 수치는 **정성 원리**로만 사용, 우리 reward의 절대 임계는 측정(다른 worker/실험)으로 정해야 함.
- **"분할점프가 동물 순차 stepping보다 나쁘다"는 정량 비교 부재**: 생체역학적 정성 유추 [LIT/추정]이며, 우리 Go2에서 어떤 fore-hind timing이 에너지 최적인지는 미측정.
- **stotting flight 중 에너지 손실의 정량(%)**: 우리 case 미측정, 동물 일반 수치도 본 조사에서 단일 합의값 미확보.

---

## 출처 목록 (links)

- Hoyt & Taylor (1981) *Nature* 292:239 — https://www.nature.com/articles/292239a0
- Alexander & Jayes (1983) dynamic similarity — https://www.researchgate.net/publication/230222551
- Dynamically similar locomotion in horses (2006) *JEB* 209:455 — https://journals.biologists.com/jeb/article/209/3/455
- Models and scaling of energy costs (2005) *JEB* 208:1645 — https://journals.biologists.com/jeb/article/208/9/1645
- Energetic Analysis on Optimal Bounding Gaits (arXiv:2303.04861) — https://arxiv.org/pdf/2303.04861
- Control of underactuated planar pronking (2010) Auton. Robots — https://link.springer.com/article/10.1007/s10514-010-9216-x
- Stotting (Caro 1986; Maynard Smith & Harper 2003) — https://en.wikipedia.org/wiki/Stotting
- Hind limb stepping over obstacles in the horse (2009) — https://pubmed.ncbi.nlm.nih.gov/19071161/
- McVea & Pearson, obstacle working memory / Area 5 (cat) — https://pmc.ncbi.nlm.nih.gov/articles/PMC6665557/
- Joints with angle-dependent damping reduce impact (2025) *Sci. Rep.* — https://www.nature.com/articles/s41598-025-13055-7
- Peak vGRF two-leg landing systematic review — https://pmc.ncbi.nlm.nih.gov/articles/PMC4160626/
- Drop-landing GRF variability (107 adults) — https://pmc.ncbi.nlm.nih.gov/articles/PMC8826326/
- Extension to collisional model / trot–canter transition — https://pmc.ncbi.nlm.nih.gov/articles/PMC6916616/

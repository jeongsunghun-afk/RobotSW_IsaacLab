# 05 — 사용자 제안 알고리즘 ①/②-1/②-2 비판적 점검 (critic)

> 작성: team worker `critic` (Task #5) · 2026-06-12
> 입력: 01_motion_latent_papers.md · 02_terrain_imitation_papers.md · 03_ood_skill_papers.md · 04_codebase_assets.md
> 추가 정독(본 task에서 수행): Motion Priors Reimagined 본문(arXiv html 2505.16084v2), PULSE 본문(2310.04582v2), AdaptNet(검색 스니펫 + Roblox 연구페이지). MDPI Terrain-Conditional AMP는 403으로 **재차 본문 미확인 → load-bearing 인용에서 제외**.
> 입장: 본 문서는 제안을 옹호하지 않는다. 근거(논문/코드 경로) 없는 판정은 "가설"로 표기한다.

---

## (0) 판정 요약표

| 항목 | 판정 | 한 줄 근거 | 핵심 위험 |
|------|------|-----------|-----------|
| **① 모션 AE** | ✅ 적절 | anchor vMF-VAE(2507.00677), PULSE cVAE(2310.04582), Lifelike VQ-VAE(2308.15143)로 검증 | 평지 데이터 풍부 → 문제 없음 |
| **① 지형 AE (decoder 포함 standalone)** | ❌ 부적절(과잉) | 선행연구는 지형을 **encoder**(heightmap→feature)로만 사용, 지형 재구성 decoder 필요 사례 없음. 코드에 `scan_encoder(187→32)` 이미 존재(04 §2.2) | 지형은 주어지는 입력 → 생성/복원 불필요. AE의 decoder 반쪽 낭비 |
| **① (지형,모션) joint AE (대칭 3종)** | ⚠️ 조건부(재설계 필요) | 문헌 수렴점은 "대칭 joint AE"가 아니라 **terrain-conditioned motion prior**(PULSE learnable prior / CALM conditioning). 대칭 joint AE 직접 선례 미발견 | paired 데이터 희소(jump 소량) → 대칭 manifold 학습 불가능에 가까움 |
| **②-1 (AE decoder로 reference 생성 → IL)** | ⚠️ 조건부 | "decode 후 IL"은 해석에 따라 위험도 극과 극. retargeting+품질게이트 없으면 IL 오염 | (i) AE 외삽=비물리 모션, (ii) 순수 IL(BC)=covariate shift, HIL(2505.12619)이 "tracking 단독 실패" 명시 |
| **②-2 (frozen latent + RL → OOD를 latent에 새 skill mapping)** | ⚠️ 조건부(원안 핵심 가정 기각, residual+self-imitation으로 재설계) | **순수 latent re-mapping을 보증한 선행연구 없음**(조사C 결론). 최근접 레퍼런스(2505.16084)는 **action-space residual** 사용 + 사용자 OOD(gap/crawl/jump) **미해결로 명시** | OOD가 latent coverage 밖이면 RL이 새 skill을 "발명"하지 못함(PULSE 경고) |
| **D. 제약 적합성** | ❌ 다수 충돌 | latent-action 치환(제약#4)·obs 팽창(제약#2)과 정면충돌 가능. 지형 모션 데이터 **현재 0** | "latent를 policy 입력에 넣기"는 조건부만 허용(아래 §4) |

> 종합 한 줄: **모션 AE + latent prior + RL 골격은 검증된 차용 가능 요소**지만, 사용자 원안의 3개 신규 주장 — (i) 대칭 3종 AE, (ii) AE decoder로 reference 생성, (iii) frozen latent에 OOD가 RL로 자동 mapping — 은 **셋 다 선행연구가 보증하지 않으며**, 사용자가 노리는 OOD(gap/crawl/jump)는 최근접 레퍼런스가 **정확히 미해결로 남긴 영역**이다. 살리려면 §5의 우선순위 수정이 필요.

---

## (1) ① 3종 AE 점검

### 1-1. 모션 AE — ✅ 적절
- 근거: anchor(2507.00677, Go2 실기 vMF-VAE 18D), PULSE(2310.04582, cVAE 32D), Lifelike(2308.15143, VQ-VAE 이산)가 모두 "모션 prior를 latent로 압축 후 재사용"을 실증. 평지 walk/run/trot/pace 데이터 풍부(04 §3, smr_mirror_pkl 18개) → 학습 가능.
- 단, **latent 분포 선택**(구면 vMF / Gaussian+learnable prior / VQ 이산)은 미결정. 사족 steerable 실적 최다 = 구면(anchor). OOD를 "새 코드 추가"로 다루려면 VQ가 운영상 유리(Lifelike). → 설계자가 택1.

### 1-2. 지형 AE(decoder 포함) — ❌ 부적절(과잉)
- **무엇이 과잉인가**: 지형 AE의 decoder는 "latent → heightmap 복원"을 한다. 그러나 본 파이프라인 어디에서도 지형을 *생성/복원*할 필요가 없다 — 지형은 RL/배포 시 RayCaster로 **주어지는 입력**(04 §1.2, scan 187-dim). 선행연구(2505.16084, 2306.14874, HIL)는 모두 지형을 **encoder로만**(feature 추출) 사용, 지형 재구성 decoder를 둔 사례 없음.
- 코드 현실: `actor_critic_parkour.py`에 `scan_encoder(187→32)`가 **이미 존재**(04 §2.2). 별도 지형 AE는 이 encoder의 중복.
- 유일한 정당화 가능 시나리오 = **joint AE generation(②-1)에서 "지형 latent를 조건으로 모션 decode"**할 때 지형 측 encoder가 필요. 이때도 필요한 것은 *encoder*이지 *지형 reconstruction decoder*가 아니다.
- **판정**: 지형 **encoder** = 이미 있음(추가 불요). 지형 **AE(decoder)** = 부적절, 제거 권고.

### 1-3. (지형,모션) joint AE(대칭 3종) — ⚠️ 조건부, 재설계 필요
- **신규성은 인정되나 직접 선례 부재**(02 §8-E, 01 §7-A 모두 동의). 문헌의 실제 수렴점은 대칭 joint AE가 아니라 **비대칭 conditional 구조**:
  - Motion Priors Reimagined(2505.16084): 모션만 latent, **지형은 high-level RL의 obs**(완전 분리).
  - PULSE(2310.04582): proprio-**conditioned** learnable prior R(z|s^p) — 모션 latent을 상태에 *조건화*(지형으로 확장 가능).
  - CALM(2305.02195): conditional encoder로 latent을 명령에 조건화.
  → 즉 "지형과 모션을 **대칭적으로** 한 latent에 인코딩"이 아니라 "**모션 latent을 지형에 조건화**"가 정답에 가깝다. 대칭 joint AE는 지형·모션을 같은 위상으로 두지만, 실제로 둘은 비대칭(지형=조건/원인, 모션=결과).
- **데이터 실현가능성(가장 약한 고리)**: 모션 AE는 대량 평지 MoCap으로 학습 가능하나, **(지형,모션) paired 데이터는 수동 제작 jump 소량뿐**(04 §3: 지형 모션 **현재 전무**, 사용자가 앞으로 제작 예정). 대칭 joint manifold를 희소 pair로 학습하면 overfit/mode collapse가 거의 확실. PULSE조차 99.8% coverage를 AMASS 전량으로 달성했음 — 소량 pair로 joint manifold는 비현실적.
- **판정**: 대칭 3종 AE = ❌. **terrain-conditioned motion AE 1종(conditional VAE/prior)로 통합**하면 ⚠️→차용 가능. 3개 → 1개 conditional AE로 축약 권고(모션 AE 본체 + 지형 conditioning encoder).

### ① 종합
> "3종 AE"는 **모션 AE 1종(✅) + 지형 conditioning encoder(이미 있음) = conditional motion AE 1종**으로 축약하는 것이 선행연구·데이터·코드 모두와 정합. 지형 reconstruction decoder와 대칭 joint AE는 근거 없이 복잡도만 키운다.

---

## (2) ②-1 점검 — "AE로 reference dataset 생성 → IL"

"AE를 이용해 reference dataset을 생성"의 **가능한 해석 4가지**와 각 위험:

| # | 해석 | 타당성 | 위험 |
|---|------|--------|------|
| 1 | **joint AE decoder로 새 지형에 모션 합성** (지형 latent 입력 → 모션 decode) | 낮음(원안 추정) | AE는 **재구성** 학습 → 미관측 지형 decode = **외삽**. 물리/접촉 보장 없음 → 비물리 reference. coverage 밖(gap/crawl)일수록 garbage |
| 2 | **latent 보간**(known pair 사이 interpolation으로 densify) | 중간 | convex coverage **내부**만 유효. OOD(gap/crawl) 외삽은 무효. 평지↔jump 보간이 물리적으로 의미있는지 미보장 |
| 3 | **retargeting/복원**(기존 모션을 지형에 정합, 생성 아닌 cleanup) | 높음 | 새 skill 못 만듦. STMR(2404.11557)로 품질↑ 가능하나 ②-1의 "데이터 생성" 취지와 거리 |
| 4 | **MaskedMimic식 inpainting**(지형+부분모션→완성, 2409.14393) | 높음 | 이건 plain AE decoder가 아니라 **RL로 학습된 컨트롤러** → ②-1과 ②-2를 혼동. 별도 학습 부담 |

**비판 핵심**:
1. **순환/외삽 문제**: AE decoder의 출력 품질은 학습 분포 내에서만 보증된다. ②-1의 목적이 "데이터 없는 지형의 reference 생성"이라면 그것은 정의상 **분포 밖 외삽** → AE가 가장 못하는 일. 생성물이 비물리적이면 IL이 그대로 오염된다(04 §4.1·조사C #7의 품질게이트 경고와 동일).
2. **IL(BC) 자체의 구조적 약점**: 정적 생성 데이터셋에 대한 순수 IL은 **covariate shift / compounding error**에 취약(동적 보행 task에서 치명적). HIL(2505.12619)이 명시 — "tracking(≈IL) 단독은 환경 적응 실패, AMP 병행 필요". 즉 ②-1을 *순수 IL*로 두는 것 자체가 선행연구 반례에 부딪힌다.
3. **PULSE 경고**: RL과 supervised(IL)를 한 latent에 섞으면 latent이 noisy(2310.04582). ②-1(생성/IL)과 ②-2(RL)를 **같은 latent/단계로 섞지 말 것**.

**판정**: ⚠️ 조건부. 성립 조건 = (a) 생성은 **coverage 내부 보간(해석 2)으로 제한**, OOD 외삽 금지, (b) 생성물에 **retargeting(STMR) + 물리/품질 게이트**(measure_pronk_cost 재사용, 04 §4.1) 통과분만 IL 타깃, (c) 순수 IL 대신 **tracking + AMP discriminator 병행**(HIL 구조). 조건 미충족 시(특히 "새 지형 decode → 바로 IL") = ❌.

---

## (3) ②-2 점검 — "frozen latent + RL, OOD를 latent에 새 skill로 mapping"

### 3-1. 원안 핵심 가정과 선행연구 대조

원안 가정 = **"고정 latent skill 공간 위에서 high-level RL을 학습하면 데이터 없는 OOD(계단/gap/crawl)가 자동으로 새 skill로 latent에 매핑된다"**.

조사C(03 문서) + 본 task 본문 정독으로 확정한 반증/조건:

1. **순수 latent re-mapping은 어떤 선행연구도 보증하지 않음**(조사C 결론). ASE/CALM/PULSE의 downstream RL은 **기존 skill의 보간/조합**이지 새 skill 발명이 아님.

2. **[본 task 신규 확정] 최근접 레퍼런스 Motion Priors Reimagined(2505.16084)의 residual은 latent가 아니라 ACTION space**.
   - 본문 §3.3 확인: high-level이 **16-dim latent command z_t** *와* **12-dim joint residual a_res**를 *둘 다* 출력. 최종 action = (frozen low-level의 joint target) + a_res.
   - residual penalty: `r_res = w_res · Σ_{i=1}^{12}(a_res_i)²`, **w_res = -0.1**(본문 Eq.3, §4.2.3). penalty 0이면 residual 폭주로 스타일 붕괴, 과대하면 지형 적응 실패 — trade-off 명시.
   - → 즉 OOD 적응을 견인하는 것은 **latent 좌표 탐색이 아니라 action-space residual**. 원안의 "latent에 새 skill mapping"은 이 레퍼런스에서 **action residual로 구현**된다.

3. **[본 task 신규 확정] PULSE(2310.04582)의 residual은 latent space**(`a = D(π_task(z) + μ^p)`, Eq.4). decoder D·prior R 모두 **freeze**, π_task만 PPO 학습.
   - 단 OOD 반례에 대한 03 문서의 "VR tracking에서 latent이 from-scratch보다 성능 저하" 서술은 **본 task 재확인 결과 다소 과장**일 수 있음 → 정확히는 "coverage 경계 task에서 latent이 *이점 없음* + success rate↔tracking precision 트레이드"로 읽힘(Table 2 success 동률). **[가설/부분확인]** — 어느 쪽이든 결론(coverage 밖 OOD에 latent이 도움 안 됨)은 유지되나, "엄격히 더 나쁘다"는 강한 주장은 인용 시 완화 필요.

4. **[본 task 신규 확정·치명] 사용자 OOD 타깃이 최근접 레퍼런스의 미해결 영역과 정확히 일치**.
   - 2505.16084 본문 한계: 성공 = 계단/슬로프/박스/고장애물(success 75~95%). **미해결(future work) = "gaps, stepping stones, overhanging obstacles" + low-level prior가 "jumping, crawling" 미포함**.
   - 즉 사용자가 노리는 **gap / crawl / jump가 바로 이 논문이 못 푼 곳**. residual(action이든 latent든)만으로 이 영역이 풀린다는 증거는 문헌에 **없다**.
   - AdaptNet(2310.00239, 검색 확인): latent injection = "기존 latent에 **offset 생성**"(state-embedding) + network layer 수정 2-tier, **둘 다 신규 adapter 파라미터 학습 필요** → "RL feedback만으로 즉시 새 skill"이 아님을 재확인.

### 3-2. 실패 모드 (학습 시 모니터링)
- **latent 이탈**: high-level이 coverage 밖 latent을 쏘면 frozen decoder가 비정상 모션(PULSE 보고) → KL/범위 제약 + residual clip 필수.
- **prior 오염**: self-imitation으로 OOD 모션을 무검증 추가하면 평지 prior까지 붕괴(조사C #7·04 §4.1·메모리 경고). → 품질 게이트 필수.
- **mode collapse**: 2505.16084·HIL 모두 단일 gait 습관 보고 → 사용자 "자연스러움/다양성" 목표 직접 위협.
- **탐색 빈약**: latent을 평지 데이터로만 채우면 지형 OOD 미충원(DVFB 분석) → terrain curriculum 필수.

### 3-3. 원안을 살리는 최소 수정안
> ②-2를 **"순수 frozen-latent re-mapping"에서 "frozen prior + residual + self-imitation 데이터 확장"으로** 재정의.

1. **frozen terrain-conditioned motion prior**(①에서 축약한 conditional AE). decoder freeze.
2. **high-level RL = latent command + residual**. residual 공간 선택:
   - **action-space residual**(2505.16084) — RMA+DAGGER·env 무재작성과 가장 충돌 적음(joint target에 더하기). **1순위 권고**.
   - latent-space residual(PULSE z+μ) — 단 이는 사실상 action을 latent로 치환 → **제약#4(latent-action 치환 금지)와 충돌 위험**(§4). 비권고.
3. **residual penalty**(w_res≈-0.1)로 자연스러움 보존(2505.16084 Eq.3).
4. **coverage-far OOD(gap/crawl/jump)** = residual로 안 됨 → **self-imitation 루프**(조사C #7): RL 성공 rollout → 품질게이트(measure_pronk_cost) → (지형,모션) pkl 승격 → prior 재학습 → coverage 단계 확장. **이 루프가 원안에서 빠진 가장 중요한 조각**.
5. **terrain curriculum**(parkour 기존 curriculum 재사용) + latent KL/범위 제약.

**판정**: ⚠️ 조건부. 원안의 "RL이 latent에 새 skill 자동 mapping"은 ❌(기각). 위 5점 수정 시 차용 가능. 단 **residual을 latent에 두면 제약#4와 충돌** → action-residual 권고.

---

## (4) 제약/데이터 적합성 점검 (04 §5 hard 제약 8개 대조)

| 제약 | 충돌 여부 | 상세 |
|------|----------|------|
| #1 env 무재작성, additive only | ⚠️ 조건부 | 새 env 생성 금지. **`Go2-ParkourImitation-v0` 위에 additive로** 얹어야(04 §1.1: AMP·flat_env_mask·DiscriminatorLSD 이미 존재). 새 latent env 신설 = 위반 |
| **#2 policy obs 팽창 금지** | ⚠️ **핵심 쟁점** | 아래 별도 분석 |
| #3 contact sensor obs 금지 | ✅ 무관 | 기존 contact_filt 외 추가 안 하면 무관 |
| **#4 latent-ACTION 치환 금지** | ❌ **충돌 위험** | PULSE식 `a=D(z+μ)`는 **action을 latent로 치환** → RMA+DAGGER·env 무재작성과 정면충돌(메모리 명시). ASE/CALM/MCP식 hierarchical latent-action 구조 거부됨. → ②-2를 **action-residual**(2505.16084)로 구현해야 회피 가능 |
| #5 python 직접 실행 금지 | ✅ 무관 | 학습 시 `./isaaclab.sh -p` |
| #6 conda env | ✅ 무관 | isaac-parkour |
| #7 코어 수정 금지 | ✅ 무관 | rsl_rl/env 레벨만 |
| #8 reward clip(min=0) | ⚠️ 주의 | style/latent 신호는 **clip 바깥(AMP additive) 또는 constraint 채널**로(04 §1.2·메모리). latent reward를 task reward에 더하면 floor에 죽음 |

### 4-1. "latent를 정책 입력에 넣기" vs obs 팽창(제약#2) — 명시적 판정
- **위반 케이스**: terrain/motion latent code를 **deploy actor(46-dim)에 새 차원으로 concat** → deploy obs 팽창 → **제약#2 위반**.
- **허용 케이스**(2가지):
  1. latent을 **discriminator(amp_obs)에만** 주입 → policy 입력 아님 → 팽창 비해당(04 §5 예외, §6-A). **가장 안전**.
  2. latent을 **기존 encoder 경로(scan_latent/priv_latent) 내부에 녹여** deploy 시 history로 distill → 46-dim 불변이면 허용(04 §6-B). 단 "deploy actor 입력 차원 불변" 검증 필수.
- **판정**: "latent를 policy 입력에 넣는다"는 **무조건 위반 아님**. 단 (a) 새 concat 차원으로 넣으면 위반, (b) discriminator-only 또는 기존 encoder/history-distill 경로로만 허용. 설계자는 **어느 경로인지 명시**해야 하며, 기본은 **discriminator-side conditioning**(04 §6-A, amp_parkour 조사 권고와 일치).

### 4-2. 데이터 현실 갭 (가장 큰 실무 리스크)
- **지형 모션 = 현재 0**(04 §3: parkour_imitation는 평지 trot/walk 12개뿐, jump/stair/gap **전무**). 사용자가 수동 제작 예정인 jump-paired 데이터도 **소량**.
- → ①(joint/conditional AE), ②-1(생성 source)이 **존재하지 않는 데이터에 의존**. 사용자의 "수동 장애물 + height paired" 방식은 HIL(2505.12619)이 정당화하나, HIL은 YouTube 19클립으로도 빠듯했고 사족·실로봇이 아님.
- **선결 과제**: paired 데이터 제작 → 품질 검증이 ①②의 모든 것에 선행. 데이터 없이는 joint AE도 reference 생성도 불가. 이 갭이 해소되기 전엔 ②-2의 **self-imitation 루프(§3-3 #4)가 데이터 부트스트랩의 유일한 현실 경로**.

---

## (5) 개선사항 우선순위 목록 (최종 프레임워크 설계자용)

**[P0 — 설계 골격, 즉시 반영]**
1. **3종 AE → conditional motion AE 1종으로 축약**. 지형 reconstruction decoder 제거, 대칭 joint AE 폐기. 지형은 **conditioning encoder**로만(PULSE learnable prior / CALM conditioning). 코드 `scan_encoder` 재사용.
2. **②-2를 action-space residual로 구현**(2505.16084: latent command 16d + joint residual 12d + residual penalty w_res≈-0.1). latent-space residual(PULSE)은 **제약#4 충돌**로 비권고.
3. **latent은 discriminator-side(amp_obs)에 conditioning**, policy 입력 팽창 회피(제약#2). 불가피하게 actor에 넣을 땐 기존 encoder/history-distill 경로로 deploy 46-dim 불변 보장.
4. **`Go2-ParkourImitation-v0` 위 additive**로 구축(새 env 금지, 제약#1). flat_env_mask·DiscriminatorLSD/ContDIAYN 재사용(04 §1.3·§2.2·§6).

**[P1 — OOD를 실제로 풀기 위한 필수]**
5. **self-imitation 데이터 확장 루프**(조사C #7) 추가 — 원안에 없던 가장 중요한 조각. RL 성공 rollout → **품질 게이트(measure_pronk_cost)** → (지형,모션) pkl 승격 → prior 재학습. gap/crawl/jump는 residual만으로 안 됨(2505.16084 미해결 영역).
6. **품질 게이트를 ②-1 생성·②-2 self-imitation 양쪽에 의무화**. 게이트 없으면 prior 오염(메모리·04 §4.1 반복 경고).
7. **terrain curriculum + latent KL/범위 제약 + residual clip**(latent 이탈·mode collapse 방지).

**[P2 — 품질/검증]**
8. **②-1을 순수 IL로 두지 말 것** → tracking + AMP 병행(HIL 2505.12619). 생성은 coverage 내부 보간으로 제한, retargeting(STMR 2404.11557) 후처리.
9. **분리형 baseline 우선 수립 후 joint/conditional의 이득을 ablation으로 입증**(01 §7-A 권고). "conditional이 분리형보다 낫다"는 미입증 가설.
10. mode collapse 모니터링(fore-hind 협응, PSI 2505.12619), 자연스러움 지표화.

---

## (6) 미해결 질문 (사용자 확인 필요)

1. **②-1 "AE로 reference 생성"의 정확한 의도** = §2 해석 1~4 중 무엇? (특히 "미관측 지형에 모션을 decode로 새로 만든다"는 의미인지 / "기존 모션을 지형에 맞춰 정제"인지) — 위험도가 극과 극이라 설계 분기.
2. **latent을 policy 입력에 넣을 생각인지, discriminator에만 넣을 생각인지** — 제약#2/#4 충돌 여부가 여기서 갈림.
3. **residual을 latent space(PULSE)로 둘 의향인지** — 그렇다면 제약#4(latent-action 치환 금지)와의 충돌을 사용자가 수용하는지 확인 필요(메모리상 과거 거부 이력).
4. **수동 제작 예정 (지형,모션) paired 데이터의 규모/종류** — joint/conditional AE 학습 가능성의 결정 변수. jump 외 stair/gap/crawl도 제작하는지.
5. **OOD 목표 우선순위** — gap/crawl/jump 중 무엇이 1순위? (최근접 레퍼런스가 셋 다 미해결 → 가장 가치 높은 1개에 self-imitation 집중 권고)

---

### 검증 기록 (정직성)
- **확인(본문 정독)**: Motion Priors Reimagined 2505.16084v2 — residual=action space(12d joint)+latent command(16d), w_res=-0.1, frozen decoder, 미해결=gap/stepping-stone/overhang/jump/crawl. PULSE 2310.04582v2 — residual=latent(z+μ), decoder+prior freeze. AdaptNet 2310.00239 — latent offset+layer 수정, 신규 adapter 파라미터 필요(검색 스니펫·Roblox 연구페이지).
- **부분확인/가설**: PULSE "VR tracking이 from-scratch보다 나쁘다"는 03 문서 서술 → 본 task 재확인 시 "이점 없음+precision 트레이드"로 완화됨(WebFetch 요약 기반, 원문 Table 재대조 권고). **[가설]** 표기.
- **미확인**: MDPI Terrain-Conditional AMP(2076-3417/16/7/3448) — 403 Forbidden 재발생, 본문 미확인 → terrain-conditioned discriminator의 **개념 방향성만** 차용, 구체 아키텍처/Go2적용/실로봇 검증은 인용 금지.

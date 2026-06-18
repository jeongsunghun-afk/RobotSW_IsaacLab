# SPO 구현 충실도 검증 — 논문 + GitHub vs 우리 코드

**작성일**: 2026-06-09
**대상**: `surrogate_type=="spo"` 분기 (`ppo_parkour.py`, `ppo.py`) + `Go2ParkourSPOPPORunnerCfg`
**레퍼런스**: Xie et al., *Simple Policy Optimization*, arXiv:2401.16025 (v9, 9개 버전) / GitHub `MyRepositories-hub/Simple-Policy-Optimization`

---

## 최종 판정: **PASS** (surrogate 구현은 논문·코드에 충실)

우리 SPO surrogate는 **논문 원문 Eq.16/Eq.18 + Algorithm 1 line-13, 그리고 GitHub 레퍼런스 코드와 모두 verbatim 일치**한다 (부호·계수·|A|·2ε·(r−1)²·전체 −mean 전부). 1차 출처(PDF 직접 판독)로 확정.
iter~2636 surrogate spike→발산은 **구현 버그가 아니라**, (a) SPO quadratic penalty의 구조적 특성(unbounded·non-saturating) + (b) parkour 도메인(heavy-tailed |A|: 많은 reward 항·RMA priv_reg·time-out bootstrap) + (c) 일부 하이퍼파라미터/프로토콜 차이가 결합된 **세팅/도메인 발산**이다.

세부 판정 (차원별):
- **PASS** — surrogate 식이 논문(Eq.16/18/Alg1)·GitHub 코드와 verbatim 일치.
- **PASS** — 코드 버그 아님 (부호/계수/식 오류 없음).
- **세팅 차이** — 발산은 도메인(heavy-tailed |A|) + 우리가 약화/변경한 stabilizer(max_grad_norm 1.0 vs 0.5, global vs per-minibatch adv-norm, ε 0.4 vs 0.2)로 설명.

---

## 1. Surrogate 식 — verbatim 대조 (PASS)

### GitHub 레퍼런스 (`mujoco/trainer.py`, `compute_policy_loss`)
```python
policy_loss = -(mb_advantages * ratios
                - torch.abs(mb_advantages) * torch.pow(ratios - 1, 2) / (2 * self.args.epsilon)).mean()
```

### 우리 코드 (`ppo_parkour.py:364-368`, `ppo.py:321-325` — 동일)
```python
if self.surrogate_type == "spo":
    surrogate_obj = ratio * adv - torch.abs(adv) * torch.square(ratio - 1.0) / (2.0 * self.spo_epsilon)
    surrogate_loss = -surrogate_obj.mean()
```

| 항 | GitHub | 우리 | 일치 |
|----|--------|------|------|
| advantage-weighted ratio | `mb_advantages * ratios` | `ratio * adv` | ✅ |
| penalty 부호 | `−` | `−` | ✅ |
| `|A|` factor | `torch.abs(mb_advantages)` | `torch.abs(adv)` | ✅ |
| `(r−1)²` | `torch.pow(ratios - 1, 2)` | `torch.square(ratio - 1.0)` | ✅ |
| `2ε` 분모 | `/ (2 * epsilon)` | `/ (2.0 * spo_epsilon)` | ✅ |
| 전체 부호 (maximize→minimize) | `-( ... ).mean()` | `-( ... ).mean()` | ✅ |

`ratio = exp(logp_new − logp_old)` 계산도 양쪽 동일(`log_ratio.exp()`). **부호/계수/2ε/|A| 전부 정확히 일치.**

### 논문과의 정합성 — 원문(PDF v9) 1차 출처로 확정
arXiv PDF를 직접 읽어 식을 verbatim 확인했다 (fast-model 추출 아님):

- **Eq.16** (p.5): `J(θ) = E_{(s,a)~π_old}{ r_t(θ)·Â_t − (|Â_t|/2ε)·[r_t(θ)−1]² }` → 우리 식과 **정확히 일치**. trust-region knob = **ε** (우리 `spo_epsilon`와 동일 기호·의미).
- **Eq.18** (p.6): `f_spo = rA − (|A|/2ε)·(r−1)²` → **정확히 일치**.
- **Algorithm 1, line 13-14** (p.5): 주석 *"Compute policy loss L_p **(This is the only difference between SPO and PPO)**"* + `L_p ← −(1/N)Σ{ (π_θ/π_old)Â − (|Â|/2ε)[(π_θ/π_old)−1]² }`. → 우리 `-( … ).mean()`(=−1/N·Σ)와 **부호·구조 완전 일치**. 논문 주석이 명시: **policy loss 외에는 PPO와 동일.**
- (해소된 혼선) ar5iv 자동추출이 반환한 KL/TV-clip 식(Eq.9~12, Thm 4.3 `O·(d_clip/d+σ−1)·σ`)은 SPO를 동기화하는 **이론적 유도(TV/KL lower bound)**일 뿐, **구현 objective는 Eq.16/18의 quadratic 형태**다. 둘 다 같은 논문에 존재하며 모순 없음. ar5iv의 "no (r−1)² term"은 **false negative**.
- **Definition 5.1 + Theorem 5.2 (p.6)**: 논문은 f_spo가 *"ε-aligned"*(미분가능·convex in r, 최댓값 `r* = 1+sign(A)·ε`)임을 증명. 우리도 `∂f/∂r = A − |A|(r−1)/ε = 0 → r* = 1+ε·sign(A)`로 동일 확인. **최적점이 PPO clip 경계 1±ε에 위치하는 smooth surrogate** — clip의 미분가능 대체가 설계 의도(Fig.3: PPO는 경계 밖 gradient=0, SPO는 모든 점이 경계로 향하는 gradient 보유).

---

## 2. Surrogate 외 항목 — 우리가 빠뜨린/다르게 한 것

### 2-1. KL-adaptive LR — 빠뜨린 것 **아님** (오해 해소)
GitHub 레퍼런스의 KL-adaptive lr는 **PPO 전용으로 게이트**되어 있다:
```python
if self.args.adaptive_learning_rate and self.args.algo == 'ppo':   # ← SPO는 제외
    if kl.mean() > desired_kl * 2.0: lr = max(1e-5, lr/1.5)
    if kl.mean() < desired_kl / 2.0: lr = min(1e-2, lr*1.5)
```
→ **레퍼런스 SPO는 adaptive lr를 쓰지 않는다.** 우리 SPO screen이 `schedule="fixed"`로 둔 것은 **레퍼런스와 일치(충실)**. (base parkour PPO는 adaptive지만 SPO cfg에서 fixed로 덮음 — 올바름.)

### 2-2. 누락/약화된 안정화 장치 (실재하는 차이)
| 장치 | 레퍼런스(GitHub mujoco) | 논문 Table 1 | 우리 SPO screen | 평가 |
|------|------|------|------|------|
| advantage norm | **per-minibatch** (`advantage_normalization=True`) | (명시X) | **global/rollout** (`normalize_advantage_per_mini_batch=False`) | 둘 다 ON, **방식 차이**. per-minibatch는 minibatch별 \|A\| 분포를 재정규화→SPO penalty 입력을 더 균질하게. 우리 global은 outlier \|A\|가 minibatch에 남음 → SPO에 불리할 수 있음 |
| max_grad_norm | **0.5** | (명시X) | **1.0** | 우리가 **2배 느슨** → surrogate spike가 grad로 더 크게 전파 가능. 약화된 stabilizer |
| ε (penalty) | 0.2 | 0.2 (PPO clip)/d_max 0.02 | **0.4** | 우리가 더 큼 → (1/ε) spike 크기는↓ 이지만 trust-region은 더 **느슨**. 양날 |
| num_epochs | 10 | 8 | **2** | 우리가 더 **보수적**(ratio drift↓). screen이 의도적으로 줄임 — 안정화 방향 ✅ |
| entropy c₂ | 0.0 | 0.01 | 0.01 | 논문과 일치 ✅ |
| value c₁ | 0.5 | 1.0 | 1.0 | 논문과 일치 ✅ |
| lr anneal | True(linear) | linear decay | 없음(fixed) | 후반 미세 안정화 부재(영향 작음) |
| lr | 3e-4 | 2.5e-4 | 2e-4 | 유사 |

### 2-3. 도메인/구조 차이 (레퍼런스에 없는 우리 추가물 — surrogate 외)
레퍼런스 SPO(MuJoCo/Atari, 8 envs·256 steps·minibatch 256)에는 없고 우리 parkour에만 있는 것:
- **time-out bootstrapping** (`process_env_step`): advantage 분포에 영향
- **priv_reg_loss** 를 total loss에 가산 (RMA adaptation)
- **estimator / dagger(update_dagger)** 분리 학습
- **수십 개 reward 항** → advantage 절댓값 \|A\|이 heavy-tailed (큰 outlier)
- **대규모 batch** (4096×24=98304, 4 minibatch=24576/batch) vs 레퍼런스 256

이들은 surrogate 식과 무관하지만 **\|A\| 분포를 레퍼런스 검증 영역 밖으로 끌고 간다.**

---

## 3. "논문은 SPO가 더 안정적" vs "우리는 SPO만 발산" — 메커니즘 규명

### 핵심: SPO penalty는 **무한(unbounded)**, PPO clip은 **gradient 포화(saturating)**
- **PPO clip**: ratio가 `1±ε`를 벗어나면 해당 샘플의 surrogate gradient = **0** (clamp 평탄부). 큰 \|A\| outlier가 ratio를 밀어도 **하드 브레이크**가 걸림.
- **SPO quadratic**: penalty gradient = `−|A|(r−1)/ε`. (r−1)에 **선형 증가**, \|A\|에 **비례**, 포화 없음. surrogate 값 자체도 `|A|(r−1)²/2ε`로 **2차 발산**.

→ parkour처럼 **\|A\|이 크고 heavy-tailed**한 환경에서, ratio가 살짝만 drift해도 소수 outlier 샘플이 거대한 surrogate spike를 만든다. 이것이 iter~2636 spike→발산의 **구조적 원인**. PPO baseline이 안정적인 이유도 같은 동전의 양면 — clip이 그 outlier들의 gradient를 0으로 죽인다.

### 논문 안정성 주장의 정확한 전제 (원문 §6)
논문의 "더 안정적" 주장은 **구체적 조건부**다:
- **헤드라인은 "deep network scaling"** (§6.2, Fig.5, Table 1): policy network를 3→7 layer로 깊게 하면 *"the performance of PPO collapses in most environments, with uncontrollable probability ratio deviations. In contrast, SPO... constrains the probability ratio deviation effectively."* Table 1 ratio deviation: SPO 0.07~0.19 < PPO 0.16~0.23.
- **실험 프로토콜** (§6.1): *"the only modification in SPO is replacing the PPO's objective with (16), **no further code-level tuning is applied to SPO**."* 즉 **"policy loss만 바꾸고 나머지는 PPO와 동일하게 두라"**가 논문 처방. Algorithm 1 line-13도 동일 명시.
- 검증 도메인은 표준 MuJoCo-v4 / Atari (정규화 advantage, moderate reward scale, moderate \|A\|).

**우리 발산과의 차이 두 갈래:**
1. **\|A\| 분포**: parkour의 수십 개 reward 항 + RMA priv_reg + time-out bootstrap → **heavy-tailed advantage**. 논문이 검증한 well-behaved \|A\| 영역 밖. unbounded penalty `|A|(r−1)²/2ε`는 well-behaved \|A\|에선 ratio를 가두지만(논문 Table 1), heavy-tailed \|A\|에선 outlier가 **surrogate spike 생성기**로 뒤집힘 (iter~2636).
2. **프로토콜 위반(의도적)**: 논문은 "policy loss만 교체"를 처방하나, 우리 SPO screen은 그 외에도 schedule(adaptive→fixed)·epochs(5→2)·ε(0.2→0.4)를 바꿨다. 단 이는 **초기 발산(iter~2636, minimal-swap에 가까운 세팅) 이후의 안정화 시도**이지 발산 원인이 아니다 (메모리 실측: epochs=2는 발산을 ~2636→~3100으로 **지연만**).

---

## 4. 결론 및 권고

### 판정: **PASS — surrogate 구현은 논문·GitHub에 충실 (버그 아님).**
발산은 **(b) 세팅/도메인 차이**이지 (a) 구현 버그가 아니다. 코드 라인에 부호·계수·식 오류 없음.

### 발산을 줄이려면 (코드 수정 아닌 cfg/세팅 권고만)
레퍼런스가 쓰는데 우리가 약화/누락한 stabilizer를 레퍼런스 쪽으로 되돌리는 방향:
1. **max_grad_norm 1.0 → 0.5** (레퍼런스 값). spike의 grad 전파를 직접 차단. 가장 저비용·고효과 후보.
2. **advantage normalization을 per-minibatch로** (`normalize_advantage_per_mini_batch=True`). minibatch별 \|A\| 재정규화로 SPO penalty 입력의 outlier 완화 — 레퍼런스와 정합.
3. **ε** (trade-off 정밀히): penalty 계수는 `1/2ε`. ε↓(0.4→0.2)는 **stiffer spring** — 최적점 `r*=1+ε·sign(A)`을 1에 더 가깝게 당겨 ratio drift를 더 좁게 묶지만(논문 안정화 메커니즘), drift 발생 시 gradient `|A|(r−1)/ε`와 penalty 값이 **더 커져 overshoot spike를 증폭**. 따라서 **ε=0.2 복귀는 무조건이 아니라 1번(grad_norm 0.5)이 step overshoot를 잡은 *뒤에만* 안전**. 순서: ① grad_norm 0.5 → ② per-minibatch adv-norm → ③ 그 다음 ε=0.2 재시도.
4. (선택) reward scale 점검 — \|A\| heavy-tail 자체를 줄이면 unbounded penalty 리스크 감소.

> 단, 이는 **세팅 튜닝 권고**일 뿐 구현 정합성과 무관. SPO 식 자체는 손대지 말 것(논문·코드 verbatim).

---

## 부록: 근거 출처
- 논문: arXiv:2401.16025 (Xie, Zhang, Yang, Hutter, Xu). 9 versions(v1 2024-01-29 ~ v9 2025-07-26). **PDF 직접 판독**: Eq.16 `J(θ)=E{r·Â−(|Â|/2ε)[r−1]²}`, Eq.18 `f_spo=rA−(|A|/2ε)(r−1)²`, Algorithm 1 line-13 *"This is the only difference between SPO and PPO"*, Def 5.1/Thm 5.2 (f_spo ε-aligned, `r*=1+sign(A)ε`), §6.1 *"only modification... no further code-level tuning"*, §6.2/Fig.5/Table 1 (deep-network scaling; SPO ratio deviation 0.07~0.19 < PPO 0.16~0.23). 이론적 유도(별개): Eq.9~12 TV/KL lower bound.
- 논문 Table 1(추출): lr 2.5e-4, epochs 8, minibatch 256, GAE λ 0.95, γ 0.99, c₁ 1, c₂ 0.01, d_max(SPO) 0.02, PPO clip ε 0.2.
- GitHub `mujoco/trainer.py`: policy_loss verbatim(위 §1), KL-adaptive lr는 `algo=='ppo'` 게이트, advantage_normalization per-minibatch, clip_grad_norm_(max_grad_norm).
- GitHub `mujoco/main.py` 기본값: epsilon 0.2, update_epochs 10, mini_batches 4, lr 3e-4, c₁ 0.5, c₂ 0.0, advantage_normalization True, desired_kl 0.01, anneal_lr True, max_grad_norm 0.5, num_envs 8, num_steps 256.
- 우리 코드: `ppo_parkour.py:364-368`, `ppo.py:321-325`; cfg `Go2ParkourSPOPPORunnerCfg`(spo_epsilon 0.4, schedule fixed, num_learning_epochs 2; 상속: entropy 0.01, lr 2e-4, num_mini_batches 4, max_grad_norm 1.0, clip_param 0.2, normalize_advantage_per_mini_batch False).

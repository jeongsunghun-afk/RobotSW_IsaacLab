# WASABI (Wasserstein Adversarial Motion Prior) — IsaacLab/rsl_rl 통합 구현 명세서

> 작성: 2026-06-08 · 상태: **명세 only (코드 미수정)**
> 목적: 현재 AMP discriminator의 LS-GAN(MSE)/BCE loss를 **Wasserstein critic**(WGAN-GP)으로 교체할 수 있도록 `disc_loss_type="wgan"` opt-in 경로를 추가하는 정확한 구현 지시서.

---

## 0. 핵심 결론 (TL;DR)

| 항목 | 결과 |
|------|------|
| 논문 검증 | arXiv:2206.11693 **실재·정확**. WGAN critic loss = `E[D(π)] - E[D(M)]`, GP는 **reference(expert) only, coef 5.0**, reward = `(D - μ̂)/σ̂` (zero-mean normalized). |
| 출력 layer 변경 | **불필요**. 현재 `MLP(input_dim, 1, hidden_dims, activation)`은 `last_activation=None` → 마지막 `nn.Linear(…,1)`이 이미 unbounded linear. |
| 코드 delta | loss 분기 3줄(×2 복사본) + reward 분기 1개 + reward normalizer 1개. GP 인프라 **재사용**. |
| 새 stateful 컴포넌트 | reward normalizer(`EmpiricalNormalization(1)`) — 유일한 신규 학습 버퍼. checkpoint 영속됨. |
| **최대 통합 리스크** | **WGAN reward는 signed(부호 있음)**. LS-GAN/BCE는 항상 ≥0이었음. lerp fusion 및 parkour `total_reward.clip(min=0)`과 충돌 가능. |
| 난이도 | **low-to-moderate** (이전 "moderate" 추정에서 하향). |
| deploy 영향 | **없음**. discriminator는 학습 전용, deploy는 policy만. |

> **출처 주의**: WASABI 본 repo(`martius-lab/wasabi`)의 `learning/` 디렉토리는 discriminator 파일을 깔끔히 노출하지 않았다. 코드 패턴 검증은 **동일 연구실·동일 Wasserstein 정식화**인 후속작 **CASSI**(`martius-lab/cassi`, ICRA 2023)에서 확보했고, 논문 Eq 2/3/4와 상호 일치함을 확인했다. 아래 코드 인용은 CASSI 기준이며 논문 수식과 corroborate된다.

---

## 1. 논문/코드 검증

### 1.1 논문 (arXiv:2206.11693, Li et al., CoRL 2022 best-paper finalist)

제목: *Learning Agile Skills via Adversarial Imitation of Rough Partial Demonstrations*. 인용 실재·정확 확인.

검증된 핵심 수식 (ar5iv HTML 본문에서 verbatim 추출):

- **Critic(discriminator) loss — Eq 2**
  ```
  arg min_D  − E_{d_M}[ D(o, o') ]  +  E_{d_π}[ D(Φ(s), Φ(s')) ]
  ```
  즉 reference(M) transition에 대한 critic 출력은 **올리고**, policy(π) transition은 **내린다**. 이것이 Wasserstein-1 거리의 Kantorovich-Rubinstein 쌍대형. `Φ`는 partial state mapping(아래 1.3).

- **Gradient penalty — Eq 3**
  ```
  w_GP · E_{d_M}[ ‖ ∇_Ω D(Ω)|_{Ω=o^H} ‖²₂ ]      with  w_GP = 5.0
  ```
  **reference(expert) 샘플에만** 적용하는 **zero-centered** R1-style penalty (`‖∇D‖²`를 직접 최소화). interpolation 기반 classic WGAN-GP가 **아님**.

- **Imitation reward — Eq 4**
  ```
  r_I = ( D(Φ(s^H)) − μ̂ ) / σ̂
  ```
  critic 출력을 running mean `μ̂` / running std `σ̂`로 정규화 → **zero-mean, unit-variance** reward. **결과적으로 약 절반이 음수.**

- **출력 activation**: discriminator는 ReLU MLP(Table S7)이나 본문 §3.3에서 "unbounded discriminator output" 명시 → **마지막 layer linear(무활성)**.

- **총 reward 합성**: `r = w_I·(r_I + r_T) + r_R` (r_T=task, r_R=regularization). `w_I`는 모션별 grid-search(Table S6, 0.1–8.0).

### 1.2 코드 (CASSI `learning/algorithms/cassi.py`, `learning/modules/discriminator.py`)

discriminator loss 분기 (verbatim):
```python
if self.discriminator_loss_function == "BCEWithLogitsLoss":
    expert_loss = torch.nn.BCEWithLogitsLoss()(expert_d, torch.ones_like(expert_d))
    policy_loss = torch.nn.BCEWithLogitsLoss()(policy_d, torch.zeros_like(policy_d))
elif self.discriminator_loss_function == "MSELoss":                      # == 우리의 ls_gan
    expert_loss = torch.nn.MSELoss()(expert_d, torch.ones(...,  device=...))
    policy_loss = torch.nn.MSELoss()(policy_d, -1 * torch.ones(..., device=...))
elif self.discriminator_loss_function == "WassersteinLoss":              # == 추가할 wgan
    expert_loss = -expert_d.mean()
    policy_loss =  policy_d.mean()
# 공통:  cassi_loss = 0.5 * (expert_loss + policy_loss)
```
→ **우리 코드(`ppo_amp.py`)의 LS-GAN/BCE 분기와 1:1 동형**. WGAN은 `elif` 한 가지만 추가하면 됨.

출력 layer (discriminator.py): `discriminator_layers.append(nn.Linear(self.shape[-1], 1))` — 무활성 linear.

grad pen (discriminator.py):
```python
def compute_grad_pen(self, expert_state_buf, lambda_=10):
    ...
    grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()   # zero-centered, expert only
```

reward 매핑(`predict_cassi_reward`): quadratic(LS-GAN형 `clamp(1-¼(d-1)², min=0)`) / log(BCE형) / **Wasserstein**(`reward_normalizer.normalize()` 적용 + 갱신). → 논문 Eq 4 = CASSI의 reward_normalizer 동작과 일치.

### 1.3 partial demonstration 처리

reference 데이터는 base 정보만 포함(`v, ω, base height z` 등 local velocities) — joint 데이터 없이 손으로 흔든(hand-held) human demo 가능. policy state `s`를 동일 부분공간으로 사상하는 mapping `Φ`로 정렬하여 discriminator 입력을 만든다. **본 통합 명세에서는 `Φ`가 곧 우리 환경의 `get_amp_observations()`가 반환하는 AMP obs 부분집합에 해당**하며, WGAN 전환은 `Φ` 정의에 독립이다(loss/reward만 바꿈). partial-demo는 reference 데이터 설계 문제이지 critic loss 종류와 직교한다.

---

## 2. 코드베이스 매핑 (정확한 파일:라인)

### 2.1 `rsl_rl/rsl_rl/modules/amp_discriminator.py`

- L34–36: `self.trunk = MLP(input_dim, 1, hidden_dims, activation)` — `last_activation` 미지정 → **마지막 layer bare linear**. (검증: `networks/mlp.py` L66–78, `last_activation_mod is None`이면 activation 미추가.) → **WGAN용 출력 변경 불필요.**
- L49–72 `compute_amp_reward(amp_obs)`: 현재 분기
  - `bce`(L62–67): `-log(1 - sigmoid(logit))` → 항상 ≥0.
  - else(`ls_gan`, L68–70): `clamp(1 - 0.25*(d-1)², min=0)` → 항상 ≥0.
  - L72: `return reward.squeeze(-1) * self.amp_reward_coef`.
  - **여기에 `wgan` 분기 추가** (§3.3).
- L74–77 `get_logits`: 정규화 후 `self.trunk(norm_obs)` 반환 — WGAN critic 값도 이 함수로 그대로 얻음(변경 없음).
- L29 `self.amp_reward_coef = 1.5` (생성자 default; ppo_amp가 cfg값으로 덮어씀).
- L31–32 `amp_obs_normalizer = EmpiricalNormalization(input_dim)` — **reward normalizer 추가 시 동일 클래스 `shape=1`로 신설** (§3.4).

### 2.2 `rsl_rl/rsl_rl/algorithms/ppo_amp.py`

> **주의: `update_amp`가 두 클래스에 복제되어 있다 — `PPOAMPBase`(L93–183)와 `PPOAMP`(L285–376). 두 곳 모두 동일하게 수정해야 한다.**

- loss 분기: `PPOAMPBase` L104–113 / `PPOAMP` L296–305 — 현재 `bce` vs `ls_gan`(else). **여기에 `wgan` 분기 추가** (§3.2).
- GP: L115–140 / L307–332 — expert+policy **양쪽**에 zero-centered `‖∇‖²`, `grad_penalty = 0.5 * coef * (expert + policy)`, default `amp_gradient_penalty_coef=10.0` (L30/L219). **WGAN에서도 이 인프라 그대로 재사용** (§3.5).
- logit reg: L142–149 / L334–341 — WGAN과 무관, 기존 그대로(WGAN에서는 logit_reg=0 권장).
- total: L151 / L343 `0.5*(expert_loss+policy_loss) + grad_penalty + logit_reg` — WGAN에서도 이 결합식 **그대로** (WassersteinLoss가 expert_loss/policy_loss 값만 바꿈).
- output monitoring L162–170 / L355–363: `bce`면 sigmoid, else raw logit. **wgan은 raw critic값을 그대로 로깅**(별도 분기 또는 else 재사용; sigmoid 적용 금지).
- cfg 파싱: L36/L225 `disc_loss_type`, L43/L232 `disc_reward_type` — `"wgan"` 허용값 추가(§3.6).

### 2.3 caller — `rsl_rl/rsl_rl/runners/on_policy_runner_amp.py`

- L114 / L328: `amp_reward = self.alg.discriminator.compute_amp_reward(corrected_amp_obs).detach()` — reward 추출부. WGAN이면 여기서 normalizer **갱신 타이밍** 고려(§3.4 옵션).
- L119–120 / L331: **lerp fusion** `total_reward = lerp*rewards + (1-lerp)*amp_reward`. → WGAN의 signed reward가 여기로 들어감(§4 리스크).
- L178 / L388: `update_amp(expert_pool[batch_ids], policy_pool[batch_ids])` — disc 학습 호출(변경 불필요).
- 동일 패턴: `on_policy_runner_parkour_amp.py` L140(reward), L199(update). parkour AMP 경로도 동일하게 영향.

---

## 3. 구현 명세 (산출물 핵심)

### 3.1 출력 layer 변경 — **없음**

현재 trunk = `MLP(input_dim, 1, hidden_dims, activation, last_activation=None)` → 마지막이 무활성 `nn.Linear(…,1)`. LS-GAN 경로가 이미 raw logit을 `(d-1)²`에 직접 먹이고, BCE 경로가 자체 sigmoid를 적용한다는 사실이 곧 출력이 unbounded linear임을 증명. **WGAN critic이 요구하는 unbounded 출력 조건을 코드가 이미 만족.** 변경 0줄.

### 3.2 Loss 분기 추가 (`ppo_amp.py`, 두 복사본)

현재:
```python
if self.disc_loss_type == "bce":
    bce = nn.BCEWithLogitsLoss()
    expert_loss = bce(expert_logits, torch.ones_like(expert_logits))
    policy_loss = bce(policy_logits, torch.zeros_like(policy_logits))
else:  # ls_gan
    expert_loss = nn.MSELoss()(expert_logits, torch.ones_like(expert_logits))
    policy_loss = nn.MSELoss()(policy_logits, -1 * torch.ones_like(policy_logits))
```
추가(WGAN — CASSI `WassersteinLoss`와 verbatim 동치):
```python
elif self.disc_loss_type == "wgan":
    expert_loss = -expert_logits.mean()   # reference critic ↑
    policy_loss =  policy_logits.mean()    # policy critic ↓
```
이후 `total_loss = 0.5*(expert_loss+policy_loss) + grad_penalty + logit_reg` 식은 **그대로**. 수학적으로 `0.5*(−E[D_exp]+E[D_pol])`가 Eq 2의 `E[D(π)]−E[D(M)]`에 0.5 스케일이 곱해진 형태 — 일관, optimizer가 흡수.

### 3.3 Reward 변환 (`amp_discriminator.py compute_amp_reward`)

논문 Eq 4: `r_I = (D − μ̂)/σ̂`. 추가 분기:
```python
elif self.disc_reward_type == "wgan":
    reward = self.reward_normalizer(disc_logits)        # (D - μ̂)/σ̂  (running stats)
    # (학습 중) normalizer 통계 갱신은 호출부 또는 여기서 self.reward_normalizer.update(disc_logits)
```
- `disc_logits` = critic raw 출력. clamp/activation 없음.
- 마지막 `reward.squeeze(-1) * self.amp_reward_coef`는 공통 적용. **단 WGAN reward는 이미 unit-variance이므로 `amp_reward_coef`는 LS-GAN 시절(1.5–2.0)보다 작게 재튜닝 필요**(§4, §5). 시작값 `reward_coef≈0.3–1.0` 권장.
- **부호 보존**: LS-GAN/BCE의 `clamp(min=0)`을 WGAN 경로에 **절대 적용하지 말 것**. signed가 의도.

### 3.4 Reward normalizer (신규 stateful 버퍼 — 유일한 신규 컴포넌트)

- `amp_discriminator.py __init__`에 추가:
  ```python
  self.reward_normalizer = EmpiricalNormalization(1).to(self.device)
  ```
  (이미 `from rsl_rl.networks import ... EmpiricalNormalization` import 되어 있음. `EmpiricalNormalization.forward`는 `(x-μ̂)/(σ̂+eps)` — Eq 4의 eps-안정화 버전, 충분히 동치.)
- **checkpoint 영속**: `EmpiricalNormalization`은 `register_buffer`로 `_mean/_var/_std/count` 보유 → discriminator의 `state_dict()`에 자동 포함, save/load·multi-GPU broadcast(`ppo_amp.py` L185–190/L378–384 `state_dict()` 경로)에 자동 동행. 별도 직렬화 코드 불필요.
- **갱신 타이밍**(택1):
  - (A) `compute_amp_reward` 내부에서 `wgan`일 때 `self.reward_normalizer.update(disc_logits)` 호출. 간단하나 매 step 갱신.
  - (B) runner에서 reward 추출 직후 1회 갱신(L114/L328 인접). collection 빈도 제어 용이. **권장: (A)**, EmpiricalNormalization이 `self.training` 가드로 eval 시 자동 무갱신이라 안전.
- **주의**: `wgan` 경로에서 reward는 `.detach()`되어 reward로만 쓰이므로(runner L114) normalizer 갱신이 critic gradient에 누설되지 않음. 단 `update`는 `forward`와 별도 호출 — `compute_amp_reward`가 `forward`(정규화 출력)와 `update`(통계 갱신)를 모두 호출하도록 구성.

### 3.5 Gradient penalty — **기존 인프라 그대로 재사용** (over-engineering 금지)

- 현재 코드의 GP는 expert+policy 양쪽에 **zero-centered** `‖∇D‖²`(R1형) — 이는 WGAN-GP가 요구하는 **올바른 종류**의 penalty다. classic interpolation WGAN-GP(`(‖∇‖−1)²`)로 **바꾸지 말 것.**
- 논문/CASSI는 **expert-only, coef 5.0**; 우리 코드는 **both-sided, coef 10.0**. 차이를 문서화하되 재작성 불필요.
- **권장**: WGAN 경로에서 `amp_gradient_penalty_coef`를 5.0 부근으로 낮춰 시작(both-sided이므로 effective penalty가 더 큼). tunable로 노출, 코드 분기 불필요.
- (선택적 정밀화) 논문 정확 재현을 원하면 wgan일 때 policy-side GP를 끄는 옵션을 둘 수 있으나, 1차 통합에선 불필요.

### 3.6 cfg 변경

`disc_loss_type` 및 `disc_reward_type`에 `"wgan"` 허용값 추가. 예시 amp_cfg:
```yaml
disc_loss_type: "wgan"          # ls_gan | bce | wgan  (신규)
disc_reward_type: "wgan"        # ls_gan | bce | wgan  (신규; reward 변환 일치 필요)
gradient_penalty_coef: 5.0      # WGAN-GP 필수, both-sided이므로 5.0 시작 (기존 10.0 재사용 가능)
reward_coef: 0.5                # WGAN reward는 unit-var → LS-GAN(1.5~2.0)보다 작게
disc_logit_reg: 0.0             # WGAN에서 logit reg 불필요
task_reward_lerp: 0.5           # 기존; signed reward 주의(§4)
```
> `disc_loss_type`(학습)과 `disc_reward_type`(reward 변환)은 **독립 cfg**다. WGAN 사용 시 **둘 다 `"wgan"`으로 맞춰야** loss와 reward가 정합. (불일치 방지 검증 권장: loss=wgan인데 reward=ls_gan이면 critic 출력에 `(d-1)²`를 먹이는 모순 — assert 또는 경고 추가 고려.)

### 3.7 하위호환

`ls_gan`/`bce` 경로는 **전부 보존**, WGAN은 순수 **opt-in** `elif` 추가. 기존 cfg(분기 키 미지정 시 default `ls_gan`) 동작 불변. reward_normalizer는 wgan일 때만 사용되지만 항상 생성됨(미사용 시 통계 미갱신·checkpoint에 무해한 init 버퍼만 동행).

---

## 4. 검증/실험 계획 + deploy

### 4.1 A/B 실험

| 축 | 비교 | 측정 |
|----|------|------|
| loss 종류 | `ls_gan` vs `wgan` (동 task, 동 seed) | `disc_expert_output`/`disc_policy_output` 발산 여부, `disc_grad_penalty` 안정성, motion imitation 품질(시각·tracking error) |
| GP coef | wgan에서 `{5.0, 10.0}` | critic Lipschitz 안정성 vs 학습속도 |
| reward_coef | wgan에서 `{0.3, 0.5, 1.0}` | task reward와 balance, total_reward 분포 |

- **discriminator 붕괴/발산 진단**: LS-GAN은 `expert→+1/policy→−1` saturate 시 gradient vanish(BCE도 동일 known issue). WGAN critic은 unbounded라 saturate가 없어 gradient 공급이 안정적이어야 함 — 이것이 본 프로젝트 **Go2-Imitation 고속 tracking plateau / 3-leg gait AMP 불안정** 이력에 대한 가설적 처방. 검증 지표: AMP reward 학습 곡선의 collapse(0으로 죽거나 발산) 빈도.
- 본 프로젝트 연결: 최근 커밋 "Port MimicKit AMP settings to fix Go2-Imitation high-speed tracking plateau"는 BCE/logit-reg 튜닝으로 plateau 대응 — WGAN은 동일 문제를 loss 구조 차원에서 접근하는 직교 실험.

### 4.2 deploy 영향 — **없음 (확인)**

discriminator는 **학습 시 reward 생성 전용**. deploy/play는 actor(policy)만 export. `compute_amp_reward`·`update_amp`·`reward_normalizer`는 추론 그래프에 미포함. → **sim-to-real deploy 영향 0**. (CLAUDE.md의 contact-sensor obs 금지 등 deploy 제약과도 무관 — AMP obs는 학습 reward용이며 policy obs와 분리.)

---

## 5. 리스크

1. **[최상위] signed reward × clip/fusion 충돌**
   WGAN reward(Eq 4)는 zero-mean → **약 절반이 음수**. 기존 LS-GAN/BCE는 항상 ≥0이었음.
   - (a) lerp fusion `(1-lerp)*amp_reward`(runner L120/L331)에 음수가 흘러들어 total_reward를 끌어내림 — 정상이나 reward scale 재인식 필요.
   - (b) **parkour `total_reward.clip(min=0)`**(프로젝트 메모리: A env:1108 + B parkour_reward_manager.py:38, **의도된 설계**)이 활성인 경로에서 WGAN을 쓰면 **음수 AMP reward가 통째로 0으로 잘려** 학습 신호의 절반이 소실. → parkour AMP 경로(`on_policy_runner_parkour_amp.py`)에 WGAN 적용 시 **clip(min=0)과의 상호작용을 반드시 사전 검토**. 비-parkour Go2-Imitation 경로 우선 검증 권장.

2. **WGAN 학습 불안정 / GP 민감도**: critic이 Lipschitz 제약을 GP에만 의존 → coef 과소 시 발산, 과대 시 underfit. both-sided GP라 5.0 시작 권장하되 모니터링 필수.

3. **reward scale 재튜닝**: unit-variance reward로 `amp_reward_coef`·`task_reward_lerp`의 의미가 바뀜. 기존 LS-GAN 튜닝값 직접 이식 금지. task reward와의 balance 재탐색 필요.

4. **두 복사본 동기화 누락**: `PPOAMPBase`/`PPOAMP` 양쪽 `update_amp` 동시 수정 필수(한쪽만 고치면 사용 클래스에 따라 silent 미적용).

5. **cfg 불일치**: `disc_loss_type=wgan` ↔ `disc_reward_type` 미일치 시 critic 출력에 LS-GAN reward식 적용되는 모순. assert/경고 권장(§3.6).

---

## 부록 A. 최종 반환 요약

**(a) 논문검증 + LS-GAN 대비 코드 delta**
- arXiv:2206.11693 실재·정확. WGAN loss `E[D(π)]−E[D(M)]`(Eq 2), GP expert-only coef 5.0 zero-centered(Eq 3), reward `(D−μ̂)/σ̂`(Eq 4), 출력 linear. CASSI 코드로 corroborate.
- **코드 delta (LS-GAN→WGAN)**: ① loss — MSE(`d`,±1) → `−E[D_exp]`/`+E[D_pol]` (CASSI `WassersteinLoss` verbatim), `ppo_amp.py` 두 복사본에 `elif "wgan"` 추가. ② reward — `clamp(1−¼(d−1)²,min=0)` → `(D−μ̂)/σ̂` (signed), `amp_discriminator.py`에 분기. ③ **출력 layer 변경 0줄**(이미 unbounded linear). ④ GP **재사용**(coef만 10→5 권장). ⑤ 신규 `reward_normalizer=EmpiricalNormalization(1)` 1개.

**(b) reward 변환 방식**
critic raw 출력 `D`를 running mean/std로 정규화 → zero-mean·unit-var(`EmpiricalNormalization(1)`). clamp 없음(부호 보존). `amp_reward_coef`로 스케일(LS-GAN보다 작게, 0.3–1.0). lerp fusion·`reward_coef` 인프라와 호환되나 signed라 재튜닝·clip 충돌 주의.

**(c) 구현 난이도 재평가**
**low-to-moderate** (당초 moderate→하향). 순수 추가량: loss `elif` 3줄×2, reward `elif` 1개, normalizer 1줄+갱신 1줄, cfg 허용값. GP·출력 layer·state_dict 영속·multi-GPU broadcast 전부 **기존 인프라 무수정 재사용**. 유일한 신규 개념은 reward normalizer와 **signed-reward 통합**(clip/lerp 검토)이며, 이것이 코드량이 아닌 **튜닝·검증 비용**의 대부분.

# WASABI (WGAN-AMP) iter~32k 붕괴 — 레퍼런스 대조 검증

**목적**: 우리 WGAN-AMP 구현이 iter~32k에서 붕괴한 두 진단
((1) critic 절대출력 common-mode 음 drift, anchor 부재 / (2) count-based reward normalizer가 sustained drift 추종 실패)
에 대해, **CASSI/WASABI 레퍼런스가 이 둘을 어떻게 처리하는지**를 검증하고 reference-faithful fix를 확정한다.
코드 수정 없음. 검증·보고만.

근거: WASABI 논문 arXiv:2206.11693 (ar5iv) + CASSI 코드 `martius-lab/cassi`(main) verbatim + 우리 코드 인용.

---

## 핵심 결론 (TL;DR)

| 항목 | 레퍼런스(CASSI/WASABI) | 우리 구현 | 판정 |
|------|------------------------|-----------|------|
| **critic anchor** | **있음** — discriminator optimizer **L2 weight decay**(코드 default `0.0005`, 논문 Table S5 `0.001`). 논문이 명시적으로 "weight regularization이 출력 scale을 제어 → stable imitation reward"라고 credit. | **없음** — `optim.Adam(..., )` weight_decay 미설정 + `disc_logit_reg=0.0`(E[D²] reg 비활성). | **우리가 빠뜨림(deviation).** 이것이 reference-faithful fix. |
| **reward normalizer** | **count-based Welford** running mean/var (`rate = batch/tot_count`, count cap 없음, clip 10.0). 논문: "running mean μ̂, variance σ̂² 유지". EMA 아님. | **count-based Welford** (`EmpiricalNormalization`, `rate = count_x/count`, cap 없음). | **동일. 편차 아님.** EMA로 바꾸면 오히려 novel deviation. |

**한 줄 요약**: drift를 normalizer로 *쫓는* 게 아니라, **critic을 anchor해서 drift 자체를 없애는** 것이 레퍼런스 방식이다.
CASSI도 우리와 똑같은 count-based normalizer를 쓰지만 붕괴하지 않는 이유는 **weight decay anchor**가 critic 출력을 stationary하게 유지하기 때문. 우리는 그 anchor만 빠뜨렸다.

---

## 1. critic anchor — 레퍼런스 유무 + 형태

### 1.1 WASABI 논문 (arXiv:2206.11693)

- **Eq 2 (critic loss)** — anchor 항 **없음**:
  ```
  arg min_D  -E_{d^M}[D(o,o')] + E_{d^π}[D(Φ(s),Φ(s'))]
  ```
- **Eq 3 (full disc objective)** — gradient penalty는 **expert-only**(reference 분포 d^M), `w^D = 0.5`, `w^GP = 5.0`:
  ```
  arg min_D  w^D ( -E_{d^M}[D(o^H)] + E_{d^π}[D(Φ(s^H))] )
             + w^GP · E_{d^M}[ ||∇_Ω D(Ω)|_{Ω=o^H}||²₂ ]
  ```
- **anchor의 실체 = weight decay.** 논문 본문 명시:
  > "we apply **L2 regularization on the discriminator** for the sake of simplicity. In addition, **discriminator weight regularization also controls the scale of its output, which results in stable imitation rewards**."

  Table S5: **discriminator weight decay = 0.001**.
  → anchor는 loss 식 안의 epsilon·E[D²] 항이 **아니라** optimizer의 weight decay로 구현됨.

### 1.2 CASSI 코드 (`martius-lab/cassi`, verbatim)

`learning/algorithms/cassi.py`:
```python
self.discriminator_optimizer = discriminator_optimizer(
    self.discriminator.parameters(),
    lr=self.discriminator_learning_rate,
    momentum=self.discriminator_momentum,
    weight_decay=self.discriminator_weight_decay,   # ← anchor
)
# discriminator_weight_decay default = 0.0005

cassi_loss = 0.5 * (expert_loss + policy_loss)               # w^D = 0.5
grad_pen_loss = self.discriminator.compute_grad_pen(
    sample_cassi_expert,                                     # expert-only
    lambda_=self.discriminator_gradient_penalty_coef)       # default 5
discriminator_loss = cassi_loss + grad_pen_loss
```
`learning/modules/discriminator.py`:
```python
expert_loss = -expert_d.mean()
policy_loss =  policy_d.mean()
grad_pen = lambda_ * (grad.norm(2, dim=1) - 0).pow(2).mean()
```
→ **loss 식 안에는 critic 출력 anchor 항이 없다.** 유일한 anchor는 **optimizer weight_decay(0.0005)**. (논문 0.001 / 코드 0.0005, 둘 다 ~1e-3 order.)

### 1.3 우리 구현 (`rsl_rl/algorithms/ppo_amp.py`)

```python
# PPOAMPBase L65 / PPOAMP L281
self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.amp_discriminator_lr)
#                                                                  ↑ weight_decay 인자 없음 → 0.0
```
```python
# disc_logit_reg default = 0.0  (L75 / L297)
# → 아래 E[D²] 형태의 anchor가 곱해지지만 계수 0이라 비활성
logit_reg = self.disc_logit_reg * (expert_logits.pow(2).mean() + policy_logits.pow(2).mean())
total_loss = 0.5 * (expert_loss + policy_loss) + grad_penalty + logit_reg
```
→ **anchor 두 경로 모두 OFF**: Adam weight_decay 없음 + logit_reg 계수 0.
→ critic 절대레벨을 잡아주는 힘이 0 → common-mode drift가 자유롭게 누적 → iter~32k 붕괴와 정합.
→ **이것이 우리가 빠뜨린 reference-faithful 항.**

#### 1.3.1 실제 붕괴 run config로 재확인 (코드 default 아님)

붕괴한 run은 `go2_imitation/agents/rsl_rl_ppo_cfg.py`의 `Go2ImitationWASABIPPORunnerCfg.__post_init__`:
```python
self.amp["disc_loss_type"]  = "wgan"
self.amp["disc_reward_type"] = "wgan"
self.amp["gradient_penalty_coef"] = 5.0   # both-sided GP
self.amp["reward_coef"] = 0.5
self.amp["disc_logit_reg"] = 0.0          # ← 주석: "WGAN에서 logit reg 불필요"
```
→ base BCE config는 `disc_logit_reg=0.01`로 켜놨지만, **WGAN 변형이 이를 명시적으로 0으로 끔**
("WGAN에서 logit reg 불필요"라는 판단). weight_decay는 어디에도 없음.
→ **실제 run에서 critic 출력 anchor가 두 경로 모두 OFF임이 코드 default가 아니라 run-cfg로 확정됨.**
오히려 WGAN 변형이 유일하게 갖고 있던 출력 L2 anchor(logit_reg)를 "불필요"로 판단해 의도적으로 제거한 것이
CASSI가 weight_decay로 채우는 anchor 공백을 그대로 노출시켰다.

(참고: 우리 GP는 expert+policy **both-sided**, CASSI/논문은 **expert-only**. 우리 코드 주석도 이미 인지하고 `wgan 사용 시 gradient_penalty_coef=5.0 권장`. drift root cause는 아니지만 reference-faithful 하려면 함께 정렬 가능.)

---

## 2. reward normalizer — 레퍼런스 update rule vs 우리 count-based

### 2.1 WASABI 논문 Eq 4
```
r^I = ( D(Φ(s^H)) - μ̂ ) / σ̂
```
> "we normalize the reward to have zero mean and unit variance in the policy training loop by maintaining its **running mean μ̂ and variance σ̂²**."

EMA/momentum 명시 없음 → "running mean/var". 코드로 확인:

### 2.2 CASSI 코드 `learning/modules/normalizer.py` (verbatim)
```python
def __init__(self, input_dim, device, epsilon=1e-2, clip=10.0):
    self.mean  = torch.zeros(input_dim, device=device)
    self.var   = torch.ones(input_dim, device=device)
    self.count = epsilon

def update_from_moments(self, batch_mean, batch_var, batch_count):
    delta = batch_mean - self.mean
    tot_count = self.count + batch_count
    new_mean = self.mean + delta * batch_count / tot_count       # rate = batch/tot_count
    new_var = (self.var*self.count + batch_var*batch_count
               + torch.square(delta)*self.count*batch_count/tot_count) / tot_count
    self.mean, self.var, self.count = new_mean, new_var, tot_count   # count cap 없음
```
디스크리미네이터 경로(`discriminator.py`, wasserstein_mapping):
```python
style_reward = self.reward_normalizer.normalize(d.clone())
self.reward_normalizer.update(d)
```
→ **count-based Welford**, `rate = batch_count / tot_count`, **count cap 없음(무한 누적)**, clip 10.0. **EMA 아님.**

### 2.3 우리 구현 `rsl_rl/networks/normalization.py` — `EmpiricalNormalization`
```python
count_x = x.shape[0]
self.count += count_x
rate = count_x / self.count                                      # = batch/tot_count
delta_mean = mean_x - self._mean
self._mean += rate * delta_mean
self._var  += rate * (var_x - self._var + delta_mean*(mean_x - self._mean))
```
`amp_discriminator.py`:
```python
if self.training:
    self.reward_normalizer.update(disc_logits.detach())
reward = self.reward_normalizer(disc_logits)                     # (D - μ̂)/σ̂
```

### 2.4 대조
**완전히 동일한 알고리즘**(count-based Welford, rate=batch/tot_count, cap 없음).
→ reward normalizer는 **우리 편차가 아니다.** 논문/코드 모두 EMA가 아니라 count-based running stat을 쓴다.
→ **normalizer를 EMA로 바꾸는 것은 reference-faithful가 아니라 novel deviation.**

**왜 CASSI는 같은 normalizer로 안 터지나**: count가 커지면(iter↑) `rate→0`이라 normalizer가 사실상 **frozen**되는 건 CASSI도 동일. 하지만 CASSI는 weight_decay anchor 덕에 **critic 출력 분포가 stationary**라서, frozen된 μ̂/σ̂가 여전히 유효 → 보상 정상.
우리는 anchor가 없어 critic이 drift → frozen normalizer가 비정상 분포를 정규화 → 보상 오염.
**즉 normalizer 문제(진단 2)는 anchor 부재(진단 1)의 *증상*이다. 둘 중 진짜 fix는 anchor.**

---

## 3. reference-faithful fix 정확한 명세

### ✅ FIX (reference-faithful, 적용 권장)
**discriminator optimizer에 L2 weight decay 추가** — CASSI verbatim 방식.

- 파일: `rsl_rl/rsl_rl/algorithms/ppo_amp.py`
- 위치: `PPOAMPBase.__init__` **L65** 와 `PPOAMP.__init__` **L281** (둘 다)
- 변경:
  ```python
  # 현재
  self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.amp_discriminator_lr)
  # → 변경
  self.disc_optimizer = optim.Adam(
      self.discriminator.parameters(),
      lr=self.amp_discriminator_lr,
      weight_decay=amp_cfg.get("disc_weight_decay", 5e-4),   # CASSI code 0.0005 / 논문 0.001
  )
  ```
- 하이퍼파라미터: **`weight_decay = 5e-4`** (CASSI 코드 default). 보수적으로 가려면 논문 Table S5의 **1e-3**. WGAN 분기에서만 켜고 ls_gan/bce는 기존(0) 유지 권장.

> ⚠ **optimizer 차이로 값 그대로 이식 금지(empirical calibration 필요).**
> CASSI의 disc optimizer는 `momentum=` 인자를 받는다 → **Adam이 아니라 SGD/RMSprop 계열**.
> 거기서 `weight_decay`는 표준 decoupled-ish L2다. 우리는 `optim.Adam(weight_decay=)`인데, 이는
> **coupled L2**(gradient에 합산)라 adaptive moment와 상호작용이 달라 anchor 효과가 약하기로 악명 높다(AdamW 문제).
> → CASSI의 `5e-4`가 우리 Adam에 그대로 전이된다는 보장 없음. **값은 실측 calibration 필요**(예: 5e-4에서
> critic 출력 mean이 안정되는지 보고 1e-3까지 상향).
> → Adam 유지 시 **`optim.AdamW(..., weight_decay=...)`** 가 anchor 의도에 더 충실(decoupled).
>   AdamW로 가면 5e-4~1e-2 범위에서 탐색 권장.

> 주: 우리엔 이미 `disc_logit_reg`(E[D²] logit reg) 경로가 있어 이걸 0.05 정도로 켜는 것도 *유효한 anchor*다.
> 다만 **CASSI-faithful한 형태는 weight decay**(파라미터 L2)이고, logit_reg(출력 L2)는 AMP(Peng 2021) 계열 변형이다. reference 충실도 우선이면 weight_decay를 1순위로.

### ❌ NON-FIX (적용 금지 — reference 위반)
**reward normalizer를 EMA로 교체.** 레퍼런스는 우리와 동일한 count-based Welford를 쓴다.
EMA 전환은 레퍼런스에 없는 novel stabilizer이며, drift의 root cause(anchor 부재)를 가리는 우회책일 뿐이다.

### (선택) reference 정렬 추가 항목
- gradient penalty를 **expert-only, coef 5.0**으로(현재 both-sided coef 10.0). drift root cause는 아니지만 논문/CASSI와 정렬.
- normalizer clip: CASSI는 reward normalizer에 clip 10.0. 우리는 reward_normalizer에 clip 미적용(`norm_clip`은 amp_obs_normalizer 전용). 필요 시 reward 경로에도 ±10 clip 추가 가능(2차).

---

## 4. 진단 매핑 정리

- **진단 (1) critic common-mode drift, anchor 부재** → **확인. 우리 deviation.** 레퍼런스는 weight_decay(~1e-3)로 anchor. 우리는 weight_decay=0 + logit_reg=0. **→ 1순위 fix.**
- **진단 (2) count-based normalizer가 drift 추종 실패** → **현상은 맞으나 root cause 아님.** 레퍼런스도 동일 count-based normalizer 사용. normalizer 자체는 편차 아님. (1)을 고치면 critic이 stationary해져 frozen normalizer로도 정상 동작. **→ EMA 교체 불필요/금지.**

**결론**: 단일 reference-faithful fix = **disc optimizer weight_decay=5e-4 (또는 1e-3) 추가.** normalizer는 건드리지 않는다.

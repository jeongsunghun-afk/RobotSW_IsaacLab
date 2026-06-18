# SPO (Simple Policy Optimization) → rsl_rl PPO 통합 구현 명세서

> 상태: **명세만** (코드 미수정). 본 문서는 구현 지시서이며, 실제 적용 전 worker dispatch 필요.
> 작성 근거: arXiv:2401.16025 **원문 PDF 직접 판독**(Algorithm 1 / Eq.16 / Eq.18) + 공개 구현 GitHub 코드 verbatim + 코드베이스 직접 읽기.

---

## 0. 요약 (TL;DR)

- 논문/저자/연도 **검증 완료** — 실재함 (§1.1).
- **SPO surrogate 확정 (PDF 직접 판독):** SPO는 PPO의 ratio-clip을 **quadratic ratio penalty**로 교체한다.
  paper Eq.16/Algorithm-1 line14/Eq.18 (verbatim):
  ```
  f_spo = r·Â  −  (|Â| / 2ε)·(r − 1)²
  ```
  - 논문 본문(Alg.1 line13)에 *"This is the only difference between SPO and PPO"* 라고 명시.
  - **공개 GitHub 구현의 `spo` 분기와 정확히 동일** → 논문↔코드 일치 확인됨.
  - ε는 PPO clip ε와 동일 역할의 trust-region 하이퍼파라미터. `f_spo`는 ε-aligned, 최적 `r* = 1 + sign(Â)·ε` (Eq.18/19).
  - **advantage 부호 분기 불필요** — `|Â|`로 부호 무관 penalty, `r·Â` 항이 부호를 자동 처리.
- **⚠ 정정 (process note):** 초기에 ar5iv HTML WebFetch(소형 요약모델)가 "KL-clip / d_clip·d / advantage 부호 piecewise" 형태를 보고했으나 **PDF 원문 판독 결과 그것은 환각/오독**이었다. 그 piecewise 수식은 SPO 알고리즘이 아니라 §4의 **TV/KL lower-bound 이론 유도(Thm 4.3, Eq.11–14)** 였다. 실제 알고리즘 objective는 Eq.16 quadratic. task 문구("ratio-clip을 KL-clip으로")는 *직관적 설명*이며, 실제 구현은 KL을 직접 쓰지 않는 quadratic penalty다.
- 변경 표면: surrogate 블록 1곳. 단 **`ppo.py`와 `ppo_parkour.py` 두 파일 모두** 손대야 전체 알고리즘 커버됨(상속 아님, 독립 클래스).
- deploy 영향 **없음** (순수 학습 알고리즘 내부). opt-in 플래그로 기존 PPO 동작 100% 보존 가능.
- **난이도: LOW.** KL piecewise 형태였다면 미분가능 KL 계산이 필요해 중간 난이도였으나, 실제 quadratic 형태는 ratio·advantage만 쓰므로 in-scope 변수로 **3~5줄 추가**면 끝. KL=0 division도 없음.

---

## 1. 논문/코드 검증 결과

### 1.1 논문 메타데이터 (검증됨)
- **Title**: *Simple Policy Optimization* — 정확.
- **Authors**: Zhengpeng Xie, Qiang Zhang, Fan Yang, Marco Hutter, Renjing Xu — task가 적은 "Xie/Zhang/Yang/Hutter/Xu"와 **일치**.
- **arXiv**: 2401.16025 (submit 2024-01-29, 최신 rev v9 2025-07-26). 연도 2024 정확.
- Abstract(원문 인용): SPO는 *"a novel unconstrained first-order algorithm. By slightly modifying the policy loss used in PPO... improves upon ratio clipping, offering stronger theoretical properties and better constraining the probability ratio within the trust region."*
- 인용 실재성: **확인됨** (가짜 인용 아님).

### 1.2 SPO 정식 objective (Eq.16 / Eq.18) — **PDF 원문 직접 판독**

논문 §4 Methodology, Eq.16 (verbatim):
```
J(θ) = E_{(s_t,a_t)~π_θold} [ r_t(θ)·Â_t  −  (|Â_t| / 2ε)·[r_t(θ) − 1]² ]
```
Eq.18 (objective class 비교, verbatim):
```
f_ppo = min( r·A,  clip(r, 1−ε, 1+ε)·A )      # 기존 PPO
f_spo = r·A  −  (|A| / 2ε)·(r − 1)²            # SPO
```
- `r = π_θ(a|s)/π_θold(a|s)` (probability ratio), `A = Â` (advantage), ε = hyperparameter (PPO clip ε와 같은 역할/스케일).
- Algorithm 1 line 13 주석(원문): *"Compute policy loss L_p (This is the only difference between SPO and PPO)"* — **policy loss 한 줄만 다름**.
- **ε-aligned (Thm 5.2, Eq.19)**: `∂f_spo/∂r = A − (|A|/ε)(r−1) = 0` → 최적 `r* = 1 + sign(A)·ε`. 즉 PPO clip이 노리는 동일한 r 경계를 **미분가능·convex** 방식으로 달성.
- **trust region 밖 gradient 처리 (PPO 대비 핵심 delta, 논문 Fig.3 / §5.2)**:
  - PPO `f_ppo`는 `r`이 `[1−ε,1+ε]` 밖이면 `clip` 때문에 해당 샘플 gradient가 **0** ("gray circle, 회복 불가").
  - SPO `f_spo`는 quadratic이라 **모든 샘플이 항상 0이 아닌 gradient**를 가져 r=1 경계 쪽으로 되돌아오는 corrective gradient 유지. → "PPO는 일부 데이터가 gradient 기여를 멈추지만 SPO는 전부 기여" (논문 주장).

### 1.3 공개 구현 (GitHub MyRepositories-hub/Simple-Policy-Optimization) — 논문과 일치 확인
- 구조: cleanrl 기반. `mujoco/trainer.py`, `atari/trainer.py`에 핵심 loss.
- **`spo` 분기 (verbatim)** — Eq.16과 정확히 동일:
```python
if self.args.algo == 'spo':
    policy_loss = -(
            mb_advantages * ratios -
            torch.abs(mb_advantages) * torch.pow(ratios - 1, 2) / (2 * self.args.epsilon)
    ).mean()
```
- `ppo` 분기 (참고, 기존 PPO와 동일):
```python
policy_loss_1 = mb_advantages * ratios
policy_loss_2 = mb_advantages * torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
```
- `compute_kld` 함수는 SPO loss에 **미사용** — `ppo` 분기의 adaptive-LR 로깅 전용. SPO objective는 **KL을 계산하지 않는다**(quadratic ratio penalty만).

### 1.4 process note — ar5iv 환각 정정 (재현 시 주의)
- 1차 조사에서 ar5iv HTML을 소형 요약모델로 WebFetch했을 때 "`d=KL(old‖new)`, `d_clip=clip(d,0,d_max)`, `J=O·(d_clip/d)` (Â>0) / `O·(−d_clip/d+2)` (Â<0)" 라는 **piecewise KL-clip 형태**가 보고됨.
- **PDF 원문 직접 판독 결과 이는 오독/환각.** 해당 `d_clip/d` piecewise는 SPO 알고리즘이 아니며, 논문 §4의 **TV/KL divergence lower-bound 이론(Prop 4.2, Thm 4.3, Eq.11–14)** 의 일부였다(이론적 동기 부여용, 알고리즘 아님). 실제 알고리즘 objective = Eq.16 quadratic.
- 교훈: **수식은 요약모델 paraphrase로 확정 금지. PDF/원문 직접 판독으로 검증.** 본 명세의 모든 SPO 수식은 PDF 직접 판독본이다.

---

## 2. 코드베이스 매핑 (정확한 파일:라인)

### 2.1 클래스 상속 구조 (검증됨 — 중요)
```
PPO            (ppo.py)            ── 독립 클래스
PPOParkour     (ppo_parkour.py)    ── 독립 클래스 (PPO 상속 아님! surrogate 블록 복제됨)
PPOAMPBase(PPO)         (ppo_amp.py)        ── PPO.update() 그대로 사용 (surrogate override 안 함)
PPOAMP(PPOParkour)      (ppo_amp.py)        ── PPOParkour.update() 그대로 사용
Distillation   (distillation.py)   ── surrogate 없음 (BC 류), 무관
ppo_parkour_original.py: 레거시 `class PPO` (RMA용). __init__.py에서 import 안 함 → 비활성. 손대지 말 것.
```
**결론**: surrogate를 바꾸려면 **`ppo.py`와 `ppo_parkour.py` 두 곳을 동일하게** 수정해야 PPO/PPOAMPBase/PPOParkour/PPOAMP 모두 커버. AMP 두 클래스는 `update()`를 override하지 않으므로 자동 상속.

### 2.2 변경 지점 — `rsl_rl/rsl_rl/algorithms/ppo.py`

| 영역 | 라인 | 현재 내용 | SPO 관련성 |
|---|---|---|---|
| `__init__` 시그니처 | 45–48 | `schedule`, `desired_kl`, `normalize_advantage_per_mini_batch`, `device` | **여기에 `surrogate_type`, `spo_epsilon` kwarg 추가** (default 지정 → 하위호환) |
| 파라미터 저장 | 126–129 | `self.desired_kl=...` 등 | `self.surrogate_type=...`, `self.spo_epsilon=...` 저장 |
| KL 계산부 (adaptive LR) | 263–272 | per-sample `kl` → `kl_mean`, `torch.inference_mode()` 안 | **SPO와 무관 — 손대지 않음.** adaptive-LR 그대로 동작(로깅/LR 스케줄용) |
| **surrogate 블록** | **298–304** | ratio-clip surrogate | **유일한 주 변경 지점.** SPO 분기 삽입 |

surrogate 블록 in-scope 변수 (298행 시점에 모두 존재 — 검증됨):
- `actions_log_prob_batch` (new logπ, 미분가능) — 255
- `old_actions_log_prob_batch` — generator
- `ratio = exp(new − old)` — 299 (미분가능)
- `advantages_batch` — generator/정규화됨
- `self.clip_param` — PPO clip (ppo 분기 전용)
→ **SPO quadratic objective가 필요로 하는 양(ratio, advantage)이 전부 in-scope.** mu/sigma·KL 불필요, 추가 텐서 불필요.

### 2.3 변경 지점 — `rsl_rl/rsl_rl/algorithms/ppo_parkour.py`
완전 동일 패턴:
| 영역 | 라인 |
|---|---|
| `__init__` 시그니처 | 45–48 (`schedule`/`desired_kl`/`normalize_advantage_per_mini_batch`/`device`) |
| 파라미터 저장 | 143–146 |
| KL 계산부 | 304–313 (inference_mode) |
| **surrogate 블록** | **340–346** |
in-scope 변수: `ratio`(341), `advantages_batch`. 동일하게 충분(mu/sigma 불필요).

### 2.4 변경 지점 — cfg 데이터클래스
`source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` → `class RslRlPpoAlgorithmCfg` (76–129).
- 새 필드를 **128행(`symmetry_cfg` 위) 또는 116행(clip_param 아래)** 에 추가:
```python
surrogate_type: str = "ppo"          # "ppo" | "spo"
"""Surrogate objective. 'ppo'=ratio-clip(기본), 'spo'=quadratic ratio penalty(Eq.16)."""

spo_epsilon: float = 0.2             # 논문 ε (PPO clip ε와 동일 스케일)
"""SPO quadratic penalty 계수 ε. surrogate_type='spo'일 때만 사용. f_spo = rA − (|A|/2ε)(r−1)²."""
```
- **propagation 경로 검증됨**: `on_policy_runner.py:286-289`에서 `alg_class(..., **self.alg_cfg, ...)`로 splat → cfg 필드가 그대로 `__init__` kwarg가 됨. 따라서 `__init__`에 동명 파라미터+default가 **반드시** 있어야 함(없으면 `TypeError: unexpected keyword`).
- **parkour 경로 커버 확인됨**: parkour agent cfg(`.../direct/parkour/agents/rsl_rl_ppo_cfg.py:71-72`)는 `algorithm = RslRlPpoAlgorithmCfg(class_name="PPOParkour", ...)` 로 **동일 dataclass를 재사용**한다(커스텀 cfg 아님). 따라서 위 필드 추가만으로 parkour/go2/R_Skeleton/hind_leg/motion_jig 등 `RslRlPpoAlgorithmCfg` 사용하는 모든 task에서 SPO opt-in 가능.
- YAML 사용자(`example_config.yaml` 등): `algorithm:` 블록에 `surrogate_type: spo` / `spo_epsilon: 0.2` 추가 시 동작. **미지정 시 default `"ppo"`로 기존 동작 유지.**

---

## 3. 구현 명세 (산출물 핵심)

### 3.1 `__init__` 변경 (ppo.py & ppo_parkour.py 둘 다)
시그니처에 추가 (default가 하위호환 핵심):
```python
        surrogate_type: str = "ppo",
        spo_epsilon: float = 0.2,
```
저장:
```python
        self.surrogate_type = surrogate_type
        self.spo_epsilon = spo_epsilon
```

### 3.2 surrogate 블록 교체 — SPO quadratic objective (Eq.16)

`ppo.py` 298–304 (그리고 `ppo_parkour.py` 340–346)을 아래로 교체:

```python
            # Surrogate loss
            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            adv = torch.squeeze(advantages_batch)

            if self.surrogate_type == "spo":
                # Simple Policy Optimization (arXiv:2401.16025), Eq.16:
                #   f_spo = r·A − (|A| / 2ε)·(r − 1)^2      (maximize)
                surrogate_obj = ratio * adv - torch.abs(adv) * torch.square(ratio - 1.0) / (2.0 * self.spo_epsilon)
                surrogate_loss = -surrogate_obj.mean()             # maximize → minimize negative
            else:  # "ppo" — 기존 ratio-clip (default, 기존 동작 보존)
                surrogate = -adv * ratio
                surrogate_clipped = -adv * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

핵심 포인트:
- SPO objective는 **`ratio`(미분가능, 299행)와 `adv`만** 사용 → KL 계산·mu/sigma 불필요. in-scope 변수만으로 완결.
- **`ratio`는 graph 안에서 계산되어야 함** (현행 299행이 이미 미분가능 — 그대로 사용). `surrogate_loss.backward()`가 policy까지 전파됨.
- `|adv|` penalty라 **advantage 부호 분기 불필요** — `r·A` 항이 부호 방향을 처리, penalty는 부호 무관 r=1 끌어당김.
- **KL=0 division 없음** — 분모는 상수 `2ε`라 안전. (이전 KL-clip 형태의 `d_clip/d` division 리스크는 quadratic 형태에서 **존재하지 않음**.)
- `self.spo_epsilon`은 SPO trust-region 하이퍼파라미터(논문 ε). PPO의 `clip_param`과 같은 스케일(≈0.2)로 시작 권장.

### 3.3 cfg (§2.4와 동일) + YAML 예시
```yaml
algorithm:
  class_name: PPO
  surrogate_type: spo      # 신규 (default ppo)
  spo_epsilon: 0.2         # 신규 (= 논문 ε, PPO clip 스케일)
  # ... 기존 필드 유지 (clip_param은 spo 모드에서 미사용이나 cfg에는 남겨둠)
```

### 3.4 edge case
- **division by zero 없음**: SPO 분모는 상수 `2·spo_epsilon`. `spo_epsilon>0` (cfg validation 권장) 외에는 NaN 경로 없음. PPO와 동일한 수치 안정성.
- **`ε` 작을 때**: `spo_epsilon→0`이면 penalty 항이 폭증해 사실상 r=1 고정(업데이트 약화). 너무 크면 trust region 느슨. 0.2(PPO clip)에서 시작 권장, sweep으로 조정.
- **schedule="adaptive" 상호작용**: 논문 repo는 `spo` 모드에서 adaptive-LR 분기를 끔(`algo=='ppo'`일 때만 LR 조정). 본 명세는 adaptive-LR 코드(263–296)를 **건드리지 않음** — SPO surrogate와 독립이라 동시 동작해도 충돌 아님(adaptive-LR는 inference KL 기반 LR 스케일, SPO는 loss 형태). **권장: SPO 모드에서 `schedule: fixed`로 두면 논문 설정에 더 가까움.** adaptive 유지도 무방하나 §4 ablation으로 확인.

### 3.5 하위호환 (backward compat) — 보장 메커니즘
- `surrogate_type` default `"ppo"` → 분기 `else`로 떨어져 **기존 PPO와 수치적으로 동일** (변수명 `surrogate`/`surrogate_clipped` 유지; `adv = squeeze(advantages_batch)`만 factor-out → 1-step diff=0 테스트로 검증).
- cfg 미지정 → dataclass default `"ppo"`.
- `__init__` 새 kwarg에 default 존재 → 기존 호출부(splat) 안 깨짐.
- **영향 받는 알고리즘**: PPO, PPOParkour, PPOAMPBase, PPOAMP 4종이 같은 surrogate 블록 공유. 단 **opt-in이라 default로는 전부 기존 동작.** spo 켜는 건 cfg 1줄.
- `ppo_parkour_original.py`(레거시 비활성): **수정 안 함**.

### 3.6 recurrent / multi-GPU / symmetry 경로
- **recurrent**: SPO surrogate는 `ratio`/`adv`만 쓰며 둘 다 recurrent 경로에서도 동일하게 채워짐 → 안전.
- **multi-GPU**: surrogate_loss는 `loss.backward()` 후 `reduce_parameters()`로 gradient all-reduce (368–381). **SPO는 loss 형태만 바꾸고 backward/reduce 구조 불변 → multi-GPU 안전.** adaptive-LR KL all-reduce(274–292)는 inference KL 기반이며 SPO surrogate와 독립.
- **symmetry**: augmentation은 `advantages_batch`/`ratio`를 모두 num_aug배로 확장(247–250 + 255행 act가 augmented obs로 재계산). SPO는 이 **동일 크기 텐서(`ratio`, `adv`)만** 쓰므로 broadcasting/shape 충돌 없음. (이전 KL-clip 초안의 mu/sigma 슬라이싱 위험은 **quadratic 형태에서 mu/sigma를 안 쓰므로 소멸**.) 단 검증 체크리스트에 shape 확인은 유지(§4).

---

## 4. 검증 / 실험 계획

### 4.1 저위험 A/B
- 동일 task(예: 안정적 baseline이 있는 locomotion flat) + 동일 seed/네트워크/스텝.
- Run A: `surrogate_type: ppo` (현 baseline). Run B: `surrogate_type: spo, spo_epsilon: 0.2`.
- **primary A/B는 `schedule: adaptive`로 실행** — 263–272행 KL 계산이 `schedule=="adaptive"`로 gated되어 있어, KL을 로깅·비교하려면 adaptive 경로가 켜져 있어야 함(apples-to-apples). `schedule: fixed`(§3.4의 논문-근접 설정)는 별도 ablation으로 분리.
- 비교 metric:
  - **mean reward / episode length** (성능): SPO가 동등 이상인지.
  - **mean KL (Episode KL 로깅, 263–272행 inference KL — adaptive 경로에서만 산출)**: SPO가 더 tight한 ratio/KL을 유지하는지(논문 §5.2 주장 — 모든 샘플이 r=1 경계로 끌림).
  - **surrogate loss / entropy 곡선**: 발산/붕괴 없는지.
  - **학습 안정성**: reward variance, ratio 분포(`|r−1|` 분포가 PPO보다 좁은지 — 논문 Fig.2/Fig.4 주장).
- 추가 ablation: `spo_epsilon ∈ {0.1, 0.2, 0.3}`.
- adaptive-LR 상호작용 ablation: `schedule: adaptive` vs `fixed` × SPO.

### 4.2 단위 검증 (validate-code 관점)
- [ ] **opt-in default**: `surrogate_type` 미지정 시 `"ppo"` → 기존 PPO와 loss 수치 동일(같은 seed 1-step diff=0) 확인.
- [ ] **shape**: `ratio` `[B]`, `adv` `[B]`, `surrogate_obj` `[B]` 일치 → `.mean()` 스칼라.
- [ ] **symmetry+SPO shape**: `ratio`/`adv`가 동일 augmented 크기인지 확인(quadratic 형태라 mu/sigma 무관 — 충돌 가능성 낮음, 그래도 1회 검증).
- [ ] **수치 안정성**: `spo_epsilon>0` 보장(0이면 div-by-zero). cfg validation 또는 assert 권장.
- [ ] **두 파일 동기화**: ppo.py와 ppo_parkour.py surrogate 블록 동일(드리프트 방지).
- [ ] **cfg ↔ __init__ 정합**: 새 cfg 필드명 == `__init__` kwarg명. splat 시 TypeError 없음. **parkour cfg(RslRlPpoAlgorithmCfg 재사용)도 함께 동작 확인.**
- [ ] **gradient 흐름**: `surrogate_loss.backward()`가 `ratio`(→policy logπ→policy params)까지 전파되는지(grad_fn 존재, ratio가 detach 안 됨).

---

## 5. Deploy / 리스크

- **Deploy 영향: 없음.** SPO는 순수 학습-시점 optimizer 목적함수 변경. 추론(policy forward)·관측·액션·네트워크 구조 불변 → 배포 정책 산출물 동일 인터페이스. sim-to-real 표면 변화 0.
- **학습 불안정 리스크 & 완화**:
  - `spo_epsilon` 과대 → penalty 약함 → trust region 느슨 → PPO보다 큰 업데이트 → 발산. 완화: 0.2에서 시작, sweep.
  - `spo_epsilon` 과소 → penalty 폭증 → 업데이트 정체. 완화: 0.1 미만 비권장.
  - `spo_epsilon=0` → division by zero. 완화: cfg validation/assert로 0 차단.
  - adaptive-LR + SPO 동시 작동. 완화: SPO 시 `schedule: fixed` 권장 또는 ablation으로 확인.
- **롤백 용이성**: cfg 1줄(`surrogate_type: ppo`)로 즉시 원복. 코드 분기는 opt-in이라 상시 안전.

---

## 부록 A. 변경 파일 요약

| 파일 | 변경 | 라인(현재) |
|---|---|---|
| `rsl_rl/rsl_rl/algorithms/ppo.py` | __init__ 2 kwarg + 저장 2줄 + surrogate 블록 교체 | 45–48, 126–129, **298–304** |
| `rsl_rl/rsl_rl/algorithms/ppo_parkour.py` | 동일 패턴 | 45–48, 143–146, **340–346** |
| `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` | `RslRlPpoAlgorithmCfg`에 `surrogate_type`+`spo_epsilon` 2필드 | ~116 또는 ~128 |
| (선택) `rsl_rl/config/example_config.yaml` | 문서용 예시 키 | algorithm 블록(48행 desired_kl 근처) |
| `ppo_amp.py` | **변경 불필요** (PPOAMPBase/PPOAMP가 update() override 안 함 → 상속 자동 적용; AMP 러너 `on_policy_runner_amp.py:183,392`가 `self.alg.update()` 호출 확인됨) | — |
| `ppo_parkour_original.py` | **변경 금지** (`__init__.py` import 안 됨 → 비활성 레거시) | — |

## 부록 B. 출처
- arXiv:2401.16025 — Abstract(원문 검증) + **PDF 원문 직접 판독**:
  - Eq.16: `J(θ) = E[ r·Â − (|Â|/2ε)·(r−1)² ]`
  - Eq.18: `f_spo = rA − (|A|/2ε)(r−1)²`, `f_ppo = min(rA, clip(r,1−ε,1+ε)A)`
  - Eq.19 (ε-aligned): `∂f_spo/∂r = A − (|A|/ε)(r−1) = 0` → `r* = 1+sign(A)ε`
  - Algorithm 1 line13: *"This is the only difference between SPO and PPO"*
  - §4 Eq.11–14 = TV/KL lower-bound 이론(Thm 4.3), **알고리즘 아님** (ar5iv 요약모델이 이를 SPO objective로 오독했던 부분).
- GitHub MyRepositories-hub/Simple-Policy-Optimization `mujoco/trainer.py` — `spo` 분기 verbatim: `mb_advantages*ratios − torch.abs(mb_advantages)*torch.pow(ratios-1,2)/(2*epsilon)` → **Eq.16과 정확 일치**. `compute_kld`는 spo loss에 미사용(ppo adaptive-LR 전용).
- 코드베이스: `ppo.py`(독립 PPO), `ppo_parkour.py`(독립 PPOParkour), `ppo_amp.py`(PPOAMPBase(PPO)/PPOAMP(PPOParkour)), `on_policy_runner.py:286-289`(cfg splat), `rl_cfg.py:76`(cfg dataclass), `.../parkour/agents/rsl_rl_ppo_cfg.py:71`(RslRlPpoAlgorithmCfg 재사용).

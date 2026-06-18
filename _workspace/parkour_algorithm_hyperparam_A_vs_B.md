# A vs B — 알고리즘 / 하이퍼파라미터 차이 분석

> - **A** = IsaacLab 본 repo parkour Direct RL 환경 (`agents/rsl_rl_ppo_cfg.py` + 메인 `rsl_rl/`)
> - **B** = `Isaaclab_Parkour` (`extreme_parkour_task/config/go2/` + 자체 `scripts/rsl_rl/`)
> 작성일 2026-05-19 · read-only 분석
> **2026-05-19 정정**: 초판의 `learning_rate`·`empirical_normalization` 값이 오류였음. `rsl_rl_ppo_cfg.py` 직접 확인 후 교정.

---

## 0. 한눈에 보는 결론

A와 B의 **알고리즘 코어는 거의 동일**하다 — 둘 다 PPO + RMA(privileged encoder + proprio-history DAGGER distillation). 네트워크 MLP 차원, scan/priv/history encoder 구조, **모든 PPO 하이퍼파라미터**(lr·clip·entropy·value_loss·gamma·lam·epochs·mini_batches·grad_norm·desired_kl)가 일치한다. A는 B 계열(extreme_parkour RMA)에서 알고리즘을 그대로 가져왔다.

**실제 차이는 ① 학습 파이프라인 구조, ② estimator 사용 여부, ③ num_envs 3가지뿐이다.**

| 항목 | A | B |
|---|---|---|
| **학습 파이프라인** | **단일 단계** (teacher만, RMA self-distill) | **2단계 teacher→student** (teacher = scan/privileged, student = depth-camera distillation) |
| **알고리즘 클래스** | `PPOParkour` | teacher `PPOWithExtractor` / student `DistillationWithExtractor` |
| **estimator (proprio→explicit-privileged 추정망)** | **미사용** (인스턴스화·학습·정책입력 모두 안 함 — 확정) | **사용** — hidden [128,64], lr 1e-4, `train_with_estimated_states=True` |
| **num_envs** | 4096 (launch arg) | **6144** (teacher cfg) |
| **depth/vision** | 없음 | student에 DepthCNN(58×87) + GRU(512) |
| learning_rate | 2.0e-4 | 2.0e-4 (teacher) — **동일** |
| empirical_normalization | False | False — **동일** |
| Actor/Critic MLP | [512,256,128] elu | [512,256,128] elu — 동일 |
| scan/priv/history encoder | [128,64,32] / [64,20] / Conv1D | [128,64,32] / [64,20] / Conv1D — 동일 |
| clip/entropy/value_loss/gamma/lam | 0.2 / 0.01 / 1.0 / 0.99 / 0.95 | 동일 |
| num_steps_per_env / mini_batches / epochs | 24 / 4 / 5 | 24 / 4 / 5 — 동일 |
| desired_kl / schedule / max_grad_norm | 0.01 / adaptive / 1.0 | 0.01 / adaptive / 1.0 — 동일 |
| priv_reg_coef_schedule / dagger_freq | [0,0.1,2000,3000] / 20 | [0,0.1,2000,3000] / 20 — 동일 |
| max_iterations / save_interval | 50000 / 100 | 50000 / 100 — 동일 |

> ⚠️ `rsl_rl_ppo_cfg.py:67`은 `learning_rate=2.0e-4`이며 주석 `# CHANGED: 2.0e-4 → 1.0e-3 -> 1.0e-4`은 *변경 이력*일 뿐 현재값이 아니다. line 26 `empirical_normalization = False`, line 51-52 `actor/critic_obs_normalization=False`.

---

## 1. 알고리즘 — 코어는 동일, 파이프라인이 다름

### 공통 코어 (A ≈ B의 teacher)
둘 다 **PPO + RMA(Rapid Motor Adaptation)** 구조:
- privileged encoder가 domain-rand latent 파라미터(마찰·질량·강성 등)를 latent로 인코딩
- proprio-history encoder(Conv1D)가 history로부터 privileged latent를 모방하도록 DAGGER 방식으로 distill
- `priv_reg` loss로 두 latent를 정렬, coef schedule `[0,0.1,2000,3000]`, dagger_freq 20 — 동일

→ A의 학습 알고리즘은 사실상 **B의 teacher 단계와 같다**.

### 결정적 구조 차이 — B는 2단계, A는 1단계
- **B**: ① teacher(`PPOWithExtractor`)를 scan + privileged로 학습 → ② student(`DistillationWithExtractor`)가 depth 카메라 입력으로 teacher 행동을 distillation. student는 6144→192 envs.
- **A**: teacher 단계만 존재. depth/vision student 없음.

> 평지 보행 gait는 **teacher 단계에서 형성**된다. 보행 품질 비교의 유효 대상은 **A ↔ B-teacher**이며, depth student 단계는 평지 gait 품질의 원인이 아니다.

---

## 2. estimator — A·B의 핵심 알고리즘 차이 (확정)

코드 직접 추적으로 확정된, A·B의 가장 명확한 알고리즘 측 차이.

### A — estimator 미사용 (3중 확인)
- 인스턴스화: `on_policy_runner_parkour.py:41`의 `self.estimator_cfg = ...`가 **주석 처리**됨. `PPOParkour.__init__`에 estimator 파라미터 없음.
- 학습: `ppo_parkour.py`에 "estimator" 문자열 0줄 — estimator loss/optimizer 없음.
- 정책 입력: `ActorCriticRMA`의 `act()`/`act_inference()`에서 estimator 호출 없음.
→ **A는 estimator를 인스턴스화·학습·사용 모두 하지 않는다.**

### B — estimator 사용 (3중 확인)
- 인스턴스화: `ppo_with_extractor.py:65` `self.estimator = estimator`.
- 학습: `ppo_with_extractor.py:190-196` — `(priv_predicted - priv_ground_truth)²`의 MSE loss로 별도 Adam optimizer(lr 1e-4) 학습.
- 정책 입력: `train_with_estimated_states=True`일 때 `act()`에서 obs의 explicit-privileged 슬라이스를 estimator 추정값으로 **치환**한 뒤 policy에 입력.

### estimator의 역할
- **입력**: proprioceptive obs (`num_prop`, B=53-dim) — 실제 로봇이 직접 측정 가능한 자기상태.
- **출력**: explicit privileged state (`num_priv_explicit`, B=9-dim — base linear velocity 계열). 실제 로봇이 직접·신뢰성 있게 측정하기 어려운 양.
- **학습**: supervised regression. 시뮬레이터의 ground-truth privileged를 정답으로 MSE 최소화.
- **sim-to-real 논리**: 학습 시 estimator가 proprio→base-velocity 매핑을 배우고, 정책은 (ground-truth가 아니라) **추정값으로 행동하도록** 학습된다(`train_with_estimated_states=True`). deploy 시 ground-truth가 없어도 estimator가 proprio만으로 base velocity를 공급 → 정책이 동일하게 동작. 즉 explicit privileged의 sim-to-real gap을 메우는 모듈.
- **history encoder와의 분담**: RMA history encoder는 *latent* privileged(마찰/질량/강성 등 domain 파라미터)를 proprio history로부터 복원. estimator는 *explicit* privileged(base 속도)를 proprio로부터 복원. B는 두 경로를 함께 쓴다.

### A는 왜 estimator 없이 동작하나
A는 explicit base velocity를 actor obs(42-dim)에 노출하지 않는다 — base velocity는 critic obs와 priv_encoder 경로에만 들어가고, actor는 history encoder가 복원한 latent로 privileged 정보를 간접 수령한다. 따라서 A의 actor는 explicit base-velocity 추정망이 *구조적으로 불필요*하다. B는 explicit base velocity를 actor 입력으로 노출하는 설계라 estimator가 필요하다. → 같은 RMA 계열이지만 privileged routing 설계가 갈린다.

---

## 3. empirical_normalization — 차이 없음 (정정)

초판은 A를 True로 잘못 기재했다. 실제로는 **A·B 모두 `empirical_normalization=False`**:
- A: `rsl_rl_ppo_cfg.py:26` `empirical_normalization = False`, line 51-52 `actor/critic_obs_normalization=False` → `ActorCriticRMA`의 모든 obs normalizer가 `torch.nn.Identity()`로 설정(정규화 미적용).
- B: teacher/student cfg 모두 `empirical_normalization=False`.
→ 양쪽 다 env 단계의 수동 스케일(ang_vel ×0.25, joint_vel ×0.05 등)만 적용. **이중 정규화 없음. 차이 아님.**

---

## 4. 평지 보행 품질 질문(직전 분석)과의 연결

- **알고리즘·하이퍼파라미터는 평지 보행 어색함의 원인이 아니다.** A와 B-teacher는 동일한 PPO+RMA, 동일한 네트워크, 동일한 모든 PPO 하이퍼파라미터(lr 포함)를 쓴다. "Manager vs Direct 프레임워크가 원인 아님"과 같은 결론.
- 알고리즘 측 실제 divergence는 **학습 파이프라인(1단계 vs 2단계)**, **estimator 사용 여부**, **num_envs(4096 vs 6144)** 3가지. 이 중 파이프라인·estimator는 sim-to-real / vision 관련 설계이지 평지 gait 품질 결정 요인이 아니다. num_envs는 데이터 다양성에 영향을 주는 보조 요인.
- 따라서 직전 분석의 결론(평지 drag 보행의 1순위 원인 = reward 측 `dof_error ×10` + `feet_air_time` 비활성, 이미 수정 완료)은 **유지된다.** 알고리즘 비교가 새 1순위 원인을 만들지 않는다.

## 5. 종합

| 구분 | 판정 |
|---|---|
| 알고리즘 코어 (PPO+RMA, 네트워크, **모든** PPO 하이퍼파라미터) | **A ≈ B-teacher, 동일** |
| 학습 파이프라인 | A 1단계 / B 2단계(teacher→depth student) — 구조적 최대 차이, 평지 gait와 무관 |
| estimator | A 미사용 / B 사용 (proprio→explicit-privileged 추정, sim-to-real 모듈) — privileged routing 설계 차이 |
| num_envs | 4096 vs 6144 — 보조 요인 |
| empirical_normalization | 둘 다 False — 차이 없음 (초판 오류 정정) |
| learning_rate | 둘 다 2.0e-4 — 차이 없음 (초판 오류 정정) |
| 평지 보행 어색함의 원인 | 알고리즘 측 아님 — reward 측(직전 수정 완료)이 1순위로 유지 |

# R_Skeleton AMP 구현 검토 보고서

**작성일:** 2026-03-20
**검토자:** Claude Code (claude-sonnet-4-6)
**검토 대상:** R_Skeleton-AMP-v0 (Adversarial Motion Prior) 구현 전체

---

## 1. 개요

본 보고서는 R_Skeleton 로봇에 적용된 AMP(Adversarial Motion Prior) 구현이 논문
("AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control", Peng et al., SIGGRAPH 2021)의 핵심 개념과 일치하는지 검토한 결과를 정리합니다.

### 검토 파일 목록

| 파일 | 역할 |
|------|------|
| `skeleton_amp_env.py` | AMP 환경 클래스 (Isaac Lab DirectRLEnv 상속) |
| `skeleton_amp_env_cfg.py` | 환경 설정 |
| `motion_loader.py` | DeepMimic 포맷 모션 데이터 로더 |
| `agents/rsl_rl_ppo_cfg.py` | RSL-RL 학습 설정 |
| `rsl_rl/runners/on_policy_runner_amp.py` | AMP 학습 루너 |
| `rsl_rl/algorithms/ppo_amp.py` | PPOAMP 알고리즘 (Discriminator 학습 포함) |
| `rsl_rl/modules/amp_discriminator.py` | Discriminator 네트워크 |

---

## 2. AMP 알고리즘 개요

AMP는 모션 캡처 데이터를 "전문가 시연"으로 활용하여, GAN 방식으로 로봇이 자연스러운 움직임을 모방하도록 학습시키는 방법입니다.

```
┌──────────────────────────────────────────────────────────────┐
│  Policy (Actor)                                              │
│    └→ actions → Isaac Sim → state transition                 │
│                                                              │
│  Task Reward (r_task): 속도 추종 등 태스크 목표              │
│  Style Reward (r_style): Discriminator 기반 모션 자연스러움  │
│                                                              │
│  Total = task_lerp * r_task + (1-task_lerp) * r_style        │
│                                                              │
│  Discriminator: expert 모션 vs policy 모션 구분              │
│    - expert → 1, policy → 0 (BCE Loss)                       │
│    - WGAN-GP Gradient Penalty로 정규화                       │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. 올바르게 구현된 부분

### 3.1 AMP 관측 벡터 구조 (99차원)

**파일:** `skeleton_amp_env.py:556-588` (`compute_obs` 함수)

```
dof_pos(34) + dof_vel(34) + root_height(1) + root_lin_vel(3) + root_ang_vel(3)
+ key_body_pos_local(12) + key_body_vel_local(12) = 99차원
```

- 논문의 AMP state `s` 정의와 일치
- key body(발끝 4개) 위치/속도를 **루트 로컬 좌표계**로 올바르게 변환
  - 시뮬레이션 (`skeleton_amp_env.py:182-190`): `quat_apply_inverse` 사용
  - 레퍼런스 모션: txt 파일의 `toe_pos_local` 직접 활용

### 3.2 전체 학습 흐름

**파일:** `on_policy_runner_amp.py:47-167`

```
[매 iteration]
1. rollout 수집 (num_steps_per_env = 24 스텝)
2. extras["amp_obs"] 추출 → Discriminator로 Style reward 계산
3. total_reward = 0.5 * task_reward + 0.5 * style_reward
4. Discriminator 업데이트 (expert vs policy 구분 학습)
5. Policy & Critic 업데이트 (PPO)
6. DAgger 업데이트 (history encoder 학습)
```

논문의 학습 루프와 일치 ✅

### 3.3 AMP 보상 공식

**파일:** `amp_discriminator.py:50-54`

```python
prob = torch.sigmoid(disc_logits)
reward = -torch.log(torch.maximum(1 - prob, tensor(0.0001)))
reward = torch.clamp(reward, 0.0, 10.0)
return reward * amp_reward_coef
```

논문 수식 `r_style = -log(1 - D(s))` 와 일치 ✅
수치 안정성을 위해 `max(1-prob, 0.0001)` 처리 및 `[0, 10]` 클램핑 적용 ✅

### 3.4 Discriminator 손실 함수

**파일:** `ppo_amp.py:69-70`

```python
expert_loss = BCEWithLogitsLoss()(expert_logits, torch.ones_like(expert_logits))
policy_loss = BCEWithLogitsLoss()(policy_logits, torch.zeros_like(policy_logits))
total_loss = 0.5 * (expert_loss + policy_loss) + gradient_penalty_coef * 0.5 * grad_penalty
```

논문의 GAN 학습 방식과 일치 ✅

### 3.5 Discriminator 입력 차원 일관성

| 항목 | 값 |
|------|-----|
| `amp_observation_space` (단일 프레임) | 99 |
| `num_amp_observations` (히스토리 프레임 수) | 2 |
| `amp_observation_size` (Discriminator 입력) | 2 × 99 = **198** |
| `gym.spaces.Box shape` | (198,) |
| Runner 탐지 값 | 198 |
| `AMPDiscriminator(input_dim)` | 198 |
| `extras["amp_obs"].shape` | `(num_envs, 198)` |

모든 경로에서 198차원으로 통일 ✅

### 3.6 Save/Load에 Discriminator 포함

**파일:** `on_policy_runner_amp.py:169-189`

```python
saved_dict = {
    "model_state_dict":           ...,   # Policy 네트워크
    "discriminator_state_dict":   ...,   # Discriminator ✅
    "optimizer_state_dict":       ...,   # Policy 옵티마이저
    "disc_optimizer_state_dict":  ...,   # Discriminator 옵티마이저 ✅
    "iter":                       ...,
}
```

체크포인트 재개 시 Discriminator 상태도 복원 ✅

### 3.7 Expert 샘플러 인터페이스

**파일:** `skeleton_amp_env.py:458-464`

```python
def get_amp_observations(self, num_samples: int) -> torch.Tensor:
    return self.collect_reference_motions(num_samples)
```

Runner가 `hasattr(env, "get_amp_observations")` 로 존재 여부를 확인하고 호출 ✅

### 3.8 Inference Mode에서 AMP Reward 계산

**파일:** `on_policy_runner_amp.py:70`

```python
with torch.inference_mode():  # rollout 수집 구간
    ...
    amp_reward = self.alg.discriminator.compute_amp_reward(agent_amp_obs)
```

Discriminator gradient가 Policy 업데이트에 영향 주지 않음 ✅

### 3.9 RSI (Reference State Initialization) 구현

**파일:** `skeleton_amp_env.py:375-419`

`reset_strategy = "random"` 선택 시 모션 데이터의 임의 시점에서 에피소드 초기화.
논문의 핵심 학습 안정화 트릭 구현됨 ✅

### 3.10 모션 보간 (선형 + SLERP)

**파일:** `motion_loader.py:283-344`

- 관절 위치/속도, body 위치/속도: 선형 보간 (`_interpolate`)
- 쿼터니언 회전: SLERP (`_slerp`)

불연속 모션 없이 임의 시점 샘플링 가능 ✅

---

## 4. 발견된 문제 및 수정 내역

### 4.1 [수정 완료] Normalizer가 Policy Obs만으로 업데이트

**파일:** `rsl_rl/rsl_rl/algorithms/ppo_amp.py:101`
**심각도:** 중간

**문제:**
```python
# 이전 코드
self.discriminator.update_normalization(policy_batch)
```
Policy 데이터만으로 정규화 통계를 갱신하면, Expert obs와 Policy obs가 서로 다른 분포로 정규화되어 Discriminator가 편향된 판별을 학습하게 됩니다.

**수정:**
```python
# 수정된 코드
self.discriminator.update_normalization(torch.cat([expert_batch, policy_batch]))
```
Expert + Policy 합산 데이터로 정규화 통계 갱신 → 두 분포가 동일한 기준으로 정규화됨

---

### 4.2 [수정 완료] Expert 샘플링 수 과다

**파일:** `rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:124`
**심각도:** 성능

**문제:**
```python
# 이전 코드
num_samples = policy_amp_obs_batch.shape[0]  # 4096 × 24 = 98,304
```
매 iteration마다 98,304개의 Expert 모션을 샘플링하면:
- `collect_reference_motions(98304)` → 196,608번 보간 연산
- 학습 속도 심각하게 저하

**수정:**
```python
# 수정된 코드
num_samples = min(policy_amp_obs_batch.shape[0], 4096)
```
최대 4,096개로 제한 → 연산량 약 24배 절감

---

### 4.3 [검증 필요] DOF_NAMES 클래스 변수 순서 불일치

**파일:** `motion_loader.py:59-98` vs `motion_loader.py:391-431`
**심각도:** 심각 (현재 동작에는 영향 없을 수 있으나 혼동 유발)

**문제:**
```python
# DOF_NAMES (클래스 변수): FL이 인덱스 0-5
DOF_NAMES = ["FL_joint1_hip_yaw", ...]  # index 0 = FL

# get_dof_index 함수: FL → 7번부터 매핑
elif name.startswith('FL_joint'):
    idx = 7 + j  # ❗ 0이 아닌 7
```

`get_dof_index`의 주석에 따르면 실제 txt 파일의 관절 순서:
```
인덱스 0-6:   N_joint (목)
인덱스 7-12:  FL_joint (앞왼쪽 다리)
인덱스 14-19: FR_joint (앞오른쪽 다리)
인덱스 21-23: W_joint (허리)
인덱스 24-29: HL_joint (뒷왼쪽 다리)
인덱스 31-36: HR_joint (뒷오른쪽 다리)
```

`DOF_NAMES`는 이 순서를 반영하지 않음 → 코드 가독성 저하 및 잠재적 혼동.

**확인 방법:** txt 파일 첫 번째 관절이 N_joint인지 직접 확인 필요.

---

### 4.4 [검증 필요] priv_obs 차원 하드코딩

**파일:** `skeleton_amp_env.py:240-249`, `skeleton_amp_env_cfg.py:53`
**심각도:** 중간 (런타임 에러 가능)

**문제:**
```python
# skeleton_amp_env_cfg.py
num_priv_obs = 114  # 하드코딩

# skeleton_amp_env.py
priv_obs = torch.cat([
    root_lin_vel_b,                                              # 3
    root_ang_vel_b,                                             # 3
    get_masses().reshape(num_envs, -1),                         # 링크 수
    get_material_properties().reshape(num_envs, -1),            # 링크 수 × 3
])
```

실제 차원 = 3 + 3 + (링크 수) + (링크 수 × 3)이며, 링크 수는 URDF에 따라 다름.
R_Skeleton 링크 수가 27개라면: 3+3+27+27×3 = 114 (일치)
링크 수가 다를 경우 critic 네트워크 입력 차원 불일치 → 런타임 에러 발생.

**확인 방법:**
```python
print(priv_obs.shape)  # 실제 priv_obs 차원 확인
```

---

### 4.5 [검증 필요] collect_reference_motions의 key body velocity 일관성

**파일:** `skeleton_amp_env.py:447-455`
**심각도:** 중간 (Discriminator 학습 편향 가능)

**시뮬레이션** (`_get_observations`):
```python
# 루트 기준 상대 속도 → 로컬 프레임 변환
rel_vel = body_lin_vel_w[key] - root_lin_vel_w  # 상대 속도 (world frame)
local_key_body_vel = quat_apply_inverse(root_quat, rel_vel)  # 로컬 변환
```

**레퍼런스 모션** (`collect_reference_motions`):
```python
# txt 파일의 toe_vel_local 직접 사용
body_linear_velocities[:, self.motion_key_body_indexes]
```

만약 txt 파일의 `toe_vel_local`이 **절대 속도**라면 (루트 속도 미차감),
시뮬레이션의 **상대 속도**와 불일치 → Discriminator가 잘못된 기준으로 학습됨.

**확인 방법:**
- 정지 상태(lin_vel ≈ 0) 모션 프레임에서 `toe_vel_local ≈ [0, 0, 0]`이면 상대 속도
- `toe_vel_local ≈ lin_vel_root`이면 절대 속도 → 수정 필요

---

### 4.6 [경미] Gradient Penalty 공식이 표준 WGAN-GP와 다름

**파일:** `ppo_amp.py:93`
**심각도:** 경미

```python
# 현재: 단순 L2 제곱합
grad_penalty = torch.sum(torch.square(gradients), dim=-1).mean()

# 표준 WGAN-GP: 1-Lipschitz 강제
grad_penalty = ((gradients.norm(2, dim=-1) - 1) ** 2).mean()
```

현재 구현은 `||grad||²`를 최소화하는 방식으로, Discriminator의 gradient를 0으로 당깁니다.
표준 WGAN-GP는 `(||grad||₂ - 1)²`를 최소화하여 1-Lipschitz 조건을 강제합니다.
학습이 불안정하거나 Discriminator가 너무 강해지는 경우 이 공식 수정을 고려하세요.

---

### 4.7 [경미] 클래스 docstring 오기

**파일:** `skeleton_amp_env.py:35-38`

```python
# 현재 (오기):
"""
AMP 관측 벡터 (amp_observation_space = 105):
    ... + tangent_normal(6) ...
"""

# 실제:
# amp_observation_space = 99
# tangent_normal은 미사용
```

코드와 주석 불일치 → 혼동 유발.

---

## 5. 전체 검토 결과 요약

| 영역 | 상태 | 파일 위치 | 비고 |
|------|------|-----------|------|
| 학습 루프 흐름 | ✅ 정상 | `on_policy_runner_amp.py` | 논문과 일치 |
| AMP 보상 공식 | ✅ 정상 | `amp_discriminator.py:50-54` | `-log(1-D(x))` 정확 |
| Discriminator 손실 | ✅ 정상 | `ppo_amp.py:69-70` | BCE Loss 정확 |
| 관측 차원 일관성 | ✅ 정상 | 환경-Runner 연동 | 198차원 통일 |
| Save/Load | ✅ 정상 | `on_policy_runner_amp.py` | Discriminator 포함 |
| 모션 보간 | ✅ 정상 | `motion_loader.py` | 선형 + SLERP 정확 |
| RSI 구현 | ✅ 정상 | `skeleton_amp_env.py` | "random" 전략 지원 |
| Normalizer 업데이트 | ✅ 수정됨 | `ppo_amp.py:101` | Expert+Policy 합산으로 변경 |
| Expert 샘플링 수 | ✅ 수정됨 | `on_policy_runner_amp.py:124` | 4096 상한 적용 |
| DOF_NAMES 순서 | ⚠️ 불일치 | `motion_loader.py:59-98` | 검증 필요 |
| priv_obs 차원 | ⚠️ 미확인 | `skeleton_amp_env_cfg.py:53` | 런타임 확인 필요 |
| key body velocity | ⚠️ 미확인 | `skeleton_amp_env.py:454` | 상대/절대 속도 검증 필요 |
| Gradient Penalty | ⚠️ 비표준 | `ppo_amp.py:93` | 비표준이나 동작 가능 |
| reset_strategy 기본값 | ⚠️ 비권장 | `skeleton_amp_env_cfg.py:63` | "random" 권장 |
| docstring | ❌ 오기 | `skeleton_amp_env.py:35` | 105→99 수정 필요 |

---

## 6. 남은 권장 수정 사항

### 우선순위 1 (기능 검증)
1. `motion_loader.py` 에서 실제 txt 파일을 열어 관절 순서 확인 후 `DOF_NAMES` 업데이트
2. 환경 실행 시 `priv_obs.shape` 출력하여 `num_priv_obs = 114` 검증
3. 정지 모션 프레임에서 `toe_vel_local ≈ 0` 확인 (상대 속도 여부)

### 우선순위 2 (품질 개선)
4. `skeleton_amp_env_cfg.py:63` — `reset_strategy = "random"` 으로 변경 (RSI 활성화)
5. `skeleton_amp_env.py:35-38` — docstring 오기 수정 (`105 → 99`)

### 우선순위 3 (선택적)
6. `ppo_amp.py:93` — Gradient Penalty를 표준 WGAN-GP 공식으로 변경
   `((gradients.norm(2, dim=-1) - 1) ** 2).mean()`

---

## 7. 결론

R_Skeleton AMP 구현은 **논문의 핵심 알고리즘(AMP 보상, GAN 학습, RSI)을 올바르게 구현**하고 있습니다.
이번 검토를 통해 수정된 2가지 버그(Normalizer 편향, Expert 과다 샘플링)가 적용되었으며,
검증이 필요한 3가지 항목이 남아있습니다. 해당 항목들은 학습 실행 시 확인 가능합니다.

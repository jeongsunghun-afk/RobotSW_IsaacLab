# RSL-RL 알고리즘 컨텍스트

## 디렉토리 구조

```
rsl_rl/rsl_rl/
├── algorithms/
│   ├── ppo.py                  # 기본 PPO
│   └── ppo_amp.py              # AMP 확장 PPO (discriminator loss 포함)
├── modules/
│   ├── actor_critic.py         # 기본 Actor-Critic
│   ├── actor_critic_recurrent.py
│   └── amp_discriminator.py    # AMP Discriminator (MLP)
├── networks/                   # 네트워크 빌딩 블록
└── runners/
    ├── on_policy_runner.py         # 기본 PPO 러너
    └── on_policy_runner_amp.py     # AMP 러너 (expert buffer 관리)
```

## AMP 학습 파이프라인

```
[on_policy_runner_amp.py]
  1. env.step(actions) → (obs, reward, done, info)
  2. amp_obs 수집 (env에서 추출)
  3. expert_batch 샘플링 (motion_loader에서)
  4. ppo_amp.update(rollout, expert_batch) 호출

[ppo_amp.py::update()]
  5. policy loss 계산 (표준 PPO)
  6. discriminator loss 계산:
     - policy_score = D(amp_obs_from_policy)
     - expert_score = D(amp_obs_from_expert)
     - loss = BCE(policy_score, 0) + BCE(expert_score, 1)
  7. amp_reward = clamp(1 - 0.25 * policy_score, min=0)
  8. total_reward = lerp(task_reward, amp_reward, 1-task_reward_lerp)
```

## 핵심 파라미터 위치

### agents/rsl_rl_ppo_cfg.py (환경별)
```python
@configclass
class PPORunnerCfg:
    num_steps_per_env: int = 24      # rollout 길이
    max_iterations: int = 5000
    save_interval: int = 100
    logger: str = "tensorboard"      # "wandb" 권장

@configclass
class PPOCfg:
    clip_param: float = 0.2
    entropy_coef: float = 0.01
    learning_rate: float = 1e-3
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95                # GAE lambda

@configclass
class AMPCfg:
    task_reward_lerp: float = 0.5    # 0: pure AMP, 1: pure task
    reward_coef: float = 0.04        # AMP reward 스케일 (2.0 * 0.02)
    motion_files: list = [...]
    reset_strategy: str = "random"   # RSI 활성화
```

## amp_discriminator.py 핵심
```python
class AMPDiscriminator:
    # input_dim = amp_observation_space × num_amp_observations
    # Go2: 55 × 2 = 110
    # R_Skeleton: 99 × 2 = 198

    def compute_grad_pen(self, ...):  # gradient penalty (학습 안정화)
    def predict_amp_scores(self, obs): # D(obs) → scalar score
    def update_normalization(self, batch): # running mean/std 업데이트
```

## 수정 빈도별 파일
| 빈도 | 파일 | 이유 |
|------|------|------|
| 자주 | `agents/rsl_rl_ppo_cfg.py` | 하이퍼파라미터 튜닝 |
| 가끔 | `ppo_amp.py` | reward 결합 방식 변경 |
| 가끔 | `amp_discriminator.py` | 구조/정규화 변경 |
| 드물게 | `on_policy_runner_amp.py` | 데이터 수집 방식 변경 |

## 디버깅 가이드
- **AMP reward → 0**: discriminator obs shape 불일치 또는 reference 속도 범위 초과
- **policy loss NaN**: learning_rate 너무 크거나 reward 폭발 (clip 미적용)
- **단일 gait 수렴**: task_reward_lerp > 0.7 이거나 reference data 단조로움
- **학습 불안정**: reset_strategy="default" (RSI 비활성) → "random"으로 변경

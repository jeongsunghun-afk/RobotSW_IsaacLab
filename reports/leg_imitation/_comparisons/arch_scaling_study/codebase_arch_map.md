# RSL-RL 정책 아키텍처 Seams: Leg-Imitation-Tracking-RMA-v0 분석

**문서 목적:** Leg-Imitation-Tracking-RMA-v0 환경에 새로운 actor-critic 아키텍처 변형(deeper MLP, residual blocks, mixture-of-Gaussians 등)을 추가하기 위한 최소 침투적 변경 경로를 매핑한다.

**작성일:** 2026-08-26 | **대상:** IsaacLab 6.0 vendored rsl_rl 3.2.0

---

## 1. 모듈 인벤토리

### 1.1 Actor-Critic 클래스

| 클래스 | 파일:줄수 | 서명 | 설명 |
|--------|---------|------|-----|
| **ActorCritic** | `rsl_rl/modules/actor_critic.py:22-238` | `__init__(obs, obs_groups, num_actions, actor_hidden_dims=[256,256,256], critic_hidden_dims=[256,256,256], activation="elu", ...)` | 기본 피드포워드; actor와 critic 분리; obs_groups["policy"] vs ["critic"] |
| **ActorCriticRecurrent** | `rsl_rl/modules/actor_critic_recurrent.py:23-228` | `__init__(..., rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1)` | LSTM/GRU 래핑; Memory 모듈 사용 |
| **ActorCriticRMA** | `rsl_rl/modules/actor_critic_parkour.py:81-~500` | `__init__(..., scan_encoder_dims=[128,64,32], priv_encoder_dims=[64,20])` | Domain Randomization + 적응; history_encoder (StateHistoryEncoder 사용) + priv_encoder; scanbot 지원 |
| **ActorCriticRMALidar** | `rsl_rl/modules/actor_critic_parkour.py:~800+` | (ActorCriticRMA 서브클래스) | LidarEncoder 추가 (parkour_imitation_lidar) |
| **ActorCriticRMAVoxel** | `rsl_rl/modules/actor_cricket_parkour.py:~1000+` | (ActorCriticRMA 서브클래스) | VoxelEncoder 변형 (parkour_imitation voxel) |

### 1.2 보조 모듈

| 모듈 | 파일:줄수 | 목적 |
|------|---------|------|
| **MLP** | `rsl_rl/modules/mlp.py:21-101` | 다층 퍼셉트론: `Linear → activation → ... → Linear` 순차 구성. hidden_dims=-1 추론 지원, 출력 reshape 지원 |
| **AMPDiscriminator** | `rsl_rl/modules/amp_discriminator.py:12-99` | 동작 품질 분류기: `input_dim → hidden_dims → 1 logit`. "ls_gan", "bce", "wgan" 보상 타입 지원 |
| **Estimator** | `rsl_rl/modules/estimator.py:13-37` | RMA 특권 상태 추정기: simple MLP (`input_dim → hidden_dims → output_dim`) |
| **StateHistoryEncoder** | `rsl_rl/modules/actor_cricket_parkour.py:23-78` | 시간 이력 인코더: `Linear → Conv1d chains → Flatten → output`. tsteps ∈ {10, 20, 50}별 조정 |

### 1.3 활성화 함수 해석기

**파일:** `rsl_rl/rsl_rl/utils/utils.py:37-71`

```python
def resolve_nn_activation(act_name: str) -> torch.nn.Module:
    """지원: elu, selu, relu, crelu, lrelu, tanh, sigmoid, softplus, gelu, swish, mish, identity"""
```

**정규화 지원:** 없음 (LayerNorm, BatchNorm 미포함). 정규화는 `EmpiricalNormalization` 모듈로 별도 관리 (actor-critic 클래스 내에서 `actor_obs_normalizer`, `priv_explicit_obs_normalizer` 등).

---

## 2. MLP 빌더 & 확장 포인트

**파일:** `rsl_rl/modules/mlp.py:21-101`

```python
class MLP(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | tuple | list,
        hidden_dims: tuple | list,
        activation: str = "elu",
        last_activation: str | None = None,
    ):
        # layers = [Linear(input_dim, hidden_dims[0]), activation,
        #           Linear(...), activation, ..., Linear(..., output_dim)]
        # last_activation 지정 시 마지막에 적용
```

**제약:**
- 순차 구성만 지원 (residual 및 parallel 경로 없음)
- 정규화 레이어 미내장 (외부에서 별도 추가 필요)

---

## 3. 인스턴스화 경로

### 3.1 클래스 해석

**러너 → 정책 클래스:**

`scripts/reinforcement_learning/rsl_rl/train.py:227-316`에서 runner class를 dispatch:
- `agent_cfg.class_name = "OnPolicyRunnerAMPBase"` → `OnPolicyRunnerAMPBase` 인스턴스화
- 내부 `_construct_algorithm()` (line 40-91, `rsl_rl/runners/on_policy_runner_amp.py`)에서:
  ```python
  actor_critic_class = self._get_actor_critic_class()  # 오버라이드 가능
  actor_critic = actor_critic_class(obs, self.cfg["obs_groups"],
                                      self.env.num_actions, **self.policy_cfg)
  ```

**정책 cfg 전달:**

`RslRlPpoActorCriticCfg` (deprecated, 3.2.0 호환) 필드:
```python
class_name: str = "ActorCritic"
activation: str = MISSING
actor_hidden_dims: list[int] = MISSING
critic_hidden_dims: list[int] = MISSING
init_noise_std: float = MISSING
...
```

**kwargs 처리:** 미지정 파라미터는 기본적으로 **무시됨** (경고만 출력). 새 아키텍처 하이퍼파라미터는 cfg 스키마 변경 불필요.

### 3.2 Leg-Imitation-Tracking-RMA 설정 예시

**파일:** `source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/agents/rsl_rl_ppo_cfg.py:50-58`

```python
policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
    class_name="ActorCriticRMA",
    actor_hidden_dims=[512, 256, 128],
    critic_hidden_dims=[512, 256, 128],
    activation="elu",
    init_noise_std=0.25,
    ...
)
```

---

## 4. Leg-Imitation-Tracking 환경 사양

**env cfg 파일:** `source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/leg_imitation_tracking_env_cfg.py:39-72`

| 항목 | 값 | 설명 |
|------|-----|------|
| **observation_space** | 63 | root_lin_vel_b(3) + root_ang_vel_b(3) + gravity_b(3) + cmd(2+1) + joint_pos_offset(17) + joint_vel(17) + actions(17) |
| **action_space** | 17 | 관절 토크 (4족 + 허리) |
| **amp_observation_space** | 59 | per-step disc obs: dof_pos(17) + dof_vel(17) + root_height(1) + lin_vel(3) + ang_vel(3) + foot_pos(12) + rot_tan_norm(6) |
| **num_amp_observations** | 10 | 판별자 히스토리 깊이 (59 × 10 = 590 입력) |
| **scene.num_envs** | 4096 | 병렬 환경 |
| **episode_length_s** | 10.0 | 에피소드 지속시간 |
| **sim_dt_hz** | 200 | 물리 시뮬레이션 |
| **policy_dt_hz** | 50 | 정책 제어 (decimation=4) |

### 4.1 RMA 환경 obs_groups

**파일:** `leg_imitation_tracking/agents/rsl_rl_ppo_cfg.py:160-166`

```python
obs_groups = {
    "policy": ["policy"],                      # 57 dims
    "critic": ["policy", "priv_explicit", "priv"],  # 57+6+38=101
    "history": ["history"],                    # (10, 57)
    "priv": ["priv"],                          # 38 dims
    "priv_explicit": ["priv_explicit"],        # 6 dims
}
```

**priv_latent 구성 (38 dims, episode-constant):**
`leg_imitation_tracking_rma_env.py:87-107`
```python
base_mass(1) + base_com(3) + joint_stiffness_ratio(17) + joint_damping_ratio(17)
```
모두 에피소드 초기화 시 DR로 샘플링되며, 에피소드 중 변하지 않음.

---

## 5. ⭐ 중요 발견: RMA Actor의 시간 정보 라우팅

**결론:** **RMA actor는 사실상 memoryless이다. PPO는 시간 정보를 최적화하지 않는다.**

### 5.1 학습 롤아웃: 95% priv_latent, 5% history_latent

**파일:** `rsl_rl/runners/on_policy_runner_parkour_amp.py:110`

```python
hist_encoding = it % 20 == 0  # DAGGER 주기
```

**의미:**
- 반복 `it % 20 == 0`일 때만 (5%) `act(obs, hist_encoding=True)` → history_latent 경로
- 나머지 95%: `act(obs, hist_encoding=False, ...)` (암묵적 기본값) → priv_latent 경로

### 5.2 PPO 손실: priv_latent 경로만 최적화

**파일:** `rsl_rl/algorithms/ppo_parkour.py:325`

```python
self.policy.act(obs_batch, masks=masks_batch, hidden_state=...)
# hist_encoding 파라미터 ABSENT → False 기본값
```

**PPO surrogate 손실** (line 384-396):
```python
ratio = torch.exp(actions_log_prob_batch - old_actions_log_prob_batch)
surrogate_loss = torch.max(...).mean()  # 이 gradient는 priv_latent 경로만 통과
```

**별도 DAGGER 업데이트** (line 646-662):
```python
self.policy.act(obs_batch, hist_encoding=True, ...)  # history_latent 경로만
hist_latent_loss = (priv_latent_batch.detach() - hist_latent_batch).norm(...)
# history_encoder만 역전파 (actor MLP NOT 포함)
```

### 5.3 배포 추론: 항상 history_latent (train/eval mismatch)

**파일:** `rsl_rl/runners/on_policy_runner_parkour.py:257`

```python
def get_inference_policy(self, device):
    return self.alg.policy.act_inference  # ← 배포용
```

**act_inference 구현:** `rsl_rl/modules/actor_cricket_parkour.py:305-320`

```python
def act_inference(self, obs: TensorDict) -> torch.Tensor:
    obs_actor = self.actor_obs_normalizer(self.get_actor_obs(obs))
    priv_explicit = self.priv_explicit_obs_normalizer(self.get_priv_explicit_obs(obs))
    history_latent = self.get_hist_latent(obs)  # ← 항상 사용
    obs_actor = torch.cat([obs_actor, priv_explicit, history_latent], dim=-1)
    ...
    return self.actor(obs_actor)
```

**Train/eval 불일치:**
```
학습:    95% priv_latent (정적, episode-constant)
        5% history_latent (시간 동적)
        PPO 그래디언트: 100% priv_latent 경로

배포:    100% history_latent (시간 동적)
```

play 스크립트 (`_workspace/leg/speed_ramp_record_rma.py:227, 340`)도 동일하게 `act_inference` 사용.

### 5.4 신규: act_inference_priv() 메소드 검증

**파일:** `rsl_rl/modules/actor_cricket_parkour.py:322-343` (새로 추가됨)

```python
def act_inference_priv(self, obs: TensorDict) -> torch.Tensor:
    """학습 시간과 동일한 입력(priv_latent)을 사용하는 결정적 추론."""
    obs_actor = self.actor_obs_normalizer(self.get_actor_obs(obs))
    priv_explicit = self.priv_explicit_obs_normalizer(self.get_priv_explicit_obs(obs))
    obs_actor = torch.cat([obs_actor, priv_explicit, self.get_priv_latent(obs)], dim=-1)
    # ...
    return self.actor(obs_actor)
```

**검증 결과: ⚠️ 불완전 (minor discrepancy)**

학습 `act(hist_encoding=False)` (line 283-303) 대비:

| 구성요소 | 학습 | act_inference_priv | 일치 |
|---------|------|-------------------|------|
| `get_actor_obs` 정규화 | ✓ line 285 | ✓ line 333 | ✓ |
| `get_priv_explicit_obs` 정규화 | ✓ line 286 | ✓ line 334 | ✓ |
| `get_priv_latent` (정규화 아님) | ✓ line 291 | ✓ line 335 | ✓ |
| `scandot_encoder` 연결 | ✓ line 294-297 | ✓ line 337-338 | ✓ |
| **nan_to_num 호출** | ✓ line 301 | ❌ 없음 | ❌ **불일치** |
| `state_dependent_std` 처리 | ✓ line 317-320 | ✓ line 340-343 | ✓ |

**발견된 문제:**
학습에서는 `torch.nan_to_num(obs_actor, nan=0.0)`을 line 301에서 호출하지만, `act_inference_priv`는 이 호출을 **빠뜨렸다**.
- 이것은 first-frame 센서 오류나 물리 불안정성에서 NaN이 발생할 경우, actor 입력이 다를 수 있음을 의미한다.
- forward pass 결과가 변할 수 있으므로, 현재 실행 중인 ramp A/B 비교가 **약간 오염**될 수 있다.
- **권장:** `act_inference_priv` line 339 앞에 `obs_actor = torch.nan_to_num(obs_actor, nan=0.0)` 추가.

### 5.5 속도 램프 플래그

**파일:** `_workspace/leg/speed_ramp_record_rma.py:28-32`

```python
parser.add_argument(
    "--use_priv_latent",
    action="store_true",
    help="actor 입력을 history latent 대신 privileged latent로 준다(sim 전용)..."
)
```

**사용처:** line 235-240

```python
if args_cli.use_priv_latent:
    policy = runner.alg.policy.act_inference_priv
else:
    policy = runner.get_inference_policy(device=env.unwrapped.device)
```

**평가:** ✓ 정확함 (위 NaN 문제 제외).

---

## 6. AMP Runner & 정책 구성

**파일:** `rsl_rl/runners/on_policy_runner_amp.py:21-91`

### 6.1 정책 구성 (_construct_algorithm)

```python
actor_critic = actor_critic_class(obs, self.cfg["obs_groups"],
                                   self.env.num_actions, **self.policy_cfg)
```

- `obs`: 환경으로부터 얻은 초기 obs TensorDict
- `obs_groups`: cfg의 dict, 관찰 라우팅 지정
- `**self.policy_cfg`: 활성화, 숨김 차원, 정규화 플래그 등 언팩

### 6.2 Estimator 구성

**조건:** `estimator_cfg` ≠ None (line 70)

```python
estimator = Estimator(
    input_dim=obs["policy"].shape[-1],
    output_dim=sum(obs[k].shape[-1] for k in obs_groups["priv_explicit"]),
    hidden_dims=estimator_cfg["hidden_dims"],
    activation=self.policy_cfg.get("activation", "elu"),
)
```

- Leg-Imitation-Tracking-RMA: input=57 (policy), output=6 (priv_explicit)
- 파일: `leg_imitation_tracking/agents/rsl_rl_ppo_cfg.py:168-173`

### 6.3 Discriminator 입력 차원 자동 감지

**파일:** `on_policy_runner_amp.py:59-62`

```python
if hasattr(self.env.unwrapped, "amp_observation_space"):
    amp_cfg["amp_observation_space"] = self.env.unwrapped.amp_observation_space.shape[0]
```

cfg 값은 무시되고, 런타임에 env에서 읽음.

---

## 7. 대칭 Augmentation & 정책 상호작용

**파일:** `source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/mdp/symmetry.py:186-236`

### 7.1 함수 시그니처

```python
@torch.no_grad()
def compute_leg_symmetric_states(*, env, obs: TensorDict | None = None,
                                   actions: torch.Tensor | None = None):
    """num_aug=2로 관찰/동작 L/R 미러링"""
    # 반환: ([original B; mirrored B], [...])
```

### 7.2 호환성

| 정책 클래스 | 호환 | 이유 |
|----------|------|------|
| **ActorCritic** | ✓ | 순수 피드포워드; 미러링은 텐서 변환만 |
| **ActorCriticRMA** | ✓ | 적응 모듈도 상태 비의존적 |
| **ActorCriticRecurrent** | ❌ | 히든 상태가 미러링되지 않음 (히스토리 불일치) |

### 7.3 RMA 에서 대칭 aug 활성화

**파일:** `leg_imitation_tracking/agents/rsl_rl_ppo_cfg.py:243-276` (__post_init__)

```python
self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
    use_data_augmentation=True,
    data_augmentation_func="isaaclab_tasks.direct.leg_imitation_tracking.mdp.symmetry:compute_leg_symmetric_states",
    use_mirror_loss=False,  # 기본값; env var LEG_MIRROR_LOSS=1로 활성화
)
```

---

## 8. 정책 Export & 배포

### 8.1 Standard Export (actor만)

**호출:** `runner.get_inference_policy()` → actor MLP 단독

**제약:**
- Estimator/history_encoder 미포함
- RMA의 경우 estimator 필요하므로 부족

### 8.2 전체 배포 Wrapper

**파일:** `scripts/real2sim/export_deployable_bipedleg.py:41-63`

```python
class DeployablePolicy(nn.Module):
    def __init__(self, policy, estimator):
        self.actor = policy.actor
        self.history_encoder = policy.history_encoder
        self.estimator = estimator
        ...

    def forward(self, proprio, history):
        obs_actor = self.actor_obs_normalizer(proprio)
        priv_explicit = self.estimator(proprio)
        history_latent = self.history_encoder(history)
        return self.actor(torch.cat([obs_actor, priv_explicit, history_latent], dim=-1))
```

**Export:** `torch.jit.script(DeployablePolicy)`

**제약:** RNN (ActorCriticRecurrent) 미지원 (TorchScript LSTM/GRU 제약).

---

## 9. ⭐ 아키텍처 후보 평가

### 9.1 평가 기준

| 기준 | 정의 |
|------|------|
| **구현됨** | 현재 코드베이스에 존재 |
| **최소 변경** | 필요한 파일과 줄수 (0 = cfg만, ~N = 코드 변경) |
| **Export** | TorchScript/ONNX 호환성 |
| **대칭 Aug** | `compute_leg_symmetric_states` 호환 |
| **AMP Runner** | OnPolicyRunnerAMPBase/ParkourAMP 지원 |
| **비용** | 개발/테스트 난이도 |

### 9.2 후보별 판정

#### Candidate 1: Deeper/Wider MLP (예: [1024,1024,512,256])

| 평가 | 결과 |
|------|------|
| **구현됨** | ✓ YES (trivial) |
| **최소 변경** | 0 줄 — cfg만 수정: `actor_hidden_dims=[1024, 1024, 512, 256]` |
| **Export** | ✓ YES (순수 Linear → activation sequence) |
| **대칭 Aug** | ✓ YES (모델과 무관) |
| **AMP Runner** | ✓ YES (표준 경로) |
| **비용** | ✓ **극저** |

**평가:** ✅ **권장** — 즉시 테스트 가능, 리스크 없음.

---

#### Candidate 2: Residual Blocks + LayerNorm (Pre-LN 형식)

| 평가 | 결과 |
|------|------|
| **구현됨** | ❌ NO |
| **최소 변경** | ~50–100 줄 (새 클래스 `MLPWithResiduals` + `ResidualBlock`) 추가 위치: `rsl_rl/modules/mlp.py` 뒤 |
| **Export** | ✓ YES (LayerNorm + 덧셈은 scriptable) |
| **대칭 Aug** | ✓ YES (상태 무관) |
| **AMP Runner** | ✓ YES (표준 경로) |
| **비용** | ⚠️ **중간** — residual 구현, actor-critic 호출점 변경 필요 |

**세부:**
```python
# rsl_rl/modules/mlp.py에 추가
class ResidualBlock(nn.Module):
    def __init__(self, dim, activation):
        self.ln = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = activation

    def forward(self, x):
        return x + self.activation(self.linear(self.ln(x)))

class MLPWithResiduals(nn.Module):
    # residual blocks 내장
```

actor-critic.py line 60, 73 등에서 `MLP(...)`를 `MLPWithResiduals(...)`로 변경.

**평가:** ✓ **실행 가능** — 개발 비용 중간, 효과 불확실.

---

#### Candidate 3: ActorCriticRecurrent (GRU/LSTM)

| 평가 | 결과 |
|------|------|
| **구현됨** | ✓ YES (`actor_critic_recurrent.py`) |
| **최소 변경** | 1 필드 (cfg: `class_name="ActorCriticRecurrent"` + rnn 파라미터) |
| **Export** | ❌ **NO** — TorchScript LSTM/GRU 미지원 (torch.jit.script 실패) |
| **대칭 Aug** | ❌ **NO** — 히든 상태 미러링 불가 (sym aug 설정에서 env var로 비활성화 강제) |
| **AMP Runner** | ⚠️ **부분** — 지원하지만, 부분 done의 히든 상태 리셋 미흡 |
| **비용** | ❌ **극고** (export 불가 = 배포 경로 재설계) |

**근거:**
- Export (line 139, `export_deployable_bipedleg.py`): `torch.jit.script(deploy)` 실패 시 대체 경로 필요
- 대칭 aug (line 243–265, `leg_imitation_tracking_rma_env.py`): `symmetry_cfg = None`로 강제 비활성화되거야 할 것
- 학습 (line 110–115): 95% hist_encoding=False 경로에서만 히든 상태 활용 → 학습 신호 극약

**평가:** ❌ **기각** — Export, 대칭, 학습 모두 문제. Leg-Imitation 불가능.

---

#### Candidate 4: Mixture-of-Gaussians Action Head (K components)

| 평가 | 결과 |
|------|------|
| **구현됨** | ❌ NO |
| **최소 변경** | ~200 줄 (new distribution class + actor-critic 호출점) 위치: `rsl_rl/modules/` + `actor_critic_parkour.py` 5+ 사이트 |
| **Export** | ⚠️ **조건부** — torch ops로 구현 시 scriptable, 그러나 배포 추론 복잡화 |
| **대칭 Aug** | ✓ YES (분포 선택은 정책 내부) |
| **AMP Runner** | ✓ YES (표준 경로) |
| **비용** | ❌ **극고** — 분포 설계, .mean/.entropy/.log_prob 모두 재정의, 배포 추론 규칙 선택 필요 |

**문제점:**
- `.action_mean` 속성 모호 (최빈값? 성분 평균?)
- PPO loss line 318 (`entropy.mean()`) 모호화
- Export line 63 (`self.actor(obs_actor)`) 분포 디코딩 추가 필요

**평가:** ❌ **기각** — 이익 대비 복잡도 극고. 우선순위 낮음.

---

#### Candidate 5: 분리된 Actor/Critic 용량 (big critic, small actor)

| 평가 | 결과 |
|------|------|
| **구현됨** | ✓ YES (이미 독립적) |
| **최소 변경** | 0 줄 — cfg만: `actor_hidden_dims=[256,128]`, `critic_hidden_dims=[1024,512]` |
| **Export** | ✓ YES (export는 actor만 사용) |
| **대칭 Aug** | ✓ YES (상태 무관) |
| **AMP Runner** | ✓ YES (표준 경로) |
| **비용** | ✓ **극저** |

**평가:** ✓ **trivial** — 즉시 테스트 가능.

---

### 9.3 최종 권장 3-arm 배분

```
Arm A (기준):        [512, 256, 128] actor/critic (현재)
Arm B (깊이):        [1024, 1024, 512, 256] actor/critic (Candidate 1)
Arm C (구조):        [512, 512, 512] + Pre-LN residual blocks (Candidate 2)
```

모두:
- ✓ Export 가능 (JIT/ONNX)
- ✓ 대칭 aug 호환
- ✓ AMP runner 호환
- ✓ 0 또는 중간 개발 비용
- ✓ Leg-Imitation-Tracking-RMA-v0에서 즉시 테스트 가능

---

## 10. noise_std 구현 세부

### 10.1 학습 가능 파라미터

**파일:** `rsl_rl/modules/actor_cricket.py:83-101`

| 타입 | 구현 | 로깅 포인트 |
|------|------|-----------|
| **상태 독립 scalar** | `self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))` (line 97) | `distribution.stddev` (property line 122) |
| **상태 독립 log** | `self.log_std = nn.Parameter(torch.log(...))` (line 99) | `exp(self.log_std)` → stddev |
| **상태 의존 head** | Actor MLP 마지막이 `[mean, log_std]` 출력 (line 60) | actor[-2] 출력 (line 86–92 init) |

### 10.2 ActorCriticRMA 확장

**파일:** `rsl_rl/modules/actor_cricket_parkour.py:215-229`

추가 옵션: `noise_std_type="fixed"`
```python
self.register_buffer("std", init_noise_std * torch.ones(num_actions))
# optimizer 터치 안 함, state_dict 저장/복원
```

### 10.3 로깅 없음

현재 코드베이스에서 std 궤적 자동 로깅 없음. 필요 시 추가:
```python
self.logger.log("policy/std_mean", torch.mean(self.actor.std).item())
```

---

## 11. Adaptive LR & Schedule

**파일:** `rsl_rl/algorithms/ppo.py:52-57, 274-299`

### 11.1 적응 로직

```python
schedule: str = "adaptive"  # 기본
desired_kl: float = 0.01

if self.desired_kl is not None and self.schedule == "adaptive":
    kl_mean = ...
    if kl_mean > desired_kl * 2.0:
        learning_rate = max(1e-5, learning_rate / 1.5)
    elif kl_mean < desired_kl / 2.0:
        learning_rate = min(1e-2, learning_rate * 1.5)
```

### 11.2 상수 LR 스위치

```python
schedule="fixed"  # cfg에서 지정
# → adaptive 블록 스킵, learning_rate 불변
```

**Leg-Imitation-Tracking-RMA cfg:** line 198, `schedule="adaptive"` (기본값).

상수 LR 테스트:
```python
algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
    schedule="fixed",  # 변경
    learning_rate=2e-4,
    ...
)
```

---

## 12. 계산 비용 분석

### 12.1 롤아웃 vs 그래디언트

**Per iteration:**
```
Rollout:      4096 envs × 24 steps = 98,304 전환
Policy update: 98,304 batch
              5 epochs × 4 minibatches
              = 491,520 sample forward+backward
              (약 5× batch size)
```

**벽시계 분포:**
- Rollout: ~15–20% (GPU 시뮬레이션 병목)
- Update (그래디언트): ~80–85%

### 12.2 네트워크 크기 영향

| 변경 | Rollout | Update | 총합 |
|------|---------|--------|------|
| 2× 깊이 | +10% | +50–100% | **+40–60%** |
| 2× 너비 | +10% | +50–100% | **+40–60%** |

**결론:** 네트워크 크기 증가는 **주로 PPO update에 영향** (80% 집중).

### 12.3 예상 반복 시간

```
현재 (mid-size):  ~4–7s per iteration
2× 크기:          ~7–14s per iteration

50,000 iter:
  현재: 55–97시간
  2× 크기: 97–194시간
```

---

## 13. 주요 파일 참고 목록

| 목적 | 파일 | 줄수 |
|------|------|------|
| Actor-Critic base | `rsl_rl/modules/actor_critic.py` | 22–238 |
| RMA variant | `rsl_rl/modules/actor_cricket_parkour.py` | 81–500 |
| MLP builder | `rsl_rl/modules/mlp.py` | 21–101 |
| Runner (AMP) | `rsl_rl/runners/on_policy_runner_amp.py` | 21–91 |
| PPO algorithm | `rsl_rl/algorithms/ppo_parkour.py` | 181–415 |
| Env cfg | `leg_imitation_tracking_env_cfg.py` | 39–72 |
| Agent cfg | `agents/rsl_rl_ppo_cfg.py` | 31–277 |
| Symmetry | `mdp/symmetry.py` | 186–236 |
| Export | `scripts/real2sim/export_deployable_bipedleg.py` | 41–63 |
| Play/Ramp | `_workspace/leg/speed_ramp_record_rma.py` | 227–340 |

---

## 14. 결론

### 14.1 아키텍처 확장 최소 경로

**새 variant 추가:**
1. `MLPWithResiduals` 클래스 작성 (선택사항, residual만 필요)
2. cfg에서 `actor_hidden_dims` 변경
3. cfg에서 `class_name` 지정 (기존="ActorCritic", 새="ActorCriticRMA")
4. 표준 runner/algorithm 사용 (변경 불필요)

**검증:**
- `./isaaclab.sh train --rl_library rsl_rl --task Leg-Imitation-Tracking-RMA-v0 --agent <cfg> --max_iterations 50000`
- TensorBoard `Algorithm/hist_latent_loss` 모니터 (DAGGER 오류)
- Speed ramp: `--use_priv_latent` 플래그로 train/eval 분리 비교

### 14.2 가장 놀라운 발견

**RMA actor는 시간 정보를 배우지 않는다:**
- PPO 최적화: 95% priv_latent (episode-constant, static)
- 배포 추론: 100% history_latent (temporal)
- History encoder: DAGGER 증류 대상일 뿐, PPO 그래디언트 미수신

**함의:** 시간 구조를 actor에 추가하려면 (예: RNN), 단순히 `class_name="ActorCriticRecurrent"`만으로는 부족. `hist_encoding = True` (항상) 또는 raw history 직접 입력 필요. 아니면 env에 gait-phase 신호 추가.

---

**문서 작성자:** Claude AI 분석 (codebase-arch-map) | **확인일:** 2026-08-26

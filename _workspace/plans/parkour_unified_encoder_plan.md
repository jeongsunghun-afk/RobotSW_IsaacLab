# Parkour Unified History Encoder — 구현 계획서

**상태**: 설계 확정, **구현 보류 (사용자 트리거 대기)**
**작성일**: 2026-05-21
**Stage 1 검증 완료**: validate-method (CONDITIONAL PASS) + Explore (코드 스펙 수집)
**참조 task list**: `~/.claude/tasks/<team>/3-7.json` (#3~#7 pending, 의존성 설정 완료)

---

## 0. 한 줄 요약

현재 parkour의 **single-step estimator + 별도 history_encoder + DAGGER** 3-track 구조를, **shared history trunk + explicit head (vel + contact, supervised) + latent head (DAGGER 유지)** 로 통합한 신규 알고리즘. **새 파일로 분리** (`*_parkour_unified.py`)하여 기존 v0와 자유 전환. env cfg에 **3그룹 bool selector** (PolicyObsCfg / PrivExplicitObsCfg / PrivLatentObsCfg) 도입하여 obs 구성 ablation 가능.

---

## 1. 배경 — 왜 이 설계인가

### 1.1 현재 parkour 구조의 한계
- `estimator`: single-step proprio(42) → vel(6) MSE. **Contact 예측 불가** (touchdown/liftoff = 시간 미분 사건, 한 프레임으로 신호 없음).
- `history_encoder`: 10-step history → 20-dim latent, DAGGER로 priv_encoder(priv_latent)와 매칭. 비지도(implicit) 트랙.
- 두 트랙이 **독립적** → 파라미터/representation 중복, contact 같은 시간 신호를 어디에도 못 짜넣음.

### 1.2 새 설계의 명시적 가치
1. **Contact 예측 가능해짐**: 10-step history는 contact 추출에 충분 (Concurrent Training Ji et al. 2022 선례)
2. **Shared trunk**: explicit(vel/contact supervised) + implicit(DAGGER) 양쪽 그래디언트로 정규화 → 더 풍부한 representation
3. **DAGGER 유지**: DR adaptation(friction/mass/com) 신호 보존 → sim-to-real 능력 유지
4. **Sim-to-real 양립**: deploy 시 contact 센서 불필요 (estimator output 사용). 메모리 §1(Contact obs 금지) 통과.

### 1.3 문헌 정합성 (validate-method 결과)
- **가장 가까운 선례**: HIM (Long et al. 2023, arXiv:2312.11460) — history → shared MLP trunk → explicit vel + implicit stability head, **별도 optimizer**, loss scale=1.0
- 추가 참조: Concurrent Training (Ji et al. 2022, 2202.05481), PCGrad (2001.06782), Robust Robot Walker (2409.07409)
- WTW/Extreme Parkour 상세 비교는 PDF 직접 확인 필요 (논문 abstract로 미검증)

---

## 2. 확정 아키텍처

```
                      history [N, 10, 42]
                              ↓
                      ┌───────────────┐
                      │  SharedTrunk  │  ← StateHistoryEncoder 재사용 (Conv1d 스택)
                      │  (history_enc)│  → trunk_features [N, trunk_dim]
                      └───────┬───────┘
                              │
              ┌───────────────┴───────────────┐
              ↓                               ↓
       ┌─────────────┐                ┌─────────────┐
       │explicit_head│                │ latent_head │
       │  (MLP)      │                │   (MLP)     │
       └──────┬──────┘                └──────┬──────┘
              ↓                               ↓
   (v̂_lin(3), v̂_ang(3),               z (20-dim latent)
    ĉ_contact_logits(4))                       │
              │                                ↓
              │ supervised:               supervised:
              │  MSE(vel)                  DAGGER L2 vs
              │  BCEWithLogits(contact)    priv_encoder(priv_latent).detach()
              │                            + RL gradient (actor 통해)
              │                                │
              └───────────────┬────────────────┘
                              ↓
                  ┌───────────────────────┐
                  │   actor input cat:     │
                  │ [proprio_t(42),        │
                  │  v̂(6), σ(ĉ)(4),        │   ← contact은 sigmoid 거쳐 [0,1]
                  │  z(20),                │
                  │  scan_latent(32)]      │
                  │  = [N, 114]            │
                  └───────────────────────┘

critic: 기존 그대로 (asymmetric, full priv info)
```

### 2.1 손실 함수
```
L_PPO     = surrogate + value_coef × value_loss + entropy_coef × entropy
L_explicit = MSE(v̂, v_true) + BCE_with_logits(ĉ_logits, c_true)
            └── (vel target은 priv_explicit[:, :6] 기존대로)
            └── (contact target은 신규 obs_dict["contact_fill"] {0,1})
L_DAGGER  = ((z - priv_encoder(priv_latent).detach()) ** 2).mean()

# 별도 optimizer (HIM 표준, K_epoch 증폭 회피)
unified_optimizer.step(): L_explicit + dagger_weight × L_DAGGER
ppo_optimizer.step()    : L_PPO
priv_encoder_optimizer.step(): trainable 유지 (DAGGER target 생성)
```

**aux 업데이트 빈도**: rollout당 1회 pass (PPO inner K-epoch loop 밖). 이로써 gradient ×K 증폭 회피.

### 2.2 핵심 결정 5가지 (validate-method 권고 반영)

| # | 결정 | 근거 |
|---|---|---|
| 1 | aux는 별도 optimizer, PPO K-epoch 밖 1회 update | HIM/Concurrent Training 표준. K_epoch ×5 증폭 회피 |
| 2 | Contact target: `{0,1}` + `BCEWithLogitsLoss` | 확률론적 표준, 수치 안정 |
| 3 | aux_loss_weight = 1.0, dagger_weight = 1.0 시작 | HIM 보고값 |
| 4 | `explicit_pred → actor` 시 **no-detach** | HIM 표준. actor가 estimator 품질 향상에 기여 |
| 5 | latent collapse 모니터링: `std(z)` per-dim logging | 전체 dim < 0.01이면 trunk가 supervision에 독점됨 |

---

## 3. 파일 변경 계획

### 3.1 신규 파일 (3개 모듈 + 1개 cfg)

| 파일 | 역할 |
|---|---|
| `rsl_rl/rsl_rl/modules/actor_critic_parkour_unified.py` | `ActorCriticParkourUnified` + `UnifiedHistoryEncoder` 클래스 |
| `rsl_rl/rsl_rl/algorithms/ppo_parkour_unified.py` | `PPOParkourUnified` (aux loss + 별도 optimizer + DAGGER 유지) |
| `rsl_rl/rsl_rl/runners/on_policy_runner_parkour_unified.py` | `OnPolicyRunnerParkourUnified` |
| `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/agents/rsl_rl_ppo_unified_cfg.py` | `Go2ParkourUnifiedPPORunnerCfg` (class_name=Unified 시리즈, aux_weight, pos_weight 등) |

### 3.2 기존 파일 수정 (env 측)

| 파일 | 변경 내용 |
|---|---|
| `parkour_env_cfg.py` | 3개 그룹 cfg 클래스 추가 (`PolicyObsCfg`, `PrivExplicitObsCfg`, `PrivLatentObsCfg`) + `ParkourEnvCfg`에 멤버 추가. 기본값 = 현 behavior |
| `parkour_env.py` | `_get_observations`에서 flag 기반 동적 cat. `__init__`에서 `num_proprio/num_priv_explicit/num_priv_latent` 동적 계산. `obs_dict["contact_fill"]` 신규 key 추가 (policy obs 미오염) |
| `parkour/__init__.py` | `Isaac-Parkour-Unified-v0` (or `Go2-Parkour-Unified-Direct-v0`) 신규 task 등록 |

### 3.3 절대 손대지 않을 파일
- `actor_critic_parkour.py`, `ppo_parkour.py`, `on_policy_runner_parkour.py`, `estimator.py`
- `agents/rsl_rl_ppo_cfg.py` (기존 v0 cfg)
- `Go2-Parkour-Direct-v0` 등록 (기존)

→ **기존 task `Go2-Parkour-Direct-v0` 학습 결과 100% 불변** 보장.

---

## 4. cfg 3그룹 상세 spec

### 4.1 `parkour_env_cfg.py`에 추가할 클래스

```python
@configclass
class PolicyObsCfg:
    """policy obs(proprio) 구성요소 toggle. UnifiedEncoder의 history도 이 구성을 따름."""
    include_yaw_diff: bool = True           # 1-dim
    include_next_yaw_diff: bool = True      # 1-dim
    include_projected_gravity: bool = True  # 3-dim
    include_commands: bool = True           # 1-dim (현재 commands[:, 0:1])
    include_joint_pos_delta: bool = True    # 12-dim
    include_joint_vel: bool = True          # 12-dim (×0.05 scale)
    include_actions: bool = True            # 12-dim

@configclass
class PrivExplicitObsCfg:
    """priv_explicit (supervised estimation target) 구성요소 toggle."""
    include_lin_vel: bool = True           # 3-dim (×2.0)
    include_ang_vel: bool = True           # 3-dim (×0.25)
    include_contact_fill: bool = False     # NEW: 4-dim {0,1}, Unified에서 True로 override

@configclass
class PrivLatentObsCfg:
    """priv_latent (DAGGER target via priv_encoder) 구성요소 toggle."""
    include_foot_friction: bool = True      # 8-dim
    include_base_mass: bool = True          # 1-dim
    include_base_com: bool = True           # 3-dim

@configclass
class ParkourEnvCfg(...):
    # 기존 필드들 유지
    policy_obs:        PolicyObsCfg        = PolicyObsCfg()
    priv_explicit_obs: PrivExplicitObsCfg  = PrivExplicitObsCfg()
    priv_latent_obs:   PrivLatentObsCfg    = PrivLatentObsCfg()
```

### 4.2 env._get_observations 동적 cat 로직

현재 (line 852-864):
```python
proprio = torch.cat([self._yaw_diff[:, None], self._next_yaw_diff[:, None], ...], dim=-1)
```

→ 신규: helper 함수 도입
```python
def _build_proprio(self) -> torch.Tensor:
    parts = []
    cfg = self.cfg.policy_obs
    if cfg.include_yaw_diff:           parts.append(self._yaw_diff[:, None])
    if cfg.include_next_yaw_diff:      parts.append(self._next_yaw_diff[:, None])
    if cfg.include_projected_gravity:  parts.append(self._robot.data.projected_gravity_b)
    if cfg.include_commands:           parts.append(self._commands[:, 0:1])
    if cfg.include_joint_pos_delta:    parts.append(self._robot.data.joint_pos - self._robot.data.default_joint_pos)
    if cfg.include_joint_vel:          parts.append(self._robot.data.joint_vel * 0.05)
    if cfg.include_actions:            parts.append(self._actions)
    return torch.cat(parts, dim=-1)

# 동일 패턴: _build_priv_explicit, _build_priv_latent
```

### 4.3 dim 동적 계산 (`__init__`)

```python
# parkour_env.py __init__ 에서
self.cfg.num_proprio = self._compute_proprio_dim()
self.cfg.num_priv_obs = self._compute_priv_obs_dim()  # priv_explicit + priv_latent
```

contact_fill은 priv_explicit에 포함시 +4, 미포함시 0.

### 4.4 backward compat 보장
- 모든 bool 기본값 = True (단 `include_contact_fill = False`)
- Unified cfg에서만 `priv_explicit_obs.include_contact_fill = True` override
- 기존 `Go2-Parkour-Direct-v0`는 dim 변화 없음 → 학습 결과 동일

---

## 5. 알고리즘 측 spec

### 5.1 `UnifiedHistoryEncoder` (신규 클래스)

```python
class UnifiedHistoryEncoder(nn.Module):
    def __init__(
        self,
        num_proprio: int,
        num_history: int = 10,
        explicit_vel_dim: int = 6,
        explicit_contact_dim: int = 4,
        latent_dim: int = 20,
        trunk_dims: list[int] = [128, 64],
        head_dims: list[int] = [64],
    ):
        super().__init__()
        # trunk: 기존 StateHistoryEncoder 재사용 또는 동일 패턴 (Conv1d 스택)
        self.trunk = StateHistoryEncoder(
            num_proprio=num_proprio, tsteps=num_history,
            output_size=trunk_dims[-1],
        )
        # heads
        self.explicit_head = build_mlp(trunk_dims[-1], head_dims, explicit_vel_dim + explicit_contact_dim)
        self.latent_head   = build_mlp(trunk_dims[-1], head_dims, latent_dim)
        self.explicit_vel_dim = explicit_vel_dim
        self.explicit_contact_dim = explicit_contact_dim

    def forward(self, history_flat):
        # history_flat: [N, num_history * num_proprio]
        trunk_feat = self.trunk(history_flat)  # [N, trunk_dims[-1]]
        explicit = self.explicit_head(trunk_feat)  # [N, vel_dim + contact_dim]
        latent   = self.latent_head(trunk_feat)    # [N, latent_dim]
        return explicit, latent

    def split_explicit(self, explicit):
        """vel은 raw, contact은 logits (BCEWithLogits용). actor에 줄 때는 sigmoid 적용."""
        vel    = explicit[:, :self.explicit_vel_dim]
        c_logits = explicit[:, self.explicit_vel_dim:]
        return vel, c_logits
```

### 5.2 `ActorCriticParkourUnified` (신규 클래스)

기존 `ActorCriticRMA` 베이스로:
- `self.estimator` 제거
- `self.history_encoder = UnifiedHistoryEncoder(...)`
- `self.priv_encoder` 유지 (DAGGER target 생성용)
- `act()` 메서드:
  ```python
  def act(self, obs, hist_encoding=True):
      proprio = self.actor_obs_normalizer(self.get_actor_obs(obs))
      history_flat = self.get_history_obs(obs).reshape(N, -1)
      explicit, latent = self.history_encoder(self.history_obs_normalizer(history_flat))
      vel, c_logits = self.history_encoder.split_explicit(explicit)
      explicit_for_actor = torch.cat([vel, torch.sigmoid(c_logits)], dim=-1)
      
      if hist_encoding:
          z = latent
      else:
          # 학습 초기 hist_encoding=False phase: priv_encoder 출력 사용
          z = self.priv_encoder(self.get_priv_obs(obs))
      
      actor_input = torch.cat([proprio, explicit_for_actor, z], dim=-1)
      if self.scandot_encoder is not None:
          actor_input = torch.cat([actor_input, self.scandot_encoder(...)], dim=-1)
      return self._update_distribution(actor_input).sample()
  ```

### 5.3 `PPOParkourUnified` (신규 클래스)

기존 `PPOParkour` 베이스로:
- `self.estimator` / `self.estimator_optimizer` 제거
- `self.unified_optimizer = Adam(self.policy.history_encoder.parameters(), lr=unified_lr)`
- `self.hist_encoder_optimizer` 제거 (unified가 흡수)
- `self.priv_encoder_optimizer` 유지 (DAGGER target 학습용; 기존 동일)
- `update()` 메서드 변경:
  ```python
  # 1) 기존 PPO loop (K epochs × minibatches): policy loss만
  for epoch in range(K_epochs):
      for mb in storage.mini_batch_generator(...):
          loss = surrogate + value + entropy
          ppo_optimizer.step(loss)
  
  # 2) AUX update (1 pass, K_epoch 밖)
  for mb in storage.mini_batch_generator(num_mini_batches=N, num_epochs=1):
      history_flat = mb["history"].reshape(N, -1)
      explicit, latent = self.policy.history_encoder(history_flat)
      vel_pred, c_logits = self.policy.history_encoder.split_explicit(explicit)
      
      vel_target = mb["priv_explicit"][:, :vel_dim]
      c_target   = mb["contact_fill"]  # {0,1}
      z_target   = self.policy.priv_encoder(mb["priv_latent"]).detach()
      
      L_vel     = (vel_pred - vel_target).pow(2).mean()
      L_contact = F.binary_cross_entropy_with_logits(c_logits, c_target, pos_weight=pos_w)
      L_dagger  = (latent - z_target).pow(2).mean()
      
      loss = aux_w * (L_vel + L_contact) + dagger_w * L_dagger
      self.unified_optimizer.zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm_(self.policy.history_encoder.parameters(), max_grad_norm)
      self.unified_optimizer.step()
  
  # 3) priv_encoder는 DAGGER target만 만드므로 frozen (or optional update)
  ```

- 로깅 추가: `vel_mse`, `contact_bce`, `dagger_l2`, `std(z)` per-dim mean

### 5.4 `OnPolicyRunnerParkourUnified` (신규 클래스)
- 기존 `OnPolicyRunnerParkour` 베이스로
- estimator 인스턴스화 제거
- `ActorCriticParkourUnified` 인스턴스화 시 unified encoder 자체 포함
- DAGGER update 호출 (rollout 후) 유지

### 5.5 `Go2ParkourUnifiedPPORunnerCfg` (신규 cfg)

```python
@configclass
class Go2ParkourUnifiedPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "OnPolicyRunnerParkourUnified"
    
    obs_groups = {
        "policy":        ["policy"],
        "critic":        ["policy", "scan", "priv_explicit", "priv_latent"],
        "scan":          ["scan"],
        "history":       ["history"],
        "priv":          ["priv_latent"],
        "priv_explicit": ["priv_explicit"],
        "contact_fill":  ["contact_fill"],   # NEW
    }
    
    # unified encoder cfg
    unified_encoder = {
        "trunk_dims": [128, 64],
        "head_dims": [64],
        "latent_dim": 20,
        "learning_rate": 1.0e-3,
    }
    
    policy = ...  # class_name = "ActorCriticParkourUnified"
    algorithm = ...  # class_name = "PPOParkourUnified",
                     # aux_loss_weight: float = 1.0
                     # dagger_loss_weight: float = 1.0
                     # contact_pos_weight: float = 1.0  # class imbalance 시 조정
```

### 5.6 env cfg override
Unified task는 env cfg에서 `priv_explicit_obs.include_contact_fill = True`로 활성화.

---

## 6. 작업 분담 (재개 시 dispatch 순서)

이 계획서 기반으로 worker dispatch:

```
[Stage 2 — 병렬]
  ├─ Task #3 → cfg-worker + obs-worker (env cfg 3그룹 + contact_fill obs export)
  └─ Task #4 → network-worker (actor_critic_parkour_unified.py)

[Stage 3 — 순차]
  ├─ Task #5 → loss-worker (ppo_parkour_unified.py, aux losses, 별도 optimizer)
  └─ Task #6 → cfg-worker (runner + agents/cfg + __init__.py 등록)

[Stage 4 — gate]
  └─ Task #7 → validate-code (dim/buffer/reset 정합성, backward compat)
```

각 worker prompt에 들어갈 핵심 정보는 본 문서 §3, §4, §5 인용.

---

## 7. Risks & 모니터링 (validate-method 결과)

학습 중 아래 시그널 이상 시 **즉시 재설계 검토**:

| 시그널 | 기준 | 해석 |
|---|---|---|
| `vel_mse` 3000 iter 후 plateau > 0.5 m²/s² | < 0.2 기대 | trunk capacity 부족 또는 history 신호 부족 |
| `contact_bce` 지속 > 0.3 | < 0.1 기대 | contact 신호 부족 또는 class imbalance (pos_weight 조정) |
| `std(z)` per-dim < 0.01 collapse | ≥ 0.05 기대 | latent collapse — explicit supervision이 trunk 독점 |
| Policy entropy 평소 대비 2× 빠른 감소 | baseline 대비 | aux gradient가 exploration 조기 억제 |
| `dagger_l2` 상승 (학습 동안) | 감소 기대 | latent_head이 priv_encoder와 발산 — DAGGER weight ↑ 필요 |
| `vel_mse` 하락하나 policy return 정체 | 연동 기대 | aux가 trunk overfitting — RL gradient 배제됨 |

---

## 8. 사용자 트리거 시 즉시 실행 가능한 명령

이 계획 그대로 진행하려면 사용자가 다음 중 하나로 요청:

> "Parkour unified 구현 시작"
> "_workspace/plans/parkour_unified_encoder_plan.md 대로 dispatch"
> "Task #3, #4부터 진행해줘"

→ 메인 agent는:
1. `TaskList`로 #3, #4 in_progress 표시
2. 본 문서 §3, §4, §5의 spec을 prompt에 포함시켜 **cfg-worker + obs-worker + network-worker** 병렬 dispatch
3. 결과 도착 후 #5, #6 순차 dispatch
4. #7 validate-code로 정합성 확인 후 사용자 보고

---

## 9. 참조

- **validate-method 결과 (요약)**: 본 문서 §1.3, §2.2, §7
- **Explore 결과 (요약)**: 본 문서 §3.2, §4.2, §5 (file:line 인용 포함)
- **상세 Explore raw 출력**: 본 세션 메시지 트랜스크립트 (Stage 1 dispatch 응답)
- **현재 task list 항목**: #1, #2 completed / #3~#7 pending with dependencies
- **금지 제약 메모**: `~/.claude/projects/-home-lgb-IsaacLab/memory/project_parkour_analysis_constraints.md` (Contact obs §1 통과 — estimator output은 obs로 OK)

---

## 10. 결정 로그

| 일시 | 결정 | 근거 |
|---|---|---|
| 2026-05-21 | DAGGER 유지 (latent_head 대상으로) | 사용자 질문에서 환기 — DR adaptation 능력 손실 방지 |
| 2026-05-21 | `_unified` suffix, `Isaac-Parkour-Unified-v0` 등록 | 사용자 답변 (1, 2) |
| 2026-05-21 | 3그룹 cfg (PolicyObsCfg/PrivExplicitObsCfg/PrivLatentObsCfg) | 사용자 답변 (3 — "Observation, priv_latent, priv_explicit 모두") |
| 2026-05-21 | validate-method 먼저, 그 다음 구현 | 사용자 답변 (4) |
| 2026-05-21 | aux loss 별도 optimizer | validate-method 권고 (K_epoch 증폭 회피) |
| 2026-05-21 | Contact target `{0,1}` + BCEWithLogits | validate-method 권고 |
| 2026-05-21 | aux_weight=1.0, dagger_weight=1.0 시작 | HIM 표준값 |
| 2026-05-21 | explicit_pred → actor no-detach | HIM 표준 |
| 2026-05-21 | **구현 보류, 사용자 트리거 대기** | 사용자 답변 (Stage 2 진행 전 보류) |

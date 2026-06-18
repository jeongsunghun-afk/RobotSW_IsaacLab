# 구현 계약: A 환경에 B-스타일 Estimator 도입

> 모든 구현 worker는 이 문서를 단일 계약으로 따른다. 차원·키·메서드 시그니처는 추가 추론 없이 그대로 사용.
> 스코프: **옵션 A** — 기존 14D `priv` 텐서만 분리 (EventCfg 미사용 DR 파라미터 신규 추가 안 함).

## 0. 검증된 사실 (코드 확인값)

| 항목 | 값 | 출처 |
|---|---|---|
| `_get_observations()` 반환 키 | `{"policy","scan","priv","history"}` (`critic` 키 주석 처리됨) | `parkour_env.py:921-927` |
| `policy`(proprio) 차원 | 42 | `parkour_env.py:847-859` |
| `priv` 차원 | 14 = `root_lin_vel_b*2.0`(3) + `root_ang_vel_b*0.25`(3) + `foot_friction`(8) | `parkour_env.py:889-896` |
| `scan` 차원 | 187 | `parkour_env.py:902` |
| `history` 차원 | (N, 10, 42) | `parkour_env.py:175-177,926` |

**STALE 경고**: `rsl_rl_ppo_cfg.py:34-45` 주석, `parkour_env_cfg.py:386-392`의 `num_priv_obs=20`/인라인 주석은 STALE — 무시. `ppo_parkour_original.py`/`on_policy_runner_parkour_original.py`는 legacy flat-tensor 버전 — 복붙 금지. 현재 패턴은 TensorDict 기반.

## 1. obs dict 키 설계 (최종)

`_get_observations()` 반환 dict:
```
"policy"        : proprio        (N, 42)   — estimator 입력으로도 사용
"scan"          : self._scan     (N, 187)
"priv_explicit" : base_lin_vel   (N, 3)    — root_lin_vel_b * 2.0
"priv_latent"   : ang_vel+fric   (N, 11)   — root_ang_vel_b*0.25 (3) + foot_friction (8)
"history"       : proprio_history(N, 10, 42)
```
기존 `"priv"` 키 **삭제**, `priv_explicit`+`priv_latent` 두 키로 대체. estimator 입력 proprio = 기존 `"policy"` 키 그대로(별도 키 불필요).

**obs_groups 최종 (`rsl_rl_ppo_cfg.py`)**:
```python
obs_groups = {
    "policy":        ["policy"],
    "critic":        ["policy", "scan", "priv_explicit", "priv_latent"],
    "scan":          ["scan"],
    "history":       ["history"],
    "priv":          ["priv_latent"],      # 그룹 키 이름 "priv" 유지, 값만 priv_latent
    "priv_explicit": ["priv_explicit"],    # 신규 그룹
}
```
그룹 키 `"priv"`는 이름 유지 → `ActorCriticRMA`의 기존 priv_encoder 경로가 이름 변경 없이 priv_latent를 인코딩. `resolve_obs_groups`가 obs_groups 그룹 ↔ env obs dict 키 일치를 검증하므로 obs-worker와 cfg-worker 변경은 함께 머지되어야 함.

## 2. Double-counting 방지 / 인코더 영향
- base_lin_vel을 priv_explicit로 분리 → priv_latent(11D)에 미포함. 중복 없음.
- `priv_encoder` 입력 14→11 자동 변경 (`ActorCriticRMA`가 obs에서 차원 유도). `priv_encoder_dims=[64,20]` 출력 20D 불변.
- history encoder 출력 20D 불변 → `priv_reg_loss`/`update_dagger()` L2 차원 계약 유지.

## 3. Actor 입력 변경 (`actor_critic_parkour.py`)
변경 후 actor 입력: `cat([proprio(42), priv_explicit(3), latent(20), scan_encoded(32)])` = **97D** (기존 94D).
network-worker 변경:
1. `__init__`: `num_priv_explicit`(=obs_groups["priv_explicit"] 그룹 합=3) 계산, `actor_input_dim = num_actor_obs + num_priv_explicit + priv_encoder_dims[-1] + scan_latent_dim`.
2. 신규 셀렉터 `get_priv_explicit_obs(obs)` — 기존 `get_*_obs` 패턴.
3. `act()`: `obs_actor` 조립을 `[actor_obs, priv_explicit, latent, scan]` 순서로. priv_explicit는 normalizer 통과 후 actor_obs 직후 concat.
4. `act_inference()`: 동일하게 priv_explicit concat — **act()와 반드시 동기화**.

## 4. Estimator 알고리즘 연동
### 4.1 인스턴스화
`OnPolicyRunnerParkour._construct_algorithm()`에서 `Estimator(input_dim=42, output_dim=3, hidden_dims=estimator_cfg["hidden_dims"], activation=...)` 생성 → `PPOParkour.__init__`에 전달. `Estimator` 클래스(`rsl_rl/modules/estimator.py`)는 기존 그대로 사용. `on_policy_runner_parkour.py:41` 주석 해제.

### 4.2 PPOParkour `__init__` (`ppo_parkour.py`)
`estimator: nn.Module`, `estimator_cfg: dict` 인자 추가. `self.estimator = estimator.to(device)`, `self.estimator_optimizer = optim.Adam(self.estimator.parameters(), lr=estimator_cfg["learning_rate"])` (PPO optimizer와 완전 분리), `self.train_with_estimated_states = estimator_cfg["train_with_estimated_states"]`.

### 4.3 act() — train_with_estimated_states 치환 (TensorDict 키 교체)
```python
if self.train_with_estimated_states:
    obs_est = obs.clone()
    obs_est["priv_explicit"] = self.estimator(obs["policy"])   # (N,3)
    self.transition.actions = self.policy.act(obs_est, hist_encoding=hist_encoding).detach()
else:
    self.transition.actions = self.policy.act(obs, hist_encoding=hist_encoding).detach()
```
`self.transition.observations = obs` (원본) **변경 금지** — storage에 참값 priv_explicit 보존. `evaluate()`(critic)는 항상 원본 obs.

### 4.4 update() — MSE loss (별도 optimizer)
미니배치 루프 내:
```python
priv_explicit_pred = self.estimator(obs_batch["policy"])
estimator_loss = (priv_explicit_pred - obs_batch["priv_explicit"]).pow(2).mean()
self.estimator_optimizer.zero_grad()
estimator_loss.backward()
nn.utils.clip_grad_norm_(self.estimator.parameters(), self.max_grad_norm)
self.estimator_optimizer.step()
```
estimator_loss는 PPO `loss`에 **미포함**. `mean_estimator_loss` 누적 → `loss_dict["estimator"]`.

### 4.5 save/load (`on_policy_runner_parkour.py`)
`save()`에 `estimator_state_dict`, `estimator_optimizer_state_dict` 추가. `load()`에서 복원.

## 5. cfg 변경 (`rsl_rl_ppo_cfg.py`)
obs_groups를 §1로 교체. `estimator` dict 추가:
```python
estimator = {
    "hidden_dims": [128, 64],
    "learning_rate": 1.0e-3,
    "train_with_estimated_states": False,   # 초기엔 estimator 출력 부정확 → 게이트 off
}
```
STALE 주석(line 34-45) 정리.

## 6. 파일별 변경 + 의존 순서
| Wave | Worker | 파일 | 변경 |
|---|---|---|---|
| 1 | obs-worker | `parkour_env.py` | `_get_observations()` dict: `priv` 삭제 → `priv_explicit`(3)/`priv_latent`(11) |
| 1 | cfg-worker | `rsl_rl_ppo_cfg.py` (+ `parkour_env_cfg.py` 주석) | obs_groups §1, estimator dict §5, STALE 주석 정리 |
| 2 | network-worker | `actor_critic_parkour.py`, `on_policy_runner_parkour.py` | §3 actor 변경, §4.1 estimator 생성, §4.5 save/load |
| 2 | loss-worker | `ppo_parkour.py` | §4.2 __init__, §4.3 act(), §4.4 update() |

## 7. 불변 규칙 체크리스트
- 새 persistent buffer 없음 → `_reset_idx` 수정 불필요. estimator는 stateless module.
- priv_explicit(3)+priv_latent(11)=14 = 기존 priv. critic 합 42+187+3+11=243 = 기존. 값 보존.
- `act()`/`act_inference()` priv_explicit concat 위치 동일 — 동기화 필수.
- estimator loss gradient/optimizer PPO와 완전 분리. clip_grad_norm 대상은 estimator 파라미터만.
- estimator 학습 타깃 = storage의 참값 `obs_batch["priv_explicit"]`.
- critic `evaluate()`는 항상 원본(참) obs.
- obs_groups 그룹 키 `"priv"` 이름 유지, 값만 `["priv_latent"]`.
- `train_with_estimated_states` 기본 False.

---

# 반복 2: B Domain Randomization 추가 + priv 재구성 (2026-05-19)

## 변경 요약
1. **ang_vel을 priv_latent → priv_explicit으로 이동**
2. **B의 push DR event 추가**
3. **DR 파라미터(mass, com)를 priv_latent에 추가** (현재 randomize되나 priv에 미노출 → 읽어들임)

## 새 obs 차원 (변경 2)
```
"priv_explicit" : base_lin_vel(3) + base_ang_vel(3)          = 6D   (기존 3D)
"priv_latent"   : foot_friction(8) + base_mass(1) + base_com(3) = 12D  (기존 11D)
```
- 합 6+12 = 18D (기존 14D 대비 +4D = mass 1 + com 3).
- estimator 출력 자동 3D→6D, actor_input_dim 자동 97→100D — **obs에서 차원 유도되므로 ActorCriticRMA/PPOParkour/runner 코드 수정 불필요**. validate-code로 확인.

## DR event (변경 2)
`parkour_env_cfg.py`의 `EventCfg`에 push event 추가 (B `parkour_mdp_cfg.py:321-327` 패턴):
- `isaaclab.envs.mdp.push_by_setting_velocity`, `mode="interval"`, `interval_range_s=(8.0, 8.0)`, `params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}`.
- actuator(stiffness/damping) randomization은 **추가 안 함** (B에서 비활성, 메모리 §5).
- mass/com/friction event는 A에 이미 존재 — 신규 event 불필요, priv에 읽어들이기만.

## priv_latent의 mass/com 읽기 (변경 2)
B `observations.py:_get_priv_latent` 패턴 참고:
- base_mass: `root_physx_view.get_masses()[:, base_body_id]` (1D)
- base_com: `data.com_pos_b[:, base_body_id]` (3D)
- 둘 다 startup-randomize라 per-env 상수 — priv_latent(준정적 domain 파라미터)에 적합.

## 불변 규칙 (변경 2)
- priv_explicit(6)+priv_latent(12)=18 = 기존 14 + 4. critic 합 42+187+18=247.
- estimator MSE 타깃 `obs_batch["priv_explicit"]`이 6D로 자동 — estimator output_dim도 obs 유도.
- 새 persistent buffer 없음 (mass/com은 매 step 조회). push event는 EventManager가 관리.
- obs_groups 키 불변 (`priv_explicit`/`priv_latent` 그대로, 내용 차원만 변경).

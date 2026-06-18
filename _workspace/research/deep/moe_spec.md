# MoE-Loco Actor — IsaacLab/rsl_rl 구현 명세서

> 작성: 2026-06-08 · 산출물: `ActorCriticMoE` 신규 module 명세 (구현 X, 명세 only)
> 대상 코드베이스: `/home/lgb/IsaacLab/rsl_rl` (커스텀 rsl_rl)

---

## 0. TL;DR (3-line)

- **논문**: arXiv:2503.08564 *MoE-Loco* (Huang*, Zhu*, Du, Zhao 2025, Tsinghua IIIS / Shanghai Qi Zhi / Harvard). 실재·정확.
- **핵심 구조**: N=6 expert, **dense softmax gating** (top-k 아님), gating = MLP[128] on encoder output, actor+critic 양쪽에 MoE. **load-balancing aux loss 없음** → PPO loop 무수정 drop-in.
- **난이도**: 낮음. 신규 module 1개 + cfg subclass 1개 + import 2줄. `ppo.py` 무수정.

---

## 1. 논문 / 코드 검증

### 1.1 인용 실재성

| 항목 | 검증 결과 |
|------|----------|
| arXiv ID | 2503.08564 — 실재 |
| 제목 | *MoE-Loco: Mixture of Experts for Multitask Locomotion* |
| 저자 | Runhan Huang\*, Shaoting Zhu\* (Tsinghua IIIS; Shanghai Qi Zhi), Yilun Du (Harvard), Hang Zhao† (Tsinghua IIIS; Shanghai Qi Zhi) |
| project page | moe-loco.github.io — 실재, dense softmax + manual weight 합성 확인 |

> **소스 신뢰도 주의**: 본문 PDF/abstract 직접 추출은 실패(바이너리). 아래 수치는 **ar5iv HTML 요약 + project page** 기반이며, 특히 "load-balancing loss 없음"은 **요약기의 부재 주장(absence claim)**이다. 단, 아래 §1.4의 메커니즘 논증으로 이 결론은 소스와 무관하게 robust하다.

### 1.2 핵심 설계 (논문)

| 요소 | 값 / 방식 |
|------|----------|
| expert 수 N | **6** ("We select expert number N_exp as 6") |
| gating 입력 | encoder output ĥ_t (LSTM/RMA가 proprio + task feature 인코딩) |
| gating 구조 | **MLP [128]** (Table VI) |
| 결합 방식 | **dense softmax**: `ĝ_i = softmax(g(ĥ_t))[i]`, `a_t = Σ_i ĝ_i · f_i(ĥ_t)` — 모든 expert가 매 forward 참여 |
| top-k sparsity | **없음** |
| skill 합성 | inference 시 수동 가중: `ĝ_i = w[i] · softmax(g(ĥ_t))[i]` |
| MoE 위치 | **actor + critic 양쪽** |
| RL 알고리즘 | **PPO** |
| 전체 objective | `L_surro + L_value + L_recon` — **MoE 전용 항 없음** |
| expert 특화 | task별 gating 분포 상이; t-SNE에서 bipedal vs quadrupedal이 distinct cluster 형성 |
| gradient conflict 처리 | **명시적 항 없음** — implicit routing(task마다 다른 expert에 gradient 분배)으로 완화 |
| task 수 | 9 (quadruped: bar/pit crossing, baffle crawl, stair climb, slope walk · biped: stand-up, plane walk, slope walk, stair descend) |
| deploy | Unitree Go2 + Jetson Orin onboard, 50Hz zero-shot. 추론 latency 수치 미공개 |

### 1.3 PPO 결합 — load-balancing loss가 PPO loop를 건드리는가?

**건드리지 않는다.** 논문 objective는 `L_surro + L_value + L_recon`이며:
- `L_recon`은 MoE가 아니라 그들의 **LSTM/RMA encoder reconstruction**에서 나온 항 (RMA 계열 표준). basic `ActorCritic` + MoE에는 애초에 없음.
- MoE 자체에 대한 별도 aux loss(load-balancing/entropy regularizer) 없음.

→ MoE는 standard MoE 문헌의 router-balancing(Switch Transformer류)을 **쓰지 않는다**.

### 1.4 왜 load-balancing이 없어도 되는가 (메커니즘 논증 — 소스 독립)

load-balancing aux loss는 **sparse top-k routing**에서 "선택 안 된 expert가 학습 신호를 못 받아 dead expert가 되는" 문제를 막으려고 존재한다. MoE-Loco는 **dense softmax**(매 forward에 6 expert 전부 forward + 가중합)이므로:
- 모든 expert가 매 step gradient를 받는다 → routing starvation 구조적으로 발생 불가.
- 따라서 load-balancing이 **메커니즘적으로 불필요**.

이 논증 덕분에 "aux loss 없음"이라는 결론은 요약기의 negative에 의존하지 않는다.

### 1.5 soft-MoE(dense) vs sparse top-k — trade-off

| 측면 | dense softmax (논문/권장) | sparse top-k |
|------|--------------------------|--------------|
| 연산량 | N expert 전부 forward (N×expert MLP) | top-k expert만 |
| deploy 비용 | expert당 작은 MLP면 6×도 onboard 타당 | 더 저렴하나 routing 분기 |
| dead expert | 없음 (전부 학습) | load-balancing loss 필요 |
| PPO 무수정 | ✅ (aux loss 불필요) | ❌ 사실상 load-balancing 추가 → ppo.py 또는 policy.forward에서 loss 노출 필요 |
| gradient 안정성 | 미분가능·연속 | router argmax 비연속 |

> **결론**: drop-in + Go2 onboard 목표에는 **dense softmax**가 정답. top-k는 PPO loop 또는 aux-loss 배관을 강제하므로 "drop-in" 전제를 깬다. 본 명세는 dense로 간다.

---

## 2. 코드베이스 매핑 (정확한 file:line)

### 2.1 `ActorCritic` 인터페이스 (`rsl_rl/rsl_rl/modules/actor_critic.py`)

PPO가 policy 객체에서 호출하는 **전체 surface** (`rsl_rl/rsl_rl/algorithms/ppo.py` 실측):

| 멤버 | 정의 위치 (actor_critic.py) | PPO 호출 위치 (ppo.py) |
|------|----------------------------|------------------------|
| `act(obs)` | L152 | L135, L254 |
| `evaluate(obs)` | L166 | L136, L178, L256 |
| `act_inference(obs)` | L158 | L330 (symmetry) |
| `get_actions_log_prob(a)` | L179 | L137, L255 |
| `action_mean` (property) | L116 | L138, L258 |
| `action_std` (property) | L120 | L139, L259 |
| `entropy` (property) | L124 | L260 |
| `update_normalization(obs)` | L182 | L148 |
| `reset(dones)` | L110 | L173 |
| `is_recurrent` (class attr) | L23 | L132, L208 |
| `parameters()`, `state_dict()` | nn.Module | L110, L380, L426 |

**actor가 호출되는 유일 지점**: `_update_distribution` (L131 `self.actor(obs)`, L141 `mean = self.actor(obs)`), `act_inference` (L162/L164 `self.actor(obs)`). 즉 **`self.actor`는 `obs -> mean` (또는 `state_dependent_std`면 `obs -> [.., 2, num_actions]`) signature를 갖는 단일 callable nn.Module**이면 충분.

→ **drop-in 계약**: `self.actor`를 동일 forward signature의 MoE nn.Module로 교체하면 `act` / `act_inference` / `_update_distribution` / `evaluate`(critic은 별개) 전부 **override 불필요**. `__init__`만 바뀐다.

### 2.2 MLP building block (`rsl_rl/rsl_rl/networks/mlp.py`)

- `MLP(input_dim, output_dim, hidden_dims, activation, last_activation=None)` (L31). `nn.Sequential` 상속, `forward(x)->x` (L95).
- **expert = `MLP(num_actor_obs, num_actions, expert_hidden_dims, activation)`** N개를 `nn.ModuleList`로 보유하면 됨.
- **gating = `MLP(num_actor_obs, num_experts, gating_hidden_dims, activation)`** 1개.
- `output_dim`이 tuple이면 자동 reshape (L67–74) → `state_dependent_std`의 `[2, num_actions]` 출력도 expert가 그대로 지원 가능 (단 §3.5 참조).

### 2.3 module 선택/등록 메커니즘

- **선택**: `on_policy_runner.py:275` `actor_critic_class = eval(self.policy_cfg.pop("class_name"))`. cfg의 `class_name` 문자열을 runner 모듈 globals에서 `eval`로 resolve.
- **import 배관** (`on_policy_runner.py:21–27`):
  ```python
  from rsl_rl.modules import (ActorCritic, ActorCriticCNN, ActorCriticRecurrent, resolve_rnd_config, resolve_symmetry_config)
  ```
  → `eval`이 보는 namespace는 **이 import 블록**. ⚠️ `ActorCriticRMA`는 `modules/__init__.py`에는 있지만 이 블록엔 **없다** (parkour는 별도 runner `OnPolicyRunnerParkour` 사용). 따라서 **신규 클래스는 두 곳 모두에 등록 필요**:
  1. `modules/__init__.py` (L13–21 import + L23–33 `__all__`)
  2. `on_policy_runner.py:21` import 블록
- **cfg 노출**: `RslRlPpoActorCriticCfg` (`source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py:22`)는 `@configclass`(엄격 dataclass). `num_experts` 같은 신규 필드는 선언 없이는 통과 못 함 → **cfg subclass 필요** (§3.4).

---

## 3. 구현 명세

### 3.1 신규 파일: `rsl_rl/rsl_rl/modules/actor_critic_moe.py`

`ActorCritic`을 상속. **`self.actor`만 MoE nn.Module로 교체**하고 나머지(std 처리, normalizer, critic, 모든 property/method)는 부모 그대로 상속.

```
class MoEActor(nn.Module):
    """obs -> action_mean. ActorCritic.self.actor 자리에 drop-in."""
    def __init__(self, num_obs, num_actions, num_experts,
                 gating_hidden_dims, expert_hidden_dims, activation,
                 gating_temperature=1.0):
        self.gating  = MLP(num_obs, num_experts, gating_hidden_dims, activation)   # -> [B, N]
        self.experts = nn.ModuleList([
            MLP(num_obs, num_actions, expert_hidden_dims, activation) for _ in range(N)
        ])
        # 모니터링용 버퍼 (학습 loss엔 미사용)
        self.last_gate_weights = None

    def forward(self, obs):                       # obs: [B, num_obs]
        logits = self.gating(obs) / temperature   # [B, N]
        w = softmax(logits, dim=-1)               # dense, all experts
        self.last_gate_weights = w.detach()       # gating entropy 모니터링용
        outs = torch.stack([e(obs) for e in self.experts], dim=1)  # [B, N, num_actions]
        mean = (w.unsqueeze(-1) * outs).sum(dim=1)                  # [B, num_actions]
        return mean
```

```
class ActorCriticMoE(ActorCritic):
    def __init__(self, obs, obs_groups, num_actions,
                 num_experts=6, gating_hidden_dims=[128],
                 expert_hidden_dims=[256,256,256], gating_temperature=1.0,
                 **kwargs):
        super().__init__(obs, obs_groups, num_actions,
                         actor_hidden_dims=expert_hidden_dims, **kwargs)
        # super가 만든 self.actor(단일 MLP)를 MoE로 교체
        num_actor_obs = <super가 계산한 num_actor_obs 재계산 or 보존>
        self.actor = MoEActor(num_actor_obs, num_actions, num_experts,
                              gating_hidden_dims, expert_hidden_dims,
                              self._activation, gating_temperature)
```

**구현 주의 (super 재사용 시)**:
- 부모 `__init__`은 `num_actor_obs`를 로컬 변수로 계산(actor_critic.py:48–51)하고 저장하지 않는다. MoE에서 동일 로직으로 재계산하거나(obs_groups 합), 부모 코드 복제. 단순함을 위해 **obs dim 재계산 헬퍼**를 둔다.
- 부모가 `actor_hidden_dims`로 만든 `self.actor`(단일 MLP)는 즉시 버려지고 MoE로 교체됨. 약간 낭비지만 std/normalizer/critic 초기화를 부모에 위임할 수 있어 net 이득. (대안: 부모 `__init__` 일부만 호출하는 정교한 분기 — 불필요한 복잡도, 비권장.)
- `activation` 문자열은 부모가 보관 안 하므로 MoE가 kwargs에서 직접 받음.

### 3.2 std / 인터페이스 보존

- `action_mean`/`action_std`/`entropy`/`get_actions_log_prob`는 **전부 부모 `self.distribution` 기반** (actor_critic.py L116–126, L179) → MoE가 `self.actor`만 교체하면 **자동 보존**. 별도 override 0개.
- `act`/`act_inference`/`_update_distribution`는 `self.actor(obs)`만 부르므로 **override 불필요**.

### 3.3 PPO 무수정 보장

- load-balancing loss **없음** (§1.4) → `ppo.py` 무수정.
- gating entropy 모니터링이 필요하면 **loss가 아니라 로깅**으로 처리: `MoEActor.last_gate_weights`를 runner의 log dict에 노출 (선택). PPO objective(ppo.py:317)는 손대지 않음.
- ⚠️ **만약** 나중에 expert collapse 방지용 gating-entropy bonus를 *학습 신호로* 넣고 싶어지면 → 그 순간 `ppo.py:317` loss line 또는 policy가 loss term을 반환하는 배관이 필요 → "drop-in" 깨짐. **1차 명세에서는 도입하지 않음.** collapse는 먼저 temperature/init/expert수로 대응(§5).

### 3.4 cfg 추가

신규 `@configclass` (위치: `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py`, `RslRlPpoActorCriticCfg` 바로 아래):

```python
@configclass
class RslRlPpoActorCriticMoECfg(RslRlPpoActorCriticCfg):
    class_name: str = "ActorCriticMoE"
    num_experts: int = 6
    gating_hidden_dims: list[int] = [128]
    expert_hidden_dims: list[int] = MISSING   # actor_hidden_dims 대체 역할
    gating_temperature: float = 1.0
```
- 상속으로 `init_noise_std`, `noise_std_type`, `*_obs_normalization`, `activation` 등 기존 필드 전부 유지.
- `actor_hidden_dims`는 `expert_hidden_dims`로 의미 대체. (혼동 방지: 둘 다 두되 명세에선 expert_hidden_dims 사용. 또는 actor_hidden_dims를 expert가 그대로 쓰도록 §3.1처럼 매핑.)
- agent cfg에서 `policy = RslRlPpoActorCriticMoECfg(num_experts=6, ...)`로 교체하면 끝. 예: Go2 manager-based velocity cfg `source/.../velocity/config/go2/agents/rsl_rl_ppo_cfg.py:17` 의 `policy=`를 본 cfg로.

### 3.5 critic은 MoE로?

- **1차: actor-only** (task의 "보통 actor만" 지침 + critic은 deploy 시 폐기). critic은 부모 단일 MLP 그대로.
- 논문은 actor+critic 양쪽이나, `evaluate`도 동일하게 `self.critic(obs)` 단일 호출 → critic-MoE도 drop-in 가능. **선택적 후속**: `critic_num_experts` 플래그로 critic도 MoE화. 우선순위 낮음.

### 3.6 parkour(ActorCriticRMA) 연결 — **후속 분리**

- 본 명세는 **basic `ActorCritic` 위 MoE만** 다룬다.
- `ActorCriticRMA`(`modules/actor_critic_parkour.py`)는 encoder+adaptation+별도 runner(`OnPolicyRunnerParkour`)·별도 PPO(`PPOParkour`) 결합이라 복잡도 별개. RMA의 latent를 gating 입력으로 쓰는 `ActorCriticRMAMoE`는 **후속 명세**로 분리. (논문의 gating 입력 ĥ_t = encoder output 패턴은 RMA latent와 정확히 대응하므로 후속 결합 시 자연스럽다.)

### 3.7 parkour 3-leg gait와 MoE — ⚠️ framing 교정 (중요)

> **이 연결은 신중히. 과대주장 금지.**
- MoE-Loco의 gradient-conflict 완화 이득은 **distinct task 간**(bipedal vs quadrupedal — t-SNE cluster가 그 증거)에서 나온다.
- 그러나 본 프로젝트 메모리의 3-leg 진단은 **단일 terrain·reward-side L/R 비대칭**(deficit clamp(min=0)이 RL duty≥target에서 gradient 0 → floor 정박)으로, **cross-task conflict가 아니다**.
- 따라서: **MoE가 3-leg gait를 고친다고 주장하지 않는다.** MoE는 parkour가 *실제로 distinct regime을 span할 때*(다중 terrain type / 다중 gait: 평지 trot vs gap jump vs stair) 그 regime별 expert 분화로 도움될 수 있다. reward 비대칭 local optimum은 MoE의 표적이 아니며, 그것은 reward-side 수정(메모리의 fix proposal)이 정답이다.
- **검증 가능한 가설**(주장 아님): multi-terrain parkour에서 MoE actor가 terrain별 gating 분화를 보이고, 단일 MLP 대비 per-terrain reward 분산이 줄면 그때 이득 인정. 그 전엔 미검증 가설로 표기.

---

## 4. 검증 / 실험 계획 + deploy

### 4.1 A/B 비교

| 축 | baseline | treatment |
|----|----------|-----------|
| actor | 단일 MLP `ActorCritic` | `ActorCriticMoE` (N=6) |
| 환경 | multi-terrain locomotion (velocity 또는 parkour multi-terrain) | 동일 |
| 동일 통제 | seed, num_envs, PPO hyperparam, obs/reward 전부 동일 | |

지표:
- **per-terrain reward attribution** (기존 tooling 재사용 — 메모리: 5798f… per-terrain 도구 존재).
- 3-leg gait 지표(per-foot contact duty L/R 대칭). ⚠️ §3.7대로 MoE가 이걸 고칠 거란 기대는 하지 않음 — 관찰만.
- 학습 곡선(수렴 속도/최종 return), action smoothness.

### 4.2 expert / gating 모니터링

- **expert 수 sweep**: N ∈ {2, 4, 6, 8}. (논문 6.)
- **gating entropy** 로깅: `H(ĝ) = -Σ ĝ_i log ĝ_i`의 batch 평균. 시간에 따라 0으로 붕괴하면 **expert collapse**(1개 expert 독점) 경보.
- **per-task/per-terrain gating 분포** 시각화(t-SNE 또는 terrain별 평균 gating bar) → expert 특화 확인.

### 4.3 deploy 타당성 (Go2 onboard)

- dense MoE 추론 비용 = gating MLP[128] 1회 + expert MLP N개 forward. expert가 [256,256,256]·N=6이면 단일 [512,256,128] actor 대비 연산 약 수 배 — 그래도 **Jetson Orin 50Hz 충분** (논문이 동일 Go2+Orin에서 50Hz 실증).
- **추가 센서 0개**: gating 입력 = 기존 actor obs. 신규 관측/센서 없음 → sim-to-real 제약(메모리: contact obs 금지 등) 불위반.
- expert가 작으면(e.g. [128,128]) N=6도 단일 큰 MLP와 파라미터·연산 비슷하게 맞출 수 있음 → deploy budget 튜닝 여지.

---

## 5. 리스크

| 리스크 | 메커니즘 | 완화 |
|--------|----------|------|
| **expert collapse** | gating이 1개 expert만 사용 → MoE가 단일 MLP로 퇴화 | gating entropy 모니터링(§4.2); temperature↑로 초기 분포 평탄화; gating/expert 직교 init; expert 수 과다 회피. **aux entropy loss는 PPO 무수정 깨므로 1차 보류**(§3.3) |
| **학습 초기 불안정** | dense 합산이라 초기 expert 출력 분산 합쳐짐 | expert MLP `init_weights` 작은 gain; init_noise_std 보수적; gating 초기 균등(logits≈0) |
| **parameter / 연산 증가** | N× expert | expert 폭 축소로 단일 MLP와 budget 맞춤; critic은 MoE化 안 함(§3.5) |
| **소스 불확실성** | load-balancing 부재가 요약기 negative | §1.4 메커니즘 논증으로 보강 완료. 단 정밀 수치(N=6, MLP[128])는 ar5iv 요약 기반 — 구현 전 본문 재확인 권장 |
| **framing 과대주장** | 3-leg를 MoE로 고친다는 오해 | §3.7 — multitask 아님 명시, 미검증 가설로 표기 |

---

## 6. 구현 체크리스트 (난이도: 낮음)

1. [신규] `rsl_rl/rsl_rl/modules/actor_critic_moe.py` — `MoEActor` + `ActorCriticMoE`.
2. [편집] `rsl_rl/rsl_rl/modules/__init__.py` — import + `__all__`에 `ActorCriticMoE`.
3. [편집] `rsl_rl/rsl_rl/runners/on_policy_runner.py:21` import 블록에 `ActorCriticMoE` 추가 (⚠️ 이거 없으면 `eval` resolve 실패).
4. [신규] `source/isaaclab_rl/isaaclab_rl/rsl_rl/rl_cfg.py` — `RslRlPpoActorCriticMoECfg`.
5. [편집] 대상 task agent cfg — `policy=RslRlPpoActorCriticMoECfg(...)`.
6. [무수정] `ppo.py`, `RolloutStorage`, runner train loop.

**불변 규칙 준수**: actor forward signature 보존, num_actions·obs dim 일치, std 인터페이스 부모 위임, 신규 buffer(`last_gate_weights`)는 학습 상태 아님(reset 불필요·detach).

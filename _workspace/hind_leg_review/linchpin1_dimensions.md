# Hind_Leg Linchpin #1 — Dimension Contract Review (Active History Path)

검증 대상: `HindLeg-Direct-v0` → env_cfg `HindLegHistoryEnvCfg`, runner `HindLegParkourPPORunnerCfg` (class_name="OnPolicyRunnerParkour").
read-only 리뷰. 코드 수정 없음.

---

## 0. 가장 중요한 프레이밍 정정 — 활성 경로는 `_original`이 아니다

태스크 지시는 `*_parkour_original.py` 파일들을 읽으라 했으나, **학습 스크립트가 실제로 import 하는 것은 non-`original` 모듈**이다.

- [검증됨: scripts/reinforcement_learning/rsl_rl/train.py:88] `from rsl_rl.runners import ... OnPolicyRunnerParkour`
- [검증됨: rsl_rl/rsl_rl/runners/__init__.py:15] `from .on_policy_runner_parkour import OnPolicyRunnerParkour` → **`on_policy_runner_parkour.py:70`** (NOT `_original`)
- [검증됨: rsl_rl/rsl_rl/algorithms/__init__.py:15] `from .ppo_parkour import PPOParkour` → **`ppo_parkour.py:25`**
- [검증됨: rsl_rl/rsl_rl/modules/__init__.py:15] `from .actor_critic_parkour import ActorCriticRMA` → **`actor_critic_parkour.py:81`**
- [검증됨: train.py:218-221] `class_name == "OnPolicyRunnerParkour"` 분기에서 위 import된 클래스를 인스턴스화

`*_original.py` 들은 정의만 존재할 뿐 이 task의 import 경로에 없다. 결론은 전적으로 **non-original** 코드에 근거한다. (`_original` 파일이 cfg num_priv_latent를 읽는 것은 사실이나, 그 경로는 실행되지 않는다.)

---

## 1. 결론 한 줄

**priv_latent 차원 불일치(cfg 선언 5 vs 실제 ~36)는 런타임 crash가 아니라 무해한 dead config다.** 활성 스택은 cfg의 `num_priv_latent`를 일절 읽지 않고, 모든 그룹 차원을 환경이 반환한 실제 텐서의 `shape[-1]`에서 런타임 추론한다.

핵심 근거:
- [검증됨: actor_critic_parkour.py:107-133] `ActorCriticRMA.__init__`가 priv/priv_explicit/history/policy/critic 차원을 전부 `obs[obs_group].shape[-1]` (priv는 `[-1]`, history는 `[-2]`)로 계산. cfg num_priv_latent 미참조.
- [검증됨: actor_critic_parkour.py:165] `self.priv_encoder = MLP(num_priv_obs, ...)` — `num_priv_obs`는 실제 priv_latent 텐서 폭(~36)에서 옴. 5가 아니라 ~36으로 빌드되므로 shape mismatch 자체가 발생하지 않음.
- [검증됨: on_policy_runner_parkour.py:364-365] estimator 입출력도 `obs["policy"].shape[-1]` / `sum(obs[k].shape[-1] ...)`로 런타임 추론.
- [검증됨 grep, 전체 repo non-build] 활성 경로(hind_leg + non-original rsl_rl)에서 `num_priv_latent`를 소비하는 코드 없음. 등장 위치는 (a) `hind_leg_env_cfg.py`의 cfg 정의 + 주석 처리된 line 258, (b) `*_original.py`(미import), (c) go2/R_Skeleton 등 무관 환경뿐.
- [검증됨: rl_cfg.py:118 commented + 258] `HindLegHistoryEnvCfg.observation_space = num_prio_obs`(=30)만 활성. priv_latent 포함 합산식(line 258)은 주석 처리됨.
- [검증됨: vecenv_wrapper.py:153-159] env obs dict를 `TensorDict`로 그대로 wrap만 함. `observation_space`와의 정합성 검증 없음 → 추가 키(history/priv_explicit/priv_latent)가 거부되지 않음.
- [검증됨: direct_rl_env.py:592-597] core는 `observation_space`를 `single_observation_space["policy"]` gym 메타데이터 구성에만 사용. 반환 obs dict shape를 assert 하지 않음.
- [검증됨: hind_leg/__init__.py:21] `disable_env_checker=True` → gym 레벨 space 검증도 비활성.

---

## 2. obs_groups 계약 vs 환경 반환 obs dict 정합성

환경이 활성(History) 경로에서 반환하는 obs dict 키:
[검증됨: hind_leg_env.py:144,161,171,187] `policy`, `history`, `priv_explicit`, `priv_latent`. (`scan`은 line 147 `isinstance(cfg, HindLegRoughEnvCfg)` 가드 안에서만 추가 → 활성 cfg는 History이므로 미생성.)

`resolve_obs_groups`는 [검증됨: utils.py:258-264] obs_groups가 참조하는 모든 group이 obs dict의 키로 존재하는지만 검사(차원 무관). 누락 시 ValueError.

| obs_groups set | 참조 group | obs dict에 존재? | 실제 차원(런타임 추론) | 정합 |
|---|---|---|---|---|
| policy | ["policy"] | 예 (env:144) | 30 = grav(3)+cmd(3)+(jpos-def)(8)+jvel(8)+act(8) [검증됨: env:115-120] | OK |
| critic | ["policy","priv_explicit","priv_latent"] | 예 (144/171/187) | 30+6+~36 = ~72 | OK (런타임 cat) |
| history | ["history"] | 예 (env:161) | (N,10,30) 3D, encoder는 shape[-2]=10 사용 [검증됨: a_c_parkour.py:122-125] | OK |
| priv | ["priv_latent"] | 예 (env:187) | ~36 (masses + material reshape) | OK (priv_encoder=MLP(~36,...)) |
| priv_explicit | ["priv_explicit"] | 예 (env:171) | 6 = root_lin_vel_b(3)+root_ang_vel_b(3) [검증됨: env:166-167] | OK |
| (scan) | 주석 처리됨 | n/a | scan_encoder는 `"scan" in obs_groups`일 때만 생성 [검증됨: a_c_parkour.py:118] → 미생성 | OK (일관) |

- policy obs = 30: [검증됨] cfg `num_prio_obs = 3+3+action_space*3`, action_space=8 → 3+3+24 = 30. (line 244, 233). 환경 cat 결과와 일치.
- priv_explicit = 6: [검증됨] cfg `num_priv = 6`(line 253), 환경 cat = 3+3 = 6 일치.
- history group 소비: [검증됨: a_c_parkour.py:124] history는 3D(`len(shape)==3`) assert. env가 (num_envs, history_len=10, num_prio_obs=30) 반환(env:48-49, 161) → 정합. `num_history = shape[-2] = 10` → StateHistoryEncoder tsteps=10.
- **결론: 모든 obs_groups 참조 키가 obs dict에 존재. resolve_obs_groups 통과. 차원은 전부 런타임 cat/shape로 결정되어 정합.**

`critic`은 default_sets에 포함되지만 [검증됨: runner:283] obs_groups에 이미 명시되어 있어 default 분기(utils.py:267-284) 미발동.

---

## 3. 발견된 Active 이슈 목록

| # | 심각도 | 이슈 | 위치 | 라벨 |
|---|---|---|---|---|
| I1 | **Low (dead config)** | `num_priv_latent=5`(cfg 선언) vs 실제 priv_latent 텐서 ~36 불일치. crash 아님 — 활성 스택 미참조. 주석 처리된 `observation_space` 합산식(line 258)에만 영향, 그 줄은 비활성. 혼동 유발 dead code. | hind_leg_env_cfg.py:255, 258 | 검증됨 (grep + 코드) |
| I2 | **High (training-quality, NOT dimension)** | `train_with_estimated_states=True`로 학습 step 0부터 actor의 priv_explicit 슬롯에 **미수렴 estimator 출력**을 주입. cfg 인라인 주석("수렴 전 게이트 off; 수렴 후 True 전환")과 모순. 차원은 6=6으로 정합(스코프 외)이나 초기 학습 신호를 오염시킬 수 있음. | rsl_rl_ppo_cfg.py:60; 소비 ppo_parkour.py:152-154 | 검증됨 |
| I3 | **Medium (perf/correctness)** | `torch.tensor(get_masses())` + `torch.tensor(get_material_properties().reshape(...))` 매 step 호출. `get_masses()`/`get_material_properties()`는 이미 CPU 텐서를 반환 → `torch.tensor()` 재래핑은 매 step CPU→GPU copy + UserWarning("To copy construct..."). priv_latent은 도메인 randomization 갱신 없으면 사실상 정적값인데 매 step 재구성. `torch.as_tensor`/`.to()` 또는 1회 캐싱 권장. | hind_leg_env.py:177-181 | 검증됨 (코드); 정적성은 [가설] |
| I4 | **Low (latent fragility)** | StateHistoryEncoder는 tsteps ∈ {10,20,50}만 허용, 그 외 ValueError. 현재 history_len=10 통과하나 cfg 변경 시 깨질 수 있는 숨은 제약. | actor_critic_parkour.py:64-65; history_len 정의 env_cfg.py:256 | 검증됨 |
| I5 | **Low (debug leftover)** | `print(obs)` 잔존 디버그 출력. 매 runner 초기화 시 전체 obs TensorDict 출력. | on_policy_runner_parkour.py:87 | 검증됨 |

스코프 노트: I2는 "차원 계약" 범위 밖의 학습-품질 모순이지만, train_with_estimated_states=True가 priv_explicit 슬롯을 estimator로 치환하는 동작이 dimension 경로와 맞물려 있어 기록함. 차원 자체는 정합(6=6, ppo_parkour.py:436-437 estimator가 priv_explicit 6-dim 예측).

---

## 4. 차원 추론 메커니즘 요약 (runner가 각 그룹 차원을 어떻게 결정하는가)

1. [검증됨: runner:86] `obs = self.env.get_observations()` — 실제 텐서 dict 1회 획득(reset 후).
2. [검증됨: runner:88] `resolve_obs_groups(obs, obs_groups, ["critic"])` — **키 존재성만** 검증·보정. 차원 미관여.
3. [검증됨: runner:353-356] `ActorCriticRMA(obs, obs_groups, num_actions, **policy_cfg)` 생성. 내부에서 [a_c_parkour.py:107-133] 각 set의 차원을 `obs[group].shape[-1]`(history는 `[-2]`)로 합산:
   - actor 입력 = num_actor_obs(30) + num_priv_explicit(6) + priv_encoder_dims[-1](=20) [+ scan_latent if scan]. [검증됨: a_c_parkour.py:139]
   - priv_encoder = MLP(num_priv_obs=~36 → priv_encoder_dims[-1]=20). [검증됨: 165]
   - history_encoder = StateHistoryEncoder(input=num_actor_obs=30, tsteps=num_history=10, out=20). [검증됨: 181]
   - critic = MLP(num_critic_obs=~72 → 1). [검증됨: 190]
4. [검증됨: runner:359-361] RolloutStorage가 `obs`(실제 dict)로 초기화 → 저장 차원도 런타임 텐서 기준.
5. [검증됨: runner:364-371] Estimator(input=obs["policy"].shape[-1]=30, output=Σ obs[priv_explicit].shape[-1]=6).
6. [검증됨: ppo_parkour.py:436-437] estimator는 `obs_orig["policy"]`(30)→`obs_orig["priv_explicit"]`(6) 예측 학습. [ppo_parkour.py:537] dagger는 hist_latent를 priv_latent encoder 출력에 회귀.

즉 **cfg의 num_prio_obs / num_priv / num_priv_latent / observation_space는 모델·estimator·storage 차원 결정에 일절 쓰이지 않는다.** 유일하게 active한 cfg-유래 차원은 DirectRLEnv가 gym `observation_space` 메타데이터를 구성할 때의 policy 폭(30)뿐이며, 이는 반환 dict와 대조·검증되지 않는다.

[검증됨: rl_cfg.py:125] `rnd_cfg` 기본 None → 본 cfg에 미설정이므로 RND 경로 비활성. estimator-only 경로 확정.

---

## 부록: 오탐 회피 확인
지시대로 다음 3개는 플래그하지 않음 — action_rate 순서(정상), action_space=8(정상, env_cfg.py:233), isinstance가 History를 Rough로 안 잡음(env:147 `isinstance(cfg, HindLegRoughEnvCfg)` → History cfg는 별도 클래스라 False, 무해).

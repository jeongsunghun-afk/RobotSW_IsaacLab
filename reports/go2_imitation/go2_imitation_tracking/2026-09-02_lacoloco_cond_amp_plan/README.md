# LaCoLoco 참고 — {go2, leg}_imitation_tracking 개선 구현 계획 (2026-09-02)

> 대상 task: `go2_imitation_tracking`, `leg_imitation_tracking` (두 task 공통 계획이라 더 관련 깊은 go2 쪽에 둔다).
> 참고 저장소: https://github.com/Gepetto/LaCoLoco (arXiv 2509.16061, Humanoids 2025). 클론 사본은 세션 scratchpad에만 두고 저장소에 넣지 않았다.
> 이 문서는 **계획**이다. 실측 수치는 없고, 인용한 수치는 전부 논문 표 또는 기존 memory/report에서 옮긴 것이다.

## 0. 한 줄 결론

LaCoLoco 에서 가져올 것은 세 가지이고, 우선순위는 **① 속도 조건부 discriminator → ② DRAIL 확산 discriminator → ③ ASE 식 latent 조건부 정책** 순이다. ①은 우리 코드 기준 env 쪽 수정만으로 끝나고(러너가 disc 입력 차원을 env에서 읽어감), ②는 rsl_rl 모듈 하나 추가, ③은 2단계 학습 체계라 가장 크다. CaT(constraints as terminations)는 질문 범위 밖이지만 pedipulation 힘 안전 판정에 맞아 부록으로 적는다.

## 1. LaCoLoco 가 실제로 하는 것 (코드 기준)

| 구성 | LaCoLoco 구현 | 파일 |
|---|---|---|
| 정책 조건 | 7-dim 단위구 latent `z`. actor는 `z → style MLP [512,256] → tanh → 7` 을 obs에 concat, critic은 `z` 직접 concat | `common_ase_network_builder.py` `AMPStyleCatNet1`, `AMPMLPNet` |
| latent 스케줄 | env별 1~150 step(`latent_steps_min/max`)마다 `z` 재샘플 | `common_ase_agent.py` `_update_latents` |
| 인코더 보상 | `q(z|s,s')` vMF, `r_E = clamp(mu_q · z, 0)` | `_calc_enc_rewards` |
| 다양성 손실 | `((|mu(s,z1)-mu(s,z2)|^2 / (0.5-0.5 z1·z2)) - 1)^2`, 계수 0.01 | `_diversity_loss` |
| discriminator (ASE) | MLP [1024,1024,512] → 1 logit, BCE, expert-side GP 5, logit reg 0.01 | `ase_agent.py` |
| discriminator (DRAIL) | 확산 노이즈 예측 MLP [256]×4 elu, T=1000 cosine, label 10-dim 상수(1.0/0.0), `D = sigma(L_pi - L_M)`, BCE on D, **GP 없음** | `ase_drail_network_builder.py`, `ase_drail_agent.py` |
| 보상 합성 | `task_w 0.0 + disc_w 0.5 + enc_w 0.5` (LLC), disc obs = `[base_h, grav, v, w, q, qdot]` **history 2** | yaml, `solo_env_ase_cfg.py` |
| HLC | 별도 PPO, 10 Hz 로 `z` 출력, `r_g = exp(-10 |x_eef - x_d|)` | `hrl/` |
| CaT | 위반량→종료확률 `p = p_max·clip(c+/c_max,0,1)`, `reward·(1-p)`, `dones=p`(soft) | `cat/constraint_manager.py`, `ase_env.py` |

기호:

```
z        latent (단위구, dim 7)
mu_q     인코더가 낸 단위벡터
L_M, L_pi  라벨 M(데이터)/pi(정책)로 조건한 확산 노이즈 예측 오차
D        판별 확률 = sigmoid(L_pi - L_M)
r_D      -log(1 - D)
c+       제약 위반량 max(0, c), c_max 는 Polyak 평균 최대치
```

논문 결과 요약(Table I, 5시드 best): Solo12 dog imitation 이 가장 어려운 셋이었고 tracking 오차 ASE 19.98 cm → ASE+DRAIL 9.74 cm(절반). 단 dog-imitation LLC는 실기 배포 못 했다(지지다각형 작아 불안정). 실기 배포된 합성 모션 셋에서는 DRAIL+CaT 가 5.62 cm / 낙상 0%.

**주의: LaCoLoco 의 discriminator 는 `z` 로 조건화되지 않는다.** ASE 계열이라 D는 무조건부, 조건은 인코더 보상 쪽에 있다. "conditioned discriminator" 는 CALM(2305.02195) 방식이고, DRAIL 의 `MLPConditionDiffusion(amp_obs, conditional, t)` 가 조건 벡터 입력을 이미 갖고 있어 확장하기 쉽다는 점이 우리에겐 실질적 의미다.

## 2. 우리 코드 현황과 대응 관계

| 항목 | 현재 | 근거 |
|---|---|---|
| 정책 조건 | 속도 명령(vx, vy, yaw)이 policy obs 에 이미 들어감 | go2 `policy[3:6]`, leg `policy[9:12]` |
| disc 조건 | **없음**. `_compute_amp_obs` 는 kinematics 만 | go2 env `:1135`, leg env `:722` |
| expert 샘플 라벨 | `_compute_reference_buffers` 가 `motion_ids` 를 내부 샘플 후 **버림** | go2 `:989`, leg `:609` |
| disc 입력 차원 | 러너가 env `amp_observation_space` 에서 런타임에 덮어씀 → env 가 열을 늘리면 yaml 수정 불필요 | `on_policy_runner_amp.py:59-62, 337-340` |
| disc 모듈 | `AMPDiscriminator` = MLP [1024,512] → 1, bce / ls_gan / wgan, GP 양쪽 | `amp_discriminator.py:40`, `ppo_amp.py:117` |
| 속도 라벨 인프라 | leg `motion_lib` 에 `motion_mean_speeds`(:505) `sample_motions_near_speed`(:520) `set_motion_weights_command_uniform`(:550) 있음, **go2 motion_lib 에는 없음** | grep 확인 |
| per-env style 게이팅 | go2 만 `extras["style_weight"/"style_substitute"]` 방출(:395), leg 는 없음 | |
| latent 선행 시도 | 동결 motion-VAE latent KL 보상(`ppo_latent.py`, `motion_encoder.py`) — Phase 1/2 실패, 60k 런 판정 불가 | [[project_go2_motion_vae_latent_speed_axis]] [[project_latentkl_run1_confounded]] |
| 조건부 D 선행 검토 | 2026-08-10 설계 검토: 다중 disc 대신 **단일 disc + 정규화 명령속도 조건** 권장, 미실행 | [[project_leg_multidisc_amp_design_review]], `arch_scaling_study/research_onpolicy_scaling.md` Arm 3 |

이 계획이 겨냥하는 기존 문제:

- 보행이 명령이 아니라 **초기 상태**로 결정됨(RSI 출발 gallop 21.9% vs 정지 출발 0.9%) → [[project_leg_gait_attractor_ramp_blindspot]]
- 저속 개시 히스테리시스, cmd 4.0 에서 참조에 없는 4 Hz trot 회귀(go2 lerp 0.8) → [[project_go2_style_weight_is_the_speed_ceiling]]
- 무조건부 D 는 "이 속도에서 어떤 걸음이어야 하는가"를 전혀 모른다. `command_uniform` 은 D 가 보는 expert 분포만 바꾸지 D 에게 무엇을 보는지 알려주지 않는다.

## 3. Phase 0 — 선행 정비 (0.5~1일)

1. **go2 motion_lib 를 leg 와 동등하게**: `motion_mean_speeds`, `_motion_names`, `sample_motions_near_speed`, `set_motion_weights_command_uniform` 이식. go2 클립은 이름(`walk/trot/run`)만 있고 속도 메타가 없으니 프레임에서 계산.
2. **클립→속도·gait 라벨 표** 를 `metrics/clip_labels.csv` 로 뽑아 검토. 열: `name, mean_speed, mean_yaw_rate, gait(filename), frames, weight`. 명령 범위(0~4.0 또는 0~3.2)에 대한 커버리지 그림 1장.
3. `_compute_reference_buffers` 가 `motion_ids` 와 샘플 시각을 **함께 반환**하도록 시그니처 확장(기본 호출은 obs 만 쓰므로 하위호환).
4. leg env 에 go2 와 같은 `style_weight/style_substitute` extras 방출 추가(정지 env 게이팅 대칭화). 이건 조건부 D 와 독립이지만 두 task 를 같은 코드 경로로 두기 위한 전제.

## 4. Phase 1 — 속도 조건부 discriminator (핵심, 2일)

### 설계

```
disc 입력 = [amp_obs (10 step kinematics) , c]
c_policy  = [ |v_cmd| / v_max ,  yaw_cmd / yaw_max ]        (정책 샘플·terminal 샘플)
c_expert  = [ clip_mean_speed / v_max , clip_mean_yaw / yaw_max ]   (motion_ids 로 조회)
```

- **조건은 명령(set-point) 의미**로 둔다. expert 라벨을 창 순간속도가 아니라 **클립 평균**으로 두는 이유: 정책 쪽 조건이 명령이라 순간속도가 아니고, 참조 클립도 대체로 등속이라 클립 평균이 명령의 가장 가까운 대응이다. 순간속도 라벨은 ablation 항목으로만 둔다.
- **조건 드롭아웃** `p_drop` 0.1~0.2: 조건 열을 0 으로 두고 별도 "조건 유효" 비트를 붙인다(classifier-free guidance 식). 이유: 정책이 아직 그 속도를 못 내는 초기에 조건부 D 가 style 보상을 0 으로 눌러 학습 신호가 죽는 것을 막는다. 무조건부 경로가 남아 있으면 기존 AMP 로 자연 퇴화한다.
- **GP 마스크**: gradient penalty 는 kinematics 열에만 걸고 조건 열은 제외(조건은 이산에 가까운 값이라 GP 가 의미 없고 오히려 조건 민감도를 죽인다).
- 정지 env: 정지 참조 클립이 없으므로 기존 `style_substitute` 유지. 조건 속도 0 은 걷기 클립 최저속(0.09)과 겹치지 않게 드롭아웃 처리.
- yaw 조건은 v1 에서 **선택**. leg 는 yaw 명령 ±1.5 에 `walk_turn` 이 있어 의미 있고, go2 는 `go2_walk_turn` 1개뿐이라 v1 은 속도만.

### 수정 위치

| 파일 | 변경 |
|---|---|
| `{go2,leg}_..._env_cfg.py` | `amp_cond_mode: "none" \| "speed" \| "speed_yaw"`, `amp_cond_dropout`, `amp_cond_v_max` |
| `{go2,leg}_..._env.py` `_get_observations` | `extras["amp_obs"]`, `extras["terminal_amp_obs"]` 끝에 `c_policy` 붙임. `amp_observation_space` 를 `__init__` 에서 `+cond_dim(+1 valid bit)` |
| `collect_reference_motions` / `get_amp_observations` | 반환 obs 끝에 `c_expert` 붙임(Phase 0 의 `motion_ids` 반환 사용) |
| `rsl_rl/modules/amp_discriminator.py` | `cond_dim` 인자, GP 마스크용 `kin_dim` 노출. 정규화기는 조건 열 제외 |
| `rsl_rl/algorithms/ppo_amp.py` `update_amp` | GP 를 `[:, :kin_dim]` 에만. `PPOAMP`/`PPOAMPBase` 둘 다(근사 중복 클래스) |
| `agents/rsl_rl_ppo_cfg.py` | `amp_cond_dim`(러너가 env 에서 읽어 덮어써도 되지만 로그용 명시) |

러너 3곳(`OnPolicyRunnerAMP`, `OnPolicyRunnerAMPBase`, `OnPolicyRunnerParkourAMP`)은 obs 에 열이 붙어 있으면 **무수정**이다. 이것이 env 쪽에 조건을 붙이는 방식을 고른 이유다.

### 검증 게이트

- 스모크: `amp_observation_space` 가 `490+3`/`590+3` 으로 잡히고 러너 disc 입력이 일치.
- 오프라인 판별 검사: expert 배치를 **라벨을 섞어서** 넣었을 때 D 가 real 확률을 낮추는지(조건을 실제로 쓰는지). 못 낮추면 조건 무시 상태라 `p_drop`·정규화 재점검.
- 램프 A/B(같은 iter 15k·50k, 헤드리스, ≥4회, `--all_stand` 대조 짝): 달성률 + **보행 종류(Hilbert 위상차)** + hip 좌우 쏠림 + base_h/부호반전 Hz/토크 캡 도달률. 8k 판정 금지.
- 기대 효과 순서: leg RSI 출발 vs 정지 출발 gallop 비율 격차 축소, go2 cmd 4.0 에서 4 Hz trot 대신 bound 유지.

## 5. Phase 2 — DRAIL 확산 discriminator (2~3일)

### 설계 (LaCoLoco 그대로, 조건 입력만 확장)

```
eps_hat = MLP_theta( sqrt(a_t)·x + sqrt(1-a_t)·eps ,  [label(10) , c] ,  t )
L(x, label, c) = | eps - eps_hat |^2  (dim 평균)
D(x, c) = sigmoid( L(x, label_pi, c) - L(x, label_M, c) )
loss    = BCE(D(x_M), 1) + BCE(D(x_pi), 0)         # GP·logit reg 없음
r_D     = reward_coef · ( -log(1 - D) )
```

- `t` 샘플: 절반은 U[0,T), 나머지 절반은 `T-1-t` (antithetic). `T=1000` cosine β.
- 크기: [256]×4 elu (LaCoLoco 값). 우리 obs 가 490/590-dim 이라 [512]×4 도 후보.
- 학습률: LaCoLoco 는 전체 2e-5 단일. 우리 disc lr 2.5e-4 는 확산 D 에 높을 가능성 → 1e-4 부터.
- 정규화: 기존 `EmpiricalNormalization` 유지(LaCoLoco 도 amp 입력 정규화).

### 수정 위치

| 파일 | 변경 |
|---|---|
| `rsl_rl/modules/amp_diffusion_discriminator.py` (신규) | `AMPDiffusionDiscriminator`: `get_logits`→확률, `compute_amp_reward`, `update_normalization`, `cond_dim` |
| `ppo_amp.py` | `disc_loss_type="drail"` 분기: BCE on prob, GP/logit reg 건너뜀. `disc_reward_type="drail"` |
| `agents/rsl_rl_ppo_cfg.py` | `disc_arch: "mlp" \| "drail"`, `drail_steps`, `drail_label_dim`, `drail_hidden_dims` |

인터페이스를 기존 `AMPDiscriminator` 와 동일하게 두면 러너·알고리즘 나머지는 그대로다.

### 검증 게이트

- 단위: 같은 배치에서 `D(x_M)` 평균 > `D(x_pi)` 평균이 수 백 step 안에 벌어지는지, 확률이 0/1 로 포화하지 않는지(포화하면 `t` 전략을 `constant` 로 바꿔 진단).
- 비용: forward 2회(라벨 1/0)라 disc 갱신 시간 ~2배. `disc_num_epochs 2`, `disc_mini_batch 4096` 유지 시 iter 당 증가분 기록.
- 2×2 A/B: `{mlp, drail} × {uncond, speed-cond}`, 같은 시드·같은 플랜트(`use_pace_params=false` 명시 — [[project_latentkl_run1_confounded]] 재발 방지). 판정 지표는 Phase 1 과 동일 + 관절속도 부호반전 Hz(논문이 주장한 "더 부드러운 모션"의 대응 지표).

## 6. Phase 3 — ASE 식 latent 조건부 정책 (선택, 1~2주)

Phase 1·2 로 "명령→보행" 결합이 안 풀릴 때만 간다. 두 변형:

**3-A 하이브리드(단일 단계, 권장 시작점)**: actor 입력 = `[obs, cmd, style(z)]`, `z` 는 7~8-dim 단위구, env 별 `latent_steps` 마다 재샘플. 보상 `lerp·task + (1-lerp)·(w_D·r_D + w_E·r_E)`. 인코더 `q(z|amp_obs)` 를 disc trunk 와 분리 학습(`enc.separate: True`), 다양성 손실 0.01. 기대: MI 항이 `z` 를 gait 축으로 벌려 놓아 같은 명령에서도 `z` 로 걸음을 고를 수 있게 된다(현재는 초기 상태가 고른다). 평가 시 `z` 를 gait 별 대표값으로 고정하고 램프.

**3-B LaCoLoco 정석(2단계)**: LLC `task_w=0` 으로 모션 프라이어만 학습 → HLC(10 Hz)가 속도 명령을 받아 `z` 출력. 우리 문제에서 HLC 는 "속도 명령→z" 회귀에 가깝다. 단점: 논문에서 dog-imitation LLC 가 가장 약했고 실기 배포 실패. 우리 참조도 dog 리타게팅이라 같은 함정.

수정 위치: `actor_critic_parkour.py` `ActorCriticRMA` 에 latent 입력 폭 추가(obs-group 기반 폭 추론이라 `num_latent` 그룹만 늘리면 됨), `ppo_amp.py` 에 enc 손실·다양성 손실, 러너에 latent 버퍼·재샘플. 기존 `ppo_latent.py`(동결 VAE)는 **재사용하지 않는다** — 온라인 인코더와 설계가 다르다.

## 7. 부록 — CaT (질문 범위 밖, 2~3일)

`DirectRLEnv` 에는 constraint manager 가 없으므로 env 에 `_compute_constraints()` 를 두고 `reward·(1-p)`, `extras["cstr_prob"]=p` 를 내보낸 뒤 rsl_rl `RolloutStorage` 의 `dones` 를 float 로 받아 GAE 의 `(1-done)` 에 곱하는 식이 최소 침습이다. 대상 후보: go2 calf 토크 캡 도달률(현재 정책 판정 지표로만 씀), HindLeg 목표각 한계 초과, pedipulation 발 접촉력 140 N 상한([[project_go2_pedipulation_plan]] 의 "hard 제약" 요구와 정확히 일치). LaCoLoco 는 `p_max` 를 0→0.1~0.2 로 30k step 선형 커리큘럼.

## 8. 실행 순서와 산출물

| 순서 | 작업 | 산출물 |
|---|---|---|
| 0 | Phase 0 정비 + 라벨 표 | `metrics/clip_labels_{go2,leg}.csv`, `figures/cmd_coverage.png` |
| 1 | leg 속도조건 D 학습(leg 가 속도 인프라·보행 문제를 다 갖고 있음) | `reports/leg_imitation/_comparisons/cond_disc_speed/` |
| 2 | go2 속도조건 D | `reports/go2_imitation/_comparisons/cond_disc_speed/` |
| 3 | DRAIL 2×2 | `reports/{task}/_comparisons/drail_2x2/` |
| 4 | (조건부) ASE 하이브리드 | 별도 계획 문서 |

학습은 `--video` 없이, 영상은 체크포인트에서 `report_video.py`. 램프는 헤드리스·≥4회·`--all_stand` 짝. 새 experiment_name 을 만들면 `reports/README.md` 와 `report_video.py` 의 `EXPERIMENT_TASK_MAP` 둘 다 갱신.

---

## 9. 구현 상태 (2026-09-02, 같은 날 착수)

Phase 0·1·2 코드는 **완료**, 학습 A/B 는 **미착수**(4 GPU 전부 다른 학습 7 개가 점유 중). Phase 0.4(leg `style_weight` extras)는
leg 에 정지 대체 보상이 없어 의미가 없으므로 **생략**했다.

### 켜는 법 (기본값이면 기존 run 과 수학적으로 동일)

| 플래그 | 값 | 뜻 |
|---|---|---|
| `env.amp_cond_mode` | `none`(기본) / `speed` / `speed_yaw` | AMP obs 끝에 `[|v_cmd|/v_max, (yaw_cmd/yaw_max), valid]` 를 붙인다 (+2 / +3 열) |
| `env.amp_cond_v_max`, `env.amp_cond_yaw_max` | None | 정규화 상한. None 이면 `lin_vel_x_max`, `max(|yaw_vel_*|)` |
| `agent.amp.amp_cond_dropout` | 0.1 | disc 학습 시 조건을 지우는 행 비율 (무조건부 경로 동시 학습) |
| `agent.amp.amp_cond_reward_blend` | 0.0 | 보상에 무조건부 평가를 섞는 비율 |
| `agent.amp.disc_arch` | `mlp`(기본) / `drail` | DRAIL 은 bce/bce 전용, GP·logit reg 자동 생략 |
| `agent.amp.drail_hidden_dims` 등 | `[256]*4`, elu, T=1000, label 10, antithetic | LaCoLoco yaml 값 |

```bash
# leg, 속도 조건부 MLP disc (현역 런과 같은 데이터·플래그 위에 얹는다)
CUDA_VISIBLE_DEVICES=<gpu> python scripts/reinforcement_learning/rsl_rl/train.py --task Leg-Imitation-Tracking-RMA-v0 \
  --headless --num_envs 4096 --device cuda:0 env.amp_cond_mode=speed agent.run_name=condspeed_<base>
# 같은 것 + DRAIL
... env.amp_cond_mode=speed agent.amp.disc_arch=drail agent.run_name=condspeed_drail_<base>
# go2 (플랜트 명시 필수)
python scripts/reinforcement_learning/rsl_rl/train.py --task Go2-Imitation-Tracking-v0 --headless --num_envs 4096 \
  --device cuda:<n> env.use_pace_params=false env.amp_cond_mode=speed agent.run_name=condspeed_stock
```

### 바뀐 파일

| 파일 | 내용 |
|---|---|
| `rsl_rl/rsl_rl/algorithms/ppo_amp.py` | `PPOAMP`/`PPOAMPBase` 중복을 `_AMPDiscriminatorMixin` 으로 통합. 조건 dropout, GP 를 kinematics 열만, `disc_arch` 분기 |
| `rsl_rl/rsl_rl/modules/amp_discriminator.py` | `cond_dim` 인자. 정규화기는 kinematics 열만. `drop_condition`, `cond_reward_blend` |
| `rsl_rl/rsl_rl/modules/amp_diffusion_discriminator.py` | **신규** `AMPDiffusionDiscriminator` (DRAIL). `get_logits = L_pi − L_M` 라 기존 BCE·bce 보상 그대로 |
| `rsl_rl/rsl_rl/runners/on_policy_runner_amp.py` | `amp_cfg["amp_cond_dim"] = env.amp_cond_dim` 주입 (`_env_amp_cond_dim`) |
| `source/.../direct/amp_command_condition.py` | **신규** env mixin — 조건 열 생성(정책=명령, expert=클립 평균, 정지 env valid=0) |
| `go2_imitation_tracking/motion_lib.py` | `motion_names`·`motion_mean_speeds`·`motion_mean_yaw_rates`·`sample_motions_near_speed`·`set_motion_weights_command_uniform` 이식 |
| `leg_imitation_tracking/motion_lib.py` | `motion_mean_yaw_rates` 추가 |
| `{go2,leg}_..._env.py` | mixin 상속, `amp_observation_size += cond`, `amp_obs`/`terminal_amp_obs`/expert 에 조건 열, `collect_reference_motions` 가 motion_ids 를 먼저 뽑음 |
| `{go2,leg}_..._env_cfg.py`, `agents/rsl_rl_ppo_cfg.py` | 위 플래그 |
| `scripts/imitation_learning/dump_clip_labels.py` | **신규** 라벨 표 + 커버리지 그림 (sim 불필요) |

⚠ `rsl_rl/rsl_rl/modules/__init__.py` 의 export 추가는 `.gitignore:90 (**/__*)` 때문에 **git 이 추적하지 않는다**(기존 상태).

### 검증

- 합성 데이터 단위 테스트(4096 × 590, cuda:3, 200 step): MLP 6.0 ms/step, DRAIL 7.2 ms/step. 둘 다 `D(e)≈1, D(pi)≈0` 분리.
- 스모크(64 env · 4 iter, GPU3): `leg cond+mlp`, `leg cond(speed_yaw)+drail`, `leg 무옵션`, `go2 cond+mlp`, `go2 cond+drail`, `leg-RMA cond` — **6/6 통과**, traceback 0.
  - 로그의 `[AMPCommandCondition] mode=speed cond_dim=2` 로 disc 입력 592/492 확인.
  - ⚠ leg-RMA 를 `--device cuda:3` 로 띄우면 `mdp/symmetry.py` 가 cuda:0 인덱스를 써서 죽는다 — **기존 버그**(이 변경과 무관). 현역 런처럼 `CUDA_VISIBLE_DEVICES=3 --device cuda:0` 으로 띄우면 통과.
- pre-commit: 변경 파일 대상 통과.

### 라벨 표 (`metrics/`, `figures/`)

| 데이터셋 | 클립 | 속도 [m/s] 분포 | 비고 |
|---|---|---|---|
| go2 `smr_mirror_pkl` (18) | walk 0.15/0.56/0.74, walk_turn 0.53(yaw ±0.94), trot 1.67×2(중복), run 2.27/2.54/**3.81** | 2.6~3.8 공백, 3.81 은 run2 하나(길이 가중 2.66%×2) | `clip_labels_go2.csv`, `cmd_coverage_go2.png` |
| leg `merged_leg_pkl` (7, 기본) | walk 0.51, walk_turn 0.68, trot 1.30/1.97, run 2.69/3.03/4.98 | 미러 없음 | `clip_labels_leg.csv` |
| leg `new_smr_leg_pkl` (14, 현역 런 데이터, command_uniform·v_max 3.2) | walk 0.11~0.94, walk_turn 0.32/0.68, trot 1.97, run 2.69/3.03 | `leg_walk1`(0.94) 이 20.2% | `metrics/new_smr_leg_pkl/`, `cmd_coverage_leg_new_smr_wcmd.png` |

클립 평균 yaw 는 walk_turn 을 제외하면 전부 |0.3| 미만이라, `speed_yaw` 모드에서 yaw 조건은 사실상 "선회 클립 vs 직진 클립" 2값에 가깝다. v1 은 `speed` 로 시작한다.

### 다음

1. GPU 가 비면 leg 2×2(`{mlp,drail}×{none,speed}`, 같은 시드·같은 데이터·`velscale15_ds14_wcmd_vmax32` 베이스) 착수 → `reports/leg_imitation/_comparisons/cond_disc_speed/`.
2. "라벨 섞기" 게이트 스크립트: 학습된 disc 에 expert 배치의 조건을 뒤섞어 넣어 real 확률이 떨어지는지 (조건을 실제로 쓰는지) 확인.

---

## 10. 1차 학습 실패와 원인 확정 — 조건 라벨 누설 (2026-09-02 17:44~18:20)

### 증상 (같은 iter, tensorboard 실측)

| run | iter | noise_std | mean_R | amp(style) | lin_vel | D(expert) | D(policy) |
|---|---|---|---|---|---|---|---|
| 기준선 `{mlp, none}` | 10000 | **0.402** | 653.7 | **21.2** | 44.8 | 0.81 | 0.19 |
| run A `mlp+speed` (clip_mean) | 10000 | **6.880** | 501.5 | 6.8 | 44.4 | 0.94 | 0.07 |
| run B `drail+speed` (clip_mean) | 4000 | **3.234** | 434.0 | 1.7 | 42.6 | 0.99 | 0.01 |

두 조건부 run 모두 σ 가 단조 발산(기준선은 0.62→0.40 수렴)하고 style 보상이 1/3·1/10 로 죽었다. 속도 추종 보상은 셋 다 같으므로 차이는 판별기 쪽이다. 사용자 결정으로 두 run 을 17:5x 에 중단.

### 원인 (검증됨 — `scripts/imitation_learning/check_cond_disc_leak.py`, run A `model_10000`, expert 4096 샘플)

| expert 라벨 | D(expert) |
|---|---|
| clip_mean (학습 라벨, 고유값 **9개**) | 0.935 |
| shuffled (라벨을 배치 안에서 뒤섞음) | 0.753 |
| **uniform (정책처럼 U[0,1] 연속값)** | **0.270** |
| matched (command_matched 샘플링) | 0.360 |
| dropped (조건 제거) | 0.897 |

kinematics–조건 정합성의 기여는 0.935→0.753 의 0.18 뿐이고, 라벨 **값 분포**만으로 0.935→0.270 이 갈린다. 정책 쪽 조건은 연속 균등분포, expert 쪽은 클립 평균이라 9개 이산값 — 판별기는 "이 속도의 걸음인가"가 아니라 "라벨이 격자값인가"를 배웠다. 조건 dropout 10% 는 무조건부 경로(0.897)는 살렸지만 누설은 못 막았고, GP 를 조건 열에서 뺀 것은 이 방향의 날카로움을 방치했다. σ 발산은 style gradient 가 평평해져 entropy 보너스만 남은 하류 증상이다(latentKL 60k 와 같은 모양).

### 처방 (구현, 기본값 승격)

`env.amp_cond_expert_sampling="command_matched"`: expert 배치마다 조건 `c ~ U[lin_vel_x_min, lin_vel_x_max]`(yaw 는 U[yaw_min, yaw_max])를 **먼저** 뽑고, `sample_motions_near_speed(c, τ=amp_cond_match_temperature=0.5)` 로 클립을 고른 뒤 라벨을 `c` 로 붙인다. 조건의 주변분포가 정책과 정확히 같아져 D 는 kinematics–조건 정합성으로만 판별해야 한다(8월 설계검토의 "속도 기반 확률적 real 샘플링"). `clip_mean` 은 비교용으로만 남긴다. 구현: `amp_command_condition.py` `_sample_expert_motions()`, 두 env 의 `collect_reference_motions`.

### 재착수 (18:19)

- run A' `mlp+speed+matched`: `logs/rsl_rl/leg_imitation_tracking_rma/2026-09-02_18-1*_condmatch_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32` (GPU3)
- run B' `drail+speed+matched`: `…_condmatch_drail_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32` (GPU1)
- 기준선과 단일변수 비교를 지키기 위해 `schedule` 은 기준선과 같은 adaptive 로 두었다. σ 가 다시 발산하면 그때 `agent.algorithm.schedule=fixed` 를 건다.
- 게이트: 5k 에서 leak 스크립트 재실행 — `uniform ≈ matched` 이고 σ 가 0.4 대로 수렴해야 통과.

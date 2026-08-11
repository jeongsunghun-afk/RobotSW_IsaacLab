# 2026-08-11_11-14-27_noamp_stock

한 줄 목적: **AMP style reward 를 정책에서 완전히 떼어내고**(pure task reward) 학습했을 때
`go2_imitation_tracking` 의 속도 천장이 어떻게 움직이는지 본다. **스톡 플랜트** 쪽.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-11_11-14-27_noamp_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-11 11:14 · GPU1 (leg run 과 공유) · ETA 약 15 h · 60000 iter
- 액추에이터: `ImplicitActuatorCfg(effort_limit=None, kp 25, kd 0.5)` — `implicit_stock` 과 동일
- **직전 세대(`2026-08-10_10-45-34_implicit_stock`) 대비 바뀐 것은 AMP 혼합 하나** — 깨끗한 A/B

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name noamp_stock env.use_pace_params=false \
  agent.amp.task_reward_lerp=1.0 agent.amp.task_reward_lerp_start=1.0 \
  agent.amp.task_reward_lerp_anneal_iters=0
```

### AMP 를 끄는 방식

러너는 `if "amp_obs" in extras:` 로 게이트되어 있고 env 가 항상 `amp_obs` 를 넣어 주므로
"AMP 를 끈다"는 스위치는 없다. 대신 혼합식

```
total_reward = task_reward_lerp * rewards + (1 - task_reward_lerp) * style_term
```

에서 `task_reward_lerp = 1.0` 으로 고정하면 **style term 의 계수가 정확히 0** 이 되어 정책이
받는 보상은 순수 task reward 다 (`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:174`).
`_start` 와 `_anneal_iters` 까지 같이 넘겨야 Stage 1 (`_start=0.5`) 이 살아나지 않는다.

**검증 완료** — hydra 오버라이드가 조용히 씹히는 사고(`agent.resume` 전례)를 배제하기 위해 둘 다 확인:

- `params/agent.yaml` 82~84 줄: `task_reward_lerp: 1.0` / `_start: 1.0` / `_anneal_iters: 0`
- TensorBoard **런타임** 값 `Loss/amp_task_reward_lerp` = **1.0 @ iter 0~4** (설정값이 아니라
  실제로 곱해지는 값)

⚠ **`Episode_Reward/amp_reward` 는 계속 기록된다** — discriminator 는 그대로 학습하기 때문이다.
정책 보상에는 **기여가 0** 이므로 이 스칼라를 보상 성분으로 읽지 말 것.

## 판정 규칙

★ 이 세대군은 **40k 이후 점으로만** 판정한다. 8k~24k 차이는 48k 이후를 예측하지 못한다는 것이
`implicit_stock` 에서 실측됐다(Δ −41 → +6 으로 반전). 근거:
`../_comparisons/mimickit_vs_60_actuator_limit/README.md`

## 비교 대상 (`implicit_stock`, AMP on, 동일 플랜트·액추에이터)

| cmd | 지표 | 40k | 48k | 56k |
|---|---|---:|---:|---:|
| 2.5 | 중앙 | 1.389 | 1.598 | 1.569 |
| 3.0 | 중앙 | 0.686 | **1.686** | 1.623 |
| 3.5 | 달성률 | 0 | 0 | 0 |

## 결과 요약

학습 진행 중 — 램프 미측정.

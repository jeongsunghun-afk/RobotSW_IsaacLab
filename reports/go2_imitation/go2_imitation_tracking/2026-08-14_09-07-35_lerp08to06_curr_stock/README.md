# 2026-08-14_09-07-35_lerp08to06_curr_stock

한 줄 목적: **style 가중치 커리큘럼** — `lerp08_stock` 40k 에서 이어받아 20k 에 걸쳐
`task_reward_lerp` **0.8 → 0.6** (style 0.2 → 0.4)으로 조인다. 속도를 먼저 얻고 나서
걸음을 다듬는다.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-14_09-07-35_lerp08to06_curr_stock`
- 시작: 2026-08-14 09:07 · GPU1 · `--max_iterations 20000` (= 40k → 60k, **추가** iter 수)
- 분기점: `2026-08-12_10-19-34_lerp08_stock/model_40000.pt` (그 시점에 `cmd 4.0` 98% 확인됨)

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 20000 --run_name lerp08to06_curr_stock env.use_pace_params=false \
  --resume --load_run 2026-08-12_10-19-34_lerp08_stock --checkpoint model_40000.pt \
  agent.amp.task_reward_lerp_start=0.8 agent.amp.task_reward_lerp=0.6 \
  agent.amp.task_reward_lerp_anneal_iters=20000
```

### 코드 수정 없이 커리큘럼이 되는 이유

러너의 anneal 은 `progress = min(1, (it − start_it) / anneal_iters)` 이고
`start_it = self.current_learning_iteration` 이다
(`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:105-122`). **resume 하면 anneal 시계가 그
지점에서 다시 시작**하므로, `_start`/`_end`/`_anneal_iters` 만 주면 그대로 커리큘럼이 된다.

⚠ resume 은 argparse 플래그로만 먹는다 — hydra `agent.resume` 은 조용히 무시된다.

**런타임 검증**: `Loss/amp_task_reward_lerp` 가 iteration **40000 에서 0.8**,
47193 에서 **0.7281**. 이론값 `0.8 − 0.2 × (47193−40000)/20000 = 0.7281` 과 일치한다.
`current_learning_iteration` 도 40000 으로 복원됐다(스텝 번호가 40000 부터 시작).

## 결과 (완주, 60k) — 판정 구간 3 점

**★ scratch 0.6 과 사실상 동률이다.** `cmd 4.0`, 40k 이후 점 평균:

| arm | vx 평균 | 달성률 | `base_h` 평균 (범위) | h std 평균 |
|---|---:|---:|---:|---:|
| 고정 0.8 | **3.870** | 98 | 0.319 (**0.279~0.365**) | 0.071 |
| **0.8 → 0.6 (이 런)** | 3.746 | 97 | **0.392** (0.384~0.398) | 0.062 |
| 0.6 scratch | 3.812 | 95 | 0.376 (0.374~0.378) | 0.066 |

체크포인트별:

| iter | vx | % | `base_h` | h std | flip Hz |
|---|---:|---:|---:|---:|---:|
| 48k (lerp≈0.72) | 3.756 | 100 | 0.384 | 0.051 | 11.7 |
| 56k (lerp≈0.64) | 3.756 | 95 | 0.393 | 0.070 | 12.6 |
| 59999 (lerp 0.6) | 3.724 | 95 | 0.398 | 0.065 | 12.6 |

- **`base_h` 가 iteration 에 따라 단조 상승**(0.384 → 0.398)한다. style 을 조일수록 몸통이
  더 선다는 뜻이고, 고정 0.8 이 `cmd 4.0` 에서 0.279~0.365 로 주저앉는 것과 대비된다.
- 속도 대가는 고정 0.8 대비 **−3%** (3.870 → 3.746).

## ⚠ 철회 — "몸통 진동을 절반으로 줄인다"

48k **한 점**만 보고 h std 0.074 → 0.051 이라고 적었으나, 56k·59999 에서 0.070 / 0.065 로
고정 0.8(0.072 / 0.072)과 거의 같아졌다. 판정 구간 평균은 **0.062 vs 0.071** — 개선은
있지만 절반이 아니다. `cmd 3.5` 에서는 오히려 커리큘럼이 근소하게 나쁘다(0.053 vs 0.047).

★ 이 실수의 형태가 이 세대군에서 반복된다 — **한 점으로 배율을 주장하면 안 된다.**
견고하게 남는 차이는 `base_h` 쪽이다(3 점 전부, 범위도 겹치지 않음).

## ★ 커리큘럼 자체의 이득은 측정되지 않았다

처음부터 0.6 으로 학습한 `lerp06_stock` 이 같은 곳에 도착한다 — vx 3.812 vs 3.746,
`base_h` 0.376 vs 0.392. 속도는 scratch 가 조금 빠르고 자세는 커리큘럼이 조금 낫다.
**"속도를 먼저 얻고 나서 조인다"는 절차의 추가 이득은 이 데이터로 확인되지 않는다.**

실무상 장점은 남는다 — 40k 짜리 사전 학습을 재활용하므로 20k 만 더 돌리면 된다
(scratch 는 60k). 결과 품질이 아니라 **비용** 쪽 논거다.

## 산출물


### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_48000__curr_ramp_0to4ms__20260814.mp4`](videos/model_48000__curr_ramp_0to4ms__20260814.mp4) | `model_48000.pt` | `ramp_0to4ms` (체이스캠, 0→4 m/s) | 2026-08-14 |
<!-- /report-video:videos -->

원자료: `../../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_curr/`

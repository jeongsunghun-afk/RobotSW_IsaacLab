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

## 결과 (48k, 커리큘럼 8k 진행 · 현재 lerp ≈ 0.72)

**★ 고정 0.8 에서 `cmd 4.0` 만 무너지던 것이 사라졌다.** 같은 48k 끼리 비교:

| cmd | | 고정 0.8 | **커리큘럼** | |
|---|---|---:|---:|---|
| | | vx / 달성률 / `base_h` / h std / flip Hz | vx / 달성률 / `base_h` / h std / flip Hz | |
| 2.0 | | 1.747 / 98 / 0.351 / **0.034** / 8.6 | 1.605 / **100** / 0.353 / **0.010** / 8.1 | |
| 2.5 | | 2.371 / 98 / 0.362 / **0.037** / 9.8 | 2.290 / **100** / 0.368 / **0.019** / 9.7 | |
| 3.0 | | 2.893 / 98 / 0.392 / **0.045** / 10.8 | 2.868 / **100** / 0.395 / **0.027** / 11.4 | |
| 3.5 | | 3.338 / 98 / 0.388 / **0.047** / 11.2 | 3.319 / **100** / 0.399 / **0.029** / 11.5 | |
| 4.0 | | 3.852 / 98 / **0.338** / **0.074** / 12.5 | 3.756 / **100** / **0.384** / **0.051** / 11.7 | |

- **몸통 상하 진동(h std)이 전 구간에서 거의 절반**이다 (0.034→0.010, 0.037→0.019,
  0.045→0.027, 0.047→0.029, 0.074→0.051).
- `cmd 4.0` 에서 고정 0.8 은 몸통이 **0.388 → 0.338 로 주저앉는데** 커리큘럼은 **0.384 를 유지**한다.
- 달성률이 98 → **100%** 로 올랐다.
- 대가는 속도 **−2.5%** (`cmd 4.0` 3.852 → 3.756).

![cmd 4.0 고정 vs 커리큘럼](../_comparisons/mimickit_vs_60_actuator_limit/figures/cmd40_fixed_vs_curriculum.png)

0.36 s 구간. 위(고정)는 몸통이 낮고 다리가 벌어지며, 아래(커리큘럼)는 몸통이 떠 있고 자세가 모여 있다.

## 판정 규칙

★ 40k 이후 점으로만 판정한다. 재현 산포 ±8~15%p / ±0.02 m/s.
★★ 속도와 **걸음 품질을 같이** 읽는다 — `base_h`, `h std`, 관절속도 부호반전 Hz, 토크 캡 도달률.

⚠ 48k 한 점이다(커리큘럼 8k 시점, lerp 아직 0.72로 목표 0.6 에 미달). 56k·60k 로 확인한다.

## 산출물

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_48000__curr_ramp_0to4ms__20260814.mp4`](videos/model_48000__curr_ramp_0to4ms__20260814.mp4) | `model_48000.pt` | `ramp_0to4ms` (체이스캠, 0→4 m/s) | 2026-08-14 |
<!-- /report-video:videos -->

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_curr/`

# 조건부 AMP discriminator 학습 곡선 수치 분석

> 작성 일시: 2026-09-04 10:45
> 비교 대상: baseline(무조건부 mlp) vs A'(cond-mlp) vs B'(cond+drail) — discriminator 조건화 여부와 disc 아키텍처

## 0. 읽은 로그와 시점

읽은 시각 2026-09-04 10:38~10:42 KST. TF event 파일을 `EventFileLoader` 로 스트리밍해 필요한 19개 tag 만 npz 로 캐시했다 (`metrics/train_curves/*.npz`, 합계 12.2 MB).

| run | 로그 디렉토리 (`logs/rsl_rl/leg_imitation_tracking_rma/` 아래) | iter 범위 (읽은 시점) |
|---|---|---|
| baseline | `2026-09-02_09-18-43_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32` | 0 ~ 49,999 (완료) |
| A' cond-mlp | `2026-09-02_17-56-53_condmatch_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32` | 0 ~ 49,999 (완료) |
| B' 원본 | `2026-09-02_17-56-53_condmatch_drail_cmdchg4s_ep20_velscale15_ds14_wcmd_vmax32` | 0 ~ 16,748 (중단) |
| B' resume | `2026-09-03_12-51-22_condmatch_drail_resume16k7_...` | 16,700 ~ **48,475** (진행 중) |

B' 는 원본의 `iter < 16,700` 과 resume 전체를 이어 붙였다. resume 로그의 step 축은 0 이 아니라 16,700 에서 시작하므로 오프셋 보정은 하지 않았다. 이동평균은 seam 을 가로지르지 않게 **구간별로** 계산했다.

## 1. 설정 (run params yaml 실측)

세 run 공통: `disc_loss_type=bce`, `disc_reward_type=bce`, `reward_coef=2.0`, `task_reward_lerp=0.5`, `disc lr=2.5e-4`, `disc_num_epochs=2`, `disc_mini_batch_size=4096`, `replay_buffer_size=200000`, `amp_weight=1.0`.

| | baseline | A' | B' |
|---|---|---|---|
| `disc_arch` | (없음 = mlp) | `mlp` | `drail` |
| `amp_cond_mode` (env) | 미설정 | `speed` | `speed` |
| `amp_cond_expert_sampling` | 미설정 | `command_matched` | `command_matched` |
| `amp_cond_dropout` | 미설정 | 0.1 | 0.1 |
| `amp_cond_reward_blend` | 미설정 | 0.0 | 0.0 |
| grad penalty / logit reg 실제 적용 | 적용 (coef 5.0 / 0.01 weight) | 적용 | **미적용** (`ppo_amp.py:225` `use_regularizers = disc_arch != "drail"`) |

B' 의 `Loss/disc_grad_penalty` 는 전 구간 정확히 0.0 이다. 이는 측정값이 아니라 drail 코드 경로가 0 텐서를 그대로 로깅하기 때문이다 (`ppo_amp.py:232`). cfg 의 `gradient_penalty_coef=5.0` 과 `disc_logit_reg=0.01` 은 B' 에서 **읽히지 않는다**.

## 2. 보상 식 (코드 실측)

`amp_discriminator.py:_reward_from_logits` (bce 분기):

```
prob   = sigmoid(logit)
reward = -log(clamp(1 - prob, min=1e-4))        # 스텝당 style 보상
reward = reward * amp_reward_coef               # amp_reward_coef = 2.0 (cfg 가 모듈 기본 1.5 를 덮어씀)
```

`on_policy_runner_amp.py:186` 의 보상 융합:

```
total_reward = task_reward_lerp * task_reward + (1 - task_reward_lerp) * amp_reward
             = 0.5 * task + 0.5 * amp
```

로깅 스케일 (`on_policy_runner_amp.py:199`):

```
Episode_Reward/amp_reward = mean_over_done_envs( sum_over_episode(amp_reward) ) / max_episode_length_s
```

`max_episode_length_s = 20.0` 이다. env 쪽 `Episode_Reward/*` 도 같은 나눗셈을 쓴다 (`leg_imitation_tracking_env.py:486`). 즉 두 계열 모두 **초당 보상**이고 `task_reward_lerp` 적용 **전** 값이므로, 6절의 비율은 스케일이 일치한다. 총 보상 기여로 환산하려면 둘 다 0.5 를 곱하면 된다.

## 3. D 가 학습되는가 — expert/policy 출력과 margin

`disc_expert_output` / `disc_policy_output` 은 sigmoid(logit) 배치 평균이다. D 의 목표는 expert→1, policy→0. 정책이 완전히 속이면 둘 다 0.5, margin 0.

초기 급등 (25-iter 구간 평균, 원자료):

| iter | baseline margin | A' margin | B' margin |
|---|---|---|---|
| 0 | 0.457 | 0.489 | 0.829 |
| 10 | 0.579 | 0.615 | 0.855 |
| 50 | 0.698 | 0.732 | 0.883 |
| 100 | 0.719 | 0.757 | 0.915 |
| 500 | 0.741 | 0.798 | 0.892 |
| 1000 | 0.761 | 0.735 | 0.873 |

이동평균이 정점의 절반을 넘는 첫 iter 는 baseline 14, A' 13, B' 0 이다. 세 run 모두 D 는 수십 iter 만에 거의 다 학습된다.

정점 (500-iter 이동평균 기준):

| run | margin 정점 | 정점 iter | 원자료 최대 | 최종 이동평균 |
|---|---|---|---|---|
| baseline | 0.7618 | 1,078 | 0.7765 | 0.5129 |
| A' | 0.7975 | 838 | 0.8275 | 0.5012 |
| B' | 0.9022 | 567 | 0.9292 | 0.8805 |

baseline 과 A' 는 1k 부근 정점 뒤 **단조 하강**해 50k 에서 0.51 / 0.50 이 된다. D 가 점점 덜 이긴다 = 정책이 참조 모션에 가까워진다. B' 는 567 iter 의 0.902 에서 48k 의 0.880 까지 2.4% 만 내려온다. drail D 는 48k 동안 사실상 계속 이긴다.

## 4. 정책이 속이는가 — policy_output 과 amp_reward

`disc_policy_output` 이 0.5 쪽으로 오를수록 정책이 D 를 속이는 것이다 (표 값은 해당 iter 이전 500-iter 후행 평균):

| iter | baseline | A' | B' |
|---|---|---|---|
| 5,000 | 0.1897 | 0.1857 | 0.0607 |
| 10,000 | 0.1903 | 0.1905 | 0.0596 |
| 20,000 | 0.1976 | 0.2094 | 0.0605 |
| 30,000 | 0.2200 | 0.2307 | 0.0606 |
| 40,000 | 0.2314 | 0.2412 | 0.0601 |
| 최종 | **0.2437** (50k) | **0.2494** (50k) | **0.0598** (48.5k) |

`Episode_Reward/amp_reward` (초당):

| iter | baseline | A' | B' |
|---|---|---|---|
| 5,000 | 21.46 | 20.91 | 8.86 |
| 20,000 | 22.70 | 23.89 | 8.82 |
| 30,000 | 25.35 | 26.59 | 8.84 |
| 40,000 | 26.74 | 27.93 | 8.81 |
| 최종 | **28.29** | **29.00** | **8.77** |

**주의 — B' 의 숫자는 A'/baseline 과 같은 축이 아니다.** drail 의 logit 은 diffusion loss 차 `L_pi − L_M` 이므로 sigmoid 값이 MLP D 의 sigmoid 와 같은 척도가 아니다. 따라서 `0.060 < 0.244` 나 `8.77 < 29.0` 을 "B' 모션이 더 나쁘다"로 읽으면 안 된다. 아키텍처와 무관하게 비교 가능한 것은 **곡선의 움직임**이다.

두 지표가 같은 방향으로 움직인다. A' 는 20k 이후 baseline 을 앞지르고 최종 amp_reward 가 2.5% 높다 (29.00 vs 28.29), policy_output 도 0.2494 vs 0.2437 로 A' 가 위다. B' 는 8.77 에 **완전히 평평**하다. 초당 8.77 은 스텝당 8.77/50 = 0.175, `reward_coef=2.0` 을 빼면 `-log(1-p) = 0.0877` → `p ≈ 0.084` 로 로깅된 `disc_policy_output` 0.060 과 같은 자릿수다 (replay buffer 와 온-폴리시 시점 차이만큼 어긋난다).

## 5. 수렴 시점

정의: 구간별 500-iter 이동평균이 최종값 ±5% 밴드에 들어와 이후 벗어나지 않는 첫 iter. 밴드 폭 = |최종값|×0.05 를 함께 적는다.

| 곡선 | baseline | A' | B' | 비고 |
|---|---|---|---|---|
| margin | 39,821 (밴드 ±0.0256) | 36,150 (±0.0251) | 32 (±0.0440) | B' 는 시작부터 평평해 판정이 **퇴화**했다 |
| amp_reward | 40,513 (±1.414) | 38,030 (±1.450) | 17,196 (±0.438) | B' 는 seam 직후 |
| lin_vel_reward | 1,948 (±2.252) | 1,287 (±2.184) | 17,208 (±2.246) | B' 는 43.03(5k)→44.93(최종)로 계속 오르는 중이라 밴드 진입이 늦다 |
| noise_std σ | 43,751 (±0.0155) | 38,063 (±0.0138) | **미수렴 (발산 중)** | B' 에 대한 기계적 답 46,834 은 발산 곡선이라 무의미하다 |
| disc_total_loss | 37,003 (±0.0204) | 33,296 (±0.0208) | 16,708 (±0.0043) | |

baseline·A' 의 style 계열(margin, amp_reward, policy σ, total_loss)은 36k~44k 에 가서야 밴드에 들어온다. 50k 예산의 3분의 2 를 개선에 쓰고 있다는 뜻이며, A' 가 baseline 보다 3.7k~5.7k 이르다. B' 의 이른 "수렴"(margin 32, total_loss 16,708)은 반대로 개선이 처음부터 멈춰 있다는 뜻이므로 같은 단어를 반대로 읽어야 한다. B' 의 밴드 폭 ±0.0043 은 세 run 중 가장 좁아, 판정 자체가 그만큼 평평한 곡선 위에서 내려졌다.

## 6. 손실

BCE 우연 수준은 ln2 = 0.6931 이다.

| iter | baseline exp / pol | A' exp / pol | B' exp / pol |
|---|---|---|---|
| 5,000 | 0.2244 / 0.2172 | 0.2183 / 0.2123 | 0.0870 / 0.0884 |
| 20,000 | 0.2353 / 0.2280 | 0.2475 / 0.2405 | 0.0858 / 0.0884 |
| 40,000 | 0.2741 / 0.2679 | 0.2880 / 0.2805 | 0.0850 / 0.0881 |
| 최종 | 0.2904 / 0.2839 | 0.2989 / 0.2915 | 0.0844 / 0.0877 |

비대칭은 **거의 없다**. baseline·A' 는 expert 쪽이 0.006~0.007 만큼 항상 크고 (expert 를 1 로 밀어붙이는 게 근소하게 더 어렵다), B' 는 반대로 policy 쪽이 0.003 크다. 세 run 모두 ln2 의 12~43% 수준으로, D 가 우연보다 훨씬 잘 맞힌다.

`disc_total_loss = 0.5(expert+policy) + grad_penalty + logit_reg`:

| iter | baseline | A' | B' |
|---|---|---|---|
| 5,000 | 0.3323 | 0.3256 | 0.0877 |
| 최종 | 0.4087 | 0.4161 | 0.0860 |

baseline·A' 의 total 이 0.5(e+p) 보다 0.11~0.12 큰 것이 grad penalty 몫이다. `disc_grad_penalty` 는 baseline 0.1115→0.1215, A' 0.1102→0.1209 로 50k 동안 9% 만 커진다. 발산 징후는 없다. `disc_logit_reg` 는 로깅되지 않아 total 에서 GP 를 뺀 잔차(0.001 미만)로만 추정된다 — **미확인**.

σ 와의 시간 상관 (1차 차분 Pearson, 인과 아님):

| 쌍 | r | n | 창 |
|---|---|---|---|
| B' Δσ vs Δpolicy_output | −0.0076 | 31,275 | 17,200~48,475 |
| B' Δσ vs Δamp_reward | −0.0057 | 31,275 | 17,200~48,475 |
| B' Δσ vs Δlin_vel_reward | +0.0066 | 30,770 | 17,200~48,475 |
| B' Δσ vs Δlearning_rate | +0.0281 | 31,275 | 17,200~48,475 |
| A' Δσ vs Δpolicy_output | −0.0315 | 49,999 | 0~49,999 |
| A' Δσ vs Δamp_reward | −0.0506 | 49,999 | 0~49,999 |

전부 |r| < 0.06 이다. B' 의 σ 발산은 판별기 출력이나 style 보상의 **iter 단위 요동과 동기화되어 있지 않다**. GP 부재와 σ 발산을 잇는 정황은 이 상관에서는 얻어지지 않는다. 다만 이 검정은 iter 스케일의 빠른 결합만 본다. "GP 가 없어 48k 에 걸쳐 σ 가 서서히 밀려 올라간다" 같은 **run 스케일의 느린 기전은 이 null 로 배제되지 않는다**.

## 7. A' vs B' 고정 iter 비교

값은 해당 iter 이전 **500-iter 후행 평균**이다. 20,000 행은 seam(16,700) 에서 3.3k 밖에 지나지 않았다.

| iter | margin A' / B' | amp_reward A' / B' | lin_vel A' / B' | σ A' / B' |
|---|---|---|---|---|
| 5,000 | 0.629 / 0.878 | 20.91 / 8.86 | 43.55 / 43.03 | 0.382 / 1.472 |
| 10,000 | 0.619 / 0.880 | 21.49 / 8.60 | 43.82 / 43.92 | 0.375 / 2.192 |
| 16,700 | 0.609 / 0.881 | 22.26 / 8.56 | 43.63 / 44.00 | 0.371 / 3.544 |
| 20,000 | 0.581 / 0.879 | 23.89 / 8.82 | 43.79 / 44.35 | 0.347 / 4.335 |
| 30,000 | 0.539 / 0.879 | 26.59 / 8.84 | 43.95 / 44.57 | 0.306 / 7.239 |
| 40,000 | 0.518 / 0.880 | 27.93 / 8.81 | 43.84 / 44.79 | 0.286 / 10.81 |
| 45,000 | 0.509 / 0.880 | 28.44 / 8.75 | 43.75 / 44.73 | 0.279 / 12.87 |
| 48,000 | 0.501 / 0.881 | 29.05 / 8.77 | 43.70 / 44.87 | 0.274 / 14.15 |
| 최종 | 0.501 (50k) / 0.880 (48.5k) | 29.00 / 8.77 | 43.67 / 44.93 | 0.276 / 14.37 |

**σ 발산은 resume 이후에 시작된 것이 아니다.** 원본 run 의 σ 는 이미 1k 에서 1.01, 5k 에서 1.50, 10k 에서 2.22, 16k 에서 3.41 이다 (200-iter 후행 평균). 같은 시점 A' 는 0.37~0.38 대에 머문다.

(baseline σ, 500-iter 후행 평균: 5k 0.392 / 16k 0.410 / 최종 0.309)

| iter | A' σ | B' σ | 배수 |
|---|---|---|---|
| 1,000 | 0.574 (정점 0.616 @704) | 1.013 | 1.8× |
| 5,000 | 0.383 | 1.495 | 3.9× |
| 10,000 | 0.375 | 2.217 | 5.9× |
| 16,000 | 0.367 | 3.413 | 9.3× |
| 16,700 (seam 직전 / 직후) | 0.372 | 3.582 / 3.604 | 9.6× |
| 48,000 | 0.274 | 14.21 | 51.9× |

seam 앞뒤 3.582 → 3.604 는 연속이다. **발산은 원본 run 의 500 iter 부근부터 단조로 진행됐고 resume 은 무관하다.** 실무적 함의: `drail_lr25e6` 재시도는 50k 를 기다리지 않고 **5k 에서 σ 가 1.5 근처인지만 봐도 판정된다**.

**σ 가 커지는 동안 margin 은 움직이지 않는다.** 아래 창 16,700~48,475 는 margin 비교 구간에 맞춘 것이지 발산이 거기서 시작해서가 아니다. σ 가 3.60 → 14.44 로 4.0배 커지는 이 구간 동안 margin 은 0.8788 → 0.8805 (+0.2%), `D(expert)` 0.9395 → 0.9403, `D(policy)` 0.0606 → 0.0598 이다.

resume seam 재워밍 아티팩트는 **약 50 iter 로 짧다**: amp_reward 가 [16,700, 16,750) 에서 평균 5.32 (최소 0.594) 로 떨어졌다가 [16,750, 16,900) 에서 8.66 으로 복귀한다. seam 직전 [16,200, 16,700) 평균은 8.57 이다. margin 은 seam 앞뒤로 0.8788 / 0.8813 으로 사실상 끊김이 없다.

부수 지표:

| iter | entropy base/A'/B' | lr base/A'/B' | ep_len base/A'/B' | mean_reward base/A'/B' |
|---|---|---|---|---|
| 5,000 | 5.43 / 5.07 / 27.0 | 4.30e−5 / 4.80e−5 / 1.22e−4 | 986 / 989 / 992 | 654 / 644 / 515 |
| 최종 | 1.33 / −0.45 / 35.6 | 4.24e−5 / 4.74e−5 / 1.79e−4 | 995 / 996 / 997 | 726 / 726 / 532 |

세 run 모두 에피소드 길이가 995 스텝 근처로 종료 없이 완주한다. B' 의 entropy 35.6 과 lr 1.79e−4 는 σ 14.4 의 직접 귀결이다 (가우시안 엔트로피는 Σlog σ, adaptive lr 은 KL 목표를 맞추려 lr 을 올린다).

## 8. task 보상과 style 보상의 균형

`lin_vel_reward / amp_reward` (둘 다 초당, lerp 전):

| iter | baseline | A' | B' |
|---|---|---|---|
| 5,000 | 2.074 | 2.084 | 4.858 |
| 10,000 | 2.079 | 2.040 | 5.111 |
| 20,000 | 1.982 | 1.833 | 5.031 |
| 30,000 | 1.782 | 1.653 | 5.047 |
| 40,000 | 1.689 | 1.570 | 5.091 |
| 최종 | 1.592 | 1.506 | 5.127 |

baseline 과 A' 는 task 가 45 근처에 고정된 채 style 이 21→28~29 로 올라오면서 비율이 2.08 → 1.59/1.51 로 내려간다. style 이 예산을 회수하는 정상 진행이다. A' 가 baseline 보다 항상 아래에 있다 = 같은 task 성능에서 style 비중이 더 크다. B' 는 5.0~5.1 에 고정이다. task 는 44~45 로 정상인데 style 만 8.8 에 묶여 있어, 실제 총 보상에서 style 이 차지하는 몫이 A' 의 3분의 1 이하다.

`torque_penalty` 는 세 run 전부 전 구간 정확히 0.0 이다. env.yaml 의 `torque_penalty_w: 0.0` 이 원인으로, 항이 꺼져 있는 것이다.

## 9. 관찰

1. **A' 는 baseline 대비 style 학습이 근소하게 낫다.** 최종 amp_reward 29.00 vs 28.29 (+2.5%), policy_output 0.2494 vs 0.2437, margin 0.501 vs 0.513. 조건화 자체가 판별기를 망가뜨리지 않았고 20k 이후 격차가 벌어진다. 대신 lin_vel 은 43.67 vs 45.03 으로 3.0% 낮다.
2. **B' 의 적대 게임이 43k 동안 정지해 있다.** `D(policy)` 0.060 이 5k 부터 48k 까지 ±0.001 이내로 평평하고 margin 도 0.880 에 고정이다. 정책이 style 을 학습해 D 를 압박하는 흔적이 없다. 단 이는 **곡선이 안 움직인다**는 주장이지 "B' 모션이 더 나쁘다"가 아니다. 절대값은 arch 간 비교 불가이고, 실제로 B' 의 lin_vel(44.93)과 ep_len(997)은 세 run 중 가장 높다. 모션 품질 판정은 롤아웃 쪽 증거가 필요하다.
2b. **다만 style 이 받는 최적화 예산은 실제로 작다.** `total = 0.5·task + 0.5·style` 에서 style 항이 8.8 인데 task 는 44.9 이므로, B' 의 총 보상에서 style 이 차지하는 몫은 A' 의 3분의 1 이하다. 이것은 스케일 문제가 아니라 config 수준의 사실이다.
3. **B' 의 σ 발산은 원본 run 500 iter 부근에서 시작됐고 resume 과 무관하다.** 5k 에서 이미 A' 의 3.9배(1.495 vs 0.383), 16k 에서 9.3배이고 seam 앞뒤는 3.582 → 3.604 로 연속이다. 재시도 판정은 5k 만 봐도 된다.
3b. **σ 발산과 판별기 지표는 시간적으로 분리돼 있다.** σ 가 4.0배 커지는 동안 margin 은 0.2% 만 변하고, B' 의 1차 차분 상관은 전부 |r| < 0.03 (A' 는 최대 0.05) 이다. "D 가 너무 강해서 σ 가 터졌다" 는 서술은 이 로그로는 뒷받침되지 않는다. 다만 B' 만 grad penalty 와 logit reg 가 꺼져 있고 B' 만 σ 가 발산한 것은 사실이다 (n=1 정황).
4. **task 보상은 세 run 이 사실상 같다.** lin_vel 최종 45.03 / 43.67 / 44.93, ep_len 전부 995 근처. B' 의 mean_reward 532 가 낮은 것은 lerp 로 절반을 차지하는 style 항이 8.8 에 머물러서다.
5. **B' resume 의 재워밍은 50 iter 로 짧다.** replay buffer 소실 영향은 amp_reward 한 점(16,700 부근)에만 남았고 이후 곡선은 seam 이전 궤적을 그대로 잇는다. seam 을 이유로 B' 후반 수치를 의심할 근거는 없다.

## 10. 산출 파일

- 그림: `reports/leg_imitation/_comparisons/conditional_discriminator/figures/train_disc_outputs.png`, `train_disc_losses.png`, `train_policy_stats.png`, `train_reward_balance.png`
- 캐시: `reports/leg_imitation/_comparisons/conditional_discriminator/metrics/train_curves/{base,A,B1,B2}.npz` (총 12.2 MB, tag 별 `<tag>__step` / `<tag>__val` 배열)

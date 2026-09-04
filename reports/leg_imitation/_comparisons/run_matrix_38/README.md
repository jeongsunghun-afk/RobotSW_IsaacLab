# leg_imitation_tracking_rma 38 run — 설정 diff 행렬과 성과 결합표

> 작성 일시: 2026-09-04 10:05
> 비교 대상: `logs/rsl_rl/leg_imitation_tracking_rma/` 의 38 run 전수 — `params/{env,agent}.yaml` 기계적 diff (특정 두 run 비교가 아님)

이 문서는 판정 문서가 아니다. "어느 run 이 무엇이 달랐고 어떤 학습 수치로 끝났나" 를 한 곳에서 조회하기
위한 **레지스트리**다. 각 축의 판정은 해당 비교 문서를 본다.

생성 방법: 두 yaml 을 `env.`/`agent.` 접두사로 재귀 flatten 해 dotted key 로 diff. 노이즈 키
(`seed`, `log_dir`, `run_name`, `video_recorder.*`, `*_files`) 제외. 학습 지표는 각 run 의
`events.out.tfevents.*` 를 스트리밍 파싱해 최종값을 뽑았다.

## 1. config 스키마에 기능이 추가된 시점 = 프로젝트 연표

기존 run 에 없던 키가 생겼다는 것은 그 시점에 새 기능이 config 에 노출됐다는 뜻이다.

| 최초 등장 run | 추가된 키 | 기능 |
|---|---|---|
| `2026-07-28_09-52-11_rma_dr_v4` | `agent.amp.fusion`, `agent.amp.task_reward_lerp` | task 보상 ↔ AMP style 보상 lerp 혼합 |
| `2026-07-31_10-01-38_symmetry_merged` | `agent.algorithm.symmetry_cfg.{use_data_augmentation, use_mirror_loss, mirror_loss_coeff, data_augmentation_func}` | 좌우 대칭 augmentation / mirror loss |
| `2026-07-31_18-16-01_torque_pen_1e5_smr` | `env.torque_penalty_w` | 토크 패널티 |
| `2026-08-04_13-47-09_..._stand_deadzone` | `env.cmd_deadzone`, `env.rel_stand_envs`, `env.stand_reset_joint_noise` | 정지 출발 리셋 전략 |
| `2026-08-18_09-28-11_..._wcmd` | `env.motion_weight_mode` | 클립 샘플링 가중치 |
| `2026-08-28_13-20-04_rsimatch_...` | `env.rsi_match_command`, `env.rsi_match_temperature` | RSI 시 명령 속도 매칭 |
| `2026-09-01_09-04-53_cmdchg_...` | `env.resample_command_in_episode` | 에피소드 중 명령 재샘플 (죽은 훅 부활) |
| `2026-09-02_10-36-37_condspeed_...` | `env.amp_cond_mode`, `env.amp_cond_v_max`, `env.amp_cond_yaw_max`, `agent.amp.disc_arch`, `agent.amp.amp_cond_dropout`, `agent.amp.amp_cond_reward_blend` | 조건부 discriminator + `drail` 구조 |
| `2026-09-02_17-56-53_condmatch_...` | `env.amp_cond_expert_sampling`, `env.amp_cond_match_temperature` | expert 조건 샘플링 (라벨 누설 수정) |

위 목록에 없는 키(`vel_err_scale`, `lin_vel_x_max`, `episode_length_s`, `tar_change_time_*`,
actuator `effort_limit_sim`, `amp_weight`, `reward_coef`, `discriminator_learning_rate` 등)는
1번 run 부터 이미 존재했고 값만 바뀌었다.

## 2. 실제로 값이 변한 축

| 축 | 값의 변천 | 기본값(run1) |
|---|---|---|
| `env.motion_file` | smr → merged → new_smr → run2t14 → trot110150 → waistfix_ds14 → new_smr | `smr_leg_pkl` |
| actuator `effort_limit_sim` (hip/thigh/waist · calf · feet) | 23.8 / 35.7 / 47.6 → **71.4 / 107.1 / 142.8** (run5 부터 영구) | 23.8 / 35.7 / 47.6 |
| `env.vel_err_scale` | 0.5 → 1.5 → 1.0 → **1.5** | 0.5 |
| `env.lin_vel_x_max` | 4.0 → **3.2** (run20 부터) | 4.0 |
| `env.motion_weight_mode` | 없음 → `command_uniform` ↔ `length` ↔ `command_uniform_mirror` | 미도입 |
| `agent.amp.task_reward_lerp` | 없음 → 0.5 → 0.65 → 0.3 → **0.5** (run17 부터 고정) | 미도입 |
| `env.torque_penalty_w` | 없음 → 1e-05 ↔ 0.0 (run25 부터 0.0) | 미도입 |
| `symmetry_cfg` | `None` → aug → aug+mirror(1.0) → aug → `None` → **aug** (run25 부터) | `None` |
| `env.reset_strategy` | `random` → `random_stand`(run14) → `random`(run25) | `random` |
| `env.cmd_deadzone` | 없음 → 0.1 → 0.0 | 미도입 |
| `env.episode_length_s` | 10.0 → **20.0** (run32 부터) | 10.0 |
| `env.tar_change_time_max` | 7.0 → **4.0** (run32 부터, `min` 은 4.0 불변) | 7.0 |
| `env.amp_cond_mode` | 없음 → `speed` ↔ `none` | 미도입 |
| `agent.amp.disc_arch` | 없음(=mlp) → `mlp` ↔ `drail` | 미도입 |
| `agent.amp.discriminator_learning_rate` | 2.5e-4 → 2.5e-5(run37) → 2.5e-4(run38) | 2.5e-4 |
| `agent.policy.init_noise_std` + obs 정규화 | (1.0, False, False) → **(0.25, True, True)** (run2·3 부터 고정) | (1.0, False, False) |
| `agent.amp.amp_weight` + `reward_coef` | (0.3, 0.08) → **(1.0, 2.0)** (run2 부터 고정) | (0.3, 0.08) |

run 38개 중 resume 은 2건(run7 ← run5, run38 ← run36)뿐이고 나머지는 전부 fresh start 다.

## 3. ⚠️ 이 시리즈의 학습 로그를 읽을 때의 함정

`episode_length_s` 가 run32(2026-09-02)부터 10 → 20 으로 바뀌어 **두 지표가 다르게 움직인다.**

- `Train/mean_reward` 는 **에피소드 합**이라 에피소드가 2배 길어지면 값도 자동으로 2배가 된다.
  초당으로 나눠야 비교된다(389.6/10 = 38.96 vs 730.8/20 = 36.54 — 원값만 보면 "1.9배 개선"으로 오독).
- `Episode_Reward/*` 는 rsl_rl 이 이미 `episode_length_s` 로 나눈 **초당 평균**이라 그대로 비교된다.
  실측: ep20 run 의 `lin_vel_reward` 45.357 vs ep10 run 46.122 = 0.983배(2배가 아님).
- `Train/mean_episode_length` 의 상한도 500 step → 1000 step 으로 다르다.

## 4. run → 설정·성과 결합표

`max_iter >= 10000` 인 34 run. `mean_rwd/s` = `Train/mean_reward` 최종값 / **명목** `episode_length_s`
(조기 종료 run 은 분모가 명목값이라 과소평가된다). `amp`·`lin_vel` 은 `Episode_Reward/*` 원값(초당).
`ep_len` 은 최종값/상한 [step]. `σ` 는 `Policy/mean_noise_std` 최종값이며 최댓값 > 2.0 이면 ⚠ 표기.

| run (날짜 접두사 생략) | max_iter | dataset | vel_err | cond/disc | mean_rwd/s | amp | lin_vel | ep_len | σ |
|---|---|---|---|---|---|---|---|---|---|
| `rma_dr` | 49600 | smr | 0.5 | –/mlp | 21.65 | 0.02 | 22.90 | 244/500 | 9.98 ⚠ |
| `rma_dr_v3` | 12400 | smr | 0.5 | –/mlp | 67.75 | 24.39 | 43.72 | 494/500 | 0.24 |
| `2026-07-29_13-12-09` | 46300 | merged | 0.5 | –/mlp | 30.26 | 30.57 | 27.57 | 454/500 | 0.23 |
| `rma_dr_urdf85` (resume) | 23400 | smr | 0.5 | –/mlp | 35.91 | 29.85 | 43.72 | 490/500 | 0.21 |
| `symmetry_merged` | 49999 | merged | 0.5 | –/mlp | 35.90 | 28.42 | 44.41 | 485/500 | 0.25 |
| `symmetry_smr` | 30500 | smr | 0.5 | –/mlp | 35.01 | 26.65 | 44.65 | 490/500 | 0.28 |
| `torque_pen_1e5_smr` | 49999 | smr | 0.5 | –/mlp | 34.41 | 27.89 | 44.03 | 481/500 | 0.26 |
| `symmetry_smr_mirrorloss` | 19600 | smr | 0.5 | –/mlp | 33.39 | 22.28 | 44.87 | 494/500 | 0.30 |
| `torque_pen_1e5_merged` | 24100 | merged | 0.5 | –/mlp | 34.16 | 25.66 | 44.04 | 490/500 | 0.25 |
| `torque1e5_newsmr_nosym` | 49999 | new_smr | 0.5 | –/mlp | 37.84 | 43.73 | 30.60 | 494/500 | 0.14 |
| `torque1e5_newsmr_stand_deadzone` | 49999 | new_smr | 0.5 | –/mlp | 37.60 | 42.19 | 29.69 | 494/500 | 0.14 |
| `..._stand_dz_lerp065` | 34400 | new_smr | 0.5 | –/mlp | 39.04 | 33.84 | 44.18 | 490/500 | 0.20 |
| `..._stand_dz_lerp030` | 14200 | new_smr | 0.5 | –/mlp | 34.33 | 37.21 | 23.36 | 476/500 | 0.17 |
| `torque1e5_trotmirror_stand_dz` | 49999 | new_smr | 0.5 | –/mlp | 39.73 | 41.70 | 35.47 | 494/500 | 0.14 |
| `..._trotmirror_stand_dz_ds14` | 49999 | new_smr | 0.5 | –/mlp | 40.86 | 45.51 | 32.74 | 499/500 | 0.12 |
| `..._trotmirror_stand_dz_run2t14` | 33600 | run2t14 | 0.5 | –/mlp | 39.02 | 44.99 | 30.35 | 499/500 | 0.12 |
| `..._trotmirror_stand_dz_ds14_vmax32` | 49999 | new_smr | 0.5 | –/mlp | 40.99 | 44.13 | 36.95 | 495/500 | 0.11 |
| `..._vmax32_trot110150` | 49999 | trot110150 | 0.5 | –/mlp | 42.90 | 44.48 | 42.11 | 499/500 | 0.13 |
| `..._vmax32_trot110150_wcmd` | 49999 | trot110150 | 0.5 | –/mlp | 41.13 | 38.41 | 46.39 | 499/500 | 0.14 |
| `..._vmax32_waistfix` | 49999 | waistfix_ds14 | 0.5 | –/mlp | 41.89 | 46.19 | 35.69 | 499/500 | 0.11 |
| `..._vmax32_waistfix_wcmd` | 49999 | waistfix_ds14 | 0.5 | –/mlp | 39.38 | 35.19 | 45.76 | 499/500 | 0.13 |
| `..._vmax32_ds14_wcmd` | 49999 | new_smr | 0.5 | –/mlp | 39.95 | 34.06 | 46.41 | 499/500 | 0.22 |
| **`velscale15_ds14_wcmd_vmax32`** | 49999 | new_smr | **1.5** | –/mlp | 38.96 | 31.77 | 46.12 | 499/500 | 0.21 |
| `velscale10_ds14_wcmd_vmax32` | 49999 | new_smr | 1.0 | –/mlp | 39.01 | 32.37 | 46.29 | 499/500 | 0.22 |
| `velscale10_ds14_cumirror_vmax32` | 49999 | new_smr | 1.0 | –/mlp | 41.31 | 36.59 | 46.04 | 499/500 | 0.17 |
| `rsimatch_velscale15_ds14_wcmd_vmax32` | 49999 | new_smr | 1.5 | –/mlp | 39.96 | 34.43 | 46.50 | 499/500 | 0.22 |
| `cmdchg_velscale15_ds14_wcmd_vmax32` | 42400 | new_smr | 1.5 | –/mlp | 36.92 | 29.39 | 45.23 | 499/500 | 0.30 |
| `velscale15_ds14_cumirror_vmax32` | 49999 | new_smr | 1.5 | –/mlp | 40.75 | 36.27 | 45.64 | 496/500 | 0.18 |
| **`cmdchg4s_ep20_...`** (uncond) | 49999 | new_smr | 1.5 | none/mlp | 36.54 | 28.45 | 45.36 | 999/1000 | 0.31 |
| `condspeed_cmdchg4s_ep20_...` | 10400 | new_smr | 1.5 | speed/mlp | 24.89 | 6.47 | 44.46 | 999/1000 | 7.31 ⚠ |
| **`condmatch_cmdchg4s_ep20_...`** (cond-mlp) | 49999 | new_smr | 1.5 | speed/mlp | 36.36 | 28.90 | 43.93 | 999/1000 | 0.27 |
| `condmatch_drail_cmdchg4s_ep20_...` | 16700 | new_smr | 1.5 | speed/drail | 25.74 | 8.39 | 43.47 | 986/1000 | 3.63 ⚠ |
| `drail_lr25e6_cmdchg4s_ep20_...` | 29900 | new_smr | 1.5 | none/drail | 27.22 | 11.54 | 43.98 | 989/1000 | 5.06 ⚠ |
| `condmatch_drail_resume16k7_...` | 47400 | new_smr | 1.5 | speed/drail | 26.26 | 8.55 | 44.27 | 987/1000 | 14.17 ⚠ |

`max_iter < 10000` 이라 제외한 4 run: `rma_dr_v2`(2700) · `rma_dr_v4`(1100) · `rma_dr_urdf85`(6700) ·
`condspeed_drail_cmdchg4s_ep20_...`(4700).

### σ 는 최종값이 아니라 **꺾였는지**로 봐야 한다

⚠ 표시된 5 run 은 **전부 `σ최종 == σ최대`** 다 — 끝까지 단조 상승했고 정점을 지나 내려온 적이 없다
(`rma_dr` 9.977 · `condspeed_cmdchg4s` 7.314 · `drail_lr25e6` 5.057 · `condmatch_drail` 3.626 ·
`condmatch_drail_resume16k7` 14.168).

나머지 29 run 은 전부 `σ최종 < σ최대` 로 뚜렷이 다르다. 예: `velscale15_ds14_wcmd_vmax32` 는
0.214 vs 0.371, `cmdchg4s_ep20` 은 0.310 vs 0.632 다.

즉 **한 시점의 σ 값만으로는 발산과 "정점을 찍고 회복 중"을 구분할 수 없다.** `cmdchg4s` 가
정확히 그 사례로, 20k 에 σ 0.407(정점 부근)이라 나빠 보였지만 50k 에는 0.310 으로 내려왔다 —
같은 시점에 내린 기각 판정이 뒤집힌 것과 같은 구간이다.

### 학습 로그의 마지막 step ≠ 마지막 체크포인트

34 run 중 15 run 에서 `Train/mean_reward` 의 마지막 step 이 `max_iter`(체크포인트 번호)보다 크다
(Δ 7~478). `save_interval: 100` 이라 마지막 체크포인트 이후에도 로그가 더 쌓인 것이다. 위 표의
`max_iter` 는 **체크포인트 기준**이며, 평가에 쓸 수 있는 마지막 지점도 이쪽이다.

## 5. 이 표로 판정하지 말 것

이 프로젝트에서 **학습 지표가 램프·조사 능력을 예측한 적이 없다.** 실례:

- `velscale15`(1.5) 와 `velscale10`(1.0) 의 `mean_rwd/s` 는 38.96 vs 39.01 로 사실상 같은데,
  램프에서는 한쪽이 평균 달성률 90.0%, 다른 쪽이 50k 에서 4/8 붕괴한다
  (→ [`arch_scaling_study`](../arch_scaling_study/README.md)).
- `cmdchg4s`(uncond) 와 `condmatch`(cond-mlp) 의 최종 `amp` 는 28.45 vs 28.90 으로 근소한데,
  명령 전환 추종률은 6.2% vs 43.8% 로 **7배** 차이다
  (→ [`conditional_discriminator`](../conditional_discriminator/README.md)).
- `σ 발산` 플래그가 붙은 run 이 곧 거동 실패는 아니다 — `condmatch_drail_resume` 는 σ 14.17 에도
  낙상 0 이고 고속 gallop 74.7% 다(단 저속 `other` 60.5%).

능력 판정은 헤드리스 램프(`speed_ramp_record_rma.py`, 정점을 `lin_vel_x_max` 에 맞출 것)와
대량 조사(`gait_survey_multienv.py`)로 한다.

## 산출물

- 원자료 JSON·생성 스크립트는 세션 scratchpad 에만 있고 저장소에 두지 않았다. 재생성하려면
  각 run 의 `params/{env,agent}.yaml` 을 flatten diff 하고 tfevents 를 스트리밍 파싱하면 된다.
- 이 문서의 서사판: Notion 「[2026-09-04] Leg 모방학습 트래킹 회고」.

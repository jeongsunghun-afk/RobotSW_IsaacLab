# Phase 2 — latent style reward env 구축 · **학습 착수 가능 (조건부)**

2026-08-28 · 브랜치 `isaac-6.0` · Task ID `Go2-Imitation-Latent-v0`
상위 [`../2026-08-27_latent_imitation_plan/STATUS.md`](../2026-08-27_latent_imitation_plan/STATUS.md)
선행 [`../2026-08-27_phase2_style_reward_validation/`](../2026-08-27_phase2_style_reward_validation/)

---

## 0. 결론

AMP discriminator 를 **동결 motion-VAE 인코더 기반 env 별 시간창 marginal KL** 로 교체한
태스크를 만들었고, 학습 전 게이트 V1~V5 와 추가 게이트 V6 를 통과했다.

| 게이트 | 결과 | 핵심 수치 |
|---|---|---|
| V1a 순열 대조 | **PASS** | 7 블록 중 6 개가 **정확히 0.0** 오차의 순열. `rot6d` 는 순열이 **아니고**, 같은 회전의 다른 매개화임을 1e-7 로 확인 |
| V1b sim↔motion 채널 편차 | **PASS** | 계통 편차 없음. 중앙 오차 `foot_pos` **0.77 mm** · `base_height` **0.21 mm** · `joint_pos` **0.0044 rad** |
| V2 스모크 | **PASS** | `--num_envs 64 --max_iterations 20` Traceback 0, 20.7 s, 체크포인트 2 개. `--resume` 5 iter 도 통과 |
| V3 env 간 산포 | **PASS** | 한 스텝 env 간 std **최소 0.391**, 평균 변동계수 **0.938**. 학습 중(iter 19) std **0.129** / mean 0.878 |
| V4 정지 페널티 | **PASS** | 정지 env 가 held-out expert 보다 **4.79 × expert std** 낮음 (기준 ≥ 2). `D_e` 중앙 **94.7 vs 13.7** |
| V5 참조 통계 출처 | **PASS** | train 35 세션으로만 적합, val 8 세션 배제, **교집합 0** — 파일 내용으로 확인 |
| V6 리셋 warm-up 편향 (추가) | **PASS(조건부)** | warm-up 이 전체 step 의 **7.3 %**, 에피소드 나이 ↔ `r_style` 상관 **+0.332** (리셋이 아니라 **생존**이 보상받는 방향) |

**단, 학습 착수 전에 읽어야 할 한계가 두 개 있다** (§7). 하나는 이 신호가 **정지는 강하게
잡지만 "그럴듯하지만 다른 걸음걸이" 는 못 잡는다**는 것(Phase 2 오프라인 검증의 재확인),
다른 하나는 **`c_kl` 이 학습 결과를 좌우하는 자유 파라미터**라는 것이다.

검증 과정에서 baseline 승계가 **조용히 틀리는** 지점 세 개를 찾아 고쳤다 —
`lerp` 스케줄의 resume 되감기(§7-5), warm-up 중립값의 리셋 유인(§7-4),
`standing_pose_reward_w` 의 스케일 불일치(§7-6). 셋 다 20 iter 스모크로는 드러나지 않는다.

**학습은 띄우지 않았다.** 스모크 20 iter + resume 5 iter 까지만 했다.

---

## 1. 기호

| 기호 | 의미 | 단위 |
|---|---|---|
| `x_vae` | 프레임 상태 특징 49-D | 혼합 |
| `z` | 인코더 사후분포 평균 18-D | 무차원 |
| `N` (`window_n`) | env 별 시간창 길이 (전이 쌍 수) | 개 |
| `mu_ref`, `var_ref` | **train 세션** expert latent 의 평균·대각 분산 | 무차원 |
| `D_e` | env `e` 의 창 marginal KL | nat |
| `c_kl` | 보상 민감도 | 1/nat |
| `offset` | `D_e` 오프셋 (expert 창 중앙값) | nat |
| `lerp` | task 보상 비중 (`task_reward_lerp`) | 무차원 |

보상:

```
env e 의 최근 N 전이쌍에서   mu_e, var_e   (대각)

D_e = KL( N(mu_e, diag var_e) || N(mu_ref, diag var_ref) )
    = 0.5 * sum_d [ log(var_ref_d / var_e_d)
                    + (var_e_d + (mu_e_d - mu_ref_d)^2) / var_ref_d - 1 ]

r_style_e = exp( -c_kl * (D_e - offset) )

total_e   = lerp * task_e + (1 - lerp) * r_style_e
```

★ 스텝 로그밀도 `d_step = -log N(z; mu_ref, Sigma_ref)` 는 **쓰지 않았다**. 정지가 expert
보다 높은 점수를 받는 것이 오프라인에서 확인됐다(클립 AUROC `cat_stand` 0.055).
정지 판별의 기전은 창 안 `z` 분산의 축퇴(`var_e << var_ref`)이고, 그것은 분포 수준
발산에서만 보인다.

★ 배치 전체에 같은 상수를 주면 PPO advantage 에서 value baseline 에 흡수되어 정책
gradient 기여가 0 이다. 그래서 창을 **env 마다** 둔다 (V3 가 이것을 잰다).

---

## 2. 확정된 설계 값

| 항목 | 값 | 출처 |
|---|---|---|
| 인코더 | `2026-08-27_phase1_motion_vae/gaussfix_reg/runs/base_s0/motion_vae.pt` | Phase 1 지시 |
| | β=0 Gaussian · `gauss_fixed_logvar` −4.0 · latent 18 · stride 2 · `predict_ahead` 1 | 체크포인트 `args` |
| 인코더 사용 | **인코더만** 로드 · `eval()` + `requires_grad_(False)` · 디코더 미사용 | |
| `N` (`window_n`) | **8** | 오프라인 창 스윕에서 8 쌍까지 정지 판별 유지 |
| `mu_ref`, `var_ref` | train 35 세션 전이쌍 **45,325** 개 | `latent_ref_stats.pt` |
| `offset` | **10.888** nat (expert 창 `D_e` 중앙값) | 같은 파일 |
| `c_kl` | **0.01589** 1/nat | §3 에서 데이터로 정함 |
| 전이 쌍 간격 | **33.33 ms** (= stride 2 × 1/60 × k 1) | 인코더 학습 설정 |
| policy dt | **20 ms** (50 Hz) | env cfg |
| `lerp` 스케줄 | 1.0 → 0.5, anneal 5000 iter, **절대 iter 기준** | AMP baseline 승계 + §7-5 수정 |
| `neutral_reward` | **0.9** (= expert 창 p75 의 보상) | §7-4 |
| `standing_style_substitute` | **False** (baseline 은 True) | §7-6 |

---

## 3. ★ `c_kl` 은 Phase 2 §5 의 0.01 을 쓰면 안 된다 — 다시 잡았다

§5 의 스윕은 `d_step`(p5/p50/p95 ≈ −27 / −22 / +6.5 nat) 기준이다. `D_e` 는 **완전히 다른
스칼라**이므로 그 값을 옮겨 쓸 수 없다. `build_go2_latent_ref_stats.py` 가 train expert 에서
길이 정확히 `N` = 8 인 창 **41,440** 개를 뽑아 분포를 잰다:

| 분위 | p1 | p5 | p25 | **p50** | p75 | p95 | p99 |
|---|---|---|---|---|---|---|---|
| `D_e` [nat] | 4.17 | 5.17 | 7.62 | **10.89** | 17.52 | 49.55 | 63.57 |

**첫 시도(p95 기준)는 실패했고 실측으로 폐기했다.** `c_kl = -ln(0.8)/(p95−p50) = 0.00577`
로 잡으면 정책이 실제로 머무는 구간에서 보상이 눌린다 — 스모크 iter 19 에서
`style_reward` 평균 **0.951**, held-out expert 평균 **0.965**. 격차 1.5 % 다.
원인은 `N` = 8 창의 `D_e` 분포 꼬리가 매우 무겁고(p50 10.9 대 p95 49.6) **그 꼬리가 스타일
차이가 아니라 짧은 창의 분산 추정 잡음**이라는 데 있다.

상사분위 기준으로 다시 잡았다:

```
c_kl = -ln(0.9) / (p75 - p50) = 0.10536 / 6.631 = 0.01589
```

| 값 | 이전 (p95 기준) | **채택 (p75 기준)** |
|---|---|---|
| `c_kl` | 0.00577 | **0.01589** |
| expert (held-out val) `r_style` 평균 | 0.965 | **0.916** |
| 랜덤 액션 sim `r_style` 평균 | 0.725 | **0.684** |
| 정지 sim `r_style` 평균 | 0.347 | **0.171** |
| 학습 중(iter 19) `style_reward` std / mean | 0.060 / 0.951 | **0.129 / 0.878** |

`c_kl` 은 cfg (`latent.c_kl`) 로 노출돼 있고 `None` 이면 참조 통계의 값을 쓴다.
(학습 중 수치는 §7-6 반영 후 재측정한 스모크 값이다.)

---

## 4. V1 — `x_vae` 와 AMP obs 의 관계

### 4-1. V1a: 두 **motion** 경로 대조 (`walk__D1_001_...F630_F830_011.pkl`, 198 프레임)

`x_vae` 는 `train_go2_motion_vae.py:build_x_vae`, AMP 는 baseline env 의
`_compute_amp_obs` + `_apply_root_rot_tan_norm` 을 같은 프레임에 태웠다.

| `x_vae` 블록 | AMP slice | max\|err\| |
|---|---|---|
| `base_height` [0:1] | [24:25] | **0.000e+00** |
| `lin_vel` [7:10] | [25:28] | **0.000e+00** |
| `ang_vel` [10:13] | [28:31] | **0.000e+00** |
| `foot_pos` [13:25] | [31:43] | **0.000e+00** |
| `joint_pos` [25:37] | [0:12] | **0.000e+00** |
| `joint_vel` [37:49] | [12:24] | **0.000e+00** |

**`rot6d` [1:7] 만 순열이 아니다.**

```
x_vae  heading_relative_rot6d : [ R·x , R·y ]     회전행렬의 1·2 열
AMP    root_rot_tan_norm      : [ R·x , R·z ]     tan=(1,0,0)·norm=(0,0,1) 회전
```

| 대조 | max\|err\| | 읽기 |
|---|---|---|
| 직접 대조 `x_vae[1:7]` vs `AMP[43:49]` | **1.056e+00** | 같은 값이 아니다 |
| 1 열: `x_vae[1:4]` vs AMP tan | **2.30e-07** | 첫 3 성분은 같다 |
| `(R·x) × (R·y)` vs AMP norm | **3.43e-07** | **같은 회전의 다른 매개화**임이 확인됨 |

정보량은 같지만(`R·z = R·x × R·y`) 선형 재배열로는 못 만든다. **인코더가 `x_vae` 순서로
학습됐으므로 env 는 `x_vae` 규약을 직접 계산한다** (`_heading_relative_rot6d`).
AMP obs 를 재배열해 쓰는 경로는 채택하지 않았다.

함께 확정한 두 가지 (추측하지 않고 왕복 검증했다):

- **쿼터니언 규약**: env `body_quat_w` 는 xyzw, `motion_lib` 는 wxyz. `reference_x_vae` 가
  소비 경계에서 `convert_quat` 한다. V1a 의 1 열 일치(2.3e-7)가 이 규약이 맞다는 증거다.
- **관절 순서**: `x_vae` 는 motion `DOF_NAMES` 순서, env 는 IsaacLab 순서.
  `self._motion_dof_indices` 는 motion→IsaacLab 방향이라 **역순열**이 필요하고,
  `torch.argsort` 로 만들었다. (실측 `motion_dof_indices = [0,3,6,9, 1,4,7,10, 2,5,8,11]`)
  ★ **이 역순열을 실제로 검증하는 것은 V1a 가 아니라 V1b 다.** V1a 는 양쪽 경로 모두
  `DOF_NAMES` 순서를 그대로 쓰므로 `_x_vae_dof_indices` 를 한 번도 지나지 않는다.
  V1b 는 env 가 `joint_pos[:, _x_vae_dof_indices]` 를 쓰고 참조는 무순열인데 중앙 오차가
  0.0044 rad 이다 — 순열이 틀렸다면 관절이 뒤바뀌어 rad 급 오차가 났을 것이다.
  **인코더를 갈아끼울 때 V1a 만 다시 돌리면 안 된다.**

### 4-2. ★ V1b: **sim 실측** `x_vae` vs **motion** `x_vae`

V1a 는 두 경로 모두 `motion_lib` 파생이라 **학습 중 실제로 벌어지는 대조를 재지 않는다**.
`mu_ref` 는 motion 파생이고 정책 롤아웃 `z` 는 sim 실측 파생이다. 채널 하나라도 계통
편차가 있으면 `D_e` 가 스타일이 아니라 sim/motion 특징 편차를 재게 된다.
구체적 용의자: `motion_lib._go2_fk_foot_pos` 는 관절각에서 **FK 로 계산한** 발 위치,
env 는 `body_pos_w` 를 base-local 로 돌린 **sim 실측** 발 위치다.

RSI 로 참조 프레임에 세운 로봇(96 env)의 env-계산 `x_vae` 를 같은 프레임의 motion `x_vae`
와 대조했다. physics view 갱신을 위해 substep 하나(5 ms)를 돌리되, 그동안 액추에이터가
관절을 기본 자세로 끌지 않도록 PD 목표를 **리셋 자세 그대로** 주었다.

| 블록 | 단위 | max\|err\| | p95\|err\| | **median\|err\|** | 참조 RMS |
|---|---|---|---|---|---|
| `base_height` | m | 1.04e-02 | 2.47e-03 | **2.07e-04** | 3.69e-01 |
| `rot6d` | – | 6.69e-01 | 7.13e-03 | **1.26e-04** | 5.78e-01 |
| `lin_vel` | m/s | 4.43e-01 | 4.83e-02 | **5.62e-03** | 4.45e-01 |
| `ang_vel` | rad/s | 1.21e+01 | 3.65e-01 | **4.84e-02** | 5.22e-01 |
| **`foot_pos`** | m | 1.35e-02 | 4.26e-03 | **7.67e-04** | 1.99e-01 |
| `joint_pos` | rad | 5.18e-02 | 2.26e-02 | **4.38e-03** | 1.18e+00 |
| `joint_vel` | rad/s | 9.58e+00 | 2.07e+00 | **2.65e-01** | 1.37e+00 |

**판정: 계통 편차 없음.** 가장 걱정했던 `foot_pos`(FK 대 sim)의 중앙 오차가 **0.77 mm**
= 참조 RMS 의 0.39 % 다. FK 링크 오프셋이 USD 와 어긋났다면 12 채널 전체에 cm 급 상수
오프셋이 나타났을 텐데 그렇지 않다.

max 열이 큰 것은 5 ms substep 동안 접촉·중력으로 튄 소수 env 때문이다 — 속도 블록에서
특히 크고(`joint_vel` max 9.58 rad/s), 이는 채널 정의 문제가 아니라 **물리 시간 진행**의
결과다. median 과 p95 가 작다는 것이 그 해석을 뒷받침한다.

---

## 5. V3 / V4 — 보상이 env 마다 다른가, 정지가 벌점을 받는가

세 군을 비교했다. **sim 군에 상태를 인위로 주입하지 않았다** — 액션만 바꿨다.

| 군 | 만드는 법 |
|---|---|
| `expert` | **held-out val 세션** expert 창 22,733 개 (sim 아님). 정책이 도달해야 할 값 |
| `static` | sim 48 env, 액션 0 → PD 가 기본 자세를 잡아 로봇이 **실제로 서 있다** |
| `free` | sim 48 env, 랜덤 액션(σ 0.4) → 움직인다 |

sim 은 200 policy step 을 굴리고 **뒤 100 step** 만 썼다(RSI 전이 구간 배제).
`early_termination` / `domain_rand` / `standing_style_substitute` 를 끄고 `rel_rest_init` 0.

| 군 | `r_style` 평균 | std | 변동계수 | `D_e` 중앙 [nat] |
|---|---|---|---|---|
| `expert` (held-out val) | **0.9158** | 0.1557 | 0.170 | **13.74** |
| `static` (sim, act = 0) | **0.1705** | 0.1673 | 0.981 | **94.67** |
| `free` (sim, random) | 0.6843 | 0.4020 | 0.588 | 17.66 |

### V3 — 한 스텝 안의 env 간 산포

| 지표 | 값 |
|---|---|
| 한 스텝 env 간 std, **최소** | **0.3907** |
| 한 스텝 env 간 std, 평균 | 0.4008 |
| 평균 **변동계수** (std / mean) | **0.938** |
| 정지 군만의 env 간 std, 최소 | 0.1461 |
| 학습 중 (스모크 iter 19, 64 env) std / mean | **0.1082 / 0.8901** |

"std > 0" 은 부동소수 잡음으로도 통과하므로 효과크기를 병기했다. 통제 프로브에서는
변동계수 0.94, **실제 학습 초기(정책이 균질할 때)에도 0.12** 다. 배치 상수가 아니다.

### V4 — 정지 페널티

```
(expert 평균 - static 평균) / expert std = (0.9158 - 0.1705) / 0.1557 = 4.79
```

**기준 ≥ 2 를 통과한다.** `D_e` 중앙값으로 보면 정지가 94.7 로 expert 13.7 의 6.9 배다 —
창 안 `z` 분산이 축퇴해 `log(var_ref/var_e)` 항이 폭증하는, 설계가 의도한 그 기전이다.
`static` vs `free` 는 1.28 × free std 인데, `free` 군의 std 가 0.40 으로 큰 것은 랜덤 액션
로봇이 "허우적댐" 과 "쓰러져 정지" 사이를 오가기 때문이다.

### `var_floor` 진단

`var_floor` = 1e-4 가 신호를 나르고 있지 않은지 확인했다 — expert 창에서 클램프에 걸린
비율 **0.0093 %**, 전역 참조 통계에서는 18 차원 중 **0 개**다. 보상이 이 임의 하이퍼
파라미터의 함수가 되어 있지는 않다.

---

## 6. V2 / V5

### V2 — 스모크

```
CUDA_VISIBLE_DEVICES=1 /home/user/miniconda3/envs/isaac-6.0/bin/python \
  scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-Latent-v0 --num_envs 64 --max_iterations 20 --headless
```

Traceback 0, 20.7 s, `logs/rsl_rl/go2_imitation_latent/2026-08-28_11-23-24/`
에 `model_0.pt` / `model_19.pt` 저장. iter 19 로그:

| 항목 | 값 |
|---|---|
| `task_reward_lerp` | 0.9981 (anneal 5000 iter 의 19/5000 지점) |
| `style_reward_mean` / `style_reward_std` | 0.8775 / **0.1292** |
| `latent_kl_mean` | 46.44 nat (창이 찬 env 만, 롤아웃 평균 — §7-7) |
| `Episode_Reward/*` | `lin_vel_reward` · `yaw_vel_reward` · `style_reward` 정상 기록 |
| `estimator_lin` / `estimator_ang` | RMA estimator 정상 학습 |
| `hist_latent_loss` | 0.56 (DAgger 경로 정상) |

resume 도 확인했다 — `--resume --load_run <run> --checkpoint model_19.pt --max_iterations 5`
가 Traceback 없이 완주하고 `task_reward_lerp` 가 되감기지 않는다 (§7-5).

### V5 — 참조 통계 출처

`latent_ref_stats.pt` 안에 적합에 쓴 세션 목록이 그대로 들어 있다.

| 항목 | 값 |
|---|---|
| train 세션 | **35** |
| 배제된 val 세션 | **8** |
| **교집합** | **0** |
| 전이쌍 / 창 | 45,325 / 41,440 |
| ckpt sha256 (앞 16) | `10cfc9a121477e50` |
| split json | `2026-08-27_phase0_dataset_build/metrics/session_split.json` |

§5 의 `expert` 군은 **val** 세션인데, 참조는 train 으로만 적합했으므로 누수가 아니다 —
오히려 held-out 기준선이라 판정을 어렵게 만드는 방향이다.

---

## 7. ★ 한계 — 학습 착수 판정과 함께 읽을 것

### 7-1. 이 신호는 "그럴듯하지만 다른 걸음걸이" 를 못 잡는다

Phase 2 오프라인 검증에서 창 marginal KL 은 좌우 다리 교환 0.547 · 앞다리 교환 0.578 ·
base-다리 위상 분리 0.536~0.567 로 **우연 수준**이었다. 창 길이를 8~64 로 바꿔도 같다.
원인은 창이 아니라 **인코더 입력이 한 스텝 전이**라 보행 위상을 볼 수 없다는 구조다.

이번 env 실측이 같은 방향을 재확인한다 — 랜덤 액션 로봇의 `D_e` 중앙값이 **17.66** 으로
held-out expert 13.74 와 크게 다르지 않다. **정지는 6.9 배로 확실히 벌하지만, 움직이기만
하면 걸음걸이의 질을 거의 구분하지 못한다.**

따라서 이 arm 이 기대할 수 있는 것은 "AMP 만큼 좋은 걸음걸이" 가 아니라
**"AMP 의 정지 exploit 없이 task 를 푸는 것"** 에 가깝다. STATUS.md §6-2 권고 1
(인코더 입력 창 확대 재학습, `w` = 10)이 이 한계를 직접 겨냥하며, 그 재학습 비용은 15 초다.
**본 학습을 띄우기 전에 그것을 먼저 할지는 팀리드의 결정 사항이다** — env 는 `latent.encoder_ckpt`
/ `latent.ref_stats` 두 경로만 바꾸면 새 인코더로 갈아끼울 수 있게 만들어 두었다.

### 7-2. `c_kl` 은 결과를 좌우하는 자유 파라미터다

§3 이 보인 대로 보정 기준을 p95 에서 p75 로 바꾸는 것만으로 정지 군 보상이 0.347 → 0.171
로 두 배 넘게 갈렸다. 데이터에서 정하는 절차를 코드에 박아 뒀지만(`build_go2_latent_ref_stats.py`),
"p75 → 0.9" 라는 선택 자체는 **원리에서 유도된 값이 아니다**. 본 학습에서 이 축은
`lerp` 와 함께 A/B 대상으로 남겨야 한다.

### 7-3. 전이 쌍 간격은 보간이다

인코더 전이 간격 33.33 ms 가 policy dt 20 ms 의 정수배가 아니므로, `x_prev` 는 링버퍼의
두 프레임(`t−20 ms`, `t−40 ms`)을 α = 2/3 으로 **선형보간**해 만든다. `rot6d` 블록의
선형보간은 엄밀히 유효회전이 아니지만 20 ms 구간 각변화가 작아 오차가 무시 가능하다.
2 스텝(40 ms)을 쓰면 20 % 계통 편향이 정책 `z` 전체에 걸리므로 그 쪽을 택하지 않았다.

### 7-4. ★ 리셋 warm-up 중립값 — 1.0 은 리셋을 보상한다 (V6)

창이 찰 때까지 약 **11 step**(0.22 s) 동안 env 는 `neutral_reward` 를 받는다.
처음엔 이것을 1.0(expert 창 중앙값의 보상)으로 두고 "에피소드가 20 초라 영향 1 %" 라고
썼는데 **틀렸다**. `episode_length_s = 20` 은 time-out 상한이고, 초기 학습의 실측 평균
에피소드 길이는 **107~112 step (약 2.2 s)** 이다. 게다가 1.0 은 `r_style` 의 사실상 상한이라
"종료 → 리셋 → 공짜 최댓값" 이 exploit 경로가 된다 — 정지가 0.17 을 받는 신호 옆에서다.

`neutral_reward` 를 **0.9** 로 내렸다. `c_kl` 보정 정의상 이 값은 **expert 창 상사분위(p75)**
의 보상이고, 초기 정책 실측 평균(0.878)과 held-out expert 평균(0.916) 사이라 어느 쪽으로도
유인을 만들지 않는다.

V3/V4 는 `early_termination=False` 로 측정하므로 이 항을 **구조적으로 볼 수 없다.**
그래서 게이트를 하나 더 넣었다 — `early_termination=True` 로 96 env × 200 step:

| 항목 | 값 |
|---|---|
| warm-up step 비율 | **7.31 %** |
| 평균 에피소드 나이 | 77.0 step |
| `r_style` — warm-up | **0.9000** (= `neutral_reward`) |
| `r_style` — 창이 찬 뒤 | 0.7460 |
| **에피소드 나이 ↔ `r_style` 상관** | **+0.332** |

상관이 **양수**다 — 오래 산 env 가 더 높은 `r_style` 을 받는다. 즉 순효과는 리셋이 아니라
**생존**을 보상하는 방향이다. 다만 랜덤 액션 정책 기준 warm-up 0.900 vs 이후 0.746 이라
국소적으로는 여전히 +0.154 의 잔여 편향이 있다. 정책이 0.9 근처를 달성하게 되면 이 항은
사라진다. **잔여 편향이 남아 있으므로 V6 는 PASS(조건부)** 로 기록한다.

### 7-5. ★ `lerp` 스케줄의 resume 되감기 (수정함)

AMP baseline 은 `task_reward_lerp_start` 와 `task_reward_lerp` 가 **둘 다 0.5** 라 스케줄이
no-op 이었다. 이 arm 은 start 1.0 → end 0.5 로 **이 스케줄을 처음 활성화**한다. AMP 러너의

```
progress = min(1, (it - start_it) / anneal_iters)
```

를 그대로 승계했다면 `--resume` 마다 `lerp` 가 **1.0(스타일 0)** 으로 되감기고 anneal 5000
iter 를 다시 올라간다. 이 프로젝트는 장기 run 이 조용히 중단된 이력이 있어(STATUS.md:
iter 44,041 중단) 실제로 밟게 되는 경로다.

`PPOLatent.update_lerp_schedule` 은 **절대 iter** 로 progress 를 잰다. 스케줄은 학습
진행도의 함수여야 하고 `learn` 호출 경계의 함수가 아니다. 실측 확인:

```
model_19.pt 에서 --resume --max_iterations 5
  iter 19 → 0.9981   20 → 0.9980   21 → 0.9979   22 → 0.9978   23 → 0.9977
```

되감기 없이 이어진다(되감겼다면 iter 19 에서 1.0000 이 찍힌다).

### 7-6. ★ `standing_style_substitute` 를 기본 False 로 내렸다

융합식이 `style_term = w·r_style + (1−w)·standing_pose_reward` 인데,
`standing_pose_reward_w = 0.85` 는 `reward_coef = 2.0` 을 지난 **AMP reward 범위**에 맞춰
튜닝된 값이고 `r_style` 은 [0, ~1.07] 이다. 두 항이 **같은 `(1−lerp)` 예산 안에서 서로를
대체**하므로 스케일이 어긋나면 정지 명령 구간과 그 외 구간의 실효 보상 크기가 달라진다.
§3 이 잡아낸 "`d_step` 기준 상수를 `D_e` 에 옮겨 쓰면 안 된다" 와 **정확히 같은 실패 유형**이고,
V3/V4 는 이 경로를 껐으므로 측정된 바가 없다.

정지 **명령** 구간 처리(정책이 스스로 정지하는 것과는 다른 문제다)는 **별개 arm** 으로
분리한다. 되살릴 때는 `standing_pose_reward_w` 를 `r_style` 범위에 맞춰 다시 잡아야 한다.

### 7-7. 그밖에

- **RSI 로 창을 채우지 않는다**: AMP 경로는 `amp_observation_buffer` 를 참조 모션으로
  채웠지만, 그렇게 하면 리셋 직후 expert 품질 `z` 가 창에 들어가 `r_style` 이 부풀려진다.
- **DR 은 `x_vae` 에 들어가지 않는다**: `joint_pos_noise` / `encoder_bias` 는 policy obs
  경로에만 실린다. 스타일 채점은 GT 상태로 한다.
- **로깅 주의 2 건**: `latent_kl_mean` 은 창이 **찬 env 만** 평균한다(안 찬 env 의 `D_e` 는
  0 이라 섞으면 아래로 편향된다 — 초기 구현이 그랬고 19.31 로 찍혔다). `style_reward_mean`
  / `style_reward_std` 는 롤아웃 **전체 평균**이다(마지막 스텝 하나가 아니다).
- **`D_e` 는 평균과 중앙값이 크게 다르다**: 꼬리가 무거워 `latent_kl_mean`(평균)이 §5 의
  중앙값보다 훨씬 크게 찍힌다. run 판정에는 같은 통계량끼리만 비교할 것.

## 8. 산출물

### 코드 (전부 신규 — baseline `go2_imitation_tracking` 은 한 줄도 건드리지 않았다)

| 경로 | 내용 |
|---|---|
| `source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_latent/__init__.py` | `Go2-Imitation-Latent-v0` 등록 |
| `.../go2_imitation_latent_env.py` | env. `_compute_x_vae` / `_heading_relative_rot6d` / `_push_x_vae` / `_update_style_reward` / `reference_x_vae` |
| `.../go2_imitation_latent_env_cfg.py` | cfg. `LatentStyleCfg` (`encoder_ckpt` · `ref_stats` · `window_n` · `c_kl` · `kl_offset` · `neutral_reward`) |
| `.../agents/rsl_rl_ppo_cfg.py` | `Go2ImitationLatentPPORunnerCfg` — `class_name="OnPolicyRunnerLatent"`, `style` 블록 |
| `rsl_rl/rsl_rl/algorithms/ppo_latent.py` | `PPOLatent(PPOParkour)` — estimator/RMA 승계, discriminator 없음 |
| `rsl_rl/rsl_rl/runners/on_policy_runner_latent.py` | `OnPolicyRunnerLatent` — disc reward 자리에 `extras["style_reward"]` |
| `rsl_rl/rsl_rl/{algorithms,runners}/__init__.py` | export 추가 (기존 항목 불변) |
| `scripts/reinforcement_learning/rsl_rl/train.py` | `class_name == "OnPolicyRunnerLatent"` 분기 1 개 추가 |
| `scripts/imitation_learning/build_go2_latent_ref_stats.py` | 참조 통계 산출 (`mu_ref`/`var_ref`/`c_kl`/`offset`/세션 목록/`D_e` 분위수) |
| `scripts/imitation_learning/verify_go2_latent_env.py` | V1a·V1b·V3·V4·V5·V6 검증 |

수정 금지 대상(`ppo_amp.py`, `on_policy_runner_amp.py`, `motion_lib.py`, 기존
`scripts/imitation_learning/*`)은 전부 그대로다.

`./isaaclab.sh -f` — **내 파일에 대한 지적 0 건**. (브랜치 전체는 기존 파일들 때문에 여전히
FAIL 이며 그것은 이 작업의 것이 아니다.)

### 산출 데이터

| 경로 | 내용 |
|---|---|
| `.../2026-08-27_phase1_motion_vae/gaussfix_reg/runs/base_s0/latent_ref_stats.pt` | 참조 통계 |
| `metrics/latent_env_verification.json` | V1a·V1b·V3·V4·V5·V6 원자료 |

---

## 9. 재현

```bash
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python

# 1) 참조 통계 (train 세션만)
CUDA_VISIBLE_DEVICES=1 $PY scripts/imitation_learning/build_go2_latent_ref_stats.py

# 2) 게이트 V1a / V1b / V3 / V4 / V5 / V6
CUDA_VISIBLE_DEVICES=1 $PY scripts/imitation_learning/verify_go2_latent_env.py \
  --headless --num_envs 96 --steps 200 \
  --out_dir reports/go2_imitation/go2_imitation_tracking/2026-08-28_phase2_latent_env/metrics

# 3) V2 스모크
CUDA_VISIBLE_DEVICES=1 $PY scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-Latent-v0 --num_envs 64 --max_iterations 20 --headless

# 4) resume 이 lerp 스케줄을 되감지 않는지 (§7-5)
CUDA_VISIBLE_DEVICES=1 $PY scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-Latent-v0 --num_envs 64 --max_iterations 5 --headless \
  --resume --load_run <run> --checkpoint model_19.pt
```

본 학습 (팀리드 결정 후):

```bash
setsid env CUDA_VISIBLE_DEVICES=1 $PY scripts/reinforcement_learning/rsl_rl/train.py \
  --task Go2-Imitation-Latent-v0 --num_envs 4096 --headless \
  --logger wandb --wandb-project IsaacLab-locomotion \
  > logs/latent_run.log 2>&1 &
```

resume 은 argparse 플래그만 작동한다 — `--resume --load_run <run> --checkpoint model_X.pt`.

---

## 10. 부수 사항

`./isaaclab.sh -f` 는 `--all-files` 로 돌기 때문에, 이 작업이 건드리지 않은 파일 4 개에
공백·EOF·SPDX 헤더 수정이 들어갔다 — `_workspace/leg/ref_motion_gait.py`(SPDX 5 줄) 와
`reports/leg_imitation/_comparisons/arch_scaling_study/metrics/{priv_vs_history_ramp,
velscale_ramp_40k,velscale_ramp_50k}.md`(각 EOF 개행 1 줄). AGENTS.md 규약대로 돌린 결과이며
되돌리지 않았다. 커밋 시 분리할지는 팀리드가 정한다.

브랜치 전체의 `./isaaclab.sh -f` 는 여전히 FAIL 이다 — 기존 파일들(`ppo_amp.py` 등)의
ANN/D 계열 지적 때문이고, **이 작업의 파일에 대한 지적은 0 건**이다.

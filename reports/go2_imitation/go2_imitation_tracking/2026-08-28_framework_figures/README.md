# [2026-08-28] latent-space imitation 프레임워크 그림 7종

Notion 보고서에 붙일 논문 스타일 그림. 라벨은 전부 영어다 — 이 머신에 CJK 폰트가 없어
matplotlib 한글이 □로 깨진다.

## 그림 목록

| 파일 | 내용 | 출처 |
|---|---|---|
| `fig1_framework.png` | 2단계 프레임워크 (offline VAE 학습 → frozen 산출물 → online PPO) | 설계 도식 |
| `fig2_amp_vs_latent.png` | AMP(공동학습 discriminator) vs latent KL(동결 인코더+고정 참조) 대비 | 설계 도식 |
| `fig3_style_reward.png` | (a) 창 모먼트 vs 고정 참조의 1σ 타원 (b) 보상 곡선과 실측 `D_e` 마커 | `latent_ref_stats.pt`, val latent 실값 |
| `fig4_tsne_wrong_gate.png` | t-SNE가 **틀린 게이트**인 이유 — 망가뜨린 모델이 오히려 더 깨끗하게 갈린다 | 실 체크포인트 2종 인코딩 + sklearn t-SNE |
| `fig5_manifold_gate.png` | M4/M5 다양체 게이트, arm별 시드 산포 | `manifold_gate/*.json` 20개 |
| `fig6_dataset_coverage.png` | `vx × wz` bin별 기여 세션 수 | `phase0/metrics/bin_session_diversity.csv` |
| `fig7_training_progress.png` | `latentKL_stock` 학습 곡선 (단일 run, 리셋 없음) | `logs/train_latent_style_stock.log` |

## 재현

```bash
PY=/home/user/miniconda3/envs/isaac-6.0/bin/python
# fig1, fig2 (도식)
$PY make_framework_figures.py
# fig3~fig7 (실데이터) — fig4 는 아래 인코딩 + t-SNE 선행 필요
$PY encode_latents.py --ckpt <phase1>/vmf/gaussfix_b0_s0/motion_vae.pt --out $S/z_win.npz
$PY encode_latents.py --ckpt <phase1>/gaussfix_reg/runs/ne0.40_s0/motion_vae.pt --out $S/z_bad.npz
$PY encode_latents.py --ckpt <phase1>/gaussfix_reg/runs/base_s0/motion_vae.pt --out $S/z_base_full.npz --max_points 999999
/home/user/miniconda3/envs/stmr/bin/python  # sklearn 이 isaac-6.0 에 없어 t-SNE 만 별도 env
$PY make_data_figures.py
```

## 그림을 만들며 새로 확인한 것

### 1. `base_s0` 와 `gaussfix_b0_s0` 은 같은 가중치다
env 의 `ref_stats` 는 `gaussfix_reg/runs/base_s0` 에서 만들어졌고 manifold 게이트의
승자는 `vmf/gaussfix_b0_s0` 인데, 두 체크포인트로 같은 val 클립을 인코딩하니
latent 최대 차이 **0.0** 이다. 서로 다른 경로에 있는 같은 모델이므로 그림들은
전부 배포에 실제로 들어간 인코더 기준이다.

### 2. ★ 학습 곡선이 보고서 서술보다 10배 큰 이야기를 하고 있었다
로그를 전량 파싱하니(`Learning iteration` 14,500+ 행, iteration 리셋 0회 = 단일 run):

| 시점 | `latent_kl_mean` |
|---|---|
| iter 0 | 72.5 |
| iter 4 | 114.98  ← 보고서가 "초기값"으로 쓴 값 |
| **iter 986 (최대)** | **1,061.6** |
| iter 5,150 | 157.6 |
| 최근 500 iter 중앙값 | **18.16** |

즉 "초기 114.98 → 18.54" 는 **최대 excursion 을 놓친 서술**이다. 실제로는
`task_reward_lerp` 가 1.0 근처(=task 항이 지배)인 구간에서 정책이 expert 다양체에서
**정지(94.7)보다 11배 멀리** 벗어났다가, lerp anneal 이 끝나 style 이 만가중이 되는
iter 5,000 에서 급락한다. 이건 "스타일 항이 실제로 정책을 끌고 있다"의 **더 강한 증거**다
(task 항만으로는 오히려 멀어졌으므로).

⚠️ 다만 여전히 **보상이 살아 있다는 증거이지 걸음걸이가 좋다는 증거가 아니다.**
이동 능력 판정은 40k 이후 속도 램프로만 한다.

### 3. ★ t-SNE 는 승자를 **거꾸로** 고른다
같은 val 표본을 승자(`gaussfix_b0`, held-out 재구성 0.00184)와 의도적으로 망가뜨린
arm(`ne0.40`, 0.01010 = **5.5배 나쁨**)으로 각각 인코딩해 t-SNE 를 돌렸더니,
**망가진 쪽이 gait 별로 더 깨끗하게 갈린다.** 클립 정체성으로 색을 칠하면 군집의 정체가
드러난다 — 군집은 gait 가 아니라 개별 클립이다. 이 세션에서 폐기한 "틀린 게이트 6건"
중 1번의 직접 증거다.

### 4. 그림에 쓰지 않기로 한 것
`d_step` 분포·잡음 민감도·창 확대 AUROC 그림은 이미 각 실험 폴더에 있고 보고서 본문에
임베드돼 있어 중복이라 뺐다.

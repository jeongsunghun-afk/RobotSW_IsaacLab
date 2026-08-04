# crawl을 빼면 3D 지각이 2.5D height_scan보다 낫다

작성 2026-08-03.

## 비교 대상 run

| 이름 | run 경로 | gym task id | 체크포인트 |
|---|---|---|---|
| height_scan (2.5D) | [`logs/rsl_rl/parkour_imitation_go2_symmetry_random_goal/2026-07-29_11-14-23/`](../../../logs/rsl_rl/parkour_imitation_go2_symmetry_random_goal/2026-07-29_11-14-23/) | `Go2-ParkourImitation-Symmetry-RandomGoal-EasyEntry-v0` | `model_49999_FINAL_BEST.pt` |
| 3D clearance | [`logs/rsl_rl/parkour_imitation_go2_teacher3d/2026-07-27_10-34-02_teacher3d_clearance_easyentry_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_teacher3d/2026-07-27_10-34-02_teacher3d_clearance_easyentry_50k/) | `Go2-ParkourImitation-Teacher3D-EasyEntry-v0` | `model_49999.pt` |
| 3D voxel (ray-scatter) | [`logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel/2026-07-27_10-34-04_teacher3d_voxel_binary_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel/2026-07-27_10-34-04_teacher3d_voxel_binary_50k/) | `Go2-ParkourImitation-Teacher3DVoxel-EasyEntry-v0` | `model_49999.pt` |
| 3D voxel GT (기둥 열거) | [`logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-07-31_17-48-16_teacher3d_voxel_gt_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-07-31_17-48-16_teacher3d_voxel_gt_50k/) | `Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0` | `model_49999.pt` |

넷 다 EasyEntry rung, 50k, seed 1.

## 왜 로그의 전체 평균을 그대로 읽으면 안 되나

지형 비율이 다르다. height_scan baseline 은 **5지형 × 0.20, crawl 없음**이고, 3D 셋은
**5지형 × 0.15 + crawl 0.25**다. crawl 은 3D 의 최약 지형인데 baseline 은 시도조차 하지 않으므로,
**baseline 이 치르지 않는 벌점을 3D 만 평균에 진다.**

비-crawl 5지형은 양쪽 다 서로 동일 가중(0.20씩 / 0.15씩)이므로, crawl 만 빼면 단순 평균이 공정 비교다.

## 지형별 (마지막 1000 iter 평균, `curriculum/mean_terrain_level_*`)

| 지형 | height_scan | 3D clearance | 3D voxel(ray) | 3D voxel GT |
|---|---:|---:|---:|---:|
| flat | **5.53** | 4.78 | 5.22 | 5.06 |
| gap | 3.78 | 6.14 | 6.41 | **6.93** |
| hurdle | 6.06 | **6.18** | 6.01 | 6.04 |
| stair | 6.21 | 5.49 | **6.61** | 2.68 |
| step | 5.93 | **7.39** | 6.54 | 5.93 |
| **crawl 제외 평균** | **5.50** | **6.00** | **6.16** | 5.33 |
| *(참고) crawl* | *지형 없음* | *2.72* | *4.94* | *3.20* |
| *(참고) 로그상 전체* | *5.49* | *5.01* | *5.86* | *4.84* |
| Train/mean_reward | **22.91** | 21.37 | 19.35 | 17.35 |

**crawl 을 빼면 voxel(ray) 이 height_scan 보다 +0.66, clearance 가 +0.49 높다.** 로그의 전체 평균만
보면 clearance 5.01 < height_scan 5.49 로 반대로 읽힌다.

가장 큰 격차는 **gap**(3.78 vs 6.41, +2.6 레벨). 구멍의 3D 기하는 하향 격자로 잡기 어렵고, 이것이
비-crawl 평균 차이의 대부분을 만든다. flat 만 height_scan 우위인데, 장애물이 없어 dense 하향
격자의 이점만 남는 지형이다.

## 창 길이 민감도 — 흔들리는 것은 height_scan 뿐이다

지형 레벨은 iteration 마다 요동치므로 "마지막 몇 iter 를 평균했나"로 값이 달라진다. 세 창으로 재봤다.

| | @200 | @1000 | @3000 |
|---|---:|---:|---:|
| height_scan gap | **6.45** | 3.78 | 3.84 |
| height_scan 비-crawl 평균 | 6.00 | 5.50 | 5.57 |
| clearance 비-crawl 평균 | 6.03 | 6.00 | 6.00 |
| **voxel(ray) 비-crawl 평균** | **6.24** | **6.16** | **6.11** |

1000 과 3000 이 서로 일치하고 **200 만 튄다** — 이 정책이 마지막 400 iter 에 gap 을 회복한 구간을
200 창이 통째로 잡은 것이라, 6.45 는 실력이 아니라 운 좋은 스냅샷이다. 그리고 **불안정한 쪽은
height_scan 뿐**이고 3D 둘은 ±0.05 안에 있다. **voxel 은 세 창 전부에서 height_scan 보다 위다.**

## 구 레시피에서는 부호가 반대지만 그 표는 못 쓴다

2026-07-06 ablation(마지막 10000 iter): height_scan 5.78 > clearance 5.34 > voxel 5.25 로 3D 가 진다.
그러나 그 run 들은 flat 표본이 n≈9,900 인 반면 **hurdle 이 n=72~183** 이다 — 로봇 대부분이 평지에
갇힌 6.0 curriculum floor-pinning 상태이고, 장애물 지형 수치는 소수 표본의 잡음이다.
EasyEntry 는 바로 그 floor-pinning 을 풀려고 도입한 rung 이고 지형당 n=194~678 로 고르다.

**즉 "3D 가 2.5D 보다 나쁘다"는 floor-pinning 상태의 관측이고, 커리큘럼이 실제로 올라가는
조건에서는 반대가 된다.**

## voxel GT 열은 버그 있는 관측의 결과다

GT 열(gap 6.93 / stair 2.68)은 **허위 천장 버그가 있던 관측**으로 학습됐다. stair 붕괴의 원인이
그 버그이고(pinned stair 에서 567 기둥 중 236 개가 허위 천장), gap 이 전 arm 최고인 것은 gap 이
마운트보다 높은 기하가 없어 그 버그가 발생하지 않는 유일한 지형이기 때문이다.
상세는 [`../../rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/`](../../rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/).
수정 후 재학습본은 **별도 baseline family** 이므로 이 표에 추가하지 않는다.

## 한계

* **seed 1개씩.** 이 표의 어떤 차이도 seed 분산에 대해 검정되지 않았다.
* **학습 이력이 대등하지 않다.** height_scan run 은 파국을 한 번 겪은 4-run·~52k 계보의 3차 resume
  이고 `model_49999_FINAL_BEST` 는 사람이 고른 체크포인트다. num_envs 도 4096 vs 1024 로 다르다.
* 커리큘럼 레벨은 정책이 스스로 고른 난이도이므로 완주율과 같은 것을 재지 않는다.

## 영상

<!-- report-video:videos -->

| 파일 | 내용 |
|---|---|
| `videos/gap_3up_heightscan_vs_voxel_vs_voxelGT.mp4` | gap 3분할 (1920×404) |
| `videos/step_3up_heightscan_vs_voxel_vs_voxelGT.mp4` | step 3분할 |
| `videos/{gap,step}_{heightscan,clearance,voxel,voxelGT}.mp4` | 방법별 단독, 각자 센서 오버레이 |
| `videos_fixed/{step,gap}_voxelGT_before_vs_after.mp4` | 허위 천장 수정 전/후 |
| `videos_fixed/crawl_{voxelGT_fixed,voxel_ray}.mp4` | crawl |

<!-- /report-video:videos -->

`videos_lifted/` 는 오버레이를 1 m 띄워 깜빡임을 없애려던 시도인데 gap 이 28.3% → 51.3% 로
악화되어(화각 밖으로 나감) 폐기했다. 코드 변경도 되돌렸다.

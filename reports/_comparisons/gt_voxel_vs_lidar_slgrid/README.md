# 배포 가능한 LiDAR 정책 vs 버그를 고친 privileged teacher

작성 2026-08-05.

## 비교 대상

| 이름 | run 경로 | gym task id | actor 입력 | 배포 |
|---|---|---|---|---|
| **GT voxel (법선)** | [`…_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/) | `Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0` | 지형 메시에서 열거한 GT voxel | **불가** |
| **SL-Grid (LiDAR)** | [`…_lidar_sl_grid/2026-07-31_10-28-47_slgrid_scratch_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid/2026-07-31_10-28-47_slgrid_scratch_50k/) | `Go2-ParkourImitation-Lidar-SL-Grid-EasyEntry-v0` | 로봇 자신의 Mid-360 산란 격자 | **가능** |

둘 다 `model_49999.pt`, from-scratch RL 50k, seed 1, **distillation 없음**.
측정: `probe_stair.py --level 6 --num_envs 64 --steps 2000 --sensor none`.

**"어느 센서가 나은가"가 아니라 "배포 가능한 정책이 privileged teacher 에 얼마나 근접하는가"** 를
재는 비교다. GT voxel 은 시뮬레이터 지형 메시를 직접 읽으므로 실기에 옮길 수 없다.

### 왜 이 체크포인트이고 crawl 은 왜 따로 다루나

세션 중간에 crawl 지형을 바꿨다(`num_crawls` 3 → 8, 코스 10.95 → 17.79 m). 그래서

* GT 쪽은 `crawl8_ft1k` fine-tune 이 아니라 **그 이전 체크포인트**를 썼다. fine-tune 은 바뀐
  지형에서 1,000 iter 를 더 돈 것이라 지형 변경이 confound 로 섞인다.
* **crawl 은 아래 별도 절에서 다룬다** — 지형을 학습 당시 상태로 되돌려 재기도 했으나, 애초에
  SL-Grid 의 학습 지형에 crawl 이 없어 비교가 성립하지 않는다.

gap·stair·step·hurdle 은 이번 변경과 무관하므로 두 정책 모두에 공정하다.
SL-Grid 는 `voxel_gt_columns` 게이트가 꺼진 별도 env 이므로 이번 세션의 코드 변경에 영향받지 않는다.

## 결과 — pinned level 6 완주율

| 지형 | **GT voxel** (privileged) | **SL-Grid** (배포 가능) | 차이 |
|---|---:|---:|---:|
| gap | **0.393** | 0.353 | +0.040 |
| stair | **0.390** | 0.296 | **+0.094** |
| step | **0.371** | 0.363 | +0.008 |
| hurdle | **0.338** | 0.299 | +0.039 |
| **평균** | **0.373** | **0.328** | **+0.045** |

부수 지표:

| 지형 | goal_idx (GT / SL) | 전진 [m/s] (GT / SL) | time_out (GT / SL) |
|---|---|---|---|
| gap | 2.900 / 2.966 | 1.106 / 1.006 | 0.600 / 0.624 |
| stair | 1.437 / 1.339 | 1.162 / 0.997 | 0.610 / 0.693 |
| step | 2.846 / 2.808 | 1.024 / 0.989 | 0.592 / 0.629 |
| hurdle | 2.562 / 2.519 | 0.987 / 0.985 | 0.662 / 0.701 |

전 조건에서 실패는 낙상이 아니라 `time_out`(59~70%)이다. tilt ≤ 3.7%, base_contact 0%.

## 읽히는 것

### privileged teacher 가 근소하게 앞선다 — 그러나 대부분 잡음 안이다

네 지형 모두 GT 가 위이고 평균 +4.5 pt. 다만 에피소드 194~275 개에서 p≈0.35 의 이항 표준오차가
약 0.030 이므로 **차이의 표준오차는 약 0.045** 다. 각 지형을 따로 보면:

| 지형 | 차이 | SE 대비 |
|---|---:|---:|
| stair | +0.094 | **2.1 SE** |
| gap | +0.040 | 0.9 SE |
| hurdle | +0.039 | 0.9 SE |
| step | +0.008 | 0.2 SE |

**개별로 유의한 것은 stair 하나뿐**이다. 나머지 셋은 잡음 구간에 있다. 다만 네 지형이 모두 같은
방향을 가리키는 것 자체가 약한 증거이고(부호검정 4/4, 단측 p=0.0625), stair 는 GT 가 전진속도도
1.162 vs 0.997 로 확연히 빠르다.

### ★ "teacher 는 필요 없다"는 결론이 약해졌다

`../../rsl_rl/parkour_imitation_go2_lidar_distill/_comparisons/matched-50k/` 는 pinned gap L6 에서
**SL-Grid 0.353 vs 당시 teacher 0.357** 로 동률임을 근거로 "배포 가능한 센서만으로 privileged
teacher 를 따라잡는다"고 적었다. 그 teacher 는 ray-hit scatter 관측을 쓰던 것이다.

**관측 버그를 고친 teacher 는 같은 지형에서 0.393 이다.** 즉 이전의 동률은 teacher 쪽이
성긴·부분적으로 잘못된 관측으로 성능이 눌려 있었기 때문일 수 있고, 관측이 정확해지자 격차가
다시 열렸다(+4.0 pt, 0.9 SE). 이 차이 자체는 유의하지 않으나 **"동률"이라는 근거는 더 이상
같은 힘을 갖지 못한다.**

SL-Grid 의 gap 완주율은 이번 재측정에서도 **0.353 으로 이전 기록과 정확히 일치**한다(같은
체크포인트·같은 프로토콜) — 재현성 확인.

## crawl — 비교가 성립하지 않는다

두 정책의 **학습 지형 구성이 다르다**:

| 정책 | 지형 비율 |
|---|---|
| SL-Grid | flat·hurdle·step·gap·stair 각 **0.20**, **crawl 없음** |
| GT voxel | 위 5지형 각 0.15 + **crawl 0.25** |

**SL-Grid 는 crawl 을 한 번도 보지 않았다.** 아래 수치는 능력 비교가 아니라 미학습 지형에서의
zero-shot 이다.

| 정책 / 지형 | 완주율 | tilt | time_out | 전진 [m/s] | goal_idx |
|---|---:|---:|---:|---:|---:|
| GT voxel / crawl-3 (학습 지형) | **0.430** | 0.003 | 0.559 | 0.855 | 1.156 |
| GT voxel / crawl-8 (새 지형) | 0.237 | 0.000 | 0.737 | 0.786 | 2.248 |
| SL-Grid / crawl-3 | **0.000** | **0.428** | 0.565 | 0.334 | 0.004 |
| SL-Grid / crawl-8 | **0.000** | **0.443** | 0.556 | 0.278 | 0.018 |

읽을 수 있는 것은 두 가지뿐이다.

* **새 crawl 지형은 실제로 더 어렵다.** 같은 GT 정책이 0.430 → 0.237 로 떨어진다. 터널이 3 → 8 개로
  늘고 회복 간격이 절반이 된 결과이며, 코스를 늘린 것이 아니라 밀도를 올린 설계와 정합한다.
  `goal_idx` 가 1.156 → 2.248 로 오른 것은 서로 다른 goal 이 3 개에서 8 개가 됐기 때문이지 성능이
  아니다.
* **미학습 지형에서 SL-Grid 의 실패 양상은 다른 곳과 다르다.** 다른 네 지형에서는 두 정책 모두
  tilt ≤ 3.7% 에 실패가 전부 `time_out` 이었는데, crawl 에서는 **tilt 43%** 로 넘어진다. 정지가
  아니라 낙상이다. 다만 학습한 적 없는 지형이므로 이것으로 LiDAR 관측의 한계를 논할 수 없다.

crawl 을 공정하게 비교하려면 crawl 이 포함된 지형 구성으로 SL-Grid 를 다시 학습해야 한다.

## 한계

* **비-crawl 네 지형도 노출량이 다르다.** GT 는 crawl 에 0.25 를 쓰므로 각 지형에 0.15,
  SL-Grid 는 0.20 이다. GT 가 **더 적게 본 지형에서 더 잘한** 셈이라 위 +4.5 pt 는 노출량 차이로
  설명되지 않는다 — 다만 이 역시 통제된 비교는 아니다.
* **seed 1개씩, level 6 한 점.** 레벨 스윕을 하지 않았다.
* 두 정책은 관측만 다른 것이 아니다. 서로 다른 task·env cfg 를 쓰므로 이 표는 "이 두 체크포인트"의
  비교이지 관측 하나를 격리하지 않는다.
* crawl 은 쟀으나 학습 지형 구성이 달라 비교로 쓸 수 없다(위 §).
* 네 지형에서 실패가 전부 `time_out` 이므로 완주율은 "제한 시간 안에 끝냈나"를 재며, 통과 능력과 속도가 섞여 있다.

## 영상

<!-- report-video:videos -->

| 파일 | 내용 |
|---|---|
| `videos/stair_gtvoxel_vs_slgrid.mp4` | **stair 좌우 비교** (1280×404) — 유일하게 유의했던 지형 |
| `videos/gap_gtvoxel_vs_slgrid.mp4` | **gap 좌우 비교** |
| `videos/{stair,gap}_gtvoxel.mp4` | GT voxel 단독, 주황 = privileged GT 격자 |
| `videos/{stair,gap}_slgrid.mp4` | SL-Grid 단독, 청록 = 로봇 자신의 Mid-360 산란 격자 |

<!-- /report-video:videos -->

각 정책이 **자기가 실제로 읽는 것**을 오버레이한 상태다. 두 오버레이는 같은 격자 좌표계
(yaw-aligned, 0.1 m, clearance 마운트 원점)를 쓰므로 직접 겹쳐 볼 수 있다.

## 방법론 메모 — hydra override 는 조용히 무시된다

crawl 을 학습 지형(터널 3개)에서 재려고
`env.terrain.terrain_generator.sub_terrains.parkour_crawl.num_crawls=3` 을 넘겼더니 exit 0 에 JSON 까지
정상 생성됐다. 그런데 **존재하지 않는 키를 넘겨도 똑같이 exit 0** 이다 — 이 경로의 hydra 는 struct
모드가 아니라 미지의 키를 버린다. 즉 override 는 처음부터 적용되지 않았고, "에러 없이 돌았으니
반영됐다"고 넘어갔으면 바꾸지도 않은 지형에서 잰 값을 "옛 지형 결과"로 보고할 뻔했다.

그래서 cfg 파일을 실제로 편집하는 방식으로 바꾸고, 전환할 때마다 `PARKOUR_GOALS_REGISTRY` 로
**실제 코스 길이를 재측정해 로그에 남겼다**(3개 → 11.08 m, 8개 → 17.79 m). 측정 후 8개로 복원하고
같은 방법으로 재확인했다.

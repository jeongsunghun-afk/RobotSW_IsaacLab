# GT voxel teacher vs SL-Grid + crawl — 학습 곡선·지형별 주행 비교

작성 2026-08-07.

crawl 지형을 학습 분포에 넣은 첫 배포 가능 정책이 나왔으므로, privileged teacher와
**6지형 전부**를 나란히 놓고 비교한다. 직전 비교
([`gt_voxel_vs_lidar_slgrid/`](../gt_voxel_vs_lidar_slgrid/))는 SL-Grid가 crawl을 학습한 적이 없어
crawl 비교가 성립하지 않았다.

## 비교 대상

| 이름 | run | gym task id | actor 입력 | 배포 |
|---|---|---|---|---|
| **GT voxel** | [`…_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/`](../../../../logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/) | `Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0` | 지형 메시에서 열거한 GT voxel | **불가** |
| **SL-Grid + crawl** | [`…_lidar_sl_grid_crawl/2026-08-05_12-57-11_slgrid_crawl_scratch_50k/`](../../../../logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl/2026-08-05_12-57-11_slgrid_crawl_scratch_50k/) | `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-EasyEntry-v0` | 로봇 자신의 Mid-360 산란 격자 | **가능** |

둘 다 `model_49999.pt`, from-scratch RL 50k, seed 1, distillation 없음.

**이 비교는 정성(주행 영상) + 학습 곡선이다.** 레벨 고정 probe로 재는 정량 완주율 비교는
아직 돌리지 않았다 — 직전 비교의 `slgrid_*_l6.json` 수치는 **구 SL-Grid**
(`2026-07-31_10-28-47`, crawl 미학습)의 것이므로 이 정책의 성적으로 읽으면 안 된다.

### 공정성 — 무엇이 비교 가능하고 무엇이 아닌가

| 항목 | GT voxel | SL-Grid + crawl | 비교 |
|---|---|---|---|
| 지형 비율 | 5지형 각 0.15 + crawl **0.25** | flat 0.20 + 5지형 각 **0.16** | 다름 |
| crawl 코스 | crawl-3 (10.95 m) | crawl-8 (17.79 m) | **다른 지형** |
| 종료 원인·비-AMP reward 항 | 동일 정의 | 동일 정의 | **가능** |
| `mean_reward` | 자체 AMP discriminator | 자체 AMP discriminator | **불가** |

★ 영상 렌더는 `play_per_terrain.py`가 **현재 소스 cfg**로 env를 만든다(각 run의 `params/env.yaml`이
아니다). 소스 cfg는 지금 `num_crawls=8`이므로 **crawl 영상에서 GT voxel은 학습한 적 없는 코스를
달린다(OOD)**. 나머지 다섯 지형은 이번 변경과 무관해 양쪽 모두에 공정하다.

## 지형별 주행 영상

좌 = GT voxel(privileged), 우 = SL-Grid + crawl(배포 가능). 각 오버레이는 **그 정책이 실제로
쓰는 입력**이다 — 좌의 **노란** 면은 지형 메시에서 열거한 GT voxel, 우의 **청록** 셀은 로봇 자신의
Mid-360 산란 격자다.

두 오버레이를 나란히 놓으면 이 비교의 전제가 그대로 보인다. stair에서 GT는 계단면을 빈틈없는
노란 판으로 덮는 반면, SL의 청록 격자는 듬성듬성하고 불규칙하다 — ray가 닿은 셀만 켜지는
산란 격자의 성질이다. 아래 정량 결과는 **그 정보 격차를 안고도 종단 성능이 어디까지 붙는가**를
재는 것이다.

| 지형 | 파일 | 비고 |
|---|---|---|
| flat | `videos/flat_gtvoxel_vs_slgrid_crawl.mp4` | |
| hurdle | `videos/hurdle_gtvoxel_vs_slgrid_crawl.mp4` | |
| step | `videos/step_gtvoxel_vs_slgrid_crawl.mp4` | |
| gap | `videos/gap_gtvoxel_vs_slgrid_crawl.mp4` | |
| stair | `videos/stair_gtvoxel_vs_slgrid_crawl.mp4` | |
| crawl | `videos/crawl_gtvoxel_vs_slgrid_crawl.mp4` | ⚠ GT는 OOD(crawl-3 학습, crawl-8 주행) |

개별 클립(합치기 전)은 `_workspace/gt_vs_slgrid_crawl/videos/`에 남아 있다.

**crawl 클립에서 눈에 띄는 것** — SL은 터널 앞에서 몸을 낮춰 엎드린 자세로 진입하는 반면,
GT는 터널 구간에서 옆으로 넘어진 프레임이 나온다. 커리큘럼 레벨(GT 1.54 / SL 6.01)과 방향이
일치하지만, **이는 GT가 crawl을 못한다는 뜻이 아니라 학습한 적 없는 코스에 놓였다는 뜻**이다.
또한 카메라가 따라가는 것은 12 env 중 한 개체라 이 장면만으로 일반화할 수 없다.

## 학습 곡선

### 1. 물리적 결과 — 정의가 같아 직접 비교 가능

![물리적 결과 지표 비교](figures/fig1_physical_outcomes.png)

45,000~50,000 iteration 평균 (각 n=5,000):

| 지표 | GT voxel | SL-Grid + crawl | 우위 |
|---|---:|---:|---|
| goal 도달 종료 / episode | **1.435** | 1.228 | GT |
| 낙상 종료 (base contact) | **0.0132** | 0.0274 | GT (2.1배) |
| 전복 종료 (tilt) | **0.179** | 0.231 | GT |
| episode 길이 [step] | 720.6 | **790.4** | SL |
| collision 페널티 | −1.646 | **−0.962** | SL |
| feet_stumble 페널티 | **−0.0050** | −0.0090 | GT |

읽히는 것:

- **GT가 goal을 더 많이 딴다** (1.435 vs 1.228). GT는 20k 부근에서 0.95 → 1.4로 계단식으로 뛰고
  그대로 평평해지는 반면, SL은 완만하게 오르며 **50k에서도 아직 상승 중**이다. 추가 iteration의
  여지는 SL 쪽에 있다.
- **SL의 낙상은 중반에 훨씬 많았다가 후반에 크게 좁혔다.** base_contact가 20~25k 구간 0.134로
  정점을 찍어 GT의 최악 구간(5~10k, 0.028)의 약 5배였는데, 40k 이후 급감해 종단 0.027까지 내려왔다.
  **다만 종단에서도 GT(0.013)의 2.1배로 여전히 뒤진다.** 중간 스냅샷만 봤다면 "LiDAR 관측으로는
  낙상을 못 줄인다"고, 후반만 봤다면 "따라잡았다"고 결론냈을 구간이다 — 둘 다 과하다.
- **안정성은 GT, 지속성·충돌은 SL이 낫다.** 낙상·전복·stumble 세 항목이 모두 GT 쪽이고,
  에피소드 길이와 collision은 SL 쪽이다. GT가 goal을 더 따면서도 에피소드가 짧은 것은
  더 빨리 주파하기 때문으로 보이나, 전진 속도를 이 로그에서 분리하지 않았다.
- ★ 지형 비율이 다르다는 점을 여기서 다시 짚어야 한다. GT는 crawl을 0.25로 더 많이 밟는데,
  GT가 학습한 crawl-3은 코스가 짧아 오히려 쉬운 지형이었을 수 있다. 즉 위 표의 GT 우위 중
  일부는 **지형 구성 차이일 가능성**을 배제하지 못한다.
- GT의 collision에 20k·38k 두 번의 급락 스파이크(−5.0, −3.0)가 있다. SL에는 없다. 원인 미조사.

### 2. 지형 커리큘럼 레벨 — 완주율이 아니다

![지형별 커리큘럼 레벨](figures/fig2_terrain_curriculum.png)

★ 이 값은 **승급 문턱을 넘은 결과**이지 완주율이 아니다. 문턱은 코스 길이와 명령 속도에 좌우된다.

45,000~50,000 iteration 평균:

| 지형 | GT voxel | SL-Grid + crawl | 차이 |
|---|---:|---:|---:|
| stair | 6.40 | 6.44 | +0.04 |
| gap | **6.32** | 6.17 | −0.15 |
| hurdle | 5.88 | **6.17** | +0.29 |
| step | 6.48 | **6.96** | +0.48 |
| flat | 4.88 | 4.95 | +0.07 |
| **crawl** | 1.54 | **6.01** | *(비교 불가 — 아래)* |

- **다섯 지형 모두 종단에서 사실상 동률이다** (최대 차 0.48, 대부분 0.3 이내). privileged 관측의
  이점은 종단 높이가 아니라 **속도**로 나타난다 — GT가 초반 5~15k에 먼저 올라가고 SL이 뒤따라 붙는다.
  step만 SL이 끝까지 올라 6.48 → 6.96으로 앞선다.
- **crawl 패널은 능력 비교가 아니라 커리큘럼 수정의 증거다.** GT는 20k 이후 **1.5 부근으로 주저앉아**
  50k까지 그대로인 반면 SL은 6.01에서 안정적이다. 이것이 직전 세션에서 규명한 정체 현상 그 자체다 —
  GT가 학습한 crawl-3 코스(10.95 m)는 다른 지형의 61%라 완주해도 승급선에 못 미치고, 명령이 빠르면
  완주가 곧 강등이 됐다. `num_crawls` 3→8로 코스를 맞춘 SL 쪽에서는 그 함정이 사라졌다.
  **두 곡선의 차이는 정책 성능 차이가 아니라 지형 정의 차이다.**

### 3. 최적화 지표

![최적화 지표](figures/fig3_optimization.png)

- **`mean_reward`는 두 런을 같은 축에 놓을 수 없다** — AMP discriminator가 런마다 다르고 지형 비율도
  다르다. 각 런 내부 추세로만 읽는다(둘 다 단조 상승 후 후반 평탄).
- **★ `action noise std`의 거동이 정반대다.** GT는 0.60 → **0.529**로 평평한데,
  SL-Grid + crawl은 0.60 → **1.837**로 50k 내내 단조 증가한다(+0.035/1,000 iter, 가속 없음).
  같은 알고리즘·같은 하이퍼파라미터(`entropy_coef` 0.01, `desired_kl` 0.01)인데 이 차이가 났다.
  **이 std 상승은 SL-Grid + crawl 런 고유의 현상이다.**
- 다만 **붕괴는 아니다.** value loss 종단이 GT 0.0081 / SL 0.0090으로 사실상 같고(SL이 11% 높다),
  SL의 물리 지표는 같은 기간 계속 개선됐다. 과거 파국 사례는 std 0.93에서 value loss가 36까지 튀었다.
- 원인은 이 두 런만으로 특정할 수 없다 — task·관측·지형 비율이 동시에 다르다.

## 다음 단계 (미실행)

- [ ] **레벨 고정 probe** — 6지형 완주율 정량 비교. crawl 포함 비교가 처음으로 성립한다.
      GT는 crawl-3/crawl-8 양쪽에서 재야 OOD 효과가 분리된다.
- [ ] **std 상승의 원인 분리** — 관측(SL-Grid) 때문인지 crawl 지형 추가 때문인지.
      기존 SL-Grid(crawl 미포함, `2026-07-31_10-28-47`)의 std 곡선을 같은 축에 올리면 한 축이 갈린다.
- [ ] step은 SL이 50k에서도 상승 중 — 추가 iteration의 이득 확인
- [ ] GT collision의 20k·38k 스파이크 원인

## 재현

```bash
# 지형별 렌더 (12편, GPU0, 약 16분)
_workspace/gt_vs_slgrid_crawl/record_pair.sh <OUT_DIR> 0 400
# 나란히 합치기
python3 _workspace/gt_vs_slgrid_crawl/make_sidebyside.py
# 학습 곡선 추출 (tfevents -> CSV) 및 플롯
python3 _workspace/gt_vs_slgrid_crawl/extract_curves.py
python3 _workspace/gt_vs_slgrid_crawl/make_plots.py
```

원자료 CSV는 `_workspace/gt_vs_slgrid_crawl/curves_{gtvoxel,slgrid_crawl}.csv`
(각 50,000 iteration × 18 태그).

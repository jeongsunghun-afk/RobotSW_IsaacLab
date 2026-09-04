# Isaac Sim 렌더 — 지형별로 LiDAR-grid 정책이 실제로 받는 지형 입력

> 작성 일시: 2026-09-04 10:50
> 비교 대상: `2026-08-10_12-35-38_slgrid_crawl_sym_scratch_50k` (baseline) vs
> `2026-09-03_18-17-33_slgrid_crawl_sym_scratch_50k_realsensor_dr22-30` (realsensor) —
> 센서 모델만 다르고 obs 차원·러너 설정은 동일

두 arm이 같은 지형에서 **무엇을 보는가**를 시뮬레이터 안에서 그대로 그린다. 기존
`render_mask_v2_isaac/`가 마스크 하나의 차이를 보여줬다면, 여기서는 정책의 지형 입력 전체
(`obs["lidar"]` 27×21×13 점유 격자)와 그 격자를 만든 LiDAR 리턴을 한 프레임에 같이 그리고,
baseline과 realsensor를 나란히 놓는다.

## 무엇을 그리나

한 스텝마다 프레임 하나를 캡처하고, 오버레이 두 종류를 **env의 상태에서 그대로 읽어** 얹는다.
재계산하는 것은 없다.

| 오버레이 | 내용 | 출처 |
|---|---|---|
| 마젠타 구 | 이 스텝에 정책이 남기는 LiDAR 리턴 | `_mid360._data.ray_hits_w[view_env]`를 env가 캐시한 `_lidar_grid_last_hit_valid`로 마스킹 |
| 주황 큐브 | 켜진 격자 셀 중 지면 시트보다 0.1 m 이상 높은 것 — 정책이 넘어야 할 것 | `_lidar_grid_last[view_env] > 0.5`, 높이 인덱스 ≥ 지면 + 1 |
| 파란 큐브 | 켜진 셀 중 **나머지 전부** | 위와 동일. 지면 시트가 대부분이지만 지면보다 낮은 셀(참호 바닥 등)도 파란색이다 |

셀 중심의 월드 변환은 env의 `_draw_lidar_grid`와 **글자 그대로 같은 식**이다.

```
quat_apply_yaw(root_quat_w, lo + idx * res) + grid_origin
```

## 패널 4개

`composed/<terrain>_side_by_side.png`와 `.gif`는 지형마다 네 칸이다.

1. **baseline 렌더** — Isaac Sim 프레임 + 오버레이
2. **baseline top-down 격자** — 같은 격자를 z축 max로 눌러 위에서 본 것. 색 = 그 칸에서 가장 높이 켜진 z 인덱스, 흰색 = 리턴이 하나도 없는 칸
3. **realsensor 렌더**
4. **realsensor top-down 격자**

축은 네 칸 모두 같다. x 전방 −0.6…2.0 m, y 좌측 −1.0…1.0 m, 로봇 원점은 빨간 사각형.
색 스케일은 **지형별로 두 arm이 공유**한다. 같은 색이 두 패널에서 다른 높이를 뜻하면 비교가
성립하지 않기 때문이다.

`composed/all_terrains_montage.png`는 이 네 칸 행을 지형 5개로 쌓은 것이다.

## 두 arm의 차이가 그림에서 보이는 곳

top-down 패널의 **흰 영역**이 센서가 아무것도 돌려주지 않은 칸이다. realsensor는 blind
0.8 m와 후방 120° 크롭 때문에 로봇 주변의 흰 렌즈가 baseline보다 훨씬 크고, 마젠타 리턴도
rolling 스캔 특유의 성긴 호 모양으로 흩어진다.

수치로 보면 realsensor가 한 스텝에 남기는 리턴은 baseline의 **1/4 수준**이다(지형별 평균
2 994–4 393 대 12 983–19 118). 그런데 격자에 켜지는 셀은 오히려 realsensor 쪽이 더 많다
— step 761 대 532, gap 772 대 522, stair 789 대 533, crawl 1 058 대 766. hurdle 하나만
반대다(463 대 568).

누적이 실제로 셀을 더 채운다는 것 자체는 env의 진단 로그가 따로 뒷받침한다. 다만 **그 로그는
이 그림의 패널을 재는 것이 아니다.** `_log_lidar_grid_diagnostics`는 env 인덱스를 0으로
하드코딩하는데, 여기서 기록한 env는 전부 1이다(열 셀 모두 `counts.json`의 `view_idx: 1`).
그리고 env 0은 이 스크립트가 지형을 격리할 때 남겨 두는 **예비 flat 컬럼**이다. 즉 이 진단은
다섯 지형이 아니라 **같은 평지를 다섯 번** 잰 값이다. 로그의 `teacher_occupied`가 79–85로
거의 변하지 않는 것이 그 방증이다.

샘플도 런당 3개뿐이고(`_LIDAR_GRID_DIAG_STEPS`의 5·50·150), 첫 샘플은 step 5–8 —
`warmup=40`보다 먼저, 즉 첫 기록 프레임보다도 먼저 찍힌다. 그 구간의 gain은
+0.000…+0.037로, 리셋 직후 누적기가 단일 스캔에서 다시 시작하는 과도 구간이다
(`clean_mask`가 20프레임을 버리는 바로 그 구간이다). warmup 이후 정상 구간 샘플 10개만
보면 단일 프레임 recall 0.247–0.306에 누적 gain **+0.101 … +0.518**이다.

그러니 이 값은 "평지에서, env 0에 대해, 누적기가 현재 스캔만으로는 못 채우는 셀을 실제로
채운다"까지만 말한다. 메커니즘을 뒷받침할 뿐 패널의 수치를 재지는 않는다. recall 자체도
teacher 격자에 점유된 셀(79–85개)만 분모로 삼고 같은 로그의 precision은 0.047–0.138이므로,
켜진 셀 대부분은 teacher가 점유로 표시하지 않은 셀이다. "누적기가 채우는 셀 전반"으로
일반화하면 안 된다.

그래도 두 arm의 지형 입력 차이가 "정보가 적다/많다"가 아니라는 것은 표만으로 성립한다.
리턴은 1/4인데 켜진 셀은 다섯 중 넷에서 realsensor가 더 많다. baseline은 조밀하고 그 순간의
관측만으로 채워지고, realsensor는 성기지만 지나간 관측이 남아 채워진다. 성질이 다르다.

## 함정 셋 — 전부 측정으로 잡았고, 각각 증상이 다르다

**1. 지면 기준을 격자 원점에 두면 안 된다. — 증상: 큐브가 전부 한 색.**
격자 원점은 로봇의 clearance 마운트다. 평지 프레임의 `ground_idx`(2–3)와 `grid_lo[2]`
(−0.6), `res`(0.1)로 역산하면 지면은 원점보다 0.30–0.40 m 아래에 있다. 기록된 500프레임
전체로는 `ground_idx`가 2–7까지 퍼지는데, 이는 stair·step에서 로봇이 오르면서 지면 시트가
같이 올라가기 때문이고 코드가 의도한 동작이다. 처음에는 로컬 z ≥ 0을
"높다"로 잡았는데, 스모크 런에서 켜진 셀 510개가 **전부** 원점 아래였다. 지형 자체가 원점
근처까지 올라오지 않는다.

**2. 지면 기준을 칸의 *최하* 셀로 잡아도 안 된다. — 증상: 로봇이 앞으로 숙이면 지면 전체가
주황.** 지면 시트는 스캔이 셀 경계를 걸칠 때 두 셀 두께가 되므로, 최하 셀 기준으로는 시트의
윗층이 스스로 "높은 셀"이 된다. level-3 hurdle에서 프레임당 "높은 셀" 117개가 나왔는데 실제
바는 컬럼 4개였다. 지금은 **칸별 최상 셀의 중앙값**을 지면으로 쓴다. 이 값은 몸통 pitch에
흔들리지 않는다.

**3. gap은 "높은 셀" 규칙이 아예 통하지 않는다. — 증상: 50프레임 중 46개가 미결정.**
gap은 음의 지형이라 올라온 것이 없다. 대안 둘을 붙여 써 봤고 baseline_gap에서 50프레임 중
**46개가 nan**, 창에 드는 것이 0개였다. "지면보다 0.2 m 아래로 꺼진 칸"은 baseline_gap의
점유된 높이 인덱스가 1–3뿐이라 성립할 수가 없고(realsensor_gap은 0–4로 조금 넓다),
"리턴 없는 첫 칸"은 realsensor에서 센서 자신의 blind 렌즈(전방 약 0.8 m)가 매 프레임 먼저
걸린다.

참호가 실제로 하는 일은 **그 컬럼의 리턴 개수를 떨어뜨리는 것**이다. 광선이 참호 위를 넘어
그 너머에 꽂힌다. baseline_gap에서 컬럼별 채워진 셀 수는 가까운 지면에서 21(전 폭)을 유지하다
참호에서 한 자릿수로 꺼지고 다시 회복한다. 그래서 신호를 **국소 최소** — 직전 두 컬럼의 절반
이하로 떨어지는 컬럼, 단 그 두 컬럼이 충분히 차 있을 것 — 으로 정의했다. 이 조건이 거리 감쇠와
blind 렌즈를 둘 다 배제한다. baseline_gap에서는 50프레임 중 16개가 0.5–1.0 m 창에 들어오고,
접근하는 동안 거리가 단조 감소하다 통과 후 리셋되는 패턴이 나온다.

**다만 이 규칙도 realsensor_gap은 못 풀었다.** 그 셀은 50프레임 중 창에 드는 것이 0개, nan이
23개다(baseline_gap은 각각 16개, 11개). blind 0.8 m가 참호 앞 컬럼들을 통째로 비워서
"직전 두 컬럼이 충분히 차 있을 것" 조건을 만족하는 지점이 참호보다 멀리서만 나타난다. 그래서 그 패널만 대체 경로
(`pick_frame`의 fallback: 유한한 거리 중 0.75 m에 가장 가까운 것)로 골랐고, 실제 거리는
1.20 m다. 패널 캡션의 `d_obst` 값이 그것을 그대로 적는다.

## 재현

10개 셀(지형 5 × arm 2)을 GPU 0에서 순차로 돌린다. GPU 0은 남과 공유하는 카드이고 여유
메모리가 분 단위로 몇 GB씩 흔들려서, 실제로 두 번은 `_build_edge_mask`에서
`torch.OutOfMemoryError`로 죽었다. `run_all.sh`가 셀마다 여유 메모리를 기다리고 실패하면
재시도한다.

```bash
source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate isaac-6.0
cd /home/lgb/IsaacLab-6.0

bash reports/presentation/05_parkour_learning/assets/render_terrain_input/run_all.sh \
  "$PWD/reports/presentation/05_parkour_learning/assets/render_terrain_input/raw" 200
```

셀 하나만 다시 돌릴 때는 아래처럼 직접 부른다.

```bash
CUDA_VISIBLE_DEVICES=0 env -u DISPLAY ./isaaclab.sh -p \
  reports/presentation/05_parkour_learning/assets/render_terrain_input/play_terrain_input.py \
  --task Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0 \
  --checkpoint logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl_sym/2026-08-10_12-35-38_slgrid_crawl_sym_scratch_50k/model_20000.pt \
  --arm baseline --terrain hurdle --pin_level 3 \
  --num_envs 12 --warmup 40 --steps 200 --save_every 4 --seed 1 \
  --out_dir <out>/baseline_hurdle --headless
```

합성은 시뮬레이터 없이 돈다(`stmr` 또는 `isaac-6.0`).

```bash
python reports/presentation/05_parkour_learning/assets/render_terrain_input/compose_terrain_input.py \
  --raw_dir <raw> --out_dir <composed>
```

## 체크포인트와 설정

| | baseline | realsensor |
|---|---|---|
| task | `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0` | `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-EasyEntry-v0` |
| 체크포인트 | `logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl_sym/2026-08-10_12-35-38_slgrid_crawl_sym_scratch_50k/model_20000.pt` | `logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl_sym/2026-09-03_18-17-33_slgrid_crawl_sym_scratch_50k_realsensor_dr22-30/model_20000.pt` |
| 센서 | 정적 24k-ray Mid-360 | 비반복 rolling 20k-ray, dropout 38% |
| 몸체 마스크 | v1 사용 | 사용 안 함 |
| blind / 후방 크롭 | 없음 / 없음 | 0.8 m / 120° |
| 마운트 pitch | 30° 고정 | env마다 22–30° 추첨 |
| 누적 | 없음 (단일 프레임) | decayed-max, α 0.94 |

공통: 지형 level 3 고정, num_envs 12, warmup 40, 기록 200 스텝, 4스텝마다 저장(50 프레임),
seed 1.

## 측정값

기록된 프레임에서 직접 집계한 값이다. `composed/counts_table.md`가 원본이고, 이 표는 그것을
그대로 옮긴 것이다.

| terrain | arm | level | rays | frames used / total | mean kept returns | mean lit cells | mean raised cells |
|---|---|---|---|---|---|---|---|
| hurdle | baseline | 3 | 24000 | 50 / 50 | 15155 | 568 | 74 |
| hurdle | realsensor | 3 | 20000 | 50 / 50 | 4251 | 463 | 70 |
| step | baseline | 3 | 24000 | 50 / 50 | 15271 | 532 | 131 |
| step | realsensor | 3 | 20000 | 50 / 50 | 3398 | 761 | 85 |
| gap | baseline | 3 | 24000 | 40 / 50 | 12983 | 522 | 26 |
| gap | realsensor | 3 | 20000 | 50 / 50 | 2994 | 772 | 70 |
| stair | baseline | 3 | 24000 | 50 / 50 | 14769 | 533 | 117 |
| stair | realsensor | 3 | 20000 | 50 / 50 | 3418 | 789 | 185 |
| crawl | baseline | 3 | 24000 | 50 / 50 | 19118 | 766 | 312 |
| crawl | realsensor | 3 | 20000 | 50 / 50 | 4393 | 1058 | 302 |

- **kept returns** — 그 스텝에 `hit_valid`를 통과한 광선 수. baseline 24 000발, realsensor
  20 000발 중.
- **lit cells** — `obs["lidar"]`에서 0.5를 넘는 셀 수. 전체 7 371개 중.
- **raised cells** — 그중 지면 시트보다 0.1 m 이상 높은 셀.
- **frames used** — 리셋 직후 구간을 뺀 프레임 수. gap/baseline만 리셋이 한 번 있어 40개다.

## 파일

- `play_terrain_input.py` — Isaac 런. 원본 PNG, `counts.json`, `grids.npz`만 남긴다.
  `render_mask_v2_isaac/play_mask_compare.py`에서 포크했고 원본과 env 파일은 수정하지 않았다.
- `run_all.sh` — 10개 셀을 GPU 여유 메모리 게이트와 재시도를 걸어 순차 실행한다.
- `compose_terrain_input.py` — 시뮬레이터 없이 스틸·GIF·몽타주·표를 만든다. 캡션이나 대표
  프레임 규칙을 고쳐도 sim을 다시 띄우지 않는다.
- `raw/<arm>_<terrain>/` — 프레임 PNG, `counts.json`, `grids.npz`, `run.log`.
- `composed/` — 스틸, GIF, 몽타주, `summary.json`, `counts_table.md`.

## 확인하지 않은 것 / 주의할 것

- **두 arm은 서로 다른 롤아웃이다.** 같은 seed·같은 지형·같은 level이지만 정책이 다르므로
  궤적이 다르다. 좌우 두 렌더는 "같은 순간"이 아니라 각 arm에서 장애물 0.5–1.0 m 앞이라는
  같은 *조건*을 만족하는 프레임이다. 선택된 스텝 번호는 패널에 찍혀 있다.
- **10개 패널 중 9개만 그 창을 실제로 만족한다.** realsensor_gap 하나는 창에 드는 프레임이
  없어 fallback으로 골랐고 거리가 1.20 m다(위 함정 3 참조). 각 패널의 실제 거리는 캡션의
  `d_obst`에, 전 프레임 거리 배열은 `composed/summary.json`의 `obstacle_distance_m`에 있다.
- **마젠타 점의 개수는 두 arm 사이에서 비교하면 안 된다.** 렌더 슬롯 한도 때문에 광선을
  정적 stride로 솎는데, 그 stride가 광선 수에서 계산되어 baseline 3, realsensor 2로 다르다.
  hurdle에서 실제 남은 리턴은 15 155 대 4 251(3.6:1)이지만 화면에 찍힌 점은 5 047 대
  2 127(2.4:1)이다. 개수는 표를 보고, 그림에서는 **분포 모양**만 읽어야 한다.
- **realsensor의 마운트 pitch는 22–30° 중 뽑힌 값 하나다.** 셀마다 `counts.json`의
  `mount_pitch_deg`에 기록돼 있다. 이 그림은 그 구간의 대표가 아니라 표본 하나다.
- **마젠타 점의 위치에는 거리 노이즈가 반영돼 있지 않다.** 센서는 노이즈와 dropout을
  `distances`에만 적용하고 `ray_hits_w`에는 적용하지 않는다. 어떤 광선이 살아남는지는
  노이즈를 반영한 판정이지만, 살아남은 점이 찍히는 위치는 노이즈 없는 기하학적 거리다.
- **level 3 고정은 커리큘럼을 무력화해서 만든 상태다.** `max_init_terrain_level`만으로는
  초기 추첨만 제한되고 첫 성공 에피소드에서 승급한다(스모크 런에서 level 4가 나왔다).
  그래서 `_update_terrain_curriculum`을 인스턴스 단위로 고정 버전으로 갈아끼웠다. env 파일은
  건드리지 않았지만, 정책이 학습된 조건 그대로는 아니다.
- **crawl은 터널 안에서 카메라가 벽에 묻힌다.** baseline 50프레임 중 8개, realsensor 1개에
  오버레이가 전혀 보이지 않는다. 대표 스틸은 그런 프레임을 고르지 않고, GIF는 baseline 기준
  42프레임에서 끊는다. 포크 원본 README에도 같은 현상이 적혀 있다.
- **카메라는 한 시점뿐이다.** 좌후방 위 3/4 뷰 하나로 끝냈다. top-down 시점은 별도 렌더 대신
  격자 raster 패널로 대체했다.
- **"어느 arm이 더 낫다"는 판단은 여기서 하지 않는다.** 이 산출물은 입력이 어떻게 다른지만
  보여준다. 성공률 비교는 다른 곳에 있다.

# 법선 기반 GT voxel — pinned 완주율로 다시 재기

작성 2026-08-04.

## 대상

| 이름 | run 경로 | gym task id | 체크포인트 |
|---|---|---|---|
| **GT fixed (법선)** | [`logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/`](../../../../logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/2026-08-03_18-06-39_voxel_gt_normals_50k/) | `Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0` | `model_49999.pt` |
| crawl 기준선 | [`…_teacher3d_voxel/2026-07-27_10-34-04_teacher3d_voxel_binary_50k/`](../../../../logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel/2026-07-27_10-34-04_teacher3d_voxel_binary_50k/) | `Go2-ParkourImitation-Teacher3DVoxel-EasyEntry-v0` | `model_49999.pt` |
| stair 기준선 | [`…_teacher3d/2026-07-27_10-34-02_teacher3d_clearance_easyentry_50k/`](../../../../logs/rsl_rl/parkour_imitation_go2_teacher3d/2026-07-27_10-34-02_teacher3d_clearance_easyentry_50k/) 외 | `Go2-ParkourImitation-Teacher3D-EasyEntry-v0` | `model_49999.pt` |

측정: `probe_stair.py --terrain {crawl,stair} --level {3,5,7,9} --num_envs 64 --steps 2000 --sensor none`.
기준선 수치는 [`../crawl_difficulty_probe/`](../crawl_difficulty_probe/) 와
[`../stair_level_probe_clearance_vs_voxel/`](../stair_level_probe_clearance_vs_voxel/) 에서 그대로 가져왔고,
셋 다 **동일 프로토콜**이다. 원자료 `metrics/`.

**버그판 GT 는 재지 않았다.** 그 체크포인트의 관측 코드가 이번에 바뀌었으므로 지금 로드하면 학습
때와 다른 입력을 받는다 — 나오는 값은 분포 밖 동작이지 "버그판이 달성했던 성능"이 아니다.
버그판의 성적은 커리큘럼 레벨(stair 2.68 / crawl 3.52)로만 남는다.

## crawl 완주율

| level | 천장 [m] | ray-scatter voxel | **GT fixed** |
|---:|---:|---:|---:|
| 3 | 0.430 | 0.405 | 0.382 |
| 5 | 0.390 | 0.413 | **0.420** |
| 7 | 0.350 | 0.190 | **0.442** |
| 9 | 0.310 | 0.267 | **0.405** |

## stair 완주율

| level | clearance | voxel(ray) | **GT fixed** |
|---:|---:|---:|---:|
| 3 | 0.349 | 0.260 | **0.355** |
| 5 | 0.333 | 0.311 | **0.381** |
| 7 | 0.198 | 0.338 | **0.384** |
| 9 | **0.000** | 0.288 | **0.358** |

## 읽히는 것

### 능력 절벽이 사라졌다

기준선 정책들은 난이도가 오르면 무너진다 — crawl 은 L5→L7 에서 0.413→0.190, stair 는 clearance 가
L7→L9 에서 0.198→**0.000**. **GT fixed 는 두 지형 모두 L3~L9 에서 평평하다** (crawl 0.382~0.442,
stair 0.355~0.384). 레벨이 올라도 성적이 떨어지지 않는다.

`../crawl_difficulty_probe/` 가 "능력 절벽은 L5→L7" 이라고 적은 것은 **그 정책의 한계였지 지형의
한계가 아니었다.** 같은 지형·같은 레벨에서 다른 정책이 절벽 없이 통과한다.

### ★ 커리큘럼 레벨과 완주율이 crawl 에서 정반대를 가리킨다

| | 버그판 GT | GT fixed |
|---|---:|---:|
| crawl 커리큘럼 레벨 | 3.52 | **1.32** |
| crawl 완주율 (L9 고정) | 미측정 | **0.405** |

커리큘럼만 보면 crawl 이 절반 이하로 나빠진 것처럼 읽히는데, **완주율로 재면 L9 에서도 40.5%로
ray-scatter(0.267) 보다 높다.** 커리큘럼 레벨은 정책이 스스로 머무는 난이도이지 통과 능력이 아니다.

앞서 이 하락을 "버그판은 복도가 막혀 회피했고 수정판은 시도하다 걸린다"로 해석했는데 **그 가설은
지지되지 않는다** — 수정판은 L5 이상 모든 레벨에서 기준선보다 잘 통과한다.

### ★★ 원인: crawl 코스가 다른 지형의 61% 길이인데 승급 문턱은 속도로만 정해진다

`parkour_env.py:1848-1855` 의 커리큘럼은 지형을 보지 않는다.

```
dis_to_origin = ‖root_xy − spawn_xy‖              스폰에서의 직선 변위
expected_dist = |commanded_vx| × episode_length_s  = cmd × 20
move_up   = dis > 0.8 × expected  = 16 × cmd
move_down = dis < 0.4 × expected  =  8 × cmd
```

그런데 goal 코스 길이는 지형마다 다르다. `PARKOUR_GOALS_REGISTRY` 에서 실측(난이도 0.6, 20회):

| 지형 | 마지막 goal x [m] | 서로 다른 goal x |
|---|---:|---:|
| **crawl** | **10.95** (10.30~12.03) | **3** |
| hurdle | 17.78 | 8 |
| step | 17.44 | 8 |
| stair | 18.00 | 4 |
| gap | 17.86 | 8 |

**crawl 만 ~10.95 m 로 끝나고 나머지는 ~17.4~18.0 m 다.** 원인은 `parkour_crawl_terrain` 이
`num_crawls=3` 로 goal 을 **3개만** 만들고, `num_goals=8` 을 맞추려 마지막 goal 을 5번 복제하기
때문이다(`parkour_terrains.py`: `_raw = vstack([_raw, tile(_raw[-1:], num_goals - len(_raw))])`).
다른 지형은 장애물이 8개다.

명령 `lin_vel_x ∈ [0.3, 1.5]` 균등을 넣으면:

| | crawl (코스 10.95 m) | 그 외 (코스 ~17.8 m) |
|---|---:|---:|
| 승급 가능 조건 | cmd < **0.68** m/s | cmd < **1.11** m/s |
| 명령 분포에서 그 비율 | **32.0%** | **67.7%** |
| **완주해도 강등되는 조건** | cmd > **1.37** m/s | cmd > 2.22 (도달 불가) |
| 그 비율 | **10.9%** | **0%** |

즉 crawl 은 ① 승급이 가능한 명령 구간이 다른 지형의 절반이고, ② **코스를 완벽히 완주해도
명령의 11% 에서 강등된다** — 다른 지형은 절대 겪지 않는 구조적 불이익이다. 정책이 아무리
잘 기어도 레벨은 낮은 곳으로 끌려간다.

**따라서 crawl 커리큘럼 레벨은 이 지형에서 능력 지표로 쓸 수 없다.** 버그판 3.52 와 수정판 1.32 의
차이도 능력차로 읽으면 안 된다. crawl 능력은 pinned 완주율로만 판단해야 하고, 그 기준으로는
수정판이 L5 이상 전 레벨에서 기준선보다 낫다.

### 실패 양상은 여전히 정지다

전 조건에서 tilt ≤ 2.0%, low_height 0%, base_contact ≤ 2.4%. 실패의 전량이 `time_out`
(crawl 54.6~61.5%, stair 61.6~64.5%) 이다. 로봇은 천장이나 계단에 부딪혀 넘어지지 않고, 제한 시간
안에 통과하지 못한다. `../crawl_difficulty_probe/` 에서 관측된 서명과 같다.

stair 전진속도가 1.047~1.179 m/s 로 crawl(0.741~0.887) 보다 확연히 빠른데, 이는 crawl 이 웅크린
자세로 이동해야 하는 지형이라는 점과 정합한다.

## 한계

* **seed 1개씩, 정책 1개씩.** 어떤 차이도 seed 분산에 대해 검정되지 않았다.
* 기준선 두 개는 **다른 관측을 쓰는 다른 task** 다. 관측·학습이 함께 다르므로 이 표는 "어느
  체크포인트가 이 지형을 잘 넘나"를 재는 것이지 관측 하나를 격리하지 않는다. 관측만 격리한 비교는
  같은 task 의 버그판↔수정판이고, 그것은 커리큘럼 레벨로만 가능하다(위 §).
* 레벨 4·6·8·10 은 재지 않았다.
* 위 승급/강등 비율은 **명령이 균등분포이고 로봇이 코스 끝까지 간다**고 놓은 계산이다. 실제로는
  명령이 4초마다 재표집되고(`resampling_time_s=4.0`) 커리큘럼은 리셋 시점의 마지막 명령 하나만
  보므로, 분포는 맞아도 개별 에피소드의 판정은 달라질 수 있다. `curriculum_diag/` 계측이 gap
  에만 있어 crawl 의 실제 frac_up/frac_down 은 측정하지 않았다 — 계산은 검증되지 않은 예측이다.

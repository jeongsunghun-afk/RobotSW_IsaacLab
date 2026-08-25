# Ground-truth voxel teacher — ray 표집을 열거로 대체

작성 2026-07-31. 상위 맥락은
`../_comparisons/program-summary-0727-0731/`.

## 왜

기존 voxel 격자는 clearance 광선의 **hit point가 떨어진 칸 하나**만 1로 찍는다
(`voxel_occupancy.py`: `grid_flat[e, occ_flat] = 1`). 294개 발산 광선으로 7,371칸 중
**70~80칸**만 점유가 되고, 세 가지 결함이 따라온다.

| 결함 | 내용 |
|---|---|
| **거짓 음성** | 광선이 안 맞은 단단한 지형이 `0` — 이진화 후 허공과 구별 불가 |
| **표면만, 부피 아님** | 지면 *아래*가 안 채워져, 바닥의 구멍과 단단한 바닥이 얇은 껍질 차이뿐 |
| **자세 의존 flicker** | 로봇이 걸으면 hit point가 이웃 칸으로 옮겨가 정지 지형에서도 1-집합이 진동 |

셋째는 2026-07-27에 `unknown(-1)` 채널을 없애며(`binary_occupancy`) 한 번 다뤘던 문제가
**occupied 채널로 옮겨간 것**이다. 이진화는 표집 자체를 고치지 않았다.

## 무엇을 바꿨나

**표집을 열거로 바꿨다.** 격자 기둥마다 광선 하나씩 — 아래로 하나, 위로 하나
(27 × 21 × 2 = **1,134개**). 어떤 기둥도 누락될 수 없고, 채우기가 hit point 산란이 아니라
**기둥별 표면 높이와의 비교**가 된다.

```
occupied(ix,iy,iz)  ⇔  z_iz ≤ ground_z(ix,iy)   또는   z_iz ≥ ceil_z(ix,iy)
```

두 광선 모두 **원점에서 출발**한다. 로봇 base는 그 기둥에서 자유 공간임이 보장된 유일한
점이기 때문이다. 볼륨 위에서 아래로 쏘면 crawl 터널의 **천장을 지면으로 오독**한다.

* 패턴: `voxel_column_pattern.py` (신규)
* 채우기: `voxel_occupancy.fill_voxel_grid_gt()` (신규)
* 게이트: `ParkourEnvCfg.voxel_gt_columns` (기본 False)
* task: `Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0`

## 가정과 그 근거

기둥당 solid 구간을 **최대 2개**(지면 이하, 천장 이상)로 모델링한다. 공중에 뜬 판(위아래 모두
자유 공간)은 표현할 수 없다.

`parkour_terrains.py`의 모든 생성기가 이 모델 안에 들어간다 — hurdle·stepping stones·
balance beam·slope·zigzag·rough blocks·stair는 전부 지면에서 위로 solid이고, **crawl조차**
천장 슬래브에 top-blocker가 붙어 (`ceiling_z_low + thickness + extra ≥ 1.0 m`) 격자 상단까지
solid로 이어진다. 격자 z 범위가 마운트 기준 ±0.6 m이므로 그 안에서 천장 위는 계속 solid다.

## 검증

**단위 검증 12/12** (`fill_voxel_grid_gt`·`voxel_column_pattern` 실물 호출):
형상 27×21×13, 평지 부피 채움(껍질 아님), gap 기둥 완전 공백, 천장 층수, 기둥 인덱스
`c = ix*ny + iy` ↔ `(ix,iy)` 일치, 패턴 1,134개(전반 down/후반 up), 기둥 좌표가 셀 중심.

**실행 검증** (무음 no-op이면 기존 fill과 로그가 구별되지 않으므로 양성 수치로 확인):

```
[VoxelGT step=5]  occ=2268/7371  columns_without_ground=0/567  columns_with_ceiling=0
[VoxelGT step=50] occ=2268/7371  columns_without_ground=0/567  columns_with_ceiling=0
```

2268 = 567 기둥 × 4층. 기존 산란 방식의 **72~80칸 대비 28~31배**이고, 누락 기둥 0이다.
env 0은 예약 평지라 천장 0이 맞다.

## 학습

`logs/rsl_rl/parkour_imitation_go2_teacher3d_voxel_gt/…_teacher3d_voxel_gt_50k`.
**1024 env, 50k, seed 1 — 원본 voxel teacher와 동일 레시피**이고 `voxel_gt_columns` 하나만
다르다.

## ★ 이 체크포인트는 기존 teacher와 호환되지 않는다

actor의 지형 입력 의미가 바뀌고 점유 셀이 약 28배 늘어난다. 따라서
`parkour_imitation_go2_teacher3d_voxel`(0.357)에 맞춰 측정한 이번 주 distillation arm
전부(K=3/5/10/15, A1-0, A1-1, Ceiling)는 **별도 baseline family**이며 같은 표의 행으로
비교할 수 없다.

## 기대 — 측정 전에 적어둔다

**gap 지형이 가장 크게 움직일 것으로 본다.** 바닥의 구멍이 기존에는 "표면 셀이 몇 개 없는 것"
이었는데 이제 **기둥 전체가 비어 있는 것**이 된다. 관측 가능한 신호의 크기가 가장 크게 달라지는
지형이다.

단 이런 예측은 이미 한 번 빗나갔다 — A1-0에서 step을 예상했다가 gap이었고, 원인은 결손의
존재와 성능 병목을 혼동한 것이었다. 이번 근거는 결손이 아니라 **신호 대비 변화량**이라는 점에서
다르지만, 틀릴 수 있는 형태로 남겨둔다.

## 결과 (2026-08-02, 50k 완주)

### crawl 천장 — 작동한다. 앞선 우려는 표본 탓이었다

혼합 지형에서 `envs_with_any_ceiling`이 0~9/256으로 나와 천장을 놓치는 것 아닌지 의심했다.
**crawl 지형·레벨 8로 고정해 재니 반증된다:**

| step | 천장을 본 env | 최대 천장 기둥 수 | occ max |
|---|---|---|---|
| 5 | 0/64 | 0 | 1701 |
| 50 | 31/64 | 210 | 2988 |
| 300 | 58/64 | 338 | 3612 |
| 2000 | **63/64** | **335 / 567** | 3420 |

혼합 지형의 낮은 수치는 crawl 비중이 0.25이고 초반에는 전 로봇이 평지 스폰 패드 위에 있기
때문이었다. **천장 검출에 결함 없음.**

### ★ pinned gap L6 — GT teacher가 기존 teacher를 넘지 못했다

| arm | 관측 | 완주율 | tilt | goal_idx |
|---|---|---|---|---|
| 기존 voxel teacher | privileged, ray 산란 | **0.357** | 0.000 | 2.824 |
| **GT voxel teacher** | privileged, **기둥 열거** | **0.326** | 0.021 | 2.800 |
| **SL-Grid** (teacher 없음) | **LiDAR 유래 격자** | **0.353** | 0.024 | **2.966** |

**예측이 빗나갔다.** "gap이 가장 크게 움직일 것"으로 적어뒀는데 GT teacher는 gap에서 오히려
3.1 pt **낮다**. 관측을 더 정확하게 만든 것이 teacher 성능으로 번역되지 않았다.

이는 이 문서 앞부분의 미확정 항목 — *"기존 teacher는 성긴 입력으로 0.357을 낸다, 즉 이
난이도에서는 충분했다"* — 을 지지한다. **표집 성김은 teacher의 병목이 아니었다.**

### ★★ 그리고 teacher 자체가 필요 없었다

**SL-Grid는 teacher도 distillation도 없이 from-scratch RL만으로 0.353을 낸다** — privileged
clearance를 보는 기존 teacher(0.357)와 사실상 동률이고, goal_idx는 2.966으로 전 arm 최고다.
그 actor가 읽는 것은 **로봇 자신의 Mid-360에서 산란한 격자**이므로 배포 가능하다
(critic만 privileged clearance를 쓰는 비대칭 구조이고 critic은 배포에서 버려진다).

같은 50k에서:

* privileged teacher 0.357
* **배포 가능 센서 + teacher 없음 0.353**

## 미확정

* teacher 성능이 실제로 오르는지. 기존 teacher는 성긴 입력으로 pinned gap L6 **0.357**을 낸다 —
  이 과제 난이도에서는 충분했다는 뜻이다. "성기니까 못한다"는 아직 검증되지 않았다.
* distillation이 쉬워지는지. Ceiling arm은 student가 *어느 294개 광선이 어디 떨어졌는지*를
  맞혀야 해서 8.8 pt를 잃는다고 지목했다. GT 격자는 **조밀한 LiDAR로 근사 가능한 진짜 기하**이므로
  그 격차가 줄어야 한다는 것이 이 rung의 핵심 가설이고, 새 teacher로 distill을 다시 돌려야 갈린다.

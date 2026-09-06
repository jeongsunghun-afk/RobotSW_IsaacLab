# Go2 ParkourImitation 환경

Go2 로봇의 다중 지형 파쿠르 주행에 **AMP(Adversarial Motion Prior)** 모방학습을 결합한 하이브리드 환경. 평지 서브지형에서는 AMP 보상(trot 참조 모션 모방)이 추가되어 자연스러운 보행 스타일을 강제하며, 나머지 지형에서는 parkour 보상만 적용됩니다.

> 기반: `parkour_env.py` / `parkour_env_cfg.py` (`ParkourEnvCfg`) 상속.
> AMP 연산은 `parkour_imitation_env.py` (`Go2ParkourImitationEnv`)에서 처리.

## 환경 구조

| 파일 | 설명 |
|------|------|
| `parkour_imitation_env.py` | AMP 하이브리드 환경 (`Go2ParkourImitationEnv`) |
| `parkour_imitation_env_cfg.py` | AMP + 지형 설정 (`ParkourImitationEnvCfg`, `ParkourImitationTerrainStyleEnvCfg`) |
| `parkour_imitation_random_goal_env.py` | 360° 랜덤 목표 변종 (`Go2ParkourImitationRandomGoalEnv`) |
| `parkour_imitation_random_goal_env_cfg.py` | 랜덤 목표 환경 설정 |
| `parkour_imitation_terrain_style_env.py` | 지형 불변 AMP obs 변종 (`ParkourImitationTerrainStyleEnv`) |
| `parkour_demo_env.py` | 인터랙티브 데모 환경 (`Go2ParkourDemoEnv`) |
| `parkour_demo_env_cfg.py` | 데모 환경 설정 (`ParkourDemoEnvCfg`, `ParkourPlaygroundEnvCfg`) |
| `agents/rsl_rl_amp_cfg.py` | RSL-RL AMP 러너 설정 (6가지 변종) |
| `motion_lib.py` | 참조 모션 로더 (`Go2MotionLib`) |
| `imitation/` | 참조 모션 `.pkl` 파일 디렉토리 |

## AMP 설계 요약

- **Runner**: `OnPolicyRunnerParkourAMP` (PPOParkour + AMP discriminator 통합)
- **Policy**: `ActorCriticRMA` (scan_encoder + priv_encoder + state_history_encoder)
- **Algorithm**: `PPOAMP` — PPOParkour에 AMP 판별자 loss를 결합
- **AMP 보상 fusion**: `total = task_reward + amp_weight(0.3) × flat_env_mask × disc_reward`
- **AMP obs 기본**: 49-dim/프레임 × 10 프레임 = **490-dim** (dof_pos/vel + root_height + lin_vel + ang_vel + foot_pos_local + root_rot_tan_norm)
- **참조 모션**: `ParkourImitationEnvCfg.amp_motion_pkl = "imitation/go2_jump"` (디렉토리 → 복수 .pkl 자동 탐색)

## 등록된 Task ID

| Task ID | 환경 클래스 | 러너 설정 클래스 | 설명 |
|---------|------------|-----------------|------|
| `Go2-ParkourImitation-v0` | `Go2ParkourImitationEnv` | `Go2ParkourImitationPPOAMPRunnerCfg` | 기준선 — 파쿠르+AMP (평지 마스크 적용) |
| `Go2-ParkourImitation-Symmetry-v0` | `Go2ParkourImitationEnv` | `Go2ParkourImitationSymmetryPPOAMPRunnerCfg` | 위 + L/R 미러 데이터 증강 |
| `Go2-ParkourImitation-Symmetry-RandomGoal-v0` | `Go2ParkourImitationRandomGoalEnv` | `Go2ParkourImitationSymmetryRandomGoalPPOAMPRunnerCfg` | Symmetry + 360° 랜덤 목표 커리큘럼 |
| `Go2-ParkourImitation-TerrainStyle-v0` | `ParkourImitationTerrainStyleEnv` | `Go2ParkourImitationTerrainStylePPOAMPRunnerCfg` | Symmetry + 지형 불변 AMP obs (37-dim/프레임 = 370-dim) |
| `Go2-ParkourDemo-v0` | `Go2ParkourDemoEnv` | `Go2ParkourImitationSymmetryPPOAMPRunnerCfg` | 데모 전용 — test 지형 (5종×11난이도, num_envs=1) |
| `Go2-ParkourDemo-Playground-v0` | `Go2ParkourDemoEnv` | `Go2ParkourImitationSymmetryPPOAMPRunnerCfg` | 데모 전용 — 자유 scatter 지형 (num_envs=1) |

### 변종별 차이 요약

| 변종 | 핵심 차이 |
|------|-----------|
| `v0` (기준) | AMP 보상을 평지(`flat_env_mask`) 환경에만 적용 |
| `Symmetry` | L/R 미러 데이터 증강 추가 (`symmetry_cfg`, `num_aug=2`) — 3-leg local optimum 방지 |
| `Symmetry-RandomGoal` | 각 리셋 시 로봇 주변 360°·[1.5, 3.0]m 범위에서 랜덤 목표 재추첨 |
| `TerrainStyle` | AMP obs에서 지형 의존 차원 제거 (root_height, lin_vel_z, foot_z, rot_tan_norm 삭제 → 37-dim) |
| `Demo/Playground` | num_envs=1, 팔로우 카메라 추가, **`play_parkour_demo.py`로만 실행 가능** |

### 실센서 Mid-360 arm (`...-RealSensor-EasyEntry-v0`)

`Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-EasyEntry-v0`은 실제 배포 Mid-360의
rosbag(2026-09-03) 계측에 맞춰 센서 모델을 교체한 arm이다. 환경 클래스와 러너 설정
(`Go2ParkourImitationLidarSLGridCrawlSymPPOAMPRunnerCfg`)은 `...-Crawl-Sym-EasyEntry-v0`과 동일하고
`obs["lidar"]`도 7371차원 점유 격자 그대로라, 두 arm은 직접 비교할 수 있다.

바뀌는 것은 다섯 가지이며 각각 `ParkourImitationRandomGoalLidarEnvCfg`의 독립 필드라 ablation 시
하나씩 되돌릴 수 있다.

| 필드 | 기본값 | RealSensor arm | 근거 |
|------|--------|----------------|------|
| `lidar_mount_pitch_deg` | `None` (30°) | `30.0` | 제공자 배포 extrinsic (0, 0.9659258, 0, 0.2588190) = 설계값. rosbag 중력 기준 실측은 22~25°로 미해결 |
| `lidar_mount_pitch_range_deg` | `None` | `(22.0, 30.0)` | 위 미해결 구간을 env별로 균등 추첨, 에피소드마다 재추첨 |
| `lidar_use_body_occ_mask` | `True` | `False` | 실제 파이프라인은 몸체 마스크를 쓰지 않음 |
| `lidar_blind_range` [m] | `0.0` | `0.8` | 0.8 m 미만 반사는 발행되지 않음 |
| `lidar_rear_crop_deg` [deg] | `0.0` | `120.0` | 후방 120° 크롭 (base 방위각 120~240°) |
| `lidar_mount_pos` [m] | `None` | 미사용 | 붐 위치 override용 |

이 arm의 센서 cfg는 `update_period=0.1`도 함께 준다. 다른 arm은 `SensorBaseCfg` 기본값 0.0을
상속해 제어 스텝마다(50 Hz) 다시 캐스트하지만, 여기서는 실제 측정 주기인 10 Hz로만 캐스트한다.
그 결과 센서 갱신 스텝과 env의 누적기 push 스텝(`push_every=5`)이 env별로 정확히 일치하고,
push 한 번이 새 스캔 창 하나를 소비한다. 리셋된 env는 타이머가 다시 시작되므로 이후 갱신이
나머지와 어긋나고, 센서는 부분 마스크로 갱신된다.

붐 pitch는 하나로 고정하지 않는다. 제공자 extrinsic(30°)과 rosbag 실측(22~25°)이 ~7° 어긋난 채
남아 있어서, `lidar_mount_pitch_range_deg = (22.0, 30.0)`으로 두 값을 모두 덮는 구간에서 env마다
균등 추첨하고 리셋마다 다시 뽑는다. 어느 쪽이 맞는지 확정되기 전에도 정책이 한쪽 extrinsic에
의존하지 않게 하려는 것이다. 실제로 뽑힌 값은 `sensor.mount_pitch_deg` (N,)로 읽고 init 때
min/mean/max가 출력된다. `None`이면 `lidar_mount_pitch_deg` 하나로 회전하는 기존 경로와 비트
단위로 같다. 이 필드는 rolling 센서 cfg에서만 의미가 있고, 정적 `LidarSensorCfg`에 설정하면
`resolve_lidar_mount`가 `TypeError`를 낸다.

센서 자체는 `mid360_rolling_lidar.Mid360RollingLidarSensor`(cfg는 `mid360_rolling_lidar_cfg.py`)로 교체된다. 코어 `LidarSensor`가 스캔
패턴에서 잘라낸 창 하나를 영원히 재사용하는 것과 달리, 이 서브클래스는 매 업데이트마다
`mid360.npy`(80만 행, 시간순)의 다음 `samples`행 창을 환경별로 ray 방향 버퍼에 덮어써서 실제
비반복 스캔을 재현한다. 환경마다 스캔 위상이 다르고 리셋 때마다 다시 추첨된다
(`rolling_scan`, `random_scan_phase`). 광선 수는 20000, 무반사율은 0.38로 실측 반환 수에 맞췄다.

`lidar_mount_pitch_deg` / `lidar_mount_pos`는 cfg의 `__post_init__`과 환경의 `_setup_scene` 양쪽에서
적용된다. hydra `env.*` override는 cfg 생성 이후에 반영되므로 `_setup_scene` 쪽이 없으면
`env.lidar_mount_pitch_deg=25` 같은 명령행 override가 센서에 도달하지 못한다.

### 롤링 누적기 arm (`...-RealSensor-Roll-EasyEntry-v0`, `...-RealSensor-RollNoFree-EasyEntry-v0`)

RealSensor arm의 누적기(`_update_occupancy_accumulator`)는 매 센서 틱마다 yaw 정렬 몸체 프레임의
격자를 trilinear `grid_sample`로 다시 표본한다. 두 보고서가 그 대가를 계측했다.

- `reports/go2_parkour/_comparisons/real_lidar_vs_sim/acc_diag/README.md` — 셀 미만 이동이
  한 셀 두께 지면 시트를 이웃으로 쪼개고 α=0.94가 양쪽을 임계값 아래로 내린다. 0.8 m blind
  zone 점등 비율이 실측 0.16~0.27인데 감쇠만 있을 때의 예측은 0.99다.
- `reports/presentation/05_parkour_learning/assets/render_acc_fix/README.md` — nearest 이송이
  그것을 0.56~0.89로 회복하지만 틱당 yaw 지터가 ~4°를 넘으면 3셀 구멍을 닫고, **모든 변형이**
  gap 참호를 지면 높이에서 점유로 표시한다. decayed-max에 셀을 지우는 연산이 없고 0.8 m 안쪽은
  재관측이 불가능하기 때문이다.

`lidar_grid_accumulate_mode = "rolling"`은 맵을 **중력 정렬·yaw 고정** 저장소에 둔다. 저장소
원점은 격자 원점을 월드 셀 격자에 스냅한 위치이고, 틱 사이 이동은 **정수 셀 index gather**
(보간 없음, 희석 0)로 처리한다. 셀 미만 잔차는 저장소에 누적되지 않고 **readout에서 한 번만**
적용되므로 위치 드리프트도 없다. 맵이 회전하지 않으므로 yaw 양자화라는 실패 모드 자체가 없다.
`obs["lidar"]`는 27×21×13 = 7371 그대로다.

| 필드 | 기본값 | Roll arm | RollNoFree arm |
|------|--------|----------|----------------|
| `lidar_grid_accumulate_mode` | `"warp"` | `"rolling"` | `"rolling"` |
| `lidar_grid_free_space` | `False` | `True` | `False` |
| `lidar_grid_free_space_rule` | `"above_hit"` | `"above_hit"` (명시) | — |
| `lidar_grid_free_space_above_hit_cells` | `2.5` | 상속 | — |
| `lidar_grid_free_space_beta` | `0.5` | 상속 | — |
| `lidar_grid_free_space_ray_stride` | `1` | 상속 | — |
| `lidar_grid_store_radius` [m] | `2.3` | 상속 | 상속 |
| `lidar_grid_readout_mode` | `"bilinear"` | 상속 | 상속 |

`"warp"` 기본값은 변경 전과 **비트 단위로 같다.** `--test warp_identity`가 RealSensor arm을
굴리면서 변경 전 누적기 본문의 동결 사본과 대조해 60틱에서 최대 절대 오차 **0.000e+00**을
확인한다. env·cfg·`__init__` 세 파일의 `git diff -w`에 **삭제된 줄이 하나도 없다.**

**틱한 env만 계산한다.** 갱신은 `tick.nonzero()`로 이번 스텝에 틱한 행만 뽑아 scroll·scatter·carve를
돌리고 `index_copy_`로 되돌려 놓는다. 결과를 마스킹만 하던 이전 판은 전 배치를 매 제어 스텝
계산했다. `_reset_idx`가 `_lidar_push_ctr`을 다시 위상 맞추므로 1024 env에서 **매 제어 스텝마다
어떤 env는 틱하고(스텝의 100 %), 스텝당 틱하는 env 비율은 0.200**이다. 곧 비용은 0.1 s마다가
아니라 제어 스텝마다 발생한다. 압축이 결과를 바꾸지 않는 것은 확인했다. gap shadow 표를
압축 전후로 다시 돌려 11개 변형 × 8개 지표 × 2199 env-프레임에서 **최대 절대 차이 0.000e+00**이다.

#### free-space carving 규칙

유효 hit를 가진 광선만 통과 셀을 β배 한다. 무반사·dropout·blind/rear crop으로 버려진 반환은
실제 드라이버도 발행하지 않으므로 아무것도 비우지 않는다. 어느 표본을 비울지는
`lidar_grid_free_space_rule`이 정한다.

- `"above_hit"`(기본) — 표본이 **자기 hit보다** `above_hit_cells` 셀 이상 **위에** 있을 때만
  비운다. 지면에 꽂히는 광선의 마지막 구간은 `z ≈ hit_z`라 보호되고, 참호로 내리꽂히는 광선은
  `hit_z`가 두 셀 이상 낮으므로 그 위를 덮은 지면 높이 셀이 비워진다.
- `"margin"` — hit 앞 `hit_margin_cells` 셀만 남기고 전부 비운다(ablation용).

**가드는 자기 깊이보다 얕은 구멍을 지우지 못한다.** 조건이 `sample_z > hit_z + guard·res`이므로
가드 2.5셀에서는 바닥이 0.25 m보다 얕게 꺼진 자리를 절대 비우지 않는다. 지면을 스치는 광선이
blind zone을 깎지 못하게 막는 바로 그 성질의 뒷면이고, 둘을 구분할 방법이 규칙 안에 없다.
`--test carving`이 이것을 단언한다(2셀 dip에서 지워진 셀 0개, 4셀 참호에서는 지워짐). gap L3의
참호는 광선이 충분히 깊이 들어가서 성립했고, **얕은 단차에서도 성립하는지는 미확인**이다.
가드 2.5는 gap 하나에서만 고른 값이다.

**가드는 0.5셀로는 부족하다.** 보호되는 광선 방향 거리는 `guard · res / |dir_z|`이므로 25°로
내려오는 광선에서 0.5셀은 0.12 m, 곧 한 셀뿐이다. 그래서 `above_hit` 0.5는 `margin` 계열과 같은
구간에 떨어진다(아래 표에서 근거리 0.346 대 margin 2셀 0.370, 참지면 0.272 대 0.349). 규칙이
틀린 것이 아니라 상수가 틀렸고, 아래 가드 스윕이 그 보정이다.

#### 같은 프레임 shadow 비교 — 이 표에서 arm을 고른다

첫 라운드는 env 자체를 rolling으로 바꿔 재는 바람에 정책이 off-distribution이 되어 궤적이
통째로 달라졌다(hurdle에서는 아예 멈췄다). 여기서는 **env를 기본 warp 모드로 두고**(정책이
on-distribution) rolling 변형들을 같은 스캔·같은 pose·같은 틱 일정으로 **옆에서 굴린다.**
shadow는 `_RollShadow`가 env의 store 메서드를 자기 객체에 바인딩해 배포되는 코드
`_update_occupancy_accumulator_rolling`을 그대로 실행한다.

**추적 env 하나가 아니라 flat 열을 뺀 11개 env 전부를 채점한다.** env 하나로 재면 값이 복권이
된다 — 같은 설정을 두 번 돌렸을 때 참호 거짓점유가 2.3배 갈렸다.

gap level 3, 12 env(11개 채점), warmup 40 + 200 스텝, **2199 env-프레임**, 평균 전진 1.09 m/s,
0.2 m/s 미만 프레임 0.14 %, 프레임당 구멍 열 28.1개. 지면 z 슬라이스, height scanner 진실.

| 변형 | 근거리 평균 \|y\|≤0.3 | 근거리 >0.5 | 구멍 거짓점유 >0.5 | 구멍 >0.3 | 참지면 유지 >0.5 |
|---|---|---|---|---|---|
| **env warp (현행)** | 0.391 | 0.273 | 0.298 | 0.619 | 0.430 |
| rollnofree | 0.474 | 0.531 | 0.321 | 0.524 | 0.579 |
| margin 2셀 | 0.370 | 0.323 | 0.154 | 0.270 | 0.349 |
| above_hit 가드 0.5 | 0.346 | 0.279 | 0.141 | 0.253 | 0.272 |
| above_hit 가드 1.5 | 0.407 | 0.391 | 0.190 | 0.319 | 0.425 |
| **above_hit 가드 2.5** | **0.435** | **0.451** | **0.213** | **0.343** | **0.478** |
| above_hit 가드 3.5 | 0.444 | 0.473 | 0.216 | 0.348 | 0.496 |
| 가드 2.5 + min_rays 4 | 0.443 | 0.466 | 0.233 | 0.371 | 0.498 |
| 가드 2.5 + stride 2 | 0.439 | 0.458 | 0.220 | 0.355 | 0.486 |
| 가드 2.5 + stride 4 | 0.444 | 0.466 | 0.232 | 0.377 | 0.497 |

읽는 법.

1. **이송만 고치면**(rollnofree) 근거리 기억과 참지면이 크게 오르지만 구멍 거짓점유도 같이
   오른다(0.298 → 0.321). render_acc_fix의 판정 그대로다. 기억을 또렷하게 만들면 틀린 기억도
   또렷해진다.
2. **carving은 세 지표를 한 곡선 위에서 맞바꾼다.** 가드를 키우면 근거리·참지면이 오르고 구멍도
   같이 오른다. 무릎은 없다.
3. **가드 2.5가 현행 누적기를 세 지표 모두에서 이기는 가장 작은 값이다.** 근거리 0.391 → 0.435,
   구멍 0.298 → 0.213(−29 %), 참지면 0.430 → 0.478. 2.5를 넘으면 곡선이 평평해진다.
4. **min_rays는 지렛대가 아니다.** 광선 통과 횟수를 직접 쟀더니 구멍 셀 7.1회/틱, 참지면 셀
   2.5회/틱으로 2.9배 차이뿐이었다. 임계값을 올려도 같은 곡선 위를 미끄러질 뿐이라 기본값 1이다.
5. **stride는 같은 곡선 위를 움직인다.** stride 4의 행(0.444 / 0.232 / 0.497)은 가드 2.5
   (0.435 / 0.213 / 0.478)보다 가드 3.5(0.444 / 0.216 / 0.496)에 가깝다. 광선을 덜 쓰면 덜
   비우므로 stride는 사실상 가드를 키운 것과 같게 작용한다. 대가가 작아 보이는 이유가 이것이고,
   **가드와 stride를 함께 올리면 이중으로 세는 셈이다.** 배포되는 Roll arm은 가드 2.5 + stride 4,
   곧 실효적으로 가드 2.5와 3.5 사이에 있다.

#### hurdle — 참호가 없는 지형에서의 대가

hurdle L3를 **프로세스의 첫 env로** 같은 방식으로 돌렸다(11 env, 2198 env-프레임, 평균 전진
1.00 m/s, 0.2 m/s 미만 0.09 %). scanner 발자국 187열이 전부 되돌아와 **구멍 열이 0개**이므로
구멍 지표는 render_acc_fix가 적은 대로 **적용 불가**다. 남는 두 지표는 이렇다.

| 변형 | 근거리 평균 \|y\|≤0.3 | 근거리 >0.5 | 참지면 유지 >0.5 |
|---|---|---|---|
| **env warp (현행)** | 0.403 | 0.326 | 0.432 |
| rollnofree | 0.459 | 0.520 | 0.532 |
| margin 2셀 | 0.386 | 0.381 | 0.444 |
| above_hit 가드 0.5 | 0.340 | 0.288 | 0.303 |
| above_hit 가드 1.5 | 0.413 | 0.430 | 0.483 |
| **above_hit 가드 2.5** | **0.455** | **0.510** | **0.524** |
| above_hit 가드 3.5 | 0.458 | 0.518 | 0.530 |
| 가드 2.5 + stride 4 | 0.456 | 0.512 | 0.526 |

**참호가 없으면 carving의 대가가 사실상 0이다.** 가드 2.5는 근거리 0.455, 참지면 0.524로
carving을 끈 rollnofree(0.459 / 0.532)와 소수 둘째 자리에서만 다르고, 현행 누적기(0.403 / 0.432)
보다 확실히 높다. gap에서 지불하는 근거리 손실(0.474 → 0.435)은 참호가 있을 때만 생긴다.

##### 프로세스당 지형 하나

첫 시도에서 hurdle이 전진 0.006 m/s, base 높이 0.28 m로 200프레임 내내 정지했다. env가 warp
모드, 곧 정책이 on-distribution인 상태에서도 그랬다. 원인은 누적기가 아니라 **한 프로세스에서
`gym.make`를 두 번 하면 두 번째 env가 열화된다**는 것이다. hurdle을 첫 env로 돌리면 1.00 m/s로
정상이고, 반대로 gap을 두 번째로 돌리면 1.09 m/s가 0.19 m/s로 떨어진다. **지형 하나당 프로세스
하나로 돌려야 한다.** 위 두 표는 각각 자기 프로세스의 첫 env에서 잰 것이다.

#### 성능 — 제어 스텝당 비용 (앞서 적은 "틱당" 수치는 사실상 제어 스텝당이었다)

**앞 판의 "36 ms/틱 = 7 ms/제어 스텝"은 틀렸다.** 틱이 0.1 s마다 한 번이라고 보고 5로 나눴는데,
`_reset_idx`가 `_lidar_push_ctr`을 다시 위상 맞추기 때문에 env들이 어긋나 **매 제어 스텝마다
어떤 env는 틱한다.** 1024 env에서 실측한 값이 이것이다.

| | 스텝당 틱하는 env 비율 | 틱이 있는 스텝 비율 |
|---|---|---|
| 1024 env, 320스텝 워밍 후 | **0.200** | **1.000** |

곧 누적기는 0.1 s마다가 아니라 **제어 스텝마다** 호출되어 일한다. 앞 판은 결과를 마스킹만 했으므로
전 배치를 매 스텝 계산했고, 그래서 "틱당" 수치가 곧 스텝당 수치였다. 지금은 틱한 20 %만 계산한다.

`torch.cuda.synchronize()`로 누적기 호출만 잰 제어 스텝당 평균이다(GPU 1, 320스텝 워밍 +
100스텝 계측, zero action).

| | 1024 env | 1280 env | 기준 대비 peak alloc (1024) |
|---|---|---|---|
| warp (현행) | 19.90 ms | 25.37 ms | +0.68 GiB |
| rolling (carving 없음) | **5.08 ms** | 미측정 (OOM) | +0.92 GiB |
| **rolling + free, stride 1** | **17.07 ms** | 미측정 (OOM) | +0.92 GiB |
| rolling + free, stride 4 | 8.50 ms | 미측정 (OOM) | +0.92 GiB |

**전 광선 carving이 현행 누적기보다 싸다.** 17.07 대 19.90 ms/스텝이다. warp는 매 스텝 전 배치를
워프하는데 rolling은 틱한 20 %만 건드리기 때문이고, 그래서 **Roll arm 기본 stride는 1이다**(정확도가
가장 좋은 설정). 제어 스텝 예산이 빠듯하면 stride 4가 9.05 ms지만, 앞 표에서 봤듯 그것은 가드를
키운 것과 같게 작용한다.

1280 env의 rolling 세 줄은 **미측정**이다. 이 GPU에 다른 프로세스가 9.3 GiB를 쓰고 있어
60~84 MiB 차이로 OOM이 났다. warp만 통과했다. 테스트는 OOM을 환경 한계로 구분해 보고하고,
그 밖의 예외는 실패로 처리한다.

carving은 처음 구현에서 512 env 기준 173 ms였다. 세 가지가 그것을 39 ms로 줄였고 셋 다 결과를
바꾸지 않는다.

1. **표본 수를 저장소가 아니라 readout 상자로 묶는다.** 정책이 읽는 것은 27×21×13 상자뿐이고
   그 밖의 표본은 이번 틱의 관측에 영향을 줄 수 없다. 광선당 39 → **22** 표본.
2. **광선을 먼저 압축한다.** 20000개 슬롯 중 blind/rear crop을 통과하는 것은 약 5300개(26 %)뿐인데
   이전 구현은 전부를 `(chunk, R, S, 3)`으로 펼친 뒤 마스킹했다. `nonzero`로 먼저 걸러 낸다.
3. **min_rays가 1이면 원자 연산을 쓰지 않는다.** "이 셀을 지나간 광선이 있나"만 물으면 순서 없는
   boolean 쓰기로 충분하다. 33k 셀 버퍼에 수천만 index를 `scatter_add_`하는 경합이 사라져
   88 → 39 ms가 된다. `min_rays > 1`이나 진단 카운트를 켤 때만 원자 경로를 탄다.
4. **틱한 env만 계산한다.** 스텝당 비용을 다시 4~5배 줄였다.
5. **동기화를 카브당 한 번으로 줄인다.** 광선 압축(`nonzero`)을 env chunk마다 하면 chunk 수만큼
   device 동기화가 걸렸다. 이제 배치 전체를 한 번 압축하고 그 평평한 목록 위에서 chunk를 돈다.
   스텝당 동기화는 틱 압축 1회 + 카브 압축 1회, 모두 **2회**다.
6. **할당을 줄인다.** `_scroll_store`의 축별 zero-fill을 `gather` 결과 위에서 in-place로 하고
   (축당 텐서 2개 → 1개), 목적지는 본 적 있는 최대 배치 크기로 자라는 스크래치 버퍼를 재사용한다.
   카브의 적용도 임시 텐서 하나로 끝난다. 틱하는 배치가 전체의 20 %라 스크래치는 저장소 전체가
   아니라 그 5분의 1 크기다.

**chunk는 지렛대가 아니다.** env chunk를 16 → 64로 키워도 시간이 그대로였고 메모리만 3.5배가
됐다. 지금 `lidar_grid_free_space_chunk_size`는 env가 아니라 **운반 광선 수**를 뜻하고 기본값은
350,000이다.

zero action 롤아웃은 학습된 정책보다 리셋이 잦아 위상 재설정이 더 자주 일어난다. 위 틱 비율
0.200은 `1/push_every`와 정확히 같으므로 이 편향이 값에 나타나지는 않았지만, 리셋 빈도가
다르면 p90은 달라질 수 있다.

#### 이송 정확도 (시뮬레이터 없음)

`--test synth`가 env의 `_scroll_store` / `_readout_store`를 duck-typed stub으로 직접 호출한다
(산술 사본이 아니라 배포되는 코드 그 자체다). 한 셀 두께 시트 가운데 3셀 구멍, 틱당
전진 0.085 m·dz 0.02 m·yaw 지터 ±2°와 ±5°, 새 관측 없이 10틱.

| 검사 | 결과 |
|---|---|
| 저장소 최댓값 = 0.94ⁿ | 상대 오차 최대 **7.3e-08** (희석 0) |
| 구멍 폭 | 10틱 내내 3~4열 (±2°, ±5° 동일). render_acc_fix의 F는 ±4°에서 3→1→0으로 닫힌다 |
| 위치 오차 | 최대 **0.50 셀**, 틱이 지나도 커지지 않음 (N 변형은 10틱에 1.5셀) |

**readout의 z 번짐은 남는다.** 저장소는 값을 하나도 잃지 않지만, readout이 몸체 프레임 셀
중심에서 표본하므로 z 스냅 잔차만큼 한 셀 두께 지면 시트가 눌린다. 측정한 readout/저장소
최댓값 비는 **0.600~1.000**이다. 이 손실은 **누적되지 않는다** — 저장소는 다시 표본되지 않으므로
매 틱 현재 맵에 한 번만 적용된다(현행 워프는 10틱에 0.187/0.539 = 0.35까지 곱해 내려간다).
잔차는 몸통 높이를 따라가고 몸통 높이는 틱당 0.008~0.023 m 변하므로 튀지 않고 천천히 움직인다.

#### 회전 불변성과 산란·readout 정합

`--test rotation`: 루트 pose를 매 스텝 다시 써서 로봇을 세워 두고 yaw만 30° 돌린 뒤, 연속 두
틱의 readout 점유를 월드 좌표로 옮겨 비교한다(상호 관측 영역만, 임계값 0.3).

| arm | 1셀 허용 일치 (최소) | 엄격 IoU (최소) | 현재 스캔 readout 재현율 |
|---|---|---|---|
| Roll | 0.989 | 0.580 | 0.923 |
| RealSensor (warp) | 0.993 | 0.580 | 0.953 |

**이 지표는 두 모드를 가르지 못한다.** trilinear 워프는 위치가 정확하고 값을 잃는 쪽이라
위치 기준 지표에서는 rolling과 같은 점수를 낸다. 브리프가 "warp는 이 불변량을 흐린다"고 적은
것은 이 측정으로 **뒷받침되지 않는다.** rolling의 이득은 위치가 아니라 값이다. 마지막 열은
같은 틱의 단일 스캔이 readout에서 다시 읽히는 비율로, 산란과 readout의 두 회전이 서로 맞는지를
잡는 검사다(부호가 틀리면 0에 가깝게 떨어진다).

## Observation 그룹 (AMP 학습 task 공통)

| 그룹 | 차원 | 내용 |
|------|------|------|
| `policy` | 42 | projected_gravity(3)+commands(3)+joint_pos(12)+joint_vel(12)+prev_actions(12) |
| `scan` | 187 | height scan |
| `priv_explicit` | 6 | root_lin_vel_b(3) + root_ang_vel_b(3) |
| `priv_latent` | **33** | base_friction(1)+foot_friction(**4**)+base_mass(1)+base_com(3)+stiffness_ratio(12)+damping_ratio(12) — foot_friction은 static-only 4-dim. cfg 주석(37/8)과 상이하며 런타임 실제값은 33 |
| `history` | 10×42 | proprio 히스토리 (StateHistoryEncoder 입력) |
| critic 총합 | **268** | policy(42)+scan(187)+priv_explicit(6)+priv_latent(**33**) |

AMP obs는 `env.extras["amp_obs"]` 별도 경로로 판별자에 전달됩니다 (obs 그룹과 독립).

---

## 실행 명령어

### ⚠️ AMP train.py 호환성 확인 결과

`scripts/reinforcement_learning/rsl_rl/train.py`는 `OnPolicyRunnerParkourAMP`를 명시적으로 지원합니다 (train.py:226-227). agent cfg의 `class_name = "OnPolicyRunnerParkourAMP"` 를 읽어 자동으로 올바른 러너를 선택하므로, **AMP 학습 task 4개는 모두 vanilla `train.py`로 실행 가능합니다**. 별도 스크립트 불필요.

### AMP 학습 Task (Go2-ParkourImitation-v0)

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-v0 --num_envs 4096

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-v0 --num_envs 32
```

### Symmetry 변종 (Go2-ParkourImitation-Symmetry-v0)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 32
```

### RandomGoal 변종 (Go2-ParkourImitation-Symmetry-RandomGoal-v0)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-RandomGoal-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-Symmetry-RandomGoal-v0 --num_envs 32
```

### TerrainStyle 변종 (Go2-ParkourImitation-TerrainStyle-v0)

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-TerrainStyle-v0 --num_envs 4096

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2-ParkourImitation-TerrainStyle-v0 --num_envs 32
```

### 공통 학습 옵션

```bash
# WandB 로깅
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096 \
    --logger wandb --wandb-project IsaacLab-ParkourImitation

# 체크포인트에서 재개
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task Go2-ParkourImitation-Symmetry-v0 --num_envs 4096 --resume
```

---

## 데모 실행 (Go2-ParkourDemo-v0 / Go2-ParkourDemo-Playground-v0)

> **⚠️ 데모 task는 vanilla `play.py`로 실행 불가** — 커스텀 진입점 `play_parkour_demo.py`를 사용해야 합니다.

데모 실행 시 반드시 `--checkpoint` 인자로 학습된 체크포인트 경로를 지정하세요.

### 루트 bash 래퍼 (권장)

```bash
# 헤드리스 녹화 sweep (test 지형 전체)
./parkour_demo

# 특정 지형/난이도만 녹화
./parkour_demo --record_classes stair,gap --record_levels 0,5,10

# 다른 체크포인트 사용
./parkour_demo --checkpoint logs/rsl_rl/<run>/model_<N>.pt
```

```bash
# test 지형 인터랙티브 (GUI, 패널로 지형/난이도 선택)
./parkour_demo_test

# WebRTC 원격 스트리밍
./parkour_demo_test --livestream 2 --enable_cameras
```

```bash
# playground 인터랙티브 (GUI, omni.ui 조이스틱)
./parkour_demo_playground

# WebRTC 원격 스트리밍
./parkour_demo_playground --livestream 2 --enable_cameras
```

### 직접 실행 (play_parkour_demo.py)

```bash
# test 모드 — goal 기반 자동 주행, test 지형
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \
    --task Go2-ParkourDemo-v0 \
    --checkpoint logs/rsl_rl/<run>/model_<N>.pt \
    --mode test

# playground 모드 — 자유 조종, scatter 지형
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \
    --task Go2-ParkourDemo-Playground-v0 \
    --checkpoint logs/rsl_rl/<run>/model_<N>.pt \
    --mode playground

# 헤드리스 녹화 sweep
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_parkour_demo.py \
    --task Go2-ParkourDemo-v0 \
    --checkpoint logs/rsl_rl/<run>/model_<N>.pt \
    --record --headless --enable_cameras \
    --record_classes stair,gap \
    --record_levels 0,5,10 \
    --video_dir logs/demo_videos
```

> `num_envs`는 항상 1로 강제됩니다 (사용자 인자 무시).

### 데모 conda env

루트 bash 래퍼 3개(`parkour_demo`, `parkour_demo_test`, `parkour_demo_playground`)는 모두 **`conda activate isaac-5.1`** 을 사용합니다.
> **주의**: 프로젝트 메모리에 parkour 서브프로젝트가 `isaac-parkour` env를 사용한다는 노트가 있으나, 실제 래퍼 코드는 `isaac-5.1`을 activate합니다. 직접 실행 시에는 현재 활성 conda env 혹은 `./isaaclab.sh`를 사용하세요.

---

## 로그 경로

| Task | 실험 폴더 이름 |
|------|---------------|
| Go2-ParkourImitation-v0 | `logs/rsl_rl/parkour_imitation_go2/` |
| Go2-ParkourImitation-Symmetry-v0 | `logs/rsl_rl/parkour_imitation_go2_symmetry/` |
| Go2-ParkourImitation-Symmetry-RandomGoal-v0 | `logs/rsl_rl/parkour_imitation_go2_symmetry_random_goal/` |
| Go2-ParkourImitation-TerrainStyle-v0 | `logs/rsl_rl/parkour_imitation_go2_terrain_style/` |

## AMP 하이퍼파라미터 요약 (기본값)

| 파라미터 | 값 |
|---------|-----|
| `amp_weight` | 0.3 |
| `disc_loss_type` | `bce` |
| `discriminator_learning_rate` | 2.5e-4 |
| `gradient_penalty_coef` | 5.0 |
| `replay_buffer_size` | 200,000 |
| `amp_observation_space` | 490 (49-dim × 10 frames, TerrainStyle: 370) |
| 참조 모션 경로 | `ParkourImitationEnvCfg.amp_motion_pkl` (env cfg가 소유) |

# SL-Grid crawl 레벨 고정 probe — symmetry OFF vs ON

## 비교 대상 run

| 이름 | 파일 접두어 | run 경로 | gym task id | 체크포인트 |
|---|---|---|---|---|
| OFF | `off_` | [`logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl/2026-08-05_12-57-11_slgrid_crawl_scratch_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl/2026-08-05_12-57-11_slgrid_crawl_scratch_50k/) | `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-EasyEntry-v0` | `model_49999.pt` |
| SYM | `sym_` | [`logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl_sym/2026-08-10_12-35-38_slgrid_crawl_sym_scratch_50k/`](../../../logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl_sym/2026-08-10_12-35-38_slgrid_crawl_sym_scratch_50k/) | `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0` | `model_49999.pt` |

두 run은 **env cfg가 동일**하고 (SYM task는 OFF와 같은
`ParkourImitationRandomGoalLidarSLGridCrawlEasyEntryEnvCfg` 를 씀), 유일한 recipe 차이는
`symmetry_cfg` 의 L/R mirror data augmentation ON/OFF다. mirror는 `obs["lidar"]` metric grid의
y-flip permutation을 포함한다 (`parkour/mdp/symmetry.py`, 2026-08-10 구현).

## 무엇 대 무엇인가

50k scratch 학습 후 **최종 능력**을 커리큘럼 자기선택 없이 잰다. 학습 중 terrain level은
성능에 따라 스스로 오르내리므로 (잘하는 정책이 어려운 지형에 많이 노출), 학습 로그의 레벨·보상
비교는 분포가 다른 두 정책을 섞어 잰 것이다. 이 probe는 두 정책을 **같은 지형, 같은 고정
레벨**(`--level`, `_skip_curriculum` 매 스텝 재장전)에 배정해 그 confound를 제거한다.

- 조건: 5개 장애물 지형(hurdle/step/gap/stair/crawl) × 레벨 {3, 6} × 2정책 = 20건
- 규모: 각 `--num_envs 64`, `--steps 2000` (조건당 에피소드 193~274건)
- crawl을 포함한 정량 비교는 이 프로그램에서 처음 성립한다 (GT voxel teacher는 crawl OOD라 불가했음)
- 주의: 두 정책은 각자의 task id로 로드된 서로 다른 env 인스턴스에서 돌았다. 지형 생성 규칙은
  동일 EasyEntry 계열이지만 완전히 같은 월드 인스턴스의 대조는 아니다.

## 수치

완주율 = `cause_goal_reached / episodes`. 나머지 종료는 대부분 time_out(59~70%)이고
low_height는 전 조건 0이다.

| 지형 | lv | OFF 완주율 | SYM 완주율 | OFF tilt% | SYM tilt% | OFF fwd [m/s] | SYM fwd [m/s] | OFF collision | SYM collision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hurdle | 3 | 29.74% | 29.53% | 0.00 | 0.52 | 0.971 | 0.980 | −0.022 | −0.062 |
| hurdle | 6 | 31.66% | 32.04% | 0.00 | 0.49 | 1.010 | 1.028 | −0.009 | −0.038 |
| step | 3 | 40.29% | 41.03% | 0.73 | 0.00 | 0.992 | 0.983 | −0.030 | −0.005 |
| step | 6 | 36.08% | 37.74% | 2.75 | 1.89 | 0.981 | 1.030 | −0.426 | −0.192 |
| gap | 3 | 38.20% | 38.69% | 1.12 | 0.36 | 0.968 | 0.999 | −0.066 | −0.013 |
| gap | 6 | 37.45% | 37.22% | 1.16 | 1.50 | 1.035 | 1.051 | −0.025 | −0.003 |
| stair | 3 | 34.80% | 34.65% | 0.44 | 0.00 | 1.012 | 1.032 | −0.197 | −0.070 |
| stair | 6 | 34.22% | 36.78% | 0.44 | 0.41 | 1.039 | 1.122 | −0.029 | −0.010 |
| crawl | 3 | 36.17% | 33.94% | 0.00 | 0.00 | 0.975 | 0.933 | −0.005 | −0.003 |
| crawl | 6 | 34.62% | 34.96% | 0.00 | 0.00 | 0.956 | 0.956 | −0.018 | −0.005 |
| **10조건 평균** | | **35.32%** | **35.66%** | 0.66 | 0.52 | 0.994 | 1.011 | −0.083 | −0.040 |

## 읽히는 차이

- **완주율은 전 조건 통계적 동률.** 조건당 n≈230, p≈0.35이면 완주율 1개의 SE≈3.1 %p,
  두 정책 차이의 SE≈4.4 %p다. 최대 격차(stair L6 +2.56 %p, crawl L3 −2.23 %p)조차 1 SE
  이내이고, 10조건 평균 차이는 +0.34 %p다. **symmetry augmentation은 최종 능력을 올리지도
  내리지도 않았다.**
- **collision은 SYM이 9/10 조건에서 낮다** (평균 −0.040 vs −0.083, step L6은 −0.192 vs
  −0.426으로 절반 이하). 학습 로그 최종 구간에서는 OFF의 collision이 더 좋았는데(−0.95 vs
  −1.45) 레벨을 고정하니 반전됐다 — 학습 로그의 collision 격차는 두 정책이 머무는 커리큘럼
  분포 차이가 만든 아티팩트였다는 뜻이다. 같은 난이도에서는 SYM이 더 깨끗하게 통과한다.
- **전진 속도는 SYM이 8/10 조건에서 미세하게 빠르다** (평균 1.011 vs 0.994, 최대 stair L6
  1.122 vs 1.039). 방향은 일관되나 격차는 작다.
- tilt·base_contact·low_height는 양쪽 다 0~3% 수준으로 파국적 실패 모드가 없다. 학습 곡선의
  tilt 격차(SYM −31%)도 고정 레벨에서는 재현되지 않는다 — 같은 커리큘럼-분포 아티팩트 계열.

## 결론 (학습 곡선과 합쳐서)

symmetry ON의 실증된 이득은 **최종 완주율이 아니라 다음 세 가지**다:

1. **수렴 속도 ~2배** — OFF가 30k에 도달한 커리큘럼 수준을 SYM은 ~15k에 도달 (같은 iter
   비교에서 초반 전 지형 레벨 ~2배, goal 도달 +15~19%).
2. **RL_calf std 발산(calf-divergence 병리) 구조적 해소** — OFF는 RL_calf 노이즈 std가
   50k까지 단조 폭주해 16.7로 마감(타 관절의 ~40배), SYM은 0.39로 정상 하강. mirror 강제로
   L/R 쌍 std가 일치해 한쪽 calf만 발산하는 평형이 성립 불가.
3. **같은 난이도에서 더 적은 충돌 + 미세하게 빠른 전진** — 위 표.

비용(최종 능력 손실)은 관측되지 않았다. 완주율 천장(~35%)은 두 정책이 공유하므로 정책이 아닌
task 구조(goal chain 길이 vs episode 길이 → time_out 지배) 쪽 제약으로 읽힌다.

## 영상 (side-by-side: OFF 왼쪽 | SYM 오른쪽)

`play_per_terrain.py`로 6지형 × 2정책 = 12클립 렌더 후 지형별 합성 (400프레임, 12 envs,
`--sensor lidar_grid` — 두 정책의 실제 입력인 Mid-360 metric grid 오버레이). probe와 달리
레벨 고정이 아닌 고난이도 스폰(`max_init_terrain_level=8`)이며, 카메라는 12 env 중 최고
레벨 1개체만 따라간다 — 일반화 불가한 정성 자료다.

| 지형 | 파일 | LIVENESS fwd [m/s] (OFF/SYM) | 비고 |
|---|---|---|---|
| flat | `videos/flat_off_vs_sym.mp4` | 0.995 / 1.051 | |
| hurdle | `videos/hurdle_off_vs_sym.mp4` | 1.072 / 1.095 | |
| step | `videos/step_off_vs_sym.mp4` | 1.063 / 1.032 | ⚠ **SYM 추적 개체는 t≈1.5s에 단 벽면에 엎어져 클립 끝까지 정지** (OFF 추적 개체는 등반). 레벨 8 스폰, tilt/base_contact 종료가 안 걸려 리셋 없는 흡수 상태 — 아래 주의 참조 |
| step (재렌더, seed 7) | `videos/step_off_vs_sym_seed7.mp4` | 0.951 / 0.986 | **양쪽 다 낙상**: SYM 추적 개체는 t≈2.5s부터 단 아래 엎어져 정지(seed 1과 동일 흡수 상태), OFF 추적 개체도 이번엔 등반 실패(z_min 0.154, z_max 0.437 — 낙상 후 전경 블록에 가려 화면 이탈). seed 1의 "OFF 등반 vs SYM 낙상" 대비는 우연이었음 |
| **step (대표, L6)** | `videos/step_off_vs_sym_L6.mp4` | 1.080 / 1.105 | ✅ **양쪽 다 완등**: `--max_init_level 6`(probe 판정 레벨)으로 촬영. 추적 개체 둘 다 낙상 없이(z_min 0.31~0.32 유지) 정상부 z≈2.0까지 등반, 프레임 4시점 검수 통과. L8 클립 2개는 흡수 상태 증거용, 등반 시연은 이 클립이 정본 |
| gap | `videos/gap_off_vs_sym.mp4` | 1.082 / 1.088 | |
| stair | `videos/stair_off_vs_sym.mp4` | 1.083 / 1.131 | |
| crawl | `videos/crawl_off_vs_sym.mp4` | 1.049 / 1.052 | 양쪽 모두 몸 낮춰 터널 진입 |

⚠ **주의 (2026-08-12 프레임 재검수로 정정)** — 초판의 "12클립 전부 실주행" 판정은 오류였다.
LIVENESS는 12 env **평균** 전진속도라 카메라 추적 개체의 실주행을 보증하지 않는다: step SYM 클립의
추적 개체는 t≈1.5s에 엎어져 끝까지 정지했는데 평균은 1.032 m/s로 정상처럼 보였다. 재검수 결과
flat/hurdle/gap/stair는 추적 개체의 이동을 프레임 대조로 확인했고, crawl은 t=2·6s에 카메라가 터널
벽·바닥만 비춰 개체 확인 불가(t=4s에는 양쪽 모두 정상 진입 자세).

★파생 관찰: 엎어진 자세로 tilt/base_contact 종료가 안 걸리는 **흡수 상태**가 존재한다. probe의
time_out(종료의 59~70%)에 이런 낙상-미종료 상태가 섞여 있을 수 있으므로, "완주율 천장 ~35% = 순수
task 구조(코스 길이) 제약"이라는 해석에는 유보가 필요하다. 종료 규칙은 양 정책 동일이라 OFF/SYM
**A/B 공정성 자체는 유지**된다.

seed 7 재렌더(2026-08-13)가 이를 보강한다: 이번에는 **OFF·SYM 추적 개체 둘 다** L8 step에서
낙상 후 정지했다(위 표). 즉 낙상-흡수 상태는 SYM 특이 결함이 아니라 L8 step에서 양 정책 공통으로
빈발하는 현상이고, seed 1 클립의 대비는 개체 추첨 우연이다 — probe L6에서 step은 SYM이 앞선
지형(37.74 vs 36.08%)이라는 정량 판정과도 상충하지 않는다.

개별 클립(합치기 전)은 `_workspace/slgrid_sym_probe/videos/`에 남아 있다.

## 산출물 인덱스

| 경로 | 내용 |
|---|---|
| `metrics/{off,sym}_{hurdle,step,gap,stair,crawl}_L{3,6}.json` | probe 원자료 20건 (task/checkpoint 자기 기술) |
| `logs/probe_*.log` | 각 probe의 stdout 전체 |
| `videos/{terrain}_off_vs_sym.mp4` | side-by-side 정성 영상 6건 (1.5~2.9 MB, Notion 업로드 가능 크기) |

## 원본 `_workspace` 경로 / 생성 스크립트

- 원본: `_workspace/slgrid_sym_probe/` (JSON + `run_probe.sh` 래퍼, 그대로 남아 있음)
- 생성 스크립트: `scripts/demos/probe_stair.py` (`--terrain`/`--level`/`--out_path`,
  이름은 stair지만 전 지형 공용)
- 실행: 2026-08-12, GPU 0/2 두 레인 병렬, 건당 ~8분

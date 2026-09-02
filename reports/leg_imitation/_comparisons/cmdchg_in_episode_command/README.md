# 에피소드 도중 명령 재샘플링(`cmdchg`) — 속도-보행 전환이 생기는가 (2026-09-01~)

## 배경

[`rsi_command_matched_gait`](../rsi_command_matched_gait/README.md) 에서 확정한 것:

1. 걸음을 고르는 것은 **초기 자세**이고, RSI 명령-매칭은 초기화 artifact 만 만든다.
2. 진짜 원인은 `_post_physics_step` 이 호출되지 않아 **명령이 에피소드 내내 고정**된 것 —
   정책이 "달리는 중 명령 변화"를 한 번도 겪은 적이 없어 전환을 학습할 신호가 없었다.

그래서 훅을 되살렸다(`resample_command_in_episode`, 기본 False). 이 문서는 그 A/B 다.

- 새 팔 `cmdchg` : `2026-09-01_09-04-53_cmdchg_velscale15_ds14_wcmd_vmax32`
- 대조 `baseline`: `2026-08-26_16-51-43_velscale15_ds14_wcmd_vmax32`
- cfg 실질 차이는 **`resample_command_in_episode: true` 하나**

## 상태: 학습 중 (40.0k / 50k)

## 판정 요약 (반복 측정 — 매 지점 iteration 을 맞춰 잰다)

| 지표 | cmdchg@20k | cmdchg@30k | cmdchg@40k | (같은 iter baseline) |
|---|---|---|---|---|
| 전환 **저속→고속** (pace→gallop) | **0/193** | **1/188** | **0/204** | 표본 없음 |
| 전환 고속→저속 (gallop→pace) | 23/64 (36%) | 23/91 (25%) | 16/84 (19%) | 표본 없음 |
| 속도 선택성 (1.5\~2.5 → 2.5\~3.5) | 17.9 → 14.4 | 14.3 → 13.5 | 15.9 → 19.6 | 23.7 → 22.2 |
| 초기자세 대조 (RSI vs 정지) | 14.4 vs **0.3** | 13.5 vs **0.2** | 19.6 vs **0.0** | 22.2 vs **0.0** |

**저속→고속 전환은 20k·30k·40k 내내 0 근처다**(합쳐 1/585 = 0.2%). 아래쪽 방향은 계속
19\~36% 로 살아 있으니 프로브가 못 잡는 게 아니다. 초기자세 대조도 40k 에서 다시 **0.0%** —
정지에서 출발하면 gallop 이 아예 없다.

## ★ 40k 판정 — gallop 이 **저속 쪽**에 몰린다

![cmdchg 40k 판정](figures/gait_verdict_cmdchg40k.png)

왼쪽 곡선의 **최고점이 cmd 1.0 (24.8%)** 이다. 참조상 gallop 은 2.69 m/s 이상의 걸음인데,
정책은 그 아래에서 gallop 을 더 많이 낸다. baseline 은 저속에서 낮고(12.4%) 고속에서 높다
(22\~24%) — 약하게나마 옳은 방향인데, cmdchg 는 **거꾸로**다.

즉 명령 재샘플링이 만든 것은 속도-보행 사다리가 아니라 **속도와 무관한, 오히려 저속에 치우친
gallop** 이다. 전환 프로브 0/204 와 합치면 그림이 일관된다: 걸음은 여전히 초기 자세가 정하고,
명령이 바뀌어도 따라가지 않는다.

### 영상 (추적 카메라, model_40000, 고정 3.0 m/s)

| 파일 | 출발 | 실제 보행 |
|---|---|---|
| `videos/cmdchg40k_const30_reference_start_GALLOP.mp4` | 참조 프레임 | **gallop** 2.29 Hz |
| `videos/cmdchg40k_const30_standing_start_PACE.mp4` | 정지 기립 | **pace** 2.36 Hz |
| `videos/cmdchg40k_ramp_standing_start_PACE.mp4` | 정지, 0→3.2 램프 | 전 구간 **pace** |

⚠️ 참조 프레임 출발 2 회 중 1 회가 gallop 이었다(19.6% 조건). 위 개체는 그 조건의 예시이지
전형이 아니다. 정지 출발은 4096 env 에서 **0.0%** 라 pace 가 곧 전형이다.

## 30k 판정

![cmdchg 30k 판정](figures/gait_verdict_cmdchg30k.png)

왼쪽 곡선이 **평평해지다 못해 내려간다**(14.3 → 13.5). baseline 은 같은 지점에서 21.2 → 24.8 로
올라간다. 즉 명령 재샘플링은 속도 선택성을 만들기는커녕 gallop 을 전반적으로 줄였다.

오른쪽에서 위쪽 막대(원하는 전환)는 20k 0/193 에서 30k **1/188** 로 사실상 그대로다.
아래쪽 막대는 계속 20\~36% 라 프로브는 정상 작동한다.

### 영상 (추적 카메라, model_30000, 고정 3.0 m/s)

| 파일 | 출발 | 실제 보행 |
|---|---|---|
| `videos/cmdchg30k_const30_reference_start_PACE.mp4` | 참조 프레임 | **pace** 2.37 Hz |
| `videos/cmdchg30k_const30_standing_start_PACE.mp4` | 정지 기립 | **pace** 2.36 Hz |
| `videos/cmdchg30k_ramp_standing_start_PACE.mp4` | 정지, 0→3.2 램프 | 전 구간 **pace** |

⚠️ 이번엔 **참조 프레임 출발도 pace 다.** 뽑기 운이 아니라 그 조건의 gallop 비율이 13.5% 로
내려왔기 때문이다(20k 에는 같은 조건 2 회 중 1 회가 gallop 이었다). 같은 조건 2 회를 찍었는데
둘 다 pace 였고, 대표성이 있는 쪽은 이제 pace 다.

## 20k 중간 판정 (기록)

## ★ 20k 판정: 전환은 **한 방향으로만** 일어난다

![cmdchg 20k 판정](../rsi_command_matched_gait/figures/gait_verdict_cmdchg_20k.png)

전환 프로브가 처음으로 실질 표본을 얻었다(명령이 바뀌어야 표본이 생긴다). 472 개 중:

| 명령 변화 | 걸음이 따라 바뀐 비율 |
|---|---|
| 저속 → 고속 (**pace → gallop**, 원하는 전환) | **0 / 193 = 0.0%** |
| 고속 → 저속 (gallop → pace) | **23 / 64 = 35.9%** |

**아래쪽 방향은 프로브가 전환을 실제로 감지할 수 있다는 양성 대조다** — 도구가 못 잡는 게
아니라 위쪽 전환이 **일어나지 않는다.** 걸음 유지율은 저속→고속에서 99.2%.

가속 시 gallop 으로 못 올라가지만 감속 시 gallop 에서 떨어져 나오기는 한다. gallop 이 저속에서
유지 불가능한 걸음이라는 점을 생각하면 자연스러운 비대칭이다 — **우물 밖으로 밀려나는 것은
되지만, 우물 안으로 올라가는 것은 안 된다.**

## 속도 선택성도 안 생겼다

| cmd vx | baseline@20k | cmdchg@20k |
|---|---|---|
| 1.5~2.5 | 20.0% | 17.9% |
| 2.5~3.5 | 23.1% | **14.4%** |

baseline 과 마찬가지로 평평하고, 오히려 고속 칸이 낮다. 참조 gallop 속도(2.69 m/s) 위에서
올라가는 단조 증가는 없다.

## 초기 자세 대조도 그대로

| model_20000, cmd 2.5~3.5 | RSI 출발 | 정지 출발 |
|---|---|---|
| baseline | 23.1% | **0.0%** |
| cmdchg | 14.4% | **0.3%** |

명령이 도중에 바뀌게 만들어도 **정지에서 출발하면 여전히 gallop 이 0** 이다. 초기 자세가
걸음을 잠그는 구조는 깨지지 않았다.

## 학습 지표 — 이번엔 실제로 물렸다

`rsimatch` 는 모든 지표가 1% 안쪽이었는데(그래서 안 물린 걸 의심했어야 했다), 이번엔 다르다:

| 9.5k~10k 평균 | baseline | cmdchg | Δ |
|---|---|---|---|
| `mean_reward` | 362.83 | 342.13 | −20.71 (−5.7%) |
| `amp_reward` | 27.29 | 23.96 | −3.32 (−12%) |
| `mean_noise_std` | 0.2302 | 0.3558 | **+55%** |

다만 **속도 추종은 손상되지 않았다** — 명령 구간별 실제 속도가 0.17/0.90/1.98/2.73 으로
baseline(0.15/0.88/1.99/2.74)과 사실상 같다. 보상 하락은 정상상태 추종이 나빠져서가 아니라
**명령 변경 직후 과도구간**이 새로 생겼기 때문으로 읽는 게 맞다.

## 영상 (추적 카메라, model_20000)

| 파일 | 조건 | 실제 보행 |
|---|---|---|
| `../rsi_command_matched_gait/videos/cmdchg20k_const30_reference_start_GALLOP.mp4` | 고정 3.0, 참조 프레임 출발 | **gallop** 2.66 Hz |
| `../rsi_command_matched_gait/videos/cmdchg20k_const30_standing_start_PACE.mp4` | 고정 3.0, 정지 출발 | **pace** 2.37 Hz |
| `../rsi_command_matched_gait/videos/cmdchg20k_ramp_standing_start_PACE.mp4` | 0→3.2 램프, 정지 출발 | 전 구간 **pace** (1.82→2.34 Hz) |

⚠️ 캡션 근거: 세 클립 모두 `gait_classify.py` 로 분류해 확인했다. 고정 3.0 · RSI 출발은
같은 조건 2 회 중 1 회가 gallop 이었다(다른 하나는 pace) — 위 개체는 대표값이지 100% 가 아니다.
정지 출발은 4096 env 에서 0.3% 라 pace 가 전형이다.

## 아직 판정이 아니다

20k 는 학습의 40% 지점이다. baseline 의 gallop 발현이 10k\~20k 에 몰려 있었으므로 그 뒤로
전환 능력이 따로 자랄 여지는 남아 있다. 다만 저속→고속 0/193 은 표본이 충분한 **하드 제로**라,
"조금씩 생기는 중"으로 보기는 어렵다.

30k / 40k / 50k 에서 같은 세 지표(전환 프로브 · 속도 선택성 · 초기자세 대조)를 반복한다.

## 산출물

- `figures/`(rsi_command_matched_gait 쪽) `gait_verdict_cmdchg_20k.png`
- `_workspace/leg/plot_gait_verdict.py` — 판정 그림 생성기(npz 직접 읽음, 양방향 전환 패널)
- `_workspace/leg/gait_transition_probe.py` — 명령 변경 전후 걸음 비교

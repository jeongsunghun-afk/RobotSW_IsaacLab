# 속도 조건부 discriminator — 걸음이 처음으로 명령을 따라간다 (2026-09-02~)

> 작성 일시: 2026-09-03 09:13 (소급 · 근거: 디렉터리 내 최초 산출물 mtime)
> 비교 대상: uncond(cmdchg4s) / cond-mlp(condmatch) / cond-drail(condmatch_drail) — AMP discriminator 속도 조건화 여부·구조

`amp_cond_mode: speed` — AMP obs 끝에 `[ |v_cmd| / v_max , valid ]` 를 붙여 판별자를 **명령
속도에 조건화**한다. 나머지는 `cmdchg4s` 와 같다(20 s 에피소드 · 4 s 고정 주기).

| run | disc | iter |
|---|---|---|
| `condmatch_cmdchg4s_ep20_...` | mlp + speed 조건 | 20k 측정 |
| `condmatch_drail_cmdchg4s_ep20_...` | **drail** + speed 조건 | 10k 측정 |
| 대조 `cmdchg4s_ep20_...` | mlp, 무조건부 | 20k |

## 결론 — 앞선 처방 셋이 못 넘던 벽을 넘었다

![조건부 D 판정](figures/cond_disc_verdict.png)

| model, cmd 2.5\~3.5 | RSI 출발 gallop | **정지 출발 gallop** | 저속→고속 전환 |
|---|---|---|---|
| cmdchg4s (무조건부) @20k | 0.0% | **0.0%** | **0/135** |
| **condmatch @20k** | 30.3% | **28.4%** | **6/90 (6.7%)** |
| **cond+drail @10k** | 13.8% | **10.6%** | **19/162 (11.7%)** |

세 지표가 **동시에** 처음으로 움직였다:

1. **속도 선택성** — condmatch 는 0.0%(cmd 1.0) → 12.5%(1.5\~2.5) → **30.3%**(2.5\~3.5) 로
   단조 증가한다. 참조 gallop 속도(2.69 m/s) 위에서 가장 높다.
2. **초기 자세 의존성이 깨졌다** — RSI 30.3% vs 정지 28.4%. 이 프로젝트에서 정지 출발 gallop 이
   0.0% 가 아닌 것은 **이번이 처음**이다(이전 모든 팔에서 0.0\~0.3%).
3. **위쪽 전환이 0 이 아니다** — 6.7% / 11.7%. 앞선 세 팔은 합쳐 1/671 (0.15%) 이었다.

### 걸음 구성 (condmatch@20k, RSI 출발)

| cmd vx | pace | trot | gallop | other | stand |
|---|---|---|---|---|---|
| 0.0\~0.5 | 0.0 | 0.0 | 12.9 | 30.6 | 56.5 |
| 0.5\~1.5 | **91.2** | 4.8 | 0.0 | 2.0 | 2.0 |
| 1.5\~2.5 | 85.9 | 0.5 | **12.5** | 1.0 | 0.2 |
| 2.5\~3.5 | 69.7 | 0.0 | **30.3** | 0.0 | 0.0 |

저속은 pace, 고속으로 갈수록 gallop 이 섞인다 — 참조의 사다리와 같은 방향이다. 다만 고속에서도
pace 가 69.7% 로 여전히 다수라 **사다리가 완성된 것은 아니다.**

## ★ iteration 을 맞춘 3자 비교 (전부 model_16000)

![16k 맞춤 비교](figures/cond_disc_matched_16k.png)

위 표는 팔마다 iteration 이 달라 그대로 비교할 수 없었다. 세 팔 모두 있는 16k 로 맞춰 다시 쟀다:

| model_16000, cmd 2.5\~3.5 | RSI 출발 | 정지 출발 | 저속→고속 전환 |
|---|---|---|---|
| uncond (cmdchg4s) | **0.0%** | **0.0%** | **0/127** |
| cond (mlp) | 2.2% | 3.2% | 1/97 (1.0%) |
| **cond + drail** | **12.2%** | **12.0%** | **20/147 (13.6%)** |

조건화만 다른 같은 지점에서 **무조건부는 완전히 0**, 조건부 둘은 0 이 아니다. 순서도 분명하다 —
drail 이 mlp 조건부보다 앞선다.

★ 앞 절에서 "condmatch 30.3%(20k) vs drail 13.8%(10k)" 를 나란히 적은 것은 오해를 부를 수 있었다.
같은 16k 에서 보면 **drail 12.2% > cond 2.2%** 로 순서가 반대다. condmatch 의 30.3% 는 16k→20k
사이의 급성장이지 drail 보다 강해서가 아니다.

### cond+drail@16k 걸음 구성 — `other` 가 크다

| cmd vx | pace | trot | gallop | other |
|---|---|---|---|---|
| 0.0\~0.5 | 2.9 | 6.9 | 13.7 | **73.7** |
| 0.5\~1.5 | 23.3 | 7.0 | 1.9 | **67.9** |
| 1.5\~2.5 | 88.3 | 0.0 | 2.2 | 8.8 |
| 2.5\~3.5 | 49.5 | 0.0 | **12.2** | **32.7** |

고속 칸의 32.7% 가 `other` 다 — trot 도 pace 도 gallop 도 아닌 중간 위상이다. gallop 이 늘어난
것과 별개로 **걸음이 아직 정돈되지 않았다**는 뜻이므로, 이 수치를 "gallop 을 배웠다" 로만 읽으면
안 된다. 학습이 더 간 뒤 다시 봐야 한다.

## 유보 사항

- **condmatch 는 10k 에 아직 0.0%, 16k 에 2.2%, 20k 에 30.3%** 로 급격히 변한다. 측정 지점에
  따라 인상이 크게 달라지므로 한 점으로 판단하면 안 된다.
- baseline 의 gallop 발현도 10k\~20k 구간이었다. 다만 **무조건부 팔은 같은 16k 에서 0.0%** 이므로,
  "그 시점이면 원래 생긴다" 로는 설명되지 않는다.
- 조사는 다른 학습 4 개가 GPU 를 쓰는 중이라 **2048 env** 로 쟀다.
- drail 팔의 `other` 가 고속에서 19.6% 로 높다. trot/pace 중간 위상이 이쪽으로 떨어지므로,
  걸음이 정돈되지 않은 상태일 수 있다. 학습이 더 진행된 뒤 다시 봐야 한다.

## 도구 수정 3건 — 없었으면 이 팔들을 아예 못 쟀다

1. **`amp_cond_mode` 가 조사 스크립트의 cfg 복원 목록에 없었다.** 판별자 입력이 592 인데 590 으로
   env 를 지어 `size mismatch` 로 죽었다.
2. **run 의 `agent.yaml` 을 복원하지 않았다.** `disc_arch: drail` 이 반영되지 않아 mlp 판별자에
   drail 가중치를 넣으려다 죽었다. 이제 `algorithm`/`policy`/`amp`/`estimator` 를 얹는다.
   단 `amp_observation_space` 등 **env 가 런타임에 계산하는 값은 제외**한다 — 저장된 590 을
   그대로 넣으면 조건부 팔이 다시 깨진다.
3. **`agent_cfg.amp` 는 객체가 아니라 dict** 라 `hasattr` 기반 덮어쓰기가 조용히 건너뛰었다.
   dict 와 configclass 를 모두 다루도록 고쳤다.

★ 1·2 는 즉시 죽어서 발견됐지만, 이 프로젝트에서 같은 부류의 **조용한** 누락
(`motion_weight_mode`)은 결론을 뒤집을 뻔했다. cfg 복원 목록은 새 cfg 키가 생길 때마다
갱신해야 한다.

## 산출물

- `figures/cond_disc_verdict.png` — 4 패널(선택성 · 초기자세 대조 · 전환 · 걸음 구성)
- `metrics/` — 조사 원자료
- `videos/` — 추적 카메라 클립

## 영상 (추적 카메라, condmatch model_20000, 고정 3.0 m/s)

| 파일 | 출발 | 실제 보행 |
|---|---|---|
| `videos/condmatch20k_const30_standing_start_GALLOP.mp4` | **정지 기립** | **gallop** 2.31 Hz |
| `videos/condmatch20k_const30_reference_start_PACE.mp4` | 참조 프레임 | pace 2.25 Hz |
| `videos/condmatch20k_ramp_standing_start_PACE.mp4` | 정지, 0→3.2 램프 | 전 구간 pace |

★ 첫 줄이 이 조사 통틀어 **처음 보는 장면**이다 — 정지에서 출발한 로봇이 고속 명령에서
gallop 한다. 이전 모든 팔에서 이 조건의 gallop 비율은 0.0% 였다.

⚠️ 둘째 줄이 pace 인 것은 모순이 아니다. 그 조건의 gallop 비율이 30.3% 라 pace 가 더 흔하다.
셋째 줄(램프)이 pace 인 것도 예상대로다 — 램프는 명령을 0 에서 시작해 초반을 저속으로 보내므로
gallop attractor 에 못 들어간다(앞선 문서에서 확인).
세 클립 모두 `gait_classify.py` 로 분류해 캡션을 검증했다.

## 동일 시점 추적 (2026-09-03 09:45, 2048 env · 정지/RSI 출발) — 격차가 벌어진다

앞 절의 유보("baseline 도 10k~20k 에 gallop 이 생기므로 시점 효과일 수 있다")를 **더 늦은
체크포인트**로 가른다. baseline 은 25k·30k, condmatch 는 23k(측정 시점의 최신)를 쟀다.

![동일 시점 추적](figures/cond_disc_same_iter.png)

| cmd 2.5~3.5 gallop | 정지 출발 | RSI 출발 | 저속 cmd 0.5~1.5 gallop (RSI) |
|---|---|---|---|
| baseline @20k (앞 절) | 0.0% | 0.0% | — |
| baseline @25k | **0.0%** | 미측정 | — |
| baseline @30k | **4.3%** | 20.9% | **40.0%** |
| condmatch @20k (앞 절) | 28.4% | 30.3% | 0.0% |
| condmatch @23k | **55.4%** | 44.4% | 0.0% |

1. **시점 효과가 아니다.** baseline 은 30k 까지 가도 정지 출발 gallop 이 4.3% 다. condmatch 는
   20k → 23k 사이에 28.4 → 55.4% 로 계속 오른다.
2. **baseline @30k 의 RSI gallop 20.9% 는 선택성이 없다** — cmd 0.5~1.5 에서 40.0%, 1.5~2.5 에서
   29.5%, 2.5~3.5 에서 20.9% 로 **속도가 오를수록 줄어든다.** 초기 프레임이 gallop 이면 명령과
   무관하게 그 걸음을 이어가는 `../gait_attractor_pace_vs_gallop/` 의 그림이다. condmatch 는 정지
   출발에서 0.0 → 20.4 → 55.4% 로 단조 증가하고 저속에서는 0 이다.
3. **대가: 정지 출발 저·중속에서 `stand` 가 늘었다.** condmatch @23k 정지 출발은 cmd 0.5~1.5 에서
   22.0%, 1.5~2.5 에서 9.5% 가 2 s 창 끝까지 서 있다(baseline @30k 0.6% / 0.0%, condmatch RSI 출발
   2.6% / 0.8%). 정지→보행 진입 실패(08-26 저속 개시 히스테리시스 판정)가 조건부 팔에서 더
   자주 나온다. 실제 평균 vx 는 0.82 / 1.92 로 baseline(0.93 / 2.02) 보다 약간 낮다.

★ 판별자 게이트(`check_cond_disc_leak.py`, shuffled ≈ clip_mean)가 "D 는 조건을 안 본다" 고 읽은
것과 이 결과는 여전히 상충한다. D(expert) 의 배치 평균은 조건 열의 기여를 잴 수 있는 통계가
아닐 수 있다 — 조건이 바꾸는 것은 **정책 쪽** 보상 지형(어느 kinematics 가 어느 명령에서 점수를
받는가)이지 expert 배치의 평균 점수가 아니다. 게이트를 정책 표본으로 다시 재야 한다.

원자료: `metrics/same_iter_surveys.md`, `metrics/same_iter/*.npz`. 조사 도구는 앞 절과 같다
(`_workspace/leg/gait_survey_multienv.py`, `--n_envs 2048 --dur_s 8`, 정지 출발은 `--all_stand`).

## run B'(cond+drail) resume 후 20k — gallop 12 → 39%, 상향 전환 13.6 → 20.2% (2026-09-03 16:51, 2048 env, GPU2)

16.7k 에서 σ 발산만 보고 중단했던 `condmatch_drail` 을 `model_16700` 에서 resume 해
(`2026-09-03_12-51-22_condmatch_drail_resume16k7_…`, 설정 동일) `model_20000` 을 쟀다. 학습 지표는
σ 4.2 · amp 8.4~9.0 으로 중단 전 추세 그대로인데, 거동은 계속 좋아진다.

![run B' resume 20k](figures/cond_disc_bresume20k.png)

| cmd 2.5~3.5 gallop | RSI 출발 | 정지 출발 | 저속→고속 전환 | 고속→저속 전환 | 고속 `other` |
|---|---|---|---|---|---|
| cond+drail @16k (앞 절) | 12.2% | 12.0% | 20/147 (13.6%) | — | 32.7% |
| **cond+drail @20k (resume)** | **39.5%** | **39.2%** | **24/119 (20.2%)** | 48/51 (94.1%) | 33.7% |
| cond(mlp) @23k (앞 절) | 44.4% | 55.4% | — | — | 0.2% |
| baseline @30k (앞 절) | 20.9% | 4.3% | — | — | 0.0% |

1. **σ 4.2 인 채로 거동이 좋아졌다.** 정지 출발 gallop 이 3.3 배, 상향 전환이 1.5 배 늘었고, 낙상 0 ·
   `stand` 0 이다. 중단 근거였던 σ 발산은 이 arm 에서 거동 실패를 뜻하지 않는다 — "학습 지표로
   판정하지 말 것" 이 다시 확인됐다.
2. **상향 전환은 세 arm 중 최고다**(20.2% vs cond-mlp 20k 6.7% · uncond 0%). 하향 전환 94.1% 는
   저속 명령에서 gallop 을 버린다는 뜻이므로 선택성도 있다.
3. **걸음은 여전히 정돈되지 않았다.** 고속 `other` 33.7% 는 16k 와 같고, 저속(cmd 0.5~1.5)은
   `other` 69% 로 pace 24% 보다 많다 — 추종(실제 vx 0.98) 은 되는데 걸음 위상이 pace/trot 어느
   쪽도 아니다. cond-mlp 23k 는 같은 구간 `other` 가 0.2~9% 다. **gallop 수치만 보면 안 되는 이유.**
4. cond-mlp 23k(55.4%) 와의 우열은 아직 못 가린다 — iteration 이 다르고 `other` 축이 반대다.
   25k·30k 에서 같은 표를 다시 채운다.

원자료: `metrics/bresume20k_surveys.md`, `metrics/bresume20k/{rsi,stand,tr}.npz`.

## 최종 체크포인트 3자 비교 — 조건화 고유 효과는 "명령 전환 추종" (2026-09-04 09:09, 4096 env, GPU3)

기준선과 cond-mlp 는 50k 완주(`model_49999`), cond+drail 은 resume run 의 `model_45000`(σ 13.75)을
같은 도구로 쟀다. 세 조사 모두 낙상 0.

![최종 3자 비교](figures/cond_disc_final50k.png)

| cmd 2.5~3.5 | RSI 출발 gallop | 정지 출발 gallop | 저속(0.5~1.5) gallop | 저속→고속 전환 | 고속→저속 전환 | 고속 `other` | 저속 `other` |
|---|---|---|---|---|---|---|---|
| uncond @50k | 46.4% | 37.9% | 10.6% | **6.2%** (10/160, 잠김 94.6%) | 45.3% | 0.0% | 19.2% |
| **cond-mlp @50k** | 74.1% | **72.3%** | 0.3% | **43.8%** (84/192) | 64.4% | 0.3% | 9.7% |
| cond+drail @45k | 74.5% | **74.7%** | 1.6% | 33.2% (81/244) | 88.3% | 8.9% | **60.5%** |

1. **기준선도 50k 에는 gallop 이 생긴다**(정지 출발 30k 4.3 → 50k 37.9%). 그러므로 "gallop 비율"
   은 더 이상 조건화의 증거가 아니다. 갈리는 축은 둘이다.
2. **명령 전환 추종** — 에피소드 도중 저속→고속으로 바뀐 표본에서 pace→gallop 으로 따라 바뀐 비율이
   uncond 6.2% vs cond-mlp **43.8%** vs cond+drail 33.2%. 기준선은 94.6% 가 처음 걸음에 잠긴다.
   이것이 조건부 D 가 만든 고유 효과이며, 이 조사의 원래 목표("걸음이 명령을 따라가는가")의 답이다.
3. **속도 선택성** — uncond 는 cmd 0.5~1.5 에서도 gallop 10.6% 로 무차별이고, cond 두 arm 은 0.3 /
   1.6% 다. 조건부 arm 만 "저속 pace / 고속 gallop" 사다리를 탄다.
4. **cond-mlp 가 우선 후보다.** gallop 은 cond+drail 과 같은데(72 vs 75) 걸음이 정돈돼 있고(`other`
   0.3 / 9.7%), 23k 에 22% 였던 정지 출발 저속 `stand` 도 5.3% 로 내려왔다.
5. **cond+drail 은 σ 13.75 에도 거동이 살아 있다** — 낙상 0, 고속 `other` 는 20k 33.7 → 8.9% 로
   정돈됐다. 그러나 **저속 `other` 60.5%** 는 20k(69%)부터 그대로다. 저속에서 pace/trot 어느 위상도
   아닌 걸음을 유지하는 것이 DRAIL 팔의 특성이며, 하향 전환 88.3% 가 높은 것도 "gallop 을 버리고
   `other` 로 간다" 여서 장점으로 읽으면 안 된다.
6. 학습 지표는 이 격차를 못 봤다 — 최종 amp 보상은 cond-mlp 28.9 vs uncond 26.3 으로 근소한데
   전환율은 7 배 차이다.

원자료: `metrics/final50k_surveys.md`, `metrics/final50k/{cmdchg4s50k,condmatch50k,bresume45k}_{rsi,stand,tr}.npz`.

### 영상 (2026-09-04 09:32~09:42, 추적 카메라, seed 0, 정지 출발, 각 1 롤아웃)

영상은 측정이 아니다(카메라 ON 이 낙상률을 바꾼다는 09-01 판정). 아래 캡션은 `gait_classify.py` 로
같은 롤아웃의 npz 를 분류한 결과이며, 조사 표의 비율과 다를 수 있는 **개체 하나**다. 6 편 모두 완주.

| 파일 | 명령 | 실제 보행 | 비고 |
|---|---|---|---|
| `videos/uncond50k_const30_stand/speed_ramp-step-0.mp4` | 3.0 고정 8 s | **pace** 2.30 Hz | vx 2.86 |
| `videos/condmlp50k_const30_stand/speed_ramp-step-0.mp4` | 3.0 고정 8 s | **gallop** 2.26 Hz | vx 3.02 |
| `videos/conddrail45k_const30_stand/speed_ramp-step-0.mp4` | 3.0 고정 8 s | pace 2.40 Hz | 조사는 74.7% gallop 인데 이 개체는 pace |
| `videos/uncond50k_steps_stand/speed_ramp-step-0.mp4` | 0.5→3.0 계단(0.5 씩, 3 s) | 전 구간 pace | 1.79→2.26 Hz |
| `videos/condmlp50k_steps_stand/speed_ramp-step-0.mp4` | 0.5→3.0 계단 | 2.0 이상 pace, 1.5 이하 3 사이클 미만 | 이 개체는 전환 안 함(조사 43.8%) |
| `videos/conddrail45k_steps_stand/speed_ramp-step-0.mp4` | 0.5→3.0 계단 | 2.5 까지 pace → **3.0 에서 gallop** | 저속→고속 전환의 실례 |

`speed_ramp_record_rma.py` 는 run 의 `agent.yaml` 을 복원하지 않아 DRAIL 체크포인트를 못 읽었다
(`disc_arch` 미반영 → state_dict mismatch). `gait_survey_multienv.py` 와 같은 overlay 를 넣어 고쳤다.

## 심층 비교 — cond-mlp vs cond+drail (2026-09-04 10:34~10:45)

두 조건부 arm 을 (a) 긴 속도 램프에서 추종·토크·관절속도·걸음으로, (b) 학습 로그에서 판별기·손실·보상
곡선으로 다시 봤다. 원자료와 표는 `metrics/long_ramp_surveys.md`, `metrics/train_curves.md`.

### (a) 긴 램프 0→3.5→0 (hold 6 s / ramp 2 s, 정지 출발, seed 0/1/2, `--no_video`, GPU1/2)

`speed_ramp_record_rma.py --vx_max 3.5 --hold_s 6 --ramp_s 2 --force_stand`, 15 단계 120 s. 명령은 0.5 씩
2 s 선형 상승 — gait survey(명령 도약) 와 **명령 형태가 다르다**. 6 롤아웃 전부 낙상 없음(`ramp_fall_probe`
는 heading_hold 없이 재서 yaw 드리프트를 낙상으로 오판하므로 관절 접힘 판정으로 대체).

| 항목 (시드 평균) | cond-mlp `model_49999` | cond+drail `model_45000` |
|---|---|---|
| 달성률 vx/cmd, 상승 1.5/2.0/2.5/3.0/**3.5** | 0.83/0.94/0.92/0.92/**0.90** | 0.98/0.97/0.98/1.02/**0.96** |
| 저속 개시(cmd 0.5~1.0) | 시드 1/3 이 vx≈0 (진입 실패, 1.5 부터 회복) | 3/3 개시 (0.83~0.94) |
| 걸음, 상승 2.0~2.5 / 3.0 / 3.5 (1.5 는 mlp other 2/3) | pace / **pace** / pace (3/3) | pace / **gallop** / gallop (3/3) |
| 걸음, 하강 3.0 / 2.5 / 2.0 / 1.5 | pace 전부 | gallop 3/3 / 3/3 / 2/3 / pace — 히스테리시스 |
| 토크 포화율(전 관절, \|τ\|>0.95 limit) | ≤1.0% (속도 따라 증가) | **2.7~3.8%, cmd 0 부터** |
| 관절속도 peak, cmd 0 / 3.0 (limit 30 rad/s) | 0.6 / 12.3 | **15.6 / 30.4** (HL 시드 최대 31.7, 초과) |
| 3.5(OOD) 에서 3.0 대비 | 토크 peak +23~43% (여력 사용) | 토크·속도 거의 평평 (3.0 에서 이미 천장) |

1. **cond-mlp 은 점진 램프에서 gallop 이 한 번도 안 열린다**(2.0~3.5 상승·하강 3/3 pace, 1.5 상승은 other 2/3). 조사에서
   정지 출발 cmd 2.5~3.5 gallop 72.3%·상향 전환 43.8% 였던 것과 어긋나는데, 조건이 다르다 — 조사는
   명령이 도약(RSI 프레임, 4 s 재샘플 점프)하고 램프는 0.5 씩 2 s 에 걸쳐 오른다. 영상 절의 "3.0 고정
   → gallop / 0.5 계단 → pace" 도 같은 방향이다. 즉 cond-mlp 의 전환은 **명령 점프 크기에 의존**한다는
   정황(n=3 시드, 확정 아님). cond+drail 은 램프에서도 3.0 에서 3/3 gallop 으로 바뀌고 하강 시
   2.0~2.5 까지 유지한다.
2. **추종은 cond+drail 이 전 구간 더 정확**(달성률 0.94~1.04 vs 0.83~0.94) 하고 저속 개시도 안정적이다.
   두 정책 다 3.5 에서 실속하지 않는다.
3. **그 대가는 액추에이터 부하다.** cond+drail 은 정지 명령에서도 다리 관절속도 RMS 2.85 rad/s(시드 2.6~3.0; cond-mlp 0.08)·peak 15.6,
   토크 포화 3%대로 다리를 계속 흔들고, 3.0 이상에서 관절속도가 `velocity_limit_sim` 30 rad/s 에 닿거나
   넘는다. 램프는 `act_inference_priv`(결정론적 mean 액션) 경로라 σ 표본 잡음이 아니라 정책 mean 자체의 거동이다. cond-mlp 은 같은 구간
   peak 12~13 으로 절반 이하. `torque_penalty_w=0` 이라 이를 누르는 항이 학습에 없다.

그림: `figures/long_ramp_tracking.png`, `long_ramp_torque.png`, `long_ramp_jointvel.png`, `long_ramp_gait.png`.
플롯 스크립트 `_workspace/leg/plot_long_ramp_compare.py`.

### (b) 학습 곡선 — 판별기·손실·보상 (tfevents, baseline / A' 50k 완료, B' 원본+resume 48,475 시점)

세 run 공통 `bce` 손실·보상, `reward_coef 2.0`, `task_reward_lerp 0.5`, disc lr 2.5e-4. drail 만 grad
penalty·logit reg 가 코드 경로상 **미적용**(`ppo_amp.py` `use_regularizers`). `disc_*_output` 은
sigmoid 배치 평균(expert→1, policy→0 이 D 의 목표), 보상은 `-log(1-D)·2.0` 을 초당 합산.

| 500-iter 후행 평균 | baseline (uncond) | A' cond-mlp | B' cond+drail |
|---|---|---|---|
| margin D(e)−D(p): 정점 → 최종 | 0.76 @1.1k → 0.51 | 0.80 @0.8k → 0.50 | 0.90 @0.6k → **0.88** |
| D(policy) 5k → 최종 | 0.190 → 0.244 | 0.186 → 0.249 | 0.061 → **0.060** |
| amp_reward 5k → 최종 [/s] | 21.5 → 28.3 | 20.9 → **29.0** | 8.9 → 8.8 |
| lin_vel_reward 최종 [/s] | **45.0** | 43.7 (−3.0%) | 44.9 |
| 수렴 iter (margin / amp_reward / σ) | 39.8k / 40.5k / 43.8k | 36.2k / 38.0k / 38.1k | 퇴화(32) / 17.2k / 발산 |
| σ 5k / 16k / 최종 | 0.39 / 0.41 / 0.31 | 0.38 / 0.37 / 0.28 | **1.50 / 3.41 / 14.4** |
| grad penalty 최종 | 0.12 | 0.12 | 0 (미적용) |

1. **mlp 계열(baseline, A')은 정상적인 적대 게임이다.** D 는 수십 iter 에 학습돼 1k 부근 margin 정점을
   찍고, 이후 정책이 서서히 속여 50k 까지 단조 하강(0.76→0.51). D(policy) 0.19→0.25, amp_reward
   21→28~29 로 style 이 계속 개선되며 **36k~44k 에 가서야** ±5% 밴드에 든다 — 50k 예산의 2/3 가 실제
   개선 구간이다. A' 는 baseline 보다 style 이 근소하게 낫고(29.0 vs 28.3, 3.7k~5.7k 이른 수렴) task 는
   3% 낮다. 조건화가 D 를 망가뜨리지 않았다.
2. **B' 의 적대 게임은 5k 이후 정지해 있다.** D(expert) 0.94 / D(policy) 0.060 / margin 0.88 이 43k
   동안 ±0.001 로 고정, amp_reward 8.8 평평. drail logit(`L_pi−L_M`)의 sigmoid 는 mlp 와 척도가 달라
   절대값(0.06 vs 0.25)으로 모션 품질을 비교할 수는 없지만, "정책이 D 를 밀어내는 흔적이 없다" 는
   사실이다. 총 보상에서 style 이 차지하는 몫도 A' 의 1/3 이하(8.8 vs 44.9 의 lerp).
3. **B' σ 발산은 resume 과 무관하다.** 원본 run 500 iter 부근부터 단조 상승 — 5k 에 1.50(A' 의 3.9배),
   16k 3.41, seam 앞뒤 3.582→3.604 연속. resume 재워밍은 amp_reward 한 점(16,700~16,750, 5.3)뿐이다.
   σ 가 4배 커지는 동안 margin 은 0.2% 만 변하고 B' 의 1차 차분 상관은 전부 |r|<0.03(A' 는 최대 0.05) — "D 가 강해서 σ 가
   터졌다" 는 서술은 이 로그로 뒷받침되지 않는다. 다만 drail 만 GP/logit reg 가 꺼져 있고 drail 만
   발산한 것은 사실(n=1 정황). **실무 함의: DRAIL 계열 재시도는 5k 시점 σ(≈1.5 vs 0.38)만으로 판정 가능.**
4. task 보상은 세 run 이 같다(43.7~45.0, ep_len 995+). B' 의 mean_reward 532 가 낮은 것은 style 항 때문.

그림: `figures/train_disc_outputs.png`, `train_disc_losses.png`, `train_policy_stats.png`,
`train_reward_balance.png`. 추출·플롯 `_workspace/leg/tb_extract_cond_disc.py`,
`plot_train_curves_cond_disc.py`, 캐시 `metrics/train_curves/*.npz`.

### 판정 갱신

- **cond-mlp 우선 후보는 유지**하되 단서 둘: 점진 램프에서는 gallop 전환이 안 열림(명령 점프 의존 정황),
  저속 개시 실패 1/3 시드. 학습 곡선은 건강하다(정상 적대 게임, 수렴 38k).
- **cond+drail 은 배포 후보가 아니다.** 추종·전환·개시는 우위지만 관절속도 한계 도달·정지 시 떨림·σ 발산·
  D 정지가 한 묶음이다. DRAIL 을 살리려면 GP/entropy 상한(σ clamp) 같은 정규화 실험이 먼저다.
- 다음: cond-mlp 명령 점프 크기 스윕(0.5→3.0 직접 vs 0.5 단계, 헤드리스 n≥8) 으로 1 을 확정;
  `torque_penalty_w`/관절속도 페널티 도입 검토; run C 는 5k σ 기준으로 조기 판정.

## D 출력과 영상 인상의 괴리 — 판별기가 보는 것 vs 눈이 보는 것 (2026-09-04 11:10~14:35)

질문: DRAIL 판별기는 expert/policy 를 거의 완벽히 가르는데(학습 로그 D(policy) 0.06, amp 보상 8.8)
영상에서는 cond+drail 이 더 잘 따라하는 듯 보인다. 세 가지 측정으로 갈랐다. 원자료·표는
`metrics/clip_probe_surveys.md`(참조 속도-명령 프로브 + 교차 판별 + 확률적 행동 프로브),
`metrics/gap_spectral.md`(주파수 분해), `metrics/gap_ampobs.md`(amp 관측 공간 분리도).

### 측정 1 — 참조 클립 속도를 그대로 명령으로 (`_workspace/leg/amp_obs_clip_probe.py`, 256 env, 14 클립 × rsi/stand)

클립의 body-frame (vx, yaw) 를 매 스텝 명령으로 주고(학습 범위로 clip, vy 는 학습 범위가 0 이라 버림)
expert 와 policy 의 amp 관측을 같은 조건에서 5000 개씩 모아 **두 판별기 모두**로 채점했다. expert 의
조건 열은 클립 평균이 아니라 policy 와 같은 프레임별 명령값으로 맞췄다(안 맞추면 조건 열에서 갈린다).

| D(policy) 평균, rsi 출발 | mlp(A' 의 D) 로 채점 | drail(B' 의 D) 로 채점 |
|---|---|---|
| A' cond-mlp, 고속 클립 | **0.295** | 0.070 |
| B' cond+drail, 고속 클립 | 0.033 | **0.113** |
| A' cond-mlp, 저속 클립 | 0.418 | 0.191 |
| B' cond+drail, 저속 클립 | 0.068 | 0.193 |

1. **D 로는 두 정책을 서열화할 수 없다.** mlp 는 자기 정책 A' 를 8.9배 높게, drail 은 자기 정책 B' 를
   1.6배 높게 준다. 순위가 판별기 구조를 따라 뒤집힌다. 두 판별기 다 expert/policy AUROC ≈1.0 으로
   포화 상태라 "누가 더 쉬운가" 를 잴 여지도 없다.
2. **판별기 무관 지표(명령 추종)에서는 B' 가 고속 우위.** |Δvx| 정지 출발 고속 0.330 vs 0.671 m/s,
   rsi 고속 0.301 vs 0.361. 저속은 A' 가 근소 우위(0.129 vs 0.147). 가장 느린 클립(0.11 m/s)은 B' 가
   안 움직이고(달성률 0.01) A' 는 과속(1.56) — 저속 개시는 둘 다 부정확하고 방향이 반대다.
3. **학습 로그의 0.06 은 행동 잡음을 잰 값이다.** 같은 클립·같은 DRAIL D 로 B' 를 세 경로로 굴리면
   결정론 mean 0.092/0.156/0.185(run1/trot0/walk1) → 학습 입력 경로(priv latent+estimator) mean
   0.092/0.153/0.191 → 학습 경로 **표본 추출** 0.062/0.073/0.072. 경로 분기는 무죄, 표본 추출만 D 를
   내린다. B' σ 12.98 은 A' 0.273 의 47.6배이고 표본 행동의 24~27% 가 ±4 clip 에 붙는다. 추종은 잡음에
   거의 안 무너진다(|Δvx| 0.298→0.306).

### 측정 2 — 주파수 분해 (긴 램프 npz + 참조 pkl, `gap_spectral.py`)

| cmd 3.0 관절속도 파워 비율 | 0~3 Hz | 3~8 Hz | **8~25 Hz** | 8~25 Hz 절대 [(rad/s)²] |
|---|---|---|---|---|
| 참조 (leg_run0/1) | 0.37 | 0.47 | 0.16 | 49 |
| A' cond-mlp | 0.33 | 0.64 | **0.04** | 7 |
| B' cond+drail | 0.20 | 0.42 | **0.39** | 218 |

B' 는 모든 속도에서 8~25 Hz 가 총 파워의 39~76%(A' 2~4%, 참조 1~16%), 정지 구간에서는 81%(RMS 11 rad/s,
최대 27 rad/s). 참조와의 관절속도 Wasserstein 거리는 원 신호에서 B' 가 A' 의 **1.9~4.7배**인데 5 Hz
저역통과 후 **0.97~1.48배**로 붕괴 — 초과 거리가 전부 5 Hz 위에 있다. 관절 **위치** 거리는 반대로
B' 가 A' 보다 1.7~2.8배 가깝다. (프레임 단위 C2ST 는 같은 정책 시드끼리도 0.67~0.99 로 갈려 음성
대조 실패 → 기각, 분류기 없는 거리로 대체.)

### 측정 3 — amp 관측 공간 (측정 1 의 npz, `gap_analysis_ampobs.py`; 음성 대조 expert/expert C2ST = 0.500 통과)

| 필드별 W1 (expert vs policy, 비미러 21 파일 평균) | A' | B' | B'/A' |
|---|---|---|---|
| dof_pos | 0.124 | 0.062 | 0.50 |
| **dof_vel** | 0.411 | **1.424** | **3.47** |
| root_h | 0.022 | 0.008 | 0.36 |
| lin_vel | 0.129 | 0.093 | 0.72 |
| ang_vel | 0.182 | 0.232 | 1.28 |
| foot_pos | 0.029 | 0.020 | 0.70 |
| rot_tan_norm | 0.025 | 0.011 | 0.43 |

- **B' 는 7개 필드 중 5개에서 참조에 더 가깝고, `dof_vel` 하나에서 3.5배 멀다.** 전체 특징 C2ST 는
  짝지은 24 파일 중 23 에서 B' 가 A' 보다 **덜** 갈리고(0.59~0.96 vs 0.88~1.00), `dof_vel` 열만
  쓰면 12/12 에서 B' 가 더 갈린다. 결정론 mean 롤아웃(`_trainmean`)에서도 유지(run1 0.914 vs 0.814)
  — 떨림은 σ 표본이 아니라 **학습된 mean 행동 자체**에 있다. 불일치 채널이 확정됐다.
- A' 의 mlp D 는 B' 표본의 83~89%(달리기·트롯)를 0.05 아래로 내린다 — 자기 학습 중 본 적 없는 떨림을
  가혹하게 벌한다. B' 의 drail D 는 A' 와 B' 를 비슷하게(때로 A' 를 더 낮게) 채점한다.

### 판정

- 괴리는 실제이며 세 조각으로 풀린다. (1) 학습 로그의 D 0.06·amp 8.8 은 **σ 13 의 확률적 행동**을
  잰 값이고 결정론 mean 은 0.09~0.26 이다. (2) D 절대값은 판별기마다 자가 편향·척도가 달라 정책 간
  비교가 안 된다. (3) 눈이 보는 채널(관절 위치·발 위치·몸통 자세·높이)에서는 B' 가 실제로 참조에
  더 가깝고, D 가 보는 채널 중 `dof_vel` 의 5 Hz 위 성분에서만 B' 가 멀다.
- "B' 의 떨림이 D 에게 더 쉬운 신호" 라는 평문 가설은 **기각**(전체 특징에서는 B' 가 덜 갈림, 23/24).
  살아남는 주장: **B' 의 불일치는 관절속도 고주파 한 채널에 국한된다.**
- 떨림의 원인을 DRAIL 정규화 부재로 귀속할 수는 없다(판별기 종류와 iter 가 함께 다름). σ 13 학습
  롤아웃과의 관계는 정황뿐.
- 다음 후보: (a) 배포 시 행동 저역통과(5 Hz) 후 `dof_vel` 거리·토크·관절속도 재측정 — 떨림이 mean 에
  박힌 것인지 필터로 지워지는지; (b) B' 계열의 σ 상한(entropy clamp) 재학습 후 같은 프로브;
  (c) 판별기 입력에서 `dof_vel` 저역화 또는 가중 — style 보상이 위상·자세를 보게 하는 변형.

도구: `_workspace/leg/amp_obs_clip_probe.py`(+`_table.py`, `_stoch_table.py`), `gap_spectral.py`,
`gap_spectral_dist.py`, `gap_analysis_ampobs.py`. 그림 `figures/gap_{psd,w1_lowpass,c2st_lowpass,c2st_ampobs,feature_distance,d_hist}.png`.
npz `metrics/clip_probe/{condmlp50k,conddrail45k}/*.npz` (68 개).

## 판별기는 무엇을 보고 무엇을 보상하는가 — DRAIL 판별기의 역할 (2026-09-04 14:40~15:45)

"DRAIL 판별기가 어떤 역할을 하는지 납득이 안 된다" 는 질문에 세 측정으로 답했다. 원자료·표는
`metrics/gait_reward_surveys.md`(걸음별 style 보상, 4096 env), `metrics/disc_role_surveys.md`
(채널 민감도·절제·저역통과 재채점·조건 열 치환, `_workspace/leg/disc_offline.py` 로 Isaac 없이 판별기 로드).

### 측정 1 — 판별기가 보는 채널 (saliency `|∂logit/∂x|·σ`, 절제)

| 표준화 기여 비율, policy 표본 | dof_pos | **dof_vel** | root_h | lin_vel | ang_vel | foot_pos | rot_tan_norm | 히스토리 가중 |
|---|---|---|---|---|---|---|---|---|
| mlp D (A') | 1~3% | **94~96%** | 0 | 1% | 2% | 0% | 0% | 10 슬롯 균등 |
| DRAIL D (B') | 23~34% | 32~47% | 1~3% | 4~5% | 5~7% | 10~16% | 7~10% | 최신 프레임 12~19% 로 기움 |

절제(policy 의 한 필드를 같은 index 의 expert 값으로 치환했을 때 ΔD): mlp D 는 `dof_vel` 치환 하나로
A' +0.07~0.42(leg_walk 0.07, 나머지 0.30~0.42), B' +0.46~0.74 가 열리고 다른 필드는 0. DRAIL D 는 **A' 를 두고는 `dof_pos`**(+0.21~0.46,
`dof_vel` 0), **B' 를 두고는 `dof_vel`**(+0.24~0.47) 을 짚는다.

1. **mlp 판별기는 사실상 관절속도 판별기다.** 기여의 95% 가 `dof_vel` 이고 위치·발·자세는 거의 안
   본다. A' 가 관절속도는 깨끗한데 관절 위치·발 위치·높이에서 B' 보다 참조에서 먼 것(§괴리 측정 3)과
   맞물린다 — mlp D 가 밀지 않은 채널이다.
2. **DRAIL 판별기는 위치·속도·발 위치를 함께 보는 넓은 스펙트럼의 style 비평가다.** B' 가 5/7 필드에서
   참조에 가까운 것은 이 D 가 그 채널들을 실제로 보기 때문이라는 정황과 맞는다.

### 측정 2 — 떨림을 지우면 DRAIL 판정이 어디까지 오르는가 (관절속도 5 Hz 저역통과 후 재채점)

| DRAIL D(policy), B' | orig | 3 Hz | **5 Hz** | 8 Hz | ang_vel 만 5 Hz (대조) | expert |
|---|---|---|---|---|---|---|
| leg_run1 | 0.090 | 0.568 | **0.507** | 0.350 | 0.100 | 0.951 |
| leg_trot0 | 0.153 | 0.678 | **0.666** | 0.574 | 0.170 | 0.936 |
| leg_walk1 | 0.187 | 0.673 | **0.641** | 0.592 | 0.209 | 0.946 |

`dof_vel` 5 Hz 저역통과 하나로 DRAIL 의 B' 점수가 3.4~5.6배 올라 expert 와의 격차 48/66/60% 가 닫힌다.
같은 처리가 DRAIL 로 잰 A' 에는 아무 영향이 없고(0.070→0.086, 0.068→0.069, 0.085→0.087 — 절제에서
A' 의 문제는 `dof_pos` 였음과 일치), `ang_vel` 만 거르면 거의 안 움직인다. 즉 **DRAIL 이 B' 를 낮게
보는 이유의 절반 이상은 관절속도 >5 Hz 성분 하나**이고, 그것을 빼면 B' 는 DRAIL 에게 0.5~0.68 의
"절반쯤 expert" 다. 어느 조합도 expert 0.94 에는 못 미친다(최대 0.678). 필터본은 정책이 낼 수 있는
궤적이 아니라 판별기가 그 대역을 얼마나 쓰는지의 측정이다.

### 측정 3 — 조건에 맞는 걸음을 더 쳐주는가 (4096 env, cmd 구간 × 걸음, 두 판별기 × 두 정책)

| cmd 2.5~3.5, gallop − pace | 자기 D | Cohen d | 상대 D | d |
|---|---|---|---|---|
| A' 롤아웃 | mlp **+0.155** | +1.97 | drail +0.099 | +1.32 |
| B' 롤아웃 | drail **−0.017 (ns)** / stand −0.041 | −0.14 / −0.34 | mlp +0.010 | +0.78 |

- mlp D 는 고속에서 gallop 을 pace 보다 확실히 더 준다. DRAIL D 는 **자기 정책 롤아웃에서는 그 차이가
  없거나 pace 가 근소하게 높다.** B' 의 gallop 선택은 DRAIL D 가 gallop 을 우대해서가 아니다.
- 저속(0.5~1.5)에 gallop 이 없어 "pace > gallop" 은 판정 불가. 실제로 있는 pace vs trot 은 **8칸 전부
  pace 가 진다.** 어느 D 도 "조건에 맞는 걸음" 을 더 주지 않는다.
- 가장 강한 반증: mlp D 는 cmd 2.5~3.5 아래 **서 있는** env 에 1.64~1.72 를 주고 같은 칸 gallop 은
  0.75 다(n 5~20). 조건과 운동학이 최대로 어긋난 표본이 최고점이다.

### 측정 4 — 조건 열을 치환하면 (정책 표본, 4 클립, 처치 7종)

조건 열 = `|lin_vel_cmd|/3.2` + valid. 클립 속도에 **맞는** 라벨과 **어긋난** 라벨(예: 2.86 m/s 롤아웃에
0.32 m/s 라벨)을 붙여 두 판별기로 재채점했다.

| D(맞춤) − D(어긋남) | 부호 | 단일 처치 최대 ΔD (policy) | expert 3라벨 스윙 | pool 셔플 상승 | drop |
|---|---|---|---|---|---|
| mlp D | **24칸 중 21칸 음수** (맞는 라벨이 더 낮음, 두 D 합산) | +0.156 | 0.028~0.101 | 16/16 | ±0.04 |
| DRAIL D | 같은 방향 | +0.09 | 0.009~0.030 (거의 무반응) | 13/16 | ±0.03 |

- **조건 열은 무시되지 않지만 방향이 반대다.** 맞는 라벨을 붙이면 D 가 낮고 엉뚱한 라벨·셔플이 높다.
  "상수 라벨이 조합 밖이라 0.5 로 밀린다" 는 대안은 원본 D 0.5~0.75 구간에서도 셔플이 D 를 올려(+0.02~
  0.07) 기각됐다. 09-03 의 expert 배치 평균 게이트("D 는 조건을 안 본다")가 정책 표본에서도 재현됐고,
  측정 3 의 "서 있는 env 최고점" 과 같은 그림이다.
- 설계상 이유가 보인다. `command_matched` 샘플링에서는 expert 라벨도 policy 라벨도 **같은 명령 분포**에서
  나오므로 라벨은 클래스 정보를 거의 갖지 않는다. 조건부 D 가 사실상 명령으로 재가중된 expert 혼합에 대한
  무조건부 D 로 퇴화한다는 정황이다(원인 함수형은 미확인).

### 판정 — DRAIL 판별기의 역할

1. **style 비평가로서는 mlp 보다 넓게 본다.** 위치·속도·발·자세를 고루 보고 최신 프레임에 가중한다.
   B' 가 위치·발·높이·자세 채널에서 참조에 더 가까운 것은 이 D 의 신호와 방향이 맞는다.
2. **그 판정의 절반 이상은 B' 의 관절속도 고주파 하나에 걸려 있다.** 저역통과만으로 0.09→0.51~0.67.
   그럼에도 정책이 그 떨림을 못 없앤 것은 보상이 포화 꼬리(D≈0.1, softplus 기울기 0.1)에 있어 압력이
   약하고 style 몫이 총 보상의 16% 라는 정황과 맞는다(§괴리 판정). 인과는 미확인.
3. **조건부 기능은 두 판별기 모두에서 작동하지 않는다.** 조건에 맞는 걸음을 더 주지 않고(측정 3),
   조건 라벨의 부호가 반대다(측정 4). 따라서 §"최종 체크포인트 3자 비교" 에서 "조건화의 고유 효과" 로
   읽은 명령 전환 추종 격차(6.2 vs 43.8/33.2%)는 **D 의 조건 사용으로는 설명되지 않는다.** 남는 후보는
   `command_matched` 가 바꾸는 expert 클립 혼합, `amp_cond_dropout`, 그리고 run 당 n=1 의 시드 분산이다.
   시드 반복 없이는 그 격차를 조건화 효과로 부를 수 없다.
4. B' 의 gallop 선택은 DRAIL D 가 gallop 을 우대해서가 아니다(측정 3). 고속 추종이 더 좋은 정책이 고속에서
   gallop 을 고른 것이 task 쪽 이유일 가능성이 있으나 이 조사로는 확정 못 한다.

다음: (a) 조건부 arm 시드 2개 추가로 전환 추종 격차 재현 여부; (b) 조건 열이 클래스 정보를 갖도록
expert 라벨을 클립 평균 속도로 되돌리되 누설 없이(라벨 잡음 또는 `amp_cond_reward_blend`) 재설계;
(c) 행동 저역통과 배포 실험(측정 2 의 반사실을 실제 궤적으로).

도구: `_workspace/leg/gait_reward_by_class.py`, `disc_offline.py`, `disc_saliency.py`,
`lowpass_rescore.py`, `disc_cond_swap.py`; `gait_survey_multienv.py --other_checkpoint`,
`amp_obs_clip_probe.py --save_seq`. 그림 `figures/{gait_reward_by_class,disc_saliency,disc_occlusion,lowpass_rescore,disc_cond_swap}.png`.

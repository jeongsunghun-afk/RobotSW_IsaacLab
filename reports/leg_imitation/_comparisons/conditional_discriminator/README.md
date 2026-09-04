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

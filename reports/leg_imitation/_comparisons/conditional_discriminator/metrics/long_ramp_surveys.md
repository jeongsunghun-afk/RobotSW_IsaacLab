# 긴 속도 램프(0→3.5→0) — cond-mlp(A') vs cond+DRAIL(B') 원자료

> 작성 일시: 2026-09-04 10:44
> 비교 대상: `condmatch_cmdchg4s_...vmax32/model_49999`(A', cond-mlp) vs
> `condmatch_drail_resume16k7_cmdchg4s_...vmax32/model_45000`(B', cond+DRAIL) — 조건부 discriminator 구조(mlp vs DRAIL)

## 실행 조건

- 스크립트: `_workspace/leg/speed_ramp_record_rma.py` (`--run_params`로 학습 run의 `env.yaml` 복원,
  `agent.yaml`로 B'의 `disc_arch=drail` 자동 복원)
- 명령 프로파일: `--vx_max 3.5` → 0, 0.5, …, 3.0, 3.5, 3.0, …, 0 (0.5 m/s 단위, 15 stage)
- 타이밍: `--hold_s 6.0 --ramp_s 2.0` (stage 8 s, 전체 120 s)
- `--force_stand`(정지 출발) · `--no_video`(렌더 없음) · `--seed 0/1/2` · `heading_hold` 등 다른 플래그 없음
- GPU: A' = `CUDA_VISIBLE_DEVICES=1`, B' = `CUDA_VISIBLE_DEVICES=2` (GPU0/3 미사용, 두 GPU 병렬·GPU 내부 순차)
- 실행 시각: 2026-09-04 10:34~10:44 (KST), 6 run(정책 2 × 시드 3) 전부 정상 종료(exit 0)
- 산출: `metrics/long_ramp/{condmlp50k,conddrail45k}_s{0,1,2}/ramp_data.npz`
- 레벨/방향(상승·하강) 판정은 cmd 값이 아니라 `vx_profile` 순서 + `ramp_s`/`hold_s` 로 계산한 **stage 시간
  구간**으로 나눴다(cmd 값은 상승·하강에서 중복되므로 값 매칭만으로는 방향을 못 가른다). 레벨별 집계는
  각 stage hold 구간의 **후반 4 s**만 평균했다(`plot_long_ramp_compare.py`, `TAIL_S=4.0`).

## 낙상 판정 — ramp_fall_probe 는 이 측정에서 신뢰 불가, 자체 판정으로 대체

`ramp_fall_probe`(heading 30° 이상 이탈 + 전진 정지를 "낙상"으로 봄)는 6 개 중 4 개를 FALL, 2 개를
"?? (증거 상충)"으로 냈다. 그러나 그 스크립트는 `heading_hold` 사용을 전제한다(설계 주석: "정상 주행은
heading_hold 로 ±10° 이내"). 이번 측정은 팀 지시대로 `heading_hold`를 **끄고** 쟀으므로 120 s 동안
world-frame heading 이 자연히 7~171° 까지 돈다 — 정책이 몸이 향한 방향(body-frame)으로는 계속 명령을
따라가면서 그냥 다른 쪽을 보고 걷는 것이다(위 tracking 플롯에서 body-frame vx 추종은 6 run 내내 정상).
`adv_post_peak` 가 음수(예: −77.6 m)로 나오는 것도 "뒤로 넘어짐"이 아니라 "돌아서 반대 방향으로 걸음"이다.

그래서 **자체 판정**을 별도로 넣었다: 각 stage의 tail 4 s 창에서 thigh/calf 관절이 `FOLDED_RAD`
(−0.6 rad) 아래로 접힌 비율이 0.5 를 넘으면 그 stage 를 "낙상 후 누움"으로 표시(`SeedRun.stages[i]["fallen"]`).
**6 run × 15 stage = 90 stage 전부 folded_frac 0.0 — 어떤 stage 도 낙상으로 표시되지 않았다.** 즉 6 회
롤아웃 전부 물리적으로 넘어진 적이 없고, 아래 모든 집계(달성률·토크·관절속도·보행)는 낙상 제외 없이
90/90 stage 를 그대로 썼다.

| run | ramp_fall_probe 판정 | stop_t / cmd | yaw_end [deg] | adv_post_peak [m] | 자체 folded stage |
|---|---|---|---|---|---|
| condmlp50k_s0 | FALL | 58.0 / 3.5 | +87.3 | −43.1 | 0/15 |
| condmlp50k_s1 | FALL | 58.0 / 3.5 | +54.1 | −77.6 | 0/15 |
| condmlp50k_s2 | ?? (상충) | 58.0 / 3.5 | +7.0 | −41.2 | 0/15 |
| conddrail45k_s0 | FALL | 71.0 / 3.0 | −82.1 | +32.1 | 0/15 |
| conddrail45k_s1 | ?? (상충) | 107.7 / 0.5 | +117.0 | +39.9 | 0/15 |
| conddrail45k_s2 | FALL | 58.0 / 3.5 | −171.1 | −32.7 | 0/15 |

## 레벨별 달성률 vx/vx_cmd (mean±std, seed n=3)

| policy/방향 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| cond-mlp 상승 | - | 0.607±0.444 | 0.576±0.408 | 0.834±0.066 | 0.939±0.032 | 0.924±0.004 | 0.919±0.002 | 0.904±0.008 |
| cond-mlp 하강 | - | 1.071±0.006 | 0.858±0.018 | 0.935±0.018 | 0.948±0.024 | 0.924±0.004 | 0.921±0.003 | - |
| cond+DRAIL 상승 | - | 0.832±0.036 | 0.936±0.028 | 0.975±0.005 | 0.971±0.005 | 0.979±0.004 | 1.022±0.033 | 0.961±0.023 |
| cond+DRAIL 하강 | - | 0.816±0.079 | 0.949±0.009 | 0.990±0.019 | 0.989±0.018 | 1.040±0.031 | 1.006±0.017 | - |

cond-mlp 상승의 cmd 0.5/1.0 표준편차(0.44/0.41)가 유별나게 크다 — 시드별로 뜯어보면 원인이 단조롭다:

| run | cmd 0.5 rate | cmd 1.0 rate |
|---|---|---|
| condmlp50k_s0 | 0.790 | 0.871 |
| condmlp50k_s1 | 1.035 | 0.858 |
| **condmlp50k_s2** | **−0.004** | **−0.001** |

`condmlp50k_s2` 는 cmd 0.5→1.0 구간(t≈8~24 s) 내내 vx≈0 으로 **정지 상태를 벗어나지 못한다**(같은
정책·같은 cmd 프로파일에서 s0/s1 은 정상 개시). cmd 1.5 부터 0.781 로 정상 추종을 회복한다.
[[project_leg_lowspeed_gait_onset_hysteresis]] 에서 본 "정지→보행 진입 실패"가 이번 3-시드 표본에도
1/3 확률로 재현됐다. cond+DRAIL 은 3 시드 전부 cmd 0.5 부터 0.82~0.88 로 정상 개시했다(같은 실패 없음,
n=3 이라 결정적이지는 않다).

## 레벨별 절대 오차 |vx-vx_cmd| [m/s] (mean±std)

| policy/방향 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| cond-mlp 상승 | 0.009±0.008 | 0.219±0.201 | 0.424±0.408 | 0.252±0.096 | 0.124±0.062 | 0.191±0.009 | 0.242±0.005 | 0.334±0.028 |
| cond-mlp 하강 | 0.119±0.011 | 0.052±0.004 | 0.142±0.018 | 0.111±0.015 | 0.106±0.046 | 0.191±0.010 | 0.238±0.008 | - |
| cond+DRAIL 상승 | 0.136±0.011 | 0.110±0.012 | 0.112±0.014 | 0.098±0.012 | 0.086±0.008 | 0.086±0.003 | 0.185±0.052 | 0.203±0.026 |
| cond+DRAIL 하강 | 0.150±0.017 | 0.125±0.025 | 0.096±0.007 | 0.088±0.004 | 0.114±0.021 | 0.171±0.057 | 0.159±0.032 | - |

cond-mlp 의 큰 오차(0.5/1.0 상승)는 위 저속 개시 실패 시드(s2)가 끌어올린 것이다. 그 구간을 빼면 두
정책의 절대 오차는 0.1~0.2 m/s 대로 비슷하고, cmd 3.0~3.5 로 갈수록 cond-mlp 쪽이 조금 더 커진다
(0.24~0.33 vs 0.16~0.20).

## 토크 포화율 frac(|tau|>0.95·effort_limit), 전 관절 평균 (mean±std)

| policy/방향 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| cond-mlp 상승 | 0.000±0.000 | 0.000±0.000 | 0.002±0.002 | 0.010±0.002 | 0.008±0.005 | 0.004±0.001 | 0.003±0.000 | 0.003±0.000 |
| cond-mlp 하강 | 0.000±0.000 | 0.000±0.000 | 0.002±0.001 | 0.007±0.003 | 0.007±0.005 | 0.004±0.001 | 0.003±0.000 | - |
| cond+DRAIL 상승 | 0.038±0.018 | 0.035±0.008 | 0.034±0.004 | 0.034±0.003 | 0.035±0.004 | 0.034±0.003 | 0.027±0.003 | 0.031±0.006 |
| cond+DRAIL 하강 | 0.033±0.010 | 0.034±0.003 | 0.036±0.004 | 0.038±0.003 | 0.028±0.007 | 0.028±0.004 | 0.029±0.003 | - |

cond+DRAIL 은 **cmd 0(정지) 부터** 포화율이 3.3~3.8% 로, cond-mlp 의 전 구간 최댓값(1.0%, cmd 1.5)보다
높다. 속도에 거의 무관하게 2.7~3.8% 대를 유지한다 — cond-mlp 는 속도가 오를수록 포화율이 오르는
정상적인 패턴인 반면, cond+DRAIL 은 이미 정지 상태부터 토크를 상시 크게 쓰고 있다는 뜻이다(그룹별
RMS/peak 는 `figures/long_ramp_torque.png` — cond+DRAIL 은 cmd 0 에서도 다리 관절 RMS 26~29 N·m,
peak 77~83 N·m 로 cond-mlp(RMS 4.6~19.6, peak 10.2~36.8, 그룹마다 편차가 큼)의 1.5~6 배).

## 관절 각속도 peak [rad/s] — 그룹 중 최댓값 (mean±std). velocity_limit_sim = 30.0 rad/s

| policy/방향 | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 |
|---|---|---|---|---|---|---|---|---|
| cond-mlp 상승 | 0.61±0.78 | 4.16±2.90 | 6.14±4.39 | 8.34±0.90 | 11.76±1.78 | 12.78±0.76 | 12.33±0.76 | 13.00±2.13 |
| cond-mlp 하강 | 4.32±0.85 | 7.06±0.56 | 9.06±0.81 | 8.03±0.73 | 11.93±1.29 | 12.89±0.65 | 12.32±0.72 | - |
| cond+DRAIL 상승 | 15.57±2.45 | 23.81±3.36 | 24.23±1.43 | 24.79±3.26 | 26.07±2.88 | 26.90±2.46 | 30.42±1.37 | 29.30±0.83 |
| cond+DRAIL 하강 | 17.65±7.07 | 23.38±3.84 | 24.12±4.39 | 23.35±0.96 | 27.82±1.58 | 28.54±1.76 | 29.09±0.52 | - |

cond+DRAIL 은 cmd 3.0(상승, HL 그룹)에서 개별 시드 최댓값 **31.7 rad/s** 로 `velocity_limit_sim`
(30.0 rad/s)을 **실제로 넘겼다**(FL 그룹도 3.0 하강에서 29.8, FR 3.0 상승 29.9 등 다수 그룹이 근접·초과).
cmd 3.0~3.5 구간에서 FL/FR/HL/HR 4 그룹 모두 27~32 rad/s 대에 몰려 있어 — 사실상 **관절 각속도
포화 상태로 학습·평가되고 있다.** cond-mlp 는 같은 구간에서 최대 13~16 rad/s 로 한계의 절반 이하다.
cmd 0(정지) 에서도 cond+DRAIL 이 15.6 rad/s(cond-mlp 0.6 rad/s)로, 다리 관절 RMS 도 2.85 rad/s(시드 3.04/2.89/2.63; cond-mlp 0.08)로, 위 토크 포화율과 같은 패턴 —
cond+DRAIL 은 정지 명령에서도 다리를 크게 흔든다.

## 레벨별 보행 분류 (`gait_classify` 재사용, hold 구간 전체 기준)

| run | 0.0↑ | 0.5↑ | 1.0↑ | 1.5↑ | 2.0↑ | 2.5↑ | 3.0↑ | 3.5↑ | 3.0↓ | 2.5↓ | 2.0↓ | 1.5↓ | 1.0↓ | 0.5↓ | 0.0↓ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| condmlp50k_s0 | gallop* | other | other | pace | pace | pace | pace | pace | pace | pace | pace | pace | other | other | gallop* |
| condmlp50k_s1 | stand | other | other | other | pace | pace | pace | pace | pace | pace | pace | pace | other | other | other |
| condmlp50k_s2 | other | -† | stand‡ | other | pace | pace | pace | pace | pace | pace | pace | pace | other | other | gallop* |
| conddrail45k_s0 | gallop* | other | other | pace | pace | pace | **gallop** | **gallop** | **gallop** | **gallop** | **gallop** | pace | other | trot§ | other |
| conddrail45k_s1 | other | other | other | pace | pace | pace | **gallop** | **gallop** | **gallop** | **gallop** | pace | pace | other | trot§ | other |
| conddrail45k_s2 | other | other | other | pace | pace | pace | **gallop** | **gallop** | **gallop** | **gallop** | **gallop** | pace | other | other | gallop* |

\* cmd=0.0 의 gallop/stand 라벨은 신뢰 낮음 — 정지 명령 구간은 진폭이 `MIN_AMP_RAD` 근처라 정착 시
떨림이 3 cycle 이상 잡히면 분류기가 임의 라벨을 낸다(같은 정책·같은 cmd 인데 seed 마다 gallop/stand/other
로 갈리는 것 자체가 증거). 실제로는 "서 있음"에 가깝다고 보는 게 맞다.
† `condmlp50k_s2` cmd 0.5 상승은 위 저속 개시 실패로 진폭 자체가 없어 `-`(사이클 부족).
‡ 같은 이유로 cmd 1.0 상승도 `stand`.
§ cond+DRAIL 의 cmd 0.5 하강에서 2/3 시드가 `trot` — 저속 하강 구간의 유일한 non-pace/gallop/other
정상 걸음 라벨인데, 짧은 구간(hold 6 s, 저속이라 사이클도 적다)이라 안정적 걸음인지 과도기 잡음인지는
이 데이터만으론 확언 못 한다(미확인).

**cond-mlp는 cmd 2.0~3.5 전 구간(상승·하강 모두) 3/3 시드가 pace 고정**(1.5 상승은 s0 만 pace, s1/s2 는 other) — gallop 전환이 한 번도
없다. **cond+DRAIL은 cmd 3.0~3.5 상승에서 3/3 시드가 gallop**으로 전환하고, 하강에서는 cmd
3.0(3/3)·2.5(3/3)·2.0(2/3) 까지 gallop 을 유지하다가 1.5 에서 3/3 전부 pace 로 돌아간다 — 상승 때
gallop 진입 문턱(3.0)보다 하강에서 gallop 이 버티는 하한(2.0~2.5)이 더 낮다. 한번 gallop 에 들어가면
속도가 줄어도 곧바로 pace 로 안 돌아가는 **히스테리시스**가 3 시드 모두에서 일관되게 보인다.

## 3.5 (OOD, 학습 상한 3.2 초과) 구간에서 3.0 대비 달라지는 것

1. **속도 자체는 계속 따라간다** — cond-mlp 0.919→0.904, cond+DRAIL 1.022→0.961 로 양쪽 다 큰 붕괴
   없이 3.5 도 추종한다(하드 실속 없음). cond-mlp 의 절대 오차는 이 sweep 전체 최댓값(0.334 m/s)을
   3.5 에서 찍는다.
2. **cond-mlp 는 토크 peak 이 3.5 에서 그룹마다 +23~43% 뛴다**(FL 77.2→94.9, FR 73.8→91.3,
   HL 79.6→**114.2** N·m(+43%), HR 80.5→108.7) — 3.0 이 아직 토크 여력의 끝이 아니었다. 관절속도
   peak 은 같은 구간에서 +3~9%(FL 12.3→13.0, HR 9.9→10.6 rad/s)로 늘긴 하지만 폭이 훨씬 작다 —
   cond-mlp 의 3.0→3.5 확장은 **토크 쪽 여력을 주로 쓴다.**
3. **cond+DRAIL 은 3.0 에서 이미 토크 천장에 닿아 3.5 에서 거의 안 오른다** — 토크 peak FL +0%
   (106.0→106.4), FR +8%(98.3→105.9), HL +6%, waist +3%, HR 은 오히려 −14%(101.0→86.9, 시드
   n=3 편차가 커 잡음일 수 있다). 관절속도 peak 도 FL +5%·FR +2% 로 거의 평평하고 HL(−15%)·HR(−7%)은
   오히려 준다 — 다만 이 그룹들은 시드 간 표준편차가 크다(예: HL 3.5 는 30.1/16.4/31.0 로 한 시드만
   크게 낮음). 3.0 시점에 이미 velocity_limit_sim 근방이라 3.5 로 더 밀어붙일 토크·속도 여력이
   cond-mlp 만큼 남아 있지 않다는 방향은 분명하지만, 그룹별 정확한 증감폭은 n=3 표본으로 확정하기 어렵다.
4. **걸음은 유지** — cond-mlp 는 3.5 에서도 pace 그대로(gallop 전환 없음), cond+DRAIL 은 3.5 에서도
   3/3 gallop 유지(3.0 에서 이어짐). OOD 로 넘어간다고 걸음이 새로 바뀌지는 않는다.
5. 하강 방향에는 3.5 stage 가 없다(사다리꼴 정점이라 한 번만 지남) — 3.5 에서의 히스테리시스는 이
   측정으로는 볼 수 없다.

## 산출 파일

- `figures/long_ramp_tracking.png` — 속도 추종(시드별 얇은 선 + 시드평균 굵은 선) + 레벨별 달성률/오차 막대
- `figures/long_ramp_torque.png` — 관절 그룹별 토크 RMS/peak + 포화율
- `figures/long_ramp_jointvel.png` — 관절 그룹별 각속도 RMS/peak (velocity_limit_sim 30.0 rad/s 점선)
- `figures/long_ramp_gait.png` — 레벨×시드 보행 격자(상승/하강 분리)
- `metrics/long_ramp/{condmlp50k,conddrail45k}_s{0,1,2}/ramp_data.npz` — 원자료

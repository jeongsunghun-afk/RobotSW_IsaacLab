# Parkour 3-leg fix: `max_air` cap 데이터 분석

**목적**: 발별 air-time(연속 공중 시간) 패널티의 상한값 `max_air`(초)를 데이터로 확정.
정상 swing/jump flight phase는 처벌하지 않고, RL이 영구히 든 발(dropped foot)만 처벌하는 cap을 추천.

**데이터**: `rank0_verify_result.npz` — 성숙 정책(model_21900) 롤아웃, 2000 step × 64 env × 4 foot contact bool (force > 2N).
Foot 순서 `[FL, FR, RL, RR]`. RL이 dropped foot (접지율 0.106).

---

## 1. step_dt (제어 주기)

```
step_dt = sim.dt × decimation = (1/200) × 4 = 0.02 s   (50 Hz)
```
**출처**:
- `parkour_env_cfg.py:436` — `dt: float = 1 / 200` (physics 200 Hz)
- `parkour_env_cfg.py:408` — `decimation: int = 4`
- `parkour_env_cfg.py:603` 주석에서도 명시: "Policy step dt = decimation(4) / physics_rate(200 Hz) = 0.02 s"

모든 air-time = (연속 non-contact step 수) × 0.02 s.

---

## 2. 발별 air-time 분포 (전체 지형, 초)

| foot | n_seg | median | p90 | p95 | p99 | max |
|------|------:|-------:|----:|----:|----:|----:|
| FL_foot | 8160 | 0.080 | 0.180 | 0.200 | 0.240 | 1.560 |
| FR_foot | 7927 | 0.140 | 0.280 | 0.340 | 0.420 | 1.700 |
| RR_foot | 7817 | 0.120 | 0.180 | 0.220 | 0.300 | 0.500 |
| **RL_foot (dropped)** | 4749 | **0.260** | **1.244** | **1.680** | **2.880** | **5.600** |

**정상 발(FL/FR/RR) 합산**: n=23904, median 0.120, p90 0.200, p95 0.260, **p99 0.360**, p99.9 0.540, p99.99 0.804, **max 1.700**.

정상 발은 대부분의 swing이 sub-0.4s. 0.5s를 넘는 segment는 전체 23,904개 중 **단 34개(0.14%)**.
RL 발은 median(0.26s)부터 이미 정상 발 p99 수준이고, 13.3%의 segment가 1.0s를 초과, 최대 5.6s까지 영구 부양.

---

## 3. 정상 발 × 지형별 (점프 지형 flight phase 확인)

정상 발 합산, terrain class(segment 시작 시점 기준):

| terrain | n | p95 | p99 | max |
|---------|--:|----:|----:|----:|
| flat | 4892 | 0.160 | 0.200 | 0.340 |
| hurdle | 4803 | 0.320 | 0.360 | 0.540 |
| step | 4826 | 0.280 | 0.360 | 0.740 |
| **gap** | 4727 | 0.240 | 0.320 | **1.700** |
| stair | 4656 | 0.260 | 0.449 | 0.640 |

점프 지형(gap)에서 꼬리(max)가 가장 길다. **상위 정상-발 outlier Top 2는 모두 동일 env42, gap 지형, 인접 step(1449/1456)에서 발생** (FR 1.70s, FL 1.56s) — 두 앞발이 동시에 긴 gap 도약 flight phase를 겪은 **실제 점프 비행**으로, contamination이 아닌 진짜 동작. 그 다음은 0.82s(gap), 0.78s(gap)로 급감.

→ 정상 점프를 처벌하지 않으려면 cap은 gap의 극단적 flight phase(~1.7s)를 고려해야 함. 다만 이런 1.5s+ 도약은 23,904개 중 2개로 극히 드물다.

## 3b. RL(dropped) 발 × 지형

| terrain | n | p95 | p99 | max |
|---------|--:|----:|----:|----:|
| flat | 1800 | 0.600 | 0.920 | 1.640 |
| hurdle | 734 | 1.740 | 2.187 | 3.180 |
| step | 910 | 1.471 | 2.615 | 4.940 |
| gap | 717 | 1.896 | 3.122 | 4.120 |
| stair | 588 | 2.893 | 3.646 | 5.600 |

지형 극복 구간일수록 dropped 발의 부양 시간이 길다 (stair p95 2.89s). 이것이 3-leg gait의 본질.

---

## 4. 핵심: 두 anchor가 역전한다 (실질 결론)

과제는 cap이 "정상-발 점프 포함 max보다 위 + RL 전형 air-time보다 아래"이길 요구하지만, 데이터상 두 기준이 **역전(overlap)** 되어 둘 다 만족 불가:

```
정상-발 max (gap 도약)  = 1.70 s
RL 발 p90              = 1.24 s     ← 정상 max보다 작다
```

즉 cap을 1.7s 위에 두면 RL 부양의 상당 부분(p90~p95 구간)을 못 잡고, RL을 확실히 잡으려 낮추면 극히 드문 gap 도약 2건을 clip하게 된다. **이 trade-off를 선택하는 것이 본 분석의 산출물**이다.

### 패널티 형태 가정
과제 명칭이 "상한 **패널티**"(termination 아님)이고, env가 이미 `current_air_time`을 per-foot로 노출(`parkour_env.py:1149`, `track_air_time=True` @ `cfg:582`)하므로 **graded per-step 패널티** 형태로 가정:
```
penalty = - scale × Σ_foot  max(0, current_air_time[foot] - max_air)
```
graded이면 드문 gap 도약 2건은 0.5~0.7s만큼의 일회성 소액 overage만 발생 → 무시 가능. 따라서 **RL을 실제로 무는 쪽**을 택한다.

---

## 5. 추천 `max_air`

### 추천값(primary): **`max_air = 1.0 s`** (graded penalty 가정)

근거 (정상 vs RL segment가 cap을 초과하는 비율):

| cap (s) | 정상-발 초과 | RL-발 초과 |
|--------:|------------:|-----------:|
| 0.30 | 2.71 % | 37.6 % |
| 0.40 | 0.37 % | 31.7 % |
| 0.50 | 0.14 % | 29.7 % |
| 0.60 | 0.038 % | 25.6 % |
| 0.80 | 0.013 % | 17.9 % |
| **1.00** | **0.008 % (2건)** | **13.3 %** |

- 1.0s는 정상-발 p99.99(0.804s) 위로 여유가 있어, 잡히는 정상 segment는 gap 대도약 **2/23,904(0.008%)** 뿐이고 그나마 graded라 overage가 미미.
- 동시에 RL의 **multi-second 병리적 꼬리 전체**(p99=2.88s, max 5.6s)를 포함한 13.3%의 segment를 처벌 → 영구 부양을 직접 타겟.
- 정상 점프(hurdle/step/stair max ≤ 0.74s)는 전혀 건드리지 않음.

### 범위
- **보수(conservative): 1.5 ~ 1.8 s** — 정상 gap 도약(max 1.70s)까지 완전 무처벌. 단 RL의 극단 꼬리(~5~7% segment)만 잡혀 약함. termination 형태라면 이 범위(>1.7s) 필수.
- **공격(aggressive): 0.6 s** — RL segment 25.6% 처벌, 정상은 여전히 0.038%(9건). 학습 신호를 강하게 주되 일부 긴 stair/gap swing(0.6~0.74s)을 살짝 건드릴 위험.

**권고**: graded penalty로 구현하고 **`max_air = 1.0 s`** 로 시작. 학습 중 정상 보행이 위축되면 1.2~1.5s로 완화, dropped foot 교정이 약하면 0.8s로 강화.

---

## 6. Caveat — reset 경계 미보정

npz에 episode reset 경계 정보가 없다. Episode 길이 = 20s / 0.02s = **1000 step**이므로 2000-step 윈도우 안에서 각 env는 **약 1~2회 reset**을 겪는다. Reset 직후 발이 떠 있으면 인접 air segment가 합쳐져 air-time이 과대평가될 수 있다(특히 RL 꼬리).

영향 판단: 그래도 **정상 발 꼬리가 매우 타이트**(p99.99 = 0.80s, 0.5s 초과 단 34/23,904)하여 reset contamination이 추천값(1.0s)을 움직일 수 없다. RL 발은 reset과 무관하게 본질적으로 부양 상태이므로(접지율 0.106) 결론 불변. 다만 표의 절대 max(특히 RL 5.6s, 정상 1.7s)는 reset로 일부 부풀 수 있음을 명시한다.

---

## 부록: 재현
- 계산 스크립트: `_workspace/3leg_gait_research/_airtime_calc.py`
- 실행: `conda run -n isaac python _workspace/3leg_gait_research/_airtime_calc.py`
- 의존성: numpy only (시뮬 미실행)

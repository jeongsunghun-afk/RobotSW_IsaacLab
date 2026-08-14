> # ⚠⚠ 이 문서의 수치는 무효다 — [`GEAR_K_INVALIDATION.md`](GEAR_K_INVALIDATION.md) 먼저 읽을 것
>
> 브리지(`real_runner_bipedleg.cpp:73 motor_deg_to_sim`)가 `gear_k` 로 나누지 않아,
> 아래 모든 적합이 **calf 1.5배 / foot 1.2배 어긋난 각도** 위에서 돌았다. 커플링 계수 +1 도
> 모델각이 아니라 채널각 공간에 적용됐다(`RL_INTERFACE.md` §2-(a) 가 경고하는 0.8배 어긋남).
> **살아남는 결론은 §6 의 에너지 논증 하나뿐이다** — foot viscous 3.9 는 물리적으로 불가능하다.
> §3 의 "공선성" 기전 설명과 §5 의 권고는 폐기한다.

# PACE biped-leg — foot viscous 는 물성인가 커플링 잔차인가 (2026-08-13)

**판정: 커플링 잔차다.** 전체 chirp 적합이 낸 `foot viscous ≈ 3.7~4.1 N·m·s/rad` 는 foot 전달계의
물리량이 아니다. calf 를 고정한 foot 단독 chirp 로 재적합하면 같은 계수가 **~2e-4 로 붕괴**한다
(1/17,000~1/24,000). viscous 는 관절 고유 물성이므로 이웃 관절의 운동 여부에 의존할 수 없다.

> ⚠ **인용 출처 주의**: CMA-ES 로그에 찍히는 `Armature:`/`Viscous:` 블록은 그 세대의
> **best individual** 이고, 배포용 `export_pace_params.py` 가 읽는 것은 `mean_*.pt`(분포 평균)다.
> 둘은 다른 객체이며 잘 식별되지 않는 파라미터에서 최대 14배까지 벌어진다(예: 전체 적합
> `viscous[HL_calf]` best 0.0015 vs mean 0.0214). 아래 표는 전부 **mean 체크포인트** 값이다.

## 1. 두 적합 비교 (foot 관절만, mean 체크포인트)

| 파라미터 | 전체 chirp 적합 | foot 단독 부분적합 | 비 |
|---|---|---|---|
| armature HL_foot [kg·m²] | 0.2309 | 0.0301 | ×0.130 |
| armature HR_foot [kg·m²] | 0.2235 | 0.0297 | ×0.133 |
| **viscous HL_foot** [N·m·s/rad] | **3.699** | **0.000091** | **×2.5e-5** |
| **viscous HR_foot** [N·m·s/rad] | **4.138** | **0.000134** | **×3.2e-5** |
| coulomb HL_foot [N·m] | 0.3787 | 0.3880 | ×1.02 |
| coulomb HR_foot [N·m] | 0.0282 | 0.3192 | ×11.3 |
| bias HL_foot [rad] | +0.0109 | −0.0205 | 부호 반전 |
| bias HR_foot [rad] | −0.0092 | +0.0244 | 부호 반전 |
| delay (전역) [ms] | 2.18 | 2.80 | ×1.3 |

- 전체 chirp 적합: `logs/pace/bipedleg_real_coupled/26_08_13_09-16-45/mean_090.pt` — 종료(gen 91),
  min score **0.02640**. 데이터셋 4종 (`chirp_gui_20260812_1725~1729`).
- foot 단독 부분적합: `logs/pace/bipedleg_foot_probe/26_08_13_13-15-59/mean_114.pt` — CMA-ES 자체
  수렴 판정으로 gen 115 에서 종료(max_iteration 200 미도달), min score **0.000981**.
  `--fit_joints foot --freeze_from <위 mean_090> --datasets chirp_gui_20260813_1313{06,44}`.
  나머지 관절 파라미터는 전체 적합값으로 동결(bounds 폭 0). 마지막 40세대 동안 score 가
  0.000982→0.000981 로 4자리 정체, mean_110↔mean_114 도 소수 4자리까지 동일.
- score 는 데이터셋이 다르므로 **적합 품질의 직접 비교가 아니다**(잔차 스케일이 다름).

## 2. 교란 차단 — "여기(excitation) 부족" 가설 기각

foot 단독 chirp 에서 foot 이 덜 움직여서 viscous 가 식별 불가였을 가능성을 배제해야 한다.
데이터셋에는 속도가 없어 위치의 수치미분으로 계산했다 (leg-major 순서, 200 Hz 재격자 후):

| 데이터셋 | foot **raw** v_rms | calf v_rms | foot **관절** v_rms (raw−calf) | corr(raw, calf) |
|---|---|---|---|---|
| foot 단독 `131344` | 1.78 | 0.005 | **1.78** | +0.02 |
| foot 단독 `131306` | 0.79 | 0.005 | 0.79 | — |
| 전체 `172508` | 1.63 | 1.44 | 0.68 | **+0.909** |
| 전체 `172620` | 2.31 | 2.92 | 1.39 | **+0.885** |
| 전체 `172956` | 2.31 | 2.92 | 1.38 | **+0.885** |

- foot 관절 속도는 foot 단독 쪽이 **오히려 1.3~2.6배 크다**. 여기 부족이 아니다.
  1.3~2.6배 차이로 20,000배 계수 차이를 살 수는 없다.
- PD 게인은 네 캡처 전부 동일 (`kp [100,50,50,20]×2`, `kd 5`). viscous 는 kd 오차를 통째로
  흡수하므로 이게 달랐으면 비교 자체가 무효였다 — 동일하므로 설명 불가.

## 3. 기전 — 커플링이 만든 공선성(ill-conditioning)

전체 chirp 에서는 foot **raw** 속도와 calf 속도의 상관이 **+0.89~+0.91** 이다. 커플링
`q_raw = q_foot + q_calf` 때문에 foot 감쇠항이 calf 쪽 항들과 거의 축퇴되고, 옵티마이저는
점수를 거의 바꾸지 않으면서 계수를 아무 데나 세울 수 있다. foot 단독 chirp 는 calf 를 세워
이 축퇴를 깬다 (상관 +0.02).

**정합성 교차검증 — foot 단독 값이 물리적으로 더 그럴듯하다:**

- 좌우 대칭: armature 0.0301 / 0.0297 (전체 적합 0.231 / 0.224 는 값 자체가 이상치),
  coulomb 0.388 / 0.319 (전체 적합은 0.379 / **0.028** — 좌우 13배 차이는 기구적으로 설명 불가)
- 잘 식별된 파라미터는 두 적합이 **일치**한다: `coulomb HL_foot` 0.379 → 0.388 (×1.02).
  즉 foot 단독 적합이 전부를 갈아엎은 게 아니라, 축퇴됐던 항만 제자리를 찾았다.
- 관절 계열 내 정합: foot 단독 armature 0.030 은 hip/thigh/calf (0.015~0.032) 와 같은 대역.
  전체 적합의 0.22~0.23 은 혼자 한 자릿수 위로 튀는 이상치였다.
- viscous 상한이 5.0 인데 전체 적합은 3.7~4.1 (상한의 74~83%) 로 붙어 올라갔다. 다른 어떤
  관절도 0.28 을 넘지 않는다.

## 4. 남은 열린 문제 (이 실험으로는 판정 불가)

foot 단독 조건에서는 raw 좌표와 관절 좌표가 **일치**하므로 (`v_calf ≈ 0.005 rad/s`), 수동
물성(armature/마찰)이 raw(모터) 좌표에서 작용하는지 관절 좌표에서 작용하는지 구분할 수 없다.
현재 sim 은 관절 좌표에 적용한다. 이걸 판정하려면 **calf 만 구동하고 foot 명령은 정지시킨**
데이터셋이 필요하다 (그때 raw 는 움직이고 관절각은 0 유지 — 두 모델이 서로 다른 예측을 낸다).

**단, 이 미결 문제는 §5 권고를 흔들지 않는다.** 두 좌표가 일치하는 데이터셋이므로 크기 ~4 의
감쇠항이 *어느 좌표에 있든* 식별됐어야 하는데 ~2e-4 가 나왔다. "실제 foot 감쇠 ≈ 0" 은 좌표
문제가 어느 쪽으로 결론나든 살아남는다.

## 5. 권고

1. foot 파라미터는 **foot 단독 적합값**을 채택한다
   (armature 0.0301/0.0297, viscous ≈0, coulomb 0.388/0.319, bias −0.0206/+0.0244).
2. 그 값을 동결하고 나머지 관절을 전체 chirp 로 재적합해 배포 파라미터를 갱신한다. 전체 적합의
   non-foot 값들도 foot 의 가짜 감쇠를 보상하느라 편향됐을 수 있다. → **실행함**(2026-08-13 15:10):

   ```bash
   CUDA_VISIBLE_DEVICES=3 python scripts/real2sim/fit_bipedleg.py --headless --num_envs 1024 \
     --robot_name bipedleg_footfrozen \
     --fit_joints hip,thigh,calf \
     --freeze_from logs/pace/bipedleg_foot_probe/26_08_13_13-15-59/mean_114.pt \
     --datasets bipedleg_real_raw/chirp_gui_20260812_1725{08,39}.pt \
                bipedleg_real_raw/chirp_gui_20260812_172956.pt \
     --holdout  bipedleg_real_raw/chirp_gui_20260812_172620.pt \
                bipedleg_real_raw/chirp_gui_20260812_172716.pt
   ```

   데이터셋·hold-out 은 전체 적합(`bipedleg_real_coupled`)과 **동일**하게 두어 score 를 직접
   비교할 수 있게 했다. 결과는 §6 에 기록한다.
3. §4 판정용 "calf 단독 chirp" 캡처를 추가한다.

## 6. foot 고정 재적합 결과 — 점수는 나빠졌고, 그게 오히려 결론을 굳혔다

`logs/pace/bipedleg_footfrozen/26_08_13_15-10-49/mean_083.pt` — CMA-ES 자체 수렴으로 gen 84 종료.

| | 적합 score |
|---|---|
| 전체 적합 (foot 자유) gen 84 | 0.0265 |
| **foot 고정 재적합 gen 84** | **0.0361** (37% 나쁨, 마지막 10세대 하강폭 9.3e-5 = 수렴) |

foot 을 물리적으로 옳은 값에 고정하니 오히려 데이터를 못 맞춘다. 해석이 둘로 갈렸다 —
(가) 커플링 **모델**이 틀렸다 vs (나) foot 단독 적합값이 틀렸다. 적합 점수로는 구분 불가하고,
**hold-out 으로도 구분되지 않는다** — 이번 hold-out 은 게인이 전부 같아서(kp[100,50,50,20]/kd 5)
같은 계통 오차가 hold-out 에도 그대로 실려 있기 때문이다.

### 판정: 에너지 보존이 (나)를 배제한다

foot 단독 chirp 에서는 calf 가 정지(v_calf rms 0.005)라 foot 관절에 일을 하는 동력원이
**foot 모터뿐**이다. 점성은 항상 소산이고 중력은 보존력이라 주기 chirp 에서 순 일이 0 이므로,
가정 없이 부등식이 선다:

$$\langle \tau\,\dot q\rangle \;\le\; \tau_{rms}\,\dot q_{rms} \qquad\text{vs}\qquad P_{diss} = b\,\dot q_{rms}^2$$

τ 는 MIT 지령토크 `kp(des−q) + kd(0−q̇)` 로 데이터에서 직접 나온다.

| 데이터셋 | 관절 | τ_rms [N·m] | q̇_rms [rad/s] | 공급 상한 [W] | b=3.9 요구 [W] | 초과 |
|---|---|---|---|---|---|---|
| `131306` | HL_foot | 0.60 | 0.79 | 0.48 | 2.42 | **5.1×** |
| `131306` | HR_foot | 0.57 | 0.79 | 0.45 | 2.42 | **5.4×** |
| `131344` | HL_foot | 1.33 | 1.78 | 2.36 | 12.30 | **5.2×** |
| `131344` | HR_foot | 1.31 | 1.78 | 2.32 | 12.32 | **5.3×** |

`viscous = 3.9` 는 실기 모터가 공급할 수 있는 파워의 **5배**를 마찰로만 요구한다. 그런데 실기는
같은 구간에서 1.25 rad 진폭 chirp 를 실제로 추종했다 — **물리적으로 불가능**하다. 적합 점수와
무관하게 배제된다.

같은 계산으로 foot 단독 적합값(b≈1e-4, coulomb 0.35)은 예산의 **23~52%** 만 쓴다 — 여유 있음.

**따라서 (가)가 맞다**: 재적합의 37% 잔차는 foot 값이 틀려서가 아니라 **커플링 모델이 틀려서**다.
전체 적합의 `foot viscous 3.9` 는 그 모델 갭을 메우던 가짜 파라미터였다.

### 부수 발견 — calf coulomb 도 커플링 잔차였다

foot 을 고정하자 calf 의 Coulomb 마찰이 함께 무너졌다:

| 관절·블록 | 전체 적합 | foot 고정 재적합 | 비 |
|---|---|---|---|
| HL_calf coulomb [N·m] | 1.193 | 0.035 | ×0.03 |
| HR_calf coulomb [N·m] | 1.127 | 0.034 | ×0.03 |
| HL/HR_calf viscous | 0.021 / 0.031 | 0.009 / 0.011 | ×0.43 / ×0.34 |
| HL/HR_calf armature | 0.0143 / 0.0151 | 0.0260 / 0.0265 | ×1.8 |

전치 토크 경로로 foot 의 가짜 감쇠가 calf 마찰까지 부풀렸던 것이다. hip/thigh 는 ±10% 안쪽으로
거의 안 움직였다 — 오염이 커플링 경로에 한정됐다는 뜻이라 정합적이다.

### 모델 갭의 정체

지금 sim 은 비가역 전달기구를 **foot PD 게인(kp=20)짜리 스프링**으로만 흉내내고, relax 일 때만
`coupling_hold_kp=200` 으로 올린다. 실기 벨트+감속기는 그보다 훨씬 뻣뻣하다. 평상시에도 이
잠금을 걸어야 하는데, **그 크기는 아직 실측되지 않았다** — 감으로 정하면 지금 문제를 만든 과정을
반복하게 된다. §4 의 calf 단독 chirp 가 이 값을 준다.

⚠ 캡처 설계 주의: GUI chirp 는 관절각으로 설계해 발행 시 raw 로 변환하므로(`pose[f] += pose[c]`),
calf 만 마스크하면 **foot 모터가 calf 를 따라 돌아** 원하는 조건이 안 된다. **foot 의 kp/kd 를 0 으로
내린 채 calf 만 chirp** 해야 모터가 손을 떼고 감속기 마찰만 남는다 (GAIN 패킷이 관절별이라 GUI
슬라이더로 바로 됨).

## 재현

```bash
python reports/_comparisons/pace_bipedleg_foot_coupling_probe/logs/compare_foot_fits.py
# → metrics/foot_fit_comparison.json  (mean_*.pt 에서 직접 읽음)
python reports/_comparisons/pace_bipedleg_foot_coupling_probe/logs/torque_budget_check.py
# → metrics/torque_budget_check.json  (에너지 논증)
```

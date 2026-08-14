# 토크 좌표 — 채널기준 ↔ 관절기준 (r2s_biped_leg)

> **한 줄**: 실기가 보고하는 토크는 **채널기준**이고 sim 이 내는 토크는 **관절기준**이라, 그대로
> 겹쳐 보면 calf 1.5배 · foot 1.2배 어긋난다. 올리는 식은 **감속비 + 커플링 전치** 두 조각이다.

관련 문서: `RL_INTERFACE.md`(실기 사양) · `COORDINATE_MIGRATION_CHECKLIST.md`(정합성) ·
`DEPLOYMENT_CHECKLIST.md`(배포). 이 문서는 **토크 좌표**만 다룬다 (각도·속도는 위 문서들).

---

## 1. 왜 필요한가

`monitor.py` 는 sim 토크와 실기 토크를 **같은 축에 스케일 없이** 겹쳐 그린다. 그런데 둘의 좌표가 다르다.

| | 출처 | 좌표 |
|---|---|---|
| sim tau | `robot.data.applied_torque` (`get_lowstate`) | **관절** |
| real tau | 드라이버 `fTorque` — 브리지가 **무변환 통과** | **채널** |

게다가 sim 은 GUI 가 보낸 **채널 게인**을 관절 drive 게인으로 그대로 쓴다. 그래서 같은 추종오차에 대해

```
real 채널 :  τ = kp_ch · Δq_ch    = kp_ch · k · Δq_joint
sim  관절 :  τ = kp_ch · Δq_joint
          ⇒  real 트레이스 = k × sim 트레이스
```

**calf 1.5배 · foot 1.2배 · hip/thigh 1.0배.** hip·thigh 가 맞아떨어져서 오래 안 보였다 —
`gear_k` 원 버그와 **똑같은 서명**(k≠1 축에만 몰림)이다.

---

## 2. 좌표 4층 (요약)

| 층 | 기호 | 정의 | 감속비 | 커플링 |
|---|---|---|---|---|
| 모터축각 | $\theta_m$ | 실제 모터 샤프트 각. 직접 관측 불가 | 이전 | 미해제 |
| **채널각** | $q_{ch}$ | 드라이버가 보고/수신 $=\theta_m/7$ — **전 축을 7:1 로 가정** | 이전 | 미해제 |
| raw 각 | $q_{raw}$ | 부호·감속비·영점 제거 후 | 이후 | **미해제** |
| **모델각** | $q_{joint}$ | MJCF `qpos`. 정책·GUI 가 쓰는 유일한 단위 | 이후 | 해제됨 |

$$k_i \;=\; \frac{\text{실제 감속비}_i}{7}$$

| 축 | 실제 감속비 | $k$ |
|---|---|---|
| hip | 7 | 1.0 |
| thigh | 7 | 1.0 |
| **calf** | 10.5 | **1.5** |
| **foot** | 8.4 | **1.2** |

---

## 3. 토크 변환

### 3-1. 감속비

$$\tau_{raw,i} \;=\; k_i^{\,n-1}\,\tau_{ch,i}$$

지수가 $n-1$ 인 이유는 §5 참고. **`n=2` 면 배율이 $k$, `n=1` 이면 $1$**(감속비 보정이 사라진다).

### 3-2. 커플링 전치

커플링을 $q_{joint} = A\,q_{raw}$, $A = I - C$ 로 쓰면 ($C$ 는 `(foot, calf)` 성분만 $+1$):

$$q_{joint,foot} = q_{raw,foot} - q_{raw,calf}, \qquad q_{joint,calf} = q_{raw,calf}$$

일률 보존 $\tau^{\top}\dot q$ 불변에서 **정방향**(관절 → 모터 명령):

$$\tau_{raw} = A^{\top}\tau_{joint}
\quad\Longrightarrow\quad
\tau_{raw,calf} = \tau_{joint,calf} - \tau_{joint,foot}$$

우리가 쓸 **역방향**(드라이브 → 관절)은 $A^{-\top} = (I+C)^{\top}$ 이므로:

$$\boxed{\;\tau_{joint,calf} = \tau_{raw,calf} + \tau_{raw,foot}, \qquad
\tau_{joint,foot} = \tau_{raw,foot}\;}$$

> ⚠ **부호 혼동 주의.** `calib_bipedleg.hpp:25` 의 `τ_raw_calf −= τ_foot` 은 **정방향**이고
> 위 `+` 는 **역방향**이다. 같은 식의 양쪽이라 모순이 아니다. 방향을 안 적으면 반드시 헷갈린다.

### 3-3. 합친 식 (실제 적용한 것)

$$\tau_{joint,i} = k_i^{\,n-1}\,\tau_{ch,i} \quad (\text{hip, thigh, foot})$$
$$\tau_{joint,calf} = k_{calf}^{\,n-1}\,\tau_{ch,calf} \;+\; k_{foot}^{\,n-1}\,\tau_{ch,foot}$$

`n=2` 를 넣으면 (현재 기본값):

```
τ_joint_hip   = τ_ch_hip
τ_joint_thigh = τ_ch_thigh
τ_joint_calf  = 1.5·τ_ch_calf + 1.2·τ_ch_foot     ← 전치항
τ_joint_foot  = 1.2·τ_ch_foot
```

### 3-4. 수치 예제

채널기준 입력 (leg-major 8-벡터):

| 슬롯 | 0 HL_hip | 1 HL_thigh | 2 HL_calf | 3 HL_foot | 4 HR_hip | 5 HR_thigh | 6 HR_calf | 7 HR_foot |
|---|---|---|---|---|---|---|---|---|
| 채널 τ | 1.0 | 2.0 | 10.0 | 5.0 | 1.0 | 2.0 | 20.0 | 4.0 |
| **관절 τ** | 1.0 | 2.0 | **21.0** | **6.0** | 1.0 | 2.0 | **34.8** | **4.8** |

- HL_calf $= 1.5\times10.0 + 1.2\times5.0 = 21.0$
- HL_foot $= 1.2\times5.0 = 6.0$
- HR_calf $= 1.5\times20.0 + 1.2\times4.0 = 34.8$

---

## 4. 전치항이 필요한 근거 (실측)

전치항을 넣을지는 **sim 의 `applied_torque` 가 effort target 을 계상하는가**에 달렸다.
sim 은 전치분을 `set_joint_effort_target_index` 로 calf 에 싣는다(`_apply_foot_coupling`).

**판정법**: calf 게인을 $k_p=k_d=0$ 으로 두면 calf 액추에이터 PD 기여가 정확히 0 이므로,
`applied_torque[calf]` 에 남을 수 있는 것은 effort target 뿐이다.

| `foot_transpose` | `applied_torque[calf]` (HL/HR) | 전치항 예측 | 비 |
|---|---|---|---|
| ON | −2.4988 / −1.4744 | −2.4710 / −1.4504 | **1.011 / 1.017** |
| OFF | **0.0000 / 0.0000** | — | — |

⇒ **계상된다.** sim calf 에는 전치항이 들어 있다. 반면 실기 보고값은 **지령 토크 에코**라
calf 채널 자신의 PD 법칙만 담고 전치항이 없다. 그래서 실기를 올릴 때 전치항을 더해 줘야 맞는다.

> ### ⚠ `foot_transpose` True/False A/B 는 이 질문에 답하지 못한다
>
> 전치를 켜면 **운동 자체가 바뀌므로** calf 토크는 어차피 달라진다. 관측된 Δcalf 가 "전치항"인지
> "궤적 발산"인지 분리되지 않는다. 실측 $\Delta\tau_{calf}/\tau_{foot}$:
>
> | | HL | HR |
> |---|---|---|
> | 40 스텝 | +1.047 | **−0.010** |
> | 3 스텝 | −3.978 | −0.791 |
>
> 전치가 실리면 $\approx +1$ 이어야 하는데 **다리마다·스텝수마다 제각각**이다(40스텝 HL 은 우연히
> +1 에 가깝지만 같은 조건 HR 은 0 이다). 즉 이 설계로는 아무것도 확정할 수 없다 —
> **게인을 0 으로 죽이는 설계라야 분리된다.**

재현: `reports/_comparisons/pace_bipedleg_foot_coupling_probe/logs/transpose_in_applied_torque.py --headless`

---

## 5. 지수 $n$ — 왜 $k^{n-1}$ 인가, 그리고 왜 브리지가 아닌가

드라이버 PD 가 **채널각 오차**에 $k_p$ 를 곱한다는 것은 실측 확정이다 (준정적 구간,
$|\tau|/|\text{pred}_{naive}|$ — `pred_naive` 는 관절각으로 계산한 순진한 PD 예측):

| 관절 | $k$ | 실측 비 | vs $k$ | vs $k^2$ |
|---|---|---|---|---|
| HL_hip | 1.0 | 1.0008 | 0.08% | 0.08% |
| HL_thigh | 1.0 | 1.0029 | 0.29% | 0.29% |
| **HL_calf** | 1.5 | **1.5008** | **0.06%** | −33.30% |
| HR_hip | 1.0 | 1.0006 | 0.06% | 0.06% |
| HR_thigh | 1.0 | 1.0014 | 0.14% | 0.14% |
| **HR_calf** | 1.5 | **1.4992** | **−0.05%** | −33.37% |

즉 보고 토크 $= k_p\,\Delta q_{ch} = k_p\,k\,\Delta q_{joint}$ — **정확히 $k^1$**($k^2$ 는 −33% 로 배제).
이것이 지수의 ① 조각(**실측**)이다.

**하지만** 그 숫자가 *채널측 토크*인지 *이미 관절측 토크*인지는 안 정해졌다 — ② 조각(**유도**).
채널측이면 $\times k$ 해야 관절토크($n=2$), 이미 관절측이면 그대로($n=1$). 그래서 감속비 배율이
$k^{n-1}$ 이고, **$n$ 은 이 세션에서 "현 데이터로 원리적 판정 불가"로 확정된 바로 그 미지수**다.

$n=2$ 로 두면 두 유도가 일치한다:

$$\tau_{joint} = k\,\tau_{ch} = k_p\,k^2\,\Delta q_{joint} = k_{p,\text{eff}}\,\Delta q_{joint},
\qquad k_{p,\text{eff}} = k_p\,k^2$$

즉 컨버터의 `GAIN_GEAR_SCALE = 2.0` 과 **같은 베팅**이다.

### 그래서 변환을 브리지에 두지 않는다

브리지가 $\times k$ 를 해 버리면 **미판정 지수에 대한 베팅이 선로 위 데이터에 구워져** 되돌릴 수 없다.
raw 로 통과시키면 소비자가 나중에 어느 쪽으로든 갈 수 있다 — 즉 안 하는 게 반대편 베팅이 아니라
**베팅 유보**다. (브리지가 이 변환까지 하게 되면 `convention_version` 은 **2** 가 된다.)

추가로 tau 는 **제어 경로에 없다**: `POLICY_STATE` 에 tau 필드 자체가 없고, `fit_bipedleg.py` 의
목적함수에도 tau 참조가 0 건이다. 컨버터에서 유일하게 쓰는 곳(지연 추정)은 hip/thigh 4관절만
쓰는데 그 축들은 $k=1$ 이라 애초에 영향이 없다.

---

## 6. ⚠ 순서 함정 — leg-major 와 articulation 은 짝이 다르다

| 순서 | 배치 | foot 의 calf 짝 |
|---|---|---|
| **leg-major** (UDP · monitor · `motions.JOINT_NAMES`) | HL{hip,thigh,calf,foot} → HR{…} | **`i−1`** |
| articulation (Isaac 내부) | {HL,HR}hip → {HL,HR}thigh → … | `i−2` |

`gui_controller` 가 중계 전에 재배열한다:
`tau_lm = [t["tau"][a] for a in _ART_FOR_LEGMAJOR]`.

**`i−2` 규칙을 leg-major 에 쓰면 반대 다리 토크를 더한다.** `r2s_udp.py` 자체 테스트에 이 경우가
들어 있다 — HL_foot 만 100 을 넣고 HR_calf 가 0 으로 남는지 본다.

---

## 7. 구현 위치

| 대상 | 위치 |
|---|---|
| 변환 함수 | `r2s_udp.channel_tau_to_joint(tau_ch, gain_exponent=2.0)` |
| 상수 | `r2s_udp.GEAR_K` (leg-major), `r2s_udp.DEFAULT_GAIN_EXPONENT` |
| 적용 | `monitor.py` `_on_recv_tick()` — 실기 8-벡터에 적용 후 관절 선택 |
| 표시 | tau 축 라벨 `Joint torque [Nm] (real lifted, n=2)` |

★환산은 **8-벡터 전체에** 해야 한다. calf 는 같은 다리 foot 값이 필요하므로 관절 선택(`i`) 뒤에
하면 계산 자체가 불가능하다.

검증 (2026-08-14 실행):

| | 채널 입력 | 표시값 | 기대 |
|---|---|---|---|
| HL_calf | 10.0 | 21.0 | $1.5\times10 + 1.2\times5$ |
| HR_calf | 7.0 | 10.5 | $1.5\times7$ (좌우 안 섞임) |

```bash
python3 scripts/real2sim/r2s_biped_leg/r2s_udp.py   # 자체 테스트 (환산 + 좌우 분리 포함)
```

---

## 8. 아직 안 정해진 것

- **$n \in \{1,2\}$** — 판정 경로는 "발끝 기지질량 + 준정적 디더" 캡처 하나뿐이다
  (`NEXT_CAPTURES.md` 캡처 A). 그전까지 `n=2` 는 **채택이지 확정이 아니다**.
- **전치 토크의 크기** — 존재는 기구 구속에서 강제되지만 크기는 미실측 (캡처 C: calf 단독 chirp).
- ⚠ `real_runner/r2s_packets.hpp:36` 주석이 *"채널기준으로 **실측 확정** — 소비자가 ×gear 하면
  관절토크"* 라고 적어 두었는데, 실측된 것은 ① 이고 "×gear 하면 관절토크"는 ② **유도**다.
  같은 명제를 이 문서와 보고서는 `⚠ 유도, 미실측` 으로 매긴다 — **등급이 어긋나 있다.**

### 안전 — 토크 트립도 채널기준이다

설정 15 N·m 이 채널기준이라 **calf 실제 관절토크는 22.5 N·m**, foot 은 18 N·m 이다.
트립을 관절 기준으로 착각하면 15 에서 보호받는다고 믿지만 관절은 22.5 까지 간다.

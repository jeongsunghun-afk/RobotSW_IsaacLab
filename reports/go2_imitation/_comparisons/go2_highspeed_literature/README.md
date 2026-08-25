# Go2 고속 주행 선행연구 조사 — 다들 액추에이터를 어떻게 쓰나

**요청 (2026-08-10)**: "go2를 이용해서 고속으로 달리는 것에 대한 연구들을 찾아보고 거기서는
어떻게 사용했는지 우선 조사"

**결론**: **실기 Go2 는 4 m/s 를 낸다.** 우리 1.72 m/s 천장은 로봇의 한계가 아니라
**우리 액추에이터 모델이 만든 한계**다. 직전에 내렸던 "MimicKit 의 4 m/s 는 sim 아티팩트"
결론을 **철회한다** — 모터 데이터시트의 24 V 를 로봇 버스 전압으로 착각한 계산이었다.

---

## 1. 실기 Go2 가 4 m/s 를 낸다는 독립 근거 두 개

| 근거 | 내용 |
|---|---|
| **Unitree 공식 사양** | Go2 **EDU 최고 5 m/s**, Pro 3.5~3.7 m/s, Air 2.5 m/s |
| **arXiv:2602.00678** (MoE, 2026) | **물리 로봇에서 4.01 m/s**, 2.16 s 내 도달 (§VI-B) |

MoE 논문의 설정: IsaacGym 학습 / MuJoCo 평가, **kp 20 · kd 0.5**, 제어 50 Hz · sim 200 Hz.
**관절별 토크 한계도 속도 한계도 토크-속도 곡선도 명시하지 않는다.** DR 은
`Actuator strength [0.8, 1.2] × Nominal` 하나뿐. 고속에서의 모터 포화 언급도 없다.
→ **토크-속도 모델 없이도 실기 4 m/s 가 나온다.**

## 2. 내 계산이 틀렸던 지점 — 버스 전압

`GO-M8010-6` 데이터시트의 "Maximum Revolution Speed **30 rad/s (24V** power supply)" 를
그대로 썼는데, **Go2 배터리는 29.6 V 정격 / 33.6 V 충전 상한**이다(Unitree Go2 Battery 매뉴얼;
"모터 효율·출력·안정성 개선을 위해 28.8 V 로 상향" 이라는 기술도 있다).
무부하 속도는 전압에 선형이므로 24 V 기준 값은 실제보다 **20~40% 낮다**.

### 다시 계산 (`logs/motor_operating_region.py`)

데이터시트 "30 rad/s @24V" 는 두 가지로 읽힌다:

| 해석 | hip·thigh @29.6 V | calf @29.6 V |
|---|---|---|
| **(a)** 30 = 무부하 속도 | ω₀ = **37.0** rad/s | ω₀ = **19.3** |
| **(b)** 30 = 정격토크 유지 코너 | 코너 **38.8** / ω₀ **46.3** | 코너 **20.2** / ω₀ **24.2** |

`K_t = 0.63895 Nm/A`(출력축, `K_t·I_max = 25.6 ≈ 최대토크 23.7` 로 확인),
calf 는 같은 모터 + **1.917:1** 추가 감속(`45.43/23.7`, `30.1/1.917 = 15.70` ✓).

### MimicKit 작동점(thigh q̇ 20 / calf q̇ 15)에서 낼 수 있는 토크

| 모델 | thigh@20 | calf@15 |
|---|---:|---:|
| **현행 cfg** (23.5/30.1, 35.5/30.0) | **7.89** | 17.75 |
| (a) 29.6 V 선형 | 10.89 | 10.13 |
| (b) 29.6 V 코너까지 평탄 | **23.70** | **45.43** |
| MimicKit `ImplicitActuator` (평탄) | 23.70 | 35.50 |
| *MimicKit 이 실제로 쓰는 값* | *23.0* | *32.0* |

**(b) 해석이면 실기가 그 토크를 낼 수 있다.** 실기 4.01 m/s 라는 경험적 앵커가 (b) 쪽을
지지한다. 현행 cfg 는 thigh 를 **34%** 로 깎는다.

⚠ (a)/(b) 를 확정하려면 권선 저항 `R` 이 필요하다(데이터시트에 없음). 실기 스톨 전류·무부하
spin 측정이면 결정된다.

## 3. 선행연구가 액추에이터를 다루는 방식

| 연구 | 액추에이터 처리 | 결과 |
|---|---|---|
| **MoE Go2** (arXiv:2602.00678) | **모델 없음.** `actuator strength ×[0.8,1.2]` DR 만 | 실기 **4.01 m/s** |
| **MimicKit** | `ImplicitActuator(effort_limit=None)` → USD `maxForce` 평탄 (23.7 / calf 35.5), **속도 제한 없음** | sim 4.0 m/s |
| **KAIST Hound** (arXiv:2312.17507) | **모터 작동영역(motor operating region) 명시 모델링**: `−V_bus ≤ (R/K_t)τ + ω/K_v ≤ V_bus` + peak torque cap(선형에서 20% 이탈 지점), **4 사분면 전부**(회생제동 포함). 파라미터 = 데이터시트 + 단축 벤치 실측. V_bus 81 V | 실기 **6.5 m/s** (트레드밀), 야외 5.9 m/s |
| **IsaacLab 기본** | `DCMotorCfg` Go2 `23.5/23.5/**30.0**`, A1 `33.5/33.5/**21.0**` | — |

### ★ KAIST Hound 의 ablation 이 핵심이다

- **작동영역 모델 없이**: sim 6.5 m/s 인데 실기는 5 m/s 에서 낙상, 3.5 m/s 부터 sim2real 격차 급증
- **작동영역 모델 있으면**: 실기 6.5 m/s. ablation 표에서 없으면 최고 4.5 vs 있으면 6.5

→ 토크-속도 제약은 **넣는 것 자체가 아니라 맞게 넣는 것**이 이득이다. 그리고 그들의 작동영역 모델은
**평탄 구간 + 하강 구간의 2 구간**이지 0 rad/s 부터 선형 하강이 아니다.

### ★ IsaacLab 의 체계적 관행이 문제다

Go2 `23.5 × 30.0`, A1 `33.5 × 21.0` — 둘 다 **"스펙 최대토크 × 스펙 최대속도"** 를 그대로
`DCMotorCfg` 에 넣었다. 그런데 `velocity_limit` 의 의미는 **무부하 속도(x 절편)** 다.
정격 최대속도를 x 절편에 넣으면 "모터가 자기 정격 속도에서 토크 0" 이 되어 곡선이 과도하게
가파르다. **로봇 두 종에 걸친 같은 패턴이므로 우리 task 만의 실수가 아니다.**

또한 `saturation_effort == effort_limit` 이라 **정전류 평탄 구간이 없다** — 실제 PMSM 은
저속에서 평평하다가 코너에서 꺾인다. KAIST 식 2 구간 모델과 형태가 다르다.

## 4. 우리에게 주는 처방

| 안 | 내용 | 성격 |
|---|---|---|
| **최소 수정** | `velocity_limit` 을 로봇 전압 기준 무부하 속도로: hip·thigh **37~46**, calf **19~24** | 지금 곡선의 기울기 오류만 제거 |
| **정석** | `saturation_effort > effort_limit` 로 **평탄+하강 2 구간** 구성 (KAIST 식). `R` 실측 필요 | 물리적으로 옳음 |
| **재현/상한 확인** | `ImplicitActuatorCfg(kp 25, kd 1.0, effort_limit=None)` — MimicKit·MoE 와 같은 조건 | "플랜트가 전부였나" 를 가림 |

⚠ calf `effort_limit` 도 공식 URDF 는 **45.43**(현행 35.5 는 MimicKit 자산 값). 다만 이건
`saturation`/`velocity_limit` 을 고친 뒤에 별도로 볼 것 — 한 번에 두 개를 바꾸면 또 분리가 안 된다.

## 5. 철회

**"MimicKit 의 4 m/s 는 실기 불가능한 sim 아티팩트"(2026-08-10) 를 철회한다.**
근거였던 무부하 속도 19.6 rad/s(calf)는 **24 V 가정**의 산물이고, 로봇은 29.6 V 로 돈다.
실기 Go2 는 4 m/s 를 내며(MoE 논문, Unitree 사양) 그 gait 는 물리적으로 가능하다.

**출처**
- [Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion (arXiv:2602.00678)](https://arxiv.org/html/2602.00678)
- [Actuator-Constrained RL for High-Speed Quadrupedal Locomotion — KAIST Hound (arXiv:2312.17507)](https://arxiv.org/html/2312.17507)
- [GO-M8010-6 Motor User Manual V1.0](https://techshare.co.jp/faq/wp-content/uploads/2023/12/GO-M8010-6_Motor_Data_User_Manual_V1.0.pdf)
- [Go2 Battery and Charger User Manual](https://static.generation-robots.com/media/Go2-Battery-and-Charger.pdf)
- [Unitree Go2 제품 페이지](https://www.unitree.com/go2/)
- 로컬: `/home/lgb/go2_description/urdf/go2_description.urdf`, `source/isaaclab_assets/.../unitree.py`

관련: [[project_go2_effort_limit_blocks_high_speed]]

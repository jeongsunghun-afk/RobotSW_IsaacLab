# Go2 Sim-to-Real Gap 비교 실험 계획 제안서

> 작성: 2026-06-15 | 대상: IsaacLab(Isaac Sim) ↔ Unitree Go2 real robot
> 목적: "동일 정책/명령을 sim과 real에 적용했을 때 무엇이 다른가"를 **계층별로 격리·정량화**하는 실험 설계
> 산출 근거: 코드베이스 조사(A) + 문헌 조사(B) + 실험설계 조사(C) 종합

---

## 0. 핵심 원칙 (먼저 합의)

1. **Gap은 단일 수치가 아니라 계층이다.** `actuator → single-joint → leg → full-body → policy` 순으로 분해해야 원인 격리가 된다. ANYmal(Hwangbo 2019)·PACE(Braun 2025) 모두 "actuator를 먼저 닫는다"가 표준.
2. **비교의 전제는 "동일 입력의 시간 정렬"이다.** 같은 명령을 같은 시각에 줬다는 보장이 없으면 어떤 metric도 무의미 → §4(인프라/동기화)가 사실상 1순위 선행 작업.
3. **Go2는 위치명령(target q)→내부 PD→QDD 모터 구조다.** sim의 PD게인/actuator 모델이 real Go2 내부 PD와 일치하는지가 1차 gap 소스. (Go2는 토크센서 없음 → encoder-only SysID가 현실적)
4. **프로젝트 제약 준수:** contact sensor를 **정책 obs에 넣지 않는다**(sim-to-real). 아래 실험의 발접촉 측정은 전부 **평가용 측정**이지 정책 입력이 아니다. input 팽창·env 재작성 금지 원칙도 유지.

---

## 1. Gap 원천 분류 (7대 카테고리)

| # | Gap 원천 | 중요도 | 본질 |
|---|---------|-------|------|
| G1 | **Actuator dynamics** | ★★★ | sim=이상적 토크소스 가정 vs real=저역통과+비선형포화+마찰+back-EMF. 단일 최대 원천 |
| G2 | **Latency / 통신지연** | ★★★ | 명령→실현 지연. 문헌 식별값 Td≈7.5ms, DR범위 [0,40]ms. 50Hz정책↔500Hz저수준 주파수차도 유효지연 생성 |
| G3 | **질량/관성/CoM 오차** | ★★ | CAD 과소평가(등가관성 실측 ≈4×), 배터리/페이로드. base mass·CoM·관성 |
| G4 | **접촉/마찰 모델** | ★★ | contact solver 강성·감쇠·μ 오차 → 지지력·보행타이밍 직격 |
| G5 | **센서 노이즈/바이어스** | ★ | IMU drift·encoder 노이즈·속도 수치미분 노이즈 |
| G6 | **제어주파수 불일치** | ★ | sim 200Hz물리/50Hz정책 vs real 500Hz저수준. ZOH 배포 시 보간오차 |
| G7 | **관절한계·backlash·온도** | ½ | 토크포화 형태차, 기어유격 dead-zone, 장시간 토크저하 |

---

## 2. 현재 우리 코드의 현황 — "gap의 구멍" 진단

조사 A 결과 기준. **각 항목이 sim에서 어떻게 모델링/랜덤화되는지**를 gap 카테고리에 매핑.

### 2.1 Actuator (G1)
- Go2 기본: `DCMotorCfg`, **Kp=25.0 / Kd=0.5**, effort/saturation=23.5 N·m, vel_limit=30 rad/s, armature=0.01, friction=0.0
  - 파일: `source/isaaclab_assets/.../robots/unitree.py:140-184`
- parkour는 `_actuator_mode=2`로 동일값 유지(`parkour_env_cfg.py:488-576`)
- **구멍 ①:** 문헌에서 보고된 Go2 배포 PD는 **Kp≈35 / Kd≈0.52**인데 우리 sim은 **Kp=25 / Kd=0.5**. → real 내부 PD와 sim PD가 다를 가능성. (단, 우리는 의도적으로 약화된 actuator로 학습 성공 이력이 있음 → §6 주의 참조, 실측 전 단정 금지)
- **구멍 ②:** `friction=0.0` (관절 쿨롱/점성 마찰 미모델링). real QDD는 마찰 존재 → 작은 명령에서 sim은 움직이고 real은 안 움직이는 정지마찰 gap 예상.
- **구멍 ③:** **action delay/latency 모델링 코드 없음**(A에서 "확인 안 됨"). G2 전체가 sim에서 누락.

### 2.2 Domain Randomization (G3/G4/G1)
환경별 편차가 큼:

| DR 항목 | Go2(WTW) | Parkour | Go2 Imitation(+Tracking) |
|--------|----------|---------|--------------------------|
| friction | (0.6,0.6) 고정 | (0.3,1.2) | **없음** |
| base mass | -1~+10kg | -1~+3kg | **없음** |
| CoM | ±0.05~0.15 | ±0.02~0.08 | **없음** |
| actuator gain | log_uniform ±50% | uniform **±2.5%** | **없음** |
| push | 없음 | interval 8s, ±0.5m/s | **없음** |
| action noise | std 0.05 | 없음 | 없음 |
| obs noise | std 0.002 | 없음 | 없음 |

- **구멍 ④:** parkour actuator gain DR이 **±2.5%로 극히 좁음** (약화 actuator 불안정 우려로 보수설정). G1 robustness 여지 적음.
- **구멍 ⑤:** parkour friction이 (0.3,1.2) — 문헌 권장 μ∈[0.5,2.0] 대비 하한이 낮고 상한이 좁음.
- **구멍 ⑥:** **go2_imitation / tracking 계열은 DR이 전무**. AMP의 암묵 robustness에만 의존 → sim-to-real 위험 최고.
- **구멍 ⑦:** latency DR 없음(전 환경 공통, G2).

### 2.3 Observation (G5)
- privileged 분리는 잘 되어 있음: parkour `priv_latent`(37-dim: base/foot friction·mass·CoM·stiffness/damping ratio)는 **real에서 제거** 설계. height_scan도 privileged.
- **구멍 ⑧:** parkour/imitation 계열에 **obs noise 미적용**(Go2 WTW만 std 0.002). real IMU/encoder 노이즈 분포 미반영.

### 2.4 제어주파수 (G6)
- 전 환경 공통: 물리 200Hz / 정책 50Hz / decimation=4 / step_dt=0.02s.
- **점검 필요:** real 배포 시 정책 주파수를 **50Hz로 일치**시키는지, 저수준 ZOH가 맞는지 deploy 코드 확인 대상.

### 진단 요약
> **가장 큰 구멍 = G2 latency 완전 누락 + go2_imitation 계열 DR 전무 + 관절마찰 0.** 이 셋이 sim-to-real 실패의 최우선 후보. 단, PD게인(구멍①)은 학습성공 이력이 있어 **실측 전 root cause로 단정 금지**.

---

## 3. 실험 매트릭스

### 3.1 Open-loop 실험 (정책 없이 — 순수 dynamics gap 격리)

| # | 실험 | 목적/격리 gap | sim 측정 | real 측정 | 비교 metric |
|---|------|--------------|----------|-----------|-------------|
| O1 | static hold-pose | G1/G3 정지마찰·중력보상 | 정착 q, 토크 | SDK q, 추정 tau | steady-state pos error, 토크 bias |
| O2 | single-joint **step response** | G1 PD추종 동특성 | q(t),dq(t),tau(t) | 동일(100Hz로깅) | rise/settling time, overshoot%, q(t) RMSE |
| O3 | single-joint **chirp sweep**(0.1–10Hz) | G1+G2 대역폭·**latency** | q_des↔q gain/phase | 동일 | Bode mag/phase RMSE, -3dB BW, **phase→Td 추정** |
| O4 | swing/pendulum(free-swing) | G1 마찰·댐핑·관성 | 감쇠 envelope·주기 | encoder 동일 | decay rate(d), 주기(Iₐ), 마찰계수 |
| O5 | const-velocity sweep | G1 torque-velocity map | 각 dq의 정상 tau | 추정 tau/전류 | tau-dq 곡선 RMSE, Coulomb/viscous 계수 |
| O6 | IMU 정지 노이즈 | G5 센서모델 | DR 노이즈 std | acc/gyro std·bias, Allan분산 | per-axis noise std, bias offset |
| O7 | drop/free-fall | G4 접촉강성·반발+G3 질량 | base z(t), 임펄스 | base z(외부추적), 발force | bounce height, peak impact force |
| O8 | battery sag(SoC별 O2) | G7 전압강하→토크저하 | (DR motor strength 근사) | 전압·전류·실현토크 vs SoC | 토크 vs 배터리% 곡선 |

### 3.2 Closed-loop 실험 (동일 RL 정책 배포)

> real 직전 **sim-to-sim**(다른 DR/seed)로 예측가능성 먼저 확인(RoboGauge 식).

| # | 실험 | 목적/격리 | 비교 metric |
|---|------|----------|-------------|
| C1 | command tracking 정속 vx∈{0.5,1.0,1.5} | 속도추종 transfer | lin/ang vel tracking RMSE, steady bias |
| C2 | step/ramp 명령 | 과도 vs 정상 분리 | settling time, overshoot, ss error |
| C3 | posture stability(정지·정속) | 자세안정 품질 | base height/roll/pitch의 std, drift |
| C4 | gait/contact 패턴 | 보행위상 일치 | duty factor차, Hildebrand 위상거리, contact-timing 상관 |
| C5 | 에너지 / **CoT** | G1 손실 누락정도 | CoT비(real/sim)=∫max(0,τ·dq)/(mgd) vs Vi/(mgv) |
| C6 | survival/robustness(다seed/terrain) | 누적 gap | survival time비, success-rate차, fall rate |
| C7 | push recovery(정량 임펄스) | 강인성 마진 | recovery time, max 이탈, 회복성공률 |
| C8 | terrain 일반화(램프/계단/저μ) | G4 terrain의존 | 지형별 C1·C6의 sim-real 차 |

**공통 기록 신호:** target q, q, dq, 추정 tau, (real)전류·온도, IMU(quat/ang_vel/lin_acc), foot force, battery SoC, timestamp.

---

## 4. 측정 인프라 & 동기화 (성패 좌우)

**real Go2(Unitree SDK2, CycloneDDS):** LowState의 12모터 q/dq/추정tau/온도/전류, IMU quat·gyro·accel, foot force(추정→임계 캘리브 필요), battery 전압·전류·SoC. 보통 50Hz target-q 배포.

**sim 대응(IsaacLab):** `articulation.data`의 joint_pos/vel/applied_torque, root_quat/ang_vel/lin_acc, contact sensor net_forces_w, step_dt 시각.

**동기화 체크리스트:**
1. **동일 입력 재생** — open-loop는 명령 시퀀스를 파일화해 양쪽 재생, closed-loop는 동일 명령 프로파일.
2. **이벤트 정렬** — 절대시각 대신 명령 인가순간을 t=0으로.
3. **공통 그리드 리샘플(200Hz) + ZOH** 후 metric 계산.
4. **latency 정렬** — O3에서 추정한 Td만큼 real shift한 "지연보정 RMSE"와 "raw RMSE" **둘 다 보고** → G2와 G1 분리.
5. **좌표계/단위 통일** — IMU 축·회전순서, q 부호·영점, 토크 단위, world vs base frame.
6. **외부 ground truth 권장** — base pose/vel은 IMU+odom drift → 가능하면 mocap.

---

## 5. 단계별 실험 순서 (격리 쉬운 것 → 어려운 것)

```
1. O6 (IMU 정지노이즈)        ── 센서만, 최易
2. O1 (static hold)          ── 중력보상·정지마찰
3. O2→O3→O4→O5 (single-joint) ── PD/actuator/마찰/관성/latency 식별 ★actuator를 여기서 최대한 닫기
4. O7 (drop)                 ── 접촉모델 격리
5. O8 (battery)              ── 토크한계
6. sim-to-sim (C1·C6)        ── real 전 예측가능성
7. C1/C2 (속도추종)           ── full-body 단순명령
8. C3/C4/C5 (자세·gait·에너지) ── 보행품질
9. C7/C8 (외란·지형)          ── 최통합·고위험, 마지막
```
각 단계: gap 측정 → sim 모델/DR 조정 → 재측정 루프. 상위 실패 시 항상 하위(특히 3번 actuator)로 회귀.

---

## 6. 처방 (gap 측정 후 적용)

| Gap | 측정법 | 처방 |
|-----|--------|------|
| G1 actuator | O2/O3/O4/O5 | **PACE/SPI-Active식 encoder-only SysID**(공중고정+chirp 20~60초로 {Iₐ,d,τf,Td} 식별) → sim armature/damping/friction을 식별값으로 교체. 그 후 DR 범위 ±20%로 좁힘 |
| G2 latency | O3 phase lag | action delay buffer 추가 + DR [0,40]ms episodic + [-5,5]ms per-step |
| G3 mass/CoM | O7, swing | base mass [0.7,1.3]×, payload [-1,3]kg DR |
| G4 friction | O7, slip | μ~U(0.5,2.0)로 확대(현 parkour 0.3~1.2 조정 검토) |
| G5 sensor | O6 Allan | parkour/imitation에 obs noise injection 추가(훈련 중반부터) |
| G6 freq | timing log | deploy 50Hz 일치 + 500Hz ZOH 확인 |
| G7 backlash/온도 | O5, O8 | tanh 토크포화 모델(SPI-Active, forward jump 오차 -45%), motor strength DR로 온도 흡수 |

**우선순위 처방 (현 코드 구멍 기준):**
1. **latency 모델링 추가**(G2, 전 환경 누락) — 가장 임팩트 큰 누락.
2. **go2_imitation/tracking에 기본 DR 세트 추가**(friction/mass/CoM/gain) — 현재 전무.
3. **관절 마찰 비0화 + actuator SysID** — O2~O5 실측 후.

---

## 7. 주의 / 프로젝트 정합성

- **PD게인(Kp25/Kd0.5)을 실측 전 gap root cause로 단정 금지.** 약화 actuator로 학습 성공 이력 존재(메모리 `project_parkour_actuator_mode2_verified`). 토크 envelope 정량추론은 실측 앞에서 Tier2 아님.
- **contact는 평가용 측정만**, 정책 obs 추가 금지(sim-to-real 제약).
- **input 팽창·env 재작성 금지** — 처방은 actuator param·DR·latency 등 additive only.
- 모든 실험은 **조사/제안**이며 코드 변경 미포함. real Go2 하드웨어 가용성에 따라 O계열(실측) 착수.

---

## 참고문헌
- Tan et al. 2018, *Sim-to-Real: Learning Agile Locomotion for Quadruped Robots* (RSS) — DR+actuator+latency 3종 조합
- Hwangbo et al. 2019, *Learning agile and dynamic motor skills* (Science Robotics) — actuator net(SEA, 토크센서 필요)
- Braun et al. 2025, *PACE* (arXiv:2509.06342) — encoder-only SysID, Td≈7.5ms, 등가관성 4×
- Zhao et al. 2025, *SPI-Active* (arXiv:2505.14266) — **Go2 특화**, tanh 포화, FIM active exploration, 42~63%↑
- Kumar et al. 2021, *RMA* (RSS) — teacher-student latent adaptation, A1 zero-shot
- He et al. 2025, *ASAP* (RSS) — residual delta action model, G1 -52.7%
- RoboGauge / Hildebrand(1965) gait 정량 / CoT 측정법

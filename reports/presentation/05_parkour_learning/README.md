# Parkour Learning — 사족보행 로봇의 3D 지형 극복 학습

**대상 로봇**: Unitree Go2 (12 DOF 사족보행)
**기간**: 2026-04-29 ~ 2026-08-18 (약 4개월, 세션 보고서 60여 건)
**플랫폼**: IsaacLab (Isaac Sim 5.1 → 6.0), rsl_rl
**최종 산출물**: 실기 배포 가능한 LiDAR 기반 3D 지형 극복 정책 (`Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0`)

---

## 대표 영상 (Showcase)

[프로젝트 대표 영상](assets/videos/00_parkour_showcase.mp4) — 6개 지형(flat · hurdle · step · gap · stair · crawl)의 student policy 주행을 6분할 화면으로 동시 재생합니다 (각 셀 하단 라벨 참조).

## 0. 한눈에 보기

이 프로젝트는 "로봇이 카메라·LiDAR로 앞을 보고, 계단·구멍·장애물·터널을 스스로 넘어가게 만든다"는 하나의 목표를 4개월에 걸쳐 다섯 단계로 밀어붙인 기록이다.

가장 중요한 결과 세 가지를 먼저 적는다.

1. **실기에 옮길 수 있는 센서(Livox Mid-360 LiDAR)만으로, 특권 정보를 쓰는 시뮬레이터 전용 teacher와 사실상 대등한 성능을 냈다.** 잔여 격차가 통계적으로 유의한 지형은 stair 하나뿐이다. 6개 지형 전부에서 전진속도 1.0 m/s 안팎, 계단 등반 높이도 1.541 m vs 1.556 m로 동일하다.
2. **그 돌파구는 "더 좋은 센서"가 아니라 "더 좋은 표현(representation)"이었다.** 같은 LiDAR 히트를 각도 range image 대신 미터 단위 3D 점유 격자로 바꾸자, 관측 차원이 13,824 → 7,371로 **작아지면서** 성능이 올랐다.
3. **teacher-student distillation을 결국 버리고 from-scratch RL로 돌아왔다.** distillation은 구조적으로 넘을 수 없는 천장(conditional-mean ceiling)을 물려받는데, 배포 가능한 표현을 확보한 뒤에는 그 천장을 우회할 이유가 없었다.

### 진화 타임라인

| 단계 | 시기 | 무엇을 했나 | 해결한 큰 문제 |
|---|---|---|---|
| **1. 기본 환경 구축** | 2026-04~05 | Isaac Gym/Genesis 기반 parkour를 IsaacLab Direct Env로 이식 | 학습이 아예 안 되던 3대 원인 규명 (actuator 한계·reward weight·"정지+정면" local optimum) |
| **2. Symmetry 도입** | 2026-06 | L/R mirror data augmentation | 3-leg gait — 왼쪽 뒷다리 한 개를 계속 접고 다니는 고착 평형 |
| **3. Imitation (AMP) 결합** | 2026-06 | Adversarial Motion Prior로 보행 스타일 학습 | 부자연스러운 pronk/split-jump 보행. estimator 누락으로 배포 불가였던 구조 결함도 함께 수정 |
| **3.5. EasyEntry** | 2026-07 | 커리큘럼 level 0 하한 인하 | Isaac Sim 6.0에서 from-scratch 학습이 장애물을 **하나도** 못 넘던 문제 (커리큘럼 재진입 문턱) |
| **4. 3D GT perception** | 2026-06~08 | height scan(2.5D) → clearance / voxel 점유 격자(3D) | 터널·오버행처럼 한 (x,y)에 바닥과 천장이 동시에 있는 지형을 2.5D로는 **원리적으로 표현 불가** |
| **5. LiDAR 전환** | 2026-07~08 | range image student 붕괴 → distillation 실험 → SL-Grid from-scratch RL | 특권 정보 없이 실기 센서만으로 학습. 병목이 센서가 아니라 **표현**임을 격리 실험으로 확정 |
| **6. Symmetry 재결합** | 2026-08 | LiDAR 격자에 y-flip mirror 적용 | RL_calf 관절 노이즈 std 폭주(16.7, 타 관절의 40배) 구조적 소멸 + 수렴 속도 약 2배 |

---

## 1. 배경 — parkour learning이란 무엇이고 왜 어려운가

### 1.1 과제 정의

일반적인 사족보행(locomotion) 학습은 "명령한 속도로 걸어라"를 배운다. Parkour는 여기에 두 가지가 더 붙는다.

- **목표점 추종(goal waypoint)**: 코스에 8개의 waypoint가 놓여 있고, 로봇은 그것을 순서대로 통과해야 한다. 보상은 "다음 목표 방향으로의 전진속도"다.
- **3D 장애물**: 허들(hurdle), 단차(step), 구멍(gap), 계단(stair), 터널(crawl), 평지(flat) 여섯 종류. 각 지형은 11단계 난이도(curriculum level 0~10)를 갖는다.

이 프로젝트의 기준 논문은 **Extreme Parkour** (Cheng et al. 2023, arXiv:2309.14341)이며, 초기 코드는 Isaac Gym 기반 `RobotSW_Parkour`와 Genesis 기반 `RobotSW_Genesis`를 IsaacLab으로 이식한 것이다.

### 1.2 왜 어려운가 — 세 겹의 난이도

**(a) 지형이 관측되지 않으면 넘을 수 없다.** 로봇이 앞의 구멍을 못 보면 그냥 빠진다. 그래서 "지형을 어떤 형태로 정책에 넣느냐"가 이 과제의 절반이다. 이 프로젝트의 후반부 전체가 이 질문이다.

**(b) 커리큘럼이 자기 자신을 강화한다.** 난이도는 성능에 따라 자동으로 오르내린다:

```
move_up   = dis_to_origin > 0.8 × expected_dist
move_down = dis_to_origin < 0.4 × expected_dist
```
(`parkour_env.py:1794`, 레벨은 0에서 clamp)

잘하는 정책은 어려운 지형에 많이 노출되고, 못하는 정책은 쉬운 지형에 갇힌다. 그래서 **학습 로그의 평균값으로 두 정책을 비교하면 서로 다른 분포를 섞어 재는 셈**이 된다. 이 사실이 이 프로젝트의 측정 방법론 전체를 지배한다 (→ §11).

**(c) 실기에 옮겨야 한다.** 시뮬레이터는 지형 메시를 직접 읽을 수 있지만 실제 로봇은 못 읽는다. 시뮬레이터 전용 정보를 **특권(privileged) 관측**이라 부르고, 그것에 의존하는 정책은 아무리 잘해도 배포할 수 없다. 이 프로젝트의 최종 목표는 "특권 관측 없이 그 성능에 얼마나 근접하는가"다.

---

## 2. 학습 프레임워크

이 절은 **전 단계에 공통으로 깔린 뼈대**다. 단계별로 바뀐 부분은 각 단계 절에서 다시 짚는다.

### 2.1 전체 구조 — 비대칭 actor-critic (RMA)

```
                    ┌──────────────── ENVIRONMENT (Go2ParkourImitationEnv) ────────────────┐
                    │ 6 지형 × 11 난이도 curriculum · goal waypoint 8개 · 50 Hz 제어        │
                    └──────────────────────────────┬───────────────────────────────────────┘
                                                   │ observation (dict)
      ┌────────────────────────────────────────────┼────────────────────────────────────────┐
      │                                            │                                        │
  policy(42~46)      history(10×42~46)     지형 관측 (표현별로 교체)              priv_latent(37)
  proprioception     ring buffer           scan(187) / clearance(294)             friction·mass·gain
  + contact                                / voxel(7371) / lidar grid(7371)       (특권, 배포 불가)
      │                     │                       │                                       │
      ▼                     ▼                       ▼                                       ▼
 ┌─────────┐        ┌──────────────┐      ┌──────────────────┐                    ┌──────────────┐
 │ (직결)  │        │ Conv1D       │      │ scan MLP[128,64, │                    │ priv_encoder │
 │         │        │ history_enc  │      │ 32] 또는         │                    │ MLP[64,20]   │
 │         │        │ 460 → 20     │      │ VoxelEncoder     │                    │ 37 → 20      │
 │         │        │ (student)    │      │ (Z-grouped 2D    │                    │ (teacher 전용)│
 │         │        │              │      │  CNN)            │                    │              │
 └────┬────┘        └──────┬───────┘      └────────┬─────────┘                    └──────┬───────┘
      │                    │                       │                                     │
      └────────────────────┴───────────┬───────────┴─────────────────────────────────────┘
                                       ▼
                        ┌──────────────────────────────────┐
                        │  ACTOR MLP [512, 256, 128] → 12  │   ← 배포되는 것은 이것뿐
                        │  Gaussian, log_std learnable     │
                        └───────────────┬──────────────────┘
                                        │ a ∈ R^12
                                        ▼
                        ┌──────────────────────────────────┐
                        │  q_des = q_default + 0.25·(clip(a,−10,10) ⊙ h)   h_hip=0.5, else 1 │
                        │  → DCMotor PD (Kp=40, Kd=1.0, sat=35 N·m)        │
                        └──────────────────────────────────┘

  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ CRITIC MLP [512,256,128] → 1   입력: policy + 지형관측 + priv_explicit + priv_latent │
  │ ★ critic은 특권 정보를 전부 본다. 학습에만 쓰이고 배포 시 버려진다.                    │
  └───────────────────────────────────────────────────────────────────────────────┘
```

이 **비대칭 구조가 프로젝트의 핵심 장치**다. actor(배포되는 부분)는 실기에서 얻을 수 있는 것만 읽고, critic(학습 전용)은 시뮬레이터의 모든 것을 본다. 가치 추정이 정확해지면 정책 gradient의 분산이 줄어드므로, actor의 관측이 빈약해도 학습이 성립한다. 최종 SL-Grid 정책이 teacher 없이 from-scratch로 학습될 수 있었던 이유가 이것이다.

### 2.2 Input — 관측 그룹

| 그룹 | 차원 | 내용 | 갱신 | 배포 |
|---|---|---|---|---|
| `policy` | 42~46 | yaw_diff(1) + next_yaw_diff(1) + projected_gravity(3) + cmd_x(1) + joint_pos(12) + joint_vel×0.05(12) + actions(12) + contact_filt(4) | 50 Hz | ✅ |
| `history` | 10 × 42~46 | `policy` obs의 최근 10 step ring buffer | 50 Hz | ✅ |
| `scan` / `voxel` / `lidar` | 187 / 294 / 7,371 | **지형 관측 — 이 슬롯이 프로젝트 후반의 전장이다** | 10 Hz | 표현에 따라 다름 |
| `priv_explicit` | 6 | base 선속도×2.0(3) + 각속도×0.25(3) | 50 Hz | ⚠️ estimator로 추정 |
| `priv_latent` | 37 | base_friction(1) + foot_friction(8) + base_mass(1) + base_com(3) + joint stiffness ratio(12) + damping ratio(12) | startup | ❌ 특권 |

`priv_explicit`(base 선속도)은 실제 로봇에서 직접 측정할 수 없다. 그래서 **estimator** 네트워크가 `policy` 관측만으로 이를 추정하고, actor는 추정값을 읽는다. critic과 AMP discriminator는 실제값을 그대로 쓴다 (학습 정확도 보존).

> 2026-06-15에 이 estimator가 AMP 러너 경로에서 **생성조차 되지 않고 있던** 버그를 발견·수정했다. `OnPolicyRunnerAMP._construct_algorithm`이 부모 러너의 estimator 빌드 로직을 오버라이드하며 생략해, actor가 시뮬레이터 ground-truth 속도를 그대로 읽고 있었다. 즉 그때까지의 모든 AMP 정책은 **원리적으로 배포 불가**였다. 상속 관계가 정상이어도 객체 생성 책임이 러너에 있으면 상속만으로는 부족하다는 사례다.

### 2.3 지형 관측 — 네 가지 표현

이 프로젝트가 실제로 비교한 네 가지 표현이다. §7~§8 전체가 이 표를 채워가는 과정이다.

| 표현 | 역할 | 형태 | 차원 | 프레임 | 천장 표현 | 배포 |
|---|---|---|---|---|---|---|
| **height_scan** | teacher | 2.5D 하향 높이 격자 (17×11) | 187 | base 하향, yaw 정렬 | ✗ | 가능(깊이센서) |
| **3D clearance** | teacher | 방향별 거리 벡터 (21 az × 14 el) | 294 | base+0.05 m, yaw 정렬 | ✓ | 불가(특권 ray) |
| **voxel occupancy** | teacher | 3D 점유 격자 (27×21×13) | 7,371 | base+0.05 m, yaw 정렬 | ✓ | 불가(특권 ray/메시) |
| **LiDAR SL-Grid** | **배포 정책** | **같은 27×21×13 격자를 Mid-360 히트로 채움** | **7,371** | base+0.05 m, yaw 정렬 | ✓ | **가능** |

height_scan은 한 (x, y) 좌표에 스칼라 높이 하나만 담는다. 터널(crawl)처럼 같은 (x, y)에 바닥과 천장이 동시에 존재하는 지형은 **표현할 수 없다** — 하향 ray가 천장 윗면을 찍어 관측이 오염된다. 이것이 3D 전환의 근본 이유다.

### 2.4 Output — action

```
a ~ N(μ_θ(o), exp(log σ))   ∈ R^12          (FL/FR/RL/RR × hip/thigh/calf)
a' = clip(a, −10, 10)
q_des = q_default + 0.25 · (a' ⊙ h),        h_i = 0.5 (hip), 1.0 (else)
τ = DCMotor_PD(q_des; Kp=40, Kd=1.0, sat=35 N·m, v_lim=30 rad/s)
```

hip(외전) 관절에만 0.5배를 걸어 다리가 옆으로 벌어지는 범위를 좁힌다. 결과적 action 범위는 `q_default ± 2.5 rad`.

### 2.5 학습 알고리즘 — PPO + AMP

기본은 PPO(clip surrogate)에 세 가지가 얹힌 형태다.

**(1) priv_reg loss — RMA 증류**
history encoder가 privileged encoder의 latent를 모방하도록 강제한다:

$$\mathcal{L}_{\text{priv}} = \lambda(t)\,\big\|\, z_{\text{hist}} - \text{sg}[z_{\text{priv}}] \,\big\|^2$$

$\lambda(t)$는 2000~3000 iteration 구간에서 0 → 0.1로 선형 상승한다. 별도 optimizer로 도는 DAGGER 업데이트도 있다 (lr = 2e-4).

**(2) AMP discriminator — 보행 스타일 imitation** (→ §5에서 상술)

$$r = r_{\text{task}} + w_{\text{amp}} \cdot c_{\text{amp}} \cdot \mathbb{1}_{\text{flat}} \cdot r_{\text{disc}}, \qquad w_{\text{amp}} = 0.3, \quad c_{\text{amp}} = 0.08$$

(2026-09-02 정정: agent.yaml `amp.reward_coef` 0.08이 `discriminator.amp_reward_coef`로 들어가 실효 계수는 0.024. 코드 기준 $r_{\text{disc}} = -\log(\max(1 - \sigma(D_\phi(x)),\, 10^{-4}))$ — clamp 때문에 $[0,\ \log 10^4] \approx [0,\ 9.21]$로 유계이고, task 보상에만 `clip(min=0)`이 걸리며 AMP 항은 env 밖(runner)에서 더해져 재클립되지 않는다. $\mathbb{1}_{\text{flat}}$ 표기는 남겨 두었지만, 이 리포트가 다루는 AMP run은 전부 `apply_amp_on_terrain = True`라 실제로는 모든 env에서 1이다)

**(3) Symmetry data augmentation** (→ §4에서 상술)

| 하이퍼파라미터 | 값 |
|---|---|
| learning_rate | 2e-4 (adaptive, floor 1e-5 / ceiling 1e-2, factor 1.5) |
| num_steps_per_env | 24 |
| num_learning_epochs / mini_batches | 5 / 4 |
| clip_param / γ / λ_GAE | 0.2 / 0.99 / 0.95 |
| entropy_coef / desired_kl | 0.01 / 0.01 |
| discriminator lr / hidden | 2.5e-4 / [1024, 512] |
| num_envs (표준) | 4096 (teacher/LiDAR 계열은 1024) |
| 학습량 | 50,000 iteration (약 60~65시간 wall-clock) |

### 2.6 Reward — 18개 항

| 항목 | weight | 비고 |
|---|---|---|
| tracking_goal_vel | **+1.5** | 다음 goal 방향 전진속도 / 명령속도. 주 신호 |
| tracking_yaw | **+0.5** | heading 오차. **정지 시 감쇠 게이팅** (→ §3) |
| termination | **−100.0** | 낙상 페널티 |
| collision | −10.0 | base/thigh 등 비정상 접촉 |
| feet_stumble / feet_edge | −1.0 / −1.0 | 발 측면 충격 / 엣지 접촉 |
| lin_vel_z_l2 / orientation_l2 | −1.0 / −1.0 | 평지에서만 완전 적용, 장애물 지형은 ×0.5 또는 해제 |
| hip_pos / dof_error_l2 / action_rate_l2 | −0.5 / −0.04 / −0.1 | 자세·평활성 |
| feet_dragging | −0.1 | 뒷다리 끌림 (RL, RR만) |
| ang_vel_xy_l2, dof_acc_l2, torques_l2, delta_torques | 소값 | 정규화 항 |

최종 보상은 `total_reward.clamp(min=0)` — 음수 에피소드를 막는 의도된 설계다.

---

## 3. 단계 1 — 기본 환경 구축과 "학습이 아예 안 되던" 시기 (2026-04 ~ 05)

### 3.1 무엇을 만들었나

2026-04-29에 Isaac Gym 기반 `RobotSW_Parkour`와 Genesis 기반 `RobotSW_Genesis`를 분석해 IsaacLab Direct Environment로 새로 구축했다.

- Genesis의 heightfield 기반 parkour 지형 4종(hurdle/step/gap/stair)을 **IsaacLab의 trimesh 기반으로 포팅**. 패러다임이 다르다 — heightfield는 2D 격자에 높이를 채우고, trimesh는 박스를 직접 배치한다. 통로 구조는 "좌/우 두 박스 + 가운데 빈 공간"으로, gap은 "ground plane을 안 만들어서 실제 추락"으로 옮겼다.
- PhysX GPU 버퍼 overflow 해결: 기본값 `gpu_max_rigid_patch_count = 163,840`이 4096 env × 복잡 mesh 지형에서 즉시 넘친다. $2^{20}$으로 상향 (요구치 536,321 대비 1.95배 마진).

### 3.2 해결한 큰 문제 — `tracking_goal_vel`이 0.05에 갇히다

초기 환경은 학습이 진행되지 않았다. 주 보상인 `tracking_goal_vel`이 0.05~0.1에서 정체 — 명령 속도의 5~10%만 전진한다는 뜻이다. 2026-05-11 세션에서 세 가지 원인이 확정됐다.

**(a) Actuator가 물리적으로 명령을 못 따라간다.** stock Go2 config는 `stiffness=25, effort=23.5 N`인데, 이것으로는 1.0 m/s 명령을 추종할 수 없다. 성공하는 참조 구현은 `stiffness=40, effort=35~40 N`을 쓰고 있었다. **정책이 아무리 학습해도 액추에이터가 못 따라가면 보상에 천장이 생긴다.**

**(b) Reward weight가 원본과 달랐다.** cfg 주석은 "Genesis original"이라 적혀 있었지만 실제 값이 달랐다:

| 항목 | Genesis 원본 | 당시 값 | 수정 |
|---|---|---|---|
| tracking_goal_vel | 1.5 | 1.2 | → 1.5 |
| tracking_yaw | 0.5 | 0.7 | → 0.5 |
| goal/yaw 비율 | **3.0** | **1.71** | → 3.0 |
| termination | −100.0 | −0.0 | → −100.0 |

yaw 보상이 상대적으로 과다해 전진 동기가 희석돼 있었다.

**(c) "정지 + 정면" local optimum.** 이것이 가장 미묘하다. 구현에서 `yaw_diff`를 `atan2(sin, cos)`로 wrap 처리하고, reset 시 로봇 heading = 0, 첫 goal이 정면에 놓인다. 그러면:

1. 가만히 있으면 `yaw_diff = 0` → `tracking_yaw = 1.0` (최대값)
2. 정지 상태에서는 속도·자세 관련 페널티가 거의 전부 0
3. `termination = −0`이라 넘어져도 손해가 없다

**결과: 아무것도 안 하는 것이 안전한 최적점이 된다.** Genesis 원본은 wrap을 하지 않아 학습 초기 yaw 신호가 −π~π로 요동쳐 정지가 자연히 억제됐다.

수정은 yaw 보상에 **이동 게이팅**을 넣는 것이었다:

$$\tilde r_{\text{yaw}} = e^{-|\Delta\psi|} \cdot \mathrm{clamp}\!\left(\frac{\|v_{xy}\| - v_{lo}}{v_{hi} - v_{lo}},\, 0,\, 1\right), \qquad v_{lo}=0.05,\; v_{hi}=0.15\ \text{m/s}$$

**교훈**: 이 세 원인 중 어느 것도 "알고리즘"이 아니다. 액추에이터 능력, 상수 오타, 보상 기하학이다. 이후 프로젝트 전반에서 "알고리즘을 바꾸기 전에 환경을 의심한다"는 순서가 굳어졌다.

또 이 시점에 **지형별 커리큘럼 로깅**(`curriculum/mean_terrain_level_{flat,hurdle,step,gap,stair}`)을 넣었다. 전체 평균만으로는 어느 지형에서 막히는지 알 수 없다. 이 인프라가 이후 모든 진단의 기반이 된다.

---

## 4. 단계 2 — Symmetry: 3-leg gait 고착을 깨다 (2026-06)

### 4.1 문제 — 왼쪽 뒷다리만 계속 접고 다닌다

평지 보행이 되기 시작하자 새로운 병리가 나타났다. 로봇이 **세 다리로만 걷는다.** 왼쪽 뒷다리(RL)의 접촉 duty가 0.336으로 주저앉고 나머지 셋은 정상(FL 0.78)이다.

`contact_duty` 보상의 목표값을 0.30 → 0.42 → 0.5로 올려봤지만 매번 다시 나타났다. **seed를 바꿔도 항상 같은 다리**였다. 무작위 초기화의 문제가 아니라는 뜻이다.

### 4.2 진단 — 코드 비대칭은 없었다

read-only 감사로 세 갈래를 전부 닫았다.

- `--seed`는 지형까지 완전히 재생성한다(`use_cache=False`). 그런데도 항상 RL → 시드 무관한 구조적 tiebreak.
- 방향성 bias 없음: command 대칭, 유일한 회전 지형(zigzag)은 `proportion=0.0`으로 비활성, 활성 지형 전부 직선 좌우대칭 corridor.
- rear-left를 특정하는 코드 비대칭 없음: 초기 자세의 hip은 L/R 미러(FL에도 영향), thigh는 front/rear 분할(RR에도 영향), 보상은 전부 per-foot 대칭, height-scan도 y-대칭.

**판정**: 대칭 시스템의 **emergent deterministic equilibrium**이다. 4발 대칭 보상은 3-leg local optimum을 허용하고, "어느 다리를 접을까"의 동점은 고정된 PhysX body-index / solver order가 깨뜨려 항상 같은 다리로 간다. 버그가 아니다.

따라서 **보상을 더 추가해도 소용없다** — equilibrium 자체를 없애지 못하고 "어느 다리를 고를지"만 바꾼다. 정책 수준에서 좌우 등변성(equivariance)을 강제해야 한다.

### 4.3 방법 — mirror data augmentation

미러 연산자 $M$을 정의하고, PPO 배치를 $B \to 2B$로 늘려 원본과 미러본을 함께 학습시킨다 (`num_aug = 2`, 앞의 $B$는 원본 보존). 목표는 정책의 좌우 등변성:

$$\pi\big(M_a a \,\big|\, M_s s\big) \;=\; \pi(a \mid s)$$

구현은 `parkour/mdp/symmetry.py`. 각 관측 그룹의 변환은 다음과 같다.

**관절 (12 DOF)** — L↔R 스왑 순열 $\sigma$ + hip(외전) 부호 반전:

$$(M_q q)_i = \epsilon_i \, q_{\sigma(i)}, \qquad \epsilon_i = \begin{cases} -1 & i \in \text{hip} \\ +1 & \text{otherwise}\end{cases}$$

$\sigma$는 하드코딩이 아니라 **런타임 `joint_names` 이름 매칭으로 동적 구성**하고, 유효한 순열인지 검사한다.

**★ magnitude 예외**: `priv_latent`의 stiffness/damping(24개)은 크기 값이므로 **스왑만 하고 부호는 뒤집지 않는다**(`_swap_joints_no_flip`). 여기에 부호 반전을 넣으면 조용히 틀린다.

**벡터량** — 중력 투영 $g$, 선속도 $v$, 각속도 $\omega$:

$$g \mapsto (g_x,\,-g_y,\,g_z), \qquad v \mapsto (v_x,\,-v_y,\,v_z), \qquad \omega \mapsto (-\omega_x,\,\omega_y,\,-\omega_z)$$

yaw 오차는 부호 반전, 명령 속도(전진 성분)는 불변, `contact_filt`는 발 스왑만.

**지형 관측** — 표현마다 다르다. 이것이 make-or-break였다:

| 표현 | 차원 | 미러 연산 |
|---|---|---|
| height_scan | 187 = 11 lat × 17 lon | 측방 축 flip |
| clearance | 294 = 21 az × 14 el | 방위각 역순 순열 (값은 거리 → 부호 변환 없음) |
| **voxel / lidar 격자** | **7,371 = 27 × 21 × 13** | **y-flip 순열** |

격자는 C-order flat 인덱스 $k = i_x (n_y n_z) + i_y n_z + i_z$를 쓰므로, 미러는

$$i_y \;\mapsto\; n_y - 1 - i_y, \qquad n_y = 21$$

$n_y$가 홀수라 중앙 슬라이스($i_y = 10$)는 자기 자신으로 사상된다. 점유값은 스칼라 categorical이므로 순열만 적용하고 부호는 건드리지 않는다.

**검증**: involution 단위 테스트(두 번 미러 = 원본), 순열 유효성 assert, 그룹별 shape 게이트.

### 4.4 결과

`Go2-Parkour-Symmetry`로 등록해 3-leg gait를 해소했다. 다만 이 시점의 판정은 구현 검증까지이고, symmetry의 **정량적 이득은 단계 6에서 lidar 격자에 다시 적용했을 때 통제된 A/B로 처음 측정된다** (→ §8).

**시각 비교 (2026-08-24 재렌더)** — 설정 차이가 symmetry 하나뿐인 통제 A/B 체크포인트 쌍을 같은 평지·같은 전진 명령(vx 1.0)으로 재생해 나란히 붙였다: OFF = `go2_parkour/2026-06-09_13-18-14_contact_duty_5.0_flat` iter 15,100, ON = `go2_parkour_symmetry/2026-06-09_17-53-09` iter 15,000 (5.1 저장소).

🎥 [3-leg gait 전/후 비교](assets/videos/01_flat_3leg_symOFF_vs_symON.mp4) — 왼쪽(OFF)은 RL(왼쪽 뒷) 다리를 끌고 걷는 3-leg gait, 오른쪽(ON)은 네 발 접지가 균등한 보행.

텐서보드에서 복원한 단계 2 정량 수치(`contact_duty_deficit`, 목표 duty 미달분 합 — 0에 가까울수록 충족): iter 15k 기준 **OFF −0.0199 vs ON −0.0005** (iter 5k: −0.0275 vs −0.0021). 에피소드 길이는 양쪽 동일(≈749)해 교란 없음. 단, 단일 보상항 지표·arm당 seed 1개라는 한계는 §11.1과 동일하다.

---

## 5. 단계 3 — Imitation (AMP) 결합 (2026-06)

### 5.1 문제 — 지표는 오르는데 보행이 이상하다

지형은 넘어가는데 동작이 부자연스러웠다. 계단에서 네 발을 동시에 모아 뛰는 **pronk** 도약이 27.7% airborne으로 측정됐다 — 이동 비용(CoT)이 1.49배로 비효율이지만 충격은 안전 임계 미달(3.1 BW)이라, 보상만으로는 억제 압력이 잘 걸리지 않는 영역이었다.

### 5.2 방법 — Adversarial Motion Prior

AMP는 "전문가 모션 데이터와 구별되지 않는 움직임"에 보상을 준다. discriminator $D_\phi$가 실제 롤아웃과 참조 모션을 구별하도록 학습하고, 정책은 그것을 속이려 한다:

$$r_{\text{total}} = r_{\text{task}} + w_{\text{amp}} \cdot c_{\text{amp}} \cdot \mathbb{1}_{\text{flat}} \cdot r_{\text{disc}}, \qquad w_{\text{amp}} = 0.3, \quad c_{\text{amp}} = 0.08$$

(2026-09-02 정정: agent.yaml `amp.reward_coef` 0.08이 `discriminator.amp_reward_coef`로 들어가 실효 계수는 0.024. 코드 기준 $r_{\text{disc}} = -\log(\max(1 - \sigma(D_\phi(x)),\, 10^{-4}))$ — clamp 때문에 $[0,\ \log 10^4] \approx [0,\ 9.21]$로 유계이고, task 보상에만 `clip(min=0)`이 걸리며 AMP 항은 env 밖(runner)에서 더해져 재클립되지 않는다. $\mathbb{1}_{\text{flat}}$ 표기는 남겨 두었지만, 이 리포트가 다루는 AMP run은 전부 `apply_amp_on_terrain = True`라 실제로는 모든 env에서 1이다)

code 기준 정확한 정의는 다음과 같다(`amp_discriminator.py` `_reward_from_logits`, `on_policy_runner_parkour_amp.py` fusion, `parkour_env.py` L1672).

$$r_{\text{disc}} = -\log\big(\max(1 - \sigma(D_\phi(x)),\; 10^{-4})\big), \qquad r_{\text{task}} = \max\Big(\sum_i w_i r_i,\; 0\Big)$$

- $\sigma$는 sigmoid, $D_\phi(x)$는 discriminator의 logit 출력이다(BCE / MimicKit 방식).
- `1e-4` clamp 때문에 $r_{\text{disc}} \in [0,\; \log 10^4] \approx [0,\; 9.21]$이고, $w_{\text{amp}} \cdot c_{\text{amp}}$를 곱한 총보상 기여는 최대 약 0.22다.
- 클램프는 task 보상에만 걸린다. AMP 항은 env 밖(runner)에서 더해지므로 다시 클립되지 않는다.
- $\mathbb{1}_{\text{flat}}$는 `apply_amp_on_terrain = True`일 때 모든 env에서 1이다(→ §5.3).
- 코드의 `compute_amp_reward()`는 $c_{\text{amp}}$까지 곱한 값을 돌려준다. §5.4-1 표의 `r_disc` 열이 바로 그 값이므로, 총보상 기여는 거기에 $w_{\text{amp}} = 0.3$만 더 곱하면 된다.

- **참조 모션**: `imitation/go2_jump` (`ParkourImitationEnvCfg.amp_motion_pkl`). **(2026-09-02 정정) 이름과 달리 jump 전용 폴더가 아니다.** 이 값은 파일이 아니라 **디렉토리**이고 `Go2MotionLib`이 내부 `*.pkl` **30개를 전부 glob**해 클립 길이로 가중 샘플링한다(`motion_lib.py:268-271`). 실제 구성은 총 65.3 s이고, 보행 계열이 67.1 %로 과반수다:

  | gait 계열 | 클립 수 | 총 길이 | 참조 분포 비중 |
  |---|---|---|---|
  | jump | 12 | 21.5 s | **32.9 %** |
  | walk | 8 | 18.9 s | 28.9 % |
  | trot | 4 | 13.8 s | 21.1 % |
  | run | 6 | 11.1 s | 17.0 % |
  | **합계** | **30** | **65.3 s** | **100 %** (보행 67.1 %) |

  단 trot 4개는 md5가 같은 2쌍이라 고유 클립은 2개뿐이고, 그만큼 샘플링 가중치가 두 배로 잡혀 있다. 계열별 discriminator 점수는 §5.4-1의 「참조 모션 계열별 비교」 소절에서 다룬다. 모션 데이터의 원 출처와 캡처 방식은 여전히 {확인 필요}.
- **AMP 관측**: frame당 49차원 × history 10 = 490. 구성은 `dof_pos(12) + dof_vel(12) + root_height(1) + root_lin_vel_b(3) + root_ang_vel_b(3) + foot_pos_local(12) + root_rot_tan_norm(6)`.
- discriminator는 BCE loss (WGAN 비활성), hidden `[1024, 512]`, lr 2.5e-4.
- **AMP 관측은 PPO의 obs TensorDict에 들어가지 않는다.** `env.extras["amp_obs"]`라는 별도 경로로만 전달되므로, symmetry data augmentation 함수는 AMP 관측을 미러하지 **않고**, 해서도 안 된다.

**표준 AMP를 어디까지 그대로 썼고, parkour용으로 무엇을 바꿨나 (2026-09-03 추가)**

판별자·손실·학습 루프는 표준 AMP 그대로고, parkour 전용 변형은 **관측 설계와 보상 융합 층**에만 있다. "다리 움직임 위주" 같은 판별자 변형은 구현만 되고 최종 파이프라인에 채택되지 않았다.

- **그대로 쓴 것** — 판별자 MLP `[1024, 512]`, BCE 손실, gradient penalty 5.0, weight 정규화 0.01, 관측 정규화 클립 10, 보상 $-\log(1-D)$. 이 값들은 `go2_imitation` 설정을 그대로 가져온 것이고, 2026-06-15 첫 AMP run부터 2026-08-10 최종 run까지 동일하다. 관측 49차원 × history 10도 일반적인 사족 AMP 구성이며(다리 관련 항이 36/49이지만 가중은 없다), 다리나 발에 가중을 둔 판별자는 없다.
- **parkour용으로 바꾼 것 (학습에 반영됨)** — (1) 보상 융합을 가산형 `r_task + w_amp·c_amp·mask·r_disc`로 하고 mask를 평지 env만 1 → `apply_amp_on_terrain`으로 전 env 1 (§5.3); 판별자 학습 샘플도 같은 mask를 따른다. (2) 종료 프레임의 AMP 관측을 리셋 이전 값으로 보정(`terminal_amp_obs`). (3) symmetry augmentation에서 AMP 관측 제외(위 bullet). (4) 참조 세트를 jump·walk·trot·run 30클립 혼합으로.
- **구현했지만 채택하지 않은 것** — `TerrainStyle-v0`(37차원, §5.3)가 "관절 협응만 보는" 다리 위주 변형에 가장 가깝다. 2026-06-15 세션에서 구현·등록하고 A/B(v0 / mask 제거 / TerrainStyle)를 계획했으나, 그 다음 날의 run부터 최종 run까지 모두 49차원 + `apply_amp_on_terrain` 경로를 썼다. 두 저장소(5.1 / 6.0) 로그 전체에 37차원으로 학습한 run이 없고 결과를 적은 세션 보고서도 없다. {확인 필요}: 시도 후 성능이 미흡해 폐기된 것으로 기억되나, 로그·보고서 근거는 남아 있지 않다. 밖에 명령 조건부 판별자(`amp_command_condition.py`)와 DRAIL 확산 판별자도 코드에 있지만 go2/leg imitation_tracking 전용이고 parkour 설정에는 옵션이 연결되어 있지 않다.
- **없는 것** — 부위별 가중 판별자, 발 궤적·접촉 패턴 전용 판별자, 지형별 self-imitation demo(§5.3 보류), 지형·난이도에 따른 `amp_weight` 스케줄.

> §5.4-1과 같이 읽으면: 정책이 관측 공간상 trot에 가장 가까운데도 D가 0.38에 머물고 계단에서 무너지므로, 다리 위주 변형(TerrainStyle)은 재시도 여지가 있는 미검증 옵션이다. 다만 σ-이동 분석에서 지형 의존 12열이 아니라 불변 37열의 이동이 D 순서와 정렬되므로, 12열을 빼는 것만으로 지형 간 격차가 사라질지는 불확실하다.

### 5.3 해결한 큰 문제 — 지형에서 AMP가 꺼져 있었다

`parkour_imitation`은 평지에서만 AMP를 학습하고 장애물 구간에서 스타일이 붕괴하고 있었다. 원인은 `_flat_env_mask`였다. 이 마스크가 discriminator 학습을 평지 env로 제한하고, 보상 융합에서도 `amp_contribution = amp_weight × flat_mask × disc_reward`이므로 **장애물 지형은 AMP 보상이 정확히 0**이었다.

두 경로로 풀었다.

1. **`apply_amp_on_terrain` flag** — True면 마스크 전체를 True로 만들어 모든 지형에서 AMP를 켠다. 기본값 False로 기존 거동 100% 보존.
2. **`TerrainStyle-v0` — 지형 불변 AMP 관측 (49 → 37차원)**. 평지 보행 · 점프 혼합 demo에서 배운 스타일을 장애물 지형에 적용하면 지형 의존 feature가 OOD를 만든다. 그래서 지형에 종속된 12개를 제거했다:

   제거: `root_height(1)` + `root_lin_vel_b.z(1)` + `foot_pos_local.z × 4(4)` + `root_rot_tan_norm`의 pitch/roll(6)
   잔존 37 = `dof_pos(12) + dof_vel(12) + root_lin_vel_b.xy(2) + root_ang_vel_b(3) + foot_pos_local.xy(8)`

   즉 **"어느 높이에서 어떤 자세로"는 버리고 "관절이 어떻게 협응하는가"만 남긴다.** 이것이 지형 간 전이 가능한 스타일이다.

> 한계로 명시된 것: 계단 도약처럼 **지형 고유의 우아함**은 평지 보행 · 점프 혼합 demo에 정보 자체가 없어 학습할 수 없다. 이를 위해서는 지형 self-imitation demo 파이프라인이 필요하고, 보류 상태다.

### 5.4 시각 비교 — 계단 pronk 전/후 (2026-09-02 재렌더 v2)

🎥 [AMP 전/후 계단 비교 v2](assets/videos/02_stair_pronk_ampOFF_vs_ampON_v2.mp4) — 같은 계단 전용 지형(단높이 0.15 m 고정), 목표 주도 주행. 왼쪽은 AMP 이전(symmetry-only, `go2_parkour_symmetry/2026-06-09_17-53-09` iter 30,000)으로 네 발을 모아 뛰는 pronk 도약이 보이고, 오른쪽은 AMP 결합(`parkour_imitation_go2_symmetry_random_goal/2026-07-29_11-14-23`, `model_49999_FINAL_BEST.pt`)이 8단 계단을 올라 완주하는 장면이다.

> **왜 오른쪽 arm을 바꿨나.** 구 영상([`02_stair_pronk_ampOFF_vs_ampON.mp4`](assets/videos/02_stair_pronk_ampOFF_vs_ampON.mp4), 보존)의 AMP arm은 `parkour_imitation_go2_symmetry/2026-06-16_10-17-39_amp_4.0_symmetry` iter 50,800이었고 **클립 안에서 계단에 실패한다** — 4 fps 프레임 검수 기준 우측 화면에서 t≈5.25\~7.25 s 배를 깔고 끌리다가 **t≈7.75 s에 완전히 전복**되고 t≈8.0 s 프레임에서는 로봇이 사라진다(episode 리셋). "AMP 도입 후의 계단 거동"을 보여주는 자리에 실패 장면이 들어가 있어 전시로서 성립하지 않았다.

교체 arm의 계단 성능 측정 — [`assets/render_02_v2/probe_stair_fixed.py`](assets/render_02_v2/probe_stair_fixed.py) (`scripts/demos/probe_stair.py` 포크, 원 출력 JSON은 [`assets/render_02_v2/probe_ampON_49999.json`](assets/render_02_v2/probe_ampON_49999.json)), 계단 전용 지형 · 단높이 0.15 m 고정 · level 8 고정, 16 env × 2,000 step, 43 episode:

| 지표 | 값 |
|---|---|
| 완주 (`cause_goal_reached`) | 32 / 43 = **74.4 %** |
| 목표 체인 절반 이상 통과 (`completion_half`) | **100 %** (27 / 27) |
| 전복 · 저높이 · 몸통접촉 종료 | **0 건** |
| 평균 전진속도 | 1.277 m/s |

렌더된 클립의 추적 개체도 base 높이가 z 0.288 → 1.519 m로 오른다. 계단 정상이 8단 × 0.15 m = 1.2 m이므로 **실제로 끝까지 올라간 궤적**이다.

> **읽을 때 주의**: AMP-only arm은 여전히 존재하지 않는다(도입 순서가 symmetry → AMP였고, 첫 AMP run부터 symmetry가 켜져 있었다). 따라서 이 비교는 "AMP 도입 전후의 계단 거동" 전시이지 AMP 단독 효과의 분리 측정이 아니다.
> **교체로 교란 요인이 늘었다**: 왼쪽은 Isaac Sim **5.1** / random-goal 이전 symmetry task / iter 30,000, 오른쪽은 Isaac Sim **6.0** / `Go2-ParkourImitation-Symmetry-RandomGoal-EasyEntry-v0` / iter 49,999다. 즉 학습량 차이(30k vs 50k) 위에 **엔진 버전 · EasyEntry 커리큘럼(§6) · random-goal 도입**이 함께 얹힌다. 구 영상은 좌우 모두 5.1이었으므로 엔진 교란은 이번에 새로 생긴 것이다. 반면 AMP 하이퍼파라미터 자체는 두 AMP run의 저장 cfg가 동일하다(49차원 AMP 관측, `amp_weight` 0.3, `apply_amp_on_terrain=true`).

**렌더 조건**: `scripts/demos/play_per_terrain.py`의 포크 [`assets/render_02_v2/play_stair_fixed.py`](assets/render_02_v2/play_stair_fixed.py)(추가한 것은 계단 단높이 고정 옵션과 카메라 오프셋 옵션뿐, 리포 소스는 수정하지 않았다), `--terrain stair --num_envs 2 --video_length 520 --max_init_level 8 --sensor none`, 계단 단높이를 전 레벨 0.15 m로 고정. 추적 카메라는 `eye=(0,-2.8,1.05)` / `lookat=(0,0,0.1)`로 바꿨다 — 6.0 기본 뷰(`(0,-2.5,0.8)`)는 로봇이 화면 하단 가장자리에 걸려 왼쪽 화면과 구도가 맞지 않는다. **왼쪽 절반은 구 영상에서 그대로 잘라 썼다**: AMP-off 체크포인트는 5.1 리포에만 있고 6.0에서 재생할 수 없다. 최종 합성은 1920×574 · 50 fps · 9.9 s로 구 영상과 동일하다.

### 5.4-1 discriminator 출력 — 평지 vs 지형 vs reference (2026-09-02 신규 측정)

§5.3은 "지형에서 AMP가 꺼져 있었다"를 코드 레벨에서 고쳤다. 그러면 켜고 난 뒤 discriminator는 지형에서 실제로 무엇을 보고 있는가.

§5.4 영상과 **같은 모델**을 썼다: `parkour_imitation_go2_symmetry_random_goal/2026-07-29_11-14-23/model_49999_FINAL_BEST.pt` (iter 49,999, `Go2-ParkourImitation-Symmetry-RandomGoal-EasyEntry-v0`, AMP 관측 49차원 × history 10, `apply_amp_on_terrain = True`). 정책과 discriminator를 체크포인트에서 그대로 로드해 혼합 지형 격자에서 96개 env를 1,200 step 굴리고, 매 step의 `extras["amp_obs"]`에 discriminator를 먹여 D(x)와 r_disc를 지형 클래스별로 모았다. 참조 모션(`imitation/go2_jump`)은 env 자신의 `get_amp_observations()`로 40,960개를 뽑아 같은 경로로 평가했다 — 관측 구성과 ring-buffer 규약이 롤아웃과 동일하다.

> **(2026-09-02 정정) 이 절에서 "참조 모션"은 jump 클립이 아니다.** `imitation/go2_jump`은 파일이 아니라 디렉토리이고 `Go2MotionLib`이 내부 `*.pkl` 30개를 전부 glob한다. jump는 그중 32.9 %이고 나머지 67.1 %가 walk · trot · run이다. 아래 표의 reference 행은 **그 30클립 혼합 분포**의 값이지 jump 단독 값이 아니다. 계열별 분해는 아래 「참조 모션 계열별 비교」 소절을 보라.

**측정 위생 두 가지.** (1) `_amp_obs_buf`는 reset 때 0으로 지워지고 10 frame에 걸쳐 다시 찬다. 리셋 직후 샘플은 zero-padding history라 D가 무조건 낮게 준다. 지형 env가 평지보다 자주 종료하면 이 아티팩트가 **지형 클래스와 상관**돼 "지형이 낮다"는 결론을 저절로 만들 수 있다. 그래서 `steps_since_reset < 20`인 샘플을 모든 버킷에서 버렸다(전체 115,200개 중 4,421개). 실제 제외율은 버킷 간 차이가 거의 없어(3.6\~3.9%) 우려한 방향의 편향은 나타나지 않았다 — 가드는 걸었고, 점검은 깨끗했다. (2) 커리큘럼 레벨 0\~2 패치는 거의 평지라 지형 버킷을 희석한다. 헤드라인 표는 **레벨 3 이상**으로 제한했다(추가로 34,853개 제외, 최종 75,926개). 정규화기는 체크포인트에서 그대로 로드했고 `amp_obs_normalizer.count = 3.93e10`로 통계가 채워져 있음을 로드 직후 assert로 확인했다.

표의 `r_disc` 열은 코드의 `compute_amp_reward()` 반환값이다 — c_amp = 0.08이 이미 곱해져 있고, 총보상 기여 열은 거기에 w_amp = 0.3을 더 곱한 값이다(§2.5 참조). 블록 A가 §5.4 영상과 같은 모델이고, 블록 B는 §9의 최종 배포 정책(LiDAR SL-Grid)이다. **두 블록을 가로로 비교하지는 말 것** — B는 관측 파이프라인(LiDAR 격자)과 지형 구성(crawl 포함)이 달라 discriminator도 다르게 학습됐다. 블록 안의 지형 순서만 비교 대상이고, 그 순서는 두 블록에서 계단이 최저라는 점까지 일치한다.

| 대상 | 표본 수 | 평균 D(x) | 평균 r_disc (±std) | 총보상 기여 (0.3·r_disc) | AUROC (ref vs 정책) |
|---|---|---|---|---|---|
| **A. 영상 모델 — `2026-07-29_11-14-23/model_49999_FINAL_BEST.pt` (§5.4 영상과 동일)** | | | | | |
| reference (`go2_jump` 폴더 30클립 — jump 32.9 % + 보행 67.1 %, **2026-09-02 정정**) | 40,960 | **0.799** | 0.1310 ± 0.0206 | — | — |
| flat | 16,167 | 0.440 | 0.0492 ± 0.0217 | 0.0148 | 0.9999 |
| gap | 12,598 | 0.433 | 0.0479 ± 0.0201 | 0.0144 | 0.9995 |
| step | 12,482 | 0.388 | 0.0433 ± 0.0256 | 0.0130 | 0.9998 |
| hurdle | 18,481 | 0.369 | 0.0401 ± 0.0236 | 0.0120 | 0.9997 |
| **stair** | 16,198 | **0.271** | 0.0297 ± 0.0277 | 0.0089 | 0.9998 |
| **B. 최종 배포 정책 — `2026-08-10_..._slgrid_crawl_sym_scratch_50k/model_49999.pt` (§9, LiDAR SL-Grid)** | | | | | |
| reference (`go2_jump` 폴더 30클립 — jump 32.9 % + 보행 67.1 %, **2026-09-02 정정**) | 40,960 | **0.813** | 0.1376 ± 0.0249 | — | — |
| flat | 13,153 | 0.405 | 0.0442 ± 0.0208 | 0.0133 | ≥0.9999 |
| crawl | 14,409 | 0.370 | 0.0398 ± 0.0218 | 0.0119 | ≥0.9999 |
| gap | 10,186 | 0.349 | 0.0372 ± 0.0218 | 0.0112 | ≥0.9999 |
| step | 11,550 | 0.283 | 0.0295 ± 0.0224 | 0.0089 | ≥0.9999 |
| hurdle | 13,152 | 0.279 | 0.0289 ± 0.0216 | 0.0087 | ≥0.9999 |
| **stair** | 11,570 | **0.254** | 0.0262 ± 0.0222 | 0.0079 | ≥0.9999 |

![reference / 지형별 D(x) 분포](assets/figures/amp_disc_D_violin.png)

*참조 모션(빨강)과 정책 롤아웃(파랑)의 D(x) 분포. 가로 점선은 D=0.5, 검은 가로선이 평균.*

![지형별 AMP style 보상](assets/figures/amp_disc_reward_by_terrain.png)

*지형별 r_disc 평균 ± 표준편차. 빨간 점선은 참조 모션의 평균.*

![난이도 레벨별 D(x)](assets/figures/amp_disc_D_by_level.png)

*커리큘럼 레벨별 평균 D(x). 표본 200개 미만 구간은 생략.*

**측정된 사실.**

- 참조 모션의 D는 **0.799**, 정책 롤아웃은 평지에서도 **0.440**이다. discriminator는 포화하지 않았다(참조가 1.0에, 정책이 0.0에 붙어 있지 않다). 다만 두 분포의 겹침은 거의 없어 **AUROC가 모든 지형에서 0.999를 넘는다**(최소 0.9995, gap).
- 지형별 순서는 flat > gap > step > hurdle > stair 다. 가장 낮은 stair의 r_disc는 평지 대비 **40% 낮다**(0.0297 vs 0.0492).
- **레벨을 맞춰도 순서는 유지된다.** 레벨 3 이상만 자르면 버킷마다 상단 레벨 분포가 달라 혼합 효과가 남는다. 다섯 버킷이 모두 겹치는 레벨 3\~6 구간으로 다시 자르면 평균 D는 flat 0.441, gap 0.433, step 0.391, hurdle 0.371, stair 0.285로 순서가 그대로다.
- 난이도(커리큘럼 레벨)와의 관계는 **단조롭지 않다**. 계단이 가장 뚜렷하게 내려가지만 중간 레벨에서 오르내림이 있고, 평지는 레벨과 무관한 대역을 유지한다. 고레벨 구간은 표본이 얇다.

![계단 통과 중 style 보상 시계열](assets/figures/amp_disc_stair_timeseries.png)

*계단 한 판(env 79, 레벨 8). 파랑이 r_disc, 초록이 base 높이. 1.66 m를 내려가는 동안 style 보상이 바닥에 붙어 있다가 평지에 닿자 회복한다. 스크립트가 base 높이 변화가 가장 큰 창을 고르므로 전형이 아니라 극단 사례다.*

- 계단 한 판의 시계열은 더 뚜렷하다. 하강 시작 전 평탄 구간에서 r_disc ≈ 0.034\~0.049이던 것이, 1.66 m를 내려가는 구간 내내 **≈0.001\~0.005로 붕괴**했다가 바닥의 평지에 닿자 0.085까지 회복한다. 다만 이 창은 **base 높이 변화가 가장 큰 구간을 골라 그린 극단 사례**다 — 표의 stair 평균 0.0297보다 한 자릿수 낮은 것은 그 때문이며, 전형적인 계단 주행이 아니다.

#### 참조 모션 계열별 비교 — 참조 세트는 jump 단독이 아니었다 (2026-09-02 신규 측정)

**질문: "사족보행 AMP는 대부분 trot 참조를 쓰는데 우리는 jump를 썼다. 판별자가 편향된 것 아닌가?"**

**답: 우리도 trot을 쓰고 있었다.** `imitation/go2_jump`은 파일이 아니라 디렉토리고 `Go2MotionLib`이 그 안의 30개 클립을 전부 읽는다 — **trot이 이미 21.1 % 들어 있고 보행 전체로는 67.1 %다.** 그리고 학습된 판별자는 이 네 계열을 모두 "진짜"로 인정한다 — 평균 D(x)가 **run 0.856 > jump 0.807 > walk 0.795 > trot 0.749**로 전부 0.5보다 높고, **정책 롤아웃 0.377**과는 뚜렷하게 분리된다. 계열 간 편차보다 참조-정책 간격이 훨씬 크다는 뜻이다.

> **검산 한 줄.** 30개 클립을 따로 채점해 길이 가중으로 다시 합치면 D(x) = **0.7994**로, 위 §5.4-1 표의 참조 값 **0.799**와 일치한다. 아래 수치 전부가 이 검산에 기대고 있다.

**(1) `imitation/go2_jump`는 jump 전용 폴더가 아니다.** `ParkourImitationEnvCfg.amp_motion_pkl = "imitation/go2_jump"`는 파일이 아니라 **디렉토리**이고, `Go2MotionLib`이 그 안의 `*.pkl`을 전부 glob으로 읽는다(`motion_lib.py:268-271`). 실제 내용물은 30개 클립 · 총 65.3초이고, **폴더 이름과 달리 jump는 그중 3분의 1뿐이다.** 샘플링 가중치는 `weights=None`일 때 클립 길이(`motion_lengths`)이므로, 아래 지속시간 비중이 곧 참조 분포에서의 비중이다.

| gait 계열 | 클립 수 | 총 길이 | 참조 분포 비중 |
|---|---|---|---|
| jump | 12 | 21.5 s | **32.9 %** |
| walk | 8 | 18.9 s | 28.9 % |
| trot | 4 | 13.8 s | 21.1 % |
| run | 6 | 11.1 s | 17.0 % |
| **합계** | **30** | **65.3 s** | **100 %** (보행 계열 67.1 %) |

즉 **참조 분포의 3분의 2가 이미 보행이고 trot만 21%다.** 다른 연구가 trot을 쓴다면 이 프로젝트도 쓰고 있었다 — 폴더 이름이 오해를 만든 것이지 jump 전용 prior가 아니다.

> **부기 — trot 클립은 실제로 2개뿐이다.** `trot0.pkl`과 `trot1.pkl`은 md5가 같은 **동일 파일**이고(`447f232b…`), mirror 쌍도 서로 같다. 고유 내용은 2개인데 4개로 세어져 **trot의 샘플링 가중치가 두 배로 잡혀 있다.** 의도된 중복인지 실수인지는 확인하지 못했다.

**(2) 학습된 discriminator는 trot을 거부하지 않는다 — 다만 계열 중 가장 낮게 준다.**

측정 방법: §5.4-1과 **같은 체크포인트**를 로드한 뒤, env의 `_motion_lib`을 **클립 하나짜리 라이브러리로 갈아끼우고** env 자신의 `get_amp_observations()`로 클립당 8,192개 window를 뽑아 같은 discriminator · 같은 정규화기로 채점했다. 49차원 관측 구성, history 10 ring-buffer 규약, tan_norm 경로가 롤아웃과 완전히 동일하다(클립에 따라 달라지는 것은 `_motion_lib`뿐이다 — DOF 재정렬 인덱스는 `motion_lib.py`의 모듈 상수 `DOF_NAMES`에서 나오므로 파일과 무관하다).

> **검산.** 30개 클립 결과를 길이 가중으로 다시 합치면 평균 D(x) = **0.7994**이고, §5.4-1 표의 참조 값 **0.799**와 소수 넷째 자리까지 일치한다. 클립 단위로 쪼갠 관측 구성이 원 측정과 조용히 갈라지지 않았다는 뜻이다.

| 대상 | 참조 분포 비중 | 평균 D(x) | 평균 logit | 평균 r_disc (±std) | 총보상 기여 (0.3·r_disc) |
|---|---|---|---|---|---|
| ref: run | 17.0 % | **0.856** | +1.82 | 0.1578 ± 0.0211 | 0.0473 |
| ref: jump | 32.9 % | 0.807 | +1.45 | 0.1334 ± 0.0177 | 0.0400 |
| ref: walk | 28.9 % | 0.795 | +1.36 | 0.1272 ± 0.0091 | 0.0382 |
| **ref: trot** | 21.1 % | **0.749** | +1.09 | 0.1107 ± 0.0055 | 0.0332 |
| ref 전체 (길이 가중) | 100 % | 0.799 | +1.41 | 0.1310 ± 0.0206 | 0.0393 |
| 재리타깃 trot (학습 밖) | — | 0.725 | +0.97 | 0.1035 | 0.0310 |
| 재리타깃 walk (학습 밖) | — | 0.794 | +1.35 | 0.1268 | 0.0381 |
| **정책 롤아웃 (전 지형 풀, stair 포함)** | — | **0.377** | −0.70 | 0.0416 | 0.0125 |

**계열 안에서의 포폭** (클립당 8,192 window 균등 표본 기준, 평균 D(x)): run 0.844(`run0_mirror`)~0.861(`run1`) · jump 0.779(`jump_up_mirror`)~0.836(`jump_mirror`) · walk 0.764(`walk1_mirror`)~0.804(`walk_mirror`) · trot 0.749~0.749(네 파일이 사실상 2개의 중복이라 포폭이 없다). **jump의 최저값(0.779)이 walk의 최고값(0.804)보다 낮다** — 계열 경계가 깔끔하게 갈리지 않는다는 뜻이기도 하다. 42개 클립 전체 표는 이 문서 끝의 부표와 `assets/data/README.md`에 있다.

![참조 gait 계열별 · 정책 지형별 D(x) 분포](assets/figures/amp_disc_ref_family_violin.png)

**읽는 법.** 네 계열 **모두 D > 0.5로 "진짜" 쪽에 있다.** 판별자가 jump만 참으로 보고 trot을 가짜로 밀어내는 일은 일어나지 않는다. 계열 간 차이는 분명히 있다 — trot이 0.749로 최저, run이 0.856으로 최고다. 하지만 그 폭(logit +1.09~+1.82)은 **참조 전체와 정책 롤아웃 사이의 폭(logit +1.41 vs −0.70)에 비하면 작다.** 이 판별자의 결정 경계는 "어떤 gait냐"가 아니라 **"참조 모션이냐, 정책이 실제로 내는 움직임이냐"**를 가른다.

trot이 계열 중 최저인 것은 따로 설명이 필요하다. trot은 표준편차도 가장 작다(logit σ 0.092, jump는 0.271). 3.45초로 가장 길고 주기적이라 window 다양성이 낮은 탓이 크다. 낮고 좁은 점수는 "판별자가 trot을 싫어한다"보다 **"trot이 참조 분포 안에서 정책 쪽에 가장 가까운 영역을 차지한다"**로 읽는 편이 자료와 맞는다 — 근거는 아래 (4)다.

![클립 단위 D(x)](assets/figures/amp_disc_ref_per_clip.png)

**(3) 재리타깃 판본은 "held-out gait"가 아니다.** `imitation/go2/`에는 이름만 다른 trot · walk 클립이 12개 더 있고 학습에 들어가지 않았다. 스키마 · 프레임 수 · fps(60)는 `go2_jump/` 쪽과 같다. 그러나 이것을 다른 gait라고 부를 수는 없다 — `go2/go2_trot0.pkl`을 `go2_jump/trot0.pkl`과 프레임별로 빼면 차이의 대부분이 **root 높이 상수 오프셋 +0.054 m**(표준편차 0.004 m)이고 관절각 차이는 최대 0.39 rad이다. `go2_walk.pkl`은 사실상 동일하다(Δz −0.0005 m). **같은 모션의 재리타깃 판본**이다. 점수도 그에 걸맞게 움직인다 — 재리타깃 trot 0.725(학습 내 trot 0.749 대비 −0.024), 재리타깃 walk 0.794(0.795 대비 −0.001). **몸통이 5 cm 높아진 trot만 눈에 띄게 깎였다.** 이 항목은 gait 일반화의 증거가 아니라 **자세 통계(특히 몸통 높이) 민감도** 측정으로 읽어야 한다.

**(4) 그러면 정책은 무엇처럼 움직이고 있나.** 판별자 정규화기 통계로 49차원 최신 프레임을 표준화한 뒤(= D가 실제로 보는 공간) 계열별 중심점을 잡고, 정책 롤아웃 프레임을 최근접 중심에 배정했다.

> **이건 거친 분류기다.** "정책이 trot을 한다"는 판정이 아니라 **정규화된 49차원 공간에서 어느 계열 중심에 더 가까운가**를 잴 뿐이다. 최신 프레임 하나만 쓰므로 history 10 frame의 시간 구조(접촉 순서 · 위상)를 보지 않고, 계열 중심이 등방적이라고 가정한다. gait 위상 판별기로 읽지 말 것.

| 정책 지형 | jump | run | walk | **trot** |
|---|---|---|---|---|
| flat | 0 % | 0 % | 8 % | **92 %** |
| hurdle | 0 % | 0.5 % | 20 % | **80 %** |
| stair | 0.04 % | 5 % | 21 % | **74 %** |
| step | 1 % | 3 % | 31 % | **65 %** |
| gap | 0 % | 0 % | 51 % | 49 % |

![trot과 jump를 가르는 특징, 그리고 정책 프레임의 최근접 중심 배정](assets/figures/amp_disc_ref_feature_space.png)

> **위 표의 정책 행과 §5.4-1 첫 표의 관계.** 여기의 0.377은 **지형을 가리지 않고 합친 값**이라 첫 표의 flat 0.440 · gap 0.433보다 낮게 나온다 — 가장 낮은 stair(0.271, 표본 16,198개)가 풀을 끌어내리기 때문이다. 두 표는 같은 롤아웃 · 같은 필터를 쓰며 집계 단위만 다르다.

**정책의 움직임은 특징 공간에서 압도적으로 trot · walk 쪽에 붙어 있고, jump 중심에 배정되는 프레임은 사실상 0이다.** 그런데 같은 프레임들의 D는 0.377로 참조 어느 계열보다도 낮다. 두 사실을 합치면 결론은 하나다 — **판별자가 정책을 낮게 보는 이유는 "정책이 trot을 안 해서"가 아니다.** 정책은 이미 trot에 가장 가까운 무언가를 하고 있고, 판별자는 그 "가장 trot 비슷한 것"조차 참조 trot과 구별해 낸다. §5.4-1에서 AUROC가 모든 지형에서 0.999를 넘은 것과 같은 이야기를 gait 축에서 다시 본 셈이다.

**측정과 해석을 갈라 두면**: 측정된 것은 ① 정책 프레임의 계열 중심 거리 배정과 ② 그 프레임들의 D 값 둘뿐이다. 두 것이 동시에 성립하는 이유는 **추측의 영역**이다 — 평균 자세는 trot과 같은 이웃에 있으나 진폭 · 주기 · 네 발 접촉 타이밍 같은 **2차 통계**가 참조와 달라 D가 갈라내는 것일 수 있다. 본 측정은 최신 프레임 하나만 보므로 이 가설을 확인하지도 반박하지도 못한다. 확인하려면 history 10 frame 전체를 쓰는 별도 분석이 필요하다.

**(5) §5.1 pronk 문제와의 연결 — 해석.** 여기서 나온 측정값은 **"trot 참조를 썼더라면 pronk가 고쳐졌을 것"이라는 가설을 지지하지 않는다.** trot 참조는 이미 21% 들어 있었고 판별자는 그것을 참으로 인정한다. pronk가 남은 것은 참조 클립 구성보다 **style 보상의 크기** 쪽 문제에 가깝다 — trot을 완벽히 재현했을 때의 총보상 기여가 0.0332이고 정책이 실제로 받는 값이 0.0125이므로, 참조를 무엇으로 바꾸든 **여지는 0.021**이다(§2.5의 task 보상 규모와 나란히 놓고 읽어야 한다). 다만 이는 **체크포인트 한 개의 사후 측정**이고 trot 단독 참조로 학습한 arm이 없으므로 **인과 주장은 할 수 없다.**

**(6) trot 참조로 학습한 run은 존재하지 않는다.** 두 리포의 로그(`/home/lgb/IsaacLab-6.0/logs/rsl_rl`, `/home/lgb/IsaacLab/logs/rsl_rl`)에서 `params/env.yaml`의 `amp_motion_pkl`을 전수 확인한 결과 **79개 run이 전부 `imitation/go2_jump`**였다. 다른 참조로 학습한 판별자가 없으니, 아래 stairfail 격자처럼 "다른 참조로 배운 비평자로 교차 채점"하는 대조군은 만들 수 없었다. (`go2_amp` 계열 cfg의 `motion_files`에 보이는 `default.csv` · `sit.csv` 등은 별개 task의 것이다.)

> **측정의 한계.** (a) 클립당 8,192 window를 **균등하게** 뽑았으므로 클립별 수치는 균등 표본, 계열·전체 수치는 **길이 가중**이다 — 두 축을 섞어 읽으면 안 된다. (b) `sample_times`는 학습과 동일하게 truncation 없이 뽑으므로 짧은 클립은 history window가 경계에서 clamp된다. `walk1`(0.60초)은 30%가 clamp라 그림에 별표로 표시했고 다른 클립보다 잡음이 크다. 이 동작은 학습과 같게 두려고 일부러 고치지 않았다. (c) 최근접 중심 배정은 49차원 **최신 프레임 하나**만 쓴다 — history 10 frame의 시간 구조를 보지 않으므로 gait 위상 판별 도구로는 거칠고, 계열 중심이 등방적이라고 가정하는 것도 단순화다.

**재현**: `scripts/demos/amp_disc_ref_probe.py`(클립별 채점) + `scripts/demos/amp_disc_ref_plot.py`(그림·표).

```bash
CKPT=logs/rsl_rl/parkour_imitation_go2_symmetry_random_goal/2026-07-29_11-14-23/model_49999_FINAL_BEST.pt
D=reports/presentation/05_parkour_learning/assets
CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p scripts/demos/amp_disc_ref_probe.py --headless \
  --task Go2-ParkourImitation-Symmetry-RandomGoal-EasyEntry-v0 --checkpoint $CKPT \
  --num_envs 16 --samples_per_clip 8192 \
  --ref train=imitation/go2_jump --ref retarget=imitation/go2 \
  --out $D/data/amp_disc_ref_A_0729.npz
python scripts/demos/amp_disc_ref_plot.py --ref_npz $D/data/amp_disc_ref_A_0729.npz \
  --policy_npz $D/data/amp_disc_A_0729.npz --outdir $D/figures --warmup 20 --min_level 3
```

원자료는 `assets/data/amp_disc_ref_A_0729.npz`와 `amp_disc_ref_summary.json`이고, 정책 쪽은 §5.4-1의 `amp_disc_A_0729.npz`를 그대로 재사용했다(같은 롤아웃, 같은 `ssr ≥ 20` · `level ≥ 3` 필터).

##### 부표 — 참조 클립 42개 전체 (2026-09-02)

클립별 수치는 **클립당 8,192 window 균등 표본**이다. 계열·전체 수치(위 본문 표)는 길이 가중이므로 두 축을 섞어 읽으면 안 된다. `clamp 비율`은 history window가 클립 시작 경계에서 frozen frame으로 채워지는 표본의 비율이다.

| 클립 | 그룹 | 계열 | 길이 | 프레임 | clamp 비율 | 평균 D(x) | 평균 logit (±std) | 평균 r_disc |
|---|---|---|---|---|---|---|---|---|
| `jump` | 학습 | jump | 0.98 s | 60 | 18.3 % | 0.835 | +1.644 ± 0.253 | 0.1460 |
| `jump2` | 학습 | jump | 1.93 s | 117 | 9.3 % | 0.810 | +1.467 ± 0.226 | 0.1342 |
| `jump2_mirror` | 학습 | jump | 1.93 s | 117 | 9.3 % | 0.812 | +1.484 ± 0.241 | 0.1354 |
| `jump_down` | 학습 | jump | 1.73 s | 105 | 10.4 % | 0.824 | +1.569 ± 0.295 | 0.1411 |
| `jump_down_mirror` | 학습 | jump | 1.73 s | 105 | 10.4 % | 0.825 | +1.573 ± 0.280 | 0.1414 |
| `jump_down_slow` | 학습 | jump | 2.90 s | 175 | 6.2 % | 0.811 | +1.477 ± 0.280 | 0.1351 |
| `jump_down_slow_mirror` | 학습 | jump | 2.90 s | 175 | 6.2 % | 0.810 | +1.474 ± 0.268 | 0.1348 |
| `jump_mirror` | 학습 | jump | 0.98 s | 60 | 18.3 % | 0.836 | +1.650 ± 0.254 | 0.1464 |
| `jump_up` | 학습 | jump | 1.25 s | 76 | 14.4 % | 0.780 | +1.282 ± 0.238 | 0.1225 |
| `jump_up_mirror` | 학습 | jump | 1.25 s | 76 | 14.4 % | 0.779 | +1.270 ± 0.204 | 0.1217 |
| `jump_walk` | 학습 | jump | 1.95 s | 118 | 9.2 % | 0.786 | +1.307 ± 0.162 | 0.1239 |
| `jump_walk_mirror` | 학습 | jump | 1.95 s | 118 | 9.2 % | 0.783 | +1.292 ± 0.147 | 0.1229 |
| `run0` | 학습 | run | 1.45 s | 88 | 12.4 % | 0.847 | +1.750 ± 0.337 | 0.1534 |
| `run0_mirror` | 학습 | run | 1.45 s | 88 | 12.4 % | 0.844 | +1.731 ± 0.333 | 0.1521 |
| `run1` | 학습 | run | 2.93 s | 177 | 6.1 % | 0.861 | +1.861 ± 0.306 | 0.1609 |
| `run1_mirror` | 학습 | run | 2.93 s | 177 | 6.1 % | 0.859 | +1.838 ± 0.296 | 0.1593 |
| `run2` | 학습 | run | 1.17 s | 71 | 15.4 % | 0.857 | +1.824 ± 0.310 | 0.1584 |
| `run2_mirror` | 학습 | run | 1.17 s | 71 | 15.4 % | 0.857 | +1.826 ± 0.298 | 0.1584 |
| `walk` | 학습 | walk | 3.28 s | 198 | 5.5 % | 0.804 | +1.412 ± 0.057 | 0.1304 |
| `walk1` | 학습 | walk | 0.60 s | 37 | 30.0 % | 0.766 | +1.196 ± 0.199 | 0.1171 |
| `walk1_mirror` | 학습 | walk | 0.60 s | 37 | 30.0 % | 0.764 | +1.185 ± 0.172 | 0.1164 |
| `walk2` | 학습 | walk | 2.12 s | 128 | 8.5 % | 0.803 | +1.413 ± 0.140 | 0.1306 |
| `walk2_mirror` | 학습 | walk | 2.12 s | 128 | 8.5 % | 0.802 | +1.403 ± 0.141 | 0.1300 |
| `walk_mirror` | 학습 | walk | 3.28 s | 198 | 5.5 % | 0.804 | +1.415 ± 0.062 | 0.1306 |
| `walk_turn` | 학습 | walk | 3.45 s | 208 | 5.2 % | 0.786 | +1.306 ± 0.151 | 0.1238 |
| `walk_turn_mirror` | 학습 | walk | 3.45 s | 208 | 5.2 % | 0.787 | +1.312 ± 0.155 | 0.1242 |
| `trot0` | 학습 | trot | 3.45 s | 208 | 5.2 % | 0.749 | +1.094 ± 0.086 | 0.1107 |
| `trot0_mirror` | 학습 | trot | 3.45 s | 208 | 5.2 % | 0.749 | +1.095 ± 0.098 | 0.1108 |
| `trot1` | 학습 | trot | 3.45 s | 208 | 5.2 % | 0.749 | +1.095 ± 0.085 | 0.1107 |
| `trot1_mirror` | 학습 | trot | 3.45 s | 208 | 5.2 % | 0.749 | +1.094 ± 0.097 | 0.1107 |
| `go2_walk` | 재리타깃 | walk | 3.28 s | 198 | 5.5 % | 0.804 | +1.412 ± 0.061 | 0.1304 |
| `go2_walk1` | 재리타깃 | walk | 0.60 s | 37 | 30.0 % | 0.762 | +1.172 ± 0.189 | 0.1156 |
| `go2_walk1_mirror` | 재리타깃 | walk | 0.60 s | 37 | 30.0 % | 0.762 | +1.170 ± 0.162 | 0.1154 |
| `go2_walk2` | 재리타깃 | walk | 2.12 s | 128 | 8.5 % | 0.800 | +1.392 ± 0.142 | 0.1292 |
| `go2_walk2_mirror` | 재리타깃 | walk | 2.12 s | 128 | 8.5 % | 0.798 | +1.380 ± 0.143 | 0.1285 |
| `go2_walk_mirror` | 재리타깃 | walk | 3.28 s | 198 | 5.5 % | 0.804 | +1.415 ± 0.066 | 0.1306 |
| `go2_walk_turn` | 재리타깃 | walk | 3.45 s | 208 | 5.2 % | 0.786 | +1.306 ± 0.137 | 0.1238 |
| `go2_walk_turn_mirror` | 재리타깃 | walk | 3.45 s | 208 | 5.2 % | 0.787 | +1.315 ± 0.142 | 0.1244 |
| `go2_trot0` | 재리타깃 | trot | 3.45 s | 208 | 5.2 % | 0.726 | +0.974 ± 0.078 | 0.1036 |
| `go2_trot0_mirror` | 재리타깃 | trot | 3.45 s | 208 | 5.2 % | 0.725 | +0.970 ± 0.091 | 0.1034 |
| `go2_trot1` | 재리타깃 | trot | 3.45 s | 208 | 5.2 % | 0.725 | +0.973 ± 0.077 | 0.1035 |
| `go2_trot1_mirror` | 재리타깃 | trot | 3.45 s | 208 | 5.2 % | 0.725 | +0.971 ± 0.092 | 0.1035 |

#### 대조군 — 계단 실패 모델과의 정책 × 비평자 격자

§5.4의 "AMP 이전" arm 대신, **계단을 넘지 못하는 모델과 넘는 모델**을 짝지었다. `parkour_imitation_go2_symmetry/2026-06-16_10-17-39_amp_4.0_symmetry/model_50800.pt`(이하 stairfail)와 위 모델(이하 final)이다. 둘 다 AMP 관측 49차원, `apply_amp_on_terrain = True`, `disc_reward_type = bce`, `reward_coef = 0.08`이라 **discriminator 입력 규약이 완전히 같고**, actor/critic state_dict의 키와 shape도 완전히 일치한다. 그래서 두 정책을 **같은 6.0 EasyEntry 환경**에서 각각 굴리고, 각 롤아웃의 amp_obs를 **두 discriminator 모두**로 채점했다. 정책 효과(행)와 비평자 효과(열)가 분리된다.

| 지형 | final 정책 / D_final | final 정책 / D_stairfail | stairfail 정책 / D_final | stairfail 정책 / D_stairfail |
|---|---|---|---|---|
| **reference 모션** | 0.799 | 0.824 | 0.799 | 0.824 |
| flat | **0.441** | 0.656 | 0.290 | 0.286 |
| gap | 0.433 | 0.576 | 0.558 | 0.419 |
| step | 0.391 | 0.509 | 0.407 | 0.341 |
| hurdle | 0.371 | 0.515 | 0.412 | 0.348 |
| **stair** | **0.285** | **0.383** | **0.282** | **0.285** |

![정책 × 비평자 격자](assets/figures/amp_disc_compare_grid.png)

*지형별 평균 D(x). 막대 그룹이 지형, 색이 비평자(파랑 D_final / 주황 D_stairfail), 빗금이 stairfail 정책. 레벨 3\~6으로 맞춘 구간.*

- **계단은 네 조합 모두에서 최저다** (0.282\~0.383). 어느 정책을 어느 비평자로 재든 계단의 스타일 점수가 가장 낮다는 결론은 바뀌지 않는다.
- **평지에서 두 정책의 차이가 가장 크다.** final 정책은 자기 비평자에게 0.441, stairfail 정책은 0.286을 받는다. stairfail 정책은 **자기 자신의 비평자에게조차** 평지에서 낮은 점수를 받는다.
- **비평자는 서로 다른 관대함을 가진다.** stairfail의 discriminator는 final 정책을 평지에서 0.656으로 후하게 본다(자기 비평자 기준 0.441). 더 약한 생성자를 상대로 학습한 판별자의 결정 경계가 느슨하다는 신호로 읽힌다. 두 비평자가 참조 모션에 주는 점수는 0.799 / 0.824로 거의 같아, 이 차이가 비평자 척도의 전역 이동 때문만은 아니다.
- stairfail 정책은 hurdle·step·gap에서 평지보다 **높은** 점수를 받는다(0.348 / 0.341 / 0.419 vs 0.286). 평지 보행 자체가 참조에서 멀다는 뜻으로, §5.1의 "지표는 오르는데 보행이 이상하다"와 방향이 맞는다.
- 주행 활력도 두 모델이 다르다. 같은 1,200 step × 96 env에서 final은 평균 전진 속도 1.02 m/s에 리셋 131회, stairfail은 0.62 m/s에 리셋 327회다.

**해석과 그 한계.**

이 수치는 "지형에서 스타일 신호가 약해진다"를 정량화한 것이지, 그 원인을 특정하지는 않는다. 특히 **"지형이 D에게 OOD다"라는 읽기는 자동으로 따라오지 않는다** — 두 run 모두 `apply_amp_on_terrain = True`로 학습돼서 discriminator가 지형 롤아웃도 negative 샘플로 보며 학습했기 때문이다. 즉 D는 지형 동작을 본 적이 있고, 그럼에도 참조와 덜 닮았다고 판정한다.

기전을 좁히려고 §5.3의 TerrainStyle이 버리는 12개 지형 의존 열(root_height, lin_vel_b.z, 발끝 z 4개, pitch/roll 6개)과 남는 37개 열을, 참조 분포 기준 σ 단위 평균 이동량으로 각각 재봤다(final 정책, 레벨 3 이상).

| 지형 | 지형 의존 12열 (σ) | 지형 불변 37열 (σ) | 평균 D(x) |
|---|---|---|---|
| flat | 0.454 | **0.879** | 0.440 |
| gap | 0.722 | 0.955 | 0.433 |
| step | 0.831 | 0.963 | 0.388 |
| hurdle | 0.435 | 0.972 | 0.369 |
| stair | 0.703 | **1.104** | 0.271 |

**지형 불변 37열의 이동량이 D 순서와 정확히 역순으로 정렬된다**(flat 0.879 < gap 0.955 < step 0.963 < hurdle 0.972 < stair 1.104, D는 0.440 > 0.433 > 0.388 > 0.369 > 0.271). 지형 의존 12열은 그런 정렬을 보이지 않는다(hurdle이 최소, step이 최대). 지형 5개짜리 순위 일치일 뿐 인과는 아니지만, 방향을 하나 시사한다: **격차의 소재가 "어느 높이에서 어떤 자세로"가 아니라 "관절이 어떻게 협응하는가" 쪽에 있다면, 지형 의존 열을 지우는 TerrainStyle-v0만으로는 이 격차가 사라지지 않는다.** 확정하려면 입력 열을 지운 counterfactual 평가가 필요하고, 여기서는 하지 않았다.

**caveat.** ① seed 1개, 체크포인트 각 1개다. ② stairfail arm은 **5.1 저장소에서 학습된 체크포인트를 6.0 환경에서 굴린 것**이다. 정책 네트워크의 키·shape가 완전히 일치해 로드는 무손실이지만, 시뮬레이터·지형 생성이 5.1→6.0으로 바뀐 도메인 시프트가 그 arm의 낮은 점수에 얼마나 기여하는지는 분리하지 못했다(§6의 "6.0에서 from-scratch가 장애물을 못 넘었다"가 그 시프트의 크기를 보여준다). "계단 실패 모델" 라벨은 이 한계를 안고 읽어야 한다. ③ `apply_amp_on_terrain = False`로 학습된 parkour AMP 체크포인트는 IsaacLab / IsaacLab-6.0 양쪽 로그에 **하나도 남아 있지 않다**(2026-06-16 run의 `params/env.yaml`도 run 시작 시각에 기록된 `true`다). 따라서 "§5.3 수정 전후"를 체크포인트로 대조하는 arm은 존재하지 않는다. ④ 두 run의 AMP 관측은 49차원이다 — TerrainStyle-v0(37차원)이 아니므로, 위 수치는 TerrainStyle의 효과 측정이 아니다. ⑤ 표는 레벨 3 이상만 담는다. 레벨을 섞으면 지형 버킷이 평지 쪽으로 희석돼 격차가 줄어든다. ⑥ 롤아웃은 `enable_random_goal = False`로 돌렸다 — 켜두면 졸업한 env가 예약 flat 열로 순간이동해 샘플 라벨이 뒤바뀌기 때문이다. 모든 버킷에 똑같이 적용되므로 버킷 간 비교는 유효하지만, 정책이 학습된 명령 분포와는 다르므로 **D의 절대값**은 학습 시점 값이 아니다.

**블록 B(최종 배포 정책)에 대하여.** 표의 블록 B는 §9의 LiDAR SL-Grid 정책을 같은 프로토콜로 잰 것이다. 원자료는 `assets/data/amp_disc_C_0810_lidar.npz`, 수치는 `amp_disc_C_0810_lidar_summary.json`이다. 아래 그림 네 장(violin, 지형별 보상, 계단 시계열, 레벨별 D)은 모두 **블록 A** 기준이며, 블록 B의 같은 그림은 만들지 않았다.

**재현** — 스크립트는 `scripts/demos/amp_disc_probe.py`(롤아웃 + discriminator 평가, `--extra_disc name=ckpt`로 다른 run의 판별자를 같은 관측에 교차 적용), `scripts/demos/amp_disc_plot.py`(단일 run 그림·표), `scripts/demos/amp_disc_compare.py`(정책 × 비평자 격자)다. 원자료는 `reports/presentation/05_parkour_learning/assets/data/`의 `amp_disc_A_0729.npz` / `amp_disc_B_0616.npz`, 수치는 같은 폴더의 `amp_disc_summary.json` / `amp_disc_compare_summary.json`.

```bash
A=logs/rsl_rl/parkour_imitation_go2_symmetry_random_goal/2026-07-29_11-14-23/model_49999_FINAL_BEST.pt
B=/home/lgb/IsaacLab/logs/rsl_rl/parkour_imitation_go2_symmetry/2026-06-16_10-17-39_amp_4.0_symmetry/model_50800.pt
T=Go2-ParkourImitation-Symmetry-RandomGoal-EasyEntry-v0

CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p scripts/demos/amp_disc_probe.py --headless --task $T \
  --checkpoint $A --extra_disc "stairfail=$B" --num_envs 96 --rollout_steps 1200 \
  --expert_samples 40960 --max_init_level 6 --out .../amp_disc_A_0729.npz
CUDA_VISIBLE_DEVICES=3 ./isaaclab.sh -p scripts/demos/amp_disc_probe.py --headless --task $T \
  --checkpoint $B --extra_disc "final=$A" --num_envs 96 --rollout_steps 1200 \
  --expert_samples 40960 --max_init_level 6 --out .../amp_disc_B_0616.npz

./isaaclab.sh -p scripts/demos/amp_disc_plot.py --npz .../amp_disc_A_0729.npz \
  --outdir .../assets/figures --warmup 20 --min_level 3
./isaaclab.sh -p scripts/demos/amp_disc_compare.py --npz_a .../amp_disc_A_0729.npz \
  --npz_b .../amp_disc_B_0616.npz --label_a final --label_b stairfail \
  --cross_key_a stairfail --cross_key_b final --outdir .../assets/figures \
  --warmup 20 --min_level 3 --max_level 6
```

### 5.5 부수 성과 — estimator 누락 버그 (§2.2 참조)

같은 세션에서 AMP 러너가 estimator를 아예 만들지 않아 actor가 시뮬레이터 ground-truth 속도를 읽고 있던 것을 발견·수정했다. 이 수정 이전의 모든 AMP 체크포인트는 배포 불가였다.

---

## 6. 단계 3.5 — EasyEntry: 커리큘럼에 첫 계단을 놓다 (2026-07)

Isaac Sim을 5.1 → 6.0으로 올린 뒤, from-scratch 학습이 **장애물을 하나도 넘지 못하는** 상태가 됐다. 이 단계는 나머지 전부를 가능하게 만든 전제조건이라 따로 적는다.

### 6.1 진단 — "seed 운"이 아니라 "재진입 실패"

seed 1/2/3을 동시에 돌려 iteration 1600에서 판정했다. 세 시드 전부 동일하게 실패:

| 지형 | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| flat | 5.87 | 5.78 | 5.90 |
| hurdle | 0.00 | 0.01 | 0.00 |
| gap | 0.06 | 0.03 | 0.03 |
| step | 0.00 | 0.00 | 0.00 |
| stair | 0.00 | 0.00 | 0.00 |

지형별 시계열을 5.1과 나란히 놓자 진짜 병목이 드러났다.

| iter | 6.0 flat | 6.0 hurdle | 6.0 gap | 5.1 flat | 5.1 hurdle | 5.1 gap |
|---|---|---|---|---|---|---|
| 100 | 0.58 | 0.21 | 0.16 | 0.55 | 0.25 | 0.35 |
| 300 | 3.18 | 0.00 | 0.03 | 2.87 | 0.00 | 0.02 |
| 800 | 6.41 | 0.00 | 0.04 | 6.37 | 0.05 | 0.52 |
| 1600 | 5.94 | **0.00** | **0.05** | 5.71 | **7.20** | **6.31** |

**iteration 300의 장애물 붕괴는 양 엔진 공통의 정상 동작이다.** 로봇이 먼저 평지 보행을 배우는 동안 커리큘럼이 장애물을 level 0으로 강등시킨다. 갈리는 지점은 iteration ~800의 **재진입**이다 — 5.1은 다시 올라가고, 6.0은 영원히 0에 머문다.

원인은 커리큘럼 사다리의 **첫 칸이 너무 높았다**는 것이다. level 0의 장애물 크기는 각 range의 하한값이고, 레벨은 0에서 clamp된다. 6.0 walker는 level 0조차 통과하지 못하는데 그 아래 칸이 없으므로 **gradient가 존재하지 않는다.**

### 6.2 수정 — 하한만 낮춘다

`ParkourImitationRandomGoalEasyEntryEnvCfg`를 새 task로 등록하고, 각 range의 **하한만** 낮췄다. 상한·보상·AMP·센서·random-goal은 전부 상속 그대로 두어 단일변수 대조가 되게 했다.

| 지형 | 기존 하한 | EasyEntry 하한 | 상한 (불변) |
|---|---|---|---|
| hurdle | 0.05 m | **0.01 m** | 0.30 m |
| gap | 0.05 m | **0.02 m** | 0.80 m |
| step | 0.10 m | **0.02 m** | 0.60 m |
| stair | 0.05 m | **0.02 m** | 0.25 m |

결과 (iteration ~1900): step **0.373 m**, stair **0.165 m** 통과 — 어떤 6.0 from-scratch run도 움직인 적 없는 지표다. baseline은 0.100 m / 0.050 m를 끝내 못 넘었다.

이후 모든 task ID에 `EasyEntry`가 붙는 것은 이 때문이다.

> **방법론 교훈 (이 세션에서 오탐 2건 발생)**: (1) iteration 55의 초기 transient를 재진입으로 오독 → 판정 게이트를 붕괴 완료 이후로 이동. (2) 커리큘럼 **레벨 인덱스를 config 간에 직접 비교**했는데, EasyEntry가 사다리를 재척도했으므로 무효다. 반드시 물리 단위로 환산해야 한다: `size = lo + (level/10) × (hi − lo)`.

---

## 7. 단계 4 — 2.5D height scan에서 3D 지각으로 (2026-06 ~ 08)

### 7.1 왜 — 2.5D는 터널을 표현할 수 없다

height_scan은 각 (x, y)에 스칼라 하나다. 터널(crawl)이나 오버행은 같은 (x, y)에 바닥과 천장이 동시에 있으므로 **원리적으로 담을 수 없고**, 하향 ray가 천장 윗면을 찍어 관측이 오염된다. 3D 전환의 근거는 성능이 아니라 **표현력**이다.

**실측 시각화 (2026-08-25)** — 같은 crawl 터널 tile·같은 base 위치에서 다섯 인지 방법이 실제로 보는 것을 덤프해 나란히 그렸다.

![crawl 인지 5-way 비교](assets/figures/crawl_perception_5way_inside.png)
*터널 내부 자세 기준. 윗줄은 각 방법의 고유 표현, 아랫줄은 같은 관측을 터널 x–z 단면(회색 = 실제 기하)에 투영. height_scan은 187칸 중 132칸이 바닥이 아니라 천장 윗면(0.99 m)을 찍어 **터널 내부가 통벽이 된다**. 진입 전 자세 버전: [`crawl_perception_5way_approach.png`](assets/figures/crawl_perception_5way_approach.png)*

실측에서 결함이 한 단계 더 강하게 확인됐다: 천장 칸의 관측값($0.2675 - 0.986 - 0.3 = -1.018$)이 clip(−1, 1) 하한 아래라 **정확히 −1.0으로 포화**된다. 187개 관측값이 단 2종류로 붕괴해 **1 m 낭떠러지와 관측상 구분이 불가능**하고, 로봇이 딛어야 할 터널 바닥은 한 칸도 관측에 들어오지 않는다 ([상세 플롯](assets/figures/crawl_heightscan_pollution.png)). 단, 이 오염 관측을 실제 입력으로 받아 학습한 정책은 없다 — 2.5D baseline은 crawl 비율이 0이고, 3D teacher는 scan 슬롯이 네트워크에 연결되지 않는다. 즉 순수하게 표현 수준의 결과이며, 그것이 2.5D arm에 crawl을 넣지 않은 이유다.

| 주행 중 센서 오버레이 영상 (crawl) | 내용 |
|---|---|
| [height_scan 오버레이](assets/videos/23_crawl_percep_heightscan_overlay.mp4) | 하향 격자가 터널에서 천장 윗면으로 올라타 2.5D 관측이 통벽이 되는 순간. ⚠ **주행 정책은 voxel teacher**(2.5D 정책은 crawl 미학습) — 화면에 그리는 관측만 height_scan |
| [SL-Grid student 오버레이](assets/videos/24_crawl_percep_slgrid_student_overlay.mp4) | 배포 student가 Mid-360 히트만으로 천장을 인지하며 웅크려 통과 (1.04 m/s, 웅크림 z_range 0.088) |

원자료·나머지 오버레이 3편(clearance / voxel ray / GT voxel)·생성 스크립트: `reports/go2_parkour/_comparisons/crawl_perception_visualization/`.

두 후보를 teacher로 추가했다.

- **3D clearance (294차원)**: 전방 반구를 21 방위각(−100°~+100°) × 14 고도각(−75°~+60°)으로 쏘아 방향별 첫 지오메트리까지의 거리. `max_distance=4.0 m`, `(d − 2.0)/2.0`로 정규화. `clearance_as_scan=True`로 height_scan과 **같은 `obs["scan"]` 슬롯을 대체**하므로 통제 실험이 된다.
- **voxel occupancy (7,371차원)**: clearance와 **동일한 294 ray, 동일한 거리값을 재사용**하되 3D 격자에 통과시켜 점유로 인코딩한다. actor의 지형 인코더만 scandot MLP → VoxelEncoder(Z-grouped 2D-CNN)로 교체.

### 7.2 계산 비용 — 3D 전환은 공짜다

50k iteration 기준 wall-clock:

| 방법 | 총 경과 | s/iter | 상대속도 |
|---|---|---|---|
| height_scan | 63.11 h | 4.544 | **1.00×** |
| clearance | 61.02 h | 4.394 | 0.97× |
| voxel | 62.48 h | 4.498 | 1.01× |

**3D 전환의 계산 오버헤드는 사실상 0이다.** 이것이 이후 선택을 자유롭게 만들었다.

### 7.3 성능 — "3D가 2.5D보다 나쁘다"는 관측 조건의 산물이었다

2026-07-06 ablation에서는 height_scan 5.78 > clearance 5.34 > voxel 5.25로 3D가 졌다. 그러나 그 run들은 **커리큘럼 floor-pinning 상태**였다 — flat 표본이 n≈9,900인 반면 hurdle은 n=72~183, 즉 로봇 대부분이 평지에 갇혀 있고 장애물 지형 수치는 소수 표본의 잡음이었다.

EasyEntry(§6)로 floor-pinning을 푼 뒤 다시 재면 부호가 뒤집힌다. 지형당 표본이 n=194~678로 고르다.

**지형별 커리큘럼 레벨 (마지막 1000 iteration 평균)**

| 지형 | height_scan | 3D clearance | 3D voxel(ray) | 3D voxel GT |
|---|---:|---:|---:|---:|
| flat | **5.53** | 4.78 | 5.22 | 5.06 |
| gap | 3.78 | 6.14 | 6.41 | **6.93** |
| hurdle | 6.06 | **6.18** | 6.01 | 6.04 |
| stair | 6.21 | 5.49 | **6.61** | 2.68 |
| step | 5.93 | **7.39** | 6.54 | 5.93 |
| **crawl 제외 평균** | **5.50** | **6.00** | **6.16** | 5.33 |
| *(참고) crawl* | *지형 없음* | *2.72* | *4.94* | *3.20* |

> ★ **crawl을 빼야 공정하다.** height_scan baseline은 5지형 × 0.20으로 crawl을 아예 안 밟고, 3D 셋은 5지형 × 0.15 + crawl 0.25다. crawl은 3D의 최약 지형인데 baseline은 시도조차 하지 않으므로, 그대로 평균 내면 **3D만 baseline이 치르지 않는 벌점을 진다.** 로그의 전체 평균만 보면 clearance 5.01 < height_scan 5.49로 정반대로 읽힌다.

crawl을 빼면 voxel이 height_scan보다 **+0.66**, clearance가 **+0.49** 높다. 가장 큰 격차는 **gap(3.78 vs 6.41, +2.6 레벨)** — 구멍의 3D 기하는 하향 격자로 잡기 어렵고, 이것이 차이의 대부분을 만든다.

**창 길이 민감도** — 흔들리는 것은 height_scan뿐이다:

| | @200 | @1000 | @3000 |
|---|---:|---:|---:|
| height_scan gap | **6.45** | 3.78 | 3.84 |
| height_scan 비-crawl 평균 | 6.00 | 5.50 | 5.57 |
| clearance 비-crawl 평균 | 6.03 | 6.00 | 6.00 |
| **voxel(ray) 비-crawl 평균** | **6.24** | **6.16** | **6.11** |

1000과 3000이 일치하고 200만 튄다 — 마지막 400 iteration의 회복 구간을 200 창이 통째로 잡은 것이라 6.45는 실력이 아니라 운 좋은 스냅샷이다. 3D 둘은 ±0.05 안에 있다.

**영상**

| 파일 | 비교 내용 | 결론 / 주의 |
|---|---|---|
| [`assets/videos/03_gap_3up_heightscan_vs_voxel_vs_voxelGT.mp4`](assets/videos/03_gap_3up_heightscan_vs_voxel_vs_voxelGT.mp4) | gap 지형 3분할 — height_scan / voxel / voxel GT | 각 정책이 실제로 읽는 것을 오버레이. gap에서 2.5D와 3D의 격차가 가장 크게 드러나는 지형 |
| [`assets/videos/04_step_3up_heightscan_vs_voxel_vs_voxelGT.mp4`](assets/videos/04_step_3up_heightscan_vs_voxel_vs_voxelGT.mp4) | step 지형 3분할 | 커리큘럼 레벨은 자기선택 난이도이므로 완주율과 같은 것을 재지 않는다 |

![crawl 터널 통과 — voxel teacher](assets/figures/voxel_crawl_best.png)
*3D voxel teacher가 crawl 터널을 통과하는 프레임. 2.5D height_scan은 이 지형을 학습 분포에 넣을 수조차 없다 (하향 ray가 천장 윗면을 찍음). 좌우 비교본은 [`clearance_crawl_best.png`](assets/figures/clearance_crawl_best.png).*

### 7.4 GT voxel — 표집을 열거로 바꾸다

기존 voxel 격자는 clearance 광선의 **hit point가 떨어진 칸 하나만** 1로 찍는다. 294개 발산 광선으로 7,371칸 중 **70~80칸**만 점유가 되고, 세 가지 결함이 따라온다.

| 결함 | 내용 |
|---|---|
| **거짓 음성** | 광선이 안 맞은 단단한 지형이 `0` — 이진화 후 허공과 구별 불가 |
| **표면만, 부피 아님** | 지면 *아래*가 안 채워져, 바닥의 구멍과 단단한 바닥이 얇은 껍질 차이뿐 |
| **자세 의존 flicker** | 로봇이 걸으면 hit point가 이웃 칸으로 옮겨가 정지 지형에서도 점유 집합이 진동 |

**표집을 열거로 바꿨다.** 격자 기둥마다 광선 하나씩 — 아래로 하나, 위로 하나 (27 × 21 × 2 = **1,134개**):

$$\text{occupied}(i_x, i_y, i_z) \iff z_{i_z} \le z_{\text{ground}}(i_x,i_y) \;\;\lor\;\; z_{i_z} \ge z_{\text{ceil}}(i_x,i_y)$$

두 광선 모두 **원점에서 출발한다** — 로봇 base는 그 기둥에서 자유 공간임이 보장된 유일한 점이기 때문이다. 볼륨 위에서 아래로 쏘면 crawl 터널의 천장을 지면으로 오독한다.

결과: 점유 셀이 72~80칸 → **2,268칸(28~31배)**, 누락 기둥 0.

### 7.5 관측 버그 2건 — 허위 천장, 허위 바닥

2026-08-03에 GT voxel 관측에서 **두 개의 결함**을 발견해 법선 기반 분류로 해소했다: 옆 블록을 천장으로 오인하는 것과, 천장 슬래브를 지면으로 오인하는 것.

이 버그의 영향은 §7.3 표의 voxel GT 열에서 바로 보인다 — **stair 2.68로 붕괴**(pinned stair에서 567 기둥 중 236개가 허위 천장), **gap 6.93으로 전 arm 최고**(gap은 마운트보다 높은 기하가 없어 이 버그가 발생하지 않는 유일한 지형).

| 파일 | 내용 |
|---|---|
| [`assets/videos/05_voxelGT_falseceiling_before_vs_after_step.mp4`](assets/videos/05_voxelGT_falseceiling_before_vs_after_step.mp4) | 허위 천장 수정 전/후 (step 지형). 수정 후 재학습본은 **별도 baseline family**이므로 §7.3 표에 섞어 읽으면 안 된다 |

### 7.6 예측이 빗나간 기록

GT voxel을 만들며 **측정 전에** "gap이 가장 크게 좋아질 것"이라 적어뒀다 — 바닥 구멍이 "표면 셀 몇 개 없음"에서 "기둥 전체가 비어 있음"으로 바뀌어 신호 변화량이 가장 크기 때문이다.

**틀렸다.** GT teacher는 pinned gap에서 오히려 기존 teacher보다 낮았다. 즉 **표집이 성긴 것은 teacher의 병목이 아니었다** — 기존 teacher는 성긴 입력만으로도 이 난이도에서 충분했다.

---

## 8. 단계 5 — LiDAR 전환: 붕괴, distillation, 그리고 표현의 발견 (2026-07 ~ 08)

이 단계가 프로젝트의 중심이다. 목표는 명확했다 — **시뮬레이터 특권 정보 없이, 실제 Go2에 달 수 있는 센서만으로 같은 성능을 낸다.**

### 8.1 센서 — Livox Mid-360

| 항목 | 값 |
|---|---|
| 마운트 | `pos = (0.3336, −0.0005, 0.0501)`, `rot` = dome-down 180° ⊗ pitch-up 30° (SLAM 실측 붐 마운트) |
| 광선 | 24,000 samples, `ray_alignment="base"` — **광선이 몸통과 함께 기울어진다** |
| 측정률 | `update_frequency = 10.0` Hz (실기와 동일) |
| 노이즈 | 거리 ±2 cm (1σ, 데이터시트), dropout 10% (sim-to-real DR) |

시뮬레이션은 warp `RayCaster` 기반이며 RTX 렌더링이 아니다.

### 8.2 1차 시도 — 완전 붕괴

첫 LiDAR student는 point cloud를 **각도 range image**(`K=3, C=2, H=24, W=96` → 13,824차원)로 인코딩해 정책에 연결했다. 결과는 negative result 중에서도 극단이었다.

| 지표 | LiDAR student | teacher 3종 범위 | 악화 |
|---|---|---|---|
| terrain_overall | **0.020** | 4.69 ~ 5.01 | −99.6% |
| mean_reward | **2.566** | 24.25 ~ 26.73 | −90% |
| collision | **−15.277** | −0.30 ~ −1.17 | 13~50배 |
| goal_reached | **0.018** | 4.10 ~ 4.79 | ≈0% |

**flat(평지)조차 0.020**이라는 것이 핵심이다. "지형이 어려워서 못 넘는" 문제가 아니라 **기립·전진 자체를 학습하지 못한** 총체적 실패다.

동일 재생 하네스로 측정한 전진속도가 5지형 전부 −0.005 ~ −0.023 m/s(≈0/음수)인 반면 teacher 3종은 전 지형 ~1 m/s였다. **하네스 버그가 아니라 정책 붕괴임이 정량 확증됐다.**

![distillation student가 읽는 raw Mid-360 히트](assets/figures/distill_k3_rangeimage_sheet.png)
*각도 range image 계열 student(K=3)의 지형별 프레임 contact sheet. 자홍색 점군이 Mid-360 raw 히트다. 로봇 주변은 빽빽하고 멀수록 급격히 성기다 — 이 기하가 §8.4의 "행 간격 붕괴"를 만든다.*

### 8.3 2차 시도 — distillation 프로그램 (2026-07-27 ~ 31)

당시 가설은 **"정면 blind band(로봇 바로 앞 사각지대) 때문에 실패한다"**였다. 그래서 frozen voxel teacher를 두고 student를 distill하며 축을 하나씩 격리했다.

$$\mathcal{L}_{\text{distill}} = \big\| \pi_{\text{student}}(o_{\text{lidar}}) - \pi_{\text{teacher}}(o_{\text{voxel}}) \big\|^2 \quad \text{(per-step action MSE)}$$

| arm | 표현 | 격리한 축 |
|---|---|---|
| K=3 / K=5 / K=10 / K=15 | 각도 range image, 프레임 스택 길이 변화 | 시간 창 길이 |
| **A1-0** | **미터 격자 (teacher와 동일 프레임)** | **표현** |
| A1-1 | 격자 + 정합 시간 누적 | blind band 채우기 |
| Ceiling | teacher 격자를 student에 그대로 주입 | 관측 불일치 자체 |

### 8.4 ★ 발견 — 병목은 blind band가 아니라 표현이었다

**A1-0(미터 격자)은 blind band를 전혀 채우지 않는다** — 밴드 점유율 0.0129로 원래 blind 측정과 같다. 그런데 teacher와의 격차 대부분을 닫는다.
**A1-1은 밴드를 9.8배 채운다**(recall 0.09 → 0.90). 그런데 추가 이득이 **0**이다.

원인은 **각도 binning이 이미 보이는 정보를 파괴한 것**이었다. range image의 결함 두 가지:

1. **깊이 축이 뭉개진다.** `scatter_reduce(amin)`이 "방향은 비슷한데 깊이는 다른" 광선을 가장 가까운 것 하나로 줄여, 전방 히트 약 13,813개가 약 1,348 bin으로 축소된다.
2. **행 간격이 거리에 따라 붕괴한다.** 0.74 m에서 0.11 m였던 간격이 4.05 m에서 **1.69 m**로 벌어지고, 24행 중 17행만 지면을 본다.

> **이 90%를 "정보 손실"로 읽으면 안 된다.** 개수의 상당 부분은 같은 표면을 때린 중복 광선이고, 미터 격자도 이진이라 한 voxel의 여러 히트를 하나로 병합한다. **두 표현 다 압축한다.** 차이는 **어느 축을 붕괴시키느냐**다.

| 표현 | 병합하는 것 | 기하학적 의미 |
|---|---|---|
| 각도 bin (`amin`) | 거의 같은 방향의 **서로 다른 깊이** | 결정적인 축을 파괴 |
| 미터 voxel | **같은 3D 위치**의 중복 히트 | 중복 축을 병합 (무손실에 가까움) |

그리고 0~2 m 볼륨에서 student 격자는 privileged teacher보다 **6.6~11배 촘촘하다** (Mid-360 약 24,000 광선 vs clearance 294). **정보는 원래 student 쪽에 더 많았고, 표현이 그것을 버리고 있었다.** 관측이 13,824 → 7,371로 **작아지면서** 성능이 오른 것이 이 해석과 정합한다.

**이득의 정체 — tilt 제거와 일대일 대응**

같은 조건(레벨 고정)에서 지형별로 재면, 완주 이득이 tilt(전복) 감소량을 거의 그대로 따라간다:

| 지형 | 완주율 배율 (K=3 → A1-0) | tilt 변화 |
|---|---|---|
| **gap** | **2.9×** | 0.350 → 0.150 (**−0.200**) |
| **step** | 1.25× | 0.060 → 0.017 (−0.043) |
| **hurdle** | 1.09× | 0.000 → 0.000 (0.000) |

tilt가 애초에 없는 hurdle에서는 이득도 거의 없다. 즉 미터 격자의 효과는 "전반적으로 더 잘 본다"가 아니라 **"void 엣지에서 넘어지는 것을 막는다"**이다. 인과 사슬:

> **각도 binning이 근거리 기하를 파괴 → void 엣지 근처 발 배치 오류 → 몸이 platform보다 0.094 m 아래로 가라앉음 → 0.14초 만에 tilt.** (teacher는 동일 조건에서 tilt 0건)

**빗나간 예측도 기록해 둔다.** 측정 전에는 "step에서 gap 이상의 이득"을 예측했다 — yaw 정렬 격자가 step의 pitch 의존 결손(상면이 2.6~2.7° 더 nose-up에서야 처음 보임)을 구조적으로 없애기 때문이다. 틀렸다. **결손의 존재를 성능 병목과 혼동한 것이다.**

### 8.5 SL-Grid — teacher를 버리고 처음부터 RL

표현이 병목이라면, 그 표현으로 **처음부터 RL을 하면 된다.** distillation은 구조적 천장을 물려받기 때문이다:

> student의 격자는 실제 센서에서, teacher의 격자는 특권 clearance 광선에서 온다. 따라서 `obs_student → action_teacher`는 함수가 아니라 **분포**이고, MSE를 최소화하면 조건부 평균으로 수렴한다. RL에는 그런 천장이 없다 — 정책이 **자기 자신의 관측 가능성 하에서** 달성 가능한 것을 최적화한다.

이 판단을 뒷받침한 것이 **Ceiling arm**이다. student에게 teacher의 격자를 그대로 주면 자기 teacher를 근소하게 앞선다. 즉 **distillation의 손실은 목적함수가 아니라 관측 불일치에서 온다.**

**SL-Grid(Scattered LiDAR Grid)**는 실기 Mid-360 히트를 teacher의 복셀 격자와 **똑같은 좌표계·똑같은 텐서 형태**로 직접 채우는 방법이다.

#### 산란 파이프라인 — 4단계

![SL-Grid 산란 파이프라인 4단계](assets/figures/slgrid_pipeline_overview.png)
*계단 L6 실주행 한 프레임(t=100, 피치 −26.7°)의 실제 데이터로 그린 파이프라인. 왼쪽부터 raw Mid-360 스캔(24,000 광선, `ray_alignment="base"`) → **①** 자기 가림 마스크(2,504 광선 폐기, 이 프레임 유효 히트 16,965) → **②** yaw 정렬(빨강 = base 정렬, 초록 = yaw 정렬, 주황 사각 = teacher 복셀) → **③** 27 × 21 × 13 = 7,371 이진 격자(이 프레임 568 셀 점등, 색 = 그 열에서 점유된 가장 높은 z). 맨 아래 띠가 텐서 형태의 흐름과 `obs_groups` 라우팅이다. 본문의 "약 24k 광선"은 **총 광선 수**를 가리키고, 실제로 격자에 들어가는 **유효 히트**는 260 프레임 평균 17,282개(72.0%)다.*


**① 자기 가림 제거.** 몸통이 자기 광선을 막는 영역을 미리 계산한 정적 방위/고도 격자(`body_occ_azel_grid.npy`, 180×360, 1°/bin)로 마스킹한다. **방향 기반**이라 센서 회전 누적에 면역이다.

$$\text{hit valid} = (d < d_{\max}) \wedge \neg\,\text{occluded}$$

![① 자기 가림 az/el 격자](assets/figures/slgrid_pipeline_mask.png)
***좌**: `body_occ_azel_grid.npy` 원본. 64,800 bin 중 **2,607개(4.02%)** 가 가림으로 표시돼 있고, 가려진 고도는 −48°~+24°에 걸치며 그중 63%가 수평선 아래다. 2,607개 중 2,002개가 전방 ±60° 섹터에 몰려 있다. **우**: 같은 마스크를 실제 Mid-360 광선 24,000개에 먹인 결과 — **2,504개(10.4%)** 가 버려진다. bin 비율(4.02%)과 광선 비율(10.4%)이 다른 것은 광선 밀도가 방위·고도에 균일하지 않기 때문이다. ⚠️ ⚠️ 로브가 **전방**에 있는 것은 물리적 자기 가림이 아니다 — 원인은 아래 정정 참조.*

> **(2026-09-03 정정) 전방 로브의 원인 — 마스크는 boom 마운트가 아니라 base 원점에서 계산됐다.** 생성 스크립트는 5.1 저장소 `~/IsaacLab/_workspace/parkour_imitation_lidar/r1_throughput/compute_body_mask.py`(2026-06-30)에서 찾았다. 이 스크립트는 센서 위치로 `lidar.data.pos_w`를 쓰는데, IsaacLab RayCaster의 `data.pos_w`는 **마운트 offset이 빠진 부모 프림(`/Robot/base`) 위치**다(offset 포함 위치는 5.1 `lidar_sensor.py`의 `_get_true_sensor_pos()`가 따로 있다). 그래서 광선이 0.334 m 앞의 Mid-360 마운트가 아니라 몸통 중심에서 출발했고, `min_range` 0.2 m만큼 옮긴 시작점이 머리 껍데기 안쪽·힙 근처에 놓여 **앞쪽의 머리 내벽과 네 다리**가 "가림"으로 기록됐다. MuJoCo Go2 모델로 독립 재계산하면(아래 그림, `assets/render_slgrid_pipeline/occlusion_mask_check.py`): base 원점에서 쏜 마스크는 저장본과 IoU 0.60에 전방/후방 광선 4,539/373으로 같은 패턴이고, 의도한 마운트 위치에서 쏜 마스크는 IoU 0.06에 **전부 후방**(1,547개, 6.4%, 고도 −62°~−13°)이다. 결과적으로 현행 마스크는 (a) 실제로 몸통·앞다리에 가려지는 후방 광선은 거의 거르지 않고(그 광선들은 6.0 ray caster가 지면만 캐스팅하므로 몸통을 통과해 뒤쪽 지면을 찍는다), (b) 가려지지 않는 **전방 ±60°, 고도 −20°~+23°의 광선 10.4%를 버린다** — 정면 상부 반구(허들 윗면·crawl 천장 높이)의 시야가 학습·배포 모두에서 통째로 빠져 있다는 뜻이다. 마스크는 학습·배포에 동일하게 적용되므로 sim2real 불일치는 아니지만, 재생성 시 관측 분포가 바뀌어 기존 체크포인트에 그대로 끼울 수는 없다.

![① 마스크 전방 로브의 원인 — 저장본 vs MuJoCo 재계산(마운트 / base 원점)](assets/figures/slgrid_pipeline_mask_rootcause.png)
*좌: 저장된 마스크. 중: 의도한 boom 마운트(0.334, 0, 0.05)에서 캐스팅 — 후방·하방에만 가림. 우: 스크립트가 실제로 쓴 base 원점에서 캐스팅 — 저장본과 같은 전방 로브 + 다리 줄무늬 4개. IoU는 1° 팽창 후 저장본 대비.*



> **(2026-09-03) 마스크 v2 재생성과 구/신 비교.** 위 정정에 따라 마스크를 다시 만들었다. 생성 스크립트는 `assets/render_slgrid_pipeline/compute_body_mask_v2.py`이고, v1과 마찬가지로 5.1 저장소 환경에서 Go2 USD 시각 메시에 대고 캐스팅하되 세 가지를 고쳤다. (a) 센서 위치를 `lidar._get_true_sensor_pos()`로 바꿨다 — 로그에 찍힌 센서−base 델타가 `(0.3336, −0.0005, 0.0503)`으로 `cfg.offset.pos`와 정확히 일치한다. 캐스팅은 **6.0 base 프레임 방향 24,000개를 5.1 씬의 base 쿼터니언으로 월드로 돌린 뒤 그 마운트 위치에서 직접 쏜 것**이고, 가려진 방향은 다시 **6.0 base 프레임 방향**으로 binning했다 — 5.1 방향으로 쏜 결과를 6.0 방향에 재매핑한 것이 아니다. 마스크 런의 base 쿼터니언은 wxyz `(1.0, 0.00077, −0.00028, 0.00007)`로 yaw ≈ +0.008°라 world ≈ base다(별도 프로브 런도 yaw −0.054°로 같다). v1 생성 시에도 yaw ≈ 0이었다는 근거는, 저장된 v1 마스크의 전방 로브가 az = 0에 대칭이고 정의상 yaw = 0인 base 프레임 MuJoCo 재계산과 팽창 IoU 0.60으로 맞는다는 점이다 — yaw가 0이 아니었다면 로브가 az = 0에서 돌아가 이 일치가 깨진다. (b) **격자를 만드는 기준 방향을 5.1 센서가 아니라 6.0 런타임 덤프에서 가져왔다.** 5.1 센서의 `ray_directions`와 6.0 런타임이 마스크를 조회할 때 쓰는 방향 집합(`_mid360.ray_directions.torch`)은 **같은 인덱스끼리 비교하면 중앙값 129.8° 어긋나지만, 집합으로는 같다** — 6.0 방향 하나하나에 대해 가장 가까운 5.1 방향까지의 각도가 중앙값 0.425°(p99 1.72°)이고 반대 방향도 같다. 고도 분포도 5.1 [−77.8°, 34.4°] 평균 −21.57° 대 6.0 [−77.1°, 34.5°] 평균 −21.33°로 일치한다. 즉 인덱스 순서만 다른 같은 rosette이며(최적 Z축 회전 +179.76°를 먹여도 인덱스별 중앙값이 47.2°로 남아 강체 회전도 아니다), **v1이 5.1 방향을 쓴 것 자체는 커버리지 결함이 아니다. v1의 실질적 결함은 원점 하나뿐이다.** 그럼에도 6.0 방향으로 격자를 만들어야 하는 이유는 **격자가 희소하기 때문**이다 — 같은 물리적 로브를 5.1 방향으로 binning하면 842개가 아니라 653개 bin만 차고, 이 격자를 6.0 방향으로 조회하면 1,404개 중 817개만 걸러져 **마스크의 42 %가 조용히 사라진다.** 5.1 cfg의 `offset`은 pos·rot 모두 6.0과 동일하다(`rot=(0.0, 0.96593, 0.0, 0.25882)`). 참고로 연속 3스텝의 5.1 `ray_directions`가 완전히 동일해, v1 docstring이 경고한 `_update_dynamic_rays` 누적 Z회전은 이 런에서 나타나지 않았다. (c) 히트 임계값 0.55 m는 base 원점 기준으로 정해진 값이라 재검증했다 — 0.30/0.40/0.55/0.80/1.20 m 스윕에서 가림 광선 수가 1,367~1,381로 평탄해 0.55를 유지했다.
>
> 결과: 가림 광선 **1,381개(5.75%)**, 방위 127°~235°, 고도 −60.3°~−12.7°로 **전부 후방·하방**이고 전방 섹터는 0개다. MuJoCo Go2 모델로 마운트에서 독립 재계산한 마스크와의 **광선 단위 IoU는 v2가 0.589, 구 마스크가 0.032**다(`mask_v2_vs_mujoco.py`). 0.59가 1.0이 아닌 것은 USD 시각 메시와 MuJoCo 충돌 지오메트리가 다르기 때문이며, 두 방법이 같은 로브를 짚는다는 확인으로는 충분하다. 구 마스크와 v2는 사실상 배타적이다 — 24,000 광선 중 **양쪽이 함께 버리는 것은 121개**뿐이다.

![구 마스크와 v2 마스크의 az/el 격자 비교](assets/figures/mask_v2_overview.png)
*좌: 구 `body_occ_azel_grid.npy`(2,607 bin, 전방 로브). 중: 신 `body_occ_azel_grid_v2.npy`(842 bin, 후방·하방). 우: 겹침 — 빨강이 구 전용, 파랑이 신 전용, 겹치는 영역은 거의 없다.*

**구/신 마스크를 같은 롤아웃에 먹인 비교.** 정책은 구 마스크로 학습됐으므로 롤아웃 자체(행동·궤적)는 구 마스크 관측으로 굴렸다. 바꾼 것은 **같은 raw 스캔에 두 마스크를 각각 먹여 격자를 두 번 만든 것**뿐이라(`dump_mask_compare.py`, `env._compute_lidar_occupancy_grid`를 hit_valid만 바꿔 두 번 호출), 모든 차이가 마스크에만 귀속된다. 런타임의 하드코딩 마스크 파일은 건드리지 않았다. 지형 3종 × 200 프레임 평균. 추적 env는 고립시킨 지형에서 **레벨이 가장 높은 env**를 고르므로, 세 런 모두 `--max_init_level 6`으로 돌렸음에도 warmup 40 스텝 동안의 커리큘럼 승급으로 hurdle·crawl은 L7에서 찍혔다 — 세 행의 난이도는 같지 않다. 또 마스크 생성 로그의 광선 수 1,381과 아래 그림의 1,404가 다른 것은 az/el 1° bin 양자화 때문이다 — 격자로 저장했다가 다시 조회하면 같은 bin을 공유하는 광선 23개가 더 딸려난다:

| 지형 | 유효 히트 구/신 | 점등 셀 구/신 | v2가 지우는 셀 | v2가 더하는 셀 | 변경 셀 / 점등 셀(구) | teacher recall 구/신 | teacher precision 구/신 |
|---|---|---|---|---|---|---|---|
| stair L6 | 16,025 / 16,432 | 550.5 / 526.5 | −49.9 | +25.9 | 13.5 % | 0.756 / 0.757 | 0.128 / 0.135 |
| hurdle L7 | 16,155 / 16,598 | 548.2 / 524.5 | −41.8 | +18.1 | 10.7 % | 0.750 / 0.750 | 0.120 / 0.127 |
| crawl L7 | 19,297 / 20,288 | 676.3 / 682.8 | −39.2 | +45.8 | 12.6 % | 0.659 / 0.664 | 0.128 / 0.128 |

![stair L6 구/신 마스크 비교](assets/figures/mask_v2_compare_stair.png)
*stair L6. (a1)(a2) 3 m 이내 raw 히트를 마스크별로 색칠했다 — 회색은 양쪽이 살리는 히트, 빨강은 구 마스크만 버리는 전방 광선(v2가 되살림), 파랑은 v2만 버리는 후방·하방 광선(구 마스크가 통과시키던 유령 지면). (a3) 두 마스크가 버리는 광선을 방위·고도 평면에 겹쳐 그렸다. (b1)(b2) 격자의 위·옆 투영, 빨강이 v2가 지우는 셀, 파랑이 더하는 셀.*

![hurdle L7 구/신 마스크 비교](assets/figures/mask_v2_compare_hurdle.png)
*hurdle L7. 같은 구성. 이 지형에서 v2가 더하는 셀은 전부 마운트 높이 아래(장애물 앞면·바닥)에 있다.*

![crawl L7 구/신 마스크 비교](assets/figures/mask_v2_compare_crawl.png)
*crawl L7. 유효 히트가 세 지형 중 가장 크게 늘고(+991), 유일하게 점등 셀 총수도 늘어난다.*

**바뀌는 셀이 어디에 있나 — 이것이 재학습 판단의 핵심이다.** v2가 **지우는** 셀은 세 지형 모두 **100 %가 몸통 근처(격자 x ≤ 0.3 m)** 다. 구 마스크가 후방·하방 자기 가림을 거르지 않아, 몸통을 통과해 뒤쪽 지면을 찍은 광선이 로봇 자기 몸 위치에 유령 셀을 쓰고 있었다는 뜻이다. v2가 **더하는** 셀은 **99.9 %가 전방(x > 0.3 m)** 이고, 그중 마운트 높이보다 위(z > 0)인 비율이 **stair 61.7 %, crawl 12.2 %, hurdle 0 %** 다. 즉 stair에서는 되살아난 전방 광선이 계단 윗면(장애물 상단)을 실제로 채우고, hurdle에서는 허들 앞면·바닥만 채운다. teacher 격자 대비 recall은 세 지형 모두 사실상 불변(±0.005)이고, precision은 stair +0.007, hurdle +0.007, crawl 불변으로 소폭 개선된다.

**재학습에 대한 판단과 한계.** 정책이 읽는 격자의 **10.7~13.5 %**가 바뀌므로 관측 분포 변화는 무시할 수준이 아니고, 기존 체크포인트에 v2 마스크를 그대로 끼우는 것은 off-distribution이다. 근거상 재학습할 이유는 있다 — 몸통 위치의 유령 셀이 사라지고, 전방 상부(계단·장애물 윗면) 셀이 새로 들어온다. 다만 **재학습 없이는 성능이 좋아진다고 말할 수 없다.** teacher 대비 recall이 거의 그대로라는 것은 v2가 teacher가 아는 기하를 더 많이 잡아주지는 않는다는 뜻이고, precision 개선폭(+0.007)도 완주율로 옮겨질지는 이 측정으로 알 수 없다. 또 마스크가 학습·배포에 동일하게 적용되므로 구 마스크가 sim2real 불일치를 만들지는 않았다. 결론적으로 v2는 **다음 학습부터 적용**할 대상이고, 판단 근거는 "관측이 물리적으로 옳아진다"이지 "지금 정책이 더 잘 걷는다"가 아니다.

> ⚠️ **v2 격자는 이 24,000 광선 패턴 전용이다.** 64,800 bin 중 842개만 채워져 있고(구 마스크도 2,607개로 마찬가지로 희소하다), 이는 24,000 광선이 구면을 드문드문 샘플링하기 때문이다. 런타임은 그때그때의 `ray_directions`로 bin을 조회하므로, **Mid-360 광선 패턴이나 `num_rays`가 바뀌면 조회가 빈 bin에 떨어져 마스크가 조용히 꺼진다.** 패턴을 바꾸면 마스크를 반드시 같이 재생성해야 한다.

*재현: `assets/render_slgrid_pipeline/`의 `compute_body_mask_v2.py`(5.1 환경, 마스크 생성) → `mask_v2_vs_mujoco.py`(교차검증) → `dump_mask_compare.py`(6.0 환경, 지형별 롤아웃 덤프) → `render_mask_compare.py`(그림·GIF). 수치는 `assets/data/mask_v2_compare_stats.json`, 마스크 진단은 `body_occ_mask_v2_summary.json`. 롤아웃 GIF는 `assets/videos/mask_v2_{stair,crawl}_oldvsnew.gif`.*

**Isaac Sim 렌더 — 같은 프레임, 구 마스크 vs v2.** 위 산점도가 보여주지 못한 것을 시뮬레이터 안에서 확인한 그림이다. 한 컨트롤 스텝 안에서 물리를 진행시키지 않고 마커만 바꿔 뷰포트를 세 번 캡처했으므로 로봇 자세도 카메라도 동일하고, 두 패널의 차이는 전부 마스크 차이다. **정책은 여전히 구 마스크로 학습된 그 체크포인트**이고(`model_49999.pt`), env가 로드하는 마스크도 구 마스크 그대로다 — 다시 계산한 것은 오버레이뿐이다.

![stair L7 접근 구간(step 20). 좌: 구 마스크(v1)가 살리는 히트 16,872개, 우: v2가 살리는 히트 17,857개. 둘 다 마젠타로 같은 색을 쓴 것은 "그 마스크에서 정책이 실제로 받는 점 전체"를 보여주기 위해서다. 계단 앞면·바닥의 점 밀도가 오른쪽에서 뚜렷하게 올라간다.](assets/figures/mask_v2_isaac_stair_0020.png)

![같은 프레임을 한 패널에 3색으로. 흰색은 양쪽 마스크가 모두 살리는 히트, 빨강 2,122개는 구 마스크만 버리던 광선(v2가 되살리는 전방 띠 — 로봇 앞 계단 앞면과 바닥에 몰려 있다), 파랑 1,137개는 v2만 버리는 광선이다. 파랑은 로봇 몸 위의 점이 아니라 **몸통을 통과해 뒤쪽·아래쪽 지면을 찍은 유령 리턴**이고, 시뮬레이터의 레이캐스터가 로봇 자체를 충돌체로 보지 않기 때문에 생긴다.](assets/figures/mask_v2_isaac_stair_0020_3color.png)

![crawl L7 접근 구간(step 12). 되살아나는 빨강 광선이 크롤 터널 앞면과 바닥 위에 띠로 놓인다 — 정확히 정책이 통과 높이를 판단해야 하는 곳이다. 터널 안으로 들어간 뒤의 프레임은 추종 카메라가 벽 바깥에 있어 그림으로 쓸 수 없다.](assets/figures/mask_v2_isaac_crawl_0012.png)

![hurdle L7 장애물 위(step 44). v2가 되살리는 1,863개 광선이 장애물 앞면과 그 너머 바닥에 놓인다. 오른쪽 패널에서 로봇 앞쪽 점 밀도가 눈에 띄게 올라가는 것이 그 차이다.](assets/figures/mask_v2_isaac_hurdle_0044.png)

stair와 hurdle은 접근·장애물 위·통과 후 세 프레임을, crawl은 접근 프레임 하나만 뽑았다(터널 안에서는 추종 카메라가 벽에 가린다) — `assets/figures/mask_v2_isaac_{stair_0020,stair_0100,stair_0158,hurdle_0016,hurdle_0044,hurdle_0096,crawl_0012}.png`와 각각의 `_3color.png`. 위에 실은 것은 그중 차이가 가장 잘 보이는 프레임이다.

stair 롤아웃 전체를 좌우로 붙인 영상: `assets/videos/mask_v2_isaac_stair.mp4` (170 프레임, 20 fps). 계단을 올라가는 동안 되살아나는 전방 히트 수가 접근 구간 ~2,100개에서 등반 중 ~620개로 줄었다가 정상 평지에서 ~1,570개로 돌아오는 것이 보인다 — 되살아나는 광선이 **로봇 앞의 지면**을 찍던 광선이라는 것과 일관된다. 반면 v2가 버리는 후방·하방 유령 리턴은 이 stair 롤아웃 170 프레임 내내 1,118~1,180개로 자세와 거의 무관하게 일정하다.

*재현: `assets/render_mask_v2_isaac/`의 `play_mask_compare.py`(6.0 환경, 원본 PNG + 프레임별 카운트) → `compose_mask_compare.py`(라벨·범례·영상, sim 불필요). 렌더링에서 조용히 그림을 망가뜨리는 함정 두 가지(패널별 독립 서브샘플링, PointInstancer 프로토타입 배열의 부분 갱신)는 같은 폴더 README에 정리해 뒀다.*

**② 좌표계 정렬 — 이 방법의 핵심.** Mid-360은 `ray_alignment="base"`라 광선이 롤·피치로 기울지만, teacher의 복셀 격자는 **yaw 정렬**(롤·피치 제거)이다. 둘을 같은 프레임에 올리는 회전:

$$q_{yb} \;=\; \mathrm{conj}\big(\mathrm{yaw}(q_{wb})\big)\cdot q_{wb}$$

여기서 $q_{wb}$는 base→world 쿼터니언, $q_{yb}$는 base→yaw-aligned. 이 정렬만으로 **공짜 이득**이 생긴다 — 계단에서 "올라선 단의 윗면이 2.6~2.7° 더 코를 든 자세에서야 보이던" 피치 의존 결손이 사라진다.

> ⚠️ **이 회전은 조용히 틀린다.** 학습은 돌아가고 loss 곡선도 그럴듯하다. 불변식 두 개로만 잡힌다 — **제자리 회전 시 격자 좌표에서 지형이 움직이지 않아야 하고**, 비영 피치에서 회전된 산란이 teacher 격자와 더 잘 일치해야 한다.

![② base 정렬 vs yaw 정렬 — teacher 격자 대비](assets/figures/slgrid_pipeline_alignment.png)
*계단 L6, 260 프레임. **좌·중**은 같은 프레임(피치 −26.7°)의 같은 히트를 base 프레임과 yaw 정렬 프레임에 각각 산란한 것이고 주황 사각이 teacher 복셀 격자다. 회전을 빼면 점군이 통째로 기울어 계단면이 격자 아래로 흘러내린다 — recall **0.750 → 0.292**, IoU **0.122 → 0.041**. 260 프레임 평균은 recall **0.770 vs 0.615**, IoU **0.126 vs 0.094**. **우측 막대가 "공짜 이득"의 정체다**: |피치| < 3°에서는 두 정렬이 사실상 같고(0.795 vs 0.767) |피치| > 10°에서만 갈라진다(0.731 vs 0.453). |피치|와 recall 이득의 상관계수는 **0.75**. recall 정의는 학습 중 `[LidarGrid]` 진단이 찍는 것과 같다(teacher 점유 셀 대비). teacher 격자가 프레임당 평균 85.7 셀만 점유하므로 precision·IoU는 원래 낮게 나오며, 여기서 읽을 것은 절대값이 아니라 두 정렬의 격차다.*


**③ 격자 산란.**

| 항목 | 값 |
|---|---|
| 범위 | x [−0.6, 2.0], y [−1.0, 1.0], z [−0.6, 0.6] m |
| 해상도 | 0.1 m |
| 형태 | **27 × 21 × 13 = 7,371** (C-order flat, binary) |
| 원점 | clearance 마운트 = base + 0.05 m |

원점 오프셋이 teacher의 clearance 스캐너 마운트와 **정확히 같아야 한다.** 어긋나면 두 격자가 평행이동돼 비교가 무의미해진다. 메모리는 `(chunk, R≈24k, 3)` 중간 텐서 때문에 256 env 단위로 쪼개 처리한다 (1024 env 일괄이면 약 295 MB).

![③ 격자 산란 — 히트에서 7,371 이진 셀까지](assets/figures/slgrid_pipeline_scatter.png)
*같은 프레임(t=100, 피치 −26.7°). **좌상**: yaw 정렬 프레임의 원시 히트를 위에서 본 것(색 = z), 점선이 crop box다. 이 프레임 24,000 광선 중 16,965개가 유효하고 대부분은 상자 밖으로 떨어진다. **중상**: 격자를 위에서 본 것, 색은 각 열에서 점유된 가장 높은 z — 계단의 단 높이가 그대로 줄무늬로 보인다(567 열 중 332 열 점등). **우상**: 측면(|y| < 0.25 m) — 히트와 격자 셀이 같은 계단 프로파일 위에 겹친다. **하단**: z 슬라이스 네 장. 이 프레임 568/7,371 셀(7.7%)이 켜지고, 260 프레임 평균은 511 셀(6.9%)이다.*


**④ (옵션) 시간 누적.** 매 센서 틱마다 누적기를 새 yaw 프레임으로 워프한 뒤 새 스캔을 섞는다. 디테일 두 가지: 평행이동은 **이전** yaw로 회전해야 하고(새 원점을 옛 프레임에서 표현하므로), 혼합은 단순 EMA가 아니라 **decayed max**여야 한다(lerp면 지금 보이는 점유 셀을 실제보다 낮게 보고한다). 유효 감쇠 대역은 $\alpha \in [0.93, 0.955]$ (반감기 1.0~1.5 s). *현행 crawl SYM arm은 누적 OFF.*

![④ decayed max vs EMA](assets/figures/slgrid_pipeline_accumulation.png)
*⚠️ **현행 arm은 누적 OFF이므로 이 그림은 덤프한 틱 시퀀스로 오프라인에서 굴린 것**이다(워프는 env의 `_warp_occupancy_acc`를 그대로 호출). α = 0.94, 센서 틱 5 제어 스텝(0.1 s), 총 52 틱. **좌상**: 전방 셀 하나(x, y, z) = (+0.30, −0.60, −0.40) m의 값. 회색 계단이 그 틱의 원시 관측(0/1)이다. decayed max는 처음 보이는 순간 1.00에 닿고, 같은 α의 lerp는 이 구간 내내 **최고 0.41**에 그쳐 0.5 임계를 한 번도 넘지 못한다. **우상**: 임계 초과 셀 수 — EMA는 52 틱 내내 **0셀**, decayed max는 단일 프레임보다 꾸준히 위에 있다. **하단**: 마지막 틱의 측면 투영. 단일 프레임 384셀 / decayed max 550셀 / EMA 0셀(> 0.5). 이것이 본문의 "lerp면 지금 보이는 점유 셀을 실제보다 낮게 보고한다"의 정량적 형태다.*


#### 라우팅 — 사실상 한 줄

```python
self.obs_groups = {**self.obs_groups, "lidar": ["lidar"], "voxel": ["lidar"]}
```

`obs_groups`의 `"voxel"` 슬롯이 `["lidar"]`를 가리켜, **actor의 지형 인코더가 특권 격자 대신 자기 LiDAR 격자를 읽는다.** critic은 손대지 않고 특권 clearance(294)를 계속 본다. 이 비대칭이 **from-scratch 학습을 가능하게 만드는 장치**다.

![raw Mid-360 히트 vs 정책이 실제로 읽는 격자 — 계단 L6](assets/figures/sensor_raw_vs_grid_stair.png)
*계단 지형 L6, 같은 롤아웃을 두 오버레이로 렌더. **좌**: Mid-360이 돌려주는 raw 히트 약 24k 광선(자홍). **우**: 그것을 산란해 만든 27×21×13 = 7,371 점유 격자(청록) — 정책이 실제로 읽는 것. 점군은 멀리까지 뻗지만 격자는 x [−0.6, 2.0] · y [−1.0, 1.0] · z [−0.6, 0.6] m로 잘리고 0.1 m로 양자화된다. 이 두 장의 차이가 산란 파이프라인 ②→③ 단계 그 자체다. 관찰 두 가지: (a) 점군은 거리에 따라 급격히 성기다 — 각도 range image의 행 간격 붕괴가 이 기하에서 나온다. (b) 격자는 밟고 넘을 영역만 담는다 — 멀리 보이는 계단은 정책 입력에서 사라지며, 그래서 시간 누적(④)이 의미를 갖는다.*

동영상판: [`assets/videos/06_sensor_raw_vs_grid_stair.mp4`](assets/videos/06_sensor_raw_vs_grid_stair.mp4) · step 지형 정지컷: [`assets/figures/sensor_raw_vs_grid_step.png`](assets/figures/sensor_raw_vs_grid_step.png)

#### 결과 — 표현이 binding defect였음이 확정된다

distillation 기준 pinned 완주율이 **0.078 → 0.238**로 이동한다(각도 range image → 미터 격자, 10k 시점). 그리고 SL-Grid를 teacher 없이 from-scratch로 돌린 것이 **모든 distillation arm을 큰 폭으로 앞선다.**

> ⚠️ 이 두 절대값도 §11.4의 probe 분모 버그 영향을 받아 과소 표기다. **배율(약 3배)과 방향만 읽어야 한다.** 또한 바로 아래 "학습량을 맞추자 두 결론이 뒤집혔다"에서 보듯 이 배율 자체가 10k 스냅샷의 값이고, 50k로 맞추면 1.23배로 줄어든다.

| 파일 | 비교 내용 | 결론 |
|---|---|---|
| [`assets/videos/07_distill_matched50k_4up.mp4`](assets/videos/07_distill_matched50k_4up.mp4) | 같은 gap 지형, 4분할 — K=10 각도 / A1-0 격자 / SL-Grid(teacher 없음) / privileged teacher | 각 arm이 실제로 읽는 것을 오버레이. **보라 = 원시 Mid-360, 청록 = student 격자, 주황 = teacher 격자.** 학습량을 50k로 맞춘 뒤에도 SL-Grid가 최선의 distillation arm을 큰 폭으로 앞선다 |

> ⚠️ **주의**: 이 4분할 영상이 나온 `matched-50k` 비교의 절대 완주율 수치는 후술하는 probe 분모 버그(§11.4)의 영향을 받는다. **arm 간 순서와 격차의 방향은 유효하지만 절대값은 과소 표기다.**

#### ★ 학습량을 맞추자 두 결론이 뒤집혔다

이 프로그램에서 가장 값진 교훈이다. distillation arm들은 10k에서, teacher와 SL-Grid는 50k에서 측정돼 있었다. 학습량이 5배 다른 표를 한 줄로 읽고 있었던 것이다. 전 arm을 50k로 맞추자:

1. **"창 길이는 지렛대가 아니다"가 뒤집혔다.** 10k에서는 K=10 < K=3이라 "순서 없음 = 노이즈"로 판정했는데, 50k에서는 K=10이 K=3보다 확실히 높다. **더 긴 창은 쓸모가 있는데, 그것을 쓰는 법을 배우는 데 10k로는 부족했던 것이다.**
2. **"표현 교체가 2.9배"의 대부분이 수렴 속도였다.** 격자 arm은 10k에서 이미 거의 포화했고(4배 학습에 +2.6 pt) 각도 arm들은 계속 올랐다. 최종 배율은 2.9배 → **1.23배**로 줄었다. 우위가 사라진 것은 아니고(tilt는 여전히 확실히 낮다) 크기가 달라졌다.

> **같은 실수를 이 프로그램에서 세 번 했다**: A1-1 5k(2.1배) → 10k(1.17배), teacher 기준점 0.260 → 0.357, 그리고 이번. 매번 원인은 **이른 시점의 스냅샷을 정상 상태로 읽은 것**이다.

**유지된 것**: 시간 누적(A1-1)은 50k에서도 A1-0을 넘지 못했다. **blind band를 9.8배 메워도 소용없다는 결론은 학습량을 맞춰도 그대로다.**

### 8.6 crawl을 학습 분포에 넣다

기존 SL-Grid는 5지형 균등(각 0.20)이라 **crawl을 한 번도 본 적이 없었다.** 그 상태로 crawl에서 재면 542 pinned 에피소드 중 0완주·43% 전복인데, 이는 능력이 아니라 **zero-shot transfer**를 잰 것이다. 다른 네 지형에서는 두 정책 모두 tilt ≤ 3.7%에 실패가 전부 시간초과인데, crawl에서만 tilt 43%로 넘어진다 — 실패 양상 자체가 다르다.

그래서 crawl arm은 flat 0.20 + 나머지 5개 0.16의 **별개 baseline**으로 새로 시작했다.

같은 시기에 **crawl 커리큘럼 정체의 원인도 규명됐다.** GT teacher의 crawl 레벨이 20k 이후 1.5 부근에 주저앉아 50k까지 그대로였는데, 원인은 능력이 아니라 **코스 길이 결함**이었다 — crawl-3 코스(10.95 m)가 다른 지형의 61%라 완주해도 승급선에 못 미치고, 명령이 빠르면 완주가 곧 강등이 됐다. `num_crawls`를 3 → 8(17.79 m)로 맞추자 그 함정이 사라지고 다른 지형과 같은 대역(6.01)에 진입한다.

![지형별 커리큘럼 레벨 — GT voxel teacher vs SL-Grid + crawl](assets/figures/curve_terrain_curriculum.png)
*45,000~50,000 iteration 평균. **다섯 지형(stair/gap/hurdle/step/flat) 모두 종단에서 사실상 동률**이다 — 최대 차 0.48, 대부분 0.3 이내. 특권 관측의 이점은 종단 높이가 아니라 **속도**로 나타난다: GT가 초반 5~15k에 먼저 올라가고 SL이 뒤따라 붙는다. **★ crawl 패널은 능력 비교가 아니라 커리큘럼 수정의 증거다** — GT의 1.54는 crawl-3 코스 길이 결함의 결과이고, 두 곡선의 차이는 정책 성능 차이가 아니라 지형 정의 차이다. 주의: 커리큘럼 레벨은 승급 문턱을 넘은 결과이지 완주율이 아니며, 문턱은 코스 길이와 명령 속도에 좌우된다.*

![물리적 결과 지표 — GT voxel teacher vs SL-Grid + crawl](assets/figures/curve_physical_outcomes.png)
*종료 원인과 페널티는 두 run에서 정의가 같아 직접 비교 가능하다 (45k~50k 평균, 각 n=5,000). **안정성은 GT, 지속성·충돌은 SL이 낫다** — 낙상(0.0132 vs 0.0274)·전복(0.179 vs 0.231)·stumble은 GT 쪽이고, 에피소드 길이(720.6 vs 790.4)와 collision 페널티(−1.646 vs −0.962)는 SL 쪽이다. **SL의 낙상은 20~25k 구간에서 0.134로 정점을 찍어 GT 최악 구간의 약 5배였다가 40k 이후 급감했다** — 중간 스냅샷만 봤다면 "LiDAR로는 낙상을 못 줄인다", 후반만 봤다면 "따라잡았다"고 결론냈을 구간이다. 둘 다 과하다. 주의: 지형 비율이 달라(GT는 crawl 0.25) GT 우위의 일부는 지형 구성 차이일 가능성을 배제하지 못한다.*

---

### 8.7 실기 Mid-360 데이터와의 대조 — sim LiDAR는 얼마나 다른가 (2026-09-03)
실기 Go2에 달린 Mid-360의 rosbag(142 s, `/livox/lidar` PointCloud2 10 Hz 1,426프레임, `/livox/imu` 200 Hz)을 받아 sim 센서 모델과 나란히 쟀다. 제공자 메모: "뒤쪽 120°는 crop해서 안 씀, go2 몸통은 blind 거리(예: 0.8 m 이내 무시)로 제거 가능". 원자료 `source/.../parkour_imitation/real_lidar/`, 분석 `reports/go2_parkour/_comparisons/real_lidar_vs_sim/` (`analyze_rosbag.py`, `analyze_frames.py`, `scatter_rosbag.py`).

| 항목 | 실기 (rosbag) | sim 현행 (`mid360_lidar` cfg) |
|---|---|---|
| 프레임 rate | 10 Hz | 10 Hz — 일치 |
| 프레임당 슬롯 / **유효 반환** | 20,000 / **약 12,000** (무반환 (0,0,0) 38 %) | 24,000 / 24,000 (지면 메시가 어디에나 있어 miss 없음, dropout 10 %) |
| 0.8 m 이내 점 | 유효 점의 **28 %** (몸통 + 센서 바로 아래 지면) | min_range 0.2 m |
| 1.5 m 이상 환경 점 | 약 5,500~6,700 / 프레임 | 약 20,000 이상 |
| 거리 분포 | median 1.16 m, p99 8.1 m, max 17.6 m | max 40 m |
| 스캔 방식 | **비반복**: 한 프레임은 FOV의 띠 일부, 4프레임(0.4 s)이면 거의 채움 | **고정** 24,000 방향을 매 프레임 동일 캐스팅 (`mid360.npy` 800k 중 고정 창) |
| 마운트 | 돔 아래 방향, 기울기 IMU 22° / 바닥 평면 30°, 센서 높이 0.39 m. 돔 축은 sim과 같은 "아래+앞" | 돔 아래 + pitch-up 30°, 높이 base+0.05 |
| 후방 크롭 | 이미 반영됨 (az 150~210° 원거리 점 0개) | 없음 |

"sparse"의 실체는 세 가지가 겹친 것이다 — 유효 반환이 프레임당 절반, 그중 28 %가 0.8 m 이내라 환경 점은 약 9,000개(sim의 1/2.7), 그리고 한 프레임이 FOV 일부 띠만 덮는 비반복 스캔. sim은 세 가지 모두 반대다.
![실기 vs sim 개요. 좌상: 실기 점의 방위/고도 밀도(base 프레임), 우상: sim 패턴의 방위/고도 밀도, 좌하: 실기 거리 히스토그램(28 %가 0.8 m 이내, 스파이크 0.12 m·0.5~0.8 m = 몸통), 우하: 0.8 m 이내 점의 방위/고도.](assets/figures/real_vs_sim_overview.png)
![단일 프레임 비교(센서 프레임). 좌: 실기 100 ms 한 프레임(색 = 거리), 중: 실기 연속 4프레임 겹침 — 비반복 스캔이 0.4 s면 FOV를 채운다, 우: sim이 매 프레임 동일하게 캐스팅하는 고정 24,000 방향.](assets/figures/real_vs_sim_single_frame.png)
#### 오프라인 산란 검증 — rosbag 프레임을 sim 격자 규칙에 넣어 봄
Isaac Sim 없이, 실기 프레임을 §8.5 ③과 같은 규칙(lo (−0.6, −1, −0.6), 0.1 m, 27×21×13, `round((p−lo)/res)`)으로 격자에 넣었다. 좌표는 발행 프레임 → **프레임별 바닥 평면 적합으로 roll/pitch 제거**(= sim의 tilt removal, 1,292/1,426 프레임 성공, 센서 높이 중앙값 0.39 m) → 후방 크롭 빈 구간이 az 150~210°에 오도록 yaw → 격자 원점 기준 평행이동(센서 = 원점 + (0.334, 0, 0)). 전방·마운트 높이는 데이터에서 추정한 값이라 제공자 확인 전이다.

| 변형 | 점등 셀/프레임 (p5~p95) | 전방(x>0.3) 셀 | 시간 누적 α=0.94 후 |
|---|---|---|---|
| raw | 555 (404~849) | 411 | — |
| 후방 120° 크롭 | 474 (330~739) | 407 | — |
| 크롭 + blind 0.8 m | 389 (238~621) | 346 | 928 |
| 크롭 + v2 방향 마스크 + min 0.3 m | 473 (326~739) | 406 | 1,075 |
| (참고) sim 단일 프레임, v1 마스크 | stair 550 / hurdle 548 / crawl 676 |  | teacher 격자 stair 86 |

![실기 3프레임을 sim 격자 규칙으로 산란. 열: 실기 점(base 프레임, 색 = z, 점선 = 격자 상자) / 후방 크롭 / 크롭+blind 0.8 / 크롭+v2 마스크+min 0.3. 색 = 최고 점유 z층. 정면 약 0.7~1 m의 빈 구역은 Mid-360의 물리적 blind cone이며 sim에도 같은 형태로 있다.](assets/figures/scatter_rosbag_frames.png)
![좌: 프레임별 점등 셀 수(점선 = sim stair 550, 점점선 = teacher 86). 우: 시간 누적(파이프라인 ④, α 0.94) 후 셀 수 — 비반복 스캔에서는 누적이 셀을 2~2.5배로 늘린다.](assets/figures/scatter_rosbag_timeseries.png)
- **실기 프레임은 sim 격자에 sim과 같은 자릿수로 들어간다.** 후방 크롭 후 474셀/프레임(sim stair 550). 프레임 간 변동(330~739)이 큰데 이는 비반복 스캔의 띠가 프레임마다 다른 곳을 덮기 때문 — 고정 패턴인 sim에는 없는 성질이다.
- **blind 0.8 m는 셀의 18 %를 지운다**(474 → 389). 지워지는 것은 전방 0.35~0.8 m의 바닥·발밑 셀(발 디딜 자리, crawl 천장 높이)이다. v2 방향 마스크 + min 0.3 m는 셀을 거의 보존한다(473).
- **정면 blind hole은 실기에도 있고 sim과 형태가 같다.** 바닥 평면 inlier의 방위별 최저 고도로 잰 "바닥이 보이기 시작하는 거리": 실기 전방 0.64~0.77 m / 측면 0.21~0.29 m / 후방 0.17~0.19 m, sim 0.82 / 0.12~0.20 / 0.08 m. sim의 "돔 아래 + 앞기울기" 마운트 가정은 실기와 맞고, 실기의 기울기가 30°보다 조금 작아(IMU 22°) 전 방위에서 blind 반경이 0.05~0.1 m 넓다. **마운트 pitch를 22~25°로 줄이는 보정**이면 프로파일이 맞을 것으로 보이며 정확한 값은 제공자 extrinsic으로 확정한다.
- **시간 누적을 켜면 셀이 2~2.5배로 는다.** 4프레임이면 FOV가 거의 채워지므로, 비반복 스캔을 재현하는 sim에서는 누적 ON(현행 crawl SYM arm은 OFF)이 실기 특성과 맞는다.
#### 수정 계획 (arm 분리, 한 번에 하나씩)
0. **좌표계 확정(전제)** — 제공자에게 드라이버 config의 extrinsic, 마운트 사진/CAD, 실제 배포 파이프라인의 크롭·blind 설정값을 받아 우리 측정(IMU up, 바닥 평면, 후방 빈 구간)과 대조.
1. **sim 센서 모델** — 마운트 pitch 30° → 22~25°, 20,000 슬롯 + 거리 의존 dropout(1차 근사 0.38), 비반복 스캔(`rolling_window_start`를 프레임마다 전진, `lidar_sensor.py` 소폭 변경), 후방 120° 크롭(base az 120~240° 무효).
2. **blind 정책 A/B** — (B) blind 0.8 m(실기 파이프라인이 그렇게 쓰면 sim도 같아야 함) vs (C) v2 방향 마스크 + min_range 0.3 m. 어느 쪽이든 마스크는 새 패턴·마운트로 재생성(희소 격자, §8.5 ①).
3. **격자 산란** — 시간 누적 ON을 기본 후보로, teacher 격자 대비 recall 재측정.
4. **학습 전 게이트** — sim 프레임 통계(유효 점 9k±2k, 거리 히스토그램, az/el 밀도)를 실기와 같은 정의로 표 한 장에.
5. **재학습** — arm (A) 마운트+비반복+dropout+크롭, (B) A+blind 0.8, (C) A+v2 마스크+min 0.3. 10k 스냅샷 pinned probe(5지형 × L3/L6, 64 env × 2,000 step)로 조기 비교 후 유망 arm만 50k. 기준선 2026-08-10 run. 도구: `reports/go2_parkour/_comparisons/maskv1_vs_maskv2_pinned_probe/{run_probe_grid.sh, compare_probes.py}`, cfg 필드 `body_occ_grid_path`.
> 경위: v2 마스크만 바꾼 재학습을 2026-09-03 13:20에 시작했으나(1024 env, seed 1, GPU 2), 실기 데이터 검토 결과 센서 모델 자체를 먼저 고쳐야 한다고 판단해 1,200 iter에서 중단하고 run을 삭제했다. 원인 분리 원칙(한 번에 한 변경)에 따라 위 arm 구성으로 다시 시작한다.

---

#### 조치 — RealSensor arm 구현 · 학습 전 게이트 · 재학습 시작 (2026-09-03 16:51)

제공자 답변으로 0단계가 정리됐다. 드라이버 extrinsic **미사용**(rosbag 점은 센서 원좌표 그대로이므로 위에서 바닥면으로 잰 기울기 22~25°가 마운트 pitch 자체), 몸통 가림은 **0.8 m blind로 처리**(A안 확정, "실제로 가려지는 부분이 많다"), 마운트 사진·도면은 미도착이라 각도·위치는 cfg 필드로 열어 두고 값이 오면 교체한다.

**구현 (코어 `source/isaaclab/` 무수정, 전부 task 폴더).** 새 arm `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-RealSensor-EasyEntry-v0`(cfg `ParkourImitationRandomGoalLidarSLGridCrawlRealSensorEnvCfg`). 필드가 전부 독립 토글이라 ablation은 하나씩 되돌리면 된다.

| 항목 | 구현 | 값 |
|---|---|---|
| 비반복 스캔 | `mid360_rolling_lidar.py` — `LidarSensor` 서브클래스. 800k행 패턴 전체를 GPU에 두고 갱신마다 env별 20k행 창을 광선 방향 버퍼에 제자리 기록(warp 커널, env별 위상 무작위, 리셋 시 재추첨) | 창 전진 20,000행/프레임, 4 s 주기 |
| 마운트 pitch | `lidar_mount_pitch_deg`(명목) + `lidar_mount_pitch_range_deg`(env별 균등 추첨, 에피소드마다 재추첨) → `quat_mul(Rx(180°), Ry(p)) = (0, cos p/2, 0, sin p/2)` | 명목 30°, 범위 22~30° |
| blind | `lidar_blind_range` — 반환 자체를 무효화 | 0.8 m |
| 후방 크롭 | `lidar_rear_crop_deg` — base az \|az\| ≥ 120° 무효 | 120° |
| 자기 가림 마스크 | `lidar_use_body_occ_mask` — 실기가 안 쓰므로 끔 | False |
| 밀도·주기 | `samples`, `pixel_dropout_prob`, `update_period` | 20,000 / 0.38 / 0.09 s |
| 누적 | `lidar_grid_accumulate` (α 0.94) | True |

두 가지를 짚어 둔다. (1) 기존 arm은 `update_period=0`이라 센서를 제어 스텝마다 캐스트하고 있었다(측정치 1개당 5회). 정적 패턴이라 결과는 같았지만 롤링 스캔에서는 창이 5배 빨리 돌게 되므로 진짜 10 Hz로 맞췄다. 0.1이 아니라 0.09인 이유는 센서 타임스탬프가 float32로 0.005 s씩 누적돼 16 s 이후 판정이 6스텝마다 발화하며 누적기 push(5스텝)와 어긋나기 때문이다(fresh-context 리뷰에서 발견). (2) 센서 `.data.distances`는 접근할 때마다 노이즈·dropout을 다시 뽑는다. env는 한 번만 읽지만, 진단 코드가 두 번 읽으면 dropout 확률만큼의 가짜 불일치가 난다.

스모크 테스트(`real_lidar/smoke_real_sensor.py`, 16 env × 40 스텝): 창 전진 정확히 20k행, 0.8 m 미만·후방 웨지 유효 hit 0건, 부분 리셋 뒤 env별 부분 갱신 경로에서 `ray_starts + d·dir`와 `ray_hits` 최대 오차 1.9e-5 m, 512 env 갱신 8.7 ms(기존 24k 정적 10.6 ms). 기존 arm은 영향 없음(정적 방향, 게이트 0 %, 마스크 정상 로드).

**학습 전 게이트 — 실기와 같은 정의로 sim 프레임 통계** (`sim_frame_stats.py`, 32 env 정지 자세 40프레임; 실기 `real_frame_hist.py`):

| 조건 | 슬롯 | 반환/프레임 | 유효(크롭·blind 후) | <0.8 m 비율 | 후방 비율 | 격자 점등 | z>0 셀 |
|---|---|---|---|---|---|---|---|
| 실기 rosbag | 20,000 | 12,152 | 8,046 | 0.284 | 0.183 | 389 | 73 |
| sim 기존 센서 flat L0 | 24,000 | 19,161 | 17,325 | 0.509 | 0.268 | 580 | 0 |
| sim 기존 센서 stair L6 | 24,000 | 19,689 | 17,680 | 0.497 | 0.261 | 599 | 6 |
| sim RealSensor flat L0 | 20,000 | 10,998 | 4,923 | 0.466 | 0.308 | 296 | 0 |
| sim RealSensor stair L6 | 20,000 | 11,405 | 5,300 | 0.453 | 0.297 | 358 | 13 |

![게이트 비교 — 윗줄: 유효 반환의 base az/el 밀도(실기 / 기존 센서 / RealSensor). 실기 유효 반환은 el −30°~+28° 띠에 들어오고 RealSensor도 −25°~+30°인 반면 기존 센서(pitch 30°)는 +35°/−78°까지 벌어져 있다. 아랫줄: 거리 히스토그램(0.8 m 점선)과 프레임당 격자 점등 셀.](assets/figures/real_vs_sim_gate_compare.png)

읽는 법: 반환 수는 맞았다(12.2k vs 11.0~11.4k, 기존 센서는 1.6배). 시야 상하 경계가 맞았다 — 마운트 pitch 23°의 독립 확인. 유효 수의 남은 차이(8.0k vs 4.9k)는 센서가 아니라 장면이다. 실기 녹화 장소의 벽·천장이 el>0 광선을 1~8 m에서 되돌리지만(격자 안 z>0 셀 73) sim 지면 평면에서는 하늘로 빠진다. 지면 셀만 비교하면 실기 약 316 vs sim 296으로 10 % 안쪽. sim의 <0.8 m 비율이 높은 것은 몸통을 캐스트하지 않아 광선이 몸통을 뚫고 0.5 m 지면에 닿기 때문이고, 어느 쪽이든 blind가 지운다. dropout 0.38은 유지. **판정: 통과.**

**마운트 각도 정정과 무작위화 (2026-09-03 18:17).** 제공자가 준 라이다 extrinsic은 `rot(wxyz) = (0, 0.9659258, 0, 0.2588190)`, 즉 sim이 원래 쓰던 **30° 설계값 그대로**다. 그런데 rosbag의 두 독립 측정 — Mid-360 자체 IMU의 중력 방향(돔 축에서 21.7°)과 프레임별 바닥면 피팅(22~25°) — 은 녹화 당시 센서가 중력 기준 22~25°에 있었다고 말한다. 위 게이트에서 시야 상하 경계가 23°와 맞은 것도 같은 사실이다. 원인은 둘 중 하나다: 녹화 중 몸통이 7~8° 들려 있었거나, 붐이 설계 각도에 있지 않다. 사진·도면 없이는 가릴 수 없어 **명목 30°를 쓰되 env마다 22~30°에서 균등 추첨**(에피소드마다 재추첨)하는 것으로 두 경우를 모두 덮는다. 30° 고정 게이트도 통과(flat 302 / stair 349 셀, 반환 10.7~11.1k). 구현은 창 기록 warp 커널에 env별 쿼터니언 회전을 넣은 것으로, 축퇴 범위 (30, 30)에서 단일 회전 경로와 2.4e-7, 범위 없음(`None`)에서 비트 단위 동일을 확인했다.

**재학습.** 23° 고정 run(16:51, iter 1,149)과 30° 고정 run(17:54, iter 363)은 중단했고 디렉토리는 남겨 두었다. 현재 run: `logs/rsl_rl/parkour_imitation_go2_lidar_sl_grid_crawl_sym/2026-09-03_18-17-33_slgrid_crawl_sym_scratch_50k_realsensor_dr22-30` — **1280 env**(GPU 2 여유에 맞춘 최대: 프로세스 12.2 GB, 다른 사용자 점유 ~8 GB, 여유 4 GB), seed 1, 50k iter, 2.9 s/iter(10k ≈ 8 h). 10k 스냅샷에서 정본 pinned probe(5지형 × L3/L6, 64 env × 2,000 step)로 2026-08-10 기준선과 조기 비교한다. env 수가 기준선(1024)과 달라 배치 구성이 다르다는 점은 비교 때 적는다. 원자료·스크립트: `reports/go2_parkour/_comparisons/real_lidar_vs_sim/` §4.


## 9. 단계 6 — Symmetry를 LiDAR 격자에 재적용 (2026-08)

### 9.1 왜 다시 symmetry인가 — std 폭주라는 단서

SL-Grid + crawl 학습 곡선에서 **다른 어떤 arm에도 없던 현상**이 보였다.

![최적화 지표 — action noise std의 거동이 정반대](assets/figures/curve_optimization.png)
*★ **`action noise std`의 거동이 정반대다.** GT teacher는 0.60 → 0.529로 평평한데, SL-Grid + crawl은 0.60 → **1.837**로 50k 내내 단조 증가한다(+0.035/1,000 iter, 가속 없음). 같은 알고리즘·같은 하이퍼파라미터(`entropy_coef` 0.01, `desired_kl` 0.01)인데 이 차이가 났다. 다만 **붕괴는 아니다** — value loss 종단이 GT 0.0081 / SL 0.0090으로 사실상 같고 물리 지표는 같은 기간 계속 개선됐다 (과거 파국 사례는 std 0.93에서 value loss가 36까지 튀었다). 주의: `mean_reward`는 AMP discriminator가 run마다 다르고 지형 비율도 달라 **두 run을 같은 축에 놓을 수 없다** — 각 run 내부 추세로만 읽는다.*

이 미스터리의 정체는 나중에 밝혀진다. std 단조 증가는 정책 전체의 현상이 아니라 **`RL_calf` 관절 딱 하나의 폭주**였다 — 종단 16.7로 타 관절의 약 40배.

### 9.2 방법 — lidar 격자 y-flip mirror

symmetry mirror가 `obs["lidar"]` 키를 덮지 않으면 **지형 격자만 미러되지 않은 채** 나머지가 뒤집히므로 augmentation이 오히려 해가 된다. 그래서 2026-08-10에 lidar 격자용 y-flip 순열을 구현했다.

다행히 **SL-Grid 격자는 teacher voxel과 레이아웃이 동일**(27×21×13, C-order flat)하므로, §4.3의 voxel 순열이 **그대로 적용된다**:

$$i_y \mapsto n_y - 1 - i_y, \qquad k = i_x(n_yn_z) + i_yn_z + i_z$$

SYM arm만 `use_data_augmentation=True`로 켜고, env cfg는 OFF arm과 **완전히 동일**하다. 즉 **유일한 recipe 차이가 mirror augmentation 하나뿐인 통제된 A/B**다.

### 9.3 결과 — 능력은 동률, 병리는 소멸

50k 학습 후 두 정책을 **같은 지형, 같은 고정 레벨**에 배정해 커리큘럼 자기선택을 제거하고 쟀다 (5 장애물 지형 × 레벨 {3, 6} × 2 정책 = 20 조건, 조건당 193~274 에피소드).

**완주율 (2026-08-18 정정 반영 — 이 문서의 유일한 정정 완료 절대값이다)**

| | OFF | SYM | 격차 |
|---|---:|---:|---|
| 풀링 완주율 (전 조건 에피소드 합산) | **55.97%** (n=1508) | **56.82%** (n=1517) | **+0.85 %p, SE 1.80 %p, z = 0.47** |

지형별 (OFF / SYM): hurdle L3 42.34 / 42.22 · L6 46.32 / 47.83 · step L3 68.32 / 69.57 · L6 58.97 / 62.50 · gap L3 62.96 / 63.47 · L6 61.01 / 61.11 · stair L3 53.74 / 53.02 · L6 52.38 / 58.55 · crawl L3 56.67 / 51.37 · L6 52.94 / 53.74

**완주율은 전 조건 통계적 동률이다.** symmetry augmentation은 최종 능력을 올리지도 내리지도 않았다.

그러면 무엇이 좋아졌나. 세 가지다.

1. **수렴 속도 약 2배.** OFF가 30k에 도달한 커리큘럼 수준을 SYM은 ~15k에 도달한다 (같은 iteration 비교에서 초반 전 지형 레벨 약 2배, goal 도달 +15~19%).
2. **★ RL_calf std 폭주의 구조적 소멸.** OFF는 `RL_calf` 노이즈 std가 50k까지 단조 폭주해 **16.7**로 마감했고(타 관절의 약 40배), SYM은 **0.39**로 정상 하강한다. mirror가 L/R 쌍의 std를 강제로 일치시키므로 **한쪽 calf만 발산하는 평형이 성립 불가능해진다.** §9.1의 std 미스터리가 여기서 해소된다.
3. **같은 난이도에서 더 적은 충돌 + 미세하게 빠른 전진.** collision은 SYM이 **9/10 조건에서 낮다**(평균 −0.040 vs −0.083, step L6은 −0.192 vs −0.426으로 절반 이하). 전진속도는 8/10 조건에서 미세하게 빠르다(평균 1.011 vs 0.994).

비용(최종 능력 손실)은 관측되지 않았다.

### 9.4 ★ 커리큘럼 아티팩트가 반전된다 — 이 프로젝트의 방법론적 핵심

학습 로그 최종 구간에서는 **OFF의 collision이 더 좋았다**(−0.95 vs −1.45). 그런데 레벨을 고정하니 **반전됐다.**

학습 곡선의 tilt 격차(SYM −31%)도 고정 레벨에서는 재현되지 않는다.

**해석**: 학습 로그의 격차는 두 정책이 머무는 **커리큘럼 분포의 차이**가 만든 아티팩트였다. 잘하는 정책은 더 어려운 지형에 있으므로 충돌도 더 많다. **같은 난이도에 놓으면 SYM이 더 깨끗하게 통과한다.**

> 이 발견 이후 **"cross-run 능력 비교는 pinned probe가 정본"**이 프로젝트의 규칙이 됐다.

**영상**

| 파일 | 비교 내용 | 결론 / 주의 |
|---|---|---|
| [`assets/videos/08_step_symOFF_vs_symON_L6.mp4`](assets/videos/08_step_symOFF_vs_symON_L6.mp4) | step 지형, 좌 = symmetry OFF / 우 = ON, **레벨 6(판정 레벨)** | ✅ **양쪽 다 완등** — 낙상 없이(z_min 0.31~0.32 유지) 정상부 z≈2.0까지 등반, 프레임 4시점 검수 통과. **등반 시연은 이 클립이 정본이다** (L8 클립들은 아래 흡수 상태 증거용) |
| [`assets/videos/09_crawl_symOFF_vs_symON.mp4`](assets/videos/09_crawl_symOFF_vs_symON.mp4) | crawl 터널, OFF vs SYM | 양쪽 모두 몸을 낮춰 터널에 진입한다. ⚠ 측벽 때문에 추적 카메라가 벽 안쪽에 갇히는 구간이 있다 |

> ⚠️ **파생 관찰 — 낙상 후 종료되지 않는 흡수 상태가 존재한다.** L8 step 클립에서 추적 개체가 엎어진 자세로 tilt/base_contact 종료가 걸리지 않고 클립 끝까지 정지한 사례가 양 정책 모두에서 관측됐다(seed 1, seed 7 재렌더 모두). 종료 규칙은 양 정책 동일이라 A/B 공정성 자체는 유지되지만, probe의 `time_out` 집계에 이런 낙상-미종료 상태가 섞여 있을 수 있다.

---

## 10. 최종 성적 — 배포 가능한 정책은 특권 teacher에 얼마나 근접했나

2026-08-18 기준, 양쪽 arm을 **같은 날 같은 소스 cfg로 새로 렌더**해 비교했다.

| | GT 3D Voxel teacher | SL-Grid + SYM |
|---|---|---|
| task | `Go2-ParkourImitation-Teacher3DVoxelGT-EasyEntry-v0` | `Go2-ParkourImitation-Lidar-SL-Grid-Crawl-Sym-EasyEntry-v0` |
| 지각 입력 | 지형 메시에서 열거한 GT 복셀 (**특권, 배포 불가**) | Mid-360 히트를 teacher 프레임에 산란 (**배포 가능**) |
| 오버레이 색 | 노랑 | 청록 |

**공정성 실측**: 카메라는 대상 지형의 최고 레벨 env를 추적하는데, 두 arm이 **모든 지형에서 같은 env 인덱스와 같은 레벨**을 잡았다. 즉 같은 지형 배치의 같은 개체를 비교한다.

| 지형 | 추적 env / 레벨 | voxel 전진속도 | SYM 전진속도 | voxel z-range | SYM z-range |
|---|---|---:|---:|---:|---:|
| flat | 0 / L3 | 0.991 | 1.051 | 0.031 | 0.045 |
| hurdle | 6 / L8 | 1.060 | 1.095 | 0.148 | 0.163 |
| step | 8 / L8 | 1.038 | 1.032 | 0.195 | 0.289 |
| gap | 2 / L8 | 1.131 | 1.088 | 0.119 | 0.088 |
| stair | 5 / L7 | 1.158 | 1.131 | **1.541** | **1.556** |
| crawl | 2 / L8 | 0.965 | 1.052 | 0.114 | 0.175 |

**전진속도는 6개 지형 모두 1.0 m/s 안팎으로 육안 차이를 만들 만한 격차가 없다.** 계단 z-range도 1.541 vs 1.556으로 사실상 동일하다. 이 영상들은 "특권 teacher가 눈에 띄게 낫다"를 보여주지 **않는다 — 그 자체가 결과다.**

### 10.1 다만 "teacher는 필요 없다"의 강한 버전은 폐기됐다

2026-08-03 시점에는 SL-Grid가 pinned gap에서 당시 teacher와 **동률**이라는 근거로 "배포 가능한 센서만으로 privileged teacher를 따라잡는다"고 적었다. 그러나 그 teacher는 **관측 버그가 있던 상태**였다.

버그를 고친 teacher를 같은 프로토콜로 재측정하니 네 지형(gap/stair/step/hurdle) 모두에서 teacher가 위였고, 평균 격차 **+4.5 pt**였다. 다만 조건당 에피소드 194~275개에서 차이의 표준오차가 약 0.045이므로:

| 지형 | 격차 | SE 대비 |
|---|---:|---:|
| stair | +0.094 | **2.1 SE** |
| gap | +0.040 | 0.9 SE |
| hurdle | +0.039 | 0.9 SE |
| step | +0.008 | 0.2 SE |

**개별로 유의한 것은 stair 하나뿐**이다. 나머지 셋은 잡음 구간에 있다. 다만 네 지형이 모두 같은 방향을 가리키는 것 자체가 약한 증거이고(부호검정 4/4, 단측 p = 0.0625), stair는 전진속도도 확연히 빠르다.

> ⚠️ 위 표의 격차는 상대값이므로 유효하다. 그러나 그 근거가 된 문서의 **절대 완주율 수치는 §11.4의 probe 분모 버그 영향을 받아 과소 표기**돼 있다.

**정확한 표현**: 배포 가능한 센서만으로 from-scratch RL을 하면 특권 teacher와 **종단 거동에서 근접 동률**에 도달하며, **stair 한 지형에서만 유의한 잔여 격차**가 남는다. "teacher를 이겼다"도 "teacher가 불필요함이 증명됐다"도 아니다.

### 10.2 지형별 최종 주행 영상

좌 = GT 3D voxel teacher(특권, 노랑 오버레이), 우 = SL-Grid + SYM(배포 가능, 청록 오버레이). 라벨은 화면에 각인돼 있다. **오버레이 색이 다른 것은 의도적이다** — 각 정책이 실제로 보는 입력을 그리기 때문이며, 영상만으로 어느 쪽이 특권 정보인지 알 수 있다.

| 파일 | 난이도 | 결론 / 주의 |
|---|---|---|
| [`10_final_flat_voxelGT_vs_slgridSYM.mp4`](assets/videos/10_final_flat_voxelGT_vs_slgridSYM.mp4) | L3 | 평지 기준선 |
| [`11_final_hurdle_voxelGT_vs_slgridSYM.mp4`](assets/videos/11_final_hurdle_voxelGT_vs_slgridSYM.mp4) | L8 | |
| [`12_final_step_voxelGT_vs_slgridSYM.mp4`](assets/videos/12_final_step_voxelGT_vs_slgridSYM.mp4) | L8 | 프레임 검수 완료 — 양쪽 정상 주행 |
| [`13_final_gap_voxelGT_vs_slgridSYM.mp4`](assets/videos/13_final_gap_voxelGT_vs_slgridSYM.mp4) | L8 | |
| [`14_final_stair_voxelGT_vs_slgridSYM.mp4`](assets/videos/14_final_stair_voxelGT_vs_slgridSYM.mp4) | L7 | 프레임 검수 완료 — 양쪽 계단 완주. **정량 판정에서 teacher가 유일하게 유의하게 앞선 지형** |
| [`25_final_crawl_voxelGT_vs_slgridSYM_fixedcam.mp4`](assets/videos/25_final_crawl_voxelGT_vs_slgridSYM_fixedcam.mp4) — **2026-08-25 재렌더** | L8 | 카메라 픽스 재렌더(카메라 이동 + 측벽만 렌더 제외, 천장·top blocker 유지) — 로봇 가시율 voxel 99.8% / student 82.8%로 **영상 판정 가능**해짐(student의 잔여 부재 프레임은 터널 내부 통과 중 천장 차폐 = 통과의 증거). ⚠ 단 voxel teacher가 crawl-3 학습이라 crawl-8에서 **OOD**인 점은 그대로이므로 좌우 우열 비교로 읽지 말 것. 추종 개체는 구 영상과 다른 env(둘 다 env 4 · L8로 상호 동일). 구 영상(카메라 갇힘, [`15_...`](assets/videos/15_final_crawl_voxelGT_vs_slgridSYM.mp4))은 보존. 재렌더 조건: `reports/go2_parkour/_comparisons/voxel_vs_slgrid_sym/videos_fixed_camera/NOTES.md` |
| [`16_stair_gtvoxel_vs_slgrid.mp4`](assets/videos/16_stair_gtvoxel_vs_slgrid.mp4) | L6 | 이전 세대(symmetry 이전) SL-Grid와의 stair 좌우 비교. 두 오버레이가 **같은 격자 좌표계**(yaw 정렬, 0.1 m, clearance 마운트 원점)를 쓰므로 직접 겹쳐 볼 수 있다 |

> ⚠️ **전진속도는 12-env 평균이라 추적 개체를 보증하지 못한다.** 이 세션에서 평균 1.032 m/s인 클립의 추적 개체가 실제로는 넘어져 있던 사례를 겪었다. 그래서 stair·step·crawl 클립은 0.5~1.0초 간격 프레임 스트립으로 직접 검수했다.

### 10.2.1 ★ 단독 주행 클립 (2026-08-19 신규) — 로봇 1대 · 외란 없음

위 좌우 비교 6편과 별개로, **최종 정책(SL-Grid + SYM) 단독**을 지형별로 다시 찍었다.
비교 영상은 두 정책을 나란히 놓느라 화면이 절반씩이고 배경에 이웃 env 로봇이 작게 들어오는데,
발표에서 "이 정책이 이 지형을 어떻게 통과하는가"만 보여줄 때는 이쪽이 읽기 쉽다.

| 파일 | 지형 / 난이도 | 비고 |
|---|---|---|
| [`17_solo_flat_slgridSYM_followcam.mp4`](assets/videos/17_solo_flat_slgridSYM_followcam.mp4) | flat · L3 | 기준선 |
| [`18_solo_hurdle_slgridSYM_followcam.mp4`](assets/videos/18_solo_hurdle_slgridSYM_followcam.mp4) | hurdle · **L8** | 최고 난이도 |
| [`19_solo_step_slgridSYM_followcam.mp4`](assets/videos/19_solo_step_slgridSYM_followcam.mp4) | step · L7 | |
| [`20_solo_gap_slgridSYM_followcam.mp4`](assets/videos/20_solo_gap_slgridSYM_followcam.mp4) | gap · L6 | |
| [`21_solo_stair_slgridSYM_followcam.mp4`](assets/videos/21_solo_stair_slgridSYM_followcam.mp4) | stair · L4 | |
| [`22_solo_crawl_slgridSYM_followcam.mp4`](assets/videos/22_solo_crawl_slgridSYM_followcam.mp4) | crawl · L3 | **터널 내부 추적 카메라.** 기어가는 자세가 전 구간 보인다 (아래 참조) |

**렌더 조건**: `scripts/demos/play_per_terrain.py`, `--num_envs 2 --video_length 400
--max_init_level 8 --sensor lidar_grid`, run `parkour_imitation_go2_lidar_sl_grid_crawl_sym/
2026-08-10_12-35-38_slgrid_crawl_sym_scratch_50k`, `model_49999.pt`.
런타임 외란은 `env.events.push_robot.interval_range_s=[1000.0,1000.0]`으로 껐다
(기본은 8초 주기 ±0.5 m/s 임펄스). 청록 오버레이는 정책이 실제로 읽는 SL-Grid 격자다.

**crawl만 카메라가 다르다.** 나머지 다섯은 parkour env 기본 뷰(`eye=(0, -2.5, 0.8)`, 로봇 yaw
프레임 오프셋)를 쓰지만, crawl은 폭 1.2 m 통로에 **높이 1.0 m 측벽**과 불투명 천장 구간
(바닥면 0.28~0.50 m, 길이 1.2 m × 3구간)이 있어 기본 뷰가 측벽 바깥에서 지붕만 비춘다.
그래서 crawl은 **터널 안에서 따라가는 카메라**로 바꿨다:
`env.viewer.eye=[-1.75,-0.30,-0.04] env.viewer.lookat=[0.7,0.0,-0.01]`.

여기서 z 오프셋이 **음수**인 것이 요점이다. 이 오프셋은 지면이 아니라 **로봇 root 기준**인데
crawl 중 root 높이가 0.15~0.30 m로 오르내리므로, 양수 오프셋(예: +0.15)을 쓰면 로봇이 몸을
세우는 순간 카메라가 월드 z 0.45로 올라가 천장 박스 안에 파묻힌다(실제로 그 구성에서 프레임의
절반이 검게 나왔다). 음수 오프셋은 카메라를 항상 천장 아래·바닥 위에 묶어 둔다.

0.4초 간격 20표본으로 로봇 가시 프레임을 세어 비교하면 **기본 뷰 11/20 → 터널 내부 뷰 16/20**이다.
남은 4표본(4.4~4.8 s, 6.4~6.8 s)은 로봇이 불투명 천장 구간 **바로 아래**를 지나는 순간으로,
외부 어느 시점에서도 보이지 않는 지형 자체의 성질이다. 리드 검수 시점인 t=1 / 3 / 6 s는 셋 다
로봇이 선명하게 보인다.

**`--num_envs 2`가 핵심이다.** `_isolate_terrain`이 지형 컬럼 0을 flat 전용으로 예약하므로
env 0은 flat, env 1이 대상 지형에 놓인다. 즉 대상 지형 위의 로봇은 **정확히 1대**이고,
flat 컬럼의 나머지 1대는 화면 밖이다 — 6편 전부 프레임 검수로 배경 로봇 없음을 확인했다.
대신 추적 개체가 하나뿐이라 난이도 레벨이 추첨 1회로 정해지는데, 이 추첨은 시드에 묶여
**같은 시드면 같은 레벨이 나온다**(재실행 3회로 확인). hurdle과 gap은 기본 시드에서 L1 / L0이
나와 `--seed 21`로 다시 뽑아 L8 / L6을 얻었다. 나머지 넷은 기본 시드 결과다.

⚠️ 이 클립들은 §10.2 비교표와 **다른 렌더**다. 레벨도 추적 개체도 다르므로 §10.2의
전진속도·z-range 수치와 섞어 읽지 말 것.

### 10.2.2 §10.2 좌우 비교 6편을 재렌더하지 않은 이유

위 단독 클립을 새로 찍는 대신 §10.2의 비교 6편은 **그대로 두었다.** 판단 근거:

1. 이 6편의 값은 **두 정책의 좌우 대조**에 있고, 그 역할은 §10.2.1의 단독 클립이 대체하지 못한다.
   "로봇 1대를 카메라가 따라가는 화면"이라는 요구는 §10.2.1로 충족했다.
2. 화면 상단 지평선 근처에 이웃 env 로봇이 작은 점으로 보이는 것은 사실이다. 다만 이는
   `--num_envs 12`에서 `_pick_target_view_env`가 대상 지형의 **최고 레벨 env**를 고르는
   구조의 부산물이고, env 수를 줄이면 추적 개체가 서는 지형 레벨 자체가 달라진다.
   §10.2.1에서 실제로 그렇게 됐다 — 같은 기본 시드에서 hurdle이 **L8 → L1**, gap이 **L8 → L0**,
   stair가 **L7 → L4**로 내려갔다(§10.2.1은 hurdle·gap만 다른 시드로 다시 뽑아 L8·L6을 얻었다).
3. 재렌더는 영상만 바꾸는 일이 아니다. 위 표의 **L3/L7/L8 난이도 표기**,
   [`_comparisons/voxel_vs_slgrid_sym/`](../../go2_parkour/_comparisons/voxel_vs_slgrid_sym/)의
   **공정성 표**(두 arm이 지형별로 같은 env 인덱스·같은 레벨을 잡았다는 실측)와
   **전진속도·z-range 12개 수치**, 그리고 클립별 프레임 검수를 전부 재도출해야 한다.
4. crawl 클립의 측벽 카메라 차폐(위 표 참조)는 재렌더로 해결되지 않는다.
5. parkour env에도 `events.push_robot`이 있으나 `interval_range_s=(8.0, 8.0)`이고 클립이 8초라
   **최대 1회**다. 다른 프로젝트에서 문제가 된 3~5초 주기 외란과 성격이 다르다.

재렌더가 필요해지면 12편 렌더 + 6편 hstack 합성 + 공정성·수치 재도출 + 지형별 프레임 검수를
**하나의 배치로 함께** 돌려야 한다(§10.2 위 주의 참조: 이 스크립트는 run의 저장 cfg가 아니라
현재 소스 cfg로 env를 만들므로 두 arm을 같은 날 함께 찍어야 한다).

---

## 11. 측정 방법론 — 이 프로젝트가 자기 자신을 정정한 기록

이 문서 전체에 걸쳐 "정정", "빗나간 예측", "뒤집혔다"가 반복해서 나온다. **그것이 우연이 아니라 이 프로젝트의 방식이다.** 여기 다섯 가지로 모은다. 발표에서 결과를 인용할 때 반드시 함께 읽어야 한다.

### 11.1 seed는 전부 1개다

이 문서의 **어떤 표의 어떤 차이도 seed 분산에 대해 검정되지 않았다.** 유일한 예외가 §6.1의 EasyEntry 진단(seed 1/2/3)이다. pinned probe 자체는 거의 결정적이지만(4회 반복 중 md5까지 동일), **학습 자체의 seed 분산은 재지 않았다.**

### 11.2 커리큘럼 레벨은 완주율이 아니다

레벨은 정책이 **스스로 고른 난이도**다. 잘하는 정책은 어려운 지형에 많이 노출된다. 따라서:

- 학습 로그의 레벨·보상·collision 비교는 **분포가 다른 두 정책을 섞어 잰 것**이다.
- 이 아티팩트가 실제로 **부호를 뒤집은 사례**가 있다 (§9.4 — collision, tilt).
- **cross-run 능력 비교는 pinned probe가 정본이다.**
- 커리큘럼 레벨 인덱스는 **config에 종속된 상대값**이라 config가 다르면(EasyEntry 등) 직접 비교가 무효다. 반드시 물리 단위로 환산한다.

### 11.3 이른 스냅샷을 정상 상태로 읽는 실수를 세 번 했다

| 사례 | 이른 시점 판정 | 학습량을 맞춘 뒤 |
|---|---|---|
| A1-1(시간 누적) 이득 | 5k에서 2.1배 | 10k에서 1.17배 |
| teacher 기준점 | 0.260 (단일 pin 측정) | 0.357 (동일 프로토콜 재측정) |
| A1-0(격자) 표현 이득 | 10k에서 2.9배 | 50k에서 1.23배 |
| K=10 창 길이 | 10k에서 "효과 없음" | 50k에서 K=3보다 명확히 위 (**부호 반전**) |

**같은 실수가 세 번 반복됐다.** 원인은 매번 동일하다 — 이른 시점의 수렴 속도 차이를 정상 상태 성능 차이로 오독한 것이다.

또한 sub-terrain 지표는 std 1.5~2.1로 노이즈가 크다. **tail-N 스냅샷 비교 금지**, 대표본 robust 평균만 판정에 사용한다. (실제로 "3D teacher가 hurdle에서 퇴행한다"는 판정이 이 착시였고 정정됐다.)

### 11.4 ★ probe 분모 버그 — 절대 완주율 수치의 취급

**2026-08-18에 발견된 측정 버그다. 이 문서를 인용할 때 가장 주의해야 할 항목이다.**

`probe_stair.py`가 완주율의 분모를 **종료 원인 카운터 5개의 합**으로 잡았는데, env가 `time_out = reset_all`을 반환하는 구조라(`parkour_env.py:1699`, `terminated = zeros`) **성공 에피소드가 `cause_goal_reached`와 `time_out` 양쪽에 중복 집계**됐다. `time_out` 카운터는 실패 지표가 아니라 **총 에피소드 수**였다.

수정 커밋 `a3650351847`. 영향:

- **이 도구로 만든 과거 완주율 절대값은 전부 약 1.6배 과소 표기돼 있다.**
- **A/B 판정은 전부 불변이다** — 두 arm이 같은 배율로 스케일되므로 격차·순서·부호·SE 비교는 그대로 유효하다.
- 그러나 다음 두 서술은 **폐기됐다**:
  1. **"완주율 천장 약 35%"는 존재하지 않는다.** 정정 후 지형별로 42~70%로 넓게 갈린다 (hurdle이 가장 낮고 step이 가장 높다). "공유된 단일 천장"이라는 그림 자체가 성립하지 않는다.
  2. **"time_out이 종료의 59~70%를 지배"도 같은 버그의 산물이다.** 그 비율은 실패율이 아니라 분모를 자기 자신으로 나눈 값에 가까웠다.

**이 문서에서 정정이 완료된 유일한 절대 완주율은 §9.3의 SL-Grid symmetry A/B(55.97% / 56.82%)다.** §8과 §10의 다른 절대값들(0.078, 0.238, 0.353, 0.357, 0.373, 0.328 등)은 **상대 비교로만 읽어야 한다.**

### 11.5 영상의 LIVENESS 지표는 추적 개체를 보증하지 않는다

`play_per_terrain.py`가 로깅하는 전진속도는 **12 env 평균**이다. step SYM 클립의 추적 개체는 t≈1.5s에 엎어져 끝까지 정지했는데 평균은 1.032 m/s로 정상처럼 보였다. **영상 판정은 반드시 프레임 스트립 직접 검수를 거쳐야 한다.**

또한 `play_per_terrain.py`는 각 run의 `params/env.yaml`이 아니라 **현재 소스 cfg**로 env를 만든다. 다른 날 찍은 클립은 서로 다른 지형 위에 놓여 있을 수 있으므로, 비교 영상은 같은 날 함께 다시 찍어야 한다.

### 11.6 그 밖의 함정 (재현 시 필독)

1. **자기 가림 마스크 파일이 없으면 마스크가 조용히 꺼진다** (`body_occ_azel_grid.npy`) — 경고만 내고 학습은 계속된다. **절대 삭제·이동 금지.**
2. **좌표 정렬 오류는 무증상이다** — 학습이 돌고 loss도 그럴듯하다. §8.5의 불변식 두 개로만 잡힌다.
3. **hydra override가 조용히 무시된다.** 존재하지 않는 키를 넘겨도 exit 0에 JSON까지 정상 생성된다 — 이 경로의 hydra는 struct 모드가 아니라 미지의 키를 버린다. "에러 없이 돌았으니 반영됐다"는 추론이 실제로 틀린 결과를 보고할 뻔했다. 이후 cfg 파일을 직접 편집하고 `PARKOUR_GOALS_REGISTRY`로 **실제 코스 길이를 재측정해 로그에 남기는** 방식으로 바꿨다.
4. **`.gitignore`의 `**/__*` 규칙이 gym 등록 `__init__.py`를 추적에서 제외한다.** 즉 task 등록 코드가 커밋에 안 들어간다. 2026-08-12 커밋 `35bfc694146`에서 강제 추가로 해소됐다.
5. **frame stack은 상속되지만 사용되지 않는다** — SL-Grid의 격자 경로가 range image 경로를 통째로 건너뛴다.

---

## 12. 미해결 항목 · {확인 필요}

### 12.1 미측정 (Notion 현황판에 명시된 것)

- **(2026-09-03) LiDAR 센서 모델 실기 정합 — RealSensor arm(마운트 22~30° 무작위) 학습 중(§8.7 조치).** 제공자 확인(extrinsic 미사용, blind 0.8 m = A안, 마운트 rot = 30° 설계값)으로 센서 모델을 고쳐 18:17에 재학습을 시작했다(run `2026-09-03_18-17-33_slgrid_crawl_sym_scratch_50k_realsensor_dr22-30`, 1280 env). 남은 것: 10k 스냅샷 pinned probe, 그리고 설계 30° vs rosbag 실측 22~25°의 불일치 원인(녹화 중 몸통 pitch인지 붐 각도인지 — 사진·도면 필요). 이전 항목: **LiDAR 자기 가림 마스크 v2 — 재학습 보류, 센서 모델 수정 선행.** 원점 오류를 고친 v2 마스크(`assets/render_slgrid_pipeline/body_occ_azel_grid_v2.npy`, 1,381 광선 / 5.75 %, 전부 후방·하방)는 준비됐고 cfg 필드 `body_occ_grid_path`로 넘길 수 있다. v2만 바꾼 재학습을 13:20에 시작했다가 실기 rosbag 검토(§8.7) 결과 비반복 스캔·유효 반환 절반·마운트 pitch·후방 크롭·blind 정책을 먼저 고쳐야 해서 1,200 iter에서 중단하고 run을 삭제했다. 다음 학습은 §8.7 계획의 arm A/B/C. 제공자 확인 대기: 드라이버 extrinsic, 마운트, 배포 파이프라인의 크롭·blind 설정, 실기 경로가 같은 마스크 파일을 쓰는지.
- **지형별 완주율 격차의 원인.** hurdle(42~48%)이 step(59~70%)보다 약 20 %p 낮다. 왜인지 규명되지 않았다.
- **crawl 커리큘럼 포화 6.0이 상한인지 능력 한계인지.**
- **distill arm 다수가 teacher 미학습 지형 분포에서 라벨받았을 가능성.** docstring과 코드의 불일치가 있으며 미수정 상태다. 사실이라면 §8.3의 distillation arm 비교 일부가 영향을 받는다.
- **SL-Grid + crawl의 action std 상승 원인 분리** — 관측(SL-Grid) 때문인지 crawl 지형 추가 때문인지. (symmetry가 `RL_calf` 폭주를 해소한 것은 확인됐으나, std 상승 자체의 인과는 분리되지 않았다.)
- **GT teacher collision의 20k·38k 급락 스파이크** 원인 미조사.

### 12.2 {확인 필요} — 이 리포트 작성 중 확정하지 못한 것

- **AMP 참조 모션 `imitation/go2_jump`의 출처와 캡처 방식** {확인 필요}. cfg 경로(`ParkourImitationEnvCfg.amp_motion_pkl`)는 확인했으나, 실제 모션 데이터가 실기 캡처인지 다른 시뮬레이션 정책의 self-imitation인지 확인하지 못했다.
- **단계 2(2026-06) symmetry의 3-leg gait 해소에 대한 정량 A/B 수치** {확인 필요}. 해당 세션 보고서는 구현·정합성 검증까지이고 학습 결과 수치가 없다. symmetry의 정량 이득은 단계 6(2026-08, LiDAR 격자)에서 처음 통제 측정됐다.
- **`Go2-ParkourImitation-TerrainStyle-v0`(37차원 지형 불변 AMP)의 A/B 학습 결과** {확인 필요}. 환경 구현과 검증은 완료됐으나, 이후 세션에서 이 arm의 학습 결과를 인용한 문서를 찾지 못했다.
- **실기(real Go2) 배포 실험 여부** {확인 필요}. 모든 결과가 시뮬레이션이며, 이 문서의 "배포 가능"은 **관측이 실기에서 얻을 수 있는 것으로만 구성됐다**는 의미이지 실기 검증을 마쳤다는 뜻이 아니다.
- **단계 1~2 시기의 baseline(`Go2-Parkour-Direct-v0`) 장기 학습 최종 성능** {확인 필요}. 2026-06-11 프레임워크 문서에 "smoke500 PASS, 장기 학습 결과 미검증"으로 남아 있다.

---

## 13. 발표 슬라이드 구성 제안 (10장)

각 장에 넣을 자료와 "한 문장으로 무슨 말을 하는가"를 함께 적는다.

| # | 제목 | 핵심 메시지 (한 문장) | 자료 |
|---|---|---|---|
| **1** | **Parkour Learning — 4개월의 궤적** | 사족보행 로봇이 계단·구멍·터널을 스스로 넘게 만들되, **실기에 옮길 수 있는 센서만으로** 하는 것이 목표였다. | 진화 타임라인 표 (§0) + 최종 주행 영상 1클립(`21_solo_stair_slgridSYM_followcam.mp4` — 로봇 1대 단독, §10.2.1) |
| **2** | **과제와 세 겹의 난이도** | 지형을 못 보면 못 넘고, 커리큘럼은 자기 자신을 강화하며, 시뮬레이터 정보는 실기에 없다. | §1.2 세 항목 + 지형 6종 스크린샷 |
| **3** | **학습 프레임워크 — 비대칭 actor-critic** | actor는 실기에서 얻을 수 있는 것만 보고, critic은 시뮬레이터의 모든 것을 본다 — 이 비대칭이 전부를 가능하게 한다. | §2.1 파이프라인 도식 + obs/action 표 |
| **4** | **1단계 · 학습이 아예 안 되던 시기** | 원인은 알고리즘이 아니라 액추에이터 능력·상수 오타·"정지가 최적점"이라는 보상 기하학이었다. | §3.2 세 원인 + yaw 게이팅 수식 |
| **5** | **2단계 · Symmetry — 대칭 시스템이 만든 비대칭 병리** | 왼쪽 뒷다리 고착은 버그가 아니라 대칭 보상이 허용한 emergent equilibrium이었고, 보상이 아니라 **정책의 등변성**으로 풀었다. | §4.2 진단 3갈래 + §4.3 미러 수식($M_q$, y-flip 순열) |
| **6** | **3단계 · Imitation(AMP)과 EasyEntry** | 스타일은 discriminator로, 학습 불능은 커리큘럼 사다리의 첫 칸을 낮춰서 풀었다. | §5.2 AMP 보상식 + §5.3 37차원 지형 불변 feature + §6.2 EasyEntry 하한 표 |
| **7** | **4단계 · 2.5D에서 3D로** | 터널은 2.5D로 **원리적으로 표현 불가**하고, 3D 전환의 계산 비용은 사실상 0이며, crawl을 빼면 3D가 2.5D보다 낫다. | §7.3 지형별 표(crawl 제외 평균 강조) + `gap_3up_...mp4` + `voxel_crawl_best.png` |
| **8** | **★ 5단계 · 병목은 센서가 아니라 표현이었다** | 같은 LiDAR 히트를 각도 이미지 대신 미터 격자로 바꾸자, 관측이 **작아지면서** 성능이 올랐다 — blind band 가설은 틀렸다. | **`sensor_raw_vs_grid_stair.png`(메인 시각자료)** + §8.4 축 비교 표 + tilt 대응 표 |
| **9** | **★ 6단계 · 결과 — 특권 teacher와의 근접 동률** | 배포 가능한 센서만으로 from-scratch RL을 하면 특권 teacher와 종단 거동이 붙고, stair 한 지형에만 유의한 잔여 격차가 남는다. | §10 전진속도·z-range 표 + `final_stair_...mp4` 좌우 비교(teacher 대조가 이 장의 논지) + §9.3 symmetry 3대 이득 |
| **10** | **측정을 믿을 수 있게 만든 것들** | 이 프로젝트의 결과가 신뢰할 만한 이유는 **스스로를 네 번 정정했기 때문**이다. | §11 다섯 항목 요약 (특히 커리큘럼 아티팩트 반전, 이른 스냅샷 3회, probe 분모 버그) + §12 미해결 |

**12장 확장안** (시간이 허락하면):

- **8-1** 사이에 **"distillation을 왜 버렸나"** 1장 추가 — Ceiling arm이 teacher를 넘은 것과 conditional-mean 천장 논증 (§8.5). `07_distill_matched50k_4up.mp4` 활용.
- **9-10** 사이에 **"학습 곡선이 말해주지 않는 것"** 1장 추가 — `curve_optimization.png`의 std 폭주가 사실 관절 하나의 문제였고 symmetry가 구조적으로 없앴다는 서사 (§9.1 → §9.3). 발표에서 가장 "탐정물"처럼 들리는 대목이다.

**발표 시 주의**: 절대 완주율 숫자를 슬라이드에 올릴 때는 §11.4를 반드시 확인할 것. 안전한 절대값은 §9.3의 55.97% / 56.82%뿐이고, 나머지는 배율·격차·부호로 말해야 한다.

---

## 부록 — 참조

### 정본 코드

| 역할 | 파일 |
|---|---|
| 기본 parkour env / cfg | `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`, `parkour_env_cfg.py` |
| 지형 생성 | `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_terrains.py` |
| symmetry 미러 (수식 구현) | `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/mdp/symmetry.py` |
| 복셀 격자 규격 / GT 열거 | `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/voxel_occupancy.py`, `voxel_column_pattern.py` |
| clearance 3D 패턴 | `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/clearance_3d_pattern.py` |
| AMP imitation env | `source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/parkour_imitation_env.py`, `_env_cfg.py` |
| EasyEntry cfg | `.../parkour_imitation_random_goal_env_cfg.py` (`ParkourImitationRandomGoalEasyEntryEnvCfg`) |
| **LiDAR 산란·누적** | `.../parkour_imitation_random_goal_lidar_env.py`, `_env_cfg.py` |
| 러너·라우팅 (`obs_groups`) | `.../parkour_imitation/agents/rsl_rl_amp_cfg.py` |
| 지형별 재생 하네스 | `scripts/demos/play_per_terrain.py` |
| 레벨 고정 probe | `scripts/demos/probe_stair.py` (이름은 stair지만 전 지형 공용) |

### 원자료 위치

| 내용 | 경로 |
|---|---|
| 학습 run 보고서 | `reports/go2_parkour/parkour_imitation_go2_{teacher3d,teacher3d_voxel,teacher3d_voxel_gt,lidar_distill,lidar_sl_grid_crawl,symmetry_random_goal}/` |
| 비교 실험 | `reports/go2_parkour/_comparisons/{height_scan_vs_3d_noncrawl,gt_voxel_vs_lidar_slgrid,gt_voxel_vs_slgrid_crawl,voxel_vs_slgrid_sym,slgrid_sym_pinned_probe,per_terrain_video_clearance_vs_voxel,...}/` |
| distillation 프로그램 | `reports/go2_parkour/_comparisons/{a0-window-length,a1-representation,a1-blind-band,matched-50k,program-summary-0727-0731}/` |
| Notion 프로젝트 페이지 | [Parkour Learning](https://app.notion.com/p/352fcb97742f80c9acb4ffa4d974f091) |
</content>
</invoke>

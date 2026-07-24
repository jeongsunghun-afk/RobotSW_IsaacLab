# Leg Imitation Tracking 환경

`go2_imitation_tracking`(AMP 모방학습 + body-frame 속도추종)을 **Leg_URDF2 로봇(17-DOF)** 으로 이식한 환경.
알고리즘·러너·AMP 구조는 동일하고, 로봇 형상과 모션 데이터셋만 교체했다.

**Task ID**: `Leg-Imitation-Tracking-v0`

---

## 로봇: Leg_URDF2 (17-DOF)

| 항목 | 값 |
|------|-----|
| 다리 | 4개 × (hip, thigh, calf, **foot**) = 16 |
| 몸통 | `FB_waist_joint` (z축 허리 관절) = 1 |
| 총 DOF | **17** |
| 총 질량 | 약 38 kg |
| 기립 base 높이 | 약 0.50 m (영자세 기준) |
| Articulation cfg | `isaaclab_assets.robots.rga.LEG_CFG` |

Go2와 달리 **다리마다 발목(foot) 관절이 하나 더** 있고 **허리 관절**이 있다.
허리는 정책이 직접 제어한다 (레퍼런스 모션에 척추 굽힘이 실제로 들어있어 AMP가 이를 모방하려면 제어 가능해야 함).

### ⚠ USD 자산 주의

원본 `data/Robots/Leg/Leg/Leg.usda` 는 **rigid body + joint + inertia 만 있고 visual/collision mesh 가 전혀 없다**
(`CollisionAPI` 0개, `Mesh` prim 0개). 이 자산으로는 로봇이 지면을 통과해 자유낙하한다.

따라서 URDF 를 재임포트한 자산을 사용한다:

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /home/lgb/Dog_Motion_data_3D/robots/Leg_URDF2/urdf/Leg.urdf \
  source/isaaclab_assets/data/Robots/Leg/Leg_gen/Leg.usd \
  --joint-target-type position --headless
```

→ `LEG_CFG.usd_path = .../Leg/Leg_gen/Leg.usd/Leg/Leg.usda` (collision 21개 생성됨).
`--fix-base` 는 절대 쓰지 않는다 (floating-base 4족).

---

## 모션 데이터셋

원본: `/home/lgb/Dog_Motion_data_3D/mujoco_retarget_leg/smr_dataset_spin/new_dataset/*.txt`
(DeepMimic JSON, 프레임당 71값 = root_pos 3 + quat_xyzw 4 + qpos 17 + qvel 23 + foot 24)

변환:

```bash
python /home/lgb/Dog_Motion_data_3D/mujoco_retarget_leg/convert_smr_to_pkl.py \
  --src_dir /home/lgb/Dog_Motion_data_3D/mujoco_retarget_leg/smr_dataset_spin/new_dataset \
  --dst_dir source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/imitation/smr_leg_pkl \
  --prefix leg_ --include walk trot run --overwrite
```

PKL 프레임 = root_pos(3) + root_exp_map(3) + joint_pos(17) = **23값**.

사용 모션 9개 (보행 계열만 — 점프/pace/spin 제외):

| 파일 | 프레임 | vx 범위 [m/s] |
|------|--------|--------------|
| `leg_walk.pkl` | 198 | -0.01 ~ 0.49 |
| `leg_walk1.pkl` | 37 | 0.80 ~ 1.07 |
| `leg_walk2.pkl` | 128 | 0.36 ~ 0.92 |
| `leg_walk_turn.pkl` | 208 | 0.16 ~ 1.49 |
| `leg_trot0.pkl` | 208 | 0.72 ~ 2.49 |
| `leg_trot1.pkl` | 208 | 0.72 ~ 2.49 |
| `leg_run0.pkl` | 88 | 1.22 ~ 4.00 |
| `leg_run1.pkl` | 177 | 1.53 ~ 4.07 |
| `leg_run2.pkl` | 71 | 3.74 ~ 5.86 |

> `leg_trot0` 과 `leg_trot1` 은 원본 txt 가 동일하다 (중복).

### 관절 순서 (중요)

PKL 의 관절 순서(= MuJoCo qpos 순서)와 IsaacLab articulation 순서는 **완전히 다르다**.

```
PKL(DOF_NAMES):   HL_hip,thigh,calf,foot | HR_… | FB_waist | FL_… | FR_…
IsaacLab(BFS 깊이순): HL_hip, HR_hip, FB_waist, HL_thigh, HR_thigh, FL_hip, FR_hip, …
→ motion_dof_indices = [0, 4, 8, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 12, 16]
```

`motion_lib.get_dof_index()` 가 관절 **이름**으로 런타임 재매핑하므로 인덱스를 손으로 관리하지 않는다.
PKL 에는 `dof_names` 키가 함께 저장되어 로드 시 순서 일치를 assert 한다.

### exp-map vs Euler (go2 대비 수정)

go2 파이프라인은 변환 스크립트가 `[3:6]` 에 **exponential map** 을 쓰는데 `motion_lib` 은 이를 **ZYX Euler** 로
읽는 불일치가 있었다. Leg 버전은 exp-map 으로 일관되게 해석하고, 각속도도 Euler rate 근사가 아닌
**quaternion finite difference** 로 계산한다.

---

## 환경 스펙

| 항목 | Go2 버전 | **Leg 버전** |
|------|---------|-------------|
| DOF | 12 | **17** |
| `observation_space` | 48 | **63** |
| `action_space` | 12 | **17** |
| AMP obs (per-step) | 49 | **59** |
| `amp_observation_space` | 490 (49×10) | **590 (59×10)** |
| `reference_body` | `base` | **`Base`** |
| Domain Randomization / PACE | 있음 | **제거** |

Physics 200 Hz / policy 50 Hz (decimation 4), episode 10 s 는 동일.

### Observation Space (63-dim)

| idx | 성분 | dim |
|-----|------|-----|
| 0–2 | `root_lin_vel_b` | 3 |
| 3–5 | `root_ang_vel_b` | 3 |
| 6–8 | `projected_gravity_b` | 3 |
| 9–10 | `lin_vel_cmd` (vx, vy) | 2 |
| 11 | `yaw_vel_cmd` | 1 |
| 12–28 | `joint_pos - default` | 17 |
| 29–45 | `joint_vel` | 17 |
| 46–62 | `actions` | 17 |

### AMP observation (59-dim per step)

`dof_pos(17) + dof_vel(17) + root_height(1) + root_lin_vel(3) + root_ang_vel(3) + foot_pos_local(12) + root_rot_tan_norm(6)`

`foot_pos_local` 은 **`*_foot_link` 원점(발목)** 위치를 base frame 으로 옮긴 값이다.
- sim 측: `body_pos_w[KEY_BODY_NAMES]` 를 root quat 으로 역회전
- 레퍼런스 측: `motion_lib._leg_fk_foot_pos()` 순운동학 (URDF 링크 오프셋 그대로, 앞다리는 허리 회전 반영)

발끝(toe)이 아닌 발목 원점을 쓰는 이유는 sim 에 toe 강체가 없어 양측을 동일 정의로 맞추기 위함이다.
발목 관절 각도 자체는 `dof_pos` 에 포함되어 정보 손실이 없다.

순서는 양측 모두 **`[FL, FR, HL, HR]`** (`motion_lib.FOOT_ORDER` == `env.KEY_BODY_NAMES`).

---

## Command 범위 / 보상

Go2 버전과 동일한 구조.

| 명령 | 범위 |
|------|------|
| `vx` | [0.0, 4.0] m/s |
| `vy` | 0.0 고정 |
| `yaw_rate` | [−1.5, 1.5] rad/s |
| 재샘플링 주기 | [4.0, 7.0] s |

```
reward = 0.7 × exp(-0.5·‖lin_vel_cmd − lin_vel_b‖²) + 0.3 × exp(-0.5·(yaw_vel_cmd − yaw_vel_b)²)
```

AMP reward 는 `task_reward_lerp` 스케줄로 혼합 (`agents/rsl_rl_ppo_cfg.py`).

---

## 실행

```bash
# 학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Leg-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --video --video_length 1000 --logger wandb --wandb-project IsaacLab-locomotion

# 평가
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task Leg-Imitation-Tracking-v0 --num_envs 32
```

---

## 파일 구조

```
leg_imitation_tracking/
├── __init__.py                          ← task 등록 (Leg-Imitation-Tracking-v0)
├── leg_imitation_tracking_env.py        ← 환경 본체
├── leg_imitation_tracking_env_cfg.py    ← 환경 config
├── motion_lib.py                        ← 17-DOF PKL 로더 + Leg 순운동학
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py               ← LegImitationTrackingPPORunnerCfg
└── imitation/
    └── smr_leg_pkl/*.pkl               ← 변환된 모션 9개
```

---

## go2 버전 대비 수정한 결함 2건

1. **exp-map / Euler 불일치** — 위 "모션 데이터셋" 절 참조.
2. **link vs COM 속도 불일치** — `motion_lib` 은 base **link** frame 속도를 주는데,
   go2 env 는 `write_root_com_velocity_to_sim_index` 로 쓰고 obs/AMP 에서 `root_lin_vel_b`(COM 기준)를 읽는다.
   Go2 는 base COM 오프셋이 작아 무해했으나 Leg 는 COM 이 `(-0.130, 0, 0.046)` 이라
   ω=4 rad/s 에서 0.5 m/s 급 오차가 AMP reference↔policy 사이에 생긴다.
   - 검증된 관계식: `v_com = v_link + ω × r_com`
   - fix: obs/AMP/reward 를 `root_link_{lin,ang}_vel_b` 로 교체 + RSI write 에 `ω × r_com` 보정 삽입
   - 실측: 보정 전 0.389 → 보정 후 0.030 m/s

> `go2_imitation_tracking` 은 이 2건 모두 **미수정** 상태다.

## env 간 충돌 — 이미 필터링됨 (확인 완료)

`replicate_physics=True` 이므로 direct 워크플로에서 cross-env 충돌이 자동 필터링된다.
실측: 두 로봇을 정확히 같은 좌표에 겹쳐 놓고 0.3 s 돌려도 중심 거리 0.023 m 유지, 상호 접촉력 0 N.

영상에서 로봇들이 겹쳐 보이는 것은 **시각적 현상**이다 — `vx` 최대 4 m/s × 10 s 에피소드면
최대 40 m 를 이동해 `env_spacing=5.0` 셀을 벗어나지만, 물리적 상호작용은 없다.
`env_spacing` 을 키울 이유가 없다.

## 정합성 검증 방법 (재현용)

리셋 **직후**에는 자식 링크 pose 와 속도 버퍼가 physx 에 전파되지 않아 reference↔sim 대조가
무의미한 큰 오차를 낸다 (`sim.forward()` / `scene.update(0)` 으로도 해결되지 않음).
FK 검증은 **1 스텝 진행 후 동일 시점 데이터**로 해야 한다.

| 항목 | 결과 |
|------|------|
| `dof_pos` / `dof_vel` remap | 0.0 (완전 일치) |
| `root_pos` / `root_height` / `root_quat` | ≤ 1e-4 |
| `foot_pos_local` FK vs sim geometry (1스텝 후) | **3e-6** (float32 노이즈) |
| RSI 속도 write (COM 보정 후) | 0.030 m/s |

## PD 게인 — 유효관성 I_eff 기반

`_workspace/leg/compute_leg_ieff.py` 로 USD 에서 관절별 유효관성(locked distal-subtree 관성
= mass matrix 대각 `H[j,j]`, rest pose)을 해석적으로 계산해 도출했다. 물리 씬을 만들지 않고
USD stage 만 파싱하므로 GPU 경합과 무관하게 돌아간다.

```bash
./isaaclab.sh -p _workspace/leg/compute_leg_ieff.py --headless
```

| type | I_eff+arm [kg·m²] | Kp | Kd | ω_n [rad/s] | 근거 |
|------|------------------|-----|-----|------|------|
| hip | 0.17725 | 71 | 6.4 | 20.0 | inertia-scaled |
| thigh | 0.13302 | 53 | 4.8 | 20.0 | inertia-scaled |
| calf | 0.03782 | 134 | 4.1 | 59.5 | 접지 floor (τ_trot 33.5 / 0.25 rad) |
| foot | 0.01157 | 59 | 1.5 | 71.2 | 접지 floor (τ_trot 14.7 / 0.25 rad) |
| waist | 1.28139 | 112 | 21.6 | 9.3 | 권한 cap (τ_max 28 / 0.25) |

**I_eff 스프레드가 922배**라 균일 게인은 관절마다 감쇠비가 제각각이 된다. 정책:

- `Kd = 2ζ·√(Kp · I_tot)`, ζ = 0.9 — **모든 관절에서 감쇠비를 동일하게 유지**한다. I_eff 측정의 핵심 payoff.
- **hip / thigh** — 자유 스윙 관절. `Kp = ω_n²·I_tot` (ω_n = 20 rad/s) 그대로.
- **calf / foot** — 접지 관절. 관성 기반 Kp(15 / 4.6)는 스탠스 하중에 붕괴하므로
  trot(2다리 지지) 정적 토크 / 허용처짐 0.25 rad 로 **floor**.
- **waist** — 관성 기반 Kp 513 은 τ_max=28 대비 0.055 rad 에서 포화하므로 τ_max/0.25 로 **cap**.

### 검증 결과

| 항목 | 균일 kp=300/kd=5 | **I_eff 기반** |
|------|------------------|---------------|
| 학습 스모크 (64env, 20iter) episode length | 62.6 | **75.9** |
| 같은 조건 reward | 53.3 | **62.7** |
| 정적 기립 (영자세, 2 s 정착) | — | base 높이 0.497 m 유지, 최대 sag 0.166 rad |
| 관절 평균 토크 포화율 | — | 4.4% |
| 채터링 | — | 없음 (`tau_std` = 0.000) |

### 알려진 제약

- **앞다리 thigh 가 토크 envelope 에 근접**한다 (포화율 29~30%, 평균 |τ| ≈ 68% of 28 N·m).
  게인 과강성이 아니라 **하중** 때문이다 — trunk CoM 이 앞쪽이라 앞다리가 무게의 약 2/3 를 받는데
  thigh τ_limit 은 28 N·m 다 (정적 기립 시 sag 는 0.02 rad 로 작다 = 필요 토크를 실제로 내고 있다).
  Kp 를 낮추면 sag 만 커지고 필요 토크는 그대로다. 학습이 앞다리에서 막히면 **하드웨어 제약**으로 볼 것.
- 4발이 지면에 물린 폐쇄연쇄라 정적 기립에서도 일부 관절에 **정적 내부력**이 크게 걸린다
  (좌우 sag 는 대칭인데 토크는 비대칭). 채터링은 아니며 정책이 움직이면 해소된다.
- **armature** — 0.01 (nominal). 실기 식별값 없음.
- **contact sensor 가 강체 1개(`Base`)만 잡는다 — 미수정.**
  URDF 재임포트 USD 는 중첩 계층(`Robot/Geometry/Base/HL_hip_link/HL_thigh_link/…`)인데
  `contact_sensor.prim_path = "/World/envs/env_.*/Robot/.*"` 는 한 단계만 매칭한다
  (Go2 USD 는 평평한 계층이라 문제가 없었다).
  실측: 로봇을 뒤집어 바닥에 눕혀도 접촉력 0 N → `_get_dones` 의 **base 접촉 종료 조건이 무력**하다.
  높이(0.25 m)·자세(70°) 종료 조건이 중복 커버하므로 학습 자체는 정상 진행된다.
  **발 접촉 기반 reward(air time, gait pairing 등)를 추가하려면 반드시 먼저 고쳐야 한다.**
- **Domain Randomization** — 제거된 상태. 실기 이식 단계에서 재도입 필요.
- **mirror augmentation** — 미적용 (허리 + 다리당 4관절 미러 매핑 필요).

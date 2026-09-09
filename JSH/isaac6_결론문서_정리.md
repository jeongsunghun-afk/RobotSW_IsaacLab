# isaac-6.0 결론 문서 정리 (2026-09-09)

`upstream/isaac-6.0` = 회사(Gangbok2) 활성 개발 라인. **`main`과는 2026-02-21에 갈라진 별개 브랜치**
(IsaacLab 3.0.0 vs main 2.3.2, main에 없는 커밋 1020개). 로컬 `feat/quad17-dtc-gaptest`(내 DTC/quad17 작업)와
merge 대상이 아니라 **참조·이식 대상**이다.

- 워크트리: `/home/jsh/문서/jsh/RobotSW_IsaacLab_isaac6` (브랜치 `isaac-6.0`, 94MB)
- 최신 커밋: `bea85978d` 2026-09-07 "Confirm HindLeg HR_hip saturation and tune torque penalty 2x"
- 원문 위치: 워크트리의 `reports/{leg_imitation,hindleg_locomotion,real2sim}/`

> ⚠ **저장소에 없는 것**: ① USD 에셋(rga.py가 `/home/lgb/...` 절대경로 참조) ② R.pet AMP용
> s16 Phase-T 데이터셋 16클립(`dataset_rpet_walktrot_T.yaml`·`rpet_usd_order/`). `smr_leg_pkl/`엔
> 구버전 9개(`leg_walk*`/`leg_trot*`/`leg_run*`)만 있다. 재현하려면 둘 다 서버에서 따로 받아야 한다.

---

## A. R.pet 17-DOF — `leg_imitation_tracking` (AMP + RMA, 평지 속도추종)

태스크 `Leg-Imitation-Tracking-RMA-v0`. 4족 12관절 + 허리 1 + 발목 4 = 17-DOF.
골격 = rsl_rl PPO + AMP discriminator + RMA privileged encoder. 아래는 그 위에 누적된 항/데이터 변경의 판정.

| # | 문서 | 결론 |
|---|---|---|
| 1 | `imitation_method_comparison` (08-26) | **단일 지표로 방법 순위를 못 매긴다.** 정상주행(cmd 2~4)·정지출발(cmd 0.5)·자세대칭 세 축의 1위가 전부 다른 방법. **정지출발은 `reset_strategy`가 결정**(추종 품질과 무관). ★**RMA estimator가 평가·배포 경로에서 빠져 있었다** — 배포조건 재측정 시 정상주행은 8개 방법 전부 무손실(\|Δ\|≤1.6pp)인데 **정지출발만 평균 −6.2pp** |
| 2 | `lowspeed_gait_onset_hysteresis` (08-26) | cmd 0.5 실패 = "0.5 m/s를 못 낸다"가 아니라 **"정지에서 보행으로 진입을 못 한다"**. 램프 상승 1~64% vs 하강 61~77%. 기전 = `reset_strategy: random`이면 정지 리셋을 0회 겪음 → `random`(−0.6/1.9/30.0%) vs `random_stand`(43.7~68.5%)로 완벽 분리 |
| 3 | `arch_scaling_study` (08-26) | **깊이 스케일링 스윕 안 함.** 아키텍처 가설 3개(관측성·증류/estimator OOD·표현력) 전부 새 학습 없이 기존 데이터 재분석으로 반증. 진짜 원인 = **보상 모양**(cmd 0.5에서 정지가 task 보상의 88.25%를 받고 탈출 유인이 고속 대비 8.5배 얕음). **레버 = `vel_err_scale`** |
| 4 | `velscale_deploy_and_symmetry` (50k) | **`vel_err_scale=1.5`가 세 축 모두 1등**(steady 96.6 / start 71.8 / skew 0.123). `1.0`은 기준보다 나쁨(45k에 symmetry loss 0.630, cmd 4.0에서 8회 중 4회 낙상). **처방 = 명령을 3.2로 캡 + s=1.5**(외삽 3.5~4.0은 배포조건에서 붕괴 2배). ★**영상은 측정이 아니다** — 카메라 켜면 같은 플래그인데 붕괴율 25%→80% |
| 5 | `gait_attractor_pace_vs_gallop` (08-28) | **걸음 종류를 고르는 것은 명령이 아니라 에피소드 초기 상태.** 참조리셋 gallop 21.9% vs 정지리셋 0.9%(명령 분포 동일). 램프 평가가 항상 `--force_stand`라 pace attractor만 표집해 왔다. 직선 gallop은 존재(\|yaw\|<0.05에서 24.2%) |
| 6 | `rsi_command_matched_gait` | 처방②(RSI 클립을 명령속도에 매칭) **기각 — 초기화 artifact**. 대신 **근본원인 발견: `_post_physics_step`이 호출되지 않아 명령이 에피소드 내내 고정**(정책이 "달리는 중 명령 변화"를 겪은 적 없음) |
| 7 | `cmdchg_in_episode_command` | 처방③(에피소드 중 명령 재샘플) 20k에서 기각했다가 **50k에서 뒤집힘**: 정지출발 gallop 0.0%(20k) → **37.9%(50k)**. gallop은 소멸이 아니라 **지연**됐다 |
| 8 | ★ `conditional_discriminator` (09-02~) | **돌파.** `amp_cond_mode: speed` — AMP obs 끝에 `[\|v_cmd\|/v_max, valid]`를 붙여 판별자를 명령속도에 조건화. 세 지표가 동시에 처음 움직임: 속도선택성(cmd 1.0의 0.0% → 2.5~3.5의 30.3% 단조증가), **정지출발 gallop 28.4%**(프로젝트 최초로 0%가 아님, 이후 72.3%), 저속→고속 전환 6.7~11.7%(앞선 세 팔 합계 0.15%). iter 맞춤(16k) 비교에선 **cond+drail**이 최고(12.2/12.0/13.6%) |
| 9 | `cumirror_weighting` (09-03) | `command_uniform` → `command_uniform_mirror`(짝 없는 클립의 미러본을 먼저 합성 → expert 좌우편향 구조적 0). 옆으로 흐른 거리 중앙 1.636 → **0.519 m**(Mann-Whitney p=0.038). **겨눈 축은 고쳤으나 더 나은 정책은 아님**(추종·정지출발이 안 따라와 s=1.5를 대체 못 함) |
| 10 | `run_matrix_38` (09-04) | 판정 문서가 아니라 **레지스트리** — 38 run의 `params/*.yaml` 기계적 diff. config에 키가 처음 등장한 시점 = 프로젝트 연표 |

### ★ 반복해서 나오는 방법론 교훈

**중간 체크포인트로 판정하지 말 것 — 40k 이후만.** 이 프로젝트에서 뒤집힌 사례 4건:

| 무엇 | 중간 시점 | 최종 |
|---|---|---|
| `cmdchg4s` gallop | 20k 0.0% (→ 처방 기각) | **50k 37.9%** (기각 무효) |
| `vel_err_scale=1.0` symmetry loss | 24k 0.10 (기준보다 대칭적) | 45k **0.630** (기준 0.325) |
| `trot110150` cmd 2.0 | 15k +11%p | 50k **−13%p** |
| `ds14` 저속 우위 | 8k 67% | 15k **0%** |

`gait_transition_investigation`은 이 위반으로 본문 결론(래칫 기전)이 통째로 무효화됐고, 문서 상단에
정정 블록을 달고 본문은 기록으로 보존하는 방식을 쓴다. 다른 문서들도 같은 패턴(정정/철회 블록)을 쓴다.

---

## B. hind_leg 8-DOF (2족) — `HindLeg-Direct-v0`

내 biped 트랙과 **같은 로봇**. 회사 쪽은 r2s_bridge 검증 플랜트(URDF3_SignFix)로 env를 개편한 세대다.

- **현행 baseline** = `2026-08-13_12-04-06_colmesh_v2_gpu2` (400 → **50,399 완주**, 15.1h).
  `track_lin_vel_xy_exp` 0.7616(+28%), mean reward 47.63(+35%). cmd 0.5에서 실제 0.487(**97%**).
  `_sole_rest_z` 수정 후 구 플랜트 성공 run을 **4배 빠른 iter로** 따라잡음.
- ★ `fall_asymmetry_termination` (08-27) — **`_get_dones`가 `base.*` 접촉만 봤다.** 다리가 몸통 아래로
  뻗은 구조라 **뒤로 자빠지면 다리 링크가 몸통을 받쳐 base가 지면에 안 닿는다** → 그 자세 종료율
  **0.000%**, 정책이 에피소드의 **92.9%를 누운 채** 보냄(time-out 674건 전부 넘어진 채).
  기울기 종료(60°) 추가하니 에피소드 길이 708.6 → 69.9 step. **이전 토크 측정치 전부 무효화**
  ("걷는 로봇"이 아니라 "누워 있는 로봇"의 토크였다).
- ★ `hip_saturation_rootcause` (09-02) — **HR_hip은 제어축이 아니다.** 클램프 *이전* 출력이
  상수 **−12.605**(\|mean\|/std = 9.83, 다른 축은 0.09~1.04), **99.9% 하단 클램프**, 가동폭 0.468 rad인데
  **8배를 초과 요구**, gait clock 위상상관 +0.22(다른 축 0.62~0.93).
- `reset_noise_ab` (09-02) — 초기자세 노이즈(0.1 rad / 5°) fine-tune. **겨냥한 증상은 안 고쳐졌고,
  그 과정에서 원인이 초기화가 아님이 드러났다**(추세 0이라 2,201 iter에서 중단).
- `torque_penalty_2x` (09-07, **최신 커밋**) — `joint_torque_reward_scale` −0.0002 → −0.0004로 5k fine-tune.
  **`sum(τ²)` −15.7%**, 고정명령 재측정 시 **실제 전진속도 무손실**(±0.01 m/s), 토크 감소는 속도가
  붙을수록 커짐(−6.5/−10.1/−12.1%). **단 이건 sim 토크** — 근거였던 "실기 thigh가 채널 15 N·m를 6.9% 초과"는
  새 실기 캡처 없이는 확인 불가.

---

## C. real2sim — 실기 브리지 + PACE 액추에이터 식별

내 `sim2real 체크리스트` A항(액추에이터 물리)·C항(미모델)에 직접 대응. 실기 캡처가 이미 돌아가고 있다.

- ★★ `GEAR_K_INVALIDATION.md` — **2026-08-13 이전 biped_leg 실기 캡처·PACE 적합 전량 무효.**
  브리지(`real_runner_bipedleg.cpp:73`)가 **채널각을 관절각으로 취급**했다(`gear_k`로 나누는 항 없음).
  실제 감속비 hip 7 / thigh 7 / calf 10.5 / foot 8.4 → `gear_k` calf **1.5** · foot **1.2**.
  즉 두 축의 모든 각도·명령이 각각 1.5배/1.2배 어긋난 채로 데이터를 모으고 그 위에서 적합을 돌렸다.
- `pace_bipedleg_foot_coupling_probe/README.md` (247K, 39절) — PACE 식별 연대기. 주요 매듭:
  §12 반사관성 off-diagonal은 무시 못 하나 **명시적 보정토크는 불안정** / §15 **기록된 명령이 실제 적용된
  명령이 아니었다**(고진폭 붕괴 원인) / §23 재적합 14~32× 개선하나 **armature 미식별** → §24 **원인은
  시간축**, 세 판정 통과 / §35 encoder bias 8개는 잉여가 아니라 **해로웠다** / §38 stock 대비 **5.4배**,
  `rga.py`에 반영하고 학습 착수 / §39 **학습이 안 붙는 범인은 PACE가 아니라 08-14 구조항 3개**.
- ★ `CONTACT_CAPTURES.md` (08-27) — **접촉 영역 실기 데이터가 한 건도 없다**(PACE 적합에 쓴 13캡처가
  전부 `fix_base` 공중 고정). 그런데 학습을 막는 항이 정확히 그 영역에 산다: `corr = I_off·q̈_foot`가
  calf effort limit(126 N·m)을 넘는 스텝의 **100%가 발 접촉 개시 스텝**(기저율 3.38% 대비 **29.6배**).
  게다가 그 크기는 물리량이 아니다 — 제어율 고정한 채 `sim.dt`만 5→1.25 ms로 줄이면 **5.64배**로 커진다.
- `bipedleg_sensor_pipeline_20260901` — 브리지 센서 변환 전수 추적. 아핀변환·foot-calf 커플링 해제·
  projected gravity는 손 유도 검증됐으나 **미검증 가정 4건**이 실사용 중: **STATE 패킷에 IMU valid bit 부재**
  (08-28 캡처의 IMU 482.9초 무수신이 안 드러난 기전), **`zero_deg` 8축 전부 0인 채 TRACK 모드 운용**.
- `bipedleg_policy_capture_20260904` — **실기에서 HR_hip −12.72 확인**(클램프 98.2%). sim에서만 보던
  −12.6이 실기 폐루프에서 재현. **엔코더 영점 오차 가설 기각**(당일 측정 `zero_deg` hip은 3.2°뿐,
  보정해도 −0.220 → −0.164 rad인데 정책 요구는 −0.995 rad, soft limit −0.234 rad).

---

## D. 참고 — go2 계열(방법 개발 라인)

R.pet 직접 대상은 아니지만 위 방법들이 먼저 검증된 곳. `go2_imitation/`(MimicKit 정렬·tan-norm 인코딩·
latent imitation·motion VAE), `go2_pedipulation/`(발끝 목표추종, hip-scale 10x 채택),
`go2_parkour/`(SL-Grid vs GT voxel vs LiDAR 지형인지 비교 — **내 DTC/지형 트랙과 겹치는 유일한 지점**).

---

## E. 내 트랙과의 접점

| 내 자산 | isaac-6.0 대응 | 관계 |
|---|---|---|
| hind_leg RL 재균형 #4 (2026-07-28) | `hindLeg_history_direct` colmesh_v2_gpu2 baseline | **같은 로봇·다른 계보**. 저쪽은 URDF3_SignFix 플랜트 + PACE. 낙상 종료 비대칭·hip 포화는 내 쪽에도 있을 가능성 |
| `biped/emb` 실기 배포(ctypes+SHM) | `scripts/real2sim/r2s_biped_leg/` (C++ 브리지, TELEM/RELAX/GAIN) | **정면 중복**. 저쪽이 실기 운용 중이고 gear_k·zero_deg 함정을 이미 밟았다 |
| PACE sim2real ("하드웨어 확보 시") | `real2sim/_comparisons/pace_...` 39절 | **이미 실기 캡처로 진행 중**. 내 메모의 "실물 데이터 대기"는 갱신 필요 |
| DTC 17-DOF (지형·발판) | `go2_parkour/` 지형인지 비교 | 저쪽엔 R.pet 지형 트랙이 **없다** — 이산지형은 내 쪽이 앞서 있다 |

---

## F. 이주 결정 (2026-09-09) — isaac-6.0 라인 확정

**내 quad17/DTC 트랙을 isaac-6.0(IsaacLab 3.0.0)으로 이주한다.** main 라인(2.3.2)에 남지 않는다.
근거: AMP 자산이 1020커밋으로 훨씬 크고, R.pet AMP 데이터셋·실기 브리지가 전부 그쪽에 있다.

### F-1. R.pet USD = `LEG_CFG` (rga.py) — **repo에 없다**

```
usd_path = /home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Leg/Leg_gen/Leg.usd/Leg/Leg.usda
```

- 원본 `Robots/Leg/Leg/Leg.usda` 는 **물리 골격만 있고 visual/collision mesh 가 없어 지면을 통과**한다.
  그래서 `scripts/tools/convert_urdf.py` 로 **`Leg_URDF2` 를 재임포트한 자산(`Leg_gen`)** 을 쓴다.
- `source/isaaclab_assets/data/Robots/` 에 **`Leg/` 디렉토리 자체가 없다**(Go2Neck2, Hind_Leg,
  Hind_Leg_URDF3, Hind_Leg_URDF3_SignFix, MotionJig, R_Skeleton_Collision4,
  R_Skeleton_Hind_Leg_Fixed_Filpped 뿐). lgb 개인 머신에만 존재 → **확보 필요**.
- 구성: 다리 4 × (hip, thigh, calf, foot) + `FB_waist_joint` = 17-DOF. 총 **38.0 kg**.
  self-collision ON(convex-hull 기준이라 false-positive 가능), 영자세 = 발끝이 base 아래 0.50 m 좌우대칭 기립.

### F-2. ★ 회사 R.pet 과 내 quad17 은 **같은 로봇**이다

| 근거 | 회사 `LEG_CFG` | 내 `QUAD_17DOF_CFG` / MJCF |
|---|---|---|
| 관절명 | `.*_{hip,thigh,calf,foot}_joint` + `FB_waist_joint` | **완전 동일** |
| 총 질량 | 38.0 kg | 38.02 kg (`quad_real_17dof_waist_sphere.mjcf`) |
| URDF effort | 84 / 84 / 126 / 168 | 동일 |
| URDF velocity | 29.6 / 29.6 / 19.7 / 14.8 | 동일 |

같은 02_Leg URDF 계보다. 다른 것은 **임포트 경로와 게인·토크 값**뿐이다.

### F-3. ★★ 불일치 1건 — foot(발목) 토크. 회사가 stale 값을 쓰고 있다

| | hip | thigh | calf | **foot** |
|---|---|---|---|---|
| URDF `<limit effort>` | 84 | 84 | 126 | 168 |
| 회사 `LEG_CFG` (URDF × 85%) | 71.4 | 71.4 | 107.1 | **142.8** |
| 내 값 (재기어 실측) | 84 | 84 | 126 | **100.8** |

내 100.8 = **발목 8.4:1 재기어 실측값**(→ `02leg-motor-spec`). URDF 의 168 은 stale 이다.
즉 **회사 쪽은 실기의 1.42 배 발목 토크로 학습 중**이다. 이주 시 이 값을 반드시 가져가야 한다 —
아니면 sim 에서 되는 것이 실기에서 안 된다.

게인은 반대로 **저쪽이 더 원칙적**이다(`compute_leg_ieff.py` 로 USD 에서 I_eff 실측 → `Kd=2ζ√(Kp·I_tot)`,
접지관절 calf/foot 은 관성기반 Kp 가 스탠스 하중에 붕괴하므로 정적토크/처짐 0.25 rad 로 floor):
calf Kp **134**(내 12) · foot **59**(내 20). 채택 검토.

### F-4. ★ 관절 순서 — AMP 데이터셋이 이미 두 순서를 다 갖고 있다

`leg_imitation_tracking/README.md`:

| 폴더 | 순서 |
|---|---|
| `rpet_xmlorder/` | **MuJoCo XML** `[HL_hip,HL_thigh,HL_calf,HL_foot, HR_*, waist, FL_*, FR_*]` |
| `rpet_usd_order/` | Isaac USD `[waist, FL_hip..FL_foot, FR_*, HL_*, HR_*]` (학습 로더용) |

`rpet_xmlorder` 가 **내 MJCF 의 body 순서와 정확히 일치**한다(HL → HR → waist → FL → FR).
즉 데이터셋이 우리 MuJoCo 모델 계보에서 나왔다 — 관절 매핑 리스크가 낮다.

### F-5. USD 확보 옵션

| 옵션 | 내용 | 판단 |
|---|---|---|
| A. 내 `quad_17dof.usd` 사용 | GPU 서버 `/mnt/ssd1/jsh/RobotSW_IsaacLab/JSH/quad_17dof/` (로컬엔 없음, `JSH/` 엔 Hind_Leg 66M·Hind_Leg_Flat 66M 뿐) | 내 DTC 트랙에서 검증됨. **단 AMP 로더는 USD 관절순서를 기대** → 순서 확인 필요 |
| B. 회사 `Leg_gen` 확보 | lgb 머신에서 복사 | AMP 데이터셋(`rpet_usd_order`)과 정합 보장 |
| C. URDF 에서 재임포트 | `Leg_URDF2/urdf/Leg.urdf` → `convert_urdf.py` | URDF 원본도 repo 에 없음 |

**어느 쪽을 택하든 하나로 통일해야 한다.** 같은 로봇이라도 USD 가 다르면 관절 인덱스·링크명·
collision 이 달라 AMP 참조와 DTC 캐시가 서로 다른 좌표를 보게 된다.

---

## G. 이주 조사 결과 (2026-09-09)

**확정 사항**
- USD·관절순서는 **AMP 데이터셋 기준(옵션 B, 회사 `Leg_gen`)** 에 맞춘다.
- **발목 토크만 우리 값 `100.8` 을 쓴다**(사용자 확정). 회사의 142.8 = URDF stale 168 × 85%.
  USD 가 아니라 CFG 값이므로 데이터셋 정합을 깨지 않는다.

### G-1. 이름 규약 — 거의 그대로 맞는다

| | 회사 (`LEG_CFG` / `leg_imitation_tracking`) | 내 (`quad17`) | 판정 |
|---|---|---|---|
| root body | `Base` | `Base` | **동일** |
| 관절 | `{FL,FR,HL,HR}_{hip,thigh,calf,foot}_joint` + `FB_waist_joint` | 동일 | **동일** |
| 발 링크 | `{FL,FR,HL,HR}_foot_link` (`KEY_BODY_NAMES`) | `.*_foot_contact_link` | **★ 수정 필요 (패턴 1곳)** |

★ 내 `quad17_env.py` 는 **전부 이름 기반**(`find_bodies` / `find_joints`)으로 인덱스를 유도한다 —
발 위상 오프셋·nominal 발 오프셋·IK Jacobian 열까지 전부 body 이름의 다리 접두사에서 파싱한다.
**하드코딩된 관절 인덱스가 없다.** 따라서 USD 의 관절 순서가 달라도 코드가 자동 흡수하고,
실제 수정은 `.*_foot_contact_link` → `.*_foot_link` 패턴 하나다.

### G-2. IsaacLab 2.3.2 → 3.0.0 — **import 수준 갭 없음**

3.0.0 은 `lazy_export` + `.pyi` 스텁 구조라 `__init__.py` 본문에는 심볼이 안 보이지만 전부 해석된다.
내 quad17 이 쓰고 회사 env 는 안 쓰는 것들까지 전수 확인:

| API | 3.0.0 |
|---|---|
| `isaaclab.sensors` : `RayCaster`, `RayCasterCfg`, `patterns` (heightmap) | OK |
| `isaaclab.markers` : `VisualizationMarkers` | OK |
| `isaaclab.markers.config` : `BLUE_ARROW_X_MARKER_CFG`, `GREEN_ARROW_X_MARKER_CFG` | OK (`config.py` → `config/` 디렉토리로 바뀜) |
| `isaaclab.utils.noise` : `GaussianNoiseCfg`, `NoiseModelWithAdditiveBiasCfg` | OK |
| `isaaclab.utils` : `configclass` | OK (회사는 `isaaclab.utils.configclass` 경로를 쓴다 — 둘 다 유효) |
| `isaaclab.terrains.TerrainImporterCfg`, `isaaclab.envs.DirectRLEnv/Cfg` | OK |

### G-3. 물리 백엔드 — 바뀌지 않는다

3.0.0 에 Newton 이 들어왔지만 `SimulationCfg.use_newton_actuators = False` 가 기본이고,
회사 R.pet env 도 기본 `SimulationCfg`(dt 1/200, decimation 4 → 정책 50 Hz)를 쓴다. **PhysX 유지** —
내 DTC 결과와 백엔드가 같다.

### G-4. 남은 진짜 리스크 (import 이 아니라 거동)

1. ★ **발 형상.** 내 17-DOF 는 **sphere 발이 필수**였는데(→ `sim2real-checklist-17dof`) `Leg_gen` 은
   `Leg_URDF2` 를 `convert_urdf.py` 로 임포트한 것이라 발이 URDF mesh 다.
   **이산지형은 발 형상이 접촉을 지배한다** — stepping-stone/gap 에서 먼저 확인할 항목.
2. **self-collision false-positive.** `LEG_CFG` 주석이 직접 경고한다 — convex-hull 기준이라 실제 mesh 보다
   부풀 수 있어 초기 학습 관찰이 필요하다.
3. **게인 차이가 지형 거동에 미치는 영향.** 회사 calf Kp 134 / foot 59 vs 내 12 / 20. 평지 AMP 기준으로
   뽑힌 값이라 DTC 지형에서 재확인 필요.
4. **회사 env 의 종료 조건**: `termination_height 0.35`, tilt 판정은 `projected_gravity_b[2]`,
   그리고 `contact_force_threshold 500.0` 에 **"contact sensor 결함으로 현재 사실상 무력"** 이라는
   주석이 달려 있다 — hind_leg 의 낙상 종료 비대칭(§B)과 같은 계열 문제일 수 있다.

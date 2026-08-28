# go2_imitation_tracking — AMP → Latent-Space Imitation 전환 계획

작성 2026-08-27 · 대상 태스크 `Go2-Imitation-Tracking-v0` · 브랜치 `isaac-6.0`

---

## 0. 한 줄 판정

**착수 가능하다.** 초기 판정은 데이터 부족을 이유로 Red 였으나,
`/home/lgb/Dog_Motion_data_3D/mujoco_retarget_go2/smr_txt_retarget_dataset_go2_v0/` 에
**이미 Go2 로 리타게팅된 보행 데이터**가 있음이 확인되었다.
규모는 **세션 합집합 기준 49,613 프레임(13.78분) — 논문 1의 3.80배**이고,
2D 명령 커버리지는 프레임 기준 48/48 이다(단 세션 기준으로는 §4-3-b 의 제약이 있다).
현행 저장소 데이터셋(1,115 프레임)은 이 자산에서 **카테고리당 대표 1개씩만 뽑은 부분집합**이었다.

따라서 계획은 **논문 1 완전판(안 A)을 목표로 하되, 안 B 의 Phase 1~2 를 그 토대로 삼아
단계적으로 쌓는다.** 첫 작업은 데이터 확보가 아니라 **큐레이션·변환**이다.

**판정 변경 이력** — 이 문서는 하루 안에 두 번의 판정을 거쳤다:

| 시점 | 판정 | 근거 | 무효화 사유 |
|---|---|---|---|
| 1차 | Red, 안 B 권고 | 저장소 데이터 1,115 프레임 / 커버리지 41.7% | 데이터 탐색 범위가 **저장소 안으로 한정**되어 있었음 |
| 2차 | Green, 안 A 권고 | 외부 자산 187,312 프레임 / 커버리지 100% | 클립 수를 다양성으로 오독 (md5 기준) |
| **3차 (현재)** | **Green(얇음), 안 A 권고 유지** | 세션 합집합 **49,613 프레임 / 46 세션** | — |

**3차에서 하향 정정된 것**: 1,122 클립의 배후 원 녹화 세션은 **46개**뿐이고,
슬라이딩 윈도우와 미러로 부풀려져 있었다. 논문 1 대비 배율은 14.3배가 아니라 **3.80배**다.
여전히 여유는 있으나 **강한 후진(`v_x < -1.0`)은 실질적으로 데이터가 없고**,
고속 영역 7개 bin 은 세션 1~2개에만 의존한다.

## 1. 참조 논문 정리

### 1-1. P1 — Walk Like Dogs (arXiv 2507.00677, Kang et al., ETH Zürich)

Unitree **Go2** 대상. 3단 스택이다.

| 단계 | 내용 |
|---|---|
| 1 | kino-dynamic retargeting: 제약 IK → MJPC(iLQG, horizon 2.0s) 로 동역학 정합 |
| 2 | **hyperspherical(vMF) VAE** 로 상태전이 `(x[t-1], x[t])` 를 latent 18-D에 임베딩 |
| 3-a | motion synthesis 정책: latent 를 항해해 command 를 따르는 **참조 모션을 생성** |
| 3-b | motion tracking 정책: 생성된 참조를 **residual 관절각**으로 추종 |

VAE 상세 (원문 PDF 표 대조 완료):

| 항목 | 값 |
|---|---|
| latent 차원 | 18 (vMF, 구 표면) |
| encoder | FC 2층 × 256, ELU |
| decoder | Mixture of Experts, expert 6개 (각 FC 2층×256) + gating FC 2층×64 |
| 손실 가중치 | `beta = 0.05` |
| 학습 일정 | 상태전이로 20 epoch → autoregressive 로 60 epoch 전환 |

```
L_vae = || x[t] - x_hat[t] ||^2  +  beta * KL( q(z | x[t-1], x[t]) || p_vMF(z) )
```

synthesis 정책 보상:

```
r_syn = exp( -(v_fwd - c_fwd)^2 / 0.25  -  (yaw_rate - c_turn)^2 / 0.1 )
```

tracking 정책:

```
obs      = [ o_state[t-4..t] , o_ref[t-4..t] , z[t] ]      # (42 + 26) * 5 + 18 = 358
theta_pd = theta_ref + 0.15 * a                            # a in R^12
r_track  = 1.0 * r_imitation + 0.5 * r_world + 0.1 * r_reg
```

기호 정의 — 상태/명령:

| 기호 | 의미 | 단위 |
|---|---|---|
| `x[t]` | VAE 입력 상태 (base 높이·6D 회전·선속도·각속도·발위치·관절각·관절속도) | 혼합 |
| `z[t]` | latent 벡터, 구 표면에 정규화 `z = z_tilde / ||z_tilde||` | 무차원 |
| `v_fwd` | 지면투영 전진속도 | m/s |
| `yaw_rate` | base yaw 각속도 | rad/s |
| `c_fwd`, `c_turn` | 사용자 명령 | m/s, rad/s |
| `theta_ref` | 참조 모션 관절각 | rad |
| `a` | 정책 출력 residual | rad (스케일 0.15) |

> 주의: 논문 표의 `KL target 0.01` 은 VAE 항이 아니라 **PPO adaptive-LR 의 KL target** 이다.
> 웹 요약이 이 둘을 혼동하기 쉬우니 구현 시 `beta = 0.05` 를 쓸 것.

### 1-2. P2 — APT-RL (arXiv 2607.13579, Kang et al., KAIST)

trajectory optimization 으로 2D 모션 18만 궤적(5,560,286 스텝, 약 15.5시간)을 생성 →
**Transformer VAE** 로 latent 16-D 임베딩 → gait 별 **torque decoder** → PPO 정책이
`a_latent(16) + a_aux(12) + gait logit(1)` 을 출력.

```
tau = tau_dec(z)  +  kp * ( q_default - q + 0.2 * a_aux )  -  kd * q_dot
kp = 80,  kd = 2
```

**이 저장소에서는 실격이다.** 근거 두 가지:

1. **토크 레이블이 없다.** P2의 decoder는 `z → tau` 를 지도학습하는데, 이는 trajectory
   optimization 이 상태와 함께 토크를 뱉기 때문에 가능했다. `smr_mirror_pkl` 은
   `frames(N,18) = root_pos(3) + root_euler(3) + joint_pos(12)` 뿐인 순수 kinematic
   mocap 이고, 속도·발위치조차 파생값이다. 토크는 복원 불가.
2. **데이터 규모가 4자릿수 다르다.** P2는 15.5시간, 이 저장소는 37.2초.
3. (부수) torque decoder 직접 구동은 이 저장소의 effort-limit / `ImplicitActuator`
   연구선(`project_go2_effort_limit_blocks_high_speed`,
   `project_go2_implicit_actuator_generation`)과 정면충돌한다.

P2에서 **가져올 만한 것은 아이디어 두 개**다: (a) 부족한 expert 데이터를 궤적최적화로
**합성**한다는 발상, (b) latent action 에 약한 KL 정규화(2.5e-6)를 걸어 OOD 를 억제.

---

## 2. 선행연구 지형 — 무엇을 latent 로 삼는가

| 방법 | latent 대상 | latent 종류 | 판별기 | 4족보행 실증 |
|---|---|---|---|---|
| MVAE (Ling 2020) | 상태전이 | Gaussian | 없음 | ✗ |
| Won et al. (CVAE 2022) | 모션 시퀀스 | Gaussian | 없음 | ✗ |
| ASE (2022) / CALM (2023) | 스킬 | 구 표면 연속 | **있음(AMP 확장)** | ✗ |
| NCP (2023) | 모션 클립 | **이산 VQ codebook** | 있음 | ✗ |
| PULSE (2024) | tracking 정책 자체 | Gaussian, proprio-조건부 prior | 없음 | ✗ |
| **P1 Walk Like Dogs (2025)** | 상태전이 | **vMF 18-D** | 없음 | **✓ Go2** |
| **LatentMimic (2026, 2604.16440)** | 모션 윈도우 | Gaussian, MLP [256,128] | **없음 — explicit KL** | **✓ Go1** |
| SLMP (2026, 2603.01294) | tracking 정책 | spherical | 있음(보조) | ✗ humanoid |

### LatentMimic 이 이 프로젝트에 가장 중요한 이유

Unitree **Go1**, Isaac Gym, PPO, 개 mocap(Zhang 2018) — 플랫폼·스택·데이터 성격이
현재 저장소와 사실상 동일하고, **AMP 를 명시적으로 대체**한다.

핵심은 KL 을 chain rule 로 쪼개고 **latent marginal 항만 남기는** 것이다:

```
KL( P_ref(X) || P_pi(X) )  =  KL( P_ref(Z) || P_pi(Z) )                 <- 이것만 사용
                            + E_Z[ KL( P_ref(X|Z) || P_pi(X|Z) ) ]      <- 버림
```

| 항 | 의미 | 왜 이렇게 |
|---|---|---|
| marginal latent 항 | gait 의 위상구조·리듬 | 스타일은 반드시 지켜야 함 |
| conditional 기하 항 | 발 궤적의 구체적 형상 | 계단에서 발을 더 드는 등 **적응을 벌하면 안 됨** |

보상:

```
r_mimic  = exp( -0.01 * KL( z_target || z_sim ) )
r_total  = r_mimic + r_task + r_anchor
r_anchor = exp( w_anchor * KL( pi_theta(a|o) || pi_style(a|o) ) )
```

즉 **AMP discriminator 를 latent KL 로 바꾸는 것만으로** 사용자가 요구한
"expert 를 latent 에 임베딩 → 그 공간에서 명령 추종" 이 성립한다.
env 구조를 갈아엎지 않아도 된다는 뜻이다.

---

## 3. 코드베이스 seam (전수 조사 결과)

### 3-1. 그대로 재사용되는 것

| 자산 | 위치 | 비고 |
|---|---|---|
| `motion_lib` 7-tuple API | `motion_lib.py:342-371` | `x_vae` 49-D **전 필드 구성 가능** |
| AMP obs 계산기 | `env.py:1119-1152` (`_compute_amp_obs`, jit) | 43/49-D, **그대로 VAE encoder 입력**으로 전용 가능 |
| heading-relative 6D 회전 | `env.py:1155-1197` | 재사용 |
| `EmpiricalNormalization` | `rsl_rl/networks/normalization.py` | VAE 입력 정규화 |
| `Estimator` MLP 골격 | `rsl_rl/modules/estimator.py:13-37` | encoder/decoder 뼈대 |
| RMA/estimator 전체 | `ppo_parkour.py:528-539` 등 | `PPOAMP` 가 `PPOParkour` 상속이므로 그대로 승계 |
| history 인코더 | `actor_critic_parkour.py:23-78` | 불변 |

`x_vae` 49-D 구성 가능성 — 필드 단위 판정:

| 필드 | 출처 | 판정 |
|---|---|---|
| base height | `root_pos[:,2]` | 직접 |
| 6D orientation | `root_quat`(wxyz) → rotmat 앞 2열 | 변환 필요 |
| base lin/ang vel | `root_lin_vel`, `root_ang_vel` (body frame) | 파생(finite-diff) |
| foot position (4×3) | `_go2_fk_foot_pos` 해석적 FK | 파생, 오차 <0.5 mm |
| joint pos / vel (12+12) | `dof_pos`, `dof_vel` | 직접 / 파생 |

### 3-2. 교체 대상

| 위치 | 현재 | 전환 후 |
|---|---|---|
| `env.py:192-231, 338-384` | amp_observation_buffer, `extras["amp_obs"]` | latent obs 버퍼 + `extras["style_obs"]` |
| `env.py:1026-1060` | `collect_reference_motions`, `get_amp_observations` | expert 배치 샘플러 (시그니처 보존 권장) |
| `rsl_rl/modules/amp_discriminator.py` | `AMPDiscriminator` | 신규 `MotionEncoder` / `MotionVAE` |
| `rsl_rl/algorithms/ppo_amp.py:242-310, 341-442` | discriminator 생성·`update_amp` | 신규 알고리즘 파일 (**병렬 신설, 기존 파일 수정 금지**) |
| `rsl_rl/runners/on_policy_runner_amp.py:118-262` | disc reward·replay·anneal | 신규 러너 (**가장 큰 교체 지점**) |
| `agents/rsl_rl_ppo_cfg.py:41-134` | `class_name="OnPolicyRunnerAMP"`, `amp:{...}` | 신규 러너 이름 + `latent:{...}` |
| `train.py` (227-280 부근) | class_name 분기 | 분기 1개 추가 |

### 3-3. 구조 변경이 필요한 곳 (단순 seam 아님)

**액션 경로.** 현재는 `env.py:290` 에서

```
target = 0.25 * a + default_joint_pos
```

P1식 residual (`theta_ref + 0.15 * a`) 로 가려면 **매 스텝의 참조 관절각**이 필요한데,
현재 env 는 RSI 시점에만 모션을 참조하고 **이후 phase 를 추적·저장하지 않는다**.
따라서 모션 phase 상태(클립 id + 시간)를 새로 도입하고 매 스텝 갱신해야 한다.
이것이 P1 완전판의 실질 비용이다.

### 3-4. 저장소 금지사항 (조용히 위반하면 안 됨)

- `motion_lib.py` **절대 수정 금지** (태스크 CLAUDE.md, byte-identical 유지 요구)
- AMP observation 490-D **불변**
- `observation_space` 는 cfg 상수가 아니라 **env `__init__` 이 `super().__init__` 전에** 재계산 (`env.py:114-123`)
- `_apply_obs_dr` 인덱스 **하드코딩 금지** — `self._obs_idx_*` 에서 유도
  (과거 ang_vel 제거로 슬라이스가 밀려 σ=0.2 노이즈가 σ=0.05 자리에 들어간 4배 증폭 사고 이력)
- obs idx 30–41 `actions` 블록은 리터럴 0 **dead channel** — 새 레이아웃이 승계하지 말 것
- resume 은 argparse 플래그만 작동 (`--resume --load_run --checkpoint`), hydra 는 조용히 무시
- 학습은 `setsid` 로 띄울 것 (세션 종료 시 SIGHUP 동반 사망 이력)
- 학습에 `--video` 붙이지 말 것 — 영상은 체크포인트에서 별도 렌더

**따라서 in-place 개조가 아니라 새 태스크 디렉터리를 만든다:**
`go2_imitation_latent/`, Task ID `Go2-Imitation-Latent-v0`, `motion_lib.py` 자체 사본(yaw 수정 포함).
기존 `Go2-Imitation-Tracking-v0` 은 **동시 진행 baseline arm** 으로 살려 둔다.

---

## 4. 데이터 — 무엇이 있고 무엇을 쓸 것인가

### 4-1. 현행 저장소 데이터셋 (사용 중, 대체 예정)

`source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation/imitation/smr_mirror_pkl/`

| 항목 | 값 |
|---|---|
| 파일 | 18개 (9 base + 9 mirror) |
| **중복** | `go2_trot1.pkl` 이 `go2_trot0.pkl` 과 md5 동일 |
| 고유 클립 / 프레임 | **8개 / 1,115** (18.6 s @60 fps) |
| `v_x` | min -0.001 / mean 1.38 / max 4.94 — **후진 전무** |
| `yaw_rate` (unwrap) | -0.95 ~ 2.24 — 고yaw 는 `go2_walk_turn` 단 한 클립 |
| 2D 커버리지 (8×6 = 48 bin) | **20/48 = 41.7%** |
| NaN / 프레임드롭 | 없음 |

### 4-2. ★ 발견 — 이미 리타게팅된 대규모 자산이 존재한다

`/home/lgb/Dog_Motion_data_3D/mujoco_retarget_go2/smr_txt_retarget_dataset_go2_v0/`

| 항목 | 값 |
|---|---|
| 전체 | 38 카테고리 / 3,524 파일 / 706,916 프레임 (196.4분) |
| 보행 관련 (22 카테고리) | 1,430 파일 |
| 포맷 | 61열 DeepMimic JSON `.txt`, `[7:19]` 가 joint_dof 12개 |
| 변환 | `convert_smr_to_pkl.py` **기존재** → `frames(N,18)`. 비용 거의 0 |
| NaN / 파싱실패 / 관절한계초과 | 각 0건 (샘플 65K 관절-프레임) |

**관절 순서 검증 완료** (조용한 오학습 방지):
`smr_go2/new_dataset/walk.txt` → 변환 → `smr_go2_pkl/walk.pkl` 이 **diff = 0.0** 으로 일치하고,
저장소 `smr_mirror_pkl/go2_walk.pkl` 과도 max diff 0.047 로 일치.
→ 전 구간 **동일 관절 컨벤션, 리매핑 불필요**.

**계보** — 현행 8클립은 이 자산의 부분집합이다:

```
smr_txt_retarget_dataset_go2_v0   (수천 raw 클립)
  -> smr_go2/new_dataset          (카테고리 대표 15개만 큐레이션)
    -> smr_go2_pkl
      -> 저장소 smr_mirror_pkl    (9개)
```

`walk/` 폴더에만 클립이 744개 있는데 저장소는 그중 **1개**만 쓰고 있었다.

### 4-3. ★★ 규모를 셀 때 클립 수를 믿으면 3~4배 과장된다

파일명이 `[M_]D<n>_<seq>_<subj>_<take>_F<start>_F<end>_<idx>` 스키마다(파싱 실패 0건).
파싱해 보면 **클립 수는 다양성이 아니다**:

| 지표 | 값 |
|---|---|
| md5 dedup 후 고유 클립 | 1,122 |
| 그중 미러(`M_`) 제외 | **733** |
| **배후 원 녹화 세션 수** | **46개** |
| md5 기준 총 프레임 | 187,312 |
| **F구간 합집합 고유 프레임 (진짜 상한)** | **49,613 (13.78분)** |

> **[Phase 0 과의 수치 차이]** 위 49,613 은 **보행 관련 22개 카테고리 전체, 봉투 필터 없음** 기준이다.
> Phase 0 이 실제로 채택한 데이터셋은 **15개 카테고리(`spin` 계열 제외) + 봉투 필터** 를 거쳐
> **39,813 프레임 / 43 세션 / 논문 대비 3.04배** 다. 둘 다 맞고 범위가 다르다.

같은 세션(`D1_007_KAN01_001`)에 `F286_F325`, `F286_F343`, `F305_F325` … 식으로
**슬라이딩 윈도우가 촘촘히 겹쳐** 있다. md5 는 서로 다르지만 프레임 대부분이 동일하다.
미러(35.9%)도 좌우대칭 복제라 정보량 기여는 0으로 본다.

**따라서 배율은 다음과 같이 정정한다:**

| 기준 | 정정 전 (md5) | **정정 후 (세션 합집합)** |
|---|---|---|
| 논문 1 (13,076) 대비 | 14.3배 | **3.80배** |
| Phase 0 목표 (10,000) 대비 | 18.7배 | **4.96배** |

**규약: 이 문서와 이후 보고서는 프레임 수를 항상 `(md5-dedup 수 / 세션-합집합 수)` 로 병기한다.**
하나만 쓰면 3~4배 과장이 된다.

### 4-3-b. 커버리지 — 프레임 기준과 세션 기준이 다르다

| | 고유 프레임 | 2D bin 커버리지 |
|---|---|---|
| 현행 저장소 | 1,115 | 20/48 = 41.7% |
| 논문 1 | 13,076 | — |
| 신규 자산 (프레임 기준) | 187,312 / **49,613** | 48/48 = 100% |
| **신규 자산 (세션 기준)** | — | **아래 참조** |

프레임 기준 48/48 은 **세션 기준으로 보면 무너진다**:

| 영역 | 기여 세션 수 | 판정 |
|---|---|---|
| `v_x ∈ [0, 1.5)` × `wz ∈ [-1.0, 1.0)` (저·중속) | **30개 이상** | 튼튼함 |
| 고속 영역 위주 7개 bin | **1~2개** | 얇음 — 클립 암기 위험 |
| **`v_x < -1.0` (강한 후진) 4개 bin** | **0개** | **실질적으로 데이터 없음** |

강한 후진 bin 이 프레임 기준으로 채워져 보였던 것은
**어느 세션의 경계 프레임 2개가 우연히 걸친 것**이었다.
`v_x < -1.0` 명령은 데이터 공백이므로 **명령 범위에서 제외하거나 합성해야 한다.**

### 4-3-c. `csv_dataset_locomotion` 은 추가 확보처가 아니다

`/home/lgb/Dog_Motion_data_3D/csv_dataset_locomotion/` (raw 개 mocap, 1,676 클립, 2.7G)을
독립적으로 분석한 결과 **세션 수 46개, 카테고리별 내역(walk 34 / run 12 / trot 6 / spin 3)이
리타게팅 자산과 정확히 일치**했다. → **동일 원본에서 파생된 같은 데이터**다.

| 함의 | 내용 |
|---|---|
| 리타게팅해도 다양성이 늘지 않는다 | 같은 46세션이다. 프레임만 늘고 세션은 그대로 |
| 리타게팅 비용은 낮다 (참고) | `stmr_go2.py` → `convert_smr_to_pkl.py`, 새 코드 불필요. 고유 838클립 기준 단일 프로세스 1.1시간, 8코어 8~17분 |
| **진짜 예비 확보처는 `csv_dataset/`** | 7.5G, DogML 전체(8,048 클립 / 8 동작 클래스). 여기에 **다른 세션**이 있는지가 관건 |

즉 "데이터를 더 붓는" 처방은 무효다. 필요한 것은 **다른 세션**이다.


### 4-4. ★ 새 데이터가 만든 새 문제 — 명령 범위를 넘어선다

이전에는 데이터가 명령보다 **좁았지만**, 이제는 **넓다**. 종류가 다른 문제다.

| | 데이터 범위 | 현행 명령 범위 |
|---|---|---|
| `v_x` | ~ 5.89 m/s | [-1, 3] m/s |
| `\|wz\|` | ~ 6.30 rad/s | [-1.5, 1.5] rad/s |

명령으로 도달할 수 없는 영역의 expert 를 그대로 두면 VAE 가
**synthesis 정책이 결코 보상받지 못할 latent 영역에 표현 용량을 쓴다.**
특히 `|wz|` 6.30 rad/s 는 로봇의 추종 가능 범위를 넘는다.

| 선택지 | 내용 | 함의 |
|---|---|---|
| (a) 데이터를 명령 봉투로 클리핑 | `v_x <= 3`, `\|wz\| <= 1.5` 만 사용 | latent 조밀. 손실 프레임 수 측정 필요 |
| (b) 명령 범위 확대 | `v_x ∈ [-1, 4]`, `\|wz\| <= 2.5` | 데이터를 살리나 AMP baseline 과 비교조건 달라짐 |
| **(c) 절충 — 권고** | `v_x` 는 확대, `\|wz\|` 는 클리핑 | 고속은 이 저장소의 오랜 목표이고 데이터가 지원함. 고yaw 는 추종 불가 |

**추가 제약 (§4-3-b).** 데이터가 명령보다 넓은 쪽만 문제가 아니다.
**강한 후진 `v_x < -1.0` 은 기여 세션이 0개**이므로 명령 하한을 `v_x >= -1.0` 이하로
내리면 안 된다. 현행 하한 -1.0 은 경계선이며, 그 근방도 세션이 얇다.
권고 명령 범위: **`v_x ∈ [-0.5, 4.0]`, `\|wz\| <= 1.5`**.

### 4-5. 커버리지는 프레임 단위지 세션 단위가 아니다 (§4-3-b 에서 실측됨)

48/48 은 "모든 bin 에 프레임이 있다" 는 뜻이지
"각 bin 에 서로 다른 클립이 충분하다" 는 뜻이 아니다.
클립 정체성 암기를 배제하려면 **bin 별 고유 클립 수**를 세야 한다.
→ Phase 1 의 G1/G2 게이트(§6)가 여기서 결정적이며,
클립이 8개에서 1,120개로 늘어난 지금 그 게이트는 **비로소 통계적으로 유의미해진다**.

### 4-6. 제외 대상과 사용 순서

| 제외 | 사유 |
|---|---|
| `jump*` 카테고리 | root z 최대 3.52 m — 순수 보행과 이질적 |
| md5 중복 | dedup |
| **`spin` 계열** | **Phase 0 실측으로 확정** — 봉투 내부비율 중앙값 0.03 (§Phase 0) |

> **[정정]** 초기 조사에서 `lie_walk` 의 한 클립이 `\|wz\| = 16.16 rad/s` 로 보고되었으나,
> 그 값은 **결함 있던 euler 경로로 계산된 것**이다. 수정 후 전 데이터셋 최대는 **7.64 rad/s** 다.
> 16.16 인 클립은 존재하지 않으므로 찾지 말 것.

**사용 순서**: `walk`(158K 프레임, 후진 포함) → `run`(고속) → `trot`(표준) → `spin`(고yaw 전담).
이 4개만으로 목표를 초과한다.
`stand_walk` / `sit_walk` / `lie_walk` 는 전이구간 트리밍 후 **mode 간 transition 학습용 보강재**로 —
논문 1의 핵심 주장이 바로 mode transition 이므로 성급히 버리지 않는다.


### 4-7. ★★★ motion_lib 포팅 회귀 — 파생값을 재계산하지 말 것

Phase 0-1(yaw unwrap 수정)을 하러 들어갔다가 **더 근본적인 문제**를 확정했다.

#### 확정된 사실

`motion_lib.py` 가 pkl `frames[:,3:6]` 을 **exponential map 인데 ZYX euler 로 읽는다.**

| 대조 | 결과 |
|---|---|
| 저장소 pkl **9/9** vs 원본 `.txt` (위치오차 `0.00e+00` 으로 매칭) | exp map 오차 **1e-17~1e-16**, euler 오차 4.2e-3 ~ **1.5e-1** |
| MimicKit 원본 `motion_lib.py:110` | `exp_map_to_quat(root_rot)` — euler 사용처 **0건** |
| MimicKit `:175-176` 각속도 | `quat_diff(q[t],q[t+1])` → `fps * quat_to_exp_map(dq)` |
| MimicKit `torch_util.py:75` | `quat_pos` 가 `w<0` 부호반전 → **unwrap 문제가 원천적으로 없음** |

**yaw wrap 결함은 독립 버그가 아니라 euler 전환의 파생이다.**
쿼터니언에는 ±π 경계가 없으므로, 원본 방식이었다면 애초에 생기지 않았다.

| 위치 | 현행 | 올바름 |
|---|---|---|
| `:18, 437` | `[3:6]` = `root_euler` | `root_rot` (exp map) |
| `:440` | `_euler_to_quat_wxyz` | `exp_map_to_quat` |
| `:442, 446` | euler finite-diff + 야코비안 | 쿼터니언 차분 |

#### ★ 그런데 고칠 필요조차 없다 — 정답이 이미 데이터에 있다

`convert_smr_to_pkl.py` 가 `.txt` **61열 중 19열만 남기고 42열을 버린다.**
버려지는 것이 정확히 우리가 필요한 값들이다:

| 열 | 내용 | 현재 처리 |
|---|---|---|
| `[19:31]` | `toes_local` — **MuJoCo FK 실측** 발 위치 | 버림 → 해석적 FK 근사로 대체 |
| `[31:34]` | `lin_vel_local` (body frame) | 버림 → 재계산 |
| `[34:37]` | `ang_vel_local` — `stmr_go2.py:551-564` 가 **쿼터니언 차분**으로 계산 | 버림 → euler-rate 로 재계산 |
| `[37:49]` | `joint_vel` | 버림 → 재계산 |
| `[49:61]` | `toe_vel_local` | 버림 |

`stmr_go2.py` 의 각속도 계산은 **MimicKit 과 동일한 알고리즘**이다.

실측 (`go2_walk_turn`, 207 프레임):

| | `\|wz\|` max | `\|wz\|` mean |
|---|---|---|
| **txt 원본** | **2.238** | 0.953 |
| motion_lib 재계산 | **373.312** | 2.749 |

`ang_vel` 최대오차 **374.08 rad/s**, `lin_vel` 최대오차 **0.0894 m/s**.
`\|wz\|>10` 프레임은 원본 0개 / motion_lib 1개 — 각속도 불일치는 **wrap 한 프레임이 지배**하고,
`lin_vel` 오차 0.089 m/s 가 회전 표현 오독분이다.

#### 처방 — Phase 0 의 성격이 바뀐다

**"motion_lib 의 재계산을 고친다" 가 아니라 "재계산을 하지 않는다".**

| | 기존 계획 | **수정** |
|---|---|---|
| 회전 | euler → exp map 수정 | `.txt` 쿼터니언 직접 사용 |
| 각속도 | 쿼터니언 차분으로 재구현 | `.txt` `[34:37]` 그대로 |
| 선속도 | 유지 | `.txt` `[31:34]` 그대로 |
| 발 위치 | 해석적 FK 유지 (오차 <0.5mm) | `.txt` `[19:31]` **MuJoCo FK 실측** |

파생값을 다시 만들지 않으므로 결함이 생길 자리가 없고,
`x_vae` 49-D 가 전부 리타게팅 파이프라인 산출물로 채워진다.

#### 현행 AMP 에 대한 영향

| 경로 | 위치 | 오염 |
|---|---|---|
| RSI 스폰 | `env.py:871-886` | 초기 자세·초기 각속도 |
| Discriminator expert 샘플 | `env.py:994` → `collect_reference_motions` → `get_amp_observations` | 49-D 중 **12개**(`ang_vel` 3 + `rot_tan_norm` 6 + `lin_vel` 3) |

나머지 37차원(`dof_pos`/`dof_vel`/`root_height`/`foot_pos`)은 회전과 무관해 영향 없다.
현행 데이터는 직진 위주라 자세 오차가 8/9 클립에서 <2.8° 이지만,
`go2_walk_turn`(고유 프레임의 18.7%)에서 12.08° 이고 `wz` 는 167배다.
`EmpiricalNormalization` 이 expert+policy 를 합쳐 std 를 fit 하므로
**wz 채널의 std 가 팽창해 discriminator 의 yaw 감각이 사실상 무력화**된다.
1/49 차원이고 yaw 축이라 이 저장소가 쫓아온 속도 천장의 원인일 가능성은 낮다.

**`spin` 제외 여부에 대한 함의**: 수정 후에는 회전이 큰 데이터도 원본과 동일한 정확도로
처리되므로, **결함을 피하려고 `spin` 을 뺄 이유는 사라진다.**


## 4-B. OOD 를 어떻게 볼 것인가 (설계 선택의 실질 축)

"데이터 커버리지가 41.7% 다" 는 곧 "명령 OOD 를 어떻게 다룰 것인가" 의 문제다.
**OOD 를 두 종류로 분리하지 않으면 결론이 뒤집힌다.**

| 종류 | 정의 | 발생 조건 |
|---|---|---|
| **명령 OOD** | 데이터에 없는 명령 (`v_x < 0`, 고속×고yaw) | 사용자가 명령 범위를 데이터보다 넓게 잡을 때 |
| **latent OOD** | 정책이 학습 안 된 latent 영역을 방문 | 위 명령 OOD 가 latent 로 전파될 때 |

### 안 A — 명령 OOD 가 latent OOD 로 **직결**된다 (기전은 유효, 판정은 무효화됨)

참조 모션을 **decoder 가 생성**하므로 경로가 이어진다:

```
명령 OOD  ->  synthesis 정책이 미학습 latent 영역으로 이동
          ->  decoder 가 무의미한 참조 상태를 출력
          ->  tracking 정책이 그 참조를 충실히 추종하려다 붕괴
```

논문 1의 Fig. 6 **회색 셀("적절한 gait mode 로 전환 실패")이 이 현상**이다.

**단, 이것이 "안 A 는 벽" 이라는 판정의 근거가 되지는 않는다.**
그 판정은 커버리지 41.7% 를 전제로 했는데, §4-2 의 자산에서 커버리지는 **48/48 = 100%** 다.
회색 셀은 **더 이상 예상 운용점이 아니라 예외 상황**이다.
기전 자체는 유효하므로, §4-4 의 명령 봉투 결정과 §6 Phase 1 의 G1/G2 게이트로 관리한다.


### 안 B — 소프트 열화로 끝난다

인코더는 **스타일 점수를 매기는 척도**일 뿐 행동을 생성하지 않는다.
명령 OOD 가 오면 `r_style` 이 약해지거나 무의미해질 뿐,
정책은 `r_task` 를 따라 자유롭게 데이터 밖으로 나간다.
성능이 절벽에서 떨어지는 것이 아니라 **스타일이 서서히 풀린다.**

### 이 저장소의 실증 — 이미 측정된 사실

현행 AMP 가 바로 안 B 구조이고, 다음이 이미 관측되었다:

| 관측 | 함의 |
|---|---|
| 참조에 없는 고속 명령에서 달성률 97~98% (`task_reward_lerp` 0.6~0.8) | **명령 OOD 외삽은 실제로 일어난다** |
| 그때의 보행이 참조에 없는 4 Hz trot (참조 run 은 2.54 Hz bound) | **스타일은 대가로 무너진다** |
| AMP 를 완전히 빼면 더 빠르지만 34 Hz 진동 포복 → 실기 불가 | **스타일 신호가 유일한 제동장치** |

### 결론

- "데이터 밖 명령이면 로봇이 못 움직인다" 는 **거짓**이다 (안 B 기준).
- "데이터 밖 명령에서 expert 스타일을 유지한다" 는 **불가능**하다.
- 커버리지가 100% 가 된 지금, 남은 OOD 위험은 **명령 봉투 밖**(§4-4)과
  **bin 별 클립 다양성 부족**(§4-5)의 두 가지로 좁혀졌다.


### 완화책 (문헌)

| 기법 | 출처 | 내용 |
|---|---|---|
| `r_anchor` | LatentMimic | 스타일 정책과의 KL 로 정책 drift 억제 |
| proprio-조건부 prior | PULSE | 고정 prior 대신 상태의존 prior 로 도달 가능 latent 영역을 좁힘 |
| conditional 기하 항 폐기 | LatentMimic | **애초에 지형 적응 같은 OOD 변형을 허용하려는 설계** |
| 약한 latent KL 정규화 (2.5e-6) | APT-RL | latent action 이 사전학습 클러스터에서 크게 벗어나지 않게 |
| Latent Policy Barrier | arXiv 2508.05941 | expert latent 임베딩을 in-distribution 경계로 사용 |

**이것이 §5 에서 안 B 를 1순위로 두는 세 번째 근거다** — 데이터 커버리지가 41.7% 인 상황에서
OOD 를 하드 실패가 아닌 소프트 열화로 만드는 설계는 안 B 뿐이다.

---

## 5. 설계 3안 — 재평가

초기 권고는 안 B 였고, 그 근거는 세 다리였다. 새 데이터로 각각을 재검증한다:

| 안 B 의 근거 | 187,312 프레임에서 |
|---|---|
| "37초로는 생성모델 학습 불가" | **소멸** — 논문 1의 3.80배 (세션 합집합 49,613 프레임) |
| "커버리지 41.7% 라 OOD 가 하드 실패" | **대부분 소멸** — 프레임 기준 48/48. 단 강한 후진 4 bin 은 세션 0개 |
| "롤백이 싸다" | **유효** — 그러나 이제 **유일하게 남은** 근거 |

근거가 하나뿐이면 그건 판정이 아니라 선호다. 따라서 다시 묻는다:
**안 A 의 추가 기계장치가 안 B 로는 못 얻는 무언가를 주는가?**

**준다. 그리고 그것이 정확히 요청받은 것이다.**
요청은 "embedding space 에서 **명령어에 따라** task 를 tracking" 이었다.
안 B 의 인코더는 **스타일 채점기**일 뿐이고, 명령 조건화는 여전히 task reward 에 남는다.
latent 는 정책이 **조종하는 대상이 되지 않는다.**
안 A 의 synthesis 정책은 문자 그대로 "명령에 따라 latent 공간을 항해" 한다.
**안 A 가 요청의 충실한 구현이고, 안 B 는 더 싼 인접물이다.**

| | **A. 논문 1 완전판 (권고)** | B. 흡수형 | C. VQ-VAE 이산 |
|---|---|---|---|
| latent | vMF 18-D, MoE decoder 6 experts | Gaussian encoder (LatentMimic식) | 이산 codebook |
| 단계 수 | 3단 (VAE → synthesis → tracking) | 1단 | 2단 |
| 액션 | `theta_ref + 0.15*a` | 현행 유지 | latent 직접 |
| **latent 가 명령에 따라 조종되는가** | **예 — 요청의 직역** | 아니오 (스타일 채점만) | 부분적 |
| env 변경량 | **대** (모션 phase 추적, obs 358-D 재설계) | 소 | 중 |
| 49,613 프레임(세션합집합)에서 | **성립** (논문의 3.80배) | 성립 | overfit 위험은 해소 |
| 4족보행 실증 | 논문 1 (Go2) | LatentMimic (Go1) | 없음 |
| 연속 속도명령 궁합 | 좋음 | 좋음 | **나쁨** (코드 경계 불연속) |

### 결정을 가르는 것은 이제 데이터가 아니라 §3-3 의 구조 비용이다

안 A 는 **매 스텝 모션 phase 추적**(`motion_id`, `motion_time`)과 obs 재설계,
reward 교체가 필요하다. 이 비용은 데이터가 늘어도 줄지 않는다.
**이건 블로커가 아니라 받아들이거나 거절할 비용이다** — 판단은 사용자 몫이다.

### 권고: A, 단 B 의 Phase 1~2 를 토대로 쌓는다

포크가 아니라 **누적**이다:

```
Phase 1 (인코더 학습)      -> 안 A 가 필요로 하는 바로 그 인코더
Phase 2 (latent style RL)  -> AMP 제거 검증, 롤백 지점
Phase 3 (command->latent)  -> synthesis 정책 = 요청의 핵심
Phase 4 (residual track)   -> 논문 1 완전판
```

Phase 1~2 의 산출물이 Phase 3~4 에서 **버려지지 않는다.**
이렇게 하면 안 B 의 실질적 미덕이었던 **싼 롤백**을 유지한 채 안 A 에 도달한다.
Phase 2 에서 멈춰도 동작하는 시스템이 남고, Phase 3 에서 멈춰도 마찬가지다.

**C 는 배제.** 연속 `m/s` 명령 추종이라는 이 태스크의 핵심 목적과 구조적으로 마찰한다.


## 6. 단계별 구현 계획

### Phase 0 — **완료 (2026-08-27)**

산출물과 실측은 두 보고서에 있다:

| 보고서 | 내용 |
|---|---|
| `../2026-08-27_motion_lib_rotation_fix/` | motion_lib 결함 2건 수정 + V1~V4 검증 (PASS) |
| `../2026-08-27_phase0_dataset_build/` | expert 데이터셋 구축 754 pkl + 커버리지·세션 통계 |

**결과 요약**

| 지표 | 값 |
|---|---|
| 유지 클립 | 754 (미러 제외 476) |
| 프레임 | md5-dedup **140,130** / 세션합집합 **39,813** (11.06분) |
| 원 녹화 세션 | 43 |
| 논문 1 대비 | **3.04배** |
| bin 채움 / 세션≥3 | 50/54 / **43/54** |
| train / val | 세션 37 / 6 |

**확정된 결정**

| 항목 | 결정 | 근거 |
|---|---|---|
| `spin` | **제외** | 회전 결함 때문이 **아니다**(그건 수정됨). 제자리 선회가 `\|wz\|` 5~6.29 rad/s 라 추종 불가이고, 봉투 내부비율 중앙값이 **0.03** 이라 트리밍으로도 못 살린다 |
| 명령 범위 | **`vx ∈ [-0.5, 3.0]`, `\|wz\| ≤ 1.5`** | `vx < -0.5` 세션 0~2개, `vx > 3.0` 극단 yaw 세션 0~2개 |
| 필터 단위 | **클립 단위** (내부비율 ≥0.90) | 프레임 단위로 자르면 상태전이 시계열이 끊긴다 |
| split | **세션 단위** | 클립 단위는 슬라이딩 윈도우 누수로 G1 이 가짜 게이트가 된다 |

**★ 남은 한계 (판정 시 반드시 참조)**: `vx > 2.0` 과 `\|wz\| > 1.0` 이 겹치는 영역은
기여 세션이 2~9개로 얇다. 이 영역 실패는 **설계 탓과 데이터 탓을 분리할 수 없다.**

**★ 파일명 파싱 함정**: `seq` 필드에 `ex03`, `047z` 변형이 있어 `D\d+_\d+_` 정규식을 쓰면
**35.6%가 조용히 누락**되고 세션 수가 46 → 26 으로 잘못 나온다. 빌더는 파싱 실패 시 중단한다.

<details>
<summary>원래 Phase 0 계획 (참고용)</summary>

### (원안) 데이터 변환 및 motion_lib 수정


> **경로 결정: txt → pkl 변환을 유지한다** (기존 방식). 따라서 파생값(속도·발위치)은
> `motion_lib` 이 재계산하며, §4-7 의 결함 2건을 **수정해야 한다**.
> 61열 txt 를 직접 읽는 대안은 채택하지 않았다 — 아래 손실 평가 참조.

| # | 작업 | 산출물 | 완료 기준 |
|---|---|---|---|
| 0-1 | 카테고리 선별 (`jump*` 제외, `spin` 은 §4-7 에 따라 재고) + md5 dedup + 미러 태깅 + **세션 ID 파싱** | 선별 목록 | 세션 46개 확인, 중복 0 |
| 0-2 | `lie_walk` 의 `\|wz\| = 16.16` 클립(+미러) 제외 | 제외 목록 | 스파이크 클립 0 |
| 0-3 | `convert_smr_to_pkl.py` 일괄 실행 (61열 txt → `frames(N,18)`) | 신규 pkl 세트 | 샘플 재검증 회전필드 diff = 0.0 |
| 0-4 | **신규 `go2_imitation_latent/motion_lib.py`** — 결함 2건 수정 | 새 파일 | V1~V4 (아래) 통과 |
| 0-5 | **명령 봉투 결정**(§4-4) 및 데이터 클리핑 | cfg + 클리핑 리포트 | 손실 프레임 비율 명시 |
| 0-6 | **bin 별 고유 세션 수** 집계 (§4-3-b) | `metrics/bin_session_diversity.csv` | 명령 봉투 안 모든 bin 에 **세션 ≥ 3** |
| 0-7 | **세션 단위 train/val split** 목록 | `metrics/session_split.json` | 클립 누수 0 (같은 세션이 양쪽에 없을 것) |

#### 0-4 의 수정 알고리즘 (MimicKit 원본 방식)

**(1) 회전 — exp map 으로 읽는다**

```
현재:  root_quat = _euler_to_quat_wxyz(frames[:,3:6])       # 오독
수정:  root_quat = _exp_map_to_quat_wxyz(frames[:,3:6])
       angle = |v| ,  axis = v/|v|  (|v| < 1e-8 이면 axis = e_z, angle = 0)
       q = [cos(angle/2), axis*sin(angle/2)]
```

**(2) 각속도 — euler-rate 야코비안을 버리고 쿼터니언 차분**

```
dq         = conj(q[t]) ⊗ q[t+1]        # body frame
dq         <- -dq  if  dq.w < 0          # quat_pos, 최단호 → wrap 원천 제거
angle      = 2·atan2(|dq.vec|, dq.w)
omega_body = (dq.vec/|dq.vec|) · angle / dt
마지막 프레임은 직전 값 복사 (기존 _finite_diff 규약 유지)
```

| 기호 | 의미 | 단위 |
|---|---|---|
| `q[t]` | world←body 쿼터니언 (wxyz) | — |
| `dq` | 한 스텝 상대회전 | — |
| `omega_body` | body frame 각속도 | rad/s |
| `dt` | `1/fps` | s |

MimicKit 은 `quat_diff(q0,q1) = q1 ⊗ conj(q0)` 로 **world frame** 각속도를 낸다.
이 저장소는 `motion_lib` 이 **body frame** 을 반환하는 계약이므로(docstring, `env.py:886`
에서 `quat_apply` 로 world 변환) 곱 순서를 뒤집어 body 로 맞춘다. **그대로 베끼면 프레임이 뒤바뀐다.**

**(3) 선속도는 구조 유지** — world 차분 후 body 변환. (1)의 회전 수정 혜택만 받는다.

`_euler_to_quat_wxyz` 와 `_euler_rates_to_body_angvel` 은 호출부가 사라지므로 제거한다.

#### 검증 (0-4 직후 즉시)

| # | 검증 | 통과 기준 |
|---|---|---|
| V1 | 회전 — 9개 클립 `q_new` vs 원본 txt 쿼터니언 | 오차 < 1e-6 |
| V2 | yaw 스파이크 — `go2_walk_turn` `\|wz\|` max | 373.31 → **2.238** (txt 원본값과 일치) |
| V3 | 각속도 전체 — 재계산값 vs txt `[34:37]` | 오차 < 1e-4 rad/s (전 클립) |
| V4 | 선속도 — 재계산값 vs txt `[31:34]` | 오차 < 1e-4 m/s |
| V5 | 회전 큰 데이터 — `spin` 카테고리 `\|wz\|` | 물리적 범위 내, txt 원본과 일치 |

V3/V4 가 통과하면 **pkl 경유로 인한 정보 손실이 없음이 증명된다** — txt 가 이미 갖고 있던
값을 motion_lib 이 동일하게 복원한다는 뜻이기 때문이다.

#### pkl 경유의 손실 평가 (수정 후 기준)

| 항목 | 손실 | 판정 |
|---|---|---|
| 회전 | 없음 — exp map 은 쿼터니언과 가역 | 무해 |
| 각속도 / 선속도 / 관절속도 | 재계산이지만 txt 와 동일 알고리즘 | V3/V4 로 검증 |
| **발 위치** | txt 의 MuJoCo FK 실측 → 해석적 FK 근사 | 문서화된 오차 **<0.5 mm** — 무해 |
| `toe_vel_local` | 폐기 | `x_vae` 49-D 에 미포함 — 무관 |

향후 발 속도나 접촉 정보가 필요해지면 pkl 포맷을 확장하는 것이 대안이다(현 단계에선 불필요).

**`csv_dataset_locomotion` 은 예비 확보처가 아니다** (§4-3-c) — 같은 46세션이다.
세션 다양성이 부족하면 처방은 리타게팅이 아니라
**`csv_dataset/`(7.5G, DogML 8,048 클립)에서 새 세션을 찾는 것**이며, 별도 트랙으로 둔다.


</details>

### Phase 1 — Motion Encoder 학습 (오프라인, 시뮬레이터 불필요)

신규 파일 `rsl_rl/rsl_rl/modules/motion_encoder.py`:

```
MotionEncoder:
  입력  : x_vae 윈도우 (w 프레임)
  구조  : EmpiricalNormalization -> MLP [256, 128] -> (mu, logvar)
  출력  : z ~ N(mu, diag(exp(logvar))),  d_latent = 16 ~ 32
  손실  : L = || x - decode(z) ||^2  +  beta * KL( q(z|x) || N(0, I) )
```

**입력 차원은 상수가 아니다.** `_compute_amp_obs` 는 기본 **43-D** 를 내고,
`_apply_root_rot_tan_norm` 이 heading-relative 6D 를 붙여야 **49-D** 가 된다.
여기에 `joint_pos_tan_norm=True` 면 관절 블록이 12 → 72 로 늘어 **103 → 109-D** 가 된다.

| `joint_pos_tan_norm` | 프레임당 | `w=10` 스택 |
|---|---|---|
| False (기본) | 49 | 490 |
| True | 109 | 1090 |

**본 계획은 `joint_pos_tan_norm=False` (49-D / 490-D) 를 전제**한다.
플래그를 켤 경우 인코더 입력 차원과 정규화 통계를 함께 갱신해야 한다.

기호:

| 기호 | 의미 | 값 |
|---|---|---|
| `w` | 인코더 입력 윈도우 길이 | 10 (기존 AMP history 재사용) |
| `d_latent` | latent 차원 | 16 (초기값) |
| `beta` | KL 가중치 | 0.05 (P1 준용), free-bits 병용 |

- 오프라인 스크립트: `scripts/imitation_learning/train_motion_encoder.py`
- **검증 게이트 (두 번째 게이트).** t-SNE/UMAP 로 "mode 가 갈리는지" 보는 것은
  **게이트로 쓰면 안 된다** — 고유 클립이 8개뿐이라 인코더가 아무것도 학습하지 못해도
  클립 정체성만으로 군집이 깨끗하게 갈린다. 즉 **항상 통과하는 가짜 게이트**다.
  대신 과적합에서 크게 실패하는 두 지표를 쓴다:

  | 게이트 | 방법 | 통과 기준 |
  |---|---|---|
  | G1. **leave-one-session-out** 재구성 | **원 녹화 세션 1개**를 통째로 빼고 학습 → 뺀 세션 재구성 오차 | held-out 오차가 in-sample 오차의 **2배 이내** |
  | G2. latent 보간 단조성 | 같은 mode·다른 속도 두 클립의 `z` 를 선형 보간 → 디코드 후 `v_x` 측정 | 보간 계수에 대해 `v_x` 가 **단조 증가** |

  t-SNE 는 게이트가 아니라 **참고용 시각화**로만 남긴다.

  **★ split 은 반드시 세션 단위여야 한다.** 클립 단위 split 은 누수가 난다 —
  같은 세션의 `F286_F325` 와 `F286_F343` 은 md5 가 달라 dedup 을 통과하지만
  프레임의 대부분이 동일하다. 클립 단위로 나누면 held-out 클립의 프레임이
  train 에 이미 들어가 있어 **G1 이 항상 통과하는 가짜 게이트가 된다**(§4-3).
  train/val 은 46개 세션을 단위로 나눈다.
- posterior collapse 방어: free-bits, KL annealing, 낮은 `beta`

### Phase 2 — Latent Style Reward 로 AMP 대체

신규 `rsl_rl/rsl_rl/algorithms/ppo_latent.py`, `rsl_rl/rsl_rl/runners/on_policy_runner_latent.py`
(**기존 `ppo_amp.py` / `on_policy_runner_amp.py` 는 손대지 않고 나란히 신설**).

**★ 발산 항의 정의 — 여기서 갈린다.** LatentMimic 표기 `KL(z_target || z_sim)` 를
글자 그대로 읽으면 안 된다. `z` 는 **표본**이고 두 점 사이의 KL 은 정의되지 않는다.
실제로 계산 가능한 것은 인코더가 뱉는 **분포끼리의 KL** 이며, 어떤 분포를 짝지을지에
따라 구현 비용이 완전히 달라진다.

**(ii) 배치 marginal — 본 계획의 기본값.** 롤아웃 배치 전체의 `z` 분포와
expert 배치의 `z` 분포를 맞춘다. 이것이 "marginal latent divergence" 의 문자 그대로의
의미이고, **phase 짝짓기가 필요 없어 env 구조 변경이 실제로 0** 이다.

```
mu_ref,  Sigma_ref  = 배치통계( Encoder(expert 윈도우 batch) )
mu_pi,   Sigma_pi   = 배치통계( Encoder(정책 롤아웃 윈도우 batch) )

D_marg  = KL_gauss( N(mu_pi, Sigma_pi) || N(mu_ref, Sigma_ref) )     # 대각 근사
r_style = exp( -c_kl * d_step )
```

여기서 `d_step` 은 스텝별 신호가 필요하므로 배치 통계를 그대로 못 쓴다.
스텝별로는 **expert 배치로 적합한 참조 분포에 대한 로그밀도**를 쓴다:

```
d_step[t] = -log N( z_sim[t] ; mu_ref, Sigma_ref )      # 낮을수록 expert 분포 안쪽
r_style   = exp( -c_kl * d_step )
```

배치 marginal `D_marg` 는 스텝 보상이 아니라 **모니터링 지표**로 로깅한다.

**(i) 위상 짝지음 — 대안, 비용 큼.** 같은 phase 의 expert 윈도우와 정책 윈도우를
짝지어 인코더의 두 posterior 사이 닫힌형 Gaussian KL 을 쓴다:

```
KL(q_ref || q_pi) = sum_i [ log(s_pi/s_ref) + (s_ref^2 + (m_ref - m_pi)^2)/(2*s_pi^2) - 0.5 ]
```

LatentMimic 원본에 더 충실하지만, **§3-3 의 모션 phase 추적 구조를 Phase 2 에서
바로 도입해야 한다**(현재 env 는 RSI 이후 phase 를 갱신·저장하지 않는다).
즉 이 선택지를 고르면 "안 B = env 변경 소" 라는 §5 의 평가가 더 이상 성립하지 않고,
Phase 4 의 구조 비용을 앞당겨 치르는 것이 된다.

**결정: (ii) 로 시작한다.** (ii) 는 스텝별 스타일 신호가 (i) 보다 약하고 노이즈가 크지만,
안 B 의 존재 이유인 "낮은 롤백 비용" 을 지킨다. (ii) 에서 스타일 신호가 너무 약해
gait 가 붕괴하면 그때 (i) 로 승격한다.

```
r_total = w_task * r_task  +  w_style * r_style
```

| 기호 | 의미 | 초기값 |
|---|---|---|
| `c_kl` | latent KL 민감도 | 0.01 |
| `w_task` | 속도추종 가중치 | 기존 `task_reward_lerp` 스케줄 승계 |
| `w_style` | 스타일 가중치 | `1 - w_task` |

- `w_task` 어닐링은 **기존 AMP 스케줄을 그대로 승계**한다. 과거 실측에서
  `task_reward_lerp 0.6~0.8` 이 `cmd 4.0` 달성률을 97~98% 로 끌어올렸고,
  **고정 0.8 은 `cmd 4.0` 에서 4 Hz trot 회귀**를 일으켰다. 이 지식은 latent 로 바꿔도
  "스타일 대 과제" 라는 축이 같으므로 그대로 유효할 가능성이 높다 —
  **초기 스윕에 반드시 포함할 것.**
- latent OOD 억제: P2 식 약한 KL 정규화(계수 2.5e-6) 또는 latent norm penalty.

### Phase 3 — Command-conditioned Latent Navigation (★ 요청의 핵심)

Phase 2 가 통과하면, 정책이 latent 를 **명령의 함수로 명시적으로 조작**하도록 확장한다.

```
z_cmd[t] = SynthesisPolicy( c[t], z[t-1] )         # c = [c_fwd, c_turn]
r_syn    = exp( -(v_fwd - c_fwd)^2 / 0.25  -  (yaw_rate - c_turn)^2 / 0.1 )
```

이 시점에서 vMF latent 로 전환한다 (논문 1은 이것 없이는 스타일 일관성을 잃는다고 명시).
구현은 `nicola-decao/s-vae-pytorch` (MIT) 의 vMF 분포·rejection 샘플러를 이식한다.
vMF 의 장점은 **액션 공간이 구 표면으로 유계**라 synthesis 정책의 탐색이 안정된다는 점이고,
P1 은 "이것 없이는 스타일 일관성을 잃거나 gait mode 를 재현하지 못한다"고 명시한다.

### Phase 4 — 논문 1 완전판 (residual tracking)

`theta_pd = theta_ref + 0.15 * a` 로 전환. 필요한 신규 구조:

1. env 에 **모션 phase 상태** (`motion_id`, `motion_time`) 추가 및 매 스텝 갱신
2. obs 를 `[o_state(H) , o_ref(H) , z]` 로 재설계 (P1: 358-D)
3. reward 를 `1.0*r_imitation + 0.5*r_world + 0.1*r_reg` 로 교체

P1 보상 민감도 (원문 표 III, `exp(-err^2 / sigma)` 의 `sigma`):

| 항 | sigma |
|---|---|
| base 선속도 | 0.2 |
| base 각속도 | 0.25 |
| base 높이 | 0.1 |
| base 자세 | 0.8 |
| 발 위치 | [0.3, 0.3, 0.1] (성분별) |
| world 위치 | 0.5 |
| world 자세 | 0.5 |
| action rate | 2.0 |
| action scale | 10.0 |
| 발 slip | 0.1 |

---

## 7. 평가 프로토콜 (이 저장소의 반복된 오판을 피하기 위해)

**학습 지표로 이동 능력을 판정하지 않는다.** 과거 실측에서 `lin_ps` / `R` / `ep_len` / `std`
네 지표 모두 실제 붕괴를 놓쳤고, `lin_ps` 는 붕괴 후 오히려 상승했다.

| 규칙 | 근거 |
|---|---|
| **속도 램프 실측**으로만 판정 | 학습지표 무효 이력 |
| 동일 조건 **≥4회 반복** | 램프 편차 |
| **40k iter 이후 점만** 사용 | 24k 이하에서 "3점 연속·두 플랜트·부호 6/6" 판정이 40k 에 반전된 이력 |
| AMP 를 **동시 진행 arm** 으로 유지 | 죽은 baseline 과 비교 금지 |
| 램프 실행 시 `--no_pace` 등 플랜트 플래그 확인 | 램프는 run params 가 아닌 **현재 소스 cfg** 로 env 를 만듦 (cross-plant 측정 사고 이력) |
| **보행 종류(위상차)까지 측정** | 달성률만 보면 4 Hz trot 회귀를 놓침 |
| 좌우 대칭(hip 쏠림) 확인 | `command_uniform` 가중치가 달성률은 올리고 대칭은 발산시킨 이력 |

산출물은 `reports/go2_imitation/go2_imitation_tracking/{experiment}/` 규약
(플롯 `figures/`, 수치 `metrics/` + 원자료).

---

## 8. 리스크

| 리스크 | 확률 | 완화 |
|---|---|---|
| ~~데이터 부족으로 latent 가 단일모드 퇴화~~ | ~~높음~~ → **해소** | 187,312 프레임 / 커버리지 100% 확보(§4-2) |
| **세션 다양성 부족 (46 세션)으로 세션 암기** | **높음** | Phase 0-7 세션 집계, Phase 1 의 G1(**leave-one-session-out**)·G2(보간 단조성) |
| **고속 영역이 세션 1~2개에만 의존** | 높음 | 해당 bin 을 명시적으로 추적, 고속 실패 시 데이터 탓인지 설계 탓인지 분리해 판정 |
| **강한 후진(`v_x < -1.0`) 데이터 부재** | 확실 | 명령 하한을 -0.5 로 설정(§4-4), 또는 합성 |
| **명령 봉투 밖 expert 가 latent 용량을 낭비** | 중 | Phase 0-6 클리핑(§4-4 (c)) |
| **안 A 의 구조 비용**(phase 추적·obs 재설계)이 예상보다 큼 | 중 | Phase 2 에서 동작하는 시스템을 확보해 두고 진입 — 실패 시 그 지점으로 롤백 |
| posterior collapse | 중 | free-bits, `beta` 낮게, KL annealing |
| latent OOD (정책이 미학습 latent 영역 방문) | 중 | 약한 latent KL 정규화, LatentMimic 의 `r_anchor` |
| autoregressive drift (Phase 3~4) | 중 | scheduled sampling, P1 의 20+60 epoch 전환 일정 준용 |
| latent 방식이 AMP 보다 나쁨 | 중 | AMP arm 동시 유지, 롤백 비용을 낮게 유지(안 B 선택 이유) |
| **동결 인코더가 exploit 당함** | 중 | AMP discriminator 는 정책의 현재 실패양상에 **적응**하지만 사전학습 후 동결된 인코더는 그러지 못한다. 정책이 "인코더가 자신 있게 expert 로 매핑하지만 실제로는 데이터와 전혀 다른" latent 영역을 찾아낼 수 있다. 완화: 인코더를 주기적으로 expert+정책 데이터로 재적합, 또는 `r_style` 상한. **롤아웃 영상으로 육안 확인 필수** |
| yaw 수정으로 expert 분포가 바뀌어 과거 비교 무효 | **확실** | 별도 arm 으로 명시, 과거 수치와 직접 비교 금지 |

---

## 9. 참고문헌

- P1: [arXiv 2507.00677](https://arxiv.org/abs/2507.00677) — Walk Like Dogs (Go2, vMF VAE)
- P2: [arXiv 2607.13579](https://arxiv.org/abs/2607.13579) — APT-RL (transformer VAE, torque decoder)
- [arXiv 2604.16440](https://arxiv.org/abs/2604.16440) — LatentMimic (Go1, marginal latent KL, AMP 대체)
- [arXiv 2603.01294](https://arxiv.org/abs/2603.01294) — Spherical Latent Motion Prior
- [arXiv 1804.00891](https://arxiv.org/abs/1804.00891) — Hyperspherical VAE (Davidson 2018) · 구현 [s-vae-pytorch](https://github.com/nicola-decao/s-vae-pytorch)
- [arXiv 2103.14274](https://arxiv.org/abs/2103.14274) — Motion VAE (Ling 2020)
- [arXiv 2205.01906](https://arxiv.org/abs/2205.01906) — ASE · [arXiv 2305.02195](https://arxiv.org/abs/2305.02195) — CALM
- [arXiv 2308.07200](https://arxiv.org/abs/2308.07200) — NCP (VQ-VAE) · [코드](https://github.com/Tencent-RoboticsX/NCP)
- [arXiv 2310.04582](https://arxiv.org/abs/2310.04582) — PULSE
- 데이터셋 실측: `reports/go2_imitation/go2_imitation_tracking/2026-08-27_13-23-12_vae_dataset_coverage/`

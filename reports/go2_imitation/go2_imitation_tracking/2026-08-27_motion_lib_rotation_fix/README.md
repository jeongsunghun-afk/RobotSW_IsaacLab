# motion_lib 회전 표현 / 각속도 수정 — 포팅 회귀 정정

2026-08-27 · 브랜치 `isaac-6.0`

---

## 요약

`motion_lib` 이 참조 모션의 root 회전을 **exponential map 인데 ZYX euler 로 읽고** 있었다.
MimicKit 원본 대비 **포팅 회귀**이며, 알려져 있던 "yaw wrap 으로 각속도 373 rad/s" 결함은
독립 버그가 아니라 이 오독의 **파생**이었다.

수정 후 재계산값이 리타게팅 원본이 이미 갖고 있던 속도값과
**원본 자체의 내부 정합성 한계까지 일치**한다(유효숫자 3자리 동일).

---

## 1. 결함

| # | 위치 (`motion_lib.py`) | 현행 | 올바름 |
|---|---|---|---|
| 1 | `:18, 437` | `[3:6]` = `root_euler` | `root_rot` (exponential map) |
| 2 | `:440` | `_euler_to_quat_wxyz` | `_exp_map_to_quat_wxyz` |
| 3 | `:442, 446` | euler finite-diff + 야코비안 | 쿼터니언 차분 |

---

## 2. 확정 근거

### 2-1. 원본 대조 (9/9 클립)

저장소 `smr_mirror_pkl/*.pkl` 9개 전부에 대해 리타게팅 원본 `.txt` 를
**위치 오차 `0.00e+00`** 으로 찾아 회전 필드를 대조했다.

| clip | vs exp map | vs euler |
|---|---|---|
| walk / walk1 / walk2 | ~1e-17 | 4.2e-3 / 5.7e-3 / 4.9e-2 |
| **walk_turn** | **4.4e-16** | **1.5e-1** |
| run0 / run1 / run2 | ~1e-17 | 6.9e-3 ~ 1.9e-2 |
| trot0 / trot1 | ~1e-17 | 1.7e-2 |

**9/9 exp map.** 재현: `verify_representation.py`

### 2-2. 생성 파이프라인

```
DogML 개 mocap csv  (27 keypoint, 60fps, 46 세션)
      ↓  stmr_go2.py    root_quat 을 wxyz→xyzw 로 변환해 기록 (stmr_go2.py:588, :612 주석)
   .txt  61열
      ↓  convert_smr_to_pkl.py     [3:6] = quat_to_exp_map(root_quat)
   .pkl  18열
      ↓  motion_lib.py             [3:6] 을 euler 로 읽음   ← 결함
```

### 2-3. MimicKit 원본

| 항목 | MimicKit | 포트 (수정 전) |
|---|---|---|
| `[3:6]` | `exp_map_to_quat` (`motion_lib.py:110`) | `_euler_to_quat_wxyz` |
| 각속도 | `quat_diff(q[t],q[t+1])` → `fps * quat_to_exp_map` (`:175-176`) | euler finite-diff + 야코비안 |
| 최단호 | `quat_pos` — `w<0` 부호반전 (`torch_util.py:75`) | 없음 |
| euler 사용 | **0건** | roll/pitch/yaw 로 해석 |

**쿼터니언에는 ±π 경계가 없으므로 원본 방식이었다면 wrap 결함이 애초에 생기지 않는다.**

저장소 안에도 이미 올바른 구현이 둘 있었다 — `go2_amp/motion_lib.py`(MimicKit 그대로),
`leg_imitation_tracking/motion_lib.py`(`_exp_map_to_quat_wxyz` 보유, 주석에
`convert_smr_to_pkl.quat_to_exp_map 의 역변환`이라 명시). **go2_imitation 계열만 회귀 상태였다.**

---

## 3. 영향 규모 (수정 전)

| clip | `\|wz\|` max 현행 → 정정 | `lin_vel_b` 오차 | 자세 오차 |
|---|---|---|---|
| run0/1/2 | 0.65 → 0.57 (1.1×) | ≤0.033 m/s | ≤1.10° |
| trot0/1 | 0.49 → 0.48 (1.0×) | 0.016 m/s | 1.01° |
| walk / walk1 | 변화 없음 | ≤0.003 m/s | ≤0.34° |
| walk2 | 0.95 → 0.98 | 0.006 m/s | 2.79° |
| **walk_turn** | **373.39 → 2.24 (167×)** | **0.090 m/s** | **12.08°** |

### 오염 경로

| 경로 | 위치 | 오염 |
|---|---|---|
| RSI 스폰 | `env.py:871-886` | 초기 자세·초기 각속도 |
| Discriminator expert 샘플 | `env.py:994` → `collect_reference_motions` → `get_amp_observations` | 49-D 중 **12개** |

오염 차원: `ang_vel`(3) + `rot_tan_norm`(6) + `lin_vel`(3) = 12.
`dof_pos`/`dof_vel`/`root_height`/`foot_pos`(37)는 회전과 무관해 영향 없음.

`go2_walk_turn` 은 고유 프레임의 **18.7%** 이고, `EmpiricalNormalization` 이 expert+policy 를
합쳐 std 를 fit 하므로 **wz 채널 std 가 팽창해 discriminator 의 yaw 감각이 사실상 무력화**된다.
1/49 차원이고 yaw 축이라 이 저장소가 쫓아온 속도 천장의 원인일 가능성은 낮다.

---

## 4. 수정 내용

**(1) 회전**

```
root_quat = _exp_map_to_quat_wxyz(frames[:, 3:6])
    angle = |v| ,  axis = v/|v|   (|v| < 1e-8 이면 axis = e_z, angle = 0)
    q     = [cos(angle/2), axis*sin(angle/2)]
```

**(2) 각속도 — 쿼터니언 차분 (unwrap 불필요)**

```
dq         = conj(q[t]) ⊗ q[t+1]        # body frame
dq        <- -dq  if dq.w < 0            # double cover 제거 = 최단호
omega_body = 2 · vec(dq) / dt
```

| 기호 | 의미 | 단위 |
|---|---|---|
| `q[t]` | world←body 쿼터니언 (wxyz) | — |
| `dq` | 한 스텝 상대회전 | — |
| `omega_body` | body frame 각속도 | rad/s |
| `dt` | `1/fps` | s |

> MimicKit 은 `q1 ⊗ conj(q0)` 로 **world frame** 을 낸다. 이 저장소 `motion_lib` 은
> **body frame** 반환 계약이므로 곱 순서를 뒤집었다. 그대로 베끼면 프레임이 뒤바뀐다.

**(3) 선속도** — 구조 유지(world 차분 → body 변환), (1)의 혜택만 받음.

`_euler_to_quat_wxyz` / `_euler_rates_to_body_angvel` 은 호출부가 사라져 제거.

### 수정 파일

| 파일 | 비고 |
|---|---|
| `direct/go2_imitation/motion_lib.py` | 원본 |
| `direct/go2_imitation_tracking/motion_lib.py` | byte-identical 동기화 (태스크 CLAUDE.md 요구) |
| `direct/go2_imitation_tracking_legacy48/motion_lib.py` | 동일 (git 미추적) |
| `direct/go2_imitation_tracking_obs45/motion_lib.py` | 동일 (git 미추적) |
| `direct/parkour_imitation/motion_lib.py` | 데이터가 exp map 임을 확인 후 적용 (5.5e-17) |

**미수정 — `go2_amp/go2_motion_loader.py`**: 같은 euler 결함이 있으나 **휴면 상태**다.
데이터의 yaw 변동폭이 최대 123.5° 로 ±π 를 넘지 않아 wrap 스파이크가 없고,
euler 방식과 쿼터니언 차분의 `|wz|` 차이가 0.12 rad/s 미만이다.
해당 pkl 의 원본을 찾지 못해 표현 포맷을 확정할 수 없어 손대지 않았다.
(`go2_amp/motion_lib.py` 는 MimicKit 방식으로 이미 올바르나 그 태스크에서 미사용.)

---

## 5. 검증 — PASS

`verify_fix.py` (재현 가능). 실행: `/home/user/miniconda3/envs/isaac-6.0/bin/python verify_fix.py`

**★ 절대 임계값을 쓰면 안 된다.** txt 의 저장된 속도는 **txt 자신의** 쿼터니언·위치로
재계산한 값과도 ~1.2e-3 rad/s 어긋난다(`stmr_go2.py` 내부에서 스무딩 전후 궤적이 섞인 것으로
보임). 따라서 **원본 자체의 내부 정합성(floor)** 을 먼저 재고, 우리 오차가 그 이내인지로 판정한다.

| clip | V1 회전 | V3 각속도 ours / floor | V4 선속도 ours / floor |
|---|---|---|---|
| walk | 5.03e-06 | 1.18e-03 / 1.18e-03 | 6.13e-04 / 6.13e-04 |
| walk1 | 5.12e-06 | 1.15e-03 / 1.16e-03 | 6.89e-04 / 6.89e-04 |
| walk2 | 4.97e-06 | 1.38e-03 / 1.39e-03 | 6.53e-04 / 6.53e-04 |
| walk_turn | 5.31e-06 | 1.40e-03 / 1.40e-03 | 7.04e-04 / 7.03e-04 |
| run0 | 5.01e-06 | 1.28e-03 / 1.29e-03 | 9.50e-04 / 9.54e-04 |
| run1 | 5.37e-06 | 1.49e-03 / 1.49e-03 | 1.04e-03 / 1.04e-03 |
| run2 | 4.96e-06 | 1.41e-03 / 1.41e-03 | 1.26e-03 / 1.26e-03 |
| trot0/1 | 5.27e-06 | 1.19e-03 / 1.18e-03 | 9.05e-04 / 9.09e-04 |

**V2 `go2_walk_turn` `|wz|` max = 2.237 rad/s** (수정 전 373.31, txt 원본 2.238)

V1 잔차 5e-6 은 `motion_lib` 출력이 float32 인 데서 온다(float64 로 계산하면 사라짐).
**V3/V4 가 floor 와 유효숫자 3자리까지 같다 — 소스가 허용하는 최대 정확도로 복원된다.**

원자료: `metrics/verify_output.txt`

---

## 6. 재학습

수정 전 코드로 돌던 `2026-08-27_10-54-18_ampTanNorm_stock`(iter 7300/60000)을 중단하고,
**동일 설정 + 수정본**으로 재시작했다.

```bash
CUDA_VISIBLE_DEVICES=0 setsid nohup python -u \
  scripts/reinforcement_learning/train.py --rl_library rsl_rl \
  --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name ampTanNorm_rotfix_stock \
  env.use_pace_params=false env.amp_joint_tan_norm=true
```

#### ★ 비교 대상과 그 한계 — 반드시 함께 읽을 것

| 항목 | 내용 |
|---|---|
| 비교 대상 | `2026-08-26_14-04-56_tannorm_stock` (42,500 iter) — tan-norm arm 의 **유일한** 장기 run |
| 그 run 의 상태 | **수정 전 motion_lib** 사용 |
| 표본 수 | **N = 1** |

즉 "rotfix 가 X 를 개선했다" 는 결론을 이 비교만으로 낼 수 없다.
§7 평가 프로토콜이 요구하는 **동일 조건 ≥4회 반복**을 만족하지 않는다.
중단된 `ampTanNorm_stock` 은 iter 7,300 이라 비교 대상이 못 된다.

**★ 과거 AMP 수치와 직접 비교 금지.** expert 분포가 바뀌었으므로 별도 arm 이다.
판정은 학습 지표가 아니라 **속도 램프 실측**으로 하고, **40k iter 이후 점만** 쓴다.

#### 현행 AMP 데이터셋을 Phase 0 봉투로 재평가

이 재학습은 여전히 `smr_mirror_pkl` 9클립을 쓴다. Phase 0 의 봉투 기준
(`vx ∈ [-0.5,4.0]`, `\|wz\| ≤ 2.0`, 내부비율 ≥0.90)으로 재보면 2개가 탈락한다:

| clip | 내부비율 | 사유 |
|---|---|---|
| `go2_run2` | 0.543 | `vx` 4.94 > 봉투 상한 4.0 (고속이라 잘림) |
| `go2_walk_turn` | 0.879 | `\|wz\|` 2.24 > 2.0, 임계값에 0.021 부족 |
| (대조) `spin` | 0.03 | 추종 불가 — **질적으로 다름** |

둘 다 **임계값 경계 탈락**이지 쓸 수 없는 데이터가 아니다
(`go2_run2` 는 이 저장소가 추구해온 고속 구간, `\|wz\|` 2.24 는 추종 가능).
**AMP baseline 은 latent arm 과 비교 가능하다.**

---

## 7. 재현 스크립트

| 파일 | 용도 |
|---|---|
| `verify_representation.py` | pkl `[3:6]` 이 exp map 인지 euler 인지 원본 quat 과 대조 |
| `verify_fix.py` | V1~V4 전체 검증 (floor 기반 판정) |

# Leg 17-DOF imitation — 방법별 설명과 비교

**2026-08-26** · task `Leg-Imitation-Tracking-RMA-v0` · 17-DOF (4족 + 허리) · rsl_rl PPO + AMP + RMA

로봇은 4족 12관절에 허리 1축, 발목 4축을 더한 17-DOF다. 모든 방법은 같은 골격
(PPO + AMP discriminator + RMA privileged encoder) 위에 얹힌 **추가 항** 또는 **데이터 구성 변경**이며,
아래 순서대로 하나씩 누적되며 도입됐다.

## BLUF

- **단일 지표로 방법 순위를 매길 수 없다.** 정상 주행(cmd 2.0~4.0), 정지 출발(cmd 0.5), 자세 대칭성은
  서로 독립적으로 움직인다. 실제로 이번 비교에서 세 축의 1위가 전부 다른 방법이다.
- **정지 출발 능력은 `reset_strategy`가 결정한다.** 학습 중 정지 리셋을 겪지 않은 정책은 정지에서
  보행으로 진입하지 못한다. 이는 추종 품질과 무관하다.
- **`command_uniform` 가중치는 그 자체로 쏠림을 만들지 않는다.** ds18에서는 쏠림 0.52를 만들었지만
  ds14에서는 0.15로 기준선 수준이다. 다만 아래 "교란 요인"에 적힌 이유로 데이터셋 단독 효과로
  단정할 수는 없다.
- **RMA estimator는 학습에만 반영되고 평가·배포 경로에는 빠져 있었다**(§7). 배포 조건으로 다시 재니
  정상 주행은 8개 방법 전부 무손실(|Δ| ≤ 1.6pp)인데 **정지 출발만 평균 −6.2pp** 떨어진다.
  §3의 표는 특권 정보(GT) 조건이므로 실기 성능이 아니다.

---

## 1. 각 방법 설명

### 1.1 골격 — RMA + AMP

**RMA (Rapid Motor Adaptation)**: 관측을 4개 그룹으로 나눠 실기에서 못 재는 값을 학습 시에만 준다.

| 그룹 | 차원 | 구성 |
|---|---|---|
| `policy` / `proprio` | 57 | gravity(3) + lin_vel_cmd(2) + yaw_vel_cmd(1) + joint_pos_off(17) + joint_vel(17) + actions(17) |
| `priv_explicit` | 6 | 실기 추정 대상 (base 선속도 등) |
| `priv` | 38 | base_mass(1) + base_com(3) + stiffness_ratio(17) + damping_ratio(17) |
| `history` | 10 x 57 | proprio 10 프레임 |

`priv`는 배포 시 사라지므로 history로부터 추정하도록 encoder를 학습시킨다.

**AMP (Adversarial Motion Prior)**: 참조 모션의 상태 전이를 진짜로, 정책의 전이를 가짜로 판별하는
discriminator를 두고, 그 출력을 보상에 섞는다. 섞는 비율이 `task_reward_lerp`다.

```
reward = task_reward_lerp * task_reward + (1 - task_reward_lerp) * amp_reward
```

`task_reward_lerp_start=0.5` → `task_reward_lerp=0.5`, `anneal_iters=5000`. 즉 이 계열은 전 구간
**task 50% + AMP 50%** 로 고정돼 있다. `lerp065` run만 0.65(task 쪽 가중)로 바꾼 A/B다.

### 1.2 L/R symmetry augmentation

좌우 미러가 물리적으로 등가라는 사전지식을 데이터 증강으로 주입한다. 관절 이름 기반으로 좌우를
swap하고, 미러 M = diag(1,−1,1) 아래 부호가 뒤집히는 관절만 flip한다.

```
M · R(a, th) · M = R(M·a, -th)
파트너 축 a' == +M·a  ->  th' = -th   (FLIP)
파트너 축 a' == -M·a  ->  th' = +th   (flip 없음)
```

현재 플랜트(Leg_URDF2)에서 FLIP 대상은 **hip 4개 + waist + 뒷발(HL/HR_foot)** 이다. 이 규칙은
2026-08-25 세션에서 6가지 방법(obs 레이아웃, 영자세, 구현 16개 테스트, 데이터셋 mirror 쌍, USD 축,
물리 엔진)으로 검증했고 **index 버그는 없다**.

변형: `use_data_augmentation`(배치를 2배로 늘려 학습) vs `use_mirror_loss`(미러 출력 간 손실 추가).
현재 기본은 data augmentation ON, mirror loss OFF다.

### 1.3 torque penalty

```
reward += -torque_penalty_w * sum(applied_torque^2)
```

`torque_penalty_w = 1e-5`. 관절 토크 과다 사용을 억제해 실기 이식성을 높이려는 항이다.

### 1.4 command deadzone

```
if |vx_cmd| <= cmd_deadzone:  vx_cmd = 0
```

`cmd_deadzone = 0.1`. 아주 작은 명령을 0으로 눌러 "정지"를 명확한 모드로 만든다.
**cmd 0.5는 deadzone(0.1) 밖이라 직접 영향을 받지 않는다.**

### 1.5 정지 리셋 (`reset_strategy` + `rel_stand_envs`)

```
# leg_imitation_tracking_env.py:426
if "stand" in self.cfg.reset_strategy and self.cfg.rel_stand_envs > 0.0:
    ...  # rel_stand_envs 비율만큼 default_pos + noise, root 속도 0 으로 리셋
```

기본 리셋은 RSI(Reference State Initialization) — 참조 모션의 임의 프레임에서 시작하므로 **로봇이
이미 움직이는 상태로 태어난다.** `reset_strategy="random_stand"` 를 주면 `rel_stand_envs`(=0.1) 비율의
env가 정지 자세에서 시작한다.

**주의: `reset_strategy` 문자열에 `stand` 가 없으면 `rel_stand_envs` 값은 무시된다.** yaml에
`rel_stand_envs: 0.1` 이 적혀 있어도 `reset_strategy: random` 이면 정지 리셋은 **0회**다.

### 1.6 모션 샘플링 가중치 (`motion_weight_mode`)

- **`length`** (기본): 클립 길이에 비례. 길이가 곧 분포가 되어 긴 클립의 속도대가 과대 대표된다.
- **`command_uniform`**: 속도축에서 각 클립이 담당하는 **최근접 셀(Voronoi) 폭**을 가중치로 준다.
  셀 경계는 이웃 속도와의 중점이고 `[0, vel_max]`로 잘린다. 같은 속도의 mirror 쌍은 셀을 균등 분할한다.
  목표는 expert 속도 분포를 `U[0, vel_max]`에 맞추는 것이다.

부작용: 속도축에서 **고립된** 클립이 넓은 셀을 받는다. ds14에서 `leg_walk1`(0.91 m/s, **0.62초**)은
이웃이 0.44와 1.96으로 멀어 셀 폭이 가장 넓고, 그 결과 0.6초짜리 클립 하나가 전체 가중치의 약 20%를
가져간다.

### 1.7 명령 상한 (`lin_vel_x_max`)

4.0 → **3.2**. 참조 모션의 최고 속도가 3.01 m/s라 4.0 명령은 참조가 없는 외삽 구간이었다.

### 1.8 데이터셋 계보

| 이름 | 클립 수 | 비고 |
|---|---|---|
| `smr_leg_pkl` | 9 | 초기. mirror 클립 없음 |
| `new_smr_leg_pkl` (**ds14**) | 14 | mirror 쌍 + stmr 변형 추가. 속도 0.05~3.01 m/s |
| `new_smr_trot110150_leg_pkl` (**ds18**) | 18 | ds14 + trot 보간 클립 4개 |
| `waistfix_ds14_leg_pkl` | 14 | ds14의 허리 관절 리타게팅 수정판 |
| `new_smr_run2t14_leg_pkl` | 16 | ds14 + run2 계열 |

ds14의 속도 분포에는 **0.91 ~ 1.96 m/s 공백**이 있다.

---

## 2. 측정 프로토콜과 비교 가능성

평가는 속도 램프다. 정지에서 시작해 `vx` 명령을 0 → 4.0 → 0으로 0.5 단위 사다리꼴(각 단계
상승 2초 + 유지 3초)로 움직이며 실제 `vx`를 잰다. 판정은 각 단계 **유지 구간**의 중앙값만 쓴다.

```bash
python _workspace/leg/speed_ramp_record_rma.py \
  --checkpoint <run>/model_<iter>.pt --run_params <run>/params/env.yaml \
  --force_stand --heading_hold --headless
```

### 프로토콜 감사 결과 (중요)

램프 npz에 기록된 메타데이터(`heading_hold`, `no_dr`, `no_push`, `only_push`, `reset_strategy`,
`hold_s`, `ramp_s`, `dt`, `vx_profile`)로 전 측정을 그룹핑한 결과 **16개 프로토콜 그룹**이 나왔다.

초기 방법 ablation(`rma_dr`, `sym_aug`, `mirrorloss`, `torque1e5`)은 전부 **구 프로토콜**
(hold 6.0초, 명령 상승 구간 없음, heading_hold 없음)에서 측정돼 현대 run과 **직접 비교가 불가능**했다.
따라서 이번 보고서를 위해 49999까지 완주한 구형 run들을 **현대 프로토콜로 재측정**했고, 프로토콜
필드가 기준 run과 완전히 일치함을 확인한 뒤에만 표에 넣었다.

미완주 run(`rma_dr` 23400, `sym_aug` 30500, `mirrorloss` 19600, `lerp065` 34400, `lerp030` 14200)은
같은 iter 비교가 불가능하므로 **표에서 제외**했다.

### 판정 지표를 세 축으로 분리한 이유

`ramp_summary.py`의 평균 달성률은 `per_rollout`이 `range(1, n_up)`로 **상승 구간만** 집계하므로
사실상 "정지에서 가속하는 능력"이고, 저속 한 점(cmd 0.5)이 값을 지배한다. 그래서 아래 표는 세 축을
분리한다.

| 축 | 정의 |
|---|---|
| `steady` | cmd 2.0~4.0 상승 유지 구간의 달성률 — 정상 주행 추종 |
| `start` | cmd 0.5 상승 유지 구간의 달성률 — 정지에서 보행 진입 |
| `skew` | hip 좌우 관절각 합의 절대값 (앞 + 뒤) — 자세 대칭성, 0이 대칭 |

`start`는 `duty`(유지 구간에서 `|jvel|RMS > 0.4`인 스텝 비율)와 함께 본다. duty가 0이면
"느리게 걷기 실패"가 아니라 **"아예 걷지 않음"** 이라 원인이 다르다.

---

## 3. 비교 결과

모두 **49999 iter 완주** 체크포인트, 동일 프로토콜, 롤아웃 4회(ds14+vmax32만 8회) 중앙값이다.
`cond=GT`는 actor 가 env 의 ground-truth base 속도를 받은 조건이다(§7 참조).

| 방법 | steady (2.0~4.0) | start (0.5, 정지출발) | duty | skew | 변위 | reset |
|---|---|---|---|---|---|---|
| torque1e5 (smr) | **86.6%** | **−0.6%** | 0.02 | 0.439 | 130.2 m | random |
| nosym | 70.3% | 1.9% | 0.17 | 0.227 | 79.7 m | random |
| stand+dz | 82.9% | 45.6% | 0.80 | **0.531** | 122.9 m | random_stand |
| trotmirror | 69.2% | 61.2% | 0.99 | 0.349 | 86.3 m | random_stand |
| ds14 | 69.9% | 43.7% | 0.64 | 0.156 | 77.9 m | random_stand |
| ds14+vmax32 | 83.8% | 40.7% | 0.76 | 0.285 | 123.1 m | random_stand |
| trot110150 | 81.7% | 46.3% | 0.96 | 0.163 | 128.8 m | random_stand |
| trot110150+wcmd | 84.0% | **68.5%** | 0.97 | 0.520 | **134.4 m** | random_stand |
| waistfix | 80.1% | 45.3% | 0.92 | 0.329 | 121.7 m | random_stand |
| waistfix+wcmd | 66.2% | 37.1% | 0.69 | 0.398 | 60.4 m | random_stand |
| **ds14+wcmd** | **87.7%** | 30.0% | 0.43 | **0.147** | **134.8 m** | random |

![방법 비교](figures/method_comparison.png)

### 3.1 세 축의 1위가 서로 다르다

- **steady 1위 = `ds14+wcmd`(87.7%)**, 근소하게 `torque1e5`(86.6%)·`trot110150+wcmd`(84.0%)
- **start 1위 = `trot110150+wcmd`(68.5%)**, 그 다음 `trotmirror`(61.2%)
- **skew 1위 = `ds14+wcmd`(0.147)**, 근소하게 `ds14`(0.156)·`trot110150`(0.163)

`torque1e5`가 특히 극적이다 — **정상 주행 86.6%로 2위인데 정지 출발은 −0.6%** 다. 단일 평균으로
순위를 매기면 이 정책은 "중간"으로 뭉개진다.

### 3.2 정지 출발은 `reset_strategy`가 가른다

| `reset_strategy` | start 범위 | duty 범위 |
|---|---|---|
| `random` (정지 리셋 0회) | −0.6% ~ 30.0% | 0.02 ~ 0.43 |
| `random_stand` (10% 정지 리셋) | 37.1% ~ 68.5% | 0.64 ~ 0.99 |

**겹치는 구간이 없다.** 그림 (b)에서 빨강 3점이 전부 아래, 파랑 8점이 전부 위에 있다.

`reset_strategy`는 `cmd_deadzone`과 완전히 공선적(dz 없음/0 ⟺ reset random)이라 이 표만으로는
둘을 분리할 수 없다. 판별은 **상승/하강 대조**로 했다 — deadzone(0.1)은 cmd 0.5의 1/5이라
전이 특이적 결손을 만들 수 없기 때문이다.

| 방법 | reset | 상승 0.5 (duty) | 하강 0.5 (duty) |
|---|---|---|---|
| torque1e5 (smr) | random | **−1% (0.02)** | 74% (0.98) |
| nosym | random | **2% (0.17)** | 54% (0.98) |
| ds14+wcmd | random | 30% (0.43) | 70% (1.00) |
| stand+dz | random_stand | 46% (0.80) | 52% (0.95) |
| trotmirror | random_stand | 55% (0.91) | 63% (1.00) |
| ds14 | random_stand | 44% (0.64) | 62% (0.99) |
| trot110150 | random_stand | 46% (0.96) | 51% (0.93) |
| trot110150+wcmd | random_stand | 69% (0.97) | 72% (0.99) |

`random` run들은 같은 명령 0.5를 **내려올 때는 54~74%로 정상 수행**한다. 걸음 자체는 있고
정지에서 진입만 못 한다 → deadzone이 아니라 정지 리셋 부재로 설명된다.

`torque_penalty_w`는 공선적이지 않다 — `nosym`·`torque1e5` 모두 1e-5인데 start가 무너진다.
따라서 start에 관한 한 `ds14+wcmd`의 4축 교란은 **reset과 deadzone 2축으로 좁혀진다.**

---

## 4. 교란 요인 — 반드시 함께 읽을 것

`ds14+wcmd` run은 이름이 `torque1e5_stand_dz_vmax32_ds14_wcmd`지만, `params/env.yaml`의 실값은
다음과 같이 **비교 대상과 4개 축이 다르다.**

| 축 | ds14 / waistfix / trot110150 | **ds14+wcmd** |
|---|---|---|
| `motion_weight_mode` | length | **command_uniform** (의도한 변경) |
| `cmd_deadzone` | 0.1 | **0.0** |
| `torque_penalty_w` | 1e-5 | **0.0** |
| `reset_strategy` | random_stand | **random** (정지 리셋 0회) |

원인은 학습 실행 시 hydra 오버라이드(`env.cmd_deadzone=0.1` 등)를 넘기지 않아 세 값이 **cfg 기본값**으로
떨어진 것이다. run 이름은 이전 실험에서 복사돼 실제 설정을 반영하지 않는다.

따라서 **"ds14+wcmd가 좋아진 것은 데이터셋 덕분"이라고 단정할 수 없다.** 토크 페널티 제거와 정지 리셋
비활성도 같은 방향으로 기여했을 수 있다. 단, `command_uniform` 자체가 쏠림의 원인이 아니라는 결론은
별도의 단일 변수 A/B(같은 ds18 클립, 가중치만 교체)에서 이미 확보돼 있어 유지된다.

---

## 5. 이번 세션에서 정정한 것

**쏠림 수치는 정의에 따라 크게 달라진다.** 롤아웃별 중앙값을 먼저 구하면 서로 반대 방향으로 기운
롤아웃이 상쇄된다.

| 방법 | A: 롤아웃 중앙값 먼저 | B: 롤아웃별 절대값 먼저 | 롤아웃별 앞쏠림 부호 |
|---|---|---|---|
| ds14 | 0.149 | 0.156 | −0.00 −0.02 −0.03 +0.02 (일관) |
| trot110150 | 0.151 | 0.163 | −0.10 −0.11 −0.10 −0.13 (일관) |
| trot110150+wcmd | 0.502 | 0.520 | +0.45 +0.49 +0.38 +0.41 (일관, 큼) |
| **ds14+wcmd 50k** | **0.082** | **0.147** | −0.13 +0.03 +0.03 +0.04 (**부호 불일치**) |

`ds14+wcmd`만 롤아웃 간 부호가 갈려 A에서 값이 절반으로 줄었다. 상쇄가 없는 B 기준으로는
**0.147로 ds14(0.156) / trot110150(0.163)과 사실상 동률**이다.

→ 이전에 "쏠림 0.047~0.117로 역대 최저"라고 한 것은 **정의 아티팩트**였다. 정확한 서술은
"ds18+wcmd의 큰 일관 쏠림(0.52)을 없애고 기존 기준선(~0.15)으로 되돌렸다"이다.

기타 정정:
- "cmd 0.5가 단조 붕괴" → 아니다. 25k가 최저점이고 40k·50k에서 회복한다(롤아웃 분산이 큰 구간).
- "걷기 시작은 cmd 0.88부터" → 25k 롤아웃 1 하나의 관찰이었다. 집계하면 대부분 cmd 0.5에서 이미
  다리를 움직인다.

---

## 6. 영상

쏠림이 가장 큰 정책(`ds18+wcmd`, skew 0.520)과 가장 작은 정책(`ds14+wcmd`, 0.147)을 같은 시점끼리
좌우로 붙였다. 둘 다 50k 체크포인트, 동일 램프(0 → 4.0 → 0, 85초)다.

| 파일 | 시점 | 무엇이 보이나 |
|---|---|---|
| `videos/compare_top.mp4` | 상공 | **가장 명확하다.** 오른쪽은 몸통이 진행 방향과 어긋나 비스듬히 나아간다 |
| `videos/compare_side.mp4` | 우측면 | 보행 자세와 발 궤적, 몸통 높이 |
| `videos/compare_diag.mp4` | 뒤측면 3/4 | 전체 자세와 좌우 다리 배치 |

원본 램프 영상은 `_workspace/leg/view_ds14wcmd_{side,top,diag}/` 와
`_workspace/leg/view_wcmd_{side,top,diag}/` 에 있다(1280x720, 85초). 비교본은 중앙을 잘라
확대한 것이다 — 체이스캠 기본 화각에서는 로봇이 작아 보행 자세가 읽히지 않는다.

---

## 7. 측정 조건 — estimator를 쓰면 수치가 달라진다

### 7.1 무엇이 문제였나

RMA estimator는 **학습 롤아웃에만 반영되고 평가·배포 경로에는 반영되지 않는다.**

- 학습: `ppo_parkour.py:185`가 `obs_est["priv_explicit"] = self.estimator(obs["policy"])`로 덮어써
  actor가 추정값을 소비한다(`train_with_estimated_states=True`). critic은 GT를 본다.
- 평가: `actor_critic_parkour.py:305 act_inference`가 `get_priv_explicit_obs(obs)`로 **env의 GT**를
  그대로 concat한다. `get_inference_policy`(`on_policy_runner_parkour.py:257`)는 estimator를 감싸지 않는다.

즉 `priv_latent`는 `history_latent`로 대체돼 배포 형태를 지키는데 **`priv_explicit`만 특권 정보가
들어간다.** estimator 자체는 정상 학습됐다 — 체크포인트에 `57→128→64→6`, `actor.0.weight`=(512, **83**)
= 57+6+20, `Mean estimator loss` 0.5003 → **0.0087**.

그래서 `speed_ramp_record_rma.py`에 **`--use_estimator`** 를 추가해 배포 조건을 재현했다.
npz에 `use_estimator`를 기록하고, `method_compare.py`는 GT/EST 혼합 시 `SystemExit`으로 막는다.

### 7.2 결과 — 정상 주행은 무손실, 정지 출발만 무너진다

| 방법 | reset | steady GT → EST | Δ | start GT → EST | Δ | skew GT → EST |
|---|---|---|---|---|---|---|
| torque1e5 (smr) | random | 86.6% → 88.0% | +1.4 | −0.6% → −0.3% | +0.3 | 0.439 → 0.518 |
| stand+dz | random_stand | 82.9% → 83.4% | +0.4 | 45.6% → 46.7% | +1.1 | 0.531 → 0.529 |
| ds14 | random_stand | 69.9% → 68.6% | −1.2 | 43.7% → 40.9% | −2.9 | 0.156 → 0.155 |
| ds14+vmax32 | random_stand | 83.8% → 85.2% | +1.4 | 40.7% → 27.6% | **−13.2** | 0.285 → 0.301 |
| trot110150 | random_stand | 81.7% → 81.4% | −0.3 | 46.3% → 47.4% | +1.1 | 0.163 → 0.208 |
| trot110150+wcmd | random_stand | 84.0% → 84.1% | +0.1 | 68.5% → 59.8% | **−8.8** | 0.520 → 0.530 |
| waistfix | random_stand | 80.1% → 80.7% | +0.6 | 45.3% → 42.4% | −2.9 | 0.329 → 0.313 |
| ds14+wcmd | random | 87.7% → 86.1% | −1.6 | 30.0% → 6.0% | **−24.0** | 0.147 → 0.210 |

**평균 Δ: steady +0.1pp, start −6.2pp.**

- **정상 주행(cmd 2.0~4.0)은 8개 방법 전부 |Δ| ≤ 1.6pp** — 속도가 실린 구간에서 estimator는 충분히
  정확하다. 이 결론은 강건하다.
- **정지 출발은 대부분 나빠지고, 폭이 방법마다 크게 다르다**(−24.0 ~ +1.1). 정지 부근에서는 base
  속도가 0에 가까워 상대 오차가 크고, 하필 그 신호가 "보행을 시작할지"를 정하는 데 필요하다.
- skew는 절반에서 소폭 악화된다(가장 큰 변화 0.147 → 0.210).

**주의 — start의 방법별 Δ는 신뢰도가 낮다.** §3에서 보였듯 상승 0.5는 롤아웃 간 이봉성이 강해
(예: `ds14+wcmd` GT가 `[0, 1, 59, 66]`) n=4 중앙값이 두 모드 사이에서 튄다. 개별 −24.0pp를
방법의 속성으로 읽으면 안 되고, "정지 출발이 estimator 오차에 취약하다"는 방향성만 취해야 한다.

`reset_strategy`로도 설명되지 않는다 — `ds14+vmax32`와 `trot110150+wcmd`는 `random_stand`인데도
각각 13.2pp, 8.8pp를 잃는다.

### 7.3 배포 export가 없다

`scripts/real2sim/`에 `export_deployable_go2.py`와 `export_deployable_bipedleg.py`는 estimator와
history_encoder를 내장한 self-contained jit을 만든다("표준 exporter는 actor MLP만 뽑아
estimator/history_encoder가 빠진다"). **17-DOF `leg_imitation_tracking`용은 없다.**

## 산출물

- `figures/method_comparison.png` — 세 축 비교, steady-start 산점도, 명령별 달성 곡선
- `metrics/method_table_GT.txt` — §3 표의 원자료 (11개 방법, GT 조건)
- `metrics/gt_vs_estimator.txt` — §7 표의 원자료 (8개 방법, GT vs EST)
- `videos/compare_{top,side,diag}.mp4` — 쏠림 최대/최소 정책의 3시점 좌우 비교
- 생성 스크립트: `_workspace/leg/method_compare.py`, `_workspace/leg/plot_method_compare.py`,
  `_workspace/leg/make_view_compare.py`
- 관련 보고서: `../lowspeed_gait_onset_hysteresis/` (저속 실패의 원인 규명)

### 재현

```bash
# GT 조건 (기존 기본값 — 특권 base 속도)
python _workspace/leg/speed_ramp_record_rma.py \
  --checkpoint <run>/model_49999.pt --run_params <run>/params/env.yaml \
  --out_dir <out> --force_stand --heading_hold --headless --no_video

# 배포 조건 (estimator 추정)
python _workspace/leg/speed_ramp_record_rma.py ... --use_estimator
```

# Policy 아키텍처 스케일링 실험 — 후보 선정 및 결정 프레임

**2026-08-26** · 상태: **완료 — 결론 = 깊이 스케일링 스윕 하지 않음, 레버는 `vel_err_scale`** (→ §7)

![판정 요약](figures/arch_scaling_verdict.png)

*(a) latent 경로를 학습 시점(priv)으로 되돌려도 cmd 0.5 상승 실패가 남는다 — 증류 오차 가설 기각.
(b) cmd 0.5에서 보상이 거의 평탄한 이유와, 두 arm이 그것을 세우는 방식.
생성: `_workspace/leg/plot_arch_scaling_verdict.py`*

**대상 run** (전부 `logs/rsl_rl/leg_imitation_tracking_rma/`):

| 역할 | run | 사용 |
|---|---|---|
| 주 분석 대상 | `2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd` | cfg 실측, A/B(`model_49999`) |
| reset_strategy 대조 | `2026-08-10_10-38-59` ~ `2026-08-22_18-08-31` (7 run) | §7-4b |
| duty 재분석 | 위 run들 + `2026-08-18_09-28-11_..._trot110150_wcmd` | §A-1 반증 (19 롤아웃) |
| 램프 원자료 | `_workspace/leg/ds14_vmax32_final_*`, `ds14_0810_final_*`, `wcmd_ramp_15000_*` 등 | §A-2, §A-1 반증 |
| A/B 산출 | `_workspace/leg/ab_privhist/{hist,priv}_{1..4}/ramp_data.npz` | §7-5 |

교차 참조: `_comparisons/lowspeed_gait_onset_hysteresis/` (이 문서가 출발점으로 삼은 실패 모드).
관련 task: 축 설계·상수 LR 선례는 `reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit/` §15.

이 문서는 "모델 구조/규모를 바꾸면 leg_imitation_tracking이 좋아지는가"를 실험하기 전에,
**무엇을 측정할 것인지 / 몇 개를 돌릴 수 있는지 / 무엇을 고정할 것인지**를 먼저 못박은
결정 프레임이다. 리서치 결과는 이 프레임 안으로 들어와야 하며, 프레임을 넓히려면 근거가 필요하다.

> **결론 요약 (§7 전문)**
> 1. 깊이 스케일링 전제는 PPO로 **전이되지 않는다**. 1000-layer 결과는 InfoNCE 목적함수 특유이고
>    (같은 논문의 SAC·TD3는 4층 넘으면 악화, offline은 저자가 직접 부정), vanilla PPO는 7층에서
>    **실측 붕괴**한다(Ant-v4 5323 → 1003).
> 2. BC mystery 글은 **표현력 주장을 하지 않는다**. 저자 본인이 미해결이라고 쓴다.
>    깊이와 표현력은 직교축이며, 대각 가우시안은 깊이와 무관하게 단봉이다.
> 3. 겨냥하려던 저속 실패는 **보상 모양** 문제다. cmd 0.5에서 정지가 이미 task 보상의 88.25%를
>    받고, 탈출 유인이 고속 대비 8.5배 얕으며, 140% 과속이 정지보다 6배 싸다.
> 4. **아키텍처 쪽 가설 3개가 전부 반증됐다** — 전부 새 학습 없이 기존 데이터 재분석으로:
>    관측성(A-1)은 cmd 1.0에서 19/19 성공으로, 증류/estimator OOD(A-1c)는 명령 경계면에서
>    완전 정지 상태인데도 0.5초 만에 걷기 시작하는 것으로. 남은 것은 보상 gradient 하나다.
> 5. 따라서 순서를 바꾼다: **`vel_err_scale` 보상 arm 먼저**, 아키텍처(LSTM 하나)는 그 다음.

---

## 0. 현재 baseline

| 항목 | 값 |
|---|---|
| task | `Leg-Imitation-Tracking-v0` (17-DOF 4족+허리) |
| policy | `ActorCritic`, `actor_hidden_dims=[512,256,128]`, ELU |
| critic | 동일 `[512,256,128]` |
| obs / act | 63 / 17 |
| 정규화 | actor·critic 모두 empirical normalization ON |
| 알고리즘 | `PPOAMPBase` (PPO + AMP discriminator `[1024,512]`) |
| LR | 2e-4, `schedule="adaptive"`, `desired_kl=0.01` |
| 탐색 | `init_noise_std=0.25`, `entropy_coef=0.005` |
| 대칭 | `LEG_SYMMETRY_AUG=1` (data-aug, `num_aug=2`) |
| 롤아웃 | `num_envs=4096`, `num_steps_per_env=24`, 5 epochs × 4 minibatch |
| 학습량 | `max_iterations=50000` |

actor 파라미터 수(대략): `63×512 + 512×256 + 256×128 + 128×17 ≈ 199k`.

---

## 1. 이 실험이 겨냥하는 실패 모드

아키텍처 실험의 arm은 **반드시 아래 중 하나의 명명된 실패 모드**에 붙어야 한다.
붙지 않는 arm은 숫자가 나와도 해석할 수 없으므로 후보에서 제외한다.

### (A) 저속 보행 개시 이봉 — 1순위
`_comparisons/lowspeed_gait_onset_hysteresis/` 참조.

같은 정책·같은 명령(cmd 0.5)인데 램프 **상승**(정지→출발)은 1~64%, **하강**(4.0에서 감속,
이미 걷는 중)은 61~77%. 7 run 전부 하강 duty가 0.98~1.00. 롤아웃 단위로 보면 duty≈1(걷는다)과
duty≈0(선다)으로 **이봉 분포**를 이룬다.

> **[2026-08-26 정정]** 처음에는 이것을 "한 관측에서 두 행동 모드가 필요한 다중모드 문제"로 읽고
> 표현력(다봉 정책·잠재 모드·순환 상태) 축이 답이라고 봤다. **이 독해는 §A-1d 에서 뒤집혔다.**
> 이봉으로 보이는 것은 정책의 출력 분포가 이봉이어서가 아니라 **보상 지형에 거의 대등한 두
> 끌개가 있어서**다(정지 0.8825 vs 완주 1.0000). 아래 A-1 → A-1e 를 순서대로 읽을 것.

#### (A-1) 기계적 가설: 정지 근방에서 **보행 위상이 관측 불가**하다 — **[반증됨, 아래 §A-1 반증]**

baseline `Leg-Imitation-Tracking-v0`의 policy 관측은 63차원이고 **history도 gait phase도 없다**:

```
root_lin_vel_b(3) + root_ang_vel_b(3) + projected_gravity_b(3)
+ lin_vel_cmd(2) + yaw_vel_cmd(1)
+ joint_pos_offset(17) + joint_vel(17) + actions(17)          = 63
```

`obs_groups = {"policy": ["policy"], "critic": ["policy"]}` — 즉 순수 memoryless 정책이다.

보행은 극한주기(limit cycle)이므로 어느 위상에서 출발할지를 정해야 하는데, 정지 상태에서는
`joint_vel ≈ 0` 이라 위상을 순간 관측에서 복원할 수 없다. 반면 이미 걷는 중(하강 구간)에는
`joint_vel`이 위상을 그대로 실어 나른다. **이 비대칭이 상승/하강 격차의 방향과 일치한다.**

이 가설이 맞다면 처방은 **깊이가 아니라 메모리/위상 구조**다. 단봉 가우시안을 더 깊게 쌓아도
관측이 같은 두 지점을 구분할 수 없다. 단, 아직 **가설**이며 아래 예비 관측은 통제된 A/B가 아니다.

#### (A-1b) ★ 정정 및 강화 — 현행 run은 RMA인데, **history가 actor에 도달하지 않는다**

먼저 정정한다. 히스테리시스를 측정한 run들은 baseline이 아니라 **RMA 변형**이다
(`logs/rsl_rl/leg_imitation_tracking_rma/2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd`).
`obs_groups`에 `history: [history]`가 있고 `history_len=10`, `num_proprio=57`이므로 겉보기에는
history가 있다. **그런데 PPO가 최적화하는 actor는 그 history를 보지 않는다.**

`rsl_rl/modules/actor_critic_parkour.py:283-292`:

```python
def act(self, obs, hist_encoding=False, **kwargs):
    obs_actor = self.actor_obs_normalizer(self.get_actor_obs(obs))
    priv_explicit = self.priv_explicit_obs_normalizer(self.get_priv_explicit_obs(obs))
    if hist_encoding:
        history_latent = self.get_hist_latent(obs)          # ← deploy 경로
        obs_actor = torch.cat([obs_actor, priv_explicit, history_latent], dim=-1)
    else:
        priv_latent = self.get_priv_latent(obs)             # ← 학습 rollout 경로
        obs_actor = torch.cat([obs_actor, priv_explicit, priv_latent], dim=-1)
```

그리고 `rsl_rl/runners/on_policy_runner_parkour_amp.py:110`:

```python
hist_encoding = it % 20 == 0          # dagger_update_freq
```

즉 **20 iteration 중 1회(5%)만** history 경로로 롤아웃하고, 나머지 95%는 `priv_latent` 경로다.
history encoder는 PPO의 정책 입력이 아니라 **DAGGER 증류 대상**일 뿐이다
(history → priv_latent를 모사하도록 학습).

그리고 `priv_latent`가 무엇인지 보면 (`leg_imitation_tracking_rma_env.py:87-107`):

```
priv_latent(38) = base_mass(1) + base_com(3)
                + joint_stiffness_ratio(17) + joint_damping_ratio(17)
```

**전부 에피소드 내 상수인 물리 파라미터다.** 시간 정보가 0이다.

따라서 PPO가 실제로 최적화하는 actor의 입력은

```
[ proprio(57, 순간값) , priv_explicit(6, base 선/각속도) , enc(priv_latent)(에피소드 상수) ]
```

이며, **보행 위상을 담은 항이 없다.** 유일한 시간 단서는 proprio 안의 `joint_vel(17)`과
`prev_actions(17)`인데, 정지 상태에서는 둘 다 0 근방으로 붕괴한다.

**이 배선은 버그가 아니라 RMA의 정본 설계다.** Kumar et al.의 RMA는 RL actor를 privileged latent로
학습시키고 history 경로는 사후 증류한다. `it % 20 == 0`은 Extreme Parkour 계열이 그 증류를
학습 중간에 끼워 넣는 방식이다. 따라서 이것은 "고쳐야 할 결함"이 아니라, **on-policy actor에 시간
정보를 넣는 것이 의도적인 아키텍처 선택지**라는 뜻이다 — 그래서 후보가 된다.

→ (A-1) 가설은 정정 후에도 살아남는다. 다만 **가설로 남겨둔다**. 반례가 분명하기 때문이다:
memoryless 정책도 정지에서 잘 출발하는 사례가 많고, 가우시안 탐색 잡음이 고정점을 깨며,
`prev_actions(17)`은 이미 1-step history다. 진짜 질문은 "위상을 표현할 수 있는가"가 아니라
**"정지 상태에서 학습된 평균 action이 스텝 쪽을 가리키는가"** 이고, 그것은 아래 §2.5의
보상 지형 문제다.

#### (A-1c) ★★ 관문 테스트 — **[해결됨: 가설 기각, §7-5 참조]**

램프 평가는 `get_inference_policy()` → `act_inference`(history 경로)로 돈다
(`rsl_rl/runners/on_policy_runner_parkour.py:257`). 학습은 95%가 priv 경로다.
**학습과 평가의 actor 입력이 다르다.**

그리고 history encoder가 회귀하도록 요구받는 대상은 `base_mass·base_com·stiffness·damping`,
즉 **플랜트 식별**이다. 그런데 **정지 상태에는 플랜트를 식별할 여기(excitation)가 없다.**
→ DAGGER 오차는 구조적으로 **정지 상태에서 최대**이고 보행 중에는 최소다.

이것은 상승/하강 비대칭에 대한 **완전히 다른 설명**이다. 표현력·용량·위상과 무관하다:
정지에서는 actor가 OOD latent를 받아 얼어붙고, 보행 중에는 좋은 latent를 받아 추종한다.

**테스트 (학습 불필요, 기존 체크포인트로 가능):**

같은 체크포인트에서 latent 경로만 바꿔 램프를 재측정한다. sim에서는 priv를 관측할 수 있으므로
학습 시점의 입력을 그대로 재현할 수 있다.

- 구현: `ActorCriticRMA.act_inference_priv()` 추가(평가 전용, 기존 경로 무변경)
  + `speed_ramp_record_rma.py --use_priv_latent`
- 대상: `2026-08-24_17-27-53_..._ds14_wcmd/model_49999.pt`, `--force_stand`, 4 repeat × 2 경로

| 결과 | 해석 | 함의 |
|---|---|---|
| 상승/하강 격차가 **사라진다** | 히스테리시스는 **증류 아티팩트** | 준비 중이던 아키텍처 arm 전부가 **엉뚱한 곳을 겨냥**하고 있었다. 실험 취소하고 증류를 고친다 |
| 격차가 **남는다** | 완벽한 adaptation module로도 안 된다 | 정책 구조를 볼 자격을 얻는다. arm 진행 |

★★ 부수 함의(사용자에게 보고 필요): 불일치가 실재한다면 **이 프로젝트 RMA 계열의 모든 램프 수치가
학습된 것과 다른 actor 입력에서 측정된 것**이다. 이번 실험뿐 아니라 §2의 평가 프로토콜 자체의
유효성 문제다.

#### (A-1d) ★★★ 보상 지형 — **실측 확인 완료. 두 실패를 모두 설명한다**

`leg_imitation_tracking_env.py:299-315` 실제 구현:

```python
lin_vel_err    = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
lin_vel_reward = torch.exp(-self.cfg.vel_err_scale * lin_vel_err)     # vel_err_scale = 0.5
reward         = 0.7 * lin_vel_reward + 0.3 * yaw_vel_reward + torque_penalty
```

| 기호 | 의미 | 값/단위 |
|---|---|---|
| `cmd` | 명령 선속도 | m/s |
| `vx` | 실제 body-frame 선속도 | m/s |
| `vel_err_scale` | 지수 스케일 | 0.5 |
| `lin_vel_reward_w` | 선속도 항 가중치 | 0.7 |
| `task_reward_lerp` | task 비중 (나머지는 AMP) | 0.5 |

```
lin_vel_reward = exp(-0.5 * (cmd - vx)^2)
Δtotal         = 0.5 * 0.7 * (완주 보상 - 정지 보상)
```

**(i) 저속에서 걷기 시작할 유인이 8.5배 약하다**

```
  cmd     정지 r     완주 r       Δr    Δtask   Δtotal    cmd 0.5 대비
------------------------------------------------------------------
  0.5   0.8825   1.0000   0.1175   0.0823   0.0411       1.0x
  1.0   0.6065   1.0000   0.3935   0.2754   0.1377       3.3x
  2.0   0.1353   1.0000   0.8647   0.6053   0.3026       7.4x
  3.0   0.0111   1.0000   0.9889   0.6922   0.3461       8.4x
  3.2   0.0060   1.0000   0.9940   0.6958   0.3479       8.5x
```

**cmd 0.5에서 가만히 서 있어도 이미 최대 task 보상의 88.25%를 받는다.** 걷기까지 가서 얻는 것은
step당 total 기준 **0.0411** 뿐이다. cmd 3.2의 0.3479와 비교하면 **8.5배 얕은 경사**다.
정책이 저속에서만 정지에 갇히는 것은 이것으로 충분히 설명된다.

**(ii) 과잉(overshoot)은 사실상 공짜다 — RMA run의 125~142%가 여기서 나온다**

```
cmd 0.5 기준, 완주(vx=0.5) 대비 손실 Δtotal
  vx=0.60 (120%)   0.0017
  vx=0.70 (140%)   0.0069     <- RMA run 실측 대역
  vx=0.75 (150%)   0.0108
  vx=0.00 (  0%)   0.0411     <- 정지
```

**140%로 과속하는 비용(0.0069)이 정지하는 비용(0.0411)의 1/6이다.** 즉 정책 입장에서
"천천히 정확히 걷기"보다 "적당히 빨리 걸어버리기"가 훨씬 싸다. `exp(-0.5·err²)`는 작은 오차
근방에서 지나치게 평탄해서 저속 정밀도에 사실상 보상을 안 준다.

**(iii) ★★★ 이것은 상승/하강 비대칭까지 설명한다 — 쌍안정(bistability)**

이 부분이 핵심이다. cmd 0.5에서 정지는 0.8825, 완주는 1.0000 — **둘 다 최적에 가깝다.**
이건 "한 개의 끌개(attractor)로 가는 얕은 경사"가 아니라 **거의 대등한 두 끌개가 있는 평탄한 지형**,
즉 전형적인 **쌍안정 / 경로의존(path dependence) 서명**이다.

롤아웃이 어느 분지(basin)에 들어가든 그 안에 머문다. 빠져나가는 비용이 얻는 것(0.0411)보다 크기
때문이다. **상승은 정지에서 진입 → 정지에 머문다. 하강은 보행에서 진입 → 보행에 머문다.**
이것이 관측된 히스테리시스의 방향과 정확히 일치한다.

**따라서 상승/하강 비대칭에 대한 경쟁 가설은 이제 셋이다:**

| # | 가설 | 기전 | 검증 상태 |
|---|---|---|---|
| A-1 | state aliasing | 정지 시 `joint_vel≈0` → 보행 위상 관측 불가 | **반증됨** (아래) |
| A-1c | 증류 오차 | 정지에는 플랜트 식별 여기가 없어 history latent가 OOD | **반증됨** (경계면 분석) |
| **A-1d** | **보상 gradient** | **정지 0.8825 vs 완주 1.0000 — 탈출 유인 0.0411** | **일관** |

#### ★★★ A-1(aliasing) 반증 — 비용 0의 재분석으로 끝났다

`is_arch_the_lever.md` §6. 기존 램프 npz를 **상승 구간 전체(cmd 0→4.0)** 로 다시 잘랐다
(종전 분석은 cmd 0.5 구간만 봤다). 19개 상승 롤아웃 · 5개 체크포인트 · 3개 학습 계열:

```
cmd 0.5 :  duty 0.00 ~ 1.00   (완전한 이봉, 다수 실패)
cmd 1.0 :  duty 1.00          (19/19, 예외 없음)
cmd 1.5+:  duty 1.00
```

**결정적 논리**: cmd 0.5에서 정지 출발과 cmd 1.0에서 정지 출발은 **관측상 완전히 동일**하다 —
위상 정보 부재는 명령 크기와 무관하다. aliasing이 병목이라면 cmd 1.0에서도 실패해야 한다.
그런데 19/19 성공한다. 유일하게 달라지는 것은 **정지가 얻는 task 보상 비율**이다
(88.2% → 60.7%). **표현력/관측성 가설은 직접 반증되고, 보상 gradient 가설과 정확히 일치한다.**

부수 확인: 학습 시 정지 노출이 0%였던 체크포인트(8k/25k/50k, `reset_strategy="random"`)와
정지 노출이 있었던 체크포인트(ds14_0810, wcmd_40k, `random_stand`)가 cmd 0.5에서
**구분되지 않는 동일한 불안정 패턴**을 보인다 → 정지 노출량도 1차 원인이 아니다.

#### ★★★ A-1c(증류 오차 / estimator OOD) 반증 — 명령 경계면 분석

`is_arch_the_lever.md` §7-1. 이것도 새 학습 없이 기존 데이터로 끝났다.

cmd 0.5에서 **실패해 계속 서 있던** 5개 사례를 골라, **명령이 1.0으로 바뀌는 그 순간**의 물리
상태를 봤다. 그 순간 로봇은 여전히 **완전 정지**다:

```
vx ≈ 0 ,  |jvel| RMS < 0.1 ,  플랜트 식별 여기(excitation) 전무
```

즉 history encoder 입장에서 **정확히 같은 OOD 조건**이다. 그런데 명령이 1.0이 되면
**0.5초 안에 안정적으로 걷기 시작한다**(vx 0.2~0.63, duty 1.00).

→ "정지 상태라 latent가 OOD여서 얼어붙는다"가 원인이라면 명령이 커져도 여전히 얼어붙어야 한다.
**바뀌는 것은 명령값 하나, 즉 보상 gradient 하나뿐이다.**

이 결과는 별도로 돌린 램프 A/B(§7-5)와도 방향이 같다 — 거기서도 증류를 **제거한**(priv 경로)
쪽이 오히려 더 못 걸었다.

#### AMP style 항에 대한 정정 (중요)

`is_arch_the_lever.md` §2-2가 측정 run이 실제로 쓴 데이터셋(`new_smr_leg_pkl`, 14클립)을 직접
로드해 확인: **정지(v≈0) 클립이 하나도 없다.** 최저는 `leg_walk_stmr` 0.108 m/s이고 그마저
"느린 걸음"이지 "서 있기"가 아니다. 0.6 m/s 미만 합산 가중치는 length 52.0% / command_uniform
17.7% — **저속 커버리지 자체는 나쁘지 않다. 없는 것은 "느린 걸음"이 아니라 "정지"다.**

따라서 "정지가 보상적으로 유리하다"는 단순화는 **틀렸다**. discriminator에게 정지는 off-support라
style 항은 정지를 오히려 벌준다. 올바른 서술은:

> task 항은 cmd 0.5 부근에서 걷기와 정지를 거의 구분하지 못하고(88% vs 100%), 정지를 이기게 하려면
> style 항이 그 차이를 메워야 하는데 **style 항의 폭은 명령 크기와 무관하게 대략 일정**하다.
> task 항의 폭만 명령에 비례해 커지므로, **어떤 문턱 명령값 위에서 정지→보행 전환이 신뢰성 있게
> 일어나기 시작하는 구조**가 나온다. 실측 문턱은 cmd 0.5와 1.0 사이다.

**★ 남은 가설은 A-1d 하나이고, 그 기전만이 정량적으로 확인되었다.**

**★★ 그리고 A-1d는 아키텍처로 바뀌지 않는다.** 깊이도, mixture head도, LSTM도, 배선도 없는
경사를 만들어내지 못한다. AMP가 나머지 절반을 차지하므로 참조 데이터에 정지 클립이 있는지는
별도 확인이 필요하다 (`is_arch_the_lever` 담당).

#### (A-1e) 참고 — 처음 세웠던 근사는 틀렸다 (기록용)

초기 추정은 `exp(-0.25·err²)`(정지 시 0.939)였으나 실제 구현은 `exp(-0.5·err²)`(정지 시 0.8825)다.
경사는 추정보다 조금 **가파르지만**, 결론(저속에서 정지가 보상의 대부분을 받는다)은 그대로다.

### (B) 좌우 쏠림 / 자세 대칭 — 2순위 (게이트 지표)
`project_leg_multidisc_amp_design_review`에서 달성률 승자 판정이 **철회**된 원인.
hip 좌우 합이 최신 run 0.047~0.117. **arm이 달성률을 올려도 쏠림이 나빠지면 승자가 아니다.**

### (C) 고속(cmd 3.5~4.0) 천장 — 3순위
현재 vmax32 기준 cmd 2.0~3.5에서 82~91%. 여기서의 개선은 부수적 관심사.

---

## 2. 평가 프로토콜 (실험 전에 확정, 사후 변경 금지)

**학습 곡선(`Loss/*`, `Episode_Reward/*`, `lin_ps`, `ep_len`, `noise_std`)은 판정 근거가 아니다.**
근거: `project_no_train_metric_detects_ramp_collapse` — 52k 붕괴를 네 지표 모두 놓쳤고 lin_ps는
붕괴 후 오히려 상승했다. 학습 지표는 **모니터링·중단 판단용**으로만 쓴다.

판정은 오직 실측 롤아웃으로 한다.

| # | 지표 | 도구 | 판정 |
|---|---|---|---|
| 1 | 속도 램프 달성률 (cmd별) | `_workspace/leg/speed_ramp_record.py` → `ramp_summary.py` | **≥4 repeat 평균**. 단일 run 금지 |
| 2 | cmd 0.5 **상승** duty | 위 램프의 상승 구간 `\|jvel\| RMS > 0.4` 비율 | (A) 전용 지표. 이봉 해소 = duty 분포가 단봉화 |
| 3 | cmd 0.5 **하강** 달성률 | 램프 하강 구간 | 대조군. 여기가 나빠지면 회귀 |
| 4 | hip 좌우 쏠림 | `_workspace/leg/analyze_lateral_skew.py` | **게이트**. baseline 대비 악화 시 승자 불가 |
| 5 | 보행 종류(위상차) | 접지 위상 | `project_go2_style_weight_is_the_speed_ceiling` 교훈 — 달성률만 보면 걸음 종류 회귀를 놓친다 |
| 6 | 추론 비용 | 파라미터 수 + JIT/ONNX export 후 1-step latency | 실기 배포 제약. export 불가 = 후보 탈락 |

**체크포인트 선택 규칙**: 판정은 **40k 이후 체크포인트만** 사용한다.
근거: `project_go2_implicit_actuator_generation` — "3점 연속·두 플랜트·부호 6/6" 라는 강한 근거로 낸
24k 판정이 40k에서 부호가 뒤집혔다(−41 → +6). 중간 체크포인트는 진행 모니터링용이다.

---

## 3. 실험 예산 → arm 수 = **3 + baseline**

50k iteration × (arm당 램프 ≥4회) × 40k 이후 체크포인트 판정.
이 비용 구조가 arm 수를 결정한다. 아이디어가 9개 나와도 **3개만 돌린다.**
따라서 리서치의 산출물은 "읽을거리 목록"이 아니라 **순위가 매겨진 3개 + 탈락 사유**여야 한다.

> **[결론 후 갱신]** 이 3-arm 예산은 유지하되, **아키텍처에 3개를 다 쓰지 않는다.**
> §7 결론에 따라 1번은 `vel_err_scale` 보상 arm, 2번은 (필요 시) LSTM arm 이다.
> 남는 한 자리는 1번 결과를 보고 정한다 — 미리 채우지 않는다.

---

## 4. 후보 필터 (순위 매기기 **전에** 적용하는 하드 필터)

### 필터 1 — on-policy 증거 유무
Simba / SimbaV2 / BRO 계열의 residual+normalization 스케일링은 대부분 **off-policy 또는 offline**
결과다. 사용자가 물은 질문 자체가 "그게 on-policy에서도 되냐"이므로, PPO(가급적 locomotion)
실측이 없는 후보는 **`extrapolated`로 명시**하고 순위를 내린다. 세탁하지 않는다.

### 필터 2 — export 경로 생존 여부
배포는 JIT/ONNX → 임베디드 `real_runner`. RNN/stateful, MoE/routing 계열은 export와 latency에
실제 문제가 있다. `codebase-arch-map`이 후보별 export 가부를 yes/no로 답해야 하며,
no면 그 자리에서 탈락(또는 "sim 전용 상한 측정" 으로 격하).

---

## 5. 반드시 고정해야 하는 교란 변수

용량을 바꾸면 아래가 **조용히** 같이 바뀐다. arm마다 명시적으로 고정하거나 로깅한다.

| 교란 | 왜 문제인가 | 처리 |
|---|---|---|
| adaptive LR + `desired_kl=0.01` | **이 프로젝트에서 이미 실증된 교란이다**(아래 참조) | **전 arm 상수 LR(`schedule="fixed"`)로 통일**을 기본안으로 한다 |
| `init_noise_std=0.25`, `entropy_coef=0.005` | 이 값들은 `[512,256,128]` **에 맞춰** 튜닝됐다. cfg 주석: leg는 std 1.0에서 9.98로 단조 발산 | std 궤적을 arm별 1급 지표로 로깅. 큰 net의 "실패"가 실은 std 발산일 수 있다 |
| symmetry aug `num_aug=2` | 미니배치가 2배 → 같은 KL 스케줄을 통해 용량과 상호작용 | 전 arm 동일. ON으로 통일하고 그렇게 명시 |
| 파라미터 수 vs 깊이 | "깊이+residual이 좋았다"와 "파라미터가 많아서 좋았다"가 구분 안 된다 | **파라미터 수 매칭 width-only arm을 반드시 1개 포함** |
| seed | 램프 분산이 크다 | 램프 ≥4 repeat로 흡수. arm당 seed는 동일 고정 |

### 5-1. adaptive LR 교란은 **가설이 아니라 이미 관측된 사고**다

`reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit/README.md` §15-a (커밋 `353a37647bf`):

go2 쪽에서 정책의 action std를 학습값에서 고정 0.1로 바꾼 arm(`fixedstd_stock`)이 48k/51k에서
**붕괴**했다. 원인은 std 자체가 아니었다. 고정 σ에서 KL은 대략

```
KL ~ (Δμ)^2 / (2 σ^2)
```

| 기호 | 의미 | 단위 |
|---|---|---|
| `Δμ` | update 전후 정책 평균 action 변화 | rad (action_scale 적용 전) |
| `σ`  | action 표준편차 | 동일 |
| `KL` | `desired_kl`과 비교되는 측정 KL | nats |

baseline은 σ를 초기 0.42까지 올렸다가 0.15로 내리는데, 이를 0.1로 못박으면 **같은 평균 변화가 훨씬
큰 KL로 측정**된다 → adaptive 스케줄러가 LR을 바닥값 1e-5에 고정 → surrogate 발산(0.29) → 붕괴.
`schedule="fixed"`(상수 LR 2e-4) 단독 arm은 60k를 붕괴 없이 완주해 이 진단을 뒷받침했다.

**아키텍처 변경은 정확히 같은 경로를 탄다.** 용량·정규화·residual은 모두 update당 Δμ 분포를 바꾸고,
그것이 측정 KL을 통해 LR 스케줄을 조용히 바꾼다. 이 상태에서 나온 arm 간 차이는 "아키텍처 효과"가
아니라 "우연히 걸린 LR 스케줄 차이"일 수 있다.

부수적으로 그 arm에서 상수 LR은 cmd 3.0 달성률을 **58% → 95%**로 올렸다(median vx는 1.686 vs
1.673으로 동일). 즉 상수 LR은 그 자체로 큰 효과가 있는 변수다 — **아키텍처 축과 절대 섞으면 안 된다.**

### 5-2. 설계 원칙: 축을 섞지 않는다

같은 문서 §15-c에 확립된 이 프로젝트의 실험 설계 원칙을 그대로 따른다.

1. **극단값으로 축을 먼저 소거한다.** "액추에이터 부족" 가설을 토크-속도 곡선을 통째로 제거하는
   극단 실험으로 먼저 반증했다. 아키텍처 축도 마찬가지로 **가장 극단적인 arm을 먼저** 찍어야
   "단조 관계인가, 최적점이 있는 축인가"를 알 수 있다.
2. **arm당 변경은 하나.** 위 표의 11개 arm이 전부 "변경한 것 하나" 열을 갖는다.
3. **판정은 40k 이후 4점 램프.**

---

## 6. 리서치 산출물

- [x] `research_onpolicy_scaling.md` — on-policy/PPO locomotion 아키텍처 서베이
- [ ] `research_depth_papers.md` — 1000-layer 논문 + BC mystery 정독 및 on-policy 전이 비판
- [ ] `codebase_arch_map.md` — rsl_rl 아키텍처 seam 지도 + export 가부
- [ ] `is_arch_the_lever.md` — 보상/데이터가 진짜 병목인지 판정

### 6-1. ★★★ 원 질문에 대한 답: 깊이 스케일링은 PPO로 **전이되지 않는다**

사용자가 인용한 두 소스를 정독한 결과, 전제부터 틀렸다.

**(1) 1000-layer 논문은 offline RL 이야기가 아니다.**
`arXiv:2503.14858` (Wang et al., NeurIPS 2025). 알고리즘은 **Contrastive RL** — replay buffer를 쓰는
**off-policy online** 방식이고, value function 자리에 InfoNCE **분류기**가 들어간다.
구조는 4층(Dense→LayerNorm→Swish) 블록의 residual 스택, depth 4~1024, Swish, LayerNorm 필수
(빼면 스케일링 이득이 붕괴). depth 4→64에서 Humanoid 12.6→649 (52배).

★ 그런데 **같은 논문 안에서 SAC·TD3는 4층 넘어가면 이득이 없거나 악화**한다. 그리고
**offline 설정에서는 저자들이 직접 "이득 없음"을 보고**한다. 즉 이 결과는
"offline/self-supervised는 되고 online RL은 안 된다"가 아니라, **분류형(InfoNCE) 목적함수에 특유**하고
**TD 부트스트랩 목적함수에는 안 통한다**는 것이다. PPO의 critic은 GAE 기반 TD 회귀다 —
논문 자체 실험에서 이미 실패한 쪽 계열이다.

**(2) PPO는 깊게 쌓으면 그냥 붕괴한다 — 직접 실측된 반례가 있다.**
`arXiv:2401.16025` (Simple Policy Optimization). MuJoCo에서 vanilla PPO를 3층 vs 7층으로 비교:

```
                     3층      7층
Ant-v4            5323  ->   1003
HalfCheetah-v4    4550  ->   2242
Hopper-v4         1119  ->    976
HalfCheetah ratio 편차 최대   0.225 -> 1675
```

기전: PPO의 clip은 클리핑된 샘플의 gradient를 **0으로 만든다**. 한 번 밖으로 나간 점은 되돌릴 힘을
못 받는데, 표현력이 큰 네트워크일수록 ratio가 신뢰영역 밖으로 멀리 표류한다. clip을 연속 페널티로
바꾼 SPO는 7층에서 3층 PPO보다도 좋아진다(Ant 4672). 보완 진단으로
`arXiv:2405.00662`(No Representation, No Trust)는 표현 rank 붕괴가 같은 실패를 만든다고 보고한다.

**(3) BC mystery 글은 표현력 주장을 하지 않는다.**
`seohong.me/blog/behavioral-cloning-mystery/`를 재확인한 결과, 다중모드(multimodality)라는 말이
**아예 없다**. 데이터를 오히려 "narrowly distributed, highly temporally correlated"라고 기술한다.
저자 본인의 결론은 *"I don't have a great answer for this mystery"* 이고 후보로 든 것은
"BC가 원래 어렵다"(용량)와 "flow BC가 비효율적이다"(최적화) 둘뿐이다. 깊이 vs 폭 ablation도 없다
(`[4096]×8`은 둘을 섞은 값). 표현력 주장은 이 글이 아니라 그의 **논문 FQL**에 있다.

**(4) 깊이와 표현력은 대체재가 아니라 직교축이다.**
대각 가우시안의 밀도는 **깊이와 무관하게 항상 단봉**이다. 층을 더 쌓으면 "관측→분포 파라미터"
사상이 날카로워질 뿐, **한 관측에서의 출력 분포가 이봉이 되지는 않는다.** 이건 최적화 한계가 아니라
분포족(distribution family)의 구조적 한계다.

### 6-2. 다중모드 vs 부분관측 — 그리고 **둘 다 아니었다**

리서치 단계에서 두 에이전트가 독립적으로 "이건 다중모드가 아니라 **부분관측**(state aliasing)"
이라는 결론에 도달했다. 그 구분 자체는 유효하고 기록해 둘 가치가 있다:

- 부분관측이면 → mixture/flow/diffusion head는 **도움이 안 된다**. 같은 빈약한 관측에 조건부인
  다봉 분포는 애매한 지점에서 stand/walk를 **동전 던지기**할 뿐이다. 필요한 건 메모리/위상.
- 진짜 다중모드면 → 깊이는 구조적으로 무력하고 mixture head가 맞는 레버.

**★ 다만 이후 §A-1 반증으로 부분관측 가설도 탈락했다.** cmd 1.0에서 정지 출발은 cmd 0.5와
관측상 완전히 동일한데 19/19 성공한다. 따라서 이 절의 두 분기 중 어느 쪽도 우리 실패가 아니며,
**mixture head도 메모리/위상도 이 실패 모드의 처방이 아니다.**
이 절은 "왜 표현력 계열 후보를 전부 탈락시켰는가"의 근거로만 읽을 것.

### 6-3. RMA latent를 고쳐도 위상은 안 생긴다 (에이전트 확인)

RMA 원논문(`arXiv:2107.04034`)의 extrinsics는 질량·모터강도·마찰·지형높이 = **전부 준정적 환경
파라미터**다. Extreme Parkour, Rapid Locomotion(`arXiv:2205.02824`)도 동일하다.
history encoder는 일관되게 **system-ID 대체물**이지 gait phase 운반체가 아니다.
→ §A-1c의 증류 불일치를 고치면 마찰·질량 추정이 좋아질 뿐, **위상 인지 actor가 되지는 않는다.**

### 6-4. 실제 배포된 legged PPO는 무엇을 쓰는가

| 시스템 | actor 구조 |
|---|---|
| rsl_rl / IsaacLab 기본 | MLP `[256,256,256]` 또는 `[512,256,128]` ELU, 정규화·residual 없음 |
| Extreme/Humanoid Parkour | MLP `[512,256,128]` CELU + GRU(256, 1층) — **우리와 동일한 폭** |
| Walk These Ways | MLP + 별도 estimator `[256,128]` ELU |
| Cassie (Siekmann, RSS 2020) | **LSTM** actor/critic, PPO, 실기 |
| Unitree H1_2 / G1 | MLP `[256,128,64]` tanh / `[512,256,128]`, ONNX 50 Hz |
| Booster T1 | asymmetric actor-critic, JIT 온보드 CPU 50 Hz |
| DreamWaQ | MLP actor + GRU 기반 estimator |

**deep residual·transformer·MoE를 core policy로 쓴 실기 배포 사례가 하나도 없다.**
plain MLP에서 벗어난 유일한 일관된 패턴은 **작은 recurrent unit**이다.

### 6-5. ★ 저속 실패가 아키텍처 문제가 아닐 수 있는 근거 (반드시 병행 확인)

- **정지는 알려진 local optimum이다.** energy·action smoothness·joint velocity 페널티는 전부
  무운동에서 최소화된다. 이 분야의 표준 처방은 **페널티 가중치 커리큘럼**이지 네트워크 변경이 아니다.
- **저속 자체가 학습하기 어렵다**는 보고가 별도로 있다(저속에서 pitch forward, local minima).
- **극한주기 보행에는 물리적 최저 속도 하한이 있다.** 임계 아래에서는 limit cycle 자체가 불안정하다.
  형태학적 하한이 0.5 m/s 근처라면 **어떤 아키텍처도 못 고친다.**
- **AMP 참조 데이터 커버리지.** 고속 쪽에서는 "sprinting 참조 부족 → 학습 불안정"이 문헌에 있다
  (저속 쪽은 대칭 추론이며 직접 인용 아님).

→ 에이전트의 정직한 결론: 예비 관측의 **과잉(125~142%)은 아키텍처 arm으로 설명하지 말 것.**
onset(걷기 시작하는가)과 overshoot(천천히 걸을 수 있는가)은 **다른 실패**이고, 후자는 보상/데이터
쪽 조사 트랙으로 **병행**해야 한다. 아니면 한쪽의 null 결과가 다른 쪽에 오귀인된다.

### 6-6. 배포 제약 실측 하나

IsaacLab exporter(`isaaclab_rl/rsl_rl/exporter.py`)는 LSTM의 (hidden, cell) 튜플 전달을
하드코딩해서 **GRU export가 깨진다** — isaac-sim/IsaacLab issue #3008, 미해결.
→ recurrent arm을 간다면 **`rnn_type="lstm"`**. GRU를 쓰려면 exporter부터 고쳐야 한다.

## 7. 최종 결정 — **깊이 스케일링 스윕은 하지 않는다. 순서를 바꾼다.**

### 7-1. 원 질문에 대한 답

> "1000 Layer Networks / BC mystery 에서 말하는 모델 구조를 키우는 이야기를 on-policy PPO
>  locomotion 에 가져오면 되는가?"

**전이되지 않는다.** 근거 세 줄:

1. **1000-layer 결과는 offline RL 이야기가 아니라 InfoNCE(분류형 목적함수) 이야기다.**
   같은 논문 안에서 SAC·TD3는 4층 넘으면 이득 없거나 악화하고, offline 설정은 저자들이 직접
   "이득 없음"을 보고한다. PPO critic은 TD 회귀 — 이미 실패한 쪽 계열이다. (§6-1)
2. **PPO는 깊게 쌓으면 실측으로 붕괴한다.** Ant-v4 3층 5323 → 7층 **1003**
   (`arXiv:2401.16025`). 기전은 clip이 클리핑된 샘플의 gradient를 0으로 만들어 ratio가
   신뢰영역 밖으로 표류하는 것. 표현력이 클수록 심해진다.
3. **실기 배포된 legged PPO 중 deep residual / transformer / MoE 를 core policy 로 쓴 사례가
   하나도 없다.** plain MLP에서 벗어난 유일한 일관된 패턴은 **작은 recurrent unit**이다. (§6-4)

덧붙여 **BC mystery 글은 표현력 주장을 하지 않는다.** 다중모드라는 말이 글에 없고, 저자 본인이
*"I don't have a great answer for this mystery"* 라고 쓴다. 그리고 깊이와 표현력은 **직교축**이다 —
대각 가우시안은 깊이와 무관하게 항상 단봉이다. (§6-1)

### 7-2. 그리고 지금 겨냥하려던 실패는 **보상 모양** 문제다

cmd 0.5에서 정지 0.8825 vs 완주 1.0000. 탈출 유인은 step당 total 0.0411이고, cmd 3.2의
0.3479 대비 **8.5배 얕다**. 게다가 140% 과속 비용(0.0069)이 정지 비용의 1/6이라 "천천히 정확히"보다
"빨리 걸어버리기"가 싸다. 이건 두 개의 거의 대등한 끌개 = **쌍안정**이고, 상승/하강 비대칭의
방향과 일치한다. (§A-1d)

★★ **함의**: 정지가 최대 task 보상의 88%를 받는 상태에서 아키텍처 arm을 cmd 0.5 달성률로 평가하면,
**목적함수 자체가 두 행동을 거의 구분하지 못하는 지표로 측정하는 것**이다. LSTM이 위상 표현을
진짜로 고쳐도 오를 경사가 없어서 null 이 나올 수 있다. 이 순서로는 3×50k iteration을 써서
**해석 불가능한 숫자**를 얻는다.

### 7-3. 권고 순서

**1단계 (먼저) — `vel_err_scale` 보상 arm. 정량 예측이 딸려 있다.**

`is_arch_the_lever.md` §9-1의 예측: `vel_err_scale` 0.5 → **1.5** 로 올리면

```
cmd 0.5, 정지:  exp(-1.5 * 0.25) = 0.687
```

즉 cmd 0.5에서 정지가 얻는 보상 비율이, **이미 19/19 신뢰성 있게 개시가 일어나는 cmd 1.0 수준
(현행 스케일에서 0.607)** 까지 떨어진다. 예측이 맞으면 **cmd 0.5 duty의 이봉이 좁아져야 한다.**
cfg 한 줄 변경에 반증 가능한 정량 예측이 붙은, 이 시리즈에서 가장 값싼 arm이다.
부작용으로 고속 추종 보상도 뾰족해지므로 전 명령대를 §6과 같은 방법으로 함께 확인한다.

아키텍처가 아니라 보상 축이다. 형제 task 에 **이미 측정된 선례**가 있다:
go2 arm #7 `velscale1_stock` 이 `vel_err_scale` 0.5 → 1.0 으로 cmd 2.5 를 1.50 → 1.87 로 올렸고
대신 cmd 3.5+ 가 0% 가 됐다(`reports/go2_imitation/_comparisons/mimickit_vs_60_actuator_limit`
§15-c). 한 축만 움직인 실측 trade-off 가 이미 있으므로 가장 싸고 근거가 좋다.
leg 에서는 저속 분해능이 목표이므로 상향이 맞는 방향이며, 고속 손실을 램프로 확인한다.

**2단계 (그 다음) — 필요하면 아키텍처 축, LSTM 하나.**
recurrent 가 유일하게 on-policy locomotion **실기 선례**가 있는 후보다
(Cassie/RSS 2020, ZSL-RPPO `arXiv:2403.01928`, Gait-Conditioned RL `arXiv:2505.20619` — 후자는
Unitree G1 실기에서 stand↔walk 전이를 LSTM+MLP 로 보고). **`rnn_type="lstm"` 고정** —
GRU 는 IsaacLab exporter 미해결 버그(issue #3008)로 export 가 깨진다. (§6-6)
단, **1단계로 경사를 만든 뒤에** 돌려야 신호가 나온다.

**병행 (학습 불필요) — 보상/데이터 감사.**
AMP 참조 데이터의 정지·저속 클립 커버리지, 페널티 항 커리큘럼 유무,
그리고 0.5 m/s 가 이 형태학의 안정 극한주기 범위 안에 있는지. (§6-5)

### 7-4. 하지 않기로 한 것과 이유

| 후보 | 탈락 사유 |
|---|---|
| deep/wide residual MLP (SimBa·BRO·SimbaV2) | on-policy locomotion 증거 없음. PPO 는 깊이 7에서 붕괴 실측 |
| transformer actor | 추론 지연이 50 Hz 임베디드 목표와 정면 충돌. 문헌도 배포 전 distill 로 되돌림 |
| MoE | 입증된 이득은 **multitask** gradient conflict 해소. 단일 task 인 우리에겐 근거 약함 + 분기 추론 비용 |
| mixture-of-Gaussians head | 부분관측이면 애매 지점에서 stand/walk 를 동전 던지기 할 뿐. 문헌상 on-policy 수치 불안정 |
| diffusion / flow policy | PPO ratio 에 필요한 밀도가 비싸거나 부재. multi-step 샘플링은 50 Hz 임베디드 불가 |
| 파라미터 수 매칭 width-only arm | 용량이 축이라는 전제 위의 대조군이었는데, 그 전제가 §7-2 로 무너짐 |

### 7-4b. ★ 부수 발견 — 최신 run만 `random_stand`가 빠졌다 (재현성 플래그)

판정을 바꾸지는 않지만 별도로 처리해야 할 문제다. `params/env.yaml` 실측:

```
2026-08-10 ... _stand_dz_ds14              reset_strategy: random_stand
2026-08-11 ... _stand_dz_run2t14           reset_strategy: random_stand
2026-08-12 ... _stand_dz_ds14_vmax32       reset_strategy: random_stand
2026-08-14 ... _stand_dz_vmax32_trot110150 reset_strategy: random_stand
2026-08-18 ... _trot110150_wcmd            reset_strategy: random_stand
2026-08-18 ... _waistfix                   reset_strategy: random_stand
2026-08-22 ... _waistfix_wcmd              reset_strategy: random_stand
2026-08-24 ... _ds14_wcmd                  reset_strategy: random        <- 여기서만
```

**run 이름에는 여전히 `stand_dz`가 들어 있는데 실제 설정은 `random`이다.**

원인은 cfg 편집이 아니다. `leg_imitation_tracking_env_cfg.py`의 기본값은 HEAD와 워킹트리 **양쪽 다
`"random"`** 이다. 즉 08-10~08-22 run들은 CLI/hydra override로 `random_stand`를 켰던 것이고,
**08-24 run은 그 플래그를 빠뜨렸다.** (env.yaml은 해석된 최종 설정을 기록하므로 override가 있었다면
그대로 찍힌다.)

그리고 `"stand" in "random"` 이 `False`이므로 `rel_stand_envs=0.1`은 **완전히 무력**하다
(`_reset_idx`의 분기 전체가 스킵된다). RSI 소스인 모션 클립에도 정지 프레임이 없으므로
**이 run은 학습 중 정지 출발을 0% 경험했다.**

이는 이 프로젝트의 기존 함정과 같은 계열이다 —
`project_ramp_uses_source_cfg_not_run_params`(`--no_pace` 누락이 조용한 cross-plant 측정이 된 사고).
**run 이름이 설정을 보증하지 않는다.** 처방: 학습 스크립트가 run 이름의 토큰과 해석된 cfg를
대조해 불일치 시 경고하거나, 이 축들을 cfg 기본값으로 승격시킬 것.

다만 **판정은 바뀌지 않는다**: `random_stand`가 실제로 켜져 있던 run들도 cmd 0.5 duty는
0.05~0.98로 똑같이 불안정했다(§A-1 반증 표).

### 7-5. ★★ A/B 최종 결과 (4 repeat × 2 경로, 완료) — **격차가 남는다**

원자료 `metrics/priv_vs_history_ramp.md`.

```
             cmd 0.5              cmd 1.0        cmd 2.0        cmd 3.0
             duty   ach           duty   ach     duty   ach     duty   ach
history  up  0.54   37%           1.00   75%     1.00   96%     1.00   87%
        down 1.00   71%           1.00   69%     1.00   95%     1.00   87%
priv     up  0.21   17%           1.00   71%     1.00   94%     1.00   86%
        down 0.99   67%           1.00   70%     1.00   97%     1.00   86%
```

**(1) 증류 오차 가설(A-1c) 최종 기각.**
가설은 "정지에서 history latent가 OOD라 얼어붙는다"였다. 그렇다면 학습 시점 입력을 그대로 준
priv 경로에서 **격차가 사라져야** 한다. 사라지지 않는다 — 오히려 priv 쪽 상승 duty가 0.21로
history의 0.54보다 **더 나쁘다.** 완벽한 adaptation module로도 히스테리시스는 남는다.
(예비 n=1에서 hist up 0.99가 나왔던 것은 이봉 분포의 한쪽 극단을 뽑은 것이었다 — **4 repeat 규약이
왜 필요한지 보여주는 사례.**)

**(2) 실패는 cmd 0.5에만 있다 — latent 경로와 무관하게.**
cmd ≥ 1.0에서는 **두 경로 × 두 방향 전부 duty 1.00**(각 16/16). §A-1 반증의 19 롤아웃 결과와
독립적으로 재현됐다. 관측 조건도 latent 경로도 아닌, **명령값 하나만이 성패를 가른다.**

**(3) 부수 확인 — train/eval 불일치의 실제 크기는 작다.**
cmd ≥ 1.0에서 두 경로의 달성률 차이는 1~4%p에 그친다(예: cmd 2.0에서 96% vs 94%).
즉 **여기가 있는 상황에서는 증류가 잘 되고 있다.** 따라서 §7-6에서 우려했던
"기존 RMA 램프 수치 전부 재검토" 는 **불필요하다** — 불일치는 정지 근방에서만 유의하고,
그 구간은 어차피 두 경로 모두 실패한다.

**결론: 아키텍처 쪽 가설 3개가 모두 반증됐고, 보상 gradient 하나만 남는다.**

### 7-5b. 영상 — `videos/priv_vs_history_cmd05.mp4` (35초, 좌우 비교)

같은 체크포인트(`model_49999`)로 latent 경로만 바꿔 다시 렌더했다. 측정(§7-5)은 `--no_video`로
돌려 mp4가 없었기 때문이다. 명령 프로파일은 `0 → 0.5 → 1.0 → 1.5 → 1.0 → 0.5 → 0`으로 줄였다 —
실패 구간(cmd 0.5 상승)·성공 구간(cmd 1.0/1.5)·성공하는 하강 0.5를 모두 담으면서 전체 램프보다
2.4배 짧다. **따라서 이 영상은 예시이지 §7-5 측정의 재현이 아니다**(측정은 `vx_max 4.0`).

★★ **해석 주의 — 이 렌더 쌍은 측정 평균과 방향이 반대다.**

```
              상승 duty        측정 4-repeat 평균
history        0.00   (선다)        0.54
priv           0.80   (걷는다)      0.21
```

cmd 0.5는 이봉 분포라 **한 번의 롤아웃이 어느 쪽으로 떨어질지는 경로가 결정하지 않는다.**
이 쌍이 반대로 나온 것 자체가 그 증거다. 영상 라벨에 경로뿐 아니라 **그 롤아웃의 실제 결과**를
같이 적은 이유이며, "history는 걷고 priv는 선다"로 읽으면 안 된다.

영상이 실제로 보여주는 것은 §7-5의 핵심과 같다: **같은 정책·같은 명령인데 결과가 갈리고,
cmd 1.0 이상에서는 두 클립 모두 예외 없이 걷는다.** 즉 성패를 가르는 것은 latent 경로가 아니라
명령값이다.

생성: `_workspace/leg/make_ab_sidebyside.py` (원본 클립 `_workspace/leg/ab_video/{hist,priv}/`)

> 구현 메모 2건 — 번들 `imageio_ffmpeg` 빌드에는 `drawtext`가 없어(libfreetype 미포함) 자막을
> 못 굽는다. PIL로 배너 PNG를 그려 `vstack`으로 얹었다. 그리고 `-loop 1` 로 넣은 배너는
> `vstack=...:shortest=1` 이 없으면 끝나지 않아 **3시간 48분짜리 58 MB 파일**이 나온다(실측).

### 7-6. A/B 방법 메모

- **비교 가능성**: 두 팔 모두 `--force_stand`로 돌렸는데, 기존 램프 npz의 `reset_strategy`
  메타데이터를 읽어보면 `ds14_vmax32_final_1` 등도 전부 `random_stand`다.
  **기존 측정과 동일한 조건**이므로 §1 (A)의 표와 교차 비교해도 된다.
- **`--no_video`**: A/B용으로 `speed_ramp_record_rma.py`에 추가했다. 렌더가 램프 비용을 지배해
  run당 45분 → 14분이 됐다. npz만 필요할 때 쓴다. (카메라를 끄면 viewport 확장이 로드되지 않아
  `omni.kit.viewport` import를 조건부로 바꿔야 했다.)
- **`nan_to_num` 가드**: `codebase_arch_map.md` §5.4 지적대로 `act_inference_priv`가 학습 경로
  (`act`, line 301)의 `torch.nan_to_num`을 빠뜨렸었다. 이번 A/B에는 영향이 없었고
  (비교 대상 `act_inference`도 이 호출이 없어 **두 팔이 대칭**이었고, 완료된 롤아웃의
  `vx/jvel/jpos/jtau`에서 **NaN이 0건**), 스윕 중 코드를 바꾸면 repeat 간 조건이 달라지므로
  **스윕 완료 후** 추가했다. 이후 실행은 학습 입력을 더 정확히 재현한다.

### 7-7. 실행 — `vel_err_scale` 축 착수 (2026-08-26)

**3점 곡선으로 간다.** §5-2의 설계 원칙("단조 관계인가 최적점이 있는 축인가를 알려면 끝점을
찍어야 한다" — go2 `lerp` 축에서 0.5/0.6/0.8/1.0을 채워 최적점을 찾은 방식)을 그대로 따른다.

| arm | `vel_err_scale` | run | GPU |
|---|---|---|---|
| baseline | 0.5 | `2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd` (기존) | — |
| A | **1.0** | `2026-08-26_..._velscale10_ds14_wcmd_vmax32` | 3 |
| B | **1.5** | `2026-08-26_16-51-43_velscale15_ds14_wcmd_vmax32` | 2 |

cmd 0.5에서 정지가 얻는 task 보상 비율:

```
vel_err_scale   정지 r = exp(-k * 0.25)    현행 스케일 기준 등가 명령
    0.5 (base)         0.8825                    cmd 0.5
    1.0 (arm A)        0.7788                    cmd ~0.7
    1.5 (arm B)        0.6873                    cmd ~1.0  <- 19/19 성공 구간
```

arm A는 **중간점**이라 "얼마나 올려야 충분한가"를 알려주고, 곡선이 단조인지 최적점이 있는지도
두 점이 있어야 구분된다. 둘 다 50k iter, `--video` 없음.

**축 하나만 움직였음을 전수 대조로 확인했다.** baseline run(`2026-08-24_17-27-53_...ds14_wcmd`)의
`env.yaml`과 현재 소스 cfg를 필드 단위로 비교한 결과, 표현식-vs-평가값 차이를 걷어내면 실제로
다른 항목은 **3개뿐**이었고(`motion_file` / `motion_weight_mode` / `lin_vel_x_max`) 전부 baseline
값으로 되돌렸다. 따라서 새로 움직인 것은 `vel_err_scale` 하나다.

```
--rl_library rsl_rl --task Leg-Imitation-Tracking-RMA-v0 --headless
--run_name velscale15_ds14_wcmd_vmax32
env.motion_file=<...>/imitation/new_smr_leg_pkl
env.motion_weight_mode=command_uniform
env.lin_vel_x_max=3.2
env.vel_err_scale=1.5
```

dump된 `params/env.yaml` 실측 확인: `vel_err_scale: 1.5`, `motion_weight_mode: command_uniform`,
`reset_strategy: random`, `rel_stand_envs: 0.1`, `lin_vel_x_max: 3.2`, `torque_penalty_w: 0.0`,
`yaw_vel_err_scale: 0.5`(미변경 — 선속도 스케일만 움직였다).
`params/agent.yaml`은 baseline과 **`max_iterations` 외 완전 일치**(symmetry aug, `task_reward_lerp`
0.5, `schedule: adaptive`, `seed: 1` 포함).

**의도적으로 함께 바꾸지 않은 것**: §5-1의 adaptive LR 교란이 알려져 있지만, baseline이 adaptive
이므로 여기서 상수 LR로 바꾸면 축이 둘이 된다. 대신 **arm별 KL·LR trace를 판정 시 함께 비교**한다.

**판정 기준** (§2 규약): 40k 이후 체크포인트, 램프 ≥4 repeat,
`ramp_onset_duty.py`로 cmd 0.5 **상승** duty 분포가 좁아지는지(현행 0.00~1.00 → 0.8 이상 수렴).
고속(cmd 2.0~3.2) 달성률 손실과 hip 좌우 쏠림을 게이트로 함께 본다.

**예측**: cmd 0.5에서 정지가 얻는 보상이 0.8825 → `exp(-1.5·0.25)` = **0.687** 로 떨어져,
이미 19/19 성공하는 cmd 1.0 수준(현행 스케일 0.607)에 근접한다. 이 예측이 빗나가면
(duty가 여전히 이봉이면) 보상 gradient 가설이 반증되고, 그때 §7-3의 2단계(LSTM)로 넘어간다.

### 7-7b. ★ 중간 점검 (10k) — adaptive LR 교란이 **실측으로 잡혔다**

두 arm 모두 10k 부근에서 건강하다(에러 0, `ep_len` 499/499 = 만점, `lin_vel_reward`가 running max,
`noise_std` 0.37/0.39 → 0.23/0.24로 **수렴** — leg의 알려진 std 발산 실패 모드는 안 나타났다).

그런데 §5-1에서 "가설이 아니라 이미 관측된 사고"라고 적었던 adaptive LR 교란이 실제로 측정된다.
**같은 0~10k 구간**으로 잘라 baseline과 비교:

| arm | `vel_err_scale` | LR median | LR 바닥(1e-5) 체류 | KL median |
|---|---|---|---|---|
| baseline | 0.5 | 5.06e-5 | 7.7% | 0.0136 |
| arm A | 1.0 | **3.37e-5** | **14.9%** | 0.0133 |
| arm B | 1.5 | **3.37e-5** | **10.9%** | 0.0133 |

**KL은 셋 다 0.0133~0.0136으로 사실상 동일하다** — 스케줄러가 `desired_kl=0.01` 근처를 지키느라
LR을 깎고 있다는 뜻이다. 기전은 명확하다: 보상을 뾰족하게 만들면 advantage 크기가 커지고,
같은 LR에서 정책이 더 크게 움직여 측정 KL이 올라가므로, 스케줄러가 LR을 낮춰 상쇄한다.

→ **두 arm은 baseline보다 실효 LR이 약 2/3, 바닥 체류는 1.4~1.9배다.**

★ **판정 시 반드시 고려할 것**: arm이 baseline보다 나쁘게 나오면 그것이 보상 모양 때문인지
**낮아진 실효 LR 때문인지 구분되지 않는다.** 두 해석을 가르려면 필요 시 상수 LR 대조군
(`agent.algorithm.schedule=fixed`)을 추가해야 한다 — 다만 그건 별도 축이므로 지금 arm에는 섞지 않는다.

☑ 안심 근거: go2 붕괴 사례는 바닥 체류 **77%** 였다. 10~15%는 그 수준이 아니고 KL도 목표에서
안 벗어난다. 즉 **주의 항목이지 중단 사유는 아니다.**

⚠ 관찰(10k 시점): arm A의 바닥 체류가 사분위별로 11.4 → 14.4 → 15.9 → **17.8%** 로 단조 증가.
"40k 이후에도 이어지면 다시 본다"로 남겨뒀다.

**☑ 해소(20k → 30k 시점)**: 5k 구간으로 다시 자르니 **단조 증가가 아니라 요동이었고**, 30k까지
이어보면 두 arm 다 정점을 지나 내려온다.

```
arm                0-5k    5-10k   10-15k   15-20k   20-25k   25-30k
baseline(0.5)      7.7%     7.6%     9.5%     8.3%
armA(1.0)         12.8%    17.0%    16.8%    14.3%    14.1%    13.2%   <- 정점 후 하강
armB(1.5)         10.6%    11.2%    11.5%    11.7%    11.2%    10.7%
```

★ **진행 중인 run 의 마지막 구간은 항상 미완성이다.** 20k 시점에 잰 armB 15-20k 는 12.1% 였는데,
구간이 다 차고 나니 11.7% 였다(n=5000 확정). 살아 있는 run 을 구간 통계로 볼 때는 **마지막 칸을
빼고 읽어야 한다** — 그러지 않으면 없는 추세가 보인다.

두 arm 모두 baseline보다 높지만(arm A ~1.7배, arm B ~1.4배) **안정적이고 go2 붕괴 영역(77%)으로
표류하지 않는다.** §7-7b의 "실효 LR이 낮다"는 판정 시 고려사항은 유지하되, 추세 우려는 철회한다.

20k 건강 상태(최근 5k 구간 vs 그 이전 running max):

| 지표 | arm A (1.0) | arm B (1.5) | 판정 |
|---|---|---|---|
| `lin_vel_reward` | 47.049 / prior 47.144 | 46.675 / prior 46.681 | 정체 아님, running max 부근 |
| `mean_reward` | **385.7** / prior 384.3 | **386.4** / prior 382.9 | 여전히 신기록 갱신 중 |
| `noise_std` | 0.233 (0.386→) | 0.229 (0.371→) | 수렴. 발산 모드 없음 |
| `ep_len` | 499 (min 460.9) | 499 (min 458.3) | 만점 |
| `surrogate` | 최근 max 1.049 / prior 3.207 | 최근 max 0.699 / prior 3.348 | 스파이크가 오히려 **감소** |

30k 건강 상태 — 위 표는 arm 마다 "최근 5k"의 끝점이 달라 기준이 어긋난다. **두 arm 을 같은 구간
(25k 이후)의 같은 통계로 다시 쟀다**(iter 33.8k / 30.6k 시점):

| 지표 (25k 이후) | arm A (1.0) | arm B (1.5) |
|---|---|---|
| `mean_reward` 평균 / 최대 | 379.1 / 398.7 | 379.1 / 397.4 |
| `ep_len` 평균 / 최소 | 495.9 / 460.6 | 496.8 / 454.6 |

★ **두 arm 의 학습 지표는 30k 에서 구분되지 않는다** — `mean_reward` 평균이 소수점 첫째 자리까지
같다. `ep_len` 도 둘 다 만점 499 를 자주 찍되 455~460 까지 내려가는 요동을 공유하므로, 한쪽만
조기 종료가 늘었다는 신호는 없다. 이 실험의 판정은 학습 지표가 아니라 램프에서 나온다(§7-7).

### 7-7c. ★★★ 40k 판정 — `vel_err_scale=1.5` 가 이봉을 **없앴다**, 고속 손실 없이

세 팔 모두 **`model_40000` 매칭 체크포인트**, 4 repeat, 동일 플래그(`--force_stand --heading_hold
--headless --no_video`). 체크포인트를 절대 iter 로 고정한 이유는 GPU 경합으로 arm 들이 서로 다른
속도로 돌아 "각자 최신"끼리 비교하면 기준이 자동으로 어긋나기 때문이다(§7-7b 의 교훈).

**게이트 1 — cmd 0.5 개시 이봉** (이 실험이 겨냥한 것):

| arm | 상승 duty | spread | 롤아웃별 | 달성률 |
|---|---|---|---|---|
| baseline 0.5 | 0.54 | 0.88 | 0.00 · 0.63 · 0.65 · 0.88 | 29% |
| arm A 1.0 | 0.50 | **1.00** | 0.00 · 0.00 · 1.00 · 1.00 | 43% |
| **arm B 1.5** | **1.00** | **0.00** | 1.00 · 1.00 · 1.00 · 1.00 | **79%** |

§7-7 의 예측은 "이봉이 좁아진다" 였는데 arm B 는 **소멸**시켰다. 4/4 전부 정지→보행 전이에 성공한다.

**게이트 2·3 — 고속 손실 / 자세 대칭**:

| arm | 0.5 | 2.0 | 3.0 | 4.0 | 평균 | 쏠림 | 변위 | 허리 |
|---|---|---|---|---|---|---|---|---|
| baseline 0.5 | 18% | 95% | 87% | 82% | 71.4% | 0.117 | 134.2 m | 0.43 |
| arm A 1.0 | 40% | 100% | 92% | 42%✗ | 82.9%✗ | 0.131 | 73.3 m✗ | 0.78 |
| **arm B 1.5** | **68%** | 98% | **97%** | **89%** | **88.8%** | **0.092** | **149.9 m** | **0.28** |

**트레이드오프가 없다** — arm B 는 전 명령에서 baseline 이상이고 쏠림·허리 비틀림도 최소다.
이 프로젝트에서 5회 반복되던 "한쪽 올리면 다른 쪽 내려간다" 가 이번엔 나타나지 않았다.

★ **arm A(1.0) 는 4/4 붕괴한다.** 정점(cmd 4.0, t=42.0s) 도달 직후 t=42.0\~50.2s 에 넘어져
못 일어난다 — 종료 시 thigh/calf 가 −1.0 rad 로 접히고 yaw 가 +84°/−108° 로 돌아간다.
정점 이후 전진이 0.01\~28.3 m(baseline 72.8\~77.1, arm B 81.5\~82.4). ✗ 표시 수치는 이 붕괴에
오염된 값이라 **능력으로 읽으면 안 된다**. 세 팔 모두 학습 상한 `lin_vel_x_max=3.2` 라 램프
정점 4.0 은 외삽 구간이고, 붕괴는 정확히 거기서 난다. 해석: 보상을 뾰족하게 하면 추종이
공격적이 되어 최고속이 오르는데(arm A 정점 3.57 vs baseline 3.25 m/s) 1.0 은 그 공격성을
감당할 안정성을 못 얻었다. 1.5 는 둘 다 얻었다.

⚠️ **0.5 → 1.0 → 1.5 가 단조가 아니다.** 중간값이 가장 나쁘다. "1.0 이라는 값이 나쁘다" 인지
"arm A 의 40k 체크포인트가 나쁘다" 인지는 **체크포인트 하나로 못 가른다** — 50k 재확인 필요.

⚠️ **학습 지표는 이 차이를 전혀 못 봤다.** 같은 시점 두 arm 의 `mean_reward` 평균이 379.1 로
소수점 첫째 자리까지 같고 `ep_len` 도 495.9/496.8 이었다(§7-7b). 그런데 램프에서는 한쪽이 4/4
붕괴, 한쪽이 역대 최고다. [[project_no_train_metric_detects_ramp_collapse]] 의 재확인 사례다.

☑ **플래그 교란 배제(확정)**: baseline 기존 데이터는 영상 켜고, arm 들은 `--no_video` 로 쟀다.
같은 체크포인트를 `--no_video` 로 다시 잰 대조군 `basenv_it40000_*` 이 **4/4 정상**이고 정점 이후
74.1\~78.2 m 로 영상 켠 쪽(72.8\~77.1 m)과 같다. 플래그는 넘어짐과 무관하며, arm A 의 붕괴는 실제다.
(arm B 도 같은 플래그로 재고 4/4 정상인 것이 1차 반증이었다.)

### 7-7d. ★★ 부수 소득 — 같은 정책을 두 번 재서 **n=4 의 잡음 바닥**을 측정했다

위 대조군은 baseline 과 **동일 체크포인트·동일 정책**인데 개시 지표가 이만큼 다르다:

| baseline `model_40000` | 상승 duty | spread | 롤아웃별 | 달성률 |
|---|---|---|---|---|
| video-ON | 0.54 | 0.88 | 0.00 · 0.63 · 0.65 · 0.88 | 29% |
| no_video | 0.37 | 0.97 | 0.00 · 0.00 · 0.49 · 0.97 | 21% |

플래그 효과가 아니다(넘어짐·변위·자세는 동일했다). cmd 0.5 상승이 이봉이라 **n=4 로는 0.37 과
0.54 를 가를 수 없다**는 뜻이다. → **잡음 바닥 = duty ±0.17, 달성률 ±8pp. 이 미만 차이는 근거로
쓰지 말 것.** (메모리의 "start 지표는 n=4 로 방법 간 비교 불안정, 8\~12회 필요" 가 수치로 확인됐다.)

☑ **arm B 판정은 이 잡음보다 훨씬 크다.** baseline 두 세트를 합쳐 n=8 로 보면:

```
baseline (n=8)  0.00  0.00  0.00  0.49  0.63  0.65  0.88  0.97   평균 0.45 · 8회 중 3회 완전 실패
arm B    (n=4)  1.00  1.00  1.00  1.00                            평균 1.00 · 예외 없음
```

±0.17 잡음으로는 이 분리가 만들어지지 않는다. 다만 **arm B 자체는 아직 n=4** 이므로 50k 에서
반복 수를 늘려(≥8) 재확인한다.

★ **측정 함정 — 넘어진 롤아웃이 "정지 자세 유지" 로 집계된다.** `ramp_onset_duty.py` 와
`ramp_summary.py` 는 명령 단계별 hold 구간만 보므로, 정점에서 넘어져 남은 구간을 누워서 보내면
duty 0.00 = 정지로 잡힌다 — 서 있는 것과 누워 있는 것이 같은 숫자다. arm A 의 하강 duty 0.00 을
하마터면 "저속에서 안 걷는다" 로 읽을 뻔했다(실제 원인은 저속과 무관). 진단기를
`_workspace/leg/ramp_fall_probe.py` 로 분리했다 — 전진 정지·종료 자세·heading 셋을 교차 확인해
"램프 끝에서 명령 0 이라 멈춘 것" 과 구분한다.

### 7-8. 남은 열린 문제 (이 실험의 범위 밖)

1. **저속 과잉(overshoot)** — 걷기 시작해도 cmd 0.5에서 달성률이 17~71%로 흩어지고 RMA 계열에서는
   125~142% 과속이 나온다. 이건 개시(onset)와 **다른 실패**다. 문헌상 비-아키텍처 설명이 최소 넷
   있고(정지 local optimum, 저속 학습의 일반적 난이도, limit-cycle 보행의 물리적 최저속도 하한,
   AMP 참조 커버리지) 어느 것도 배제되지 않았다. **아키텍처 arm의 성과로 귀속시키지 말 것.**
2. **`random_stand` 플래그 누락** (§7-4b) — run 이름과 실제 cfg가 어긋난다. 학습 스크립트가
   run 이름 토큰과 해석된 cfg를 대조해 경고하게 하거나, 이 축을 cfg 기본값으로 승격시킬 것.
3. **DAGGER 혼합비 5%** — `it % 20 == 0`은 이 분야 관례보다 희박하다(보통 커리큘럼으로 점증).
   cmd ≥ 1.0에서는 증류가 잘 되고 있어 급하지 않지만, 정지 근방 정확도를 원하면 손볼 축이다.

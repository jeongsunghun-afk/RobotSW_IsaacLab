# 아키텍처(정책 규모)가 저속 보행 개시 실패의 원인인가

**2026-08-26** · 판정: **reward+data가 결정 요인이다. 순수 폭/깊이(capacity) 스윕은 이 지표를 움직이지 않는다.**
(단, 별도 축인 "RMA 배선 문제"는 유효한 저비용 수정 후보로 남는다 — §7 참조)

읽기 전 요약(TL;DR): cmd 0.5에서 정지가 task reward의 88%를 이미 확보해 gradient가 약하다는 것,
그리고 학습이 진짜 정지 상태에서 출발하는 경험을 거의 받지 못한다는 것을 코드로 확인했다.
그런데 이것만으로는 "아키텍처가 아니다"를 증명하지 못한다 — **결정적 증거는 기존 램프 npz를
커맨드별로 재분석**해서 나왔다(§6): cmd 0.5는 duty 0.00~1.00으로 불안정하지만, **cmd ≥ 1.0에서는
19개 롤아웃(5개 체크포인트 × 3개 학습 계열, stand-reset 유무 무관) 전부 duty=1.00**이다. 위상이
관측 안 되는 문제라면 cmd 1.0에서도 실패해야 하는데 실패하지 않는다. 따라서 이것은 표현력 문제가
아니라 **보상 gradient 크기의 문제**다.

같은 디렉터리의 `README.md`(다른 agent, 독립 조사)도 같은 코드에서 같은 수식(`exp(-0.5·err²)`,
정지 시 cmd0.5=0.8825)을 도출해 **깊이 스케일링 스윕을 하지 않기로** 결론 냈다 — 두 agent가
독립적으로 수렴했다. 그 문서가 "A/B 진행 중"으로 남긴 세 번째 경쟁 가설(§A-1c, "정지=estimator
OOD")은 §7-1에서 기존 데이터만으로 추가 검증했다.

---

## 0. 기호

| 기호 | 의미 | 단위 |
|---|---|---|
| `v_cmd` | 명령 선속도 (body-frame x) | m/s |
| `v_x` | 실제 선속도 (body-frame x) | m/s |
| `w_lin` | `lin_vel_reward_w` (코드 기본값 0.7) | - |
| `w_yaw` | `yaw_vel_reward_w` (코드 기본값 0.3) | - |
| `k_lin` | `vel_err_scale` (코드 기본값 0.5) | (m/s)⁻² |
| `r_lin` | `exp(-k_lin * (v_cmd - v_x)^2)` | - |
| `R_task` | `w_lin*r_lin + w_yaw*r_yaw` (+ torque penalty, 이 run은 0) | - |
| `lerp` | `task_reward_lerp` (코드 기본값 0.5) | - |
| `R_total` | `lerp*R_task + (1-lerp)*style_term` | - |
| `duty` | hold 구간에서 `‖jvel‖_RMS > 0.4` 인 스텝의 비율 (0=정지, 1=연속보행) | - |

코드 출처: `leg_imitation_tracking_env.py:299-322`(`_get_rewards`), `leg_imitation_tracking_env_cfg.py:104-112`,
`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:160-174`(fusion).

---

## 1. Reward 수치 분석 — task 항만 (§1의 핵심 질문)

### 1-1. 코드에서 그대로 가져온 수식

```python
lin_vel_b = self._robot.data.root_link_lin_vel_b[:, :2]
lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
lin_vel_reward = torch.exp(-self.cfg.vel_err_scale * lin_vel_err)

yaw_vel_b = self._robot.data.root_link_ang_vel_b[:, 2]
yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
yaw_vel_reward = torch.exp(-self.cfg.yaw_vel_err_scale * yaw_vel_err)

reward = (
    self.cfg.lin_vel_reward_w * lin_vel_reward
    + self.cfg.yaw_vel_reward_w * yaw_vel_reward
    + torque_penalty   # 이 run은 torque_penalty_w=0.0 → 0
)
```

(`leg_imitation_tracking_env.py:299-322`). `vy` 명령은 항상 0이므로 `lin_vel_err`는 사실상 `(v_cmd - v_x)^2`
1차원으로 축약된다. yaw 항은 `w_lin`·`k_lin`과 완전히 독립이라 아래 표에서는 lin 항만 격리해 본다
(직진 명령을 가정하면 yaw 항은 stand/walk 선택과 무관하게 동일하다).

측정에 쓰인 run(`2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd`)의 `params/env.yaml`에서
`lin_vel_reward_w=0.7, vel_err_scale=0.5`를 직접 확인했다 — 기본값과 동일.

### 1-2. "정지"(v_x=0)가 얻는 task reward — 명령별

```
v_cmd | r_lin(정지) | R_lin(정지)=w_lin*r_lin | 정지가 얻는 비율 | 완벽추종과의 gap
------|------------|------------------------|-----------------|------------------
 0.5  |   0.8825   |   0.6177 / 0.7          |      88.2%      |  0.0823 (11.8%)
 0.7  |   0.7827   |   0.5479 / 0.7          |      78.3%      |  0.1521 (21.7%)
 1.0  |   0.6065   |   0.4246 / 0.7          |      60.7%      |  0.2754 (39.3%)
 1.5  |   0.3247   |   0.2273 / 0.7          |      32.5%      |  0.4727 (67.5%)
 2.0  |   0.1353   |   0.0947 / 0.7          |      13.5%      |  0.6053 (86.5%)
 3.0  |   0.0111   |   0.0078 / 0.7          |       1.1%      |  0.6922 (98.9%)
```

핵심: **정지가 확보하는 reward 비율은 명령이 커질수록 지수적으로 붕괴한다.** cmd 0.5에서 정지는
"거의 다 먹고" 시작하고, cmd 2.0에서는 거의 아무것도 못 먹는다. 이것이 저속(0.5 부근)에서만
정지-보행 이봉이 나타나고 고속에서는 전혀 문제없이 학습되는 이유를 정량적으로 설명한다.

추가로, cmd 0.5에서는 "제대로 못 걸어도" 거의 만점에 가깝다는 점도 중요하다:

```
v_cmd=0.5, 절반 속도(0.25)로만 걸어도  → r_lin=0.9692 → R_lin/w_lin = 96.9%
v_cmd=1.0, 절반 속도(0.5)로만 걸어도   → r_lin=0.8825 → R_lin/w_lin = 88.2%
v_cmd=2.0, 절반 속도(1.0)로만 걸어도   → r_lin=0.6065 → R_lin/w_lin = 60.7%
```

즉 cmd 0.5 부근은 task reward 곡면 자체가 **거의 평평하다** — 서 있어도, 절반만 걸어도, 완벽히
걸어도 reward 차이가 크지 않다. "gradient가 약하다"고 표현하는 것이 정확하다.

### 1-3. ★ 정정(advisor 지적 반영) — 이것만으로 "정지가 보상적으로 유리하다"고 말하면 틀린다

Total reward는 task 단독이 아니라 `R_total = 0.5*R_task + 0.5*style_term`이다(§2). §3에서 확인하듯
이 태스크의 참조 모션(ds14, 14클립)에는 **정지(v≈0) 클립이 단 하나도 없다** — 최저 속도가
0.108 m/s(`leg_walk_stmr`)이고 그나마도 "느리게 걷는" 클립이지 "서 있는" 클립이 아니다. AMP
discriminator는 정지 상태에 대한 참조 지지(support)가 전혀 없으므로, 정지 상태는 discriminator
입장에서 분포 밖(off-support)이고, `style_term`은 정지를 **선호하지 않고 오히려 불리하게** 만들
것으로 기대된다(정확한 수치는 §2에서 논의하듯 학습된 discriminator 가중치에 의존해 코드만으로
계산할 수 없다 — 이 방향성은 추론).

따라서 올바른 결론은 "정지가 보상-양(positive)이다"가 아니라: **task 항은 cmd 0.5 부근에서
걷기와 정지를 거의 구분하지 못하고(88% vs 100%), 그 gradient가 무엇이든 정지를 이기게 하려면
style 항(AMP)이 그 차이를 메워야 하는데, style 항은 명령 크기와 무관하게 대략 일정한 폭으로만
정지를 벌준다.** 명령이 커질수록 task 항의 gap이 기하급수로 벌어지므로(0.5→2.0에서 11.8%→86.5%),
**"항상 일정한 style 항의 폭"과 "명령에 비례해 커지는 task 항의 폭"을 더하면, 어떤 문턱
명령값 부근에서 정지→보행 전환이 신뢰성 있게 일어나기 시작하는 구조**가 자연스럽게 나온다.
이 구조적 예측은 §6의 실측과 정확히 들어맞는다.

---

## 2. AMP이 저속에서 기여하는 바 (§2)

### 2-1. Fusion 수식 (코드 확인)

`rsl_rl/rsl_rl/runners/on_policy_runner_amp.py:160-174`:

```python
task_reward_lerp = self.alg.amp_task_reward_lerp   # = 0.5 (cfg에서 확인)
...
total_reward = task_reward_lerp * rewards + (1.0 - task_reward_lerp) * style_term
```

`leg_imitation_tracking/agents/rsl_rl_ppo_cfg.py:224`에서 RMA 러너의 `task_reward_lerp=0.5`,
`fusion="lerp"` 확인. Discriminator는 `disc_loss_type="bce"`, `disc_reward_type="bce"`
(`rsl_rl/rsl_rl/algorithms/ppo_amp.py:37,129-133`) — sigmoid 출력 기반 reward이며 정확한 스칼라 값은
학습된 disc 가중치에 의존해 코드만으로 도출할 수 없다. 이 보고서는 **방향성만** 코드+데이터로 주장한다.

### 2-2. 참조 데이터셋에 "정지" 클립이 있는가 — 직접 로드해서 확인

측정 run이 실제로 쓴 데이터셋(`imitation/new_smr_leg_pkl`, 14클립, "ds14")을 `LegMotionLib`으로
직접 로드해 클립별 평균 속도와 `command_uniform`(이 run이 쓴 가중 모드) 샘플링 가중치를 계산했다:

```
클립                              속도[m/s]   length-weight   command_uniform-weight(vmax=3.2)
leg_run0(+mirror)                  2.687         3.01% ×2         8.27% ×2
leg_run1(+mirror)                  3.032         6.08% ×2         5.32% ×2
leg_trot0(+mirror)                 1.974         7.15% ×2        13.63% ×2
leg_walk                           0.186         6.80%             3.29%
leg_walk1                          0.942         1.24%            20.16%
leg_walk1_stmr                     0.447         2.66%             5.70%
leg_walk_stmr                      0.108        11.71%             4.60%
leg_walk_turn(+mirror)             0.684         7.15% ×2          3.87% ×2
leg_walk_turn_stmr(+mirror)        0.319        15.41% ×2          2.04% ×2
```

**0 m/s(정지) 클립은 없다.** 최저 속도는 0.108 m/s고 그마저 "느린 걸음"이지 "정지"가 아니다.
0.6 m/s 미만 클립의 합산 가중치는 length 모드 52.0% / command_uniform 모드 17.7%다 — 즉 저속 대역
자체의 커버리지는 나쁘지 않다(오히려 length 모드에서는 과대표집). **누락된 것은 "느린 걸음"이
아니라 "정지"다.**

이는 §1-3의 추론을 뒷받침한다: discriminator가 학습하는 "그럴듯한 동작" 분포에 정지가 아예
없으므로, 정지는 style 보상을 받을 근거가 없다 — style 항이 정지를 밀어준다는 가설은 데이터로
기각된다. (다른 팀 문서의 "0.05~3.01 m/s" 표기와 이 표의 최저값 0.108이 다른 것은 관측 오차가
아니라, 그쪽이 참조한 것이 waistfix 데이터셋 등 다른 버전일 수 있음 — 본 계산은 실제 측정 run이
쓴 `new_smr_leg_pkl`을 직접 로드해서 얻은 값이다.)

---

## 3. Command 샘플링 / reset 전략 (§3)

### 3-1. 코드 확인 — `reset_strategy`

```python
# leg_imitation_tracking_env_cfg.py (현재 워킹트리, uncommitted)
reset_strategy: str = "random"   # "random" | "random_start" | "random_stand"
rel_stand_envs: float = 0.1
```

`_reset_idx`에서:

```python
stand_ids = env_ids[:0]
rsi_ids = env_ids
if "stand" in self.cfg.reset_strategy and self.cfg.rel_stand_envs > 0.0:
    is_stand = torch.rand(len(env_ids), device=self.device) < self.cfg.rel_stand_envs
    stand_ids = env_ids[is_stand]
    rsi_ids = env_ids[~is_stand]
```

`"stand" in "random"` 은 `False`이므로, `reset_strategy="random"`(기본값)일 때 `rel_stand_envs`는
**완전히 무시**되고 100% RSI(모션 프레임에서 시작, 즉 이미 움직이는 상태)로 리셋된다. 그리고
RSI 소스인 모션 클립에는 §2에서 확인했듯 정지 프레임이 없다. **결론: `reset_strategy="random"`일
때 학습 중 에피소드는 단 한 번도 "진짜 정지 상태"에서 시작하지 않는다.**

이건 소스 파일의 기본값 이야기만이 아니다 — 실측 run(`2026-08-24_17-27-53_..._ds14_wcmd`)의
`params/env.yaml`을 직접 열어 확인했다:

```
reset_strategy: random
rel_stand_envs: 0.1
cmd_deadzone: 0.0
torque_penalty_w: 0.0
```

**이 run은 실제로 `reset_strategy=random`으로 돌았다.** 즉 이 run으로 측정된 히스테리시스는
학습 중 정지 출발 경험이 0%인 정책에서 나온 것이다.

### 3-2. ★ 결정적으로 중요한 발견 — 이 run의 이름과 실제 설정이 어긋난다

`2026-08-24_17-27-53_torque1e5_stand_dz_vmax32_ds14_wcmd`라는 디렉터리 이름은
`torque1e5`(torque_penalty_w=1e-5), `stand`(reset_strategy=random_stand),
`dz`(cmd_deadzone>0)를 암시한다. 그런데 실제 `env.yaml`은 `torque_penalty_w=0.0`,
`reset_strategy=random`, `cmd_deadzone=0.0`이다. **이름과 설정이 불일치한다.**

이 lineage의 **이전 9개 run 전부**(2026-08-04 ~ 2026-08-22)를 `env.yaml`로 직접 대조했다:

```
run (날짜_태그)                                           reset_strategy  rel_stand  cmd_dz  torque_pen_w
2026-08-04_torque1e5_newsmr_stand_deadzone                random_stand    0.1        0.1     1e-5
2026-08-05_torque1e5_newsmr_stand_dz_lerp065               random_stand    0.1        0.1     1e-5
2026-08-06_torque1e5_newsmr_stand_dz_lerp030               random_stand    0.1        0.1     1e-5
2026-08-06_torque1e5_trotmirror_stand_dz                   random_stand    0.1        0.1     1e-5
2026-08-10_torque1e5_trotmirror_stand_dz_ds14               random_stand    0.1        0.1     1e-5
2026-08-11_torque1e5_trotmirror_stand_dz_run2t14            random_stand    0.1        0.1     1e-5
2026-08-12_torque1e5_trotmirror_stand_dz_ds14_vmax32        random_stand    0.1        0.1     1e-5
2026-08-14_torque1e5_stand_dz_vmax32_trot110150             random_stand    0.1        0.1     1e-5
2026-08-18_torque1e5_stand_dz_vmax32_trot110150_wcmd        random_stand    0.1        0.1     1e-5
2026-08-18_torque1e5_stand_dz_vmax32_waistfix               random_stand    0.1        0.1     1e-5
2026-08-22_torque1e5_stand_dz_vmax32_waistfix_wcmd          random_stand    0.1        0.1     1e-5
--------------------------------------------------------------------------------------------------
2026-08-24_torque1e5_stand_dz_vmax32_ds14_wcmd (측정 run)   random          0.1        0.0     0.0   ← 어긋남
```

3주(9~11회 run) 동안 `reset_strategy=random_stand`가 실제로 켜져 있었는데, 2026-08-26 히스테리시스
보고서를 만든 바로 그 run에서만 꺼져 있다. 학습 launch 로그(`_workspace/leg/train_logs/ds14_wcmd.log`)에
`reset_strategy`를 언급하는 CLI override가 전혀 없으므로, 이건 명시적 ablation이 아니라 **소스
기본값이 이 run 직전에 되돌아간 것**으로 보인다(현재 워킹트리에서도 `leg_imitation_tracking_env_cfg.py`
전체가 커밋되지 않은 로컬 수정 상태이고, `git blame`은 모든 관련 라인을 "Not Committed Yet"으로
표시한다 — 이 파일은 이 브랜치의 2446개 커밋 중 단 1개(최초 이식 커밋)만 건드렸을 뿐, 3주간의
반복은 전부 커밋되지 않은 로컬 편집으로 진행됐다). **왜 되돌아갔는지는 알아내지 못했다(추론
불가) — 사실로 확인한 것은 "설정이 달랐다"까지다.**

이건 히스테리시스 보고서 자체에 대한 **데이터 무결성 플래그**다: 그 보고서 안의 비교행
"ds14(length)"(2026-08-10 run, `ds14_0810_final_*/ramp_data.npz`로 확인)와 "wcmd(ds18)"(체크포인트
문자열이 `2026-08-18_09-28-11_torque1e5_stand_dz_vmax32_trot110150_wcmd/model_40000.pt`임을
npz에서 직접 확인)는 `reset_strategy=random_stand`로 돌았고, "8k/15k/25k/40k/50k" 행은 전부
`reset_strategy=random`인 2026-08-24 run **한 run의 체크포인트들**이다. 즉 같은 표 안에서 리셋
전략이라는 통제되지 않은 축이 섞여 있다.

### 3-3. 그런데 — random_stand가 켜져 있었어도 문제가 해소된 적은 없다

위 표는 "고쳐지지 않은 원인이 stand-reset이 꺼졌기 때문"이라고 성급히 결론내리게 만들 수 있다.
그렇지 않다는 것을 §6의 재분석이 직접 보여준다: `reset_strategy=random_stand`로 돌았던
2026-08-10 run과 2026-08-18 run의 체크포인트도 cmd 0.5에서 동일하게 불안정한 duty(0.05~0.98)를
보인다. **10% 확률의 stand-reset는 이 문제를 눈에 띄게 고치지 못했다.**

이유를 추정할 수 있다: stand-reset가 걸려도 명령은 여전히 `[0, lin_vel_x_max]`에서 균등 샘플링된다
(`_resample_steering`은 stand/RSI 구분 없이 모든 `env_ids`에 동일하게 적용됨, `leg_imitation_tracking_env.py:544-572`,
`_reset_idx`의 마지막 부분에서 stand_ids/rsi_ids 통합 후 호출). `lin_vel_x_max=3.2`(vmax32 run들 기준)일 때,
"정지에서 시작 **그리고** 명령이 0.5 부근"이 되는 경우는 대략:

```
P(stand) * P(cmd ∈ 대략 0.5 근방 폭 ±0.25) ≈ 0.10 * (0.5 / 3.2) ≈ 0.10 * 0.156 ≈ 1.6%
```

(폭 ±0.25는 램프 평가의 hold 구간 폭을 본뜬 대략적 추정치이며, 정확한 학습 시 "0.5 부근"의 정의는
없다 — 이 숫자는 추론/근사다.) 즉 "정지에서 출발해서 딱 이 gradient가 약한 구간을 걸어야 하는"
경험은 전체 리셋의 **1.6% 남짓**뿐이었다. 명목상 커리큘럼은 있었지만 **저속 구간에 집중되도록
설계되지 않아 사실상 희석됐다.**

---

## 4. 조기 종료 (Termination) — 정지는 무위험이다 (§4)

```python
died = base_height < self.cfg.termination_height       # 0.35 m
died = died | tilted                                     # gz > -cos(45°)
died = died | bad_contacts                                # base 접촉(사실상 무력, threshold=500N)
```

(`leg_imitation_tracking_env.py:324-354`) 기본 자세(default_joint_pos)로 서 있으면 base 높이는
termination_height(0.35m)보다 훨씬 높고 tilt는 0에 가까우므로 **정지는 종료 조건을 전혀 건드리지
않는다.** 반대로 걷기 시도는 넘어짐 → 조기 종료 → 남은 에피소드(최대 10s) reward 상실 위험을
수반한다. 이는 §1의 "약한 gradient"에 **비대칭 위험**을 추가한다: 특히 학습 초반, 서투른 걸음의
기댓값이 그 위험 때문에 정지보다 낮게 평가될 수 있다. (정성적 논증, 수치화하지 않음.)

---

## 5. Git / reports 이력 — random_stand arm이 실제로 돌았는가 (§5)

- `git log --follow`로 `leg_imitation_tracking_env_cfg.py`를 추적하면 커밋 1개(포트 커밋)뿐이다.
  3주간의 모든 하이퍼파라미터 반복(stand/dz/torque_penalty/dataset/weight_mode)은 **커밋되지
  않은 로컬 편집**으로 진행됐다 — `git blame` 전 라인이 "Not Committed Yet".
- `logs/rsl_rl/leg_imitation_tracking_rma/`의 `params/env.yaml`을 직접 열어 대조한 결과
  (§3-2 표), `reset_strategy="random_stand"` arm은 **실제로 9~11회 실행됐다**(2026-08-04~08-22).
  "한 번도 시도되지 않은 저비용 실험"이 아니다.
- 그 9~11회 중 어느 것도 §6에서 쓴 "cmd별 duty 분해" 방식으로 up/down을 분리 측정하지 않았다
  (이 분석 스크립트 `plot_lowspeed_hysteresis.py`와 `_workspace/leg/*/ramp_data.npz`에 저장된
  `reset_strategy` 메타데이터는 2026-08-26 세션에 새로 만들어진 것으로 보인다). 즉 "껐다 켰다"는
  했지만 **정확히 이 실패 모드를 겨냥해 격리 측정한 적은 없었다.**
- `reports/leg_imitation/_comparisons/arch_scaling_study/` 안에 이미 다른 agent가
  작성한 `README.md`가 있으며, 거기서는 "RMA actor가 PPO 롤아웃의 95%에서 history를 보지 않고
  episode-상수인 `priv_latent`만 본다"는 별도 가설을 다룬다 — §7에서 검증하고 논한다.

---

## 6. ★ 결정적 재분석 — 기존 램프 데이터를 커맨드별로 다시 잘라본다 (신규 분석, 이 세션에서 수행)

### 6-1. 방법

`_workspace/leg/*/ramp_data.npz`는 각 20ms 스텝의 `vx_cmd`, `vx`, `jvel`(17관절)을 저장한다. 램프는
`[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 3.5, ..., 0.5, 0.0]` 순서의 17개 구간(각 hold 3.0s +
ramp 2.0s)으로 구성된다(`hold_s`, `ramp_s`, `vx_cmd`를 npz에서 직접 읽어 검증). **각 파일은
`reset_strategy` 메타데이터를 저장하고 있어, 전부 `random_stand`(진짜 정지에서 램프 시작)로
평가됐음을 직접 확인했다** — 즉 "상승 = 정지에서 출발"이라는 이 보고서 전체의 전제는 확인된
사실이다(advisor가 검증을 요구한 지점).

기존 `plot_lowspeed_hysteresis.py`는 cmd 0.5 구간(index 1, 15)만 봤다. 여기서는 **상승 구간 전체
(index 0~8, cmd 0→4.0)**의 duty를 모든 available 체크포인트에 대해 계산했다 — 새 학습 없이 기존
파일을 다시 자르기만 하면 되는, 비용 0의 재분석이다.

### 6-2. 결과 — 19개 롤아웃, 5개 체크포인트, 3개 학습 계열

**주의**: 아래 표의 "평가"는 램프 npz에 저장된 `reset_strategy` 필드(§6-1에서 확인한 대로, 램프는
항상 `random_stand`로 강제되어 정지에서 시작한다)이고, "학습"은 그 체크포인트를 만든 학습 run의
`params/env.yaml`에서 읽은 값(§3-2 표)이다. 이 둘은 서로 다른 축이며 혼동하면 안 된다 — 특히
ds14wcmd 세 체크포인트는 **학습 중에는 정지 노출이 0%였는데, 평가는 항상 정지에서 시작**한다.

| 체크포인트 | 학습 시 reset_strategy | 평가 시 reset_strategy | n | cmd=0.5 duty 범위 | cmd≥1.0 duty |
|---|---|---|---|---|---|
| 8k (2026-08-24 run) | `random` | `random_stand`(강제) | 4 | 0.09 ~ 1.00 | **전부 1.00** |
| 25k (2026-08-24 run) | `random` | `random_stand`(강제) | 4 | 0.00 ~ 0.43 | **전부 1.00** |
| 50k (2026-08-24 run) | `random` | `random_stand`(강제) | 4 | 0.00 ~ 0.99 | **전부 1.00** |
| ds14_0810 (2026-08-10 run) | `random_stand` | `random_stand`(강제) | 4 | 0.61 ~ 0.91 | **전부 1.00**(단 cmd=4.0에서만 0.33~0.45로 재붕괴 — 별개 현상, §6-3) |
| wcmd_it40000 (checkpoint = `2026-08-18_09-28-11_torque1e5_stand_dz_vmax32_trot110150_wcmd/model_40000.pt`, npz에서 직접 확인) | `random_stand` | `random_stand`(강제) | 4 | 0.05 ~ 0.98 | **전부 1.00** |

즉 학습 시 정지 노출이 0%였던 세 체크포인트(8k/25k/50k)와 정지 노출이 있었던 두 체크포인트
(ds14_0810, wcmd_it40000)가 cmd 0.5에서 **구분되지 않는 동일한 불안정 패턴**을 보인다 — 이것이
"10% stand-reset가 이 문제를 눈에 띄게 고치지 못했다"(§3-3)는 주장의 직접적 근거다.

19개 상승-롤아웃 전부에서: **cmd=0.5는 duty 0.00~1.00으로 완전히 불안정**(이봉), **cmd=1.0부터는
19/19 모두 duty=1.00**(예외 없음). 원자료:

```
예) ds14wcmd_it25000_1: cmd0.5 duty=0.03 → cmd1.0 duty=1.00 → cmd1.5 duty=1.00 → ...
    ds14wcmd_it25000_2: cmd0.5 duty=0.00 → cmd1.0 duty=1.00 → cmd1.5 duty=1.00 → ...
    ds14_0810_final_2 :  cmd0.5 duty=0.63 → cmd1.0 duty=1.00 → cmd1.5 duty=1.00 → ...
```

### 6-3. 이것이 판정에 결정적인 이유

**"정지에서 gait phase를 관측할 수 없다"는 가설이 맞다면, cmd 1.0에서도 실패해야 한다.** cmd
0.5에서 정지 상태(관절속도≈0)로 시작하는 것과 cmd 1.0에서 정지 상태로 시작하는 것은 **관측
관점에서 완전히 동일한 상황**이다(위상 정보 부재는 명령 크기와 무관). 그런데 정책은 cmd 0.5에서는
19번 중 다수 실패하고 cmd 1.0에서는 19번 전부 성공한다. 유일하게 달라지는 것은 §1에서 계산한
**보상 gradient의 크기**(정지가 얻는 task reward 비율이 88.2%→60.7%로 떨어짐)다. 이는 순수
표현력/관측성 가설을 직접 반증하고, 보상 gradient 가설과 정확히 일치한다.

(cmd=4.0에서 ds14_0810 계열만 duty가 0.33~0.45로 떨어지는 것은 §1-2 표에서 정지가 얻는 reward가
0%에 가까워 정지로의 회귀가 설명되지 않는, **완전히 다른 현상**이다 — 아마 최고속 근처의 토크/추종
한계다. 이 재붕괴는 wcmd_it40000 계열에서는 재현되지 않아(4/4 모두 cmd 4.0도 duty=1.00) run별
현상이며, 저속 이봉과는 별개 이슈이니 이 보고서의 판정 범위 밖에 둔다.)

---

## 7. 별도 축 — RMA actor의 history 배선 문제 (검증됨, 그러나 1차 원인은 아님)

같은 디렉터리의 `README.md`(다른 agent 작성)가 제기한 주장을 직접 코드에서 검증했다:

```python
# rsl_rl/rsl_rl/modules/actor_critic_parkour.py:283-292 (직접 확인)
def act(self, obs, hist_encoding=False, **kwargs):
    ...
    if hist_encoding:
        history_latent = self.get_hist_latent(obs)          # ← 배포/추론 경로
        obs_actor = torch.cat([obs_actor, priv_explicit, history_latent], dim=-1)
    else:
        priv_latent = self.get_priv_latent(obs)              # ← PPO 롤아웃 경로
        obs_actor = torch.cat([obs_actor, priv_explicit, priv_latent], dim=-1)

# rsl_rl/rsl_rl/runners/on_policy_runner_parkour_amp.py:110 (직접 확인)
hist_encoding = it % 20 == 0   # dagger_update_freq — 20 iter 중 1회(5%)만 history 경로
```

그리고 `priv_latent`가 `base_mass(1) + base_com(3) + joint_stiffness_ratio(17) +
joint_damping_ratio(17)`(에피소드 내 상수)임을 `leg_imitation_tracking_rma_env.py:87-107`에서
직접 확인했다. **이 세 가지 주장은 전부 코드로 확인된 사실이다** — 다른 agent의 주장을 검증
없이 인용하지 않는다는 규약(`feedback_worker_absence_claim_must_verify`)에 따라 재검증했다.

즉 PPO가 실제로 최적화하는 RMA actor 입력에는 시간/위상 정보가 없다는 것은 사실이다. 하지만
§6이 보여주듯, **바로 그 동일한 actor**가 cmd≥1.0에서는 19/19 신뢰성 있게 보행을 개시한다.
따라서 이 배선 문제는 "정지에서 위상을 복원할 수 없어 보행을 개시 못 한다"는 결론으로 이어지지
않는다 — 개시는 하고 있다, 다만 명령이 충분히 클 때만. 이 배선 이슈는 **독립적으로 고칠 가치가
있는 저비용 수정**(history를 실제 PPO 경로에 흘리는 것)이지만, cmd 0.5 이봉의 1차 원인은 아니다.
또한 이 문제는 RMA(`leg_imitation_tracking_rma_env_cfg`) 변형에만 있고, 사용자가 원래 지목한
`Leg-Imitation-Tracking-v0`(plain `LegImitationTrackingEnvCfg`, `obs_groups={"policy":["policy"]}`,
history 자체가 없음)에는 애초에 해당하지 않는다 — plain baseline은 이 배선 버그가 있을 수조차 없다.

### 7-1. ★ 같은 디렉터리의 A-1c 가설("증류 오차로 정지 시 latent가 OOD") — 게이트 테스트를 데이터로 대신 통과시킨다

`README.md` §A-1c는 세 번째 경쟁 가설을 제기한다: "정지 상태에는 플랜트를 식별할 여기(excitation)가
없어 history-distillation으로 학습된 `priv_latent` 인코더가 OOD 값을 내고, 그래서 actor가 얼어붙는다"
— 그리고 이를 확인하려면 `priv_latent` 경로 강제 vs `history_latent` 경로(평가 기본값) 비교가
필요하다며 "A/B 진행 중"으로 남겨뒀다. 새 코드 없이 기존 npz로 이 가설을 강하게 압박할 수 있다.

만약 "정지 상태 자체가 estimator를 OOD로 만든다"가 핵심 기전이라면, cmd=0.5 구간이 실패해
로봇이 계속 정지해 있다가 cmd가 1.0으로 넘어가는 그 순간에도 **로봇은 여전히 정지 상태**다 —
즉 estimator 입력 관점에서는 cmd=0.5 실패 직전과 cmd=1.0 진입 직후가 물리적으로 구분되지 않는다.
그런데도 개시에 성공한다면, "정지 상태의 estimator OOD"가 독립 원인일 가능성은 낮아진다.

경계면을 직접 측정했다(cmd 0.5 구간의 마지막 0.5초 vs cmd 1.0 구간의 처음 0.5초, cmd0.5에서
`duty<0.2`로 사실상 계속 정지해 있던 5개 사례):

```
run                    | cmd0.5 종료 시점            | cmd1.0 시작 후 0.5초
                       | vx      jvelRMS  duty       | vx      jvelRMS  duty(전체)
-----------------------|------------------------------|------------------------------
ds14wcmd_it25000_1     | -0.014  0.067    0.03        | 0.548   1.149    1.00
ds14wcmd_it25000_2     |  0.017  0.075    0.00        | 0.551   1.368    1.00
ds14wcmd_it25000_3     | -0.016  0.092    0.09        | 0.602   1.309    1.00
ds14wcmd_it49999_2     | -0.005  0.060    0.09        | 0.633   1.228    1.00
ds14wcmd_it49999_3     | -0.004  0.056    0.00        | 0.206   0.755    1.00
```

5건 전부 cmd=0.5 구간 끝에서 `vx≈0, jvelRMS<0.1`(완전 정지, 플랜트 식별 여기가 전혀 없는 상태)
이었는데, cmd가 1.0으로 바뀐 뒤 0.5초 안에 이미 유의미한 속도(vx 0.2~0.63)와 stepping
(jvelRMS 0.76~1.37)으로 전환되고, 그 구간 전체 duty는 5/5 모두 1.00이다. **"정지 상태라 estimator가
OOD"라는 전제 자체는 cmd 0.5 진입과 cmd 1.0 진입에서 동일한데 결과만 갈린다** — 명령 크기가
전환되는 순간, 물리 상태는 그대로인데 정책의 행동만 바뀐다. 이는 §6과 같은 결론(명령 크기에
따른 보상 gradient)을 가리키고, "정지=estimator 붕괴"라는 독립 기전을 약화시킨다. `priv_latent`
강제 A/B를 직접 도는 것보다 약하지만(진짜 게이트 테스트는 여전히 유효한 후속 검증), 기존
데이터만으로 얻을 수 있는 최선의 사전 신호는 이쪽을 가리킨다.

---

## 8. 판정

**reward+data가 결정 요인이다. 아키텍처(폭/깊이) 스윕은 이 지표를 움직이지 않는다.**

근거를 다시 정리하면:

1. task reward 곡면이 cmd 0.5 부근에서 거의 평평하다(정지 88.2% vs 완주 100%) — 코드에서 계산한 사실.
2. 참조 데이터셋에는 정지 클립이 없다 — style 항이 정지를 밀어준다는 대안 가설은 데이터로 기각된다(§2).
3. `reset_strategy=random`(기본값이자 측정 run의 실제값)에서는 학습이 진짜 정지 출발을 전혀
   경험하지 않는다 — 코드+실측 env.yaml로 확인한 사실(§3).
4. `random_stand`가 실제로 9~11회 켜져서 돌았지만(§3-2), 명령이 stand-여부와 무관하게 전체
   범위에서 균등 샘플링되어 "정지+cmd 0.5 근방"의 실제 노출은 ~1.6%에 불과했고(§3-3, 추정치),
   그 켜진 run들도 cmd 0.5 duty는 여전히 불안정했다(§6-2) — 즉 기존 커리큘럼은 시도됐지만
   저속에 집중되도록 설계되지 않아 사실상 무효화됐다.
5. **결정적으로**: 동일 정책·동일 관측 구조가 cmd≥1.0에서는 정지 노출·아키텍처 변화 없이도
   19/19 완벽하게 보행을 개시한다(§6). 위상 비관측성이 진짜 병목이라면 cmd 크기와 무관하게
   실패해야 하는데 그렇지 않다. 실패는 정확히 §1에서 계산한 "gradient가 약한 구간"에서만
   일어난다.
6. RMA actor의 history 배선 문제는 사실로 확인되지만(§7), 같은 actor가 cmd≥1.0에서 신뢰성 있게
   작동하므로 1차 원인이 될 수 없다.

명시적으로: **정책 폭/깊이(`[512,256,128]` 등)를 바꾸는 것을 겨냥한 실험 예산은 이 실패 모드에
대해서는 근거가 없다.** `arch_scaling_study/README.md`가 제안한 "다봉 정책/잠재 모드 변수" 계열
arm도, 표현력이 병목이 아니라는 §6의 직접 반증 앞에서는 이 실패 모드에 대한 근거를 잃는다(다른
실패 모드, 예: (B) 좌우 쏠림이나 (C) 고속 천장에 대해서는 이 보고서가 다루지 않았으므로 판단하지
않는다).

---

## 9. 가장 저렴한 다음 실험

새 학습이 필요 없는 부분은 이미 §6에서 끝냈다. §6이 실제로 짚어낸 병목은 "정지 노출 부족"이
아니라 **보상 gradient의 크기**다 — 같은 정책이 정지 노출 없이도(8k/25k/50k) cmd≥1.0에서는
완벽히 작동한다. 따라서 가장 직접적인 다음 실험은 노출량을 늘리는 것이 아니라 **gradient
자체를 조작하는 것**이다.

### 9-1. 1순위(주 arm) — `vel_err_scale` 증가

`k_lin = vel_err_scale`을 올리면 `exp(-k_lin*(v_cmd-v_x)^2)`의 falloff가 가팔라져, 정지가 cmd 0.5
에서 얻는 task reward 비율이 낮아진다. 예를 들어 `k_lin: 0.5 → 1.5`이면

```
v_cmd=0.5, 정지: r_lin = exp(-1.5*0.25) = 0.687  (현재 0.5 스케일에서의 0.8825 대비 대폭 하락,
                                                   §1-2의 cmd=1.0/k=0.5 수준인 0.607에 근접)
```

즉 `k_lin`을 3배 올리면 cmd 0.5에서 정지가 얻는 reward 비율이, §6에서 **이미 개시가 신뢰성
있게 일어나는** cmd=1.0 수준(60.7%, k=0.5 기준)까지 떨어진다. 이건 코드 한 줄(`vel_err_scale`
cfg 값) 변경에 정량적 예측이 딸린 실험이다 — 이 예측이 맞으면 cmd 0.5 duty가 좁아져야 한다.
(부작용: `k_lin`은 전 명령대에 동일하게 적용되므로 고속 추종 reward도 더 뾰족해진다 — 부수
효과를 §6과 동일한 방법으로 전 명령대에 대해 확인해야 한다.)

### 9-2. 2순위(보조 arm) — stand-reset 노출량 증가

기존에 "껐다 켰다"만 했던 stand-reset을 **저속에 집중해서** 다시 태워본다. 코드 변경은 필요 없다
(`rel_stand_envs`, `reset_strategy`, 커맨드 리샘플 범위 모두 이미 존재하는 파라미터다):

1. `reset_strategy="random_stand"`, `rel_stand_envs`를 0.1 → **0.3~0.5**로 올린다.
2. (선택) stand-reset된 env에 한해 명령 샘플링 범위를 `[0, 1.5]`처럼 저속 대역으로 좁힌다.

9-1이 §6이 짚어낸 메커니즘(gradient 크기)을 직접 조작하는 반면, 9-2는 이미 부분적으로 시도됐고
(§3-2, §6-2) 뚜렷한 효과가 안 보였던 축이다 — 우선순위는 9-1이 높다.

### 9-3. 판정

두 arm 모두 §6과 동일한 방식(램프 npz를 커맨드별로 재분해)으로 cmd 0.5 duty 분포가 좁아지는지
(0.00~1.00 → 예: 0.8 이상으로 수렴) 확인한다. `arch_scaling_study/README.md`가 이미 정한 규약대로
**40k 이후 체크포인트, ≥4 repeat**를 지킨다.

9-1이 duty를 안정화시키면 reward+data(구체적으로 gradient 크기) 판정이 최종 확정된다. 만약
9-1·9-2를 모두 시도했는데도 cmd 0.5 duty가 여전히 이봉이면(이 보고서의 예측과 반대), 그때 비로소
§7의 history 배선 수정(값싼 아키텍처 변경 — history를 PPO 롤아웃 경로에도 흘리기)을 다음 후보로
올리는 것이 맞다. 어느 쪽이든 폭/깊이 스윕이 정당화되는 경로는 이 단계들을 모두 통과하지
못했을 때뿐이다.

---

## 부록 — 사용한 원자료/스크립트

- 모션 데이터셋 로드/가중치 계산: `LegMotionLib`을 직접 import해 `imitation/new_smr_leg_pkl`(측정
  run이 실제로 쓴 14클립)과 `imitation/merged_leg_pkl`(현재 워킹트리 기본값, 7클립)을 각각 로드.
  실행 환경은 `numpy==2.3.1`이 필요해(pkl이 numpy 2.x로 pickle됨) `isaac-6.0` conda env
  (`/home/user/miniconda3/envs/isaac-6.0/bin/python`)를 사용했다 — 저장소 `AGENTS.md`가 권장하는
  `./isaaclab.sh -p`는 이 worktree에서 `python3`(가상환경 미활성)로 fallback해 torch가 없어 실패했다.
- §6 재분석 스크립트: 이 세션에서 작성, `_workspace/leg/*/ramp_data.npz`의 `vx_cmd/vx/jvel/dt/hold_s/
  ramp_s/reset_strategy` 필드를 읽어 세그먼트별 duty를 계산(기존 `plot_lowspeed_hysteresis.py`의
  `hold()` 함수와 동일한 정의, index 범위만 0~8로 확장).
- `env.yaml` 대조 대상 run 11개: `logs/rsl_rl/leg_imitation_tracking_rma/2026-08-{04,05,06,06,10,11,12,14,18,18,22,24}*`.

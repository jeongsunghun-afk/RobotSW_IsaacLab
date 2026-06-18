# AMP 이론 노트 & 비교 채점 기준
**작성자**: worker-theory (team go2-amp-compare)
**용도**: IsaacLab `go2_imitation` vs MimicKit `amp_go2_track` 비교에서 *paper-spec 대비 어디가 어긋났는가*를 점수화하기 위한 reference.

> 모든 인용은 (paper / 파일경로:line / URL) 형태로 표시. 직접 출처 없는 항목은 "추정".

---

## 1. AMP 핵심 정의 (Peng et al. 2021)

원논문: *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*, SIGGRAPH 2021. arXiv: https://arxiv.org/abs/2104.02180 (PDF: https://arxiv.org/pdf/2104.02180).
프로젝트 페이지: https://xbpeng.github.io/projects/AMP/

### 1.1 Discriminator 입력 (state-transition features)

Discriminator는 (s, s′) 페어가 motion dataset 분포 `d_M`에서 왔는지, 정책 분포 `d_π`에서 왔는지 판별한다. 그러나 raw state 대신 **관측 추출 함수** `Φ(s)`를 거쳐서 들어간다. 즉 `D(Φ(s), Φ(s′))`.

**Φ(s)에 들어가는 feature** (paper §5.3, /tmp/amp_paper.txt:540–574):
- Root의 linear / angular velocity, **character의 local coordinate frame** 으로 표현.
- 각 joint의 local rotation.
- 각 joint의 local velocity.
- End-effector (hands, feet)의 3D position, **character의 local frame**.

> Local frame 정의 (paper §5.3, /tmp/amp_paper.txt:561–564): 원점은 root (pelvis), x축은 root link의 facing direction, y축은 global up.

> **task feature는 의도적으로 빠진다.** "The observations also do not include any task-specific features, thus enabling the motion prior to be trained without requiring task-specific annotation of the reference motions" (paper §5.3, /tmp/amp_paper.txt:570–574). → goal/command/phase는 discriminator 입력에서 분리되어야 한다.

### 1.2 Discriminator loss (LSGAN + zero-centered Gradient Penalty)

Paper에서 채택한 형태는 **least-squares GAN (LSGAN)** + **Mescheder-style zero-centered GP on real samples**.

LSGAN loss (paper Eq. 6, /tmp/amp_paper.txt:512–522):

```
L_D = E_{d_M(s,s′)}[ (D(s,s′) − 1)^2 ] + E_{d_π(s,s′)}[ (D(s,s′) + 1)^2 ]
```

→ demo target = +1, policy target = −1.

GP를 더하면 (paper Eq. 8, /tmp/amp_paper.txt:592–607):

```
L_D_full = E_{d_M}[ (D − 1)^2 ] + E_{d_π}[ (D + 1)^2 ]
         + (w_gp / 2) · E_{d_M}[ || ∇_φ D(φ) |_{φ=(Φ(s),Φ(s′))} ||^2 ]
```

**중요 포인트**:
- GP는 **demo (real) sample 위에서만** 계산 (zero-centered, Mescheder 2018 스타일) — Gulrajani WGAN-GP의 "interpolation 위 1-centered"와는 다르다. (paper §5.4, /tmp/amp_paper.txt:587–611)
- GP는 raw state가 아니라 **Φ(observation features)에 대한 gradient**.
- 실제 사용된 `w_gp = 10` (paper §8.1, /tmp/amp_paper.txt:825 및 Table 4, /tmp/amp_paper.txt:1957).

### 1.3 Style reward (clip 형태)

Paper Eq. 7 (/tmp/amp_paper.txt:530):

```
r_S(s_t, s_{t+1}) = max( 0, 1 − 0.25 · (D(s_t, s_{t+1}) − 1)^2 )
```

→ `D=+1`일 때 r_S=1, `D=−1`일 때 r_S=0, `|D−1|>2`이면 clip돼서 0.

### 1.4 Total reward

Paper §6.3 / Algorithm 1 (/tmp/amp_paper.txt:650–652):

```
r_t = w_G · r_t^G  +  w_S · r_t^S
```

모든 task에 대해 **`w_G = 0.5, w_S = 0.5`** 사용 (paper §8.2, /tmp/amp_paper.txt:867–868 및 Table 4, /tmp/amp_paper.txt:1955–1956).

### 1.5 Reference State Initialization (RSI) + Early Termination

(paper §6.3, /tmp/amp_paper.txt:734–741):
- **RSI**: 매 episode 시작 시 character를 dataset의 motion clip 중에서 무작위 frame으로 초기화. → cold-start 회피, 다양한 phase 분포 보장.
- **Early termination**: 발 이외의 body part가 ground와 contact하면 종료 (대부분 task). 매우 contact-heavy task (rolling, getup)는 비활성.
- RSI는 DeepMimic (Peng 2018a)에서 가져온 기법으로 paper 본인이 "we incorporate reference state initialization"이라고 명시.

### 1.6 Sampling / Replay Buffer

(paper §6.3, /tmp/amp_paper.txt:742–751):
- Policy의 rollout trajectory를 replay buffer **B**에 저장.
- Discriminator는 `M`(demo dataset)과 **B**(policy replay)에서 minibatch를 뽑아 update.
- Replay buffer는 "discriminator가 가장 최근 batch에 overfit하는 것을 방지".
- Buffer size = `10^5` (Table 4, /tmp/amp_paper.txt:1966).
- Samples per update iteration = 4096, batch = 256, disc batch (K) = 256 (Table 4).
- → demo:policy 샘플 ratio는 1:1 (한 minibatch 안에서 동일 K). MimicKit `amp_agent.py:169` `replay_data = self._disc_buffer.sample(disc_obs.shape[0])`도 1:1 매칭 확인.

### 1.7 Network / Optimization 기본값

(Table 4, /tmp/amp_paper.txt:1932–1972):
- PPO clip threshold: 0.02 (paper 값; 일반적 0.2와 다름 — paper는 작은 stepsize).
- GAE(λ) = 0.95, TD(λ) = 0.95, SGD momentum 0.9.
- Discount γ = 0.95 (single-clip imitation), 0.99 (task).
- Policy / value / disc stepsize는 1e-5 ~ 1e-4 범위로 매우 작음 (SGD with momentum, **Adam 아님**).
- Discriminator stepsize 10^-5.

### 1.8 PD control / actuator

(paper §8.1, /tmp/amp_paper.txt:820–822):
- 시뮬레이터 Bullet, sim freq = 1.2 kHz, **policy freq = 30 Hz**.
- Action = PD controller의 **target joint position**. → motion dataset이 녹화된 character와 PD gain / mass 도메인이 일치해야 의미가 있다 (sim-to-sim mismatch 시 style reward signal이 왜곡).

### 1.9 LSGAN vs BCE (구현 차이 주의)

Paper는 LSGAN least-squares form을 명시하지만, 일부 reference 구현 (MimicKit `mimickit/learning/amp_agent.py:220–228`)은 **BCEWithLogitsLoss** + sigmoid logit, style reward는 `r = -log(1 - sigmoid(D))` (`amp_agent.py:242–244`) — 즉 GAIL-original 형태로 구현되어 있다. Paper Eq.7과 동치는 아니지만, label-smoothing + GP가 들어가면 비슷한 안정성. 비교 시 "어느 loss form을 쓰는가"는 **구현 검증 항목**으로 둔다 (둘 다 valid).

---

## 2. MimicKit 프로젝트 개요

**Repo 루트**: `/home/lgb/MimicKit/`
**Starter Guide / 자체 논문**: Peng 2025, *MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control*, arXiv: https://arxiv.org/abs/2510.13794 (MimicKit/README.md:9, CITATION 항목 README.md:147–156).

### 2.1 무엇인가
- Xue Bin Peng (AMP 저자) 그룹이 직접 유지하는 **motion imitation 알고리즘 모음 코드베이스**. ProtoMotions의 경량 버전 (`README.md:9`).
- 지원 알고리즘: DeepMimic, **AMP**, AWR, ASE, LCP, ADD (`README.md:12–17`).
- 지원 엔진: Isaac Gym, **Isaac Lab**, Newton (`README.md:25–48`). Isaac Lab은 commit `2ed331acfcbb1b96c47b190564476511836c3754`에서 테스트 (`README.md:37`).
- 지원 character: Humanoid, G1, Pi Plus, SMPL, **Go2** (Unitree quadruped) (`CLAUDE.md:9–17`).
- Go2용 AMP env / agent config:
  - env: `data/envs/amp_go2_env.yaml`, `amp_go2_tracking_env.yaml`, `amp_go2_steering_env.yaml`, `amp_go2_location_env.yaml`
  - agent: `data/agents/amp_go2_agent.yaml`, `amp_go2_task_agent.yaml`

### 2.2 코드 계층 (`CLAUDE.md:58–75`)

```
BaseEnv → SimEnv → CharEnv → DeepMimicEnv → AMPEnv → TaskSteeringEnv
                                       → ADDEnv
BaseAgent → PPOAgent → AMPAgent → ADDAgent
                              → ASEAgent
```

→ AMPEnv는 DeepMimicEnv를 상속하므로 **RSI / early termination / key-body 추적 / motion lib 로딩**이 DeepMimic에서 그대로 상속된다.

### 2.3 Go2 task 적용 (`data/envs/amp_go2_tracking_env.yaml`)

발췌 (`amp_go2_tracking_env.yaml:1–60`):
- `num_disc_obs_steps: 10` → discriminator 입력은 단순 (s, s′) 한 쌍이 아니라 **10-step transition window**.
- `key_bodies: ["FR_foot","FL_foot","RR_foot","RL_foot"]` → 발 4개 위치를 Φ(s)에 포함.
- `enable_early_termination: True` → AMP paper 권장과 일치.
- `rand_reset: True` → RSI 활성화 (motion clip에서 random initial state 샘플).
- `enable_phase_obs: False`, `enable_tar_obs: False`, `global_obs: False`, `root_height_obs: False` → **discriminator 입력에서 phase/goal/global 위치 정보 제거** (paper §5.3 권장과 일치).
- `hip_scale_reduction: true, factor: 0.5` → hip joint action 출력 스케일 절반 (Go2에서 yaw drift 억제용).
- Task reward: tracking command (`lin_x_vel`, `ang_yaw_vel`) + pose/vel/key_pos tracking, 합 = 1.0.

### 2.4 Agent hyperparameters (`data/agents/amp_go2_task_agent.yaml`)

```
disc_grad_penalty: 5        # paper 값 10에서 절반
disc_logit_reg: 0.01        # L2 on logit weights
disc_reward_scale: 2        # paper에는 없는 추가 scaling
disc_buffer_size: 200000    # paper 1e5보다 2배 큼
disc_replay_samples: 1000
task_reward_weight: 0.5     # paper 권장과 일치
disc_reward_weight: 0.5     # paper 권장과 일치
ppo_clip_ratio: 0.2         # paper 0.02보다 크지만 통상적
td_lambda: 0.95             # paper와 동일
discount: 0.99              # task 모드 paper 값과 일치
disc_epochs: 2, disc_batch_size: 2  # epoch x batch (실제 minibatch는 4 x bs)
actor_optimizer.type: SGD, lr 2e-4
disc_optimizer.type: SGD, lr 2.5e-4, weight_decay 1e-4
action_std: 0.1 (FIXED)     # exploration std
action_bound_weight: 10.0
```

(amp_humanoid_agent.yaml은 `task_reward_weight: 0.0, disc_reward_weight: 1.0` — 순수 single-clip imitation, AMP 본문 single-clip 세팅과 일치.)

### 2.5 Loss 구현체 (`mimickit/learning/amp_agent.py`)

- `_disc_loss_pos/neg` → **BCEWithLogitsLoss** (Eq.6 LSGAN과 다른 GAIL form, line 220–228).
- GP는 **demo + policy 양쪽**에서 zero-centered로 계산 후 평균 (`amp_agent.py:185–195`) — paper는 demo only지만 양쪽도 흔한 변형, ranking에서 critical 아님.
- Style reward: `r = -log(1 - sigmoid(D))` (line 242–244), Eq.7 clip form과는 다르지만 dense·양수 보장.
- Discriminator는 `disc_epochs × disc_batch_size` (2×2 = 4 minibatch / iter)로 학습.
- Replay buffer (`amp_agent.py:53–54, 100–114, 169`)에서 demo와 policy를 1:1로 샘플.

---

## 3. AMP 학습 성공의 필수 체크리스트 (비교 채점에 사용)

각 항목은 **paper-spec 근거**와 **위반 시 증상**을 함께 제시. IsaacLab 측 / MimicKit 측 각각에 ✓/✗를 매기는 데 사용.

| # | 체크 항목 | 왜 중요한가 (paper 근거) | 위반 시 증상 |
|---|----------|------------------------|--------------|
| C1 | **Discriminator input이 character-local frame**으로 변환 (root vel/ang vel/end-effector pos 모두 local) | §5.3, line 548–564: 모든 vel·position이 root facing frame. world frame이면 yaw에 따라 같은 모션이 다른 feature로 보임 | yaw drift, heading-dependent style, disc collapse |
| C2 | **Discriminator input에 task/goal/phase feature 미포함** | §5.3, line 569–574: "observations do not include any task-specific features" | style이 goal에 종속, command 변경 시 reward 폭주, mode collapse |
| C3 | **Discriminator loss = LSGAN(±1) 또는 BCE + zero-centered GP on real samples** | §5.4 Eq.8, line 592–607 | GP 없으면 disc 폭주, real-side GP 누락 시 generator overshoot |
| C4 | **`w_gp ≈ 10` 정도** (5~20 권장; 0이나 0.1 같은 미세값은 위험) | §8.1, line 825 (`w_gp=10`); MimicKit Go2는 5 사용 | GP 너무 작 → disc 폭주, style reward→0; 너무 크 → disc 학습 안 됨 |
| C5 | **Style reward `r_S = max(0, 1 − 0.25·(D−1)^2)` 또는 동치 dense·양수 형태** | Eq.7, line 530 | reward range·sign이 바뀌면 PPO advantage 분포 왜곡 |
| C6 | **Total reward = w_G·r_G + w_S·r_S, magnitude 같은 order** (paper 0.5/0.5) | §6.3 Eq.4, §8.2 line 867–868 | task만 따라가 style 무시, 또는 demo poses에 갇혀 task 실패 |
| C7 | **Reference state initialization (RSI) 사용** — motion clip에서 random frame으로 reset | §6.3 line 734–737 | cold-start 시 demo distribution과 mismatch → disc가 trivially policy 구분 → reward 0 → 학습 정지 |
| C8 | **Early termination** (발 이외 contact 시 종료) — RSI와 짝 | §6.3 line 738–741 | 종료 너무 늦으면 fallen pose가 replay buffer에 누적 → disc가 fall 자체를 negative로 학습해 정상 motion까지 음의 score |
| C9 | **Demo:policy 샘플 1:1 매 disc update** + replay buffer (1e5 order) | §6.3 line 742–751, Table 4 line 1966 | 1:N 비대칭 시 disc 편향, replay 없으면 disc가 가장 최근 batch에 overfit |
| C10 | **Disc는 매 PPO update와 동기**(시간당 disc step ≈ policy step order) | Algorithm 1 line 634–730 | disc가 너무 자주 update → 폭주, 너무 드물 → reward stale |
| C11 | **PD gain / armature / mass가 motion이 녹화된 character와 일치** (Go2의 경우 retarget된 motion이면 더 critical) | §8.1 line 820–822 (PD controller, 30Hz policy) | sim-to-sim mismatch → demo state distribution에 도달 불가 → disc reward 항상 낮음 |
| C12 | **Disc obs / state obs 모두 running normalization** (mean/std) | §5.3·§6.1, MimicKit `amp_agent.py:166,172` `_disc_obs_norm.normalize` | unnormalized 입력 + GP → gradient scale 폭주, disc lr 튜닝 무력화 |
| C13 | **Action = default joint position offset에 더해진 PD target** (motion data와 같은 도메인) | §6.1 line 626–632; Go2 retarget motion은 default pose 기준 offset이 일반적 | action이 absolute target인데 motion이 relative이거나 그 반대면 style reward signal 모두 noise |
| C14 | **Disc obs window가 (s, s′) 한 쌍 또는 num_disc_obs_steps>1 stacking 일관** | MimicKit `amp_go2_tracking_env.yaml:12` `num_disc_obs_steps: 10` (multi-step variant) | 한쪽이 1-step, 다른 쪽이 multi-step이면 demo와 policy 분포가 구조적으로 다름 → disc trivially 분리 |

> 비교 우선순위: **C1, C2, C4, C7, C9, C12, C13, C14**가 "어긋나면 학습 자체가 안 됨" 그룹 — 이 8개를 isaac/mimickit 양측 체크. 나머지는 미세 튜닝 영역.

---

## 4. 자주 보고되는 실패 원인 카탈로그

각 항목: **원인 — 증상 — 진단 방법**.

1. **Discriminator collapse (perfect discriminator)**
   - 원인: GP λ 너무 작거나 0, disc capacity가 policy 대비 과대, disc lr가 policy lr 대비 너무 큼.
   - 증상: `disc_agent_acc → 1.0`, `disc_demo_acc → 1.0`, style reward → 0, episode return 평탄.
   - 진단: log에 `Disc_Agent_Acc`/`Disc_Demo_Acc` (`MimicKit/CLAUDE.md:81–86` 권장 범위 0.5–0.8) 추적. 둘 다 >0.9면 collapse. (보강: https://www.emergentmind.com/topics/adversarial-motion-priors-amp)

2. **Mode collapse on multi-clip dataset**
   - 원인: dataset에 다양한 skill (trot+gallop+turn)이 섞여 있고 disc가 그중 한 mode만 학습.
   - 증상: agent가 한 가지 gait만 사용, dataset 다양성을 못 살림.
   - 진단: motion lib에서 sampling되는 clip별 style reward 평균을 분리해서 plot. (보강: https://arxiv.org/html/2509.21810 Conditional AMP 논문이 이 문제 motivation)

3. **Reward magnitude imbalance**
   - 원인: task reward가 [0,10] 같은 큰 scale인데 style reward는 [0,1]. (또는 그 반대.)
   - 증상: 학습이 task만 따라가 demo 무시 (task=10, style=0.1), 또는 정반대.
   - 진단: log의 평균 `r_task`, `r_style` 비교. 같은 order인지 확인.

4. **Motion data normalization / domain mismatch**
   - 원인: motion이 다른 character로 녹화된 후 retarget되었는데 joint 순서·축·scale이 코드와 다름. 또는 root height/offset이 motion clip과 env 초기 pose 사이 어긋남.
   - 증상: RSI 직후 character가 ground를 통과하거나 점프 — early termination이 즉시 trigger → episode 길이 ~1, disc 학습 무력화.
   - 진단: view_motion 또는 motion preview로 dataset 시각화. RSI 직후 z-height, joint angle log.

5. **PD gain / actuator mismatch (sim-to-sim)**
   - 원인: motion이 강한 PD로 트래킹돼서 만들어졌는데 sim에서 약한 PD (낮은 stiffness, 큰 damping, 작은 effort limit) 사용. 또는 armature/inertia 값 차이.
   - 증상: 같은 joint target을 줘도 실제 trajectory가 motion보다 느려 항상 disc에 lag → style reward 낮게 고정.
   - 진단: motion clip의 desired joint pos vs 실제 joint pos 비교 plot. (Go2의 경우 deploy 제약 때문에 PD가 인위적으로 약하면 더 자주 발생.)

6. **Early termination too aggressive**
   - 원인: 발 이외 contact 외에 추가 종료 조건 (height < X, roll > θ 등)이 motion에 등장하는 자연 pose까지 잡아냄.
   - 증상: episode length 짧고 균일, replay buffer가 같은 시작 segment로 채워짐, disc가 segment-bias 학습.
   - 진단: termination cause별 카운트, episode length 분포.

7. **RSI 미사용**
   - 원인: env가 `rand_reset: False` 또는 항상 default standing pose에서 시작.
   - 증상: cold-start 직후 disc는 standing만 보고, motion의 dynamic phase는 거의 못 봄 → style reward signal stale.
   - 진단: env config에서 `rand_reset` / `enable_phase_obs` / motion lib initial sampling 확인.

8. **Discriminator input에 task feature가 섞임**
   - 원인: AMP env가 obs를 `[motion_obs, command, phase]` 통째로 disc에 넘김.
   - 증상: command/phase 분포가 demo와 policy에서 다르므로 disc가 style 대신 task feature로 trivial하게 분리 → style signal 0.
   - 진단: disc input dim, 그 안에 command 차원이 들어가 있는지 확인 (MimicKit `amp_go2_tracking_env.yaml`은 `enable_tar_obs:False`로 명시 분리).

9. **World vs character frame**
   - 원인: root linear velocity를 world frame으로 disc에 넣음.
   - 증상: 같은 trot이라도 yaw 각도가 다르면 disc 입력이 달라져 disc가 yaw 자체를 학습 → policy는 학습된 yaw만 사용 (yaw drift / preferred heading).
   - 진단: disc obs 추출 함수에 `quat_rotate_inverse(root_quat, vel)` 같은 local 변환 코드가 있는지 확인.

10. **Replay buffer 미사용 또는 너무 작음**
    - 원인: 매 PPO iteration에서 가장 최근 batch만 disc에 사용.
    - 증상: disc loss가 진동, disc accuracy oscillation.
    - 진단: `disc_buffer_size`, `disc_replay_samples` 설정 확인.

11. **action / motion data 도메인 차이 (default offset)**
    - 원인: policy action이 default joint position에 더해지는 offset인데, motion data는 absolute joint target. 또는 반대.
    - 증상: 같은 action=0이 motion의 standing이 아니라 ground crash.
    - 진단: env step에서 PD target 계산식 grep, motion lib initial pose 비교.

12. **observation normalization off**
    - 원인: disc obs / state obs running stats 미적용. 특히 GP가 raw scale에서 계산되면 gradient norm이 폭주.
    - 증상: disc loss NaN, GP loss가 main loss 압도, 또는 disc lr를 아무리 낮춰도 학습 안 됨.
    - 진단: `_disc_obs_norm.normalize` 호출 / `EmpiricalNormalization` 위치 확인.

13. **num_disc_obs_steps mismatch**
    - 원인: env yaml에서 `num_disc_obs_steps=10`인데 코드에서 single-step (s,s′)만 stack.
    - 증상: disc input dim mismatch 또는 demo와 policy의 window 길이가 다름 → trivial 분리.
    - 진단: disc network input dim과 env가 만들어주는 disc obs tensor shape 일치 확인.

---

## 5. 출처

- **Peng et al. 2021**, *AMP: Adversarial Motion Priors for Stylized Physics-Based Character Control*, arXiv: https://arxiv.org/abs/2104.02180, PDF (로컬 추출본 `/tmp/amp_paper.txt`). 인용 line은 본 추출본 기준.
- **AMP Project page**: https://xbpeng.github.io/projects/AMP/
- **Peng 2025**, *MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control*, arXiv: https://arxiv.org/abs/2510.13794
- **MimicKit repo**: `/home/lgb/MimicKit/`
  - `README.md` (overview)
  - `CLAUDE.md` (구조 및 로그 지표 권장 범위)
  - `docs/README_AMP.md`
  - `data/agents/amp_humanoid_agent.yaml`, `amp_go2_task_agent.yaml`
  - `data/envs/amp_humanoid_env.yaml`, `amp_go2_tracking_env.yaml`
  - `mimickit/learning/amp_agent.py` (disc loss / GP / replay 구현)
- **AMP failure modes 보강 (web)**:
  - https://www.emergentmind.com/topics/adversarial-motion-priors-amp (overview of training instabilities & GP role)
  - https://arxiv.org/html/2509.21810 (Conditional AMP — multi-skill mode collapse 문제 motivation)
  - https://arxiv.org/html/2407.02282v1 (Bipedal-on-quadruped AMP, Go2-인접)
  - https://www.mdpi.com/2076-3417/15/6/3356 (BCAMP — Go2 AMP variant)
  - https://skrl.readthedocs.io/en/latest/api/agents/amp.html (skrl AMP 구현 reference)

> **추정 표시**: 본 노트의 모든 핵심 수식·체크리스트 항목은 paper 본문 또는 MimicKit 코드 line으로 grounding 되었음. C11 (PD gain mismatch)의 Go2 특이성, C13 (action offset 도메인) 등 일부 일반론은 "흔한 sim-to-sim 실패 유형"으로 web 출처 + 통상 관행 기반 — 비교 채점 시 "추정"으로 표시 권장.

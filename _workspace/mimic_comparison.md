# AMP (Adversarial Motion Priors) 구현 비교 분석

**분석 일자**: 2026-04-02
**비교 대상**:
- **현재 구현** (IsaacLab Go2AMP): `/home/lgb/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/direct/go2_amp/`
- **참조 구현** (MimicKit): `/home/lgb/MimicKit/`

---

## A. 알고리즘 (PPO-AMP)

### A-1. Discriminator 구조

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| 레이어 구성 | Linear(55×2→1024)→ReLU→Linear(1024→512)→ReLU→Linear(512→1) | Linear(disc_obs→1024)→act→Linear(1024→512)→act→Linear(512→1) |
| 기본 hidden dims | [1024, 512] | [1024, 512] (fc_2layers_1024units.py) |
| Activation | ReLU (rsl_rl MLP 기본값) | 설정 가능 (`activation` 파라미터, 기본 ELU 계열) |
| 입력 크기 | `amp_observation_space=55` × `num_amp_observations=2` = **110** | `num_disc_obs_steps=10` × obs_per_step (환경 의존) |
| 출력 | 스칼라 logit (1) | 스칼라 logit (1) |
| 가중치 초기화 | MLP 기본값 | Uniform(-1.0, 1.0) + bias=0 (disc_logits 레이어만) |
| 입력 정규화 | `EmpiricalNormalization` (rsl_rl 내장) | 별도 `Normalizer` 클래스 (온라인 평균·분산 추적) |

**핵심 차이:**
- MimicKit은 disc_logits(출력 레이어) 가중치를 Uniform(-1.0, 1.0)으로 별도 초기화하여 초기 logit이 무작위 분포를 갖도록 한다.
- MimicKit의 Normalizer는 `clip=10.0`이 명시적으로 설정되어 있으나, 현재 구현의 EmpiricalNormalization은 clip 값 설정 여부가 불명확.
- MimicKit은 obs normalizer(policy)와 disc_obs normalizer를 **별개로** 유지하며, disc_obs_norm에 expert demo 데이터도 함께 포함시켜 업데이트한다.

---

### A-2. Discriminator 학습 방식

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| Loss 함수 | **LS-GAN** (MSELoss, expert=+1, policy=-1) | **BCE** (BCEWithLogitsLoss, expert=1.0, policy=0.0) |
| Loss 수식 | `0.5*(MSE(D(expert),+1)+MSE(D(policy),-1)) + GP` | `0.5*(BCE(D(expert),1)+BCE(D(policy),0)) + GP + logit_reg` |
| Gradient Penalty 적용 대상 | **expert 데이터만** | **expert + policy 데이터 양쪽** |
| Gradient Penalty 계산 | `‖∇D(expert)‖₂²` (L2 norm squared) | `0.5*(mean(∑dx²_demo) + mean(∑dx²_agent))` (sum of squares) |
| Gradient Penalty 계수 | `gradient_penalty_coef=5.0` | `disc_grad_penalty=5` |
| Logit 정규화 | 없음 | `disc_logit_reg=0.01` (출력 레이어 가중치 L2) |
| Discriminator 업데이트 주기 | **매 iteration, 1회** (policy 업데이트 전) | 별도 `disc_epochs×disc_batches` 스텝 (매 iteration 복수 업데이트) |
| Replay buffer 사용 | **없음** (현재 iteration 데이터만 사용) | **있음** (`disc_buffer_size=200000`, `disc_replay_samples=1000`) |

**핵심 차이:**
- **Loss 함수**: 현재 구현은 LS-GAN (더 안정적인 학습, gradient vanishing 감소), MimicKit은 원본 AMP 논문의 BCE. LS-GAN은 학습 안정성 측면에서 유리하나 보상 스케일이 다름.
- **Gradient Penalty 범위**: MimicKit은 expert+policy 양쪽에 gradient penalty를 적용해 더 강한 정규화 효과를 얻음. 현재 구현은 expert만 적용.
- **Replay Buffer 부재**: 현재 구현에는 replay buffer가 없어 discriminator가 과거 데이터로 정규화되지 않음. MimicKit은 200K 크기의 replay buffer에서 1000개를 추가 샘플링하여 catastrophic forgetting 방지.
- **Disc 업데이트 횟수**: MimicKit은 `disc_epochs=2, disc_batch_size=4`로 매 iteration에 복수 gradient step 수행. 현재 구현은 1회만.

---

### A-3. PPO Policy 학습 방식 (AMP reward 통합)

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| AMP 보상 공식 | LS-GAN 기반: `clamp(1 - 0.25*(D-1)², 0) × amp_reward_coef` | BCE 기반: `-log(max(1-sigmoid(D), 0.0001)) × disc_reward_scale` |
| Task reward 통합 | Lerp: `lerp×r_task + (1-lerp)×r_amp` (annealing) | 가중합: `task_weight×r_task + disc_weight×r_disc` |
| 초기 단계 전략 | Task reward 위주 시작 (`lerp_start=0.95`) → AMP로 점진 전환 | `task_reward_weight=0.0`, `disc_reward_weight=1.0` (pure AMP) |
| Annealing 지원 | 있음 (10000 iter, lerp 0.95→0.3) | 없음 (고정 비율) |
| PPO 업데이트 | `num_learning_epochs=5, num_mini_batches=4` | `actor_epochs=5, critic_epochs=2` (actor/critic 분리) |
| Actor/Critic 분리 | 아니오 (joint update) | 예 (actor_optimizer, critic_optimizer 별도) |
| Value function target | TD-lambda (GAE 기반) | TD-lambda (`td_lambda=0.95`) |
| Advantage 정규화 | GAE 기반 (RSL-RL 표준) | 전체 배치 정규화 후 clip (`norm_adv_clip=4.0`) |
| Action bound loss | 없음 | 있음 (`action_bound_weight=10.0`) |
| Action entropy | `entropy_coef=0.007` | `action_entropy_weight=0.0` (비활성) |
| Exploration | Gaussian noise (고정 std) | 확률적 탐색 (`exp_anneal_samples` decay 지원) |

**핵심 차이:**
- **Annealing 전략**: 현재 구현은 task reward 위주 초기화 후 AMP로 전환하는 2단계 학습 방식 채택. MimicKit은 처음부터 pure AMP(task_reward_weight=0).
- **Actor/Critic 분리 최적화**: MimicKit은 actor와 critic을 별도 optimizer로 관리하여 학습률 독립 제어 가능.
- **Action Bound Loss**: MimicKit은 action이 범위를 벗어날 경우 패널티를 부여(weight=10.0). 현재 구현에는 없음.

---

### A-4. Replay Buffer / Motion Sampling 전략

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| Policy replay buffer | 없음 | `disc_buffer_size=200000` (CircularBuffer 기반) |
| Demo sampling | 매 iter `env.get_amp_observations()` 직접 호출 | `env.fetch_disc_obs_demo(n)`으로 motion_lib에서 온-더-플라이 샘플링 |
| 속도 균등 샘플링 | 있음 (`_build_velocity_sample_weights`, 5구간 균등화) | 없음 (길이 비례 가중치로 motion 선택 후 uniform time) |
| 다중 모션 파일 | 있음 (전체 concatenate 후 통합 샘플링) | 있음 (motion별 가중치 부여, `multinomial` 선택) |
| 시간 보간 | Linear interpolation + SLERP (quaternion) | Linear interpolation + SLERP (별도 torch_util) |
| Loop 모드 | 없음 (clamp only) | WRAP / CLAMP 선택 가능, WRAP 시 position offset 보정 |

**핵심 차이:**
- **Replay Buffer**: MimicKit의 가장 큰 장점 중 하나. Policy 과거 데이터를 buffer에 쌓아 discriminator가 "과거 실수"를 기억하게 함. 현재 구현 부재로 discriminator가 현재 iteration 데이터에만 의존.
- **속도 균등 샘플링**: 현재 구현은 학습 중 특정 속도 구간 편향을 방지하는 velocity-balanced sampling을 독자적으로 구현. MimicKit에는 없는 기능.
- **Loop Mode**: MimicKit은 모션이 끊김 없이 반복될 때(WRAP) root position offset을 자동 보정. 현재 구현은 모션 끝에서 clamp.

---

## B. 시뮬레이션 환경

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| 시뮬레이터 | Isaac Sim (Isaac Lab DirectRLEnv) | Isaac Gym / Isaac Lab / Newton (엔진 추상화) |
| 로봇 모델 | Unitree Go2, **12 DOF** | Unitree Go2 (MJCF go2.xml), **12 DOF** |
| Physics dt | 1/200 Hz (5ms) | 엔진 설정 의존 (Isaac Gym 기준 ~1/60 Hz) |
| Decimation | 4 (정책 주기 50Hz) | 직접 설정 없음 (step_per_iter로 제어) |
| 에피소드 길이 | 20.0초 | 10.0초 |
| Ground friction | static=1.0, dynamic=1.0 | 설정 파일 미명시 (`ground_contact_height=0.15`만 지정) |
| 조기 종료 조건 | 높이<0.15m OR 뒤집힘(gravity_z>0) OR base 충돌 | root_rot 이탈 OR body_pos 이탈 OR 바닥 외 접촉 |
| 포즈 기반 종료 | 없음 | `pose_termination=False` (옵션 있음, 비활성) |
| RSI (Reference State Init) | 있음 (`reset_strategy=random`) | 있음 (`rand_reset=True`) |
| 리셋 전략 | `default` / `random` / `random-start` 선택 | random (motion_lib에서 시간 샘플링) |
| 환경 수 | 4096 (default) | 32 (test_episodes), 학습 시 다름 |
| 물리 파라미터 무작위화 | Domain Randomization 없음 (현재) | 없음 |

**핵심 차이:**
- **엔진 추상화**: MimicKit은 Isaac Gym / Isaac Lab / Newton을 모두 지원하는 추상 엔진 레이어를 보유. 현재 구현은 Isaac Lab에 특화.
- **포즈 기반 종료**: MimicKit의 DeepMimicEnv는 레퍼런스 모션 대비 포즈 차이가 임계값 초과 시 종료하는 `pose_termination` 옵션 보유. 현재 구현에는 없음.
- **에피소드 길이**: 현재 구현이 2배 긴 에피소드(20초)로, 더 다양한 속도 조건을 경험할 수 있으나 학습이 느려질 수 있음.

---

## C. Observation

### C-1. Policy Observation

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| 구성 | gravity_b(3)+commands(3)+joint_pos_diff(12)+joint_vel(12)+actions(12) = **42** | 환경별 상이 (deepmimic_env 기반, body-relative obs) |
| 속도 명령 포함 | 있음 (lin_vel_x,y + ang_vel_yaw 3개) | 없음 (pure imitation, 속도 목표 없음) |
| 이력 관측치 | 있음 (`history_len=50` 프레임 → 별도 history 텐서) | 없음 |
| Privileged 관측치 | 있음 (lin_vel+ang_vel+mass+material 총 74차원) | 없음 (standard PPO) |
| 정규화 | Actor/Critic별 EmpiricalNormalization | 별도 Normalizer (`normalizer_samples=100M`) |

### C-2. AMP Observation (Discriminator 입력)

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| 구성 | dof_pos(12)+dof_vel(12)+root_height(1)+lin_vel(3)+ang_vel(3)+key_body_pos(12)+key_body_vel(12) = **55** | root_vel(3)+root_ang_vel(3)+dof_vel(num_dof)+[pos_obs(relative)] = 환경별 |
| 시간 스텝 수 | `num_amp_observations=2` (현재+이전 1프레임) | `num_disc_obs_steps=10` (10프레임 이력) |
| 루트 위치 표현 | 절대 높이(z)만 사용 | body-relative 위치 (heading-aligned 로컬 프레임) |
| key body 포함 | 발 4개 위치 + 속도 (로컬 프레임) | key_bodies=[FL,FR,RL,RR] (로컬 프레임, FK 기반) |
| 관절 표현 | DOF 각도/속도 (Euler 기반) | joint rotation (quaternion 기반, `dof_to_rot` 변환) |
| 좌표계 | 루트 쿼터니언 역변환으로 로컬 변환 | heading quaternion 역변환으로 heading-aligned 로컬 |

**핵심 차이:**
- **시간 이력 깊이**: MimicKit은 10프레임(~167ms @ 60Hz)의 이력을 쌓아 temporal pattern 학습. 현재 구현은 2프레임만 사용하여 temporal information이 제한적.
- **관절 표현**: MimicKit은 DOF를 quaternion rotation으로 변환하여 불연속성이 없는 표현 사용. 현재 구현은 Euler angle(DOF) 직접 사용.
- **속도 관측**: MimicKit은 heading-aligned local frame으로 변환하여 방향 독립적인 표현 제공. 현재 구현은 body frame 직접 사용.

### C-3. Observation 정규화

| 항목 | 현재 구현 (IsaacLab Go2AMP) | MimicKit |
|------|---------------------------|----------|
| 방법 | `EmpiricalNormalization` (rsl_rl 내장 running stats) | `Normalizer` 클래스 (온라인 mean/variance, clip=10.0) |
| AMP obs 별도 정규화 | 있음 (`amp_obs_normalizer`) | 있음 (`disc_obs_norm`, expert+policy 혼합) |
| 분산 학습 시 정규화 동기화 | `mp_util.reduce_inplace_sum` 지원 | `mp_util`로 multi-process 동기화 |

---

## D. Motion Loader

### D-1. 지원 데이터 포맷

| 항목 | 현재 구현 (Go2MotionLoader) | MimicKit (MotionLib) |
|------|---------------------------|---------------------|
| 지원 포맷 | `.txt` (DeepMimic JSON), `.pkl` (MuJoCo) | `.pkl` (MimicKit 전용 pickle), `.yaml` (모션 목록) |
| 파일 목록 지정 | 디렉토리 또는 파일 직접 지정 | `.yaml` 파일로 가중치 포함 목록 관리 |
| 데이터 구조 (txt) | 61값/프레임 (root_pos+rot+joint_pos+toe_pos+vel 등) | 해당 없음 |
| 데이터 구조 (pkl) | 18값/프레임 (root_pos+euler+joint_pos, 속도 자동 계산) | 프레임별 (root_pos+root_rot_expmap+joint_dof) |
| 회전 표현 | quaternion (wxyz) | exponential map → quaternion 변환 |

### D-2. 모션 데이터 구조

| 항목 | 현재 구현 (Go2MotionLoader) | MimicKit (MotionLib) |
|------|---------------------------|---------------------|
| 루트 상태 | position(3) + quaternion(4) + lin_vel(3) + ang_vel(3) | position(3) + quaternion(4) + 속도는 처리 시 계산 |
| 관절 표현 | DOF angle (Euler) 12개 | joint rotation (quaternion, `dof_to_rot` 변환) |
| 발 위치 | toe_pos 직접 저장 + FK 계산 (pkl의 경우) | 순방향 운동학 (`forward_kinematics` from kin_char_model) |
| 발 속도 | finite difference로 자동 계산 | `compute_frame_dof_vel` (처리 단계에서 계산) |
| 속도 계산 방법 | forward finite difference | frame_fps × (pos[t+1]-pos[t]) (FPS 기반) |

### D-3. 프레임 샘플링 방식

| 항목 | 현재 구현 (Go2MotionLoader) | MimicKit (MotionLib) |
|------|---------------------------|---------------------|
| 모션 선택 | 단일 concatenated 데이터셋 (균등 샘플링) | 모션별 가중치 기반 multinomial 선택 |
| 시간 샘플링 | **velocity-balanced**: 속도 구간 균등화 or uniform | uniform in [0, motion_length] |
| 보간 | Linear + SLERP (quaternion) | Linear + SLERP |
| Loop 처리 | clamp (경계에서 마지막 프레임) | WRAP / CLAMP 모드, WRAP 시 position offset 누적 |
| 다중 모션 처리 | 파일별 concatenate 후 통합 관리 | 모션별 start_idx 관리, `motion_id` 기반 인덱싱 |

### D-4. 데이터 증강

| 항목 | 현재 구현 (Go2MotionLoader) | MimicKit (MotionLib) |
|------|---------------------------|---------------------|
| 미러링 | 없음 | 없음 |
| 속도 균등화 | 있음 (5구간 velocity bin 기반 가중치) | 없음 |
| 노이즈 추가 | 없음 | 없음 |
| Loop offset 보정 | 없음 | WRAP 모드에서 position delta 보정 |

---

## 종합 비교 요약

### 핵심 차이점 요약 표

| 구분 | 현재 구현 특징 | MimicKit 특징 | MimicKit 장점 |
|------|--------------|--------------|--------------|
| **Disc Loss** | LS-GAN (MSE) | BCE (원본 AMP) | BCE가 확률 해석 명확, LS-GAN이 학습 안정성 유리 |
| **Gradient Penalty** | expert만 | expert+policy 양쪽 | 더 강한 정규화로 discriminator 안정화 |
| **Replay Buffer** | 없음 | 200K 크기 replay buffer | Catastrophic forgetting 방지, 더 일관된 disc 학습 |
| **Disc 업데이트** | 1회/iter | 복수 epoch | Disc가 policy를 더 잘 추적 |
| **Logit 정규화** | 없음 | disc_logit_reg=0.01 | Discriminator 과포화 방지 |
| **AMP obs 이력** | 2프레임 | 10프레임 | Temporal pattern 인식 강화 |
| **관절 표현** | Euler DOF | Quaternion rotation | 불연속성 없는 표현 |
| **보상 통합** | Lerp + annealing | 고정 가중합 | Annealing이 초기 수렴 안정화 |
| **Task reward** | 속도 tracking 포함 | Pure imitation (0.0) | 현재 구현이 조종 가능성 우위 |
| **velocity sampling** | velocity-balanced | 균등 random | 현재 구현이 속도 다양성 커버리지 우위 |
| **Loop 처리** | clamp only | WRAP/CLAMP 모드 | 순환 모션 seamless 재생 |
| **Actor/Critic opt** | 공유 optimizer | 분리 optimizer | 학습률 독립 제어 가능 |
| **포즈 종료 조건** | 없음 | 있음 (옵션) | 심각한 pose drift 조기 종료 가능 |
| **엔진 지원** | Isaac Lab 전용 | Isaac Gym/Lab/Newton | 이식성 우위 |

---

## 개선 권장사항

현재 구현에서 MimicKit의 장점을 도입하여 개선할 수 있는 항목:

### 우선순위 High
1. **Replay Buffer 도입**: `disc_buffer_size=100000` 정도의 circular buffer로 과거 policy 데이터를 보존하면 discriminator 학습이 더 안정적으로 됨.
2. **Gradient Penalty 확장**: expert 데이터뿐만 아니라 policy 데이터에도 gradient penalty 적용 (`0.5*(GP_expert + GP_policy)`).
3. **AMP obs 이력 확장**: `num_amp_observations=2` → `10` 이상으로 늘려 temporal pattern 인식 강화. (단, discriminator 입력 크기 증가로 메모리 주의)

### 우선순위 Medium
4. **Disc Logit 정규화**: `disc_logit_reg=0.01` 추가 — discriminator 출력 레이어 가중치 L2 정규화.
5. **Disc 복수 업데이트**: 매 iteration에 discriminator를 2 epoch 이상 업데이트.
6. **Action Bound Loss**: 관절 범위 초과 패널티 추가 (`action_bound_weight=10.0`).

### 우선순위 Low
7. **Loop 모드 지원**: WRAP 모드 구현으로 순환 모션(trot, pace)의 seamless 재생.
8. **관절 표현 quaternion 변환**: AMP obs에서 DOF → quaternion 변환 검토 (구현 복잡도 증가).

---

*이 보고서는 코드 분석 기반으로 작성되었으며, 하이퍼파라미터 최적화는 실험을 통해 검증이 필요합니다.*

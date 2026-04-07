## IL/AMP 디버그 결과 보고서

### 1. `reset` 및 학습 저하 원인 분석 (가장 중요)

**진단 결과**: `reset` 메서드의 좌표계 변환이나 상태 초기화 수학 로직에는 문제가 없습니다. 로봇이 바닥에 기울어진 채 다리만 허우적대는 현상의 근본적인 원인은 **AMP 관측 벡터에서 발 속도(`key_body_lin_vel`)가 제거되었기 때문**입니다.

1. **AMP Observation의 정보 누락 (Local Minimum 발생)**:
   - Git Diff 확인 결과, 최근 수정에서 AMP 관측 공간(55차원 $\rightarrow$ 43차원)을 축소하며 `local_key_body_vel`(발의 로컬 선속도, 12차원)을 제거했습니다.
   - 이로 인해 Discriminator는 발이 올바른 궤적으로 "이동"하고 있는지는 판단하지 못하고, 오직 발의 "위치(`key_body_pos`)"만 보게 됩니다.
   - 결과적으로 로봇은 걷기 위해 균형을 잡는 어려운 학습을 포기하고, **넘어진 상태에서 발의 위치만 흉내 내며 허우적거려도 Discriminator를 속이고 높은 AMP 보상을 받을 수 있는 꼼수(Survival Local Minimum)** 에 빠진 것입니다.

2. **비활성화된 Pose Termination**:
   - `go2_amp_env.py`에 Reference Pose와 실제 로봇의 Pose 거리가 멀어지면 에피소드를 종료하는 `pose_termination` 로직이 성공적으로 추가되었으나, `go2_amp_env_cfg.py`에서 `pose_termination = False`로 설정되어 있습니다.
   - 허우적대는 상태를 조기에 강제 종료(Terminate) 시키지 못해 Task Reward가 0인 상태로 에피소드가 낭비되고 있습니다.

3. **Reset 로직 검증 (정상)**:
   - `quat_apply(root_rot, body_linear_velocities)`는 `stmr_go2.py`가 출력하는 Base Frame 속도를 World Frame으로 올바르게 변환하고 있습니다.
   - `root_rot` 쿼터니언 포맷(`[w, x, y, z]`) 매핑과 `joint_pos` 관절 순서 매핑 모두 IsaacLab의 규격과 정상적으로 일치합니다.

---

### 2. MimicKit 방법론과의 차이점 및 문제 확인

1. **Observation 구성 차이**:
   - **MimicKit**은 Discriminator가 모션의 동적인 특성(Dynamics)을 정확히 판별할 수 있도록 End-effector(발)의 위치뿐만 아니라 **속도(Velocity)** 도 필수적으로 포함합니다. 현재 `key_body_lin_vel`을 제거한 상태는 MimicKit의 AMP 설계 철학과 맞지 않아 모방 성능이 크게 떨어집니다.

2. **LCP (Lipschitz-Constrained Policies) 구현 상태**:
   - 현재 `ppo_amp.py` 알고리즘은 LCP가 적용되지 않은 표준 PPO입니다.
   - **MimicKit의 LCP**는 Actor 신경망의 출력(Action)이 급격히 튀지 않도록 선형 레이어(Linear Layer)들에 Spectral Normalization을 걸고 Gradient Penalty를 주어 부드러움(Smoothness)을 강제합니다.
   - 허우적대는 현상을 억제하고 발 충격을 줄이는 자연스러운 보행을 원한다면, RSL-RL 프레임워크의 Actor 네트워크(`ActorCritic`)에 LCP (Spectral Norm 등)를 도입하는 추가 확장이 필요합니다.

3. **Discriminator 정규화**:
   - MimicKit에서 사용하는 Logit Regularization(`disc_logit_reg`)과 Gradient Penalty는 현재 `ppo_amp.py`에 올바르게 구현되어 정상 작동 중입니다.

---

### 3. 수정 계획 및 권장 사항 (Claude에게 전달)

1. **[Critical] 발 속도 관측치 복구**:
   - `go2_amp_env.py`의 `compute_obs` 및 `_get_observations`에 `local_key_body_vel` 로직을 다시 추가하십시오.
   - `go2_amp_env_cfg.py`의 `amp_observation_space`를 43에서 55로 롤백해야 합니다.

2. **[Warning] Pose Termination 활성화**:
   - 허우적대는 생존 꼼수를 방지하기 위해 `go2_amp_env_cfg.py`에서 `pose_termination = True`로 변경하여, 레퍼런스 모션에서 크게 벗어나면 에피소드가 초기화되도록 수정하십시오.

3. **[Feature] LCP(Spectral Norm) 포팅 검토**:
   - 모방학습 시 안정성을 극대화하기 위해, RSL-RL의 Policy 네트워크에 MimicKit의 LCP 구조(Spectral Normalization)를 이식하는 작업을 고려하십시오.
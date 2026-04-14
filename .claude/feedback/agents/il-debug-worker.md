# il-debug-worker Feedback Log

## Active Rules
<!-- 최대 10개. session-summarizer에서 관리. agent는 Deprecated 섹션 무시 -->
- [2026-04-03] `projected_gravity_b`에서 roll을 계산할 때 `atan2(gy, gz)` 가 아닌 `atan2(gy, -gz)` 를 사용할 것. 정립 시 gz=-1이므로 gz 부호 반전 없이 계산하면 roll=±π가 출력되어 termination 조건이 항상 만족되는 버그 발생. ZYX Euler 기준: gravity_b = [-sin(p), cos(p)sin(r), -cos(p)cos(r)] → roll = atan2(gy, -gz). (위반 시: 로봇이 정립 상태에서도 즉시 termination)
- [2026-04-03] AMP 환경에서 pose_termination(frame-by-frame reference 비교 종료)은 기본 비활성화 권장. AMP는 distribution matching 방식이므로 특정 시점 reference와의 차이로 종료하는 것은 철학적으로 부적합. 코드는 구현하되 config에서 False로 기본값 설정. (위반 시: AMP 학습 불안정, 과도한 early termination)
- [2026-04-09] AMP 환경에서 RSI 구현 시 robot state 초기화와 AMP 버퍼 초기화에 반드시 동일한 motion_ids를 사용할 것. robot state와 AMP 버퍼가 서로 다른 motion에서 샘플링되면 Discriminator가 reset 직후 데이터를 쉽게 구분 → 조기 수렴 → amp_reward = 0 고착. (위반 시: disc 조기 수렴, policy AMP 신호 완전 소실)
- [2026-04-09] AMP 환경 이식 시 Expert obs와 Policy obs가 동일한 joint 순서를 사용하는지 반드시 확인할 것. IsaacLab `robot.data.joint_names`는 알파벳 또는 URDF 순서이며 motion data DOF_NAMES와 다를 수 있음. 두 obs 간 joint 위치가 다르면 Discriminator가 스타일 차이가 아닌 순서 차이만으로 완벽히 구분 → amp_reward = 0. 기존 구현(go2_amp_env.py)의 `motion_dof_indexes` 패턴 참조. (위반 시: disc 완전 수렴, amp_reward 지속 0)

- [2026-04-13] Go2-Imitation-v0 AMP에서 "policy가 disc를 못 속이는" 문제를 `init_noise_std` 축소로 해결하려 하지 말 것. 사용자 확인 결과 이 하이퍼파라미터 변경은 실질적인 효과 없음. 대신 disc 학습 구조(mini-batch 방식), replay buffer 설계 등 알고리즘 구조적 문제를 먼저 확인할 것.
- [2026-04-13] AMP disc 학습 구조를 MimicKit과 비교할 때 반드시 iteration당 optimizer step 수와 batch size를 함께 확인할 것. `disc_num_epochs=2`가 같아도 MimicKit은 24+ mini-batch step(8192/step)인 반면 large-batch 방식은 2 step(~100K/step). Large-batch는 disc가 한 번에 완벽 수렴 → policy가 따라가기 전에 disc 승리 고착화. Mini-batch(disc_mini_batch_size=4096 등)로 교체하면 disc 수렴 속도 조절 가능. (위반 시: disc 항상 이김, amp_reward=0 지속)
- [2026-04-13] AMP 환경에서 terminal step의 `extras["amp_obs"]`는 post-reset RSI 상태 기반이므로 discriminator에 직접 입력하면 reward 오염 발생. `_reset_idx()` 진입 시점(RSI 전)에서 terminal state obs를 별도 캡처(`_terminal_amp_obs`)하고, runner에서 done env의 amp_obs를 교체해야 함. (위반 시: terminal step에서 amp_reward 인위적 과대평가 → policy 조기 종료 학습)

- [2026-04-14] AMP 환경에서 history_length(과거 프레임 수)를 너무 크게 설정하면 역효과가 날 수 있음. 10프레임 대신 2프레임으로 줄였을 때 학습 결과가 개선된 사례 있음. 긴 히스토리는 노이즈 누적으로 Discriminator 학습을 방해할 수 있으므로, 짧은 히스토리(2~3)부터 시작하여 점진적으로 늘려가는 것을 권장. (위반 시: 과도한 히스토리로 disc 수렴 방해)

## Deprecated
<!-- 감사 추적 전용 -->
- [2026-04-14] AMP Discriminator가 "도메인 갭(Domain Gap) 꼼수"를 부릴 때, Expert obs와 Policy obs에 **완전히 동일한 변환 파이프라인**을 적용하는 것이 핵심 해법임. 특히 X/Y 2D 상대 궤적(`root_pos - ref_root_pos` 후 heading 기준 로컬 변환)을 Expert와 Policy 양쪽에 동일하게 계산해야 하며, 한쪽만 다른 방식으로 계산하면 도메인 갭이 지속됨. (위반 시: disc가 모션 품질이 아닌 수치 오차로 구분 → policy 학습 신호 없음) 변환했으나 딱히 영향 없는 것 같음.

# Global Lessons
<!-- 모든 agent에 적용되는 교차 규칙. 최대 10개 Active Rules. -->

## Active Rules
- [2026-04-07] Replay Buffer처럼 add/sample 함수가 존재해도 실제 호출 경로(runner loop)에서 올바르게 사용되는지 별도 확인할 것. "코드에 구현됨"과 "실제로 동작함"은 다름. (위반 시: 버그가 구현된 것처럼 보이지만 실제로는 비활성화된 상태)
- [2026-04-03] 외부 환경(MimicKit 등) 분석 시 "코드에 존재" vs "실제 활성화" 를 구분할 것. config 기본값과 실제 학습 설정을 함께 확인해야 하며, 코드 존재만으로 동작을 가정하지 말 것. (위반 시: 비활성화 기능 기반 잘못된 구현 이식)
- [2026-04-03] 코드 수정 제안 전 반드시 해당 코드를 직접 읽고 근거를 확인할 것. 추측으로 원인을 제시했다가 사용자가 실제 코드를 지목하면 번복해야 하는 상황이 발생함. (위반 시: 잘못된 분석으로 불필요한 수정 유발)
- [2026-04-03] `projected_gravity_b`에서 roll/pitch 계산 시 부호 주의: roll = atan2(gy, **-gz**). Isaac Lab의 정립 상태에서 gz=-1이므로 부호 반전 없이 atan2(gy, gz)를 사용하면 roll=±π 출력. 이는 센서 좌표계 특성에서 비롯된 함정이며 각도 기반 termination/reward 구현 시 항상 검증 필요. (위반 시: 정상 상태를 이상 상태로 오판)
- [2026-04-06] 외부 구현을 이식할 때 flat tensor vs TensorDict 구조 차이를 반드시 확인할 것. `requires_grad_()`를 flat tensor에 거는 코드를 TensorDict 구조로 이식하면, leaf variable의 위치가 달라져 gradient 의미가 완전히 바뀐다. 이 패턴은 논문 구현 포팅 시 자주 발생하는 silent bug임. (위반 시: 원 논문 의도와 다른 regularization 적용)
- [2026-04-08] 알고리즘/정책 클래스를 교체할 때 러너(Runner)도 반드시 함께 검토할 것. 러너는 정책 클래스와 알고리즘 메서드(hist_encoding, update_dagger 등)에 강하게 결합되어 있으며 하드코딩된 경우가 많다. 정책만 바꾸고 러너를 그대로 두면 AttributeError/TypeError가 발생한다. (위반 시: 런타임 오류로 학습 시작 불가)
- [2026-04-09] 외부 환경(MimicKit 등)을 IsaacLab으로 이식할 때 config 기본값이 실제 학습에서 비활성화 상태인지 반드시 확인할 것. 특히 `task_reward_lerp=1.0`처럼 AMP reward 비중을 0으로 만드는 기본값은 코드가 존재해도 기능이 죽어있는 상태다. (위반 시: 이식 후에도 동일하게 학습 실패)
- [2026-04-09] AMP 환경에서 RSI 구현 시 robot state 초기화와 AMP 버퍼 초기화는 반드시 동일한 motion_ids로 수행할 것. 두 초기화를 별도 함수에서 독립적으로 수행하면 motion_ids 불일치가 발생한다. 패턴: `collect_reference_motions(n, times, motion_ids=motion_ids)` 형태로 motion_ids를 명시적으로 전달. (위반 시: disc 조기 수렴, amp_reward=0 고착)
- [2026-04-10] 외부 AMP 구현 이식 시 Loss 함수와 Reward 수식을 쌍으로 확인할 것. MimicKit BCE: loss target 0/1 + reward=-log(1-sigmoid(logit)); LS-GAN: loss target -1/+1 + reward=clamp(1-0.25*(d-1)²,0). 두 수식이 불일치하면 disc 수렴 방향과 policy gradient 방향이 어긋남. 또한 Gradient Penalty는 expert+policy 양쪽에 적용하는 것이 MimicKit 원본임. (위반 시: disc 수렴해도 policy 학습 신호 없음)
- [2026-04-13] AMP 환경에서 `extras["amp_obs"]`는 `_get_observations()` 호출 시점 기준이므로, reset이 일어난 step에서는 post-reset RSI obs가 담김. Terminal step의 reward는 이전 에피소드 것, amp_obs는 다음 에피소드 것이 되는 에피소드 간 오염이 발생. 수정 패턴: `_reset_idx()`에서 RSI 이전 terminal obs를 별도 저장 → extras에 포함 → runner에서 done env에 교체 적용. (위반 시: terminal step amp_reward 과대평가 → 조기 종료 선호 학습)
- [2026-04-13] 외부 AMP 구현과 disc 학습 루프를 비교할 때 `disc_num_epochs` 수치만 보지 말고 iteration당 실제 optimizer step 수와 batch size를 함께 확인할 것. 같은 epoch 수라도 mini-batch(작은 batch, 많은 step)와 large-batch(큰 batch, 적은 step)는 disc 수렴 속도가 전혀 다름. Large-batch는 disc가 너무 빨리 수렴해 policy gradient 신호가 사라질 수 있음. (위반 시: disc 항상 이김, policy amp 학습 불가)
- [2026-04-06] TensorDict 환경에서 `torch.cat` 결과를 autograd gradient target으로 사용하면 안 됨. `torch.cat`은 호출할 때마다 새 tensor 인스턴스를 생성하므로, act() 내부에서 만든 cat tensor와 외부에서 다시 만든 cat tensor는 다른 computation graph node임. gradient target은 반드시 cat의 재료인 TensorDict leaf tensor(`obs_batch_gp[group]`)여야 함. (위반 시: RuntimeError "not used in the graph")
- [2026-04-14] 히스토리 길이는 짧을수록(2~3 프레임) 노이즈 없이 명확한 신호를 전달하는 경향이 있음. (위반 시: disc가 실제 보행 패턴이 아닌 수치 오차에 의존)

## Deprecated
- [2026-04-14] AMP Discriminator 도메인 갭 해결의 핵심은 Expert와 Policy obs 추출에 **완전히 동일한 변환 파이프라인** 적용. Z축 절대 높이보다 X/Y 2D 상대 궤적(현재 위치 기준 과거 프레임 차분 + heading 로컬 변환)이 더 강력한 "보행 형태" 신호를 제공함. 
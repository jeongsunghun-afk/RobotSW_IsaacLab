# Global Lessons
<!-- 모든 agent에 적용되는 교차 규칙. 최대 10개 Active Rules. -->

## Active Rules
- [2026-04-07] Replay Buffer처럼 add/sample 함수가 존재해도 실제 호출 경로(runner loop)에서 올바르게 사용되는지 별도 확인할 것. "코드에 구현됨"과 "실제로 동작함"은 다름. (위반 시: 버그가 구현된 것처럼 보이지만 실제로는 비활성화된 상태)
- [2026-04-03] 외부 환경(MimicKit 등) 분석 시 "코드에 존재" vs "실제 활성화" 를 구분할 것. config 기본값과 실제 학습 설정을 함께 확인해야 하며, 코드 존재만으로 동작을 가정하지 말 것. (위반 시: 비활성화 기능 기반 잘못된 구현 이식)
- [2026-04-03] 코드 수정 제안 전 반드시 해당 코드를 직접 읽고 근거를 확인할 것. 추측으로 원인을 제시했다가 사용자가 실제 코드를 지목하면 번복해야 하는 상황이 발생함. (위반 시: 잘못된 분석으로 불필요한 수정 유발)
- [2026-04-03] `projected_gravity_b`에서 roll/pitch 계산 시 부호 주의: roll = atan2(gy, **-gz**). Isaac Lab의 정립 상태에서 gz=-1이므로 부호 반전 없이 atan2(gy, gz)를 사용하면 roll=±π 출력. 이는 센서 좌표계 특성에서 비롯된 함정이며 각도 기반 termination/reward 구현 시 항상 검증 필요. (위반 시: 정상 상태를 이상 상태로 오판)
- [2026-04-06] 외부 구현을 이식할 때 flat tensor vs TensorDict 구조 차이를 반드시 확인할 것. `requires_grad_()`를 flat tensor에 거는 코드를 TensorDict 구조로 이식하면, leaf variable의 위치가 달라져 gradient 의미가 완전히 바뀐다. 이 패턴은 논문 구현 포팅 시 자주 발생하는 silent bug임. (위반 시: 원 논문 의도와 다른 regularization 적용)
- [2026-04-06] TensorDict 환경에서 `torch.cat` 결과를 autograd gradient target으로 사용하면 안 됨. `torch.cat`은 호출할 때마다 새 tensor 인스턴스를 생성하므로, act() 내부에서 만든 cat tensor와 외부에서 다시 만든 cat tensor는 다른 computation graph node임. gradient target은 반드시 cat의 재료인 TensorDict leaf tensor(`obs_batch_gp[group]`)여야 함. (위반 시: RuntimeError "not used in the graph")

## Deprecated

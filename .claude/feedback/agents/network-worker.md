# network-worker Feedback Log

## Active Rules
<!-- 최대 10개. /report Step 4에서 관리. -->
- [2026-04-10] AMP Discriminator에 새 파라미터(disc_reward_type, norm_clip 등) 추가 시 `get_logits()`과 `compute_amp_reward()` 양쪽에 동일하게 적용되는지 확인할 것. 특히 normalizer clip은 `_normalize()` 같은 공통 메서드로 분리하여 두 경로 모두 통과하도록 해야 함. (위반 시: 학습 시 clip 적용되지만 reward 계산 시 미적용 등 불일치)
- [2026-04-10] MimicKit 방식의 logit regularization은 출력값이 아닌 출력 레이어 가중치를 타겟으로 함. `get_output_layer_weights()`처럼 `trunk.modules()`를 역순 순회하여 마지막 `nn.Linear` 가중치를 반환하는 메서드를 discriminator에 노출해야 loss-worker가 사용 가능. (위반 시: logit reg이 의도와 다른 대상에 적용됨)
- [2026-04-08] ActorCriticRMA → ActorCritic 전환 시 해당 러너(OnPolicyRunnerAMP 등)도 반드시 함께 검토할 것. 러너는 내부적으로 정책 클래스를 하드코딩하고 hist_encoding/update_dagger를 직접 호출하므로, 정책 클래스만 바꾸면 런타임 오류가 발생한다. (위반 시: AttributeError 또는 TypeError)

## Deprecated
<!-- agent 무시 — 감사 추적 전용 -->

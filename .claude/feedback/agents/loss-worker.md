# loss-worker Feedback Log

## Active Rules
<!-- 최대 10개. /report Step 4에서 관리. -->
- [2026-04-07] Replay Buffer가 코드에 존재해도 실제 호출 경로에서 올바르게 사용되는지 별도 확인할 것. add/sample 함수가 정의되어 있어도 caller에서 replace 방식으로 잘못 사용될 수 있음. MimicKit 방식: 현재 N개 + 과거 N개를 torch.cat으로 합쳐 2N개로 Discriminator 학습. (위반 시: Catastrophic Forgetting으로 모션 모방 실패)
- [2026-04-06] ppo_parkour.py LCP GP 최종 구현: (1) `obs_batch_gp = obs_batch.clone().apply(requires_grad_(True))`로 TensorDict leaf tensor에 grad 활성화, (2) `act(obs_batch_gp)`로 메인 forward pass, (3) `policy_tensors = [obs_batch_gp[g] for g in policy.obs_groups["policy"]]`로 leaf tensor 직접 추출, (4) `autograd.grad(log_prob, policy_tensors)`로 미분. `get_actor_obs()`를 gradient target으로 사용하면 안 됨 — 내부 `torch.cat`이 새 tensor 인스턴스 생성해 graph 끊김. (위반 시: RuntimeError "not used in graph")

## Deprecated
<!-- agent 무시 — 감사 추적 전용 -->
- [2026-04-03] LCP Gradient Penalty 계산 시 actor 입력 tensor에 `requires_grad_(True)` 설정 후 `_update_distribution()` → `get_actions_log_prob()` → `autograd.grad()` 순서로 호출할 것. (폐기 사유: ppo_parkour는 단일 forward pass + leaf tensor 직접 미분 방식으로 대체됨)
- [2026-04-03] priv_latent는 `torch.no_grad()` 블록 내에서 detach하여 LCP gradient가 priv encoder를 통해 역전파되지 않도록 할 것. (폐기 사유: GP를 obs_batch_gp 전체에 적용하는 방식으로 변경되어 priv_latent도 포함됨)
- [2026-04-06] LCP gradient penalty의 leaf variable은 반드시 정규화 이전 raw obs(`get_actor_obs().detach().requires_grad_(True)`)여야 한다. (폐기 사유: obs_batch_gp TensorDict leaf tensor 방식으로 대체, chain rule은 자동 보장됨)
- [2026-04-06] `_update_distribution()` 직접 호출 시 scandot_encoder가 있으면 scandot_latent도 포함해야 차원이 맞는다. (폐기 사유: 별도 `_update_distribution()` 호출 자체가 제거됨)

# rl-debug-worker Feedback Log

## Active Rules
<!-- 최대 10개. session-summarizer에서 관리. agent는 Deprecated 섹션 무시 -->
- [2026-04-06] LCP gradient penalty 버그 진단 시 `requires_grad_(True)` 위치를 반드시 확인할 것. leaf variable이 `[o_norm, z]`이면 z(priv_latent)가 독립 변수로 취급되어 actor가 priv_latent를 무시하도록 학습되고 task reward가 저하된다. leaf는 반드시 정규화 이전 raw obs여야 함. (위반 시: z를 독립 변수로 취급하는 오진단)
- [2026-04-06] gradient penalty 발산 + entropy 동시 증가 패턴은 policy가 σ를 키워 penalty를 회피하는 신호다. L_GP ∝ 1/σ² 이므로 σ 증가로 쉽게 penalty를 낮출 수 있고, 이때 entropy도 함께 증가한다. 이 패턴이 관측되면 penalty가 잘못 계산(과대 산출)되고 있음을 의심할 것. (위반 시: entropy 증가를 별개 문제로 오진단)

## Deprecated
<!-- 감사 추적 전용 -->

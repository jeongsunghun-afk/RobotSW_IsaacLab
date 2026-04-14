# reward-worker Feedback Log

## Active Rules
<!-- 최대 10개. /report Step 4에서 관리. -->
- [2026-04-09] AMP 환경에서 velocity tracking reward는 MSE 대신 `exp(-scale‖tar_vel - actual_vel‖²)` 형태를 사용할 것. MSE는 목표 속도=0일 때 정지로 수렴하여 보행 패턴이 없어도 최대 reward 달성 가능. (위반 시: AMP discriminator 학습 무의미해짐)

## Deprecated
<!-- agent 무시 — 감사 추적 전용 -->

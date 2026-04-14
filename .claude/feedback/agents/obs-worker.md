# obs-worker Feedback Log

## Active Rules
<!-- 최대 10개. /report Step 4에서 관리. -->
- [2026-04-09] Steering task obs에서 tar_dir/face_dir는 반드시 heading-relative frame으로 변환할 것. `calc_heading_quat_inv(root_quat)`로 yaw만 제거한 quaternion으로 quat_apply. roll/pitch가 포함된 full quat_inv를 사용하면 경사나 요동 시 obs가 불안정해진다. (위반 시: obs 일반화 실패, 기울어진 지형에서 행동 이상)

## Deprecated
<!-- agent 무시 — 감사 추적 전용 -->

# il-debug-worker Feedback Log

## Active Rules
<!-- 최대 10개. session-summarizer에서 관리. agent는 Deprecated 섹션 무시 -->
- [2026-04-03] `projected_gravity_b`에서 roll을 계산할 때 `atan2(gy, gz)` 가 아닌 `atan2(gy, -gz)` 를 사용할 것. 정립 시 gz=-1이므로 gz 부호 반전 없이 계산하면 roll=±π가 출력되어 termination 조건이 항상 만족되는 버그 발생. ZYX Euler 기준: gravity_b = [-sin(p), cos(p)sin(r), -cos(p)cos(r)] → roll = atan2(gy, -gz). (위반 시: 로봇이 정립 상태에서도 즉시 termination)
- [2026-04-03] AMP 환경에서 pose_termination(frame-by-frame reference 비교 종료)은 기본 비활성화 권장. AMP는 distribution matching 방식이므로 특정 시점 reference와의 차이로 종료하는 것은 철학적으로 부적합. 코드는 구현하되 config에서 False로 기본값 설정. (위반 시: AMP 학습 불안정, 과도한 early termination)

## Deprecated
<!-- 감사 추적 전용 -->

# cfg-worker Feedback Log

## Active Rules
<!-- 최대 10개. /report Step 4에서 관리. -->
- [2026-04-08] 새 환경 cfg 추가 시 기존 cfg를 상속 + 달라지는 필드만 오버라이드하는 패턴을 우선 사용할 것. 전체 필드를 복사하면 유지보수 시 양쪽 동기화가 깨진다. (위반 시: 설정 불일치로 인한 디버깅 비용 증가)

## Deprecated
<!-- agent 무시 — 감사 추적 전용 -->

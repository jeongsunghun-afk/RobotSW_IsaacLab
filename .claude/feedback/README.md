# Feedback System 운영 방침

## 목적
사용자와의 대화에서 발생한 피드백/결론을 다음 세션 agent에 전달하여 동일한 실수 반복 방지.

## 구조
```
.claude/feedback/
├── README.md           ← 이 파일
├── global_lessons.md   ← 교차-agent 공통 규칙
├── agents/             ← agent별 누적 피드백 (최대 10개 Active Rules)
│   ├── reward-worker.md
│   ├── obs-worker.md
│   └── ...
└── sessions/           ← 세션별 원본 요약 (감사 추적, agent는 읽지 않음)
    └── YYYY-MM-DD.md
```

## 업데이트 시점
`/report` 명령 실행 시 Step 4에서 자동 업데이트 (별도 명령 불필요).

## Pruning 규칙 (report Step 4에서 적용)
- Active Rules는 **최대 10개** 유지
- 새 규칙이 기존 규칙을 포함/대체 → 기존 규칙을 Deprecated로 이동
- 10개 초과 시 → 가장 오래된 non-critical 규칙을 Deprecated로
- Deprecated 섹션은 agent가 읽지 않음 (감사 추적 전용)

## Agent별 피드백 파일 포맷
```markdown
# {agent-name} Feedback Log

## Active Rules
<!-- 최대 10개. session이 관리. -->
- [YYYY-MM-DD] 규칙 (위반 시 결과)

## Deprecated
<!-- agent 무시 — 감사 추적 전용 -->
- [날짜] ~~구 규칙~~ → 대체 이유
```

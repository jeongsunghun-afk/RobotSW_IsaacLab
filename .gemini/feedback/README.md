# Gemini Feedback System 운영 방침

## 목적
Gemini 에이전트들의 코드 분석 및 디버깅 과정에서 얻은 교훈, 실수 패턴, 주의사항을 기록하고 다음 세션에서 반영하여 오류의 반복을 방지한다. Claude의 피드백 시스템과 유사하게 동작한다.

## 구조
```
.gemini/feedback/
├── README.md           ← 이 파일
├── global_lessons.md   ← 모든 Gemini 에이전트 공통 적용 규칙
├── agents/             ← 각 에이전트별 누적 피드백 (최대 10개 Active Rules)
│   └── debug-worker.md
└── sessions/           ← 세션별 요약 (감사 추적 전용)
```

## 업데이트 규칙 (debug-analyzer 스킬에서 수행)
- **Active Rules**는 최대 10개 유지. 초과 시 오래된 것은 Deprecated로 이동.
- 포맷: `- [YYYY-MM-DD] 규칙 설명 (위반 시 결과)`
- 에이전트는 작업 시작 전 반드시 해당 피드백을 먼저 읽어야 한다.

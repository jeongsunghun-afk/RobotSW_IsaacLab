# .claude/feedback/

이 디렉토리는 **Notion 저장 실패 시 로컬 fallback 저장소**로만 사용됩니다.

## 구조

```
feedback/
└── sessions/
    └── YYYY-MM-DD-{slug}.md   # /report 가 Notion 저장 실패 시 여기에 저장
```

## 정책 (2026-04-30 개편)

- **폐기**: `global_lessons.md` 및 agent별 누적 규칙(`agents/`)은 더 이상 사용하지 않습니다. 보고 시스템은 글로벌 `~/.claude/agents/session-report-writer.md` agent와 `~/.claude/commands/report.md` 명령어로 통합되었습니다.
- **유지**: `sessions/` — `/report` 실행 중 Notion API 호출이 실패하면 보고서가 이곳에 자동 저장됩니다.
- **수동 정리**: 더 이상 필요 없는 fallback 파일은 자유롭게 삭제 가능합니다.

## 관련 위치

- 보고서 작성 명령: `~/.claude/commands/report.md`
- 보고서 작성 agent: `~/.claude/agents/session-report-writer.md`
- Notion 저장 대상: `~/.claude/notion_targets.yaml`

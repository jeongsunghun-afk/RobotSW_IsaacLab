이 세션에서 수행한 작업을 마크다운 보고서로 정리하고 Notion에 저장합니다.

## 사용법
`/report {프로젝트명}` — 예: `/report Go2`, `/report R_Skeleton`, `/report Retargeting`

인수 없이 `/report` 만 호출한 경우 → 현재 작업 맥락에서 프로젝트를 추론하거나, 사용자에게 물어보세요.

---

## 실행 절차

### Step 1: 대상 Notion 페이지 결정

`.claude/notion_targets.yaml` 파일을 Read 도구로 읽어 프로젝트명에 해당하는 `page_id`를 확인합니다.

- 인수로 받은 프로젝트명과 aliases 중 하나가 일치하면 해당 page_id 사용
- 매칭 안 되는 경우: 사용자에게 "어느 Notion 페이지에 저장할까요? (Go2 / R_Skeleton / Retargeting)" 질문

### Step 2: 보고서 내용 생성

이 대화 세션 전체를 검토하여 아래 구조로 보고서를 작성합니다.
모든 섹션은 실제 내용이 있는 경우에만 포함하고, 없으면 생략합니다.

```
# [YYYY-MM-DD] {작업 제목 — 핵심 변경사항 1줄 요약}

## 작업 요약
- **목적**: 이번 세션의 목표
- **결과**: 성공 / 부분 성공 / 진행 중

## 잘 된 것
- 성공한 구현, 검증된 아이디어, 해결된 버그 목록
- 각 항목에 구체적인 결과 포함 (예: "reward 수렴 확인", "shape 오류 해결")

## 잘 안 된 것 (시도했지만 문제 있음)
| 시도한 것 | 발생한 문제 | 원인 추정 / 상태 |
|-----------|-------------|-----------------|
| 예: contact_force reward 추가 | reward explosion | weight 스케일 문제, 다음 세션에 재시도 |

## 변경된 파일
| 파일 | 주요 변경 내용 |
|------|---------------|
| `경로/파일명.py` | 변경 내용 요약 |

## 분석 / 인사이트
- 이번 세션에서 발견한 중요한 사실, 주의사항, 설계 결정 이유

## 다음 세션 TODO
- [ ] 미완성 작업
- [ ] 재시도 필요한 것
- [ ] 확인 필요한 것
```

### Step 3: Notion에 저장

`mcp__claude_ai_Notion__notion-create-pages` 도구를 호출하여 보고서를 저장합니다.

- **parent**: Step 1에서 확인한 page_id
- **title**: 보고서 제목 (예: `[2026-03-30] Go2 AMP 보상 함수 수정`)
- **content**: Step 2에서 작성한 보고서 전체 (마크다운 형식)

저장 완료 후 생성된 페이지 URL을 사용자에게 알려줍니다.

### Step 4: Agent 피드백 업데이트

Step 2에서 생성한 보고서를 그대로 활용하여 (대화 재읽기 없이) agent 피드백 파일을 업데이트합니다.

**4-1. 영향받는 agent 분류**

보고서의 "잘 된 것 / 잘 안 된 것 / 분석·인사이트" 섹션을 보고, 아래 agent 중 관련된 것을 파악합니다:
- ENV: reward-worker, obs-worker, cfg-worker
- ALGO: network-worker, loss-worker, hyperparam-worker
- 검증: validate-code, validate-method
- 디버그: wtw-debug-worker, amp-debug-worker
- 분석: log-analyzer
- 교차: global_lessons (모든 agent에 적용되는 규칙)

**4-2. 각 피드백 파일 업데이트**

영향받는 각 agent의 `.claude/feedback/agents/{agent-name}.md`를 Read 후 Edit합니다.

규칙 추가 시 Pruning 적용:
- 새 규칙이 기존 규칙을 포함/대체 → 기존 규칙을 Deprecated 섹션으로 이동
- Active Rules가 10개 초과 → 가장 오래된 non-critical 규칙을 Deprecated로 이동
- 이미 동일한 규칙이 있으면 추가하지 않음

추가할 규칙 포맷: `- [YYYY-MM-DD] 규칙 설명 (위반 시 결과)`

**4-3. global_lessons.md 업데이트**

`.claude/feedback/global_lessons.md`를 Read 후, 특정 agent에 국한되지 않는 교차 규칙만 추가합니다. (동일한 pruning 규칙 적용)

**4-4. 세션 요약 저장**

`.claude/feedback/sessions/YYYY-MM-DD.md`를 Write합니다:

```markdown
# Session: YYYY-MM-DD — {보고서 제목}

## 업데이트된 Agent
- {agent-name}: {추가된 규칙 요약}

## context.md 업데이트 필요 여부
- [ ] {환경명}_context.md: {사유} (필요한 경우만)
```

피드백 업데이트 완료 후 사용자에게 업데이트된 agent 목록을 간략히 알립니다.

---

## 주의사항
- 보고서는 사실에 기반해 작성합니다. 시도하지 않은 내용은 포함하지 않습니다.
- "잘 안 된 것"은 부정적인 것이 아니라 다음 세션의 단서입니다. 구체적으로 기록합니다.
- 파일 경로는 프로젝트 루트 기준 상대경로로 작성합니다.

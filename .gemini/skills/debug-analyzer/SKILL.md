---
name: debug-analyzer
description: 분석/디버그 요청을 받아 문제를 진단하고, Claude에게 전달할 보고서를 작성한 뒤 피드백 루프를 업데이트하는 스킬입니다. "분석해줘", "디버그해줘", "학습 결과 디버깅" 요청 시 실행됩니다.
user-invocable: true
---

## 실행 모드
Claude가 디버깅 또는 분석을 요청할 때 이 스킬을 호출한다.
분석 실행 → 리포트 작성 → 피드백 업데이트를 자동화한다.

## 사전 준비
1. `_workspace/` 디렉토리가 존재하는지 확인 (없으면 생성)
2. 대상 환경에 맞는 피드백 로그 (`.gemini/feedback/agents/rl-debug-worker.md` 또는 `il-debug-worker.md`) 및 `.gemini/feedback/global_lessons.md` 읽기.

## 워크플로우

### Phase 1: 현황 파악 및 분석 위임
사용자/Claude가 요청한 증상, 대상 환경(go2_amp, go2_wtw, R_Skeleton 등), 관련된 파일 및 로그 경로를 수집한다.
대상 환경명에 따라 적절한 에이전트를 판단하여 Agent 도구를 통해 분석 작업을 지시한다.
- 환경명에 `amp`, `il`, `imitation`이 포함된 경우: `@il-debug-worker` 호출
- 일반 강화학습 환경인 경우: `@rl-debug-worker` 호출

```
[프롬프트 내용]
- 환경: {환경명}
- 문제 증상: {증상 설명}
- 대상 파일/로그: {파일 또는 로그 경로}
```

### Phase 2: 분석 리포트 작성
호출된 에이전트의 분석 결과를 기반으로 `_workspace/debug_report.md` 파일을 생성한다.
이 문서는 Claude가 코드를 수정하는 데 직접적인 지침이 된다.

### Phase 3: 피드백 루프 업데이트
분석 과정에서 새롭게 얻은 교훈, 자주 발생하는 휴먼/AI 에러 패턴, 주의사항을 추출한다.
- **Agent별 피드백**: `.gemini/feedback/agents/rl-debug-worker.md` 또는 `il-debug-worker.md`
- **공통/교차 피드백**: `.gemini/feedback/global_lessons.md`

#### 피드백 업데이트 규칙
1. 기존 Active Rules가 10개를 초과하면 가장 오래된 규칙을 Deprecated 섹션으로 이동한다.
2. 새 규칙 포맷: `- [YYYY-MM-DD] 규칙 설명 (위반 시 결과)`
3. 파일 수정에는 파일 쓰기/치환 도구를 사용한다.

### Phase 4: 완료 알림
Claude에게 다음 포맷으로 완료 메시지를 반환한다.
```
디버깅 및 분석이 완료되었습니다.
- 결과 보고서: _workspace/debug_report.md
- 피드백 업데이트 내역: [추가된 피드백 요약]
```

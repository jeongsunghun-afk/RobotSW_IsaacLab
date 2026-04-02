---
name: research-team
description: 사족보행 로봇 보행 학습에 필요한 논문/기법을 조사하는 리서치 팀을 구성한다. research-surveyor(Gemini CLI 조사)와 research-advisor(교수 역할 검토)가 토론하여 최종 구현 제안안을 생성한다. "리서치 팀 돌려줘", "논문 조사", "최신 방법 조사", "어떤 기법이 좋을까", "research team" 요청 시 반드시 이 스킬을 사용할 것. /research와 달리 두 에이전트가 팀으로 협업하여 더 깊이 있는 검토를 수행한다.
user-invocable: true
---

## 실행 모드
에이전트 팀 모드 — surveyor + advisor가 SendMessage로 직접 토론, 최대 2라운드.

## 사전 준비

다음 파일들을 읽어 프로젝트 현재 상태 파악:
- `CLAUDE.md` — 프로젝트 미션 및 현재 구조
- `_workspace/` 디렉토리가 없으면 생성

## 워크플로우

### Phase 1: 팀 구성

TeamCreate로 리서치 팀을 구성한다:
```
팀명: research-team
팀원: research-surveyor, research-advisor
```

### Phase 2: 작업 할당

TaskCreate로 작업을 할당한다:
```
Task 1 (research-surveyor): 연구 질문 조사
  - 입력: 사용자 연구 질문 + 프로젝트 컨텍스트
  - 출력: _workspace/01_surveyor_research.md
  - 완료 조건: research-advisor에게 SendMessage 전송

Task 2 (research-advisor): 조사 결과 검토 및 최종 승인
  - 의존: Task 1 완료 후
  - 입력: _workspace/01_surveyor_research.md
  - 출력: _workspace/02_advisor_final_proposal.md
  - 완료 조건: 오케스트레이터에게 "완료" SendMessage
```

### Phase 3: 토론 모니터링

팀원들이 자체 조율하는 동안 대기 (최대 2라운드 허용):
- Round 1: surveyor 조사 → advisor 검토 → 추가 조사 요청 가능
- Round 2: surveyor 보완 → advisor 최종 승인
- "완료" 메시지 수신 시 Phase 4로 진행

### Phase 4: 결과 정리

`_workspace/02_advisor_final_proposal.md`를 읽어 사용자에게 선택지 제시:

```
## 리서치 팀 결과

| 우선순위 | 방법 | 적용 가능성 | 예상 효과 | 구현 범위 |
|--------|------|------------|---------|---------|

어떤 방법으로 진행하시겠습니까?
1. 위 방법 중 선택하여 /locomotion-loop 실행
2. 추가 조사 요청
3. 직접 방법을 지정하여 /locomotion-loop 실행
```

### Phase 5: 팀 정리

팀원들에게 작업 완료 알림 후 팀 해체.

## 에러 핸들링

- **Gemini CLI 타임아웃**: surveyor가 `_workspace/research_brief.md` 저장 후 advisor에게 알림 → WebSearch만으로 진행
- **합의 실패 (2라운드 후)**: advisor가 가용 정보로 최선의 안 확정, "불확실성 있음" 명시
- **파일 저장 실패**: 메시지로 내용 전달 후 계속 진행

## 출력 파일
- `_workspace/01_surveyor_research.md` — 조사 원문
- `_workspace/02_advisor_final_proposal.md` — 최종 제안안

## 테스트 시나리오

**정상 흐름**: "발 충격을 줄이는 최신 보상 함수 기법 조사해줘"
→ surveyor가 Gemini로 조사 → advisor가 코드베이스 기반 검토 → 최종 3가지 방법 제안

**에러 흐름**: Gemini CLI 오류
→ surveyor가 WebSearch로만 조사 → advisor가 검토 → 불확실성 명시된 제안안 생성

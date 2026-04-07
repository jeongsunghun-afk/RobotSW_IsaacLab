---
name: research-team
description: 사족보행 로봇 보행 학습에 필요한 논문/기법을 조사하는 리서치 팀을 구성한다. Gemini CLI를 통해 research-surveyor(조사)와 research-advisor(검증)가 협업하며, Claude는 결과 파일을 읽어 최종 제안을 사용자에게 전달한다. "리서치 팀 돌려줘", "논문 조사", "최신 방법 조사", "어떤 기법이 좋을까", "research team" 요청 시 반드시 이 스킬을 사용할 것.
user-invocable: true
---

## 실행 모드

Gemini CLI 위임 모드 — Claude는 `gemini -p` 로 단일 호출을 보내고,
Gemini 내부에서 GEMINI.md 라우팅 규칙에 따라 `@research-surveyor` → `@research-advisor` 협업이 자동 진행된다.
Claude는 결과 파일만 읽어 사용자에게 전달한다.

## 아키텍처

```
Claude (오케스트레이터)
  │
  └─ Bash: gemini -p "..." -y
              │
              ├─ @research-surveyor  (웹 검색 + 논문 조사 + 적용 아이디어)
              │        │ _workspace/01_surveyor_research.md 저장
              │        ▼
              └─ @research-advisor   (코드베이스 기반 비판적 검증)
                       │ _workspace/02_advisor_final_proposal.md 저장
                       ▼
  Claude: Read 결과 파일 → 사용자에게 요약 전달
```

## 사전 준비

- CLAUDE.md를 읽어 프로젝트 미션·구조 파악
- `_workspace/` 디렉토리 존재 확인 (없으면 `mkdir -p .claude/skills/research-team/_workspace` 실행)

## 워크플로우

### Phase 1: 연구 질문 정제

사용자 요청에서 핵심 연구 질문을 추출하고, 아래 형식으로 Gemini 프롬프트를 구성한다:

```
[연구 질문]
{사용자 질문을 한 문장으로 요약}

[프로젝트 컨텍스트]
- 목표: Go2 / R_Skeleton 사족보행 자연스럽고 강건한 보행 (RL + AMP)
- 프레임워크: IsaacLab + RSL-RL
- 현재 상태: {CLAUDE.md에서 파악한 현재 이슈 또는 개선 목표}

[요청]
1. @research-surveyor 에게 위 연구 질문에 대한 최신 논문/기법 조사 및 IsaacLab 적용 아이디어 도출을 요청하고,
   결과를 .claude/skills/research-team/_workspace/01_surveyor_research.md 에 저장해줘.
2. @research-advisor 에게 surveyor 결과를 코드베이스 기반으로 비판적 검증하게 하고,
   최종 합의된 제안안을 .claude/skills/research-team/_workspace/02_advisor_final_proposal.md 에 저장해줘.
3. 저장 완료 후 "RESEARCH_COMPLETE" 를 출력해줘.
```

### Phase 2: Gemini CLI 실행

```bash
cd /home/lgb/IsaacLab && gemini -p "{Phase 1에서 구성한 프롬프트}" -y
```

**주의사항:**
- 반드시 `--yolo`(`-y`) 플래그 사용 (파일 저장 확인 프롬프트 자동 승인)
- 타임아웃은 최대 10분 (복잡한 리서치는 시간이 걸릴 수 있음)
- 출력에서 "RESEARCH_COMPLETE" 확인 시 다음 단계 진행

### Phase 3: 결과 파일 읽기

```
Read: .claude/skills/research-team/_workspace/02_advisor_final_proposal.md
```

파일이 없거나 비어 있으면:
1. `_workspace/01_surveyor_research.md` 먼저 확인
2. 있으면 해당 내용으로 진행 (advisor 단계 미완료 명시)
3. 둘 다 없으면 에러 처리 → Phase 2 재시도 or 사용자에게 알림

### Phase 4: 사용자에게 결과 전달

최종 제안안을 아래 형식으로 요약 제시:

```markdown
## 리서치 팀 결과

| 우선순위 | 방법 | 적용 가능성 | 예상 효과 | 구현 범위 |
|--------|------|------------|---------|---------|
| 1 | ... | 상/중/하 | ... | ENV / ALGO |

**Advisor 검증 의견:** {advisor의 핵심 코멘트 1~2줄}

---
어떤 방법으로 진행하시겠습니까?
1. 위 방법 중 선택하여 `/locomotion-loop` 실행
2. 추가 조사 요청 (더 깊이 파고들 주제 지정)
3. 직접 방법을 지정하여 `/locomotion-loop` 실행
```

## 에러 핸들링

| 상황 | 대응 |
|------|------|
| `gemini` 명령어 실패 (exit code != 0) | 에러 메시지 확인 후 사용자에게 보고 |
| 결과 파일 미생성 | Gemini stdout에서 내용 추출 시도 후 임시 파일로 저장 |
| 타임아웃 (10분 초과) | 중간 파일(`01_surveyor_research.md`)이라도 읽어 진행 |
| "@research-surveyor 에이전트 없음" 오류 | `.gemini/agents/` 경로 확인 및 `settings.json`의 `enableAgents: true` 점검 안내 |

## 출력 파일 위치

- `.claude/skills/research-team/_workspace/01_surveyor_research.md` — Surveyor 조사 원문
- `.claude/skills/research-team/_workspace/02_advisor_final_proposal.md` — Advisor 최종 제안안

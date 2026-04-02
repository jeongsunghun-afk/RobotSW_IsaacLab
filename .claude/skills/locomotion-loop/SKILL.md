---
name: locomotion-loop
description: 사족보행 보행 학습의 전체 반복 사이클을 오케스트레이션한다. 제안 방안(리서치팀 결과 또는 사용자 직접 지정)을 받아 환경 설계→알고리즘 설계→검증→[학습]→평가→피드백→반복 루프를 수행한다. "학습 루프 시작", "보행 학습 시작", "이터레이션 돌려줘", "locomotion loop", "ENV+ALGO 전체 파이프라인" 요청 시 반드시 이 스킬을 사용할 것.
user-invocable: true
---

## 실행 모드
파이프라인 + 반복 루프 — 각 단계를 순차 실행, 학습 단계에서 사용자 개입, 평가 후 반복 여부 결정.

## 사전 준비

다음을 확인한다:
- `_workspace/` 디렉토리 생성 (없으면)
- 이터레이션 번호 초기화 (N=1) 또는 기존 이터레이션 번호 이어받기
- 제안 방안 파악:
  - `/research-team` 결과 있으면: `_workspace/02_advisor_final_proposal.md` 읽기
  - 없으면: 사용자에게 변경 사항 직접 요청

## 메인 루프 (이터레이션 N)

---

### Stage 1: 변경 사항 파악

이번 이터레이션에서 수행할 변경 사항 확인:
- **첫 이터레이션**: 제안 방안에서 1순위 방법 선택
- **2회차 이상**: `_workspace/feedback_iter_{N-1}.md` 읽기

변경 범위 분류:
- ENV만 → Stage 2-A만
- ALGO만 → Stage 2-B만  
- 둘 다 → Stage 2-A, 2-B 병렬 실행

---

### Stage 2-A: 환경 설계 (ENV 변경 있을 때)

Agent 도구로 general-purpose 에이전트를 spawn하여 ENV 작업 수행:

```
프롬프트 내용:
- CLAUDE.md의 /env-coordinator 절차를 따라 다음 변경을 수행:
  [변경 사항 목록]
- 환경명: [go2 / go2_amp / R_Skeleton / R_Skeleton_amp]
- 완료 후 변경된 파일 목록 반환
```

### Stage 2-B: 알고리즘 설계 (ALGO 변경 있을 때)

Agent 도구로 general-purpose 에이전트를 spawn하여 ALGO 작업 수행:

```
프롬프트 내용:
- CLAUDE.md의 /algo-coordinator 절차를 따라 다음 변경을 수행:
  [변경 사항 목록]
- 완료 후 변경된 파일 목록 반환
```

---

### Stage 3: 코드 정합성 검증

Agent 도구로 validate-code 에이전트 실행:

```
프롬프트: Stage 2에서 변경된 파일들에 대해 체크리스트 A~D를 검증하라.
변경 파일: [Stage 2 결과]
```

Red 판정 항목 발견 시: 해당 worker에게 수정 요청 후 재검증.

---

### Stage 4: 학습 실행 (사용자 개입)

사용자에게 학습 명령어 제시:

```bash
# 학습 실행 (권장 설정)
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task [TASK_NAME] --num_envs 4096 \
  --logger wandb --wandb-project IsaacLab-locomotion

# 완료 후 로그 경로 확인
ls logs/rsl_rl/[task_name]/
```

학습 완료 후 로그 경로를 알려달라고 안내.

---

### Stage 5: 평가

Agent 도구로 training-evaluator 에이전트 실행:

```
프롬프트:
- 로그 디렉토리: [사용자가 제공한 경로]
- 이터레이션 번호: N
- 이전 이터레이션 결과: _workspace/eval_iter_{N-1}.md (있으면)
- 평가 보고서를 _workspace/eval_iter_{N}.md에 저장
```

---

### Stage 6: 피드백 생성

Agent 도구로 feedback-generator 에이전트 실행:

```
프롬프트:
- 평가 보고서: _workspace/eval_iter_{N}.md
- 이전 피드백 이력: _workspace/feedback_iter_*.md (있으면)
- 원본 제안: _workspace/02_advisor_final_proposal.md (있으면)
- 이터레이션 번호: N
- 피드백을 _workspace/feedback_iter_{N}.md에 저장
```

---

### Stage 7: 루프 결정

사용자에게 현황과 선택지 제시:

```
## 이터레이션 N 완료

종합 판정: [Green/Yellow/Red]
주요 지표: [요약]

다음 행동:
1. 이터레이션 N+1 시작 (권장 변경: [핵심 1~2개])
2. 루프 종료 (현재 결과로 배포)
3. 리서치 팀 다시 호출하여 방향 재검토
```

**Green 판정**: "목표 달성 — 루프 종료를 권장합니다" 제안  
**3회 연속 Yellow**: "점진적 개선 중 — 계속 진행?" 제안  
**Red 판정 3회 반복**: "근본적 방향 변경 필요 — /research-team 재호출 권장" 제안

사용자 선택에 따라 N+1로 이어가거나 종료.

---

## 에러 핸들링

- **validate-code Red**: Stage 3에서 블로킹 — 수정 완료 후 Stage 4 진행
- **학습 실행 실패**: Stage 4 건너뛰고 이전 로그로 Stage 5 진행 (있을 경우)
- **로그 없음**: training-evaluator가 "데이터 없음" 보고 → 사용자에게 로그 경로 재확인 요청
- **Stage 2 병렬 충돌**: ENV/ALGO가 동일 파일 수정 시 순차 실행으로 전환

## 작업 산출물

각 이터레이션마다 생성:
- `_workspace/env_changes_iter_{N}.md` — ENV 변경 내역
- `_workspace/algo_changes_iter_{N}.md` — ALGO 변경 내역
- `_workspace/eval_iter_{N}.md` — 평가 보고서
- `_workspace/feedback_iter_{N}.md` — 다음 이터레이션 피드백

## 테스트 시나리오

**정상 흐름 (첫 이터레이션)**:
"발 충격 보상 추가 방안으로 locomotion-loop 시작"
→ ENV만 변경 → validate-code → 사용자 학습 → 평가 Yellow → 피드백 생성 → N+1 제안

**에러 흐름 (Red 반복)**:
3회 연속 Red → "리서치 팀 재호출 권장" 메시지와 함께 루프 일시 중단

---
name: locomotion-loop
description: 학습 전체 반복 사이클(환경 설계→알고리즘 설계→검증→[학습]→평가→피드백→반복)을 오케스트레이션. 보행뿐 아니라 모든 robot/task에 적용 가능. "학습 루프 시작", "이터레이션 돌려줘", "ENV+ALGO 전체 파이프라인" 요청 시 본 스킬 사용.
user-invocable: true
---

## 실행 모드
파이프라인 + 반복 루프 — 각 단계 순차 실행, 학습 단계에서 사용자 개입, 평가 후 반복 여부 결정.

## 사전 준비
- `_workspace/` 디렉토리 생성 (없으면)
- 이터레이션 번호 초기화 (N=1) 또는 기존 이터레이션 이어받기
- 변경 제안 파악:
  - 사용자가 직접 지정한 변경 사항 또는
  - 이전 이터레이션의 `_workspace/feedback_iter_{N-1}.md` 사용

## 메인 루프 (이터레이션 N)

---

### Stage 1: 변경 사항 파악

이번 이터레이션에서 수행할 변경 사항 결정
- **첫 이터레이션**: 사용자 입력 또는 명시적 제안에서 1순위 변경 추출
- **2회차 이상**: 직전 `_workspace/feedback_iter_{N-1}.md` 읽기

변경 범위 분류
- ENV만 → Stage 2-A
- ALGO만 → Stage 2-B
- 둘 다 → Stage 2-A, 2-B 병렬

---

### Stage 2-A: 환경 설계 (ENV 변경 있을 때)

`/env-coordinator` 절차로 적절한 worker dispatch:

```
Agent(
  subagent_type: "general-purpose",
  description: "ENV 작업 오케스트레이션",
  prompt: """
  /env-coordinator 명령에 따라 다음 변경 수행:
  - task: <task_name>
  - 변경 사항: <list>
  - 완료 후 변경된 파일 목록 반환
  """
)
```

### Stage 2-B: 알고리즘 설계 (ALGO 변경 있을 때)

```
Agent(
  subagent_type: "general-purpose",
  description: "ALGO 작업 오케스트레이션",
  prompt: """
  /algo-coordinator 명령에 따라 다음 변경 수행:
  - 변경 사항: <list>
  - 완료 후 변경된 파일 목록 반환
  """
)
```

---

### Stage 3: 코드 정합성 검증

```
Agent(
  subagent_type: "validate-code",
  description: "정합성 검증",
  prompt: "Stage 2 변경 파일들에 대해 체크리스트 A~D 검증."
)
```

FAIL 항목이 있으면 해당 worker에게 수정 요청 후 재검증.

---

### Stage 4: 학습 실행 (사용자 개입)

사용자에게 학습 명령어 안내:

```bash
# rsl_rl 예시
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task <Task-Name> --num_envs 4096 \
  --logger wandb --wandb-project IsaacLab-<your-project>

# skrl 예시
./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
  --task <Task-Name> --num_envs 4096 --algorithm <PPO|AMP|...>

# 완료 후 로그 경로 확인
ls logs/<framework>/<task>/
```

학습 완료 후 사용자에게 로그 경로 요청.

---

### Stage 5: 평가

```
Agent(
  subagent_type: "training-evaluator",
  description: "학습 결과 평가",
  prompt: """
  로그 디렉토리: <user_provided_path>
  task 유형: <locomotion|manipulation|...>
  이터레이션 번호: N
  이전 결과 (있으면): _workspace/eval_iter_{N-1}.md
  보고서 저장 위치: _workspace/eval_iter_{N}.md
  """
)
```

---

### Stage 6: 피드백 생성

```
Agent(
  subagent_type: "feedback-generator",
  description: "평가 → 다음 이터레이션 작업 지시서",
  prompt: """
  평가 보고서: _workspace/eval_iter_{N}.md
  이전 피드백: _workspace/feedback_iter_*.md (있으면)
  이터레이션 번호: N
  저장 위치: _workspace/feedback_iter_{N}.md
  """
)
```

---

### Stage 7: 루프 결정

사용자에게 현황 + 선택지 제시:

```
## 이터레이션 N 완료
종합 판정: <Green/Yellow/Red>
주요 지표: <요약>

다음 행동
1. 이터레이션 N+1 시작 (권장 변경: <핵심 1~2개>)
2. 루프 종료 (현재 결과로 배포)
3. 방향 재검토 (큰 설계 변경 후 재시작)
```

권고 패턴
- **Green**: 종료 권고
- **3회 연속 Yellow**: "충분히 수렴했을 가능성 — 종료 또는 다른 task로 확장 고려"
- **3회 연속 Red**: "근본적 방향 변경 필요 — 사용자 의사결정 요청"

사용자 선택에 따라 N+1로 이어가거나 종료.

---

## 에러 핸들링
- **validate-code FAIL**: Stage 3에서 블로킹 — 수정 후 Stage 4 진행
- **학습 실패/Crash**: `debug-worker` 호출 권고 (로그 경로 + 증상 전달)
- **로그 없음**: training-evaluator가 "데이터 부족" 보고 → 사용자에게 경로 재확인 요청
- **Stage 2 병렬 충돌**: ENV/ALGO가 동일 파일 만지면 순차 전환

## 작업 산출물

각 이터레이션 N마다:
- `_workspace/env_changes_iter_{N}.md` — ENV 변경 내역
- `_workspace/algo_changes_iter_{N}.md` — ALGO 변경 내역
- `_workspace/eval_iter_{N}.md` — 평가 보고서
- `_workspace/feedback_iter_{N}.md` — 다음 이터레이션 작업 지시서

## 적용 task 예시
- **Locomotion**: 보행 명령 추종, gait 다양성, foot contact 안정
- **Manipulation**: pick-and-place, target reaching, grasp success rate
- **Hybrid (IL+RL)**: AMP 기반 자연스러움 + RL 기반 task 추종

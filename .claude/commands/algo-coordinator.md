ALGO 작업을 분석하고 적절한 worker를 Agent tool로 dispatch합니다.

## Step 0: Dispatch 모드 결정

매 dispatch 시점에 환경변수를 확인:

```bash
[[ "${ISAAC_WORKER_PANES:-0}" == "1" ]] && echo PANE_MODE || echo AGENT_MODE
```

- **AGENT_MODE (default)**: 아래 Step 2의 `Agent(subagent_type=...)` 직접 호출 — in-process sub-agent
- **PANE_MODE** (`ISAAC_WORKER_PANES=1`): tmux split pane에서 별도 `claude` CLI 프로세스로 worker 실행 (실시간 작업 화면 관찰 가능)

### PANE_MODE dispatch 패턴

순차 (단일 또는 의존):
```bash
RES=$(scripts/isaac_worker/dispatch.sh <agent-name> "<task-text>")
```

병렬 (독립 작업 N개):
```bash
JOB1=$(scripts/isaac_worker/spawn_pane.sh <agent-A> "<task-A>")
JOB2=$(scripts/isaac_worker/spawn_pane.sh <agent-B> "<task-B>")
scripts/isaac_worker/wait_pane.sh "$JOB1" "$JOB2"
RES1=$(scripts/isaac_worker/get_result.sh "$JOB1" --result)
RES2=$(scripts/isaac_worker/get_result.sh "$JOB2" --result)
```

비용 주의: pane mode는 worker당 별도 claude CLI 세션 — 무거운 worker나 작업 과정 직접 관찰이 필요할 때만 활성화.

## Step 1: 알고리즘 컨텍스트 파악

대상 알고리즘 식별:
- rsl_rl 환경: `rsl_rl/rsl_rl/algorithms/<algo>.py` + `modules/`
- skrl 환경: `skrl/skrl/agents/torch/<algo>/<algo>.py` + agent YAML
- 환경별 agent cfg: `source/isaaclab_tasks/.../<task>/agents/`

알고리즘 패러다임:
- 일반 RL (PPO, SAC 등)
- 모방학습 (BC, GAIL, AMP 등)
- 하이브리드 (RL + IL 보조 loss)

## Step 2: 작업 분류 및 Worker dispatch

### network-worker
**트리거**: actor/critic/discriminator 등 네트워크 아키텍처 변경 (레이어, activation, init)

**Agent 호출:**
```
Agent(
  subagent_type: "network-worker",
  description: "네트워크 아키텍처 변경",
  prompt: """
  변경 대상: {file_path or yaml_path}
  변경 사항: {description}
  현재 input shape: {input_shape}
  현재 output shape: {output_shape}
  """
)
```

### loss-worker
**트리거**: loss 함수 항 추가/제거, gradient 흐름 수정 (detach, optimizer 분리 등)

**Agent 호출:**
```
Agent(
  subagent_type: "loss-worker",
  description: "Loss 함수 수정",
  prompt: """
  알고리즘 파일: {algo_file}
  변경 사항: {description}
  영향 받는 loss 항: {affected_terms}
  """
)
```

### hyperparam-worker
**트리거**: 학습률, clip_param, entropy_coef, num_steps, task vs style 비율 등 스칼라 조정

**Agent 호출:**
```
Agent(
  subagent_type: "hyperparam-worker",
  description: "학습 하이퍼파라미터 조정",
  prompt: """
  cfg 파일: {cfg_file}
  변경 파라미터: {param_list}
  변경 의도: {rationale}
  """
)
```

## Step 3: 상호의존성 주의

- network + loss 동시 변경: **순차 dispatch** (network 먼저 → output shape 결정 → loss에 전달)
- network + hyperparam: 보통 독립이므로 병렬 가능
- 동일 파일 수정 시: 순차

## Step 4: validate-code 자동 실행

모든 worker 완료 후:

```
Agent(
  subagent_type: "validate-code",
  description: "코드 정합성 검증",
  prompt: """
  변경된 파일 목록: {changed_files}
  변경 요약: {changes_summary}

  체크리스트 A~D 중 적용 가능한 항목 실행 + PASS/FAIL 판정.
  """
)
```

## Step 5: validate-method 자동 실행 (설계 변경 시)

```
Agent(
  subagent_type: "validate-method",
  description: "방법론 타당성 검증",
  prompt: """
  설계 변경 요약: {design_changes}
  변경 배경: {rationale}

  체크리스트 E~H 중 알고리즘 패러다임에 해당하는 항목 적용:
    - RL 기본: 체크리스트 E, G, H
    - IL/AMP: 체크리스트 F 추가
  Green/Yellow/Red 판정.
  """
)
```

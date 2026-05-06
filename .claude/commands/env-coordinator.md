ENV 작업을 분석하고 적절한 worker를 Agent tool로 dispatch합니다.

## Step 0: Dispatch 모드 결정

매 dispatch 시점에 환경변수를 확인:

```bash
[[ "${ISAAC_WORKER_PANES:-0}" == "1" ]] && echo PANE_MODE || echo AGENT_MODE
```

- **AGENT_MODE (default)**: 아래 Step 2의 `Agent(subagent_type=...)` 직접 호출 — in-process sub-agent
- **PANE_MODE** (`ISAAC_WORKER_PANES=1`): 아래 패턴으로 대체. tmux split pane에서 별도 `claude` CLI 프로세스로 worker 실행 (실시간 작업 화면 관찰 가능)

### PANE_MODE dispatch 패턴

순차 (단일 또는 의존):
```bash
RES=$(scripts/isaac_worker/dispatch.sh <agent-name> "<task-text>")  # 결과 stdout
```

병렬 (독립 작업 N개):
```bash
JOB1=$(scripts/isaac_worker/spawn_pane.sh <agent-A> "<task-A>")
JOB2=$(scripts/isaac_worker/spawn_pane.sh <agent-B> "<task-B>")
scripts/isaac_worker/wait_pane.sh "$JOB1" "$JOB2"
RES1=$(scripts/isaac_worker/get_result.sh "$JOB1" --result)
RES2=$(scripts/isaac_worker/get_result.sh "$JOB2" --result)
```

비용 주의: pane mode는 worker당 별도 claude CLI 세션 (스폰 ~3-5s + API 풀 비용). 무거운 worker(`debug-worker`, `training-evaluator` 등)나 작업 과정 직접 관찰이 필요할 때만 활성화 권장.

## Step 1: 환경 context 파악

사용자가 지정한 task/환경의 디렉토리를 식별:
- `source/isaaclab_tasks/isaaclab_tasks/direct/<task_name>/`
- 해당 디렉토리에 `CLAUDE.md`가 있으면 먼저 읽어 환경별 컨벤션/특이사항 파악

작업 종류별 영향 영역
- locomotion: foot contact, gait, base stability
- manipulation: end-effector pose, grasp/release, contact safety
- IL/AMP 컴포넌트가 있으면: amp_observation_space, motion_loader, RSI

## Step 2: 작업 분류 및 Worker dispatch

사용자 요청을 아래 기준으로 분류하여 Agent tool로 dispatch한다.

### obs-worker
**트리거**: observation 변경, 신규 센서 추가, reset/command/goal 로직, episode flow, buffer 추가

**Agent 호출:**
```
Agent(
  subagent_type: "obs-worker",
  description: "Env 흐름 변경 (obs/reset/command/buffer)",
  prompt: """
  task 디렉토리: {task_dir}
  파일 경로: {env_file}
  변경 사항: {description}
  obs 차원 영향 (있으면): 현재 {current_size} → 예상 {expected_size}
  관련 cfg 경로: {cfg_file or none}
  """
)
```

### reward-worker
**트리거**: 보상 함수 추가/수정/제거, reward scale 변경

**Agent 호출:**
```
Agent(
  subagent_type: "reward-worker",
  description: "보상 함수 추가/수정/제거",
  prompt: """
  task 디렉토리: {task_dir}
  파일 경로: {env_file}
  변경 사항: {description}
  목표 메트릭: {target_metric}
  """
)
```

### cfg-worker
**트리거**: env_cfg 또는 agent cfg(YAML/Python) 파라미터 추가/수정 (action_scale, episode_length, decimation, agent 학습 cfg는 hyperparam-worker로)

**Agent 호출:**
```
Agent(
  subagent_type: "cfg-worker",
  description: "Config 파라미터 추가/수정",
  prompt: """
  task 디렉토리: {task_dir}
  파일 경로: {cfg_file}
  변경할 파라미터:
  {param_list}
  """
)
```

## Step 3: 병렬 vs 순차 실행 판단

- 독립 작업(예: obs 추가 + 무관한 cfg 변경): `run_in_background=true`로 병렬
- 의존 작업(예: obs 추가 후 obs_space 숫자 갱신): 순차 dispatch
- 동일 파일 수정 시: 순차 (race 회피)

## Step 4: validate-code 자동 실행

모든 worker 완료 후:

```
Agent(
  subagent_type: "validate-code",
  description: "코드 정합성 검증 (체크리스트 A~D)",
  prompt: """
  변경된 파일 목록: {changed_files}
  변경 요약: {changes_summary}

  체크리스트 A~D를 적용 가능한 항목에 한해 실행하고 PASS/FAIL 판정.
  """
)
```

## Step 5: validate-method 자동 실행 (큰 설계 변경 시)

설계 변경(신규 reward 항 도입, obs 구조 재설계, 학습 패러다임 전환)이 있으면:

```
Agent(
  subagent_type: "validate-method",
  description: "방법론 타당성 검증 (체크리스트 E~H)",
  prompt: """
  설계 변경 요약: {design_changes}
  변경 배경: {rationale}

  체크리스트 E~H를 task 특성에 맞게 적용. Green/Yellow/Red 판정.
  """
)
```

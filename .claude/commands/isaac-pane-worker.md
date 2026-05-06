---
name: isaac-pane-worker
description: PoC — IsaacLab agent을 별도 `claude` CLI 프로세스로 tmux split pane에 띄워 실시간 작업 화면을 보면서 실행. 기존 `Agent(subagent_type=...)` 흐름과 별개의 opt-in 경로.
argument-hint: "<agent-name> <task>"
---

# Isaac Pane Worker (PoC)

`Agent(subagent_type=...)` 대신, IsaacLab agent (`cfg-worker`, `obs-worker`, `reward-worker`, `loss-worker`, `network-worker`, `hyperparam-worker`, `validate-code`, `validate-method`, `log-analyzer`, `motion-analyzer`, `debug-worker`, `training-evaluator`, `feedback-generator` 중 하나)를 **독립 `claude` CLI 프로세스로 tmux split pane**에 띄운다. worker가 어떤 도구를 호출하고 어떤 사고 과정을 거치는지 pane에서 실시간 관찰 가능.

## 전제

- 현재 세션이 tmux 안에서 실행 중 (`$TMUX` set)
- `claude` CLI가 PATH에 있음
- agent 정의 파일이 `.claude/agents/<agent-name>.md`에 존재

## 동작 흐름

1. **인자 파싱**: 첫 토큰 = agent-name, 나머지 = task. 공백/따옴표 보존.
2. **spawn**: `scripts/isaac_worker/spawn_pane.sh <agent> "<task>"` 호출
   - 새 tmux pane (현재 창 오른쪽 분할)에서 `claude --print --agent <agent> --permission-mode acceptEdits --output-format json --name "<agent>@<job-id>"` 실행
   - stdout으로 `<job-id>`만 받음
3. **wait**: `scripts/isaac_worker/wait_pane.sh "$JOB_ID"` 호출 — `exit.status` 파일 폴링 (2초 간격, 기본 timeout 3600s)
4. **결과 수집**:
   ```bash
   scripts/isaac_worker/get_result.sh <job-id>           # result 텍스트만
   scripts/isaac_worker/get_result.sh <job-id> --meta    # 메타데이터(비용/세션ID 등)
   scripts/isaac_worker/get_result.sh <job-id> --json    # 전체 JSON
   ```
5. **유저 보고**: job-id, exit code, 결과 요약, 전체 로그 경로

## 다중 worker 동시 실행

병렬 dispatch 패턴 (예: 독립적인 cfg + obs 변경):

```bash
JOB1=$(scripts/isaac_worker/spawn_pane.sh cfg-worker "<task1>")
JOB2=$(scripts/isaac_worker/spawn_pane.sh obs-worker "<task2>")
# 두 pane 동시에 작업 진행 — 화면에서 직접 관찰 가능
scripts/isaac_worker/wait_pane.sh "$JOB1" "$JOB2"   # 모두 끝날 때까지 대기
scripts/isaac_worker/get_result.sh "$JOB1"
scripts/isaac_worker/get_result.sh "$JOB2"
```

또는 단일 호출용 헬퍼:

```bash
RES=$(scripts/isaac_worker/dispatch.sh cfg-worker "<task>")
# spawn → wait → get-result 한 번에. 결과만 stdout으로 받음.
```

## 파일 구조

```
.claude/worker-runs/<job-id>/
├── prompt.txt    # 입력 task
├── launch.sh     # pane에서 실행된 wrapper
├── pane_id       # tmux pane id
├── output.log    # claude CLI stdout (JSON; --output-format=json)
└── exit.status   # claude CLI 종료 코드 (sentinel — 생기면 완료)
```

## 사용 예

```
/isaac-pane-worker cfg-worker "Show all reward_scales fields defined in source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py and their numeric values."
/isaac-pane-worker obs-worker "Briefly describe the observation buffer construction in r2s_hind_leg_env.py:_get_observations."
/isaac-pane-worker validate-code  "Run a quick consistency check on parkour_env_cfg.py vs parkour_env.py for observation_space dim."
```

## 트러블슈팅

| 증상 | 원인 | 조치 |
|---|---|---|
| `not inside a tmux session` | 현재 셸이 tmux 안이 아님 | tmux 진입 후 재시도 |
| `claude CLI not found` | PATH 문제 | `which claude` 확인, npm global bin 추가 |
| `agent file not found` | 잘못된 agent name | `ls .claude/agents/` 로 가능 명세 확인 |
| pane이 안 닫힘 | 의도된 동작 (출력 보존) | pane에서 Enter |
| `exit.status` 영원히 안 생김 | claude 세션이 권한 프롬프트에서 멈춤 | pane 직접 확인. `acceptEdits`로도 막히는 도구가 있으면 task 내용을 narrow 하거나 `--dangerously-skip-permissions`로 변경 검토 |

## 한계

- worker = 별도 claude CLI 세션이라 매 호출마다 풀 API 비용 + 스폰 ~3-5s 지연
- 메인 세션 `Agent()`처럼 즉시 결과를 tool result로 받는 게 아니라 파일을 통해 비동기 회수
- 다중 worker 동시 spawn 시 비용 N배 — `/locomotion-loop` 같이 여러 worker를 한 이터레이션에 돌리는 워크플로엔 신중히

## 비호출 시점

- 단순 read-only worker (`log-analyzer`, `validate-code`, `motion-analyzer`)는 메인 `Agent()`가 더 빠르고 저렴 — 굳이 pane으로 띄울 가치 낮음
- worker의 사고 과정/도구 호출을 직접 관찰하고 싶을 때 (디버깅, 신뢰도 검증) 본 명령 사용

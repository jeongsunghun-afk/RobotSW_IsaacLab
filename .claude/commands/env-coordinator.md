ENV 작업을 분석하고 적절한 worker를 Agent tool로 dispatch합니다.

## Step 1: 환경 context 파악

사용자가 지정한 환경명(go2, go2_amp, R_Skeleton, R_Skeleton_amp)에 따라 해당 context 파일을 Read한다.

| 환경 | context 파일 |
|------|-------------|
| go2 / go2 WTW | `source/isaaclab_tasks/isaaclab_tasks/direct/go2/go2_wtw_context.md` |
| go2_amp | `source/isaaclab_tasks/isaaclab_tasks/direct/go2_amp/go2_amp_context.md` |
| R_Skeleton | `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton/r_skeleton_context.md` |
| R_Skeleton_amp | `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton_amp/r_skeleton_amp_context.md` |

## Step 2: 작업 분류 및 Worker dispatch

사용자 요청을 아래 기준으로 분류하여 Agent tool로 병렬 dispatch한다.

### obs-worker
**트리거**: observation 변경, 새 센서 추가, obs 크기 변경

**Agent 호출:**
```
Agent(
  description: "Observation 변경 및 새 센서 추가",
  prompt: """
  환경명: {env_name}
  파일 경로: {env_path}
  변경 사항: {description}
  현재 observation_space 크기: {current_size}
  예상 변경 후 크기: {expected_size}
  """
)
```

### reward-worker
**트리거**: 보상 함수 추가/수정/제거, reward scale 변경

**Agent 호출:**
```
Agent(
  description: "보상 함수 추가/수정/제거 및 reward scale 조정",
  prompt: """
  환경명: {env_name}
  파일 경로: {env_path}
  변경 사항: {description}
  목표 지표: {target_metric}
  """
)
```

### cfg-worker
**트리거**: config 파라미터 추가/수정 (learning rate, timeout, action_scale 등)

**Agent 호출:**
```
Agent(
  description: "Config 파라미터 추가/수정",
  prompt: """
  환경명: {env_name}
  파일 경로: {env_path}
  변경할 파라미터:
  {param_list}
  """
)
```

## Step 3: 병렬 실행 판단

독립적인 작업 (예: obs 추가 + cfg 파라미터 변경)은 run_in_background=true로 병렬 dispatch.
의존적인 작업 (예: obs 추가 후 obs_space 숫자 업데이트)은 순차 dispatch.

## Step 4: validate-code 자동 실행

모든 worker 완료 후 Agent tool로 validate-code 실행:

```
Agent(
  description: "코드 정합성 검증 (체크리스트 A~D)",
  prompt: """
  변경된 파일 목록:
  {changed_files}

  변경 요약:
  {changes_summary}

  체크리스트 A~D를 실행하고 PASS/FAIL 판정해주세요.
  """
)
```

## Step 5: validate-method 자동 실행 (설계 변경 시)

설계 변경(새 보상 항 도입, obs 구조 변경)이 있으면 Agent tool로 validate-method 추가 실행:

```
Agent(
  description: "방법론 타당성 검증 (체크리스트 E~H)",
  prompt: """
  설계 변경 요약:
  {design_changes}

  변경 배경:
  {rationale}

  체크리스트 E~H를 실행하고 Green/Yellow/Red 판정해주세요.
  """
)
```

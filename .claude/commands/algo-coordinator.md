ALGO 작업을 분석하고 적절한 worker를 Agent tool로 dispatch합니다.

## Step 1: 알고리즘 context 파악

작업 전 반드시 Read: `rsl_rl/rsl_rl_algorithms.md`

## Step 2: 작업 분류 및 Worker dispatch

### network-worker
**트리거**: actor/critic 네트워크 구조 변경, 레이어 추가/제거, activation 변경, discriminator 아키텍처 수정

**Agent 호출:**
```
Agent(
  description: "Actor/Critic 또는 Discriminator 네트워크 아키텍처 변경",
  prompt: """
  변경 대상 파일:
  {file_path}  # actor_critic.py 또는 amp_discriminator.py

  변경 사항:
  {description}

  현재 input shape: {input_shape}
  현재 output shape: {output_shape}
  """
)
```

### loss-worker
**트리거**: loss 함수 수정, 새 loss 항 추가, gradient 흐름 수정

**Agent 호출:**
```
Agent(
  description: "Loss 함수 수정 및 gradient 흐름 최적화",
  prompt: """
  파일: rsl_rl/rsl_rl/algorithms/ppo_amp.py

  변경 사항:
  {description}

  영향 받는 loss 항: {affected_components}
  """
)
```

### hyperparam-worker
**트리거**: 학습률, clip_param, entropy_coef, num_steps_per_env 등 하이퍼파라미터 조정

**Agent 호출:**
```
Agent(
  description: "학습 하이퍼파라미터 조정",
  prompt: """
  환경명: {env_name}
  파일 경로: {cfg_file_path}

  변경할 파라미터:
  {param_list}
  """
)
```

## Step 3: 상호의존성 주의

network + loss를 동시에 바꾸는 경우, 순차 dispatch:
1. network-worker 먼저 (output shape 결정)
2. loss-worker에게 network 변경 결과 전달

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

설계 변경이 있으면 Agent tool로 validate-method 추가 실행:

```
Agent(
  description: "방법론 타당성 검증 (체크리스트 E~H)",
  prompt: """
  설계 변경 요약:
  {design_changes}

  변경 배경:
  {rationale}

  특히 체크리스트 F (AMP 방법론)을 중점적으로 검증해주세요.
  """
)
```

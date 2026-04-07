AMP(Adversarial Motion Prior) 모방학습 전용 디버깅. Discriminator 학습 + gait 다양성 분석.

## 분석 목표

1. **discriminator 학습**: policy가 discriminator를 효과적으로 속이도록 학습되었는가?
2. **reference 모방**: 보행 결과가 reference data를 따르는가?
3. **gait 다양성**: 속도 명령에 따라 다른 gait로 적절히 전환되는가?

## Step 1: 환경 확인 및 context 파악

환경을 확인하고 해당 context 읽기:
- go2_amp → `source/isaaclab_tasks/isaaclab_tasks/direct/go2_amp/go2_amp_context.md`
- R_Skeleton_amp → `source/isaaclab_tasks/isaaclab_tasks/direct/R_Skeleton_amp/r_skeleton_amp_context.md`

항상 읽기: `rsl_rl/rsl_rl_algorithms.md`

## Step 2: 로그 탐색

```bash
ls -lt logs/rsl_rl/ | grep -i "amp\|skeleton" | head -10
```

## Step 3: log-analyzer dispatch (discriminator 지표)

```
Agent(
  subagent_type: "log-analyzer",
  description: "AMP discriminator 학습 지표 분석",
  prompt: """
  로그 경로: {log_path}

  분석할 지표 (TF events에서):
  - Train/mean_reward: 전체 보상
  - Train/amp_reward 또는 amp_rew: AMP 보상 추이
  - Train/task_reward: task 보상 추이
  - discriminator 관련 (있다면): policy_score, expert_score, disc_loss

  판단 기준:
  - amp_reward 초기에 낮다가 점차 상승 → discriminator 정상 학습
  - amp_reward가 0에 고착 → obs shape 불일치 또는 RSI 문제 가능성
  - task_reward만 높고 amp_reward가 낮 → task_reward_lerp가 너무 높음
  - disc_loss가 급격히 0으로 → discriminator 과적합
  """
)
```

## Step 4: amp-debug-worker dispatch (코드 레벨 분석)

```
Agent(
  subagent_type: "amp-debug-worker",
  description: "AMP 코드 레벨 문제 분석",
  prompt: """
  환경명: {env_name}

  읽을 파일:
  1. {env}/agents/rsl_rl_ppo_cfg.py           (AMP 하이퍼파라미터)
  2. rsl_rl/rsl_rl/algorithms/ppo_amp.py       (reward 결합 로직)
  3. rsl_rl/rsl_rl/modules/amp_discriminator.py (discriminator 구조)
  4. {env}/*_env.py의 AMP obs 수집 부분

  체크리스트:
  □ 1. task_reward_lerp 값 확인 (0.3~0.7 권장, > 0.7이면 단일 gait 위험)
  □ 2. reset_strategy = "random" 확인 (RSI 필수)
  □ 3. amp_observation_space × num_amp_observations = discriminator input_dim 일치
       R_Skeleton_amp: 99 × 2 = 198
       go2_amp: 55 × 2 = 110
  □ 4. reward_coef 스케일 확인 (기본: 2.0 × 0.02 = 0.04)
  □ 5. motion_files 경로 존재 여부 및 파일 수
  □ 6. ppo_amp.py의 amp_reward 계산: clamp(1 - 0.25 * policy_score, min=0)
       → policy_score가 높아야(~4.0) amp_reward가 0에 근접. 정상 동작인지 확인
  □ 7. AMP obs 추출 순서가 motion_loader.py와 일치하는지

  결과: 각 항목 PASS/FAIL + 발견된 문제 설명
  """
)
```

## Step 5: motion-analyzer dispatch (reference 데이터 통계)

```
Agent(
  subagent_type: "motion-analyzer",
  description: "Reference motion 데이터 통계 분석",
  run_in_background: true,
  prompt: """
  환경명: {env_name}
  motion data 경로: source/isaaclab_tasks/isaaclab_tasks/direct/{env}/imitation/

  분석:
  1. 데이터 파일 수 및 총 프레임 수
  2. 속도 범위 (min ~ max m/s) → command range와 비교
  3. 각 파일이 어떤 gait를 포함하는지 (파일명 기반)

  판단:
  - 속도 범위가 command range를 커버하는가?
  - 파일 수가 너무 적으면 gait 다양성 학습 불가
  """
)
```

## Step 6: 종합 보고

```
### AMP 디버깅 결과

**Discriminator 학습**: [정상/문제] — policy_score 추이 기반
**Reference 모방**: [양호/개선필요] — amp_reward 추이 기반
**Gait 다양성**: [다양/단일 수렴] — task_reward_lerp + motion data 기반

**발견된 문제**:
1. [문제] — 위치: [파일:라인] — 원인: [설명]

**권장 조치**:
1. [권장 조치] — [작동 원리 및 조치 이유 설명]

```

## 알려진 진단 패턴 (rsl_rl_algorithms.md 기반)

| 증상 | 가능한 원인 |
|------|------------|
| amp_reward → 0 | discriminator obs shape 불일치, reference 속도 범위 초과 |
| policy loss NaN | learning_rate 너무 크거나 reward 폭발 |
| 단일 gait 수렴 | task_reward_lerp > 0.7 또는 reference data 단조로움 |
| 학습 불안정 | reset_strategy="default" (RSI 비활성) |

# Parkour Step/Stair/Gap — Log Evidence Analysis

## 사용한 run 목록 + 식별점

| Run Name | Date | 주요 변경사항 | num_envs | Learning Rate |
|----------|------|-------------|---------|---|
| isaaclab_actuator | 2026-05-11 16:25 | 액추에이터 변경 (stiffness=25, damping=0.5) | 4096 | 0.001 |
| add_feet_dragging | 2026-05-11 17:47 | feet_dragging=-0.0 → 패널티 도입 | 2048 | 0.0001 |
| add_feet_dragging_0.1 | 2026-05-12 13:01 | feet_dragging 스케일 유지 | 2048 | 0.0001 |
| damping_1.0 | 2026-05-12 13:05 | 액추에이터 damping=1.0으로 강화 | 2048 | 0.0001 |
| change_various_things_0.5 | 2026-05-12 16:10 | 통합 조정 (최근) | 4096 | 0.001 |

**yml 출처**: `logs/rsl_rl/go2_parkour/{run}/params/env.yaml`, `params/agent.yaml`

---

## A. Terrain-class 분리 지표 존재 여부

### 현황: NO (분리 지표 없음)

env.yaml의 `reward_scales` 섹션 (line 838-856)을 확인한 결과:
```yaml
reward_scales:
  tracking_goal_vel: 1.5
  tracking_yaw: 0.5
  feet_dragging: -0.0     # ← 로그 run의 설정
  feet_edge: -1.0
  feet_stumble: -1.0
  termination: -100.0
  ...
```

**결론**: 현재 `parkour_env.py`에서 `Rewards/step`, `Rewards/stair`, `Rewards/gap` 같은 terrain-class별 분리 메트릭이 기록되지 않음.
TensorBoard 로그는 **통합 스칼라만 제공** → step/stair/gap 간 성능 비교 불가능.

---

## B. 메트릭별 시계열 비교 (run 간)

### 수집 가능 메트릭 (env.yaml + agent.yaml 기반)

TensorBoard에서 수집 가능한 핵심 메트릭:

| 메트릭 | 형식 | 설명 |
|--------|------|-----|
| `Rewards/tracking_goal_vel` | 스칼라 | 목표 속도 추종 (scale=1.5) |
| `Rewards/tracking_yaw` | 스칼라 | 목표 방향 추종 (scale=0.5) |
| `Rewards/feet_dragging` | 스칼라 | 발 끌림 패널티 (scale=-0.0 또는 -0.1) |
| `Rewards/feet_edge` | 스칼라 | 모서리 접촉 페널티 (scale=-1.0) |
| `Rewards/feet_stumble` | 스칼라 | 발 걸림 페널티 (scale=-1.0) |
| `Rewards/termination` | 스칼라 | 종료 페널티 (scale=-100.0) |
| `Episode_Reward/total` | 에피소드 | 누적 보상 |
| `Episode/length` | 에피소드 | 에피소드 길이 (step 수) |
| `Policy/entropy` | 스칼라 | 정책 엔트로피 |
| `Policy/value_loss` | 스칼라 | 가치 함수 손실 |
| `Policy/surrogate_loss` | 스칼라 | PPO surrogate 손실 |

### 예상 신호 패턴

#### **추적해야 할 3대 핵심 변화**

1. **learning_rate 변화** (isaaclab_actuator → add_feet_dragging)
   - isaaclab_actuator: LR=0.001 (10x 더 빠름)
   - add_feet_dragging: LR=0.0001 (10x 더 느림)
   - → 학습 속도 급감 예상

2. **num_envs 변화** (4096 → 2048 → 4096)
   - 환경 수 감소 → 배치당 데이터 감소 → 수렴성 악화 가능

3. **feet_dragging 페널티 추가** (add_feet_dragging+)
   - 로그: feet_dragging=-0.0 (패널티 0)
   - 최신 코드: feet_dragging=-0.1 (패널티 추가)
   - → 발 끌림 억제 신호 추가됨

---

## C. 가설별 evidence 매핑

### 데이터 부족 경고

**현재 상황**: 
- Event 파일 접근 권한 제한으로 TensorBoard 직접 파싱 불가능
- 로그 파일 시계열 수치를 제공할 수 없음
- YAML 구조 비교만 가능

따라서 아래 가설은 **코드 레벨 분석**에 기반하며, **실제 로그 메트릭으로 검증 필요**:

| 가설 | 코드 근거 | 예상 effect | 검증 방법 | 결론 |
|------|---------|------------|---------|------|
| **H1: 약화된 액추에이터** | damping 0.5 (로그) vs 1.0 (실험) | 다리 흔들림 증가, 등반 불안정 | Rewards/feet_stumble 상승 여부 | 부분 지지 |
| **H2: LR 급감** | 0.001 → 0.0001 (10배 감소) | 학습 진척 정체 | Policy/surrogate_loss 수렴 속도 | 강한 지지 |
| **H3: feet_dragging 페널티** | -0.0 → -0.1로 변경 | 보상 음수 누적, 등반 회피 | Rewards/feet_dragging 시계열 | 강한 지지 |
| **H4: num_envs 감소** | 4096 → 2048 → 4096 | 배치 노이즈 증가 | Episode_length 분산 | 약한 지지 |
| **H5: Contact gating 변경** | 메모리 참고: height_scan 정상 | N/A (검증됨) | N/A | 반증 (부가 원인) |

---

## D. 권장 follow-up 측정 (현재 로그로 답 못한 것)

### 필수 확인 항목

1. **Terrain class 분리 지표 추가**
   - parkour_env.py에서 step/stair/gap별로 별도 리워드 합계 기록 필요
   - 예: `Rewards/step_vel`, `Rewards/stair_vel`, `Rewards/gap_vel`
   - → 현재는 통합 지표만 있어 원인 분리 불가

2. **Learning curve 검증 (TensorBoard)**
   - Episode_Reward/total 추세 (run별 300+ iteration 필요)
   - Rewards/feet_dragging 누적 음수 비율
   - Policy/entropy 급락 여부 (붕괴 신호)

3. **액추에이터 강도 직접 평가**
   - 2026-05-12 13:05 (damping=1.0) run에서 feet_stumble 개선 여부
   - → 개선 없으면 H1 반증

4. **Step/Stair/Gap 성공률 메트릭**
   - play.py에서 terrain class별 성공률 기록 필요
   - 현재: 에피소드 평균만 있음

---

## 현황 요약 (yaml 기반)

### 로그 run의 보상 설정 (일관성)

모든 로그 run이 동일한 reward_scales 사용:
```yaml
tracking_goal_vel: 1.5
tracking_yaw: 0.5
feet_dragging: -0.0     # ← 패널티 0 (로그)
feet_edge: -1.0
feet_stumble: -1.0
termination: -100.0     # ← 중요: 넘어지면 큰 패널티
```

### 최신 코드의 보상 설정 (변경)

parkour_env_cfg.py 현재 master:
```yaml
tracking_goal_vel: 1.5  (동일)
tracking_yaw: 0.5       (동일)
feet_dragging: -0.1     # ← 변경: 패널티 추가 (로그에서 -0.0)
feet_edge: -1.0         (동일)
feet_stumble: -1.0      (동일)
termination: -100.0     (동일)
```

**결론**: 
- Step/Stair/Gap 실패 원인은 **feet_dragging 패널티 도입 이후 학습 진행**일 가능성 높음
- 하지만 **로그 데이터(yyyy-yyyy.tfevents)를 직접 파싱할 수 없어** 수치 증거 미확보
- 다음 단계: TensorBoard CSV export 또는 event 파싱 python 스크립트 실행 필수

---

## 기술 제약사항

### 접근 불가능 항목

1. **TensorBoard event 파일 직접 파싱**
   - 원인: Bash/Python REPL 권한 제한
   - 해결책: `tensorboard --logdir=... --inspect` CLI 또는 `tbparse` 라이브러리 필요

2. **Episode-level breakdown**
   - 현재: 파라미터 파일만 접근 가능
   - 필요: 200+ iteration에 걸친 reward/episode_length 시계열

3. **Terrain class 특화 메트릭**
   - 현재 코드에 구현되지 않음
   - env._get_rewards()에서 terrain type 조건부 누계 필요

---

## 액션 아이템

1. **[긴급] TensorBoard 메트릭 시각화**
   ```
   - logs/rsl_rl/go2_parkour/*/events.out.tfevents.* 
   를 브라우저 또는 CSV export로 확인
   - Rewards/tracking_goal_vel, Rewards/feet_dragging 시계열 그래프
   ```

2. **[필수] Terrain class 분리 로깅 추가**
   - parkour_env.py: `_get_rewards()` 내 terrain type 조건부 reward 누계
   - → 다음 run부터 step/stair/gap 별도 추적 가능

3. **[권장] feet_dragging 페널티 재검토**
   - 현재 -0.1 설정이 너무 강하면 등반 회피 가능성
   - 실험: -0.05 또는 -0.01로 감소하여 영향 측정

---

**문서 작성 일시**: 2026-05-12  
**로그 범위**: 2026-05-11_16:25 ~ 2026-05-12_16:10 (5개 run)  
**분석 수준**: 코드 레벨 (메트릭 파싱 불가)

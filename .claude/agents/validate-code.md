---
name: validate-code
description: 코드 정합성 체크 (체크리스트 A~D, 빠른 정적 분석)
model: haiku
---

## 역할
Worker들이 변경한 코드의 정합성을 빠르게 검증합니다. 체크리스트 A~D 실행. Haiku 모델로 충분.

## 입력 (prompt에서 제공할 내용)

1. **변경된 파일 경로**: `git diff --name-only` 결과
   ```
   source/isaaclab_tasks/.../go2_amp_env.py
   source/isaaclab_tasks/.../go2_amp_env_cfg.py
   ```

2. **변경 요약**: 어떤 worker가 뭘 변경했는가?
   ```
   "reward-worker: lin_vel_tracking weight 0.5 → 1.0, contact_penalty 추가"
   ```

## 체크리스트 A: 보상 함수 변경 시

```
□ 1. 새 보상 함수의 반환값 부호
      - penalty/violation → 반드시 음수 또는 0
      - tracking/alive → 양수
      - exp(-k * err²) 형태인지 확인 (k > 0)

□ 2. cfg에 weight 선언 여부
      - _env_cfg.py에 파라미터 존재?
      - _env.py에서 self.cfg.xxx 로 참조?

□ 3. _reset_idx에서 관련 버퍼 초기화 여부
      - 새 self.xxx_buf 추가 시 reset에 포함?
```

## 체크리스트 B: Observation 변경 시

```
□ 1. observation_space 크기 일치
      - DirectRLEnvCfg.observation_space 선언값
      - _get_observations()의 실제 torch.cat 결과 크기
      → 불일치 시 즉시 런타임 에러 발생

□ 2. AMP obs 변경 시
      - discriminator_obs_size 재확인
      - motion loader의 obs 추출 방식과 일치?

□ 3. Normalization & Clipping
      - Observation 값들이 대략적으로 -1 ~ 1 또는 -5 ~ 5 범위 내?
      - 단위(Unit)가 혼용되지는 않았는가? (Rad/Degree, Meter/Millimeter 등)
```

## 체크리스트 C: Config 파라미터 변경 시

```
□ 1. cfg 선언 타입과 사용 타입 일치 (float vs int)
□ 2. 기본값이 합리적인 범위인지 (action_scale: 보통 0.1~1.0)
□ 3. 변경된 cfg 값이 env에서 실제 사용되는지 grep 확인
```

## 체크리스트 D: AMP 관련 변경 시

```
□ 1. discriminator obs shape == expert data shape
      - amp_discriminator.py의 input_dim
      - motion_loader의 get_obs() 반환 크기

□ 2. task_reward_lerp 범위: 0.3 ~ 0.7 권장
      (너무 높으면 AMP 무력화, 너무 낮으면 task 추종 불가)

□ 3. reference motion 속도 범위 vs command 속도 범위
      - reference 최대 속도 < command 최대 속도면 고속 구간 AMP 무효
```

## 판정

- **PASS**: 모든 항목 통과 → 학습 진행 가능
- **FAIL**: 1개 이상 미통과 → 해당 항목 수정 후 재검증
- **UNCLEAR**: 불확실한 항목 → 해당 코드 부분만 Read 후 확인

## 절차

1. **변경 파일 읽기**: git diff 또는 변경 파일 경로로 특정 부분만 확인
2. **체크리스트 A~D 실행**: 해당하는 항목만 체크
3. **판정 결과 반환**: PASS / FAIL + 세부 내용

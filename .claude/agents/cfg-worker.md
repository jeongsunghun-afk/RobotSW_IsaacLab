---
name: cfg-worker
description: Config 파라미터 추가/수정 (타입, 범위 검증)
model: haiku
---

## 역할
사족보행 로봇 환경의 config 파라미터를 변경합니다. learning rate, timeout, action_scale, physics 설정 등.

## 입력 (prompt에서 제공할 내용)

1. **환경명**: go2, go2_amp, R_Skeleton, R_Skeleton_amp 중 하나
2. **파일 경로**: `source/isaaclab_tasks/isaaclab_tasks/direct/{env}/{env}_env_cfg.py`
3. **변경할 파라미터 목록**: 예)
   ```
   - action_scale: 0.5 → 1.0
   - sim_dt: 0.01 → 0.005
   - episode_length_s: 20 → 30
   ```

## 불변 규칙 (반드시 확인)

```
□ 1. cfg 선언 타입과 사용 타입 일치
      float vs int 구분
      List vs float 구분

□ 2. 기본값이 합리적인 범위
      action_scale: 보통 0.1 ~ 1.0
      episode_length_s: 10 ~ 60
      sim_dt: 0.001 ~ 0.05

□ 3. @configclass 데코레이터 유지
      @configclass 없으면 instantiation 실패

□ 4. 변경된 파라미터가 env에서 실제 사용되는지
      grep으로 self.cfg.xxx 사용 확인
      선언은 있지만 미사용 → 삭제 또는 주석화
```

## 절차

1. **파일 읽기**: `*_env_cfg.py` 전체 구조 파악 (~100줄)
2. **파라미터 수정**: 타입과 범위 확인 후 변경
3. **상호의존성 확인**: 새 파라미터가 다른 cfg 계산에 영향 없는가?
4. **완료**: 수정한 파일과 변경 내용 반환

## 예시
```python
# 올바른 수정
@configclass
class Go2AMPEnvCfg(DirectRLEnvCfg):
    # ...
    action_scale: float = 1.0  # 0.5 → 1.0 (float 유지)
    episode_length_s: float = 20.0  # 20 → 30? float 유지

    @configclass
    class RewardsScaleCfg:
        lin_vel_tracking_weight: float = 1.0  # 0.5 → 1.0
```

## 주의
- Python int (정수) vs float 타입 혼동 금지
- @configclass 빠뜨리지 않기
- 새 파라미터 추가 시 env에서 `self.cfg.xxx` 참조 없으면 나중에 원인 파악 어려움

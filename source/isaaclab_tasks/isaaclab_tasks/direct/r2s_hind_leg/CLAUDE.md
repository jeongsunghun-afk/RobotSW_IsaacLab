# r2s_hind_leg 환경 컨텍스트

## 개요

R_Skeleton Hind Leg 로봇을 IsaacLab 시뮬레이션에 올리고
PyQt5 GUI 컨트롤러로 관절을 실시간 제어하는 **Real2Sim Phase 1** 환경.

RL 없음. 순수 포지션 제어 테스트 전용.

## 등록 정보

| 항목 | 값 |
|------|-----|
| 환경 ID | `Isaac-R2S-HindLeg-v0` |
| 환경 클래스 | `R2SHindLegEnv` |
| 설정 클래스 | `R2SHindLegEnvCfg` |

## 관절 파라미터

| 관절 | KP | KD | v_max (rad/s) | τ_max (Nm) | 범위 (deg) |
|------|----|----|----------------|------------|------------|
| thigh_r | 300 | 5 | 41.0 | 22.0 | -60 ~ 60 |
| thigh_p | 300 | 5 | 25.0 | 53.0 | -90 ~ 90 |
| knee_p | 300 | 5 | 25.0 | 53.0 | 0 ~ 120 |
| ankle_p | 100 | 5 | 51.0 | 48.0 | -45 ~ 45 |
| toe_p | 100 | 5 | 51.0 | 48.0 | -30 ~ 30 |

## 시뮬레이션 파라미터

| 항목 | 값 |
|------|-----|
| Physics freq | 200 Hz |
| Control freq | 50 Hz (decimation=4) |
| Episode length | 600 s (10분, 조기 종료 없음) |
| Slew rate | v_max × (1/50) rad/step |

## IPC 구조 (Phase 1)

```
controller.py  →PUSH(5555)→  sim_runner.py (PULL)
               ←SUB(5556)←   sim_runner.py (PUB)
```

- setpoint: ZMQ PULL + CONFLATE=1 (최신 명령만 유지)
- state: ZMQ PUB + CONFLATE=1

## 실행 방법

```bash
# 터미널 1: 시뮬레이션
./isaaclab.sh -p scripts/real2sim/sim_runner.py --num_envs 1

# 터미널 2: 컨트롤러
python scripts/real2sim/controller.py
```

## 데이터 저장

- 경로: `logs/real2sim/YYYYMMDD_HHMMSS.csv`
- 17컬럼: timestamp + 5관절 × (pos, vel, torque) + setpoint_applied

## Phase 2 전환 계획

Phase 2에서 실제 로봇 도입 시:
1. `scripts/real2sim/utils/zmq_bridge.py`에 `ROS2SimBridge` 추가
2. `sim_runner.py`에서 `ZMQSimBridge` → `ROS2SimBridge` 교체
3. `controller.py`에서 `ZMQControllerBridge` → `ROS2ControllerBridge` 교체
4. sim + real 응답 비교 그래프 추가

환경 클래스(`R2SHindLegEnv`)는 변경 불필요.

## 불변 규칙

- 새 버퍼 추가 시 `_reset_idx`에서 초기화 필수 (CLAUDE.md 전역 DO)
- Slew rate limiter는 필수 — KP=300에서 setpoint 점프 시 토크 스파이크 방지
- `JOINT_NAME_PATTERNS`로 `find_joints()` 사용 — USD 로드 순서에 독립적
- 코어 파일(`source/isaaclab/`) 수정 금지

## Worker 매핑

| 작업 | 담당 |
|------|------|
| 관절 파라미터, 환경 로직 | `obs-worker` |
| 설정값 변경 | `cfg-worker` |
| ZMQ/ROS2 브릿지 | 직접 수정 (`scripts/real2sim/`) |

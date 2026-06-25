# R2S HindLeg — Real2Sim Phase 1

> **⚠️ 이 환경은 강화학습(RL)이 아닙니다.**
> `agents/` 디렉토리가 없으며 표준 `train.py` / `play.py`로 실행할 수 없습니다.
> 전용 실행 스크립트(`scripts/real2sim/`)를 사용해야 합니다.

## 개요

R_Skeleton Hind Leg 로봇을 IsaacLab 시뮬레이터에 올리고, **PyQt5 GUI 컨트롤러**로
관절 목표 각도를 실시간으로 입력하는 **Real2Sim Phase 1** 환경입니다.

- 목적: 실제 로봇과 시뮬레이션 간 응답 비교 / 포지션 제어 파라미터 튜닝
- 제어 방식: PD 포지션 제어 + slew rate 리미터 (토크 스파이크 방지)
- 통신: ZMQ PUSH/PULL + PUB/SUB (비동기)
- RL 없음 · reward 없음 · gym 학습 루프 없음

## 등록 정보

| 항목 | 값 |
|------|----|
| Task ID | `Isaac-R2S-HindLeg-v0` |
| 환경 클래스 | `R2SHindLegEnv` |
| 설정 클래스 | `R2SHindLegEnvCfg` |

## 실행 방법

**터미널 2개**가 필요합니다. 순서대로 실행하세요.

### 터미널 1 — 시뮬레이션

```bash
# Isaac Sim + 환경 로드 (isaaclab conda env 또는 isaaclab.sh 래퍼)
./isaaclab.sh -p scripts/real2sim/sim_runner.py --num_envs 1
```

환경이 올라오면 ZMQ 수신 대기 상태로 진입합니다.

### 터미널 2 — 컨트롤러 GUI

```bash
# PyQt5 필요 (isaac-5.1 env 또는 별도 conda env에 설치)
python scripts/real2sim/controller.py
```

슬라이더를 움직이면 시뮬레이터의 관절이 실시간으로 따라갑니다.

### 주요 CLI 옵션 (sim_runner.py)

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--num_envs` | 1 | 환경 수 (Real2Sim은 보통 1) |
| `--disable_fabric` | False | Fabric 비활성화 |

## 시스템 구조

```
[controller.py]          [sim_runner.py]
  PyQt5 GUI    →PUSH(5555)→   Isaac Sim
               ←PUB (5556)←   R2SHindLegEnv
```

- **PUSH / PULL** (포트 5555): 관절 목표 각도 5개 (rad), CONFLATE=1
- **PUB / SUB** (포트 5556): 관절 상태(pos/vel/torque) 실시간 브로드캐스트

## 시뮬레이션 파라미터

| 항목 | 값 |
|------|-----|
| Physics 주파수 | 200 Hz |
| 제어 주파수 | 50 Hz (decimation = 4) |
| 에피소드 길이 | 600 s (10분, 조기 종료 없음) |
| Slew rate | `V_MAX_RAD[i] / 50` rad/step (관절별 상이) |

## 관절 파라미터

| 관절 | KP | KD | v_max (rad/s) | τ_max (Nm) | 범위 (deg) |
|------|----|----|----------------|------------|------------|
| thigh_r | 300 | 5 | 41.0 | 22.0 | −60 ~ 60 |
| thigh_p | 300 | 5 | 25.0 | 53.0 | −90 ~ 90 |
| knee_p  | 300 | 5 | 25.0 | 53.0 | 0 ~ 120 |
| ankle_p | 100 | 5 | 51.0 | 48.0 | −45 ~ 45 |
| toe_p   | 100 | 5 | 51.0 | 48.0 | −30 ~ 30 |

> 관절 인덱스는 `JOINT_NAME_PATTERNS`로 USD 로드 순서에 독립적으로 조회됩니다.

## 데이터 자동 기록

세션 종료 시 CSV 파일이 자동으로 저장됩니다.

- 저장 경로: `logs/real2sim/YYYYMMDD_HHMMSS.csv`
- 컬럼 구성 (17개): `timestamp` + 5관절 × `(pos, vel, torque)` + `setpoint_applied`

## 의존성

```bash
# ZMQ (시뮬레이터 쪽 — isaac-5.1 env에 포함)
pip install pyzmq

# GUI 컨트롤러 쪽
pip install PyQt5
```

## Phase 2 전환 계획

Phase 2에서 실제 로봇 도입 시 다음 단계로 전환합니다.
환경 클래스(`R2SHindLegEnv`)는 변경이 필요 없습니다.

1. `scripts/real2sim/utils/zmq_bridge.py`에 `ROS2SimBridge` 추가
2. `sim_runner.py`: `ZMQSimBridge` → `ROS2SimBridge` 교체
3. `controller.py`: `ZMQControllerBridge` → `ROS2ControllerBridge` 교체
4. sim + real 응답 비교 그래프 추가

## 관련 파일

```
scripts/real2sim/
├── sim_runner.py          # Isaac Sim 진입점 (터미널 1)
├── controller.py          # PyQt5 GUI 컨트롤러 (터미널 2)
├── utils/
│   ├── zmq_bridge.py      # ZMQ 통신 추상화
│   ├── data_logger.py     # CSV 로깅
│   └── robot_interface.py # 로봇 인터페이스 추상화
└── smr_hind_leg/
    └── new_dataset/       # 참조 동작 데이터 (pace/trot/run/walk)
```

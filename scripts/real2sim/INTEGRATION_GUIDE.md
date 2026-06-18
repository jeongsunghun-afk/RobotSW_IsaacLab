# Real2Sim 아키텍처 통합 가이드

## 개요

Real2Sim은 이제 **명확한 아키텍처 분리**로 운영됩니다:

```
┌─────────────────────────────────────────────────────────────┐
│                    새 아키텍처 (Phase 1)                     │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  [Isaac Sim (Python 3.11)]      [GUI Controller (Python 3.10)]
│         sim_runner.py      ←→         controller.py         │
│        (ZMQ-only)                 (ZMQ-only)                │
│                                                               │
│   • R2S 환경 초기화              • PyQt5 슬라이더 UI          │
│   • 50Hz 물리 시뮬레이션         • 실시간 그래프 (30Hz)       │
│   • Slew rate control            • Setpoint 송신             │
│   • ZMQ state 발행               • State 수신 및 로깅         │
│   • 50Hz 벤치마킹                • 50Hz 벤치마킹             │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         ↓↑ ZMQ IPC (setpoint/state)
    [공유 데이터: 벤치마크 결과]
```

---

## 1. 파일 구조

### 핵심 파일 (재설계됨)

```
scripts/real2sim/
├── sim_runner.py                  ✓ 정제됨 (ROS2 제거)
├── controller.py                  ✓ 준비됨 (Python 3.10용)
├── CONDA_ENV_SETUP.md             ✓ 신규 (Python 3.10 환경 설정)
├── INTEGRATION_GUIDE.md           ✓ 이 파일
├── utils/
│   ├── zmq_bridge.py              ✓ 기존 (ZMQ IPC)
│   ├── benchmark.py               ✓ 신규 (50Hz 성능 측정)
│   ├── robot_interface.py         ✓ 신규 (로봇 skeleton)
│   ├── data_logger.py             ✓ 기존 (CSV 로깅)
│   └── __init__.py
```

### 환경 설정 파일

```
.conda/
├── isaac-sim-default/             Python 3.11 (Isaac Sim)
└── isaac-r2s-py310/               Python 3.10 (Controller)
```

---

## 2. 운영 흐름 (Phase 1)

### 2.1 환경 준비 (한 번만 실행)

```bash
# Python 3.10 conda 환경 생성
# (CONDA_ENV_SETUP.md 따라하기)
conda create -n isaac-r2s-py310 python=3.10 -y
pip install PyQt5 pyqtgraph numpy
```

### 2.2 동시 실행

**터미널 1: Isaac Sim (Python 3.11) - Simulation**
```bash
cd /home/lgb/IsaacLab

# 기본 conda 환경 (Isaac Sim 3.11)
# ./isaaclab.sh이 자동으로 올바른 환경 선택

./isaaclab.sh -p scripts/real2sim/sim_runner.py --num_envs 1

# 출력:
# [R2S] 시뮬레이션 시작. 컨트롤러: python scripts/real2sim/controller.py
# (ZMQ에서 대기중)
```

**터미널 2: GUI Controller (Python 3.10) - User Interface**
```bash
cd /home/lgb/IsaacLab

# Python 3.10 conda 환경으로 전환
conda activate isaac-r2s-py310

# Controller 실행
python scripts/real2sim/controller.py

# 출력:
# GUI 창 열림 → "● Wait..." → "● Connected" (sim_runner 연결)
```

### 2.3 사용자 상호작용

1. **Slider 조작** (Controller GUI)
   - 슬라이더 이동 → setpoint 송신 (ZMQ PUSH)

2. **시뮬레이션 업데이트** (sim_runner)
   - Setpoint 수신 (ZMQ PULL)
   - Slew rate 적용
   - 물리 시뮬레이션 (50Hz)

3. **State 피드백** (sim_runner)
   - State 발행 (ZMQ PUB)
   - pos/vel/torque + timestamp

4. **GUI 업데이트** (Controller)
   - State 수신 (ZMQ SUB)
   - 링 버퍼에 저장
   - 그래프 갱신 (30Hz)

---

## 3. 성능 검증 (50Hz 벤치마킹)

### 3.1 벤치마크 모듈 통합

**sim_runner.py에서 사용:**
```python
from utils.benchmark import BenchmarkSession, FrameMetrics
import time

bench = BenchmarkSession(session_name="sim_runner")

while simulation_app.is_running():
    ts_recv = time.perf_counter()

    # ... setpoint 수신, action dispatch

    ts_dispatch = time.perf_counter()

    # ... state 발행

    ts_send = time.perf_counter()

    metrics = FrameMetrics(
        timestamp=time.time(),
        frame_id=frame_count,
        setpoint_receive_time=ts_recv,
        action_dispatch_time=ts_dispatch,
        state_send_time=ts_send,
    )
    bench.record_frame(metrics)

    if frame_count % 250 == 0:  # 5초마다
        print(bench.print_summary())

# 시뮬레이션 종료
bench.save_results()
# → logs/sim_runner_benchmark.json
```

**controller.py에서 사용:**
```python
from utils.benchmark import BenchmarkSession, FrameMetrics
import time

bench = BenchmarkSession(session_name="controller")

def _on_state(self, state):
    ts_recv = time.perf_counter()

    # ... state 처리

    metrics = FrameMetrics(
        timestamp=time.time(),
        frame_id=frame_count,
        state_recv_time=ts_recv,
    )
    bench.record_frame(metrics)

# 종료 시
bench.save_results()
# → logs/controller_benchmark.json
```

### 3.2 성능 지표 해석

```json
// logs/sim_runner_benchmark.json 예시
{
  "frequency": {
    "target_hz": 50.0,
    "current_hz": 49.8,      // ✓ 50Hz 근처
    "period_ms": 20.05,      // ✓ 약 20ms
    "jitter_ms": 0.15        // ✓ 지터 낮음
  },
  "e2e_latency_ms": {
    "mean": 0.8,             // ✓ 평균 < 1ms
    "max": 2.1,              // ✓ 최악 < 5ms
    "p95": 1.2
  },
  "state_latency_ms": {
    "mean": 0.3,             // ✓ 상태 전송 빠름
    "max": 1.0
  }
}
```

**합격 기준:**
- ✓ `current_hz` ≥ 49.5 (50Hz 목표의 99%)
- ✓ `mean_latency` < 5ms
- ✓ `max_latency` < 15ms
- ✓ `jitter` < 1ms

---

## 4. Phase 2: 실제 로봇 통신 (향후)

### 4.1 로봇 인터페이스 확장

```python
# controller.py (향후 Phase 2)
from utils.zmq_bridge import ZMQControllerBridge
from utils.robot_interface import DualTransportBridge

# 현재 (Phase 1)
zmq_bridge = ZMQControllerBridge()

# 향후 (Phase 2): 로봇 인터페이스 추가 후 DualTransportBridge로 확장
```

---

## 5. 코드 정리 요약

### 5.1 sim_runner.py 변경사항

| 항목 | 이전 (ROS2 지원) | 현재 (ZMQ-only) |
|------|-----------------|-----------------|
| `--use-ros2` flag | ✓ 있음 | ✗ 제거됨 |
| 브릿지 선택 | 동적 (runtime) | 고정 (ZMQ) |
| 임포트 | `get_transport_bridge()` | `ZMQSimBridge` |
| 파일 라인 수 | 157 | 150 (단순화) |

### 5.2 controller.py 준비

| 항목 | 현재 상태 |
|------|---------|
| **Python 버전** | Python 3.10 (conda 환경) |
| **ZMQ 지원** | ✓ 활성 (그대로) |
| **GUI** | ✓ PyQt5 + pyqtgraph (변경 없음) |
| **50Hz 벤치마킹** | ✓ benchmark.py 통합 가능 |

---

## 6. 트러블슈팅

### 문제 1: "ModuleNotFoundError: zmq"
```
sim_runner.py에서 zmq_bridge import 실패
→ Isaac Sim 환경에 zmq 설치 필요
```

**해결:**
```bash
./isaaclab.sh -i  # 개발 모드 설치 (zmq 포함)
```

### 문제 2: Controller GUI "Wait..." 상태 지속
```
Controller가 시뮬레이션 연결 못함
→ sim_runner가 시작되지 않았거나 포트 충돌
```

**해결:**
```bash
# 포트 확인
lsof -i :5555  # ZMQ setpoint
lsof -i :5556  # ZMQ state

# 기존 프로세스 종료
pkill -f sim_runner.py
pkill -f controller.py

# 다시 시작
```

### 문제 3: "Python 3.11 vs 3.10" 혼동
```
sim_runner.py를 Python 3.10 환경에서 실행 시도
→ Isaac Sim 패키지 호환성 오류
```

**확인:**
```bash
# Isaac Sim 환경 확인
./isaaclab.sh -p -c "import sys; print(sys.version)"

# Controller 환경 확인
conda activate isaac-r2s-py310
python -c "import sys; print(sys.version)"
```

---

## 7. 검증 체크리스트

```
[ ] Isaac Sim (Python 3.11) 환경 준비
    [ ] isaaclab.sh 실행 가능
    [ ] R2S 환경 레지스터됨

[ ] Python 3.10 conda 환경 생성
    [ ] PyQt5, pyqtgraph 설치

[ ] ZMQ 포트 확인
    [ ] 5555 (setpoint), 5556 (state) 사용 가능

[ ] 초기 실행
    [ ] sim_runner.py 시작 (ZMQ 대기)
    [ ] controller.py 시작 (GUI 표시)
    [ ] "Connected" 상태 도달

[ ] 슬라이더 테스트
    [ ] Slider 이동 → state 그래프 업데이트
    [ ] 에러 메시지 없음

[ ] 50Hz 벤치마킹
    [ ] 로그 생성 (logs/\*_benchmark.json)
    [ ] frequency ≥ 49.5 Hz
    [ ] latency < 5ms
```

---

## 참고

| 문서 | 내용 |
|------|------|
| CONDA_ENV_SETUP.md | Python 3.10 환경 설정 |
| utils/benchmark.py | 50Hz 성능 측정 API |
| utils/robot_interface.py | 로봇 통신 skeleton |
| utils/zmq_bridge.py | ZMQ IPC (기존) |

---

**지원:** gb.lee@rgarobot.com

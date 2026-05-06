# Python 3.10 Conda Environment Setup for Controller (ROS2)

## 개요
- **sim_runner.py**: Isaac Sim 환경 (Python 3.11, ZMQ-only)
- **controller.py**: 실시간 GUI 컨트롤러 (Python 3.10, ROS2 지원)

두 프로세스가 ZMQ IPC를 통해 통신하므로, 별도 Python 환경에서 각각 실행합니다.

---

## 1. Python 3.10 Conda 환경 생성

```bash
# 환경 생성
conda create -n isaac-r2s-py310 python=3.10 -y

# 활성화
conda activate isaac-r2s-py310
```

---

## 2. PyQt5 설치 (GUI 필수)

```bash
# PyQt5 및 pyqtgraph 설치
pip install PyQt5>=5.15 pyqtgraph>=0.13
```

**검증:**
```bash
python -c "from PyQt5 import QtWidgets; print('✓ PyQt5 OK')"
python -c "import pyqtgraph as pg; print('✓ pyqtgraph OK')"
```

---

## 3. NumPy 및 기본 라이브러리

```bash
pip install numpy torch  # torch는 선택사항 (현재 필요 없음)
```

---

## 4. ROS2 Humble 설치 (향후 로봇 통신용)

### 4.1 ROS2 공식 저장소 추가 (Ubuntu)

```bash
# 키 설정
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

# 저장소 추가
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# 업데이트
sudo apt update
```

### 4.2 ROS2 Humble 설치

```bash
# ROS2 Humble desktop 설치
sudo apt install ros-humble-desktop -y

# ROS2 환경 소싱
source /opt/ros/humble/setup.bash

# conda 환경과 호환성을 위해 .bashrc에 추가
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
```

---

## 5. Conda 환경에서 ROS2 사용 설정

ROS2의 C 확장(cp310 ABI)을 conda 환경에서 접근 가능하게 설정:

### 5.1 환경 활성화 후 ROS2 경로 추가

```bash
conda activate isaac-r2s-py310

# ROS2 경로를 Python path에 추가
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
```

### 5.2 자동화: conda activate hook 생성

```bash
# Conda 환경 디렉토리 확인
conda info --envs | grep isaac-r2s-py310

# 예: /home/lgb/miniconda3/envs/isaac-r2s-py310

# activate.d 디렉토리 생성
mkdir -p /home/lgb/miniconda3/envs/isaac-r2s-py310/etc/conda/activate.d

# 활성화 스크립트 생성
cat > /home/lgb/miniconda3/envs/isaac-r2s-py310/etc/conda/activate.d/ros2_setup.sh << 'EOF'
#!/bin/bash
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/ros/humble/lib:$LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
EOF

chmod +x /home/lgb/miniconda3/envs/isaac-r2s-py310/etc/conda/activate.d/ros2_setup.sh
```

이제 `conda activate isaac-r2s-py310`할 때마다 자동으로 ROS2 경로가 설정됩니다.

---

## 6. 검증

```bash
conda activate isaac-r2s-py310

# ROS2 확인
python -c "import rclpy; print('✓ ROS2 (rclpy) OK')"

# PyQt5 확인
python -c "from PyQt5 import QtWidgets; print('✓ PyQt5 OK')"

# 기본 임포트 확인
python -c "import numpy as np; print('✓ NumPy OK')"
```

---

## 7. Controller 실행 (Python 3.10 환경)

```bash
# 먼저 sim_runner를 Isaac Sim 환경에서 시작
# (별도 터미널)
cd /home/lgb/IsaacLab
./isaaclab.sh -p scripts/real2sim/sim_runner.py --num_envs 1

# 그 다음 controller를 Python 3.10 환경에서 실행
conda activate isaac-r2s-py310
cd /home/lgb/IsaacLab
python scripts/real2sim/controller.py
```

---

## 8. 트러블슈팅

### 문제 1: rclpy import 실패
**증상:** `ModuleNotFoundError: No module named 'rclpy'`

**해결:**
```bash
# 1) 환경 확인
conda activate isaac-r2s-py310
python -c "import sys; print(sys.path)"

# 2) PYTHONPATH 수동 설정
export PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:$PYTHONPATH

# 3) ROS2 환경 소싱 확인
source /opt/ros/humble/setup.bash
```

### 문제 2: C 확장 ABI 불일치
**증상:** `_rclpy_pybind11.cpython-310-*.so` 로드 실패

**원인:** ROS2 Humble이 Python 3.10으로 빌드되었으므로, conda 환경도 정확히 Python 3.10이어야 함.

**해결:**
```bash
# 정확한 Python 버전 확인
python --version  # 3.10.x 확인

# 버전 불일치 시 환경 다시 생성
conda env remove -n isaac-r2s-py310
conda create -n isaac-r2s-py310 python=3.10.13 -y  # 특정 3.10.x 버전 지정
```

### 문제 3: PyQt5 디스플레이 이슈
**증상:** `Could not connect to display` (원격 연결 시)

**해결:** 로컬 디스플레이 또는 X11 forwarding 설정 필요

```bash
# X11 forwarding 확인
echo $DISPLAY  # 일반적으로 :0 또는 :1

# 필요시 DISPLAY 수동 설정
export DISPLAY=:0
python scripts/real2sim/controller.py
```

---

## 9. 통신 검증

### 50Hz 동시 운영 테스트

controller와 sim_runner가 각각의 환경에서 실행 중일 때:

```bash
# sim_runner 터미널
[R2S] 시뮬레이션 시작. 컨트롤러: python scripts/real2sim/controller.py

# controller 터미널 (Python 3.10)
python scripts/real2sim/controller.py
# GUI 시작 → Sliders로 관절 제어 → Graph에서 state 수신 확인
```

**체크포인트:**
- ✓ GUI가 응답함 (슬라이더 움직임)
- ✓ 상태 그래프가 업데이트됨 (30Hz 갱신)
- ✓ 콘솔에 에러 메시지 없음

---

## 10. 주의사항

| 항목 | 주의 |
|------|------|
| **Python 버전** | controller는 **반드시** Python 3.10, sim_runner는 Python 3.11 |
| **ZMQ 포트** | setpoint: 5555, state: 5556 (충돌 주의) |
| **50Hz 타이밍** | sim_runner는 물리 시뮬레이션에서 보장, controller GUI는 30Hz |
| **ROS2 활성화** | 처음부터 모든 명령어 전에 `source /opt/ros/humble/setup.bash` 필요 |

---

## 11. 향후 로봇 통신 (Phase 2)

현재 skeleton 구조에서 ROS2를 이용한 실제 로봇 통신으로 확장:

```python
# controller.py (향후)
from rclpy.node import Node
from real2sim_msgs.srv import JointCommand

class R2SControllerROS(Node):
    def __init__(self):
        super().__init__('r2s_controller')
        # ROS2 서비스/토픽 설정
        # → 로봇 하드웨어와 ZMQ/ROS2 동시 운영
```

현재는 ZMQ만 사용하고, ROS2 임포트/통신 코드는 미포함. Python 3.10 환경이 준비되어 있으므로 필요시 즉시 추가 가능.

---

## 빠른 시작

```bash
# 1. 환경 생성 및 활성화
conda create -n isaac-r2s-py310 python=3.10 -y
conda activate isaac-r2s-py310

# 2. 라이브러리 설치
pip install PyQt5 pyqtgraph numpy

# 3. ROS2 경로 설정 (activate hook 구성 후)
source /opt/ros/humble/setup.bash

# 4. Controller 실행
python scripts/real2sim/controller.py
```

Done! ✓

# r2s_go2 환경 컨텍스트

## 개요

Unitree GO2 로봇을 IsaacLab 시뮬레이션에 올리고 ROS2 GUI 컨트롤러로
관절을 실시간 제어하는 **Real2Sim Milestone 1** 환경.

RL 없음. 순수 포지션 제어 테스트 전용. 고정 계약은 이 디렉토리의 `CONTRACT.md` 참고.

## 등록 정보

| 항목 | 값 |
|------|-----|
| 환경 ID | `Isaac-R2S-Go2-v0` |
| 환경 클래스 | `R2SGo2Env` |
| 설정 클래스 | `R2SGo2EnvCfg` |

## 조인트 파라미터

12관절 전부 `UNITREE_GO2_CFG`의 `DCMotorCfg base_legs` 공통 설정 사용 (CONTRACT §6).

| 항목 | 값 |
|------|-----|
| KP | 25.0 |
| KD | 0.5 |
| effort_limit / saturation_effort | 23.5 Nm |
| velocity_limit (v_max) | 30.0 rad/s |
| armature | 0.01 |

관절 순서는 `JOINT_ORDER`(`r2s_go2_env_cfg.py`)로 고정 — CONTRACT §2와 100% 일치해야 함.

## 시뮬레이션 파라미터

| 항목 | 값 |
|------|-----|
| Physics freq | 200 Hz |
| Control freq | 50 Hz (decimation=4) |
| Episode length | 600 s (10분, 조기 종료 없음) |
| Slew rate | v_max(30 rad/s) × (1/50) rad/step, 전 관절 동일 |

## 통신 구조 (Milestone 1)

```
gui_controller.py  →/lowcmd(ROS2)→  sim_bridge.py  →UDP:9871→  sim_runner_go2.py
                    ←/lowstate(ROS2)←              ←UDP:9872←
```

- sim 쪽(Isaac 3.12)은 UDP만 사용. rclpy/ros 임포트 절대 금지(§1 함정).
- UDP는 latest-wins: sim_runner의 recv는 non-blocking, 매 스텝 큐를 비우고 마지막 패킷만 반영.
- 명령/상태 패킷 스키마는 `CONTRACT.md` §4 참고.

## 실행 방법

```bash
# 터미널 1: 시뮬레이션
./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py --num_envs 1 [--fix_base]

# 터미널 2: ROS2 브릿지 (시스템 python3.10 + humble)
/usr/bin/python3 scripts/real2sim/r2s_go2/sim_bridge.py

# 터미널 3: GUI 컨트롤러 (시스템 python3.10 + humble)
/usr/bin/python3 scripts/real2sim/r2s_go2/gui_controller.py
```

## `--fix_base` 구현 메모

- `ArticulationRootBaseCfg.fix_root_link`(solver-common, `isaaclab.sim.schemas`)를
  `True`로 설정 → world와 base 사이에 `FixedJoint` 생성.
- `UNITREE_GO2_CFG`가 쓰는 (deprecated) `sim_utils.ArticulationRootPropertiesCfg`도
  이 필드를 상속하므로 로봇 cfg 교체 없이 `robot.spawn.articulation_props.fix_root_link`
  직접 대입으로 충분.
- `fix_base=True`일 때 spawn 높이를 z=0.5로 올림(다리 스윙 여유). `configclass`가
  필드 기본값을 인스턴스마다 deepcopy하므로 `__post_init__`에서 안전하게 mutate 가능
  (전역 `UNITREE_GO2_CFG` 오염 없음 — 헤드리스 검증으로 확인됨).
- 검증 결과: `fix_base=True`에서 20 step 이후 `root_pos_w[2] == 0.5` 그대로 유지,
  IMU quat=identity, gyro/acc=0(부동 상태이므로 정상).

## get_lowstate() IMU quat 변환 주의

- IsaacLab 6.0부터 `robot.data.root_quat_w`는 **(x, y, z, w)** 순서로 반환된다
  (5.1 이전엔 wxyz였음 — `[[project_parkour_imitation_quat_convention_bug]]`와 동일 계열 함정).
- Unitree `LowState.imu_state.quaternion`은 **(w, x, y, z)** 순서(CONTRACT §3).
- `get_lowstate()`는 `quat_xyzw[[3, 0, 1, 2]]`로 재배열해서 반환한다. 이 변환을
  건드릴 때는 반드시 quat norm≈1 + 정지 상태에서 quat_w≈1(upright) 검증할 것.
- base 선가속도는 `root_lin_acc_w` 같은 shorthand가 없어 `body_lin_acc_w[:, base_id, :]`
  (world frame)를 `find_bodies("base")`로 얻은 인덱스로 조회한 뒤
  `quat_apply_inverse`로 body frame으로 회전해서 사용한다.

## joint_acc (ddq) 데이터 소스

- `robot.data.joint_acc`가 실제로 존재함(fallback 불필요) — `base_articulation_data.py`에
  추상 property로 선언되어 있고 PhysX 백엔드(`isaaclab_physx`)가 워프 커널로 구현.
  헤드리스 실행에서 non-zero 값 확인됨.

## 불변 규칙

공통 불변 규칙: `.claude/rules/r2s.md`

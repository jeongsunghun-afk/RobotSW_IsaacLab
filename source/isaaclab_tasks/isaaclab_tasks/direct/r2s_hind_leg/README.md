# R2S HindLeg — Real2Sim (env 태스크)

> **⚠️ 이 환경은 강화학습(RL)이 아닙니다.** `agents/`가 없고 `train.py`/`play.py`로 실행하지
> 않습니다. 전용 도구(`scripts/real2sim/r2s_hind_leg/`)로 구동합니다.

R_Skeleton 뒷다리(고정베이스 5-DOF)를 Isaac Sim에 올리고, 순수 UDP로 PyQt5 GUI/monitor와
통신해 관절 위치·PD 응답을 실시간 제어/비교하는 Real2Sim 환경. r2s_go2 패턴 기반.

- 목적: 시뮬 vs (향후) 실로봇 응답 비교 / PD 파라미터 튜닝
- 제어: PD 위치 제어 + slew rate limiter + **faithful PD**(GUI kp/kd → sim drive 게인)
- 통신: 순수 UDP (ROS2는 향후, 메시지 형식 미정 — `CONTRACT.md` §3 seam)

## 등록 정보

| 항목 | 값 |
|---|---|
| Task ID | `Isaac-R2S-HindLeg-v0` |
| 환경 클래스 | `R2SHindLegEnv` |
| 설정 클래스 | `R2SHindLegEnvCfg` |

## env 인터페이스

- `set_setpoint(q, dq, kp, kd, tau)` — PD 목표 주입(q는 slew 통과, kp/kd는 faithful_pd=True면 반영).
- `get_lowstate()` — 5관절 `(q, dq, ddq, tau_est)` numpy. 고정베이스라 IMU 없음.

## 실행

```bash
# 터미널 1 (Isaac)
./isaaclab.sh -p scripts/real2sim/sim_runner_hindleg.py --num_envs 1 [--headless]
# 터미널 2 (시스템 python, GUI)
bash scripts/real2sim/r2s_hind_leg/run_gui_controller.sh
```

운영 상세: [`scripts/real2sim/r2s_hind_leg/README.md`](../../../../../scripts/real2sim/r2s_hind_leg/README.md).
고정 계약(조인트/패킷/게인): [`CONTRACT.md`](CONTRACT.md).

## 관절 파라미터 (실측 2026-07-13)

| 관절 | 실측명 | KP | KD | v_max [rad/s] | τ_max [Nm] | soft limit [rad] |
|---|---|---|---|---|---|---|
| thigh_r | HL_joint2_thigh_r | 300 | 5 | 41.0 | 22.0 | ±2.826 |
| thigh_p | HL_joint3_thigh_p | 300 | 5 | 25.0 | 53.0 | ±1.413 |
| knee_p | HL_joint4_knee_p | 300 | 5 | 25.0 | 53.0 | ±1.413 |
| ankle_p | HL_joint5_ankle_p | 100 | 5 | 51.0 | 48.0 | ±1.413 |
| toe_p | HL_joint6_toe_p | 100 | 5 | 51.0 | 48.0 | ±1.413 |

> 관절 인덱스는 `JOINT_NAME_PATTERNS`로 USD 로드 순서에 독립적으로 조회됩니다.
> soft limit은 실측 USD 값(`soft_joint_pos_limit_factor=0.9` 반영)입니다.

## 시뮬레이션 파라미터

| 항목 | 값 |
|---|---|
| Physics 주파수 | 200 Hz |
| 제어 주파수 | 50 Hz (decimation=4) |
| 에피소드 길이 | 600 s (조기 종료 없음) |
| Slew rate | `V_MAX_RAD[i] / 50` rad/step (관절별 상이) |

## 불변 규칙

- 새 버퍼는 `_reset_idx`에서 초기화 필수 (CLAUDE.md 전역 DO)
- slew rate limiter 필수 — KP=300에서 setpoint 점프 시 토크 스파이크 방지
- `JOINT_NAME_PATTERNS`로 `find_joints(preserve_order=True)` — USD 로드 순서 독립
- 게인/soft limit은 `scripts/real2sim/r2s_hind_leg/motions.py`와 값 일치 필수(중복 정의)
- 코어 파일(`source/isaaclab/`) 수정 금지

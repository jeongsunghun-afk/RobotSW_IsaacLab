# R_Skeleton 환경 컨텍스트

## 환경 개요
이족보행 로봇(R_Skeleton) 기본 보행 환경. Go2 WTW의 구조를 확장.
DOF 34, 발 4개 (사족보행 형태).

## 등록 ID & 파일

| ID | 클래스 | 설명 |
|----|--------|------|
| R_Skeleton-v0 | SkeletonEnv | 기본 (history 포함) |
| R_Skeleton-v1 | SkeletonEnv | fixed variant |
| R_Skeleton-WTW-v0 | SkeletonWTWEnv | Walk-the-walk |

- 설정: `skeleton_env_cfg.py`
- 구현: `skeleton_env.py`
- PPO 설정: `agents/rsl_rl_ppo_cfg.py`

## 로봇 스펙
- DOF: 34 (action_space = 34)
- 발 링크: `FL_link7_toe`, `FR_link7_toe`, `HL_link7_toe`, `HR_link7_toe`
- 종료 조건: base height < 0.3m
- 에피소드: 20초, decimation: 4 (50Hz 정책)
- action_scale: 0.25

## Observation 구성
Go2 대비 DOF 크기 변경에 주의:
```
projected_gravity_b(3) + commands(3) +
joint_pos_error(34) + joint_vel(34) + actions(34)
= 기본 108 (history 없을 때)
```

## 수정 시 주의사항
- DOF 34이므로 Go2(12) 코드를 복붙할 때 shape 불일치 주의
- `_setup_scene`에서 발 링크 이름 4개 확인: `FL/FR/HL/HR_link7_toe`
- R_Skeleton_amp로 확장 시 AMP obs 크기 재계산 필요

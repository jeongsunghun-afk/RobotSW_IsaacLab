# 2026-08-10_10-38-59_torque1e5_trotmirror_stand_dz_ds14

한 줄 목적: trot mirror 클립을 넣은 14 클립 데이터셋에서, **정지 리셋 + 명령 데드존 +
토크 페널티**를 동시에 켜고 저속(0.5 m/s) 달성률과 대칭 붕괴를 함께 노린다.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/leg_imitation_tracking_rma/2026-08-10_10-38-59_torque1e5_trotmirror_stand_dz_ds14`
- gym task id: `Leg-Imitation-Tracking-RMA-v0`
- rl_library: `rsl_rl`
- experiment_name: `leg_imitation_tracking_rma`
- 시작: 2026-08-10 10:38 · `--num_envs 4096 --max_iterations 50000` · GPU2
- 주요 설정 (기본값에서 바뀐 것만)

  | 항목 | 값 | 기본값 | 의도 |
  |---|---|---|---|
  | `motion_file` | `new_smr_leg_pkl` (**14 클립**) | `merged_leg_pkl` | run/trot/walk + **mirror 짝** |
  | `reset_strategy` | `random_stand` | `random` | `rel_stand_envs` 0.1 만큼 정지 리셋 |
  | `cmd_deadzone` | **0.1** m/s | 0.0 | `\|vx cmd\| ≤ 0.1` → 0 클램프 |
  | `torque_penalty_w` | **1e-5** | 0.0 | `reward −= w·Σ τ²` |

  `task_reward_lerp` 0.5, symmetry data-aug ON(mirror loss OFF) 은 기본값 그대로다.

  ⚠ **네 가지를 동시에 바꿨으므로 이 런 단독으로는 어느 것의 효과인지 분리되지 않는다.**

## 결과 요약

**진행 중 (11.7k / 50k, 2026-08-10 17:00 기준).** 아래는 스냅샷이고 판정이 아니다 —
이 task 의 이동 능력·대칭은 학습 지표로 판정되지 않으며 속도 램프 실측이 필요하다.

| 지표 | 2k | 6k | 10k | last (11.7k) |
|---|---:|---:|---:|---:|
| `lin_vel_reward` (ep sum) | 17.41 | 26.43 | 30.24 | 26.14 |
| `amp_reward` (ep sum) | 33.10 | 35.47 | 42.04 | 38.45 |
| `torque_penalty` (ep sum) | −1.13 | −1.26 | −1.43 | −1.31 |
| `mean_episode_length` | 478 | 455 | 499 | 434 |
| policy std | 0.211 | 0.150 | 0.144 | 0.143 |

`torque_penalty` 는 episode 합으로 −1.3 수준이라 `lin_vel_reward` 26~30 대비 **약 5%** 다 —
토크를 억제하되 추종을 압도하지는 않는 크기다. 램프는 아직 측정하지 않았다.

## 산출물

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_11500__leg_trotmirror_stand_dz_11k5__20260810-170555.mp4`](videos/model_11500__leg_trotmirror_stand_dz_11k5__20260810-170555.mp4) | `model_11500.pt` | `leg_trotmirror_stand_dz_11k5` | 2026-08-10 17:06:52 |
<!-- /report-video:videos -->

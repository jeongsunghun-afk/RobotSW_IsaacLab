# Go2 Imitation Tracking 환경 컨텍스트

## 개요

`go2_imitation`에서 파생된 환경으로, command 구조를 **steering(tar_dir·tar_speed·face_dir)**에서
**body-frame 속도추종(vx, vy=0, yaw_rate)**으로 교체한 버전이다.
AMP/discriminator/reset 로직은 go2_imitation과 동일. `motion_lib.py`는 go2_imitation 것을 베이스로
tracking 전용 메서드(`motion_names`, `motion_mean_speeds`, `motion_mean_yaw_rates`,
`sample_motions_near_speed`, `set_motion_weights_command_uniform` 등, 340-431줄)가 추가된
확장본(564줄, go2_imitation은 469줄)이며 byte-identical이 아니다.

**Task ID**: `Go2-Imitation-Tracking-v0`

---

## 파일 구조

```
go2_imitation_tracking/
├── __init__.py                         ← task 등록 (Go2-Imitation-Tracking-v0)
├── go2_imitation_tracking_env.py       ← 환경 본체
├── go2_imitation_tracking_env_cfg.py   ← 환경 config
├── motion_lib.py                       ← go2_imitation 기반 + tracking 전용 메서드 추가 (564줄)
├── agents/
│   ├── __init__.py
│   └── rsl_rl_ppo_cfg.py              ← PPO/AMP 러너 config
└── imitation/
    └── smr_mirror_pkl/                 ← go2_imitation/imitation/smr_mirror_pkl 의 실제 디렉토리 복사본 (심링크 아님)
```

---

## go2_imitation과의 차이점

| 항목 | go2_imitation | go2_imitation_tracking |
|------|---------------|------------------------|
| command 타입 | steering (tar_dir + tar_speed + face_dir) | body-frame 속도 (vx, vy=0, yaw_rate) |
| obs command block | `local_tar_dir(2) + tar_speed(1) + local_face_dir(2)` = 5 | `lin_vel_cmd(2) + yaw_vel_cmd(1)` = 3 |
| obs 총 차원 (policy) | 50 | **42** |
| command 변환 | `quat_apply` (heading relative) | 변환 없음 (이미 body-relative) |
| reward | tar_reward + face_reward | lin_vel_reward + yaw_vel_reward |
| AMP 경로 | 동일 | 동일 (불변) |

---

## Observation (RMA dict obs — 2026-07 개편)

배포 가능성을 위해 단일 48-dim 텐서에서 dict obs로 바뀌었다. `root_lin_vel_b`는 실기
GO2에서 측정할 수 없으므로 policy obs에서 빠지고 estimator가 추정한다.

**policy (42-dim)** — priv_explicit로 분리한 root 속도 항은 제외. `_apply_obs_dr`로 노이즈가 실린다.

| idx | 성분 | dim |
|-----|------|-----|
| 0–2 | `projected_gravity_b` | 3 |
| 3–4 | `lin_vel_cmd` (vx, vy) | 2 |
| 5 | `yaw_vel_cmd` | 1 |
| 6–17 | `joint_pos - default` | 12 |
| 18–29 | `joint_vel` | 12 |
| 30–41 | `actions` | 12 |
| **합계** | | **42** |

- `actions` 블록(30–41)은 학습 내내 리터럴 0인 dead channel이다. 배포 시에도 0.0 고정할 것. (2026-09-03 코드 대조 시 미확인)
- **`joint_pos_tan_norm=True`** 면 관절 블록이 12 → **72**(관절별 회전의 tan-norm 6D)로 늘어
  policy가 **102**, history가 (10, 102)가 된다. MimicKit actor proprio(117) 중 96이 이 표현이라
  대조하려고 만든 플래그다(`_comparisons/mimickit_vs_60_actuator_limit/README.md` §16).
  - `observation_space`는 cfg 상수가 아니라 **env `__init__`이 `super().__init__` 전에** 다시
    계산한다 — hydra 오버라이드가 확정되는 시점이 거기다.
  - `_apply_obs_dr`의 인덱스를 **하드코딩하지 말 것.** `self._obs_idx_joint_vel`에서 유도한다.
  - 관절각 노이즈(`joint_pos_noise`)와 `encoder_bias`는 `_apply_obs_dr`이 아니라
    **`_get_observations`에서 라디안 공간에** 더한다. tan-norm 출력에 라디안을 더할 수 없다.
    raw 경로에서는 인코딩이 항등이라 예전 동작과 bit-identical이다.
  - Go2 축(hip=x, thigh/calf=y) 때문에 72 중 **32가 θ와 무관한 상수**다(실측: 상수 차원
    raw 13 → tan-norm 45). `EmpiricalNormalization`이 `std + 1e-2`로 나눠 NaN은 아니고 0이 된다.
    정보량은 관절당 (cos θ, sin θ)뿐이라 **차원 맞춘 대조군은 cos/sin 24**다.
- **2026-07-30 변경**: `root_ang_vel_b`가 policy obs에서 제거되어 인덱스가 3씩 앞으로 밀렸다.
  `_apply_obs_dr`의 노이즈 슬라이스도 함께 이동했고, `dr.ang_vel_noise`는 주입할 자리가
  없어져 **dead config**가 되었다(되살릴 때 인덱스만 밀어 남기면 σ=0.2가 σ=0.05인
  `projected_gravity_b`에 들어가 4배 증폭되므로 주의).

**priv_explicit (6-dim)** — 노이즈 없는 GT. critic 입력이자 estimator target.

| idx | 성분 | 비고 |
|-----|------|------|
| 0–2 | `root_lin_vel_b × priv_explicit_lin_vel_scale` | 실측 불가 |
| 3–5 | `root_ang_vel_b × priv_explicit_ang_vel_scale` | 설계 일관성을 위해 obs에서 제외 |

- 둘 다 policy obs에 없으므로 estimator 과제는 **미관측 상태 추정**이다(denoising 아님).
  평균예측 MSE 기준선: lin 0.381 / ang 0.0201 (`s²·Var(target)`, 2026-07-30 실측). (2026-09-03 코드 대조 시 미확인)

- 두 scale은 actor/critic 입력에선 normalizer를 지나므로, 실질 역할은 estimator MSE에서의
  **블록 간 상대 gradient 가중치**다. `Loss/estimator_lin` vs `Loss/estimator_ang`로 확인한다.

**priv_latent (19-dim)** — domain-rand 파라미터 (armature/friction/mass/foot_fric/kp/kd/action_delay + encoder_bias 12).

**history (10 × 42)** — policy proprio 링버퍼 (노이즈 포함).

- command 변환(quat_apply) 제거. 버퍼를 그대로 concat.
- AMP observation (490-dim = 49×10 history) 은 go2_imitation과 동일, 절대 수정 금지.

---

## Command 버퍼

```python
self._lin_vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, vy) — vy 항상 0
self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)     # yaw rate [rad/s]
self._tar_timer   = torch.zeros(self.num_envs, device=self.device)     # 재샘플링 타이머
```

범위 (cfg 참조):
- `lin_vel_x`: [0.0, 4.0] m/s
- `lin_vel_y`: [0.0, 0.0] (항상 0)
- `yaw_vel`: [-1.0, 1.0] rad/s
- `tar_timer`: [4.0, 7.0] s마다 재샘플링 (2026-09-03 코드 대조 시 미확인 — `_post_physics_step`이 훅으로 자동 호출되지 않아 한때 dead 였던 경위가 있음)

---

## Reward 함수

```python
# lin_vel tracking (body frame 직접 비교)
lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]
lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
lin_vel_reward = torch.exp(-cfg.vel_err_scale * lin_vel_err)

# yaw_vel tracking
yaw_vel_b = self._robot.data.root_ang_vel_b[:, 2]
yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
yaw_vel_reward = torch.exp(-cfg.yaw_vel_err_scale * yaw_vel_err)

reward = cfg.lin_vel_reward_w * lin_vel_reward + cfg.yaw_vel_reward_w * yaw_vel_reward
# 기본: 0.7 * lin_vel_reward + 0.3 * yaw_vel_reward
```

---

## 알고리즘 설정

go2_imitation과 동일 (AMP dict / 네트워크 / 하이퍼파라미터 값 동일).

---

## 학습 실행

학습·렌더 실행 방법은 `.claude/rules/training.md`.

---

## 수정 가이드

| 수정 항목 | 파일 | 주의사항 |
|-----------|------|---------|
| reward 가중치 | `go2_imitation_tracking_env_cfg.py` | `lin_vel_reward_w + yaw_vel_reward_w` 합계 권장 = 1.0 |
| 명령 범위 | `go2_imitation_tracking_env_cfg.py` | `lin_vel_x_min/max`, `yaw_vel_min/max` |
| Termination 임계값 | `go2_imitation_tracking_env_cfg.py` | go2_imitation과 동일 구조 |
| AMP anneal 일정 | `agents/rsl_rl_ppo_cfg.py` | go2_imitation과 동일 |
| 새 텐서 추가 | `go2_imitation_tracking_env.py` | `_reset_idx`에서 반드시 초기화 |
| motion 데이터 | `imitation/smr_mirror_pkl/*.pkl` | go2_imitation과 별도의 실제 디렉토리 복사본 (심링크 아님) — 수정 시 두 경로 모두 갱신 필요 |

## 절대 수정 금지

- AMP observation 관련 코드 — 490-dim (49×10) 불변

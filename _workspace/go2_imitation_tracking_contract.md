# go2_imitation_tracking — 구현 계약서 (CONTRACT)

> 모든 worker는 이 문서를 **단일 진실의 원천(SSOT)** 으로 따른다. 임의로 이름/차원/값을 바꾸지 말 것.
> 충돌 시 team-lead에게 SendMessage로 확인.

## 0. 목표 (한 줄)
`go2_imitation`을 복제해 `go2_imitation_tracking` 새 환경을 만든다.
command 구조를 **steering(tar_dir·tar_speed·face_dir)** → **body-frame 속도추종(vx, vy=0, yaw_rate)** 으로 교체한다.

## 1. 설계 결정 (확정 — 변경 금지)
- **Frame**: body/heading frame. lin_vel 명령 = `(vx, vy)`, 단 **vy는 항상 0** (vy range = (0.0, 0.0)).
- **yaw_velocity**: angular velocity 명령 (rad/s), `root_ang_vel_b[:, 2]`로 추종.
- **obs 차원**: 기존 50 → **48** (command block: 5 → 3).
  - 기존 command block: `local_tar_dir(2) + tar_speed(1) + local_face_dir(2)` = 5
  - 신규 command block: `lin_vel_cmd(2) + yaw_vel_cmd(1)` = 3  ← vy(=0) 포함해서 2
- **명령은 이미 body-relative** 이므로 obs에 넣을 때 `_calc_heading_quat_inv`/`quat_apply` 변환 **제거**, 버퍼를 그대로 obs에 삽입.
- **AMP 경로 불변**: `amp_observation_space=490`, AMP/discriminator/motion/reset 로직 전부 byte-identical 유지.
- **WASABI 변형 제외**: 새 환경에서 WASABI cfg/등록 복제하지 않는다.

## 2. 명명 규약 (확정)
| 항목 | 값 |
|------|----|
| 디렉토리 | `source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_tracking/` |
| env 파일 | `go2_imitation_tracking_env.py` |
| env 클래스 | `Go2ImitationTrackingEnv` |
| cfg 파일 | `go2_imitation_tracking_env_cfg.py` |
| cfg 클래스 | `Go2ImitationTrackingEnvCfg` |
| runner cfg 파일 | `agents/rsl_rl_ppo_cfg.py` |
| runner cfg 클래스 | `Go2ImitationTrackingPPORunnerCfg` |
| gym id | `Go2-Imitation-Tracking-v0` |
| experiment_name | `go2_imitation_tracking` |

## 3. command 버퍼 (env.py)
기존 `_tar_dir[N,2]`, `_tar_speed[N]`, `_face_dir[N,2]`, `_tar_timer[N]` 를 **다음으로 교체**:
```python
self._lin_vel_cmd = torch.zeros(self.num_envs, 2, device=self.device)  # (vx, vy) — vy 항상 0
self._yaw_vel_cmd = torch.zeros(self.num_envs, device=self.device)     # yaw rate [rad/s]
self._tar_timer   = torch.zeros(self.num_envs, device=self.device)     # 재샘플링 타이머 (유지)
```
- **재샘플링 함수** (`_resample_steering` 자리, 이름 유지 가능):
  ```python
  self._lin_vel_cmd[env_ids, 0] = U(cfg.lin_vel_x_min, cfg.lin_vel_x_max)  # vx
  self._lin_vel_cmd[env_ids, 1] = U(cfg.lin_vel_y_min, cfg.lin_vel_y_max)  # vy → (0,0)이므로 항상 0
  self._yaw_vel_cmd[env_ids]    = U(cfg.yaw_vel_min, cfg.yaw_vel_max)
  self._tar_timer[env_ids]      = U(cfg.tar_change_time_min, cfg.tar_change_time_max)
  ```
  (U = uniform). 타이머/주기 재샘플링 로직과 `_reset_idx` 호출은 기존 구조 그대로 유지.
- **`_reset_idx`** 에서 위 3개 버퍼 반드시 초기화(재샘플 호출로 충족). (프로젝트 불변규칙)

## 4. observation 레이아웃 (env.py `_get_observations`) — 총 48
| idx | 성분 | dim |
|-----|------|-----|
| 0–2 | `root_lin_vel_b` | 3 |
| 3–5 | `root_ang_vel_b` | 3 |
| 6–8 | `projected_gravity_b` | 3 |
| **9–10** | **`self._lin_vel_cmd`** (vx, vy) | **2** |
| **11** | **`self._yaw_vel_cmd`** (unsqueeze) | **1** |
| 12–23 | `joint_pos - default` | 12 |
| 24–35 | `joint_vel` | 12 |
| 36–47 | `actions` | 12 |
| **합계** | | **48** |
- command 변환(quat_apply) 제거. 버퍼 그대로 concat.
- AMP observation 조립부는 **절대 건드리지 않는다**.

## 5. reward (env.py `_get_rewards`) — 2개 항
```python
# (1) lin_vel tracking — body frame 직접 비교
lin_vel_b = self._robot.data.root_lin_vel_b[:, :2]          # (vx, vy) actual
lin_vel_err = torch.sum((self._lin_vel_cmd - lin_vel_b) ** 2, dim=-1)
lin_vel_reward = torch.exp(-cfg.vel_err_scale * lin_vel_err)
# proj_speed>=0 backward-gating 제거 (squared error가 처리)

# (2) yaw_vel tracking
yaw_vel_b = self._robot.data.root_ang_vel_b[:, 2]
yaw_vel_err = (self._yaw_vel_cmd - yaw_vel_b) ** 2
yaw_vel_reward = torch.exp(-cfg.yaw_vel_err_scale * yaw_vel_err)

reward = cfg.lin_vel_reward_w * lin_vel_reward + cfg.yaw_vel_reward_w * yaw_vel_reward
```
- `_episode_sums` / 로깅 키: `face_reward` → `yaw_vel_reward`, `tar_reward` → `lin_vel_reward` (기존 키 이름 추적해 일관 교체).
- `_calc_heading_quat`(forward 방향 계산)는 reward에서 더 이상 불필요 → reward 경로에서 사용 제거. (단 obs용 `_calc_heading_quat_inv`도 command엔 불필요해짐. helper 함수 정의 자체는 남겨도 무방.)

## 6. cfg 파라미터 (env_cfg.py) — 기존 steering 필드 교체
```python
observation_space: int = 3 + 3 + 3 + 2 + 1 + 12 + 12 + 12  # = 48  (명시적 합으로 작성)
# ... AMP/motion/episode/sim 필드는 전부 그대로 유지 ...

# ── 속도추종 command 범위 ──
lin_vel_x_min: float = -1.0   # vx 최소 (m/s)  ※ 합리적 기본값
lin_vel_x_max: float = 3.0    # vx 최대 (m/s)
lin_vel_y_min: float = 0.0    # vy 항상 0
lin_vel_y_max: float = 0.0    # vy 항상 0
yaw_vel_min: float   = -1.5   # yaw rate 최소 (rad/s)
yaw_vel_max: float   = 1.5    # yaw rate 최대 (rad/s)
tar_change_time_min: float = 4.0  # (유지)
tar_change_time_max: float = 7.0  # (유지)

# ── reward 가중치 ──
lin_vel_reward_w: float = 0.7
yaw_vel_reward_w: float = 0.3
vel_err_scale: float = 0.5        # lin_vel reward 지수 스케일 (유지)
yaw_vel_err_scale: float = 0.5    # yaw_vel reward 지수 스케일 (신규)
```
- 기존 `tar_speed_min/max`, `tar_reward_w`, `face_reward_w` 는 **삭제**.
- docstring의 observation 설명도 48 레이아웃으로 갱신.
- `motion_file = MOTION_FILES_DIR` 유지. `MOTION_FILES_DIR`는 `_THIS_DIR/imitation/smr_mirror_pkl` ( `__file__` 상대 → 심링크된 새 디렉토리 자동 인식).

## 7. agents/rsl_rl_ppo_cfg.py
- `Go2ImitationPPORunnerCfg` 를 그대로 복제하되 클래스명 `Go2ImitationTrackingPPORunnerCfg`, `experiment_name="go2_imitation_tracking"`.
- AMP dict / 네트워크 / 하이퍼파라미터 **값 동일 유지**.
- **WASABI cfg 클래스는 복제하지 않는다.**

## 8. __init__.py (등록)
```python
import gymnasium as gym
from . import agents

gym.register(
    id="Go2-Imitation-Tracking-v0",
    entry_point=f"{__name__}.go2_imitation_tracking_env:Go2ImitationTrackingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_imitation_tracking_env_cfg:Go2ImitationTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ImitationTrackingPPORunnerCfg",
    },
)
```
- WASABI 등록 제외. 상위 `isaaclab_tasks/__init__.py`는 자동 import이므로 수정 불필요 (blacklist에 없음).

## 9. motion_lib.py & 데이터
- `motion_lib.py` 는 `go2_imitation/motion_lib.py` 를 **byte-identical 복사** (수정 금지).
- imitation 데이터: **전체 트리 복사 금지** (비디오 포함 거대). `imitation/smr_mirror_pkl` **서브디렉토리만 심링크**:
  ```bash
  mkdir -p <new>/imitation
  ln -s ../../go2_imitation/imitation/smr_mirror_pkl <new>/imitation/smr_mirror_pkl
  ```
  (상대 심링크 경로 정확성 검증할 것. `<new>/imitation/smr_mirror_pkl/`에서 .pkl이 보여야 함.)

## 10. 최종 검증 게이트 (lead가 수행)
- `observation_space(48) == _get_observations concat 실제 크기` 일치.
- import 시 gym id 등록 성공.
- 새 버퍼 `_reset_idx` 초기화 확인.

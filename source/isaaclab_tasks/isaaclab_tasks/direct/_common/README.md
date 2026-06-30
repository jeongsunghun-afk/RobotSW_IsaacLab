# `_common` — 공유 인프라

IsaacLab direct RL 환경 전반에서 공통으로 사용하는 helper 모음.
`source/isaaclab/` 코어를 수정하지 않고 composition 방식으로 기능을 추가한다.

---

## DebugViewer

viewer/키보드/디버그 시각화를 env에 1줄로 붙이는 composition helper.

### 임포트

```python
from isaaclab_tasks.direct._common import DebugViewer, DebugViewerCfg, DebugKeyBindingCfg
```

---

## 최소 통합 패턴

```python
class MyEnv(DirectRLEnv):
    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # (1) viewer 활성화 — 기본 cfg로 충분
        self._debug_viewer = DebugViewer(self, cfg=DebugViewerCfg())

    def _get_observations(self):
        obs = {...}
        # (2) step 끝에 update (free-fly 카메라 적분, 키 이벤트 처리)
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
            self._debug_viewer.update(self.step_dt)
        return obs

    def __del__(self):
        # (3) cleanup — idempotent, 중복 호출 안전
        if hasattr(self, "_debug_viewer") and self._debug_viewer is not None:
            self._debug_viewer.close()
```

headless 모드나 `viewport_camera_controller is None`이면 `DebugViewer.__init__` 에서 자동으로 비활성화되므로 if-guard 불필요.

---

## DebugViewerCfg 필드

| 필드 | 타입 | 기본값 | 설명 |
|------|------|--------|------|
| `fly_speed` | `float` | `5.0` | free-fly 카메라 이동 속도 (m/s) |
| `fly_speed_boost` | `float` | `3.0` | LEFT_SHIFT 홀드 시 속도 배율 |
| `key_bindings` | `DebugKeyBindingCfg` | (아래 참조) | 내장 키 매핑 커스터마이징 |

### DebugKeyBindingCfg 필드

| 필드 | 기본 키 | 역할 |
|------|---------|------|
| `toggle_free_fly` | `F` | free-fly ↔ tracking 카메라 전환 |
| `move_forward` | `W` | free-fly 앞 이동 |
| `move_backward` | `S` | free-fly 뒤 이동 |
| `move_left` | `A` | free-fly 왼쪽 이동 |
| `move_right` | `D` | free-fly 오른쪽 이동 |
| `move_up` | `E` | free-fly 위쪽 이동 |
| `move_down` | `Q` | free-fly 아래쪽 이동 |
| `env_index_prev` | `[` | 이전 env_index로 전환 |
| `env_index_next` | `]` | 다음 env_index로 전환 |
| `speed_boost` | `LEFT_SHIFT` | 홀드 시 이동 속도 부스트 |

방향키(`UP`/`DOWN`/`LEFT`/`RIGHT`)는 `W`/`S`/`A`/`D`와 동일하게 동작.

---

## task-specific 키 등록

```python
self._debug_viewer.register_key(
    name="reset_env",
    on_press=self._manual_reset,   # 키 눌릴 때 1회 호출
    on_release=None,               # 키 떼어질 때 (선택)
    on_repeat=None,                # 홀드 중 매 frame (선택)
)
```

- `name`은 사람이 읽을 수 있는 레이블 (로그/debug용)
- 내장 키와 동일한 키를 등록하면 내장 동작에 추가로 콜백이 실행됨

---

## task-specific 시각화 등록

`sensor_base.set_debug_vis` 컨벤션과 동일한 on/off 인터페이스.

```python
# 등록
self._debug_viewer.register_debug_vis(
    name="terrain_edges",
    callback=self._draw_terrain_edges,   # enabled=True일 때 매 update() 호출
    default_on=False,
)

# 런타임 토글
self._debug_viewer.set_debug_vis("terrain_edges", True)

# 현재 상태 쿼리
if self._debug_viewer.vis_is_enabled("terrain_edges"):
    ...
```

---

## 카메라 follow 콜백과 충돌 방지

env가 자체 카메라 tracking 콜백(예: robot-yaw 추적)을 등록하는 경우,
free-fly 모드에서 충돌을 막으려면 콜백 첫 줄에 가드를 추가:

```python
def _camera_follow_callback(self, event):
    if self._debug_viewer.is_free_fly_camera:
        return  # free-fly 중에는 tracking 중단
    # ... 기존 카메라 이동 로직
```

---

## property 요약

| property | 타입 | 설명 |
|----------|------|------|
| `is_free_fly_camera` | `bool` | 현재 free-fly 모드 여부 |
| `env_index` | `int` | 현재 추적 중인 env 인덱스 |
| `is_active` | `bool` | viewer가 활성 상태인지 (headless면 False) |

---

## FAQ

**Q. headless 학습 중에 DebugViewer를 생성해도 괜찮나?**
A. 괜찮다. `viewport_camera_controller is None` 이면 `is_active=False`가 되어 키/시각화 등록은 모두 no-op이다.

**Q. `update()`를 `_get_observations` 말고 다른 곳에서 호출해도 되나?**
A. 가능하다. `_get_observations` 또는 `_post_physics_step` 중 env의 step 흐름상 마지막 지점에서 1회 호출하면 된다. 2회 이상 호출은 카메라가 double-integrate되므로 피할 것.

**Q. `DebugViewerCfg`를 env_cfg에 필드로 넣어도 되나?**
A. 권장한다. env_cfg에 `debug_viewer: DebugViewerCfg = DebugViewerCfg()` 필드를 추가하면 cfg-worker가 키/속도를 task별로 조정할 수 있다.

```python
# env_cfg.py
@configclass
class MyEnvCfg(DirectRLEnvCfg):
    debug_viewer: DebugViewerCfg = DebugViewerCfg()

# env.py
self._debug_viewer = DebugViewer(self, cfg=self.cfg.debug_viewer)
```

---

## 참고 구현

- `source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env.py`
  — DebugViewer + parkour-specific 카메라 yaw tracking + terrain 시각화 결합 사례

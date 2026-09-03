---
paths:
  - "source/isaaclab_tasks/isaaclab_tasks/direct/r2s_*/**"
---

# r2s_* 공통 불변 규칙

- 새 버퍼 추가 시 `_reset_idx`에서 초기화 필수.
- Slew rate limiter 필수 — setpoint 점프 시 토크 스파이크 방지.
- 관절 인덱스는 `JOINT_NAME_PATTERNS`/`JOINT_ORDER`로 `find_joints(preserve_order=True)` 사용 —
  USD 로드 순서 독립.
- `from isaaclab.utils.configclass import configclass` 사용 — `from isaaclab.utils import configclass`는
  6.0에서 `TypeError: 'module' object is not callable`.
- 코어 파일(`source/isaaclab/`) 수정 금지.

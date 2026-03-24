# Findings

## contact_sensor 데이터 구조
- `contact_sensor.data.net_forces_w`: shape `(num_envs, num_bodies, 3)` (world frame)
- `contact_sensor.data.net_forces_w_history`: shape `(num_envs, history_length, num_bodies, 3)`
- body 이름 접근: `robot.data.body_names` (list[str])
- contact_sensor 등록 경로: `prim_path="/World/envs/env_.*/Robot/.*"` → 모든 링크 포함

## 현재 report_data_recorder.py 수집 항목
- joint torque, position, velocity
- processed actions
- base linear velocity, angular velocity
- **contact force: 미수집**

## 환경별 contact_sensor 접근
- `env.unwrapped.contact_sensor` 속성으로 접근
- `go2_interaction_env.py`, `skeleton_amp_env.py` 모두 동일 패턴

## IsaacLab VisualizationMarkers
- import: `from isaaclab.markers import VisualizationMarkers, SPHERE_MARKER_CFG`
- `go2_interaction_env.py`에서 이미 import하여 사용 중
- marker 색상: `VisualizationMarkersCfg`에서 `ColorType.RGBA` 지정
- 위치 업데이트: `markers.visualize(translations=positions, scales=scales)`

## body names 패턴 (R_Skeleton 기준)
- 발(toe): `FL_link7_toe`, `FR_link7_toe`, `HL_link7_toe`, `HR_link7_toe`
- 기타 링크: base, neck, shoulder, thigh, knee, calf 등

## Plot 저장 구조 (현재)
```
results/{task}/{run}/report_{ts}/
  env0_{label}/
    robot_data.csv
    rms_summary.csv
    joint_torques_{left,right,waist_neck}.png
    joint_positions_*.png
    joint_velocities_*.png
    processed_actions_*.png
    base_velocity.png
```
→ 새로 추가: `contact_forces.png`, `contact_forces.csv`

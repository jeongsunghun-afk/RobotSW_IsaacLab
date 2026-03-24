# Task Plan: Collision Force Visualization in report.py

## Goal
`report.py` 실행 중 self-collision 및 외부 collision force를 수집/시각화한다.

## Context
- `ContactSensor.data.net_forces_w`: shape `(num_envs, num_bodies, 3)` — 각 body에 가해지는 world-frame 힘
- `skeleton_amp_env.py`에서 이미 `contact_sensor`를 사용하지만, `report_data_recorder.py`는 이 데이터를 수집/저장하지 않음
- 시각화 방향: (A) 오프라인 Plot 저장 + (B) IsaacLab VisualizationMarkers 실시간 시각화

## Phases

### Phase 1: 데이터 구조 파악 [COMPLETE]
- contact_sensor 데이터 구조 확인
- report_data_recorder.py 현재 수집 항목 확인
- skeleton_amp_env.py의 body_name 접근 패턴 확인

### Phase 2: report_data_recorder.py 에 collision force 수집 추가 [TODO]
- `record()` 메서드에서 `contact_sensor` 데이터 수집
- 버퍼 추가: `_contact_forces` (per body, per env)
- body 이름 저장
- `save()` / `_save_csv()` 에 collision force 데이터 포함

### Phase 3: Collision Force Plot 시각화 [TODO]
- 각 env별 contact force magnitude plot (body별)
- 발(foot) vs 비발 부위 분리 표시
- 임계값 초과 강조 표시 (red zone)

### Phase 4: report.py 에 실시간 VisualizationMarkers 추가 [TODO]
- 각 body position에 sphere marker 오버레이
- contact force 크기에 따라 색상/크기 변화
- IsaacLab VisualizationMarkers API 활용

## Decisions
- `contact_sensor`가 없는 env에 대해 graceful skip
- force magnitude = norm of (x,y,z) per body
- threshold: 기본 1.0 N (기존 코드와 동일)

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| - | - | - |

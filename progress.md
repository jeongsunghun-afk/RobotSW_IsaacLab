# Progress

## Session 1 — 2026-03-23

### Completed
- [x] Phase 1: 데이터 구조 파악
  - contact_sensor.data.net_forces_w shape 확인
  - report_data_recorder.py 현재 구조 확인
  - 실시간 마커 API (VisualizationMarkers) 확인

### Completed
- [x] Phase 2: report_data_recorder.py — contact_forces 버퍼, record(), save(), _save_csv(), _plot_contact_forces() 추가
- [x] Phase 3: Collision Force Plot — body별 force magnitude, 발/비발 색상 구분, threshold 초과 배경 강조
- [x] Phase 4: report.py — VisualizationMarkers 실시간 sphere 마커 (force>1N body에 빨간 구체)

### Notes
- 기존 코드 패턴 유지하며 최소 변경
- contact_sensor 없는 env → graceful skip

# Progress Log

## 2026-03-26

### Session 4 - Deploy 시 발 착지 충격(Soft Landing) 계획 수립
- [x] go2_wtw_env.py 전체 코드 분석 (보상 구조, Bezier 궤적, contact shaping)
- [x] go2_env_cfg.py 설정 파악 (reward scales, gait params)
- [x] 발 충격 원인 3가지 도출 (c2 z값 급하강, 착지 z속도 패널티 부재, impact force 미반영)
- [x] task_plan.md 새 계획으로 업데이트 (방안 A~D 우선순위 포함)
- [x] findings.md에 soft landing 원인 분석 및 코드 스니펫 추가
- [x] 방안 A 구현: go2_wtw_env.py L610 c2 z값 수정 (height_cmd * 0.1)
- [x] 방안 B 구현: foot_landing_vel 보상 추가 (go2_wtw_env.py + go2_env_cfg.py)

---

### Session 3 - Bezier Swing Start 편향 분석 및 수정 계획
- [x] go2_wtw_env.py Bezier c0/c3 계산 코드 분석 (line 537~618)
- [x] swing start(c0) 편향 메커니즘 파악
- [x] 수정안 (c0 = hip - x_corr) 도출 및 수치 검증
- [x] task_plan.md, findings.md 업데이트
- [ ] go2_wtw_env.py c0 계산 수정 구현 (사용자 승인 후)

---

## 2026-03-25

### Session 2 - Go2 AMP Environment 구현 계획 수립
- [x] R_Skeleton_amp 전체 구조 분석 (skeleton_amp_env.py, env_cfg.py, motion_loader.py)
- [x] Go2 기존 환경 분석 (go2_env_cfg.py, UNITREE_GO2_CFG)
- [x] stmr_go2.py 분석 (joint names, body names, 출력 포맷)
- [x] task_plan.md, findings.md 작성
- [x] stmr_go2.py 실제 출력 포맷 확인 → 61개 값/프레임
- [x] Go2 body 이름 확인 → "base" 확인
- [x] go2_motion_loader.py 구현
- [x] go2_amp_env_cfg.py 구현
- [x] go2_amp_env.py 구현
- [x] __init__.py 업데이트 ("Go2-AMP-v0" 등록)
- [x] agents/rsl_rl_ppo_cfg.py 업데이트 (Go2AmpPPORunnerCfg 추가)

### Session 1 - Bezier swing_start_pos 문제 해결
- [x] go2_wtw_env.py Bezier 관련 코드 전체 분석
- [x] 문제 메커니즘 파악
- [x] 해결 방안 5가지 도출 (findings.md)
- [x] 사용자와 솔루션 선택 논의 → Body Frame 기반 방안 선택
- [x] 구현 완료

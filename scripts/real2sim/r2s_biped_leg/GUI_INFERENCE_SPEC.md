# GUI 직접 inference 통합 SPEC (executor 작업 지시서)

## 목표
`gui_controller.py`가 **deployable jit 모델을 직접 로드해 policy mode inference를 수행**하도록 수정한다.
현재는 별도 `policy_runner_bipedleg.py`가 추론하지만, 이를 gui 내부로 흡수한다. 목적: 실제 로봇 배포와
같은 self-contained 모델을 gui에서 구동해 sim/real input·action을 비교.

## 확정된 사실 (검증 완료 — 재검증 불필요)
- **deployable model**: `<run>/exported/deployable_policy.pt` (torch.jit.script). 생성:
  `python scripts/real2sim/export_deployable_bipedleg.py --run_dir <run> --checkpoint model_23100.pt`.
  인터페이스: `model(proprio(B,34), history(B,10,34)) → action(B,8)`. estimator/history_encoder 내장.
  bit-exact 검증됨(policy_runner 방식과 diff 0).
- **obs/lockstep 로직은 `scripts/real2sim/policy_runner_bipedleg.py`에 검증된 채로 존재** — 이식 소스로 사용.

## ⚠ 환경 (2026-07-24 정정 — 사용자 제약)
gui는 **conda가 아니라 시스템 python3(`/usr/bin/python3`, py3.10)에서 실행**한다. 이유: 추후 ros2로 gui를
실행해야 하고, ros2 gui는 r2s_go2 패턴상 `/opt/ros/humble` + 시스템 python3(conda 완전 비활성화 — conda가
시스템 python3를 shadow하면 rclpy가 깨짐)에서 돈다. 따라서:
- **run_gui_controller.sh는 기존대로 conda 비활성화 + /usr/bin/python3 유지** (conda isaac-6.0 전환 금지).
- **torch는 시스템 python3에 2.10.0+cu128(py3.10 wheel, --user)로 설치** — lead가 설치 진행. 워커는
  `import torch`가 시스템 python3에서 된다는 전제로 코드 작성. conda에 아무것도 설치하지 말 것.
- matplotlib/monitor/position 호환은 **시스템 python3** 기준(기존 gui가 이미 시스템 python3라 그대로 동작).

## 반드시 재사용할 것 (policy_runner_bipedleg.py에서 그대로 이식)
1. **PolicyState 클래스** (obs 구성): gravity3+cmd3+(q−default)8+dq8+prev_action8+clock4=34, **articulation 순서**.
   clock φ+=0.02/0.6/step, HL=φ/HR=φ+0.5. history 첫스텝 10×복제 후 roll. **한 글자도 바꾸지 말 것** (bit-parity).
   단, deployable jit는 obs["priv_explicit"]가 불필요(estimator 내장) → jit 호출은 `model(proprio, history)`만.
2. **요청-응답 lockstep**: action 송신 → sim state blocking 대기 → inference → 다음 action. `1 action = 1 sim step`.
   free-running/자유타이머 금지(로봇 넘어짐). policy_runner의 `recv_state_blocking`/`send_action` 패턴 이식.
3. **UDP 계약**: `r2s_udp.py` 그대로. 포트 gui↔sim: POLICY_ACT(9886 송신), POLICY_STATE(9885 수신).
   real: REAL_ACT(9887), REAL_STATE(9888). target=0.25·action+default(0).
4. **시동/교착 방지**: run 진입 시 default target 송신(sim이 회신주소 학습). state 없으면 default 재송신.

## 핵심 설계 — Qt 이벤트 루프 + lockstep
gui는 Qt 이벤트 루프라 blocking recv를 메인 스레드에서 하면 UI가 얼어붙는다.
→ **policy inference를 `QThread`에서 돌린다**. 스레드가 lockstep 루프(blocking recv), 결과는
Qt signal로 메인 스레드에 전달해 상태바/monitor 갱신. mode/source/x_vel/yaw는 스레드가 공유 변수로 읽음.
스레드는 mode=policy & Run일 때만 활성, Stop/Position이면 정지.

## gui 수정 항목
1. **torch/jit lazy import**: policy mode 진입 시에만 `import torch`. position mode는 torch 없이 동작해야
   하지만 gui가 conda python이면 torch가 있으므로 무해. import는 policy 스레드 시작 시.
2. **모델 경로 인자/기본값**: `--model <path>` (기본 `<기본 run>/exported/deployable_policy.pt`). 없으면 policy UI 비활성 + 안내.
3. **PolicyInferenceThread(QThread)**: PolicyState + deployable jit + lockstep 루프. sim/real state 수신, inference, fan-out.
   grav_z 등 상태를 signal로 방출.
4. **기존 policy UI(Mode/Run/Stop/source/x_vel/yaw)와 연결**: 기존 `_on_policy_*` 핸들러가 스레드 공유 변수 갱신.
   `_on_publish_tick`의 policy 분기(현재 POLICY_CMD 송신)는 제거 — 이제 gui가 직접 추론하므로 POLICY_CMD 불필요.
5. **run_gui_controller.sh는 시스템 python3(/usr/bin/python3) 유지** (conda 전환 금지 — 위 "환경" 참고).
   torch는 시스템 python3에 lead가 설치. position/monitor는 기존대로 시스템 python3에서 동작(변경 불필요).
   policy mode용 인자(--model 등)만 추가.

## 비교 기능 (이번 스코프 = 최소)
- gui inference 스레드가 매 step의 (input=state q/gravity, action=target)을 기존 monitor로 relay(action vs sim).
- **sim/real dual 비교(양쪽 state를 각각 추론해 나란히 비교)는 후속** — 이번엔 gui 직접 inference 성립까지.
  real은 seam(--real_host 미지정 시 no-op) 유지.

## 검증 (executor가 수행)
1. `export_deployable_bipedleg.py`로 모델 생성 확인.
2. 통합 스모크: `scratchpad/smoke_policy_integration.py` 패턴 참조하되 **policy_runner 대신 gui inference 스레드**를
   검증. 헤드리스로는 gui 없이 스레드 로직만 떼어내 sim_runner(--policy_mode)와 폐루프 돌려 grav_z 확인 가능.
   **PASS 기준: 정지·전진 구간 grav_z ≈ -0.97~-0.99 (직립 유지)**. policy_runner 스모크와 동일 기준.
3. gui를 conda로 실제 띄워 position mode(관절 스텝)와 policy mode(Run) 둘 다 동작 확인.

## 참조 파일
- `scripts/real2sim/policy_runner_bipedleg.py` — obs/lockstep 이식 소스 (**핵심**)
- `scripts/real2sim/export_deployable_bipedleg.py` — 모델 입출력 계약
- `scripts/real2sim/r2s_biped_leg/POLICY_MODE_SPEC.md` — 전체 아키텍처/obs parity
- `scripts/real2sim/r2s_biped_leg/r2s_udp.py` — UDP 패킷 (수정 금지)
- `/tmp/.../scratchpad/smoke_policy_integration.py` — 통합 스모크 패턴

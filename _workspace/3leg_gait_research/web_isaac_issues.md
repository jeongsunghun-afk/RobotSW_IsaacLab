# Web 조사: Isaac Sim / Isaac Lab 엔진·시뮬레이터 차원의 3-leg gait / foot stuck / dragging 이슈

> 조사자: worker-web-isaac · 작성일: 2026-06-02
> 목적: Go2 parkour 3-leg gait(지형 극복 시 한 다리를 잘 안 쓰는 보행)가 **시뮬레이터/엔진(PhysX, contact, articulation)** 차원의 알려진 이슈에서 기인할 수 있는지 확인.

---

## ★ 핵심 결론 (먼저 읽을 것)

- **Isaac Sim 5.1 specific 한, "quadruped가 다리 3개로 걷는다"를 직접 유발하는 확정된 엔진 버그는 발견되지 않음.**
  - 5.1 release notes / known issues / GitHub issues 어디에도 "quadruped 3-leg gait를 일으키는 PhysX/contact 버그"라고 명시된 항목 없음.
  - 5.1 known issues 중 articulation 관련 항목은 **gripper(평행 그리퍼) 링크가 안 움직임**뿐이며, legged/feet 관련 항목 없음.
- 다만 **간접적으로 비대칭/이상 보행에 기여할 수 있는 엔진 레벨 이슈 후보 2개**는 실재함 (아래 HIGH/MEDIUM 항목). 특히 **GPU + triangle-mesh terrain에서 contact force 누락** 이슈는 parkour rough terrain의 gait reward 신뢰도에 영향을 줄 수 있어 주목할 만함.
- confirmation bias 방지를 위해: 아래 케이스들은 "3-leg gait의 원인이다"가 아니라 "엔진 차원에서 비대칭 보행을 만들 *수 있는* 메커니즘 후보"로만 제시함. 대부분은 학습(RL) artifact이거나 사용자 설정 오류로 판명되었음.

---

## HIGH 관련도

### 1. GPU + triangle-mesh terrain에서 contact force 누락/축소 (엔진 레벨, 5.1 specific 아님)
- 출처:
  - legged_gym README (원조 보고): https://github.com/leggedrobotics/legged_gym
  - Isaac Lab Discussion #2697 "Is the Legged Gym issue with force sensor still there?": https://github.com/isaac-sim/IsaacLab/discussions/2697
- 요지:
  - `net_contact_force_tensor`로 보고되는 contact force가 **GPU + triangle mesh terrain**에서 신뢰 불가. triangle edge와의 상호작용으로 **일부 timestep에서 force가 크게 줄거나 아예 누락**됨.
  - 과거 workaround는 발끝(feet/end-effector)에만 force sensor 부착. 단 NVIDIA는 **force sensor를 deprecate**하고, 대신 `ArticulationData.body_incoming_joint_wrench_b`(link incoming joint force, load-cell 방식)로 대체함. ContactSensor도 권장 경로.
- Go2 parkour 3-leg gait 관련성: **높음**
  - parkour는 정확히 **triangle-mesh rough/parkour terrain** 위에서 학습됨. 만약 발-지형 contact가 간헐적으로 누락되면 **contact 기반 gait reward(feet_air_time, contact, foot clearance 등)의 신호가 다리별로 비대칭하게 왜곡**될 수 있고, 이는 정책이 "신호가 깨끗한 다리"만 쓰는 비대칭/3-leg 보행으로 수렴하는 메커니즘이 될 수 있음.
  - **검증 필요/주의**: 본 프로젝트는 height_scan이 정상 검증됨, contact sensor obs 추가는 sim-to-real 이유로 금지됨(MEMORY). 이 항목은 *obs*가 아니라 *reward 신호의 다리별 신뢰도* 문제로 봐야 함. code-worker가 parkour의 contact 측정 방식(ContactSensor vs net_contact_force)과 terrain collider 타입(trimesh vs heightfield)을 실제로 확인해 줘야 결론 가능. (가설로만 표기)

---

## MEDIUM 관련도

### 2. Isaac Sim 5.1.0 "G1 한쪽 다리가 매우 높이 접혀 스킵하듯 걸음" (#4014)
- 출처: https://github.com/isaac-sim/IsaacLab/issues/4014
- 요지:
  - `Velocity-G1-Flat-v0` RSL_RL 기본 학습을 10K~50K iter 돌리면 **한쪽 다리가 매우 높이 접혀(folded high) 스킵(skipping)하는 단일-다리 이상 보행** 발생.
  - 환경: **Isaac Sim 5.1.0**, Ubuntu 22.04, L40, CUDA 12.4, 4096 envs.
  - 상태: **open, 라벨/담당자/maintainer 응답 없음, RCA 미규명.**
- 관련성: **중간**
  - "Isaac Sim 5.1 기본 학습에서 단일-다리 이상 보행이 실제로 관측된다"는 직접적 data point. 단 **(a) G1 = 휴머노이드(quadruped 아님), (b) 엔진 버그로 확정된 바 없고 RL artifact일 가능성이 큼.** 따라서 "5.1 엔진 때문이다"의 근거로 단정 금지. "default training에서도 single-leg 이상 보행이 흔히 나온다"는 정황 증거로만 사용.

### 3. Go2 sim2real: 발 끌기(dragging) / 발을 충분히 안 듦 — 바닥 friction 모델링 (#1784)
- 출처: https://github.com/isaac-sim/IsaacLab/issues/1784
- 요지:
  - 정책이 sim에서 **발을 끌거나(dragging) 충분히 들지 않고** 걷는 것을 학습 → 실기 전이 실패.
  - 원인으로 **시뮬레이터 바닥 friction 모델링 부족**(발을 끌어도 이동 가능)을 지목. **foot static/dynamic friction randomization 시도했으나 효과 없었음.**
- 관련성: **중간**
  - "발 끌기/안 듦"은 3-leg gait와 인접한 증상이나, 본 케이스는 **대칭적 dragging**이지 한 다리만 안 쓰는 비대칭은 아님. friction이 다리별로 다르게 randomize되면 비대칭 유발 가능성은 있으나 본 issue엔 그 언급 없음. friction/마찰 모델은 5.0/5.1에서 변경됨(아래 5번 참조)이라 교차 확인 가치 있음.

---

## LOW 관련도 (참고/배제용)

### 4. "legged robot 비정상 joint 움직임" 포럼 — 사용자 설정 오류로 판명
- 출처: https://forums.developer.nvidia.com/t/the-issue-of-abnormal-joint-movement-in-the-legged-robot-in-isaac-lab/317165
- 요지: hip joint target 설정 시 다리가 몸을 끌고 가는 이상 동작 → **원인은 매 프레임 `write_joint_state_to_sim()` 호출하며 `reset()` 누락한 control loop 설정 오류.** `reset()` 한 번 추가로 해결.
- 관련성: **낮음** — 엔진 버그 아님, RL 학습 환경과도 무관. 배제.

### 5. Isaac Sim 5.0/5.1 physics 변경점 (release notes) — 직접 3-leg 유발 근거 없음
- 출처:
  - Isaac Sim 5.1 release notes: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html
  - Isaac Sim 5.1 known issues: https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html
  - Isaac Lab release notes: https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html
- 발견된 physics 변경점(참고용, 모두 3-leg와 직접 연결 안 됨):
  - **Isaac Sim 5.0: 새 actuator 모델 (drive model + friction model 개선)** — 일부 state-based env에서 ~20% slowdown. → per-joint dynamics가 미묘하게 바뀔 여지는 있으나 비대칭 보행과의 인과 근거 없음(추정 Tier 3).
  - **Isaac Lab v2.2.0: PhysX 최신 API 기반 joint friction modeling** 도입.
  - **Isaac Sim 5.1: articulation collision contact를 마지막에 푸는 옵션** 추가 — 명시적으로 **gripper penetration 개선용**. legged와 무관.
  - 5.1 known issues의 articulation 항목: **gripper 평행기구 링크가 안 움직임**뿐. quadruped/feet 항목 **없음**.
  - ContactSensor 기능 확장: friction force reporting(v2.3.2), contact point location, force_matrix_w_history 등 추가됨.
- 관련성: **낮음** — "5.1에서 quadruped 3-leg를 일으킨다"는 release note/known issue 근거는 **없음**.

### 6. solver iteration count — 일반 안정화 파라미터 (배경지식)
- 출처: https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html
- 요지: `solver_position_iteration_count`(통상 8) / `solver_velocity_iteration_count`(0~1)는 articulation 단위로 설정(per-link 불가). 진동/불안정 시 position iteration 증가가 일반 트러블슈팅. legged 통상값 확인용 배경지식.
- 관련성: **낮음** — 직접 3-leg 유발 근거 없음.

### 7. [OVPHYSX] rough terrain locomotion 지원 (#5321) — 신규 backend 통합 작업
- 출처: https://github.com/isaac-sim/IsaacLab/issues/5321
- 요지: OVPhysX backend에서 rough terrain locomotion 검증 작업(draft, 2026-04-20). "**ContactSensor가 critical dependency — 없으면 gait reward용 foot contact를 못 잡는다**"고 명시. 단 이는 신규 backend 통합 이슈이지 기존 gait 비대칭 버그가 아님.
- 관련성: **낮음(정황)** — "foot contact 감지가 gait reward의 핵심 의존성"이라는 점을 NVIDIA도 인정한다는 보조 근거. 1번 항목을 뒷받침.

---

## 종합 판단 & code-worker에게 넘길 포인트

1. **Isaac Sim 5.1 specific 3-leg gait 엔진 이슈는 발견 안 됨.** (명시적으로 보고)
2. 엔진 차원에서 비대칭/3-leg 보행에 *기여할 수 있는* 가장 유력한 후보는 **#1 (GPU + triangle-mesh terrain의 contact force 누락)** — parkour terrain collider가 trimesh이고 gait reward가 net_contact_force_tensor 기반이면 다리별 reward 신호 왜곡 가능. → **code-worker가 parkour의 (a) terrain collider 타입, (b) contact 측정 경로(ContactSensor vs net_contact_force vs body_incoming_joint_wrench_b)를 확인** 필요. (가설, 미검증)
3. #4014는 "5.1 default 학습에서 single-leg 이상 보행이 실제로 흔하다"는 정황 증거이나 humanoid + RL artifact 가능성 → **엔진 원인으로 단정 금지.**
4. friction 모델은 5.0/5.1에서 변경되었고(#5), Go2 dragging(#1784)과도 연관 → reward/friction worker가 교차 확인할 가치 있음(단, 대칭 dragging이라 3-leg 직접 근거는 약함).

---

## Sources
- https://github.com/isaac-sim/IsaacLab/issues/4014 (Isaac Sim 5.1.0 G1 single-leg folded/skipping)
- https://github.com/isaac-sim/IsaacLab/issues/1784 (Go2 sim2real feet dragging / friction)
- https://github.com/isaac-sim/IsaacLab/discussions/2697 (GPU trimesh contact force unreliability; force sensor deprecated → body_incoming_joint_wrench_b)
- https://github.com/leggedrobotics/legged_gym (원조 contact force 누락 보고)
- https://github.com/isaac-sim/IsaacLab/issues/5321 (OVPHYSX rough terrain; ContactSensor가 gait reward critical dependency)
- https://forums.developer.nvidia.com/t/the-issue-of-abnormal-joint-movement-in-the-legged-robot-in-isaac-lab/317165 (사용자 설정 오류, 배제)
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/release_notes.html
- https://docs.isaacsim.omniverse.nvidia.com/5.1.0/overview/known_issues.html
- https://isaac-sim.github.io/IsaacLab/main/source/refs/release_notes.html
- https://isaac-sim.github.io/IsaacLab/main/source/api/lab/isaaclab.sim.schemas.html

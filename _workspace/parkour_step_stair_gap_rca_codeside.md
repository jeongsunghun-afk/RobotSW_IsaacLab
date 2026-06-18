# Parkour Step/Stair/Gap RCA — Code & Config Side

분석 범위: 최근 두 커밋(`c3a7c4395af`, `2aab9ee224b`) + uncommitted 변경(`parkour_env_cfg.py`,
`source/isaaclab/.../mesh_terrains_cfg.py`)이 step/stair/gap 학습 실패를 일으키는 메커니즘을
코드/cfg 표면에서 분석. 학습 로그는 미참조(코드 기반 RCA). 모든 결론은 evidence 강도 라벨링.

증거 위치
- `parkour_env.py:883–1078` — `_get_rewards()` 전체
- `parkour_env.py:1080–1103` — `_get_dones()` (termination)
- `parkour_env.py:580–622` — height_scan & yaw_diff 캐시(+ `recently_reset` 게이트)
- `parkour_env.py:986–1018` — feet_stumble / feet_edge contact 추출
- `parkour_env.py:1024–1031` — feet_dragging 항
- `parkour_env_cfg.py:396–450` — `_actuator_mode = 2` (uncommitted)
- `parkour_env_cfg.py:474–496` — reward_scales (`feet_dragging -0.1` uncommitted)
- `parkour_env_cfg.py:466–471` — command 범위 vx∈[0.3,1.0]
- `parkour_env_cfg.py:44–132` — PARKOUR_TERRAINS_CFG (step/stair/gap height/length 범위)
- `source/isaaclab/.../mesh_terrains_cfg.py:345` — `y_offset_range` (uncommitted, gap 한정)

---

## 1. 변경 사항 요약 표

지형별 영향 (직접영향 = 학습 봉쇄 가능 / 간접영향 = 정책 mode 왜곡 / 도움 = 학습을 쉽게 함)

| # | 변경 | 상태 | step (0.10–0.45 m) | stair (0.05–0.20 m × width 0.25–0.40) | gap (0.05–0.5 m) | flat | hurdle (≤0.30 m) |
|---|------|------|--------------------|----------------------------------------|------------------|------|-------------------|
| A | actuator stiffness 40→25, damping 1.0→0.5, saturation_effort 35→23.5, effort_limit hip/thigh/calf {35,40,40}→{23.5,23.5,23.5}, velocity_limit 52.4/30.1→30/30, **+armature 0.01** | uncommitted (`_actuator_mode=2`) | **직접 (강)** — high steps (≥0.25 m) push-off 토크 부족 가능성 | **직접 (강)** — 좁은 step width(0.25 m) × 다단 등반 시 빠른 회복 토크 부족 | **직접 (강)** — gap 0.4–0.5 m 도약 시 calf/thigh peak 토크가 한계 근접/초과 | tolerable (low-torque envelope) | tolerable (hurdle ≤0.30 m, 1회성 jump) |
| B | `tracking_yaw`에서 `moving_mask`(horizontal speed gating) 제거. 정지 상태에서도 yaw reward = `exp(-|Δyaw|)` 발생 | committed (c3a7c4395af) | **간접** — A로 정지하면 yaw만 정렬 → free reward → standstill trap | **간접** — 동일 | **간접** — 동일, 점프 commit이 가장 위험한 비결정 행동이므로 회피 incentive 강함 | 영향 미미 (전진 쉬움) | 영향 미미 |
| C | `feet_dragging` reward scale 0.0 → **-0.1** | uncommitted | 작음 (간접) — 등반 도중 발 미끄러짐 시 페널티, 그러나 magnitude는 작음(아래 §2-H4 정량) | 작음 (간접) | 매우 작음 — 점프 중에는 contact=False이므로 firing 거의 없음 | 미미 | 미미 |
| D | feet_edge/stumble contact 입력: `max over history` → `net_contact_forces[:,0]` (latest substep), `contact_filt = curr OR last` | committed (2aab9ee224b) | 도움 (약) — feet_stumble은 selective해짐, 등반 시 false-positive 감소 | 도움 (약) — 동일 | 도움 (약) — gap edge 짧은 접촉도 `contact_filt` 2-sample OR로 검출 유지 | 도움 | 도움 |
| E | `_scan`/`_yaw_diff`/`_next_yaw_diff` zero-alloc + `recently_reset` per-env 패치 | committed (2aab9ee224b) | 거의 영향 없음 — 모든 지형 동일 적용. 어려운 지형에서 reset 빈도 증가하지만 한 step만 stale | 동일 | 동일 | 동일 | 동일 |
| F | `next_goal_threshold` 0.1 → 0.2 | uncommitted | 도움 — 완화 | 도움 | 도움 (gap 후 다음 platform에서 더 일찍 goal 인정) | 도움 | 도움 |
| G | `debug_vis_edge_mask` True → False | uncommitted | perf 개선 외 영향 없음 | 동일 | 동일 | 동일 | 동일 |
| H | `MeshParkourGapTerrainCfg.y_offset_range` (-1.2,1.2) → (-0.4,0.4) | uncommitted (`source/isaaclab/.../mesh_terrains_cfg.py:345`) | 무관 | 무관 | 도움 — gap platform의 lateral offset 축소 → 횡방향 정렬 부담 감소 | 무관 | 무관 |

핵심 split: A는 step/stair/gap에서 **직접영향**, flat/hurdle에서는 **tolerable**. 이것이
{학습 실패, 학습 진행} 구분을 가장 잘 설명함. B는 A에 conditional하게만 supplementary 효과.

---

## 2. 가설 (evidence 강도 + 메커니즘 + 검증 방법)

### H1. Actuator 약화(A)가 step/stair/gap 등반에 필요한 peak 토크/속도 envelope을 침범 — **강한 가설 (primary)**

Evidence
- `parkour_env_cfg.py:425–432`: effort_limit이 hip/thigh/calf 모두 **23.5 N·m**로 평탄화됨. 이전(_actuator_mode=1)은 hip=35, thigh=40, calf=40 — 즉 추진 관절(thigh/calf) effort_limit이 **40→23.5 (41% 감소)**.
- `saturation_effort=23.5`(DCMotor의 토크-속도 곡선상 stall torque) → 고속 회전 중 가용 토크는 추가로 떨어짐 (velocity_limit=30 rad/s에서 0).
- `stiffness=25`, `damping=0.5`로 PD가 모두 약화. 1/2 수준. action_scale=0.25(`parkour_env_cfg.py:319`)와 결합되어 목표 관절 변위 추종 강도(Kp·Δq) 자체가 절반 이하로 약해짐.
- `armature=0.01` 추가는 반사 관성 → 빠른 가속 시 effective inertia 증가, 응답 둔감화(보조 효과, 미검증).

대략적 정량 추정 (Go2 mass ≈ 15 kg, 다리당 분담 ≈ 3.7 kg)
- step_height=0.45 m 등반 시 push-off 단계에서 calf joint peak 토크 요구량은 Genesis 보고치/일반 사족 동역학에서 25–40 N·m 범위가 흔함. **23.5 N·m saturation은 이 요구량 하단/중앙 침범**.
- gap_length=0.5 m 도약(체공 시간 ~0.3 s, 도약 속도 vx≈1.0 m/s 가정 시 takeoff 수직 임펄스 m·v_z ≈ 15·1.5 = 22.5 N·s)을 0.1 s 안에 만들려면 다리 전체에서 ≈225 N 추력 → calf 토크 환산 시 saturation 한계 근접/초과.
- 반면 hurdle_height ≤0.30 m + flat은 등반 없이 1회성 jump 혹은 거의 정상 보행 envelope → tolerable.

메커니즘 (왜 학습이 봉쇄되는가)
1) 정책이 step push-off 시도 → 토크 saturation으로 충분한 상승 속도 미발생 → 발이 step에 걸려 termination(base contact / tilt / low_height) → 음수 reward(`termination=-100`) 누적.
2) 반복적 termination → curriculum 강등 (`_terrain_levels` -= 1, `parkour_env.py:1221`) → 더 쉬운 row로 회귀 → 그러나 step/stair 컬럼 자체는 sub_terrain proportion(0.2/0.2/0.3)이 고정이라 column이 바뀌지 않음, 즉 같은 지형 클래스 안에서 difficulty만 0–최저로 머무름.
3) Level 0의 step/stair/gap도 saturation에 닿으면(0.10 m step도 1대 다리만 사용 시 임계 근접) standstill 수렴.

검증 (가장 작은 단일 실험)
- `parkour_env_cfg.py:399`에서 `_actuator_mode = 2` → `_actuator_mode = 1`로만 변경. **다른 모든 uncommitted 변경(C feet_dragging, F threshold, H y_offset)은 유지**.
- 동일 step 수(짧게 — 1k–2k iter)로 학습, per-class curriculum metric(`Episode_Reward/curriculum/mean_terrain_level_step|stair|gap`)이 회복하면 H1 확정.
- 부수 metric: applied_torque saturation% (calf joint), feet_air_time, base_z trajectory.

---

### H2. `moving_mask` 제거(B)가 H1과 결합되어 standstill+yaw-aligned local optimum을 강화 — **강한 가설 (H1 conditional)**

Evidence
- `parkour_env.py:912–918`:
  ```python
  tracking_yaw = torch.exp(-|yaw_diff|)
  ```
  속도 게이팅 없음. weight=0.5.
- `tracking_goal_vel`(weight=1.5)은 `min(proj_forward, commanded_speed)/commanded_speed`로 정지 시 0이지만, 정지가 강요되는 환경에서는 두 항을 모두 0/max로 비교하면 yaw 항만 free.
- step/stair/gap 앞 standstill: tracking_goal_vel=0, tracking_yaw=exp(0)=1.0 (정면 정렬 시), 페널티는 hip_pos/dof_error_l2(작음)/torques_l2(거의 0)만 발생. **종합 reward가 음수보다 0 근처 작은 양수에 머무름**, 시도→termination(-100)보다 유리.

메커니즘
- H1이 step/stair/gap 등반을 거의 봉쇄 → 시도하면 termination, 시도 안 하면 yaw reward만 받음 → expected-return greedy 정책은 시도 회피로 수렴.
- flat/hurdle에서는 정상 전진이 saturation 안에서 가능 → tracking_goal_vel이 항상 ≥0.7 수준의 신호를 주므로 yaw-only 정책이 dominate되지 않음. **이것이 학습 split을 가속함**.

조건성
- H2는 H1이 참일 때만 충분조건. 액추에이터가 강하면 1.5:0.5 비율이 standstill trap을 충분히 막음 — advisor 검토와 일치.

검증
- H1 검증 실험에서 함께 회복하면 H2도 동시 해결. 별도 분리 실험 불필요.

---

### H3. `feet_dragging` -0.1(C)이 step/stair 등반 도중 미끄러짐을 가중 페널티 — **약한 가설**

Evidence
- `parkour_env.py:1024–1031`:
  ```python
  feet_dragging = Σ feet_speed · contact_filt · (feet_speed > 0.05) [/leg]
  ```
- 등반 중 발이 step edge에서 미끄러지면 contact=True + feet_speed>0.05 → 발화.
- 그러나 magnitude 산정:
  - per-step contribution = `0.1 · step_dt · Σ feet_speed`
  - step_dt = 0.02 (decimation=4, physics 200 Hz)
  - 4발 모두 0.3 m/s 드래깅 가정 → `0.1 · 0.02 · 1.2 = 0.0024` / step
  - tracking_goal_vel max 기여(weight=1.5, value=1) `= 1.5·0.02·1.0 = 0.030` / step
  - **드래깅 페널티는 tracking_goal_vel의 ~8%** — gait quality 조정에 충분하지만 학습 자체를 봉쇄할 magnitude는 아님.

조건성
- 단독으로는 학습을 차단하지 않음. H1+H2로 robot이 step 앞에서 yaw만 정렬하는 standstill에 빠질 때, 발이 살짝 미끄러지면 dragging 페널티가 추가로 음수를 만들 수 있으나 standstill 자체의 expected value가 여전히 시도(-100)보다 큼.

검증
- H1 확정 후, gait 품질이 어색하면(예: 잠금-끌기 패턴) 0.0 또는 -0.05로 완화 검토. 단독 실험 불필요.

---

### H4. Contact gating `[:, 0]` 전환(D)이 step/gap edge에서 feet_edge/feet_stumble 누락 — **약한 가설 (실제로 도움 방향 가능성)**

Evidence
- `parkour_env.py:998–1001`:
  ```python
  contact = ||F[:, 0, feet]|| > 2.0          # latest substep만
  contact_filt = contact OR self._last_contacts  # 2-sample 정책-step OR
  ```
- 기존 `max over history`는 substep 어디서든 임계 초과면 True → 짧은 false-positive 빈발.
- 신 코드: latest substep만 검사하지만, `contact_filt = curr OR last_policy_step`이 이미 50 Hz 정책-step 간 OR debouncing 제공.
- 따라서 짧은 step edge 충격이 누락될 확률은 substep 단위가 아니라 policy-step 단위에서 평가됨 — Isaac에서는 policy-step(20 ms) 안에 발이 step edge에 접촉하면 latest substep 한 번이라도 임계 초과할 확률이 높음.
- feet_stumble은 selective해진 결과 등반 시 false-positive(짧은 lateral force 스파이크)가 줄어 **오히려 학습에 도움**일 가능성.

판정
- step/stair/gap 학습 봉쇄 가설로는 약함. 만약 도움 방향이면 H1 해결 후 자연스럽게 정상 작동.

검증
- 별도 실험 불필요. H1 해결 시 episode_reward/feet_stumble 및 feet_edge 메트릭이 정상 음수 범위로 안정화되는지 확인.

---

### H5. `recently_reset` 게이트(E)가 어려운 지형에서 노이즈 유발 — **매우 약한 가설**

Evidence
- `parkour_env.py:593`: `recently_reset = (episode_length_buf <= 1)`. reset 직후 한 policy-step만 per-env 패치.
- 패치 자체는 fresh sensor data 기반 (`scan_new = ...` 전체 envs 계산 후 인덱싱), stale 데이터 주입 아님.
- 어려운 지형(자주 termination)에서 reset 빈도 ↑ → 더 자주 fresh scan 갱신 → 오히려 cadence가 효과적으로 빨라짐. 부정적 메커니즘 미관찰.

판정
- step/stair/gap에 특이적 영향 없음. 모든 클래스에 동일 적용되므로 split 설명력 0.

---

## 3. 우선순위

### 1순위: **H1 (actuator 약화)** — primary blocker
- `_actuator_mode = 2`(uncommitted)가 step/stair/gap에 필요한 peak 토크 envelope을 침범.
- {flat, hurdle ≤0.30 m}와 {step ≥0.25 m, stair 다단, gap ≥0.4 m} 사이의 학습 split을 단독으로 가장 잘 설명.
- H2/H3는 H1에 conditional / 보조적.

### 2순위: **H2 (moving_mask 제거)** — H1과 결합되었을 때만 활성
- H1 단독 해결로 자동 완화 가능. 만약 H1 복원 후에도 standstill trap이 잔존하면 별도 조치.

### 3순위: **H3 (feet_dragging -0.1)** — gait 품질 측면 효과만, 학습 봉쇄 아님
- H1 해결 후 정성적 gait 관찰이 어색하면 추후 조정.

### 검증 실험 (가장 작게 1개)

```text
실험 X1: actuator 단독 복원
  변경: source/isaaclab_tasks/isaaclab_tasks/direct/parkour/parkour_env_cfg.py:399
        `_actuator_mode = 2` → `_actuator_mode = 1`
  유지: feet_dragging=-0.1, next_goal_threshold=0.2, y_offset_range=(-0.4,0.4) 등
        다른 uncommitted 변경 전부 유지 (confound 회피 — H1만 isolate)
  학습: ~1k–2k PPO iter, num_envs=4096, seed 1개
  관측 메트릭 (`Episode_Reward/...`, `curriculum/...`):
    - curriculum/mean_terrain_level_step
    - curriculum/mean_terrain_level_stair
    - curriculum/mean_terrain_level_gap
    - curriculum/mean_terrain_level_flat  (기준선 — 회귀 없어야 함)
    - curriculum/mean_terrain_level_hurdle (기준선)
    - Episode_Termination/cause_base_contact, cause_tilt, cause_low_height
  성공 기준:
    step/stair/gap mean_terrain_level이 기존 학습 대비 의미 있게 상승 (예: 0→2 이상)
    AND flat/hurdle 회귀 없음
  → 성공 시 H1 확정. _actuator_mode=2를 사용하려면 별도 보상 재튜닝 필요.
  → 실패 시 H2/H3 단독 실험 또는 다른 root cause로 escalate.
```

대안: 정량적 신호를 더 빨리 얻고 싶으면 학습 없이 play 모드에서 `_actuator_mode=2` vs `=1`로 random/scripted action을 step 0.45 m / gap 0.5 m에 적용해 base_z 도달 가능 여부, calf joint applied_torque 분포를 비교(applied_torque/saturation_effort 비율 > 0.95% 빈도). 단 학습 정책이 아니라 결과 해석이 약함.

---

## 4. 추천 후속 액션 (메인이 결정. worker별 task 형태로 정리)

### Step A — 검증 실험 X1 실행 (debug-worker는 분석만, 실제 학습 트리거는 메인 또는 locomotion-loop)
- cfg-worker 호출: `_actuator_mode = 1`로 변경하는 한 줄 patch (parkour_env_cfg.py:399).
- locomotion-loop or 메인이 `./isaaclab.sh -p .../rsl_rl/train.py --task Go2-Parkour-Direct-v0 --num_envs 4096 --logger wandb` 실행.
- log-analyzer에 위 메트릭 추출 + Green/Yellow/Red 판정 위임.

### Step B (X1이 성공한 경우)
- _actuator_mode=2를 정말 운영에서 쓰려면 별도 보상/curriculum 재설계 필요:
  - reward-worker: step_height/gap_length curriculum의 max 한계를 actuator envelope 안에 맞춰 축소
  - 또는 hyperparam-worker: action_scale↑로 PD 약화를 부분 보상
  - 의사결정은 메인.

### Step C (X1이 실패한 경우 — H1만으로는 부족)
- 다음 단일 실험: `feet_dragging` -0.1 → 0.0 단독 복원, _actuator_mode=2 유지. H3 검정.
- 그래도 회복 안 되면 termination 빈도 / 보상 분포 wandb 로그에 대해 log-analyzer에 deeper RCA 위임.

### 비-액션 (수정 금지 권고)
- `recently_reset` 게이트(E)와 contact gating 전환(D)은 step/stair/gap 가설로서 약함 — 되돌리지 말 것.
- `next_goal_threshold` 0.2(F)와 `y_offset_range` 축소(H)는 도움 방향 — 되돌리지 말 것.

---

## 부록: 라벨링 정책 재확인

- 확정: 없음 (학습 로그/사용자 측정치 미확보, 코드 표면 추론만)
- 강한 가설: H1, H2 (H2는 H1 conditional)
- 약한 가설: H3, H4
- 매우 약한 가설: H5

CLAUDE.md memory 규칙("검증되지 않은 코드 표면 추론을 root cause로 단정하지 말 것")에 따라
모든 H#는 "확정"이 아니며, 단일 검증 실험(X1)로만 H1을 confirm/reject 가능함.

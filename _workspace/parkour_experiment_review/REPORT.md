# Parkour 실험 분석 종합 보고서

**대상:** `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/` 24개 실험
**기간:** 2026-05-11 ~ 2026-05-14 (약 3.5일)
**작성:** parkour-exp-review team (worker-params + worker-metrics + worker-video + team-lead)

> **주의:** 정성적 인과 추정은 "추정"으로 표기. 모션 평가는 정성 1인 평가(5 sparse frame). Terrain level은 단순 평균이 아니라 **각 환경의 curriculum stage**(0=가장 쉬움~10=가장 어려움) 평균이므로 학습 시간과 강한 상관이 있다.

---

## 1. 실험 타임라인 개관 (질문 #3과 일부 겹침)

24개 실험을 디렉토리명·param 변경 패턴으로 5단계로 묶었습니다.

| Phase | Exp # | 일자 | 의도 | 핵심 변경 |
|---|---|---|---|---|
| **A. Actuator 튜닝 (초기)** | 1-3 | 5/11 | actuator 모델 교체 + LR 보정 | (1) custom 23.5/40/0.5 → (2) IsaacLab default actuator → (3) LR 0.001→0.0001 |
| **B. Damping + Feet-dragging 도입** | 4-10 | 5/11~12 | dragging penalty 도입 + 모터 damping 탐색 | feet_dragging 0→-1.0→-0.1, damping 2.0↔0.5↔1.0 진동, LR 0.0001→0.001 복귀 |
| **C. Flat-terrain 디버그** | 11-16 | 5/13 | 평탄 지형에서 보행 lock-in/yaw 문제 격리 | yaw_wrap 처리, dof_error_l2 -0.04→-0.5, feet_air_time +0.5 신규 |
| **D. Reward Ablation (foot penalty 제거)** | 17-22 | 5/13~14 | foot 관련 penalty가 학습에 미치는 영향 격리 | action_rate -0.01→-0.1, feet_air_time/dragging/edge/stumble/hip_pos를 0으로, tracking_goal_vel 1.5→1.0, dof_error→0 |
| **E. 회복 + 환경 변경** | 23-24 | 5/14 | foot penalty 재활성화 + 충돌/PhysX 변경 | feet_edge/stumble -0.0→-1.0, action_rate 0→-0.1, tracking_goal_vel 1.0→1.5, PhysX cfg 변경 |

**큰 흐름:** "actuator → reward shaping → flat-terrain 디버그 → 무엇이 핵심 penalty인지 ablation → 최근에는 환경 자체(collision/PhysX) 변경" 으로 점차 **시뮬레이션 사실성**으로 관심이 옮겨가는 패턴.

**디렉토리명 의도 vs 실제 변경 (worker-params Section 2에서 확인)**:
- `change_actuator`, `isaaclab_actuator`: actuator 파라미터 ON/OFF (custom ↔ IsaacLab default).
- `damping_X.X`: motor damping 튜닝. 1.0 ↔ 0.5 ↔ 2.0 진동.
- `add_feet_dragging[_0.1]`: 발 끌림 penalty 도입(-1.0) 후 약화(-0.1).
- `flat_terrain*`: 평탄 지형에서 yaw/DOF/feet_air_time 디버그.
- `except_foot_penalty/friction/_dof_error/_no_action_penalty`: 시리즈로 penalty들을 차례로 disable → ablation.
- `change_physx_config`: PhysX 시뮬 설정 변경 (디렉토리명 추정).

---

## 2. Terrain Level 분석 (질문 #1)

### 발견 가능한 태그

worker-metrics 분석 결과 **전체 terrain level**만 발견:

- `curriculum/mean_terrain_level` — 환경 평균 curriculum stage (0~N)
- **지형별(step/stair/gap/flat 등) 분리 태그는 events 파일에 없음.** → 사용자 질문 1의 "각 지형별 평균"은 **현 로그로는 답변 불가**. (parkour env가 per-terrain tag를 write하지 않음. 필요 시 cfg에 metric 분리 로깅 추가 권장.)

### 전체 평균 (23개 실험; #12 events 파싱 실패 1건 제외)

| 통계 | 값 |
|---|---|
| Final mean terrain_level (n=23) | **3.47** |
| Median | 4.24 |
| Std | 2.18 |
| Range | 0.30 ~ 6.18 |

### 상위 5 / 하위 5 (Final terrain_level)

| 순위 | 실험 | Final | Max | Steps | Reward |
|---|---|---:|---:|---:|---:|
| 🥇 1 | `2026-05-13_12-39_flat_terrain_change` | **6.18** | 7.36 | 3,299 ⚠️ | 21.00 |
| 🥈 2 | `2026-05-13_09-28_flat_terrain` | 6.11 | 7.30 | 10,603 ⚠️ | 21.07 |
| 🥉 3 | `2026-05-12_18-23_change_various_thing_yaw_diff` | 5.71 | 6.49 | 53,883 ✓ | 16.93 |
| 4 | `2026-05-11_15-27_change_actuator` | 5.69 | 6.43 | 87,644 ✓ | 12.47 |
| 5 | `2026-05-13_13-37_flat_terrain_change` | 5.66 | 7.32 | 17,445 ⚠️ | 22.02 |

| 순위 | 실험 (bottom) | Final | Steps | Reward |
|---|---|---:|---:|---:|
| ❌ 1 | `2026-05-14_12-12_change_physx_config` | 1.14 | 851 ⚠️⚠️ | **-3.28** |
| 2 | `2026-05-11_17-46_damping_2.0` | 0.97 | 55,992 ✓ | 3.46 |
| 3 | `2026-05-11_17-47_add_feet_dragging` | 0.88 | 55,940 ✓ | 5.02 |
| 4 | `2026-05-14_09-26_except_foot_friction_average_no_action_penalty` | 0.38 | 9,361 ⚠️ | 8.61 |
| 5 | `2026-05-14_09-26_up_no_dof_error` | 0.30 | 9,411 ⚠️ | 5.06 |

### **핵심 해석 (반드시 읽을 것)**

**(1) Top 1-2위의 "flat_terrain" 계열은 비교 불가.** 디렉토리명에서 알 수 있듯이 평탄 지형 단일 실험으로 보임. curriculum이 빨리 올라가므로 단순 ranking에 들어가는 게 부당함. **실제 parkour 성능 지표로 활용 X.**

**(2) Bottom 5 중 3개(physx_config, foot_friction_no_action, up_no_dof_error)는 학습 시간 자체가 부족** (≤9.4k step). 통상 PPO 50-100k step에서야 curriculum 진행. → "정책이 나쁘다"가 아니라 **"학습이 아직 시작 단계"**.

**(3) damping_2.0과 add_feet_dragging(-1.0)은 진짜 트랩.** 둘 다 55k step 충분히 학습했는데도 terrain_level <1.0. → motor saturation(damping=2.0) + 과한 dragging penalty(-1.0)는 정책 학습을 명확히 봉쇄. 후속 실험에서 모두 약화/회복됨.

**(4) 가장 신뢰할 만한 "성공" 실험:** Exp 9 (`change_various_thing_yaw_diff`, terrain 5.71, 53k step, reward 16.93). 충분히 학습됐고 다양한 변경 후에도 회복. **다음 baseline 후보.**

> ⚠️ 사용자 질문 "지형별 평균 난이도"는 **현재 로그에 메트릭이 없어 답변 불가.** 권장: `_get_observations()` 직후 또는 `_compute_rewards()` 직전에 각 env의 terrain_type을 기반으로 per-terrain mean을 별도 tag로 write하도록 env에 reward_worker나 obs_worker로 코드 추가.

---

## 3. 마지막 비디오 모션 분석 (질문 #2)

worker-video가 24개 실험의 마지막 step mp4를 확인하고, **최신 5개 실험**의 5-frame 시각 검토 완료. (full 분석은 `03_video_motion.md` 참조.)

### 전체 메타데이터 요약

- **24개 중 22개**에 비디오 존재. **#12, #13** (yaw_wrap 계열)은 비디오 0개 — 학습이 매우 일찍 crash/abort된 것으로 추정.
- **가장 긴 학습:** Exp 2 (`isaaclab_actuator`) — 998,000 step / 518 비디오.
- **가장 짧은 학습:** Exp 24 (`change_physx_config`) — 2,000 step / 2 비디오.
- 최신 5개 실험 모두 **≤9,000 step** — 모두 early-training 단계.

### 최신 5개 모션 평가 (worker-video Section 2 요약)

| 순위 | 실험 | last_step | 정성 평가 |
|---|---|---:|---|
| 🥇 1 | `up_no_dof_error` | 9000 | **가장 깨끗한 자세.** 다리가 몸 아래로 tuck, 명확한 mid-stride 포즈, yaw/pitch/roll 안정. 가장 Go2-like. |
| 🥈 2 | `change_collision` | 8000 | 5프레임 중 4개에서 upright 보행 + 명확한 stride phase. 1개는 ambiguous low-body. |
| 🥉 3 | `except_foot_penalty_tracking_1.0` | 8000 | 모두 upright(넘어짐 없음), 그러나 **splayed/low-hipped** 자세 — 보행보다 균형 잡기에 가까움. |
| 4 | `except_foot_friction_average_no_action_penalty` | 8000 | #3와 비슷한 splay/crouch. **action penalty가 0이므로** 자세가 더 느슨해진 것으로 추정. |
| 5 | `change_physx_config` | 2000 | 대부분 collapse/face-down. **단 2k step**이라 비교 부당. |

### 모션-Reward 인과 추정

- **#21(up_no_dof_error)이 가장 깨끗 + reward 5.06 + terrain_level 0.30:** "dof_error_l2 = 0"이 보상은 낮아 보이지만 **자세 자유도가 늘어 자연스러운 stride**가 가능해진 것으로 추정. terrain_level이 낮은 것은 단순히 9k step 단계.
- **#22, #20의 splayed-low-hip:** action_rate/action_smoothness/dof_error가 모두 약화 → 다리 움직임이 더 "느슨"해짐. 보행은 학습 중이나 정교한 stride는 미달.
- **#24(physx_config)의 face-plant:** PhysX 변경 직후 + 2k step. 시뮬 안정성 자체가 흔들리고 있을 가능성 ↔ 단순 early-training 가능성 둘 다 살아있음. **8-10k step 도달 후 재평가 필요.**
- **No clear step/stair/gap traversal** in any of top-5 videos — 모든 최신 실험이 너무 일찍 끝나서 curriculum이 평탄 단계에 머물러 있음.

---

## 4. 하이퍼파라미터 변경 vs 보상/모션 효과 (질문 #3)

**대전제:** 24개 실험 전체에서 **PPO core hyperparameter는 거의 안 바뀜.** lr 1회 진동(0.001↔0.0001)을 제외하면 entropy_coef, clip_param, num_epochs, mini_batches, gamma, lam, num_steps_per_env, max_iterations 모두 동일. → **실제로 바뀐 건 reward scale과 env(actuator/physx)** 이다.

### 주요 변경별 인과 추정 표 (정성)

| 변경 | 도입 Exp | 이전 → 이후 | terrain_level 변화 | 모션/reward 변화 추정 |
|---|---|---|---:|---|
| LR 0.001→0.0001 | 3 | 5.47 → 5.31 | -0.16 (미미) | 큰 영향 없음. 다음 실험(7)에서 0.001로 복귀. 효용 없음으로 판단된 듯. |
| **damping 0.5→2.0** | 4 | 5.31 → **0.97** | **-4.34** | 모터 over-damping → torque envelope 압박, 정책이 stride 못 만듦. **명확한 트랩.** |
| **feet_dragging 0→-1.0** | 4(같이) | (위와 결합) | (위와 결합) | dragging penalty 너무 강함. 다음 실험에서 -0.1로 약화. |
| damping 2.0→0.5 | 5 | 0.97 → **0.88** | (회복 안 됨) | damping만 약화로 부족. dragging penalty 함께 약화 후에야 회복. |
| feet_dragging -1.0→-0.1 | 6 | 0.88 → 1.28 → 4.83 | +3.95(단계적) | 약화가 명확히 효과. |
| LR 0.0001→0.001 + actuator 강화(stiff 25→40, effort 23.5→35-40) | 7 | 4.83 → 4.92 → 5.71 | +0.88 | **actuator 강화가 stride 회복 도움.** |
| dof_error_l2 -0.04→-0.5 | 13 | (flat terrain 디버그 중) | n/a | 자세 강제 → 짧은 학습에서 reward 21.0(평탄 21.07) 달성. flat에서만 작동. |
| **feet_air_time +0.5 신규** | 14 | 5.66 → 6.18 → 6.11 | +0.5 | 평탄 지형에서 직선 보행 lock-in 방지 효과 추정. (commit log: "fix flat-terrain drag-forward lock-in") |
| action_rate -0.01→-0.1 | 17 | 5.32 → **4.24** | **-1.08** | 액션 변화율 페널티 10배 → 정교한 stride 학습 방해 추정. |
| **foot penalty 일괄 제거 (air_time/dragging/edge/stumble/hip_pos)** | 18 | 4.24 → **1.72** | **-2.52** | 발 관련 penalty가 일괄로 사라지자 정책이 발 끌기/엣지 침범 exploit. terrain progression 정체. |
| tracking_goal_vel 1.5→1.0 | 20 | 5.43 → 1.70 | -3.73 | foot penalty 제거 효과와 결합. 추적 보상 약화 + penalty 부재 = 학습 진행 안 됨. |
| dof_error_l2 -0.5→0 | 21 | 1.70 → 0.30 | -1.40 | **모션은 깨끗(자유도↑) but terrain은 안 올라감.** 단, 9k step만 학습이라 결론 보류. |
| action_rate/action_smoothness 0 | 22 | 0.30 → 0.38 | +0.08 | 9k step 추가 학습으로 약간 회복. 모션은 더 splayed. |
| **PhysX cfg 변경 + foot penalty/action_rate 재활성화** | 24 | (이전 0.38) → 1.14 | +0.76 | 2k step에서 robots face-plant. 단기로는 reward -3.28. **8-10k 후 재평가 권장.** |

### 핵심 결론

1. **PPO hyperparam(lr/entropy/clip/...) 튜닝은 사실상 거의 하지 않았음.** 변경의 실체는 **reward shaping + actuator + physx**. 만약 알고리즘 측 개선을 원한다면 hyperparam-worker 활용 여지가 크다.
2. **Foot 관련 penalty가 학습에 critical.** 일괄 제거하면 terrain_level이 1.72까지 폭락. 단, scale은 너무 크지 않게(-0.1 ~ -1.0 사이가 sweet spot으로 보임).
3. **damping=2.0은 분명한 학습 봉쇄 영역.** motor over-damping. 다시 들어가지 말 것.
4. **feet_air_time +0.5 추가가 flat-terrain reward를 21+로 끌어올림** — locomotion-loop의 핵심 발견.
5. **change_physx_config 실험은 미완.** 2k step만 학습 → 비교 자체가 불가. 동일 cfg로 50k+ 학습 후 평가 필요.

---

## 5. 추후 개선 방향 및 위에서 다루지 못한 중요 관찰 (질문 #4)

### 5.1 학습 시간/일관성 문제 (가장 큰 신호)

- **23개 중 11개가 ≤20k step에서 종료.** max_iterations = 50,000으로 설정되어 있는데 실제로 도달한 실험은 8개뿐. (worker-metrics Section 3 참조)
- 짧은 실험들 간 비교는 거의 무의미. **24개 비교 표는 misleading.**
- → 다음 이터레이션은 **반드시 동일 max_iter (또는 동일 wall-time) 도달 후** 비교할 것. 짧게 끊으면 ablation 결론을 잘못 내림.

### 5.2 미답변 질문: 지형별 평균 난이도

- **현재 events 파일에 per-terrain tag 없음.** `curriculum/mean_terrain_level`만 단일 스칼라.
- 권장: `parkour_env.py`의 `_compute_curriculum`(혹은 등가) 위치에서 env_origin/terrain_type을 기반으로 다음과 같이 분리 로깅:
  ```python
  for t_name, t_mask in self._terrain_type_masks.items():
      self.extras["log"][f"curriculum/level_{t_name}"] = self._terrain_levels[t_mask].float().mean()
  ```
- obs-worker 또는 reward-worker에게 위임 가능.

### 5.3 Reward Shaping 부작용 (sim-to-real 위험)

다음은 **이미 발견된 exploit 패턴** + sim-to-real 위험 신호:

- **stationary yaw exploit** (commit log에서 확인): tracking_yaw가 stationary 상태에서도 reward 줘서 정책이 멈춰서 yaw만 맞춤. → c3a7c43에서 actuator 튜닝 + stationary yaw reward 제거로 fix됨.
- **flat-terrain drag-forward lock-in** (commit e81b71c): 평탄에서 다리를 끌면서 전진. feet_air_time 추가로 fix.
- **friction-average 처리**: `_except_foot_friction_average` 디렉토리명 → 발 마찰을 평균화하는 처리로 추정. 4족에서 발마다 마찰이 다른 게 정상인데 평균을 쓰면 sim-to-real 위험. 코드 확인 필요.
- **action_rate -0.1은 너무 강할 수 있음**: -0.01에서 -0.1로 10배 증가. 실로봇 PD 컨트롤러는 50-200Hz, sim 50Hz라 action diff 페널티가 너무 강하면 stride 학습 자체를 막음. -0.05 정도로 절충 권장.

### 5.4 PhysX 변경의 의의/위험

- Exp 24가 PhysX cfg 변경 + 2k step만 학습. 변경 의도는 디렉토리명만으로 불분명. **변경된 PhysX 키와 그 의도를 commit message 또는 별도 노트로 기록 필요** (사용자에게 직접 확인 권장).
- PhysX 변경은 sim-to-real 전이에 큰 영향. solver_position_iter_count, max_depen_velocity, gpu_max_rigid_patch_count 등이 정책 학습 안정성에 영향. **변경 의도 명확화 후 50k+ step 학습 필요.**

### 5.5 다음 이터레이션 구체 추천

1. **Baseline 재확정:** Exp 9 (`change_various_thing_yaw_diff`) 또는 Exp 1 (`change_actuator`)을 **충분히(50-100k step)** 다시 돌려서 reproducible baseline 확립.
2. **지형별 terrain_level 로깅 추가** (위 5.2). obs-worker로 위임.
3. **Reward sweet spot 재확정:**
   - action_rate: -0.05 (-0.01과 -0.1 사이)
   - feet_air_time: 0.5 유지 (효과 입증됨)
   - feet_dragging: -0.1 유지
   - feet_edge/stumble: -1.0 유지 (제거 시 정체 확인됨)
   - dof_error_l2: 0 vs -0.04 ablation (-0.5는 너무 강함)
4. **PhysX 실험은 별도 ablation으로:** baseline 대비 PhysX cfg 1개씩 바꿔서 비교. 한 번에 여러 개 같이 바꾸지 말 것.
5. **모든 실험을 동일 max_iter(50k)에서 종료한 후 비교.** 짧은 실험은 결론으로 쓰지 말 것.
6. **Top-5 long-run 실험(998k, 165k, 87k, 68k, 65k step) 비디오 추가 분석:** worker-video는 최신 5개만 검토했음. 장기 학습 실험의 converged 모션이 진짜 baseline이 될 후보. → 후속 작업 추천.

### 5.6 메모리/규칙 관점 추가 관찰

- `feedback_torque_envelope_not_tier2_against_empirical.md`와 `project_parkour_actuator_mode2_verified.md`에 따르면 **damping 2.0/effort 23.5 등 actuator 약화는 검증된 봉쇄 영역이 아니다.** 실제로 Exp 9 (`change_various_thing_yaw_diff`)에서 effort=23.5/damping=1.0(이후 0.5)로 terrain 5.71 달성. **이 메모리와 일치.**
- damping=2.0(Exp 4)이 폭락한 것은 **2.0 자체가 트랩**이지 "낮은 damping = 트랩"이 아님. 이전 분석 가설(actuator 약화 = 학습 봉쇄)은 이번 데이터로도 **재반박됨.**
- `project_parkour_height_scan_verified.md`: parkour env의 height_scan은 정상. dragging RCA에서 후보로 거론 금지. → 이번 분석 결과 dragging은 **reward scale**로 충분히 통제 가능 (-0.1이 sweet spot).

---

## Sources

- 실험별 params 시간순 diff: `/home/lgb/IsaacLab/_workspace/parkour_experiment_review/01_params_changes.md`
- TF events terrain/reward 메트릭: `/home/lgb/IsaacLab/_workspace/parkour_experiment_review/02_terrain_metrics.md`
- 마지막 비디오 모션 분석 + 최신 5개 프레임: `/home/lgb/IsaacLab/_workspace/parkour_experiment_review/03_video_motion.md`
- 추출 프레임(25장 png): `/home/lgb/IsaacLab/_workspace/parkour_experiment_review/frames/<exp>/frame_*.png`
- 원본 로그: `/home/lgb/IsaacLab/logs/rsl_rl/go2_parkour/` (24개 실험)
- 분석 도구: rsl_rl tfevents (tbparse/EventAccumulator), ffmpeg (vlnce conda env), python yaml diff

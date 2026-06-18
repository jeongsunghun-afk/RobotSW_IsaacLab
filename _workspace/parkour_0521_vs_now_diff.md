# Go2 Parkour: `2026-05-21_15-48-19_feet_dragging` (당시) vs 현재 코드 비교

> 목적: 사용자가 "상대적으로 가장 좋게 나온" run으로 지목한 `2026-05-21_15-48-19_feet_dragging` 시점의
> 코드/설정을 정확히 복원하고, 현재(working tree)와의 차이를 사실 기반으로 나열한다.
> ⚠️ 프레이밍 주의: direct/parkour(A)는 known-good 학습 결과를 낸 적이 없으므로 "회귀(regression)/버그
> 도입" 프레임이 아니라 **중립적 시점 간 차이**로 기술한다. 미검증 추론은 버그로 단정하지 않는다.

## 1. "당시 코드"의 정의 (복원 방법)
- 학습 시작: 2026-05-21 15:48, 당시 HEAD = `dff065bccf9` ("Place parkour_hurdle goals between hurdles")
- 로그 디렉토리에 캡처된 ground truth:
  - `git/IsaacLab.diff` (1.28MB) — 학습 시점 working-tree 전체 diff
  - `params/env.yaml` (27KB), `params/agent.yaml` — **실제 사용된 resolved config** (가장 신뢰도 높음)
- 복원: base commit 파일 + 캡처 diff hunk 적용 → `/tmp/parkour_then/then_parkour_env*.py`
- 검증: 복원본의 reward set/차원/파라미터가 `params/env.yaml`·`agent.yaml`과 100% 일치 확인.

## 2. 핵심 차이 요약 (당시 → 현재)

### A. Reward term 2개 신규 추가 (14-term → 16-term)
당시 reward set에는 없던 항목이 현재 추가됨 (params/env.yaml에 부재 확인):
1. **`feet_gait_pairing`** (scale `0.0`) — Spot GaitReward 스타일 대각쌍(FL+RR, FR+RL) sync/async 위상 동기 보상.
   현재 scale=0.0 이므로 학습에 영향 없음(코드만 추가된 상태).
2. **`air_time_cap`** (scale `-0.1`) — 발별 연속 체공시간이 `air_time_cap_max_s=1.0s` 초과분에 graded penalty.
   3-leg gait(들린 다리) 억제 목적으로 06-02 추가.
   - ※ 2026-06-05 사용자 실측 메모: air_time_cap은 micro-tap escape로 거의 무력(-0.0033)으로 확인됨.

연관 신규 파라미터: `air_time_cap_max_s=1.0`, `feet_gait_std=0.2`, `feet_gait_max_err=0.3`, `feet_gait_velocity_threshold=0.3`.

### B. 기존 reward scale 변경
| term | 당시(5/21) | 현재 | 비고 |
|---|---|---|---|
| `action_rate_l2` | **-0.05** | **-0.1** | 2배 강화 (주석은 "WAS -0.1, 10x error"로 모순 — 값이 -0.1로 되돌려짐) |

그 외 14-term의 scale은 동일.

### C. Observation 차원 변경 ⚠️ (sim-to-real 고려 필요)
- **proprio(actor obs)에 `contact_filt` 4-dim 추가**: `(contact_filt.float()-0.5)`.
  - `num_proprio` 42 → 46, `observation_space` 42 → 46.
  - 4발 net contact force(>2.0N) 기반 binary contact를 **actor(배포되는 policy) 입력**에 포함.
  - ⚠️ 프로젝트 제약 메모(`project_parkour_analysis_constraints`)의 "Contact sensor obs 추가 금지(sim-to-real)"에
    해당하는 변경. 버그 단정이 아니라 **배포 관점에서 검토가 필요한 차이**로 플래그함. (당시 run에는 없음)

### D. priv_latent(critic 전용, privileged) 확장: 12 → 37
- 당시: `foot_friction(8) + base_mass(1) + base_com(3)` = 12 (`num_priv_obs`=18: priv_explicit 6 포함)
- 현재: `base_friction(1) + foot_friction(8) + base_mass(1) + base_com(3) + joint_stiffness_ratio(12) + joint_damping_ratio(12)` = 37 (`num_priv_obs`=43)
- `foot_friction`이 `[static,dynamic]`(8) → static-only로 추출 방식 변경(주석상). critic 전용이라 배포 영향 없음.
- 관련: agent cfg obs_groups critic total 247 → 272.

### E. Domain Randomization(EventCfg) 변경
- `foot_physics_material` (foot friction DR) → **주석 처리(비활성)**.
- base friction DR: body_names `"base"` → `".*"` (전 body), range `(0.6,1.2)/(0.5,1.0)` → `(0.4,1.5)/(0.3,1.2)`.
- **`randomize_actuator_gains` 신규 활성** (±2.5% scale, startup). 당시엔 "actuator_mode=2와 충돌"로 의도적 제외였으나
  현재는 priv_latent의 joint_stiffness/damping_ratio에 비영 신호 공급 목적으로 켜짐(보수적 범위).

### F. Command / 에피소드 동역학 변경
| 항목 | 당시 | 현재 |
|---|---|---|
| `command_cfg.lin_vel_x_range` | [0.3, **1.0**] | [0.3, **1.5**] (상한 속도 ↑) |
| `resampling_time_s` | **6.0** | **4.0** (명령 재샘플 빈도 ↑) |

### G. Actuator / sim 설정
- `_actuator_mode=2` 동일(당시·현재 모두). 단 현재는 `_actuator_mode==3` 분기 코드가 추가됨(현재 미사용).
- terrain physics: `restitution=0.0` 라인 제거(기본값 사용).

### H. Agent(PPO/네트워크) 설정
- `noise_std_type`: **`scalar`(당시) → `log`(현재)** — action noise std 파라미터화 방식 변경(탐험 동역학에 영향 가능).
- 그 외 PPO 하이퍼파라미터(lr 2e-4, entropy 0.01, KL 0.01, epochs 5, minibatch 4, gamma 0.99 등) 동일.

### I. 분석/디버그용 비기능 변경 (학습 무영향)
- `_last_reward_breakdown_per_env` 버퍼 + `_reward_term_names` (per-terrain reward attribution 도구용).
- `_gait_synced_pair_*`, `_base_shape_index` 인덱스 캐시.
- ⚠️ `parkour_env.py` 디버그 함수에 `print(forces)` leftover 추가됨 → 스팸 로그 유발 가능, 정리 권장.

## 3. 변경 동기(commit narrative)
5/21 이후 parkour 디렉토리 commit:
- `798d9b5ebe5` (05-28) per-terrain reward attribution tooling (3-leg gait 분석용)
- `90f82298ef1` (06-02) air_time_cap penalty (3-leg gait fix 시도)
즉 5/21 이후 변경의 대부분은 **3-leg gait(들린 다리) 문제 분석/억제 시도**에서 비롯됨.
단, 2026-06-05 실측 메모상 air_time_cap은 효과 미미로 확인됨.

## 4. 한 줄 결론
"가장 좋았던" 5/21 run은 **16-term이 아닌 14-term reward + 42-dim proprio(contact obs 없음) + priv_latent 12 +
DR 최소(actuator gain DR off) + cmd속도 1.0 상한 + noise_std scalar** 구성이었다.
이후 변경은 (1) reward 2종 추가(gait_pairing=0, air_time_cap=-0.1), (2) actor obs에 contact 4-dim 추가,
(3) priv_latent/critic 확장, (4) DR 확대+actuator gain DR on, (5) cmd 속도/resample 강화, (6) noise_std log 전환 으로 요약된다.
이 중 학습 거동에 실제 영향을 줄 수 있는 차이는 **C(actor contact obs), B(action_rate 2배), F(cmd 속도·resample), E(DR 확대), H(noise_std)** 이며,
A의 air_time_cap은 실측상 영향 미미(feet_gait_pairing은 scale 0).

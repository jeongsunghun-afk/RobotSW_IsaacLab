# 02 — go2_parkour_symmetry 8런 Config 변화 비교

> 작성: worker-2 (task #2) · 소스: `logs/rsl_rl/go2_parkour_symmetry/<run>/params/{env.yaml, agent.yaml}`
> 방법: 8런 env.yaml/agent.yaml 전체를 baseline(run1) 대비 `diff`. step_dt 재곱 없음(reward_scales는 weight 원본값).

## 런 라벨 (시간순)

| # | 디렉토리 | 단축명 |
|---|----------|--------|
| 1 | `2026-06-09_17-53-09` | **baseline** |
| 2 | `2026-06-10_17-00-32_gap_step_more_difficult` | terrain↑ |
| 3 | `2026-06-10_17-38-06_torque_penalty_0.0001` | torque×10 |
| 4 | `2026-06-11_10-08-37_no_duty_time_cap` | air/duty OFF |
| 5 | `2026-06-11_12-13-02_positive_work` | pw −3e-4 |
| 6 | `2026-06-11_16-21-24_no_duty_time_cap` | pw 0 (control) |
| 7 | `2026-06-11_17-28-23_positive_work_0.01` | pw −1e-2 |
| 8 | `2026-06-12_09-53-06_positive_work_0.001` | pw −1e-3 |

---

## A. reward_scales 전체 비교표 (항 × 런)

변화 없는 항은 윗줄에 한 번만, 런 간 달라지는 항만 행으로 강조.

### 불변 항 (8런 전부 동일)
```
tracking_goal_vel: 1.5    tracking_yaw: 0.5      lin_vel_z_l2: -1.0
ang_vel_xy_l2: -0.05      orientation_l2: -1.0   dof_acc_l2: -2.5e-07
collision: -10.0          action_rate_l2: -0.1   delta_torques: -1.0e-07
hip_pos: -0.5             dof_error_l2: -0.04    feet_stumble: -1.0
feet_edge: -1.0           feet_dragging: -0.1    feet_gait_pairing: 0.0
```
> **feet_gait_pairing = 0.0 (전 런 비활성)**, tracking_goal_vel/tracking_yaw은 전 런 1.5/0.5 고정.

### 변화 항

| reward 항 | run1 | run2 | run3 | run4 | run5 | run6 | run7 | run8 |
|-----------|------|------|------|------|------|------|------|------|
| `torques_l2` | -1e-5 | -1e-5 | **-1e-4** | **-1e-4** | -1e-5 | -1e-5 | -1e-5 | -1e-5 |
| `air_time_cap` | -0.1 | -0.1 | -0.1 | **-0.0** | -0.0 | -0.0 | -0.0 | -0.0 |
| `contact_duty_deficit` | -0.5 | -0.5 | -0.5 | **-0.0** | -0.0 | -0.0 | -0.0 | -0.0 |
| `positive_work` | (없음) | (없음) | (없음) | (없음) | **-3e-4** | -0.0 | **-1e-2** | **-1e-3** |

읽는 법(누적 변화 타임라인):
- **run3**: baseline + 터레인↑(run2 계승) + torques_l2 −1e-5→−1e-4(×10).
- **run4**: run3 계승(torque −1e-4 유지) + air_time_cap·contact_duty_deficit 둘 다 OFF(0).
- **run5~8**: torque를 **−1e-5로 원복**(run3/4의 ×10 해제), air/duty는 **계속 OFF**, `positive_work` 항 신규 추가.
  - run5 pw −3e-4 / run6 pw 0(control) / run7 pw −1e-2 / run8 pw −1e-3.

---

## B. reward 외 env_config 변화

env.yaml diff에서 reward_scales·log_dir 제외하고 바뀐 것은 **터레인 난이도 2개 필드뿐**.

| 필드 | 위치 | run1 | run2~run8 |
|------|------|------|-----------|
| `parkour_step` → `step_height_range` 상한 | terrain_generator sub_terrains | 0.45 | **0.6** |
| `parkour_gap` → `gap_length_range` 상한 | terrain_generator sub_terrains | 0.6 | **0.8** |

> **핵심 confound**: 터레인은 run2에서 한 번 어렵게 바뀐 뒤 **run8까지 원복 안 됨**. 즉 baseline(run1)만 쉬운 터레인, run2~8 전부 어려운 터레인. baseline과 이후 런의 직접 비교 시 터레인 차이를 반드시 보정해야 함.

### 변화 없음 확인된 env 파라미터 (전 런 동일)
- `decimation: 4`, `episode_length_s: 20.0`, `action_scale: 0.25`
- command: `command_cfg` 블록 동일, `resampling_time_s: 4.0`
- domain randomization 전부 동일: `randomize_rigid_body_material`(friction static/dynamic 1.0), `randomize_rigid_body_mass`, `randomize_com`, `push_robot`, `randomize_actuator_gains`
- `_actuator_mode: 2`, `terrain_curriculum: true`, `base_height_target: 0.34`, `termination_height/grace`, `max_tilt` 동일
- gait/duty 관련 보조 파라미터(`contact_duty_target: 0.5`, `contact_duty_tau: 1.0`, `air_time_cap_max_s: 1.0`, `feet_gait_std/max_err` 등)는 **scale을 0으로 껐어도 파라미터 값 자체는 그대로** 남아있음.
- `rel_standing_envs`: 본 parkour env에는 **필드 없음**(hind_leg 전용 개념). 해당 없음.

---

## C. agent.yaml (PPO / symmetry / estimator) 변화

**8런 전부 완전 동일** (diff 결과 log_dir·run_name 외 차이 0).

| 항목 | 값 (전 런 고정) |
|------|------|
| `num_steps_per_env` | 24 |
| `max_iterations` | 50000 |
| `num_learning_epochs` | 5 |
| `num_mini_batches` | 4 |
| `learning_rate` (policy) | 2e-4 |
| `gamma` / `lam` | 0.99 / 0.95 |
| `entropy_coef` | 0.01 |
| `desired_kl` | 0.01 |
| `clip_param` | 0.2 |
| `surrogate_type` | ppo (`spo_epsilon: 0.2`, `lcp_cfg: null`) |
| `symmetry_cfg` | `use_data_augmentation: true`, `use_mirror_loss: false`, `mirror_loss_coeff: 0.0`, func=`...mdp.symmetry:compute_parkour_symmetric_states` |
| `estimator.learning_rate` | 1e-3 (HIM-style state estimator, policy LR와 별개) |

> 결론: **알고리즘/하이퍼/대칭 설정은 변수가 아님.** 8런 차이는 전부 env.yaml(터레인 2필드 + reward_scales 4항).

---

## D. 런 이름 vs 실제 yaml 검증

| 런 | 이름이 시사 | 실제 yaml | 판정 |
|----|------------|-----------|------|
| 2 | gap·step 더 어렵게 | step 0.45→0.6, gap 0.6→0.8 | ✅ 일치 |
| 3 | torque penalty 0.0001 | `torques_l2: -1e-4` | ✅ 일치 (단, 터레인↑도 계승 — 이름에 없음) |
| 4 | no_duty_time_cap | air_time_cap·contact_duty_deficit → 0 | ✅ 일치하나 **torque −1e-4도 유지**(이름 미표기) |
| 5 | positive_work | pw −3e-4 추가 + **air/duty 계속 OFF** + torque 원복(−1e-5) | ⚠️ 이름은 pw만 시사, 실제론 air/duty OFF·torque 원복 동반 |
| 6 | no_duty_time_cap | run5와 동일 config, **pw 0** | ⚠️ **이름 오해**: 실제론 positive_work 시리즈의 pw=0 control. air/duty OFF는 run4부터 공통이라 이 이름은 변별력 없음 |
| 7 | positive_work_0.01 | pw −1e-2 | ✅ 일치 |
| 8 | positive_work_0.001 | pw −1e-3 | ✅ 일치 |

**이름 함정 요약**
- "no_duty_time_cap"은 run4·run6 두 번 쓰였고, air/duty OFF는 **run4부터 run8까지 전부 공통**이라 이 라벨만으로 런을 구분할 수 없다. run6의 진짜 정체는 **positive_work 시리즈의 pw=0 대조군**(run5/7/8과 air/duty OFF·torque −1e-5 동일, pw만 0).
- torque ×10(−1e-4)은 run3·run4에만 적용되고 run5부터 원복(−1e-5)됐다. 즉 positive_work 시리즈(run5~8)는 torque penalty 측면에서 baseline과 같다.

---

## E. "런 간 실제로 바뀐 변수" 최종 정리

env.yaml 단 두 그룹(터레인 2필드 + reward 4항)만 변수. 누적 적용 형태:

```
run1 baseline:  터레인 easy | torque -1e-5 | air -0.1 | duty -0.5 | (pw 없음)
run2 +터레인:   터레인 HARD | (reward는 baseline과 동일)
run3 +torque:   터레인 HARD | torque -1e-4 | air -0.1 | duty -0.5
run4 +air/dutyOFF: 터레인 HARD | torque -1e-4 | air 0   | duty 0
run5 pw -3e-4:  터레인 HARD | torque -1e-5(원복) | air 0 | duty 0 | pw -3e-4
run6 pw 0:      = run5 with pw 0   (positive_work control)
run7 pw -1e-2:  = run5 with pw -1e-2
run8 pw -1e-3:  = run5 with pw -1e-3
```

**비교 분석 시 주의(confound)**
1. **터레인**: run1만 easy. run2~8 비교는 동일 HARD 터레인이므로 공정, baseline 대비는 보정 필요.
2. **air_time_cap·contact_duty_deficit**: run4부터 전부 OFF. 기존 gait-shaping(duty/air) 신호가 positive_work 시리즈엔 없음 → positive_work 효과는 "gait 신호 부재 상태"에서 측정된 것.
3. **torque penalty**: run3/4만 −1e-4, run5~8은 −1e-5. positive_work A/B(run5~8)는 torque 동일.
4. **clean A/B 가능 구간**: run5↔run6↔run7↔run8은 `positive_work`만 다른 4점 sweep(0, −3e-4, −1e-3, −1e-2) — 다른 모든 변수 동일. positive_work 효과의 가장 깨끗한 비교군.

> calibration(이전): pw −3e-4 안전 / −1e-3 경계(stair p99 tracking 82%) / −1e-2 붕괴. 위 run5/8/7이 각각 이에 대응(run6=control). 코드 미수정.

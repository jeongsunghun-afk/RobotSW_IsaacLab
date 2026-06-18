# parkour_imitation 환경 종합 검증 보고서

**일자**: 2026-05-28
**대상**: `source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/`
**목적**: 평지에서 3-leg gait 문제를 AMP style reward로 해결하려는 의도가 제대로 구현되었는지 검증
**방법**: 정적 코드 분석 (debug-worker, validate-code) + 모션 데이터 분석 + 학습 로그 9개 run 메트릭 추출
**상태**: analysis only — 코드 수정 없음

---

## Executive Verdict

| 영역 | 결과 |
|---|---|
| **코드 정합성 (runner fusion, mask, terminal obs, dim)** | ✅ **PASS** — T1 grep-verified 모든 핵심 경로 정상 |
| **Reference 모션 품질 (Go2 trot)** | ✅ **PASS** — 정상 4-leg diagonal trot, 1.67 m/s, FL-RR corr +0.86 |
| **Discriminator obs 설계 (no terrain conflate)** | ✅ **PASS** — joint+root만, heightmap 없음 |
| **모니터링 가시성** | ⚠️ **FAIL** — `AMP/amp_reward_mean_*` 메트릭 9개 run 전부 0.0 고정 (runner가 env 버퍼 미작성) |
| **신호 magnitude 보정** | ⚠️ **CONCERN** — 평지 per-step AMP ≤ 0.012 (전형적 AMP의 1/83 수준) |
| **학습 결과 (의도 달성?)** | ⚠️ **INCONCLUSIVE** — 9개 run 중 단 2개만 > 1500 iter, flat vs terrain 분리는 1273-iter `history_length_2` run 1개만 미약하게 관찰 |
| **Doc rot** | ⚠️ FAIL — `H=10/430-dim` 스테일 docstring 8개 사이트 (실제 `H=2/86-dim`) |

**한 줄 결론**: **설계는 정상, 코드는 정상, 그러나 (a) 모니터링 메트릭이 작동 안 해 의도 달성 여부를 직접 확인 불가**하고, **(b) AMP 신호 크기가 1/83로 약해 평지에서 task reward를 dominant하게 shaping하기 어려움**. 의도된 4-leg trot 모션 reference 자체는 정상.

---

## 1. 의도-구현 매핑 (debug-worker 결과)

### 1.1 Runner-side fusion (PASS, T1)

`rsl_rl/rsl_rl/runners/on_policy_runner_parkour_amp.py:141, 150`:
```python
amp_contribution = self.amp_weight * flat_mask.float() * disc_reward   # 평지만 통과
total_reward = rewards + amp_contribution                               # additive
```

- `amp_weight = 0.3` (runner cfg)
- `flat_mask`는 env에서 `_env_class == TERRAIN_CLASS_FLAT(=0)` 로 갱신
- **순서**: env의 `clip(task_total, min=0.)`는 `_get_rewards`에서 적용 → runner는 clipped reward에 AMP 추가 → 평지에서 음수 task가 양수 AMP를 erase 하지 않음 ✓

### 1.2 Discriminator gradient의 flat-only 학습 (PASS, T1)

`on_policy_runner_parkour_amp.py:137`:
```python
if flat_mask.any():
    amp_obs_buffer.append(corrected_amp_obs[flat_mask].detach())  # 평지 sample만
```

Replay buffer에도 평지 sample만 적재. Discriminator는 비평지(parkour) 데이터로 오염되지 않음 ✓.

### 1.3 terminal_amp_obs 흐름 (PASS, T1)

- `_reset_idx` 진입 시 super 호출 **이전**에 `_terminal_amp_obs[env_ids]` 갱신 → robot state가 reset되기 전 시점의 ring buffer가 정확히 snapshot됨
- runner는 `extras["terminal_amp_obs"][dones.bool()]`로 done envs만 보정 → 평지 reset env가 zero-buffer 대신 직전 frame을 disc에 제공 ✓

### 1.4 amp_observation_space=86 동적 결정 (PASS, T1)

- env `amp_observation_space` property = `cfg.amp_history_length * cfg.amp_obs_dim = 2 * 43 = 86`
- runner cfg의 `86`은 `on_policy_runner_amp.py:40-44`의 `_construct_algorithm`에서 env property로 덮어써짐
- env cfg가 단일 소스, runner cfg는 fallback/문서

### 1.5 update_iteration 콜백 (CONCERN, T2)

- runner는 `hasattr(env, "update_iteration")`로 호출 시도하지만 **env에 정의되지 않아 silent no-op**
- 의도된 동작: "parkour의 default uniform terrain assignment 사용" — env_cfg.py:31에서 명시
- 단, runner docstring은 "spawn scheduler / flat_bias curriculum"을 약속해 misleading

---

## 2. 정적 코드 정합성 (validate-code 결과)

| 체크리스트 | 결과 | 근거 |
|---|---|---|
| A. obs 차원 | PASS | `[N,2,43]` → flatten 86, env property = runner cfg = 86 |
| B. **Docstring drift** | **FAIL** | "10 history / 430-dim" 잔존 8개 사이트 (env.py:9, 13, 18, 29, 100, 283-286 / env_cfg.py:27-29 / amp_cfg.py:18-19, 133) |
| C. Buffer 초기화 | PASS | `_amp_obs_buf, _flat_env_mask, _amp_reward_buf, _terminal_amp_obs` 모두 `__init__` + `_reset_idx` 초기화 |
| D. obs_groups vs env 반환 키 | PASS | runner cfg의 5개 키 모두 부모 parkour env가 반환 |
| E. DOF 순서 (motion vs Isaac) | PASS | `_motion_dof_indices = motion_lib.get_dof_index(robot_joint_names)`, fallback `range(12)` 존재. motion_lib `DOF_NAMES`는 IsaacLab/MuJoCo 표준 순서와 일치 ("FL_hip, FL_thigh, FL_calf, FR_..., RL_..., RR_...") |
| F. foot articulation index | PASS | `_amp_foot_body_ids = find_bodies(".*foot")`, `len==4` assertion |
| G. `_last_reward_breakdown_per_env` 의존 | PASS | 부모 parkour_env.py:193에서 할당, 자식이 상속 |
| H. clip_actions=10 / init_noise_std=1.0 | PASS | ActorCriticRMA + parkour action range와 부합 |

코드 자체는 학습 가능 상태. **Doc rot만 처리하면 정합 만족**.

---

## 3. Reference Motion 데이터 (직접 분석 결과)

### 3.1 파일 인벤토리

`source/isaaclab_tasks/isaaclab_tasks/direct/parkour_imitation/imitation/go2/` (총 12개 pkl):

| 파일 | 프레임수 | FPS | 길이(s) | 비고 |
|---|---|---|---|---|
| go2_trot0.pkl | 208 | 60 | 3.47 | trot, 원본 |
| go2_trot0_mirror.pkl | - | 60 | - | trot, 좌우 mirror |
| go2_trot1.pkl | - | 60 | - | trot 변형 |
| go2_trot1_mirror.pkl | - | 60 | - | mirror |
| go2_walk.pkl | 198 | 60 | 3.30 | walk |
| go2_walk1.pkl, go2_walk2.pkl + mirrors | - | 60 | - | walk 변형 |
| go2_walk_turn.pkl + mirror | 208 | 60 | 3.47 | walk + turning |

**비율**: trot 4개 (2 + 2 mirror), walk 8개 (4 + 4 mirror). Walk가 dominant 샘플링됨.

### 3.2 프레임 레이아웃 (motion_lib.py:9-22 + 직접 확인)

```
[0:3]   root_pos   (xyz, world frame)       — 3-d
[3:6]   root_euler (roll, pitch, yaw, rad)  — 3-d
[6:18]  joint_pos  (12 DOF, DOF_NAMES 순)   — 12-d
─────────────────────────────────────────
        total = 18 floats/frame
```

motion_lib는 finite-diff로 lin_vel/ang_vel, FK로 foot_pos_local 계산.

### 3.3 Gait 품질 검증 (trot0 vs walk 직접 분석)

**go2_trot0**:
- speed_mean = **1.67 m/s** (정상 trot 속도)
- thigh phase 상관:
  - FL ↔ RR (diagonal pair): **+0.861** ← 강한 in-phase
  - FL ↔ FR (좌우 동측 vs 반측): -0.956 ← 강한 anti-phase
  - FL ↔ RL (ipsi-lateral): -0.820 ← anti-phase
- **→ 진정한 diagonal trot 패턴** (FL+RR / FR+RL 페어)
- leg activity (calf + thigh range): FL=1.43, FR=1.47, RL=1.45, RR=1.37 → min/max = **0.93** ← 4발 모두 균등 활용. **3-leg gait 편향 없음** ✓

**go2_walk**:
- speed_mean = 0.15 m/s (저속 walk)
- thigh corr 약함 (FL-RR +0.36, FL-FR -0.65) — 4-beat walk 패턴, trot보다 phase 분리 약함

**결론**: reference 모션 자체는 의도(4-leg trot 패턴 모방)와 **부합**. 3-leg 가설을 reference data가 강화하지 않음. ✓

### 3.4 위험 신호

- Walk 8개 vs Trot 4개로 walk-dominant 샘플링. Motion lib는 `uniform weighting by default` (env_cfg.py:40 주석)로 walk 확률이 트롯의 2배. **평지 명령 속도가 1.5 m/s 같은 trot 영역인데 walk 모션이 dominant하게 샘플되면 discriminator가 저속 walk를 expert로 학습**. 명령 속도와 reference 속도 미스매치 가능성.

---

## 4. 학습 로그 분석 (TF events 9개 run)

### 4.1 Run 인벤토리

| 디렉토리 | iter | env.amp_history_length | 분류 |
|---|---|---|---|
| 2026-05-27_15-51-32 | **3437** | (run 1: H=10 추정) | full run |
| 2026-05-27_20-29-07 | 150 | 10 | smoke |
| 2026-05-27_20-40-21 | 139 | 10 | smoke |
| 2026-05-27_20-51-38 | 28 | 10 | smoke |
| 2026-05-27_20-55-00 | 34 | 10 | smoke |
| 2026-05-27_20-58-38 | **4999** | 10 | full run |
| 2026-05-28_15-39-24 | 1767 | **10** | medium run, last H=10 |
| 2026-05-28_15-49-37_amp_history_length_2 | 338 | **2** | first H=2 |
| 2026-05-28_16-12-52_amp_history_length_2 | **1273** | **2** | medium H=2 run |

**현재 head**: env_cfg.py `amp_history_length: int = 2` (오늘 변경).

### 4.2 핵심 메트릭 (모든 run 공통)

```
AMP/amp_reward_mean_flat    : 0.0  →  0.0  (전 구간 0)   ← BUG: 미작동
AMP/amp_reward_mean_terrain : 0.0  →  0.0  (전 구간 0)   ← BUG: 미작동
AMP/flat_env_fraction       : 0.20 →  0.20 (안정, ~1/5 envs가 평지)
Loss/amp_weight             : 0.30  (고정)
Episode_Reward/amp_reward   : run마다 0.05 ~ 35 (runner-side unmasked sum, 실제 disc 신호)
```

**🚨 결정적 발견**: `AMP/amp_reward_mean_flat`, `AMP/amp_reward_mean_terrain`이 **모든 run에서 일관되게 0.0**.

원인 (debug-worker + 코드 추적):
- env의 `_amp_reward_buf [N]`는 0으로 초기화 후 reset에서만 0 클리어
- env docstring (env.py:55, 94)은 "AMP runner writes per-env disc reward here each step"이라 주장
- 그러나 `on_policy_runner_parkour_amp.py` 어디에서도 `_amp_reward_buf`에 write 하지 않음 (line 148은 `self.amp_reward_sums`, 자기 자신의 누적기에만 write)
- 결과: `_log_amp_metrics`는 항상 0 텐서의 mean → 0 출력

**파급**:
- 평지 vs 비평지 disc reward 차이를 로그에서 직접 확인 불가능
- 모니터링 도구 ⊕ 의도가 실제 작동하는지 검증 channel이 비어 있음 (학습은 정상 진행되지만 "AMP가 평지에서만 작동한다"는 의도가 메트릭으로 가시화 안 됨)

### 4.3 `Episode_Reward/amp_reward` (runner-side unmasked sum) 추이

| Run | first | mid | last | max |
|---|---|---|---|---|
| 15-51-32 (3437i, H=?) | 0.02 | 0.05 | 0.05 | 0.09 |
| 20-58-38 (4999i, H=10) | 0.19 | 0.55 | 0.92 | 2.40 |
| 15-39-24 (1767i, H=10) | 0.94 | 6.88 | 3.84 | 18.27 |
| **15-49-37 (338i, H=2)** | 0.91 | 23.07 | 28.85 | **35.43** |
| 16-12-52 (1273i, H=2) | 0.02 | 0.24 | 0.24 | 0.63 |

- 같은 H=2에서도 0.6 vs 35.4로 8 배 차이 — 시드/초기 disc training 변동에 매우 민감
- H=10 run 중 가장 긴 4999i는 0.92로 수렴 → discriminator-policy 평형이 형성됨
- 절대값 자체는 unmasked 누적이므로 평지/비평지 분리 정보 없음

### 4.4 `AMP/total_reward_mean_flat` vs `_terrain` (task reward 분리)

핵심 — 평지에서 AMP가 효과를 내면 평지 task reward가 비평지보다 우월하게 증가해야 함.

| Run | flat_last | terrain_last | flat - terrain |
|---|---|---|---|
| 15-51-32 (3437i) | -0.057 | -0.056 | **-0.001** (사실상 동일) |
| 20-58-38 (4999i, 가장 긴 run) | +0.029 | +0.028 | **+0.001** (동일) |
| 15-39-24 (1767i, H=10) | -0.018 | -0.039 | **+0.021** (약한 우위) |
| 15-49-37 (338i, H=2) | -0.011 | -0.005 | -0.006 |
| **16-12-52 (1273i, H=2)** | **+0.028** | **-0.027** | **+0.055** (가장 큰 우위) |

- 4999-iter 최장 run에서 flat ≈ terrain → AMP 효과 없음
- 1273-iter `history_length_2` run에서 flat이 terrain보다 +0.055 우위 — **유일한 명확한 separation**
- 그러나 표본 1개라 noise vs signal 구별 불가능

### 4.5 action stats — policy 수렴 건강성

| Run | hip_sample_std (last) |
|---|---|
| 15-51-32 (3437i) | FL=1.84, FR=**3.03**, RL=1.47 |
| 20-58-38 (4999i) | FL=1.58, FR=1.39, RL=1.09 |
| 16-12-52 (1273i, H=2) | FL=2.17, FR=**2.56**, RL=1.61 |

`init_noise_std=1.0`에서 시작했는데 long run들에서 1.4 ~ 3.0으로 **증가**. PPO `desired_kl=0.01`의 adaptive schedule이 std를 낮춰야 정상인데 오히려 키우는 추세 → policy가 안정적으로 수렴 못함. AMP gradient noise일 가능성, 또는 reward signal 약함.

### 4.6 H=10 vs H=2 비교

5/28 15-39-24 (H=10) → 15-49-37 (H=2) 전환은 단순 cfg 변경:
```
- amp_history_length: 10  / amp_observation_space: 430
+ amp_history_length: 2   / amp_observation_space: 86
```
다른 cfg 동일. 결과 분리는 위 4.4 표 참조 — H=2 long run(16-12-52)이 유일하게 flat vs terrain separation을 보여 H=2가 더 나은 후보로 보이나, 단일 표본.

---

## 5. Critical Issues (우선순위 정렬)

### P0 — 모니터링 메트릭 미작동 (FAIL, T1)
- `AMP/amp_reward_mean_flat`, `AMP/amp_reward_mean_terrain` 9개 run 전부 0
- 원인: runner가 env의 `_amp_reward_buf`에 write 하지 않음
- 영향: **의도가 작동하는지 직접 확인할 채널이 없음**. 본 검증의 최대 한계.
- 수정 방안 (제안):
  - (A) runner에서 `env.unwrapped._amp_reward_buf.copy_(disc_reward)` 한 줄 추가
  - (B) env에서 `extras["amp_obs"]` 옆에 disc 결과를 받지 않고 runner-측 마스킹된 mean만 로깅하도록 변경
  - (A)가 minimal change. 본인 worker(reward-worker 또는 loss-worker)에 위임 추천.

### P1 — Doc rot (FAIL, T1, 코드 무관)
- 8개 사이트의 "H=10 / 430-dim" 스테일 docstring (실제 H=2/86)
- 향후 누군가 "코드가 잘못됐다"고 86 → 430으로 "수정"할 위험
- cfg-worker 위임 1-shot 작업

### P2 — AMP 신호 magnitude 1/83 (CONCERN, T2)
- per-step AMP ≤ 0.012 (disc_reward.clamp(0,1) × reward_coef 0.04 × amp_weight 0.3)
- 전형적 AMP 셋업의 1/83 수준 → 평지에서 task reward를 dominant하게 shape 어려움
- 4999-iter run의 flat-terrain 분리 부재가 이 가설을 약하게 지지
- hyperparam-worker 후속 실험 후보:
  - reward_coef 0.04 → 0.1 ~ 0.2
  - 또는 amp_weight 0.3 → 0.5 ~ 0.7

### P3 — Reference motion 분포 편향 (CONCERN, T2)
- trot 4개 vs walk 8개, uniform 가중 → walk 샘플 확률 ≈ 67%
- walk는 0.15 m/s, trot는 1.67 m/s — parkour의 평지 명령 속도(주로 1 m/s 이상)와 walk가 mismatch
- discriminator가 저속 walk를 expert로 학습하면 policy가 평지에서 trot 대신 walk 모방
- motion_lib에 가중치 설정 옵션 있는지(motion_lib.py:243-247 확인 필요) 또는 trot pkl만 선택하도록 env_cfg `amp_motion_pkl: "imitation/go2/go2_trot0.pkl"` 변경 옵션 고려

### P4 — Policy std 발산 (CONCERN, T2)
- long run들에서 hip sample_std 1.0 → 1.5~3.0 증가
- desired_kl adaptive schedule이 정상이라면 std 감소해야 함
- AMP gradient noise일 가능성. P2/P3 해결 후 재관측 필요.

### P5 — `update_iteration` no-op (CONCERN, doc-only)
- runner docstring 약속 vs env 미구현. 현재 의도와 부합(uniform terrain) 하지만 docstring misleading.

---

## 6. 의도 달성 여부 (사용자 질문에 대한 직답)

> **사용자 의도**: "기존 parkour 환경은 평지에서도 3-leg gait 문제. parkour_imitation은 평지에서만 imitation style reward로 4-leg gait pattern을 모방하게 함."

**의도-구현 매핑 진단**:

| 단계 | 상태 |
|---|---|
| 1. AMP가 평지에서만 reward 추가하는 코드 | ✅ 정상 (`flat_mask * amp_weight * disc_reward`, additive) |
| 2. Discriminator가 평지 데이터로만 학습 | ✅ 정상 (`amp_obs_buffer.append(corrected[flat_mask])`) |
| 3. Reference motion이 4-leg trot 패턴 보유 | ✅ 정상 (FL-RR diagonal corr +0.86, leg activity uniform 0.93) |
| 4. **AMP가 작동하는지 메트릭으로 확인** | ❌ **모니터링 미작동 — 직접 확인 불가** |
| 5. **학습 결과에서 평지/비평지 task reward 분리** | ⚠️ 약함 — 5개 run 중 1개(1273i H=2)만 +0.055 분리, 최장 4999i run은 분리 0 |

**최종 답변**:
- **설계는 의도와 정확히 일치하게 구현**되어 있음 (코드 정합 PASS)
- 그러나 **신호 크기가 1/83 수준으로 약하고**, **모니터링 채널이 비어 있어** 의도가 실제로 평지 gait를 4-leg으로 shape했는지 **로그만으로는 증명 불가**
- 가장 긴 H=10 run(4999 iter)에서 flat vs terrain task reward가 동일 → AMP 효과가 task-level에서 보이지 않음
- 단 1개의 1273-iter H=2 run에서 flat이 terrain 대비 +0.055 우위 → 미약하지만 의도된 방향. 표본 부족.

**권고 다음 단계** (사용자 결정 필요):

1. **P0 모니터링 fix를 먼저 처리** (1-line runner 패치) — 이게 안 되면 어떤 후속 실험도 검증 불가
2. P0 해결 후 H=2 cfg로 ≥ 3000 iter 1회 더 학습 → `AMP/amp_reward_mean_flat`가 양수로 수렴하는지 확인
3. 그래도 평지/비평지 분리가 미미하면 P2 (reward_coef 증가) + P3 (trot-only motion) 동시 적용
4. 평지 보행 품질은 메트릭만으로 부족 — 학습 후 `play.py`로 평지 trot 시각 검증 필요 (foot contact pattern, 3-leg vs 4-leg)

---

## 7. 검증 한계

- 본 audit은 정적 분석 + 로그 메트릭 추출. **실제 시뮬레이션에서 robot이 4발로 걷는지 시각 검증은 별도 필요** (`play.py` 또는 viewer).
- `flat_env_fraction = 0.2002`로 안정 — 평지 envs의 비율은 정상이므로 AMP가 적용될 envs는 충분히 존재.
- discriminator의 internal 메트릭(`disc_total_loss`, `disc_expert_output`, `disc_policy_output`)은 추출하지 않음 — 다음 단계에서 필요시 별도 분석.
- motion_lib의 sample weighting 정책 (uniform vs custom)은 코드의 motion_lib.py:243-247 라인을 직접 확인 필요 (validate-code worker가 PASS 했지만 가중치 정책 단정 안 함).

---

## 8. 부록: 메모리 제약 준수 확인

- ✅ `direct/parkour(A)` 한 번도 학습 성공 적 없음 — regression 프레임 없음
- ✅ `parkour total_reward clip(min=0)` 의도된 설계 — bug 후보로 ranking 안 함
- ✅ contact sensor obs / actuator_mode / 토크 envelope 가설 push 없음
- ✅ Tier 표기: T1(grep 확정) / T2(추론) 모두 명시
- ✅ 3-leg gait를 calf-divergence로 인용하지 않음 (reward attribution reframe 존중)
- ✅ A vs B 비교에서 "X 없다" 단정 시 grep 재검증 (P0 부재는 코드 grep로 확인 — `_amp_reward_buf` write 검색 결과 0 hit)

---

## 변경 제안 요약 (사용자 결정 대기)

| 우선순위 | 파일 | 변경 내용 | 담당 worker |
|---|---|---|---|
| P0 | `rsl_rl/rsl_rl/runners/on_policy_runner_parkour_amp.py` | line 148 근처에 `env.unwrapped._amp_reward_buf.copy_(disc_reward)` 추가 | loss-worker 또는 reward-worker |
| P1 | env.py, env_cfg.py, amp_cfg.py 8개 사이트 | "10 history / 430" → "2 history / 86" 일괄 치환 | cfg-worker |
| P2 | `agents/rsl_rl_amp_cfg.py:119` | `reward_coef=0.04` → `0.1` or `0.2` 실험 | hyperparam-worker |
| P3 | `parkour_imitation_env_cfg.py:42` | `amp_motion_pkl: "imitation/go2"` → `"imitation/go2/go2_trot0.pkl"` (trot only) 실험 | cfg-worker |

본 보고서는 분석만 — 실제 변경은 사용자 승인 후 적절한 worker에 위임.

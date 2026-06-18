# Go2-Imitation 고속 under-tracking — AMP/reward/obs lane 진단

**Symptom**: MimicKit Go2(steering/tracking)는 2.5~4 m/s 고속을 학습하나, IsaacLab `Go2-Imitation-v0`는 고속 명령에서 actual velocity가 ~1.8–2.0 m/s로 포화.

**Lane scope**: AMP discriminator + reward + observation 파이프라인. (curriculum/데이터 sparsity 제외 — 사용자 지정)

**전제 검증 결과**: Task(steering) reward 공식은 양쪽 *완전 동일* → task reward는 원인 아님(아래 §3). 차이는 **AMP disc obs 표현 + disc reward 변환식 + disc obs 정규화 clip**에 집중됨.

---

## ⚠️ 먼저 실행할 가르는 실험 (lane 귀속 판정)

이 lane이 정답을 소유하는지부터 1줄 cfg로 판정:

> `rsl_rl_ppo_cfg.py` amp dict에서 `task_reward_lerp=1.0`, `task_reward_lerp_start=1.0` (AMP off, pure task)
> + `go2_imitation_env_cfg.py` `tar_speed_max=4.0~5.0` (MimicKit는 5.0; IsaacLab 3.0이라 3.0 초과 명령 자체가 없음)

- **고속(3~4 m/s) 도달 → 천장은 AMP가 만든 것**. 본 lane 후보(§1,§2,§5)가 정답 사냥터.
- **여전히 ~2 m/s 포화 → AMP/reward/obs 아님**. 물리(action_scale/PD gain/torque)·cross-lane(§6)으로 이관.

본 보고서의 AMP 결론은 *이 실험이 AMP를 gate로 확인하는 것에 조건부*.

---

## 1. [CRITICAL] AMP disc reward 변환식: LS-GAN(bounded) vs BCE(unbounded)

| | IsaacLab (활성) | MimicKit |
|---|---|---|
| 식 | `clamp(1 − 0.25·(d−1)², min=0) · 2.0` | `−log(1 − sigmoid(logit)) · 2.0` |
| 범위 | **[0, 2.0] (상한 고정)** | **[0, +∞) (상한 없음)** |
| 파일:라인 | `rsl_rl/rsl_rl/modules/amp_discriminator.py:69-70` (`disc_reward_type="ls_gan"`, cfg:104에서 활성) | `mimickit/learning/amp_agent.py:242-244` |

**고속 영향(메커니즘)**: BCE reward는 disc가 generated를 expert로 "확신"할수록 보상이 무한정 커져, 정책이 disc를 더 잘 속이는 방향(=고속 reference 스타일 추종)으로 강한 gradient를 받음. LS-GAN reward는 d=1에서 포화(2.0 상한)되어, disc가 약간만 헷갈려도 보상이 평탄해져 "더 expert답게" 갈 인센티브가 약함. 고속처럼 disc가 generated를 쉽게 구분하는 영역(낮은 reward)에서 BCE는 더 가파른 회복 gradient를 준다.
**신뢰도**: 식은 검증(양쪽 코드 직접 확인). 고속 천장 인과는 §실험으로 확정 필요.

비고: `amp_discriminator.py:29`의 하드코드 `amp_reward_coef=1.5`는 `ppo_amp.py:46`에서 cfg `reward_coef=2.0`으로 **덮어써짐** → 실효 2.0, MimicKit `disc_reward_scale=2`와 일치. **phantom diff 아님(검증 완료).**

---

## 2. [CRITICAL] AMP disc observation: 표현·차원·history 깊이 전면 불일치

### 2-a. per-frame feature 구성
IsaacLab per-step (49D) — `go2_imitation_env.py:642-665` + `:668-710`:
- dof_pos **raw 12** + dof_vel 12 + root_height 1 + root_lin_vel(body) 3 + root_ang_vel(body) 3 + foot_pos_local 12 + root_rot_tan_norm **6** = 49

MimicKit per-step (111D) — `amp_env.py:327-348` + `deepmimic_env.py:722-774`:
- root_pos_obs 3 (heading-local xy + z) + root_rot_tan_norm **6** + **joint_rot_tan_norm 12×6 = 72** + key_pos 12 + root_vel(heading-local) 3 + root_ang_vel(heading-local) 3 + dof_vel 12 = **111**

핵심 차이:
1. **관절 표현**: MimicKit는 각 관절 회전을 **6D tan_norm(72D)**으로; IsaacLab는 **raw joint angle(12D)**. → disc가 보는 자세 표현의 해상도/기하가 근본적으로 다름.
2. **root_vel 좌표계**: MimicKit는 root_lin/ang_vel을 **heading-inverse로 회전(heading-local)** (`amp_env.py:304-317`). IsaacLab는 `root_lin_vel_b`/`root_ang_vel_b` (**full body frame**, pitch/roll 포함) (`go2_imitation_env.py:198-199, 269-270`). 고속에서 pitch가 생기면 body-frame vel은 heading-local vel과 어긋남 → disc 입력 분포 차이.

### 2-b. history 깊이
- IsaacLab `num_amp_observations=2` → 98-dim (`go2_imitation_env_cfg.py:64`)
- MimicKit `num_disc_obs_steps=10` → 약 1110-dim (`amp_go2_steering_env.yaml:num_disc_obs_steps: 10`)

**고속 영향**: 2 프레임만으로는 disc가 stride 리듬/주기를 모델링 못 함 → 순간 피처(특히 root velocity)에 의존. 고속에서 disc가 "빠른 속도 자체"를 generated 판별 단서로 쓰기 쉬워져 고속 AMP reward를 낮춤. 10프레임은 보행 주기 전체를 봐서 속도 크기보다 gait 패턴으로 판별.
**신뢰도**: 차원/표현/좌표계 차이는 검증(양쪽 코드 직접 확인). MimicKit 1110-dim은 손계산(111×10); 정확값은 `get_disc_obs_space()`가 `fetch_disc_obs_demo(1)` 런타임 shape로 산출(`amp_env.py:17-27`)하므로 표현 차이 결론에는 영향 없음. 고속 인과는 가설(메커니즘 명시).

---

## 3. [검증: 차이 없음] Task(steering) reward — 원인에서 제외

IsaacLab `go2_imitation_env.py:284-317` ≡ MimicKit `task_steering_mixin.py:compute_steering_reward`:
- `tar_reward = exp(−vel_err_scale·‖tar_speed·tar_dir − root_vel_xy‖²)`, **vel_err_scale=0.5 동일**
- `proj_speed < 0 → tar_reward=0` 게이팅 동일
- 속도 정의: 양쪽 **world planar 위치 차분/dt** (`root_vel_xy`) 동일
- 가중치 **tar 0.7 / face 0.3 동일**, face_reward = `clamp_min(dot(face_dir, char_fwd), 0)` 동일

→ **task reward gradient는 고속에서 양쪽 동일**. 본 lane에서 task reward는 root cause 아님. (신뢰도: 검증)

---

## 4. [검증: 차이 없음] AMP vs task 혼합 비율 — stale narrative 정정

- IsaacLab: `task_reward_lerp=0.5`, `task_reward_lerp_start=0.5` → **둘 다 0.5, 사실상 상수** (`rsl_rl_ppo_cfg.py:76-77`). CLAUDE.md/docstring의 "1.0→0.5 anneal" 서술은 **stale** (현재 코드는 anneal 없음).
- MimicKit: `task_reward_weight=0.5 / disc_reward_weight=0.5` (`amp_go2_task_agent.yaml`).

→ 혼합비 **양쪽 50/50 일치** → 원인 아님. (검증) 단, §1의 reward *변환식* 차이로 같은 50% AMP라도 실효 AMP gradient 크기는 다름.

---

## 5. [WARNING] disc obs normalization clip: None vs ±10

| | IsaacLab (활성) | MimicKit |
|---|---|---|
| clip | `disc_norm_clip=None` → **clip 없음** | `Normalizer(clip=10.0)` → **±10σ clip** |
| 파일:라인 | `rsl_rl_ppo_cfg.py:106` → `amp_discriminator.py:41-43` | `amp_agent.py:63` |

**고속 영향(메커니즘)**: clip 없으면 고속 시 root_lin_vel·root_height 같은 피처가 학습 초기 통계 대비 다수-σ 이상치로 disc에 입력됨 → disc가 고속 모션을 trivially "fake"로 판별 → 고속 AMP reward→0 → 정책이 고속에서 끌어내려짐. MimicKit ±10 clip이 정확히 이 OOD 고속 피처를 압축. **§2의 history 2-frame(순간 속도 의존)과 결합 시 효과 증폭.**
**신뢰도**: clip 값 차이 검증. 효과 강도는 고속 reference 커버리지와 상호작용(제외된 데이터 질문과 인접) → 가설, caveat 명시.

---

## 6. [Cross-lane 이관 — 본 lane 아님, 누락 방지]
가장 *직접적* 속도 제한 요인이라 명시 이관:
- `action_scale=0.25` (`go2_imitation_env_cfg.py:95`) + Go2 PD gain/torque limit → 물리적 속도 천장. MimicKit는 `hip_scale_reduction_factor=0.5`(hip만 축소) 같은 별도 액션 매핑 보유(steering yaml) — IsaacLab cfg엔 대응 항목 부재(추정, env action 매핑 재확인 필요).
- `tar_speed_max`: IsaacLab **3.0** vs MimicKit **5.0** → IsaacLab는 3.0 초과를 학습 중 명령조차 안 함. 고속 천장 관측 자체가 불가능 → §실험에서 반드시 상향.
→ **physics/action lane(또는 cfg-worker)** 으로 이관 권고.

---

## Root cause 랭킹 (AMP/reward/obs lane, lerp=1.0 실험이 AMP를 gate로 확인하는 조건부)

| 순위 | 후보 | 파일:라인 (IsaacLab) | 고속 천장 메커니즘 | 신뢰도 |
|---|---|---|---|---|
| 1 | disc reward LS-GAN(상한 2.0) vs BCE(무한) | `amp_discriminator.py:69-70`, cfg:104 | 고속 저-reward 영역서 LS-GAN gradient 평탄 → expert화 인센티브 약함 | 식 검증 / 인과 가설 |
| 2 | disc obs history 2 vs 10 + 관절 raw vs 6D + vel 좌표계 | `env_cfg.py:64`, `env.py:198-199,642-665` | 2프레임은 gait 주기 못 봄→순간 속도로 판별→고속 reward↓ | 차이 검증 / 인과 가설 |
| 3 | disc_norm_clip None vs ±10 | `cfg:106`, `amp_discriminator.py:41-43` | 고속 OOD 피처 미압축→disc가 고속을 fake 판별→reward 0 | 차이 검증 / 인과 가설 |

**가르는 실험(우선순위)**:
1. (lane 귀속) `task_reward_lerp=1.0` + `tar_speed_max=5.0` → 고속 도달? → AMP gate 여부 확정.
2. AMP가 gate면, **한 번에 하나씩**: ① `disc_loss_type/reward_type="bce"`로 전환 → 고속 회복? ② `disc_norm_clip=10.0` 단독 → 고속 회복? ③ `num_amp_observations` 상향(예: 6~10) → 고속 회복?

**위임 권고 (수정 금지, 분석만)**:
- [critical] hyperparam-worker: `rsl_rl_ppo_cfg.py:103-106` — bce 4종 세트 전환 ablation 1개씩.
- [warning] cfg-worker: `go2_imitation_env_cfg.py:77` `tar_speed_max` 5.0 상향(실험 전제), `:64` `num_amp_observations` ablation.
- [cross-lane] physics/action lane: `action_scale`/PD/torque + hip_scale 대응 부재 확인.

## 동일 패턴 주의 위치
- terminal/live/expert 3경로 모두 `_apply_root_rot_tan_norm(ref=frame[0]=newest)` 대칭 호출 — MimicKit `compute_tar_obs`도 `ref_root_rot`=윈도우 ref로 대칭. "fix disc root orientation"(629f32b) 의도대로 대칭 (검증). root orientation 처리는 추가 root cause 후보 아님.

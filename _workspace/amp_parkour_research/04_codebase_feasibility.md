# Task #4 — 우리 코드베이스 통합 가능성 분석 (AMP 인프라 + parkour env)

**작성**: code-feasibility worker · **날짜**: 2026-06-11
**질문**: "평지 자연 보행 데이터(go2_imitation의 AMP expert)를 parkour 학습에 적용할 수 있나?"

---

## ⭐ 헤드라인 (TL;DR)

**이미 구현·등록·학습까지 되어 있다.** 사용자가 묻는 통합은 신규 작업이 아니라 **기존 자산의 활용/확장** 문제다.

- Task: **`Go2-ParkourImitation-v0`** (`parkour_imitation/__init__.py:21-29`)
- Env: `Go2ParkourImitationEnv(Go2ParkourEnv)` (`parkour_imitation_env.py:49`)
- Algo: `PPOAMP(PPOParkour)` — PPOParkour(RMA+DAGGER+estimator) + AMP discriminator (`ppo_amp.py:242`)
- Runner: `OnPolicyRunnerParkourAMP(OnPolicyRunnerAMP→Parkour)` (`on_policy_runner_parkour_amp.py:56`)
- Cfg: `Go2ParkourImitationPPOAMPRunnerCfg` (`agents/rsl_rl_amp_cfg.py:33`)
- Expert 모션: **이미 평지 trot/walk pkl** (`parkour_imitation/imitation/go2/` — go2_trot0/1, go2_walk/walk1/walk2/walk_turn + mirror)
- **학습 로그 존재**: `logs/rsl_rl/parkour_imitation_go2/` 에 2026-05-27~28 run 9개 (일부 `_amp_history_length_2` 라벨). → smoke 이상으로 돌려본 흔적.

즉 "평지 자연 보행 데이터를 parkour에 AMP로 주입" = **이미 코드로 존재하며 expert 데이터도 평지 walk/trot 그 자체**.

---

## 1. AMP 인프라 (ppo_amp.py / amp_discriminator.py)

### 1.1 Discriminator 입력 feature 구성
`_compute_amp_obs` (go2_imitation_env.py:639-663) 및 parkour 측 `_update_amp_obs_buf` (parkour_imitation_env.py:151-195)가 **동일 레이아웃**:

| feature | dim | 비고 |
|---|---|---|
| dof_pos | 12 | joint 각도 |
| dof_vel | 12 | joint 속도 |
| **root_height** | 1 | root z (world) — **root height 포함** |
| root_lin_vel_b | 3 | body-frame 선속도 |
| root_ang_vel_b | 3 | body-frame 각속도 |
| foot_pos_local | 12 | [FL,FR,RL,RR]×3 base-local |
| **per-step 합** | **43** | |

- **orientation**: go2_imitation은 R4에서 `root_rot_tan_norm`(6D, heading-relative) 추가 → 49-dim (`go2_imitation_env.py:108`, `_apply_root_rot_tan_norm` :666). **parkour_imitation은 이 6D를 채택하지 않음** → per-step 43-dim 유지. (orientation은 root_height + foot_pos_local로 간접 표현)
- **History**: parkour 측 `amp_history_length=2` (`parkour_imitation_env_cfg.py:49`) → flatten **86-dim**. (go2_imitation은 10 → 490)
- joint-only 아님: root height + body-frame vel + foot pos 포함. proprioceptive style 신호.

### 1.2 AMP reward ↔ task reward 결합
- **discriminator 자체**: `compute_amp_reward` (amp_discriminator.py:54-87). reward type 3종 분기:
  - `ls_gan`: `clamp(1 − 0.25(d−1)², min=0)·coef` → **항상 ≥0**
  - `bce`: `−log(1−σ(d))` → ≥0
  - `wgan`: `(D−μ̂)/σ̂` → **signed (절반 음수)**
- **결합 위치는 알고리즘이 아니라 runner**. parkour 경로는 `OnPolicyRunnerParkourAMP.learn()`에서:
  ```
  total_reward = rewards + amp_weight · flat_env_mask.float() · disc_reward   (L141, L150)
  ```
  - **additive + flat-env 마스킹** (go2_imitation의 lerp fusion과 다름; `task_reward_lerp` 무시됨, L14-15 주석 명시)
  - `amp_weight=0.3`, `reward_coef=2.0*0.02=0.04` (cfg :117, :121) → 기여 매우 작게 튜닝됨.

### 1.3 WASABI(WGAN) opt-in 현황
- 플래그 살아있음: `disc_loss_type` / `disc_reward_type` / `disc_logit_reg_type` (ppo_amp.py:258-261). WGAN 불일치 가드(:264-268) 포함.
- parkour_imitation cfg 현재값: **`ls_gan`** (rsl_rl_amp_cfg.py:127-128). WGAN 미활성.

---

## 2. go2_imitation 모션 데이터

- 포맷: **PKL** (npz 아님). `Go2MotionLib` (motion_lib.py:237) — dir 지정 시 `*.pkl` 자동 탐색·uniform 가중 (:245-248).
- 프레임 레이아웃 18-col: `[0:3]root_pos, [3:6]rpy, [6:18]dof_pos(12)` — 속도/발위치는 FK+finite-diff로 계산 (CLAUDE.md).
- go2_imitation 원본: 7개 (pace/run/trot/walk0~3). **전부 평지 자연 보행**.
- env가 amp_obs 노출하는 법: `extras["amp_obs"]`, `extras["terminal_amp_obs"]` + `get_amp_observations(n)`(expert) + `amp_observation_space` property. parkour_imitation도 **동일 인터페이스 구현 완료** (parkour_imitation_env.py:146-147, 289-339, 280-287).

---

## 3. parkour 스택과의 충돌점 — **이미 해결됨**

### 3.1 storage / update() / hidden state
- `PPOAMP(PPOParkour)`: `update()`는 PPOParkour 그대로(RMA encoder + estimator + priv_reg + DAGGER). disc는 **별도 메서드 `update_amp()` + 별도 optimizer `disc_optimizer`** (ppo_amp.py:290). PPO storage(`RolloutStorage`)와 **완전 분리** — disc는 `amp_obs_buffer`(list) + 별도 replay buffer(:299) 사용. → storage 충돌 없음.
- runner가 3-스텝 순차 호출: `update_amp()` → `update()` → `update_dagger()` (runner L199-206). gradient 격리.
- **hidden state**: `ActorCriticRMA.is_recurrent=False` (actor_critic_parkour.py:82). history는 RNN이 아니라 **1D-CNN `StateHistoryEncoder`** (:23-78). 따라서 recurrent hidden-state 동기화 문제 **해당 없음**. (PPOParkour.update의 recurrent 분기는 미사용)

### 3.2 total_reward clip(min=0) ↔ AMP reward 충돌
- parkour task reward는 `_get_rewards` 끝에서 `torch.clip(total_reward, min=0.0)` (parkour_env.py:1272) — **task reward만** clip.
- AMP는 **clip 바깥**(runner, env.step 반환 후 L150)에서 더해짐. 따라서:
  - 현재 `ls_gan`(≥0): task(≥0) + amp(≥0) → 문제 없음.
  - **WGAN signed로 바꿔도** AMP가 clip 밖에서 더해지므로 음수 신호가 통째로 죽지 않음 → additive 합은 음수 가능(PPO advantage엔 정상).
- ⚠ **정정**: `IMPLEMENTATION_STATUS.md` WASABI 섹션(:65)의 "parkour clip(min=0)이 signed AMP 신호 절반을 죽인다" 우려는 **이 additive-outside-clip 설계에는 적용되지 않음**. (그 우려는 AMP가 clip을 통과하는 lerp/내부합 가정). parkour_imitation 경로는 구조적으로 회피.

### 3.3 policy 입력 불변 (사용자의 input팽창 dealbreaker)
- **확인됨, 무위반.** `amp_obs`는 `self.extras["amp_obs"]`로만 노출(parkour_imitation_env.py:146). policy obs dict(`policy/scan/priv_explicit/priv_latent/history`)에 **추가 안 됨**. obs_groups `"policy":["policy"]` = 42-dim 불변(rsl_rl_amp_cfg.py:63). discriminator 전용. → **input팽창 아님, dealbreaker 비해당**.

---

## 4. PPOParkour ↔ PPOAMP 클래스 관계

- `ppo_amp.py`에 **두 클래스 공존**:
  - `PPOAMPBase(PPO)` — encoder 없는 단순 AMP (go2_imitation steering용)
  - **`PPOAMP(PPOParkour)`** (:242) — parkour RMA 스택 + AMP. **이게 합쳐진 클래스**.
- 즉 "PPOParkourAMP를 새로 만들 필요 없음" — **상속으로 이미 존재**. update() 충돌 지점 0 (disc는 override 메서드+분리 optimizer). 변경 규모: **신규 알고리즘 코드 불필요**.

---

## 5. 통합 경로 (비용 오름차순) — 대부분 "이미 됨" 위에서의 선택지

### 경로 A — 기존 자산 그대로 사용 (비용 ≈ 0)
- **무엇**: `--task Go2-ParkourImitation-v0` 로 바로 학습. expert pkl만 큐레이션(평지 trot/walk).
- 바뀌는 파일: 없음 (모션 추가 시 `imitation/go2/*.pkl`만).
- LOC: 0~5.
- **리스크/한계**: **AMP가 `_flat_env_mask`로 평지 env에만 적용**(env.py:345-354, runner L141). 즉 *평지 구간에서만 자연 보행 style 강제*, **장애물 통과 동작에는 style 미적용**. 사용자 목표가 "평지에서 자연스러움 유지 + 장애물은 parkour reward"라면 이 경로가 정확히 그 설계. **학습 결과 품질은 미검증**(로그 존재하나 평가 안 됨).

### 경로 B — AMP 게이트를 장애물 지형으로 확장 (비용 小~中)
- **무엇**: `_flat_env_mask` 게이팅을 제거/확장해 disc reward를 장애물 env에도 적용.
- 바뀌는 파일: `parkour_imitation_env.py`(`_update_flat_env_mask` 로직), `on_policy_runner_parkour_amp.py`(L137 buffer 필터 + L141 reward 마스킹). ~20-50 LOC.
- **리스크 (높음)**: 평지 walk 데이터는 **점프/계단 동작에 대해 OOD** → discriminator가 정당한 parkour 동작(공중 자세, 비대칭 deck)을 "fake"로 처벌 → parkour task reward와 **직접 충돌**. 지형별 disc 또는 지형-적합 모션 없이는 역효과. (이게 Task #1~#3 문헌조사가 다루는 핵심 난제와 직결)

### 경로 C — signed/WGAN 또는 style-distill (비용 中~大)
- **C1 WGAN**: cfg `disc_loss_type/disc_reward_type="wgan"`, `gradient_penalty_coef=5.0`, `reward_coef` 재튜닝(rsl_rl_amp_cfg.py 수정 ~5 LOC). additive-outside-clip이라 clip 충돌은 구조적 회피(§3.2). 리스크: WGAN collapse/스케일 재튜닝, 미검증.
- **C2 distill**: go2_imitation에서 style 정책 학습 후 parkour에 DAGGER distill. 신규 distill 파이프라인 필요(~수백 LOC). 가장 큼. 현 코드에 직접 경로 없음.

---

## 6. 발견된 결함/주의 (비차단)

1. **문서 staleness (비차단)**: env.py:29/100/283/286, rsl_rl_amp_cfg.py:18-20/112/135 주석이 "10 history × 43 = 430" 이라고 적었으나 **실제 cfg `amp_history_length=2` → 86-dim**. 런타임은 runner가 `env.unwrapped.amp_observation_space.shape[0]`(=86)로 disc 입력차원을 **자동 덮어씀**(on_policy_runner_amp.py:41-44) → **실제 dim 불일치 버그 아님**, 주석만 stale(go2_imitation에서 copy 잔재). cfg `amp_observation_space=86`은 올바름(주석만 "×10"으로 틀림).
2. **학습 결과 미평가**: 로그 9개 존재하나 본 task는 코드 실태 조사 범위 — 성능/수렴은 log-analyzer 영역.
3. parkour 분석 제약(MEMORY): contact-sensor obs 추가 금지 등 — 단, AMP는 **policy obs 불변**이므로 sim-to-real deploy 영향 0(discriminator는 학습 전용, deploy 미사용).

---

## 결론

사용자 질문에 대한 코드 실태 답: **"가능하다"가 아니라 "이미 구현되어 있고 expert 데이터도 평지 walk/trot이며 학습까지 돌려본 적 있다."**
- 신규 알고리즘/클래스 작업 불필요 (`PPOAMP(PPOParkour)` + `OnPolicyRunnerParkourAMP` 완비).
- input팽창 dealbreaker **비해당**(amp_obs는 extras 전용, policy 입력 42-dim 불변).
- clip(min=0) 충돌 **구조적 회피**(AMP는 clip 바깥 additive).
- **진짜 미해결 질문은 "통합 가능성"이 아니라 "장애물 지형으로 style을 확장할 때의 OOD 충돌"**(경로 B 리스크) — 이게 문헌조사 Task #1~#3가 답해야 할 부분.

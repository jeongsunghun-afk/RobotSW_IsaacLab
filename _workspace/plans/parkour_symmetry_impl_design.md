# Go2-Parkour-Symmetry — Soft Symmetry Augmentation 구현 설계 조사

작성: 2026-06-09 / debug-worker (분석 전용, 코드 미수정)
대상: 새 task `Go2-Parkour-Symmetry` — PPO mirror data-augmentation (좌우 L/R equivariance soft 강제)

> 배경(확정): RL(왼쪽 뒷다리) 단독 park은 대칭 시스템의 emergent equilibrium.
> Fix = policy L/R equivariance를 data-augmentation 모드로 soft 강제. contact_duty reward 유지,
> parkour_env 동역학 불변. 알고리즘/cfg 측만 변경.

표기 규칙: **[검증됨]** = 코드/실행 1차 근거 / **[추정]** = 표면 추론(검증 방법 명시) / **[미검증-부재]** = grep으로 부재 확인했으나 단정 회피.

---

## 0. 핵심 요약 (가장 먼저 읽을 것)

- 이 task는 **L/R(좌우) 단일 대칭만** 적용한다. anymal 레퍼런스의 4-way(original/left-right/front-back/diagonal)는 **그대로 쓰면 안 된다.** parkour는 지형/명령이 전방 지향(forward-only command, scan offset +0.375m forward)이라 front-back은 task의 대칭이 아니다. → `num_aug = 2` (원본 + 좌우미러 1개).
- 인프라는 **이미 전부 존재**한다. `RslRlPpoAlgorithmCfg.symmetry_cfg` 필드(`rl_cfg.py:196`), `resolve_symmetry_config`(`symmetry.py:16`), `PPOParkour`의 data-aug/mirror-loss 처리(`ppo_parkour.py:284-292, 436-465`)가 모두 동작 상태. **새로 작성할 것은 (1) parkour용 `data_augmentation_func` 1개, (2) symmetry_cfg를 켠 RunnerCfg 1개, (3) gym.register 1개**뿐.
- parkour obs는 anymal과 달리 **flat tensor가 아니라 TensorDict**(`policy/scan/priv_explicit/priv_latent/history` 5그룹). aug 함수는 5그룹 전부를 일관되게 미러해야 한다(critic/encoder/estimator가 소비).
- **부재 확인**: parkour 쪽 기존 `data_augmentation_func`/`get_symmetric_states` **[미검증-부재]** — grep 결과 parkour 디렉토리·rsl_rl에 parkour용 대칭 함수 없음. 유일한 4족 레퍼런스는 anymal(manager-based). 즉 새로 작성 필요.

---

## A. symmetry_cfg → ppo_parkour `data_augmentation_func` 인터페이스

### A.1 `symmetry_cfg` dict 기대 키 **[검증됨]**
`ppo_parkour.py:101-123` 및 `:284-291, :436-465`에서 PPOParkour가 읽는 키:

| 키 | 타입 | 사용처(파일:line) |
|----|------|------|
| `use_data_augmentation` | bool | `ppo_parkour.py:104, 284, 440` |
| `use_mirror_loss` | bool | `ppo_parkour.py:104, 464` |
| `data_augmentation_func` | callable \| str | `ppo_parkour.py:109-116, 286, 441, 454` (str이면 `string_to_callable`로 resolve) |
| `mirror_loss_coeff` | float | `ppo_parkour.py:465` |
| `_env` | VecEnv | `resolve_symmetry_config`(`symmetry.py:29`)가 주입 → `ppo_parkour.py:291, 442, 455`에서 `env=` 인자로 전달 |

cfg dataclass 측 정의: `RslRlSymmetryCfg`(`source/isaaclab_rl/isaaclab_rl/rsl_rl/symmetry_cfg.py:11-51`) — `use_data_augmentation`(:28), `use_mirror_loss`(:31), `data_augmentation_func`(:34, MISSING), `mirror_loss_coeff`(:50). `_env`는 dataclass에 없고 런타임에 dict로 주입됨.

학습 모드 결정 **[검증됨]**:
- data-augmentation 모드(이번 task 권장): `use_data_augmentation=True`. obs/actions를 미러 복제해 batch를 2배로 늘려 surrogate/value loss를 그대로 학습(`ppo_parkour.py:284-299`). 추가 loss 항 없음 — "soft"하게 equivariance 유도.
- mirror-loss 모드(대안): `use_mirror_loss=True`. `mirror_loss_coeff * MSE(π(mirror(s)), mirror(π(s)))`를 loss에 가산(`:460-465`).
- 둘 다 False면 로깅 전용(`:106-107`).

> 권장: 우선 `use_data_augmentation=True, use_mirror_loss=False`로 시작. data-aug가 batch 2배 → KL/lr adaptive에 영향. mirror-loss는 batch 증가 없이 soft penalty만 추가하므로, 메모리 부담 시 대안.

### A.2 `data_augmentation_func` 정확한 시그니처/반환 **[검증됨]**

**호출 지점 1 — data augmentation** (`ppo_parkour.py:286-292`):
```python
data_augmentation_func = self.symmetry["data_augmentation_func"]
obs_batch, actions_batch = data_augmentation_func(
    obs=obs_batch,           # TensorDict, batch_size=[B]
    actions=actions_batch,   # Tensor [B, 12]
    env=self.symmetry["_env"],
)
num_aug = int(obs_batch.batch_size[0] / original_batch_size)   # :294 — 반환 obs로 num_aug 산정
```
**호출 지점 2 — mirror loss** (`ppo_parkour.py:441-442, 454-456`):
```python
obs_batch, _ = data_augmentation_func(obs=obs_batch, actions=None, env=...)   # actions=None 가능
...
_, actions_mean_symm_batch = data_augmentation_func(obs=None, actions=action_mean_orig, env=...)  # obs=None 가능
```

→ **시그니처 (확정)**:
```
def func(*, env, obs: TensorDict | None, actions: Tensor | None) -> tuple[TensorDict | None, Tensor | None]
```
- 키워드 인자 `obs=`, `actions=`, `env=` (호출이 키워드로 함 → positional 순서 무관하지만 이름은 정확히 맞춰야 함).
- `obs`가 None이면 반환 obs도 None, `actions`가 None이면 반환 actions도 None (anymal 패턴 `anymal.py:64-65, 80-81` 동일).
- 반환 batch는 `[B*num_aug, ...]`. 첫 B개는 **반드시 원본**(`ppo_parkour.py:307-309, 453, 461`이 `[:original_batch_size]`를 원본으로 가정 — entropy/mu/sigma 계산 + mirror loss target). 이번 task는 num_aug=2 → `[원본 B; 미러 B]`.
- `@torch.no_grad()` 권장(anymal `:23`). data-aug 결과는 storage 값처럼 취급되며 그래프 불필요.

> **주의 — 시그니처 docstring 불일치 [검증됨]**: `RslRlSymmetryCfg` docstring(`symmetry_cfg.py:43`)은 인자명을 `action`(단수)로 적었으나, 실제 호출(`ppo_parkour.py:288, 442, 455`)은 `actions=`(복수). **호출부를 따라 `actions`(복수)로 작성할 것.**

### A.3 cfg → PPOParkour 전달 경로 **[검증됨]**
1. RunnerCfg.algorithm = `RslRlPpoAlgorithmCfg(..., symmetry_cfg=RslRlSymmetryCfg(...))` (`rl_cfg.py:196` 필드 존재).
2. 러너가 cfg를 dict로 변환(`to_dict`) → `self.alg_cfg`.
3. `OnPolicyRunnerParkour._construct_algorithm`(`on_policy_runner_parkour.py:339`): `resolve_symmetry_config(self.alg_cfg, self.env)` → `symmetry_cfg` dict에 `_env` 주입(`symmetry.py:29`). symmetry_cfg가 None이면 그대로 None(`symmetry.py:31`).
4. `on_policy_runner_parkour.py:376-384`: `alg_class(actor_critic, storage, ..., **self.alg_cfg, ...)` — dict 스프레드로 `symmetry_cfg=`가 `PPOParkour.__init__`(`ppo_parkour.py:55`)에 전달.
5. `ppo_parkour.py:109-110`: `data_augmentation_func`가 str이면 import-path로 resolve. → cfg에 함수 객체 직접 넣어도, "module.path:func" 문자열로 넣어도 동작.

### A.4 IsaacLab 표준 rsl_rl과의 비교 **[검증됨]**
- 표준 `ppo.py`의 symmetry 처리는 `ppo_parkour.py`와 **로직 동일**(`ppo.py:87-106` init, `:254-261` data-aug, `:389-419` mirror-loss). 차이: `ppo_parkour`는 priv_reg/estimator/dagger/LCP 등 추가 항이 끼어있을 뿐, symmetry 인터페이스는 100% 호환.
- 레퍼런스 `data_augmentation_func` 구현체 (repo 내 존재) **[검증됨]**:
  - `source/isaaclab_tasks/.../locomotion/velocity/mdp/symmetry/anymal.py:24` `compute_symmetric_states(env, obs, actions)` — **4족 좌우/전후 대칭의 정석 레퍼런스**. obs flat tensor 슬라이싱 + joint swap + height-scan flip 패턴 전부 포함.
  - `source/isaaclab_tasks/.../classic/cartpole/mdp/symmetry.py` — 1-DOF 단순 예.
  - 등록 사용처: `anymal_b/c/d/__init__.py` + `agents/rsl_rl_ppo_cfg.py` (예: anymal cfg에서 `RslRlSymmetryCfg(use_data_augmentation=..., data_augmentation_func=compute_symmetric_states)`).
- **차이로 인한 주의**: anymal은 obs가 **단일 flat tensor**(`anymal.py:53` `obs["policy"]` 단일 그룹, slice index 하드코딩). parkour는 **TensorDict 5그룹** + RMA encoder 구조. anymal 슬라이스 인덱스(lin_vel 0:3부터 시작)는 parkour proprio 레이아웃(yaw_diff부터 시작, lin_vel 없음)과 **완전히 다름** → 인덱스 재작성 필수(섹션 B 참조).

---

## B. parkour observation 레이아웃 (mirror map 작성용)

### B.1 obs TensorDict 구조 **[검증됨]** (`parkour_env.py:1031-1038`)
반환 dict 키 5개:
```python
{ "policy": proprio(N,46), "scan": self._scan(N,187),
  "priv_explicit": (N,6), "priv_latent": (N,37), "history": (N,10,46) }
```
obs_groups 매핑(`rsl_rl_ppo_cfg.py:48-55`): policy←[policy], critic←[policy,scan,priv_explicit,priv_latent], scan←[scan], history←[history], priv←[priv_latent], priv_explicit←[priv_explicit].

> **aug 함수는 이 5그룹을 전부 미러해야 한다.** critic/scan_encoder/priv_encoder/history_encoder/estimator가 각 그룹을 소비하므로, 일부만 미러하면 advantage/value/latent가 오염된다.

### B.2 `policy`(proprio) 인덱스 맵 **[검증됨]** (`parkour_env.py:905-917`)
> 주의: cfg 주석(`parkour_env_cfg.py:417`)과 `_get_observations` 주석(`parkour_env.py:918`)은 "42"라 적었으나, cat에 `contact_filt(4)`가 포함되어 **실제 46-dim**. cfg `num_proprio = 42 + 4`(`parkour_env_cfg.py:430`), `observation_space = 42 + 4`(:426)로 46 확정.

| idx | 구성요소 | dim | 출처 line |
|-----|---------|-----|-----------|
| 0   | `yaw_diff` (target−heading, wrapped) | 1 | :907 |
| 1   | `next_yaw_diff` | 1 | :908 |
| 2:5 | `projected_gravity_b` | 3 | :909 |
| 5   | `commands[:,0:1]` (forward-only) | 1 | :910 |
| 6:18 | `joint_pos − default_joint_pos` | 12 | :911 |
| 18:30 | `joint_vel * 0.05` | 12 | :912 |
| 30:42 | `_actions` (this step) | 12 | :913 |
| 42:46 | `contact_filt − 0.5` (per-foot bool) | 4 | :914 |

### B.3 `scan`(height-scan) 레이아웃 **[검증됨 grid / 추정 lateral 축]**
- scanner: `RayCasterCfg(... pattern_cfg=GridPatternCfg(resolution=0.1, size=[1.6, 1.0]), ray_alignment="yaw")`, offset `pos=(0.375, 0.0, 20.0)` (`parkour_env_cfg.py:585-589`). num_scan_obs=187 (`:431`).
- grid 점 개수 **[검증됨]** (`patterns.py:45-47`): `x = arange(-0.8, 0.8, 0.1)` → 17점, `y = arange(-0.5, 0.5, 0.1)` → 11점. 17×11 = 187 ✓.
- ordering 기본 `"xy"` (`patterns_cfg.py:55`, parkour는 미지정 → 기본값). `indexing="xy"` (`patterns.py:43`) → `grid_x, grid_y = meshgrid(x, y, indexing="xy")` → 출력 shape `(len(y)=11, len(x)=17)` → flatten 시 **앞 축(11)=y(lateral), 뒤 축(17)=x(forward)**.
- **좌우 flip permutation [검증됨, anymal 대조]**: scan을 `(-1, 11, 17)`로 reshape 후 **dim=1(lateral, 크기 11) flip**. anymal 레퍼런스가 동일 grid(size `[1.6,1.0]`, ordering `xy`)에 대해 left-right=`view(-1,11,17).flip(dims=[1])`(`anymal.py:128`), front-back=`flip(dims=[2])`(`:169`)로 하드코딩 → parkour와 grid 형상·ordering 일치하므로 **그대로 `flip(dims=[1])` 적용 가능**.
- lateral offset이 0 (`pos=(0.375, 0.0, ...)` → y-offset 0)이므로 미러축이 깨끗(좌우 대칭 중심이 grid 중앙). x-offset 0.375는 forward 축이라 좌우 미러에 무관. **[추정→검증방법]**: scan 한 프레임을 `flip(dims=[1])` 후 좌우 평탄지형에서 동일해야 함 — smoke 학습 전 단위 assert 권장.

### B.4 joint/action 순서 (12 DOF) **[검증됨 부분 / 미검증-실행]**
- `_robot.data.joint_names` 순서 = PhysX articulation DOF 순서. **GPU 포화로 실행 검증 실패**(섹션 D).
- action 순서 = joint 순서와 동일 **[검증됨]**: `parkour_env.py:573` `self._processed_actions = action_scale*scaled + default_joint_pos` → `set_joint_position_target` (`:576`). action[j]가 joint[j]에 직접 매핑.
- hip joint 인덱스는 **런타임 name-match로 동적 산정** **[검증됨]** (`parkour_env.py:311-316`: `[i for i,n in enumerate(joint_names) if "hip" in n]`). 즉 코드가 순서에 비의존적. → **aug 함수도 하드코딩 대신 `env.unwrapped._robot.data.joint_names`에서 swap 인덱스를 구성하면 순서 ambiguity를 완전히 회피**(권장).
- feet 순서 **[검증됨, 런타임 assert로 보증]**: `parkour_env.py:1146` 주석 `[FL(0), FR(1), RL(2), RR(3)]`, `:207-210`이 매 실행 `_feet_ids[2:4]`가 RL/RR인지 assert. contact_filt(proprio 42:46)는 이 feet 순서.

---

## C. L/R mirror 변환 규칙 (B 기반 정성 설계)

> 단일 대칭 = 좌우(L↔R). num_aug=2. 좌표계: body frame, x=forward, y=left(+), z=up (IsaacLab 표준).
> "좌우 미러"는 y축 부호반전 + L/R 다리 교환.

### C.1 그룹별 변환 (요약 표)

| 그룹 / 구성요소 | 변환 | 근거 / 비고 |
|----------------|------|------|
| **policy[0] yaw_diff** | `*= -1` | **[추정]** heading 오차의 좌우 반전. 검증: 좌측 target→우측 target. anymal 무대응(velocity cmd 사용). |
| **policy[1] next_yaw_diff** | `*= -1` | 동상 **[추정]** |
| **policy[2:5] projected_gravity_b** | `* [1,-1,1]` | y(lateral) 반전. anymal `:116` 동일 |
| **policy[5] commands (forward)** | **불변** | forward-only(`parkour_env.py:910`). **[검증됨-부재]** vy/wz 항 없음 → task brief의 "vy/wz 부호반전"은 parkour엔 해당 없음(구현자 혼동 방지) |
| **policy[6:18] joint_pos−default** | L/R joint swap + **hip 부호반전** | C.2 |
| **policy[18:30] joint_vel*0.05** | L/R joint swap + **hip 부호반전** | C.2 (위치·속도 모두 동일 규칙) |
| **policy[30:42] actions** | L/R joint swap + **hip 부호반전** | C.2 |
| **policy[42:46] contact_filt** | foot swap FL↔FR, RL↔RR (부호 없음) | feet 순서 [FL,FR,RL,RR] → swap (0↔1, 2↔3) |
| **scan (187)** | `view(-1,11,17).flip(dims=[1])` | B.3, anymal `:128` 동일 |
| **priv_explicit[0:3] root_lin_vel_b*2** | `* [1,-1,1]` | y 반전. anymal `:112` 동일 |
| **priv_explicit[3:6] root_ang_vel_b*0.25** | `* [-1,1,-1]` | anymal `:114` 동일 (roll·yaw 반전, pitch 불변) |
| **priv_latent[0] base_friction** | 불변 | 스칼라 물성 |
| **priv_latent[1:5] foot_friction(4)** | foot swap (FL↔FR, RL↔RR), 부호 없음 | foot당 static 1값. **실측 4-dim (아래 함정 참조)** |
| **priv_latent[5] base_mass** | 불변 | |
| **priv_latent[6:9] base_com** | `* [1,-1,1]` (y 반전) | COM offset의 lateral 성분 반전 **[추정]** |
| **priv_latent[9:21] joint_stiffness_ratio** | L/R joint swap, **부호반전 없음** | **물성 크기** — hip 부호 절대 반전 금지 |
| **priv_latent[21:33] joint_damping_ratio** | L/R joint swap, **부호반전 없음** | 동상 |
| **history (N,10,46)** | 각 시간프레임에 policy 변환 동일 적용 | `[:, :, :]` 마지막 축이 proprio 46. 단 `history[:,:,0:2]`(yaw)는 env가 0으로 마스킹(`parkour_env.py:1020`)이라 부호반전 무해 |

> **★ priv_latent dim 함정 [검증됨 — 파일이 자기모순]**: `_get_observations`의 cat 주석(`parkour_env.py:998 "foot_friction (N,8)"`), cfg 주석(`parkour_env_cfg.py:420 "37"`, `num_priv_obs=43`)은 모두 **STALE**. 실행 코드 ground-truth:
> - `_foot_shape_indices`는 **shape (4,)** — foot당 first-shape 1개 (`parkour_env.py:264-270`, 주석 `# shape: (4,)`).
> - `foot_friction = _mat_all[:, self._foot_shape_indices, 0].reshape(N,-1)` → static만, 4개 index → **(N, 4)** (`parkour_env.py:947`, `, 0]` 슬라이스).
> ∴ **실제 priv_latent = base_friction(1)+foot_friction(4)+base_mass(1)+base_com(3)+stiff(12)+damp(12) = 33** (37 아님).
> 인덱스 출처 **[검증됨]**: `parkour_env.py:947`(foot 4), `:995-1005`(cat 순서), `:264-270`(shape 4).
> cfg `num_priv_obs=43`/`37` 주석은 증거 아님 — 러너가 encoder를 cfg가 아닌 실제 `obs[k].shape[-1]`로 사이징(`on_policy_runner_parkour.py:365-366`)하므로 잘못돼도 크래시 안 남. **반드시 Task 1에서 런타임 assert로 확정**(아래).

> **핵심 함정 [advisor 지적, 검증됨]**: priv_latent의 stiffness/damping ratio(13:37)는 **joint swap만, hip 부호반전 금지**. 이들은 게인 크기(magnitude)지 각도가 아니다. C.2의 joint-swap helper를 재사용하되 **hip 부호반전 분기는 제외한 별도 helper**를 써야 한다.

### C.2 joint L/R swap + hip 부호반전 규칙 **[검증됨 부호근거 / 미검증-인덱스]**

부호 근거 **[검증됨]**: Go2 기본자세에서 `.*L_hip_joint = +0.1`, `.*R_hip_joint = -0.1` (`unitree.py:77-78`) → **hip(abduction) 관절은 L/R이 부호 반대**. 따라서 좌우 미러 시 hip 성분은 swap 후 **부호반전 필요**. thigh/calf(pitch 회전, sagittal plane)는 좌우 대칭이라 **swap만, 부호 불변** (anymal `:245` 도 HAA만 부호반전, HFE/KFE는 front-back에서만).

swap 인덱스 산정 — **하드코딩 대신 동적 구성 권장 [검증됨 메커니즘]**:
```
joint_names = env.unwrapped._robot.data.joint_names   # 런타임 ground-truth
# 각 joint n에 대해 L↔R 짝을 이름으로 매칭:
#   "FL_hip_joint" ↔ "FR_hip_joint", "RL_..." ↔ "RR_...", thigh/calf 동일
# 'L'↔'R'만 치환한 이름의 인덱스를 perm[i]에 저장 → swap_idx 텐서.
# hip 여부도 ("hip" in n)로 판정 → sign_flip 마스크.
```
이렇게 하면 leg-grouped([FL_h,FL_t,FL_c,FR_h,...]) vs joint-type-grouped 중 **어느 실제 순서든 자동 대응**. (워크스페이스 선행 문서들이 순서를 두고 상충 — `parkour_indexing_audit.md`는 leg-grouped 주장, 미검증. 동적 구성이 이 분쟁을 회피.)

### C.3 action mirror **[검증됨 규칙]**
action은 joint 순서와 동일(C.1 B.4) → policy[30:42]와 **완전히 같은 변환**: L/R swap + hip 부호반전. `data_augmentation_func`의 actions 경로는 이 한 변환만 적용(anymal `:179-195` 패턴, 단 left-right만).

---

## D. body_names 정렬 확인 — **실행 검증 실패, 정적 분석으로 대체 [미검증-실행]**

- **실행 시도**: `_workspace/_tmp_parkour_jointorder_probe.py`(작성 후 삭제)로 isaac-5.1 / isaac-parkour 두 env에서 `Go2-Parkour-Direct-v0` init → `joint_names`/`body_names`/`_feet_ids`/`_hip_joint_ids` 출력 시도.
- **실패 원인 [검증됨]**: GPU 메모리 포화. `nvidia-smi`: 23.6/24.5 GB 사용 중, 다중 학습 프로세스 활성. PhysX backend가 `create_articulation_view`/`get_dof_velocities`에서 `NoneType`/`Failed to get DOF velocities` 예외. 환경 동역학 문제 아님 — 단순 자원 부족(다른 학습 방해 위험으로 강제 실행 회피).
- **정적 분석 결론**:
  - feet 순서 `[FL,FR,RL,RR]` **[검증됨, 런타임 self-assert로 보증]** (`parkour_env.py:207-210, 1146`).
  - hip 인덱스/joint 순서: env가 **이름 매칭으로 동적 산정**(`:311-316`)하므로 코드 자체는 순서 비의존. 단, **aug 함수의 swap 인덱스는 별도 산정 필요** → C.2의 동적 구성 방식 채택하면 contact-sensor vs articulation body_names 정렬 불일치 위험까지 제거(둘 다 이름으로 매칭).
  - contact_filt는 `_feet_ids`(contact sensor view) 기준(`parkour_env.py:900,914`), foot_friction은 `_foot_shape_indices`(articulation physx view) 기준(`:947`). **두 view의 foot 순서가 같은지 [미검증-실행]** — 정적으로 양쪽 다 `.*foot` 정규식 매칭이고 동일 USD라 일치할 개연성 높으나 **단정 회피**.
- **재실행 명령 (GPU 여유 시)**:
  ```bash
  /home/user/miniconda3/envs/isaac-5.1/bin/python <probe.py> --headless --num_envs 4
  # 출력 확인 항목: joint_names(12) 정확 순서, _hip_joint_ids, _feet_ids,
  #   robot.body_names vs contact_sensor.body_names의 foot 순서 일치 여부,
  #   default_joint_pos[0] (hip L=+0.1, R=-0.1 부호 확인)
  ```
- **구현 가드 권장 [추정]**: aug 함수 1회차 호출 시 `joint_names`로 구성한 swap_idx가 permutation인지(중복/누락 없음) assert. foot_friction swap도 `_foot_shape_indices`가 contact `_feet_ids`와 같은 leg 순서인지 1회 assert.

---

## E. 새 task 등록 방식 **[검증됨 패턴]**

### E.1 기존 등록 패턴 (`source/isaaclab_tasks/.../direct/parkour/__init__.py`)
SPO/LCP/MoE 3개 variant가 **동일 env_cfg + 다른 RunnerCfg**로 등록됨(`__init__.py:28-56`):
```python
gym.register(
    id="Go2-Parkour-Direct-SPO",
    entry_point=f"{__name__}.parkour_env:Go2ParkourEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.parkour_env_cfg:ParkourEnvCfg",         # 재사용
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2ParkourSPOPPORunnerCfg",  # 신규
    },
)
```
RunnerCfg 패턴 (`agents/rsl_rl_ppo_cfg.py:95-110` SPO 예): `Go2ParkourPPORunnerCfg`를 상속, `__post_init__`에서 algorithm 필드만 수정.

### E.2 Go2-Parkour-Symmetry 추가 결론
- **(a) 새 env_cfg 불필요** — 동역학/reward/contact_duty 전부 유지. 기존 `ParkourEnvCfg` 재사용 **[검증됨, 요구사항 일치]**. (obs 자체는 안 바뀜; aug는 알고리즘 내부에서만 발생.)
- **(b) 새 agent cfg 필요** — `Go2ParkourSymmetryPPORunnerCfg`를 `rsl_rl_ppo_cfg.py`에 추가, `Go2ParkourPPORunnerCfg` 상속 + `__post_init__`에서:
  ```python
  self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
      use_data_augmentation=True,
      use_mirror_loss=False,
      mirror_loss_coeff=0.0,
      data_augmentation_func=<parkour mirror func, 객체 또는 "module:func" 문자열>,
  )
  ```
  import 추가: `from isaaclab_rl.rsl_rl import RslRlSymmetryCfg` (현재 `rsl_rl_ppo_cfg.py:8-14`에 미포함 → 추가 필요).
- aug 함수 파일 위치 권장: `source/isaaclab_tasks/.../direct/parkour/mdp/symmetry.py` (anymal 패턴 `velocity/mdp/symmetry/anymal.py`과 동형). parkour 디렉토리에 `mdp/` 존재 여부는 구현 worker가 확인 후 없으면 신설.

---

## 구현 task 분해 제안 (의존성 포함)

> 모든 변경은 **알고리즘/cfg/등록 측**. `parkour_env.py` 동역학·reward 불변. validate-method 선행 권장(soft symmetry 타당성 + parkour 단일대칭 적정성).

### Task 1 — parkour mirror 함수 작성 (network/algo 계열 worker) ★ 핵심, 선행
- 파일: `source/isaaclab_tasks/.../direct/parkour/mdp/symmetry.py` (신규, 없으면 `mdp/` 신설)
- 산출: `compute_parkour_symmetric_states(*, env, obs, actions)` (시그니처는 A.2 의사코드).
- 의존: 없음(설계 본 문서 B/C). 단, swap 인덱스는 **C.2 동적 구성** 사용 → D의 실행 미검증 항을 우회.
- 검증 게이트(함수 내부 1회 assert): swap_idx permutation 무결성, **`obs["scan"].shape[-1]==187`, `obs["policy"].shape[-1]==46`, `obs["priv_latent"].shape[-1]==33`** (파일 주석 자기모순 — loud assert로 silent index-shift 차단), num_aug=2. priv_latent가 33이 아니면 즉시 실패시키고 실측값으로 인덱스 재산정.

의사코드:
```python
@torch.no_grad()
def compute_parkour_symmetric_states(*, env, obs=None, actions=None):
    u = env.unwrapped
    swap, hip_mask = _build_joint_perm(u._robot.data.joint_names)   # 동적, 캐시 권장
    foot_swap = [1, 0, 3, 2]   # FL,FR,RL,RR -> 좌우 swap (런타임 _feet_ids 이름으로 검증)
    if obs is not None:
        out = obs.repeat(2)                      # [원본; 미러] (TensorDict.repeat)
        out["policy"][B:]        = _mirror_proprio(obs["policy"], swap, hip_mask, foot_swap)
        out["scan"][B:]          = obs["scan"].view(-1,11,17).flip(dims=[1]).reshape(-1,187)
        out["priv_explicit"][B:] = _mirror_priv_explicit(obs["priv_explicit"])
        out["priv_latent"][B:]   = _mirror_priv_latent(obs["priv_latent"], swap, foot_swap)
        out["history"][B:]       = _mirror_history(obs["history"], swap, hip_mask, foot_swap)
    else: out = None
    if actions is not None:
        a = actions.repeat(2, 1)
        a[B:] = _swap_joints(actions, swap, hip_mask)   # swap + hip 부호반전
    else: a = None
    return out, a
# _mirror_proprio: idx 0,1 *= -1; 2:5 *=[1,-1,1]; 5 불변; 6:18,18:30,30:42 = _swap_joints(.,hip부호);
#                  42:46 = contact[foot_swap]
# _mirror_priv_latent (dim=33): 0 불변; 1:5 foot_friction[foot_swap]; 5 불변;
#                      6:9 base_com *=[1,-1,1]; 9:21 stiff, 21:33 damp = swap만(hip 부호 X)
```

### Task 2 — Symmetry RunnerCfg 추가 (cfg-worker)
- 파일: `source/isaaclab_tasks/.../direct/parkour/agents/rsl_rl_ppo_cfg.py`
- 산출: `Go2ParkourSymmetryPPORunnerCfg(Go2ParkourPPORunnerCfg)` + `RslRlSymmetryCfg` import.
- 의존: **Task 1** (data_augmentation_func 참조). E.2 코드.
- 주의: `use_data_augmentation=True`면 batch 2배 → `desired_kl` adaptive lr와 상호작용. hyperparam 변경은 별도(이 cfg에서 PPO 하이퍼는 상속 유지).

### Task 3 — gym.register (cfg-worker, Task 2 의존)
- 파일: `.../direct/parkour/__init__.py`
- 산출: `Go2-Parkour-Symmetry` 블록 추가(E.1 패턴, env_cfg=`ParkourEnvCfg` 재사용, rsl_rl_cfg=`Go2ParkourSymmetryPPORunnerCfg`).

### Task 4 — 검증 (validate-code → smoke)
- validate-code: obs 그룹 5개 전부 미러 적용 확인, num_aug=2, priv_latent stiffness/damping에 hip 부호반전 없음(C.1 함정), proprio 46 인덱스 정확.
- D 재실행(GPU 여유 시): joint_names/feet 순서 실측 → C.2 동적 swap이 기대대로 동작하는지 확인.
- smoke 학습(짧게): symmetry loss/logging 출력, batch 2배 반영, KL 발산 없음 확인.

### 의존성 그래프
```
Task1 (mirror func) ──> Task2 (RunnerCfg) ──> Task3 (register) ──> Task4 (validate+smoke)
        └────────────────────────────────────────────────────────────┘ (D 재실행은 Task4 내 병행)
```

---

## 미해결 / 검증 필요 (구현 worker에게 인계)
1. **[미검증-실행]** joint_names 실제 순서 + contact `_feet_ids` vs articulation foot 순서 일치 — D 재실행 또는 Task1 내부 assert로 해소. (동적 swap 채택 시 치명도 낮음.)
2. **[추정]** yaw_diff/next_yaw_diff 좌우 부호반전 — body-frame heading 오차 정의상 타당하나, 좌우 평탄지형에서 미러 obs로 정책 출력 좌우대칭 단위테스트로 검증 권장.
3. **[추정]** base_com y 부호반전 — COM offset lateral 성분. DR 비활성 시 0이라 무해, 활성 시 검증.
4. data-aug vs mirror-loss 모드 선택: 우선 data-aug. 메모리/KL 문제 시 mirror-loss(coeff 0.5~1.0 권장 시작)로 전환 — hyperparam-worker.

# hind_leg 환경 점검 종합 보고서

점검일: 2026-06-02
대상: `source/isaaclab_tasks/isaaclab_tasks/direct/hind_leg/`
로봇: `HIND_LEG_CFG` — 8관절 2족(뒷다리만) 보행기 (HL/HR × Hip/Thigh/Calf/Foot)
등록 환경: `HindLeg-Direct-v0` → cfg=`HindLegHistoryEnvCfg`, runner=`HindLegParkourPPORunnerCfg` (parkour RMA 스택)

> 점검 원칙: **Active(등록된 History 경로에서 실제 실행됨) vs Latent(미등록 Flat/Rough 전용)** 분리.
> 모든 주장에 [검증됨]/[가설] 라벨. 상세 근거는 동일 폴더의 `linchpin1_dimensions.md`, `linchpin2_bodynames.md`, `method_validity.md` 참조.

---

## 한 줄 결론

**(정적 분석 기반 — 실행 미수행)** 정적 분석상 등록된 유일 경로는 `__init__`에서 `ValueError`로 죽는다(body name typo, A-1). 이 한 줄을 고치면 실행은 되지만, 그 다음엔 reward 설계의 정적 attractor(feet_air_time 0.5s + base_height -10)로 "가만히 서기"에 수렴할 **위험(예측)**이 높다 — 이 코드베이스에 동일 메커니즘의 catastrophic failure 전력이 있어 근거가 약하지 않다.

> 본 리뷰는 코드/USD/URDF 정적 분석이다. smoke run을 돌리지 않았다. "학습 시작 불가"라는 결론 자체는 robust하다(init 단계 blocker가 A-1·C-2 등 복수). 다만 **그 crash 지점이 정확히 A-1**이라는 특정은 위 A-1 체인에 근거한다. 또한 reward 위험(B-1/B-2)은 측정값이 아니라 **이 gait에 대한 설계 예측 + 과거 전례**다.

전반적으로 **구조(parkour RMA 스택 연결, obs_groups, 차원 추론)는 의외로 견고**하나, **이름 정합성 1건(치명)과 reward 튜닝(고위험)**이 발목을 잡는다.

---

## A. 즉시 수정 필요 (Active 경로 = 등록된 History cfg에서 실제 실행)

### A-1. [CRITICAL · 검증됨] body name typo → `__init__` crash
- 위치: `hind_leg_env_cfg.py:272` `HindLegHistoryEnvCfg.penalzied_body_names`
- `"HR_Foot_Link_Link"` (Link 중복) — 실제 body 이름은 `"HR_Foot_Link"`
- 근거 체인 (전부 [검증됨]):
  1. 실제 body 11개에 `HR_Foot_Link`만 존재, `HR_Foot_Link_Link`는 없음 [URDF `Hind_Leg_URDF_v3/.../Hind_Leg.urdf`, `merge_fixed_joints:false`]
  2. `find_bodies(list)`는 인자를 그대로 `resolve_matching_names`에 전달 [`contact_sensor.py:180`]
  3. `resolve_matching_names`는 매칭 없는 패턴이 하나라도 있으면 `ValueError` raise (빈 리스트 silent 반환 아님) [`isaaclab/utils/string.py:259-268`]
- 결과: `self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(self.cfg.penalzied_body_names)` (env:75)에서 즉시 crash
- **수정**: `"HR_Foot_Link_Link"` → `"HR_Foot_Link"` (한 글자)
- 참고: `.*_Foot_Contact_Link`(env:73)는 정상 — 실제로 `HL/HR_Foot_Contact_Link` 2개 body 존재 [검증됨]. 초기 의심과 달리 이 패턴은 문제 없음.
- regression 여부: 형제 `r2s_hind_leg`는 **완전히 다른 로봇**(`R_SKELETON_HIND_LEG_CFG`, 5관절, ContactSensor 자체가 없음)이라 oracle로 못 씀 [검증됨]. 따라서 이 typo가 "복사 중 유입"인지 "처음부터"인지는 판별 불가 — 다만 두 정황(아래 헤드라인 주석)상 **아직 실행해보지 않았을 가능성**이 높음.

### A-2. [High · 검증됨] estimator를 step 0부터 actor에 주입
- 위치: `agents/rsl_rl_ppo_cfg.py:60` `train_with_estimated_states=True`
- 같은 줄 주석은 "estimator 수렴 전 게이트 off; 수렴 후 True 전환"이라 적혀 있으나 **실제 값은 처음부터 True** — 주석과 모순
- 미수렴 estimator 출력(추정 priv_explicit)이 학습 초기부터 actor 입력에 들어가 초기 학습 불안정 유발 가능
- **판단 필요**: 의도대로면 False로 시작하거나, RMA 2단계 학습 흐름을 명시적으로 확인

### A-3. [Medium · 검증됨] `torch.tensor(get_masses())` 매 step 재래핑
- 위치: `hind_leg_env.py:177-181` (priv_latent obs)
- `root_physx_view.get_masses()`는 이미 텐서(CPU) 반환 → `torch.tensor(...)`로 감싸면 매 step CPU→GPU copy + UserWarning
- 단, priv_latent는 **dead config**(A섹션 vs C-1 참조)이므로 실제로는 critic 입력으로만 쓰임 → 성능 손해만 있고 기능 오류는 아님. `.to(device)` 또는 캐싱 권장

### A-4. [Low · 검증됨] 디버그 잔존물
- `print(obs)` 류 잔존 (parkour runner:87), env:106 `# print(self._robot.joint_names)` 주석

---

## B. 고위험 — 실행은 되지만 학습이 "정적 attractor"로 수렴할 가능성 (Method)

총평: **RED**. reward 구조 안에 "가만히 서서 명령 무시"가 국소 최적이 되는 신호가 누적.

### B-1. [RED · 설계 smell(예측)] feet_air_time threshold = 0.5s — 과거 전례 있음
- `hind_leg_env.py:211` `air_time = sum((last_air_time - 0.5) * first_contact) * (||cmd||>0.1)`
- 예측: 2족 보행 air_time이 0.5s보다 짧으면 매 발걸음마다 음수 reward → 발을 안 들면 0, 들면 음수 → "발 들지 마" 편향 (※ 실제 gait air_time은 미측정 — 학습 로그로 확인 필요)
- 증거: 이 코드베이스 commit `fc1b0a874dd`에서 threshold=0.3s 버전이 동일 메커니즘으로 catastrophic failure 유발 → 0.5s는 더 빡빡한 방향
- 권장: 짧은 run의 feet_air_time 로그로 실제 air_time 분포 확인 후 threshold 재검토

### B-2. [RED · 설계 smell(예측)] base_height scale = -10 이 lin_vel tracking(scale 1.0) 대비 큼
- `cfg:325` base_height_reward_scale=-10.0, target=default z(0.6). 예측상 dz≈0.32m면 완벽한 속도추종 보상을 상쇄하는 크기
- 우려: 2족 보행의 자연스러운 CoM 상하 진동이 과하게 패널티화되어 정적 stance가 상대적으로 유리해질 수 있음 (※ 실제 진동폭 미측정)

### B-3. [YELLOW 누적] 기타 튜닝 신호
- exp tracking `sigma=0.1` (표준 0.25 대비 매우 빡빡): `cfg`/env:194,197 `/0.1`
- `flat_orientation_reward_scale=-0.0` (History에서 OFF) — 2족 자세 안정 신호 제거됨
- command `lin_vel_x=[-2.0, 1.0]` 비대칭(후진이 전진의 2배) + `command_curriculum=False` (커리큘럼 미사용)
- actuator damping DR `(0.3, 3.0)` = 10배 범위 — 2족에 과도할 수 있음
- `lin_vel_y=[0,0]`인데 reward는 `commands[:, :2]` 2D 추종 (y는 항상 0이라 무해하나 불필요)

### B-4. [미확인 우려] observation_noise_model이 priv/history 그룹까지 오염하는가
- `cfg:335` `observation_noise_model`(Gaussian std 0.002 + bias)이 DirectRLEnv에서 obs dict 전체에 적용되는지, "policy" 그룹에만 적용되는지 **미검증**
- 만약 전 그룹 적용이면 critic 전용 privileged 입력(priv_explicit/priv_latent)과 history에도 노이즈가 들어가 privileged 신호의 의미가 희석됨
- 권장: DirectRLEnv의 noise model 적용 범위를 확인 (이번 점검에서 추적 안 함)

상세/근거: `method_validity.md`

---

## C. Latent — 현재 무해(미등록), 그러나 해당 cfg를 쓰려면 고쳐야 함

### C-1. [정보 · 검증됨] priv_latent 차원 선언 불일치는 무해
- `HindLegHistoryEnvCfg`: `num_priv_latent=5` 선언 vs 실제 obs ~36차원
- **crash 아님**: 활성 스택(`OnPolicyRunnerParkour`/`ActorCriticRMA`/`PPOParkour`, train.py:88 import 경로)은 cfg의 `num_priv_latent`를 읽지 않고 **env가 반환한 실제 텐서 shape에서 차원을 런타임 추론** [검증됨]
- 주의: `*_parkour_original.py` 모듈만 cfg 차원을 소비하나 **그 경로는 import되지 않음** → dead config일 뿐
- obs_groups 5개(policy/critic/history/priv/priv_explicit)가 참조하는 키가 모두 env obs dict에 존재 → 계약 통과. policy=30, priv_explicit=6 정합 [검증됨]

### C-2. [High · 검증됨] Flat/Rough cfg는 사용 시 crash (현재 미등록)
- `HindLegFlatEnvCfg.penalzied_body_names` (cfg:122-146): `FL_link_1`~`HR_link_5` 등 존재하지 않는 이름 23개 → 사용 시 `find_bodies` ValueError
- `HindLegFlatEnvCfg.action_space=26` — 실제 로봇 8관절과 불일치 (History는 8로 정확)
- `termination_reward_scale`은 History cfg에만 존재 → Flat/Rough에서 `_get_rewards` 접근 시 AttributeError
- → Flat/Rough는 미완성 템플릿 잔재. 현재 등록 경로가 아니므로 **지금은 무해**하나, 쓰려면 전면 정리 필요

---

## D. 정상 확인됨 (오탐 방지 — 플래그하지 않음)

- action_rate 계산 순서 정상 [검증됨]: step 순서상 `_previous_actions`가 reward 시점에 `a_{t-1}` 담음
- `action_space=8` = 로봇 8관절 정확히 일치
- `.*_Foot_Contact_Link` 매칭 정상 (2개 body 존재)
- `find_bodies("base")` 단일 매칭 정상
- priv_explicit(6), policy obs(30) 차원 정합
- isinstance가 History를 Rough로 안 잡음 → scan 미사용이라 무해

---

## 권장 조치 순서

1. **A-1 즉시 수정** (1글자) — 이거 없이는 아무것도 안 돌아감
2. 짧은 smoke run으로 `__init__`/첫 forward 통과 확인
3. **B-1, B-2 reward 재튜닝** — 학습 품질의 핵심. 특히 feet_air_time은 전례가 있으니 우선
4. A-2(estimator 게이팅) 의도 확정
5. (선택) C-2 Flat/Rough 정리 또는 삭제

# Go2 Parkour 3-Leg Gait — 종합 조사 보고서

**대상 run**: `logs/rsl_rl/go2_parkour/2026-06-01_14-06-08_use_default_usd` (~13,800+ iter)
**증상**: parkour 지형 극복 시 다리 3개만 사용하는 비대칭 보행 (RL = rear-left 발 접지율 낮음)
**조사일**: 2026-06-02 · 3-worker 팀 (web-isaac / web-rl / code)

---

## 0. TL;DR

1. **Isaac Sim 5.1에 "3-leg gait를 직접 유발하는 확정 엔진 버그"는 없다.** (5.1 release notes/known issues 어디에도 legged/feet 항목 없음)
2. **이건 시뮬레이터 버그가 아니라 RL local optimum 문제다.** velocity reward만 강하고(reward의 ~80%) 4발 대칭을 요구·보상하는 항이 실질적으로 없어서, 정책이 "3다리로도 전진 가능한" 국소 최적해에 고착.
3. **`use_default_usd` 실험이 이를 뒷받침.** custom USD 비대칭을 의심해 default Go2 USD로 바꿔도 여전히 3-leg → 에셋 비대칭 가설은 약화, 원인은 reward/learning 쪽.
4. **권장 first move 순서(코드 검증 후 정정됨):** ① 저비용 **검증** (운동학 vs contact 접지 비교 + use_default_usd run 새 attribution 캡처) → ② **발별 air-time 상한 패널티** + **gait_pairing 재활성** (contact-기반, deploy-safe) → ③ **Symmetry Augmentation** (leg-agnostic이라 가치 높으나 **parkour용 미러 매핑 함수 신규 구현 필요 → 중간~높은 비용**, 단순 플래그 아님).

> ⚠️ **초기 권고 정정**: web 조사 단계에서 "Symmetry Augmentation = deploy 리스크 0, 1순위"였으나, 코드 검증 결과 parkour에는 `data_augmentation_func`(좌우 미러 매핑)가 **없어서** 플래그만 켜면 에러. deploy 리스크는 0이지만 **구현 비용/정합성 리스크는 0이 아님.** 따라서 1순위에서 내려옴.

---

## 1. 외부 케이스 조사 결과

### 1-A. Isaac Sim / Isaac 계열 (worker-web-isaac)
- **5.1 specific 엔진 버그: 발견 안 됨.** (confirmation bias 경계, 억지 연결 안 함)
- 정황/메커니즘 후보:
  - **[HIGH·미검증]** GPU + **trimesh terrain**에서 `net_contact_force_tensor`가 triangle edge에서 contact force를 누락/축소 (legged_gym 원조 이슈 + IsaacLab Discussion #2697). parkour는 정확히 trimesh rough terrain. 사실이면 contact 기반 gait reward가 **다리별로 비대칭 왜곡** → 정책이 "신호 깨끗한 다리"만 사용. NVIDIA는 `net_contact_force` deprecate, `ContactSensor`/`body_incoming_joint_wrench_b` 권장.
  - **[MEDIUM]** Issue #4014: Isaac Sim **5.1.0**에서 G1(휴머노이드) 한쪽 다리 접고 스킵하는 single-leg 이상 보행이 default 학습에서 관측 (open). → "5.1 default 학습에서 single-leg 이상 보행이 드물지 않다"는 정황 증거.
  - **[MEDIUM]** Go2 sim2real #1784: 발 dragging (friction 모델링, 5.0/5.1에서 friction 모델 변경됨). 단 대칭 dragging이라 3-leg 직접 근거 약함.

### 1-B. 일반 Quadruped RL (worker-web-rl)
3-leg/비대칭 gait은 거의 항상 **"velocity reward만 강하고 gait shaping 부재 → 3다리 local optimum"**. 5개 해결책 카테고리(모두 출처 확보):
1. **Gait/periodic reward** — Siekmann 2021 first-contact gating + per-leg phase clock. swing/stance contact penalty로 끌기 차단.
2. **★Symmetry augmentation/loss** — IsaacLab `RslRlSymmetryCfg` 기본 지원 (`use_data_augmentation`/`use_mirror_loss`/`mirror_loss_coeff`). **학습 전용 → deploy 리스크 0. 1순위.**
3. **feet_air_time 함정** — IsaacLab Discussion #1977: first_contact gating 없거나 weight 과도하면 "아예 안 딛는" 정책으로 붕괴(공식 미해결). feet_air_time 단독은 좌우 균형 보장 못함.
4. **Curriculum** — promotion 과공격적이면 "버티기" 도피 gait. flat warm-up + 성능기반 demotion 권고.
5. **Exploration** — PPO local optimum 고착. entropy↑, action smoothness penalty 과다 점검(과하면 minimal-motion 도피).

---

## 2. 코드 내부 분석 결과 (worker-code, 제약 7항목 준수)

### 검증으로 기각된 가설
- **air_time 무한적립 버그: 현재 parkour env에 없음.** raw air-time 직접 보상 항 없음. air_time은 `feet_gait_pairing`에만 쓰이고 그마저 `scale=0.0` + `is_flat` gate. (git의 fc1b0a 버그는 다른 경로)
- reset 자세 대칭(env:1312), hip-scale 대칭(560) — 대칭 확인. clip(min=0)은 의도된 설계.

### 실증 (attribution npz)
- ⚠️ **수치 출처 정정**: 재활용한 attribution npz는 mtime 2026-05-27, **이번 `use_default_usd` run(2026-06-01)이 아닌 이전(≤5/27) 체크포인트**. 따라서 아래 접지율은 "이전 체크포인트" 기준.
- 이전(≤5/27) 체크포인트: 5개 지형 롤아웃 전부에서 **RL(rear-left) 발 접지율 0.18~0.28 vs 나머지 0.5~0.66** → 학습된 3-leg 비대칭. (지형당 단일-env 롤아웃이라 "RL 다리 특이"는 모집단 단정 불가 — 핵심은 "한 다리로 깨진다"는 것 자체.)
- **use_default_usd run의 3-leg는 사용자 육안 관찰로 재현 보고됨.** 단 어느 다리/정확한 수치는 run마다 다를 수 있음 → 이 leg-agnostic 성격이 symmetry 류 해결책의 가치를 높임(단 구현 비용은 별개, §3-③). **새 attribution 캡처로 use_default_usd run을 직접 재측정 권장.**
- reward의 **~80%가 `tracking_goal_vel`(+0.030) + `tracking_yaw`(+0.008)**. gait/effort 규제항은 전부 |≤2e-3|. → **4발 대칭을 요구·보상하는 항이 실질적으로 없음.** (구조적, run 무관)

### 유발 후보 (가설, 데이터 뒷받침)
| ID | 후보 | 위치/근거 |
|----|------|-----------|
| A | `feet_gait_pairing * is_flat` → 장애물 지형에서 gait shaping = 0 | env.py:1178 |
| B | `feet_gait_pairing scale=0.0` → 평지에서도 비활성 | cfg:623 |
| C(보조) | `feet_dragging` hind-only+contact-gated → 든 뒷발 막는 항 *부재*(permissive, 능동 인센티브 아님) | — |
| D | **PPO symmetry augmentation 부재** (agent cfg grep 0건) → 대칭파괴 복원 압력 없음 | agent cfg |

---

## 3. 권장 해결책 (Deploy-safe, 제약 self-check 통과)

> 모두 contact 정보를 **reward 내부에서만** 사용. policy obs/proprio 추가 0 (제약 §1 준수). scale 단독 튜닝/actuator·latency·height_scan blame 없음.
> **순서는 코드 검증(Q1~Q3) 후 정정됨.** contact-기반 fix(①②)는 신호 신뢰도 게이팅 필요, symmetry(③)는 구현 비용 큼.

### 우선순위 0 — 저비용 검증 먼저 (반나절, 비용 거의 0) 🔍
1. **contact 신호 신뢰도 검증 (§E 게이팅)**: 발별로 ContactSensor 접지 bool vs 운동학적 접지(발 z vs height_scan 지면) 비교. 한 다리만 체계적 불일치면 trimesh edge contact-underreporting 시사 → 그 경우 ①②를 운동학 기반 신호로 전환해야 함(역효과 방지).
2. **use_default_usd run 새 attribution 캡처**: 기존 npz는 ≤5/27 이전 체크포인트라 라벨 불가. 이번 run에서 per-foot 접지율 직접 재측정해 어느 다리/얼마나 깨지는지 확정.

### 우선순위 1 — 발별 air-time 상한 패널티 신규항 (저비용, 단 우선순위0-1 선행)
- `current_air_time`을 **reward 내부에서만** 사용, 4발 대칭, `max_single_air_time_s ≈ 0.5`. 영구히 든 다리를 직접 봉쇄.
- 제약 §1 준수(obs 추가 아님), §2의 "구조 변경 동반"이므로 단독 scale 튜닝 아님.
- ⚠️ contact-기반이라 §E 검증 결과에 따라 운동학 신호로 구현 (trimesh edge 강건).

### 우선순위 2 — `feet_gait_pairing` 재활성 + `is_flat` gate 제거 (저비용)
- 기존 자산 재활용. 현재 `scale=0.0`(cfg:623) + `* is_flat`(env:1178)로 사실상 비활성 → 장애물 지형에서도 gait shaping이 살아있도록. parkour task 다양성(점프/등반) 고려해 soft phase 또는 swing/stance penalty 형태로 제한 권고(web-rl 주의사항).

### 우선순위 3 — PPO Symmetry Augmentation (leg-agnostic, deploy 리스크 0, **단 구현 비용 큼**)
- **근거**: web-rl(#1)·worker-code(D) 양쪽 독립 수렴. 어느 다리가 깨지든 무관하게 작동 → run마다 깨지는 다리가 달라도 유효.
- **인프라 있음**: PPOParkour가 `symmetry_cfg` 받음(ppo_parkour.py:54-102), OnPolicyRunnerParkour가 `resolve_symmetry_config` 호출(:337).
- ⚠️ **그러나 `data_augmentation_func`(좌우 미러 매핑)가 parkour에 부재(grep 0건) → 필수 필드 MISSING → 플래그만 켜면 에러.** parkour obs(proprio 42 + height_scan 187 격자 + priv + history 10×42)·12-dim action에 맞는 미러 매핑(관절 좌↔우 permutation+부호, height_scan 187 격자 좌우반전, lateral cmd/yaw 부호)을 **신규 구현해야 함 → 중간~높은 난이도/리스크**. 미러 함수를 잘못 짜면 학습이 조용히 망가짐.
- deploy 리스크는 0(학습 시점만, 정책 구조/obs 불변).

### 보조 후보 — 후보 E (trimesh edge contact-underreporting, 미검증 가설)
- (a) terrain = trimesh 확정(env_cfg:457). (b) reward의 contact는 ContactSensor 경유지만 내부가 동일 PhysX GPU tensor를 래핑(core:373) → **edge-underreporting 면역 안 됨**. 메커니즘 코드상 성립 가능.
- 단 반대 증거 3: ①edge-miss는 *착지 위치*(공간)의 함수지 *다리*의 함수 아님 ②평탄영역에서도 RL 저접지 지속 ③영향 항 가중치 ≤2e-3로 leverage 거의 없음. → 주원인 확률 낮음, 배제는 안 함. **우선순위0-1 검증으로 판정.**

---

## 4. 한 줄 요약

> **5.1 엔진 버그 아님. velocity-only reward(reward의 ~80%)의 3-leg local optimum.** 4발 대칭을 보상하는 항이 사실상 없음.
> first move = **저비용 검증**(운동학 vs contact 접지 비교 + use_default_usd run 새 attribution) → **발별 air-time 상한 패널티 + gait_pairing 재활성**(contact-기반, 검증 결과에 따라 운동학 신호로) → **Symmetry Augmentation**(leg-agnostic, deploy 리스크 0이나 parkour용 미러 매핑 함수 신규 구현 필요 = 고비용).
> "symmetry 플래그만 켜면 끝"은 **틀림** — parkour엔 `data_augmentation_func`가 없어 구현 필요.

**상세 보고서**: `web_isaac_issues.md` · `web_rl_methods.md` · `code_internal_solutions.md` (동일 디렉토리)

---

## 5. 실측 검증 결과 (순위 0 완료, 2026-06-02)

검증 스크립트 `scripts/reinforcement_learning/rsl_rl/verify_3leg_contact.py` (v3, height_scanner 기반 운동학 ground-truth)로 측정. 로그: `v3_run.log`, 결과: `rank0_verify_result.{md,npz}`.

### 검증 과정의 함정 2개 (기록용)
- **운동학 검출기 v1 버그**: 초기엔 `_edge_mask_height_field`(edge 국소 필드)를 ground-z로 써서 4발 중 RR만 동작 → height_scanner `ray_hits_w` 최근접 + 발별 sole_offset 자동보정으로 수정.
- **잘못된 체크포인트 자동선택**: "최신 체크포인트" 자동선택이 오늘 새로 시작된 `2026-06-02_12-32-54_use_default_usd_4.5/model_400`(400 iter 아기 정책)을 잡아 대칭 결과를 냄. `--checkpoint`로 올바른 run(`2026-06-01_14-06-08_use_default_usd/model_21900`)에 **고정**해서 해결. → 교훈: 원본 run이 **아직 학습 중**(model_21900 계속 증가)이므로 비교 시 체크포인트 고정 필수.

### 확정 결과 (성숙 정책 model_21900)
| foot | sensor_ratio | kin_ratio | both | neither |
|------|---|---|---|---|
| FL | 0.674 | 0.373 | 0.282 | 0.236 |
| FR | 0.513 | 0.462 | 0.232 | 0.257 |
| **RL** | **0.106** | **0.124** | 0.041 | **0.811** |
| RR | 0.602 | 0.612 | 0.585 | 0.371 |

- **(A) 3-leg 확정 — RL(rear-left)이 dropped foot**: sensor 0.106 vs 나머지 0.51–0.67, asymmetry std=0.22. v1/v2(model_21300)와 일치, 사용자 육안과 일치, **정책 의존적**(아기 정책 model_400은 대칭이었음 → reward/policy가 원인이지 엔진/에셋/센서 아님). scanner coverage 정상(ray_valid=100%).

- **(B) edge-underreporting 기각 → contact 기반 fix 안전**: 발별 calibration이 RL 판정 임계를 "지면 0.18m 이내=접지"로 **느슨하게** 잡았는데도(즉 접지를 **과대**계상만 가능) kin_ratio[RL]=0.124. 따라서 RL 실제 접지 ≤12.4%, sensor는 독립적으로 10.6%. **느슨한 검출기는 dropped foot를 숨길 수 없고 접지를 부풀릴 뿐인데도 접지가 여전히 작음 → RL-dropped 견고, 센서가 RL 접지를 누락하는 것 아님.** 만약 센서가 누락했다면 loose kin이 sensor보다 훨씬 큰 접지를 보여야 하나 차이는 1.8pp뿐. ⇒ **순위 1/2 fix를 contact 신호 기반으로 구현 OK.**

- **부수 관측(결정에 무관, 추적 안 함)**: FL/FR/RL 발이 nearest-ground 대비 0.15–0.33m 위에 위치(RR만 0.02m). 아기/성숙 두 정책에서 거의 동일 → 학습된 gait가 아니라 발 body-origin offset/spawn 자세 등 **측정 특성**으로 추정. sensor_ratio는 높이와 독립이라 (A) 결론에 영향 없음. 닫고 싶으면 flat에서 `foot_z vs base_z` 2분 점검(선택, critical path 아님).

### 결론 → fix 방향
- RL이 **일관되게**(model_21300·21900 공통) dropped → leg-agnostic한 symmetry augmentation의 가치는 상대적으로 덜 결정적. **targeted 발별 air-time 상한 패널티(순위 1, contact 기반으로 clear됨)가 RL을 직접 잡는 싸고 정확한 first move.** symmetry(순위 3)는 더 견고하나 parkour용 mirror 함수 신규 구현 비용 큼.
- ⚠️ post-fix 비교 전 반드시 체크포인트 고정. 원본 run 학습 진행 중 + 새 `use_default_usd_4.5`(Isaac 4.5?) run 존재.

---

## 6. 구현 완료 (순위 1 — air_time_cap, 2026-06-02)

`max_air` 데이터 분석(`max_air_analysis.md`) → reward-worker 구현 → validate-code **PASS**.

### max_air 결정
- step_dt = 0.02s (50Hz). 정상 발(FL/FR/RR) air-time p99=0.36s/max=1.70s(gap 도약), RL p90=1.24s/max=5.60s.
- **anchor 역전**: 정상 max(1.70s) > RL p90(1.24s) → 깨끗한 분리 불가 → **graded penalty 필수**.
- **`max_air = 1.0s`** 채택 (graded). 정상 도약 0.008%만 걸림, RL 병리꼬리 13.3% 처벌.

### 코드 변경 (parkour env)
- `parkour_env.py:1199-1228` — `_get_rewards()`에 `air_time_cap = Σ_4발 clamp(current_air_time[foot] − air_time_cap_max_s, min=0)` 추가. 기존 `air_time` 핸들 재사용, 4발 대칭, velocity-ungated.
- `parkour_env_cfg.py:630` — `reward_scales["air_time_cap"] = -0.1` (penalty). `:648` — `air_time_cap_max_s = 1.0`.
- 검증: obs_space 불변(air_time는 reward 내부만, §1 준수), 새 buffer 없음(ContactSensor 관리), 키 정합 17개, 부호(value≥0×weight<0=penalty≤0), 기존 scale 불변. py_compile/validate-code PASS.

### 조정 레버 (효과 부족 시, 다음 이터레이션)
- weight `-0.1 → -0.25` (압박 강화) 또는 `air_time_cap_max_s 1.0 → 0.6s` (RL 25.6% 처벌, 정상 도약 약간 희생).
- 한계: air-time cap은 "연속 공중 >max_air"만 막음 → RL이 ~0.9s마다 톡 딛으며 절뚝거리면 회피 가능. 그 경우 순위 3(symmetry) 추가 필요.

### 다음: 학습 + 검증 루프
1. 수정된 reward로 학습 (resume from model_21900 또는 fresh — 사용자 결정).
2. verify_3leg_contact.py를 **고정 체크포인트**로 재측정 → RL sensor_ratio가 0.106 → 0.4~0.5로 회복하는지 확인.

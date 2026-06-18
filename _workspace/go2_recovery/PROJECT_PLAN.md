# Go2 Fall-Recovery 프로젝트 계획서

> 목표: 넘어진 Go2를 default pose로 **안전하게(천천히·저jerk)** 복구하는 정책 학습.
> 참조: `/home/lgb/RobotSW_Genesis/recovery` (Genesis 기반 Go1 recovery).
> 작성일: 2026-06-16 · 멀티에이전트 조사(참조분석/Go2자산/안전복구 리서치) 종합.

---

## 0. 한 줄 요약

기존 recovery reward는 "빨리 일어서기"에 최적화되어 deploy 시 급격한 동작이 위험.
→ **사용자 원안(고정 T초 느린 복구)을 HumanUP식 2-stage로 디벨롭**하여
"복구 성공률(Stage I)"과 "동작 속도/안전(Stage II)"을 **분리 학습**한다.
단, Stage I/II는 1주차에 바로 가지 않고, **Tier 0~1(단일 env)로 baseline을 먼저 확보**한 뒤 단계적으로 올린다.

---

## 1. 참조 프로젝트(Genesis Go1)에서 가져올 것 / 버릴 것

| 항목 | 참조 구현 (Genesis Go1) | IsaacLab Go2 이식 방침 |
|------|------------------------|------------------------|
| Fall 초기상태 | 80% 공중낙하(0.5~0.6m)+Euler roll±135°/랜덤 joint, 10% standing, 10% sitting, init_step=100 물리수렴 | **그대로 채택**. `write_root_pose_to_sim`+`quat_from_axis_angle`로 구현 |
| 복구 reward | `_reward_reset` = 0.5·r_roll(upright) + 0.5·r_stand(0.2·height+0.6·pose+0.2·vel) | **구조 채택**. Go2 default pose(h=0.27m, calf −1.5)로 목표값 교체 |
| 안전 메커니즘 | Hip action scale 0.5×, action_smoothness 1/2차, dof_acc/torque penalty | **채택**. 단 "직접 속도제한 없음"이 약점 → 본 프로젝트가 보강하는 지점 |
| Termination | height < −0.25m or 시간초과 (성공판정 explicit 없음) | 채택 + **success 명시판정 추가**(upright & near-default 유지 N step) |
| 알고리즘 | PPO(ActorCriticRMA), obs 451(history10+priv_latent22) | **rsl_rl PPO**. 초기엔 priv/history 없이 단순 obs로 시작, 필요시 확장 |

**핵심 차이점(우리가 더하는 것)**: 참조 프로젝트는 속도/안전을 *간접 penalty*로만 다룸. 본 프로젝트는 **시간 파라미터화 + curriculum + (선택)keyframe tracking**으로 동작 속도를 *명시적으로* 제어한다.

---

## 2. IsaacLab 구현 기반 (조사 확정)

- **로봇 cfg**: `UNITREE_GO2_CFG` @ `source/isaaclab_assets/isaaclab_assets/robots/unitree.py:140-184`
  - default pose = 복구 목표: pos z=0.27, hip ±0.1, thigh 0.8/1.0, calf −1.5
  - actuator(DCMotor): effort 23.5, vel_limit 30.0, Kp 25, Kd 0.5, armature 0.01
- **베이스 템플릿**: `Go2WTW`/`go2_env.py`(가장 단순한 flat-ground) → 복구 env로 개조
- **신규 디렉토리**: `source/isaaclab_tasks/isaaclab_tasks/direct/go2_recovery/`
  - `__init__.py`(gym.register `Go2Recovery-v0`), `go2_recovery_env.py`, `go2_recovery_env_cfg.py`, `agents/rsl_rl_ppo_cfg.py`, `CLAUDE.md`
- **Fall 초기화 API**: `write_root_pose_to_sim`(pos+quat) / `write_root_velocity_to_sim` / `write_joint_state_to_sim`, `isaaclab.utils.math.quat_from_axis_angle`
- **DebugViewer**: 필수 3줄 패턴(`_common`) — fall/upright 인디케이터 시각화 등록
- **불변규칙**: `source/isaaclab/` 코어 수정 금지, 새 buffer는 `_reset_idx` 초기화, `python` 직접실행 금지(`./isaaclab.sh -p`)

---

## 3. 안전·저속 복구 방법론 (사용자 아이디어 디벨롭)

### 3.1 사용자 원안 평가
- **직관(맞음)**: 복구시간↑ → 단위시간 운동량↓ → 안전.
- **구현상 함정**: 고정 T를 *episode 종료 + sparse reward*로 두면 탐색 실패, 자세 다양성에 경직(쉬운자세 낭비/어려운자세 실패), "빠른복구 vs 느린동작"이 reward에서 충돌.

### 3.2 채택 스택 (Tier 0 → 2, 점진 적용)

**Tier 0 — 즉시(1~2일, 난이도 하): 안전의 하한선**
1. **Actuator/velocity 단 제한**: `velocity_limit`을 locomotion의 50~70%로, Kd 1.5~2× ↑ → 급격 토크 출력 억제 (reward 독립, sim-to-real gap 자체 축소)
2. **Smoothness penalty 강화(FR-Net 값 기준)**: `action_rate_l2 −0.05`, `dof_acc_l2 −5e-6`, `dof_vel_l2 −5e-3`, `torque −5e-4`. 성공률 drop 시 `action_rate` 우선 완화.
3. **Action clip 축소**: ±0.25 → 필요시 ±0.3 내, hip scale 0.5×(참조 채택)

**Tier 1 — 핵심(1주, 난이도 중): 시간 제어 진입**
4. **Progressive Penalty Curriculum**: 학습 step 기준 penalty weight를 3구간 자동 증가(초기 약→후기 강). reward 항 추가 없이 cfg weight만 변경(HiFAR 방식). → 탐색은 초반에 허용, 부드러움은 후반에 확보.
5. **(선택) Keyframe Dense Tracking 간소화**: prone/supine별 중간자세 2개만 정의, 현재 pose와 quaternion 오차를 dense reward로. subgoal로 탐색효율↑ + 단계별 속도 제어.

**Tier 2 — 품질 극대화(2~3주, 난이도 상): 사용자 "T초"의 정석 구현**
6. **Two-Stage Curriculum (HumanUP 패턴) ★권장 최종형**
   - Stage I: weak penalty, 시간제한 X → 빠른 복구 동작 **발견**
   - 성공 에피소드 rollout 기록 → **시간 보간으로 6~8초로 연장**(= 사용자의 "T초")
   - Stage II: 연장 궤적을 strong penalty + dense tracking으로 추적
   - 프로젝트에 이미 `go2_imitation` AMP 파이프라인 존재 → Stage II를 **AMP discriminator로 대체 가능**(demo=느린 getup 궤적)

**보완재(항상 병행)**
- Domain Randomization(stiffness/damping ±30%, payload, slope) → robust = 여유 동작 = jerk↓
- Recovery 전용 obs: linear velocity 추정 제거, roll/pitch/ang_vel/dof_pos/dof_vel만(HumanUP)

### 3.3 방법별 Trade-off (조사 종합)

| 방법 | 성공률 | 부드러움 | 난이도 | 시간제어 직접성 |
|------|:---:|:---:|:---:|:---:|
| Two-Stage Curriculum | ★★★★★ | ★★★★★ | 중-상 | 높음 |
| Progressive Penalty | ★★★★ | ★★★★ | 하 | 낮음 |
| AMP Motion Imitation | ★★★★ | ★★★★★ | 상 | 데모의존 |
| Action/vel 제한 | ★★★ | ★★★ | 하 | 낮음 |
| Keyframe Dense Track | ★★★★ | ★★★★ | 중 | 높음 |
| 사용자 원안(고정T sparse) | ★★ | ★★★ | 하 | 높음(경직) |

---

## 4. 단계별 로드맵 (마일스톤)

### M1. Env 골격 + Fall 초기화 (1~2일)
- `go2_recovery/` 생성, `Go2Recovery-v0` 등록
- `_reset_idx`: 공중낙하+랜덤 orientation+랜덤 joint (참조 비율 80/10/10)
- obs(고유감각 전용), `_get_dones`(height/tilt+timeout), DebugViewer 3줄
- **검증**: smoke run(num_envs 소수)로 로봇이 실제 넘어진 상태로 초기화되는지 viewer 확인

### M2. 복구 reward + Tier 0 안전 (2~3일)
- `_reward_reset`(roll+stand) 이식, Go2 default pose 목표화
- Tier 0(actuator vel limit↓, smoothness 강화, action clip) 적용
- success 명시판정 추가
- **검증**: validate-code(shape/buffer) + 단기 학습으로 복구 성공률 곡선 확인

### M3. Tier 1 Curriculum + 안전 정량화 (1주)
- Progressive penalty curriculum, (선택)keyframe 2개
- **deploy 위험도 지표 정의·측정**: peak joint velocity, peak torque, jerk(dof_acc), CoM 속도, 복구 소요시간 → 학습 로그에 기록
- **검증**: training-evaluator로 "성공률 vs 안전지표" trade-off 평가

### M4. Tier 2 Two-Stage (옵션, 2~3주)
- Stage I 정책 → 성공 rollout 기록 → 시간보간 6~8초 → Stage II 추적(또는 AMP)
- **검증**: 동일 안전지표로 Tier 1 대비 동작 부드러움 개선 정량 비교

### M5. Sim-to-real 강건화 (병행)
- Domain randomization 강화, action latency 모델링(참조 채택), 최종 deploy export

---

## 5. 확정된 의사결정 (2026-06-16 사용자 확인)

1. **목표 우선순위 = 균형**: 성공률을 크게 희생 않는 선에서 최대한 부드럽게. → **Tier 0~1까지 적용 후 안전지표 보며 weight 조정**. Tier 2는 균형이 안 맞을 때만 진입.
2. **복구 소요시간 = 6~8초**: HumanUP 기준 느린 복구. → Stage II 타임스케일 6~8초, 안전 KPI에 "복구 소요시간 6~8초 도달" 명시.
3. **AMP = 미사용, 순수 RL curriculum**: demo 생성 병목 회피. Tier 2도 progressive penalty + 2-stage *RL* tracking으로 구현(AMP discriminator 경로 제외).
4. **착지 범위 = 임의 자세 전체**: 참조와 동일하게 roll±135°·랜덤 joint·랜덤 yaw 전체 커버.
5. **로봇 = Go2 확정**: default pose/actuator는 Go2 기준 이미 반영.

### 확정 경로 (실행 순서)
M1(env골격, 임의자세 fall init) → M2(reward + Tier0 안전) → M3(Tier1 progressive penalty curriculum + 안전지표 정량화) → **여기서 균형 평가** → 불충분 시 M4(순수 RL 2-stage, 6~8초 타임스케일). AMP·keyframe은 보류(필요 시 재논의).

---

## 6. Worker 위임 매핑 (구현 단계)

| 작업 | Worker |
|------|--------|
| env 골격/reset/obs/done | `obs-worker` |
| 복구 reward + smoothness 항 | `reward-worker` |
| env_cfg / actuator vel limit / penalty weight / curriculum | `cfg-worker` |
| PPO 하이퍼파라미터 | `hyperparam-worker` |
| (Tier2) AMP discriminator/network | `network-worker` + `loss-worker` |
| 정적 검증 | `validate-code` / 큰 설계변경 시 `validate-method` |
| 학습 평가/피드백 | `training-evaluator` → `feedback-generator` (`/locomotion-loop`) |

---

## 부록. 핵심 코드 근거
- 참조 fall init: `legged_env_recovery.py:923-1001`, reward: `:1197-1276`, smoothness: `:1284-1294`
- Go2 cfg: `unitree.py:140-184` / 베이스 env: `direct/go2/go2_env.py`
- Fall API: `write_root_pose_to_sim`, `quat_from_axis_angle`(`isaaclab/utils/math.py`)
- 논문: FR-Net(2509.11504, Go2 reward 수치), HumanUP(2502.12152, 2-stage 슬로우다운), HiFAR(2502.20061, stage penalty)

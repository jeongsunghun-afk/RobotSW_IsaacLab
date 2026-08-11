# 2026-08-11_13-58-01_depen3_stock

한 줄 목적: **`max_depenetration_velocity` 1.0 → 3.0**(PhysX 스키마 기본값)으로 올렸을 때
고속 구간이 열리는지 본다. **스톡 플랜트** 쪽. `implicit_stock` 대비 **변수 1 개**.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-11_13-58-01_depen3_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-11 13:58 · GPU3 (전용) · 60000 iter
- 액추에이터: `ImplicitActuatorCfg(effort_limit=None, kp 25, kd 0.5)` — `implicit_stock` 과 동일
- AMP: 기본값 그대로 (`task_reward_lerp` 0.5, Stage 1 5000 iter) — `noamp_stock` 과 달리 **켜져 있다**

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/reinforcement_learning/train.py \
  --rl_library rsl_rl --task Go2-Imitation-Tracking-v0 --num_envs 4096 --headless \
  --max_iterations 60000 --run_name depen3_stock env.use_pace_params=false \
  env.robot.spawn.rigid_props.max_depenetration_velocity=3.0
```

소스 파일은 건드리지 않았다 — hydra 로 `robot.spawn.rigid_props` 를 직접 덮으므로
`UNITREE_GO2_CFG` 는 그대로고 다른 Go2 task 에 영향이 없다.

## `max_depenetration_velocity` 가 무엇인가

두 강체가 서로 파고들었을 때 **솔버가 밀어내는 속도의 상한** [m/s]. 상한이 작으면 관통이
여러 스텝에 걸쳐 천천히 해소되고(지면이 물렁해지는 방향), 크면 즉시 뽑혀 나온다.

| | 값 | 출처 |
|---|---:|---|
| PhysX 스키마 기본 | **3.0** | `physxRigidBody:maxDepenetrationVelocity = 3` (`PhysxSchema/resources/generatedSchema.usda`) |
| 기존 Go2 | **1.0** | `source/isaaclab_assets/isaaclab_assets/robots/unitree.py:153` |
| 이 런 | **3.0** | hydra override |
| MimicKit | 10.0 | `mimickit/engines/isaac_lab_engine.py:941,976` |

가설: 빠를수록 스텝당 발 관통 깊이가 커지는데 상한 1.0 은 그것을 충분히 빨리 못 뽑아내
**고속에서만 지면이 물렁해진다**. 저속 구간에는 영향이 거의 없어야 한다 — 즉 `cmd 2.5` 는
그대로이고 `cmd 3.0` 이상만 움직인다면 이 기전을 지지한다. **인과 미검증 가설이다.**

## 설정 반영 검증

hydra 오버라이드가 조용히 씹히는 사고(`agent.resume` 전례)를 배제하기 위해 두 단계 확인:

1. `params/env.yaml:229` → `max_depenetration_velocity: 3.0` (`use_pace_params: false` 도 확인)
2. **USD 프림 실측** — 같은 오버라이드로 Go2 를 스폰해 속성을 되읽었다:
   ```
   [CHECK] /World/baseline/base  authored=True  value=1.0
   [CHECK] /World/override/base  authored=True  value=3.0
   ```
   cfg 값이 실제로 `physxRigidBody:maxDepenetrationVelocity` 로 써진다는 것까지 확인.

## 판정 규칙

★ 이 세대군은 **40k 이후 점으로만** 판정한다. 8k~24k 차이는 48k 이후를 예측하지 못한다는 것이
`implicit_stock` 에서 실측됐다(Δ −41 → +6 으로 반전).
근거: `../_comparisons/mimickit_vs_60_actuator_limit/README.md`

## 비교 대상 (`2026-08-10_10-45-34_implicit_stock`, depen 1.0, 나머지 동일)

| cmd | 지표 | 40k | 48k | 56k |
|---|---|---:|---:|---:|
| 2.5 | 중앙 | 1.389 | 1.598 | 1.569 |
| 3.0 | 중앙 | 0.686 | **1.686** | 1.623 |
| 3.5 | 달성률 | 0 | 0 | 0 |

## 결과 요약

학습 진행 중 — 램프 미측정.

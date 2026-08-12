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
**고속에서만 지면이 물렁해진다**. **인과 미검증 가설이다.**

판정 기준 — 이 task 의 재현 산포는 달성률 **±8~15%p**, 중앙값 ±0.02 m/s 다. 따라서:

- **지지**: 40k 이후 점에서 `cmd 3.0` 달성률이 **+15%p 를 넘어** 개선.
- **반증**: 같은 구간에서 **−15%p 를 넘어** 악화.
- ⚠ "`cmd 2.5` 가 안 움직였다"는 **판별 근거가 아니다** — `implicit_stock` 의 `cmd 2.5` Δ 도
  40k~56k 에서 +2/+0/+6 으로 이미 밴드 안이라, 기전이 있든 없든 참이 된다.

## 설정 반영 검증

hydra 오버라이드가 조용히 씹히는 사고(`agent.resume` 전례)를 배제하기 위해 두 단계 확인:

1. `params/env.yaml:229` → `max_depenetration_velocity: 3.0` (`use_pace_params: false` 도 확인)
2. **USD 프림 실측** — 같은 오버라이드로 Go2 를 스폰해 속성을 되읽었다:
   ```
   [CHECK] /World/baseline/base  authored=True  value=1.0
   [CHECK] /World/override/base  authored=True  value=3.0
   ```
   cfg 값이 실제로 `physxRigidBody:maxDepenetrationVelocity` 로 써진다는 것까지 확인.

## ★★ 램프 측정 시 `--sync_spawn_props` 필수

램프(`_workspace/go2_tracking/speed_ramp_record.py`)는 run 의 params 가 아니라 **현재 소스 cfg**
로 env 를 만든다. 이 런의 3.0 은 hydra 오버라이드라 `unitree.py` 에는 없으므로, 아무 조치 없이
재면 **3.0 에서 학습한 정책을 1.0 에서 재는** cross-plant 측정이 된다(`--no_pace` 누락으로 같은
사고를 낸 적이 있다).

그래서 램프 스크립트에 가드를 넣었다 — run 의 `params/env.yaml` 과 소스 cfg 의
`robot.spawn.rigid_props` 가 다르면 **실행을 거부**하고, `--sync_spawn_props` 를 주면 run 값으로
맞춘 뒤 무엇을 바꿨는지 출력한다. 세 경로 모두 smoke 로 실측 확인:

- 플래그 없이 이 런 → `error: ... max_depenetration_velocity: (1.0, 3.0) ... cross-plant 측정이다`
- 플래그 주고 이 런 → `>>> [sync_spawn_props] ... 1.0 → 3.0 (run 값으로 맞춤)` 후 정상 완주
- `implicit_stock`(오버라이드 없음) → 오탐 없이 그냥 통과

```bash
python _workspace/go2_tracking/speed_ramp_record.py \
  --checkpoint logs/rsl_rl/go2_imitation_tracking/2026-08-11_13-58-01_depen3_stock/model_40000.pt \
  --task Go2-Imitation-Tracking-v0 --num_envs 64 --no_video --no_pace --sync_spawn_props \
  --out_dir reports/rsl_rl/go2_imitation_tracking/_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_depen3/stock_40000
```

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

## 결과 요약 (중간, 49.8k / 60k)

**★ 반증 쪽이다 — 3.0 은 `implicit_stock`(1.0) 보다 나쁘다.** 판정 가능 구간(40k 이후) 램프 2 점.

| cmd | 지표 | 40k base | 40k **3.0** | Δ | 48k base | 48k **3.0** | Δ |
|---|---|---:|---:|---:|---:|---:|---:|
| 2.0 | 중앙 | 1.315 | 1.296 | −0.019 | 1.375 | 1.294 | −0.081 |
| 2.5 | 중앙 | 1.389 | 1.329 | −0.060 | **1.598** | 1.323 | **−0.275** |
| 2.5 | 달성률 | 94 | 62 | **−31** | 97 | 64 | **−33** |
| 3.0 | 중앙 | 0.686 | 0.439 | −0.247 | **1.686** | 0.445 | **−1.241** |
| 3.0 | 달성률 | 41 | 38 | −3 | 58 | 38 | **−20** |
| 3.5 | 달성률 | 0 | 0 | +0 | 0 | 0 | +0 |
| — | top | 1.389@2.5 | 1.329@2.5 | | **1.686@3.0** | 1.323@**2.5** | |

- 사전에 정한 **반증 조건(40k 이후 `cmd 3.0` 달성률 −15%p 초과 악화)을 48k 에서 −20%p 로 충족**.
  40k 는 −3%p 로 밴드 안이라 2 점 중 1 점만 넘겼다.
- 더 강한 신호는 **`cmd 2.5` 가 두 점 모두 −31/−33%p** 라는 것이다. 재현 산포 ±8~15%p 의
  두 배를 넘고 부호가 일치한다.
- 48k 에서 base 의 top 이 `cmd 3.0` 1.686 이었는데 3.0 쪽은 **`cmd 2.5` 1.323 으로 되돌아갔다**.
  `implicit_stock` 이 이 세대에서 얻은 유일한 성과(top 이 2.5→3.0 으로 올라간 것)가 사라진다.

★ **가설의 형태도 틀렸다.** "고속에서만 지면이 물렁해진다"였는데, 실제로는 `cmd 2.5` 같은
중속 구간이 더 크게 무너졌다. 즉 관측된 열화는 고속-특이적이지 않다.

⚠ 아직 최종이 아니다 — 56k 점이 남아 있고, 판정 근거는 2 점뿐이다. 이 세대군에서 **3 개 연속 ·
두 플랜트 · 부호 6/6 이라는 더 강한 근거가 뒤집힌 전례**가 있으므로 56k 까지 보고 확정한다.

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_depen3/`

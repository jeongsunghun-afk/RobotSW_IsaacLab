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

## 결과 요약 (완주, 60k)

**결론: 3.0 을 채택할 근거가 없다.** 다만 "확실히 나쁘다"까지는 아니다 — 마지막 점에서 격차가
크게 좁혀졌다. 판정 구간(40k 이후) 매치드 램프 **4 점**.

| cmd | 지표 | 40k base | 40k **3.0** | 48k base | 48k **3.0** | 56k base | 56k **3.0** | 59999 base | 59999 **3.0** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | 중앙 | 1.389 | 1.329 | **1.598** | 1.323 | 1.569 | 1.339 | 1.440 | **1.498** |
| 2.5 | 달성률 | 94 | 62 (**−31**) | 97 | 64 (**−33**) | 97 | 64 (**−33**) | 98 | 83 (**−16**) |
| 3.0 | 중앙 | 0.686 | 0.439 | **1.686** | 0.445 | 1.623 | 0.531 | 1.240 | **1.317** |
| 3.0 | 달성률 | 41 | 38 (−3) | 58 | 38 (−20) | 53 | 45 (−8) | 42 | 47 (**+5**) |
| 3.5 | 달성률 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| — | top | 1.389@2.5 | 1.329@2.5 | **1.686@3.0** | 1.323@2.5 | **1.623@3.0** | 1.339@2.5 | 1.440@2.5 | 1.498@2.5 |

### 무엇이 견고하고 무엇이 아닌가

**견고하다:**
- `cmd 2.5` 달성률이 **4 점 전부 음수**(−31/−33/−33/−16). 부호가 한 번도 안 뒤집혔고
  재현 산포 ±8~15%p 를 전부 넘는다.
- **3.0 은 base 의 최고점을 한 번도 못 넘는다.** base 최고 **1.686 @ cmd 3.0**(48k) vs
  3.0 최고 **1.498 @ cmd 2.5**(59999) — 11% 낮고 명령 단계도 한 칸 아래다.
- `cmd 3.5` 는 양쪽 다 전 점 0%.

**견고하지 않다 — 초기 판정을 완화한다:**
- 59999 에서 **중앙값은 3.0 이 오히려 앞선다**(2.5 에서 1.498 vs 1.440, 3.0 에서 1.317 vs 1.240).
  `cmd 3.0` 달성률도 +5%p 다.
- `cmd 3.0` 달성률 Δ 는 −3/−20/−8/+5 로 **부호가 뒤집힌다**. 40k~56k 만 보고 "−20%p 로 반증
  충족"이라고 적었던 것은 **과했다** — base 의 48k·56k 가 그 런의 정점이었고, 3.0 은 정점이
  59999 로 늦게 왔을 뿐일 수 있다.
- 즉 정확한 표현은 "3.0 이 무너뜨린다"가 아니라 **"3.0 은 `cmd 2.5` 안정성을 일관되게 깎고,
  base 가 도달한 `cmd 3.0` 영역에는 끝내 못 올라갔다"** 이다.

★ **가설의 형태는 여전히 틀렸다.** "고속에서만 지면이 물렁해진다"였는데, 실제로 일관되게
움직인 것은 `cmd 2.5` 중속 구간이다. 고속-특이적 기전이 아니다.

→ **소스 기본값 1.0 을 유지한다.** 채택 근거가 없다. 세 접촉 노브 중 유일하게 기전이 그려졌던
항목이 이 정도이므로 `angular_damping`·`bounce_threshold_velocity` 도 우선순위를 내린다.

원자료: `../_comparisons/mimickit_vs_60_actuator_limit/metrics/ramp_depen3/`
(base 59999 점은 이번에 새로 쟀다: `.../ramp_implicit/stock_59999/`)

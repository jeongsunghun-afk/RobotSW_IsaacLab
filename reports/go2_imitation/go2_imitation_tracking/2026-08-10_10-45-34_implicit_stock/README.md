# 2026-08-10_10-45-34_implicit_stock

한 줄 목적: `DCMotorCfg` → `ImplicitActuatorCfg` 로 **토크-속도 곡선을 제거**했을 때
`cmd 3.5` 벽이 열리는지 본다. **스톡 플랜트**(`use_pace_params=false`) 쪽 A/B 짝.

## 기본 정보

- 원본 로그: `/home/lgb/IsaacLab-6.0/logs/rsl_rl/go2_imitation_tracking/2026-08-10_10-45-34_implicit_stock`
- gym task id: `Go2-Imitation-Tracking-v0`
- rl_library: `rsl_rl`
- experiment_name: `go2_imitation_tracking`
- 시작: 2026-08-10 10:45 · `--num_envs 4096 --max_iterations 60000 env.use_pace_params=false` · GPU1
- 액추에이터: `ImplicitActuatorCfg(joint_names_expr=[".*"], effort_limit=None, kp 25, kd 0.5)`
  → USD `drive maxForce` 를 그대로 캡으로 쓴다 (hip/thigh **23.70**, calf **45.43** N·m)
- 직전 세대(`yaw10vn15_stock`, `DCMotorCfg` calf 35.5/`velocity_limit` 30.0) 대비
  **바뀐 것은 액추에이터 모델 하나** — yaw ±1.0 · `joint_vel_noise` 0.15 동일

## 결과 요약

**56k / 60k.** 램프 7 점 전부 확보. **★ 이 플랜트에서는 ImplicitActuator 가 확실히 낫다.**

| cmd | 지표 | 8k | 16k | 24k | 32k | 40k | 48k | 56k |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 2.5 | DC 중앙 | 1.540 | 1.554 | 1.556 | 1.564 | 1.494 | 1.553 | 1.495 |
| 2.5 | **IMP 중앙** | 1.398 | 1.524 | 1.256 | 1.373 | 1.389 | **1.598** | **1.569** |
| 2.5 | Δ 달성률 | −27 | −39 | −41 | −14 | +2 | +0 | +6 |
| 3.0 | DC 중앙 | 1.451 | 0.743 | 0.721 | 0.757 | 0.673 | 1.387 | 1.074 |
| 3.0 | **IMP 중앙** | 1.122 | 1.347 | 0.366 | 0.860 | 0.686 | **1.686** | **1.623** |
| 3.0 | Δ 달성률 | −20 | **+14** | **+16** | **+22** | **+28** | **+19** | **+30** |
| 3.5 | IMP 달성률 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

- **`cmd 3.0` 이 16k 이후 6 개 연속 앞선다**(+14~+30%p). 48k 에서 **top 이 처음으로
  `cmd 2.5` 가 아니라 `cmd 3.0` 으로 올라갔다 — 1.686 m/s (58%)**. DC 의 같은 구간 최고는 1.387.
- `cmd 2.5` 도 40k 이후로는 DC 와 동률이거나 앞선다(48k 1.598 vs 1.553, 달성률 97% vs 97%).
- ★ **`cmd 3.5` 는 7 점 전부 0%.** 이 세대를 시작한 판정 기준은 미달이다.

⚠⚠ **8k~24k 로 내렸던 "`cmd 2.5` 도달률 하락 확정" 판정을 철회한다.** Δ 가 −27/−39/−41 →
−14/+2/+0/+6 으로 뒤집혔다. 3 개 연속 점 · 부호 6/6 일치 · 폭 20~41%p 라는 **꽤 강한 근거를
갖고도** 틀렸으므로, 이 세대군의 판정은 **40k 이후 점으로만** 한다.

토크(48k, cmd 3.0, 달린 env 37/64): **thigh 4 개 전부 p95 23.70 에 캡 도달 13.5~21.8%** 로
포화가 더 깊어졌다. calf `|q̇|p95` 4.0~6.4 → USD 클립 15.70 은 여기서도 안 걸린다.

곡선을 없앤 대가로 벽이 **평탄한 `thigh maxForce = 23.7 N·m`** 로 옮겨갔고, 이 값은 Unitree
공식 peak 라 올릴 근거가 없다.

전체 표·그림: `../../_comparisons/mimickit_vs_60_actuator_limit/metrics/implicit_full_trajectory.md`
· `.../figures/implicit_full_trajectory.png`

## 산출물

### videos

<!-- report-video:videos -->
| 파일 | 체크포인트 | 태그 | 렌더 시각 |
|---|---|---|---|
| [`model_16000__implicit_stock_16k__20260810-170343.mp4`](videos/model_16000__implicit_stock_16k__20260810-170343.mp4) | `model_16000.pt` | `implicit_stock_16k` | 2026-08-10 17:04:37 |
| [`model_48000__implicit_stock_48k_best__20260811-090503.mp4`](videos/model_48000__implicit_stock_48k_best__20260811-090503.mp4) | `model_48000.pt` | `implicit_stock_48k_best` | 2026-08-11 09:05:58 |
| [`model_59999__implicit_stock_ramp_0to4ms__20260814.mp4`](videos/model_59999__implicit_stock_ramp_0to4ms__20260814.mp4) | `model_59999.pt` | `ramp_0to4ms` (체이스캠, 0→4 m/s) | 2026-08-14 |
| [`model_48000__followcmd40__20260819-104650.mp4`](videos/model_48000__followcmd40__20260819-104650.mp4) | `model_48000.pt` | `followcmd40` | 2026-08-19 10:47:46 |
| [`model_48000__ndcmd40__20260819-112045.mp4`](videos/model_48000__ndcmd40__20260819-112045.mp4) | `model_48000.pt` | `ndcmd40` | 2026-08-19 11:21:40 |
| [`model_48000__c2cmd40__20260824-085818.mp4`](videos/model_48000__c2cmd40__20260824-085818.mp4) | `model_48000.pt` | `c2cmd40` | 2026-08-24 08:59:00 |
<!-- /report-video:videos -->

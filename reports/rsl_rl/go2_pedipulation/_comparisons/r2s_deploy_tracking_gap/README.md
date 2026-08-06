# r2s 배포에서 발 추종이 무너지는 원인 — sim_runner 자유 구동

**판정: 원인은 정책도 런타임도 플랜트도 아니고, `sim_runner_go2.py` 가 정책과 동기화 없이
자유 구동한다는 것.** `--lockstep` 을 켜면 추종 오차가 **264 mm → 10~14 mm**(3회 반복)로
떨어지고, pedipulation 구간의 slew limiter 개입이 **726회 → 4회**로 사라진다.
최대 토크는 자유 구동 14.98 N·m vs lockstep 8.40 N·m — 다만 자유 구동 쪽 값은 아래 "함정"의
분산에 걸려 있으므로 **참고치**로만 볼 것.

- 정책: `logs/rsl_rl/go2_pedipulation/2026-08-03_15-53-29_hipscale_scratch_s10x/exported/pedipulation_policy.pt`
- 런타임: `scripts/real2sim/r2s_go2/pedipulation_runtime.py` (obs 83 / action 28)
- 기준점: 같은 정책의 IsaacLab 게이트 hold 오차 **9.3 mm** (`play_pedipulation_gates.py --mode hold`)
- 측정일: 2026-08-06

## 증상

sim 화면에서 조작 발이 목표 구(초록)에 닿지 못하고, 토크가 과해 보인다.

## 결론에 이르는 측정

명령 대본은 전부 동일하다 — 조작 다리 FL, 목표 = latch 된 nominal + (0, 0, 0.15) m,
정착 후 조작 다리를 RR 로 교체.

| 구성 | FL 정착오차 | RR 정착오차 | 비고 |
|---|---:|---:|---|
| r2s 3-프로세스, 자유 구동 (run 7) | 6.5 mm | 110.0 mm | latch 가 학습 nominal 과 최대 100 mm 어긋난 상태 |
| r2s 3-프로세스, 자유 구동 (run 10) | 264.3 mm | 262.7 mm | latch 를 9.4 mm 까지 맞춘 상태 |
| **r2s 3-프로세스, `--lockstep` (run 11)** | **11.0 mm** | **13.6 mm** | latch 는 오히려 155.8 mm 어긋남 |
| **`--lockstep` 반복 (run 12)** | **10.3 mm** | **12.6 mm** | |
| **`--lockstep` 반복 (run 13)** | **11.5 mm** | **12.5 mm** | |
| 전송 경로 제거, 런타임이 env 를 직접 구동 | 13.9 mm | 10.5 mm | `verify_pedipulation_closedloop_go2.py`, UDP/ROS 없음 |
| IsaacLab 게이트 hold (기준점) | — | — | 전 다리 평균 9.3 mm |

**분산 자체가 증거다.** 자유 구동은 같은 구성에서 6.5~264 mm 로 40배 흩어지는데,
lockstep 3회는 FL 10.3~11.5 / RR 12.5~13.6 mm 로 거의 겹친다. 크기뿐 아니라 재현성이
같이 회복된다.

## 배제한 원인

- **`action_scale` · `hip_scale_reduction` 미적용** — 런타임과 env 의 관절 목표가 1.9e-6 rad
  일치한다. 둘 중 하나라도 빠지면 0.25배 / 2배 차이로 즉시 드러난다.
- **nominal latch 어긋남** — 가장 그럴듯했고, 실제로 크게 어긋나 있었다(뒷발이 학습 자세보다
  100~155 mm 앞·위). 그런데 latch 를 9.4 mm 까지 맞춘 run 10 이 **가장 나빴고**, 155.8 mm
  어긋난 run 11 이 11 mm 로 추종했다. 명령이 latch 기준의 상대 offset 이라 기준계가 통째로
  옮겨가도 정책은 목표에 도달한다. **기각.**
- **런타임 적분 wind-up** — `action_to_target_art()` 는 관절한계로 클램프한 값을
  `state.manip_joint_target` 에 되쓴다. env `_pre_physics_step` 과 동일하다. 감김 없음.
- **플랜트 차이** — 양쪽 다 200 Hz physics / 50 Hz control, `UNITREE_GO2_CFG`(kp 25 / kd 0.5,
  effort 23.5), PACE `set2`. 서류상 동일하고 로그의 프리셋도 `plant=set2` 로 확인했다.
- **slew rate limiter 자체** — r2s 에만 있는 것은 맞지만, `R2S_NO_SLEW=1` A/B 는
  아래 "함정" 때문에 무효다. lockstep 을 켜면 limiter 개입이 726회 → 4회로 사라져
  **limiter 는 원인이 아니라 desync 의 증상**이었다.

## 원인

`sim_runner_go2.py` 의 메인 루프는 페이싱도 lockstep 도 없이 `env.step()` 을 최대 속도로 돈다.
**정책 tick 과 sim 스텝 사이에 어떤 동기화도 없다** — 주기도, 위상도, 왕복 지연도 묶여 있지
않다. 측정된 증상은 sim **62.5 Hz** vs 정책 발행 **50 Hz**(1.25배)이고, 그 위에 다중 홉
(UDP → ROS2 → GUI → ROS2 → UDP) 왕복 지연이 얹힌다. 학습이 견디도록 배운 지연은
`max_action_delay_steps = 1`(20 ms) 뿐이다.

⚠ **하위 기전은 분리하지 않았다.** `--lockstep` 은 주기·위상·지연 상한을 동시에 고친다.
왕복 지연을 step 단위로 실측하지 않았으므로 "주기 어긋남이 원인"이라고 좁혀 말할 수 없다.
확정된 것은 "동기화 부재가 원인이고 lockstep 이 해소한다"까지다.

**왜 지금까지 안 드러났나**: Policy·Recovery 모드의 정책은 목표가 **절대값**
(`target = a·scale + default`)이라 늦거나 중복된 액션이 자기 보정된다. pedipulation 의 조작
다리만 **적분형**(`target += delta`)이라 타이밍 오차가 누적되고 되돌아오지 않는다. 이 경로로
배포된 첫 상태 보유 정책이다.

## 처방

`sim_runner_go2.py` 가 수신한 `/lowcmd` 패킷 하나당 정확히 한 번 스텝한다. 타임아웃(0.1 s)이면
그냥 스텝해 publisher 가 없을 때 sim 이 얼지 않게 한다.

**2026-08-06 부로 기본값이 lockstep 이다.** 자유 구동은 `--no_lockstep` 으로만 재현한다 —
이 날짜 이전에 수집한 r2s 데이터는 전부 자유 구동 타이밍이므로, 그 데이터와 나란히 비교할
때만 쓴다. 기동 로그의 `stepping=` 필드로 어느 쪽인지 확인할 수 있다.

⚠ 부작용: publisher 가 없으면 sim 이 0.1 s 타임아웃마다 한 번씩만 돌아 **느린 화면**으로
보인다. 멈춘 게 아니다.

## 함정 (다음 사람이 반복하기 쉬운 것)

- **같은 설정이 6.5 mm ~ 264 mm 를 오간다.** 자유 구동에서는 sim 속도가 렌더 부하에 따라
  흔들려 결과가 실행마다 다르다. 스냅샷 한 번으로 방법을 비교하면 안 된다.
- 이 분산 때문에 앞서 뽑은 **slew limiter A/B(135.8 mm vs 269.9 mm)는 무효**다. 두 팔이
  같은 조건이 아니라 같은 조건의 분산 안에 있었다.
- **정적 대조로는 절대 안 잡힌다.** obs 5.6e-7, 관절목표 1.9e-6, FK 0.0004 mm 로 전부 깨끗한데
  폐루프에서만 무너진다. 배포 검증에 폐루프 롤아웃이 반드시 있어야 한다.
- 부수 관측: r2s 의 기립(prone → Stand Up → Default)이 뒷다리를 자주 덜 편다
  (`|q − DEFAULT|` 최대 0.46 rad, 발 위치 100~155 mm 이탈). 추종 문제의 원인은 아니지만
  실기 정합 관점에서 별개로 볼 것.

## 재현

```bash
# 전송 경로 없는 폐루프 기준점
./isaaclab.sh -p scripts/real2sim/verify_pedipulation_closedloop_go2.py

# 3-프로세스 (터미널 3개). lockstep 은 기본값이라 플래그가 없다 —
# 자유 구동 쪽을 재현하려면 --no_lockstep 을 붙인다.
R2S_LOG_MARKER=1 R2S_LOG_SLEW=1 ./isaaclab.sh -p scripts/real2sim/sim_runner_go2.py \
    --num_envs 1 --plant set2 --camera follow
R2S_SIM_STATE_TOPIC=/lowstate python3 scripts/real2sim/r2s_go2/sim_bridge.py
python3 scripts/real2sim/r2s_go2/gui_controller.py
```

원자료: 세션 스크래치패드 `sim_runner{5,6,7,8,9}.log`, `gui_demo{8,9,10,11,12,13}.log`.

## 혼동 주의 — 이웃 문서

`reports/_comparisons/r2s_control_rate_seam/` 는 **다른 질문**이다. 그쪽은 env 안의
control rate(50/100/200 Hz)와 실기 드라이버 kHz PD 사이의 이음매를 다룬다. 이 문서는
env 의 control rate 는 50 Hz 로 고정된 채, **sim 프로세스와 정책 프로세스가 서로 동기화되지
않는다**는 별개의 문제다.

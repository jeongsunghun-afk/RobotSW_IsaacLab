# HindLeg 정책 배포 런북 (2026-08-28)

배포 후보 = `gainclamp_ft` **model_33900**. 아래 수치는 **전부 이 체크포인트 실측**이다 —
다른 체크포인트를 쓰면 임계·달성률이 달라지므로 프로브를 다시 돌려야 한다.

```
정책   logs/rsl_rl/hindLeg_history_direct/2026-08-28_09-39-20_gainclamp_ft/
         exported/deployable_policy.pt            (GUI 가 최신순으로 자동 탐색)
         exported/deployable_policy_model_33900.pt (같은 파일, 출처 고정용 사본)
검증   wrapper diff 0.0e+00 · jit reload diff 0.0e+00 · GUI 계약검사 OK (plant=PACE)
```

⚠ **학습은 아직 진행 중**(33900 / 39800)이다. 이 export 는 중간 스냅샷이고, 완주 후에는
재-export + 프로브 재측정이 필요하다.

---

## 1. 실기 설정값 — 이 두 개만 맞추면 된다

### 1-a. PD 게인 — **변환하지 말 것**

```
채널(드라이버에 넣는 값)   hip 100/5   thigh 50/5   calf 50/5   foot 20/5
```

`motions.DEFAULT_KP/KD` 그대로다. GUI Gains 그룹이 이 값을 GAIN(R2PK) 으로 발행하고
`real_runner` 가 무변환으로 드라이버에 넣는다.

★ **관절 좌표로 환산해서 넣으면 안 된다.** sim 이 쓰는 `calf 112.5 / foot 28.8` 은 같은 채널값이
관절에서 나타나는 **이미지**이지 전송값이 아니다(`kp_joint = k²·kp_ch`, k = 1/1/1.5/1.2).
2026-08-28 이전 학습 env 가 이 둘을 혼동해 채널값을 관절 게인으로 쓰고 있었다 — 그게
`gainclamp_ft` 로 재학습한 이유다.

### 1-b. 토크 트립 임계 — 속도별로 다르다

펌웨어 기본값은 채널 **15 N·m / 50 ms**. 그 값으로는 **모든 속도에서 걸린다.**

| 명령 | 필요 채널 임계 | 근거 (≥3 스텝 런, 관절-env 열 256 개 중) |
|---|---|---|
| `cmd 0.3` | **35** | 35 에서 hip·thigh·calf·foot 전부 0 |
| `cmd 0.5` | **45** | 35 에서 calf 2 · foot 2 남음 |
| `cmd 1.0` | **45** | 35 에서 hip 1 · 45 에서도 hip 1 (60 ms) |

채널 상한은 **84**(모터 12 N·m × 7)이므로 35 = 42 % · 45 = 54 % 지점이다.
**보호를 없애는 게 아니라 옮기는 것**이다. 다만 트립이 열 보호인지 기계 보호인지는
우리가 판단할 수 없다 — 듀티가 낮다는 것까지가 sim 이 말할 수 있는 전부다.

⚠ `cmd 1.0` 의 hip 1 건은 **256 열 중 1 건**이라 잡음 구간이다. 같은 조건을 다시 재면
꼬리 통계가 흔들린 전례가 있다(같은 정책·다른 롤아웃에서 calf p99 32.3 vs 43.6).
**한 번의 측정으로 45 를 확정하지 말고, 첫 실기에서 `cmd 0.3`(35 로 깨끗) 부터 간다.**

---

## 2. 절차

### 2-a. sim 에서 먼저 (실기 없이 전 구간 확인)

```bash
# ① r2s sim — policy mode, 자유베이스
scripts/real2sim/r2s_biped_leg/run_policy_sim.sh

# ② GUI (system python3 + PyQt5)
scripts/real2sim/r2s_biped_leg/run_gui_controller.sh
```

GUI 에서:
1. **Policy 그룹** — 모델 목록 맨 위가 `2026-08-28_09-39-20_gainclamp_ft` 인지 확인
2. **Gains** — §1-a 값 입력 (hip 100/5 · thigh 50/5 · calf 50/5 · foot 20/5)
3. **Action scale = 1.00 유지** — ★낮추면 트립이 **악화**된다(§3)
4. 라우팅 `action from SIM obs → sim`, 명령 `x_vel 0.3`
5. Relax → Engage → 걷는지 확인

### 2-b. 실기 (sim 이 정상일 때만)

```bash
# ①·② 는 동일. GUI 만 real_host 를 준다
scripts/real2sim/r2s_biped_leg/run_gui_controller.sh --real_host 192.168.60.5
```

파이 쪽 선행 조건(`project_bipedleg_real_runner_bridge`):
- `RobotEmbedded` 를 **먼저 sudo 실행**, 모터 상태 100 회 수신 후에만 명령 enable
- `real_runner` 는 최신 빌드여야 한다(구버전은 TELEM 미송신)

사다리 — **각 단계에서 트립 0 을 확인한 뒤에만 다음으로**:

| # | 상태 | 명령 | 트립 임계 | 확인할 것 |
|---|---|---|---|---|
| 0 | RELAX | — | — | TELEM 수신 · 8 관절 각도가 sim 과 같은 부호 |
| 1 | ENGAGE, 정지 | `x_vel 0` | 35 | 자세 유지 · `cmd_clamp_bits` 상시 켜짐 여부 |
| 2 | 보행 | `x_vel 0.3` | 35 | 좌우 교대 · limp 래치 0 회 |
| 3 | 보행 | `x_vel 0.5` | 45 | 같음 |
| 4 | 보행 | `x_vel 1.0` | 45 | 같음. **hip 토크를 특히 볼 것** |

★ 단계 1 에서 **`cmd_clamp_bits` 를 반드시 확인한다.** 이 정책은 목표각 클램프를 **학습에
반영한** 첫 정책이므로, 클램프 신고가 상시 켜져 있으면 sim 과 실기의 한계값이 어긋난 것이다
(이전 정책은 목표의 81 % 가 한계 밖이라 항상 켜졌을 것이다).

---

## 3. 하면 안 되는 것

**`Action scale` 을 낮춰 "약하게" 시험하지 말 것.** 게인 노브가 아니라 분포 이동이다.
sim 실측(cmd 1.0, ≥3 스텝 런):

```
scale        1.0    0.8     0.6
foot 런       48    376    2527      ← 53 배
foot 평균     7.03   7.48   9.19 N·m ← 토크가 **오른다**
달성률        92 %   95 %    66 %
```

배율은 목표각을 기본 자세 쪽으로 당기므로 다리가 덜 뻗고 더 세게 착지한다.
GUI 툴팁에 같은 표를 넣어 뒀다. 약하게 가고 싶으면 **명령 속도를 낮춘다**(cmd 0.3).

---

## 4. 이 정책이 sim 에서 내는 값 (대조용)

실기에서 이 범위를 크게 벗어나면 sim2real 갭이 남아 있다는 신호다.

```
cmd     달성률   |tau| p99 [N·m]              양발접지   좌우교대
0.3      92 %    hip 18.9 thigh 24.1 calf 23.5 foot 20.7    ~13 %   43/44 %
0.5      96 %    hip 21.0 thigh 26.4 calf 28.3 foot 22.3    ~12 %   43/44 %
1.0      94 %    hip 24.1 thigh 30.9 calf 43.9 foot 26.6    ~12 %   44/44 %
```

⚠ `cmd 2.0` 은 sim 에서도 실패한다(달성률 −1 %). 이번 수정과 무관한 별건이므로
**실기에서 시도하지 말 것.**

---

## 5. 아직 모르는 것

- **발목 raw 가동범위** — 벨트라 모터축은 `q_foot + q_calf` 이고, 관절별 클램프로는 raw 가
  두 한계의 합(±2.329 rad, foot 단독 1.92)까지 간다. 실기 브리지도 관절별로만 자르므로
  **양쪽이 같은 규약**이지만, raw 기준 기구 한계가 따로 있으면 둘 다 안 지키는 것이다.
  단계 1 의 `cmd_clamp_bits` 와 실제 발목 거동으로 간접 확인할 수 있다.
- **접촉 영역 플랜트** — PACE 적합 캡처 13 개가 전부 공중 고정이라 **착지 응답은 검증된 적이
  없다**. 반사관성 1 스텝 지연의 실기 영향도 여기서만 드러난다
  (`reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/CONTACT_CAPTURES.md`).

---

재측정 도구:

```bash
python _workspace/hindleg_trip_probe.py --checkpoint <model.pt> --trip_nm 25 35 45 --cmd_x 0.3 0.5 1.0
python _workspace/hindleg_gait_probe.py --checkpoint <model.pt>
```

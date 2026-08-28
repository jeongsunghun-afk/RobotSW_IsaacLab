# Go2 Imitation Tracking — VAE latent embedding용 reference motion 데이터셋 충분성 분석

- **분석 대상**: `source/isaaclab_tasks/isaaclab_tasks/direct/go2_imitation_tracking/imitation/smr_mirror_pkl/*.pkl`
  (`go2_imitation/imitation/smr_mirror_pkl`의 심링크)
- **로더**: `go2_imitation_tracking/motion_lib.py` (`go2_imitation/motion_lib.py`와 byte-identical)
- **목적**: 현재 AMP discriminator가 쓰는 이 reference 데이터를, VAE 임베딩("Walk Like Dogs" 스타일
  x^vae 49-D + MoE decoder + 18-D latent) 학습으로 전환할 때 데이터가 충분한지 판정.
- **판정: Red** — 원본 데이터로는 VAE 전환이 사실상 불가능. 후진(v_x<0) 전무, 고 yaw 구간
  희박, 유니크 클립 8개·18.6초(미러 포함 37.2초)뿐이라 커버리지·규모 모두 크게 미달.

## 1. 인벤토리

디렉토리에 파일 18개(9 base + 9 `_mirror`). 단 **`go2_trot1.pkl`은 `go2_trot0.pkl`과 완전히
동일한 바이트(md5 `ef6128c6...` 동일)** — 별개 클립이 아니라 중복 파일이다. 따라서 실제
고유 클립은 **8개**.

| 파일 | 프레임 | fps | 길이(s) | 비고 |
|---|---|---|---|---|
| go2_run0.pkl | 88 | 60 | 1.45 | |
| go2_run1.pkl | 177 | 60 | 2.93 | |
| go2_run2.pkl | 71 | 60 | 1.17 | |
| go2_trot0.pkl | 208 | 60 | 3.45 | |
| go2_trot1.pkl | 208 | 60 | 3.45 | **trot0과 완전 중복(md5 동일)** |
| go2_walk.pkl | 198 | 60 | 3.28 | |
| go2_walk1.pkl | 37 | 60 | 0.60 | |
| go2_walk2.pkl | 128 | 60 | 2.12 | |
| go2_walk_turn.pkl | 208 | 60 | 3.45 | 유일한 실질 회전 클립 |

각 `_mirror` 파일은 좌우 대칭 증강(같은 프레임 수, y/roll/yaw 부호 반전 — 예: run0/run0_mirror가
yaw 부호만 반대인 것으로 확인됨). `.pkl` 딕셔너리 키는 `['loop_mode', 'fps', 'frames']`이고
`frames`는 `(N, 18)` — `[0:3] root_pos`, `[3:6] root_euler(rpy)`, `[6:18] joint_pos(12 DOF)`.
속도·발위치는 저장돼 있지 않고 `motion_lib.py`가 finite-difference + FK로 매번 파생한다.

- 고유 base 클립 8개, 프레임 합 **1,115** (18.58 s)
- 미러 포함(중복 trot1 계열 제외) **2,230 프레임 (37.2 s)**
- 파일 그대로(18개, 중복 포함) 로드 시 **2,646 프레임**

## 2. 필드 가용성 — x^vae 49-D 구성 가능 여부

PKL에는 `root_pos(3) + root_euler(3) + joint_pos(12)` 18열만 있고, 나머지는 전부 파생값이다.

| x^vae 성분 | 원본 PKL에 존재? | 파생 방법 | 판정 |
|---|---|---|---|
| base height (1) | 아니오 | `root_pos[:,2]` 그대로 | 가능 |
| 6D orientation (6) | 아니오 | `root_euler → quat → rotation matrix 앞 2열` | 가능 |
| base lin vel body-frame (3) | 아니오 | `finite_diff(root_pos) → world_vel_to_body(quat)` | 가능, **wrap 버그 무관**(quat은 euler를 매 프레임 독립 변환하므로 미분 대상이 아님) |
| base ang vel body-frame (3) | 아니오 | `finite_diff(root_euler) → euler_rates_to_body_angvel` | 가능하나 **§6 yaw-wrap 결함으로 오염** |
| foot pos ×4 (12) | 아니오 | `_go2_fk_foot_pos(joint_pos)` (해석적 FK, Go2 URDF 파라미터 사용) | 가능 |
| joint pos (12) | 예 | 직접 | 가능 |
| joint vel (12) | 아니오 | `finite_diff(joint_pos)` | 가능 |

**결론**: 필드 자체는 49-D 전부 구성 가능하다(전부 결정론적 파생 — FK/quat 변환에 학습되지
않은 파라미터 없음). 단, ang vel 파생 경로가 `go2_walk_turn`(및 그 미러)에서 깨져 있어
그대로 쓰면 안 된다(§6).

## 3. 커버리지 통계 — v_x, yaw rate, 2D joint coverage

task 명령 범위: `lin_vel_x ∈ [-1.0, 3.0]` m/s, `yaw_vel ∈ [-1.5, 1.5]` rad/s, `lin_vel_y = 0`(고정).

전체 클립(고유 8개, unwrap된 yaw로 계산) 프레임 기준(N=1,115):

| 신호 | min | p5 | mean | p95 | max |
|---|---|---|---|---|---|
| v_x [m/s] | **-0.001** | 0.037 | 1.378 | 3.226 | 4.943 |
| v_y [m/s] | -0.506 | — | ~0 | — | 0.572 |
| yaw rate(unwrap) [rad/s] | -0.952 | -0.464 | 0.122 | 1.466 | 2.244 |

- **v_x < 0 (후진) 데이터가 사실상 전무**하다(최소값이 -0.001로 수치 오차 수준). task 범위
  하한 -1.0 m/s는 **완전히 비어 있다** — ✗ 미달.
- v_x 상단은 go2_run2가 4.94 m/s까지 나와 task 상한 3.0을 초과 커버(런닝 클립 덕분) — ✓.
- v_y는 명령 범위가 항상 0으로 고정돼 있으므로(태스크가 옆걸음을 요구하지 않음) 이 데이터의
  잔류 v_y(±0.5 m/s, 의도치 않은 궤적 잡음)는 문제 되지 않는다 — 참고용.
- yaw rate: unwrap 기준으로도 |wz|>1.0 rad/s 구간은 `go2_walk_turn` 클립 하나에서만 나온다.
  task 상한 1.5 rad/s의 절반 이상을 커버하는 유일한 소스이며, 그 클립 자체가 §6 결함으로
  오염돼 있다 — 고쳐서 써야 하는 조건부 ✓.

2D joint coverage (`v_x` 8 bin × `yaw_rate` 6 bin, task 범위 그대로 bin 경계 사용):

```
vx_bins: [-1.0 -0.5  0.0  0.5  1.0  1.5  2.0  2.5  3.0]
wz_bins: [-1.5 -1.0 -0.5  0.0  0.5  1.0  1.5]

              wz:[-1.5,-1) [-1,-0.5) [-0.5,0) [0,0.5) [0.5,1) [1,1.5)
vx [-1.0,-0.5)        0         0        0        0       0       0
vx [-0.5, 0.0)        0         0        4        0       0       0
vx [ 0.0, 0.5)        0        28      140       87      18      24
vx [ 0.5, 1.0)        0        19       80      108      30       6
vx [ 1.0, 1.5)        0         0       78       29       0       0
vx [ 1.5, 2.0)        0         0      194       70       0       0
vx [ 2.0, 2.5)        0         0       75       55       0       0
vx [ 2.5, 3.0)        0         0       33       90      13       0
```

- 비어 있지 않은 셀: **20/48 (41.7%)**
- vx<0 행(2개 bin, 전체 16.7%)이 **완전히 0** — 후진 명령 전 구간 미커버
- `|wz| ≥ 1.0` 열(4개 bin 중 2개, `[-1.5,-1)`와 `[1,1.5)`)은 각각 0과 극소수(24, 13)뿐 — 고회전
  구간은 저속(`vx<1.0`)에서만, 그것도 희박하게 존재
- `vx ∈ [1.0, 3.0)` 구간(고속) 전체가 `wz`는 `[-0.5, 0.5)`에만 몰려 있음 — **고속+회전 조합은
  전무**(달리면서 도는 궤적이 없음)

## 4. 보행 종류(gait) 판별

발 접촉(발 높이 임계값)과 stride frequency(FL 발높이 신호의 FFT peak)로 추정(★추정 — 실측
force-plate 없이 FK 발위치 임계값 기반이라 duty factor 절대값은 근사치, gait *상대* 구분 용도로만
신뢰):

| 클립 | duty[FL,FR,RL,RR] | stride freq | 추정 gait |
|---|---|---|---|
| go2_run0 | 0.15 / 0.33 / 0.15 / 0.33 | 2.05 Hz | bound/gallop 계열(비대칭 낮은 duty) — ★검증 필요 |
| go2_run1 | 0.20 / 0.29 / 0.09 / 0.28 | 2.37 Hz | 상동 |
| go2_run2 | 0.08 / 0.15 / 0.35 / 0.27 | 2.54 Hz | 상동, 가장 빠름(4.94 m/s) |
| go2_trot0 (=trot1 중복) | 0.41 / 0.45 / 0.40 / 0.42 | 2.31 Hz | **trot**(4족 duty 균등 — 대각쌍 동위상 전형) |
| go2_walk | 0.79 / 0.69 / 0.82 / 0.89 | 0.30 Hz | **walk**(정적 안정, 저속 v_x≈0.14) |
| go2_walk1 | 0.70 / 0.27 / 0.35 / 0.49 | 1.62 Hz | 전이 구간(walk→trot 사이) — ★추정 |
| go2_walk2 | 0.52 / 0.47 / 0.52 / 0.34 | 0.94 Hz | 전이 구간, wz 변동 큼(-0.95~0.72) |
| go2_walk_turn | 0.58 / 0.36 / 0.50 / 0.41 | 1.15 Hz | walk 계열 + 회전(유일) |

**실재 모드 수**: 크게 3종 — **run(bound류) / trot / walk**, 그리고 walk_turn이 유일한 "회전
포함" 변형. `pace`는 이 디렉토리에는 없다(다른 go2_imitation 원본 세트에 `go2_pace.pkl`이
있었다는 CLAUDE.md 기록과 달리 `smr_mirror_pkl`에는 pace 클립이 없음 — 확인 필요 사항).

VAE가 "behavioral mode"를 발견할 여지: 3~4개 대분류는 duty factor로 뚜렷이 갈리므로 latent가
그 정도 클러스터는 찾아낼 수 있을 것으로 보이나, **클립 수(8개)가 모드당 1~3개뿐**이라 같은
모드 내에서의 style variation(예: 같은 trot이라도 다른 속도/보폭)을 학습할 표본이 사실상 없다.
`trot1`이 `trot0`과 완전 중복이라는 것 자체가 데이터 다양성 착시(파일 개수는 9개지만 실제
고유 정보는 8개)를 보여준다.

## 5. 데이터 규모 판정

- 총 상태전이 쌍: 고유 클립 기준 `1,115 - 8 ≈ 1,107`쌍, 미러 포함 `2,230 - 16 ≈ 2,214`쌍
  (파일 그대로 18개 로드 시 `2,646 - 18 = 2,628`쌍이지만 이 중 416쌍은 trot1/trot1_mirror
  중복이라 실질적으로는 앞의 2,214가 맞다)
- 총 재생시간: 고유 18.6 s, 미러 포함 37.2 s

**판정: 명백히 부족.** 근거:
- MoE decoder(6 experts) + 18-D latent + 2×256 encoder는 파라미터 수만도 대략 수십만~
  100만 단위인데, 학습 신호는 실질적으로 2천여 개의 (state, next-state) 쌍뿐이다. epoch당
  batch 다양성이 극히 낮아 encoder가 8개 클립의 시간 인덱스를 그대로 외우는(memorization)
  방향으로 붕괴할 위험이 크다 — 특히 walk_turn처럼 유일한 표본인 모드는 validation split을
  두는 순간 해당 모드가 train/val 어느 한쪽에서 통째로 사라진다.
- AMP discriminator는 프레임 단위 marginal(짧은 window)만 판별하면 되므로 이 정도 데이터로도
  "그럴듯한 자세 분포"를 학습할 수 있었지만, VAE는 궤적 전체의 매끄러운 latent 다양체를 학습해야
  하므로 요구 데이터량이 질적으로 다르다(참고 문헌들은 보통 수 시간~수십 시간의 mocap을 씀).
- 미러링은 **좌우 대칭 증강만**이며(부호 반전 확인됨) 실질적으로 독립적인 새 모드/속도/보폭을
  추가하지 않는다 — 규모 문제 해결에 기여도 낮음.

**필요 규모(추정)**: 최소한 모드별(walk/trot/run/turn) 각 5~10개 이상의 서로 다른 속도·보폭
변형 클립, 총 재생시간 5~10분(현재의 10~20배) 이상을 권고. 특히:
1. **후진(v_x<0)** 클립 — 현재 0개, 최소 walk/trot 각 1~2개 필요
2. **고 yaw(|wz|>1.0) × 중고속(v_x>1.0)** 조합 — 현재 0개
3. trot/run 각각 최소 2~3개의 서로 다른 속도 변형(현재 trot는 사실상 1개뿐)

## 6. 결함 점검

- **NaN/프레임 드롭**: 18개 파일 전부 NaN 0개. 프레임 간 최대 xy 변위도 클립별 속도 대역과
  일관돼 프레임 드롭 징후 없음(예: run2 0.083 m/frame×60fps≈4.97 m/s ≈ 실제 v_x_max 4.94와 일치).
- **yaw unwrap 결함 — 여전히 존재, `go2_walk_turn.pkl`(+ mirror)에서 실증**:
  `motion_lib.py`의 `_load_pkl`(`go2_imitation_tracking/motion_lib.py:442`, `go2_imitation/motion_lib.py`도
  byte-identical)이 `euler_rates = _finite_diff(root_euler, dt)`를 **raw yaw(±π wrap)에 그대로**
  적용한다. `go2_walk_turn`은 실제로 yaw가 -π/+π 경계를 넘나드는 유일한 클립이라 여기서 그대로
  터진다.
  - raw(현재 코드 그대로): `wz_max_abs = 373.39 rad/s` (= 2π × 60fps, 프레임 경계에서 순간 랩)
  - `np.unwrap` 적용 후: `wz_max_abs = 2.24 rad/s` (물리적으로 타당)
  - 다른 7개 클립은 yaw가 좁은 범위(±0.6 rad 이내)만 오가므로 wrap이 발생하지 않아 raw==unwrap로
    동일함 — 이 결함은 **turn 계열에만, 그러나 정확히 turn 계열이 yaw rate 다양성의 유일한
    소스라는 점에서 치명적**이다.
  - **영향 범위**: root ang_vel(z)만 오염되며, root lin_vel/foot_pos/joint 신호는 quat 기반
    프레임별 변환이라 영향 없음(§2 표 참고).
  - **VAE 학습에 대한 영향**: `x^vae`에 base ang vel 3성분이 포함되므로, 이 결함을 고치지 않고
    데이터셋을 만들면 2,214개 전이쌍 중 walk_turn 유래 208쌍(~9%)의 target ang_vel_z가
    ±373 rad/s라는 물리적으로 불가능한 값이 되어 (a) 정규화 통계(mean/std)가 완전히 왜곡되고
    (b) VAE가 이 극단치를 맞추려다 다른 98%의 정상 구간 재구성 품질을 희생하게 된다. 반드시
    VAE 데이터 파이프라인에서 `np.unwrap(yaw)` 후 미분하도록 고쳐야 한다(AMP는 이 결함을
    그대로 물려받아 왔으나 별도 이슈 — 이번 분석 범위 밖).

## 산출물

- `metrics/per_clip_stats.csv` — 클립별 v_x/v_y/yaw_rate min/mean/max, duty factor, stride
  frequency 원자료

## 요약(3줄)

1. 실질 고유 클립 8개·18.6초(trot1은 trot0과 완전 중복 파일), 미러 포함해도 37초 — VAE+MoE
   규모 대비 최소 1~2 자릿수 부족.
2. task 명령 범위의 후진(v_x<0)이 완전히 비었고, 고속×고yaw 조합도 전무 — 2D coverage
   20/48 bin(41.7%)만 커버.
3. yaw-wrap 결함(과거 AMP 분석에서 발견된 것과 동일 코드 경로)이 `go2_walk_turn` 클립에서
   여전히 살아 있어(wz 373 rad/s spike), 이 클립이 yaw rate 다양성의 유일한 소스라는 점과
   맞물려 VAE 학습 전 반드시 수정 필요.

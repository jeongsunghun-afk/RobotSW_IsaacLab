# 01 — Code Diff Analysis: go2_parkour_symmetry 3 runs

**Scope (Task #1):** Per-run code snapshot (`<run>/git/IsaacLab.diff`) 분석 → 런별 코드 변경점,
3런 상호 + 현재 워킹트리 대비 차이. 관심 항목: `parkour_env.py::_get_rewards()` positive_work 항,
`parkour_env_cfg.py` reward_scales.

**Method:** 각 런 디렉토리에는 `git/IsaacLab.diff` 단일 파일(코드 스냅샷 = working tree diff vs 해당 시점 HEAD)만 존재.
런별 base HEAD가 다름에 유의 (아래 타임라인).

---

## 0. 핵심 결론 (TL;DR)

- **3런의 유일한 parkour-관련 차이는 `reward_scales["positive_work"]` 한 값뿐이다.**
  - 런1 = **-3e-4** (권장 보수값)
  - 런2 = **0** (positive_work 비활성 = 순수 baseline)
  - 런3 = **-1e-2** (권장값의 **약 33×**, "과대" 의심값)
- **`air_time_cap`, `contact_duty_deficit`, `feet_gait_pairing` = 세 런 모두 0 (OFF).** anti-pronk 신호는
  세 런 전부 꺼져 있었다. (런2 이름 "no_duty_time_cap"이 마치 런2에서만 끈 것처럼 보이나, **실제로는 세 런 공통**)
- positive_work 수식 = `Σ_j max(0, τ_j · q̇_j)` over 12 joints (코드 동일, 세 런 불변).
- 런2 vs 런3 디퍼런스 중 `rel_standing_envs`(0.2 vs 0.05)는 **`hind_leg` env 파일**이라 parkour 학습과 무관(동시 작업 잔재).
- 현재 워킹트리는 `positive_work = -1e-2` (런3 값이 미반영/미복원 상태로 남아 있음).

---

## 1. 타임라인 / base HEAD

| 시각 | 이벤트 | positive_work | air_time_cap | contact_duty_deficit |
|------|--------|---------------|--------------|----------------------|
| `b0c21f9` 10:35 | 직전 commit (런1의 base HEAD) | (없음) | **-0.1** | **-0.5** |
| **런1** 12:13 `_positive_work` | working tree | **-3e-4** (신규 추가) | **0** | **0** |
| `50b0da2` 12:19 | 런1 상태 commit ("Add reference-free positive-work…") | -3e-4 | -0.0 | -0.0 |
| **런2** 16:21 `_no_duty_time_cap` | working tree (base=50b0da2) | **0** | 0 | 0 |
| **런3** 17:28 `_positive_work_0.01` | working tree (base=50b0da2) | **-1e-2** | 0 | 0 |
| 현재 | working tree | -1e-2 (런3 잔재) | -0.0 | -0.0 |

> 런1은 자신의 디프에서 직전 baseline의 `air_time_cap=-0.1`/`contact_duty_deficit=-0.5`를 **0으로 끄면서**
> 동시에 `positive_work=-3e-4`를 **새로 추가**했다. 즉 런1 = "anti-pronk 신호 제거 + positive_work 도입" 스왑.
> 이후 런1 상태가 `50b0da2`로 커밋되어 런2/런3의 base가 됨 → 런2/런3 디프는 `parkour_env.py`를 건드리지 않음
> (positive_work 코드가 이미 커밋되어 HEAD와 동일하므로 디프에 안 나타남).

---

## 2. reward_scales 변경점 (parkour_env_cfg.py, ParkourEnvCfg)

### 런1 (`2026-06-11_12-13-02_positive_work`)  — diff vs `b0c21f9`
```diff
-        "air_time_cap": -0.1,        # graded per-foot penalty
+        "air_time_cap": -0.0,        # ← OFF
-        "contact_duty_deficit": -0.5,  # 1순위 headline fix
+        "contact_duty_deficit": -0.0,  # ← OFF
+        # positive_work: Fu et al. 2021 positive mechanical work efficiency penalty.
+        "positive_work": -3e-4,      # ← 신규 추가 (권장 보수값)
```
- 그 외 reward_scales(tracking_goal_vel 1.5, tracking_yaw 0.5, feet_stumble -1.0, feet_edge -1.0,
  feet_dragging -0.1, feet_gait_pairing 0.0, dof_error_l2 -0.04 …)는 baseline과 동일 (변경 없음, 나머지는 ruff 줄바꿈 포맷 변경뿐).

### 런2 (`2026-06-11_16-21-24_no_duty_time_cap`)  — diff vs `50b0da2`
```diff
-        "positive_work": -3e-4,
+        "positive_work": -0e-4,   # = 0, positive_work 비활성
```
- 단 한 줄. air_time_cap / contact_duty_deficit는 이미 커밋된 0에서 변동 없음 → **런2 = 모든 gait-shaping OFF (순수 baseline)**.

### 런3 (`2026-06-11_17-28-23_positive_work_0.01`)  — diff vs `50b0da2`
```diff
-        "positive_work": -3e-4,
+        "positive_work": -1e-2,   # 권장 -3e-4 대비 ≈33× 강화 ("과대" 의심)
```
- 단 한 줄.

### 세 런 reward_scales 정렬표 (parkour 관련 항만)

| reward term | 런1 | 런2 | 런3 | 현재 워킹트리 |
|-------------|-----|-----|-----|----------------|
| **positive_work** | **-3e-4** | **0** | **-1e-2** | -1e-2 |
| air_time_cap | 0 | 0 | 0 | -0.0 |
| contact_duty_deficit | 0 | 0 | 0 | -0.0 |
| feet_gait_pairing | 0 | 0 | 0 | 0 |
| tracking_goal_vel | 1.5 | 1.5 | 1.5 | 1.5 |
| tracking_yaw | 0.5 | 0.5 | 0.5 | 0.5 |
| feet_dragging | -0.1 | -0.1 | -0.1 | -0.1 |
| feet_stumble / feet_edge | -1.0 / -1.0 | = | = | = |

---

## 3. positive_work 항 코드 (`parkour_env.py::_get_rewards()`)

런1 디프에서 신규 추가, 이후 커밋되어 런2/런3에서도 동일하게 실행됨 (코드 자체는 세 런 불변):

```python
# === Positive mechanical work (Fu et al. 2021 efficiency penalty) ===
# Penalising Σ max(0, τ_j·q̇_j) over 12 joints creates efficiency pressure
# (Fu et al. 2021) sufficient to induce natural gaits without motion references.
# Only positive (motoring) work counted; clamp at 0 prevents braking phases
# from contributing a spurious *bonus* under a negative weight.
# Stateless: reuses applied_torque (same source as torques_l2).
positive_work = torch.sum(
    torch.clamp(self._robot.data.applied_torque * self._robot.data.joint_vel, min=0.0), dim=1
)  # shape (N,), units: W (positive mechanical power); always >= 0
```
- 항상 ≥ 0 → weight<0이면 항상 penalty. 새 buffer 없음(stateless) → reset 초기화 불필요. 차원 안전.
- assemble 시 `reward_terms["positive_work"] = positive_work`로 등록(런1 line ~32747).
- cfg 주석에 기록된 calibration (pronk_cost_result.npz, 256 env×3000 step, dt=0.02):
  - stair per-step power mean=101 W, p90=235 W, **p99=857 W**.
  - tracking_goal_vel 전형 per-step 기여 ≈ 1.5×0.02×0.70 = **0.021**.
  - weight=-3e-4 → stair mean 0.0006(tracking의 2.9%), p90 0.0014(6.7%) — 보수적/안전.
  - weight=-1e-3 → stair p99=0.017(82%), max=0.028(132%) → **total_reward floor(clip min=0 @ env.py:1261) 위험**.
  - ⇒ **런3의 -1e-2는 -1e-3보다 다시 10× 강함**: stair mean 기여만 0.02(≈tracking 전부와 동급),
    p90/p99에서는 tracking을 압도 → clip(min=0) floor 상시 발동 가능 영역 (정량 위험, 학습곡선은 03에서 확인).

---

## 4. 비-parkour / 무관 변경 (노이즈 격리)

- **`rel_standing_envs` (0.2 vs 0.05, 런2 vs 런3):** `direct/hind_leg/hind_leg_env_cfg.py` 소속 →
  go2_parkour_symmetry 학습과 **무관**. 동시 진행된 hind_leg 작업의 워킹트리 잔재일 뿐.
- 세 디프 공통으로 들어있는 대형 블록(MoE-RMA `Go2ParkourMoEPPORunnerCfg`, LCP `RslRlLcpCfg`,
  `ppo_parkour*`, `actor_critic_parkour*`, exporter, r2s_hind_leg 라이선스 헤더, 다수 `_workspace/*.md`)은
  **세 런 모두 동일**하며 default-OFF 연구 트랙/문서/포맷 변경 → 런 간 차이 아님(런별 변별력 0).
- `gap_length_range`, observation_space 주석, dof_error_l2/hind_xy_vel_norm 등은 ruff 줄바꿈 포맷 변경(동작 불변).

---

## 5. Task #4(종합)로 넘길 핵심 인풋

1. **3런 = 단일 변수 실험 (positive_work scale: -3e-4 / 0 / -1e-2).** 다른 모든 parkour reward 동일.
2. **anti-pronk 신호(air_time_cap, contact_duty_deficit)는 3런 전부 OFF** — "런2에서만 껐다"는 이름은 오해 소지.
   따라서 런 간 행동 차이를 duty/time_cap on→off로 귀인할 수 없음(이미 셋 다 off).
3. **런3 positive_work=-1e-2는 cfg 자체 calibration 주석 기준으로도 명백한 과대(-1e-3조차 floor 위험이라 경고)** →
   `total_reward = clip(min=0)` floor 상시화 → gradient 소멸/"로봇 망가짐" 1순위 코드측 용의자.
4. 현재 워킹트리에 -1e-2가 그대로 남아 있음 → 다음 런 전 복원 필요(권고는 04에서).

> Evidence tier: §1–§4 = **확정**(diff 직접 인용). §3 마지막 -1e-2 floor 위험 = **정량 추론**(cfg calibration 주석 근거,
> 실제 floor 발동 여부는 학습로그(03)로 확증 필요).

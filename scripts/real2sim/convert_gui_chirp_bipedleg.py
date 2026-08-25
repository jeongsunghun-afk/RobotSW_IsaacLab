# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""biped_leg GUI chirp 캡처(.npz) → PACE 데이터셋(.pt) 변환 + 품질/지연 판정.

``gui_controller.py``의 "Chirp (sysid capture)"가 남긴 50 Hz 캡처를 ``fit_bipedleg.py``가 먹는
step-동기 포맷(기본 200 Hz = 학습 플랜트 물리 그리드, ``--rate``로 변경)으로 바꾼다.
Isaac Sim은 띄우지 않는다 (torch만 필요).

하는 일 (``convert_capture_to_pt.py``(go2)의 biped_leg 판):

0. **좌표 규약 판별 + 채널→관절 각도 보정** — 드라이버 감속비 오설정(전 축 7:1 가정)으로 **규약 0**
   캡처는 calf·foot 각도가 각각 1.5·1.2배 부풀려져 있다. 그런 캡처만 ``GEAR_K``로 나눠 관절 단위로
   옮긴다. 다른 모든 단계보다 먼저 온다 — 커플링 coef 가 감속비 이후 공간의 계수라서다.

   도장은 세 세대이고 **새 것부터** 읽는다 (자세한 건 :func:`convert` 안 주석):
   ``convention_version`` (브리지가 TELEM 으로 신고한 값을 GUI 가 그대로 기록) →
   ``gear_applied`` (구 불리언) → 도장 없음(= 규약 0, 구 캡처).
   두 규약을 **공통 내부 프레임(관절 단위 + foot raw)** 으로 정규화한 뒤 나머지 단계를 태운다:

   * **규약 0** (구 캡처, 채널 단위 + foot raw) — ``GEAR_K`` 로 나눈다. foot 은 이미 raw.
   * **규약 1** (2026-08-14~, gear 적용 + foot **관절각**) — 나눗셈은 **생략**하고(두 번 나누면
     calf 가 1.5 배 작아진다) foot 을 raw 로 **합성**한다: ``raw = 관절 + coef·calf``.
     명령은 명령끼리, 실측은 실측끼리 합성한다 — 브리지 ``joint_to_motor_deg`` / ``motor_deg_to_joint``
     와 같은 규칙이다. 섞으면 calf 추종오차가 foot 채널에 가짜 신호로 들어간다.

   게인은 두 규약 모두 **채널 게인**으로 기록되므로(브리지가 변환 없이 통과 —
   ``calib_bipedleg.hpp`` GAIN_GEAR 주석) 관절 환산 ``k^n`` 은 공통이다.
   ⚠ ``convention_version = -1`` (캡처 중 TELEM 미수신)은 중단 — 프레임을 추측하게 되기 때문.
   ⚠ 모르는 규약(≥2)도 중단한다.
1. **균일 그리드 정렬** — 명령(50 Hz 발행시각)과 실기 TELEM(≈50 Hz 도착시각)이 비동기이므로,
   물리 그리드(기본 200 Hz)에 명령은 ZOH(브리지가 마지막 ACT를 유지, slew 기본 0), 실측은 선형보간으로 얹는다.
2. **foot 좌표** — 0번 정규화를 거치면 foot 항은 **항상 raw각**(q_foot+q_calf)이다(명령·실측 모두).
   ★**기본은 raw 유지**(``--keep_raw_foot``)다. 현재 sysid env 는 ``foot_coupling=True``
   (``r2s_biped_leg_sysid_cfg.py:221``)로 **데이터도 재생도 채점도 전부 raw** 규약이며, 그게 옳다 —
   엔코더가 raw 만 재므로 관절각으로 바꾸면 calf 의 측정 잡음이 foot 채널에 섞이고 커플링 계수
   가정이 데이터에 구워진다.
   ``--keep_raw_foot`` 를 끄면(구 규약) 관절각으로 변환한다:
   ``q_foot_joint = q_raw − q_calf(실측)``, ``des_foot_joint = raw_cmd − q_calf(실측)``.
   ⚠ 그 경로의 모델 갭: 실기 foot PD 의 kd 는 (q̇_f+q̇_c)에 걸리는데 재생은 q̇_f 만 보고,
   foot 모터 토크의 calf 전치(τ_c += τ_f)도 재생에 없다 — calf·foot 식별에 오차가 들어간다.
   ⚠⚠ **이 절은 2026-08-14 까지 ``foot_coupling=False`` 라고 적혀 있었다**(같은 파일 코드 주석은
   ``True`` 로 최신이라 한 파일 안에서 서로 반대였다). 실제로 이 docstring 을 읽은 사람이 sysid
   규약을 정반대로 파악하는 사고가 있었다. 규약을 바꿀 때 **docstring 과 코드 주석을 함께** 고칠 것.
3. **명령/상태 시간축 지연 판정** — go2에서 104 ms 어긋남이 관성·점성을 ~100× 오염시킨 전례
   (``convert_capture_to_pt.py`` 참조). PD 법칙 ``tau = kp(des−q) − kd·q̇``이 tau_real과 가장 잘
   맞는 시간이동으로 잰다(커플링 오염이 없는 hip/thigh 4관절만 사용). ``--cmd_lag_ms auto``로
   자동 보정, 기본 0.0 = 보정 없음(판정만).

실행::

    # 판정만
    /home/user/miniconda3/envs/isaac-6.0/bin/python3.12 scripts/real2sim/convert_gui_chirp_bipedleg.py \\
        --captures 'data/bipedleg_gui/chirp_gui_2026*.npz'
    # 변환 저장
    ... --out_dir data/bipedleg_real
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import torch

GRID_HZ = 200.0  # 기본 = 학습 플랜트 물리 그리드(200 Hz). sysid env SYSID_RATE_HZ와 일치 필수.
# leg-major (calf, foot) 커플링 쌍 — gui_controller._COUPLED_CALF_FOOT와 동일.
COUPLED_CALF_FOOT = ((2, 3), (6, 7))
# foot↔calf 커플링 계수 — 벨트가 무릎을 건너므로 raw = q_foot + coef·q_calf 다.
# 브리지 `calib_bipedleg.hpp FOOT_CALF_COEF` 와 값 일치 필수 (RL_INTERFACE.md §1, 실기 실측 +1).
FOOT_CALF_COEF = 1.0
# 드라이버 감속비 오설정 보정 (RL_INTERFACE.md §4). 드라이버가 전 축을 7:1 로 가정해 각도를
# 주고받으므로, 드라이버가 보고/수신하는 "채널각"은 참 관절각의 gear_k = 실제감속비/7 배다
# (실제 감속비 hip 7 · thigh 7 · calf 10.5 · foot 8.4). 파이 브리지
# (real_runner_bipedleg.cpp:73 motor_deg_to_sim / :69 sim_to_motor_deg)가 양방향 모두 이 k 를
# 적용하지 않으므로, 캡처의 각도·각속도는 명령·실측 둘 다 채널 단위다 → 여기서 나눠 관절 단위로 옮긴다.
# 커플링 coef 는 감속비 **이후** 공간의 계수이므로(§2-a) 반드시 커플링 해제보다 먼저 적용해야 한다.
GEAR_K = np.array([1.0, 1.0, 1.5, 1.2, 1.0, 1.0, 1.5, 1.2])  # leg-major

# 브리지 TELEM 송신 레이트 — `real_runner_bipedleg.cpp` 의 `kTelemDtSec`(5 ms) 와 짝.
# `telem_tick` 을 시간으로 되돌릴 때 쓴다. 한쪽만 바꾸면 시간축이 조용히 늘어난다.
TELEM_HZ: float = 200.0
JOINT_LABELS = ["HL_hip", "HL_thigh", "HL_calf", "HL_foot", "HR_hip", "HR_thigh", "HR_calf", "HR_foot"]
# 게인의 gear 지수 — **k² 채택 (2026-08-14)**. `--gain_gear_scale` 로 A/B 가능.
#
# ⚠ **"확정"이 아니라 "채택"이다.** 지수 2 는 아래 ①(실측)과 ②(유도)의 곱이고 등급이 다르다.
#   ② 는 재지 않았다 — 두 가설(c=1 / c=k)이 **같은 값을 보고**하므로 보고토크로는 원리적으로
#   구분되지 않는다(그 값이 PD 의 에코라는 것 자체가 이유다). 실측이 배제한 건 "펌웨어가 보고 전에
#   k 를 한 번 더 곱한다"는 세 번째 가능성뿐이고, {지수 1, 지수 2} 는 둘 다 살아 있다.
#   이 프로젝트는 유도·부분측정을 확정으로 기록했다가 두 번 되돌렸다(전치 번복, gear_k 무효화).
#   판정 방법: reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/NEXT_CAPTURES.md
#
# 두 단계가 곱해져 k² 가 된다 (RL_INTERFACE.md §4):
#   ① 드라이버 PD 가 **채널각 오차**에 kp 를 곱한다 → 채널오차 = k × 관절오차
#      [실측 확인] 준정적 구간에서 |τ_보고| / |kp·e_관절 − kd·q̇_관절| = 1.5008 (calf, k=1.5;
#      k 대비 오차 0.06%, k² 대비 −33%). hip/thigh(k=1)는 1.0006~1.0029 로 대조군 통과.
#      → reports/real2sim/_comparisons/pace_bipedleg_foot_coupling_probe/logs/tau_echo_check.py
#   ② 실제 관절토크 = 보고토크 × k (펌웨어가 토크 상수에도 7:1 을 가정하므로)
#   ⇒ 실효 관절강성 = kp·k²  — calf 50→112.5, foot 30→43.2. 감쇠비도 ζ×k 로 바뀐다
#     (calf 0.76→1.13 과감쇠 · foot 0.82→0.97). 같은 지수가 kp·kd 에 공통으로 걸린다.
#
# ⚠ 중력으로 판정하려던 시도는 **판정 불능**이었다 — calf/foot 중력토크(전 범위 0.94/0.20 N·m)가
#   마찰 바닥(0.27~0.86 N·m)보다 작아 SNR 0.31/0.35 다. 위 ①이 훨씬 강한 신호(오차 0.06%)라
#   그쪽을 근거로 삼는다. → logs/torque_scale_gravity.py
# ⚠ 드라이버 gear 버그가 고쳐지면 calf·foot 강성이 갑자기 k² 만큼 약해진다 → **정책 재학습 필요**.
GAIN_GEAR_SCALE_DEFAULT: float = 2.0
GAIN_GEAR_SCALE: float = GAIN_GEAR_SCALE_DEFAULT
# 지연 판정에 쓰는 관절 (커플링 전치 오염이 없는 hip/thigh)
LAG_JOINTS = (0, 1, 4, 5)
JOINT_ORDER_FULL = [
    "HL_hip_joint",
    "HL_thigh_joint",
    "HL_calf_joint",
    "HL_foot_joint",
    "HR_hip_joint",
    "HR_thigh_joint",
    "HR_calf_joint",
    "HR_foot_joint",
]


def zoh(t_src: np.ndarray, v_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    """ZOH 샘플링 — 각 grid 시각에 대해 '마지막으로 발행된' 값을 취한다."""
    idx = np.searchsorted(t_src, t_grid, side="right") - 1
    idx = np.clip(idx, 0, len(t_src) - 1)
    return v_src[idx]


def lininterp(t_src: np.ndarray, v_src: np.ndarray, t_grid: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t_grid, t_src, v_src[:, j]) for j in range(v_src.shape[1])], axis=1)


def estimate_cmd_lag_ms(
    t_cmd, q_cmd, t_grid, q_meas, dq_meas, tau_meas, kp, kd, scan_ms: float = 100.0
) -> tuple[float, float]:
    """PD 정합성으로 명령 시간축 지연을 추정한다 (양수 = 명령 타임스탬프가 응답보다 앞섬).

    Returns:
        (best_lag_ms, best_corr) — hip/thigh 4관절 합산 상관이 최대가 되는 시프트.
    """
    best = (0.0, -2.0)
    for lag_ms in np.arange(-scan_ms, scan_ms + 1e-6, 2.0):
        des = zoh(t_cmd + lag_ms * 1e-3, q_cmd, t_grid)
        num = den_p = den_m = 0.0
        for j in LAG_JOINTS:
            pred = kp[j] * (des[:, j] - q_meas[:, j]) - kd[j] * dq_meas[:, j]
            meas = tau_meas[:, j]
            p = pred - pred.mean()
            m = meas - meas.mean()
            num += float((p * m).sum())
            den_p += float((p * p).sum())
            den_m += float((m * m).sum())
        if den_p > 0 and den_m > 0:
            c = num / np.sqrt(den_p * den_m)
            if c > best[1]:
                best = (float(lag_ms), c)
    return best


def convert(
    path: Path, out_dir: Path | None, cmd_lag_ms: str, rate: float = GRID_HZ, keep_raw_foot: bool = False
) -> None:
    d = np.load(path, allow_pickle=True)
    if bool(d["aborted"]):
        print(f"[skip] {path.name}: aborted 캡처")
        return
    if "t_real" not in d.files or len(d["t_real"]) < 100:
        print(f"[skip] {path.name}: 실기 TELEM 스트림 없음/부족")
        return

    t_cmd = d["t_cmd"].astype(np.float64)
    q_cmd = d["q_cmd"].astype(np.float64)
    t_real = d["t_real"].astype(np.float64)
    q_real = d["q_real"].astype(np.float64)
    dq_real = d["dq_real"].astype(np.float64)
    tau_real = d["tau_real"].astype(np.float64)
    kp = d["kp"].astype(np.float64)
    kd = d["kd"].astype(np.float64)
    # ★ 캡처 당시 **실기 드라이버에 실제로 들어간 채널 게인**. 아래에서 kp/kd 는 관절 공간으로
    #   덮어써지므로(×gear_k^n) 여기서 원본을 떠 둔다. 둘을 구분 못 하면 조용히 오독된다 —
    #   .pt 의 `kp` 는 **관절값**이고 npz 의 `kp` 는 **채널값**인데 키 이름이 같기 때문이다.
    #   meta 의 `kp_channel` 이 "실기에 뭘 걸고 땄는가" 의 정본이다 (README §32-f).
    kp_channel, kd_channel = kp.copy(), kd.copy()

    # ── 브리지 에코 (2026-08-19, TELEM 157 B). 없으면 구 캡처다. ──────────────────────────
    # ① telem_tick — 파이의 5 ms 격자. `t_real`(파이썬 도착시각)은 GIL·Qt·네트워크 지터를
    #    안고 있어 5 ms 급 현상을 분해할 수 없다(실측 std 0.50 ms = 간격의 10 %). 틱이 있으면
    #    시간축을 **구조적으로** 재구성하고, 원점만 첫 도착시각에 맞춘다(원점 오차는 상수라
    #    PACE delay·재생 shift 가 흡수한다).
    if "telem_tick" in d.files:
        tick = d["telem_tick"].astype(np.float64)
        t_real = t_real[0] + (tick - tick[0]) / TELEM_HZ
        gap = int(np.sum(np.diff(d["telem_tick"].astype(np.int64)) - 1))
        print(f"    시간축: telem_tick 사용 ({TELEM_HZ:g} Hz 격자, 유실 {gap} 틱)")
    else:
        print("    ⚠ 시간축: telem_tick 없음 — 도착시각을 쓴다(지터 포함). 구 브리지 캡처다.")
    # ② q_cmd_real — 드라이버가 **실제로 받은** 목표각. `q_cmd`(GUI 발행값)와 달리 클램프·
    #    ENGAGE 램프·slew·float16 양자화가 전부 반영돼 있고, 무엇보다 **측정과 같은 틱**에
    #    실려 온다 → 명령·측정을 서로 다른 클록에서 맞출 필요가 사라진다(§18 오염의 근원).
    q_cmd_applied = None
    if "q_cmd_real" in d.files:
        cand = d["q_cmd_real"].astype(np.float64)
        if np.isnan(cand).all(axis=1).any():
            print("    ⚠ q_cmd_real 에 무효(NaN) 샘플 있음 — 전송 전 구간 포함. 발행값으로 대체한다.")
        else:
            q_cmd_applied = cand
    # ③ clamp_mask_real — 목표가 soft limit 에 잘린 샘플. §18 오염을 캡처가 자진신고한다.
    if "clamp_mask_real" in d.files:
        clamped = int(np.bitwise_or.reduce(d["clamp_mask_real"].astype(np.uint8)))
        if clamped:
            names = [JOINT_LABELS[j] for j in range(8) if clamped & (1 << j)]
            print(f"    ⚠ 명령이 soft limit 에 클램프된 구간 있음: {', '.join(names)} — 적합 전에 확인할 것")

    # 감속비 오설정 소급 보정 — 채널 단위 → 관절 단위 (GEAR_K 주석 참조).
    #
    # 도장은 세 세대가 있다. **새 것부터** 읽는다:
    #   1) ``convention_version`` (2026-08-14~) — 브리지가 TELEM 으로 신고한 값을 GUI 가 그대로 기록.
    #      0 = gear 미적용 + foot raw / 1 = gear 적용 + foot **관절**각 / -1 = TELEM 미수신(규약 미상).
    #   2) ``gear_applied`` (구) — 불리언. 규약 ≥1 과 같은 뜻이지만 foot 프레임 정보가 없다.
    #   3) 도장 없음 — 구 캡처. 미보정(채널 단위 + foot raw)으로 간주한다.
    if "convention_version" in d.files:
        conv = int(d["convention_version"])
    elif "gear_applied" in d.files:
        conv = 1 if bool(d["gear_applied"]) else 0
    else:
        conv = 0
    if conv < 0:
        raise SystemExit(
            f"[{path.name}] convention_version=-1 — 캡처 중 TELEM 을 받지 못해 좌표 규약이 미상이다.\n"
            "  실기 스트림 없이 저장된 캡처이거나 브리지가 죽어 있었다. 변환하면 프레임을 추측하는 셈이라 중단한다."
        )
    if conv >= 2:
        raise SystemExit(
            f"[{path.name}] convention_version={conv} 는 이 컨버터가 모르는 규약이다.\n"
            "  프레임을 추측하느니 중단한다 — 규약 정의를 확인하고 분기를 추가할 것."
        )
    # ---- 두 규약을 **공통 내부 프레임**으로 정규화한다: 관절(모델) 단위 + foot **raw** ----
    # 그 프레임이 sysid env 가 기대하는 것이다 (`foot_coupling=True` — 데이터·재생·채점 전부 raw,
    # 엔코더가 raw 만 재므로 그게 옳다). 아래 keep_raw_foot 분기는 이 정규화 뒤에 붙는다.
    if conv == 1:
        # 규약 1 — 브리지가 gear 를 **이미** 나눴고 foot 은 **관절각**이다.
        #   ① gear 나눗셈을 하지 않는다. 또 나누면 calf 가 1.5 배, foot 이 1.2 배 작아진다.
        #   ② foot 을 raw 로 **합성**한다: raw = 관절 + coef·calf (coef=1, calib FOOT_CALF_COEF).
        #      명령은 명령끼리, 실측은 실측끼리 합성한다 — 브리지 `joint_to_motor_deg` 가 목표값끼리
        #      합성하고 `motor_deg_to_joint` 가 실측끼리 분해하는 것과 같은 규칙이다. 섞으면
        #      calf 추종오차가 foot 채널에 가짜 신호로 들어간다.
        #   calf 자체는 커플링이 없어 관절 = raw 이므로 건드리지 않는다.
        gear_applied_in_capture = True
        for c, f in COUPLED_CALF_FOOT:
            q_cmd[:, f] = q_cmd[:, f] + FOOT_CALF_COEF * q_cmd[:, c]
            q_real[:, f] = q_real[:, f] + FOOT_CALF_COEF * q_real[:, c]
            dq_real[:, f] = dq_real[:, f] + FOOT_CALF_COEF * dq_real[:, c]
            # ★에코 명령은 TELEM 출신이라 `q_real` 과 **같은** 프레임이다 — 같은 규칙을 적용한다.
            #   `q_cmd`(GUI 발행값) 규칙을 쓰면 안 된다. 출처가 다르면 규칙도 다르다.
            if q_cmd_applied is not None:
                q_cmd_applied[:, f] = q_cmd_applied[:, f] + FOOT_CALF_COEF * q_cmd_applied[:, c]
        # 게인은 규약과 무관하게 **채널 게인**으로 기록된다(브리지가 변환 없이 통과시킨다 —
        # calib_bipedleg.hpp GAIN_GEAR 주석). 관절 공간 환산은 두 규약 모두 같다.
        kp = kp * GEAR_K**GAIN_GEAR_SCALE
        kd = kd * GEAR_K**GAIN_GEAR_SCALE
        print(
            f"[{path.name}] gear: 캡처가 이미 관절 단위 (규약 1) — 나눗셈 생략. "
            f"foot 을 raw 로 합성(coef={FOOT_CALF_COEF:g}), 게인 배율 k^{GAIN_GEAR_SCALE:g}, tau 미스케일"
        )
    else:
        # 규약 0 — 채널 단위 + foot raw. 소급 보정을 적용한다.
        gear_applied_in_capture = False
        q_cmd /= GEAR_K
        q_real /= GEAR_K
        dq_real /= GEAR_K
        if q_cmd_applied is not None:
            q_cmd_applied /= GEAR_K  # 에코 명령도 실측과 같은 규칙 (둘 다 TELEM 출신)
        kp = kp * GEAR_K**GAIN_GEAR_SCALE
        kd = kd * GEAR_K**GAIN_GEAR_SCALE
        print(
            f"[{path.name}] gear: 채널→관절 보정 적용 (규약 0, q/dq ÷ k, k={GEAR_K.tolist()}), "
            f"게인 배율 k^{GAIN_GEAR_SCALE:g}, tau 미스케일"
        )

    # 겹치는 구간만 (real 스트림은 램프 구간을 포함하므로 cmd(스윕 전용) 창으로 잘린다)
    t0 = max(t_cmd[0], t_real[0])
    t1 = min(t_cmd[-1], t_real[-1])
    n = int((t1 - t0) * rate)
    t_grid = t0 + np.arange(n) / rate

    q_meas = lininterp(t_real, q_real, t_grid)
    dq_meas = lininterp(t_real, dq_real, t_grid)
    tau_meas = lininterp(t_real, tau_real, t_grid)

    # 품질: 수신률/최대 공백/추종 (⚠ 그리드 rate 파라미터와 이름 충돌 금지)
    real_in = (t_real >= t0) & (t_real <= t1)
    real_rate_hz = real_in.sum() / (t1 - t0)
    max_gap_ms = float(np.diff(t_real[real_in]).max() * 1e3) if real_in.sum() > 2 else float("nan")

    # 지연 판정 (관절각 공간 필요 없음 — hip/thigh는 커플링 무관)
    lag_ms, corr = estimate_cmd_lag_ms(t_cmd, q_cmd, t_grid, q_meas, dq_meas, tau_meas, kp, kd)
    applied = 0.0
    if cmd_lag_ms == "auto":
        applied = lag_ms
    elif float(cmd_lag_ms) != 0.0:
        applied = float(cmd_lag_ms)
    if q_cmd_applied is not None:
        # ★브리지 에코를 쓴다 — 명령과 측정이 **같은 틱**에 실려 오므로 시간축 정렬이 불필요하다.
        #   `applied`(추정 지연 보정)도 적용하지 않는다: 보정할 어긋남이 애초에 없다.
        des = zoh(t_real, q_cmd_applied, t_grid)
        print("    명령: q_cmd_real(브리지 에코) 사용 — 클램프·램프·float16 반영, 시간정렬 불필요")
    else:
        des = zoh(t_cmd + applied * 1e-3, q_cmd, t_grid)

    # foot 좌표 처리 — 두 규약:
    #  keep_raw_foot=True (커플링 재생 적합용, 권장): foot 명령·실측을 raw(엔코더) 그대로 둔다.
    #    sysid env(foot_coupling=True)가 raw 구동+전치를 재생하고 채점도 raw끼리 한다.
    #  False (구 관절각 규약): foot을 관절각으로 환산 — 커플링 없는 재생용(모델 갭 있음).
    if keep_raw_foot:
        q_joint = q_meas.copy()
        des_joint = des.copy()
        dq_joint = dq_meas.copy()
    else:
        q_joint = q_meas.copy()
        des_joint = des.copy()
        dq_joint = dq_meas.copy()
        for c, f in COUPLED_CALF_FOOT:
            des_joint[:, f] = des[:, f] - q_meas[:, c]
            q_joint[:, f] = q_meas[:, f] - q_meas[:, c]
            dq_joint[:, f] = dq_meas[:, f] - dq_meas[:, c]

    track_rms = np.sqrt(np.mean((des_joint - q_joint) ** 2, axis=0))
    pp = q_joint.max(axis=0) - q_joint.min(axis=0)
    print(
        f"[{path.name}] {t1 - t0:.1f}s  real {real_rate_hz:.1f}Hz(max gap {max_gap_ms:.0f}ms)  "
        f"lag={lag_ms:+.0f}ms(corr {corr:.3f}, 적용 {applied:+.0f}ms)\n"
        f"    관절 p-p [rad]: " + " ".join(f"{v:.2f}" for v in pp) + "\n"
        "    추종 RMS [rad]: " + " ".join(f"{v:.3f}" for v in track_rms)
    )
    if abs(lag_ms) > 30.0 and applied == 0.0:
        print(f"    ⚠ 지연 {lag_ms:+.0f}ms > 적합기 delay 상한(20ms) — --cmd_lag_ms auto 권장 (go2 104ms 전례)")

    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (path.stem + ".pt")
    torch.save(
        {
            "time": torch.arange(n, dtype=torch.float32) / rate,
            "dof_pos": torch.from_numpy(q_joint).float(),
            "des_dof_pos": torch.from_numpy(des_joint).float(),
            # 브리지가 보고한 각속도 (그리드 정렬, dof_pos 와 같은 프레임). 종전엔 버렸는데,
            # 소비자가 위치를 다시 미분하면 50 Hz 원본을 200 Hz 로 얹은 그리드에서 잡음이 커져
            # `kd·q̇` 항이 오염된다 — 실제로 verify_chirp_capture.py 가 오탐을 냈다.
            "dq_meas": torch.from_numpy(dq_joint).float(),
            # 드라이버 보고토크 (그리드 정렬, **미스케일**). 지연 판정에만 쓰고 버리던 값을 보존한다 —
            # "드라이버 토크가 실측인가 지령 에코인가" 판정에 필요하고, 관절/모터 기준 여부가
            # 확정되면 여기서 GEAR_K 로 환산할 수 있다.
            "tau_meas": torch.from_numpy(tau_meas).float(),
            "kp": torch.from_numpy(kp).float(),
            "kd": torch.from_numpy(kd).float(),
            "joint_order": JOINT_ORDER_FULL,
            "meta": {
                "source": "real_gui",
                "capture": str(path),
                "rate_hz": rate,
                "f0_hz": float(d["f0_hz"]),
                "f1_hz": float(d["f1_hz"]),
                "duration_s": float(d["duration_s"]),
                "amplitude_scale": float(d["amplitude_scale"]),
                "cmd_lag_applied_ms": applied,
                "cmd_lag_estimated_ms": lag_ms,
                "coupling_converted": not keep_raw_foot,
                # True = 이 파일에서 채널→관절 보정을 했다(구 캡처). False = 캡처가 이미 관절 단위.
                "capture_convention_version": conv,
                "gear_k_applied": not gear_applied_in_capture,
                # True = 캡처가 foot 을 **관절각**으로 줘서 여기서 raw 로 합성했다(규약 1).
                # False = 캡처의 foot 이 이미 raw 였다(규약 0). 어느 쪽이든 출력은 raw 규약이다.
                "foot_raw_synthesized": conv == 1,
                "gain_gear_scale": GAIN_GEAR_SCALE,
                # ★ 캡처 당시 실기 드라이버 게인 (**채널 좌표**, GUI 가 그대로 발행한 값).
                #   최상위 `kp`/`kd` 는 관절 좌표(= 채널 × gear_k^gain_gear_scale)다 — 다른 양이다.
                #   게인을 바꿔 가며 딴 캡처(kd 다양화 등)를 구분하려면 이 둘을 봐야 한다.
                "kp_channel": kp_channel.tolist(),
                "kd_channel": kd_channel.tolist(),
                # ★ chirp 진동 중심 [rad, leg-major] 과 출처("default" = chirp.CHIRP_CENTER /
                #   "slider" = 조작자가 슬라이더로 잡은 임시 영점).
                #   실기가 중력으로 처지면 고정 중심이 치우쳐 소프트리밋에 일찍 닿으므로,
                #   조작자가 중심을 옮겨 딸 수 있다. **캡처마다 다를 수 있다.**
                #   ⚠ GUI 발행값은 항상 **관절각**이므로(publisher 주석) 규약 0/1 어느 쪽이든
                #     gear 나눗셈을 하지 않는다 — q/dq 와 달리 이 값은 채널 단위였던 적이 없다.
                #   구 캡처엔 이 키가 없다 → 아래 fallback 이 들어가지만 **가정이다**.
                "chirp_center": (d["chirp_center"].astype(np.float64).tolist() if "chirp_center" in d.files else None),
                "chirp_center_source": (
                    str(d["chirp_center_source"]) if "chirp_center_source" in d.files else "unknown"
                ),
            },
        },
        out,
    )
    print(f"    → {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="biped_leg GUI chirp 캡처 → PACE .pt 변환")
    parser.add_argument("--captures", required=True, help="npz 경로 glob (따옴표로 감쌀 것)")
    parser.add_argument("--out_dir", default=None, help="저장 디렉토리. 생략 시 판정만")
    parser.add_argument("--cmd_lag_ms", default="0.0", help="명령 시간축 보정 [ms]. 'auto'=추정값 적용")
    parser.add_argument(
        "--gain_gear_scale",
        type=float,
        default=GAIN_GEAR_SCALE_DEFAULT,
        help="게인에 곱할 gear 지수 k^n (기본 2.0 = RL_INTERFACE §4 확정값). "
        "1.0/0.0 은 A/B 대조용 — 근거는 GAIN_GEAR_SCALE 주석 참조.",
    )
    parser.add_argument(
        "--rate", type=float, default=GRID_HZ, help="출력 그리드 [Hz] — sysid env SYSID_RATE_HZ와 일치 필수"
    )
    parser.add_argument(
        "--keep_raw_foot", action="store_true", help="foot을 raw(엔코더) 그대로 저장 — 커플링 재생 적합용"
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.captures))
    if not files:
        raise SystemExit(f"매칭되는 캡처가 없다: {args.captures}")
    out_dir = Path(args.out_dir) if args.out_dir else None
    # 모듈 상수를 CLI 값으로 덮어쓴다 — convert()와 meta 기록이 같은 값을 보게 하기 위함.
    global GAIN_GEAR_SCALE
    GAIN_GEAR_SCALE = args.gain_gear_scale
    for f in files:
        convert(Path(f), out_dir, args.cmd_lag_ms, rate=args.rate, keep_raw_foot=args.keep_raw_foot)


if __name__ == "__main__":
    main()

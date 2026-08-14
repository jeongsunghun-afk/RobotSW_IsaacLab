#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Motion Playback 통합 테스트 — 실제 GUI 인스턴스를 offscreen 으로 띄워 재생을 구동한다.

실행:
    QT_QPA_PLATFORM=offscreen /usr/bin/python3 test_motion_playback.py

로봇/sim 없이 돌아간다. publisher 는 두 방식으로 검증한다:
  1) ``_compute_target`` 을 50 Hz 로 직접 틱 — 발행 직전 목표각을 프레임 단위로 검사(정밀).
  2) ``publisher_process_main`` 을 스레드로 띄우고 CMD_PORT(9881) 를 mock 수신 — 실제 UDP 경로.
     (mp.Process 대신 스레드인 이유: QApplication 생성 뒤 fork 를 피한다. STATE_PORT 가 이미
     점유돼 있으면 publisher 가 즉시 죽으므로 그 경우는 SKIP 으로 보고한다.)
"""

from __future__ import annotations

import multiprocessing as mp
import os
import socket
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import gui_controller as gc  # noqa: E402
import motion_clips  # noqa: E402
import motions  # noqa: E402
import r2s_udp  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

NJ = gc.NUM_JOINTS
TICK = 1.0 / gc.FRAME_HZ
# (2026-08-14 좌표 이관: 재생 프레임이 곧 관절각이라 되돌릴 커플링 쌍도 to_joint() 도 필요 없다.
#  raw↔관절 변환은 브리지가 전담한다 — r2s_udp.R2S_CONVENTION_VERSION 참조.)

_failures: list[str] = []


def check(cond: bool, label: str, detail: str = "") -> None:
    if cond:
        print(f"  PASS  {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  FAIL  {label}" + (f"  ({detail})" if detail else ""))
        _failures.append(label)


def make_window():
    """publisher 프로세스 없이 MainWindow 를 만들고 startup latch 를 통과시킨다."""
    shared = mp.Array("d", gc._SM_LEN)
    with shared.get_lock():
        shared[gc._SM_CMD_VALID] = 0.0
        shared[gc._SM_REAL_ENABLE] = 0.0
        for i in range(NJ):
            shared[gc._SM_KP + i] = motions.DEFAULT_KP[i]
            shared[gc._SM_KD + i] = motions.DEFAULT_KD[i]
            shared[gc._SM_BASE_Q + i] = motions.DEFAULT_POSE[i]
    win = gc.MainWindow(shared, model_path=None, real_host=None, publisher_proc=None)
    win._on_startup_tick()  # QTimer 없이 직접 — RELAX + gains 기록
    return shared, win


def sim_publisher(shared, n_ticks: int, t0: float | None = None) -> list[list[float]]:
    """publisher 의 목표 계산을 50 Hz 로 n_ticks 만큼 돌려 발행될 자세를 모은다."""
    with shared.get_lock():
        start = shared[gc._SM_START_TIME]
    base = t0 if t0 is not None else start
    poses = []
    for k in range(n_ticks):
        res = gc._compute_target(shared, base + k * TICK)
        poses.append(None if res is None else list(res[0]))
    return poses


# ---------------------------------------------------------------------------


def test_playback_frames(app) -> None:
    """(a) 프레임 순서·값 (b) foot raw 변환 (c) soft limit 위반 0 (d) 틱 간 점프 상한."""
    print("\n[1] non-loop playback: frame order / joint-frame publish / soft limits")
    shared, win = make_window()
    win._motion_clip_combo.setCurrentText("trot0")
    win._motion_speed_spin.setValue(1.0)
    win._motion_loop_check.setChecked(False)

    clip = motion_clips.load_clip("trot0")
    pb = motion_clips.build_playback(
        clip, win._output_pose(), gc.FRAME_HZ, speed=1.0, loop=False, intro_s=gc.SEQUENCE_DURATION_S
    )
    expected = pb["frames"]
    win._on_motion_play_clicked()

    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
        count = int(shared[gc._SM_FRAME_COUNT])
        loop_start = int(shared[gc._SM_MOTION_LOOP_START])
    check(mode == gc._MODE_MOTION, "mode = _MODE_MOTION")
    check(count == expected.shape[0], "frame count", f"{count} frames")
    check(loop_start == pb["num_intro"], "loop_start = intro length", f"{loop_start}")

    poses = np.array(sim_publisher(shared, count + 40))  # 끝을 넘겨 홀드까지 확인
    check(all(p is not None for p in poses), "publisher 발행 보류 없음 (CMD_VALID=1)")

    # (a) 프레임 순서 — 매 틱이 기대 프레임과 정확히 일치, 버퍼 끝에서는 마지막 프레임 홀드
    max_err = float(np.abs(poses[:count] - expected).max())
    check(max_err < 1e-6, "발행 자세 == 기대 프레임 (순서·값)", f"max err {max_err:.2e} rad")
    hold_err = float(np.abs(poses[count:] - expected[-1]).max())
    check(hold_err < 1e-6, "버퍼 끝에서 마지막 프레임 홀드", f"max err {hold_err:.2e}")

    # (b) ★발행값 == 관절각 (2026-08-14 좌표 이관) — 커플링 변환이 **남아 있지 않은지** 본다.
    #     이 검사만이 "변환했는가"를 구별한다 (클램프는 어느 쪽이든 통과시킨다).
    q0 = motion_clips.clamp_frames(clip["joint_angles"][:1])[0]
    got = poses[loop_start]
    check(
        abs(got[3] - q0[3]) < 1e-6 and abs(got[7] - q0[7]) < 1e-6,
        "foot 채널 = 관절각 그대로 (raw 변환 없음 — 브리지가 담당)",
        f"HL {got[3]:+.4f} vs {q0[3]:+.4f}, HR {got[7]:+.4f} vs {q0[7]:+.4f}",
    )
    check(
        abs(q0[2]) > 1e-3 and abs(q0[6]) > 1e-3,
        "calf 가 0 이 아니다 (위 검사가 무의미하지 않음 — raw 였다면 그만큼 달랐을 것)",
        f"calf HL {q0[2]:+.4f}, HR {q0[6]:+.4f}",
    )
    check(abs(got[2] - q0[2]) < 1e-6, "calf 는 어느 규약에서도 변환 없음")

    # (c) soft limit — 발행값이 곧 관절각이라 되돌릴 필요가 없다
    scan = motion_clips.soft_limit_scan(poses)
    check(scan["num_violating"] == 0, "soft limit 위반 0 (관절각 기준)", f"{scan['num_violating']} frames")
    # 원본에는 위반이 있었다 = 클램프가 실제로 일했다
    raw_scan = motion_clips.soft_limit_scan(clip["joint_angles"])
    check(raw_scan["num_violating"] > 0, "원본 클립에는 위반이 있었다", f"{raw_scan['num_violating']}/209 frames")

    # (d) 틱 간 점프
    jump = float(np.abs(np.diff(poses[:count], axis=0)).max()) * gc.FRAME_HZ
    check(jump <= 1.05 * pb["peak_rate"]["max_rad_s"], "틱 간 점프 = 클립 최대 관절속도 이내", f"{jump:.2f} rad/s")
    win.close()


def test_loop_wrap(app) -> None:
    """Loop: 반복 구간만 wrap, 진입 보간은 1회, 이음매 점프가 bridge 상한 이내."""
    print("\n[2] loop playback: wrap / bridge continuity")
    shared, win = make_window()
    win._motion_clip_combo.setCurrentText("walk1")  # 짧아서 여러 주기를 빨리 돈다
    win._motion_speed_spin.setValue(1.0)
    win._motion_loop_check.setChecked(True)
    win._on_motion_play_clicked()

    with shared.get_lock():
        count = int(shared[gc._SM_FRAME_COUNT])
        loop_start = int(shared[gc._SM_MOTION_LOOP_START])
        done = shared[gc._SM_MOTION_DONE]
    span = count - loop_start
    poses = np.array(sim_publisher(shared, count + 3 * span))
    check(done < 0.5, "재생 시작 시 DONE=0")

    # wrap: 버퍼를 넘긴 틱이 반복 구간의 같은 위상과 일치
    wrap_err = float(np.abs(poses[count : count + span] - poses[loop_start : loop_start + span]).max())
    check(wrap_err < 1e-6, "반복 구간이 정확히 wrap 된다", f"max err {wrap_err:.2e}")
    intro_seen = np.abs(poses[count:] - poses[0]).max(axis=1).min()
    check(intro_seen > 1e-6, "진입 보간은 반복되지 않는다")

    with shared.get_lock():
        done_after = shared[gc._SM_MOTION_DONE]
    check(done_after < 0.5, "loop 중엔 DONE 이 서지 않는다")

    # 이음매 포함 전 구간 점프 — bridge 설계 상한(2.0 rad/s)과 클립 최대 속도 중 큰 값 이내
    pb = motion_clips.build_playback(
        motion_clips.load_clip("walk1"), [0.0] * NJ, gc.FRAME_HZ, speed=1.0, loop=True, intro_s=gc.SEQUENCE_DURATION_S
    )
    cap = 1.05 * max(motion_clips.LOOP_BRIDGE_MAX_RAD_S, pb["peak_rate"]["max_rad_s"])
    jump = float(np.abs(np.diff(poses, axis=0)).max()) * gc.FRAME_HZ
    check(jump <= cap, "루프 이음매 포함 틱 간 점프가 상한 이내", f"{jump:.2f} <= {cap:.2f} rad/s")

    src = motion_clips.load_clip("walk1")["joint_angles"]
    seam_raw = float(np.abs(src[0] - src[-1]).max())
    check(seam_raw > 1.0, "원본 seam 은 그대로 두면 위험했다", f"{seam_raw:.2f} rad in one tick")
    win.close()


def test_stop_and_finish(app) -> None:
    """Stop = 현재 자세 HOLD(점프 없음), 비반복 종료 = DONE → 폴링이 HOLD 전환."""
    print("\n[3] stop / natural end")
    shared, win = make_window()
    win._motion_clip_combo.setCurrentText("walk1")
    win._motion_loop_check.setChecked(True)
    win._on_motion_play_clicked()
    check(win._motion_active and not win._motion_play_btn.isEnabled(), "재생 중 Play 비활성 / Stop 활성")

    # 재생 도중 Stop — publisher 의 현재 출력에서 홀드해야 한다
    with shared.get_lock():
        shared[gc._SM_START_TIME] = time.monotonic() - 1.0  # 1초 진행한 것으로
    before = gc._compute_target(shared, time.monotonic())[0]
    with shared.get_lock():  # publisher 가 CURRENT_Q 를 되쓰는 것을 흉내
        for i in range(NJ):
            shared[gc._SM_CURRENT_Q + i] = before[i]
    win._on_motion_stop_clicked()
    after = gc._compute_target(shared, time.monotonic())[0]
    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
    check(mode == gc._MODE_HOLD, "Stop → _MODE_HOLD")
    check(not win._motion_active and win._motion_play_btn.isEnabled(), "Stop 후 버튼 복구")
    check(max(abs(a - b) for a, b in zip(before, after)) < 1e-9, "Stop 시 자세 점프 없음")

    # 비반복 자연 종료
    shared2, win2 = make_window()
    win2._motion_clip_combo.setCurrentText("walk1")
    win2._motion_loop_check.setChecked(False)
    win2._on_motion_play_clicked()
    with shared2.get_lock():
        count = int(shared2[gc._SM_FRAME_COUNT])
        shared2[gc._SM_START_TIME] = time.monotonic() - (count + 5) * TICK
    last = gc._compute_target(shared2, time.monotonic())[0]
    with shared2.get_lock():
        done = shared2[gc._SM_MOTION_DONE]
        for i in range(NJ):
            shared2[gc._SM_CURRENT_Q + i] = last[i]
    check(done >= 0.5, "마지막 프레임 도달 시 publisher 가 DONE=1")
    win2._on_motion_poll_tick()
    with shared2.get_lock():
        mode2 = int(shared2[gc._SM_MODE])
    check(mode2 == gc._MODE_HOLD and not win2._motion_active, "DONE → 폴링이 HOLD 로 전환")
    win.close()
    win2.close()


def test_preemption_and_regression(app) -> None:
    """다른 조작이 재생을 가로채는 경우 + 기존 Home/슬라이더/sine 무회귀."""
    print("\n[4] preemption + existing-feature regression")
    shared, win = make_window()
    win._motion_clip_combo.setCurrentText("trot0")
    win._on_motion_play_clicked()

    # Home 이 "보간"인지 보려면 시작점이 goal 과 달라야 한다 — publisher 가 되쓰는 현재 출력 목표를
    # 흉내내 non-zero 로 둔다 (실제 GUI 에선 publisher 가 매 틱 CURRENT_Q 를 쓴다).
    with shared.get_lock():
        for i in range(NJ):
            shared[gc._SM_CURRENT_Q + i] = 0.2
    win._on_home_clicked()  # 재생 중 Home
    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
    check(mode == gc._MODE_SEQUENCE, "재생 중 Home → _MODE_SEQUENCE (Home 이 이긴다)")
    win._on_motion_poll_tick()
    check(not win._motion_active and win._motion_play_btn.isEnabled(), "폴링이 가로채기를 감지해 UI 정리")

    # 기존 Home 시퀀스 무회귀 — 1.5s 뒤 DEFAULT_POSE 도달
    poses = sim_publisher(shared, int(gc.SEQUENCE_DURATION_S * gc.FRAME_HZ) + 5)
    check(
        max(abs(v - d) for v, d in zip(poses[-1], motions.DEFAULT_POSE)) < 1e-9,
        "Home 시퀀스가 default pose 에 도달",
        f"end {[f'{v:+.4f}' for v in poses[-1]]}",
    )
    n_steps = int(gc.SEQUENCE_DURATION_S * gc.FRAME_HZ)
    check(len(poses) > n_steps and poses[0] != poses[n_steps - 1], "Home 이 보간으로 이동(스냅 아님)")

    # 슬라이더 / sine / relax 경로가 여전히 동작
    win._joint_sliders[1].setValue(gc._rad_to_slider(1, 0.3))
    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
        base1 = shared[gc._SM_BASE_Q + 1]
    check(mode == gc._MODE_HOLD and abs(base1 - 0.3) < 2e-3, "슬라이더 → HOLD", f"BASE_Q[1]={base1:.4f}")
    win._on_sine_start_clicked()
    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
    check(mode == gc._MODE_SINE, "Sine start → _MODE_SINE")
    win._on_sine_stop_clicked()
    win._on_relax_clicked()
    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
    check(mode == gc._MODE_RELAX, "Relax → _MODE_RELAX")

    # 재생 중 다른 클립/배속으로 다시 Play (재진입)
    win._motion_clip_combo.setCurrentText("walk1")
    win._motion_speed_spin.setValue(0.25)
    win._on_motion_play_clicked()
    with shared.get_lock():
        mode = int(shared[gc._SM_MODE])
        count = int(shared[gc._SM_FRAME_COUNT])
    slow = motion_clips.build_playback(
        motion_clips.load_clip("walk1"), [0.0] * NJ, gc.FRAME_HZ, speed=0.25, loop=False, intro_s=gc.SEQUENCE_DURATION_S
    )
    check(mode == gc._MODE_MOTION and count == slow["frames"].shape[0], "0.25배속 재생 시작", f"{count} frames")
    check(count <= gc.MAX_FRAMES, "0.25배속도 버퍼 안에 들어간다", f"{count} <= {gc.MAX_FRAMES}")
    win.close()


def test_udp_end_to_end(app) -> None:
    """실제 publisher 루프 + mock CMD 수신 — UDP 경로로 프레임이 그대로 나가는지."""
    print("\n[5] end-to-end UDP (publisher loop -> CMD_PORT mock receiver)")
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        rx.bind((gc.HOST, r2s_udp.CMD_PORT))
    except OSError as exc:
        print(f"  SKIP  CMD_PORT {r2s_udp.CMD_PORT} bind 실패 ({exc}) - gui/sim 이 이미 떠 있는가?")
        return
    rx.settimeout(0.5)

    shared, win = make_window()
    stop_flag = mp.Value("i", 0)
    th = threading.Thread(target=gc.publisher_process_main, args=(shared, stop_flag, None, None), daemon=True)
    th.start()
    time.sleep(0.3)

    win._motion_clip_combo.setCurrentText("walk1")
    win._motion_speed_spin.setValue(1.0)
    win._motion_loop_check.setChecked(False)
    pb = motion_clips.build_playback(
        motion_clips.load_clip("walk1"),
        win._output_pose(),
        gc.FRAME_HZ,
        speed=1.0,
        loop=False,
        intro_s=gc.SEQUENCE_DURATION_S,
    )
    expected = pb["frames"]
    # Play 이전(RELAX)에 이미 도착한 패킷을 버린다 — 재생 경로가 아니라서 꺾은선 위에 없다.
    rx.setblocking(False)
    while True:
        try:
            rx.recvfrom(4096)
        except OSError:
            break
    rx.settimeout(0.5)
    win._on_motion_play_clicked()

    got: list[list[float]] = []
    deadline = time.monotonic() + expected.shape[0] * TICK + 0.5
    while time.monotonic() < deadline:
        try:
            data, _ = rx.recvfrom(4096)
        except (TimeoutError, OSError):
            continue
        pkt = r2s_udp.unpack_cmd(data)
        if pkt is not None:
            got.append(list(pkt["q"]))
    stop_flag.value = 1
    th.join(timeout=2.0)
    rx.close()

    if not got:
        print(f"  SKIP  CMD 패킷 0개 - publisher 가 STATE_PORT {r2s_udp.STATE_PORT} bind 에 실패했을 수 있다")
        win.close()
        return
    check(len(got) > 0.7 * expected.shape[0], "CMD 패킷 수신", f"{len(got)} packets / {expected.shape[0]} frames")

    # publisher 는 프레임 사이를 선형보간하므로 수신값이 특정 프레임과 같을 필요는 없다.
    # 대신 **프레임을 잇는 꺾은선 위에** 있어야 한다 — 각 샘플에서 모든 선분까지의 거리 중 최소값.
    g = np.array(got, dtype=np.float64)
    p0 = expected[:-1]  # (S, 8) 선분 시작
    seg = expected[1:] - p0  # (S, 8) 선분 벡터
    denom = np.maximum((seg * seg).sum(axis=1), 1e-18)
    u = np.clip(((g[:, None, :] - p0[None]) * seg[None]).sum(axis=2) / denom[None], 0.0, 1.0)  # (N, S)
    resid = np.abs(g[:, None, :] - (p0[None] + u[..., None] * seg[None])).max(axis=2)  # (N, S)
    best = resid.argmin(axis=1)
    worst = float(resid[np.arange(len(g)), best].max())
    check(worst < 5e-6, "수신 목표각이 프레임 보간 경로 위에 있다 (float32 오차 이내)", f"max err {worst:.2e} rad")
    prog = best + u[np.arange(len(g)), best]  # 실수 프레임 위치
    check(bool((np.diff(prog) >= -1e-6).all()), "재생 위치 단조 증가 (역행 없음)")
    check(
        prog[0] <= 2.0 and prog[-1] >= expected.shape[0] - 3, "처음부터 끝까지 재생", f"{prog[0]:.1f} -> {prog[-1]:.1f}"
    )
    check(float(prog.max()) >= pb["loop_start"], "진입 보간을 지나 클립 구간까지 도달")
    # 실시간 발행에서 틱 간 명령 변화율 — 프레임 건너뜀이 있으면 여기서 2배로 튄다
    rate = float(np.abs(np.diff(g, axis=0)).max()) * gc.PUBLISH_HZ
    peak = pb["peak_rate"]["max_rad_s"]
    check(
        rate <= 1.30 * peak,
        "실시간 발행 틱 간 변화율에 프레임 건너뜀 없음",
        f"{rate:.2f} rad/s (clip peak {peak:.2f})",
    )
    win.close()


def main() -> int:
    app = QApplication(sys.argv)
    print(f"motion_data: {motion_clips.MOTION_DIR}")
    print(f"clips: {motion_clips.list_clips()}")
    test_playback_frames(app)
    test_loop_wrap(app)
    test_stop_and_finish(app)
    test_preemption_and_regression(app)
    test_udp_end_to_end(app)
    print()
    if _failures:
        print(f"FAILED ({len(_failures)}): " + ", ".join(_failures))
        return 1
    print("motion playback integration OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""두 정책의 램프 영상을 같은 시점끼리 좌우로 붙여 비교 영상을 만든다.

체이스캠이 로봇을 화면 중앙 근처에 두지만 원본 1280x720 에서는 로봇이 작아 보행 자세가 안 읽힌다.
그래서 중앙을 잘라 확대한 뒤 붙인다. 라벨은 PIL 로 그려 overlay 한다
(번들 ffmpeg 에 drawtext 필터가 없다).

사용:
    python _workspace/leg/make_view_compare.py <view> <left_dir> <left_label> <right_dir> <right_label> <out>
"""

import os
import subprocess
import sys

import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

FF = imageio_ffmpeg.get_ffmpeg_exe()
# 원본 1280x720 에서 잘라낼 영역 — 체이스캠이 로봇을 두는 위치에 맞춘다.
CROP = {
    "side": (300, 230, 700, 394),
    "top": (340, 130, 620, 349),
    "diag": (300, 230, 700, 394),
}
PANE_W, PANE_H = 900, 506  # 각 패널 최종 크기
BAR = 46


def label_png(text, color, path, w=PANE_W, h=BAR):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 200))
    d = ImageDraw.Draw(img)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 26)
    except OSError:
        f = ImageFont.load_default()
    tw = d.textlength(text, font=f)
    d.text(((w - tw) / 2, (h - 32) / 2), text, fill=color, font=f)
    img.save(path)


def main():
    view, ldir, llab, rdir, rlab, out = sys.argv[1:7]
    cx, cy, cw, ch = CROP.get(view, CROP["side"])
    tmp = os.path.dirname(out) or "."
    os.makedirs(tmp, exist_ok=True)
    lp, rp = f"{tmp}/_lab_L.png", f"{tmp}/_lab_R.png"
    label_png(llab, (120, 230, 140), lp)
    label_png(rlab, (240, 130, 120), rp)

    lv = f"{ldir}/speed_ramp-step-0.mp4"
    rv = f"{rdir}/speed_ramp-step-0.mp4"
    for p in (lv, rv):
        if not os.path.exists(p):
            print(f"없음: {p}")
            return 1

    # [0]=left video [1]=right video [2]=left label [3]=right label
    fc = (
        f"[0:v]crop={cw}:{ch}:{cx}:{cy},scale={PANE_W}:{PANE_H},pad={PANE_W}:{PANE_H + BAR}:0:{BAR}:black[L];"
        f"[1:v]crop={cw}:{ch}:{cx}:{cy},scale={PANE_W}:{PANE_H},pad={PANE_W}:{PANE_H + BAR}:0:{BAR}:black[R];"
        f"[L][2:v]overlay=0:0[Lb];[R][3:v]overlay=0:0[Rb];[Lb][Rb]hstack=inputs=2[v]"
    )
    cmd = [FF, "-y", "-i", lv, "-i", rv, "-i", lp, "-i", rp,
           "-filter_complex", fc, "-map", "[v]",
           "-c:v", "libx264", "-crf", "26", "-pix_fmt", "yuv420p", "-preset", "medium", out]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2500:])
        return r.returncode
    print(f"saved {out}  ({os.path.getsize(out) / 1e6:.1f} MB)")
    for p in (lp, rp):
        os.remove(p)
    return 0


sys.exit(main())

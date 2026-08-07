"""Stack each terrain's two renders side by side: GT voxel (left) | SL-Grid+crawl (right).

Both clips come from the same terrain and the same --video_length, so no time alignment
is needed. Output is scaled so the pair fits comfortably under Notion's 20 MiB single-part
upload limit.
"""

import pathlib
import subprocess
import sys

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = pathlib.Path("/home/lgb/IsaacLab-6.0/_workspace/gt_vs_slgrid_crawl/videos")
DST = pathlib.Path("/home/lgb/IsaacLab-6.0/reports/_comparisons/gt_voxel_vs_slgrid_crawl/videos")
TERRAINS = ["flat", "hurdle", "step", "gap", "stair", "crawl"]


def main() -> int:
    DST.mkdir(parents=True, exist_ok=True)
    failed = []
    for ter in TERRAINS:
        left, right = SRC / f"{ter}_gtvoxel.mp4", SRC / f"{ter}_slgrid.mp4"
        if not (left.exists() and right.exists()):
            print(f"[{ter}] SKIP — missing {'GT' if not left.exists() else 'SL'} clip")
            failed.append(ter)
            continue
        out = DST / f"{ter}_gtvoxel_vs_slgrid_crawl.mp4"
        cmd = [
            FF, "-y", "-i", str(left), "-i", str(right),
            "-filter_complex",
            # pad each to an even height, then place them side by side with a thin divider
            "[0:v]scale=-2:540,setsar=1[l];[1:v]scale=-2:540,setsar=1[r];"
            "[l][r]hstack=inputs=2[v]",
            "-map", "[v]", "-c:v", "libx264", "-crf", "26", "-preset", "medium",
            "-pix_fmt", "yuv420p", str(out),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            print(f"[{ter}] FAILED rc={r.returncode}\n{r.stderr[-600:]}")
            failed.append(ter)
        else:
            print(f"[{ter}] OK {out.stat().st_size / 1e6:.2f} MB -> {out.name}")
    if failed:
        print("failed/skipped:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

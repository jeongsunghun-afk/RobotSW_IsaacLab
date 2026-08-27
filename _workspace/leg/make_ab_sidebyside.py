"""priv vs history 경로 A/B 램프 두 클립을 좌우로 붙여 한 영상으로 만든다.

왼쪽 = history latent(배포 평가 경로), 오른쪽 = priv latent(학습 롤아웃 경로).
같은 체크포인트·같은 명령 프로파일이라 시간 정렬이 필요 없다.

라벨은 영어로 둔다 — 렌더 환경에 CJK 폰트가 없어 한글이 □ 로 깨진다.
Notion 단일 업로드 한계(20 MiB)에 맞춰 스케일·비트레이트를 낮춘다.

    python _workspace/leg/make_ab_sidebyside.py
"""

import pathlib
import subprocess
import sys

import imageio_ffmpeg

FF = imageio_ffmpeg.get_ffmpeg_exe()
SRC = pathlib.Path("/home/lgb/IsaacLab-6.0/_workspace/leg/ab_video")
DST = pathlib.Path("/home/lgb/IsaacLab-6.0/reports/leg_imitation/_comparisons/arch_scaling_study/videos")
OUT = DST / "priv_vs_history_cmd05.mp4"

# 좌우 라벨. drawtext 는 폰트 파일이 필요하므로 없으면 라벨 없이 붙인다.
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
# 라벨은 경로와 **이 롤아웃의 실제 결과**를 함께 적는다. cmd 0.5 는 이봉이라 경로가 결과를
# 결정하지 않으며(4 repeat 평균은 이 쌍과 반대 방향이다), 라벨에 경로만 쓰면 오해를 부른다.
LEFT_LABEL = "history latent  -  cmd 0.5: STANDS (duty 0.00)"
RIGHT_LABEL = "priv latent  -  cmd 0.5: WALKS (duty 0.80)"


W = 640  # 각 클립 가로 폭 [px]
BANNER_H = 34


def _banner(text: str, path: pathlib.Path) -> pathlib.Path:
    """라벨 배너 PNG 를 만든다.

    번들 imageio_ffmpeg 빌드에는 `drawtext` 가 없어(libfreetype 미포함) 자막을 못 굽는다.
    대신 PIL 로 배너를 그려 `vstack` 으로 영상 위에 얹는다 — core 필터만 쓴다.
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (W, BANNER_H), (16, 16, 16))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT, 17)
    except OSError:
        font = ImageFont.load_default()
    tw = draw.textbbox((0, 0), text, font=font)[2]
    draw.text(((W - tw) // 2, (BANNER_H - 21) // 2), text, fill=(255, 255, 255), font=font)
    img.save(path)
    return path


def _one(mode: str) -> pathlib.Path | None:
    hits = sorted((SRC / mode).glob("*.mp4"))
    return hits[0] if hits else None


def main() -> int:
    left, right = _one("hist"), _one("priv")
    if left is None or right is None:
        print(f"[error] 클립 없음: hist={left} priv={right}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)

    bl = _banner(LEFT_LABEL, DST / "_banner_l.png")
    br = _banner(RIGHT_LABEL, DST / "_banner_r.png")
    # ★ vstack 의 shortest=1 이 없으면 `-loop 1` 로 넣은 배너 PNG 가 끝나지 않아
    #   출력이 무한정 길어진다(실측: 3시간 48분짜리 58 MB 파일이 나왔다).
    lab = (
        f"[0:v]scale={W}:-2[v0];[1:v]scale={W}:-2[v1];"
        "[2:v][v0]vstack=inputs=2:shortest=1[l];[3:v][v1]vstack=inputs=2:shortest=1[r];"
        "[l][r]hstack=inputs=2[v]"
    )
    cmd = [FF, "-y", "-i", str(left), "-i", str(right),
           "-loop", "1", "-i", str(bl), "-loop", "1", "-i", str(br),
           "-filter_complex", lab, "-map", "[v]", "-shortest",
           "-c:v", "libx264", "-preset", "medium", "-crf", "26", "-pix_fmt", "yuv420p",
           str(OUT)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-2000:], file=sys.stderr)
        return r.returncode
    mib = OUT.stat().st_size / 1024**2
    print(f"saved: {OUT}  ({mib:.1f} MiB)")
    if mib > 19:
        print("[warn] 20 MiB 한계에 근접 — Notion 업로드 전 재압축 필요")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

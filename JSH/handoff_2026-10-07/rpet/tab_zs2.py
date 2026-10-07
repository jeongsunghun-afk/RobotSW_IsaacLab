import re, io, json
t = io.open("/mnt/ssd1/jsh/rpet_zs2.log", encoding="utf-8", errors="ignore").read().replace("\r", "")
rows, cur = [], None
for line in t.split("\n"):
    m = re.match(r"^CELL (\S+) (\S+) vx([\d.]+)\s*$", line)
    if m:
        cur = (m.group(1), m.group(2), m.group(3)); continue
    m = re.match(r"^RESULT_JSON (\{.*\})\s*$", line)
    if m and cur:
        rows.append((cur, json.loads(m.group(1)))); cur = None
print("R.pet 험지 zero-shot (교정판: yaw=0 고정 · 16 env · 400 step · 재학습 0)\n")
print("%-9s %-3s %8s %8s %10s %11s %10s" % ("지형", "L", "vx실측", "추종률", "이동거리m", "플랫폼이탈", "낙상env%"))
print("  " + "-" * 66)
for (k, l, v), d in rows:
    r = d["vx_gt"] / d["vx_cmd"]
    print("%-9s %-3s %8.3f %7.1f%% %10.2f %10.0f%% %8.0f%%  (term %d)" % (
        k, l, d["vx_gt"], 100 * r, d["far_mean"],
        100 * d["left_platform_frac"], 100 * d["died_env_frac"], d["terminations"]))
print("\n  이동거리 = env별 최대 변위 평균 [m] · 플랫폼이탈 = 변위>1.0m env 비율")
print("  낙상env% = 1회 이상 종료한 env 비율 · term = 누적 종료 횟수")
print("  (명령 vx 1.0, 400 step = 8 s → 무손실이면 약 8 m)")

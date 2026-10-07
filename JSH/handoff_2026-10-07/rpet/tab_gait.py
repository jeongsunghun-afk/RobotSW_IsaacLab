import re, io, json
t = io.open("/mnt/ssd1/jsh/rpet_gait_terrain.log", encoding="utf-8", errors="ignore").read().replace("\r", "")
rows, cur = [], ("flat", "-")
for line in t.split("\n"):
    m = re.match(r"^GCELL (\S+) L(\S+)\s*$", line)
    if m:
        cur = (m.group(1), m.group(2)); continue
    m = re.match(r"^GAIT_JSON (\{.*\})\s*$", line)
    if m:
        rows.append((cur, json.loads(m.group(1))))
print("R.pet 험지 걸음 분류 (vx 1.0 · 64 env · 500 step · 마지막 2 s 창)")
print("★분류기는 저속에서만 검증됨 — 평지 pace 73.4% vs 문서 70~71%. 고속(vx3.0)은 검증 실패.\n")
print("%-9s %-3s %7s %7s %12s %7s %8s %8s" % ("지형","L","pace","trot","bound/gallop","stand","적합도","낙상env%"))
print("  " + "-" * 68)
for (k, l), d in rows:
    g = d["gait"]
    print("%-9s %-3s %6.1f%% %6.1f%% %11.1f%% %6.1f%% %8.3f %7.0f%%" % (
        k, l, g.get("pace", 0), g.get("trot", 0), g.get("bound/gallop", 0), g.get("stand", 0),
        d["pair_score_med"], 100 * d["died_env_frac"]))
print("\n  적합도 = 쌍구조 점수 중앙값 (작을수록 그 걸음에 잘 맞음. 평지 0.246 이 기준)")

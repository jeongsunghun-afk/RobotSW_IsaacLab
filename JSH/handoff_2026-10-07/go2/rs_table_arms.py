#!/usr/bin/env python3
"""[ROBUST-SUITE] tier-1 outcome + tier-2 traverse quality + metric-redundancy report."""
import numpy as np, os, re, glob, json, sys
sys.path.insert(0, "/tmp")
import rs_metrics as RM

SEEDS = [int(x) for x in os.environ.get("RS_SEEDS", "42,7,1,3,2,5,11,23").split(",")]
# ↑ 시드 집합을 RS_SEEDS 로 인자화.  고정 [42,7,1,3,2,5,11,23] 는 d-계열 정책의 규약이고,
#   팔 비교(armG/armT/armTS)는 1~8 로 돌렸다.  교집합만 세면 n=5 로 줄어 8-seed 규약이 깨진다.
DUMPDIR = os.environ.get("RS_DUMPDIR", "/tmp/legsym")
RM.DUMPDIR = DUMPDIR
RM.DRAWS = RM.load_draws(os.path.join(DUMPDIR, "rand_d9.log"))
ONLY = sys.argv[1] if len(sys.argv) > 1 else ""

rows = []
for fd in sorted(glob.glob(os.path.join(DUMPDIR, "*_footdiag.txt"))):
    tag = os.path.basename(fd)[:-13]
    m = re.match(r"(d\d+|arm[A-Za-z0-9]+)_L(\d+)_s(\d+)(?:_vx([\d.]+))?(?:_ep(\d+))?$", tag)
    if not m or (ONLY and not re.search(ONLY, tag)): continue
    pol, L, s = m.group(1), int(m.group(2)), int(m.group(3))
    if s not in SEEDS: continue
    vx = float(m.group(4)) if m.group(4) else 0.4
    ep = int(m.group(5)) if m.group(5) else 30
    try: r = RM.analyse(tag, L, s, DUMPDIR)
    except Exception as e: print("ERR", tag, e, file=sys.stderr); continue
    if r is None: continue
    r.update(pol=pol, vx=vx, ep=ep); rows.append(r)
json.dump(rows, open("/tmp/rs_rows.json", "w"))
print(f"# {len(rows)} rollouts from {DUMPDIR}\n")
keys = sorted({(r["pol"], r["level"], r["vx"], r["ep"]) for r in rows})
def sel(k): return [r for r in rows if (r["pol"], r["level"], r["vx"], r["ep"]) == k]

print("=" * 116)
print("TIER 1 -- OUTCOME.  R1 CROSS / R2 FALLS / R3 STALL.  All binary + geometry-relative: no distance-per-time.")
print("=" * 116)
print(f"{'pol':4}{'L':>3}{'vx':>6}{'ep':>5}{'n':>3} | {'R1 CROSS':>9} {'R2 FALLn':>9} {'TRUNC':>5} {'R3 STALL':>6} {'ENTER':>7} "
      f"{'maxx_med':>9} {'R4 ceil':>8}  missing")
for k in keys:
    S = sel(k); n = len(S)
    oc = {c: sum(r["outcome"] == c for r in S) for c in ("CROSS", "FALL", "TRUNC", "STALL")}
    # R2 = NUMBER OF SEEDS whose FIRST episode ended in a base-contact death.  NOT the total reset
    # count: after a death the env restarts and keeps running to the episode budget, so a raw reset
    # tally scales with EPLEN (a 120 s run gets 4x the exposure of a 30 s run) and with the dead
    # time a finished policy spends PARKED at the corridor end.  One seed = one episode = one
    # Bernoulli trial keeps R2 exposure-normalised and consistent with CROSS/STALL/TRUNC.
    cr, fa = sum(r["crossed"] for r in S), oc["FALL"]
    ev = sum(r["falls"] for r in S)   # raw reset events, exposure-dependent -- diagnostic only
    en = sum(r["entered"] for r in S)
    ok = "PASS" if (n >= 8 and cr >= 7 and fa == 0) else ""
    miss = sorted(set(SEEDS) - {r["seed"] for r in S})
    print(f"{k[0]:4}{k[1]:3d}{k[2]:6.2f}{k[3]:5d}{n:3d} | {cr:4d}/{n:<4d} {fa:8d} {oc['TRUNC']:5d} {oc['STALL']:6d} "
          f"{en:4d}/{n:<2d} {np.median([r['max_x'] for r in S]):9.2f} {ok:>8}  "
          f"{'(reset-events %d)' % ev if ev != fa else ''}{miss if miss else ''}")

for WIN in ("cruise", "strict"):
    print("\n" + "=" * 116)
    print(f"TIER 2 -- TRAVERSE QUALITY, window = {WIN}."
          + ("  adaptive: first stone -> min(3.0, maxx), >=0.8 m of real progress required."
             if WIN == "cruise" else "  fixed x 1.0 -> 3.0 (cross-check)."))
    print("         Stalled / parked rollouts are N/A here -- standing still can never score a clean gait.")
    print("=" * 116)
    print(f"{'pol':4}{'L':>3}{'vx':>6}{'ep':>5}{'nval':>5} | {'R5 voidTD%':>11} {'voidDW%':>8} {'R6 recov%':>10} "
          f"{'R7 ydev':>8} {'R8 bz':>7} {'n_sup':>6} {'flight%':>8} | {'dist':>5} {'vx_cr':>6}")
    for k in keys:
        S = [r for r in sel(k) if r.get(WIN, {}).get("valid")]
        if not S:
            print(f"{k[0]:4}{k[1]:3d}{k[2]:6.2f}{k[3]:5d}{0:5d} |  -- no seed traversed the window --"); continue
        g = lambda f: np.array([r[WIN][f] for r in S if r[WIN].get(f) is not None], float)
        rc = g("void_recov")
        print(f"{k[0]:4}{k[1]:3d}{k[2]:6.2f}{k[3]:5d}{len(S):5d} | {g('void_td').mean()*100:11.2f} "
              f"{g('void_dwell').mean()*100:8.2f} {(rc.mean()*100 if len(rc) else float('nan')):10.1f} "
              f"{g('ydev').mean():8.3f} {g('bz_med').mean():7.3f} {g('n_sup').mean():6.2f} "
              f"{g('flight').mean()*100:8.2f} | {g('dist').mean():5.2f} {g('vx').mean():6.3f}")

print("\n" + "=" * 116)
print("METRIC REDUNDANCY (Pearson r, every cruise-valid rollout).  |r|>0.9 => keep only one.")
print("=" * 116)
V = [r for r in rows if r.get("cruise", {}).get("valid")]
names = ["void_td", "void_dwell", "ydev", "bz_med", "n_sup", "flight", "marg_com", "marg_link", "vx"]
Mx = np.array([[r["cruise"][n] for n in names] for r in V], float)
print("            " + "".join(f"{n[:9]:>10}" for n in names))
for i, n in enumerate(names):
    print(f"{n:>11} " + "".join(f"{np.corrcoef(Mx[:, i], Mx[:, j])[0,1]:10.2f}" for j in range(len(names))))
print(f"\n[n = {len(V)} cruise-valid rollouts]")

# ---------------------------------------------------------------- R4 robust ceiling
print("\n" + "=" * 116)
print("R4 -- ROBUST CEILING.  Highest level a policy clears on the 8-seed DR draw.")
print("     strict : CROSS >= 7/8 AND falls == 0   (the rule as specified)")
print("     <=1fall: CROSS >= 7/8 AND falls <= 1   (same rule, one seed of slack -- see note)")
print("=" * 116)
POLS = sorted({r["pol"] for r in rows})
for p_ in POLS:
    per = {}
    for k in keys:
        if k[0] != p_: continue
        S = sel(k)
        if len(S) < 8: continue
        cr = sum(r["crossed"] for r in S); fa = sum(r["outcome"] == "FALL" for r in S)
        # keep the most favourable (vx, ep) cell per level -- a policy is judged at its best setting
        cur = per.get(k[1])
        if cur is None or (cr, -fa) > (cur[0], -cur[1]): per[k[1]] = (cr, fa, k[2], k[3])
    if not per: print(f"  {p_:5} no 8-seed cell"); continue
    strict = [L for L, (cr, fa, _, _) in per.items() if cr >= 7 and fa == 0]
    loose  = [L for L, (cr, fa, _, _) in per.items() if cr >= 7 and fa <= 1]
    detail = "  ".join(f"L{L}:{cr}/8,f{fa}@vx{vx}/ep{ep}" for L, (cr, fa, vx, ep) in sorted(per.items()))
    print(f"  {p_:5} strict={('L%d' % max(strict)) if strict else 'none':>5}   "
          f"<=1fall={('L%d' % max(loose)) if loose else 'none':>5}   |  {detail}")
print("\n  Each level is scored at the policy's BEST (vx,ep) cell -- the cell is printed, because")
print("  it matters: D6 clears L7 with 0 falls only in its vx=0.20 cell (vx=0.40 costs it 1 fall).")
print("  NOTE: the strict rule is brittle at n=8 -- D6 fails L7 (1 fall, seed 2) yet passes L8")
print("        (8/8, 0 falls), i.e. it is NON-MONOTONE in level.  Report the whole level curve;")
print("        if a single scalar is required, the <=1-fall variant is monotone here.")

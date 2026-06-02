import numpy as np

d = np.load('/home/lgb/IsaacLab/_workspace/3leg_gait_research/rank0_verify_result.npz')
sensor = d['arr_sensor'].astype(bool)   # (2000,64,4) contact True (force>2N)
cls = d['arr_class']                     # (2000,64)
foot_names = [str(x) for x in d['foot_names']]
STEP_DT = 4 / 200.0  # 0.02 s
T, E, F = sensor.shape
print("shape", sensor.shape, "step_dt", STEP_DT)
print("foot contact rate:", dict(zip(foot_names, sensor.reshape(-1, 4).mean(0).round(4))))


def runs_of_false(col):
    out = []
    n = len(col)
    i = 0
    while i < n:
        if not col[i]:
            j = i
            while j < n and not col[j]:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


air_by_foot = {f: [] for f in foot_names}
air_by_foot_terr = {f: {c: [] for c in range(5)} for f in foot_names}

for e in range(E):
    for fi in range(F):
        col = sensor[:, e, fi]
        for (st, ln) in runs_of_false(col):
            air_by_foot[foot_names[fi]].append(ln)
            c0 = int(cls[st, e])
            air_by_foot_terr[foot_names[fi]][c0].append(ln)

terr_names = {0: 'flat', 1: 'hurdle', 2: 'step', 3: 'gap', 4: 'stair'}


def stats(arr_steps):
    a = np.array(arr_steps, dtype=float) * STEP_DT
    if len(a) == 0:
        return None
    return dict(n=len(a), median=np.median(a), p90=np.percentile(a, 90),
                p95=np.percentile(a, 95), p99=np.percentile(a, 99),
                max=a.max(), mean=a.mean())


print("\n=== PER-FOOT AIR-TIME (seconds), all terrain ===")
print(f"{'foot':8} {'n':>7} {'median':>8} {'p90':>8} {'p95':>8} {'p99':>8} {'max':>8}")
for f in foot_names:
    s = stats(air_by_foot[f])
    print(f"{f:8} {s['n']:>7} {s['median']:>8.3f} {s['p90']:>8.3f} {s['p95']:>8.3f} {s['p99']:>8.3f} {s['max']:>8.3f}")

print("\n=== PER-FOOT x TERRAIN p95/p99/max (seconds) ===")
for f in foot_names:
    print(f"\n-- {f} --")
    print(f"{'terrain':8} {'n':>7} {'p95':>8} {'p99':>8} {'max':>8}")
    for c in range(5):
        s = stats(air_by_foot_terr[f][c])
        if s is None:
            print(f"{terr_names[c]:8} {'0':>7}")
            continue
        print(f"{terr_names[c]:8} {s['n']:>7} {s['p95']:>8.3f} {s['p99']:>8.3f} {s['max']:>8.3f}")

# Normal feet aggregate (FL, FR, RR) -- exclude RL (dropped)
normal = ['FL_foot', 'FR_foot', 'RR_foot']
print("\n=== NORMAL FEET (FL/FR/RR) AGGREGATE ===")
agg_all = []
for f in normal:
    agg_all += air_by_foot[f]
s = stats(agg_all)
print(f"ALL terrain: n={s['n']} median={s['median']:.3f} p90={s['p90']:.3f} p95={s['p95']:.3f} p99={s['p99']:.3f} max={s['max']:.3f}")
print("\nNormal-feet aggregate per terrain:")
print(f"{'terrain':8} {'n':>7} {'p95':>8} {'p99':>8} {'max':>8}")
for c in range(5):
    agg = []
    for f in normal:
        agg += air_by_foot_terr[f][c]
    s = stats(agg)
    print(f"{terr_names[c]:8} {s['n']:>7} {s['p95']:>8.3f} {s['p99']:>8.3f} {s['max']:>8.3f}")

# RL (dropped foot) for reference
print("\n=== RL_foot (DROPPED) for contrast ===")
s = stats(air_by_foot['RL_foot'])
print(f"ALL: n={s['n']} median={s['median']:.3f} p90={s['p90']:.3f} p95={s['p95']:.3f} p99={s['p99']:.3f} max={s['max']:.3f} mean={s['mean']:.3f}")
# RL air-time histogram in seconds
rl = np.array(air_by_foot['RL_foot'], dtype=float) * STEP_DT
print("RL air segs >= 1.0s:", (rl >= 1.0).sum(), "/", len(rl), f"({(rl>=1.0).mean()*100:.1f}%)")
print("RL air segs >= 2.0s:", (rl >= 2.0).sum(), f"({(rl>=2.0).mean()*100:.1f}%)")
print("RL air segs >= 5.0s:", (rl >= 5.0).sum(), f"({(rl>=5.0).mean()*100:.1f}%)")

# Normal feet: fraction of air segments above candidate caps
print("\n=== Normal-feet air segments above candidate caps ===")
agg = np.array(agg_all, dtype=float) * STEP_DT
for cap in [0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
    print(f"cap {cap:.2f}s: normal segs above = {(agg>cap).sum()} ({(agg>cap).mean()*100:.3f}%); RL segs above = {(rl>cap).sum()} ({(rl>cap).mean()*100:.1f}%)")

# ---- tail / outlier inspection for normal feet ----
print("\n=== NORMAL-FEET TAIL & OUTLIERS ===")
segs = []
idx = {f: i for i, f in enumerate(foot_names)}
for e in range(E):
    for f in normal:
        for st, ln in runs_of_false(sensor[:, e, idx[f]]):
            segs.append((ln * STEP_DT, f, e, st, int(cls[st, e])))
segs.sort(reverse=True)
print("Top 15 normal-foot air segments (sec, foot, env, start_step, terrain):")
for s in segs[:15]:
    print(f"  {s[0]:.2f}s {s[1]} env{s[2]} step{s[3]} {terr_names[s[4]]}")
arr = np.array([s[0] for s in segs])
print("\nNormal-feet fine tail percentiles:")
for p in [99, 99.5, 99.9, 99.95, 99.99]:
    print(f"  p{p}: {np.percentile(arr, p):.3f}s")
print(f"  max: {arr.max():.3f}s ; segs>0.5s: {(arr>0.5).sum()} of {len(arr)}")

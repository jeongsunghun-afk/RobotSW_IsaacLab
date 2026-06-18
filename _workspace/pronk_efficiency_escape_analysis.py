"""Post-hoc: does positive_work (Σ max(0,τ·q̇)) penalty make flight a low-power ESCAPE -> pronk?

numpy-only, no Isaac. Uses pronk_cost_result.npz (model_16700, 256 envs x 3000 steps).
Tests hypothesis: policy can lower positive_work penalty by spending more time airborne.

Key arrays:
  arr_mech_pow (T,N)   = Σ max(0, τ·q̇) over 12 joints [W]   == positive_work quantity
  arr_airborne (T,N)   = all-4-feet-off (pronk flight)        uint8
  arr_contact  (T,N,4) = per-foot contact (force>thresh)      uint8
  arr_class    (T,N)   = terrain class 0..4
  valid        (T,N)   = non-failed-episode mask              uint8
"""
import numpy as np

NPZ = "/home/lgb/IsaacLab/_workspace/pronk_cost_result.npz"
NAMES = {0: "flat", 1: "hurdle", 2: "step", 3: "gap", 4: "stair"}

d = np.load(NPZ, allow_pickle=True)
print("=== (0) npz keys / shapes ===")
for k in d.files:
    a = d[k]
    print(f"  {k:<14} shape={str(a.shape):<16} dtype={a.dtype}")

pw   = d["arr_mech_pow"].astype(np.float64)      # (T,N) W
air  = d["arr_airborne"].astype(bool)            # (T,N)
con  = d["arr_contact"].astype(bool)             # (T,N,4)
cls  = d["arr_class"].astype(np.int64)           # (T,N)
valid= d["valid"].astype(bool)                   # (T,N)
T, N = pw.shape
nfeet = con.sum(axis=-1)                          # (T,N) #feet down 0..4
class_ids = sorted({int(c) for c in np.unique(cls)})

def stats(x):
    if x.size == 0:
        return (np.nan, np.nan, np.nan, 0)
    return (float(np.mean(x)), float(np.median(x)), float(np.percentile(x, 90)), int(x.size))

print(f"\nT={T} N={N}  valid_frac={valid.mean():.3f}")

# ============================================================
# (1) sanity: airborne_frac & mean power per terrain (CONFOUNDED cross-terrain view)
# ============================================================
print("\n=== (1) per-terrain airborne_frac & mean power (cross-terrain = CONFOUNDED) ===")
print(f"  {'terrain':<8}{'n_valid':>10}{'air_frac':>10}{'meanP_all':>12}")
for c in class_ids:
    m = (cls == c) & valid
    if m.sum() == 0:
        continue
    print(f"  {NAMES.get(c,c):<8}{int(m.sum()):>10}{air[m].mean():>10.4f}{pw[m].mean():>12.2f}")

# ============================================================
# (2) CORE: airborne vs stance per-step power, PER TERRAIN (within-terrain control)
# ============================================================
print("\n=== (2) airborne vs stance per-step power [W], PER TERRAIN (within-terrain) ===")
print(f"  {'terrain':<8}{'grp':<9}{'mean':>9}{'median':>9}{'p90':>9}{'n':>10}")
core = {}
for c in class_ids:
    base = (cls == c) & valid
    if base.sum() == 0:
        continue
    a_m = base & air
    s_m = base & ~air
    pa, pa_med, pa_90, na = stats(pw[a_m])
    ps, ps_med, ps_90, ns = stats(pw[s_m])
    core[c] = (pa, ps, na, ns)
    print(f"  {NAMES.get(c,c):<8}{'airborne':<9}{pa:>9.2f}{pa_med:>9.2f}{pa_90:>9.2f}{na:>10}")
    print(f"  {NAMES.get(c,c):<8}{'stance':<9}{ps:>9.2f}{ps_med:>9.2f}{ps_90:>9.2f}{ns:>10}")
    if not np.isnan(pa) and ps > 0:
        print(f"  {'':8}{'-> airborne/stance mean ratio = '}{pa/ps:.3f}")

# ============================================================
# (3) PHASE decomposition: takeoff / flight / landing / stance  (advisor: the real test)
#   takeoff  = contact[t-1] (>=1 foot) -> airborne[t]
#   flight   = airborne[t-1] & airborne[t]  (sustained ballistic)
#   landing  = airborne[t-1] & contact[t] (>=1 foot)
#   stance   = contact[t-1] & contact[t]   (no airborne adjacent)
# ============================================================
print("\n=== (3) PHASE decomposition of per-step power [W], PER TERRAIN ===")
print("  phases: takeoff(push-off) / flight(ballistic) / landing(impact) / stance(ground)")
prev_air = np.zeros_like(air); prev_air[1:] = air[:-1]
prev_con = np.zeros_like(air); prev_con[1:] = con.any(axis=-1)[:-1]
cur_con  = con.any(axis=-1)
takeoff = prev_con & air & valid
flight  = prev_air & air & valid
landing = prev_air & cur_con & valid
stance  = prev_con & cur_con & valid
# exclude step 0 (no prev)
for msk in (takeoff, flight, landing, stance):
    msk[0, :] = False
print(f"  {'terrain':<8}{'takeoff':>10}{'flight':>10}{'landing':>10}{'stance':>10}")
print(f"  {'(mean W over steps in phase)':<48}")
for c in class_ids:
    cm = (cls == c)
    row = []
    for msk in (takeoff, flight, landing, stance):
        mm = msk & cm
        row.append(pw[mm].mean() if mm.sum() else np.nan)
    print(f"  {NAMES.get(c,c):<8}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>10.2f}{row[3]:>10.2f}")
print(f"  {'(n steps in phase)':<48}")
for c in class_ids:
    cm = (cls == c)
    row = [int((msk & cm).sum()) for msk in (takeoff, flight, landing, stance)]
    print(f"  {NAMES.get(c,c):<8}{row[0]:>10}{row[1]:>10}{row[2]:>10}{row[3]:>10}")

# ============================================================
# (4) #feet-in-contact vs power (monotonic? fewer feet -> lower power?), per terrain
# ============================================================
print("\n=== (4) per-step power [W] by #feet in contact (0..4), PER TERRAIN ===")
print(f"  {'terrain':<8}{'0ft':>9}{'1ft':>9}{'2ft':>9}{'3ft':>9}{'4ft':>9}")
print(f"  {'(mean W)':<8}")
for c in class_ids:
    base = (cls == c) & valid
    if base.sum() == 0:
        continue
    row = []
    for k in range(5):
        mm = base & (nfeet == k)
        row.append(pw[mm].mean() if mm.sum() else np.nan)
    print(f"  {NAMES.get(c,c):<8}" + "".join(f"{v:>9.2f}" if not np.isnan(v) else f"{'--':>9}" for v in row))
print(f"  {'(n steps)':<8}")
for c in class_ids:
    base = (cls == c) & valid
    if base.sum() == 0:
        continue
    row = [int((base & (nfeet == k)).sum()) for k in range(5)]
    print(f"  {NAMES.get(c,c):<8}" + "".join(f"{v:>9}" for v in row))

# ============================================================
# (5) ESCAPE DISCRIMINATOR (advisor): within STAIR, per-env mean power vs per-env airborne frac.
#     Negative slope -> going airborne lowers positive_work -> supports HYP.
#     Repeat for each terrain. Use envs with enough valid samples on that terrain.
# ============================================================
print("\n=== (5) ESCAPE SLOPE: per-env meanP vs per-env airborne_frac, within terrain ===")
print("  (negative slope => more airborne -> lower positive_work = escape mechanism supports HYP)")
print(f"  {'terrain':<8}{'n_env':>7}{'slope_W/frac':>14}{'pearson_r':>11}{'meanP_lo':>10}{'meanP_hi':>10}")
MIN_SAMP = 50
for c in class_ids:
    af_list, mp_list = [], []
    for n in range(N):
        col = (cls[:, n] == c) & valid[:, n]
        ns = int(col.sum())
        if ns < MIN_SAMP:
            continue
        af_list.append(air[col, n].mean())
        mp_list.append(pw[col, n].mean())
    if len(af_list) < 5:
        print(f"  {NAMES.get(c,c):<8}{len(af_list):>7}  (too few envs)")
        continue
    af = np.array(af_list); mp = np.array(mp_list)
    # linear fit meanP ~ slope*air_frac + b
    A = np.vstack([af, np.ones_like(af)]).T
    slope, b = np.linalg.lstsq(A, mp, rcond=None)[0]
    r = np.corrcoef(af, mp)[0, 1] if af.std() > 0 else np.nan
    # tertile contrast: low vs high airborne-frac envs
    order = np.argsort(af)
    nlo = max(1, len(af) // 3)
    lo = mp[order[:nlo]].mean(); hi = mp[order[-nlo:]].mean()
    print(f"  {NAMES.get(c,c):<8}{len(af):>7}{slope:>14.2f}{r:>11.3f}{lo:>10.2f}{hi:>10.2f}")

print("\n[done]")

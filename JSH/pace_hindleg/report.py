import sys, glob, os, torch
from tensorboard.backend.event_processing import event_accumulator as EA

robot = sys.argv[1] if len(sys.argv) > 1 else "hindleg_sim"
base = "/mnt/ssd1/jsh/pace-sim2real/logs/pace/" + robot
d = sorted(glob.glob(base + "/*/"))[-1]
means = sorted(glob.glob(d + "mean_*.pt"))
ev = sorted(glob.glob(d + "events*"))[-1]
ea = EA.EventAccumulator(ev, size_guidance={"scalars": 0})
ea.Reload()
tags = ea.Tags()["scalars"]


def last(tag):
    s = ea.Scalars(tag)
    return s[-1].step, s[-1].value


it, score = last("0_Episode/score")
ck = [os.path.basename(m) for m in means]
print("robot=%s dir=%s iters=%d best_score=%.5f checkpoints=%s" % (robot, os.path.basename(d.rstrip("/")), it, score, ck))


def joints(prefix):
    ts = sorted([t for t in tags if t.startswith(prefix)])
    return [(t.split("/")[-1].replace("best_", ""), ea.Scalars(t)[-1].value) for t in ts]


for label, pref in [("armature", "1_Armature/best_"), ("viscous", "2_Viscous_Friction/best_"),
                    ("coulomb", "3_Static_Dynamic_Friction/best_"), ("bias", "4_Bias/best_")]:
    js = joints(pref)
    print("  %-9s:" % label, ["%s=%.4f" % (n, v) for n, v in js])
if "0_Delay/best" in tags:
    print("  delay    :", ea.Scalars("0_Delay/best")[-1].value)
if means:
    m = torch.load(means[-1])
    print("  MEAN %s (len %d):" % (os.path.basename(means[-1]), len(m)), [round(float(x), 4) for x in m.tolist()])

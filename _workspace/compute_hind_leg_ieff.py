"""Compute per-joint effective inertia I_eff for hind_leg analytically from USD.

Launches a minimal headless kit app ONLY to get `pxr` (USD libs). Does NOT build a
physics scene or step — just opens the USD stage, reads mass/inertia/joint data, and
computes the locked-base distal-subtree inertia about each joint axis (= mass-matrix
diagonal H[j,j]) at the USD rest pose.

Run (pin to freest GPU):
    CUDA_VISIBLE_DEVICES=2 ./isaaclab.sh -p _workspace/compute_hind_leg_ieff.py --headless
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app
print(">>> APP LAUNCHED", flush=True)

import numpy as np  # noqa: E402
from pxr import Usd, UsdGeom, UsdPhysics, Gf  # noqa: E402

print(">>> pxr imported", flush=True)

USD = "/home/lgb/IsaacLab/source/isaaclab_assets/data/Robots/Hind_Leg/hind_leg.usd"
stage = Usd.Stage.Open(USD)
if stage is None:
    print("FAILED to open", USD, flush=True); simulation_app.close(); sys.exit(1)
print(">>> STAGE OPENED:", USD, flush=True)


def quat_to_R(q):
    """pxr Gf.Quat* -> 3x3 numpy rotation."""
    if q is None:
        return np.eye(3)
    w = q.GetReal()
    im = q.GetImaginary()
    x, y, z = im[0], im[1], im[2]
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def m44_to_Rt(m):
    """Gf.Matrix4d -> (R 3x3, t 3) numpy. USD row-vector convention: world = local * M."""
    R = np.array([[m[i][j] for j in range(3)] for i in range(3)])  # rows = basis
    t = np.array([m[3][0], m[3][1], m[3][2]])
    # Row-vector: a point p_world = p_local . M_upper3x3 + t  => effective rotation is R^T for col vectors
    return R, t


def xform_point(R, t, p):
    """Apply USD row-vector transform to a local point p (3,) -> world."""
    p = np.asarray(p, float)
    return p @ R + t


def xform_vec(R, v):
    v = np.asarray(v, float)
    return v @ R


# ── Gather rigid bodies ──────────────────────────────────────────────────────
bodies = {}   # path -> dict(mass, diag, com_local, princ_R, Rworld, tworld)
for prim in stage.Traverse():
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        continue
    path = str(prim.GetPath())
    mass = None; diag = None; com = Gf.Vec3f(0, 0, 0); princ = None
    if prim.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(prim)
        mass = m.GetMassAttr().Get()
        diag = m.GetDiagonalInertiaAttr().Get()
        com = m.GetCenterOfMassAttr().Get() or Gf.Vec3f(0, 0, 0)
        princ = m.GetPrincipalAxesAttr().Get()
    xf = UsdGeom.Xformable(prim)
    world = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    R, t = m44_to_Rt(world)
    bodies[path] = dict(
        mass=float(mass) if mass else 0.0,
        diag=np.array([diag[0], diag[1], diag[2]]) if diag else np.zeros(3),
        com_local=np.array([com[0], com[1], com[2]]),
        princ_R=quat_to_R(princ),
        Rworld=R, tworld=t,
    )

print(f">>> {len(bodies)} rigid bodies", flush=True)
for p, b in bodies.items():
    print(f"    {p:<50} m={b['mass']:.4f}  diagI={b['diag']}  com={b['com_local']}", flush=True)

# ── Gather joints, build parent->child tree ──────────────────────────────────
AXIS = {"X": np.array([1., 0, 0]), "Y": np.array([0, 1., 0]), "Z": np.array([0, 0, 1.])}
joints = {}   # name -> dict(parent, child, axis_world, pos_world, axis_letter)
children = {}  # parent_path -> [child_path]
for prim in stage.Traverse():
    if "Joint" not in str(prim.GetTypeName()):
        continue
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        # still record tree edges for fixed joints
        pass
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    if not b0 or not b1:
        continue
    parent = str(b0[0]); child = str(b1[0])
    children.setdefault(parent, []).append(child)
    name = prim.GetName()
    if prim.IsA(UsdPhysics.RevoluteJoint):
        rj = UsdPhysics.RevoluteJoint(prim)
        axis_letter = rj.GetAxisAttr().Get()
        lp0 = j.GetLocalPos0Attr().Get() or Gf.Vec3f(0, 0, 0)
        # joint world position via parent body transform
        pb = bodies.get(parent)
        if pb is None:
            continue
        pos_world = xform_point(pb["Rworld"], pb["tworld"], [lp0[0], lp0[1], lp0[2]])
        axis_local = AXIS.get(str(axis_letter), np.array([1., 0, 0]))
        axis_world = xform_vec(pb["Rworld"], axis_local)
        axis_world = axis_world / (np.linalg.norm(axis_world) + 1e-12)
        joints[name] = dict(parent=parent, child=child,
                            axis_world=axis_world, pos_world=pos_world,
                            axis_letter=str(axis_letter), lower=rj.GetLowerLimitAttr().Get(),
                            upper=rj.GetUpperLimitAttr().Get())

print(f"\n>>> {len(joints)} revolute joints", flush=True)
for n, jj in joints.items():
    print(f"    {n:<22} parent={jj['parent'].split('/')[-1]:<14} child={jj['child'].split('/')[-1]:<14} "
          f"axis={jj['axis_letter']} axis_w={np.round(jj['axis_world'],3)} pos_w={np.round(jj['pos_world'],4)}",
          flush=True)


# ── Distal subtree collection ────────────────────────────────────────────────
def subtree(root):
    out = []
    stack = [root]
    while stack:
        node = stack.pop()
        out.append(node)
        for c in children.get(node, []):
            stack.append(c)
    return out


def inertia_about_axis(body, axis_w, pos_w):
    """Moment of inertia of one body about world axis (dir axis_w through pos_w)."""
    b = bodies[body]
    m = b["mass"]
    if m <= 0:
        return 0.0
    # CoM world
    com_w = xform_point(b["Rworld"], b["tworld"], b["com_local"])
    # I_cm in world frame: R_world^T? USD row-vec: world basis rows are R. For col-vec math use Rc = R^T
    Rc = b["Rworld"].T               # column-vector rotation body->world
    Rp = b["princ_R"]                # principal axes (col-vec) within body frame
    I_diag = np.diag(b["diag"])
    I_cm_body = Rp @ I_diag @ Rp.T
    I_cm_world = Rc @ I_cm_body @ Rc.T
    a = axis_w / (np.linalg.norm(axis_w) + 1e-12)
    I_axis_cm = a @ I_cm_world @ a
    d = com_w - pos_w
    d_perp2 = d @ d - (d @ a) ** 2
    return I_axis_cm + m * d_perp2


print(f"\n{'='*78}", flush=True)
print("  I_eff per joint (locked distal-subtree inertia about joint axis), USD rest pose", flush=True)
print(f"{'='*78}", flush=True)
print(f"  {'joint':<22} {'I_eff (kg·m²)':>14}  {'#distal links':>13}  {'distal':<30}", flush=True)
results = {}
for n, jj in joints.items():
    distal = subtree(jj["child"])
    I_eff = sum(inertia_about_axis(bp, jj["axis_world"], jj["pos_world"])
                for bp in distal if bp in bodies)
    results[n] = I_eff
    dl = ",".join(d.split("/")[-1] for d in distal)
    print(f"  {n:<22} {I_eff:>14.6f}  {len(distal):>13}  {dl:<30}", flush=True)

print(f"\n  total robot mass = {sum(b['mass'] for b in bodies.values()):.4f} kg", flush=True)
print("\n>>> DONE", flush=True)
simulation_app.close()

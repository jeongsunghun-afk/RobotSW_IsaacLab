"""USD structure dump for hind_leg — no kit app, pure pxr. Fast, no GPU.

Reads per-link mass + inertia + CoM and joint (axis, frames, body refs) so we can
compute effective inertia I_eff about each joint axis analytically.
"""
import sys

from pxr import Usd, UsdGeom, UsdPhysics, Gf  # noqa

USD = "/home/lgb/IsaacLab/source/isaaclab_assets/data/Robots/Hind_Leg/hind_leg.usd"

stage = Usd.Stage.Open(USD)
if stage is None:
    print("FAILED to open", USD); sys.exit(1)

print("=" * 80)
print("RIGID BODIES (mass / inertia / CoM)")
print("=" * 80)
print(f"{'prim path':<55} {'mass':>8} {'diagInertia (Ixx,Iyy,Izz)':>32}")
bodies = {}
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.MassAPI):
        m = UsdPhysics.MassAPI(prim)
        mass = m.GetMassAttr().Get()
        diag = m.GetDiagonalInertiaAttr().Get()
        com = m.GetCenterOfMassAttr().Get()
        princ = m.GetPrincipalAxesAttr().Get()
        bodies[str(prim.GetPath())] = dict(mass=mass, diag=diag, com=com, princ=princ)
        ds = f"({diag[0]:.5f},{diag[1]:.5f},{diag[2]:.5f})" if diag else "None"
        print(f"{str(prim.GetPath()):<55} {str(mass):>8} {ds:>32}")
        print(f"{'   CoM=':<10}{com}   principalAxes={princ}")

print()
print("=" * 80)
print("RIGID BODY API prims (xform / world transform)")
print("=" * 80)
for prim in stage.Traverse():
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        xf = UsdGeom.Xformable(prim)
        try:
            world = xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = world.ExtractTranslation()
            print(f"{str(prim.GetPath()):<55} worldT=({t[0]:.4f},{t[1]:.4f},{t[2]:.4f})")
        except Exception as e:
            print(f"{str(prim.GetPath()):<55} xform-err {e}")

print()
print("=" * 80)
print("JOINTS (type / axis / bodies / local frames)")
print("=" * 80)
for prim in stage.Traverse():
    tn = prim.GetTypeName()
    if "Joint" not in str(tn):
        continue
    print(f"\n--- {prim.GetPath()}  type={tn}")
    j = UsdPhysics.Joint(prim)
    b0 = j.GetBody0Rel().GetTargets()
    b1 = j.GetBody1Rel().GetTargets()
    print(f"    body0={b0}  body1={b1}")
    lp0 = j.GetLocalPos0Attr().Get(); lp1 = j.GetLocalPos1Attr().Get()
    print(f"    localPos0={lp0}  localPos1={lp1}")
    if prim.IsA(UsdPhysics.RevoluteJoint):
        rj = UsdPhysics.RevoluteJoint(prim)
        print(f"    REVOLUTE axis={rj.GetAxisAttr().Get()}  "
              f"lower={rj.GetLowerLimitAttr().Get()}  upper={rj.GetUpperLimitAttr().Get()}")

print("\nDONE")

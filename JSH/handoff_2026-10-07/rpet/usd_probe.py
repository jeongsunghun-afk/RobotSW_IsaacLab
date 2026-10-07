import os, sys
from pxr import Usd, UsdPhysics
p = os.environ["USD_PATH"]
st = Usd.Stage.Open(p)
rb = [x.GetPath().pathString for x in st.Traverse() if x.HasAPI(UsdPhysics.RigidBodyAPI)]
print("RigidBodyAPI prim %d개" % len(rb))
for x in rb[:10]:
    print("   ", x)
if len(rb) > 10:
    print("    ... (+%d)" % (len(rb) - 10))
if rb:
    print("  깊이(슬래시 수) 분포:", sorted(set(x.count("/") for x in rb)))
    # 루트 아래 상대 경로 형태
    root = st.GetDefaultPrim().GetPath().pathString if st.GetDefaultPrim() else ""
    print("  defaultPrim:", root)
    print("  예시 상대:", [x.replace(root, "<root>") for x in rb[:4]])
feet = [x for x in rb if "foot" in x.lower()]
print("  발 관련 강체 %d개:" % len(feet), [x.split("/")[-1] for x in feet])

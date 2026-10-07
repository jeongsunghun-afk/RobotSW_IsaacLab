import io, ast
p="/mnt/ssd1/jsh/RobotSW_IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/rga_leg.py"
s=io.open(p,encoding="utf-8").read()
old='''        usd_path=os.path.join(
            os.environ.get("LEG_ASSET_ROOT", _LEG_ASSET_ROOT_DEFAULT), "Leg_gen", "Leg.usd", "Leg", "Leg.usda"
        ),'''
assert s.count(old)==1, s.count(old)
new='''        # [JSH 2026-10-01] RPET_USD_PATH 로 USD 를 직접 지정할 수 있게 한다(기본 동작 불변).
        #   왜: 동봉 Leg_gen USD 는 Isaac Sim **6.0** 의 URDF 변환기 산출물이고
        #   payloads/Physics/{mujoco,physics,physx}.usda 레이아웃이다. Isaac Sim 5.1 에서 로드하면
        #   PhysicsUSD "CreateJoint - no bodies defined" 가 관절마다 나고 아티큘레이션이 안 선다.
        #   같은 URDF(assets/urdf/Leg_URDF2)를 우리 5.1 convert_urdf.py 로 재임포트한 USD 를 쓰기 위한 훅.
        usd_path=os.environ.get("RPET_USD_PATH") or os.path.join(
            os.environ.get("LEG_ASSET_ROOT", _LEG_ASSET_ROOT_DEFAULT), "Leg_gen", "Leg.usd", "Leg", "Leg.usda"
        ),'''
s=s.replace(old,new,1); ast.parse(s)
io.open(p,"w",encoding="utf-8").write(s); print("RPET_USD_PATH 훅 추가 OK")

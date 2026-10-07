import io, ast
p="/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6/source/isaaclab_assets/isaaclab_assets/robots/rga.py"
s=io.open(p,encoding="utf-8").read()
old='        usd_path="/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Leg/Leg_gen/Leg.usd/Leg/Leg.usda",'
assert s.count(old)==1, s.count(old)
new=('        # [JSH 2026-10-02] 원본은 송신자 머신 절대경로다(이 repo 에 USD 가 없다 — 09-09 조사 F-1).\n'
     '        # 인계 패키지가 동봉한 같은 자산을 LEG_ASSET_ROOT 아래에서 찾는다. 미설정 시 원본 경로 유지.\n'
     '        usd_path=__import__("os").path.join(\n'
     '            __import__("os").environ["LEG_ASSET_ROOT"], "Leg_gen", "Leg.usd", "Leg", "Leg.usda"\n'
     '        ) if __import__("os").environ.get("LEG_ASSET_ROOT")\n'
     '        else "/home/lgb/IsaacLab-6.0/source/isaaclab_assets/data/Robots/Leg/Leg_gen/Leg.usd/Leg/Leg.usda",')
s=s.replace(old,new,1); ast.parse(s)
io.open(p,"w",encoding="utf-8").write(s); print("LEG_CFG usd_path → LEG_ASSET_ROOT 훅 적용")

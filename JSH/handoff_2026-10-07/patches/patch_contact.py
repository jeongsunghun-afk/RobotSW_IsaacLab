import io, ast
p=("/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6/source/isaaclab_tasks/isaaclab_tasks/"
   "direct/leg_imitation_tracking/leg_imitation_tracking_env_cfg.py")
s=io.open(p,encoding="utf-8").read()
old='        prim_path="/World/envs/env_.*/Robot/.*",'
assert s.count(old)==1, s.count(old)
new=('        # [JSH 2026-10-06] 원본 "/Robot/.*" 는 **한 단계만** 매칭한다. 이 USD 의 강체는\n'
     '        # 운동학 사슬대로 중첩돼 있어(<root>/Geometry/Base/HL_hip_link/.../HL_foot_contact_link,\n'
     '        # 깊이 3~9) 한 단계로는 강체가 아닌 Geometry 만 걸린다. 실측: 추적 body 1개(Base),\n'
     '        # 접촉력 120 스텝 전부 0.00 N — 센서가 사실상 무력했다(09-09 조사 §G-4 경고와 일치).\n'
     '        # "/Robot/.*/.*" 식 고정 깊이가 아니라 재귀 매칭으로 전 강체 22개를 잡는다.\n'
     '        prim_path="/World/envs/env_.*/Robot/.*(/.*)*",')
s=s.replace(old,new,1); ast.parse(s)
io.open(p,"w",encoding="utf-8").write(s); print("contact prim_path 재귀 매칭으로 수정")

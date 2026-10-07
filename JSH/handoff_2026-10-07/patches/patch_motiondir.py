import io, ast, os
p="/mnt/ssd1/jsh/RobotSW_IsaacLab_isaac6/source/isaaclab_tasks/isaaclab_tasks/direct/leg_imitation_tracking/leg_imitation_tracking_env_cfg.py"
s=io.open(p,encoding="utf-8").read()
old='MOTION_FILES_DIR = os.path.join(_THIS_DIR, "imitation", "merged_leg_pkl")'
assert s.count(old)==1, s.count(old)
new=('# [JSH 2026-10-02] 기본값이 가리키던 merged_leg_pkl 은 이 repo 에 없다(송신자 머신 자산).\n'
     '# 인계 패키지가 동봉한 new_smr_leg_pkl(14 클립)로 기본값을 바꾼다.\n'
     '#   play 경로는 run 의 env.yaml 로 motion_file 을 덮어쓰므로 영향 없고,\n'
     '#   레지스트리 기본 cfg 로 env 를 만드는 경로(test_parity --with_env)만 이 값을 쓴다.\n'
     '_MOTION_DEFAULT = os.path.join(_THIS_DIR, "imitation", "merged_leg_pkl")\n'
     '_MOTION_FALLBACK = os.path.join(_THIS_DIR, "imitation", "new_smr_leg_pkl")\n'
     'MOTION_FILES_DIR = _MOTION_DEFAULT if os.path.exists(_MOTION_DEFAULT) else _MOTION_FALLBACK')
s=s.replace(old,new,1); ast.parse(s)
io.open(p,"w",encoding="utf-8").write(s); print("MOTION_FILES_DIR 폴백 적용")

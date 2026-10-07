import io
p="/mnt/ssd1/jsh/robust_suite_arms.sh"; s=io.open(p,encoding="utf-8").read()
h="    armG|armT|armTS)"
assert s.count(h)==1
s=s.replace(h,"    armG|armT|armTS|armN)",1)
old='''         export GO2_STEP_TAMOLS=tamols_cache_go2_stepping_ref GO2_STEP_TAMOLS_BASE=1'''
assert s.count(old)==1, s.count(old)
s=s.replace(old,'''         if [ "$1" = "armN" ]; then
           unset GO2_STEP_TAMOLS GO2_STEP_TAMOLS_BASE 2>/dev/null || true   # 팔N = TAMOLS 완전 제거
         else
           export GO2_STEP_TAMOLS=tamols_cache_go2_stepping_ref GO2_STEP_TAMOLS_BASE=1
         fi''',1)
old2='           armTS) PAT="*armTS_tamols_snap" ;;'
assert s.count(old2)==1
s=s.replace(old2, old2+'\n           armN)  PAT="*armN_notamols" ;;',1)
io.open(p,"w",encoding="utf-8").write(s); print("armN case 추가 OK")

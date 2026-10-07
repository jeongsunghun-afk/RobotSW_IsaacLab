import io
p="/mnt/ssd1/jsh/robust_suite_arms.sh"; s=io.open(p,encoding="utf-8").read()
h="    armG|armT|armTS|armN)"
assert s.count(h)==1
s=s.replace(h,"    armG|armT|armTS|armN|armS)",1)
o='           armN)  PAT="*armN_notamols" ;;'
assert s.count(o)==1
s=s.replace(o,o+'\n           armS)  PAT="*armS_analytic" ;;',1)
o2='         unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP 2>/dev/null || true'
assert s.count(o2)==1, s.count(o2)
s=s.replace(o2,o2+'\n         unset GO2_SREF_BASE_ANALYTIC 2>/dev/null || true\n'
 '         [ "$1" = "armS" ] && export GO2_SREF_BASE_ANALYTIC=1 GO2_SREF_SWAY_AMP=0.04',1)
io.open(p,"w",encoding="utf-8").write(s); print("armS case 추가 OK")

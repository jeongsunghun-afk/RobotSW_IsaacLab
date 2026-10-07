import io
p="/mnt/ssd1/jsh/robust_suite_arms.sh"; s=io.open(p,encoding="utf-8").read()
h="    armG|armT|armTS|armN|armS)"
assert s.count(h)==1
s=s.replace(h,"    armG|armT|armTS|armN|armS|armS2)",1)
o='           armS)  PAT="*armS_analytic" ;;'
assert s.count(o)==1
s=s.replace(o,o+'\n           armS2) PAT="*armS2_cyclemean" ;;',1)
o2='         [ "$1" = "armS" ] && export GO2_SREF_BASE_ANALYTIC=1 GO2_SREF_SWAY_AMP=0.04'
assert s.count(o2)==1
s=s.replace(o2,'         case "$1" in armS|armS2) export GO2_SREF_BASE_ANALYTIC=1 ;; esac',1)
io.open(p,"w",encoding="utf-8").write(s); print("armS2 case 추가 OK")

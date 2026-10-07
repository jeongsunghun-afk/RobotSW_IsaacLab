import io
p="/mnt/ssd1/jsh/robust_suite_arms.sh"; s=io.open(p,encoding="utf-8").read()
h="    armG|armT|armTS|armN|armS|armS2)"
assert s.count(h)==1
s=s.replace(h,"    armG|armT|armTS|armN|armS|armS2|armG7|armS27)",1)
o='           armS2) PAT="*armS2_cyclemean" ;;'
assert s.count(o)==1
# ★armG 의 PAT 가 armG7 런도 먹지 않도록 seed7 런은 이름 자체를 분리(armG7_geom)
s=s.replace(o,o+'\n           armG7)  PAT="*armG7_geom" ;;\n           armS27) PAT="*armS27_cyclemean" ;;',1)
o2='         case "$1" in armS|armS2) export GO2_SREF_BASE_ANALYTIC=1 ;; esac'
assert s.count(o2)==1
s=s.replace(o2,'         case "$1" in armS|armS2|armS27) export GO2_SREF_BASE_ANALYTIC=1 ;; esac',1)
io.open(p,"w",encoding="utf-8").write(s); print("armG7/armS27 case 추가 OK")

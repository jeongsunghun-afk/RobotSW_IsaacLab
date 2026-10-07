import io
p="/mnt/ssd1/jsh/robust_suite_arms.sh"; s=io.open(p,encoding="utf-8").read()
names = [f"armGs{i}" for i in (1,2,3)] + [f"armNs{i}" for i in (7,1,2,3)]
h = "    armG|armT|armTS|armN|armS|armS2|armG7|armS27)"
assert s.count(h)==1, s.count(h)
s = s.replace(h, "    " + "|".join(["armG","armT","armTS","armN","armS","armS2","armG7","armS27"]+names) + ")", 1)
# 몸통참조 제거 조건에 armNs* 포함
o = '''         if [ "$1" = "armN" ]; then'''
assert s.count(o)==1
s = s.replace(o, '''         case "$1" in armN*) _NOTAM=1 ;; *) _NOTAM=0 ;; esac
         if [ "$_NOTAM" = "1" ]; then''', 1)
# PAT
o2 = '           armS27) PAT="*armS27_cyclemean" ;;'
assert s.count(o2)==1
s = s.replace(o2, o2 + "\n" + "\n".join(
    f'           {n})  PAT="*{n}_*" ;;' for n in names), 1)
io.open(p,"w",encoding="utf-8").write(s)
print("다중 seed case %d개 추가 OK"%len(names))

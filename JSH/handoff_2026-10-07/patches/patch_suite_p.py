import io
p="/mnt/ssd1/jsh/robust_suite_arms.sh"; s=io.open(p,encoding="utf-8").read()
names=[f"armPs{i}" for i in (42,7,1,2,3)]
import re
m=re.search(r"^(    armG\|[^\)]*)\)$", s, re.M)
assert m, "case 헤더 못 찾음"
s=s.replace(m.group(0), m.group(1)+"|"+"|".join(names)+")",1)
anchor='           armNs3) PAT="*armNs3_*" ;;'
if anchor not in s:
    cand=[l for l in s.split("\n") if 'armNs3)' in l and 'PAT=' in l]
    assert cand, "armNs3 PAT 행 못 찾음"; anchor=cand[0]
s=s.replace(anchor, anchor+"\n"+"\n".join(f'           {n})  PAT="*{n}_*" ;;' for n in names),1)
# 팔 P 는 몸통참조를 정책+보상으로
o='         case "$1" in armS|armS2|armS27) export GO2_SREF_BASE_ANALYTIC=1 ;; esac'
assert s.count(o)==1
s=s.replace(o, o+'\n         case "$1" in armPs*) export GO2_STEP_TAMOLS_BASE_POLICY=1 ;; *) unset GO2_STEP_TAMOLS_BASE_POLICY 2>/dev/null || true ;; esac',1)
io.open(p,"w",encoding="utf-8").write(s); print("armP case 5개 추가 OK")

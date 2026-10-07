import io
p = "/mnt/ssd1/jsh/robust_suite_arms.sh"
s = io.open(p, encoding="utf-8").read()

old_head = "    armG|armT)"
assert s.count(old_head) == 1, ("head", s.count(old_head))
s = s.replace(old_head, "    armG|armT|armTS)", 1)

old_fh = '''         if [ "$1" = "armT" ]; then export GO2_STEP_TAMOLS_FH=1          # ★단일변수
         else                        unset  GO2_STEP_TAMOLS_FH 2>/dev/null || true; fi
         PAT=$([ "$1" = "armT" ] && echo "*armT_tamols" || echo "*armG_geom")'''
assert s.count(old_fh) == 1, ("fh", s.count(old_fh))
new_fh = '''         # 팔 G  = 기하 스냅            (FH off)
         # 팔 T  = TAMOLS, 스냅 없음    (FH on,  SNAP off)  ← 스냅이 제2변수로 섞인 무효 비교
         # 팔 TS = TAMOLS + 동일 스냅   (FH on,  SNAP on )  ← 단일변수 비교
         unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP 2>/dev/null || true
         case "$1" in
           armT)  export GO2_STEP_TAMOLS_FH=1 ;;
           armTS) export GO2_STEP_TAMOLS_FH=1 GO2_STEP_TAMOLS_FH_SNAP=1 ;;
         esac
         case "$1" in
           armG)  PAT="*armG_geom" ;;
           armT)  PAT="*armT_tamols" ;;
           armTS) PAT="*armTS_tamols_snap" ;;
         esac'''
s = s.replace(old_fh, new_fh, 1)
io.open(p, "w", encoding="utf-8").write(s)
print("armTS case 추가 OK")

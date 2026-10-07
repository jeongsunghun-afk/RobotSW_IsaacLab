import io
p='/mnt/ssd1/jsh/robust_suite_arms.sh'
s=io.open('/tmp/robust_suite.sh',encoding='utf-8').read()
anchor='    *)   echo "RS_BAD_POLICY $1"; exit 2 ;;'
assert s.count(anchor)==1
add = '''    armG|armT)
         # ①-비교 두 팔.  train_arm_common.inc 와 동일한 env 를 play 에도 건다.
         #  - GO2_GAIT_FREQ 는 공유블록의 1.25 핀을 그대로 둔다(학습밴드 [1.0,2.5] 안. 이 스위트의
         #    기존 정책 d9/d16 도 같은 규약이라 비교가능. FREQ 와 FREQ_RANGE 는 상호배타라 둘 다 주면 에러).
         #  - GO2_VOID_FIX=die 는 학습 그대로 켠다(=학습한 규칙으로 평가).  그 결과 R5(missed-step)는
         #    "보이드 밟으면 사망"이라 에피소드당 최대 1회가 되어 사실상 R2/R3 로 흡수된다 -- 의도된 해석변경.
         unset GO2_PRIV_COM 2>/dev/null || true
         export GO2_TRACK_SIGMA=0.04
         export GO2_VOID_FIX=die GO2_LANE_DIE=1 GO2_LANE_DIE_X=1.0 GO2_PROMOTE_ALIVE=1
         export GO2_FH_LATCH_SCORE=1 GO2_FH_LATCH_SCORE_K=1.0384 GO2_FH_EPS=1e-5
         export GO2_STEP_TAMOLS=tamols_cache_go2_stepping_ref GO2_STEP_TAMOLS_BASE=1
         export GO2_JUMP_TERRAIN=1
         if [ "$1" = "armT" ]; then export GO2_STEP_TAMOLS_FH=1          # ★단일변수
         else                        unset  GO2_STEP_TAMOLS_FH 2>/dev/null || true; fi
         PAT=$([ "$1" = "armT" ] && echo "*armT_tamols" || echo "*armG_geom")
         RUN=$(ls -1dt $B/$PAT 2>/dev/null | head -1) ; DEFVX=0.35 ;;
'''
s=s.replace(anchor, add+anchor,1)
# sweep 가 자기 자신을 /tmp/robust_suite.sh 로 재귀호출하므로 복사본 경로로 교정
s=s.replace('/tmp/robust_suite.sh batch','"$0" batch')
io.open(p,'w',encoding='utf-8').write(s)
print('robust_suite_arms.sh 생성 (armG/armT case 추가, sweep 자기참조 $0 교정)')

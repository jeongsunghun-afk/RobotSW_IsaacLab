#!/bin/bash
# =============================================================================
# [ROBUST-SUITE]  Robustness-first evaluation of Go2 stepping-stone policies.
#
# WHY THIS EXISTS.  Speed-derived scores broke three times on this terrain:
#   1. terrain_level   -- the curriculum promotes on FORWARD PROGRESS, so it scores speed, not
#                         terrain capability (a slow policy at its ceiling reads 1.1 levels low).
#   2. "reached 2.8 m in 30 s" -- 2.8 m == promote_frac(0.8) * corridor_len(3.5); a 30 s budget
#                         makes a slow-but-capable policy look like a terrain failure.
#   3. window-mean speed -- the corridor ENDS at x=4.25 and there is NO end-of-corridor
#                         termination, so a policy that finishes PARKS at x~3.85 for the rest of
#                         the episode and its episode-mean speed collapses toward 0.
# The fix is structural, not cosmetic:
#   * OUTCOME metrics (tier 1) are binary/geometry-relative, never distance-per-time.
#   * QUALITY metrics (tier 2) are per-EVENT or per-STEP inside a window the robot actually
#     TRAVERSED.  A policy that parks or stalls fails tier 1 and is N/A in tier 2 -- it can never
#     score a clean gait by standing still (D9 @ L8 stalls at x=0.78 and would otherwise report a
#     perfect 0 % missed-step rate).
#   * The time budget is generous (EPLEN=120 s default) so TIME is never the binding constraint.
#
# usage:
#   /tmp/robust_suite.sh roll  <pol> <level> <seed>          # one rollout -> dumps
#   "$0" batch <pol> <level> "<seeds>"       # JOBS-concurrent seed sweep
#   /tmp/robust_suite.sh sweep "<pols>" "<levels>" "<seeds>" # full grid
#   /tmp/robust_suite.sh rand  <pol> "<seeds>"               # realized startup DR draw per seed
#   /tmp/robust_suite.sh table                               # offline metrics table (CPU, no GPU)
#   /tmp/robust_suite.sh report                              # table + verdict + ceiling + audits
#   /tmp/robust_suite.sh audit                               # re-validate the fall / void detectors
# env knobs: JOBS(2)  PLAYVX(per-policy default)  EPLEN(120)  RS_DUMPDIR(/tmp/legsym)
#
# ---------------------------------------------------------------------------
# CONFIRMED METRIC SET (kept -- each shown to discriminate D6/D9/D10 and to survive the
# parking audit, i.e. full-episode vs cruise ranking is unchanged):
#   R1 CROSS n/8   base reaches (last stone row - 0.25 m).  Geometry-relative, not a distance/time.
#   R2 FALLS       mid-episode base-contact deaths (detected as a base_x snap-back to spawn;
#                  validated: every one is preceded by base_z collapsing 0.44 -> 0.22..0.32).
#   R3 STALL/TRUNC did-not-finish split by whether the robot was STILL ADVANCING at the buzzer
#                  (>0.25 m in the last 5 s).  TRUNC = out of clock (re-run with bigger EPLEN);
#                  STALL = stopped on its own = a real refusal.  WITHOUT this split a slow-but-
#                  capable policy is scored identically to one that quit -- that IS the speed bias.
#   R4 CEILING     highest level with CROSS>=7/8 and falls==0.  Brittle at n=8 (non-monotone for
#                  D6); report the whole level curve, use the <=1-fall variant if a scalar is needed.
#   R5 MISSED-STEP % of touchdowns landing on the z=0 void floor instead of a stone top.  Per-EVENT,
#                  so step frequency and speed divide out (corr with cruise speed = 0.16).
#                  Detected by foot z at first-contact; the touchdown-z histogram is cleanly bimodal
#                  (stone top 0.15+-hvar vs void 0.00) with an EMPTY bin at the cut.
#   R7 LANE DEV    max |base_y - lane_y| in the window.  Report the MEDIAN over seeds (mean is
#                  outlier-driven).  Independent of R5 (r=0.61).
#
# DROPPED (measured, then rejected -- do not re-add without new evidence):
#   support-polygon margin   whole-body-CoM vs base-LINK reference correlate r=1.00 and differ by
#                            5 mm (max 25 mm), flipping sign in 2/126 rollouts -- the CoM correction
#                            changes nothing.  Worse, margin is r=-0.98 with flight-phase and r=0.85
#                            with support count: it is a GAIT DESCRIPTOR, not an independent
#                            robustness signal, and it is structurally unfair across gaits (a trot
#                            is *designed* to be statically unstable, so it scores negative by
#                            construction).  Use n_sup descriptively instead.
#   void-dwell               r=0.98 with R5.        flight-phase  same cluster as margin.
#   ride height (base z)     0.42..0.45 for every non-catastrophic cell; only moves at total
#                            collapse.  Diagnostic only.
#   void-recovery            100 % in 9 of 12 cells -- a missed step almost never kills within 2 s.
#                            Low power; diagnostic only.
#
# NOT MEASURABLE HERE (honest gaps):
#   * PUSH / DISTURBANCE RECOVERY.  go2_env_cfg.py EventCfg has NO push term (no
#     push_by_setting_velocity, no apply_external_force).  Its 4 terms are physics_material
#     (startup, but the friction range is DEGENERATE: static (0.8,0.8) dynamic (0.6,0.6) -- friction
#     is NOT randomised despite being listed), add_base_mass (startup, -1..+10 kg), randomize_com
#     (startup, x+-0.15 y+-0.05 z+-0.05 m) and actuator gains (RESET, kp x[0.75,1.5] kd x[0.3,3.0]
#     log-uniform).  So external-disturbance rejection is UNMEASURED by this suite.
#   * roll/pitch.  Recoverable exactly from the dump (target_w - foot_w = R.[(th)-(fh)]_base, Kabsch
#     over the 4 feet; the recovered yaw matches dump col 4 to 0.00 deg) -- BUT the 4 vectors are
#     rank-deficient during stance, so only 6..68 % of steps solve and the solvable fraction is
#     itself policy-dependent.  Biased sample => fall forensics only, not a scored metric.
#   * TERMINATION.  _get_dones is ONLY (base-link contact force > 1.0 N) OR timeout.  There is no
#     attitude/height termination and NO end-of-corridor termination -- hence the parking artifact.
# ---------------------------------------------------------------------------
# OPERATING RULES learned the hard way in this session:
#   * NEVER edit/scp over this file while a copy of it is executing.  bash reads a script
#     incrementally by BYTE OFFSET, so overwriting it mid-run makes the running shell resume at a
#     garbage offset.  It happened: the batch loop re-ran a finished sweep and emitted
#     "line 128: operating: command not found" (text from a comment added by the overwrite).
#     Work on a copy and deploy only when nothing is running; `cp` to rs_frozen.sh to pin a run.
#   * A finished rollout always has exactly EPLEN*50 dump rows (a fall resets the env but the dump
#     continues to the budget).  rs_metrics.py REJECTS short dumps -- a killed run would otherwise
#     be scored as a fake early STALL/TRUNC outcome.
#
# GPU: CUDA_VISIBLE_DEVICES=2 and --num_envs 1 ONLY -- GPU2 is shared with another user's 4096-env
# training job.  Keep JOBS<=3.  Never touch another user's processes.
# Server files are READ-ONLY: everything here lives in /tmp.
# =============================================================================
set -u
MODE=$1; shift
DUMP=${RS_DUMPDIR:-/tmp/legsym}
B=/mnt/ssd1/jsh/RobotSW_IsaacLab/logs/rsl_rl/go2_wtw_direct

# ---- per-policy trained-env block.  Copied VERBATIM from the train_*.sh that produced each ckpt;
#      the shared block is train_d9_crawl.sh.  DEFVX = that policy's own trained forward-speed band
#      midpoint -- robustness is scored at each policy's OWN operating point, because the question
#      is "can it cross", not "how fast".  (Cross-vx cells are the OOD test, metric E.)
_setenv() {
  source /mnt/ssd1/jsh/miniconda3/etc/profile.d/conda.sh
  conda activate isaac-5.1
  export PYTHONPATH=/mnt/ssd1/jsh/np126_shim:${PYTHONPATH:-}
  cd /mnt/ssd1/jsh/RobotSW_IsaacLab
  export CUDA_VISIBLE_DEVICES=2 OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
  export GO2_TERRAIN=stepping GO2_HEIGHTMAP=1
  export GO2_GAIT=walk GO2_GAIT_DUTY=0.78 GO2_GAIT_FREQ=1.25 GO2_STEP_VX=0.25,0.45
  export GO2_STONE_HVAR=0.05
  export GO2_USE_TAMOLS_CACHE=1 GO2_FOOTHOLD_TRACK_SCALE=0.3
  export GO2_FH_Z3D=1 GO2_SWING_TERRAIN=1 GO2_FH_Z_FROM_HMAP=1
  export GO2_SWING_TRACK=1 GO2_LANE_VX_GATE=1
  export GO2_FH_LOOKAHEAD_PITCH=0.5 GO2_FH_LATCH=1 GO2_FH_AHEAD=1
  export GO2_FH_HMAP_ONSTONE=1 GO2_FH_DZ_MAX=0.10
  case "$1" in
    d6)  # TROT.  trained vx[0.3,0.8], gait_frequency[1.5,4.0] randomised; no duty/freq/latch pins.
         unset GO2_PRIV_COM GO2_GAIT_DUTY GO2_GAIT_FREQ GO2_STEP_VX GO2_FH_LATCH 2>/dev/null || true
         export GO2_GAIT=trot
         RUN=$B/2026-08-28_11-42-21_stepH_std_dtc_v6 ; DEFVX=0.4 ;;
    d9)  # WALK.  trained vx[0.25,0.45].
         unset GO2_PRIV_COM 2>/dev/null || true
         RUN=$B/2026-09-01_09-55-50_stepH_std_dtc_v9_crawl ; DEFVX=0.4 ;;
    d10) # WALK-SLOW (crawl-match).  trained vx[0.15,0.25], gait_frequency 0.63.
         unset GO2_PRIV_COM 2>/dev/null || true
         export GO2_GAIT=walk GO2_GAIT_DUTY=0.78 GO2_GAIT_FREQ=0.63 GO2_STEP_VX=0.15,0.25
         RUN=$B/2026-09-01_16-30-30_stepH_std_dtc_v10_crawlmatch ; DEFVX=0.2 ;;
    d16) # WALK + the single D16 delta: linear-velocity tracking kernel sigma_v 0.50 -> 0.20 m/s
         # (GO2_TRACK_SIGMA=0.04) with the level-preserving gain k=1.7997 applied automatically.
         # This env var MUST be set at PLAY time too -- without it the loaded policy runs against a
         # different reward kernel than it trained on.  Everything else byte-identical to d9.
         unset GO2_PRIV_COM 2>/dev/null || true
         export GO2_TRACK_SIGMA=0.04
         RUN=$B/2026-09-04_19-47-55_stepH_std_dtc_v16_sigma ; DEFVX=0.4 ;;
    d14) export GO2_PRIV_COM=1
         RUN=$(ls -d $B/*stepH_std_dtc_v14_privcom 2>/dev/null | tail -1) ; DEFVX=0.4 ;;
    armG|armT|armTS|armN|armS|armS2|armG7|armS27|armGs1|armGs2|armGs3|armNs7|armNs1|armNs2|armNs3|armPs42|armPs7|armPs1|armPs2|armPs3)
         # ①-비교 두 팔.  train_arm_common.inc 와 동일한 env 를 play 에도 건다.
         #  - GO2_GAIT_FREQ 는 공유블록의 1.25 핀을 그대로 둔다(학습밴드 [1.0,2.5] 안. 이 스위트의
         #    기존 정책 d9/d16 도 같은 규약이라 비교가능. FREQ 와 FREQ_RANGE 는 상호배타라 둘 다 주면 에러).
         #  - GO2_VOID_FIX=die 는 학습 그대로 켠다(=학습한 규칙으로 평가).  그 결과 R5(missed-step)는
         #    "보이드 밟으면 사망"이라 에피소드당 최대 1회가 되어 사실상 R2/R3 로 흡수된다 -- 의도된 해석변경.
         unset GO2_PRIV_COM 2>/dev/null || true
         export GO2_TRACK_SIGMA=0.04
         export GO2_VOID_FIX=die GO2_LANE_DIE=1 GO2_LANE_DIE_X=1.0 GO2_PROMOTE_ALIVE=1
         export GO2_FH_LATCH_SCORE=1 GO2_FH_LATCH_SCORE_K=1.0384 GO2_FH_EPS=1e-5
         case "$1" in armN*) _NOTAM=1 ;; *) _NOTAM=0 ;; esac
         if [ "$_NOTAM" = "1" ]; then
           unset GO2_STEP_TAMOLS GO2_STEP_TAMOLS_BASE 2>/dev/null || true   # 팔N = TAMOLS 완전 제거
         else
           export GO2_STEP_TAMOLS=tamols_cache_go2_stepping_ref GO2_STEP_TAMOLS_BASE=1
         fi
         export GO2_JUMP_TERRAIN=1
         # 팔 G  = 기하 스냅            (FH off)
         # 팔 T  = TAMOLS, 스냅 없음    (FH on,  SNAP off)  ← 스냅이 제2변수로 섞인 무효 비교
         # 팔 TS = TAMOLS + 동일 스냅   (FH on,  SNAP on )  ← 단일변수 비교
         unset GO2_STEP_TAMOLS_FH GO2_STEP_TAMOLS_FH_SNAP 2>/dev/null || true
         unset GO2_SREF_BASE_ANALYTIC 2>/dev/null || true
         case "$1" in armS|armS2|armS27) export GO2_SREF_BASE_ANALYTIC=1 ;; esac
         case "$1" in armPs*) export GO2_STEP_TAMOLS_BASE_POLICY=1 ;; *) unset GO2_STEP_TAMOLS_BASE_POLICY 2>/dev/null || true ;; esac
         case "$1" in
           armT)  export GO2_STEP_TAMOLS_FH=1 ;;
           armTS) export GO2_STEP_TAMOLS_FH=1 GO2_STEP_TAMOLS_FH_SNAP=1 ;;
         esac
         case "$1" in
           armG)  PAT="*armG_geom" ;;
           armT)  PAT="*armT_tamols" ;;
           armTS) PAT="*armTS_tamols_snap" ;;
           armN)  PAT="*armN_notamols" ;;
           armS)  PAT="*armS_analytic" ;;
           armS2) PAT="*armS2_cyclemean" ;;
           armG7)  PAT="*armG7_geom" ;;
           armS27) PAT="*armS27_cyclemean" ;;
           armGs1)  PAT="*armGs1_*" ;;
           armGs2)  PAT="*armGs2_*" ;;
           armGs3)  PAT="*armGs3_*" ;;
           armNs7)  PAT="*armNs7_*" ;;
           armNs1)  PAT="*armNs1_*" ;;
           armNs2)  PAT="*armNs2_*" ;;
           armNs3)  PAT="*armNs3_*" ;;
           armPs42)  PAT="*armPs42_*" ;;
           armPs7)  PAT="*armPs7_*" ;;
           armPs1)  PAT="*armPs1_*" ;;
           armPs2)  PAT="*armPs2_*" ;;
           armPs3)  PAT="*armPs3_*" ;;
         esac
         RUN=$(ls -1dt $B/$PAT 2>/dev/null | head -1) ; DEFVX=0.35 ;;
    *)   echo "RS_BAD_POLICY $1"; exit 2 ;;
  esac
}

case "$MODE" in
roll)
  POL=$1; LEVEL=$2; SEED=$3
  _setenv "$POL"
  CKPT=${CKPT:-$RUN/model_2999.pt}
  PLAYVX=${PLAYVX:-$DEFVX}
  EPLEN=${EPLEN:-120}                 # generous by DEFAULT: time must not be the binding constraint
  export EPISODE_LEN_S=$EPLEN
  export GO2_PLAY_TERRAIN_LEVEL=$LEVEL GO2_PLAY_PIN_VX=1 GO2_PLAY_VX=$PLAYVX GO2_PLAY_FULL_EP=1
  export GO2_FH_MAX_STEPS=$((50 * EPLEN)) GO2_RAY_VIS=0
  mkdir -p "$DUMP"
  SUF=""; [ "$PLAYVX" = "0.4" ] || SUF="_vx${PLAYVX}"; [ "$EPLEN" = "30" ] || SUF="${SUF}_ep${EPLEN}"
  TAG=${POL}_L${LEVEL}_s${SEED}${SUF}
  export GO2_BASE_X_DUMP=1 GO2_BASE_X_FILE=$DUMP/${TAG}_basex.txt
  export GO2_FOOT_DIAG=1   GO2_FOOT_DIAG_FILE=$DUMP/${TAG}_footdiag.txt
  rm -f "$GO2_BASE_X_FILE" "$GO2_FOOT_DIAG_FILE"
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
    --task Go2WTW --num_envs 1 --headless --seed $SEED --checkpoint "$CKPT" > $DUMP/${TAG}.log 2>&1
  echo "RS_DONE $TAG exit=$? fd=$(wc -l < $GO2_FOOT_DIAG_FILE 2>/dev/null || echo 0)"
  ;;
batch)
  # Job control uses REAL PIDs + `wait`, not a pgrep name match.  The pgrep form was fragile:
  # it matched by script NAME, so a renamed copy did not gate, and the count could momentarily
  # read 0 between two rolls and let a queued chain jump in.  `wait -n` has neither failure mode.
  POL=$1; LEVEL=$2; SEEDS=$3; J=${JOBS:-2}
  mkdir -p "$DUMP"; PIDS=()
  for S in $SEEDS; do
    while [ "${#PIDS[@]}" -ge "$J" ]; do
      wait -n 2>/dev/null || true
      NEW=(); for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && NEW+=("$p"); done; PIDS=("${NEW[@]}")
    done
    echo "[rs] START $POL L$LEVEL s$S $(date +%H:%M:%S)"
    PLAYVX=${PLAYVX:-} EPLEN=${EPLEN:-} "$0" roll "$POL" "$LEVEL" "$S" >> "$DUMP/rs_batch.log" 2>&1 &
    PIDS+=($!)
  done
  wait
  echo "[rs] DONE $POL L$LEVEL $(date +%H:%M:%S)"
  ;;
sweep)
  POLS=$1; LEVELS=$2; SEEDS=$3
  for P in $POLS; do for L in $LEVELS; do "$0" batch $P $L "$SEEDS"; done; done
  echo "[rs] SWEEP DONE $(date +%H:%M:%S)"
  ;;
rand)
  POL=$1; SEEDS=$2; mkdir -p "$DUMP"; : > $DUMP/rs_rand_${POL}.log
  for S in $SEEDS; do   # one Isaac app per seed: the probe's own --seeds loop hangs on gym.make #2
    echo "@@@@ seed $S" >> $DUMP/rs_rand_${POL}.log
    ( _setenv "$POL"; export PYTHONUNBUFFERED=1
      export GO2_PLAY_TERRAIN_LEVEL=7 GO2_PLAY_PIN_VX=1 GO2_PLAY_VX=$DEFVX GO2_PLAY_FULL_EP=1
      ./isaaclab.sh -p /tmp/go2_rand_probe.py --seeds "$S" --num_envs 1 2>&1 \
        | grep -E "### SEED |base mass|TOTAL mass|base COM local|randomize_com draw" ) >> $DUMP/rs_rand_${POL}.log
  done
  echo "RS_RAND_DONE $POL"
  ;;
table)
  # rs_table.py MUST run first: it (re)builds /tmp/rs_rows.json, which every other section reads.
  RS_DUMPDIR=$DUMP python3 /tmp/rs_table.py "$@" || exit 1
  ;;
report)
  # one-command full report: outcome + quality + redundancy + ceiling, then the audits.
  RS_DUMPDIR=$DUMP python3 /tmp/rs_table.py || exit 1
  for s in rs_verdict rs_discrim rs_slope rs_park rs_ood rs_lead; do
    [ -f /tmp/$s.py ] && { echo; echo "############ $s ############"; python3 /tmp/$s.py 2>/dev/null; }
  done
  ;;
audit)
  # detector-validation pass: fall detector vs base_z collapse, void cut vs touchdown-z histogram,
  # the D9 zero-miss claim, and whether roll/pitch is recoverable at all.
  for s in rs_validate rs_check0 rs_attitude; do
    [ -f /tmp/$s.py ] && { echo "############ $s ############"; python3 /tmp/$s.py 2>/dev/null; echo; }
  done
  ;;
*) echo "RS_BAD_MODE $MODE"; exit 2 ;;
esac

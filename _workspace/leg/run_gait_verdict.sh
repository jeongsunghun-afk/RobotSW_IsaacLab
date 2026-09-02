#!/usr/bin/env bash
# 보행 전환 판정 한 벌 — 4096 env 조사 3종 + 전환 프로브 + 그림 + 추적카메라 영상.
#
# 판정마다 손으로 커맨드를 조립하면 매번 플래그 하나씩 빠진다. 실제로 그렇게
#   `motion_weight_mode` 누락(램프가 학습과 다른 클립 분포를 씀)
#   `--dur_s 10` 이 에피소드 경계와 겹쳐 표본 4096→1
# 두 건을 겪었다. 그래서 한 곳에 고정한다.
#
# ★ 조사 길이가 두 가지인 이유:
#     격자(dur 8) : 마지막 2 s 창이 10 s 에피소드 리셋을 걸치면 안 된다.
#     프로브(dur 10): 명령 변경(4~7 s) 앞뒤로 2 s 창이 둘 다 들어가야 한다.
#
# 사용:
#   bash _workspace/leg/run_gait_verdict.sh <RUN_DIR> <ITER> <TAG> [BASE_RSI_NPZ] [BASE_STAND_NPZ]
set -euo pipefail
cd /home/lgb/IsaacLab-6.0

RUN=$1; IT=$2; TAG=$3
BASE_RSI=${4:-}; BASE_STAND=${5:-}
CKPT=$RUN/model_$IT.pt
PARAMS=$RUN/params/env.yaml
[ -f "$CKPT" ] || { echo "체크포인트 없음: $CKPT" >&2; exit 1; }

W=_workspace/leg
D=reports/leg_imitation/_comparisons/cmdchg_in_episode_command
LOG=${SCRATCH:-/tmp}/gait_verdict_$TAG
# 학습이 도는 GPU 를 피해야 한다 — 기본 0,1,2 이지만 GA/GB/GC 로 덮어쓸 수 있다.
GA=${GA:-0}; GB=${GB:-1}; GC=${GC:-2}
# 프로브 창 길이. 명령 재샘플 주기가 짧으면(예: 4 s) 창 안에 변경이 여러 번 들어오는데,
# 프로브는 변경마다 표본을 만들므로 길수록 표본이 많다. 단 에피소드 길이를 넘기면 안 된다.
PROBE_DUR=${PROBE_DUR:-10}
mkdir -p "$LOG" "$D/figures" "$D/videos" "$D/metrics"

export CONDA_PREFIX=/home/user/miniconda3/envs/isaac-6.0
PY=$CONDA_PREFIX/bin/python
unset DISPLAY   # DISPLAY 가 있으면 headless 여도 GLX 를 잡으려다 GLXBadFBConfig 로 죽는다

srv() { # gpu dur extra out
  CUDA_VISIBLE_DEVICES=$1 ./isaaclab.sh -p $W/gait_survey_multienv.py \
    --checkpoint "$CKPT" --run_params "$PARAMS" --n_envs 4096 --dur_s $2 $3 \
    --out_dir "$W/$4" > "$LOG/$4.log" 2>&1
}
vid() { # gpu extra out
  CUDA_VISIBLE_DEVICES=$1 ./isaaclab.sh -p $W/speed_ramp_record_rma.py \
    --checkpoint "$CKPT" --run_params "$PARAMS" --cam_view chase $2 \
    --out_dir "$D/videos/$3" > "$LOG/$3.log" 2>&1
}

echo ">>> [1/3] 4096 env 조사 3종 + 영상"
( srv $GA 8          ""            "gv_${TAG}_rsi"
  srv $GA $PROBE_DUR "" "gv_${TAG}_tr" ) &
( srv $GB 8 "--all_stand" "gv_${TAG}_stand"
  vid $GB "--vx_const 3.0 --hold_s 8.0 --ramp_s 0 --force_stand" "${TAG}_stand" ) &
( vid $GC "--vx_const 3.0 --hold_s 8.0 --ramp_s 0" "${TAG}_rsi_a"
  vid $GC "--vx_const 3.0 --hold_s 8.0 --ramp_s 0" "${TAG}_rsi_b"
  vid $GC "--vx_max 3.2 --hold_s 3.0 --ramp_s 2.0 --force_stand" "${TAG}_ramp" ) &
wait

echo ">>> [2/3] 그림"
SEL=(); CON=(); TRA=()
[ -n "$BASE_RSI" ] && SEL+=("baseline=$BASE_RSI")
SEL+=("cmdchg@$IT=$W/gv_${TAG}_rsi/gait_survey.npz")
[ -n "$BASE_RSI" ] && [ -n "$BASE_STAND" ] && CON+=("baseline=$BASE_RSI,$BASE_STAND")
CON+=("cmdchg@$IT=$W/gv_${TAG}_rsi/gait_survey.npz,$W/gv_${TAG}_stand/gait_survey.npz")
TRA+=("cmdchg@$IT=$W/gv_${TAG}_tr/gait_survey.npz")
$PY $W/plot_gait_verdict.py --out "$D/figures/gait_verdict_$TAG.png" \
  --title "In-episode command changes (cmdchg) at $IT" \
  --selectivity "${SEL[@]}" --contrast "${CON[@]}" --transition "${TRA[@]}"

echo ">>> [3/3] 원자료 + 영상 보행 검증 (캡션 근거)"
{ echo "# cmdchg 판정 원자료 — $TAG (model_$IT)"; echo
  for t in rsi stand; do echo "## gv_${TAG}_$t"; echo '```'
    $PY $W/gait_survey_analyze.py "$W/gv_${TAG}_$t/gait_survey.npz" 2>&1; echo '```'; echo; done
  echo "## 전환 프로브"; echo '```'
  $PY $W/gait_transition_probe.py "$W/gv_${TAG}_tr/gait_survey.npz" 2>&1; echo '```'; echo
  echo "## 영상 클립의 실제 보행 (캡션은 반드시 이 표에 맞출 것)"; echo '```'
  $PY $W/gait_classify.py \
    "${TAG}_rsi_a=$D/videos/${TAG}_rsi_a/ramp_data.npz" \
    "${TAG}_rsi_b=$D/videos/${TAG}_rsi_b/ramp_data.npz" \
    "${TAG}_stand=$D/videos/${TAG}_stand/ramp_data.npz" --cmds 3.0 --keep_fallen 2>&1
  $PY $W/gait_classify.py "${TAG}_ramp=$D/videos/${TAG}_ramp/ramp_data.npz" \
    --cmds 0.5 1.0 1.5 2.0 2.5 3.0 --keep_fallen 2>&1
  echo '```'; } > "$D/metrics/verdict_$TAG.md"

echo ">>> 완료: $D/figures/gait_verdict_$TAG.png · $D/metrics/verdict_$TAG.md"
echo ">>> 영상은 $D/videos/${TAG}_* — 위 표로 보행을 확인한 뒤 이름을 붙일 것"

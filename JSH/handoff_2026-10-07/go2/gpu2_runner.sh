#!/bin/bash
# GPU2 전용 직렬 큐 (상주형).  큐가 비어도 종료하지 않고 폴링한다 —
# 이전판은 비면 종료해서 나중에 넣은 작업을 못 집어갔고 GPU 가 3시간 놀았다.
# 정지: /mnt/ssd1/jsh/gpu2q/STOP 파일 생성.
Q=/mnt/ssd1/jsh/gpu2q; S=/mnt/ssd1/jsh/gpu2_queue_status.txt; mkdir -p $Q/done
[ -n "${1:-}" ] && { echo "[q] 선행 pid=$1 대기 $(date '+%F %T')" >> $S
  while kill -0 "$1" 2>/dev/null; do sleep 60; done; }
echo "[q] 러너 기동(상주형) $(date '+%F %T')" >> $S
IDLE=0
while true; do
  [ -f $Q/STOP ] && { echo "[q] STOP 감지 — 종료 $(date '+%F %T')" >> $S; rm -f $Q/STOP; break; }
  J=$(ls -1 $Q/[0-9]*.sh 2>/dev/null | head -1)
  if [ -z "$J" ]; then
    IDLE=$((IDLE+1)); [ $((IDLE % 30)) -eq 1 ] && echo "[q] 큐 비어있음 — 대기중 $(date '+%F %T')" >> $S
    sleep 60; continue
  fi
  IDLE=0; N=$(basename $J)
  echo "[q] ▶ $N 시작 $(date '+%F %T')" >> $S
  bash "$J" >> /mnt/ssd1/jsh/gpu2q_${N%.sh}.log 2>&1; RC=$?
  echo "[q] ■ $N 종료 exit=$RC $(date '+%F %T')" >> $S
  mv "$J" $Q/done/
done

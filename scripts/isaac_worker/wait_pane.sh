#!/usr/bin/env bash
# Poll exit.status sentinel files for one or more isaac-worker jobs.
#
# Usage:
#   scripts/isaac_worker/wait_pane.sh <job-id> [<job-id> ...]
#
# Env:
#   POLL_INTERVAL  seconds between polls (default: 2)
#   TIMEOUT        max wait seconds (default: 3600)
#
# Output:
#   one line per finished job:  <job-id> <exit-code>
#   non-zero overall exit if any job timed out.

set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <job-id> [<job-id> ...]" >&2
  exit 64
fi

POLL_INTERVAL="${POLL_INTERVAL:-2}"
TIMEOUT="${TIMEOUT:-3600}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

START_TS=$(date +%s)
declare -A DONE=()

while true; do
  ALL_DONE=1
  for JOB_ID in "$@"; do
    if [[ -n "${DONE[$JOB_ID]:-}" ]]; then
      continue
    fi
    STATUS_FILE="${PROJECT_ROOT}/.claude/worker-runs/${JOB_ID}/exit.status"
    if [[ -f "${STATUS_FILE}" ]]; then
      RC="$(cat "${STATUS_FILE}" | tr -d '[:space:]')"
      DONE["${JOB_ID}"]="${RC}"
      echo "${JOB_ID} ${RC}"
    else
      ALL_DONE=0
    fi
  done

  (( ALL_DONE == 1 )) && exit 0

  NOW=$(date +%s)
  if (( NOW - START_TS > TIMEOUT )); then
    for JOB_ID in "$@"; do
      if [[ -z "${DONE[$JOB_ID]:-}" ]]; then
        echo "${JOB_ID} TIMEOUT"
      fi
    done
    exit 124
  fi

  sleep "${POLL_INTERVAL}"
done

#!/usr/bin/env bash
# Synchronous Agent()-equivalent: spawn one isaac-worker pane, wait, print result.
#
# Usage:
#   scripts/isaac_worker/dispatch.sh <agent> <task>
#
# Stdout: the worker's `result` text (single chunk).
# Stderr: human-readable progress (job-id, timing, exit code).
# Exit code:
#   0  — worker succeeded
#   2  — claude session reported is_error=true
#   N  — propagated from spawn / wait / extract

set -euo pipefail

AGENT="${1:?usage: dispatch.sh <agent> <task>}"
TASK="${2:?usage: dispatch.sh <agent> <task>}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPAWN="${ROOT}/scripts/isaac_worker/spawn_pane.sh"
WAIT="${ROOT}/scripts/isaac_worker/wait_pane.sh"
GETR="${ROOT}/scripts/isaac_worker/get_result.sh"

JOB=$("${SPAWN}" "${AGENT}" "${TASK}")
echo "[dispatch] spawned job=${JOB}" >&2

LINE=$("${WAIT}" "${JOB}")
RC=$(awk '{print $2}' <<<"${LINE}")
echo "[dispatch] worker exited rc=${RC}" >&2

# get_result.sh exits 2 if is_error=true; preserve that
"${GETR}" "${JOB}" --result || GETR_RC=$?
GETR_RC="${GETR_RC:-0}"

# Final exit: prefer worker rc unless it's 0 and result has is_error
if [[ "${RC}" != "0" ]]; then
  exit "${RC}"
fi
exit "${GETR_RC}"

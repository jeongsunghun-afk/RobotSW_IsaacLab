#!/usr/bin/env bash
# Extract result / metadata from a finished isaac-worker job.
#
# Usage:
#   scripts/isaac_worker/get_result.sh <job-id>            # default: print result text
#   scripts/isaac_worker/get_result.sh <job-id> --json     # print full JSON
#   scripts/isaac_worker/get_result.sh <job-id> --meta     # print metadata summary
#   scripts/isaac_worker/get_result.sh <job-id> --result   # print result text (explicit)
#
# Exit codes:
#   0  — success
#   1  — log not found
#   2  — claude session reported is_error=true (result text is the error message)

set -euo pipefail

JOB_ID="${1:?usage: get_result.sh <job-id> [--json|--meta|--result]}"
MODE="${2:---result}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG="${PROJECT_ROOT}/.claude/worker-runs/${JOB_ID}/output.log"

if [[ ! -f "${LOG}" ]]; then
  echo "ERROR: log not found: ${LOG}" >&2
  exit 1
fi

case "${MODE}" in
  --json|-j)
    cat "${LOG}"
    ;;
  --meta|-m)
    python3 - "${LOG}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"result      : {repr(d.get('result'))[:200]}")
print(f"is_error    : {d.get('is_error')}")
print(f"num_turns   : {d.get('num_turns')}")
print(f"duration_ms : {d.get('duration_ms')}")
print(f"cost_usd    : {round(d.get('total_cost_usd', 0), 5)}")
print(f"session_id  : {d.get('session_id')}")
print(f"model       : {list(d.get('modelUsage', {}).keys())}")
PY
    ;;
  --result|-r|*)
    python3 - "${LOG}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
text = d.get('result', '')
print(text)
sys.exit(2 if d.get('is_error') else 0)
PY
    ;;
esac

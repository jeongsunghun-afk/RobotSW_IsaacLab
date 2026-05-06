#!/usr/bin/env bash
# Spawn one IsaacLab agent as a `claude` CLI process in a tmux split pane.
#
# Usage:
#   scripts/isaac_worker/spawn_pane.sh <agent-name> <task-text>
#
# Outputs the generated <job-id> on stdout.
#
# Side effects:
#   - Creates  .claude/worker-runs/<job-id>/{prompt.txt, launch.sh, pane_id}
#   - On worker exit, the pane writes  .claude/worker-runs/<job-id>/{output.log, exit.status}
#
# Requires: running inside an existing tmux session ($TMUX set).

set -euo pipefail

AGENT="${1:?usage: spawn_pane.sh <agent-name> <task-text>}"
TASK="${2:?usage: spawn_pane.sh <agent-name> <task-text>}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AGENT_FILE="${PROJECT_ROOT}/.claude/agents/${AGENT}.md"

if [[ ! -f "${AGENT_FILE}" ]]; then
  echo "ERROR: agent file not found: ${AGENT_FILE}" >&2
  exit 1
fi

if [[ -z "${TMUX:-}" ]]; then
  echo "ERROR: not inside a tmux session (\$TMUX is empty)" >&2
  exit 2
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: claude CLI not found in PATH" >&2
  exit 3
fi

RAND_SUFFIX="$(head -c 4 /dev/urandom | xxd -p)"
JOB_ID="$(date +%Y%m%d-%H%M%S)-${AGENT}-${RAND_SUFFIX}"
RUN_DIR="${PROJECT_ROOT}/.claude/worker-runs/${JOB_ID}"
mkdir -p "${RUN_DIR}"

printf '%s' "${TASK}" > "${RUN_DIR}/prompt.txt"

cat > "${RUN_DIR}/launch.sh" <<'LAUNCH_EOF'
#!/usr/bin/env bash
set -uo pipefail
RUN_DIR="$1"
AGENT="$2"
PROJECT_ROOT="$3"

cd "${PROJECT_ROOT}"

JOB_ID="$(basename "${RUN_DIR}")"
echo "=== isaac-worker | agent=${AGENT} | job=${JOB_ID} ==="
echo "=== task ==="
cat "${RUN_DIR}/prompt.txt"
echo
echo "=== claude session start ==="

claude --print \
  --agent "${AGENT}" \
  --permission-mode acceptEdits \
  --output-format json \
  --name "${AGENT}@${JOB_ID}" \
  "$(cat "${RUN_DIR}/prompt.txt")" \
  2>&1 | tee "${RUN_DIR}/output.log"
RC=${PIPESTATUS[0]}

echo "${RC}" > "${RUN_DIR}/exit.status"
echo
echo "=== worker exited (rc=${RC}); press Enter to close pane ==="
read -r _ || true
LAUNCH_EOF
chmod +x "${RUN_DIR}/launch.sh"

PANE_ID=$(tmux split-window -h -P -F '#{pane_id}' \
  "${RUN_DIR}/launch.sh" "${RUN_DIR}" "${AGENT}" "${PROJECT_ROOT}")
echo "${PANE_ID}" > "${RUN_DIR}/pane_id"

echo "${JOB_ID}"

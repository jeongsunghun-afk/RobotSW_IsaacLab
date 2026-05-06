#!/usr/bin/env bash
# run_iteration.sh — Autoresearch cron driver for amp-locomotion-style-autoresearch
#
# Cron line (every 3 hours):
#   0 */3 * * * /home/lgb/IsaacLab/scripts/autoresearch/run_iteration.sh >> /home/lgb/IsaacLab/.omc/autoresearch/amp-locomotion-style-autoresearch/runs/run-2026-04-30T03-55Z/cron.log 2>&1
#
# On unexpected error (exit 1), cron mails to the local user (MAILTO in crontab).

set -euo pipefail

# Conda env activation for non-interactive (cron / sub-shell) invocation.
# Interactive shells get this via ~/.bashrc; cron does not, hence the explicit init.
CONDA_INIT="/home/user/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="isaac"
if [[ -f "${CONDA_INIT}" ]]; then
    set +u
    # shellcheck disable=SC1090
    source "${CONDA_INIT}"
    conda activate "${CONDA_ENV_NAME}"
    set -u
else
    echo "FATAL: conda init script not found at ${CONDA_INIT}" >&2
    exit 1
fi

# ────────────────────────────────────────────────────────────────────────────────
# CONSTANTS (tune here)
# ────────────────────────────────────────────────────────────────────────────────
ITER_PER_FIRING="${ITER_PER_FIRING:-1000}"    # max PPO iterations per firing
TRAIN_TIMEOUT="${TRAIN_TIMEOUT:-7200}"        # seconds = 2h; leaves ~30 min for eval
RECORD_VIDEO="${RECORD_VIDEO:-1}"             # 1 = pass --video to train.py, 0 = headless-only
GPU_PIN="${GPU_PIN:-}"       # set to a numeric index (e.g. "1") to pin; empty = auto-pick
GPU_UTIL_LIMIT=50            # % utilization above which a GPU is considered busy
GPU_MEM_LIMIT_MB=4096        # MB memory used above which a GPU is considered busy

ISAACLAB_ROOT="/home/lgb/IsaacLab"
ISAACLAB_SH="${ISAACLAB_ROOT}/isaaclab.sh"
MISSION_SLUG="${MISSION_SLUG:-amp-locomotion-style-autoresearch}"
RUN_ID="${RUN_ID:-run-2026-04-30T03-55Z}"
MISSION_DIR="${ISAACLAB_ROOT}/.omc/autoresearch/${MISSION_SLUG}"
RUN_DIR="${MISSION_DIR}/runs/${RUN_ID}"
STATE_FILE="${RUN_DIR}/state.json"
MUTATIONS_FILE="${MISSION_DIR}/mutations.json"
BASELINE_CKPT_DEFAULT="${BASELINE_CKPT_DEFAULT:-logs/skrl/go2_skrl_amp/2026-04-22_15-07-02_amp_torch_skrl_amp_add_lin_vel/checkpoints/best_agent.pt}"
DECISION_LOG="${RUN_DIR}/decision-log.md"
EVAL_SCRIPT="scripts/autoresearch/eval_amp_style.py"
TASK="Isaac-Go2-AMP-Direct-v0"
CMD_GRID="vx=0.5,1.0,1.5,2.0;vy=0;wz=-0.5,0,0.5"
NUM_SEEDS=4
ROLLOUT_STEPS=500
LOCK_FILE="/tmp/autoresearch-amp-locomotion.lock"

# ────────────────────────────────────────────────────────────────────────────────
# LOCK: only one firing at a time
# ────────────────────────────────────────────────────────────────────────────────
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Another firing is running — exiting." >&2
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────────────
# ENV SETUP: cron has minimal PATH; source conda/bashrc
# ────────────────────────────────────────────────────────────────────────────────
export HOME="/home/lgb"
# shellcheck source=/dev/null
if [[ -f "${HOME}/.bashrc" ]]; then
    source "${HOME}/.bashrc" 2>/dev/null || true
fi
# Ensure conda is on PATH if available
if [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true
elif [[ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null || true
fi

cd "${ISAACLAB_ROOT}"

# Ensure mission/run directories exist (so first firing of a new mission can
# write state.json, decision-log, and per-iter artifacts without errors).
mkdir -p "${MISSION_DIR}" "${RUN_DIR}/evaluations"
[[ -f "${RUN_DIR}/decision-log.md" ]] || cat > "${RUN_DIR}/decision-log.md" <<EOF
# Autoresearch Decision Log — \`${MISSION_SLUG}\`

**Run ID:** \`${RUN_ID}\`
**Mission dir:** \`${MISSION_DIR}\`
EOF

# ────────────────────────────────────────────────────────────────────────────────
# HELPERS
# ────────────────────────────────────────────────────────────────────────────────

# Atomic append to decision-log via temp file
dlog_append() {
    local tmp
    tmp="$(mktemp "${MISSION_DIR}/.dltemp.XXXXXX")"
    cat > "${tmp}"
    cat "${tmp}" >> "${DECISION_LOG}"
    rm -f "${tmp}"
}

# UTC ISO8601 timestamp
now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Python state reader: state_get <key>
state_get() {
    python3 -c "
import json, sys
with open('${STATE_FILE}') as f:
    d = json.load(f)
val = d.get('$1', None)
if val is None:
    sys.stdout.write('null')
else:
    sys.stdout.write(str(val))
"
}

# Python state writer: state_set <key> <value_as_python_literal>
# value_as_python_literal: True, False, None, 1, 1.5, '"string"'
state_set() {
    local key="$1"
    local value="$2"
    python3 -c "
import json, os, tempfile
key = '${key}'
value = ${value}
with open('${STATE_FILE}') as f:
    d = json.load(f)
d[key] = value
tmp = '${STATE_FILE}.tmp'
with open(tmp, 'w') as f:
    json.dump(d, f, indent=2)
    f.write('\n')
os.replace(tmp, '${STATE_FILE}')
"
}

# Multi-key state update: state_update <python_dict_literal>
state_update() {
    local updates="$1"
    python3 -c "
import json, os
updates = ${updates}
with open('${STATE_FILE}') as f:
    d = json.load(f)
d.update(updates)
tmp = '${STATE_FILE}.tmp'
with open(tmp, 'w') as f:
    json.dump(d, f, indent=2)
    f.write('\n')
os.replace(tmp, '${STATE_FILE}')
"
}

# Apply a single mutation entry using op dispatch
# apply_mutation <file> <op> <yaml_path_or_field> <new_value_python> [<index>]
apply_mutation_entry() {
    local file="$1"
    local op="$2"
    local path_or_field="$3"
    local new_value="$4"
    local dict_index="${5:-}"
    python3 - "${file}" "${op}" "${path_or_field}" "${new_value}" "${dict_index}" <<'PYEOF'
import sys, re, pathlib

filepath = sys.argv[1]
op       = sys.argv[2]
path_or_field = sys.argv[3]
new_value_raw = sys.argv[4]   # Python-literal string, e.g. "0.85" or "'hello'"
dict_index_raw = sys.argv[5]  # for py_dict_index: "1" (int index into list)

text = pathlib.Path(filepath).read_text()

def parse_val(s):
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s in ("True", "False"):
        return s == "True"
    return s.strip("'\"")

new_val = parse_val(new_value_raw)

if op == "yaml_set":
    # path_or_field is dot-separated key chain, e.g. "agent.style_reward_scale"
    keys = path_or_field.split(".")
    leaf = keys[-1]
    # Match the leaf key in YAML (scalar value on same line)
    pattern = re.compile(
        r'^(\s*' + re.escape(leaf) + r'\s*:\s*)([^\n#]+)',
        re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        print(f"ERROR: yaml_set key '{leaf}' not found in {filepath}", file=sys.stderr)
        sys.exit(1)
    if isinstance(new_val, float):
        new_str = f"{new_val:.2e}" if (abs(new_val) < 0.01 or abs(new_val) >= 1e5) else str(new_val)
    else:
        new_str = str(new_val)
    new_text = text[:m.start(2)] + new_str + text[m.end(2):]
    pathlib.Path(filepath).write_text(new_text)

elif op == "py_scalar":
    # path_or_field is the bare attribute name, e.g. "termination_height"
    field = path_or_field
    # Match "    field = <value>" pattern (class-body scalar assignment)
    pattern = re.compile(
        r'^(\s*' + re.escape(field) + r'\s*=\s*)([^\n#]+)',
        re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        print(f"ERROR: py_scalar field '{field}' not found in {filepath}", file=sys.stderr)
        sys.exit(1)
    if isinstance(new_val, float):
        new_str = str(new_val)
    else:
        new_str = str(new_val)
    new_text = text[:m.start(2)] + new_str + text[m.end(2):]
    pathlib.Path(filepath).write_text(new_text)

elif op == "py_dict_index":
    # path_or_field is the dict key, dict_index_raw is the list index to change
    # e.g. field="lin_vel_x_range", index=1, new_val=3.0
    # Matches: "lin_vel_x_range": [<old0>, <old1>]
    field = path_or_field
    idx = int(dict_index_raw)
    # Regex: capture the list literal under this key
    pattern = re.compile(
        r'("' + re.escape(field) + r'"\s*:\s*\[)([^\]]+)(\])',
        re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        print(f"ERROR: py_dict_index key '{field}' not found in {filepath}", file=sys.stderr)
        sys.exit(1)
    elements = [e.strip() for e in m.group(2).split(",")]
    if isinstance(new_val, float):
        elements[idx] = str(new_val)
    else:
        elements[idx] = str(new_val)
    new_list_body = ", ".join(elements)
    new_text = text[:m.start(2)] + new_list_body + text[m.end(2):]
    pathlib.Path(filepath).write_text(new_text)

else:
    print(f"ERROR: unknown op '{op}'", file=sys.stderr)
    sys.exit(1)

print(f"[apply_mutation] {op} {filepath}:{path_or_field} -> {new_val}")
PYEOF
}

# ────────────────────────────────────────────────────────────────────────────────
# STATE INIT (if absent)
# ────────────────────────────────────────────────────────────────────────────────
if [[ ! -f "${STATE_FILE}" ]]; then
    echo "[$(now_utc)] state.json absent — initialising." >&2
    MISSION_HOURS="${MISSION_HOURS:-48}" \
    MISSION_SLUG_INIT="${MISSION_SLUG}" \
    RUN_ID_INIT="${RUN_ID}" \
    BASELINE_CKPT_INIT="${BASELINE_CKPT_DEFAULT}" \
    STATE_FILE_INIT="${STATE_FILE}" \
    python3 -c "
import json, os
from datetime import datetime, timedelta, timezone
hours = int(os.environ['MISSION_HOURS'])
now = datetime.now(timezone.utc).replace(microsecond=0)
deadline = now + timedelta(hours=hours)
d = {
    'mission_slug': os.environ['MISSION_SLUG_INIT'],
    'run_id': os.environ['RUN_ID_INIT'],
    'active': True,
    'iteration': 0,
    'iteration_zero_pending': True,
    'started_at': now.isoformat().replace('+00:00', 'Z'),
    'deadline': deadline.isoformat().replace('+00:00', 'Z'),
    'baseline_checkpoint': os.environ['BASELINE_CKPT_INIT'],
    'baseline_score_primary': None,
    'baseline_score_disjoint': None,
    'baseline_std': None,
    'best_score': None,
    'best_iteration': None,
}
tmp = os.environ['STATE_FILE_INIT'] + '.tmp'
with open(tmp, 'w') as f:
    json.dump(d, f, indent=2)
    f.write('\n')
os.replace(tmp, os.environ['STATE_FILE_INIT'])
"
fi

# ────────────────────────────────────────────────────────────────────────────────
# READ STATE
# ────────────────────────────────────────────────────────────────────────────────
ACTIVE="$(state_get active)"
DEADLINE="$(state_get deadline)"
ITERATION="$(state_get iteration)"
ITER0_PENDING="$(state_get iteration_zero_pending)"
BASELINE_CKPT="$(state_get baseline_checkpoint)"

if [[ "${ACTIVE}" != "True" ]]; then
    echo "[$(now_utc)] Mission inactive — exiting." >&2
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────────────
# DEADLINE CHECK
# ────────────────────────────────────────────────────────────────────────────────
NOW_EPOCH="$(date -u +%s)"
DEADLINE_EPOCH="$(date -u -d "${DEADLINE}" +%s 2>/dev/null || python3 -c "
from datetime import datetime, timezone
import sys
dt = datetime.fromisoformat('${DEADLINE}'.replace('Z','+00:00'))
print(int(dt.timestamp()))
")"

if [[ "${NOW_EPOCH}" -ge "${DEADLINE_EPOCH}" ]]; then
    dlog_append <<DLOG

## Mission STOP — deadline reached

**Time:** $(now_utc)
**Iteration reached:** ${ITERATION}

Deadline ${DEADLINE} exceeded. No further iterations will run.
DLOG
    state_set "active" "False"
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────────────
# GPU SELECTION  (auto-pick a free GPU unless GPU_PIN overrides)
# ────────────────────────────────────────────────────────────────────────────────
if [[ -n "${GPU_PIN}" ]]; then
    # Pinned mode: use the specified GPU iff it is below busy limits
    GPU_INFO="$(nvidia-smi -i "${GPU_PIN}" --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null || echo "999, 999999")"
    GPU_UTIL="$(echo "${GPU_INFO}" | awk -F',' '{print int($1)}')"
    GPU_MEM="$(echo "${GPU_INFO}" | awk -F',' '{print int($2)}')"
    if [[ "${GPU_UTIL}" -gt "${GPU_UTIL_LIMIT}" ]] || [[ "${GPU_MEM}" -gt "${GPU_MEM_LIMIT_MB}" ]]; then
        dlog_append <<DLOG

Iteration ${ITERATION} skipped — pinned GPU ${GPU_PIN} busy (util=${GPU_UTIL}%, mem=${GPU_MEM}MB) at $(now_utc)
DLOG
        exit 0
    fi
    SELECTED_GPU="${GPU_PIN}"
else
    # Auto mode: scan all GPUs, pick the one with lowest memory use among those below busy limits
    SELECTED_GPU=""
    SELECTED_MEM=999999
    while IFS=, read -r gpu_idx gpu_util gpu_mem; do
        gpu_idx="${gpu_idx// /}"
        gpu_util="${gpu_util// /}"
        gpu_mem="${gpu_mem// /}"
        [[ -z "${gpu_idx}" ]] && continue
        if [[ "${gpu_util}" -le "${GPU_UTIL_LIMIT}" ]] && [[ "${gpu_mem}" -le "${GPU_MEM_LIMIT_MB}" ]]; then
            if [[ "${gpu_mem}" -lt "${SELECTED_MEM}" ]]; then
                SELECTED_MEM="${gpu_mem}"
                SELECTED_GPU="${gpu_idx}"
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null)

    if [[ -z "${SELECTED_GPU}" ]]; then
        dlog_append <<DLOG

Iteration ${ITERATION} skipped — no free GPU found (all above util>${GPU_UTIL_LIMIT}% or mem>${GPU_MEM_LIMIT_MB}MB) at $(now_utc)
DLOG
        exit 0
    fi
fi

export CUDA_VISIBLE_DEVICES="${SELECTED_GPU}"
echo "[$(now_utc)] Selected GPU ${SELECTED_GPU} (mode: ${GPU_PIN:+pinned}${GPU_PIN:-auto})" >&2

# ────────────────────────────────────────────────────────────────────────────────
# TRAP for unexpected errors
# ────────────────────────────────────────────────────────────────────────────────
ITER_DIR=""
BACKUPS_TO_RESTORE=()

cleanup_on_error() {
    local exit_code=$?
    local lineno="${BASH_LINENO[0]}"
    local msg="exit ${exit_code} at line ${lineno}"
    echo "[$(now_utc)] UNEXPECTED ERROR: ${msg}" >&2
    dlog_append <<DLOG

## Iteration ${ITERATION} — UNEXPECTED ERROR: ${msg}

**Time:** $(now_utc)
**Exit code:** ${exit_code}
**Line:** ${lineno}
DLOG
    # Best-effort backup restoration
    for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
        local orig="${pair%%:::*}"
        local bak="${pair##*:::}"
        if [[ -f "${bak}" ]]; then
            cp "${bak}" "${orig}" && echo "[cleanup] Restored ${orig}" >&2 || true
        fi
    done
    exit 1
}
trap cleanup_on_error ERR

# ────────────────────────────────────────────────────────────────────────────────
# ITERATION 0: BASELINE MEASUREMENT
# ────────────────────────────────────────────────────────────────────────────────
if [[ "${ITER0_PENDING}" == "True" ]]; then
    ITER_DIR="${RUN_DIR}/iter-0000"
    mkdir -p "${ITER_DIR}"

    dlog_append <<DLOG

## Iteration 0 — running baseline measurement

**Started:** $(now_utc)
DLOG

    EVAL_OUT_PRIMARY="${RUN_DIR}/evaluations/iteration-0000-primary.json"
    EVAL_OUT_DISJOINT="${RUN_DIR}/evaluations/iteration-0000-disjoint.json"
    mkdir -p "${RUN_DIR}/evaluations"

    # Primary seed list
    echo "[$(now_utc)] Iter 0: running primary eval..." >&2
    "${ISAACLAB_SH}" -p "${EVAL_SCRIPT}" \
        --task "${TASK}" \
        --checkpoint "${BASELINE_CKPT}" \
        --cmd-grid "${CMD_GRID}" \
        --num-seeds "${NUM_SEEDS}" \
        --rollout-steps "${ROLLOUT_STEPS}" \
        --seed-list-mode primary \
        --output-json "${EVAL_OUT_PRIMARY}" \
        2>&1 | tee "${ITER_DIR}/eval-primary.log"

    if [[ ! -f "${EVAL_OUT_PRIMARY}" ]]; then
        echo "[$(now_utc)] ERROR: primary eval did not produce output JSON." >&2
        exit 1
    fi

    # Disjoint seed list
    echo "[$(now_utc)] Iter 0: running disjoint eval..." >&2
    "${ISAACLAB_SH}" -p "${EVAL_SCRIPT}" \
        --task "${TASK}" \
        --checkpoint "${BASELINE_CKPT}" \
        --cmd-grid "${CMD_GRID}" \
        --num-seeds "${NUM_SEEDS}" \
        --rollout-steps "${ROLLOUT_STEPS}" \
        --seed-list-mode disjoint \
        --output-json "${EVAL_OUT_DISJOINT}" \
        2>&1 | tee "${ITER_DIR}/eval-disjoint.log"

    if [[ ! -f "${EVAL_OUT_DISJOINT}" ]]; then
        echo "[$(now_utc)] ERROR: disjoint eval did not produce output JSON." >&2
        exit 1
    fi

    # Parse results and update state
    python3 -c "
import json, os

with open('${EVAL_OUT_PRIMARY}') as f:
    p = json.load(f)
with open('${EVAL_OUT_DISJOINT}') as f:
    d = json.load(f)

score_p   = p.get('score', float('nan'))
score_d   = d.get('score', float('nan'))
std_p     = p.get('score_std_across_seeds', 0.0)
disc_p    = p.get('disc_collapsed', False)
disc_d    = d.get('disc_collapsed', False)

print(f'PRIMARY:  score={score_p:.4f}  std={std_p:.4f}  disc_collapsed={disc_p}')
print(f'DISJOINT: score={score_d:.4f}  disc_collapsed={disc_d}')

with open('${STATE_FILE}') as f:
    state = json.load(f)
state['iteration_zero_pending'] = False
state['iteration'] = 1
state['baseline_score_primary'] = score_p
state['baseline_score_disjoint'] = score_d
state['baseline_std'] = std_p
tmp = '${STATE_FILE}.tmp'
with open(tmp, 'w') as f:
    json.dump(state, f, indent=2)
    f.write('\n')
os.replace(tmp, '${STATE_FILE}')

# Write decision-log entry to stdout for capture
entry = f'''
## Iteration 0 — Baseline measurement (DONE)

**Completed:** \$(date -u +%Y-%m-%dT%H:%M:%SZ)

| Run | Score | Std across seeds | disc_collapsed |
|-----|-------|-----------------|----------------|
| Primary [0,1,2,3] | {score_p:.4f} | {std_p:.4f} | {disc_p} |
| Disjoint [10,11,12,13] | {score_d:.4f} | N/A | {disc_d} |

Pass threshold for any iteration: score > {score_p:.4f} + 1.0 * {std_p:.4f} = {score_p + std_p:.4f}
Stability threshold: score_disjoint > {score_p:.4f} + 0.5 * {std_p:.4f} = {score_p + 0.5*std_p:.4f}
'''
print(entry)
" >> "${DECISION_LOG}"

    echo "[$(now_utc)] Iter 0 complete." >&2
    exit 0
fi

# ────────────────────────────────────────────────────────────────────────────────
# MUTATION ITERATIONS (iteration >= 1)
# ────────────────────────────────────────────────────────────────────────────────
ITER_NUM="${ITERATION}"
ITER_PAD="$(printf '%04d' "${ITER_NUM}")"
ITER_DIR="${RUN_DIR}/iter-${ITER_PAD}"
mkdir -p "${ITER_DIR}"

echo "[$(now_utc)] Starting iteration ${ITER_NUM}..." >&2

# Load mutation for this iteration
MUTATION_JSON="$(python3 -c "
import json, sys
with open('${MUTATIONS_FILE}') as f:
    data = json.load(f)
mutations = data.get('mutations', [])
match = [m for m in mutations if m.get('iteration') == ${ITER_NUM}]
if not match:
    print('GRID_EXHAUSTED')
else:
    print(json.dumps(match[0]))
")"

if [[ "${MUTATION_JSON}" == "GRID_EXHAUSTED" ]]; then
    dlog_append <<DLOG

## Mission STOP — mutation grid exhausted

**Time:** $(now_utc)
**Iteration reached:** ${ITER_NUM}

No mutation defined for iteration ${ITER_NUM}. All ${MUTATIONS_FILE} entries consumed.
DLOG
    state_set "active" "False"
    exit 0
fi

MUT_ID="$(echo "${MUTATION_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['id'])")"
MUT_RATIONALE="$(echo "${MUTATION_JSON}" | python3 -c "import json,sys; print(json.load(sys.stdin)['rationale'])")"

dlog_append <<DLOG

## Iteration ${ITER_NUM} — ${MUT_ID}

**Started:** $(now_utc)
**Mutation ID:** ${MUT_ID}
**Rationale:** ${MUT_RATIONALE}
DLOG

# ── Apply mutation, backing up each file ──────────────────────────────────────
echo "${MUTATION_JSON}" | python3 -c "
import json, sys, os, shutil, pathlib

mutation = json.load(sys.stdin)
iter_dir = '${ITER_DIR}'
os.makedirs(iter_dir, exist_ok=True)

for applies in mutation.get('applies', []):
    filepath  = applies['file']
    op        = applies['op']
    path_fld  = applies['yaml_path']
    new_val   = applies['to']
    dict_idx  = str(applies.get('dict_index', ''))

    # Backup
    bak_name  = pathlib.Path(filepath).name + '.bak'
    bak_path  = os.path.join(iter_dir, bak_name)
    shutil.copy2(filepath, bak_path)
    print(f'BACKUP {filepath} -> {bak_path}')

print('APPLIES_DONE')
" > "${ITER_DIR}/backup.log" 2>&1

# Read backup pairs for potential restore
mapfile -t BACKUP_LINES < <(grep "^BACKUP " "${ITER_DIR}/backup.log" || true)
for line in "${BACKUP_LINES[@]+"${BACKUP_LINES[@]}"}"; do
    # Format: "BACKUP <orig> -> <bak>"
    orig="$(echo "${line}" | awk '{print $2}')"
    bak="$(echo "${line}" | awk '{print $4}')"
    BACKUPS_TO_RESTORE+=("${orig}:::${bak}")
done

# Apply each entry
MUTATION_INPUT_FILE="${ITER_DIR}/_mutation_input.json"
printf '%s' "${MUTATION_JSON}" > "${MUTATION_INPUT_FILE}"
MUTATION_INPUT_FILE="${MUTATION_INPUT_FILE}" python3 - <<'APPLY_PY'
import json, sys, re, pathlib, os

mutation = json.load(open(os.environ['MUTATION_INPUT_FILE']))

def parse_val(v):
    return v  # already native Python from JSON

for applies in mutation.get("applies", []):
    filepath   = applies["file"]
    op         = applies["op"]
    path_fld   = applies["yaml_path"]
    new_val    = parse_val(applies["to"])
    dict_idx   = applies.get("dict_index", None)

    text = pathlib.Path(filepath).read_text()

    if op == "yaml_set":
        leaf = path_fld.split(".")[-1]
        pattern = re.compile(
            r"^(\s*" + re.escape(leaf) + r"\s*:\s*)([^\n#]+)",
            re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            print(f"ERROR: yaml_set key '{leaf}' not found in {filepath}", file=sys.stderr)
            sys.exit(1)
        if isinstance(new_val, float):
            fstr = f"{new_val:.2e}" if (abs(new_val) < 0.01 or abs(new_val) >= 1e5) else str(new_val)
        else:
            fstr = str(new_val)
        new_text = text[: m.start(2)] + fstr + text[m.end(2) :]
        pathlib.Path(filepath).write_text(new_text)
        print(f"[yaml_set] {filepath}:{leaf} -> {fstr}")

    elif op == "py_scalar":
        field = path_fld
        pattern = re.compile(
            r"^(\s*" + re.escape(field) + r"\s*=\s*)([^\n#]+)",
            re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            print(f"ERROR: py_scalar field '{field}' not found in {filepath}", file=sys.stderr)
            sys.exit(1)
        new_text = text[: m.start(2)] + str(new_val) + text[m.end(2) :]
        pathlib.Path(filepath).write_text(new_text)
        print(f"[py_scalar] {filepath}:{field} -> {new_val}")

    elif op == "py_dict_index":
        field = path_fld
        idx = int(dict_idx)
        pattern = re.compile(
            r'("' + re.escape(field) + r'"\s*:\s*\[)([^\]]+)(\])',
            re.MULTILINE,
        )
        m = pattern.search(text)
        if not m:
            print(f"ERROR: py_dict_index key '{field}' not found in {filepath}", file=sys.stderr)
            sys.exit(1)
        elements = [e.strip() for e in m.group(2).split(",")]
        elements[idx] = str(new_val)
        new_text = text[: m.start(2)] + ", ".join(elements) + text[m.end(2) :]
        pathlib.Path(filepath).write_text(new_text)
        print(f"[py_dict_index] {filepath}:{field}[{idx}] -> {new_val}")

    else:
        print(f"ERROR: unknown op '{op}'", file=sys.stderr)
        sys.exit(1)
APPLY_PY

echo "[$(now_utc)] Mutation ${MUT_ID} applied." >&2

# ── Pre-train compile check ───────────────────────────────────────────────────
# Module-level import is impossible without booting Isaac Sim (pxr depends on it).
# Instead we validate file syntax for each mutated file. Real runtime validation
# is left to the first ~60s of Isaac Sim training startup + 100k-step divergence check.
echo "[$(now_utc)] Running compile check (syntax-only)..." >&2
COMPILE_OK=true
MUTATION_INPUT_FILE="${ITER_DIR}/_mutation_input.json" python3 - > "${ITER_DIR}/compile-check.log" 2>&1 <<'COMPILE_PY' || COMPILE_OK=false
import ast, json, os, sys
import yaml  # Available in isaac conda env via PyYAML
mutation = json.load(open(os.environ['MUTATION_INPUT_FILE']))
for applies in mutation.get('applies', []):
    fp = applies['file']
    try:
        text = open(fp).read()
    except Exception as exc:
        print(f'OPEN FAIL {fp}: {exc}', file=sys.stderr)
        sys.exit(1)
    if fp.endswith(('.yaml', '.yml')):
        try:
            yaml.safe_load(text)
        except yaml.YAMLError as exc:
            print(f'YAML SYNTAX FAIL {fp}: {exc}', file=sys.stderr)
            sys.exit(1)
        print(f'YAML OK {fp}')
    elif fp.endswith('.py'):
        try:
            ast.parse(text, filename=fp)
        except SyntaxError as exc:
            print(f'PY SYNTAX FAIL {fp}: {exc}', file=sys.stderr)
            sys.exit(1)
        print(f'PY OK {fp}')
    else:
        print(f'SKIP {fp} (unknown extension)')
print('COMPILE_PASS')
COMPILE_PY

if [[ "${COMPILE_OK}" == "false" ]]; then
    dlog_append <<DLOG
- **Pre-train compile check:** FAIL (see iter-${ITER_PAD}/compile-check.log)
- **Decision:** REJECTED — reverting mutation.
DLOG
    # Restore backups
    for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
        orig="${pair%%:::*}"
        bak="${pair##*:::}"
        [[ -f "${bak}" ]] && cp "${bak}" "${orig}" && echo "[restore] ${orig}" >&2 || true
    done
    state_set "iteration" "$((ITER_NUM + 1))"
    exit 0
fi

dlog_append <<DLOG
- **Pre-train compile check:** PASS
DLOG

# ── Training ──────────────────────────────────────────────────────────────────
echo "[$(now_utc)] Starting training (timeout ${TRAIN_TIMEOUT}s, ${ITER_PER_FIRING} iters)..." >&2
TRAIN_LOG="${ITER_DIR}/train.log"

TRAIN_EXIT=0
TRAIN_VIDEO_ARGS=()
if [[ "${RECORD_VIDEO}" == "1" ]]; then
    TRAIN_VIDEO_ARGS=(--video --video_length 200 --video_interval 2000)
fi
timeout "${TRAIN_TIMEOUT}" "${ISAACLAB_SH}" -p scripts/reinforcement_learning/skrl/train.py \
    --task "${TASK}" \
    --algorithm AMP \
    --num_envs 4096 \
    --max_iterations "${ITER_PER_FIRING}" \
    --checkpoint "${BASELINE_CKPT}" \
    --headless \
    "${TRAIN_VIDEO_ARGS[@]}" \
    2>&1 | tee "${TRAIN_LOG}" || TRAIN_EXIT=$?

if [[ "${TRAIN_EXIT}" -eq 124 ]]; then
    echo "[$(now_utc)] Training timed out after ${TRAIN_TIMEOUT}s — using last checkpoint." >&2
    dlog_append <<DLOG
- **Training:** timed out at ${TRAIN_TIMEOUT}s — using last available checkpoint.
DLOG
elif [[ "${TRAIN_EXIT}" -ne 0 ]]; then
    dlog_append <<DLOG
- **Training:** FAILED (exit code ${TRAIN_EXIT}) — reverting mutation.
DLOG
    for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
        orig="${pair%%:::*}"
        bak="${pair##*:::}"
        [[ -f "${bak}" ]] && cp "${bak}" "${orig}" || true
    done
    state_set "iteration" "$((ITER_NUM + 1))"
    exit 0
fi

# ── Find latest checkpoint ────────────────────────────────────────────────────
NEW_CKPT="$(find "${ISAACLAB_ROOT}/logs/skrl/go2_skrl_amp" \
    -name "agent_*.pt" -newer "${BASELINE_CKPT}" \
    -printf '%T@ %p\n' 2>/dev/null \
    | sort -n | tail -1 | awk '{print $2}' || true)"

if [[ -z "${NEW_CKPT}" ]]; then
    # Fallback: find any best_agent.pt newer than baseline
    NEW_CKPT="$(find "${ISAACLAB_ROOT}/logs/skrl/go2_skrl_amp" \
        -name "best_agent.pt" -newer "${BASELINE_CKPT}" \
        -printf '%T@ %p\n' 2>/dev/null \
        | sort -n | tail -1 | awk '{print $2}' || true)"
fi

if [[ -z "${NEW_CKPT}" ]]; then
    dlog_append <<DLOG
- **Checkpoint search:** FAILED — no new checkpoint found after training. Reverting.
DLOG
    for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
        orig="${pair%%:::*}"
        bak="${pair##*:::}"
        [[ -f "${bak}" ]] && cp "${bak}" "${orig}" || true
    done
    state_set "iteration" "$((ITER_NUM + 1))"
    exit 0
fi

echo "[$(now_utc)] New checkpoint: ${NEW_CKPT}" >&2
dlog_append <<DLOG
- **Training:** completed (exit ${TRAIN_EXIT})
- **New checkpoint:** ${NEW_CKPT}
DLOG

# ── 100k-step divergence check ────────────────────────────────────────────────
DIVERGED=false
if grep -qiE "nan|loss.*nan|reward.*nan" "${TRAIN_LOG}" 2>/dev/null; then
    DIVERGED=true
fi
# Episode length collapse: look for episode_len lines and check for very short episodes
if python3 -c "
import re, sys
text = open('${TRAIN_LOG}').read()
# Extract episode_length values (logged as 'episode_length_mean' or similar)
vals = [float(x) for x in re.findall(r'episode_length[_\s]+(?:mean)?[\s:=]+([0-9.]+)', text, re.IGNORECASE)]
if vals and any(v < 50 for v in vals[-10:]):  # last 10 readings
    sys.exit(1)
sys.exit(0)
" 2>/dev/null; then
    true
else
    DIVERGED=true
fi

if [[ "${DIVERGED}" == "true" ]]; then
    dlog_append <<DLOG
- **100k-step divergence check:** FAIL (NaN or episode_len collapse detected)
- **Decision:** REJECTED — reverting mutation.
DLOG
    for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
        orig="${pair%%:::*}"
        bak="${pair##*:::}"
        [[ -f "${bak}" ]] && cp "${bak}" "${orig}" || true
    done
    state_set "iteration" "$((ITER_NUM + 1))"
    exit 0
fi

dlog_append <<DLOG
- **100k-step divergence check:** PASS
DLOG

# ── Evaluator (primary) ───────────────────────────────────────────────────────
EVAL_OUT_PRIMARY="${RUN_DIR}/evaluations/iteration-${ITER_PAD}-primary.json"
mkdir -p "${RUN_DIR}/evaluations"

echo "[$(now_utc)] Running primary evaluator..." >&2
"${ISAACLAB_SH}" -p "${EVAL_SCRIPT}" \
    --task "${TASK}" \
    --checkpoint "${NEW_CKPT}" \
    --cmd-grid "${CMD_GRID}" \
    --num-seeds "${NUM_SEEDS}" \
    --rollout-steps "${ROLLOUT_STEPS}" \
    --seed-list-mode primary \
    --output-json "${EVAL_OUT_PRIMARY}" \
    2>&1 | tee "${ITER_DIR}/eval-primary.log"

if [[ ! -f "${EVAL_OUT_PRIMARY}" ]]; then
    echo "[$(now_utc)] ERROR: primary eval JSON absent." >&2
    exit 1
fi

# Parse primary results
EVAL_PARSED="$(python3 -c "
import json
with open('${EVAL_OUT_PRIMARY}') as f:
    d = json.load(f)
print(d.get('score', float('nan')))
print(d.get('score_std_across_seeds', 0.0))
print(d.get('disc_collapsed', False))
")"

SCORE_PRIMARY="$(echo "${EVAL_PARSED}" | sed -n '1p')"
SCORE_STD="$(echo "${EVAL_PARSED}" | sed -n '2p')"
DISC_COLLAPSED="$(echo "${EVAL_PARSED}" | sed -n '3p')"

dlog_append <<DLOG
- **Evaluator (primary):** score=${SCORE_PRIMARY}, disc_collapsed=${DISC_COLLAPSED}, JSON=${EVAL_OUT_PRIMARY}
DLOG

# Discriminator collapse guard
if [[ "${DISC_COLLAPSED}" == "True" ]]; then
    dlog_append <<DLOG
- **Decision:** REJECTED — discriminator collapse detected. Reverting mutation.
DLOG
    for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
        orig="${pair%%:::*}"
        bak="${pair##*:::}"
        [[ -f "${bak}" ]] && cp "${bak}" "${orig}" || true
    done
    state_set "iteration" "$((ITER_NUM + 1))"
    exit 0
fi

# ── Pass criterion check ──────────────────────────────────────────────────────
PASS_RESULT="$(python3 -c "
import json
with open('${STATE_FILE}') as f:
    state = json.load(f)

baseline = state['baseline_score_primary']
baseline_std = state['baseline_std']
score = ${SCORE_PRIMARY}
threshold = baseline + 1.0 * baseline_std
print(f'baseline={baseline:.4f}  threshold={threshold:.4f}  score={score:.4f}')
print('PASS' if score > threshold else 'FAIL')
")"

echo "[$(now_utc)] Pass criterion: ${PASS_RESULT}" >&2

PASS_STATUS="$(echo "${PASS_RESULT}" | tail -1)"

dlog_append <<DLOG
- **Pass criterion (primary):** ${PASS_RESULT}
DLOG

if [[ "${PASS_STATUS}" == "PASS" ]]; then
    # ── Stability check (disjoint seeds) ────────────────────────────────────
    EVAL_OUT_DISJOINT="${RUN_DIR}/evaluations/iteration-${ITER_PAD}-disjoint.json"

    echo "[$(now_utc)] Running disjoint stability check..." >&2
    "${ISAACLAB_SH}" -p "${EVAL_SCRIPT}" \
        --task "${TASK}" \
        --checkpoint "${NEW_CKPT}" \
        --cmd-grid "${CMD_GRID}" \
        --num-seeds "${NUM_SEEDS}" \
        --rollout-steps "${ROLLOUT_STEPS}" \
        --seed-list-mode disjoint \
        --output-json "${EVAL_OUT_DISJOINT}" \
        2>&1 | tee "${ITER_DIR}/eval-disjoint.log"

    STABILITY_PARSED="$(python3 -c "
import json
with open('${EVAL_OUT_DISJOINT}') as f:
    d = json.load(f)
print(d.get('score', float('nan')))
print(d.get('disc_collapsed', False))
")"

    SCORE_DISJOINT="$(echo "${STABILITY_PARSED}" | sed -n '1p')"
    DISC_COLLAPSED_D="$(echo "${STABILITY_PARSED}" | sed -n '2p')"

    STABILITY_OK="$(python3 -c "
import json
with open('${STATE_FILE}') as f:
    state = json.load(f)
baseline = state['baseline_score_primary']
baseline_std = state['baseline_std']
score_d = ${SCORE_DISJOINT}
disc_d = '${DISC_COLLAPSED_D}' == 'True'
threshold_d = baseline + 0.5 * baseline_std
ok = (score_d > threshold_d) and (not disc_d)
print(f'disjoint_score={score_d:.4f}  threshold={threshold_d:.4f}  disc_collapsed={disc_d}')
print('STABLE' if ok else 'NOT_STABLE')
")"

    STABLE_STATUS="$(echo "${STABILITY_OK}" | tail -1)"

    dlog_append <<DLOG
- **Stability check (disjoint):** score_disjoint=${SCORE_DISJOINT}, disc_collapsed=${DISC_COLLAPSED_D}
- **Stability result:** ${STABILITY_OK}
DLOG

    if [[ "${STABLE_STATUS}" == "STABLE" ]]; then
        dlog_append <<DLOG

## Mission SUCCESS — iteration ${ITER_NUM} pass criterion met

**Time:** $(now_utc)
**Mutation:** ${MUT_ID}
**Primary score:** ${SCORE_PRIMARY}
**Disjoint score:** ${SCORE_DISJOINT}
**New checkpoint:** ${NEW_CKPT}

Mutation kept. Mission complete within budget.
DLOG
        state_update "{'active': False, 'best_score': ${SCORE_PRIMARY}, 'best_iteration': ${ITER_NUM}}"
        # Mutation is KEPT (no restore)
        exit 0
    else
        dlog_append <<DLOG
- **Decision:** improvement not stable — reverting mutation.
DLOG
    fi
else
    dlog_append <<DLOG
- **Decision:** primary criterion not met — reverting mutation.
DLOG
fi

# ── Revert mutation (stability fail or primary fail) ──────────────────────────
for pair in "${BACKUPS_TO_RESTORE[@]+"${BACKUPS_TO_RESTORE[@]}"}"; do
    orig="${pair%%:::*}"
    bak="${pair##*:::}"
    [[ -f "${bak}" ]] && cp "${bak}" "${orig}" && echo "[restore] ${orig}" >&2 || true
done
echo "[$(now_utc)] Mutation reverted." >&2

# ── Advance iteration counter ─────────────────────────────────────────────────
state_set "iteration" "$((ITER_NUM + 1))"

dlog_append <<DLOG
- **Iteration ${ITER_NUM} closed at:** $(now_utc)
DLOG

echo "[$(now_utc)] Iteration ${ITER_NUM} complete." >&2
exit 0

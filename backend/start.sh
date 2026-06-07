#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
#  start.sh — launch the Magenta-RT backend for the track-selection extension.
#
#  Layout (after the user moved magenta-realtime into track-selection):
#    track-selection/
#      backend/             <- this script
#      magenta-realtime/    <- in-tree copy of the magenta-rt source
#      ...
#
#  We:
#    1. Find a Python venv with `magenta_rt` + `mlx` installed.
#    2. Force imports to resolve `magenta_rt` from track-selection/magenta-realtime
#       (so the in-tree copy is the source of truth).
#    3. Install FastAPI/uvicorn into that venv if missing (uv pip > python -m pip).
#    4. Run server.py on http://127.0.0.1:8765 (override with HOST/PORT).
#
#  Usage:
#    ./start.sh                 # mrt2_small (real model)
#    ./start.sh --dry-run       # sine+kick stub, no MLX needed
#    ./start.sh --model mrt2_base
#    PORT=9000 ./start.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# track-selection/
TRACK_SELECTION_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# hackathon/
HACKATHON_DIR="$(cd "$TRACK_SELECTION_DIR/.." && pwd)"

# In-tree magenta source we want imports to resolve to.
LOCAL_MAGENTA_DIR="$TRACK_SELECTION_DIR/magenta-realtime"
LEGACY_MAGENTA_DIR="$HACKATHON_DIR/magenta-realtime"
# In-tree Stable Audio 3 source (optional second engine).
LOCAL_SA3_DIR="$TRACK_SELECTION_DIR/stable-audio-3"

PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"

# ─── 1. Pick a venv that already has magenta_rt + mlx ─────────────────────────
# Priority:
#   (a) track-selection/magenta-realtime/.venv  (if the user created one here)
#   (b) track-selection/.venv
#   (c) hackathon/.venv                          (current install in this repo)
#   (d) ambient $VIRTUAL_ENV
# We probe with PYTHONPATH already pointing at the in-tree magenta source,
# because the venv's editable `.pth` may target an old (now-missing) location
# after the user moved magenta-realtime into track-selection/.
candidate_venvs=(
    "$LOCAL_MAGENTA_DIR/.venv"
    "$TRACK_SELECTION_DIR/.venv"
    "$HACKATHON_DIR/.venv"
    "$LEGACY_MAGENTA_DIR/.venv"
)

probe_python() {
    PYTHONPATH="$LOCAL_MAGENTA_DIR${PYTHONPATH:+:$PYTHONPATH}" \
        "$1" -c "import magenta_rt, mlx" 2>/dev/null
}

VENV=""
for v in "${candidate_venvs[@]}"; do
    if [[ -x "$v/bin/python" ]] && probe_python "$v/bin/python"; then
        VENV="$v"
        break
    fi
done

if [[ -z "$VENV" && -n "${VIRTUAL_ENV:-}" ]] \
        && probe_python "$VIRTUAL_ENV/bin/python"; then
    VENV="$VIRTUAL_ENV"
fi

if [[ -z "$VENV" ]]; then
    echo "✖ Could not find a Python venv with 'magenta_rt' installed."
    echo "  Looked in:"
    for v in "${candidate_venvs[@]}"; do echo "    - $v"; done
    echo "  Set up one (from $LOCAL_MAGENTA_DIR):"
    echo "    cd $LOCAL_MAGENTA_DIR"
    echo "    uv venv --python 3.12 && source .venv/bin/activate"
    echo "    uv pip install -e \".[mlx]\""
    exit 1
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
PY="$VENV/bin/python"
echo "[start.sh] venv     : $VENV"
echo "[start.sh] python   : $($PY -c 'import sys; print(sys.executable)')"

# ─── 2. Force in-tree magenta source ──────────────────────────────────────────
# Prepend the local magenta-realtime/ to PYTHONPATH so `import magenta_rt`
# resolves to the in-tree copy, even if the venv's site-packages has a
# differently-pointed editable install.
if [[ -d "$LOCAL_MAGENTA_DIR/magenta_rt" ]]; then
    export PYTHONPATH="$LOCAL_MAGENTA_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "[start.sh] magenta  : $LOCAL_MAGENTA_DIR (in-tree)"
else
    echo "[start.sh] magenta  : (in-tree copy not found — falling back to venv install)"
fi

# Confirm and report which path magenta_rt actually resolves to.
"$PY" -c "import magenta_rt, os; print(f'[start.sh] magenta_rt -> {os.path.dirname(magenta_rt.__file__)}')" || {
    echo "✖ 'import magenta_rt' failed in $VENV"
    exit 1
}

# Optional: prepend in-tree Stable Audio 3 to PYTHONPATH so `import stable_audio_3`
# works without `uv sync` inside that subfolder. Torch/torchaudio still need to
# be installed in the active venv for SA3 endpoints to actually serve requests.
if [[ -d "$LOCAL_SA3_DIR/stable_audio_3" ]]; then
    export PYTHONPATH="$LOCAL_SA3_DIR${PYTHONPATH:+:$PYTHONPATH}"
    echo "[start.sh] sa3      : $LOCAL_SA3_DIR (in-tree)"
    if "$PY" -c "import torch, stable_audio_3" 2>/dev/null; then
        echo "[start.sh] sa3 deps : OK (torch + stable_audio_3 importable)"
    else
        echo "[start.sh] sa3 deps : MISSING — /sa3/* will be disabled"
        echo "             install with:  cd $LOCAL_SA3_DIR && uv sync"
    fi
else
    echo "[start.sh] sa3      : (not found — Stable Audio 3 engine disabled)"
fi

# ─── 3. Ensure FastAPI deps are present ───────────────────────────────────────
need_install=0
"$PY" -c "import fastapi, uvicorn, pydantic" 2>/dev/null || need_install=1

if [[ $need_install -eq 1 ]]; then
    echo "[start.sh] installing FastAPI deps into $VENV ..."
    if "$PY" -m pip --version >/dev/null 2>&1; then
        "$PY" -m pip install -q -r "$SCRIPT_DIR/requirements.txt"
    elif command -v uv >/dev/null 2>&1; then
        VIRTUAL_ENV="$VENV" uv pip install -q -r "$SCRIPT_DIR/requirements.txt"
    else
        echo "✖ Neither 'pip' (in venv) nor 'uv' (on PATH) is available."
        echo "  Install uv (https://docs.astral.sh/uv/) or repair the venv with pip,"
        echo "  then re-run this script."
        exit 1
    fi
fi

# ─── 4. Run the server ────────────────────────────────────────────────────────
exec "$PY" "$SCRIPT_DIR/server.py" --host "$HOST" --port "$PORT" "$@"

#!/usr/bin/env bash
set -euo pipefail

cd /app

if [[ "${SKIP_CHECKPOINT_PREP:-0}" =~ ^(1|true|yes)$ ]]; then
  exec "$@"
fi

# RunPod shared volume layout:
#   /runpod-volume/skintokens/experiments/
#   /runpod-volume/skintokens/models/
VOLUME_ROOT="${SKINTOKENS_CKPTS_ROOT:-}"
if [[ -z "$VOLUME_ROOT" && -d /runpod-volume/skintokens ]]; then
  VOLUME_ROOT=/runpod-volume/skintokens
fi

if [[ -n "$VOLUME_ROOT" && -d "$VOLUME_ROOT" ]]; then
  VOLUME_ROOT="$(readlink -f "$VOLUME_ROOT")"

  if [[ -d "$VOLUME_ROOT/experiments" ]]; then
    CKPTS_SRC="$(readlink -f "$VOLUME_ROOT/experiments")"
    if [[ -L /app/experiments ]] && [[ "$(readlink -f /app/experiments)" == "$CKPTS_SRC" ]]; then
      echo ">>> /app/experiments already linked -> $CKPTS_SRC"
    else
      rm -rf /app/experiments
      ln -sfn "$CKPTS_SRC" /app/experiments
      echo ">>> Linked /app/experiments -> $CKPTS_SRC"
    fi
  fi

  if [[ -d "$VOLUME_ROOT/models" ]]; then
    MODELS_SRC="$(readlink -f "$VOLUME_ROOT/models")"
    if [[ -L /app/models ]] && [[ "$(readlink -f /app/models)" == "$MODELS_SRC" ]]; then
      echo ">>> /app/models already linked -> $MODELS_SRC"
    else
      rm -rf /app/models
      ln -sfn "$MODELS_SRC" /app/models
      echo ">>> Linked /app/models -> $MODELS_SRC"
    fi
  fi
fi

if ! python scripts/ensure_checkpoints.py; then
  exit 1
fi

# Start bpy_server sidecar (required for mesh load/export).
python bpy_server.py &
BPY_PID=$!

cleanup() {
  if kill -0 "$BPY_PID" 2>/dev/null; then
    echo ">>> Stopping bpy_server (pid=$BPY_PID)"
    kill "$BPY_PID" 2>/dev/null || true
    wait "$BPY_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo ">>> Waiting for bpy_server on port 59876..."
for _ in $(seq 1 60); do
  if curl -sf http://127.0.0.1:59876/ping >/dev/null 2>&1; then
    echo ">>> bpy_server is ready"
    break
  fi
  if ! kill -0 "$BPY_PID" 2>/dev/null; then
    echo ">>> ERROR: bpy_server exited before becoming ready" >&2
    exit 1
  fi
  sleep 0.5
done

if ! curl -sf http://127.0.0.1:59876/ping >/dev/null 2>&1; then
  echo ">>> ERROR: bpy_server failed to start within 30s" >&2
  exit 1
fi

exec "$@"

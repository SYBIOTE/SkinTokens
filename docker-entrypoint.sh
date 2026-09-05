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

# bpy_server is started and supervised by whichever process runs in the
# foreground (bpy_supervisor.py): FastAPI's lifespan for the HTTP image,
# handler.py at import for the queue image. Either way exactly one foreground
# process remains, which is what RunPod Serverless expects.
exec "$@"
